"""Small frictionless hard-normal contact active-set solvers.

The routines here implement the algebraic hard-contact layer needed before the
full Lagrangian-SDF surface contact path can support Abaqus-style
``pressure-overclosure=HARD`` semantics.  They are intentionally geometry
agnostic: callers provide a linear stiffness matrix, external force vector,
gap offsets, and gap Jacobian rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator, cg, splu, spsolve


@dataclass(frozen=True, slots=True)
class HardContactSolution:
    """Result from a frictionless hard-normal active-set solve."""

    displacement: np.ndarray
    multipliers: np.ndarray
    gaps: np.ndarray
    active: np.ndarray
    iterations: int
    converged: bool


class _IterativeHardContactSolveFailed(RuntimeError):
    """Internal signal used to fall back to the direct Schur path."""


def solve_linear_hard_contact_active_set(
    stiffness: np.ndarray,
    external_force: np.ndarray,
    gap_offset: np.ndarray,
    gap_jacobian: np.ndarray,
    *,
    equilibrium_jacobian_scale: float = 1.0,
    normal_compliance: np.ndarray | float | None = None,
    initial_active: np.ndarray | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 20,
) -> HardContactSolution:
    """Solve a linear frictionless hard-contact complementarity problem.

    The gap is ``g = gap_offset + J u`` and must satisfy ``g >= 0``.  The
    contact multiplier is non-negative and contributes ``J.T lambda`` to the
    resisting force.  With zero compliance, the active set is solved from

    ``K u - s J_active.T lambda = f``,
    ``J_active u = -gap_offset_active``.

    A nonzero ``normal_compliance`` solves the regularized hard-enforcement
    system ``g + C lambda = 0`` on active constraints.  This models solver
    contact-enforcement compliance without changing the geometric gap query.
    The optional scale ``s`` is used by HHT-style equilibrium, where current
    contact forces enter the residual with the same ``1 + alpha`` weight as the
    rest of the current static force balance while the geometric constraint
    equation remains unscaled.
    """

    K = _dense_matrix(stiffness)
    f = np.asarray(external_force, dtype=float).reshape(-1)
    g0 = np.asarray(gap_offset, dtype=float).reshape(-1)
    J = np.asarray(gap_jacobian, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("stiffness must be square")
    if f.shape != (K.shape[0],):
        raise ValueError("external_force length must match stiffness")
    if J.ndim != 2 or J.shape[1] != K.shape[0]:
        raise ValueError("gap_jacobian must have shape (n_constraints, n_dofs)")
    if g0.shape != (J.shape[0],):
        raise ValueError("gap_offset length must match number of constraints")
    if float(tolerance) < 0.0:
        raise ValueError("tolerance must be non-negative")
    scale = float(equilibrium_jacobian_scale)
    if scale <= 0.0:
        raise ValueError("equilibrium_jacobian_scale must be positive")
    compliance = _as_constraint_compliance(normal_compliance, J.shape[0])
    if initial_active is None:
        try:
            unconstrained = np.linalg.solve(K, f)
        except np.linalg.LinAlgError:
            unconstrained, *_ = np.linalg.lstsq(K, f, rcond=None)
        active = g0 + J @ unconstrained < -float(tolerance)
    else:
        active = np.asarray(initial_active, dtype=bool).reshape(-1)
        if active.shape != g0.shape:
            raise ValueError("initial_active must match constraints")
    u = np.zeros(K.shape[0], dtype=float)
    lam = np.zeros(J.shape[0], dtype=float)
    converged = False
    iterations = 0
    for iterations in range(1, max(1, int(max_iterations)) + 1):
        active_ids = np.flatnonzero(active)
        if active_ids.size:
            Ja = J[active_ids]
            Ca = np.diag(compliance[active_ids])
            matrix = np.block(
                [
                    [K, -scale * Ja.T],
                    [Ja, Ca],
                ]
            )
            rhs = np.concatenate((f, -g0[active_ids]))
            try:
                sol = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                sol, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
            u = sol[: K.shape[0]]
            lam = np.zeros(J.shape[0], dtype=float)
            lam[active_ids] = sol[K.shape[0] :]
        else:
            try:
                u = np.linalg.solve(K, f)
            except np.linalg.LinAlgError:
                u, *_ = np.linalg.lstsq(K, f, rcond=None)
            lam = np.zeros(J.shape[0], dtype=float)
        gaps = g0 + J @ u
        effective_gaps = gaps + compliance * lam
        next_active = active.copy()
        next_active[effective_gaps < -float(tolerance)] = True
        next_active[lam < -float(tolerance)] = False
        if np.array_equal(next_active, active) and np.all(effective_gaps >= -float(tolerance)) and np.all(lam >= -float(tolerance)):
            converged = True
            break
        active = next_active
    gaps = g0 + J @ u
    lam[np.abs(lam) <= float(tolerance)] = 0.0
    return HardContactSolution(
        displacement=u,
        multipliers=lam,
        gaps=gaps,
        active=active,
        iterations=int(iterations),
        converged=bool(converged),
    )


def hard_contact_gap_jacobian_from_samples(
    samples: Iterable[Any],
    *,
    n_total_dofs: int,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense hard-contact gap offsets and Jacobian rows from samples.

    Each sample must expose the same attributes as
    :class:`sfc.fem.calculix_aligned.ContactSample`: ``gap``, ``normal``,
    ``node_ids``, ``shape_weights``, and optional ``master_node_ids`` /
    ``master_shape_weights``.  The row convention is

    ``g = g0 + J u``,

    with positive slave-side normal motion increasing the gap and master-side
    motion along the same normal decreasing it.
    """

    n_dofs = int(n_total_dofs)
    if n_dofs <= 0:
        raise ValueError("n_total_dofs must be positive")
    g0: list[float] = []
    rows: list[np.ndarray] = []
    for sample in samples:
        normal = np.asarray(sample.normal, dtype=float).reshape(-1)
        if normal.shape != (3,):
            raise ValueError("sample normal must have shape (3,)")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 0.0:
            raise ValueError("sample normal must be nonzero")
        normal = normal / normal_norm
        row = np.zeros(n_dofs, dtype=float)
        slave_nodes = np.asarray(sample.node_ids, dtype=np.int64).reshape(-1)
        slave_weights = np.asarray(sample.shape_weights, dtype=float).reshape(-1)
        if slave_nodes.shape != slave_weights.shape:
            raise ValueError("sample slave nodes and weights must have matching lengths")
        _scatter_gap_row(row, slave_nodes, slave_weights, normal, int(slave_dof_offset), sign=1.0)
        master_nodes = getattr(sample, "master_node_ids", None)
        master_weights = getattr(sample, "master_shape_weights", None)
        if master_nodes is not None and master_weights is not None:
            master_ids = np.asarray(master_nodes, dtype=np.int64).reshape(-1)
            master_w = np.asarray(master_weights, dtype=float).reshape(-1)
            if master_ids.shape != master_w.shape:
                raise ValueError("sample master nodes and weights must have matching lengths")
            _scatter_gap_row(row, master_ids, master_w, normal, int(master_dof_offset), sign=-1.0)
        rows.append(row)
        g0.append(float(sample.gap))
    if not rows:
        return np.empty((0,), dtype=float), np.empty((0, n_dofs), dtype=float)
    return np.asarray(g0, dtype=float), np.vstack(rows)


def solve_linear_hard_contact_from_samples(
    stiffness: np.ndarray,
    external_force: np.ndarray,
    samples: Iterable[Any],
    *,
    n_total_dofs: int,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
    equilibrium_jacobian_scale: float = 1.0,
    normal_compliance: np.ndarray | float | None = None,
    initial_active: np.ndarray | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 20,
) -> HardContactSolution:
    """Solve linear hard contact using contact-sample gap/Jacobian data."""

    gap_offset, gap_jacobian = hard_contact_gap_jacobian_from_samples(
        samples,
        n_total_dofs=int(n_total_dofs),
        slave_dof_offset=int(slave_dof_offset),
        master_dof_offset=int(master_dof_offset),
    )
    return solve_linear_hard_contact_active_set(
        stiffness,
        external_force,
        gap_offset,
        gap_jacobian,
        equilibrium_jacobian_scale=equilibrium_jacobian_scale,
        normal_compliance=normal_compliance,
        initial_active=initial_active,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def solve_linear_hard_contact_with_dirichlet(
    stiffness: np.ndarray,
    external_force: np.ndarray,
    gap_offset: np.ndarray,
    gap_jacobian: np.ndarray,
    *,
    fixed_dofs: np.ndarray,
    fixed_values: np.ndarray | None = None,
    equilibrium_jacobian_scale: float = 1.0,
    normal_compliance: np.ndarray | float | None = None,
    initial_active: np.ndarray | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 20,
) -> HardContactSolution:
    """Solve linear hard contact with prescribed displacement constraints.

    Dirichlet values are eliminated before the active-set solve.  The returned
    displacement vector is expanded back to the full DOF space, while contact
    multipliers and gaps remain associated with the original constraint rows.
    """

    K = _dense_matrix(stiffness)
    f = np.asarray(external_force, dtype=float).reshape(-1)
    g0 = np.asarray(gap_offset, dtype=float).reshape(-1)
    J = np.asarray(gap_jacobian, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("stiffness must be square")
    if f.shape != (K.shape[0],):
        raise ValueError("external_force length must match stiffness")
    if J.ndim != 2 or J.shape[1] != K.shape[0]:
        raise ValueError("gap_jacobian must have shape (n_constraints, n_dofs)")
    if g0.shape != (J.shape[0],):
        raise ValueError("gap_offset length must match number of constraints")
    compliance = _as_constraint_compliance(normal_compliance, J.shape[0])
    scale = float(equilibrium_jacobian_scale)
    if scale <= 0.0:
        raise ValueError("equilibrium_jacobian_scale must be positive")
    fixed = np.asarray(fixed_dofs, dtype=np.int64).reshape(-1)
    if np.any(fixed < 0) or (fixed.size and int(fixed.max()) >= K.shape[0]):
        raise ValueError("fixed_dofs reference a dof outside stiffness")
    if np.unique(fixed).shape[0] != fixed.shape[0]:
        raise ValueError("fixed_dofs must not contain duplicates")
    if fixed_values is None:
        values = np.zeros(fixed.shape[0], dtype=float)
    else:
        values = np.asarray(fixed_values, dtype=float).reshape(-1)
        if values.shape != fixed.shape:
            raise ValueError("fixed_values must match fixed_dofs")

    full_u = np.zeros(K.shape[0], dtype=float)
    full_u[fixed] = values
    free_mask = np.ones(K.shape[0], dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    if free.size == 0:
        gaps = g0 + J @ full_u
        active = gaps < -float(tolerance)
        return HardContactSolution(
            displacement=full_u,
            multipliers=np.zeros(J.shape[0], dtype=float),
            gaps=gaps,
            active=active,
            iterations=0,
            converged=not bool(np.any(active)),
        )

    Kff = K[np.ix_(free, free)]
    Kfc = K[np.ix_(free, fixed)] if fixed.size else np.empty((free.size, 0), dtype=float)
    f_eff = f[free] - (Kfc @ values if fixed.size else 0.0)
    Jf = J[:, free]
    g_eff = g0 + (J[:, fixed] @ values if fixed.size else 0.0)
    reduced = solve_linear_hard_contact_active_set(
        Kff,
        f_eff,
        g_eff,
        Jf,
        equilibrium_jacobian_scale=scale,
        normal_compliance=compliance,
        initial_active=initial_active,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    full_u[free] = reduced.displacement
    gaps = g0 + J @ full_u
    return HardContactSolution(
        displacement=full_u,
        multipliers=reduced.multipliers.copy(),
        gaps=gaps,
        active=reduced.active.copy(),
        iterations=int(reduced.iterations),
        converged=bool(reduced.converged and np.all(gaps + compliance * reduced.multipliers >= -float(tolerance))),
    )


def solve_linear_hard_contact_with_dirichlet_sparse(
    stiffness,
    external_force: np.ndarray,
    gap_offset: np.ndarray,
    gap_jacobian: np.ndarray,
    *,
    fixed_dofs: np.ndarray,
    fixed_values: np.ndarray | None = None,
    equilibrium_jacobian_scale: float = 1.0,
    normal_compliance: np.ndarray | float | None = None,
    initial_active: np.ndarray | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 20,
    linear_solver: str = "direct",
    iterative_tolerance: float | None = None,
    iterative_max_iterations: int | None = None,
    iterative_fallback_to_direct: bool = True,
) -> HardContactSolution:
    """Sparse counterpart of :func:`solve_linear_hard_contact_with_dirichlet`.

    This keeps the global FEM tangent sparse and forms only the small active
    contact KKT border.  It is intended for full-size validation models where
    the dense active-set helper would exhaust memory.  The algebra and sign
    convention are identical to the dense implementation.
    """

    K = stiffness.tocsr() if issparse(stiffness) else csr_matrix(np.asarray(stiffness, dtype=float))
    f = np.asarray(external_force, dtype=float).reshape(-1)
    g0 = np.asarray(gap_offset, dtype=float).reshape(-1)
    J = np.asarray(gap_jacobian, dtype=float)
    if K.shape[0] != K.shape[1]:
        raise ValueError("stiffness must be square")
    if f.shape != (K.shape[0],):
        raise ValueError("external_force length must match stiffness")
    if J.ndim != 2 or J.shape[1] != K.shape[0]:
        raise ValueError("gap_jacobian must have shape (n_constraints, n_dofs)")
    if g0.shape != (J.shape[0],):
        raise ValueError("gap_offset length must match number of constraints")
    compliance = _as_constraint_compliance(normal_compliance, J.shape[0])
    scale = float(equilibrium_jacobian_scale)
    if scale <= 0.0:
        raise ValueError("equilibrium_jacobian_scale must be positive")
    fixed = np.asarray(fixed_dofs, dtype=np.int64).reshape(-1)
    if np.any(fixed < 0) or (fixed.size and int(fixed.max()) >= K.shape[0]):
        raise ValueError("fixed_dofs reference a dof outside stiffness")
    if np.unique(fixed).shape[0] != fixed.shape[0]:
        raise ValueError("fixed_dofs must not contain duplicates")
    values = np.zeros(fixed.shape[0], dtype=float) if fixed_values is None else np.asarray(fixed_values, dtype=float).reshape(-1)
    if values.shape != fixed.shape:
        raise ValueError("fixed_values must match fixed_dofs")

    full_u = np.zeros(K.shape[0], dtype=float)
    full_u[fixed] = values
    free_mask = np.ones(K.shape[0], dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    if free.size == 0:
        gaps = g0 + J @ full_u
        active = gaps < -float(tolerance)
        return HardContactSolution(
            displacement=full_u,
            multipliers=np.zeros(J.shape[0], dtype=float),
            gaps=gaps,
            active=active,
            iterations=0,
            converged=not bool(np.any(active)),
        )

    f_eff = np.asarray(f[free] - (K[free][:, fixed] @ values if fixed.size else 0.0), dtype=float).reshape(-1)
    Jf_dense = J[:, free]
    Jc_values = J[:, fixed] @ values if fixed.size else 0.0
    g_eff = np.asarray(g0 + Jc_values, dtype=float).reshape(-1)
    solver_mode = str(linear_solver).lower()
    if solver_mode not in {"direct", "primal_cg", "cg", "auto"}:
        raise ValueError("linear_solver must be 'direct', 'primal_cg', 'cg', or 'auto'")
    if solver_mode in {"primal_cg", "cg", "auto"} and J.shape[0] and np.all(compliance > 0.0):
        Kff_csr = K[free][:, free].tocsr()
        try:
            reduced = _solve_regularized_hard_contact_primal_cg(
                Kff_csr,
                f_eff,
                g_eff,
                Jf_dense,
                compliance,
                scale=scale,
                initial_active=initial_active,
                tolerance=float(tolerance),
                max_iterations=max_iterations,
                iterative_tolerance=iterative_tolerance,
                iterative_max_iterations=iterative_max_iterations,
            )
            full_u[free] = reduced.displacement
            gaps = g0 + J @ full_u
            return HardContactSolution(
                displacement=full_u,
                multipliers=reduced.multipliers.copy(),
                gaps=gaps,
                active=reduced.active.copy(),
                iterations=int(reduced.iterations),
                converged=bool(reduced.converged and np.all(gaps + compliance * reduced.multipliers >= -float(tolerance))),
            )
        except _IterativeHardContactSolveFailed:
            if not iterative_fallback_to_direct:
                raise

    Kff = K[free][:, free].tocsc()
    lu = splu(Kff)
    unconstrained = np.asarray(lu.solve(f_eff), dtype=float).reshape(-1)
    if initial_active is None:
        active = g_eff + Jf_dense @ unconstrained < -float(tolerance)
    else:
        active = np.asarray(initial_active, dtype=bool).reshape(-1)
        if active.shape != g_eff.shape:
            raise ValueError("initial_active must match constraints")

    u_free = unconstrained.copy()
    lam = np.zeros(J.shape[0], dtype=float)
    converged = False
    iterations = 0
    for iterations in range(1, max(1, int(max_iterations)) + 1):
        active_ids = np.flatnonzero(active)
        if active_ids.size:
            Ja_dense = Jf_dense[active_ids]
            influence = np.asarray(lu.solve(Ja_dense.T), dtype=float)
            schur = float(scale) * (Ja_dense @ influence) + np.diag(compliance[active_ids])
            rhs = -g_eff[active_ids] - Ja_dense @ unconstrained
            try:
                lam_active = np.linalg.solve(schur, rhs)
            except np.linalg.LinAlgError:
                lam_active, *_ = np.linalg.lstsq(schur, rhs, rcond=None)
            u_free = unconstrained + float(scale) * (influence @ lam_active)
            lam = np.zeros(J.shape[0], dtype=float)
            lam[active_ids] = lam_active
        else:
            u_free = unconstrained.copy()
            lam = np.zeros(J.shape[0], dtype=float)
        gaps_free = g_eff + Jf_dense @ u_free
        effective_gaps = gaps_free + compliance * lam
        next_active = active.copy()
        next_active[effective_gaps < -float(tolerance)] = True
        next_active[lam < -float(tolerance)] = False
        if np.array_equal(next_active, active) and np.all(effective_gaps >= -float(tolerance)) and np.all(lam >= -float(tolerance)):
            converged = True
            break
        active = next_active

    full_u[free] = u_free
    gaps = g0 + J @ full_u
    lam[np.abs(lam) <= float(tolerance)] = 0.0
    return HardContactSolution(
        displacement=full_u,
        multipliers=lam.copy(),
        gaps=gaps,
        active=active.copy(),
        iterations=int(iterations),
        converged=bool(converged and np.all(gaps + compliance * lam >= -float(tolerance))),
    )


def solve_linear_hard_contact_from_samples_with_dirichlet(
    stiffness: np.ndarray,
    external_force: np.ndarray,
    samples: Iterable[Any],
    *,
    n_total_dofs: int,
    fixed_dofs: np.ndarray,
    fixed_values: np.ndarray | None = None,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
    equilibrium_jacobian_scale: float = 1.0,
    normal_compliance: np.ndarray | float | None = None,
    initial_active: np.ndarray | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 20,
) -> HardContactSolution:
    """Solve sample-based hard contact with prescribed DOFs."""

    gap_offset, gap_jacobian = hard_contact_gap_jacobian_from_samples(
        samples,
        n_total_dofs=int(n_total_dofs),
        slave_dof_offset=int(slave_dof_offset),
        master_dof_offset=int(master_dof_offset),
    )
    return solve_linear_hard_contact_with_dirichlet(
        stiffness,
        external_force,
        gap_offset,
        gap_jacobian,
        fixed_dofs=fixed_dofs,
        fixed_values=fixed_values,
        equilibrium_jacobian_scale=equilibrium_jacobian_scale,
        normal_compliance=normal_compliance,
        initial_active=initial_active,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _solve_regularized_hard_contact_primal_cg(
    Kff: csr_matrix,
    f_eff: np.ndarray,
    g_eff: np.ndarray,
    Jf_dense: np.ndarray,
    compliance: np.ndarray,
    *,
    scale: float,
    initial_active: np.ndarray | None,
    tolerance: float,
    max_iterations: int,
    iterative_tolerance: float | None,
    iterative_max_iterations: int | None,
) -> HardContactSolution:
    """Solve the compliant active set through a primal SPD CG system.

    For positive normal compliance, eliminating the multipliers from
    ``K u - s J^T lambda = f`` and ``g0 + J u + C lambda = 0`` gives the
    equivalent primal system
    ``(K + s J_a^T C_a^-1 J_a) u = f - s J_a^T C_a^-1 g_a``.
    This avoids a sparse LU factorization of ``K`` while preserving the same
    active-set equations.  Callers fall back to the direct Schur path if the
    iterative residual does not meet the requested tolerance.
    """

    f = np.asarray(f_eff, dtype=float).reshape(-1)
    g0 = np.asarray(g_eff, dtype=float).reshape(-1)
    J = np.asarray(Jf_dense, dtype=float)
    compliance_arr = np.asarray(compliance, dtype=float).reshape(-1)
    if np.any(compliance_arr <= 0.0):
        raise _IterativeHardContactSolveFailed("primal CG requires positive compliance")
    cg_rtol = 1.0e-10 if iterative_tolerance is None else float(iterative_tolerance)
    cg_atol = max(float(tolerance) * 1.0e-2, 1.0e-12)
    maxiter = int(iterative_max_iterations) if iterative_max_iterations is not None else max(200, min(2000, Kff.shape[0] // 10))
    unconstrained = _cg_checked(
        Kff,
        f,
        rtol=cg_rtol,
        atol=cg_atol,
        maxiter=maxiter,
        x0=None,
        diagonal_preconditioner=np.asarray(Kff.diagonal(), dtype=float),
    )
    if initial_active is None:
        active = g0 + J @ unconstrained < -float(tolerance)
    else:
        active = np.asarray(initial_active, dtype=bool).reshape(-1)
        if active.shape != g0.shape:
            raise ValueError("initial_active must match constraints")
    u = unconstrained.copy()
    lam = np.zeros(J.shape[0], dtype=float)
    converged = False
    iterations = 0
    for iterations in range(1, max(1, int(max_iterations)) + 1):
        active_ids = np.flatnonzero(active)
        if active_ids.size:
            Ja = J[active_ids]
            inv_c = 1.0 / compliance_arr[active_ids]

            def matvec(value: np.ndarray) -> np.ndarray:
                v = np.asarray(value, dtype=float)
                return np.asarray(Kff @ v, dtype=float) + float(scale) * np.asarray(Ja.T @ (inv_c * (Ja @ v)), dtype=float)

            rhs = f - float(scale) * np.asarray(Ja.T @ (inv_c * g0[active_ids]), dtype=float)
            diag = np.asarray(Kff.diagonal(), dtype=float) + float(scale) * np.sum((Ja * Ja) * inv_c[:, None], axis=0)
            u = _cg_checked(
                LinearOperator(Kff.shape, matvec=matvec, dtype=float),
                rhs,
                rtol=cg_rtol,
                atol=cg_atol,
                maxiter=maxiter,
                x0=u,
                diagonal_preconditioner=diag,
            )
            lam = np.zeros(J.shape[0], dtype=float)
            lam[active_ids] = -inv_c * (g0[active_ids] + Ja @ u)
        else:
            u = unconstrained.copy()
            lam = np.zeros(J.shape[0], dtype=float)
        gaps = g0 + J @ u
        effective_gaps = gaps + compliance_arr * lam
        next_active = active.copy()
        next_active[effective_gaps < -float(tolerance)] = True
        next_active[lam < -float(tolerance)] = False
        if np.array_equal(next_active, active) and np.all(effective_gaps >= -float(tolerance)) and np.all(lam >= -float(tolerance)):
            converged = True
            break
        active = next_active
    lam[np.abs(lam) <= float(tolerance)] = 0.0
    return HardContactSolution(
        displacement=u,
        multipliers=lam,
        gaps=g0 + J @ u,
        active=active,
        iterations=int(iterations),
        converged=bool(converged),
    )


def _cg_checked(
    operator,
    rhs: np.ndarray,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
    x0: np.ndarray | None,
    diagonal_preconditioner: np.ndarray | None = None,
) -> np.ndarray:
    """Run CG and require the achieved algebraic residual to satisfy tolerance."""

    b = np.asarray(rhs, dtype=float).reshape(-1)
    M = None
    if diagonal_preconditioner is not None:
        diag = np.asarray(diagonal_preconditioner, dtype=float).reshape(-1)
        safe_diag = np.where(np.abs(diag) > 1.0e-30, diag, 1.0)
        M = LinearOperator((b.size, b.size), matvec=lambda value: np.asarray(value, dtype=float) / safe_diag, dtype=float)
    solution, info = cg(operator, b, x0=x0, rtol=float(rtol), atol=float(atol), maxiter=int(maxiter), M=M)
    if info != 0:
        raise _IterativeHardContactSolveFailed(f"CG did not converge, info={info}")
    residual = np.asarray(operator @ solution, dtype=float) - b
    residual_norm = float(np.linalg.norm(residual))
    allowed = float(atol) + float(rtol) * max(1.0, float(np.linalg.norm(b)))
    if residual_norm > max(10.0 * allowed, 1.0e-9):
        raise _IterativeHardContactSolveFailed(f"CG residual {residual_norm:.6e} exceeds {allowed:.6e}")
    return np.asarray(solution, dtype=float).reshape(-1)


def hard_contact_pressure_compliance_from_samples(
    samples: Iterable[Any],
    *,
    pressure_stiffness: float | None = None,
) -> np.ndarray:
    """Return force-multiplier compliance from sample pressure stiffness.

    ``pressure_stiffness`` has pressure-overclosure units.  Since hard-contact
    multipliers are nodal/contact-resultant forces, each sample compliance is
    ``1 / (pressure_stiffness * area)``.  When ``pressure_stiffness`` is not
    supplied, each sample may provide a positive ``stiffness`` attribute.
    """

    values: list[float] = []
    for sample in samples:
        area = float(getattr(sample, "area", 0.0))
        if area <= 0.0:
            raise ValueError("sample area must be positive for pressure compliance")
        k = float(pressure_stiffness) if pressure_stiffness is not None else float(getattr(sample, "stiffness", 0.0))
        if k <= 0.0:
            raise ValueError("pressure stiffness must be positive for pressure compliance")
        values.append(1.0 / (k * area))
    return np.asarray(values, dtype=float)


def _scatter_gap_row(row: np.ndarray, nodes: np.ndarray, weights: np.ndarray, normal: np.ndarray, offset: int, *, sign: float) -> None:
    for node, weight in zip(nodes, weights, strict=True):
        base = int(offset) + 3 * int(node)
        if base < 0 or base + 2 >= row.shape[0]:
            raise ValueError("sample node references a dof outside n_total_dofs")
        row[base : base + 3] += float(sign) * float(weight) * normal


def _dense_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray(), dtype=float)
    return np.asarray(value, dtype=float)


def _sparse_solve_vector(matrix, rhs: np.ndarray) -> np.ndarray:
    solution = spsolve(matrix, np.asarray(rhs, dtype=float).reshape(-1))
    return np.asarray(solution, dtype=float).reshape(-1)


def _as_constraint_compliance(value: np.ndarray | float | None, n_constraints: int) -> np.ndarray:
    n = int(n_constraints)
    if value is None:
        return np.zeros(n, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        arr = np.full(n, float(arr), dtype=float)
    else:
        arr = arr.reshape(-1)
    if arr.shape != (n,):
        raise ValueError("normal_compliance must be scalar or match number of constraints")
    if np.any(arr < 0.0):
        raise ValueError("normal_compliance must be non-negative")
    return arr
