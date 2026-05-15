"""Clean-room CalculiX-aligned nonlinear TET4 mechanics backend.

This module does not copy or import CalculiX source code.  It implements the
mechanics side of the validation model using public finite-element formulas and
CalculiX-compatible modeling choices where practical:

- C3D4/TET4 topology.
- Total-Lagrangian St. Venant-Kirchhoff response for ``*ELASTIC, NLGEOM`` style
  diagnostics.
- Consistent TET4 mass matrix for direct dynamics alignment.
- HHT/Newmark parameters matching the CalculiX ``*DYNAMIC, ALPHA=...``
  convention.
- A swappable contact-geometry interface so the contact query can be replaced
  by the SFC dynamic SDF while the mechanics backend remains unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

TRIANGLE_QUADRATURE_BARYCENTRIC = np.asarray(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ],
    dtype=float,
)
TRIANGLE_QUADRATURE_WEIGHTS = np.full(3, 1.0 / 3.0, dtype=float)


@dataclass(frozen=True, slots=True)
class ContactSample:
    """One slave-surface contact quadrature sample."""

    node_ids: np.ndarray
    shape_weights: np.ndarray
    gap: float
    normal: np.ndarray
    area: float
    stiffness: float


class ContactGeometry(Protocol):
    """Protocol for a contact-query provider.

    The mechanics backend only consumes contact samples.  A provider may use a
    rigid plane, direct closest-point projection, or the project-specific
    dynamic SDF to compute the gap and normal.
    """

    def samples(self, x_current: np.ndarray) -> Iterable[ContactSample]:
        """Yield contact quadrature samples for the current configuration."""


@dataclass(frozen=True, slots=True)
class PlaneContactGeometry:
    """Rigid horizontal plane contact geometry with triangle quadrature."""

    faces: np.ndarray
    plane_z: float
    stiffness: float
    normal: np.ndarray | None = None

    def samples(self, x_current: np.ndarray) -> Iterable[ContactSample]:
        faces = np.asarray(self.faces, dtype=np.int64)
        normal = np.asarray([0.0, 0.0, 1.0] if self.normal is None else self.normal, dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 0.0:
            raise ValueError("contact normal must be nonzero")
        normal = normal / normal_norm
        bary = TRIANGLE_QUADRATURE_BARYCENTRIC
        weights = TRIANGLE_QUADRATURE_WEIGHTS
        for face in faces:
            tri = x_current[face]
            area = 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
            if area <= 0.0:
                continue
            for shape, q_weight in zip(bary, weights, strict=True):
                point = shape @ tri
                gap = float((point - np.asarray([0.0, 0.0, self.plane_z], dtype=float)) @ normal)
                yield ContactSample(
                    node_ids=np.asarray(face, dtype=np.int64),
                    shape_weights=np.asarray(shape, dtype=float),
                    gap=gap,
                    normal=normal,
                    area=area * float(q_weight),
                    stiffness=float(self.stiffness),
                )


@dataclass(frozen=True, slots=True)
class MechanicsModel:
    """Reference TET4 model data for the aligned backend."""

    X: np.ndarray
    elements: np.ndarray
    E: float
    nu: float
    density: float
    volumes: np.ndarray
    shape_grads: np.ndarray
    mass_matrix: csr_matrix

    @classmethod
    def from_tet4_mesh(
        cls,
        X: np.ndarray,
        elements: np.ndarray,
        *,
        E: float,
        nu: float,
        density: float,
    ) -> "MechanicsModel":
        X_arr = np.asarray(X, dtype=float)
        elements_arr = np.asarray(elements, dtype=np.int64)
        volumes, grads = tet4_reference_data(X_arr, elements_arr)
        mass = assemble_consistent_mass(X_arr.shape[0], elements_arr, volumes, density)
        return cls(X_arr, elements_arr, float(E), float(nu), float(density), volumes, grads, mass)

    @property
    def n_nodes(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_dofs(self) -> int:
        return 3 * self.n_nodes


@dataclass(slots=True)
class MechanicsState:
    """Current displacement state for HHT/Newmark dynamics."""

    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    time: float = 0.0


@dataclass(slots=True)
class InternalResponse:
    """StVK internal response and tangent."""

    force: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    von_mises: np.ndarray
    strain_energy: float
    material_tangent: csr_matrix
    geometric_tangent: csr_matrix

    @property
    def tangent(self) -> csr_matrix:
        return (self.material_tangent + self.geometric_tangent).tocsr()


@dataclass(slots=True)
class ContactResponse:
    """Penalty contact force, tangent, and diagnostics."""

    force: np.ndarray
    tangent: csr_matrix
    min_gap: float
    max_penetration: float
    active_count: int
    normal_force: float
    energy: float


@dataclass(slots=True)
class StaticForceState:
    """Static force bookkeeping in both SFC and CalculiX residual signs.

    ``residual`` is the SFC residual convention used by local tangent checks:
    ``f_int - f_ext``.  Contact is stored as an internal spring contribution,
    so ``f_int`` includes ``-f_contact``.

    ``calculix_rhs_balance`` is the CalculiX ``calcresidual.c`` history sign:
    ``f_ext - f_int``.  It is the quantity stored as ``fextini - fini`` after
    an accepted increment and used in the HHT alpha history term.
    """

    residual: np.ndarray
    calculix_rhs_balance: np.ndarray
    internal_force: np.ndarray
    external_force: np.ndarray
    contact_internal_force: np.ndarray
    contact_force: np.ndarray
    tangent: csr_matrix


@dataclass(frozen=True, slots=True)
class NewtonAcceptanceMetrics:
    """Clean-room CalculiX-style Newton acceptance metrics for one iteration."""

    iteration: int
    ram: float
    qam: float
    cam: float
    uam: float
    residual_ratio: float
    correction_ratio: float
    active_contact_count: int
    contact_element_change: int
    contact_energy: float
    total_energy: float
    energy_residual: float
    energy_stabilization: bool
    accepted_by_calculix_style: bool
    reason: str


@dataclass(slots=True)
class StepDiagnostics:
    """Combined mechanics/contact diagnostics for one state."""

    internal: InternalResponse
    contact: ContactResponse
    kinetic_energy: float
    gravitational_energy: float
    total_energy: float
    newton_iterations: int = 0
    newton_residual_norm: float = 0.0
    newton_acceptance_policy: str = "relative_correction"
    newton_acceptance_reason: str = ""
    newton_acceptance_metrics: list[NewtonAcceptanceMetrics] = field(default_factory=list)


def hht_newmark_parameters(alpha: float) -> tuple[float, float]:
    """Return HHT/Newmark ``beta`` and ``gamma`` for CalculiX alpha syntax."""

    alpha_value = float(alpha)
    if not (-1.0 / 3.0 <= alpha_value <= 0.0):
        raise ValueError("HHT alpha must lie in [-1/3, 0]")
    return 0.25 * (1.0 - alpha_value) ** 2, 0.5 - alpha_value


def tet4_reference_data(X: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return positive TET4 volumes and reference shape gradients."""

    parent_grads = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    volumes = np.zeros(elements.shape[0], dtype=float)
    grads = np.zeros((elements.shape[0], 4, 3), dtype=float)
    for e, element in enumerate(elements):
        Xe = X[element]
        Dm = np.column_stack((Xe[1] - Xe[0], Xe[2] - Xe[0], Xe[3] - Xe[0]))
        det = float(np.linalg.det(Dm))
        if det <= 0.0:
            raise ValueError("TET4 reference element must have positive orientation")
        volumes[e] = det / 6.0
        grads[e] = parent_grads @ np.linalg.inv(Dm)
    return volumes, grads


def assemble_consistent_mass(n_nodes: int, elements: np.ndarray, volumes: np.ndarray, density: float) -> csr_matrix:
    """Assemble the consistent C3D4/TET4 mass matrix."""

    rho = float(density)
    if rho <= 0.0:
        raise ValueError("density must be positive")
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    scalar = np.full((4, 4), 1.0, dtype=float)
    np.fill_diagonal(scalar, 2.0)
    for element, volume in zip(elements, volumes, strict=True):
        dofs = _element_dofs(element)
        Me = np.kron(rho * float(volume) * scalar / 20.0, np.eye(3))
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(rr.ravel())
        cols.append(cc.ravel())
        data.append(Me.ravel())
    if not data:
        return csr_matrix((3 * n_nodes, 3 * n_nodes), dtype=float)
    return coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * n_nodes, 3 * n_nodes),
    ).tocsr()


def stvk_internal_response(model: MechanicsModel, x_current: np.ndarray, *, assemble_tangent: bool = True) -> InternalResponse:
    """Return total-Lagrangian StVK internal force and tangent."""

    x = np.asarray(x_current, dtype=float)
    lam, mu = _lame_parameters(model.E, model.nu)
    identity = np.eye(3)
    force = np.zeros_like(x)
    strains = np.zeros((model.elements.shape[0], 3, 3), dtype=float)
    stresses = np.zeros((model.elements.shape[0], 3, 3), dtype=float)
    vm = np.zeros(model.elements.shape[0], dtype=float)
    energy = 0.0
    mat_rows: list[int] = []
    mat_cols: list[int] = []
    mat_data: list[float] = []
    geo_rows: list[int] = []
    geo_cols: list[int] = []
    geo_data: list[float] = []
    for e, element in enumerate(model.elements):
        xe = x[element]
        grad = model.shape_grads[e]
        F = xe.T @ grad
        green = 0.5 * (F.T @ F - identity)
        second_piola = lam * float(np.trace(green)) * identity + 2.0 * mu * green
        first_piola = F @ second_piola
        for a in range(4):
            force[int(element[a])] += model.volumes[e] * (first_piola @ grad[a])
        if assemble_tangent:
            _append_stvk_tangent_blocks(
                model.elements[e],
                model.volumes[e],
                grad,
                F,
                second_piola,
                lam,
                mu,
                mat_rows,
                mat_cols,
                mat_data,
                geo_rows,
                geo_cols,
                geo_data,
            )
        J = max(float(np.linalg.det(F)), 1.0e-12)
        cauchy = (first_piola @ F.T) / J
        strains[e] = green
        stresses[e] = cauchy
        vm[e] = von_mises(cauchy)
        energy += model.volumes[e] * 0.5 * float(_voigt_strain(green) @ _voigt_stress(second_piola))
    material = coo_matrix((mat_data, (mat_rows, mat_cols)), shape=(model.n_dofs, model.n_dofs)).tocsr()
    geometric = coo_matrix((geo_data, (geo_rows, geo_cols)), shape=(model.n_dofs, model.n_dofs)).tocsr()
    return InternalResponse(force, strains, stresses, vm, float(energy), material, geometric)


def assemble_contact_response(samples: Iterable[ContactSample], n_nodes: int) -> ContactResponse:
    """Assemble penalty contact force and residual tangent from contact samples."""

    force = np.zeros((n_nodes, 3), dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    min_gap = np.inf
    max_penetration = 0.0
    active_count = 0
    normal_force = 0.0
    energy = 0.0
    for sample in samples:
        gap = float(sample.gap)
        min_gap = min(min_gap, gap)
        penetration = max(-gap, 0.0)
        if penetration <= 0.0:
            continue
        node_ids = np.asarray(sample.node_ids, dtype=np.int64)
        weights = np.asarray(sample.shape_weights, dtype=float)
        normal = np.asarray(sample.normal, dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-30)
        area = float(sample.area)
        stiffness = float(sample.stiffness)
        lam = stiffness * area * penetration
        tangent_scale = stiffness * area
        active_count += 1
        normal_force += lam
        max_penetration = max(max_penetration, penetration)
        energy += 0.5 * stiffness * area * penetration * penetration
        for local, node in enumerate(node_ids):
            force[int(node)] += weights[local] * lam * normal
        for a, node_a in enumerate(node_ids):
            for b, node_b in enumerate(node_ids):
                block = weights[a] * weights[b] * tangent_scale * np.outer(normal, normal)
                for i in range(3):
                    row = 3 * int(node_a) + i
                    for j in range(3):
                        col = 3 * int(node_b) + j
                        rows.append(row)
                        cols.append(col)
                        data.append(float(block[i, j]))
    if not np.isfinite(min_gap):
        min_gap = 0.0
    tangent = coo_matrix((data, (rows, cols)), shape=(3 * n_nodes, 3 * n_nodes)).tocsr()
    return ContactResponse(force, tangent, float(min_gap), float(max_penetration), active_count, float(normal_force), float(energy))


def evaluate_state(
    model: MechanicsModel,
    state: MechanicsState,
    contact_geometry: ContactGeometry,
    *,
    gravity: float,
    assemble_tangent: bool = True,
) -> StepDiagnostics:
    """Evaluate internal/contact response and energies for a state."""

    internal = stvk_internal_response(model, state.x, assemble_tangent=assemble_tangent)
    contact = assemble_contact_response(contact_geometry.samples(state.x), model.n_nodes)
    kinetic = 0.5 * float(state.v.reshape(-1) @ (model.mass_matrix @ state.v.reshape(-1)))
    gravitational = float(np.sum(_nodal_gravity_loads(model, gravity).reshape((-1, 3))[:, 2] * -state.x[:, 2]))
    total = kinetic + internal.strain_energy + gravitational + contact.energy
    return StepDiagnostics(internal, contact, kinetic, gravitational, total)


def initial_state(
    model: MechanicsModel,
    contact_geometry: ContactGeometry,
    *,
    gravity: float,
    initial_velocity: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[MechanicsState, np.ndarray]:
    """Return initial state and static residual for HHT stepping."""

    velocity = np.zeros_like(model.X)
    velocity[:] = np.asarray(initial_velocity, dtype=float)
    state = MechanicsState(model.X.copy(), velocity, np.zeros_like(model.X), time=0.0)
    static_state = static_force_state(model, state.x, contact_geometry, gravity=gravity)
    acceleration = np.asarray(spsolve(model.mass_matrix.tocsc(), -static_state.residual), dtype=float).reshape((-1, 3))
    state.a = acceleration
    return state, static_state.calculix_rhs_balance


def static_residual_and_tangent(
    model: MechanicsModel,
    x_current: np.ndarray,
    contact_geometry: ContactGeometry,
    *,
    gravity: float,
) -> tuple[np.ndarray, csr_matrix]:
    """Return ``f_int - f_ext - f_contact`` and its tangent."""

    state = static_force_state(model, x_current, contact_geometry, gravity=gravity)
    return state.residual, state.tangent


def static_force_state(
    model: MechanicsModel,
    x_current: np.ndarray,
    contact_geometry: ContactGeometry,
    *,
    gravity: float,
) -> StaticForceState:
    """Return static force history with CalculiX-style contact bookkeeping.

    CalculiX face-to-face penalty contact is assembled as generated spring
    elements.  For the validation backend, the positive ``contact.force`` is
    the physical reaction force applied to the slave body, while the equivalent
    internal spring contribution stored in ``fini`` is ``-contact.force``.
    """

    internal = stvk_internal_response(model, x_current, assemble_tangent=True)
    contact = assemble_contact_response(contact_geometry.samples(x_current), model.n_nodes)
    external = _nodal_gravity_loads(model, gravity)
    contact_force = contact.force.reshape(-1)
    contact_internal = -contact_force
    total_internal = internal.force.reshape(-1) + contact_internal
    residual = total_internal - external
    tangent = (internal.tangent + contact.tangent).tocsr()
    return StaticForceState(
        residual=np.asarray(residual, dtype=float),
        calculix_rhs_balance=np.asarray(external - total_internal, dtype=float),
        internal_force=np.asarray(total_internal, dtype=float),
        external_force=np.asarray(external, dtype=float),
        contact_internal_force=np.asarray(contact_internal, dtype=float),
        contact_force=np.asarray(contact_force, dtype=float),
        tangent=tangent,
    )


def hht_step(
    model: MechanicsModel,
    state: MechanicsState,
    previous_static_residual: np.ndarray,
    contact_geometry: ContactGeometry,
    *,
    dt: float,
    gravity: float,
    alpha: float = -0.05,
    max_iterations: int = 12,
    tolerance: float = 1.0e-10,
    acceptance_policy: str = "relative_correction",
) -> tuple[MechanicsState, np.ndarray, StepDiagnostics]:
    """Advance one implicit HHT/Newmark step using Newton iterations.

    ``previous_static_residual`` is kept for API compatibility, but its
    semantics are CalculiX-style ``fextini - fini``.  Contact spring forces are
    stored in ``fini`` through their internal-force sign, and the returned
    history vector is recomputed at the accepted state.
    """

    if acceptance_policy not in {"relative_correction", "calculix_multicriteria"}:
        raise ValueError("acceptance_policy must be 'relative_correction' or 'calculix_multicriteria'")
    beta, gamma = hht_newmark_parameters(alpha)
    c0 = 1.0 / (beta * dt * dt)
    u = (state.x - model.X).reshape(-1)
    v = state.v.reshape(-1)
    a = state.a.reshape(-1)
    u_pred = u + dt * v + dt * dt * (0.5 - beta) * a
    v_pred = v + dt * (1.0 - gamma) * a
    u_guess = u_pred.copy()
    previous_rhs_balance = np.asarray(previous_static_residual, dtype=float)
    residual_norm = np.inf
    iteration_count = 0
    acceptance_reason = "iteration_limit"
    acceptance_metrics: list[NewtonAcceptanceMetrics] = []
    previous_energy = evaluate_state(model, state, contact_geometry, gravity=gravity, assemble_tangent=False).total_energy
    previous_ram: float | None = None
    previous_active_count: int | None = None
    for iteration in range(max(1, int(max_iterations))):
        x_guess = model.X + u_guess.reshape((-1, 3))
        static_state = static_force_state(model, x_guess, contact_geometry, gravity=gravity)
        a_guess = c0 * (u_guess - u_pred)
        dynamic_residual = (
            model.mass_matrix @ a_guess
            - (1.0 + float(alpha)) * static_state.calculix_rhs_balance
            + float(alpha) * previous_rhs_balance
        )
        residual_norm = float(np.linalg.norm(dynamic_residual))
        tangent = (model.mass_matrix * c0 + static_state.tangent * (1.0 + float(alpha))).tocsc()
        correction = np.asarray(spsolve(tangent, -dynamic_residual), dtype=float)
        u_guess += correction
        iteration_count = iteration + 1
        a_iter = c0 * (u_guess - u_pred)
        v_iter = v_pred + gamma * dt * a_iter
        trial_state = MechanicsState(
            model.X + u_guess.reshape((-1, 3)),
            v_iter.reshape((-1, 3)),
            a_iter.reshape((-1, 3)),
            time=state.time + float(dt),
        )
        trial_diagnostics = evaluate_state(model, trial_state, contact_geometry, gravity=gravity, assemble_tangent=False)
        metric = _newton_acceptance_metrics(
            model,
            dynamic_residual,
            correction,
            u_guess - u,
            trial_diagnostics,
            previous_energy=previous_energy,
            previous_ram=previous_ram,
            previous_active_count=previous_active_count,
            iteration=iteration_count,
            gravity=gravity,
        )
        acceptance_metrics.append(metric)
        previous_ram = metric.ram
        previous_active_count = metric.active_contact_count
        relative_correction_accept = np.linalg.norm(correction) <= tolerance * max(1.0, np.linalg.norm(u_guess))
        if acceptance_policy == "relative_correction" and relative_correction_accept:
            acceptance_reason = "relative_correction"
            break
        if acceptance_policy == "calculix_multicriteria" and metric.accepted_by_calculix_style:
            acceptance_reason = metric.reason
            break
    a_new = c0 * (u_guess - u_pred)
    v_new = v_pred + gamma * dt * a_new
    next_state = MechanicsState(
        model.X + u_guess.reshape((-1, 3)),
        v_new.reshape((-1, 3)),
        a_new.reshape((-1, 3)),
        time=state.time + float(dt),
    )
    diagnostics = evaluate_state(model, next_state, contact_geometry, gravity=gravity, assemble_tangent=True)
    accepted_static_state = static_force_state(model, next_state.x, contact_geometry, gravity=gravity)
    diagnostics.newton_iterations = iteration_count
    diagnostics.newton_residual_norm = residual_norm
    diagnostics.newton_acceptance_policy = acceptance_policy
    diagnostics.newton_acceptance_reason = acceptance_reason
    diagnostics.newton_acceptance_metrics = acceptance_metrics
    return next_state, accepted_static_state.calculix_rhs_balance, diagnostics


def _newton_acceptance_metrics(
    model: MechanicsModel,
    residual: np.ndarray,
    correction: np.ndarray,
    increment: np.ndarray,
    diagnostics: StepDiagnostics,
    *,
    previous_energy: float,
    previous_ram: float | None,
    previous_active_count: int | None,
    iteration: int,
    gravity: float,
) -> NewtonAcceptanceMetrics:
    """Return clean-room proxies for CalculiX `ram/qam/cam/uam` checks."""

    ram = float(np.max(np.abs(residual))) if residual.size else 0.0
    qam = _force_reference_scale(model, diagnostics, gravity)
    cam = float(np.max(np.abs(correction))) if correction.size else 0.0
    uam = float(np.max(np.abs(increment))) if increment.size else 0.0
    residual_ratio = ram / max(qam, 1.0e-30)
    correction_ratio = cam / max(uam, 1.0e-30)
    previous_active = diagnostics.contact.active_count if previous_active_count is None else int(previous_active_count)
    contact_change = abs(int(diagnostics.contact.active_count) - previous_active)
    energy_residual = float(diagnostics.total_energy - previous_energy)
    energy_scale = max(abs(float(previous_energy)), abs(float(diagnostics.total_energy)), 1.0)
    energy_stabilization = bool(diagnostics.contact.active_count > 0 and abs(energy_residual) > 5.0e-2 * energy_scale)

    residual_ok = residual_ratio <= 5.0e-3
    correction_ok = correction_ratio <= 1.0e-2
    relaxed_ok = (
        previous_ram is not None
        and previous_ram > 0.0
        and ram * cam <= 1.0e-2 * max(uam, 1.0e-30) * previous_ram
    )
    loose_residual_ok = residual_ratio <= 2.0e-2
    contact_stable = contact_change == 0
    accepted = bool(
        iteration > 1
        and residual_ok
        and contact_stable
        and not energy_stabilization
        and (correction_ok or relaxed_ok or loose_residual_ok or cam < 1.0e-8)
    )
    if accepted:
        reason = "calculix_multicriteria"
    elif iteration <= 1:
        reason = "need_second_iteration"
    elif not residual_ok:
        reason = "residual_ratio"
    elif not contact_stable:
        reason = "contact_element_change"
    elif energy_stabilization:
        reason = "energy_contact_stabilization"
    else:
        reason = "correction_ratio"
    return NewtonAcceptanceMetrics(
        iteration=int(iteration),
        ram=ram,
        qam=qam,
        cam=cam,
        uam=uam,
        residual_ratio=float(residual_ratio),
        correction_ratio=float(correction_ratio),
        active_contact_count=int(diagnostics.contact.active_count),
        contact_element_change=int(contact_change),
        contact_energy=float(diagnostics.contact.energy),
        total_energy=float(diagnostics.total_energy),
        energy_residual=energy_residual,
        energy_stabilization=energy_stabilization,
        accepted_by_calculix_style=accepted,
        reason=reason,
    )


def _force_reference_scale(model: MechanicsModel, diagnostics: StepDiagnostics, gravity: float) -> float:
    gravity_load = _nodal_gravity_loads(model, gravity)
    parts = [
        np.abs(gravity_load),
        np.abs(diagnostics.internal.force.reshape(-1)),
        np.abs(diagnostics.contact.force.reshape(-1)),
    ]
    values = np.concatenate([part.ravel() for part in parts])
    positive = values[values > 0.0]
    if positive.size == 0:
        return 1.0
    return max(float(np.mean(positive)), 1.0e-12)


def von_mises(stress: np.ndarray) -> float:
    """Return the von Mises equivalent stress for a 3-by-3 stress tensor."""

    s = np.asarray(stress, dtype=float)
    return float(
        np.sqrt(
            0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2)
            + 3.0 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[0, 2] ** 2)
        )
    )


def _lame_parameters(E: float, nu: float) -> tuple[float, float]:
    mu = float(E) / (2.0 * (1.0 + float(nu)))
    lam = float(E) * float(nu) / ((1.0 + float(nu)) * (1.0 - 2.0 * float(nu)))
    return lam, mu


def _voigt_strain(tensor: np.ndarray) -> np.ndarray:
    return np.asarray([tensor[0, 0], tensor[1, 1], tensor[2, 2], 2.0 * tensor[0, 1], 2.0 * tensor[1, 2], 2.0 * tensor[0, 2]])


def _voigt_stress(tensor: np.ndarray) -> np.ndarray:
    return np.asarray([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[1, 2], tensor[0, 2]])


def _element_dofs(element: np.ndarray) -> np.ndarray:
    conn = np.asarray(element, dtype=np.int64)
    return np.repeat(conn, 3) * 3 + np.tile(np.arange(3, dtype=np.int64), conn.size)


def _nodal_gravity_loads(model: MechanicsModel, gravity: float) -> np.ndarray:
    f = np.zeros(model.n_dofs, dtype=float)
    for element, volume in zip(model.elements, model.volumes, strict=True):
        nodal = model.density * float(volume) * np.asarray([0.0, 0.0, -float(gravity)], dtype=float) / 4.0
        for node in element:
            start = 3 * int(node)
            f[start : start + 3] += nodal
    return f


def _append_stvk_tangent_blocks(
    element: np.ndarray,
    volume: float,
    grad: np.ndarray,
    F: np.ndarray,
    second_piola: np.ndarray,
    lam: float,
    mu: float,
    mat_rows: list[int],
    mat_cols: list[int],
    mat_data: list[float],
    geo_rows: list[int],
    geo_cols: list[int],
    geo_data: list[float],
) -> None:
    fft = F @ F.T
    for a in range(4):
        ga = grad[a]
        Fga = F @ ga
        for b in range(4):
            gb = grad[b]
            Fgb = F @ gb
            material_block = volume * (lam * np.outer(Fga, Fgb) + mu * np.outer(Fgb, Fga) + mu * float(ga @ gb) * fft)
            geometric_block = volume * float(gb @ second_piola @ ga) * np.eye(3)
            for i in range(3):
                row = 3 * int(element[a]) + i
                for j in range(3):
                    col = 3 * int(element[b]) + j
                    mat_rows.append(row)
                    mat_cols.append(col)
                    mat_data.append(float(material_block[i, j]))
                    geo_rows.append(row)
                    geo_cols.append(col)
                    geo_data.append(float(geometric_block[i, j]))
