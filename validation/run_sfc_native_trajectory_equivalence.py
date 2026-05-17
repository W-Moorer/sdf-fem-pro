"""Native SFC full-trajectory equivalence for true field contact.

This runner builds all geometry, finite-element matrices, contact queries, and
time/load trajectories inside SFC.  It compares two SFC contact paths:

* candidate: ``DynamicNarrowBandSDF + field_contact`` interpolation queries;
* reference: exact analytic rigid-plane gap/normal with the same penalty law.

The purpose is a scoped full-solve trajectory equivalence check.  It uses no
external solver output.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact import SurfaceSample, compute_field_contact_constraints, field_penalty_contact_response  # noqa: E402
from sfc.fem import DeformableBody, assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.fem.constraints import eliminate_fixed_dofs, expand_reduced_vector, fixed_dofs_from_node_set  # noqa: E402
from sfc.fem.hex8 import hex8_center_strain_stress  # noqa: E402
from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.fem.tet4 import tet4_strain_displacement_matrix  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402
from validation.run_phase3_validation import structured_tet_block  # noqa: E402

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class NativeModel:
    mesh: VolumeMesh
    body: DeformableBody
    top_nodes: np.ndarray
    bottom_nodes: np.ndarray
    element_type: str


@dataclass(frozen=True, slots=True)
class ContactState:
    gaps: np.ndarray
    force: np.ndarray
    query_seconds: float
    active_count: int


@dataclass(frozen=True, slots=True)
class StaticSolveState:
    u: np.ndarray
    contact: ContactState
    iterations: int
    converged: bool
    correction_norm: float


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _structured_hex_block(nx: int, ny: int, nz: int, *, size: tuple[float, float, float]) -> VolumeMesh:
    if min(nx, ny, nz) <= 0:
        raise ValueError("mesh resolution must be positive")
    lx, ly, lz = size
    nodes: list[tuple[float, float, float]] = []

    def node_id(i: int, j: int, k: int) -> int:
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes.append((lx * i / nx, ly * j / ny, lz * k / nz))

    elements: list[tuple[int, int, int, int, int, int, int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements.append(
                    (
                        node_id(i, j, k),
                        node_id(i + 1, j, k),
                        node_id(i + 1, j + 1, k),
                        node_id(i, j + 1, k),
                        node_id(i, j, k + 1),
                        node_id(i + 1, j, k + 1),
                        node_id(i + 1, j + 1, k + 1),
                        node_id(i, j + 1, k + 1),
                    )
                )
    return VolumeMesh(np.asarray(nodes, dtype=float), np.asarray(elements, dtype=np.int64), element_type="hex8")


def _native_model(element_type: str, *, quick: bool) -> NativeModel:
    nx = ny = 1 if quick else 2
    nz = 1 if quick else 2
    size = (1.0, 1.0, 0.25)
    if element_type == "tet4":
        base = structured_tet_block(nx, ny, nz, size=size)
    elif element_type == "hex8":
        base = _structured_hex_block(nx, ny, nz, size=size)
    else:
        raise ValueError("element_type must be tet4 or hex8")

    X = base.X.copy()
    X[:, 2] += 0.055
    mesh = VolumeMesh(X, base.elements, element_type=element_type)
    body = DeformableBody(mesh=mesh, material={"E": 220.0, "nu": 0.30}, density=1.0)
    top_nodes = np.nonzero(np.isclose(mesh.X[:, 2], float(np.max(mesh.X[:, 2]))))[0]
    bottom_nodes = np.nonzero(np.isclose(mesh.X[:, 2], float(np.min(mesh.X[:, 2]))))[0]
    return NativeModel(mesh=mesh, body=body, top_nodes=top_nodes, bottom_nodes=bottom_nodes, element_type=element_type)


def _master_plane_sdf() -> DynamicNarrowBandSDF:
    X = np.asarray(
        [
            [-0.35, -0.35, 0.0],
            [1.35, -0.35, 0.0],
            [1.35, 1.35, 0.0],
            [-0.35, 1.35, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return DynamicNarrowBandSDF.build(
        X,
        faces,
        spacing=0.085,
        band_radius=0.32,
        origin=np.asarray([-0.35, -0.35, -0.22], dtype=float),
        shape=(21, 21, 10),
        cell_size=0.20,
        gradient_mode="finite_difference",
    )


def _bottom_node_samples(bottom_nodes: np.ndarray) -> list[SurfaceSample]:
    return [
        SurfaceSample(np.asarray([int(node)], dtype=np.int64), np.asarray([1.0], dtype=float), np.asarray([0], dtype=np.int64))
        for node in np.asarray(bottom_nodes, dtype=np.int64)
    ]


def _top_fixed_dofs_and_values(model: NativeModel, prescribed_z: float) -> tuple[np.ndarray, np.ndarray]:
    fixed = fixed_dofs_from_node_set(model.top_nodes, "xyz")
    values = np.zeros(fixed.size, dtype=float)
    values[np.asarray(fixed) % 3 == 2] = float(prescribed_z)
    return fixed, values


def _field_contact_state(
    x_current: np.ndarray,
    samples: list[SurfaceSample],
    sdf: DynamicNarrowBandSDF,
    *,
    stiffness: float,
) -> ContactState:
    start = perf_counter()
    constraints = compute_field_contact_constraints(x_current, samples, sdf)
    force, _Kc = field_penalty_contact_response(
        constraints,
        stiffness=stiffness,
        n_total_dofs=x_current.shape[0] * 3 + sdf.x_current.shape[0] * 3,
        slave_dof_offset=0,
        master_dof_offset=x_current.shape[0] * 3,
    )
    query_seconds = perf_counter() - start
    gaps = np.asarray([constraint.g for constraint in constraints], dtype=float)
    return ContactState(
        gaps=gaps,
        force=force[: x_current.shape[0] * 3],
        query_seconds=query_seconds,
        active_count=int(np.count_nonzero(gaps < 0.0)),
    )


def _analytic_plane_contact_state(
    x_current: np.ndarray,
    samples: list[SurfaceSample],
    *,
    stiffness: float,
) -> ContactState:
    gaps = np.zeros(len(samples), dtype=float)
    force = np.zeros(x_current.shape[0] * 3, dtype=float)
    start = perf_counter()
    for row, sample in enumerate(samples):
        point = sample.point(x_current)
        gap = float(point[2])
        gaps[row] = gap
        penetration = max(-gap, 0.0)
        if penetration <= 0.0:
            continue
        for node, weight in zip(sample.node_ids, sample.weights, strict=True):
            force[int(node) * 3 + 2] += float(stiffness) * penetration * float(weight)
    return ContactState(
        gaps=gaps,
        force=force,
        query_seconds=perf_counter() - start,
        active_count=int(np.count_nonzero(gaps < 0.0)),
    )


def _contact_state(
    mode: str,
    x_current: np.ndarray,
    samples: list[SurfaceSample],
    sdf: DynamicNarrowBandSDF,
    *,
    stiffness: float,
) -> ContactState:
    if mode == "field":
        return _field_contact_state(x_current, samples, sdf, stiffness=stiffness)
    if mode == "analytic":
        return _analytic_plane_contact_state(x_current, samples, stiffness=stiffness)
    raise ValueError("mode must be field or analytic")


def _solve_static_increment(
    model: NativeModel,
    stiffness_matrix,
    samples: list[SurfaceSample],
    sdf: DynamicNarrowBandSDF,
    *,
    mode: str,
    u0: np.ndarray,
    prescribed_top_z: float,
    contact_stiffness: float,
    max_iterations: int,
    tolerance: float,
) -> StaticSolveState:
    fixed, values = _top_fixed_dofs_and_values(model, prescribed_top_z)
    u = u0.copy()
    converged = False
    correction_norm = np.inf
    contact = _contact_state(mode, model.mesh.X + u.reshape((-1, 3)), samples, sdf, stiffness=contact_stiffness)
    for iteration in range(max_iterations):
        x_current = model.mesh.X + u.reshape((-1, 3))
        contact = _contact_state(mode, x_current, samples, sdf, stiffness=contact_stiffness)
        reduced_matrix, reduced_rhs, free = eliminate_fixed_dofs(stiffness_matrix, contact.force, fixed, values)
        if free.size:
            reduced_u = np.asarray(spsolve(reduced_matrix, reduced_rhs), dtype=float)
        else:
            reduced_u = np.empty(0, dtype=float)
        u_new = expand_reduced_vector(reduced_u, free, model.body.n_dofs, fixed, values)
        correction_norm = float(np.linalg.norm(u_new - u))
        u = 0.65 * u_new + 0.35 * u
        if correction_norm <= tolerance * max(1.0, float(np.linalg.norm(u_new))):
            u = u_new
            converged = True
            break
    contact = _contact_state(mode, model.mesh.X + u.reshape((-1, 3)), samples, sdf, stiffness=contact_stiffness)
    return StaticSolveState(
        u=u,
        contact=contact,
        iterations=iteration + 1,
        converged=converged,
        correction_norm=correction_norm,
    )


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    if denom <= 1.0e-30:
        return float(np.linalg.norm(value - reference))
    return float(np.linalg.norm(value - reference) / denom)


def _von_mises_voigt(stress: np.ndarray) -> float:
    s = np.asarray(stress, dtype=float).ravel()
    if s.size != 6:
        raise ValueError("stress must be Voigt-6")
    sxx, syy, szz, sxy, syz, sxz = s
    return float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + syz**2 + sxz**2)))


def _stress_metrics(model: NativeModel, u: np.ndarray) -> tuple[float, float]:
    E = float(model.body.material["E"])
    nu = float(model.body.material["nu"])
    values: list[float] = []
    strain_norms: list[float] = []
    u_nodes = u.reshape((-1, 3))
    if model.element_type == "tet4":
        C = isotropic_linear_elasticity_matrix(E, nu)
        for element in model.mesh.elements:
            B = tet4_strain_displacement_matrix(model.mesh.X[element])
            strain = B @ u_nodes[element].reshape(12)
            stress = C @ strain
            values.append(_von_mises_voigt(stress))
            strain_norms.append(float(np.linalg.norm(strain)))
    else:
        for element in model.mesh.elements:
            strain, stress = hex8_center_strain_stress(model.mesh.X[element], u_nodes[element], E, nu)
            values.append(_von_mises_voigt(stress))
            strain_norms.append(float(np.linalg.norm(strain)))
    return float(np.max(values)) if values else 0.0, float(np.max(strain_norms)) if strain_norms else 0.0


def _case_status(max_state_error: float, max_gap_error: float, max_force_error: float, converged: bool = True) -> str:
    if converged and max_state_error <= 1.0e-10 and max_gap_error <= 1.0e-10 and max_force_error <= 1.0e-10:
        return "passed"
    return "failed"


def _static_trajectory(
    element_type: str,
    *,
    quick: bool,
    sdf: DynamicNarrowBandSDF,
    contact_stiffness: float,
) -> tuple[Row, list[Row], list[Row]]:
    model = _native_model(element_type, quick=quick)
    samples = _bottom_node_samples(model.bottom_nodes)
    stiffness = assemble_stiffness_matrix(model.body).tocsr()
    n_steps = 7 if quick else 12
    max_top_displacement = -0.105 if quick else -0.120
    u_field = np.zeros(model.body.n_dofs, dtype=float)
    u_ref = np.zeros(model.body.n_dofs, dtype=float)
    history_rows: list[Row] = []
    sample_rows: list[Row] = []
    max_state_error = 0.0
    max_gap_error = 0.0
    max_force_error = 0.0
    max_reaction_error = 0.0
    total_field_query = 0.0
    total_ref_query = 0.0
    max_iterations = 0
    all_converged = True

    for step in range(n_steps):
        load_factor = (step + 1) / n_steps
        prescribed_z = max_top_displacement * load_factor
        field_state = _solve_static_increment(
            model,
            stiffness,
            samples,
            sdf,
            mode="field",
            u0=u_field,
            prescribed_top_z=prescribed_z,
            contact_stiffness=contact_stiffness,
            max_iterations=160,
            tolerance=1.0e-12,
        )
        ref_state = _solve_static_increment(
            model,
            stiffness,
            samples,
            sdf,
            mode="analytic",
            u0=u_ref,
            prescribed_top_z=prescribed_z,
            contact_stiffness=contact_stiffness,
            max_iterations=160,
            tolerance=1.0e-12,
        )
        u_field = field_state.u
        u_ref = ref_state.u
        state_error = float(np.linalg.norm(u_field - u_ref))
        gap_error = float(np.max(np.abs(field_state.contact.gaps - ref_state.contact.gaps)))
        force_error = _relative_error(field_state.contact.force, ref_state.contact.force)
        reaction_field = float(np.sum(field_state.contact.force[2::3]))
        reaction_ref = float(np.sum(ref_state.contact.force[2::3]))
        reaction_error = abs(reaction_field - reaction_ref)
        max_state_error = max(max_state_error, state_error)
        max_gap_error = max(max_gap_error, gap_error)
        max_force_error = max(max_force_error, force_error)
        max_reaction_error = max(max_reaction_error, reaction_error)
        total_field_query += field_state.contact.query_seconds
        total_ref_query += ref_state.contact.query_seconds
        max_iterations = max(max_iterations, field_state.iterations, ref_state.iterations)
        all_converged = all_converged and field_state.converged and ref_state.converged
        case_id = f"static_linear_{element_type}_native_sfc_trajectory_equivalence"
        history_rows.append(
            {
                "case_id": case_id,
                "step": step,
                "time_or_load": load_factor,
                "prescribed_top_z": prescribed_z,
                "min_gap_field": float(np.min(field_state.contact.gaps)),
                "min_gap_reference": float(np.min(ref_state.contact.gaps)),
                "reaction_z_field": reaction_field,
                "reaction_z_reference": reaction_ref,
                "state_l2_abs_error": state_error,
                "gap_max_abs_error": gap_error,
                "force_l2_rel_error": force_error,
                "active_count_field": field_state.contact.active_count,
                "active_count_reference": ref_state.contact.active_count,
                "field_iterations": field_state.iterations,
                "reference_iterations": ref_state.iterations,
                "field_converged": str(field_state.converged).lower(),
                "reference_converged": str(ref_state.converged).lower(),
            }
        )
        for sample_id, (gap_field, gap_ref) in enumerate(zip(field_state.contact.gaps, ref_state.contact.gaps, strict=True)):
            sample_rows.append(
                {
                    "case_id": case_id,
                    "step": step,
                    "sample_id": sample_id,
                    "gap_field": float(gap_field),
                    "gap_reference": float(gap_ref),
                    "active_field": int(gap_field < 0.0),
                    "active_reference": int(gap_ref < 0.0),
                }
            )

    max_vm, max_strain = _stress_metrics(model, u_field)
    case_row = {
        "case_id": f"static_linear_{element_type}_native_sfc_trajectory_equivalence",
        "regime": "static_incremental",
        "element_type": element_type,
        "geometry": "sfc_native_block_on_rigid_plane",
        "candidate_path": "DynamicNarrowBandSDF+field_contact",
        "reference_path": "SFC analytic plane same penalty law",
        "node_count": model.mesh.X.shape[0],
        "element_count": model.mesh.elements.shape[0],
        "trajectory_steps": n_steps,
        "sample_count": len(samples),
        "field_update_seconds": sdf.field_update_cost,
        "field_query_seconds_total": total_field_query,
        "reference_query_seconds_total": total_ref_query,
        "max_state_l2_abs_error": max_state_error,
        "max_gap_abs_error": max_gap_error,
        "max_force_l2_rel_error": max_force_error,
        "max_reaction_z_abs_error": max_reaction_error,
        "max_solver_iterations": max_iterations,
        "all_increments_converged": str(all_converged).lower(),
        "max_von_mises": max_vm,
        "max_strain_norm": max_strain,
        "status": _case_status(max_state_error, max_gap_error, max_force_error, all_converged),
    }
    return case_row, history_rows, sample_rows


def _dynamic_trajectory(
    element_type: str,
    *,
    quick: bool,
    sdf: DynamicNarrowBandSDF,
    contact_stiffness: float,
) -> tuple[Row, list[Row], list[Row]]:
    model = _native_model(element_type, quick=quick)
    samples = _bottom_node_samples(model.bottom_nodes)
    stiffness = assemble_stiffness_matrix(model.body).tocsr()
    mass = assemble_mass_matrix(model.body, kind="lumped").tocsr()
    mass_diag = np.maximum(mass.diagonal(), 1.0e-12)
    dt = 8.0e-4
    n_steps = 90 if quick else 140
    u_field = np.zeros(model.body.n_dofs, dtype=float)
    v_field = np.zeros(model.body.n_dofs, dtype=float)
    u_ref = np.zeros(model.body.n_dofs, dtype=float)
    v_ref = np.zeros(model.body.n_dofs, dtype=float)
    v_field[2::3] = -0.92
    v_ref[2::3] = -0.92

    history_rows: list[Row] = []
    sample_rows: list[Row] = []
    max_state_error = 0.0
    max_velocity_error = 0.0
    max_gap_error = 0.0
    max_force_error = 0.0
    max_reaction_error = 0.0
    total_field_query = 0.0
    total_ref_query = 0.0
    peak_reaction_field = 0.0
    peak_reaction_ref = 0.0
    max_penetration_field = 0.0
    max_penetration_ref = 0.0
    case_id = f"dynamic_linear_{element_type}_native_sfc_trajectory_equivalence"

    for step in range(n_steps):
        x_field = model.mesh.X + u_field.reshape((-1, 3))
        field_contact = _field_contact_state(x_field, samples, sdf, stiffness=contact_stiffness)
        x_ref = model.mesh.X + u_ref.reshape((-1, 3))
        ref_contact = _analytic_plane_contact_state(x_ref, samples, stiffness=contact_stiffness)

        total_field_query += field_contact.query_seconds
        total_ref_query += ref_contact.query_seconds
        gap_error = float(np.max(np.abs(field_contact.gaps - ref_contact.gaps)))
        force_error = _relative_error(field_contact.force, ref_contact.force)
        reaction_field = float(np.sum(field_contact.force[2::3]))
        reaction_ref = float(np.sum(ref_contact.force[2::3]))

        internal_field = stiffness @ u_field
        internal_ref = stiffness @ u_ref
        a_field = (field_contact.force - internal_field) / mass_diag
        a_ref = (ref_contact.force - internal_ref) / mass_diag
        v_field = v_field + dt * a_field
        v_ref = v_ref + dt * a_ref
        u_field = u_field + dt * v_field
        u_ref = u_ref + dt * v_ref

        state_error = float(np.linalg.norm(u_field - u_ref))
        velocity_error = float(np.linalg.norm(v_field - v_ref))
        max_state_error = max(max_state_error, state_error)
        max_velocity_error = max(max_velocity_error, velocity_error)
        max_gap_error = max(max_gap_error, gap_error)
        max_force_error = max(max_force_error, force_error)
        max_reaction_error = max(max_reaction_error, abs(reaction_field - reaction_ref))
        peak_reaction_field = max(peak_reaction_field, reaction_field)
        peak_reaction_ref = max(peak_reaction_ref, reaction_ref)
        max_penetration_field = max(max_penetration_field, max(-float(np.min(field_contact.gaps)), 0.0))
        max_penetration_ref = max(max_penetration_ref, max(-float(np.min(ref_contact.gaps)), 0.0))
        history_rows.append(
            {
                "case_id": case_id,
                "step": step,
                "time_or_load": step * dt,
                "min_gap_field": float(np.min(field_contact.gaps)),
                "min_gap_reference": float(np.min(ref_contact.gaps)),
                "reaction_z_field": reaction_field,
                "reaction_z_reference": reaction_ref,
                "state_l2_abs_error": state_error,
                "velocity_l2_abs_error": velocity_error,
                "gap_max_abs_error": gap_error,
                "force_l2_rel_error": force_error,
                "active_count_field": field_contact.active_count,
                "active_count_reference": ref_contact.active_count,
            }
        )
        if step == n_steps - 1:
            for sample_id, (gap_field, gap_ref) in enumerate(zip(field_contact.gaps, ref_contact.gaps, strict=True)):
                sample_rows.append(
                    {
                        "case_id": case_id,
                        "step": step,
                        "sample_id": sample_id,
                        "gap_field": float(gap_field),
                        "gap_reference": float(gap_ref),
                        "active_field": int(gap_field < 0.0),
                        "active_reference": int(gap_ref < 0.0),
                    }
                )

    max_vm, max_strain = _stress_metrics(model, u_field)
    case_row = {
        "case_id": case_id,
        "regime": "dynamic_explicit",
        "element_type": element_type,
        "geometry": "sfc_native_block_on_rigid_plane",
        "candidate_path": "DynamicNarrowBandSDF+field_contact",
        "reference_path": "SFC analytic plane same penalty law",
        "node_count": model.mesh.X.shape[0],
        "element_count": model.mesh.elements.shape[0],
        "trajectory_steps": n_steps,
        "sample_count": len(samples),
        "field_update_seconds": sdf.field_update_cost,
        "field_query_seconds_total": total_field_query,
        "reference_query_seconds_total": total_ref_query,
        "max_state_l2_abs_error": max_state_error,
        "max_velocity_l2_abs_error": max_velocity_error,
        "max_gap_abs_error": max_gap_error,
        "max_force_l2_rel_error": max_force_error,
        "max_reaction_z_abs_error": max_reaction_error,
        "max_penetration_field": max_penetration_field,
        "max_penetration_reference": max_penetration_ref,
        "peak_reaction_z_field": peak_reaction_field,
        "peak_reaction_z_reference": peak_reaction_ref,
        "max_von_mises": max_vm,
        "max_strain_norm": max_strain,
        "status": _case_status(max_state_error, max_gap_error, max_force_error),
    }
    return case_row, history_rows, sample_rows


def _write_summary(path: Path, case_rows: list[Row], outputs: dict[str, Path]) -> None:
    passed = sum(1 for row in case_rows if row["status"] == "passed")
    lines = [
        "# SFC Native Full-Trajectory Equivalence",
        "",
        "This validation is fully native to SFC: geometry construction, FEM assembly, contact evaluation, and trajectory integration/solve are all performed locally.",
        "",
        "## Definition",
        "",
        "- Candidate path: `DynamicNarrowBandSDF + field_contact`.",
        "- Reference path: SFC analytic rigid-plane gap/normal with the same frictionless penalty law.",
        "- Both paths use the same SFC mesh, material, stiffness, mass, load history, and integration/solve loop.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        if key != "summary":
            lines.append(f"- `{value.relative_to(path.parent).as_posix()}`")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Case | Status | State error | Gap error | Force error | Steps |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in case_rows:
        lines.append(
            f"| `{row['case_id']}` | {row['status']} | "
            f"{float(row['max_state_l2_abs_error']):.6e} | "
            f"{float(row['max_gap_abs_error']):.6e} | "
            f"{float(row['max_force_l2_rel_error']):.6e} | "
            f"{int(row['trajectory_steps'])} |"
        )
    lines.extend(
        [
            "",
            "## Claim Gate",
            "",
            f"- Cases passed: `{passed}/{len(case_rows)}`.",
            f"- Candidate path used in every case: `{str(all(row['candidate_path'] == 'DynamicNarrowBandSDF+field_contact' for row in case_rows)).lower()}`.",
            "- Projection is not a contact query API in this runner; it appears only inside SDF field construction.",
            "",
            "## Scope",
            "",
            "- Supports scoped full-trajectory equivalence of true field contact against an exact SFC analytic contact path.",
            "- Does not claim equivalence to external solver, friction, self-contact, barrier contact, or nonlinear FEM trajectories.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(out_dir: Path, *, quick: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf = _master_plane_sdf()
    contact_stiffness = 55.0
    case_rows: list[Row] = []
    history_rows: list[Row] = []
    sample_rows: list[Row] = []
    for element_type in ("tet4", "hex8"):
        row, history, samples = _static_trajectory(element_type, quick=quick, sdf=sdf, contact_stiffness=contact_stiffness)
        case_rows.append(row)
        history_rows.extend(history)
        sample_rows.extend(samples)
        row, history, samples = _dynamic_trajectory(element_type, quick=quick, sdf=sdf, contact_stiffness=contact_stiffness)
        case_rows.append(row)
        history_rows.extend(history)
        sample_rows.extend(samples)

    outputs = {
        "cases": out_dir / "sfc_native_trajectory_equivalence_cases.csv",
        "history": out_dir / "sfc_native_trajectory_equivalence_history.csv",
        "samples": out_dir / "sfc_native_trajectory_equivalence_samples.csv",
        "summary": out_dir / "sfc_native_trajectory_equivalence_summary.md",
    }
    _write_csv(outputs["cases"], case_rows)
    _write_csv(outputs["history"], history_rows)
    _write_csv(outputs["samples"], sample_rows)
    _write_summary(outputs["summary"], case_rows, outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "sfc_native_trajectory_equivalence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(args.out_dir, quick=bool(args.quick))
    print("SFC native full-trajectory equivalence complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
