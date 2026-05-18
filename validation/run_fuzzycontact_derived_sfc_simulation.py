"""Run SFC simulations from FuzzyContact-derived working conditions.

The public FuzzyContact dataset provides VTU solution fields rather than the
original FEniCS input decks.  This runner reconstructs an auditable SFC working
condition from those fields:

* reference coordinates: ``X0 = x_current - displacement``;
* linear elastic material: least-squares fit from stress/strain tensors;
* support and driver faces: inferred from paired-body contact geometry;
* driver displacement: mean final displacement on the inferred driver face;
* contact: SFC ``DynamicNarrowBandSDF + field_contact``.

This is a FuzzyContact-derived SFC reproduction, not an official source-input
equivalence claim.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.sparse import block_diag
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact import SurfaceSample, compute_field_contact_constraints, field_penalty_contact_response  # noqa: E402
from sfc.fem import DeformableBody, assemble_stiffness_matrix  # noqa: E402
from sfc.fem.constraints import eliminate_fixed_dofs, expand_reduced_vector, fixed_dofs_from_node_set  # noqa: E402
from sfc.fem.hex8 import hex8_center_strain_stress  # noqa: E402
from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.fem.tet4 import tet4_strain_displacement_matrix  # noqa: E402
from sfc.mesh import VolumeMesh, extract_boundary_triangles  # noqa: E402
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402
from validation.run_fuzzycontact_vtu_reference import _real_vtu_files, _replay_pairs  # noqa: E402
from validation.run_fuzzycontact_first_frame_sfc_models import (  # noqa: E402
    FirstFrameMaterial,
    _candidate_reference_points,
    _default_reference_rule,
    _reference_candidate_score,
    _strain_tensor_norm_from_voigt,
)

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class BodyCase:
    name: str
    mesh: VolumeMesh
    reference_rule: str
    reference_score: float
    reference_score_details: str
    target_displacement: np.ndarray
    E: float
    nu: float
    stress_scale_label: str
    target_max_von_mises: float


@dataclass(frozen=True, slots=True)
class DerivedCase:
    case_id: str
    problem: str
    scenario: str
    master: BodyCase
    slave: BodyCase
    axis: int
    direction: float


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


def _point_field(point_data: dict[str, np.ndarray], kind: str) -> tuple[str, np.ndarray]:
    aliases = {
        "displacement": ("displacement", "u_"),
        "strain": ("strain", "e_"),
        "stress": ("stress", "s_"),
    }[kind]
    for name, values in point_data.items():
        lower = name.lower()
        if any(token in lower for token in aliases):
            return name, np.asarray(values, dtype=float)
    raise ValueError(f"missing {kind} field")


def _cells_element_type(cell_types: list[int]) -> str:
    unique = {int(v) for v in cell_types}
    if unique == {10}:
        return "tet4"
    if unique == {12}:
        return "hex8"
    raise ValueError(f"unsupported VTU cell types: {sorted(unique)}")


def _fit_isotropic_material(strain: np.ndarray, stress: np.ndarray, *, max_points: int = 5000) -> tuple[float, float]:
    eps = np.asarray(strain, dtype=float).reshape((-1, 3, 3))
    sig = np.asarray(stress, dtype=float).reshape((-1, 3, 3))
    if eps.shape[0] > max_points:
        ids = np.linspace(0, eps.shape[0] - 1, max_points, dtype=int)
        eps = eps[ids]
        sig = sig[ids]
    A: list[list[float]] = []
    b: list[float] = []
    for e, s in zip(eps, sig, strict=True):
        tr = float(np.trace(e))
        for i in range(3):
            A.append([tr, 2.0 * float(e[i, i])])
            b.append(float(s[i, i]))
        for i, j in ((0, 1), (0, 2), (1, 2)):
            A.append([0.0, 2.0 * float(0.5 * (e[i, j] + e[j, i]))])
            b.append(float(0.5 * (s[i, j] + s[j, i])))
    sol, *_ = np.linalg.lstsq(np.asarray(A, dtype=float), np.asarray(b, dtype=float), rcond=None)
    lam = float(sol[0])
    mu = float(sol[1])
    if not math.isfinite(lam) or not math.isfinite(mu) or mu <= 0.0 or lam + mu <= 0.0:
        return 1.0, 0.30
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    if not math.isfinite(E) or E <= 0.0 or not (-0.99 < nu < 0.49):
        return max(mu, 1.0), 0.30
    return float(E), float(np.clip(nu, -0.95, 0.48))


def _select_reference_mesh(data: Any, problem: str, displacement: np.ndarray, E: float, nu: float) -> tuple[str, VolumeMesh, float, str]:
    material = FirstFrameMaterial(E=E, nu=nu, density=1.0, unit_system="dataset")
    default = _default_reference_rule(problem) if problem in {"problem_1", "problem_3", "problem_4"} else "points_minus_displacement"
    scored: list[tuple[float, str, VolumeMesh]] = []
    for rule, X in _candidate_reference_points(np.asarray(data.points, dtype=float), displacement).items():
        try:
            mesh = VolumeMesh(X, np.asarray(data.cells, dtype=np.int64), element_type=_cells_element_type(data.cell_types))
            score = _reference_candidate_score(mesh, displacement, material, data)
        except Exception:
            score = math.inf
            mesh = VolumeMesh(X, np.asarray(data.cells, dtype=np.int64), element_type=_cells_element_type(data.cell_types))
        scored.append((score, rule, mesh))
    scored.sort(key=lambda item: (item[0], 0 if item[1] == default else 1))
    best_score, best_rule, best_mesh = scored[0]
    for score, rule, mesh in scored:
        if rule == default and math.isfinite(score) and score <= best_score + 1.0e-3:
            best_score, best_rule, best_mesh = score, rule, mesh
            break
    details = ";".join(f"{rule}:{score:.6e}" for score, rule, _mesh in scored)
    return best_rule, best_mesh, float(best_score), details


def _body_case(name: str, data: Any, problem: str) -> BodyCase:
    disp_name, displacement = _point_field(data.point_data, "displacement")
    _strain_name, strain = _point_field(data.point_data, "strain")
    stress_name, stress = _point_field(data.point_data, "stress")
    E, nu = _fit_isotropic_material(strain, stress)
    rule, mesh, score, details = _select_reference_mesh(data, problem, np.asarray(displacement, dtype=float), E, nu)
    return BodyCase(
        name=name,
        mesh=mesh,
        reference_rule=rule,
        reference_score=score,
        reference_score_details=details,
        target_displacement=np.asarray(displacement, dtype=float),
        E=E,
        nu=nu,
        stress_scale_label=stress_name,
        target_max_von_mises=_max_tensor_von_mises(stress),
    )


def _bbox_contact_axis(master_points: np.ndarray, slave_points: np.ndarray) -> tuple[int, float]:
    master_min = np.min(master_points, axis=0)
    master_max = np.max(master_points, axis=0)
    slave_min = np.min(slave_points, axis=0)
    slave_max = np.max(slave_points, axis=0)
    separated: list[tuple[float, int, float]] = []
    for axis in range(3):
        if master_max[axis] <= slave_min[axis]:
            separated.append((float(slave_min[axis] - master_max[axis]), axis, 1.0))
        elif slave_max[axis] <= master_min[axis]:
            separated.append((float(master_min[axis] - slave_max[axis]), axis, -1.0))
    if separated:
        _gap, axis, direction = min(separated, key=lambda item: item[0])
        return axis, direction
    centers = np.mean(slave_points, axis=0) - np.mean(master_points, axis=0)
    axis = int(np.argmax(np.abs(centers)))
    return axis, 1.0 if centers[axis] >= 0.0 else -1.0


def _extreme_nodes(points: np.ndarray, *, axis: int, side: str, quantile: float = 0.08) -> np.ndarray:
    values = np.asarray(points, dtype=float)[:, axis]
    if side == "min":
        threshold = float(np.quantile(values, quantile))
        ids = np.nonzero(values <= threshold + 1.0e-12)[0]
    else:
        threshold = float(np.quantile(values, 1.0 - quantile))
        ids = np.nonzero(values >= threshold - 1.0e-12)[0]
    if ids.size == 0:
        ids = np.asarray([int(np.argmin(values) if side == "min" else np.argmax(values))], dtype=np.int64)
    return ids.astype(np.int64)


def _select_even(ids: np.ndarray, max_count: int) -> np.ndarray:
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size <= max_count:
        return arr
    pick = np.unique(np.linspace(0, arr.size - 1, int(max_count), dtype=int))
    return arr[pick]


def _contact_triangles(mesh: VolumeMesh, x_current: np.ndarray, *, axis: int, side: str) -> np.ndarray:
    triangles, _adjacent = extract_boundary_triangles(mesh.elements, x_current, element_type=mesh.element_type)
    if triangles.size == 0:
        return triangles
    centroids = np.mean(x_current[triangles], axis=1)
    values = centroids[:, axis]
    if side == "min":
        threshold = float(np.quantile(values, 0.16))
        selected = triangles[values <= threshold + 1.0e-12]
    else:
        threshold = float(np.quantile(values, 0.84))
        selected = triangles[values >= threshold - 1.0e-12]
    return selected if selected.size else triangles


def _load_cases(source_dir: Path, *, max_cases: int) -> list[DerivedCase]:
    pairs = _replay_pairs(_real_vtu_files(source_dir, 10_000), max_groups=max_cases)
    cases: list[DerivedCase] = []
    for pair in pairs:
        master = _body_case(pair.master_body, pair.master, pair.problem)
        slave = _body_case(pair.slave_body, pair.slave, pair.problem)
        axis, direction = _bbox_contact_axis(master.mesh.X, slave.mesh.X)
        cases.append(
            DerivedCase(
                case_id=f"{pair.problem}_{pair.scenario}_{pair.master_body}_to_{pair.slave_body}",
                problem=pair.problem,
                scenario=pair.scenario,
                master=master,
                slave=slave,
                axis=axis,
                direction=direction,
            )
        )
    return cases


def _block_stiffness(master: BodyCase, slave: BodyCase):
    master_body = DeformableBody(master.mesh, {"E": master.E, "nu": master.nu}, density=1.0)
    slave_body = DeformableBody(slave.mesh, {"E": slave.E, "nu": slave.nu}, density=1.0)
    return block_diag((assemble_stiffness_matrix(master_body), assemble_stiffness_matrix(slave_body)), format="csr")


def _mean_disp_norm(displacement: np.ndarray, ids: np.ndarray) -> float:
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(np.asarray(displacement, dtype=float)[arr], axis=1)))


def _select_support_nodes(body: BodyCase, *, axis: int) -> tuple[np.ndarray, str]:
    disp_norm = np.linalg.norm(body.target_displacement, axis=1)
    candidates: list[tuple[float, str, np.ndarray]] = []
    for side in ("min", "max"):
        ids = _extreme_nodes(body.mesh.X, axis=axis, side=side)
        score = float(np.mean(disp_norm[ids])) if ids.size else math.inf
        candidates.append((score, f"{side}_low_displacement_face", ids))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][2], candidates[0][1]


def _top_fraction(ids: np.ndarray, scores: np.ndarray, max_count: int) -> np.ndarray:
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size == 0:
        return arr
    order = np.argsort(-np.asarray(scores, dtype=float)[arr])
    count = min(max(1, int(max_count)), arr.size)
    return arr[order[:count]]


def _select_driver_nodes(body: BodyCase, *, axis: int, direction: float) -> tuple[np.ndarray, str]:
    points = body.mesh.X
    displacement = body.target_displacement
    norm = np.linalg.norm(displacement, axis=1)
    outer_side = "max" if direction >= 0.0 else "min"
    contact_side = "min" if direction >= 0.0 else "max"
    face_outer = _extreme_nodes(points, axis=axis, side=outer_side)
    face_contact = _extreme_nodes(points, axis=axis, side=contact_side)
    high_norm = _top_fraction(np.arange(points.shape[0], dtype=np.int64), norm, max(1, points.shape[0] // 12))
    high_axis = _top_fraction(np.arange(points.shape[0], dtype=np.int64), np.abs(displacement[:, axis]), max(1, points.shape[0] // 12))
    candidates = [
        ("outer_extreme_face", face_outer),
        ("contact_extreme_face", face_contact),
        ("largest_displacement_norm", high_norm),
        ("largest_axis_displacement", high_axis),
    ]
    scored: list[tuple[float, str, np.ndarray]] = []
    for label, ids in candidates:
        if ids.size == 0:
            continue
        score = _mean_disp_norm(displacement, ids)
        scored.append((score, label, ids))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][2], scored[0][1]


def _fixed_dofs(case: DerivedCase, load_factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    axis = case.axis
    master_support, support_strategy = _select_support_nodes(case.master, axis=axis)
    slave_driver, driver_strategy = _select_driver_nodes(case.slave, axis=axis, direction=case.direction)

    master_fixed = fixed_dofs_from_node_set(master_support, "xyz")
    slave_fixed_local = fixed_dofs_from_node_set(slave_driver, "xyz")
    offset = case.master.mesh.X.shape[0] * 3
    fixed = np.concatenate((master_fixed, offset + slave_fixed_local))
    values = np.zeros(fixed.size, dtype=float)
    driver_targets = case.slave.target_displacement[slave_driver] * float(load_factor)
    driver_target = np.mean(driver_targets, axis=0) if driver_targets.size else np.zeros(3, dtype=float)
    for local_id, dof in enumerate(slave_fixed_local):
        node_rank = int(local_id // 3)
        component = int(dof) % 3
        values[master_fixed.size + local_id] = driver_targets[node_rank, component]
    return fixed, values, master_support, slave_driver, driver_target, support_strategy, driver_strategy


def _contact_state(
    case: DerivedCase,
    u: np.ndarray,
    *,
    contact_stiffness: float,
    max_contact_samples: int,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    n_master = case.master.mesh.X.shape[0]
    master_dofs = n_master * 3
    master_x = case.master.mesh.X + u[:master_dofs].reshape((-1, 3))
    slave_x = case.slave.mesh.X + u[master_dofs:].reshape((-1, 3))
    slave_contact_side = "min" if case.direction >= 0.0 else "max"
    master_contact_side = "max" if case.direction >= 0.0 else "min"
    contact_nodes = _select_even(_extreme_nodes(slave_x, axis=case.axis, side=slave_contact_side), max_contact_samples)
    samples = [
        SurfaceSample(np.asarray([int(node)], dtype=np.int64), np.asarray([1.0], dtype=float), np.asarray([0], dtype=np.int64))
        for node in contact_nodes
    ]
    sample_points = slave_x[contact_nodes]
    triangles = _contact_triangles(case.master.mesh, master_x, axis=case.axis, side=master_contact_side)
    span = np.ptp(master_x, axis=0)
    spacing = max(float(np.linalg.norm(span)) / 46.0, 1.0e-8)
    separated_gap = max(0.0, float(np.min(sample_points[:, case.axis]) - np.max(master_x[:, case.axis]))) if case.direction >= 0 else max(0.0, float(np.min(master_x[:, case.axis]) - np.max(sample_points[:, case.axis])))
    band = max(8.0 * spacing, separated_gap + 4.0 * spacing)
    start = perf_counter()
    sdf = DynamicNarrowBandSDF.build_required_points(
        master_x,
        triangles,
        sample_points,
        spacing=spacing,
        band_radius=band,
        padding=band,
        cell_size=max(spacing, band),
    )
    constraints = compute_field_contact_constraints(slave_x, samples, sdf)
    force, _Kc = field_penalty_contact_response(
        constraints,
        stiffness=contact_stiffness,
        n_total_dofs=u.size,
        slave_dof_offset=master_dofs,
        master_dof_offset=0,
    )
    elapsed = perf_counter() - start
    gaps = np.asarray([c.g for c in constraints], dtype=float)
    return gaps, force, int(np.count_nonzero(gaps < 0.0)), elapsed, float(sdf.stats.valid_node_count)


def _simulate_case(
    case: DerivedCase,
    *,
    quick: bool,
    max_contact_samples: int,
    load_steps: int,
    iterations: int,
) -> tuple[Row, list[Row]]:
    K = _block_stiffness(case.master, case.slave)
    n_dofs = K.shape[0]
    u = np.zeros(n_dofs, dtype=float)
    # Nodal penalty units are unavailable in the public dataset; use the fitted
    # modulus as a scale and keep the same rule for all derived cases.
    contact_stiffness = 0.04 * max(case.master.E, case.slave.E)
    history: list[Row] = []
    total_query = 0.0
    max_correction = 0.0
    max_active = 0
    last_gaps = np.empty(0, dtype=float)
    last_force = np.zeros(n_dofs, dtype=float)
    final_driver = np.zeros(3, dtype=float)
    final_support_strategy = ""
    final_driver_strategy = ""
    final_support_count = 0
    final_driver_count = 0
    for step in range(max(1, int(load_steps))):
        load = (step + 1) / max(1, int(load_steps))
        fixed, values, master_support, slave_driver, driver, support_strategy, driver_strategy = _fixed_dofs(case, load)
        final_driver = driver
        final_support_strategy = support_strategy
        final_driver_strategy = driver_strategy
        final_support_count = int(master_support.size)
        final_driver_count = int(slave_driver.size)
        for iteration in range(max(1, int(iterations))):
            gaps, contact_force, active, query_seconds, valid_nodes = _contact_state(
                case,
                u,
                contact_stiffness=contact_stiffness,
                max_contact_samples=max_contact_samples,
            )
            total_query += query_seconds
            A, rhs, free = eliminate_fixed_dofs(K, contact_force, fixed, values)
            reduced = np.asarray(spsolve(A, rhs), dtype=float) if free.size else np.empty(0, dtype=float)
            u_new = expand_reduced_vector(reduced, free, n_dofs, fixed, values)
            correction = float(np.linalg.norm(u_new - u))
            max_correction = max(max_correction, correction)
            u = 0.55 * u_new + 0.45 * u
            last_gaps = gaps
            last_force = contact_force
            max_active = max(max_active, active)
            history.append(
                {
                    "case_id": case.case_id,
                    "load_step": step,
                    "iteration": iteration,
                    "load_factor": load,
                    "active_contact_samples": active,
                    "min_gap": float(np.min(gaps)) if gaps.size else "",
                    "mean_gap": float(np.mean(gaps)) if gaps.size else "",
                    "reaction_norm": float(np.linalg.norm(contact_force)),
                    "correction_norm": correction,
                    "field_query_build_seconds": query_seconds,
                    "valid_grid_nodes": valid_nodes,
                }
            )
            if correction <= 1.0e-9 * max(1.0, float(np.linalg.norm(u_new))):
                u = u_new
                break
    master_n = case.master.mesh.X.shape[0]
    master_u = u[: master_n * 3].reshape((-1, 3))
    slave_u = u[master_n * 3 :].reshape((-1, 3))
    master_disp_error = _relative_error(master_u, case.master.target_displacement)
    slave_disp_error = _relative_error(slave_u, case.slave.target_displacement)
    master_vm, master_strain = _stress_metrics(case.master, master_u)
    slave_vm, slave_strain = _stress_metrics(case.slave, slave_u)
    target_master_vm = _target_von_mises(case.master)
    target_slave_vm = _target_von_mises(case.slave)
    row = {
        "case_id": case.case_id,
        "problem": case.problem,
        "scenario": case.scenario,
        "axis": case.axis,
        "direction": case.direction,
        "master_nodes": case.master.mesh.X.shape[0],
        "slave_nodes": case.slave.mesh.X.shape[0],
        "master_elements": case.master.mesh.elements.shape[0],
        "slave_elements": case.slave.mesh.elements.shape[0],
        "master_reference_rule": case.master.reference_rule,
        "slave_reference_rule": case.slave.reference_rule,
        "master_reference_score": case.master.reference_score,
        "slave_reference_score": case.slave.reference_score,
        "master_reference_score_details": case.master.reference_score_details,
        "slave_reference_score_details": case.slave.reference_score_details,
        "master_E_fit": case.master.E,
        "master_nu_fit": case.master.nu,
        "slave_E_fit": case.slave.E,
        "slave_nu_fit": case.slave.nu,
        "contact_stiffness": contact_stiffness,
        "driver_displacement_x": float(final_driver[0]),
        "driver_displacement_y": float(final_driver[1]),
        "driver_displacement_z": float(final_driver[2]),
        "support_selection": final_support_strategy,
        "driver_selection": final_driver_strategy,
        "support_node_count": final_support_count,
        "driver_node_count": final_driver_count,
        "driver_mean_displacement_norm": float(np.linalg.norm(final_driver)),
        "loading_history_status": "not_available_in_public_vtu; final displacement-control reconstruction only",
        "max_contact_samples": max_contact_samples,
        "load_steps": load_steps,
        "iterations_per_step": iterations,
        "active_contact_samples_max": max_active,
        "final_min_gap": float(np.min(last_gaps)) if last_gaps.size else "",
        "final_mean_gap": float(np.mean(last_gaps)) if last_gaps.size else "",
        "final_reaction_norm": float(np.linalg.norm(last_force)),
        "field_query_build_seconds_total": total_query,
        "max_correction_norm": max_correction,
        "master_displacement_rel_error_vs_fuzzycontact": master_disp_error,
        "slave_displacement_rel_error_vs_fuzzycontact": slave_disp_error,
        "master_sfc_max_von_mises": master_vm,
        "slave_sfc_max_von_mises": slave_vm,
        "master_fuzzycontact_max_von_mises": target_master_vm,
        "slave_fuzzycontact_max_von_mises": target_slave_vm,
        "master_sfc_max_strain_norm": master_strain,
        "slave_sfc_max_strain_norm": slave_strain,
        "trajectory_equivalence_status": "not_supported",
        "status": "completed",
        "claim_scope": "fuzzycontact-derived SFC simulation; inferred BC/material from public VTU fields",
    }
    return row, history


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    if denom <= 1.0e-30:
        return float(np.linalg.norm(value - reference))
    return float(np.linalg.norm(value - reference) / denom)


def _target_von_mises(body: BodyCase) -> float:
    return float(body.target_max_von_mises)


def _max_tensor_von_mises(stress: np.ndarray) -> float:
    tensors = np.asarray(stress, dtype=float).reshape((-1, 3, 3))
    values = []
    for s in tensors:
        voigt = np.asarray([s[0, 0], s[1, 1], s[2, 2], 0.5 * (s[0, 1] + s[1, 0]), 0.5 * (s[1, 2] + s[2, 1]), 0.5 * (s[0, 2] + s[2, 0])])
        values.append(_von_mises_voigt(voigt))
    return float(np.max(values)) if values else math.nan


def _von_mises_voigt(stress: np.ndarray) -> float:
    s = np.asarray(stress, dtype=float).ravel()
    sxx, syy, szz, sxy, syz, sxz = s
    return float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + syz**2 + sxz**2)))


def _stress_metrics(body: BodyCase, u_nodes: np.ndarray) -> tuple[float, float]:
    values: list[float] = []
    strain_norms: list[float] = []
    if body.mesh.element_type == "tet4":
        C = isotropic_linear_elasticity_matrix(body.E, body.nu)
        for element in body.mesh.elements:
            B = tet4_strain_displacement_matrix(body.mesh.X[element])
            strain = B @ u_nodes[element].reshape(12)
            stress = C @ strain
            values.append(_von_mises_voigt(stress))
            strain_norms.append(_strain_tensor_norm_from_voigt(strain))
    else:
        for element in body.mesh.elements:
            strain, stress = hex8_center_strain_stress(body.mesh.X[element], u_nodes[element], body.E, body.nu)
            values.append(_von_mises_voigt(stress))
            strain_norms.append(float(np.linalg.norm(strain)))
    return float(np.max(values)) if values else 0.0, float(np.max(strain_norms)) if strain_norms else 0.0


def _write_summary(path: Path, case_rows: list[Row], outputs: dict[str, Path]) -> None:
    lines = [
        "# FuzzyContact-Derived SFC Simulation",
        "",
        "This runner reconstructs SFC working conditions from the public FuzzyContact VTU solution fields.",
        "It does not use the original FEniCS decks because they are not included in the downloaded dataset.",
        "",
        "## Reconstruction",
        "",
        "- Reference mesh: selected by scoring `points` and `points_minus_displacement` against the public VTU strain/stress fields.",
        "- Material: isotropic linear fit from reported stress/strain tensors.",
        "- Boundary conditions: support nodes are selected from low-displacement boundary faces; driver nodes are selected from the largest displacement response nodes.",
        "- Driver displacement: nodal final displacement values on the inferred response driver set.",
        "- Loading history: not present in the public VTU files; this runner performs only final-state displacement-control reconstruction.",
        "- Contact: SFC `DynamicNarrowBandSDF + field_contact`.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        if key != "summary":
            lines.append(f"- `{value.relative_to(path.parent).as_posix()}`")
    lines.extend(["", "## Cases", "", "| Case | Status | Trajectory equivalence | Active samples | Final min gap | Slave disp. rel. error |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for row in case_rows:
        lines.append(
            f"| `{row['case_id']}` | {row['status']} | {row['trajectory_equivalence_status']} | {row['active_contact_samples_max']} | "
            f"{float(row['final_min_gap']):.6e} | {float(row['slave_displacement_rel_error_vs_fuzzycontact']):.6e} |"
        )
    lines.extend(
        [
            "",
            "## Reconstruction Diagnostics",
            "",
            "- Coordinate convention, support selection, driver selection, driver displacement norm, and loading-history status are written to `fuzzycontact_derived_sfc_cases.csv`.",
            "- Large displacement errors are interpreted as missing working-condition evidence, not as a dynamic-SDF field-query failure.",
        ]
    )
    lines.extend(
        [
            "",
            "## Claim Scope",
            "",
            "- Supports an auditable reconstruction attempt from FuzzyContact-derived geometry and final displacement fields.",
            "- Does not support complete displacement/stress/strain trajectory equivalence; rows explicitly set `trajectory_equivalence_status=not_supported`.",
            "- Does not support official FuzzyContact solver equivalence because original material cards, penalty-law parameters, boundary conditions, and load histories are not present in the public zip.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    out_dir: Path,
    *,
    source_dir: Path,
    quick: bool = False,
    max_cases: int = 5,
    max_contact_samples: int = 48,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(source_dir, max_cases=max_cases)
    case_rows: list[Row] = []
    history_rows: list[Row] = []
    load_steps = 2 if quick else 4
    iterations = 2 if quick else 5
    samples = min(max_contact_samples, 24 if quick else max_contact_samples)
    for case in cases:
        row, history = _simulate_case(
            case,
            quick=quick,
            max_contact_samples=samples,
            load_steps=load_steps,
            iterations=iterations,
        )
        case_rows.append(row)
        history_rows.extend(history)
    outputs = {
        "cases": out_dir / "fuzzycontact_derived_sfc_cases.csv",
        "history": out_dir / "fuzzycontact_derived_sfc_history.csv",
        "summary": out_dir / "fuzzycontact_derived_sfc_summary.md",
    }
    _write_csv(outputs["cases"], case_rows)
    _write_csv(outputs["history"], history_rows)
    _write_summary(outputs["summary"], case_rows, outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "results" / "fuzzycontact_vtu_reference_full" / "downloads" / "Dataset_for_Solutions")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "fuzzycontact_derived_sfc")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--max-contact-samples", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(
        args.out_dir,
        source_dir=args.source_dir,
        quick=bool(args.quick),
        max_cases=int(args.max_cases),
        max_contact_samples=int(args.max_contact_samples),
    )
    print("FuzzyContact-derived SFC simulation complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
