"""Build SFC first-frame models from FuzzyContact VTU benchmark files.

The public FuzzyContact dataset provides solution VTU files, not the original
input decks.  This runner uses the first available displacement VTU for
benchmarks 1, 3, and 4 as a mesh/model entry point, reconstructs a reference
configuration where needed, and evaluates SFC's true dynamic narrow-band SDF
field-contact path on the corresponding deformed configurations.

This is an external benchmark mesh reproduction.  It does not claim source-level
FuzzyContact equivalence, and it does not introduce friction, self-contact,
barrier contact, GPU, POD, neural SDF, Abaqus, or CalculiX dependencies.
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from sfc.contact import SurfaceSample, compute_field_contact_constraints, field_penalty_contact_response  # noqa: E402
from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.fem.tet4 import tet4_strain_displacement_matrix  # noqa: E402
from sfc.mesh import VolumeMesh, extract_boundary_triangles  # noqa: E402
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402
from validation.run_fuzzycontact_vtu_reference import (  # noqa: E402
    ReplayPair,
    VTUData,
    _real_vtu_files,
    _replay_pairs,
)

Row = dict[str, Any]

PROBLEM_ORDER = ("problem_1", "problem_3", "problem_4")


@dataclass(frozen=True, slots=True)
class FirstFrameMaterial:
    E: float
    nu: float
    density: float
    unit_system: str


@dataclass(frozen=True, slots=True)
class FirstFrameBody:
    name: str
    mesh: VolumeMesh
    reference_rule: str
    reference_score: float
    reference_score_details: str
    target_displacement: np.ndarray
    reference_points_raw: np.ndarray
    material: FirstFrameMaterial
    vtu: VTUData


@dataclass(frozen=True, slots=True)
class FirstFrameCase:
    case_id: str
    problem: str
    master: FirstFrameBody
    slave: FirstFrameBody
    axis: int
    direction: float
    contact_stiffness: float
    reference_metric_name: str
    reference_metric_value: float


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


def _problem_material(problem: str) -> FirstFrameMaterial:
    if problem in {"problem_1", "problem_3"}:
        return FirstFrameMaterial(E=70_000.0, nu=0.33, density=2700.0, unit_system="mm-N-MPa")
    if problem == "problem_4":
        return FirstFrameMaterial(E=195_223.0, nu=0.3698, density=7850.0, unit_system="mm-N-MPa")
    raise ValueError(f"unsupported problem: {problem}")


def _problem_contact_stiffness(problem: str) -> float:
    if problem == "problem_1":
        return 1000.0
    if problem in {"problem_3", "problem_4"}:
        return 250.0
    raise ValueError(f"unsupported problem: {problem}")


def _problem_reference_metric(problem: str) -> tuple[str, float]:
    if problem == "problem_1":
        return "FuzzyContact sphere indentation depth (mm)", 0.03303
    if problem == "problem_3":
        return "FuzzyContact V-indenter indentation depth (mm)", 0.001876
    if problem == "problem_4":
        return "FuzzyContact effective Young's modulus (MPa)", 196_691.0
    raise ValueError(f"unsupported problem: {problem}")


def _displacement_field(data: VTUData) -> tuple[str, np.ndarray]:
    for name, values in data.point_data.items():
        if "disp" in name.lower():
            arr = np.asarray(values, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"{data.path} displacement field must have 3 components")
            return name, arr
    raise ValueError(f"{data.path} does not contain a displacement field")


def _default_reference_rule(problem: str) -> str:
    if problem == "problem_1":
        return "points_minus_displacement"
    if problem in {"problem_3", "problem_4"}:
        return "points"
    raise ValueError(f"unsupported problem: {problem}")


def _candidate_reference_points(points: np.ndarray, displacement: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "points": np.asarray(points, dtype=float).copy(),
        "points_minus_displacement": np.asarray(points, dtype=float) - np.asarray(displacement, dtype=float),
    }


def _strain_tensor_norm_from_voigt(strain: np.ndarray) -> float:
    """Return Frobenius norm of the symmetric strain tensor from engineering strain."""

    e = np.asarray(strain, dtype=float).ravel()
    return float(np.sqrt(e[0] ** 2 + e[1] ** 2 + e[2] ** 2 + 2.0 * ((0.5 * e[3]) ** 2 + (0.5 * e[4]) ** 2 + (0.5 * e[5]) ** 2)))


def _strain_norm_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.abs(arr.reshape((-1,)))
    if arr.shape[-1] == 9:
        tensors = arr.reshape((-1, 3, 3))
        sym = 0.5 * (tensors + np.swapaxes(tensors, 1, 2))
        return np.linalg.norm(sym, axis=(1, 2))
    if arr.shape[-1] == 6:
        e = arr.reshape((-1, 6))
        return np.sqrt(e[:, 0] ** 2 + e[:, 1] ** 2 + e[:, 2] ** 2 + 2.0 * (e[:, 3] ** 2 + e[:, 4] ** 2 + e[:, 5] ** 2))
    return np.linalg.norm(arr.reshape((arr.shape[0], -1)), axis=1)


def _summary_pair(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float).reshape((-1,))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.max(arr)), float(np.quantile(arr, 0.95))


def _sfc_stress_metric_values(mesh: VolumeMesh, displacement: np.ndarray, material: FirstFrameMaterial) -> tuple[np.ndarray, np.ndarray]:
    if mesh.element_type != "tet4":
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    C = isotropic_linear_elasticity_matrix(material.E, material.nu)
    vm: list[float] = []
    strain_norm: list[float] = []
    for element in mesh.elements:
        B = tet4_strain_displacement_matrix(mesh.X[element])
        strain = B @ displacement[element].reshape(12)
        stress = C @ strain
        vm.append(_von_mises_voigt(stress))
        strain_norm.append(_strain_tensor_norm_from_voigt(strain))
    return np.asarray(vm, dtype=float), np.asarray(strain_norm, dtype=float)


def _reference_stress_metric_values(data: VTUData) -> tuple[np.ndarray, np.ndarray]:
    strain_field = _tensor_field(data, "strain")
    strain = _strain_norm_values(strain_field) if strain_field is not None else np.zeros(data.points.shape[0], dtype=float)
    mises = _field_norm(data, "mises")
    if not np.any(mises):
        stress = _tensor_field(data, "stress")
        mises = _von_mises_tensor(stress) if stress is not None else np.zeros(data.points.shape[0], dtype=float)
    return np.asarray(mises, dtype=float), np.asarray(strain, dtype=float)


def _log_scale_error(value: float, reference: float) -> float:
    eps = 1.0e-30
    return abs(math.log((abs(float(value)) + eps) / (abs(float(reference)) + eps)))


def _reference_candidate_score(mesh: VolumeMesh, displacement: np.ndarray, material: FirstFrameMaterial, data: VTUData) -> float:
    sfc_vm, sfc_strain = _sfc_stress_metric_values(mesh, displacement, material)
    ref_vm, ref_strain = _reference_stress_metric_values(data)
    _sfc_vm_max, sfc_vm_p95 = _summary_pair(sfc_vm)
    _ref_vm_max, ref_vm_p95 = _summary_pair(ref_vm)
    _sfc_strain_max, sfc_strain_p95 = _summary_pair(sfc_strain)
    _ref_strain_max, ref_strain_p95 = _summary_pair(ref_strain)
    return _log_scale_error(sfc_vm_p95, ref_vm_p95) + _log_scale_error(sfc_strain_p95, ref_strain_p95)


def _reference_points(data: VTUData, problem: str) -> tuple[str, np.ndarray, np.ndarray, float, str]:
    _name, displacement = _displacement_field(data)
    points = np.asarray(data.points, dtype=float)
    material = _problem_material(problem)
    default = _default_reference_rule(problem)
    candidates = _candidate_reference_points(points, displacement)
    scored: list[tuple[float, str, np.ndarray]] = []
    for rule, X in candidates.items():
        try:
            mesh = VolumeMesh(X, np.asarray(data.cells, dtype=np.int64), element_type=_element_type(data.cell_types))
            score = _reference_candidate_score(mesh, displacement, material, data)
        except Exception:
            score = math.inf
        scored.append((score, rule, X))
    scored.sort(key=lambda item: (item[0], 0 if item[1] == default else 1))
    best_score, best_rule, best_points = scored[0]
    # Keep the literature/PDF-derived default if the automatic score is
    # numerically indistinguishable.  This prevents tiny roundoff differences in
    # a nearly linear field from flipping the recorded coordinate convention.
    for score, rule, X in scored:
        if rule == default and math.isfinite(score) and score <= best_score + 1.0e-3:
            best_score, best_rule, best_points = score, rule, X
            break
    details = ";".join(f"{rule}:{score:.6e}" for score, rule, _X in scored)
    return best_rule, best_points, displacement, float(best_score), details


def _element_type(cell_types: list[int]) -> str:
    unique = {int(v) for v in cell_types}
    if unique == {10}:
        return "tet4"
    if unique == {12}:
        return "hex8"
    raise ValueError(f"unsupported VTU cell types: {sorted(unique)}")


def _body_from_vtu(name: str, data: VTUData, problem: str) -> FirstFrameBody:
    rule, X, displacement, score, details = _reference_points(data, problem)
    mesh = VolumeMesh(X, np.asarray(data.cells, dtype=np.int64), element_type=_element_type(data.cell_types))
    return FirstFrameBody(
        name=name,
        mesh=mesh,
        reference_rule=rule,
        reference_score=score,
        reference_score_details=details,
        target_displacement=displacement,
        reference_points_raw=np.asarray(data.points, dtype=float),
        material=_problem_material(problem),
        vtu=data,
    )


def _axis_and_direction(master: FirstFrameBody, slave: FirstFrameBody) -> tuple[int, float]:
    delta = np.mean(slave.mesh.X, axis=0) - np.mean(master.mesh.X, axis=0)
    axis = int(np.argmax(np.abs(delta)))
    direction = 1.0 if delta[axis] >= 0.0 else -1.0
    return axis, direction


def _load_cases(source_dir: Path, problems: tuple[str, ...]) -> list[FirstFrameCase]:
    pairs = _replay_pairs(_real_vtu_files(source_dir, 100_000), max_groups=100)
    by_problem: dict[str, ReplayPair] = {pair.problem: pair for pair in pairs if pair.problem in problems}
    cases: list[FirstFrameCase] = []
    for problem in problems:
        pair = by_problem.get(problem)
        if pair is None:
            continue
        master = _body_from_vtu(pair.master_body, pair.master, problem)
        slave = _body_from_vtu(pair.slave_body, pair.slave, problem)
        axis, direction = _axis_and_direction(master, slave)
        metric_name, metric_value = _problem_reference_metric(problem)
        cases.append(
            FirstFrameCase(
                case_id=f"{problem}_{pair.master_body}_to_{pair.slave_body}",
                problem=problem,
                master=master,
                slave=slave,
                axis=axis,
                direction=direction,
                contact_stiffness=_problem_contact_stiffness(problem),
                reference_metric_name=metric_name,
                reference_metric_value=metric_value,
            )
        )
    return cases


def _boundary_triangles(mesh: VolumeMesh, x_current: np.ndarray) -> np.ndarray:
    triangles, _adjacent = extract_boundary_triangles(mesh.elements, x_current, element_type=mesh.element_type)
    return np.asarray(triangles, dtype=np.int64)


def _select_contact_triangles(
    mesh: VolumeMesh,
    x_current: np.ndarray,
    *,
    axis: int,
    side: str,
    quantile: float,
    max_faces: int,
) -> np.ndarray:
    faces = _boundary_triangles(mesh, x_current)
    if faces.size == 0:
        return faces
    centroids = np.mean(x_current[faces], axis=1)
    values = centroids[:, axis]
    if side == "min":
        order = np.argsort(values)
    else:
        order = np.argsort(-values)
    selected = faces[order[: max(1, min(int(max_faces), faces.shape[0]))]]
    return np.asarray(selected, dtype=np.int64)


def _select_master_contact_triangles(
    mesh: VolumeMesh,
    x_current: np.ndarray,
    *,
    axis: int,
    side: str,
    target_center: np.ndarray,
    max_faces: int,
) -> np.ndarray:
    faces = _boundary_triangles(mesh, x_current)
    if faces.size == 0:
        return faces
    centroids = np.mean(x_current[faces], axis=1)
    axis_values = centroids[:, axis]
    axis_span = max(float(np.max(axis_values) - np.min(axis_values)), 1.0e-12)
    if side == "min":
        threshold = float(np.min(axis_values) + 0.25 * axis_span)
        pool = np.nonzero(axis_values <= threshold + 1.0e-12)[0]
    else:
        threshold = float(np.max(axis_values) - 0.25 * axis_span)
        pool = np.nonzero(axis_values >= threshold - 1.0e-12)[0]
    if pool.size == 0:
        pool = np.arange(faces.shape[0])
    tangent_axes = [idx for idx in range(3) if idx != axis]
    center = np.asarray(target_center, dtype=float)
    tangent_delta = centroids[pool][:, tangent_axes] - center[tangent_axes]
    order = pool[np.argsort(np.linalg.norm(tangent_delta, axis=1))]
    return faces[order[: max(1, min(int(max_faces), order.size))]]


def _select_local_contact_triangles(
    mesh: VolumeMesh,
    x_current: np.ndarray,
    *,
    axis: int,
    side: str,
    target_center: np.ndarray,
    max_faces: int,
) -> np.ndarray:
    faces = _boundary_triangles(mesh, x_current)
    if faces.size == 0:
        return faces
    centroids = np.mean(x_current[faces], axis=1)
    axis_values = centroids[:, axis]
    axis_span = max(float(np.max(axis_values) - np.min(axis_values)), 1.0e-12)
    if side == "min":
        threshold = float(np.min(axis_values) + 0.25 * axis_span)
        pool = np.nonzero(axis_values <= threshold + 1.0e-12)[0]
    else:
        threshold = float(np.max(axis_values) - 0.25 * axis_span)
        pool = np.nonzero(axis_values >= threshold - 1.0e-12)[0]
    if pool.size == 0:
        pool = np.arange(faces.shape[0])
    tangent_axes = [idx for idx in range(3) if idx != axis]
    center = np.asarray(target_center, dtype=float)
    tangent_delta = centroids[pool][:, tangent_axes] - center[tangent_axes]
    order = pool[np.argsort(np.linalg.norm(tangent_delta, axis=1))]
    return faces[order[: max(1, min(int(max_faces), order.size))]]


def _surface_samples(slave_faces: np.ndarray) -> list[SurfaceSample]:
    return [
        SurfaceSample(
            np.asarray(face, dtype=np.int64),
            np.full(3, 1.0 / 3.0, dtype=float),
            np.asarray([0], dtype=np.int64),
        )
        for face in np.asarray(slave_faces, dtype=np.int64)
    ]


def _contact_sides(direction: float) -> tuple[str, str]:
    if direction >= 0.0:
        return "max", "min"
    return "min", "max"


def _estimate_spacing(points: np.ndarray, *, quick: bool, problem: str) -> float:
    span = np.ptp(np.asarray(points, dtype=float), axis=0)
    diag = max(float(np.linalg.norm(span)), 1.0e-8)
    divisor = 20.0 if quick else 34.0
    caps = {
        "problem_1": 0.05,
        "problem_3": 0.08,
        "problem_4": 0.20,
    }
    return max(min(diag / divisor, caps.get(problem, diag / divisor)), 1.0e-7)


def _axis_gap(master_x: np.ndarray, query_points: np.ndarray, *, axis: int, direction: float) -> float:
    if direction >= 0.0:
        return float(np.min(query_points[:, axis]) - np.max(master_x[:, axis]))
    return float(np.min(master_x[:, axis]) - np.max(query_points[:, axis]))


def _evaluate_field_contact(
    case: FirstFrameCase,
    *,
    load_factor: float,
    state_name: str,
    closure_offset: float = 0.0,
    quick: bool,
    max_contact_samples: int,
    max_master_faces: int,
) -> tuple[Row, list[Row]]:
    master_x = case.master.mesh.X + float(load_factor) * case.master.target_displacement
    slave_x = case.slave.mesh.X + float(load_factor) * case.slave.target_displacement
    if closure_offset != 0.0:
        slave_x = slave_x.copy()
        slave_x[:, case.axis] -= case.direction * float(closure_offset)
    master_side, slave_side = _contact_sides(case.direction)
    slave_faces = _select_local_contact_triangles(
        case.slave.mesh,
        slave_x,
        axis=case.axis,
        side=slave_side,
        target_center=np.mean(master_x, axis=0),
        max_faces=max_contact_samples,
    )
    samples = _surface_samples(slave_faces)
    query_points = np.asarray([sample.point(slave_x) for sample in samples], dtype=float)
    master_faces = _select_master_contact_triangles(
        case.master.mesh,
        master_x,
        axis=case.axis,
        side=master_side,
        target_center=np.mean(query_points, axis=0),
        max_faces=max_master_faces,
    )
    spacing = _estimate_spacing(
        master_x[master_faces.ravel()] if master_faces.size else master_x,
        quick=quick,
        problem=case.problem,
    )
    estimated_gap = _axis_gap(master_x, query_points, axis=case.axis, direction=case.direction) if query_points.size else 0.0
    band = max(8.0 * spacing, abs(estimated_gap) + 6.0 * spacing)
    start = perf_counter()
    constraints = []
    sdf: DynamicNarrowBandSDF | None = None
    build_seconds = 0.0
    for scale in (1.0, 2.0, 4.0):
        scaled_band = band * scale
        sdf = DynamicNarrowBandSDF.build_required_points(
            master_x,
            master_faces,
            query_points,
            spacing=spacing,
            band_radius=scaled_band,
            padding=scaled_band,
            cell_size=spacing,
        )
        try:
            constraints = compute_field_contact_constraints(slave_x, samples, sdf)
            band = scaled_band
            break
        except ValueError:
            if scale == 4.0:
                raise
    build_seconds = perf_counter() - start
    if sdf is None:
        raise RuntimeError("SDF field build did not produce a field")
    query_start = perf_counter()
    constraints = compute_field_contact_constraints(slave_x, samples, sdf)
    query_seconds = perf_counter() - query_start
    force, _Kc = field_penalty_contact_response(
        constraints,
        stiffness=case.contact_stiffness,
        n_total_dofs=slave_x.shape[0] * 3 + master_x.shape[0] * 3,
        slave_dof_offset=0,
        master_dof_offset=slave_x.shape[0] * 3,
    )
    gaps = np.asarray([constraint.g for constraint in constraints], dtype=float)
    sample_rows = [
        {
            "case_id": case.case_id,
            "state": state_name,
            "load_factor": load_factor,
            "closure_offset": closure_offset,
            "sample_id": sample_id,
            "x": float(constraint.x_slave[0]),
            "y": float(constraint.x_slave[1]),
            "z": float(constraint.x_slave[2]),
            "phi": float(constraint.g),
            "normal_x": float(constraint.normal[0]),
            "normal_y": float(constraint.normal[1]),
            "normal_z": float(constraint.normal[2]),
            "active": int(constraint.active),
        }
        for sample_id, constraint in enumerate(constraints)
    ]
    row = {
        "case_id": case.case_id,
        "problem": case.problem,
        "state": state_name,
        "load_factor": float(load_factor),
        "closure_offset": float(closure_offset),
        "master_body": case.master.name,
        "slave_body": case.slave.name,
        "master_reference_rule": case.master.reference_rule,
        "slave_reference_rule": case.slave.reference_rule,
        "axis": int(case.axis),
        "direction": float(case.direction),
        "master_nodes": int(case.master.mesh.X.shape[0]),
        "slave_nodes": int(case.slave.mesh.X.shape[0]),
        "master_elements": int(case.master.mesh.elements.shape[0]),
        "slave_elements": int(case.slave.mesh.elements.shape[0]),
        "master_contact_faces": int(master_faces.shape[0]),
        "slave_contact_samples": int(len(samples)),
        "spacing": float(spacing),
        "band_radius": float(band),
        "grid_node_count": int(sdf.stats.grid_node_count),
        "valid_grid_node_count": int(sdf.stats.valid_node_count),
        "field_update_seconds": float(build_seconds),
        "field_query_seconds_total": float(query_seconds),
        "field_query_seconds_per_sample": float(query_seconds / max(len(constraints), 1)),
        "phi_min": float(np.min(gaps)) if gaps.size else math.nan,
        "phi_mean": float(np.mean(gaps)) if gaps.size else math.nan,
        "phi_max": float(np.max(gaps)) if gaps.size else math.nan,
        "active_samples": int(np.count_nonzero(gaps < 0.0)),
        "contact_force_norm": float(np.linalg.norm(force)),
        "contact_force_axis_sum": float(np.sum(force[case.axis::3])),
        "query_path": "DynamicNarrowBandSDF.build_required_points -> field_contact interpolation",
    }
    return row, sample_rows


def _tensor_field(data: VTUData, kind: str) -> np.ndarray | None:
    tokens = {
        "strain": ("strain", "e_"),
        "stress": ("stress", "s_"),
        "mises": ("mises", "von"),
    }[kind]
    for name, values in data.point_data.items():
        lower = name.lower()
        if any(token in lower for token in tokens):
            return np.asarray(values, dtype=float)
    return None


def _von_mises_voigt(stress: np.ndarray) -> float:
    sxx, syy, szz, sxy, syz, sxz = np.asarray(stress, dtype=float).ravel()
    return float(
        np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy**2 + syz**2 + sxz**2)
        )
    )


def _von_mises_tensor(tensor: np.ndarray) -> np.ndarray:
    arr = np.asarray(tensor, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.shape[-1] == 1:
        return arr.reshape(-1)
    if arr.shape[-1] == 6:
        return np.asarray([_von_mises_voigt(row) for row in arr.reshape((-1, 6))], dtype=float)
    tensors = arr.reshape((-1, 3, 3))
    values = []
    for s in tensors:
        sym = 0.5 * (s + s.T)
        values.append(_von_mises_voigt([sym[0, 0], sym[1, 1], sym[2, 2], sym[0, 1], sym[1, 2], sym[0, 2]]))
    return np.asarray(values, dtype=float)


def _field_norm(data: VTUData, kind: str) -> np.ndarray:
    values = _tensor_field(data, kind)
    if values is None:
        return np.zeros(data.points.shape[0], dtype=float)
    arr = np.asarray(values, dtype=float)
    if kind == "mises":
        return _von_mises_tensor(arr)
    if kind == "strain":
        return _strain_norm_values(arr)
    return np.linalg.norm(arr.reshape((arr.shape[0], -1)), axis=1)


def _sfc_stress_metrics(body: FirstFrameBody) -> tuple[float, float]:
    vm, strain_norm = _sfc_stress_metric_values(body.mesh, body.target_displacement, body.material)
    vm_max, _vm_p95 = _summary_pair(vm)
    strain_max, _strain_p95 = _summary_pair(strain_norm)
    return vm_max, strain_max


def _sfc_stress_metric_summaries(body: FirstFrameBody) -> tuple[float, float, float, float]:
    vm, strain_norm = _sfc_stress_metric_values(body.mesh, body.target_displacement, body.material)
    vm_max, vm_p95 = _summary_pair(vm)
    strain_max, strain_p95 = _summary_pair(strain_norm)
    return vm_max, vm_p95, strain_max, strain_p95


def _reference_stress_metrics(body: FirstFrameBody) -> tuple[float, float]:
    mises, strain = _reference_stress_metric_values(body.vtu)
    vm_max, _vm_p95 = _summary_pair(mises)
    strain_max, _strain_p95 = _summary_pair(strain)
    return vm_max, strain_max


def _reference_stress_metric_summaries(body: FirstFrameBody) -> tuple[float, float, float, float]:
    mises, strain = _reference_stress_metric_values(body.vtu)
    vm_max, vm_p95 = _summary_pair(mises)
    strain_max, strain_p95 = _summary_pair(strain)
    return vm_max, vm_p95, strain_max, strain_p95


def _safe_rel_error(value: float, reference: float) -> float:
    denom = abs(float(reference))
    if denom <= 1.0e-30:
        return abs(float(value) - float(reference))
    return abs(float(value) - float(reference)) / denom


def _model_row(case: FirstFrameCase) -> Row:
    master_sfc_vm, master_sfc_vm_p95, master_sfc_strain, master_sfc_strain_p95 = _sfc_stress_metric_summaries(case.master)
    slave_sfc_vm, slave_sfc_vm_p95, slave_sfc_strain, slave_sfc_strain_p95 = _sfc_stress_metric_summaries(case.slave)
    master_ref_vm, master_ref_vm_p95, master_ref_strain, master_ref_strain_p95 = _reference_stress_metric_summaries(case.master)
    slave_ref_vm, slave_ref_vm_p95, slave_ref_strain, slave_ref_strain_p95 = _reference_stress_metric_summaries(case.slave)
    slave_disp = np.linalg.norm(case.slave.target_displacement, axis=1)
    master_disp = np.linalg.norm(case.master.target_displacement, axis=1)
    return {
        "case_id": case.case_id,
        "problem": case.problem,
        "master_body": case.master.name,
        "slave_body": case.slave.name,
        "master_reference_rule": case.master.reference_rule,
        "slave_reference_rule": case.slave.reference_rule,
        "master_reference_score": case.master.reference_score,
        "slave_reference_score": case.slave.reference_score,
        "master_reference_score_details": case.master.reference_score_details,
        "slave_reference_score_details": case.slave.reference_score_details,
        "master_nodes": int(case.master.mesh.X.shape[0]),
        "slave_nodes": int(case.slave.mesh.X.shape[0]),
        "master_elements": int(case.master.mesh.elements.shape[0]),
        "slave_elements": int(case.slave.mesh.elements.shape[0]),
        "element_type": case.master.mesh.element_type,
        "E_MPa": case.master.material.E,
        "nu": case.master.material.nu,
        "contact_stiffness": case.contact_stiffness,
        "reference_metric_name": case.reference_metric_name,
        "reference_metric_value": case.reference_metric_value,
        "master_target_disp_max": float(np.max(master_disp)),
        "slave_target_disp_max": float(np.max(slave_disp)),
        "master_sfc_postprocess_max_von_mises": master_sfc_vm,
        "master_sfc_postprocess_p95_von_mises": master_sfc_vm_p95,
        "master_reference_max_von_mises": master_ref_vm,
        "master_reference_p95_von_mises": master_ref_vm_p95,
        "master_von_mises_rel_error": _safe_rel_error(master_sfc_vm, master_ref_vm),
        "master_von_mises_p95_rel_error": _safe_rel_error(master_sfc_vm_p95, master_ref_vm_p95),
        "slave_sfc_postprocess_max_von_mises": slave_sfc_vm,
        "slave_sfc_postprocess_p95_von_mises": slave_sfc_vm_p95,
        "slave_reference_max_von_mises": slave_ref_vm,
        "slave_reference_p95_von_mises": slave_ref_vm_p95,
        "slave_von_mises_rel_error": _safe_rel_error(slave_sfc_vm, slave_ref_vm),
        "slave_von_mises_p95_rel_error": _safe_rel_error(slave_sfc_vm_p95, slave_ref_vm_p95),
        "master_sfc_postprocess_max_strain_norm": master_sfc_strain,
        "master_sfc_postprocess_p95_strain_norm": master_sfc_strain_p95,
        "master_reference_max_strain_norm": master_ref_strain,
        "master_reference_p95_strain_norm": master_ref_strain_p95,
        "master_strain_norm_rel_error": _safe_rel_error(master_sfc_strain, master_ref_strain),
        "master_strain_norm_p95_rel_error": _safe_rel_error(master_sfc_strain_p95, master_ref_strain_p95),
        "slave_sfc_postprocess_max_strain_norm": slave_sfc_strain,
        "slave_sfc_postprocess_p95_strain_norm": slave_sfc_strain_p95,
        "slave_reference_max_strain_norm": slave_ref_strain,
        "slave_reference_p95_strain_norm": slave_ref_strain_p95,
        "slave_strain_norm_rel_error": _safe_rel_error(slave_sfc_strain, slave_ref_strain),
        "slave_strain_norm_p95_rel_error": _safe_rel_error(slave_sfc_strain_p95, slave_ref_strain_p95),
        "claim_scope": "first-frame external benchmark mesh reproduction with SFC dynamic SDF field contact",
    }


def _downsample_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if faces.shape[0] <= max_faces:
        return faces
    ids = np.unique(np.linspace(0, faces.shape[0] - 1, max_faces, dtype=int))
    return faces[ids]


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * max(float(np.max(maxs - mins)), 1.0e-9)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _draw_surface_field(
    ax,
    x_current: np.ndarray,
    faces: np.ndarray,
    nodal_values: np.ndarray,
    *,
    norm: Normalize,
    title: str,
    max_faces: int,
) -> Poly3DCollection:
    shown = _downsample_faces(faces, max_faces)
    values = np.mean(np.asarray(nodal_values, dtype=float)[shown], axis=1)
    collection = Poly3DCollection(
        [x_current[face] for face in shown],
        cmap="viridis",
        norm=norm,
        linewidths=0.0,
        edgecolors="none",
        antialiaseds=False,
    )
    collection.set_array(values)
    ax.add_collection3d(collection)
    ax.set_title(title, fontsize=9)
    _set_equal_axes(ax, x_current)
    ax.view_init(elev=24, azim=-52)
    return collection


def _write_case_figure(case: FirstFrameCase, out_dir: Path, *, quick: bool) -> Row:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    master_x = case.master.mesh.X + case.master.target_displacement
    slave_x = case.slave.mesh.X + case.slave.target_displacement
    master_faces = _boundary_triangles(case.master.mesh, master_x)
    slave_faces = _boundary_triangles(case.slave.mesh, slave_x)
    master_disp = np.linalg.norm(case.master.target_displacement, axis=1)
    slave_disp = np.linalg.norm(case.slave.target_displacement, axis=1)
    master_strain = _field_norm(case.master.vtu, "strain")
    slave_strain = _field_norm(case.slave.vtu, "strain")
    master_vm = _field_norm(case.master.vtu, "mises")
    slave_vm = _field_norm(case.slave.vtu, "mises")
    if not np.any(master_vm):
        stress = _tensor_field(case.master.vtu, "stress")
        master_vm = _von_mises_tensor(stress) if stress is not None else np.zeros(case.master.mesh.X.shape[0])
    if not np.any(slave_vm):
        stress = _tensor_field(case.slave.vtu, "stress")
        slave_vm = _von_mises_tensor(stress) if stress is not None else np.zeros(case.slave.mesh.X.shape[0])

    fields = [
        ("Displacement magnitude", master_disp, slave_disp),
        ("Strain norm", master_strain, slave_strain),
        ("von Mises stress", master_vm, slave_vm),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), subplot_kw={"projection": "3d"})
    # Paper-facing figures should show continuous surfaces.  Earlier quick
    # previews downsampled faces aggressively, which made the surface look like
    # disconnected triangular flakes even though the underlying TET boundary is
    # continuous.  Keep essentially all boundary faces here; the contact-query
    # sampling remains controlled separately by the validation runner options.
    max_faces = 100_000 if quick else 1_000_000
    for ax, (title, master_values, slave_values) in zip(axes, fields, strict=True):
        all_values = np.concatenate([master_values, slave_values])
        lo = float(np.nanmin(all_values))
        hi = float(np.nanmax(all_values))
        if not math.isfinite(lo) or not math.isfinite(hi) or np.isclose(lo, hi):
            hi = lo + 1.0
        norm = Normalize(vmin=lo, vmax=hi)
        col = _draw_surface_field(
            ax,
            master_x,
            master_faces,
            master_values,
            norm=norm,
            title=title,
            max_faces=max_faces,
        )
        _draw_surface_field(
            ax,
            slave_x,
            slave_faces,
            slave_values,
            norm=norm,
            title=title,
            max_faces=max_faces,
        )
        fig.colorbar(col, ax=ax, shrink=0.58, pad=0.02)
        _set_equal_axes(ax, np.vstack([master_x, slave_x]))
    fig.suptitle(f"{case.problem}: FuzzyContact first-frame fields on SFC model", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.03, top=0.86, wspace=0.05)
    png = fig_dir / f"{case.problem}_first_frame_fields_3d.png"
    pdf = fig_dir / f"{case.problem}_first_frame_fields_3d.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "case_id": case.case_id,
        "plot": f"{case.problem}_first_frame_fields_3d",
        "png": str(png.relative_to(out_dir)),
        "pdf": str(pdf.relative_to(out_dir)),
        "status": "ok",
        "description": "3D displacement, strain, and von Mises fields rendered with Times-style font.",
    }


def _write_summary(
    path: Path,
    *,
    model_rows: list[Row],
    contact_rows: list[Row],
    plot_rows: list[Row],
    outputs: dict[str, Path],
) -> None:
    lines = [
        "# FuzzyContact First-Frame SFC Models",
        "",
        "Benchmarks are processed in the requested order: problem 1, problem 3, then problem 4.",
        "",
        "## Scope",
        "",
        "- Model entry: public FuzzyContact VTU first displacement files.",
        "- Solver-side contact path: `DynamicNarrowBandSDF.build_required_points(...) -> field_contact`.",
        "- Projection is used only inside SDF field population, not as a contact query API.",
        "- This is external benchmark mesh reproduction, not a source-level FuzzyContact solver equivalence claim.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        if key != "summary":
            lines.append(f"- `{value.relative_to(path.parent).as_posix()}`")
    lines.extend(["", "## Model Reconstruction", "", "| Case | Reference rule | Nodes | Elements | Reference metric |", "| --- | --- | ---: | ---: | --- |"])
    for row in model_rows:
        nodes = int(row["master_nodes"]) + int(row["slave_nodes"])
        elements = int(row["master_elements"]) + int(row["slave_elements"])
        lines.append(
            f"| `{row['case_id']}` | {row['master_reference_rule']} / {row['slave_reference_rule']} | "
            f"{nodes} | {elements} | {row['reference_metric_name']} = {float(row['reference_metric_value']):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Coordinate and Field Convention Diagnostics",
            "",
            "- Reference coordinates are selected by scoring `points` and `points_minus_displacement` against the public VTU strain/stress fields.",
            "- Strain comparisons use tensor Frobenius norms. SFC engineering shear strains are converted to tensor shear before comparison.",
            "- Both maximum and p95 errors are written to `fuzzycontact_first_frame_models.csv`; p95 is less sensitive to isolated contact-tip peaks.",
            "",
            "## Field-Contact Evaluation",
            "",
            "| Case | state | closure | samples | phi min | active | update s | query ms/sample |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in contact_rows:
        lines.append(
            f"| `{row['case_id']}` | {row['state']} | {float(row['closure_offset']):.6e} | {int(row['slave_contact_samples'])} | "
            f"{float(row['phi_min']):.6e} | {int(row['active_samples'])} | "
            f"{float(row['field_update_seconds']):.6f} | {1000.0 * float(row['field_query_seconds_per_sample']):.6f} |"
        )
    lines.extend(["", "## Figures", ""])
    for row in plot_rows:
        lines.append(f"- `{row['png']}` and `{row['pdf']}`: {row['description']}")
    lines.extend(
        [
            "",
            "## Unsupported Claims",
            "",
            "- Full FuzzyContact sigmoidal traction-law equivalence.",
            "- Frictional stick-slip, self-contact, barrier contact, GPU acceleration, POD, or neural SDF.",
            "- A claim that VTU points are always undeformed reference coordinates; problem-specific reference rules are recorded in the CSV.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    out_dir: Path,
    *,
    source_dir: Path,
    quick: bool = False,
    problems: tuple[str, ...] = PROBLEM_ORDER,
    max_contact_samples: int = 48,
    max_master_faces: int = 1800,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(source_dir, problems)
    if not cases:
        raise RuntimeError(f"no requested FuzzyContact cases found under {source_dir}")
    if quick:
        max_contact_samples = min(int(max_contact_samples), 24)
        max_master_faces = min(int(max_master_faces), 650)

    model_rows: list[Row] = []
    contact_rows: list[Row] = []
    sample_rows: list[Row] = []
    plot_rows: list[Row] = []
    for case in cases:
        model_rows.append(_model_row(case))
        row, samples = _evaluate_field_contact(
            case,
            load_factor=1.0,
            state_name="reference_displacement",
            quick=quick,
            max_contact_samples=max_contact_samples,
            max_master_faces=max_master_faces,
        )
        contact_rows.append(row)
        sample_rows.extend(samples)
        if float(row["phi_min"]) >= 0.0:
            closure = float(row["phi_min"]) + max(0.25 * float(row["spacing"]), 1.0e-6)
            closed_row, closed_samples = _evaluate_field_contact(
                case,
                load_factor=1.0,
                state_name="sfc_closed_contact",
                closure_offset=closure,
                quick=quick,
                max_contact_samples=max_contact_samples,
                max_master_faces=max_master_faces,
            )
            contact_rows.append(closed_row)
            sample_rows.extend(closed_samples)
        plot_rows.append(_write_case_figure(case, out_dir, quick=quick))

    outputs = {
        "models": out_dir / "fuzzycontact_first_frame_models.csv",
        "contact": out_dir / "fuzzycontact_first_frame_field_contact.csv",
        "samples": out_dir / "fuzzycontact_first_frame_samples.csv",
        "plots": out_dir / "fuzzycontact_first_frame_plots.csv",
        "summary": out_dir / "fuzzycontact_first_frame_summary.md",
    }
    _write_csv(outputs["models"], model_rows)
    _write_csv(outputs["contact"], contact_rows)
    _write_csv(outputs["samples"], sample_rows)
    _write_csv(outputs["plots"], plot_rows)
    _write_summary(outputs["summary"], model_rows=model_rows, contact_rows=contact_rows, plot_rows=plot_rows, outputs=outputs)
    return outputs


def _parse_problems(value: str) -> tuple[str, ...]:
    if not value:
        return PROBLEM_ORDER
    out = tuple(part.strip() for part in value.split(",") if part.strip())
    for problem in out:
        if problem not in PROBLEM_ORDER:
            raise argparse.ArgumentTypeError(f"problem must be one of {', '.join(PROBLEM_ORDER)}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "results" / "fuzzycontact_vtu_reference_full" / "downloads" / "Dataset_for_Solutions",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "fuzzycontact_first_frame_sfc")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--problems", type=_parse_problems, default=PROBLEM_ORDER)
    parser.add_argument("--max-contact-samples", type=int, default=48)
    parser.add_argument("--max-master-faces", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(
        args.out_dir,
        source_dir=args.source_dir,
        quick=bool(args.quick),
        problems=args.problems,
        max_contact_samples=int(args.max_contact_samples),
        max_master_faces=int(args.max_master_faces),
    )
    print("FuzzyContact first-frame SFC model validation complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
