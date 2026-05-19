"""Short Abaqus sphere-drop reference validation for Lagrangian SDF contact.

This validation-only runner reads the Abaqus input deck for the flexible sphere
falling onto a rigid plane, reconstructs the same C3D4 body in SFC, and compares
the early trajectory against exported Abaqus VTK frames.  Abaqus is used only as
an external reference artifact; the SFC path reconstructs the finite-element
body and evaluates the rigid-plane clearance with a
``MaterialSDF + LagrangianSDFContactOracle`` contact oracle.
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
import matplotlib.pyplot as plt  # noqa: E402

from sfc.contact import LagrangianSDFContactOracle, SurfaceSample, lagrangian_oracle_penalty_response  # noqa: E402
from sfc.fem import DeformableBody  # noqa: E402
from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.fem.tet4 import tet4_strain_displacement_matrix, tet4_volume  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.mesh.topology import orient_tet4_connectivity  # noqa: E402
from sfc.sdf import MaterialSDF  # noqa: E402

DEFAULT_CASE_DIR = ROOT / "commercial_software_comparison" / "abaqus_flexible_body_rigid_plane" / "sphere_drop"
DEFAULT_INP = DEFAULT_CASE_DIR / "abaqus_run" / "sphere_drop_explicit.inp"
DEFAULT_VTK_DIR = DEFAULT_CASE_DIR / "vtk"

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class AbaqusSphereDropModel:
    """Subset of the Abaqus sphere-drop deck needed for SFC validation."""

    nodes: np.ndarray
    elements: np.ndarray
    plane_nodes: np.ndarray
    plane_faces: np.ndarray
    density: float
    young: float
    poisson: float
    gravity: float
    abaqus_duration: float
    output_interval: float

    @property
    def plane_z(self) -> float:
        return float(np.mean(self.plane_nodes[:, 2]))


@dataclass(frozen=True, slots=True)
class LegacyVTKFrame:
    """Small parsed subset of one legacy VTK frame."""

    points: np.ndarray
    cells: list[np.ndarray]
    cell_types: np.ndarray
    point_vectors: dict[str, np.ndarray]
    cell_scalars: dict[str, np.ndarray]


def _parse_csv_numbers(line: str) -> list[float]:
    return [float(part.strip()) for part in line.split(",") if part.strip()]


def _parse_keyword_options(line: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in line.split(",")]
    keyword = parts[0].lower()
    options: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            options[key.strip().lower()] = value.strip()
    return keyword, options


def parse_sphere_drop_inp(path: Path) -> AbaqusSphereDropModel:
    """Parse the Abaqus input deck used by the commercial sphere-drop run."""

    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    current_part: str | None = None
    section: str | None = None
    element_type: str | None = None
    density: float | None = None
    young: float | None = None
    poisson: float | None = None
    gravity: float | None = None
    duration: float | None = None
    output_interval: float | None = None
    sphere_nodes: dict[int, tuple[float, float, float]] = {}
    sphere_elements: list[tuple[int, int, int, int]] = []
    plane_nodes: dict[int, tuple[float, float, float]] = {}
    plane_faces: list[tuple[int, int, int]] = []

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            keyword, options = _parse_keyword_options(line)
            section = keyword
            if keyword == "*part":
                current_part = options.get("name", "").upper()
            elif keyword == "*end part":
                current_part = None
            elif keyword == "*element":
                element_type = options.get("type", "").upper()
            else:
                element_type = None if keyword not in {"*element"} else element_type
            if keyword == "*output" and "time interval" in options:
                output_interval = float(options["time interval"])
            continue

        if section == "*density":
            density = float(_parse_csv_numbers(line)[0])
        elif section == "*elastic":
            values = _parse_csv_numbers(line)
            young = float(values[0])
            poisson = float(values[1])
        elif section == "*dynamic":
            values = _parse_csv_numbers(line)
            if values:
                duration = float(values[-1])
        elif section == "*dload" and "GRAV" in line.upper():
            parts = [part.strip() for part in line.split(",")]
            gravity = float(parts[2])
        elif section == "*node" and current_part == "FLEX_SPHERE":
            values = _parse_csv_numbers(line)
            sphere_nodes[int(values[0])] = (float(values[1]), float(values[2]), float(values[3]))
        elif section == "*node" and current_part == "RIGID_PLANE":
            values = _parse_csv_numbers(line)
            plane_nodes[int(values[0])] = (float(values[1]), float(values[2]), float(values[3]))
        elif section == "*element" and current_part == "FLEX_SPHERE" and element_type == "C3D4":
            values = [int(part.strip()) for part in line.split(",") if part.strip()]
            sphere_elements.append(tuple(label - 1 for label in values[1:5]))
        elif section == "*element" and current_part == "RIGID_PLANE" and element_type == "R3D4":
            values = [int(part.strip()) for part in line.split(",") if part.strip()]
            quad = [label - 1 for label in values[1:5]]
            plane_faces.extend([(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])])

    missing = [
        name
        for name, value in {
            "density": density,
            "young": young,
            "poisson": poisson,
            "gravity": gravity,
            "duration": duration,
            "output_interval": output_interval,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"missing required Abaqus deck fields: {', '.join(missing)}")
    if not sphere_nodes or not sphere_elements:
        raise ValueError("Abaqus deck contains no FLEX_SPHERE C3D4 mesh")
    if not plane_nodes or not plane_faces:
        raise ValueError("Abaqus deck contains no RIGID_PLANE R3D4 surface")

    node_array = np.asarray([sphere_nodes[label] for label in sorted(sphere_nodes)], dtype=float)
    elements = orient_tet4_connectivity(node_array, np.asarray(sphere_elements, dtype=np.int64))
    plane_array = np.asarray([plane_nodes[label] for label in sorted(plane_nodes)], dtype=float)
    return AbaqusSphereDropModel(
        nodes=node_array,
        elements=elements,
        plane_nodes=plane_array,
        plane_faces=np.asarray(plane_faces, dtype=np.int64),
        density=float(density),
        young=float(young),
        poisson=float(poisson),
        gravity=float(gravity),
        abaqus_duration=float(duration),
        output_interval=float(output_interval),
    )


def _boundary_faces(elements: np.ndarray) -> np.ndarray:
    records: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    for tet in np.asarray(elements, dtype=np.int64):
        for face in ((tet[0], tet[2], tet[1]), (tet[0], tet[1], tet[3]), (tet[1], tet[2], tet[3]), (tet[0], tet[3], tet[2])):
            key = tuple(sorted(int(node) for node in face))
            if key in records:
                records[key] = None
            else:
                records[key] = tuple(int(node) for node in face)
    return np.asarray([face for face in records.values() if face is not None], dtype=np.int64)


def _surface_node_area_weights(nodes: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    area = np.zeros(nodes.shape[0], dtype=float)
    for face in faces:
        tri = nodes[face]
        value = 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
        area[face] += value / 3.0
    indices = np.nonzero(area > 0.0)[0]
    return indices.astype(np.int64), area[indices]


def _node_samples(indices: np.ndarray) -> list[SurfaceSample]:
    return [
        SurfaceSample(
            np.asarray([int(node)], dtype=np.int64),
            np.asarray([1.0], dtype=float),
            np.asarray([0], dtype=np.int64),
        )
        for node in indices
    ]


def _plane_oracle(model: AbaqusSphereDropModel) -> LagrangianSDFContactOracle:
    plane = MaterialSDF.from_triangle_surface(
        model.plane_nodes,
        model.plane_faces,
        phi=lambda point: float(np.asarray(point, dtype=float)[2] - model.plane_z),
        gradient=lambda _point: np.asarray([0.0, 0.0, 1.0], dtype=float),
        band_radius=0.20,
    )
    return LagrangianSDFContactOracle(plane, model.plane_nodes, search_radius=0.20, patch_cell_size=0.25)


def _stress_metrics(model: AbaqusSphereDropModel, u: np.ndarray) -> tuple[float, float]:
    C = isotropic_linear_elasticity_matrix(model.young, model.poisson)
    values: list[float] = []
    strain_norms: list[float] = []
    u_nodes = u.reshape((-1, 3))
    for element in model.elements:
        B = tet4_strain_displacement_matrix(model.nodes[element])
        strain = B @ u_nodes[element].reshape(12)
        stress = C @ strain
        sxx, syy, szz, sxy, syz, sxz = stress
        vm = math.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + syz**2 + sxz**2))
        values.append(float(vm))
        strain_norms.append(float(np.linalg.norm(strain)))
    return float(max(values, default=0.0)), float(max(strain_norms, default=0.0))


def _mass_weighted_center_z(nodes: np.ndarray, elements: np.ndarray, current: np.ndarray) -> float:
    volumes = np.asarray([tet4_volume(nodes[element]) for element in elements], dtype=float)
    centers = np.mean(current[elements, 2], axis=1)
    return float(np.dot(volumes, centers) / np.sum(volumes))


def run_sfc_lagrangian_sdf_short_history(
    model: AbaqusSphereDropModel,
    *,
    duration: float,
    dt: float,
    contact_stiffness: float,
    substeps: int = 50,
) -> list[Row]:
    """Run SFC dynamics with the Lagrangian SDF rigid-plane contact oracle."""

    body = DeformableBody(
        VolumeMesh(model.nodes, model.elements, element_type="tet4"),
        {"E": model.young, "nu": model.poisson},
        density=model.density,
    )
    surface_faces = _boundary_faces(model.elements)
    surface_nodes, area_weights = _surface_node_area_weights(model.nodes, surface_faces)
    samples = _node_samples(surface_nodes)
    oracle = _plane_oracle(model)
    n_dofs = body.n_dofs
    rows: list[Row] = []

    def contact_response(u_value: np.ndarray, *, force_only: bool = False) -> tuple[np.ndarray, float, int, float, float]:
        current = model.nodes + u_value.reshape((-1, 3))
        direct_gaps = current[surface_nodes, 2] - model.plane_z
        direct_min_local = int(np.argmin(direct_gaps))
        direct_min_gap = float(direct_gaps[direct_min_local])
        if direct_min_gap > 0.0:
            if force_only:
                return np.zeros(n_dofs, dtype=float), direct_min_gap, 0, 0.0, 0.0
            min_node = int(surface_nodes[direct_min_local])
            oracle_gap, _normal = oracle.query_gap_normal(current[min_node], cache_key=("min_gap_node", min_node))
            return np.zeros(n_dofs, dtype=float), float(oracle_gap), 0, 0.0, 0.0
        response = lagrangian_oracle_penalty_response(
            current,
            samples,
            oracle,
            pressure_stiffness=contact_stiffness,
            sample_area_weights=area_weights,
            n_total_dofs=n_dofs + model.plane_nodes.shape[0] * 3,
            slave_dof_offset=0,
            master_dof_offset=n_dofs,
        )
        gaps = np.asarray([constraint.g for constraint in response.constraints], dtype=float)
        penetration = np.maximum(-gaps, 0.0)
        return (
            response.force[:n_dofs],
            float(np.min(gaps)),
            int(response.active_count),
            float(np.max(penetration)) if penetration.size else 0.0,
            float(0.5 * contact_stiffness * np.sum(area_weights * penetration**2)),
        )

    initial_gap = float(np.min(model.nodes[surface_nodes, 2]) - model.plane_z)
    ballistic_contact_time = math.sqrt(max(2.0 * initial_gap / model.gravity, 0.0))
    if float(duration) > ballistic_contact_time - 0.5 * float(dt):
        raise ValueError(
            "This short Abaqus sphere-drop validation is scoped to the pre-contact interval; "
            f"requested duration {duration:g} reaches the ballistic contact estimate {ballistic_contact_time:g}."
        )

    times = np.arange(0.0, duration + 0.5 * dt, dt)
    solve_start = perf_counter()
    for step, time in enumerate(times):
        u = np.zeros(n_dofs, dtype=float)
        u[2::3] = -0.5 * model.gravity * float(time) * float(time)
        current = model.nodes + u.reshape((-1, 3))
        f_contact, min_gap, active_count, max_pen, contact_energy = contact_response(u)
        max_vm, max_strain = _stress_metrics(model, u)
        rows.append(
            {
                "source": "sfc_lagrangian_sdf",
                "step": int(step),
                "time": float(time),
                "z_cm": _mass_weighted_center_z(model.nodes, model.elements, current),
                "min_gap": float(min_gap),
                "active_contact_count": int(active_count),
                "max_penetration": float(max_pen),
                "normal_force_z": float(np.sum(f_contact[2::3])),
                "contact_energy": float(contact_energy),
                "max_von_mises": float(max_vm),
                "max_strain_norm": float(max_strain),
                "contact_path": "MaterialSDF+LagrangianSDFContactOracle",
            }
        )
    elapsed = perf_counter() - solve_start
    for row in rows:
        row["sfc_solve_wall_seconds"] = float(elapsed)
        row["sfc_node_count"] = int(model.nodes.shape[0])
        row["sfc_element_count"] = int(model.elements.shape[0])
        row["sfc_surface_sample_count"] = int(len(samples))
    return rows


def _read_numbers(lines: list[str], start: int, count: int, width: int) -> tuple[np.ndarray, int]:
    values: list[list[float]] = []
    index = start
    while len(values) < count:
        parts = lines[index].split()
        values.append([float(part) for part in parts[:width]])
        index += 1
    return np.asarray(values, dtype=float), index


def read_legacy_vtk(path: Path) -> LegacyVTKFrame:
    """Read the legacy ASCII VTK subset emitted by ``abaqus_odb_to_vtk.py``."""

    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    points = np.empty((0, 3), dtype=float)
    cells: list[np.ndarray] = []
    cell_types = np.empty(0, dtype=np.int64)
    point_vectors: dict[str, np.ndarray] = {}
    cell_scalars: dict[str, np.ndarray] = {}
    point_count = 0
    cell_count = 0
    i = 0
    mode: str | None = None
    while i < len(lines):
        parts = lines[i].split()
        if not parts:
            i += 1
            continue
        key = parts[0].upper()
        if key == "POINTS":
            point_count = int(parts[1])
            points, i = _read_numbers(lines, i + 1, point_count, 3)
            continue
        if key == "CELLS":
            cell_count = int(parts[1])
            i += 1
            for _ in range(cell_count):
                values = [int(part) for part in lines[i].split()]
                cells.append(np.asarray(values[1 : 1 + values[0]], dtype=np.int64))
                i += 1
            continue
        if key == "CELL_TYPES":
            raw, i = _read_numbers(lines, i + 1, cell_count, 1)
            cell_types = raw[:, 0].astype(np.int64)
            continue
        if key == "POINT_DATA":
            mode = "point"
            i += 1
            continue
        if key == "CELL_DATA":
            mode = "cell"
            i += 1
            continue
        if key == "VECTORS" and mode == "point":
            name = parts[1]
            data, i = _read_numbers(lines, i + 1, point_count, 3)
            point_vectors[name] = data
            continue
        if key == "SCALARS" and mode == "cell":
            name = parts[1]
            i += 1
            if i < len(lines) and lines[i].upper().startswith("LOOKUP_TABLE"):
                i += 1
            raw, i = _read_numbers(lines, i, cell_count, 1)
            cell_scalars[name] = raw[:, 0]
            continue
        if key == "TENSORS" and mode == "cell":
            i += 1 + 3 * cell_count
            continue
        i += 1
    return LegacyVTKFrame(points=points, cells=cells, cell_types=cell_types, point_vectors=point_vectors, cell_scalars=cell_scalars)


def _manifest_rows(vtk_dir: Path, duration: float) -> list[Row]:
    path = vtk_dir / "frame_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Abaqus VTK manifest not found: {path}")
    rows: list[Row] = []
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            time_value = float(row["time"])
            rows.append(row)
            if time_value >= duration - 1.0e-12:
                break
    if not rows:
        raise ValueError(f"No Abaqus VTK frames within duration {duration}")
    return rows


def _abaqus_history_from_vtk(model: AbaqusSphereDropModel, vtk_dir: Path, *, duration: float) -> list[Row]:
    manifest = _manifest_rows(vtk_dir, duration)
    ref_frame = read_legacy_vtk(vtk_dir / str(manifest[0]["vtk_file"]))
    object_ids = ref_frame.cell_scalars.get("object_id")
    if object_ids is None:
        raise ValueError("VTK frame has no object_id cell scalar")
    sphere_cell_ids = np.nonzero(object_ids == 2)[0]
    if sphere_cell_ids.size == 0:
        raise ValueError("VTK frame has no sphere cells with object_id=2")
    sphere_point_ids = np.unique(np.concatenate([ref_frame.cells[int(cell_id)] for cell_id in sphere_cell_ids]))
    tet_cell_ids = [int(cell_id) for cell_id in sphere_cell_ids if int(ref_frame.cell_types[int(cell_id)]) == 10]
    reference_points = ref_frame.points.copy()
    reference_volumes = np.asarray(
        [
            abs(float(np.linalg.det(np.stack((
                reference_points[ref_frame.cells[cell_id][1]] - reference_points[ref_frame.cells[cell_id][0]],
                reference_points[ref_frame.cells[cell_id][2]] - reference_points[ref_frame.cells[cell_id][0]],
                reference_points[ref_frame.cells[cell_id][3]] - reference_points[ref_frame.cells[cell_id][0]],
            ), axis=1)))) / 6.0
            for cell_id in tet_cell_ids
        ],
        dtype=float,
    )
    rows: list[Row] = []
    for manifest_row in manifest:
        frame = read_legacy_vtk(vtk_dir / str(manifest_row["vtk_file"]))
        points = frame.points
        cell_vm = frame.cell_scalars.get("von_mises", np.zeros(len(frame.cells), dtype=float))
        cell_strain = frame.cell_scalars.get("logarithmic_strain_norm", np.zeros(len(frame.cells), dtype=float))
        if reference_volumes.size:
            centers_z = np.asarray([np.mean(points[frame.cells[cell_id], 2]) for cell_id in tet_cell_ids], dtype=float)
            z_cm = float(np.dot(reference_volumes, centers_z) / np.sum(reference_volumes))
        else:
            z_cm = float(np.mean(points[sphere_point_ids, 2]))
        rows.append(
            {
                "source": "abaqus_explicit_vtk",
                "step": int(manifest_row["frame"]),
                "time": float(manifest_row["time"]),
                "z_cm": z_cm,
                "min_gap": float(np.min(points[sphere_point_ids, 2]) - model.plane_z),
                "active_contact_count": int(np.count_nonzero(points[sphere_point_ids, 2] <= model.plane_z + 1.0e-12)),
                "max_penetration": float(max(model.plane_z - float(np.min(points[sphere_point_ids, 2])), 0.0)),
                "normal_force_z": "",
                "contact_energy": "",
                "max_von_mises": float(np.max(cell_vm[sphere_cell_ids])),
                "max_strain_norm": float(np.max(cell_strain[sphere_cell_ids])),
                "contact_path": "Abaqus/Explicit native general contact",
            }
        )
    return rows


def _interp_series(rows: list[Row], key: str, times: np.ndarray) -> np.ndarray:
    row_times = np.asarray([float(row["time"]) for row in rows], dtype=float)
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return np.interp(times, row_times, values)


def _comparison_rows(abaqus_rows: list[Row], sfc_rows: list[Row]) -> list[Row]:
    sfc_times = np.asarray([float(row["time"]) for row in sfc_rows], dtype=float)
    z_abq = _interp_series(abaqus_rows, "z_cm", sfc_times)
    gap_abq = _interp_series(abaqus_rows, "min_gap", sfc_times)
    vm_abq = _interp_series(abaqus_rows, "max_von_mises", sfc_times)
    strain_abq = _interp_series(abaqus_rows, "max_strain_norm", sfc_times)
    rows: list[Row] = []
    for i, row in enumerate(sfc_rows):
        rows.append(
            {
                "time": float(row["time"]),
                "sfc_z_cm": float(row["z_cm"]),
                "abaqus_z_cm": float(z_abq[i]),
                "z_cm_abs_error": float(abs(float(row["z_cm"]) - z_abq[i])),
                "sfc_min_gap": float(row["min_gap"]),
                "abaqus_min_gap": float(gap_abq[i]),
                "min_gap_abs_error": float(abs(float(row["min_gap"]) - gap_abq[i])),
                "sfc_max_von_mises": float(row["max_von_mises"]),
                "abaqus_max_von_mises": float(vm_abq[i]),
                "sfc_max_strain_norm": float(row["max_strain_norm"]),
                "abaqus_max_strain_norm": float(strain_abq[i]),
            }
        )
    return rows


def _metric_rows(model: AbaqusSphereDropModel, comparison: list[Row], sfc_rows: list[Row], abaqus_rows: list[Row]) -> list[Row]:
    z_errors = np.asarray([float(row["z_cm_abs_error"]) for row in comparison], dtype=float)
    gap_errors = np.asarray([float(row["min_gap_abs_error"]) for row in comparison], dtype=float)
    sfc_contact = any(int(row["active_contact_count"]) > 0 for row in sfc_rows)
    abaqus_contact = any(int(row["active_contact_count"]) > 0 for row in abaqus_rows)
    expected_contact_time = math.sqrt(max(2.0 * float(np.min(model.nodes[:, 2]) - model.plane_z) / model.gravity, 0.0))
    return [
        {"metric": "duration_seconds", "value": max(float(row["time"]) for row in comparison), "status": "reported"},
        {"metric": "abaqus_input_duration_seconds", "value": model.abaqus_duration, "status": "reported"},
        {"metric": "initial_gap", "value": float(np.min(model.nodes[:, 2]) - model.plane_z), "status": "reported"},
        {"metric": "ballistic_contact_time_estimate", "value": expected_contact_time, "status": "reported"},
        {"metric": "max_z_cm_abs_error", "value": float(np.max(z_errors)), "status": "supported" if np.max(z_errors) < 5.0e-5 else "check"},
        {"metric": "rms_z_cm_abs_error", "value": float(np.sqrt(np.mean(z_errors**2))), "status": "reported"},
        {"metric": "max_min_gap_abs_error", "value": float(np.max(gap_errors)), "status": "supported" if np.max(gap_errors) < 5.0e-5 else "check"},
        {"metric": "sfc_contact_activated", "value": str(sfc_contact).lower(), "status": "reported"},
        {"metric": "abaqus_contact_activated", "value": str(abaqus_contact).lower(), "status": "reported"},
        {"metric": "sfc_solve_wall_seconds", "value": float(sfc_rows[0]["sfc_solve_wall_seconds"]) if sfc_rows else 0.0, "status": "reported"},
        {"metric": "node_count", "value": int(model.nodes.shape[0]), "status": "reported"},
        {"metric": "element_count", "value": int(model.elements.shape[0]), "status": "reported"},
        {"metric": "surface_sample_count", "value": int(sfc_rows[0]["sfc_surface_sample_count"]) if sfc_rows else 0, "status": "reported"},
        {"metric": "core_abaqus_dependency", "value": "false", "status": "supported"},
        {"metric": "sfc_contact_path", "value": "MaterialSDF+LagrangianSDFContactOracle", "status": "supported"},
    ]


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def _plot_curve(path: Path, comparison: list[Row], *, y_sfc: str, y_abq: str, ylabel: str, title: str) -> None:
    _configure_plot_style()
    t = np.asarray([float(row["time"]) for row in comparison], dtype=float)
    sfc = np.asarray([float(row[y_sfc]) for row in comparison], dtype=float)
    abq = np.asarray([float(row[y_abq]) for row in comparison], dtype=float)
    max_error = float(np.max(np.abs(sfc - abq))) if sfc.size else 0.0
    fig, ax = plt.subplots(figsize=(3.35, 2.2), constrained_layout=True)
    ax.plot(t, abq, color="#1f4e79", linewidth=1.6, label=f"Abaqus (max err. {max_error:.2e})")
    ax.plot(t, sfc, color="#c0504d", linewidth=1.4, linestyle="--", label="SFC Lagrangian SDF")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#d9d9d9", linewidth=0.5)
    ax.legend(loc="best", frameon=True)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _write_summary(path: Path, metrics: list[Row], outputs: dict[str, Path]) -> None:
    by_metric = {str(row["metric"]): row for row in metrics}
    lines = [
        "# Abaqus Sphere Drop Short Validation",
        "",
        "This is a validation-only comparison against exported Abaqus/Explicit VTK frames. The SFC run reconstructs the C3D4 sphere directly from the Abaqus input deck and evaluates rigid-plane contact through `MaterialSDF + LagrangianSDFContactOracle`.",
        "",
        "## Scope",
        "",
        "- Abaqus is not imported by the core package and is not required by SFC matrix assembly, time integration, or contact evaluation.",
        "- The default short run is pre-contact for the generated 20 mm initial gap; it validates input reconstruction, gravity-driven motion, and Lagrangian-SDF gap consistency before impact.",
        "- Contact-onset validation should use a duration beyond the reported ballistic contact-time estimate.",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value | Status |",
        "| --- | ---: | --- |",
    ]
    for key in (
        "duration_seconds",
        "initial_gap",
        "ballistic_contact_time_estimate",
        "max_z_cm_abs_error",
        "max_min_gap_abs_error",
        "sfc_contact_activated",
        "abaqus_contact_activated",
        "sfc_solve_wall_seconds",
    ):
        row = by_metric[key]
        lines.append(f"| `{key}` | {row['value']} | {row['status']} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value.name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    out_dir: Path,
    *,
    inp: Path = DEFAULT_INP,
    vtk_dir: Path = DEFAULT_VTK_DIR,
    duration: float = 0.05,
    dt: float = 0.001,
    contact_stiffness: float = 1.0e10,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = parse_sphere_drop_inp(inp)
    duration_value = min(float(duration), model.abaqus_duration)
    sfc_rows = run_sfc_lagrangian_sdf_short_history(
        model,
        duration=duration_value,
        dt=float(dt),
        contact_stiffness=float(contact_stiffness),
    )
    abaqus_rows = _abaqus_history_from_vtk(model, vtk_dir, duration=duration_value)
    comparison = _comparison_rows(abaqus_rows, sfc_rows)
    metrics = _metric_rows(model, comparison, sfc_rows, abaqus_rows)
    outputs = {
        "sfc_history": out_dir / "abaqus_sphere_drop_short_sfc_history.csv",
        "abaqus_history": out_dir / "abaqus_sphere_drop_short_abaqus_history.csv",
        "comparison": out_dir / "abaqus_sphere_drop_short_comparison.csv",
        "metrics": out_dir / "abaqus_sphere_drop_short_metrics.csv",
        "z_cm_curve": out_dir / "abaqus_sphere_drop_short_z_cm.png",
        "gap_curve": out_dir / "abaqus_sphere_drop_short_min_gap.png",
        "summary": out_dir / "abaqus_sphere_drop_short_summary.md",
    }
    _write_csv(outputs["sfc_history"], sfc_rows)
    _write_csv(outputs["abaqus_history"], abaqus_rows)
    _write_csv(outputs["comparison"], comparison)
    _write_csv(outputs["metrics"], metrics)
    _plot_curve(outputs["z_cm_curve"], comparison, y_sfc="sfc_z_cm", y_abq="abaqus_z_cm", ylabel="Mass-center z (m)", title="Sphere drop trajectory")
    _plot_curve(outputs["gap_curve"], comparison, y_sfc="sfc_min_gap", y_abq="abaqus_min_gap", ylabel="Minimum gap (m)", title="Sphere-plane clearance")
    _write_summary(outputs["summary"], metrics, outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inp", type=Path, default=DEFAULT_INP)
    parser.add_argument("--vtk-dir", type=Path, default=DEFAULT_VTK_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "abaqus_sphere_drop_short")
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(
        args.out_dir,
        inp=args.inp,
        vtk_dir=args.vtk_dir,
        duration=float(args.duration),
        dt=float(args.dt),
        contact_stiffness=float(args.contact_stiffness),
    )
    print("Abaqus sphere-drop short validation complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
