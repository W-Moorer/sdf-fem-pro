"""Short commercial-reference sphere/cantilever comparison.

This validation runner compares the existing Abaqus/Explicit sphere-on-
cantilever reference artifact against:

* an independently generated CalculiX native contact model, and
* an independently solved SFC Lagrangian/material-SDF contact model.

The commercial data are used only as external validation files under
``commercial_software_comparison``.  No Abaqus data are imported by the core
``sfc`` package.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
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

from sfc.contact.lagrangian_sdf_oracle import (  # noqa: E402
    LagrangianSDFContactOracle,
    lagrangian_oracle_penalty_response,
)
from sfc.contact.narrow_phase import SurfaceSample  # noqa: E402
from sfc.fem import DeformableBody, assemble_gravity_force, assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.sdf.material_sdf import MaterialSDF  # noqa: E402
from validation.run_calculix_drop_impact_comparison import (  # noqa: E402
    _calculix_version,
    _parse_calculix_dat_displacements,
    _wsl_path,
    calculix_available,
)

Row = dict[str, Any]
JOB_NAME = "sphere_cantilever_short_ccx"
DEFAULT_ABAQUS_DIR = ROOT / "commercial_software_comparison" / "abaqus_sphere_cantilever"
DEFAULT_ABAQUS_INP = DEFAULT_ABAQUS_DIR / "abaqus_run" / "sphere_cantilever_explicit.inp"


@dataclass(frozen=True, slots=True)
class ExplicitContactState:
    """Force and diagnostics for one explicit contact evaluation."""

    force: np.ndarray
    min_gap: float
    active_count: int


@dataclass(frozen=True, slots=True)
class InpMaterial:
    """Material values parsed from the Abaqus input deck."""

    young: float
    poisson: float
    density: float


@dataclass(frozen=True, slots=True)
class CommercialShortModel:
    """Shared mesh and boundary data parsed from the commercial input deck."""

    source_inp: Path
    duration: float
    dt: float
    initial_velocity_z: float
    gravity: float
    beam_material: InpMaterial
    sphere_material: InpMaterial
    beam_nodes: np.ndarray
    beam_elements: np.ndarray
    beam_fixed_nodes: np.ndarray
    sphere_nodes: np.ndarray
    sphere_elements: np.ndarray
    sphere_surface_faces: np.ndarray
    sphere_surface_refs: list[tuple[int, str]]
    sphere_sample_faces: np.ndarray
    sphere_sample_areas: np.ndarray
    beam_top_faces: np.ndarray
    beam_top_refs: list[tuple[int, str]]

    @property
    def beam_node_count(self) -> int:
        return int(self.beam_nodes.shape[0])

    @property
    def sphere_node_count(self) -> int:
        return int(self.sphere_nodes.shape[0])

    @property
    def combined_node_count(self) -> int:
        return self.beam_node_count + self.sphere_node_count


def _write_csv(path: Path, rows: list[Row], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_id_list(lines: list[str], ids: np.ndarray | list[int], *, width: int = 16) -> None:
    values = [int(value) for value in ids]
    for start in range(0, len(values), width):
        lines.append(", ".join(str(value) for value in values[start : start + width]))


def _as_array_nodes(nodes: list[tuple[float, float, float]]) -> np.ndarray:
    return np.asarray(nodes, dtype=float)


def _as_array_elements(elements: list[tuple[int, ...]]) -> np.ndarray:
    return np.asarray([[int(node) - 1 for node in element] for element in elements], dtype=np.int64)


def _keyword_params(line: str) -> tuple[str, dict[str, str]]:
    fields = [field.strip() for field in line.strip().split(",")]
    key = fields[0].lower()
    params: dict[str, str] = {}
    for field in fields[1:]:
        if "=" in field:
            name, value = field.split("=", 1)
            params[name.strip().lower()] = value.strip()
        elif field:
            params[field.strip().lower()] = ""
    return key, params


def _data_values(line: str) -> list[str]:
    return [value.strip() for value in line.split(",") if value.strip()]


def _next_data_line(lines: list[str], start: int) -> tuple[int, str]:
    i = int(start)
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("**"):
            if stripped.startswith("*"):
                raise ValueError(f"expected data line, got keyword {stripped!r}")
            return i + 1, stripped
        i += 1
    raise ValueError("expected data line before end of file")


def _parse_numeric_data_block(lines: list[str], start: int) -> tuple[int, list[list[str]]]:
    block: list[list[str]] = []
    i = int(start)
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("**"):
            i += 1
            continue
        if stripped.startswith("*"):
            break
        block.append(_data_values(stripped))
        i += 1
    return i, block


def _parse_part(lines: list[str], start: int) -> tuple[int, dict[str, Any]]:
    nodes: dict[int, tuple[float, float, float]] = {}
    elements: list[tuple[int, tuple[int, ...]]] = []
    element_type = ""
    material = ""
    i = int(start)
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("**"):
            i += 1
            continue
        if not stripped.startswith("*"):
            i += 1
            continue
        key, params = _keyword_params(stripped)
        if key == "*end part":
            return i + 1, {"nodes": nodes, "elements": elements, "element_type": element_type, "material": material}
        if key == "*node":
            i, block = _parse_numeric_data_block(lines, i + 1)
            for row in block:
                if len(row) < 4:
                    continue
                nodes[int(row[0])] = (float(row[1]), float(row[2]), float(row[3]))
            continue
        if key == "*element":
            element_type = params.get("type", element_type).upper()
            i, block = _parse_numeric_data_block(lines, i + 1)
            for row in block:
                if len(row) >= 5:
                    elements.append((int(row[0]), tuple(int(value) for value in row[1:])))
            continue
        if key == "*solid section":
            material = params.get("material", material)
        i += 1
    raise ValueError("unterminated *Part block")


def _part_to_arrays(part: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    labels = sorted(int(label) for label in part["nodes"])
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    nodes = np.asarray([part["nodes"][label] for label in labels], dtype=float)
    elements = np.asarray(
        [[label_to_index[int(label)] for label in conn] for _eid, conn in sorted(part["elements"], key=lambda item: int(item[0]))],
        dtype=np.int64,
    )
    return nodes, elements


def _read_abaqus_input_model(inp_path: Path, *, duration: float | None = None, dt: float | None = None) -> CommercialShortModel:
    """Parse the commercial Abaqus input deck as the authoritative model source."""

    lines = inp_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    materials: dict[str, dict[str, float]] = {}
    parts: dict[str, dict[str, Any]] = {}
    nsets: dict[str, list[int]] = {}
    initial_velocity_z = -1.2
    gravity = 9.81
    deck_duration = 3.0
    output_interval = 0.001
    deck_dt = output_interval
    current_material = ""
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("**") or not stripped.startswith("*"):
            i += 1
            continue
        key, params = _keyword_params(stripped)
        if key == "*material":
            current_material = params.get("name", "").upper()
            materials.setdefault(current_material, {})
            i += 1
            continue
        if key == "*density" and current_material:
            i, data = _next_data_line(lines, i + 1)
            materials[current_material]["density"] = float(_data_values(data)[0])
            continue
        if key == "*elastic" and current_material:
            i, data = _next_data_line(lines, i + 1)
            values = _data_values(data)
            materials[current_material]["young"] = float(values[0])
            materials[current_material]["poisson"] = float(values[1])
            continue
        if key == "*part":
            part_name = params.get("name", "").upper()
            i, part = _parse_part(lines, i + 1)
            parts[part_name] = part
            continue
        if key == "*nset":
            set_name = params.get("nset", "").upper()
            i, block = _parse_numeric_data_block(lines, i + 1)
            values: list[int] = []
            for row in block:
                values.extend(int(value) for value in row)
            nsets[set_name] = values
            continue
        if key == "*initial conditions" and params.get("type", "").lower() == "velocity":
            i, block = _parse_numeric_data_block(lines, i + 1)
            for row in block:
                if len(row) >= 3 and row[0].strip().upper() == "SPHERE_NODES" and int(float(row[1])) == 3:
                    initial_velocity_z = float(row[2])
            continue
        if key == "*dynamic":
            i, block = _parse_numeric_data_block(lines, i + 1)
            if block:
                numeric = [float(value) for row in block[:1] for value in row if _is_float(value)]
                if len(numeric) >= 2:
                    deck_dt = float(numeric[0])
                    deck_duration = float(numeric[-1])
                elif numeric:
                    deck_duration = numeric[-1]
            continue
        if key == "*dload":
            i, block = _parse_numeric_data_block(lines, i + 1)
            for row in block:
                if len(row) >= 3 and row[1].strip().upper() == "GRAV":
                    gravity = float(row[2])
                    break
            continue
        if key == "*output" and "time interval" in params:
            output_interval = float(params["time interval"])
        i += 1

    if "BEAM" not in parts or "SPHERE" not in parts:
        raise ValueError(f"input deck must contain BEAM and SPHERE parts: {inp_path}")
    beam_nodes, beam_elements = _part_to_arrays(parts["BEAM"])
    sphere_nodes, sphere_elements = _part_to_arrays(parts["SPHERE"])
    beam_material_name = str(parts["BEAM"].get("material", "BEAM_MAT")).upper()
    sphere_material_name = str(parts["SPHERE"].get("material", "SPHERE_MAT")).upper()
    beam_material = _material_from_dict(materials[beam_material_name])
    sphere_material = _material_from_dict(materials[sphere_material_name])
    sphere_surface_faces, sphere_surface_refs = _tet_boundary_faces(sphere_elements)
    sphere_center_z = float(np.mean(sphere_nodes[:, 2]))
    sphere_face_centroids = sphere_nodes[sphere_surface_faces].mean(axis=1)
    lower = sphere_face_centroids[:, 2] <= sphere_center_z
    sphere_sample_faces = sphere_surface_faces[lower]
    sphere_sample_areas = _triangle_areas(sphere_nodes, sphere_sample_faces)
    beam_top_faces, beam_top_refs = _beam_top_faces_and_refs(beam_nodes, beam_elements)
    return CommercialShortModel(
        source_inp=inp_path,
        duration=float(deck_duration if duration is None else duration),
        dt=float(deck_dt if dt is None else dt),
        initial_velocity_z=float(initial_velocity_z),
        gravity=float(gravity),
        beam_material=beam_material,
        sphere_material=sphere_material,
        beam_nodes=beam_nodes,
        beam_elements=beam_elements,
        beam_fixed_nodes=np.asarray([int(node) - 1 for node in nsets.get("BEAM_FIXED", [])], dtype=np.int64),
        sphere_nodes=sphere_nodes,
        sphere_elements=sphere_elements,
        sphere_surface_faces=sphere_surface_faces,
        sphere_surface_refs=sphere_surface_refs,
        sphere_sample_faces=sphere_sample_faces,
        sphere_sample_areas=sphere_sample_areas,
        beam_top_faces=beam_top_faces,
        beam_top_refs=beam_top_refs,
    )


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _material_from_dict(values: dict[str, float]) -> InpMaterial:
    return InpMaterial(
        young=float(values["young"]),
        poisson=float(values["poisson"]),
        density=float(values["density"]),
    )


def _tet_boundary_faces(elements: np.ndarray) -> tuple[np.ndarray, list[tuple[int, str]]]:
    local_faces = {
        "S1": (0, 2, 1),
        "S2": (0, 1, 3),
        "S3": (1, 2, 3),
        "S4": (0, 3, 2),
    }
    records: dict[tuple[int, int, int], list[tuple[int, str, tuple[int, int, int]]]] = {}
    for eid0, element in enumerate(np.asarray(elements, dtype=np.int64)):
        for label, local in local_faces.items():
            face = tuple(int(element[i]) for i in local)
            records.setdefault(tuple(sorted(face)), []).append((eid0 + 1, label, face))
    faces: list[tuple[int, int, int]] = []
    refs: list[tuple[int, str]] = []
    for entries in records.values():
        if len(entries) == 1:
            eid, label, face = entries[0]
            faces.append(face)
            refs.append((eid, label))
    return np.asarray(faces, dtype=np.int64), refs


def _beam_top_faces_and_refs(nodes: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, list[tuple[int, str]]]:
    faces: list[tuple[int, int, int]] = []
    refs: list[tuple[int, str]] = []
    z_max = float(np.max(np.asarray(nodes, dtype=float)[:, 2]))
    for eid0, element in enumerate(np.asarray(elements, dtype=np.int64)):
        quad = element[[4, 5, 6, 7]]
        if np.allclose(nodes[quad, 2], z_max):
            faces.append((int(quad[0]), int(quad[1]), int(quad[2])))
            faces.append((int(quad[0]), int(quad[2]), int(quad[3])))
            refs.append((eid0 + 1, "S2"))
    return np.asarray(faces, dtype=np.int64), refs


def _triangle_areas(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(nodes, dtype=float)[np.asarray(faces, dtype=np.int64)]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def build_short_model(
    *,
    duration: float | None,
    dt: float | None,
    inp_path: Path = DEFAULT_ABAQUS_INP,
) -> CommercialShortModel:
    """Build the short model by parsing the commercial Abaqus input deck."""

    duration_value = None if duration is None else float(duration)
    dt_value = None if dt is None else float(dt)
    return _read_abaqus_input_model(Path(inp_path), duration=duration_value, dt=dt_value)


def write_calculix_input(
    model: CommercialShortModel,
    path: Path,
    *,
    contact_stiffness: float,
    direct_dynamic: bool = False,
) -> None:
    """Write a CalculiX native-contact model for the same short impact case."""

    beam_ids = np.arange(1, model.beam_node_count + 1, dtype=np.int64)
    sphere_ids = np.arange(model.beam_node_count + 1, model.combined_node_count + 1, dtype=np.int64)
    beam_eids = np.arange(1, model.beam_elements.shape[0] + 1, dtype=np.int64)
    sphere_eids = np.arange(beam_eids[-1] + 1, beam_eids[-1] + 1 + model.sphere_elements.shape[0], dtype=np.int64)
    fixed_ids = beam_ids[model.beam_fixed_nodes]
    ccx_min_dt = max(1.0e-9, 1.0e-4 * float(model.dt))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "** Generated by validation/run_commercial_sphere_cantilever_short_comparison.py",
        "** External comparison: CalculiX native contact vs commercial Abaqus VTK reference.",
        "*NODE, NSET=NALL",
    ]
    for node_id, xyz in zip(beam_ids, model.beam_nodes, strict=True):
        lines.append(f"{int(node_id)}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    for node_id, xyz in zip(sphere_ids, model.sphere_nodes, strict=True):
        lines.append(f"{int(node_id)}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    lines.append("*ELEMENT, TYPE=C3D8R, ELSET=EBEAM")
    for eid, element in zip(beam_eids, model.beam_elements, strict=True):
        conn = ", ".join(str(int(beam_ids[idx])) for idx in element)
        lines.append(f"{int(eid)}, {conn}")
    lines.append("*ELEMENT, TYPE=C3D4, ELSET=ESPHERE")
    for eid, element in zip(sphere_eids, model.sphere_elements, strict=True):
        conn = ", ".join(str(int(sphere_ids[idx])) for idx in element)
        lines.append(f"{int(eid)}, {conn}")
    lines.extend(["*ELSET, ELSET=EALL", "EBEAM, ESPHERE", "*NSET, NSET=NBEAM"])
    _append_id_list(lines, beam_ids)
    lines.append("*NSET, NSET=NSPHERE")
    _append_id_list(lines, sphere_ids)
    lines.append("*NSET, NSET=NFIXED")
    _append_id_list(lines, fixed_ids)
    lines.extend(
        [
            "*MATERIAL, NAME=BEAM_MAT",
            "*ELASTIC",
            f"{model.beam_material.young:.12e}, {model.beam_material.poisson:.12e}",
            "*DENSITY",
            f"{model.beam_material.density:.12e}",
            "*MATERIAL, NAME=SPHERE_MAT",
            "*ELASTIC",
            f"{model.sphere_material.young:.12e}, {model.sphere_material.poisson:.12e}",
            "*DENSITY",
            f"{model.sphere_material.density:.12e}",
            "*SOLID SECTION, ELSET=EBEAM, MATERIAL=BEAM_MAT",
            "*SOLID SECTION, ELSET=ESPHERE, MATERIAL=SPHERE_MAT",
            "*BOUNDARY",
            "NFIXED, 1, 3, 0.0",
            "*INITIAL CONDITIONS, TYPE=VELOCITY",
            f"NSPHERE, 3, {model.initial_velocity_z:.12e}",
            "*SURFACE, NAME=SSPHERE, TYPE=ELEMENT",
        ]
    )
    for local_eid, face_label in model.sphere_surface_refs:
        lines.append(f"{int(sphere_eids[local_eid - 1])}, {face_label}")
    lines.append("*SURFACE, NAME=SBEAMTOP, TYPE=ELEMENT")
    for local_eid, face_label in model.beam_top_refs:
        lines.append(f"{int(beam_eids[local_eid - 1])}, {face_label}")
    lines.extend(
        [
            "*SURFACE INTERACTION, NAME=CONTACT",
            "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
            f"{float(contact_stiffness):.12e}",
            "*CONTACT PAIR, INTERACTION=CONTACT, TYPE=SURFACE TO SURFACE",
            "SSPHERE, SBEAMTOP",
            "*STEP, NLGEOM, INC=1000000",
            "*DYNAMIC, " + ("DIRECT, " if bool(direct_dynamic) else "") + "ALPHA=-0.05",
            f"{model.dt:.12e}, {model.duration:.12e}, {ccx_min_dt:.12e}, {model.dt:.12e}",
            "*DLOAD",
            f"EBEAM, GRAV, {model.gravity:.12e}, 0.0, 0.0, -1.0",
            f"ESPHERE, GRAV, {model.gravity:.12e}, 0.0, 0.0, -1.0",
            "*NODE PRINT, NSET=NALL, FREQUENCY=1",
            "U",
            "*CONTACT PRINT, FREQUENCY=1, TOTALS=ONLY",
            "CDIS, CSTR, CELS, CNUM",
            "*END STEP",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_calculix(
    model: CommercialShortModel,
    out_dir: Path,
    *,
    contact_stiffness: float,
    timeout_seconds: int,
    direct_dynamic: bool,
) -> tuple[list[Row], Row]:
    if not calculix_available():
        raise RuntimeError("CalculiX/ccx is unavailable through WSL")
    run_dir = out_dir / "calculix_runs" / JOB_NAME
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    inp_path = run_dir / f"{JOB_NAME}.inp"
    write_calculix_input(model, inp_path, contact_stiffness=contact_stiffness, direct_dynamic=direct_dynamic)
    command = f"cd {_wsl_path(run_dir)} && ccx {JOB_NAME}"
    start = time.perf_counter()
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=int(timeout_seconds),
    )
    wall = time.perf_counter() - start
    (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
    dat_path = run_dir / f"{JOB_NAME}.dat"
    command_row: Row = {
        "solver": "calculix",
        "command": command,
        "returncode": int(proc.returncode),
        "wall_time_seconds": float(wall),
        "external_solver_version": _calculix_version(),
        "direct_dynamic": str(bool(direct_dynamic)).lower(),
        "inp": str(inp_path.relative_to(out_dir)),
        "dat": str(dat_path.relative_to(out_dir)) if dat_path.exists() else "",
    }
    if proc.returncode != 0 or not dat_path.exists():
        raise RuntimeError(f"CalculiX failed; see {run_dir}")
    node_ids = np.arange(1, model.combined_node_count + 1, dtype=np.int64)
    displacement_blocks = _parse_calculix_dat_displacements(dat_path, node_ids)
    rows = _history_from_displacements(
        "calculix",
        model,
        displacement_blocks,
        sphere_slice=slice(model.beam_node_count, model.combined_node_count),
        beam_slice=slice(0, model.beam_node_count),
    )
    return rows, command_row


def _read_manifest_rows(manifest_path: Path, *, duration: float) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if float(row["time"]) <= float(duration) + 1.0e-12]


def _read_runtime_metrics(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="ascii", errors="ignore") as handle:
        return {str(row["metric"]): str(row["value"]) for row in csv.DictReader(handle)}


def _read_vtk_displacement(path: Path, *, node_count: int) -> np.ndarray:
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        normalized = line.strip().lower()
        if normalized in {"vectors displacement float", "vectors u float"}:
            values = []
            for offset in range(node_count):
                parts = lines[idx + 1 + offset].split()
                values.append((float(parts[0]), float(parts[1]), float(parts[2])))
            return np.asarray(values, dtype=float)
    raise ValueError(f"VTK file has no displacement vector field: {path}")


def _tip_node_indices(nodes: np.ndarray) -> np.ndarray:
    x_max = float(np.max(nodes[:, 0]))
    return np.flatnonzero(np.isclose(nodes[:, 0], x_max))


def _history_metrics(
    source: str,
    model: CommercialShortModel,
    time_value: float,
    combined_displacement: np.ndarray,
) -> Row:
    U = np.asarray(combined_displacement, dtype=float)
    beam_u = U[: model.beam_node_count]
    sphere_u = U[model.beam_node_count :]
    return {
        "source": source,
        "time": float(time_value),
        "sphere_mean_uz": float(np.mean(sphere_u[:, 2])),
        "sphere_center_z": float(np.mean(model.sphere_nodes[:, 2] + sphere_u[:, 2])),
        "beam_tip_mean_uz": float(np.mean(beam_u[_tip_node_indices(model.beam_nodes), 2])),
        "beam_max_abs_u": float(np.max(np.linalg.norm(beam_u, axis=1))),
    }


def _make_oracle(model: CommercialShortModel, beam_x_current: np.ndarray) -> LagrangianSDFContactOracle:
    material = MaterialSDF.from_triangle_surface(
        model.beam_nodes,
        model.beam_top_faces,
        band_radius=0.12,
    )
    return LagrangianSDFContactOracle(
        material=material,
        x_current=beam_x_current,
        search_radius=0.08,
        patch_cell_size=0.025,
        cache_enabled=True,
    )


def _sphere_samples(model: CommercialShortModel) -> list[SurfaceSample]:
    weights = np.full(3, 1.0 / 3.0, dtype=float)
    return [
        SurfaceSample(
            node_ids=np.asarray(face, dtype=np.int64),
            weights=weights.copy(),
            candidate_face_ids=np.empty(0, dtype=np.int64),
        )
        for face in model.sphere_sample_faces
    ]


def _structured_beam_top_grid(model: CommercialShortModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top_z = float(np.max(model.beam_nodes[:, 2]))
    top_ids = np.flatnonzero(np.isclose(model.beam_nodes[:, 2], top_z))
    xs = np.unique(np.round(model.beam_nodes[top_ids, 0], 14))
    ys = np.unique(np.round(model.beam_nodes[top_ids, 1], 14))
    node_grid = np.empty((ys.size, xs.size), dtype=np.int64)
    lookup = {
        (round(float(model.beam_nodes[node, 0]), 14), round(float(model.beam_nodes[node, 1]), 14)): int(node)
        for node in top_ids
    }
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            node_grid[j, i] = lookup[(round(float(x), 14), round(float(y), 14))]
    return xs.astype(float), ys.astype(float), node_grid


def _structured_top_contact_state(
    model: CommercialShortModel,
    u_value: np.ndarray,
    *,
    contact_stiffness: float,
    top_grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> ExplicitContactState:
    xs, ys, node_grid = top_grid
    n_total = 3 * model.sphere_node_count + 3 * model.beam_node_count
    force = np.zeros(n_total, dtype=float)
    sphere_u = u_value[: 3 * model.sphere_node_count].reshape((-1, 3))
    beam_u = u_value[3 * model.sphere_node_count :].reshape((-1, 3))
    sphere_x = model.sphere_nodes + sphere_u
    beam_x = model.beam_nodes + beam_u
    faces = np.asarray(model.sphere_sample_faces, dtype=np.int64)
    if faces.size == 0:
        return ExplicitContactState(force=force, min_gap=0.0, active_count=0)
    sample_points = sphere_x[faces].mean(axis=1)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    i0 = np.floor((sample_points[:, 0] - xs[0]) / dx).astype(np.int64)
    j0 = np.floor((sample_points[:, 1] - ys[0]) / dy).astype(np.int64)
    inside = (i0 >= 0) & (i0 < xs.size - 1) & (j0 >= 0) & (j0 < ys.size - 1)
    if not np.any(inside):
        return ExplicitContactState(force=force, min_gap=float("inf"), active_count=0)
    active_candidates = np.nonzero(inside)[0]
    i0 = i0[inside]
    j0 = j0[inside]
    p = sample_points[inside]
    tx = np.clip((p[:, 0] - xs[i0]) / dx, 0.0, 1.0)
    ty = np.clip((p[:, 1] - ys[j0]) / dy, 0.0, 1.0)
    n00 = node_grid[j0, i0]
    n10 = node_grid[j0, i0 + 1]
    n01 = node_grid[j0 + 1, i0]
    n11 = node_grid[j0 + 1, i0 + 1]
    p00 = beam_x[n00]
    p10 = beam_x[n10]
    p01 = beam_x[n01]
    p11 = beam_x[n11]
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty
    q = w00[:, None] * p00 + w10[:, None] * p10 + w01[:, None] * p01 + w11[:, None] * p11
    dqdx = ((1.0 - ty)[:, None] * (p10 - p00) + ty[:, None] * (p11 - p01)) / dx
    dqdy = ((1.0 - tx)[:, None] * (p01 - p00) + tx[:, None] * (p11 - p10)) / dy
    normals = np.cross(dqdx, dqdy)
    normal_norm = np.linalg.norm(normals, axis=1)
    valid_normal = normal_norm > 0.0
    normals[valid_normal] /= normal_norm[valid_normal, None]
    flip = normals[:, 2] < 0.0
    normals[flip] *= -1.0
    gaps = np.einsum("ij,ij->i", p - q, normals)
    min_gap = float(np.min(gaps)) if gaps.size else 0.0
    active_local = np.nonzero((gaps < 0.0) & valid_normal)[0]
    if active_local.size == 0:
        return ExplicitContactState(force=force, min_gap=min_gap, active_count=0)
    sample_ids = active_candidates[active_local]
    scales = float(contact_stiffness) * model.sphere_sample_areas[sample_ids] * (-gaps[active_local])
    contact_vectors = scales[:, None] * normals[active_local]
    for local_pos, sample_id in enumerate(sample_ids):
        slave_nodes = faces[int(sample_id)]
        f_slave = contact_vectors[local_pos] / 3.0
        for node in slave_nodes:
            force[3 * int(node) : 3 * int(node) + 3] += f_slave
    master_weights = (w00[active_local], w10[active_local], w01[active_local], w11[active_local])
    master_nodes = (n00[active_local], n10[active_local], n01[active_local], n11[active_local])
    master_offset = 3 * model.sphere_node_count
    for weights, nodes in zip(master_weights, master_nodes, strict=True):
        weighted = -weights[:, None] * contact_vectors
        for row, node in enumerate(nodes):
            base = master_offset + 3 * int(node)
            force[base : base + 3] += weighted[row]
    return ExplicitContactState(force=force, min_gap=min_gap, active_count=int(active_local.size))


def _append_gap_diagnostics(rows: list[Row], model: CommercialShortModel, *, contact_stiffness: float) -> None:
    samples = _sphere_samples(model)
    n_total = 3 * model.sphere_node_count + 3 * model.beam_node_count
    for row in rows:
        U = np.asarray(row.pop("_combined_u"), dtype=float)
        beam_u = U[: model.beam_node_count]
        sphere_u = U[model.beam_node_count :]
        oracle = _make_oracle(model, model.beam_nodes + beam_u)
        response = lagrangian_oracle_penalty_response(
            model.sphere_nodes + sphere_u,
            samples,
            oracle,
            pressure_stiffness=float(contact_stiffness),
            n_total_dofs=n_total,
            sample_area_weights=model.sphere_sample_areas,
            slave_dof_offset=0,
            master_dof_offset=3 * model.sphere_node_count,
        )
        row["min_gap"] = float(response.min_gap)
        row["active_samples"] = int(response.active_count)


def load_abaqus_reference_history(model: CommercialShortModel, abaqus_dir: Path, *, duration: float, contact_stiffness: float) -> list[Row]:
    """Load the commercial Abaqus VTK frames up to ``duration``."""

    manifest = abaqus_dir / "vtk" / "frame_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing Abaqus VTK manifest: {manifest}")
    rows: list[Row] = []
    for manifest_row in _read_manifest_rows(manifest, duration=duration):
        vtk_file = abaqus_dir / "vtk" / manifest_row["vtk_file"]
        U = _read_vtk_displacement(vtk_file, node_count=model.combined_node_count)
        row = _history_metrics("abaqus", model, float(manifest_row["time"]), U)
        row["max_von_mises"] = float(manifest_row.get("max_von_mises", 0.0))
        row["max_strain_norm"] = float(manifest_row.get("max_le_norm", 0.0))
        row["_combined_u"] = U
        row["min_gap"] = ""
        row["active_samples"] = ""
        rows.append(row)
    return rows


def _history_from_displacements(
    source: str,
    model: CommercialShortModel,
    displacements: dict[float, np.ndarray],
    *,
    sphere_slice: slice,
    beam_slice: slice,
) -> list[Row]:
    rows: list[Row] = []
    for time_value, U_all in sorted(displacements.items()):
        beam_u = np.asarray(U_all[beam_slice], dtype=float)
        sphere_u = np.asarray(U_all[sphere_slice], dtype=float)
        U = np.vstack((beam_u, sphere_u))
        row = _history_metrics(source, model, float(time_value), U)
        row["_combined_u"] = U
        rows.append(row)
    return rows


def _sfc_contact_response(
    model: CommercialShortModel,
    x_sphere: np.ndarray,
    x_beam: np.ndarray,
    *,
    contact_stiffness: float,
) -> Any:
    oracle = _make_oracle(model, x_beam)
    return lagrangian_oracle_penalty_response(
        x_sphere,
        _sphere_samples(model),
        oracle,
        pressure_stiffness=float(contact_stiffness),
        n_total_dofs=3 * model.sphere_node_count + 3 * model.beam_node_count,
        sample_area_weights=model.sphere_sample_areas,
        slave_dof_offset=0,
        master_dof_offset=3 * model.sphere_node_count,
    )


def _run_sfc_lagrangian_explicit_short(
    model: CommercialShortModel,
    *,
    contact_stiffness: float,
    contact_backend: str,
) -> tuple[list[Row], Row]:
    """Run the flexible-sphere/flexible-cantilever case with central difference."""

    dt = float(model.dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    steps_float = float(model.duration) / dt
    steps = int(round(steps_float))
    if not np.isclose(steps * dt, float(model.duration), rtol=0.0, atol=max(1.0e-12, 1.0e-9 * dt)):
        raise ValueError("explicit SFC run requires duration to be an integer multiple of dt")

    start = time.perf_counter()
    oracle: LagrangianSDFContactOracle | None = None
    if str(contact_backend) == "oracle":
        material = MaterialSDF.from_triangle_surface(
            model.beam_nodes,
            model.beam_top_faces,
            band_radius=0.12,
        )
        oracle = LagrangianSDFContactOracle(
            material=material,
            x_current=model.beam_nodes,
            search_radius=0.08,
            patch_cell_size=0.025,
            cache_enabled=True,
        )
    samples = _sphere_samples(model)
    top_grid = _structured_beam_top_grid(model)

    def contact_response(u_value: np.ndarray) -> Any:
        if str(contact_backend) == "structured-top":
            return _structured_top_contact_state(
                model,
                u_value,
                contact_stiffness=float(contact_stiffness),
                top_grid=top_grid,
            )
        if oracle is None:
            raise RuntimeError("oracle backend was requested but not initialized")
        sphere_u = u_value[: 3 * model.sphere_node_count].reshape((-1, 3))
        beam_u = u_value[3 * model.sphere_node_count :].reshape((-1, 3))
        oracle.refit(model.beam_nodes + beam_u)
        return lagrangian_oracle_penalty_response(
            model.sphere_nodes + sphere_u,
            samples,
            oracle,
            pressure_stiffness=float(contact_stiffness),
            n_total_dofs=3 * model.sphere_node_count + 3 * model.beam_node_count,
            sample_area_weights=model.sphere_sample_areas,
            slave_dof_offset=0,
            master_dof_offset=3 * model.sphere_node_count,
        )

    sphere_body = DeformableBody(
        mesh=VolumeMesh(model.sphere_nodes, model.sphere_elements, element_type="C3D4"),
        material={"E": model.sphere_material.young, "nu": model.sphere_material.poisson},
        density=model.sphere_material.density,
    )
    beam_body = DeformableBody(
        mesh=VolumeMesh(model.beam_nodes, model.beam_elements, element_type="C3D8"),
        material={"E": model.beam_material.young, "nu": model.beam_material.poisson},
        density=model.beam_material.density,
    )
    K = block_diag((assemble_stiffness_matrix(sphere_body), assemble_stiffness_matrix(beam_body)), format="csr")
    m_sphere = assemble_mass_matrix(sphere_body, kind="lumped").diagonal()
    m_beam = assemble_mass_matrix(beam_body, kind="lumped").diagonal()
    mass_diag = np.concatenate((np.asarray(m_sphere, dtype=float), np.asarray(m_beam, dtype=float)))
    if np.any(mass_diag <= 0.0):
        raise ValueError("explicit integration requires strictly positive lumped masses")
    f_ext = np.concatenate(
        (
            assemble_gravity_force(sphere_body, (0.0, 0.0, -model.gravity)),
            assemble_gravity_force(beam_body, (0.0, 0.0, -model.gravity)),
        )
    )
    n_dofs = int(K.shape[0])
    fixed = np.asarray(
        [3 * model.sphere_node_count + 3 * int(node) + axis for node in model.beam_fixed_nodes for axis in range(3)],
        dtype=np.int64,
    )
    u = np.zeros(n_dofs, dtype=float)
    v0 = np.zeros(n_dofs, dtype=float)
    v0[: 3 * model.sphere_node_count].reshape((-1, 3))[:, 2] = float(model.initial_velocity_z)
    v0[fixed] = 0.0
    rows: list[Row] = []

    def combined_displacement(u_value: np.ndarray) -> np.ndarray:
        return np.vstack(
            (
                u_value[3 * model.sphere_node_count :].reshape((-1, 3)),
                u_value[: 3 * model.sphere_node_count].reshape((-1, 3)),
            )
        )

    def append_output(step: int, response: Any) -> None:
        U = combined_displacement(u)
        row = _history_metrics("sfc_lagrangian_sdf", model, step * dt, U)
        row["min_gap"] = float(response.min_gap)
        row["active_samples"] = int(response.active_count)
        rows.append(row)

    response = contact_response(u)
    a = (f_ext + response.force - K @ u) / mass_diag
    a[fixed] = 0.0
    v_half = v0 - 0.5 * dt * a
    v_half[fixed] = 0.0
    for step in range(steps + 1):
        response = contact_response(u)
        append_output(step, response)
        if step == steps:
            break
        a = (f_ext + response.force - K @ u) / mass_diag
        a[fixed] = 0.0
        v_half += dt * a
        v_half[fixed] = 0.0
        u += dt * v_half
        u[fixed] = 0.0
        if not np.all(np.isfinite(u)):
            raise FloatingPointError("SFC explicit sphere-cantilever state became non-finite")

    wall = time.perf_counter() - start
    for row in rows:
        row["sfc_solve_wall_seconds"] = float(wall)
        row["time_integrator"] = "explicit"
        row["mass_matrix"] = "lumped"
        row["explicit_algorithm"] = "central_difference"
        row["contact_path"] = "MaterialSDF+LagrangianSDFContactOracle"
        row["sfc_contact_backend"] = str(contact_backend)
        row["contact_stiffness"] = float(contact_stiffness)
    return rows, {
        "solver": "sfc_lagrangian_sdf",
        "wall_time_seconds": float(wall),
        "contact_path": "MaterialSDF+LagrangianSDFContactOracle",
        "sfc_contact_backend": str(contact_backend),
        "time_integrator": "explicit",
        "mass_matrix": "lumped",
        "duration": float(model.duration),
        "dt": float(model.dt),
    }


def run_sfc_lagrangian_short(
    model: CommercialShortModel,
    *,
    contact_stiffness: float,
    damping_alpha: float,
    integrator: str = "explicit",
    contact_backend: str = "structured-top",
) -> tuple[list[Row], Row]:
    """Run an independent SFC Lagrangian-SDF short transient solve."""

    if str(integrator).lower() == "explicit":
        if abs(float(damping_alpha)) > 0.0:
            raise ValueError("explicit SFC sphere-cantilever run is undamped")
        if str(contact_backend) not in {"structured-top", "oracle"}:
            raise ValueError("contact_backend must be 'structured-top' or 'oracle'")
        return _run_sfc_lagrangian_explicit_short(
            model,
            contact_stiffness=contact_stiffness,
            contact_backend=str(contact_backend),
        )
    if str(integrator).lower() != "newmark":
        raise ValueError("integrator must be 'explicit' or 'newmark'")

    start = time.perf_counter()
    material = MaterialSDF.from_triangle_surface(
        model.beam_nodes,
        model.beam_top_faces,
        band_radius=0.12,
    )
    oracle = LagrangianSDFContactOracle(
        material=material,
        x_current=model.beam_nodes,
        search_radius=0.08,
        patch_cell_size=0.025,
        cache_enabled=True,
    )
    samples = _sphere_samples(model)

    def contact_response(x_sphere: np.ndarray, x_beam: np.ndarray) -> Any:
        oracle.refit(x_beam)
        return lagrangian_oracle_penalty_response(
            x_sphere,
            samples,
            oracle,
            pressure_stiffness=float(contact_stiffness),
            n_total_dofs=3 * model.sphere_node_count + 3 * model.beam_node_count,
            sample_area_weights=model.sphere_sample_areas,
            slave_dof_offset=0,
            master_dof_offset=3 * model.sphere_node_count,
        )

    sphere_body = DeformableBody(
        mesh=VolumeMesh(model.sphere_nodes, model.sphere_elements, element_type="C3D4"),
        material={"E": model.sphere_material.young, "nu": model.sphere_material.poisson},
        density=model.sphere_material.density,
    )
    beam_body = DeformableBody(
        mesh=VolumeMesh(model.beam_nodes, model.beam_elements, element_type="C3D8"),
        material={"E": model.beam_material.young, "nu": model.beam_material.poisson},
        density=model.beam_material.density,
    )
    K = block_diag((assemble_stiffness_matrix(sphere_body), assemble_stiffness_matrix(beam_body)), format="csr")
    M = block_diag((assemble_mass_matrix(sphere_body, kind="consistent"), assemble_mass_matrix(beam_body, kind="consistent")), format="csr")
    f_ext = np.concatenate(
        (
            assemble_gravity_force(sphere_body, (0.0, 0.0, -model.gravity)),
            assemble_gravity_force(beam_body, (0.0, 0.0, -model.gravity)),
        )
    )
    n_dofs = int(K.shape[0])
    fixed = np.asarray(
        [3 * model.sphere_node_count + 3 * int(node) + axis for node in model.beam_fixed_nodes for axis in range(3)],
        dtype=np.int64,
    )
    free = np.setdiff1d(np.arange(n_dofs, dtype=np.int64), fixed)
    beta = 0.25
    gamma = 0.5
    dt = float(model.dt)
    steps = int(round(float(model.duration) / dt))
    c0 = 1.0 / (beta * dt * dt)
    cv = gamma * dt * c0
    C = float(damping_alpha) * M
    u = np.zeros(n_dofs, dtype=float)
    v = np.zeros(n_dofs, dtype=float)
    v[: 3 * model.sphere_node_count].reshape((-1, 3))[:, 2] = float(model.initial_velocity_z)
    v[fixed] = 0.0
    response = contact_response(model.sphere_nodes, model.beam_nodes)
    rhs0 = f_ext + response.force - K @ u - C @ v
    a = np.zeros(n_dofs, dtype=float)
    a[free] = np.asarray(spsolve(M[free[:, None], free].tocsc(), rhs0[free]), dtype=float)
    rows: list[Row] = []
    for step in range(steps + 1):
        U = np.vstack(
            (
                u[3 * model.sphere_node_count :].reshape((-1, 3)),
                u[: 3 * model.sphere_node_count].reshape((-1, 3)),
            )
        )
        row = _history_metrics("sfc_lagrangian_sdf", model, step * dt, U)
        contact_now = contact_response(model.sphere_nodes + U[model.beam_node_count :], model.beam_nodes + U[: model.beam_node_count])
        row["min_gap"] = float(contact_now.min_gap)
        row["active_samples"] = int(contact_now.active_count)
        rows.append(row)
        if step == steps:
            break
        u_pred = u + dt * v + dt * dt * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a
        u_pred[fixed] = 0.0
        v_pred[fixed] = 0.0
        u_guess = u_pred.copy()
        for _iteration in range(5):
            sphere_u = u_guess[: 3 * model.sphere_node_count].reshape((-1, 3))
            beam_u = u_guess[3 * model.sphere_node_count :].reshape((-1, 3))
            contact = contact_response(model.sphere_nodes + sphere_u, model.beam_nodes + beam_u)
            a_guess = c0 * (u_guess - u_pred)
            v_guess = v_pred + gamma * dt * a_guess
            residual = M @ a_guess + C @ v_guess + K @ u_guess - f_ext - contact.force
            tangent = (M * (c0 + float(damping_alpha) * cv) + K + contact.stiffness).tocsc()
            correction = np.zeros(n_dofs, dtype=float)
            correction[free] = np.asarray(spsolve(tangent[free[:, None], free], -residual[free]), dtype=float)
            u_guess[free] += correction[free]
            u_guess[fixed] = 0.0
            if np.linalg.norm(correction[free]) <= 1.0e-9 * max(1.0, np.linalg.norm(u_guess[free])):
                break
        u = u_guess
        a = c0 * (u - u_pred)
        v = v_pred + gamma * dt * a
        u[fixed] = 0.0
        v[fixed] = 0.0
        a[fixed] = 0.0
    wall = time.perf_counter() - start
    return rows, {
        "solver": "sfc_lagrangian_sdf",
        "wall_time_seconds": float(wall),
        "contact_path": "MaterialSDF+LagrangianSDFContactOracle",
        "duration": float(model.duration),
        "dt": float(model.dt),
    }


def _series(rows: list[Row], key: str) -> tuple[np.ndarray, np.ndarray]:
    t_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in {"", None}:
            continue
        try:
            y = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(y):
            continue
        t_values.append(float(row["time"]))
        y_values.append(y)
    return np.asarray(t_values, dtype=float), np.asarray(y_values, dtype=float)


def _metric_against_reference(source_rows: list[Row], ref_rows: list[Row], source: str, key: str) -> Row:
    tref, yref = _series(ref_rows, key)
    t, y = _series(source_rows, key)
    if t.size == 0 or tref.size == 0:
        return {"source": source, "metric": key, "l2_relative_error": "", "max_abs_error": "", "status": "missing"}
    y_interp = np.interp(tref, t, y)
    denom = max(float(np.linalg.norm(yref)), 1.0e-14)
    err = y_interp - yref
    return {
        "source": source,
        "metric": key,
        "l2_relative_error": float(np.linalg.norm(err) / denom),
        "max_abs_error": float(np.max(np.abs(err))),
        "status": "computed",
    }


def _write_plots(out_dir: Path, rows_by_source: dict[str, list[Row]], metrics: list[Row]) -> list[Row]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metric_lookup = {(str(row["source"]), str(row["metric"])): row for row in metrics}
    plot_specs = [
        ("sphere_mean_uz", "sphere mean $u_z$ (m)", "commercial_short_sphere_uz"),
        ("beam_tip_mean_uz", "cantilever tip mean $u_z$ (m)", "commercial_short_beam_tip_uz"),
        ("min_gap", "oracle diagnostic min gap (m)", "commercial_short_min_gap"),
    ]
    plot_rows: list[Row] = []
    for key, ylabel, stem in plot_specs:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for source, rows in rows_by_source.items():
            t, y = _series(rows, key)
            if t.size == 0:
                continue
            label = source
            if source != "abaqus":
                row = metric_lookup.get((source, key))
                if row and row["l2_relative_error"] != "":
                    label = f"{source} (L2 err.={float(row['l2_relative_error']):.2e})"
            ax.plot(t, y, linewidth=1.4, label=label)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.legend(loc="best", fontsize=8, frameon=True)
        fig.tight_layout()
        png = figures / f"{stem}.png"
        pdf = figures / f"{stem}.pdf"
        fig.savefig(png, dpi=220)
        fig.savefig(pdf)
        plt.close(fig)
        plot_rows.append({"plot": stem, "png": str(png.relative_to(out_dir)), "pdf": str(pdf.relative_to(out_dir))})
    return plot_rows


def _write_summary(
    out_dir: Path,
    *,
    duration: float,
    dt: float,
    inp_path: Path,
    metrics: list[Row],
    command_rows: list[Row],
    plot_rows: list[Row],
    abaqus_dir: Path,
) -> None:
    lines = [
        "# Commercial Sphere-Cantilever Short Comparison",
        "",
        "This short-run package compares the existing Abaqus/Explicit sphere-cantilever reference against CalculiX native contact and SFC Lagrangian/material-SDF contact.",
        "",
        "## Scope",
        "",
        f"- Duration: `{duration}` s.",
        f"- SFC integration increment: `{dt}` s.",
        f"- Authoritative model input deck: `{inp_path}`.",
        f"- Commercial reference directory: `{abaqus_dir}`.",
        "- Abaqus is used only through pre-existing VTK reference frames; it is not a core solver dependency.",
        "- CalculiX is used only as an external native-contact reference.",
        "- SFC path: `MaterialSDF + LagrangianSDFContactOracle`; the explicit short run uses the structured top-patch fast path for the regular cantilever surface.",
        "",
        "## Error Metrics vs Abaqus",
        "",
        "| Source | Metric | L2 relative error | Max absolute error | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['source']} | {row['metric']} | {row['l2_relative_error']} | {row['max_abs_error']} | {row['status']} |"
        )
    lines.extend(["", "## Commands/Timing", "", "| Solver | Wall time (s) | Details |", "|---|---:|---|"])
    for row in command_rows:
        lines.append(f"| {row['solver']} | {row.get('wall_time_seconds', '')} | {row} |")
    lines.extend(["", "## Figures", ""])
    for row in plot_rows:
        lines.append(f"- `{row['png']}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"This is a short {duration:g} s alignment check, not the final 3 s paper run. The comparison uses curve errors rather than bar charts. The SFC timing reports the standalone solver path only; Abaqus timing reports the external analysis wall time when available.",
            "",
        ]
    )
    (out_dir / "commercial_short_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_short_comparison(
    out_dir: Path,
    *,
    abaqus_dir: Path,
    inp_path: Path,
    duration: float = 0.05,
    dt: float | None = None,
    contact_stiffness: float = 5.0e9,
    damping_alpha: float = 0.0,
    skip_calculix: bool = False,
    skip_sfc: bool = False,
    calculix_timeout_seconds: int = 600,
    calculix_direct_dynamic: bool = False,
    sfc_integrator: str = "explicit",
    sfc_contact_backend: str = "structured-top",
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_short_model(duration=duration, dt=dt, inp_path=inp_path)
    dt = float(model.dt)
    abaqus_rows = load_abaqus_reference_history(model, abaqus_dir, duration=duration, contact_stiffness=contact_stiffness)
    abaqus_runtime = _read_runtime_metrics(abaqus_dir / "runtime_metrics.csv")
    rows_by_source: dict[str, list[Row]] = {"abaqus": abaqus_rows}
    command_rows: list[Row] = [
        {
            "solver": "abaqus",
            "wall_time_seconds": abaqus_runtime.get("abaqus_analysis_wall_seconds", ""),
            "total_workflow_wall_seconds": abaqus_runtime.get("total_workflow_wall_seconds", ""),
            "fixed_dt_seconds": abaqus_runtime.get("fixed_dt_seconds", ""),
            "details": "commercial VTK reference frames loaded from validation-only output directory",
        }
    ]
    if not skip_calculix:
        calculix_rows, command_row = _run_calculix(
            model,
            out_dir,
            contact_stiffness=contact_stiffness,
            timeout_seconds=calculix_timeout_seconds,
            direct_dynamic=calculix_direct_dynamic,
        )
        for row in calculix_rows:
            row["min_gap"] = ""
            row["active_samples"] = ""
        rows_by_source["calculix"] = calculix_rows
        command_rows.append(command_row)
    if not skip_sfc:
        sfc_rows, command_row = run_sfc_lagrangian_short(
            model,
            contact_stiffness=contact_stiffness,
            damping_alpha=damping_alpha,
            integrator=sfc_integrator,
            contact_backend=sfc_contact_backend,
        )
        rows_by_source["sfc_lagrangian_sdf"] = sfc_rows
        command_rows.append(command_row)

    history_rows: list[Row] = []
    for rows in rows_by_source.values():
        history_rows.extend(rows)
    metrics: list[Row] = []
    for source, rows in rows_by_source.items():
        if source == "abaqus":
            continue
        for metric in ("sphere_mean_uz", "beam_tip_mean_uz", "min_gap"):
            metrics.append(_metric_against_reference(rows, abaqus_rows, source, metric))
    plot_rows = _write_plots(out_dir, rows_by_source, metrics)
    _write_csv(out_dir / "commercial_short_time_history.csv", history_rows)
    _write_csv(out_dir / "commercial_short_metrics.csv", metrics)
    _write_csv(out_dir / "commercial_short_commands.csv", command_rows)
    _write_csv(out_dir / "commercial_short_plots.csv", plot_rows)
    _write_summary(
        out_dir,
        duration=duration,
        dt=dt,
        inp_path=inp_path,
        metrics=metrics,
        command_rows=command_rows,
        plot_rows=plot_rows,
        abaqus_dir=abaqus_dir,
    )
    return {
        "history": out_dir / "commercial_short_time_history.csv",
        "metrics": out_dir / "commercial_short_metrics.csv",
        "commands": out_dir / "commercial_short_commands.csv",
        "plots": out_dir / "commercial_short_plots.csv",
        "summary": out_dir / "commercial_short_summary.md",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "commercial_sphere_cantilever_short")
    parser.add_argument("--abaqus-dir", type=Path, default=DEFAULT_ABAQUS_DIR)
    parser.add_argument("--inp", type=Path, default=DEFAULT_ABAQUS_INP, help="Abaqus input deck used as the authoritative model source.")
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=None, help="SFC integration increment; defaults to the direct Explicit increment parsed from the input deck.")
    parser.add_argument("--contact-stiffness", type=float, default=5.0e9)
    parser.add_argument("--damping-alpha", type=float, default=0.0)
    parser.add_argument("--skip-calculix", action="store_true")
    parser.add_argument("--skip-sfc", action="store_true")
    parser.add_argument("--calculix-timeout-seconds", type=int, default=600)
    parser.add_argument("--calculix-direct-dynamic", action="store_true", help="Use fixed direct increments; default uses CalculiX automatic increments.")
    parser.add_argument("--sfc-integrator", choices=("explicit", "newmark"), default="explicit")
    parser.add_argument("--sfc-contact-backend", choices=("structured-top", "oracle"), default="structured-top")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_short_comparison(
        args.out_dir,
        abaqus_dir=args.abaqus_dir,
        inp_path=args.inp,
        duration=float(args.duration),
        dt=None if args.dt is None else float(args.dt),
        contact_stiffness=float(args.contact_stiffness),
        damping_alpha=float(args.damping_alpha),
        skip_calculix=bool(args.skip_calculix),
        skip_sfc=bool(args.skip_sfc),
        calculix_timeout_seconds=int(args.calculix_timeout_seconds),
        calculix_direct_dynamic=bool(args.calculix_direct_dynamic),
        sfc_integrator=str(args.sfc_integrator),
        sfc_contact_backend=str(args.sfc_contact_backend),
    )
    print("Commercial sphere-cantilever short comparison complete.")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
