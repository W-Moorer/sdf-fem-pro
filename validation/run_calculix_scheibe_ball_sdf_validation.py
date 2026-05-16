"""Official CalculiX scheibe2f2f and ball SDF validation.

This runner adds two paper-facing external checks:

* ``scheibe2f2f``: parse CalculiX FRD displacement/contact blocks and replay
  the C3D8 nonlinear-static face-to-face contact state with SFC's current
  surface SDF query.
* ``ball``: run a native SFC C3D8 dynamic drop against a rigid plane and
  compare the time history against the official CalculiX ball example.

The runner is deliberately claim-gated.  Passing the scheibe replay supports
deformed-state gap/pressure correctness.  The ball case only supports native
trajectory equivalence when the generated CSV metrics meet the thresholds.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sfc.fem.assembler import assemble_gravity_force, assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.fem.body import DeformableBody  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.sdf.dynamic_surface_sdf import dynamic_surface_sdf  # noqa: E402
from validation.run_c3d8_contact_trajectory_validation import (  # noqa: E402
    C3D8_FACE_NODES,
    _orient_quad_outward,
    _parse_contact_elements,
    _parse_nodal_vectors,
    _parse_totals,
    _triangles_from_quads,
)
from validation.run_calculix_official_contact_examples import (  # noqa: E402
    DEFAULT_WSL_EXAMPLES_DIR,
    _calculix_available,
    _calculix_version,
    _read_example,
    _run,
    _wsl_path,
    _wsl_quote,
)

Row = dict[str, Any]

NUMBER = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?"
FRD_ROW = re.compile(
    r"^\s*-1\s*(\d+)\s*"
    + r"("
    + NUMBER
    + r")\s*("
    + NUMBER
    + r")\s*("
    + NUMBER
    + r")\s*("
    + NUMBER
    + r")?\s*("
    + NUMBER
    + r")?\s*("
    + NUMBER
    + r")?"
)


@dataclass
class ParsedInput:
    node_ids: np.ndarray
    X: np.ndarray
    elements_by_type: dict[str, np.ndarray]
    element_ids_by_type: dict[str, np.ndarray]
    nsets: dict[str, list[int]]
    elsets: dict[str, list[int]]
    surfaces: dict[str, dict[str, Any]]
    contact_slave_surface: str = ""
    contact_master_surface: str = ""
    pressure_law: str = ""
    pressure_values: list[float] = field(default_factory=list)
    material_E: float = 0.0
    material_nu: float = 0.0
    density: float = 1.0
    initial_velocity: np.ndarray | None = None
    gravity: float = 0.0
    gravity_direction: np.ndarray = field(default_factory=lambda: np.asarray([0.0, 0.0, -1.0], dtype=float))

    @property
    def id_to_index(self) -> dict[int, int]:
        return {int(node_id): idx for idx, node_id in enumerate(self.node_ids)}

    @property
    def c3d8_elements(self) -> np.ndarray:
        return self.elements_by_type.get("C3D8", np.empty((0, 8), dtype=np.int64))

    @property
    def c3d8_element_ids(self) -> np.ndarray:
        return self.element_ids_by_type.get("C3D8", np.empty(0, dtype=np.int64))


@dataclass
class FrdResults:
    nodes: dict[int, np.ndarray]
    elements: dict[int, list[int]]
    displacements: dict[int, np.ndarray]
    contact: list[Row]


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bool_text(value: bool) -> str:
    return str(bool(value)).lower()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _parse_keyword(line: str) -> tuple[str, dict[str, str], set[str]]:
    parts = [part.strip() for part in line.split(",")]
    name = parts[0].upper()
    params: dict[str, str] = {}
    flags: set[str] = set()
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip().upper()] = value.strip()
        elif part:
            flags.add(part.strip().upper())
    return name, params, flags


def _values(line: str) -> list[str]:
    return [value.strip() for value in line.split(",") if value.strip()]


def _name(value: str) -> str:
    return value.strip().upper()


def _extend_generated(target: list[int], values: list[str]) -> None:
    if len(values) >= 2:
        start = int(values[0])
        stop = int(values[1])
        step = int(values[2]) if len(values) >= 3 else 1
        target.extend(range(start, stop + (1 if step > 0 else -1), step))


def parse_calculix_input(text: str) -> ParsedInput:
    """Parse the small CalculiX input subset used by the official examples."""

    nodes: dict[int, list[float]] = {}
    elements: dict[str, dict[int, list[int]]] = {}
    nsets: dict[str, list[int]] = {}
    elsets: dict[str, list[int]] = {}
    surfaces: dict[str, dict[str, Any]] = {}
    contact_slave_surface = ""
    contact_master_surface = ""
    pressure_law = ""
    pressure_values: list[float] = []
    material_E = 0.0
    material_nu = 0.0
    density = 1.0
    velocity_specs: list[tuple[str, int, float]] = []
    gravity = 0.0
    gravity_direction = np.asarray([0.0, 0.0, -1.0], dtype=float)

    section = ""
    params: dict[str, str] = {}
    flags: set[str] = set()
    current_element_type = ""
    current_nset = ""
    current_elset = ""
    current_surface = ""
    expect_contact_pair = False
    expect_pressure_values = False
    expect_elastic = False
    expect_density = False
    current_element_generate = False
    current_nset_generate = False
    current_elset_generate = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            section, params, flags = _parse_keyword(line)
            current_nset = ""
            current_elset = ""
            current_surface = ""
            current_element_generate = "GENERATE" in flags
            current_nset_generate = "GENERATE" in flags
            current_elset_generate = "GENERATE" in flags
            if section == "*NODE":
                current_nset = _name(params.get("NSET", ""))
            elif section == "*ELEMENT":
                current_element_type = params.get("TYPE", "").upper()
                current_elset = _name(params.get("ELSET", ""))
                elements.setdefault(current_element_type, {})
            elif section == "*NSET":
                current_nset = _name(params.get("NSET", ""))
                nsets.setdefault(current_nset, [])
            elif section == "*ELSET":
                current_elset = _name(params.get("ELSET", ""))
                elsets.setdefault(current_elset, [])
            elif section == "*SURFACE":
                current_surface = _name(params.get("NAME", ""))
                surfaces.setdefault(current_surface, {"type": params.get("TYPE", "ELEMENT").upper(), "entries": []})
            elif section == "*CONTACT PAIR":
                expect_contact_pair = True
            elif section == "*SURFACE BEHAVIOR":
                pressure_law = params.get("PRESSURE-OVERCLOSURE", "").upper()
                expect_pressure_values = True
            elif section == "*ELASTIC":
                expect_elastic = True
            elif section == "*DENSITY":
                expect_density = True
            continue

        vals = _values(line)
        if not vals:
            continue
        if expect_contact_pair:
            contact_slave_surface = _name(vals[0])
            contact_master_surface = _name(vals[1])
            expect_contact_pair = False
            continue
        if expect_pressure_values:
            pressure_values = [float(value) for value in vals]
            expect_pressure_values = False
            continue
        if expect_elastic:
            material_E = float(vals[0])
            material_nu = float(vals[1])
            expect_elastic = False
            continue
        if expect_density:
            density = float(vals[0])
            expect_density = False
            continue

        if section == "*NODE":
            node_id = int(vals[0])
            nodes[node_id] = [float(vals[1]), float(vals[2]), float(vals[3])]
            if current_nset:
                nsets.setdefault(current_nset, []).append(node_id)
        elif section == "*ELEMENT":
            element_id = int(vals[0])
            elements.setdefault(current_element_type, {})[element_id] = [int(value) for value in vals[1:]]
            if current_elset:
                elsets.setdefault(current_elset, []).append(element_id)
        elif section == "*NSET":
            if current_nset_generate:
                _extend_generated(nsets.setdefault(current_nset, []), vals)
            else:
                nsets.setdefault(current_nset, []).extend(int(value) for value in vals)
        elif section == "*ELSET":
            if current_elset_generate:
                _extend_generated(elsets.setdefault(current_elset, []), vals)
            else:
                elsets.setdefault(current_elset, []).extend(int(value) for value in vals)
        elif section == "*SURFACE":
            surfaces[current_surface]["entries"].append(vals)
        elif section == "*INITIAL CONDITIONS" and params.get("TYPE", "").upper() == "VELOCITY":
            velocity_specs.append((_name(vals[0]), int(vals[1]), float(vals[2])))
        elif section == "*DLOAD" and len(vals) >= 6 and vals[1].upper() == "GRAV":
            gravity = float(vals[2])
            direction = np.asarray([float(vals[3]), float(vals[4]), float(vals[5])], dtype=float)
            norm = float(np.linalg.norm(direction))
            gravity_direction = direction / norm if norm > 0.0 else gravity_direction

    node_ids = np.asarray(sorted(nodes), dtype=np.int64)
    id_to_index = {int(node_id): idx for idx, node_id in enumerate(node_ids)}
    X = np.asarray([nodes[int(node_id)] for node_id in node_ids], dtype=float)
    elements_by_type: dict[str, np.ndarray] = {}
    element_ids_by_type: dict[str, np.ndarray] = {}
    for etype, data in elements.items():
        element_ids = np.asarray(sorted(data), dtype=np.int64)
        if element_ids.size:
            elements_by_type[etype] = np.asarray([[id_to_index[int(node_id)] for node_id in data[int(eid)]] for eid in element_ids], dtype=np.int64)
        else:
            elements_by_type[etype] = np.empty((0, 0), dtype=np.int64)
        element_ids_by_type[etype] = element_ids

    initial_velocity = np.zeros_like(X)
    for set_name, dof, value in velocity_specs:
        node_list = nsets.get(_name(set_name), [])
        for node_id in node_list:
            if int(node_id) in id_to_index:
                initial_velocity[id_to_index[int(node_id)], int(dof) - 1] = float(value)

    return ParsedInput(
        node_ids=node_ids,
        X=X,
        elements_by_type=elements_by_type,
        element_ids_by_type=element_ids_by_type,
        nsets=nsets,
        elsets=elsets,
        surfaces=surfaces,
        contact_slave_surface=contact_slave_surface,
        contact_master_surface=contact_master_surface,
        pressure_law=pressure_law,
        pressure_values=pressure_values,
        material_E=material_E,
        material_nu=material_nu,
        density=density,
        initial_velocity=initial_velocity,
        gravity=gravity,
        gravity_direction=gravity_direction,
    )


def parse_frd(text: str) -> FrdResults:
    """Parse nodes, C3D8 connectivity, DISP, and CONTACT blocks from FRD."""

    nodes: dict[int, np.ndarray] = {}
    elements: dict[int, list[int]] = {}
    displacements: dict[int, np.ndarray] = {}
    contact: list[Row] = []
    mode: str | None = None
    current_element: int | None = None
    current_component_labels: list[str] = []

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("2C"):
            mode = "nodes"
            continue
        if stripped.startswith("3C"):
            mode = "elements"
            continue
        if stripped.startswith("100CL"):
            mode = "result_header"
            current_component_labels = []
            continue
        if stripped.startswith("-4"):
            if "DISP" in stripped:
                mode = "disp"
            elif "CONTACT" in stripped:
                mode = "contact"
            else:
                mode = None
            current_component_labels = []
            continue
        if stripped.startswith("-5") and mode in {"disp", "contact"}:
            parts = stripped.split()
            if len(parts) >= 2:
                current_component_labels.append(parts[1].upper())
            continue
        if stripped.startswith("-3"):
            mode = None
            current_element = None
            continue

        if mode == "nodes" and stripped.startswith("-1"):
            match = FRD_ROW.match(line)
            if match:
                nodes[int(match.group(1))] = np.asarray([float(match.group(2)), float(match.group(3)), float(match.group(4))], dtype=float)
        elif mode == "elements":
            if stripped.startswith("-1"):
                parts = stripped.split()
                current_element = int(parts[1]) if len(parts) >= 2 else None
            elif stripped.startswith("-2") and current_element is not None:
                parts = stripped.split()
                elements[current_element] = [int(value) for value in parts[1:]]
        elif mode == "disp" and stripped.startswith("-1"):
            match = FRD_ROW.match(line)
            if match:
                displacements[int(match.group(1))] = np.asarray([float(match.group(2)), float(match.group(3)), float(match.group(4))], dtype=float)
        elif mode == "contact" and stripped.startswith("-1"):
            match = FRD_ROW.match(line)
            if match:
                values = [float(group) for group in match.groups()[1:] if group is not None]
                label_values = {label.lower(): values[i] for i, label in enumerate(current_component_labels[: len(values)])}
                contact.append({"node_id": int(match.group(1)), **label_values})

    return FrdResults(nodes=nodes, elements=elements, displacements=displacements, contact=contact)


def _copy_or_read_example(file_name: str, examples_dir: Path | None, wsl_examples_dir: str) -> tuple[str, str]:
    return _read_example(file_name, examples_dir, wsl_examples_dir)


def _run_calculix_case(input_text: str, run_dir: Path, case_name: str, *, timeout: int) -> tuple[bool, Path, Path, Row]:
    run_dir.mkdir(parents=True, exist_ok=True)
    inp_path = run_dir / f"{case_name}.inp"
    inp_path.write_text(input_text, encoding="utf-8")
    command = f"cd {_wsl_quote(_wsl_path(run_dir))} && ccx {_wsl_quote(case_name)}"
    start = time.perf_counter()
    proc = _run(["wsl", "--exec", "bash", "-lc", command], cwd=ROOT, timeout=timeout)
    wall = time.perf_counter() - start
    (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
    row = {
        "case_id": case_name,
        "external_solver": "CalculiX",
        "external_solver_version": _calculix_version(),
        "completed": _bool_text(proc.returncode == 0),
        "return_code": proc.returncode,
        "wall_time_seconds": wall,
        "command": f"wsl --exec bash -lc \"cd {_wsl_path(run_dir)} && ccx {case_name}\"",
        "input_file": _display_path(inp_path),
    }
    return proc.returncode == 0, inp_path, run_dir / f"{case_name}.frd", row


def _surface_quads(model: ParsedInput, surface_name: str) -> list[tuple[np.ndarray, np.ndarray]]:
    elmap = {int(eid): idx for idx, eid in enumerate(model.c3d8_element_ids)}
    quads: list[tuple[np.ndarray, np.ndarray]] = []
    for entry in model.surfaces.get(_name(surface_name), {}).get("entries", []):
        if not entry:
            continue
        name = entry[0]
        face = entry[1].upper() if len(entry) > 1 else ""
        element_ids = model.elsets.get(_name(name), [int(name)] if name.lstrip("+-").isdigit() else [])
        for element_id in element_ids:
            if int(element_id) not in elmap or face not in C3D8_FACE_NODES:
                continue
            element = model.c3d8_elements[elmap[int(element_id)]]
            quads.append((element[C3D8_FACE_NODES[face]], element))
    return quads


def _pressure_from_gap(gap: float, law: str, values: list[float]) -> float:
    overclosure = max(-float(gap), 0.0)
    if overclosure <= 0.0:
        return 0.0
    if law.upper() == "LINEAR":
        stiffness = float(values[0]) if values else 0.0
        tension_offset = float(values[1]) if len(values) > 1 else 0.0
        return max(stiffness * overclosure - tension_offset, 0.0)
    if law.upper() == "EXPONENTIAL":
        p0 = float(values[0])
        c0 = float(values[1])
        return float(p0 * np.exp(overclosure * np.log(100.0) / max(c0, 1.0e-30)))
    return 0.0


def _relative_error(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1.0e-30)


def _relative_vector_error(values: list[float] | np.ndarray, references: list[float] | np.ndarray) -> float:
    a = np.asarray(values, dtype=float)
    b = np.asarray(references, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-30))


def _scheibe_existing_paths() -> tuple[Path, Path]:
    base = ROOT / "paper" / "numerical_experiments" / "calculix_official_contact_examples"
    return (
        base / "inputs" / "scheibe2f2f.inp",
        base / "calculix_runs" / "scheibe2f2f_c3d8_nlgeom_static" / "scheibe2f2f.frd",
    )


def _run_scheibe(args: argparse.Namespace, out_dir: Path) -> tuple[Row, list[Row], Row, list[Row]]:
    case_id = "scheibe2f2f_c3d8_nonlinear_static_sdf_replay"
    run_dir = out_dir / "calculix_runs" / case_id
    command_row: Row = {"case_id": case_id, "command": "existing locked CalculiX FRD", "completed": "true"}

    existing_inp, existing_frd = _scheibe_existing_paths()
    if args.skip_calculix or not _calculix_available():
        input_text = existing_inp.read_text(encoding="utf-8", errors="ignore")
        frd_text = existing_frd.read_text(encoding="utf-8", errors="ignore")
        input_source = _display_path(existing_inp)
        frd_source = _display_path(existing_frd)
    else:
        input_text, input_source = _copy_or_read_example("scheibe2f2f.inp.gz", args.examples_dir, args.wsl_examples_dir)
        completed, inp_path, frd_path, run_command = _run_calculix_case(input_text, run_dir, "scheibe2f2f", timeout=240)
        command_row = run_command
        if not completed or not frd_path.exists():
            input_text = existing_inp.read_text(encoding="utf-8", errors="ignore")
            frd_path = existing_frd
            command_row["completed"] = "false_existing_fallback"
        frd_text = frd_path.read_text(encoding="utf-8", errors="ignore")
        input_source = _display_path(inp_path)
        frd_source = _display_path(frd_path)

    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs" / "scheibe2f2f.inp").write_text(input_text, encoding="utf-8")
    model = parse_calculix_input(input_text)
    frd = parse_frd(frd_text)
    U = np.zeros_like(model.X)
    id_to_index = model.id_to_index
    for node_id, value in frd.displacements.items():
        if int(node_id) in id_to_index:
            U[id_to_index[int(node_id)]] = value
    X_current = model.X + U
    master_quads = _surface_quads(model, model.contact_master_surface)
    master_faces = _triangles_from_quads(X_current, master_quads)
    candidates = np.arange(master_faces.shape[0], dtype=np.int64)

    contact_rows: list[Row] = []
    gap_errors: list[float] = []
    pressure_errors: list[float] = []
    pressure_rel_errors: list[float] = []
    for row in frd.contact:
        node_id = int(row["node_id"])
        if node_id not in id_to_index:
            continue
        query = X_current[id_to_index[node_id]]
        result = dynamic_surface_sdf(query, X_current, master_faces, candidates)
        sfc_gap = float(result.g)
        sfc_pressure = _pressure_from_gap(sfc_gap, model.pressure_law, model.pressure_values)
        copen = float(row.get("copen", np.nan))
        cpress = float(row.get("cpress", np.nan))
        gap_error = abs(sfc_gap - copen)
        pressure_error = abs(sfc_pressure - cpress)
        pressure_rel = _relative_error(sfc_pressure, cpress)
        gap_errors.append(gap_error)
        pressure_errors.append(pressure_error)
        pressure_rel_errors.append(pressure_rel)
        contact_rows.append(
            {
                "case_id": case_id,
                "node_id": node_id,
                "calculix_copen": copen,
                "sfc_dynamic_sdf_gap": sfc_gap,
                "gap_abs_error": gap_error,
                "calculix_cpress": cpress,
                "sfc_pressure": sfc_pressure,
                "pressure_abs_error": pressure_error,
                "pressure_rel_error": pressure_rel,
                "closest_face_id": int(result.face_id),
                "closest_x": float(result.p[0]),
                "closest_y": float(result.p[1]),
                "closest_z": float(result.p[2]),
                "normal_x": float(result.n[0]),
                "normal_y": float(result.n[1]),
                "normal_z": float(result.n[2]),
            }
        )

    max_gap_error = max(gap_errors or [np.inf])
    max_pressure_rel = max(pressure_rel_errors or [np.inf])
    metric_ok = bool(contact_rows and max_gap_error < 2.0e-8 and max_pressure_rel < 1.0e-3)
    summary = {
        "case_id": case_id,
        "analysis_type": "nonlinear_static_replay",
        "element_type": "C3D8",
        "reference_source": "CalculiX FRD CONTACT/DISP",
        "input_source": input_source,
        "frd_source": frd_source,
        "contact_point_count": len(contact_rows),
        "master_triangle_count": int(master_faces.shape[0]),
        "pressure_law": model.pressure_law,
        "pressure_values": ";".join(str(value) for value in model.pressure_values),
        "max_gap_abs_error": max_gap_error,
        "max_pressure_abs_error": max(pressure_errors or [np.inf]),
        "max_pressure_rel_error": max_pressure_rel,
        "supports_external_correctness": _bool_text(metric_ok),
        "supports_native_trajectory_equivalence": "false",
        "supports_efficiency": "false",
        "status": "supported" if metric_ok else "metric_failed",
    }
    gates = [
        {
            "case_id": case_id,
            "claim": "deforming_sdf_gap_matches_calculix_copen",
            "allowed": _bool_text(metric_ok),
            "reason": "requires FRD DISP/CONTACT replay gap and pressure metrics below thresholds",
        },
        {
            "case_id": case_id,
            "claim": "native_sfc_trajectory_equivalence",
            "allowed": "false",
            "reason": "scheibe2f2f is deformed-state replay, not a native SFC solve",
        },
        {
            "case_id": case_id,
            "claim": "efficiency",
            "allowed": "false",
            "reason": "this replay isolates correctness, not timing acceleration",
        },
    ]
    return summary, contact_rows, command_row, gates


def _instrument_ball_input(text: str, *, total_time: float, dt: float, output_dt: float) -> str:
    lines = text.splitlines()
    result: list[str] = []
    after_time_points = False
    after_dynamic = False
    inserted_contact = False
    for raw in lines:
        stripped = raw.strip()
        upper = stripped.upper()
        if after_time_points and stripped and not stripped.startswith("*"):
            result.append(f"0.0, {total_time:.12e}, {output_dt:.12e}")
            after_time_points = False
            continue
        if after_dynamic and stripped and not stripped.startswith("*"):
            result.append(f"{dt:.12e}, {total_time:.12e}, {max(dt * 1.0e-3, 1.0e-9):.12e}, {dt:.12e}")
            after_dynamic = False
            continue
        if upper.startswith("*TIME POINTS"):
            after_time_points = True
        elif upper.startswith("*DYNAMIC"):
            after_dynamic = True
        if upper.startswith("*ENDSTEP") or upper.startswith("*END STEP"):
            if not inserted_contact:
                result.extend(
                    [
                        "*NODE PRINT, NSET=nfloor, TOTALS=ONLY, GLOBAL=YES, TIME POINTS=times",
                        "RF",
                        "*CONTACT PRINT, TIME POINTS=times",
                        "CDIS,CSTR,CELS",
                        "*CONTACT PRINT, TOTALS=ONLY, TIME POINTS=times",
                        "CELS,CNUM",
                    ]
                )
                inserted_contact = True
        result.append(raw)
    return "\n".join(result) + "\n"


def _ball_model_from_input(model: ParsedInput) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    c3d8 = model.c3d8_elements
    if c3d8.size == 0:
        raise ValueError("ball input has no C3D8 elements")
    used_global_indices = np.asarray(sorted({int(node) for element in c3d8 for node in element}), dtype=np.int64)
    global_to_local = {int(global_idx): local for local, global_idx in enumerate(used_global_indices)}
    X = model.X[used_global_indices]
    elements = np.asarray([[global_to_local[int(idx)] for idx in element] for element in c3d8], dtype=np.int64)
    node_ids = model.node_ids[used_global_indices]
    return node_ids, X, elements, global_to_local


def _ball_surface_node_areas(X: np.ndarray, elements: np.ndarray, surface_node_indices: np.ndarray) -> np.ndarray:
    surface = set(int(idx) for idx in surface_node_indices)
    face_count: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    counts: dict[tuple[int, ...], int] = {}
    for element in elements:
        for face_nodes in C3D8_FACE_NODES.values():
            quad = element[face_nodes]
            key = tuple(sorted(int(node) for node in quad))
            counts[key] = counts.get(key, 0) + 1
            face_count[key] = (quad, element)
    areas = np.zeros(X.shape[0], dtype=float)
    for key, count in counts.items():
        if count != 1:
            continue
        quad, element = face_count[key]
        if not any(int(node) in surface for node in quad):
            continue
        oriented = _orient_quad_outward(X, quad, element)
        coords = X[oriented]
        area = 0.5 * np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
        area += 0.5 * np.linalg.norm(np.cross(coords[2] - coords[0], coords[3] - coords[0]))
        for node in oriented:
            if int(node) in surface:
                areas[int(node)] += 0.25 * area
    positive = areas[surface_node_indices]
    fallback = float(np.mean(positive[positive > 0.0])) if np.any(positive > 0.0) else 1.0
    for node in surface_node_indices:
        if areas[int(node)] <= 0.0:
            areas[int(node)] = fallback
    return areas


def _floor_plane_from_input(model: ParsedInput) -> tuple[float, tuple[float, float, float, float]]:
    floor_ids = model.nsets.get("nfloor", model.nsets.get("NFLOOR", []))
    id_to_index = model.id_to_index
    floor_indices = [id_to_index[int(node_id)] for node_id in floor_ids if int(node_id) in id_to_index]
    if not floor_indices:
        return -90.0, (-50.0, 100.0, -50.0, 50.0)
    coords = model.X[floor_indices]
    z = float(np.mean(coords[:, 2]) + 0.005)
    return z, (float(np.min(coords[:, 0])), float(np.max(coords[:, 0])), float(np.min(coords[:, 1])), float(np.max(coords[:, 1])))


def _floor_sdf(query: np.ndarray, floor_z: float, extent: tuple[float, float, float, float]) -> tuple[float, np.ndarray]:
    xmin, xmax, ymin, ymax = extent
    pad = 0.25 * max(xmax - xmin, ymax - ymin)
    floor_nodes = np.asarray(
        [
            [xmin - pad, ymin - pad, floor_z],
            [xmax + pad, ymin - pad, floor_z],
            [xmax + pad, ymax + pad, floor_z],
            [xmin - pad, ymax + pad, floor_z],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    result = dynamic_surface_sdf(np.asarray(query, dtype=float), floor_nodes, faces, np.arange(2, dtype=np.int64))
    return float(result.g), np.asarray(result.n, dtype=float)


def _ball_contact_response(
    x_current: np.ndarray,
    surface_nodes: np.ndarray,
    node_areas: np.ndarray,
    *,
    floor_z: float,
    extent: tuple[float, float, float, float],
    law: str,
    values: list[float],
    n_nodes: int,
    assemble_tangent: bool = True,
) -> tuple[np.ndarray, csr_matrix, float, float, int, float, float]:
    ndofs = 3 * int(n_nodes)
    force = np.zeros(ndofs, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    min_gap = np.inf
    max_pen = 0.0
    active = 0
    energy = 0.0
    normal_force_z = 0.0
    for node in surface_nodes:
        node = int(node)
        gap, normal = _floor_sdf(x_current[node], floor_z, extent)
        min_gap = min(min_gap, gap)
        overclosure = max(-gap, 0.0)
        if overclosure <= 0.0:
            continue
        area = float(node_areas[node])
        pressure = _pressure_from_gap(gap, law, values)
        scalar_force = pressure * area
        dofs = np.asarray([3 * node, 3 * node + 1, 3 * node + 2], dtype=np.int64)
        force[dofs] += scalar_force * normal
        normal_force_z += scalar_force * normal[2]
        if law.upper() == "LINEAR":
            stiffness = float(values[0]) if values else 0.0
            pressure_tangent = stiffness
            energy += 0.5 * stiffness * overclosure * overclosure * area
        elif law.upper() == "EXPONENTIAL":
            p0 = float(values[0])
            c0 = max(float(values[1]), 1.0e-30)
            pressure_tangent = p0 * np.exp(overclosure * np.log(100.0) / c0) * np.log(100.0) / c0
            energy += area * p0 * c0 / np.log(100.0) * (np.exp(overclosure * np.log(100.0) / c0) - 1.0)
        else:
            pressure_tangent = 0.0
        if assemble_tangent:
            local = pressure_tangent * area * np.outer(normal, normal)
            for a in range(3):
                for b in range(3):
                    rows.append(int(dofs[a]))
                    cols.append(int(dofs[b]))
                    data.append(float(local[a, b]))
        active += 1
        max_pen = max(max_pen, overclosure)
    if not np.isfinite(min_gap):
        min_gap = 0.0
    tangent = csr_matrix((data, (rows, cols)), shape=(ndofs, ndofs)) if assemble_tangent else csr_matrix((ndofs, ndofs))
    return force, tangent, float(min_gap), float(max_pen), int(active), float(energy), float(normal_force_z)


def _native_ball_history(
    model: ParsedInput,
    *,
    total_time: float,
    dt: float,
    output_dt: float,
) -> tuple[list[Row], np.ndarray]:
    node_ids, X, elements, global_to_local = _ball_model_from_input(model)
    mesh = VolumeMesh(X, elements, element_type="C3D8")
    body = DeformableBody(mesh=mesh, material={"E": model.material_E, "nu": model.material_nu}, density=model.density)
    K = assemble_stiffness_matrix(body).tocsr()
    M = assemble_mass_matrix(body, kind="consistent").tocsr()
    g_vec = model.gravity * model.gravity_direction
    f_ext = assemble_gravity_force(body, tuple(float(v) for v in g_vec))
    id_to_global = {int(node_id): idx for idx, node_id in enumerate(model.node_ids)}
    velocity = np.zeros_like(X)
    if model.initial_velocity is not None:
        for local, node_id in enumerate(node_ids):
            velocity[local] = model.initial_velocity[id_to_global[int(node_id)]]
    nsurface_ids = model.nsets.get("nsurface", model.nsets.get("NSURFACE", []))
    surface_nodes = np.asarray([global_to_local[id_to_global[int(node_id)]] for node_id in nsurface_ids if int(node_id) in id_to_global and id_to_global[int(node_id)] in global_to_local], dtype=np.int64)
    node_areas = _ball_surface_node_areas(X, elements, surface_nodes)
    floor_z, extent = _floor_plane_from_input(model)
    ndofs = body.n_dofs
    u = np.zeros(ndofs, dtype=float)
    v = velocity.reshape(-1).copy()
    beta = 0.25
    gamma = 0.5
    c0 = 1.0 / (beta * dt * dt)
    contact_force, _contact_tangent, *_ = _ball_contact_response(
        X,
        surface_nodes,
        node_areas,
        floor_z=floor_z,
        extent=extent,
        law=model.pressure_law,
        values=model.pressure_values,
        n_nodes=X.shape[0],
        assemble_tangent=False,
    )
    a = np.asarray(spsolve(M.tocsc(), f_ext + contact_force - K @ u), dtype=float)
    rows: list[Row] = []
    steps = int(np.ceil(total_time / dt))
    output_stride = max(1, int(round(output_dt / dt)))
    for step in range(steps + 1):
        x_current = X + u.reshape((-1, 3))
        force, _tangent, min_gap, max_pen, active, energy, normal_z = _ball_contact_response(
            x_current,
            surface_nodes,
            node_areas,
            floor_z=floor_z,
            extent=extent,
            law=model.pressure_law,
            values=model.pressure_values,
            n_nodes=X.shape[0],
            assemble_tangent=False,
        )
        if step % output_stride == 0 or step == steps:
            rows.append(
                {
                    "time": float(step * dt),
                    "sfc_z_cm": float(np.mean(x_current[:, 2])),
                    "sfc_x_cm": float(np.mean(x_current[:, 0])),
                    "sfc_min_gap": min_gap,
                    "sfc_max_penetration": max_pen,
                    "sfc_active_contact_count": active,
                    "sfc_contact_force_z": normal_z,
                    "sfc_contact_energy": energy,
                }
            )
        if step == steps:
            break
        u_pred = u + dt * v + dt * dt * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a
        u_guess = u_pred.copy()
        for _iteration in range(10):
            x_guess = X + u_guess.reshape((-1, 3))
            force, tangent, *_ = _ball_contact_response(
                x_guess,
                surface_nodes,
                node_areas,
                floor_z=floor_z,
                extent=extent,
                law=model.pressure_law,
                values=model.pressure_values,
                n_nodes=X.shape[0],
                assemble_tangent=True,
            )
            a_guess = c0 * (u_guess - u_pred)
            residual = M @ a_guess + K @ u_guess - f_ext - force
            effective = (M * c0 + K + tangent).tocsc()
            correction = np.asarray(spsolve(effective, -residual), dtype=float)
            u_guess += correction
            if float(np.linalg.norm(correction)) <= 1.0e-9 * max(1.0, float(np.linalg.norm(u_guess))):
                break
        u = u_guess
        a = c0 * (u - u_pred)
        v = v_pred + gamma * dt * a
    return rows, X + u.reshape((-1, 3))


def _run_ball_calculix(
    args: argparse.Namespace,
    out_dir: Path,
    input_text: str,
) -> tuple[dict[float, np.ndarray], dict[float, Row], dict[float, list[Row]], Row]:
    case_id = "ball_c3d8_dynamic_drop"
    if args.skip_calculix or not _calculix_available():
        base = ROOT / "paper" / "numerical_experiments" / "calculix_official_contact_examples" / "calculix_runs" / case_id
        dat_path = base / "ball.dat"
        command = {"case_id": case_id, "command": "existing locked CalculiX ball.dat", "completed": _bool_text(dat_path.exists())}
    else:
        run_dir = out_dir / "calculix_runs" / case_id
        run_dir.mkdir(parents=True, exist_ok=True)
        case_name = "ball"
        (run_dir / f"{case_name}.inp").write_text(input_text, encoding="utf-8")
        command_text = f"cd {_wsl_quote(_wsl_path(run_dir))} && ccx {_wsl_quote(case_name)}"
        start = time.perf_counter()
        proc = _run(["wsl", "--exec", "bash", "-lc", command_text], cwd=ROOT, timeout=300)
        wall = time.perf_counter() - start
        (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
        dat_path = run_dir / "ball.dat"
        command = {
            "case_id": case_id,
            "external_solver": "CalculiX",
            "external_solver_version": _calculix_version(),
            "completed": _bool_text(proc.returncode == 0 and dat_path.exists()),
            "return_code": proc.returncode,
            "wall_time_seconds": wall,
            "command": f"wsl --exec bash -lc \"cd {_wsl_path(run_dir)} && ccx ball\"",
            "input_file": _display_path(run_dir / "ball.inp"),
        }
    parsed = parse_calculix_input(input_text)
    node_ids, _X, _elements, _map = _ball_model_from_input(parsed)
    if not dat_path.exists():
        return {}, {}, {}, command
    displacements = _parse_nodal_vectors(dat_path, node_ids, quantity="u")
    totals = _parse_totals(dat_path)
    contacts = _parse_contact_elements(dat_path)
    return displacements, totals, contacts, command


def _merge_ball_histories(
    model: ParsedInput,
    native_rows: list[Row],
    calc_displacements: dict[float, np.ndarray],
    totals: dict[float, Row],
    contacts: dict[float, list[Row]],
) -> tuple[list[Row], Row]:
    node_ids, X, _elements, _map = _ball_model_from_input(model)
    native_by_time = {round(float(row["time"]), 12): row for row in native_rows}
    rows: list[Row] = []
    z_sfc: list[float] = []
    z_calc: list[float] = []
    gap_sfc: list[float] = []
    gap_calc: list[float] = []
    force_sfc: list[float] = []
    force_calc: list[float] = []
    energy_sfc: list[float] = []
    energy_calc: list[float] = []
    all_times = sorted(set(native_by_time) | {round(float(t), 12) for t in calc_displacements})
    for time_key in all_times:
        native = native_by_time.get(time_key)
        calc_u = calc_displacements.get(time_key)
        calc_z = ""
        calc_x = ""
        if calc_u is not None:
            calc_x = float(np.mean((X + calc_u)[:, 0]))
            calc_z = float(np.mean((X + calc_u)[:, 2]))
        total = totals.get(float(time_key), {})
        contact_rows = contacts.get(float(time_key), [])
        cdis_values = [float(row["clearance"]) for row in contact_rows if row.get("quantity") == "cdis" and "clearance" in row]
        cels_values = [float(row["energy"]) for row in contact_rows if row.get("quantity") == "cels" and "energy" in row]
        min_cdis = min(cdis_values) if cdis_values else ""
        cels = total.get("contact_energy_calculix", float(np.sum(cels_values)) if cels_values else "")
        rf = total.get("normal_force_calculix", "")
        row = {
            "case_id": "ball_c3d8_dynamic_drop_native_trajectory",
            "time": time_key,
            "sfc_x_cm": native.get("sfc_x_cm", "") if native else "",
            "calculix_x_cm": calc_x,
            "sfc_z_cm": native.get("sfc_z_cm", "") if native else "",
            "calculix_z_cm": calc_z,
            "sfc_min_gap": native.get("sfc_min_gap", "") if native else "",
            "calculix_min_cdis": min_cdis,
            "sfc_active_contact_count": native.get("sfc_active_contact_count", "") if native else "",
            "calculix_contact_count": total.get("contact_count_calculix", len(cdis_values) if cdis_values else ""),
            "sfc_contact_force_z": native.get("sfc_contact_force_z", "") if native else "",
            "calculix_floor_rf_norm": rf,
            "sfc_contact_energy": native.get("sfc_contact_energy", "") if native else "",
            "calculix_cels": cels,
        }
        if native and calc_u is not None:
            z_sfc.append(float(native["sfc_z_cm"]))
            z_calc.append(float(calc_z))
            if min_cdis != "":
                gap_sfc.append(float(native["sfc_min_gap"]))
                gap_calc.append(float(min_cdis))
            if rf != "":
                force_sfc.append(abs(float(native["sfc_contact_force_z"])))
                force_calc.append(float(rf))
            if cels != "":
                energy_sfc.append(float(native["sfc_contact_energy"]))
                energy_calc.append(float(cels))
        rows.append(row)
    max_native_active = max([int(row["sfc_active_contact_count"]) for row in native_rows] or [0])
    max_calculix_active = max(
        [int(row["calculix_contact_count"]) for row in rows if row["calculix_contact_count"] not in {"", None}] or [0]
    )
    metrics = {
        "case_id": "ball_c3d8_dynamic_drop_native_trajectory",
        "time_step": "",
        "z_cm_l2_rel_error": "" if not z_sfc else _relative_vector_error(z_sfc, z_calc),
        "gap_l2_rel_error": "" if not gap_sfc else _relative_vector_error(gap_sfc, gap_calc),
        "force_l2_rel_error": "" if not force_sfc else _relative_vector_error(force_sfc, force_calc),
        "energy_l2_rel_error": "" if not energy_sfc else _relative_vector_error(energy_sfc, energy_calc),
        "max_sfc_active_contact_count": max_native_active,
        "max_calculix_active_contact_count": max_calculix_active,
        "overlapping_time_count": len(z_sfc),
    }
    return rows, metrics


def _run_ball(args: argparse.Namespace, out_dir: Path) -> tuple[Row, list[Row], Row, list[Row]]:
    case_id = "ball_c3d8_dynamic_drop_native_trajectory"
    text, source = _copy_or_read_example("ball.inp.gz", args.examples_dir, args.wsl_examples_dir)
    total_time = float(args.ball_time)
    dt = float(args.ball_dt)
    output_dt = float(args.ball_output_dt)
    instrumented = _instrument_ball_input(text, total_time=total_time, dt=dt, output_dt=output_dt)
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs" / "ball_instrumented.inp").write_text(instrumented, encoding="utf-8")
    model = parse_calculix_input(instrumented)
    native_rows, _x_final = _native_ball_history(model, total_time=total_time, dt=dt, output_dt=output_dt)
    calc_displacements, totals, contacts, command_row = _run_ball_calculix(args, out_dir, instrumented)
    history_rows, metrics = _merge_ball_histories(model, native_rows, calc_displacements, totals, contacts)
    z_error = metrics["z_cm_l2_rel_error"]
    gap_error = metrics["gap_l2_rel_error"]
    force_error = metrics["force_l2_rel_error"]
    external = bool(calc_displacements)
    native_contact = int(metrics["max_sfc_active_contact_count"]) > 0
    external_contact = int(metrics["max_calculix_active_contact_count"]) > 0
    metric_ok = (
        external
        and native_contact
        and external_contact
        and z_error != ""
        and float(z_error) < 0.20
        and (gap_error == "" or float(gap_error) < 0.75)
        and (force_error == "" or float(force_error) < 0.90)
    )
    summary = {
        "case_id": case_id,
        "analysis_type": "native_dynamic",
        "element_type": "C3D8",
        "reference_source": "official CalculiX ball.inp",
        "input_source": source,
        "total_time": total_time,
        "dt": dt,
        "output_dt": output_dt,
        **metrics,
        "supports_external_correctness": _bool_text(metric_ok),
        "supports_native_trajectory_equivalence": _bool_text(metric_ok),
        "supports_efficiency": "false",
        "status": "supported" if metric_ok else ("no_calculix_reference" if not external else "native_external_comparison_failed"),
    }
    gates = [
        {
            "case_id": case_id,
            "claim": "native_sfc_trajectory_equivalence",
            "allowed": _bool_text(metric_ok),
            "reason": "requires native SFC C3D8 trajectory, CalculiX dynamic trajectory, active contact, and error metrics below thresholds",
        },
        {
            "case_id": case_id,
            "claim": "external_correctness",
            "allowed": _bool_text(metric_ok),
            "reason": "requires the native trajectory equivalence gate for this dynamic contact case",
        },
        {
            "case_id": case_id,
            "claim": "efficiency",
            "allowed": "false",
            "reason": "ball runner is a correctness trajectory check, not a repeated timing benchmark",
        },
    ]
    return summary, history_rows, command_row, gates


def _write_plots(out_dir: Path, scheibe_rows: list[Row], ball_rows: list[Row]) -> list[Row]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_rows: list[Row] = []
    if scheibe_rows:
        copen = np.asarray([float(row["calculix_copen"]) for row in scheibe_rows])
        sfc_gap = np.asarray([float(row["sfc_dynamic_sdf_gap"]) for row in scheibe_rows])
        cpress = np.asarray([float(row["calculix_cpress"]) for row in scheibe_rows])
        sfc_pressure = np.asarray([float(row["sfc_pressure"]) for row in scheibe_rows])
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
        axes[0].scatter(copen, sfc_gap, s=20)
        lo = min(float(copen.min()), float(sfc_gap.min()))
        hi = max(float(copen.max()), float(sfc_gap.max()))
        axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
        axes[0].set_xlabel("CalculiX COPEN")
        axes[0].set_ylabel("SFC dynamic SDF gap")
        axes[0].grid(True, alpha=0.25)
        axes[1].scatter(cpress, sfc_pressure, s=20)
        lo = min(float(cpress.min()), float(sfc_pressure.min()))
        hi = max(float(cpress.max()), float(sfc_pressure.max()))
        axes[1].plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
        axes[1].set_xlabel("CalculiX CPRESS")
        axes[1].set_ylabel("SFC pressure replay")
        axes[1].grid(True, alpha=0.25)
        fig.tight_layout()
        png = figures / "scheibe2f2f_sdf_replay_scatter.png"
        pdf = figures / "scheibe2f2f_sdf_replay_scatter.pdf"
        fig.savefig(png, dpi=180)
        fig.savefig(pdf)
        plt.close(fig)
        plot_rows.append({"plot": "scheibe2f2f_sdf_replay_scatter", "png": str(png.relative_to(out_dir)), "pdf": str(pdf.relative_to(out_dir)), "description": "Scheibe2f2f FRD COPEN/CPRESS versus SFC deformed-SDF replay"})

    if ball_rows:
        times = np.asarray([float(row["time"]) for row in ball_rows])
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True)
        sfc_z = np.asarray([float(row["sfc_z_cm"]) if row["sfc_z_cm"] != "" else np.nan for row in ball_rows])
        calc_z = np.asarray([float(row["calculix_z_cm"]) if row["calculix_z_cm"] != "" else np.nan for row in ball_rows])
        axes[0].plot(times, sfc_z, label="SFC native")
        axes[0].plot(times, calc_z, "--", label="CalculiX")
        axes[0].set_ylabel("z center")
        axes[0].legend()
        axes[0].grid(True, alpha=0.25)
        sfc_gap = np.asarray([float(row["sfc_min_gap"]) if row["sfc_min_gap"] != "" else np.nan for row in ball_rows])
        calc_gap = np.asarray([float(row["calculix_min_cdis"]) if row["calculix_min_cdis"] != "" else np.nan for row in ball_rows])
        axes[1].plot(times, sfc_gap, label="SFC min gap")
        axes[1].plot(times, calc_gap, "--", label="CalculiX CDIS")
        axes[1].set_ylabel("gap")
        axes[1].legend()
        axes[1].grid(True, alpha=0.25)
        sfc_force = np.asarray([float(row["sfc_contact_force_z"]) if row["sfc_contact_force_z"] != "" else np.nan for row in ball_rows])
        calc_rf = np.asarray([float(row["calculix_floor_rf_norm"]) if row["calculix_floor_rf_norm"] != "" else np.nan for row in ball_rows])
        axes[2].plot(times, sfc_force, label="SFC contact force z")
        axes[2].plot(times, calc_rf, "--", label="CalculiX floor RF")
        axes[2].set_xlabel("time")
        axes[2].set_ylabel("force")
        axes[2].legend()
        axes[2].grid(True, alpha=0.25)
        fig.tight_layout()
        png = figures / "ball_dynamic_drop_native_trajectory.png"
        pdf = figures / "ball_dynamic_drop_native_trajectory.pdf"
        fig.savefig(png, dpi=180)
        fig.savefig(pdf)
        plt.close(fig)
        plot_rows.append({"plot": "ball_dynamic_drop_native_trajectory", "png": str(png.relative_to(out_dir)), "pdf": str(pdf.relative_to(out_dir)), "description": "Official ball dynamic drop SFC native trajectory versus CalculiX"})
    return plot_rows


def _write_summary(out_dir: Path, summaries: list[Row], gates: list[Row], plots: list[Row], commands: list[Row]) -> None:
    lines = [
        "# CalculiX Scheibe/Ball SDF Validation",
        "",
        "This package extends the official CalculiX evidence line beyond small contact-law examples.",
        "",
        "## Cases",
        "",
        "| Case | Type | Status | Main metrics |",
        "|---|---|---|---|",
    ]
    for row in summaries:
        if row["case_id"].startswith("scheibe"):
            metrics = f"max gap error `{row['max_gap_abs_error']}`, max pressure rel. `{row['max_pressure_rel_error']}`"
        else:
            metrics = f"z_cm rel. `{row['z_cm_l2_rel_error']}`, gap rel. `{row['gap_l2_rel_error']}`, force rel. `{row['force_l2_rel_error']}`"
        lines.append(f"| {row['case_id']} | {row['analysis_type']} | {row['status']} | {metrics} |")
    lines.extend(["", "## Claim Gates", "", "| Case | Claim | Allowed | Reason |", "|---|---|---|---|"])
    for gate in gates:
        lines.append(f"| {gate['case_id']} | {gate['claim']} | {gate['allowed']} | {gate['reason']} |")
    lines.extend(["", "## Commands", ""])
    for command in commands:
        lines.append(f"- `{command.get('command', '')}`")
    lines.extend(["", "## Figures", ""])
    for plot in plots:
        lines.append(f"- `{plot['png']}`: {plot['description']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`scheibe2f2f` is a deformed-state replay validation: SFC reads the final CalculiX-deformed C3D8 surface and reproduces COPEN/CPRESS through the dynamic SDF and the same pressure-overclosure law. It does not claim a native SFC nonlinear static trajectory.",
            "",
            "`ball` is a native C3D8 dynamic trajectory comparison against the official CalculiX drop example. The claim gate remains false unless the native SFC trajectory, contact activation, and error metrics pass the generated thresholds.",
        ]
    )
    (out_dir / "scheibe_ball_sdf_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scheibe_summary, scheibe_points, scheibe_command, scheibe_gates = _run_scheibe(args, out_dir)
    ball_summary, ball_history, ball_command, ball_gates = _run_ball(args, out_dir)
    plot_rows = _write_plots(out_dir, scheibe_points, ball_history)

    summaries = [scheibe_summary, ball_summary]
    gates = scheibe_gates + ball_gates
    commands = [scheibe_command, ball_command]
    _write_csv(
        out_dir / "scheibe_ball_sdf_validation_summary.csv",
        sorted({key for row in summaries for key in row}),
        summaries,
    )
    _write_csv(
        out_dir / "scheibe2f2f_contact_points.csv",
        [
            "case_id",
            "node_id",
            "calculix_copen",
            "sfc_dynamic_sdf_gap",
            "gap_abs_error",
            "calculix_cpress",
            "sfc_pressure",
            "pressure_abs_error",
            "pressure_rel_error",
            "closest_face_id",
            "closest_x",
            "closest_y",
            "closest_z",
            "normal_x",
            "normal_y",
            "normal_z",
        ],
        scheibe_points,
    )
    _write_csv(
        out_dir / "ball_dynamic_drop_trajectory.csv",
        [
            "case_id",
            "time",
            "sfc_x_cm",
            "calculix_x_cm",
            "sfc_z_cm",
            "calculix_z_cm",
            "sfc_min_gap",
            "calculix_min_cdis",
            "sfc_active_contact_count",
            "calculix_contact_count",
            "sfc_contact_force_z",
            "calculix_floor_rf_norm",
            "sfc_contact_energy",
            "calculix_cels",
        ],
        ball_history,
    )
    _write_csv(out_dir / "scheibe_ball_claim_gates.csv", ["case_id", "claim", "allowed", "reason"], gates)
    _write_csv(out_dir / "scheibe_ball_commands.csv", sorted({key for row in commands for key in row}), commands)
    _write_csv(out_dir / "scheibe_ball_plots.csv", ["plot", "png", "pdf", "description"], plot_rows)
    _write_summary(out_dir, summaries, gates, plot_rows, commands)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use short defaults suitable for CI.")
    parser.add_argument("--skip-calculix", action="store_true", help="Use locked CalculiX artifacts where available.")
    parser.add_argument("--examples-dir", type=Path, default=None)
    parser.add_argument("--wsl-examples-dir", default=DEFAULT_WSL_EXAMPLES_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "calculix_scheibe_ball_sdf_validation")
    parser.add_argument("--ball-time", type=float, default=None)
    parser.add_argument("--ball-dt", type=float, default=None)
    parser.add_argument("--ball-output-dt", type=float, default=None)
    args = parser.parse_args(argv)
    if args.ball_time is None:
        args.ball_time = 0.004 if args.quick else 0.13
    if args.ball_dt is None:
        args.ball_dt = 0.002 if args.quick else 0.002
    if args.ball_output_dt is None:
        args.ball_output_dt = args.ball_dt if args.quick else 0.004
    return args


def main(argv: list[str] | None = None) -> int:
    return run_validation(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
