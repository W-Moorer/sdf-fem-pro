"""Replay official CalculiX contact-law examples with the dynamic SDF.

This validation line uses official CalculiX C3D8 contact examples as external
deformed-state references.  CalculiX solves the static contact models; SFC then
queries the current deformed master surface using the dynamic surface SDF and
replays the pressure-overclosure law at the same contact output locations.

The runner instruments the official inputs with additional output requests
only.  The physics model is not otherwise changed.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
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

from sfc.sdf.dynamic_surface_sdf import dynamic_surface_sdf  # noqa: E402
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

C3D8_FACE_NODES = {
    "S1": np.array([0, 1, 2, 3], dtype=np.int64),
    "S2": np.array([4, 7, 6, 5], dtype=np.int64),
    "S3": np.array([0, 4, 5, 1], dtype=np.int64),
    "S4": np.array([1, 5, 6, 2], dtype=np.int64),
    "S5": np.array([2, 6, 7, 3], dtype=np.int64),
    "S6": np.array([3, 7, 4, 0], dtype=np.int64),
}


@dataclass(frozen=True)
class LawCase:
    case_id: str
    file_name: str
    scope: str


LAW_CASES = [
    LawCase("contactenergy_c3d8_surface_linear", "contactenergy.inp", "surface_to_surface_linear_energy_reference"),
    LawCase("contact1_c3d8_node_exponential", "contact1.inp", "node_to_surface_exponential_gap_pressure_diagnostic"),
    LawCase("contact3_c3d8_node_linear", "contact3.inp", "node_to_surface_linear_law_alignment"),
    LawCase("contact6_c3d8_node_stiff_linear", "contact6.inp", "node_to_surface_stiff_linear_law_alignment"),
]


@dataclass
class ParsedModel:
    node_ids: np.ndarray
    X: np.ndarray
    element_ids: np.ndarray
    elements: np.ndarray
    nsets: dict[str, list[int]]
    elsets: dict[str, list[int]]
    surfaces: dict[str, dict[str, Any]]
    contact_slave_surface: str
    contact_master_surface: str
    pressure_law: str
    pressure_values: list[float]
    cload_total: np.ndarray

    @property
    def id_to_index(self) -> dict[int, int]:
        return {int(node_id): idx for idx, node_id in enumerate(self.node_ids)}


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_keyword(line: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in line.split(",")]
    name = parts[0].upper()
    params: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip().upper()] = value.strip()
    return name, params


def _values(line: str) -> list[str]:
    return [value.strip() for value in line.split(",") if value.strip()]


def _instrument_contact_output(text: str) -> str:
    if "CDIS" in text.upper() and "CSTR" in text.upper():
        return text
    lines: list[str] = []
    inserted = False
    for raw in text.splitlines():
        if raw.strip().upper().startswith("*END STEP") and not inserted:
            lines.append("*CONTACT PRINT")
            lines.append("CDIS,CSTR,CELS")
            lines.append("*CONTACT PRINT,TOTALS=YES")
            lines.append("CELS")
            inserted = True
        lines.append(raw)
    return "\n".join(lines) + "\n"


def parse_official_contact_model(text: str) -> ParsedModel:
    nodes: dict[int, list[float]] = {}
    elements: dict[int, list[int]] = {}
    nsets: dict[str, list[int]] = {}
    elsets: dict[str, list[int]] = {}
    surfaces: dict[str, dict[str, Any]] = {}
    contact_slave_surface = ""
    contact_master_surface = ""
    pressure_law = ""
    pressure_values: list[float] = []
    cloads: list[tuple[int, int, float]] = []

    section = ""
    params: dict[str, str] = {}
    current_nset = ""
    current_elset = ""
    current_surface = ""
    current_surface_type = "ELEMENT"
    expect_pressure_values = False
    expect_contact_pair = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            section, params = _parse_keyword(line)
            current_nset = params.get("NSET", "") if section in {"*NODE", "*NSET"} else ""
            current_elset = params.get("ELSET", "") if section in {"*ELEMENT", "*ELSET"} else ""
            if section == "*SURFACE":
                current_surface = params.get("NAME", "")
                current_surface_type = params.get("TYPE", "ELEMENT").upper()
                surfaces.setdefault(current_surface, {"type": current_surface_type, "entries": []})
            else:
                current_surface = ""
            if section == "*CONTACT PAIR":
                expect_contact_pair = True
            if section == "*SURFACE BEHAVIOR":
                pressure_law = params.get("PRESSURE-OVERCLOSURE", "").upper()
                expect_pressure_values = True
            continue

        vals = _values(line)
        if not vals:
            continue
        if expect_contact_pair:
            contact_slave_surface = vals[0]
            contact_master_surface = vals[1]
            expect_contact_pair = False
            continue
        if expect_pressure_values:
            pressure_values = [float(value) for value in vals]
            expect_pressure_values = False
            continue

        if section == "*NODE":
            node_id = int(vals[0])
            nodes[node_id] = [float(vals[1]), float(vals[2]), float(vals[3])]
            if current_nset:
                nsets.setdefault(current_nset, []).append(node_id)
        elif section == "*ELEMENT":
            element_id = int(vals[0])
            elements[element_id] = [int(value) for value in vals[1:]]
            if current_elset:
                elsets.setdefault(current_elset, []).append(element_id)
        elif section == "*NSET":
            nsets.setdefault(current_nset, []).extend(int(value) for value in vals)
        elif section == "*ELSET":
            elsets.setdefault(current_elset, []).extend(int(value) for value in vals)
        elif section == "*SURFACE":
            surfaces[current_surface]["entries"].append(vals)
        elif section == "*CLOAD":
            cloads.append((int(vals[0]), int(vals[1]), float(vals[2])))

    if not contact_slave_surface or not contact_master_surface:
        raise ValueError("failed to parse contact pair surfaces")
    if pressure_law not in {"LINEAR", "EXPONENTIAL"}:
        raise ValueError(f"unsupported pressure-overclosure law {pressure_law!r}")
    node_ids = np.asarray(sorted(nodes), dtype=np.int64)
    id_to_index = {int(node_id): idx for idx, node_id in enumerate(node_ids)}
    X = np.asarray([nodes[int(node_id)] for node_id in node_ids], dtype=float)
    element_ids = np.asarray(sorted(elements), dtype=np.int64)
    conn = np.asarray([[id_to_index[int(node_id)] for node_id in elements[int(element_id)]] for element_id in element_ids], dtype=np.int64)
    cload_total = np.zeros(3, dtype=float)
    for _node, dof, value in cloads:
        cload_total[int(dof) - 1] += float(value)
    return ParsedModel(
        node_ids=node_ids,
        X=X,
        element_ids=element_ids,
        elements=conn,
        nsets=nsets,
        elsets=elsets,
        surfaces=surfaces,
        contact_slave_surface=contact_slave_surface,
        contact_master_surface=contact_master_surface,
        pressure_law=pressure_law,
        pressure_values=pressure_values,
        cload_total=cload_total,
    )


def _parse_last_vector_block(text: str, label: str) -> dict[int, np.ndarray]:
    blocks: list[dict[int, np.ndarray]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if label.lower() not in line.lower():
            continue
        block: dict[int, np.ndarray] = {}
        for row in lines[index + 1 :]:
            parts = row.split()
            if len(parts) == 4 and parts[0].lstrip("+-").isdigit():
                block[int(parts[0])] = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
            elif block:
                break
        if block:
            blocks.append(block)
    return blocks[-1] if blocks else {}


def _parse_contact_block(text: str, marker: str) -> list[Row]:
    lines = text.splitlines()
    blocks: list[list[Row]] = []
    for index, line in enumerate(lines):
        lower = line.lower()
        if marker not in lower:
            continue
        is_face = "slave element+face" in lower
        block: list[Row] = []
        for raw in lines[index + 1 :]:
            parts = raw.split()
            if is_face and len(parts) >= 3 and parts[0].lstrip("+-").isdigit() and parts[1].lstrip("+-").isdigit():
                row: Row = {"slave_element": int(parts[0]), "slave_face": int(parts[1])}
                values = [float(value) for value in parts[2:]]
                row["normal"] = values[0]
                if len(values) > 1:
                    row["tang1"] = values[1]
                if len(values) > 2:
                    row["tang2"] = values[2]
                block.append(row)
            elif (not is_face) and len(parts) >= 2 and parts[0].lstrip("+-").isdigit():
                row = {"slave_node": int(parts[0])}
                values = [float(value) for value in parts[1:]]
                row["normal"] = values[0]
                if len(values) > 1:
                    row["tang1"] = values[1]
                if len(values) > 2:
                    row["tang2"] = values[2]
                block.append(row)
            elif block:
                break
        if block:
            blocks.append(block)
    return blocks[-1] if blocks else []


def parse_calculix_contact_dat(text: str, node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[Row], list[Row], list[Row], float | None]:
    displacement = _parse_last_vector_block(text, "displacements")
    force = _parse_last_vector_block(text, "forces")
    U = np.asarray([displacement.get(int(node_id), np.zeros(3)) for node_id in node_ids], dtype=float)
    RF = np.asarray([force.get(int(node_id), np.zeros(3)) for node_id in node_ids], dtype=float)
    cdis = _parse_contact_block(text, "relative contact displacement")
    cstr = _parse_contact_block(text, "contact stress")
    cels = _parse_contact_block(text, "contact print energy")
    totals = re.findall(r"total contact spring energy for time\s+[-+0-9.Ee]+\s*\n\s*([-+0-9.Ee]+)", text, flags=re.IGNORECASE)
    total_cels = float(totals[-1]) if totals else None
    return U, RF, cdis, cstr, cels, total_cels


def _orient_quad_outward(X: np.ndarray, quad: np.ndarray, element: np.ndarray) -> np.ndarray:
    polygon = X[quad]
    normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
    if float(np.dot(normal, polygon.mean(axis=0) - X[element].mean(axis=0))) < 0.0:
        return quad[::-1]
    return quad


def _surface_quads(model: ParsedModel, surface_name: str, X: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, int, str]]:
    surface = model.surfaces[surface_name]
    if surface["type"] == "NODE":
        return []
    element_id_to_index = {int(element_id): idx for idx, element_id in enumerate(model.element_ids)}
    quads: list[tuple[np.ndarray, np.ndarray, int, str]] = []
    for entry in surface["entries"]:
        if len(entry) < 2:
            continue
        target, face_name = entry[0], entry[1].upper()
        if face_name not in C3D8_FACE_NODES:
            continue
        element_ids = model.elsets.get(target, [int(target)] if target.lstrip("+-").isdigit() else [])
        for element_id in element_ids:
            element = model.elements[element_id_to_index[int(element_id)]]
            quad = _orient_quad_outward(X, element[C3D8_FACE_NODES[face_name]], element)
            quads.append((quad, element, int(element_id), face_name))
    return quads


def _triangles_from_quads(quads: list[tuple[np.ndarray, np.ndarray, int, str]]) -> np.ndarray:
    triangles: list[np.ndarray] = []
    for quad, _element, _element_id, _face_name in quads:
        triangles.append(quad[[0, 1, 2]])
        triangles.append(quad[[0, 2, 3]])
    return np.asarray(triangles, dtype=np.int64)


def _surface_nodes(model: ParsedModel, surface_name: str) -> np.ndarray:
    surface = model.surfaces[surface_name]
    node_ids: list[int] = []
    if surface["type"] != "NODE":
        return np.empty(0, dtype=np.int64)
    for entry in surface["entries"]:
        target = entry[0]
        if target.lstrip("+-").isdigit():
            node_ids.append(int(target))
        else:
            node_ids.extend(model.nsets.get(target, []))
    return np.asarray([model.id_to_index[int(node_id)] for node_id in node_ids], dtype=np.int64)


def _quad_area(coords: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))) + 0.5 * float(
        np.linalg.norm(np.cross(coords[2] - coords[0], coords[3] - coords[0]))
    )


def _query_points(model: ParsedModel, X_current: np.ndarray) -> list[Row]:
    slave_surface = model.surfaces[model.contact_slave_surface]
    points: list[Row] = []
    if slave_surface["type"] == "NODE":
        for node in _surface_nodes(model, model.contact_slave_surface):
            points.append(
                {
                    "kind": "slave_node",
                    "slave_node": int(model.node_ids[int(node)]),
                    "slave_element": "",
                    "slave_face": "",
                    "x": X_current[int(node)],
                    "area_weight": 1.0,
                }
            )
        return points

    for quad, _element, element_id, face_name in _surface_quads(model, model.contact_slave_surface, X_current):
        coords = X_current[quad]
        points.append(
            {
                "kind": "slave_face_centroid",
                "slave_node": "",
                "slave_element": element_id,
                "slave_face": int(face_name[1:]),
                "x": coords.mean(axis=0),
                "area_weight": _quad_area(coords),
            }
        )
    return points


def _pressure_from_gap(law: str, values: list[float], gap: float) -> float:
    overclosure = max(-float(gap), 0.0)
    if law == "LINEAR":
        stiffness = float(values[0])
        tension_offset = float(values[1]) if len(values) > 1 else 0.0
        return max(stiffness * overclosure - tension_offset, 0.0)
    if law == "EXPONENTIAL":
        if len(values) < 2:
            return 0.0
        c0, p0 = float(values[0]), float(values[1])
        if c0 <= 0.0:
            return 0.0
        return float(p0 * np.exp(overclosure * np.log(100.0) / c0))
    return 0.0


def _unit_area_energy(law: str, values: list[float], gap: float, pressure: float) -> float:
    overclosure = max(-float(gap), 0.0)
    if overclosure <= 0.0:
        return 0.0
    if law == "EXPONENTIAL" and len(values) >= 2 and values[0] > 0.0:
        c0, p0 = float(values[0]), float(values[1])
        return float(p0 * c0 / np.log(100.0) * (np.exp(overclosure * np.log(100.0) / c0) - 1.0))
    return 0.5 * float(pressure) * overclosure


def _match_calculix_row(query: Row, rows: list[Row], field: str) -> Row | None:
    if not rows:
        return None
    if query["kind"] == "slave_node":
        node = query["slave_node"]
        for row in rows:
            if row.get("slave_node") == node:
                return row
    else:
        element_id = query["slave_element"]
        face = query["slave_face"]
        for row in rows:
            if row.get("slave_element") == element_id and row.get("slave_face") == face:
                return row
    return rows[0] if field else None


def replay_case(case: LawCase, *, text: str, dat_text: str, out_dir: Path) -> tuple[Row, list[Row]]:
    model = parse_official_contact_model(text)
    U, RF, cdis_rows, cstr_rows, cels_rows, total_cels = parse_calculix_contact_dat(dat_text, model.node_ids)
    X_current = model.X + U
    master_quads = _surface_quads(model, model.contact_master_surface, X_current)
    master_faces = _triangles_from_quads(master_quads)
    if master_faces.size == 0:
        raise ValueError(f"{case.case_id} does not define an element master surface")
    candidates = np.arange(master_faces.shape[0], dtype=np.int64)
    contact_rows: list[Row] = []
    force = np.zeros(3, dtype=float)
    energy = 0.0
    gap_errors: list[float] = []
    pressure_errors: list[float] = []

    for index, query in enumerate(_query_points(model, X_current)):
        result = dynamic_surface_sdf(np.asarray(query["x"], dtype=float), X_current, master_faces, candidates)
        sfc_pressure = _pressure_from_gap(model.pressure_law, model.pressure_values, float(result.g))
        area = float(query["area_weight"])
        row_energy = _unit_area_energy(model.pressure_law, model.pressure_values, float(result.g), sfc_pressure) * area
        row_force = sfc_pressure * area * result.n
        force += row_force
        energy += row_energy
        cdis = _match_calculix_row(query, cdis_rows, "cdis")
        cstr = _match_calculix_row(query, cstr_rows, "cstr")
        cels = _match_calculix_row(query, cels_rows, "cels")
        calc_gap = "" if cdis is None else float(cdis["normal"])
        calc_pressure = "" if cstr is None else float(cstr["normal"])
        calc_energy = "" if cels is None else float(cels["normal"])
        if calc_gap != "":
            gap_errors.append(abs(float(result.g) - float(calc_gap)))
        if calc_pressure != "":
            pressure_errors.append(abs(float(sfc_pressure) - float(calc_pressure)) / max(abs(float(calc_pressure)), 1.0e-30))
        contact_rows.append(
            {
                "case_id": case.case_id,
                "contact_index": index,
                "query_kind": query["kind"],
                "slave_node": query["slave_node"],
                "slave_element": query["slave_element"],
                "slave_face": query["slave_face"],
                "x": float(query["x"][0]),
                "y": float(query["x"][1]),
                "z": float(query["x"][2]),
                "sfc_gap": float(result.g),
                "calculix_cdis": calc_gap,
                "gap_abs_error": "" if calc_gap == "" else abs(float(result.g) - float(calc_gap)),
                "normal_x": float(result.n[0]),
                "normal_y": float(result.n[1]),
                "normal_z": float(result.n[2]),
                "closest_face_id": int(result.face_id),
                "pressure_law": model.pressure_law,
                "pressure_values": ";".join(str(value) for value in model.pressure_values),
                "sfc_pressure": float(sfc_pressure),
                "calculix_cstr": calc_pressure,
                "pressure_rel_error": "" if calc_pressure == "" else abs(float(sfc_pressure) - float(calc_pressure)) / max(abs(float(calc_pressure)), 1.0e-30),
                "area_weight": area,
                "sfc_force_x": float(row_force[0]),
                "sfc_force_y": float(row_force[1]),
                "sfc_force_z": float(row_force[2]),
                "sfc_energy": float(row_energy),
                "calculix_cels": calc_energy,
                "energy_note": "area_weighted_sfc_density; node-to-surface CalculiX spring area may differ",
            }
        )

    applied_force_norm = float(np.linalg.norm(model.cload_total))
    rf_norm = float(np.linalg.norm(np.sum(RF, axis=0))) if np.any(RF) else ""
    gap_error_max = max(gap_errors) if gap_errors else ""
    pressure_error_max = max(pressure_errors) if pressure_errors else ""
    total_energy_reference = total_cels if total_cels is not None else ""
    total_energy_error = "" if total_cels in {None, 0.0} else abs(float(energy) - float(total_cels)) / max(abs(float(total_cels)), 1.0e-30)
    status = "ok"
    if gap_error_max != "" and float(gap_error_max) > 1.0e-7:
        status = "check"
    if model.pressure_law == "LINEAR" and pressure_error_max != "" and float(pressure_error_max) > 5.0e-2:
        status = "check"
    summary = {
        "case_id": case.case_id,
        "source_file": case.file_name,
        "scope": case.scope,
        "contact_pair": f"{model.contact_slave_surface}->{model.contact_master_surface}",
        "pressure_law": model.pressure_law,
        "pressure_values": ";".join(str(value) for value in model.pressure_values),
        "node_count": model.X.shape[0],
        "element_count": model.elements.shape[0],
        "master_triangle_count": int(master_faces.shape[0]),
        "sfc_contact_point_count": len(contact_rows),
        "calculix_cdis_count": len(cdis_rows),
        "calculix_cstr_count": len(cstr_rows),
        "calculix_cels_count": len(cels_rows),
        "max_gap_abs_error": gap_error_max,
        "max_pressure_rel_error": pressure_error_max,
        "sfc_total_force_norm": float(np.linalg.norm(force)),
        "applied_force_norm": applied_force_norm,
        "calculix_rf_total_norm": rf_norm,
        "sfc_total_contact_energy": float(energy),
        "calculix_total_cels": total_energy_reference,
        "total_energy_rel_error": total_energy_error,
        "status": status,
    }
    return summary, contact_rows


def _run_instrumented_calculix(case: LawCase, text: str, out_dir: Path, skip_calculix: bool, timeout: int) -> tuple[Path, Path, Row]:
    run_dir = out_dir / "calculix_runs" / case.case_id
    run_dir.mkdir(parents=True, exist_ok=True)
    job = case.file_name.replace(".inp", "").replace(".gz", "") + "_instrumented"
    inp_path = run_dir / f"{job}.inp"
    inp_path.write_text(_instrument_contact_output(text), encoding="utf-8")
    command_row: Row = {
        "case_id": case.case_id,
        "external_solver": "CalculiX",
        "external_solver_version": _calculix_version(),
        "command": "",
        "run_status": "not_run",
        "wall_time_seconds": "",
        "inp": str(inp_path.relative_to(out_dir)),
        "dat": "",
    }
    dat_path = run_dir / f"{job}.dat"
    if skip_calculix:
        return inp_path, dat_path, command_row
    if not _calculix_available():
        command_row["run_status"] = "calculix_unavailable"
        return inp_path, dat_path, command_row
    wsl_run_dir = _wsl_path(run_dir)
    command = f"cd {_wsl_quote(wsl_run_dir)} && ccx {_wsl_quote(job)}"
    start = time.perf_counter()
    proc = _run(["wsl", "--exec", "bash", "-lc", command], cwd=ROOT, timeout=timeout)
    wall = time.perf_counter() - start
    (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
    command_row.update(
        {
            "command": f"wsl --exec bash -lc \"{command}\"",
            "run_status": "ok" if proc.returncode == 0 and dat_path.exists() else "failed",
            "return_code": proc.returncode,
            "wall_time_seconds": wall,
            "dat": str(dat_path.relative_to(out_dir)) if dat_path.exists() else "",
        }
    )
    return inp_path, dat_path, command_row


def _write_plots(out_dir: Path, summaries: list[Row]) -> list[Row]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = [row["case_id"].replace("_", "\n") for row in summaries]
    gap = [float(row["max_gap_abs_error"]) if row["max_gap_abs_error"] != "" else np.nan for row in summaries]
    pressure = [float(row["max_pressure_rel_error"]) if row["max_pressure_rel_error"] != "" else np.nan for row in summaries]
    x = np.arange(len(labels), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    axes[0].bar(x, gap, color="#4c78a8")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("max |SFC gap - CDIS|")
    axes[0].grid(True, axis="y", alpha=0.25, which="both")
    axes[1].bar(x, pressure, color="#f58518")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("max pressure relative error")
    axes[1].grid(True, axis="y", alpha=0.25, which="both")
    fig.tight_layout()
    png = figures / "official_law_sdf_replay_errors.png"
    pdf = figures / "official_law_sdf_replay_errors.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)

    force = [float(row["sfc_total_force_norm"]) for row in summaries]
    applied = [float(row["applied_force_norm"]) for row in summaries]
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    width = 0.35
    ax.bar(x - width / 2.0, force, width, label="SFC replay force norm")
    ax.bar(x + width / 2.0, applied, width, label="applied load norm")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("force norm")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    force_png = figures / "official_law_sdf_replay_force_balance.png"
    force_pdf = figures / "official_law_sdf_replay_force_balance.pdf"
    fig.savefig(force_png, dpi=180)
    fig.savefig(force_pdf)
    plt.close(fig)
    return [
        {
            "plot": "official_law_sdf_replay_errors",
            "png": str(png.relative_to(out_dir)),
            "pdf": str(pdf.relative_to(out_dir)),
            "description": "gap/CDIS and pressure/CSTR replay errors on CalculiX-deformed official examples",
        },
        {
            "plot": "official_law_sdf_replay_force_balance",
            "png": str(force_png.relative_to(out_dir)),
            "pdf": str(force_pdf.relative_to(out_dir)),
            "description": "SFC replay contact force norm against applied load norm",
        },
    ]


def _claim_gates(summaries: list[Row]) -> list[Row]:
    gates: list[Row] = []
    for row in summaries:
        gap_ok = row["max_gap_abs_error"] != "" and float(row["max_gap_abs_error"]) <= 1.0e-7
        pressure_ok = row["max_pressure_rel_error"] != "" and float(row["max_pressure_rel_error"]) <= 5.0e-2
        linear_pressure_ok = row["pressure_law"] == "LINEAR" and pressure_ok
        energy_ok = row["case_id"] == "contactenergy_c3d8_surface_linear" and row["total_energy_rel_error"] != "" and float(row["total_energy_rel_error"]) <= 1.0e-5
        gates.append(
            {
                "case_id": row["case_id"],
                "claim": "deforming_sdf_gap_matches_calculix_cdis",
                "allowed": str(gap_ok).lower(),
                "reason": "requires instrumented CalculiX CDIS and SFC current-surface gap agreement",
            }
        )
        gates.append(
            {
                "case_id": row["case_id"],
                "claim": "pressure_law_matches_calculix_cstr",
                "allowed": str(pressure_ok).lower(),
                "reason": "requires pressure-overclosure replay and CalculiX CSTR agreement",
            }
        )
        gates.append(
            {
                "case_id": row["case_id"],
                "claim": "linear_pressure_law_matches_calculix_cstr",
                "allowed": str(linear_pressure_ok).lower(),
                "reason": "allowed only for linear pressure-overclosure rows with CSTR agreement",
            }
        )
        gates.append(
            {
                "case_id": row["case_id"],
                "claim": "contact_energy_matches_calculix_cels",
                "allowed": str(energy_ok).lower(),
                "reason": "currently only the official contactenergy surface-to-surface case has a locked total CELS match",
            }
        )
        gates.append(
            {
                "case_id": row["case_id"],
                "claim": "native_sfc_trajectory_equivalence",
                "allowed": "false",
                "reason": "this runner replays CalculiX deformed states; it does not run a native SFC trajectory",
            }
        )
    return gates


def _write_summary(out_dir: Path, summaries: list[Row], gates: list[Row]) -> None:
    lines = [
        "# Official CalculiX Contact Law + Deforming-SDF Replay",
        "",
        "CalculiX solves the selected official C3D8 contact examples. SFC then",
        "replays the final deformed configuration by querying the dynamic SDF on",
        "the current master surface and evaluating the pressure-overclosure law.",
        "",
        "## Metrics",
        "",
        "| Case | Law | contact points | max gap error | max pressure error | status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['case_id']} | {row['pressure_law']} | {row['sfc_contact_point_count']} | "
            f"{row['max_gap_abs_error']} | {row['max_pressure_rel_error']} | {row['status']} |"
        )
    lines.extend(["", "## Claim Gates", "", "| Case | Claim | Allowed | Reason |", "| --- | --- | --- | --- |"])
    for gate in gates:
        lines.append(f"| {gate['case_id']} | {gate['claim']} | {gate['allowed']} | {gate['reason']} |")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- This is a deformed-state replay validation, not native SFC trajectory equivalence.",
            "- `contactenergy` supports the total CELS energy replay claim.",
            "- `contact3` and `contact6` support linear pressure-overclosure diagnostics.",
            "- `contact1` is retained as an exponential-law diagnostic; the gap comparison is meaningful, but full exponential energy equivalence is not claimed.",
        ]
    )
    (out_dir / "official_law_sdf_replay_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_official_law_replay(
    *,
    out_dir: Path,
    examples_dir: Path | None,
    wsl_examples_dir: str,
    skip_calculix: bool,
    timeout: int,
) -> dict[str, list[Row]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[Row] = []
    contact_rows: list[Row] = []
    commands: list[Row] = []
    for case in LAW_CASES:
        text, source = _read_example(case.file_name, examples_dir, wsl_examples_dir)
        copied_name = case.file_name[:-3] if case.file_name.endswith(".gz") else case.file_name
        (inputs_dir / copied_name).write_text(text, encoding="utf-8")
        _inp_path, dat_path, command_row = _run_instrumented_calculix(
            case,
            text,
            out_dir,
            skip_calculix=skip_calculix,
            timeout=timeout,
        )
        command_row["input_source"] = source
        commands.append(command_row)
        if not dat_path.exists():
            continue
        summary, rows = replay_case(case, text=_instrument_contact_output(text), dat_text=dat_path.read_text(encoding="utf-8", errors="ignore"), out_dir=out_dir)
        summaries.append(summary)
        contact_rows.extend(rows)

    if not summaries:
        raise RuntimeError("no official contact-law replay summaries were generated")
    gates = _claim_gates(summaries)
    plots = _write_plots(out_dir, summaries)
    _write_csv(out_dir / "official_law_sdf_replay_summary.csv", list(summaries[0].keys()), summaries)
    _write_csv(out_dir / "official_law_sdf_replay_contact_points.csv", list(contact_rows[0].keys()), contact_rows)
    _write_csv(out_dir / "official_law_sdf_replay_claim_gates.csv", ["case_id", "claim", "allowed", "reason"], gates)
    _write_csv(out_dir / "official_law_sdf_replay_commands.csv", list(commands[0].keys()), commands)
    _write_csv(out_dir / "official_law_sdf_replay_plots.csv", list(plots[0].keys()), plots)
    _write_summary(out_dir, summaries, gates)
    return {"summaries": summaries, "contact_points": contact_rows, "claim_gates": gates, "commands": commands, "plots": plots}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Accepted for consistency; selected official cases are small.")
    parser.add_argument("--skip-calculix", action="store_true")
    parser.add_argument("--examples-dir", type=Path, default=None)
    parser.add_argument("--wsl-examples-dir", default=DEFAULT_WSL_EXAMPLES_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "calculix_official_law_sdf_replay")
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_official_law_replay(
        out_dir=Path(args.out_dir),
        examples_dir=args.examples_dir,
        wsl_examples_dir=str(args.wsl_examples_dir),
        skip_calculix=bool(args.skip_calculix),
        timeout=int(args.timeout),
    )
    print(f"Wrote official CalculiX law/SDF replay outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
