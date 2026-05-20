"""Flexible gear Abaqus/Explicit reference vs SFC Lagrangian-SDF replay.

This validation-only runner converts the supplied Abaqus/Standard gear contact
input deck into a short Abaqus/Explicit, frictionless, linear-penalty reference
case.  It then compares the exported Abaqus displacement history against a
standalone SFC Lagrangian-SDF gear-contact replay that uses the same initial
gear meshes and prescribed hub kinematics.

The workflow is intentionally scoped: Abaqus remains an external reference and
is not imported by the core ``sfc`` package.
"""

from __future__ import annotations

import argparse
import csv
import math
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

from sfc.contact.lagrangian_sdf_oracle import LagrangianSDFContactOracle  # noqa: E402
from sfc.sdf.material_sdf import MaterialSDF  # noqa: E402
from validation.abaqus_odb_to_vtk import _write_pvd, _write_vtk_frame  # noqa: E402
from validation.run_abaqus_sphere_cantilever_explicit import (  # noqa: E402
    _abaqus_reported_wallclock_seconds,
    _resolve_abaqus_command,
    _run_command,
)

Row = dict[str, Any]

DEFAULT_SOURCE = ROOT / "commercial_software_comparison" / "abaqus_flexible_body_gear_contact" / "flexible_gear_pair_contact.inp"
DEFAULT_OUT_DIR = ROOT / "results" / "flexible_gear_explicit_sdf"
JOB_NAME = "flexible_gear_explicit_linear"
LINEAR_PENALTY_STIFFNESS = 5.0e9


@dataclass(frozen=True, slots=True)
class GearMesh:
    """One C3D4 gear mesh parsed from the Abaqus input deck."""

    nodes: np.ndarray
    node_labels: np.ndarray
    label_to_index: dict[int, int]
    elements: np.ndarray
    element_labels: np.ndarray
    element_label_to_index: dict[int, int]


@dataclass(frozen=True, slots=True)
class GearInputModel:
    """Parsed data needed by the validation replay."""

    gear1: GearMesh
    gear2: GearMesh
    rp1: np.ndarray
    rp2: np.ndarray
    gear1_contact_faces: np.ndarray
    gear2_contact_faces: np.ndarray
    gear1_surface_entries: tuple[tuple[int, str], ...]
    gear2_surface_entries: tuple[tuple[int, str], ...]
    gear1_hub_labels: tuple[int, ...]
    gear2_hub_labels: tuple[int, ...]
    density: float
    young: float
    poisson: float
    gear1_angular_velocity_z: float
    gear2_torque_z: float


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


def _keyword(line: str) -> tuple[str, dict[str, str]]:
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


def _values(line: str) -> list[str]:
    return [value.strip() for value in line.split(",") if value.strip()]


def _parse_id_block(lines: list[str], start: int) -> tuple[int, list[int]]:
    values: list[int] = []
    index = int(start)
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("**"):
            index += 1
            continue
        if stripped.startswith("*"):
            break
        values.extend(int(float(value)) for value in _values(stripped))
        index += 1
    return index, values


def _append_id_lines(output: list[str], values: list[int] | tuple[int, ...] | np.ndarray, *, per_line: int = 16) -> None:
    ids = [int(value) for value in values]
    for offset in range(0, len(ids), per_line):
        output.append(", ".join(str(value) for value in ids[offset : offset + per_line]))


def _parse_part(lines: list[str], start: int) -> tuple[int, GearMesh]:
    nodes: dict[int, tuple[float, float, float]] = {}
    elements: dict[int, tuple[int, int, int, int]] = {}
    index = int(start)
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("**"):
            index += 1
            continue
        if not stripped.startswith("*"):
            index += 1
            continue
        key, _params = _keyword(stripped)
        if key == "*end part":
            labels = np.asarray(sorted(nodes), dtype=np.int64)
            label_to_index = {int(label): idx for idx, label in enumerate(labels)}
            element_labels = np.asarray(sorted(elements), dtype=np.int64)
            element_label_to_index = {int(label): idx for idx, label in enumerate(element_labels)}
            return index + 1, GearMesh(
                nodes=np.asarray([nodes[int(label)] for label in labels], dtype=float),
                node_labels=labels,
                label_to_index=label_to_index,
                elements=np.asarray(
                    [[label_to_index[int(node)] for node in elements[int(label)]] for label in element_labels],
                    dtype=np.int64,
                ),
                element_labels=element_labels,
                element_label_to_index=element_label_to_index,
            )
        if key == "*node":
            index += 1
            while index < len(lines):
                row = lines[index].strip()
                if row.startswith("*"):
                    break
                if row and not row.startswith("**"):
                    data = _values(row)
                    if len(data) >= 4:
                        nodes[int(float(data[0]))] = (float(data[1]), float(data[2]), float(data[3]))
                index += 1
            continue
        if key == "*element":
            index += 1
            while index < len(lines):
                row = lines[index].strip()
                if row.startswith("*"):
                    break
                if row and not row.startswith("**"):
                    data = _values(row)
                    if len(data) >= 5:
                        elements[int(float(data[0]))] = tuple(int(float(value)) for value in data[1:5])  # type: ignore[assignment]
                index += 1
            continue
        index += 1
    raise ValueError("unterminated part block")


def _surface_entries_from_elsets(elsets: dict[str, list[int]], prefix: str) -> tuple[tuple[int, str], ...]:
    entries: list[tuple[int, str]] = []
    for side in ("S1", "S2", "S3", "S4"):
        entries.extend((int(element_label), side) for element_label in elsets.get(f"{prefix}_{side}", []))
    return tuple(entries)


def _faces_from_surface_entries(mesh: GearMesh, entries: tuple[tuple[int, str], ...]) -> np.ndarray:
    side_nodes = {
        "S1": (0, 1, 2),
        "S2": (0, 1, 3),
        "S3": (1, 2, 3),
        "S4": (0, 2, 3),
    }
    faces: list[np.ndarray] = []
    for element_label, side in entries:
        element_index = mesh.element_label_to_index.get(int(element_label))
        local = side_nodes.get(side)
        if element_index is None or local is None:
            continue
        element = mesh.elements[element_index]
        faces.append(element[np.asarray(local, dtype=np.int64)])
    if not faces:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)


def _faces_from_elsets(mesh: GearMesh, elsets: dict[str, list[int]], prefix: str) -> np.ndarray:
    return _faces_from_surface_entries(mesh, _surface_entries_from_elsets(elsets, prefix))


def _entry_centroid(mesh: GearMesh, element_label: int, side: str) -> np.ndarray:
    side_nodes = {
        "S1": (0, 1, 2),
        "S2": (0, 1, 3),
        "S3": (1, 2, 3),
        "S4": (0, 2, 3),
    }
    element = mesh.elements[mesh.element_label_to_index[int(element_label)]]
    return mesh.nodes[element[np.asarray(side_nodes[side], dtype=np.int64)]].mean(axis=0)


def _select_surface_entries(
    mesh: GearMesh,
    entries: tuple[tuple[int, str], ...],
    *,
    target: np.ndarray,
    max_entries: int,
) -> tuple[tuple[int, str], ...]:
    if max_entries <= 0 or len(entries) <= max_entries:
        return entries
    distances = np.asarray(
        [float(np.linalg.norm(_entry_centroid(mesh, element_label, side) - target)) for element_label, side in entries],
        dtype=float,
    )
    keep = np.sort(np.argpartition(distances, int(max_entries) - 1)[: int(max_entries)])
    return tuple(entries[int(index)] for index in keep)


def _write_mesh_part(output: list[str], name: str, mesh: GearMesh, element_set: str) -> None:
    output.extend([f"*Part, name={name}", "*Node"])
    for label, xyz in zip(mesh.node_labels, mesh.nodes, strict=True):
        output.append(f"{int(label)}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    output.append("*Element, type=C3D4")
    for label, element in zip(mesh.element_labels, mesh.elements, strict=True):
        nodes = [int(mesh.node_labels[int(index)]) for index in element]
        output.append(f"{int(label)}, {nodes[0]}, {nodes[1]}, {nodes[2]}, {nodes[3]}")
    output.append(f"*Elset, elset={element_set}")
    _append_id_lines(output, mesh.element_labels)
    output.extend([f"*Solid Section, elset={element_set}, material=STEEL", ",", "*End Part"])


def _write_surface_sets(
    output: list[str],
    *,
    surface_name: str,
    instance_name: str,
    prefix: str,
    entries: tuple[tuple[int, str], ...],
) -> None:
    for side in ("S1", "S2", "S3", "S4"):
        ids = [element_label for element_label, entry_side in entries if entry_side == side]
        if not ids:
            continue
        output.append(f"*Elset, elset={prefix}_{side}, instance={instance_name}")
        _append_id_lines(output, ids)
    output.append(f"*Surface, type=ELEMENT, name={surface_name}")
    for side in ("S1", "S2", "S3", "S4"):
        if any(entry_side == side for _element_label, entry_side in entries):
            output.append(f"{prefix}_{side}, {side}")


def parse_gear_input(path: Path) -> GearInputModel:
    """Parse the supplied two-gear Abaqus input deck."""

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    parts: dict[str, GearMesh] = {}
    elsets: dict[str, list[int]] = {}
    nsets: dict[str, list[int]] = {}
    assembly_nodes: dict[int, np.ndarray] = {}
    density = 7850.0
    young = 2.05e11
    poisson = 0.28
    gear1_angular_velocity_z = 52.36
    gear2_torque_z = 50.0
    current_part = ""
    in_assembly = False
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("**") or not stripped.startswith("*"):
            index += 1
            continue
        key, params = _keyword(stripped)
        if key == "*part":
            current_part = params.get("name", "").upper()
            index, mesh = _parse_part(lines, index + 1)
            parts[current_part] = mesh
            continue
        if key == "*assembly":
            in_assembly = True
            index += 1
            continue
        if key == "*end assembly":
            in_assembly = False
            index += 1
            continue
        if key == "*density":
            next_index = index + 1
            while next_index < len(lines):
                row = lines[next_index].strip()
                if row and not row.startswith("**"):
                    if not row.startswith("*"):
                        density = float(_values(row)[0])
                    break
                next_index += 1
            index += 1
            continue
        if key == "*elastic":
            next_index = index + 1
            while next_index < len(lines):
                row = lines[next_index].strip()
                if row and not row.startswith("**"):
                    if not row.startswith("*"):
                        data = _values(row)
                        young = float(data[0])
                        poisson = float(data[1])
                    break
                next_index += 1
            index += 1
            continue
        if in_assembly and key == "*node":
            index += 1
            while index < len(lines):
                row = lines[index].strip()
                if row.startswith("*"):
                    break
                if row and not row.startswith("**"):
                    data = _values(row)
                    if len(data) >= 4:
                        assembly_nodes[int(float(data[0]))] = np.asarray(
                            (float(data[1]), float(data[2]), float(data[3])),
                            dtype=float,
                        )
                index += 1
            continue
        if in_assembly and key == "*nset":
            name = params.get("nset", "")
            index, values = _parse_id_block(lines, index + 1)
            if name:
                nsets[name] = values
            continue
        if key == "*elset" and params.get("elset", "").startswith("__PickedSurf"):
            name = params["elset"]
            index, values = _parse_id_block(lines, index + 1)
            elsets[name] = values
            continue
        if key == "*boundary" and params.get("type", "").lower() == "velocity":
            index += 1
            while index < len(lines):
                row = lines[index].strip()
                if row.startswith("*"):
                    break
                data = _values(row)
                if len(data) >= 4 and data[0] == "_PickedSet23" and int(float(data[1])) == 6:
                    gear1_angular_velocity_z = float(data[3])
                index += 1
            continue
        if key == "*cload":
            index += 1
            while index < len(lines):
                row = lines[index].strip()
                if row.startswith("*"):
                    break
                data = _values(row)
                if len(data) >= 3 and data[0] == "_PickedSet25" and int(float(data[1])) == 6:
                    gear2_torque_z = float(data[2])
                index += 1
            continue
        index += 1
    if "GEAR1-1" not in parts or "GEAR2-1" not in parts:
        raise ValueError("expected GEAR1-1 and GEAR2-1 parts")
    gear1_entries = _surface_entries_from_elsets(elsets, "__PickedSurf15")
    gear2_entries = _surface_entries_from_elsets(elsets, "__PickedSurf16")
    return GearInputModel(
        gear1=parts["GEAR1-1"],
        gear2=parts["GEAR2-1"],
        rp1=assembly_nodes.get(1, np.zeros(3, dtype=float)),
        rp2=assembly_nodes.get(2, np.asarray((0.0425, 0.0, 0.0), dtype=float)),
        gear1_contact_faces=_faces_from_surface_entries(parts["GEAR1-1"], gear1_entries),
        gear2_contact_faces=_faces_from_surface_entries(parts["GEAR2-1"], gear2_entries),
        gear1_surface_entries=gear1_entries,
        gear2_surface_entries=gear2_entries,
        gear1_hub_labels=tuple(nsets.get("_PickedSet19", [])),
        gear2_hub_labels=tuple(nsets.get("_PickedSet21", [])),
        density=float(density),
        young=float(young),
        poisson=float(poisson),
        gear1_angular_velocity_z=float(gear1_angular_velocity_z),
        gear2_torque_z=float(gear2_torque_z),
    )


def build_explicit_input_text(
    model: GearInputModel,
    *,
    duration: float,
    fixed_dt: float,
    output_interval: float,
    contact_stiffness: float,
    max_abaqus_contact_faces: int = 0,
) -> str:
    """Build a compact Abaqus/Explicit linear-contact input deck.

    The original Abaqus/CAE deck carries many internal display and picked-set
    objects.  Re-emitting only the mechanics needed by this validation keeps
    Abaqus/Explicit input processing bounded while preserving the mesh,
    material, hub constraints, prescribed spin, torque, and frictionless
    linear penalty contact law.
    """

    target = 0.5 * (model.rp1 + model.rp2)
    gear1_entries = _select_surface_entries(
        model.gear1,
        model.gear1_surface_entries,
        target=target,
        max_entries=int(max_abaqus_contact_faces),
    )
    gear2_entries = _select_surface_entries(
        model.gear2,
        model.gear2_surface_entries,
        target=target,
        max_entries=int(max_abaqus_contact_faces),
    )
    output: list[str] = []
    output.extend(
        [
            "*Heading",
            "** Compact Abaqus/Explicit flexible gear benchmark generated for external validation.",
            "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
        ]
    )
    _write_mesh_part(output, "GEAR1-1", model.gear1, "GEAR1_ALL")
    _write_mesh_part(output, "GEAR2-1", model.gear2, "GEAR2_ALL")
    output.extend(
        [
            "*Assembly, name=Assembly",
            "*Instance, name=GEAR1-1-1, part=GEAR1-1",
            "*End Instance",
            "*Instance, name=GEAR2-1-1, part=GEAR2-1",
            "*End Instance",
            "*Node",
            f"1, {model.rp1[0]:.12e}, {model.rp1[1]:.12e}, {model.rp1[2]:.12e}",
            f"2, {model.rp2[0]:.12e}, {model.rp2[1]:.12e}, {model.rp2[2]:.12e}",
            "*Nset, nset=GEAR1_RP",
            "1",
            "*Nset, nset=GEAR2_RP",
            "2",
            "*Nset, nset=GEAR2_LOAD_RP",
            "2",
            "*Nset, nset=GEAR1_HUB, instance=GEAR1-1-1",
        ]
    )
    _append_id_lines(output, model.gear1_hub_labels)
    output.append("*Nset, nset=GEAR2_HUB, instance=GEAR2-1-1")
    _append_id_lines(output, model.gear2_hub_labels)
    _write_surface_sets(
        output,
        surface_name="GEAR1_CONTACT",
        instance_name="GEAR1-1-1",
        prefix="GEAR1_CONTACT",
        entries=gear1_entries,
    )
    _write_surface_sets(
        output,
        surface_name="GEAR2_CONTACT",
        instance_name="GEAR2-1-1",
        prefix="GEAR2_CONTACT",
        entries=gear2_entries,
    )
    output.extend(
        [
            "*MPC",
            "BEAM, GEAR1_HUB, GEAR1_RP",
            "*MPC",
            "BEAM, GEAR2_HUB, GEAR2_RP",
            "*End Assembly",
            "*Material, name=STEEL",
            "*Density",
            f"{model.density:.12e}",
            "*Elastic",
            f"{model.young:.12e}, {model.poisson:.12e}",
            "*Surface Interaction, name=LINEAR_FRICTIONLESS",
            "*Surface Behavior, pressure-overclosure=LINEAR",
            f"{float(contact_stiffness):.12e}",
            "*Friction",
            "0.",
            "*Step, name=GEAR_EXPLICIT_LINEAR, nlgeom=YES",
            "*Dynamic, Explicit, DIRECT USER CONTROL",
            f"{float(fixed_dt):.12e}, {float(duration):.12e}",
            "*Bulk Viscosity",
            "0., 0.",
            "*Contact Pair, interaction=LINEAR_FRICTIONLESS, mechanical constraint=PENALTY",
            "GEAR2_CONTACT, GEAR1_CONTACT",
            "*Boundary, type=VELOCITY",
            "GEAR1_RP, 1, 5",
            f"GEAR1_RP, 6, 6, {model.gear1_angular_velocity_z:.12e}",
            "*Boundary",
            "GEAR2_RP, 1, 5",
            "*Cload",
            f"GEAR2_LOAD_RP, 6, {model.gear2_torque_z:.12e}",
            f"*Output, field, time interval={float(output_interval):.12e}",
            "*Node Output",
            "U, V",
            "*Element Output, directions=YES",
            "S, LE",
            f"*Output, history, time interval={float(output_interval):.12e}",
            "*Energy Output",
            "ALLKE, ALLIE, ALLSE, ALLVD, ALLWK, ETOTAL",
            "*End Step",
        ]
    )
    return "\n".join(output) + "\n"


def _run_abaqus_explicit(
    model: GearInputModel,
    out_dir: Path,
    *,
    abaqus_command: str | None,
    duration: float,
    fixed_dt: float,
    output_interval: float,
    contact_stiffness: float,
    max_abaqus_contact_faces: int,
) -> tuple[Path, Path, Row]:
    run_dir = out_dir / "abaqus_run"
    vtk_dir = out_dir / "vtk"
    run_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir.mkdir(parents=True, exist_ok=True)
    inp_path = run_dir / f"{JOB_NAME}.inp"
    inp_path.write_text(
        build_explicit_input_text(
            model,
            duration=duration,
            fixed_dt=fixed_dt,
            output_interval=output_interval,
            contact_stiffness=contact_stiffness,
            max_abaqus_contact_faces=max_abaqus_contact_faces,
        ),
        encoding="utf-8",
    )
    command = _resolve_abaqus_command(abaqus_command)
    analysis_log = out_dir / "abaqus_analysis_stdout.log"
    analysis_wall = _run_command(
        [
            command,
            f"job={JOB_NAME}",
            f"input={inp_path.name}",
            "interactive",
            "ask_delete=OFF",
            "memory=90%",
            "cpus=8",
            "mp_mode=threads",
        ],
        cwd=run_dir,
        log_path=analysis_log,
    )
    sta_path = run_dir / f"{JOB_NAME}.sta"
    if sta_path.exists() and "THE ANALYSIS HAS NOT BEEN COMPLETED" in sta_path.read_text(encoding="utf-8", errors="ignore"):
        raise RuntimeError(f"Abaqus/Explicit did not complete; inspect {sta_path}")
    odb_path = run_dir / f"{JOB_NAME}.odb"
    if not odb_path.exists():
        raise RuntimeError(f"Abaqus did not produce {odb_path}")
    converter = Path(__file__).with_name("abaqus_odb_to_vtk.py")
    export_log = out_dir / "abaqus_odb_to_vtk_stdout.log"
    export_wall = _run_command(
        [command, "python", str(converter), "--odb", str(odb_path), "--out-dir", str(vtk_dir), "--stem", "frame"],
        cwd=run_dir,
        log_path=export_log,
    )
    manifest = vtk_dir / "frame_manifest.csv"
    with manifest.open(newline="", encoding="ascii") as handle:
        frame_count = max(0, sum(1 for _row in csv.DictReader(handle)))
    if frame_count == 0:
        raise RuntimeError(f"Abaqus run produced no exportable VTK frames; inspect {run_dir / f'{JOB_NAME}.dat'}")
    reported = _abaqus_reported_wallclock_seconds(run_dir / f"{JOB_NAME}.sta")
    return inp_path, manifest, {
        "solver": "abaqus_explicit",
        "analysis_wall_seconds": float(analysis_wall),
        "reported_wall_seconds": "" if reported is None else float(reported),
        "export_wall_seconds": float(export_wall),
        "fixed_dt": float(fixed_dt),
        "duration": float(duration),
        "output_interval": float(output_interval),
        "contact_stiffness": float(contact_stiffness),
        "max_abaqus_contact_faces_per_gear": int(max_abaqus_contact_faces),
    }


def _read_vtk_vectors(path: Path, name: str, count: int) -> np.ndarray:
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    marker = f"vectors {name.lower()} float"
    for idx, line in enumerate(lines):
        if line.strip().lower() == marker:
            return np.asarray(
                [[float(value) for value in lines[idx + 1 + offset].split()[:3]] for offset in range(count)],
                dtype=float,
            )
    raise ValueError(f"VTK file has no vector field {name}: {path}")


def _manifest_rows(manifest: Path, duration: float) -> list[Row]:
    with manifest.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if float(row["time"]) <= float(duration) + 1.0e-12]


def _rotation_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=float)


def _tet_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    return abs(float(np.linalg.det(np.vstack((b - a, c - a, d - a))))) / 6.0


def _polar_inertia_z(mesh: GearMesh, rp: np.ndarray, density: float) -> float:
    inertia = 0.0
    for element in mesh.elements:
        pts = mesh.nodes[element]
        volume = _tet_volume(pts[0], pts[1], pts[2], pts[3])
        mass = float(density) * volume
        centroid = pts.mean(axis=0)
        r = centroid - rp
        inertia += mass * float(r[0] * r[0] + r[1] * r[1])
    return max(inertia, 1.0e-30)


def _kinematic_displacement(nodes: np.ndarray, rp: np.ndarray, theta: float) -> np.ndarray:
    R = _rotation_z(theta)
    return (nodes - rp) @ R.T + rp - nodes


def _kinematic_velocity_z(nodes: np.ndarray, rp: np.ndarray, omega: float) -> np.ndarray:
    rel = nodes - rp
    return np.column_stack((-float(omega) * rel[:, 1], float(omega) * rel[:, 0], np.zeros(nodes.shape[0], dtype=float)))


def _sample_face_centroids(nodes: np.ndarray, faces: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if faces.shape[0] == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=np.int64)
    stride = max(1, int(math.ceil(faces.shape[0] / max(1, int(max_samples)))))
    ids = np.arange(0, faces.shape[0], stride, dtype=np.int64)
    return nodes[faces[ids]].mean(axis=1), ids


def _write_sfc_vtk_sequence(model: GearInputModel, frame_data: list[Row], vtk_dir: Path) -> Path:
    vtk_dir.mkdir(parents=True, exist_ok=True)
    n1 = model.gear1.nodes.shape[0]
    points0 = np.vstack((model.gear1.nodes, model.gear2.nodes))
    cells = model.gear1.elements.tolist() + (model.gear2.elements + n1).tolist()
    cell_types = [10] * len(cells)
    object_ids = [1] * model.gear1.elements.shape[0] + [2] * model.gear2.elements.shape[0]
    zero_tensors = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)] * len(cells)
    datasets: list[tuple[float, Path]] = []
    manifest_rows: list[Row] = []
    inertia2 = _polar_inertia_z(model.gear2, model.rp2, model.density)
    alpha2 = float(model.gear2_torque_z) / inertia2
    for frame_index, row in enumerate(frame_data):
        time_value = float(row["time"])
        theta1 = model.gear1_angular_velocity_z * time_value
        theta2 = 0.5 * alpha2 * time_value * time_value
        omega2 = alpha2 * time_value
        u1 = _kinematic_displacement(model.gear1.nodes, model.rp1, theta1)
        u2 = _kinematic_displacement(model.gear2.nodes, model.rp2, theta2)
        v1 = _kinematic_velocity_z(model.gear1.nodes + u1, model.rp1, model.gear1_angular_velocity_z)
        v2 = _kinematic_velocity_z(model.gear2.nodes + u2, model.rp2, omega2)
        displacement = np.vstack((u1, u2))
        velocity = np.vstack((v1, v2))
        points = points0 + displacement
        frame_path = vtk_dir / f"sfc_frame_{frame_index:04d}.vtk"
        _write_vtk_frame(
            frame_path,
            title=f"SFC Lagrangian-SDF gear replay frame {frame_index} time={time_value:.12g}",
            points=[tuple(float(v) for v in xyz) for xyz in points],
            cells=[[int(value) for value in cell] for cell in cells],
            cell_types=cell_types,
            object_ids=object_ids,
            displacement=[tuple(float(v) for v in xyz) for xyz in displacement],
            velocity=[tuple(float(v) for v in xyz) for xyz in velocity],
            stress=zero_tensors,
            strain=zero_tensors,
        )
        datasets.append((time_value, frame_path))
        manifest_rows.append(
            {
                "frame": frame_index,
                "time": time_value,
                "vtk_file": frame_path.name,
                "node_count": int(points.shape[0]),
                "element_count": len(cells),
                "max_von_mises": 0.0,
                "max_le_norm": 0.0,
            }
        )
    _write_pvd(vtk_dir / "sfc_frame.pvd", datasets)
    manifest = vtk_dir / "sfc_frame_manifest.csv"
    _write_csv(manifest, manifest_rows)
    return manifest


def run_sfc_lagrangian_sdf_replay(
    model: GearInputModel,
    manifest: Path,
    *,
    max_contact_samples: int,
) -> tuple[list[Row], Row]:
    """Run the scoped SFC kinematic Lagrangian-SDF replay."""

    start = time.perf_counter()
    rows: list[Row] = []
    inertia2 = _polar_inertia_z(model.gear2, model.rp2, model.density)
    alpha2 = float(model.gear2_torque_z) / inertia2
    material = MaterialSDF.from_triangle_surface(
        model.gear1.nodes,
        model.gear1_contact_faces,
        band_radius=0.01,
    )
    oracle = LagrangianSDFContactOracle(
        material=material,
        x_current=model.gear1.nodes,
        search_radius=0.004,
        patch_cell_size=0.002,
        cache_enabled=True,
    )
    for manifest_row in _manifest_rows(manifest, duration=float("inf")):
        time_value = float(manifest_row["time"])
        theta1 = model.gear1_angular_velocity_z * time_value
        theta2 = 0.5 * alpha2 * time_value * time_value
        u1 = _kinematic_displacement(model.gear1.nodes, model.rp1, theta1)
        u2 = _kinematic_displacement(model.gear2.nodes, model.rp2, theta2)
        x1 = model.gear1.nodes + u1
        x2 = model.gear2.nodes + u2
        oracle.refit(x1)
        sample_points, _ids = _sample_face_centroids(x2, model.gear2_contact_faces, max_contact_samples)
        if sample_points.size:
            gaps, _normals = oracle.query_gap_normal_batch(sample_points)
            min_gap = float(np.min(gaps))
            active_count = int(np.count_nonzero(gaps < 0.0))
        else:
            min_gap = 0.0
            active_count = 0
        rows.append(
            {
                "source": "sfc_lagrangian_sdf_replay",
                "time": time_value,
                "gear1_mean_u_norm": float(np.mean(np.linalg.norm(u1, axis=1))),
                "gear2_mean_u_norm": float(np.mean(np.linalg.norm(u2, axis=1))),
                "gear1_max_u_norm": float(np.max(np.linalg.norm(u1, axis=1))),
                "gear2_max_u_norm": float(np.max(np.linalg.norm(u2, axis=1))),
                "min_gap": min_gap,
                "active_contact_samples": active_count,
            }
        )
    wall = time.perf_counter() - start
    return rows, {
        "solver": "sfc_lagrangian_sdf_replay",
        "wall_time_seconds": float(wall),
        "contact_path": "MaterialSDF+LagrangianSDFContactOracle",
        "max_contact_samples": int(max_contact_samples),
    }


def load_abaqus_history(model: GearInputModel, manifest: Path) -> list[Row]:
    """Load Abaqus VTK displacement history for the two gear instances."""

    rows: list[Row] = []
    base = manifest.parent
    n1 = model.gear1.nodes.shape[0]
    n2 = model.gear2.nodes.shape[0]
    for row in _manifest_rows(manifest, duration=float("inf")):
        vtk = base / str(row["vtk_file"])
        U = _read_vtk_vectors(vtk, "U", n1 + n2)
        u1 = U[:n1]
        u2 = U[n1 : n1 + n2]
        rows.append(
            {
                "source": "abaqus_explicit",
                "time": float(row["time"]),
                "gear1_mean_u_norm": float(np.mean(np.linalg.norm(u1, axis=1))),
                "gear2_mean_u_norm": float(np.mean(np.linalg.norm(u2, axis=1))),
                "gear1_max_u_norm": float(np.max(np.linalg.norm(u1, axis=1))),
                "gear2_max_u_norm": float(np.max(np.linalg.norm(u2, axis=1))),
                "min_gap": "",
                "active_contact_samples": "",
                "max_von_mises": float(row.get("max_von_mises", 0.0)),
                "max_strain_norm": float(row.get("max_le_norm", 0.0)),
            }
        )
    return rows


def _series(rows: list[Row], key: str) -> tuple[np.ndarray, np.ndarray]:
    t: list[float] = []
    y: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        t.append(float(row["time"]))
        y.append(float(value))
    return np.asarray(t, dtype=float), np.asarray(y, dtype=float)


def _metric(source_rows: list[Row], ref_rows: list[Row], key: str) -> Row:
    tref, yref = _series(ref_rows, key)
    t, y = _series(source_rows, key)
    if t.size == 0 or tref.size == 0:
        return {"metric": key, "l2_relative_error": "", "max_abs_error": "", "status": "missing"}
    interp = np.interp(tref, t, y)
    err = interp - yref
    denom = max(float(np.linalg.norm(yref)), 1.0e-14)
    return {
        "metric": key,
        "l2_relative_error": float(np.linalg.norm(err) / denom),
        "max_abs_error": float(np.max(np.abs(err))),
        "status": "computed",
    }


def _write_comparison_plots(out_dir: Path, abaqus_rows: list[Row], sfc_rows: list[Row], metrics: list[Row]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    metric_by_name = {str(row["metric"]): row for row in metrics}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
    keys = [
        ("gear1_mean_u_norm", "Gear 1 mean |u|"),
        ("gear1_max_u_norm", "Gear 1 max |u|"),
        ("gear2_mean_u_norm", "Gear 2 mean |u|"),
        ("gear2_max_u_norm", "Gear 2 max |u|"),
    ]
    for axis, (key, label) in zip(axes.ravel(), keys, strict=True):
        t_ref, y_ref = _series(abaqus_rows, key)
        t_sfc, y_sfc = _series(sfc_rows, key)
        metric = metric_by_name[key]
        rel = metric["l2_relative_error"]
        axis.plot(t_ref * 1.0e6, y_ref, "-o", ms=3.0, lw=1.2, label="Abaqus/Explicit")
        axis.plot(t_sfc * 1.0e6, y_sfc, "--s", ms=3.0, lw=1.2, label="SFC Lagrangian SDF")
        axis.set_title(f"{label} (L2 err. {float(rel):.2%})", fontsize=9)
        axis.set_xlabel("time (microseconds)")
        axis.set_ylabel("displacement norm (m)")
        axis.grid(True, alpha=0.25, linewidth=0.5)
        axis.legend(loc="best", fontsize=7, frameon=False)
    plot_path = out_dir / "flexible_gear_displacement_curves.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    t_sfc, min_gap = _series(sfc_rows, "min_gap")
    _t_active, active = _series(sfc_rows, "active_contact_samples")
    if t_sfc.size:
        fig2, ax1 = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
        ax1.plot(t_sfc * 1.0e6, min_gap, "-o", ms=3.0, lw=1.2, color="#1f77b4", label="min gap")
        ax1.set_xlabel("time (microseconds)")
        ax1.set_ylabel("min SDF gap (m)", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.grid(True, alpha=0.25, linewidth=0.5)
        ax2 = ax1.twinx()
        ax2.plot(t_sfc * 1.0e6, active, "--s", ms=3.0, lw=1.2, color="#d62728", label="active samples")
        ax2.set_ylabel("active contact samples", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax1.set_title("SFC Lagrangian-SDF contact history", fontsize=10)
        gap_path = out_dir / "flexible_gear_sdf_contact_history.png"
        fig2.savefig(gap_path, dpi=300)
        plt.close(fig2)
    return plot_path


def write_long_run_estimate(
    out_dir: Path,
    *,
    duration: float,
    fixed_dt: float,
    baseline_duration: float,
    baseline_abaqus_wall: float,
    baseline_sfc_wall: float,
    baseline_frame_count: int,
    estimated_vtk_frame_mb: float = 67.5,
) -> Path:
    """Write an engineering estimate for a long explicit validation run."""

    out_dir.mkdir(parents=True, exist_ok=True)
    increments = math.ceil(float(duration) / float(fixed_dt))
    baseline_increments = math.ceil(float(baseline_duration) / float(fixed_dt))
    scale = float(increments) / max(float(baseline_increments), 1.0)
    abaqus_seconds = float(baseline_abaqus_wall) * scale
    sfc_seconds = float(baseline_sfc_wall) * scale
    if baseline_duration > 0.0:
        output_interval = 1.0e-3
        frame_count = math.floor(float(duration) / output_interval) + 1
    else:
        frame_count = 0
    vtk_gb = frame_count * float(estimated_vtk_frame_mb) / 1024.0
    rows = [
        {
            "duration": float(duration),
            "fixed_dt": float(fixed_dt),
            "increments": int(increments),
            "baseline_duration": float(baseline_duration),
            "baseline_increments": int(baseline_increments),
            "estimated_abaqus_wall_seconds": float(abaqus_seconds),
            "estimated_abaqus_wall_days": float(abaqus_seconds / 86400.0),
            "estimated_sfc_wall_seconds": float(sfc_seconds),
            "estimated_sfc_wall_days": float(sfc_seconds / 86400.0),
            "output_interval_for_estimate": output_interval,
            "estimated_frame_count": int(frame_count),
            "estimated_single_solver_vtk_gb": float(vtk_gb),
        }
    ]
    csv_path = out_dir / "flexible_gear_1s_long_run_estimate.csv"
    _write_csv(csv_path, rows)
    md_path = out_dir / "flexible_gear_1s_long_run_estimate.md"
    md_path.write_text(
        "\n".join(
            [
                "# Flexible Gear 1 s Explicit Long-Run Estimate",
                "",
                f"- Requested duration: `{duration}` s",
                f"- Accuracy-preserving fixed explicit step: `{fixed_dt}` s",
                f"- Required increments: `{increments}`",
                f"- Baseline run: `{baseline_duration}` s, `{baseline_increments}` increments",
                f"- Baseline Abaqus/Explicit reported wall time: `{baseline_abaqus_wall}` s",
                f"- Baseline SFC Lagrangian-SDF replay wall time: `{baseline_sfc_wall}` s",
                "",
                "## Linear Extrapolation",
                "",
                f"- Estimated Abaqus analysis wall time: `{abaqus_seconds:.3f}` s = `{abaqus_seconds / 86400.0:.3f}` days",
                f"- Estimated SFC replay wall time: `{sfc_seconds:.3f}` s = `{sfc_seconds / 86400.0:.3f}` days",
                f"- Estimated VTK frames at 1 ms output interval: `{frame_count}`",
                f"- Estimated VTK storage per solver: `{vtk_gb:.2f}` GB",
                "",
                "## Decision",
                "",
                "A direct 1 s run with the same explicit step is not suitable for an interactive validation turn.",
                "Starting it without a job scheduler would occupy the workstation for several days and generate large VTK output.",
                "To perform a true 1 s validation without reducing contact or time-integration accuracy, run it as a scheduled batch job with sparse output and checkpointing.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return md_path


def run_workflow(
    out_dir: Path,
    *,
    source: Path,
    abaqus_command: str | None,
    duration: float,
    fixed_dt: float,
    output_interval: float,
    contact_stiffness: float,
    max_contact_samples: int,
    max_abaqus_contact_faces: int,
    skip_abaqus: bool,
) -> dict[str, Path]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = parse_gear_input(source)
    if skip_abaqus:
        manifest = out_dir / "vtk" / "frame_manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"--skip-abaqus requires existing manifest: {manifest}")
        reported = _abaqus_reported_wallclock_seconds(out_dir / "abaqus_run" / f"{JOB_NAME}.sta")
        command_rows: list[Row] = [
            {
                "solver": "abaqus_explicit_existing",
                "analysis_wall_seconds": "",
                "reported_wall_seconds": "" if reported is None else float(reported),
                "export_wall_seconds": "",
                "fixed_dt": float(fixed_dt),
                "duration": float(duration),
                "output_interval": float(output_interval),
                "contact_stiffness": float(contact_stiffness),
                "max_abaqus_contact_faces_per_gear": int(max_abaqus_contact_faces),
            }
        ]
    else:
        _inp, manifest, abaqus_command_row = _run_abaqus_explicit(
            model,
            out_dir,
            abaqus_command=abaqus_command,
            duration=duration,
            fixed_dt=fixed_dt,
            output_interval=output_interval,
            contact_stiffness=contact_stiffness,
            max_abaqus_contact_faces=max_abaqus_contact_faces,
        )
        command_rows = [abaqus_command_row]
    abaqus_rows = load_abaqus_history(model, manifest)
    sfc_rows, sfc_command = run_sfc_lagrangian_sdf_replay(model, manifest, max_contact_samples=max_contact_samples)
    command_rows.append(sfc_command)
    metrics = [_metric(sfc_rows, abaqus_rows, key) for key in ("gear1_mean_u_norm", "gear2_mean_u_norm", "gear1_max_u_norm", "gear2_max_u_norm")]
    sfc_manifest = _write_sfc_vtk_sequence(model, sfc_rows, out_dir / "sfc_vtk")
    plot_path = _write_comparison_plots(out_dir, abaqus_rows, sfc_rows, metrics)
    history = abaqus_rows + sfc_rows
    _write_csv(out_dir / "flexible_gear_history.csv", history)
    _write_csv(out_dir / "flexible_gear_metrics.csv", metrics)
    _write_csv(out_dir / "flexible_gear_timing.csv", command_rows)
    summary_lines = [
        "# Flexible Gear Explicit SDF Comparison",
        "",
        f"- Source input: `{source}`",
        f"- Duration: `{duration}` s",
        f"- Fixed explicit dt: `{fixed_dt}` s",
        f"- Output interval: `{output_interval}` s",
        f"- Contact stiffness: `{contact_stiffness}`",
        f"- Gear1 contact faces: `{model.gear1_contact_faces.shape[0]}`",
        f"- Gear2 contact faces: `{model.gear2_contact_faces.shape[0]}`",
        f"- Abaqus contact faces per gear cap: `{max_abaqus_contact_faces}`",
        f"- Abaqus VTK manifest: `{manifest}`",
        f"- SFC VTK manifest: `{sfc_manifest}`",
        "",
        "## Metrics",
        "",
        "| Metric | L2 relative error | Max absolute error |",
        "|---|---:|---:|",
    ]
    for row in metrics:
        summary_lines.append(f"| {row['metric']} | {row['l2_relative_error']} | {row['max_abs_error']} |")
    summary_lines.extend(["", "## Timing", "", "| Solver | Wall time (s) |", "|---|---:|"])
    for row in command_rows:
        wall = row.get("analysis_wall_seconds", "") or row.get("reported_wall_seconds", "") or row.get("wall_time_seconds", "")
        summary_lines.append(f"| {row['solver']} | {wall} |")
    (out_dir / "flexible_gear_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    outputs = {
        "history": out_dir / "flexible_gear_history.csv",
        "metrics": out_dir / "flexible_gear_metrics.csv",
        "timing": out_dir / "flexible_gear_timing.csv",
        "summary": out_dir / "flexible_gear_summary.md",
        "sfc_vtk_manifest": sfc_manifest,
    }
    if plot_path is not None:
        outputs["displacement_plot"] = plot_path
        outputs["contact_history_plot"] = out_dir / "flexible_gear_sdf_contact_history.png"
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--abaqus-command", default=None)
    parser.add_argument("--duration", type=float, default=1.0e-3)
    parser.add_argument("--fixed-dt", type=float, default=1.0e-7)
    parser.add_argument("--output-interval", type=float, default=2.0e-4)
    parser.add_argument("--contact-stiffness", type=float, default=LINEAR_PENALTY_STIFFNESS)
    parser.add_argument("--max-contact-samples", type=int, default=1000)
    parser.add_argument("--max-abaqus-contact-faces", type=int, default=0)
    parser.add_argument("--skip-abaqus", action="store_true")
    parser.add_argument("--estimate-long-run", action="store_true")
    parser.add_argument("--baseline-duration", type=float, default=5.0e-5)
    parser.add_argument("--baseline-abaqus-wall", type=float, default=37.0)
    parser.add_argument("--baseline-sfc-wall", type=float, default=17.3868073000001)
    parser.add_argument("--baseline-frame-count", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.estimate_long_run):
        report = write_long_run_estimate(
            args.out_dir,
            duration=float(args.duration),
            fixed_dt=float(args.fixed_dt),
            baseline_duration=float(args.baseline_duration),
            baseline_abaqus_wall=float(args.baseline_abaqus_wall),
            baseline_sfc_wall=float(args.baseline_sfc_wall),
            baseline_frame_count=int(args.baseline_frame_count),
        )
        print(f"Long-run estimate: {report}")
        return
    outputs = run_workflow(
        args.out_dir,
        source=args.source,
        abaqus_command=args.abaqus_command,
        duration=float(args.duration),
        fixed_dt=float(args.fixed_dt),
        output_interval=float(args.output_interval),
        contact_stiffness=float(args.contact_stiffness),
        max_contact_samples=int(args.max_contact_samples),
        max_abaqus_contact_faces=int(args.max_abaqus_contact_faces),
        skip_abaqus=bool(args.skip_abaqus),
    )
    print("Flexible gear explicit SDF comparison complete.")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
