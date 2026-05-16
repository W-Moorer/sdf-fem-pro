"""Convert the RecurDyn gear RMD model into a CalculiX validation deck.

This script is intentionally kept in ``validation/``.  It is an external
reference runner for a user-provided RecurDyn model, not a dependency of the
core SFC solver.  The generated CalculiX deck maps the flexible RecurDyn
``FTETRA4`` body to C3D4 elements, maps the rigid gear triangle surface to an
S3 rigid body, drives the rigid gear by a rotational boundary condition, and
prints displacement, velocity, contact, stress, and strain data for post-run
VTK stress/strain clouds.

The conversion is mechanical and claim-gated: it records the mapping
assumptions in CSV/Markdown so the model can be reviewed before using the
results as paper evidence.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from validation.run_c3d8_contact_trajectory_validation import (  # noqa: E402
    _parse_contact_elements,
    _parse_nodal_vectors,
    _parse_totals,
)

Row = dict[str, Any]

NUMBER = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?"
FNODE_RE = re.compile(r"^FNODE/\s*(\d+).*?QG\s*=\s*(" + NUMBER + r")\s*,\s*(" + NUMBER + r")\s*,\s*(" + NUMBER + r")", re.I)
FTET_RE = re.compile(r"^FTETRA4/\s*(\d+).*?NODE\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", re.I)
MATERIAL_RE = re.compile(r"^MATERIAL\s*/\s*(\d+)", re.I)
MARKER_RE = re.compile(r"^MARKER\s*/\s*(\d+)", re.I)
GGEOM_RE = re.compile(r"^GGEOM\s*/\s*(\d+)", re.I)
FRBE_RE = re.compile(r"^FRBE/\s*(\d+)", re.I)
CONTACT_RE = re.compile(r"^GGEOMCONTACT\s*/\s*(\d+)", re.I)
C3D4_FACE_LABELS = {
    (0, 1, 2): "S1",
    (0, 1, 3): "S2",
    (1, 2, 3): "S3",
    (0, 2, 3): "S4",
}


@dataclass
class Material:
    E: float = 200000.0
    nu: float = 0.285
    rho_kg_per_mm3: float = 7.85e-6
    zeta: float = 1.0e-4

    @property
    def rho_tonne_per_mm3(self) -> float:
        """CalculiX N-mm-s density unit is tonne/mm^3."""

        return self.rho_kg_per_mm3 / 1000.0


@dataclass
class RigidSurface:
    marker_id: int = 0
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    patches: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))
    nodes_local: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))

    @property
    def nodes_global(self) -> np.ndarray:
        return self.nodes_local + self.origin[None, :]


@dataclass
class ContactParameters:
    stiffness: float = 100000.0
    damping: float = 10.0
    order: int = 2
    max_penetration: float = 0.06
    bpen: float = 1.0e-2


@dataclass
class RecurDynGearModel:
    nodes: dict[int, np.ndarray]
    elements: dict[int, tuple[int, int, int, int]]
    material: Material
    rigid_surface: RigidSurface
    flexible_surface_patches: np.ndarray
    hub_nodes: list[int]
    hub_reference_node: int | None
    contact: ContactParameters
    angular_velocity: float
    units: dict[str, str]

    @property
    def flexible_node_ids(self) -> np.ndarray:
        return np.asarray(sorted(self.nodes), dtype=np.int64)

    @property
    def element_ids(self) -> np.ndarray:
        return np.asarray(sorted(self.elements), dtype=np.int64)

    @property
    def X(self) -> np.ndarray:
        return np.vstack([self.nodes[int(node_id)] for node_id in self.flexible_node_ids])

    @property
    def connectivity_ids(self) -> np.ndarray:
        return np.asarray([self.elements[int(eid)] for eid in self.element_ids], dtype=np.int64)

    @property
    def id_to_index(self) -> dict[int, int]:
        return {int(node_id): idx for idx, node_id in enumerate(self.flexible_node_ids)}

    @property
    def connectivity_zero_based(self) -> np.ndarray:
        id_to_index = self.id_to_index
        return np.asarray([[id_to_index[int(node)] for node in element] for element in self.connectivity_ids], dtype=np.int64)


def _to_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "E"))


def _numbers(line: str) -> list[float]:
    return [_to_float(match.group(0)) for match in re.finditer(NUMBER, line)]


def _ints(line: str) -> list[int]:
    return [int(round(value)) for value in _numbers(line)]


def _parse_key_float(line: str, key: str) -> float | None:
    match = re.search(r"\b" + re.escape(key) + r"\s*=\s*(" + NUMBER + r")", line, re.I)
    return _to_float(match.group(1)) if match else None


def _parse_marker_origin(lines: list[str]) -> dict[int, np.ndarray]:
    markers: dict[int, np.ndarray] = {}
    current_marker: int | None = None
    for line in lines:
        match = MARKER_RE.match(line.strip())
        if match:
            current_marker = int(match.group(1))
            continue
        if current_marker is not None and "QP" in line:
            values = _numbers(line)
            if len(values) >= 3:
                markers[current_marker] = np.asarray(values[:3], dtype=float)
            current_marker = None
    return markers


def parse_recurdyn_rmd(path: Path) -> RecurDynGearModel:
    """Parse the RMD subset needed for the gear contact validation deck."""

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    markers = _parse_marker_origin(lines)

    nodes: dict[int, np.ndarray] = {}
    elements: dict[int, tuple[int, int, int, int]] = {}
    material = Material()
    materials: dict[int, Material] = {}
    rigid_surface = RigidSurface()
    flexible_patches: list[tuple[int, int, int]] = []
    hub_nodes: list[int] = []
    hub_reference_node: int | None = None
    contact = ContactParameters()
    units: dict[str, str] = {}
    angular_velocity = 1.0

    current_material: int | None = None
    current_ggeom: int | None = None
    ggeom_patch_mode = False
    ggeom_node_mode = False
    rigid_patches: list[tuple[int, int, int]] = []
    rigid_nodes: list[np.ndarray] = []
    in_frbe = False
    read_frbe_inodes = False
    read_contact = False
    read_units = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        marker = GGEOM_RE.match(line)
        if marker:
            current_ggeom = int(marker.group(1))
            current_material = None
            in_frbe = False
            ggeom_patch_mode = False
            ggeom_node_mode = False
            read_frbe_inodes = False
            read_contact = False
            read_units = False
            continue

        mat_match = MATERIAL_RE.match(line)
        if mat_match:
            current_material = int(mat_match.group(1))
            materials.setdefault(current_material, Material())
            current_ggeom = None
            ggeom_patch_mode = False
            ggeom_node_mode = False
            continue

        if FRBE_RE.match(line):
            current_material = None
            current_ggeom = None
            in_frbe = True
            ggeom_patch_mode = False
            ggeom_node_mode = False
            read_frbe_inodes = False
            continue

        if CONTACT_RE.match(line):
            read_contact = True
            in_frbe = False
            read_frbe_inodes = False
            current_material = None
            current_ggeom = None
            continue

        if line.startswith("UNITS"):
            read_units = True
            read_contact = False
            in_frbe = False
            read_frbe_inodes = False
            current_material = None
            current_ggeom = None
            continue

        fnode = FNODE_RE.match(line)
        if fnode:
            nodes[int(fnode.group(1))] = np.asarray([_to_float(fnode.group(i)) for i in range(2, 5)], dtype=float)
            continue

        ftet = FTET_RE.match(line)
        if ftet:
            elements[int(ftet.group(1))] = tuple(int(ftet.group(i)) for i in range(2, 6))
            continue

        if current_material is not None:
            E = _parse_key_float(line, "E")
            nu = _parse_key_float(line, "NU")
            rho = _parse_key_float(line, "RHO")
            zeta = _parse_key_float(line, "ZETA")
            mat = materials[current_material]
            if E is not None:
                mat.E = E
            if nu is not None:
                mat.nu = nu
            if rho is not None:
                mat.rho_kg_per_mm3 = rho
            if zeta is not None:
                mat.zeta = zeta
            continue

        if current_ggeom in {2, 3}:
            if "RM" in line and current_ggeom == 2:
                values = _ints(line)
                if values:
                    rigid_surface.marker_id = values[0]
                    rigid_surface.origin = markers.get(values[0], np.zeros(3))
            if "PATCHES" in line:
                ggeom_patch_mode = True
                ggeom_node_mode = False
                continue
            if "NODES" in line:
                ggeom_node_mode = True
                ggeom_patch_mode = False
                continue
            if line.startswith("!") or re.match(r"^[A-Z_]+\s*/", line, re.I):
                current_ggeom = None
                ggeom_patch_mode = False
                ggeom_node_mode = False
                continue
            if ggeom_patch_mode:
                values = _ints(line)
                if len(values) >= 4 and values[0] == 3:
                    patch = tuple(values[1:4])
                    if current_ggeom == 2:
                        rigid_patches.append(patch)
                    else:
                        flexible_patches.append(patch)
                continue
            if ggeom_node_mode and current_ggeom == 2:
                values = _numbers(line)
                if len(values) >= 3:
                    rigid_nodes.append(np.asarray(values[:3], dtype=float))
                continue

        if in_frbe and "JNODE" in line and hub_reference_node is None:
            value = _parse_key_float(line, "JNODE")
            if value is not None:
                hub_reference_node = int(round(value))

        if in_frbe and re.search(r"\bINODE\s*=", line):
            read_frbe_inodes = True
        if in_frbe and read_frbe_inodes:
            values = _ints(line)
            hub_nodes.extend(values)
            if re.search(r"\bTYPE\s*=", line) or re.search(r"^\s*,?\s*E\s*=", line):
                read_frbe_inodes = False
                in_frbe = False

        if read_contact:
            K = _parse_key_float(line, "K")
            C = _parse_key_float(line, "C")
            KORDER = _parse_key_float(line, "KORDER")
            MAXPEN = _parse_key_float(line, "MAXPEN")
            BPEN = _parse_key_float(line, "BPEN")
            if K is not None:
                contact.stiffness = K
            if C is not None:
                contact.damping = C
            if KORDER is not None:
                contact.order = int(round(KORDER))
            if MAXPEN is not None:
                contact.max_penetration = MAXPEN
            if BPEN is not None:
                contact.bpen = BPEN

        if read_units:
            for key in ("FORCE", "MASS", "LENGTH", "TIME"):
                match = re.search(r"\b" + key + r"\s*=\s*'?([^,']+)'?", line, re.I)
                if match:
                    units[key.lower()] = match.group(1).strip()

    if 3 in materials:
        material = materials[3]
    elif materials:
        material = next(iter(materials.values()))

    # RecurDyn expression Ex1 is FUNCTION=1\, a constant rotational velocity.
    angular_velocity = 1.0

    valid_hub_nodes = sorted({node for node in hub_nodes if node in nodes})
    rigid_surface.patches = np.asarray(rigid_patches, dtype=np.int64)
    rigid_surface.nodes_local = np.vstack(rigid_nodes) if rigid_nodes else np.empty((0, 3), dtype=float)

    return RecurDynGearModel(
        nodes=nodes,
        elements=elements,
        material=material,
        rigid_surface=rigid_surface,
        flexible_surface_patches=np.asarray(flexible_patches, dtype=np.int64),
        hub_nodes=valid_hub_nodes,
        hub_reference_node=hub_reference_node,
        contact=contact,
        angular_velocity=angular_velocity,
        units=units,
    )


def _append_id_list(lines: list[str], ids: list[int] | np.ndarray, *, per_line: int = 16) -> None:
    values = [int(value) for value in ids]
    for start in range(0, len(values), per_line):
        lines.append(", ".join(str(value) for value in values[start : start + per_line]))


def _axis_index(axis: str) -> int:
    return {"x": 1, "y": 2, "z": 3}[axis.lower()]


def _axis_vector(axis: str) -> np.ndarray:
    vectors = {
        "x": np.asarray([1.0, 0.0, 0.0]),
        "y": np.asarray([0.0, 1.0, 0.0]),
        "z": np.asarray([0.0, 0.0, 1.0]),
    }
    return vectors[axis.lower()]


def _rigid_rotation_displacement(points: np.ndarray, *, center: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Return final displacement for a rigid rotation about ``center``."""

    pts = np.asarray(points, dtype=float)
    c = np.asarray(center, dtype=float)
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    skew = np.asarray(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ],
        dtype=float,
    )
    R = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    return (pts - c) @ R.T + c - pts


def _c3d4_surface_face_refs(model: RecurDynGearModel) -> list[tuple[int, str]]:
    """Map RecurDyn surface triangles to CalculiX C3D4 element-face labels."""

    face_to_ref: dict[tuple[int, int, int], tuple[int, str]] = {}
    for element_id, element in model.elements.items():
        nodes = tuple(int(node) for node in element)
        for local_face, label in C3D4_FACE_LABELS.items():
            face = tuple(nodes[i] for i in local_face)
            key = tuple(sorted(face))
            face_to_ref.setdefault(key, (int(element_id), label))
    refs: list[tuple[int, str]] = []
    used: set[tuple[int, str]] = set()
    for patch in model.flexible_surface_patches:
        key = tuple(sorted(int(node) for node in patch))
        ref = face_to_ref.get(key)
        if ref is not None and ref not in used:
            refs.append(ref)
            used.add(ref)
    return refs


def _write_calculix_input(
    path: Path,
    model: RecurDynGearModel,
    *,
    duration: float,
    dt: float,
    output_every: int,
    rotation_axis: str,
    contact_law: str,
    direct_dynamic: bool,
    explicit_dynamic: bool,
    drive_mode: str,
    contact_stiffness_scale: float,
    slave_surface_mode: str,
) -> dict[str, Any]:
    """Write a CalculiX input deck and return generation metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rigid_node_offset = 200000
    rigid_element_offset = 300000
    ref_node = 900001
    rot_node = 900002
    axis_dof = _axis_index(rotation_axis)
    axis = _axis_vector(rotation_axis)
    ref_point = np.asarray([20.9136192988016, 0.503589451286037, 1.36499444260163], dtype=float)
    rot_point = ref_point + axis

    flexible_nodes = model.flexible_node_ids.tolist()
    contact_nodes = sorted({int(node) for patch in model.flexible_surface_patches for node in patch if int(node) in model.nodes})
    slave_face_refs = _c3d4_surface_face_refs(model) if slave_surface_mode == "element-face" else []
    hub_nodes = model.hub_nodes or [int(model.hub_reference_node)] if model.hub_reference_node in model.nodes else []
    rigid_node_ids = [rigid_node_offset + i + 1 for i in range(model.rigid_surface.nodes_global.shape[0])]
    rigid_element_ids = [rigid_element_offset + i + 1 for i in range(model.rigid_surface.patches.shape[0])]

    total_rotation = model.angular_velocity * duration
    rigid_final_displacement = _rigid_rotation_displacement(
        model.rigid_surface.nodes_global,
        center=ref_point,
        axis=axis,
        angle=total_rotation,
    )
    effective_contact_stiffness = model.contact.stiffness * contact_stiffness_scale
    pressure_line = f"{effective_contact_stiffness:.12e}"
    pressure_keyword = "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR"
    if contact_law == "exponential":
        # This is only an engineering approximation of RecurDyn's KORDER=2 law.
        pressure_keyword = "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=EXPONENTIAL"
        pressure_line = f"{max(model.contact.bpen, 1.0e-9):.12e}, {max(effective_contact_stiffness / 100000.0, 1.0e-9):.12e}"

    if explicit_dynamic:
        dynamic_keyword = "*DYNAMIC, DIRECT, EXPLICIT"
    else:
        dynamic_keyword = "*DYNAMIC, DIRECT" if direct_dynamic else "*DYNAMIC"
    dynamic_line = f"{dt:.12e}, {duration:.12e}" if direct_dynamic else f"{dt:.12e}, {duration:.12e}, {0.01 * dt:.12e}, {dt:.12e}"

    lines: list[str] = [
        "** Generated from assets/jiandanjiaolian.rmd by validation/run_recurdyn_gear_calculix.py",
        "** Mapping: RecurDyn FTETRA4 -> CalculiX C3D4; rigid GGEOM triangles -> S3 rigid body.",
        "** Units: N, mm, s; density converted from kg/mm^3 to tonne/mm^3.",
        "*NODE, NSET=GEAR22_NALL",
    ]
    for node_id in flexible_nodes:
        x, y, z = model.nodes[int(node_id)]
        lines.append(f"{int(node_id)}, {x:.12e}, {y:.12e}, {z:.12e}")
    lines.append("*NODE, NSET=GEAR21_RIGID_NODES")
    for node_id, xyz in zip(rigid_node_ids, model.rigid_surface.nodes_global):
        x, y, z = xyz
        lines.append(f"{int(node_id)}, {x:.12e}, {y:.12e}, {z:.12e}")
    lines.append("*NODE, NSET=GEAR21_CONTROL_NODES")
    lines.append(f"{ref_node}, {ref_point[0]:.12e}, {ref_point[1]:.12e}, {ref_point[2]:.12e}")
    lines.append(f"{rot_node}, {rot_point[0]:.12e}, {rot_point[1]:.12e}, {rot_point[2]:.12e}")

    lines.append("*ELEMENT, TYPE=C3D4, ELSET=GEAR22_SOLID")
    for element_id in model.element_ids:
        n1, n2, n3, n4 = model.elements[int(element_id)]
        lines.append(f"{int(element_id)}, {n1}, {n2}, {n3}, {n4}")
    lines.append("*ELEMENT, TYPE=S3, ELSET=GEAR21_SURF")
    for element_id, patch in zip(rigid_element_ids, model.rigid_surface.patches):
        n1, n2, n3 = (rigid_node_offset + int(local_id) for local_id in patch)
        lines.append(f"{int(element_id)}, {n1}, {n2}, {n3}")

    lines.append("*NSET, NSET=GEAR22_CONTACT_NODES")
    _append_id_list(lines, contact_nodes)
    lines.append("*NSET, NSET=GEAR22_HUB")
    _append_id_list(lines, hub_nodes)
    lines.extend(
        [
            "*MATERIAL, NAME=GEAR22_MAT",
            "*ELASTIC",
            f"{model.material.E:.12e}, {model.material.nu:.12e}",
            "*DENSITY",
            f"{model.material.rho_tonne_per_mm3:.12e}",
            "*MATERIAL, NAME=RIGID_GEAR_MAT",
            "*ELASTIC",
            f"{model.material.E:.12e}, {model.material.nu:.12e}",
            "*DENSITY",
            f"{model.material.rho_tonne_per_mm3:.12e}",
            "*SOLID SECTION, ELSET=GEAR22_SOLID, MATERIAL=GEAR22_MAT",
            "*SHELL SECTION, ELSET=GEAR21_SURF, MATERIAL=RIGID_GEAR_MAT",
            "1.0e-3",
        ]
    )
    if drive_mode == "rigid-body":
        lines.append(f"*RIGID BODY, NSET=GEAR21_RIGID_NODES, REF NODE={ref_node}, ROT NODE={rot_node}")
    lines.extend(
        [
        ]
    )
    if slave_surface_mode == "element-face":
        lines.append("*SURFACE, NAME=GEAR22_SLAVE, TYPE=ELEMENT")
        for element_id, label in slave_face_refs:
            lines.append(f"{element_id}, {label}")
    else:
        lines.extend(["*SURFACE, NAME=GEAR22_SLAVE, TYPE=NODE", "GEAR22_CONTACT_NODES"])
    lines.extend(
        [
            "*SURFACE, NAME=GEAR21_MASTER, TYPE=ELEMENT",
            "GEAR21_SURF, SPOS",
            "*SURFACE INTERACTION, NAME=GEAR_CONTACT",
            pressure_keyword,
            pressure_line,
            "*CONTACT PAIR, INTERACTION=GEAR_CONTACT, TYPE=SURFACE TO SURFACE"
            if slave_surface_mode == "element-face"
            else "*CONTACT PAIR, INTERACTION=GEAR_CONTACT, TYPE=NODE TO SURFACE",
            "GEAR22_SLAVE, GEAR21_MASTER",
            "*AMPLITUDE, NAME=ROTAMP",
            f"0.0, 0.0, {duration:.12e}, 1.0",
            "*BOUNDARY",
            "GEAR22_HUB, 1, 3, 0.0",
        ]
    )
    if drive_mode == "rigid-body":
        lines.extend(
            [
                f"{ref_node}, 1, 3, 0.0",
                f"{rot_node}, {2 if axis_dof == 1 else 1}, {2 if axis_dof == 1 else 1}, 0.0",
                f"{rot_node}, {3 if axis_dof in {1, 2} else 2}, {3 if axis_dof in {1, 2} else 2}, 0.0",
            ]
        )
    lines.extend(
        [
            "*STEP, NLGEOM, INC=1000000",
            dynamic_keyword,
            dynamic_line,
            "*BOUNDARY, AMPLITUDE=ROTAMP",
        ]
    )
    if drive_mode == "rigid-body":
        lines.append(f"{rot_node}, {axis_dof}, {axis_dof}, {total_rotation:.12e}")
    elif drive_mode == "prescribed-surface":
        for node_id, displacement in zip(rigid_node_ids, rigid_final_displacement):
            ux, uy, uz = displacement
            lines.append(f"{int(node_id)}, 1, 1, {ux:.12e}")
            lines.append(f"{int(node_id)}, 2, 2, {uy:.12e}")
            lines.append(f"{int(node_id)}, 3, 3, {uz:.12e}")
    else:
        lines.append("GEAR21_RIGID_NODES, 1, 3, 0.0")
    lines.extend(
        [
            f"*NODE PRINT, NSET=GEAR22_NALL, FREQUENCY={output_every}",
            "U",
            f"*NODE PRINT, NSET=GEAR22_NALL, FREQUENCY={output_every}",
            "V",
            f"*NODE PRINT, NSET=GEAR22_HUB, TOTALS=ONLY, GLOBAL=YES, FREQUENCY={output_every}",
            "RF",
            f"*CONTACT PRINT, FREQUENCY={output_every}",
            "CDIS,CSTR,CELS",
            f"*CONTACT PRINT, TOTALS=ONLY, FREQUENCY={output_every}",
            "CELS,CNUM",
            f"*EL PRINT, ELSET=GEAR22_SOLID, FREQUENCY={output_every}",
            "S,E",
            "*END STEP",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "inp_path": str(path),
        "flexible_nodes": len(flexible_nodes),
        "tet4_elements": len(model.elements),
        "flexible_contact_nodes": len(contact_nodes),
        "flexible_contact_element_faces": len(slave_face_refs),
        "rigid_surface_nodes": len(rigid_node_ids),
        "rigid_surface_triangles": len(rigid_element_ids),
        "hub_fixed_nodes": len(hub_nodes),
        "duration": duration,
        "dt": dt,
        "output_every": output_every,
        "rotation_axis": rotation_axis,
        "total_rotation": total_rotation,
        "contact_law": contact_law,
        "effective_contact_stiffness": effective_contact_stiffness,
        "contact_stiffness_scale": contact_stiffness_scale,
        "direct_dynamic": direct_dynamic,
        "explicit_dynamic": explicit_dynamic,
        "drive_mode": drive_mode,
        "slave_surface_mode": slave_surface_mode,
    }


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def _run_calculix(inp_path: Path, log_path: Path, *, timeout: int) -> Row:
    job_name = inp_path.stem
    cwd = inp_path.parent
    cmd = f"cd {_shell_quote(_wsl_path(cwd))} && ccx -i {_shell_quote(job_name)}"
    command = ["wsl", "--exec", "bash", "-lc", cmd]
    start = time.perf_counter()
    try:
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        elapsed = time.perf_counter() - start
        return_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return_code = 124
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        timed_out = True
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                "$ " + " ".join(command),
                "",
                "STDOUT:",
                stdout,
                "",
                "STDERR:",
                stderr,
            ]
        ),
        encoding="utf-8",
    )
    return {
        "command": " ".join(command),
        "return_code": return_code,
        "timed_out": str(timed_out).lower(),
        "wall_time_seconds": elapsed,
        "stdout_tail": "\n".join(stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-20:]),
    }


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _parse_element_tensor(path: Path, *, quantity: str) -> dict[float, dict[int, np.ndarray]]:
    header_words = {"stress": "stresses", "strain": "strains"}
    if quantity not in header_words:
        raise ValueError("quantity must be 'stress' or 'strain'")
    number = r"([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    header = re.compile(header_words[quantity] + r".*time\s+" + number, re.IGNORECASE)
    row = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+"
        + number
        + r"\s+"
        + number
        + r"\s+"
        + number
        + r"\s+"
        + number
        + r"\s+"
        + number
        + r"\s+"
        + number
    )
    values: dict[float, dict[int, list[np.ndarray]]] = {}
    current_time: float | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = header.search(line)
        if match:
            current_time = float(match.group(1))
            values.setdefault(current_time, {})
            continue
        if current_time is None:
            continue
        match = row.match(line)
        if match is None:
            if re.search(r"[A-Za-z]", line):
                current_time = None
            continue
        element_id = int(match.group(1))
        xx, yy, zz, xy, xz, yz = [float(match.group(i)) for i in range(3, 9)]
        values[current_time].setdefault(element_id, []).append(np.asarray([xx, yy, zz, xy, yz, xz], dtype=float))
    return {time: {eid: np.mean(tensors, axis=0) for eid, tensors in element_map.items()} for time, element_map in values.items()}


def _von_mises(stress: np.ndarray) -> float:
    sxx, syy, szz, sxy, syz, sxz = np.asarray(stress, dtype=float)
    return float(math.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + syz**2 + sxz**2)))


def _write_tet_vtk(
    path: Path,
    points: np.ndarray,
    elements: np.ndarray,
    *,
    displacement: np.ndarray,
    stress: np.ndarray,
    strain: np.ndarray,
    von_mises: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# vtk DataFile Version 3.0",
        "RecurDyn gear converted CalculiX stress strain cloud",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {points.shape[0]} float",
    ]
    lines.extend(f"{x:.12e} {y:.12e} {z:.12e}" for x, y, z in points)
    lines.append(f"CELLS {elements.shape[0]} {elements.shape[0] * 5}")
    lines.extend("4 " + " ".join(str(int(node)) for node in element) for element in elements)
    lines.append(f"CELL_TYPES {elements.shape[0]}")
    lines.extend("10" for _ in range(elements.shape[0]))
    lines.append(f"POINT_DATA {points.shape[0]}")
    lines.append("VECTORS displacement float")
    lines.extend(f"{u:.12e} {v:.12e} {w:.12e}" for u, v, w in displacement)
    lines.append(f"CELL_DATA {elements.shape[0]}")
    lines.append("SCALARS von_mises float 1")
    lines.append("LOOKUP_TABLE default")
    lines.extend(f"{float(value):.12e}" for value in von_mises)
    lines.append("SCALARS strain_norm float 1")
    lines.append("LOOKUP_TABLE default")
    lines.extend(f"{float(np.linalg.norm(value)):.12e}" for value in strain)
    lines.append("TENSORS stress_voigt_symmetric float")
    for sxx, syy, szz, sxy, syz, sxz in stress:
        lines.append(f"{sxx:.12e} {sxy:.12e} {sxz:.12e}")
        lines.append(f"{sxy:.12e} {syy:.12e} {syz:.12e}")
        lines.append(f"{sxz:.12e} {syz:.12e} {szz:.12e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_placeholder_vtk(path: Path, model: RecurDynGearModel) -> None:
    zeros_nodes = np.zeros_like(model.X)
    zeros_cells = np.zeros((len(model.element_ids), 6), dtype=float)
    _write_tet_vtk(
        path,
        model.X,
        model.connectivity_zero_based,
        displacement=zeros_nodes,
        stress=zeros_cells,
        strain=zeros_cells,
        von_mises=np.zeros(len(model.element_ids), dtype=float),
    )


def _export_vtk_frames(out_dir: Path, model: RecurDynGearModel, dat_path: Path, *, max_frames: int | None) -> list[Row]:
    frames_dir = out_dir / "vtk"
    node_ids = model.flexible_node_ids
    element_ids = model.element_ids
    id_to_element_row = {int(eid): idx for idx, eid in enumerate(element_ids)}

    if not dat_path.exists():
        placeholder = frames_dir / "recurdyn_gear_calculix_frame_0000.vtk"
        _write_placeholder_vtk(placeholder, model)
        return [
            {
                "frame_index": 0,
                "time": 0.0,
                "vtk_path": str(placeholder),
                "source": "placeholder_initial_geometry_no_calculix_dat",
                "max_von_mises": 0.0,
                "max_strain_norm": 0.0,
            }
        ]

    u_blocks = _parse_nodal_vectors(dat_path, node_ids, quantity="u")
    stress_blocks = _parse_element_tensor(dat_path, quantity="stress")
    strain_blocks = _parse_element_tensor(dat_path, quantity="strain")
    times = sorted(set(u_blocks) | set(stress_blocks) | set(strain_blocks))
    if max_frames is not None and len(times) > max_frames:
        indices = np.linspace(0, len(times) - 1, max_frames, dtype=int)
        times = [times[int(i)] for i in indices]
    if not times:
        placeholder = frames_dir / "recurdyn_gear_calculix_frame_0000.vtk"
        _write_placeholder_vtk(placeholder, model)
        return [
            {
                "frame_index": 0,
                "time": 0.0,
                "vtk_path": str(placeholder),
                "source": "placeholder_initial_geometry_empty_dat_parse",
                "max_von_mises": 0.0,
                "max_strain_norm": 0.0,
            }
        ]

    rows: list[Row] = []
    X = model.X
    elements = model.connectivity_zero_based
    for frame_index, time_value in enumerate(times):
        displacement = u_blocks.get(time_value, np.zeros_like(X))
        stress = np.zeros((len(element_ids), 6), dtype=float)
        strain = np.zeros((len(element_ids), 6), dtype=float)
        for element_id, tensor in stress_blocks.get(time_value, {}).items():
            row = id_to_element_row.get(int(element_id))
            if row is not None:
                stress[row] = tensor
        for element_id, tensor in strain_blocks.get(time_value, {}).items():
            row = id_to_element_row.get(int(element_id))
            if row is not None:
                strain[row] = tensor
        vm = np.asarray([_von_mises(value) for value in stress], dtype=float)
        path = frames_dir / f"recurdyn_gear_calculix_frame_{frame_index:04d}.vtk"
        _write_tet_vtk(path, X + displacement, elements, displacement=displacement, stress=stress, strain=strain, von_mises=vm)
        rows.append(
            {
                "frame_index": frame_index,
                "time": time_value,
                "vtk_path": str(path),
                "source": "calculix_dat",
                "max_von_mises": float(np.max(vm)) if vm.size else 0.0,
                "max_strain_norm": float(np.max(np.linalg.norm(strain, axis=1))) if strain.size else 0.0,
            }
        )
    return rows


def _boundary_faces(elements: np.ndarray) -> np.ndarray:
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    seen: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for element in elements:
        for face in faces:
            f = tuple(int(element[i]) for i in face)
            seen.setdefault(tuple(sorted(f)), []).append(f)
    return np.asarray([records[0] for records in seen.values() if len(records) == 1], dtype=np.int64)


def _write_preview_png(path: Path, model: RecurDynGearModel, frame_rows: list[Row]) -> Row:
    """Write a light 3D preview of the final VTK frame geometry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    X = model.X
    faces = _boundary_faces(model.connectivity_zero_based)
    # Subsample for a fast review image; VTK remains the authoritative cloud.
    if faces.shape[0] > 5000:
        step = max(1, faces.shape[0] // 5000)
        faces = faces[::step]
    polys = [X[face] for face in faces]
    colors = np.linalg.norm(X[faces].mean(axis=1) - X.mean(axis=0), axis=1)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(polys, linewidths=0.02, alpha=0.95)
    collection.set_array(colors)
    collection.set_cmap("viridis")
    ax.add_collection3d(collection)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("Converted RecurDyn flexible gear surface preview")
    fig.colorbar(collection, ax=ax, shrink=0.7, label="radial coordinate preview")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"figure_path": str(path), "source": "geometry_preview", "boundary_faces_plotted": int(len(polys)), "vtk_frames": len(frame_rows)}


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _write_markdown(
    path: Path,
    *,
    model: RecurDynGearModel,
    generation: Row,
    run_row: Row | None,
    frame_rows: list[Row],
    totals: dict[float, Row],
    contact_rows: dict[float, list[Row]],
    command_line: str,
) -> None:
    run_completed = run_row is not None and int(run_row.get("return_code", -1)) == 0
    vtk_status = "generated_from_calculix_dat" if frame_rows and frame_rows[0]["source"] == "calculix_dat" else "placeholder_or_no_dat"
    text = [
        "# RecurDyn Gear to CalculiX 1s Validation Handoff",
        "",
        "## Command",
        "",
        f"`{command_line}`",
        "",
        "## Parsed RMD model",
        "",
        f"- Flexible nodes: {generation['flexible_nodes']}",
        f"- C3D4 elements: {generation['tet4_elements']}",
        f"- Flexible slave contact nodes: {generation['flexible_contact_nodes']}",
        f"- Rigid gear surface triangles: {generation['rigid_surface_triangles']}",
        f"- Fixed hub nodes from FRBE: {generation['hub_fixed_nodes']}",
        f"- Material: E={model.material.E:g} N/mm^2, nu={model.material.nu:g}, density={model.material.rho_tonne_per_mm3:.6e} tonne/mm^3",
        f"- RecurDyn contact law: K={model.contact.stiffness:g}, order={model.contact.order}, damping={model.contact.damping:g}",
        "",
        "## CalculiX run",
        "",
        f"- Input deck: `{_display(Path(generation['inp_path']))}`",
        f"- Duration: {generation['duration']} s",
        f"- Time step request: {generation['dt']} s",
        f"- Output frequency: every {generation['output_every']} increments",
        f"- Rotation axis: {generation['rotation_axis']}",
        f"- Rigid surface drive mode: {generation['drive_mode']}",
        f"- Slave surface mode: {generation['slave_surface_mode']}",
        f"- Effective CalculiX contact stiffness: {float(generation['effective_contact_stiffness']):.6e}",
        f"- Total prescribed rigid gear rotation: {generation['total_rotation']} rad",
        f"- CalculiX completed: {str(run_completed).lower()}",
    ]
    if run_row is not None:
        text.extend(
            [
                f"- Return code: {run_row['return_code']}",
                f"- Wall time: {float(run_row['wall_time_seconds']):.3f} s",
            ]
        )
    text.extend(
        [
            "",
            "## Outputs",
            "",
            f"- VTK status: {vtk_status}",
            f"- VTK frames: {len(frame_rows)}",
            f"- Contact totals time rows: {len(totals)}",
            f"- Contact element time rows: {len(contact_rows)}",
            "",
            "## Acceptance status",
            "",
        ]
    )
    if run_completed and frame_rows and frame_rows[0]["source"] == "calculix_dat":
        text.append("PASS: the RMD model was converted, CalculiX completed, and stress/strain VTK frames were generated.")
    elif run_completed:
        text.append("PARTIAL: CalculiX completed, but the `.dat` stress/strain parse did not produce authoritative VTK field frames.")
    else:
        text.append("BLOCKED: the CalculiX run did not complete. Generated input and any partial VTK frames are available for debugging.")
    text.extend(
        [
            "",
            "## Important limitations",
            "",
            "- RecurDyn's quadratic contact parameter `KORDER=2` is approximated by a CalculiX pressure-overclosure law.",
            "- The rigid gear drive can be mapped either as a CalculiX rigid body or as prescribed rigid-surface nodal motion; prescribed-surface mode is an engineering approximation.",
            "- The flexible gear hub is fixed using the FRBE `INODE` list rather than translating every RecurDyn joint/bushing equation.",
            "- This validates a CalculiX reproduction path for the RMD model; it is not yet an SFC-vs-RecurDyn trajectory equivalence result.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmd", type=Path, default=ROOT / "assets" / "jiandanjiaolian.rmd")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "recurdyn_gear_calculix")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--output-every", type=int, default=10)
    parser.add_argument("--rotation-axis", choices=["x", "y", "z"], default="x")
    parser.add_argument("--contact-law", choices=["linear", "exponential"], default="linear")
    parser.add_argument(
        "--contact-stiffness-scale",
        type=float,
        default=1.0,
        help="Scale RecurDyn K for CalculiX engineering smoke runs; 1.0 preserves the parsed value.",
    )
    parser.add_argument(
        "--drive-mode",
        choices=["rigid-body", "prescribed-surface", "fixed"],
        default="rigid-body",
        help="CalculiX mapping for the rigid RecurDyn gear surface.",
    )
    parser.add_argument(
        "--slave-surface-mode",
        choices=["node", "element-face"],
        default="node",
        help="Use RecurDyn node contact or map the surface triangles back to C3D4 element faces.",
    )
    parser.add_argument(
        "--automatic-increment",
        action="store_true",
        help="Use CalculiX automatic dynamic incrementation instead of fixed *DYNAMIC,DIRECT.",
    )
    parser.add_argument(
        "--explicit",
        action="store_true",
        help="Use CalculiX *DYNAMIC,DIRECT,EXPLICIT for a fixed-step engineering stress-cloud run.",
    )
    parser.add_argument("--skip-calculix", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Generate a short smoke deck unless duration/dt are explicitly overridden.")
    parser.add_argument("--max-vtk-frames", type=int, default=101)
    parser.add_argument("--timeout", type=int, default=3600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    duration = 0.002 if args.quick and args.duration == 1.0 else args.duration
    output_every = max(1, int(args.output_every))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = parse_recurdyn_rmd(args.rmd)
    inp_path = out_dir / "jiandanjiaolian_calculix_1s.inp"
    generation = _write_calculix_input(
        inp_path,
        model,
        duration=duration,
        dt=args.dt,
        output_every=output_every,
        rotation_axis=args.rotation_axis,
        contact_law=args.contact_law,
        direct_dynamic=not args.automatic_increment,
        explicit_dynamic=args.explicit,
        drive_mode=args.drive_mode,
        contact_stiffness_scale=args.contact_stiffness_scale,
        slave_surface_mode=args.slave_surface_mode,
    )

    run_row: Row | None = None
    if not args.skip_calculix:
        run_row = _run_calculix(inp_path, out_dir / "logs" / "calculix_run.log", timeout=args.timeout)

    dat_path = inp_path.with_suffix(".dat")
    frame_rows = _export_vtk_frames(out_dir, model, dat_path, max_frames=args.max_vtk_frames)
    preview = _write_preview_png(out_dir / "figures" / "recurdyn_gear_surface_preview.png", model, frame_rows)
    totals = _parse_totals(dat_path) if dat_path.exists() else {}
    contact_rows = _parse_contact_elements(dat_path) if dat_path.exists() else {}

    metadata_rows = [
        {"key": key, "value": value}
        for key, value in {
            **generation,
            "rmd_path": str(args.rmd),
            "run_completed": run_row is not None and int(run_row.get("return_code", -1)) == 0,
            "dat_path": str(dat_path),
            "vtk_frame_count": len(frame_rows),
        }.items()
    ]
    _write_csv(out_dir / "recurdyn_gear_calculix_metadata.csv", ["key", "value"], metadata_rows)
    _write_csv(out_dir / "recurdyn_gear_calculix_vtk_frames.csv", list(frame_rows[0].keys()), frame_rows)
    _write_csv(out_dir / "recurdyn_gear_calculix_figures.csv", list(preview.keys()), [preview])

    summary_rows = [
        {
            "case_id": "recurdyn_jiandanjiaolian_calculix_1s",
            "rmd_parsed": "true",
            "calculix_completed": str(run_row is not None and int(run_row.get("return_code", -1)) == 0).lower(),
            "duration_s": duration,
            "dt_s": args.dt,
            "vtk_frames": len(frame_rows),
            "stress_strain_vtk_generated": str(frame_rows and frame_rows[0]["source"] == "calculix_dat").lower(),
            "contact_totals_rows": len(totals),
            "acceptance": "pass"
            if run_row is not None and int(run_row.get("return_code", -1)) == 0 and frame_rows and frame_rows[0]["source"] == "calculix_dat"
            else "blocked_or_partial",
        }
    ]
    _write_csv(out_dir / "recurdyn_gear_calculix_summary.csv", list(summary_rows[0].keys()), summary_rows)
    if run_row is not None:
        _write_csv(out_dir / "recurdyn_gear_calculix_run.csv", list(run_row.keys()), [run_row])

    command_line = "python " + " ".join([str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]])
    _write_markdown(
        out_dir / "recurdyn_gear_calculix_summary.md",
        model=model,
        generation=generation,
        run_row=run_row,
        frame_rows=frame_rows,
        totals=totals,
        contact_rows=contact_rows,
        command_line=command_line,
    )
    print(f"Wrote {out_dir / 'recurdyn_gear_calculix_summary.md'}")
    return 0 if summary_rows[0]["acceptance"] == "pass" or args.skip_calculix else 1


if __name__ == "__main__":
    raise SystemExit(main())
