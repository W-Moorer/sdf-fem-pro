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
import shutil
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
from scipy.spatial import cKDTree  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from sfc.sdf.local_projection import closest_point_on_triangle  # noqa: E402
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
PART_RE = re.compile(r"^PART\s*/\s*(\d+)", re.I)
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
class PartPose:
    qg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reuler: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class MarkerPose:
    part_id: int | None = None
    qp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reuler: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class RigidSurface:
    marker_id: int = 0
    marker_pose: MarkerPose = field(default_factory=MarkerPose)
    part_pose: PartPose = field(default_factory=PartPose)
    patches: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))
    nodes_local: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))

    @property
    def nodes_global(self) -> np.ndarray:
        marker_points = _transform_points(self.nodes_local, self.marker_pose.qp, self.marker_pose.reuler)
        return _transform_points(marker_points, self.part_pose.qg, self.part_pose.reuler)


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


def _parse_key_vector(line: str, key: str) -> np.ndarray | None:
    match = re.search(
        r"\b" + re.escape(key) + r"\s*=\s*("
        + NUMBER
        + r")\s*,\s*("
        + NUMBER
        + r")\s*,\s*("
        + NUMBER
        + r")",
        line,
        re.I,
    )
    if not match:
        return None
    return np.asarray([_to_float(match.group(i)) for i in range(1, 4)], dtype=float)


def _rotation_matrix_from_reuler(reuler: np.ndarray) -> np.ndarray:
    """Return a rotation matrix for RecurDyn-style XYZ Euler angles.

    The RMD gear surface exercised by this runner has zero marker rotation and
    a small body pitch; for that case the convention-dependent ambiguity is not
    material.  The implementation keeps a standard intrinsic XYZ composition so
    all parsed rotations are applied consistently rather than ignored.
    """

    rx, ry, rz = (float(value) for value in reuler)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    Ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    Rz = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return Rz @ Ry @ Rx


def _transform_points(points: np.ndarray, translation: np.ndarray, reuler: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape((-1, 3))
    R = _rotation_matrix_from_reuler(np.asarray(reuler, dtype=float))
    return pts @ R.T + np.asarray(translation, dtype=float)[None, :]


def _parse_part_marker_poses(lines: list[str]) -> tuple[dict[int, PartPose], dict[int, MarkerPose]]:
    parts: dict[int, PartPose] = {}
    markers: dict[int, MarkerPose] = {}
    current_kind: str | None = None
    current_id: int | None = None
    for line in lines:
        stripped = line.strip()
        part_match = PART_RE.match(stripped)
        marker_match = MARKER_RE.match(stripped)
        if part_match:
            current_kind = "part"
            current_id = int(part_match.group(1))
            parts.setdefault(current_id, PartPose())
            continue
        if marker_match:
            current_kind = "marker"
            current_id = int(marker_match.group(1))
            markers.setdefault(current_id, MarkerPose())
            continue
        if re.match(r"^[A-Z_]+\s*/", stripped, re.I):
            current_kind = None
            current_id = None
            continue
        if current_kind == "part" and current_id is not None:
            pose = parts.setdefault(current_id, PartPose())
            qg = _parse_key_vector(line, "QG")
            reuler = _parse_key_vector(line, "REULER")
            if qg is not None:
                pose.qg = qg
            if reuler is not None:
                pose.reuler = reuler
        elif current_kind == "marker" and current_id is not None:
            pose = markers.setdefault(current_id, MarkerPose())
            part_id = _parse_key_float(line, "PART")
            qp = _parse_key_vector(line, "QP")
            reuler = _parse_key_vector(line, "REULER")
            if part_id is not None:
                pose.part_id = int(round(part_id))
            if qp is not None:
                pose.qp = qp
            if reuler is not None:
                pose.reuler = reuler
    return parts, markers


def parse_recurdyn_rmd(path: Path) -> RecurDynGearModel:
    """Parse the RMD subset needed for the gear contact validation deck."""

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    parts, markers = _parse_part_marker_poses(lines)

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
                    rigid_surface.marker_pose = markers.get(values[0], MarkerPose())
                    rigid_surface.part_pose = parts.get(rigid_surface.marker_pose.part_id or 0, PartPose())
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


def _oriented_rigid_surface(model: RecurDynGearModel) -> tuple[np.ndarray, np.ndarray]:
    """Return rigid nodes and outward-ish oriented triangle connectivity."""

    points = model.rigid_surface.nodes_global
    faces = np.asarray(model.rigid_surface.patches, dtype=np.int64) - 1
    center = np.mean(points, axis=0)
    oriented = faces.copy()
    for i, face in enumerate(oriented):
        tri = points[face]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.linalg.norm(normal) <= 0.0:
            continue
        if float(np.dot(normal, np.mean(tri, axis=0) - center)) < 0.0:
            oriented[i, [1, 2]] = oriented[i, [2, 1]]
    return points, oriented


def _initial_contact_query_points(model: RecurDynGearModel) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Return flexible contact nodes and face centroids for initial gap checks."""

    point_rows: list[np.ndarray] = []
    labels: list[tuple[str, int]] = []
    contact_nodes = sorted({int(node) for patch in model.flexible_surface_patches for node in patch if int(node) in model.nodes})
    for node_id in contact_nodes:
        point_rows.append(model.nodes[node_id])
        labels.append(("node", node_id))
    for i, patch in enumerate(model.flexible_surface_patches):
        coords = [model.nodes[int(node)] for node in patch if int(node) in model.nodes]
        if len(coords) == 3:
            point_rows.append(np.mean(np.vstack(coords), axis=0))
            labels.append(("face_centroid", i))
    return np.vstack(point_rows), labels


def evaluate_initial_gap_diagnostics(
    model: RecurDynGearModel,
    *,
    candidate_count: int = 96,
) -> tuple[list[Row], list[Row]]:
    """Evaluate initial flexible contact-surface gaps to the rigid gear surface."""

    rigid_points, rigid_faces = _oriented_rigid_surface(model)
    query_points, labels = _initial_contact_query_points(model)
    triangles = rigid_points[rigid_faces]
    centroids = np.mean(triangles, axis=1)
    tree = cKDTree(centroids)
    k = min(max(1, int(candidate_count)), len(triangles))

    sample_rows: list[Row] = []
    for point_index, point in enumerate(query_points):
        _, candidate_ids = tree.query(point, k=k)
        candidate_array = np.atleast_1d(candidate_ids).astype(np.int64)
        best: tuple[float, float, int, np.ndarray, np.ndarray] | None = None
        for face_id in candidate_array:
            tri = triangles[int(face_id)]
            p, _, dist2, _ = closest_point_on_triangle(point, tri[0], tri[1], tri[2])
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = np.linalg.norm(normal)
            if norm <= 0.0:
                continue
            normal = normal / norm
            distance = float(np.sqrt(dist2))
            signed_gap = float(np.dot(point - p, normal))
            # If closest feature is an edge/vertex, the oriented local surface
            # sign remains only diagnostic; unsigned distance is authoritative.
            if best is None or dist2 < best[0]:
                best = (float(dist2), signed_gap, int(face_id), p, normal)
        if best is None:
            continue
        dist2, signed_gap, face_id, p, normal = best
        sample_type, sample_id = labels[point_index]
        sample_rows.append(
            {
                "sample_index": point_index,
                "sample_type": sample_type,
                "sample_id": sample_id,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "nearest_face": face_id,
                "signed_gap": signed_gap,
                "unsigned_distance": float(np.sqrt(dist2)),
                "projection_x": float(p[0]),
                "projection_y": float(p[1]),
                "projection_z": float(p[2]),
                "normal_x": float(normal[0]),
                "normal_y": float(normal[1]),
                "normal_z": float(normal[2]),
            }
        )

    summary_rows: list[Row] = []
    for sample_type in ("node", "face_centroid", "all"):
        rows = sample_rows if sample_type == "all" else [row for row in sample_rows if row["sample_type"] == sample_type]
        if not rows:
            continue
        gaps = np.asarray([float(row["signed_gap"]) for row in rows], dtype=float)
        distances = np.asarray([float(row["unsigned_distance"]) for row in rows], dtype=float)
        summary_rows.append(
            {
                "sample_type": sample_type,
                "count": len(rows),
                "candidate_count": k,
                "min_signed_gap": float(np.min(gaps)),
                "max_signed_gap": float(np.max(gaps)),
                "mean_signed_gap": float(np.mean(gaps)),
                "negative_signed_gap_count": int(np.sum(gaps < 0.0)),
                "min_unsigned_distance": float(np.min(distances)),
                "max_unsigned_distance": float(np.max(distances)),
                "mean_unsigned_distance": float(np.mean(distances)),
                "near_zero_distance_count_1e_6": int(np.sum(distances < 1.0e-6)),
                "near_zero_distance_count_1e_3": int(np.sum(distances < 1.0e-3)),
            }
        )
    return sample_rows, summary_rows


def _write_gap_histogram(path: Path, sample_rows: list[Row]) -> Row:
    path.parent.mkdir(parents=True, exist_ok=True)
    gaps = np.asarray([float(row["signed_gap"]) for row in sample_rows], dtype=float)
    distances = np.asarray([float(row["unsigned_distance"]) for row in sample_rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(gaps, bins=80, color="#3b82f6", alpha=0.85)
    axes[0].axvline(0.0, color="black", linewidth=1.0)
    axes[0].set_title("Local signed side diagnostic")
    axes[0].set_xlabel("local signed gap [mm]")
    axes[0].set_ylabel("count")
    axes[1].hist(distances, bins=80, color="#10b981", alpha=0.85)
    axes[1].set_title("Unsigned closest distance")
    axes[1].set_xlabel("distance [mm]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"figure_path": str(path), "source": "initial_gap_diagnostic", "sample_count": len(sample_rows)}


def evaluate_contact_law_alignment(model: RecurDynGearModel, *, samples: int = 101) -> tuple[list[Row], list[Row]]:
    """Generate a reviewable RecurDyn-vs-CalculiX contact-law proxy table."""

    max_pen = max(float(model.contact.max_penetration), 1.0e-9)
    overclosure = np.linspace(0.0, max_pen, max(3, int(samples)))
    K = float(model.contact.stiffness)
    order = max(1, int(model.contact.order))

    # RecurDyn's exact internal law is proprietary to RecurDyn.  This proxy
    # records the common polynomial interpretation implied by KORDER and makes
    # the approximation explicit in CSV/Markdown instead of hiding it in the
    # CalculiX deck.
    recur_proxy = K * overclosure**order
    endpoint_slope = float(recur_proxy[-1] / max_pen)
    lsq_slope = float(np.dot(overclosure, recur_proxy) / max(np.dot(overclosure, overclosure), 1.0e-30))
    parsed_slope = K
    rows: list[Row] = []
    for d, reference in zip(overclosure, recur_proxy):
        rows.append(
            {
                "overclosure": float(d),
                "recurdyn_polynomial_proxy": float(reference),
                "calculix_linear_parsed_K": float(parsed_slope * d),
                "calculix_linear_endpoint_fit": float(endpoint_slope * d),
                "calculix_linear_lsq_fit": float(lsq_slope * d),
            }
        )
    summary: list[Row] = []
    for name, slope in (
        ("parsed_K", parsed_slope),
        ("endpoint_fit", endpoint_slope),
        ("least_squares_fit", lsq_slope),
    ):
        pred = slope * overclosure
        err = pred - recur_proxy
        summary.append(
            {
                "fit": name,
                "linear_slope": float(slope),
                "max_abs_error": float(np.max(np.abs(err))),
                "rms_error": float(np.sqrt(np.mean(err * err))),
                "relative_rms_error": float(np.sqrt(np.mean(err * err)) / max(np.sqrt(np.mean(recur_proxy * recur_proxy)), 1.0e-30)),
                "law_scope": "proxy_only_recurdyn_KORDER_not_exact",
            }
        )
    return rows, summary


def _write_contact_law_plot(path: Path, rows: list[Row]) -> Row:
    path.parent.mkdir(parents=True, exist_ok=True)
    overclosure = np.asarray([float(row["overclosure"]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(overclosure, [float(row["recurdyn_polynomial_proxy"]) for row in rows], label="RecurDyn KORDER proxy", linewidth=2.0)
    ax.plot(overclosure, [float(row["calculix_linear_parsed_K"]) for row in rows], label="CalculiX linear parsed K", linestyle="--")
    ax.plot(overclosure, [float(row["calculix_linear_endpoint_fit"]) for row in rows], label="CalculiX endpoint fit", linestyle=":")
    ax.plot(overclosure, [float(row["calculix_linear_lsq_fit"]) for row in rows], label="CalculiX LSQ fit", linestyle="-.")
    ax.set_xlabel("overclosure [mm]")
    ax.set_ylabel("pressure/force proxy")
    ax.set_title("Contact law alignment proxy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"figure_path": str(path), "source": "contact_law_alignment_proxy", "sample_count": len(rows)}


def _recurdyn_korder_force(model: RecurDynGearModel, overclosure: np.ndarray | float) -> np.ndarray:
    values = np.asarray(overclosure, dtype=float)
    order = max(1, int(model.contact.order))
    clipped = np.clip(values, 0.0, max(float(model.contact.max_penetration), 1.0e-12))
    return float(model.contact.stiffness) * clipped**order


def evaluate_contact_initialization(
    model: RecurDynGearModel,
    gap_rows: list[Row],
    *,
    activation_distance: float | None = None,
) -> tuple[list[Row], list[Row]]:
    """Initialize contact candidates from unsigned distance and the parsed RecurDyn law."""

    distance_limit = float(model.contact.bpen if activation_distance is None else activation_distance)
    distance_limit = max(distance_limit, 0.0)
    max_pen = max(float(model.contact.max_penetration), 1.0e-12)
    rows: list[Row] = []
    for row in gap_rows:
        unsigned_distance = float(row["unsigned_distance"])
        local_signed_gap = float(row["signed_gap"])
        overclosure = max(distance_limit - unsigned_distance, 0.0)
        clamped_overclosure = min(overclosure, max_pen)
        force_proxy = float(_recurdyn_korder_force(model, clamped_overclosure))
        active = overclosure > 0.0
        rows.append(
            {
                "sample_index": row["sample_index"],
                "sample_type": row["sample_type"],
                "sample_id": row["sample_id"],
                "nearest_face": row["nearest_face"],
                "unsigned_distance": unsigned_distance,
                "local_signed_gap": local_signed_gap,
                "activation_distance": distance_limit,
                "initial_overclosure_from_unsigned_distance": overclosure,
                "clamped_overclosure": clamped_overclosure,
                "overclosure_to_maxpen_ratio": clamped_overclosure / max_pen,
                "recurdyn_korder": int(model.contact.order),
                "recurdyn_k": float(model.contact.stiffness),
                "recurdyn_force_proxy": force_proxy,
                "active_contact_candidate": str(active).lower(),
                "local_negative_side_candidate": str(local_signed_gap < 0.0 and active).lower(),
            }
        )

    summary_rows: list[Row] = []
    for sample_type in ("node", "face_centroid", "all"):
        subset = rows if sample_type == "all" else [row for row in rows if row["sample_type"] == sample_type]
        if not subset:
            continue
        active_rows = [row for row in subset if row["active_contact_candidate"] == "true"]
        force = np.asarray([float(row["recurdyn_force_proxy"]) for row in active_rows], dtype=float) if active_rows else np.asarray([], dtype=float)
        overclosure = (
            np.asarray([float(row["initial_overclosure_from_unsigned_distance"]) for row in active_rows], dtype=float)
            if active_rows
            else np.asarray([], dtype=float)
        )
        local_negative_rows = [row for row in active_rows if row["local_negative_side_candidate"] == "true"]
        preload_recommended = bool(active_rows)
        contact_adjust_recommended = False
        if not active_rows:
            recommendation = "direct_dynamic_without_preload"
            reason = "No samples fall inside the unsigned-distance contact activation band."
        else:
            recommendation = "static_preload_then_dynamic_restart"
            reason = (
                "Unsigned-distance contact candidates exist under the RecurDyn BPEN activation band; "
                "initialize contact history with a static preload before dynamic continuation. "
                "CONTACT ADJUST is not recommended from this evidence alone because the open-surface local signed side is not a robust penetration classifier."
            )
        summary_rows.append(
            {
                "sample_type": sample_type,
                "sample_count": len(subset),
                "activation_distance": distance_limit,
                "active_candidate_count": len(active_rows),
                "local_negative_active_candidate_count": len(local_negative_rows),
                "max_initial_overclosure": float(np.max(overclosure)) if overclosure.size else 0.0,
                "mean_initial_overclosure": float(np.mean(overclosure)) if overclosure.size else 0.0,
                "max_recurdyn_force_proxy": float(np.max(force)) if force.size else 0.0,
                "sum_recurdyn_force_proxy": float(np.sum(force)) if force.size else 0.0,
                "preload_recommended": str(preload_recommended).lower(),
                "contact_adjust_recommended": str(contact_adjust_recommended).lower(),
                "recommended_initialization": recommendation,
                "decision_reason": reason,
            }
        )
    return rows, summary_rows


def _write_contact_initialization_plot(path: Path, rows: list[Row]) -> Row:
    path.parent.mkdir(parents=True, exist_ok=True)
    distances = np.asarray([float(row["unsigned_distance"]) for row in rows], dtype=float)
    forces = np.asarray([float(row["recurdyn_force_proxy"]) for row in rows], dtype=float)
    activation_distance = float(rows[0]["activation_distance"]) if rows else 0.0
    active_forces = forces[forces > 0.0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(distances, bins=100, color="#2563eb", alpha=0.85)
    axes[0].axvline(activation_distance, color="black", linewidth=1.0, label="activation distance")
    axes[0].set_xlim(0.0, max(activation_distance * 4.0, float(np.percentile(distances, 5)) if distances.size else 1.0e-3))
    axes[0].set_xlabel("unsigned closest distance [mm]")
    axes[0].set_ylabel("count")
    axes[0].set_title("Contact initialization band")
    axes[0].legend()
    if active_forces.size:
        axes[1].hist(active_forces, bins=60, color="#dc2626", alpha=0.85)
    else:
        axes[1].bar([0.0], [0.0], color="#dc2626")
    axes[1].set_xlabel("RecurDyn KORDER force proxy")
    axes[1].set_ylabel("count")
    axes[1].set_title("Active candidate force proxy")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"figure_path": str(path), "source": "contact_initialization", "sample_count": len(rows)}


def _contact_surface_behavior_lines(
    model: RecurDynGearModel,
    *,
    contact_law: str,
    effective_contact_stiffness: float,
) -> list[str]:
    """Return CalculiX pressure-overclosure lines for the selected proxy law.

    The parsed RecurDyn model uses ``KORDER=2``.  RecurDyn's exact commercial
    contact law is not reproduced here; these are explicit CalculiX-side proxy
    laws used to test which external-reference mapping is numerically usable.
    """

    if contact_law == "linear":
        return [
            "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
            f"{effective_contact_stiffness:.12e}",
        ]
    if contact_law == "exponential":
        return [
            "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=EXPONENTIAL",
            f"{max(model.contact.bpen, 1.0e-9):.12e}, {max(effective_contact_stiffness / 100000.0, 1.0e-9):.12e}",
        ]
    if contact_law == "tabular":
        max_pen = max(float(model.contact.max_penetration), float(model.contact.bpen), 1.0e-9)
        order = max(1, int(model.contact.order))
        rows = ["*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=TABULAR"]
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            overclosure = max_pen * fraction
            pressure = effective_contact_stiffness * overclosure**order
            rows.append(f"{pressure:.12e}, {overclosure:.12e}")
        return rows
    raise ValueError(f"unsupported contact law: {contact_law}")


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
    analysis: str,
    preload_rotation: float,
    contact_adjust: str | None,
    write_restart: bool = False,
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

    if analysis in {"static-preload", "preload-restart-dynamic"}:
        total_rotation = preload_rotation
    elif analysis == "preload-dynamic":
        total_rotation = preload_rotation + model.angular_velocity * duration
    else:
        total_rotation = model.angular_velocity * duration
    effective_contact_stiffness = model.contact.stiffness * contact_stiffness_scale
    pressure_lines = _contact_surface_behavior_lines(
        model,
        contact_law=contact_law,
        effective_contact_stiffness=effective_contact_stiffness,
    )

    if analysis in {"static-preload", "preload-restart-dynamic"}:
        dynamic_keyword = "*STATIC"
        dynamic_line = "1.0e-1, 1.0, 1.0e-8, 1.0e-1"
    elif explicit_dynamic:
        dynamic_keyword = "*DYNAMIC, DIRECT, EXPLICIT"
        dynamic_line = f"{dt:.12e}, {duration:.12e}"
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
    contact_pair_keyword = (
        "*CONTACT PAIR, INTERACTION=GEAR_CONTACT, TYPE=SURFACE TO SURFACE"
        if slave_surface_mode == "element-face"
        else "*CONTACT PAIR, INTERACTION=GEAR_CONTACT, TYPE=NODE TO SURFACE"
    )
    if contact_adjust:
        contact_pair_keyword += f", ADJUST={contact_adjust}"

    lines.extend(
        [
            "*SURFACE, NAME=GEAR21_MASTER, TYPE=ELEMENT",
            "GEAR21_SURF, SPOS",
            "*SURFACE INTERACTION, NAME=GEAR_CONTACT",
            *pressure_lines,
            contact_pair_keyword,
            "GEAR22_SLAVE, GEAR21_MASTER",
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
    def _append_motion_boundary(angle: float) -> None:
        if drive_mode == "rigid-body":
            lines.append(f"{rot_node}, {axis_dof}, {axis_dof}, {angle:.12e}")
        elif drive_mode == "prescribed-surface":
            displacement_rows = _rigid_rotation_displacement(
                model.rigid_surface.nodes_global,
                center=ref_point,
                axis=axis,
                angle=angle,
            )
            for node_id, displacement in zip(rigid_node_ids, displacement_rows):
                ux, uy, uz = displacement
                lines.append(f"{int(node_id)}, 1, 1, {ux:.12e}")
                lines.append(f"{int(node_id)}, 2, 2, {uy:.12e}")
                lines.append(f"{int(node_id)}, 3, 3, {uz:.12e}")
        else:
            lines.append("GEAR21_RIGID_NODES, 1, 3, 0.0")

    def _append_print_requests(include_velocity: bool) -> None:
        lines.extend([f"*NODE PRINT, NSET=GEAR22_NALL, FREQUENCY={output_every}", "U"])
        if include_velocity:
            lines.extend([f"*NODE PRINT, NSET=GEAR22_NALL, FREQUENCY={output_every}", "V"])
        lines.extend(
            [
                f"*NODE PRINT, NSET=GEAR22_HUB, TOTALS=ONLY, GLOBAL=YES, FREQUENCY={output_every}",
                "RF",
                f"*CONTACT PRINT, FREQUENCY={output_every}",
                "CDIS,CSTR,CELS",
                f"*CONTACT PRINT, TOTALS=ONLY, FREQUENCY={output_every}",
                "CELS,CNUM",
                f"*EL PRINT, ELSET=GEAR22_SOLID, FREQUENCY={output_every}",
                "S,E",
            ]
        )

    def _append_step(keyword: str, line: str, *, amplitude: str, angle: float, include_velocity: bool, restart_write: bool = False) -> None:
        lines.extend(["*STEP, NLGEOM, INC=1000000", keyword, line, f"*BOUNDARY, AMPLITUDE={amplitude}"])
        _append_motion_boundary(angle)
        _append_print_requests(include_velocity)
        if restart_write:
            lines.append("*RESTART, WRITE, FREQUENCY=1")
        lines.append("*END STEP")

    if analysis == "preload-dynamic":
        preload_fraction = preload_rotation / total_rotation if abs(total_rotation) > 1.0e-30 else 0.0
        lines.extend(
            [
                "*AMPLITUDE, NAME=PRELOADAMP",
                "0.0, 0.0, 1.0, 1.0",
                "*AMPLITUDE, NAME=ROTAMP",
                f"1.0, {preload_fraction:.12e}, {1.0 + duration:.12e}, 1.0",
            ]
        )
        _append_step("*STATIC", "1.0e-1, 1.0, 1.0e-8, 1.0e-1", amplitude="PRELOADAMP", angle=preload_rotation, include_velocity=False)
        _append_step(dynamic_keyword, dynamic_line, amplitude="ROTAMP", angle=total_rotation, include_velocity=True)
    else:
        lines.extend(["*AMPLITUDE, NAME=ROTAMP", f"0.0, 0.0, {duration:.12e}, 1.0"])
        _append_step(
            dynamic_keyword,
            dynamic_line,
            amplitude="ROTAMP",
            angle=total_rotation,
            include_velocity=analysis not in {"static-preload", "preload-restart-dynamic"},
            restart_write=write_restart,
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
        "rigid_surface_marker_id": model.rigid_surface.marker_id,
        "rigid_surface_marker_part_id": model.rigid_surface.marker_pose.part_id or "",
        "rigid_surface_marker_qp_x": model.rigid_surface.marker_pose.qp[0],
        "rigid_surface_marker_qp_y": model.rigid_surface.marker_pose.qp[1],
        "rigid_surface_marker_qp_z": model.rigid_surface.marker_pose.qp[2],
        "rigid_surface_marker_reuler_x": model.rigid_surface.marker_pose.reuler[0],
        "rigid_surface_marker_reuler_y": model.rigid_surface.marker_pose.reuler[1],
        "rigid_surface_marker_reuler_z": model.rigid_surface.marker_pose.reuler[2],
        "rigid_surface_part_qg_x": model.rigid_surface.part_pose.qg[0],
        "rigid_surface_part_qg_y": model.rigid_surface.part_pose.qg[1],
        "rigid_surface_part_qg_z": model.rigid_surface.part_pose.qg[2],
        "rigid_surface_part_reuler_x": model.rigid_surface.part_pose.reuler[0],
        "rigid_surface_part_reuler_y": model.rigid_surface.part_pose.reuler[1],
        "rigid_surface_part_reuler_z": model.rigid_surface.part_pose.reuler[2],
        "analysis": analysis,
        "preload_rotation": preload_rotation,
        "contact_adjust": contact_adjust or "",
        "restart_write": str(write_restart).lower(),
    }


def _write_calculix_restart_input(
    path: Path,
    model: RecurDynGearModel,
    *,
    duration: float,
    dt: float,
    output_every: int,
    rotation_axis: str,
    direct_dynamic: bool,
    explicit_dynamic: bool,
    drive_mode: str,
    preload_rotation: float,
) -> dict[str, Any]:
    """Write the dynamic restart step following a completed preload job."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rigid_node_offset = 200000
    ref_node = 900001
    rot_node = 900002
    axis_dof = _axis_index(rotation_axis)
    axis = _axis_vector(rotation_axis)
    ref_point = np.asarray([20.9136192988016, 0.503589451286037, 1.36499444260163], dtype=float)
    rigid_node_ids = [rigid_node_offset + i + 1 for i in range(model.rigid_surface.nodes_global.shape[0])]
    total_rotation = preload_rotation + model.angular_velocity * duration
    preload_fraction = preload_rotation / total_rotation if abs(total_rotation) > 1.0e-30 else 0.0

    if explicit_dynamic:
        dynamic_keyword = "*DYNAMIC, DIRECT, EXPLICIT"
        dynamic_line = f"{dt:.12e}, {duration:.12e}"
    else:
        dynamic_keyword = "*DYNAMIC, DIRECT" if direct_dynamic else "*DYNAMIC"
        dynamic_line = f"{dt:.12e}, {duration:.12e}" if direct_dynamic else f"{dt:.12e}, {duration:.12e}, {0.01 * dt:.12e}, {dt:.12e}"

    lines: list[str] = [
        "** Dynamic restart deck generated from the static preload job.",
        "** Copy the preload .rout file to this job name as .rin before running ccx.",
        "*RESTART, READ, STEP=1",
        "*AMPLITUDE, NAME=ROTAMP",
        f"0.0, {preload_fraction:.12e}, {duration:.12e}, 1.0",
        "*STEP, NLGEOM, INC=1000000",
        dynamic_keyword,
        dynamic_line,
        "*BOUNDARY, AMPLITUDE=ROTAMP",
    ]

    if drive_mode == "rigid-body":
        lines.append(f"{rot_node}, {axis_dof}, {axis_dof}, {total_rotation:.12e}")
    elif drive_mode == "prescribed-surface":
        displacement_rows = _rigid_rotation_displacement(
            model.rigid_surface.nodes_global,
            center=ref_point,
            axis=axis,
            angle=total_rotation,
        )
        for node_id, displacement in zip(rigid_node_ids, displacement_rows):
            ux, uy, uz = displacement
            lines.append(f"{int(node_id)}, 1, 1, {ux:.12e}")
            lines.append(f"{int(node_id)}, 2, 2, {uy:.12e}")
            lines.append(f"{int(node_id)}, 3, 3, {uz:.12e}")

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
        "restart_inp_path": str(path),
        "restart_duration": duration,
        "restart_dt": dt,
        "restart_total_rotation": total_rotation,
        "restart_output_every": output_every,
        "restart_dynamic_keyword": dynamic_keyword,
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


def _combine_restart_run_rows(preload: Row, restart: Row | None, *, restart_file_copied: bool) -> Row:
    preload_return = int(preload.get("return_code", -1))
    restart_return = int(restart.get("return_code", -1)) if restart is not None else preload_return
    timed_out = str(preload.get("timed_out", "false")).lower() == "true"
    if restart is not None:
        timed_out = timed_out or str(restart.get("timed_out", "false")).lower() == "true"
    return {
        "command": preload.get("command", "") + (" && " + str(restart.get("command", "")) if restart is not None else ""),
        "return_code": restart_return,
        "timed_out": str(timed_out).lower(),
        "wall_time_seconds": float(preload.get("wall_time_seconds", 0.0))
        + (float(restart.get("wall_time_seconds", 0.0)) if restart is not None else 0.0),
        "stdout_tail": "PRELOAD:\n"
        + str(preload.get("stdout_tail", ""))
        + ("\n\nRESTART:\n" + str(restart.get("stdout_tail", "")) if restart is not None else "\n\nRESTART: not run"),
        "stderr_tail": "PRELOAD:\n"
        + str(preload.get("stderr_tail", ""))
        + ("\n\nRESTART:\n" + str(restart.get("stderr_tail", "")) if restart is not None else "\n\nRESTART: not run"),
        "preload_return_code": preload_return,
        "restart_return_code": restart_return if restart is not None else "",
        "restart_file_copied": str(restart_file_copied).lower(),
        "restart_attempted": str(restart is not None).lower(),
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
    gap_summary: list[Row],
    law_summary: list[Row],
    init_summary: list[Row],
    run_row: Row | None,
    frame_rows: list[Row],
    totals: dict[float, Row],
    contact_rows: dict[float, list[Row]],
    command_line: str,
) -> None:
    run_completed = run_row is not None and int(run_row.get("return_code", -1)) == 0
    vtk_status = "generated_from_calculix_dat" if frame_rows and frame_rows[0]["source"] == "calculix_dat" else "placeholder_or_no_dat"
    all_gap = gap_summary[-1] if gap_summary else {}
    best_law = min(law_summary, key=lambda row: float(row["relative_rms_error"])) if law_summary else {}
    init_all = init_summary[-1] if init_summary else {}
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
        "## Initial contact-distance diagnostic",
        "",
        f"- Samples: {all_gap.get('count', 0)}",
        f"- Minimum unsigned closest distance: {float(all_gap.get('min_unsigned_distance', 0.0)):.6e} mm",
        f"- Near-zero unsigned distance count (`<1e-3 mm`): {all_gap.get('near_zero_distance_count_1e_3', 0)}",
        f"- Minimum local signed gap: {float(all_gap.get('min_signed_gap', 0.0)):.6e} mm",
        f"- Negative local signed-gap count: {all_gap.get('negative_signed_gap_count', 0)}",
        "- Interpretation: unsigned distance is the geometric closest-distance metric. The signed local gap only indicates the side of the nearest oriented open surface patch and is not, by itself, a robust penetration classification for this open gear surface.",
        "",
        "## RMD rigid-surface transform",
        "",
        f"- Rigid GGEOM marker id: {generation['rigid_surface_marker_id']}",
        f"- Marker PART id: {generation['rigid_surface_marker_part_id']}",
        f"- Marker QP: ({float(generation['rigid_surface_marker_qp_x']):.6e}, {float(generation['rigid_surface_marker_qp_y']):.6e}, {float(generation['rigid_surface_marker_qp_z']):.6e})",
        f"- Marker REULER: ({float(generation['rigid_surface_marker_reuler_x']):.6e}, {float(generation['rigid_surface_marker_reuler_y']):.6e}, {float(generation['rigid_surface_marker_reuler_z']):.6e})",
        f"- Part QG: ({float(generation['rigid_surface_part_qg_x']):.6e}, {float(generation['rigid_surface_part_qg_y']):.6e}, {float(generation['rigid_surface_part_qg_z']):.6e})",
        f"- Part REULER: ({float(generation['rigid_surface_part_reuler_x']):.6e}, {float(generation['rigid_surface_part_reuler_y']):.6e}, {float(generation['rigid_surface_part_reuler_z']):.6e})",
        "- Rigid GGEOM nodes are transformed as `x_global = T_part * T_marker * x_local`.",
        "",
        "## Contact law alignment proxy",
        "",
        f"- Best linear proxy fit: {best_law.get('fit', 'n/a')}",
        f"- Best linear proxy slope: {float(best_law.get('linear_slope', 0.0)):.6e}",
        f"- Relative RMS proxy error: {float(best_law.get('relative_rms_error', 0.0)):.6e}",
        "- This is a proxy only; RecurDyn's exact `KORDER=2` internal law is not assumed to equal this polynomial model.",
        "",
        "## Contact initialization from unsigned distance",
        "",
        f"- Activation distance: {float(init_all.get('activation_distance', model.contact.bpen)):.6e} mm",
        f"- Active unsigned-distance candidates: {init_all.get('active_candidate_count', 0)} / {init_all.get('sample_count', 0)}",
        f"- Maximum initialization overclosure: {float(init_all.get('max_initial_overclosure', 0.0)):.6e} mm",
        f"- Maximum RecurDyn KORDER force proxy: {float(init_all.get('max_recurdyn_force_proxy', 0.0)):.6e}",
        f"- Sum RecurDyn KORDER force proxy: {float(init_all.get('sum_recurdyn_force_proxy', 0.0)):.6e}",
        f"- Preload recommended: {init_all.get('preload_recommended', 'false')}",
        f"- CONTACT ADJUST recommended: {init_all.get('contact_adjust_recommended', 'false')}",
        f"- Recommended initialization: {init_all.get('recommended_initialization', 'n/a')}",
        f"- Decision reason: {init_all.get('decision_reason', 'n/a')}",
        "- The initialization force is a sample-level RecurDyn `KORDER` proxy for deciding start-up strategy; it is not a physical integrated resultant.",
        "",
        "## CalculiX run",
        "",
        f"- Input deck: `{_display(Path(generation['inp_path']))}`",
        f"- Restart dynamic deck: `{_display(Path(str(generation.get('restart_inp_path', generation['inp_path']))))}`" if generation.get("restart_inp_path") else "- Restart dynamic deck: not used",
        f"- Analysis mode: {generation['analysis']}",
        f"- Duration: {generation['duration']} s",
        f"- Time step request: {generation['dt']} s",
        f"- Output frequency: every {generation['output_every']} increments",
        f"- Rotation axis: {generation['rotation_axis']}",
        f"- Rigid surface drive mode: {generation['drive_mode']}",
        f"- Slave surface mode: {generation['slave_surface_mode']}",
        f"- Contact law proxy: {generation['contact_law']}",
        f"- Effective CalculiX contact stiffness: {float(generation['effective_contact_stiffness']):.6e}",
        f"- Total prescribed rigid gear rotation: {generation['total_rotation']} rad",
        f"- Restart target total rotation: {generation.get('restart_total_rotation', 'not used')} rad",
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
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--output-every", type=int, default=20)
    parser.add_argument("--analysis", choices=["dynamic", "static-preload", "preload-dynamic", "preload-restart-dynamic"], default="dynamic")
    parser.add_argument("--preload-rotation", type=float, default=0.0)
    parser.add_argument("--rotation-axis", choices=["x", "y", "z"], default="x")
    parser.add_argument("--contact-law", choices=["linear", "exponential", "tabular"], default="linear")
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
        "--contact-adjust",
        default=None,
        help="Optional CalculiX CONTACT PAIR ADJUST value, e.g. 0.0 to adjust initially overclosed slave nodes.",
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
    parser.add_argument("--gap-diagnostic-only", action="store_true")
    parser.add_argument("--gap-candidate-count", type=int, default=96)
    parser.add_argument(
        "--contact-init-distance",
        type=float,
        default=None,
        help="Unsigned closest-distance band used to initialize contact candidates; defaults to RecurDyn BPEN.",
    )
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
    gap_rows, gap_summary = evaluate_initial_gap_diagnostics(model, candidate_count=args.gap_candidate_count)
    law_rows, law_summary = evaluate_contact_law_alignment(model)
    init_rows, init_summary = evaluate_contact_initialization(model, gap_rows, activation_distance=args.contact_init_distance)
    _write_csv(out_dir / "initial_gap_samples.csv", list(gap_rows[0].keys()), gap_rows)
    _write_csv(out_dir / "initial_gap_summary.csv", list(gap_summary[0].keys()), gap_summary)
    gap_plot = _write_gap_histogram(out_dir / "figures" / "initial_gap_histogram.png", gap_rows)
    _write_csv(out_dir / "contact_law_alignment.csv", list(law_rows[0].keys()), law_rows)
    _write_csv(out_dir / "contact_law_alignment_summary.csv", list(law_summary[0].keys()), law_summary)
    law_plot = _write_contact_law_plot(out_dir / "figures" / "contact_law_alignment.png", law_rows)
    _write_csv(out_dir / "contact_initialization_samples.csv", list(init_rows[0].keys()), init_rows)
    _write_csv(out_dir / "contact_initialization_summary.csv", list(init_summary[0].keys()), init_summary)
    init_plot = _write_contact_initialization_plot(out_dir / "figures" / "contact_initialization.png", init_rows)

    inp_path = out_dir / "jiandanjiaolian_calculix_1s.inp"
    is_restart_strategy = args.analysis == "preload-restart-dynamic"
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
        analysis=args.analysis,
        preload_rotation=args.preload_rotation,
        contact_adjust=args.contact_adjust,
        write_restart=is_restart_strategy,
    )
    restart_inp_path: Path | None = None
    if is_restart_strategy:
        restart_inp_path = out_dir / "jiandanjiaolian_calculix_1s_restart.inp"
        restart_generation = _write_calculix_restart_input(
            restart_inp_path,
            model,
            duration=duration,
            dt=args.dt,
            output_every=output_every,
            rotation_axis=args.rotation_axis,
            direct_dynamic=not args.automatic_increment,
            explicit_dynamic=args.explicit,
            drive_mode=args.drive_mode,
            preload_rotation=args.preload_rotation,
        )
        generation.update(restart_generation)

    run_row: Row | None = None
    if not args.skip_calculix and not args.gap_diagnostic_only:
        if is_restart_strategy and restart_inp_path is not None:
            preload_row = _run_calculix(inp_path, out_dir / "logs" / "calculix_preload_run.log", timeout=args.timeout)
            restart_row: Row | None = None
            restart_file_copied = False
            if int(preload_row.get("return_code", -1)) == 0:
                rout_path = inp_path.with_suffix(".rout")
                rin_path = restart_inp_path.with_suffix(".rin")
                if rout_path.exists():
                    shutil.copyfile(rout_path, rin_path)
                    restart_file_copied = True
                    restart_row = _run_calculix(restart_inp_path, out_dir / "logs" / "calculix_restart_run.log", timeout=args.timeout)
            run_row = _combine_restart_run_rows(preload_row, restart_row, restart_file_copied=restart_file_copied)
        else:
            run_row = _run_calculix(inp_path, out_dir / "logs" / "calculix_run.log", timeout=args.timeout)

    dat_path = inp_path.with_suffix(".dat")
    if restart_inp_path is not None and restart_inp_path.with_suffix(".dat").exists():
        dat_path = restart_inp_path.with_suffix(".dat")
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
    figure_rows = [preview, gap_plot, law_plot, init_plot]
    _write_csv(out_dir / "recurdyn_gear_calculix_figures.csv", sorted({key for row in figure_rows for key in row}), figure_rows)

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
            "min_initial_unsigned_distance": gap_summary[-1]["min_unsigned_distance"],
            "near_zero_initial_distance_count_1e_3": gap_summary[-1]["near_zero_distance_count_1e_3"],
            "min_initial_local_signed_gap": gap_summary[-1]["min_signed_gap"],
            "negative_initial_local_signed_gap_count": gap_summary[-1]["negative_signed_gap_count"],
            "contact_law_best_lsq_slope": law_summary[-1]["linear_slope"],
            "contact_init_active_candidate_count": init_summary[-1]["active_candidate_count"],
            "contact_init_max_force_proxy": init_summary[-1]["max_recurdyn_force_proxy"],
            "contact_init_preload_recommended": init_summary[-1]["preload_recommended"],
            "contact_init_adjust_recommended": init_summary[-1]["contact_adjust_recommended"],
            "contact_init_recommendation": init_summary[-1]["recommended_initialization"],
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
        gap_summary=gap_summary,
        law_summary=law_summary,
        init_summary=init_summary,
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
