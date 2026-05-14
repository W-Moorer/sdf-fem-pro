"""CalculiX drop-impact reference comparison.

This validation-only runner uses CalculiX/ccx as an external transient dynamic
contact reference.  The runner generates a small deterministic TET4
sphere-like elastic body impacting a fixed rigid plane, then compares CalculiX
time-history data against the current SFC linear TET4 HHT-alpha/penalty-contact
prototype on the same mesh, material, gravity, initial velocity, and rigid-plane
location.

The comparison is deliberately scoped.  It is external solver evidence for a
clean transient dynamic contact scenario, not a claim that the current SFC
prototype matches all CalculiX nonlinear contact details.
"""

from __future__ import annotations

import argparse
import csv
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
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

from sfc.fem import DeformableBody, assemble_gravity_force, assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.fem.tet4 import tet4_volume  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.mesh.topology import orient_tet4_connectivity  # noqa: E402
from validation.run_phase4_paper_validation import _format_float  # noqa: E402

Row = dict[str, Any]

DEFAULT_CASE_NAME = "sfc_calculix_ball_drop"
FLOOR_SHELL_THICKNESS = 1.0e-2
CALCULIX_MASTER_SURFACE_OFFSET = 0.5 * FLOOR_SHELL_THICKNESS
CONTACT_SMOOTHING_EPSILON = 2.5e-3
CALCULIX_DEFAULT_HHT_ALPHA = -0.05
REBOUND_HEIGHT_TOLERANCE = 1.0e-6
TRIANGLE_QUADRATURE_BARYCENTRIC = np.asarray(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ],
    dtype=float,
)
TRIANGLE_QUADRATURE_WEIGHTS = np.full(3, 1.0 / 3.0, dtype=float)

C3D4_FACE_LABELS = {
    (0, 1, 2): "S1",
    (0, 1, 3): "S2",
    (1, 2, 3): "S3",
    (0, 2, 3): "S4",
}
LOCAL_FACE_NODES = {
    "S1": (0, 1, 2),
    "S2": (0, 1, 3),
    "S3": (1, 2, 3),
    "S4": (0, 2, 3),
}


@dataclass(slots=True)
class DropModel:
    """Generated drop-impact model shared by CalculiX and SFC."""

    case: str
    resolution: int
    node_ids: np.ndarray
    nodes: np.ndarray
    tet_elements: np.ndarray
    surface_faces: np.ndarray
    slave_face_refs: list[tuple[int, str]]
    surface_node_ids: np.ndarray
    surface_node_areas: np.ndarray
    floor_nodes: np.ndarray
    floor_element_ids: np.ndarray
    floor_z: float
    E: float
    nu: float
    density: float
    gravity: float
    initial_velocity_z: float
    contact_stiffness: float
    total_time: float
    dt: float
    output_frequency: int
    direct_dynamic: bool
    sfc_substeps: int
    contact_smoothing_epsilon: float
    hht_alpha: float
    model_source: str

    @property
    def id_to_index(self) -> dict[int, int]:
        return {int(node_id): i for i, node_id in enumerate(self.node_ids)}

    @property
    def surface_indices(self) -> np.ndarray:
        lookup = self.id_to_index
        return np.asarray([lookup[int(node_id)] for node_id in self.surface_node_ids], dtype=np.int64)


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=timeout)


def _wsl_available() -> bool:
    if shutil.which("wsl") is None:
        return False
    try:
        _run(["wsl", "--exec", "bash", "-lc", "true"], cwd=ROOT, timeout=15)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def calculix_available() -> bool:
    """Return whether CalculiX/ccx is available through WSL."""

    if not _wsl_available():
        return False
    try:
        _run(["wsl", "--exec", "bash", "-lc", "command -v ccx >/dev/null"], cwd=ROOT, timeout=20)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _calculix_version() -> str:
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", "ccx -v 2>&1"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "unknown"


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    parts = [part for part in resolved.parts[1:]]
    if not drive:
        return resolved.as_posix()
    return "/mnt/" + drive + "/" + "/".join(part.replace("\\", "/") for part in parts)


def _boundary_face_data(nodes: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, list[tuple[int, str]], np.ndarray, np.ndarray]:
    """Return boundary faces, CalculiX face refs, node ids, and nodal areas."""

    face_records: dict[tuple[int, int, int], list[tuple[int, str, tuple[int, int, int]]]] = {}
    for element_id, element in enumerate(elements, start=1):
        for label, local in LOCAL_FACE_NODES.items():
            face = tuple(int(element[i]) for i in local)
            key = tuple(sorted(face))
            face_records.setdefault(key, []).append((element_id, label, face))

    faces: list[tuple[int, int, int]] = []
    refs: list[tuple[int, str]] = []
    for records in face_records.values():
        if len(records) == 1:
            element_id, label, face = records[0]
            faces.append(face)
            refs.append((element_id, label))

    surface_faces = np.asarray(faces, dtype=np.int64)
    surface_indices = np.unique(surface_faces.ravel())
    areas = np.zeros(nodes.shape[0], dtype=float)
    for face in surface_faces:
        a, b, c = nodes[face]
        area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        areas[face] += area / 3.0
    return surface_faces, refs, surface_indices, areas


def _floor(half_width: float, z: float, *, start_id: int) -> tuple[np.ndarray, np.ndarray]:
    floor_nodes = np.asarray(
        [
            [start_id, -half_width, -half_width, z],
            [start_id + 1, half_width, -half_width, z],
            [start_id + 2, half_width, half_width, z],
            [start_id + 3, -half_width, half_width, z],
        ],
        dtype=float,
    )
    floor_elements = np.asarray([[0, start_id, start_id + 1, start_id + 2, start_id + 3]], dtype=np.int64)
    return floor_nodes, floor_elements


def _structured_block_mesh(resolution: int, *, size: tuple[float, float, float], bottom_z: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(resolution)
    lx, ly, lz = size
    nodes: list[tuple[float, float, float]] = []
    node_id: dict[tuple[int, int, int], int] = {}
    for k in range(n + 1):
        z = bottom_z + lz * k / n
        for j in range(n + 1):
            y = -0.5 * ly + ly * j / n
            for i in range(n + 1):
                x = -0.5 * lx + lx * i / n
                node_id[(i, j, k)] = len(nodes)
                nodes.append((x, y, z))

    tets: list[tuple[int, int, int, int]] = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                v000 = node_id[(i, j, k)]
                v100 = node_id[(i + 1, j, k)]
                v010 = node_id[(i, j + 1, k)]
                v110 = node_id[(i + 1, j + 1, k)]
                v001 = node_id[(i, j, k + 1)]
                v101 = node_id[(i + 1, j, k + 1)]
                v011 = node_id[(i, j + 1, k + 1)]
                v111 = node_id[(i + 1, j + 1, k + 1)]
                tets.extend(
                    [
                        (v000, v100, v010, v001),
                        (v100, v110, v010, v111),
                        (v100, v010, v001, v111),
                        (v010, v001, v011, v111),
                        (v100, v001, v101, v111),
                    ]
                )
    nodes_arr = np.asarray(nodes, dtype=float)
    return nodes_arr, orient_tet4_connectivity(nodes_arr, np.asarray(tets, dtype=np.int64))


def _sphere_like_mesh(resolution: int, *, radius: float, center_z: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import Delaunay

    level = max(1, int(resolution))
    n_theta = 12 + 4 * level
    n_surface_layers = 5 + 2 * level
    flat_radius = 0.24 * radius
    bottom_z = center_z - radius
    cap_z = center_z - np.sqrt(max(radius * radius - flat_radius * flat_radius, 0.0))

    pts: list[tuple[float, float, float]] = []

    def add_ring(z: float, ring_radius: float, count: int = n_theta) -> None:
        for i in range(count):
            theta = 2.0 * np.pi * i / count
            pts.append((float(ring_radius * np.cos(theta)), float(ring_radius * np.sin(theta)), float(z)))

    # A tiny flat contact patch avoids the vertex-only contact that makes
    # CalculiX face-to-face contact activate after large geometric penetration.
    pts.append((0.0, 0.0, float(bottom_z)))
    add_ring(float(bottom_z), 0.50 * flat_radius)
    add_ring(float(bottom_z), flat_radius)

    # Regular latitude rings on the rounded part of the sphere-like body.
    z_values = np.linspace(float(cap_z), float(center_z + radius), n_surface_layers)
    for z in z_values[:-1]:
        dz = z - center_z
        ring_radius = np.sqrt(max(radius * radius - dz * dz, 0.0))
        add_ring(float(z), float(max(ring_radius, flat_radius)))
    pts.append((0.0, 0.0, float(center_z + radius)))

    # Interior axis and half-radius rings make the Delaunay tetrahedra more
    # regular without changing the boundary contact patch.
    interior_z_values = np.linspace(float(bottom_z + 0.15 * radius), float(center_z + 0.75 * radius), max(3, level + 3))
    for z in interior_z_values:
        dz = z - center_z
        outer_radius = np.sqrt(max(radius * radius - dz * dz, 0.0))
        pts.append((0.0, 0.0, float(z)))
        if outer_radius > 0.12 * radius:
            add_ring(float(z), 0.45 * float(outer_radius), max(8, n_theta // 2))

    nodes = np.unique(np.asarray(pts, dtype=float), axis=0)
    tets = np.asarray(Delaunay(nodes).simplices, dtype=np.int64)
    coords = nodes[tets]
    dets = np.linalg.det(np.stack((coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0], coords[:, 3] - coords[:, 0]), axis=2))
    tets = tets[np.abs(dets) > 1.0e-12]
    return nodes, orient_tet4_connectivity(nodes, tets)


def build_drop_model(
    *,
    quick: bool,
    case: str = "sphere_like_drop",
    resolution: int = 0,
    duration: float | None = None,
    dt: float | None = None,
    output_frequency: int = 1,
    direct_dynamic: bool = True,
    initial_velocity_z: float | None = None,
    gravity: float | None = None,
    contact_stiffness_override: float | None = None,
    hht_alpha: float | None = None,
) -> DropModel:
    """Build the shared CalculiX/SFC drop-impact model."""

    radius = 0.5
    initial_gap = 0.05
    floor_z = 0.0
    if case == "sphere_like_drop":
        center_z = floor_z + initial_gap + radius
        nodes, tet_elements = _sphere_like_mesh(int(resolution), radius=radius, center_z=center_z)
        floor_half_width = 1.25
        model_source = "regularized_sphere_like_tet4_with_bottom_contact_patch"
        total_time = 0.05 if quick else 0.06
        contact_stiffness = 2.0e5
    elif case == "block_drop":
        block_height = 0.8
        nodes, tet_elements = _structured_block_mesh(max(1, int(resolution)), size=(0.8, 0.8, block_height), bottom_z=floor_z + initial_gap)
        floor_half_width = 1.15
        model_source = "generated_structured_block_tet4"
        total_time = 0.08 if quick else 0.14
        contact_stiffness = 2.0e4
    else:
        raise ValueError("case must be 'sphere_like_drop' or 'block_drop'")

    surface_faces, slave_face_refs, surface_indices, surface_areas = _boundary_face_data(nodes, tet_elements)
    node_ids = np.arange(1, nodes.shape[0] + 1, dtype=np.int64)
    surface_node_ids = node_ids[surface_indices]
    floor_nodes, floor_elements = _floor(floor_half_width, floor_z, start_id=int(node_ids[-1]) + 1)

    if duration is not None:
        total_time = float(duration)
    dt_value = 5.0e-4 if dt is None else float(dt)
    velocity_value = -2.0 if initial_velocity_z is None else float(initial_velocity_z)
    gravity_value = 9.81 if gravity is None else float(gravity)
    contact_stiffness_value = float(contact_stiffness) if contact_stiffness_override is None else float(contact_stiffness_override)
    hht_alpha_value = CALCULIX_DEFAULT_HHT_ALPHA if hht_alpha is None else float(hht_alpha)
    _hht_newmark_parameters(hht_alpha_value)

    return DropModel(
        case=case,
        resolution=int(resolution),
        node_ids=node_ids,
        nodes=nodes,
        tet_elements=tet_elements,
        surface_faces=surface_faces,
        slave_face_refs=slave_face_refs,
        surface_node_ids=surface_node_ids,
        surface_node_areas=surface_areas,
        floor_nodes=floor_nodes,
        floor_element_ids=floor_elements,
        floor_z=floor_z,
        E=1000.0,
        nu=0.3,
        density=1.0,
        gravity=gravity_value,
        initial_velocity_z=velocity_value,
        contact_stiffness=contact_stiffness_value,
        total_time=total_time,
        dt=dt_value,
        output_frequency=max(1, int(output_frequency)),
        direct_dynamic=bool(direct_dynamic),
        sfc_substeps=2 if quick else 4,
        contact_smoothing_epsilon=CONTACT_SMOOTHING_EPSILON,
        hht_alpha=hht_alpha_value,
        model_source=model_source,
    )


def build_model_suite(
    *,
    quick: bool,
    duration: float | None = None,
    dt: float | None = None,
    output_frequency: int = 1,
    direct_dynamic: bool = True,
    cases: list[str] | tuple[str, ...] | None = None,
    resolutions: list[int] | tuple[int, ...] | None = None,
    initial_velocity_z: float | None = None,
    gravity: float | None = None,
    contact_stiffness_override: float | None = None,
    hht_alpha: float | None = None,
) -> list[DropModel]:
    """Return the model suite for quick or paper-scale external comparison."""

    case_names = ["sphere_like_drop", "block_drop"] if cases is None else list(cases)
    resolution_values = [1] if quick else [1, 2, 3]
    if resolutions is not None:
        resolution_values = [int(value) for value in resolutions]
    models: list[DropModel] = []
    for case_name in case_names:
        for resolution in resolution_values:
            models.append(
                build_drop_model(
                    quick=quick,
                    case=case_name,
                    resolution=resolution,
                    duration=duration,
                    dt=dt,
                    output_frequency=output_frequency,
                    direct_dynamic=direct_dynamic,
                    initial_velocity_z=initial_velocity_z,
                    gravity=gravity,
                    contact_stiffness_override=contact_stiffness_override,
                    hht_alpha=hht_alpha,
                )
            )
    return models


def _format_id_list(ids: np.ndarray, *, width: int = 10) -> list[str]:
    values = [str(int(value)) for value in ids]
    return [", ".join(values[i : i + width]) + "," for i in range(0, len(values), width)]


def write_calculix_input(model: DropModel, path: Path) -> None:
    """Write a deterministic CalculiX C3D4 drop-impact input file."""

    id_from_index = model.node_ids
    dynamic_keyword = f"*dynamic, alpha={model.hht_alpha:.12g}"
    if model.direct_dynamic:
        dynamic_keyword = f"*dynamic, direct, alpha={model.hht_alpha:.12g}"
    min_dt = max(1.0e-9, 1.0e-4 * model.dt)
    output_frequency = max(1, int(model.output_frequency))
    lines: list[str] = [
        "** Generated by validation/run_calculix_drop_impact_comparison.py",
        "** External reference: CalculiX/ccx transient dynamic surface-to-surface contact.",
        "*NODE, NSET=Nall",
    ]
    for node_id, coord in zip(model.node_ids, model.nodes, strict=True):
        lines.append(f"{int(node_id)}, {coord[0]:.12e}, {coord[1]:.12e}, {coord[2]:.12e}")

    lines.append("*ELEMENT, TYPE=C3D4, ELSET=elall")
    for element_id, tet in enumerate(model.tet_elements, start=1):
        ids = [int(id_from_index[int(i)]) for i in tet]
        lines.append(f"{element_id}, {ids[0]}, {ids[1]}, {ids[2]}, {ids[3]}")

    lines.append("*node, nset=nfloor")
    for row in model.floor_nodes:
        lines.append(f"{int(row[0])}, {row[1]:.12e}, {row[2]:.12e}, {row[3]:.12e}")
    lines.append("*element, type=s4, elset=efloor")
    floor_element_start = model.tet_elements.shape[0] + 1
    for offset, row in enumerate(model.floor_element_ids):
        values = [floor_element_start + offset] + [int(value) for value in row[1:]]
        lines.append(", ".join(str(int(value)) for value in values))

    lines.append("*NSET,NSET=nsurface")
    lines.extend(_format_id_list(model.surface_node_ids))
    lines.append("*surface, name=ball, type=element")
    for element_id, label in model.slave_face_refs:
        lines.append(f"{int(element_id)}, {label}")
    lines.extend(
        [
            "*boundary",
            "nfloor, 1, 6,",
            "*material, name=gummi",
            "*elastic",
            f"{model.E:.12g}, {model.nu:.12g}",
            "*density",
            f"{model.density:.12g}",
            "*solid section, elset=elall, material=gummi",
            "*shell section, elset=efloor, material=gummi",
            f"{FLOOR_SHELL_THICKNESS:.12g}",
            "*initial conditions, type=velocity",
            f"nall, 3, {model.initial_velocity_z:.12g}",
            "*surface, name=floor, type=element",
            "efloor, SPOS",
            "*contact pair, interaction=contact,TYPE=SURFACE TO SURFACE",
            "ball, floor",
            "*surface interaction, name=contact",
            "*surface behavior, pressure-overclosure=linear",
            f"{model.contact_stiffness:.12g}",
            "*step, nlgeom, inc=1000000",
            dynamic_keyword,
            f"{model.dt:.12g}, {model.total_time:.12g}, {min_dt:.12g}, {model.dt:.12g}",
            "*dload",
            f"elall, grav, {model.gravity:.12g}, 0.0, 0.0, -1.0",
            f"*node print, nset=nall, frequency={output_frequency}",
            "u",
            f"*node print, nset=nfloor, totals=only, global=yes, frequency={output_frequency}",
            "rf",
            f"*contact print, frequency={output_frequency}, totals=only",
            "cdis, cstr, cels, cnum",
            "*endstep",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_calculix(
    model: DropModel,
    out_dir: Path,
    *,
    case_name: str | None = None,
    timeout_seconds: int | None = None,
) -> tuple[Path, Row]:
    """Run CalculiX through WSL and return the generated .dat path."""

    if not calculix_available():
        raise SystemExit("CalculiX/ccx is not available in WSL.")

    case_name = case_name or f"{model.case}_r{model.resolution}"
    run_dir = out_dir / "calculix_runs" / case_name
    run_dir.mkdir(parents=True, exist_ok=True)
    inp_path = run_dir / f"{case_name}.inp"
    write_calculix_input(model, inp_path)

    wsl_run_dir = _wsl_path(run_dir)
    command = f"cd {wsl_run_dir} && ccx {case_name}"
    nominal_steps = max(1, int(np.ceil(model.total_time / model.dt)))
    timeout = int(timeout_seconds) if timeout_seconds is not None else max(240, min(3600, 120 + 2 * nominal_steps))
    timed_out = False
    try:
        proc = subprocess.run(
            ["wsl", "--exec", "bash", "-lc", command],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        return_code: int | str = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr)
        return_code = "timeout"
    (run_dir / "calculix_stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(stderr, encoding="utf-8")

    dat_path = run_dir / f"{case_name}.dat"
    if not dat_path.is_file():
        raise RuntimeError(f"CalculiX did not produce {dat_path}")

    command_row = {
        "case": model.case,
        "resolution": model.resolution,
        "external_solver": "CalculiX",
        "external_solver_version": _calculix_version(),
        "command": f"wsl --exec bash -lc \"cd {wsl_run_dir} && ccx {case_name}\"",
        "input_file": str(inp_path.relative_to(out_dir)),
        "dat_file": str(dat_path.relative_to(out_dir)),
        "stdout_log": str((run_dir / "calculix_stdout.log").relative_to(out_dir)),
        "stderr_log": str((run_dir / "calculix_stderr.log").relative_to(out_dir)),
        "return_code": return_code,
        "timed_out": str(timed_out).lower(),
        "completed": str(return_code == 0).lower(),
    }
    return dat_path, command_row


def _parse_calculix_dat_displacements(path: Path, node_ids: np.ndarray) -> dict[float, np.ndarray]:
    """Parse CalculiX node-print displacement blocks from a .dat file."""

    id_to_row = {int(node_id): i for i, node_id in enumerate(node_ids)}
    blocks: dict[float, np.ndarray] = {}
    current_time: float | None = None
    current = np.zeros((len(node_ids), 3), dtype=float)
    seen = 0
    header = re.compile(r"displacements.*time\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE)
    row_pattern = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    )

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = header.search(line)
        if match:
            if current_time is not None and seen:
                blocks[current_time] = current.copy()
            current_time = float(match.group(1))
            current = np.zeros((len(node_ids), 3), dtype=float)
            seen = 0
            continue

        if current_time is None:
            continue
        row = row_pattern.match(line)
        if row is None:
            continue
        node_id = int(row.group(1))
        if node_id not in id_to_row:
            continue
        current[id_to_row[node_id], :] = [float(row.group(2)), float(row.group(3)), float(row.group(4))]
        seen += 1

    if current_time is not None and seen:
        blocks[current_time] = current.copy()
    if not blocks:
        raise RuntimeError(f"no displacement blocks parsed from {path}")
    return dict(sorted(blocks.items()))


def _parse_calculix_dat_contact_totals(path: Path) -> dict[float, dict[str, float | int]]:
    """Parse CalculiX total RF/contact energy/contact count from .dat output."""

    totals: dict[float, dict[str, float | int]] = {}
    pending: tuple[str, float] | None = None
    number = r"([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    headers = {
        "floor_rf": re.compile(r"total force .* time\s+" + number, re.IGNORECASE),
        "contact_energy": re.compile(r"total contact spring energy for time\s+" + number, re.IGNORECASE),
        "contact_count": re.compile(r"total number of contact elements for time\s+" + number, re.IGNORECASE),
    }
    vector_pattern = re.compile(r"^\s*" + number + r"\s+" + number + r"\s+" + number + r"\s*$")
    scalar_pattern = re.compile(r"^\s*" + number + r"\s*$")
    integer_pattern = re.compile(r"^\s*(\d+)\s*$")

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        matched_header = False
        for kind, pattern in headers.items():
            match = pattern.search(line)
            if match:
                pending = (kind, float(match.group(1)))
                totals.setdefault(float(match.group(1)), {})
                matched_header = True
                break
        if matched_header or pending is None:
            continue

        kind, time = pending
        if kind == "floor_rf":
            match = vector_pattern.match(line)
            if match:
                fx, fy, fz = (float(match.group(i)) for i in range(1, 4))
                totals[time].update({"floor_rf_x": fx, "floor_rf_y": fy, "floor_rf_z": fz, "normal_force_from_rf": abs(fz)})
                pending = None
        elif kind == "contact_energy":
            match = scalar_pattern.match(line)
            if match:
                totals[time]["calculix_contact_energy"] = float(match.group(1))
                pending = None
        elif kind == "contact_count":
            match = integer_pattern.match(line)
            if match:
                totals[time]["calculix_contact_count"] = int(match.group(1))
                pending = None
    return dict(sorted(totals.items()))


def _history_from_displacements(
    source: str,
    model: DropModel,
    displacement_blocks: dict[float, np.ndarray],
    contact_totals: dict[float, dict[str, float | int]] | None = None,
) -> list[Row]:
    """Convert displacement blocks into comparable drop-impact diagnostics."""

    rows: list[Row] = []
    times = [0.0] + [float(t) for t in displacement_blocks]
    displacements = [np.zeros_like(model.nodes)] + [displacement_blocks[t] for t in displacement_blocks]
    previous_z_cm: float | None = None
    previous_time: float | None = None
    for time, u_nodes in zip(times, displacements, strict=True):
        current = model.nodes + u_nodes
        z_cm = _mass_weighted_center_z(model, current)
        if previous_z_cm is None:
            v_cm_z = model.initial_velocity_z
        else:
            v_cm_z = (z_cm - previous_z_cm) / max(float(time) - float(previous_time), 1.0e-30)
        gaps = _surface_face_gaps_to_contact_plane(model, current)
        penetration = np.maximum(-gaps, 0.0)
        active_mask = penetration > 0.0
        totals = {} if contact_totals is None else contact_totals.get(float(time), {})
        if source == "calculix":
            active_count_value = int(totals.get("calculix_contact_count", 0))
            normal_force: float | str = ""
            contact_energy: float | str = ""
            force_source = "calculix_contact_output_unavailable"
            if "normal_force_from_rf" in totals:
                normal_force = float(totals["normal_force_from_rf"])
                force_source = "floor_rf_total"
            if "calculix_contact_energy" in totals:
                contact_energy = float(totals["calculix_contact_energy"])
        else:
            active_count_value = int(np.count_nonzero(active_mask))
            normal_force = float(model.contact_stiffness * np.sum(penetration))
            contact_energy = float(0.5 * model.contact_stiffness * np.sum(penetration**2))
            force_source = "penalty_gap"
        rows.append(
            {
                "case": model.case,
                "resolution": model.resolution,
                "source": source,
                "time": float(time),
                "z_cm": z_cm,
                "v_cm_z": float(v_cm_z),
                "min_gap": float(np.min(gaps)),
                "max_penetration": float(np.max(penetration)),
                "active_contact_count": active_count_value,
                "normal_force_proxy": normal_force,
                "normal_force_source": force_source,
                "calculix_floor_rf_z": totals.get("floor_rf_z", ""),
                "calculix_contact_count": totals.get("calculix_contact_count", ""),
                "kinetic_energy_proxy": "",
                "strain_energy": "",
                "gravitational_potential_energy": "",
                "total_mechanical_energy_proxy": "",
                "contact_energy_proxy": contact_energy,
                "details": "postprocessed from CalculiX displacement, floor RF, and CONTACT PRINT output"
                if source == "calculix"
                else "computed by SFC Newmark linear TET4 prototype",
            }
        )
        previous_z_cm = z_cm
        previous_time = float(time)
    return rows


def _total_mass(model: DropModel) -> float:
    return float(model.density * sum(tet4_volume(model.nodes[element]) for element in model.tet_elements))


def _contact_plane_z(model: DropModel) -> float:
    """Return the CalculiX shell SPOS master contact plane."""

    return float(model.floor_z + CALCULIX_MASTER_SURFACE_OFFSET)


def _surface_face_geometry(current_nodes: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return boundary-face centroids and current triangle areas."""

    tri = current_nodes[faces]
    centroids = np.mean(tri, axis=1)
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return centroids, areas


def _surface_face_gaps_to_contact_plane(model: DropModel, current_nodes: np.ndarray) -> np.ndarray:
    """Return face-integration-point gaps to the CalculiX master contact plane."""

    centroids, _ = _surface_face_geometry(current_nodes, model.surface_faces)
    return centroids[:, 2] - _contact_plane_z(model)


def _reference_element_volumes(model: DropModel) -> np.ndarray:
    """Return positive reference volumes for all TET4 elements."""

    return np.asarray([tet4_volume(model.nodes[element]) for element in model.tet_elements], dtype=float)


def _mass_weighted_center_z(model: DropModel, current_nodes: np.ndarray) -> float:
    """Return the uniform-density TET4 mass center height."""

    volumes = _reference_element_volumes(model)
    element_center_z = np.mean(current_nodes[model.tet_elements, 2], axis=1)
    return float(np.dot(volumes, element_center_z) / np.sum(volumes))


def _mass_weighted_velocity_z(model: DropModel, nodal_velocity: np.ndarray) -> float:
    """Return the uniform-density TET4 mass-averaged vertical velocity."""

    volumes = _reference_element_volumes(model)
    element_velocity_z = np.mean(nodal_velocity.reshape((-1, 3))[model.tet_elements, 2], axis=1)
    return float(np.dot(volumes, element_velocity_z) / np.sum(volumes))


def _fill_acceleration_force_proxy(rows: list[Row], model: DropModel) -> None:
    """Infer a normal-force proxy from center-of-mass acceleration."""

    total_mass = _total_mass(model)
    by_source: dict[str, list[Row]] = {}
    for row in rows:
        by_source.setdefault(str(row["source"]), []).append(row)

    for source_rows in by_source.values():
        source_rows.sort(key=lambda row: float(row["time"]))
        for i, row in enumerate(source_rows):
            if i == 0:
                continue
            previous = source_rows[i - 1]
            dt = float(row["time"]) - float(previous["time"])
            if dt <= 0.0:
                continue
            acceleration = (float(row["v_cm_z"]) - float(previous["v_cm_z"])) / dt
            force_proxy = total_mass * (acceleration + model.gravity)
            if (
                str(row["source"]) == "calculix"
                and int(row["active_contact_count"]) > 0
                and row.get("normal_force_source") != "floor_rf_total"
            ):
                row["normal_force_proxy"] = float(max(force_proxy, 0.0))
                row["normal_force_source"] = "acceleration_proxy_fallback"


def _plane_contact_force(
    x_current: np.ndarray,
    surface_indices: np.ndarray,
    *,
    floor_z: float,
    stiffness: float,
) -> tuple[np.ndarray, float, int, float, float]:
    """Return nodal force for a frictionless rigid horizontal plane."""

    force = np.zeros(3 * x_current.shape[0], dtype=float)
    gaps = x_current[surface_indices, 2] - floor_z
    penetration = np.maximum(-gaps, 0.0)
    for node, depth in zip(surface_indices, penetration, strict=True):
        if depth > 0.0:
            force[3 * int(node) + 2] += stiffness * float(depth)
    return (
        force,
        float(np.min(gaps)),
        int(np.count_nonzero(penetration > 0.0)),
        float(np.max(penetration)),
        float(0.5 * stiffness * np.sum(penetration**2)),
    )


def _smooth_overclosure_response(d: np.ndarray, *, stiffness: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CalculiX-inspired smooth non-attractive linear pressure-overclosure law."""

    eps = max(float(epsilon), 1.0e-12)
    s = d / eps
    smooth_heaviside = 0.5 + np.arctan(s) / np.pi
    raw_lambda = stiffness * d * smooth_heaviside
    raw_tangent = stiffness * (smooth_heaviside + s / (np.pi * (1.0 + s * s)))
    active = raw_lambda > 0.0
    lambdas = np.where(active, raw_lambda, 0.0)
    tangents = np.where(active, np.maximum(raw_tangent, 0.0), 0.0)
    energy = 0.5 * np.where(active, stiffness * d * d * smooth_heaviside, 0.0)
    return lambdas, tangents, energy


def _calculix_aligned_plane_contact(
    model: DropModel,
    x_current: np.ndarray,
) -> tuple[np.ndarray, csr_matrix, float, int, float, float]:
    """Return face-based smooth penalty force and tangent for a rigid plane.

    CalculiX face-to-face contact evaluates clearance at slave-face integration
    points and scales the linear overclosure pressure by spring area.  This
    validation approximation uses three triangle quadrature points per boundary
    face and distributes each quadrature force back to the TET4 face nodes.
    """

    n_dofs = 3 * x_current.shape[0]
    faces = model.surface_faces
    face_nodes = x_current[faces]
    _, areas = _surface_face_geometry(x_current, faces)
    bary = TRIANGLE_QUADRATURE_BARYCENTRIC
    q_weights = TRIANGLE_QUADRATURE_WEIGHTS
    q_points = np.einsum("qn,fnc->fqc", bary, face_nodes)
    gaps = q_points[:, :, 2] - _contact_plane_z(model)
    overclosure = -gaps
    q_areas = areas[:, None] * q_weights[None, :]
    q_stiffness = model.contact_stiffness * q_areas
    lambdas, tangents, energies = _smooth_overclosure_response(
        overclosure,
        stiffness=q_stiffness,
        epsilon=model.contact_smoothing_epsilon,
    )

    force = np.zeros(n_dofs, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for face, face_lambdas, face_tangents in zip(faces, lambdas, tangents, strict=True):
        z_dofs = [3 * int(node) + 2 for node in face]
        for shape_weights, lam, tangent in zip(bary, face_lambdas, face_tangents, strict=True):
            for dof, weight in zip(z_dofs, shape_weights, strict=True):
                force[dof] += float(weight * lam)
            if tangent > 0.0:
                for row_dof, row_weight in zip(z_dofs, shape_weights, strict=True):
                    for col_dof, col_weight in zip(z_dofs, shape_weights, strict=True):
                        rows.append(row_dof)
                        cols.append(col_dof)
                        data.append(float(row_weight * col_weight * tangent))

    Kc = coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    physical_penetration = np.maximum(-gaps, 0.0)
    return (
        force,
        Kc,
        float(np.min(gaps)),
        int(np.count_nonzero(lambdas > 1.0e-12)),
        float(np.max(physical_penetration)),
        float(np.sum(energies)),
    )


def _hht_newmark_parameters(alpha: float) -> tuple[float, float]:
    """Return Hilber-Hughes-Taylor Newmark parameters used by CalculiX dynamics.

    CalculiX documents *DYNAMIC implicit integration as the HHT alpha method;
    alpha=0 gives classical Newmark, and the default keyword value is -0.05.
    The validation runner uses these parameters only for the CalculiX-aligned
    external comparison mode.
    """

    if not (-1.0 / 3.0 <= float(alpha) <= 0.0):
        raise ValueError("HHT alpha must lie in [-1/3, 0]")
    gamma = 0.5 - float(alpha)
    beta = 0.25 * (1.0 - float(alpha)) ** 2
    return beta, gamma


def run_sfc_drop_history(model: DropModel) -> list[Row]:
    """Run the matching SFC TET4 HHT-alpha model with SDF contact.

    Everything outside contact is intentionally aligned to the generated
    CalculiX model: C3D4/TET4 linear elasticity, consistent mass, zero physical
    damping, constant gravity, HHT alpha integration, fixed direct time
    increments, and Newton iterations on the implicit step.  Contact remains the
    SFC validation contact model, because that is the method under comparison.
    """

    mesh = VolumeMesh(
        model.nodes,
        model.tet_elements,
        node_sets={"surface": model.surface_indices},
    )
    body = DeformableBody(mesh, {"E": model.E, "nu": model.nu}, density=model.density)
    K = assemble_stiffness_matrix(body).tocsr()
    M = assemble_mass_matrix(body, kind="consistent").tocsr()
    C = csr_matrix(K.shape, dtype=float)
    f_gravity = assemble_gravity_force(body, (0.0, 0.0, -model.gravity))

    n_dofs = body.n_dofs
    u = np.zeros(n_dofs, dtype=float)
    v = np.zeros(n_dofs, dtype=float)
    v[2::3] = model.initial_velocity_z
    x_current = model.nodes + u.reshape((-1, 3))
    f_contact, _, _, _, _, _ = _calculix_aligned_plane_contact(model, x_current)
    a = np.asarray(spsolve(M, f_gravity + f_contact - K @ u - C @ v), dtype=float)

    alpha = float(model.hht_alpha)
    beta, gamma = _hht_newmark_parameters(alpha)
    output_dt = model.dt
    substeps = max(1, int(model.sfc_substeps))
    times = np.arange(0.0, model.total_time + 0.5 * output_dt, output_dt)
    rows: list[Row] = []

    for step, time in enumerate(times):
        x_current = model.nodes + u.reshape((-1, 3))
        f_contact, _, min_gap, active_count, max_penetration, contact_energy = _calculix_aligned_plane_contact(model, x_current)
        z_cm = _mass_weighted_center_z(model, x_current)
        kinetic_energy = float(0.5 * v @ (M @ v))
        strain_energy = float(0.5 * u @ (K @ u))
        gravitational_energy = float(_total_mass(model) * model.gravity * z_cm)
        total_energy = kinetic_energy + strain_energy + contact_energy + gravitational_energy
        rows.append(
            {
                "case": model.case,
                "resolution": model.resolution,
                "source": "sfc_calculix_aligned",
                "time": float(time),
                "z_cm": z_cm,
                "v_cm_z": _mass_weighted_velocity_z(model, v),
                "min_gap": min_gap,
                "max_penetration": max_penetration,
                "active_contact_count": active_count,
                "normal_force_proxy": float(np.sum(f_contact[2::3])),
                "normal_force_source": "smooth_penalty_tangent",
                "calculix_floor_rf_z": "",
                "calculix_contact_count": "",
                "kinetic_energy_proxy": kinetic_energy,
                "strain_energy": strain_energy,
                "gravitational_potential_energy": gravitational_energy,
                "contact_energy_proxy": contact_energy,
                "total_mechanical_energy_proxy": total_energy,
                "details": (
                    "CalculiX-aligned C3D4/TET4 linear dynamics with consistent mass, "
                    f"HHT alpha={alpha:g}, three-point triangle quadrature contact, area weighting, tangent contact, and {substeps} substeps"
                ),
            }
        )
        if step == len(times) - 1:
            break

        for _ in range(substeps):
            dt = output_dt / substeps
            c0 = 1.0 / (beta * dt * dt)
            c1 = gamma / (beta * dt)
            u_pred = u + dt * v + dt * dt * (0.5 - beta) * a
            v_pred = v + dt * (1.0 - gamma) * a
            x_prev = model.nodes + u.reshape((-1, 3))
            f_contact_prev, _, *_ = _calculix_aligned_plane_contact(model, x_prev)
            previous_static_residual = C @ v + K @ u - f_gravity - f_contact_prev
            u_guess = u_pred.copy()
            for _iteration in range(8):
                x_guess = model.nodes + u_guess.reshape((-1, 3))
                f_contact, K_contact, *_ = _calculix_aligned_plane_contact(model, x_guess)
                a_guess = c0 * (u_guess - u_pred)
                v_guess = v_pred + gamma * dt * a_guess
                current_static_residual = C @ v_guess + K @ u_guess - f_gravity - f_contact
                residual = M @ a_guess + (1.0 + alpha) * current_static_residual - alpha * previous_static_residual
                tangent = csc_matrix(c0 * M + (1.0 + alpha) * (K + c1 * C + K_contact))
                correction = np.asarray(spsolve(tangent, -residual), dtype=float)
                u_guess += correction
                if np.linalg.norm(correction) <= 1.0e-10 * max(1.0, np.linalg.norm(u_guess)):
                    break
            u_new = u_guess
            a_new = c0 * (u_new - u_pred)
            v_new = v_pred + gamma * dt * a_new
            u, v, a = u_new, v_new, a_new

    return rows


def _first_contact_time(rows: list[Row]) -> float | None:
    active: list[Row] = []
    for row in rows:
        if str(row.get("source", "")) == "calculix":
            force = _optional_float(row.get("normal_force_proxy", ""))
            if int(row["active_contact_count"]) > 0 or (force is not None and force > 1.0e-12):
                active.append(row)
        elif int(row["active_contact_count"]) > 0 or float(row["max_penetration"]) > 0.0:
            active.append(row)
    return None if not active else float(active[0]["time"])


def _optional_float(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rebound_physicality(model: DropModel, rows: list[Row]) -> Row:
    """Return a rebound-height physicality check.

    With an initial downward velocity, a rebound above the starting height can
    be physically admissible. The conservative ceiling used here is the
    starting mass-center height plus the ballistic height implied by the
    initial kinetic energy in the gravity field.
    """

    if not rows:
        return {
            "initial_z": "",
            "allowed_rebound_z": "",
            "first_contact_time": "",
            "max_rebound_z": "",
            "rebound_overshoot": "",
            "status": "not_available",
        }
    sorted_rows = sorted(rows, key=lambda row: float(row["time"]))
    initial_z = float(sorted_rows[0]["z_cm"])
    allowed_rebound_z = initial_z
    if float(model.gravity) > 0.0 and float(model.initial_velocity_z) < 0.0:
        allowed_rebound_z += float(model.initial_velocity_z) ** 2 / (2.0 * float(model.gravity))
    first_contact = _first_contact_time(sorted_rows)
    if first_contact is None:
        return {
            "initial_z": initial_z,
            "allowed_rebound_z": allowed_rebound_z,
            "first_contact_time": "",
            "max_rebound_z": "",
            "rebound_overshoot": "",
            "status": "not_available",
        }
    post_contact = [row for row in sorted_rows if float(row["time"]) >= float(first_contact)]
    max_rebound_z = max(float(row["z_cm"]) for row in post_contact) if post_contact else initial_z
    overshoot = max(0.0, max_rebound_z - allowed_rebound_z)
    return {
        "initial_z": initial_z,
        "allowed_rebound_z": allowed_rebound_z,
        "first_contact_time": first_contact,
        "max_rebound_z": max_rebound_z,
        "rebound_overshoot": overshoot,
        "status": "supported" if overshoot <= REBOUND_HEIGHT_TOLERANCE else "check",
    }


def comparison_metrics(
    model: DropModel,
    calculix_rows: list[Row],
    sfc_rows: list[Row],
    *,
    calculix_completed: bool = True,
) -> list[Row]:
    """Return summary metrics comparing CalculiX and SFC time histories."""

    cx_times_all = np.asarray([float(row["time"]) for row in calculix_rows], dtype=float)
    sfc_times_all = np.asarray([float(row["time"]) for row in sfc_rows], dtype=float)
    if cx_times_all.size and sfc_times_all.size:
        max_common_time = min(float(np.max(cx_times_all)), float(np.max(sfc_times_all)))
        min_common_time = max(float(np.min(cx_times_all)), float(np.min(sfc_times_all)))
        common_mask = (cx_times_all >= min_common_time) & (cx_times_all <= max_common_time)
        cx_times = cx_times_all[common_mask]
        cx_z = np.asarray([float(row["z_cm"]) for row, keep in zip(calculix_rows, common_mask, strict=True) if keep], dtype=float)
        cx_gap = np.asarray([float(row["min_gap"]) for row, keep in zip(calculix_rows, common_mask, strict=True) if keep], dtype=float)
        sfc_z = np.interp(cx_times, sfc_times_all, np.asarray([float(row["z_cm"]) for row in sfc_rows], dtype=float))
        sfc_gap = np.interp(cx_times, sfc_times_all, np.asarray([float(row["min_gap"]) for row in sfc_rows], dtype=float))
        z_errors = sfc_z - cx_z
        gap_errors = sfc_gap - cx_gap
    else:
        cx_times = np.asarray([], dtype=float)
        z_errors = np.asarray([], dtype=float)
        gap_errors = np.asarray([], dtype=float)

    cx_first = _first_contact_time(calculix_rows)
    sfc_first = _first_contact_time(sfc_rows)
    cx_rebound = _rebound_physicality(model, calculix_rows)
    sfc_rebound = _rebound_physicality(model, sfc_rows)
    first_contact_abs_error = "" if cx_first is None or sfc_first is None else abs(sfc_first - cx_first)
    both_contact = cx_first is not None and sfc_first is not None
    cx_contact_output_available = any(
        int(row["active_contact_count"]) > 0
        or ((_optional_float(row.get("normal_force_proxy", "")) or 0.0) > 1.0e-12)
        for row in calculix_rows
    )
    trajectory_rel = float(np.linalg.norm(z_errors) / max(np.linalg.norm(cx_z) if cx_times.size else 0.0, 1.0e-30))
    gap_linf = float(np.max(np.abs(gap_errors))) if gap_errors.size else float("nan")
    if not calculix_completed:
        contact_status = "incomplete"
    elif not cx_contact_output_available:
        contact_status = "not_available"
    elif both_contact and first_contact_abs_error != "" and float(first_contact_abs_error) <= 3.0e-3:
        contact_status = "supported"
    else:
        contact_status = "check"

    if not calculix_completed:
        physical_status = "incomplete"
    elif cx_rebound["status"] == "supported" and sfc_rebound["status"] == "supported":
        physical_status = "supported"
    elif cx_rebound["status"] == "not_available" or sfc_rebound["status"] == "not_available":
        physical_status = "not_available"
    else:
        physical_status = "check"

    if contact_status == "supported" and physical_status != "supported":
        status = physical_status
    else:
        status = contact_status

    return [
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "external_solver",
            "value": "CalculiX",
            "status": "evidence",
            "details": "external transient dynamic contact solver used through WSL ccx",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "calculix_run_completed",
            "value": str(bool(calculix_completed)).lower(),
            "status": "evidence" if calculix_completed else "incomplete",
            "details": "false means CalculiX returned a nonzero exit code; partial .dat data may still be parsed for diagnostics",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "first_contact_time_calculix",
            "value": "" if cx_first is None else cx_first,
            "status": "evidence",
            "details": "first time with exported CalculiX RF/contact count",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "first_contact_time_sfc",
            "value": "" if sfc_first is None else sfc_first,
            "status": "evidence",
            "details": "first time with positive SFC penalty contact",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "first_contact_time_abs_error",
            "value": first_contact_abs_error,
            "status": contact_status,
            "details": "gate uses 3 ms tolerance in quick generated model",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "z_cm_l2_relative_error",
            "value": trajectory_rel,
            "status": "evidence",
            "details": "center-of-mass trajectory difference after interpolating SFC onto CalculiX output times",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "min_gap_linf_abs_error",
            "value": gap_linf,
            "status": "evidence",
            "details": "minimum gap difference after interpolating SFC onto CalculiX output times",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "interpolated_comparison_sample_count",
            "value": int(cx_times.size),
            "status": "evidence",
            "details": "number of CalculiX output samples used for interpolated trajectory/gap metrics",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "initial_z_calculix",
            "value": cx_rebound["initial_z"],
            "status": "evidence",
            "details": "mass-weighted center height at the first exported CalculiX sample",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "allowed_rebound_z_calculix",
            "value": cx_rebound["allowed_rebound_z"],
            "status": "evidence",
            "details": "starting height plus ballistic height from initial downward kinetic energy",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "max_rebound_z_after_contact_calculix",
            "value": cx_rebound["max_rebound_z"],
            "status": cx_rebound["status"],
            "details": "maximum CalculiX mass-center height after first contact",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "rebound_overshoot_calculix",
            "value": cx_rebound["rebound_overshoot"],
            "status": cx_rebound["status"],
            "details": "positive value means CalculiX rebound exceeded the ballistic height bound",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "initial_z_sfc",
            "value": sfc_rebound["initial_z"],
            "status": "evidence",
            "details": "mass-weighted center height at the first SFC sample",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "allowed_rebound_z_sfc",
            "value": sfc_rebound["allowed_rebound_z"],
            "status": "evidence",
            "details": "starting height plus ballistic height from initial downward kinetic energy",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "max_rebound_z_after_contact_sfc",
            "value": sfc_rebound["max_rebound_z"],
            "status": sfc_rebound["status"],
            "details": "maximum SFC mass-center height after first contact",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "rebound_overshoot_sfc",
            "value": sfc_rebound["rebound_overshoot"],
            "status": sfc_rebound["status"],
            "details": "positive value means SFC rebound exceeded the ballistic height bound",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "physical_rebound_height_claim",
            "value": physical_status,
            "status": physical_status,
            "details": "supported only if both histories keep post-contact rebound height within the ballistic height bound",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "calculix_contact_output_available",
            "value": str(cx_contact_output_available).lower(),
            "status": "evidence" if cx_contact_output_available else "not_available",
            "details": "true only when CalculiX exports nonzero RF/contact count; otherwise force/energy comparison is skipped",
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "metric": "external_dynamic_contact_claim",
            "value": "supported" if status == "supported" else status,
            "status": "supported" if status == "supported" else status,
            "details": "supported only if contact timing agrees on a completed model and rebound height passes the physicality gate",
        },
    ]


def write_plots(out_dir: Path, history_rows: list[Row]) -> list[Row]:
    """Write paper-facing diagnostic plots for the external drop comparison."""

    out_dir.mkdir(parents=True, exist_ok=True)
    plots: list[Row] = []

    groups = sorted({(str(row["case"]), int(row["resolution"])) for row in history_rows})

    def series(rows_for_group: list[Row], source: str, field: str) -> tuple[list[float], list[float]]:
        rows = [row for row in rows_for_group if row["source"] == source]
        x: list[float] = []
        y: list[float] = []
        for row in rows:
            value = _optional_float(row.get(field, ""))
            if value is None:
                continue
            x.append(float(row["time"]))
            y.append(value)
        return x, y

    fields = [
        ("z_cm", "center-of-mass z", "calculix_drop_z_cm"),
        ("min_gap", "minimum gap to rigid plane", "calculix_drop_min_gap"),
        ("normal_force_proxy", "normal force proxy", "calculix_drop_force_proxy"),
        ("contact_energy_proxy", "contact energy proxy", "calculix_drop_contact_energy"),
        ("total_mechanical_energy_proxy", "SFC total mechanical energy proxy", "calculix_drop_total_energy"),
    ]
    for group_index, (case, resolution) in enumerate(groups):
        rows_for_group = [row for row in history_rows if row["case"] == case and int(row["resolution"]) == resolution]
        sources = sorted({str(row["source"]) for row in rows_for_group})
        for field, ylabel, base_name in fields:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            for source in sources:
                x, y = series(rows_for_group, source, field)
                if not x:
                    continue
                ax.plot(x, y, marker="o", markersize=3.0, linewidth=1.2, label=source)
            ax.axhline(0.0, color="0.35", linewidth=0.8)
            ax.set_xlabel("time")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{case} r{resolution}")
            ax.legend()
            ax.grid(True, color="0.9", linewidth=0.5)
            fig.tight_layout()
            name = base_name if group_index == 0 else f"{base_name}_{case}_r{resolution}"
            png = out_dir / f"{name}.png"
            pdf = out_dir / f"{name}.pdf"
            fig.savefig(png, dpi=180)
            fig.savefig(pdf)
            plt.close(fig)
            plots.append({"plot": name, "case": case, "resolution": resolution, "png": png.name, "pdf": pdf.name, "status": "ok"})

    return plots


def _fieldnames(rows: list[Row]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def write_markdown(
    path: Path,
    *,
    models: list[DropModel],
    command_rows: list[Row],
    metric_rows: list[Row],
    plots: list[Row],
    quick: bool,
) -> None:
    claim_rows = [row for row in metric_rows if row["metric"] == "external_dynamic_contact_claim"]
    all_supported = bool(claim_rows) and all(row["status"] == "supported" for row in claim_rows)
    physical_rows = [row for row in metric_rows if row["metric"] == "physical_rebound_height_claim"]
    all_physical = bool(physical_rows) and all(row["status"] == "supported" for row in physical_rows)
    first_command = command_rows[0] if command_rows else {"external_solver_version": "unknown", "command": ""}
    reproduce_parts = ["python", "validation/run_calculix_drop_impact_comparison.py"]
    if quick:
        reproduce_parts.append("--quick")
    if models and len({round(float(model.total_time), 12) for model in models}) == 1:
        reproduce_parts.extend(["--duration", f"{models[0].total_time:g}"])
    if models and len({round(float(model.dt), 12) for model in models}) == 1:
        reproduce_parts.extend(["--dt", f"{models[0].dt:g}"])
    if models and len({int(model.output_frequency) for model in models}) == 1 and models[0].output_frequency != 1:
        reproduce_parts.extend(["--output-frequency", str(models[0].output_frequency)])
    if models:
        case_names = sorted({model.case for model in models})
        if case_names != ["block_drop", "sphere_like_drop"]:
            for case_name in case_names:
                reproduce_parts.extend(["--case", case_name])
        resolution_values = sorted({int(model.resolution) for model in models})
        default_resolution_values = [1] if quick else [1, 2, 3]
        if resolution_values != default_resolution_values:
            for resolution in resolution_values:
                reproduce_parts.extend(["--resolution", str(resolution)])
    if models and len({round(float(model.initial_velocity_z), 12) for model in models}) == 1 and models[0].initial_velocity_z != -2.0:
        reproduce_parts.extend(["--initial-velocity-z", f"{models[0].initial_velocity_z:g}"])
    if models and len({round(float(model.gravity), 12) for model in models}) == 1 and models[0].gravity != 9.81:
        reproduce_parts.extend(["--gravity", f"{models[0].gravity:g}"])
    if models and len({round(float(model.contact_stiffness), 12) for model in models}) == 1:
        default_stiffness = 2.0e5 if models[0].case == "sphere_like_drop" else 2.0e4
        if models[0].contact_stiffness != default_stiffness:
            reproduce_parts.extend(["--contact-stiffness", f"{models[0].contact_stiffness:g}"])
    if models and len({round(float(model.hht_alpha), 12) for model in models}) == 1 and models[0].hht_alpha != CALCULIX_DEFAULT_HHT_ALPHA:
        reproduce_parts.extend(["--hht-alpha", f"{models[0].hht_alpha:g}"])
    if models and not models[0].direct_dynamic:
        reproduce_parts.append("--calculix-auto-step")
    reproduce_parts.extend(["--out-dir", path.parent.as_posix()])
    lines = [
        "# CalculiX Drop-Impact Comparison",
        "",
        "This validation uses CalculiX/ccx as an external open-source transient dynamic contact reference. The generated models are vertical elastic drop-impact cases against a fixed rigid plane. SFC uses the same TET4 meshes, material constants, gravity, initial velocity, HHT-alpha integration parameters, and rigid-plane location as each CalculiX input.",
        "",
        "## Reproduce",
        "",
        "```bash",
        " ".join(reproduce_parts),
        "```",
        "",
        "## External Solver",
        "",
        f"- Solver: CalculiX/ccx",
        f"- Version: `{first_command['external_solver_version']}`",
        f"- Example command: `{first_command['command']}`",
        "",
        "## Models",
        "",
        "| Case | Resolution | Nodes | TET4 elements | Boundary faces | HHT alpha | SFC substeps | Model source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in models:
        lines.append(
            f"| {model.case} | {model.resolution} | {model.nodes.shape[0]} | {model.tet_elements.shape[0]} | {model.surface_faces.shape[0]} | {model.hht_alpha:g} | {model.sfc_substeps} | {model.model_source} |"
        )
    lines.extend(
        [
            "",
            "Common setup:",
            "",
            f"- Initial vertical velocity: {_format_float(models[0].initial_velocity_z)}",
            f"- Rigid shell mid-surface z: {_format_float(models[0].floor_z)}",
            f"- CalculiX SPOS contact plane z: {_format_float(_contact_plane_z(models[0]))}",
            f"- Linear material: E={_format_float(models[0].E)}, nu={_format_float(models[0].nu)}, density={_format_float(models[0].density)}",
            f"- Nominal time increment: {_format_float(models[0].dt)}",
            f"- CalculiX dynamic mode: {'direct fixed increment' if models[0].direct_dynamic else 'adaptive increment'}",
            f"- CalculiX output frequency: {models[0].output_frequency}",
            "- CalculiX slave surfaces are face-based element surfaces; SFC uses matching boundary-face area weights.",
            "- SFC dynamics uses the same generated C3D4/TET4 mesh, isotropic linear elastic constants, consistent mass, gravity, initial velocity, and HHT-alpha parameters as the CalculiX input.",
            "- SFC contact uses a CalculiX-aligned three-point triangle quadrature pressure-overclosure law, area weighting, contact tangent, Newton iterations, and substepping.",
        ]
    )
    lines.extend(
        [
            "",
            "## Claim Gate",
            "",
            "| Claim | Status | Evidence | Gate |",
            "| --- | --- | --- | --- |",
            f"| external dynamic contact time-scale agreement | {'supported' if all_supported else 'not_supported'} | calculix_drop_metrics.csv::external_dynamic_contact_claim | {'all_supported' if all_supported else 'some_not_supported'} |",
            f"| rebound height within ballistic energy bound | {'supported' if all_physical else 'not_supported'} | calculix_drop_metrics.csv::physical_rebound_height_claim | {'all_supported' if all_physical else 'some_not_supported'} |",
            "<!-- evidence csv=calculix_drop_metrics.csv field=value -->",
            "",
            "## Metrics",
            "",
            "| Metric | Value | Status | Details |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in metric_rows:
        lines.append(f"| {row['metric']} | {row['value']} | {row['status']} | {row['details']} |")
    lines.extend(["", "## Plots", ""])
    for plot in plots:
        lines.append(f"- `{plot['png']}` and `{plot['pdf']}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- CalculiX performs an independent transient dynamic contact solve.",
            "- The CalculiX force history uses the fixed-floor RF total when available; SFC reports its assembled smooth penalty normal force.",
            "- The accepted claim is limited to the generated linear TET4 normal-impact cases. It is not a validation of friction, self-contact, nonlinear FEM, or production contact algorithms.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _calculix_dynamic_keyword(model: DropModel) -> str:
    mode = "DIRECT," if model.direct_dynamic else ""
    return f"*DYNAMIC,{mode}ALPHA={model.hht_alpha:g}"


def run_comparison(
    out_dir: Path,
    *,
    quick: bool,
    duration: float | None = None,
    dt: float | None = None,
    output_frequency: int = 1,
    direct_dynamic: bool = True,
    cases: list[str] | tuple[str, ...] | None = None,
    resolutions: list[int] | tuple[int, ...] | None = None,
    initial_velocity_z: float | None = None,
    gravity: float | None = None,
    contact_stiffness: float | None = None,
    calculix_timeout_seconds: int | None = None,
    hht_alpha: float | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("calculix_drop_*"):
        if stale.is_file():
            stale.unlink()
    run_dir = out_dir / "calculix_runs"
    if run_dir.is_dir():
        shutil.rmtree(run_dir)
    models = build_model_suite(
        quick=quick,
        duration=duration,
        dt=dt,
        output_frequency=output_frequency,
        direct_dynamic=direct_dynamic,
        cases=cases,
        resolutions=resolutions,
        initial_velocity_z=initial_velocity_z,
        gravity=gravity,
        contact_stiffness_override=contact_stiffness,
        hht_alpha=hht_alpha,
    )
    history_rows: list[Row] = []
    metric_rows: list[Row] = []
    command_rows: list[Row] = []
    for model in models:
        dat_path, command_row = run_calculix(model, out_dir, timeout_seconds=calculix_timeout_seconds)
        displacement_blocks = _parse_calculix_dat_displacements(dat_path, model.node_ids)
        contact_totals = _parse_calculix_dat_contact_totals(dat_path)
        calculix_rows = _history_from_displacements("calculix", model, displacement_blocks, contact_totals)
        sfc_rows = run_sfc_drop_history(model)
        history_rows.extend(calculix_rows)
        history_rows.extend(sfc_rows)
        command_rows.append(command_row)
        metric_rows.extend(
            comparison_metrics(
                model,
                calculix_rows,
                sfc_rows,
                calculix_completed=str(command_row.get("completed", "false")).lower() == "true",
            )
        )

    history_rows = sorted(history_rows, key=lambda row: (str(row["case"]), int(row["resolution"]), float(row["time"]), str(row["source"])))
    plots = write_plots(out_dir, history_rows)

    history_csv = out_dir / "calculix_drop_time_history.csv"
    metrics_csv = out_dir / "calculix_drop_metrics.csv"
    commands_csv = out_dir / "calculix_drop_commands.csv"
    plots_csv = out_dir / "calculix_drop_plots.csv"
    metadata_csv = out_dir / "calculix_drop_metadata.csv"
    summary_md = out_dir / "calculix_drop_summary.md"

    _write_csv(history_csv, _fieldnames(history_rows), history_rows)
    _write_csv(metrics_csv, _fieldnames(metric_rows), metric_rows)
    _write_csv(commands_csv, _fieldnames(command_rows), command_rows)
    _write_csv(plots_csv, _fieldnames(plots), plots)
    _write_csv(
        metadata_csv,
        ["key", "value"],
        [
            {"key": "python", "value": sys.version.split()[0]},
            {"key": "platform", "value": platform.platform()},
            {"key": "calculix_version", "value": command_rows[0]["external_solver_version"] if command_rows else "unknown"},
            {"key": "quick", "value": str(bool(quick)).lower()},
            {"key": "model_count", "value": len(models)},
            {"key": "hht_alpha", "value": models[0].hht_alpha if models else CALCULIX_DEFAULT_HHT_ALPHA},
            {"key": "sfc_mass_matrix", "value": "consistent"},
            {"key": "duration_override", "value": "" if duration is None else duration},
            {"key": "dt_override", "value": "" if dt is None else dt},
            {"key": "output_frequency", "value": output_frequency},
            {"key": "resolutions_override", "value": "" if resolutions is None else ",".join(str(int(value)) for value in resolutions)},
            {"key": "initial_velocity_z_override", "value": "" if initial_velocity_z is None else initial_velocity_z},
            {"key": "gravity_override", "value": "" if gravity is None else gravity},
            {"key": "contact_stiffness_override", "value": "" if contact_stiffness is None else contact_stiffness},
            {"key": "calculix_timeout_seconds", "value": "" if calculix_timeout_seconds is None else calculix_timeout_seconds},
            {"key": "hht_alpha_override", "value": "" if hht_alpha is None else hht_alpha},
            {"key": "calculix_direct_dynamic", "value": str(bool(direct_dynamic)).lower()},
            {"key": "calculix_dynamic_keyword", "value": _calculix_dynamic_keyword(models[0]) if models else "*DYNAMIC,DIRECT,ALPHA=-0.05"},
            {"key": "cases", "value": ",".join(f"{model.case}:r{model.resolution}" for model in models)},
        ],
    )
    write_markdown(summary_md, models=models, command_rows=command_rows, metric_rows=metric_rows, plots=plots, quick=quick)
    return {
        "history": history_csv,
        "metrics": metrics_csv,
        "commands": commands_csv,
        "plots": plots_csv,
        "metadata": metadata_csv,
        "summary": summary_md,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a short deterministic case for CI and smoke checks.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "calculix_drop_impact")
    parser.add_argument("--duration", type=float, default=None, help="Override total simulated time for diagnostic runs.")
    parser.add_argument("--dt", type=float, default=None, help="Override the nominal integration/output increment.")
    parser.add_argument("--output-frequency", type=int, default=1, help="CalculiX print frequency in increments.")
    parser.add_argument(
        "--case",
        action="append",
        choices=["sphere_like_drop", "block_drop"],
        default=None,
        help="Limit the run to one or more cases. Repeat the option for multiple cases.",
    )
    parser.add_argument("--resolution", action="append", type=int, default=None, help="Limit the run to one or more mesh resolutions.")
    parser.add_argument("--initial-velocity-z", type=float, default=None, help="Override the initial vertical velocity.")
    parser.add_argument("--gravity", type=float, default=None, help="Override gravitational acceleration magnitude.")
    parser.add_argument("--contact-stiffness", type=float, default=None, help="Override the pressure-overclosure stiffness.")
    parser.add_argument("--calculix-timeout-seconds", type=int, default=None, help="Override per-case CalculiX timeout.")
    parser.add_argument("--hht-alpha", type=float, default=None, help="Override HHT alpha; use 0 for classical Newmark.")
    parser.add_argument(
        "--calculix-auto-step",
        action="store_true",
        help="Use CalculiX adaptive *DYNAMIC instead of fixed direct increments for long diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_comparison(
        args.out_dir,
        quick=bool(args.quick),
        duration=args.duration,
        dt=args.dt,
        output_frequency=args.output_frequency,
        direct_dynamic=not bool(args.calculix_auto_step),
        cases=args.case,
        resolutions=args.resolution,
        initial_velocity_z=args.initial_velocity_z,
        gravity=args.gravity,
        contact_stiffness=args.contact_stiffness,
        calculix_timeout_seconds=args.calculix_timeout_seconds,
        hht_alpha=args.hht_alpha,
    )
    print("CalculiX drop-impact comparison complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
