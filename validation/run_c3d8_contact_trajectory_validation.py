"""C3D8 dynamic contact trajectory validation against CalculiX output.

This runner generates two small transient C3D8 contact cases:

* an elastic HEX8/C3D8 block contacting a fixed rigid plane;
* an elastic HEX8/C3D8 block contacting a fixed elastic C3D8 master block.

CalculiX is the external trajectory source.  SFC does not claim to reproduce
CalculiX's full nonlinear dynamic solve here.  Instead, the CalculiX-deformed
coordinates are replayed with the SFC current-surface dynamic SDF query on the
same C3D8 boundary faces, and the resulting gap, contact force, contact energy,
and stress-cloud quantities are compared against CalculiX outputs where those
outputs are available.
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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

from sfc.fem.hex8 import hex8_center_strain_stress, hex8_volume  # noqa: E402
from sfc.mesh import extract_boundary_triangles  # noqa: E402
from sfc.contact.field_contact import field_contact_constraint_from_sample  # noqa: E402
from sfc.contact.narrow_phase import SurfaceSample  # noqa: E402
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402

Row = dict[str, Any]
_TRAJECTORY_FIELD_CACHE: dict[tuple[tuple[int, ...], tuple[int, int], bytes], DynamicNarrowBandSDF] = {}

FLOOR_SHELL_THICKNESS = 1.0e-2
FLOOR_CONTACT_OFFSET = 0.5 * FLOOR_SHELL_THICKNESS

C3D8_FACE_NODES = {
    "S1": np.array([0, 1, 2, 3], dtype=np.int64),
    "S2": np.array([4, 7, 6, 5], dtype=np.int64),
    "S3": np.array([0, 4, 5, 1], dtype=np.int64),
    "S4": np.array([1, 5, 6, 2], dtype=np.int64),
    "S5": np.array([2, 6, 7, 3], dtype=np.int64),
    "S6": np.array([3, 7, 4, 0], dtype=np.int64),
}

QUAD_GAUSS = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))


@dataclass(slots=True)
class HexBlock:
    node_ids: np.ndarray
    X: np.ndarray
    elements: np.ndarray
    element_ids: np.ndarray
    face_refs: dict[str, list[tuple[int, str]]] = field(default_factory=dict)


@dataclass(slots=True)
class C3D8TrajectoryModel:
    case: str
    resolution: int
    node_ids: np.ndarray
    X: np.ndarray
    elements: np.ndarray
    element_ids: np.ndarray
    slave_surface: list[tuple[int, str]]
    master_surface: list[tuple[int, str]]
    slave_node_indices: np.ndarray
    fixed_node_ids: np.ndarray
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
    master_kind: str

    @property
    def id_to_index(self) -> dict[int, int]:
        return {int(node_id): idx for idx, node_id in enumerate(self.node_ids)}


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
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", "command -v ccx >/dev/null"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return proc.returncode == 0


def _calculix_version() -> str:
    if not calculix_available():
        return "unavailable"
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", "ccx -v 2>&1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    lines = [line.strip() for line in (proc.stdout + proc.stderr).splitlines() if line.strip()]
    return lines[0] if lines else "CalculiX"


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        return resolved.as_posix()
    rest = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{rest}"


def _write_hex_block_input_lines(lines: list[str], model: C3D8TrajectoryModel) -> None:
    lines.append("*NODE, NSET=NALL")
    for node_id, xyz in zip(model.node_ids, model.X, strict=True):
        lines.append(f"{int(node_id)}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    if model.master_kind == "rigid_plane":
        lines.append("*NODE, NSET=NFLOOR")
        for row in model.floor_nodes:
            node_id = int(row[0])
            x, y, z = row[1:]
            lines.append(f"{node_id}, {x:.12e}, {y:.12e}, {z:.12e}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=ELALL")
    for element_id, element in zip(model.element_ids, model.elements, strict=True):
        conn = ", ".join(str(int(model.node_ids[idx])) for idx in element)
        lines.append(f"{int(element_id)}, {conn}")
    if model.master_kind == "rigid_plane":
        lines.append("*ELEMENT, TYPE=S4, ELSET=FLOOR")
        for floor_element in model.floor_element_ids:
            lines.append(", ".join(str(int(value)) for value in floor_element))


def _append_id_list(lines: list[str], values: np.ndarray, *, per_line: int = 12) -> None:
    ids = [str(int(value)) for value in np.asarray(values).ravel()]
    for start in range(0, len(ids), per_line):
        lines.append(", ".join(ids[start : start + per_line]))


def write_calculix_input(model: C3D8TrajectoryModel, path: Path) -> None:
    """Write the C3D8 dynamic contact case as a CalculiX input file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "** Generated by validation/run_c3d8_contact_trajectory_validation.py",
        "*HEADING",
        f"C3D8 dynamic contact trajectory validation: {model.case}",
    ]
    _write_hex_block_input_lines(lines, model)
    lines.extend(
        [
            "*NSET, NSET=NSLAVE",
        ]
    )
    _append_id_list(lines, model.node_ids[model.slave_node_indices])
    lines.extend(
        [
            "*NSET, NSET=NFIXED",
        ]
    )
    _append_id_list(lines, model.fixed_node_ids)
    lines.extend(
        [
            "*MATERIAL, NAME=MAT",
            "*ELASTIC",
            f"{model.E:.12e}, {model.nu:.12e}",
            "*DENSITY",
            f"{model.density:.12e}",
            "*SOLID SECTION, ELSET=ELALL, MATERIAL=MAT",
        ]
    )
    if model.master_kind == "rigid_plane":
        lines.extend(
            [
                "*MATERIAL, NAME=FLOORMAT",
                "*ELASTIC",
                f"{model.E:.12e}, {model.nu:.12e}",
                "*SHELL SECTION, ELSET=FLOOR, MATERIAL=FLOORMAT",
                f"{FLOOR_SHELL_THICKNESS:.12e}",
            ]
        )

    lines.append("*SURFACE, NAME=SSLAVE, TYPE=ELEMENT")
    for element_index, face in model.slave_surface:
        lines.append(f"{int(model.element_ids[element_index])}, {face}")
    lines.append("*SURFACE, NAME=SMASTER, TYPE=ELEMENT")
    if model.master_kind == "rigid_plane":
        lines.append("FLOOR, SPOS")
    else:
        for element_index, face in model.master_surface:
            lines.append(f"{int(model.element_ids[element_index])}, {face}")
    lines.extend(
        [
            "*SURFACE INTERACTION, NAME=CONTACT",
            "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
            f"{model.contact_stiffness:.12e}",
            "*CONTACT PAIR, INTERACTION=CONTACT, TYPE=SURFACE TO SURFACE",
            "SSLAVE, SMASTER",
            "*INITIAL CONDITIONS, TYPE=VELOCITY",
            f"NSLAVE, 3, {model.initial_velocity_z:.12e}",
            "*BOUNDARY",
            "NFIXED, 1, 3, 0.0",
            "*STEP, NLGEOM",
            "*DYNAMIC",
            f"{model.dt:.12e}, {model.total_time:.12e}, {0.01 * model.dt:.12e}, {model.dt:.12e}",
            "*DLOAD",
            f"ELALL, GRAV, {model.gravity:.12e}, 0.0, 0.0, -1.0",
            f"*NODE PRINT, NSET=NALL, FREQUENCY={model.output_frequency}",
            "U",
            f"*NODE PRINT, NSET=NALL, FREQUENCY={model.output_frequency}",
            "V",
            f"*NODE PRINT, NSET=NFIXED, TOTALS=ONLY, GLOBAL=YES, FREQUENCY={model.output_frequency}",
            "RF",
            f"*CONTACT PRINT, FREQUENCY={model.output_frequency}",
            "CDIS,CSTR,CELS",
            f"*CONTACT PRINT, TOTALS=ONLY, FREQUENCY={model.output_frequency}",
            "CELS,CNUM",
            f"*EL PRINT, ELSET=ELALL, FREQUENCY={model.output_frequency}",
            "S,E",
            "*END STEP",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _structured_hex_block(
    resolution: int,
    *,
    origin: tuple[float, float, float],
    size: tuple[float, float, float],
    node_start: int,
    element_start: int,
) -> HexBlock:
    n = max(1, int(resolution))
    lx, ly, lz = size
    ox, oy, oz = origin
    nodes: list[tuple[float, float, float]] = []
    node_id: dict[tuple[int, int, int], int] = {}
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                node_id[(i, j, k)] = len(nodes)
                nodes.append((ox + lx * i / n, oy + ly * j / n, oz + lz * k / n))
    elements: list[tuple[int, ...]] = []
    element_ids: list[int] = []
    face_refs: dict[str, list[tuple[int, str]]] = {label: [] for label in C3D8_FACE_NODES}
    for k in range(n):
        for j in range(n):
            for i in range(n):
                local_index = len(elements)
                element_ids.append(element_start + local_index)
                conn = (
                    node_id[(i, j, k)],
                    node_id[(i + 1, j, k)],
                    node_id[(i + 1, j + 1, k)],
                    node_id[(i, j + 1, k)],
                    node_id[(i, j, k + 1)],
                    node_id[(i + 1, j, k + 1)],
                    node_id[(i + 1, j + 1, k + 1)],
                    node_id[(i, j + 1, k + 1)],
                )
                elements.append(conn)
                if k == 0:
                    face_refs["S1"].append((local_index, "S1"))
                if k == n - 1:
                    face_refs["S2"].append((local_index, "S2"))
                if j == 0:
                    face_refs["S3"].append((local_index, "S3"))
                if i == n - 1:
                    face_refs["S4"].append((local_index, "S4"))
                if j == n - 1:
                    face_refs["S5"].append((local_index, "S5"))
                if i == 0:
                    face_refs["S6"].append((local_index, "S6"))
    return HexBlock(
        node_ids=np.arange(node_start, node_start + len(nodes), dtype=np.int64),
        X=np.asarray(nodes, dtype=float),
        elements=np.asarray(elements, dtype=np.int64),
        element_ids=np.asarray(element_ids, dtype=np.int64),
        face_refs=face_refs,
    )


def _floor(half_width: float, z: float, *, start_id: int, element_id: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(
        [
            [start_id, -half_width, -half_width, z],
            [start_id + 1, half_width, -half_width, z],
            [start_id + 2, half_width, half_width, z],
            [start_id + 3, -half_width, half_width, z],
        ],
        dtype=float,
    )
    elements = np.asarray([[element_id, start_id, start_id + 1, start_id + 2, start_id + 3]], dtype=np.int64)
    return nodes, elements


def _offset_face_refs(refs: list[tuple[int, str]], element_offset: int) -> list[tuple[int, str]]:
    return [(element_offset + int(element_index), face) for element_index, face in refs]


def build_c3d8_model(
    *,
    case: str,
    resolution: int = 1,
    quick: bool = False,
    total_time: float | None = None,
    dt: float | None = None,
) -> C3D8TrajectoryModel:
    """Build a deterministic C3D8 dynamic contact validation model."""

    res = max(1, int(resolution))
    duration = float(total_time if total_time is not None else (0.08 if quick else 0.12))
    step = float(dt if dt is not None else (0.002 if quick else 0.001))
    output_frequency = 1 if quick else 2
    E = 1.0e5
    nu = 0.30
    density = 1.0
    gravity = 9.81
    initial_velocity_z = -0.30
    contact_stiffness = 5.0e4

    if case == "block_plane_c3d8":
        block = _structured_hex_block(
            res,
            origin=(-0.25, -0.25, 0.020),
            size=(0.50, 0.50, 0.30),
            node_start=1,
            element_start=1,
        )
        floor_nodes, floor_elements = _floor(
            0.75,
            0.0,
            start_id=int(block.node_ids[-1]) + 1,
            element_id=int(block.element_ids[-1]) + 1,
        )
        return C3D8TrajectoryModel(
            case=case,
            resolution=res,
            node_ids=block.node_ids,
            X=block.X,
            elements=block.elements,
            element_ids=block.element_ids,
            slave_surface=block.face_refs["S1"],
            master_surface=[],
            slave_node_indices=np.arange(block.X.shape[0], dtype=np.int64),
            fixed_node_ids=floor_nodes[:, 0].astype(np.int64),
            floor_nodes=floor_nodes,
            floor_element_ids=floor_elements,
            floor_z=0.0,
            E=E,
            nu=nu,
            density=density,
            gravity=gravity,
            initial_velocity_z=initial_velocity_z,
            contact_stiffness=contact_stiffness,
            total_time=duration,
            dt=step,
            output_frequency=output_frequency,
            master_kind="rigid_plane",
        )
    if case == "block_block_c3d8":
        master = _structured_hex_block(
            res,
            origin=(-0.25, -0.25, 0.0),
            size=(0.50, 0.50, 0.24),
            node_start=1,
            element_start=1,
        )
        slave = _structured_hex_block(
            res,
            origin=(-0.25, -0.25, 0.255),
            size=(0.50, 0.50, 0.26),
            node_start=int(master.node_ids[-1]) + 1,
            element_start=int(master.element_ids[-1]) + 1,
        )
        element_offset = master.elements.shape[0]
        X = np.vstack([master.X, slave.X])
        elements = np.vstack([master.elements, slave.elements + master.X.shape[0]])
        node_ids = np.concatenate([master.node_ids, slave.node_ids])
        element_ids = np.concatenate([master.element_ids, slave.element_ids])
        slave_indices = np.arange(master.X.shape[0], X.shape[0], dtype=np.int64)
        fixed_node_ids = master.node_ids.copy()
        return C3D8TrajectoryModel(
            case=case,
            resolution=res,
            node_ids=node_ids,
            X=X,
            elements=elements,
            element_ids=element_ids,
            slave_surface=_offset_face_refs(slave.face_refs["S1"], element_offset),
            master_surface=master.face_refs["S2"],
            slave_node_indices=slave_indices,
            fixed_node_ids=fixed_node_ids,
            floor_nodes=np.empty((0, 4), dtype=float),
            floor_element_ids=np.empty((0, 5), dtype=np.int64),
            floor_z=0.0,
            E=E,
            nu=nu,
            density=density,
            gravity=gravity,
            initial_velocity_z=initial_velocity_z,
            contact_stiffness=contact_stiffness,
            total_time=duration,
            dt=step,
            output_frequency=output_frequency,
            master_kind="deformable_block",
        )
    raise ValueError("case must be 'block_plane_c3d8' or 'block_block_c3d8'")


def build_model_suite(*, quick: bool, cases: list[str] | None = None, resolutions: list[int] | None = None) -> list[C3D8TrajectoryModel]:
    selected_cases = cases or ["block_plane_c3d8", "block_block_c3d8"]
    selected_resolutions = resolutions or ([1] if quick else [1, 2])
    return [
        build_c3d8_model(case=case, resolution=resolution, quick=quick)
        for case in selected_cases
        for resolution in selected_resolutions
    ]


def run_calculix(model: C3D8TrajectoryModel, out_dir: Path, *, timeout_seconds: int = 240) -> tuple[Path, Row]:
    """Run CalculiX for one generated C3D8 case."""

    if not calculix_available():
        raise RuntimeError("CalculiX/ccx is unavailable through WSL")
    case_name = f"{model.case}_r{model.resolution}"
    run_dir = out_dir / "calculix_runs" / case_name
    run_dir.mkdir(parents=True, exist_ok=True)
    inp_path = run_dir / f"{case_name}.inp"
    write_calculix_input(model, inp_path)
    wsl_run_dir = _wsl_path(run_dir)
    command = f"cd {wsl_run_dir} && ccx {case_name}"
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            ["wsl", "--exec", "bash", "-lc", command],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return_code: int | str = proc.returncode
        timed_out = False
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        return_code = "timeout"
        timed_out = True
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    wall_time = time.perf_counter() - start
    (run_dir / "calculix_stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(stderr, encoding="utf-8")
    dat_path = run_dir / f"{case_name}.dat"
    if return_code != 0 or not dat_path.exists() or dat_path.stat().st_size == 0:
        raise RuntimeError(f"CalculiX did not produce {dat_path}")
    return dat_path, {
        "case": model.case,
        "resolution": model.resolution,
        "external_solver": "CalculiX",
        "external_solver_version": _calculix_version(),
        "command": f"wsl --exec bash -lc \"{command}\"",
        "input_file": str(inp_path.relative_to(out_dir)),
        "dat_file": str(dat_path.relative_to(out_dir)),
        "stdout_log": str((run_dir / "calculix_stdout.log").relative_to(out_dir)),
        "stderr_log": str((run_dir / "calculix_stderr.log").relative_to(out_dir)),
        "return_code": return_code,
        "timed_out": str(timed_out).lower(),
        "completed": str(return_code == 0).lower(),
        "wall_time_seconds": wall_time,
    }


def _parse_nodal_vectors(path: Path, node_ids: np.ndarray, *, quantity: str) -> dict[float, np.ndarray]:
    labels = {
        "u": r"displacements",
        "v": r"velocities",
        "a": r"accelerations",
    }
    if quantity not in labels:
        raise ValueError("quantity must be 'u', 'v', or 'a'")
    id_to_row = {int(node_id): idx for idx, node_id in enumerate(node_ids)}
    blocks: dict[float, np.ndarray] = {}
    header = re.compile(labels[quantity] + r".*time\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE)
    row = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    )
    current_time: float | None = None
    current = np.zeros((len(node_ids), 3), dtype=float)
    seen: set[int] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = header.search(line)
        if match:
            if current_time is not None and seen:
                blocks[current_time] = current.copy()
            current_time = float(match.group(1))
            current = np.zeros((len(node_ids), 3), dtype=float)
            seen = set()
            continue
        if current_time is None:
            continue
        match = row.match(line)
        if match is None:
            continue
        node_id = int(match.group(1))
        if node_id in id_to_row and node_id not in seen:
            current[id_to_row[node_id]] = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
            seen.add(node_id)
    if current_time is not None and seen:
        blocks[current_time] = current.copy()
    return dict(sorted(blocks.items()))


def _parse_totals(path: Path) -> dict[float, Row]:
    totals: dict[float, Row] = {}
    pending: tuple[str, float] | None = None
    number = r"([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    headers = {
        "rf": re.compile(r"total force .* time\s+" + number, re.IGNORECASE),
        "cels": re.compile(r"total contact spring energy for time\s+" + number, re.IGNORECASE),
        "cnum": re.compile(r"total number of contact elements for time\s+" + number, re.IGNORECASE),
    }
    vector = re.compile(r"^\s*" + number + r"\s+" + number + r"\s+" + number + r"\s*$")
    scalar = re.compile(r"^\s*" + number + r"\s*$")
    integer = re.compile(r"^\s*(\d+)\s*$")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for kind, pattern in headers.items():
            match = pattern.search(line)
            if match:
                pending = (kind, float(match.group(1)))
                totals.setdefault(float(match.group(1)), {})
                break
        else:
            if pending is None:
                continue
            kind, time = pending
            if kind == "rf":
                match = vector.match(line)
                if match:
                    fx, fy, fz = (float(match.group(i)) for i in range(1, 4))
                    totals[time].update({"rf_x": fx, "rf_y": fy, "rf_z": fz, "normal_force_calculix": abs(fz)})
                    pending = None
            elif kind == "cels":
                match = scalar.match(line)
                if match:
                    totals[time]["contact_energy_calculix"] = float(match.group(1))
                    pending = None
            else:
                match = integer.match(line)
                if match:
                    totals[time]["contact_count_calculix"] = int(match.group(1))
                    pending = None
    return dict(sorted(totals.items()))


def _parse_contact_elements(path: Path) -> dict[float, list[Row]]:
    number = r"([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    headers = {
        "cdis": re.compile(r"relative contact displacement .* time\s+" + number, re.IGNORECASE),
        "cstr": re.compile(r"contact stress .* time\s+" + number, re.IGNORECASE),
        "cels": re.compile(r"contact (?:spring|print) energy .* time\s+" + number, re.IGNORECASE),
    }
    vector = re.compile(r"^\s*(\d+)\s+(\d+)\s+" + number + r"\s+" + number + r"\s+" + number + r"\s*$")
    scalar = re.compile(r"^\s*(\d+)\s+(\d+)\s+" + number + r"\s*$")
    rows_by_time: dict[float, list[Row]] = {}
    current_kind: str | None = None
    current_time: float | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        header_found = False
        for kind, pattern in headers.items():
            match = pattern.search(line)
            if match:
                current_kind = kind
                current_time = float(match.group(1))
                rows_by_time.setdefault(current_time, [])
                header_found = True
                break
        if header_found or current_kind is None or current_time is None:
            continue
        if current_kind in {"cdis", "cstr"}:
            match = vector.match(line)
            if match is None:
                if re.search(r"[A-Za-z]", line):
                    current_kind = None
                    current_time = None
                continue
            key = "clearance" if current_kind == "cdis" else "pressure"
            rows_by_time[current_time].append(
                {
                    "quantity": current_kind,
                    "slave_element": int(match.group(1)),
                    "slave_face": int(match.group(2)),
                    key: float(match.group(3)),
                    "tangential_1": float(match.group(4)),
                    "tangential_2": float(match.group(5)),
                }
            )
        else:
            match = scalar.match(line)
            if match is None:
                if re.search(r"[A-Za-z]", line):
                    current_kind = None
                    current_time = None
                continue
            rows_by_time[current_time].append(
                {
                    "quantity": current_kind,
                    "slave_element": int(match.group(1)),
                    "slave_face": int(match.group(2)),
                    "energy": float(match.group(3)),
                }
            )
    return dict(sorted(rows_by_time.items()))


def _parse_element_stress(path: Path) -> dict[float, dict[int, np.ndarray]]:
    number = r"([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    header = re.compile(r"stresses .* time\s+" + number, re.IGNORECASE)
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
        # CalculiX prints sxx,syy,szz,sxy,sxz,syz. Convert to the SFC ordering
        # sxx,syy,szz,sxy,syz,sxz before von Mises evaluation.
        sxx, syy, szz, sxy, sxz, syz = [float(match.group(i)) for i in range(3, 9)]
        values[current_time].setdefault(element_id, []).append(np.asarray([sxx, syy, szz, sxy, syz, sxz], dtype=float))
    return {time: {eid: np.mean(stresses, axis=0) for eid, stresses in element_map.items()} for time, element_map in values.items()}


def _quad_shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = np.asarray(
        [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )
    dxi = np.asarray(
        [
            -0.25 * (1.0 - eta),
            0.25 * (1.0 - eta),
            0.25 * (1.0 + eta),
            -0.25 * (1.0 + eta),
        ],
        dtype=float,
    )
    deta = np.asarray(
        [
            -0.25 * (1.0 - xi),
            -0.25 * (1.0 + xi),
            0.25 * (1.0 + xi),
            0.25 * (1.0 - xi),
        ],
        dtype=float,
    )
    return shape, dxi, deta


def _surface_quads(model: C3D8TrajectoryModel, refs: list[tuple[int, str]]) -> list[tuple[np.ndarray, np.ndarray]]:
    quads: list[tuple[np.ndarray, np.ndarray]] = []
    for element_index, face in refs:
        element = model.elements[int(element_index)]
        quads.append((element[C3D8_FACE_NODES[face]], element))
    return quads


def _orient_quad_outward(X: np.ndarray, quad: np.ndarray, element: np.ndarray) -> np.ndarray:
    coords = X[quad]
    normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
    if float(np.dot(normal, coords.mean(axis=0) - X[element].mean(axis=0))) < 0.0:
        return quad[::-1]
    return quad


def _triangles_from_quads(X: np.ndarray, quads: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    triangles: list[np.ndarray] = []
    for quad, element in quads:
        oriented = _orient_quad_outward(X, quad, element)
        triangles.append(oriented[[0, 1, 2]])
        triangles.append(oriented[[0, 2, 3]])
    if not triangles:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)


def _master_query_geometry(model: C3D8TrajectoryModel, X_current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if model.master_kind == "rigid_plane":
        floor = np.asarray(
            [
                [-0.75, -0.75, model.floor_z + FLOOR_CONTACT_OFFSET],
                [0.75, -0.75, model.floor_z + FLOOR_CONTACT_OFFSET],
                [0.75, 0.75, model.floor_z + FLOOR_CONTACT_OFFSET],
                [-0.75, 0.75, model.floor_z + FLOOR_CONTACT_OFFSET],
            ],
            dtype=float,
        )
        start = X_current.shape[0]
        nodes = np.vstack([X_current, floor])
        faces = np.asarray([[start, start + 1, start + 2], [start, start + 2, start + 3]], dtype=np.int64)
        return nodes, faces
    faces = _triangles_from_quads(X_current, _surface_quads(model, model.master_surface))
    return X_current, faces


def _trajectory_sdf_grid_parameters(nodes: np.ndarray, faces: np.ndarray) -> tuple[float, float]:
    if faces.size == 0:
        raise ValueError("master field requires at least one triangle")
    used = np.unique(np.asarray(faces, dtype=np.int64).ravel())
    X = np.asarray(nodes, dtype=float)[used]
    extent = np.ptp(X, axis=0)
    diag = max(float(np.linalg.norm(extent)), 1.0e-6)
    spacing = max(diag / 28.0, 0.02)
    band_radius = max(6.0 * spacing, 0.35 * diag, 0.12)
    return spacing, band_radius


def _trajectory_field_cache_key(nodes: np.ndarray, faces: np.ndarray) -> tuple[tuple[int, ...], tuple[int, int], bytes]:
    face_ids = np.asarray(faces, dtype=np.int64)
    used = np.unique(face_ids.ravel())
    coords = np.round(np.asarray(nodes, dtype=float)[used], decimals=12)
    return tuple(int(v) for v in face_ids.ravel()), tuple(coords.shape), coords.tobytes()


def _build_trajectory_master_field(nodes: np.ndarray, faces: np.ndarray) -> DynamicNarrowBandSDF:
    key = _trajectory_field_cache_key(nodes, faces)
    cached = _TRAJECTORY_FIELD_CACHE.get(key)
    if cached is not None:
        return cached
    spacing, band_radius = _trajectory_sdf_grid_parameters(nodes, faces)
    field = DynamicNarrowBandSDF.build(
        nodes,
        faces,
        spacing=spacing,
        band_radius=band_radius,
        padding=band_radius,
        cell_size=max(2.0 * spacing, band_radius),
        gradient_mode="finite_difference",
    )
    _TRAJECTORY_FIELD_CACHE[key] = field
    return field


def _von_mises(stress: np.ndarray) -> float:
    sxx, syy, szz, sxy, syz, sxz = np.asarray(stress, dtype=float)
    return float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + syz**2 + sxz**2)))


def _replay_dynamic_sdf(model: C3D8TrajectoryModel, U: np.ndarray) -> Row:
    X_current = model.X + U
    query_nodes, master_faces = _master_query_geometry(model, X_current)
    master_sdf = _build_trajectory_master_field(query_nodes, master_faces)
    candidates = np.arange(master_faces.shape[0], dtype=np.int64)
    min_gap = np.inf
    max_penetration = 0.0
    active_count = 0
    force = np.zeros(3, dtype=float)
    contact_energy = 0.0
    slave_quads = _surface_quads(model, model.slave_surface)
    for quad, element in slave_quads:
        face = _orient_quad_outward(X_current, quad, element)
        coords = X_current[face]
        for xi in QUAD_GAUSS:
            for eta in QUAD_GAUSS:
                shape, dxi, deta = _quad_shape(xi, eta)
                xq = shape @ coords
                jac = np.cross(dxi @ coords, deta @ coords)
                area_weight = float(np.linalg.norm(jac))
                sample = SurfaceSample(
                    node_ids=np.asarray(face, dtype=np.int64),
                    weights=np.asarray(shape, dtype=float),
                    candidate_face_ids=candidates,
                )
                constraint = field_contact_constraint_from_sample(X_current, sample, master_sdf)
                gap = float(constraint.g)
                min_gap = min(min_gap, gap)
                overclosure = max(-gap, 0.0)
                max_penetration = max(max_penetration, overclosure)
                if overclosure <= 0.0:
                    continue
                active_count += 1
                scalar_force = model.contact_stiffness * overclosure * area_weight
                force += scalar_force * constraint.normal
                contact_energy += 0.5 * model.contact_stiffness * overclosure * overclosure * area_weight
    if not np.isfinite(min_gap):
        min_gap = 0.0
    return {
        "sfc_min_gap": float(min_gap),
        "sfc_max_penetration": float(max_penetration),
        "sfc_active_contact_count": int(active_count),
        "sfc_normal_force_x": float(force[0]),
        "sfc_normal_force_y": float(force[1]),
        "sfc_normal_force_z": float(force[2]),
        "sfc_normal_force_norm": float(np.linalg.norm(force)),
        "sfc_contact_energy": float(contact_energy),
        "sfc_master_triangle_count": int(master_faces.shape[0]),
        "sfc_slave_quadrature_count": int(4 * len(slave_quads)),
    }


def _synthetic_reference(model: C3D8TrajectoryModel) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[float, np.ndarray], dict[float, Row], dict[float, list[Row]], dict[float, dict[int, np.ndarray]], Row]:
    times = np.arange(model.dt, model.total_time + 0.5 * model.dt, model.dt)
    displacements: dict[float, np.ndarray] = {}
    velocities: dict[float, np.ndarray] = {}
    accelerations: dict[float, np.ndarray] = {}
    totals: dict[float, Row] = {}
    contact_rows: dict[float, list[Row]] = {}
    stresses: dict[float, dict[int, np.ndarray]] = {}
    for time in times:
        progress = float(time / model.total_time)
        drop = 0.070 * progress * progress
        velocity = -0.140 * progress / model.total_time
        U = np.zeros_like(model.X)
        V = np.zeros_like(model.X)
        A = np.zeros_like(model.X)
        U[model.slave_node_indices, 2] -= drop
        V[model.slave_node_indices, 2] = velocity
        A[model.slave_node_indices, 2] = -0.140 / model.total_time
        replay = _replay_dynamic_sdf(model, U)
        displacements[float(time)] = U
        velocities[float(time)] = V
        accelerations[float(time)] = A
        totals[float(time)] = {
            "normal_force_calculix": abs(float(replay["sfc_normal_force_z"])),
            "contact_energy_calculix": float(replay["sfc_contact_energy"]),
            "contact_count_calculix": int(replay["sfc_active_contact_count"]),
        }
        contact_rows[float(time)] = [
            {"quantity": "cdis", "slave_element": 0, "slave_face": 0, "clearance": float(replay["sfc_min_gap"])}
        ]
        stress_map: dict[int, np.ndarray] = {}
        for element_id, element in zip(model.element_ids, model.elements, strict=True):
            _strain, stress = hex8_center_strain_stress(model.X[element], U[element], model.E, model.nu)
            stress_map[int(element_id)] = stress
        stresses[float(time)] = stress_map
    command_row: Row = {
        "case": model.case,
        "resolution": model.resolution,
        "external_solver": "synthetic_calculix_like_reference",
        "external_solver_version": "not_run",
        "command": "synthetic reference generated by --skip-calculix",
        "input_file": "",
        "dat_file": "",
        "stdout_log": "",
        "stderr_log": "",
        "return_code": "not_run",
        "timed_out": "false",
        "completed": "false",
        "wall_time_seconds": "",
    }
    return displacements, velocities, accelerations, totals, contact_rows, stresses, command_row


def _center_z(model: C3D8TrajectoryModel, U: np.ndarray) -> float:
    masses = np.zeros(model.X.shape[0], dtype=float)
    for element in model.elements:
        volume = hex8_volume(model.X[element])
        masses[element] += model.density * volume / 8.0
    slave_masses = masses[model.slave_node_indices]
    return float(np.dot(slave_masses, (model.X + U)[model.slave_node_indices, 2]) / np.sum(slave_masses))


def _strain_stress_cloud(model: C3D8TrajectoryModel, U: np.ndarray, stress_reference: dict[int, np.ndarray] | None) -> list[Row]:
    rows: list[Row] = []
    element_lookup = {int(element_id): i for i, element_id in enumerate(model.element_ids)}
    for element_id in model.element_ids:
        element = model.elements[element_lookup[int(element_id)]]
        strain, stress_sfc = hex8_center_strain_stress(model.X[element], U[element], model.E, model.nu)
        stress_calc = stress_reference.get(int(element_id), stress_sfc) if stress_reference is not None else stress_sfc
        rows.append(
            {
                "case": model.case,
                "resolution": model.resolution,
                "element_id": int(element_id),
                "engineering_strain_norm": float(np.linalg.norm(strain)),
                "sfc_von_mises": _von_mises(stress_sfc),
                "calculix_von_mises": _von_mises(stress_calc),
                "von_mises_abs_error": abs(_von_mises(stress_sfc) - _von_mises(stress_calc)),
            }
        )
    return rows


def _rows_for_model(
    model: C3D8TrajectoryModel,
    displacements: dict[float, np.ndarray],
    velocities: dict[float, np.ndarray],
    accelerations: dict[float, np.ndarray],
    totals: dict[float, Row],
    contact_rows: dict[float, list[Row]],
    stresses: dict[float, dict[int, np.ndarray]],
    *,
    reference_source: str,
) -> tuple[list[Row], list[Row]]:
    rows: list[Row] = []
    stress_rows: list[Row] = []
    for time, U in displacements.items():
        replay = _replay_dynamic_sdf(model, U)
        total = totals.get(float(time), {})
        contacts = contact_rows.get(float(time), [])
        cdis_values = [float(row["clearance"]) for row in contacts if row.get("quantity") == "cdis" and "clearance" in row]
        min_cdis = min(cdis_values) if cdis_values else ""
        normal_force_calculix = total.get("normal_force_calculix", "")
        cels = total.get("contact_energy_calculix", "")
        if cels == "":
            cels_values = [float(row["energy"]) for row in contacts if row.get("quantity") == "cels" and "energy" in row]
            if cels_values:
                cels = float(np.sum(cels_values))
        cnum = total.get("contact_count_calculix", "")
        force_error = ""
        energy_error = ""
        gap_error = ""
        if normal_force_calculix != "":
            force_error = abs(float(normal_force_calculix) - abs(float(replay["sfc_normal_force_z"])))
        if cels != "":
            energy_error = abs(float(cels) - float(replay["sfc_contact_energy"]))
        if min_cdis != "":
            gap_error = abs(float(min_cdis) - float(replay["sfc_min_gap"]))
        V = velocities.get(float(time), np.zeros_like(U))
        A = accelerations.get(float(time), np.zeros_like(U))
        rows.append(
            {
                "case": model.case,
                "resolution": model.resolution,
                "time": float(time),
                "reference_source": reference_source,
                "z_cm": _center_z(model, U),
                "v_cm_z": float(np.mean(V[model.slave_node_indices, 2])),
                "a_cm_z": float(np.mean(A[model.slave_node_indices, 2])),
                "calculix_min_cdis": min_cdis,
                "sfc_min_gap": replay["sfc_min_gap"],
                "gap_abs_error": gap_error,
                "calculix_contact_count": cnum,
                "sfc_active_contact_count": replay["sfc_active_contact_count"],
                "calculix_normal_force": normal_force_calculix,
                "sfc_normal_force_z": replay["sfc_normal_force_z"],
                "normal_force_abs_error": force_error,
                "calculix_contact_energy": cels,
                "sfc_contact_energy": replay["sfc_contact_energy"],
                "contact_energy_abs_error": energy_error,
                "sfc_max_penetration": replay["sfc_max_penetration"],
                "sfc_master_triangle_count": replay["sfc_master_triangle_count"],
                "sfc_slave_quadrature_count": replay["sfc_slave_quadrature_count"],
            }
        )
    if displacements:
        last_time = max(displacements)
        stress_rows = _strain_stress_cloud(model, displacements[last_time], stresses.get(float(last_time)))
    return rows, stress_rows


def _safe_rel(error: float, reference: float) -> float:
    return float(error / max(abs(reference), 1.0e-30))


def _summary_rows(history: list[Row]) -> list[Row]:
    rows: list[Row] = []
    for key in sorted({(row["case"], row["resolution"]) for row in history}):
        case, resolution = key
        subset = [row for row in history if row["case"] == case and row["resolution"] == resolution]
        completed_external = all(row["reference_source"] == "calculix_dat" for row in subset)
        active = [row for row in subset if int(row["sfc_active_contact_count"]) > 0]
        force_errors = [float(row["normal_force_abs_error"]) for row in subset if row["normal_force_abs_error"] != ""]
        energy_errors = [float(row["contact_energy_abs_error"]) for row in subset if row["contact_energy_abs_error"] != ""]
        gap_errors = [float(row["gap_abs_error"]) for row in subset if row["gap_abs_error"] != ""]
        peak_force_ref = max([float(row["calculix_normal_force"]) for row in subset if row["calculix_normal_force"] != ""] or [0.0])
        peak_energy_ref = max([float(row["calculix_contact_energy"]) for row in subset if row["calculix_contact_energy"] != ""] or [0.0])
        max_gap_ref = max([abs(float(row["calculix_min_cdis"])) for row in subset if row["calculix_min_cdis"] != ""] or [0.0])
        rows.append(
            {
                "case": case,
                "resolution": resolution,
                "reference_source": "calculix_dat" if completed_external else "synthetic_calculix_like_reference",
                "time_rows": len(subset),
                "active_sfc_rows": len(active),
                "max_sfc_penetration": max(float(row["sfc_max_penetration"]) for row in subset),
                "peak_calculix_normal_force": peak_force_ref,
                "peak_sfc_normal_force_z_abs": max(abs(float(row["sfc_normal_force_z"])) for row in subset),
                "peak_force_abs_error": max(force_errors or [0.0]),
                "peak_force_rel_error": _safe_rel(max(force_errors or [0.0]), peak_force_ref),
                "peak_calculix_contact_energy": peak_energy_ref,
                "peak_sfc_contact_energy": max(float(row["sfc_contact_energy"]) for row in subset),
                "peak_energy_abs_error": max(energy_errors or [0.0]),
                "peak_energy_rel_error": _safe_rel(max(energy_errors or [0.0]), peak_energy_ref),
                "min_gap_abs_error": max(gap_errors or [0.0]),
                "min_gap_rel_error": _safe_rel(max(gap_errors or [0.0]), max_gap_ref),
                "status": "external_replay" if completed_external else "synthetic_quick_check",
            }
        )
    return rows


def _nodal_average(elements: np.ndarray, values: np.ndarray, n_nodes: int) -> np.ndarray:
    sums = np.zeros(n_nodes, dtype=float)
    counts = np.zeros(n_nodes, dtype=float)
    for element, value in zip(elements, values, strict=True):
        sums[element] += float(value)
        counts[element] += 1.0
    return sums / np.maximum(counts, 1.0)


def _draw_surface(ax, X_ref: np.ndarray, X_plot: np.ndarray, faces: np.ndarray, values: np.ndarray, *, norm: Normalize, title: str) -> Poly3DCollection:
    collection = Poly3DCollection([X_plot[face] for face in faces], cmap="viridis", norm=norm, linewidths=0.15, edgecolors=(0.1, 0.1, 0.1, 0.25))
    collection.set_array(np.asarray([np.mean(values[face]) for face in faces], dtype=float))
    ax.add_collection3d(collection)
    wire = Poly3DCollection([X_ref[face] for face in faces], facecolors=(0.7, 0.7, 0.7, 0.02), edgecolors=(0.0, 0.0, 0.0, 0.12), linewidths=0.2)
    ax.add_collection3d(wire)
    bounds = np.vstack([X_ref, X_plot])
    mins = bounds.min(axis=0)
    maxs = bounds.max(axis=0)
    pad = np.maximum(0.05 * (maxs - mins), 1.0e-4)
    ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
    ax.set_zlim(float(mins[2] - pad[2]), float(maxs[2] + pad[2]))
    ax.set_box_aspect((maxs - mins + 2.0 * pad).clip(min=1.0e-12))
    ax.view_init(elev=22.0, azim=-50.0)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.set_title(title)
    ax.set_axis_off()
    return collection


def _write_plots(out_dir: Path, history: list[Row], stress_rows: list[Row], models_by_key: dict[tuple[str, int], C3D8TrajectoryModel], displacements_by_key: dict[tuple[str, int], dict[float, np.ndarray]]) -> list[Row]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_rows: list[Row] = []

    def save(fig, name: str, description: str) -> None:
        png = figures / f"{name}.png"
        pdf = figures / f"{name}.pdf"
        fig.savefig(png, dpi=180, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        plot_rows.append({"plot": name, "png": str(png.relative_to(out_dir)), "pdf": str(pdf.relative_to(out_dir)), "description": description})

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for case in sorted({row["case"] for row in history}):
        subset = [row for row in history if row["case"] == case]
        ax.plot([float(row["time"]) for row in subset], [float(row["z_cm"]) for row in subset], marker="o", markersize=2.5, label=case)
    ax.set_xlabel("time")
    ax.set_ylabel("slave body z center")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, "c3d8_contact_trajectory_z_cm", "C3D8 CalculiX trajectory center height")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for case in sorted({row["case"] for row in history}):
        subset = [row for row in history if row["case"] == case]
        ax.plot([float(row["time"]) for row in subset], [float(row["sfc_min_gap"]) for row in subset], marker="o", markersize=2.5, label=f"{case} SFC gap")
        cdis = [(float(row["time"]), float(row["calculix_min_cdis"])) for row in subset if row["calculix_min_cdis"] != ""]
        if cdis:
            ax.plot([t for t, _ in cdis], [g for _, g in cdis], linestyle="--", label=f"{case} CalculiX CDIS")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("gap / clearance")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "c3d8_contact_trajectory_gap", "SFC dynamic-SDF gap and CalculiX CDIS where available")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for case in sorted({row["case"] for row in history}):
        subset = [row for row in history if row["case"] == case]
        t = [float(row["time"]) for row in subset]
        axes[0].plot(t, [abs(float(row["sfc_normal_force_z"])) for row in subset], marker="o", markersize=2.0, label=f"{case} SFC")
        calc_force = [(float(row["time"]), float(row["calculix_normal_force"])) for row in subset if row["calculix_normal_force"] != ""]
        if calc_force:
            axes[0].plot([x for x, _ in calc_force], [y for _, y in calc_force], linestyle="--", label=f"{case} CalculiX")
        axes[1].plot(t, [float(row["sfc_contact_energy"]) for row in subset], marker="o", markersize=2.0, label=f"{case} SFC")
        calc_energy = [(float(row["time"]), float(row["calculix_contact_energy"])) for row in subset if row["calculix_contact_energy"] != ""]
        if calc_energy:
            axes[1].plot([x for x, _ in calc_energy], [y for _, y in calc_energy], linestyle="--", label=f"{case} CalculiX")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("normal force magnitude")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("contact energy")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    save(fig, "c3d8_contact_trajectory_force_energy", "Normal reaction and contact spring energy")

    if stress_rows:
        keys = sorted(models_by_key)[:2]
        fig, axes = plt.subplots(1, len(keys), figsize=(5.0 * len(keys), 4.0), subplot_kw={"projection": "3d"})
        if len(keys) == 1:
            axes = np.asarray([axes])
        max_vm = max(float(row["calculix_von_mises"]) for row in stress_rows) if stress_rows else 1.0
        norm = Normalize(vmin=0.0, vmax=max(max_vm, 1.0e-30))
        for ax, key in zip(axes, keys, strict=True):
            model = models_by_key[key]
            U = displacements_by_key[key][max(displacements_by_key[key])]
            element_values = []
            for element_id in model.element_ids:
                matches = [row for row in stress_rows if row["case"] == model.case and int(row["resolution"]) == model.resolution and int(row["element_id"]) == int(element_id)]
                element_values.append(float(matches[0]["calculix_von_mises"]) if matches else 0.0)
            nodal = _nodal_average(model.elements, np.asarray(element_values, dtype=float), model.X.shape[0])
            faces, _ = extract_boundary_triangles(model.elements, model.X, element_type="hex8")
            collection = _draw_surface(ax, model.X, model.X + 3.0 * U, faces, nodal, norm=norm, title=f"{model.case} final stress")
        fig.colorbar(collection, ax=list(axes), shrink=0.62, pad=0.03, label="von Mises stress")
        save(fig, "c3d8_contact_trajectory_stress_cloud", "Final C3D8 stress cloud from CalculiX displacement/stress output")
    return plot_rows


def _write_summary(out_dir: Path, summaries: list[Row], commands: list[Row], plots: list[Row]) -> None:
    lines = [
        "# C3D8 Dynamic Contact Trajectory Validation",
        "",
        "This validation generates C3D8 block-plane and block-block transient contact cases. CalculiX supplies the external trajectory; SFC replays the current deformed C3D8 boundary with dynamic SDF queries for gap, normal-force, contact-energy, and stress-cloud diagnostics.",
        "",
        "## Commands",
        "",
    ]
    for row in commands:
        lines.append(f"- `{row['command']}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Case | r | Source | Rows | Active rows | Peak force rel. error | Peak CELS rel. error | Gap rel. error | Status |",
            "|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['case']} | {row['resolution']} | {row['reference_source']} | {row['time_rows']} | "
            f"{row['active_sfc_rows']} | {float(row['peak_force_rel_error']):.6e} | "
            f"{float(row['peak_energy_rel_error']):.6e} | {float(row['min_gap_rel_error']):.6e} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `c3d8_contact_trajectory.csv`",
            "- `c3d8_contact_trajectory_summary.csv`",
            "- `c3d8_contact_trajectory_stress_cloud.csv`",
            "- `c3d8_contact_trajectory_commands.csv`",
            "- `c3d8_contact_trajectory_plots.csv`",
            "- `c3d8_contact_trajectory_claims.csv`",
            "",
            "## Scope",
            "",
            "Supported claim: current-surface dynamic SDF can replay C3D8 deformed contact trajectories exported by CalculiX and compute comparable gap, force, energy, and stress-cloud diagnostics in the tested cases.",
            "",
            "Unsupported claim: native SFC nonlinear C3D8 dynamics is exactly equivalent to CalculiX. The runner intentionally keeps CalculiX as the external trajectory source.",
            "",
            "## Figures",
            "",
        ]
    )
    for row in plots:
        lines.append(f"- `{row['png']}`: {row['description']}")
    (out_dir / "c3d8_contact_trajectory_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = build_model_suite(quick=args.quick, cases=args.case, resolutions=args.resolution)
    history: list[Row] = []
    stress_rows: list[Row] = []
    command_rows: list[Row] = []
    models_by_key: dict[tuple[str, int], C3D8TrajectoryModel] = {}
    displacements_by_key: dict[tuple[str, int], dict[float, np.ndarray]] = {}

    for model in models:
        models_by_key[(model.case, model.resolution)] = model
        reference_source = "calculix_dat"
        if args.skip_calculix:
            U, V, A, totals, contact_rows, stresses, command = _synthetic_reference(model)
            reference_source = "synthetic_calculix_like_reference"
        else:
            dat_path, command = run_calculix(model, out_dir, timeout_seconds=args.timeout_seconds)
            U = _parse_nodal_vectors(dat_path, model.node_ids, quantity="u")
            V = _parse_nodal_vectors(dat_path, model.node_ids, quantity="v")
            A = _parse_nodal_vectors(dat_path, model.node_ids, quantity="a")
            totals = _parse_totals(dat_path)
            contact_rows = _parse_contact_elements(dat_path)
            stresses = _parse_element_stress(dat_path)
        command_rows.append(command)
        displacements_by_key[(model.case, model.resolution)] = U
        model_rows, model_stress_rows = _rows_for_model(model, U, V, A, totals, contact_rows, stresses, reference_source=reference_source)
        history.extend(model_rows)
        stress_rows.extend(model_stress_rows)

    summaries = _summary_rows(history)
    plot_rows = _write_plots(out_dir, history, stress_rows, models_by_key, displacements_by_key)
    claims = [
        {
            "claim": "c3d8_dynamic_sdf_external_trajectory_replay",
            "supported": str(any(row["reference_source"] == "calculix_dat" for row in summaries)).lower(),
            "evidence_file": "c3d8_contact_trajectory.csv",
            "details": "CalculiX C3D8 trajectory replay with current-surface dynamic SDF; synthetic quick mode does not support the external claim.",
        },
        {
            "claim": "native_sfc_nonlinear_c3d8_trajectory_equivalence",
            "supported": "false",
            "evidence_file": "",
            "details": "This runner replays CalculiX trajectories; it is not a native nonlinear C3D8 solve-equivalence claim.",
        },
    ]

    _write_csv(
        out_dir / "c3d8_contact_trajectory.csv",
        [
            "case",
            "resolution",
            "time",
            "reference_source",
            "z_cm",
            "v_cm_z",
            "a_cm_z",
            "calculix_min_cdis",
            "sfc_min_gap",
            "gap_abs_error",
            "calculix_contact_count",
            "sfc_active_contact_count",
            "calculix_normal_force",
            "sfc_normal_force_z",
            "normal_force_abs_error",
            "calculix_contact_energy",
            "sfc_contact_energy",
            "contact_energy_abs_error",
            "sfc_max_penetration",
            "sfc_master_triangle_count",
            "sfc_slave_quadrature_count",
        ],
        history,
    )
    _write_csv(out_dir / "c3d8_contact_trajectory_summary.csv", list(summaries[0].keys()) if summaries else ["case"], summaries)
    _write_csv(
        out_dir / "c3d8_contact_trajectory_stress_cloud.csv",
        ["case", "resolution", "element_id", "engineering_strain_norm", "sfc_von_mises", "calculix_von_mises", "von_mises_abs_error"],
        stress_rows,
    )
    _write_csv(out_dir / "c3d8_contact_trajectory_commands.csv", list(command_rows[0].keys()) if command_rows else ["case"], command_rows)
    _write_csv(out_dir / "c3d8_contact_trajectory_plots.csv", ["plot", "png", "pdf", "description"], plot_rows)
    _write_csv(out_dir / "c3d8_contact_trajectory_claims.csv", ["claim", "supported", "evidence_file", "details"], claims)
    _write_summary(out_dir, summaries, command_rows, plot_rows)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run the smallest deterministic cases.")
    parser.add_argument("--skip-calculix", action="store_true", help="Use a synthetic CalculiX-like trajectory for CI-only output checks.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "c3d8_contact_trajectory_validation")
    parser.add_argument("--case", action="append", choices=["block_plane_c3d8", "block_block_c3d8"], help="Restrict to a case; may be repeated.")
    parser.add_argument("--resolution", action="append", type=int, help="Restrict to a C3D8 block resolution; may be repeated.")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_validation(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
