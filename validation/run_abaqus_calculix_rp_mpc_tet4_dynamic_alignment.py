"""Abaqus vs CalculiX RP/hub MPC TET4 implicit-dynamics alignment.

This validation runner builds one deterministic TET4 beam-like model with a
hub/reference-point rigid kinematic coupling.  The left end is fixed; the
right-end hub is tied to an RP and driven by a ramped transverse displacement
plus a small rotation.  The same input topology, material, implicit dynamic
step, and output requests are run in Abaqus/Standard and CalculiX/ccx.

The goal is an external solver-to-solver alignment case for the modeling
features needed before SFC Lagrangian-SDF gear/contact comparisons:

- C3D4/TET4 dynamics.
- Abaqus-style hub/RP coupling represented with the solver-native rigid-body
  coupling syntax in Abaqus and CalculiX.
- Displacement, stress, and strain history extraction.

Abaqus and CalculiX are used only as external validation references.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from validation.run_phase3_validation import structured_tet_block  # noqa: E402

Row = dict[str, Any]

DEFAULT_OUT_DIR = ROOT / "results" / "abaqus_calculix_rp_mpc_tet4_dynamic"


@dataclass(frozen=True, slots=True)
class RPHubDynamicModel:
    nodes: np.ndarray
    elements: np.ndarray
    left_nodes: np.ndarray
    hub_nodes: np.ndarray
    rp_node_id: int
    rp: np.ndarray
    young: float = 2.0e7
    poisson: float = 0.3
    density: float = 1200.0
    duration: float = 0.01
    dt: float = 0.002
    transverse_disp_y: float = 0.002
    rotation_z: float = 0.01

    @property
    def node_ids(self) -> np.ndarray:
        return np.arange(1, self.nodes.shape[0] + 1, dtype=np.int64)

    @property
    def element_ids(self) -> np.ndarray:
        return np.arange(1, self.elements.shape[0] + 1, dtype=np.int64)


def build_model(*, resolution: int = 2, duration: float = 0.01, dt: float = 0.002) -> RPHubDynamicModel:
    mesh = structured_tet_block(max(2, int(resolution) * 2), max(1, int(resolution)), max(1, int(resolution)), size=(1.0, 0.18, 0.18))
    nodes = np.asarray(mesh.X, dtype=float)
    nodes[:, 1] -= 0.09
    nodes[:, 2] -= 0.09
    elements = np.asarray(mesh.elements, dtype=np.int64)
    left = np.flatnonzero(np.isclose(nodes[:, 0], nodes[:, 0].min())) + 1
    hub = np.flatnonzero(np.isclose(nodes[:, 0], nodes[:, 0].max())) + 1
    rp = np.asarray([nodes[:, 0].max(), 0.0, 0.0], dtype=float)
    return RPHubDynamicModel(
        nodes=nodes,
        elements=elements,
        left_nodes=left.astype(np.int64),
        hub_nodes=hub.astype(np.int64),
        rp_node_id=int(nodes.shape[0] + 1),
        rp=rp,
        duration=float(duration),
        dt=float(dt),
    )


def _format_id_lines(ids: np.ndarray, *, per_line: int = 16) -> list[str]:
    values = [int(v) for v in np.asarray(ids, dtype=np.int64).ravel()]
    return [", ".join(str(v) for v in values[start : start + per_line]) for start in range(0, len(values), per_line)]


def write_common_input(model: RPHubDynamicModel, path: Path, *, solver: str) -> None:
    """Write an Abaqus/CalculiX-compatible input file."""

    min_increment = min(float(model.dt) * 1.0e-4, 1.0e-8)
    lines: list[str] = [
        "*Heading",
        f"RP hub MPC TET4 dynamic alignment for {solver}",
        "*Node, NSET=NALL",
    ]
    for node_id, xyz in zip(model.node_ids, model.nodes, strict=True):
        lines.append(f"{int(node_id)}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    lines.append(f"{model.rp_node_id}, {model.rp[0]:.12e}, {model.rp[1]:.12e}, {model.rp[2]:.12e}")
    lines.append("*Element, type=C3D4, ELSET=EALL")
    for element_id, tet in zip(model.element_ids, model.elements, strict=True):
        ids = [int(v) + 1 for v in tet]
        lines.append(f"{int(element_id)}, {ids[0]}, {ids[1]}, {ids[2]}, {ids[3]}")
    lines.append("*Nset, nset=LEFT")
    lines.extend(_format_id_lines(model.left_nodes))
    lines.append("*Nset, nset=HUB")
    lines.extend(_format_id_lines(model.hub_nodes))
    lines.append("*Nset, nset=RP")
    lines.append(str(model.rp_node_id))
    if solver.lower().startswith("calculix"):
        rot_node_id = model.rp_node_id + 1
        lines.append("*Node")
        lines.append(f"{rot_node_id}, {model.rp[0]:.12e}, {model.rp[1]:.12e}, {model.rp[2]:.12e}")
        lines.append("*Nset, nset=ROT")
        lines.append(str(rot_node_id))
        lines.append(f"*Rigid Body, NSET=HUB, REF NODE={model.rp_node_id}, ROT NODE={rot_node_id}")
    else:
        lines.append(f"*Rigid Body, REF NODE={model.rp_node_id}, TIE NSET=HUB")
    lines.extend(
        [
            "*Material, name=MAT",
            "*Density",
            f"{model.density:.12e}",
            "*Elastic",
            f"{model.young:.12e}, {model.poisson:.12e}",
            "*Solid Section, elset=EALL, material=MAT",
            ",",
            "*Amplitude, name=RAMP",
            f"0., 0., {model.duration:.12e}, 1.",
            "*Step, name=dynamic_alignment, nlgeom=NO, inc=1000",
            "*Dynamic",
            f"{model.dt:.12e}, {model.duration:.12e}, {min_increment:.12e}, {model.dt:.12e}",
            "*Boundary",
            "LEFT, 1, 3, 0.",
            "RP, 1, 1, 0.",
            "RP, 3, 3, 0.",
            "RP, 4, 5, 0.",
            "*Boundary, amplitude=RAMP",
            f"RP, 2, 2, {model.transverse_disp_y:.12e}",
            f"RP, 6, 6, {model.rotation_z:.12e}",
            "*Node Print, nset=NALL, frequency=1",
            "U",
            "*El Print, elset=EALL, frequency=1",
            "S",
            "E",
            "*End Step",
            "",
        ]
    )
    if not solver.lower().startswith("calculix"):
        insert_at = lines.index("*Node Print, nset=NALL, frequency=1")
        lines[insert_at:insert_at] = [
            "*Output, field, frequency=1",
            "*Node Output",
            "U",
            "*Element Output",
            "S, E",
        ]
    else:
        boundary_index = lines.index("*Boundary")
        lines[boundary_index + 5 : boundary_index + 5] = [
            "ROT, 1, 2, 0.",
        ]
        amp_index = lines.index("*Boundary, amplitude=RAMP")
        lines[amp_index + 2] = f"ROT, 3, 3, {model.rotation_z:.12e}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


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


def _resolve_abaqus_command(command: str | None) -> str:
    if command:
        return command
    for candidate in ("abaqus", "abq2024"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Abaqus command not found; pass --abaqus-command")


def _run_command(command: list[str], *, cwd: Path, log_path: Path, timeout: int = 300) -> float:
    start = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=timeout)
    elapsed = time.perf_counter() - start
    log_path.write_text(result.stdout, encoding="utf-8", errors="ignore")
    print(result.stdout)
    text = result.stdout.lower()
    if result.returncode != 0 or "exited with errors" in text or "fatal errors" in text or "traceback" in text:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}")
    return elapsed


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    parts = [part for part in resolved.parts[1:]]
    return "/mnt/" + drive + "/" + "/".join(part.replace("\\", "/") for part in parts)


def _run_wsl(command: str, *, cwd: Path, log_path: Path, timeout: int = 300) -> float:
    start = time.perf_counter()
    proc = subprocess.run(["wsl", "--exec", "bash", "-lc", command], cwd=cwd, capture_output=True, text=True, check=False, timeout=timeout)
    elapsed = time.perf_counter() - start
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8", errors="ignore")
    print(proc.stdout)
    print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"WSL command failed with exit code {proc.returncode}: {command}")
    return elapsed


def _abaqus_export_script() -> str:
    return r'''
from __future__ import print_function

import csv
import math
import sys

from odbAccess import openOdb


def norm3(data):
    return math.sqrt(float(data[0]) ** 2 + float(data[1]) ** 2 + float(data[2]) ** 2)


def strain_tensor_norm(data):
    values = [float(v) for v in data]
    if len(values) < 6:
        return 0.0
    # Abaqus strain field outputs use engineering shear components.  Convert
    # them to tensor shear before comparing with CalculiX EL PRINT strains.
    values[3] *= 0.5
    values[4] *= 0.5
    values[5] *= 0.5
    return math.sqrt(values[0] ** 2 + values[1] ** 2 + values[2] ** 2 + 2.0 * (values[3] ** 2 + values[4] ** 2 + values[5] ** 2))


def von_mises(data):
    values = [float(v) for v in data]
    if len(values) < 6:
        return 0.0
    s11, s22, s33, s12, s13, s23 = values[:6]
    return math.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * (s12 * s12 + s13 * s13 + s23 * s23))


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct / 100.0))
    return float(ordered[index])


odb = openOdb(path=sys.argv[1], readOnly=True)
out_path = sys.argv[2]
max_node_label = int(sys.argv[3])
try:
    step = odb.steps[list(odb.steps.keys())[0]]
    with open(out_path, "w", newline="") as handle:
        fieldnames = ["time", "max_displacement_norm", "p95_von_mises", "max_von_mises", "p95_strain_norm", "max_strain_norm"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame in step.frames:
            u_norm = []
            vm = []
            strain = []
            if "U" in frame.fieldOutputs:
                for value in frame.fieldOutputs["U"].values:
                    if int(value.nodeLabel) <= max_node_label:
                        u_norm.append(norm3(value.data))
            if "S" in frame.fieldOutputs:
                for value in frame.fieldOutputs["S"].values:
                    try:
                        vm.append(float(value.mises))
                    except Exception:
                        vm.append(von_mises(value.data))
            strain_output = frame.fieldOutputs["E"] if "E" in frame.fieldOutputs else (frame.fieldOutputs["LE"] if "LE" in frame.fieldOutputs else None)
            if strain_output is not None:
                for value in strain_output.values:
                    strain.append(strain_tensor_norm(value.data))
            writer.writerow({
                "time": float(frame.frameValue),
                "max_displacement_norm": max(u_norm) if u_norm else 0.0,
                "p95_von_mises": percentile(vm, 95.0),
                "max_von_mises": max(vm) if vm else 0.0,
                "p95_strain_norm": percentile(strain, 95.0),
                "max_strain_norm": max(strain) if strain else 0.0,
            })
finally:
    odb.close()
'''


def run_abaqus(model: RPHubDynamicModel, out_dir: Path, *, abaqus_command: str | None, timeout: int) -> tuple[Path, Row]:
    run_dir = out_dir / "abaqus_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    job = "rp_mpc_tet4_dynamic_abaqus"
    for old in run_dir.glob(f"{job}.*"):
        old.unlink()
    inp = run_dir / f"{job}.inp"
    write_common_input(model, inp, solver="Abaqus")
    command = _resolve_abaqus_command(abaqus_command)
    analysis_wall = _run_command([command, f"job={job}", f"input={inp.name}", "interactive"], cwd=run_dir, log_path=out_dir / "abaqus_stdout.log", timeout=timeout)
    odb = run_dir / f"{job}.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    script = run_dir / "export_abaqus_metrics.py"
    script.write_text(_abaqus_export_script(), encoding="ascii")
    metrics = out_dir / "abaqus_rp_mpc_tet4_dynamic_metrics.csv"
    export_wall = _run_command(
        [command, "python", str(script.resolve()), str(odb.resolve()), str(metrics.resolve()), str(model.nodes.shape[0])],
        cwd=run_dir,
        log_path=out_dir / "abaqus_export_stdout.log",
        timeout=timeout,
    )
    return metrics, {"solver": "abaqus", "analysis_wall_seconds": analysis_wall, "export_wall_seconds": export_wall, "status": "completed"}


def _parse_calculix_dat_displacements(path: Path, node_ids: np.ndarray) -> dict[float, np.ndarray]:
    id_to_row = {int(node_id): i for i, node_id in enumerate(node_ids)}
    blocks: dict[float, np.ndarray] = {}
    current_time: float | None = None
    current = np.zeros((len(node_ids), 3), dtype=float)
    seen: set[int] = set()
    header = re.compile(r"displacements.*time\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE)
    row = re.compile(r"^\s*(\d+)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)")
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
        if not match:
            continue
        node_id = int(match.group(1))
        if node_id not in id_to_row or node_id in seen:
            continue
        current[id_to_row[node_id], :] = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
        seen.add(node_id)
    if current_time is not None and seen:
        blocks[current_time] = current.copy()
    if not blocks:
        raise RuntimeError(f"no displacement blocks parsed from {path}")
    return dict(sorted(blocks.items()))


def _parse_calculix_element_metric_blocks(path: Path, *, element_count: int, kind: str) -> dict[float, np.ndarray]:
    """Parse simple CalculiX ``*EL PRINT`` stress/strain numeric blocks."""

    if kind == "stress":
        header = re.compile(r"stresses.*time\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE)
    elif kind == "strain":
        header = re.compile(r"strains.*time\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE)
    else:
        raise ValueError("kind must be stress or strain")
    row = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    )
    blocks: dict[float, list[list[float]]] = {}
    current_time: float | None = None
    values: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = header.search(line)
        if match:
            if current_time is not None and values:
                blocks[current_time] = values
            current_time = float(match.group(1))
            values = []
            continue
        if current_time is None:
            continue
        match = row.match(line)
        if not match:
            continue
        values.append([float(match.group(i)) for i in range(3, 9)])
        if len(values) >= element_count:
            blocks[current_time] = values
            current_time = None
            values = []
    if current_time is not None and values:
        blocks[current_time] = values
    return {time: np.asarray(vals, dtype=float) for time, vals in sorted(blocks.items())}


def _von_mises_from_voigt(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty(0, dtype=float)
    s11, s22, s33, s12, s13, s23 = (values[:, i] for i in range(6))
    return np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * (s12 * s12 + s13 * s13 + s23 * s23))


def _tensor_norm_from_voigt(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty(0, dtype=float)
    return np.sqrt(values[:, 0] ** 2 + values[:, 1] ** 2 + values[:, 2] ** 2 + 2.0 * (values[:, 3] ** 2 + values[:, 4] ** 2 + values[:, 5] ** 2))


def run_calculix(model: RPHubDynamicModel, out_dir: Path, *, timeout: int) -> tuple[Path, Row]:
    run_dir = out_dir / "calculix_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    job = "rp_mpc_tet4_dynamic_calculix"
    for old in run_dir.glob(f"{job}.*"):
        old.unlink()
    inp = run_dir / f"{job}.inp"
    write_common_input(model, inp, solver="CalculiX")
    wall = _run_wsl(f"cd {_wsl_path(run_dir)} && ccx {job}", cwd=ROOT, log_path=out_dir / "calculix_stdout.log", timeout=timeout)
    dat = run_dir / f"{job}.dat"
    if not dat.exists():
        raise FileNotFoundError(dat)
    disp = _parse_calculix_dat_displacements(dat, model.node_ids)
    stresses = _parse_calculix_element_metric_blocks(dat, element_count=model.elements.shape[0], kind="stress")
    strains = _parse_calculix_element_metric_blocks(dat, element_count=model.elements.shape[0], kind="strain")
    rows: list[Row] = [{"time": 0.0, "max_displacement_norm": 0.0, "p95_von_mises": 0.0, "max_von_mises": 0.0, "p95_strain_norm": 0.0, "max_strain_norm": 0.0}]
    for time_value, u in disp.items():
        vm = _von_mises_from_voigt(stresses.get(time_value, np.empty((0, 6), dtype=float)))
        en = _tensor_norm_from_voigt(strains.get(time_value, np.empty((0, 6), dtype=float)))
        rows.append(
            {
                "time": float(time_value),
                "max_displacement_norm": float(np.max(np.linalg.norm(u, axis=1))),
                "p95_von_mises": float(np.percentile(vm, 95.0)) if vm.size else 0.0,
                "max_von_mises": float(np.max(vm)) if vm.size else 0.0,
                "p95_strain_norm": float(np.percentile(en, 95.0)) if en.size else 0.0,
                "max_strain_norm": float(np.max(en)) if en.size else 0.0,
            }
        )
    metrics = out_dir / "calculix_rp_mpc_tet4_dynamic_metrics.csv"
    _write_csv(metrics, rows)
    return metrics, {"solver": "calculix", "analysis_wall_seconds": wall, "export_wall_seconds": 0.0, "status": "completed"}


def _read_rows(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float_column(rows: list[Row], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0) or 0.0) for row in rows], dtype=float)


def _relative_error(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(b)), 1.0e-12)


def compare_metrics(abaqus_path: Path, calculix_path: Path, out_path: Path) -> list[Row]:
    abq = _read_rows(abaqus_path)
    ccx = _read_rows(calculix_path)
    t_abq = _float_column(abq, "time")
    t_ccx = _float_column(ccx, "time")
    if t_abq.size == 0 or t_ccx.size == 0:
        raise RuntimeError("solver metric CSV contains no rows")
    metrics = ["max_displacement_norm", "p95_von_mises", "max_von_mises", "p95_strain_norm", "max_strain_norm"]
    rows: list[Row] = []
    for i, time_value in enumerate(t_abq):
        row: Row = {"time": float(time_value)}
        for key in metrics:
            abq_value = float(abq[i].get(key, 0.0) or 0.0)
            ccx_value = float(np.interp(time_value, t_ccx, _float_column(ccx, key)))
            row[f"abaqus_{key}"] = abq_value
            row[f"calculix_{key}"] = ccx_value
            row[f"{key}_abs_error"] = abs(abq_value - ccx_value)
            row[f"{key}_rel_error"] = _relative_error(ccx_value, abq_value)
        rows.append(row)
    _write_csv(out_path, rows)
    return rows


def plot_curves(abaqus_path: Path, calculix_path: Path, error_rows: list[Row], out_path: Path) -> None:
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "axes.unicode_minus": False})
    abq = _read_rows(abaqus_path)
    ccx = _read_rows(calculix_path)
    t_abq = _float_column(abq, "time")
    t_ccx = _float_column(ccx, "time")
    panels = [
        ("max_displacement_norm", "max displacement norm"),
        ("p95_von_mises", "p95 von Mises stress"),
        ("p95_strain_norm", "p95 strain norm"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7), constrained_layout=True)
    for ax, (key, title) in zip(axes.ravel(), panels, strict=True):
        final_error = float(error_rows[-1].get(f"{key}_rel_error", 0.0)) if error_rows else 0.0
        ax.plot(t_abq, _float_column(abq, key), color="#1f77b4", linewidth=1.8, label="Abaqus/Standard")
        ax.plot(t_ccx, _float_column(ccx, key), color="#ff7f0e", linewidth=1.8, linestyle="--", label=f"CalculiX ({100.0 * final_error:.2f}% final err.)")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.legend(loc="best", fontsize=7, frameon=False)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_summary(path: Path, model: RPHubDynamicModel, abaqus_row: Row, calculix_row: Row, error_rows: list[Row], figure: Path) -> None:
    final = error_rows[-1]
    text = [
        "# Abaqus-CalculiX RP/Hub TET4 Dynamic Alignment",
        "",
        "The same C3D4/TET4 beam, solver-native rigid hub/RP coupling, implicit dynamic step, and ramped RP motion were run in Abaqus/Standard and CalculiX.",
        "",
        f"- nodes/elements: {model.nodes.shape[0]} / {model.elements.shape[0]}",
        f"- duration/dt: {model.duration:.6g} / {model.dt:.6g}",
        f"- Abaqus wall time: {float(abaqus_row['analysis_wall_seconds']):.6f} s",
        f"- CalculiX wall time: {float(calculix_row['analysis_wall_seconds']):.6f} s",
        "",
        "## Final-Time Errors",
        "",
        f"- max displacement norm: {100.0 * float(final.get('max_displacement_norm_rel_error', 0.0)):.3f}%",
        f"- p95 von Mises: {100.0 * float(final.get('p95_von_mises_rel_error', 0.0)):.3f}%",
        f"- p95 strain norm: {100.0 * float(final.get('p95_strain_norm_rel_error', 0.0)):.3f}%",
        "",
        "The strain norm comparison uses a tensor-shear convention; Abaqus engineering shear strain components are converted before the norm is evaluated.",
        "",
        f"- figure: `{figure.name}`",
        "",
        "This is an external solver alignment case only; neither Abaqus nor CalculiX is imported by the SFC core solver.",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--duration", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--abaqus-command", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(resolution=int(args.resolution), duration=float(args.duration), dt=float(args.dt))
    abaqus_metrics, abaqus_row = run_abaqus(model, out_dir, abaqus_command=args.abaqus_command, timeout=int(args.timeout))
    calculix_metrics, calculix_row = run_calculix(model, out_dir, timeout=int(args.timeout))
    _write_csv(out_dir / "solver_runtime.csv", [abaqus_row, calculix_row])
    error_rows = compare_metrics(abaqus_metrics, calculix_metrics, out_dir / "abaqus_vs_calculix_rp_mpc_tet4_dynamic_errors.csv")
    figure = out_dir / "abaqus_vs_calculix_rp_mpc_tet4_dynamic_curves.png"
    plot_curves(abaqus_metrics, calculix_metrics, error_rows, figure)
    summary = out_dir / "abaqus_calculix_rp_mpc_tet4_dynamic_summary.md"
    write_summary(summary, model, abaqus_row, calculix_row, error_rows, figure)
    print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
