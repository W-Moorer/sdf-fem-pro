"""Abaqus BEAM-MPC RP angular velocity boundary vs SFC RP history.

This small layer validates the gear-model feature where an angular velocity is
prescribed on a reference-point rotational degree of freedom.  The Abaqus case
ties a C3D4 hub to an RP using ``*MPC, BEAM`` and applies
``*Boundary, type=VELOCITY`` to RP dof 6.  SFC evaluates the same prescribed
rotation history directly.

Abaqus is an external reference only; no Abaqus module is imported by SFC core.
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.fem.rp_mpc import constant_angular_velocity_rotation_history  # noqa: E402

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "abaqus_rp_angular_velocity_alignment"


@dataclass(frozen=True, slots=True)
class RPAngularVelocityCase:
    nodes: np.ndarray
    rp: np.ndarray
    angular_velocity: np.ndarray
    duration: float = 0.02
    dt: float = 0.002
    young: float = 1.0e7
    poisson: float = 0.3
    density: float = 1000.0


def build_case() -> RPAngularVelocityCase:
    nodes = np.asarray(
        [
            [0.10, -0.18, -0.12],
            [0.72, 0.06, -0.08],
            [0.18, 0.44, 0.04],
            [0.22, -0.02, 0.52],
        ],
        dtype=float,
    )
    return RPAngularVelocityCase(
        nodes=nodes,
        rp=np.asarray([0.18, -0.03, 0.02], dtype=float),
        angular_velocity=np.asarray([0.0, 0.0, 5.236], dtype=float),
    )


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_abaqus_deck(case: RPAngularVelocityCase, path: Path) -> None:
    min_dt = min(float(case.dt) * 1.0e-4, 1.0e-8)
    lines = [
        "*Heading",
        "RP angular velocity boundary alignment with BEAM MPC",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
        "*Node",
    ]
    for idx, xyz in enumerate(case.nodes, start=1):
        lines.append(f"{idx}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    lines.append(f"100, {case.rp[0]:.12e}, {case.rp[1]:.12e}, {case.rp[2]:.12e}")
    lines.extend(
        [
            "*Element, type=C3D4, elset=EALL",
            "1, 1, 2, 3, 4",
            "*Nset, nset=HUB",
            "1, 2, 3, 4",
            "*Nset, nset=RP",
            "100",
            "*MPC",
            "BEAM, HUB, RP",
            "*Material, name=MAT",
            "*Density",
            f"{case.density:.12e}",
            "*Elastic",
            f"{case.young:.12e}, {case.poisson:.12e}",
            "*Solid Section, elset=EALL, material=MAT",
            ",",
            "*Step, name=rp_angular_velocity, nlgeom=YES, inc=1000",
            "*Dynamic",
            f"{case.dt:.12e}, {case.duration:.12e}, {min_dt:.12e}, {case.dt:.12e}",
            "*Boundary",
            "RP, 1, 5, 0.",
            "*Boundary, type=VELOCITY",
            f"RP, 6, 6, {case.angular_velocity[2]:.12e}",
            "*Output, field, frequency=1",
            "*Node Output, nset=RP",
            "U, UR, V, VR, A, AR",
            "*End Step",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def _resolve_abaqus_command(command: str | None) -> str:
    if command:
        return command
    for candidate in ("abaqus", "abq2024"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Abaqus command not found; pass --abaqus-command")


def _run_command(command: list[str], *, cwd: Path, log_path: Path, timeout: int) -> float:
    start = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=timeout)
    elapsed = time.perf_counter() - start
    log_path.write_text(result.stdout, encoding="utf-8", errors="ignore")
    print(result.stdout)
    text = result.stdout.lower()
    if result.returncode != 0 or "exited with errors" in text or "fatal errors" in text or "traceback" in text:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}")
    return elapsed


def _abaqus_export_script() -> str:
    return r'''
from __future__ import print_function

import csv
import sys

from odbAccess import openOdb

odb = openOdb(path=sys.argv[1], readOnly=True)
out_path = sys.argv[2]
try:
    step = odb.steps[list(odb.steps.keys())[0]]
    rows = []
    for frame in step.frames:
        rz = 0.0
        wz = 0.0
        if "UR" in frame.fieldOutputs:
            for value in frame.fieldOutputs["UR"].values:
                if int(value.nodeLabel) == 100:
                    rz = float(value.data[2])
        if "VR" in frame.fieldOutputs:
            for value in frame.fieldOutputs["VR"].values:
                if int(value.nodeLabel) == 100:
                    wz = float(value.data[2])
        rows.append({"time": float(frame.frameValue), "rotation_z": rz, "angular_velocity_z": wz})
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "rotation_z", "angular_velocity_z"])
        writer.writeheader()
        writer.writerows(rows)
finally:
    odb.close()
'''


def run_abaqus(case: RPAngularVelocityCase, out_dir: Path, *, abaqus_command: str | None, timeout: int) -> tuple[Path, Row]:
    run_dir = out_dir / "abaqus_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    job = "rp_angular_velocity"
    for old in run_dir.glob(f"{job}.*"):
        old.unlink()
    deck = run_dir / f"{job}.inp"
    write_abaqus_deck(case, deck)
    command = _resolve_abaqus_command(abaqus_command)
    analysis_wall = _run_command([command, f"job={job}", f"input={deck.name}", "interactive"], cwd=run_dir, log_path=out_dir / "abaqus_stdout.log", timeout=timeout)
    odb = run_dir / f"{job}.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    script = run_dir / "export_rp_angular_velocity.py"
    script.write_text(_abaqus_export_script(), encoding="ascii")
    metrics = out_dir / "abaqus_rp_angular_velocity_history.csv"
    export_wall = _run_command([command, "python", str(script.resolve()), str(odb.resolve()), str(metrics.resolve())], cwd=run_dir, log_path=out_dir / "abaqus_export_stdout.log", timeout=timeout)
    return metrics, {"solver": "abaqus", "analysis_wall_seconds": analysis_wall, "export_wall_seconds": export_wall}


def run_sfc(case: RPAngularVelocityCase, out_path: Path) -> tuple[Path, Row]:
    start = time.perf_counter()
    times, rotations, angular_velocities = constant_angular_velocity_rotation_history(
        case.angular_velocity,
        duration=case.duration,
        dt=case.dt,
    )
    wall = time.perf_counter() - start
    rows = [
        {"time": float(t), "rotation_z": float(r[2]), "angular_velocity_z": float(w[2])}
        for t, r, w in zip(times, rotations, angular_velocities, strict=True)
    ]
    _write_csv(out_path, rows)
    return out_path, {"solver": "sfc_prescribed_rp_velocity", "analysis_wall_seconds": wall}


def _read_rows(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _col(rows: list[Row], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0) or 0.0) for row in rows], dtype=float)


def compare(abaqus_path: Path, sfc_path: Path, out_path: Path) -> list[Row]:
    abq = _read_rows(abaqus_path)
    sfc = _read_rows(sfc_path)
    t_abq = _col(abq, "time")
    rows: list[Row] = []
    for row in sfc:
        t = float(row["time"])
        rz_sfc = float(row["rotation_z"])
        wz_sfc = float(row["angular_velocity_z"])
        rz_abq = float(np.interp(t, t_abq, _col(abq, "rotation_z")))
        wz_abq = float(np.interp(t, t_abq, _col(abq, "angular_velocity_z")))
        rows.append(
            {
                "time": t,
                "abaqus_rotation_z": rz_abq,
                "sfc_rotation_z": rz_sfc,
                "rotation_z_abs_error": abs(rz_sfc - rz_abq),
                "rotation_z_rel_error": abs(rz_sfc - rz_abq) / max(abs(rz_abq), 1.0e-14),
                "abaqus_angular_velocity_z": wz_abq,
                "sfc_angular_velocity_z": wz_sfc,
                "angular_velocity_z_abs_error": abs(wz_sfc - wz_abq),
                "angular_velocity_z_rel_error": abs(wz_sfc - wz_abq) / max(abs(wz_abq), 1.0e-14),
            }
        )
    _write_csv(out_path, rows)
    return rows


def write_summary(path: Path, rows: list[Row], abaqus_runtime: Row, sfc_runtime: Row) -> None:
    final = rows[-1]
    text = [
        "# Abaqus RP Angular Velocity Alignment",
        "",
        "A C3D4 hub is tied to an RP with Abaqus `*MPC, BEAM`; RP dof 6 receives a prescribed angular velocity and is compared with SFC prescribed RP rotation history.",
        "",
        f"- Abaqus wall time: {float(abaqus_runtime.get('analysis_wall_seconds', 0.0)):.6f} s",
        f"- SFC prescribed-RP wall time: {float(sfc_runtime.get('analysis_wall_seconds', 0.0)):.6e} s",
        f"- final rotation z rel. error: {100.0 * float(final.get('rotation_z_rel_error', 0.0)):.6f}%",
        f"- final angular velocity z rel. error: {100.0 * float(final.get('angular_velocity_z_rel_error', 0.0)):.6f}%",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--abaqus-command", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case = build_case()
    abaqus_path, abaqus_runtime = run_abaqus(case, out_dir, abaqus_command=args.abaqus_command, timeout=int(args.timeout))
    sfc_path, sfc_runtime = run_sfc(case, out_dir / "sfc_rp_angular_velocity_history.csv")
    rows = compare(abaqus_path, sfc_path, out_dir / "abaqus_vs_sfc_rp_angular_velocity_errors.csv")
    _write_csv(out_dir / "solver_runtime.csv", [abaqus_runtime, sfc_runtime])
    summary = out_dir / "abaqus_rp_angular_velocity_summary.md"
    write_summary(summary, rows, abaqus_runtime, sfc_runtime)
    print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
