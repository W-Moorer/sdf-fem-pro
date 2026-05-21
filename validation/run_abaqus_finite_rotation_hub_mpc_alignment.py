"""Abaqus BEAM-MPC vs SFC finite-rotation hub kinematics.

This small external validation case checks the first missing gear feature:
finite hub/reference-point rotation.  A single C3D4 tetrahedron has all four
nodes tied to an RP by an Abaqus ``*MPC, BEAM`` constraint.  The RP is given a
finite rotation and translation in an ``nlgeom=YES`` static step.  The exported
hub-node displacements are compared with
``sfc.fem.rp_mpc.FiniteRotationRigidHubMPC``.

Abaqus is used only as an external reference for this validation runner.
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

from sfc.fem.rp_mpc import FiniteRotationRigidHubMPC  # noqa: E402

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "abaqus_finite_rotation_hub_mpc_alignment"


@dataclass(frozen=True, slots=True)
class FiniteRotationHubCase:
    nodes: np.ndarray
    rp: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray
    young: float = 1.0e7
    poisson: float = 0.3
    density: float = 1000.0


def build_case() -> FiniteRotationHubCase:
    return FiniteRotationHubCase(
        nodes=np.asarray(
            [
                [0.35, -0.18, 0.05],
                [0.82, 0.07, -0.12],
                [0.11, 0.46, 0.18],
                [0.27, -0.04, 0.55],
            ],
            dtype=float,
        ),
        rp=np.asarray([0.18, -0.06, 0.04], dtype=float),
        translation=np.asarray([0.012, -0.008, 0.015], dtype=float),
        rotation=np.asarray([0.11, -0.07, 0.31], dtype=float),
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


def write_abaqus_deck(case: FiniteRotationHubCase, path: Path) -> None:
    lines = [
        "*Heading",
        "Finite rotation BEAM-MPC hub kinematic alignment",
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
            "*Step, name=finite_rotation, nlgeom=YES, inc=100",
            "*Static",
            "1., 1., 1e-08, 1.",
            "*Boundary",
            f"RP, 1, 1, {case.translation[0]:.12e}",
            f"RP, 2, 2, {case.translation[1]:.12e}",
            f"RP, 3, 3, {case.translation[2]:.12e}",
            f"RP, 4, 4, {case.rotation[0]:.12e}",
            f"RP, 5, 5, {case.rotation[1]:.12e}",
            f"RP, 6, 6, {case.rotation[2]:.12e}",
            "*Output, field, frequency=1",
            "*Node Output, nset=HUB",
            "U",
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
    lowered = result.stdout.lower()
    if result.returncode != 0 or "exited with errors" in lowered or "fatal errors" in lowered or "traceback" in lowered:
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
    frame = step.frames[-1]
    rows = []
    for value in frame.fieldOutputs["U"].values:
        label = int(value.nodeLabel)
        if 1 <= label <= 4:
            rows.append({"node_id": label, "ux": value.data[0], "uy": value.data[1], "uz": value.data[2]})
    rows.sort(key=lambda row: row["node_id"])
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_id", "ux", "uy", "uz"])
        writer.writeheader()
        writer.writerows(rows)
finally:
    odb.close()
'''


def run_abaqus(case: FiniteRotationHubCase, out_dir: Path, *, abaqus_command: str | None, timeout: int) -> tuple[Path, Row]:
    run_dir = out_dir / "abaqus_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    job = "finite_rotation_hub_mpc"
    for old in run_dir.glob(f"{job}.*"):
        old.unlink()
    deck = run_dir / f"{job}.inp"
    write_abaqus_deck(case, deck)
    command = _resolve_abaqus_command(abaqus_command)
    analysis_wall = _run_command([command, f"job={job}", f"input={deck.name}", "interactive"], cwd=run_dir, log_path=out_dir / "abaqus_stdout.log", timeout=timeout)
    odb = run_dir / f"{job}.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    script = run_dir / "export_hub_u.py"
    script.write_text(_abaqus_export_script(), encoding="ascii")
    metrics = out_dir / "abaqus_hub_displacements.csv"
    export_wall = _run_command([command, "python", str(script.resolve()), str(odb.resolve()), str(metrics.resolve())], cwd=run_dir, log_path=out_dir / "abaqus_export_stdout.log", timeout=timeout)
    return metrics, {"solver": "abaqus", "analysis_wall_seconds": analysis_wall, "export_wall_seconds": export_wall}


def sfc_displacements(case: FiniteRotationHubCase, out_path: Path) -> Path:
    hub = FiniteRotationRigidHubMPC(np.arange(case.nodes.shape[0], dtype=np.int64), case.nodes, case.rp)
    disp = hub.nodal_displacements(translation=case.translation, rotation=case.rotation)
    rows = [
        {"node_id": idx + 1, "ux": float(value[0]), "uy": float(value[1]), "uz": float(value[2])}
        for idx, value in enumerate(disp)
    ]
    _write_csv(out_path, rows)
    return out_path


def _read_disp(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append([float(row["ux"]), float(row["uy"]), float(row["uz"])])
    return np.asarray(rows, dtype=float)


def compare(abaqus_path: Path, sfc_path: Path, out_path: Path) -> list[Row]:
    abaqus = _read_disp(abaqus_path)
    sfc = _read_disp(sfc_path)
    if abaqus.shape != sfc.shape:
        raise ValueError("Abaqus and SFC displacement arrays have different shapes")
    diff = sfc - abaqus
    rows: list[Row] = []
    for idx in range(abaqus.shape[0]):
        abs_error = float(np.linalg.norm(diff[idx]))
        ref_norm = float(np.linalg.norm(abaqus[idx]))
        rows.append(
            {
                "node_id": idx + 1,
                "abaqus_norm": ref_norm,
                "sfc_norm": float(np.linalg.norm(sfc[idx])),
                "abs_error": abs_error,
                "rel_error": abs_error / max(ref_norm, 1.0e-14),
            }
        )
    rows.append(
        {
            "node_id": "max",
            "abaqus_norm": float(np.max(np.linalg.norm(abaqus, axis=1))),
            "sfc_norm": float(np.max(np.linalg.norm(sfc, axis=1))),
            "abs_error": float(np.max(np.linalg.norm(diff, axis=1))),
            "rel_error": float(np.linalg.norm(diff) / max(np.linalg.norm(abaqus), 1.0e-14)),
        }
    )
    _write_csv(out_path, rows)
    return rows


def write_summary(path: Path, rows: list[Row], runtime: Row) -> None:
    final = rows[-1]
    text = [
        "# Abaqus Finite-Rotation Hub MPC Alignment",
        "",
        "A single C3D4 hub is tied to an RP with Abaqus `*MPC, BEAM` and compared with SFC finite-rotation hub kinematics.",
        "",
        f"- Abaqus analysis wall time: {float(runtime.get('analysis_wall_seconds', 0.0)):.6f} s",
        f"- Abaqus export wall time: {float(runtime.get('export_wall_seconds', 0.0)):.6f} s",
        f"- max nodal displacement abs error: {float(final['abs_error']):.6e}",
        f"- global relative displacement error: {100.0 * float(final['rel_error']):.6f}%",
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
    abaqus_path, runtime = run_abaqus(case, out_dir, abaqus_command=args.abaqus_command, timeout=int(args.timeout))
    sfc_path = sfc_displacements(case, out_dir / "sfc_hub_displacements.csv")
    rows = compare(abaqus_path, sfc_path, out_dir / "abaqus_vs_sfc_hub_displacement_errors.csv")
    _write_csv(out_dir / "solver_runtime.csv", [runtime])
    summary = out_dir / "abaqus_finite_rotation_hub_mpc_summary.md"
    write_summary(summary, rows, runtime)
    print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
