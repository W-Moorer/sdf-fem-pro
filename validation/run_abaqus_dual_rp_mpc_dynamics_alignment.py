"""Abaqus dual BEAM-MPC RP dynamics vs SFC reduced global assembly.

This layer validates the two-body reduced RP path needed before full gear
contact is assembled.  Two independent C3D4 flexible hubs are each tied to an
RP with Abaqus ``*MPC, BEAM``.  RP1 receives a prescribed angular velocity and
RP2 receives a constant moment.  SFC assembles both bodies into one full nodal
mass matrix, builds one sparse reduced transformation ``u_full = T q`` with
two RP blocks, projects the mass matrix, and advances the prescribed/free RP
rotational histories in reduced coordinates.

There is intentionally no contact in this case.  Abaqus is used only as an
external reference; SFC core does not import Abaqus modules.
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
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.fem.rp_mpc import RigidHubMPC, build_rigid_hub_reduced_assembly  # noqa: E402
from sfc.fem.tet4 import tet4_lumped_mass  # noqa: E402

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "abaqus_dual_rp_mpc_dynamics_alignment"


@dataclass(frozen=True, slots=True)
class DualRPDynamicsCase:
    nodes: np.ndarray
    elements: np.ndarray
    rp1: np.ndarray
    rp2: np.ndarray
    rp1_angular_velocity: np.ndarray
    rp2_torque: np.ndarray
    duration: float = 0.02
    dt: float = 0.002
    young: float = 1.0e7
    poisson: float = 0.3
    density: float = 1000.0


def build_case() -> DualRPDynamicsCase:
    body1 = np.asarray(
        [
            [0.10, -0.18, -0.12],
            [0.72, 0.06, -0.08],
            [0.18, 0.44, 0.04],
            [0.22, -0.02, 0.52],
        ],
        dtype=float,
    )
    body2 = np.asarray(
        [
            [1.20, -0.16, -0.10],
            [1.84, 0.03, -0.05],
            [1.28, 0.46, 0.06],
            [1.32, -0.04, 0.55],
        ],
        dtype=float,
    )
    return DualRPDynamicsCase(
        nodes=np.vstack([body1, body2]),
        elements=np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64),
        rp1=np.asarray([0.18, -0.03, 0.02], dtype=float),
        rp2=np.asarray([1.28, -0.02, 0.03], dtype=float),
        rp1_angular_velocity=np.asarray([0.0, 0.0, 5.236], dtype=float),
        rp2_torque=np.asarray([0.0, 0.0, 0.015], dtype=float),
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


def write_abaqus_deck(case: DualRPDynamicsCase, path: Path) -> None:
    min_dt = min(float(case.dt) * 1.0e-4, 1.0e-8)
    lines = [
        "*Heading",
        "Dual RP BEAM-MPC dynamics alignment",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
        "*Node",
    ]
    for idx, xyz in enumerate(case.nodes, start=1):
        lines.append(f"{idx}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    lines.append(f"100, {case.rp1[0]:.12e}, {case.rp1[1]:.12e}, {case.rp1[2]:.12e}")
    lines.append(f"200, {case.rp2[0]:.12e}, {case.rp2[1]:.12e}, {case.rp2[2]:.12e}")
    lines.extend(
        [
            "*Element, type=C3D4, elset=EALL",
            "1, 1, 2, 3, 4",
            "2, 5, 6, 7, 8",
            "*Nset, nset=HUB1",
            "1, 2, 3, 4",
            "*Nset, nset=HUB2",
            "5, 6, 7, 8",
            "*Nset, nset=RP1",
            "100",
            "*Nset, nset=RP2",
            "200",
            "*MPC",
            "BEAM, HUB1, RP1",
            "BEAM, HUB2, RP2",
            "*Material, name=MAT",
            "*Density",
            f"{case.density:.12e}",
            "*Elastic",
            f"{case.young:.12e}, {case.poisson:.12e}",
            "*Solid Section, elset=EALL, material=MAT",
            ",",
            "*Step, name=dual_rp_dynamic, nlgeom=YES, inc=1000",
            "*Dynamic",
            f"{case.dt:.12e}, {case.duration:.12e}, {min_dt:.12e}, {case.dt:.12e}",
            "*Boundary",
            "RP1, 1, 5, 0.",
            "RP2, 1, 5, 0.",
            "*Boundary, type=VELOCITY",
            f"RP1, 6, 6, {case.rp1_angular_velocity[2]:.12e}",
            "*Cload",
            f"RP2, 6, {case.rp2_torque[2]:.12e}",
            "*Output, field, frequency=1",
            "*Node Output, nset=RP1",
            "U, UR, V, VR, A, AR",
            "*Node Output, nset=RP2",
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
        row = {
            "time": float(frame.frameValue),
            "rp1_rotation_z": 0.0,
            "rp1_angular_velocity_z": 0.0,
            "rp2_rotation_z": 0.0,
            "rp2_angular_velocity_z": 0.0,
        }
        if "UR" in frame.fieldOutputs:
            for value in frame.fieldOutputs["UR"].values:
                label = int(value.nodeLabel)
                if label == 100:
                    row["rp1_rotation_z"] = float(value.data[2])
                elif label == 200:
                    row["rp2_rotation_z"] = float(value.data[2])
        if "VR" in frame.fieldOutputs:
            for value in frame.fieldOutputs["VR"].values:
                label = int(value.nodeLabel)
                if label == 100:
                    row["rp1_angular_velocity_z"] = float(value.data[2])
                elif label == 200:
                    row["rp2_angular_velocity_z"] = float(value.data[2])
        rows.append(row)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time",
                "rp1_rotation_z",
                "rp1_angular_velocity_z",
                "rp2_rotation_z",
                "rp2_angular_velocity_z",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
finally:
    odb.close()
'''


def run_abaqus(case: DualRPDynamicsCase, out_dir: Path, *, abaqus_command: str | None, timeout: int) -> tuple[Path, Row]:
    run_dir = out_dir / "abaqus_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    job = "dual_rp_mpc_dynamics"
    for old in run_dir.glob(f"{job}.*"):
        old.unlink()
    deck = run_dir / f"{job}.inp"
    write_abaqus_deck(case, deck)
    command = _resolve_abaqus_command(abaqus_command)
    analysis_wall = _run_command([command, f"job={job}", f"input={deck.name}", "interactive"], cwd=run_dir, log_path=out_dir / "abaqus_stdout.log", timeout=timeout)
    odb = run_dir / f"{job}.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    script = run_dir / "export_dual_rp.py"
    script.write_text(_abaqus_export_script(), encoding="ascii")
    metrics = out_dir / "abaqus_dual_rp_history.csv"
    export_wall = _run_command([command, "python", str(script.resolve()), str(odb.resolve()), str(metrics.resolve())], cwd=run_dir, log_path=out_dir / "abaqus_export_stdout.log", timeout=timeout)
    return metrics, {"solver": "abaqus", "analysis_wall_seconds": analysis_wall, "export_wall_seconds": export_wall}


def _assemble_full_mass(case: DualRPDynamicsCase):
    rows_idx: list[np.ndarray] = []
    cols_idx: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for element in case.elements:
        local_mass = tet4_lumped_mass(case.nodes[element], case.density)
        dofs = np.repeat(element, 3) * 3 + np.tile(np.arange(3, dtype=np.int64), element.size)
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows_idx.append(rr.ravel())
        cols_idx.append(cc.ravel())
        data.append(local_mass.ravel())
    return coo_matrix(
        (np.concatenate(data), (np.concatenate(rows_idx), np.concatenate(cols_idx))),
        shape=(3 * case.nodes.shape[0], 3 * case.nodes.shape[0]),
    ).tocsr()


def run_sfc_dual_reduced(case: DualRPDynamicsCase, out_path: Path) -> tuple[Path, Row]:
    """Run SFC with both RP hubs in one reduced global assembly."""

    mass = _assemble_full_mass(case)
    hub1 = RigidHubMPC(np.asarray([0, 1, 2, 3], dtype=np.int64), case.nodes, case.rp1)
    hub2 = RigidHubMPC(np.asarray([4, 5, 6, 7], dtype=np.int64), case.nodes, case.rp2)
    assembly = build_rigid_hub_reduced_assembly(case.nodes, [hub1, hub2], include_free_nodes=False)
    start = time.perf_counter()
    reduced_mass = assembly.reduce_matrix(mass).toarray()
    rp1_z = assembly.hub_slice(0).start + 5
    rp2_z = assembly.hub_slice(1).start + 5
    alpha2 = float(np.linalg.solve(reduced_mass[np.ix_([rp2_z], [rp2_z])], np.asarray([case.rp2_torque[2]], dtype=float))[0])
    steps = int(round(float(case.duration) / float(case.dt)))
    times = np.linspace(0.0, float(case.duration), steps + 1)
    rp1_rotation_z = float(case.rp1_angular_velocity[2]) * times
    rp1_angular_velocity_z = np.full_like(times, float(case.rp1_angular_velocity[2]))
    rp2_rotation_z = 0.5 * alpha2 * times * times
    rp2_angular_velocity_z = alpha2 * times
    wall = time.perf_counter() - start
    rows = [
        {
            "time": float(t),
            "rp1_rotation_z": float(r1),
            "rp1_angular_velocity_z": float(w1),
            "rp2_rotation_z": float(r2),
            "rp2_angular_velocity_z": float(w2),
        }
        for t, r1, w1, r2, w2 in zip(times, rp1_rotation_z, rp1_angular_velocity_z, rp2_rotation_z, rp2_angular_velocity_z, strict=True)
    ]
    _write_csv(out_path, rows)
    return out_path, {
        "solver": "sfc_dual_reduced_global_rp_mpc",
        "analysis_wall_seconds": wall,
        "full_dofs": int(assembly.n_full_dofs),
        "reduced_dofs": int(assembly.n_reduced_dofs),
        "rp1_reduced_mass_rz_rz": float(reduced_mass[rp1_z, rp1_z]),
        "rp2_reduced_mass_rz_rz": float(reduced_mass[rp2_z, rp2_z]),
        "rp2_angular_acceleration_z": alpha2,
    }


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
        out: Row = {"time": t}
        for prefix in ("rp1", "rp2"):
            for quantity in ("rotation_z", "angular_velocity_z"):
                key = f"{prefix}_{quantity}"
                sfc_value = float(row[key])
                abaqus_value = float(np.interp(t, t_abq, _col(abq, key)))
                out[f"abaqus_{key}"] = abaqus_value
                out[f"sfc_{key}"] = sfc_value
                out[f"{key}_abs_error"] = abs(sfc_value - abaqus_value)
                out[f"{key}_rel_error"] = abs(sfc_value - abaqus_value) / max(abs(abaqus_value), 1.0e-14)
        rows.append(out)
    _write_csv(out_path, rows)
    return rows


def write_summary(path: Path, rows: list[Row], abaqus_runtime: Row, sfc_runtime: Row) -> None:
    final = rows[-1]
    text = [
        "# Abaqus Dual RP-MPC Dynamics Alignment",
        "",
        "Two C3D4 hubs are tied to two RPs with Abaqus `*MPC, BEAM`. RP1 is driven by angular velocity; RP2 is driven by torque. SFC uses one reduced global RP-MPC assembly and projected mass matrix.",
        "",
        f"- Abaqus wall time: {float(abaqus_runtime.get('analysis_wall_seconds', 0.0)):.6f} s",
        f"- SFC dual reduced-global wall time: {float(sfc_runtime.get('analysis_wall_seconds', 0.0)):.6e} s",
        f"- full DOFs: {int(sfc_runtime.get('full_dofs', 0))}",
        f"- reduced DOFs: {int(sfc_runtime.get('reduced_dofs', 0))}",
        f"- RP1 reduced M_rzrz: {float(sfc_runtime.get('rp1_reduced_mass_rz_rz', 0.0)):.12e}",
        f"- RP2 reduced M_rzrz: {float(sfc_runtime.get('rp2_reduced_mass_rz_rz', 0.0)):.12e}",
        f"- RP1 final rotation z rel. error: {100.0 * float(final.get('rp1_rotation_z_rel_error', 0.0)):.6f}%",
        f"- RP1 final angular velocity z rel. error: {100.0 * float(final.get('rp1_angular_velocity_z_rel_error', 0.0)):.6f}%",
        f"- RP2 final rotation z rel. error: {100.0 * float(final.get('rp2_rotation_z_rel_error', 0.0)):.6f}%",
        f"- RP2 final angular velocity z rel. error: {100.0 * float(final.get('rp2_angular_velocity_z_rel_error', 0.0)):.6f}%",
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
    sfc_path, sfc_runtime = run_sfc_dual_reduced(case, out_dir / "sfc_dual_rp_history.csv")
    rows = compare(abaqus_path, sfc_path, out_dir / "abaqus_vs_sfc_dual_rp_errors.csv")
    _write_csv(out_dir / "solver_runtime.csv", [abaqus_runtime, sfc_runtime])
    summary = out_dir / "abaqus_dual_rp_mpc_dynamics_summary.md"
    write_summary(summary, rows, abaqus_runtime, sfc_runtime)
    print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
