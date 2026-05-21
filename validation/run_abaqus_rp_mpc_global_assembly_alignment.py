"""Abaqus BEAM-MPC vs SFC reduced global RP-MPC assembly.

This layer validates that BEAM hub constraints are not only available as
prescribed nodal motion helpers, but can enter the SFC global assembly through
a reduced transformation ``u_full = T q``.  The Abaqus reference is the same
small C3D4 hub tied to an RP with ``*MPC, BEAM`` and loaded by a moment on RP
dof 6.  SFC assembles the full nodal mass matrix, projects it as
``M_r = T.T M T``, fixes RP dofs 1--5, and integrates the remaining rotational
DOF under the same torque.

Abaqus is used only as an external reference and is not imported by SFC core.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
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
from validation.run_abaqus_rp_torque_dynamics_alignment import (  # noqa: E402
    RPTorqueCase,
    build_case,
    compare,
    run_abaqus,
)

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "abaqus_rp_mpc_global_assembly_alignment"


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


def _assemble_full_mass(case: RPTorqueCase):
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


def run_sfc_global_assembly(case: RPTorqueCase, out_path: Path) -> tuple[Path, Row]:
    """Run the SFC reduced global RP-MPC mass projection path."""

    mass = _assemble_full_mass(case)
    hub = RigidHubMPC(np.arange(case.nodes.shape[0], dtype=np.int64), case.nodes, case.rp)
    assembly = build_rigid_hub_reduced_assembly(case.nodes, [hub], include_free_nodes=False)
    start = time.perf_counter()
    reduced_mass = assembly.reduce_matrix(mass).toarray()
    torque = np.zeros(assembly.n_reduced_dofs, dtype=float)
    torque[assembly.hub_slice(0).start + 5] = float(case.torque[2])
    active = np.asarray([assembly.hub_slice(0).start + 5], dtype=np.int64)
    angular_acceleration = float(np.linalg.solve(reduced_mass[active[:, None], active], torque[active])[0])
    steps = int(round(float(case.duration) / float(case.dt)))
    times = np.linspace(0.0, float(case.duration), steps + 1)
    rotation_z = 0.5 * angular_acceleration * times * times
    angular_velocity_z = angular_acceleration * times
    wall = time.perf_counter() - start
    rows = [
        {"time": float(t), "rotation_z": float(r), "angular_velocity_z": float(w)}
        for t, r, w in zip(times, rotation_z, angular_velocity_z, strict=True)
    ]
    _write_csv(out_path, rows)
    return out_path, {
        "solver": "sfc_reduced_global_rp_mpc",
        "analysis_wall_seconds": wall,
        "reduced_dofs": int(assembly.n_reduced_dofs),
        "full_dofs": int(assembly.n_full_dofs),
        "reduced_mass_rz_rz": float(reduced_mass[active[0], active[0]]),
        "angular_acceleration_z": angular_acceleration,
    }


def write_summary(path: Path, rows: list[Row], abaqus_runtime: Row, sfc_runtime: Row) -> None:
    final = rows[-1]
    text = [
        "# Abaqus RP-MPC Global Assembly Alignment",
        "",
        "A C3D4 hub is tied to an RP with Abaqus `*MPC, BEAM`. SFC assembles the full nodal mass matrix, projects it with the sparse RP-MPC transformation `T`, and integrates the reduced RP rotational DOF.",
        "",
        f"- Abaqus wall time: {float(abaqus_runtime.get('analysis_wall_seconds', 0.0)):.6f} s",
        f"- SFC reduced-global wall time: {float(sfc_runtime.get('analysis_wall_seconds', 0.0)):.6e} s",
        f"- full DOFs: {int(sfc_runtime.get('full_dofs', 0))}",
        f"- reduced DOFs: {int(sfc_runtime.get('reduced_dofs', 0))}",
        f"- reduced M_rzrz: {float(sfc_runtime.get('reduced_mass_rz_rz', 0.0)):.12e}",
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
    sfc_path, sfc_runtime = run_sfc_global_assembly(case, out_dir / "sfc_rp_mpc_global_assembly_history.csv")
    rows = compare(abaqus_path, sfc_path, out_dir / "abaqus_vs_sfc_rp_mpc_global_assembly_errors.csv")
    _write_csv(out_dir / "solver_runtime.csv", [abaqus_runtime, sfc_runtime])
    summary = out_dir / "abaqus_rp_mpc_global_assembly_summary.md"
    write_summary(summary, rows, abaqus_runtime, sfc_runtime)
    print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
