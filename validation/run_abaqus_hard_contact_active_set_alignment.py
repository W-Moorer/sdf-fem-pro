"""Abaqus HARD normal contact vs SFC hard-contact active set.

This is the next small alignment layer for Abaqus-style gear contact.  The
case is deliberately minimal: one vertical spring-supported contact node is
pressed into a rigid plane with ``pressure-overclosure=HARD``.  The same
one-degree-of-freedom complementarity problem is solved by
``solve_linear_hard_contact_active_set``.

Abaqus is used only as an external reference and is not imported by the core
SFC package.
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

from sfc.contact.hard_contact import solve_linear_hard_contact_from_samples  # noqa: E402
from sfc.fem.calculix_aligned import ContactSample  # noqa: E402

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "abaqus_hard_contact_active_set_alignment"


@dataclass(frozen=True, slots=True)
class HardContactCase:
    """One-DOF hard normal contact alignment case."""

    initial_gap: float = 0.10
    stiffness: float = 1000.0
    load: float = -200.0
    plane_half_width: float = 1.0


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


def build_case() -> HardContactCase:
    return HardContactCase()


def write_abaqus_deck(case: HardContactCase, path: Path) -> None:
    """Write the external Abaqus reference deck."""

    h = float(case.plane_half_width)
    lines = [
        "*Heading",
        "Hard normal contact active-set alignment",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
        "*Node",
        f"1, 0., 0., {case.initial_gap:.12e}",
        f"2, {-h:.12e}, {-h:.12e}, 0.",
        f"3, {h:.12e}, {-h:.12e}, 0.",
        f"4, {h:.12e}, {h:.12e}, 0.",
        f"5, {-h:.12e}, {h:.12e}, 0.",
        "100, 0., 0., 0.",
        "*Element, type=SPRING1, elset=SPRING_E",
        "1, 1",
        "*Spring, elset=SPRING_E",
        "3",
        f"{case.stiffness:.12e}",
        "*Element, type=R3D4, elset=PLANE_E",
        "2, 2, 3, 4, 5",
        "*Nset, nset=SLAVE_NODE",
        "1",
        "*Nset, nset=PLANE_RP",
        "100",
        "*Elset, elset=PLANE_E",
        "2",
        "*Surface, type=NODE, name=SLAVE_SURF",
        "SLAVE_NODE, 1.",
        "*Surface, type=ELEMENT, name=MASTER_SURF",
        "PLANE_E, SPOS",
        "*Rigid Body, ref node=PLANE_RP, elset=PLANE_E",
        "*Surface Interaction, name=HARD_CONTACT",
        "*Surface Behavior, pressure-overclosure=HARD",
        "*Contact Pair, interaction=HARD_CONTACT, type=NODE TO SURFACE",
        "SLAVE_SURF, MASTER_SURF",
        "*Step, name=hard_contact_static, nlgeom=YES, inc=100",
        "*Static",
        "1., 1., 1e-08, 1.",
        "*Boundary",
        "1, 1, 2, 0.",
        "PLANE_RP, 1, 6, 0.",
        "*Cload",
        f"1, 3, {case.load:.12e}",
        "*Output, field, frequency=1",
        "*Node Output, nset=SLAVE_NODE",
        "U, CF, RF",
        "*End Step",
        "",
    ]
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
    frame = step.frames[-1]
    row = {
        "u3": 0.0,
        "cf3": 0.0,
        "rf3": 0.0,
    }
    for key in ("U", "CF", "RF"):
        if key not in frame.fieldOutputs:
            continue
        for value in frame.fieldOutputs[key].values:
            if int(value.nodeLabel) == 1:
                row[key.lower() + "3"] = float(value.data[2])
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["u3", "cf3", "rf3"])
        writer.writeheader()
        writer.writerow(row)
finally:
    odb.close()
'''


def run_abaqus(case: HardContactCase, out_dir: Path, *, abaqus_command: str | None, timeout: int) -> tuple[Path, Row]:
    run_dir = out_dir / "abaqus_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    job = "hard_contact_active_set"
    for old in run_dir.glob(f"{job}.*"):
        old.unlink()
    deck = run_dir / f"{job}.inp"
    write_abaqus_deck(case, deck)
    command = _resolve_abaqus_command(abaqus_command)
    analysis_wall = _run_command([command, f"job={job}", f"input={deck.name}", "interactive"], cwd=run_dir, log_path=out_dir / "abaqus_stdout.log", timeout=timeout)
    odb = run_dir / f"{job}.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    script = run_dir / "export_hard_contact_node.py"
    script.write_text(_abaqus_export_script(), encoding="ascii")
    metrics = out_dir / "abaqus_hard_contact_node.csv"
    export_wall = _run_command([command, "python", str(script.resolve()), str(odb.resolve()), str(metrics.resolve())], cwd=run_dir, log_path=out_dir / "abaqus_export_stdout.log", timeout=timeout)
    return metrics, {"solver": "abaqus", "analysis_wall_seconds": analysis_wall, "export_wall_seconds": export_wall}


def run_sfc(case: HardContactCase, out_path: Path) -> tuple[Path, Row]:
    sample = ContactSample(
        node_ids=np.asarray([0], dtype=np.int64),
        shape_weights=np.asarray([1.0], dtype=float),
        gap=float(case.initial_gap),
        normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
        area=1.0,
        stiffness=1.0,
    )
    start = time.perf_counter()
    solution = solve_linear_hard_contact_from_samples(
        np.diag([1.0, 1.0, float(case.stiffness)]),
        np.asarray([0.0, 0.0, case.load], dtype=float),
        [sample],
        n_total_dofs=3,
    )
    wall = time.perf_counter() - start
    row = {
        "u3": float(solution.displacement[2]),
        "gap": float(solution.gaps[0]),
        "contact_multiplier": float(solution.multipliers[0]),
        "active": int(bool(solution.active[0])),
        "iterations": int(solution.iterations),
        "converged": int(bool(solution.converged)),
    }
    _write_csv(out_path, [row])
    return out_path, {"solver": "sfc_hard_contact_active_set", "analysis_wall_seconds": wall}


def _read_first(path: Path) -> Row:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows[0]


def compare(abaqus_path: Path, sfc_path: Path, out_path: Path) -> list[Row]:
    abaqus = _read_first(abaqus_path)
    sfc = _read_first(sfc_path)
    abaqus_u = float(abaqus["u3"])
    sfc_u = float(sfc["u3"])
    expected_gap = float(sfc["gap"])
    rows = [
        {
            "metric": "u3",
            "abaqus": abaqus_u,
            "sfc": sfc_u,
            "abs_error": abs(sfc_u - abaqus_u),
            "rel_error": abs(sfc_u - abaqus_u) / max(abs(abaqus_u), 1.0e-14),
        },
        {
            "metric": "gap",
            "abaqus": 0.0,
            "sfc": expected_gap,
            "abs_error": abs(expected_gap),
            "rel_error": abs(expected_gap),
        },
    ]
    _write_csv(out_path, rows)
    return rows


def write_summary(path: Path, rows: list[Row], abaqus_runtime: Row, sfc_runtime: Row) -> None:
    u_row = next(row for row in rows if row["metric"] == "u3")
    gap_row = next(row for row in rows if row["metric"] == "gap")
    text = [
        "# Abaqus HARD Contact Active-Set Alignment",
        "",
        "A one-DOF spring contact node is pressed into a rigid plane. Abaqus uses `pressure-overclosure=HARD`; SFC solves the equivalent normal complementarity problem.",
        "",
        f"- Abaqus wall time: {float(abaqus_runtime.get('analysis_wall_seconds', 0.0)):.6f} s",
        f"- SFC active-set wall time: {float(sfc_runtime.get('analysis_wall_seconds', 0.0)):.6e} s",
        f"- displacement U3 abs. error: {float(u_row['abs_error']):.6e}",
        f"- displacement U3 rel. error: {100.0 * float(u_row['rel_error']):.6f}%",
        f"- SFC residual hard-contact gap: {float(gap_row['sfc']):.6e}",
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
    sfc_path, sfc_runtime = run_sfc(case, out_dir / "sfc_hard_contact_node.csv")
    rows = compare(abaqus_path, sfc_path, out_dir / "abaqus_vs_sfc_hard_contact_errors.csv")
    _write_csv(out_dir / "solver_runtime.csv", [abaqus_runtime, sfc_runtime])
    summary = out_dir / "abaqus_hard_contact_active_set_summary.md"
    write_summary(summary, rows, abaqus_runtime, sfc_runtime)
    print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
