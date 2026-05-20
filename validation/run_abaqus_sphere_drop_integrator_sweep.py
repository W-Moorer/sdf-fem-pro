"""Sweep SFC implicit integration settings for the Abaqus sphere-drop case.

This validation-only script compares HHT/Newmark/Backward-Euler style SFC
settings against the existing Abaqus/Standard implicit sphere-drop reference.
It is designed to diagnose post-impact rebound height and contact impulse
alignment; it does not change the core solver or make Abaqus a dependency.
"""

from __future__ import annotations

import argparse
import csv
import sys
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

from validation.run_abaqus_sphere_drop_full_validation import (  # noqa: E402
    run_validation,
)
from validation.run_abaqus_sphere_drop_short_validation import (  # noqa: E402
    _write_csv,
)

DEFAULT_INP = ROOT / "commercial_software_comparison" / "abaqus_flexible_body_rigid_plane_implicit" / "sphere_drop" / "abaqus_run" / "sphere_drop_implicit.inp"
DEFAULT_VTK_DIR = ROOT / "commercial_software_comparison" / "abaqus_flexible_body_rigid_plane_implicit" / "sphere_drop" / "vtk"

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class SweepCase:
    name: str
    integrator: str
    hht_alpha: float
    mass_damping: float = 0.0
    adaptive_increments: bool = False
    contact_event_substeps: int = 8


DEFAULT_CASES: tuple[SweepCase, ...] = (
    SweepCase("newmark_no_damping", "newmark", 0.0),
    SweepCase("hht_tf_alpha_m005", "hht", -0.05),
    SweepCase("hht_alpha_m010", "hht", -0.10),
    SweepCase("hht_alpha_m030", "hht", -0.30),
    SweepCase("hht_md_alpha_m041421", "hht", -0.41421356237),
    SweepCase("bwe", "bwe", 0.0),
    SweepCase("hht_alpha_m030_mass12", "hht", -0.30, mass_damping=12.0),
    SweepCase("hht_alpha_m030_mass12_adaptive", "hht", -0.30, mass_damping=12.0, adaptive_increments=True),
)


def _read_csv(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _metric_map(path: Path) -> dict[str, str]:
    return {str(row["metric"]): str(row["value"]) for row in _read_csv(path)}


def _interp(rows: list[Row], key: str, time_value: float) -> float:
    times = np.asarray([float(row["time"]) for row in rows], dtype=float)
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.interp(float(time_value), times, values))


def _max_after(rows: list[Row], key: str, start_time: float) -> float:
    values = [float(row[key]) for row in rows if float(row["time"]) >= float(start_time)]
    return max(values) if values else float("nan")


def _contact_impulse(rows: list[Row]) -> float:
    usable = [row for row in rows if row.get("normal_force_z", "") != ""]
    if len(usable) < 2:
        return float("nan")
    times = np.asarray([float(row["time"]) for row in usable], dtype=float)
    forces = np.asarray([float(row["normal_force_z"]) for row in usable], dtype=float)
    return float(np.trapezoid(forces, times))


def _summarize_case(case: SweepCase, outputs: dict[str, Path]) -> Row:
    metrics = _metric_map(outputs["metrics"])
    comparison = _read_csv(outputs["comparison"])
    sfc_rows = _read_csv(outputs["sfc_history"])
    abaqus_rows = _read_csv(outputs["abaqus_history"])
    abaqus_first = float(metrics.get("abaqus_first_contact_time") or 0.0)
    sfc_first = float(metrics.get("sfc_first_contact_time") or 0.0)
    max_sfc_rebound = _max_after(sfc_rows, "z_cm", sfc_first)
    max_abaqus_rebound = _max_after(abaqus_rows, "z_cm", abaqus_first)
    return {
        "case": case.name,
        "integrator": case.integrator,
        "hht_alpha": case.hht_alpha,
        "mass_damping": case.mass_damping,
        "adaptive_increments": str(case.adaptive_increments).lower(),
        "contact_event_substeps": case.contact_event_substeps,
        "sfc_first_contact_time": metrics.get("sfc_first_contact_time", ""),
        "abaqus_first_contact_time": metrics.get("abaqus_first_contact_time", ""),
        "first_contact_time_abs_error": metrics.get("first_contact_time_abs_error", ""),
        "max_z_cm_abs_error": metrics.get("max_z_cm_abs_error", ""),
        "rms_z_cm_abs_error": metrics.get("rms_z_cm_abs_error", ""),
        "max_normal_force_z_abs_error": metrics.get("max_normal_force_z_abs_error", ""),
        "rms_normal_force_z_abs_error": metrics.get("rms_normal_force_z_abs_error", ""),
        "sfc_rebound_z_max": max_sfc_rebound,
        "abaqus_rebound_z_max": max_abaqus_rebound,
        "rebound_z_max_abs_error": abs(max_sfc_rebound - max_abaqus_rebound),
        "sfc_z_at_end": _interp(sfc_rows, "z_cm", float(comparison[-1]["time"])),
        "abaqus_z_at_end": _interp(abaqus_rows, "z_cm", float(comparison[-1]["time"])),
        "sfc_contact_impulse": _contact_impulse(sfc_rows),
        "abaqus_contact_impulse_proxy": _contact_impulse(abaqus_rows),
        "accepted_increment_count": metrics.get("accepted_increment_count", ""),
        "cutback_count": metrics.get("cutback_count", ""),
        "min_accepted_increment": metrics.get("min_accepted_increment", ""),
        "sfc_solve_wall_seconds": metrics.get("sfc_solve_wall_seconds", ""),
    }


def _write_summary(path: Path, rows: list[Row], *, duration: float, dt: float) -> None:
    best = min(rows, key=lambda row: float(row["rebound_z_max_abs_error"]))
    lines = [
        "# Abaqus Sphere Drop Integrator Sweep",
        "",
        "This sweep diagnoses which SFC implicit integration setting best matches the Abaqus/Standard post-impact rebound for the same sphere-drop input deck.",
        "",
        f"- Duration: `{duration:g} s`",
        f"- Output dt: `{dt:g} s`",
        f"- Best rebound case: `{best['case']}`",
        f"- Best rebound max-z error: `{float(best['rebound_z_max_abs_error']):.6e} m`",
        "",
        "| Case | Integrator | HHT alpha | Mass damping | Adaptive | First-contact error | Rebound max-z error | Max z error | Solve wall (s) |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {integrator} | {hht_alpha:.6g} | {mass_damping:.6g} | {adaptive_increments} | {first_contact_time_abs_error} | {rebound:.6e} | {max_z} | {wall} |".format(
                case=row["case"],
                integrator=row["integrator"],
                hht_alpha=float(row["hht_alpha"]),
                mass_damping=float(row["mass_damping"]),
                adaptive_increments=row["adaptive_increments"],
                first_contact_time_abs_error=row["first_contact_time_abs_error"],
                rebound=float(row["rebound_z_max_abs_error"]),
                max_z=row["max_z_cm_abs_error"],
                wall=row["sfc_solve_wall_seconds"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(
    out_dir: Path,
    *,
    inp: Path = DEFAULT_INP,
    vtk_dir: Path = DEFAULT_VTK_DIR,
    duration: float = 0.12,
    dt: float = 0.001,
    cases: tuple[SweepCase, ...] = DEFAULT_CASES,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    for case in cases:
        case_dir = out_dir / case.name
        outputs = run_validation(
            case_dir,
            inp=inp,
            vtk_dir=vtk_dir,
            duration=float(duration),
            dt=float(dt),
            contact_stiffness=5.0e9,
            mass_damping=float(case.mass_damping),
            damping_start_policy="contact-estimate",
            integrator=case.integrator,
            hht_alpha=float(case.hht_alpha),
            contact_integration="surface",
            quadrature_order=3,
            adaptive_increments=bool(case.adaptive_increments),
            min_increment=1.0e-6,
            contact_event_substeps=int(case.contact_event_substeps),
            max_newton_iterations=8,
        )
        row = _summarize_case(case, outputs)
        row["result_dir"] = str(case_dir.relative_to(out_dir))
        rows.append(row)
    summary_csv = out_dir / "integrator_sweep_summary.csv"
    summary_md = out_dir / "integrator_sweep_summary.md"
    _write_csv(summary_csv, rows)
    _write_summary(summary_md, rows, duration=float(duration), dt=float(dt))
    return {"summary_csv": summary_csv, "summary_md": summary_md}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inp", type=Path, default=DEFAULT_INP)
    parser.add_argument("--vtk-dir", type=Path, default=DEFAULT_VTK_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "abaqus_sphere_drop_integrator_sweep")
    parser.add_argument("--duration", type=float, default=0.12)
    parser.add_argument("--dt", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_sweep(
        args.out_dir,
        inp=args.inp,
        vtk_dir=args.vtk_dir,
        duration=float(args.duration),
        dt=float(args.dt),
    )
    print("Abaqus sphere-drop integrator sweep complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
