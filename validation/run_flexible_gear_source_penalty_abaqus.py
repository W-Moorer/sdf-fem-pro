"""Run the source flexible-gear Abaqus deck with linear penalty contact.

This is an external-validation utility.  It preserves the source gear model
geometry, RP-MPCs, boundary conditions, angular velocity, torque, and dynamic
time stepping, while converting contact to frictionless linear penalty and
exporting a strided VTK/PVD series for comparison with SFC.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.prepare_flexible_gear_source_penalty_deck import DEFAULT_SOURCE, write_source_penalty_deck
from validation.run_flexible_gear_full_lagrangian_sdf_comparison import (
    compare_animation_manifests,
    write_animation_color_ranges,
)
from validation.run_flexible_gear_explicit_sdf_comparison import parse_gear_input
from validation.run_flexible_gear_implicit_lagrangian_sdf_comparison import (
    _abaqus_wallclock_seconds,
    _resolve_abaqus_command,
    _run_command,
    _write_csv,
)

Row = dict[str, Any]


def _read_csv_rows(path: Path) -> list[Row]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_source_penalty_abaqus(
    *,
    source: Path,
    out_dir: Path,
    pressure_stiffness: float,
    frame_stride: int,
    dt: float | None,
    duration: float | None,
    min_dt: float | None,
    max_dt: float | None,
    fixed_increment: bool,
    require_source_timing: bool,
    abaqus_command: str | None,
    include_tensors: bool,
    sfc_manifest: Path | None = None,
) -> Row:
    """Run the source-derived Abaqus penalty deck and export VTK frames."""

    out_dir.mkdir(parents=True, exist_ok=True)
    model = parse_gear_input(source)
    run_dir = out_dir / "abaqus_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    job_name = "gear_contact_source_penalty"
    deck_path = out_dir / f"{job_name}.inp"
    write_source_penalty_deck(
        source,
        deck_path,
        pressure_stiffness=pressure_stiffness,
        frame_stride=frame_stride,
        dt=dt,
        duration=duration,
        min_dt=min_dt,
        max_dt=max_dt,
        fixed_increment=bool(fixed_increment),
        require_source_timing=bool(require_source_timing),
    )
    inp_path = run_dir / f"{job_name}.inp"
    for old in run_dir.glob(f"{job_name}.*"):
        old.unlink()
    inp_path.write_text(deck_path.read_text(encoding="cp936", errors="replace"), encoding="cp936", errors="replace")
    command = _resolve_abaqus_command(abaqus_command)
    analysis_wall = _run_command(
        [command, f"job={job_name}", f"input={inp_path.name}", "interactive"],
        cwd=run_dir,
        log_path=out_dir / "abaqus_analysis_stdout.log",
    )
    odb = run_dir / f"{job_name}.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    vtk_dir = out_dir / "abaqus_vtk"
    export_script = ROOT / "validation" / "abaqus_odb_to_vtk.py"
    # The Abaqus deck already controls output cadence through
    # ``*Output, frequency=frame_stride``.  Export every ODB frame here so the
    # VTK series matches the solver output instead of applying a second stride.
    vtk_export_stride = 1
    export_wall = _run_command(
        [
            command,
            "python",
            str(export_script.resolve()),
            "--odb",
            str(odb.resolve()),
            "--out-dir",
            str(vtk_dir.resolve()),
            "--stem",
            "abaqus",
            "--frame-stride",
            str(vtk_export_stride),
            "--young",
            f"{float(model.young):.16e}",
            "--poisson",
            f"{float(model.poisson):.16e}",
            "--include-tensors" if include_tensors else "--scalars-only",
        ],
        cwd=run_dir,
        log_path=out_dir / "abaqus_vtk_export_stdout.log",
    )
    manifest = vtk_dir / "abaqus_manifest.csv"
    frame_count = max(0, len(_read_csv_rows(manifest)))
    row: Row = {
        "status": "completed",
        "source": str(source),
        "abaqus_penalty_deck": str(deck_path),
        "abaqus_analysis_wall_seconds": float(analysis_wall),
        "abaqus_export_wall_seconds": float(export_wall),
        "abaqus_reported_wall_seconds": _abaqus_wallclock_seconds(run_dir / f"{job_name}.sta") or 0.0,
        "pressure_stiffness": float(pressure_stiffness),
        "young": float(model.young),
        "poisson": float(model.poisson),
        "dt_override": "" if dt is None else float(dt),
        "duration_override": "" if duration is None else float(duration),
        "min_dt_override": "" if min_dt is None else float(min_dt),
        "max_dt_override": "" if max_dt is None else float(max_dt),
        "fixed_increment": int(bool(fixed_increment)),
        "vtk_frame_stride": int(max(1, int(frame_stride))),
        "abaqus_vtk_export_frame_stride": int(vtk_export_stride),
        "abaqus_vtk_pvd": str(vtk_dir / "abaqus.pvd"),
        "abaqus_vtk_manifest": str(manifest),
        "abaqus_vtk_frame_count": int(frame_count),
    }
    if sfc_manifest is not None:
        sfc_manifest = Path(sfc_manifest)
        row["sfc_vtk_manifest"] = str(sfc_manifest)
        errors = out_dir / "sfc_vs_abaqus_penalty_vtk_metric_errors.csv"
        curves = out_dir / "sfc_vs_abaqus_penalty_vtk_metric_curves.png"
        compare_animation_manifests(
            sfc_manifest=sfc_manifest,
            abaqus_manifest=manifest,
            out_csv=errors,
            out_png=curves,
        )
        row["animation_metric_errors"] = str(errors)
        row["animation_metric_figure"] = str(curves)
        row["animation_color_ranges"] = str(
            write_animation_color_ranges(out_dir, sfc_manifest=sfc_manifest, abaqus_manifest=manifest)
        )
    else:
        row["animation_color_ranges"] = str(write_animation_color_ranges(out_dir, abaqus_manifest=manifest))
    _write_csv(out_dir / "abaqus_source_penalty_summary.csv", [row])
    write_source_penalty_summary(out_dir / "abaqus_source_penalty_summary.md", row)
    return row


def write_source_penalty_summary(path: Path, row: Row) -> None:
    """Write a concise markdown summary for the external validation run."""

    lines = [
        "# Source Gear Abaqus Penalty Run",
        "",
        "- Purpose: external validation reference for SFC Lagrangian-SDF penalty contact.",
        "- Model: source gear deck geometry, RP-MPCs, angular velocity, torque, and dynamic step are preserved.",
        "- Contact: frictionless linear penalty pressure-overclosure in Abaqus/Standard node-to-surface contact.",
        f"- Pressure stiffness: `{float(row['pressure_stiffness']):.6e}`",
        f"- Abaqus output frequency: every `{int(row['vtk_frame_stride'])}` increments",
        f"- Fixed increment: `{bool(int(row.get('fixed_increment', 0)))}`",
        f"- Dynamic timing override: initial=`{row.get('dt_override', '')}`, total=`{row.get('duration_override', '')}`, min=`{row.get('min_dt_override', '')}`, max=`{row.get('max_dt_override', '')}`",
        f"- VTK export frame stride: `{int(row.get('abaqus_vtk_export_frame_stride', 1))}`",
        f"- Abaqus analysis wall time: `{float(row['abaqus_analysis_wall_seconds']):.6f} s`",
        f"- Abaqus reported wall time: `{float(row['abaqus_reported_wall_seconds']):.6f} s`",
        f"- Abaqus VTK manifest: `{row['abaqus_vtk_manifest']}`",
        f"- Abaqus VTK frame count: `{int(row['abaqus_vtk_frame_count'])}`",
    ]
    if row.get("animation_metric_errors"):
        lines.extend(
            [
                "",
                "## SFC Comparison",
                "",
                f"- SFC VTK manifest: `{row['sfc_vtk_manifest']}`",
                f"- Metric errors: `{row['animation_metric_errors']}`",
                f"- Metric curves: `{row['animation_metric_figure']}`",
            ]
        )
    lines.extend(
        [
            "",
            "This script is validation-only and does not import Abaqus into the SFC core solver.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pressure-stiffness", type=float, default=5.0e9)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--min-dt", type=float, default=None)
    parser.add_argument("--max-dt", type=float, default=None)
    parser.add_argument(
        "--adaptive-increment",
        action="store_true",
        help="Allow Abaqus automatic cutbacks by writing independent min/max increments.",
    )
    parser.add_argument(
        "--require-source-timing",
        action="store_true",
        help="Reject dt/duration overrides that differ from the source *Dynamic row.",
    )
    parser.add_argument("--abaqus-command", type=str, default=None)
    parser.add_argument("--sfc-manifest", type=Path, default=None)
    parser.add_argument("--scalars-only", action="store_true")
    args = parser.parse_args(argv)
    row = run_source_penalty_abaqus(
        source=args.source,
        out_dir=args.out_dir,
        pressure_stiffness=float(args.pressure_stiffness),
        frame_stride=int(args.frame_stride),
        dt=args.dt,
        duration=args.duration,
        min_dt=args.min_dt,
        max_dt=args.max_dt,
        fixed_increment=not bool(args.adaptive_increment),
        require_source_timing=bool(args.require_source_timing),
        abaqus_command=args.abaqus_command,
        include_tensors=not bool(args.scalars_only),
        sfc_manifest=args.sfc_manifest,
    )
    print(Path(str(row["abaqus_vtk_manifest"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
