"""Compare source-gear SFC and Abaqus VTK manifests without rerunning solves."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flexible_gear_full_lagrangian_sdf_comparison import (  # noqa: E402
    compare_animation_manifests,
    write_animation_color_ranges,
    write_paraview_animation_setup,
)

Row = dict[str, Any]


def _read_csv_rows(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_summary(path: Path, *, errors: list[Row], artifacts: dict[str, Path]) -> None:
    final = errors[-1] if errors else {}
    lines = [
        "# Source Gear VTK Manifest Alignment",
        "",
        "This report compares the already exported SFC and Abaqus VTK animation manifests.",
        "It does not read Abaqus ODB data and does not rerun either solver.",
        "",
        f"- metric errors: `{artifacts['errors'].name}`",
        f"- metric curves: `{artifacts['curves'].name}`",
        f"- fixed color ranges: `{artifacts['ranges'].name}`",
        f"- ParaView setup script: `{artifacts['paraview'].name}`",
        "",
        "## Final Frame Errors",
        "",
        f"- displacement magnitude rel. error: {100.0 * float(final.get('max_displacement_magnitude_rel_error', 0.0)):.3f}%",
        f"- p95 node-averaged von Mises rel. error: {100.0 * float(final.get('p95_von_mises_nodeavg_rel_error', 0.0)):.3f}%",
        f"- p95 node-averaged equivalent elastic strain rel. error: {100.0 * float(final.get('p95_equivalent_elastic_strain_nodeavg_rel_error', 0.0)):.3f}%",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def align_manifests(*, sfc_manifest: Path, abaqus_manifest: Path, out_dir: Path) -> Row:
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = out_dir / "sfc_vs_abaqus_vtk_metric_errors.csv"
    curves_path = out_dir / "sfc_vs_abaqus_vtk_metric_curves.png"
    errors = compare_animation_manifests(
        sfc_manifest=sfc_manifest,
        abaqus_manifest=abaqus_manifest,
        out_csv=errors_path,
        out_png=curves_path,
    )
    ranges = write_animation_color_ranges(
        out_dir,
        sfc_manifest=sfc_manifest,
        abaqus_manifest=abaqus_manifest,
    )
    paraview = write_paraview_animation_setup(
        out_dir,
        color_ranges=ranges,
        sfc_pvd=sfc_manifest.with_name("sfc.pvd"),
        abaqus_pvd=abaqus_manifest.with_name("abaqus.pvd"),
    )
    summary_path = out_dir / "source_gear_vtk_manifest_alignment_summary.md"
    _write_summary(
        summary_path,
        errors=errors,
        artifacts={
            "errors": errors_path,
            "curves": curves_path,
            "ranges": ranges,
            "paraview": paraview,
        },
    )
    final = errors[-1] if errors else {}
    return {
        "out_dir": str(out_dir),
        "summary": str(summary_path),
        "errors": str(errors_path),
        "curves": str(curves_path),
        "ranges": str(ranges),
        "paraview": str(paraview),
        "frame_count": len(errors),
        "final_max_displacement_magnitude_rel_error": float(final.get("max_displacement_magnitude_rel_error", 0.0)),
        "final_p95_von_mises_nodeavg_rel_error": float(final.get("p95_von_mises_nodeavg_rel_error", 0.0)),
        "final_p95_equivalent_elastic_strain_nodeavg_rel_error": float(
            final.get("p95_equivalent_elastic_strain_nodeavg_rel_error", 0.0)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfc-manifest", type=Path, required=True)
    parser.add_argument("--abaqus-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = align_manifests(
        sfc_manifest=args.sfc_manifest,
        abaqus_manifest=args.abaqus_manifest,
        out_dir=args.out_dir,
    )
    print("# Source Gear VTK Manifest Alignment")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
