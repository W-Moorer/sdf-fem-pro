"""Export dynamic SFC/CalculiX C3D8 trajectories as numbered VTK frame series.

The script is visualization-only. It does not change the contact method or any
solver state. Existing CalculiX ``.dat`` files are parsed when available; SFC
native frames are regenerated from the same deterministic C3D8 models.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_c3d8_contact_trajectory_validation import (  # noqa: E402
    _parse_element_stress,
    _parse_nodal_vectors,
    build_c3d8_model,
)
from validation.run_phase9_full_contact_validation import (  # noqa: E402
    _native_c3d8_linear_dynamic_history,
    _phase9_hex_cell_fields,
)
from validation.vtk_frame_series import write_hex_frame_series  # noqa: E402

Row = dict[str, Any]


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _calculix_dat_path(source_dir: Path, case: str, resolution: int) -> Path:
    return source_dir / "c3d8_dynamic_trajectory" / "calculix_runs" / f"{case}_r{resolution}" / f"{case}_r{resolution}.dat"


def _export_sfc_case(out_dir: Path, *, case: str, resolution: int, quick: bool, frame_stride: int) -> Row:
    model = build_c3d8_model(case=case, resolution=resolution, quick=quick)
    _rows, _u_final, _vm_final, frames = _native_c3d8_linear_dynamic_history(model)
    stem = f"sfc_{case}_r{resolution}"
    outputs = write_hex_frame_series(
        out_dir / case / "sfc",
        stem=stem,
        reference_points=model.X,
        elements=model.elements,
        frames=frames,
        frame_stride=frame_stride,
        cell_scalar_fn=lambda _time, U, model=model: _phase9_hex_cell_fields(model, U),
        source_label=f"SFC native dynamic {case}",
    )
    return {
        "case": case,
        "solver": "sfc",
        "resolution": resolution,
        "source": "native_sfc_rerun",
        "frame_count": int(outputs["frame_count"]),
        "pvd": _display_path(Path(outputs["pvd"])),
        "first_frame": _display_path(Path(outputs["first_frame"])),
    }


def _export_calculix_case(
    out_dir: Path,
    *,
    source_dir: Path,
    case: str,
    resolution: int,
    quick: bool,
    frame_stride: int,
) -> Row:
    model = build_c3d8_model(case=case, resolution=resolution, quick=quick)
    dat_path = _calculix_dat_path(source_dir, case, resolution)
    if not dat_path.exists():
        return {
            "case": case,
            "solver": "calculix",
            "resolution": resolution,
            "source": _display_path(dat_path),
            "frame_count": 0,
            "pvd": "",
            "first_frame": "",
            "status": "missing_dat",
        }
    displacements = _parse_nodal_vectors(dat_path, model.node_ids, quantity="u")
    stresses = _parse_element_stress(dat_path)
    frames = [(float(time_value), U) for time_value, U in sorted(displacements.items())]
    stem = f"calculix_{case}_r{resolution}"
    outputs = write_hex_frame_series(
        out_dir / case / "calculix",
        stem=stem,
        reference_points=model.X,
        elements=model.elements,
        frames=frames,
        frame_stride=frame_stride,
        cell_scalar_fn=lambda time_value, U, model=model, stresses=stresses: _phase9_hex_cell_fields(
            model,
            U,
            stresses.get(float(time_value), {}),
        ),
        source_label=f"CalculiX dynamic {case}",
    )
    return {
        "case": case,
        "solver": "calculix",
        "resolution": resolution,
        "source": _display_path(dat_path),
        "frame_count": int(outputs["frame_count"]),
        "pvd": _display_path(Path(outputs["pvd"])),
        "first_frame": _display_path(Path(outputs["first_frame"])),
        "status": "exported",
    }


def run_export(
    out_dir: Path,
    *,
    source_dir: Path,
    cases: list[str],
    resolution: int,
    quick: bool,
    frame_stride: int,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    for case in cases:
        rows.append(_export_sfc_case(out_dir, case=case, resolution=resolution, quick=quick, frame_stride=frame_stride))
        rows.append(
            _export_calculix_case(
                out_dir,
                source_dir=source_dir,
                case=case,
                resolution=resolution,
                quick=quick,
                frame_stride=frame_stride,
            )
        )
    manifest = out_dir / "dynamic_vtk_frame_manifest.csv"
    _write_csv(manifest, rows)
    summary = out_dir / "dynamic_vtk_frame_summary.md"
    lines = [
        "# Dynamic VTK Frame Export",
        "",
        "This directory contains continuously numbered legacy VTK frame sequences and PVD time indexes for ParaView animation.",
        "",
        "| case | solver | frames | pvd |",
        "| --- | --- | ---: | --- |",
    ]
    lines.extend(f"| {row['case']} | {row['solver']} | {row['frame_count']} | `{row.get('pvd', '')}` |" for row in rows)
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"manifest": manifest, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "numerical_experiments" / "dynamic_vtk_frames")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "paper" / "numerical_experiments" / "phase9_full_contact_validation",
        help="Directory containing c3d8_dynamic_trajectory/calculix_runs.",
    )
    parser.add_argument("--case", action="append", choices=["block_plane_c3d8", "block_block_c3d8"], help="Case to export; may be repeated.")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Use quick model duration for regenerated SFC frames.")
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = args.case or ["block_plane_c3d8", "block_block_c3d8"]
    outputs = run_export(
        args.out_dir,
        source_dir=args.source_dir,
        cases=list(cases),
        resolution=int(args.resolution),
        quick=bool(args.quick),
        frame_stride=int(args.frame_stride),
    )
    print("Dynamic VTK frame export complete.")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
