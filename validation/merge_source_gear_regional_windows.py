"""Merge source-gear regional alignment windows into continuous curves."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_source_gear_vtk_regional_alignment import (  # noqa: E402
    _write_csv,
    _write_curve_plot,
    _write_metric_curve_plot,
    _write_summary,
)

Row = dict[str, Any]


def _read_csv(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_float(value: object) -> float:
    return float(value)


def merge_regional_csvs(paths: list[Path]) -> list[Row]:
    """Return de-duplicated regional-alignment rows sorted by time."""

    merged: dict[tuple[float, str, str], Row] = {}
    for window_index, path in enumerate(paths):
        for row in _read_csv(path):
            key = (round(_as_float(row["sfc_time"]), 15), str(row["region"]), str(row["metric"]))
            row = dict(row)
            row["source_window"] = str(path)
            row["source_window_index"] = int(window_index)
            merged[key] = row
    rows = sorted(merged.values(), key=lambda row: (_as_float(row["sfc_time"]), str(row["region"]), str(row["metric"])))
    time_to_index: dict[float, int] = {}
    for row in rows:
        time_key = round(_as_float(row["sfc_time"]), 15)
        if time_key not in time_to_index:
            time_to_index[time_key] = len(time_to_index)
        row["pair_index"] = int(time_to_index[time_key])
    return rows


def write_merged_outputs(*, csv_paths: list[Path], out_dir: Path) -> dict[str, Path | int]:
    """Merge regional CSV windows and write continuous CSV, summary, and plots."""

    rows = merge_regional_csvs(csv_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "source_gear_regional_merged_errors.csv"
    error_curve_path = _write_curve_plot(out_dir / "source_gear_regional_merged_p95_error_curves.png", rows)
    metric_curve_path = _write_metric_curve_plot(out_dir / "source_gear_regional_merged_p95_metric_curves.png", rows)
    summary_path = out_dir / "source_gear_regional_merged_summary.md"
    _write_csv(csv_path, rows)
    _write_summary(summary_path, rows, csv_path=csv_path, curve_path=error_curve_path, metric_curve_path=metric_curve_path)
    return {
        "row_count": len(rows),
        "frame_count": len({round(_as_float(row["sfc_time"]), 15) for row in rows}),
        "csv": csv_path,
        "summary": summary_path,
        "error_curves": error_curve_path or Path(""),
        "metric_curves": metric_curve_path or Path(""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", dest="csv_paths", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = write_merged_outputs(csv_paths=list(args.csv_paths), out_dir=args.out_dir)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
