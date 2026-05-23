"""Region-aware SFC/Abaqus VTK field comparison for source-gear validation.

This post-processing utility reads already exported legacy ASCII VTK files. It
does not read Abaqus ODB data and does not run either solver.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Row = dict[str, Any]

FIELD_PAIRS = (
    ("displacement_magnitude", "displacement_magnitude", "displacement_magnitude"),
    ("von_mises_nodeavg", "von_mises_nodeavg", "von_mises_nodeavg"),
    ("strain_norm_nodeavg", "logarithmic_strain_norm_nodeavg", "strain_norm_nodeavg"),
    ("equivalent_elastic_strain_nodeavg", "equivalent_elastic_strain_nodeavg", "equivalent_elastic_strain_nodeavg"),
)
CONTACT_FIELDS = (
    "contact_pressure_nodeavg",
    "contact_penetration_nodeavg",
    "contact_active_node",
)
REQUIRED_POINT_SCALARS = tuple(sorted({name for pair in FIELD_PAIRS for name in pair[:2]} | set(CONTACT_FIELDS)))


@dataclass(frozen=True, slots=True)
class VTKPointScalars:
    point_count: int
    scalars: dict[str, np.ndarray]


def _read_numeric_values(lines: list[str], start: int, count: int) -> tuple[np.ndarray, int]:
    values: list[float] = []
    index = start
    while len(values) < count and index < len(lines):
        for part in lines[index].split():
            values.append(float(part))
            if len(values) == count:
                break
        index += 1
    if len(values) != count:
        raise ValueError(f"expected {count} numeric values, found {len(values)}")
    return np.asarray(values, dtype=float), index


def read_vtk_point_scalars(path: Path, *, names: tuple[str, ...] = REQUIRED_POINT_SCALARS) -> VTKPointScalars:
    """Read selected point scalar arrays from a legacy ASCII VTK file."""

    wanted = set(names)
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    point_count = 0
    scalars: dict[str, np.ndarray] = {}
    mode: str | None = None
    index = 0
    while index < len(lines):
        parts = lines[index].split()
        if not parts:
            index += 1
            continue
        key = parts[0].upper()
        if key == "POINT_DATA":
            point_count = int(parts[1])
            mode = "point"
            index += 1
            continue
        if key == "CELL_DATA":
            mode = "cell"
            index += 1
            continue
        if key == "VECTORS" and mode == "point":
            index += 1 + point_count
            continue
        if key == "SCALARS" and mode == "point":
            name = parts[1]
            index += 1
            if index < len(lines) and lines[index].upper().startswith("LOOKUP_TABLE"):
                index += 1
            values, index = _read_numeric_values(lines, index, point_count)
            if name in wanted:
                scalars[name] = values
            continue
        if key == "TENSORS" and mode == "point":
            index += 1 + 3 * point_count
            continue
        index += 1
    if point_count <= 0:
        raise ValueError(f"VTK file has no POINT_DATA block: {path}")
    return VTKPointScalars(point_count=point_count, scalars=scalars)


def _zeros(frame: VTKPointScalars) -> np.ndarray:
    return np.zeros(frame.point_count, dtype=float)


def _contact_mask(frame: VTKPointScalars) -> np.ndarray:
    pressure = frame.scalars.get("contact_pressure_nodeavg", _zeros(frame))
    penetration = frame.scalars.get("contact_penetration_nodeavg", _zeros(frame))
    active = frame.scalars.get("contact_active_node", _zeros(frame))
    return (active > 0.5) | (pressure > 0.0) | (penetration > 0.0)


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values.astype(float), float(percentile)))


def _region_rows(
    *,
    sfc: VTKPointScalars,
    abaqus: VTKPointScalars,
    sfc_time: float,
    abaqus_time: float,
    sfc_vtk: Path,
    abaqus_vtk: Path,
) -> list[Row]:
    if sfc.point_count != abaqus.point_count:
        raise ValueError(f"point-count mismatch: SFC={sfc.point_count}, Abaqus={abaqus.point_count}")
    sfc_active = _contact_mask(sfc)
    abaqus_active = _contact_mask(abaqus)
    full = np.ones(sfc.point_count, dtype=bool)
    regions = {
        "full": full,
        "abaqus_active": abaqus_active,
        "sfc_active": sfc_active,
        "active_union": sfc_active | abaqus_active,
    }
    rows: list[Row] = []
    for region_name, mask in regions.items():
        sample_count = int(np.count_nonzero(mask))
        for sfc_name, abaqus_name, metric_name in FIELD_PAIRS:
            sfc_values = sfc.scalars.get(sfc_name)
            abaqus_values = abaqus.scalars.get(abaqus_name)
            if sfc_values is None or abaqus_values is None:
                continue
            sfc_region = sfc_values[mask]
            abaqus_region = abaqus_values[mask]
            diff = np.abs(sfc_region - abaqus_region)
            sfc_p95 = _percentile(sfc_region, 95.0)
            abaqus_p95 = _percentile(abaqus_region, 95.0)
            sfc_max = float(np.max(sfc_region)) if sfc_region.size else 0.0
            abaqus_max = float(np.max(abaqus_region)) if abaqus_region.size else 0.0
            denom_p95 = max(abs(abaqus_p95), 1.0e-30)
            denom_max = max(abs(abaqus_max), 1.0e-30)
            rows.append(
                {
                    "sfc_vtk": str(sfc_vtk),
                    "abaqus_vtk": str(abaqus_vtk),
                    "sfc_time": float(sfc_time),
                    "abaqus_time": float(abaqus_time),
                    "time_difference": float(sfc_time) - float(abaqus_time),
                    "region": region_name,
                    "sample_count": sample_count,
                    "metric": metric_name,
                    "sfc_max": sfc_max,
                    "abaqus_max": abaqus_max,
                    "max_rel_error": abs(sfc_max - abaqus_max) / denom_max,
                    "sfc_p95": sfc_p95,
                    "abaqus_p95": abaqus_p95,
                    "p95_rel_error": abs(sfc_p95 - abaqus_p95) / denom_p95,
                    "mean_abs_error": float(np.mean(diff)) if diff.size else 0.0,
                    "p95_abs_error": _percentile(diff, 95.0),
                    "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
                    "sfc_has_contact_field": int(any(name in sfc.scalars for name in CONTACT_FIELDS)),
                    "abaqus_has_contact_field": int(any(name in abaqus.scalars for name in CONTACT_FIELDS)),
                    "sfc_active_node_count": int(np.count_nonzero(sfc_active)),
                    "abaqus_active_node_count": int(np.count_nonzero(abaqus_active)),
                }
            )
    return rows


def compare_vtk_pair(*, sfc_vtk: Path, abaqus_vtk: Path, sfc_time: float, abaqus_time: float) -> list[Row]:
    """Compare one SFC/Abaqus VTK frame pair."""

    sfc = read_vtk_point_scalars(sfc_vtk)
    abaqus = read_vtk_point_scalars(abaqus_vtk)
    return _region_rows(
        sfc=sfc,
        abaqus=abaqus,
        sfc_time=sfc_time,
        abaqus_time=abaqus_time,
        sfc_vtk=sfc_vtk,
        abaqus_vtk=abaqus_vtk,
    )


def _read_manifest(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _vtk_path_from_manifest_row(manifest_path: Path, row: Row) -> Path:
    return manifest_path.parent / str(row["vtk_file"])


def _paired_manifest_rows(*, sfc_manifest: Path, abaqus_manifest: Path, time_tolerance: float) -> list[tuple[Row, Row]]:
    sfc_rows = _read_manifest(sfc_manifest)
    abaqus_rows = _read_manifest(abaqus_manifest)
    unused = set(range(len(abaqus_rows)))
    pairs: list[tuple[Row, Row]] = []
    for sfc_row in sfc_rows:
        sfc_time = float(sfc_row["time"])
        if not unused:
            break
        nearest = min(unused, key=lambda index: abs(float(abaqus_rows[index]["time"]) - sfc_time))
        diff = abs(float(abaqus_rows[nearest]["time"]) - sfc_time)
        if diff <= float(time_tolerance):
            pairs.append((sfc_row, abaqus_rows[nearest]))
            unused.remove(nearest)
    if not pairs:
        raise ValueError(
            f"no manifest frame pairs within tolerance {time_tolerance:g}: "
            f"{sfc_manifest} vs {abaqus_manifest}"
        )
    return pairs


def compare_vtk_manifests(*, sfc_manifest: Path, abaqus_manifest: Path, time_tolerance: float = 1.0e-9) -> list[Row]:
    """Compare a sequence of SFC/Abaqus VTK frames paired by time."""

    rows: list[Row] = []
    for pair_index, (sfc_row, abaqus_row) in enumerate(
        _paired_manifest_rows(
            sfc_manifest=sfc_manifest,
            abaqus_manifest=abaqus_manifest,
            time_tolerance=float(time_tolerance),
        )
    ):
        pair_rows = compare_vtk_pair(
            sfc_vtk=_vtk_path_from_manifest_row(sfc_manifest, sfc_row),
            abaqus_vtk=_vtk_path_from_manifest_row(abaqus_manifest, abaqus_row),
            sfc_time=float(sfc_row["time"]),
            abaqus_time=float(abaqus_row["time"]),
        )
        for row in pair_rows:
            row["pair_index"] = int(pair_index)
            row["sfc_frame"] = sfc_row.get("frame", "")
            row["abaqus_frame"] = abaqus_row.get("frame", "")
        rows.extend(pair_rows)
    return rows


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


def _write_curve_plot(path: Path, rows: list[Row]) -> Path | None:
    """Write a line plot of p95 relative error curves when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return None
    metrics = (
        "displacement_magnitude",
        "von_mises_nodeavg",
        "equivalent_elastic_strain_nodeavg",
    )
    regions = ("full", "abaqus_active")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
        }
    )
    fig, axes = plt.subplots(1, len(regions), figsize=(7.2, 2.8), sharey=True)
    if len(regions) == 1:
        axes = [axes]
    legend_handles = []
    legend_labels = []
    for axis, region in zip(axes, regions):
        for metric in metrics:
            series = [
                row
                for row in rows
                if str(row["region"]) == region and str(row["metric"]) == metric
            ]
            series.sort(key=lambda row: float(row["sfc_time"]))
            if not series:
                continue
            (line,) = axis.plot(
                [float(row["sfc_time"]) for row in series],
                [100.0 * float(row["p95_rel_error"]) for row in series],
                marker="o",
                linewidth=1.4,
                markersize=3.2,
                label=metric.replace("_nodeavg", "").replace("_", " "),
            )
            label = metric.replace("_nodeavg", "").replace("_", " ")
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)
        axis.set_title(region.replace("_", " "))
        axis.set_xlabel("time (s)")
        axis.grid(True, alpha=0.28)
    axes[0].set_ylabel("p95 relative error (%)")
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _latest_rows(rows: list[Row]) -> list[Row]:
    if not rows:
        return []
    latest_time = max(float(row["sfc_time"]) for row in rows)
    return [row for row in rows if abs(float(row["sfc_time"]) - latest_time) <= 1.0e-15]


def _max_p95_rows(rows: list[Row]) -> list[Row]:
    selected: dict[tuple[str, str], Row] = {}
    for row in rows:
        key = (str(row["region"]), str(row["metric"]))
        previous = selected.get(key)
        if previous is None or float(row["p95_rel_error"]) > float(previous["p95_rel_error"]):
            selected[key] = row
    return list(selected.values())


def _write_summary(path: Path, rows: list[Row], *, csv_path: Path, curve_path: Path | None = None) -> None:
    final_rows = [row for row in _latest_rows(rows) if row["region"] in {"full", "abaqus_active"}]
    max_rows = [row for row in _max_p95_rows(rows) if row["region"] in {"full", "abaqus_active"}]
    lines = [
        "# Source Gear Regional VTK Alignment",
        "",
        "This report compares already exported SFC and Abaqus legacy VTK frames.",
        "It is post-processing only: it does not read Abaqus ODB data and does not rerun either solver.",
        "",
        f"- CSV: `{csv_path.name}`",
        f"- p95 error curves: `{curve_path.name}`" if curve_path is not None else "- p95 error curves: not generated",
        f"- paired frames: `{len(set(str(row.get('pair_index', 0)) for row in rows))}`",
        "",
        "## Latest Paired Frame",
        "",
        "| region | metric | samples | p95 rel. error | max rel. error |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in final_rows:
        lines.append(
            "| {region} | {metric} | {sample_count} | {p95:.3f}% | {maxerr:.3f}% |".format(
                region=row["region"],
                metric=row["metric"],
                sample_count=int(row["sample_count"]),
                p95=100.0 * float(row["p95_rel_error"]),
                maxerr=100.0 * float(row["max_rel_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Maximum p95 Error Over Sequence",
            "",
            "| region | metric | time | samples | p95 rel. error |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in max_rows:
        lines.append(
            "| {region} | {metric} | {time:.12g} | {sample_count} | {p95:.3f}% |".format(
                region=row["region"],
                metric=row["metric"],
                time=float(row["sfc_time"]),
                sample_count=int(row["sample_count"]),
                p95=100.0 * float(row["p95_rel_error"]),
            )
        )
    if rows:
        first = rows[0]
        latest = _latest_rows(rows)[0]
        lines.extend(
            [
                "",
                "## Contact Field Availability",
                "",
                f"- SFC contact fields present: `{int(latest['sfc_has_contact_field'])}`",
                f"- Abaqus contact fields present: `{int(latest['abaqus_has_contact_field'])}`",
                f"- latest SFC active node count: `{int(latest['sfc_active_node_count'])}`",
                f"- latest Abaqus active node count: `{int(latest['abaqus_active_node_count'])}`",
                f"- latest time difference: `{float(latest['time_difference']):.12e}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfc-vtk", type=Path, default=None)
    parser.add_argument("--abaqus-vtk", type=Path, default=None)
    parser.add_argument("--sfc-time", type=float, default=None)
    parser.add_argument("--abaqus-time", type=float, default=None)
    parser.add_argument("--sfc-manifest", type=Path, default=None)
    parser.add_argument("--abaqus-manifest", type=Path, default=None)
    parser.add_argument("--time-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.sfc_manifest is not None or args.abaqus_manifest is not None:
        if args.sfc_manifest is None or args.abaqus_manifest is None:
            parser.error("--sfc-manifest and --abaqus-manifest must be provided together")
        rows = compare_vtk_manifests(
            sfc_manifest=args.sfc_manifest,
            abaqus_manifest=args.abaqus_manifest,
            time_tolerance=float(args.time_tolerance),
        )
    else:
        if args.sfc_vtk is None or args.abaqus_vtk is None or args.sfc_time is None or args.abaqus_time is None:
            parser.error("single-frame mode requires --sfc-vtk, --abaqus-vtk, --sfc-time, and --abaqus-time")
        rows = compare_vtk_pair(
            sfc_vtk=args.sfc_vtk,
            abaqus_vtk=args.abaqus_vtk,
            sfc_time=float(args.sfc_time),
            abaqus_time=float(args.abaqus_time),
        )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "source_gear_regional_vtk_errors.csv"
    summary_path = out_dir / "source_gear_regional_vtk_alignment_summary.md"
    curve_path = _write_curve_plot(out_dir / "source_gear_regional_p95_error_curves.png", rows)
    _write_csv(csv_path, rows)
    _write_summary(summary_path, rows, csv_path=csv_path, curve_path=curve_path)
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    if curve_path is not None:
        print(f"Curves: {curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
