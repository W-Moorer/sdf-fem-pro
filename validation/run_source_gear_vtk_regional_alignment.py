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


def _write_summary(path: Path, rows: list[Row], *, csv_path: Path) -> None:
    final_rows = [row for row in rows if row["region"] in {"full", "abaqus_active"}]
    lines = [
        "# Source Gear Regional VTK Alignment",
        "",
        "This report compares already exported SFC and Abaqus legacy VTK frames.",
        "It is post-processing only: it does not read Abaqus ODB data and does not rerun either solver.",
        "",
        f"- CSV: `{csv_path.name}`",
        "",
        "## Key Rows",
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
    if rows:
        first = rows[0]
        lines.extend(
            [
                "",
                "## Contact Field Availability",
                "",
                f"- SFC contact fields present: `{int(first['sfc_has_contact_field'])}`",
                f"- Abaqus contact fields present: `{int(first['abaqus_has_contact_field'])}`",
                f"- SFC active node count: `{int(first['sfc_active_node_count'])}`",
                f"- Abaqus active node count: `{int(first['abaqus_active_node_count'])}`",
                f"- time difference: `{float(first['time_difference']):.12e}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfc-vtk", type=Path, required=True)
    parser.add_argument("--abaqus-vtk", type=Path, required=True)
    parser.add_argument("--sfc-time", type=float, required=True)
    parser.add_argument("--abaqus-time", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

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
    _write_csv(csv_path, rows)
    _write_summary(summary_path, rows, csv_path=csv_path)
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
