"""Gate source-gear Abaqus/SFC secondary-line alignment evidence.

This checker does not run Abaqus or SFC.  It validates already generated
alignment CSV files and records whether the timestep-matched secondary-line
surface-to-surface contact path satisfies the current short-window acceptance
thresholds for displacement, stress, and strain.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

Row = dict[str, Any]


DEFAULT_RESULT_DIR = Path("results/source_gear_secondary_line_dt1e5_probe_0002")


def _read_rows(path: Path) -> list[Row]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: Row, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def check_gate(
    *,
    errors_csv: Path,
    summary_csv: Path | None,
    max_displacement_rel: float,
    max_stress_rel: float,
    max_strain_rel: float,
    expected_projection: str,
    expected_direction: str,
    expected_averaging: str,
    expected_dt: float,
) -> tuple[bool, list[Row], Row]:
    rows = _read_rows(errors_csv)
    if not rows:
        raise ValueError(f"no rows found in {errors_csv}")
    final = rows[-1]
    displacement = _float(final, "max_displacement_magnitude_rel_error")
    stress = _float(final, "p95_von_mises_nodeavg_rel_error")
    strain = _float(final, "p95_equivalent_elastic_strain_nodeavg_rel_error")
    summary: Row = {}
    if summary_csv is not None:
        summary_rows = _read_rows(summary_csv)
        if summary_rows:
            summary = summary_rows[0]
    projection_ok = str(summary.get("source_contact_projection", expected_projection)) == str(expected_projection)
    direction_ok = str(summary.get("source_contact_direction", expected_direction)) == str(expected_direction)
    averaging_ok = str(summary.get("source_contact_averaging", expected_averaging)) == str(expected_averaging)
    dt_ok = abs(_float(summary, "sfc_dt", expected_dt) - float(expected_dt)) <= max(1.0e-14, 1.0e-9 * abs(float(expected_dt)))
    checks = [
        {
            "check": "final_max_displacement_magnitude_rel_error",
            "value": displacement,
            "threshold": float(max_displacement_rel),
            "passed": bool(displacement <= float(max_displacement_rel)),
        },
        {
            "check": "final_p95_von_mises_nodeavg_rel_error",
            "value": stress,
            "threshold": float(max_stress_rel),
            "passed": bool(stress <= float(max_stress_rel)),
        },
        {
            "check": "final_p95_equivalent_elastic_strain_nodeavg_rel_error",
            "value": strain,
            "threshold": float(max_strain_rel),
            "passed": bool(strain <= float(max_strain_rel)),
        },
        {
            "check": "source_contact_projection",
            "value": str(summary.get("source_contact_projection", "")),
            "threshold": str(expected_projection),
            "passed": bool(projection_ok),
        },
        {
            "check": "source_contact_direction",
            "value": str(summary.get("source_contact_direction", "")),
            "threshold": str(expected_direction),
            "passed": bool(direction_ok),
        },
        {
            "check": "source_contact_averaging",
            "value": str(summary.get("source_contact_averaging", "")),
            "threshold": str(expected_averaging),
            "passed": bool(averaging_ok),
        },
        {
            "check": "sfc_dt",
            "value": _float(summary, "sfc_dt", expected_dt),
            "threshold": float(expected_dt),
            "passed": bool(dt_ok),
        },
    ]
    passed = all(bool(row["passed"]) for row in checks)
    report = {
        "passed": bool(passed),
        "time": _float(final, "time"),
        "max_displacement_magnitude_rel_error": displacement,
        "p95_von_mises_nodeavg_rel_error": stress,
        "p95_equivalent_elastic_strain_nodeavg_rel_error": strain,
        "errors_csv": str(errors_csv),
        "summary_csv": "" if summary_csv is None else str(summary_csv),
    }
    return passed, checks, report


def write_report(path: Path, *, checks: list[Row], report: Row) -> None:
    lines = [
        "# Source Gear Secondary-Line Alignment Gate",
        "",
        f"- passed: `{bool(report['passed'])}`",
        f"- final time: `{float(report['time']):.12g}`",
        f"- displacement rel. error: `{100.0 * float(report['max_displacement_magnitude_rel_error']):.3f}%`",
        f"- p95 von Mises rel. error: `{100.0 * float(report['p95_von_mises_nodeavg_rel_error']):.3f}%`",
        f"- p95 equivalent strain rel. error: `{100.0 * float(report['p95_equivalent_elastic_strain_nodeavg_rel_error']):.3f}%`",
        "",
        "| check | value | threshold | passed |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | `{row['value']}` | `{row['threshold']}` | `{bool(row['passed'])}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors-csv", type=Path, default=DEFAULT_RESULT_DIR / "sfc_vs_abaqus_vtk_metric_errors.csv")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_RESULT_DIR / "full_gear_lagrangian_sdf_summary.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--max-displacement-rel", type=float, default=0.10)
    parser.add_argument("--max-stress-rel", type=float, default=0.10)
    parser.add_argument("--max-strain-rel", type=float, default=0.10)
    parser.add_argument("--expected-projection", type=str, default="secondary_line")
    parser.add_argument("--expected-direction", type=str, default="secondary_average")
    parser.add_argument("--expected-averaging", type=str, default="slave_node_region_constraint")
    parser.add_argument("--expected-dt", type=float, default=1.0e-5)
    args = parser.parse_args(argv)

    passed, checks, report = check_gate(
        errors_csv=args.errors_csv,
        summary_csv=args.summary_csv,
        max_displacement_rel=float(args.max_displacement_rel),
        max_stress_rel=float(args.max_stress_rel),
        max_strain_rel=float(args.max_strain_rel),
        expected_projection=str(args.expected_projection),
        expected_direction=str(args.expected_direction),
        expected_averaging=str(args.expected_averaging),
        expected_dt=float(args.expected_dt),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "source_gear_secondary_line_alignment_gate.csv", checks)
    _write_csv(args.out_dir / "source_gear_secondary_line_alignment_gate_summary.csv", [report])
    write_report(args.out_dir / "source_gear_secondary_line_alignment_gate.md", checks=checks, report=report)
    print((args.out_dir / "source_gear_secondary_line_alignment_gate.md").read_text(encoding="utf-8"))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
