from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.check_source_gear_secondary_line_alignment import check_gate


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_secondary_line_alignment_gate_passes_expected_theory_setup(tmp_path: Path) -> None:
    errors = tmp_path / "errors.csv"
    summary = tmp_path / "summary.csv"
    _write_csv(
        errors,
        [
            {
                "frame": 1,
                "time": 0.0002,
                "max_displacement_magnitude_rel_error": 0.025,
                "p95_von_mises_nodeavg_rel_error": 0.043,
                "p95_equivalent_elastic_strain_nodeavg_rel_error": 0.043,
            }
        ],
    )
    _write_csv(
        summary,
        [
            {
                "source_contact_projection": "secondary_line",
                "source_contact_direction": "secondary_average",
                "source_contact_averaging": "slave_node_region_constraint",
                "sfc_dt": 1.0e-5,
            }
        ],
    )

    passed, checks, report = check_gate(
        errors_csv=errors,
        summary_csv=summary,
        max_displacement_rel=0.10,
        max_stress_rel=0.10,
        max_strain_rel=0.10,
        expected_projection="secondary_line",
        expected_direction="secondary_average",
        expected_averaging="slave_node_region_constraint",
        expected_dt=1.0e-5,
    )

    assert passed
    assert all(bool(row["passed"]) for row in checks)
    assert report["p95_von_mises_nodeavg_rel_error"] == 0.043


def test_secondary_line_alignment_gate_fails_wrong_projection(tmp_path: Path) -> None:
    errors = tmp_path / "errors.csv"
    summary = tmp_path / "summary.csv"
    _write_csv(
        errors,
        [
            {
                "frame": 1,
                "time": 0.0002,
                "max_displacement_magnitude_rel_error": 0.025,
                "p95_von_mises_nodeavg_rel_error": 0.043,
                "p95_equivalent_elastic_strain_nodeavg_rel_error": 0.043,
            }
        ],
    )
    _write_csv(
        summary,
        [
            {
                "source_contact_projection": "closest_feature",
                "source_contact_direction": "secondary_average",
                "source_contact_averaging": "slave_node_region_constraint",
                "sfc_dt": 1.0e-5,
            }
        ],
    )

    passed, checks, _report = check_gate(
        errors_csv=errors,
        summary_csv=summary,
        max_displacement_rel=0.10,
        max_stress_rel=0.10,
        max_strain_rel=0.10,
        expected_projection="secondary_line",
        expected_direction="secondary_average",
        expected_averaging="slave_node_region_constraint",
        expected_dt=1.0e-5,
    )

    assert not passed
    projection = [row for row in checks if row["check"] == "source_contact_projection"][0]
    assert not bool(projection["passed"])
