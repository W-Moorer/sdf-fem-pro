from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_deforming_master_sdf_reference import run_reference  # noqa: E402


def test_deforming_master_sdf_reference_quick_outputs(tmp_path: Path) -> None:
    outputs = run_reference(tmp_path, quick=True)

    for path in outputs.values():
        assert path.exists()

    with outputs["queries"].open(newline="", encoding="utf-8") as f:
        query_rows = list(csv.DictReader(f))
    with outputs["forces"].open(newline="", encoding="utf-8") as f:
        force_rows = list(csv.DictReader(f))
    with outputs["baseline"].open(newline="", encoding="utf-8") as f:
        baseline_rows = list(csv.DictReader(f))
    with outputs["claims"].open(newline="", encoding="utf-8") as f:
        claims = {row["claim"]: row for row in csv.DictReader(f)}

    assert query_rows
    assert force_rows
    assert baseline_rows
    assert all(row["status"] == "passed" for row in query_rows)
    assert all(row["status"] == "passed" for row in force_rows)
    assert max(float(row["gap_abs_difference"]) for row in baseline_rows) > 1.0e-4
    assert max(float(row["normal_angle_difference"]) for row in baseline_rows) > 1.0e-3
    assert claims["broad_phase_dynamic_sdf_matches_bruteforce_on_deforming_master"]["supported"] == "true"
    assert claims["dynamic_sdf_contact_force_matches_bruteforce_reference"]["supported"] == "true"
    assert claims["frozen_reference_sdf_differs_from_current_deforming_master_sdf"]["supported"] == "true"
