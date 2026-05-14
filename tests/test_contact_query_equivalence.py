from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_contact_query_equivalence import run_equivalence  # noqa: E402


def test_contact_query_equivalence_quick_outputs(tmp_path: Path) -> None:
    outputs = run_equivalence(tmp_path, quick=True)

    for path in outputs.values():
        assert path.exists()

    with outputs["metrics"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["status"] == "passed"
    assert float(rows[0]["max_field_abs_error"]) <= 1.0e-8

    with outputs["claims"].open(newline="", encoding="utf-8") as f:
        claims = {row["claim"]: row for row in csv.DictReader(f)}

    assert claims["same_backend_dynamic_sdf_query_matches_analytic_plane"]["supported"] == "true"
