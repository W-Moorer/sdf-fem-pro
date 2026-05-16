from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_sdf_plane_equivalence_diagnostic import run_diagnostic


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_sdf_plane_equivalence_diagnostic_outputs_backing_files(tmp_path: Path) -> None:
    outputs = run_diagnostic(
        tmp_path,
        duration=0.032,
        dt=0.008,
        resolution=1,
        initial_velocity_z=-2.0,
        stride=1,
    )

    for path in outputs.values():
        assert path.exists()

    stateless = _rows(outputs["stateless"])
    persistent = _rows(outputs["persistent"])
    summary = _rows(outputs["summary"])

    assert stateless
    assert persistent
    assert summary
    assert "dynamic FEM-induced SDF query" in outputs["markdown"].read_text(encoding="utf-8")


def test_block_drop_sdf_plane_query_matches_analytic_plane(tmp_path: Path) -> None:
    outputs = run_diagnostic(
        tmp_path,
        duration=0.032,
        dt=0.008,
        resolution=1,
        initial_velocity_z=-2.0,
        stride=1,
    )

    rows = _rows(outputs["summary"])

    assert {row["comparison"] for row in rows} == {"stateless_query", "persistent_lifecycle"}
    assert all(row["status"] == "pass" for row in rows)
    assert max(float(row["max_gap_abs_error"]) for row in rows) < 1.0e-10
    assert max(float(row["max_normal_l2_error"]) for row in rows) < 1.0e-10
    assert max(float(row["max_projection_l2_error"]) for row in rows) < 1.0e-10
    assert max(int(row["active_mismatch_count"]) for row in rows) == 0
    assert max(int(row["missing_sdf_sample_count"]) for row in rows) == 0
