from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCKED_CASE = ROOT / "paper" / "numerical_experiments" / "block_drop_dynamic_sdf_calculix_1s"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_locked_dynamic_sdf_block_drop_artifacts_exist() -> None:
    required = [
        LOCKED_CASE / "README.md",
        LOCKED_CASE / "MANIFEST.md",
        LOCKED_CASE / "locked_thresholds.csv",
        LOCKED_CASE / "runner_summary.md",
        LOCKED_CASE / "data" / "trajectory_comparison.csv",
        LOCKED_CASE / "data" / "alignment_diagnostics.csv",
        LOCKED_CASE / "data" / "claims.csv",
        LOCKED_CASE / "data" / "plots.csv",
        LOCKED_CASE / "data" / "external_solver_commands.csv",
        LOCKED_CASE / "figures" / "calculix_drop_z_cm.png",
        LOCKED_CASE / "figures" / "calculix_drop_z_cm.pdf",
        LOCKED_CASE / "figures" / "calculix_drop_min_gap.png",
        LOCKED_CASE / "figures" / "calculix_drop_force_proxy.png",
        LOCKED_CASE / "figures" / "calculix_drop_contact_energy.png",
        LOCKED_CASE / "figures" / "calculix_drop_total_energy.png",
    ]

    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path


def test_locked_dynamic_sdf_block_drop_metrics_satisfy_thresholds() -> None:
    comparison = _rows(LOCKED_CASE / "data" / "trajectory_comparison.csv")
    thresholds = _rows(LOCKED_CASE / "locked_thresholds.csv")

    assert len(comparison) == 1
    metric_row = comparison[0]
    assert metric_row["case"] == "block_drop"
    assert metric_row["contact_mode"] == "persistent_dynamic_sdf_calculix_f2f"

    for gate in thresholds:
        metric = gate["metric"]
        actual = metric_row[metric]
        expected_actual = gate["actual"]
        operator = gate["operator"]
        threshold = gate["threshold"]
        assert gate["status"] == "pass"

        if operator == "eq":
            assert actual == expected_actual
            assert actual == threshold
        elif operator == "le":
            assert abs(float(actual) - float(expected_actual)) < 1.0e-15
            assert float(actual) <= float(threshold)
        else:  # pragma: no cover - catches malformed future threshold rows
            raise AssertionError(f"unsupported operator {operator!r} for {metric}")


def test_locked_dynamic_sdf_block_drop_claim_boundary_is_explicit() -> None:
    readme = (LOCKED_CASE / "README.md").read_text(encoding="utf-8")
    claims = {row["claim"]: row for row in _rows(LOCKED_CASE / "data" / "claims.csv")}

    assert "persistent_dynamic_sdf_calculix_f2f" in readme
    assert "does not claim arbitrary contact equivalence" in readme
    assert claims["block_plane_geometric_contact_scoped_acceptance"]["supported"] == "true"
    assert claims["block_plane_geometric_contact_calculix_comparison_available"]["supported"] == "true"
