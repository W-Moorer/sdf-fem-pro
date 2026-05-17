from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_true_field_solver_timing import run_validation


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_true_field_solver_timing_quick_outputs_claim_gates(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path / "solver-timing", quick=True)

    for key in ("step_timing", "crossover", "summary"):
        path = outputs[key]
        assert path.exists(), key
        assert path.stat().st_size > 0, key

    rows = _rows(outputs["step_timing"])
    assert {row["element_type"] for row in rows} == {"tet4", "hex8"}
    assert {row["regime"] for row in rows} == {"static", "dynamic"}
    assert {row["field_path"] for row in rows} == {"DynamicNarrowBandSDF+field_contact"}
    assert {row["field_population_mode"] for row in rows} == {"required_points_exact"}
    assert {row["reference_path"] for row in rows} == {"spatial_hash_projection_timing_reference"}
    assert {row["accuracy_status"] for row in rows} == {"passed"}
    assert all(int(row["required_grid_node_count"]) < int(row["grid_node_count"]) for row in rows)
    assert all(float(row["max_gap_abs_error"]) < 1.0e-10 for row in rows)
    assert all(float(row["force_l2_rel_error"]) < 1.0e-10 for row in rows)
    assert any(row["field_faster_total_step"] == "true" for row in rows)
    assert max(float(row["step_speedup_projection_over_field"]) for row in rows) > 1.0
    assert min(float(row["query_crossover_q_star"]) for row in rows) > 0.0

    summary = outputs["summary"].read_text(encoding="utf-8")
    assert "Total-step acceleration rows passed" in summary
    assert "Does not support nonlinear FEM" in summary


def test_true_field_solver_timing_uses_projection_only_as_reference() -> None:
    source = (ROOT / "validation" / "run_true_field_solver_timing.py").read_text(encoding="utf-8")

    assert "DynamicNarrowBandSDF" in source
    assert "build_required_points" in source
    assert "compute_field_contact_constraints" in source
    assert "surface_projection_distance_kernel" in source
    assert "dynamic_surface_sdf(" not in source
