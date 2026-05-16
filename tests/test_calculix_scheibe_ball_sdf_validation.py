from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_calculix_scheibe_ball_sdf_validation.py"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_scheibe_ball_quick_outputs_claim_gated_replay(tmp_path: Path) -> None:
    out_dir = tmp_path / "scheibe_ball"
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--quick",
            "--skip-calculix",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )

    expected = [
        "scheibe_ball_sdf_validation_summary.csv",
        "scheibe2f2f_contact_points.csv",
        "ball_dynamic_drop_trajectory.csv",
        "scheibe_ball_claim_gates.csv",
        "scheibe_ball_commands.csv",
        "scheibe_ball_plots.csv",
        "scheibe_ball_sdf_validation_summary.md",
        "figures/scheibe2f2f_sdf_replay_scatter.png",
        "figures/scheibe2f2f_sdf_replay_scatter.pdf",
        "figures/ball_dynamic_drop_native_trajectory.png",
        "figures/ball_dynamic_drop_native_trajectory.pdf",
    ]
    for relative in expected:
        path = out_dir / relative
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    summary = {row["case_id"]: row for row in _rows(out_dir / "scheibe_ball_sdf_validation_summary.csv")}
    assert set(summary) == {
        "scheibe2f2f_c3d8_nonlinear_static_sdf_replay",
        "ball_c3d8_dynamic_drop_native_trajectory",
    }
    scheibe = summary["scheibe2f2f_c3d8_nonlinear_static_sdf_replay"]
    assert scheibe["status"] == "supported"
    assert float(scheibe["max_gap_abs_error"]) < 2.0e-8
    assert float(scheibe["max_pressure_rel_error"]) < 1.0e-3

    contact_points = _rows(out_dir / "scheibe2f2f_contact_points.csv")
    assert len(contact_points) == 34
    assert max(float(row["pressure_rel_error"]) for row in contact_points) < 1.0e-3

    ball = summary["ball_c3d8_dynamic_drop_native_trajectory"]
    assert ball["supports_efficiency"] == "false"
    assert ball["supports_native_trajectory_equivalence"] == "false"
    ball_history = _rows(out_dir / "ball_dynamic_drop_trajectory.csv")
    assert len(ball_history) >= 3
    assert max(float(row["sfc_x_cm"]) for row in ball_history if row["sfc_x_cm"]) > 0.0

    gates = _rows(out_dir / "scheibe_ball_claim_gates.csv")
    by_gate = {(row["case_id"], row["claim"]): row for row in gates}
    assert by_gate[
        ("scheibe2f2f_c3d8_nonlinear_static_sdf_replay", "deforming_sdf_gap_matches_calculix_copen")
    ]["allowed"] == "true"
    assert by_gate[
        ("scheibe2f2f_c3d8_nonlinear_static_sdf_replay", "native_sfc_trajectory_equivalence")
    ]["allowed"] == "false"
    assert by_gate[("ball_c3d8_dynamic_drop_native_trajectory", "efficiency")]["allowed"] == "false"
