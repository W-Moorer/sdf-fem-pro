from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_phase9_full_contact_validation.py"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_phase9_quick_smoke_outputs_claim_gated_matrix(tmp_path: Path) -> None:
    out_dir = tmp_path / "phase9"
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
        timeout=120,
    )

    expected = [
        "phase9_full_contact_validation.csv",
        "phase9_claim_gates.csv",
        "phase9_stress_strain_cloud.csv",
        "phase9_commands.csv",
        "phase9_plots.csv",
        "phase9_solver_timing.csv",
        "phase9_c3d4_c3d8_side_by_side.csv",
        "phase9_curved_nonplanar_contact_external.csv",
        "phase9_full_contact_validation_summary.md",
        "native_c3d8_linear_dynamic_block_plane.csv",
        "native_c3d8_linear_dynamic_block_block.csv",
        "native_c3d8_linear_dynamic_comparison.csv",
        "native_c3d8_nonlinear_static_contactenergy.csv",
        "native_c3d8_nonlinear_dynamic_block_plane.csv",
        "native_c3d8_nonlinear_dynamic_block_block.csv",
        "native_c3d8_nonlinear_dynamic_comparison.csv",
    ]
    for name in expected:
        path = out_dir / name
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    rows = _rows(out_dir / "phase9_full_contact_validation.csv")
    assert {row["case_id"] for row in rows} == {
        "c3d8_linear_static_contact",
        "c3d8_nonlinear_static_contact",
        "c3d8_linear_dynamic_block_plane_contact",
        "c3d8_linear_dynamic_block_block_contact",
        "c3d8_nonlinear_dynamic_block_plane_contact",
        "c3d8_nonlinear_dynamic_block_block_contact",
        "c3d8_curved_nonplanar_contact_replay",
    }
    by_case = {row["case_id"]: row for row in rows}
    assert by_case["c3d8_linear_static_contact"]["native_sfc_result"] == "true"
    assert by_case["c3d8_linear_static_contact"]["calculix_comparison"] == "true"
    assert by_case["c3d8_linear_dynamic_block_plane_contact"]["native_sfc_result"] == "true"
    assert by_case["c3d8_linear_dynamic_block_plane_contact"]["status"] in {
        "supported",
        "native_only_no_calculix",
        "native_external_comparison_failed",
    }
    assert by_case["c3d8_linear_dynamic_block_plane_contact"]["supports_trajectory_equivalence"] == "false"
    assert by_case["c3d8_linear_dynamic_block_block_contact"]["native_sfc_result"] == "true"
    assert by_case["c3d8_nonlinear_static_contact"]["native_sfc_result"] == "true"
    assert by_case["c3d8_nonlinear_static_contact"]["calculix_comparison"] == "true"
    assert by_case["c3d8_nonlinear_static_contact"]["status"] == "supported"
    assert by_case["c3d8_nonlinear_dynamic_block_plane_contact"]["native_sfc_result"] == "true"
    assert by_case["c3d8_nonlinear_dynamic_block_plane_contact"]["status"] in {
        "supported",
        "native_external_comparison_failed",
    }
    assert by_case["c3d8_nonlinear_dynamic_block_block_contact"]["native_sfc_result"] == "true"
    assert by_case["c3d8_nonlinear_dynamic_block_block_contact"]["status"] in {
        "supported",
        "native_external_comparison_failed",
    }
    assert by_case["c3d8_curved_nonplanar_contact_replay"]["native_sfc_result"] == "true"

    gates = _rows(out_dir / "phase9_claim_gates.csv")
    assert all(
        row["allowed"] == "false"
        for row in gates
        if row["claim"] == "efficiency"
    )
    dynamic_trajectory = [
        row
        for row in gates
        if row["case_id"] == "c3d8_linear_dynamic_block_plane_contact" and row["claim"] == "trajectory_equivalence"
    ][0]
    assert dynamic_trajectory["allowed"] == "false"
    nonlinear_static_external = [
        row
        for row in gates
        if row["case_id"] == "c3d8_nonlinear_static_contact" and row["claim"] == "external_correctness"
    ][0]
    assert nonlinear_static_external["allowed"] == "true"

    for figure in [
        out_dir / "figures" / "phase9_claim_gate_matrix.png",
        out_dir / "figures" / "phase9_contact_error_metrics.png",
    ]:
        assert figure.exists(), figure
        assert figure.stat().st_size > 0, figure

    vtk_files = list((out_dir / "vtk").glob("*.vtk"))
    side_by_side = _rows(out_dir / "phase9_c3d4_c3d8_side_by_side.csv")
    assert {"C3D4", "C3D8"} <= {row["element_type"] for row in side_by_side}
    timings = _rows(out_dir / "phase9_solver_timing.csv")
    assert {row["case_id"] for row in timings} >= {
        "c3d8_linear_dynamic_block_plane_contact",
        "c3d8_linear_dynamic_block_block_contact",
        "c3d8_nonlinear_dynamic_block_plane_contact",
        "c3d8_nonlinear_dynamic_block_block_contact",
    }

    vtk_files = list((out_dir / "vtk").glob("*.vtk"))
    assert len(vtk_files) >= 7
    assert all(path.stat().st_size > 0 for path in vtk_files)
