from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_c3d8_contact_trajectory_validation.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _calculix_available() -> bool:
    try:
        from validation.run_c3d8_contact_trajectory_validation import calculix_available

        return calculix_available()
    except Exception:
        return False


def test_c3d8_contact_trajectory_model_suite_contains_plane_and_block_cases() -> None:
    from validation.run_c3d8_contact_trajectory_validation import build_model_suite

    models = build_model_suite(quick=True)
    cases = {model.case for model in models}

    assert cases == {"block_plane_c3d8", "block_block_c3d8"}
    assert all(model.elements.shape[1] == 8 for model in models)
    assert all(model.slave_surface for model in models)
    assert any(model.master_kind == "rigid_plane" for model in models)
    assert any(model.master_kind == "deformable_block" for model in models)


def test_c3d8_trajectory_replay_uses_true_field_contact_path() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "dynamic_surface_sdf" not in source
    assert "DynamicNarrowBandSDF" in source
    assert "field_contact_constraint_from_sample" in source


def test_c3d8_contact_trajectory_quick_outputs_replay_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "c3d8-contact-trajectory"

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
        "c3d8_contact_trajectory.csv",
        "c3d8_contact_trajectory_summary.csv",
        "c3d8_contact_trajectory_stress_cloud.csv",
        "c3d8_contact_trajectory_commands.csv",
        "c3d8_contact_trajectory_plots.csv",
        "c3d8_contact_trajectory_claims.csv",
        "c3d8_contact_trajectory_summary.md",
    ]
    for name in expected:
        path = out_dir / name
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    history = _rows(out_dir / "c3d8_contact_trajectory.csv")
    assert {row["case"] for row in history} == {"block_plane_c3d8", "block_block_c3d8"}
    assert {row["reference_source"] for row in history} == {"synthetic_calculix_like_reference"}
    assert any(int(row["sfc_active_contact_count"]) > 0 for row in history)
    assert max(float(row["sfc_master_triangle_count"]) for row in history) > 0

    claims = {row["claim"]: row for row in _rows(out_dir / "c3d8_contact_trajectory_claims.csv")}
    assert claims["c3d8_dynamic_sdf_external_trajectory_replay"]["supported"] == "false"
    assert claims["native_sfc_nonlinear_c3d8_trajectory_equivalence"]["supported"] == "false"

    for figure in [
        out_dir / "figures" / "c3d8_contact_trajectory_z_cm.png",
        out_dir / "figures" / "c3d8_contact_trajectory_gap.png",
        out_dir / "figures" / "c3d8_contact_trajectory_force_energy.png",
        out_dir / "figures" / "c3d8_contact_trajectory_stress_cloud.png",
    ]:
        assert figure.exists(), figure
        assert figure.stat().st_size > 0, figure


@pytest.mark.skipif(not _calculix_available(), reason="CalculiX/ccx is unavailable through WSL")
def test_c3d8_contact_trajectory_quick_calculix_external_replay(tmp_path: Path) -> None:
    out_dir = tmp_path / "c3d8-contact-trajectory-calculix"

    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--quick",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )

    summaries = _rows(out_dir / "c3d8_contact_trajectory_summary.csv")
    assert {row["case"] for row in summaries} == {"block_plane_c3d8", "block_block_c3d8"}
    assert {row["reference_source"] for row in summaries} == {"calculix_dat"}
    assert all(row["status"] == "external_replay" for row in summaries)
    assert all(float(row["peak_force_rel_error"]) < 1.0e-2 for row in summaries)
    assert all(float(row["peak_energy_rel_error"]) < 1.0e-2 for row in summaries)
    assert all(float(row["min_gap_rel_error"]) < 1.0e-2 for row in summaries)
    assert all(int(row["active_sfc_rows"]) > 0 for row in summaries)

    claims = {row["claim"]: row for row in _rows(out_dir / "c3d8_contact_trajectory_claims.csv")}
    assert claims["c3d8_dynamic_sdf_external_trajectory_replay"]["supported"] == "true"
    assert claims["native_sfc_nonlinear_c3d8_trajectory_equivalence"]["supported"] == "false"
