from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_phase7_physical_validation.py"

EXPECTED_FILES = {
    "phase7_stress_strain.csv",
    "phase7_stress_strain_patch_cloud.csv",
    "phase7_cantilever_reference.csv",
    "phase7_cantilever_stress_cloud.csv",
    "phase7_contact_reference.csv",
    "phase7_acceleration_feasibility.csv",
    "phase7_claims.csv",
    "phase7_plots.csv",
    "phase7_summary.md",
    "phase7_cantilever_stress_cloud.png",
    "phase7_cantilever_stress_cloud.pdf",
    "phase7_cantilever_stress_3d.png",
    "phase7_cantilever_stress_3d.pdf",
    "phase7_stress_strain_patch_3d.png",
    "phase7_stress_strain_patch_3d.pdf",
}


@pytest.fixture(scope="session")
def phase7_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("phase7-validation")
    subprocess.run(
        [sys.executable, str(RUNNER), "--quick", "--out-dir", str(out_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return out_dir


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_phase7_quick_mode_writes_expected_outputs(phase7_output: Path) -> None:
    produced = {path.name for path in phase7_output.iterdir() if path.is_file()}

    assert EXPECTED_FILES <= produced
    assert all((phase7_output / name).stat().st_size > 0 for name in EXPECTED_FILES)


def test_phase7_stress_strain_matches_analytic_reference(phase7_output: Path) -> None:
    rows = _rows(phase7_output / "phase7_stress_strain.csv")

    assert len(rows) >= 3
    assert all(row["status"] == "ok" for row in rows)
    assert max(float(row["max_strain_l2_error"]) for row in rows) < 1.0e-14
    assert max(float(row["max_stress_l2_error"]) for row in rows) < 1.0e-8


def test_phase7_stress_strain_patch_cloud_has_zero_error_and_figures(phase7_output: Path) -> None:
    rows = _rows(phase7_output / "phase7_stress_strain_patch_cloud.csv")
    resolutions = {int(row["resolution"]) for row in rows}

    assert len(resolutions) >= 3
    assert {row["status"] for row in rows} == {"ok"}
    assert max(float(row["von_mises_abs_error"]) for row in rows) < 1.0e-8
    assert max(float(row["engineering_strain_norm_abs_error"]) for row in rows) < 1.0e-14
    assert (phase7_output / "phase7_stress_strain_patch_3d.png").stat().st_size > 0
    assert (phase7_output / "phase7_stress_strain_patch_3d.pdf").stat().st_size > 0


def test_phase7_cantilever_reports_displacement_and_stress_trends(phase7_output: Path) -> None:
    rows = _rows(phase7_output / "phase7_cantilever_reference.csv")

    assert len(rows) >= 3
    assert all(float(row["tip_displacement_z"]) < 0.0 for row in rows)
    assert all(float(row["max_von_mises"]) > 0.0 for row in rows)
    assert rows[-1]["relative_tip_error_to_finest"] == "0.0"
    assert rows[-1]["relative_max_vm_error_to_finest"] == "0.0"


def test_phase7_cantilever_stress_cloud_has_finite_von_mises_samples(phase7_output: Path) -> None:
    rows = _rows(phase7_output / "phase7_cantilever_stress_cloud.csv")
    resolutions = {int(row["resolution"]) for row in rows}

    assert len(resolutions) >= 3
    assert all(row["status"] == "ok" for row in rows)
    assert all(float(row["von_mises"]) >= 0.0 for row in rows)
    assert (phase7_output / "phase7_cantilever_stress_cloud.png").stat().st_size > 0
    assert (phase7_output / "phase7_cantilever_stress_cloud.pdf").stat().st_size > 0
    assert (phase7_output / "phase7_cantilever_stress_3d.png").stat().st_size > 0
    assert (phase7_output / "phase7_cantilever_stress_3d.pdf").stat().st_size > 0


def test_phase7_contact_force_displacement_matches_bruteforce_cpp(phase7_output: Path) -> None:
    rows = _rows(phase7_output / "phase7_contact_reference.csv")

    assert len(rows) >= 4
    assert all(row["status"] == "ok" for row in rows)
    assert max(float(row["normal_force_rel_error"]) for row in rows) < 1.0e-12
    assert max(float(row["max_gap_abs_error"]) for row in rows) < 1.0e-12
    assert all(row["brute_active_count"] == row["dynamic_active_count"] for row in rows)


def test_phase7_acceleration_distinguishes_contact_only_and_total_step(phase7_output: Path) -> None:
    rows = _rows(phase7_output / "phase7_acceleration_feasibility.csv")
    stages = {row["stage"] for row in rows}
    methods = {row["method"] for row in rows}

    assert {"contact_pipeline_only", "fem_assembly_linear_solve_plus_contact"} <= stages
    assert {"dynamic_fem_sdf_contact", "dynamic_fem_sdf_total_step"} <= methods
    assert {"mean_seconds", "std_seconds", "min_seconds", "max_seconds", "speedup_vs_bruteforce"} <= set(rows[0])


def test_phase7_claim_markers_have_backing_csv_fields(phase7_output: Path) -> None:
    marker = re.compile(r"<!--\s*evidence\s+csv=(?P<csv>\S+)\s+field=(?P<field>\S+)\s*-->")
    text = (phase7_output / "phase7_summary.md").read_text(encoding="utf-8")
    matches = list(marker.finditer(text))

    assert matches
    for match in matches:
        csv_path = phase7_output / match.group("csv")
        assert csv_path.is_file()
        with csv_path.open(newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames or []
        assert match.group("field") in fieldnames
