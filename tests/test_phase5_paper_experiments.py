from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_phase5_paper_experiments.py"

EXPECTED_FILES = {
    "phase5_metadata.csv",
    "phase5_mesh_resolution.csv",
    "phase5_contact_time_history.csv",
    "phase5_performance_scaling.csv",
    "phase5_material_space_sdf_baseline.csv",
    "phase5_external_reference_validation.csv",
    "phase5_contact_reference.csv",
    "phase5_claims.csv",
    "phase5_plots.csv",
    "phase5_summary.md",
}

EXPECTED_PLOTS = {
    "phase5_mesh_resolution_trend",
    "phase5_contact_time_history",
    "phase5_performance_scaling",
    "phase5_speedup",
    "phase5_material_space_sdf_error",
}


@pytest.fixture(scope="session")
def phase5_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("phase5-experiments")
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


def test_phase5_quick_mode_writes_expected_outputs_and_plots(phase5_output: Path) -> None:
    produced = {path.name for path in phase5_output.iterdir() if path.is_file()}

    assert EXPECTED_FILES <= produced
    assert all((phase5_output / name).stat().st_size > 0 for name in EXPECTED_FILES)
    for plot in EXPECTED_PLOTS:
        assert (phase5_output / f"{plot}.png").stat().st_size > 0
        assert (phase5_output / f"{plot}.pdf").stat().st_size > 0


def test_phase5_performance_uses_four_surface_sizes_and_repeated_stats(phase5_output: Path) -> None:
    rows = _rows(phase5_output / "phase5_performance_scaling.csv")
    sizes = {int(row["surface_resolution"]) for row in rows}
    required_fields = {"mean_seconds", "std_seconds", "min_seconds", "max_seconds", "repeats"}

    assert len(sizes) >= 4
    assert required_fields <= set(rows[0])
    timed = [row for row in rows if row["mean_seconds"]]
    assert timed
    assert all(int(row["repeats"]) >= 2 for row in timed)
    assert all(float(row["min_seconds"]) <= float(row["mean_seconds"]) <= float(row["max_seconds"]) for row in timed)


def test_phase5_metadata_records_reproducibility_context(phase5_output: Path) -> None:
    metadata = {row["key"]: row["value"] for row in _rows(phase5_output / "phase5_metadata.csv")}

    assert metadata["run_mode"] == "quick"
    assert metadata["timer"] == "time.perf_counter"
    assert "numpy_version" in metadata
    assert "scipy_version" in metadata
    assert "core_physics_scope" in metadata


def test_phase5_material_space_baseline_exists_for_stretch_and_shear(phase5_output: Path) -> None:
    rows = _rows(phase5_output / "phase5_material_space_sdf_baseline.csv")
    deformations = {row["deformation"] for row in rows}

    assert {"stretch", "shear"} <= deformations
    assert all(row["baseline_label"] == "frozen_material_space_sdf_baseline" for row in rows)
    assert max(float(row["gap_error"]) for row in rows) > 0.0
    assert max(float(row["normal_angle_error_rad"]) for row in rows if row["deformation"] == "shear") > 0.0


def test_phase5_reference_validation_outputs_analytic_and_cpp_references(phase5_output: Path) -> None:
    reference_rows = _rows(phase5_output / "phase5_external_reference_validation.csv")
    contact_rows = _rows(phase5_output / "phase5_contact_reference.csv")

    assert {row["case"] for row in reference_rows} == {"analytic_uniaxial_linear_elastic_energy"}
    assert all(float(row["relative_error"]) < 1.0e-12 for row in reference_rows)
    assert {row["case"] for row in contact_rows} == {"high_resolution_brute_force_cpp_contact_reference"}
    assert all(float(row["gap_error"]) < 1.0e-12 for row in contact_rows)
    assert all(float(row["candidate_count"]) > 0 for row in contact_rows)


def test_phase5_claim_gates_prevent_unsupported_paper_claims(phase5_output: Path) -> None:
    claims = {row["claim_id"]: row for row in _rows(phase5_output / "phase5_claims.csv")}

    assert claims["final_acceleration_claim"]["claim_status"] == "not_supported"
    assert claims["final_acceleration_claim"]["gate_value"].startswith("quick,")
    assert claims["convergence_order_claim"]["claim_status"] == "not_supported"
    assert claims["material_space_baseline_comparison"]["claim_status"] == "supported"
    assert claims["external_reference_validation"]["claim_status"] == "supported"


def test_phase5_markdown_claim_markers_have_backing_csv_fields(phase5_output: Path) -> None:
    marker = re.compile(r"<!--\s*evidence\s+csv=(?P<csv>\S+)\s+field=(?P<field>\S+)\s*-->")
    text = (phase5_output / "phase5_summary.md").read_text(encoding="utf-8")
    matches = list(marker.finditer(text))

    assert matches
    for match in matches:
        csv_path = phase5_output / match.group("csv")
        assert csv_path.is_file()
        with csv_path.open(newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames or []
        assert match.group("field") in fieldnames
