from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_phase4_paper_validation.py"

EXPECTED_FILES = {
    "phase4_mesh_resolution.csv",
    "phase4_contact_time_history.csv",
    "phase4_performance_scaling.csv",
    "phase4_claims.csv",
    "phase4_summary.md",
}


@pytest.fixture(scope="session")
def phase4_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("phase4-validation")
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


def test_phase4_quick_mode_writes_expected_outputs(phase4_output: Path) -> None:
    produced = {path.name for path in phase4_output.iterdir() if path.is_file()}

    assert EXPECTED_FILES <= produced
    assert all((phase4_output / name).stat().st_size > 0 for name in EXPECTED_FILES)


def test_phase4_mesh_resolution_uses_at_least_three_resolutions(phase4_output: Path) -> None:
    rows = _rows(phase4_output / "phase4_mesh_resolution.csv")
    for case in ("uniaxial_linear_elastic_patch", "fixed_base_gravity_elastic_block"):
        resolutions = {
            int(row["resolution"])
            for row in rows
            if row["case"] == case and row["resolution"].isdigit()
        }
        assert len(resolutions) >= 3


def test_phase4_contact_time_history_has_required_fields(phase4_output: Path) -> None:
    rows = _rows(phase4_output / "phase4_contact_time_history.csv")
    cases = {row["case"] for row in rows}
    required_fields = {
        "time",
        "min_gap",
        "max_penetration",
        "active_contact_count",
        "normal_force",
        "contact_energy",
        "action_reaction_imbalance",
    }

    assert {
        "elastic_block_pressed_against_rigid_plane",
        "deformable_deformable_contact_time_history",
    } <= cases
    assert required_fields <= set(rows[0])


def test_phase4_performance_methods_use_distinct_workloads(phase4_output: Path) -> None:
    rows = _rows(phase4_output / "phase4_performance_scaling.csv")
    compared = [
        row
        for row in rows
        if row["stage"] in {"contact_detection_end_to_end", "full_contact_pipeline_end_to_end"}
        and row["workload"] != "unavailable"
    ]

    for resolution in {row["surface_resolution"] for row in compared}:
        subset = [row for row in compared if row["surface_resolution"] == resolution]
        workloads = [row["workload"] for row in subset]
        assert len(workloads) == len(set(workloads))


def test_phase4_claims_have_backing_csv_fields(phase4_output: Path) -> None:
    marker = re.compile(r"<!--\s*evidence\s+csv=(?P<csv>\S+)\s+field=(?P<field>\S+)\s*-->")
    text = (phase4_output / "phase4_summary.md").read_text(encoding="utf-8")
    matches = list(marker.finditer(text))

    assert matches
    for match in matches:
        csv_path = phase4_output / match.group("csv")
        assert csv_path.is_file()
        with csv_path.open(newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames or []
        assert match.group("field") in fieldnames


def test_phase4_convergence_claim_requires_three_resolutions(phase4_output: Path) -> None:
    claims = _rows(phase4_output / "phase4_claims.csv")
    for claim in claims:
        if "resolution" not in claim["claim_id"] and "convergence" not in claim["claim_id"]:
            continue
        if claim["claim_status"] == "supported":
            assert int(float(claim["condition_value"])) >= 3


def test_phase4_no_acceleration_claim_when_speedup_not_greater_than_one(phase4_output: Path) -> None:
    rows = _rows(phase4_output / "phase4_performance_scaling.csv")
    for row in rows:
        if row["speedup_status"] == "no_speedup":
            assert row["speedup_vs_bruteforce"] == ""

    claims = _rows(phase4_output / "phase4_claims.csv")
    for claim in claims:
        if "acceleration" in claim["claim_id"]:
            if float(claim["condition_value"]) <= 1.0:
                assert claim["claim_status"] != "supported"
