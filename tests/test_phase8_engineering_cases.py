from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("skfem")

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_phase8_engineering_cases.py"

EXPECTED_FILES = {
    "phase8_cantilever_external.csv",
    "phase8_rigid_indenter_history.csv",
    "phase8_rigid_indenter_stress_cloud.csv",
    "phase8_deformable_deformable_history.csv",
    "phase8_deformable_deformable_stress_cloud.csv",
    "phase8_claims.csv",
    "phase8_plots.csv",
    "phase8_summary.md",
    "phase8_cantilever_external_stress_3d.png",
    "phase8_cantilever_external_stress_3d.pdf",
    "phase8_rigid_indenter_force_history.png",
    "phase8_rigid_indenter_force_history.pdf",
    "phase8_rigid_indenter_stress_3d.png",
    "phase8_rigid_indenter_stress_3d.pdf",
    "phase8_deformable_deformable_history.png",
    "phase8_deformable_deformable_history.pdf",
    "phase8_deformable_deformable_stress_3d.png",
    "phase8_deformable_deformable_stress_3d.pdf",
}


@pytest.fixture(scope="session")
def phase8_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("phase8-engineering")
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


def test_phase8_quick_mode_writes_expected_outputs(phase8_output: Path) -> None:
    produced = {path.name for path in phase8_output.iterdir() if path.is_file()}

    assert EXPECTED_FILES <= produced
    assert all((phase8_output / name).stat().st_size > 0 for name in EXPECTED_FILES)


def test_phase8_cantilever_external_fem_matches(phase8_output: Path) -> None:
    rows = _rows(phase8_output / "phase8_cantilever_external.csv")

    assert len(rows) >= 2
    assert all(row["external_solver"] == "scikit-fem" for row in rows)
    assert all(row["status"] == "ok" for row in rows)
    assert max(float(row["displacement_l2_rel_error"]) for row in rows) < 1.0e-8
    assert max(float(row["stiffness_fro_rel_error"]) for row in rows) < 1.0e-8
    assert max(float(row["stress_l2_rel_error"]) for row in rows) < 1.0e-8
    assert max(float(row["von_mises_l2_rel_error"]) for row in rows) < 1.0e-8


def test_phase8_rigid_indenter_produces_repulsive_downward_load(phase8_output: Path) -> None:
    rows = _rows(phase8_output / "phase8_rigid_indenter_history.csv")
    positive = [row for row in rows if float(row["indentation"]) > 0.0]

    assert positive
    assert all(row["status"] == "ok" for row in rows)
    assert all(int(row["active_contact_count"]) > 0 for row in positive)
    assert all(float(row["normal_force_z"]) < 0.0 for row in positive)
    force_magnitudes = [abs(float(row["normal_force_z"])) for row in rows]
    assert force_magnitudes == sorted(force_magnitudes)
    assert all(float(row["max_von_mises"]) >= 0.0 for row in rows)


def test_phase8_deformable_deformable_action_reaction_and_stress(phase8_output: Path) -> None:
    rows = _rows(phase8_output / "phase8_deformable_deformable_history.csv")
    final = rows[-1]

    assert all(row["status"] == "ok" for row in rows)
    assert int(final["active_contact_count"]) > 0
    assert float(final["upper_normal_force_z"]) > 0.0
    assert float(final["lower_normal_force_z"]) < 0.0
    assert max(float(row["action_reaction_imbalance"]) for row in rows) < 1.0e-8
    assert max(float(row["max_von_mises_lower"]) for row in rows) >= 0.0
    assert max(float(row["max_von_mises_upper"]) for row in rows) >= 0.0

    stress_rows = _rows(phase8_output / "phase8_deformable_deformable_stress_cloud.csv")
    assert {row["body"] for row in stress_rows} == {"lower", "upper"}
    assert all(float(row["von_mises"]) >= 0.0 for row in stress_rows)


def test_phase8_claim_markers_have_backing_csv_fields(phase8_output: Path) -> None:
    marker = re.compile(r"<!--\s*evidence\s+csv=(?P<csv>\S+)\s+field=(?P<field>\S+)\s*-->")
    text = (phase8_output / "phase8_summary.md").read_text(encoding="utf-8")
    matches = list(marker.finditer(text))

    assert matches
    for match in matches:
        csv_path = phase8_output / match.group("csv")
        assert csv_path.is_file()
        with csv_path.open(newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames or []
        assert match.group("field") in fieldnames


def test_phase8_claims_are_supported_in_quick_mode(phase8_output: Path) -> None:
    rows = _rows(phase8_output / "phase8_claims.csv")

    assert {row["claim_status"] for row in rows} == {"supported"}
