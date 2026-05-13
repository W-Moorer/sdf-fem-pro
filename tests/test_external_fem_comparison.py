from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("skfem")

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_external_fem_comparison.py"

EXPECTED_FILES = {
    "external_fem_comparison.csv",
    "external_fem_stress_cloud.csv",
    "external_fem_plots.csv",
    "external_fem_summary.md",
    "external_fem_error_trends.png",
    "external_fem_error_trends.pdf",
    "external_fem_stress_cloud.png",
    "external_fem_stress_cloud.pdf",
    "external_fem_stress_3d.png",
    "external_fem_stress_3d.pdf",
}


@pytest.fixture(scope="session")
def external_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("external-fem")
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


def test_external_fem_quick_mode_writes_expected_outputs(external_output: Path) -> None:
    produced = {path.name for path in external_output.iterdir() if path.is_file()}

    assert EXPECTED_FILES <= produced
    assert all((external_output / name).stat().st_size > 0 for name in EXPECTED_FILES)


def test_external_fem_displacement_stress_and_stiffness_match(external_output: Path) -> None:
    rows = _rows(external_output / "external_fem_comparison.csv")

    assert len(rows) >= 2
    assert all(row["external_solver"] == "scikit-fem" for row in rows)
    assert all(row["status"] == "ok" for row in rows)
    assert max(float(row["displacement_l2_rel_error"]) for row in rows) < 1.0e-8
    assert max(float(row["stiffness_fro_rel_error"]) for row in rows) < 1.0e-8
    assert max(float(row["stress_l2_rel_error"]) for row in rows) < 1.0e-8
    assert max(float(row["von_mises_l2_rel_error"]) for row in rows) < 1.0e-8


def test_external_fem_stress_cloud_contains_error_field(external_output: Path) -> None:
    rows = _rows(external_output / "external_fem_stress_cloud.csv")

    assert rows
    assert all(row["status"] == "ok" for row in rows)
    assert all(float(row["von_mises_sfc"]) >= 0.0 for row in rows)
    assert all(float(row["von_mises_external"]) >= 0.0 for row in rows)
    assert max(float(row["von_mises_abs_error"]) for row in rows) < 1.0e-8


def test_external_fem_summary_claim_markers_have_backing_csv_fields(external_output: Path) -> None:
    marker = re.compile(r"<!--\s*evidence\s+csv=(?P<csv>\S+)\s+field=(?P<field>\S+)\s*-->")
    text = (external_output / "external_fem_summary.md").read_text(encoding="utf-8")
    matches = list(marker.finditer(text))

    assert matches
    for match in matches:
        csv_path = external_output / match.group("csv")
        assert csv_path.is_file()
        with csv_path.open(newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames or []
        assert match.group("field") in fieldnames
