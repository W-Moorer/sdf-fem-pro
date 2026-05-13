from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_external_contact_solver_comparison.py"


def _sfepy_available() -> bool:
    if shutil.which("sfepy-run") is None or shutil.which("py") is None:
        return False
    proc = subprocess.run(
        ["py", "-3.12", "-c", "import sfepy"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return proc.returncode == 0


pytestmark = pytest.mark.skipif(not _sfepy_available(), reason="SfePy external contact solver is unavailable")

EXPECTED_FILES = {
    "external_contact_solver_comparison.csv",
    "external_contact_solver_commands.csv",
    "external_contact_solver_claims.csv",
    "external_contact_solver_plots.csv",
    "external_contact_solver_summary.md",
    "external_contact_gap_comparison.png",
    "external_contact_gap_comparison.pdf",
    "external_contact_active_count.png",
    "external_contact_active_count.pdf",
}


@pytest.fixture(scope="session")
def external_contact_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("external-contact-solver")
    subprocess.run(
        [sys.executable, str(RUNNER), "--quick", "--out-dir", str(out_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return out_dir


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_external_contact_quick_mode_writes_expected_outputs(external_contact_output: Path) -> None:
    produced = {path.name for path in external_contact_output.iterdir() if path.is_file()}

    assert EXPECTED_FILES <= produced
    assert all((external_contact_output / name).stat().st_size > 0 for name in EXPECTED_FILES)


def test_external_contact_solver_gap_replay_matches_contact_state(external_contact_output: Path) -> None:
    rows = _rows(external_contact_output / "external_contact_solver_comparison.csv")

    assert len(rows) >= 3
    assert all(row["external_solver"] == "SfePy" for row in rows)
    assert all(row["contact_state_agreement"] == "true" for row in rows)
    assert max(float(row["gap_mean_abs_difference"]) for row in rows) < 1.5e-2
    assert any(int(row["sfepy_active_count"]) > 0 for row in rows)
    assert any(int(row["sfc_replay_active_count"]) > 0 for row in rows)


def test_external_contact_claims_are_supported(external_contact_output: Path) -> None:
    rows = _rows(external_contact_output / "external_contact_solver_claims.csv")

    assert rows
    assert {row["claim_status"] for row in rows} == {"supported"}


def test_external_contact_summary_claim_markers_have_backing_csv_fields(external_contact_output: Path) -> None:
    marker = re.compile(r"<!--\s*evidence\s+csv=(?P<csv>\S+)\s+field=(?P<field>\S+)\s*-->")
    text = (external_contact_output / "external_contact_solver_summary.md").read_text(encoding="utf-8")
    matches = list(marker.finditer(text))

    assert matches
    for match in matches:
        csv_path = external_contact_output / match.group("csv")
        assert csv_path.is_file()
        with csv_path.open(newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames or []
        assert match.group("field") in fieldnames
