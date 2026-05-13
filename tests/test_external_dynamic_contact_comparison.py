from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_external_dynamic_contact_comparison.py"


def _sfepy_available() -> bool:
    if shutil.which("sfepy-run") is None or shutil.which("py") is None:
        return False
    proc = subprocess.run(
        ["py", "-3.12", "-c", "import sfepy"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode == 0


pytestmark = pytest.mark.skipif(not _sfepy_available(), reason="SfePy external FEM/contact reference is unavailable")

EXPECTED_FILES = {
    "external_dynamic_contact_time_history.csv",
    "external_dynamic_contact_comparison.csv",
    "external_dynamic_contact_claims.csv",
    "external_dynamic_contact_commands.csv",
    "external_dynamic_contact_plots.csv",
    "external_dynamic_contact_summary.md",
    "external_dynamic_contact_min_gap.png",
    "external_dynamic_contact_min_gap.pdf",
    "external_dynamic_contact_force_history.png",
    "external_dynamic_contact_force_history.pdf",
    "external_dynamic_contact_active_count.png",
    "external_dynamic_contact_active_count.pdf",
}


@pytest.fixture(scope="session")
def external_dynamic_contact_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("external-dynamic-contact")
    subprocess.run(
        [sys.executable, str(RUNNER), "--quick", "--out-dir", str(out_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=420,
    )
    return out_dir


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_external_dynamic_contact_quick_mode_writes_expected_outputs(external_dynamic_contact_output: Path) -> None:
    produced = {path.name for path in external_dynamic_contact_output.iterdir() if path.is_file()}

    assert EXPECTED_FILES <= produced
    assert all((external_dynamic_contact_output / name).stat().st_size > 0 for name in EXPECTED_FILES)


def test_external_dynamic_contact_contains_mass_newmark_and_contact_sources(external_dynamic_contact_output: Path) -> None:
    rows = _rows(external_dynamic_contact_output / "external_dynamic_contact_time_history.csv")
    sources = {row["source"] for row in rows}

    assert {
        "sfepy_transient_elastodynamic",
        "sfc_replay_on_sfepy_transient_geometry",
        "sfepy_contact_reference_snapshot",
        "sfc_contact_snapshot_replay",
        "sfc_newmark_penalty_contact",
    } <= sources
    assert any(row["integrator"] == "SfePy ts.newmark" and row["mass_matrix_included"] == "true" for row in rows)
    assert any(row["integrator"] == "sfc.newmark_beta_step" and row["mass_matrix_included"] == "true" for row in rows)
    assert all(abs(float(row["total_time"]) - 3.0) < 1.0e-12 for row in rows)
    assert any(int(row["active_contact_count"]) > 0 for row in rows if row["source"] == "sfepy_contact_reference_snapshot")
    assert any(int(row["active_contact_count"]) > 0 for row in rows if row["source"] == "sfc_newmark_penalty_contact")


def test_external_dynamic_contact_claims_are_supported(external_dynamic_contact_output: Path) -> None:
    rows = _rows(external_dynamic_contact_output / "external_dynamic_contact_claims.csv")

    assert rows
    assert {row["claim_status"] for row in rows} == {"supported"}


def test_external_dynamic_contact_summary_claim_markers_have_backing_csv_fields(external_dynamic_contact_output: Path) -> None:
    marker = re.compile(r"<!--\s*evidence\s+csv=(?P<csv>\S+)\s+field=(?P<field>\S+)\s*-->")
    text = (external_dynamic_contact_output / "external_dynamic_contact_summary.md").read_text(encoding="utf-8")
    matches = list(marker.finditer(text))

    assert matches
    for match in matches:
        csv_path = external_dynamic_contact_output / match.group("csv")
        assert csv_path.is_file()
        with csv_path.open(newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames or []
        assert match.group("field") in fieldnames


def test_external_dynamic_contact_runner_does_not_use_disallowed_rigid_body_solver() -> None:
    text = RUNNER.read_text(encoding="utf-8").lower()

    disallowed_name = "mu" + "joco"
    assert disallowed_name not in text
