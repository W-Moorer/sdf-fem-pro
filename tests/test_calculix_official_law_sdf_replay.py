from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_calculix_official_law_sdf_replay.py"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _calculix_official_examples_available() -> bool:
    if shutil.which("wsl") is None:
        return False
    command = (
        "command -v ccx >/dev/null && "
        "test -f /usr/share/doc/calculix-ccx-test/examples/test/contactenergy.inp && "
        "test -f /usr/share/doc/calculix-ccx-test/examples/test/contact1.inp && "
        "test -f /usr/share/doc/calculix-ccx-test/examples/test/contact3.inp && "
        "test -f /usr/share/doc/calculix-ccx-test/examples/test/contact6.inp"
    )
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return proc.returncode == 0


@pytest.mark.skipif(not _calculix_official_examples_available(), reason="local CalculiX official examples are unavailable")
def test_official_law_sdf_replay_quick_outputs_claim_gated_metrics(tmp_path: Path) -> None:
    out_dir = tmp_path / "law_replay"
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

    expected = [
        "official_law_sdf_replay_summary.csv",
        "official_law_sdf_replay_contact_points.csv",
        "official_law_sdf_replay_claim_gates.csv",
        "official_law_sdf_replay_commands.csv",
        "official_law_sdf_replay_plots.csv",
        "official_law_sdf_replay_summary.md",
    ]
    for name in expected:
        path = out_dir / name
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    summaries = {row["case_id"]: row for row in _rows(out_dir / "official_law_sdf_replay_summary.csv")}
    assert set(summaries) == {
        "contactenergy_c3d8_surface_linear",
        "contact1_c3d8_node_exponential",
        "contact3_c3d8_node_linear",
        "contact6_c3d8_node_stiff_linear",
    }
    assert summaries["contactenergy_c3d8_surface_linear"]["pressure_law"] == "LINEAR"
    assert summaries["contact1_c3d8_node_exponential"]["pressure_law"] == "EXPONENTIAL"
    assert float(summaries["contactenergy_c3d8_surface_linear"]["max_gap_abs_error"]) <= 1.0e-10
    assert float(summaries["contactenergy_c3d8_surface_linear"]["total_energy_rel_error"]) <= 1.0e-5
    assert float(summaries["contact3_c3d8_node_linear"]["max_pressure_rel_error"]) <= 5.0e-2
    assert float(summaries["contact6_c3d8_node_stiff_linear"]["max_pressure_rel_error"]) <= 5.0e-2

    gates = _rows(out_dir / "official_law_sdf_replay_claim_gates.csv")
    by_case_claim = {(row["case_id"], row["claim"]): row for row in gates}
    assert by_case_claim[("contactenergy_c3d8_surface_linear", "contact_energy_matches_calculix_cels")]["allowed"] == "true"
    assert by_case_claim[("contact1_c3d8_node_exponential", "pressure_law_matches_calculix_cstr")]["allowed"] == "true"
    assert by_case_claim[("contact1_c3d8_node_exponential", "native_sfc_trajectory_equivalence")]["allowed"] == "false"
    assert by_case_claim[("contact3_c3d8_node_linear", "linear_pressure_law_matches_calculix_cstr")]["allowed"] == "true"
    assert by_case_claim[("contact6_c3d8_node_stiff_linear", "linear_pressure_law_matches_calculix_cstr")]["allowed"] == "true"

    point_rows = _rows(out_dir / "official_law_sdf_replay_contact_points.csv")
    assert len(point_rows) >= 4
    assert all(row["sfc_gap"] != "" for row in point_rows)
    assert all(row["calculix_cdis"] != "" for row in point_rows)
    assert all(row["sfc_pressure"] != "" for row in point_rows)

    for figure in [
        out_dir / "figures" / "official_law_sdf_replay_errors.png",
        out_dir / "figures" / "official_law_sdf_replay_force_balance.png",
    ]:
        assert figure.exists(), figure
        assert figure.stat().st_size > 0, figure
