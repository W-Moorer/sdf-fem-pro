from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_calculix_contactenergy_replay.py"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _wsl_contactenergy_reference_available() -> bool:
    if shutil.which("wsl") is None:
        return False
    command = "test -f /tmp/sfc_calculix_source/test/contactenergy.inp && test -f /tmp/sfc_calculix_source/test/contactenergy.dat.ref"
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return proc.returncode == 0


@pytest.mark.skipif(not _wsl_contactenergy_reference_available(), reason="local CalculiX contactenergy reference is unavailable")
def test_calculix_contactenergy_replay_quick_outputs_dynamic_sdf_c3d8_metrics(tmp_path: Path) -> None:
    out_dir = tmp_path / "contactenergy"

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
        "calculix_contactenergy_replay.csv",
        "calculix_contactenergy_claims.csv",
        "calculix_contactenergy_commands.csv",
        "calculix_contactenergy_raw_cels.csv",
        "calculix_contactenergy_plots.csv",
        "calculix_contactenergy_stress_strain_cloud.csv",
        "calculix_contactenergy_error_metrics.csv",
        "calculix_contactenergy_summary.md",
    ]
    for name in expected:
        path = out_dir / name
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    row = _rows(out_dir / "calculix_contactenergy_replay.csv")[0]
    assert row["element_type"] == "c3d8"
    assert row["dynamic_sdf_backend"] == "current_surface_triangulated_c3d8_faces"
    assert row["reference_scope"] == "contact_law_energy_replay_not_tet4_trajectory"
    assert row["status"] == "ok"
    assert float(row["contact_force_rel_error"]) <= 1.0e-5
    assert float(row["contact_energy_rel_error"]) <= 1.0e-5
    assert float(row["sfc_c3d8_displacement_l2_rel_error"]) >= 0.0
    assert float(row["sfc_c3d8_stress_l2_rel_error"]) >= 0.0
    assert float(row["sfc_c3d8_von_mises_l2_rel_error"]) >= 0.0

    claims = {row["claim"]: row for row in _rows(out_dir / "calculix_contactenergy_claims.csv")}
    assert claims["dynamic_sdf_replays_c3d8_contact_energy"]["supported"] == "true"
    assert claims["dynamic_sdf_is_not_tet4_bound"]["supported"] == "true"

    for figure in [
        out_dir / "figures" / "calculix_contactenergy_stress_strain_3d.png",
        out_dir / "figures" / "calculix_contactenergy_stress_strain_3d.pdf",
        out_dir / "figures" / "calculix_contactenergy_contact_pressure_3d.png",
        out_dir / "figures" / "calculix_contactenergy_contact_pressure_3d.pdf",
        out_dir / "figures" / "calculix_contactenergy_error_metrics.png",
        out_dir / "figures" / "calculix_contactenergy_error_metrics.pdf",
        out_dir / "figures" / "calculix_contactenergy_sfc_c3d8_error_3d.png",
        out_dir / "figures" / "calculix_contactenergy_sfc_c3d8_error_3d.pdf",
    ]:
        assert figure.exists(), figure
        assert figure.stat().st_size > 0, figure

    plot_names = {row["plot"] for row in _rows(out_dir / "calculix_contactenergy_plots.csv")}
    assert "calculix_contactenergy_error_metrics" in plot_names
    assert "calculix_contactenergy_sfc_c3d8_error_3d" in plot_names

    cloud = _rows(out_dir / "calculix_contactenergy_stress_strain_cloud.csv")
    assert len(cloud) == 2
    assert max(float(row["von_mises"]) for row in cloud) > 0.0
    assert max(float(row["engineering_strain_norm"]) for row in cloud) > 0.0
    assert max(float(row["sfc_von_mises"]) for row in cloud) > 0.0
    assert max(float(row["von_mises_abs_error"]) for row in cloud) >= 0.0

    errors = _rows(out_dir / "calculix_contactenergy_error_metrics.csv")
    assert {row["quantity"] for row in errors} == {"normal_force_z", "contact_energy", "c3d8_displacement", "c3d8_von_mises"}
    by_quantity = {row["quantity"]: row for row in errors}
    assert by_quantity["normal_force_z"]["status"] == "ok"
    assert by_quantity["contact_energy"]["status"] == "ok"
    assert max(float(by_quantity[name]["relative_error"]) for name in ["normal_force_z", "contact_energy"]) <= 1.0e-5
