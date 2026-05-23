from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.diagnose_source_gear_alignment import build_diagnostic_rows, run_diagnostics, summarize_diagnostics


def _write_manifest_pair(tmp_path: Path) -> tuple[Path, Path]:
    sfc = tmp_path / "sfc_manifest.csv"
    abaqus = tmp_path / "abaqus_manifest.csv"
    sfc.write_text(
        "frame,time,vtk_file,active_contact_node_count,max_contact_pressure_nodeavg,p95_contact_pressure_nodeavg,"
        "rp1_rotation_z_rad,rp2_rotation_z_rad,rp1_angular_velocity_z_rad_per_s,rp2_angular_velocity_z_rad_per_s\n"
        "0,0.0,sfc_0000.vtk,0,0,0,0,0,52.36,0\n"
        "1,0.003,sfc_0001.vtk,0,0,0,0.15708,9.2,52.36,6150\n",
        encoding="utf-8",
    )
    abaqus.write_text(
        "frame,time,vtk_file,active_contact_node_count,max_contact_pressure_nodeavg,p95_contact_pressure_nodeavg,"
        "rp1_rotation_z_rad,rp2_rotation_z_rad,rp1_angular_velocity_z_rad_per_s,rp2_angular_velocity_z_rad_per_s\n"
        "0,0.0,abaqus_0000.vtk,0,0,0,0,0,52.36,0\n"
        "1,0.003000000001,abaqus_0001.vtk,88,4,2,0.15708,2.5,52.36,5800\n",
        encoding="utf-8",
    )
    return sfc, abaqus


def _write_regional_errors(tmp_path: Path) -> Path:
    path = tmp_path / "regional.csv"
    path.write_text(
        "sfc_time,region,metric,p95_rel_error\n"
        "0.0,full,displacement_magnitude,0\n"
        "0.0,full,von_mises_nodeavg,0\n"
        "0.0,full,strain_norm_nodeavg,0\n"
        "0.003,full,displacement_magnitude,0.034\n"
        "0.003,full,von_mises_nodeavg,0.184\n"
        "0.003,full,strain_norm_nodeavg,0.098\n"
        "0.003,full,equivalent_elastic_strain_nodeavg,0.184\n"
        "0.003,active_union,von_mises_nodeavg,2.48\n"
        "0.003,abaqus_active,von_mises_nodeavg,2.48\n",
        encoding="utf-8",
    )
    return path


def test_build_diagnostic_rows_combines_rp_contact_and_field_errors(tmp_path: Path) -> None:
    sfc, abaqus = _write_manifest_pair(tmp_path)
    regional = _write_regional_errors(tmp_path)

    rows = build_diagnostic_rows(
        sfc_manifest=sfc,
        abaqus_manifest=abaqus,
        regional_errors=regional,
        time_tolerance=1.0e-8,
    )

    assert len(rows) == 2
    final = rows[-1]
    assert final["active_contact_node_abs_diff"] == 88
    assert final["full_von_mises_p95_rel_error"] == pytest.approx(0.184)
    assert final["full_strain_norm_p95_rel_error"] == pytest.approx(0.098)
    assert final["rp2_rotation_z_rel_error"] == pytest.approx(abs(9.2 - 2.5) / 2.5)
    assert final["p95_contact_pressure_rel_error"] == pytest.approx(1.0)


def test_summarize_diagnostics_flags_incomplete_stress_gate(tmp_path: Path) -> None:
    sfc, abaqus = _write_manifest_pair(tmp_path)
    regional = _write_regional_errors(tmp_path)
    rows = build_diagnostic_rows(
        sfc_manifest=sfc,
        abaqus_manifest=abaqus,
        regional_errors=regional,
        time_tolerance=1.0e-8,
    )

    summary = summarize_diagnostics(rows)

    assert summary["latest_displacement_pass_10pct"] == 1
    assert summary["latest_von_mises_pass_10pct"] == 0
    assert summary["latest_strain_norm_pass_10pct"] == 1
    assert summary["overall_latest_gate_pass"] == 0
    assert summary["latest_active_contact_node_abs_diff"] == 88


def test_run_diagnostics_writes_summary_artifacts(tmp_path: Path) -> None:
    sfc, abaqus = _write_manifest_pair(tmp_path)
    regional = _write_regional_errors(tmp_path)

    outputs = run_diagnostics(
        sfc_manifest=sfc,
        abaqus_manifest=abaqus,
        regional_errors=regional,
        out_dir=tmp_path / "out",
        time_tolerance=1.0e-8,
    )

    assert outputs["diagnostics_csv"].exists()
    assert outputs["summary_csv"].exists()
    assert outputs["summary_md"].exists()
    with outputs["summary_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["overall_latest_gate_pass"] == "0"
    text = outputs["summary_md"].read_text(encoding="utf-8")
    assert "contact status/release" in text
