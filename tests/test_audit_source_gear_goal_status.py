from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.audit_source_gear_goal_status import (  # noqa: E402
    build_audit_rows,
    parse_abaqus_sta_completed_time,
    parse_source_deck,
    summarize_alignment,
    summarize_manifest,
)


def test_parse_source_deck_extracts_drive_and_contact_settings(tmp_path: Path) -> None:
    deck = tmp_path / "gear.inp"
    deck.write_text(
        "\n".join(
            [
                "*MPC",
                "BEAM, A, B",
                "*MPC",
                "BEAM, C, D",
                "*Surface Behavior, pressure-overclosure=LINEAR",
                "5.0e9",
                "*Contact Pair, interaction=IntProp-1, type=NODE TO SURFACE",
                "S1, S2",
                "*Step, name=Step-1, nlgeom=YES, inc=2000",
                "*Dynamic",
                "1e-05,0.05,1e-10,5e-05",
                "*Boundary, type=VELOCITY",
                "RP1, 6, 6, 52.36",
                "*Cload",
                "RP2, 6, 50.",
            ]
        ),
        encoding="utf-8",
    )

    settings = parse_source_deck(deck)

    assert settings.nlgeom == "YES"
    assert settings.dynamic_initial_dt == 1.0e-5
    assert settings.dynamic_total_time == 0.05
    assert settings.gear1_angular_velocity_z == 52.36
    assert settings.gear2_torque_z == 50.0
    assert settings.mpc_beam_count == 2
    assert settings.contact_behavior.endswith("LINEAR")
    assert "type=NODE TO SURFACE" in settings.contact_pair


def test_audit_rows_flag_partial_duration_and_large_stress_error(tmp_path: Path) -> None:
    manifest = tmp_path / "sfc_manifest.csv"
    manifest.write_text(
        "time,rotation_unit,stress_strain_postprocess,contact_pressure_nodeavg\n"
        "0.0,radian,linear_corotated,0\n"
        "0.00002,radian,linear_corotated,1\n",
        encoding="utf-8",
    )
    alignment = tmp_path / "alignment.csv"
    alignment.write_text(
        "sfc_time,region,metric,p95_rel_error\n"
        "0.00002,full,displacement_magnitude,0.01\n"
        "0.00002,full,von_mises_nodeavg,0.68\n"
        "0.00002,full,equivalent_elastic_strain_nodeavg,0.68\n",
        encoding="utf-8",
    )
    deck = tmp_path / "gear.inp"
    deck.write_text(
        "*Step, name=Step-1, nlgeom=YES, inc=2000\n"
        "*Dynamic\n"
        "1e-05,0.05,1e-10,5e-05\n"
        "*Boundary, type=VELOCITY\n"
        "RP1, 6, 6, 52.36\n"
        "*Cload\n"
        "RP2, 6, 50.\n",
        encoding="utf-8",
    )
    settings = parse_source_deck(deck)
    sfc = summarize_manifest(manifest)
    abaqus = summarize_manifest(manifest)
    align = summarize_alignment(alignment)
    sta = {"completed_time": 0.002, "completed_increment": 200}

    rows = build_audit_rows(settings, sfc, abaqus, align, sta)
    status = {row["requirement"]: row["status"] for row in rows}

    assert status["source_drive_units_radian"] == "PASS"
    assert status["abaqus_reference_duration_matches_source"] == "MISSING"
    assert status["full_source_duration_covered"] == "MISSING"
    assert status["stress_strain_alignment_currently_acceptable"] == "MISSING"


def test_parse_abaqus_sta_completed_time_reads_last_increment(tmp_path: Path) -> None:
    sta = tmp_path / "job.sta"
    sta.write_text(
        "   1   199   1     2     2     4  0.00199    0.00199    1.000e-05\n"
        "   1   200   1     2     2     4  0.00200    0.00200    1.000e-05\n"
        " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n",
        encoding="utf-8",
    )

    row = parse_abaqus_sta_completed_time(sta)

    assert row["exists"] == 1
    assert row["completed"] == 1
    assert row["completed_increment"] == 200
    assert row["completed_time"] == 0.002
