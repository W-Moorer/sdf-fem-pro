from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_source_gear_contact_status_diagnostics import (
    build_contact_status_diagnostics,
    parse_abaqus_sta_increments,
)


def test_parse_abaqus_sta_increments(tmp_path: Path) -> None:
    sta = tmp_path / "job.sta"
    sta.write_text(
        "\n".join(
            [
                " STEP  INC ATT SEVERE EQUIL TOTAL  TOTAL      STEP       INC OF",
                "   1     1   1     0     2     2  1.00e-05   1.00e-05   1.000e-05",
                "   1     2   1     1     1     2  2.00e-05   2.00e-05   1.000e-05",
            ]
        ),
        encoding="utf-8",
    )

    increments = parse_abaqus_sta_increments(sta)

    assert len(increments) == 2
    assert increments[1]["increment"] == 2
    assert increments[1]["severe_discontinuity_iterations"] == 1
    assert increments[1]["increment_size"] == 1.0e-5


def test_contact_status_diagnostics_flags_release_mismatch(tmp_path: Path) -> None:
    sfc = tmp_path / "sfc.csv"
    abaqus = tmp_path / "abaqus.csv"
    sta = tmp_path / "job.sta"
    sfc.write_text(
        "\n".join(
            [
                "time,active_contact_node_count,max_contact_pressure_nodeavg,mean_active_contact_pressure_nodeavg,min_contact_gap_node,p95_von_mises_nodeavg,p95_equivalent_elastic_strain_nodeavg,newton_iterations,newton_residual_norm",
                "0.00038,351,1.5e7,9.0e6,-0.003,1.3e7,5e-5,8,2e-7",
            ]
        ),
        encoding="utf-8",
    )
    abaqus.write_text(
        "\n".join(
            [
                "time,active_contact_node_count,max_contact_pressure_nodeavg,mean_active_contact_pressure_nodeavg,min_contact_gap_node,p95_von_mises_nodeavg,p95_equivalent_elastic_strain_nodeavg",
                "0.00038,0,0,0,0,4.4e6,1.8e-5",
            ]
        ),
        encoding="utf-8",
    )
    sta.write_text("   1    38   1     0     3     3  3.80e-04   3.80e-04   1.000e-05\n", encoding="utf-8")

    rows = build_contact_status_diagnostics(sfc_history=sfc, abaqus_manifest=abaqus, abaqus_sta=sta)

    assert len(rows) == 1
    assert rows[0]["abaqus_increment_size"] == 1.0e-5
    assert rows[0]["active_contact_node_count_delta"] == 351.0
    assert rows[0]["release_stage_mismatch"] == 1
