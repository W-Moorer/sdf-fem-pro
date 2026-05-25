from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_source_gear_contact_status_diagnostics import parse_abaqus_msg_contact_convergence


def test_parse_abaqus_msg_contact_convergence_keeps_status_and_compatibility(tmp_path: Path) -> None:
    msg = tmp_path / "job.msg"
    msg.write_text(
        """
  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-05
                   24 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   24 POINTS CHANGED FROM OPEN TO CLOSED

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 15.7307E-06  AT NODE GEAR2-1-1.1412 OF CONTACT PAIR
   MAX. CONTACT FORCE ERROR 37.8809E-03  AT NODE GEAR2-1-1.4906 OF CONTACT PAIR

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1

   MAX. PENETRATION ERROR -321.920E-12   AT NODE GEAR2-1-1.140 OF CONTACT PAIR
   MAX. CONTACT FORCE ERROR -393.472E-09   AT NODE GEAR2-1-1.140 OF CONTACT
          THE CONTACT CONSTRAINTS HAVE CONVERGED.

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH

 TIME INCREMENT COMPLETED  1.000E-05,  FRACTION OF STEP COMPLETED  5.000E-02
 STEP TIME COMPLETED       2.000E-05,  TOTAL TIME COMPLETED        2.000E-05
        """,
        encoding="utf-8",
    )

    rows = parse_abaqus_msg_contact_convergence(msg)

    assert len(rows) == 1
    row = rows[0]
    assert row["increment"] == 2
    assert row["open_to_closed_points"] == 24
    assert row["closed_to_open_points"] == 0
    assert row["severe_discontinuity_iterations_msg"] == 1
    assert row["equilibrium_iterations_msg"] == 1
    assert row["contact_converged_checks"] == 1
    assert row["total_iterations_msg"] == 2
    assert row["total_time_msg"] == pytest.approx(2.0e-5)
    assert row["severe_max_abs_penetration_error"] == pytest.approx(15.7307e-6)
    assert row["severe_max_abs_contact_force_error"] == pytest.approx(37.8809e-3)
    assert row["final_abs_penetration_error"] == pytest.approx(321.920e-12)
    assert row["final_abs_contact_force_error"] == pytest.approx(393.472e-9)
    assert row["penetration_error_reduction"] < 1.0e-4
    assert row["contact_force_error_reduction"] < 1.0e-4
