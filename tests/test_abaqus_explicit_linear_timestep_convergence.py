from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_explicit_linear_timestep_convergence import (  # noqa: E402
    ExplicitLinearConfig,
    _analysis_status_from_sta,
    _case_name,
    _energy_audit_rows,
    build_input_text,
)


def test_explicit_linear_input_uses_fixed_user_time_increment_and_no_bulk_viscosity() -> None:
    text = build_input_text(ExplicitLinearConfig(duration=1.0, output_interval=0.001, fixed_dt=1.0e-4))

    assert "*Dynamic, Explicit, DIRECT USER CONTROL" in text
    assert "1.000000000000e-04, 1.000000000000e+00" in text
    assert "*Bulk Viscosity\n0., 0." in text
    assert "*Surface Behavior, pressure-overclosure=LINEAR" in text
    assert "5.000000000000e+09" in text
    assert "*Friction\n0." in text
    assert "*Contact Inclusions, ALL EXTERIOR" in text
    assert "*Output, field, time interval=1.000000000000e-03" in text
    assert "*Energy Output" in text
    assert "ALLKE, ALLIE, ALLSE, ALLVD, ALLWK, ETOTAL" in text
    assert "*Contact Damping" not in text


def test_explicit_linear_input_can_disable_default_contact_damping() -> None:
    text = build_input_text(
        ExplicitLinearConfig(duration=1.0, output_interval=0.001, fixed_dt=1.0e-6, contact_damping_fraction=0.0)
    )

    assert "*Contact Damping, definition=CRITICAL DAMPING FRACTION" in text
    assert "0.000000000000e+00" in text
    assert text.index("*Contact Damping") > text.index("*Surface Behavior")
    assert text.index("*Contact Damping") < text.index("*Friction")


def test_explicit_linear_config_preserves_sphere_drop_material_values() -> None:
    cfg = ExplicitLinearConfig()

    assert cfg.duration == 1.0
    assert cfg.output_interval == 0.001
    assert cfg.contact_stiffness == 5.0e9
    assert cfg.density == 1200.0
    assert cfg.young == 5.0e7
    assert cfg.poisson == 0.30
    assert cfg.contact_damping_fraction is None


def test_case_name_preserves_fractional_scientific_mantissa() -> None:
    assert _case_name(5.0e-6) == "dt_5em06"
    assert _case_name(2.5e-6) == "dt_2p5em06"
    assert _case_name(1.0e-6) == "dt_1em06"


def test_analysis_status_from_sta_detects_explicit_instability(tmp_path: Path) -> None:
    sta = tmp_path / "job.sta"
    sta.write_text(
        "***ERROR: Excessive distortion of element number 1\n"
        "***ERROR: The ratio of deformation speed to wave speed exceeds 1.0000\n"
        "  THE ANALYSIS HAS NOT BEEN COMPLETED\n",
        encoding="utf-8",
    )

    status, message = _analysis_status_from_sta(sta)

    assert status == "analysis_failed"
    assert "excessive element distortion" in message
    assert "deformation speed exceeds wave speed" in message


def test_analysis_status_from_sta_detects_success(tmp_path: Path) -> None:
    sta = tmp_path / "job.sta"
    sta.write_text("  THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n", encoding="utf-8")

    status, message = _analysis_status_from_sta(sta)

    assert status == "completed"
    assert message == ""


def test_energy_audit_reports_etotal_drift() -> None:
    rows = [
        {"case": "dt_1em06", "fixed_dt": 1.0e-6, "time": 0.0, "ETOTAL": "1.0", "ALLKE": "2.0", "ALLIE": "0.5", "ALLSE": "0.5", "ALLVD": "0.0", "ALLWK": "0.0"},
        {"case": "dt_1em06", "fixed_dt": 1.0e-6, "time": 1.0, "ETOTAL": "1.2", "ALLKE": "1.0", "ALLIE": "0.7", "ALLSE": "0.7", "ALLVD": "0.0", "ALLWK": "0.1"},
    ]

    audit = _energy_audit_rows({"dt_1em06": rows})

    assert len(audit) == 1
    assert audit[0]["case"] == "dt_1em06"
    assert audit[0]["etotal_final_minus_initial"] == 0.19999999999999996
    assert audit[0]["max_abs_etotal_drift"] == 0.19999999999999996
    assert audit[0]["allvd_final"] == 0.0
