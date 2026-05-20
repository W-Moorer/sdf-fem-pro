from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_explicit_linear_timestep_convergence import (  # noqa: E402
    ExplicitLinearConfig,
    _analysis_status_from_sta,
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


def test_explicit_linear_config_preserves_sphere_drop_material_values() -> None:
    cfg = ExplicitLinearConfig()

    assert cfg.duration == 1.0
    assert cfg.output_interval == 0.001
    assert cfg.contact_stiffness == 5.0e9
    assert cfg.density == 1200.0
    assert cfg.young == 5.0e7
    assert cfg.poisson == 0.30


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
