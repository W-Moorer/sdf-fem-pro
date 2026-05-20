from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_sphere_drop_integrator_sweep import DEFAULT_CASES, _contact_impulse


def test_default_integrator_sweep_includes_abaqus_style_dissipation_cases() -> None:
    cases = {case.name: case for case in DEFAULT_CASES}

    assert cases["hht_tf_alpha_m005"].integrator == "hht"
    assert cases["hht_tf_alpha_m005"].hht_alpha == pytest.approx(-0.05)
    assert cases["hht_md_alpha_m041421"].hht_alpha == pytest.approx(-0.41421356237)
    assert cases["bwe"].integrator == "bwe"
    assert cases["hht_alpha_m030_mass12_adaptive"].adaptive_increments is True


def test_contact_impulse_uses_trapezoidal_time_integral() -> None:
    rows = [
        {"time": 0.0, "normal_force_z": 0.0},
        {"time": 0.5, "normal_force_z": 2.0},
        {"time": 1.0, "normal_force_z": 0.0},
    ]

    assert _contact_impulse(rows) == pytest.approx(1.0)
