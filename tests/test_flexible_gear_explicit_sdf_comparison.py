from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flexible_gear_explicit_sdf_comparison import (
    DEFAULT_SOURCE,
    build_explicit_input_text,
    parse_gear_input,
)


pytestmark = pytest.mark.skipif(not DEFAULT_SOURCE.exists(), reason="commercial gear input is not present")


def test_parse_flexible_gear_input_extracts_contact_and_hub_data() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)

    assert model.gear1.nodes.shape == (19322, 3)
    assert model.gear2.nodes.shape == (19562, 3)
    assert model.gear1_contact_faces.shape[0] == 11560
    assert model.gear2_contact_faces.shape[0] == 11560
    assert len(model.gear1_hub_labels) == 204
    assert len(model.gear2_hub_labels) == 204
    assert model.density == pytest.approx(7850.0)
    assert model.young == pytest.approx(2.05e11)
    assert model.poisson == pytest.approx(0.28)


def test_compact_explicit_deck_uses_penalty_linear_contact() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    text = build_explicit_input_text(
        model,
        duration=2.0e-5,
        fixed_dt=5.0e-9,
        output_interval=1.0e-5,
        contact_stiffness=5.0e9,
        max_abaqus_contact_faces=16,
    )

    assert "*Dynamic, Explicit, DIRECT USER CONTROL" in text
    assert "5.000000000000e-09, 2.000000000000e-05" in text
    assert "*Surface Behavior, pressure-overclosure=LINEAR" in text
    assert "5.000000000000e+09" in text
    assert "*Contact Pair, interaction=LINEAR_FRICTIONLESS, mechanical constraint=PENALTY" in text
    assert "type=SURFACE TO SURFACE" not in text
    assert "*Bulk Viscosity\n0., 0." in text
    assert "GEAR1_RP, 6, 6, 5.236000000000e+01" in text
    assert "GEAR2_LOAD_RP, 6, 5.000000000000e+01" in text
