from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.prepare_flexible_gear_source_penalty_deck import prepare_source_penalty_deck_text


def test_prepare_source_penalty_deck_uses_linear_penalty_and_strided_output() -> None:
    source = "\n".join(
        [
            "*Heading",
            "*Surface Interaction, name=IntProp-1",
            "1.,",
            "*Friction",
            "0.,",
            "*Surface Behavior, pressure-overclosure=HARD",
            "** INTERACTIONS",
            "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE",
            "_PickedSurf16, _PickedSurf15",
            "*Step, name=Step-1, nlgeom=YES, inc=2000",
            "*Dynamic",
            "1e-05,0.05,1e-10,5e-05",
            "*Output, field, variable=PRESELECT",
            "*Output, history, variable=PRESELECT",
            "*End Step",
        ]
    )

    text = prepare_source_penalty_deck_text(
        source,
        pressure_stiffness=5.0e9,
        frame_stride=5,
        dt=1.0e-5,
        duration=1.0e-4,
    )

    assert "*Surface Behavior, pressure-overclosure=LINEAR" in text
    assert "5.000000000000e+09" in text
    assert "pressure-overclosure=HARD" not in text
    assert "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE, mechanical constraint=PENALTY" in text
    assert "1.000000000000e-05,1.000000000000e-04,1e-10,5e-05" in text
    assert "*Output, field, variable=PRESELECT, frequency=5" in text
    assert "*Output, history, variable=PRESELECT, frequency=5" in text


def test_prepare_source_penalty_deck_replaces_existing_linear_data() -> None:
    source = "\n".join(
        [
            "*Surface Behavior, pressure-overclosure=LINEAR",
            "1.0e3",
            "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE, mechanical constraint=PENALTY",
            "S2, S1",
        ]
    )

    text = prepare_source_penalty_deck_text(source, pressure_stiffness=7.0e9, frame_stride=2)

    assert "7.000000000000e+09" in text
    assert "1.0e3" not in text
    assert text.count("mechanical constraint=PENALTY") == 1
