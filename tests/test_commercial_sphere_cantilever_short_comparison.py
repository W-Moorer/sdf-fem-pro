from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_commercial_sphere_cantilever_short_comparison import (  # noqa: E402
    DEFAULT_ABAQUS_INP,
    build_short_model,
    write_calculix_input,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_ABAQUS_INP.exists(),
    reason="commercial Abaqus input deck is not available in this checkout",
)


def test_short_comparison_calculix_input_contains_native_contact(tmp_path: Path) -> None:
    model = build_short_model(duration=0.05, dt=0.001)
    inp = tmp_path / "sphere_cantilever_short_ccx.inp"

    write_calculix_input(model, inp, contact_stiffness=5.0e10)
    text = inp.read_text(encoding="utf-8")

    assert "*CONTACT PAIR, INTERACTION=CONTACT, TYPE=SURFACE TO SURFACE" in text
    assert "*DYNAMIC, ALPHA=-0.05" in text
    assert "5.000000000000e-02" in text
    assert "*ELEMENT, TYPE=C3D8R, ELSET=EBEAM" in text
    assert "*ELEMENT, TYPE=C3D4, ELSET=ESPHERE" in text
    assert "SSPHERE, SBEAMTOP" in text


def test_short_model_uses_bottom_sphere_samples_and_beam_top_material_surface() -> None:
    model = build_short_model(duration=0.05, dt=0.001)

    assert model.source_inp.name == "sphere_cantilever_explicit.inp"
    assert model.beam_top_faces.shape[0] == 960
    assert 0 < model.sphere_sample_faces.shape[0] < model.sphere_surface_faces.shape[0]
    assert model.sphere_sample_areas.shape == (model.sphere_sample_faces.shape[0],)
    assert model.beam_elements.shape[1] == 8
    assert model.sphere_elements.shape[1] == 4
    assert model.duration == 0.05
    assert model.dt == 0.001
