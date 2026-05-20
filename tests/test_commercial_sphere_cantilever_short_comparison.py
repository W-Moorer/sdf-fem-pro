from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_commercial_sphere_cantilever_short_comparison import (  # noqa: E402
    DEFAULT_ABAQUS_INP,
    build_short_model,
    run_sfc_lagrangian_short,
    write_calculix_input,
)
from validation.run_abaqus_sphere_cantilever_explicit import ModelConfig, build_input_text  # noqa: E402

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


def test_short_model_parses_direct_explicit_dt_and_soft_sphere(tmp_path: Path) -> None:
    inp = tmp_path / "sphere_cantilever_explicit.inp"
    inp.write_text(build_input_text(ModelConfig(duration=0.05, fixed_dt=1.0e-5, sphere_young=5.0e6)), encoding="ascii")

    model = build_short_model(duration=None, dt=None, inp_path=inp)

    assert model.duration == 0.05
    assert model.dt == 1.0e-5
    assert model.sphere_material.young == pytest.approx(5.0e6)
    assert model.sphere_material.young < model.beam_material.young


def test_explicit_sfc_structured_top_backend_runs_small_soft_model(tmp_path: Path) -> None:
    inp = tmp_path / "sphere_cantilever_explicit.inp"
    cfg = ModelConfig(
        beam_nx=2,
        beam_ny=2,
        beam_nz=1,
        sphere_latitudes=6,
        sphere_longitudes=12,
        duration=2.0e-5,
        fixed_dt=1.0e-5,
        beam_young=2.5e8,
        sphere_young=5.0e6,
    )
    inp.write_text(build_input_text(cfg), encoding="ascii")
    model = build_short_model(duration=None, dt=None, inp_path=inp)

    rows, command = run_sfc_lagrangian_short(
        model,
        contact_stiffness=5.0e9,
        damping_alpha=0.0,
        integrator="explicit",
        contact_backend="structured-top",
    )

    assert len(rows) == 2
    assert command["time_integrator"] == "explicit"
    assert command["sfc_contact_backend"] == "structured-top"
    assert command["output_stride"] == 100
    assert all(np.isfinite(float(row["sphere_mean_uz"])) for row in rows)
