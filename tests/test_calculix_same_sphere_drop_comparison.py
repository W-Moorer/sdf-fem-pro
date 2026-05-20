from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_flexible_body_rigid_plane_implicit import PENALTY_NORMAL_STIFFNESS, build_implicit_input_text
from validation.run_abaqus_sphere_drop_short_validation import parse_sphere_drop_inp
from validation.run_calculix_drop_impact_comparison import CALCULIX_MASTER_SURFACE_OFFSET, write_calculix_input
from validation.run_calculix_same_sphere_drop_comparison import build_calculix_model_from_abaqus


def test_calculix_model_preserves_abaqus_implicit_sphere_condition(tmp_path: Path) -> None:
    inp = tmp_path / "sphere_drop_implicit.inp"
    inp.write_text(build_implicit_input_text("sphere_drop"), encoding="ascii")
    source = parse_sphere_drop_inp(inp)

    model = build_calculix_model_from_abaqus(source, duration=0.12, dt=0.001, hht_alpha=-0.3)

    assert model.nodes.shape == source.nodes.shape
    assert model.tet_elements.shape == source.elements.shape
    assert model.initial_velocity_z == pytest.approx(0.0)
    assert model.gravity == pytest.approx(source.gravity)
    assert model.E == pytest.approx(source.young)
    assert model.nu == pytest.approx(source.poisson)
    assert model.density == pytest.approx(source.density)
    assert model.contact_stiffness == pytest.approx(PENALTY_NORMAL_STIFFNESS)
    assert model.floor_z + CALCULIX_MASTER_SURFACE_OFFSET == pytest.approx(source.plane_z)


def test_calculix_input_uses_linear_penalty_and_zero_initial_velocity(tmp_path: Path) -> None:
    inp = tmp_path / "sphere_drop_implicit.inp"
    inp.write_text(build_implicit_input_text("sphere_drop"), encoding="ascii")
    source = parse_sphere_drop_inp(inp)
    model = build_calculix_model_from_abaqus(source, duration=0.12, dt=0.001, hht_alpha=-0.3)

    out = tmp_path / "same_condition.inp"
    write_calculix_input(model, out)
    text = out.read_text(encoding="utf-8").lower()

    assert "*dynamic, alpha=-0.3" in text
    assert "*dynamic, direct" not in text
    assert "*surface behavior, pressure-overclosure=linear" in text
    assert f"{PENALTY_NORMAL_STIFFNESS:.12g}".lower() in text
    assert "nall, 3, 0" in text
    assert "elall, grav, 9.81, 0.0, 0.0, -1.0" in text


def test_calculix_input_can_write_frictionless_explicit_dynamic(tmp_path: Path) -> None:
    inp = tmp_path / "sphere_drop_implicit.inp"
    inp.write_text(build_implicit_input_text("sphere_drop"), encoding="ascii")
    source = parse_sphere_drop_inp(inp)
    model = build_calculix_model_from_abaqus(
        source,
        duration=0.05,
        dt=5.0e-6,
        hht_alpha=0.0,
        output_frequency=200,
        direct_dynamic=True,
        explicit_dynamic=True,
        friction_coefficient=0.0,
    )

    out = tmp_path / "same_condition_explicit.inp"
    write_calculix_input(model, out)
    text = out.read_text(encoding="utf-8").lower()

    assert "*dynamic, direct, explicit" in text
    assert "5e-06, 0.05" in text
    assert "*friction" not in text
    assert "alpha=" not in text
