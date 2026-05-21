from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flexible_gear_explicit_sdf_comparison import DEFAULT_SOURCE, parse_gear_input
from validation.run_flexible_gear_implicit_lagrangian_sdf_comparison import (
    _write_abaqus_alignment_deck,
    build_cropped_pair,
    solve_sfc_cropped_pair,
    solve_sfc_cropped_pair_hard_contact,
)


pytestmark = pytest.mark.skipif(not DEFAULT_SOURCE.exists(), reason="commercial gear input is not present")


def test_cropped_gear_pair_is_small_and_has_supports() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=4, expansion_rings=0)

    assert pair.gear1.nodes.shape[0] < model.gear1.nodes.shape[0]
    assert pair.gear2.nodes.shape[0] < model.gear2.nodes.shape[0]
    assert pair.gear1.contact_faces.shape[0] == 4
    assert pair.gear2.contact_faces.shape[0] == 4
    assert pair.gear1.support_nodes.size >= 4
    assert pair.gear2.support_nodes.size >= 4
    assert pair.initial_patch_gap >= 0.0


def test_cropped_gear_sfc_lagrangian_sdf_path_runs_one_implicit_step() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
    )

    assert len(history) == 1
    assert summary["status"] == "completed"
    assert summary["nodes"] > 0
    assert summary["elements"] > 0
    assert history[-1]["newton_iterations"] >= 1


def test_cropped_gear_abaqus_deck_prescribes_matching_rp_motion(tmp_path) -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)
    path = tmp_path / "cropped_gear_alignment.inp"

    _write_abaqus_alignment_deck(
        path,
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=2.0e-3,
        dt=1.0e-3,
        target_overclosure=3.0e-4,
        rotation_rate_z=2.0,
    )

    text = path.read_text(encoding="ascii")
    assert "G2_RP, 1, 6" in text
    assert "G1_RP, 4, 5" in text
    assert "*Boundary, amplitude=RAMP" in text
    assert "*Dynamic, application=MODERATE DISSIPATION" in text
    assert "G1_RP, 6, 6, 4.000000000000e-03" in text
    assert "S, E, LE" in text
    assert "CPRESS" not in text
    assert "COPEN" not in text


def test_cropped_gear_hard_contact_path_runs_one_implicit_step() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        max_iterations=4,
        automatic_increment=False,
    )

    assert len(history) == 1
    assert summary["status"] == "completed"
    assert summary["contact_mode"] == "hard"
    assert summary["hard_enforcement"] == "abaqus_standard_penalty"
    assert float(summary["hht_alpha"]) < 0.0
    assert summary["nodes"] > 0
    assert summary["elements"] > 0
    assert int(history[-1]["hard_contact_samples"]) > 0
    assert int(history[-1]["hard_active_set_converged"]) == 1
    assert "rp_force_norm" in history[-1]
    assert float(history[-1]["rp_force_norm"]) >= 0.0
    assert "opposing_rp_force_norm" in history[-1]
    assert float(history[-1]["opposing_rp_force_norm"]) >= 0.0
    assert history[-1]["rp_reaction_definition"] == "static_physical_constraint_residual"
    assert "rp1_hht_rp_force_norm" in history[-1]
    assert "rp1_dynamic_rp_force_norm" in history[-1]
    assert "rp1_inertia_rp_force_norm" in history[-1]
    assert "rp1_contact_rp_force_norm" in history[-1]
    assert "contact_multiplier_sum" in history[-1]
    assert "p95_equivalent_elastic_strain" in history[-1]
    assert float(history[-1]["p95_equivalent_elastic_strain"]) >= 0.0


def test_cropped_gear_hard_contact_supports_surface_patch_constraint_averaging() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        max_iterations=4,
        constraint_averaging="surface_patch",
        automatic_increment=False,
    )

    assert len(history) == 1
    assert summary["constraint_averaging"] == "surface_patch"
    assert int(history[-1]["hard_constraints"]) <= int(history[-1]["hard_contact_samples"])
    assert int(history[-1]["hard_active_set_converged"]) == 1


def test_cropped_gear_hard_contact_supports_element_pressure_smoothing() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        max_iterations=4,
        hard_enforcement="element_pressure_smoothing",
        pressure_smoothing_factor=4.0,
        automatic_increment=False,
    )

    assert len(history) == 1
    assert summary["hard_enforcement"] == "element_pressure_smoothing"
    assert float(summary["effective_hard_pressure_stiffness"]) > 5.0e9
    assert float(summary["contact_patch_min_edge_length"]) > 0.0


def test_cropped_gear_hard_contact_supports_abaqus_standard_penalty() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        max_iterations=4,
        hard_enforcement="abaqus_standard_penalty",
        automatic_increment=False,
    )

    assert len(history) == 1
    assert summary["hard_enforcement"] == "abaqus_standard_penalty"
    assert float(summary["effective_hard_pressure_stiffness"]) > 5.0e9
    assert int(summary["accepted_increment_count"]) == 1


def test_cropped_gear_abaqus_deck_can_use_hard_contact(tmp_path) -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)
    path = tmp_path / "cropped_gear_hard_alignment.inp"

    _write_abaqus_alignment_deck(
        path,
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=2.0e-3,
        dt=1.0e-3,
        target_overclosure=3.0e-4,
        rotation_rate_z=2.0,
        contact_mode="hard",
    )

    text = path.read_text(encoding="ascii")
    assert "*Surface Interaction, name=SFC_HARD" in text
    assert "*Surface Behavior, pressure-overclosure=HARD" in text
    assert "interaction=SFC_HARD, type=SURFACE TO SURFACE" in text
    assert "pressure-overclosure=LINEAR" not in text
