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
    assert "G1_RP, 6, 6, 4.000000000000e-03" in text
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
    )

    assert len(history) == 1
    assert summary["status"] == "completed"
    assert summary["contact_mode"] == "hard"
    assert summary["nodes"] > 0
    assert summary["elements"] > 0
    assert int(history[-1]["hard_contact_samples"]) > 0
    assert int(history[-1]["hard_active_set_converged"]) == 1


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
