from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.prepare_flexible_gear_source_penalty_deck import prepare_source_penalty_deck_text
from validation.run_flexible_gear_source_penalty_abaqus import write_source_penalty_summary
from validation.run_flexible_gear_full_lagrangian_sdf_comparison import run_full_gear


def test_prepare_source_penalty_deck_uses_standard_linear_penalty_and_strided_output() -> None:
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
    assert "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE" in text
    assert "mechanical constraint=PENALTY" not in text
    assert "1.000000000000e-05,1.000000000000e-04,1.000000000000e-05,1.000000000000e-05" in text
    assert "*Output, field, variable=PRESELECT, frequency=5" in text
    assert "*Output, history, variable=PRESELECT, frequency=5" in text


def test_prepare_source_penalty_deck_can_preserve_automatic_increment_bounds() -> None:
    source = "\n".join(
        [
            "*Surface Behavior, pressure-overclosure=HARD",
            "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE",
            "S2, S1",
            "*Dynamic",
            "1e-05,0.05,1e-10,5e-05",
        ]
    )

    text = prepare_source_penalty_deck_text(
        source,
        pressure_stiffness=5.0e9,
        frame_stride=5,
        dt=1.0e-5,
        duration=1.0e-4,
        fixed_increment=False,
    )

    assert "1.000000000000e-05,1.000000000000e-04,1e-10,5e-05" in text


def test_prepare_source_penalty_deck_can_require_source_timing() -> None:
    source = "\n".join(
        [
            "*Surface Behavior, pressure-overclosure=HARD",
            "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE",
            "S2, S1",
            "*Dynamic",
            "1e-05,0.05,1e-10,5e-05",
        ]
    )

    text = prepare_source_penalty_deck_text(
        source,
        pressure_stiffness=5.0e9,
        frame_stride=2,
        dt=1.0e-5,
        duration=0.05,
        require_source_timing=True,
    )

    assert "1.000000000000e-05,5.000000000000e-02,1.000000000000e-05,1.000000000000e-05" in text


def test_prepare_source_penalty_deck_rejects_short_duration_when_source_timing_required() -> None:
    source = "\n".join(
        [
            "*Surface Behavior, pressure-overclosure=HARD",
            "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE",
            "S2, S1",
            "*Dynamic",
            "1e-05,0.05,1e-10,5e-05",
        ]
    )

    import pytest

    with pytest.raises(ValueError, match="requested duration"):
        prepare_source_penalty_deck_text(
            source,
            pressure_stiffness=5.0e9,
            frame_stride=2,
            dt=1.0e-5,
            duration=0.002,
            require_source_timing=True,
        )


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


def test_prepare_source_penalty_deck_can_write_explicit_style_contact_pair_parameter() -> None:
    source = "\n".join(
        [
            "*Surface Behavior, pressure-overclosure=HARD",
            "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE",
            "S2, S1",
        ]
    )

    text = prepare_source_penalty_deck_text(
        source,
        pressure_stiffness=5.0e9,
        frame_stride=1,
        contact_pair_penalty_parameter=True,
    )

    assert "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE, mechanical constraint=PENALTY" in text


def test_source_penalty_summary_states_external_validation_only(tmp_path: Path) -> None:
    path = tmp_path / "summary.md"

    write_source_penalty_summary(
        path,
        {
            "pressure_stiffness": 5.0e9,
            "vtk_frame_stride": 5,
            "abaqus_analysis_wall_seconds": 1.5,
            "abaqus_reported_wall_seconds": 1.25,
            "abaqus_vtk_manifest": "abaqus_manifest.csv",
            "abaqus_vtk_frame_count": 3,
            "abaqus_vtk_export_frame_stride": 1,
        },
    )

    text = path.read_text(encoding="utf-8")
    assert "validation-only" in text
    assert "frictionless linear penalty pressure-overclosure" in text
    assert "VTK export frame stride: `1`" in text


def test_full_gear_runner_exposes_hht_alpha_parameter(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_parse(_source: Path):
        class Model:
            young = 1.0
            poisson = 0.25
            density = 1.0
            dynamic_duration = 1.0
            dynamic_initial_dt = 0.1
            dynamic_min_dt = 0.1
            dynamic_max_dt = 0.1
            contact_pressure_overclosure = "LINEAR"
            gear1_angular_velocity_z = 2.0
            gear2_torque_z = 3.0

        return Model()

    def fake_build(*_args, **_kwargs):
        class Patch:
            nodes = __import__("numpy").zeros((1, 3))
            elements = __import__("numpy").zeros((0, 4), dtype=int)
            contact_faces = __import__("numpy").zeros((0, 3), dtype=int)
            support_nodes = __import__("numpy").zeros((1,), dtype=int)
            rp = __import__("numpy").zeros(3)

        class Pair:
            gear1 = Patch()
            gear2 = Patch()
            initial_patch_gap = 0.0

        return Pair()

    def fake_solve(*_args, **kwargs):
        captured["hht_alpha"] = float(kwargs["hht_alpha"])
        captured["tet4_mass_kind"] = str(kwargs["tet4_mass_kind"])
        captured["history_frame_stride"] = int(kwargs["history_frame_stride"])
        captured["source_stress_postprocess"] = str(kwargs["source_stress_postprocess"])
        captured["source_checkpoint_stride"] = int(kwargs["source_checkpoint_stride"])
        return (
            [{"time": 0.1}],
            {
                "status": "completed",
                "nodes": 2,
                "elements": 0,
                "gear1_contact_faces": 0,
                "gear2_contact_faces": 0,
                "gear1_support_nodes": 1,
                "gear2_support_nodes": 1,
                "sfc_wall_seconds": 0.0,
            },
        )

    monkeypatch.setattr("validation.run_flexible_gear_full_lagrangian_sdf_comparison.parse_gear_input", fake_parse)
    monkeypatch.setattr("validation.run_flexible_gear_full_lagrangian_sdf_comparison.build_full_active_pair", fake_build)
    monkeypatch.setattr("validation.run_flexible_gear_full_lagrangian_sdf_comparison.solve_sfc_source_drive_pair", fake_solve)
    monkeypatch.setattr(
        "validation.run_flexible_gear_full_lagrangian_sdf_comparison._write_abaqus_alignment_deck",
        lambda *_args, **_kwargs: None,
    )

    run_full_gear(
        source=tmp_path / "dummy.inp",
        out_dir=tmp_path,
        active_faces_per_body=0,
        active_patch_radius_factor=1.0,
        duration=0.1,
        dt=0.1,
        target_overclosure=0.0,
        rotation_rate_z=0.0,
        pressure_stiffness=1.0,
        hard_max_iterations=1,
        contact_mode="penalty",
        run_abaqus=False,
        drive_mode="source_inp",
        hht_alpha=-0.05,
        tet4_mass_kind="consistent",
        history_frame_stride=3,
        source_stress_postprocess="finite_stvk_visual",
        source_checkpoint_path=tmp_path / "checkpoint.npz",
        resume_source_checkpoint=True,
        source_checkpoint_stride=7,
    )

    assert captured["hht_alpha"] == -0.05
    assert captured["tet4_mass_kind"] == "consistent"
    assert captured["history_frame_stride"] == 3
    assert captured["source_stress_postprocess"] == "finite_stvk_visual"
    assert captured["source_checkpoint_stride"] == 7
