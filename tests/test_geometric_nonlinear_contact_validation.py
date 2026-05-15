from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_geometric_nonlinear_contact_validation import (  # noqa: E402
    _contact_cutback_enabled,
    _make_contact_geometry,
    _contact_model,
    _calculix_iterations_per_increment,
    _parse_calculix_dat_contact_elements,
    _parse_calculix_dynamic_step_options,
    _parse_calculix_stdout_diagnostics,
    contact_lifecycle_output_diagnostics,
    contact_element_clearance_lifecycle_audit,
    contact_replay_metrics,
    hht_state_definition_diagnostics,
    hht_residual_tangent_diagnostics,
    one_step_calculix_state_diagnostics,
    run_sfc_geometric_contact_history,
    run_validation,
    write_calculix_contact_input_with_stress,
)


def test_calculix_contact_input_requests_contact_stress_output(tmp_path: Path) -> None:
    model = _contact_model(resolution=1, duration=0.02, dt=0.004)
    inp = tmp_path / "contact.inp"
    write_calculix_contact_input_with_stress(model, inp)

    text = inp.read_text(encoding="utf-8").lower()
    assert "*contact pair, interaction=contact,type=surface to surface" in text
    assert "*contact print, frequency=1\ncdis\ncstr\ncels" in text
    assert "*contact print, frequency=1, totals=only\ncnum\n" in text
    assert "*el print, elset=elall" in text
    assert "\ns\n" in text
    assert "\ne\n" in text


def test_parse_calculix_dat_per_contact_cdis_cstr_cels(tmp_path: Path) -> None:
    dat = tmp_path / "case.dat"
    dat.write_text(
        "\n".join(
            [
                " relative contact displacement (slave element+face,normal,tang1,tang2) for all contact elements and time 0.1000000E+00",
                "",
                "         4          1 -1.250000E-03  2.000000E-04 -3.000000E-04",
                " contact stress (slave element+face,press,tang1,tang2) for all contact elements and time 0.1000000E+00",
                "",
                "         4          1  2.500000E+01  1.000000E+00 -2.000000E+00",
                " contact spring energy (slave element+face,energy) for all contact elements and time 0.1000000E+00",
                "",
                "         4          1  7.812500E-04",
                " total contact spring energy for time 0.1000000E+00",
                "  7.812500E-04",
            ]
        ),
        encoding="utf-8",
    )

    parsed = _parse_calculix_dat_contact_elements(dat)

    row = parsed[0.1][(4, 1, 0)]
    assert row["calculix_clearance_normal"] == -1.25e-3
    assert row["calculix_clearance_tangential_1"] == 2.0e-4
    assert row["calculix_stress_normal"] == 25.0
    assert row["calculix_stress_tangential_2"] == -2.0
    assert row["calculix_contact_energy"] == 7.8125e-4


def test_contact_element_audit_compares_calculix_native_and_replay() -> None:
    model = _contact_model(resolution=1, duration=0.02, dt=0.004)
    x = model.nodes.copy()
    x[:, 2] -= 0.0505
    native_rows: list[dict[str, object]] = []
    run_sfc_geometric_contact_history(
        model,
        contact_element_audit_rows=native_rows,
        cutback_policy="calculix_direct",
    )
    element_id, face_number = model.slave_face_refs[0]
    face_number_int = int(face_number[1:])
    displacement = np.zeros_like(model.nodes)
    displacement[:, 2] -= 0.0505
    parsed = {
        0.004: {
            (int(element_id), face_number_int, 0): {
                "calculix_clearance_normal": -5.0e-4,
                "calculix_stress_normal": 10.0,
                "calculix_contact_energy": 1.0e-6,
            }
        }
    }

    rows, summary = contact_element_clearance_lifecycle_audit(
        model,
        calculix_dat_contact=parsed,
        sfc_native_rows=native_rows,
        calculix_displacements={0.004: displacement},
    )

    assert rows
    assert any(row["calculix_present"] == "true" for row in rows)
    assert any(row["calculix_displacement_replay_active_spring"] == "true" for row in rows)
    assert any(row["calculix_per_contact_output_available"] == "true" for row in summary)


def test_sfc_geometric_contact_history_activates_contact() -> None:
    model = _contact_model(resolution=1, duration=0.12, dt=0.004)
    rows, _, state = run_sfc_geometric_contact_history(model, cutback_policy="active_retry")

    assert any(int(row["active_contact_count"]) > 0 for row in rows)
    assert {row["contact_mode"] for row in rows} == {"persistent_calculix_c3d4_f2f"}
    assert {row["normal_force_source"] for row in rows} == {"calculix_c3d4_f2f_hard_linear"}
    assert any(row["generated_contact_spring_count"] != "" for row in rows)
    active_rows = [row for row in rows if int(row["active_contact_count"]) > 0]
    assert active_rows
    assert max(int(row["generated_contact_spring_count"]) for row in active_rows) == 14
    assert max(int(row["calculix_equivalent_contact_count"]) for row in active_rows) == 14
    assert all(row["contact_cutback_recommended"] in {"true", "false"} for row in rows)
    assert all(row["contact_convergence_reason"] for row in rows)
    assert any(int(row["cutback_retry_count"]) > 0 for row in rows)
    assert min(float(row["accepted_dt"]) for row in rows[1:]) < model.dt
    assert max(float(row["normal_force_proxy"]) for row in rows) > 0.0
    assert max(float(row["contact_energy_proxy"]) for row in rows) > 0.0
    assert max(int(row["newton_iterations"]) for row in rows) > 0
    assert state.von_mises.shape[0] == model.tet_elements.shape[0]


def test_default_geometric_contact_history_uses_calculix_direct_increment_policy() -> None:
    model = _contact_model(resolution=1, duration=0.12, dt=0.004)
    rows, _, _ = run_sfc_geometric_contact_history(model)

    assert {row["cutback_policy"] for row in rows} == {"calculix_direct"}
    assert all(int(row["cutback_retry_count"]) == 0 for row in rows)
    assert abs(min(float(row["accepted_dt"]) for row in rows[1:]) - model.dt) < 1.0e-12


def test_contact_geometry_modes_expose_strict_and_three_point_discretizations() -> None:
    model = _contact_model(resolution=1, duration=0.02, dt=0.004)
    strict, _ = _make_contact_geometry(model, "calculix_c3d4_f2f")
    three_point, _ = _make_contact_geometry(model, "plane")

    assert len(list(strict.samples(model.nodes))) == 7 * model.surface_faces.shape[0]
    assert len(list(three_point.samples(model.nodes))) == 3 * model.surface_faces.shape[0]


def test_contact_replay_metrics_uses_calculix_shell_offset_and_cnum_weight() -> None:
    model = _contact_model(resolution=1, duration=0.02, dt=0.004)
    x = model.nodes.copy()
    x[:, 2] -= 0.0505

    offset_metrics = contact_replay_metrics(model, x)
    midplane_metrics = contact_replay_metrics(model, x, plane_z=model.floor_z)

    assert offset_metrics["active_force_spring_count"] == 14
    assert offset_metrics["cnum_equivalent"] == 14
    assert offset_metrics["normal_force"] > midplane_metrics["normal_force"]
    assert offset_metrics["contact_energy"] > midplane_metrics["contact_energy"]


def test_one_step_calculix_state_diagnostics_decomposes_terms() -> None:
    model = _contact_model(resolution=1, duration=0.008, dt=0.004)
    u1 = np.zeros_like(model.nodes)
    u2 = np.zeros_like(model.nodes)
    u2[:, 2] -= 0.0505
    calculix_rows = [
        {"time": 0.004, "normal_force_proxy": 0.0, "calculix_contact_count": 0},
        {"time": 0.008, "normal_force_proxy": 1.0, "calculix_contact_count": 14},
    ]

    rows = one_step_calculix_state_diagnostics(
        model,
        calculix_rows,
        {0.004: u1, 0.008: u2},
    )

    assert len(rows) == 2
    assert rows[0]["calculix_internal_force_available"] == "false"
    assert rows[0]["calculix_effective_tangent_available"] == "false"
    assert rows[-1]["dominant_one_step_difference_source"] in {
        "mass_term",
        "static_residual",
        "internal_force",
        "contact_force",
        "acceleration",
    }
    assert float(rows[-1]["sfc_on_calculix_internal_force_norm"]) >= 0.0
    assert float(rows[-1]["sfc_hht_effective_residual_norm_at_calculix_state"]) >= 0.0


def test_hht_state_definition_diagnostics_reports_precision_and_initial_state() -> None:
    model = _contact_model(resolution=1, duration=0.008, dt=0.004)
    u1 = np.zeros_like(model.nodes)
    u2 = np.zeros_like(model.nodes)
    u2[:, 2] -= 0.0505
    one_step = one_step_calculix_state_diagnostics(
        model,
        [{"time": 0.004, "normal_force_proxy": 0.0}, {"time": 0.008, "normal_force_proxy": 1.0}],
        {0.004: u1, 0.008: u2},
    )

    row = hht_state_definition_diagnostics(model, {0.004: u1, 0.008: u2}, one_step)

    assert row["calculix_completed"] == "true"
    assert row["source_beta_formula"] == "nonlingeo.c uses beta=(1-alpha)^2/4"
    assert float(row["sfc_beta"]) > 0.0
    assert float(row["sfc_gamma"]) > 0.0
    assert float(row["mass_rel_error"]) < 1.0e-12
    assert float(row["dat_precision_acceleration_uncertainty_estimate"]) >= 0.0
    assert row["diagnosis"] in {
        "precontact_hht_state_mismatch",
        "dat_displacement_precision_can_affect_reconstructed_acceleration",
        "contact_phase_effective_residual_mismatch_not_explained_by_dat_precision",
        "hht_state_definitions_consistent_for_sampled_rows",
    }


def test_parse_calculix_stdout_increment_diagnostics(tmp_path: Path) -> None:
    log = tmp_path / "ccx.log"
    log.write_text(
        "\n".join(
            [
                "increment 1 attempt 1",
                "increment size= 2.000000e-03",
                "Number of contact spring elements=14",
                " iteration 1",
                "no convergence",
                " iteration 2",
                " convergence",
                "increment 1 attempt 2",
                "increment size= 1.000000e-03",
                "Number of contact spring elements=12",
                " iteration 1",
                " convergence",
                "restoring the elastic contact stifnesses to their original values",
                "Adaption of the energy residual in persistent contact,",
                "convergence; new increment size is forced to 1.000000e-03",
            ]
        ),
        encoding="utf-8",
    )

    row = _parse_calculix_stdout_diagnostics(log)

    assert row["calculix_stdout_increment_count"] == 2
    assert row["calculix_stdout_max_attempt"] == 2
    assert row["calculix_stdout_min_increment_size"] == 0.001
    assert row["calculix_stdout_convergence_count"] == 2
    assert row["calculix_stdout_total_newton_iterations"] == 3
    assert row["calculix_stdout_max_iterations_per_increment"] == 2
    assert row["calculix_stdout_mean_iterations_per_increment"] == 1.5
    assert row["calculix_stdout_cutback_attempt_count"] == 1
    assert row["calculix_stdout_no_convergence_count"] == 1
    assert row["calculix_stdout_kscale_restore_count"] == 1
    assert row["calculix_stdout_contact_energy_stabilization_count"] == 1
    assert row["calculix_stdout_forced_increment_size_count"] == 1
    assert row["calculix_stdout_max_contact_spring_elements"] == 14


def test_parse_calculix_dynamic_step_options(tmp_path: Path) -> None:
    inp = tmp_path / "case.inp"
    inp.write_text("*step, nlgeom, inc=100\n*dynamic, direct, alpha=-0.05\n", encoding="utf-8")

    row = _parse_calculix_dynamic_step_options(inp)

    assert row["calculix_dynamic_direct"] == "true"
    assert row["calculix_nlgeom"] == "true"
    assert row["calculix_alpha"] == -0.05


def test_calculix_iterations_per_increment_counts_blocks() -> None:
    text = "\n".join(
        [
            "increment 1 attempt 1 ",
            " iteration 1",
            " no convergence",
            " iteration 2",
            " convergence",
            "increment 2 attempt 1 ",
            " iteration 1",
            " convergence",
        ]
    )

    assert _calculix_iterations_per_increment(text) == [2, 1]


def test_contact_lifecycle_output_diagnostics_detects_cnum_trajectory_difference() -> None:
    model = _contact_model(resolution=1, duration=0.12, dt=0.002)
    calculix_rows = [
        {"time": 0.0, "calculix_contact_count": 0, "normal_force_proxy": 0.0, "contact_energy_proxy": 0.0},
        {"time": 0.096, "calculix_contact_count": 14, "normal_force_proxy": 1.0, "contact_energy_proxy": 0.1},
        {"time": 0.104, "calculix_contact_count": 12, "normal_force_proxy": 3.0, "contact_energy_proxy": 0.3},
        {"time": 0.108, "calculix_contact_count": 14, "normal_force_proxy": 2.0, "contact_energy_proxy": 0.2},
    ]
    sfc_rows = [
        {"time": 0.0, "contact_mode": "persistent_calculix_c3d4_f2f", "calculix_equivalent_contact_count": 0, "normal_force_proxy": 0.0, "contact_energy_proxy": 0.0},
        {"time": 0.096, "contact_mode": "persistent_calculix_c3d4_f2f", "calculix_equivalent_contact_count": 14, "normal_force_proxy": 1.0, "contact_energy_proxy": 0.1},
        {"time": 0.104, "contact_mode": "persistent_calculix_c3d4_f2f", "calculix_equivalent_contact_count": 0, "normal_force_proxy": 2.0, "contact_energy_proxy": 0.2},
        {"time": 0.108, "contact_mode": "persistent_calculix_c3d4_f2f", "calculix_equivalent_contact_count": 14, "normal_force_proxy": 4.0, "contact_energy_proxy": 0.4},
    ]
    contact_disp = np.zeros_like(model.nodes)
    contact_disp[:, 2] -= 0.0505
    calculix_displacements = {
        0.0: np.zeros_like(model.nodes),
        0.096: contact_disp,
        0.104: contact_disp,
        0.108: contact_disp,
    }

    row = contact_lifecycle_output_diagnostics(model, calculix_rows, sfc_rows, calculix_displacements)

    assert row["calculix_cnum_sequence"] == "0->14->12->14"
    assert row["sfc_cnum_sequence"] == "0->14->0->14"
    assert row["calculix_displacement_replay_cnum_sequence"] == "0->14"
    assert row["cnum_first_mismatch_time"] == 0.104
    assert row["cnum_max_abs_error"] == 12.0
    assert row["calculix_displacement_replay_cnum_max_abs_error"] == 2.0
    assert row["trajectory_difference_explains_cnum_gap"] == "true"
    assert row["calculix_release_or_reactivation_observed"] == "true"
    assert abs(float(row["force_peak_time_abs_error"]) - 0.004) < 1.0e-12
    assert abs(float(row["energy_peak_time_abs_error"]) - 0.004) < 1.0e-12
    assert row["diagnosis"] == "contact_lifecycle_trajectory_difference_observed"


def test_calculix_direct_cutback_policy_disables_sfc_retry() -> None:
    assert _contact_cutback_enabled("persistent_calculix_c3d4_f2f", "active_retry")
    assert not _contact_cutback_enabled("persistent_calculix_c3d4_f2f", "calculix_direct")
    assert not _contact_cutback_enabled("calculix_c3d4_f2f", "active_retry")


def test_contact_validation_quick_skip_calculix_outputs(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path, quick=True, skip_calculix=True)

    for path in outputs.values():
        assert path.exists()

    with outputs["claims"].open(newline="", encoding="utf-8") as f:
        claims = {row["claim"]: row for row in csv.DictReader(f)}

    assert claims["block_plane_geometric_contact_calculix_comparison_available"]["supported"] == "false"
    assert claims["contact_mesh_convergence_trend_available"]["supported"] == "true"
    assert claims["contact_timestep_convergence_trend_available"]["supported"] == "true"
    assert claims["paraview_stress_cloud_comparison_available"]["supported"] == "false"
    assert claims["calculix_contact_force_law_replay_diagnostics_available"]["supported"] == "false"
    assert claims["hht_residual_tangent_trajectory_diagnostics_available"]["supported"] == "true"
    assert claims["calculix_contact_lifecycle_output_diagnostics_available"]["supported"] == "false"
    assert claims["calculix_mechanics_increment_acceptance_diagnostics_available"]["supported"] == "false"
    assert claims["calculix_state_one_step_mechanics_diagnostics_available"]["supported"] == "false"
    assert claims["hht_newmark_state_definition_precision_diagnostics_available"]["supported"] == "false"
    assert claims["calculix_per_contact_element_clearance_lifecycle_audit_available"]["supported"] == "false"

    with outputs["alignment"].open(newline="", encoding="utf-8") as f:
        alignment_rows = list(csv.DictReader(f))
    assert alignment_rows
    assert alignment_rows[0]["diagnosis"] == "external_unavailable"

    with outputs["lifecycle"].open(newline="", encoding="utf-8") as f:
        lifecycle_rows = list(csv.DictReader(f))
    assert lifecycle_rows
    assert lifecycle_rows[0]["diagnosis"] == "external_unavailable"
    assert lifecycle_rows[0]["sfc_cnum_sequence"]

    with outputs["mechanics"].open(newline="", encoding="utf-8") as f:
        mechanics_rows = list(csv.DictReader(f))
    assert mechanics_rows
    assert mechanics_rows[0]["diagnosis"] == "external_unavailable"

    with outputs["one_step"].open(newline="", encoding="utf-8") as f:
        one_step_rows = list(csv.DictReader(f))
    assert one_step_rows
    assert one_step_rows[0]["diagnosis"] == "external_unavailable"

    with outputs["hht_state"].open(newline="", encoding="utf-8") as f:
        hht_state_rows = list(csv.DictReader(f))
    assert hht_state_rows
    assert hht_state_rows[0]["diagnosis"] == "external_unavailable"

    with outputs["mesh"].open(newline="", encoding="utf-8") as f:
        mesh_rows = list(csv.DictReader(f))
    assert {int(row["resolution"]) for row in mesh_rows} == {1, 2, 3}

    with outputs["timestep"].open(newline="", encoding="utf-8") as f:
        timestep_rows = list(csv.DictReader(f))
    assert len(timestep_rows) == 3

    with outputs["hht"].open(newline="", encoding="utf-8") as f:
        hht_rows = list(csv.DictReader(f))
    assert hht_rows
    assert float(hht_rows[0]["effective_tangent_directional_fd_rel_error"]) < 5.0e-5
    assert float(hht_rows[0]["contact_residual_tangent_directional_fd_rel_error"]) < 5.0e-5


def test_hht_residual_tangent_diagnostics_probe_contact_sign_convention() -> None:
    model = _contact_model(resolution=1, duration=0.12, dt=0.004)
    rows = hht_residual_tangent_diagnostics(model)

    assert rows
    row = rows[0]
    assert row["calculix_calcresidual_sign_convention"] == "sfc_R_is_negative_of_CalculiX_rhs_b_with_fextini_minus_fini_history"
    assert int(row["active_contact_count"]) > 0
    assert float(row["effective_tangent_directional_fd_rel_error"]) < 5.0e-5
    assert float(row["static_tangent_directional_fd_rel_error"]) < 5.0e-5
    assert float(row["contact_residual_tangent_directional_fd_rel_error"]) < 5.0e-5
    assert float(row["previous_static_residual_update_rel_error"]) < 5.0e-8
