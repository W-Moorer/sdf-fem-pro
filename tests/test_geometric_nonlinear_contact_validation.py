from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_geometric_nonlinear_contact_validation import (  # noqa: E402
    _contact_cutback_enabled,
    _make_contact_geometry,
    _contact_model,
    _parse_calculix_stdout_diagnostics,
    contact_replay_metrics,
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
    assert "*contact print" in text
    assert "*el print, elset=elall" in text
    assert "\ns\n" in text
    assert "\ne\n" in text


def test_sfc_geometric_contact_history_activates_contact() -> None:
    model = _contact_model(resolution=1, duration=0.12, dt=0.004)
    rows, _, state = run_sfc_geometric_contact_history(model, cutback_policy="active_retry")

    assert any(int(row["active_contact_count"]) > 0 for row in rows)
    assert {row["contact_mode"] for row in rows} == {"persistent_calculix_c3d4_f2f"}
    assert {row["normal_force_source"] for row in rows} == {"calculix_c3d4_f2f_hard_linear"}
    assert any(row["generated_contact_spring_count"] != "" for row in rows)
    active_rows = [row for row in rows if int(row["active_contact_count"]) > 0]
    assert active_rows
    assert max(int(row["generated_contact_spring_count"]) for row in active_rows) == 2
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

    assert len(list(strict.samples(model.nodes))) == model.surface_faces.shape[0]
    assert len(list(three_point.samples(model.nodes))) == 3 * model.surface_faces.shape[0]


def test_contact_replay_metrics_uses_calculix_shell_offset_and_cnum_weight() -> None:
    model = _contact_model(resolution=1, duration=0.02, dt=0.004)
    x = model.nodes.copy()
    x[:, 2] -= 0.0505

    offset_metrics = contact_replay_metrics(model, x)
    midplane_metrics = contact_replay_metrics(model, x, plane_z=model.floor_z)

    assert offset_metrics["active_force_spring_count"] == 2
    assert offset_metrics["cnum_equivalent"] == 14
    assert offset_metrics["normal_force"] > midplane_metrics["normal_force"]
    assert offset_metrics["contact_energy"] > midplane_metrics["contact_energy"]


def test_parse_calculix_stdout_increment_diagnostics(tmp_path: Path) -> None:
    log = tmp_path / "ccx.log"
    log.write_text(
        "\n".join(
            [
                "increment 1 attempt 1",
                "increment size= 2.000000e-03",
                "Number of contact spring elements=14",
                "no convergence",
                "increment 1 attempt 2",
                "increment size= 1.000000e-03",
                "Number of contact spring elements=12",
                "restoring the elastic contact stifnesses to their original values",
            ]
        ),
        encoding="utf-8",
    )

    row = _parse_calculix_stdout_diagnostics(log)

    assert row["calculix_stdout_increment_count"] == 2
    assert row["calculix_stdout_max_attempt"] == 2
    assert row["calculix_stdout_min_increment_size"] == 0.001
    assert row["calculix_stdout_cutback_attempt_count"] == 1
    assert row["calculix_stdout_no_convergence_count"] == 1
    assert row["calculix_stdout_kscale_restore_count"] == 1
    assert row["calculix_stdout_max_contact_spring_elements"] == 14


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

    with outputs["alignment"].open(newline="", encoding="utf-8") as f:
        alignment_rows = list(csv.DictReader(f))
    assert alignment_rows
    assert alignment_rows[0]["diagnosis"] == "external_unavailable"

    with outputs["mesh"].open(newline="", encoding="utf-8") as f:
        mesh_rows = list(csv.DictReader(f))
    assert {int(row["resolution"]) for row in mesh_rows} == {1, 2, 3}

    with outputs["timestep"].open(newline="", encoding="utf-8") as f:
        timestep_rows = list(csv.DictReader(f))
    assert len(timestep_rows) == 3
