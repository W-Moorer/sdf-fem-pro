"""Geometric nonlinear contact validation against CalculiX.

This runner is deliberately separate from the core solver.  It uses a
clean-room CalculiX-aligned StVK geometric nonlinear dynamics backend and
compares a block-plane dynamic contact case against CalculiX contact output
when ccx is available.  The comparison records contact activation,
penetration, normal reaction, contact energy, center-of-mass motion, rebound
height, contact-zone stress, mesh/time-step trends, and ParaView VTK stress
clouds.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_calculix_drop_impact_comparison import (  # noqa: E402
    CALCULIX_MASTER_SURFACE_OFFSET,
    DropModel,
    _calculix_version,
    _contact_plane_z,
    _first_contact_time,
    _history_from_displacements,
    _mass_weighted_center_z,
    _mass_weighted_velocity_z,
    _parse_calculix_dat_contact_totals,
    _rebound_physicality,
    _surface_face_gaps_to_contact_plane,
    _total_mass,
    _wsl_path,
    build_drop_model,
    calculix_available,
    write_calculix_input,
)
from validation.run_geometric_nonlinear_acceptance import (  # noqa: E402
    _parse_calculix_dat_displacements_strict,
    _parse_final_stress_voigt,
    _voigt_von_mises,
)
from validation.calculix_f2f_contact import (  # noqa: E402
    CalculixContactConvergenceHeuristic,
    CalculixC3D4FaceToFacePlaneContactGeometry,
    CalculixC3D4FaceToFaceSDFContactGeometry,
    PersistentCalculixC3D4FaceToFacePlaneContactGeometry,
    PersistentCalculixC3D4FaceToFaceSDFContactGeometry,
    calculix_equivalent_contact_element_count,
)
from sfc.contact import DynamicSurfaceSDFContactGeometry, UniformTriangleAABBHash  # noqa: E402
from sfc.fem.calculix_aligned import (  # noqa: E402
    ContactGeometry,
    MechanicsModel,
    MechanicsState,
    PlaneContactGeometry,
    StepDiagnostics,
    assemble_contact_response,
    evaluate_state,
    hht_newmark_parameters,
    hht_step,
    initial_state,
    static_residual_and_tangent,
)
from validation.run_geometric_nonlinear_vtk import (  # noqa: E402
    NonlinearState,
    _write_vtk,
)

Row = dict[str, Any]


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _contact_model(*, resolution: int, duration: float, dt: float, output_frequency: int = 1) -> DropModel:
    return build_drop_model(
        quick=True,
        case="block_drop",
        resolution=resolution,
        duration=duration,
        dt=dt,
        output_frequency=output_frequency,
        initial_velocity_z=0.0,
        gravity=9.81,
        contact_stiffness_override=2.0e4,
        hht_alpha=-0.05,
    )


def write_calculix_contact_input_with_stress(model: DropModel, path: Path) -> None:
    """Write the standard CalculiX contact input plus element stress output."""

    write_calculix_input(model, path)
    text = path.read_text(encoding="utf-8")
    stress_block = "\n".join(
        [
            f"*el print, elset=elall, frequency={max(1, int(model.output_frequency))}",
            "s",
            "e",
        ]
    )
    text = text.replace("*endstep", stress_block + "\n*endstep")
    path.write_text(text, encoding="utf-8")


def run_calculix_contact_with_stress(
    model: DropModel,
    out_dir: Path,
    *,
    timeout: int = 300,
) -> tuple[list[Row], np.ndarray, np.ndarray, Row, dict[float, np.ndarray]]:
    """Run CalculiX contact and return history, final displacement, stress, command."""

    if not calculix_available():
        return [], np.zeros_like(model.nodes), np.zeros((model.tet_elements.shape[0], 6), dtype=float), {
            "case": model.case,
            "resolution": model.resolution,
            "completed": "false",
            "return_code": "not_available",
            "command": "",
        }, {}
    run_dir = out_dir / "calculix_runs" / f"{model.case}_r{model.resolution}_dt{model.dt:g}".replace(".", "p")
    run_dir.mkdir(parents=True, exist_ok=True)
    case_name = f"{model.case}_r{model.resolution}"
    inp = run_dir / f"{case_name}.inp"
    write_calculix_contact_input_with_stress(model, inp)
    command = f"cd {_wsl_path(run_dir)} && ccx {case_name}"
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
    dat = run_dir / f"{case_name}.dat"
    command_row = {
        "case": model.case,
        "resolution": model.resolution,
        "external_solver": "CalculiX",
        "external_solver_version": _calculix_version(),
        "command": f"wsl --exec bash -lc \"{command}\"",
        "input_file": str(inp.relative_to(out_dir)),
        "dat_file": str(dat.relative_to(out_dir)) if dat.exists() else "",
        "stdout_log": str((run_dir / "calculix_stdout.log").relative_to(out_dir)),
        "stderr_log": str((run_dir / "calculix_stderr.log").relative_to(out_dir)),
        "return_code": proc.returncode,
        "completed": str(proc.returncode == 0 and dat.exists()).lower(),
    }
    if not dat.exists():
        return [], np.zeros_like(model.nodes), np.zeros((model.tet_elements.shape[0], 6), dtype=float), command_row, {}
    displacements = _parse_calculix_dat_displacements_strict(dat, model.node_ids)
    contact_totals = _parse_calculix_dat_contact_totals(dat)
    history = _history_from_displacements("calculix", model, displacements, contact_totals)
    final_u = displacements[max(displacements)]
    final_stress = _parse_final_stress_voigt(dat, model.tet_elements.shape[0])
    return history, final_u, final_stress, command_row, displacements


def run_sfc_geometric_contact_history(
    model: DropModel,
    *,
    contact_mode: str = "persistent_calculix_c3d4_f2f",
    cutback_policy: str = "calculix_direct",
    source: str = "sfc_geometric_nonlinear",
) -> tuple[list[Row], np.ndarray, NonlinearState]:
    """Run the clean-room CalculiX-aligned StVK contact backend."""

    mechanics = MechanicsModel.from_tet4_mesh(
        model.nodes,
        model.tet_elements,
        E=model.E,
        nu=model.nu,
        density=model.density,
    )
    contact, details = _make_contact_geometry(model, contact_mode)
    state, previous_static_residual = initial_state(
        mechanics,
        contact,
        gravity=model.gravity,
        initial_velocity=(0.0, 0.0, model.initial_velocity_z),
    )
    diagnostics = evaluate_state(mechanics, state, contact, gravity=model.gravity, assemble_tangent=True)
    rows: list[Row] = []
    contact_heuristic = CalculixContactConvergenceHeuristic()
    state.time = 0.0
    initial_record = contact_heuristic.update(
        iteration=0,
        active_count=_contact_count_for_heuristic(contact, diagnostics),
        residual_norm=float(diagnostics.newton_residual_norm),
    )
    rows.append(
        _sfc_history_row(
            model,
            state,
            diagnostics,
            contact,
            source=source,
            contact_mode=contact_mode,
            convergence_record=initial_record,
            details=details,
            accepted_dt=0.0,
            cutback_retry_count=0,
            cutback_limited=False,
            cutback_policy=cutback_policy,
        )
    )
    current_time = 0.0
    step = 0
    min_dt = max(float(model.dt) / 16.0, 1.0e-8)
    max_retries = 6
    while current_time < float(model.total_time) - 1.0e-12:
        nominal_dt = min(float(model.dt), float(model.total_time) - current_time)
        trial_dt = nominal_dt
        retry_count = 0
        while True:
            lifecycle_snapshot = _snapshot_contact_lifecycle(contact)
            heuristic_snapshot = _snapshot_contact_heuristic(contact_heuristic)
            _set_contact_cutback_retry(contact, retry_count > 0)
            trial_state, trial_previous_static, trial_diagnostics = hht_step(
                mechanics,
                state,
                previous_static_residual,
                contact,
                dt=trial_dt,
                gravity=model.gravity,
                alpha=model.hht_alpha,
                max_iterations=12,
                tolerance=1.0e-10,
            )
            trial_time = current_time + trial_dt
            trial_state.time = float(trial_time)
            record = contact_heuristic.update(
                iteration=step + 1,
                active_count=_contact_count_for_heuristic(contact, trial_diagnostics),
                residual_norm=float(trial_diagnostics.newton_residual_norm),
            )
            should_retry = (
                _contact_cutback_enabled(contact_mode, cutback_policy)
                and record.recommended_cutback
                and retry_count < max_retries
                and trial_dt > min_dt * 1.000001
            )
            if should_retry:
                _restore_contact_lifecycle(contact, lifecycle_snapshot)
                _restore_contact_heuristic(contact_heuristic, heuristic_snapshot)
                trial_dt = max(min_dt, 0.5 * trial_dt)
                retry_count += 1
                continue

            cutback_limited = bool(record.recommended_cutback and retry_count >= max_retries)
            state = trial_state
            previous_static_residual = trial_previous_static
            diagnostics = trial_diagnostics
            current_time = trial_time
            step += 1
            _set_contact_cutback_retry(contact, False)
            rows.append(
                _sfc_history_row(
                    model,
                    state,
                    diagnostics,
                    contact,
                    source=source,
                    contact_mode=contact_mode,
                    convergence_record=record,
                    details=details,
                    accepted_dt=trial_dt,
                    cutback_retry_count=retry_count,
                    cutback_limited=cutback_limited,
                    cutback_policy=cutback_policy,
                )
            )
            break
    return rows, state.x, _vtk_state_from_backend(state, diagnostics)


def _sfc_history_row(
    model: DropModel,
    state: MechanicsState,
    diagnostics: StepDiagnostics,
    contact: ContactGeometry,
    *,
    source: str,
    contact_mode: str,
    convergence_record: Any,
    details: str,
    accepted_dt: float,
    cutback_retry_count: int,
    cutback_limited: bool,
    cutback_policy: str,
) -> Row:
    centroid_gaps = _surface_face_gaps_to_contact_plane(model, state.x)
    generated_count = _generated_contact_count(contact)
    cnum_equivalent = _calculix_equivalent_contact_count(contact)
    return {
        "case": model.case,
        "resolution": model.resolution,
        "source": source,
        "contact_mode": contact_mode,
        "time": float(state.time),
        "accepted_dt": float(accepted_dt),
        "nominal_dt": float(model.dt),
        "cutback_policy": cutback_policy,
        "cutback_retry_count": int(cutback_retry_count),
        "cutback_limited": str(bool(cutback_limited)).lower(),
        "z_cm": _mass_weighted_center_z(model, state.x),
        "v_cm_z": _mass_weighted_velocity_z(model, state.v.reshape(-1)),
        "min_gap": float(np.min(centroid_gaps)),
        "quadrature_min_gap": diagnostics.contact.min_gap,
        "max_penetration": max(float(diagnostics.contact.max_penetration), float(np.max(np.maximum(-centroid_gaps, 0.0)))),
        "active_contact_count": diagnostics.contact.active_count,
        "generated_contact_spring_count": generated_count,
        "calculix_equivalent_contact_count": cnum_equivalent,
        "normal_force_proxy": diagnostics.contact.normal_force,
        "sfc_floor_rf_z_equivalent": diagnostics.contact.normal_force,
        "normal_force_source": _normal_force_source(contact_mode),
        "calculix_floor_rf_z": "",
        "calculix_contact_count": "",
        "kinetic_energy_proxy": diagnostics.kinetic_energy,
        "strain_energy": diagnostics.internal.strain_energy,
        "gravitational_potential_energy": diagnostics.gravitational_energy,
        "contact_energy_proxy": diagnostics.contact.energy,
        "total_mechanical_energy_proxy": diagnostics.total_energy,
        "max_contact_zone_von_mises": _contact_zone_max_von_mises(model, diagnostics.internal.von_mises),
        "newton_iterations": diagnostics.newton_iterations,
        "newton_residual_norm": diagnostics.newton_residual_norm,
        "contact_cutback_recommended": str(convergence_record.recommended_cutback).lower(),
        "contact_convergence_reason": convergence_record.reason,
        "details": details,
    }


def contact_replay_metrics(model: DropModel, x_current: np.ndarray, *, plane_z: float | None = None) -> Row:
    """Return hard-linear contact metrics replayed on a current geometry."""

    z = _contact_plane_z(model) if plane_z is None else float(plane_z)
    tri = np.asarray(x_current, dtype=float)[model.surface_faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    gaps = np.mean(tri, axis=1)[:, 2] - z
    penetration = np.maximum(-gaps, 0.0)
    active = int(np.count_nonzero(penetration > 0.0))
    force = float(model.contact_stiffness * np.sum(areas * penetration))
    energy = float(0.5 * model.contact_stiffness * np.sum(areas * penetration * penetration))
    return {
        "min_gap": float(np.min(gaps)) if gaps.size else 0.0,
        "max_penetration": float(np.max(penetration)) if penetration.size else 0.0,
        "active_force_spring_count": active,
        "cnum_equivalent": active * 7,
        "normal_force": force,
        "contact_energy": energy,
    }


def contact_alignment_diagnostics(
    model: DropModel,
    calculix_rows: list[Row],
    sfc_rows: list[Row],
    calculix_displacements: dict[float, np.ndarray],
    command_row: Row,
    out_dir: Path,
) -> Row:
    """Diagnose whether RF/CELS differences are force-law or trajectory driven."""

    stdout_rel = str(command_row.get("stdout_log", ""))
    stdout_path = out_dir / stdout_rel if stdout_rel else None
    stdout = _parse_calculix_stdout_diagnostics(stdout_path)
    base: Row = {
        "case": model.case,
        "resolution": model.resolution,
        "contact_mode": sfc_rows[0].get("contact_mode", "") if sfc_rows else "",
        "calculix_completed": str(bool(calculix_rows and calculix_displacements)).lower(),
        "shell_master_offset": CALCULIX_MASTER_SURFACE_OFFSET,
        "contact_stiffness": model.contact_stiffness,
        **stdout,
    }
    if not calculix_rows or not calculix_displacements:
        base.update(
            {
                "force_law_mean_rel_error": "",
                "force_law_max_rel_error": "",
                "energy_law_mean_rel_error": "",
                "energy_law_max_rel_error": "",
                "cnum_equivalent_max_abs_error": "",
                "peak_rf_force_replay_rel_error": "",
                "peak_cels_energy_replay_rel_error": "",
                "peak_midplane_force_fraction": "",
                "sfc_min_accepted_dt": _min_float(sfc_rows, "accepted_dt") or "",
                "sfc_rows_with_retry": sum(1 for row in sfc_rows if int(row.get("cutback_retry_count", 0)) > 0),
                "sfc_max_retry_count": max((int(row.get("cutback_retry_count", 0)) for row in sfc_rows), default=0),
                "diagnosis": "external_unavailable",
            }
        )
        return base

    force_errors: list[float] = []
    energy_errors: list[float] = []
    cnum_errors: list[float] = []
    peak_rf_tuple: tuple[float, float] | None = None
    peak_cels_tuple: tuple[float, float] | None = None
    peak_offset_force = 0.0
    peak_midplane_force = 0.0
    for row in calculix_rows:
        time = float(row["time"])
        disp_time = min(calculix_displacements, key=lambda value: abs(value - time))
        x_current = model.nodes + calculix_displacements[disp_time]
        replay = contact_replay_metrics(model, x_current)
        midplane = contact_replay_metrics(model, x_current, plane_z=model.floor_z)
        rf = _optional_float(row.get("normal_force_proxy", ""))
        cels = _optional_float(row.get("contact_energy_proxy", ""))
        cnum = _optional_float(row.get("calculix_contact_count", ""))
        if rf is not None and abs(rf) > 1.0e-12:
            force_errors.append(abs(float(replay["normal_force"]) - rf) / max(abs(rf), 1.0e-30))
            if peak_rf_tuple is None or rf > peak_rf_tuple[0]:
                peak_rf_tuple = (rf, float(replay["normal_force"]))
            if float(replay["normal_force"]) > peak_offset_force:
                peak_offset_force = float(replay["normal_force"])
                peak_midplane_force = float(midplane["normal_force"])
        if cels is not None and abs(cels) > 1.0e-12:
            energy_errors.append(abs(float(replay["contact_energy"]) - cels) / max(abs(cels), 1.0e-30))
            if peak_cels_tuple is None or cels > peak_cels_tuple[0]:
                peak_cels_tuple = (cels, float(replay["contact_energy"]))
        if cnum is not None:
            cnum_errors.append(abs(float(replay["cnum_equivalent"]) - cnum))

    peak_force_error = (
        None
        if peak_rf_tuple is None
        else abs(peak_rf_tuple[1] - peak_rf_tuple[0]) / max(abs(peak_rf_tuple[0]), 1.0e-30)
    )
    peak_energy_error = (
        None
        if peak_cels_tuple is None
        else abs(peak_cels_tuple[1] - peak_cels_tuple[0]) / max(abs(peak_cels_tuple[0]), 1.0e-30)
    )
    force_mean_error = _mean_or_blank(force_errors)
    energy_mean_error = _mean_or_blank(energy_errors)
    force_law_aligned = (
        peak_force_error is not None
        and peak_force_error < 0.02
        and peak_energy_error is not None
        and peak_energy_error < 0.02
        and force_mean_error != ""
        and float(force_mean_error) < 0.02
        and energy_mean_error != ""
        and float(energy_mean_error) < 0.15
    )
    sfc_min_dt = _min_float([row for row in sfc_rows if float(row.get("accepted_dt", 0.0)) > 0.0], "accepted_dt")
    base.update(
        {
            "force_law_mean_rel_error": force_mean_error,
            "force_law_max_rel_error": _max_or_blank(force_errors),
            "energy_law_mean_rel_error": energy_mean_error,
            "energy_law_max_rel_error": _max_or_blank(energy_errors),
            "cnum_equivalent_max_abs_error": _max_or_blank(cnum_errors),
            "peak_rf_force_replay_rel_error": "" if peak_force_error is None else peak_force_error,
            "peak_cels_energy_replay_rel_error": "" if peak_energy_error is None else peak_energy_error,
            "peak_midplane_force_fraction": ""
            if peak_offset_force <= 0.0
            else peak_midplane_force / peak_offset_force,
            "sfc_min_accepted_dt": "" if sfc_min_dt is None else sfc_min_dt,
            "sfc_rows_with_retry": sum(1 for row in sfc_rows if int(row.get("cutback_retry_count", 0)) > 0),
            "sfc_max_retry_count": max((int(row.get("cutback_retry_count", 0)) for row in sfc_rows), default=0),
            "diagnosis": "trajectory_difference_dominant" if force_law_aligned else "force_law_or_output_difference_remains",
        }
    )
    return base


def contact_lifecycle_output_diagnostics(
    model: DropModel,
    calculix_rows: list[Row],
    sfc_rows: list[Row],
    calculix_displacements: dict[float, np.ndarray] | None = None,
) -> Row:
    """Return contact lifecycle and RF/CELS/CNUM output-definition diagnostics."""

    base: Row = {
        "case": model.case,
        "resolution": model.resolution,
        "contact_mode": sfc_rows[0].get("contact_mode", "") if sfc_rows else "",
        "calculix_completed": str(bool(calculix_rows)).lower(),
        "rf_output_definition": "CalculiX fixed-floor RF total; SFC assembled penalty normal force",
        "cels_output_definition": "CalculiX printout total contact spring energy; SFC sample-integrated penalty energy",
        "cnum_output_definition": "CalculiX generated contact spring element count; SFC reports CNUM-equivalent generated spring count when available",
        "clearance_update_definition": "CalculiX F2F stores master xi/eta and normal after contact generation; clearance is recomputed from current nodal positions plus clearini*ramp",
    }
    if not calculix_rows or not sfc_rows:
        base.update(
            {
                "calculix_cnum_sequence": "",
                "sfc_cnum_sequence": _sequence_string(_series_values(sfc_rows, "calculix_equivalent_contact_count")),
                "calculix_cnum_transition_count": "",
                "sfc_cnum_transition_count": _transition_count(_series_values(sfc_rows, "calculix_equivalent_contact_count")),
                "cnum_first_mismatch_time": "",
                "cnum_max_abs_error": "",
                "calculix_displacement_replay_cnum_sequence": "",
                "calculix_displacement_replay_cnum_max_abs_error": "",
                "sfc_trajectory_cnum_max_abs_error": "",
                "trajectory_difference_explains_cnum_gap": "",
                "calculix_release_or_reactivation_observed": "",
                "force_peak_time_calculix": "",
                "force_peak_time_sfc": _peak_time(sfc_rows, "normal_force_proxy"),
                "force_peak_time_abs_error": "",
                "energy_peak_time_calculix": "",
                "energy_peak_time_sfc": _peak_time(sfc_rows, "contact_energy_proxy"),
                "energy_peak_time_abs_error": "",
                "diagnosis": "external_unavailable",
            }
        )
        return base

    calc_cnum = _series_values(calculix_rows, "calculix_contact_count")
    sfc_cnum = _series_values(sfc_rows, "calculix_equivalent_contact_count")
    cnum_errors: list[float] = []
    mismatch_time: float | str = ""
    for calc_row in calculix_rows:
        time = float(calc_row["time"])
        calc_value = _optional_float(calc_row.get("calculix_contact_count", ""))
        sfc_row = _nearest_row(sfc_rows, time)
        sfc_value = _optional_float(sfc_row.get("calculix_equivalent_contact_count", "")) if sfc_row else None
        if calc_value is None or sfc_value is None:
            continue
        err = abs(float(sfc_value) - float(calc_value))
        cnum_errors.append(err)
        if err > 0.0 and mismatch_time == "":
            mismatch_time = time

    replay_cnum: list[float] = []
    replay_errors: list[float] = []
    if calculix_displacements:
        for calc_row in calculix_rows:
            time = float(calc_row["time"])
            disp_time = min(calculix_displacements, key=lambda value: abs(float(value) - time))
            replay = contact_replay_metrics(model, model.nodes + calculix_displacements[disp_time])
            replay_value = float(replay["cnum_equivalent"])
            replay_cnum.append(replay_value)
            calc_value = _optional_float(calc_row.get("calculix_contact_count", ""))
            if calc_value is not None:
                replay_errors.append(abs(replay_value - float(calc_value)))

    calc_force_peak = _peak_time(calculix_rows, "normal_force_proxy")
    sfc_force_peak = _peak_time(sfc_rows, "normal_force_proxy")
    calc_energy_peak = _peak_time(calculix_rows, "contact_energy_proxy")
    sfc_energy_peak = _peak_time(sfc_rows, "contact_energy_proxy")
    calc_transitions = _transition_count(calc_cnum)
    sfc_transitions = _transition_count(sfc_cnum)
    cnum_mismatch = bool(cnum_errors and max(cnum_errors) > 0.0)
    replay_max_error = _max_or_blank(replay_errors)
    sfc_max_error = _max_or_blank(cnum_errors)
    trajectory_explains_gap = (
        replay_max_error != ""
        and sfc_max_error != ""
        and float(replay_max_error) < float(sfc_max_error)
    )
    release_or_reactivation = bool(calc_transitions > 1 and len({value for value in calc_cnum if value > 0.0}) > 1)
    peak_time_shift = (
        calc_force_peak != ""
        and sfc_force_peak != ""
        and abs(float(calc_force_peak) - float(sfc_force_peak)) > 0.5 * float(model.dt)
    )
    if cnum_mismatch or release_or_reactivation:
        diagnosis = "contact_lifecycle_trajectory_difference_observed"
    elif peak_time_shift:
        diagnosis = "output_peak_timing_difference_observed"
    else:
        diagnosis = "lifecycle_output_definitions_aligned_for_sampled_rows"

    base.update(
        {
            "calculix_cnum_sequence": _sequence_string(calc_cnum),
            "sfc_cnum_sequence": _sequence_string(sfc_cnum),
            "calculix_cnum_transition_count": calc_transitions,
            "sfc_cnum_transition_count": sfc_transitions,
            "cnum_first_mismatch_time": mismatch_time,
            "cnum_max_abs_error": sfc_max_error,
            "calculix_displacement_replay_cnum_sequence": _sequence_string(replay_cnum),
            "calculix_displacement_replay_cnum_max_abs_error": replay_max_error,
            "sfc_trajectory_cnum_max_abs_error": sfc_max_error,
            "trajectory_difference_explains_cnum_gap": str(bool(trajectory_explains_gap)).lower()
            if replay_max_error != "" and sfc_max_error != ""
            else "",
            "calculix_release_or_reactivation_observed": str(release_or_reactivation).lower(),
            "force_peak_time_calculix": calc_force_peak,
            "force_peak_time_sfc": sfc_force_peak,
            "force_peak_time_abs_error": _abs_or_blank(calc_force_peak, sfc_force_peak),
            "energy_peak_time_calculix": calc_energy_peak,
            "energy_peak_time_sfc": sfc_energy_peak,
            "energy_peak_time_abs_error": _abs_or_blank(calc_energy_peak, sfc_energy_peak),
            "diagnosis": diagnosis,
        }
    )
    return base


def _parse_calculix_stdout_diagnostics(path: Path | None) -> Row:
    if path is None or not path.exists():
        return {
            "calculix_stdout_increment_count": "",
            "calculix_stdout_max_attempt": "",
            "calculix_stdout_min_increment_size": "",
            "calculix_stdout_cutback_attempt_count": "",
            "calculix_stdout_no_convergence_count": "",
            "calculix_stdout_kscale_restore_count": "",
            "calculix_stdout_contact_energy_stabilization_count": "",
            "calculix_stdout_forced_increment_size_count": "",
            "calculix_stdout_max_contact_spring_elements": "",
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    attempts = [int(value) for value in re.findall(r"increment\s+\d+\s+attempt\s+(\d+)", text)]
    sizes = [float(value) for value in re.findall(r"increment size=\s*([0-9.Ee+-]+)", text)]
    contact_counts = [int(value) for value in re.findall(r"Number of contact spring elements=(\d+)", text)]
    return {
        "calculix_stdout_increment_count": len(attempts),
        "calculix_stdout_max_attempt": max(attempts) if attempts else "",
        "calculix_stdout_min_increment_size": min(sizes) if sizes else "",
        "calculix_stdout_cutback_attempt_count": sum(1 for value in attempts if value > 1),
        "calculix_stdout_no_convergence_count": len(re.findall(r"\bno convergence\b", text)),
        "calculix_stdout_kscale_restore_count": len(re.findall(r"restoring the elastic contact stifnesses", text)),
        "calculix_stdout_contact_energy_stabilization_count": len(
            re.findall(r"Adaption of the (?:energy residual|max-decay boundary)", text)
        ),
        "calculix_stdout_forced_increment_size_count": len(re.findall(r"new increment size is forced", text)),
        "calculix_stdout_max_contact_spring_elements": max(contact_counts) if contact_counts else "",
    }


def hht_residual_tangent_diagnostics(
    model: DropModel,
    *,
    contact_mode: str = "calculix_c3d4_f2f",
) -> list[Row]:
    """Return HHT residual/tangent trajectory diagnostics for SFC contact.

    The probe mirrors the CalculiX `calcresidual.c` sign convention by checking
    the SFC residual form

    `R = M a + (1 + alpha) R_static(u_{n+1}) - alpha R_static(u_n)`

    against finite differences.  It uses a stateless contact geometry to keep
    the active set deterministic during centered finite differences.
    """

    if contact_mode.startswith("persistent_"):
        contact_mode = contact_mode.removeprefix("persistent_")
    mechanics = MechanicsModel.from_tet4_mesh(
        model.nodes,
        model.tet_elements,
        E=model.E,
        nu=model.nu,
        density=model.density,
    )
    contact, details = _make_contact_geometry(model, contact_mode)
    state, previous_static = initial_state(
        mechanics,
        contact,
        gravity=model.gravity,
        initial_velocity=(0.0, 0.0, model.initial_velocity_z),
    )
    rows: list[Row] = []
    current_time = 0.0
    step = 0
    while current_time < float(model.total_time) - 1.0e-12:
        dt = min(float(model.dt), float(model.total_time) - current_time)
        state_n = MechanicsState(state.x.copy(), state.v.copy(), state.a.copy(), time=state.time)
        previous_static_n = previous_static.copy()
        state, previous_static, diagnostics = hht_step(
            mechanics,
            state,
            previous_static,
            contact,
            dt=dt,
            gravity=model.gravity,
            alpha=model.hht_alpha,
            max_iterations=12,
            tolerance=1.0e-10,
        )
        current_time += dt
        state.time = float(current_time)
        step += 1
        if diagnostics.contact.active_count <= 0:
            continue
        rows.append(
            _hht_probe_row(
                model,
                mechanics,
                contact,
                state_n,
                previous_static_n,
                state,
                previous_static,
                diagnostics,
                step=step,
                dt=dt,
                contact_mode=contact_mode,
                details=details,
            )
        )
        # The first active step is the most useful place to diagnose activation
        # trajectory differences against CalculiX.
        break
    if rows:
        return rows
    return [
        {
            "case": model.case,
            "resolution": model.resolution,
            "contact_mode": contact_mode,
            "time": "",
            "dt": model.dt,
            "alpha": model.hht_alpha,
            "probe_step": "",
            "active_contact_count": 0,
            "max_penetration": 0.0,
            "hht_dynamic_residual_norm": "",
            "static_residual_norm_current": "",
            "static_residual_norm_previous": float(np.linalg.norm(previous_static)),
            "effective_tangent_directional_fd_rel_error": "",
            "static_tangent_directional_fd_rel_error": "",
            "contact_residual_tangent_directional_fd_rel_error": "",
            "previous_static_residual_update_abs_error": "",
            "previous_static_residual_update_rel_error": "",
            "calculix_calcresidual_sign_convention": "not_probed_no_active_contact",
            "details": details,
        }
    ]


def _hht_probe_row(
    model: DropModel,
    mechanics: MechanicsModel,
    contact: ContactGeometry,
    state_n: MechanicsState,
    previous_static_n: np.ndarray,
    state_np1: MechanicsState,
    accepted_static_residual: np.ndarray,
    diagnostics: StepDiagnostics,
    *,
    step: int,
    dt: float,
    contact_mode: str,
    details: str,
) -> Row:
    residual, tangent, static_residual, static_tangent = _hht_dynamic_residual_and_tangent(
        mechanics,
        contact,
        state_n,
        previous_static_n,
        state_np1.x.reshape(-1),
        dt=dt,
        gravity=model.gravity,
        alpha=model.hht_alpha,
    )
    direction = _contact_probe_direction(model, mechanics.n_dofs)
    eps = 1.0e-7
    r_plus, _, static_plus, _ = _hht_dynamic_residual_and_tangent(
        mechanics,
        contact,
        state_n,
        previous_static_n,
        state_np1.x.reshape(-1) + eps * direction,
        dt=dt,
        gravity=model.gravity,
        alpha=model.hht_alpha,
    )
    r_minus, _, static_minus, _ = _hht_dynamic_residual_and_tangent(
        mechanics,
        contact,
        state_n,
        previous_static_n,
        state_np1.x.reshape(-1) - eps * direction,
        dt=dt,
        gravity=model.gravity,
        alpha=model.hht_alpha,
    )
    dynamic_fd = (r_plus - r_minus) / (2.0 * eps)
    static_fd = (static_plus - static_minus) / (2.0 * eps)
    contact_tangent_error = _contact_residual_tangent_directional_error(
        contact,
        state_np1.x.reshape(-1),
        direction,
        n_nodes=mechanics.n_nodes,
        eps=eps,
    )
    update_abs = float(np.linalg.norm(accepted_static_residual - static_residual))
    update_rel = update_abs / max(float(np.linalg.norm(static_residual)), 1.0e-30)
    return {
        "case": model.case,
        "resolution": model.resolution,
        "contact_mode": contact_mode,
        "time": float(state_np1.time),
        "dt": float(dt),
        "alpha": float(model.hht_alpha),
        "probe_step": int(step),
        "active_contact_count": int(diagnostics.contact.active_count),
        "max_penetration": float(diagnostics.contact.max_penetration),
        "hht_dynamic_residual_norm": float(np.linalg.norm(residual)),
        "static_residual_norm_current": float(np.linalg.norm(static_residual)),
        "static_residual_norm_previous": float(np.linalg.norm(previous_static_n)),
        "effective_tangent_directional_fd_rel_error": _relative_vector_error(tangent @ direction, dynamic_fd),
        "static_tangent_directional_fd_rel_error": _relative_vector_error(static_tangent @ direction, static_fd),
        "contact_residual_tangent_directional_fd_rel_error": contact_tangent_error,
        "previous_static_residual_update_abs_error": update_abs,
        "previous_static_residual_update_rel_error": update_rel,
        "calculix_calcresidual_sign_convention": "sfc_R_is_negative_of_CalculiX_rhs_b",
        "details": details,
    }


def _hht_dynamic_residual_and_tangent(
    mechanics: MechanicsModel,
    contact: ContactGeometry,
    state_n: MechanicsState,
    previous_static_residual: np.ndarray,
    x_flat: np.ndarray,
    *,
    dt: float,
    gravity: float,
    alpha: float,
) -> tuple[np.ndarray, Any, np.ndarray, Any]:
    beta, _gamma = hht_newmark_parameters(alpha)
    c0 = 1.0 / (beta * float(dt) * float(dt))
    u_n = (state_n.x - mechanics.X).reshape(-1)
    v_n = state_n.v.reshape(-1)
    a_n = state_n.a.reshape(-1)
    u_pred = u_n + float(dt) * v_n + float(dt) * float(dt) * (0.5 - beta) * a_n
    u = np.asarray(x_flat, dtype=float) - mechanics.X.reshape(-1)
    a = c0 * (u - u_pred)
    static_residual, static_tangent = static_residual_and_tangent(
        mechanics,
        np.asarray(x_flat, dtype=float).reshape((-1, 3)),
        contact,
        gravity=gravity,
    )
    residual = mechanics.mass_matrix @ a + (1.0 + float(alpha)) * static_residual - float(alpha) * previous_static_residual
    tangent = (mechanics.mass_matrix * c0 + static_tangent * (1.0 + float(alpha))).tocsr()
    return np.asarray(residual, dtype=float), tangent, static_residual, static_tangent


def _contact_probe_direction(model: DropModel, n_dofs: int) -> np.ndarray:
    direction = np.zeros(int(n_dofs), dtype=float)
    nodes = np.unique(np.asarray(model.surface_faces, dtype=np.int64).ravel())
    if nodes.size == 0:
        direction[2::3] = -1.0
    else:
        for node in nodes:
            direction[3 * int(node) + 2] = -1.0
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        direction[:] = 1.0
        norm = float(np.linalg.norm(direction))
    return direction / norm


def _contact_residual_tangent_directional_error(
    contact: ContactGeometry,
    x_flat: np.ndarray,
    direction: np.ndarray,
    *,
    n_nodes: int,
    eps: float,
) -> float | str:
    def contact_residual_and_tangent(x_value: np.ndarray) -> tuple[np.ndarray, Any]:
        response = assemble_contact_response(contact.samples(x_value.reshape((-1, 3))), n_nodes)
        return -response.force.reshape(-1), response.tangent

    contact_residual, contact_tangent = contact_residual_and_tangent(np.asarray(x_flat, dtype=float))
    if contact_tangent.nnz == 0 and np.linalg.norm(contact_residual) <= 0.0:
        return ""
    plus, _ = contact_residual_and_tangent(np.asarray(x_flat, dtype=float) + eps * direction)
    minus, _ = contact_residual_and_tangent(np.asarray(x_flat, dtype=float) - eps * direction)
    finite_difference = (plus - minus) / (2.0 * eps)
    return _relative_vector_error(contact_tangent @ direction, finite_difference)


def _relative_vector_error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = np.asarray(expected, dtype=float)
    return float(np.linalg.norm(actual_arr - expected_arr) / max(np.linalg.norm(expected_arr), 1.0e-30))


def _optional_float(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _series_values(rows: list[Row], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _optional_float(row.get(key, ""))
        if value is not None:
            values.append(float(value))
    return values


def _sequence_string(values: list[float]) -> str:
    if not values:
        return ""
    sequence: list[int | float] = []
    for value in values:
        scalar: int | float = int(round(value)) if abs(value - round(value)) < 1.0e-12 else float(value)
        if not sequence or sequence[-1] != scalar:
            sequence.append(scalar)
    return "->".join(str(value) for value in sequence)


def _transition_count(values: list[float]) -> int:
    if not values:
        return 0
    return sum(1 for previous, current in zip(values, values[1:], strict=False) if abs(current - previous) > 1.0e-12)


def _nearest_row(rows: list[Row], time: float) -> Row | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row.get("time", 0.0)) - float(time)))


def _peak_time(rows: list[Row], key: str) -> float | str:
    best_time: float | None = None
    best_value: float | None = None
    for row in rows:
        value = _optional_float(row.get(key, ""))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = float(value)
            best_time = float(row["time"])
    return "" if best_time is None else best_time


def _abs_or_blank(a: Any, b: Any) -> float | str:
    if a == "" or b == "" or a is None or b is None:
        return ""
    return abs(float(a) - float(b))


def _mean_or_blank(values: list[float]) -> float | str:
    return "" if not values else float(np.mean(values))


def _max_or_blank(values: list[float]) -> float | str:
    return "" if not values else float(np.max(values))


def _make_contact_geometry(model: DropModel, contact_mode: str) -> tuple[ContactGeometry, str]:
    if contact_mode == "persistent_calculix_c3d4_f2f":
        return (
            PersistentCalculixC3D4FaceToFacePlaneContactGeometry(
                model.surface_faces,
                plane_z=_contact_plane_z(model),
                stiffness=model.contact_stiffness,
            ),
            "clean-room persistent CalculiX-style C3D4 face-to-face mode; one slave-face centroid spring; hard linear overclosure; rigid plane query",
        )
    if contact_mode == "calculix_c3d4_f2f":
        return (
            CalculixC3D4FaceToFacePlaneContactGeometry(
                model.surface_faces,
                plane_z=_contact_plane_z(model),
                stiffness=model.contact_stiffness,
            ),
            "clean-room CalculiX-style C3D4 face-to-face mode; one slave-face centroid spring; hard linear overclosure; rigid plane query",
        )
    if contact_mode == "persistent_dynamic_sdf_calculix_f2f":
        master_x, master_faces = _rigid_plane_master_surface(model)
        delta_safe = _sdf_plane_padding(model)
        broad_phase = UniformTriangleAABBHash.from_surface(
            master_x,
            master_faces,
            delta_safe=delta_safe,
            cell_size=max(0.25, delta_safe),
        )
        return (
            PersistentCalculixC3D4FaceToFaceSDFContactGeometry(
                model.surface_faces,
                master_x,
                master_faces,
                candidate_provider=broad_phase.query_point,
                stiffness=model.contact_stiffness,
                master_element_node_count=4,
            ),
            "clean-room persistent CalculiX-style C3D4 face-to-face mode; one slave-face centroid spring; hard linear overclosure; dynamic FEM-SDF plane query",
        )
    if contact_mode == "dynamic_sdf_calculix_f2f":
        master_x, master_faces = _rigid_plane_master_surface(model)
        delta_safe = _sdf_plane_padding(model)
        broad_phase = UniformTriangleAABBHash.from_surface(
            master_x,
            master_faces,
            delta_safe=delta_safe,
            cell_size=max(0.25, delta_safe),
        )
        return (
            CalculixC3D4FaceToFaceSDFContactGeometry(
                model.surface_faces,
                master_x,
                master_faces,
                candidate_provider=broad_phase.query_point,
                stiffness=model.contact_stiffness,
                master_element_node_count=4,
            ),
            "clean-room CalculiX-style C3D4 face-to-face mode; one slave-face centroid spring; hard linear overclosure; dynamic FEM-SDF plane query",
        )
    if contact_mode == "plane":
        return (
            PlaneContactGeometry(
                model.surface_faces,
                plane_z=_contact_plane_z(model),
                stiffness=model.contact_stiffness,
            ),
            "clean-room CalculiX-aligned StVK backend; rigid plane contact geometry",
        )
    if contact_mode == "dynamic_sdf_plane":
        master_x, master_faces = _rigid_plane_master_surface(model)
        delta_safe = _sdf_plane_padding(model)
        broad_phase = UniformTriangleAABBHash.from_surface(
            master_x,
            master_faces,
            delta_safe=delta_safe,
            cell_size=max(0.25, delta_safe),
        )
        return (
            DynamicSurfaceSDFContactGeometry(
                model.surface_faces,
                master_x,
                master_faces,
                candidate_provider=broad_phase.query_point,
                stiffness=model.contact_stiffness,
            ),
            "clean-room CalculiX-aligned StVK backend; dynamic FEM-SDF plane contact query",
        )
    raise ValueError(
        "contact_mode must be 'calculix_c3d4_f2f', 'persistent_calculix_c3d4_f2f', "
        "'dynamic_sdf_calculix_f2f', 'persistent_dynamic_sdf_calculix_f2f', 'plane', or 'dynamic_sdf_plane'"
    )


def _normal_force_source(contact_mode: str) -> str:
    if contact_mode in {
        "calculix_c3d4_f2f",
        "persistent_calculix_c3d4_f2f",
        "dynamic_sdf_calculix_f2f",
        "persistent_dynamic_sdf_calculix_f2f",
    }:
        return "calculix_c3d4_f2f_hard_linear"
    if contact_mode == "dynamic_sdf_plane":
        return "dynamic_sdf_three_point_penalty_tangent"
    return "three_point_penalty_tangent"


def _generated_contact_count(contact: ContactGeometry) -> int | str:
    value = getattr(contact, "generated_contact_count", None)
    if value is None:
        return ""
    return int(value)


def _calculix_equivalent_contact_count(contact: ContactGeometry) -> int | str:
    value = calculix_equivalent_contact_element_count(contact)
    if value < 0:
        return ""
    return int(value)


def _contact_count_for_heuristic(contact: ContactGeometry, diagnostics: StepDiagnostics) -> int:
    generated = _generated_contact_count(contact)
    if generated != "":
        return int(generated)
    return int(diagnostics.contact.active_count)


def _contact_cutback_enabled(contact_mode: str, cutback_policy: str = "active_retry") -> bool:
    if cutback_policy == "calculix_direct":
        return False
    if cutback_policy != "active_retry":
        raise ValueError("cutback_policy must be 'active_retry' or 'calculix_direct'")
    return contact_mode.startswith("persistent_")


def _snapshot_contact_lifecycle(contact: ContactGeometry) -> Any:
    lifecycle = getattr(contact, "lifecycle", None)
    if lifecycle is None or not hasattr(lifecycle, "snapshot"):
        return None
    return (lifecycle.snapshot(), bool(getattr(contact, "cutback_retry", False)))


def _restore_contact_lifecycle(contact: ContactGeometry, snapshot: Any) -> None:
    if snapshot is None:
        return
    lifecycle_snapshot, cutback_retry = snapshot
    lifecycle = getattr(contact, "lifecycle", None)
    if lifecycle is not None and hasattr(lifecycle, "restore"):
        lifecycle.restore(lifecycle_snapshot)
    _set_contact_cutback_retry(contact, bool(cutback_retry))


def _set_contact_cutback_retry(contact: ContactGeometry, value: bool) -> None:
    setter = getattr(contact, "set_cutback_retry", None)
    if setter is not None:
        setter(bool(value))


def _snapshot_contact_heuristic(heuristic: CalculixContactConvergenceHeuristic) -> list[Any]:
    return list(heuristic.history or [])


def _restore_contact_heuristic(heuristic: CalculixContactConvergenceHeuristic, snapshot: list[Any]) -> None:
    heuristic.history = list(snapshot)


def _rigid_plane_master_surface(model: DropModel) -> tuple[np.ndarray, np.ndarray]:
    floor_coords = np.asarray(model.floor_nodes[:, 1:4], dtype=float)
    half_width = float(np.max(np.abs(floor_coords[:, :2])))
    z = _contact_plane_z(model)
    master_x = np.asarray(
        [
            [-half_width, -half_width, z],
            [half_width, -half_width, z],
            [half_width, half_width, z],
            [-half_width, half_width, z],
        ],
        dtype=float,
    )
    master_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return master_x, master_faces


def _sdf_plane_padding(model: DropModel) -> float:
    height = float(np.max(model.nodes[:, 2]) - np.min(model.nodes[:, 2]))
    travel = abs(float(model.initial_velocity_z)) * float(model.total_time)
    travel += 0.5 * float(model.gravity) * float(model.total_time) ** 2
    return max(0.25, 1.5 * height + travel + 0.05)


def _vtk_state_from_backend(state: MechanicsState, diagnostics: StepDiagnostics) -> NonlinearState:
    return NonlinearState(
        time=state.time,
        x=state.x.copy(),
        v=state.v.copy(),
        element_strain=diagnostics.internal.strain.copy(),
        element_stress=diagnostics.internal.stress.copy(),
        von_mises=diagnostics.internal.von_mises.copy(),
        min_gap=diagnostics.contact.min_gap,
        max_penetration=diagnostics.contact.max_penetration,
        active_contact_count=diagnostics.contact.active_count,
        kinetic_energy=diagnostics.kinetic_energy,
        strain_energy=diagnostics.internal.strain_energy,
        gravitational_energy=diagnostics.gravitational_energy,
        contact_energy=diagnostics.contact.energy,
        total_energy=diagnostics.total_energy,
        material_tangent_norm=float(np.linalg.norm(diagnostics.internal.material_tangent.data))
        if diagnostics.internal.material_tangent.nnz
        else 0.0,
        geometric_tangent_norm=float(np.linalg.norm(diagnostics.internal.geometric_tangent.data))
        if diagnostics.internal.geometric_tangent.nnz
        else 0.0,
        contact_tangent_norm=float(np.linalg.norm(diagnostics.contact.tangent.data)) if diagnostics.contact.tangent.nnz else 0.0,
        newton_iterations=diagnostics.newton_iterations,
        newton_residual_norm=diagnostics.newton_residual_norm,
    )


def _contact_zone_elements(model: DropModel) -> np.ndarray:
    face_centers = np.mean(model.nodes[model.surface_faces], axis=1)
    z_min = float(np.min(face_centers[:, 2]))
    height = float(np.max(model.nodes[:, 2]) - np.min(model.nodes[:, 2]))
    mask = face_centers[:, 2] <= z_min + max(1.0e-8, 0.1 * height)
    element_ids = [int(model.slave_face_refs[i][0]) - 1 for i, keep in enumerate(mask) if keep]
    if not element_ids:
        return np.arange(model.tet_elements.shape[0], dtype=np.int64)
    return np.unique(np.asarray(element_ids, dtype=np.int64))


def _contact_zone_max_von_mises(model: DropModel, vm: np.ndarray) -> float:
    return float(np.max(vm[_contact_zone_elements(model)]))


def _max_float(rows: list[Row], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        values.append(float(value))
    return None if not values else max(values)


def _min_float(rows: list[Row], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        values.append(float(value))
    return None if not values else min(values)


def _time_series_l2_relative(reference_rows: list[Row], candidate_rows: list[Row], key: str) -> float | str:
    ref_times = np.asarray([float(row["time"]) for row in reference_rows], dtype=float)
    cand_times = np.asarray([float(row["time"]) for row in candidate_rows], dtype=float)
    if ref_times.size == 0 or cand_times.size == 0:
        return ""
    ref_values = np.asarray([float(row[key]) for row in reference_rows], dtype=float)
    cand_values = np.interp(ref_times, cand_times, np.asarray([float(row[key]) for row in candidate_rows], dtype=float))
    return float(np.linalg.norm(cand_values - ref_values) / max(np.linalg.norm(ref_values), 1.0e-30))


def compare_contact_histories(
    model: DropModel,
    calculix_rows: list[Row],
    sfc_rows: list[Row],
    calculix_stress: np.ndarray,
    sfc_state: NonlinearState,
) -> Row:
    """Return one comparison row for CalculiX/SFC contact histories."""

    cx_first = _first_contact_time(calculix_rows)
    sfc_first = _first_contact_time(sfc_rows)
    cx_vm = _voigt_von_mises(calculix_stress)
    cx_contact_vm = _contact_zone_max_von_mises(model, cx_vm) if calculix_rows else ""
    sfc_contact_vm = _contact_zone_max_von_mises(model, sfc_state.von_mises)
    cx_rebound = _rebound_physicality(model, calculix_rows)
    sfc_rebound = _rebound_physicality(model, sfc_rows)
    cx_peak_force = _max_float(calculix_rows, "normal_force_proxy")
    sfc_peak_force = _max_float(sfc_rows, "normal_force_proxy")
    cx_peak_energy = _max_float(calculix_rows, "contact_energy_proxy")
    sfc_peak_energy = _max_float(sfc_rows, "contact_energy_proxy")
    cx_max_pen = _max_float(calculix_rows, "max_penetration")
    sfc_max_pen = _max_float(sfc_rows, "max_penetration")
    cx_max_cnum = _max_float(calculix_rows, "calculix_contact_count")
    sfc_max_cnum = _max_float(sfc_rows, "calculix_equivalent_contact_count")
    row: Row = {
        "case": model.case,
        "resolution": model.resolution,
        "contact_mode": sfc_rows[0].get("contact_mode", "") if sfc_rows else "",
        "dt": model.dt,
        "duration": model.total_time,
        "calculix_completed": str(bool(calculix_rows)).lower(),
        "contact_activation_time_calculix": "" if cx_first is None else cx_first,
        "contact_activation_time_sfc": "" if sfc_first is None else sfc_first,
        "contact_activation_abs_error": "" if cx_first is None or sfc_first is None else abs(sfc_first - cx_first),
        "max_penetration_calculix": "" if cx_max_pen is None else cx_max_pen,
        "max_penetration_sfc": "" if sfc_max_pen is None else sfc_max_pen,
        "max_penetration_rel_error": _rel_or_blank(sfc_max_pen, cx_max_pen),
        "peak_normal_force_calculix": "" if cx_peak_force is None else cx_peak_force,
        "peak_normal_force_sfc": "" if sfc_peak_force is None else sfc_peak_force,
        "peak_normal_force_rel_error": _rel_or_blank(sfc_peak_force, cx_peak_force),
        "peak_contact_energy_calculix": "" if cx_peak_energy is None else cx_peak_energy,
        "peak_contact_energy_sfc": "" if sfc_peak_energy is None else sfc_peak_energy,
        "peak_contact_energy_rel_error": _rel_or_blank(sfc_peak_energy, cx_peak_energy),
        "max_cnum_calculix": "" if cx_max_cnum is None else cx_max_cnum,
        "max_cnum_equivalent_sfc": "" if sfc_max_cnum is None else sfc_max_cnum,
        "max_cnum_abs_error": "" if cx_max_cnum is None or sfc_max_cnum is None else abs(sfc_max_cnum - cx_max_cnum),
        "z_cm_l2_rel_error": _time_series_l2_relative(calculix_rows, sfc_rows, "z_cm"),
        "min_gap_l2_rel_error": _time_series_l2_relative(calculix_rows, sfc_rows, "min_gap"),
        "rebound_height_calculix": cx_rebound.get("max_rebound_z", ""),
        "rebound_height_sfc": sfc_rebound.get("max_rebound_z", ""),
        "contact_zone_max_vm_calculix": cx_contact_vm,
        "contact_zone_max_vm_sfc": sfc_contact_vm,
        "contact_zone_max_vm_rel_error": _rel_or_blank(sfc_contact_vm, cx_contact_vm if cx_contact_vm != "" else None),
        "max_sfc_newton_iterations": max(int(row["newton_iterations"]) for row in sfc_rows),
        "max_sfc_newton_residual_norm": max(float(row["newton_residual_norm"]) for row in sfc_rows),
    }
    row["acceptance_status"] = _contact_acceptance_status(row)
    return row


def _rel_or_blank(a: float | None | str, b: float | None | str) -> float | str:
    if a == "" or b == "" or a is None or b is None:
        return ""
    return abs(float(a) - float(b)) / max(abs(float(b)), 1.0e-30)


def _contact_acceptance_status(row: Row) -> str:
    if row["calculix_completed"] != "true":
        return "external_unavailable"
    required = [
        row["contact_activation_abs_error"],
        row["max_penetration_rel_error"],
        row["peak_normal_force_rel_error"],
        row["contact_zone_max_vm_rel_error"],
    ]
    if any(value == "" for value in required):
        return "evidence_incomplete"
    # These are deliberately loose diagnostic thresholds.  Passing this gate is
    # not a claim of source-level CalculiX contact equivalence.
    passed = (
        float(row["contact_activation_abs_error"]) <= max(2.0 * float(row["dt"]), 1.0e-12)
        and float(row["max_penetration_rel_error"]) <= 0.5
        and float(row["peak_normal_force_rel_error"]) <= 0.75
        and float(row["contact_zone_max_vm_rel_error"]) <= 0.5
    )
    return "passed_scoped_gate" if passed else "comparison_available_not_accepted"


def write_stress_cloud_vtks(
    out_dir: Path,
    model: DropModel,
    calculix_u: np.ndarray,
    calculix_stress: np.ndarray,
    sfc_x: np.ndarray,
    sfc_state: NonlinearState,
) -> list[Row]:
    """Write SFC and CalculiX final stress clouds with a shared VM scale."""

    vtk_dir = out_dir / "vtk_contact_stress_clouds" / f"{model.case}_r{model.resolution}"
    vtk_dir.mkdir(parents=True, exist_ok=True)
    plane_z = _contact_plane_z(model)
    calc_vm = _voigt_von_mises(calculix_stress)
    calc_stress_tensor = _voigt_to_tensor(calculix_stress)
    calc_state = replace(
        sfc_state,
        x=model.nodes + calculix_u,
        v=np.zeros_like(model.nodes),
        element_strain=np.zeros_like(sfc_state.element_strain),
        element_stress=calc_stress_tensor,
        von_mises=calc_vm,
    )
    sfc_path = vtk_dir / "sfc_frame_0000.vtk"
    calc_path = vtk_dir / "calculix_frame_0000.vtk"
    _write_vtk(sfc_path, model.nodes, sfc_x, model.tet_elements, sfc_state, plane_z=plane_z, plane_half_width=1.15)
    _write_vtk(calc_path, model.nodes, model.nodes + calculix_u, model.tet_elements, calc_state, plane_z=plane_z, plane_half_width=1.15)
    color_min = float(min(np.min(sfc_state.von_mises), np.min(calc_vm)))
    color_max = float(max(np.max(sfc_state.von_mises), np.max(calc_vm)))
    return [
        {
            "case": model.case,
            "resolution": model.resolution,
            "source": "sfc_geometric_nonlinear",
            "vtk_file": str(sfc_path.relative_to(out_dir)),
            "von_mises_color_min": color_min,
            "von_mises_color_max": color_max,
        },
        {
            "case": model.case,
            "resolution": model.resolution,
            "source": "calculix",
            "vtk_file": str(calc_path.relative_to(out_dir)),
            "von_mises_color_min": color_min,
            "von_mises_color_max": color_max,
        },
    ]


def _voigt_to_tensor(values: np.ndarray) -> np.ndarray:
    tensors = np.zeros((values.shape[0], 3, 3), dtype=float)
    for i, row in enumerate(values):
        sxx, syy, szz, sxy, sxz, syz = row
        tensors[i] = [[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]]
    return tensors


def sfc_mesh_convergence(
    resolutions: list[int],
    *,
    duration: float,
    dt: float,
    contact_mode: str = "persistent_calculix_c3d4_f2f",
    cutback_policy: str = "calculix_direct",
) -> list[Row]:
    rows: list[Row] = []
    for resolution in resolutions:
        model = _contact_model(resolution=resolution, duration=duration, dt=dt)
        history, _, state = run_sfc_geometric_contact_history(model, contact_mode=contact_mode, cutback_policy=cutback_policy)
        rows.append(
            {
                "case": "block_drop_contact_mesh",
                "resolution": resolution,
                "contact_mode": contact_mode,
                "cutback_policy": cutback_policy,
                "nodes": model.nodes.shape[0],
                "elements": model.tet_elements.shape[0],
                "max_penetration": _max_float(history, "max_penetration"),
                "peak_normal_force": _max_float(history, "normal_force_proxy"),
                "peak_contact_energy": _max_float(history, "contact_energy_proxy"),
                "contact_zone_max_von_mises": _contact_zone_max_von_mises(model, state.von_mises),
                "max_newton_iterations": max(int(row["newton_iterations"]) for row in history),
            }
        )
    return rows


def sfc_timestep_convergence(
    dts: list[float],
    *,
    duration: float,
    contact_mode: str = "persistent_calculix_c3d4_f2f",
    cutback_policy: str = "calculix_direct",
) -> list[Row]:
    rows: list[Row] = []
    for dt in dts:
        model = _contact_model(resolution=1, duration=duration, dt=dt)
        history, _, state = run_sfc_geometric_contact_history(model, contact_mode=contact_mode, cutback_policy=cutback_policy)
        rows.append(
            {
                "case": "block_drop_contact_timestep",
                "dt": dt,
                "contact_mode": contact_mode,
                "cutback_policy": cutback_policy,
                "duration": duration,
                "max_penetration": _max_float(history, "max_penetration"),
                "peak_normal_force": _max_float(history, "normal_force_proxy"),
                "peak_contact_energy": _max_float(history, "contact_energy_proxy"),
                "contact_zone_max_von_mises": _contact_zone_max_von_mises(model, state.von_mises),
                "final_z_cm": history[-1]["z_cm"],
                "max_newton_iterations": max(int(row["newton_iterations"]) for row in history),
            }
        )
    return rows


def run_validation(
    out_dir: Path,
    *,
    quick: bool = False,
    skip_calculix: bool = False,
    contact_mode: str = "persistent_calculix_c3d4_f2f",
    cutback_policy: str = "calculix_direct",
) -> dict[str, Path]:
    """Run geometric nonlinear contact validation and write CSV/Markdown."""

    out_dir.mkdir(parents=True, exist_ok=True)
    duration = 0.12 if quick else 0.14
    dt = 0.002
    comparison_resolutions = [1] if quick else [1, 2, 3]
    history_rows: list[Row] = []
    comparison_rows: list[Row] = []
    alignment_rows: list[Row] = []
    hht_rows: list[Row] = []
    lifecycle_rows: list[Row] = []
    command_rows: list[Row] = []
    vtk_rows: list[Row] = []
    for resolution in comparison_resolutions:
        model = _contact_model(resolution=resolution, duration=duration, dt=dt)
        sfc_rows, sfc_x, sfc_state = run_sfc_geometric_contact_history(
            model,
            contact_mode=contact_mode,
            cutback_policy=cutback_policy,
        )
        history_rows.extend(sfc_rows)
        if skip_calculix:
            calc_rows: list[Row] = []
            calc_u = np.zeros_like(model.nodes)
            calc_stress = np.zeros((model.tet_elements.shape[0], 6), dtype=float)
            calc_displacements: dict[float, np.ndarray] = {}
            command_rows.append({"case": model.case, "resolution": resolution, "completed": "false", "return_code": "skipped", "command": ""})
        else:
            calc_rows, calc_u, calc_stress, command, calc_displacements = run_calculix_contact_with_stress(model, out_dir)
            command_rows.append(command)
            history_rows.extend(calc_rows)
        comparison_rows.append(compare_contact_histories(model, calc_rows, sfc_rows, calc_stress, sfc_state))
        alignment_rows.append(contact_alignment_diagnostics(model, calc_rows, sfc_rows, calc_displacements, command_rows[-1], out_dir))
        hht_rows.extend(hht_residual_tangent_diagnostics(model, contact_mode=contact_mode))
        lifecycle_rows.append(contact_lifecycle_output_diagnostics(model, calc_rows, sfc_rows, calc_displacements))
        if calc_rows:
            vtk_rows.extend(write_stress_cloud_vtks(out_dir, model, calc_u, calc_stress, sfc_x, sfc_state))

    mesh_rows = sfc_mesh_convergence(
        [1, 2, 3],
        duration=duration,
        dt=dt,
        contact_mode=contact_mode,
        cutback_policy=cutback_policy,
    )
    timestep_rows = sfc_timestep_convergence(
        [0.004, 0.002, 0.001] if quick else [0.002, 0.001, 0.0005],
        duration=duration,
        contact_mode=contact_mode,
        cutback_policy=cutback_policy,
    )
    claim_rows = _claim_rows(comparison_rows, mesh_rows, timestep_rows, vtk_rows, alignment_rows, hht_rows, lifecycle_rows)
    outputs = {
        "history": out_dir / "geometric_contact_history.csv",
        "comparison": out_dir / "geometric_contact_calculix_comparison.csv",
        "alignment": out_dir / "geometric_contact_alignment_diagnostics.csv",
        "hht": out_dir / "geometric_contact_hht_residual_tangent.csv",
        "lifecycle": out_dir / "geometric_contact_lifecycle_output_diagnostics.csv",
        "mesh": out_dir / "geometric_contact_mesh_convergence.csv",
        "timestep": out_dir / "geometric_contact_timestep_convergence.csv",
        "vtk": out_dir / "geometric_contact_stress_clouds.csv",
        "commands": out_dir / "external_solver_commands.csv",
        "claims": out_dir / "geometric_contact_claims.csv",
        "summary": out_dir / "geometric_contact_validation_summary.md",
    }
    _write_csv(outputs["history"], history_rows)
    _write_csv(outputs["comparison"], comparison_rows)
    _write_csv(outputs["alignment"], alignment_rows)
    _write_csv(outputs["hht"], hht_rows)
    _write_csv(outputs["lifecycle"], lifecycle_rows)
    _write_csv(outputs["mesh"], mesh_rows)
    _write_csv(outputs["timestep"], timestep_rows)
    _write_csv(outputs["vtk"], vtk_rows)
    _write_csv(outputs["commands"], command_rows)
    _write_csv(outputs["claims"], claim_rows)
    _write_markdown(
        outputs["summary"],
        comparison_rows,
        mesh_rows,
        timestep_rows,
        vtk_rows,
        claim_rows,
        alignment_rows,
        hht_rows,
        lifecycle_rows,
        contact_mode=contact_mode,
        cutback_policy=cutback_policy,
    )
    return outputs


def _claim_rows(
    comparison_rows: list[Row],
    mesh_rows: list[Row],
    timestep_rows: list[Row],
    vtk_rows: list[Row],
    alignment_rows: list[Row],
    hht_rows: list[Row],
    lifecycle_rows: list[Row],
) -> list[Row]:
    comparison_available = any(row["calculix_completed"] == "true" for row in comparison_rows)
    accepted = any(row["acceptance_status"] == "passed_scoped_gate" for row in comparison_rows)
    replay_supported = bool(alignment_rows) and any(
        row.get("diagnosis") == "trajectory_difference_dominant" for row in alignment_rows
    )
    hht_supported = bool(hht_rows) and any(
        row.get("effective_tangent_directional_fd_rel_error") not in {"", None}
        and float(row["effective_tangent_directional_fd_rel_error"]) < 5.0e-5
        and row.get("contact_residual_tangent_directional_fd_rel_error") not in {"", None}
        and float(row["contact_residual_tangent_directional_fd_rel_error"]) < 5.0e-5
        and row.get("previous_static_residual_update_rel_error") not in {"", None}
        and float(row["previous_static_residual_update_rel_error"]) < 5.0e-8
        for row in hht_rows
    )
    lifecycle_supported = bool(lifecycle_rows) and any(
        row.get("diagnosis") not in {"", None, "external_unavailable"} for row in lifecycle_rows
    )
    return [
        {
            "claim": "block_plane_geometric_contact_calculix_comparison_available",
            "supported": str(comparison_available).lower(),
            "evidence_csv": "geometric_contact_calculix_comparison.csv",
            "details": "CalculiX dynamic contact output parsed for activation, penetration, force, contact energy, and contact-zone stress",
        },
        {
            "claim": "block_plane_geometric_contact_scoped_acceptance",
            "supported": str(accepted).lower(),
            "evidence_csv": "geometric_contact_calculix_comparison.csv",
            "details": "Uses loose diagnostic thresholds; false means evidence exists but external contact agreement is not accepted",
        },
        {
            "claim": "calculix_contact_force_law_replay_diagnostics_available",
            "supported": str(replay_supported).lower(),
            "evidence_csv": "geometric_contact_alignment_diagnostics.csv",
            "details": "Replays SFC hard-linear area-weighted contact law on CalculiX displacements to separate force-law/output differences from trajectory differences",
        },
        {
            "claim": "hht_residual_tangent_trajectory_diagnostics_available",
            "supported": str(hht_supported).lower(),
            "evidence_csv": "geometric_contact_hht_residual_tangent.csv",
            "details": "Checks the HHT residual evaluation point, previous static residual update, and contact residual tangent sign convention by finite differences",
        },
        {
            "claim": "calculix_contact_lifecycle_output_diagnostics_available",
            "supported": str(lifecycle_supported).lower(),
            "evidence_csv": "geometric_contact_lifecycle_output_diagnostics.csv",
            "details": "Tracks CNUM lifecycle transitions, RF/CELS peak timing, and source-level output definitions for CalculiX/SFC contact comparison",
        },
        {
            "claim": "contact_mesh_convergence_trend_available",
            "supported": str(len({int(row['resolution']) for row in mesh_rows}) >= 3).lower(),
            "evidence_csv": "geometric_contact_mesh_convergence.csv",
            "details": "r1/r2/r3 SFC geometric contact trend; no theoretical order claimed",
        },
        {
            "claim": "contact_timestep_convergence_trend_available",
            "supported": str(len(timestep_rows) >= 3).lower(),
            "evidence_csv": "geometric_contact_timestep_convergence.csv",
            "details": "dt, dt/2, dt/4 SFC geometric contact trend",
        },
        {
            "claim": "paraview_stress_cloud_comparison_available",
            "supported": str(bool(vtk_rows)).lower(),
            "evidence_csv": "geometric_contact_stress_clouds.csv",
            "details": "SFC and CalculiX final VTK stress clouds are written with shared von Mises color scale fields",
        },
    ]


def _write_markdown(
    path: Path,
    comparison_rows: list[Row],
    mesh_rows: list[Row],
    timestep_rows: list[Row],
    vtk_rows: list[Row],
    claim_rows: list[Row],
    alignment_rows: list[Row],
    hht_rows: list[Row],
    lifecycle_rows: list[Row],
    *,
    contact_mode: str,
    cutback_policy: str,
) -> None:
    lines = [
        "# Geometric Nonlinear Contact Validation",
        "",
        "This validation compares the clean-room CalculiX-aligned SFC StVK geometric nonlinear block-plane contact path against CalculiX dynamic contact output.",
        "",
        f"SFC contact mode: `{contact_mode}`.",
        f"SFC cutback policy: `{cutback_policy}`.",
        "",
        "The default `calculix_c3d4_f2f` mode uses one slave-face centroid spring per C3D4 boundary triangle and a hard linear pressure-overclosure law for stricter CalculiX-style validation. The older `plane` and `dynamic_sdf_plane` modes use three triangle quadrature samples and are not strict CalculiX C3D4 contact discretizations.",
        "",
        "## Outputs",
        "",
        "- `geometric_contact_history.csv`",
        "- `geometric_contact_calculix_comparison.csv`",
        "- `geometric_contact_alignment_diagnostics.csv`",
        "- `geometric_contact_hht_residual_tangent.csv`",
        "- `geometric_contact_lifecycle_output_diagnostics.csv`",
        "- `geometric_contact_mesh_convergence.csv`",
        "- `geometric_contact_timestep_convergence.csv`",
        "- `geometric_contact_stress_clouds.csv`",
        "- `geometric_contact_claims.csv`",
        "",
        "## Claim Gates",
        "",
        "| Claim | Supported | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in claim_rows:
        lines.append(f"| {row['claim']} | {row['supported']} | `{row['evidence_csv']}` |")
    lines.extend(["", "## Contact Comparison", "", "| Resolution | Status | z_cm L2 rel. | max pen. rel. | peak force rel. | CNUM abs. | contact-zone VM rel. |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in comparison_rows:
        lines.append(
            f"| {row['resolution']} | {row['acceptance_status']} | {_fmt(row['z_cm_l2_rel_error'])} | "
            f"{_fmt(row['max_penetration_rel_error'])} | {_fmt(row['peak_normal_force_rel_error'])} | "
            f"{_fmt(row['max_cnum_abs_error'])} | {_fmt(row['contact_zone_max_vm_rel_error'])} |"
        )
    lines.extend(
        [
            "",
            "## Alignment Diagnostics",
            "",
            "| Resolution | Diagnosis | peak RF replay rel. | peak CELS replay rel. | force-law max rel. | kscale restores | max attempt | min CalculiX dt | min SFC dt |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in alignment_rows:
        lines.append(
            f"| {row['resolution']} | {row['diagnosis']} | {_fmt(row['peak_rf_force_replay_rel_error'])} | "
            f"{_fmt(row['peak_cels_energy_replay_rel_error'])} | {_fmt(row['force_law_max_rel_error'])} | "
            f"{_fmt(row['calculix_stdout_kscale_restore_count'])} | "
            f"{_fmt(row['calculix_stdout_max_attempt'])} | {_fmt(row['calculix_stdout_min_increment_size'])} | "
            f"{_fmt(row['sfc_min_accepted_dt'])} |"
        )
    lines.extend(
        [
            "",
            "## HHT Residual/Tangent Diagnostics",
            "",
            "| Resolution | Contact mode | time | active | HHT tangent FD rel. | contact tangent FD rel. | previous-static update rel. |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in hht_rows:
        lines.append(
            f"| {row['resolution']} | {row['contact_mode']} | {_fmt(row['time'])} | "
            f"{_fmt(row['active_contact_count'])} | {_fmt(row['effective_tangent_directional_fd_rel_error'])} | "
            f"{_fmt(row['contact_residual_tangent_directional_fd_rel_error'])} | "
            f"{_fmt(row['previous_static_residual_update_rel_error'])} |"
        )
    lines.extend(
        [
            "",
            "## Lifecycle And Output Diagnostics",
            "",
            "| Resolution | Diagnosis | CalculiX CNUM sequence | SFC CNUM sequence | Replay-on-CalculiX CNUM sequence | SFC CNUM max abs. | Replay CNUM max abs. | RF peak time error | CELS peak time error |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in lifecycle_rows:
        lines.append(
            f"| {row['resolution']} | {row['diagnosis']} | `{row['calculix_cnum_sequence']}` | "
            f"`{row['sfc_cnum_sequence']}` | `{row.get('calculix_displacement_replay_cnum_sequence', '')}` | "
            f"{_fmt(row['cnum_max_abs_error'])} | {_fmt(row.get('calculix_displacement_replay_cnum_max_abs_error', ''))} | "
            f"{_fmt(row['force_peak_time_abs_error'])} | {_fmt(row['energy_peak_time_abs_error'])} |"
        )
    lines.extend(
        [
            "",
            "## Trend Rows",
            "",
            f"- Mesh convergence rows: `{len(mesh_rows)}`",
            f"- Time-step convergence rows: `{len(timestep_rows)}`",
            f"- VTK stress cloud rows: `{len(vtk_rows)}`",
            "",
            "## Limitations",
            "",
            "- This validates the diagnostic clean-room StVK TET4 contact path, not a copied CalculiX implementation.",
            "- The mechanics backend is swappable with SFC dynamic-SDF contact geometry; this runner uses rigid-plane contact for external alignment.",
            "- The strict mode still does not copy CalculiX source and does not reproduce every persistent active-set/cutback branch of CalculiX.",
            "- Passing the scoped gate is not a source-level CalculiX contact equivalence claim.",
            "- If CalculiX does not export RF/contact-energy data, the affected force/energy comparison fields remain blank.",
            "- Mesh/time-step trend files do not claim theoretical convergence order.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.6e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "geometric_nonlinear_contact_validation")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-calculix", action="store_true")
    parser.add_argument(
        "--contact-mode",
        choices=[
            "calculix_c3d4_f2f",
            "persistent_calculix_c3d4_f2f",
            "dynamic_sdf_calculix_f2f",
            "persistent_dynamic_sdf_calculix_f2f",
            "plane",
            "dynamic_sdf_plane",
        ],
        default="persistent_calculix_c3d4_f2f",
        help="SFC contact geometry/enforcement mode used for the validation run.",
    )
    parser.add_argument(
        "--cutback-policy",
        choices=["active_retry", "calculix_direct"],
        default="calculix_direct",
        help="SFC trial-step retry policy. 'calculix_direct' keeps fixed direct increments to match the generated CalculiX input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(
        args.out_dir,
        quick=args.quick,
        skip_calculix=args.skip_calculix,
        contact_mode=args.contact_mode,
        cutback_policy=args.cutback_policy,
    )
    print("Geometric nonlinear contact validation complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
