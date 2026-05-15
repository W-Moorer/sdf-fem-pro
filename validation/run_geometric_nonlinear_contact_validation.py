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
from scipy.sparse.linalg import spsolve

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
    CALCULIX_TRIANGLE_CONTACT_BARYCENTRIC,
    CALCULIX_TRIANGLE_CONTACT_WEIGHTS,
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
    static_force_state,
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
    frequency = max(1, int(model.output_frequency))
    per_contact_block = "\n".join(
        [
            f"*contact print, frequency={frequency}",
            "cdis",
            "cstr",
            "cels",
            f"*contact print, frequency={frequency}, totals=only",
            "cnum",
        ]
    )
    text = re.sub(
        r"\*contact print[^\n]*totals=only[^\n]*\n[ \t]*cdis[ \t]*,[ \t]*cstr[ \t]*,[ \t]*cels[ \t]*,[ \t]*cnum[ \t]*",
        per_contact_block,
        text,
        flags=re.IGNORECASE,
    )
    stress_block = "\n".join(
        [
            f"*el print, elset=elall, frequency={frequency}",
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
    per_contact = _parse_calculix_dat_contact_elements(dat)
    face_lookup = _face_ref_index(model)
    for time, records in per_contact.items():
        disp_time = min(displacements, key=lambda value: abs(float(value) - float(time))) if displacements else None
        current_areas: np.ndarray | None = None
        if disp_time is not None:
            tri = (model.nodes + displacements[disp_time])[model.surface_faces]
            current_areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        energy_values = [
            float(row["calculix_contact_energy"])
            for row in records.values()
            if row.get("calculix_contact_energy", "") not in {"", None}
        ]
        derived_energy_values: list[float] = []
        if not energy_values and current_areas is not None:
            for key, row in records.items():
                face_index = face_lookup.get((int(key[0]), int(key[1])), -1)
                pressure = _optional_float(row.get("calculix_stress_normal", ""))
                clearance = _optional_float(row.get("calculix_clearance_normal", ""))
                if face_index >= 0 and pressure is not None and clearance is not None:
                    derived_energy_values.append(
                        0.5 * abs(float(pressure)) * float(current_areas[face_index]) * max(-float(clearance), 0.0)
                    )
        if (energy_values or derived_energy_values) and "calculix_contact_energy" not in contact_totals.setdefault(float(time), {}):
            contact_totals[float(time)]["calculix_contact_energy"] = float(np.sum(energy_values or derived_energy_values))
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
    contact_element_audit_rows: list[Row] | None = None,
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
    _append_sfc_contact_element_audit_rows(
        contact_element_audit_rows,
        model,
        state,
        contact,
        source=source,
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
            _append_sfc_contact_element_audit_rows(
                contact_element_audit_rows,
                model,
                state,
                contact,
                source=source,
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
    acceptance = _last_newton_acceptance_metric(diagnostics)
    metrics = getattr(diagnostics, "newton_acceptance_metrics", [])
    max_contact_change = max((int(metric.contact_element_change) for metric in metrics), default="")
    any_energy_stabilization = any(bool(metric.energy_stabilization) for metric in metrics)
    any_calculix_style_rejection = any(not bool(metric.accepted_by_calculix_style) for metric in metrics)
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
        "newton_acceptance_policy": diagnostics.newton_acceptance_policy,
        "newton_acceptance_reason": diagnostics.newton_acceptance_reason,
        "newton_ram": "" if acceptance is None else acceptance.ram,
        "newton_qam": "" if acceptance is None else acceptance.qam,
        "newton_cam": "" if acceptance is None else acceptance.cam,
        "newton_uam": "" if acceptance is None else acceptance.uam,
        "newton_residual_ratio": "" if acceptance is None else acceptance.residual_ratio,
        "newton_correction_ratio": "" if acceptance is None else acceptance.correction_ratio,
        "newton_contact_element_change": "" if acceptance is None else acceptance.contact_element_change,
        "newton_max_contact_element_change": max_contact_change,
        "newton_energy_residual": "" if acceptance is None else acceptance.energy_residual,
        "newton_energy_stabilization": "" if acceptance is None else str(acceptance.energy_stabilization).lower(),
        "newton_any_energy_stabilization": "" if not metrics else str(any_energy_stabilization).lower(),
        "newton_calculix_style_accepted": "" if acceptance is None else str(acceptance.accepted_by_calculix_style).lower(),
        "newton_any_calculix_style_rejection": "" if not metrics else str(any_calculix_style_rejection).lower(),
        "newton_calculix_style_reason": "" if acceptance is None else acceptance.reason,
        "contact_cutback_recommended": str(convergence_record.recommended_cutback).lower(),
        "contact_convergence_reason": convergence_record.reason,
        "details": details,
    }


def _last_newton_acceptance_metric(diagnostics: StepDiagnostics) -> Any | None:
    metrics = getattr(diagnostics, "newton_acceptance_metrics", [])
    return metrics[-1] if metrics else None


def contact_replay_metrics(model: DropModel, x_current: np.ndarray, *, plane_z: float | None = None) -> Row:
    """Return hard-linear contact metrics replayed on a current geometry."""

    z = _contact_plane_z(model) if plane_z is None else float(plane_z)
    tri = np.asarray(x_current, dtype=float)[model.surface_faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    q_points = np.einsum("qn,fnc->fqc", CALCULIX_TRIANGLE_CONTACT_BARYCENTRIC, tri)
    gaps = q_points[:, :, 2] - z
    penetration = np.maximum(-gaps, 0.0)
    active = int(np.count_nonzero(penetration > 0.0))
    q_areas = areas[:, None] * CALCULIX_TRIANGLE_CONTACT_WEIGHTS[None, :]
    force = float(model.contact_stiffness * np.sum(q_areas * penetration))
    energy = float(0.5 * model.contact_stiffness * np.sum(q_areas * penetration * penetration))
    return {
        "min_gap": float(np.min(gaps)) if gaps.size else 0.0,
        "max_penetration": float(np.max(penetration)) if penetration.size else 0.0,
        "active_force_spring_count": active,
        "cnum_equivalent": active,
        "normal_force": force,
        "contact_energy": energy,
    }


def _parse_calculix_dat_contact_elements(path: Path) -> dict[float, dict[tuple[int, int, int], Row]]:
    """Parse per-contact-element CDIS/CSTR/CELS blocks from a CalculiX ``.dat`` file.

    CalculiX writes face-to-face contact rows as ``slave element, slave face``
    followed by either three components (CDIS/CSTR) or one scalar (CELS).
    Totals-only blocks are intentionally ignored here.
    """

    number = r"([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    headers = {
        "cdis": re.compile(r"relative contact displacement .* time\s+" + number, re.IGNORECASE),
        "cstr": re.compile(r"contact stress .* time\s+" + number, re.IGNORECASE),
        "cels": re.compile(r"contact spring energy .* time\s+" + number, re.IGNORECASE),
    }
    vector_row = re.compile(r"^\s*(\d+)\s+(\d+)\s+" + number + r"\s+" + number + r"\s+" + number + r"\s*$")
    scalar_row = re.compile(r"^\s*(\d+)\s+(\d+)\s+" + number + r"\s*$")
    parsed: dict[float, dict[tuple[int, int, int], Row]] = {}
    current_kind: str | None = None
    current_time: float | None = None
    occurrence_counts: dict[tuple[str, float, int, int], int] = {}

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        header_matched = False
        for kind, pattern in headers.items():
            match = pattern.search(line)
            if match:
                current_kind = kind
                current_time = float(match.group(1))
                parsed.setdefault(current_time, {})
                header_matched = True
                break
        if header_matched:
            continue
        if current_kind is None or current_time is None:
            continue
        if not line.strip():
            continue

        if current_kind in {"cdis", "cstr"}:
            match = vector_row.match(line)
            if match is None:
                if re.search(r"[A-Za-z]", line):
                    current_kind = None
                    current_time = None
                continue
            element = int(match.group(1))
            face = int(match.group(2))
            occurrence_key = (current_kind, float(current_time), element, face)
            occurrence_index = occurrence_counts.get(occurrence_key, 0)
            occurrence_counts[occurrence_key] = occurrence_index + 1
            values = [float(match.group(i)) for i in range(3, 6)]
            row = parsed.setdefault(current_time, {}).setdefault((element, face, occurrence_index), {})
            row["contact_element_index"] = occurrence_index
            prefix = "calculix_clearance" if current_kind == "cdis" else "calculix_stress"
            row[f"{prefix}_normal"] = values[0]
            row[f"{prefix}_tangential_1"] = values[1]
            row[f"{prefix}_tangential_2"] = values[2]
        else:
            match = scalar_row.match(line)
            if match is None:
                if re.search(r"[A-Za-z]", line):
                    current_kind = None
                    current_time = None
                continue
            element = int(match.group(1))
            face = int(match.group(2))
            occurrence_key = (current_kind, float(current_time), element, face)
            occurrence_index = occurrence_counts.get(occurrence_key, 0)
            occurrence_counts[occurrence_key] = occurrence_index + 1
            energy = float(match.group(3))
            row = parsed.setdefault(current_time, {}).setdefault((element, face, occurrence_index), {})
            row["contact_element_index"] = occurrence_index
            row["calculix_contact_energy"] = energy
    return dict(sorted(parsed.items()))


def _face_ref_key(model: DropModel, face_index: int) -> tuple[int, int]:
    element_id, label = model.slave_face_refs[int(face_index)]
    match = re.search(r"(\d+)", str(label))
    if match is None:
        raise ValueError(f"unsupported CalculiX face label: {label!r}")
    return int(element_id), int(match.group(1))


def _face_ref_index(model: DropModel) -> dict[tuple[int, int], int]:
    return {_face_ref_key(model, i): i for i in range(len(model.slave_face_refs))}


def _contact_springs_by_contact_element(contact: ContactGeometry, x_current: np.ndarray) -> dict[tuple[int, int], Any]:
    lifecycle = getattr(contact, "lifecycle", None)
    active_springs = getattr(lifecycle, "active_springs", None)
    if isinstance(active_springs, dict):
        indexed: dict[tuple[int, int], Any] = {}
        for key, spring in active_springs.items():
            if isinstance(key, tuple):
                indexed[(int(key[0]), int(key[1]))] = spring
            else:
                indexed[(int(key), 0)] = spring
        return indexed
    if hasattr(contact, "contact_springs"):
        springs = getattr(contact, "contact_springs")(x_current)
        return {
            (int(spring.slave_face_index), int(getattr(spring, "slave_quadrature_index", 0))): spring
            for spring in springs
            if bool(getattr(spring, "active", False))
        }
    return {}


def _per_face_plane_contact_rows(
    model: DropModel,
    x_current: np.ndarray,
    *,
    time: float,
    source: str,
    generated_springs: dict[tuple[int, int], Any] | None = None,
) -> list[Row]:
    """Return per-contact-element clearance, force, and energy rows for a plane replay."""

    x = np.asarray(x_current, dtype=float)
    tri = x[model.surface_faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    q_points = np.einsum("qn,fnc->fqc", CALCULIX_TRIANGLE_CONTACT_BARYCENTRIC, tri)
    q_gaps = q_points[:, :, 2] - _contact_plane_z(model)
    generated = {} if generated_springs is None else generated_springs
    rows: list[Row] = []
    for face_index, area in enumerate(areas):
        element_id, face_number = _face_ref_key(model, face_index)
        for q_index, q_weight in enumerate(CALCULIX_TRIANGLE_CONTACT_WEIGHTS):
            spring = generated.get((int(face_index), int(q_index)))
            if spring is not None:
                clearance = float(spring.clearance)
                spring_area = float(spring.spring_area)
                generated_flag = True
            else:
                clearance = float(q_gaps[face_index, q_index])
                spring_area = float(area * q_weight)
                generated_flag = bool(max(-clearance, 0.0) > 0.0) if source == "calculix_displacement_replay" else False
            penetration = max(-clearance, 0.0)
            force = model.contact_stiffness * spring_area * penetration if generated_flag or penetration > 0.0 else 0.0
            energy = 0.5 * model.contact_stiffness * spring_area * penetration * penetration if generated_flag or penetration > 0.0 else 0.0
            rows.append(
                {
                    "case": model.case,
                    "resolution": model.resolution,
                    "source": source,
                    "time": float(time),
                    "surface_face_index": int(face_index),
                    "contact_element_index": int(q_index),
                    "slave_element": int(element_id),
                    "slave_face": int(face_number),
                    "slave_face_label": f"S{face_number}",
                    "face_area": float(spring_area),
                    "clearance": float(clearance),
                    "active_spring": str(bool(generated_flag)).lower(),
                    "force": float(force),
                    "energy": float(energy),
                }
            )
    return rows


def _append_sfc_contact_element_audit_rows(
    rows: list[Row] | None,
    model: DropModel,
    state: MechanicsState,
    contact: ContactGeometry,
    *,
    source: str,
) -> None:
    if rows is None:
        return
    rows.extend(
        _per_face_plane_contact_rows(
            model,
            state.x,
            time=float(state.time),
            source=source,
            generated_springs=_contact_springs_by_contact_element(contact, state.x),
        )
    )


def _time_key(value: float) -> float:
    return round(float(value), 12)


def _index_contact_rows(rows: list[Row]) -> dict[float, dict[tuple[int, int, int], Row]]:
    indexed: dict[float, dict[tuple[int, int, int], Row]] = {}
    for row in rows:
        time = _time_key(float(row["time"]))
        key = (
            int(row["slave_element"]),
            int(row["slave_face"]),
            int(row.get("contact_element_index", 0)),
        )
        indexed.setdefault(time, {})[key] = row
    return indexed


def _calculix_contact_rows_from_dat(
    model: DropModel,
    parsed: dict[float, dict[tuple[int, int, int], Row]],
    replay_by_time: dict[float, dict[tuple[int, int, int], Row]],
) -> list[Row]:
    rows: list[Row] = []
    face_lookup = _face_ref_index(model)
    for time, records in parsed.items():
        time_key = _time_key(time)
        replay = replay_by_time.get(time_key, {})
        for (element_id, face_number, contact_element_index), values in records.items():
            face_index = face_lookup.get((int(element_id), int(face_number)), -1)
            area = ""
            replay_key = (int(element_id), int(face_number), int(contact_element_index))
            if replay_key in replay:
                area = replay[replay_key]["face_area"]
            elif face_index >= 0:
                tri = model.nodes[model.surface_faces[face_index]]
                area = (
                    0.5
                    * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
                    * float(CALCULIX_TRIANGLE_CONTACT_WEIGHTS[int(contact_element_index) % CALCULIX_TRIANGLE_CONTACT_WEIGHTS.size])
                )
            normal_pressure = _optional_float(values.get("calculix_stress_normal", ""))
            force = "" if normal_pressure is None or area == "" else abs(normal_pressure) * float(area)
            cels_energy = values.get("calculix_contact_energy", "")
            clearance = values.get("calculix_clearance_normal", "")
            clearance_value = _optional_float(clearance)
            if cels_energy not in {"", None}:
                energy = cels_energy
                energy_source = "cels"
            elif normal_pressure is not None and clearance_value is not None and area != "":
                energy = 0.5 * abs(normal_pressure) * float(area) * max(-float(clearance_value), 0.0)
                energy_source = "derived_from_cdis_cstr"
            else:
                energy = ""
                energy_source = "unavailable"
            rows.append(
                {
                    "case": model.case,
                    "resolution": model.resolution,
                    "source": "calculix_dat",
                    "time": float(time),
                    "surface_face_index": int(face_index),
                    "contact_element_index": int(contact_element_index),
                    "slave_element": int(element_id),
                    "slave_face": int(face_number),
                    "slave_face_label": f"S{int(face_number)}",
                    "face_area": area,
                    "clearance": clearance,
                    "active_spring": "true",
                    "force": force,
                    "energy": energy,
                    "calculix_cels_energy": cels_energy,
                    "calculix_energy_source": energy_source,
                    "calculix_clearance_tangential_1": values.get("calculix_clearance_tangential_1", ""),
                    "calculix_clearance_tangential_2": values.get("calculix_clearance_tangential_2", ""),
                    "calculix_stress_normal": values.get("calculix_stress_normal", ""),
                    "calculix_stress_tangential_1": values.get("calculix_stress_tangential_1", ""),
                    "calculix_stress_tangential_2": values.get("calculix_stress_tangential_2", ""),
                }
            )
    return rows


def contact_element_clearance_lifecycle_audit(
    model: DropModel,
    *,
    calculix_dat_contact: dict[float, dict[tuple[int, int, int], Row]],
    sfc_native_rows: list[Row],
    calculix_displacements: dict[float, np.ndarray],
) -> tuple[list[Row], list[Row]]:
    """Return per-contact-element and per-step lifecycle audit rows."""

    replay_rows: list[Row] = []
    for time, displacement in sorted(calculix_displacements.items()):
        replay_rows.extend(
            _per_face_plane_contact_rows(
                model,
                model.nodes + displacement,
                time=float(time),
                source="calculix_displacement_replay",
            )
        )

    replay_by_time = _index_contact_rows(replay_rows)
    calc_rows = _calculix_contact_rows_from_dat(model, calculix_dat_contact, replay_by_time)
    calc_by_time = _index_contact_rows(calc_rows)
    native_by_time = _index_contact_rows(sfc_native_rows)
    times = sorted(set(calc_by_time) | set(native_by_time) | set(replay_by_time))
    face_keys = sorted(
        (element_id, face_number, q_index)
        for element_id, face_number in _face_ref_index(model)
        for q_index in range(CALCULIX_TRIANGLE_CONTACT_WEIGHTS.size)
    )
    audit_rows: list[Row] = []
    summary_rows: list[Row] = []

    for time in times:
        calc = calc_by_time.get(time, {})
        native = native_by_time.get(time, {})
        replay = replay_by_time.get(time, {})
        keys = sorted(set(face_keys) | set(calc) | set(native) | set(replay))
        clearance_errors: list[float] = []
        force_errors: list[float] = []
        energy_errors: list[float] = []
        for element_id, face_number, contact_element_index in keys:
            key = (element_id, face_number, contact_element_index)
            calc_row = calc.get(key, {})
            native_row = native.get(key, {})
            replay_row = replay.get(key, {})
            face_index = native_row.get("surface_face_index", replay_row.get("surface_face_index", calc_row.get("surface_face_index", "")))
            area = native_row.get("face_area", replay_row.get("face_area", calc_row.get("face_area", "")))
            calc_clearance = calc_row.get("clearance", "")
            replay_clearance = replay_row.get("clearance", "")
            calc_force = calc_row.get("force", "")
            replay_force = replay_row.get("force", "")
            calc_energy = calc_row.get("energy", "")
            replay_energy = replay_row.get("energy", "")
            _append_abs_error(clearance_errors, calc_clearance, replay_clearance)
            _append_abs_error(force_errors, calc_force, replay_force)
            _append_abs_error(energy_errors, calc_energy, replay_energy)
            audit_rows.append(
                {
                    "case": model.case,
                    "resolution": model.resolution,
                    "time": float(time),
                    "surface_face_index": face_index,
                    "contact_element_index": int(contact_element_index),
                    "slave_element": int(element_id),
                    "slave_face": int(face_number),
                    "slave_face_label": f"S{int(face_number)}",
                    "face_area": area,
                    "calculix_present": str(bool(calc_row)).lower(),
                    "sfc_native_active_spring": native_row.get("active_spring", "false"),
                    "calculix_displacement_replay_active_spring": replay_row.get("active_spring", "false"),
                    "calculix_clearance": calc_clearance,
                    "calculix_clearance_tangential_1": calc_row.get("calculix_clearance_tangential_1", ""),
                    "calculix_clearance_tangential_2": calc_row.get("calculix_clearance_tangential_2", ""),
                    "calculix_normal_pressure": calc_row.get("calculix_stress_normal", ""),
                    "calculix_tangential_stress_1": calc_row.get("calculix_stress_tangential_1", ""),
                    "calculix_tangential_stress_2": calc_row.get("calculix_stress_tangential_2", ""),
                    "calculix_force_proxy": calc_force,
                    "calculix_contact_energy": calc_energy,
                    "calculix_cels_energy": calc_row.get("calculix_cels_energy", ""),
                    "calculix_contact_energy_source": calc_row.get("calculix_energy_source", ""),
                    "sfc_native_clearance": native_row.get("clearance", ""),
                    "sfc_native_force": native_row.get("force", ""),
                    "sfc_native_energy": native_row.get("energy", ""),
                    "calculix_displacement_replay_clearance": replay_clearance,
                    "calculix_displacement_replay_force": replay_force,
                    "calculix_displacement_replay_energy": replay_energy,
                }
            )
        calc_count = sum(1 for row in calc.values() if row.get("active_spring") == "true")
        native_count = sum(1 for row in native.values() if row.get("active_spring") == "true")
        replay_count = sum(1 for row in replay.values() if row.get("active_spring") == "true")
        summary_rows.append(
            {
                "case": model.case,
                "resolution": model.resolution,
                "time": float(time),
                "calculix_active_contact_elements": int(calc_count),
                "sfc_native_active_springs": int(native_count),
                "calculix_displacement_replay_active_springs": int(replay_count),
                "calculix_per_contact_output_available": str(bool(calc)).lower(),
                "clearance_replay_linf_error": _max_or_blank(clearance_errors),
                "force_replay_linf_error": _max_or_blank(force_errors),
                "energy_replay_linf_error": _max_or_blank(energy_errors),
                "diagnosis": _contact_element_audit_diagnosis(calc_count, native_count, replay_count),
            }
        )
    if not summary_rows:
        summary_rows.append(
            {
                "case": model.case,
                "resolution": model.resolution,
                "time": "",
                "calculix_active_contact_elements": "",
                "sfc_native_active_springs": "",
                "calculix_displacement_replay_active_springs": "",
                "calculix_per_contact_output_available": "false",
                "clearance_replay_linf_error": "",
                "force_replay_linf_error": "",
                "energy_replay_linf_error": "",
                "diagnosis": "external_unavailable",
            }
        )
    return audit_rows, summary_rows


def _append_abs_error(errors: list[float], a: Any, b: Any) -> None:
    aval = _optional_float(a)
    bval = _optional_float(b)
    if aval is not None and bval is not None:
        errors.append(abs(float(aval) - float(bval)))


def _contact_element_audit_diagnosis(calc_count: int, native_count: int, replay_count: int) -> str:
    if calc_count <= 0 and native_count <= 0 and replay_count <= 0:
        return "no_contact"
    if calc_count == replay_count == native_count:
        return "active_sets_aligned"
    if calc_count == replay_count and calc_count != native_count:
        return "native_trajectory_or_lifecycle_differs_from_calculix_displacement_replay"
    if calc_count != replay_count:
        return "calculix_contact_lifecycle_or_clearance_output_differs_from_replay"
    return "active_set_difference_observed"


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


def one_step_calculix_state_diagnostics(
    model: DropModel,
    calculix_rows: list[Row],
    calculix_displacements: dict[float, np.ndarray],
    *,
    contact_mode: str = "persistent_calculix_c3d4_f2f",
) -> list[Row]:
    """Compare one SFC step launched from each available CalculiX state.

    CalculiX `.dat` output does not expose the full global internal-force vector
    or tangent matrix.  This diagnostic therefore keeps those columns explicitly
    marked unavailable and compares the SFC decomposition evaluated on the same
    CalculiX kinematic states against the SFC one-step prediction from that
    state.
    """

    if len(calculix_displacements) < 1:
        return [_one_step_unavailable_row(model, contact_mode, "external_unavailable")]

    mechanics = MechanicsModel.from_tet4_mesh(
        model.nodes,
        model.tet_elements,
        E=model.E,
        nu=model.nu,
        density=model.density,
    )
    contact, details = _make_contact_geometry(model, contact_mode)
    beta, gamma = hht_newmark_parameters(model.hht_alpha)
    ordered = _ordered_calculix_displacements(model, calculix_displacements)
    if len(ordered) < 2:
        return [_one_step_unavailable_row(model, contact_mode, "insufficient_calculix_time_states")]

    state, previous_static = initial_state(
        mechanics,
        contact,
        gravity=model.gravity,
        initial_velocity=(0.0, 0.0, model.initial_velocity_z),
    )
    state.time = float(ordered[0][0])
    rows: list[Row] = []
    for next_time, next_u in ordered[1:]:
        dt = float(next_time) - float(state.time)
        if dt <= 0.0:
            continue
        state_n = MechanicsState(state.x.copy(), state.v.copy(), state.a.copy(), time=state.time)
        previous_static_n = previous_static.copy()
        snapshot_n = _snapshot_contact_lifecycle(contact)

        pred_state, _pred_static, pred_diag = hht_step(
            mechanics,
            state_n,
            previous_static_n,
            contact,
            dt=dt,
            gravity=model.gravity,
            alpha=model.hht_alpha,
            max_iterations=12,
            tolerance=1.0e-10,
        )
        _restore_contact_lifecycle(contact, snapshot_n)

        calc_state = _newmark_state_from_displacement(
            mechanics,
            state_n,
            np.asarray(next_u, dtype=float),
            next_time=float(next_time),
            dt=dt,
            beta=beta,
            gamma=gamma,
        )
        calc_residual, calc_tangent, calc_static, _calc_static_tangent = _hht_dynamic_residual_and_tangent(
            mechanics,
            contact,
            state_n,
            previous_static_n,
            calc_state.x.reshape(-1),
            dt=dt,
            gravity=model.gravity,
            alpha=model.hht_alpha,
        )
        calc_diag = evaluate_state(mechanics, calc_state, contact, gravity=model.gravity, assemble_tangent=True)
        rows.append(
            _one_step_row(
                model,
                mechanics,
                state_n,
                calc_state,
                pred_state,
                calc_diag,
                pred_diag,
                calc_residual,
                calc_tangent,
                previous_static_n,
                calculix_rows,
                contact_mode=contact_mode,
                details=details,
                dt=dt,
            )
        )
        state = calc_state
        previous_static = -calc_static
    if not rows:
        return [_one_step_unavailable_row(model, contact_mode, "insufficient_positive_time_steps")]
    return rows


def _ordered_calculix_displacements(
    model: DropModel,
    displacements: dict[float, np.ndarray],
) -> list[tuple[float, np.ndarray]]:
    rows: list[tuple[float, np.ndarray]] = [(0.0, np.zeros_like(model.nodes))]
    for time in sorted(displacements):
        if float(time) <= 1.0e-15:
            continue
        rows.append((float(time), np.asarray(displacements[time], dtype=float)))
    return rows


def _newmark_state_from_displacement(
    mechanics: MechanicsModel,
    previous: MechanicsState,
    displacement: np.ndarray,
    *,
    next_time: float,
    dt: float,
    beta: float,
    gamma: float,
) -> MechanicsState:
    u_n = (previous.x - mechanics.X).reshape(-1)
    v_n = previous.v.reshape(-1)
    a_n = previous.a.reshape(-1)
    u_next = np.asarray(displacement, dtype=float).reshape(-1)
    u_pred = u_n + float(dt) * v_n + float(dt) * float(dt) * (0.5 - beta) * a_n
    a_next = (u_next - u_pred) / (beta * float(dt) * float(dt))
    v_next = v_n + float(dt) * ((1.0 - gamma) * a_n + gamma * a_next)
    return MechanicsState(
        mechanics.X + u_next.reshape((-1, 3)),
        v_next.reshape((-1, 3)),
        a_next.reshape((-1, 3)),
        time=float(next_time),
    )


def _one_step_row(
    model: DropModel,
    mechanics: MechanicsModel,
    state_n: MechanicsState,
    calc_state: MechanicsState,
    pred_state: MechanicsState,
    calc_diag: StepDiagnostics,
    pred_diag: StepDiagnostics,
    calc_residual: np.ndarray,
    calc_tangent: Any,
    previous_static_n: np.ndarray,
    calculix_rows: list[Row],
    *,
    contact_mode: str,
    details: str,
    dt: float,
) -> Row:
    mass_calc = mechanics.mass_matrix @ calc_state.a.reshape(-1)
    mass_pred = mechanics.mass_matrix @ pred_state.a.reshape(-1)
    internal_calc = calc_diag.internal.force.reshape(-1)
    internal_pred = pred_diag.internal.force.reshape(-1)
    contact_calc = calc_diag.contact.force.reshape(-1)
    contact_pred = pred_diag.contact.force.reshape(-1)
    static_calc = internal_calc - _nodal_gravity_like(model, mechanics) - contact_calc
    static_pred = internal_pred - _nodal_gravity_like(model, mechanics) - contact_pred
    hht_required_mass_term = (
        (1.0 + float(model.hht_alpha)) * (-static_calc)
        - float(model.hht_alpha) * np.asarray(previous_static_n, dtype=float)
    )
    force_balance_acceleration = np.asarray(spsolve(mechanics.mass_matrix.tocsc(), hht_required_mass_term), dtype=float)
    calc_row = _nearest_row(calculix_rows, float(calc_state.time)) if calculix_rows else None
    calc_rf = _optional_float(calc_row.get("normal_force_proxy", "")) if calc_row else None
    calc_cnum = _optional_float(calc_row.get("calculix_contact_count", "")) if calc_row else None
    component_diffs = {
        "mass_term": _vector_norm(mass_pred - mass_calc),
        "static_residual": _vector_norm(static_pred - static_calc),
        "internal_force": _vector_norm(internal_pred - internal_calc),
        "contact_force": _vector_norm(contact_pred - contact_calc),
        "acceleration": _vector_norm(pred_state.a.reshape(-1) - calc_state.a.reshape(-1)),
    }
    dominant = max(component_diffs, key=component_diffs.get)
    displacement_error = _vector_norm((pred_state.x - calc_state.x).reshape(-1))
    displacement_scale = max(_vector_norm((calc_state.x - state_n.x).reshape(-1)), 1.0e-30)
    acceleration_error = _relative_vector_error(pred_state.a.reshape(-1), calc_state.a.reshape(-1))
    contact_force_rel_to_calculix_rf = (
        ""
        if calc_rf is None
        else abs(float(calc_diag.contact.normal_force) - calc_rf) / max(abs(calc_rf), 1.0e-30)
    )
    residual_scale = max(
        _vector_norm(mass_calc)
        + _vector_norm((1.0 + float(model.hht_alpha)) * static_calc)
        + _vector_norm(float(model.hht_alpha) * previous_static_n),
        1.0e-30,
    )
    return {
        "case": model.case,
        "resolution": model.resolution,
        "contact_mode": contact_mode,
        "time_n": float(state_n.time),
        "time_np1": float(calc_state.time),
        "dt": float(dt),
        "dt_matches_model_dt": str(abs(float(dt) - float(model.dt)) <= 1.0e-12).lower(),
        "calculix_internal_force_available": "false",
        "calculix_effective_tangent_available": "false",
        "calculix_unavailable_reason": "CalculiX .dat does not print full global internal force vectors or tangent matrices",
        "calculix_rf_z": "" if calc_rf is None else calc_rf,
        "calculix_cnum": "" if calc_cnum is None else calc_cnum,
        "sfc_on_calculix_active_count": int(calc_diag.contact.active_count),
        "sfc_predicted_active_count": int(pred_diag.contact.active_count),
        "sfc_on_calculix_max_penetration": float(calc_diag.contact.max_penetration),
        "sfc_predicted_max_penetration": float(pred_diag.contact.max_penetration),
        "sfc_on_calculix_contact_force_z": float(calc_diag.contact.normal_force),
        "sfc_predicted_contact_force_z": float(pred_diag.contact.normal_force),
        "contact_force_rel_error_to_calculix_rf_at_calculix_state": contact_force_rel_to_calculix_rf,
        "sfc_on_calculix_internal_force_norm": _vector_norm(internal_calc),
        "sfc_predicted_internal_force_norm": _vector_norm(internal_pred),
        "sfc_on_calculix_contact_force_norm": _vector_norm(contact_calc),
        "sfc_predicted_contact_force_norm": _vector_norm(contact_pred),
        "sfc_on_calculix_mass_term_norm": _vector_norm(mass_calc),
        "sfc_predicted_mass_term_norm": _vector_norm(mass_pred),
        "sfc_hht_required_mass_term_norm_at_calculix_state": _vector_norm(hht_required_mass_term),
        "sfc_hht_required_mass_term_rel_to_reconstructed": _relative_vector_error(hht_required_mass_term, mass_calc),
        "sfc_hht_force_balance_acceleration_norm_at_calculix_state": _vector_norm(force_balance_acceleration),
        "sfc_hht_force_balance_vs_reconstructed_acceleration_rel": _relative_vector_error(
            force_balance_acceleration,
            calc_state.a.reshape(-1),
        ),
        "sfc_on_calculix_static_residual_norm": _vector_norm(static_calc),
        "sfc_predicted_static_residual_norm": _vector_norm(static_pred),
        "sfc_hht_effective_residual_norm_at_calculix_state": _vector_norm(calc_residual),
        "sfc_hht_effective_residual_relative_at_calculix_state": _vector_norm(calc_residual) / residual_scale,
        "sfc_effective_tangent_norm_at_calculix_state": _sparse_data_norm(calc_tangent),
        "sfc_material_tangent_norm_at_calculix_state": _sparse_data_norm(calc_diag.internal.material_tangent),
        "sfc_geometric_tangent_norm_at_calculix_state": _sparse_data_norm(calc_diag.internal.geometric_tangent),
        "sfc_contact_tangent_norm_at_calculix_state": _sparse_data_norm(calc_diag.contact.tangent),
        "sfc_predicted_vs_calculix_displacement_norm": displacement_error,
        "sfc_predicted_vs_calculix_displacement_rel": displacement_error / displacement_scale,
        "sfc_predicted_vs_calculix_acceleration_rel": acceleration_error,
        "mass_term_diff_norm": component_diffs["mass_term"],
        "static_residual_diff_norm": component_diffs["static_residual"],
        "internal_force_diff_norm": component_diffs["internal_force"],
        "contact_force_diff_norm": component_diffs["contact_force"],
        "acceleration_diff_norm": component_diffs["acceleration"],
        "dominant_one_step_difference_source": dominant,
        "diagnosis": _one_step_diagnosis(calc_diag, pred_diag, displacement_error / displacement_scale, dominant),
        "details": details,
    }


def _one_step_unavailable_row(model: DropModel, contact_mode: str, diagnosis: str) -> Row:
    return {
        "case": model.case,
        "resolution": model.resolution,
        "contact_mode": contact_mode,
        "time_n": "",
        "time_np1": "",
        "dt": model.dt,
        "dt_matches_model_dt": "",
        "calculix_internal_force_available": "false",
        "calculix_effective_tangent_available": "false",
        "calculix_unavailable_reason": "CalculiX displacement/contact history unavailable",
        "calculix_rf_z": "",
        "calculix_cnum": "",
        "sfc_on_calculix_active_count": "",
        "sfc_predicted_active_count": "",
        "sfc_on_calculix_max_penetration": "",
        "sfc_predicted_max_penetration": "",
        "sfc_on_calculix_contact_force_z": "",
        "sfc_predicted_contact_force_z": "",
        "contact_force_rel_error_to_calculix_rf_at_calculix_state": "",
        "sfc_on_calculix_internal_force_norm": "",
        "sfc_predicted_internal_force_norm": "",
        "sfc_on_calculix_contact_force_norm": "",
        "sfc_predicted_contact_force_norm": "",
        "sfc_on_calculix_mass_term_norm": "",
        "sfc_predicted_mass_term_norm": "",
        "sfc_hht_required_mass_term_norm_at_calculix_state": "",
        "sfc_hht_required_mass_term_rel_to_reconstructed": "",
        "sfc_hht_force_balance_acceleration_norm_at_calculix_state": "",
        "sfc_hht_force_balance_vs_reconstructed_acceleration_rel": "",
        "sfc_on_calculix_static_residual_norm": "",
        "sfc_predicted_static_residual_norm": "",
        "sfc_hht_effective_residual_norm_at_calculix_state": "",
        "sfc_hht_effective_residual_relative_at_calculix_state": "",
        "sfc_effective_tangent_norm_at_calculix_state": "",
        "sfc_material_tangent_norm_at_calculix_state": "",
        "sfc_geometric_tangent_norm_at_calculix_state": "",
        "sfc_contact_tangent_norm_at_calculix_state": "",
        "sfc_predicted_vs_calculix_displacement_norm": "",
        "sfc_predicted_vs_calculix_displacement_rel": "",
        "sfc_predicted_vs_calculix_acceleration_rel": "",
        "mass_term_diff_norm": "",
        "static_residual_diff_norm": "",
        "internal_force_diff_norm": "",
        "contact_force_diff_norm": "",
        "acceleration_diff_norm": "",
        "dominant_one_step_difference_source": "",
        "diagnosis": diagnosis,
        "details": "",
    }


def _one_step_diagnosis(
    calc_diag: StepDiagnostics,
    pred_diag: StepDiagnostics,
    displacement_rel: float,
    dominant: str,
) -> str:
    if calc_diag.contact.active_count <= 0 and pred_diag.contact.active_count <= 0:
        return "precontact_or_no_active_contact"
    if displacement_rel <= 1.0e-3:
        return "one_step_update_close"
    if dominant == "contact_force":
        return "contact_force_difference_dominant"
    if dominant == "internal_force":
        return "internal_force_difference_dominant"
    if dominant in {"mass_term", "acceleration"}:
        return "newmark_acceleration_difference_dominant"
    return "static_residual_difference_dominant"


def _nodal_gravity_like(model: DropModel, mechanics: MechanicsModel) -> np.ndarray:
    values = np.zeros(mechanics.n_dofs, dtype=float)
    lumped = np.asarray(mechanics.mass_matrix.sum(axis=1)).reshape(-1)
    values[2::3] = -float(model.gravity) * lumped[2::3]
    return values


def _vector_norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=float)))


def _sparse_data_norm(matrix: Any) -> float:
    data = getattr(matrix, "data", np.asarray([], dtype=float))
    return float(np.linalg.norm(np.asarray(data, dtype=float))) if len(data) else 0.0


def hht_state_definition_diagnostics(
    model: DropModel,
    calculix_displacements: dict[float, np.ndarray],
    one_step_rows: list[Row],
    *,
    contact_mode: str = "persistent_calculix_c3d4_f2f",
) -> Row:
    """Summarize HHT/Newmark state-definition and precision checks."""

    mechanics = MechanicsModel.from_tet4_mesh(
        model.nodes,
        model.tet_elements,
        E=model.E,
        nu=model.nu,
        density=model.density,
    )
    contact, _details = _make_contact_geometry(model, contact_mode)
    beta, gamma = hht_newmark_parameters(model.hht_alpha)
    state0, static0 = initial_state(
        mechanics,
        contact,
        gravity=model.gravity,
        initial_velocity=(0.0, 0.0, model.initial_velocity_z),
    )
    _static_residual0, static_tangent0 = static_residual_and_tangent(
        mechanics,
        state0.x,
        contact,
        gravity=model.gravity,
    )
    dt_regularized = float(model.dt) / 10.0
    regularized = mechanics.mass_matrix + static_tangent0 * (beta * dt_regularized * dt_regularized * (1.0 + float(model.hht_alpha)))
    regularized_acceleration = np.asarray(
        np.linalg.solve(regularized.toarray(), static0),
        dtype=float,
    )
    pure_acceleration = state0.a.reshape(-1)
    initial_accel_rel = _relative_vector_error(regularized_acceleration, pure_acceleration)
    mass_from_matrix = _mass_from_consistent_matrix(mechanics)
    mass_reference = _total_mass(model)
    precision_abs = _max_dat_displacement_abs_precision(calculix_displacements)
    precision_accel = precision_abs / max(beta * float(model.dt) * float(model.dt), 1.0e-30)
    active_rows = [
        row
        for row in one_step_rows
        if _optional_float(row.get("sfc_on_calculix_max_penetration", "")) not in {None, 0.0}
        or _optional_float(row.get("sfc_predicted_max_penetration", "")) not in {None, 0.0}
    ]
    precontact_rows = [
        row
        for row in one_step_rows
        if _optional_float(row.get("sfc_on_calculix_max_penetration", "")) == 0.0
        and _optional_float(row.get("sfc_predicted_max_penetration", "")) == 0.0
    ]
    max_active_residual_rel = _max_optional_row_float(active_rows, "sfc_hht_effective_residual_relative_at_calculix_state")
    max_precontact_residual_rel = _max_optional_row_float(precontact_rows, "sfc_hht_effective_residual_relative_at_calculix_state")
    max_accel_diff = _max_optional_row_float(one_step_rows, "acceleration_diff_norm")
    precision_fraction = (
        ""
        if max_accel_diff in {"", None} or float(max_accel_diff) <= 0.0
        else float(precision_accel) / max(float(max_accel_diff), 1.0e-30)
    )
    if not calculix_displacements:
        diagnosis = "external_unavailable"
    elif max_precontact_residual_rel not in {"", None} and float(max_precontact_residual_rel) > 1.0e-6:
        diagnosis = "precontact_hht_state_mismatch"
    elif precision_fraction != "" and float(precision_fraction) > 1.0e-2:
        diagnosis = "dat_displacement_precision_can_affect_reconstructed_acceleration"
    elif max_active_residual_rel not in {"", None} and float(max_active_residual_rel) > 1.0e-2:
        diagnosis = "contact_phase_effective_residual_mismatch_not_explained_by_dat_precision"
    else:
        diagnosis = "hht_state_definitions_consistent_for_sampled_rows"
    return {
        "case": model.case,
        "resolution": model.resolution,
        "contact_mode": contact_mode,
        "calculix_completed": str(bool(calculix_displacements)).lower(),
        "source_beta_formula": "nonlingeo.c uses beta=(1-alpha)^2/4",
        "source_gamma_formula": "nonlingeo.c uses gamma=0.5-alpha",
        "source_prediction_summary": "prediction.c predicts displacement and velocity from veold/accold, then resets accold before Newton correction",
        "source_update_summary": "resultsini/iniparll treats the Newton solution vector as acceleration increment: dx=beta*dt^2*da, dv=gamma*dt*da, accold+=da",
        "source_residual_summary": "calcresidual.c forms RHS=(1+alpha)*(fext-fint)-alpha*(fextini-fini)-M*a for implicit dynamics",
        "sfc_alpha": float(model.hht_alpha),
        "sfc_beta": float(beta),
        "sfc_gamma": float(gamma),
        "initial_static_residual_norm": _vector_norm(static0),
        "initial_acceleration_pure_mass_norm": _vector_norm(pure_acceleration),
        "initial_acceleration_calculix_regularized_norm": _vector_norm(regularized_acceleration),
        "initial_acceleration_regularized_rel_diff": initial_accel_rel,
        "initial_regularized_dt": dt_regularized,
        "mass_from_consistent_matrix": mass_from_matrix,
        "mass_reference": mass_reference,
        "mass_rel_error": abs(mass_from_matrix - mass_reference) / max(abs(mass_reference), 1.0e-30),
        "dat_displacement_max_abs_precision_estimate": precision_abs,
        "dat_precision_acceleration_uncertainty_estimate": precision_accel,
        "dat_precision_fraction_of_observed_acceleration_diff": precision_fraction,
        "max_precontact_hht_residual_relative_at_calculix_state": max_precontact_residual_rel,
        "max_active_hht_residual_relative_at_calculix_state": max_active_residual_rel,
        "max_observed_acceleration_diff_norm": max_accel_diff,
        "diagnosis": diagnosis,
    }


def _mass_from_consistent_matrix(mechanics: MechanicsModel) -> float:
    z_dofs = np.arange(2, mechanics.n_dofs, 3, dtype=np.int64)
    return float(np.sum(mechanics.mass_matrix[z_dofs][:, z_dofs]))


def _max_dat_displacement_abs_precision(displacements: dict[float, np.ndarray]) -> float:
    max_precision = 0.0
    for values in displacements.values():
        arr = np.abs(np.asarray(values, dtype=float).ravel())
        nonzero = arr[arr > 0.0]
        if nonzero.size == 0:
            continue
        exponents = np.floor(np.log10(nonzero))
        # CalculiX .dat prints nodal displacements in E format with six digits
        # after the decimal in the current output path.
        max_precision = max(max_precision, float(np.max(0.5 * np.power(10.0, exponents - 6.0))))
    return max_precision


def _max_optional_row_float(rows: list[Row], key: str) -> float | str:
    values = [_optional_float(row.get(key, "")) for row in rows]
    numeric = [float(value) for value in values if value is not None]
    return _max_or_blank(numeric)


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


def mechanics_increment_acceptance_diagnostics(
    model: DropModel,
    sfc_rows: list[Row],
    command_row: Row,
    out_dir: Path,
) -> Row:
    """Return CalculiX/SFC mechanics and increment-acceptance diagnostics."""

    stdout_rel = str(command_row.get("stdout_log", ""))
    stdout_path = out_dir / stdout_rel if stdout_rel else None
    input_rel = str(command_row.get("input_file", ""))
    input_path = out_dir / input_rel if input_rel else None
    stdout = _parse_calculix_stdout_diagnostics(stdout_path)
    step_options = _parse_calculix_dynamic_step_options(input_path)
    beta, gamma = hht_newmark_parameters(model.hht_alpha)
    sfc_iterations = [int(row.get("newton_iterations", 0)) for row in sfc_rows if float(row.get("accepted_dt", 0.0)) > 0.0]
    sfc_residuals = [float(row.get("newton_residual_norm", 0.0)) for row in sfc_rows if float(row.get("accepted_dt", 0.0)) > 0.0]
    sfc_residual_ratios = _row_float_values(sfc_rows, "newton_residual_ratio")
    sfc_correction_ratios = _row_float_values(sfc_rows, "newton_correction_ratio")
    sfc_contact_changes = _row_float_values(sfc_rows, "newton_max_contact_element_change")
    sfc_energy_stabilization_count = sum(
        1 for row in sfc_rows if str(row.get("newton_any_energy_stabilization", "")).lower() == "true"
    )
    sfc_steps_with_contact_change = sum(
        1 for value in sfc_contact_changes if float(value) > 0.0
    )
    sfc_steps_with_rejected_iteration = sum(
        1
        for row in sfc_rows
        if float(row.get("accepted_dt", 0.0)) > 0.0
        and str(row.get("newton_any_calculix_style_rejection", "")).lower() == "true"
    )
    sfc_final_rejected_count = sum(
        1
        for row in sfc_rows
        if float(row.get("accepted_dt", 0.0)) > 0.0
        and str(row.get("newton_calculix_style_accepted", "")).lower() == "false"
    )
    calculix_available = bool(stdout_path is not None and stdout_path.exists())
    direct_fixed = (
        step_options.get("calculix_dynamic_direct") == "true"
        and stdout.get("calculix_stdout_max_attempt") not in {"", None}
        and int(stdout["calculix_stdout_max_attempt"]) == 1
        and int(stdout.get("calculix_stdout_cutback_attempt_count") or 0) == 0
    )
    convergence_per_increment = (
        stdout.get("calculix_stdout_increment_count") not in {"", None}
        and stdout.get("calculix_stdout_convergence_count") not in {"", None}
        and int(stdout["calculix_stdout_increment_count"]) == int(stdout["calculix_stdout_convergence_count"])
    )
    if not calculix_available:
        diagnosis = "external_unavailable"
    elif direct_fixed and convergence_per_increment:
        diagnosis = "direct_fixed_increment_accepted_after_internal_newton"
    elif direct_fixed:
        diagnosis = "direct_fixed_increment_with_incomplete_stdout_convergence_evidence"
    else:
        diagnosis = "automatic_or_cutback_increment_acceptance_not_aligned"
    return {
        "case": model.case,
        "resolution": model.resolution,
        "calculix_completed": str(bool(calculix_available)).lower(),
        "mechanics_source_definition": "CalculiX calcresidual uses b=(1+alpha)(fext-fint)-alpha(fext_n-fint_n)-M*a with convergence checked by checkconvergence residual/correction/contact criteria",
        "sfc_source_definition": "SFC hht_step uses the same residual sign after negation, but accepts Newton by relative correction tolerance or max iteration count",
        "calculix_dynamic_direct": step_options.get("calculix_dynamic_direct", ""),
        "calculix_nlgeom": step_options.get("calculix_nlgeom", ""),
        "calculix_alpha": step_options.get("calculix_alpha", ""),
        "sfc_alpha": model.hht_alpha,
        "calculix_hht_beta": "" if step_options.get("calculix_alpha", "") == "" else hht_newmark_parameters(float(step_options["calculix_alpha"]))[0],
        "calculix_hht_gamma": "" if step_options.get("calculix_alpha", "") == "" else hht_newmark_parameters(float(step_options["calculix_alpha"]))[1],
        "sfc_hht_beta": beta,
        "sfc_hht_gamma": gamma,
        **stdout,
        "sfc_accepted_step_count": len(sfc_iterations),
        "sfc_max_newton_iterations": max(sfc_iterations) if sfc_iterations else "",
        "sfc_mean_newton_iterations": _mean_or_blank([float(value) for value in sfc_iterations]),
        "sfc_max_newton_residual_norm": max(sfc_residuals) if sfc_residuals else "",
        "sfc_max_newton_residual_ratio_ram_over_qam": _max_or_blank(sfc_residual_ratios),
        "sfc_max_newton_correction_ratio_cam_over_uam": _max_or_blank(sfc_correction_ratios),
        "sfc_max_contact_element_change": _max_or_blank(sfc_contact_changes),
        "sfc_steps_with_contact_element_change": sfc_steps_with_contact_change,
        "sfc_energy_stabilization_count": sfc_energy_stabilization_count,
        "sfc_steps_with_rejected_multicriteria_iteration": sfc_steps_with_rejected_iteration,
        "sfc_final_calculix_style_rejected_step_count": sfc_final_rejected_count,
        "sfc_rows_reaching_iteration_limit_12": sum(1 for value in sfc_iterations if value >= 12),
        "direct_fixed_increment_no_retry_observed": str(bool(direct_fixed)).lower() if calculix_available else "",
        "calculix_converged_each_increment": str(bool(convergence_per_increment)).lower() if calculix_available else "",
        "acceptance_criteria_source_equivalent": "false",
        "diagnosis": diagnosis,
    }


def _parse_calculix_stdout_diagnostics(path: Path | None) -> Row:
    if path is None or not path.exists():
        return {
            "calculix_stdout_increment_count": "",
            "calculix_stdout_max_attempt": "",
            "calculix_stdout_min_increment_size": "",
            "calculix_stdout_convergence_count": "",
            "calculix_stdout_total_newton_iterations": "",
            "calculix_stdout_max_iterations_per_increment": "",
            "calculix_stdout_mean_iterations_per_increment": "",
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
    iterations_per_increment = _calculix_iterations_per_increment(text)
    return {
        "calculix_stdout_increment_count": len(attempts),
        "calculix_stdout_max_attempt": max(attempts) if attempts else "",
        "calculix_stdout_min_increment_size": min(sizes) if sizes else "",
        "calculix_stdout_convergence_count": len(re.findall(r"(?m)^\s+convergence(?:;|\s*$)", text)),
        "calculix_stdout_total_newton_iterations": sum(iterations_per_increment),
        "calculix_stdout_max_iterations_per_increment": max(iterations_per_increment) if iterations_per_increment else "",
        "calculix_stdout_mean_iterations_per_increment": _mean_or_blank(
            [float(value) for value in iterations_per_increment]
        ),
        "calculix_stdout_cutback_attempt_count": sum(1 for value in attempts if value > 1),
        "calculix_stdout_no_convergence_count": len(re.findall(r"(?m)^\s*no convergence\s*$", text)),
        "calculix_stdout_kscale_restore_count": len(re.findall(r"restoring the elastic contact stifnesses", text)),
        "calculix_stdout_contact_energy_stabilization_count": len(
            re.findall(r"Adaption of the (?:energy residual|max-decay boundary)", text)
        ),
        "calculix_stdout_forced_increment_size_count": len(re.findall(r"new increment size is forced", text)),
        "calculix_stdout_max_contact_spring_elements": max(contact_counts) if contact_counts else "",
    }


def _calculix_iterations_per_increment(stdout_text: str) -> list[int]:
    blocks = re.split(r"(?m)^\s*increment\s+\d+\s+attempt\s+\d+\s*$", stdout_text)[1:]
    return [len(re.findall(r"(?m)^\s*iteration\s+\d+\s*$", block)) for block in blocks]


def _parse_calculix_dynamic_step_options(path: Path | None) -> Row:
    if path is None or not path.exists():
        return {"calculix_dynamic_direct": "", "calculix_nlgeom": "", "calculix_alpha": ""}
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    dynamic_line = next((line for line in text.splitlines() if line.strip().startswith("*dynamic")), "")
    step_line = next((line for line in text.splitlines() if line.strip().startswith("*step")), "")
    alpha_match = re.search(r"alpha\s*=\s*([0-9.eE+-]+)", dynamic_line, flags=re.IGNORECASE)
    return {
        "calculix_dynamic_direct": str("direct" in dynamic_line).lower(),
        "calculix_nlgeom": str("nlgeom" in step_line).lower(),
        "calculix_alpha": "" if alpha_match is None else float(alpha_match.group(1)),
    }


def hht_residual_tangent_diagnostics(
    model: DropModel,
    *,
    contact_mode: str = "calculix_c3d4_f2f",
) -> list[Row]:
    """Return HHT residual/tangent trajectory diagnostics for SFC contact.

    The probe mirrors the CalculiX `calcresidual.c` sign convention by checking
    the SFC residual form

    `R = M a - (1 + alpha) B_static(u_{n+1}) + alpha B_static(u_n)`

    where `B_static = f_ext - f_int` is the CalculiX `fextini - fini`
    history sign.  The returned SFC residual is the negative of CalculiX's
    right-hand side `b`.

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
    update_abs = float(np.linalg.norm(accepted_static_residual + static_residual))
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
        "calculix_calcresidual_sign_convention": "sfc_R_is_negative_of_CalculiX_rhs_b_with_fextini_minus_fini_history",
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
    force_state = static_force_state(
        mechanics,
        np.asarray(x_flat, dtype=float).reshape((-1, 3)),
        contact,
        gravity=gravity,
    )
    residual = (
        mechanics.mass_matrix @ a
        - (1.0 + float(alpha)) * force_state.calculix_rhs_balance
        + float(alpha) * np.asarray(previous_static_residual, dtype=float)
    )
    tangent = (mechanics.mass_matrix * c0 + force_state.tangent * (1.0 + float(alpha))).tocsr()
    return np.asarray(residual, dtype=float), tangent, force_state.residual, force_state.tangent


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


def _row_float_values(rows: list[Row], key: str) -> list[float]:
    return _series_values(rows, key)


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
                quadrature="calculix_7",
                release_tolerance_scale=1.0e-3,
                contact_element_weight=1,
            ),
            "clean-room persistent CalculiX-style C3D4 face-to-face mode; seven slave-face integration springs; stored projection/normal clearance; hard linear overclosure; rigid plane query",
        )
    if contact_mode == "calculix_c3d4_f2f":
        return (
            CalculixC3D4FaceToFacePlaneContactGeometry(
                model.surface_faces,
                plane_z=_contact_plane_z(model),
                stiffness=model.contact_stiffness,
                quadrature="calculix_7",
                contact_element_weight=1,
            ),
            "clean-room CalculiX-style C3D4 face-to-face mode; seven slave-face integration springs; hard linear overclosure; rigid plane query",
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
                quadrature="calculix_7",
                release_tolerance_scale=1.0e-3,
                contact_element_weight=1,
            ),
            "clean-room persistent CalculiX-style C3D4 face-to-face mode; seven slave-face integration springs; stored projection/normal clearance; hard linear overclosure; dynamic FEM-SDF plane query",
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
                quadrature="calculix_7",
                contact_element_weight=1,
            ),
            "clean-room CalculiX-style C3D4 face-to-face mode; seven slave-face integration springs; hard linear overclosure; dynamic FEM-SDF plane query",
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
    mechanics_rows: list[Row] = []
    one_step_rows: list[Row] = []
    hht_state_rows: list[Row] = []
    contact_element_audit_rows: list[Row] = []
    contact_element_audit_summary_rows: list[Row] = []
    command_rows: list[Row] = []
    vtk_rows: list[Row] = []
    for resolution in comparison_resolutions:
        model = _contact_model(resolution=resolution, duration=duration, dt=dt)
        sfc_native_contact_element_rows: list[Row] = []
        sfc_rows, sfc_x, sfc_state = run_sfc_geometric_contact_history(
            model,
            contact_mode=contact_mode,
            cutback_policy=cutback_policy,
            contact_element_audit_rows=sfc_native_contact_element_rows,
        )
        history_rows.extend(sfc_rows)
        if skip_calculix:
            calc_rows: list[Row] = []
            calc_u = np.zeros_like(model.nodes)
            calc_stress = np.zeros((model.tet_elements.shape[0], 6), dtype=float)
            calc_displacements: dict[float, np.ndarray] = {}
            calc_contact_elements: dict[float, dict[tuple[int, int], Row]] = {}
            command_rows.append({"case": model.case, "resolution": resolution, "completed": "false", "return_code": "skipped", "command": ""})
        else:
            calc_rows, calc_u, calc_stress, command, calc_displacements = run_calculix_contact_with_stress(model, out_dir)
            command_rows.append(command)
            history_rows.extend(calc_rows)
            dat_rel = str(command.get("dat_file", ""))
            dat_path = out_dir / dat_rel if dat_rel else None
            calc_contact_elements = (
                _parse_calculix_dat_contact_elements(dat_path)
                if dat_path is not None and dat_path.exists()
                else {}
            )
        comparison_rows.append(compare_contact_histories(model, calc_rows, sfc_rows, calc_stress, sfc_state))
        alignment_rows.append(contact_alignment_diagnostics(model, calc_rows, sfc_rows, calc_displacements, command_rows[-1], out_dir))
        hht_rows.extend(hht_residual_tangent_diagnostics(model, contact_mode=contact_mode))
        lifecycle_rows.append(contact_lifecycle_output_diagnostics(model, calc_rows, sfc_rows, calc_displacements))
        mechanics_rows.append(mechanics_increment_acceptance_diagnostics(model, sfc_rows, command_rows[-1], out_dir))
        one_step_resolution_rows = one_step_calculix_state_diagnostics(
            model,
            calc_rows,
            calc_displacements,
            contact_mode=contact_mode,
        )
        one_step_rows.extend(one_step_resolution_rows)
        hht_state_rows.append(
            hht_state_definition_diagnostics(
                model,
                calc_displacements,
                one_step_resolution_rows,
                contact_mode=contact_mode,
            )
        )
        per_face_rows, per_step_rows = contact_element_clearance_lifecycle_audit(
            model,
            calculix_dat_contact=calc_contact_elements,
            sfc_native_rows=sfc_native_contact_element_rows,
            calculix_displacements=calc_displacements,
        )
        contact_element_audit_rows.extend(per_face_rows)
        contact_element_audit_summary_rows.extend(per_step_rows)
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
    claim_rows = _claim_rows(
        comparison_rows,
        mesh_rows,
        timestep_rows,
        vtk_rows,
        alignment_rows,
        hht_rows,
        lifecycle_rows,
        mechanics_rows,
        one_step_rows,
        hht_state_rows,
        contact_element_audit_summary_rows,
    )
    outputs = {
        "history": out_dir / "geometric_contact_history.csv",
        "comparison": out_dir / "geometric_contact_calculix_comparison.csv",
        "alignment": out_dir / "geometric_contact_alignment_diagnostics.csv",
        "hht": out_dir / "geometric_contact_hht_residual_tangent.csv",
        "lifecycle": out_dir / "geometric_contact_lifecycle_output_diagnostics.csv",
        "mechanics": out_dir / "geometric_contact_mechanics_increment_acceptance.csv",
        "one_step": out_dir / "geometric_contact_one_step_state_diagnostics.csv",
        "hht_state": out_dir / "geometric_contact_hht_state_definition_diagnostics.csv",
        "contact_element_audit": out_dir / "geometric_contact_element_clearance_lifecycle_audit.csv",
        "contact_element_audit_summary": out_dir / "geometric_contact_element_clearance_lifecycle_audit_summary.csv",
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
    _write_csv(outputs["mechanics"], mechanics_rows)
    _write_csv(outputs["one_step"], one_step_rows)
    _write_csv(outputs["hht_state"], hht_state_rows)
    _write_csv(outputs["contact_element_audit"], contact_element_audit_rows)
    _write_csv(outputs["contact_element_audit_summary"], contact_element_audit_summary_rows)
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
        mechanics_rows,
        one_step_rows,
        hht_state_rows,
        contact_element_audit_summary_rows,
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
    mechanics_rows: list[Row],
    one_step_rows: list[Row],
    hht_state_rows: list[Row],
    contact_element_audit_summary_rows: list[Row],
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
    mechanics_supported = bool(mechanics_rows) and any(
        row.get("diagnosis") not in {"", None, "external_unavailable"} for row in mechanics_rows
    )
    one_step_supported = bool(one_step_rows) and any(
        row.get("diagnosis") not in {"", None, "external_unavailable", "insufficient_calculix_time_states"}
        for row in one_step_rows
    )
    hht_state_supported = bool(hht_state_rows) and any(
        row.get("diagnosis") not in {"", None, "external_unavailable"} for row in hht_state_rows
    )
    contact_element_audit_supported = bool(contact_element_audit_summary_rows) and any(
        row.get("calculix_per_contact_output_available") == "true" for row in contact_element_audit_summary_rows
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
            "claim": "calculix_mechanics_increment_acceptance_diagnostics_available",
            "supported": str(mechanics_supported).lower(),
            "evidence_csv": "geometric_contact_mechanics_increment_acceptance.csv",
            "details": "Tracks CalculiX DIRECT increment attempts, Newton iterations, convergence messages, HHT alpha/beta/gamma, and SFC Newton acceptance diagnostics",
        },
        {
            "claim": "calculix_state_one_step_mechanics_diagnostics_available",
            "supported": str(one_step_supported).lower(),
            "evidence_csv": "geometric_contact_one_step_state_diagnostics.csv",
            "details": "Starts SFC from each CalculiX displacement state and decomposes internal force, contact force, mass term, HHT residual, tangent norms, and Newmark update error",
        },
        {
            "claim": "hht_newmark_state_definition_precision_diagnostics_available",
            "supported": str(hht_state_supported).lower(),
            "evidence_csv": "geometric_contact_hht_state_definition_diagnostics.csv",
            "details": "Checks CalculiX beta/gamma, prediction/update/residual state definitions, initial acceleration regularization, mass scale, and .dat displacement precision amplification",
        },
        {
            "claim": "calculix_per_contact_element_clearance_lifecycle_audit_available",
            "supported": str(contact_element_audit_supported).lower(),
            "evidence_csv": "geometric_contact_element_clearance_lifecycle_audit_summary.csv",
            "details": "Parses per-contact CDIS/CSTR/CELS from CalculiX .dat and compares CalculiX active contact elements against SFC native and CalculiX-displacement replay springs",
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
    mechanics_rows: list[Row],
    one_step_rows: list[Row],
    hht_state_rows: list[Row],
    contact_element_audit_summary_rows: list[Row],
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
        "The strict `calculix_c3d4_f2f` modes use seven slave-face integration springs per C3D4 boundary triangle, stored master projection/normal clearance, and a hard linear pressure-overclosure law for CalculiX-style validation. The older `plane` and `dynamic_sdf_plane` modes use three triangle quadrature samples and are not strict CalculiX C3D4 contact discretizations.",
        "",
        "## Outputs",
        "",
        "- `geometric_contact_history.csv`",
        "- `geometric_contact_calculix_comparison.csv`",
        "- `geometric_contact_alignment_diagnostics.csv`",
        "- `geometric_contact_hht_residual_tangent.csv`",
        "- `geometric_contact_lifecycle_output_diagnostics.csv`",
        "- `geometric_contact_mechanics_increment_acceptance.csv`",
        "- `geometric_contact_one_step_state_diagnostics.csv`",
        "- `geometric_contact_hht_state_definition_diagnostics.csv`",
        "- `geometric_contact_element_clearance_lifecycle_audit.csv`",
        "- `geometric_contact_element_clearance_lifecycle_audit_summary.csv`",
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
            "## Mechanics And Increment Acceptance Diagnostics",
            "",
            "| Resolution | Diagnosis | Direct | CalculiX inc. | CalculiX max iter/inc. | SFC max iter | max ram/qam | max cam/uam | max contact change | steps w/change | energy stab. | final rejected |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in mechanics_rows:
        lines.append(
            f"| {row['resolution']} | {row['diagnosis']} | {row['calculix_dynamic_direct']} | "
            f"{_fmt(row['calculix_stdout_increment_count'])} | "
            f"{_fmt(row['calculix_stdout_max_iterations_per_increment'])} | "
            f"{_fmt(row['sfc_max_newton_iterations'])} | "
            f"{_fmt(row.get('sfc_max_newton_residual_ratio_ram_over_qam', ''))} | "
            f"{_fmt(row.get('sfc_max_newton_correction_ratio_cam_over_uam', ''))} | "
            f"{_fmt(row.get('sfc_max_contact_element_change', ''))} | "
            f"{_fmt(row.get('sfc_steps_with_contact_element_change', ''))} | "
            f"{_fmt(row.get('sfc_energy_stabilization_count', ''))} | "
            f"{_fmt(row.get('sfc_final_calculix_style_rejected_step_count', ''))} |"
        )
    lines.extend(
        [
            "",
            "## One-Step CalculiX-State Diagnostics",
            "",
            "This table starts SFC from CalculiX displacement states. CalculiX does not print full global internal force vectors or tangent matrices in the parsed `.dat`, so the internal-force and tangent entries are SFC decompositions evaluated on CalculiX kinematics rather than direct CalculiX vectors.",
            "",
            "| Resolution | time n+1 | Diagnosis | Dominant term | disp. rel. | accel. rel. | force-balance accel rel. | residual rel. | CalculiX RF | SFC-on-CalculiX force | SFC predicted force |",
            "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _representative_one_step_rows(one_step_rows):
        lines.append(
            f"| {row['resolution']} | {_fmt(row.get('time_np1', ''))} | {row.get('diagnosis', '')} | "
            f"{row.get('dominant_one_step_difference_source', '')} | "
            f"{_fmt(row.get('sfc_predicted_vs_calculix_displacement_rel', ''))} | "
            f"{_fmt(row.get('sfc_predicted_vs_calculix_acceleration_rel', ''))} | "
            f"{_fmt(row.get('sfc_hht_force_balance_vs_reconstructed_acceleration_rel', ''))} | "
            f"{_fmt(row.get('sfc_hht_effective_residual_relative_at_calculix_state', ''))} | "
            f"{_fmt(row.get('calculix_rf_z', ''))} | "
            f"{_fmt(row.get('sfc_on_calculix_contact_force_z', ''))} | "
            f"{_fmt(row.get('sfc_predicted_contact_force_z', ''))} |"
        )
    lines.extend(
        [
            "",
            "## HHT/Newmark State Definition Diagnostics",
            "",
            "| Resolution | Diagnosis | beta | gamma | initial accel rel. | mass rel. | .dat accel uncertainty | precision/diff | precontact residual rel. | active residual rel. |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in hht_state_rows:
        lines.append(
            f"| {row['resolution']} | {row['diagnosis']} | {_fmt(row.get('sfc_beta', ''))} | "
            f"{_fmt(row.get('sfc_gamma', ''))} | "
            f"{_fmt(row.get('initial_acceleration_regularized_rel_diff', ''))} | "
            f"{_fmt(row.get('mass_rel_error', ''))} | "
            f"{_fmt(row.get('dat_precision_acceleration_uncertainty_estimate', ''))} | "
            f"{_fmt(row.get('dat_precision_fraction_of_observed_acceleration_diff', ''))} | "
            f"{_fmt(row.get('max_precontact_hht_residual_relative_at_calculix_state', ''))} | "
            f"{_fmt(row.get('max_active_hht_residual_relative_at_calculix_state', ''))} |"
        )
    lines.extend(
        [
            "",
            "## Per-Contact-Element Clearance/Lifecycle Audit",
            "",
            "| Resolution | time | CalculiX active | SFC native active | Replay active | per-contact output | clearance replay Linf | force replay Linf | energy replay Linf | Diagnosis |",
            "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in contact_element_audit_summary_rows:
        lines.append(
            f"| {row['resolution']} | {_fmt(row['time'])} | "
            f"{_fmt(row['calculix_active_contact_elements'])} | "
            f"{_fmt(row['sfc_native_active_springs'])} | "
            f"{_fmt(row['calculix_displacement_replay_active_springs'])} | "
            f"{row['calculix_per_contact_output_available']} | "
            f"{_fmt(row['clearance_replay_linf_error'])} | "
            f"{_fmt(row['force_replay_linf_error'])} | "
            f"{_fmt(row['energy_replay_linf_error'])} | "
            f"{row['diagnosis']} |"
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


def _representative_one_step_rows(rows: list[Row]) -> list[Row]:
    if not rows:
        return []
    contact_rows = [
        row
        for row in rows
        if _optional_float(row.get("sfc_on_calculix_max_penetration", "")) not in {None, 0.0}
        or _optional_float(row.get("sfc_predicted_max_penetration", "")) not in {None, 0.0}
    ]
    if not contact_rows:
        return rows[: min(3, len(rows))]
    worst = max(
        contact_rows,
        key=lambda row: float(row.get("sfc_predicted_vs_calculix_displacement_rel") or 0.0),
    )
    first = contact_rows[0]
    if first is worst:
        return [first]
    return [first, worst]


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
