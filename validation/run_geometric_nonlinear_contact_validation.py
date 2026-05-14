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
    CalculixC3D4FaceToFacePlaneContactGeometry,
    CalculixC3D4FaceToFaceSDFContactGeometry,
)
from sfc.contact import DynamicSurfaceSDFContactGeometry, UniformTriangleAABBHash  # noqa: E402
from sfc.fem.calculix_aligned import (  # noqa: E402
    ContactGeometry,
    MechanicsModel,
    MechanicsState,
    PlaneContactGeometry,
    StepDiagnostics,
    evaluate_state,
    hht_step,
    initial_state,
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


def run_calculix_contact_with_stress(model: DropModel, out_dir: Path, *, timeout: int = 300) -> tuple[list[Row], np.ndarray, np.ndarray, Row]:
    """Run CalculiX contact and return history, final displacement, stress, command."""

    if not calculix_available():
        return [], np.zeros_like(model.nodes), np.zeros((model.tet_elements.shape[0], 6), dtype=float), {
            "case": model.case,
            "resolution": model.resolution,
            "completed": "false",
            "return_code": "not_available",
            "command": "",
        }
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
        return [], np.zeros_like(model.nodes), np.zeros((model.tet_elements.shape[0], 6), dtype=float), command_row
    displacements = _parse_calculix_dat_displacements_strict(dat, model.node_ids)
    contact_totals = _parse_calculix_dat_contact_totals(dat)
    history = _history_from_displacements("calculix", model, displacements, contact_totals)
    final_u = displacements[max(displacements)]
    final_stress = _parse_final_stress_voigt(dat, model.tet_elements.shape[0])
    return history, final_u, final_stress, command_row


def run_sfc_geometric_contact_history(
    model: DropModel,
    *,
    contact_mode: str = "calculix_c3d4_f2f",
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
    times = np.arange(0.0, model.total_time + 0.5 * model.dt, model.dt)
    rows: list[Row] = []
    for step, time in enumerate(times):
        state.time = float(time)
        centroid_gaps = _surface_face_gaps_to_contact_plane(model, state.x)
        rows.append(
            {
                "case": model.case,
                "resolution": model.resolution,
                "source": source,
                "contact_mode": contact_mode,
                "time": float(time),
                "z_cm": _mass_weighted_center_z(model, state.x),
                "v_cm_z": _mass_weighted_velocity_z(model, state.v.reshape(-1)),
                "min_gap": float(np.min(centroid_gaps)),
                "quadrature_min_gap": diagnostics.contact.min_gap,
                "max_penetration": max(float(diagnostics.contact.max_penetration), float(np.max(np.maximum(-centroid_gaps, 0.0)))),
                "active_contact_count": diagnostics.contact.active_count,
                "normal_force_proxy": diagnostics.contact.normal_force,
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
                "details": details,
            }
        )
        if step == len(times) - 1:
            break
        state, previous_static_residual, diagnostics = hht_step(
            mechanics,
            state,
            previous_static_residual,
            contact,
            dt=model.dt,
            gravity=model.gravity,
            alpha=model.hht_alpha,
            max_iterations=12,
            tolerance=1.0e-10,
        )
    return rows, state.x, _vtk_state_from_backend(state, diagnostics)


def _make_contact_geometry(model: DropModel, contact_mode: str) -> tuple[ContactGeometry, str]:
    if contact_mode == "calculix_c3d4_f2f":
        return (
            CalculixC3D4FaceToFacePlaneContactGeometry(
                model.surface_faces,
                plane_z=_contact_plane_z(model),
                stiffness=model.contact_stiffness,
            ),
            "clean-room CalculiX-style C3D4 face-to-face mode; one slave-face centroid spring; hard linear overclosure; rigid plane query",
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
    raise ValueError("contact_mode must be 'calculix_c3d4_f2f', 'dynamic_sdf_calculix_f2f', 'plane', or 'dynamic_sdf_plane'")


def _normal_force_source(contact_mode: str) -> str:
    if contact_mode in {"calculix_c3d4_f2f", "dynamic_sdf_calculix_f2f"}:
        return "calculix_c3d4_f2f_hard_linear"
    if contact_mode == "dynamic_sdf_plane":
        return "dynamic_sdf_three_point_penalty_tangent"
    return "three_point_penalty_tangent"


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


def sfc_mesh_convergence(resolutions: list[int], *, duration: float, dt: float, contact_mode: str = "calculix_c3d4_f2f") -> list[Row]:
    rows: list[Row] = []
    for resolution in resolutions:
        model = _contact_model(resolution=resolution, duration=duration, dt=dt)
        history, _, state = run_sfc_geometric_contact_history(model, contact_mode=contact_mode)
        rows.append(
            {
                "case": "block_drop_contact_mesh",
                "resolution": resolution,
                "contact_mode": contact_mode,
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


def sfc_timestep_convergence(dts: list[float], *, duration: float, contact_mode: str = "calculix_c3d4_f2f") -> list[Row]:
    rows: list[Row] = []
    for dt in dts:
        model = _contact_model(resolution=1, duration=duration, dt=dt)
        history, _, state = run_sfc_geometric_contact_history(model, contact_mode=contact_mode)
        rows.append(
            {
                "case": "block_drop_contact_timestep",
                "dt": dt,
                "contact_mode": contact_mode,
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
    contact_mode: str = "calculix_c3d4_f2f",
) -> dict[str, Path]:
    """Run geometric nonlinear contact validation and write CSV/Markdown."""

    out_dir.mkdir(parents=True, exist_ok=True)
    duration = 0.12 if quick else 0.14
    dt = 0.002
    comparison_resolutions = [1] if quick else [1, 2, 3]
    history_rows: list[Row] = []
    comparison_rows: list[Row] = []
    command_rows: list[Row] = []
    vtk_rows: list[Row] = []
    for resolution in comparison_resolutions:
        model = _contact_model(resolution=resolution, duration=duration, dt=dt)
        sfc_rows, sfc_x, sfc_state = run_sfc_geometric_contact_history(model, contact_mode=contact_mode)
        history_rows.extend(sfc_rows)
        if skip_calculix:
            calc_rows: list[Row] = []
            calc_u = np.zeros_like(model.nodes)
            calc_stress = np.zeros((model.tet_elements.shape[0], 6), dtype=float)
            command_rows.append({"case": model.case, "resolution": resolution, "completed": "false", "return_code": "skipped", "command": ""})
        else:
            calc_rows, calc_u, calc_stress, command = run_calculix_contact_with_stress(model, out_dir)
            command_rows.append(command)
            history_rows.extend(calc_rows)
        comparison_rows.append(compare_contact_histories(model, calc_rows, sfc_rows, calc_stress, sfc_state))
        if calc_rows:
            vtk_rows.extend(write_stress_cloud_vtks(out_dir, model, calc_u, calc_stress, sfc_x, sfc_state))

    mesh_rows = sfc_mesh_convergence([1, 2, 3], duration=duration, dt=dt, contact_mode=contact_mode)
    timestep_rows = sfc_timestep_convergence(
        [0.004, 0.002, 0.001] if quick else [0.002, 0.001, 0.0005],
        duration=duration,
        contact_mode=contact_mode,
    )
    claim_rows = _claim_rows(comparison_rows, mesh_rows, timestep_rows, vtk_rows)
    outputs = {
        "history": out_dir / "geometric_contact_history.csv",
        "comparison": out_dir / "geometric_contact_calculix_comparison.csv",
        "mesh": out_dir / "geometric_contact_mesh_convergence.csv",
        "timestep": out_dir / "geometric_contact_timestep_convergence.csv",
        "vtk": out_dir / "geometric_contact_stress_clouds.csv",
        "commands": out_dir / "external_solver_commands.csv",
        "claims": out_dir / "geometric_contact_claims.csv",
        "summary": out_dir / "geometric_contact_validation_summary.md",
    }
    _write_csv(outputs["history"], history_rows)
    _write_csv(outputs["comparison"], comparison_rows)
    _write_csv(outputs["mesh"], mesh_rows)
    _write_csv(outputs["timestep"], timestep_rows)
    _write_csv(outputs["vtk"], vtk_rows)
    _write_csv(outputs["commands"], command_rows)
    _write_csv(outputs["claims"], claim_rows)
    _write_markdown(outputs["summary"], comparison_rows, mesh_rows, timestep_rows, vtk_rows, claim_rows, contact_mode=contact_mode)
    return outputs


def _claim_rows(comparison_rows: list[Row], mesh_rows: list[Row], timestep_rows: list[Row], vtk_rows: list[Row]) -> list[Row]:
    comparison_available = any(row["calculix_completed"] == "true" for row in comparison_rows)
    accepted = any(row["acceptance_status"] == "passed_scoped_gate" for row in comparison_rows)
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
    *,
    contact_mode: str,
) -> None:
    lines = [
        "# Geometric Nonlinear Contact Validation",
        "",
        "This validation compares the clean-room CalculiX-aligned SFC StVK geometric nonlinear block-plane contact path against CalculiX dynamic contact output.",
        "",
        f"SFC contact mode: `{contact_mode}`.",
        "",
        "The default `calculix_c3d4_f2f` mode uses one slave-face centroid spring per C3D4 boundary triangle and a hard linear pressure-overclosure law for stricter CalculiX-style validation. The older `plane` and `dynamic_sdf_plane` modes use three triangle quadrature samples and are not strict CalculiX C3D4 contact discretizations.",
        "",
        "## Outputs",
        "",
        "- `geometric_contact_history.csv`",
        "- `geometric_contact_calculix_comparison.csv`",
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
    lines.extend(["", "## Contact Comparison", "", "| Resolution | Status | z_cm L2 rel. | max pen. rel. | peak force rel. | contact-zone VM rel. |", "| ---: | --- | ---: | ---: | ---: | ---: |"])
    for row in comparison_rows:
        lines.append(
            f"| {row['resolution']} | {row['acceptance_status']} | {_fmt(row['z_cm_l2_rel_error'])} | "
            f"{_fmt(row['max_penetration_rel_error'])} | {_fmt(row['peak_normal_force_rel_error'])} | {_fmt(row['contact_zone_max_vm_rel_error'])} |"
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
        choices=["calculix_c3d4_f2f", "dynamic_sdf_calculix_f2f", "plane", "dynamic_sdf_plane"],
        default="calculix_c3d4_f2f",
        help="SFC contact geometry/enforcement mode used for the validation run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(args.out_dir, quick=args.quick, skip_calculix=args.skip_calculix, contact_mode=args.contact_mode)
    print("Geometric nonlinear contact validation complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
