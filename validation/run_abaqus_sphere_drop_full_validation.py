"""Full Abaqus sphere-drop validation for SFC Lagrangian SDF contact.

This validation-only runner extends the short pre-contact check to the full
Abaqus/Explicit sphere-drop trajectory.  It reconstructs the C3D4 sphere from
the Abaqus input deck, solves the same gravity/drop problem with internally
assembled SFC FEM matrices and a ``MaterialSDF + LagrangianSDFContactOracle``
rigid-plane contact response, then compares the full time history against the
exported Abaqus VTK frames.

Abaqus remains an external reference artifact only; the core SFC package does
not import Abaqus or read ODB files.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact import (  # noqa: E402
    LagrangianSDFContactOracle,
    SurfaceSample,
    lagrangian_oracle_penalty_response,
    triangle_surface_quadrature_cache,
)
from sfc.fem import DeformableBody, assemble_gravity_force, assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from validation.run_abaqus_sphere_drop_short_validation import (  # noqa: E402
    DEFAULT_INP,
    DEFAULT_VTK_DIR,
    AbaqusSphereDropModel,
    _abaqus_history_from_vtk,
    _boundary_faces,
    _comparison_rows,
    _configure_plot_style,
    _mass_weighted_center_z,
    _metric_rows,
    _node_samples,
    _plane_oracle,
    _plot_abs_error_curve,
    _plot_curve,
    _resolved_contact_stiffness,
    _stress_metric_summary,
    _surface_node_area_weights,
    _write_csv,
    parse_sphere_drop_inp,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Row = dict[str, Any]


def _restricted_samples(samples: list[SurfaceSample], ids: np.ndarray) -> list[SurfaceSample]:
    return [samples[int(idx)] for idx in np.asarray(ids, dtype=np.int64)]


def _surface_sample_points(samples: list[SurfaceSample], x_current: np.ndarray) -> np.ndarray:
    if not samples:
        return np.empty((0, 3), dtype=float)
    X = np.asarray(x_current, dtype=float)
    return np.vstack([sample.point(X) for sample in samples])


def _node_surface_samples(surface_nodes: np.ndarray) -> list[SurfaceSample]:
    return _node_samples(surface_nodes)


def _contact_samples(
    model: AbaqusSphereDropModel,
    surface_faces: np.ndarray,
    surface_nodes: np.ndarray,
    area_weights: np.ndarray,
    *,
    contact_integration: str,
    quadrature_order: int,
    hybrid_node_area_fraction: float,
) -> tuple[list[SurfaceSample], np.ndarray]:
    if contact_integration == "node":
        return _node_surface_samples(surface_nodes), area_weights
    if contact_integration == "surface":
        cache = triangle_surface_quadrature_cache(surface_faces, model.nodes, order=quadrature_order)
        return cache.samples(), cache.area_weights.copy()
    if contact_integration == "hybrid":
        fraction = float(hybrid_node_area_fraction)
        if not (0.0 <= fraction <= 1.0):
            raise ValueError("hybrid_node_area_fraction must lie in [0, 1]")
        cache = triangle_surface_quadrature_cache(surface_faces, model.nodes, order=quadrature_order)
        surface_samples = cache.samples()
        node_samples = _node_surface_samples(surface_nodes)
        samples = surface_samples + node_samples
        weights = np.concatenate(
            (
                (1.0 - fraction) * cache.area_weights,
                fraction * np.asarray(area_weights, dtype=float),
            )
        )
        return samples, weights
    raise ValueError("contact_integration must be 'node', 'surface', or 'hybrid'")


def _lagrangian_plane_contact_response(
    model: AbaqusSphereDropModel,
    x_current: np.ndarray,
    *,
    v_current: np.ndarray | None = None,
    area_weights: np.ndarray,
    samples: list[SurfaceSample],
    oracle: LagrangianSDFContactOracle,
    contact_stiffness: float,
    contact_damping: float = 0.0,
    velocity_tangent_factor: float = 0.0,
    n_dofs: int,
) -> tuple[np.ndarray, csr_matrix, float, int, float, float, float]:
    """Return SFC Lagrangian-SDF contact force and tangent against the plane."""

    sample_points = _surface_sample_points(samples, x_current)
    direct_gaps = sample_points[:, 2] - model.plane_z if sample_points.size else np.empty(0, dtype=float)
    min_gap = float(np.min(direct_gaps)) if direct_gaps.size else 0.0
    active_ids = np.nonzero(direct_gaps <= 0.0)[0]
    if active_ids.size == 0:
        return np.zeros(n_dofs, dtype=float), csr_matrix((n_dofs, n_dofs), dtype=float), min_gap, 0, 0.0, 0.0, 0.0

    active_samples = _restricted_samples(samples, active_ids)
    active_weights = area_weights[active_ids]
    response = lagrangian_oracle_penalty_response(
        x_current,
        active_samples,
        oracle,
        pressure_stiffness=float(contact_stiffness),
        sample_area_weights=active_weights,
        n_total_dofs=n_dofs + model.plane_nodes.shape[0] * 3,
        slave_dof_offset=0,
        master_dof_offset=n_dofs,
    )
    gaps = np.asarray([constraint.g for constraint in response.constraints], dtype=float)
    penetration = np.maximum(-gaps, 0.0)
    active_response = penetration > 0.0
    Kc = response.stiffness[:n_dofs, :n_dofs].tocsr()
    if contact_damping > 0.0 and v_current is not None and response.constraints:
        damping_force = np.zeros(n_dofs, dtype=float)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        velocity_nodes = np.asarray(v_current, dtype=float).reshape((-1, 3))
        for constraint, area in zip(response.constraints, active_weights, strict=True):
            if not constraint.active:
                continue
            normal = np.asarray(constraint.normal, dtype=float)
            sample_velocity = np.zeros(3, dtype=float)
            for node, weight in zip(constraint.slave_node_ids, constraint.slave_weights, strict=True):
                sample_velocity += float(weight) * velocity_nodes[int(node)]
            normal_velocity = float(sample_velocity @ normal)
            if normal_velocity >= 0.0:
                continue
            damping_scale = float(contact_damping) * float(area)
            lambda_damping = damping_scale * (-normal_velocity)
            tangent_scale = damping_scale * float(velocity_tangent_factor)
            dofs: list[int] = []
            vals: list[float] = []
            for node, weight in zip(constraint.slave_node_ids, constraint.slave_weights, strict=True):
                base = 3 * int(node)
                for axis in range(3):
                    dofs.append(base + axis)
                    vals.append(float(weight) * float(normal[axis]))
                    damping_force[base + axis] += float(weight) * lambda_damping * float(normal[axis])
            if tangent_scale > 0.0:
                for row_dof, row_value in zip(dofs, vals, strict=True):
                    for col_dof, col_value in zip(dofs, vals, strict=True):
                        rows.append(row_dof)
                        cols.append(col_dof)
                        data.append(tangent_scale * row_value * col_value)
        if rows:
            Kc = (Kc + coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()).tocsr()
        force = response.force[:n_dofs] + damping_force
    else:
        force = response.force[:n_dofs]
    return (
        force,
        Kc,
        float(np.min(gaps)) if gaps.size else min_gap,
        int(response.active_count),
        float(np.max(penetration)) if penetration.size else 0.0,
        float(0.5 * contact_stiffness * np.sum(active_weights * penetration**2)),
        float(np.sum(active_weights[active_response])),
    )


def _first_contact_time(rows: list[Row]) -> float | None:
    active = [float(row["time"]) for row in rows if int(row["active_contact_count"]) > 0 or float(row["max_penetration"]) > 0.0]
    return None if not active else min(active)


def run_sfc_lagrangian_sdf_full_history(
    model: AbaqusSphereDropModel,
    *,
    duration: float,
    dt: float,
    contact_stiffness: float,
    contact_damping: float = 0.0,
    mass_damping: float = 0.0,
    stiffness_damping: float = 0.0,
    damping_start_time: float = 0.0,
    integrator: str = "newmark",
    hht_alpha: float = 0.0,
    contact_integration: str = "node",
    quadrature_order: int = 3,
    hybrid_node_area_fraction: float = 0.15,
    adaptive_increments: bool = False,
    min_increment: float = 1.0e-8,
    max_increment: float | None = None,
    cutback_factor: float = 0.5,
    growth_factor: float = 1.25,
    target_newton_iterations: int = 4,
    residual_tolerance: float = 1.0e-8,
    contact_event_substeps: int = 8,
    output_stride: int = 1,
    max_newton_iterations: int = 8,
    tolerance: float = 1.0e-10,
) -> list[Row]:
    """Run the full SFC sphere-drop solve with Lagrangian-SDF contact."""

    mesh = VolumeMesh(model.nodes, model.elements, element_type="tet4")
    body = DeformableBody(mesh, {"E": model.young, "nu": model.poisson}, density=model.density)
    K = assemble_stiffness_matrix(body).tocsr()
    M = assemble_mass_matrix(body, kind="consistent").tocsr()
    C_base = float(mass_damping) * M + float(stiffness_damping) * K
    f_gravity = assemble_gravity_force(body, (0.0, 0.0, -model.gravity))
    surface_faces = _boundary_faces(model.elements)
    surface_nodes, area_weights = _surface_node_area_weights(model.nodes, surface_faces)
    samples, sample_area_weights = _contact_samples(
        model,
        surface_faces,
        surface_nodes,
        area_weights,
        contact_integration=contact_integration,
        quadrature_order=quadrature_order,
        hybrid_node_area_fraction=float(hybrid_node_area_fraction),
    )
    oracle = _plane_oracle(model)
    n_dofs = body.n_dofs
    u = np.zeros(n_dofs, dtype=float)
    v = np.zeros(n_dofs, dtype=float)
    integrator_name = str(integrator).lower()
    if integrator_name not in {"newmark", "hht"}:
        raise ValueError("integrator must be 'newmark' or 'hht'")
    alpha = float(hht_alpha) if integrator_name == "hht" else 0.0
    if not (-1.0 / 3.0 <= alpha <= 0.0):
        raise ValueError("hht_alpha must lie in [-1/3, 0]")
    output_dt = float(dt)
    min_h = float(min_increment)
    max_h = output_dt if max_increment is None else min(float(max_increment), output_dt)
    if min_h <= 0.0 or max_h <= 0.0 or min_h > max_h:
        raise ValueError("time increment bounds must satisfy 0 < min_increment <= max_increment")
    cutback = float(cutback_factor)
    growth = float(growth_factor)
    if not (0.0 < cutback < 1.0):
        raise ValueError("cutback_factor must lie in (0, 1)")
    if growth < 1.0:
        raise ValueError("growth_factor must be at least 1")
    event_floor = max(min_h, output_dt / max(1, int(contact_event_substeps)))

    x0 = model.nodes.copy()
    f_contact, K_contact, min_gap, active_count, max_pen, contact_energy, active_area = _lagrangian_plane_contact_response(
        model,
        x0,
        v_current=v.reshape((-1, 3)),
        area_weights=sample_area_weights,
        samples=samples,
        oracle=oracle,
        contact_stiffness=contact_stiffness,
        contact_damping=contact_damping,
        velocity_tangent_factor=0.0,
        n_dofs=n_dofs,
    )
    C_initial = C_base if 0.0 >= float(damping_start_time) else csr_matrix(K.shape, dtype=float)
    a = np.asarray(spsolve(M.tocsc(), f_gravity + f_contact - C_initial @ v - K @ u), dtype=float)
    f_contact_old = f_contact.copy()

    beta = 0.25 * (1.0 - alpha) ** 2
    gamma = 0.5 - alpha
    rows: list[Row] = []
    output_times = np.arange(0.0, duration + 0.5 * output_dt, output_dt)
    output_stride_value = max(1, int(output_stride))
    solve_start = perf_counter()
    total_newton_iterations = 0
    failed_newton_steps = 0
    cutback_count = 0
    accepted_increment_count = 0
    accepted_increments: list[float] = []
    current_time = 0.0
    output_index = 0
    next_h = max_h

    def append_output(step: int, time_value: float) -> None:
        current = model.nodes + u.reshape((-1, 3))
        f_out, _Kc, min_gap_out, active_count_out, max_pen_out, contact_energy_out, active_area_out = _lagrangian_plane_contact_response(
            model,
            current,
            v_current=v.reshape((-1, 3)),
            area_weights=sample_area_weights,
            samples=samples,
            oracle=oracle,
            contact_stiffness=contact_stiffness,
            contact_damping=contact_damping,
            velocity_tangent_factor=0.0,
            n_dofs=n_dofs,
        )
        stress_summary = _stress_metric_summary(model, u)
        rows.append(
            {
                "source": "sfc_lagrangian_sdf_full",
                "step": int(step),
                "time": float(time_value),
                "z_cm": _mass_weighted_center_z(model.nodes, model.elements, current),
                "min_gap": float(min_gap_out),
                "active_contact_count": int(active_count_out),
                "active_contact_area": float(active_area_out),
                "max_penetration": float(max_pen_out),
                "normal_force_z": float(np.sum(f_out[2::3])),
                "contact_energy": float(contact_energy_out),
                **stress_summary,
                "contact_path": "MaterialSDF+LagrangianSDFContactOracle",
                "contact_integration": str(contact_integration),
                "time_integrator": integrator_name,
            }
        )

    append_output(0, 0.0)
    output_index = 1
    while current_time < float(duration) - 1.0e-14:
        next_output_time = float(output_times[min(output_index, len(output_times) - 1)])
        if next_output_time <= current_time + 1.0e-14 and output_index < len(output_times) - 1:
            output_index += 1
            next_output_time = float(output_times[output_index])
        h_limit = min(max_h, float(duration) - current_time, next_output_time - current_time)
        h_trial = min(next_h, h_limit)
        h_trial = max(min_h, h_trial)
        accepted = False
        previous_active = int(active_count)
        previous_min_gap = float(min_gap)
        while not accepted:
            C = C_base if float(current_time) >= float(damping_start_time) else csr_matrix(K.shape, dtype=float)
            old_internal = C @ v + K @ u - f_gravity - f_contact_old
            c0 = 1.0 / (beta * h_trial * h_trial)
            c1 = gamma / (beta * h_trial)
            u_pred = u + h_trial * v + h_trial * h_trial * (0.5 - beta) * a
            v_pred = v + h_trial * (1.0 - gamma) * a
            u_guess = u_pred.copy()
            converged = False
            accepted_contact_force = f_contact_old
            accepted_min_gap = previous_min_gap
            accepted_active = previous_active
            active_previous_iteration: int | None = None
            iteration_used = 0
            for iteration in range(1, int(max_newton_iterations) + 1):
                current_guess = model.nodes + u_guess.reshape((-1, 3))
                a_guess = c0 * (u_guess - u_pred)
                v_guess = v_pred + gamma * h_trial * a_guess
                f_contact, K_contact, gap_guess, active_guess, *_ = _lagrangian_plane_contact_response(
                    model,
                    current_guess,
                    v_current=v_guess.reshape((-1, 3)),
                    area_weights=sample_area_weights,
                    samples=samples,
                    oracle=oracle,
                    contact_stiffness=contact_stiffness,
                    contact_damping=contact_damping,
                    velocity_tangent_factor=c1,
                    n_dofs=n_dofs,
                )
                accepted_contact_force = f_contact.copy()
                accepted_min_gap = float(gap_guess)
                accepted_active = int(active_guess)
                new_internal = C @ v_guess + K @ u_guess - f_gravity - f_contact
                residual = M @ a_guess + (1.0 + alpha) * new_internal - alpha * old_internal
                force_scale = max(1.0, float(np.linalg.norm(f_gravity)), float(np.linalg.norm(K @ u_guess)), float(np.linalg.norm(f_contact)))
                residual_ok = float(np.linalg.norm(residual)) <= float(residual_tolerance) * force_scale
                tangent = (c0 * M + (1.0 + alpha) * (c1 * C + K + K_contact)).tocsc()
                correction = np.asarray(spsolve(tangent, -residual), dtype=float)
                u_guess += correction
                total_newton_iterations += 1
                iteration_used = int(iteration)
                correction_ok = float(np.linalg.norm(correction)) <= float(tolerance) * max(1.0, float(np.linalg.norm(u_guess)))
                active_stable = active_previous_iteration is not None and int(active_previous_iteration) == int(active_guess)
                active_previous_iteration = int(active_guess)
                if correction_ok and residual_ok and (active_stable or int(active_guess) == 0):
                    converged = True
                    break
            contact_event = previous_active == 0 and accepted_active > 0 and previous_min_gap > 0.0
            if converged and bool(adaptive_increments) and contact_event and h_trial > event_floor * (1.0 + 1.0e-12):
                h_trial = max(event_floor, h_trial * cutback)
                cutback_count += 1
                continue
            if converged:
                accepted = True
                u_new = u_guess
                a_new = c0 * (u_new - u_pred)
                v_new = v_pred + gamma * h_trial * a_new
                u, v, a = u_new, v_new, a_new
                f_contact_old = accepted_contact_force
                min_gap = accepted_min_gap
                active_count = accepted_active
                current_time += h_trial
                accepted_increment_count += 1
                accepted_increments.append(float(h_trial))
                if bool(adaptive_increments):
                    if iteration_used <= int(target_newton_iterations):
                        next_h = min(max_h, max(min_h, h_trial * growth))
                    else:
                        next_h = min(max_h, max(min_h, h_trial))
                else:
                    next_h = max_h
            else:
                failed_newton_steps += 1
                if not bool(adaptive_increments) or h_trial <= min_h * (1.0 + 1.0e-12):
                    accepted = True
                    u_new = u_guess
                    a_new = c0 * (u_new - u_pred)
                    v_new = v_pred + gamma * h_trial * a_new
                    u, v, a = u_new, v_new, a_new
                    f_contact_old = accepted_contact_force
                    min_gap = accepted_min_gap
                    active_count = accepted_active
                    current_time += h_trial
                    accepted_increment_count += 1
                    accepted_increments.append(float(h_trial))
                    next_h = max_h
                else:
                    h_trial = max(min_h, h_trial * cutback)
                    cutback_count += 1
        if output_index < len(output_times) and current_time >= float(output_times[output_index]) - 1.0e-12:
            if output_index % output_stride_value == 0 or output_index == len(output_times) - 1:
                append_output(output_index, float(output_times[output_index]))
            output_index += 1

    elapsed = perf_counter() - solve_start
    min_accepted_increment = min(accepted_increments, default=0.0)
    max_accepted_increment = max(accepted_increments, default=0.0)
    for row in rows:
        row["sfc_solve_wall_seconds"] = float(elapsed)
        row["sfc_node_count"] = int(model.nodes.shape[0])
        row["sfc_element_count"] = int(model.elements.shape[0])
        row["sfc_surface_sample_count"] = int(len(samples))
        row["contact_stiffness"] = float(contact_stiffness)
        row["contact_damping"] = float(contact_damping)
        row["mass_damping"] = float(mass_damping)
        row["stiffness_damping"] = float(stiffness_damping)
        row["damping_start_time"] = float(damping_start_time)
        row["hht_alpha"] = float(alpha)
        row["quadrature_order"] = int(quadrature_order)
        row["hybrid_node_area_fraction"] = float(hybrid_node_area_fraction)
        row["adaptive_increments"] = str(bool(adaptive_increments)).lower()
        row["accepted_increment_count"] = int(accepted_increment_count)
        row["cutback_count"] = int(cutback_count)
        row["min_accepted_increment"] = float(min_accepted_increment)
        row["max_accepted_increment"] = float(max_accepted_increment)
        row["residual_tolerance"] = float(residual_tolerance)
        row["newton_iterations_total"] = int(total_newton_iterations)
        row["newton_failed_steps"] = int(failed_newton_steps)
    return rows


def _full_metric_rows(model: AbaqusSphereDropModel, comparison: list[Row], sfc_rows: list[Row], abaqus_rows: list[Row]) -> list[Row]:
    rows = _metric_rows(model, comparison, sfc_rows, abaqus_rows)
    sfc_first = _first_contact_time(sfc_rows)
    abaqus_first = _first_contact_time(abaqus_rows)
    vm_errors = np.asarray([float(row["von_mises_abs_error"]) for row in comparison], dtype=float)
    strain_errors = np.asarray([float(row["strain_norm_abs_error"]) for row in comparison], dtype=float)
    vm_p95_errors = np.asarray([float(row["p95_von_mises_abs_error"]) for row in comparison if "p95_von_mises_abs_error" in row], dtype=float)
    vm_mean_errors = np.asarray([float(row["volume_mean_von_mises_abs_error"]) for row in comparison if "volume_mean_von_mises_abs_error" in row], dtype=float)
    strain_p95_errors = np.asarray([float(row["p95_strain_norm_abs_error"]) for row in comparison if "p95_strain_norm_abs_error" in row], dtype=float)
    strain_mean_errors = np.asarray([float(row["volume_mean_strain_norm_abs_error"]) for row in comparison if "volume_mean_strain_norm_abs_error" in row], dtype=float)
    z_errors = np.asarray([float(row["z_cm_abs_error"]) for row in comparison], dtype=float)
    gap_errors = np.asarray([float(row["min_gap_abs_error"]) for row in comparison], dtype=float)
    force_errors = np.asarray([float(row["normal_force_z_abs_error"]) for row in comparison if "normal_force_z_abs_error" in row], dtype=float)
    energy_errors = np.asarray([float(row["contact_energy_abs_error"]) for row in comparison if "contact_energy_abs_error" in row], dtype=float)
    active_area_errors = np.asarray([float(row["active_contact_area_abs_error"]) for row in comparison if "active_contact_area_abs_error" in row], dtype=float)
    rows.extend(
        [
            {"metric": "sfc_first_contact_time", "value": "" if sfc_first is None else sfc_first, "status": "reported"},
            {"metric": "abaqus_first_contact_time", "value": "" if abaqus_first is None else abaqus_first, "status": "reported"},
            {
                "metric": "first_contact_time_abs_error",
                "value": "" if sfc_first is None or abaqus_first is None else abs(float(sfc_first) - float(abaqus_first)),
                "status": "reported" if sfc_first is not None and abaqus_first is not None else "not_available",
            },
            {"metric": "max_von_mises_abs_error", "value": float(np.max(vm_errors)), "status": "reported"},
            {"metric": "rms_von_mises_abs_error", "value": float(np.sqrt(np.mean(vm_errors**2))), "status": "reported"},
            {"metric": "max_p95_von_mises_abs_error", "value": "" if vm_p95_errors.size == 0 else float(np.max(vm_p95_errors)), "status": "not_available" if vm_p95_errors.size == 0 else "reported"},
            {"metric": "rms_p95_von_mises_abs_error", "value": "" if vm_p95_errors.size == 0 else float(np.sqrt(np.mean(vm_p95_errors**2))), "status": "not_available" if vm_p95_errors.size == 0 else "reported"},
            {"metric": "max_volume_mean_von_mises_abs_error", "value": "" if vm_mean_errors.size == 0 else float(np.max(vm_mean_errors)), "status": "not_available" if vm_mean_errors.size == 0 else "reported"},
            {"metric": "rms_volume_mean_von_mises_abs_error", "value": "" if vm_mean_errors.size == 0 else float(np.sqrt(np.mean(vm_mean_errors**2))), "status": "not_available" if vm_mean_errors.size == 0 else "reported"},
            {"metric": "max_strain_norm_abs_error", "value": float(np.max(strain_errors)), "status": "reported"},
            {"metric": "rms_strain_norm_abs_error", "value": float(np.sqrt(np.mean(strain_errors**2))), "status": "reported"},
            {"metric": "max_p95_strain_norm_abs_error", "value": "" if strain_p95_errors.size == 0 else float(np.max(strain_p95_errors)), "status": "not_available" if strain_p95_errors.size == 0 else "reported"},
            {"metric": "rms_p95_strain_norm_abs_error", "value": "" if strain_p95_errors.size == 0 else float(np.sqrt(np.mean(strain_p95_errors**2))), "status": "not_available" if strain_p95_errors.size == 0 else "reported"},
            {"metric": "max_volume_mean_strain_norm_abs_error", "value": "" if strain_mean_errors.size == 0 else float(np.max(strain_mean_errors)), "status": "not_available" if strain_mean_errors.size == 0 else "reported"},
            {"metric": "rms_volume_mean_strain_norm_abs_error", "value": "" if strain_mean_errors.size == 0 else float(np.sqrt(np.mean(strain_mean_errors**2))), "status": "not_available" if strain_mean_errors.size == 0 else "reported"},
            {"metric": "full_z_cm_l2_error", "value": float(np.linalg.norm(z_errors)), "status": "reported"},
            {"metric": "full_min_gap_l2_error", "value": float(np.linalg.norm(gap_errors)), "status": "reported"},
            {"metric": "max_normal_force_z_abs_error", "value": "" if force_errors.size == 0 else float(np.max(force_errors)), "status": "not_available" if force_errors.size == 0 else "reported"},
            {"metric": "rms_normal_force_z_abs_error", "value": "" if force_errors.size == 0 else float(np.sqrt(np.mean(force_errors**2))), "status": "not_available" if force_errors.size == 0 else "reported"},
            {"metric": "max_contact_energy_abs_error", "value": "" if energy_errors.size == 0 else float(np.max(energy_errors)), "status": "not_available" if energy_errors.size == 0 else "reported"},
            {"metric": "rms_contact_energy_abs_error", "value": "" if energy_errors.size == 0 else float(np.sqrt(np.mean(energy_errors**2))), "status": "not_available" if energy_errors.size == 0 else "reported"},
            {"metric": "max_active_contact_area_abs_error", "value": "" if active_area_errors.size == 0 else float(np.max(active_area_errors)), "status": "not_available" if active_area_errors.size == 0 else "reported"},
            {"metric": "rms_active_contact_area_abs_error", "value": "" if active_area_errors.size == 0 else float(np.sqrt(np.mean(active_area_errors**2))), "status": "not_available" if active_area_errors.size == 0 else "reported"},
            {"metric": "newton_iterations_total", "value": int(sfc_rows[0].get("newton_iterations_total", 0)) if sfc_rows else 0, "status": "reported"},
            {"metric": "newton_failed_steps", "value": int(sfc_rows[0].get("newton_failed_steps", 0)) if sfc_rows else 0, "status": "reported"},
            {"metric": "contact_stiffness", "value": float(sfc_rows[0].get("contact_stiffness", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "contact_damping", "value": float(sfc_rows[0].get("contact_damping", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "mass_damping", "value": float(sfc_rows[0].get("mass_damping", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "stiffness_damping", "value": float(sfc_rows[0].get("stiffness_damping", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "damping_start_time", "value": float(sfc_rows[0].get("damping_start_time", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "time_integrator", "value": str(sfc_rows[0].get("time_integrator", "")) if sfc_rows else "", "status": "reported"},
            {"metric": "hht_alpha", "value": float(sfc_rows[0].get("hht_alpha", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "contact_integration", "value": str(sfc_rows[0].get("contact_integration", "")) if sfc_rows else "", "status": "reported"},
            {"metric": "quadrature_order", "value": int(sfc_rows[0].get("quadrature_order", 0)) if sfc_rows else 0, "status": "reported"},
            {"metric": "hybrid_node_area_fraction", "value": float(sfc_rows[0].get("hybrid_node_area_fraction", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "adaptive_increments", "value": str(sfc_rows[0].get("adaptive_increments", "")) if sfc_rows else "", "status": "reported"},
            {"metric": "accepted_increment_count", "value": int(sfc_rows[0].get("accepted_increment_count", 0)) if sfc_rows else 0, "status": "reported"},
            {"metric": "cutback_count", "value": int(sfc_rows[0].get("cutback_count", 0)) if sfc_rows else 0, "status": "reported"},
            {"metric": "min_accepted_increment", "value": float(sfc_rows[0].get("min_accepted_increment", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "max_accepted_increment", "value": float(sfc_rows[0].get("max_accepted_increment", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "residual_tolerance", "value": float(sfc_rows[0].get("residual_tolerance", 0.0)) if sfc_rows else 0.0, "status": "reported"},
        ]
    )
    return rows


def _plot_contact_count(path: Path, comparison: list[Row], sfc_rows: list[Row], abaqus_rows: list[Row]) -> None:
    _configure_plot_style()
    t_sfc = np.asarray([float(row["time"]) for row in sfc_rows], dtype=float)
    t_abq = np.asarray([float(row["time"]) for row in abaqus_rows], dtype=float)
    c_sfc = np.asarray([float(row["active_contact_count"]) for row in sfc_rows], dtype=float)
    c_abq = np.asarray([float(row["active_contact_count"]) for row in abaqus_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(3.35, 2.2), constrained_layout=True)
    ax.plot(t_abq, c_abq, color="#1f4e79", linewidth=1.3, label="Abaqus")
    ax.plot(t_sfc, c_sfc, color="#c0504d", linewidth=1.3, linestyle="--", label="SFC Lagrangian SDF")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Active contact samples")
    ax.set_title("Contact activation history")
    ax.grid(True, color="#d9d9d9", linewidth=0.5)
    ax.legend(loc="best", frameon=True)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _write_full_summary(path: Path, metrics: list[Row], outputs: dict[str, Path]) -> None:
    by_metric = {str(row["metric"]): row for row in metrics}
    lines = [
        "# Abaqus Sphere Drop Full Validation",
        "",
        "This full-run validation compares an independent SFC Lagrangian-SDF penalty-contact solve against the exported Abaqus native-contact VTK trajectory for the same input deck.",
        "",
        "## Scope",
        "",
        "- SFC reconstructs the C3D4 sphere from the Abaqus `.inp` and assembles FEM matrices internally.",
        "- Abaqus is used only as an external native-contact reference trajectory.",
        "- When the Abaqus deck provides a linear penalty pressure-overclosure value, SFC uses the same normal penalty stiffness unless explicitly overridden.",
        "- The reported errors are alignment diagnostics, not a claim of source-level Abaqus equivalence.",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value | Status |",
        "| --- | ---: | --- |",
    ]
    for key in (
        "duration_seconds",
        "initial_gap",
        "sfc_first_contact_time",
        "abaqus_first_contact_time",
        "first_contact_time_abs_error",
        "max_z_cm_abs_error",
        "max_min_gap_abs_error",
        "max_von_mises_abs_error",
        "rms_p95_von_mises_abs_error",
        "rms_volume_mean_von_mises_abs_error",
        "max_strain_norm_abs_error",
        "rms_p95_strain_norm_abs_error",
        "rms_volume_mean_strain_norm_abs_error",
        "sfc_solve_wall_seconds",
        "contact_stiffness",
        "contact_damping",
        "mass_damping",
        "stiffness_damping",
        "damping_start_time",
        "time_integrator",
        "hht_alpha",
        "contact_integration",
        "quadrature_order",
        "hybrid_node_area_fraction",
        "adaptive_increments",
        "accepted_increment_count",
        "cutback_count",
        "min_accepted_increment",
        "max_accepted_increment",
        "max_normal_force_z_abs_error",
        "max_contact_energy_abs_error",
        "max_active_contact_area_abs_error",
        "newton_failed_steps",
    ):
        row = by_metric[key]
        lines.append(f"| `{key}` | {row['value']} | {row['status']} |")
    lines.extend(["", "## Outputs", ""])
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value.name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    out_dir: Path,
    *,
    inp: Path = DEFAULT_INP,
    vtk_dir: Path = DEFAULT_VTK_DIR,
    duration: float = 3.0,
    dt: float = 0.001,
    contact_stiffness: float | None = None,
    contact_damping: float = 0.0,
    mass_damping: float = 0.0,
    stiffness_damping: float = 0.0,
    damping_start_time: float = 0.0,
    integrator: str = "newmark",
    hht_alpha: float = 0.0,
    contact_integration: str = "node",
    quadrature_order: int = 3,
    hybrid_node_area_fraction: float = 0.15,
    adaptive_increments: bool = False,
    min_increment: float = 1.0e-8,
    max_increment: float | None = None,
    cutback_factor: float = 0.5,
    growth_factor: float = 1.25,
    target_newton_iterations: int = 4,
    residual_tolerance: float = 1.0e-8,
    contact_event_substeps: int = 8,
    output_stride: int = 1,
    max_newton_iterations: int = 8,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = parse_sphere_drop_inp(inp)
    duration_value = min(float(duration), model.abaqus_duration)
    contact_stiffness_value = _resolved_contact_stiffness(model, contact_stiffness, fallback=1.0e9)
    sfc_rows = run_sfc_lagrangian_sdf_full_history(
        model,
        duration=duration_value,
        dt=float(dt),
        contact_stiffness=contact_stiffness_value,
        contact_damping=float(contact_damping),
        mass_damping=float(mass_damping),
        stiffness_damping=float(stiffness_damping),
        damping_start_time=float(damping_start_time),
        integrator=str(integrator),
        hht_alpha=float(hht_alpha),
        contact_integration=str(contact_integration),
        quadrature_order=int(quadrature_order),
        hybrid_node_area_fraction=float(hybrid_node_area_fraction),
        adaptive_increments=bool(adaptive_increments),
        min_increment=float(min_increment),
        max_increment=None if max_increment is None else float(max_increment),
        cutback_factor=float(cutback_factor),
        growth_factor=float(growth_factor),
        target_newton_iterations=int(target_newton_iterations),
        residual_tolerance=float(residual_tolerance),
        contact_event_substeps=int(contact_event_substeps),
        output_stride=int(output_stride),
        max_newton_iterations=int(max_newton_iterations),
    )
    abaqus_rows = _abaqus_history_from_vtk(model, vtk_dir, duration=duration_value)
    comparison = _comparison_rows(abaqus_rows, sfc_rows)
    metrics = _full_metric_rows(model, comparison, sfc_rows, abaqus_rows)
    outputs = {
        "sfc_history": out_dir / "abaqus_sphere_drop_full_sfc_history.csv",
        "abaqus_history": out_dir / "abaqus_sphere_drop_full_abaqus_history.csv",
        "comparison": out_dir / "abaqus_sphere_drop_full_comparison.csv",
        "metrics": out_dir / "abaqus_sphere_drop_full_metrics.csv",
        "z_cm_curve": out_dir / "abaqus_sphere_drop_full_z_cm.png",
        "z_cm_error_curve": out_dir / "abaqus_sphere_drop_full_z_cm_abs_error.png",
        "gap_curve": out_dir / "abaqus_sphere_drop_full_min_gap.png",
        "gap_error_curve": out_dir / "abaqus_sphere_drop_full_min_gap_abs_error.png",
        "stress_curve": out_dir / "abaqus_sphere_drop_full_von_mises.png",
        "stress_error_curve": out_dir / "abaqus_sphere_drop_full_von_mises_abs_error.png",
        "strain_curve": out_dir / "abaqus_sphere_drop_full_strain_norm.png",
        "strain_error_curve": out_dir / "abaqus_sphere_drop_full_strain_norm_abs_error.png",
        "contact_count_curve": out_dir / "abaqus_sphere_drop_full_active_contact.png",
        "summary": out_dir / "abaqus_sphere_drop_full_summary.md",
    }
    has_force_reference = bool(comparison and "normal_force_z_abs_error" in comparison[0])
    has_energy_reference = bool(comparison and "contact_energy_abs_error" in comparison[0])
    has_active_area_reference = bool(comparison and "active_contact_area_abs_error" in comparison[0])
    has_robust_stress_metrics = bool(comparison and "p95_von_mises_abs_error" in comparison[0] and "volume_mean_von_mises_abs_error" in comparison[0])
    has_robust_strain_metrics = bool(comparison and "p95_strain_norm_abs_error" in comparison[0] and "volume_mean_strain_norm_abs_error" in comparison[0])
    if has_robust_stress_metrics:
        outputs["stress_p95_curve"] = out_dir / "abaqus_sphere_drop_full_von_mises_p95.png"
        outputs["stress_p95_error_curve"] = out_dir / "abaqus_sphere_drop_full_von_mises_p95_abs_error.png"
        outputs["stress_volume_mean_curve"] = out_dir / "abaqus_sphere_drop_full_von_mises_volume_mean.png"
        outputs["stress_volume_mean_error_curve"] = out_dir / "abaqus_sphere_drop_full_von_mises_volume_mean_abs_error.png"
    if has_robust_strain_metrics:
        outputs["strain_p95_curve"] = out_dir / "abaqus_sphere_drop_full_strain_norm_p95.png"
        outputs["strain_p95_error_curve"] = out_dir / "abaqus_sphere_drop_full_strain_norm_p95_abs_error.png"
        outputs["strain_volume_mean_curve"] = out_dir / "abaqus_sphere_drop_full_strain_norm_volume_mean.png"
        outputs["strain_volume_mean_error_curve"] = out_dir / "abaqus_sphere_drop_full_strain_norm_volume_mean_abs_error.png"
    if has_force_reference:
        outputs["normal_force_curve"] = out_dir / "abaqus_sphere_drop_full_normal_force_z.png"
        outputs["normal_force_error_curve"] = out_dir / "abaqus_sphere_drop_full_normal_force_z_abs_error.png"
    if has_energy_reference:
        outputs["contact_energy_curve"] = out_dir / "abaqus_sphere_drop_full_contact_energy.png"
        outputs["contact_energy_error_curve"] = out_dir / "abaqus_sphere_drop_full_contact_energy_abs_error.png"
    if has_active_area_reference:
        outputs["active_area_curve"] = out_dir / "abaqus_sphere_drop_full_active_area.png"
        outputs["active_area_error_curve"] = out_dir / "abaqus_sphere_drop_full_active_area_abs_error.png"
    _write_csv(outputs["sfc_history"], sfc_rows)
    _write_csv(outputs["abaqus_history"], abaqus_rows)
    _write_csv(outputs["comparison"], comparison)
    _write_csv(outputs["metrics"], metrics)
    _plot_curve(outputs["z_cm_curve"], comparison, y_sfc="sfc_z_cm", y_abq="abaqus_z_cm", ylabel="Mass-center z (m)", title="Full sphere-drop trajectory")
    _plot_abs_error_curve(outputs["z_cm_error_curve"], comparison, y_error="z_cm_abs_error", ylabel="Mass-center z abs. error (m)", title="Displacement trajectory error")
    _plot_curve(outputs["gap_curve"], comparison, y_sfc="sfc_min_gap", y_abq="abaqus_min_gap", ylabel="Minimum gap (m)", title="Full sphere-plane clearance")
    _plot_abs_error_curve(outputs["gap_error_curve"], comparison, y_error="min_gap_abs_error", ylabel="Minimum gap abs. error (m)", title="Clearance error")
    _plot_curve(outputs["stress_curve"], comparison, y_sfc="sfc_max_von_mises", y_abq="abaqus_max_von_mises", ylabel="Max von Mises stress", title="Full stress history")
    _plot_abs_error_curve(outputs["stress_error_curve"], comparison, y_error="von_mises_abs_error", ylabel="Max von Mises abs. error", title="Stress history error")
    _plot_curve(outputs["strain_curve"], comparison, y_sfc="sfc_max_strain_norm", y_abq="abaqus_max_strain_norm", ylabel="Max strain norm", title="Full strain history")
    _plot_abs_error_curve(outputs["strain_error_curve"], comparison, y_error="strain_norm_abs_error", ylabel="Max strain norm abs. error", title="Strain history error")
    if has_robust_stress_metrics:
        _plot_curve(outputs["stress_p95_curve"], comparison, y_sfc="sfc_p95_von_mises", y_abq="abaqus_p95_von_mises", ylabel="P95 von Mises stress", title="P95 stress history")
        _plot_abs_error_curve(outputs["stress_p95_error_curve"], comparison, y_error="p95_von_mises_abs_error", ylabel="P95 von Mises abs. error", title="P95 stress error")
        _plot_curve(outputs["stress_volume_mean_curve"], comparison, y_sfc="sfc_volume_mean_von_mises", y_abq="abaqus_volume_mean_von_mises", ylabel="Volume-mean von Mises stress", title="Volume-mean stress history")
        _plot_abs_error_curve(outputs["stress_volume_mean_error_curve"], comparison, y_error="volume_mean_von_mises_abs_error", ylabel="Volume-mean stress abs. error", title="Volume-mean stress error")
    if has_robust_strain_metrics:
        _plot_curve(outputs["strain_p95_curve"], comparison, y_sfc="sfc_p95_strain_norm", y_abq="abaqus_p95_strain_norm", ylabel="P95 strain norm", title="P95 strain history")
        _plot_abs_error_curve(outputs["strain_p95_error_curve"], comparison, y_error="p95_strain_norm_abs_error", ylabel="P95 strain norm abs. error", title="P95 strain error")
        _plot_curve(outputs["strain_volume_mean_curve"], comparison, y_sfc="sfc_volume_mean_strain_norm", y_abq="abaqus_volume_mean_strain_norm", ylabel="Volume-mean strain norm", title="Volume-mean strain history")
        _plot_abs_error_curve(outputs["strain_volume_mean_error_curve"], comparison, y_error="volume_mean_strain_norm_abs_error", ylabel="Volume-mean strain norm abs. error", title="Volume-mean strain error")
    if has_force_reference:
        _plot_curve(outputs["normal_force_curve"], comparison, y_sfc="sfc_normal_force_z", y_abq="abaqus_normal_force_z", ylabel="Normal contact force (N)", title="Normal contact force history")
        _plot_abs_error_curve(outputs["normal_force_error_curve"], comparison, y_error="normal_force_z_abs_error", ylabel="Normal force abs. error (N)", title="Normal contact force error")
    if has_energy_reference:
        _plot_curve(outputs["contact_energy_curve"], comparison, y_sfc="sfc_contact_energy", y_abq="abaqus_contact_energy", ylabel="Contact energy (J)", title="Contact energy history")
        _plot_abs_error_curve(outputs["contact_energy_error_curve"], comparison, y_error="contact_energy_abs_error", ylabel="Contact energy abs. error (J)", title="Contact energy error")
    if has_active_area_reference:
        _plot_curve(outputs["active_area_curve"], comparison, y_sfc="sfc_active_contact_area", y_abq="abaqus_active_contact_area", ylabel="Active contact area (m$^2$)", title="Active contact area history")
        _plot_abs_error_curve(outputs["active_area_error_curve"], comparison, y_error="active_contact_area_abs_error", ylabel="Active area abs. error (m$^2$)", title="Active contact area error")
    _plot_contact_count(outputs["contact_count_curve"], comparison, sfc_rows, abaqus_rows)
    _write_full_summary(outputs["summary"], metrics, outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inp", type=Path, default=DEFAULT_INP)
    parser.add_argument("--vtk-dir", type=Path, default=DEFAULT_VTK_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "abaqus_sphere_drop_full")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--contact-stiffness", type=float, default=None)
    parser.add_argument("--contact-damping", type=float, default=0.0)
    parser.add_argument("--mass-damping", type=float, default=0.0)
    parser.add_argument("--stiffness-damping", type=float, default=0.0)
    parser.add_argument("--damping-start-time", type=float, default=0.0)
    parser.add_argument("--integrator", choices=("newmark", "hht"), default="newmark")
    parser.add_argument("--hht-alpha", type=float, default=0.0)
    parser.add_argument("--contact-integration", choices=("node", "surface", "hybrid"), default="node")
    parser.add_argument("--quadrature-order", type=int, choices=(1, 3, 7), default=3)
    parser.add_argument("--hybrid-node-area-fraction", type=float, default=0.15)
    parser.add_argument("--adaptive-increments", action="store_true")
    parser.add_argument("--min-increment", type=float, default=1.0e-8)
    parser.add_argument("--max-increment", type=float, default=None)
    parser.add_argument("--cutback-factor", type=float, default=0.5)
    parser.add_argument("--growth-factor", type=float, default=1.25)
    parser.add_argument("--target-newton-iterations", type=int, default=4)
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--contact-event-substeps", type=int, default=8)
    parser.add_argument("--output-stride", type=int, default=1)
    parser.add_argument("--max-newton-iterations", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(
        args.out_dir,
        inp=args.inp,
        vtk_dir=args.vtk_dir,
        duration=float(args.duration),
        dt=float(args.dt),
        contact_stiffness=args.contact_stiffness,
        contact_damping=float(args.contact_damping),
        mass_damping=float(args.mass_damping),
        stiffness_damping=float(args.stiffness_damping),
        damping_start_time=float(args.damping_start_time),
        integrator=str(args.integrator),
        hht_alpha=float(args.hht_alpha),
        contact_integration=str(args.contact_integration),
        quadrature_order=int(args.quadrature_order),
        hybrid_node_area_fraction=float(args.hybrid_node_area_fraction),
        adaptive_increments=bool(args.adaptive_increments),
        min_increment=float(args.min_increment),
        max_increment=args.max_increment,
        cutback_factor=float(args.cutback_factor),
        growth_factor=float(args.growth_factor),
        target_newton_iterations=int(args.target_newton_iterations),
        residual_tolerance=float(args.residual_tolerance),
        contact_event_substeps=int(args.contact_event_substeps),
        output_stride=int(args.output_stride),
        max_newton_iterations=int(args.max_newton_iterations),
    )
    print("Abaqus sphere-drop full validation complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
