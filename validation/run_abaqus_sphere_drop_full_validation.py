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

from sfc.contact import LagrangianSDFContactOracle, SurfaceSample, lagrangian_oracle_penalty_response  # noqa: E402
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
    _stress_metrics,
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


def _lagrangian_plane_contact_response(
    model: AbaqusSphereDropModel,
    x_current: np.ndarray,
    *,
    v_current: np.ndarray | None = None,
    surface_nodes: np.ndarray,
    area_weights: np.ndarray,
    samples: list[SurfaceSample],
    oracle: LagrangianSDFContactOracle,
    contact_stiffness: float,
    contact_damping: float = 0.0,
    velocity_tangent_factor: float = 0.0,
    n_dofs: int,
) -> tuple[np.ndarray, csr_matrix, float, int, float, float]:
    """Return SFC Lagrangian-SDF contact force and tangent against the plane."""

    direct_gaps = x_current[surface_nodes, 2] - model.plane_z
    min_gap = float(np.min(direct_gaps))
    active_ids = np.nonzero(direct_gaps <= 0.0)[0]
    if active_ids.size == 0:
        return np.zeros(n_dofs, dtype=float), csr_matrix((n_dofs, n_dofs), dtype=float), min_gap, 0, 0.0, 0.0

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
    samples = _node_samples(surface_nodes)
    oracle = _plane_oracle(model)
    n_dofs = body.n_dofs
    u = np.zeros(n_dofs, dtype=float)
    v = np.zeros(n_dofs, dtype=float)

    x0 = model.nodes.copy()
    f_contact, K_contact, min_gap, active_count, max_pen, contact_energy = _lagrangian_plane_contact_response(
        model,
        x0,
        v_current=v.reshape((-1, 3)),
        surface_nodes=surface_nodes,
        area_weights=area_weights,
        samples=samples,
        oracle=oracle,
        contact_stiffness=contact_stiffness,
        contact_damping=contact_damping,
        velocity_tangent_factor=0.0,
        n_dofs=n_dofs,
    )
    C_initial = C_base if 0.0 >= float(damping_start_time) else csr_matrix(K.shape, dtype=float)
    a = np.asarray(spsolve(M.tocsc(), f_gravity + f_contact - C_initial @ v - K @ u), dtype=float)

    beta = 0.25
    gamma = 0.5
    c0 = 1.0 / (beta * dt * dt)
    rows: list[Row] = []
    times = np.arange(0.0, duration + 0.5 * dt, dt)
    output_stride_value = max(1, int(output_stride))
    solve_start = perf_counter()
    total_newton_iterations = 0
    failed_newton_steps = 0

    for step, time in enumerate(times):
        if step % output_stride_value == 0 or step == len(times) - 1:
            current = model.nodes + u.reshape((-1, 3))
            f_contact, _Kc, min_gap, active_count, max_pen, contact_energy = _lagrangian_plane_contact_response(
                model,
                current,
                v_current=v.reshape((-1, 3)),
                surface_nodes=surface_nodes,
                area_weights=area_weights,
                samples=samples,
                oracle=oracle,
                contact_stiffness=contact_stiffness,
                contact_damping=contact_damping,
                velocity_tangent_factor=0.0,
                n_dofs=n_dofs,
            )
            max_vm, max_strain = _stress_metrics(model, u)
            rows.append(
                {
                    "source": "sfc_lagrangian_sdf_full",
                    "step": int(step),
                    "time": float(time),
                    "z_cm": _mass_weighted_center_z(model.nodes, model.elements, current),
                    "min_gap": float(min_gap),
                    "active_contact_count": int(active_count),
                    "max_penetration": float(max_pen),
                    "normal_force_z": float(np.sum(f_contact[2::3])),
                    "contact_energy": float(contact_energy),
                    "max_von_mises": float(max_vm),
                    "max_strain_norm": float(max_strain),
                    "contact_path": "MaterialSDF+LagrangianSDFContactOracle",
                }
            )
        if step == len(times) - 1:
            break

        C = C_base if float(time) >= float(damping_start_time) else csr_matrix(K.shape, dtype=float)
        u_pred = u + dt * v + dt * dt * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a
        c1 = gamma / (beta * dt)
        u_guess = u_pred.copy()
        converged = False
        for iteration in range(1, int(max_newton_iterations) + 1):
            current_guess = model.nodes + u_guess.reshape((-1, 3))
            a_guess = c0 * (u_guess - u_pred)
            v_guess = v_pred + gamma * dt * a_guess
            f_contact, K_contact, *_ = _lagrangian_plane_contact_response(
                model,
                current_guess,
                v_current=v_guess.reshape((-1, 3)),
                surface_nodes=surface_nodes,
                area_weights=area_weights,
                samples=samples,
                oracle=oracle,
                contact_stiffness=contact_stiffness,
                contact_damping=contact_damping,
                velocity_tangent_factor=c1,
                n_dofs=n_dofs,
            )
            residual = M @ a_guess + C @ v_guess + K @ u_guess - f_gravity - f_contact
            tangent = (c0 * M + c1 * C + K + K_contact).tocsc()
            correction = np.asarray(spsolve(tangent, -residual), dtype=float)
            u_guess += correction
            total_newton_iterations += 1
            if float(np.linalg.norm(correction)) <= float(tolerance) * max(1.0, float(np.linalg.norm(u_guess))):
                converged = True
                break
        if not converged:
            failed_newton_steps += 1
        u_new = u_guess
        a_new = c0 * (u_new - u_pred)
        v_new = v_pred + gamma * dt * a_new
        u, v, a = u_new, v_new, a_new

    elapsed = perf_counter() - solve_start
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
        row["newton_iterations_total"] = int(total_newton_iterations)
        row["newton_failed_steps"] = int(failed_newton_steps)
    return rows


def _full_metric_rows(model: AbaqusSphereDropModel, comparison: list[Row], sfc_rows: list[Row], abaqus_rows: list[Row]) -> list[Row]:
    rows = _metric_rows(model, comparison, sfc_rows, abaqus_rows)
    sfc_first = _first_contact_time(sfc_rows)
    abaqus_first = _first_contact_time(abaqus_rows)
    vm_errors = np.asarray([abs(float(row["sfc_max_von_mises"]) - float(row["abaqus_max_von_mises"])) for row in comparison], dtype=float)
    strain_errors = np.asarray([abs(float(row["sfc_max_strain_norm"]) - float(row["abaqus_max_strain_norm"])) for row in comparison], dtype=float)
    z_errors = np.asarray([float(row["z_cm_abs_error"]) for row in comparison], dtype=float)
    gap_errors = np.asarray([float(row["min_gap_abs_error"]) for row in comparison], dtype=float)
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
            {"metric": "max_strain_norm_abs_error", "value": float(np.max(strain_errors)), "status": "reported"},
            {"metric": "rms_strain_norm_abs_error", "value": float(np.sqrt(np.mean(strain_errors**2))), "status": "reported"},
            {"metric": "full_z_cm_l2_error", "value": float(np.linalg.norm(z_errors)), "status": "reported"},
            {"metric": "full_min_gap_l2_error", "value": float(np.linalg.norm(gap_errors)), "status": "reported"},
            {"metric": "newton_iterations_total", "value": int(sfc_rows[0].get("newton_iterations_total", 0)) if sfc_rows else 0, "status": "reported"},
            {"metric": "newton_failed_steps", "value": int(sfc_rows[0].get("newton_failed_steps", 0)) if sfc_rows else 0, "status": "reported"},
            {"metric": "contact_stiffness", "value": float(sfc_rows[0].get("contact_stiffness", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "contact_damping", "value": float(sfc_rows[0].get("contact_damping", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "mass_damping", "value": float(sfc_rows[0].get("mass_damping", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "stiffness_damping", "value": float(sfc_rows[0].get("stiffness_damping", 0.0)) if sfc_rows else 0.0, "status": "reported"},
            {"metric": "damping_start_time", "value": float(sfc_rows[0].get("damping_start_time", 0.0)) if sfc_rows else 0.0, "status": "reported"},
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
        "This full-run validation compares an independent SFC Lagrangian-SDF penalty-contact solve against the exported Abaqus/Explicit native-contact VTK trajectory for the same input deck.",
        "",
        "## Scope",
        "",
        "- SFC reconstructs the C3D4 sphere from the Abaqus `.inp` and assembles FEM matrices internally.",
        "- Abaqus is used only as an external native-contact reference trajectory.",
        "- The contact laws are not identical: Abaqus uses Explicit hard normal contact, while SFC uses a frictionless Lagrangian-SDF penalty response. The reported errors are therefore alignment diagnostics, not a claim of source-level Abaqus equivalence.",
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
        "max_strain_norm_abs_error",
        "sfc_solve_wall_seconds",
        "contact_stiffness",
        "contact_damping",
        "mass_damping",
        "stiffness_damping",
        "damping_start_time",
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
        output_stride=int(args.output_stride),
        max_newton_iterations=int(args.max_newton_iterations),
    )
    print("Abaqus sphere-drop full validation complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
