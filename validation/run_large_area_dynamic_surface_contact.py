"""Large-area dynamic surface-to-surface SDF contact benchmark.

This runner implements the selected "scheme B" engineering case: a dense upper
C3D8 body is loaded by a pressure on its top surface, moves freely in the
normal direction, and contacts a deformable lower C3D8 body.  The contact stays
active over a large surface patch, so the benchmark stresses the true
``RequiredPointSDFWorkspace.build(...) -> field_contact`` path with
many quadrature samples per time step.

CalculiX support is optional and used only as an external native-contact
reference.  The SFC path builds and solves its own model independently.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, factorized, spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt  # noqa: E402

from scipy.sparse import block_diag  # noqa: E402
from sfc.contact.field_contact import (  # noqa: E402
    surface_to_surface_field_penalty_response_vectorized,
    triangle_surface_quadrature_cache,
)
from sfc.fem import DeformableBody  # noqa: E402
from sfc.fem.assembler import assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF, RequiredPointSDFWorkspace  # noqa: E402
from validation.run_c3d8_contact_trajectory_validation import (  # noqa: E402
    _append_id_list,
    _parse_nodal_vectors,
    _parse_totals,
    _wsl_path,
)
from validation.run_fig6_inspired_frictionless_contact import (  # noqa: E402
    BoxMesh,
    _element_fields,
    _fixed_bottom_dofs,
    _free_dofs,
    _structured_hex_box,
    _surface_grid,
    _top_cell_ids_for_points,
    _top_triangles,
)
from validation.vtk_frame_series import write_hex_frame_series  # noqa: E402

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class DynamicSurfaceConfig:
    """Parameters for the large-area pressure-driven dynamic contact case."""

    nx: int
    ny: int
    nz: int
    driver_nx: int
    driver_ny: int
    total_time: float
    dt: float
    ramp_time: float
    initial_gap: float
    closure: float
    oscillation_amplitude: float
    oscillation_frequency: float
    spacing: float
    band_radius: float
    quadrature_order: int
    pressure_stiffness: float
    peak_pressure: float
    young_modulus: float
    poisson_ratio: float
    density: float
    damping_alpha: float
    newmark_iterations: int
    frame_stride: int
    sdf_batch_projection_threshold: int
    sdf_candidate_padding: float | None
    use_contact_tangent: bool
    use_matrix_free_contact_tangent: bool
    use_cpp_contact_tangent_solver: bool
    cpp_contact_tangent_preconditioner: str
    contact_tangent_cg_rtol: float
    contact_tangent_cg_atol: float
    contact_tangent_cg_maxiter: int
    defer_diagnostics: bool

    @property
    def output_frequency(self) -> int:
        return 1


@dataclass(slots=True)
class SFCDynamicResult:
    history: list[Row]
    timing: list[Row]
    frames: list[tuple[float, np.ndarray]]
    final_displacement: np.ndarray
    driver_ref: np.ndarray
    driver_faces: np.ndarray
    lower_mesh: BoxMesh
    upper_mesh: BoxMesh | None = None
    upper_frames: list[tuple[float, np.ndarray]] | None = None
    upper_bottom_faces: np.ndarray | None = None
    solve_wall_seconds: float = 0.0
    core_solve_seconds: float = 0.0
    diagnostics_seconds: float = 0.0
    cpp_tangent_iterations: int = 0
    cpp_tangent_solves: int = 0


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _motion_z_offset(t: float, cfg: DynamicSurfaceConfig) -> float:
    """Prescribed upper-surface z displacement."""

    time_value = float(max(t, 0.0))
    if cfg.ramp_time > 0.0 and time_value <= cfg.ramp_time:
        ramp = 0.5 * (1.0 - math.cos(math.pi * time_value / cfg.ramp_time))
        return -cfg.closure * ramp
    phase_t = max(time_value - cfg.ramp_time, 0.0)
    oscillation = cfg.oscillation_amplitude * math.sin(2.0 * math.pi * cfg.oscillation_frequency * phase_t)
    return -cfg.closure + oscillation


def _pressure_value(t: float, cfg: DynamicSurfaceConfig) -> float:
    """Top-surface pressure history used by the pressure-driven benchmark."""

    time_value = float(max(t, 0.0))
    if cfg.ramp_time > 0.0 and time_value <= cfg.ramp_time:
        ramp = 0.5 * (1.0 - math.cos(math.pi * time_value / cfg.ramp_time))
        return float(cfg.peak_pressure) * ramp
    phase_t = max(time_value - cfg.ramp_time, 0.0)
    oscillation = cfg.oscillation_amplitude * math.sin(2.0 * math.pi * cfg.oscillation_frequency * phase_t)
    return float(cfg.peak_pressure) * max(0.0, 1.0 + oscillation)


def _driver_positions(driver_ref: np.ndarray, t: float, cfg: DynamicSurfaceConfig) -> np.ndarray:
    driver = np.asarray(driver_ref, dtype=float).copy()
    driver[:, 2] += _motion_z_offset(t, cfg)
    return driver


def _make_geometry(cfg: DynamicSurfaceConfig) -> tuple[BoxMesh, np.ndarray, np.ndarray]:
    lower = _structured_hex_box(
        nx=cfg.nx,
        ny=cfg.ny,
        nz=cfg.nz,
        origin=(-10.0, -10.0, 0.0),
        size=(20.0, 20.0, 4.0),
    )
    driver_ref, driver_faces = _surface_grid(
        nx=cfg.driver_nx,
        ny=cfg.driver_ny,
        origin=(-7.5, -7.5, 4.0 + cfg.initial_gap),
        size=(15.0, 15.0),
    )
    return lower, driver_ref, driver_faces


def _make_pressure_geometry(cfg: DynamicSurfaceConfig) -> tuple[BoxMesh, BoxMesh]:
    lower = _structured_hex_box(
        nx=cfg.nx,
        ny=cfg.ny,
        nz=cfg.nz,
        origin=(-10.0, -10.0, 0.0),
        size=(20.0, 20.0, 4.0),
    )
    upper = _structured_hex_box(
        nx=cfg.driver_nx,
        ny=cfg.driver_ny,
        nz=max(1, min(cfg.nz, 2)),
        origin=(-7.5, -7.5, 4.0 + cfg.initial_gap),
        size=(15.0, 15.0, 1.0),
    )
    return lower, upper


def _bottom_triangles(mesh: BoxMesh) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for j in range(mesh.ny):
        for i in range(mesh.nx):
            eid = j * mesh.nx + i
            element = mesh.elements[eid]
            quad = element[[0, 3, 2, 1]]
            faces.append((int(quad[0]), int(quad[1]), int(quad[2])))
            faces.append((int(quad[0]), int(quad[2]), int(quad[3])))
    return np.asarray(faces, dtype=np.int64)


def _surface_pressure_force(
    x_reference: np.ndarray,
    faces: np.ndarray,
    *,
    pressure: float,
    node_count: int,
    dof_offset: int = 0,
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
    quadrature_order: int = 3,
) -> np.ndarray:
    cache = triangle_surface_quadrature_cache(faces, x_reference, order=quadrature_order)
    force = np.zeros(3 * int(node_count) + int(dof_offset), dtype=float)
    direction_vec = np.asarray(direction, dtype=float)
    for node_ids, weights, area_weight in zip(cache.node_ids, cache.weights, cache.area_weights, strict=True):
        for node, weight in zip(node_ids, weights, strict=True):
            start = int(dof_offset) + 3 * int(node)
            force[start : start + 3] += float(pressure) * float(area_weight) * float(weight) * direction_vec
    return force[int(dof_offset) :]


def _normal_force_from_response(response: Any, pressure_stiffness: float) -> float:
    gaps = np.asarray(response.gaps, dtype=float)
    penetration = np.maximum(-gaps, 0.0)
    return float(np.sum(float(pressure_stiffness) * np.asarray(response.quadrature_weights, dtype=float) * penetration))


def _pressure_contact_diagnostics_row(
    *,
    time_value: float,
    upper: BoxMesh,
    lower: BoxMesh,
    upper_displacement: np.ndarray,
    lower_displacement: np.ndarray,
    upper_bottom_faces: np.ndarray,
    lower_top_faces: np.ndarray,
    contact_cache: Any,
    sdf_workspace: RequiredPointSDFWorkspace,
    cfg: DynamicSurfaceConfig,
) -> tuple[Row, Row]:
    """Evaluate exact field-contact diagnostics for one accepted state."""

    t = float(time_value)
    u_upper = np.asarray(upper_displacement, dtype=float)
    u_lower = np.asarray(lower_displacement, dtype=float)
    x_upper = upper.X + u_upper
    x_lower = lower.X + u_lower
    q_points = contact_cache.points(x_upper)
    build_t0 = time.perf_counter()
    master_sdf = sdf_workspace.build(
        x_lower,
        lower_top_faces,
        q_points,
    )
    build_elapsed = time.perf_counter() - build_t0
    query_t0 = time.perf_counter()
    response = surface_to_surface_field_penalty_response_vectorized(
        x_upper,
        upper_bottom_faces,
        master_sdf,
        pressure_stiffness=cfg.pressure_stiffness,
        n_total_dofs=3 * upper.X.shape[0] + 3 * lower.X.shape[0],
        slave_x_reference=upper.X,
        quadrature_cache=contact_cache,
        slave_dof_offset=0,
        master_dof_offset=3 * upper.X.shape[0],
    )
    query_elapsed = time.perf_counter() - query_t0
    normal_force = _normal_force_from_response(response, cfg.pressure_stiffness)
    gaps = np.asarray(response.gaps, dtype=float)
    history_row: Row = {
        "time": t,
        "top_pressure": _pressure_value(t, cfg),
        "upper_mean_z_displacement": float(np.mean(u_upper[:, 2])),
        "upper_bottom_mean_z": float(np.mean(x_upper[np.isclose(upper.X[:, 2], upper.X[:, 2].min()), 2])),
        "lower_top_mean_z_displacement": float(np.mean(u_lower[np.isclose(lower.X[:, 2], lower.X[:, 2].max()), 2])),
        "min_gap": float(response.min_gap),
        "max_penetration": float(response.max_penetration),
        "active_samples": int(response.active_count),
        "quadrature_points": int(contact_cache.area_weights.size),
        "active_fraction": float(response.active_count / max(contact_cache.area_weights.size, 1)),
        "normal_force": normal_force,
        "contact_energy": float(0.5 * cfg.pressure_stiffness * np.sum(contact_cache.area_weights * np.maximum(-gaps, 0.0) ** 2)),
        "field_path": "RequiredPointSDFWorkspace.build -> field_contact",
        "load_type": "top_pressure",
        "accepted_response_reused": "false",
    }
    timing_row: Row = {
        "time": t,
        "field_update_seconds": float(master_sdf.stats.update_seconds),
        "field_elapsed_seconds": float(build_elapsed),
        "field_query_seconds": float(query_elapsed),
        "field_query_seconds_per_sample": float(query_elapsed / max(contact_cache.area_weights.size, 1)),
        "quadrature_points": int(contact_cache.area_weights.size),
        "active_samples": int(response.active_count),
        "accepted_response_reused": "false",
    }
    return history_row, timing_row


def _contact_cell_fields(
    mesh: BoxMesh,
    driver_x: np.ndarray,
    driver_faces: np.ndarray,
    driver_ref: np.ndarray,
    displacement: np.ndarray,
    *,
    cfg: DynamicSurfaceConfig,
) -> tuple[dict[str, np.ndarray], Row]:
    top_faces = _top_triangles(mesh)
    cache = triangle_surface_quadrature_cache(driver_faces, driver_ref, order=cfg.quadrature_order)
    q_points = cache.points(driver_x)
    x_master = mesh.X + np.asarray(displacement, dtype=float)
    field_t0 = time.perf_counter()
    master_sdf = DynamicNarrowBandSDF.build_required_points(
        x_master,
        top_faces,
        q_points,
        spacing=cfg.spacing,
        band_radius=cfg.band_radius,
        padding=cfg.band_radius,
        cell_size=max(2.0 * cfg.spacing, cfg.band_radius),
        batch_projection_threshold=cfg.sdf_batch_projection_threshold,
    )
    field_elapsed = time.perf_counter() - field_t0
    response_t0 = time.perf_counter()
    response = surface_to_surface_field_penalty_response_vectorized(
        driver_x,
        driver_faces,
        master_sdf,
        pressure_stiffness=cfg.pressure_stiffness,
        n_total_dofs=3 * driver_x.shape[0] + 3 * mesh.X.shape[0],
        slave_x_reference=driver_ref,
        quadrature_cache=cache,
        slave_dof_offset=0,
        master_dof_offset=3 * driver_x.shape[0],
    )
    query_elapsed = time.perf_counter() - response_t0
    gaps = np.asarray(response.gaps, dtype=float)
    pressure = float(cfg.pressure_stiffness) * np.maximum(-gaps, 0.0)
    top_ids = _top_cell_ids_for_points(mesh, q_points)
    n_cells = mesh.elements.shape[0]
    gap_sum = np.zeros(n_cells, dtype=float)
    pressure_sum = np.zeros(n_cells, dtype=float)
    count = np.zeros(n_cells, dtype=float)
    active = np.zeros(n_cells, dtype=float)
    for eid, gap, p in zip(top_ids, gaps, pressure, strict=True):
        gap_sum[int(eid)] += float(gap)
        pressure_sum[int(eid)] += float(p)
        count[int(eid)] += 1.0
        active[int(eid)] = max(active[int(eid)], 1.0 if p > 0.0 else 0.0)
    denom = np.maximum(count, 1.0)
    fields = {
        "sdf_gap": gap_sum / denom,
        "contact_pressure": pressure_sum / denom,
        "active_contact": active,
    }
    timing = {
        "field_update_seconds": float(master_sdf.stats.update_seconds),
        "field_elapsed_seconds": float(field_elapsed),
        "field_query_seconds": float(query_elapsed),
        "quadrature_points": int(cache.area_weights.size),
        "active_samples": int(response.active_count),
        "normal_force": _normal_force_from_response(response, cfg.pressure_stiffness),
        "min_gap": float(response.min_gap),
        "max_penetration": float(response.max_penetration),
    }
    return fields, timing


def run_sfc_dynamic(
    cfg: DynamicSurfaceConfig,
    *,
    checkpoint_dir: Path | None = None,
    checkpoint_stride: int = 50,
) -> SFCDynamicResult:
    """Run the SFC dynamic surface-to-surface field-contact trajectory."""

    lower, driver_ref, driver_faces = _make_geometry(cfg)
    volume = VolumeMesh(lower.X, lower.elements, element_type="C3D8")
    body = DeformableBody(
        mesh=volume,
        material={"E": cfg.young_modulus, "nu": cfg.poisson_ratio},
        density=cfg.density,
    )
    K = assemble_stiffness_matrix(body).tocsr()
    M = assemble_mass_matrix(body, kind="consistent").tocsr()
    fixed = _fixed_bottom_dofs(lower)
    free = _free_dofs(body.n_dofs, fixed)
    top_faces = _top_triangles(lower)
    cache = triangle_surface_quadrature_cache(driver_faces, driver_ref, order=cfg.quadrature_order)
    n_slave_dofs = 3 * driver_ref.shape[0]
    sdf_workspace = RequiredPointSDFWorkspace.from_surface(
        lower.X,
        top_faces,
        spacing=cfg.spacing,
        band_radius=cfg.band_radius,
        padding=cfg.band_radius,
        cell_size=max(2.0 * cfg.spacing, cfg.band_radius),
        batch_projection_threshold=cfg.sdf_batch_projection_threshold,
        candidate_padding=cfg.sdf_candidate_padding,
    )

    beta = 0.25
    gamma = 0.5
    dt = float(cfg.dt)
    c0 = 1.0 / (beta * dt * dt)
    cv = gamma * dt * c0
    effective = (M * (c0 + cfg.damping_alpha * cv) + K).tocsc()
    effective_free = effective[free[:, None], free]
    solve_effective_free = factorized(effective_free) if free.size else None

    u = np.zeros(body.n_dofs, dtype=float)
    v = np.zeros(body.n_dofs, dtype=float)
    a = np.zeros(body.n_dofs, dtype=float)
    history: list[Row] = []
    timing: list[Row] = []
    frames: list[tuple[float, np.ndarray]] = []
    steps = int(round(cfg.total_time / cfg.dt))

    for step in range(steps + 1):
        t = float(step * dt)
        driver_x = _driver_positions(driver_ref, t, cfg)
        x_master = lower.X + u.reshape((-1, 3))
        q_points = cache.points(driver_x)
        build_t0 = time.perf_counter()
        master_sdf = sdf_workspace.build(
            x_master,
            top_faces,
            q_points,
        )
        build_elapsed = time.perf_counter() - build_t0
        query_t0 = time.perf_counter()
        response = surface_to_surface_field_penalty_response_vectorized(
            driver_x,
            driver_faces,
            master_sdf,
            pressure_stiffness=cfg.pressure_stiffness,
            n_total_dofs=n_slave_dofs + body.n_dofs,
            slave_x_reference=driver_ref,
            quadrature_cache=cache,
            slave_dof_offset=0,
            master_dof_offset=n_slave_dofs,
        )
        query_elapsed = time.perf_counter() - query_t0
        normal_force = _normal_force_from_response(response, cfg.pressure_stiffness)
        frames.append((t, u.reshape((-1, 3)).copy()))
        history.append(
            {
                "time": t,
                "driver_z_offset": _motion_z_offset(t, cfg),
                "top_mean_z_displacement": float(np.mean(u.reshape((-1, 3))[np.isclose(lower.X[:, 2], 4.0), 2])),
                "min_gap": float(response.min_gap),
                "max_penetration": float(response.max_penetration),
                "active_samples": int(response.active_count),
                "quadrature_points": int(cache.area_weights.size),
                "active_fraction": float(response.active_count / max(cache.area_weights.size, 1)),
                "normal_force": normal_force,
                "contact_energy": float(
                    0.5
                    * cfg.pressure_stiffness
                    * np.sum(cache.area_weights * np.maximum(-np.asarray(response.gaps, dtype=float), 0.0) ** 2)
                ),
                "field_path": "RequiredPointSDFWorkspace.build -> field_contact",
            }
        )
        timing.append(
            {
                "time": t,
                "field_update_seconds": float(master_sdf.stats.update_seconds),
                "field_elapsed_seconds": float(build_elapsed),
                "field_query_seconds": float(query_elapsed),
                "field_query_seconds_per_sample": float(query_elapsed / max(cache.area_weights.size, 1)),
                "quadrature_points": int(cache.area_weights.size),
                "active_samples": int(response.active_count),
            }
        )
        if checkpoint_dir is not None and (step % max(1, int(checkpoint_stride)) == 0 or step == steps):
            _write_csv(checkpoint_dir / "large_area_dynamic_sfc_history_partial.csv", history)
            _write_csv(checkpoint_dir / "large_area_dynamic_sfc_timing_partial.csv", timing)
        if step == steps:
            break

        u_pred = u + dt * v + dt * dt * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a
        u_guess = u_pred.copy()
        if fixed.size:
            u_guess[fixed] = 0.0
        next_t = float((step + 1) * dt)
        for _iteration in range(max(1, cfg.newmark_iterations)):
            driver_next = _driver_positions(driver_ref, next_t, cfg)
            q_next = cache.points(driver_next)
            x_guess = lower.X + u_guess.reshape((-1, 3))
            sdf_next = sdf_workspace.build(
                x_guess,
                top_faces,
                q_next,
            )
            response_next = surface_to_surface_field_penalty_response_vectorized(
                driver_next,
                driver_faces,
                sdf_next,
                pressure_stiffness=cfg.pressure_stiffness,
                n_total_dofs=n_slave_dofs + body.n_dofs,
                slave_x_reference=driver_ref,
                quadrature_cache=cache,
                slave_dof_offset=0,
                master_dof_offset=n_slave_dofs,
            )
            f_contact = response_next.force[n_slave_dofs:]
            a_guess = c0 * (u_guess - u_pred)
            v_guess = v_pred + gamma * dt * a_guess
            residual = M @ a_guess + cfg.damping_alpha * (M @ v_guess) + K @ u_guess - f_contact
            correction = np.zeros(body.n_dofs, dtype=float)
            if solve_effective_free is not None:
                correction[free] = np.asarray(solve_effective_free(-residual[free]), dtype=float)
            u_guess[free] += correction[free]
            if fixed.size:
                u_guess[fixed] = 0.0
            if float(np.linalg.norm(correction[free])) <= 1.0e-9 * max(1.0, float(np.linalg.norm(u_guess[free]))):
                break
        u = u_guess
        a = c0 * (u - u_pred)
        v = v_pred + gamma * dt * a
        if fixed.size:
            u[fixed] = 0.0
            v[fixed] = 0.0
            a[fixed] = 0.0

    return SFCDynamicResult(
        history=history,
        timing=timing,
        frames=frames,
        final_displacement=u.reshape((-1, 3)).copy(),
        driver_ref=driver_ref,
        driver_faces=driver_faces,
        lower_mesh=lower,
    )


def run_sfc_pressure_dynamic(
    cfg: DynamicSurfaceConfig,
    *,
    checkpoint_dir: Path | None = None,
    checkpoint_stride: int = 50,
) -> SFCDynamicResult:
    """Run a two-body pressure-driven C3D8 dynamic contact trajectory."""

    lower, upper = _make_pressure_geometry(cfg)
    lower_body = DeformableBody(
        mesh=VolumeMesh(lower.X, lower.elements, element_type="C3D8"),
        material={"E": cfg.young_modulus, "nu": cfg.poisson_ratio},
        density=cfg.density,
    )
    upper_body = DeformableBody(
        mesh=VolumeMesh(upper.X, upper.elements, element_type="C3D8"),
        material={"E": cfg.young_modulus, "nu": cfg.poisson_ratio},
        density=cfg.density,
    )
    K = block_diag((assemble_stiffness_matrix(upper_body), assemble_stiffness_matrix(lower_body)), format="csr")
    M = block_diag((assemble_mass_matrix(upper_body, kind="consistent"), assemble_mass_matrix(lower_body, kind="consistent")), format="csr")
    upper_dofs = upper_body.n_dofs
    lower_dofs = lower_body.n_dofs
    n_dofs = upper_dofs + lower_dofs

    upper_lateral_fixed = np.asarray(
        [3 * int(node) + comp for node in range(upper.X.shape[0]) for comp in (0, 1)],
        dtype=np.int64,
    )
    lower_fixed = upper_dofs + _fixed_bottom_dofs(lower)
    fixed = np.unique(np.concatenate([upper_lateral_fixed, lower_fixed]))
    free = _free_dofs(n_dofs, fixed)

    lower_top_faces = _top_triangles(lower)
    upper_bottom_faces = _bottom_triangles(upper)
    upper_top_faces = _top_triangles(upper)
    contact_cache = triangle_surface_quadrature_cache(upper_bottom_faces, upper.X, order=cfg.quadrature_order)
    unit_pressure_force_upper = _surface_pressure_force(
        upper.X,
        upper_top_faces,
        pressure=1.0,
        node_count=upper.X.shape[0],
        quadrature_order=cfg.quadrature_order,
    )
    sdf_workspace = RequiredPointSDFWorkspace.from_surface(
        lower.X,
        lower_top_faces,
        spacing=cfg.spacing,
        band_radius=cfg.band_radius,
        padding=cfg.band_radius,
        cell_size=max(2.0 * cfg.spacing, cfg.band_radius),
        batch_projection_threshold=cfg.sdf_batch_projection_threshold,
        candidate_padding=cfg.sdf_candidate_padding,
    )

    beta = 0.25
    gamma = 0.5
    dt = float(cfg.dt)
    c0 = 1.0 / (beta * dt * dt)
    cv = gamma * dt * c0
    effective = (M * (c0 + cfg.damping_alpha * cv) + K).tocsc()
    upper_free = free[free < upper_dofs]
    lower_free = free[free >= upper_dofs]
    upper_solve = factorized(effective[upper_free[:, None], upper_free]) if upper_free.size else None
    lower_solve = factorized(effective[lower_free[:, None], lower_free]) if lower_free.size else None
    effective_free = effective[free[:, None], free].tocsr() if free.size else None
    solve_effective_free = factorized(effective_free.tocsc()) if free.size and cfg.use_matrix_free_contact_tangent else None

    u = np.zeros(n_dofs, dtype=float)
    v = np.zeros(n_dofs, dtype=float)
    a = np.zeros(n_dofs, dtype=float)
    history: list[Row] = []
    timing: list[Row] = []
    lower_frames: list[tuple[float, np.ndarray]] = []
    upper_frames: list[tuple[float, np.ndarray]] = []
    steps = int(round(cfg.total_time / cfg.dt))
    core_solve_seconds = 0.0
    diagnostics_seconds = 0.0
    cpp_tangent_iterations = 0
    cpp_tangent_solves = 0
    accepted_response_cache: tuple[float, Any, Any, float, float] | None = None
    solve_loop_start = time.perf_counter()

    for step in range(steps + 1):
        t = float(step * dt)
        pressure = _pressure_value(t, cfg)
        u_upper = u[:upper_dofs].reshape((-1, 3))
        u_lower = u[upper_dofs:].reshape((-1, 3))
        x_upper = upper.X + u_upper
        x_lower = lower.X + u_lower
        lower_frames.append((t, u_lower.copy()))
        upper_frames.append((t, u_upper.copy()))
        if not cfg.defer_diagnostics:
            reused_accepted_response = False
            if accepted_response_cache is not None and abs(float(accepted_response_cache[0]) - t) <= 1.0e-12:
                _cache_time, master_sdf, response, build_elapsed, query_elapsed = accepted_response_cache
                accepted_response_cache = None
                reused_accepted_response = True
            else:
                diagnostics_t0 = time.perf_counter()
                q_points = contact_cache.points(x_upper)
                build_t0 = time.perf_counter()
                master_sdf = sdf_workspace.build(
                    x_lower,
                    lower_top_faces,
                    q_points,
                )
                build_elapsed = time.perf_counter() - build_t0
                query_t0 = time.perf_counter()
                response = surface_to_surface_field_penalty_response_vectorized(
                    x_upper,
                    upper_bottom_faces,
                    master_sdf,
                    pressure_stiffness=cfg.pressure_stiffness,
                    n_total_dofs=n_dofs,
                    slave_x_reference=upper.X,
                    quadrature_cache=contact_cache,
                    slave_dof_offset=0,
                    master_dof_offset=upper_dofs,
                )
                query_elapsed = time.perf_counter() - query_t0
                diagnostics_seconds += time.perf_counter() - diagnostics_t0
            normal_force = _normal_force_from_response(response, cfg.pressure_stiffness)
            history.append(
                {
                    "time": t,
                    "top_pressure": pressure,
                    "upper_mean_z_displacement": float(np.mean(u_upper[:, 2])),
                    "upper_bottom_mean_z": float(np.mean(x_upper[np.isclose(upper.X[:, 2], upper.X[:, 2].min()), 2])),
                    "lower_top_mean_z_displacement": float(np.mean(u_lower[np.isclose(lower.X[:, 2], lower.X[:, 2].max()), 2])),
                    "min_gap": float(response.min_gap),
                    "max_penetration": float(response.max_penetration),
                    "active_samples": int(response.active_count),
                    "quadrature_points": int(contact_cache.area_weights.size),
                    "active_fraction": float(response.active_count / max(contact_cache.area_weights.size, 1)),
                    "normal_force": normal_force,
                    "contact_energy": float(
                        0.5
                        * cfg.pressure_stiffness
                        * np.sum(contact_cache.area_weights * np.maximum(-np.asarray(response.gaps, dtype=float), 0.0) ** 2)
                    ),
                    "field_path": "RequiredPointSDFWorkspace.build -> field_contact",
                    "load_type": "top_pressure",
                    "accepted_response_reused": str(reused_accepted_response).lower(),
                }
            )
            timing.append(
                {
                    "time": t,
                    "field_update_seconds": float(master_sdf.stats.update_seconds),
                    "field_elapsed_seconds": float(build_elapsed),
                    "field_query_seconds": float(query_elapsed),
                    "field_query_seconds_per_sample": float(query_elapsed / max(contact_cache.area_weights.size, 1)),
                    "quadrature_points": int(contact_cache.area_weights.size),
                    "active_samples": int(response.active_count),
                    "accepted_response_reused": str(reused_accepted_response).lower(),
                }
            )
        if checkpoint_dir is not None and (step % max(1, int(checkpoint_stride)) == 0 or step == steps):
            if history:
                _write_csv(checkpoint_dir / "large_area_dynamic_sfc_history_partial.csv", history)
            if timing:
                _write_csv(checkpoint_dir / "large_area_dynamic_sfc_timing_partial.csv", timing)
        if step == steps:
            break

        u_pred = u + dt * v + dt * dt * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a
        u_guess = u_pred.copy()
        if fixed.size:
            u_guess[fixed] = 0.0
        next_t = float((step + 1) * dt)
        for _iteration in range(max(1, cfg.newmark_iterations)):
            core_t0 = time.perf_counter()
            pressure_next = _pressure_value(next_t, cfg)
            u_upper_guess = u_guess[:upper_dofs].reshape((-1, 3))
            u_lower_guess = u_guess[upper_dofs:].reshape((-1, 3))
            x_upper_guess = upper.X + u_upper_guess
            x_lower_guess = lower.X + u_lower_guess
            q_next = contact_cache.points(x_upper_guess)
            sdf_next = sdf_workspace.build(
                x_lower_guess,
                lower_top_faces,
                q_next,
            )
            response_next = surface_to_surface_field_penalty_response_vectorized(
                x_upper_guess,
                upper_bottom_faces,
                sdf_next,
                pressure_stiffness=cfg.pressure_stiffness,
                n_total_dofs=n_dofs,
                slave_x_reference=upper.X,
                quadrature_cache=contact_cache,
                slave_dof_offset=0,
                master_dof_offset=upper_dofs,
                assemble_stiffness=cfg.use_contact_tangent,
                matrix_free_stiffness=cfg.use_matrix_free_contact_tangent or cfg.use_cpp_contact_tangent_solver,
            )
            f_ext = np.zeros(n_dofs, dtype=float)
            f_ext[:upper_dofs] = pressure_next * unit_pressure_force_upper
            a_guess = c0 * (u_guess - u_pred)
            v_guess = v_pred + gamma * dt * a_guess
            residual = M @ a_guess + cfg.damping_alpha * (M @ v_guess) + K @ u_guess - f_ext - response_next.force
            correction = np.zeros(n_dofs, dtype=float)
            if cfg.use_cpp_contact_tangent_solver:
                if response_next.stiffness_operator is None or effective_free is None:
                    raise RuntimeError("C++ contact tangent solver was requested but no operator is available")
                try:
                    from sfc.contact._cpp_field_contact import solve_contact_tangent_pcg
                except Exception as exc:
                    raise RuntimeError("C++ contact tangent solver is not available") from exc
                op = response_next.stiffness_operator
                solution, info, iterations, residual_norm = solve_contact_tangent_pcg(
                    effective_free.indptr,
                    effective_free.indices,
                    effective_free.data,
                    -residual[free],
                    free,
                    op.scale,
                    op.slave_node_ids,
                    op.slave_weights,
                    op.gradients,
                    op.face_node_ids,
                    op.grid_weights,
                    op.barycentric,
                    op.normals,
                    n_dofs,
                    op.slave_dof_offset,
                    op.master_dof_offset,
                    cfg.contact_tangent_cg_rtol,
                    cfg.contact_tangent_cg_atol,
                    cfg.contact_tangent_cg_maxiter,
                    preconditioner=cfg.cpp_contact_tangent_preconditioner,
                )
                if info != 0:
                    raise RuntimeError(
                        f"C++ contact tangent PCG did not converge, info={info}, "
                        f"iterations={iterations}, residual={residual_norm:.6e}"
                    )
                cpp_tangent_iterations += int(iterations)
                cpp_tangent_solves += 1
                correction[free] = solution
            elif cfg.use_matrix_free_contact_tangent:
                if response_next.stiffness_operator is None or effective_free is None or solve_effective_free is None:
                    raise RuntimeError("matrix-free contact tangent was requested but no operator is available")
                contact_free = response_next.stiffness_operator.as_linear_operator(free)

                def tangent_matvec(value: np.ndarray) -> np.ndarray:
                    return np.asarray(effective_free @ value, dtype=float) + np.asarray(contact_free @ value, dtype=float)

                tangent_operator = LinearOperator((free.size, free.size), matvec=tangent_matvec, dtype=float)
                preconditioner = LinearOperator(
                    (free.size, free.size),
                    matvec=lambda value: np.asarray(solve_effective_free(value), dtype=float),
                    dtype=float,
                )
                cg_iterations = 0

                def count_cg_iteration(_value: np.ndarray) -> None:
                    nonlocal cg_iterations
                    cg_iterations += 1

                solution, info = cg(
                    tangent_operator,
                    -residual[free],
                    M=preconditioner,
                    rtol=float(cfg.contact_tangent_cg_rtol),
                    atol=float(cfg.contact_tangent_cg_atol),
                    maxiter=int(cfg.contact_tangent_cg_maxiter),
                    callback=count_cg_iteration,
                )
                if info != 0:
                    raise RuntimeError(f"matrix-free contact tangent CG did not converge, info={info}")
                correction[free] = np.asarray(solution, dtype=float)
            elif cfg.use_contact_tangent:
                tangent = (effective + response_next.stiffness).tocsc()
                tangent_free = tangent[free[:, None], free]
                correction[free] = np.asarray(spsolve(tangent_free, -residual[free]), dtype=float)
            else:
                if upper_solve is not None:
                    correction[upper_free] = np.asarray(upper_solve(-residual[upper_free]), dtype=float)
                if lower_solve is not None:
                    correction[lower_free] = np.asarray(lower_solve(-residual[lower_free]), dtype=float)
            u_guess[free] += correction[free]
            if fixed.size:
                u_guess[fixed] = 0.0
            core_solve_seconds += time.perf_counter() - core_t0
            if float(np.linalg.norm(correction[free])) <= 1.0e-9 * max(1.0, float(np.linalg.norm(u_guess[free]))):
                break
        u = u_guess
        a = c0 * (u - u_pred)
        v = v_pred + gamma * dt * a
        if fixed.size:
            u[fixed] = 0.0
            v[fixed] = 0.0
            a[fixed] = 0.0
        if step + 1 <= steps and not cfg.defer_diagnostics:
            diagnostics_t0 = time.perf_counter()
            u_upper_final = u[:upper_dofs].reshape((-1, 3))
            u_lower_final = u[upper_dofs:].reshape((-1, 3))
            x_upper_final = upper.X + u_upper_final
            x_lower_final = lower.X + u_lower_final
            q_final = contact_cache.points(x_upper_final)
            build_t0 = time.perf_counter()
            sdf_final = sdf_workspace.build(
                x_lower_final,
                lower_top_faces,
                q_final,
            )
            build_elapsed_final = time.perf_counter() - build_t0
            query_t0 = time.perf_counter()
            response_final = surface_to_surface_field_penalty_response_vectorized(
                x_upper_final,
                upper_bottom_faces,
                sdf_final,
                pressure_stiffness=cfg.pressure_stiffness,
                n_total_dofs=n_dofs,
                slave_x_reference=upper.X,
                quadrature_cache=contact_cache,
                slave_dof_offset=0,
                master_dof_offset=upper_dofs,
            )
            query_elapsed_final = time.perf_counter() - query_t0
            diagnostics_seconds += time.perf_counter() - diagnostics_t0
            accepted_response_cache = (
                float(next_t),
                sdf_final,
                response_final,
                float(build_elapsed_final),
                float(query_elapsed_final),
            )

    solve_wall_seconds = time.perf_counter() - solve_loop_start
    if cfg.defer_diagnostics:
        diagnostics_t0 = time.perf_counter()
        for (time_value, lower_u), (_upper_time, upper_u) in zip(lower_frames, upper_frames, strict=True):
            history_row, timing_row = _pressure_contact_diagnostics_row(
                time_value=float(time_value),
                upper=upper,
                lower=lower,
                upper_displacement=upper_u,
                lower_displacement=lower_u,
                upper_bottom_faces=upper_bottom_faces,
                lower_top_faces=lower_top_faces,
                contact_cache=contact_cache,
                sdf_workspace=sdf_workspace,
                cfg=cfg,
            )
            history_row["accepted_response_reused"] = "postprocess"
            timing_row["accepted_response_reused"] = "postprocess"
            history.append(history_row)
            timing.append(timing_row)
        diagnostics_seconds += time.perf_counter() - diagnostics_t0

    return SFCDynamicResult(
        history=history,
        timing=timing,
        frames=lower_frames,
        final_displacement=u[upper_dofs:].reshape((-1, 3)).copy(),
        driver_ref=upper.X,
        driver_faces=upper_bottom_faces,
        lower_mesh=lower,
        upper_mesh=upper,
        upper_frames=upper_frames,
        upper_bottom_faces=upper_bottom_faces,
        solve_wall_seconds=float(solve_wall_seconds),
        core_solve_seconds=float(core_solve_seconds),
        diagnostics_seconds=float(diagnostics_seconds),
        cpp_tangent_iterations=int(cpp_tangent_iterations),
        cpp_tangent_solves=int(cpp_tangent_solves),
    )


def _surface_refs_top(mesh: BoxMesh) -> list[tuple[int, str]]:
    refs: list[tuple[int, str]] = []
    for j in range(mesh.ny):
        for i in range(mesh.nx):
            eid = (mesh.nz - 1) * mesh.ny * mesh.nx + j * mesh.nx + i
            refs.append((eid, "S2"))
    return refs


def _surface_refs_bottom(mesh: BoxMesh) -> list[tuple[int, str]]:
    refs: list[tuple[int, str]] = []
    for j in range(mesh.ny):
        for i in range(mesh.nx):
            eid = j * mesh.nx + i
            refs.append((eid, "S1"))
    return refs


def write_calculix_input(cfg: DynamicSurfaceConfig, path: Path) -> tuple[np.ndarray, np.ndarray, BoxMesh, BoxMesh]:
    """Write an optional native CalculiX contact model for the same pressure load."""

    lower, driver = _make_pressure_geometry(cfg)
    lower_node_ids = np.arange(1, lower.X.shape[0] + 1, dtype=np.int64)
    driver_node_ids = np.arange(lower_node_ids[-1] + 1, lower_node_ids[-1] + 1 + driver.X.shape[0], dtype=np.int64)
    lower_element_ids = np.arange(1, lower.elements.shape[0] + 1, dtype=np.int64)
    driver_element_ids = np.arange(
        lower_element_ids[-1] + 1,
        lower_element_ids[-1] + 1 + driver.elements.shape[0],
        dtype=np.int64,
    )
    bottom_nodes = np.flatnonzero(np.isclose(lower.X[:, 2], 0.0))
    lower_top_nodes = np.flatnonzero(np.isclose(lower.X[:, 2], 4.0))

    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "** Generated by validation/run_large_area_dynamic_surface_contact.py",
        "*HEADING",
        "Large-area pressure-driven dynamic surface-to-surface contact",
        "*NODE, NSET=NALL",
    ]
    for node_id, xyz in zip(lower_node_ids, lower.X, strict=True):
        lines.append(f"{int(node_id)}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    for node_id, xyz in zip(driver_node_ids, driver.X, strict=True):
        lines.append(f"{int(node_id)}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=ELOWER")
    for element_id, element in zip(lower_element_ids, lower.elements, strict=True):
        conn = ", ".join(str(int(lower_node_ids[idx])) for idx in element)
        lines.append(f"{int(element_id)}, {conn}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=EDRIVER")
    for element_id, element in zip(driver_element_ids, driver.elements, strict=True):
        conn = ", ".join(str(int(driver_node_ids[idx])) for idx in element)
        lines.append(f"{int(element_id)}, {conn}")
    lines.extend(["*ELSET, ELSET=ELALL", "ELOWER, EDRIVER"])
    lines.append("*NSET, NSET=NLOWER")
    _append_id_list(lines, lower_node_ids)
    lines.append("*NSET, NSET=NLOWER_TOP")
    _append_id_list(lines, lower_node_ids[lower_top_nodes])
    lines.append("*NSET, NSET=NFIXED")
    _append_id_list(lines, lower_node_ids[bottom_nodes])
    lines.append("*NSET, NSET=NDRIVER")
    _append_id_list(lines, driver_node_ids)
    ccx_dt = min(float(cfg.dt), 1.0e-4)
    ccx_output_frequency = max(1, int(round(float(cfg.dt) / ccx_dt)))
    lines.extend(
        [
            "*MATERIAL, NAME=MAT",
            "*ELASTIC",
            f"{cfg.young_modulus:.12e}, {cfg.poisson_ratio:.12e}",
            "*DENSITY",
            f"{cfg.density:.12e}",
            "*SOLID SECTION, ELSET=ELOWER, MATERIAL=MAT",
            "*SOLID SECTION, ELSET=EDRIVER, MATERIAL=MAT",
            "*SURFACE, NAME=SSLAVE, TYPE=ELEMENT",
        ]
    )
    for eid, face in _surface_refs_bottom(driver):
        lines.append(f"{int(driver_element_ids[eid])}, {face}")
    lines.append("*SURFACE, NAME=SMASTER, TYPE=ELEMENT")
    for eid, face in _surface_refs_top(lower):
        lines.append(f"{int(lower_element_ids[eid])}, {face}")
    lines.extend(
        [
            "*SURFACE INTERACTION, NAME=CONTACT",
            "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
            f"{cfg.pressure_stiffness:.12e}",
            "*CONTACT PAIR, INTERACTION=CONTACT, TYPE=SURFACE TO SURFACE",
            "SSLAVE, SMASTER",
            "*AMPLITUDE, NAME=TOPPRESS, TIME=TOTAL TIME",
        ]
    )
    amp_pairs: list[float] = []
    for step in range(int(round(cfg.total_time / cfg.dt)) + 1):
        t = float(step * cfg.dt)
        amp_pairs.extend([t, _pressure_value(t, cfg)])
    for start in range(0, len(amp_pairs), 8):
        lines.append(", ".join(f"{value:.12e}" for value in amp_pairs[start : start + 8]))
    unit_top_force = _surface_pressure_force(
        driver.X,
        _top_triangles(driver),
        pressure=1.0,
        node_count=driver.X.shape[0],
        quadrature_order=cfg.quadrature_order,
    )
    lines.extend(
        [
            "*BOUNDARY",
            "NFIXED, 1, 3, 0.0",
            "NDRIVER, 1, 2, 0.0",
            "*STEP, NLGEOM, INC=2000",
            "*DYNAMIC",
            f"{ccx_dt:.12e}, {cfg.total_time:.12e}, {0.1 * ccx_dt:.12e}, {ccx_dt:.12e}",
            "*CLOAD, AMPLITUDE=TOPPRESS",
        ]
    )
    for local_id, node_id in enumerate(driver_node_ids):
        fz = float(unit_top_force[3 * local_id + 2])
        if abs(fz) > 0.0:
            lines.append(f"{int(node_id)}, 3, {fz:.12e}")
    lines.extend(
        [
            f"*NODE PRINT, NSET=NALL, FREQUENCY={ccx_output_frequency}",
            "U",
            f"*CONTACT PRINT, TOTALS=ONLY, FREQUENCY={ccx_output_frequency}",
            "CELS,CNUM",
            f"*EL PRINT, ELSET=ELOWER, FREQUENCY={ccx_output_frequency}",
            "S,E",
            "*END STEP",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lower_node_ids, driver_node_ids, lower, driver


def _calculix_version_safe() -> str:
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", "ccx -v 2>&1"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = [line.strip() for line in (proc.stdout + proc.stderr).splitlines() if line.strip()]
    return lines[0] if proc.returncode == 0 and lines else "CalculiX"


def run_calculix_reference(cfg: DynamicSurfaceConfig, out_dir: Path, *, timeout_seconds: int) -> tuple[list[Row], Row]:
    """Run the optional CalculiX native-contact reference."""

    availability = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", "command -v ccx >/dev/null"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if availability.returncode != 0:
        raise RuntimeError("CalculiX/ccx is unavailable through WSL")
    run_dir = out_dir / "calculix_runs" / "large_area_dynamic_surface_contact"
    run_dir.mkdir(parents=True, exist_ok=True)
    job = "large_area_dynamic_surface_contact"
    inp = run_dir / f"{job}.inp"
    lower_node_ids, driver_node_ids, lower, driver = write_calculix_input(cfg, inp)
    wsl_run_dir = _wsl_path(run_dir)
    command = f"cd {wsl_run_dir} && ccx {job}"
    start = time.perf_counter()
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    wall = time.perf_counter() - start
    (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
    dat = run_dir / f"{job}.dat"
    status = {
        "external_solver": "CalculiX",
        "external_solver_version": _calculix_version_safe(),
        "command": f"wsl --exec bash -lc \"{command}\"",
        "input_file": str(inp.relative_to(out_dir)),
        "dat_file": str(dat.relative_to(out_dir)) if dat.exists() else "",
        "stdout_log": str((run_dir / "calculix_stdout.log").relative_to(out_dir)),
        "stderr_log": str((run_dir / "calculix_stderr.log").relative_to(out_dir)),
        "return_code": int(proc.returncode),
        "completed": str(proc.returncode == 0 and dat.exists() and dat.stat().st_size > 0).lower(),
        "partial_reference": str(proc.returncode != 0 and dat.exists() and dat.stat().st_size > 0).lower(),
        "wall_time_seconds": wall,
    }
    if not dat.exists() or dat.stat().st_size == 0:
        raise RuntimeError(f"CalculiX did not produce a usable {dat}")
    lower_blocks = _parse_nodal_vectors(dat, lower_node_ids, quantity="u")
    upper_blocks = _parse_nodal_vectors(dat, driver_node_ids, quantity="u")
    totals = _parse_totals(dat)
    top_mask = np.isclose(lower.X[:, 2], 4.0)
    rows: list[Row] = []
    for t, U in lower_blocks.items():
        U_upper = upper_blocks.get(float(t), np.zeros((driver_node_ids.size, 3), dtype=float))
        total = totals.get(float(t), {})
        rows.append(
            {
                "time": float(t),
                "lower_top_mean_z_displacement": float(np.mean(U[top_mask, 2])),
                "upper_mean_z_displacement": float(np.mean(U_upper[:, 2])),
                "upper_bottom_mean_z": float(np.mean(driver.X[:, 2] + U_upper[:, 2])),
                "normal_force": float(total.get("normal_force_calculix", np.nan)),
                "contact_energy": float(total.get("contact_energy_calculix", np.nan)),
                "contact_count": int(total.get("contact_count_calculix", 0)),
            }
        )
    return rows, status


def _write_vtk_outputs(out_dir: Path, result: SFCDynamicResult, cfg: DynamicSurfaceConfig) -> dict[str, Path | int]:
    vtk_dir = out_dir / "vtk" / "sfc"
    upper_by_time = {
        round(float(time_value), 12): np.asarray(displacement, dtype=float)
        for time_value, displacement in (result.upper_frames or [])
    }

    def cell_fields(time_value: float, U: np.ndarray) -> dict[str, np.ndarray]:
        fields = _element_fields(result.lower_mesh, U.reshape(-1), E=cfg.young_modulus, nu=cfg.poisson_ratio)
        if result.upper_mesh is not None:
            upper_u = upper_by_time.get(round(float(time_value), 12))
            if upper_u is None:
                driver_x = result.driver_ref.copy()
            else:
                driver_x = result.upper_mesh.X + upper_u
            driver_ref = result.upper_mesh.X
            driver_faces = result.upper_bottom_faces if result.upper_bottom_faces is not None else result.driver_faces
        else:
            driver_x = _driver_positions(result.driver_ref, time_value, cfg)
            driver_ref = result.driver_ref
            driver_faces = result.driver_faces
        contact_fields, _contact_row = _contact_cell_fields(
            result.lower_mesh,
            driver_x,
            driver_faces,
            driver_ref,
            U,
            cfg=cfg,
        )
        fields.update(contact_fields)
        return fields

    lower_outputs = write_hex_frame_series(
        vtk_dir,
        stem="sfc_large_area_dynamic",
        reference_points=result.lower_mesh.X,
        elements=result.lower_mesh.elements,
        frames=result.frames,
        frame_stride=cfg.frame_stride,
        cell_scalar_fn=cell_fields,
        source_label="SFC large-area dynamic surface contact",
    )
    if result.upper_mesh is not None and result.upper_frames:
        upper_outputs = write_hex_frame_series(
            vtk_dir,
            stem="sfc_large_area_dynamic_upper",
            reference_points=result.upper_mesh.X,
            elements=result.upper_mesh.elements,
            frames=result.upper_frames,
            frame_stride=cfg.frame_stride,
            cell_scalar_fn=lambda U: _element_fields(result.upper_mesh, U.reshape(-1), E=cfg.young_modulus, nu=cfg.poisson_ratio),
            source_label="SFC large-area dynamic upper body",
        )
        lower_outputs["upper_pvd"] = upper_outputs["pvd"]
        lower_outputs["upper_frame_count"] = upper_outputs["frame_count"]
    return lower_outputs


def _plot_histories(out_dir: Path, sfc_rows: list[Row], calculix_rows: list[Row] | None = None) -> dict[str, Path]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    t = np.asarray([row["time"] for row in sfc_rows], dtype=float)
    active = np.asarray([row["active_samples"] for row in sfc_rows], dtype=float)
    force = np.asarray([row["normal_force"] for row in sfc_rows], dtype=float)
    z_key = "upper_mean_z_displacement" if "upper_mean_z_displacement" in sfc_rows[0] else "top_mean_z_displacement"
    z = np.asarray([row[z_key] for row in sfc_rows], dtype=float)
    gap = np.asarray([row["min_gap"] for row in sfc_rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), constrained_layout=True)
    axes[0, 0].plot(t, z, "-o", ms=3.0, lw=1.2, label="SFC")
    if calculix_rows:
        tc = np.asarray([row["time"] for row in calculix_rows], dtype=float)
        calc_z_key = "upper_mean_z_displacement" if "upper_mean_z_displacement" in calculix_rows[0] else "top_mean_z_displacement"
        zc = np.asarray([row[calc_z_key] for row in calculix_rows], dtype=float)
        axes[0, 0].plot(tc, zc, "--", lw=1.2, label="CalculiX")
        if tc.size and zc.size:
            z_interp = np.interp(t, tc, zc)
            denom = max(float(np.max(np.abs(z_interp))), 1.0e-12)
            err = float(np.max(np.abs(z - z_interp)) / denom)
            axes[0, 0].legend(title=f"max rel. err. {err:.2e}", frameon=False, fontsize=8)
        else:
            axes[0, 0].legend(frameon=False, fontsize=8)
    else:
        axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("upper z displacement")

    axes[0, 1].plot(t, force, "-o", ms=3.0, lw=1.2)
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("normal force")

    axes[1, 0].plot(t, active, "-o", ms=3.0, lw=1.2)
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("active quadrature samples")

    axes[1, 1].plot(t, gap, "-o", ms=3.0, lw=1.2)
    axes[1, 1].axhline(0.0, color="0.35", lw=0.8)
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("minimum field gap")

    for ax in axes.ravel():
        ax.grid(True, color="0.88", lw=0.5)
    png = figures / "large_area_dynamic_histories.png"
    pdf = figures / "large_area_dynamic_histories.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def _compare_histories(sfc_rows: list[Row], calculix_rows: list[Row]) -> list[Row]:
    if not calculix_rows:
        return []
    tc = np.asarray([row["time"] for row in calculix_rows], dtype=float)
    z_key = "upper_mean_z_displacement" if "upper_mean_z_displacement" in sfc_rows[0] else "top_mean_z_displacement"
    calc_z_key = "upper_mean_z_displacement" if "upper_mean_z_displacement" in calculix_rows[0] else "top_mean_z_displacement"
    zc = np.asarray([row[calc_z_key] for row in calculix_rows], dtype=float)
    fc = np.asarray([row["normal_force"] for row in calculix_rows], dtype=float)
    rows: list[Row] = []
    for row in sfc_rows:
        t = float(row["time"])
        z_ref = float(np.interp(t, tc, zc)) if tc.size else float("nan")
        z_sfc = float(row[z_key])
        force_ref = float(np.interp(t, tc, fc)) if tc.size and np.any(np.isfinite(fc)) else float("nan")
        force_sfc = float(row["normal_force"])
        force_abs = abs(force_sfc - force_ref) if math.isfinite(force_ref) else float("nan")
        rows.append(
            {
                "time": t,
                f"sfc_{z_key}": z_sfc,
                f"calculix_{calc_z_key}": z_ref,
                "abs_displacement_error": abs(z_sfc - z_ref),
                "normal_force_sfc": force_sfc,
                "normal_force_calculix": force_ref,
                "abs_normal_force_error": force_abs,
                "rel_normal_force_error": force_abs / max(abs(force_ref), 1.0e-12) if math.isfinite(force_abs) else float("nan"),
            }
        )
    return rows


def _summary_text(
    cfg: DynamicSurfaceConfig,
    sfc: SFCDynamicResult,
    calc_status: Row | None,
    comparison_rows: list[Row],
    vtk_outputs: dict[str, Path | int],
    *,
    sfc_wall_time_seconds: float,
) -> str:
    max_active = max(int(row["active_samples"]) for row in sfc.history)
    max_quad = max(int(row["quadrature_points"]) for row in sfc.history)
    avg_update = float(np.mean([row["field_update_seconds"] for row in sfc.timing]))
    avg_query = float(np.mean([row["field_query_seconds"] for row in sfc.timing]))
    lines = [
        "# Large-area dynamic surface-to-surface SDF contact",
        "",
        "This is scheme B: pressure-driven two-body dynamics with sustained large-area contact.",
        "",
        "## Configuration",
        "",
        f"- Lower C3D8 mesh: `{cfg.nx} x {cfg.ny} x {cfg.nz}` elements.",
        f"- Upper C3D8 mesh: `{cfg.driver_nx} x {cfg.driver_ny} x {max(1, min(cfg.nz, 2))}` elements.",
        f"- Time window: `{cfg.total_time}` with `dt={cfg.dt}`.",
        f"- Load: top pressure peaks at `{cfg.peak_pressure}`, initial gap `{cfg.initial_gap}`, pressure oscillation amplitude `{cfg.oscillation_amplitude}`.",
        f"- Contact quadrature points per step: `{max_quad}`.",
        f"- Maximum active quadrature samples: `{max_active}`.",
        "",
        "## SFC path",
        "",
        "- Upper body: internally assembled C3D8 dynamics with equivalent top-pressure nodal loads.",
        "- Current lower master surface -> `RequiredPointSDFWorkspace.build(...)`.",
        "- Query/integration -> `surface_to_surface_field_penalty_response_vectorized(...)`.",
        "- Projection is used only inside field construction; field contact queries use interpolation.",
        f"- Contact tangent in SFC iterations: `{bool(cfg.use_contact_tangent)}`.",
        f"- Matrix-free contact tangent in SFC iterations: `{bool(cfg.use_matrix_free_contact_tangent)}`.",
        f"- C++ contact tangent PCG solver: `{bool(cfg.use_cpp_contact_tangent_solver)}`.",
        f"- C++ contact tangent preconditioner: `{cfg.cpp_contact_tangent_preconditioner}`.",
        f"- Deferred diagnostics/postprocess: `{bool(cfg.defer_diagnostics)}`.",
        "- Accepted-state response reuse: response is reused only after exact final-state reevaluation.",
        "",
        "## Timing",
        "",
        f"- Complete command wall time including output/postprocess: `{float(sfc_wall_time_seconds):.6e}` s.",
        f"- SFC solve-loop wall time excluding deferred diagnostics: `{float(sfc.solve_wall_seconds):.6e}` s.",
        f"- Inner correction solve time: `{float(sfc.core_solve_seconds):.6e}` s.",
        f"- Diagnostics/accepted-response time inside SFC trajectory: `{float(sfc.diagnostics_seconds):.6e}` s.",
        f"- Mean field update time, backend breakdown only: `{avg_update:.6e}` s/step.",
        f"- Mean field query/contact assembly time, backend breakdown only: `{avg_query:.6e}` s/step.",
        f"- C++ tangent PCG solves: `{int(sfc.cpp_tangent_solves)}`.",
        f"- C++ tangent PCG iterations: `{int(sfc.cpp_tangent_iterations)}`.",
        "",
        "## VTK outputs",
        "",
        f"- SFC PVD: `{vtk_outputs['pvd']}`.",
        f"- SFC frame count: `{vtk_outputs['frame_count']}`.",
        "",
    ]
    if calc_status is None:
        lines.extend(
            [
                "## CalculiX reference",
                "",
                "- Not run in this invocation.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## CalculiX reference",
                "",
                f"- Completed: `{calc_status.get('completed')}`.",
                f"- Wall time: `{float(calc_status.get('wall_time_seconds', float('nan'))):.6e}` s.",
                f"- Input: `{calc_status.get('input_file', '')}`.",
                "",
            ]
        )
        if comparison_rows:
            max_abs = max(float(row["abs_displacement_error"]) for row in comparison_rows)
            lines.append(f"- Max upper-displacement absolute error on common SFC times: `{max_abs:.6e}`.")
            finite_force = [
                float(row["rel_normal_force_error"])
                for row in comparison_rows
                if math.isfinite(float(row.get("rel_normal_force_error", float("nan"))))
            ]
            if finite_force:
                lines.append(f"- Max normal-force relative error on common SFC times: `{max(finite_force):.6e}`.")
            lines.append("")
    return "\n".join(lines) + "\n"


def default_config(*, quick: bool) -> DynamicSurfaceConfig:
    if quick:
        return DynamicSurfaceConfig(
            nx=10,
            ny=10,
            nz=2,
            driver_nx=10,
            driver_ny=10,
            total_time=0.08,
            dt=0.01,
            ramp_time=0.04,
            initial_gap=0.02,
            closure=0.12,
            oscillation_amplitude=0.01,
            oscillation_frequency=12.0,
            spacing=0.35,
            band_radius=0.7,
            quadrature_order=7,
            pressure_stiffness=5.0e3,
            peak_pressure=4.0e2,
            young_modulus=2.0e5,
            poisson_ratio=0.30,
            density=1.0,
            damping_alpha=2.5,
            newmark_iterations=2,
            frame_stride=1,
            sdf_batch_projection_threshold=250_000,
            sdf_candidate_padding=None,
            use_contact_tangent=False,
            use_matrix_free_contact_tangent=False,
            use_cpp_contact_tangent_solver=False,
            cpp_contact_tangent_preconditioner="block-sgs",
            contact_tangent_cg_rtol=1.0e-10,
            contact_tangent_cg_atol=1.0e-12,
            contact_tangent_cg_maxiter=80,
            defer_diagnostics=False,
        )
    return DynamicSurfaceConfig(
        nx=36,
        ny=36,
        nz=6,
        driver_nx=36,
        driver_ny=36,
        total_time=0.50,
        dt=0.005,
        ramp_time=0.12,
        initial_gap=0.02,
        closure=0.14,
        oscillation_amplitude=0.012,
        oscillation_frequency=8.0,
        spacing=0.18,
        band_radius=0.45,
        quadrature_order=7,
        pressure_stiffness=5.0e3,
        peak_pressure=4.0e2,
        young_modulus=2.0e5,
        poisson_ratio=0.30,
        density=1.0,
        damping_alpha=2.5,
        newmark_iterations=3,
        frame_stride=5,
        sdf_batch_projection_threshold=250_000,
        sdf_candidate_padding=None,
        use_contact_tangent=False,
        use_matrix_free_contact_tangent=False,
        use_cpp_contact_tangent_solver=False,
        cpp_contact_tangent_preconditioner="block-sgs",
        contact_tangent_cg_rtol=1.0e-10,
        contact_tangent_cg_atol=1.0e-12,
        contact_tangent_cg_maxiter=80,
        defer_diagnostics=False,
    )


def run_benchmark(
    *,
    out_dir: Path,
    quick: bool,
    skip_calculix: bool,
    timeout_seconds: int = 360,
    cfg: DynamicSurfaceConfig | None = None,
) -> dict[str, Path | int | str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = cfg if cfg is not None else default_config(quick=quick)
    start = time.perf_counter()
    checkpoint_stride = max(1, int(round(0.05 / max(float(config.dt), 1.0e-12))))
    sfc = run_sfc_pressure_dynamic(config, checkpoint_dir=out_dir, checkpoint_stride=checkpoint_stride)
    sfc_wall = time.perf_counter() - start
    _write_csv(out_dir / "large_area_dynamic_sfc_history.csv", sfc.history)
    _write_csv(out_dir / "large_area_dynamic_sfc_timing.csv", sfc.timing)
    vtk_outputs = _write_vtk_outputs(out_dir, sfc, config)

    calc_rows: list[Row] = []
    calc_status: Row | None = None
    if skip_calculix:
        write_calculix_input(config, out_dir / "calculix_runs" / "large_area_dynamic_surface_contact" / "large_area_dynamic_surface_contact.inp")
        calc_status = {
            "external_solver": "CalculiX",
            "completed": "false",
            "reason": "skipped",
            "wall_time_seconds": 0.0,
        }
    else:
        try:
            calc_rows, calc_status = run_calculix_reference(config, out_dir, timeout_seconds=timeout_seconds)
        except Exception as exc:  # pragma: no cover - depends on external solver availability
            calc_status = {
                "external_solver": "CalculiX",
                "completed": "false",
                "reason": str(exc),
                "wall_time_seconds": float("nan"),
            }
    _write_csv(out_dir / "large_area_dynamic_calculix_history.csv", calc_rows)
    _write_csv(out_dir / "large_area_dynamic_calculix_status.csv", [calc_status] if calc_status else [])
    comparison_rows = _compare_histories(sfc.history, calc_rows)
    _write_csv(out_dir / "large_area_dynamic_comparison.csv", comparison_rows)
    figure_outputs = _plot_histories(out_dir, sfc.history, calc_rows if calc_rows else None)
    summary = _summary_text(
        config,
        sfc,
        calc_status,
        comparison_rows,
        vtk_outputs,
        sfc_wall_time_seconds=sfc_wall,
    )
    (out_dir / "large_area_dynamic_summary.md").write_text(summary, encoding="utf-8")
    return {
        "out_dir": out_dir,
        "sfc_wall_time_seconds": sfc_wall,
        "sfc_history": out_dir / "large_area_dynamic_sfc_history.csv",
        "sfc_timing": out_dir / "large_area_dynamic_sfc_timing.csv",
        "calculix_history": out_dir / "large_area_dynamic_calculix_history.csv",
        "comparison": out_dir / "large_area_dynamic_comparison.csv",
        "summary": out_dir / "large_area_dynamic_summary.md",
        "figure_png": figure_outputs["png"],
        "vtk_pvd": vtk_outputs["pvd"],
        "vtk_frame_count": vtk_outputs["frame_count"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run the small deterministic smoke/paper-preview case")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "large_area_dynamic_surface_contact")
    parser.add_argument("--skip-calculix", action="store_true", help="write the CalculiX input but do not run ccx")
    parser.add_argument("--timeout-seconds", type=int, default=360)
    parser.add_argument("--total-time", type=float, default=None, help="override physical simulation time")
    parser.add_argument("--dt", type=float, default=None, help="override SFC time step")
    parser.add_argument("--nx", type=int, default=None, help="override lower C3D8 x resolution")
    parser.add_argument("--ny", type=int, default=None, help="override lower C3D8 y resolution")
    parser.add_argument("--nz", type=int, default=None, help="override lower C3D8 z resolution")
    parser.add_argument("--driver-nx", type=int, default=None, help="override driver surface x resolution")
    parser.add_argument("--driver-ny", type=int, default=None, help="override driver surface y resolution")
    parser.add_argument("--frame-stride", type=int, default=None, help="override VTK frame stride")
    parser.add_argument("--newmark-iterations", type=int, default=None, help="override SFC contact fixed-point iterations")
    parser.add_argument("--spacing", type=float, default=None, help="override SDF grid spacing")
    parser.add_argument("--band-radius", type=float, default=None, help="override SDF narrow-band radius")
    parser.add_argument("--closure", type=float, default=None, help="override driver closure")
    parser.add_argument("--ramp-time", type=float, default=None, help="override closure ramp time")
    parser.add_argument("--oscillation-amplitude", type=float, default=None, help="override post-ramp oscillation amplitude")
    parser.add_argument("--oscillation-frequency", type=float, default=None, help="override post-ramp oscillation frequency")
    parser.add_argument("--pressure-stiffness", type=float, default=None, help="override pressure-overclosure stiffness")
    parser.add_argument("--peak-pressure", type=float, default=None, help="override top pressure magnitude")
    parser.add_argument("--damping-alpha", type=float, default=None, help="override mass-proportional damping")
    parser.add_argument("--density", type=float, default=None, help="override material density")
    parser.add_argument("--sdf-batch-projection-threshold", type=int, default=None, help="override SDF all-faces batch threshold")
    parser.add_argument("--sdf-candidate-padding", type=float, default=None, help="override exact-fallback AABB candidate padding")
    parser.add_argument("--contact-tangent", action="store_true", help="assemble and solve with the field-contact tangent")
    parser.add_argument("--matrix-free-contact-tangent", action="store_true", help="solve with matrix-free field-contact tangent")
    parser.add_argument("--cpp-contact-tangent-solver", action="store_true", help="solve contact tangent with the C++ PCG backend")
    parser.add_argument(
        "--cpp-contact-tangent-preconditioner",
        choices=("sgs", "block-sgs"),
        default=None,
        help="C++ PCG preconditioner for the matrix-free contact tangent",
    )
    parser.add_argument("--contact-tangent-cg-rtol", type=float, default=None, help="matrix-free tangent CG relative tolerance")
    parser.add_argument("--contact-tangent-cg-atol", type=float, default=None, help="matrix-free tangent CG absolute tolerance")
    parser.add_argument("--contact-tangent-cg-maxiter", type=int, default=None, help="matrix-free tangent CG maximum iterations")
    parser.add_argument("--defer-diagnostics", action="store_true", help="move accepted-state contact diagnostics to postprocess")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = default_config(quick=bool(args.quick))
    overrides = {
        "total_time": args.total_time,
        "dt": args.dt,
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "driver_nx": args.driver_nx,
        "driver_ny": args.driver_ny,
        "frame_stride": args.frame_stride,
        "newmark_iterations": args.newmark_iterations,
        "spacing": args.spacing,
        "band_radius": args.band_radius,
        "closure": args.closure,
        "ramp_time": args.ramp_time,
        "oscillation_amplitude": args.oscillation_amplitude,
        "oscillation_frequency": args.oscillation_frequency,
        "pressure_stiffness": args.pressure_stiffness,
        "peak_pressure": args.peak_pressure,
        "damping_alpha": args.damping_alpha,
        "density": args.density,
        "sdf_batch_projection_threshold": args.sdf_batch_projection_threshold,
        "sdf_candidate_padding": args.sdf_candidate_padding,
        "contact_tangent_cg_rtol": args.contact_tangent_cg_rtol,
        "contact_tangent_cg_atol": args.contact_tangent_cg_atol,
        "contact_tangent_cg_maxiter": args.contact_tangent_cg_maxiter,
        "cpp_contact_tangent_preconditioner": args.cpp_contact_tangent_preconditioner,
    }
    cfg = replace(cfg, **{key: value for key, value in overrides.items() if value is not None})
    tangent_modes = sum(
        int(flag)
        for flag in (
            bool(args.contact_tangent),
            bool(args.matrix_free_contact_tangent),
            bool(args.cpp_contact_tangent_solver),
        )
    )
    if tangent_modes > 1:
        raise ValueError("--contact-tangent, --matrix-free-contact-tangent, and --cpp-contact-tangent-solver are mutually exclusive")
    if bool(args.contact_tangent):
        cfg = replace(cfg, use_contact_tangent=True)
    if bool(args.matrix_free_contact_tangent):
        cfg = replace(cfg, use_matrix_free_contact_tangent=True)
    if bool(args.cpp_contact_tangent_solver):
        cfg = replace(cfg, use_cpp_contact_tangent_solver=True)
    if bool(args.defer_diagnostics):
        cfg = replace(cfg, defer_diagnostics=True)
    outputs = run_benchmark(
        out_dir=args.out_dir,
        quick=bool(args.quick),
        skip_calculix=bool(args.skip_calculix),
        timeout_seconds=int(args.timeout_seconds),
        cfg=cfg,
    )
    print(f"Wrote {outputs['summary']}")
    print(f"SFC history: {outputs['sfc_history']}")
    print(f"SFC VTK PVD: {outputs['vtk_pvd']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
