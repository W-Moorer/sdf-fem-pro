"""Large-area dynamic surface-to-surface SDF contact benchmark.

This runner implements the selected "scheme B" engineering case: a dense,
prescribed upper surface ramps into a deformable C3D8 block, holds contact, and
then applies a small sustained oscillation.  The contact stays active over a
large surface patch, so the benchmark stresses the true
``DynamicNarrowBandSDF.build_required_points(...) -> field_contact`` path with
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
from dataclasses import dataclass
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

from sfc.contact.field_contact import (  # noqa: E402
    surface_to_surface_field_penalty_response_vectorized,
    triangle_surface_quadrature_cache,
)
from sfc.fem import DeformableBody  # noqa: E402
from sfc.fem.assembler import assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402
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
    """Parameters for the large-area prescribed dynamic indentation case."""

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
    young_modulus: float
    poisson_ratio: float
    density: float
    damping_alpha: float
    newmark_iterations: int
    frame_stride: int

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


def _normal_force_from_response(response: Any, pressure_stiffness: float) -> float:
    gaps = np.asarray(response.gaps, dtype=float)
    penetration = np.maximum(-gaps, 0.0)
    return float(np.sum(float(pressure_stiffness) * np.asarray(response.quadrature_weights, dtype=float) * penetration))


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


def run_sfc_dynamic(cfg: DynamicSurfaceConfig) -> SFCDynamicResult:
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

    beta = 0.25
    gamma = 0.5
    dt = float(cfg.dt)
    c0 = 1.0 / (beta * dt * dt)
    cv = gamma * dt * c0
    effective = (M * (c0 + cfg.damping_alpha * cv) + K).tocsc()
    effective_free = effective[free[:, None], free]

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
        master_sdf = DynamicNarrowBandSDF.build_required_points(
            x_master,
            top_faces,
            q_points,
            spacing=cfg.spacing,
            band_radius=cfg.band_radius,
            padding=cfg.band_radius,
            cell_size=max(2.0 * cfg.spacing, cfg.band_radius),
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
                "field_path": "DynamicNarrowBandSDF.build_required_points -> field_contact",
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
            sdf_next = DynamicNarrowBandSDF.build_required_points(
                x_guess,
                top_faces,
                q_next,
                spacing=cfg.spacing,
                band_radius=cfg.band_radius,
                padding=cfg.band_radius,
                cell_size=max(2.0 * cfg.spacing, cfg.band_radius),
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
            correction[free] = np.asarray(spsolve(effective_free, -residual[free]), dtype=float)
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
    """Write an optional native CalculiX contact model for the same kinematics."""

    lower, _driver_surface, _driver_faces = _make_geometry(cfg)
    driver = _structured_hex_box(
        nx=cfg.driver_nx,
        ny=cfg.driver_ny,
        nz=1,
        origin=(-7.5, -7.5, 4.0 + cfg.initial_gap),
        size=(15.0, 15.0, 0.35),
    )
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
        "Large-area prescribed dynamic surface-to-surface contact",
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
            "*AMPLITUDE, NAME=PUNCHZ, TIME=TOTAL TIME",
        ]
    )
    amp_pairs: list[float] = []
    for step in range(int(round(cfg.total_time / cfg.dt)) + 1):
        t = float(step * cfg.dt)
        amp_pairs.extend([t, _motion_z_offset(t, cfg)])
    for start in range(0, len(amp_pairs), 8):
        lines.append(", ".join(f"{value:.12e}" for value in amp_pairs[start : start + 8]))
    lines.extend(
        [
            "*BOUNDARY",
            "NFIXED, 1, 3, 0.0",
            "NDRIVER, 1, 2, 0.0",
            "*BOUNDARY, AMPLITUDE=PUNCHZ",
            "NDRIVER, 3, 3, 1.0",
            "*STEP, NLGEOM, INC=2000",
            "*DYNAMIC",
            f"{ccx_dt:.12e}, {cfg.total_time:.12e}, {0.1 * ccx_dt:.12e}, {ccx_dt:.12e}",
            f"*NODE PRINT, NSET=NLOWER, FREQUENCY={ccx_output_frequency}",
            "U",
            f"*NODE PRINT, NSET=NDRIVER, TOTALS=ONLY, GLOBAL=YES, FREQUENCY={ccx_output_frequency}",
            "RF",
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
    lower_node_ids, _driver_node_ids, lower, _driver = write_calculix_input(cfg, inp)
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
    blocks = _parse_nodal_vectors(dat, lower_node_ids, quantity="u")
    totals = _parse_totals(dat)
    top_mask = np.isclose(lower.X[:, 2], 4.0)
    rows: list[Row] = []
    for t, U in blocks.items():
        total = totals.get(float(t), {})
        rows.append(
            {
                "time": float(t),
                "top_mean_z_displacement": float(np.mean(U[top_mask, 2])),
                "normal_force": float(total.get("normal_force_calculix", np.nan)),
                "contact_energy": float(total.get("contact_energy_calculix", np.nan)),
                "contact_count": int(total.get("contact_count_calculix", 0)),
            }
        )
    return rows, status


def _write_vtk_outputs(out_dir: Path, result: SFCDynamicResult, cfg: DynamicSurfaceConfig) -> dict[str, Path | int]:
    vtk_dir = out_dir / "vtk" / "sfc"

    def cell_fields(time_value: float, U: np.ndarray) -> dict[str, np.ndarray]:
        fields = _element_fields(result.lower_mesh, U.reshape(-1), E=cfg.young_modulus, nu=cfg.poisson_ratio)
        driver_x = _driver_positions(result.driver_ref, time_value, cfg)
        contact_fields, _contact_row = _contact_cell_fields(
            result.lower_mesh,
            driver_x,
            result.driver_faces,
            result.driver_ref,
            U,
            cfg=cfg,
        )
        fields.update(contact_fields)
        return fields

    return write_hex_frame_series(
        vtk_dir,
        stem="sfc_large_area_dynamic",
        reference_points=result.lower_mesh.X,
        elements=result.lower_mesh.elements,
        frames=result.frames,
        frame_stride=cfg.frame_stride,
        cell_scalar_fn=cell_fields,
        source_label="SFC large-area dynamic surface contact",
    )


def _plot_histories(out_dir: Path, sfc_rows: list[Row], calculix_rows: list[Row] | None = None) -> dict[str, Path]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    t = np.asarray([row["time"] for row in sfc_rows], dtype=float)
    active = np.asarray([row["active_samples"] for row in sfc_rows], dtype=float)
    force = np.asarray([row["normal_force"] for row in sfc_rows], dtype=float)
    z = np.asarray([row["top_mean_z_displacement"] for row in sfc_rows], dtype=float)
    gap = np.asarray([row["min_gap"] for row in sfc_rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), constrained_layout=True)
    axes[0, 0].plot(t, z, "-o", ms=3.0, lw=1.2, label="SFC")
    if calculix_rows:
        tc = np.asarray([row["time"] for row in calculix_rows], dtype=float)
        zc = np.asarray([row["top_mean_z_displacement"] for row in calculix_rows], dtype=float)
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
    axes[0, 0].set_ylabel("top z displacement")

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
    zc = np.asarray([row["top_mean_z_displacement"] for row in calculix_rows], dtype=float)
    fc = np.asarray([row["normal_force"] for row in calculix_rows], dtype=float)
    rows: list[Row] = []
    for row in sfc_rows:
        t = float(row["time"])
        z_ref = float(np.interp(t, tc, zc)) if tc.size else float("nan")
        z_sfc = float(row["top_mean_z_displacement"])
        force_ref = float(np.interp(t, tc, fc)) if tc.size and np.any(np.isfinite(fc)) else float("nan")
        force_sfc = float(row["normal_force"])
        force_abs = abs(force_sfc - force_ref) if math.isfinite(force_ref) else float("nan")
        rows.append(
            {
                "time": t,
                "sfc_top_mean_z_displacement": z_sfc,
                "calculix_top_mean_z_displacement": z_ref,
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
) -> str:
    max_active = max(int(row["active_samples"]) for row in sfc.history)
    max_quad = max(int(row["quadrature_points"]) for row in sfc.history)
    avg_update = float(np.mean([row["field_update_seconds"] for row in sfc.timing]))
    avg_query = float(np.mean([row["field_query_seconds"] for row in sfc.timing]))
    lines = [
        "# Large-area dynamic surface-to-surface SDF contact",
        "",
        "This is scheme B: prescribed dynamic indentation with sustained large-area contact.",
        "",
        "## Configuration",
        "",
        f"- Lower C3D8 mesh: `{cfg.nx} x {cfg.ny} x {cfg.nz}` elements.",
        f"- Driver surface: `{cfg.driver_nx} x {cfg.driver_ny}` quads, triangulated for SFC quadrature.",
        f"- Time window: `{cfg.total_time}` with `dt={cfg.dt}`.",
        f"- Motion: closure `{cfg.closure}`, initial gap `{cfg.initial_gap}`, oscillation amplitude `{cfg.oscillation_amplitude}`.",
        f"- Contact quadrature points per step: `{max_quad}`.",
        f"- Maximum active quadrature samples: `{max_active}`.",
        "",
        "## SFC path",
        "",
        "- Current master surface -> `DynamicNarrowBandSDF.build_required_points(...)`.",
        "- Query/integration -> `surface_to_surface_field_penalty_response_vectorized(...)`.",
        "- Projection is used only inside field construction; field contact queries use interpolation.",
        "",
        "## Timing",
        "",
        f"- Mean field update time: `{avg_update:.6e}` s/step.",
        f"- Mean field query/contact assembly time: `{avg_query:.6e}` s/step.",
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
            lines.append(f"- Max top-displacement absolute error on common SFC times: `{max_abs:.6e}`.")
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
            young_modulus=2.0e5,
            poisson_ratio=0.30,
            density=1.0e-6,
            damping_alpha=2.5,
            newmark_iterations=2,
            frame_stride=1,
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
        young_modulus=2.0e5,
        poisson_ratio=0.30,
        density=1.0e-6,
        damping_alpha=2.5,
        newmark_iterations=3,
        frame_stride=5,
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
    sfc = run_sfc_dynamic(config)
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
    summary = _summary_text(config, sfc, calc_status, comparison_rows, vtk_outputs)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = run_benchmark(
        out_dir=args.out_dir,
        quick=bool(args.quick),
        skip_calculix=bool(args.skip_calculix),
        timeout_seconds=int(args.timeout_seconds),
    )
    print(f"Wrote {outputs['summary']}")
    print(f"SFC history: {outputs['sfc_history']}")
    print(f"SFC VTK PVD: {outputs['vtk_pvd']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
