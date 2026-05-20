"""Figure-9-style frictionless linear-contact implicit dynamic comparison.

This validation runner rebuilds the manuscript Figure 9 geometry as a small
engineering benchmark: a deformable C3D8 block fixed at its bottom face is
loaded by a frictionless rigid driver that first closes normally and then
shifts laterally.  Abaqus/Standard is used only as an external reference.  The
SFC path solves the same model internally with a dynamic SDF field contact
response, no Rayleigh damping, and no contact damping.
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
from scipy.sparse.linalg import factorized

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
from sfc.sdf.dynamic_narrow_band_sdf import RequiredPointSDFWorkspace  # noqa: E402
from validation.abaqus_odb_to_vtk import export_odb_to_vtk  # noqa: E402
from validation.run_abaqus_sphere_cantilever_explicit import (  # noqa: E402
    _abaqus_reported_wallclock_seconds,
    _resolve_abaqus_command,
    _run_command,
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

Row = dict[str, Any]
JOB_NAME = "fig9_linear_implicit"


@dataclass(frozen=True, slots=True)
class Fig9Config:
    nx: int = 12
    ny: int = 6
    nz: int = 3
    driver_nx: int = 6
    driver_ny: int = 6
    total_time: float = 0.04
    dt: float = 0.001
    closure_time: float = 0.02
    closure: float = 0.105
    shift_x: float = 2.0
    spacing: float = 0.30
    band_radius: float = 0.90
    quadrature_order: int = 3
    pressure_stiffness: float = 4500.0
    young_modulus: float = 1200.0
    poisson_ratio: float = 0.30
    density: float = 1.0
    newmark_iterations: int = 4
    output_stride: int = 1


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_geometry(cfg: Fig9Config) -> tuple[BoxMesh, np.ndarray, np.ndarray]:
    lower = _structured_hex_box(
        nx=cfg.nx,
        ny=cfg.ny,
        nz=cfg.nz,
        origin=(-10.0, -5.0, 0.0),
        size=(20.0, 10.0, 2.0),
    )
    driver_ref, driver_faces = _surface_grid(
        nx=cfg.driver_nx,
        ny=cfg.driver_ny,
        origin=(-2.25, -2.25, 2.065),
        size=(4.5, 4.5),
    )
    return lower, driver_ref, driver_faces


def _driver_offset(time_value: float, cfg: Fig9Config) -> np.ndarray:
    t = float(np.clip(time_value, 0.0, cfg.total_time))
    if cfg.closure_time > 0.0 and t <= cfg.closure_time:
        normal = 0.5 * (1.0 - math.cos(math.pi * t / cfg.closure_time))
        shear = 0.0
    else:
        normal = 1.0
        denom = max(cfg.total_time - cfg.closure_time, 1.0e-12)
        tau = np.clip((t - cfg.closure_time) / denom, 0.0, 1.0)
        shear = 0.5 * (1.0 - math.cos(math.pi * tau))
    return np.asarray((cfg.shift_x * shear, 0.0, -cfg.closure * normal), dtype=float)


def _driver_positions(driver_ref: np.ndarray, time_value: float, cfg: Fig9Config) -> np.ndarray:
    return np.asarray(driver_ref, dtype=float) + _driver_offset(time_value, cfg)


def _append_ids(lines: list[str], ids: list[int] | np.ndarray, *, per_line: int = 16) -> None:
    values = [int(value) for value in ids]
    for offset in range(0, len(values), per_line):
        lines.append(", ".join(str(value) for value in values[offset : offset + per_line]))


def _driver_quads(driver_nx: int, driver_ny: int) -> list[tuple[int, int, int, int]]:
    quads: list[tuple[int, int, int, int]] = []
    for j in range(driver_ny):
        for i in range(driver_nx):
            n00 = j * (driver_nx + 1) + i + 1
            n10 = n00 + 1
            n01 = (j + 1) * (driver_nx + 1) + i + 1
            n11 = n01 + 1
            quads.append((n00, n10, n11, n01))
    return quads


def build_abaqus_input_text(cfg: Fig9Config) -> str:
    lower, driver_ref, _driver_faces = make_geometry(cfg)
    lines: list[str] = [
        "*Heading",
        "** Figure-9-style implicit dynamic validation: frictionless linear contact, no damping.",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
        "*Part, name=LOWER",
        "*Node",
    ]
    for node_id, xyz in enumerate(lower.X, start=1):
        lines.append(f"{node_id}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    lines.append("*Element, type=C3D8")
    for element_id, element in enumerate(lower.elements, start=1):
        labels = [int(node) + 1 for node in element]
        lines.append(f"{element_id}, " + ", ".join(str(label) for label in labels))
    lines.append("*Elset, elset=LOWER_ALL")
    _append_ids(lines, np.arange(1, lower.elements.shape[0] + 1, dtype=np.int64))
    bottom_nodes = np.flatnonzero(np.isclose(lower.X[:, 2], lower.X[:, 2].min())) + 1
    lines.append("*Nset, nset=LOWER_BOTTOM")
    _append_ids(lines, bottom_nodes)
    top_element_ids = np.asarray([(cfg.nz - 1) * cfg.nx * cfg.ny + j * cfg.nx + i + 1 for j in range(cfg.ny) for i in range(cfg.nx)], dtype=np.int64)
    lines.append("*Elset, elset=LOWER_TOP")
    _append_ids(lines, top_element_ids)
    lines.extend(
        [
            "*Solid Section, elset=LOWER_ALL, material=SOFT",
            ",",
            "*End Part",
            "*Part, name=ZDRIVER",
            "*Node",
        ]
    )
    for node_id, xyz in enumerate(driver_ref, start=1):
        lines.append(f"{node_id}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    lines.append("*Element, type=R3D4")
    for element_id, quad in enumerate(_driver_quads(cfg.driver_nx, cfg.driver_ny), start=1):
        lines.append(f"{element_id}, {quad[0]}, {quad[1]}, {quad[2]}, {quad[3]}")
    lines.append("*Elset, elset=DRIVER_ALL")
    _append_ids(lines, np.arange(1, cfg.driver_nx * cfg.driver_ny + 1, dtype=np.int64))
    lines.extend(
        [
            "*End Part",
            "*Assembly, name=Assembly",
            "*Instance, name=LOWER-1, part=LOWER",
            "*End Instance",
            "*Instance, name=ZDRIVER-1, part=ZDRIVER",
            "*End Instance",
            "*Node",
            "900001, 0.000000000000e+00, 0.000000000000e+00, 2.065000000000e+00",
            "*Nset, nset=DRIVER_RP",
            "900001",
            "*Elset, elset=LOWER_TOP_ASM, instance=LOWER-1",
        ]
    )
    _append_ids(lines, top_element_ids)
    lines.append("*Elset, elset=DRIVER_ALL_ASM, instance=ZDRIVER-1")
    _append_ids(lines, np.arange(1, cfg.driver_nx * cfg.driver_ny + 1, dtype=np.int64))
    lines.extend(
        [
            "*Nset, nset=LOWER_BOTTOM_ASM, instance=LOWER-1",
        ]
    )
    _append_ids(lines, bottom_nodes)
    lines.extend(
        [
            "*Surface, type=ELEMENT, name=LOWER_TOP_SURF",
            "LOWER_TOP_ASM, S2",
            "*Surface, type=ELEMENT, name=DRIVER_SURF",
            "DRIVER_ALL_ASM, SNEG",
            "*Rigid Body, ref node=DRIVER_RP, elset=DRIVER_ALL_ASM",
            "*End Assembly",
            "*Material, name=SOFT",
            "*Density",
            f"{cfg.density:.12e}",
            "*Elastic",
            f"{cfg.young_modulus:.12e}, {cfg.poisson_ratio:.12e}",
            "*Surface Interaction, name=LINEAR_FRICTIONLESS",
            "*Surface Behavior, pressure-overclosure=LINEAR",
            f"{cfg.pressure_stiffness:.12e}",
            "*Friction",
            "0.",
            "*Contact Pair, interaction=LINEAR_FRICTIONLESS, type=SURFACE TO SURFACE",
            "LOWER_TOP_SURF, DRIVER_SURF",
            "*Amplitude, name=CLOSURE_AMP, time=TOTAL TIME",
            f"0., 0., {cfg.closure_time:.12e}, 1., {cfg.total_time:.12e}, 1.",
            "*Amplitude, name=SHIFT_AMP, time=TOTAL TIME",
            f"0., 0., {cfg.closure_time:.12e}, 0., {cfg.total_time:.12e}, 1.",
            "*Step, name=FIG9_LINEAR_IMPLICIT, nlgeom=YES, inc=10000",
            "*Dynamic, DIRECT, NOHAF",
            f"{cfg.dt:.12e}, {cfg.total_time:.12e}",
            "*Boundary",
            "LOWER_BOTTOM_ASM, 1, 3, 0.",
            "DRIVER_RP, 2, 2, 0.",
            "DRIVER_RP, 4, 6, 0.",
            "*Boundary, amplitude=CLOSURE_AMP",
            f"DRIVER_RP, 3, 3, {-cfg.closure:.12e}",
            "*Boundary, amplitude=SHIFT_AMP",
            f"DRIVER_RP, 1, 1, {cfg.shift_x:.12e}",
            f"*Output, field, time interval={cfg.dt * max(1, cfg.output_stride):.12e}",
            "*Node Output",
            "U, V",
            "*Element Output, directions=YES",
            "S, LE",
            f"*Output, history, time interval={cfg.dt * max(1, cfg.output_stride):.12e}",
            "*Energy Output",
            "ALLKE, ALLIE, ALLSE, ALLWK, ETOTAL",
            "*End Step",
        ]
    )
    return "\n".join(lines) + "\n"


def run_sfc(cfg: Fig9Config) -> tuple[list[Row], list[Row], float]:
    lower, driver_ref, driver_faces = make_geometry(cfg)
    body = DeformableBody(
        mesh=VolumeMesh(lower.X, lower.elements, element_type="C3D8"),
        material={"E": cfg.young_modulus, "nu": cfg.poisson_ratio},
        density=cfg.density,
    )
    K = assemble_stiffness_matrix(body).tocsr()
    M = assemble_mass_matrix(body, kind="consistent").tocsr()
    fixed = _fixed_bottom_dofs(lower)
    free = _free_dofs(body.n_dofs, fixed)
    lower_top_faces = _top_triangles(lower)
    contact_cache = triangle_surface_quadrature_cache(driver_faces, driver_ref, order=cfg.quadrature_order)
    sdf_workspace = RequiredPointSDFWorkspace.from_surface(
        lower.X,
        lower_top_faces,
        spacing=cfg.spacing,
        band_radius=cfg.band_radius,
        padding=cfg.band_radius,
        cell_size=max(2.0 * cfg.spacing, cfg.band_radius),
        batch_projection_threshold=256,
        candidate_padding=cfg.spacing,
    )
    beta = 0.25
    gamma = 0.5
    dt = float(cfg.dt)
    c0 = 1.0 / (beta * dt * dt)
    effective_free = (M * c0 + K).tocsc()[free[:, None], free]
    solve_free = factorized(effective_free) if free.size else None
    n_slave_dofs = 3 * driver_ref.shape[0]
    u = np.zeros(body.n_dofs, dtype=float)
    v = np.zeros(body.n_dofs, dtype=float)
    a = np.zeros(body.n_dofs, dtype=float)
    history: list[Row] = []
    timing: list[Row] = []
    top_nodes = np.flatnonzero(np.isclose(lower.X[:, 2], lower.X[:, 2].max()))
    start = time.perf_counter()
    steps = int(round(cfg.total_time / cfg.dt))
    for step in range(steps + 1):
        t = float(step * dt)
        x_master = lower.X + u.reshape((-1, 3))
        driver_x = _driver_positions(driver_ref, t, cfg)
        q_points = contact_cache.points(driver_x)
        field_t0 = time.perf_counter()
        master_sdf = sdf_workspace.build(x_master, lower_top_faces, q_points)
        field_elapsed = time.perf_counter() - field_t0
        query_t0 = time.perf_counter()
        response = surface_to_surface_field_penalty_response_vectorized(
            driver_x,
            driver_faces,
            master_sdf,
            pressure_stiffness=cfg.pressure_stiffness,
            n_total_dofs=n_slave_dofs + body.n_dofs,
            slave_x_reference=driver_ref,
            quadrature_cache=contact_cache,
            slave_dof_offset=0,
            master_dof_offset=n_slave_dofs,
        )
        query_elapsed = time.perf_counter() - query_t0
        fields = _element_fields(lower, u, E=cfg.young_modulus, nu=cfg.poisson_ratio)
        history.append(
            {
                "source": "sfc_lagrangian_sdf_implicit",
                "time": t,
                "top_mean_z_displacement": float(np.mean(u.reshape((-1, 3))[top_nodes, 2])),
                "max_displacement_norm": float(np.max(np.linalg.norm(u.reshape((-1, 3)), axis=1))),
                "max_von_mises": float(np.max(fields["von_mises"])),
                "p95_von_mises": float(np.percentile(fields["von_mises"], 95.0)),
                "max_strain_norm": float(np.max(fields["engineering_strain_norm"])),
                "p95_strain_norm": float(np.percentile(fields["engineering_strain_norm"], 95.0)),
                "min_gap": float(response.min_gap),
                "max_penetration": float(response.max_penetration),
                "active_samples": int(response.active_count),
                "normal_force": float(np.sum(cfg.pressure_stiffness * contact_cache.area_weights * np.maximum(-np.asarray(response.gaps), 0.0))),
            }
        )
        timing.append(
            {
                "time": t,
                "field_update_seconds": float(master_sdf.stats.update_seconds),
                "field_query_seconds": float(query_elapsed),
                "field_elapsed_seconds": float(field_elapsed),
                "active_samples": int(response.active_count),
                "quadrature_points": int(contact_cache.area_weights.size),
            }
        )
        if step == steps:
            break
        next_t = float((step + 1) * dt)
        u_pred = u + dt * v + dt * dt * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a
        u_guess = u_pred.copy()
        for _iteration in range(max(1, cfg.newmark_iterations)):
            x_guess = lower.X + u_guess.reshape((-1, 3))
            driver_next = _driver_positions(driver_ref, next_t, cfg)
            q_next = contact_cache.points(driver_next)
            sdf_next = sdf_workspace.build(x_guess, lower_top_faces, q_next)
            response_next = surface_to_surface_field_penalty_response_vectorized(
                driver_next,
                driver_faces,
                sdf_next,
                pressure_stiffness=cfg.pressure_stiffness,
                n_total_dofs=n_slave_dofs + body.n_dofs,
                slave_x_reference=driver_ref,
                quadrature_cache=contact_cache,
                slave_dof_offset=0,
                master_dof_offset=n_slave_dofs,
            )
            f_contact = response_next.force[n_slave_dofs:]
            a_guess = c0 * (u_guess - u_pred)
            residual = M @ a_guess + K @ u_guess - f_contact
            correction = np.zeros(body.n_dofs, dtype=float)
            if solve_free is not None:
                correction[free] = np.asarray(solve_free(-residual[free]), dtype=float)
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
    return history, timing, time.perf_counter() - start


def _read_vtk_vectors(path: Path, name: str, count: int) -> np.ndarray:
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    marker = f"vectors {name.lower()} float"
    for idx, line in enumerate(lines):
        if line.strip().lower() == marker:
            return np.asarray([[float(value) for value in lines[idx + 1 + row].split()[:3]] for row in range(count)], dtype=float)
    raise ValueError(f"VTK file has no vector field {name}: {path}")


def _read_vtk_scalar(path: Path, name: str, cell_count: int) -> np.ndarray:
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    marker = f"scalars {name.lower()} float 1"
    for idx, line in enumerate(lines):
        if line.strip().lower() == marker:
            start = idx + 2
            return np.asarray([float(lines[start + row].strip()) for row in range(cell_count)], dtype=float)
    raise ValueError(f"VTK file has no scalar field {name}: {path}")


def load_abaqus_history(manifest: Path, cfg: Fig9Config) -> list[Row]:
    lower, _driver_ref, _driver_faces = make_geometry(cfg)
    top_nodes = np.flatnonzero(np.isclose(lower.X[:, 2], lower.X[:, 2].max()))
    rows: list[Row] = []
    with manifest.open(newline="", encoding="ascii") as handle:
        manifest_rows = list(csv.DictReader(handle))
    for row in manifest_rows:
        vtk = manifest.parent / str(row["vtk_file"])
        U = _read_vtk_vectors(vtk, "U", lower.X.shape[0])
        vm = _read_vtk_scalar(vtk, "von_mises", lower.elements.shape[0])
        le = _read_vtk_scalar(vtk, "logarithmic_strain_norm", lower.elements.shape[0])
        rows.append(
            {
                "source": "abaqus_standard_implicit",
                "time": float(row["time"]),
                "top_mean_z_displacement": float(np.mean(U[top_nodes, 2])),
                "max_displacement_norm": float(np.max(np.linalg.norm(U, axis=1))),
                "max_von_mises": float(np.max(vm)),
                "p95_von_mises": float(np.percentile(vm, 95.0)),
                "max_strain_norm": float(np.max(le)),
                "p95_strain_norm": float(np.percentile(le, 95.0)),
            }
        )
    return rows


def _series(rows: list[Row], key: str) -> tuple[np.ndarray, np.ndarray]:
    t: list[float] = []
    y: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        t.append(float(row["time"]))
        y.append(float(value))
    return np.asarray(t, dtype=float), np.asarray(y, dtype=float)


def _metric(source: list[Row], reference: list[Row], key: str) -> Row:
    tref, yref = _series(reference, key)
    t, y = _series(source, key)
    if t.size == 0 or tref.size == 0:
        return {"metric": key, "l2_relative_error": "", "max_abs_error": "", "status": "missing"}
    interp = np.interp(tref, t, y)
    err = interp - yref
    denom = max(float(np.linalg.norm(yref)), 1.0e-14)
    return {
        "metric": key,
        "l2_relative_error": float(np.linalg.norm(err) / denom),
        "max_abs_error": float(np.max(np.abs(err))),
        "status": "computed",
    }


def _plot_curves(out_dir: Path, abaqus_rows: list[Row], sfc_rows: list[Row], metrics: list[Row]) -> Path:
    metric_by_name = {str(row["metric"]): row for row in metrics}
    keys = [
        ("top_mean_z_displacement", "top mean z displacement"),
        ("max_displacement_norm", "max displacement norm"),
        ("p95_von_mises", "95th percentile von Mises"),
        ("p95_strain_norm", "95th percentile strain norm"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for axis, (key, title) in zip(axes.ravel(), keys, strict=True):
        t_abq, y_abq = _series(abaqus_rows, key)
        t_sfc, y_sfc = _series(sfc_rows, key)
        err = float(metric_by_name[key]["l2_relative_error"])
        axis.plot(t_abq, y_abq, "-", lw=1.3, label="Abaqus/Standard")
        axis.plot(t_sfc, y_sfc, "--", lw=1.3, label="SFC Lagrangian SDF")
        axis.set_title(f"{title} (L2 err. {err:.2%})", fontsize=9)
        axis.set_xlabel("time (s)")
        axis.grid(True, alpha=0.25, linewidth=0.5)
        axis.legend(loc="best", fontsize=7, frameon=False)
    path = out_dir / "fig9_linear_implicit_curves.png"
    fig.savefig(path, dpi=300)
    fig.savefig(out_dir / "fig9_linear_implicit_curves.pdf")
    plt.close(fig)
    return path


def run_abaqus(cfg: Fig9Config, out_dir: Path, *, abaqus_command: str | None, skip_abaqus: bool) -> tuple[Path, Row]:
    run_dir = out_dir / "abaqus_run"
    vtk_dir = out_dir / "abaqus_vtk"
    run_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir.mkdir(parents=True, exist_ok=True)
    inp = run_dir / f"{JOB_NAME}.inp"
    if not skip_abaqus:
        inp.write_text(build_abaqus_input_text(cfg), encoding="ascii")
        command = _resolve_abaqus_command(abaqus_command)
        wall = _run_command(
            [command, f"job={JOB_NAME}", f"input={inp.name}", "interactive", "ask_delete=OFF", "cpus=8", "mp_mode=threads"],
            cwd=run_dir,
            log_path=out_dir / "abaqus_stdout.log",
        )
        odb = run_dir / f"{JOB_NAME}.odb"
        if not odb.exists():
            raise RuntimeError(f"Abaqus did not produce {odb}")
        sta = run_dir / f"{JOB_NAME}.sta"
        if sta.exists() and "THE ANALYSIS HAS NOT BEEN COMPLETED" in sta.read_text(encoding="utf-8", errors="ignore"):
            raise RuntimeError(f"Abaqus/Standard did not complete; inspect {sta}")
        command_parts = [
            command,
            "python",
            str(Path(__file__).with_name("abaqus_odb_to_vtk.py").resolve()),
            "--odb",
            str(odb.resolve()),
            "--out-dir",
            str(vtk_dir.resolve()),
            "--stem",
            "frame",
        ]
        export_wall = _run_command(command_parts, cwd=run_dir, log_path=out_dir / "abaqus_odb_to_vtk_stdout.log")
    else:
        wall = 0.0
        export_wall = 0.0
    manifest = vtk_dir / "frame_manifest.csv"
    if skip_abaqus and not manifest.exists():
        raise FileNotFoundError(f"--skip-abaqus requires an existing manifest: {manifest}")
    if not skip_abaqus and not manifest.exists():
        odb = run_dir / f"{JOB_NAME}.odb"
        export_odb_to_vtk(odb, vtk_dir, stem="frame")
    reported = _abaqus_reported_wallclock_seconds(run_dir / f"{JOB_NAME}.sta")
    return manifest, {
        "solver": "abaqus_standard_implicit",
        "analysis_wall_seconds": "" if skip_abaqus else float(wall),
        "reported_wall_seconds": "" if reported is None else float(reported),
        "export_wall_seconds": "" if skip_abaqus else float(export_wall),
    }


def run_workflow(out_dir: Path, *, cfg: Fig9Config, abaqus_command: str | None, skip_abaqus: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, abaqus_timing = run_abaqus(cfg, out_dir, abaqus_command=abaqus_command, skip_abaqus=skip_abaqus)
    abaqus_rows = load_abaqus_history(manifest, cfg)
    sfc_rows, sfc_timing_rows, sfc_wall = run_sfc(cfg)
    metrics = [
        _metric(sfc_rows, abaqus_rows, key)
        for key in ("top_mean_z_displacement", "max_displacement_norm", "p95_von_mises", "p95_strain_norm")
    ]
    timing = [
        abaqus_timing,
        {
            "solver": "sfc_lagrangian_sdf_implicit",
            "analysis_wall_seconds": float(sfc_wall),
            "reported_wall_seconds": "",
            "export_wall_seconds": "",
        },
    ]
    _write_csv(out_dir / "fig9_linear_implicit_history.csv", abaqus_rows + sfc_rows)
    _write_csv(out_dir / "fig9_linear_implicit_metrics.csv", metrics)
    _write_csv(out_dir / "fig9_linear_implicit_timing.csv", timing)
    _write_csv(out_dir / "fig9_linear_implicit_sfc_field_timing.csv", sfc_timing_rows)
    plot = _plot_curves(out_dir, abaqus_rows, sfc_rows, metrics)
    summary = out_dir / "fig9_linear_implicit_summary.md"
    lines = [
        "# Figure 9 Linear Implicit Contact Comparison",
        "",
        f"- Mesh: `{cfg.nx}x{cfg.ny}x{cfg.nz}` C3D8 lower block",
        f"- Duration: `{cfg.total_time}` s",
        f"- Fixed implicit increment: `{cfg.dt}` s",
        f"- Contact: frictionless linear pressure-overclosure, stiffness `{cfg.pressure_stiffness}`",
        "- Damping: no Rayleigh damping and no contact damping terms are introduced",
        f"- Plot: `{plot}`",
        "",
        "## Metrics",
        "",
        "| Metric | L2 relative error | Max absolute error |",
        "|---|---:|---:|",
    ]
    for row in metrics:
        lines.append(f"| {row['metric']} | {row['l2_relative_error']} | {row['max_abs_error']} |")
    lines.extend(["", "## Timing", "", "| Solver | Wall time (s) |", "|---|---:|"])
    for row in timing:
        wall = row.get("reported_wall_seconds", "") or row.get("analysis_wall_seconds", "")
        lines.append(f"| {row['solver']} | {wall} |")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summary": summary,
        "history": out_dir / "fig9_linear_implicit_history.csv",
        "metrics": out_dir / "fig9_linear_implicit_metrics.csv",
        "timing": out_dir / "fig9_linear_implicit_timing.csv",
        "field_timing": out_dir / "fig9_linear_implicit_sfc_field_timing.csv",
        "plot": plot,
        "abaqus_manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "fig9_linear_implicit_sfc_abaqus")
    parser.add_argument("--abaqus-command", default=None)
    parser.add_argument("--skip-abaqus", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nz", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--total-time", type=float, default=0.04)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Fig9Config(
        nx=8 if args.quick else int(args.nx),
        ny=4 if args.quick else int(args.ny),
        nz=2 if args.quick else int(args.nz),
        driver_nx=4 if args.quick else max(4, int(args.nx) // 2),
        driver_ny=4 if args.quick else max(4, int(args.ny)),
        dt=float(args.dt),
        total_time=float(args.total_time),
        closure_time=0.5 * float(args.total_time),
    )
    outputs = run_workflow(args.out_dir, cfg=cfg, abaqus_command=args.abaqus_command, skip_abaqus=bool(args.skip_abaqus))
    print("Figure 9 linear implicit comparison complete.")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
