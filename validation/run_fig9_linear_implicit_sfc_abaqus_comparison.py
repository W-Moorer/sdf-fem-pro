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
from scipy.sparse.linalg import factorized, spsolve

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
    SurfaceQuadratureCache,
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
    use_contact_tangent: bool = True
    contact_tolerance: float = 1.0e-9
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


def _driver_master_faces(driver_faces: np.ndarray) -> np.ndarray:
    """Flip the rigid driver surface so its SDF normal points toward lower top."""

    faces = np.asarray(driver_faces, dtype=np.int64)
    return faces[:, [0, 2, 1]].copy()


def _driver_swept_xy_bounds(driver_ref: np.ndarray, cfg: Fig9Config) -> tuple[float, float, float, float]:
    start = np.asarray(driver_ref, dtype=float)
    end = start + np.asarray((cfg.shift_x, 0.0, -cfg.closure), dtype=float)
    both = np.vstack((start, end))
    return (
        float(np.min(both[:, 0])),
        float(np.max(both[:, 0])),
        float(np.min(both[:, 1])),
        float(np.max(both[:, 1])),
    )


def _lower_contact_top_element_ids(mesh: BoxMesh, driver_ref: np.ndarray, cfg: Fig9Config) -> np.ndarray:
    """Return 1-based top element ids in the swept driver footprint plus one cell pad."""

    xmin, xmax, ymin, ymax = _driver_swept_xy_bounds(driver_ref, cfg)
    cell_pad = max(float(mesh.size[0]) / float(mesh.nx), float(mesh.size[1]) / float(mesh.ny), float(cfg.band_radius))
    ids: list[int] = []
    for j in range(mesh.ny):
        for i in range(mesh.nx):
            eid = (mesh.nz - 1) * mesh.ny * mesh.nx + j * mesh.nx + i
            centroid = np.mean(mesh.X[mesh.elements[eid]], axis=0)
            if xmin - cell_pad <= centroid[0] <= xmax + cell_pad and ymin - cell_pad <= centroid[1] <= ymax + cell_pad:
                ids.append(eid + 1)
    if not ids:
        raise RuntimeError("driver swept footprint selected no lower top elements")
    return np.asarray(ids, dtype=np.int64)


def _top_triangles_for_element_ids(mesh: BoxMesh, element_ids_1based: np.ndarray) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for element_id in np.asarray(element_ids_1based, dtype=np.int64):
        element = mesh.elements[int(element_id) - 1]
        quad = element[[4, 5, 6, 7]]
        faces.append((int(quad[0]), int(quad[1]), int(quad[2])))
        faces.append((int(quad[0]), int(quad[2]), int(quad[3])))
    return np.asarray(faces, dtype=np.int64)


def _workspace_from_contact_envelope(
    lower: BoxMesh,
    driver_ref: np.ndarray,
    lower_contact_faces: np.ndarray,
    cfg: Fig9Config,
) -> RequiredPointSDFWorkspace:
    """Create a reusable required-point grid covering the full driver sweep."""

    h = np.full(3, float(cfg.spacing), dtype=float)
    lower_contact_points = lower.X[np.asarray(lower_contact_faces, dtype=np.int64).ravel()]
    driver_start = np.asarray(driver_ref, dtype=float)
    driver_end = driver_start + np.asarray((cfg.shift_x, 0.0, -cfg.closure), dtype=float)
    envelope = np.vstack((lower_contact_points, driver_start, driver_end))
    cell_pad = max(float(lower.size[0]) / float(lower.nx), float(lower.size[1]) / float(lower.ny), float(cfg.band_radius))
    effective_band = max(float(cfg.band_radius), abs(float(cfg.shift_x)) + 2.0 * cell_pad + 2.0 * float(cfg.spacing))
    padding = effective_band + 2.0 * float(cfg.spacing)
    origin = np.min(envelope, axis=0) - padding
    upper = np.max(envelope, axis=0) + padding
    shape = tuple((np.ceil((upper - origin) / h).astype(np.int64) + 2).tolist())
    return RequiredPointSDFWorkspace(
        spacing=h,
        band_radius=effective_band,
        origin=origin,
        shape=shape,
        cell_size=max(2.0 * float(cfg.spacing), effective_band),
        batch_projection_threshold=256,
        candidate_padding=effective_band,
    )


def _driver_footprint_mask(points: np.ndarray, driver_x: np.ndarray, *, tolerance: float = 1.0e-10) -> np.ndarray:
    """Keep slave quadrature points whose projection lies on the rigid driver face."""

    p = np.asarray(points, dtype=float)
    d = np.asarray(driver_x, dtype=float)
    xmin, xmax = float(np.min(d[:, 0])), float(np.max(d[:, 0]))
    ymin, ymax = float(np.min(d[:, 1])), float(np.max(d[:, 1]))
    tol = float(tolerance)
    return (p[:, 0] >= xmin - tol) & (p[:, 0] <= xmax + tol) & (p[:, 1] >= ymin - tol) & (p[:, 1] <= ymax + tol)


def _filtered_cache(cache: SurfaceQuadratureCache, mask: np.ndarray) -> SurfaceQuadratureCache:
    keep = np.asarray(mask, dtype=bool)
    return SurfaceQuadratureCache(
        node_ids=cache.node_ids[keep].copy(),
        weights=cache.weights[keep].copy(),
        area_weights=cache.area_weights[keep].copy(),
    )


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
    top_element_ids = _lower_contact_top_element_ids(lower, driver_ref, cfg)
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
            "*Step, name=FIG9_LINEAR_IMPLICIT, nlgeom=NO, inc=10000",
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
            "S, E, LE",
            f"*Output, history, time interval={cfg.dt * max(1, cfg.output_stride):.12e}",
            "*Node Output, nset=DRIVER_RP",
            "U3, RF3, CF3",
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
    lower_contact_element_ids = _lower_contact_top_element_ids(lower, driver_ref, cfg)
    lower_contact_faces = _top_triangles_for_element_ids(lower, lower_contact_element_ids)
    driver_master_faces = _driver_master_faces(driver_faces)
    contact_cache = triangle_surface_quadrature_cache(lower_contact_faces, lower.X, order=cfg.quadrature_order)
    sdf_workspace = _workspace_from_contact_envelope(lower, driver_ref, lower_contact_faces, cfg)
    beta = 0.25
    gamma = 0.5
    dt = float(cfg.dt)
    c0 = 1.0 / (beta * dt * dt)
    effective_free = (M * c0 + K).tocsc()[free[:, None], free]
    solve_free = factorized(effective_free) if free.size else None
    n_driver_dofs = 3 * driver_ref.shape[0]
    n_total_contact_dofs = body.n_dofs + n_driver_dofs
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
        x_slave = lower.X + u.reshape((-1, 3))
        driver_x = _driver_positions(driver_ref, t, cfg)
        all_q_points = contact_cache.points(x_slave)
        active_cache = _filtered_cache(contact_cache, _driver_footprint_mask(all_q_points, driver_x, tolerance=-0.25 * cfg.spacing))
        q_points = active_cache.points(x_slave)
        field_t0 = time.perf_counter()
        master_sdf = sdf_workspace.build(driver_x, driver_master_faces, q_points)
        field_elapsed = time.perf_counter() - field_t0
        query_t0 = time.perf_counter()
        response = surface_to_surface_field_penalty_response_vectorized(
            x_slave,
            lower_contact_faces,
            master_sdf,
            pressure_stiffness=cfg.pressure_stiffness,
            n_total_dofs=n_total_contact_dofs,
            slave_x_reference=lower.X,
            quadrature_cache=active_cache,
            slave_dof_offset=0,
            master_dof_offset=body.n_dofs,
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
                "normal_force": float(np.sum(cfg.pressure_stiffness * active_cache.area_weights * np.maximum(-np.asarray(response.gaps), 0.0))),
            }
        )
        timing.append(
            {
                "time": t,
                "field_update_seconds": float(master_sdf.stats.update_seconds),
                "field_query_seconds": float(query_elapsed),
                "field_elapsed_seconds": float(field_elapsed),
                "active_samples": int(response.active_count),
                "quadrature_points": int(active_cache.area_weights.size),
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
            all_q_next = contact_cache.points(x_guess)
            active_cache_next = _filtered_cache(contact_cache, _driver_footprint_mask(all_q_next, driver_next, tolerance=-0.25 * cfg.spacing))
            q_next = active_cache_next.points(x_guess)
            sdf_next = sdf_workspace.build(driver_next, driver_master_faces, q_next)
            response_next = surface_to_surface_field_penalty_response_vectorized(
                x_guess,
                lower_contact_faces,
                sdf_next,
                pressure_stiffness=cfg.pressure_stiffness,
                n_total_dofs=n_total_contact_dofs,
                slave_x_reference=lower.X,
                quadrature_cache=active_cache_next,
                slave_dof_offset=0,
                master_dof_offset=body.n_dofs,
                assemble_stiffness=bool(cfg.use_contact_tangent),
            )
            f_contact = response_next.force[: body.n_dofs]
            a_guess = c0 * (u_guess - u_pred)
            residual = M @ a_guess + K @ u_guess - f_contact
            correction = np.zeros(body.n_dofs, dtype=float)
            if solve_free is not None:
                if bool(cfg.use_contact_tangent):
                    tangent = (M * c0 + K + response_next.stiffness[: body.n_dofs, : body.n_dofs]).tocsc()
                    correction[free] = np.asarray(spsolve(tangent[free[:, None], free], -residual[free]), dtype=float)
                else:
                    correction[free] = np.asarray(solve_free(-residual[free]), dtype=float)
            u_guess[free] += correction[free]
            if fixed.size:
                u_guess[fixed] = 0.0
            if float(np.linalg.norm(correction[free])) <= float(cfg.contact_tolerance) * max(1.0, float(np.linalg.norm(u_guess[free]))):
                break
        u = u_guess
        a = c0 * (u - u_pred)
        v = v_pred + gamma * dt * a
        if fixed.size:
            u[fixed] = 0.0
            v[fixed] = 0.0
            a[fixed] = 0.0
    return history, timing, time.perf_counter() - start


def _abaqus_driver_history_export_script() -> str:
    return r'''
from __future__ import print_function

import csv
import sys

from odbAccess import openOdb


odb_path = sys.argv[1]
out_path = sys.argv[2]
odb = openOdb(path=odb_path, readOnly=True)
try:
    step = odb.steps[list(odb.steps.keys())[0]]
    selected = None
    for region_name, region in step.historyRegions.items():
        keys = set(region.historyOutputs.keys())
        if "RF3" in keys or "CF3" in keys:
            selected = region
            break
    rows = {}
    if selected is not None:
        for key in ("U3", "RF3", "CF3"):
            if key not in selected.historyOutputs:
                continue
            output = selected.historyOutputs[key]
            for time_value, value in output.data:
                rows.setdefault(float(time_value), {})[key.lower()] = float(value)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "u3", "rf3", "cf3", "normal_force"])
        writer.writeheader()
        for time_value in sorted(rows):
            row = rows[time_value]
            rf3 = row.get("rf3", "")
            cf3 = row.get("cf3", "")
            source = rf3 if rf3 != "" else cf3
            writer.writerow(
                {
                    "time": time_value,
                    "u3": row.get("u3", ""),
                    "rf3": rf3,
                    "cf3": cf3,
                    "normal_force": "" if source == "" else abs(float(source)),
                }
            )
finally:
    odb.close()
'''


def _abaqus_frame_metrics_export_script() -> str:
    return r'''
from __future__ import print_function

import csv
import math
import sys

from odbAccess import openOdb


def tensor_norm(data):
    values = [float(v) for v in data]
    return math.sqrt(sum(v * v for v in values))


def percentile(values, pct):
    if not values:
        return ""
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct / 100.0))
    return ordered[idx]


odb_path = sys.argv[1]
out_path = sys.argv[2]
odb = openOdb(path=odb_path, readOnly=True)
try:
    step = odb.steps[list(odb.steps.keys())[0]]
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "p95_small_strain_norm", "max_small_strain_norm"])
        writer.writeheader()
        for frame in step.frames:
            strain_values = []
            if "E" in frame.fieldOutputs:
                field = frame.fieldOutputs["E"]
            elif "LE" in frame.fieldOutputs:
                field = frame.fieldOutputs["LE"]
            else:
                field = None
            if field is not None:
                for value in field.values:
                    instance = getattr(value, "instance", None)
                    if instance is not None and "LOWER" not in instance.name.upper():
                        continue
                    strain_values.append(tensor_norm(value.data))
            writer.writerow(
                {
                    "time": float(frame.frameValue),
                    "p95_small_strain_norm": percentile(strain_values, 95.0),
                    "max_small_strain_norm": max(strain_values) if strain_values else "",
                }
            )
finally:
    odb.close()
'''


def _read_abaqus_driver_history(path: Path | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ("u3", "rf3", "cf3", "normal_force"):
        t: list[float] = []
        y: list[float] = []
        for row in rows:
            value = row.get(key, "")
            if value == "":
                continue
            t.append(float(row["time"]))
            y.append(float(value))
        if t:
            result[key] = (np.asarray(t, dtype=float), np.asarray(y, dtype=float))
    return result


def _read_abaqus_frame_metrics(path: Path | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    mapping = {
        "p95_small_strain_norm": "p95_strain_norm",
        "max_small_strain_norm": "max_strain_norm",
    }
    for csv_key, public_key in mapping.items():
        t: list[float] = []
        y: list[float] = []
        for row in rows:
            value = row.get(csv_key, "")
            if value == "":
                continue
            t.append(float(row["time"]))
            y.append(float(value))
        if t:
            result[public_key] = (np.asarray(t, dtype=float), np.asarray(y, dtype=float))
    return result


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


def load_abaqus_history(
    manifest: Path,
    cfg: Fig9Config,
    *,
    driver_history: Path | None = None,
    frame_metrics: Path | None = None,
) -> list[Row]:
    lower, _driver_ref, _driver_faces = make_geometry(cfg)
    top_nodes = np.flatnonzero(np.isclose(lower.X[:, 2], lower.X[:, 2].max()))
    driver_series = _read_abaqus_driver_history(driver_history)
    frame_series = _read_abaqus_frame_metrics(frame_metrics)
    rows: list[Row] = []
    with manifest.open(newline="", encoding="ascii") as handle:
        manifest_rows = list(csv.DictReader(handle))
    for row in manifest_rows:
        vtk = manifest.parent / str(row["vtk_file"])
        U = _read_vtk_vectors(vtk, "U", lower.X.shape[0])
        vm = _read_vtk_scalar(vtk, "von_mises", lower.elements.shape[0])
        le = _read_vtk_scalar(vtk, "logarithmic_strain_norm", lower.elements.shape[0])
        time_value = float(row["time"])
        item: Row = {
            "source": "abaqus_standard_implicit",
            "time": time_value,
            "top_mean_z_displacement": float(np.mean(U[top_nodes, 2])),
            "max_displacement_norm": float(np.max(np.linalg.norm(U, axis=1))),
            "max_von_mises": float(np.max(vm)),
            "p95_von_mises": float(np.percentile(vm, 95.0)),
            "max_strain_norm": float(np.max(le)),
            "p95_strain_norm": float(np.percentile(le, 95.0)),
        }
        for key, (times, values) in driver_series.items():
            item[f"driver_{key}"] = float(np.interp(time_value, times, values))
        if "normal_force" in driver_series:
            times, values = driver_series["normal_force"]
            item["normal_force"] = float(np.interp(time_value, times, values))
        for key, (times, values) in frame_series.items():
            item[key] = float(np.interp(time_value, times, values))
        rows.append(item)
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
        ("normal_force", "driver/contact normal force"),
        ("p95_von_mises", "95th percentile von Mises"),
        ("p95_strain_norm", "95th percentile strain norm"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0), constrained_layout=True)
    flat_axes = axes.ravel()
    for axis, (key, title) in zip(flat_axes, keys, strict=False):
        t_abq, y_abq = _series(abaqus_rows, key)
        t_sfc, y_sfc = _series(sfc_rows, key)
        raw_err = metric_by_name.get(key, {}).get("l2_relative_error", "")
        err_label = "missing" if raw_err == "" else f"{float(raw_err):.2%}"
        axis.plot(t_abq, y_abq, "-", lw=1.3, label="Abaqus/Standard")
        axis.plot(t_sfc, y_sfc, "--", lw=1.3, label="SFC Lagrangian SDF")
        axis.set_title(f"{title} (L2 err. {err_label})", fontsize=9)
        axis.set_xlabel("time (s)")
        axis.grid(True, alpha=0.25, linewidth=0.5)
        axis.legend(loc="best", fontsize=7, frameon=False)
    for axis in flat_axes[len(keys) :]:
        axis.axis("off")
    path = out_dir / "fig9_linear_implicit_curves.png"
    fig.savefig(path, dpi=300)
    fig.savefig(out_dir / "fig9_linear_implicit_curves.pdf")
    plt.close(fig)
    return path


def run_abaqus(cfg: Fig9Config, out_dir: Path, *, abaqus_command: str | None, skip_abaqus: bool) -> tuple[Path, Path, Path, Row]:
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
        history_csv = out_dir / "abaqus_driver_reaction_history.csv"
        history_script = run_dir / "export_driver_reaction_history.py"
        history_script.write_text(_abaqus_driver_history_export_script(), encoding="ascii")
        history_export_wall = _run_command(
            [command, "python", str(history_script.resolve()), str(odb.resolve()), str(history_csv.resolve())],
            cwd=run_dir,
            log_path=out_dir / "abaqus_driver_history_stdout.log",
        )
        frame_metrics_csv = out_dir / "abaqus_frame_metrics.csv"
        frame_metrics_script = run_dir / "export_frame_metrics.py"
        frame_metrics_script.write_text(_abaqus_frame_metrics_export_script(), encoding="ascii")
        frame_metrics_wall = _run_command(
            [command, "python", str(frame_metrics_script.resolve()), str(odb.resolve()), str(frame_metrics_csv.resolve())],
            cwd=run_dir,
            log_path=out_dir / "abaqus_frame_metrics_stdout.log",
        )
    else:
        wall = 0.0
        export_wall = 0.0
        history_export_wall = 0.0
        history_csv = out_dir / "abaqus_driver_reaction_history.csv"
        frame_metrics_wall = 0.0
        frame_metrics_csv = out_dir / "abaqus_frame_metrics.csv"
    manifest = vtk_dir / "frame_manifest.csv"
    if skip_abaqus and not manifest.exists():
        raise FileNotFoundError(f"--skip-abaqus requires an existing manifest: {manifest}")
    if not skip_abaqus and not manifest.exists():
        odb = run_dir / f"{JOB_NAME}.odb"
        export_odb_to_vtk(odb, vtk_dir, stem="frame")
    reported = _abaqus_reported_wallclock_seconds(run_dir / f"{JOB_NAME}.sta")
    return manifest, history_csv, frame_metrics_csv, {
        "solver": "abaqus_standard_implicit",
        "analysis_wall_seconds": "" if skip_abaqus else float(wall),
        "reported_wall_seconds": "" if reported is None else float(reported),
        "export_wall_seconds": "" if skip_abaqus else float(export_wall),
        "history_export_wall_seconds": "" if skip_abaqus else float(history_export_wall),
        "frame_metrics_wall_seconds": "" if skip_abaqus else float(frame_metrics_wall),
    }


def run_workflow(out_dir: Path, *, cfg: Fig9Config, abaqus_command: str | None, skip_abaqus: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, abaqus_driver_history, abaqus_frame_metrics, abaqus_timing = run_abaqus(
        cfg,
        out_dir,
        abaqus_command=abaqus_command,
        skip_abaqus=skip_abaqus,
    )
    abaqus_rows = load_abaqus_history(manifest, cfg, driver_history=abaqus_driver_history, frame_metrics=abaqus_frame_metrics)
    sfc_rows, sfc_timing_rows, sfc_wall = run_sfc(cfg)
    metrics = [
        _metric(sfc_rows, abaqus_rows, key)
        for key in ("top_mean_z_displacement", "max_displacement_norm", "normal_force", "p95_von_mises", "p95_strain_norm")
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
        "abaqus_driver_history": abaqus_driver_history,
        "abaqus_frame_metrics": abaqus_frame_metrics,
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
