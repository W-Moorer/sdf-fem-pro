"""Two-flexible-body C3D8 contact comparison against Abaqus/Standard.

The benchmark removes the rigid open-surface driver used in the Figure 9
runner.  A flexible upper cube is driven by prescribed displacement on its top
face and contacts the top face of a flexible lower block.  SFC solves both
deformable bodies internally and uses the upper cube bottom surface as the SDF
master; Abaqus/Standard solves the same two-flexible-body contact model as an
external reference.
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
from scipy.sparse import block_diag
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
    quadrilateral_surface_quadrature_cache,
    surface_to_surface_field_penalty_response_vectorized,
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
)

Row = dict[str, Any]
JOB_NAME = "flexible_cube_sdf_contact"


@dataclass(frozen=True, slots=True)
class FlexibleCubeConfig:
    lower_nx: int = 12
    lower_ny: int = 6
    lower_nz: int = 3
    upper_nx: int = 6
    upper_ny: int = 6
    upper_nz: int = 2
    total_time: float = 0.04
    dt: float = 0.001
    closure_time: float = 0.02
    closure: float = 0.20
    initial_gap: float = -0.02
    spacing: float = 0.30
    band_radius: float = 0.90
    quadrature_order: int = 2
    pressure_stiffness: float = 4500.0
    lower_young_modulus: float = 1200.0
    upper_young_modulus: float = 1200.0
    poisson_ratio: float = 0.30
    density: float = 1.0
    newmark_iterations: int = 12
    contact_tolerance: float = 1.0e-9
    residual_tolerance: float = 1.0e-8
    force_tolerance: float = 1.0e-8
    output_stride: int = 1


def make_geometry(cfg: FlexibleCubeConfig) -> tuple[BoxMesh, BoxMesh]:
    lower = _structured_hex_box(
        nx=cfg.lower_nx,
        ny=cfg.lower_ny,
        nz=cfg.lower_nz,
        origin=(-10.0, -5.0, 0.0),
        size=(20.0, 10.0, 2.0),
    )
    upper = _structured_hex_box(
        nx=cfg.upper_nx,
        ny=cfg.upper_ny,
        nz=cfg.upper_nz,
        origin=(-2.25, -2.25, 2.0 + cfg.initial_gap),
        size=(4.5, 4.5, 2.0),
    )
    return lower, upper


def _append_ids(lines: list[str], ids: list[int] | np.ndarray, *, per_line: int = 16) -> None:
    values = [int(value) for value in ids]
    for offset in range(0, len(values), per_line):
        lines.append(", ".join(str(value) for value in values[offset : offset + per_line]))


def _top_element_ids(mesh: BoxMesh) -> np.ndarray:
    return np.asarray(
        [(mesh.nz - 1) * mesh.nx * mesh.ny + j * mesh.nx + i + 1 for j in range(mesh.ny) for i in range(mesh.nx)],
        dtype=np.int64,
    )


def _bottom_element_ids(mesh: BoxMesh) -> np.ndarray:
    return np.asarray([j * mesh.nx + i + 1 for j in range(mesh.ny) for i in range(mesh.nx)], dtype=np.int64)


def _contact_lower_top_element_ids(lower: BoxMesh, upper: BoxMesh, cfg: FlexibleCubeConfig) -> np.ndarray:
    xmin, xmax = float(np.min(upper.X[:, 0])), float(np.max(upper.X[:, 0]))
    ymin, ymax = float(np.min(upper.X[:, 1])), float(np.max(upper.X[:, 1]))
    cell_pad = max(float(lower.size[0]) / float(lower.nx), float(lower.size[1]) / float(lower.ny), float(cfg.band_radius))
    ids: list[int] = []
    for j in range(lower.ny):
        for i in range(lower.nx):
            eid = (lower.nz - 1) * lower.ny * lower.nx + j * lower.nx + i
            centroid = np.mean(lower.X[lower.elements[eid]], axis=0)
            if xmin - cell_pad <= centroid[0] <= xmax + cell_pad and ymin - cell_pad <= centroid[1] <= ymax + cell_pad:
                ids.append(eid + 1)
    if not ids:
        raise RuntimeError("upper cube footprint selected no lower top elements")
    return np.asarray(ids, dtype=np.int64)


def _top_quads_for_element_ids(mesh: BoxMesh, element_ids_1based: np.ndarray) -> np.ndarray:
    quads: list[tuple[int, int, int, int]] = []
    for element_id in np.asarray(element_ids_1based, dtype=np.int64):
        element = mesh.elements[int(element_id) - 1]
        quad = element[[4, 5, 6, 7]]
        quads.append((int(quad[0]), int(quad[1]), int(quad[2]), int(quad[3])))
    return np.asarray(quads, dtype=np.int64)


def _upper_bottom_triangles(mesh: BoxMesh) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for element_id in _bottom_element_ids(mesh):
        element = mesh.elements[int(element_id) - 1]
        quad = element[[0, 1, 2, 3]]
        faces.append((int(quad[0]), int(quad[2]), int(quad[1])))
        faces.append((int(quad[0]), int(quad[3]), int(quad[2])))
    return np.asarray(faces, dtype=np.int64)


def _upper_bottom_points(mesh: BoxMesh) -> np.ndarray:
    return mesh.X[np.flatnonzero(np.isclose(mesh.X[:, 2], float(np.min(mesh.X[:, 2]))))]


def _closure_state(time_value: float, cfg: FlexibleCubeConfig) -> tuple[float, float, float]:
    t = float(np.clip(time_value, 0.0, cfg.total_time))
    if cfg.closure_time > 0.0 and t <= cfg.closure_time:
        w = math.pi / cfg.closure_time
        amp = 0.5 * (1.0 - math.cos(w * t))
        vel = 0.5 * w * math.sin(w * t)
        acc = 0.5 * w * w * math.cos(w * t)
    else:
        amp = 1.0
        vel = 0.0
        acc = 0.0
    return -cfg.closure * amp, -cfg.closure * vel, -cfg.closure * acc


def _dirichlet_state(
    time_value: float,
    cfg: FlexibleCubeConfig,
    lower: BoxMesh,
    upper: BoxMesh,
    upper_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dofs: list[int] = []
    u_values: list[float] = []
    v_values: list[float] = []
    a_values: list[float] = []
    lower_fixed = _fixed_bottom_dofs(lower)
    for dof in lower_fixed:
        dofs.append(int(dof))
        u_values.append(0.0)
        v_values.append(0.0)
        a_values.append(0.0)
    upper_top = np.flatnonzero(np.isclose(upper.X[:, 2], float(np.max(upper.X[:, 2]))))
    uz, vz, az = _closure_state(time_value, cfg)
    for node in upper_top:
        for component, u_value, v_value, a_value in ((0, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 0.0), (2, uz, vz, az)):
            dofs.append(int(upper_offset + 3 * int(node) + component))
            u_values.append(float(u_value))
            v_values.append(float(v_value))
            a_values.append(float(a_value))
    return (
        np.asarray(dofs, dtype=np.int64),
        np.asarray(u_values, dtype=float),
        np.asarray(v_values, dtype=float),
        np.asarray(a_values, dtype=float),
    )


def _workspace_from_contact_envelope(
    lower: BoxMesh,
    upper: BoxMesh,
    lower_contact_quads: np.ndarray,
    cfg: FlexibleCubeConfig,
) -> RequiredPointSDFWorkspace:
    h = np.full(3, float(cfg.spacing), dtype=float)
    lower_points = lower.X[np.asarray(lower_contact_quads, dtype=np.int64).ravel()]
    upper_start = _upper_bottom_points(upper)
    upper_end = upper_start + np.asarray((0.0, 0.0, -cfg.closure), dtype=float)
    envelope = np.vstack((lower_points, upper_start, upper_end))
    effective_band = max(float(cfg.band_radius), abs(float(cfg.closure)) + 3.0 * float(cfg.spacing))
    padding = effective_band + 2.0 * float(cfg.spacing)
    origin = np.min(envelope, axis=0) - padding
    upper_bound = np.max(envelope, axis=0) + padding
    shape = tuple((np.ceil((upper_bound - origin) / h).astype(np.int64) + 2).tolist())
    return RequiredPointSDFWorkspace(
        spacing=h,
        band_radius=effective_band,
        origin=origin,
        shape=shape,
        cell_size=max(2.0 * float(cfg.spacing), effective_band),
        batch_projection_threshold=256,
        candidate_padding=effective_band,
    )


def _footprint_mask(points: np.ndarray, upper_x: np.ndarray, *, tolerance: float) -> np.ndarray:
    bottom = upper_x[np.isclose(upper_x[:, 2], float(np.min(upper_x[:, 2])), atol=1.0e-8)]
    p = np.asarray(points, dtype=float)
    xmin, xmax = float(np.min(bottom[:, 0])), float(np.max(bottom[:, 0]))
    ymin, ymax = float(np.min(bottom[:, 1])), float(np.max(bottom[:, 1]))
    tol = float(tolerance)
    return (p[:, 0] >= xmin - tol) & (p[:, 0] <= xmax + tol) & (p[:, 1] >= ymin - tol) & (p[:, 1] <= ymax + tol)


def _filtered_cache(cache: SurfaceQuadratureCache, mask: np.ndarray) -> SurfaceQuadratureCache:
    keep = np.asarray(mask, dtype=bool)
    return SurfaceQuadratureCache(
        node_ids=cache.node_ids[keep].copy(),
        weights=cache.weights[keep].copy(),
        area_weights=cache.area_weights[keep].copy(),
    )


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


def build_abaqus_input_text(cfg: FlexibleCubeConfig) -> str:
    lower, upper = make_geometry(cfg)
    lower_contact_ids = _contact_lower_top_element_ids(lower, upper, cfg)
    upper_bottom_ids = _bottom_element_ids(upper)
    upper_top_nodes = np.flatnonzero(np.isclose(upper.X[:, 2], float(np.max(upper.X[:, 2])))) + 1
    lower_bottom_nodes = np.flatnonzero(np.isclose(lower.X[:, 2], float(np.min(lower.X[:, 2])))) + 1
    lines: list[str] = [
        "*Heading",
        "** Two-flexible-body SDF contact validation: frictionless linear contact, no damping.",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
    ]
    for part_name, mesh in (("LOWER", lower), ("UPPER", upper)):
        lines.extend([f"*Part, name={part_name}", "*Node"])
        for node_id, xyz in enumerate(mesh.X, start=1):
            lines.append(f"{node_id}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
        lines.append("*Element, type=C3D8")
        for element_id, element in enumerate(mesh.elements, start=1):
            labels = [int(node) + 1 for node in element]
            lines.append(f"{element_id}, " + ", ".join(str(label) for label in labels))
        lines.append(f"*Elset, elset={part_name}_ALL")
        _append_ids(lines, np.arange(1, mesh.elements.shape[0] + 1, dtype=np.int64))
        material = "LOWER_MAT" if part_name == "LOWER" else "UPPER_MAT"
        lines.extend([f"*Solid Section, elset={part_name}_ALL, material={material}", ",", "*End Part"])
    lines.extend(
        [
            "*Assembly, name=Assembly",
            "*Instance, name=LOWER-1, part=LOWER",
            "*End Instance",
            "*Instance, name=UPPER-1, part=UPPER",
            "*End Instance",
            "*Elset, elset=LOWER_TOP_ASM, instance=LOWER-1",
        ]
    )
    _append_ids(lines, lower_contact_ids)
    lines.append("*Elset, elset=UPPER_BOTTOM_ASM, instance=UPPER-1")
    _append_ids(lines, upper_bottom_ids)
    lines.append("*Nset, nset=LOWER_BOTTOM_ASM, instance=LOWER-1")
    _append_ids(lines, lower_bottom_nodes)
    lines.append("*Nset, nset=UPPER_TOP_ASM, instance=UPPER-1")
    _append_ids(lines, upper_top_nodes)
    lines.extend(
        [
            "*Surface, type=ELEMENT, name=LOWER_TOP_SURF",
            "LOWER_TOP_ASM, S2",
            "*Surface, type=ELEMENT, name=UPPER_BOTTOM_SURF",
            "UPPER_BOTTOM_ASM, S1",
            "*End Assembly",
            "*Material, name=LOWER_MAT",
            "*Density",
            f"{cfg.density:.12e}",
            "*Elastic",
            f"{cfg.lower_young_modulus:.12e}, {cfg.poisson_ratio:.12e}",
            "*Material, name=UPPER_MAT",
            "*Density",
            f"{cfg.density:.12e}",
            "*Elastic",
            f"{cfg.upper_young_modulus:.12e}, {cfg.poisson_ratio:.12e}",
            "*Surface Interaction, name=LINEAR_FRICTIONLESS",
            "*Surface Behavior, pressure-overclosure=LINEAR",
            f"{cfg.pressure_stiffness:.12e}",
            "*Friction",
            "0.",
            "*Contact Pair, interaction=LINEAR_FRICTIONLESS, type=SURFACE TO SURFACE",
            "LOWER_TOP_SURF, UPPER_BOTTOM_SURF",
            "*Amplitude, name=CLOSURE_AMP, time=TOTAL TIME",
            f"0., 0., {cfg.closure_time:.12e}, 1., {cfg.total_time:.12e}, 1.",
            "*Step, name=FLEXIBLE_CUBE_IMPLICIT, nlgeom=NO, inc=10000",
            "*Dynamic, DIRECT, NOHAF",
            f"{cfg.dt:.12e}, {cfg.total_time:.12e}",
            "*Boundary",
            "LOWER_BOTTOM_ASM, 1, 3, 0.",
            "UPPER_TOP_ASM, 1, 2, 0.",
            "*Boundary, amplitude=CLOSURE_AMP",
            f"UPPER_TOP_ASM, 3, 3, {-cfg.closure:.12e}",
            f"*Output, field, time interval={cfg.dt * max(1, cfg.output_stride):.12e}",
            "*Node Output",
            "U, V, RF",
            "*Element Output, directions=YES",
            "S, E, LE",
            f"*Output, history, time interval={cfg.dt * max(1, cfg.output_stride):.12e}",
            "*Energy Output",
            "ALLKE, ALLIE, ALLSE, ALLWK, ETOTAL",
            "*End Step",
        ]
    )
    return "\n".join(lines) + "\n"


def _split_displacement(u: np.ndarray, lower: BoxMesh, upper: BoxMesh) -> tuple[np.ndarray, np.ndarray]:
    lower_dofs = 3 * lower.X.shape[0]
    return u[:lower_dofs].reshape((-1, 3)), u[lower_dofs:].reshape((-1, 3))


def run_sfc(cfg: FlexibleCubeConfig) -> tuple[list[Row], list[Row], float]:
    lower, upper = make_geometry(cfg)
    lower_body = DeformableBody(
        mesh=VolumeMesh(lower.X, lower.elements, element_type="C3D8"),
        material={"E": cfg.lower_young_modulus, "nu": cfg.poisson_ratio},
        density=cfg.density,
    )
    upper_body = DeformableBody(
        mesh=VolumeMesh(upper.X, upper.elements, element_type="C3D8"),
        material={"E": cfg.upper_young_modulus, "nu": cfg.poisson_ratio},
        density=cfg.density,
    )
    lower_dofs = lower_body.n_dofs
    upper_dofs = upper_body.n_dofs
    total_dofs = lower_dofs + upper_dofs
    K = block_diag((assemble_stiffness_matrix(lower_body).tocsr(), assemble_stiffness_matrix(upper_body).tocsr()), format="csr")
    M = block_diag((assemble_mass_matrix(lower_body, kind="consistent").tocsr(), assemble_mass_matrix(upper_body, kind="consistent").tocsr()), format="csr")
    lower_contact_ids = _contact_lower_top_element_ids(lower, upper, cfg)
    lower_contact_quads = _top_quads_for_element_ids(lower, lower_contact_ids)
    upper_bottom_faces = _upper_bottom_triangles(upper)
    contact_cache = quadrilateral_surface_quadrature_cache(lower_contact_quads, lower.X, order=max(1, min(3, cfg.quadrature_order)))
    sdf_workspace = _workspace_from_contact_envelope(lower, upper, lower_contact_quads, cfg)
    beta = 0.25
    gamma = 0.5
    dt = float(cfg.dt)
    c0 = 1.0 / (beta * dt * dt)
    u = np.zeros(total_dofs, dtype=float)
    v = np.zeros(total_dofs, dtype=float)
    a = np.zeros(total_dofs, dtype=float)
    dofs0, u0, v0, a0 = _dirichlet_state(0.0, cfg, lower, upper, lower_dofs)
    u[dofs0] = u0
    v[dofs0] = v0
    # Abaqus reports zero initial reaction before the first dynamic increment.
    # The prescribed boundary acceleration is enforced from the first step.
    a[dofs0] = 0.0
    free = _free_dofs(total_dofs, dofs0)
    upper_top_nodes = np.flatnonzero(np.isclose(upper.X[:, 2], float(np.max(upper.X[:, 2]))))
    upper_top_z_dofs = lower_dofs + 3 * upper_top_nodes + 2
    effective_free = (M * c0 + K).tocsc()[free[:, None], free]
    solve_free = factorized(effective_free) if free.size else None
    lower_top_nodes = np.flatnonzero(np.isclose(lower.X[:, 2], float(np.max(lower.X[:, 2]))))
    history: list[Row] = []
    timing: list[Row] = []
    start = time.perf_counter()
    steps = int(round(cfg.total_time / cfg.dt))
    response = None
    for step in range(steps + 1):
        t = float(step * dt)
        lower_u, upper_u = _split_displacement(u, lower, upper)
        lower_x = lower.X + lower_u
        upper_x = upper.X + upper_u
        all_q = contact_cache.points(lower_x)
        active_cache = _filtered_cache(contact_cache, _footprint_mask(all_q, upper_x, tolerance=-0.25 * cfg.spacing))
        q_points = active_cache.points(lower_x)
        field_t0 = time.perf_counter()
        master_sdf = sdf_workspace.build(upper_x, upper_bottom_faces, q_points)
        field_elapsed = time.perf_counter() - field_t0
        query_t0 = time.perf_counter()
        response = surface_to_surface_field_penalty_response_vectorized(
            lower_x,
            lower_contact_quads,
            master_sdf,
            pressure_stiffness=cfg.pressure_stiffness,
            n_total_dofs=total_dofs,
            slave_x_reference=lower.X,
            quadrature_cache=active_cache,
            slave_dof_offset=0,
            master_dof_offset=lower_dofs,
        )
        query_elapsed = time.perf_counter() - query_t0
        lower_fields = _element_fields(lower, u[:lower_dofs], E=cfg.lower_young_modulus, nu=cfg.poisson_ratio)
        upper_fields = _element_fields(upper, u[lower_dofs:], E=cfg.upper_young_modulus, nu=cfg.poisson_ratio)
        vm = np.concatenate((lower_fields["von_mises"], upper_fields["von_mises"]))
        strain = np.concatenate((lower_fields["engineering_strain_norm"], upper_fields["engineering_strain_norm"]))
        balance = M @ a + K @ u - response.force
        history.append(
            {
                "source": "sfc_lagrangian_sdf_two_flexible",
                "time": t,
                "top_mean_z_displacement": float(np.mean(lower_u[lower_top_nodes, 2])),
                "max_displacement_norm": float(max(np.max(np.linalg.norm(lower_u, axis=1)), np.max(np.linalg.norm(upper_u, axis=1)))),
                "max_von_mises": float(np.max(vm)),
                "p95_von_mises": float(np.percentile(vm, 95.0)),
                "max_strain_norm": float(np.max(strain)),
                "p95_strain_norm": float(np.percentile(strain, 95.0)),
                "min_gap": float(response.min_gap),
                "max_penetration": float(response.max_penetration),
                "active_samples": int(response.active_count),
                "contact_integral_force": float(np.sum(cfg.pressure_stiffness * active_cache.area_weights * np.maximum(-np.asarray(response.gaps), 0.0))),
                "normal_force": float(abs(np.sum(balance[upper_top_z_dofs]))),
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
        next_dofs, next_u_bc, next_v_bc, next_a_bc = _dirichlet_state(next_t, cfg, lower, upper, lower_dofs)
        u_pred = u + dt * v + dt * dt * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a
        u_guess = u_pred.copy()
        u_guess[next_dofs] = next_u_bc
        previous_contact_force: np.ndarray | None = None
        previous_active_count: int | None = None
        for _iteration in range(max(1, cfg.newmark_iterations)):
            lower_guess_u, upper_guess_u = _split_displacement(u_guess, lower, upper)
            lower_guess_x = lower.X + lower_guess_u
            upper_guess_x = upper.X + upper_guess_u
            all_q_next = contact_cache.points(lower_guess_x)
            active_cache_next = _filtered_cache(contact_cache, _footprint_mask(all_q_next, upper_guess_x, tolerance=-0.25 * cfg.spacing))
            q_next = active_cache_next.points(lower_guess_x)
            sdf_next = sdf_workspace.build(upper_guess_x, upper_bottom_faces, q_next)
            response_next = surface_to_surface_field_penalty_response_vectorized(
                lower_guess_x,
                lower_contact_quads,
                sdf_next,
                pressure_stiffness=cfg.pressure_stiffness,
                n_total_dofs=total_dofs,
                slave_x_reference=lower.X,
                quadrature_cache=active_cache_next,
                slave_dof_offset=0,
                master_dof_offset=lower_dofs,
                assemble_stiffness=True,
            )
            f_contact = response_next.force
            a_guess = c0 * (u_guess - u_pred)
            a_guess[next_dofs] = next_a_bc
            residual = M @ a_guess + K @ u_guess - f_contact
            correction = np.zeros(total_dofs, dtype=float)
            residual_norm = float(np.linalg.norm(residual[free])) if free.size else 0.0
            residual_scale = max(
                1.0,
                float(np.linalg.norm((M @ a_guess)[free])) if free.size else 0.0,
                float(np.linalg.norm((K @ u_guess)[free])) if free.size else 0.0,
                float(np.linalg.norm(f_contact[free])) if free.size else 0.0,
            )
            force_delta = 0.0 if previous_contact_force is None else float(np.linalg.norm(f_contact - previous_contact_force))
            force_scale = max(1.0, float(np.linalg.norm(f_contact)))
            active_stable = previous_active_count is not None and int(response_next.active_count) == int(previous_active_count)
            if solve_free is not None:
                tangent = (M * c0 + K + response_next.stiffness).tocsc()
                correction[free] = np.asarray(spsolve(tangent[free[:, None], free], -residual[free]), dtype=float)
            u_guess[free] += correction[free]
            u_guess[next_dofs] = next_u_bc
            correction_norm = float(np.linalg.norm(correction[free])) if free.size else 0.0
            correction_scale = max(1.0, float(np.linalg.norm(u_guess[free])) if free.size else 0.0)
            converged = (
                residual_norm <= float(cfg.residual_tolerance) * residual_scale
                and correction_norm <= float(cfg.contact_tolerance) * correction_scale
                and force_delta <= float(cfg.force_tolerance) * force_scale
                and (active_stable or int(response_next.active_count) == 0)
            )
            previous_contact_force = f_contact.copy()
            previous_active_count = int(response_next.active_count)
            if converged:
                break
        u = u_guess
        a = c0 * (u - u_pred)
        v = v_pred + gamma * dt * a
        u[next_dofs] = next_u_bc
        v[next_dofs] = next_v_bc
        a[next_dofs] = next_a_bc
    return history, timing, time.perf_counter() - start


def _abaqus_frame_metrics_export_script() -> str:
    return r'''
from __future__ import print_function

import csv
import math
import sys

from odbAccess import openOdb


def norm3(data):
    return math.sqrt(float(data[0]) ** 2 + float(data[1]) ** 2 + float(data[2]) ** 2)


def tensor_norm(data):
    return math.sqrt(sum(float(v) * float(v) for v in data))


def von_mises(data):
    s11, s22, s33, s12, s13, s23 = [float(v) for v in data]
    return math.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * (s12 * s12 + s13 * s13 + s23 * s23))


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
    assembly = odb.rootAssembly
    lower = assembly.instances["LOWER-1"]
    upper = assembly.instances["UPPER-1"]
    lower_zmax = max(float(node.coordinates[2]) for node in lower.nodes)
    upper_zmax = max(float(node.coordinates[2]) for node in upper.nodes)
    lower_top = set(int(node.label) for node in lower.nodes if abs(float(node.coordinates[2]) - lower_zmax) < 1.0e-8)
    upper_top = set(int(node.label) for node in upper.nodes if abs(float(node.coordinates[2]) - upper_zmax) < 1.0e-8)
    step = odb.steps[list(odb.steps.keys())[0]]
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time",
                "top_mean_z_displacement",
                "max_displacement_norm",
                "normal_force",
                "p95_von_mises",
                "max_von_mises",
                "p95_strain_norm",
                "max_strain_norm",
            ],
        )
        writer.writeheader()
        for frame in step.frames:
            u_values = []
            lower_top_u3 = []
            if "U" in frame.fieldOutputs:
                for value in frame.fieldOutputs["U"].values:
                    inst = getattr(value, "instance", None)
                    if inst is None or inst.name.upper() not in ("LOWER-1", "UPPER-1"):
                        continue
                    u_values.append(norm3(value.data))
                    if inst.name.upper() == "LOWER-1" and int(value.nodeLabel) in lower_top:
                        lower_top_u3.append(float(value.data[2]))
            rf3 = 0.0
            if "RF" in frame.fieldOutputs:
                for value in frame.fieldOutputs["RF"].values:
                    inst = getattr(value, "instance", None)
                    if inst is not None and inst.name.upper() == "UPPER-1" and int(value.nodeLabel) in upper_top:
                        rf3 += float(value.data[2])
            vm_values = []
            if "S" in frame.fieldOutputs:
                for value in frame.fieldOutputs["S"].values:
                    inst = getattr(value, "instance", None)
                    if inst is not None and inst.name.upper() in ("LOWER-1", "UPPER-1"):
                        vm_values.append(von_mises(value.data))
            strain_values = []
            if "E" in frame.fieldOutputs:
                field = frame.fieldOutputs["E"]
            elif "LE" in frame.fieldOutputs:
                field = frame.fieldOutputs["LE"]
            else:
                field = None
            if field is not None:
                for value in field.values:
                    inst = getattr(value, "instance", None)
                    if inst is not None and inst.name.upper() in ("LOWER-1", "UPPER-1"):
                        strain_values.append(tensor_norm(value.data))
            writer.writerow(
                {
                    "time": float(frame.frameValue),
                    "top_mean_z_displacement": sum(lower_top_u3) / len(lower_top_u3) if lower_top_u3 else "",
                    "max_displacement_norm": max(u_values) if u_values else "",
                    "normal_force": abs(rf3),
                    "p95_von_mises": percentile(vm_values, 95.0),
                    "max_von_mises": max(vm_values) if vm_values else "",
                    "p95_strain_norm": percentile(strain_values, 95.0),
                    "max_strain_norm": max(strain_values) if strain_values else "",
                }
            )
finally:
    odb.close()
'''


def run_abaqus(cfg: FlexibleCubeConfig, out_dir: Path, *, abaqus_command: str | None, skip_abaqus: bool) -> tuple[Path, Path, Row]:
    run_dir = out_dir / "abaqus_run"
    vtk_dir = out_dir / "abaqus_vtk"
    run_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir.mkdir(parents=True, exist_ok=True)
    inp = run_dir / f"{JOB_NAME}.inp"
    metrics_csv = out_dir / "abaqus_frame_metrics.csv"
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
        metrics_script = run_dir / "export_frame_metrics.py"
        metrics_script.write_text(_abaqus_frame_metrics_export_script(), encoding="ascii")
        metrics_wall = _run_command(
            [command, "python", str(metrics_script.resolve()), str(odb.resolve()), str(metrics_csv.resolve())],
            cwd=run_dir,
            log_path=out_dir / "abaqus_frame_metrics_stdout.log",
        )
    else:
        wall = 0.0
        export_wall = 0.0
        metrics_wall = 0.0
    manifest = vtk_dir / "frame_manifest.csv"
    if skip_abaqus and not metrics_csv.exists():
        raise FileNotFoundError(f"--skip-abaqus requires existing Abaqus metrics: {metrics_csv}")
    if skip_abaqus and not manifest.exists():
        raise FileNotFoundError(f"--skip-abaqus requires existing VTK manifest: {manifest}")
    if not skip_abaqus and not manifest.exists():
        export_odb_to_vtk(run_dir / f"{JOB_NAME}.odb", vtk_dir, stem="frame")
    reported = _abaqus_reported_wallclock_seconds(run_dir / f"{JOB_NAME}.sta")
    return manifest, metrics_csv, {
        "solver": "abaqus_standard_flexible_cube",
        "analysis_wall_seconds": "" if skip_abaqus else float(wall),
        "reported_wall_seconds": "" if reported is None else float(reported),
        "export_wall_seconds": "" if skip_abaqus else float(export_wall),
        "frame_metrics_wall_seconds": "" if skip_abaqus else float(metrics_wall),
    }


def load_abaqus_history(metrics_csv: Path) -> list[Row]:
    with metrics_csv.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    out: list[Row] = []
    for row in rows:
        item: Row = {"source": "abaqus_standard_flexible_cube", "time": float(row["time"])}
        for key in (
            "top_mean_z_displacement",
            "max_displacement_norm",
            "normal_force",
            "p95_von_mises",
            "max_von_mises",
            "p95_strain_norm",
            "max_strain_norm",
        ):
            item[key] = "" if row.get(key, "") == "" else float(row[key])
        out.append(item)
    return out


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
        ("top_mean_z_displacement", "lower top mean z displacement"),
        ("max_displacement_norm", "max displacement norm"),
        ("normal_force", "upper-top reaction force"),
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
    path = out_dir / "flexible_cube_sdf_abaqus_curves.png"
    fig.savefig(path, dpi=300)
    fig.savefig(out_dir / "flexible_cube_sdf_abaqus_curves.pdf")
    plt.close(fig)
    return path


def run_workflow(out_dir: Path, *, cfg: FlexibleCubeConfig, abaqus_command: str | None, skip_abaqus: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, abaqus_metrics_csv, abaqus_timing = run_abaqus(cfg, out_dir, abaqus_command=abaqus_command, skip_abaqus=skip_abaqus)
    abaqus_rows = load_abaqus_history(abaqus_metrics_csv)
    sfc_rows, sfc_timing_rows, sfc_wall = run_sfc(cfg)
    metrics = [
        _metric(sfc_rows, abaqus_rows, key)
        for key in ("top_mean_z_displacement", "max_displacement_norm", "normal_force", "p95_von_mises", "p95_strain_norm")
    ]
    timing = [
        abaqus_timing,
        {
            "solver": "sfc_lagrangian_sdf_two_flexible",
            "analysis_wall_seconds": float(sfc_wall),
            "reported_wall_seconds": "",
            "export_wall_seconds": "",
            "frame_metrics_wall_seconds": "",
        },
    ]
    _write_csv(out_dir / "flexible_cube_sdf_abaqus_history.csv", abaqus_rows + sfc_rows)
    _write_csv(out_dir / "flexible_cube_sdf_abaqus_metrics.csv", metrics)
    _write_csv(out_dir / "flexible_cube_sdf_abaqus_timing.csv", timing)
    _write_csv(out_dir / "flexible_cube_sdf_field_timing.csv", sfc_timing_rows)
    plot = _plot_curves(out_dir, abaqus_rows, sfc_rows, metrics)
    summary = out_dir / "flexible_cube_sdf_abaqus_summary.md"
    lines = [
        "# Two-Flexible-Body SDF Contact Comparison",
        "",
        f"- Lower mesh: `{cfg.lower_nx}x{cfg.lower_ny}x{cfg.lower_nz}` C3D8",
        f"- Upper mesh: `{cfg.upper_nx}x{cfg.upper_ny}x{cfg.upper_nz}` C3D8",
        f"- Duration: `{cfg.total_time}` s",
        f"- Time step: `{cfg.dt}` s",
        f"- Contact stiffness: `{cfg.pressure_stiffness}`",
        "- Contact: frictionless linear pressure-overclosure, two flexible bodies",
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
        "history": out_dir / "flexible_cube_sdf_abaqus_history.csv",
        "metrics": out_dir / "flexible_cube_sdf_abaqus_metrics.csv",
        "timing": out_dir / "flexible_cube_sdf_abaqus_timing.csv",
        "field_timing": out_dir / "flexible_cube_sdf_field_timing.csv",
        "plot": plot,
        "abaqus_manifest": manifest,
        "abaqus_frame_metrics": abaqus_metrics_csv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "flexible_cube_sdf_abaqus")
    parser.add_argument("--abaqus-command", default=None)
    parser.add_argument("--skip-abaqus", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--total-time", type=float, default=0.04)
    parser.add_argument("--closure", type=float, default=0.20)
    parser.add_argument("--initial-gap", type=float, default=-0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FlexibleCubeConfig(
        lower_nx=8 if args.quick else 12,
        lower_ny=4 if args.quick else 6,
        lower_nz=2 if args.quick else 3,
        upper_nx=4 if args.quick else 6,
        upper_ny=4 if args.quick else 6,
        upper_nz=1 if args.quick else 2,
        dt=float(args.dt),
        total_time=float(args.total_time),
        closure_time=0.5 * float(args.total_time),
        closure=float(args.closure),
        initial_gap=float(args.initial_gap),
    )
    outputs = run_workflow(args.out_dir, cfg=cfg, abaqus_command=args.abaqus_command, skip_abaqus=bool(args.skip_abaqus))
    print("Two-flexible-body SDF comparison complete.")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
