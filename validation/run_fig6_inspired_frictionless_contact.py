"""Fig. 6-inspired frictionless C3D8 SDF-contact validation.

The geometry follows the spirit of the combined loading sketch in the
FuzzyContact paper: a small rectangular upper body approaches a wider lower
body and is then laterally shifted.  This runner deliberately keeps the contact
law frictionless.  It is used for three-dimensional field/cloud visualization
and SDF-field inspection, not for a stick-slip claim.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
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
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from sfc.contact.field_contact import (  # noqa: E402
    surface_to_surface_field_penalty_response_vectorized,
    triangle_surface_quadrature_cache,
)
from sfc.fem import DeformableBody, assemble_stiffness_matrix  # noqa: E402
from sfc.fem.hex8 import hex8_center_strain_stress  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402
from validation.run_phase9_full_contact_validation import _von_mises, _write_legacy_hex_vtk  # noqa: E402

Row = dict[str, Any]

_HEX_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


@dataclass(slots=True)
class BoxMesh:
    X: np.ndarray
    elements: np.ndarray
    nx: int
    ny: int
    nz: int
    origin: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(slots=True)
class StageResult:
    name: str
    u: np.ndarray
    master_sdf: DynamicNarrowBandSDF
    fields: dict[str, np.ndarray]
    rows: list[Row]
    driver_x: np.ndarray
    driver_faces: np.ndarray
    quadrature_points: np.ndarray
    quadrature_gaps: np.ndarray


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


def _structured_hex_box(
    *,
    nx: int,
    ny: int,
    nz: int,
    origin: tuple[float, float, float],
    size: tuple[float, float, float],
) -> BoxMesh:
    nodes: list[tuple[float, float, float]] = []
    node_id: dict[tuple[int, int, int], int] = {}
    ox, oy, oz = origin
    lx, ly, lz = size
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                node_id[(i, j, k)] = len(nodes)
                nodes.append((ox + lx * i / nx, oy + ly * j / ny, oz + lz * k / nz))
    elements: list[tuple[int, ...]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements.append(
                    (
                        node_id[(i, j, k)],
                        node_id[(i + 1, j, k)],
                        node_id[(i + 1, j + 1, k)],
                        node_id[(i, j + 1, k)],
                        node_id[(i, j, k + 1)],
                        node_id[(i + 1, j, k + 1)],
                        node_id[(i + 1, j + 1, k + 1)],
                        node_id[(i, j + 1, k + 1)],
                    )
                )
    return BoxMesh(
        X=np.asarray(nodes, dtype=float),
        elements=np.asarray(elements, dtype=np.int64),
        nx=nx,
        ny=ny,
        nz=nz,
        origin=origin,
        size=size,
    )


def _surface_grid(
    *,
    nx: int,
    ny: int,
    origin: tuple[float, float, float],
    size: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    ox, oy, oz = origin
    lx, ly = size
    nodes: list[tuple[float, float, float]] = []
    node_id: dict[tuple[int, int], int] = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            node_id[(i, j)] = len(nodes)
            nodes.append((ox + lx * i / nx, oy + ly * j / ny, oz))
    faces: list[tuple[int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            n00 = node_id[(i, j)]
            n10 = node_id[(i + 1, j)]
            n11 = node_id[(i + 1, j + 1)]
            n01 = node_id[(i, j + 1)]
            faces.append((n00, n10, n11))
            faces.append((n00, n11, n01))
    return np.asarray(nodes, dtype=float), np.asarray(faces, dtype=np.int64)


def _top_triangles(mesh: BoxMesh) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for j in range(mesh.ny):
        for i in range(mesh.nx):
            eid = (mesh.nz - 1) * mesh.ny * mesh.nx + j * mesh.nx + i
            element = mesh.elements[eid]
            quad = element[[4, 5, 6, 7]]
            faces.append((int(quad[0]), int(quad[1]), int(quad[2])))
            faces.append((int(quad[0]), int(quad[2]), int(quad[3])))
    return np.asarray(faces, dtype=np.int64)


def _boundary_faces(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    owners: dict[tuple[int, ...], tuple[np.ndarray, int]] = {}
    counts: dict[tuple[int, ...], int] = {}
    for eid, element in enumerate(np.asarray(elements, dtype=np.int64)):
        for face_nodes in _HEX_FACES:
            face = np.asarray([element[i] for i in face_nodes], dtype=np.int64)
            key = tuple(sorted(int(node) for node in face))
            counts[key] = counts.get(key, 0) + 1
            owners.setdefault(key, (face, eid))
    faces: list[np.ndarray] = []
    face_owners: list[int] = []
    for key, count in counts.items():
        if count == 1:
            face, owner = owners[key]
            faces.append(face)
            face_owners.append(owner)
    return np.asarray(faces, dtype=np.int64), np.asarray(face_owners, dtype=np.int64)


def _fixed_bottom_dofs(mesh: BoxMesh) -> np.ndarray:
    zmin = float(np.min(mesh.X[:, 2]))
    nodes = np.flatnonzero(np.isclose(mesh.X[:, 2], zmin))
    return np.asarray([3 * int(node) + comp for node in nodes for comp in range(3)], dtype=np.int64)


def _free_dofs(n_dofs: int, fixed: np.ndarray) -> np.ndarray:
    return np.setdiff1d(np.arange(n_dofs, dtype=np.int64), np.asarray(fixed, dtype=np.int64))


def _element_fields(mesh: BoxMesh, u: np.ndarray, *, E: float, nu: float) -> dict[str, np.ndarray]:
    U = np.asarray(u, dtype=float).reshape((-1, 3))
    disp = np.asarray([float(np.mean(np.linalg.norm(U[element], axis=1))) for element in mesh.elements], dtype=float)
    strain_norm: list[float] = []
    vm: list[float] = []
    for element in mesh.elements:
        strain, stress = hex8_center_strain_stress(mesh.X[element], U[element], E, nu)
        strain_norm.append(float(np.linalg.norm(strain)))
        vm.append(_von_mises(stress))
    return {
        "displacement_magnitude": disp,
        "engineering_strain_norm": np.asarray(strain_norm, dtype=float),
        "von_mises": np.asarray(vm, dtype=float),
    }


def _top_cell_ids_for_points(mesh: BoxMesh, points: np.ndarray) -> np.ndarray:
    ox, oy, _oz = mesh.origin
    lx, ly, _lz = mesh.size
    p = np.asarray(points, dtype=float)
    ii = np.floor((p[:, 0] - ox) / lx * mesh.nx).astype(np.int64)
    jj = np.floor((p[:, 1] - oy) / ly * mesh.ny).astype(np.int64)
    ii = np.clip(ii, 0, mesh.nx - 1)
    jj = np.clip(jj, 0, mesh.ny - 1)
    return (mesh.nz - 1) * mesh.nx * mesh.ny + jj * mesh.nx + ii


def _solve_stage(
    mesh: BoxMesh,
    driver_ref: np.ndarray,
    driver_faces: np.ndarray,
    *,
    stage_name: str,
    closure: float,
    shift_x: float,
    spacing: float,
    band_radius: float,
    E: float,
    nu: float,
    pressure_stiffness: float,
    iterations: int,
) -> StageResult:
    volume = VolumeMesh(mesh.X, mesh.elements, element_type="C3D8")
    body = DeformableBody(mesh=volume, material={"E": E, "nu": nu}, density=1.0)
    K = assemble_stiffness_matrix(body).tocsr()
    fixed = _fixed_bottom_dofs(mesh)
    free = _free_dofs(body.n_dofs, fixed)
    top_faces = _top_triangles(mesh)
    cache = triangle_surface_quadrature_cache(driver_faces, driver_ref, order=7)
    u = np.zeros(body.n_dofs, dtype=float)
    rows: list[Row] = []
    master_sdf: DynamicNarrowBandSDF | None = None
    response = None
    driver_x = driver_ref.copy()
    driver_x[:, 0] += float(shift_x)
    driver_x[:, 2] -= float(closure)
    q_points = cache.points(driver_x)
    for iteration in range(1, iterations + 1):
        x_master = mesh.X + u.reshape((-1, 3))
        q_points = cache.points(driver_x)
        master_sdf = DynamicNarrowBandSDF.build_required_points(
            x_master,
            top_faces,
            q_points,
            spacing=spacing,
            band_radius=band_radius,
            padding=band_radius,
            cell_size=max(2.0 * spacing, band_radius),
        )
        n_slave_dofs = 3 * driver_x.shape[0]
        response = surface_to_surface_field_penalty_response_vectorized(
            driver_x,
            driver_faces,
            master_sdf,
            pressure_stiffness=pressure_stiffness,
            n_total_dofs=n_slave_dofs + body.n_dofs,
            slave_x_reference=driver_ref,
            quadrature_cache=cache,
            slave_dof_offset=0,
            master_dof_offset=n_slave_dofs,
        )
        force = response.force[n_slave_dofs:]
        rhs = force[free]
        u_new = np.zeros_like(u)
        u_new[free] = np.asarray(spsolve(K[free[:, None], free].tocsc(), rhs), dtype=float)
        correction = float(np.linalg.norm(u_new - u))
        u = 0.65 * u_new + 0.35 * u
        rows.append(
            {
                "stage": stage_name,
                "iteration": iteration,
                "closure_mm": float(closure),
                "shift_x_mm": float(shift_x),
                "min_gap_mm": float(response.min_gap),
                "max_penetration_mm": float(response.max_penetration),
                "active_samples": int(response.active_count),
                "normal_force": float(np.sum(pressure_stiffness * cache.area_weights * np.maximum(-np.asarray(response.gaps), 0.0))),
                "correction_norm": correction,
                "field_update_seconds": float(master_sdf.stats.update_seconds),
                "quadrature_points": int(cache.area_weights.size),
            }
        )
        if correction <= 1.0e-10 * max(1.0, float(np.linalg.norm(u_new))):
            u = u_new
            break
    if master_sdf is None or response is None:
        raise RuntimeError("stage solve did not produce a contact response")
    fields = _element_fields(mesh, u, E=E, nu=nu)
    q_points = cache.points(driver_x)
    gaps = np.asarray(response.gaps, dtype=float)
    pressure = pressure_stiffness * np.maximum(-gaps, 0.0)
    top_ids = _top_cell_ids_for_points(mesh, q_points)
    gap_sum = np.zeros(mesh.elements.shape[0], dtype=float)
    pressure_sum = np.zeros(mesh.elements.shape[0], dtype=float)
    count = np.zeros(mesh.elements.shape[0], dtype=float)
    active = np.zeros(mesh.elements.shape[0], dtype=float)
    for eid, gap, p in zip(top_ids, gaps, pressure, strict=True):
        gap_sum[eid] += float(gap)
        pressure_sum[eid] += float(p)
        count[eid] += 1.0
        active[eid] = max(active[eid], 1.0 if p > 0.0 else 0.0)
    denom = np.maximum(count, 1.0)
    fields["sdf_gap"] = gap_sum / denom
    fields["contact_pressure"] = pressure_sum / denom
    fields["active_contact"] = active
    return StageResult(
        name=stage_name,
        u=u,
        master_sdf=master_sdf,
        fields=fields,
        rows=rows,
        driver_x=driver_x,
        driver_faces=driver_faces,
        quadrature_points=q_points,
        quadrature_gaps=gaps,
    )


def _plot_stage_cloud(path_base: Path, mesh: BoxMesh, result: StageResult) -> dict[str, Path]:
    figures = path_base / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    faces, owners = _boundary_faces(mesh.elements)
    x_plot = mesh.X + result.u.reshape((-1, 3))
    panels = [
        ("displacement_magnitude", "displacement magnitude"),
        ("engineering_strain_norm", "strain norm"),
        ("von_mises", "von Mises"),
        ("sdf_gap", "SDF gap"),
        ("contact_pressure", "contact pressure"),
        ("active_contact", "active contact"),
    ]
    fig = plt.figure(figsize=(10.5, 6.8))
    for idx, (key, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        values = np.asarray(result.fields[key], dtype=float)
        face_values = values[owners]
        vmin = float(np.min(face_values))
        vmax = float(np.max(face_values))
        if abs(vmax - vmin) < 1.0e-14:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        collection = Poly3DCollection(
            [x_plot[face] for face in faces],
            cmap="viridis",
            norm=norm,
            linewidths=0.08,
            edgecolors=(0.1, 0.1, 0.1, 0.12),
        )
        collection.set_array(face_values)
        ax.add_collection3d(collection)
        bounds = x_plot[faces.reshape(-1)]
        mins = bounds.min(axis=0)
        maxs = bounds.max(axis=0)
        pad = np.maximum(0.04 * (maxs - mins), 1.0e-5)
        ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
        ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
        ax.set_zlim(float(mins[2] - pad[2]), float(maxs[2] + pad[2]))
        ax.set_box_aspect((maxs - mins + 2.0 * pad).clip(min=1.0e-12))
        ax.view_init(elev=22.0, azim=-48.0)
        ax.set_axis_off()
        ax.set_title(title, pad=3)
        fig.colorbar(collection, ax=ax, shrink=0.62, pad=0.02)
    fig.tight_layout()
    png = figures / f"fig6_inspired_{result.name}_clouds.png"
    pdf = figures / f"fig6_inspired_{result.name}_clouds.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def _plot_sdf_visualization(path_base: Path, mesh: BoxMesh, result: StageResult) -> dict[str, Path]:
    figures = path_base / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    x_master = mesh.X + result.u.reshape((-1, 3))
    top_faces = _top_triangles(mesh)
    top_quad_faces = [face for face in _boundary_faces(mesh.elements)[0] if np.all(np.isclose(mesh.X[face, 2], mesh.origin[2] + mesh.size[2]))]
    fig = plt.figure(figsize=(10.2, 7.2))

    ax0 = fig.add_subplot(2, 2, 1, projection="3d")
    top_collection = Poly3DCollection(
        [x_master[tri] for tri in top_faces],
        cmap="viridis",
        linewidths=0.08,
        edgecolors=(0.1, 0.1, 0.1, 0.18),
    )
    sample_pressure = np.maximum(-result.quadrature_gaps, 0.0)
    face_values = np.zeros(top_faces.shape[0], dtype=float)
    if sample_pressure.size:
        face_values[:] = float(np.max(sample_pressure))
    top_collection.set_array(face_values)
    ax0.add_collection3d(top_collection)
    driver_collection = Poly3DCollection(
        [result.driver_x[face] for face in result.driver_faces],
        facecolors=(0.2, 0.2, 0.2, 0.10),
        edgecolors=(0.0, 0.0, 0.0, 0.25),
        linewidths=0.12,
    )
    ax0.add_collection3d(driver_collection)
    bounds = np.vstack([x_master[top_faces.reshape(-1)], result.driver_x])
    mins = bounds.min(axis=0)
    maxs = bounds.max(axis=0)
    pad = np.maximum(0.05 * (maxs - mins), 1.0e-5)
    ax0.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    ax0.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
    ax0.set_zlim(float(mins[2] - pad[2]), float(maxs[2] + pad[2]))
    ax0.set_box_aspect((maxs - mins + 2.0 * pad).clip(min=1.0e-12))
    ax0.view_init(elev=24.0, azim=-48.0)
    ax0.set_axis_off()
    ax0.set_title("(a) deformed surface and driver", loc="left", fontsize=10)

    ax1 = fig.add_subplot(2, 2, 2)
    xs = np.linspace(mesh.origin[0], mesh.origin[0] + mesh.size[0], 160)
    zs = np.linspace(mesh.origin[2] + mesh.size[2] - 0.45, mesh.origin[2] + mesh.size[2] + 0.25, 100)
    yy = mesh.origin[1] + 0.5 * mesh.size[1]
    phi = np.full((zs.size, xs.size), np.nan, dtype=float)
    for iz, z in enumerate(zs):
        for ix, x in enumerate(xs):
            try:
                phi[iz, ix] = result.master_sdf.query_phi(np.asarray([x, yy, z], dtype=float))
            except ValueError:
                continue
    im = ax1.imshow(
        phi,
        extent=(xs[0], xs[-1], zs[0], zs[-1]),
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
    )
    ax1.contour(xs, zs, np.nan_to_num(phi, nan=1.0), levels=[0.0], colors="black", linewidths=1.0)
    ax1.set_title("(b) SDF $\\phi$ slice", loc="left", fontsize=10)
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("z (mm)")
    fig.colorbar(im, ax=ax1, shrink=0.82)

    ax2 = fig.add_subplot(2, 2, 3)
    valid = np.isfinite(phi).astype(float)
    im2 = ax2.imshow(
        valid,
        extent=(xs[0], xs[-1], zs[0], zs[-1]),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax2.set_title("(c) narrow-band validity", loc="left", fontsize=10)
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("z (mm)")
    fig.colorbar(im2, ax=ax2, shrink=0.82)

    ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    q = result.quadrature_points
    gaps = result.quadrature_gaps
    norm = Normalize(vmin=float(np.min(gaps)), vmax=float(np.max(gaps)) if np.max(gaps) > np.min(gaps) else float(np.min(gaps) + 1.0))
    collection = Poly3DCollection(
        [result.driver_x[face] for face in result.driver_faces],
        cmap="coolwarm",
        norm=norm,
        linewidths=0.10,
        edgecolors=(0.1, 0.1, 0.1, 0.18),
    )
    face_gaps_list: list[float] = []
    for face in result.driver_faces:
        centroid = np.mean(result.driver_x[face], axis=0)
        try:
            face_gaps_list.append(float(result.master_sdf.query_phi(centroid)))
        except ValueError:
            face_gaps_list.append(np.nan)
    face_gaps = np.asarray(face_gaps_list, dtype=float)
    if np.all(~np.isfinite(face_gaps)):
        face_gaps = np.zeros(result.driver_faces.shape[0], dtype=float)
    else:
        fill = float(np.nanmax(face_gaps))
        face_gaps = np.nan_to_num(face_gaps, nan=fill)
    collection.set_array(face_gaps)
    ax3.add_collection3d(collection)
    if q.size:
        step = max(1, q.shape[0] // 45)
        for point in q[::step]:
            try:
                normal = result.master_sdf.query_gap_normal(point)[1]
            except ValueError:
                continue
            ax3.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=0.18, color="black", linewidth=0.45)
    ax3.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    ax3.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
    ax3.set_zlim(float(mins[2] - pad[2]), float(maxs[2] + pad[2]))
    ax3.set_box_aspect((maxs - mins + 2.0 * pad).clip(min=1.0e-12))
    ax3.view_init(elev=24.0, azim=-48.0)
    ax3.set_axis_off()
    ax3.set_title("(d) driver gap and scalar-gradient normals", loc="left", fontsize=10)
    fig.colorbar(collection, ax=ax3, shrink=0.72, pad=0.02)

    fig.tight_layout()
    png = figures / "fig6_inspired_sdf_field_visualization.png"
    pdf = figures / "fig6_inspired_sdf_field_visualization.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"sdf_png": png, "sdf_pdf": pdf}


def run_validation(
    out_dir: Path,
    *,
    quick: bool = False,
    nx: int = 24,
    ny: int = 12,
    nz: int = 4,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if quick:
        nx, ny, nz = 8, 4, 2
    mesh = _structured_hex_box(
        nx=nx,
        ny=ny,
        nz=nz,
        origin=(-10.0, -5.0, 0.0),
        size=(20.0, 10.0, 2.0),
    )
    driver_ref, driver_faces = _surface_grid(
        nx=max(4, nx // 3),
        ny=max(4, ny // 2),
        origin=(-2.25, -2.25, 2.065),
        size=(4.5, 4.5),
    )
    E = 1200.0
    nu = 0.30
    pressure_stiffness = 4500.0
    spacing = 0.35 if quick else 0.25
    band_radius = 0.90
    start = perf_counter()
    normal = _solve_stage(
        mesh,
        driver_ref,
        driver_faces,
        stage_name="normal_loading",
        closure=0.105,
        shift_x=0.0,
        spacing=spacing,
        band_radius=band_radius,
        E=E,
        nu=nu,
        pressure_stiffness=pressure_stiffness,
        iterations=3 if quick else 6,
    )
    shifted = _solve_stage(
        mesh,
        driver_ref,
        driver_faces,
        stage_name="shifted_frictionless",
        closure=0.105,
        shift_x=2.0,
        spacing=spacing,
        band_radius=band_radius,
        E=E,
        nu=nu,
        pressure_stiffness=pressure_stiffness,
        iterations=3 if quick else 6,
    )
    wall = perf_counter() - start
    rows = normal.rows + shifted.rows
    for row in rows:
        row["quick"] = str(bool(quick)).lower()
        row["element_type"] = "C3D8"
        row["lower_mesh"] = f"{nx}x{ny}x{nz}"
        row["wall_time_seconds"] = wall
    outputs: dict[str, Path] = {
        "metrics": out_dir / "fig6_inspired_frictionless_metrics.csv",
        "normal_vtk": out_dir / "vtk" / "fig6_inspired_normal_loading.vtk",
        "shifted_vtk": out_dir / "vtk" / "fig6_inspired_shifted_frictionless.vtk",
        "summary": out_dir / "fig6_inspired_frictionless_summary.md",
    }
    _write_csv(outputs["metrics"], rows)
    _write_legacy_hex_vtk(outputs["normal_vtk"], mesh.X + normal.u.reshape((-1, 3)), mesh.elements, cell_values=normal.fields)
    _write_legacy_hex_vtk(outputs["shifted_vtk"], mesh.X + shifted.u.reshape((-1, 3)), mesh.elements, cell_values=shifted.fields)
    normal_figs = _plot_stage_cloud(out_dir, mesh, normal)
    shifted_figs = _plot_stage_cloud(out_dir, mesh, shifted)
    sdf_figs = _plot_sdf_visualization(out_dir, mesh, shifted)
    outputs["normal_cloud_png"] = normal_figs["png"]
    outputs["normal_cloud_pdf"] = normal_figs["pdf"]
    outputs["shifted_cloud_png"] = shifted_figs["png"]
    outputs["shifted_cloud_pdf"] = shifted_figs["pdf"]
    outputs.update(sdf_figs)
    last_normal = normal.rows[-1]
    last_shifted = shifted.rows[-1]
    lines = [
        "# Fig. 6-Inspired Frictionless C3D8 Contact",
        "",
        "This case uses the FuzzyContact Fig. 6 loading sketch only as geometry/loading inspiration.",
        "The contact law is frictionless; no stick-slip claim is made.",
        "",
        f"- lower C3D8 mesh: `{nx} x {ny} x {nz}` elements",
        f"- driver surface triangles: `{driver_faces.shape[0]}`",
        f"- dynamic SDF spacing: `{spacing}` mm",
        f"- wall time: `{wall:.6f}` s",
        f"- normal-stage min gap: `{float(last_normal['min_gap_mm']):.6e}` mm",
        f"- shifted-stage min gap: `{float(last_shifted['min_gap_mm']):.6e}` mm",
        f"- shifted-stage active samples: `{int(last_shifted['active_samples'])}`",
        "",
        "## Outputs",
        "",
    ]
    for key, path in outputs.items():
        if key != "summary":
            lines.append(f"- `{path.relative_to(out_dir).as_posix()}`")
    outputs["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "fig6_inspired_frictionless_contact")
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--nz", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_validation(args.out_dir, quick=bool(args.quick), nx=int(args.nx), ny=int(args.ny), nz=int(args.nz))
    print("Fig. 6-inspired frictionless contact validation complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
