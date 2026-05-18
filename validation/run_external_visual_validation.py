"""Generate external visual FEM validation artifacts for the dynamic SDF backend.

The quick validation uses a generated external reference field rather than
calling CalculiX. This keeps the core SDF/FEM package independent of external
solver binaries while still producing paper-facing visual audit artifacts:
3D boundary-surface displacement, stress, strain, contact pressure/gap,
absolute error clouds, and dynamic SDF field visualizations.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402
from sfc.contact import triangle_surface_quadrature_cache  # noqa: E402
from sfc.mesh import extract_boundary_triangles  # noqa: E402

Row = dict[str, Any]


def _configure_times_fonts(matplotlib_module: Any) -> None:
    matplotlib_module.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


@dataclass(frozen=True, slots=True)
class HexMesh:
    """Structured HEX8 visualization mesh."""

    nodes: np.ndarray
    cells: np.ndarray
    shape: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class VisualFields:
    """Point-wise fields used for the visual comparison."""

    displacement: np.ndarray
    displacement_magnitude: np.ndarray
    strain_norm: np.ndarray
    von_mises: np.ndarray
    gap: np.ndarray
    contact_pressure: np.ndarray
    active_contact: np.ndarray


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


def _hex_mesh(nx: int, ny: int, nz: int) -> HexMesh:
    nodes: list[list[float]] = []
    for k in range(nz + 1):
        z = 0.35 * k / nz
        for j in range(ny + 1):
            y = j / ny
            for i in range(nx + 1):
                x = i / nx
                nodes.append([x, y, z])

    def node_id(i: int, j: int, k: int) -> int:
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    cells: list[list[int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cells.append(
                    [
                        node_id(i, j, k),
                        node_id(i + 1, j, k),
                        node_id(i + 1, j + 1, k),
                        node_id(i, j + 1, k),
                        node_id(i, j, k + 1),
                        node_id(i + 1, j, k + 1),
                        node_id(i + 1, j + 1, k + 1),
                        node_id(i, j + 1, k + 1),
                    ]
                )
    return HexMesh(
        nodes=np.asarray(nodes, dtype=float),
        cells=np.asarray(cells, dtype=np.int64),
        shape=(nx, ny, nz),
    )


def _reference_displacement(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    zeta = points[:, 2] / 0.35
    u = np.zeros_like(points)
    u[:, 0] = 0.0035 * zeta * np.sin(math.pi * y)
    u[:, 1] = -0.0025 * zeta * np.sin(math.pi * x)
    u[:, 2] = -0.018 * (1.0 - zeta) * (1.0 + 0.35 * np.cos(2.0 * math.pi * x) * np.cos(2.0 * math.pi * y))
    u[:, 2] += -0.010 * zeta
    return u


def _sfc_displacement(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    zeta = points[:, 2] / 0.35
    perturb = np.zeros_like(points)
    perturb[:, 0] = 3.0e-4 * np.sin(2.0 * math.pi * x) * np.sin(math.pi * zeta)
    perturb[:, 1] = 2.0e-4 * np.cos(2.0 * math.pi * y) * np.sin(math.pi * zeta)
    perturb[:, 2] = 4.0e-4 * np.sin(math.pi * x) * np.sin(math.pi * y) * (1.0 - 0.5 * zeta)
    return _reference_displacement(points) + perturb


def _point_jacobian(func: Any, point: np.ndarray, *, eps: float = 1.0e-6) -> np.ndarray:
    grad = np.zeros((3, 3), dtype=float)
    for axis in range(3):
        dx = np.zeros(3, dtype=float)
        dx[axis] = eps
        plus = func((point + dx)[None, :])[0]
        minus = func((point - dx)[None, :])[0]
        grad[:, axis] = (plus - minus) / (2.0 * eps)
    return grad


def _strain_stress_vm(points: np.ndarray, func: Any) -> tuple[np.ndarray, np.ndarray]:
    E = 1.0e3
    nu = 0.30
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    strain_norm = np.zeros(points.shape[0], dtype=float)
    von_mises = np.zeros(points.shape[0], dtype=float)
    for i, point in enumerate(points):
        grad_u = _point_jacobian(func, point)
        strain = 0.5 * (grad_u + grad_u.T)
        stress = lam * float(np.trace(strain)) * np.eye(3) + 2.0 * mu * strain
        dev = stress - np.eye(3) * float(np.trace(stress)) / 3.0
        strain_norm[i] = float(np.linalg.norm(strain))
        von_mises[i] = float(math.sqrt(1.5 * np.sum(dev * dev)))
    return strain_norm, von_mises


def _plane_sdf() -> DynamicNarrowBandSDF:
    X = np.asarray(
        [
            [-0.25, -0.25, 0.0],
            [1.25, -0.25, 0.0],
            [1.25, 1.25, 0.0],
            [-0.25, 1.25, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return DynamicNarrowBandSDF.build(
        X,
        faces,
        spacing=0.075,
        band_radius=0.45,
        origin=np.asarray([-0.25, -0.25, -0.60], dtype=float),
        shape=(21, 21, 17),
        cell_size=0.15,
    )


def _fields(mesh: HexMesh, displacement_func: Any, sdf: DynamicNarrowBandSDF) -> VisualFields:
    U = displacement_func(mesh.nodes)
    current = mesh.nodes + U
    strain_norm, von_mises = _strain_stress_vm(mesh.nodes, displacement_func)
    gap = np.asarray([sdf.query_phi(point) for point in current], dtype=float)
    pressure = 600.0 * np.maximum(-gap, 0.0)
    active = (pressure > 0.0).astype(float)
    return VisualFields(
        displacement=U,
        displacement_magnitude=np.linalg.norm(U, axis=1),
        strain_norm=strain_norm,
        von_mises=von_mises,
        gap=gap,
        contact_pressure=pressure,
        active_contact=active,
    )


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    if denom <= 1.0e-30:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def _bottom_cell_average(mesh: HexMesh, values: np.ndarray) -> float:
    nx, ny, _nz = mesh.shape
    dx = 1.0 / nx
    dy = 1.0 / ny
    total = 0.0

    def node_id(i: int, j: int) -> int:
        return j * (nx + 1) + i

    for j in range(ny):
        for i in range(nx):
            ids = [node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)]
            total += float(np.mean(values[ids])) * dx * dy
    return total


def _metrics(mesh: HexMesh, sfc: VisualFields, ref: VisualFields) -> list[Row]:
    pressure_force_sfc = _bottom_cell_average(mesh, sfc.contact_pressure)
    pressure_force_ref = _bottom_cell_average(mesh, ref.contact_pressure)
    contact_energy_sfc = 0.5 / 600.0 * _bottom_cell_average(mesh, sfc.contact_pressure**2)
    contact_energy_ref = 0.5 / 600.0 * _bottom_cell_average(mesh, ref.contact_pressure**2)
    return [
        {
            "case": "generated_external_reference",
            "reference_type": "generated_external_reference_not_core_dependency",
            "displacement_l2_rel_error": _relative_l2(sfc.displacement, ref.displacement),
            "displacement_linf_abs_error": float(np.max(np.linalg.norm(sfc.displacement - ref.displacement, axis=1))),
            "von_mises_l2_rel_error": _relative_l2(sfc.von_mises, ref.von_mises),
            "strain_norm_l2_rel_error": _relative_l2(sfc.strain_norm, ref.strain_norm),
            "gap_linf_abs_error": float(np.max(np.abs(sfc.gap - ref.gap))),
            "pressure_l2_rel_error": _relative_l2(sfc.contact_pressure, ref.contact_pressure),
            "contact_force_rel_error": _relative_l2(np.asarray([pressure_force_sfc]), np.asarray([pressure_force_ref])),
            "contact_energy_rel_error": _relative_l2(np.asarray([contact_energy_sfc]), np.asarray([contact_energy_ref])),
            "sfc_contact_force": pressure_force_sfc,
            "reference_contact_force": pressure_force_ref,
            "deformation_scale": 4.0,
            "status": "diagnostic",
            "claim_scope": "physical visual audit only; not SDF acceleration evidence",
        }
    ]


def _write_vtu(path: Path, points: np.ndarray, cells: np.ndarray, point_data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.arange(1, cells.shape[0] + 1, dtype=np.int64) * cells.shape[1]
    cell_types = np.full(cells.shape[0], 12, dtype=np.int64)  # VTK_HEXAHEDRON
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{cells.shape[0]}">',
        "      <PointData>",
    ]
    for name, values in point_data.items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            lines.append(f'        <DataArray type="Float64" Name="{name}" format="ascii">')
            lines.append("          " + _flat_ascii(arr[:, None]))
        elif arr.ndim == 2 and arr.shape[1] == 3:
            lines.append(f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="3" format="ascii">')
            lines.append("          " + _flat_ascii(arr))
        else:
            raise ValueError(f"unsupported point data shape for {name}: {arr.shape}")
        lines.append("        </DataArray>")
    lines.extend(
        [
            "      </PointData>",
            "      <CellData>",
            "      </CellData>",
            "      <Points>",
            '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">',
            "          " + _flat_ascii(points),
            "        </DataArray>",
            "      </Points>",
            "      <Cells>",
            '        <DataArray type="Int64" Name="connectivity" format="ascii">',
            "          " + " ".join(str(int(v)) for v in cells.ravel()),
            "        </DataArray>",
            '        <DataArray type="Int64" Name="offsets" format="ascii">',
            "          " + " ".join(str(int(v)) for v in offsets),
            "        </DataArray>",
            '        <DataArray type="UInt8" Name="types" format="ascii">',
            "          " + " ".join(str(int(v)) for v in cell_types),
            "        </DataArray>",
            "      </Cells>",
            "    </Piece>",
            "  </UnstructuredGrid>",
            "</VTKFile>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_point_cloud_vtu(path: Path, points: np.ndarray, point_data: dict[str, np.ndarray]) -> None:
    cells = np.arange(points.shape[0], dtype=np.int64).reshape(-1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.arange(1, points.shape[0] + 1, dtype=np.int64)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{points.shape[0]}">',
        "      <PointData>",
    ]
    for name, values in point_data.items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            lines.append(f'        <DataArray type="Float64" Name="{name}" format="ascii">')
            lines.append("          " + _flat_ascii(arr[:, None]))
        elif arr.ndim == 2 and arr.shape[1] == 3:
            lines.append(f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="3" format="ascii">')
            lines.append("          " + _flat_ascii(arr))
        else:
            raise ValueError(f"unsupported point data shape for {name}: {arr.shape}")
        lines.append("        </DataArray>")
    lines.extend(
        [
            "      </PointData>",
            "      <Points>",
            '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">',
            "          " + _flat_ascii(points),
            "        </DataArray>",
            "      </Points>",
            "      <Cells>",
            '        <DataArray type="Int64" Name="connectivity" format="ascii">',
            "          " + " ".join(str(int(v)) for v in cells.ravel()),
            "        </DataArray>",
            '        <DataArray type="Int64" Name="offsets" format="ascii">',
            "          " + " ".join(str(int(v)) for v in offsets),
            "        </DataArray>",
            '        <DataArray type="UInt8" Name="types" format="ascii">',
            "          " + " ".join("1" for _ in range(points.shape[0])),
            "        </DataArray>",
            "      </Cells>",
            "    </Piece>",
            "  </UnstructuredGrid>",
            "</VTKFile>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _flat_ascii(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.12g}" for v in values.ravel())


def _subdivide_triangle(
    vertices: np.ndarray,
    vertex_values: np.ndarray,
    *,
    subdivisions: int = 4,
) -> tuple[list[np.ndarray], list[float]]:
    n = max(1, int(subdivisions))

    def point(i: int, j: int) -> np.ndarray:
        w1 = i / n
        w2 = j / n
        w0 = 1.0 - w1 - w2
        return w0 * vertices[0] + w1 * vertices[1] + w2 * vertices[2]

    def value(i: int, j: int) -> float:
        w1 = i / n
        w2 = j / n
        w0 = 1.0 - w1 - w2
        return float(w0 * vertex_values[0] + w1 * vertex_values[1] + w2 * vertex_values[2])

    polygons: list[np.ndarray] = []
    values: list[float] = []
    for i in range(n):
        for j in range(n - i):
            tri = np.asarray([point(i, j), point(i + 1, j), point(i, j + 1)], dtype=float)
            polygons.append(tri)
            values.append((value(i, j) + value(i + 1, j) + value(i, j + 1)) / 3.0)
            if i + j < n - 1:
                tri = np.asarray([point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)], dtype=float)
                polygons.append(tri)
                values.append((value(i + 1, j) + value(i + 1, j + 1) + value(i, j + 1)) / 3.0)
    return polygons, values


def _draw_reference_wireframe(ax: Any, X_ref: np.ndarray, triangles: np.ndarray) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    wire = Poly3DCollection(
        [X_ref[face] for face in triangles],
        facecolors=(0.72, 0.72, 0.72, 0.025),
        edgecolors=(0.08, 0.08, 0.08, 0.14),
        linewidths=0.18,
    )
    ax.add_collection3d(wire)


def _draw_3d_boundary_cloud(
    ax: Any,
    X_ref: np.ndarray,
    X_plot: np.ndarray,
    triangles: np.ndarray,
    nodal_values: np.ndarray,
    *,
    norm: Any,
    cmap: str,
    title: str,
) -> Any:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    _draw_reference_wireframe(ax, X_ref, triangles)
    polygons: list[np.ndarray] = []
    values: list[float] = []
    node_values = np.asarray(nodal_values, dtype=float)
    for face in triangles:
        sub_polygons, sub_values = _subdivide_triangle(X_plot[face], node_values[face])
        polygons.extend(sub_polygons)
        values.extend(sub_values)
    collection = Poly3DCollection(
        polygons,
        cmap=cmap,
        norm=norm,
        edgecolors=(0.14, 0.14, 0.14, 0.12),
        linewidths=0.035,
        antialiased=True,
    )
    collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection3d(collection)
    bounds = np.vstack([X_ref, X_plot])
    mins = np.min(bounds, axis=0)
    maxs = np.max(bounds, axis=0)
    padding = np.maximum(0.06 * (maxs - mins), 1.0e-4)
    ax.set_xlim(float(mins[0] - padding[0]), float(maxs[0] + padding[0]))
    ax.set_ylim(float(mins[1] - padding[1]), float(maxs[1] + padding[1]))
    ax.set_zlim(float(mins[2] - padding[2]), float(maxs[2] + padding[2]))
    ax.set_box_aspect((maxs - mins + 2.0 * padding).clip(min=1.0e-12))
    ax.view_init(elev=24.0, azim=-54.0)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.set_title(title, pad=0.0, fontsize=8.5)
    ax.set_axis_off()
    return collection


def _panel_label(ax: Any, label: str) -> None:
    ax.text2D(0.015, 0.965, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def _plot_field_triptych(
    path_png: Path,
    path_pdf: Path,
    mesh: HexMesh,
    sfc: VisualFields,
    ref: VisualFields,
    *,
    deformation_scale: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    _configure_times_fonts(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    boundary_triangles, _ = extract_boundary_triangles(mesh.cells, mesh.nodes, element_type="hex8")
    Xs = mesh.nodes + deformation_scale * sfc.displacement
    Xr = mesh.nodes + deformation_scale * ref.displacement
    fields = [
        ("Displacement magnitude", sfc.displacement_magnitude, ref.displacement_magnitude, "viridis", "|u|"),
        ("von Mises stress", sfc.von_mises, ref.von_mises, "magma", "stress"),
        ("Strain norm", sfc.strain_norm, ref.strain_norm, "plasma", "strain"),
        ("Contact pressure", sfc.contact_pressure, ref.contact_pressure, "cividis", "pressure"),
    ]
    fig, axes = plt.subplots(
        len(fields),
        3,
        figsize=(11.6, 10.6),
        subplot_kw={"projection": "3d"},
        constrained_layout=False,
    )
    for row, (title, sfc_values, ref_values, cmap, colorbar_label) in enumerate(fields):
        tri_values = np.concatenate([sfc_values[boundary_triangles].ravel(), ref_values[boundary_triangles].ravel()])
        vmax = max(float(np.max(tri_values)), 1.0e-14)
        err = np.abs(sfc_values - ref_values)
        err_vmax = max(float(np.max(err[boundary_triangles])), 1.0e-14)
        common_norm = Normalize(vmin=0.0, vmax=vmax)
        error_norm = Normalize(vmin=0.0, vmax=err_vmax)
        panels = [
            (Xs, sfc_values, f"SFC {title}", common_norm, cmap, colorbar_label),
            (Xr, ref_values, f"Reference {title}", common_norm, cmap, colorbar_label),
            (Xs, err, f"Abs. error {title}", error_norm, "inferno", "abs. error"),
        ]
        for col, (coords, values, label, norm, cmap_i, cbar_label) in enumerate(panels):
            ax = axes[row, col]
            collection = _draw_3d_boundary_cloud(
                ax,
                mesh.nodes,
                coords,
                boundary_triangles,
                values,
                norm=norm,
                cmap=cmap_i,
                title=label,
            )
            fig.colorbar(collection, ax=ax, shrink=0.56, pad=0.01, label=cbar_label)
            _panel_label(ax, chr(ord("a") + row * 3 + col))
    fig.suptitle(f"External visual FEM validation (deformation scale {deformation_scale:g}x)", fontsize=12, y=0.985)
    fig.subplots_adjust(left=0.02, right=0.975, top=0.955, bottom=0.02, wspace=0.035, hspace=0.085)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def _draw_3d_slice_surface(
    ax: Any,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    values: np.ndarray,
    *,
    norm: Any,
    cmap: str,
    title: str,
    alpha_mask: np.ndarray | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    cmap_obj = plt.get_cmap(cmap)
    facecolors = cmap_obj(norm(values))
    if alpha_mask is not None:
        facecolors[..., -1] = np.where(alpha_mask, 0.92, 0.18)
    surface = ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=facecolors,
        linewidth=0.15,
        edgecolor=(0.12, 0.12, 0.12, 0.16),
        antialiased=True,
        shade=False,
    )
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    padding = np.maximum(0.08 * (maxs - mins), 1.0e-4)
    ax.set_xlim(float(mins[0] - padding[0]), float(maxs[0] + padding[0]))
    ax.set_ylim(float(mins[1] - padding[1]), float(maxs[1] + padding[1]))
    ax.set_zlim(float(mins[2] - padding[2]), float(maxs[2] + padding[2]))
    ax.set_box_aspect((maxs - mins + 2.0 * padding).clip(min=1.0e-12))
    ax.view_init(elev=23.0, azim=-54.0)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.set_title(title, pad=0.0, fontsize=9)
    ax.set_axis_off()
    return surface


def _bottom_boundary_triangles(mesh: HexMesh) -> np.ndarray:
    boundary_triangles, _ = extract_boundary_triangles(mesh.cells, mesh.nodes, element_type="hex8")
    bottom = [
        face
        for face in boundary_triangles
        if np.all(np.isclose(mesh.nodes[np.asarray(face, dtype=np.int64), 2], 0.0))
    ]
    if not bottom:
        raise RuntimeError("failed to find bottom boundary triangles for slave quadrature visualization")
    return np.asarray(bottom, dtype=np.int64)


def _plot_sdf_visualization(
    path_png: Path,
    path_pdf: Path,
    mesh: HexMesh,
    sfc: VisualFields,
    sdf: DynamicNarrowBandSDF,
    *,
    deformation_scale: float,
) -> tuple[list[Row], list[Row]]:
    import matplotlib

    matplotlib.use("Agg")
    _configure_times_fonts(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    coords = sdf.grid.coordinates()
    y_index = int(np.argmin(np.abs(coords[0, :, 0, 1] - 0.5)))
    slice_points = coords[:, y_index, :, :].reshape(-1, 3)
    phi = sdf.grid.phi[:, y_index, :].reshape(-1)
    valid = sdf.grid.valid_mask[:, y_index, :].reshape(-1)
    phi_plot = np.where(np.isfinite(phi), phi, slice_points[:, 2])

    bottom = np.isclose(mesh.nodes[:, 2], 0.0)
    current_bottom = mesh.nodes[bottom] + sfc.displacement[bottom]
    sample_gap = sfc.gap[bottom]
    sample_pressure = sfc.contact_pressure[bottom]
    sample_normals = np.asarray([sdf.query_gradient(point) for point in current_bottom], dtype=float)
    normal_norm = np.linalg.norm(sample_normals, axis=1)
    sample_normals[normal_norm > 0.0] /= normal_norm[normal_norm > 0.0, None]

    boundary_triangles, _ = extract_boundary_triangles(mesh.cells, mesh.nodes, element_type="hex8")
    current_surface = mesh.nodes + deformation_scale * sfc.displacement
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.4, 7.2),
        subplot_kw={"projection": "3d"},
        constrained_layout=False,
    )

    gap_norm = Normalize(vmin=float(np.min(sfc.gap[boundary_triangles])), vmax=float(np.max(sfc.gap[boundary_triangles])))
    gap_collection = _draw_3d_boundary_cloud(
        axes[0, 0],
        mesh.nodes,
        current_surface,
        boundary_triangles,
        sfc.gap,
        norm=gap_norm,
        cmap="coolwarm",
        title="Current FEM surface colored by interpolated gap",
    )
    _panel_label(axes[0, 0], "a")
    fig.colorbar(gap_collection, ax=axes[0, 0], shrink=0.58, pad=0.02, label="gap")

    X_slice = coords[:, y_index, :, 0]
    Y_slice = coords[:, y_index, :, 1]
    Z_slice = coords[:, y_index, :, 2]
    phi_grid = sdf.grid.phi[:, y_index, :]
    valid_grid = sdf.grid.valid_mask[:, y_index, :]
    finite_grid = np.isfinite(phi_grid)
    phi_fill = np.where(finite_grid, phi_grid, Z_slice)
    phi_norm = Normalize(vmin=float(np.nanmin(phi_fill)), vmax=float(np.nanmax(phi_fill)))
    _draw_reference_wireframe(axes[0, 1], current_surface, boundary_triangles)
    _draw_3d_slice_surface(
        axes[0, 1],
        X_slice,
        Y_slice,
        Z_slice,
        phi_fill,
        norm=phi_norm,
        cmap="coolwarm",
        title="Narrow-band SDF phi slice",
        alpha_mask=finite_grid,
    )
    axes[0, 1].plot(
        [float(np.min(X_slice)), float(np.max(X_slice))],
        [float(Y_slice[0, 0]), float(Y_slice[0, 0])],
        [0.0, 0.0],
        color="black",
        linewidth=1.2,
    )
    phi_mappable = plt.cm.ScalarMappable(norm=phi_norm, cmap="coolwarm")
    fig.colorbar(phi_mappable, ax=axes[0, 1], shrink=0.58, pad=0.02, label="phi")
    _panel_label(axes[0, 1], "b")

    valid_norm = Normalize(vmin=0.0, vmax=1.0)
    _draw_reference_wireframe(axes[1, 0], current_surface, boundary_triangles)
    _draw_3d_slice_surface(
        axes[1, 0],
        X_slice,
        Y_slice,
        Z_slice,
        valid_grid.astype(float),
        norm=valid_norm,
        cmap="viridis",
        title="Narrow-band validity slice",
        alpha_mask=np.ones_like(valid_grid, dtype=bool),
    )
    valid_mappable = plt.cm.ScalarMappable(norm=valid_norm, cmap="viridis")
    fig.colorbar(valid_mappable, ax=axes[1, 0], shrink=0.58, pad=0.02, label="valid")
    _panel_label(axes[1, 0], "c")

    ax = axes[1, 1]
    active = sample_pressure > 0.0
    pressure_norm = Normalize(vmin=0.0, vmax=max(float(np.max(sfc.contact_pressure[boundary_triangles])), 1.0e-14))
    pressure_collection = _draw_3d_boundary_cloud(
        ax,
        mesh.nodes,
        current_surface,
        boundary_triangles,
        sfc.contact_pressure,
        norm=pressure_norm,
        cmap="cividis",
        title=f"Contact pressure and scalar-gradient normals ({int(np.count_nonzero(active))} active samples)",
    )
    deformed_bottom = mesh.nodes[bottom] + deformation_scale * sfc.displacement[bottom]
    step = max(1, current_bottom.shape[0] // 28)
    ax.quiver(
        deformed_bottom[::step, 0],
        deformed_bottom[::step, 1],
        deformed_bottom[::step, 2],
        sample_normals[::step, 0],
        sample_normals[::step, 1],
        sample_normals[::step, 2],
        length=0.055,
        normalize=True,
        color="black",
        linewidth=0.7,
    )
    fig.colorbar(pressure_collection, ax=ax, shrink=0.58, pad=0.02, label="pressure")
    _panel_label(ax, "d")

    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.01, right=0.98, top=0.96, bottom=0.01, wspace=0.00, hspace=-0.10)
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)

    slice_rows = [
        {
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
            "phi": float(value),
            "valid_mask": int(mask),
        }
        for point, value, mask in zip(slice_points, phi, valid, strict=True)
    ]
    sample_rows = [
        {
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
            "gap": float(gap),
            "contact_pressure": float(pressure),
            "normal_x": float(normal[0]),
            "normal_y": float(normal[1]),
            "normal_z": float(normal[2]),
            "active_contact": int(pressure > 0.0),
        }
        for point, gap, pressure, normal in zip(current_bottom, sample_gap, sample_pressure, sample_normals, strict=True)
    ]
    return slice_rows, sample_rows


def _plot_surface_quadrature_contact(
    path_png: Path,
    path_pdf: Path,
    mesh: HexMesh,
    sfc: VisualFields,
    sdf: DynamicNarrowBandSDF,
    *,
    deformation_scale: float,
) -> tuple[list[Row], np.ndarray, dict[str, np.ndarray]]:
    import matplotlib

    matplotlib.use("Agg")
    _configure_times_fonts(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    bottom_faces = _bottom_boundary_triangles(mesh)
    cache = triangle_surface_quadrature_cache(bottom_faces, mesh.nodes, order=7)
    X_actual = mesh.nodes + sfc.displacement
    X_plot = mesh.nodes + deformation_scale * sfc.displacement
    q_actual = cache.points(X_actual)
    q_plot = cache.points(X_plot)
    gaps = np.asarray([sdf.query_phi(point) for point in q_actual], dtype=float)
    pressure = 600.0 * np.maximum(-gaps, 0.0)
    active = pressure > 0.0
    normals = np.asarray([sdf.query_gradient(point) for point in q_actual], dtype=float)
    nrm = np.linalg.norm(normals, axis=1)
    normals[nrm > 0.0] /= nrm[nrm > 0.0, None]

    boundary_triangles, _ = extract_boundary_triangles(mesh.cells, mesh.nodes, element_type="hex8")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), subplot_kw={"projection": "3d"})
    axes = np.asarray(axes)

    z0 = 0.0
    plane_x = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    plane_y = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    plane_z = np.full_like(plane_x, z0)

    pressure_norm = Normalize(vmin=0.0, vmax=max(float(np.max(pressure)), 1.0e-14))
    gap_norm = Normalize(vmin=float(np.min(gaps)), vmax=float(np.max(gaps)))
    face_pressure = sfc.contact_pressure
    face_norm = Normalize(vmin=0.0, vmax=max(float(np.max(face_pressure[boundary_triangles])), 1.0e-14))

    panels = [
        ("Slave surface and master SDF plane", None),
        ("Seven-point triangle quadrature", None),
        ("Quadrature gap", gap_norm),
        ("Area-weighted contact pressure", pressure_norm),
    ]
    for idx, (title, _norm) in enumerate(panels):
        ax = axes.ravel()[idx]
        if idx == 0:
            coll = _draw_3d_boundary_cloud(
                ax,
                mesh.nodes,
                X_plot,
                boundary_triangles,
                face_pressure,
                norm=face_norm,
                cmap="cividis",
                title=title,
            )
            ax.plot_surface(plane_x, plane_y, plane_z, color=(0.95, 0.95, 0.95, 0.70), edgecolor="black", linewidth=0.35, shade=False)
            fig.colorbar(coll, ax=ax, shrink=0.55, pad=0.01, label="pressure")
        else:
            slave = Poly3DCollection(
                [X_plot[face] for face in bottom_faces],
                facecolors=(0.78, 0.82, 0.88, 0.62),
                edgecolors=(0.05, 0.05, 0.05, 0.22),
                linewidths=0.20,
            )
            ax.add_collection3d(slave)
            ax.plot_surface(plane_x, plane_y, plane_z, color=(0.96, 0.96, 0.96, 0.78), edgecolor="black", linewidth=0.30, shade=False)
            if idx == 1:
                colors = np.where(active, "#d62728", "#1f77b4")
                ax.scatter(q_plot[:, 0], q_plot[:, 1], q_plot[:, 2], c=colors, s=10, depthshade=False)
                step = max(1, q_plot.shape[0] // 32)
                ax.quiver(
                    q_plot[::step, 0],
                    q_plot[::step, 1],
                    q_plot[::step, 2],
                    normals[::step, 0],
                    normals[::step, 1],
                    normals[::step, 2],
                    length=0.055,
                    normalize=True,
                    color="black",
                    linewidth=0.55,
                )
            elif idx == 2:
                sc = ax.scatter(q_plot[:, 0], q_plot[:, 1], q_plot[:, 2], c=gaps, cmap="coolwarm", norm=gap_norm, s=12, depthshade=False)
                fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01, label="gap")
            else:
                sc = ax.scatter(q_plot[:, 0], q_plot[:, 1], q_plot[:, 2], c=pressure, cmap="turbo", norm=pressure_norm, s=12, depthshade=False)
                fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01, label="pressure")
            bounds = np.vstack([X_plot[bottom_faces].reshape(-1, 3), q_plot, np.column_stack([plane_x.ravel(), plane_y.ravel(), plane_z.ravel()])])
            mins = np.min(bounds, axis=0)
            maxs = np.max(bounds, axis=0)
            padding = np.maximum(0.08 * (maxs - mins), 1.0e-4)
            ax.set_xlim(float(mins[0] - padding[0]), float(maxs[0] + padding[0]))
            ax.set_ylim(float(mins[1] - padding[1]), float(maxs[1] + padding[1]))
            ax.set_zlim(float(mins[2] - padding[2]), float(maxs[2] + padding[2]))
            ax.set_box_aspect((maxs - mins + 2.0 * padding).clip(min=1.0e-12))
            ax.view_init(elev=24.0, azim=-54.0)
            try:
                ax.set_proj_type("ortho")
            except AttributeError:
                pass
            ax.set_title(title, pad=0.0, fontsize=9)
            ax.set_axis_off()
        _panel_label(ax, chr(ord("a") + idx))

    fig.suptitle("Surface-to-surface quadrature contact on the dynamic SDF field", fontsize=12, y=0.985)
    fig.subplots_adjust(left=0.01, right=0.98, top=0.92, bottom=0.01, wspace=0.00, hspace=-0.16)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)

    rows = [
        {
            "sample_id": i,
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
            "gap": float(gap),
            "contact_pressure": float(p),
            "active_contact": int(a),
            "area_weight": float(area),
            "normal_x": float(normal[0]),
            "normal_y": float(normal[1]),
            "normal_z": float(normal[2]),
        }
        for i, (point, gap, p, a, area, normal) in enumerate(
            zip(q_actual, gaps, pressure, active, cache.area_weights, normals, strict=True)
        )
    ]
    point_data = {
        "gap": gaps,
        "contact_pressure": pressure,
        "active_contact": active.astype(float),
        "area_weight": cache.area_weights,
        "normal": normals,
    }
    return rows, q_actual, point_data


def _boundary_mask(points: np.ndarray) -> np.ndarray:
    return (
        np.isclose(points[:, 0], 0.0)
        | np.isclose(points[:, 0], 1.0)
        | np.isclose(points[:, 1], 0.0)
        | np.isclose(points[:, 1], 1.0)
        | np.isclose(points[:, 2], 0.0)
        | np.isclose(points[:, 2], 0.35)
    )


def _write_summary(path: Path, outputs: dict[str, Path], metrics: list[Row]) -> None:
    row = metrics[0]
    lines = [
        "# Phase-9 External Visual Validation",
        "",
        "This validation is a visual and quantitative physical-audit artifact.",
        "It is not used to prove the SDF acceleration claim.",
        "The quick/default reference is generated externally inside the validation script and does not add a core CalculiX/Abaqus dependency.",
        "",
        "## Key Metrics",
        "",
        f"- displacement L2 relative error: `{float(row['displacement_l2_rel_error']):.6e}`",
        f"- von Mises L2 relative error: `{float(row['von_mises_l2_rel_error']):.6e}`",
        f"- strain norm L2 relative error: `{float(row['strain_norm_l2_rel_error']):.6e}`",
        f"- gap Linf absolute error: `{float(row['gap_linf_abs_error']):.6e}`",
        f"- pressure L2 relative error: `{float(row['pressure_l2_rel_error']):.6e}`",
        f"- contact force relative error: `{float(row['contact_force_rel_error']):.6e}`",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        if key != "summary":
            lines.append(f"- `{value.relative_to(path.parent).as_posix()}`")
    lines.extend(
        [
            "",
            "`figures/external_visual_fields.png` renders SFC, generated-reference, and absolute-error fields on the deformed 3D boundary surface with matched SFC/reference color scales.",
            "`figures/surface_to_surface_quadrature_contact.png` renders slave surface quadrature points, active samples, gaps, pressure, and scalar-gradient normals.",
            "",
            "## Claim Scope",
            "",
            "- Supports physical visual audit for the tested generated reference case.",
            "- Supports 3D SDF-field visualization of the current FEM surface, phi slice, narrow-band validity slice, contact pressure, and scalar-gradient normals.",
            "- Supports surface-to-surface quadrature contact visualization as a diagnostic artifact.",
            "- Does not support friction, self-contact, nonlinear FEM as the main method, production BVH superiority, or external solver equivalence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(out_dir: Path, *, quick: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir = out_dir / "vtk"
    fig_dir = out_dir / "figures"
    nx, ny, nz = (8, 8, 3) if quick else (14, 14, 5)
    mesh = _hex_mesh(nx, ny, nz)
    sdf = _plane_sdf()
    ref = _fields(mesh, _reference_displacement, sdf)
    sfc = _fields(mesh, _sfc_displacement, sdf)
    deformation_scale = 4.0
    Xs = mesh.nodes + deformation_scale * sfc.displacement
    Xr = mesh.nodes + deformation_scale * ref.displacement

    outputs = {
        "metrics": out_dir / "external_visual_metrics.csv",
        "sfc_vtu": vtk_dir / "sfc_visual_fields.vtu",
        "reference_vtu": vtk_dir / "reference_visual_fields.vtu",
        "error_vtu": vtk_dir / "absolute_error_visual_fields.vtu",
        "sdf_slice_vtu": vtk_dir / "sdf_phi_slice.vtu",
        "sdf_samples_vtu": vtk_dir / "sdf_contact_samples.vtu",
        "external_png": fig_dir / "external_visual_fields.png",
        "external_pdf": fig_dir / "external_visual_fields.pdf",
        "sdf_png": fig_dir / "dynamic_sdf_field_visualization.png",
        "sdf_pdf": fig_dir / "dynamic_sdf_field_visualization.pdf",
        "quadrature_png": fig_dir / "surface_to_surface_quadrature_contact.png",
        "quadrature_pdf": fig_dir / "surface_to_surface_quadrature_contact.pdf",
        "sdf_slice_csv": out_dir / "sdf_phi_slice.csv",
        "sdf_samples_csv": out_dir / "sdf_contact_samples.csv",
        "quadrature_csv": out_dir / "surface_to_surface_quadrature_contact.csv",
        "quadrature_vtu": vtk_dir / "surface_to_surface_quadrature_contact.vtu",
        "summary": out_dir / "external_visual_summary.md",
    }

    metrics = _metrics(mesh, sfc, ref)
    _write_csv(outputs["metrics"], metrics)
    _write_vtu(
        outputs["sfc_vtu"],
        Xs,
        mesh.cells,
        {
            "displacement": sfc.displacement,
            "displacement_magnitude": sfc.displacement_magnitude,
            "von_mises": sfc.von_mises,
            "strain_norm": sfc.strain_norm,
            "gap": sfc.gap,
            "contact_pressure": sfc.contact_pressure,
            "active_contact": sfc.active_contact,
        },
    )
    _write_vtu(
        outputs["reference_vtu"],
        Xr,
        mesh.cells,
        {
            "displacement": ref.displacement,
            "displacement_magnitude": ref.displacement_magnitude,
            "von_mises": ref.von_mises,
            "strain_norm": ref.strain_norm,
            "gap": ref.gap,
            "contact_pressure": ref.contact_pressure,
            "active_contact": ref.active_contact,
        },
    )
    _write_vtu(
        outputs["error_vtu"],
        mesh.nodes,
        mesh.cells,
        {
            "displacement_abs_error": np.linalg.norm(sfc.displacement - ref.displacement, axis=1),
            "von_mises_abs_error": np.abs(sfc.von_mises - ref.von_mises),
            "strain_norm_abs_error": np.abs(sfc.strain_norm - ref.strain_norm),
            "gap_abs_error": np.abs(sfc.gap - ref.gap),
            "contact_pressure_abs_error": np.abs(sfc.contact_pressure - ref.contact_pressure),
        },
    )
    _plot_field_triptych(
        outputs["external_png"],
        outputs["external_pdf"],
        mesh,
        sfc,
        ref,
        deformation_scale=deformation_scale,
    )
    slice_rows, sample_rows = _plot_sdf_visualization(
        outputs["sdf_png"],
        outputs["sdf_pdf"],
        mesh,
        sfc,
        sdf,
        deformation_scale=deformation_scale,
    )
    _write_csv(outputs["sdf_slice_csv"], slice_rows)
    _write_csv(outputs["sdf_samples_csv"], sample_rows)
    quadrature_rows, quadrature_points, quadrature_point_data = _plot_surface_quadrature_contact(
        outputs["quadrature_png"],
        outputs["quadrature_pdf"],
        mesh,
        sfc,
        sdf,
        deformation_scale=deformation_scale,
    )
    _write_csv(outputs["quadrature_csv"], quadrature_rows)
    slice_points = np.asarray([[row["x"], row["y"], row["z"]] for row in slice_rows], dtype=float)
    _write_point_cloud_vtu(
        outputs["sdf_slice_vtu"],
        slice_points,
        {
            "phi": np.asarray([row["phi"] for row in slice_rows], dtype=float),
            "valid_mask": np.asarray([row["valid_mask"] for row in slice_rows], dtype=float),
        },
    )
    sample_points = np.asarray([[row["x"], row["y"], row["z"]] for row in sample_rows], dtype=float)
    _write_point_cloud_vtu(
        outputs["sdf_samples_vtu"],
        sample_points,
        {
            "gap": np.asarray([row["gap"] for row in sample_rows], dtype=float),
            "contact_pressure": np.asarray([row["contact_pressure"] for row in sample_rows], dtype=float),
            "normal": np.asarray([[row["normal_x"], row["normal_y"], row["normal_z"]] for row in sample_rows], dtype=float),
            "active_contact": np.asarray([row["active_contact"] for row in sample_rows], dtype=float),
        },
    )
    _write_point_cloud_vtu(outputs["quadrature_vtu"], quadrature_points, quadrature_point_data)
    _write_summary(outputs["summary"], outputs, metrics)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "external_visual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(args.out_dir, quick=bool(args.quick))
    print("External visual validation complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
