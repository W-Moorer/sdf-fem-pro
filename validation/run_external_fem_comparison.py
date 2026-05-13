"""External open-source FEM comparison using scikit-fem.

This validation-only runner compares the standalone SFC linear TET4 assembly
against an independent open-source FEM implementation on the same mesh, material,
boundary conditions, and load.  It does not change core solver physics and it
does not introduce scikit-fem as a dependency of ``src/sfc``.
"""

from __future__ import annotations

import argparse
import csv
import sys
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
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

try:  # noqa: E402
    import skfem
    from skfem import Basis, ElementTetP1, ElementVector, MeshTet, asm, condense, solve
    from skfem.models.elasticity import lame_parameters, linear_elasticity
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without optional validation dependency
    raise SystemExit(
        "scikit-fem is required for external FEM validation. "
        "Install it with `pip install scikit-fem` or `pip install -e .[validation]`."
    ) from exc

from sfc.fem import assemble_stiffness_matrix, eliminate_fixed_dofs, fixed_dofs_from_node_set  # noqa: E402
from sfc.mesh import extract_boundary_faces  # noqa: E402
from validation.run_phase3_validation import _body, structured_tet_block  # noqa: E402
from validation.run_phase4_paper_validation import _format_float  # noqa: E402
from validation.run_phase7_physical_validation import _element_strain_stress  # noqa: E402

Row = dict[str, Any]


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-30))


def _sparse_frobenius_norm(matrix) -> float:
    return float(np.sqrt(matrix.multiply(matrix).sum()))


def _load_case(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixed_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], 0.0))
    tip_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], mesh.X[:, 0].max()))
    force = np.zeros(3 * mesh.X.shape[0], dtype=float)
    force[tip_nodes * 3 + 2] = -1.0 / tip_nodes.size
    return fixed_nodes, tip_nodes, force


def solve_sfc_cantilever(mesh, E: float, nu: float) -> tuple[np.ndarray, Any]:
    """Solve the cantilever load case using the standalone SFC assembler."""

    fixed_nodes, _, force = _load_case(mesh)
    fixed = fixed_dofs_from_node_set(fixed_nodes, "xyz")
    body = _body(mesh, E=E, nu=nu, rho=1.0)
    K = assemble_stiffness_matrix(body)
    Kff, rhs, free = eliminate_fixed_dofs(K, force, fixed)
    u = np.zeros(body.n_dofs, dtype=float)
    u[free] = np.asarray(spsolve(Kff, rhs), dtype=float)
    return u, K


def solve_scikit_fem_cantilever(mesh, E: float, nu: float) -> tuple[np.ndarray, Any]:
    """Solve the same cantilever load case using scikit-fem."""

    fixed_nodes, _, force = _load_case(mesh)
    fixed = fixed_dofs_from_node_set(fixed_nodes, "xyz")
    sk_mesh = MeshTet(mesh.X.T, mesh.elements.T)
    basis = Basis(sk_mesh, ElementVector(ElementTetP1()))
    lambda_, mu = lame_parameters(E, nu)
    K = asm(linear_elasticity(lambda_, mu), basis)
    u = solve(*condense(K, force, D=fixed))
    return np.asarray(u, dtype=float), K


def external_fem_comparison(resolutions: list[int]) -> tuple[list[Row], list[Row]]:
    """Compare SFC and scikit-fem displacement, stiffness, strain, and stress."""

    E, nu = 1.0e5, 0.3
    summary_rows: list[Row] = []
    cloud_rows: list[Row] = []
    for resolution in resolutions:
        mesh = structured_tet_block(resolution, resolution, resolution, size=(1.0, 0.2, 0.2))
        fixed_nodes, tip_nodes, _ = _load_case(mesh)
        u_sfc, K_sfc = solve_sfc_cantilever(mesh, E, nu)
        u_ext, K_ext = solve_scikit_fem_cantilever(mesh, E, nu)
        strain_sfc, stress_sfc, vm_sfc = _element_strain_stress(mesh, u_sfc, E, nu)
        strain_ext, stress_ext, vm_ext = _element_strain_stress(mesh, u_ext, E, nu)
        K_diff = K_sfc - K_ext
        stiffness_rel = _sparse_frobenius_norm(K_diff) / max(_sparse_frobenius_norm(K_ext), 1.0e-30)
        displacement_rel = _relative_error(u_sfc, u_ext)
        strain_rel = _relative_error(strain_sfc.ravel(), strain_ext.ravel())
        stress_rel = _relative_error(stress_sfc.ravel(), stress_ext.ravel())
        vm_rel = _relative_error(vm_sfc, vm_ext)
        tip_sfc = float(np.mean(u_sfc.reshape((-1, 3))[tip_nodes, 2]))
        tip_ext = float(np.mean(u_ext.reshape((-1, 3))[tip_nodes, 2]))
        tip_rel = abs(tip_sfc - tip_ext) / max(abs(tip_ext), 1.0e-30)
        summary_rows.append(
            {
                "case": "cantilever_external_open_source_fem",
                "external_solver": "scikit-fem",
                "external_solver_version": skfem.__version__,
                "resolution": resolution,
                "nodes": mesh.X.shape[0],
                "elements": mesh.elements.shape[0],
                "fixed_nodes": fixed_nodes.size,
                "tip_displacement_z_sfc": tip_sfc,
                "tip_displacement_z_external": tip_ext,
                "tip_displacement_rel_error": tip_rel,
                "displacement_l2_rel_error": displacement_rel,
                "displacement_linf_abs_error": float(np.max(np.abs(u_sfc - u_ext))),
                "stiffness_fro_rel_error": stiffness_rel,
                "strain_l2_rel_error": strain_rel,
                "stress_l2_rel_error": stress_rel,
                "von_mises_l2_rel_error": vm_rel,
                "max_von_mises_sfc": float(np.max(vm_sfc)),
                "max_von_mises_external": float(np.max(vm_ext)),
                "max_von_mises_abs_error": float(abs(np.max(vm_sfc) - np.max(vm_ext))),
                "status": "ok"
                if max(displacement_rel, stiffness_rel, strain_rel, stress_rel, vm_rel, tip_rel) < 1.0e-8
                else "check",
                "details": "same mesh/material/Dirichlet boundary/point load compared against scikit-fem",
            }
        )
        centroids = np.mean(mesh.X[mesh.elements], axis=1)
        for element_id, (centroid, sfc_value, ext_value) in enumerate(zip(centroids, vm_sfc, vm_ext, strict=True)):
            cloud_rows.append(
                {
                    "case": "cantilever_external_open_source_fem_stress_cloud",
                    "external_solver": "scikit-fem",
                    "resolution": resolution,
                    "element_id": element_id,
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "centroid_z": float(centroid[2]),
                    "von_mises_sfc": float(sfc_value),
                    "von_mises_external": float(ext_value),
                    "von_mises_abs_error": float(abs(sfc_value - ext_value)),
                    "status": "ok" if np.isfinite(float(sfc_value)) and np.isfinite(float(ext_value)) else "check",
                    "details": "element-centroid von Mises comparison for projected x-z stress cloud",
                }
            )
    return summary_rows, cloud_rows


def _plot_error_trends(out_dir: Path, rows: list[Row]) -> Row:
    plt.figure(figsize=(6.2, 4.0))
    x = [int(row["elements"]) for row in rows]
    for field, label in [
        ("displacement_l2_rel_error", "displacement"),
        ("stiffness_fro_rel_error", "stiffness"),
        ("stress_l2_rel_error", "stress"),
        ("von_mises_l2_rel_error", "von Mises"),
    ]:
        plt.semilogy(x, [max(float(row[field]), 1.0e-18) for row in rows], marker="o", label=label)
    plt.xlabel("TET4 elements")
    plt.ylabel("relative error vs scikit-fem")
    plt.legend()
    plt.grid(True, color="0.9", linewidth=0.5)
    plt.tight_layout()
    png = out_dir / "external_fem_error_trends.png"
    pdf = out_dir / "external_fem_error_trends.pdf"
    plt.savefig(png, dpi=180)
    plt.savefig(pdf)
    plt.close()
    return {"plot": "external_fem_error_trends", "png": png.name, "pdf": pdf.name, "status": "ok"}


def _nodal_average(mesh, element_values: np.ndarray) -> np.ndarray:
    values = np.asarray(element_values, dtype=float)
    sums = np.zeros(mesh.X.shape[0], dtype=float)
    counts = np.zeros(mesh.X.shape[0], dtype=float)
    for element, value in zip(mesh.elements, values, strict=True):
        sums[element] += float(value)
        counts[element] += 1.0
    return sums / np.maximum(counts, 1.0)


def _subdivide_triangle(
    vertices: np.ndarray,
    vertex_values: np.ndarray,
    *,
    subdivisions: int = 5,
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
            tri = np.asarray([point(i, j), point(i + 1, j), point(i, j + 1)])
            polygons.append(tri)
            values.append((value(i, j) + value(i + 1, j) + value(i, j + 1)) / 3.0)
            if i + j < n - 1:
                tri = np.asarray([point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)])
                polygons.append(tri)
                values.append((value(i + 1, j) + value(i + 1, j + 1) + value(i, j + 1)) / 3.0)
    return polygons, values


def _projected_side_surface(mesh) -> tuple[list[np.ndarray], np.ndarray]:
    faces, adjacent = extract_boundary_faces(mesh.elements, mesh.X)
    y_min = float(np.min(mesh.X[:, 1]))
    polygons: list[np.ndarray] = []
    adjacent_ids: list[int] = []
    for face, element_id in zip(faces, adjacent, strict=True):
        coords = mesh.X[face]
        if np.all(np.isclose(coords[:, 1], y_min)):
            polygons.append(coords[:, [0, 2]])
            adjacent_ids.append(int(element_id))
    return polygons, np.asarray(adjacent_ids, dtype=np.int64)


def _draw_projected_surface(
    ax,
    mesh,
    nodal_values: np.ndarray,
    *,
    norm: Normalize,
    cmap: str,
    title: str,
) -> PolyCollection:
    faces, _ = extract_boundary_faces(mesh.elements, mesh.X)
    y_min = float(np.min(mesh.X[:, 1]))
    polygons: list[np.ndarray] = []
    values: list[float] = []
    node_values = np.asarray(nodal_values, dtype=float)
    for face in faces:
        coords = mesh.X[face]
        if not np.all(np.isclose(coords[:, 1], y_min)):
            continue
        sub_polygons, sub_values = _subdivide_triangle(coords[:, [0, 2]], node_values[face])
        polygons.extend(sub_polygons)
        values.extend(sub_values)
    collection = PolyCollection(
        polygons,
        array=np.asarray(values, dtype=float),
        cmap=cmap,
        norm=norm,
        edgecolors=(0.15, 0.15, 0.15, 0.14),
        linewidths=0.08,
    )
    ax.add_collection(collection)
    xmin, xmax = float(np.min(mesh.X[:, 0])), float(np.max(mesh.X[:, 0]))
    zmin, zmax = float(np.min(mesh.X[:, 2])), float(np.max(mesh.X[:, 2]))
    ax.plot([xmin, xmax, xmax, xmin, xmin], [zmin, zmin, zmax, zmax, zmin], color="0.15", linewidth=0.9)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_box_aspect(0.55)
    ax.grid(False)
    return collection


def _draw_3d_surface(
    ax,
    mesh,
    nodal_values: np.ndarray,
    *,
    norm: Normalize,
    cmap: str,
    title: str,
) -> Poly3DCollection:
    faces, _ = extract_boundary_faces(mesh.elements, mesh.X)
    polygons: list[np.ndarray] = []
    values: list[float] = []
    node_values = np.asarray(nodal_values, dtype=float)
    for face in faces:
        sub_polygons, sub_values = _subdivide_triangle(mesh.X[face], node_values[face])
        polygons.extend(sub_polygons)
        values.extend(sub_values)
    collection = Poly3DCollection(
        polygons,
        cmap=cmap,
        norm=norm,
        edgecolors=(0.18, 0.18, 0.18, 0.14),
        linewidths=0.06,
    )
    collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection3d(collection)
    mins = np.min(mesh.X, axis=0)
    maxs = np.max(mesh.X, axis=0)
    ax.set_xlim(float(mins[0]), float(maxs[0]))
    ax.set_ylim(float(mins[1]), float(maxs[1]))
    ax.set_zlim(float(mins[2]), float(maxs[2]))
    ax.set_box_aspect((maxs - mins).clip(min=1.0e-12))
    ax.view_init(elev=22.0, azim=-58.0)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.set_title(title, pad=0.0)
    ax.set_axis_off()
    return collection


def _plot_stress_cloud(out_dir: Path, rows: list[Row]) -> Row:
    finest = max(int(row["resolution"]) for row in rows)
    subset = [row for row in rows if int(row["resolution"]) == finest]
    mesh = structured_tet_block(finest, finest, finest, size=(1.0, 0.2, 0.2))
    by_element = {int(row["element_id"]): row for row in subset}
    vm_sfc = np.asarray([float(by_element[element_id]["von_mises_sfc"]) for element_id in range(mesh.elements.shape[0])], dtype=float)
    vm_ext = np.asarray([float(by_element[element_id]["von_mises_external"]) for element_id in range(mesh.elements.shape[0])], dtype=float)
    err = np.asarray([float(by_element[element_id]["von_mises_abs_error"]) for element_id in range(mesh.elements.shape[0])], dtype=float)
    nodal_sfc = _nodal_average(mesh, vm_sfc)
    nodal_ext = _nodal_average(mesh, vm_ext)
    nodal_err = np.abs(nodal_sfc - nodal_ext)
    vmax = max(float(np.max(vm_sfc)), float(np.max(vm_ext)), 1.0e-30)
    err_max = max(float(np.max(err)), 1.0e-30)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharex=True, sharey=True)
    panels = [
        (nodal_sfc, "SFC", Normalize(vmin=0.0, vmax=vmax), "viridis"),
        (nodal_ext, "scikit-fem", Normalize(vmin=0.0, vmax=vmax), "viridis"),
        (nodal_err, "absolute error", Normalize(vmin=0.0, vmax=max(float(np.max(nodal_err)), 1.0e-30)), "magma"),
    ]
    stress_collection = None
    error_collection = None
    for ax, (values, title, norm, cmap) in zip(axes, panels, strict=True):
        collection = _draw_projected_surface(ax, mesh, values, norm=norm, cmap=cmap, title=title)
        if title == "absolute error":
            error_collection = collection
        else:
            stress_collection = collection
    axes[0].set_ylabel("z")
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.20, top=0.78, wspace=0.12)
    if stress_collection is not None:
        cax = fig.add_axes((0.88, 0.30, 0.018, 0.45))
        fig.colorbar(stress_collection, cax=cax, label="von Mises")
    if error_collection is not None:
        cax_err = fig.add_axes((0.94, 0.30, 0.018, 0.45))
        fig.colorbar(error_collection, cax=cax_err, label="abs. error")
    fig.suptitle(f"External FEM side-surface stress comparison, resolution {finest}")
    png = out_dir / "external_fem_stress_cloud.png"
    pdf = out_dir / "external_fem_stress_cloud.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"plot": "external_fem_stress_cloud", "png": png.name, "pdf": pdf.name, "status": "ok"}


def _plot_stress_3d(out_dir: Path, rows: list[Row]) -> Row:
    finest = max(int(row["resolution"]) for row in rows)
    subset = [row for row in rows if int(row["resolution"]) == finest]
    mesh = structured_tet_block(finest, finest, finest, size=(1.0, 0.2, 0.2))
    by_element = {int(row["element_id"]): row for row in subset}
    vm_sfc = np.asarray([float(by_element[element_id]["von_mises_sfc"]) for element_id in range(mesh.elements.shape[0])], dtype=float)
    vm_ext = np.asarray([float(by_element[element_id]["von_mises_external"]) for element_id in range(mesh.elements.shape[0])], dtype=float)
    err = np.asarray([float(by_element[element_id]["von_mises_abs_error"]) for element_id in range(mesh.elements.shape[0])], dtype=float)
    nodal_sfc = _nodal_average(mesh, vm_sfc)
    nodal_ext = _nodal_average(mesh, vm_ext)
    nodal_err = np.abs(nodal_sfc - nodal_ext)
    vmax = max(float(np.max(vm_sfc)), float(np.max(vm_ext)), 1.0e-30)
    err_max = max(float(np.max(err)), 1.0e-30)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), subplot_kw={"projection": "3d"})
    panels = [
        (nodal_sfc, "SFC", Normalize(vmin=0.0, vmax=vmax), "viridis"),
        (nodal_ext, "scikit-fem", Normalize(vmin=0.0, vmax=vmax), "viridis"),
        (nodal_err, "absolute error", Normalize(vmin=0.0, vmax=max(float(np.max(nodal_err)), 1.0e-30)), "magma"),
    ]
    stress_collection = None
    error_collection = None
    for ax, (values, title, norm, cmap) in zip(axes, panels, strict=True):
        collection = _draw_3d_surface(ax, mesh, values, norm=norm, cmap=cmap, title=title)
        if title == "absolute error":
            error_collection = collection
        else:
            stress_collection = collection
    fig.subplots_adjust(left=0.01, right=0.86, bottom=0.04, top=0.84, wspace=0.02)
    if stress_collection is not None:
        cax = fig.add_axes((0.88, 0.22, 0.018, 0.52))
        fig.colorbar(stress_collection, cax=cax, label="von Mises")
    if error_collection is not None:
        cax_err = fig.add_axes((0.94, 0.22, 0.018, 0.52))
        fig.colorbar(error_collection, cax=cax_err, label="abs. error")
    fig.suptitle(f"External FEM 3D boundary-surface stress comparison, resolution {finest}")
    png = out_dir / "external_fem_stress_3d.png"
    pdf = out_dir / "external_fem_stress_3d.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"plot": "external_fem_stress_3d", "png": png.name, "pdf": pdf.name, "status": "ok"}


def write_plots(out_dir: Path, summary_rows: list[Row], cloud_rows: list[Row]) -> list[Row]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        _plot_error_trends(out_dir, summary_rows),
        _plot_stress_cloud(out_dir, cloud_rows),
        _plot_stress_3d(out_dir, cloud_rows),
    ]


def write_markdown(path: Path, summary_rows: list[Row], cloud_rows: list[Row], plots: list[Row]) -> None:
    max_disp = max(float(row["displacement_l2_rel_error"]) for row in summary_rows)
    max_stress = max(float(row["stress_l2_rel_error"]) for row in summary_rows)
    max_vm = max(float(row["von_mises_l2_rel_error"]) for row in summary_rows)
    max_K = max(float(row["stiffness_fro_rel_error"]) for row in summary_rows)
    lines = [
        "# External Open-Source FEM Comparison",
        "",
        "This validation compares the standalone SFC linear TET4 cantilever against scikit-fem on the same mesh, material, fixed boundary, and point load.",
        "",
        "## Command",
        "",
        "```bash",
        "python validation/run_external_fem_comparison.py --out-dir results/external_fem",
        "```",
        "",
        "## External Solver",
        "",
        f"- solver: scikit-fem",
        f"- version: {skfem.__version__}",
        "- role: validation-only external open-source FEM reference",
        "",
        "## Summary Metrics",
        "",
        "| Resolution | Elements | Tip uz SFC | Tip uz scikit-fem | Displacement rel. error | Stress rel. error | VM rel. error | K rel. error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['resolution']} | {row['elements']} | {_format_float(row['tip_displacement_z_sfc'])} | "
            f"{_format_float(row['tip_displacement_z_external'])} | {_format_float(row['displacement_l2_rel_error'])} | "
            f"{_format_float(row['stress_l2_rel_error'])} | {_format_float(row['von_mises_l2_rel_error'])} | "
            f"{_format_float(row['stiffness_fro_rel_error'])} |"
        )
    lines.extend(
        [
            "",
            "## Claim Gate",
            "",
            "| Claim | Evidence | Gate value | Status |",
            "| --- | --- | ---: | --- |",
            f"| SFC agrees with external open-source FEM for the linear TET4 cantilever | external_fem_comparison.csv::displacement_l2_rel_error | {_format_float(max_disp)} | {'supported' if max(max_disp, max_stress, max_vm, max_K) < 1.0e-8 else 'check'} |",
            "<!-- evidence csv=external_fem_comparison.csv field=displacement_l2_rel_error -->",
            f"| SFC stress recovery agrees with external-solver displacement field | external_fem_comparison.csv::stress_l2_rel_error | {_format_float(max_stress)} | {'supported' if max_stress < 1.0e-8 else 'check'} |",
            "<!-- evidence csv=external_fem_comparison.csv field=stress_l2_rel_error -->",
            "",
            "## Stress-Cloud Data",
            "",
            f"- rows: {len(cloud_rows)}",
            f"- max von Mises absolute error: {_format_float(max(float(row['von_mises_abs_error']) for row in cloud_rows))}",
            "",
            "## Plots",
            "",
        ]
    )
    for plot in plots:
        lines.append(f"- `{plot['png']}` and `{plot['pdf']}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an external open-source FEM comparison, not an internal refinement-only reference.",
            "- The comparison uses identical linear TET4 mesh topology, material parameters, Dirichlet constraints, and nodal load.",
            "- Stress and strain are compared as derived element quantities from each solver's displacement field on the same TET4 elements.",
            "- Stress plots average element von Mises values to nodes and interpolate over subdivided boundary triangles for visualization.",
            "- The result does not validate nonlinear FEM, friction, self-contact, or arbitrary industrial contact algorithms.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_external_comparison(*, quick: bool, out_dir: Path) -> dict[str, list[Row]]:
    resolutions = [1, 2] if quick else [1, 2, 3, 4]
    summary_rows, cloud_rows = external_fem_comparison(resolutions)
    plots = write_plots(out_dir, summary_rows, cloud_rows)
    return {"summary": summary_rows, "cloud": cloud_rows, "plots": plots}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a reduced deterministic comparison for tests.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "external_fem")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = run_external_comparison(quick=bool(args.quick), out_dir=args.out_dir)
    _write_csv(
        args.out_dir / "external_fem_comparison.csv",
        [
            "case",
            "external_solver",
            "external_solver_version",
            "resolution",
            "nodes",
            "elements",
            "fixed_nodes",
            "tip_displacement_z_sfc",
            "tip_displacement_z_external",
            "tip_displacement_rel_error",
            "displacement_l2_rel_error",
            "displacement_linf_abs_error",
            "stiffness_fro_rel_error",
            "strain_l2_rel_error",
            "stress_l2_rel_error",
            "von_mises_l2_rel_error",
            "max_von_mises_sfc",
            "max_von_mises_external",
            "max_von_mises_abs_error",
            "status",
            "details",
        ],
        outputs["summary"],
    )
    _write_csv(
        args.out_dir / "external_fem_stress_cloud.csv",
        [
            "case",
            "external_solver",
            "resolution",
            "element_id",
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "von_mises_sfc",
            "von_mises_external",
            "von_mises_abs_error",
            "status",
            "details",
        ],
        outputs["cloud"],
    )
    _write_csv(args.out_dir / "external_fem_plots.csv", ["plot", "png", "pdf", "status"], outputs["plots"])
    write_markdown(args.out_dir / "external_fem_summary.md", outputs["summary"], outputs["cloud"], outputs["plots"])
    for name in [
        "external_fem_comparison.csv",
        "external_fem_stress_cloud.csv",
        "external_fem_plots.csv",
        "external_fem_summary.md",
    ]:
        print(f"Wrote {args.out_dir / name}")
    for plot in outputs["plots"]:
        print(f"Wrote {args.out_dir / plot['png']}")
        print(f"Wrote {args.out_dir / plot['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
