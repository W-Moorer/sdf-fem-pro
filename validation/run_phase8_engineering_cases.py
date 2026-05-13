"""Phase-8 engineering validation cases for paper-facing evidence.

These cases add more realistic engineering-style checks without changing the
core solver physics.  They exercise the existing linear TET4 FEM assembly,
current-surface SDF contact, and penalty normal contact in three settings:

1. a 3D cantilever beam compared against scikit-fem,
2. a rigid flat indenter pressing an elastic block, and
3. two deformable blocks in normal contact with action-reaction checks.

The contact examples are prescribed-displacement validation cases, not nonlinear
contact solves.  They are intended to produce physically interpretable stress
and contact-force evidence for the current minimal solver scope.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
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
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

try:  # noqa: E402
    import skfem
    from skfem import Basis, ElementTetP1, ElementVector, MeshTet, asm, condense, solve
    from skfem.models.elasticity import lame_parameters, linear_elasticity
except ModuleNotFoundError as exc:  # pragma: no cover - optional validation dependency
    raise SystemExit(
        "scikit-fem is required for Phase-8 external FEM validation. "
        "Install it with `pip install scikit-fem` or `pip install -e .[validation]`."
    ) from exc

from sfc.contact import SurfaceSample, UniformTriangleAABBHash, contact_constraint_from_sample, penalty_contact_response  # noqa: E402
from sfc.fem import assemble_stiffness_matrix, eliminate_fixed_dofs, fixed_dofs_from_node_set  # noqa: E402
from sfc.mesh import extract_boundary_faces  # noqa: E402
from validation.run_external_fem_comparison import _relative_error, _sparse_frobenius_norm  # noqa: E402
from validation.run_phase3_validation import _body, structured_tet_block  # noqa: E402
from validation.run_phase4_paper_validation import _format_float  # noqa: E402
from validation.run_phase7_physical_validation import _element_strain_stress, _nodal_average, _subdivide_triangle  # noqa: E402

Row = dict[str, Any]


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _solve_static(mesh, force: np.ndarray, fixed_nodes: np.ndarray, E: float, nu: float) -> tuple[np.ndarray, Any]:
    body = _body(mesh, E=E, nu=nu, rho=1.0)
    fixed = fixed_dofs_from_node_set(fixed_nodes, "xyz")
    K = assemble_stiffness_matrix(body)
    Kff, rhs, free = eliminate_fixed_dofs(K, force, fixed)
    u = np.zeros(body.n_dofs, dtype=float)
    if free.size:
        u[free] = np.asarray(spsolve(Kff, rhs), dtype=float)
    return u, K


def _beam_mesh(resolution: int):
    r = int(resolution)
    return structured_tet_block(3 * r, r, r, size=(2.0, 0.35, 0.35))


def _cantilever_load(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixed_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], 0.0))
    tip_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], mesh.X[:, 0].max()))
    force = np.zeros(3 * mesh.X.shape[0], dtype=float)
    force[tip_nodes * 3 + 2] = -1.0 / tip_nodes.size
    return fixed_nodes, tip_nodes, force


def _solve_sfc_cantilever(mesh, E: float, nu: float) -> tuple[np.ndarray, Any]:
    fixed_nodes, _, force = _cantilever_load(mesh)
    return _solve_static(mesh, force, fixed_nodes, E, nu)


def _solve_scikit_cantilever(mesh, E: float, nu: float) -> tuple[np.ndarray, Any]:
    fixed_nodes, _, force = _cantilever_load(mesh)
    fixed = fixed_dofs_from_node_set(fixed_nodes, "xyz")
    sk_mesh = MeshTet(mesh.X.T, mesh.elements.T)
    basis = Basis(sk_mesh, ElementVector(ElementTetP1()))
    lambda_, mu = lame_parameters(E, nu)
    K = asm(linear_elasticity(lambda_, mu), basis)
    u = solve(*condense(K, force, D=fixed))
    return np.asarray(u, dtype=float), K


def _draw_surface_stress(
    ax,
    X: np.ndarray,
    elements: np.ndarray,
    nodal_values: np.ndarray,
    *,
    norm: Normalize,
    cmap: str,
    title: str,
    alpha: float = 1.0,
) -> Poly3DCollection:
    faces, _ = extract_boundary_faces(elements, X)
    polygons: list[np.ndarray] = []
    values: list[float] = []
    node_values = np.asarray(nodal_values, dtype=float)
    for face in faces:
        sub_polygons, sub_values = _subdivide_triangle(X[face], node_values[face], subdivisions=5)
        polygons.extend(sub_polygons)
        values.extend(sub_values)
    collection = Poly3DCollection(
        polygons,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        edgecolors=(0.18, 0.18, 0.18, 0.13),
        linewidths=0.055,
    )
    collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection3d(collection)
    ax.set_title(title, pad=0.0)
    return collection


def _finish_3d_axes(ax, all_points: np.ndarray, *, elev: float = 22.0, azim: float = -58.0) -> None:
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.5 * max(span, 1.0e-12)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.set_axis_off()


def _save_figure(fig, out_dir: Path, name: str) -> Row:
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"plot": name, "png": png.name, "pdf": pdf.name, "status": "ok"}


def cantilever_external_case(resolutions: list[int]) -> tuple[list[Row], dict[str, Any]]:
    E, nu = 1.0e5, 0.3
    rows: list[Row] = []
    finest: dict[str, Any] = {}
    for resolution in resolutions:
        mesh = _beam_mesh(int(resolution))
        _, tip_nodes, _ = _cantilever_load(mesh)
        u_sfc, K_sfc = _solve_sfc_cantilever(mesh, E, nu)
        u_ext, K_ext = _solve_scikit_cantilever(mesh, E, nu)
        strain_sfc, stress_sfc, vm_sfc = _element_strain_stress(mesh, u_sfc, E, nu)
        strain_ext, stress_ext, vm_ext = _element_strain_stress(mesh, u_ext, E, nu)
        stiffness_rel = _sparse_frobenius_norm(K_sfc - K_ext) / max(_sparse_frobenius_norm(K_ext), 1.0e-30)
        displacement_rel = _relative_error(u_sfc, u_ext)
        strain_rel = _relative_error(strain_sfc.ravel(), strain_ext.ravel())
        stress_rel = _relative_error(stress_sfc.ravel(), stress_ext.ravel())
        vm_rel = _relative_error(vm_sfc, vm_ext)
        tip_sfc = float(np.mean(u_sfc.reshape((-1, 3))[tip_nodes, 2]))
        tip_ext = float(np.mean(u_ext.reshape((-1, 3))[tip_nodes, 2]))
        status = "ok" if max(displacement_rel, stiffness_rel, strain_rel, stress_rel, vm_rel) < 1.0e-8 else "check"
        rows.append(
            {
                "case": "phase8_3d_cantilever_beam_external_fem",
                "external_solver": "scikit-fem",
                "external_solver_version": skfem.__version__,
                "resolution": int(resolution),
                "nodes": mesh.X.shape[0],
                "elements": mesh.elements.shape[0],
                "tip_displacement_z_sfc": tip_sfc,
                "tip_displacement_z_external": tip_ext,
                "displacement_l2_rel_error": displacement_rel,
                "stiffness_fro_rel_error": stiffness_rel,
                "strain_l2_rel_error": strain_rel,
                "stress_l2_rel_error": stress_rel,
                "von_mises_l2_rel_error": vm_rel,
                "max_von_mises_sfc": float(np.max(vm_sfc)),
                "max_von_mises_external": float(np.max(vm_ext)),
                "max_von_mises_abs_error": float(abs(np.max(vm_sfc) - np.max(vm_ext))),
                "status": status,
                "details": "3D beam mesh, same material/load/fixed boundary compared against scikit-fem",
            }
        )
        finest = {
            "mesh": mesh,
            "u_sfc": u_sfc,
            "u_external": u_ext,
            "vm_sfc": vm_sfc,
            "vm_external": vm_ext,
        }
    return rows, finest


def _square_platen(center: tuple[float, float], half_width: float, z: float, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = center
    n = max(1, int(resolution))
    nodes: list[tuple[float, float, float]] = []
    for j in range(n + 1):
        y = cy - half_width + 2.0 * half_width * j / n
        for i in range(n + 1):
            x = cx - half_width + 2.0 * half_width * i / n
            nodes.append((x, y, z))
    faces: list[tuple[int, int, int]] = []
    for j in range(n):
        for i in range(n):
            v00 = j * (n + 1) + i
            v10 = v00 + 1
            v01 = v00 + (n + 1)
            v11 = v01 + 1
            # Downward-oriented underside triangles.
            faces.append((v00, v01, v10))
            faces.append((v10, v01, v11))
    return np.asarray(nodes, dtype=float), np.asarray(faces, dtype=np.int64)


def _platen_prism_polygons(platen_x: np.ndarray, z_top: float) -> list[np.ndarray]:
    """Return a simple visible rigid-platen box for plotting only."""

    xmin, ymin, zmin = np.min(platen_x, axis=0)
    xmax, ymax, _ = np.max(platen_x, axis=0)
    zmax = float(z_top)
    p000 = np.array([xmin, ymin, zmin], dtype=float)
    p100 = np.array([xmax, ymin, zmin], dtype=float)
    p010 = np.array([xmin, ymax, zmin], dtype=float)
    p110 = np.array([xmax, ymax, zmin], dtype=float)
    p001 = np.array([xmin, ymin, zmax], dtype=float)
    p101 = np.array([xmax, ymin, zmax], dtype=float)
    p011 = np.array([xmin, ymax, zmax], dtype=float)
    p111 = np.array([xmax, ymax, zmax], dtype=float)
    return [
        np.asarray([p001, p101, p111, p011]),  # top
        np.asarray([p000, p100, p101, p001]),
        np.asarray([p100, p110, p111, p101]),
        np.asarray([p110, p010, p011, p111]),
        np.asarray([p010, p000, p001, p011]),
    ]


def _contact_constraints_for_nodes(
    slave_x: np.ndarray,
    slave_node_ids: np.ndarray,
    master_x: np.ndarray,
    master_faces: np.ndarray,
    *,
    delta_safe: float,
    cell_size: float,
) -> list[Any]:
    index = UniformTriangleAABBHash.from_surface(master_x, master_faces, delta_safe=delta_safe, cell_size=cell_size)
    constraints = []
    for node_id in slave_node_ids:
        candidates = index.query_point(slave_x[int(node_id)])
        if candidates.size == 0:
            continue
        sample = SurfaceSample(np.array([int(node_id)], dtype=np.int64), np.array([1.0]), candidates)
        constraints.append(contact_constraint_from_sample(slave_x, sample, master_x, master_faces))
    return constraints


def _split_contact_force(
    constraints: list[Any],
    *,
    stiffness: float,
    slave_node_count: int,
    master_node_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slave_dofs = 3 * slave_node_count
    force, _ = penalty_contact_response(
        constraints,
        stiffness=stiffness,
        n_total_dofs=slave_dofs + 3 * master_node_count,
        slave_dof_offset=0,
        master_dof_offset=slave_dofs,
    )
    return force[:slave_dofs], force[slave_dofs:], force


def rigid_indenter_case(indentations: list[float], resolution: int) -> tuple[list[Row], list[Row], dict[str, Any]]:
    E, nu = 1.0e5, 0.3
    stiffness = 7.5e4
    mesh = structured_tet_block(resolution, resolution, max(1, resolution // 2), size=(1.0, 1.0, 0.35))
    top_z = float(mesh.X[:, 2].max())
    half_width = 0.38
    top_nodes = np.flatnonzero(
        np.isclose(mesh.X[:, 2], top_z)
        & (np.abs(mesh.X[:, 0] - 0.5) <= half_width + 1.0e-12)
        & (np.abs(mesh.X[:, 1] - 0.5) <= half_width + 1.0e-12)
    )
    fixed_nodes = np.flatnonzero(np.isclose(mesh.X[:, 2], 0.0))
    rows: list[Row] = []
    stress_rows: list[Row] = []
    final_state: dict[str, Any] = {}
    for indentation in indentations:
        platen_x, platen_faces = _square_platen((0.5, 0.5), half_width, top_z - float(indentation), resolution)
        constraints = _contact_constraints_for_nodes(
            mesh.X,
            top_nodes,
            platen_x,
            platen_faces,
            delta_safe=max(0.03, float(indentation) + 0.01),
            cell_size=max(2.0 * half_width / max(1, resolution), 1.0e-6),
        )
        block_force, platen_force, _ = _split_contact_force(
            constraints,
            stiffness=stiffness,
            slave_node_count=mesh.X.shape[0],
            master_node_count=platen_x.shape[0],
        )
        u, _ = _solve_static(mesh, block_force, fixed_nodes, E, nu)
        _, _, vm = _element_strain_stress(mesh, u, E, nu)
        gaps = np.asarray([float(c.g) for c in constraints], dtype=float)
        penetrations = np.maximum(-gaps, 0.0) if gaps.size else np.zeros(0, dtype=float)
        block_total = block_force.reshape((-1, 3)).sum(axis=0)
        platen_total = platen_force.reshape((-1, 3)).sum(axis=0)
        active_count = int(np.count_nonzero(penetrations > 0.0))
        normal_force_z = float(block_total[2])
        status = "ok"
        if float(indentation) > 0.0 and not (active_count > 0 and normal_force_z < 0.0):
            status = "check"
        rows.append(
            {
                "case": "rigid_flat_indenter_pressing_elastic_block",
                "indentation": float(indentation),
                "slave_sample_count": int(top_nodes.size),
                "active_contact_count": active_count,
                "min_gap": float(np.min(gaps)) if gaps.size else 0.0,
                "max_penetration": float(np.max(penetrations)) if penetrations.size else 0.0,
                "normal_force_z": normal_force_z,
                "contact_energy": 0.5 * stiffness * float(np.dot(penetrations, penetrations)),
                "action_reaction_imbalance": float(np.linalg.norm(block_total + platen_total)),
                "mean_top_displacement_z": float(np.mean(u.reshape((-1, 3))[top_nodes, 2])),
                "max_von_mises": float(np.max(vm)),
                "status": status,
                "details": "prescribed flat rigid indenter underside; block bottom fixed; linear static response to penalty contact force",
            }
        )
        final_state = {
            "mesh": mesh,
            "u": u,
            "vm": vm,
            "platen_x": platen_x,
            "platen_faces": platen_faces,
            "indentation": float(indentation),
            "platen_plot_top_z": top_z + 0.35,
        }
    if final_state:
        centroids = np.mean(mesh.X[mesh.elements], axis=1)
        for element_id, (centroid, value) in enumerate(zip(centroids, final_state["vm"], strict=True)):
            stress_rows.append(
                {
                    "case": "rigid_flat_indenter_pressing_elastic_block_stress_cloud",
                    "indentation": final_state["indentation"],
                    "element_id": int(element_id),
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "centroid_z": float(centroid[2]),
                    "von_mises": float(value),
                    "status": "ok" if np.isfinite(float(value)) and float(value) >= 0.0 else "check",
                    "details": "element von Mises stress from linear block solve under penalty contact load",
                }
            )
    return rows, stress_rows, final_state


def _top_faces(mesh) -> np.ndarray:
    faces, _ = extract_boundary_faces(mesh.elements, mesh.X)
    zmax = float(mesh.X[:, 2].max())
    return np.asarray([face for face in faces if np.all(np.isclose(mesh.X[face, 2], zmax))], dtype=np.int64)


def deformable_deformable_case(indentations: list[float], resolution: int) -> tuple[list[Row], list[Row], dict[str, Any]]:
    E, nu = 1.0e5, 0.3
    stiffness = 6.0e4
    lower = structured_tet_block(resolution, resolution, max(1, resolution // 2), size=(1.0, 1.0, 0.25))
    upper = structured_tet_block(resolution, resolution, max(1, resolution // 2), size=(1.0, 1.0, 0.25))
    gap0 = 0.035
    upper_ref = upper.X.copy()
    upper_ref[:, 2] += float(lower.X[:, 2].max()) + gap0
    lower_top_faces = _top_faces(lower)
    upper_bottom_nodes = np.flatnonzero(np.isclose(upper_ref[:, 2], upper_ref[:, 2].min()))
    lower_fixed_nodes = np.flatnonzero(np.isclose(lower.X[:, 2], 0.0))
    upper_fixed_nodes = np.flatnonzero(np.isclose(upper_ref[:, 2], upper_ref[:, 2].max()))
    rows: list[Row] = []
    stress_rows: list[Row] = []
    final_state: dict[str, Any] = {}
    for indentation in indentations:
        upper_x = upper_ref.copy()
        upper_x[:, 2] -= float(indentation)
        constraints = _contact_constraints_for_nodes(
            upper_x,
            upper_bottom_nodes,
            lower.X,
            lower_top_faces,
            delta_safe=max(0.03, abs(float(indentation) - gap0) + 0.01),
            cell_size=max(1.0 / max(1, resolution), 1.0e-6),
        )
        upper_force, lower_force, _ = _split_contact_force(
            constraints,
            stiffness=stiffness,
            slave_node_count=upper_x.shape[0],
            master_node_count=lower.X.shape[0],
        )
        u_lower, _ = _solve_static(lower, lower_force, lower_fixed_nodes, E, nu)
        upper_view = SimpleNamespace(X=upper_ref, elements=upper.elements)
        u_upper, _ = _solve_static(upper_view, upper_force, upper_fixed_nodes, E, nu)
        _, _, vm_lower = _element_strain_stress(lower, u_lower, E, nu)
        _, _, vm_upper = _element_strain_stress(upper_view, u_upper, E, nu)
        gaps = np.asarray([float(c.g) for c in constraints], dtype=float)
        penetrations = np.maximum(-gaps, 0.0) if gaps.size else np.zeros(0, dtype=float)
        upper_total = upper_force.reshape((-1, 3)).sum(axis=0)
        lower_total = lower_force.reshape((-1, 3)).sum(axis=0)
        active_count = int(np.count_nonzero(penetrations > 0.0))
        status = "ok"
        if float(indentation) > gap0 and not (active_count > 0 and np.linalg.norm(upper_total + lower_total) < 1.0e-8):
            status = "check"
        rows.append(
            {
                "case": "deformable_deformable_block_contact_action_reaction",
                "indentation": float(indentation),
                "initial_gap": gap0,
                "slave_sample_count": int(upper_bottom_nodes.size),
                "active_contact_count": active_count,
                "min_gap": float(np.min(gaps)) if gaps.size else 0.0,
                "max_penetration": float(np.max(penetrations)) if penetrations.size else 0.0,
                "upper_normal_force_z": float(upper_total[2]),
                "lower_normal_force_z": float(lower_total[2]),
                "contact_energy": 0.5 * stiffness * float(np.dot(penetrations, penetrations)),
                "action_reaction_imbalance": float(np.linalg.norm(upper_total + lower_total)),
                "max_von_mises_lower": float(np.max(vm_lower)),
                "max_von_mises_upper": float(np.max(vm_upper)),
                "status": status,
                "details": "prescribed approach between two linear elastic blocks; lower bottom and upper top fixed for stress solves",
            }
        )
        final_state = {
            "lower": lower,
            "upper": upper_view,
            "upper_x": upper_x,
            "vm_lower": vm_lower,
            "vm_upper": vm_upper,
            "indentation": float(indentation),
        }
    if final_state:
        for body_name, mesh, X_current, values in [
            ("lower", final_state["lower"], final_state["lower"].X, final_state["vm_lower"]),
            ("upper", final_state["upper"], final_state["upper_x"], final_state["vm_upper"]),
        ]:
            centroids = np.mean(X_current[mesh.elements], axis=1)
            for element_id, (centroid, value) in enumerate(zip(centroids, values, strict=True)):
                stress_rows.append(
                    {
                        "case": "deformable_deformable_block_contact_stress_cloud",
                        "body": body_name,
                        "indentation": final_state["indentation"],
                        "element_id": int(element_id),
                        "centroid_x": float(centroid[0]),
                        "centroid_y": float(centroid[1]),
                        "centroid_z": float(centroid[2]),
                        "von_mises": float(value),
                        "status": "ok" if np.isfinite(float(value)) and float(value) >= 0.0 else "check",
                        "details": "element von Mises stress for two-block normal contact validation",
                    }
                )
    return rows, stress_rows, final_state


def claim_rows(cantilever: list[Row], indenter: list[Row], deformable: list[Row]) -> list[Row]:
    max_external_error = max(
        max(
            float(row["displacement_l2_rel_error"]),
            float(row["stiffness_fro_rel_error"]),
            float(row["stress_l2_rel_error"]),
            float(row["von_mises_l2_rel_error"]),
        )
        for row in cantilever
    )
    positive_indenter = [row for row in indenter if float(row["indentation"]) > 0.0]
    indenter_force_ok = all(float(row["normal_force_z"]) < 0.0 and int(row["active_contact_count"]) > 0 for row in positive_indenter)
    indenter_monotone = all(
        abs(float(b["normal_force_z"])) >= abs(float(a["normal_force_z"])) - 1.0e-12
        for a, b in zip(indenter, indenter[1:], strict=False)
    )
    max_deformable_imbalance = max(float(row["action_reaction_imbalance"]) for row in deformable)
    final_deformable_active = int(deformable[-1]["active_contact_count"]) > 0
    return [
        {
            "claim_id": "phase8_external_cantilever_fem_agreement",
            "claim_text": "The 3D cantilever beam solution agrees with an external open-source linear TET4 FEM solver.",
            "evidence_csv": "phase8_cantilever_external.csv",
            "evidence_field": "stress_l2_rel_error",
            "gate_value": max_external_error,
            "claim_status": "supported" if max_external_error < 1.0e-8 else "not_supported",
            "details": "scikit-fem comparison on matching 3D beam meshes",
        },
        {
            "claim_id": "phase8_rigid_indenter_contact_response",
            "claim_text": "A prescribed rigid flat indenter produces downward normal force, finite stress, and monotone force growth.",
            "evidence_csv": "phase8_rigid_indenter_history.csv",
            "evidence_field": "normal_force_z",
            "gate_value": min(float(row["normal_force_z"]) for row in positive_indenter) if positive_indenter else 0.0,
            "claim_status": "supported" if indenter_force_ok and indenter_monotone else "not_supported",
            "details": "normal penalty contact from current-surface local projection; no friction or nonlinear solve",
        },
        {
            "claim_id": "phase8_deformable_deformable_action_reaction",
            "claim_text": "Two-block normal contact preserves action-reaction balance in the assembled penalty contact forces.",
            "evidence_csv": "phase8_deformable_deformable_history.csv",
            "evidence_field": "action_reaction_imbalance",
            "gate_value": max_deformable_imbalance,
            "claim_status": "supported" if max_deformable_imbalance < 1.0e-8 and final_deformable_active else "not_supported",
            "details": "slave/master force balance for deformable-deformable contact samples",
        },
    ]


def write_plots(out_dir: Path, outputs: dict[str, Any]) -> list[Row]:
    plots: list[Row] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    finest = outputs["cantilever_finest"]
    mesh = finest["mesh"]
    vm_sfc = finest["vm_sfc"]
    vm_ext = finest["vm_external"]
    vmax = max(float(np.max(vm_sfc)), float(np.max(vm_ext)), 1.0e-30)
    err = np.abs(vm_sfc - vm_ext)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), subplot_kw={"projection": "3d"})
    norm = Normalize(vmin=0.0, vmax=vmax)
    err_norm = Normalize(vmin=0.0, vmax=max(float(np.max(err)), 1.0e-30))
    col0 = _draw_surface_stress(axes[0], mesh.X, mesh.elements, _nodal_average(mesh, vm_sfc), norm=norm, cmap="viridis", title="SFC")
    _draw_surface_stress(axes[1], mesh.X, mesh.elements, _nodal_average(mesh, vm_ext), norm=norm, cmap="viridis", title="scikit-fem")
    col2 = _draw_surface_stress(axes[2], mesh.X, mesh.elements, _nodal_average(mesh, err), norm=err_norm, cmap="magma", title="abs. error")
    for ax in axes:
        _finish_3d_axes(ax, mesh.X)
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.04, top=0.86, wspace=0.02)
    fig.colorbar(col0, ax=axes[:2], fraction=0.030, pad=0.02, label="von Mises stress")
    fig.colorbar(col2, ax=axes[2], fraction=0.050, pad=0.03, label="absolute error")
    fig.suptitle("Phase-8 3D cantilever external FEM stress comparison")
    plots.append(_save_figure(fig, out_dir, "phase8_cantilever_external_stress_3d"))

    indenter_rows = outputs["indenter"]
    fig, ax1 = plt.subplots(figsize=(6.2, 4.0))
    x = [float(row["indentation"]) for row in indenter_rows]
    force = [abs(float(row["normal_force_z"])) for row in indenter_rows]
    active = [int(row["active_contact_count"]) for row in indenter_rows]
    ax1.plot(x, force, marker="o", label="|normal force|")
    ax1.set_xlabel("prescribed indentation")
    ax1.set_ylabel("normal force magnitude")
    ax2 = ax1.twinx()
    ax2.plot(x, active, marker="s", color="tab:orange", label="active contacts")
    ax2.set_ylabel("active contact count")
    fig.tight_layout()
    plots.append(_save_figure(fig, out_dir, "phase8_rigid_indenter_force_history"))

    state = outputs["indenter_final"]
    mesh = state["mesh"]
    vm = state["vm"]
    fig = plt.figure(figsize=(5.6, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=0.0, vmax=max(float(np.max(vm)), 1.0e-30))
    col = _draw_surface_stress(ax, mesh.X, mesh.elements, _nodal_average(mesh, vm), norm=norm, cmap="viridis", title="flat indenter on elastic block")
    platen = Poly3DCollection(
        _platen_prism_polygons(state["platen_x"], state["platen_plot_top_z"]),
        facecolors=(0.12, 0.12, 0.12, 0.82),
        edgecolors=(0.03, 0.03, 0.03, 0.95),
        linewidths=0.45,
    )
    ax.add_collection3d(platen)
    xmin, ymin, zmin = np.min(state["platen_x"], axis=0)
    xmax, ymax, _ = np.max(state["platen_x"], axis=0)
    zmax = state["platen_plot_top_z"]
    top_loop = np.array(
        [
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
            [xmin, ymin, zmax],
        ],
        dtype=float,
    )
    ax.plot(top_loop[:, 0], top_loop[:, 1], top_loop[:, 2], color="0.02", linewidth=1.2)
    for x_corner, y_corner in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
        ax.plot([x_corner, x_corner], [y_corner, y_corner], [zmin, zmax], color="0.02", linewidth=0.9)
    platen_plot_points = np.vstack((state["platen_x"], state["platen_x"] + np.array([0.0, 0.0, state["platen_plot_top_z"] - np.min(state["platen_x"][:, 2])])))
    _finish_3d_axes(ax, np.vstack((mesh.X, platen_plot_points)), elev=24.0, azim=-50.0)
    fig.colorbar(col, ax=ax, fraction=0.034, pad=0.02, label="von Mises stress")
    fig.tight_layout()
    plots.append(_save_figure(fig, out_dir, "phase8_rigid_indenter_stress_3d"))

    deformable_rows = outputs["deformable"]
    fig, ax1 = plt.subplots(figsize=(6.2, 4.0))
    x = [float(row["indentation"]) for row in deformable_rows]
    force = [abs(float(row["upper_normal_force_z"])) for row in deformable_rows]
    imbalance = [float(row["action_reaction_imbalance"]) for row in deformable_rows]
    ax1.plot(x, force, marker="o", label="|upper normal force|")
    ax1.set_xlabel("prescribed approach")
    ax1.set_ylabel("normal force magnitude")
    ax2 = ax1.twinx()
    ax2.semilogy(x, [max(v, 1.0e-30) for v in imbalance], marker="s", color="tab:orange", label="imbalance")
    ax2.set_ylabel("action-reaction imbalance")
    fig.tight_layout()
    plots.append(_save_figure(fig, out_dir, "phase8_deformable_deformable_history"))

    state = outputs["deformable_final"]
    lower = state["lower"]
    upper = state["upper"]
    upper_x = state["upper_x"]
    vm_lower = state["vm_lower"]
    vm_upper = state["vm_upper"]
    vmax = max(float(np.max(vm_lower)), float(np.max(vm_upper)), 1.0e-30)
    norm = Normalize(vmin=0.0, vmax=vmax)
    fig = plt.figure(figsize=(5.8, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    col = _draw_surface_stress(ax, lower.X, lower.elements, _nodal_average(lower, vm_lower), norm=norm, cmap="viridis", title="two deformable blocks")
    _draw_surface_stress(ax, upper_x, upper.elements, _nodal_average(upper, vm_upper), norm=norm, cmap="viridis", title="two deformable blocks")
    _finish_3d_axes(ax, np.vstack((lower.X, upper_x)), elev=23.0, azim=-52.0)
    fig.colorbar(col, ax=ax, fraction=0.034, pad=0.02, label="von Mises stress")
    fig.tight_layout()
    plots.append(_save_figure(fig, out_dir, "phase8_deformable_deformable_stress_3d"))

    return plots


def write_markdown(path: Path, outputs: dict[str, Any]) -> None:
    claims = outputs["claims"]
    lines = [
        "# Phase-8 Engineering Validation Summary",
        "",
        "This validation adds three paper-facing engineering cases without adding new core physics: a 3D cantilever beam external FEM comparison, a rigid flat indenter pressing an elastic block, and deformable-deformable block contact with action-reaction diagnostics.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python validation/run_phase8_engineering_cases.py --out-dir results/phase8",
        "```",
        "",
        "For CI/reduced checks:",
        "",
        "```bash",
        "python validation/run_phase8_engineering_cases.py --quick --out-dir results/phase8",
        "```",
        "",
        "## Claims",
        "",
        "| Claim | Status | Evidence | Gate value |",
        "| --- | --- | --- | ---: |",
    ]
    for claim in claims:
        lines.append(f"| {claim['claim_id']} | {claim['claim_status']} | {claim['evidence_csv']}::{claim['evidence_field']} | {_format_float(claim['gate_value'])} |")
        lines.append(f"<!-- evidence csv={claim['evidence_csv']} field={claim['evidence_field']} -->")

    lines.extend(["", "## 3D Cantilever External FEM", "", "| Resolution | Elements | Disp. rel. err. | Stress rel. err. | K rel. err. |", "| ---: | ---: | ---: | ---: | ---: |"])
    for row in outputs["cantilever"]:
        lines.append(
            f"| {row['resolution']} | {row['elements']} | {_format_float(row['displacement_l2_rel_error'])} | {_format_float(row['stress_l2_rel_error'])} | {_format_float(row['stiffness_fro_rel_error'])} |"
        )

    lines.extend(["", "## Rigid Flat Indenter", "", "| Indentation | Active contacts | Min gap | Max penetration | Normal force z | Max VM |", "| ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in outputs["indenter"]:
        lines.append(
            f"| {_format_float(row['indentation'])} | {row['active_contact_count']} | {_format_float(row['min_gap'])} | {_format_float(row['max_penetration'])} | {_format_float(row['normal_force_z'])} | {_format_float(row['max_von_mises'])} |"
        )

    lines.extend(["", "## Deformable-Deformable Contact", "", "| Approach | Active contacts | Min gap | Max penetration | Upper force z | Lower force z | Imbalance |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in outputs["deformable"]:
        lines.append(
            f"| {_format_float(row['indentation'])} | {row['active_contact_count']} | {_format_float(row['min_gap'])} | {_format_float(row['max_penetration'])} | {_format_float(row['upper_normal_force_z'])} | {_format_float(row['lower_normal_force_z'])} | {_format_float(row['action_reaction_imbalance'])} |"
        )

    lines.extend(["", "## Plots", ""])
    for plot in outputs["plots"]:
        lines.append(f"- `{plot['png']}` and `{plot['pdf']}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The rigid indenter is modeled as a prescribed flat rigid underside surface, not a curved Hertzian indenter.",
            "- The contact cases are prescribed-displacement validation cases; they are not nonlinear contact equilibrium solves.",
            "- Contact remains frictionless normal penalty contact, and the FEM model remains small-strain linear TET4.",
            "- Stress plots use nodal averaging and subdivided boundary triangles for visualization; quantitative checks use CSV fields.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase8(*, quick: bool, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cantilever_resolutions = [1, 2] if quick else [1, 2, 3]
    contact_resolution = 3 if quick else 5
    indentations = [0.0, 0.01, 0.02] if quick else [0.0, 0.006, 0.012, 0.018, 0.024]
    approaches = [0.0, 0.03, 0.055] if quick else [0.0, 0.025, 0.04, 0.055, 0.07]

    cantilever_rows, cantilever_finest = cantilever_external_case(cantilever_resolutions)
    indenter_rows, indenter_stress_rows, indenter_final = rigid_indenter_case(indentations, contact_resolution)
    deformable_rows, deformable_stress_rows, deformable_final = deformable_deformable_case(approaches, contact_resolution)
    claims = claim_rows(cantilever_rows, indenter_rows, deformable_rows)
    outputs: dict[str, Any] = {
        "cantilever": cantilever_rows,
        "cantilever_finest": cantilever_finest,
        "indenter": indenter_rows,
        "indenter_stress": indenter_stress_rows,
        "indenter_final": indenter_final,
        "deformable": deformable_rows,
        "deformable_stress": deformable_stress_rows,
        "deformable_final": deformable_final,
        "claims": claims,
    }
    outputs["plots"] = write_plots(out_dir, outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use reduced deterministic sizes for CI.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "phase8")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_phase8(quick=bool(args.quick), out_dir=args.out_dir)
    _write_csv(
        args.out_dir / "phase8_cantilever_external.csv",
        [
            "case",
            "external_solver",
            "external_solver_version",
            "resolution",
            "nodes",
            "elements",
            "tip_displacement_z_sfc",
            "tip_displacement_z_external",
            "displacement_l2_rel_error",
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
        outputs["cantilever"],
    )
    _write_csv(
        args.out_dir / "phase8_rigid_indenter_history.csv",
        [
            "case",
            "indentation",
            "slave_sample_count",
            "active_contact_count",
            "min_gap",
            "max_penetration",
            "normal_force_z",
            "contact_energy",
            "action_reaction_imbalance",
            "mean_top_displacement_z",
            "max_von_mises",
            "status",
            "details",
        ],
        outputs["indenter"],
    )
    _write_csv(
        args.out_dir / "phase8_rigid_indenter_stress_cloud.csv",
        ["case", "indentation", "element_id", "centroid_x", "centroid_y", "centroid_z", "von_mises", "status", "details"],
        outputs["indenter_stress"],
    )
    _write_csv(
        args.out_dir / "phase8_deformable_deformable_history.csv",
        [
            "case",
            "indentation",
            "initial_gap",
            "slave_sample_count",
            "active_contact_count",
            "min_gap",
            "max_penetration",
            "upper_normal_force_z",
            "lower_normal_force_z",
            "contact_energy",
            "action_reaction_imbalance",
            "max_von_mises_lower",
            "max_von_mises_upper",
            "status",
            "details",
        ],
        outputs["deformable"],
    )
    _write_csv(
        args.out_dir / "phase8_deformable_deformable_stress_cloud.csv",
        ["case", "body", "indentation", "element_id", "centroid_x", "centroid_y", "centroid_z", "von_mises", "status", "details"],
        outputs["deformable_stress"],
    )
    _write_csv(
        args.out_dir / "phase8_claims.csv",
        ["claim_id", "claim_text", "evidence_csv", "evidence_field", "gate_value", "claim_status", "details"],
        outputs["claims"],
    )
    _write_csv(args.out_dir / "phase8_plots.csv", ["plot", "png", "pdf", "status"], outputs["plots"])
    write_markdown(args.out_dir / "phase8_summary.md", outputs)
    print(f"Wrote Phase-8 engineering validation outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
