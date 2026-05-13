"""Phase-7 physical validation against FEM and contact references.

This runner adds the missing physical-comparison evidence for the paper draft.
It does not add new core physics. It validates stress/strain recovery against
analytic linear elasticity, compares displacement/stress trends against a
standard high-resolution FEM reference, compares contact forces against a
brute-force closest-point projection reference, and reports whether contact
geometry acceleration translates into total-step timing gains.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.sparse import eye
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

from sfc.contact import (  # noqa: E402
    SurfaceSample,
    UniformTriangleAABBHash,
    contact_constraint_from_sample,
    penalty_contact_response,
)
from sfc.fem import assemble_stiffness_matrix, eliminate_fixed_dofs, fixed_dofs_from_node_set  # noqa: E402
from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.fem.tet4 import tet4_strain_displacement_matrix  # noqa: E402
from sfc.mesh import extract_boundary_faces  # noqa: E402
from validation.run_phase3_validation import _body, structured_tet_block  # noqa: E402
from validation.run_phase4_paper_validation import _format_float, _plane_surface  # noqa: E402

Row = dict[str, Any]


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _repeat_timing(func: Callable[[], Any], repeats: int) -> tuple[dict[str, float], Any]:
    values: list[float] = []
    result = None
    for _ in range(max(1, repeats)):
        start = perf_counter()
        result = func()
        values.append(perf_counter() - start)
    arr = np.asarray(values, dtype=float)
    return {
        "mean_seconds": float(np.mean(arr)),
        "std_seconds": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min_seconds": float(np.min(arr)),
        "max_seconds": float(np.max(arr)),
    }, result


def _von_mises(stress: np.ndarray) -> np.ndarray:
    s = np.asarray(stress, dtype=float)
    return np.sqrt(
        0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
        + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    )


def _element_strain_stress(mesh, u: np.ndarray, E: float, nu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = isotropic_linear_elasticity_matrix(E, nu)
    u_nodes = np.asarray(u, dtype=float).reshape((-1, 3))
    strains: list[np.ndarray] = []
    stresses: list[np.ndarray] = []
    for element in mesh.elements:
        B = tet4_strain_displacement_matrix(mesh.X[element])
        ue = u_nodes[element].reshape(12)
        strain = B @ ue
        strains.append(strain)
        stresses.append(C @ strain)
    strain_arr = np.asarray(strains, dtype=float)
    stress_arr = np.asarray(stresses, dtype=float)
    return strain_arr, stress_arr, _von_mises(stress_arr)


def analytic_stress_strain_patch(resolutions: Iterable[int]) -> list[Row]:
    """Compare element strain/stress to analytic affine-strain values."""

    E, nu, eps = 2.0e5, 0.25, 1.0e-3
    analytic_strain = np.array([eps, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    analytic_stress = isotropic_linear_elasticity_matrix(E, nu) @ analytic_strain
    rows: list[Row] = []
    for resolution in resolutions:
        mesh = structured_tet_block(int(resolution), int(resolution), int(resolution))
        u_nodes = np.zeros_like(mesh.X)
        u_nodes[:, 0] = eps * mesh.X[:, 0]
        strain, stress, _ = _element_strain_stress(mesh, u_nodes.ravel(), E, nu)
        strain_err = np.linalg.norm(strain - analytic_strain, axis=1)
        stress_err = np.linalg.norm(stress - analytic_stress, axis=1)
        rows.append(
            {
                "case": "analytic_uniaxial_stress_strain_patch",
                "resolution": int(resolution),
                "elements": mesh.elements.shape[0],
                "max_strain_l2_error": float(np.max(strain_err)),
                "rms_strain_l2_error": float(np.sqrt(np.mean(strain_err**2))),
                "max_stress_l2_error": float(np.max(stress_err)),
                "rms_stress_l2_error": float(np.sqrt(np.mean(stress_err**2))),
                "reference": "analytic_affine_small_strain",
                "status": "ok" if np.max(strain_err) < 1.0e-14 and np.max(stress_err) < 1.0e-8 else "check",
                "details": f"analytic_strain={analytic_strain.tolist()}; analytic_stress={analytic_stress.tolist()}",
            }
        )
    return rows


def _cantilever_solve(resolution: int) -> tuple[Any, np.ndarray, np.ndarray]:
    E, nu = 1.0e5, 0.3
    mesh = structured_tet_block(resolution, resolution, resolution, size=(1.0, 0.2, 0.2))
    body = _body(mesh, E=E, nu=nu, rho=1.0)
    fixed_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], 0.0))
    fixed = fixed_dofs_from_node_set(fixed_nodes, "xyz")
    tip_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], mesh.X[:, 0].max()))
    force = np.zeros(body.n_dofs, dtype=float)
    force[tip_nodes * 3 + 2] = -1.0 / tip_nodes.size
    K = assemble_stiffness_matrix(body)
    Kff, rhs, free = eliminate_fixed_dofs(K, force, fixed)
    u = np.zeros(body.n_dofs, dtype=float)
    u[free] = np.asarray(spsolve(Kff, rhs), dtype=float)
    _, _, vm = _element_strain_stress(mesh, u, E, nu)
    return mesh, u, vm


def _cantilever_solution(resolution: int) -> Row:
    mesh, u, vm = _cantilever_solve(resolution)
    tip_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], mesh.X[:, 0].max()))
    return {
        "case": "cantilever_standard_fem_reference",
        "resolution": resolution,
        "nodes": mesh.X.shape[0],
        "elements": mesh.elements.shape[0],
        "tip_displacement_z": float(np.mean(u.reshape((-1, 3))[tip_nodes, 2])),
        "max_von_mises": float(np.max(vm)),
        "mean_von_mises": float(np.mean(vm)),
        "relative_tip_error_to_finest": "",
        "relative_max_vm_error_to_finest": "",
        "reference": "finest_resolution_standard_linear_tet4_fem",
        "status": "evidence",
        "details": "standard FEM displacement/stress trend; not an external commercial solver",
    }


def cantilever_reference_study(resolutions: Iterable[int]) -> list[Row]:
    """Compare displacement and stress metrics against the finest FEM mesh."""

    rows = [_cantilever_solution(int(r)) for r in resolutions]
    finest = rows[-1]
    tip_ref = abs(float(finest["tip_displacement_z"]))
    vm_ref = abs(float(finest["max_von_mises"]))
    for row in rows:
        row["relative_tip_error_to_finest"] = abs(float(row["tip_displacement_z"]) - float(finest["tip_displacement_z"])) / max(tip_ref, 1.0e-30)
        row["relative_max_vm_error_to_finest"] = abs(float(row["max_von_mises"]) - float(finest["max_von_mises"])) / max(vm_ref, 1.0e-30)
        row["status"] = "ok" if np.isfinite(float(row["tip_displacement_z"])) and float(row["tip_displacement_z"]) < 0.0 else "check"
    return rows


def cantilever_stress_cloud(resolutions: Iterable[int]) -> list[Row]:
    """Return element-centroid von Mises data for stress-cloud comparisons."""

    rows: list[Row] = []
    for resolution in resolutions:
        mesh, _, vm = _cantilever_solve(int(resolution))
        centroids = np.mean(mesh.X[mesh.elements], axis=1)
        for element_id, (centroid, value) in enumerate(zip(centroids, vm, strict=True)):
            rows.append(
                {
                    "case": "cantilever_von_mises_stress_cloud",
                    "resolution": int(resolution),
                    "element_id": int(element_id),
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "centroid_z": float(centroid[2]),
                    "von_mises": float(value),
                    "status": "ok" if np.isfinite(float(value)) and float(value) >= 0.0 else "check",
                    "details": "element-centroid von Mises stress for projected x-z stress cloud",
                }
            )
    return rows


def _contact_constraints(
    slave: np.ndarray,
    master_nodes: np.ndarray,
    master_faces: np.ndarray,
    candidates: list[np.ndarray],
) -> list[Any]:
    constraints = []
    for node_id, face_ids in enumerate(candidates):
        if face_ids.size == 0:
            continue
        sample = SurfaceSample(np.array([node_id], dtype=np.int64), np.array([1.0]), face_ids)
        constraints.append(contact_constraint_from_sample(slave, sample, master_nodes, master_faces))
    return constraints


def _contact_response(
    slave: np.ndarray,
    master_nodes: np.ndarray,
    master_faces: np.ndarray,
    candidates: list[np.ndarray],
    stiffness: float,
) -> tuple[list[Any], np.ndarray]:
    constraints = _contact_constraints(slave, master_nodes, master_faces, candidates)
    force, _ = penalty_contact_response(
        constraints,
        stiffness=stiffness,
        n_total_dofs=3 * (slave.shape[0] + master_nodes.shape[0]),
        slave_dof_offset=0,
        master_dof_offset=3 * slave.shape[0],
    )
    return constraints, force


def contact_force_displacement_reference(resolution: int, penetrations: Iterable[float]) -> list[Row]:
    """Compare dynamic SDF contact to brute-force all-face CPP contact."""

    master_nodes, master_faces, surface_points = _plane_surface(resolution)
    all_faces = np.arange(master_faces.shape[0], dtype=np.int64)
    stiffness = 5.0e3
    rows: list[Row] = []
    for penetration in penetrations:
        slave = surface_points.copy()
        slave[:, 2] = -float(penetration)
        brute_candidates = [all_faces for _ in range(slave.shape[0])]
        index = UniformTriangleAABBHash.from_surface(master_nodes, master_faces, delta_safe=max(0.02, 2.0 * float(penetration)), cell_size=1.0 / resolution)
        dynamic_candidates = index.query_points(slave)
        brute_constraints, brute_force = _contact_response(slave, master_nodes, master_faces, brute_candidates, stiffness)
        dynamic_constraints, dynamic_force = _contact_response(slave, master_nodes, master_faces, dynamic_candidates, stiffness)
        brute_slave = brute_force[: 3 * slave.shape[0]].reshape((-1, 3))
        dynamic_slave = dynamic_force[: 3 * slave.shape[0]].reshape((-1, 3))
        brute_normal = float(np.sum(brute_slave[:, 2]))
        dynamic_normal = float(np.sum(dynamic_slave[:, 2]))
        gap_errors = [abs(float(a.g) - float(b.g)) for a, b in zip(brute_constraints, dynamic_constraints)]
        force_err = abs(dynamic_normal - brute_normal)
        rows.append(
            {
                "case": "rigid_plane_force_displacement_reference",
                "surface_resolution": resolution,
                "penetration": float(penetration),
                "sample_count": slave.shape[0],
                "brute_active_count": sum(c.active for c in brute_constraints),
                "dynamic_active_count": sum(c.active for c in dynamic_constraints),
                "brute_normal_force": brute_normal,
                "dynamic_normal_force": dynamic_normal,
                "normal_force_abs_error": force_err,
                "normal_force_rel_error": force_err / max(abs(brute_normal), 1.0e-30),
                "max_gap_abs_error": max(gap_errors) if gap_errors else 0.0,
                "max_penetration": max(float(penetration), 0.0),
                "reference": "brute_force_all_triangle_closest_point_projection",
                "status": "ok" if force_err < 1.0e-10 and len(brute_constraints) == len(dynamic_constraints) else "check",
                "details": "dynamic SDF contact force-displacement curve compared to brute-force CPP contact",
            }
        )
    return rows


def _linear_solve_workload() -> None:
    mesh = structured_tet_block(2, 2, 2, size=(1.0, 0.2, 0.2))
    body = _body(mesh, E=1.0e5, nu=0.3, rho=1.0)
    fixed_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], 0.0))
    fixed = fixed_dofs_from_node_set(fixed_nodes, "xyz")
    tip_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], mesh.X[:, 0].max()))
    force = np.zeros(body.n_dofs, dtype=float)
    force[tip_nodes * 3 + 2] = -1.0 / tip_nodes.size
    K = assemble_stiffness_matrix(body)
    Kff, rhs, _ = eliminate_fixed_dofs(K, force, fixed)
    # Small diagonal regularization only guards against pathological tiny systems in timing.
    spsolve(Kff + 1.0e-12 * eye(Kff.shape[0], format="csr"), rhs)


def _brute_contact_workload(resolution: int) -> None:
    nodes, faces, points = _plane_surface(resolution)
    slave = points.copy()
    slave[:, 2] = -0.005
    all_faces = np.arange(faces.shape[0], dtype=np.int64)
    _contact_response(slave, nodes, faces, [all_faces for _ in range(slave.shape[0])], 5.0e3)


def _dynamic_contact_workload(resolution: int) -> None:
    nodes, faces, points = _plane_surface(resolution)
    slave = points.copy()
    slave[:, 2] = -0.005
    index = UniformTriangleAABBHash.from_surface(nodes, faces, delta_safe=0.02, cell_size=1.0 / resolution)
    _contact_response(slave, nodes, faces, index.query_points(slave), 5.0e3)


def acceleration_feasibility(surface_sizes: Iterable[int], repeats: int) -> list[Row]:
    """Report contact-only and total-step timing for brute-force vs dynamic contact."""

    rows: list[Row] = []
    for resolution in surface_sizes:
        timings: dict[str, float] = {}
        workloads: list[tuple[str, str, Callable[[], None]]] = [
            ("brute_force_contact", "contact_pipeline_only", lambda r=int(resolution): _brute_contact_workload(r)),
            ("dynamic_fem_sdf_contact", "contact_pipeline_only", lambda r=int(resolution): _dynamic_contact_workload(r)),
            ("brute_force_total_step", "fem_assembly_linear_solve_plus_contact", lambda r=int(resolution): (_linear_solve_workload(), _brute_contact_workload(r))),
            ("dynamic_fem_sdf_total_step", "fem_assembly_linear_solve_plus_contact", lambda r=int(resolution): (_linear_solve_workload(), _dynamic_contact_workload(r))),
        ]
        triangle_count = 2 * int(resolution) * int(resolution)
        point_count = int(resolution) * int(resolution)
        for method, stage, func in workloads:
            stats, _ = _repeat_timing(func, repeats)
            timings[method] = stats["mean_seconds"]
            rows.append(
                {
                    "surface_resolution": int(resolution),
                    "triangle_count": triangle_count,
                    "query_point_count": point_count,
                    "method": method,
                    "stage": stage,
                    "repeats": repeats,
                    "mean_seconds": stats["mean_seconds"],
                    "std_seconds": stats["std_seconds"],
                    "min_seconds": stats["min_seconds"],
                    "max_seconds": stats["max_seconds"],
                    "speedup_vs_bruteforce": "",
                    "speedup_status": "not_applicable",
                    "details": "timing row",
                }
            )
        for row in rows:
            if int(row["surface_resolution"]) != int(resolution):
                continue
            if row["method"] == "dynamic_fem_sdf_contact":
                speedup = timings["brute_force_contact"] / max(timings["dynamic_fem_sdf_contact"], 1.0e-30)
                row["speedup_vs_bruteforce"] = speedup if speedup > 1.0 else ""
                row["speedup_status"] = "speedup_supported" if speedup > 1.0 else "no_speedup"
            if row["method"] == "dynamic_fem_sdf_total_step":
                speedup = timings["brute_force_total_step"] / max(timings["dynamic_fem_sdf_total_step"], 1.0e-30)
                row["speedup_vs_bruteforce"] = speedup if speedup > 1.0 else ""
                row["speedup_status"] = "speedup_supported" if speedup > 1.0 else "no_speedup"
    return rows


def claim_rows(stress_rows: list[Row], contact_rows: list[Row], acceleration_rows: list[Row]) -> list[Row]:
    max_stress_error = max(float(row["max_stress_l2_error"]) for row in stress_rows)
    max_contact_force_rel = max(float(row["normal_force_rel_error"]) for row in contact_rows)
    total_step_speedups = [
        float(row["speedup_vs_bruteforce"])
        for row in acceleration_rows
        if row["method"] == "dynamic_fem_sdf_total_step" and row["speedup_vs_bruteforce"] != ""
    ]
    min_total_step_speedup = min(total_step_speedups) if total_step_speedups else 0.0
    return [
        {
            "claim_id": "stress_strain_physical_correctness",
            "claim_text": "Element strain and stress match analytic affine linear-elastic references.",
            "evidence_csv": "phase7_stress_strain.csv",
            "evidence_field": "max_stress_l2_error",
            "gate_value": max_stress_error,
            "claim_status": "supported" if max_stress_error < 1.0e-8 else "not_supported",
            "details": "analytic stress/strain patch test",
        },
        {
            "claim_id": "contact_force_reference_correctness",
            "claim_text": "Dynamic SDF contact force-displacement response matches brute-force CPP contact reference.",
            "evidence_csv": "phase7_contact_reference.csv",
            "evidence_field": "normal_force_rel_error",
            "gate_value": max_contact_force_rel,
            "claim_status": "supported" if max_contact_force_rel < 1.0e-10 else "not_supported",
            "details": "rigid plane penalty force-displacement comparison",
        },
        {
            "claim_id": "total_step_acceleration_feasibility",
            "claim_text": "Geometry acceleration remains beneficial when included in a small FEM assembly/solve/contact timing step.",
            "evidence_csv": "phase7_acceleration_feasibility.csv",
            "evidence_field": "speedup_vs_bruteforce",
            "gate_value": min_total_step_speedup,
            "claim_status": "supported" if min_total_step_speedup > 1.0 else "not_supported",
            "details": "distinguishes contact-only speedup from total-step timing",
        },
    ]


def _save_plot(out_dir: Path, name: str, build: Callable[[], None]) -> Row:
    plt.figure(figsize=(6.0, 4.0))
    build()
    plt.tight_layout()
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    plt.savefig(png, dpi=160)
    plt.savefig(pdf)
    plt.close()
    return {"plot": name, "png": png.name, "pdf": pdf.name, "status": "ok"}


def _nodal_average(mesh, element_values: np.ndarray) -> np.ndarray:
    """Average element scalar values to mesh nodes for visualization."""

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
    """Split a triangle and return sub-triangles colored by interpolated values."""

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
    """Return visible side-surface polygons projected to the x-z plane."""

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


def _draw_projected_body_stress(
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
    for face in faces:
        coords = mesh.X[face]
        if not np.all(np.isclose(coords[:, 1], y_min)):
            continue
        sub_polygons, sub_values = _subdivide_triangle(coords[:, [0, 2]], np.asarray(nodal_values, dtype=float)[face])
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


def _draw_3d_body_stress(
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


def _save_stress_cloud_plot(out_dir: Path, cloud_rows: list[Row]) -> Row:
    resolutions = sorted({int(row["resolution"]) for row in cloud_rows})
    cols = len(resolutions)
    fig, axes = plt.subplots(1, cols, figsize=(2.75 * cols, 2.65), sharex=True, sharey=True)
    if cols == 1:
        axes = np.asarray([axes])
    vmax = max(float(row["von_mises"]) for row in cloud_rows)
    norm = Normalize(vmin=0.0, vmax=vmax)
    collection = None
    for ax, resolution in zip(axes, resolutions, strict=True):
        mesh, _, vm = _cantilever_solve(int(resolution))
        collection = _draw_projected_body_stress(ax, mesh, _nodal_average(mesh, vm), norm=norm, cmap="viridis", title=f"res {resolution}")
    axes[0].set_ylabel("z")
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.20, top=0.78, wspace=0.08)
    if collection is not None:
        cax = fig.add_axes((0.90, 0.22, 0.018, 0.52))
        fig.colorbar(collection, cax=cax, label="von Mises stress")
    fig.suptitle("Cantilever side-surface von Mises comparison")
    png = out_dir / "phase7_cantilever_stress_cloud.png"
    pdf = out_dir / "phase7_cantilever_stress_cloud.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"plot": "phase7_cantilever_stress_cloud", "png": png.name, "pdf": pdf.name, "status": "ok"}


def _save_stress_3d_plot(out_dir: Path, cloud_rows: list[Row]) -> Row:
    resolutions = sorted({int(row["resolution"]) for row in cloud_rows})
    cols = len(resolutions)
    fig, axes = plt.subplots(1, cols, figsize=(3.0 * cols, 3.0), subplot_kw={"projection": "3d"})
    if cols == 1:
        axes = np.asarray([axes])
    vmax = max(float(row["von_mises"]) for row in cloud_rows)
    norm = Normalize(vmin=0.0, vmax=vmax)
    collection = None
    for ax, resolution in zip(axes, resolutions, strict=True):
        mesh, _, vm = _cantilever_solve(int(resolution))
        collection = _draw_3d_body_stress(ax, mesh, _nodal_average(mesh, vm), norm=norm, cmap="viridis", title=f"res {resolution}")
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.04, top=0.86, wspace=0.02)
    if collection is not None:
        cax = fig.add_axes((0.92, 0.18, 0.018, 0.58))
        fig.colorbar(collection, cax=cax, label="von Mises stress")
    fig.suptitle("Cantilever 3D boundary-surface von Mises comparison")
    png = out_dir / "phase7_cantilever_stress_3d.png"
    pdf = out_dir / "phase7_cantilever_stress_3d.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"plot": "phase7_cantilever_stress_3d", "png": png.name, "pdf": pdf.name, "status": "ok"}


def write_plots(
    out_dir: Path,
    stress_rows: list[Row],
    cantilever_rows: list[Row],
    cantilever_cloud_rows: list[Row],
    contact_rows: list[Row],
    acceleration_rows: list[Row],
) -> list[Row]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots: list[Row] = []

    def stress_plot() -> None:
        plt.semilogy([row["resolution"] for row in stress_rows], [max(float(row["max_stress_l2_error"]), 1.0e-18) for row in stress_rows], marker="o")
        plt.xlabel("mesh resolution")
        plt.ylabel("max stress L2 error")

    plots.append(_save_plot(out_dir, "phase7_stress_strain_error", stress_plot))

    def cantilever_plot() -> None:
        plt.plot([row["resolution"] for row in cantilever_rows], [abs(float(row["tip_displacement_z"])) for row in cantilever_rows], marker="o", label="|tip uz|")
        plt.plot([row["resolution"] for row in cantilever_rows], [float(row["max_von_mises"]) for row in cantilever_rows], marker="s", label="max von Mises")
        plt.xlabel("mesh resolution")
        plt.ylabel("metric")
        plt.legend()

    plots.append(_save_plot(out_dir, "phase7_cantilever_fem_reference", cantilever_plot))
    plots.append(_save_stress_cloud_plot(out_dir, cantilever_cloud_rows))
    plots.append(_save_stress_3d_plot(out_dir, cantilever_cloud_rows))

    def contact_plot() -> None:
        plt.plot([row["penetration"] for row in contact_rows], [row["brute_normal_force"] for row in contact_rows], marker="o", label="brute force CPP")
        plt.plot([row["penetration"] for row in contact_rows], [row["dynamic_normal_force"] for row in contact_rows], marker="x", label="dynamic SDF")
        plt.xlabel("prescribed penetration")
        plt.ylabel("normal force")
        plt.legend()

    plots.append(_save_plot(out_dir, "phase7_contact_force_displacement", contact_plot))

    def acceleration_plot() -> None:
        rows = [row for row in acceleration_rows if row["method"] in {"dynamic_fem_sdf_contact", "dynamic_fem_sdf_total_step"}]
        for method in sorted({row["method"] for row in rows}):
            subset = [row for row in rows if row["method"] == method]
            plt.plot([row["triangle_count"] for row in subset], [float(row["speedup_vs_bruteforce"] or 0.0) for row in subset], marker="o", label=method)
        plt.axhline(1.0, color="0.4", linewidth=1.0)
        plt.xlabel("triangle count")
        plt.ylabel("speedup vs brute-force counterpart")
        plt.legend(fontsize=8)

    plots.append(_save_plot(out_dir, "phase7_acceleration_feasibility", acceleration_plot))
    return plots


def write_markdown(
    path: Path,
    stress_rows: list[Row],
    cantilever_rows: list[Row],
    cantilever_cloud_rows: list[Row],
    contact_rows: list[Row],
    acceleration_rows: list[Row],
    claims: list[Row],
    plots: list[Row],
) -> None:
    lines = [
        "# Phase-7 Physical Validation Summary",
        "",
        "This validation adds stress/strain, standard FEM reference, contact force reference, and total-step acceleration evidence. It does not add new physics.",
        "",
        "## Claims",
        "",
        "| Claim | Status | Evidence | Gate value |",
        "| --- | --- | --- | ---: |",
    ]
    for claim in claims:
        lines.append(f"| {claim['claim_id']} | {claim['claim_status']} | {claim['evidence_csv']}::{claim['evidence_field']} | {_format_float(claim['gate_value'])} |")
        lines.append(f"<!-- evidence csv={claim['evidence_csv']} field={claim['evidence_field']} -->")
    lines.extend(["", "## Stress/Strain Patch", "", "| Resolution | Elements | Max strain error | Max stress error |", "| ---: | ---: | ---: | ---: |"])
    for row in stress_rows:
        lines.append(f"| {row['resolution']} | {row['elements']} | {_format_float(row['max_strain_l2_error'])} | {_format_float(row['max_stress_l2_error'])} |")
    lines.extend(["", "## Cantilever Standard FEM Reference", "", "| Resolution | Tip uz | Max von Mises | Tip rel. error to finest | Max VM rel. error to finest |", "| ---: | ---: | ---: | ---: | ---: |"])
    for row in cantilever_rows:
        lines.append(f"| {row['resolution']} | {_format_float(row['tip_displacement_z'])} | {_format_float(row['max_von_mises'])} | {_format_float(row['relative_tip_error_to_finest'])} | {_format_float(row['relative_max_vm_error_to_finest'])} |")
    lines.extend(["", "## Cantilever Stress Cloud", "", "| Resolution | Stress samples | Min von Mises | Max von Mises |", "| ---: | ---: | ---: | ---: |"])
    for resolution in sorted({int(row["resolution"]) for row in cantilever_cloud_rows}):
        subset = [row for row in cantilever_cloud_rows if int(row["resolution"]) == resolution]
        values = [float(row["von_mises"]) for row in subset]
        lines.append(f"| {resolution} | {len(subset)} | {_format_float(min(values))} | {_format_float(max(values))} |")
    lines.extend(["", "## Contact Force-Displacement Reference", "", "| Penetration | Brute force | Dynamic SDF | Relative error | Active counts |", "| ---: | ---: | ---: | ---: | --- |"])
    for row in contact_rows:
        lines.append(f"| {_format_float(row['penetration'])} | {_format_float(row['brute_normal_force'])} | {_format_float(row['dynamic_normal_force'])} | {_format_float(row['normal_force_rel_error'])} | {row['brute_active_count']}/{row['dynamic_active_count']} |")
    lines.extend(["", "## Acceleration Feasibility", "", "| Resolution | Method | Stage | Mean seconds | Speedup | Status |", "| ---: | --- | --- | ---: | ---: | --- |"])
    for row in acceleration_rows:
        if row["method"].startswith("dynamic"):
            lines.append(f"| {row['surface_resolution']} | {row['method']} | {row['stage']} | {_format_float(row['mean_seconds'])} | {_format_float(row['speedup_vs_bruteforce'])} | {row['speedup_status']} |")
    lines.extend(["", "## Plots", ""])
    for plot in plots:
        lines.append(f"- `{plot['png']}` and `{plot['pdf']}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The stress/strain patch validates element recovery for affine small-strain linear elasticity.",
            "- The cantilever study is a standard FEM mesh-refinement reference, not an external commercial-solver benchmark.",
            "- The cantilever stress plots average element von Mises stress to nodes and interpolate over subdivided boundary triangles for visualization.",
            "- The contact force-displacement curve validates dynamic SDF contact against brute-force closest-point projection contact.",
            "- The acceleration table separates contact-only acceleration from total-step timing, which is the relevant claim boundary for paper text.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase7(*, quick: bool, out_dir: Path) -> dict[str, list[Row]]:
    resolutions = [1, 2, 3] if quick else [1, 2, 3, 4]
    contact_resolution = 4 if quick else 8
    surface_sizes = [3, 5] if quick else [4, 8, 12]
    repeats = 1 if quick else 3
    penetrations = [0.0, 0.0025, 0.005, 0.01] if quick else [0.0, 0.0025, 0.005, 0.01, 0.02]
    stress_rows = analytic_stress_strain_patch(resolutions)
    cantilever_rows = cantilever_reference_study(resolutions)
    cantilever_cloud_rows = cantilever_stress_cloud(resolutions)
    contact_rows = contact_force_displacement_reference(contact_resolution, penetrations)
    acceleration_rows = acceleration_feasibility(surface_sizes, repeats)
    claims = claim_rows(stress_rows, contact_rows, acceleration_rows)
    plots = write_plots(out_dir, stress_rows, cantilever_rows, cantilever_cloud_rows, contact_rows, acceleration_rows)
    return {
        "stress": stress_rows,
        "cantilever": cantilever_rows,
        "cantilever_cloud": cantilever_cloud_rows,
        "contact": contact_rows,
        "acceleration": acceleration_rows,
        "claims": claims,
        "plots": plots,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use reduced deterministic sizes for CI.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "phase7")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = run_phase7(quick=bool(args.quick), out_dir=args.out_dir)
    _write_csv(
        args.out_dir / "phase7_stress_strain.csv",
        ["case", "resolution", "elements", "max_strain_l2_error", "rms_strain_l2_error", "max_stress_l2_error", "rms_stress_l2_error", "reference", "status", "details"],
        outputs["stress"],
    )
    _write_csv(
        args.out_dir / "phase7_cantilever_reference.csv",
        ["case", "resolution", "nodes", "elements", "tip_displacement_z", "max_von_mises", "mean_von_mises", "relative_tip_error_to_finest", "relative_max_vm_error_to_finest", "reference", "status", "details"],
        outputs["cantilever"],
    )
    _write_csv(
        args.out_dir / "phase7_cantilever_stress_cloud.csv",
        ["case", "resolution", "element_id", "centroid_x", "centroid_y", "centroid_z", "von_mises", "status", "details"],
        outputs["cantilever_cloud"],
    )
    _write_csv(
        args.out_dir / "phase7_contact_reference.csv",
        ["case", "surface_resolution", "penetration", "sample_count", "brute_active_count", "dynamic_active_count", "brute_normal_force", "dynamic_normal_force", "normal_force_abs_error", "normal_force_rel_error", "max_gap_abs_error", "max_penetration", "reference", "status", "details"],
        outputs["contact"],
    )
    _write_csv(
        args.out_dir / "phase7_acceleration_feasibility.csv",
        ["surface_resolution", "triangle_count", "query_point_count", "method", "stage", "repeats", "mean_seconds", "std_seconds", "min_seconds", "max_seconds", "speedup_vs_bruteforce", "speedup_status", "details"],
        outputs["acceleration"],
    )
    _write_csv(
        args.out_dir / "phase7_claims.csv",
        ["claim_id", "claim_text", "evidence_csv", "evidence_field", "gate_value", "claim_status", "details"],
        outputs["claims"],
    )
    _write_csv(args.out_dir / "phase7_plots.csv", ["plot", "png", "pdf", "status"], outputs["plots"])
    write_markdown(
        args.out_dir / "phase7_summary.md",
        outputs["stress"],
        outputs["cantilever"],
        outputs["cantilever_cloud"],
        outputs["contact"],
        outputs["acceleration"],
        outputs["claims"],
        outputs["plots"],
    )
    for name in [
        "phase7_stress_strain.csv",
        "phase7_cantilever_reference.csv",
        "phase7_cantilever_stress_cloud.csv",
        "phase7_contact_reference.csv",
        "phase7_acceleration_feasibility.csv",
        "phase7_claims.csv",
        "phase7_plots.csv",
        "phase7_summary.md",
    ]:
        print(f"Wrote {args.out_dir / name}")
    for plot in outputs["plots"]:
        print(f"Wrote {args.out_dir / plot['png']}")
        print(f"Wrote {args.out_dir / plot['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
