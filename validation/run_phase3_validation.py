"""Phase-3 validation runner for the standalone FEM/SDF contact solver.

The cases here are deterministic smoke validations. They intentionally avoid
external reference files and exercise only the standalone ``sfc`` APIs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact import (  # noqa: E402
    SurfaceSample,
    UniformTriangleAABBHash,
    assemble_contact_jacobian,
    contact_constraint_from_sample,
    penalty_contact_response,
)
from sfc.fem import (  # noqa: E402
    DeformableBody,
    assemble_gravity_force,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    eliminate_fixed_dofs,
    fixed_dofs_from_node_set,
    newmark_beta_step,
)
from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.fem.tet4 import tet4_mass, tet4_stiffness, tet4_volume  # noqa: E402
from sfc.mesh import VolumeMesh, extract_boundary_faces  # noqa: E402
from sfc.sdf.local_projection import closest_point_on_triangle  # noqa: E402

ValidationRecord = dict[str, Any]


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Small result container for a validation case."""

    case: str
    metric: str
    value: float | str
    status: str
    details: str


def _record(
    category: str,
    case: str,
    metric: str,
    value: float | str,
    *,
    status: str = "ok",
    details: str = "",
    label: str = "",
    workload: str = "",
) -> ValidationRecord:
    return {
        "category": category,
        "case": case,
        "label": label,
        "metric": metric,
        "value": value,
        "workload": workload,
        "status": status,
        "details": details,
    }


def structured_tet_block(
    nx: int,
    ny: int,
    nz: int,
    *,
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> VolumeMesh:
    """Build a deterministic TET4 block mesh by splitting each hex into tets."""

    if min(nx, ny, nz) <= 0:
        raise ValueError("mesh resolution must be positive in all directions")

    lx, ly, lz = size
    nodes: list[tuple[float, float, float]] = []
    node_id: dict[tuple[int, int, int], int] = {}
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                node_id[(i, j, k)] = len(nodes)
                nodes.append((lx * i / nx, ly * j / ny, lz * k / nz))

    tets: list[tuple[int, int, int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v000 = node_id[(i, j, k)]
                v100 = node_id[(i + 1, j, k)]
                v010 = node_id[(i, j + 1, k)]
                v110 = node_id[(i + 1, j + 1, k)]
                v001 = node_id[(i, j, k + 1)]
                v101 = node_id[(i + 1, j, k + 1)]
                v011 = node_id[(i, j + 1, k + 1)]
                v111 = node_id[(i + 1, j + 1, k + 1)]
                tets.extend(
                    [
                        (v000, v100, v010, v001),
                        (v100, v110, v010, v111),
                        (v100, v010, v001, v111),
                        (v010, v001, v011, v111),
                        (v100, v001, v101, v111),
                    ]
                )

    X = np.asarray(nodes, dtype=float)
    return VolumeMesh(X=X, elements=np.asarray(tets, dtype=np.int64))


def _body(mesh: VolumeMesh, *, E: float = 1.0e5, nu: float = 0.3, rho: float = 2.0) -> DeformableBody:
    return DeformableBody(mesh=mesh, material={"E": E, "nu": nu}, density=rho)


def _solve_static(body: DeformableBody, force: np.ndarray, fixed_dofs: np.ndarray) -> np.ndarray:
    K = assemble_stiffness_matrix(body)
    Kff, rhs, free = eliminate_fixed_dofs(K, force, fixed_dofs)
    u = np.zeros(body.n_dofs, dtype=float)
    if free.size:
        u[free] = np.asarray(spsolve(Kff, rhs), dtype=float)
    return u


def uniaxial_linear_elastic_patch() -> list[CaseResult]:
    """Validate assembled strain energy against the analytic affine value."""

    mesh = structured_tet_block(1, 1, 1, size=(1.0, 1.0, 1.0))
    E, nu, eps = 2.0e5, 0.25, 1.0e-3
    body = _body(mesh, E=E, nu=nu)
    K = assemble_stiffness_matrix(body)

    u_nodes = np.zeros_like(mesh.X)
    u_nodes[:, 0] = eps * mesh.X[:, 0]
    u = u_nodes.ravel()
    numerical = 0.5 * float(u @ (K @ u))
    C = isotropic_linear_elasticity_matrix(E, nu)
    volume = float(np.prod(np.ptp(mesh.X, axis=0)))
    analytic = 0.5 * float(C[0, 0]) * eps * eps * volume
    rel_error = abs(numerical - analytic) / max(abs(analytic), 1.0e-30)
    status = "ok" if rel_error < 1.0e-12 else "fail"
    return [
        CaseResult(
            "uniaxial_linear_elastic_patch",
            "relative_energy_error",
            rel_error,
            status,
            f"numerical={numerical:.16e}; analytic={analytic:.16e}",
        )
    ]


def gravity_fixed_base_elastic_block(resolution: int = 1) -> list[CaseResult]:
    """Run a static gravity solve with the base fixed."""

    mesh = structured_tet_block(resolution, resolution, resolution, size=(1.0, 0.5, 0.5))
    body = _body(mesh, E=5.0e4, nu=0.3, rho=3.0)
    fixed_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], 0.0))
    fixed = fixed_dofs_from_node_set(fixed_nodes, "xyz")
    f = assemble_gravity_force(body, (0.0, 0.0, -9.81))
    u = _solve_static(body, f, fixed)
    free = np.setdiff1d(np.arange(body.n_dofs), fixed)
    residual = assemble_stiffness_matrix(body) @ u - f
    rel_residual = np.linalg.norm(residual[free]) / max(np.linalg.norm(f[free]), 1.0e-30)
    tip_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], mesh.X[:, 0].max()))
    tip_uz = float(np.mean(u.reshape((-1, 3))[tip_nodes, 2]))
    status = "ok" if np.isfinite(tip_uz) and tip_uz < 0.0 and rel_residual < 1.0e-10 else "fail"
    return [
        CaseResult(
            "gravity_fixed_base_elastic_block",
            "mean_tip_z_displacement",
            tip_uz,
            status,
            f"resolution={resolution}; relative_free_residual={rel_residual:.3e}",
        )
    ]


def undamped_free_vibration_energy(steps: int = 12) -> list[CaseResult]:
    """Check energy drift for an undamped linear system under Newmark average acceleration."""

    mesh = structured_tet_block(1, 1, 1, size=(1.0, 0.4, 0.4))
    body = _body(mesh, E=2.5e4, nu=0.25, rho=1.5)
    fixed_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], 0.0))
    fixed = fixed_dofs_from_node_set(fixed_nodes, "xyz")
    M = assemble_mass_matrix(body)
    K = assemble_stiffness_matrix(body)
    C = csr_matrix(K.shape, dtype=float)
    u = np.zeros(body.n_dofs, dtype=float)
    u.reshape((-1, 3))[:, 2] = 2.0e-3 * mesh.X[:, 0]
    v = np.zeros_like(u)
    a = np.zeros_like(u)
    f = np.zeros_like(u)
    Mff, rhs_a, free = eliminate_fixed_dofs(M, -(K @ u), fixed)
    if free.size:
        a[free] = np.asarray(spsolve(Mff, rhs_a), dtype=float)

    def energy(uu: np.ndarray, vv: np.ndarray) -> float:
        return 0.5 * float(vv @ (M @ vv)) + 0.5 * float(uu @ (K @ uu))

    e0 = energy(u, v)
    for _ in range(steps):
        u, v, a = newmark_beta_step(M, C, K, u, v, a, f, dt=1.0e-4, fixed_dofs=fixed)
    e1 = energy(u, v)
    drift = abs(e1 - e0) / max(abs(e0), 1.0e-30)
    return [
        CaseResult(
            "undamped_free_vibration_energy",
            "relative_energy_drift",
            drift,
            "ok" if drift < 1.0e-8 else "fail",
            f"steps={steps}; initial={e0:.16e}; final={e1:.16e}",
        )
    ]


def deformable_block_against_rigid_plane() -> list[CaseResult]:
    """Evaluate a deformable surface sample against a fixed triangular plane."""

    slave_x = np.array([[0.25, 0.25, -0.02]], dtype=float)
    plane_x = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    plane_faces = np.array([[0, 1, 2]], dtype=np.int64)
    sample = SurfaceSample(
        node_ids=np.array([0], dtype=np.int64),
        weights=np.array([1.0]),
        candidate_face_ids=np.array([0], dtype=np.int64),
    )
    constraint = contact_constraint_from_sample(slave_x, sample, plane_x, plane_faces)
    force, _ = penalty_contact_response([constraint], stiffness=1.0e4, n_total_dofs=12, master_dof_offset=3)
    expected_penalty = 200.0
    value = float(constraint.g)
    status = "ok" if value < 0.0 and np.isclose(force[2], expected_penalty) else "fail"
    return [
        CaseResult(
            "deformable_block_against_rigid_plane",
            "signed_gap",
            value,
            status,
            f"slave_force_z={force[2]:.6e}; expected_penalty={expected_penalty:.6e}",
        )
    ]


def deformable_deformable_contact_action_reaction() -> list[CaseResult]:
    """Check equal-and-opposite assembled penalty forces across two bodies."""

    master_x = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    master_faces = np.array([[0, 1, 2]], dtype=np.int64)
    slave_x = np.array([[0.2, 0.2, -0.01], [0.4, 0.2, -0.01]], dtype=float)
    samples = [
        SurfaceSample(np.array([0], dtype=np.int64), np.array([1.0]), np.array([0], dtype=np.int64)),
        SurfaceSample(np.array([1], dtype=np.int64), np.array([1.0]), np.array([0], dtype=np.int64)),
    ]
    constraints = [
        contact_constraint_from_sample(slave_x, sample, master_x, master_faces)
        for sample in samples
    ]
    n_slave = 3 * slave_x.shape[0]
    total_dofs = n_slave + 3 * master_x.shape[0]
    force, _ = penalty_contact_response(
        constraints,
        stiffness=5.0e3,
        n_total_dofs=total_dofs,
        slave_dof_offset=0,
        master_dof_offset=n_slave,
    )
    slave_total = force[:n_slave].reshape((-1, 3)).sum(axis=0)
    master_total = force[n_slave:].reshape((-1, 3)).sum(axis=0)
    imbalance = float(np.linalg.norm(slave_total + master_total))
    status = "ok" if imbalance < 1.0e-12 and np.linalg.norm(slave_total) > 0.0 else "fail"
    return [
        CaseResult(
            "deformable_deformable_contact_action_reaction",
            "net_force_imbalance",
            imbalance,
            status,
            f"slave_total={slave_total.tolist()}; master_total={master_total.tolist()}",
        )
    ]


def required_validation_case_records() -> list[ValidationRecord]:
    """Emit the required baseline validation case IDs from AGENTS.md."""

    records: list[ValidationRecord] = []
    Xe = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    K = tet4_stiffness(Xe, 10.0, 0.25)
    rigid_modes = []
    for component in range(3):
        mode = np.zeros((4, 3), dtype=float)
        mode[:, component] = 1.0
        rigid_modes.append(mode.ravel())
    center = Xe.mean(axis=0)
    for axis in np.eye(3):
        rigid_modes.append(np.cross(axis, Xe - center).ravel())
    rigid_residual = max(float(np.linalg.norm(K @ mode)) for mode in rigid_modes)
    min_eig = float(np.linalg.eigvalsh(0.5 * (K + K.T)).min())
    records.append(
        _record(
            "required_validation",
            "tet4_stiffness_rigid_modes",
            "max_rigid_mode_residual",
            rigid_residual,
            status="ok" if rigid_residual < 1.0e-12 and min_eig > -1.0e-12 else "fail",
            details=f"minimum_eigenvalue={min_eig:.3e}",
        )
    )

    rho = 7.0
    M = tet4_mass(Xe, rho)
    total_mass = rho * tet4_volume(Xe)
    component_masses = [float(np.ones(4) @ M[c::3, c::3] @ np.ones(4)) for c in range(3)]
    mass_error = max(abs(m - total_mass) for m in component_masses) / total_mass
    records.append(
        _record(
            "required_validation",
            "tet4_mass_conservation",
            "relative_component_mass_error",
            mass_error,
            status="ok" if mass_error < 1.0e-14 else "fail",
            details=f"component_masses={component_masses}; expected={total_mass:.16e}",
        )
    )

    mesh = structured_tet_block(1, 1, 1)
    faces, adjacent = extract_boundary_faces(mesh.elements, mesh.X)
    records.append(
        _record(
            "required_validation",
            "boundary_face_extraction",
            "boundary_face_count",
            int(faces.shape[0]),
            status="ok" if faces.shape == (12, 3) and adjacent.shape == (12,) else "fail",
            details="one cube split into five tetrahedra has twelve boundary triangles",
        )
    )

    point = np.array([0.25, 0.25, 0.4], dtype=float)
    a = np.array([0.0, 0.0, 0.0], dtype=float)
    b = np.array([1.0, 0.0, 0.0], dtype=float)
    c = np.array([0.0, 1.0, 0.0], dtype=float)
    closest, weights, dist2, _ = closest_point_on_triangle(point, a, b, c)
    projection_error = float(np.linalg.norm(closest - np.array([0.25, 0.25, 0.0])))
    records.append(
        _record(
            "required_validation",
            "closest_point_projection",
            "projection_error",
            projection_error,
            status="ok" if projection_error < 1.0e-14 and np.isclose(dist2, 0.16) else "fail",
            details=f"weights={weights.tolist()}",
        )
    )

    slave_x = np.array([[0.25, 0.25, -0.02], [0.25, 0.25, 0.02]], dtype=float)
    plane_x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    plane_faces = np.array([[0, 1, 2]], dtype=np.int64)
    below = SurfaceSample(np.array([0], dtype=np.int64), np.array([1.0]), np.array([0], dtype=np.int64))
    above = SurfaceSample(np.array([1], dtype=np.int64), np.array([1.0]), np.array([0], dtype=np.int64))
    g_below = contact_constraint_from_sample(slave_x, below, plane_x, plane_faces).g
    g_above = contact_constraint_from_sample(slave_x, above, plane_x, plane_faces).g
    records.append(
        _record(
            "required_validation",
            "contact_gap_sign_convention",
            "below_above_gap_product",
            float(g_below * g_above),
            status="ok" if g_below < 0.0 < g_above else "fail",
            details=f"below={g_below:.6e}; above={g_above:.6e}",
        )
    )

    fd_error = contact_jacobian_finite_difference_error()
    records.append(
        _record(
            "required_validation",
            "contact_jacobian_finite_difference",
            "absolute_directional_error",
            fd_error,
            status="ok" if fd_error < 1.0e-8 else "fail",
            details="deterministic slave/master directional perturbation",
        )
    )

    src_files = list((ROOT / "src" / "sfc").rglob("*.py"))
    offending = [
        str(path.relative_to(ROOT))
        for path in src_files
        if "abaqus" in path.read_text(encoding="utf-8").lower()
        or "legacy_abaqus_prototype" in path.read_text(encoding="utf-8").lower()
    ]
    records.append(
        _record(
            "required_validation",
            "no_core_abaqus_imports",
            "offending_file_count",
            len(offending),
            status="ok" if not offending else "fail",
            details=";".join(offending),
        )
    )
    return records


def contact_jacobian_finite_difference_error() -> float:
    """Return a directional finite-difference check for one contact row."""

    slave_x = np.array([[0.25, 0.25, -0.02]], dtype=float)
    master_x = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    master_faces = np.array([[0, 1, 2]], dtype=np.int64)
    sample = SurfaceSample(np.array([0], dtype=np.int64), np.array([1.0]), np.array([0], dtype=np.int64))
    constraint = contact_constraint_from_sample(slave_x, sample, master_x, master_faces)
    J = assemble_contact_jacobian([constraint], n_total_dofs=12, slave_dof_offset=0, master_dof_offset=3)
    direction = np.array(
        [0.0, 0.0, 0.7, 0.1, 0.2, -0.3, -0.2, 0.1, -0.3, 0.0, -0.1, -0.3],
        dtype=float,
    )
    eps = 1.0e-7

    def gap(q: np.ndarray) -> float:
        sx = slave_x + q[:3].reshape((1, 3))
        mx = master_x + q[3:].reshape((3, 3))
        return float(contact_constraint_from_sample(sx, sample, mx, master_faces).g)

    fd = (gap(eps * direction) - gap(-eps * direction)) / (2.0 * eps)
    analytic = float((J @ direction)[0])
    return abs(fd - analytic)


def run_case_results(quick: bool) -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    records.extend(required_validation_case_records())
    case_groups = [
        uniaxial_linear_elastic_patch(),
        gravity_fixed_base_elastic_block(1),
        undamped_free_vibration_energy(6 if quick else 20),
        deformable_block_against_rigid_plane(),
        deformable_deformable_contact_action_reaction(),
    ]
    for result in (item for group in case_groups for item in group):
        records.append(
            _record(
                "validation",
                result.case,
                result.metric,
                result.value,
                status=result.status,
                details=result.details,
            )
        )
    return records


def convergence_utility(resolutions: Iterable[int]) -> list[ValidationRecord]:
    """Run a mesh-refinement utility or mark convergence unavailable."""

    unique_resolutions = sorted({int(r) for r in resolutions if int(r) > 0})
    if len(unique_resolutions) < 2:
        only = unique_resolutions[0] if unique_resolutions else ""
        return [
            _record(
                "convergence",
                "gravity_fixed_base_elastic_block",
                "convergence_rate",
                "",
                status="unavailable",
                details=f"only one mesh resolution available ({only})",
            )
        ]

    displacements: list[float] = []
    for resolution in unique_resolutions:
        result = gravity_fixed_base_elastic_block(resolution)[0]
        displacements.append(float(result.value))

    successive = [
        abs(displacements[i] - displacements[i - 1])
        for i in range(1, len(displacements))
    ]
    if len(successive) < 2 or min(successive) <= 0.0:
        rate: float | str = ""
        status = "unavailable"
        details = "need at least two nonzero successive differences for a rate"
    else:
        h_ratio = unique_resolutions[-1] / unique_resolutions[-2]
        rate = float(np.log(successive[-2] / successive[-1]) / np.log(h_ratio))
        status = "ok"
        details = f"resolutions={unique_resolutions}; tip_uz={displacements}"

    records = [
        _record(
            "convergence",
            "gravity_fixed_base_elastic_block",
            f"tip_uz_n{resolution}",
            displacement,
            status="ok",
            details="static gravity solve",
        )
        for resolution, displacement in zip(unique_resolutions, displacements)
    ]
    records.append(
        _record(
            "convergence",
            "gravity_fixed_base_elastic_block",
            "convergence_rate",
            rate,
            status=status,
            details=details,
        )
    )
    return records


def contact_performance_comparison(iterations: int) -> list[ValidationRecord]:
    """Include isolated contact timing stages using existing benchmark helpers."""

    from benchmarks.run_contact_benchmark import (  # noqa: WPS433
        build_contact_scene,
        comparison_method_specs,
        phase_benchmark_specs,
        time_stage,
    )

    records: list[ValidationRecord] = []
    scene = build_contact_scene()
    phase_specs = phase_benchmark_specs(scene)
    comparison_specs = comparison_method_specs(scene)

    for spec in phase_specs:
        timed, _ = time_stage("phase3_validation", spec, iterations=iterations)
        records.append(
            _record(
                "performance",
                timed["method"],
                timed["stage"],
                timed["mean_seconds"],
                status=timed["status"],
                details=f"workload={timed['workload']}; {timed['details']}",
            )
        )

    comparison_means: dict[str, float] = {}
    comparison_workloads: list[str] = []
    for spec in comparison_specs:
        timed, _ = time_stage("phase3_validation", spec, iterations=iterations)
        comparison_means[timed["method"]] = float(timed["mean_seconds"])
        comparison_workloads.append(timed["workload"])
        records.append(
            _record(
                "performance",
                timed["method"],
                timed["stage"],
                timed["mean_seconds"],
                status=timed["status"],
                details=timed["details"],
                label=timed["method"],
                workload=timed["workload"],
            )
        )

    brute = comparison_means.get("brute_force_all_triangle_projection")
    dynamic = comparison_means.get("dynamic_fem_induced_sdf_full_pipeline")
    if brute is None or dynamic is None or dynamic <= 0.0:
        speedup: float | str = ""
        status = "unavailable"
        details = "missing brute-force or dynamic full-pipeline timing"
    else:
        speedup = brute / dynamic
        status = "ok"
        details = "brute_force_all_triangle_projection / dynamic_fem_induced_sdf_full_pipeline"

    records.append(
        _record(
            "performance",
            "contact_pipeline_speedup",
            "brute_force_over_dynamic_full_pipeline",
            speedup,
            status=status,
            details=details,
        )
    )

    records.append(
        _record(
            "performance",
            "contact_method_workload_distinctness",
            "unique_workloads",
            len(set(comparison_workloads)),
            status="ok" if len(set(comparison_workloads)) == len(comparison_workloads) else "fail",
            details=f"compared_methods={len(comparison_workloads)}",
        )
    )
    return records


def write_csv(records: list[ValidationRecord], path: Path) -> None:
    """Write validation records as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "case", "label", "metric", "value", "workload", "status", "details"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _format_value(value: float | str) -> str:
    if value == "":
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.6e}"


def write_markdown(records: list[ValidationRecord], path: Path) -> None:
    """Write a compact Markdown validation summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase-3 Validation Summary",
        "",
        "Reproduce:",
        "",
        "```bash",
        "python validation/run_phase3_validation.py --quick --out-dir results/validation",
        "```",
        "",
        "| Category | Case | Metric | Value | Status | Details |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| {category} | {case} | {metric} | {value} | {status} | {details} |".format(
                category=record["category"],
                case=record["case"],
                metric=record["metric"],
                value=_format_value(record["value"]),
                status=record["status"],
                details=str(record["details"]).replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use CI-friendly work sizes.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "validation")
    return parser.parse_args()


def run_validation(*, quick: bool) -> list[ValidationRecord]:
    start = perf_counter()
    records = run_case_results(quick)
    records.extend(convergence_utility([1] if quick else [1, 2, 3]))
    records.extend(contact_performance_comparison(1 if quick else 5))
    records.append(
        _record(
            "summary",
            "phase3_validation_runner",
            "total_seconds",
            perf_counter() - start,
            status="ok",
            details=f"quick={quick}",
        )
    )
    return records


def main() -> int:
    args = parse_args()
    records = run_validation(quick=bool(args.quick))
    csv_path = args.out_dir / "phase3_validation.csv"
    md_path = args.out_dir / "phase3_validation.md"
    write_csv(records, csv_path)
    write_markdown(records, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
