"""Phase-4 validation evidence for the minimal standalone FEM-SDF solver.

This script generates deterministic validation and benchmark artifacts. It does
not add physics features; it exercises the current linear TET4, oriented local
SDF, spatial-hash, and penalty-contact implementation.
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact import (  # noqa: E402
    SurfaceSample,
    UniformTriangleAABBHash,
    contact_constraint_from_sample,
    penalty_contact_response,
)
from sfc.fem import (  # noqa: E402
    assemble_gravity_force,
    assemble_stiffness_matrix,
    fixed_dofs_from_node_set,
)
from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.sdf import dynamic_surface_sdf  # noqa: E402
from sfc.sdf.local_projection import closest_point_on_triangle  # noqa: E402
from validation.run_phase3_validation import (  # noqa: E402
    _body,
    _solve_static,
    structured_tet_block,
)

Row = dict[str, Any]


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _time_call(func: Callable[[], Any], iterations: int) -> tuple[float, Any]:
    start = perf_counter()
    result = None
    for _ in range(iterations):
        result = func()
    total = perf_counter() - start
    return total / max(iterations, 1), result


def _format_float(value: float | str) -> str:
    if value == "":
        return ""
    return f"{float(value):.6e}"


def _uniaxial_energy_row(resolution: int) -> Row:
    mesh = structured_tet_block(resolution, resolution, resolution)
    E, nu, eps = 2.0e5, 0.25, 1.0e-3
    body = _body(mesh, E=E, nu=nu)
    K = assemble_stiffness_matrix(body)
    u_nodes = np.zeros_like(mesh.X)
    u_nodes[:, 0] = eps * mesh.X[:, 0]
    u = u_nodes.ravel()
    numerical = 0.5 * float(u @ (K @ u))
    analytic = 0.5 * float(isotropic_linear_elasticity_matrix(E, nu)[0, 0]) * eps * eps
    rel_error = abs(numerical - analytic) / max(abs(analytic), 1.0e-30)
    return {
        "case": "uniaxial_linear_elastic_patch",
        "resolution": resolution,
        "nodes": mesh.X.shape[0],
        "elements": mesh.elements.shape[0],
        "metric": "relative_energy_error",
        "value": rel_error,
        "relative_to_finest": "",
        "trend": "affine energy exact to roundoff",
        "claim_status": "evidence",
        "details": f"numerical={numerical:.16e}; analytic={analytic:.16e}",
    }


def _gravity_tip_row(resolution: int) -> Row:
    mesh = structured_tet_block(resolution, resolution, resolution, size=(1.0, 0.5, 0.5))
    body = _body(mesh, E=5.0e4, nu=0.3, rho=3.0)
    fixed_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], 0.0))
    fixed = fixed_dofs_from_node_set(fixed_nodes, "xyz")
    u = _solve_static(body, assemble_gravity_force(body, (0.0, 0.0, -9.81)), fixed)
    tip_nodes = np.flatnonzero(np.isclose(mesh.X[:, 0], mesh.X[:, 0].max()))
    tip_uz = float(np.mean(u.reshape((-1, 3))[tip_nodes, 2]))
    return {
        "case": "fixed_base_gravity_elastic_block",
        "resolution": resolution,
        "nodes": mesh.X.shape[0],
        "elements": mesh.elements.shape[0],
        "metric": "mean_tip_z_displacement",
        "value": tip_uz,
        "relative_to_finest": "",
        "trend": "reported against finest available quick/full run",
        "claim_status": "evidence",
        "details": "static linear solve with fixed x=0 base",
    }


def mesh_resolution_study(resolutions: Iterable[int]) -> list[Row]:
    unique = sorted({int(r) for r in resolutions if int(r) > 0})
    rows = [_uniaxial_energy_row(r) for r in unique]
    gravity_rows = [_gravity_tip_row(r) for r in unique]
    finest_tip = float(gravity_rows[-1]["value"])
    for row in gravity_rows:
        row["relative_to_finest"] = abs(float(row["value"]) - finest_tip) / max(abs(finest_tip), 1.0e-30)
    rows.extend(gravity_rows)

    resolution_count = len(unique)
    claim_status = "supported" if resolution_count >= 3 else "not_supported"
    rows.append(
        {
            "case": "mesh_resolution_study",
            "resolution": "all",
            "nodes": "",
            "elements": "",
            "metric": "resolution_count",
            "value": resolution_count,
            "relative_to_finest": "",
            "trend": "three or more resolutions are required before reporting trends",
            "claim_status": claim_status,
            "details": f"resolutions={unique}; no theoretical order is claimed",
        }
    )
    return rows


def _plane() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        np.array([[0, 1, 2]], dtype=np.int64),
    )


def _contact_step(
    case: str,
    time: float,
    slave_x: np.ndarray,
    master_x: np.ndarray,
    master_faces: np.ndarray,
    stiffness: float,
) -> Row:
    samples = [
        SurfaceSample(np.array([i], dtype=np.int64), np.array([1.0]), np.array([0], dtype=np.int64))
        for i in range(slave_x.shape[0])
    ]
    constraints = [
        contact_constraint_from_sample(slave_x, sample, master_x, master_faces)
        for sample in samples
    ]
    slave_dofs = 3 * slave_x.shape[0]
    total_dofs = slave_dofs + 3 * master_x.shape[0]
    force, _ = penalty_contact_response(
        constraints,
        stiffness=stiffness,
        n_total_dofs=total_dofs,
        slave_dof_offset=0,
        master_dof_offset=slave_dofs,
    )
    gaps = np.array([float(c.g) for c in constraints])
    penetrations = np.maximum(-gaps, 0.0)
    slave_total = force[:slave_dofs].reshape((-1, 3)).sum(axis=0)
    master_total = force[slave_dofs:].reshape((-1, 3)).sum(axis=0)
    return {
        "case": case,
        "time": time,
        "min_gap": float(gaps.min()),
        "max_penetration": float(penetrations.max()),
        "active_contact_count": int(np.count_nonzero(penetrations > 0.0)),
        "normal_force": float(slave_total[2]),
        "contact_energy": 0.5 * stiffness * float(np.dot(penetrations, penetrations)),
        "action_reaction_imbalance": float(np.linalg.norm(slave_total + master_total)),
        "status": "ok",
    }


def contact_time_history(steps: int) -> list[Row]:
    master_x, master_faces = _plane()
    rows: list[Row] = []
    base_xy = np.array([[0.2, 0.2], [0.45, 0.2], [0.2, 0.45], [0.45, 0.45]], dtype=float)
    for step in range(steps):
        time = step / max(steps - 1, 1)
        z = 0.025 - 0.05 * time
        slave = np.column_stack((base_xy, np.full(base_xy.shape[0], z)))
        rows.append(
            {
                "step": step,
                **_contact_step(
                    "elastic_block_pressed_against_rigid_plane",
                    time,
                    slave,
                    master_x,
                    master_faces,
                    8.0e3,
                ),
            }
        )

    deformable_master = master_x + np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, -0.05, 0.0]])
    for step in range(steps):
        time = step / max(steps - 1, 1)
        z = 0.02 - 0.04 * time
        slave = np.array([[0.25, 0.25, z], [0.55, 0.20, z]], dtype=float)
        rows.append(
            {
                "step": step,
                **_contact_step(
                    "deformable_deformable_contact_time_history",
                    time,
                    slave,
                    deformable_master,
                    master_faces,
                    6.0e3,
                ),
            }
        )
    return rows


def _plane_surface(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = []
    for j in range(resolution + 1):
        for i in range(resolution + 1):
            nodes.append((i / resolution, j / resolution, 0.0))
    faces = []
    for j in range(resolution):
        for i in range(resolution):
            v00 = j * (resolution + 1) + i
            v10 = v00 + 1
            v01 = v00 + (resolution + 1)
            v11 = v01 + 1
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    points = [
        ((i + 0.5) / resolution, (j + 0.5) / resolution, -0.005)
        for j in range(resolution)
        for i in range(resolution)
    ]
    return np.asarray(nodes, dtype=float), np.asarray(faces, dtype=np.int64), np.asarray(points, dtype=float)


def _brute_force_surface_query(points: np.ndarray, nodes: np.ndarray, faces: np.ndarray) -> list[Any]:
    all_faces = np.arange(faces.shape[0], dtype=np.int64)
    return [dynamic_surface_sdf(point, nodes, faces, all_faces) for point in points]


def _candidate_query(points: np.ndarray, nodes: np.ndarray, faces: np.ndarray) -> tuple[UniformTriangleAABBHash, list[np.ndarray]]:
    index = UniformTriangleAABBHash.from_surface(nodes, faces, delta_safe=0.02, cell_size=1.0 / max(1, int(np.sqrt(len(points)))))
    return index, index.query_points(points)


def _local_projection_query(points: np.ndarray, nodes: np.ndarray, faces: np.ndarray, candidates: list[np.ndarray]) -> list[Any]:
    out = []
    for point, face_ids in zip(points, candidates):
        best = None
        best_dist2 = np.inf
        for face_id in face_ids:
            tri = nodes[faces[int(face_id)]]
            result = closest_point_on_triangle(point, tri[0], tri[1], tri[2])
            if result[2] < best_dist2:
                best = result
                best_dist2 = result[2]
        if best is not None:
            out.append(best)
    return out


def _dynamic_surface_query(points: np.ndarray, nodes: np.ndarray, faces: np.ndarray, candidates: list[np.ndarray]) -> list[Any]:
    return [
        dynamic_surface_sdf(point, nodes, faces, face_ids)
        for point, face_ids in zip(points, candidates)
        if face_ids.size
    ]


def _full_dynamic_contact_pipeline(points: np.ndarray, nodes: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, Any]:
    _, candidates = _candidate_query(points, nodes, faces)
    constraints = []
    for i, face_ids in enumerate(candidates):
        if face_ids.size:
            sample = SurfaceSample(np.array([i], dtype=np.int64), np.array([1.0]), face_ids)
            constraints.append(contact_constraint_from_sample(points, sample, nodes, faces))
    return penalty_contact_response(
        constraints,
        stiffness=5.0e3,
        n_total_dofs=3 * (points.shape[0] + nodes.shape[0]),
        slave_dof_offset=0,
        master_dof_offset=3 * points.shape[0],
    )


def performance_scaling(surface_sizes: Iterable[int], iterations: int) -> list[Row]:
    rows: list[Row] = []
    for resolution in surface_sizes:
        nodes, faces, points = _plane_surface(int(resolution))
        index, candidates = _candidate_query(points, nodes, faces)
        timings: dict[str, float] = {}

        stages: list[tuple[str, str, str, Callable[[], Any]]] = [
            (
                "brute_force_all_triangle_projection",
                "contact_detection_end_to_end",
                "brute_force_all_triangle_projection_workload",
                lambda n=nodes, f=faces, p=points: _brute_force_surface_query(p, n, f),
            ),
            (
                "spatial_hash_plus_local_projection",
                "build_broad_phase",
                "build_spatial_hash_workload",
                lambda n=nodes, f=faces: UniformTriangleAABBHash.from_surface(n, f, delta_safe=0.02, cell_size=1.0 / int(resolution)),
            ),
            (
                "spatial_hash_plus_local_projection",
                "candidate_query_only",
                "candidate_query_workload",
                lambda idx=index, p=points: idx.query_points(p),
            ),
            (
                "spatial_hash_plus_local_projection",
                "local_projection_only",
                "local_projection_workload",
                lambda n=nodes, f=faces, p=points, c=candidates: _local_projection_query(p, n, f, c),
            ),
            (
                "spatial_hash_plus_local_projection",
                "contact_detection_end_to_end",
                "spatial_hash_plus_local_projection_full_workload",
                lambda n=nodes, f=faces, p=points: _local_projection_query(p, n, f, _candidate_query(p, n, f)[1]),
            ),
            (
                "dynamic_fem_induced_sdf_full_pipeline",
                "dynamic_sdf_query_only",
                "dynamic_sdf_query_workload",
                lambda n=nodes, f=faces, p=points, c=candidates: _dynamic_surface_query(p, n, f, c),
            ),
            (
                "dynamic_fem_induced_sdf_full_pipeline",
                "full_contact_pipeline_end_to_end",
                "dynamic_fem_sdf_full_contact_workload",
                lambda n=nodes, f=faces, p=points: _full_dynamic_contact_pipeline(p, n, f),
            ),
        ]
        for method, stage, workload, func in stages:
            mean, _ = _time_call(func, iterations)
            timings[f"{method}:{stage}"] = mean
            rows.append(
                {
                    "surface_resolution": resolution,
                    "triangle_count": faces.shape[0],
                    "query_point_count": points.shape[0],
                    "method": method,
                    "stage": stage,
                    "workload": workload,
                    "iterations": iterations,
                    "mean_seconds": mean,
                    "speedup_vs_bruteforce": "",
                    "speedup_status": "not_applicable",
                    "status": "ok",
                    "details": "isolated timing row",
                }
            )

        brute = timings["brute_force_all_triangle_projection:contact_detection_end_to_end"]
        for row in rows:
            if row["surface_resolution"] != resolution:
                continue
            if row["stage"] not in {"contact_detection_end_to_end", "full_contact_pipeline_end_to_end"}:
                continue
            if row["method"] == "brute_force_all_triangle_projection":
                continue
            speedup = brute / float(row["mean_seconds"]) if float(row["mean_seconds"]) > 0.0 else np.inf
            if speedup > 1.0:
                row["speedup_vs_bruteforce"] = speedup
                row["speedup_status"] = "speedup_supported"
            else:
                row["speedup_vs_bruteforce"] = ""
                row["speedup_status"] = "no_speedup"
                row["details"] = "speedup <= 1; acceleration claim not supported for this row"

        rows.append(
            {
                "surface_resolution": resolution,
                "triangle_count": faces.shape[0],
                "query_point_count": points.shape[0],
                "method": "material_space_sdf_baseline",
                "stage": "contact_detection_end_to_end",
                "workload": "unavailable",
                "iterations": 0,
                "mean_seconds": "",
                "speedup_vs_bruteforce": "",
                "speedup_status": "unavailable",
                "status": "unavailable",
                "details": "material-space SDF baseline is not implemented",
            }
        )
    return rows


def claim_rows(mesh_rows: list[Row], performance_rows: list[Row]) -> list[Row]:
    resolution_count = len({int(row["resolution"]) for row in mesh_rows if str(row["resolution"]).isdigit()})
    speedup_values = [
        float(row["speedup_vs_bruteforce"])
        for row in performance_rows
        if row["speedup_vs_bruteforce"] != ""
    ]
    max_speedup = max(speedup_values) if speedup_values else 0.0
    return [
        {
            "claim_id": "mesh_resolution_trends_reported",
            "claim_text": "Mesh-resolution displacement and energy-error trends are reported without a theoretical order claim.",
            "evidence_csv": "phase4_mesh_resolution.csv",
            "evidence_field": "relative_to_finest",
            "condition_field": "resolution_count",
            "condition_value": resolution_count,
            "claim_status": "supported" if resolution_count >= 3 else "not_supported",
            "details": "requires at least three mesh resolutions",
        },
        {
            "claim_id": "contact_time_history_reported",
            "claim_text": "Contact time histories report gap, penetration, active contacts, normal force, energy, and action-reaction imbalance.",
            "evidence_csv": "phase4_contact_time_history.csv",
            "evidence_field": "action_reaction_imbalance",
            "condition_field": "time_history_case_count",
            "condition_value": 2,
            "claim_status": "supported",
            "details": "rigid-plane and deformable-deformable histories are present",
        },
        {
            "claim_id": "contact_acceleration_rows_conditional",
            "claim_text": "Acceleration is claimed only for rows where measured speedup is greater than one.",
            "evidence_csv": "phase4_performance_scaling.csv",
            "evidence_field": "speedup_vs_bruteforce",
            "condition_field": "max_speedup",
            "condition_value": max_speedup,
            "claim_status": "supported" if max_speedup > 1.0 else "not_supported",
            "details": "non-speedup rows leave speedup blank and are marked no_speedup",
        },
    ]


def write_markdown(
    path: Path,
    mesh_rows: list[Row],
    history_rows: list[Row],
    performance_rows: list[Row],
    claims: list[Row],
) -> None:
    lines = [
        "# Phase-4 Paper-Scale Validation Summary",
        "",
        "This is evidence for the current minimal solver only. It does not validate friction, self-contact, nonlinear FEM, GPU execution, or barrier contact.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python validation/run_phase4_paper_validation.py --quick --out-dir results/phase4",
        "```",
        "",
        "## Evidence Claims",
        "",
        "| Claim | Status | Evidence | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for claim in claims:
        lines.append(
            f"| {claim['claim_id']} | {claim['claim_status']} | {claim['evidence_csv']}::{claim['evidence_field']} | {claim['details']} |"
        )
        lines.append(f"<!-- evidence csv={claim['evidence_csv']} field={claim['evidence_field']} -->")

    lines.extend(
        [
            "",
            "## Mesh Resolution",
            "",
            "| Case | Resolution | Metric | Value | Relative To Finest | Claim Status |",
            "| --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in mesh_rows:
        lines.append(
            "| {case} | {resolution} | {metric} | {value} | {relative} | {claim_status} |".format(
                case=row["case"],
                resolution=row["resolution"],
                metric=row["metric"],
                value=_format_float(row["value"]),
                relative=_format_float(row["relative_to_finest"]),
                claim_status=row["claim_status"],
            )
        )

    lines.extend(
        [
            "",
            "## Contact Histories",
            "",
            "| Case | Steps | Min Gap | Max Penetration | Max Imbalance |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in sorted({row["case"] for row in history_rows}):
        rows = [row for row in history_rows if row["case"] == case]
        lines.append(
            "| {case} | {steps} | {min_gap} | {max_penetration} | {imbalance} |".format(
                case=case,
                steps=len(rows),
                min_gap=_format_float(min(float(row["min_gap"]) for row in rows)),
                max_penetration=_format_float(max(float(row["max_penetration"]) for row in rows)),
                imbalance=_format_float(max(float(row["action_reaction_imbalance"]) for row in rows)),
            )
        )

    lines.extend(
        [
            "",
            "## Performance Scaling",
            "",
            "| Surface Resolution | Method | Stage | Mean Seconds | Speedup Status | Speedup |",
            "| ---: | --- | --- | ---: | --- | ---: |",
        ]
    )
    for row in performance_rows:
        if row["stage"] not in {"contact_detection_end_to_end", "full_contact_pipeline_end_to_end"}:
            continue
        lines.append(
            "| {resolution} | {method} | {stage} | {mean} | {speedup_status} | {speedup} |".format(
                resolution=row["surface_resolution"],
                method=row["method"],
                stage=row["stage"],
                mean=_format_float(row["mean_seconds"]),
                speedup_status=row["speedup_status"],
                speedup=_format_float(row["speedup_vs_bruteforce"]),
            )
        )
    lines.extend(
        [
            "",
            "Rows with `no_speedup` or blank speedup do not support acceleration claims.",
            "",
            "## Limitations",
            "",
            "- Linear TET4 only.",
            "- Penalty contact only.",
            "- No friction, self-contact, nonlinear FEM, GPU, or barrier contact.",
            "- The current SDF remains an oriented local surface distance.",
            "- Quick mode is reduced and deterministic; larger repeated runs are still required before paper submission.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use reduced deterministic sizes for CI.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "phase4")
    return parser.parse_args()


def run_phase4(*, quick: bool) -> dict[str, list[Row]]:
    resolutions = [1, 2, 3] if quick else [1, 2, 3, 4]
    surface_sizes = [2, 4, 6] if quick else [4, 8, 12]
    steps = 6 if quick else 20
    iterations = 1 if quick else 3
    mesh_rows = mesh_resolution_study(resolutions)
    history_rows = contact_time_history(steps)
    performance_rows = performance_scaling(surface_sizes, iterations)
    claims = claim_rows(mesh_rows, performance_rows)
    return {
        "mesh": mesh_rows,
        "history": history_rows,
        "performance": performance_rows,
        "claims": claims,
    }


def main() -> int:
    args = parse_args()
    outputs = run_phase4(quick=bool(args.quick))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.out_dir / "phase4_mesh_resolution.csv",
        ["case", "resolution", "nodes", "elements", "metric", "value", "relative_to_finest", "trend", "claim_status", "details"],
        outputs["mesh"],
    )
    _write_csv(
        args.out_dir / "phase4_contact_time_history.csv",
        ["case", "step", "time", "min_gap", "max_penetration", "active_contact_count", "normal_force", "contact_energy", "action_reaction_imbalance", "status"],
        outputs["history"],
    )
    _write_csv(
        args.out_dir / "phase4_performance_scaling.csv",
        ["surface_resolution", "triangle_count", "query_point_count", "method", "stage", "workload", "iterations", "mean_seconds", "speedup_vs_bruteforce", "speedup_status", "status", "details"],
        outputs["performance"],
    )
    _write_csv(
        args.out_dir / "phase4_claims.csv",
        ["claim_id", "claim_text", "evidence_csv", "evidence_field", "condition_field", "condition_value", "claim_status", "details"],
        outputs["claims"],
    )
    write_markdown(
        args.out_dir / "phase4_summary.md",
        outputs["mesh"],
        outputs["history"],
        outputs["performance"],
        outputs["claims"],
    )
    print(f"Wrote {args.out_dir / 'phase4_mesh_resolution.csv'}")
    print(f"Wrote {args.out_dir / 'phase4_contact_time_history.csv'}")
    print(f"Wrote {args.out_dir / 'phase4_performance_scaling.csv'}")
    print(f"Wrote {args.out_dir / 'phase4_claims.csv'}")
    print(f"Wrote {args.out_dir / 'phase4_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
