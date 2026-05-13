"""Deterministic contact benchmark runner.

The default problem is intentionally small so it can be used as a CI smoke
benchmark. Timing stages are isolated: setup for a stage is precomputed outside
the timed closure unless the stage name explicitly says end-to-end.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from dataclasses import dataclass
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
)
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.sdf import dynamic_surface_sdf  # noqa: E402
from sfc.sdf.local_projection import closest_point_on_triangle  # noqa: E402

BenchmarkRecord = dict[str, Any]


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    method: str
    stage: str
    workload: str
    func: Callable[[], Any]
    details: str


def build_two_tet_body() -> DeformableBody:
    mesh = VolumeMesh(
        X=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        ),
        elements=np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int64),
    )
    return DeformableBody(mesh=mesh, material={"E": 1.0e4, "nu": 0.3}, density=2.0)


def build_contact_scene() -> dict[str, Any]:
    master_nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    master_faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    slave_points = np.array(
        [
            [0.20, 0.20, -0.020],
            [0.55, 0.15, 0.030],
            [0.15, 0.55, -0.040],
            [0.33, 0.33, 0.015],
        ],
        dtype=float,
    )
    return {
        "master_nodes": master_nodes,
        "master_faces": master_faces,
        "slave_points": slave_points,
        "delta_safe": 0.10,
        "cell_size": 0.25,
    }


def time_stage(
    benchmark: str,
    spec: WorkloadSpec,
    *,
    iterations: int,
    status: str = "ok",
) -> tuple[BenchmarkRecord, Any]:
    start = perf_counter()
    result = None
    for _ in range(iterations):
        result = spec.func()
    elapsed = perf_counter() - start
    return (
        {
            "benchmark": benchmark,
            "method": spec.method,
            "stage": spec.stage,
            "workload": spec.workload,
            "iterations": iterations,
            "total_seconds": elapsed,
            "mean_seconds": elapsed / max(iterations, 1),
            "status": status,
            "details": spec.details,
        },
        result,
    )


def fem_assembly_workload() -> tuple[Any, Any, np.ndarray]:
    body = build_two_tet_body()
    K = assemble_stiffness_matrix(body)
    M = assemble_mass_matrix(body)
    f = assemble_gravity_force(body)
    return K, M, f


def build_spatial_hash(scene: dict[str, Any]) -> UniformTriangleAABBHash:
    return UniformTriangleAABBHash.from_surface(
        scene["master_nodes"],
        scene["master_faces"],
        delta_safe=scene["delta_safe"],
        cell_size=scene["cell_size"],
    )


def query_spatial_hash_candidates(
    scene: dict[str, Any],
    index: UniformTriangleAABBHash,
) -> list[np.ndarray]:
    return index.query_points(scene["slave_points"])


def brute_force_all_triangle_projection_workload(scene: dict[str, Any]) -> list[Any]:
    all_face_ids = np.arange(scene["master_faces"].shape[0], dtype=np.int64)
    return [
        dynamic_surface_sdf(point, scene["master_nodes"], scene["master_faces"], all_face_ids)
        for point in scene["slave_points"]
    ]


def spatial_hash_candidate_only_workload(
    scene: dict[str, Any],
    index: UniformTriangleAABBHash,
) -> list[np.ndarray]:
    return query_spatial_hash_candidates(scene, index)


def spatial_hash_plus_local_projection_workload(
    scene: dict[str, Any],
    candidates: list[np.ndarray],
) -> list[Any]:
    results = []
    for point, face_ids in zip(scene["slave_points"], candidates):
        best = None
        best_dist2 = np.inf
        for face_id in face_ids:
            tri = scene["master_nodes"][scene["master_faces"][int(face_id)]]
            result = closest_point_on_triangle(point, tri[0], tri[1], tri[2])
            if result[2] < best_dist2:
                best = result
                best_dist2 = result[2]
        if best is not None:
            results.append(best)
    return results


def dynamic_sdf_query_only_workload(
    scene: dict[str, Any],
    candidates: list[np.ndarray],
) -> list[Any]:
    results = []
    for point, face_ids in zip(scene["slave_points"], candidates):
        if face_ids.size:
            results.append(
                dynamic_surface_sdf(
                    point,
                    scene["master_nodes"],
                    scene["master_faces"],
                    face_ids,
                )
            )
    return results


def dynamic_fem_induced_sdf_full_pipeline_workload(scene: dict[str, Any]) -> list[Any]:
    index = build_spatial_hash(scene)
    candidates = query_spatial_hash_candidates(scene, index)
    return dynamic_sdf_query_only_workload(scene, candidates)


def contact_samples_from_candidates(candidates: list[np.ndarray]) -> list[SurfaceSample]:
    samples = []
    for node_id, face_ids in enumerate(candidates):
        if face_ids.size == 0:
            continue
        samples.append(
            SurfaceSample(
                node_ids=np.array([node_id], dtype=np.int64),
                weights=np.array([1.0]),
                candidate_face_ids=face_ids,
            )
        )
    return samples


def contact_constraints_from_samples(
    scene: dict[str, Any],
    samples: list[SurfaceSample],
) -> list[Any]:
    return [
        contact_constraint_from_sample(
            scene["slave_points"],
            sample,
            scene["master_nodes"],
            scene["master_faces"],
        )
        for sample in samples
    ]


def contact_jacobian_assembly_workload(
    constraints: list[Any],
    n_total_dofs: int,
    master_dof_offset: int,
) -> Any:
    return assemble_contact_jacobian(
        constraints,
        n_total_dofs=n_total_dofs,
        slave_dof_offset=0,
        master_dof_offset=master_dof_offset,
    )


def penalty_force_assembly_only_workload(
    constraints: list[Any],
    jacobian,
    *,
    stiffness: float,
) -> np.ndarray:
    lambdas = np.array([stiffness * max(-float(c.g), 0.0) for c in constraints])
    return np.asarray(jacobian.T @ lambdas).ravel()


def full_contact_pipeline_workload(scene: dict[str, Any]) -> tuple[np.ndarray, Any]:
    index = build_spatial_hash(scene)
    candidates = query_spatial_hash_candidates(scene, index)
    samples = contact_samples_from_candidates(candidates)
    constraints = contact_constraints_from_samples(scene, samples)
    n_total_dofs = 3 * (scene["slave_points"].shape[0] + scene["master_nodes"].shape[0])
    return penalty_contact_response(
        constraints,
        stiffness=1.0e4,
        n_total_dofs=n_total_dofs,
        slave_dof_offset=0,
        master_dof_offset=3 * scene["slave_points"].shape[0],
    )


def build_linear_system() -> tuple[Any, np.ndarray]:
    K, M, f = fem_assembly_workload()
    A = M + 1.0e-4 * K + 1.0e-12 * eye(M.shape[0], format="csr")
    return A, f


def linear_solve_only_workload(A, f: np.ndarray) -> np.ndarray:
    return np.asarray(spsolve(A, f), dtype=float)


def phase_benchmark_specs(scene: dict[str, Any]) -> list[WorkloadSpec]:
    index = build_spatial_hash(scene)
    candidates = query_spatial_hash_candidates(scene, index)
    samples = contact_samples_from_candidates(candidates)
    constraints = contact_constraints_from_samples(scene, samples)
    n_total_dofs = 3 * (scene["slave_points"].shape[0] + scene["master_nodes"].shape[0])
    master_dof_offset = 3 * scene["slave_points"].shape[0]
    jacobian = contact_jacobian_assembly_workload(
        constraints,
        n_total_dofs,
        master_dof_offset,
    )
    A, rhs = build_linear_system()

    return [
        WorkloadSpec(
            "native_fem",
            "fem_assembly_only",
            "fem_assembly_workload",
            fem_assembly_workload,
            "assemble K, M, and gravity only",
        ),
        WorkloadSpec(
            "aabb_spatial_hash",
            "build_update_broad_phase",
            "build_spatial_hash",
            lambda: build_spatial_hash(scene),
            "build padded triangle AABB spatial hash only",
        ),
        WorkloadSpec(
            "aabb_spatial_hash",
            "candidate_query_only",
            "spatial_hash_candidate_only_workload",
            lambda: spatial_hash_candidate_only_workload(scene, index),
            "query prebuilt spatial hash only",
        ),
        WorkloadSpec(
            "triangle_projection",
            "local_projection_only",
            "spatial_hash_plus_local_projection_workload",
            lambda: spatial_hash_plus_local_projection_workload(scene, candidates),
            "closest point projection over precomputed candidate faces only",
        ),
        WorkloadSpec(
            "dynamic_fem_sdf",
            "dynamic_sdf_query_only",
            "dynamic_sdf_query_only_workload",
            lambda: dynamic_sdf_query_only_workload(scene, candidates),
            "dynamic oriented local SDF over precomputed candidate faces only",
        ),
        WorkloadSpec(
            "penalty_contact",
            "contact_jacobian_assembly_only",
            "contact_jacobian_assembly_workload",
            lambda: contact_jacobian_assembly_workload(
                constraints,
                n_total_dofs,
                master_dof_offset,
            ),
            "assemble sparse contact Jacobian from precomputed constraints only",
        ),
        WorkloadSpec(
            "penalty_contact",
            "penalty_force_assembly_only",
            "penalty_force_assembly_only_workload",
            lambda: penalty_force_assembly_only_workload(
                constraints,
                jacobian,
                stiffness=1.0e4,
            ),
            "assemble penalty force from precomputed constraints and Jacobian only",
        ),
        WorkloadSpec(
            "dynamic_fem_sdf_penalty",
            "full_contact_pipeline_end_to_end",
            "full_contact_pipeline_workload",
            lambda: full_contact_pipeline_workload(scene),
            "build broad phase, query candidates, compute constraints, assemble penalty response",
        ),
        WorkloadSpec(
            "native_fem",
            "linear_solve_only",
            "linear_solve_only_workload",
            lambda: linear_solve_only_workload(A, rhs),
            "solve preassembled sparse effective system only",
        ),
    ]


def comparison_method_specs(scene: dict[str, Any]) -> list[WorkloadSpec]:
    index = build_spatial_hash(scene)
    candidates = query_spatial_hash_candidates(scene, index)
    return [
        WorkloadSpec(
            "brute_force_all_triangle_projection",
            "contact_detection_and_gap",
            "brute_force_all_triangle_projection_workload",
            lambda: brute_force_all_triangle_projection_workload(scene),
            "project against every triangle, no broad phase",
        ),
        WorkloadSpec(
            "spatial_hash_candidate_only",
            "candidate_query_only",
            "spatial_hash_candidate_only_workload",
            lambda: spatial_hash_candidate_only_workload(scene, index),
            "query prebuilt padded AABB spatial hash only",
        ),
        WorkloadSpec(
            "spatial_hash_plus_local_projection",
            "contact_detection_and_gap",
            "spatial_hash_plus_local_projection_workload",
            lambda: spatial_hash_plus_local_projection_workload(scene, candidates),
            "precomputed hash candidates plus closest point projection",
        ),
        WorkloadSpec(
            "dynamic_fem_induced_sdf_full_pipeline",
            "contact_detection_and_gap",
            "dynamic_fem_induced_sdf_full_pipeline_workload",
            lambda: dynamic_fem_induced_sdf_full_pipeline_workload(scene),
            "build hash, query candidates, and compute current-surface oriented local SDF",
        ),
    ]


def run_phase_benchmark(iterations: int) -> list[BenchmarkRecord]:
    scene = build_contact_scene()
    records: list[BenchmarkRecord] = []
    for spec in phase_benchmark_specs(scene):
        record, _ = time_stage(
            "contact_pipeline",
            spec,
            iterations=iterations,
        )
        records.append(record)
    return records


def run_method_comparison(iterations: int) -> list[BenchmarkRecord]:
    scene = build_contact_scene()
    records: list[BenchmarkRecord] = []
    for spec in comparison_method_specs(scene):
        record, _ = time_stage(
            "contact_method_comparison",
            spec,
            iterations=iterations,
        )
        records.append(record)

    records.append(
        {
            "benchmark": "contact_method_comparison",
            "method": "material_space_sdf_baseline",
            "stage": "contact_detection_and_gap",
            "workload": "unavailable",
            "iterations": 0,
            "total_seconds": "",
            "mean_seconds": "",
            "status": "unavailable",
            "details": "no material-space SDF baseline is implemented yet",
        }
    )
    return records


def write_csv(records: list[BenchmarkRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "method",
        "stage",
        "workload",
        "iterations",
        "total_seconds",
        "mean_seconds",
        "status",
        "details",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_markdown_summary(records: list[BenchmarkRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Contact Benchmark Summary",
        "",
        "| Benchmark | Method | Stage | Workload | Mean seconds | Status | Details |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        mean = record["mean_seconds"]
        mean_text = "" if mean == "" else f"{float(mean):.6e}"
        lines.append(
            "| {benchmark} | {method} | {stage} | {workload} | {mean} | {status} | {details} |".format(
                benchmark=record["benchmark"],
                method=record["method"],
                stage=record["stage"],
                workload=record["workload"],
                mean=mean_text,
                status=record["status"],
                details=record["details"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_optional_plot(records: list[BenchmarkRecord], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    timed = [r for r in records if r["mean_seconds"] != ""]
    labels = [f"{r['method']}\n{r['stage']}" for r in timed]
    values = [float(r["mean_seconds"]) for r in timed]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 4))
    ax.bar(range(len(values)), values)
    ax.set_yscale("log")
    ax.set_ylabel("Mean seconds")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "benchmarks")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--quick", action="store_true", help="Use a small CI-friendly iteration count.")
    parser.add_argument("--plot", action="store_true", help="Write a matplotlib plot if matplotlib is installed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    iterations = 3 if args.quick else int(args.iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    records = run_phase_benchmark(iterations) + run_method_comparison(iterations)
    csv_path = args.out_dir / "contact_benchmark.csv"
    md_path = args.out_dir / "contact_benchmark.md"
    write_csv(records, csv_path)
    write_markdown_summary(records, md_path)
    if args.plot:
        write_optional_plot(records, args.out_dir / "contact_benchmark.png")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
