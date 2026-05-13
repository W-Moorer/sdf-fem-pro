"""Phase-5 paper experiment preparation for the minimal FEM-SDF solver.

This runner prepares paper-facing evidence without changing the core physics.
It exercises the existing linear TET4, current-surface dynamic SDF, spatial
hash broad phase, and penalty contact code. Friction, self-contact, nonlinear
FEM, GPU execution, and barrier contact remain out of scope.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sfc.contact import (  # noqa: E402
    SurfaceSample,
    UniformTriangleAABBHash,
    contact_constraint_from_sample,
    penalty_contact_response,
)
from sfc.fem import assemble_stiffness_matrix  # noqa: E402
from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.sdf import dynamic_surface_sdf  # noqa: E402
from validation.run_phase3_validation import _body, structured_tet_block  # noqa: E402
from validation.run_phase4_paper_validation import (  # noqa: E402
    _candidate_query,
    _dynamic_surface_query,
    _format_float,
    _full_dynamic_contact_pipeline,
    _local_projection_query,
    _plane_surface,
    _brute_force_surface_query,
    contact_time_history,
    mesh_resolution_study,
)

Row = dict[str, Any]


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _repeat_timing(func: Callable[[], Any], repeats: int) -> tuple[dict[str, float], Any]:
    times: list[float] = []
    result = None
    for _ in range(max(1, repeats)):
        start = perf_counter()
        result = func()
        times.append(perf_counter() - start)
    values = np.asarray(times, dtype=float)
    stats = {
        "mean_seconds": float(np.mean(values)),
        "std_seconds": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
    }
    return stats, result


def metadata_rows(*, run_mode: str, surface_sizes: Iterable[int], repeats: int) -> list[Row]:
    """Return reproducibility metadata that is not tied to a user path."""

    return [
        {"key": "run_mode", "value": run_mode},
        {"key": "surface_sizes", "value": ",".join(str(int(v)) for v in surface_sizes)},
        {"key": "timing_repeats", "value": str(int(repeats))},
        {"key": "python_version", "value": platform.python_version()},
        {"key": "numpy_version", "value": np.__version__},
        {"key": "scipy_version", "value": scipy.__version__},
        {"key": "matplotlib_version", "value": matplotlib.__version__},
        {"key": "platform_system", "value": platform.system()},
        {"key": "platform_release", "value": platform.release()},
        {"key": "platform_machine", "value": platform.machine()},
        {"key": "cpu_count_logical", "value": str(os.cpu_count() or "")},
        {"key": "timer", "value": "time.perf_counter"},
        {"key": "core_physics_scope", "value": "linear_tet4_oriented_dynamic_sdf_penalty_contact"},
    ]


def performance_scaling(surface_sizes: Iterable[int], repeats: int, *, run_mode: str) -> list[Row]:
    """Measure repeated contact-detection timings over several surface sizes."""

    rows: list[Row] = []
    for resolution in surface_sizes:
        r = int(resolution)
        nodes, faces, points = _plane_surface(r)
        index, candidates = _candidate_query(points, nodes, faces)
        timing_by_key: dict[str, float] = {}

        workloads: list[tuple[str, str, str, Callable[[], Any]]] = [
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
                lambda n=nodes, f=faces: UniformTriangleAABBHash.from_surface(
                    n,
                    f,
                    delta_safe=0.02,
                    cell_size=1.0 / r,
                ),
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

        for method, stage, workload, func in workloads:
            stats, _ = _repeat_timing(func, repeats)
            timing_by_key[f"{method}:{stage}"] = stats["mean_seconds"]
            rows.append(
                {
                    "run_mode": run_mode,
                    "surface_resolution": r,
                    "triangle_count": faces.shape[0],
                    "query_point_count": points.shape[0],
                    "method": method,
                    "stage": stage,
                    "workload": workload,
                    "repeats": repeats,
                    "speedup_vs_bruteforce": "",
                    "speedup_status": "not_applicable",
                    "status": "ok",
                    "details": "repeated isolated timing row",
                    **stats,
                }
            )

        brute = timing_by_key["brute_force_all_triangle_projection:contact_detection_end_to_end"]
        for row in rows:
            if int(row["surface_resolution"]) != r:
                continue
            if row["stage"] not in {"contact_detection_end_to_end", "full_contact_pipeline_end_to_end"}:
                continue
            if row["method"] == "brute_force_all_triangle_projection":
                continue
            speedup = brute / float(row["mean_seconds"]) if float(row["mean_seconds"]) > 0.0 else math.inf
            if speedup > 1.0:
                row["speedup_vs_bruteforce"] = speedup
                row["speedup_status"] = "speedup_supported"
            else:
                row["speedup_vs_bruteforce"] = ""
                row["speedup_status"] = "no_speedup"
                row["details"] = "speedup <= 1; acceleration claim not supported for this row"

        rows.append(
            {
                "run_mode": run_mode,
                "surface_resolution": r,
                "triangle_count": faces.shape[0],
                "query_point_count": points.shape[0],
                "method": "material_space_sdf_baseline",
                "stage": "contact_detection_end_to_end",
                "workload": "material_space_sdf_baseline_error_study",
                "repeats": 0,
                "mean_seconds": "",
                "std_seconds": "",
                "min_seconds": "",
                "max_seconds": "",
                "speedup_vs_bruteforce": "",
                "speedup_status": "baseline_not_timed_as_main_method",
                "status": "available_as_error_baseline",
                "details": "material-space SDF baseline is compared in phase5_material_space_sdf_baseline.csv",
            }
        )
    return rows


def _affine_deformations() -> dict[str, np.ndarray]:
    return {
        "stretch": np.array([[1.2, 0.0, 0.0], [0.0, 0.85, 0.0], [0.0, 0.0, 1.15]], dtype=float),
        "shear": np.array([[1.0, 0.25, 0.0], [0.05, 1.0, 0.0], [0.25, -0.20, 1.0]], dtype=float),
    }


def _current_surface_normal(F: np.ndarray) -> np.ndarray:
    tangent_a = F @ np.array([1.0, 0.0, 0.0])
    tangent_b = F @ np.array([0.0, 1.0, 0.0])
    n = np.cross(tangent_a, tangent_b)
    return n / np.linalg.norm(n)


def _frozen_material_space_sdf(x: np.ndarray, F_inv: np.ndarray) -> tuple[float, np.ndarray]:
    """Frozen material-space plane SDF baseline, intentionally not the main method."""

    X = F_inv @ x
    return float(X[2]), np.array([0.0, 0.0, 1.0], dtype=float)


def material_space_sdf_baseline(resolution: int) -> list[Row]:
    """Compare a frozen material-space SDF baseline against current-surface SDF."""

    rest_nodes, faces, _ = _plane_surface(resolution)
    material_samples = np.array(
        [
            [0.2, 0.2, 0.0],
            [0.5, 0.25, 0.0],
            [0.75, 0.35, 0.0],
            [0.35, 0.7, 0.0],
            [0.65, 0.65, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([-0.03, -0.01, 0.02, 0.04], dtype=float)
    all_faces = np.arange(faces.shape[0], dtype=np.int64)
    rows: list[Row] = []

    for deformation, F in _affine_deformations().items():
        F_inv = np.linalg.inv(F)
        current_nodes = rest_nodes @ F.T
        normal = _current_surface_normal(F)
        for sample_id, Xs in enumerate(material_samples):
            current_surface_point = F @ Xs
            for offset in offsets:
                x = current_surface_point + offset * normal
                reference = dynamic_surface_sdf(x, current_nodes, faces, all_faces)
                baseline_gap, baseline_normal = _frozen_material_space_sdf(x, F_inv)
                normal_dot = float(np.clip(np.dot(reference.n, baseline_normal), -1.0, 1.0))
                rows.append(
                    {
                        "deformation": deformation,
                        "surface_resolution": resolution,
                        "sample_id": sample_id,
                        "offset": offset,
                        "reference_gap": reference.g,
                        "baseline_gap": baseline_gap,
                        "gap_error": abs(float(reference.g) - baseline_gap),
                        "reference_normal_x": reference.n[0],
                        "reference_normal_y": reference.n[1],
                        "reference_normal_z": reference.n[2],
                        "baseline_normal_x": baseline_normal[0],
                        "baseline_normal_y": baseline_normal[1],
                        "baseline_normal_z": baseline_normal[2],
                        "normal_angle_error_rad": math.acos(normal_dot),
                        "baseline_label": "frozen_material_space_sdf_baseline",
                        "status": "ok",
                    }
                )
    return rows


def analytic_linear_elastic_reference(resolutions: Iterable[int]) -> list[Row]:
    """Compare affine uniaxial TET4 energy to an analytic linear-elastic value."""

    rows: list[Row] = []
    E, nu, eps = 2.0e5, 0.25, 1.0e-3
    analytic = 0.5 * float(isotropic_linear_elasticity_matrix(E, nu)[0, 0]) * eps * eps
    for resolution in resolutions:
        mesh = structured_tet_block(int(resolution), int(resolution), int(resolution))
        body = _body(mesh, E=E, nu=nu)
        K = assemble_stiffness_matrix(body)
        u_nodes = np.zeros_like(mesh.X)
        u_nodes[:, 0] = eps * mesh.X[:, 0]
        u = u_nodes.ravel()
        numerical = 0.5 * float(u @ (K @ u))
        rel_error = abs(numerical - analytic) / max(abs(analytic), 1.0e-30)
        rows.append(
            {
                "case": "analytic_uniaxial_linear_elastic_energy",
                "resolution": int(resolution),
                "nodes": mesh.X.shape[0],
                "elements": mesh.elements.shape[0],
                "numerical_value": numerical,
                "reference_value": analytic,
                "absolute_error": abs(numerical - analytic),
                "relative_error": rel_error,
                "status": "ok" if rel_error < 1.0e-12 else "check",
                "details": "affine strain energy compared to analytic small-strain linear elasticity",
            }
        )
    return rows


def contact_reference_validation(resolution: int) -> list[Row]:
    """Compare spatial-hash dynamic SDF queries to all-face brute-force CPP."""

    rest_nodes, faces, base_points = _plane_surface(resolution)
    F = _affine_deformations()["shear"]
    nodes = rest_nodes @ F.T
    normal = _current_surface_normal(F)
    points = (base_points @ F.T) + 0.012 * normal
    all_faces = np.arange(faces.shape[0], dtype=np.int64)
    index = UniformTriangleAABBHash.from_surface(nodes, faces, delta_safe=0.04, cell_size=1.0 / resolution)
    rows: list[Row] = []

    for point_id, point in enumerate(points):
        reference = dynamic_surface_sdf(point, nodes, faces, all_faces)
        candidates = index.query_point(point)
        method = dynamic_surface_sdf(point, nodes, faces, candidates)
        normal_error = float(np.linalg.norm(reference.n - method.n))
        rows.append(
            {
                "case": "high_resolution_brute_force_cpp_contact_reference",
                "surface_resolution": resolution,
                "triangle_count": faces.shape[0],
                "point_id": point_id,
                "candidate_count": candidates.size,
                "reference_gap": reference.g,
                "method_gap": method.g,
                "gap_error": abs(float(reference.g) - float(method.g)),
                "normal_error": normal_error,
                "status": "ok" if candidates.size > 0 and normal_error < 1.0e-12 else "check",
                "details": "all-face closest point projection reference versus spatial-hash candidate dynamic SDF",
            }
        )
    return rows


def _regression_order(mesh_rows: list[Row]) -> tuple[float, float, int]:
    gravity = [
        row
        for row in mesh_rows
        if row["case"] == "fixed_base_gravity_elastic_block"
        and row["resolution"] != "all"
        and row["relative_to_finest"] != ""
        and float(row["relative_to_finest"]) > 0.0
    ]
    if len(gravity) < 3:
        return 0.0, 0.0, len(gravity)
    h = np.array([1.0 / float(row["resolution"]) for row in gravity], dtype=float)
    err = np.array([float(row["relative_to_finest"]) for row in gravity], dtype=float)
    coeff = np.polyfit(np.log(h), np.log(err), 1)
    pred = np.polyval(coeff, np.log(h))
    ss_res = float(np.sum((np.log(err) - pred) ** 2))
    ss_tot = float(np.sum((np.log(err) - np.mean(np.log(err))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return float(coeff[0]), r2, len(gravity)


def claim_rows(
    *,
    run_mode: str,
    mesh_rows: list[Row],
    performance_rows: list[Row],
    material_rows: list[Row],
    reference_rows: list[Row],
    contact_reference_rows: list[Row],
) -> list[Row]:
    dynamic_rows = [
        row
        for row in performance_rows
        if row["method"] == "dynamic_fem_induced_sdf_full_pipeline"
        and row["stage"] == "full_contact_pipeline_end_to_end"
    ]
    speedups = [float(row["speedup_vs_bruteforce"]) for row in dynamic_rows if row["speedup_vs_bruteforce"] != ""]
    min_speedup = min(speedups) if speedups else 0.0
    surface_count = len({int(row["surface_resolution"]) for row in dynamic_rows})
    repeat_count = min((int(row["repeats"]) for row in dynamic_rows), default=0)
    acceleration_supported = run_mode == "full" and surface_count >= 4 and repeat_count >= 3 and min_speedup > 1.0

    order, r2, regression_points = _regression_order(mesh_rows)
    convergence_supported = regression_points >= 3 and order > 0.0 and r2 >= 0.8
    material_supported = len(material_rows) > 0 and len({row["deformation"] for row in material_rows}) >= 2
    reference_supported = len(reference_rows) > 0 and len(contact_reference_rows) > 0

    return [
        {
            "claim_id": "final_acceleration_claim",
            "claim_text": "Final acceleration claim is allowed only for repeated non-quick timing with dynamic FEM-SDF speedup greater than one.",
            "evidence_csv": "phase5_performance_scaling.csv",
            "evidence_field": "speedup_vs_bruteforce",
            "gate_field": "run_mode,min_speedup,surface_count,repeat_count",
            "gate_value": f"{run_mode},{min_speedup},{surface_count},{repeat_count}",
            "claim_status": "supported" if acceleration_supported else "not_supported",
            "details": "quick mode cannot support the final acceleration claim",
        },
        {
            "claim_id": "convergence_order_claim",
            "claim_text": "A convergence-order claim requires a log-log regression over at least three nonzero error points.",
            "evidence_csv": "phase5_mesh_resolution.csv",
            "evidence_field": "relative_to_finest",
            "gate_field": "regression_points,order,r2",
            "gate_value": f"{regression_points},{order},{r2}",
            "claim_status": "supported" if convergence_supported else "not_supported",
            "details": "trend evidence is reported separately from theoretical convergence-order claims",
        },
        {
            "claim_id": "material_space_baseline_comparison",
            "claim_text": "Material-space SDF baseline comparison is allowed only when baseline error data exists.",
            "evidence_csv": "phase5_material_space_sdf_baseline.csv",
            "evidence_field": "gap_error",
            "gate_field": "baseline_row_count",
            "gate_value": str(len(material_rows)),
            "claim_status": "supported" if material_supported else "not_supported",
            "details": "baseline is frozen material-space SDF, not the main method",
        },
        {
            "claim_id": "external_reference_validation",
            "claim_text": "Reference validation includes analytic linear elasticity and brute-force CPP contact evidence.",
            "evidence_csv": "phase5_external_reference_validation.csv;phase5_contact_reference.csv",
            "evidence_field": "relative_error;gap_error",
            "gate_field": "reference_rows,contact_reference_rows",
            "gate_value": f"{len(reference_rows)},{len(contact_reference_rows)}",
            "claim_status": "supported" if reference_supported else "not_supported",
            "details": "Abaqus is not used or required",
        },
    ]


def _save_plot(out_dir: Path, name: str, build: Callable[[], None]) -> Row:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 4.0))
    build()
    plt.tight_layout()
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    plt.savefig(png, dpi=160)
    plt.savefig(pdf)
    plt.close()
    return {"plot": name, "png": png.name, "pdf": pdf.name, "status": "ok"}


def write_plots(
    out_dir: Path,
    mesh_rows: list[Row],
    history_rows: list[Row],
    performance_rows: list[Row],
    material_rows: list[Row],
) -> list[Row]:
    plot_rows: list[Row] = []

    def mesh_plot() -> None:
        gravity = [row for row in mesh_rows if row["case"] == "fixed_base_gravity_elastic_block"]
        x = [int(row["resolution"]) for row in gravity if str(row["resolution"]).isdigit()]
        y = [abs(float(row["value"])) for row in gravity if str(row["resolution"]).isdigit()]
        plt.plot(x, y, marker="o", label="gravity tip |uz|")
        uniaxial = [row for row in mesh_rows if row["case"] == "uniaxial_linear_elastic_patch"]
        xu = [int(row["resolution"]) for row in uniaxial]
        yu = [max(float(row["value"]), 1.0e-18) for row in uniaxial]
        plt.semilogy(xu, yu, marker="s", label="uniaxial energy error")
        plt.xlabel("mesh resolution")
        plt.ylabel("metric value")
        plt.legend()

    plot_rows.append(_save_plot(out_dir, "phase5_mesh_resolution_trend", mesh_plot))

    def contact_plot() -> None:
        for case in sorted({row["case"] for row in history_rows}):
            rows = [row for row in history_rows if row["case"] == case]
            plt.plot([float(row["time"]) for row in rows], [float(row["max_penetration"]) for row in rows], marker="o", label=case)
        plt.xlabel("time")
        plt.ylabel("max penetration")
        plt.legend(fontsize=7)

    plot_rows.append(_save_plot(out_dir, "phase5_contact_time_history", contact_plot))

    endpoint_methods = {
        "brute_force_all_triangle_projection",
        "spatial_hash_plus_local_projection",
        "dynamic_fem_induced_sdf_full_pipeline",
    }

    def performance_plot() -> None:
        for method in sorted(endpoint_methods):
            rows = [
                row
                for row in performance_rows
                if row["method"] == method
                and row["stage"] in {"contact_detection_end_to_end", "full_contact_pipeline_end_to_end"}
            ]
            plt.plot(
                [int(row["triangle_count"]) for row in rows],
                [float(row["mean_seconds"]) for row in rows],
                marker="o",
                label=method,
            )
        plt.xlabel("triangle count")
        plt.ylabel("mean seconds")
        plt.yscale("log")
        plt.legend(fontsize=7)

    plot_rows.append(_save_plot(out_dir, "phase5_performance_scaling", performance_plot))

    def speedup_plot() -> None:
        for method in ("spatial_hash_plus_local_projection", "dynamic_fem_induced_sdf_full_pipeline"):
            rows = [
                row
                for row in performance_rows
                if row["method"] == method
                and row["speedup_vs_bruteforce"] != ""
            ]
            plt.plot(
                [int(row["triangle_count"]) for row in rows],
                [float(row["speedup_vs_bruteforce"]) for row in rows],
                marker="o",
                label=method,
            )
        plt.axhline(1.0, color="0.4", linewidth=1.0)
        plt.xlabel("triangle count")
        plt.ylabel("speedup vs brute force")
        plt.legend(fontsize=7)

    plot_rows.append(_save_plot(out_dir, "phase5_speedup", speedup_plot))

    def material_plot() -> None:
        labels = []
        gap_errors = []
        normal_errors = []
        for deformation in sorted({row["deformation"] for row in material_rows}):
            rows = [row for row in material_rows if row["deformation"] == deformation]
            labels.append(deformation)
            gap_errors.append(max(float(row["gap_error"]) for row in rows))
            normal_errors.append(max(float(row["normal_angle_error_rad"]) for row in rows))
        x = np.arange(len(labels))
        width = 0.36
        plt.bar(x - width / 2, gap_errors, width, label="max gap error")
        plt.bar(x + width / 2, normal_errors, width, label="max normal angle error")
        plt.xticks(x, labels)
        plt.ylabel("error")
        plt.legend()

    plot_rows.append(_save_plot(out_dir, "phase5_material_space_sdf_error", material_plot))
    return plot_rows


def _summary_by_method(performance_rows: list[Row]) -> list[Row]:
    return [
        row
        for row in performance_rows
        if row["stage"] in {"contact_detection_end_to_end", "full_contact_pipeline_end_to_end"}
    ]


def write_markdown(
    path: Path,
    *,
    run_mode: str,
    mesh_rows: list[Row],
    history_rows: list[Row],
    performance_rows: list[Row],
    material_rows: list[Row],
    reference_rows: list[Row],
    contact_reference_rows: list[Row],
    claims: list[Row],
    plots: list[Row],
) -> None:
    lines = [
        "# Phase-5 Paper Experiment Preparation Summary",
        "",
        "This report is generated evidence for the current minimal solver. It does not add friction, self-contact, nonlinear FEM, GPU execution, or barrier contact.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python validation/run_phase5_paper_experiments.py {'--quick ' if run_mode == 'quick' else ''}--out-dir results/phase5",
        "```",
        "",
        "## Claims",
        "",
        "| Claim | Status | Evidence | Gate |",
        "| --- | --- | --- | --- |",
    ]
    for claim in claims:
        lines.append(
            f"| {claim['claim_id']} | {claim['claim_status']} | {claim['evidence_csv']}::{claim['evidence_field']} | {claim['gate_field']}={claim['gate_value']} |"
        )
        csv_names = str(claim["evidence_csv"]).split(";")
        fields = str(claim["evidence_field"]).split(";")
        for csv_name, field in zip(csv_names, fields):
            lines.append(f"<!-- evidence csv={csv_name} field={field} -->")

    lines.extend(
        [
            "",
            "## Performance End-To-End Rows",
            "",
            "| Resolution | Triangles | Method | Mean | Std | Min | Max | Speedup Status | Speedup |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in _summary_by_method(performance_rows):
        lines.append(
            "| {resolution} | {triangles} | {method} | {mean} | {std} | {minv} | {maxv} | {status} | {speedup} |".format(
                resolution=row["surface_resolution"],
                triangles=row["triangle_count"],
                method=row["method"],
                mean=_format_float(row["mean_seconds"]),
                std=_format_float(row["std_seconds"]),
                minv=_format_float(row["min_seconds"]),
                maxv=_format_float(row["max_seconds"]),
                status=row["speedup_status"],
                speedup=_format_float(row["speedup_vs_bruteforce"]),
            )
        )

    lines.extend(
        [
            "",
            "## Material-Space Baseline",
            "",
            "| Deformation | Max Gap Error | Max Normal Angle Error |",
            "| --- | ---: | ---: |",
        ]
    )
    for deformation in sorted({row["deformation"] for row in material_rows}):
        rows = [row for row in material_rows if row["deformation"] == deformation]
        lines.append(
            "| {deformation} | {gap} | {normal} |".format(
                deformation=deformation,
                gap=_format_float(max(float(row["gap_error"]) for row in rows)),
                normal=_format_float(max(float(row["normal_angle_error_rad"]) for row in rows)),
            )
        )

    lines.extend(
        [
            "",
            "## Reference Validation",
            "",
            f"- Analytic linear-elastic rows: `{len(reference_rows)}`.",
            f"- Brute-force CPP contact reference rows: `{len(contact_reference_rows)}`.",
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
            "## Limitations",
            "",
            "- Quick mode cannot support the final acceleration claim.",
            "- Convergence-order claims require a supported regression over at least three nonzero error points.",
            "- The material-space SDF is a frozen baseline only; it is not the main method.",
            "- Linear TET4 and penalty normal contact remain the only validated physics in this phase.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase5(*, quick: bool, out_dir: Path) -> dict[str, list[Row]]:
    run_mode = "quick" if quick else "full"
    mesh_resolutions = [1, 2, 3] if quick else [1, 2, 3, 4]
    surface_sizes = [2, 3, 4, 5] if quick else [4, 8, 12, 16]
    repeats = 2 if quick else 5
    history_steps = 6 if quick else 30
    material_resolution = 4 if quick else 8
    contact_reference_resolution = 5 if quick else 12

    mesh_rows = mesh_resolution_study(mesh_resolutions)
    history_rows = contact_time_history(history_steps)
    performance_rows = performance_scaling(surface_sizes, repeats, run_mode=run_mode)
    material_rows = material_space_sdf_baseline(material_resolution)
    reference_rows = analytic_linear_elastic_reference(mesh_resolutions)
    contact_reference_rows = contact_reference_validation(contact_reference_resolution)
    claims = claim_rows(
        run_mode=run_mode,
        mesh_rows=mesh_rows,
        performance_rows=performance_rows,
        material_rows=material_rows,
        reference_rows=reference_rows,
        contact_reference_rows=contact_reference_rows,
    )
    plots = write_plots(out_dir, mesh_rows, history_rows, performance_rows, material_rows)
    metadata = metadata_rows(run_mode=run_mode, surface_sizes=surface_sizes, repeats=repeats)
    return {
        "mesh": mesh_rows,
        "history": history_rows,
        "performance": performance_rows,
        "material": material_rows,
        "reference": reference_rows,
        "contact_reference": contact_reference_rows,
        "claims": claims,
        "plots": plots,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use reduced deterministic sizes for CI.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "phase5")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = run_phase5(quick=bool(args.quick), out_dir=args.out_dir)

    _write_csv(
        args.out_dir / "phase5_metadata.csv",
        ["key", "value"],
        outputs["metadata"],
    )
    _write_csv(
        args.out_dir / "phase5_mesh_resolution.csv",
        ["case", "resolution", "nodes", "elements", "metric", "value", "relative_to_finest", "trend", "claim_status", "details"],
        outputs["mesh"],
    )
    _write_csv(
        args.out_dir / "phase5_contact_time_history.csv",
        ["case", "step", "time", "min_gap", "max_penetration", "active_contact_count", "normal_force", "contact_energy", "action_reaction_imbalance", "status"],
        outputs["history"],
    )
    _write_csv(
        args.out_dir / "phase5_performance_scaling.csv",
        [
            "run_mode",
            "surface_resolution",
            "triangle_count",
            "query_point_count",
            "method",
            "stage",
            "workload",
            "repeats",
            "mean_seconds",
            "std_seconds",
            "min_seconds",
            "max_seconds",
            "speedup_vs_bruteforce",
            "speedup_status",
            "status",
            "details",
        ],
        outputs["performance"],
    )
    _write_csv(
        args.out_dir / "phase5_material_space_sdf_baseline.csv",
        [
            "deformation",
            "surface_resolution",
            "sample_id",
            "offset",
            "reference_gap",
            "baseline_gap",
            "gap_error",
            "reference_normal_x",
            "reference_normal_y",
            "reference_normal_z",
            "baseline_normal_x",
            "baseline_normal_y",
            "baseline_normal_z",
            "normal_angle_error_rad",
            "baseline_label",
            "status",
        ],
        outputs["material"],
    )
    _write_csv(
        args.out_dir / "phase5_external_reference_validation.csv",
        ["case", "resolution", "nodes", "elements", "numerical_value", "reference_value", "absolute_error", "relative_error", "status", "details"],
        outputs["reference"],
    )
    _write_csv(
        args.out_dir / "phase5_contact_reference.csv",
        ["case", "surface_resolution", "triangle_count", "point_id", "candidate_count", "reference_gap", "method_gap", "gap_error", "normal_error", "status", "details"],
        outputs["contact_reference"],
    )
    _write_csv(
        args.out_dir / "phase5_claims.csv",
        ["claim_id", "claim_text", "evidence_csv", "evidence_field", "gate_field", "gate_value", "claim_status", "details"],
        outputs["claims"],
    )
    _write_csv(args.out_dir / "phase5_plots.csv", ["plot", "png", "pdf", "status"], outputs["plots"])
    write_markdown(
        args.out_dir / "phase5_summary.md",
        run_mode="quick" if args.quick else "full",
        mesh_rows=outputs["mesh"],
        history_rows=outputs["history"],
        performance_rows=outputs["performance"],
        material_rows=outputs["material"],
        reference_rows=outputs["reference"],
        contact_reference_rows=outputs["contact_reference"],
        claims=outputs["claims"],
        plots=outputs["plots"],
    )

    for name in [
        "phase5_metadata.csv",
        "phase5_mesh_resolution.csv",
        "phase5_contact_time_history.csv",
        "phase5_performance_scaling.csv",
        "phase5_material_space_sdf_baseline.csv",
        "phase5_external_reference_validation.csv",
        "phase5_contact_reference.csv",
        "phase5_claims.csv",
        "phase5_plots.csv",
        "phase5_summary.md",
    ]:
        print(f"Wrote {args.out_dir / name}")
    for plot in outputs["plots"]:
        print(f"Wrote {args.out_dir / plot['png']}")
        print(f"Wrote {args.out_dir / plot['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
