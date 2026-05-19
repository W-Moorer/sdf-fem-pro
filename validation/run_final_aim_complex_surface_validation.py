"""Validate final-aim Lagrangian SDF oracle on a deformed complex surface.

This runner measures geometric error and timing for a warped FEM surface:

    MaterialSDF + LagrangianSDFContactOracle

against a brute-force all-triangle current-surface closest-point projection
reference.  It also reports the existing current-space DynamicNarrowBandSDF
field path as a timing/error comparison.  The projection reference is used only
as validation evidence; it is not the final-aim query method.
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact import LagrangianSDFContactOracle  # noqa: E402
from sfc.mesh import VolumeMesh, extract_boundary_triangles  # noqa: E402
from sfc.sdf import DynamicNarrowBandSDF, MaterialSDF, surface_projection_distance_kernel  # noqa: E402
from validation.run_phase3_validation import structured_tet_block  # noqa: E402

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ComplexSurfaceCase:
    mesh: VolumeMesh
    x_current: np.ndarray
    top_faces: np.ndarray
    material: MaterialSDF
    query_points: np.ndarray
    query_offsets: np.ndarray


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


def _build_complex_surface_case(*, quick: bool) -> ComplexSurfaceCase:
    nx = ny = 4 if quick else 8
    nz = 1 if quick else 2
    base = structured_tet_block(nx, ny, nz, size=(2.0, 2.0, 0.35))
    X = base.X.copy()
    X[:, 0] -= 1.0
    X[:, 1] -= 1.0
    X[:, 2] -= 0.35
    mesh = VolumeMesh(X, base.elements, element_type="tet4")
    z_min = float(np.min(mesh.X[:, 2]))
    z_max = float(np.max(mesh.X[:, 2]))
    height = max(z_max - z_min, 1.0e-30)
    x = mesh.X[:, 0]
    y = mesh.X[:, 1]
    zeta = (mesh.X[:, 2] - z_min) / height

    current = mesh.X.copy()
    current[:, 0] += zeta * (0.08 * y + 0.035 * np.sin(np.pi * y))
    current[:, 1] += zeta * (-0.05 * x + 0.025 * np.sin(np.pi * x))
    current[:, 2] += zeta * (
        0.18 * np.sin(0.75 * np.pi * (x + 1.0)) * np.cos(0.65 * np.pi * (y + 1.0))
        + 0.06 * x * y
        + 0.035 * x
    )

    boundary_faces, _adjacent = extract_boundary_triangles(mesh.elements, mesh.X, element_type=mesh.element_type)
    top = np.isclose(mesh.X[boundary_faces][:, :, 2], z_max)
    top_faces = boundary_faces[np.all(top, axis=1)]
    if top_faces.size == 0:
        raise RuntimeError("failed to extract top surface faces")
    material = MaterialSDF.from_triangle_surface(
        mesh.X,
        top_faces,
        phi=lambda point: float(point[2] - z_max),
        gradient=lambda _point: np.asarray([0.0, 0.0, 1.0], dtype=float),
        band_radius=0.25,
    )
    points, offsets = _query_points_from_top_surface(mesh.X, current, top_faces, quick=quick)
    return ComplexSurfaceCase(
        mesh=mesh,
        x_current=current,
        top_faces=top_faces,
        material=material,
        query_points=points,
        query_offsets=offsets,
    )


def _query_points_from_top_surface(
    reference_nodes: np.ndarray,
    current_nodes: np.ndarray,
    faces: np.ndarray,
    *,
    quick: bool,
) -> tuple[np.ndarray, np.ndarray]:
    barycentric_samples = (
        np.asarray([0.50, 0.25, 0.25], dtype=float),
        np.asarray([0.25, 0.50, 0.25], dtype=float),
    )
    offsets = (-0.025, 0.018) if quick else (-0.030, -0.015, 0.018)
    points: list[np.ndarray] = []
    offset_values: list[float] = []
    for face in faces:
        centroid_ref = reference_nodes[face].mean(axis=0)
        if abs(float(centroid_ref[0])) > 0.82 or abs(float(centroid_ref[1])) > 0.82:
            continue
        tri = current_nodes[face]
        normal = _triangle_normal(tri)
        for bary in barycentric_samples:
            base = bary @ tri
            for offset in offsets:
                points.append(base + float(offset) * normal)
                offset_values.append(float(offset))
    if not points:
        raise RuntimeError("failed to generate interior top-surface query points")
    return np.vstack(points), np.asarray(offset_values, dtype=float)


def _triangle_normal(triangle: np.ndarray) -> np.ndarray:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("degenerate triangle")
    normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal
    return normal


def _projection_reference(case: ComplexSurfaceCase) -> tuple[list[Any], float]:
    face_ids = np.arange(case.top_faces.shape[0], dtype=np.int64)
    start = perf_counter()
    results = [
        surface_projection_distance_kernel(point, case.x_current, case.top_faces, face_ids)
        for point in case.query_points
    ]
    return results, perf_counter() - start


def _lagrangian_oracle_results(case: ComplexSurfaceCase) -> tuple[list[Any], float]:
    oracle = LagrangianSDFContactOracle(
        case.material,
        case.x_current,
        search_radius=0.04,
        max_newton_iterations=14,
        newton_tolerance=1.0e-13,
        patch_cell_size=0.08,
    )
    start = perf_counter()
    results = [oracle.query(point, cache_key=idx) for idx, point in enumerate(case.query_points)]
    return results, perf_counter() - start


def _dynamic_field_results(case: ComplexSurfaceCase, *, quick: bool) -> tuple[list[tuple[float, np.ndarray, np.ndarray]], float, float]:
    spacing = 0.075 if quick else 0.050
    band_radius = 0.20 if quick else 0.18
    field = DynamicNarrowBandSDF.build(
        case.x_current,
        case.top_faces,
        spacing=spacing,
        band_radius=band_radius,
        cell_size=0.18,
        gradient_mode="finite_difference",
    )
    start = perf_counter()
    rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    for point in case.query_points:
        gap, normal = field.query_gap_normal(point)
        payload = field.query_payload(point)
        closest = _payload_closest_point(case.x_current, payload.face_node_ids, payload.barycentric, payload.grid_weights)
        rows.append((float(gap), normal, closest))
    return rows, float(field.field_update_cost), perf_counter() - start


def _payload_closest_point(
    nodes: np.ndarray,
    face_node_ids: np.ndarray,
    barycentric: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    closest = np.zeros(3, dtype=float)
    total = 0.0
    for alpha, face, bary in zip(weights, face_node_ids, barycentric, strict=True):
        a = float(alpha)
        if a == 0.0:
            continue
        closest += a * (bary @ nodes[face])
        total += a
    if total <= 0.0:
        raise ValueError("field payload interpolation has zero total weight")
    return closest / total


def _accuracy_rows(
    projection: list[Any],
    oracle: list[Any],
    field: list[tuple[float, np.ndarray, np.ndarray]],
) -> tuple[list[Row], list[Row]]:
    oracle_gap_errors: list[float] = []
    oracle_normal_errors: list[float] = []
    oracle_point_errors: list[float] = []
    field_gap_errors: list[float] = []
    field_normal_errors: list[float] = []
    field_point_errors: list[float] = []
    query_rows: list[Row] = []
    for idx, (ref, lag, fld) in enumerate(zip(projection, oracle, field, strict=True)):
        field_gap, field_normal, field_closest = fld
        og = abs(float(lag.gap) - float(ref.g))
        on = _normal_error(lag.normal, ref.n)
        op = float(np.linalg.norm(lag.closest_point - ref.p))
        fg = abs(float(field_gap) - float(ref.g))
        fn = _normal_error(field_normal, ref.n)
        fp = float(np.linalg.norm(field_closest - ref.p))
        oracle_gap_errors.append(og)
        oracle_normal_errors.append(on)
        oracle_point_errors.append(op)
        field_gap_errors.append(fg)
        field_normal_errors.append(fn)
        field_point_errors.append(fp)
        query_rows.append(
            {
                "query_id": idx,
                "reference_gap": float(ref.g),
                "lagrangian_gap": float(lag.gap),
                "dynamic_field_gap": float(field_gap),
                "lagrangian_gap_abs_error": og,
                "dynamic_field_gap_abs_error": fg,
                "lagrangian_normal_l2_error": on,
                "dynamic_field_normal_l2_error": fn,
                "lagrangian_closest_point_l2_error": op,
                "dynamic_field_closest_point_l2_error": fp,
                "reference_face_id": int(ref.face_id),
                "lagrangian_face_id": int(lag.face_id),
                "lagrangian_candidates_evaluated": int(lag.candidates_evaluated),
                "lagrangian_iterations": int(lag.iterations),
                "lagrangian_converged": int(lag.converged),
            }
        )
    summary_rows = [
        _summary_row(
            "lagrangian_patch_oracle",
            oracle_gap_errors,
            oracle_normal_errors,
            oracle_point_errors,
            status=max(oracle_gap_errors) < 1.0e-9 and max(oracle_normal_errors) < 1.0e-7 and max(oracle_point_errors) < 1.0e-8,
        ),
        _summary_row(
            "dynamic_field",
            field_gap_errors,
            field_normal_errors,
            field_point_errors,
            status=max(field_gap_errors) < 6.0e-2 and max(field_normal_errors) < 4.0e-1 and max(field_point_errors) < 9.0e-2,
        ),
    ]
    return summary_rows, query_rows


def _normal_error(value: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(value, dtype=float)
    b = np.asarray(reference, dtype=float)
    direct = float(np.linalg.norm(a - b))
    flipped = float(np.linalg.norm(a + b))
    return min(direct, flipped)


def _summary_row(
    method: str,
    gap_errors: list[float],
    normal_errors: list[float],
    point_errors: list[float],
    *,
    status: bool,
) -> Row:
    return {
        "method": method,
        "reference": "brute_force_current_surface_projection",
        "max_gap_abs_error": float(max(gap_errors)),
        "mean_gap_abs_error": float(np.mean(gap_errors)),
        "max_normal_l2_error": float(max(normal_errors)),
        "mean_normal_l2_error": float(np.mean(normal_errors)),
        "max_closest_point_l2_error": float(max(point_errors)),
        "mean_closest_point_l2_error": float(np.mean(point_errors)),
        "status": "passed" if status else "failed",
    }


def _timing_rows(
    *,
    query_count: int,
    projection_seconds: float,
    oracle_seconds: float,
    field_update_seconds: float,
    field_query_seconds: float,
) -> list[Row]:
    field_total = field_update_seconds + field_query_seconds
    return [
        {
            "method": "brute_force_projection_reference",
            "query_count": query_count,
            "update_seconds": 0.0,
            "query_seconds_total": projection_seconds,
            "total_seconds": projection_seconds,
            "seconds_per_query": projection_seconds / max(query_count, 1),
            "speedup_projection_over_method": 1.0,
        },
        {
            "method": "lagrangian_patch_oracle",
            "query_count": query_count,
            "update_seconds": 0.0,
            "query_seconds_total": oracle_seconds,
            "total_seconds": oracle_seconds,
            "seconds_per_query": oracle_seconds / max(query_count, 1),
            "speedup_projection_over_method": projection_seconds / oracle_seconds if oracle_seconds > 0 else "",
            "speedup_dynamic_field_total_over_method": field_total / oracle_seconds if oracle_seconds > 0 else "",
        },
        {
            "method": "dynamic_field",
            "query_count": query_count,
            "update_seconds": field_update_seconds,
            "query_seconds_total": field_query_seconds,
            "total_seconds": field_total,
            "seconds_per_query": field_query_seconds / max(query_count, 1),
            "speedup_projection_over_method": projection_seconds / field_total if field_total > 0 else "",
        },
    ]


def _write_summary(
    path: Path,
    *,
    case: ComplexSurfaceCase,
    accuracy_rows: list[Row],
    timing_rows: list[Row],
    outputs: dict[str, Path],
) -> None:
    by_method = {row["method"]: row for row in accuracy_rows}
    timing = {row["method"]: row for row in timing_rows}
    lines = [
        "# Final-Aim Complex Surface Validation",
        "",
        "This validation compares the Lagrangian material-SDF oracle against a brute-force current-surface projection reference on a warped FEM top surface.",
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
            "## Geometry",
            "",
            f"- nodes: `{case.mesh.X.shape[0]}`",
            f"- tetrahedra: `{case.mesh.elements.shape[0]}`",
            f"- top surface triangles: `{case.top_faces.shape[0]}`",
            f"- query points: `{case.query_points.shape[0]}`",
            "",
            "## Accuracy",
            "",
            "| Method | Status | Max gap error | Max normal error | Max closest-point error |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for method in ("lagrangian_patch_oracle", "dynamic_field"):
        row = by_method[method]
        lines.append(
            f"| `{method}` | {row['status']} | "
            f"{float(row['max_gap_abs_error']):.6e} | "
            f"{float(row['max_normal_l2_error']):.6e} | "
            f"{float(row['max_closest_point_l2_error']):.6e} |"
        )
    lines.extend(
        [
            "",
            "## Timing",
            "",
            "| Method | Total seconds | Seconds/query | Speedup projection/method |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method in ("brute_force_projection_reference", "lagrangian_patch_oracle", "dynamic_field"):
        row = timing[method]
        speed = row.get("speedup_projection_over_method", "")
        speed_text = "" if speed == "" else f"{float(speed):.6e}"
        lines.append(
            f"| `{method}` | {float(row['total_seconds']):.6e} | "
            f"{float(row['seconds_per_query']):.6e} | {speed_text} |"
        )
    lag_speed = timing["lagrangian_patch_oracle"].get("speedup_dynamic_field_total_over_method", "")
    if lag_speed != "":
        lines.extend(
            [
                "",
                f"- Dynamic-field total / Lagrangian-oracle total: `{float(lag_speed):.6e}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- The reference projection is validation evidence, not the final query method.",
            "- The dynamic-field row is a compatibility-path comparison.",
            "- This does not claim friction, self-contact, nonlinear FEM, or CalculiX native-contact trajectory equivalence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(out_dir: Path, *, quick: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    case = _build_complex_surface_case(quick=quick)
    projection, projection_seconds = _projection_reference(case)
    oracle, oracle_seconds = _lagrangian_oracle_results(case)
    field, field_update_seconds, field_query_seconds = _dynamic_field_results(case, quick=quick)
    accuracy_rows, query_rows = _accuracy_rows(projection, oracle, field)
    timing_rows = _timing_rows(
        query_count=case.query_points.shape[0],
        projection_seconds=projection_seconds,
        oracle_seconds=oracle_seconds,
        field_update_seconds=field_update_seconds,
        field_query_seconds=field_query_seconds,
    )
    outputs = {
        "accuracy": out_dir / "final_aim_complex_surface_accuracy.csv",
        "queries": out_dir / "final_aim_complex_surface_queries.csv",
        "timing": out_dir / "final_aim_complex_surface_timing.csv",
        "summary": out_dir / "final_aim_complex_surface_summary.md",
    }
    _write_csv(outputs["accuracy"], accuracy_rows)
    _write_csv(outputs["queries"], query_rows)
    _write_csv(outputs["timing"], timing_rows)
    _write_summary(outputs["summary"], case=case, accuracy_rows=accuracy_rows, timing_rows=timing_rows, outputs=outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "final_aim_complex_surface")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(args.out_dir, quick=bool(args.quick))
    print("Final-aim complex surface validation complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
