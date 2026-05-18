"""Run solver-level timing cases for true-field FEM-SDF contact.

This runner complements the field-query crossover tests with full step-level
accounting on native linear TET4/HEX8 contact states. The comparison reference
is the internal spatial-hash projection kernel used only for timing and
accuracy reference. The field update uses exact required-node population: only
the trilinear grid corners touched by the current slave samples are populated,
and each populated node still uses the same closest-feature projection kernel
as the full grid builder. The query path then uses ``DynamicNarrowBandSDF`` by
interpolation through ``field_contact``.
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
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact import SurfaceSample, compute_field_contact_constraints  # noqa: E402
from sfc.contact.broad_phase import UniformTriangleAABBHash  # noqa: E402
from sfc.fem import DeformableBody, assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402
from sfc.sdf.dynamic_surface_sdf import surface_projection_distance_kernel  # noqa: E402
from validation.run_phase10_true_field_contact_cases import _structured_hex_block  # noqa: E402
from validation.run_phase3_validation import structured_tet_block  # noqa: E402

Row = dict[str, Any]


def _configure_times_fonts(matplotlib_module: Any) -> None:
    matplotlib_module.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


@dataclass(frozen=True, slots=True)
class TimingModel:
    mesh: VolumeMesh
    body: DeformableBody
    element_type: str
    bottom_corners: np.ndarray


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


def _plot_outputs(out_dir: Path, step_rows: list[Row]) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    _configure_times_fonts(matplotlib)
    import matplotlib.pyplot as plt

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    labels = [str(row["case_id"]).replace("_linear_", "\n") for row in step_rows]
    x = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(labels)), 4.2))
    field_update = np.asarray([float(row["field_update_seconds"]) for row in step_rows], dtype=float)
    field_contact = np.asarray([float(row["field_contact_seconds"]) for row in step_rows], dtype=float)
    field_solve = np.asarray([float(row["field_solve_or_integrate_seconds"]) for row in step_rows], dtype=float)
    fem = np.asarray([float(row["fem_assembly_seconds"]) for row in step_rows], dtype=float)
    bottom = np.zeros_like(x)
    for values, label, color in [
        (fem, "FEM assembly", "#b279a2"),
        (field_update, "SDF field update", "#4c78a8"),
        (field_contact, "field contact query", "#72b7b2"),
        (field_solve, "solve/integrate", "#f58518"),
    ]:
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    projection_total = np.asarray([float(row["projection_total_step_seconds"]) for row in step_rows], dtype=float)
    ax.scatter(x, projection_total, marker="D", color="black", zorder=5, label="projection total")
    ax.set_yscale("log")
    ax.set_ylabel("Step time (s)")
    ax.set_xlabel("Case")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.grid(True, axis="y", which="both", alpha=0.30)
    ax.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    png = figures / "solver_step_timing_breakdown.png"
    pdf = figures / "solver_step_timing_breakdown.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    outputs["solver_step_timing_breakdown_png"] = png
    outputs["solver_step_timing_breakdown_pdf"] = pdf

    fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(labels)), 3.8))
    speedup = [float(row["step_speedup_projection_over_field"]) for row in step_rows]
    colors = ["#54a24b" if float(value) > 1.0 else "#e45756" for value in speedup]
    ax.bar(x, speedup, color=colors)
    ax.axhline(1.0, color="black", linewidth=1.0)
    ax.set_ylabel("Projection total / field total")
    ax.set_xlabel("Case")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.30)
    fig.tight_layout()
    png = figures / "solver_step_speedup.png"
    pdf = figures / "solver_step_speedup.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    outputs["solver_step_speedup_png"] = png
    outputs["solver_step_speedup_pdf"] = pdf

    fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(labels)), 3.8))
    qstar = [float(row["query_crossover_q_star"]) for row in step_rows]
    ax.bar(x, qstar, color="#4c78a8")
    ax.set_yscale("log")
    ax.set_ylabel("Measured query crossover Q*")
    ax.set_xlabel("Case")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.grid(True, axis="y", which="both", alpha=0.30)
    fig.tight_layout()
    png = figures / "solver_query_crossover.png"
    pdf = figures / "solver_query_crossover.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    outputs["solver_query_crossover_png"] = png
    outputs["solver_query_crossover_pdf"] = pdf
    return outputs


def _timing_model(element_type: str, *, quick: bool) -> TimingModel:
    nx = ny = 2 if quick else 3
    nz = 1
    size = (1.0, 1.0, 0.20)
    if element_type == "tet4":
        base = structured_tet_block(nx, ny, nz, size=size)
    elif element_type == "hex8":
        base = _structured_hex_block(nx, ny, nz, size=size)
    else:
        raise ValueError("element_type must be tet4 or hex8")
    X = base.X.copy()
    X[:, 2] += 0.04
    mesh = VolumeMesh(X, base.elements, element_type=element_type)
    body = DeformableBody(mesh=mesh, material={"E": 200.0, "nu": 0.30}, density=1.0)
    bottom = np.nonzero(np.isclose(mesh.X[:, 2], float(np.min(mesh.X[:, 2]))))[0]
    bottom_xy = mesh.X[bottom, :2]

    def corner_id(x_value: float, y_value: float) -> int:
        distance = np.linalg.norm(bottom_xy - np.asarray([x_value, y_value], dtype=float), axis=1)
        return int(bottom[int(np.argmin(distance))])

    corners = np.asarray(
        [
            corner_id(0.0, 0.0),
            corner_id(1.0, 0.0),
            corner_id(1.0, 1.0),
            corner_id(0.0, 1.0),
        ],
        dtype=np.int64,
    )
    return TimingModel(mesh=mesh, body=body, element_type=element_type, bottom_corners=corners)


def _dense_plane_surface(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    nodes: list[tuple[float, float, float]] = []

    def node_id(i: int, j: int) -> int:
        return j * (resolution + 1) + i

    for j in range(resolution + 1):
        for i in range(resolution + 1):
            x = -0.10 + 1.20 * i / resolution
            y = -0.10 + 1.20 * j / resolution
            nodes.append((x, y, 0.0))
    faces: list[tuple[int, int, int]] = []
    for j in range(resolution):
        for i in range(resolution):
            n00 = node_id(i, j)
            n10 = node_id(i + 1, j)
            n01 = node_id(i, j + 1)
            n11 = node_id(i + 1, j + 1)
            faces.append((n00, n10, n11))
            faces.append((n00, n11, n01))
    return np.asarray(nodes, dtype=float), np.asarray(faces, dtype=np.int64)


def _build_master_field(query_points: np.ndarray, *, quick: bool) -> DynamicNarrowBandSDF:
    resolution = 8 if quick else 18
    nodes, faces = _dense_plane_surface(resolution)
    return DynamicNarrowBandSDF.build_required_points(
        nodes,
        faces,
        query_points,
        spacing=0.09 if quick else 0.05,
        band_radius=0.22,
        origin=np.asarray([-0.12, -0.12, -0.16], dtype=float),
        shape=(14, 14, 6) if quick else (27, 27, 9),
        cell_size=0.12,
    )


def _dense_samples(corners: np.ndarray, query_count: int) -> list[SurfaceSample]:
    side = int(np.ceil(np.sqrt(query_count)))
    samples: list[SurfaceSample] = []
    for row in range(side):
        eta = (row + 0.5) / side
        for col in range(side):
            if len(samples) >= query_count:
                return samples
            xi = (col + 0.5) / side
            weights = np.asarray(
                [
                    (1.0 - xi) * (1.0 - eta),
                    xi * (1.0 - eta),
                    xi * eta,
                    (1.0 - xi) * eta,
                ],
                dtype=float,
            )
            samples.append(
                SurfaceSample(
                    np.asarray(corners, dtype=np.int64),
                    weights,
                    np.asarray([0], dtype=np.int64),
                )
            )
    return samples


def _prescribed_state(model: TimingModel, regime: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_min = float(np.min(model.mesh.X[:, 2]))
    height = max(float(np.max(model.mesh.X[:, 2]) - z_min), 1.0e-30)
    u = np.zeros(model.body.n_dofs, dtype=float).reshape((-1, 3))
    zeta = (model.mesh.X[:, 2] - z_min) / height
    if regime == "static":
        u[:, 2] = -0.062 - 0.012 * zeta
    elif regime == "dynamic":
        u[:, 2] = -0.054 - 0.006 * zeta
    else:
        raise ValueError("regime must be static or dynamic")
    v = np.zeros(model.body.n_dofs, dtype=float)
    v[2::3] = -0.30 if regime == "dynamic" else 0.0
    return model.mesh.X + u, u.ravel(), v


def _assemble_fem(model: TimingModel, regime: str):
    start = perf_counter()
    K = assemble_stiffness_matrix(model.body).tocsr()
    M = assemble_mass_matrix(model.body, kind="lumped").tocsr() if regime == "dynamic" else None
    return K, M, perf_counter() - start


def _field_contact_force(
    x_current: np.ndarray,
    samples: list[SurfaceSample],
    sdf: DynamicNarrowBandSDF,
    *,
    stiffness: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    start = perf_counter()
    constraints = compute_field_contact_constraints(x_current, samples, sdf)
    force = np.zeros(x_current.shape[0] * 3, dtype=float)
    gaps = np.asarray([constraint.g for constraint in constraints], dtype=float)
    active = 0
    for constraint in constraints:
        penetration = max(-float(constraint.g), 0.0)
        if penetration <= 0.0:
            continue
        active += 1
        nodal = float(stiffness) * penetration * np.asarray(constraint.normal, dtype=float)
        for node, weight in zip(constraint.slave_node_ids, constraint.slave_weights, strict=True):
            force[3 * int(node) : 3 * int(node) + 3] += float(weight) * nodal
    return force, gaps, perf_counter() - start, active


def _projection_contact_force(
    x_current: np.ndarray,
    samples: list[SurfaceSample],
    master_nodes: np.ndarray,
    master_faces: np.ndarray,
    broad_phase: UniformTriangleAABBHash,
    *,
    stiffness: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    start = perf_counter()
    force = np.zeros(x_current.shape[0] * 3, dtype=float)
    gaps = np.zeros(len(samples), dtype=float)
    active = 0
    all_faces = np.arange(master_faces.shape[0], dtype=np.int64)
    for row, sample in enumerate(samples):
        point = sample.point(x_current)
        candidates = broad_phase.query_point(point)
        if candidates.size == 0:
            candidates = all_faces
        result = surface_projection_distance_kernel(point, master_nodes, master_faces, candidates)
        gaps[row] = float(result.g)
        penetration = max(-float(result.g), 0.0)
        if penetration <= 0.0:
            continue
        active += 1
        nodal = float(stiffness) * penetration * np.asarray(result.n, dtype=float)
        for node, weight in zip(sample.node_ids, sample.weights, strict=True):
            force[3 * int(node) : 3 * int(node) + 3] += float(weight) * nodal
    return force, gaps, perf_counter() - start, active


def _solve_or_integrate(
    model: TimingModel,
    regime: str,
    K,
    M,
    u: np.ndarray,
    v: np.ndarray,
    force: np.ndarray,
) -> tuple[float, float]:
    start = perf_counter()
    if regime == "dynamic":
        if M is None:
            raise ValueError("dynamic timing requires a mass matrix")
        mass_diag = np.maximum(M.diagonal(), 1.0e-12)
        acceleration = (force - K @ u) / mass_diag
        v_next = v + 1.0e-3 * acceleration
        u_next = u + 1.0e-3 * v_next
        state_norm = float(np.linalg.norm(u_next) + np.linalg.norm(v_next))
        return perf_counter() - start, state_norm

    fixed_nodes = np.nonzero(np.isclose(model.mesh.X[:, 2], float(np.max(model.mesh.X[:, 2]))))[0]
    fixed = np.asarray([3 * int(node) + comp for node in fixed_nodes for comp in range(3)], dtype=np.int64)
    all_dofs = np.arange(model.body.n_dofs, dtype=np.int64)
    free = np.setdiff1d(all_dofs, fixed)
    rhs = force[free] - K[free][:, fixed] @ np.zeros(fixed.size, dtype=float)
    Kff = K[free][:, free] + 1.0e-10 * eye(free.size, format="csr")
    solution = spsolve(Kff, rhs)
    return perf_counter() - start, float(np.linalg.norm(solution))


def _relative_norm_error(value: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    if denom <= 1.0e-30:
        return float(np.linalg.norm(value - reference))
    return float(np.linalg.norm(value - reference) / denom)


def _case_query_counts(quick: bool) -> list[int]:
    return [128, 4096] if quick else [256, 1024, 4096, 8192]


def _run_case(
    *,
    element_type: str,
    regime: str,
    query_count: int,
    master_nodes: np.ndarray,
    master_faces: np.ndarray,
    broad_phase: UniformTriangleAABBHash,
    projection_broadphase_seconds: float,
    quick: bool,
) -> Row:
    model = _timing_model(element_type, quick=quick)
    samples = _dense_samples(model.bottom_corners, query_count)
    x_current, u, v = _prescribed_state(model, regime)
    query_points = np.asarray([sample.point(x_current) for sample in samples], dtype=float)
    stiffness = 150.0

    K, M, fem_seconds = _assemble_fem(model, regime)
    sdf = _build_master_field(query_points, quick=quick)
    field_force, field_gaps, field_contact_seconds, active_field = _field_contact_force(
        x_current,
        samples,
        sdf,
        stiffness=stiffness,
    )
    field_solve_seconds, field_state_norm = _solve_or_integrate(model, regime, K, M, u, v, field_force)

    projection_force, projection_gaps, projection_contact_seconds, active_projection = _projection_contact_force(
        x_current,
        samples,
        master_nodes,
        master_faces,
        broad_phase,
        stiffness=stiffness,
    )
    projection_solve_seconds, projection_state_norm = _solve_or_integrate(
        model,
        regime,
        K,
        M,
        u,
        v,
        projection_force,
    )

    field_total = fem_seconds + sdf.field_update_cost + field_contact_seconds + field_solve_seconds
    projection_total = fem_seconds + projection_broadphase_seconds + projection_contact_seconds + projection_solve_seconds
    gap_error = float(np.max(np.abs(field_gaps - projection_gaps))) if field_gaps.size else 0.0
    force_error = _relative_norm_error(field_force, projection_force)
    field_per_query = field_contact_seconds / max(len(samples), 1)
    projection_per_query = projection_contact_seconds / max(len(samples), 1)
    denom = projection_per_query - field_per_query
    q_star = float("inf") if denom <= 0.0 else max(float(sdf.field_update_cost) / denom, 0.0)
    speedup = projection_total / max(field_total, 1.0e-30)
    accuracy_ok = gap_error < 1.0e-10 and force_error < 1.0e-10
    faster = field_total < projection_total
    return {
        "case_id": f"{regime}_linear_{element_type}_q{len(samples)}",
        "regime": regime,
        "element_type": element_type,
        "query_count": len(samples),
        "master_triangle_count": master_faces.shape[0],
        "field_path": "DynamicNarrowBandSDF+field_contact",
        "field_population_mode": "required_points_exact",
        "required_grid_node_count": sdf.grid.metadata.get("required_node_count", ""),
        "valid_grid_node_count": sdf.stats.valid_node_count,
        "grid_node_count": sdf.stats.grid_node_count,
        "reference_path": "spatial_hash_projection_timing_reference",
        "fem_assembly_seconds": fem_seconds,
        "field_update_seconds": sdf.field_update_cost,
        "projection_broadphase_build_seconds": projection_broadphase_seconds,
        "field_contact_seconds": field_contact_seconds,
        "projection_contact_seconds": projection_contact_seconds,
        "field_solve_or_integrate_seconds": field_solve_seconds,
        "projection_solve_or_integrate_seconds": projection_solve_seconds,
        "field_total_step_seconds": field_total,
        "projection_total_step_seconds": projection_total,
        "field_seconds_per_query": field_per_query,
        "projection_seconds_per_query": projection_per_query,
        "query_crossover_q_star": q_star,
        "step_speedup_projection_over_field": speedup,
        "field_faster_total_step": str(faster).lower(),
        "max_gap_abs_error": gap_error,
        "force_l2_rel_error": force_error,
        "active_field": active_field,
        "active_projection": active_projection,
        "field_state_norm": field_state_norm,
        "projection_state_norm": projection_state_norm,
        "state_norm_abs_error": abs(field_state_norm - projection_state_norm),
        "accuracy_status": "passed" if accuracy_ok else "failed",
        "acceleration_status": "passed" if faster else "below_crossover",
    }


def _write_summary(path: Path, step_rows: list[Row], outputs: dict[str, Path]) -> None:
    accelerated = sum(1 for row in step_rows if row["field_faster_total_step"] == "true")
    accurate = sum(1 for row in step_rows if row["accuracy_status"] == "passed")
    best_speedup = max(float(row["step_speedup_projection_over_field"]) for row in step_rows)
    min_q_star = min(float(row["query_crossover_q_star"]) for row in step_rows if np.isfinite(float(row["query_crossover_q_star"])))
    lines = [
        "# True Field Solver-Level Timing",
        "",
        "This validation measures full contact-step cost for native linear TET4/HEX8 static and dynamic cases.",
        "The projection path is used only as a timing reference; the paper method path is exact required-node `DynamicNarrowBandSDF+field_contact`.",
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
            "## Claim Gates",
            "",
            f"- Accuracy rows passed: `{accurate}/{len(step_rows)}`.",
            f"- Total-step acceleration rows passed: `{accelerated}/{len(step_rows)}`.",
            f"- Best total-step speedup projection/field: `{best_speedup:.6f}`.",
            f"- Minimum measured query crossover Q*: `{min_q_star:.3f}`.",
            "",
            "## Scope",
            "",
            "- Supports contact-dominated linear TET4/HEX8 static/dynamic timing claims after the measured crossover.",
            "- Does not support nonlinear FEM, friction, self-contact, GPU, barrier contact, neural/POD SDF, or production BVH claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(out_dir: Path, *, quick: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    master_resolution = 8 if quick else 18
    master_nodes, master_faces = _dense_plane_surface(master_resolution)
    broad_phase_start = perf_counter()
    broad_phase = UniformTriangleAABBHash.from_surface(
        master_nodes,
        master_faces,
        delta_safe=0.25,
        cell_size=0.08 if quick else 0.05,
    )
    projection_broadphase_seconds = perf_counter() - broad_phase_start
    step_rows: list[Row] = []
    for element_type in ("tet4", "hex8"):
        for regime in ("static", "dynamic"):
            for query_count in _case_query_counts(quick):
                step_rows.append(
                    _run_case(
                        element_type=element_type,
                        regime=regime,
                        query_count=query_count,
                        master_nodes=master_nodes,
                        master_faces=master_faces,
                        broad_phase=broad_phase,
                        projection_broadphase_seconds=projection_broadphase_seconds,
                        quick=quick,
                    )
                )

    crossover_rows: list[Row] = []
    for row in step_rows:
        crossover_rows.append(
            {
                "case_id": row["case_id"],
                "regime": row["regime"],
                "element_type": row["element_type"],
                "query_count": row["query_count"],
                "field_update_seconds": row["field_update_seconds"],
                "field_seconds_per_query": row["field_seconds_per_query"],
                "projection_seconds_per_query": row["projection_seconds_per_query"],
                "query_crossover_q_star": row["query_crossover_q_star"],
                "field_faster_total_step": row["field_faster_total_step"],
                "step_speedup_projection_over_field": row["step_speedup_projection_over_field"],
            }
        )

    outputs = {
        "step_timing": out_dir / "solver_step_timing.csv",
        "crossover": out_dir / "solver_crossover.csv",
        "summary": out_dir / "true_field_solver_timing_summary.md",
    }
    _write_csv(outputs["step_timing"], step_rows)
    _write_csv(outputs["crossover"], crossover_rows)
    outputs.update(_plot_outputs(out_dir, step_rows))
    _write_summary(outputs["summary"], step_rows, outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "true_field_solver_timing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(args.out_dir, quick=bool(args.quick))
    print("True field solver-level timing complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
