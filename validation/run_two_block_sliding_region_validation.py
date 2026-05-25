"""Two-flexible-block sliding validation for region-level contact totals.

This runner is an early validation layer before cropped/full gear contact.  It
does not compare nodal CPRESS clouds.  Instead it exercises the Lagrangian-SDF
surface contact path on a two-deformable-block sliding sequence and records the
quantities that must be aligned first: total normal force, contact virtual
work, contact energy, active area, active region count, and active-region
continuity.

Abaqus is not used by this script.  External solver comparisons can consume the
same CSV columns later.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact.lagrangian_surface_contact import LagrangianSDFSurfaceContactGeometry  # noqa: E402
from sfc.fem.calculix_aligned import ContactSample  # noqa: E402
from sfc.sdf.material_sdf import MaterialSDF  # noqa: E402
from validation.run_calculix_deformable_sdf_contact_validation import build_two_block_model  # noqa: E402
from validation.run_flexible_gear_implicit_lagrangian_sdf_comparison import (  # noqa: E402
    _aggregate_contact_sample_arrays,
    _assemble_contact_arrays_force_only,
    _contact_active_region_continuity_metrics_from_arrays,
    _contact_region_integral_metrics_from_arrays,
)

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "two_block_sliding_region_validation"


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _contact_samples_to_arrays(samples: list[ContactSample]) -> dict[str, np.ndarray]:
    if not samples:
        return {
            "sample_node_ids": np.empty((0, 3), dtype=np.int64),
            "sample_weights": np.empty((0, 3), dtype=float),
            "gaps": np.empty(0, dtype=float),
            "normals": np.empty((0, 3), dtype=float),
            "areas": np.empty(0, dtype=float),
            "master_node_ids": np.empty((0, 3), dtype=np.int64),
            "master_weights": np.empty((0, 3), dtype=float),
        }
    slave_width = max(int(np.asarray(sample.node_ids).size) for sample in samples)
    master_width = max(
        0 if sample.master_node_ids is None else int(np.asarray(sample.master_node_ids).size)
        for sample in samples
    )
    sample_node_ids = np.zeros((len(samples), slave_width), dtype=np.int64)
    sample_weights = np.zeros((len(samples), slave_width), dtype=float)
    master_node_ids = np.zeros((len(samples), master_width), dtype=np.int64)
    master_weights = np.zeros((len(samples), master_width), dtype=float)
    gaps = np.zeros(len(samples), dtype=float)
    normals = np.zeros((len(samples), 3), dtype=float)
    areas = np.zeros(len(samples), dtype=float)
    for idx, sample in enumerate(samples):
        nodes = np.asarray(sample.node_ids, dtype=np.int64).reshape(-1)
        weights = np.asarray(sample.shape_weights, dtype=float).reshape(-1)
        sample_node_ids[idx, : nodes.size] = nodes
        sample_weights[idx, : weights.size] = weights
        if sample.master_node_ids is not None and sample.master_shape_weights is not None:
            m_nodes = np.asarray(sample.master_node_ids, dtype=np.int64).reshape(-1)
            m_weights = np.asarray(sample.master_shape_weights, dtype=float).reshape(-1)
            master_node_ids[idx, : m_nodes.size] = m_nodes
            master_weights[idx, : m_weights.size] = m_weights
        gaps[idx] = float(sample.gap)
        normal = np.asarray(sample.normal, dtype=float).reshape(3)
        normal_norm = max(float(np.linalg.norm(normal)), 1.0e-30)
        normals[idx] = normal / normal_norm
        areas[idx] = float(sample.area)
    return {
        "sample_node_ids": sample_node_ids,
        "sample_weights": sample_weights,
        "gaps": gaps,
        "normals": normals,
        "areas": areas,
        "master_node_ids": master_node_ids,
        "master_weights": master_weights,
    }


def _sliding_positions(reference_nodes: np.ndarray, *, lower_node_count: int, approach: float, lateral_shift: float) -> np.ndarray:
    x = np.asarray(reference_nodes, dtype=float).copy()
    x[int(lower_node_count) :, 0] += float(lateral_shift)
    x[int(lower_node_count) :, 2] -= float(approach)
    return x


def run_validation(
    *,
    out_dir: Path,
    resolution: int = 2,
    steps: int = 9,
    gap: float = 0.035,
    penetration: float = 0.01,
    lateral_shift: float = 0.20,
    pressure_stiffness: float = 6.0e4,
    quadrature: str = "tri3",
) -> tuple[list[Row], Row]:
    """Run the sliding sequence and write region-level validation outputs."""

    if int(steps) < 2:
        raise ValueError("steps must be at least 2")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_two_block_model(resolution=int(resolution), approach=float(gap) + float(penetration), gap=float(gap))
    lower_nodes = np.asarray(model.nodes[: model.lower_node_count], dtype=float)
    master = MaterialSDF.from_triangle_surface(lower_nodes, np.asarray(model.lower_top_faces, dtype=np.int64))
    slave_faces = np.asarray(model.upper_bottom_faces, dtype=np.int64) - int(model.lower_node_count)
    contact = LagrangianSDFSurfaceContactGeometry(
        slave_faces,
        master,
        lower_nodes,
        pressure_stiffness=float(pressure_stiffness),
        slave_node_offset=int(model.lower_node_count),
        master_node_offset=0,
        quadrature=str(quadrature),
        search_radius=float(gap) + abs(float(penetration)) + abs(float(lateral_shift)) + 0.25,
        compiled_batch_projection=False,
        secondary_path_tracking=True,
    )
    previous_active_regions: tuple[int, ...] | None = None
    rows: list[Row] = []
    start = time.perf_counter()
    for step in range(int(steps)):
        alpha = float(step) / float(int(steps) - 1)
        shift = alpha * float(lateral_shift)
        x = _sliding_positions(
            model.nodes,
            lower_node_count=int(model.lower_node_count),
            approach=float(gap) + float(penetration),
            lateral_shift=shift,
        )
        samples = list(contact.samples(x))
        arrays = _contact_samples_to_arrays(samples)
        regions = _aggregate_contact_sample_arrays(arrays, "slave_node_region_constraint")
        if regions is None:
            raise RuntimeError("slave_node_region_constraint aggregation unexpectedly fell back")
        response = _assemble_contact_arrays_force_only(regions, model.nodes.shape[0], stiffness=float(pressure_stiffness))
        region_metrics = _contact_region_integral_metrics_from_arrays(regions, stiffness=float(pressure_stiffness))
        continuity, current_active_regions = _contact_active_region_continuity_metrics_from_arrays(
            regions,
            previous_active_regions,
        )
        previous_active_regions = current_active_regions
        row: Row = {
            "step": int(step),
            "time": float(alpha),
            "lateral_shift": float(shift),
            "normal_penetration": float(penetration),
            "raw_quadrature_samples": int(len(samples)),
            "constraint_regions": int(np.asarray(regions["gaps"]).size),
            "active_contact_samples": int(response.active_count),
            "normal_force": float(response.normal_force),
            "contact_energy": float(response.energy),
            "contact_virtual_work": float(2.0 * response.energy),
            "min_region_gap": float(np.min(np.asarray(regions["gaps"], dtype=float))) if np.asarray(regions["gaps"]).size else 0.0,
        }
        row.update(region_metrics)
        row.update(continuity)
        rows.append(row)
    wall = time.perf_counter() - start
    totals_path = out_dir / "two_block_sliding_contact_totals.csv"
    continuity_path = out_dir / "two_block_sliding_region_continuity.csv"
    summary_path = out_dir / "two_block_sliding_region_summary.md"
    _write_csv(totals_path, rows)
    continuity_keys = [
        "step",
        "time",
        "lateral_shift",
        "active_contact_region_count",
        "contact_active_area",
        "contact_active_region_continuity_previous_count",
        "contact_active_region_continuity_current_count",
        "contact_active_region_continuity_intersection_count",
        "contact_active_region_new_count",
        "contact_active_region_dropped_count",
        "contact_active_region_persistence_fraction",
        "contact_active_region_jaccard",
    ]
    _write_csv(continuity_path, [{key: row.get(key, "") for key in continuity_keys} for row in rows])
    summary: Row = {
        "status": "completed",
        "steps": int(steps),
        "resolution": int(resolution),
        "raw_quadrature_samples_final": int(rows[-1]["raw_quadrature_samples"]) if rows else 0,
        "constraint_regions_final": int(rows[-1]["constraint_regions"]) if rows else 0,
        "active_regions_final": int(rows[-1]["active_contact_region_count"]) if rows else 0,
        "normal_force_final": float(rows[-1]["normal_force"]) if rows else 0.0,
        "contact_energy_final": float(rows[-1]["contact_energy"]) if rows else 0.0,
        "active_region_jaccard_min": float(min(float(row["contact_active_region_jaccard"]) for row in rows[1:])) if len(rows) > 1 else 1.0,
        "analysis_wall_seconds": float(wall),
        "totals_csv": str(totals_path),
        "continuity_csv": str(continuity_path),
    }
    summary_path.write_text(
        "\n".join(
            [
                "# Two-Block Sliding Region Validation",
                "",
                "This validation reports region-level contact totals before nodal CPRESS/COPEN cloud comparisons.",
                "",
                f"- steps: {summary['steps']}",
                f"- resolution: {summary['resolution']}",
                f"- final normal force: {summary['normal_force_final']:.12e}",
                f"- final contact energy: {summary['contact_energy_final']:.12e}",
                f"- minimum active-region Jaccard after first frame: {summary['active_region_jaccard_min']:.12e}",
                f"- totals CSV: `{totals_path.name}`",
                f"- continuity CSV: `{continuity_path.name}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "two_block_sliding_region_summary.csv", [summary])
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--gap", type=float, default=0.035)
    parser.add_argument("--penetration", type=float, default=0.01)
    parser.add_argument("--lateral-shift", type=float, default=0.20)
    parser.add_argument("--pressure-stiffness", type=float, default=6.0e4)
    parser.add_argument("--quick", action="store_true", help="Use a smaller deterministic smoke case.")
    args = parser.parse_args(argv)
    steps = 5 if bool(args.quick) else int(args.steps)
    resolution = 1 if bool(args.quick) else int(args.resolution)
    rows, summary = run_validation(
        out_dir=args.out_dir,
        resolution=resolution,
        steps=steps,
        gap=float(args.gap),
        penetration=float(args.penetration),
        lateral_shift=float(args.lateral_shift),
        pressure_stiffness=float(args.pressure_stiffness),
    )
    print((args.out_dir / "two_block_sliding_region_summary.md").read_text(encoding="utf-8"))
    return 0 if rows and summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
