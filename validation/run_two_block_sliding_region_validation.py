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
from sfc.contact.validation_gates import contact_total_priority_gate_metrics  # noqa: E402
from sfc.fem.calculix_aligned import ContactSample  # noqa: E402
from sfc.sdf.material_sdf import MaterialSDF  # noqa: E402
from validation.run_calculix_deformable_sdf_contact_validation import build_two_block_model  # noqa: E402
from validation.run_flexible_gear_implicit_lagrangian_sdf_comparison import (  # noqa: E402
    _aggregate_contact_sample_arrays,
    _assemble_contact_arrays_force_only,
    _contact_active_region_continuity_metrics_from_arrays,
    _contact_region_path_tracking_metrics_from_arrays,
    _contact_region_integral_metrics_from_arrays,
    _constraint_region_contact_law_metrics_from_arrays,
    _secondary_region_contact_node_diagnostics_from_arrays,
)
from validation.run_flat_punch_region_penetration_validation import run_validation as run_flat_punch_validation  # noqa: E402

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


def two_block_sliding_region_gate_metrics(
    rows: list[Row],
    *,
    flat_punch_summary: Row | None,
    min_cache_hit_fraction: float = 0.999,
    min_cache_match_fraction: float = 0.999,
    min_active_region_jaccard: float = 0.999,
    max_master_face_switch_fraction: float = 0.60,
    max_master_barycentric_drift: float = 0.75,
) -> Row:
    """Gate the second validation layer before cropped/full gear escalation.

    This gate depends on the flat-punch RF/penetration calibration, then checks
    only region-level sliding quantities: total normal force, virtual
    work/energy consistency, active area/region count, accepted-state path
    tracking, and active-region continuity.  Nodal CPRESS/COPEN fields remain
    deferred to later visualization layers.
    """

    flat_gate = int(float((flat_punch_summary or {}).get("flat_punch_rf_penetration_gate_passed", 0))) == 1
    total_gate = contact_total_priority_gate_metrics(rows)
    total_gate_passed = int(total_gate["contact_total_gate_passed"]) == 1
    active_rows = [
        row
        for row in rows
        if int(float(row.get("active_contact_region_count", row.get("active_contact_samples", 0)))) > 0
        or float(row.get("contact_active_area", 0.0)) > 0.0
        or float(row.get("normal_force", 0.0)) > 0.0
    ]
    no_nodal_clouds = bool(rows) and all(int(float(row.get("nodal_cpress_deferred", 0))) == 1 for row in rows)
    active_totals_present = bool(active_rows) and all(
        float(row.get("normal_force", 0.0)) >= 0.0
        and float(row.get("contact_active_area", 0.0)) > 0.0
        and int(float(row.get("active_contact_region_count", 0))) > 0
        for row in active_rows
    )
    max_force_mismatch = 0.0
    max_work_mismatch = 0.0
    law_rows_ok = True
    for row in active_rows:
        normal_force = float(row.get("normal_force", 0.0))
        region_force = float(row.get("contact_region_normal_force", normal_force))
        force_scale = max(abs(normal_force), abs(region_force), 1.0)
        max_force_mismatch = max(max_force_mismatch, abs(normal_force - region_force) / force_scale)
        work = float(row.get("contact_virtual_work", 0.0))
        region_work = float(row.get("contact_region_virtual_work", work))
        work_scale = max(abs(work), abs(region_work), 1.0)
        max_work_mismatch = max(max_work_mismatch, abs(work - region_work) / work_scale)
        law_rows_ok = law_rows_ok and str(row.get("contact_constraint_law_source", "")) == "slave_node_region_constraint"
        law_rows_ok = (
            law_rows_ok
            and str(row.get("contact_constraint_open_closed_source", "")) == "signed_area_average_region_gap"
            and str(row.get("contact_constraint_normal_source", "")) == "area_average_region_normal"
            and str(row.get("contact_constraint_force_distribution", "")) == "region_area_slave_shape_master_payload"
            and int(float(row.get("contact_constraint_independent_quadrature_penalty_disabled", 0))) == 1
        )
    tracking_rows = rows[1:] if len(rows) > 1 else []
    cache_hit_values = [float(row.get("contact_path_cache_hit_fraction", 0.0)) for row in tracking_rows]
    cache_match_values = [float(row.get("contact_path_cache_match_fraction", 0.0)) for row in tracking_rows]
    switch_values = [float(row.get("contact_master_face_switch_fraction", 0.0)) for row in tracking_rows]
    barycentric_values = [float(row.get("contact_master_barycentric_drift_max", 0.0)) for row in tracking_rows]
    active_jaccard_values = [float(row.get("contact_active_region_jaccard", 0.0)) for row in tracking_rows]
    cache_hit_min = min(cache_hit_values) if cache_hit_values else 1.0
    cache_match_min = min(cache_match_values) if cache_match_values else 1.0
    master_face_switch_fraction_max = max(switch_values) if switch_values else 0.0
    master_barycentric_drift_max = max(barycentric_values) if barycentric_values else 0.0
    active_jaccard_min = min(active_jaccard_values) if active_jaccard_values else 1.0
    path_tracking_ok = (
        bool(rows)
        and cache_hit_min >= float(min_cache_hit_fraction)
        and cache_match_min >= float(min_cache_match_fraction)
        and active_jaccard_min >= float(min_active_region_jaccard)
        and master_face_switch_fraction_max <= float(max_master_face_switch_fraction)
        and master_barycentric_drift_max <= float(max_master_barycentric_drift)
    )
    active_region_continuity_ok = active_jaccard_min >= float(min_active_region_jaccard)
    force_consistency_ok = max_force_mismatch <= 1.0e-12
    work_consistency_ok = max_work_mismatch <= 1.0e-12
    gate_passed = (
        flat_gate
        and total_gate_passed
        and no_nodal_clouds
        and active_totals_present
        and law_rows_ok
        and path_tracking_ok
        and active_region_continuity_ok
        and force_consistency_ok
        and work_consistency_ok
    )
    return {
        **total_gate,
        "two_block_sliding_region_gate_passed": int(gate_passed),
        "comparison_stage": "two_block_sliding_totals_before_cropped_gear_or_clouds",
        "flat_punch_prerequisite_gate_passed": int(flat_gate),
        "two_block_active_row_count": int(len(active_rows)),
        "two_block_nodal_cpress_deferred": int(no_nodal_clouds),
        "two_block_active_totals_present": int(active_totals_present),
        "two_block_constraint_region_law_gate_passed": int(law_rows_ok),
        "two_block_path_tracking_gate_passed": int(path_tracking_ok),
        "two_block_active_region_continuity_gate_passed": int(active_region_continuity_ok),
        "two_block_force_consistency_gate_passed": int(force_consistency_ok),
        "two_block_virtual_work_consistency_gate_passed": int(work_consistency_ok),
        "two_block_force_relative_mismatch_max": float(max_force_mismatch),
        "two_block_virtual_work_relative_mismatch_max": float(max_work_mismatch),
        "two_block_path_cache_hit_fraction_min_after_first": float(cache_hit_min),
        "two_block_path_cache_match_fraction_min_after_first": float(cache_match_min),
        "two_block_active_region_jaccard_min": float(active_jaccard_min),
        "two_block_master_face_switch_fraction_max": float(master_face_switch_fraction_max),
        "two_block_master_barycentric_drift_max": float(master_barycentric_drift_max),
        "two_block_path_tracking_min_cache_hit_threshold": float(min_cache_hit_fraction),
        "two_block_path_tracking_min_cache_match_threshold": float(min_cache_match_fraction),
        "two_block_path_tracking_min_active_region_jaccard_threshold": float(min_active_region_jaccard),
        "two_block_path_tracking_max_face_switch_threshold": float(max_master_face_switch_fraction),
        "two_block_path_tracking_max_barycentric_drift_threshold": float(max_master_barycentric_drift),
        "two_block_ready_for_cropped_gear_patch": int(gate_passed),
    }


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
    min_cache_hit_fraction: float = 0.999,
    min_cache_match_fraction: float = 0.999,
    min_active_region_jaccard: float = 0.999,
    max_master_face_switch_fraction: float = 0.60,
    max_master_barycentric_drift: float = 0.75,
    run_flat_punch_prerequisite: bool = True,
) -> tuple[list[Row], Row]:
    """Run the sliding sequence and write region-level validation outputs."""

    if int(steps) < 2:
        raise ValueError("steps must be at least 2")
    if float(penetration) <= 0.0:
        raise ValueError("penetration must be positive for the sliding contact validation")
    if not (0.0 <= float(min_cache_hit_fraction) <= 1.0):
        raise ValueError("min_cache_hit_fraction must be in [0, 1]")
    if not (0.0 <= float(min_cache_match_fraction) <= 1.0):
        raise ValueError("min_cache_match_fraction must be in [0, 1]")
    if not (0.0 <= float(min_active_region_jaccard) <= 1.0):
        raise ValueError("min_active_region_jaccard must be in [0, 1]")
    if not (0.0 <= float(max_master_face_switch_fraction) <= 1.0):
        raise ValueError("max_master_face_switch_fraction must be in [0, 1]")
    if float(max_master_barycentric_drift) < 0.0:
        raise ValueError("max_master_barycentric_drift must be non-negative")
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_punch_summary: Row | None = None
    if bool(run_flat_punch_prerequisite):
        _flat_rows, flat_punch_summary = run_flat_punch_validation(
            out_dir=out_dir / "flat_punch_prerequisite",
            resolution=int(resolution),
            gap=float(gap),
            penetrations=(0.0, 0.5 * float(penetration), float(penetration)),
            pressure_stiffness=float(pressure_stiffness),
            quadrature=str(quadrature),
        )
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
        compiled_batch_projection=True,
        secondary_path_tracking=True,
    )
    previous_active_regions: tuple[int, ...] | None = None
    previous_path_region_ids: np.ndarray | None = None
    previous_master_face_ids: np.ndarray | None = None
    previous_master_barycentric: np.ndarray | None = None
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
        arrays = contact.secondary_normal_projection_sample_arrays(x)
        if arrays is None:
            samples = list(contact.samples(x))
            arrays = _contact_samples_to_arrays(samples)
        raw_sample_count = int(np.asarray(arrays.get("gaps", np.empty(0)), dtype=float).size)
        regions = _aggregate_contact_sample_arrays(arrays, "slave_node_region_constraint")
        if regions is None:
            raise RuntimeError("slave_node_region_constraint aggregation unexpectedly fell back")
        response = _assemble_contact_arrays_force_only(regions, model.nodes.shape[0], stiffness=float(pressure_stiffness))
        region_metrics = _contact_region_integral_metrics_from_arrays(regions, stiffness=float(pressure_stiffness))
        pressure_diagnostics = _secondary_region_contact_node_diagnostics_from_arrays(
            regions,
            model.nodes.shape[0],
            stiffness=float(pressure_stiffness),
        )
        law_metrics = _constraint_region_contact_law_metrics_from_arrays(
            regions,
            arrays,
            averaging_mode="slave_node_region_constraint",
        )
        (
            path_metrics,
            current_path_region_ids,
            current_master_face_ids,
            current_master_barycentric,
        ) = _contact_region_path_tracking_metrics_from_arrays(
            regions,
            previous_path_region_ids,
            previous_master_face_ids,
            previous_master_barycentric,
        )
        if current_path_region_ids is not None:
            previous_path_region_ids = current_path_region_ids
        if current_master_face_ids is not None:
            previous_master_face_ids = current_master_face_ids
        if current_master_barycentric is not None:
            previous_master_barycentric = current_master_barycentric
        committed_tracking = (
            contact.commit_secondary_tracking_from_sample_arrays(arrays)
            if hasattr(contact, "commit_secondary_tracking_from_sample_arrays")
            else 0
        )
        continuity, current_active_regions = _contact_active_region_continuity_metrics_from_arrays(
            regions,
            previous_active_regions,
        )
        previous_active_regions = current_active_regions
        row: Row = {
            "step": int(step),
            "comparison_stage": "two_block_sliding_totals_before_cropped_gear_or_clouds",
            "nodal_cpress_deferred": 1,
            "flat_punch_prerequisite_gate_passed": (
                "" if flat_punch_summary is None else int(flat_punch_summary["flat_punch_rf_penetration_gate_passed"])
            ),
            "time": float(alpha),
            "lateral_shift": float(shift),
            "normal_penetration": float(penetration),
            "raw_quadrature_samples": raw_sample_count,
            "constraint_regions": int(np.asarray(regions["gaps"]).size),
            "active_contact_samples": int(response.active_count),
            "normal_force": float(response.normal_force),
            "contact_energy": float(response.energy),
            "contact_virtual_work": float(2.0 * response.energy),
            "accepted_path_tracking_committed": int(committed_tracking),
            "min_region_gap": float(np.min(np.asarray(regions["gaps"], dtype=float))) if np.asarray(regions["gaps"]).size else 0.0,
        }
        row.update(region_metrics)
        if pressure_diagnostics is not None:
            row["contact_secondary_pressure_recovery_source"] = pressure_diagnostics["metrics"][
                "contact_secondary_pressure_recovery_source"
            ]
        row.update(law_metrics)
        row.update(path_metrics)
        row.update(continuity)
        rows.append(row)
    wall = time.perf_counter() - start
    totals_path = out_dir / "two_block_sliding_contact_totals.csv"
    continuity_path = out_dir / "two_block_sliding_region_continuity.csv"
    tracking_path = out_dir / "two_block_sliding_path_tracking.csv"
    gate_path = out_dir / "two_block_sliding_region_gate.csv"
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
    tracking_keys = [
        "step",
        "time",
        "lateral_shift",
        "contact_active_master_face_count",
        "contact_master_face_unique_count",
        "contact_master_face_tracking_comparable_count",
        "contact_master_face_switch_count",
        "contact_master_face_switch_fraction",
        "contact_path_cache_hit_count",
        "contact_path_cache_hit_fraction",
        "contact_path_cache_match_count",
        "contact_path_cache_match_fraction",
        "contact_master_barycentric_tracking_comparable_count",
        "contact_master_barycentric_drift_mean",
        "contact_master_barycentric_drift_max",
    ]
    _write_csv(tracking_path, [{key: row.get(key, "") for key in tracking_keys} for row in rows])
    tracking_rows = rows[1:] if len(rows) > 1 else rows
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
        "master_face_switch_fraction_max": (
            float(max(float(row["contact_master_face_switch_fraction"]) for row in tracking_rows))
            if tracking_rows
            else 0.0
        ),
        "path_cache_hit_fraction_min_after_first": (
            float(min(float(row["contact_path_cache_hit_fraction"]) for row in rows[1:]))
            if len(rows) > 1
            else 0.0
        ),
        "path_cache_match_fraction_min_after_first": (
            float(min(float(row["contact_path_cache_match_fraction"]) for row in rows[1:]))
            if len(rows) > 1
            else 0.0
        ),
        "master_barycentric_drift_max": (
            float(max(float(row["contact_master_barycentric_drift_max"]) for row in tracking_rows))
            if tracking_rows
            else 0.0
        ),
        "path_tracking_min_cache_hit_threshold": float(min_cache_hit_fraction),
        "path_tracking_min_cache_match_threshold": float(min_cache_match_fraction),
        "path_tracking_min_active_region_jaccard_threshold": float(min_active_region_jaccard),
        "path_tracking_max_face_switch_threshold": float(max_master_face_switch_fraction),
        "path_tracking_max_barycentric_drift_threshold": float(max_master_barycentric_drift),
        "analysis_wall_seconds": float(wall),
        "totals_csv": str(totals_path),
        "continuity_csv": str(continuity_path),
        "tracking_csv": str(tracking_path),
    }
    summary["path_tracking_gate_passed"] = int(
        float(summary["path_cache_hit_fraction_min_after_first"]) >= float(min_cache_hit_fraction)
        and float(summary["path_cache_match_fraction_min_after_first"]) >= float(min_cache_match_fraction)
        and float(summary["active_region_jaccard_min"]) >= float(min_active_region_jaccard)
        and float(summary["master_face_switch_fraction_max"]) <= float(max_master_face_switch_fraction)
        and float(summary["master_barycentric_drift_max"]) <= float(max_master_barycentric_drift)
    )
    for row in rows:
        row["path_tracking_min_cache_hit_threshold"] = float(min_cache_hit_fraction)
        row["path_tracking_min_cache_match_threshold"] = float(min_cache_match_fraction)
    gate = two_block_sliding_region_gate_metrics(
        rows,
        flat_punch_summary=flat_punch_summary,
        min_cache_hit_fraction=float(min_cache_hit_fraction),
        min_cache_match_fraction=float(min_cache_match_fraction),
        min_active_region_jaccard=float(min_active_region_jaccard),
        max_master_face_switch_fraction=float(max_master_face_switch_fraction),
        max_master_barycentric_drift=float(max_master_barycentric_drift),
    )
    _write_csv(gate_path, [gate])
    summary.update(gate)
    summary["gate_csv"] = str(gate_path)
    if flat_punch_summary is not None:
        summary["flat_punch_prerequisite_summary_csv"] = str(out_dir / "flat_punch_prerequisite" / "flat_punch_region_summary.csv")
        summary["flat_punch_prerequisite_gate_csv"] = str(out_dir / "flat_punch_prerequisite" / "flat_punch_rf_penetration_gate.csv")
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
                f"- max master-face switch fraction: {summary['master_face_switch_fraction_max']:.12e}",
                f"- min path-cache hit fraction after first frame: {summary['path_cache_hit_fraction_min_after_first']:.12e}",
                f"- min path-cache match fraction after first frame: {summary['path_cache_match_fraction_min_after_first']:.12e}",
                f"- max master-barycentric drift: {summary['master_barycentric_drift_max']:.12e}",
                f"- accepted-state path tracking gate: {'PASS' if int(summary['path_tracking_gate_passed']) else 'FAIL'}",
                f"- flat-punch prerequisite gate: {'PASS' if int(summary['flat_punch_prerequisite_gate_passed']) else 'FAIL'}",
                f"- two-block sliding region gate: {'PASS' if int(summary['two_block_sliding_region_gate_passed']) else 'FAIL'}",
                f"- nodal CPRESS/COPEN deferred: {'yes' if int(summary['two_block_nodal_cpress_deferred']) else 'no'}",
                f"- constraint-region law gate: {'PASS' if int(summary['two_block_constraint_region_law_gate_passed']) else 'FAIL'}",
                "",
                "Face switches are permitted when the smooth sliding path crosses the two-triangle split of a planar quad. "
                "The gate therefore requires accepted-state cache hit/match continuity and bounded switch fraction rather "
                "than requiring a fixed triangle id for all frames.",
                f"- totals CSV: `{totals_path.name}`",
                f"- continuity CSV: `{continuity_path.name}`",
                f"- path tracking CSV: `{tracking_path.name}`",
                f"- gate CSV: `{gate_path.name}`",
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
    parser.add_argument("--min-cache-hit-fraction", type=float, default=0.999)
    parser.add_argument("--min-cache-match-fraction", type=float, default=0.999)
    parser.add_argument("--min-active-region-jaccard", type=float, default=0.999)
    parser.add_argument("--max-master-face-switch-fraction", type=float, default=0.60)
    parser.add_argument("--max-master-barycentric-drift", type=float, default=0.75)
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
        min_cache_hit_fraction=float(args.min_cache_hit_fraction),
        min_cache_match_fraction=float(args.min_cache_match_fraction),
        min_active_region_jaccard=float(args.min_active_region_jaccard),
        max_master_face_switch_fraction=float(args.max_master_face_switch_fraction),
        max_master_barycentric_drift=float(args.max_master_barycentric_drift),
    )
    print((args.out_dir / "two_block_sliding_region_summary.md").read_text(encoding="utf-8"))
    return 0 if rows and summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
