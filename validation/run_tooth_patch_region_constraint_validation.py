"""Tilted tooth-patch validation for secondary constraint-region contact.

This runner is the next evidence layer after flat punch and two-block sliding.
It uses a non-axis-aligned Q4 tooth-like patch, but it does not compare nodal
CPRESS/COPEN clouds yet.  The gate checks that Q4 master closest-feature
payloads can be aggregated into secondary-node constraint regions and that the
region-level totals, path tracking, and active-region continuity remain stable
under smooth tangential sliding.
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

from sfc.contact.lagrangian_surface_contact import LagrangianSDFQ4MasterSurfaceContactGeometry  # noqa: E402
from sfc.contact.validation_gates import contact_total_priority_gate_metrics  # noqa: E402
from validation.run_abaqus_lagrangian_sdf_tooth_patch_hard_contact import (  # noqa: E402
    ToothPatchHardContactCase,
    _pressure_enforcement_stiffness,
    build_case,
    make_geometry,
    tooth_patch_basis,
)
from validation.run_flexible_gear_implicit_lagrangian_sdf_comparison import (  # noqa: E402
    _aggregate_contact_sample_arrays,
    _assemble_contact_arrays_force_only,
    _constraint_region_contact_law_metrics_from_arrays,
    _contact_active_region_continuity_metrics_from_arrays,
    _contact_region_path_tracking_metrics_from_arrays,
    _contact_region_integral_metrics_from_arrays,
    _promote_secondary_contact_diagnostics_to_legacy,
    _secondary_region_contact_node_diagnostics_from_arrays,
)
from validation.run_two_block_sliding_region_validation import run_validation as run_two_block_sliding_validation  # noqa: E402

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "tooth_patch_region_constraint_validation"


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


def _current_nodes(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    lower_node_count: int,
    normal: np.ndarray,
    tangent: np.ndarray,
    closure: float,
    lateral_shift: float,
) -> np.ndarray:
    x = np.vstack((lower, upper)).astype(float, copy=True)
    x[int(lower_node_count) :, :] += -float(closure) * np.asarray(normal, dtype=float).reshape(1, 3)
    x[int(lower_node_count) :, :] += float(lateral_shift) * np.asarray(tangent, dtype=float).reshape(1, 3)
    return x


def tooth_patch_region_gate_metrics(rows: list[Row], *, two_block_summary: Row | None) -> Row:
    """Gate the tilted Q4 patch before cropped/full gear escalation."""

    prerequisite = int(float((two_block_summary or {}).get("two_block_sliding_region_gate_passed", 0))) == 1
    total_gate = contact_total_priority_gate_metrics(rows)
    total_gate_passed = int(total_gate["contact_total_gate_passed"]) == 1
    active_rows = [
        row
        for row in rows
        if int(float(row.get("active_contact_region_count", 0))) > 0
        or float(row.get("contact_active_area", 0.0)) > 0.0
        or float(row.get("normal_force", 0.0)) > 0.0
    ]
    no_nodal_clouds = bool(rows) and all(int(float(row.get("nodal_cpress_deferred", 0))) == 1 for row in rows)
    law_ok = True
    q4_payload_ok = True
    max_force_mismatch = 0.0
    max_work_mismatch = 0.0
    for row in active_rows:
        law_ok = (
            law_ok
            and str(row.get("contact_constraint_law_source", "")) == "slave_node_region_constraint"
            and str(row.get("contact_constraint_open_closed_source", "")) == "signed_area_average_region_gap"
            and str(row.get("contact_constraint_normal_source", "")) == "area_average_region_normal"
            and str(row.get("contact_constraint_force_distribution", "")) == "region_area_slave_shape_master_payload"
            and int(float(row.get("contact_constraint_independent_quadrature_penalty_disabled", 0))) == 1
        )
        q4_payload_ok = q4_payload_ok and int(float(row.get("q4_master_payload_width", 0))) == 4
        normal_force = float(row.get("normal_force", 0.0))
        region_force = float(row.get("contact_region_normal_force", normal_force))
        force_scale = max(abs(normal_force), abs(region_force), 1.0)
        max_force_mismatch = max(max_force_mismatch, abs(normal_force - region_force) / force_scale)
        work = float(row.get("contact_virtual_work", 0.0))
        region_work = float(row.get("contact_region_virtual_work", work))
        work_scale = max(abs(work), abs(region_work), 1.0)
        max_work_mismatch = max(max_work_mismatch, abs(work - region_work) / work_scale)
    tracking_rows = rows[1:] if len(rows) > 1 else []
    cache_hit_min = min((float(row.get("contact_path_cache_hit_fraction", 0.0)) for row in tracking_rows), default=1.0)
    cache_match_min = min((float(row.get("contact_path_cache_match_fraction", 0.0)) for row in tracking_rows), default=1.0)
    face_switch_max = max((float(row.get("contact_master_face_switch_fraction", 0.0)) for row in tracking_rows), default=0.0)
    bary_drift_max = max((float(row.get("contact_master_barycentric_drift_max", 0.0)) for row in tracking_rows), default=0.0)
    active_jaccard_min = min((float(row.get("contact_active_region_jaccard", 0.0)) for row in tracking_rows), default=1.0)
    path_ok = (
        bool(rows)
        and cache_hit_min >= 0.999
        and cache_match_min >= 0.999
        and face_switch_max <= 0.0
        and bary_drift_max <= 0.20
        and active_jaccard_min >= 0.999
    )
    totals_ok = bool(active_rows) and max_force_mismatch <= 1.0e-12 and max_work_mismatch <= 1.0e-12
    gate_passed = prerequisite and total_gate_passed and no_nodal_clouds and law_ok and q4_payload_ok and path_ok and totals_ok
    return {
        **total_gate,
        "tooth_patch_region_gate_passed": int(gate_passed),
        "comparison_stage": "tooth_patch_region_totals_before_cropped_gear_clouds",
        "two_block_sliding_prerequisite_gate_passed": int(prerequisite),
        "tooth_patch_active_row_count": int(len(active_rows)),
        "tooth_patch_nodal_cpress_deferred": int(no_nodal_clouds),
        "tooth_patch_constraint_region_law_gate_passed": int(law_ok),
        "tooth_patch_q4_master_payload_gate_passed": int(q4_payload_ok),
        "tooth_patch_path_tracking_gate_passed": int(path_ok),
        "tooth_patch_force_consistency_gate_passed": int(max_force_mismatch <= 1.0e-12),
        "tooth_patch_virtual_work_consistency_gate_passed": int(max_work_mismatch <= 1.0e-12),
        "tooth_patch_force_relative_mismatch_max": float(max_force_mismatch),
        "tooth_patch_virtual_work_relative_mismatch_max": float(max_work_mismatch),
        "tooth_patch_path_cache_hit_fraction_min_after_first": float(cache_hit_min),
        "tooth_patch_path_cache_match_fraction_min_after_first": float(cache_match_min),
        "tooth_patch_master_face_switch_fraction_max": float(face_switch_max),
        "tooth_patch_master_weight_drift_max": float(bary_drift_max),
        "tooth_patch_active_region_jaccard_min": float(active_jaccard_min),
        "tooth_patch_ready_for_cropped_gear_patch": int(gate_passed),
    }


def run_validation(
    *,
    out_dir: Path,
    steps: int = 5,
    lateral_shift: float = 0.08,
    run_two_block_prerequisite: bool = True,
) -> tuple[list[Row], Row]:
    """Run a smooth tilted tooth-patch sliding sequence."""

    if int(steps) < 2:
        raise ValueError("steps must be at least 2")
    out_dir.mkdir(parents=True, exist_ok=True)
    two_block_summary: Row | None = None
    if bool(run_two_block_prerequisite):
        _rows, two_block_summary = run_two_block_sliding_validation(
            out_dir=out_dir / "two_block_prerequisite",
            resolution=1,
            steps=5,
            penetration=0.01,
            lateral_shift=0.10,
        )
    case: ToothPatchHardContactCase = build_case()
    lower, upper, _elements, normal = make_geometry(case)
    tangent, _t2, _n = tooth_patch_basis(case)
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1.0e-30)
    lower_top_quad = np.asarray([[4, 5, 6, 7]], dtype=np.int64)
    upper_bottom_quad = np.asarray([[0, 3, 2, 1]], dtype=np.int64)
    stiffness = _pressure_enforcement_stiffness(case)
    contact = LagrangianSDFQ4MasterSurfaceContactGeometry(
        lower_top_quad,
        upper_bottom_quad,
        upper,
        pressure_stiffness=stiffness,
        master_node_offset=lower.shape[0],
        quadrature_order=int(case.quadrature_order),
        search_radius=max(case.initial_gap, case.closure) + 0.5 * case.thickness,
        secondary_path_tracking=True,
    )
    previous_master_face_ids: np.ndarray | None = None
    previous_master_barycentric: np.ndarray | None = None
    previous_path_region_ids: np.ndarray | None = None
    previous_active_regions: tuple[int, ...] | None = None
    rows: list[Row] = []
    start = time.perf_counter()
    for step in range(int(steps)):
        alpha = float(step) / float(int(steps) - 1)
        x = _current_nodes(
            lower,
            upper,
            lower_node_count=lower.shape[0],
            normal=normal,
            tangent=tangent,
            closure=float(case.closure),
            lateral_shift=alpha * float(lateral_shift),
        )
        raw_arrays = contact.sample_arrays(x)
        regions = _aggregate_contact_sample_arrays(raw_arrays, "slave_node_region_constraint")
        if regions is None:
            raise RuntimeError("slave_node_region_constraint aggregation unexpectedly fell back")
        response = _assemble_contact_arrays_force_only(regions, x.shape[0], stiffness=stiffness)
        region_metrics = _contact_region_integral_metrics_from_arrays(regions, stiffness=stiffness)
        pressure_diagnostics = _secondary_region_contact_node_diagnostics_from_arrays(
            regions,
            x.shape[0],
            stiffness=stiffness,
        )
        law_metrics = _constraint_region_contact_law_metrics_from_arrays(
            regions,
            raw_arrays,
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
        previous_path_region_ids = current_path_region_ids if current_path_region_ids is not None else previous_path_region_ids
        previous_master_face_ids = current_master_face_ids if current_master_face_ids is not None else previous_master_face_ids
        previous_master_barycentric = (
            current_master_barycentric if current_master_barycentric is not None else previous_master_barycentric
        )
        committed_tracking = contact.commit_secondary_tracking_from_sample_arrays(raw_arrays)
        continuity, current_active_regions = _contact_active_region_continuity_metrics_from_arrays(
            regions,
            previous_active_regions,
        )
        previous_active_regions = current_active_regions
        row: Row = {
            "step": int(step),
            "comparison_stage": "tooth_patch_region_totals_before_cropped_gear_clouds",
            "nodal_cpress_deferred": 1,
            "time": float(alpha),
            "lateral_shift": float(alpha * float(lateral_shift)),
            "normal_closure": float(case.closure),
            "raw_quadrature_samples": int(np.asarray(raw_arrays["gaps"]).size),
            "constraint_regions": int(np.asarray(regions["gaps"]).size),
            "q4_master_payload_width": int(np.asarray(raw_arrays["master_weights"]).shape[1]),
            "active_contact_samples": int(response.active_count),
            "normal_force": float(response.normal_force),
            "contact_energy": float(response.energy),
            "contact_virtual_work": float(2.0 * response.energy),
            "accepted_path_tracking_committed": int(committed_tracking),
            "min_region_gap": (
                float(np.min(np.asarray(regions["gaps"], dtype=float))) if np.asarray(regions["gaps"]).size else 0.0
            ),
        }
        row.update(region_metrics)
        if pressure_diagnostics is not None:
            _promote_secondary_contact_diagnostics_to_legacy(pressure_diagnostics)
            metrics = pressure_diagnostics["metrics"]
            row["contact_secondary_pressure_recovery_source"] = metrics["contact_secondary_pressure_recovery_source"]
            row["contact_pressure_recovery_source"] = metrics["contact_pressure_recovery_source"]
            row["contact_legacy_pressure_alias_source"] = metrics["contact_legacy_pressure_alias_source"]
            row["contact_legacy_pressure_alias_matches_secondary"] = metrics[
                "contact_legacy_pressure_alias_matches_secondary"
            ]
        row.update(law_metrics)
        row.update(path_metrics)
        row.update(continuity)
        rows.append(row)
    wall = time.perf_counter() - start
    totals_path = out_dir / "tooth_patch_region_totals.csv"
    tracking_path = out_dir / "tooth_patch_region_path_tracking.csv"
    gate_path = out_dir / "tooth_patch_region_gate.csv"
    summary_csv = out_dir / "tooth_patch_region_summary.csv"
    summary_md = out_dir / "tooth_patch_region_summary.md"
    _write_csv(totals_path, rows)
    tracking_keys = [
        "step",
        "time",
        "lateral_shift",
        "contact_path_cache_hit_fraction",
        "contact_path_cache_match_fraction",
        "contact_master_face_switch_fraction",
        "contact_master_barycentric_drift_max",
        "contact_active_region_jaccard",
    ]
    _write_csv(tracking_path, [{key: row.get(key, "") for key in tracking_keys} for row in rows])
    gate = tooth_patch_region_gate_metrics(rows, two_block_summary=two_block_summary)
    _write_csv(gate_path, [gate])
    summary: Row = {
        "status": "completed",
        "steps": int(steps),
        "analysis_wall_seconds": float(wall),
        "totals_csv": str(totals_path),
        "tracking_csv": str(tracking_path),
        "gate_csv": str(gate_path),
        "normal_force_final": float(rows[-1]["normal_force"]) if rows else 0.0,
        "active_regions_final": int(rows[-1]["active_contact_region_count"]) if rows else 0,
    }
    summary.update(gate)
    if two_block_summary is not None:
        summary["two_block_prerequisite_summary_csv"] = str(out_dir / "two_block_prerequisite" / "two_block_sliding_region_summary.csv")
    _write_csv(summary_csv, [summary])
    summary_md.write_text(
        "\n".join(
            [
                "# Tooth-Patch Region Constraint Validation",
                "",
                "This validation checks a tilted Q4 tooth-like patch using secondary constraint-region totals before nodal CPRESS/COPEN clouds.",
                "",
                f"- steps: {summary['steps']}",
                f"- final normal force: {summary['normal_force_final']:.12e}",
                f"- active regions final: {summary['active_regions_final']}",
                f"- two-block prerequisite gate: {'PASS' if int(summary['two_block_sliding_prerequisite_gate_passed']) else 'FAIL'}",
                f"- tooth-patch region gate: {'PASS' if int(summary['tooth_patch_region_gate_passed']) else 'FAIL'}",
                f"- Q4 master payload gate: {'PASS' if int(summary['tooth_patch_q4_master_payload_gate_passed']) else 'FAIL'}",
                f"- path tracking gate: {'PASS' if int(summary['tooth_patch_path_tracking_gate_passed']) else 'FAIL'}",
                f"- totals CSV: `{totals_path.name}`",
                f"- tracking CSV: `{tracking_path.name}`",
                f"- gate CSV: `{gate_path.name}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lateral-shift", type=float, default=0.08)
    parser.add_argument("--skip-two-block-prerequisite", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    steps = 3 if bool(args.quick) else int(args.steps)
    rows, summary = run_validation(
        out_dir=args.out_dir,
        steps=steps,
        lateral_shift=float(args.lateral_shift),
        run_two_block_prerequisite=not bool(args.skip_two_block_prerequisite),
    )
    print((args.out_dir / "tooth_patch_region_summary.md").read_text(encoding="utf-8"))
    return 0 if rows and summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
