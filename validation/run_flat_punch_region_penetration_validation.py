"""Flat-punch two-flexible-block region-contact penetration validation.

This is the first validation layer in the contact-alignment ladder.  It uses a
stacked two-block TET4 model and the Lagrangian-SDF surface-contact path, but
checks only region-level totals: reaction force versus prescribed penetration,
contact energy, active area, and active constraint-region count.  Nodal
CPRESS/COPEN clouds are intentionally not part of this gate.

For the aligned flat punch with a linear pressure-overclosure law, the expected
total normal force is ``F_n = k_p A p`` and the expected contact energy is
``0.5 k_p A p^2``.
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
from sfc.sdf.material_sdf import MaterialSDF  # noqa: E402
from validation.run_calculix_deformable_sdf_contact_validation import build_two_block_model  # noqa: E402
from validation.run_flexible_gear_implicit_lagrangian_sdf_comparison import (  # noqa: E402
    _aggregate_contact_sample_arrays,
    _assemble_contact_arrays_force_only,
    _constraint_region_contact_law_metrics_from_arrays,
    _contact_region_integral_metrics_from_arrays,
)
from validation.run_two_block_sliding_region_validation import _contact_samples_to_arrays, _write_csv  # noqa: E402

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "flat_punch_region_penetration_validation"


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("at least one penetration value is required")
    if any(value < 0.0 for value in values):
        raise ValueError("penetrations must be non-negative")
    return values


def _flat_punch_positions(reference_nodes: np.ndarray, *, lower_node_count: int, approach: float) -> np.ndarray:
    x = np.asarray(reference_nodes, dtype=float).copy()
    x[int(lower_node_count) :, 2] -= float(approach)
    return x


def flat_punch_rf_penetration_gate_metrics(
    rows: list[Row],
    *,
    max_force_rel_error: float = 1.0e-10,
    max_energy_rel_error: float = 1.0e-10,
) -> Row:
    """Gate the first validation layer before sliding or pressure clouds.

    The flat punch layer is a total-quantity check: RF/penetration, contact
    virtual work/energy, active area, and region count.  It intentionally does
    not use nodal CPRESS/COPEN fields.  Active rows must also prove that their
    force came from secondary constraint regions, not independent quadrature
    penalties.
    """

    positive_force_errors = [
        float(row["normal_force_rel_error"])
        for row in rows
        if float(row.get("expected_normal_force", 0.0)) > 0.0
    ]
    positive_energy_errors = [
        float(row["contact_energy_rel_error"])
        for row in rows
        if float(row.get("expected_contact_energy", 0.0)) > 0.0
    ]
    active_rows = [
        row
        for row in rows
        if int(float(row.get("active_contact_region_count", row.get("active_contact_samples", 0)))) > 0
        or float(row.get("contact_active_area", 0.0)) > 0.0
        or float(row.get("normal_force", 0.0)) > 0.0
    ]
    no_nodal_clouds = all(int(float(row.get("nodal_cpress_deferred", 0))) == 1 for row in rows)
    active_area_present = bool(active_rows) and all(float(row.get("contact_active_area", 0.0)) > 0.0 for row in active_rows)
    region_count_present = bool(rows) and all(int(float(row.get("constraint_regions", 0))) >= 0 for row in rows)
    law_rows_ok = True
    for row in active_rows:
        law_rows_ok = law_rows_ok and str(row.get("contact_constraint_law_source", "")) == "slave_node_region_constraint"
        law_rows_ok = (
            law_rows_ok
            and str(row.get("contact_constraint_open_closed_source", "")) == "signed_area_average_region_gap"
            and str(row.get("contact_constraint_normal_source", "")) == "area_average_region_normal"
            and str(row.get("contact_constraint_force_distribution", "")) == "region_area_slave_shape_master_payload"
            and int(float(row.get("contact_constraint_independent_quadrature_penalty_disabled", 0))) == 1
        )
    force_error_max = float(max(positive_force_errors) if positive_force_errors else 0.0)
    energy_error_max = float(max(positive_energy_errors) if positive_energy_errors else 0.0)
    force_ok = force_error_max <= float(max_force_rel_error)
    energy_ok = energy_error_max <= float(max_energy_rel_error)
    gate_passed = bool(rows) and force_ok and energy_ok and no_nodal_clouds and active_area_present and region_count_present and law_rows_ok
    return {
        "flat_punch_rf_penetration_gate_passed": int(gate_passed),
        "comparison_stage": "flat_punch_rf_penetration_before_sliding_or_clouds",
        "flat_punch_reference_source": "analytic_linear_pressure_overclosure",
        "flat_punch_row_count": int(len(rows)),
        "flat_punch_active_row_count": int(len(active_rows)),
        "flat_punch_max_normal_force_rel_error": force_error_max,
        "flat_punch_max_contact_energy_rel_error": energy_error_max,
        "flat_punch_max_force_rel_error_threshold": float(max_force_rel_error),
        "flat_punch_max_energy_rel_error_threshold": float(max_energy_rel_error),
        "flat_punch_force_gate_passed": int(force_ok),
        "flat_punch_energy_gate_passed": int(energy_ok),
        "flat_punch_nodal_cpress_deferred": int(no_nodal_clouds),
        "flat_punch_active_area_present": int(active_area_present),
        "flat_punch_region_count_present": int(region_count_present),
        "flat_punch_constraint_region_law_gate_passed": int(law_rows_ok),
        "flat_punch_ready_for_two_block_sliding": int(gate_passed),
    }


def run_validation(
    *,
    out_dir: Path,
    resolution: int = 2,
    gap: float = 0.035,
    penetrations: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.0075, 0.01),
    pressure_stiffness: float = 6.0e4,
    quadrature: str = "tri3",
) -> tuple[list[Row], Row]:
    """Run the flat-punch penetration sweep and write RF/penetration outputs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    max_penetration = max(float(value) for value in penetrations)
    model = build_two_block_model(resolution=int(resolution), approach=float(gap) + max_penetration, gap=float(gap))
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
        search_radius=float(gap) + max_penetration + 0.25,
        compiled_batch_projection=False,
        secondary_path_tracking=False,
    )
    rows: list[Row] = []
    start = time.perf_counter()
    for step, penetration in enumerate(penetrations):
        x = _flat_punch_positions(
            model.nodes,
            lower_node_count=int(model.lower_node_count),
            approach=float(gap) + float(penetration),
        )
        samples = list(contact.samples(x))
        arrays = _contact_samples_to_arrays(samples)
        regions = _aggregate_contact_sample_arrays(arrays, "slave_node_region_constraint")
        if regions is None:
            raise RuntimeError("slave_node_region_constraint aggregation unexpectedly fell back")
        response = _assemble_contact_arrays_force_only(regions, model.nodes.shape[0], stiffness=float(pressure_stiffness))
        region_metrics = _contact_region_integral_metrics_from_arrays(regions, stiffness=float(pressure_stiffness))
        law_metrics = _constraint_region_contact_law_metrics_from_arrays(
            regions,
            arrays,
            averaging_mode="slave_node_region_constraint",
        )
        active_area = float(region_metrics["contact_active_area"])
        expected_force = float(pressure_stiffness) * active_area * float(penetration)
        expected_energy = 0.5 * float(pressure_stiffness) * active_area * float(penetration) ** 2
        force_abs_error = abs(float(response.normal_force) - expected_force)
        energy_abs_error = abs(float(response.energy) - expected_energy)
        row: Row = {
            "step": int(step),
            "comparison_stage": "flat_punch_rf_penetration_before_sliding_or_clouds",
            "nodal_cpress_deferred": 1,
            "reference_source": "analytic_linear_pressure_overclosure",
            "penetration": float(penetration),
            "raw_quadrature_samples": int(len(samples)),
            "constraint_regions": int(np.asarray(regions["gaps"]).size),
            "active_contact_samples": int(response.active_count),
            "normal_force": float(response.normal_force),
            "expected_normal_force": float(expected_force),
            "normal_force_abs_error": float(force_abs_error),
            "normal_force_rel_error": float(force_abs_error / max(abs(expected_force), 1.0e-30)),
            "contact_energy": float(response.energy),
            "expected_contact_energy": float(expected_energy),
            "contact_energy_abs_error": float(energy_abs_error),
            "contact_energy_rel_error": float(energy_abs_error / max(abs(expected_energy), 1.0e-30)),
            "min_region_gap": float(np.min(np.asarray(regions["gaps"], dtype=float))) if np.asarray(regions["gaps"]).size else 0.0,
        }
        row.update(region_metrics)
        row.update(law_metrics)
        rows.append(row)
    wall = time.perf_counter() - start
    totals_path = out_dir / "flat_punch_rf_penetration.csv"
    gate_path = out_dir / "flat_punch_rf_penetration_gate.csv"
    summary_path = out_dir / "flat_punch_region_summary.md"
    summary_csv = out_dir / "flat_punch_region_summary.csv"
    _write_csv(totals_path, rows)
    gate = flat_punch_rf_penetration_gate_metrics(rows)
    _write_csv(gate_path, [gate])
    summary: Row = {
        "status": "completed",
        "resolution": int(resolution),
        "penetration_count": int(len(penetrations)),
        "max_normal_force_rel_error": float(gate["flat_punch_max_normal_force_rel_error"]),
        "max_contact_energy_rel_error": float(gate["flat_punch_max_contact_energy_rel_error"]),
        "flat_punch_rf_penetration_gate_passed": int(gate["flat_punch_rf_penetration_gate_passed"]),
        "flat_punch_ready_for_two_block_sliding": int(gate["flat_punch_ready_for_two_block_sliding"]),
        "active_area_final": float(rows[-1]["contact_active_area"]) if rows else 0.0,
        "normal_force_final": float(rows[-1]["normal_force"]) if rows else 0.0,
        "analysis_wall_seconds": float(wall),
        "totals_csv": str(totals_path),
        "gate_csv": str(gate_path),
    }
    summary.update(gate)
    _write_csv(summary_csv, [summary])
    summary_path.write_text(
        "\n".join(
            [
                "# Flat-Punch Region Penetration Validation",
                "",
                "This validation checks RF/penetration totals before nodal CPRESS/COPEN cloud comparisons.",
                "",
                f"- resolution: {summary['resolution']}",
                f"- penetration count: {summary['penetration_count']}",
                f"- max normal-force relative error: {summary['max_normal_force_rel_error']:.12e}",
                f"- max contact-energy relative error: {summary['max_contact_energy_rel_error']:.12e}",
                f"- RF/penetration gate: {'PASS' if int(summary['flat_punch_rf_penetration_gate_passed']) else 'FAIL'}",
                f"- nodal CPRESS/COPEN deferred: {'yes' if int(summary['flat_punch_nodal_cpress_deferred']) else 'no'}",
                f"- constraint-region law gate: {'PASS' if int(summary['flat_punch_constraint_region_law_gate_passed']) else 'FAIL'}",
                f"- final active area: {summary['active_area_final']:.12e}",
                f"- final normal force: {summary['normal_force_final']:.12e}",
                f"- totals CSV: `{totals_path.name}`",
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
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--gap", type=float, default=0.035)
    parser.add_argument("--penetrations", type=str, default="0.0,0.0025,0.005,0.0075,0.01")
    parser.add_argument("--pressure-stiffness", type=float, default=6.0e4)
    parser.add_argument("--quick", action="store_true", help="Use a smaller deterministic smoke case.")
    args = parser.parse_args(argv)
    penetrations = (0.0, 0.005, 0.01) if bool(args.quick) else _parse_float_list(args.penetrations)
    resolution = 1 if bool(args.quick) else int(args.resolution)
    rows, summary = run_validation(
        out_dir=args.out_dir,
        resolution=resolution,
        gap=float(args.gap),
        penetrations=penetrations,
        pressure_stiffness=float(args.pressure_stiffness),
    )
    print((args.out_dir / "flat_punch_region_summary.md").read_text(encoding="utf-8"))
    return 0 if rows and summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
