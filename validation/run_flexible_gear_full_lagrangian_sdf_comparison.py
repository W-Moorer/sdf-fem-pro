"""Full flexible-gear implicit Lagrangian-SDF contact runner.

This validation runner keeps the complete two-gear TET4 volume meshes and
selects only the currently relevant contact surface patch for the Lagrangian
SDF contact constraints.  The active-patch selection is a broad-phase step; the
FEM solve, RP-MPC kinematics, contact gap/Jacobian, and reaction diagnostics
remain internal to SFC.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact.validation_gates import (  # noqa: E402
    contact_total_priority_gate_metrics as _core_contact_total_priority_gate_metrics,
)
from validation.run_flexible_gear_explicit_sdf_comparison import DEFAULT_SOURCE, GearInputModel, GearMesh, parse_gear_input  # noqa: E402
from validation.run_flexible_gear_implicit_lagrangian_sdf_comparison import (  # noqa: E402
    ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
    CroppedGearPair,
    CroppedGearPatch,
    _face_centroids,
    _orient_faces_toward,
    _resolve_abaqus_command,
    _run_command,
    _write_abaqus_alignment_deck,
    _write_csv,
    run_abaqus_alignment,
    compare_histories,
    plot_alignment_curves,
    solve_sfc_cropped_pair,
    solve_sfc_cropped_pair_hard_contact,
    solve_sfc_source_drive_pair,
)

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "flexible_gear_full_lagrangian_sdf"

PATCH_PREREQUISITE_GATE_KEYS: tuple[str, ...] = (
    "tooth_patch_region_gate_passed",
    "tooth_patch_ready_for_cropped_gear_patch",
    "tooth_patch_constraint_region_law_gate_passed",
    "tooth_patch_path_tracking_gate_passed",
    "tooth_patch_contact_total_gate_passed",
    "cropped_patch_gate_passed",
    "cropped_patch_pressure_stress_gate_passed",
    "cropped_patch_path_tracking_gate_passed",
    "cropped_patch_active_region_continuity_gate_passed",
    "cropped_patch_contact_total_gate_passed",
    "cropped_patch_increment_gate_passed",
    "cropped_patch_constraint_region_tangent_gate_passed",
    "cropped_patch_active_set_line_search_gate_passed",
)

CONTACT_TOTAL_PRIORITY_COLUMNS: tuple[str, ...] = (
    "time",
    "source_accepted_step",
    "normal_force",
    "rp_force_norm",
    "opposing_rp_force_norm",
    "rp_contact_force_norm",
    "opposing_rp_contact_force_norm",
    "contact_region_normal_force",
    "contact_region_virtual_work",
    "contact_region_energy",
    "contact_active_area",
    "contact_region_count",
    "active_contact_region_count",
    "contact_mean_active_region_pressure",
    "contact_active_region_jaccard",
    "contact_active_region_persistence_fraction",
    "contact_path_cache_hit_fraction",
    "contact_path_cache_match_fraction",
    "contact_master_face_switch_fraction",
    "contact_master_barycentric_drift_mean",
    "contact_master_barycentric_drift_max",
    "contact_active_region_continuity_previous_count",
    "contact_active_region_continuity_current_count",
    "contact_active_region_continuity_intersection_count",
    "contact_active_region_new_count",
    "contact_active_region_dropped_count",
    "contact_constraint_law_source",
    "contact_constraint_region_source",
    "contact_constraint_open_closed_source",
    "contact_constraint_active_status_source",
    "contact_constraint_normal_source",
    "contact_constraint_region_area_source",
    "contact_constraint_force_distribution",
    "contact_constraint_raw_sample_count",
    "contact_constraint_region_count",
    "contact_constraint_active_region_count",
    "contact_constraint_secondary_node_regions_present",
    "contact_constraint_master_payload_present",
    "contact_constraint_area_positive",
    "contact_constraint_independent_quadrature_penalty_disabled",
    "contact_secondary_pressure_recovery_source",
)

NODAL_CONTACT_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "active_contact_node_count",
    "active_contact_secondary_node_count",
    "max_contact_pressure_nodeavg",
    "max_contact_secondary_pressure_nodeavg",
    "p95_contact_pressure_nodeavg",
    "p95_contact_secondary_pressure_nodeavg",
    "mean_active_contact_pressure_nodeavg",
    "mean_active_contact_secondary_pressure_nodeavg",
)


def _read_csv_rows(path: Path) -> list[Row]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_patch_prerequisite_summary(paths: Sequence[Path]) -> Row:
    """Load explicit patch-ladder gate evidence for the full-gear entry gate.

    Full-gear validation is not allowed to infer tooth/cropped-patch evidence
    from the full solve itself.  This helper only accepts already-generated
    summary CSV files and only forwards the known patch-ladder gate fields.
    Duplicate gate keys must agree so that a later CSV cannot silently override
    a failing prerequisite.
    """

    merged: Row = {}
    resolved_paths: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        rows = _read_csv_rows(path)
        if not rows:
            raise ValueError(f"patch prerequisite summary is missing or empty: {path}")
        row = rows[-1]
        recognized = {key: row[key] for key in PATCH_PREREQUISITE_GATE_KEYS if key in row}
        if not recognized:
            raise ValueError(f"patch prerequisite summary has no recognized gate columns: {path}")
        for key, value in recognized.items():
            if key in merged and str(merged[key]) != str(value):
                raise ValueError(f"conflicting patch prerequisite gate `{key}` in {path}")
            merged[key] = value
        resolved_paths.append(str(path))
    if resolved_paths:
        merged["full_gear_patch_prerequisite_summary_count"] = int(len(resolved_paths))
        merged["full_gear_patch_prerequisite_summaries"] = ";".join(resolved_paths)
        merged["full_gear_patch_prerequisite_keys_present"] = ";".join(
            sorted(k for k in merged if k in PATCH_PREREQUISITE_GATE_KEYS)
        )
    return merged


def _max_column(rows: list[Row], column: str) -> float:
    values: list[float] = []
    for row in rows:
        value = row.get(column, "")
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values, default=0.0)


def _manifest_column_available(rows: list[Row], column: str) -> bool:
    return bool(rows) and column in rows[0]


def write_contact_total_priority_csv(history_rows: list[Row], out_path: Path) -> Path:
    """Write the contact-total gate CSV before nodal CPRESS/COPEN checks.

    This file intentionally excludes nodal pressure values.  It is the
    comparison entry point for total force, contact work/energy, active area,
    active region count, and path/active-set continuity.
    """

    rows: list[Row] = []
    for index, source in enumerate(history_rows):
        row: Row = {
            "frame": int(index),
            "comparison_stage": "region_totals_before_nodal_cpress",
            "nodal_cpress_deferred": 1,
        }
        for key in CONTACT_TOTAL_PRIORITY_COLUMNS:
            if key in source:
                row[key] = source[key]
        present_nodal = [key for key in NODAL_CONTACT_DIAGNOSTIC_COLUMNS if key in source]
        row["deferred_nodal_contact_columns"] = ";".join(present_nodal)
        rows.append(row)
    _write_csv(out_path, rows)
    return out_path


def _row_has_value(row: Row, key: str) -> bool:
    value = row.get(key)
    return value not in ("", None)


def contact_total_priority_gate_metrics(priority_rows: list[Row]) -> Row:
    """Gate for total contact quantities before nodal CPRESS/COPEN checks."""

    return dict(_core_contact_total_priority_gate_metrics(priority_rows))


def _finite_row_float(row: Row, *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value in ("", None):
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(result):
            return result
    return None


def _row_int_flag(row: Row, key: str, *, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _match_time_sequence(
    query_times: list[float],
    reference_times: list[float],
    *,
    tolerance: float,
) -> tuple[int, float]:
    """Return ordered-subsequence matches and the largest matched time error."""

    if not query_times:
        return 0, 0.0
    reference_index = 0
    matched = 0
    max_error = 0.0
    for query_time in query_times:
        best_index = -1
        best_error = np.inf
        for index in range(reference_index, len(reference_times)):
            error = abs(reference_times[index] - query_time)
            if error < best_error:
                best_error = error
                best_index = index
            if error <= tolerance:
                break
        if best_index >= 0 and best_error <= tolerance:
            reference_index = best_index + 1
            matched += 1
            max_error = max(max_error, float(best_error))
    return matched, float(max_error if np.isfinite(max_error) else np.inf)


def _time_in_list(time: float, candidates: list[float], *, tolerance: float) -> bool:
    return any(abs(candidate - time) <= tolerance for candidate in candidates)


def source_increment_trial_gate_metrics(
    history_rows: list[Row],
    trial_rows: list[Row],
    *,
    tolerance: float = 1.0e-12,
) -> Row:
    """Gate that rejected/cutback source trials never become accepted history.

    History may be frame-strided, so the accepted history times only need to be
    an ordered subset of the accepted trial end times.  This still proves the
    Abaqus-style cutback ledger is the state authority: rejected trials are
    recorded for diagnostics, but accepted output can only come from accepted
    trial states.
    """

    accepted_trials = [row for row in trial_rows if _row_int_flag(row, "source_trial_accepted") == 1]
    rejected_trials = [row for row in trial_rows if _row_int_flag(row, "source_trial_accepted") == 0]
    history_times = [
        time
        for time in (_finite_row_float(row, "time", "source_trial_end_time") for row in history_rows)
        if time is not None
    ]
    accepted_trial_times = [
        time
        for time in (_finite_row_float(row, "source_trial_end_time", "time") for row in accepted_trials)
        if time is not None
    ]
    rejected_trial_times = [
        time
        for time in (_finite_row_float(row, "source_trial_end_time", "time") for row in rejected_trials)
        if time is not None
    ]
    matched_count, max_time_error = _match_time_sequence(
        history_times,
        accepted_trial_times,
        tolerance=float(tolerance),
    )
    unmatched_history_count = max(0, len(history_times) - matched_count)
    history_all_accepted = all(_row_int_flag(row, "source_increment_accepted", default=1) == 1 for row in history_rows)
    rejected_only_history_hits = 0
    for rejected_time in rejected_trial_times:
        if not _time_in_list(rejected_time, accepted_trial_times, tolerance=float(tolerance)) and _time_in_list(
            rejected_time,
            history_times,
            tolerance=float(tolerance),
        ):
            rejected_only_history_hits += 1
    gate_passed = int(unmatched_history_count == 0 and history_all_accepted and rejected_only_history_hits == 0)
    return {
        "source_trial_gate_passed": gate_passed,
        "source_trial_history_times_in_accepted_trials": int(unmatched_history_count == 0),
        "source_trial_history_all_accepted": int(history_all_accepted),
        "source_trial_rejected_only_history_hits": int(rejected_only_history_hits),
        "source_trial_accepted_count": int(len(accepted_trials)),
        "source_trial_rejected_count": int(len(rejected_trials)),
        "source_trial_history_row_count": int(len(history_rows)),
        "source_trial_history_time_match_count": int(matched_count),
        "source_trial_unmatched_history_count": int(unmatched_history_count),
        "source_trial_accepted_history_time_linf": float(max_time_error),
        "source_trial_last_history_time": float(history_times[-1]) if history_times else 0.0,
        "source_trial_last_accepted_trial_time": float(accepted_trial_times[-1]) if accepted_trial_times else 0.0,
        "source_trial_time_tolerance": float(tolerance),
    }


def source_convergence_gate_metrics(
    summary: Row,
    history_rows: list[Row],
    *,
    tolerance: float = 1.0e-12,
) -> Row:
    """Return the full-run source convergence gate before nodal field checks.

    This gate is deliberately about accepted-state semantics, not cloud-plot
    agreement: accepted increments must come from converged source trials,
    unstable/unconverged states must not be accepted, response reuse/requery
    must cover accepted increments, and active contact should exercise the
    constraint-region tangent path.
    """

    accepted_count = _row_int_flag(summary, "source_accepted_increment_count", default=0)
    expected_count = _row_int_flag(summary, "sfc_increment_count", default=accepted_count)
    converged_count = _row_int_flag(summary, "source_step_converged_count", default=0)
    unstable_accepted = _row_int_flag(summary, "source_unstable_accepted_count", default=0)
    unconverged_accepted = _row_int_flag(summary, "source_unconverged_accepted_count", default=0)
    rejected_trials = _row_int_flag(summary, "source_rejected_trial_count", default=0)
    cutback_count = _row_int_flag(summary, "source_cutback_required_count", default=0)
    trial_gate_passed = _row_int_flag(summary, "source_trial_gate_passed", default=1)
    reuse_count = _row_int_flag(summary, "source_accepted_contact_response_reuse_count", default=0)
    requery_count = _row_int_flag(summary, "source_accepted_contact_response_requery_count", default=0)
    response_count = reuse_count + requery_count
    tangent_solve_count = _row_int_flag(summary, "source_constraint_region_tangent_solve_count", default=0)
    tangent_active_rows = _row_int_flag(summary, "source_constraint_region_tangent_active_rows_sum", default=0)
    final_active_samples = _row_int_flag(summary, "final_active_contact_samples", default=0)
    tangent_required = int(final_active_samples > 0 or tangent_active_rows > 0)
    final_time = _finite_row_float(summary, "source_final_time")
    target_time = _finite_row_float(summary, "sfc_duration")
    time_error = 0.0
    final_time_matches = True
    if final_time is not None and target_time is not None:
        time_error = abs(float(final_time) - float(target_time))
        final_time_matches = time_error <= max(float(tolerance), abs(float(target_time)) * 1.0e-9)
    history_all_accepted = all(_row_int_flag(row, "source_increment_accepted", default=1) == 1 for row in history_rows)
    history_step_count = len(history_rows)
    accepted_count_matches_expected = expected_count <= 0 or accepted_count == expected_count
    converged_count_matches_accepted = accepted_count <= 0 or converged_count == accepted_count
    response_count_matches_accepted = response_count == accepted_count or (
        response_count == 0 and accepted_count == 0
    )
    tangent_used_when_required = (not bool(tangent_required)) or tangent_solve_count > 0
    no_bad_accepted_state = unstable_accepted == 0 and unconverged_accepted == 0
    gate_passed = int(
        bool(trial_gate_passed)
        and bool(history_all_accepted)
        and bool(accepted_count_matches_expected)
        and bool(converged_count_matches_accepted)
        and bool(response_count_matches_accepted)
        and bool(tangent_used_when_required)
        and bool(final_time_matches)
        and bool(no_bad_accepted_state)
    )
    return {
        "source_convergence_gate_passed": gate_passed,
        "comparison_stage": "source_convergence_before_nodal_cpress",
        "source_convergence_history_all_accepted": int(history_all_accepted),
        "source_convergence_accepted_count_matches_expected": int(accepted_count_matches_expected),
        "source_convergence_converged_count_matches_accepted": int(converged_count_matches_accepted),
        "source_convergence_response_count_matches_accepted": int(response_count_matches_accepted),
        "source_convergence_tangent_used_when_required": int(tangent_used_when_required),
        "source_convergence_final_time_matches_duration": int(final_time_matches),
        "source_convergence_no_unstable_or_unconverged_accepted": int(no_bad_accepted_state),
        "source_convergence_trial_gate_passed": int(trial_gate_passed),
        "source_convergence_history_row_count": int(history_step_count),
        "source_convergence_expected_increment_count": int(expected_count),
        "source_convergence_accepted_increment_count": int(accepted_count),
        "source_convergence_converged_step_count": int(converged_count),
        "source_convergence_rejected_trial_count": int(rejected_trials),
        "source_convergence_cutback_required_count": int(cutback_count),
        "source_convergence_unstable_accepted_count": int(unstable_accepted),
        "source_convergence_unconverged_accepted_count": int(unconverged_accepted),
        "source_convergence_response_reuse_count": int(reuse_count),
        "source_convergence_response_requery_count": int(requery_count),
        "source_convergence_response_total_count": int(response_count),
        "source_convergence_tangent_required": int(tangent_required),
        "source_convergence_tangent_solve_count": int(tangent_solve_count),
        "source_convergence_tangent_active_rows_sum": int(tangent_active_rows),
        "source_convergence_final_time_error": float(time_error),
    }


def source_active_set_line_search_gate_metrics(summary: Row, history_rows: list[Row]) -> Row:
    """Gate active-set line-search and accepted tracking before full gear use."""

    active_rows: list[Row] = []
    for row in history_rows:
        active_regions = _row_int_flag(row, "active_contact_region_count", default=0)
        active_area = _finite_row_float(row, "contact_active_area") or 0.0
        normal_force = abs(_finite_row_float(row, "contact_region_normal_force", "normal_force") or 0.0)
        if active_regions > 0 or active_area > 0.0 or normal_force > 0.0:
            active_rows.append(row)
    active_contact_present = bool(active_rows) or _row_int_flag(summary, "final_active_contact_samples", default=0) > 0
    line_search_enabled = str(summary.get("source_active_set_line_search", "true")).lower() == "true"
    stability_enabled = str(summary.get("source_contact_active_set_stability", "true")).lower() == "true"
    summary_unstable = _row_int_flag(summary, "source_line_search_unstable_count", default=0)
    summary_trials = _row_int_flag(summary, "source_line_search_trial_count", default=0)
    summary_commit = _row_int_flag(summary, "source_accepted_tracking_commit_count", default=0)
    row_columns_present = all(
        all(
            key in row
            for key in (
                "source_line_search_trial_count",
                "source_line_search_reduced_count",
                "source_line_search_stable_count",
                "source_line_search_unstable_count",
                "source_line_search_last_alpha",
                "source_accepted_tracking_committed",
            )
        )
        for row in history_rows
    )
    row_unstable_ok = all(_row_int_flag(row, "source_line_search_unstable_count", default=0) == 0 for row in history_rows)
    row_alpha_ok = all(
        0.0 < (_finite_row_float(row, "source_line_search_last_alpha") or 1.0) <= 1.0 for row in history_rows
    )
    row_commit_ok = (not active_contact_present) or any(
        _row_int_flag(row, "source_accepted_tracking_committed", default=0) > 0 for row in active_rows
    )
    summary_commit_ok = (not active_contact_present) or summary_commit > 0
    summary_trials_ok = (not active_contact_present) or summary_trials > 0 or not bool(line_search_enabled)
    gate_passed = int(
        bool(line_search_enabled)
        and bool(stability_enabled)
        and summary_unstable == 0
        and bool(row_columns_present)
        and bool(row_unstable_ok)
        and bool(row_alpha_ok)
        and bool(row_commit_ok)
        and bool(summary_commit_ok)
        and bool(summary_trials_ok)
    )
    return {
        "source_active_set_line_search_gate_passed": gate_passed,
        "comparison_stage": "active_set_line_search_before_full_gear_sync",
        "source_active_set_line_search_active_contact_present": int(active_contact_present),
        "source_active_set_line_search_enabled": int(bool(line_search_enabled)),
        "source_active_set_stability_enabled": int(bool(stability_enabled)),
        "source_active_set_line_search_summary_unstable_count": int(summary_unstable),
        "source_active_set_line_search_summary_trial_count": int(summary_trials),
        "source_active_set_line_search_summary_commit_count": int(summary_commit),
        "source_active_set_line_search_row_columns_present": int(row_columns_present),
        "source_active_set_line_search_row_unstable_ok": int(row_unstable_ok),
        "source_active_set_line_search_row_alpha_ok": int(row_alpha_ok),
        "source_active_set_line_search_row_commit_ok": int(row_commit_ok),
        "source_active_set_line_search_summary_commit_ok": int(summary_commit_ok),
        "source_active_set_line_search_summary_trials_ok": int(summary_trials_ok),
        "source_active_set_line_search_history_row_count": int(len(history_rows)),
        "source_active_set_line_search_active_history_row_count": int(len(active_rows)),
    }


def constraint_region_tangent_gate_metrics(summary: Row, history_rows: list[Row]) -> Row:
    """Gate fixed-active-set consistent tangent evidence for region contact.

    Active accepted rows must use the constraint-region gap Jacobian, the
    linear pressure-overclosure derivative, and a source solve that actually
    exercised the region tangent.  This is intentionally checked before nodal
    pressure and cloud comparison gates.
    """

    active_rows: list[Row] = []
    for row in history_rows:
        active_regions = _row_int_flag(row, "active_contact_region_count", default=0)
        active_area = _finite_row_float(row, "contact_active_area") or 0.0
        normal_force = abs(_finite_row_float(row, "contact_region_normal_force", "normal_force") or 0.0)
        if active_regions > 0 or active_area > 0.0 or normal_force > 0.0:
            active_rows.append(row)
    active_contact_present = len(active_rows) > 0
    source_solve_count = _row_int_flag(summary, "source_constraint_region_tangent_solve_count", default=0)
    source_active_rows = _row_int_flag(summary, "source_constraint_region_tangent_active_rows_sum", default=0)
    source_j_nnz = _row_int_flag(summary, "source_constraint_region_tangent_j_nnz_sum", default=0)
    source_scale_sum = _finite_row_float(summary, "source_constraint_region_tangent_scale_sum") or 0.0
    row_source_ok = True
    row_jacobian_ok = True
    row_derivative_ok = True
    row_fixed_active_ok = True
    row_active_count_ok = True
    row_scale_ok = True
    row_solve_ok = True
    max_active_count_mismatch = 0
    for row in active_rows:
        tangent_active = _row_int_flag(row, "contact_tangent_active_region_count", default=-1)
        active_regions = _row_int_flag(row, "active_contact_region_count", default=0)
        max_active_count_mismatch = max(max_active_count_mismatch, abs(tangent_active - active_regions))
        row_source_ok = row_source_ok and str(row.get("contact_tangent_source", "")) == "constraint_region_arrays"
        row_jacobian_ok = (
            row_jacobian_ok
            and str(row.get("contact_tangent_gap_jacobian_source", "")) == "constraint_region_fixed_payload"
        )
        row_derivative_ok = (
            row_derivative_ok
            and str(row.get("contact_tangent_pressure_derivative", "")) == "linear_penalty_active_set"
        )
        row_fixed_active_ok = row_fixed_active_ok and _row_int_flag(row, "contact_tangent_fixed_active_set") == 1
        row_active_count_ok = row_active_count_ok and tangent_active == active_regions
        row_scale_ok = row_scale_ok and (_finite_row_float(row, "contact_tangent_scale_sum") or 0.0) > 0.0
        row_solve_ok = row_solve_ok and _row_int_flag(row, "source_constraint_region_tangent_solve_count") > 0
    summary_tangent_used = (not active_contact_present) or (
        source_solve_count > 0 and source_active_rows > 0 and source_j_nnz > 0 and source_scale_sum > 0.0
    )
    rows_ok = (
        row_source_ok
        and row_jacobian_ok
        and row_derivative_ok
        and row_fixed_active_ok
        and row_active_count_ok
        and row_scale_ok
        and row_solve_ok
    )
    gate_passed = int((not active_contact_present) or (bool(summary_tangent_used) and bool(rows_ok)))
    return {
        "constraint_region_tangent_gate_passed": gate_passed,
        "comparison_stage": "constraint_region_tangent_before_clouds",
        "constraint_region_tangent_active_contact_present": int(active_contact_present),
        "constraint_region_tangent_active_history_row_count": int(len(active_rows)),
        "constraint_region_tangent_summary_used": int(summary_tangent_used),
        "constraint_region_tangent_row_source_ok": int(row_source_ok),
        "constraint_region_tangent_row_jacobian_ok": int(row_jacobian_ok),
        "constraint_region_tangent_row_pressure_derivative_ok": int(row_derivative_ok),
        "constraint_region_tangent_row_fixed_active_set_ok": int(row_fixed_active_ok),
        "constraint_region_tangent_row_active_count_ok": int(row_active_count_ok),
        "constraint_region_tangent_row_scale_ok": int(row_scale_ok),
        "constraint_region_tangent_row_solve_ok": int(row_solve_ok),
        "constraint_region_tangent_max_active_count_mismatch": int(max_active_count_mismatch),
        "constraint_region_tangent_source_solve_count": int(source_solve_count),
        "constraint_region_tangent_source_active_rows_sum": int(source_active_rows),
        "constraint_region_tangent_source_j_nnz_sum": int(source_j_nnz),
        "constraint_region_tangent_source_scale_sum": float(source_scale_sum),
    }


def constraint_region_contact_law_gate_metrics(history_rows: list[Row]) -> Row:
    """Gate that accepted active contact uses region constraints, not samples.

    Active accepted rows must prove that open/closed status comes from the
    signed area-average region gap, that the contact direction is the region
    average normal, and that forces are distributed through the region area,
    slave shape-function support, and master closest-feature payload.
    """

    active_rows: list[Row] = []
    for row in history_rows:
        active_regions = _row_int_flag(row, "active_contact_region_count", default=0)
        active_area = _finite_row_float(row, "contact_active_area") or 0.0
        normal_force = abs(_finite_row_float(row, "contact_region_normal_force", "normal_force") or 0.0)
        if active_regions > 0 or active_area > 0.0 or normal_force > 0.0:
            active_rows.append(row)
    active_contact_present = len(active_rows) > 0
    row_law_ok = True
    row_gap_ok = True
    row_normal_ok = True
    row_distribution_ok = True
    row_region_area_ok = True
    row_secondary_ok = True
    row_master_payload_ok = True
    row_area_positive_ok = True
    row_no_independent_sample_penalty_ok = True
    max_raw_to_region_ratio = 0.0
    for row in active_rows:
        row_law_ok = row_law_ok and str(row.get("contact_constraint_law_source", "")) == "slave_node_region_constraint"
        row_gap_ok = (
            row_gap_ok
            and str(row.get("contact_constraint_open_closed_source", "")) == "signed_area_average_region_gap"
            and str(row.get("contact_constraint_active_status_source", "")) == "aggregated_region_gap"
        )
        row_normal_ok = row_normal_ok and str(row.get("contact_constraint_normal_source", "")) == "area_average_region_normal"
        row_region_area_ok = (
            row_region_area_ok
            and str(row.get("contact_constraint_region_area_source", "")) == "slave_shape_tributary_area"
        )
        row_distribution_ok = (
            row_distribution_ok
            and str(row.get("contact_constraint_force_distribution", "")) == "region_area_slave_shape_master_payload"
        )
        row_secondary_ok = row_secondary_ok and _row_int_flag(row, "contact_constraint_secondary_node_regions_present") == 1
        row_master_payload_ok = row_master_payload_ok and _row_int_flag(row, "contact_constraint_master_payload_present") == 1
        row_area_positive_ok = row_area_positive_ok and _row_int_flag(row, "contact_constraint_area_positive") == 1
        row_no_independent_sample_penalty_ok = (
            row_no_independent_sample_penalty_ok
            and _row_int_flag(row, "contact_constraint_independent_quadrature_penalty_disabled") == 1
        )
        raw_count = _finite_row_float(row, "contact_constraint_raw_sample_count")
        region_count = _finite_row_float(row, "contact_constraint_region_count")
        if raw_count is not None and region_count is not None and region_count > 0.0:
            max_raw_to_region_ratio = max(max_raw_to_region_ratio, float(raw_count) / float(region_count))
    rows_ok = (
        row_law_ok
        and row_gap_ok
        and row_normal_ok
        and row_distribution_ok
        and row_region_area_ok
        and row_secondary_ok
        and row_master_payload_ok
        and row_area_positive_ok
        and row_no_independent_sample_penalty_ok
    )
    return {
        "constraint_region_contact_law_gate_passed": int((not active_contact_present) or rows_ok),
        "comparison_stage": "constraint_region_contact_law_before_totals",
        "constraint_region_contact_law_active_contact_present": int(active_contact_present),
        "constraint_region_contact_law_active_history_row_count": int(len(active_rows)),
        "constraint_region_contact_law_row_law_ok": int(row_law_ok),
        "constraint_region_contact_law_row_gap_status_ok": int(row_gap_ok),
        "constraint_region_contact_law_row_normal_ok": int(row_normal_ok),
        "constraint_region_contact_law_row_region_area_ok": int(row_region_area_ok),
        "constraint_region_contact_law_row_distribution_ok": int(row_distribution_ok),
        "constraint_region_contact_law_row_secondary_region_ok": int(row_secondary_ok),
        "constraint_region_contact_law_row_master_payload_ok": int(row_master_payload_ok),
        "constraint_region_contact_law_row_area_positive_ok": int(row_area_positive_ok),
        "constraint_region_contact_law_no_independent_quadrature_penalty": int(row_no_independent_sample_penalty_ok),
        "constraint_region_contact_law_max_raw_to_region_ratio": float(max_raw_to_region_ratio),
    }


def path_tracking_gate_metrics(
    history_rows: list[Row],
    *,
    min_cache_hit_fraction: float = 0.95,
    min_cache_match_fraction: float = 0.0,
    min_active_region_jaccard: float = 0.25,
    min_active_region_persistence: float = 0.25,
    max_master_face_switch_fraction: float = 0.85,
    max_master_barycentric_drift: float = 1.50,
) -> Row:
    """Gate accepted-state master-payload path tracking before cloud checks.

    The full-gear validation must prove that each accepted secondary
    constraint region carries a coherent previous master face/barycentric
    payload.  This gate evaluates only accepted history rows and intentionally
    sits before nodal CPRESS/COPEN cloud comparison.  A single active row is
    reported as not yet continuity-observable; multi-row active contact must
    retain high cache-hit continuity, bounded face switching, bounded
    barycentric drift on same-face samples, and persistent active regions.
    """

    active_rows: list[Row] = []
    for row in history_rows:
        active_regions = _row_int_flag(row, "active_contact_region_count", default=0)
        active_area = _finite_row_float(row, "contact_active_area") or 0.0
        normal_force = abs(_finite_row_float(row, "contact_region_normal_force", "normal_force") or 0.0)
        if active_regions > 0 or active_area > 0.0 or normal_force > 0.0:
            active_rows.append(row)
    active_contact_present = len(active_rows) > 0
    tracking_rows = active_rows[1:] if len(active_rows) > 1 else []
    required_columns = (
        "contact_path_cache_hit_fraction",
        "contact_path_cache_match_fraction",
        "contact_master_face_switch_fraction",
        "contact_active_region_jaccard",
        "contact_active_region_persistence_fraction",
        "contact_master_barycentric_drift_max",
    )
    columns_present = all(_row_has_value(row, key) for row in tracking_rows for key in required_columns)
    continuity_observable = len(tracking_rows) > 0
    if tracking_rows and columns_present:
        cache_hit_min = min(float(row["contact_path_cache_hit_fraction"]) for row in tracking_rows)
        cache_match_min = min(float(row["contact_path_cache_match_fraction"]) for row in tracking_rows)
        face_switch_max = max(float(row["contact_master_face_switch_fraction"]) for row in tracking_rows)
        active_jaccard_min = min(float(row["contact_active_region_jaccard"]) for row in tracking_rows)
        active_persistence_min = min(float(row["contact_active_region_persistence_fraction"]) for row in tracking_rows)
        barycentric_drift_max = max(float(row["contact_master_barycentric_drift_max"]) for row in tracking_rows)
    else:
        cache_hit_min = 1.0 if not active_contact_present else 0.0
        cache_match_min = 1.0 if not active_contact_present else 0.0
        face_switch_max = 0.0
        active_jaccard_min = 1.0 if not active_contact_present else 0.0
        active_persistence_min = 1.0 if not active_contact_present else 0.0
        barycentric_drift_max = 0.0
    cache_hit_ok = cache_hit_min >= float(min_cache_hit_fraction)
    cache_match_ok = cache_match_min >= float(min_cache_match_fraction)
    face_switch_ok = face_switch_max <= float(max_master_face_switch_fraction)
    active_jaccard_ok = active_jaccard_min >= float(min_active_region_jaccard)
    active_persistence_ok = active_persistence_min >= float(min_active_region_persistence)
    barycentric_ok = barycentric_drift_max <= float(max_master_barycentric_drift)
    if not active_contact_present:
        gate_passed = 1
    elif not continuity_observable:
        gate_passed = 1
    else:
        gate_passed = int(
            columns_present
            and cache_hit_ok
            and cache_match_ok
            and face_switch_ok
            and active_jaccard_ok
            and active_persistence_ok
            and barycentric_ok
        )
    return {
        "path_tracking_gate_passed": int(gate_passed),
        "comparison_stage": "accepted_state_path_tracking_before_nodal_cpress",
        "path_tracking_active_contact_present": int(active_contact_present),
        "path_tracking_active_history_row_count": int(len(active_rows)),
        "path_tracking_continuity_observable": int(continuity_observable),
        "path_tracking_required_columns_present": int(columns_present),
        "path_tracking_cache_hit_gate_passed": int(cache_hit_ok),
        "path_tracking_cache_match_gate_passed": int(cache_match_ok),
        "path_tracking_face_switch_gate_passed": int(face_switch_ok),
        "path_tracking_active_region_jaccard_gate_passed": int(active_jaccard_ok),
        "path_tracking_active_region_persistence_gate_passed": int(active_persistence_ok),
        "path_tracking_barycentric_drift_gate_passed": int(barycentric_ok),
        "path_tracking_cache_hit_fraction_min_after_first_active": float(cache_hit_min),
        "path_tracking_cache_match_fraction_min_after_first_active": float(cache_match_min),
        "path_tracking_master_face_switch_fraction_max_after_first_active": float(face_switch_max),
        "path_tracking_active_region_jaccard_min_after_first_active": float(active_jaccard_min),
        "path_tracking_active_region_persistence_min_after_first_active": float(active_persistence_min),
        "path_tracking_master_barycentric_drift_max_after_first_active": float(barycentric_drift_max),
        "path_tracking_min_cache_hit_threshold": float(min_cache_hit_fraction),
        "path_tracking_min_cache_match_threshold": float(min_cache_match_fraction),
        "path_tracking_min_active_region_jaccard_threshold": float(min_active_region_jaccard),
        "path_tracking_min_active_region_persistence_threshold": float(min_active_region_persistence),
        "path_tracking_max_master_face_switch_threshold": float(max_master_face_switch_fraction),
        "path_tracking_max_master_barycentric_drift_threshold": float(max_master_barycentric_drift),
    }


def full_gear_entry_gate_metrics(summary: Row, *, min_strict_sync_steps: int = 10) -> Row:
    """Gate the full-gear comparison ladder before nodal pressure/cloud checks.

    The first paper-facing comparison for the full-gear path must be based on
    accepted source increments and region-integrated contact totals.  Nodal
    CPRESS/COPEN and stress cloud comparisons are allowed only after this gate
    confirms that convergence, cutback/trial authority, contact totals, and
    path-tracking diagnostics are all present in the accepted history.
    """

    source_trial_passed = _row_int_flag(summary, "source_trial_gate_passed", default=1)
    source_convergence_passed = _row_int_flag(summary, "source_convergence_gate_passed", default=0)
    source_line_search_passed = _row_int_flag(summary, "source_active_set_line_search_gate_passed", default=0)
    contact_law_passed = _row_int_flag(summary, "constraint_region_contact_law_gate_passed", default=1)
    tangent_passed = _row_int_flag(summary, "constraint_region_tangent_gate_passed", default=1)
    contact_total_passed = _row_int_flag(summary, "contact_total_gate_passed", default=0)
    nodal_deferred = _row_int_flag(summary, "contact_total_nodal_cpress_deferred", default=0)
    no_nodal_priority_columns = _row_int_flag(summary, "contact_total_no_nodal_priority_columns", default=0)
    active_contact_present = _row_int_flag(summary, "contact_total_active_contact_present", default=0)
    tooth_patch_passed = _row_int_flag(summary, "tooth_patch_region_gate_passed", default=0)
    tooth_patch_ready = _row_int_flag(summary, "tooth_patch_ready_for_cropped_gear_patch", default=0)
    cropped_patch_passed = _row_int_flag(summary, "cropped_patch_gate_passed", default=0)
    cropped_pressure_stress_passed = _row_int_flag(summary, "cropped_patch_pressure_stress_gate_passed", default=0)
    cropped_path_passed = _row_int_flag(summary, "cropped_patch_path_tracking_gate_passed", default=0)
    cropped_active_region_passed = _row_int_flag(
        summary,
        "cropped_patch_active_region_continuity_gate_passed",
        default=0,
    )
    patch_ladder_passed = int(
        bool(tooth_patch_passed)
        and bool(tooth_patch_ready)
        and bool(cropped_patch_passed)
        and bool(cropped_pressure_stress_passed)
        and bool(cropped_path_passed)
        and bool(cropped_active_region_passed)
    )
    path_tracking_passed = _row_int_flag(
        summary,
        "path_tracking_gate_passed",
        default=1 if not bool(active_contact_present) else 0,
    )
    path_continuity_observable = int(
        (not bool(active_contact_present))
        or bool(_row_int_flag(summary, "path_tracking_continuity_observable", default=0))
    )
    path_columns_present = _row_int_flag(summary, "contact_total_path_columns_present", default=0)
    source_final_time_ok = _row_int_flag(summary, "source_convergence_final_time_matches_duration", default=0)
    no_bad_accepted = _row_int_flag(summary, "source_convergence_no_unstable_or_unconverged_accepted", default=0)
    accepted_count = _row_int_flag(summary, "source_accepted_increment_count", default=0)
    expected_count = _row_int_flag(summary, "sfc_increment_count", default=accepted_count)
    strict_sync_min_steps_met = int(accepted_count >= int(min_strict_sync_steps))
    path_tracking_ready = int((not bool(active_contact_present)) or bool(path_columns_present))
    base_gate_passed = int(
        bool(patch_ladder_passed)
        and bool(source_trial_passed)
        and bool(source_convergence_passed)
        and bool(source_line_search_passed)
        and bool(contact_law_passed)
        and bool(tangent_passed)
        and bool(contact_total_passed)
        and bool(path_tracking_passed)
        and bool(path_continuity_observable)
        and bool(nodal_deferred)
        and bool(no_nodal_priority_columns)
        and bool(path_tracking_ready)
        and bool(source_final_time_ok)
        and bool(no_bad_accepted)
    )
    strict_sync_ready = int(bool(base_gate_passed) and bool(strict_sync_min_steps_met))
    contact_window_ready = int(bool(base_gate_passed) and bool(active_contact_present) and bool(path_continuity_observable))
    return {
        "full_gear_entry_gate_passed": base_gate_passed,
        "full_gear_entry_stage": "region_totals_before_nodal_cpress_and_clouds",
        "full_gear_entry_contact_window_gate_passed": contact_window_ready,
        "full_gear_entry_ready_for_nodal_contact_outputs": contact_window_ready,
        "full_gear_entry_ready_for_strict_sync_window": strict_sync_ready,
        "full_gear_entry_ready_for_stress_cloud_comparison": int(
            bool(strict_sync_ready) and bool(contact_window_ready)
        ),
        "full_gear_entry_patch_ladder_gate_passed": int(patch_ladder_passed),
        "full_gear_entry_tooth_patch_region_gate_passed": int(tooth_patch_passed),
        "full_gear_entry_tooth_patch_ready_for_cropped_gear_patch": int(tooth_patch_ready),
        "full_gear_entry_cropped_patch_gate_passed": int(cropped_patch_passed),
        "full_gear_entry_cropped_patch_pressure_stress_gate_passed": int(cropped_pressure_stress_passed),
        "full_gear_entry_cropped_patch_path_tracking_gate_passed": int(cropped_path_passed),
        "full_gear_entry_cropped_patch_active_region_continuity_gate_passed": int(cropped_active_region_passed),
        "full_gear_entry_source_trial_gate_passed": int(source_trial_passed),
        "full_gear_entry_source_convergence_gate_passed": int(source_convergence_passed),
        "full_gear_entry_source_active_set_line_search_gate_passed": int(source_line_search_passed),
        "full_gear_entry_constraint_region_contact_law_gate_passed": int(contact_law_passed),
        "full_gear_entry_constraint_region_tangent_gate_passed": int(tangent_passed),
        "full_gear_entry_contact_total_gate_passed": int(contact_total_passed),
        "full_gear_entry_path_tracking_gate_passed": int(path_tracking_passed),
        "full_gear_entry_path_tracking_continuity_observable": int(path_continuity_observable),
        "full_gear_entry_nodal_cpress_deferred": int(nodal_deferred),
        "full_gear_entry_no_nodal_priority_columns": int(no_nodal_priority_columns),
        "full_gear_entry_active_contact_present": int(active_contact_present),
        "full_gear_entry_path_tracking_ready": int(path_tracking_ready),
        "full_gear_entry_final_time_matches_duration": int(source_final_time_ok),
        "full_gear_entry_no_bad_accepted_state": int(no_bad_accepted),
        "full_gear_entry_accepted_increment_count": int(accepted_count),
        "full_gear_entry_expected_increment_count": int(expected_count),
        "full_gear_entry_min_strict_sync_steps": int(min_strict_sync_steps),
        "full_gear_entry_strict_sync_min_steps_met": int(strict_sync_min_steps_met),
    }


def _artifact_present(summary: Row, key: str) -> int:
    value = summary.get(key)
    return int(value not in ("", None))


def full_gear_evidence_ladder_rows(summary: Row) -> list[Row]:
    """Return the paper-evidence ladder for full-gear validation artifacts.

    Diagnostic plots may exist before all gates pass.  This manifest records
    whether each layer is allowed to support manuscript claims, preserving the
    required order: contact totals first, then nodal CPRESS/COPEN, then stress
    and strain cloud comparisons.
    """

    patch_ladder_ok = _row_int_flag(summary, "full_gear_entry_patch_ladder_gate_passed", default=0)
    source_ok = _row_int_flag(summary, "source_convergence_gate_passed", default=0)
    line_search_ok = _row_int_flag(summary, "source_active_set_line_search_gate_passed", default=0)
    law_ok = _row_int_flag(summary, "constraint_region_contact_law_gate_passed", default=0)
    tangent_ok = _row_int_flag(summary, "constraint_region_tangent_gate_passed", default=0)
    path_ok = _row_int_flag(summary, "path_tracking_gate_passed", default=0)
    totals_ok = _row_int_flag(summary, "contact_total_gate_passed", default=0)
    entry_ok = _row_int_flag(summary, "full_gear_entry_gate_passed", default=0)
    contact_window_ok = _row_int_flag(summary, "full_gear_entry_contact_window_gate_passed", default=0)
    strict_ok = _row_int_flag(summary, "full_gear_entry_ready_for_strict_sync_window", default=0)
    sfc_manifest_present = _artifact_present(summary, "sfc_vtk_manifest")
    abaqus_manifest_present = _artifact_present(summary, "abaqus_vtk_manifest")
    animation_metrics_present = _artifact_present(summary, "animation_metric_errors")
    history_metrics_present = _artifact_present(summary, "history_metric_errors")

    region_allowed = int(
        bool(patch_ladder_ok)
        and bool(source_ok)
        and bool(line_search_ok)
        and bool(law_ok)
        and bool(tangent_ok)
        and bool(path_ok)
        and bool(totals_ok)
    )
    nodal_allowed = int(
        bool(entry_ok)
        and bool(contact_window_ok)
        and bool(sfc_manifest_present)
        and bool(abaqus_manifest_present)
        and bool(animation_metrics_present)
    )
    cloud_allowed = int(
        bool(strict_ok)
        and bool(contact_window_ok)
        and bool(sfc_manifest_present)
        and bool(abaqus_manifest_present)
        and bool(animation_metrics_present)
    )
    history_allowed = int(
        bool(strict_ok) and bool(contact_window_ok) and bool(abaqus_manifest_present) and bool(history_metrics_present)
    )

    def row(
        stage: str,
        *,
        allowed: int,
        required_before: str,
        artifact_key: str,
        blocking_reason: str,
    ) -> Row:
        return {
            "evidence_stage": stage,
            "comparison_order": len(rows) + 1,
            "paper_evidence_allowed": int(allowed),
            "diagnostic_allowed": 1,
            "required_before": required_before,
            "artifact_key": artifact_key,
            "artifact_present": _artifact_present(summary, artifact_key),
            "artifact_path": str(summary.get(artifact_key, "")),
            "blocking_reason": "" if allowed else blocking_reason,
            "source_convergence_gate_passed": int(source_ok),
            "source_active_set_line_search_gate_passed": int(line_search_ok),
            "constraint_region_contact_law_gate_passed": int(law_ok),
            "constraint_region_tangent_gate_passed": int(tangent_ok),
            "path_tracking_gate_passed": int(path_ok),
            "contact_total_gate_passed": int(totals_ok),
            "patch_ladder_gate_passed": int(patch_ladder_ok),
            "full_gear_entry_gate_passed": int(entry_ok),
            "full_gear_strict_sync_ready": int(strict_ok),
        }

    rows: list[Row] = []
    rows.append(
        row(
            "patch_ladder_prerequisites",
            allowed=patch_ladder_ok,
            required_before="full_gear_10_step",
            artifact_key="cropped_patch_gate",
            blocking_reason="tooth_patch_or_cropped_patch_gate_failed",
        )
    )
    rows.append(
        row(
            "region_contact_totals",
            allowed=region_allowed,
            required_before="nodal_cpress_copen",
            artifact_key="contact_total_priority_metrics",
            blocking_reason="patch_ladder_source_convergence_or_contact_total_gate_failed",
        )
    )
    rows.append(
        row(
            "history_vs_abaqus_manifest_totals",
            allowed=history_allowed,
            required_before="stress_strain_clouds",
            artifact_key="history_metric_errors",
            blocking_reason="strict_sync_or_history_manifest_missing",
        )
    )
    rows.append(
        row(
            "nodal_cpress_copen",
            allowed=nodal_allowed,
            required_before="stress_strain_clouds",
            artifact_key="animation_metric_errors",
            blocking_reason="full_gear_entry_gate_or_vtk_manifest_missing",
        )
    )
    rows.append(
        row(
            "stress_strain_clouds",
            allowed=cloud_allowed,
            required_before="solver_timing_claim",
            artifact_key="animation_metric_errors",
            blocking_reason="strict_sync_gate_or_animation_metric_missing",
        )
    )
    return rows


def write_animation_color_ranges(
    out_dir: Path,
    *,
    sfc_manifest: Path | None = None,
    abaqus_manifest: Path | None = None,
) -> Path:
    """Write fixed ParaView color ranges for the SFC/Abaqus animation series."""

    sfc_rows = _read_csv_rows(sfc_manifest or (out_dir / "sfc_vtk" / "sfc_manifest.csv"))
    abaqus_rows = _read_csv_rows(abaqus_manifest or (out_dir / "abaqus_vtk" / "abaqus_manifest.csv"))
    range_rows: list[Row] = [
        {
            "field": "displacement_magnitude",
            "recommended_min": 0.0,
            "recommended_max": max(
                _max_column(sfc_rows, "max_displacement_magnitude"),
                _max_column(abaqus_rows, "max_displacement_magnitude"),
            ),
            "sfc_column": "max_displacement_magnitude",
            "abaqus_column": "max_displacement_magnitude",
            "paraview_note": "Use this fixed range for both SFC and Abaqus; do not rescale every timestep.",
        },
        {
            "field": "von_mises",
            "recommended_min": 0.0,
            "recommended_max": max(_max_column(sfc_rows, "max_von_mises"), _max_column(abaqus_rows, "max_von_mises")),
            "sfc_column": "max_von_mises",
            "abaqus_column": "max_von_mises",
            "paraview_note": "Fixed global range prevents stress-cloud flicker caused by per-frame color rescaling.",
        },
        {
            "field": "von_mises_nodeavg",
            "recommended_min": 0.0,
            "recommended_max": max(
                _max_column(sfc_rows, "max_von_mises_nodeavg"),
                _max_column(abaqus_rows, "max_von_mises_nodeavg"),
            ),
            "sfc_column": "max_von_mises_nodeavg",
            "abaqus_column": "max_von_mises_nodeavg",
            "paraview_note": "Use this point-data field for smooth animations while retaining cell von_mises for quantitative checks.",
        },
        {
            "field": "strain_norm",
            "recommended_min": 0.0,
            "recommended_max": max(_max_column(sfc_rows, "max_strain_norm"), _max_column(abaqus_rows, "max_le_norm")),
            "sfc_column": "max_strain_norm",
            "abaqus_column": "max_le_norm",
            "paraview_note": "SFC field is named strain_norm; Abaqus field is named logarithmic_strain_norm.",
        },
        {
            "field": "strain_norm_nodeavg / logarithmic_strain_norm_nodeavg",
            "recommended_min": 0.0,
            "recommended_max": max(
                _max_column(sfc_rows, "max_strain_norm_nodeavg"),
                _max_column(abaqus_rows, "max_le_norm_nodeavg"),
            ),
            "sfc_column": "max_strain_norm_nodeavg",
            "abaqus_column": "max_le_norm_nodeavg",
            "paraview_note": "Use the node-averaged point-data field for animation; use cell data for exact element statistics.",
        },
        {
            "field": "equivalent_elastic_strain_nodeavg",
            "recommended_min": 0.0,
            "recommended_max": max(
                _max_column(sfc_rows, "max_equivalent_elastic_strain_nodeavg"),
                _max_column(abaqus_rows, "max_equivalent_elastic_strain_nodeavg"),
            ),
            "sfc_column": "max_equivalent_elastic_strain_nodeavg",
            "abaqus_column": "max_equivalent_elastic_strain_nodeavg",
            "paraview_note": "Use for paper-facing strain curves when raw SFC and Abaqus strain tensor measures differ.",
        },
    ]
    path = out_dir / "animation_fixed_color_ranges.csv"
    _write_csv(path, range_rows)
    return path


def write_paraview_animation_setup(
    out_dir: Path,
    *,
    color_ranges: Path,
    sfc_pvd: Path | None = None,
    abaqus_pvd: Path | None = None,
) -> Path:
    """Write a ParaView helper that applies fixed stress/strain color ranges.

    The VTK/PVD files carry the raw fields only; ParaView otherwise tends to
    rescale the color transfer function per timestep.  The generated script is
    post-processing-only and makes the paper animation use the same global
    range and the same node-averaged field at every frame.
    """

    rows = _read_csv_rows(color_ranges)
    ranges: dict[str, tuple[float, float]] = {}
    for row in rows:
        field = str(row.get("field", "")).strip()
        if not field or "/" in field:
            continue
        try:
            ranges[field] = (
                float(row.get("recommended_min", 0.0) or 0.0),
                float(row.get("recommended_max", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            continue
    stable_fields = [
        "displacement_magnitude",
        "von_mises_nodeavg",
        "equivalent_elastic_strain_nodeavg",
    ]
    embedded_ranges = {
        field: ranges.get(field, (0.0, 1.0))
        for field in stable_fields
    }

    def script_relative_path(path: Path | None) -> str:
        if path is None:
            return ""
        return Path(os.path.relpath(Path(path).resolve(), out_dir.resolve())).as_posix()

    sfc_path = script_relative_path(sfc_pvd)
    abaqus_path = script_relative_path(abaqus_pvd)
    script = f'''"""ParaView setup for fixed-range SFC/Abaqus gear animations.

Run with ParaView's Python shell or pvpython from the result directory:

    pvpython {Path("paraview_fixed_range_animation.py").as_posix()}

The script intentionally uses point-data node-averaged fields for stress and
strain animation.  Cell-data fields remain in the VTK files for quantitative
element diagnostics.
"""

from pathlib import Path

from paraview.simple import *  # noqa: F401,F403

SFC_PVD = r"{sfc_path}"
ABAQUS_PVD = r"{abaqus_path}"
FIELD_RANGES = {embedded_ranges!r}
DEFAULT_FIELD = "von_mises_nodeavg"
ROOT = Path(__file__).resolve().parent


def _open_pvd(path):
    return PVDReader(FileName=str((ROOT / path).resolve())) if path else None


def _display(source, view, field_name):
    if source is None:
        return None
    shown = Show(source, view)
    ColorBy(shown, ("POINTS", field_name))
    shown.RescaleTransferFunctionToDataRange(False, True)
    shown.SetScalarBarVisibility(view, True)
    lut = GetColorTransferFunction(field_name)
    vmin, vmax = FIELD_RANGES.get(field_name, (0.0, 1.0))
    lut.RescaleTransferFunction(float(vmin), float(vmax))
    opacity = GetOpacityTransferFunction(field_name)
    opacity.RescaleTransferFunction(float(vmin), float(vmax))
    return shown


view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1400, 850]
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]

sfc = _open_pvd(SFC_PVD)
abaqus = _open_pvd(ABAQUS_PVD)
_display(sfc, view, DEFAULT_FIELD)
_display(abaqus, view, DEFAULT_FIELD)
ResetCamera(view)

print("Loaded fixed-range animation setup.")
print("Stable fields:", ", ".join(FIELD_RANGES))
print("Default field:", DEFAULT_FIELD, FIELD_RANGES.get(DEFAULT_FIELD))
print("Switch DEFAULT_FIELD to displacement_magnitude or equivalent_elastic_strain_nodeavg for other clouds.")
'''
    path = out_dir / "paraview_fixed_range_animation.py"
    path.write_text(script, encoding="utf-8")
    return path


def compare_animation_manifests(
    *,
    sfc_manifest: Path,
    abaqus_manifest: Path,
    out_csv: Path,
    out_png: Path,
) -> list[Row]:
    """Compare SFC and Abaqus VTK animation metrics on the SFC frame times.

    The VTK animation manifests are field-output summaries, not solver inputs.
    They are useful for the paper-facing displacement/stress/strain curves
    because both solvers can be compared from the exact fields that are opened
    in ParaView.
    """

    sfc_rows = _read_csv_rows(sfc_manifest)
    abaqus_rows = _read_csv_rows(abaqus_manifest)
    if not sfc_rows:
        raise RuntimeError(f"SFC VTK manifest has no frames: {sfc_manifest}")
    if not abaqus_rows:
        raise RuntimeError(f"Abaqus VTK manifest has no frames: {abaqus_manifest}")
    t_sfc = np.asarray([float(row.get("time", 0.0) or 0.0) for row in sfc_rows], dtype=float)
    t_abaqus = np.asarray([float(row.get("time", 0.0) or 0.0) for row in abaqus_rows], dtype=float)
    strain_metric = (
        ("p95_equivalent_elastic_strain_nodeavg", "p95_equivalent_elastic_strain_nodeavg", "p95 node-averaged equivalent elastic strain")
        if _manifest_column_available(sfc_rows, "p95_equivalent_elastic_strain_nodeavg")
        and _manifest_column_available(abaqus_rows, "p95_equivalent_elastic_strain_nodeavg")
        else ("max_equivalent_elastic_strain_nodeavg", "max_equivalent_elastic_strain_nodeavg", "max node-averaged equivalent elastic strain")
        if _manifest_column_available(sfc_rows, "max_equivalent_elastic_strain_nodeavg")
        and _manifest_column_available(abaqus_rows, "max_equivalent_elastic_strain_nodeavg")
        else ("max_strain_norm_nodeavg", "max_le_norm_nodeavg", "max node-averaged strain norm")
    )
    stress_metric = (
        ("p95_von_mises_nodeavg", "p95_von_mises_nodeavg", "p95 node-averaged von Mises")
        if _manifest_column_available(sfc_rows, "p95_von_mises_nodeavg")
        and _manifest_column_available(abaqus_rows, "p95_von_mises_nodeavg")
        else ("max_von_mises_nodeavg", "max_von_mises_nodeavg", "max node-averaged von Mises")
    )
    metrics = [
        ("max_displacement_magnitude", "max_displacement_magnitude", "max displacement magnitude"),
        stress_metric,
        strain_metric,
    ]
    diagnostic_metrics = list(metrics)
    for sfc_key, abaqus_key, label in (
        ("max_von_mises_nodeavg", "max_von_mises_nodeavg", "max node-averaged von Mises"),
        (
            "max_equivalent_elastic_strain_nodeavg",
            "max_equivalent_elastic_strain_nodeavg",
            "max node-averaged equivalent elastic strain",
        ),
        ("mean_von_mises_nodeavg", "mean_von_mises_nodeavg", "mean node-averaged von Mises"),
        (
            "mean_equivalent_elastic_strain_nodeavg",
            "mean_equivalent_elastic_strain_nodeavg",
            "mean node-averaged equivalent elastic strain",
        ),
    ):
        if _manifest_column_available(sfc_rows, sfc_key) and _manifest_column_available(abaqus_rows, abaqus_key):
            diagnostic_metrics.append((sfc_key, abaqus_key, label))
    for sfc_key, abaqus_key, label in (
        ("rp1_rotation_z_rad", "rp1_rotation_z_rad", "RP1 rotation about z"),
        ("rp2_rotation_z_rad", "rp2_rotation_z_rad", "RP2 rotation about z"),
        ("rp1_angular_velocity_z_rad_per_s", "rp1_angular_velocity_z_rad_per_s", "RP1 angular velocity about z"),
        ("rp2_angular_velocity_z_rad_per_s", "rp2_angular_velocity_z_rad_per_s", "RP2 angular velocity about z"),
    ):
        if _manifest_column_available(sfc_rows, sfc_key) and _manifest_column_available(abaqus_rows, abaqus_key):
            diagnostic_metrics.append((sfc_key, abaqus_key, label))
    for object_id in (1, 2):
        for sfc_key, abaqus_key, label in (
            (
                f"max_displacement_magnitude_object{object_id}",
                f"max_displacement_magnitude_object{object_id}",
                f"max displacement magnitude object {object_id}",
            ),
            (
                f"max_von_mises_nodeavg_object{object_id}",
                f"max_von_mises_nodeavg_object{object_id}",
                f"max node-averaged von Mises object {object_id}",
            ),
            (
                f"p95_von_mises_nodeavg_object{object_id}",
                f"p95_von_mises_nodeavg_object{object_id}",
                f"p95 node-averaged von Mises object {object_id}",
            ),
            (
                f"max_equivalent_elastic_strain_nodeavg_object{object_id}",
                f"max_equivalent_elastic_strain_nodeavg_object{object_id}",
                f"max node-averaged equivalent elastic strain object {object_id}",
            ),
            (
                f"p95_equivalent_elastic_strain_nodeavg_object{object_id}",
                f"p95_equivalent_elastic_strain_nodeavg_object{object_id}",
                f"p95 node-averaged equivalent elastic strain object {object_id}",
            ),
        ):
            if _manifest_column_available(sfc_rows, sfc_key) and _manifest_column_available(abaqus_rows, abaqus_key):
                diagnostic_metrics.append((sfc_key, abaqus_key, label))
    rows: list[Row] = []
    for frame, (t_value, sfc_row) in enumerate(zip(t_sfc, sfc_rows, strict=True)):
        row: Row = {"frame": int(frame), "time": float(t_value)}
        for sfc_key, abaqus_key, _label in diagnostic_metrics:
            sfc_value = float(sfc_row.get(sfc_key, 0.0) or 0.0)
            abaqus_series = np.asarray([float(item.get(abaqus_key, 0.0) or 0.0) for item in abaqus_rows], dtype=float)
            abaqus_value = float(np.interp(float(t_value), t_abaqus, abaqus_series))
            row[f"sfc_{sfc_key}"] = sfc_value
            row[f"abaqus_{abaqus_key}"] = abaqus_value
            row[f"{sfc_key}_abs_error"] = abs(sfc_value - abaqus_value)
            row[f"{sfc_key}_rel_error"] = abs(sfc_value - abaqus_value) / max(abs(abaqus_value), 1.0e-12)
        rows.append(row)
    _write_csv(out_csv, rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.75), constrained_layout=True)
    for ax, (sfc_key, abaqus_key, label) in zip(axes, metrics, strict=True):
        sfc_y = np.asarray([float(row.get(sfc_key, 0.0) or 0.0) for row in sfc_rows], dtype=float)
        abaqus_y = np.asarray([float(row.get(abaqus_key, 0.0) or 0.0) for row in abaqus_rows], dtype=float)
        interp_abaqus_y = np.asarray(
            [float(row[f"abaqus_{abaqus_key}"]) for row in rows],
            dtype=float,
        )
        rel_key = f"{sfc_key}_rel_error"
        final_error = float(rows[-1].get(rel_key, 0.0)) if rows else 0.0
        ax.plot(t_sfc, interp_abaqus_y, color="#1f77b4", linewidth=1.7, label="Abaqus/Standard")
        if t_abaqus.size <= 80:
            in_window = t_abaqus <= (float(np.max(t_sfc)) if t_sfc.size else 0.0) + 1.0e-14
            if np.any(in_window):
                ax.plot(
                    t_abaqus[in_window],
                    abaqus_y[in_window],
                    color="#1f77b4",
                    linestyle="none",
                    marker="o",
                    markersize=2.5,
                    alpha=0.75,
                )
        ax.plot(
            t_sfc,
            sfc_y,
            color="#d95f02",
            linewidth=1.7,
            linestyle="--",
            label=f"SFC ({final_error * 100:.2f}% final err.)",
        )
        if t_sfc.size > 1:
            ax.fill_between(
                t_sfc,
                np.minimum(sfc_y, interp_abaqus_y),
                np.maximum(sfc_y, interp_abaqus_y),
                color="#d95f02",
                alpha=0.10,
                linewidth=0.0,
            )
        ax.set_title(label, fontsize=9.5)
        ax.set_xlabel("time (s)", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.legend(loc="best", fontsize=7.5, frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return rows


def compare_sfc_history_to_abaqus_manifest(
    *,
    sfc_history: Path,
    abaqus_manifest: Path,
    out_csv: Path,
) -> list[Row]:
    """Compare SFC scalar history with Abaqus VTK manifest metrics by time.

    This avoids writing large SFC VTK files for every short diagnostic run while
    still checking the displacement, stress, strain, and contact fields against
    the already-exported Abaqus reference frames.
    """

    sfc_rows = _read_csv_rows(sfc_history)
    abaqus_rows = _read_csv_rows(abaqus_manifest)
    if not sfc_rows or not abaqus_rows:
        _write_csv(out_csv, [])
        return []
    t_abaqus = np.asarray([float(row.get("time", 0.0) or 0.0) for row in abaqus_rows], dtype=float)
    def sfc_key(preferred: str, fallback: str) -> str:
        return preferred if _manifest_column_available(sfc_rows, preferred) else fallback

    metric_pairs = [
        ("max_displacement_norm", "max_displacement_magnitude"),
        ("p95_von_mises_nodeavg", "p95_von_mises_nodeavg"),
        ("p95_equivalent_elastic_strain_nodeavg", "p95_equivalent_elastic_strain_nodeavg"),
        ("max_von_mises_nodeavg", "max_von_mises_nodeavg"),
        ("max_equivalent_elastic_strain_nodeavg", "max_equivalent_elastic_strain_nodeavg"),
        ("mean_von_mises_nodeavg", "mean_von_mises_nodeavg"),
        ("mean_equivalent_elastic_strain_nodeavg", "mean_equivalent_elastic_strain_nodeavg"),
        (sfc_key("active_contact_secondary_node_count", "active_contact_node_count"), "active_contact_node_count"),
        (sfc_key("max_contact_secondary_pressure_nodeavg", "max_contact_pressure_nodeavg"), "max_contact_pressure_nodeavg"),
        (sfc_key("p95_contact_secondary_pressure_nodeavg", "p95_contact_pressure_nodeavg"), "p95_contact_pressure_nodeavg"),
        (
            sfc_key("mean_active_contact_secondary_pressure_nodeavg", "mean_active_contact_pressure_nodeavg"),
            "mean_active_contact_pressure_nodeavg",
        ),
    ]
    rows: list[Row] = []
    for frame, sfc_row in enumerate(sfc_rows):
        t_value = float(sfc_row.get("time", 0.0) or 0.0)
        row: Row = {"frame": int(frame), "time": t_value}
        for sfc_key, abaqus_key in metric_pairs:
            if sfc_key not in sfc_row or not _manifest_column_available(abaqus_rows, abaqus_key):
                continue
            try:
                sfc_value = float(sfc_row.get(sfc_key, 0.0) or 0.0)
                abaqus_series = np.asarray([float(item.get(abaqus_key, 0.0) or 0.0) for item in abaqus_rows], dtype=float)
            except (TypeError, ValueError):
                continue
            abaqus_value = float(np.interp(t_value, t_abaqus, abaqus_series))
            row[f"sfc_{sfc_key}"] = sfc_value
            row[f"abaqus_{abaqus_key}"] = abaqus_value
            row[f"{sfc_key}_abs_error"] = abs(sfc_value - abaqus_value)
            row[f"{sfc_key}_rel_error"] = abs(sfc_value - abaqus_value) / max(abs(abaqus_value), 1.0e-12)
        rows.append(row)
    _write_csv(out_csv, rows)
    return rows


def _selected_surface_entries_for_full_mesh(
    mesh: GearMesh,
    entries: tuple[tuple[int, str], ...],
    selected_surface_ids: np.ndarray,
) -> tuple[tuple[int, str], ...]:
    """Map selected original Abaqus surface entries to emitted full-mesh labels."""

    out: list[tuple[int, str]] = []
    for sid in np.asarray(selected_surface_ids, dtype=np.int64).reshape(-1):
        element_label, side = entries[int(sid)]
        new_element_label = int(mesh.element_label_to_index[int(element_label)]) + 1
        out.append((new_element_label, str(side).upper()))
    return tuple(out)


def _hub_indices(mesh: GearMesh, labels: tuple[int, ...]) -> np.ndarray:
    ids = [mesh.label_to_index[int(label)] for label in labels if int(label) in mesh.label_to_index]
    if not ids:
        raise ValueError("hub node set did not map to this gear mesh")
    return np.asarray(sorted(set(ids)), dtype=np.int64)


def _nearest_ids(tree: cKDTree, point: np.ndarray, *, k: int, count: int) -> np.ndarray:
    query_count = min(max(1, int(k)), max(1, int(count)))
    return np.asarray(tree.query(point, k=query_count)[1], dtype=np.int64).reshape(-1)


def _radius_expanded_ids(
    centroids: np.ndarray,
    seed: np.ndarray,
    base_ids: np.ndarray,
    *,
    radius_factor: float,
    extra_radius: float = 0.0,
) -> np.ndarray:
    ids = np.asarray(base_ids, dtype=np.int64).reshape(-1)
    if (float(radius_factor) <= 1.0 and float(extra_radius) <= 0.0) or ids.size == 0:
        return np.unique(ids)
    offsets = np.linalg.norm(centroids[ids] - seed, axis=1)
    base_radius = float(np.max(offsets)) if offsets.size else 0.0
    if base_radius <= 0.0 and centroids.shape[0] > 1:
        distances = np.linalg.norm(centroids - seed, axis=1)
        positive = distances[distances > 0.0]
        base_radius = float(np.min(positive)) if positive.size else 0.0
    radius = max(base_radius * float(radius_factor), base_radius) + max(float(extra_radius), 0.0)
    expanded = np.flatnonzero(np.linalg.norm(centroids - seed, axis=1) <= radius + 1.0e-12)
    return np.unique(np.concatenate([ids, expanded.astype(np.int64)]))


def _nearest_active_surface_ids(
    model: GearInputModel,
    *,
    active_faces_per_body: int,
    active_patch_radius_factor: float = 1.0,
    active_patch_sweep_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    centroids1 = _face_centroids(model.gear1.nodes, model.gear1_contact_faces)
    centroids2 = _face_centroids(model.gear2.nodes, model.gear2_contact_faces)
    tree2 = cKDTree(centroids2)
    distances, ids2 = tree2.query(centroids1, k=1)
    seed1 = int(np.argmin(distances))
    seed2 = int(ids2[seed1])
    if int(active_faces_per_body) <= 0:
        ids1 = np.arange(centroids1.shape[0], dtype=np.int64)
        ids2_active = np.arange(centroids2.shape[0], dtype=np.int64)
        c1 = centroids1[seed1]
        c2 = centroids2[seed2]
        drive = c2 - c1
        drive /= max(float(np.linalg.norm(drive)), 1.0e-30)
        return ids1, ids2_active, float(distances[seed1]), drive
    n = int(active_faces_per_body)
    tree1 = cKDTree(centroids1)
    ids1 = _nearest_ids(tree1, centroids1[seed1], k=n, count=centroids1.shape[0])
    ids2_active = _nearest_ids(tree2, centroids2[seed2], k=n, count=centroids2.shape[0])
    ids1 = _radius_expanded_ids(
        centroids1,
        centroids1[seed1],
        ids1,
        radius_factor=active_patch_radius_factor,
        extra_radius=active_patch_sweep_radius,
    )
    ids2_active = _radius_expanded_ids(
        centroids2,
        centroids2[seed2],
        ids2_active,
        radius_factor=active_patch_radius_factor,
        extra_radius=active_patch_sweep_radius,
    )
    if float(active_patch_radius_factor) > 1.0:
        opposite_count = min(4, centroids2.shape[0])
        mapped2 = np.asarray(tree2.query(centroids1[ids1], k=opposite_count)[1], dtype=np.int64).reshape(-1)
        ids2_active = np.unique(np.concatenate([ids2_active, mapped2]))
        opposite_count = min(4, centroids1.shape[0])
        mapped1 = np.asarray(tree1.query(centroids2[ids2_active], k=opposite_count)[1], dtype=np.int64).reshape(-1)
        ids1 = np.unique(np.concatenate([ids1, mapped1]))
    c1 = np.mean(centroids1[ids1], axis=0)
    c2 = np.mean(centroids2[ids2_active], axis=0)
    drive = c2 - c1
    drive /= max(float(np.linalg.norm(drive)), 1.0e-30)
    return ids1, ids2_active, float(distances[seed1]), drive


def _source_dynamic_contact_window_sweep_radius(model: GearInputModel, *, duration: float) -> float:
    """Return a geometry-derived swept-patch radius for source-drive contact.

    The full source contact surface is too expensive for every diagnostic run,
    but a static initial patch can miss the contact region after finite sliding.
    This radius expands the initial closest patch by the arc length swept by the
    prescribed source RP rotation over the requested time window.  It is only a
    broad-phase contact-window size; the final gap, normal, payload, and
    pressure law still come from the current surface query and constraint-region
    contact formulation.
    """

    angle = abs(float(model.gear1_angular_velocity_z)) * max(float(duration), 0.0)
    if angle <= 0.0:
        return 0.0
    centroids = [
        _face_centroids(model.gear1.nodes, model.gear1_contact_faces),
        _face_centroids(model.gear2.nodes, model.gear2_contact_faces),
    ]
    rps = [np.asarray(model.rp1, dtype=float).reshape(3), np.asarray(model.rp2, dtype=float).reshape(3)]
    radii: list[float] = []
    for points, rp in zip(centroids, rps, strict=True):
        if points.size == 0:
            continue
        distances = np.linalg.norm(np.asarray(points, dtype=float) - rp.reshape(1, 3), axis=1)
        distances = distances[np.isfinite(distances)]
        if distances.size:
            radii.append(float(np.percentile(distances, 95.0)))
    if not radii:
        return 0.0
    return 1.25 * angle * max(radii)


def build_full_active_pair(
    model: GearInputModel,
    *,
    active_faces_per_body: int,
    active_patch_radius_factor: float = 1.0,
    active_patch_sweep_radius: float = 0.0,
) -> CroppedGearPair:
    """Return a complete two-gear volume model with active contact surfaces."""

    ids1, ids2, initial_gap, drive = _nearest_active_surface_ids(
        model,
        active_faces_per_body=active_faces_per_body,
        active_patch_radius_factor=active_patch_radius_factor,
        active_patch_sweep_radius=active_patch_sweep_radius,
    )
    gear1_faces = _orient_faces_toward(model.gear1.nodes, model.gear1_contact_faces[ids1], drive)
    gear2_faces = _orient_faces_toward(model.gear2.nodes, model.gear2_contact_faces[ids2], -drive)
    gear1 = CroppedGearPatch(
        name="gear1",
        nodes=model.gear1.nodes,
        elements=model.gear1.elements,
        contact_faces=gear1_faces,
        support_nodes=_hub_indices(model.gear1, model.gear1_hub_labels),
        surface_entries=_selected_surface_entries_for_full_mesh(model.gear1, model.gear1_surface_entries, ids1),
        element_labels=np.arange(1, model.gear1.elements.shape[0] + 1, dtype=np.int64),
        rp=model.rp1,
    )
    gear2 = CroppedGearPatch(
        name="gear2",
        nodes=model.gear2.nodes,
        elements=model.gear2.elements,
        contact_faces=gear2_faces,
        support_nodes=_hub_indices(model.gear2, model.gear2_hub_labels),
        surface_entries=_selected_surface_entries_for_full_mesh(model.gear2, model.gear2_surface_entries, ids2),
        element_labels=np.arange(1, model.gear2.elements.shape[0] + 1, dtype=np.int64),
        rp=model.rp2,
    )
    return CroppedGearPair(gear1, gear2, drive, initial_gap)


def write_full_summary(path: Path, summary: Row, history_path: Path, *, abaqus_row: Row | None = None) -> None:
    lines = [
        "# Full Gear Lagrangian-SDF Alignment",
        "",
        "This run keeps the full gear volume meshes and uses an active contact surface patch.",
        "",
        f"- nodes/elements: {summary['nodes']} / {summary['elements']}",
        f"- active contact faces: {summary['gear1_contact_faces']} / {summary['gear2_contact_faces']}",
        f"- requested active faces per body: {summary.get('requested_active_faces_per_body', '')}",
        f"- effective active faces per body: {summary.get('effective_active_faces_per_body', '')}",
        f"- active patch radius factor: {float(summary.get('active_patch_radius_factor', 1.0)):.3f}",
        f"- support nodes: {summary['gear1_support_nodes']} / {summary['gear2_support_nodes']}",
        f"- linear solver: {summary.get('linear_solver', 'sparse')}",
        f"- contact mode: {summary.get('contact_mode', '')}",
        f"- source contact averaging: {summary.get('source_contact_averaging', '')}",
        f"- source contact kinematics: {summary.get('source_contact_kinematics', '')}",
        f"- source contact normal filter: {summary.get('source_contact_normal_filter', '')}",
        f"- source contact direction: {summary.get('source_contact_direction', '')}",
        f"- source contact projection: {summary.get('source_contact_projection', '')}",
        f"- source contact pair order: {summary.get('source_contact_pair_order', '')}",
        f"- source contact search radius: {summary.get('source_contact_search_radius', '')}",
        f"- source secondary normal min projection: {summary.get('source_secondary_normal_min_projection', '')}",
        f"- source secondary line hard distance limit: {summary.get('source_secondary_line_hard_distance_limit', '')}",
        f"- source secondary path tracking: {summary.get('source_secondary_path_tracking', '')}",
        f"- source dynamic contact window: {summary.get('source_dynamic_contact_window', '')}",
        f"- source dynamic contact window applied: {summary.get('source_dynamic_contact_window_applied', '')}",
        f"- source dynamic contact sweep radius: {summary.get('source_dynamic_contact_window_sweep_radius', '')}",
        f"- source active-set stability: {summary.get('source_contact_active_set_stability', '')}",
        f"- source max nonlinear iterations: {summary.get('source_max_iterations', '')}",
        f"- source convergence gate: {summary.get('source_convergence_gate_passed', '')}",
        f"- source active-set line-search gate: {summary.get('source_active_set_line_search_gate_passed', '')}",
        f"- source converged steps: {summary.get('source_step_converged_count', '')}",
        f"- source iteration-limit steps: {summary.get('source_iteration_limit_reached_count', '')}",
        f"- source unstable accepted steps: {summary.get('source_unstable_accepted_count', '')}",
        f"- source line-search unstable count: {summary.get('source_line_search_unstable_count', '')}",
        f"- source constraint-region tangent solves: {summary.get('source_constraint_region_tangent_solve_count', '')}",
        f"- constraint-region contact-law gate: {summary.get('constraint_region_contact_law_gate_passed', '')}",
        f"- constraint-region tangent gate: {summary.get('constraint_region_tangent_gate_passed', '')}",
        f"- accepted-state path tracking gate: {summary.get('path_tracking_gate_passed', '')}",
        f"- contact-total gate: {summary.get('contact_total_gate_passed', '')}",
        f"- full-gear entry gate: {summary.get('full_gear_entry_gate_passed', '')}",
        "- full-gear strict-sync ready: "
        f"{summary.get('full_gear_entry_ready_for_strict_sync_window', '')}",
        f"- full-gear evidence manifest: {summary.get('full_gear_evidence_ladder', '')}",
        f"- source contact footprint clipping: {summary.get('source_contact_footprint_clipping', '')}",
        f"- source internal kinematics: {summary.get('source_internal_kinematics', '')}",
        f"- source rotating inertia: {summary.get('source_rotating_inertia', '')}",
        f"- penalty solver: {summary.get('penalty_solver', '')}",
        f"- material linearization: {summary.get('material_linearization', '')}",
        f"- RP reaction definition: {summary.get('rp_reaction_definition', '')}",
        f"- SFC wall time: {float(summary['sfc_wall_seconds']):.6f} s",
        f"- timing internal+tangent: {float(summary.get('timing_internal_tangent_seconds', 0.0)):.6f} s",
        f"- timing effective system: {float(summary.get('timing_effective_system_seconds', 0.0)):.6f} s",
        f"- timing hard-contact solve: {float(summary.get('timing_hard_contact_solve_seconds', 0.0)):.6f} s",
        f"- timing contact linearization: {float(summary.get('timing_contact_linearization_seconds', 0.0)):.6f} s",
        f"- timing penalty base tangent: {float(summary.get('timing_base_tangent_seconds', 0.0)):.6f} s",
        f"- timing penalty residual: {float(summary.get('timing_penalty_residual_seconds', 0.0)):.6f} s",
        f"- timing penalty low-rank solve: {float(summary.get('timing_penalty_low_rank_seconds', 0.0)):.6f} s",
        f"- final max displacement norm: {float(summary.get('final_max_displacement_norm', 0.0)):.6e}",
        f"- final p95 von Mises: {float(summary.get('final_p95_von_mises', 0.0)):.6e}",
        f"- final p95 equivalent elastic strain: {float(summary.get('final_p95_equivalent_elastic_strain', 0.0)):.6e}",
        f"- final p95 node-averaged von Mises: {float(summary.get('final_p95_von_mises_nodeavg', 0.0)):.6e}",
        "- final p95 node-averaged equivalent elastic strain: "
        f"{float(summary.get('final_p95_equivalent_elastic_strain_nodeavg', 0.0)):.6e}",
        f"- final RP reaction norm: {float(summary.get('final_rp_force_norm', 0.0)):.6e}",
        f"- final min gap: {float(summary.get('final_min_gap', 0.0)):.6e}",
        f"- rotation rate about z: {float(summary.get('rotation_rate_z_rad_per_s', 0.0)):.6e} rad/s",
        f"- final rotation about z: {float(summary.get('final_rotation_z_rad', 0.0)):.6e} rad",
        f"- history CSV: `{history_path.name}`",
    ]
    if summary.get("source_convergence_gate"):
        lines.append(f"- source convergence gate CSV: `{Path(str(summary.get('source_convergence_gate'))).name}`")
    if summary.get("source_active_set_line_search_gate"):
        lines.append(
            "- source active-set line-search gate CSV: "
            f"`{Path(str(summary.get('source_active_set_line_search_gate'))).name}`"
        )
    if summary.get("constraint_region_contact_law_gate"):
        lines.append(
            f"- constraint-region contact-law gate CSV: `{Path(str(summary.get('constraint_region_contact_law_gate'))).name}`"
        )
    if summary.get("constraint_region_tangent_gate"):
        lines.append(
            f"- constraint-region tangent gate CSV: `{Path(str(summary.get('constraint_region_tangent_gate'))).name}`"
        )
    if summary.get("path_tracking_gate"):
        lines.append(f"- accepted-state path tracking gate CSV: `{Path(str(summary.get('path_tracking_gate'))).name}`")
    if summary.get("contact_total_priority_metrics"):
        lines.append(f"- contact-total priority metrics: `{Path(str(summary.get('contact_total_priority_metrics'))).name}`")
    if summary.get("contact_total_gate"):
        lines.append(f"- contact-total gate CSV: `{Path(str(summary.get('contact_total_gate'))).name}`")
    if summary.get("full_gear_entry_gate"):
        lines.append(f"- full-gear entry gate CSV: `{Path(str(summary.get('full_gear_entry_gate'))).name}`")
    if summary.get("full_gear_evidence_ladder"):
        lines.append(f"- full-gear evidence ladder CSV: `{Path(str(summary.get('full_gear_evidence_ladder'))).name}`")
    if summary.get("source_increment_trials"):
        lines.append(f"- source increment trial ledger: `{Path(str(summary.get('source_increment_trials'))).name}`")
    if summary.get("sfc_vtk_pvd"):
        lines.extend(
            [
                f"- SFC VTK PVD: `{summary.get('sfc_vtk_pvd')}`",
                f"- SFC VTK frame count: {int(summary.get('sfc_vtk_frame_count', 0))}",
            ]
        )
    if abaqus_row is not None:
        lines.extend(
            [
                "",
                "## Abaqus Native-Contact Alignment",
                "",
                f"- Abaqus analysis wall time: {float(abaqus_row.get('abaqus_analysis_wall_seconds', 0.0)):.6f} s",
                f"- Abaqus export wall time: {float(abaqus_row.get('abaqus_export_wall_seconds', 0.0)):.6f} s",
                f"- alignment errors CSV: `{Path(str(summary.get('alignment_errors', ''))).name}`",
                f"- alignment figure: `{Path(str(summary.get('alignment_figure', ''))).name}`",
            ]
        )
    if summary.get("abaqus_vtk_pvd"):
        lines.extend(
            [
                "",
                "## VTK Animation",
                "",
                f"- Abaqus VTK PVD: `{summary.get('abaqus_vtk_pvd')}`",
                f"- Abaqus VTK frame count: {int(summary.get('abaqus_vtk_frame_count', 0))}",
            ]
        )
    if summary.get("animation_color_ranges"):
        lines.extend(
            [
                "",
                "## ParaView Color Ranges",
                "",
                f"- fixed color ranges: `{Path(str(summary.get('animation_color_ranges'))).name}`",
                f"- fixed-range setup script: `{Path(str(summary.get('paraview_animation_setup', ''))).name}`",
                "- Use fixed global ranges for stress/strain animations; per-frame auto-rescaling can look like stress flicker.",
            ]
        )
    if summary.get("animation_metric_errors"):
        lines.extend(
            [
                "",
                "## VTK Field-Curve Alignment",
                "",
                f"- VTK metric errors: `{Path(str(summary.get('animation_metric_errors'))).name}`",
                f"- VTK metric curves: `{Path(str(summary.get('animation_metric_figure'))).name}`",
                "- Curves use the same displacement/stress/strain fields written to the SFC and Abaqus VTK animations.",
            ]
        )
    if summary.get("history_metric_errors"):
        lines.extend(
            [
                "",
                "## History Field-Curve Alignment",
                "",
                f"- history metric errors: `{Path(str(summary.get('history_metric_errors'))).name}`",
                "- This lightweight check compares SFC scalar history against Abaqus VTK manifest fields without writing SFC VTK frames.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_abaqus_vtk_frames(
    *,
    out_dir: Path,
    abaqus_command: str | None,
    frame_stride: int,
    include_tensors: bool,
    young: float | None = None,
    poisson: float | None = None,
) -> Row:
    """Export the existing full-gear Abaqus ODB to a strided VTK/PVD series."""

    run_dir = out_dir / "abaqus_run"
    odb = run_dir / "cropped_gear_alignment.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    vtk_dir = out_dir / "abaqus_vtk"
    command = _resolve_abaqus_command(abaqus_command)
    script = ROOT / "validation" / "abaqus_odb_to_vtk.py"
    export_command = [
        command,
        "python",
        str(script.resolve()),
        "--odb",
        str(odb.resolve()),
        "--out-dir",
        str(vtk_dir.resolve()),
        "--stem",
        "abaqus",
        "--frame-stride",
        str(max(1, int(frame_stride))),
    ]
    if young is not None and poisson is not None:
        export_command.extend(["--young", f"{float(young):.16e}", "--poisson", f"{float(poisson):.16e}"])
    export_command.append("--include-tensors" if include_tensors else "--scalars-only")
    wall = _run_command(
        export_command,
        cwd=run_dir,
        log_path=out_dir / "abaqus_vtk_export_stdout.log",
    )
    manifest = vtk_dir / "abaqus_manifest.csv"
    frame_count = 0
    if manifest.exists():
        with manifest.open("r", newline="", encoding="ascii") as handle:
            frame_count = max(0, sum(1 for _ in handle) - 1)
    return {
        "abaqus_vtk_export_wall_seconds": float(wall),
        "abaqus_vtk_pvd": str(vtk_dir / "abaqus.pvd"),
        "abaqus_vtk_manifest": str(manifest),
        "abaqus_vtk_frame_count": int(frame_count),
        "abaqus_vtk_frame_stride": int(frame_stride),
        "abaqus_vtk_include_tensors": bool(include_tensors),
    }


def run_full_gear(
    *,
    source: Path,
    out_dir: Path,
    active_faces_per_body: int,
    active_patch_radius_factor: float,
    duration: float,
    dt: float,
    target_overclosure: float,
    rotation_rate_z: float,
    pressure_stiffness: float,
    hard_max_iterations: int,
    contact_mode: str,
    run_abaqus: bool,
    drive_mode: str = "closure",
    hht_alpha: float = ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
    tet4_mass_kind: str = "consistent",
    history_frame_stride: int = 1,
    use_source_timing: bool = False,
    abaqus_command: str | None = None,
    write_sfc_vtk: bool = False,
    sfc_vtk_dir: Path | None = None,
    vtk_frame_stride: int = 1,
    vtk_include_tensors: bool = True,
    source_stress_postprocess: str = "linear_corotated",
    source_contact_averaging: str = "slave_node_region_constraint",
    source_contact_kinematics: str = "finite_rp_corotated",
    source_contact_normal_filter: str = "opposing",
    source_contact_direction: str = "secondary_average",
    source_contact_projection: str = "secondary_line",
    source_contact_pair_order: str = "gear2_slave",
    source_contact_search_radius: float | None = None,
    source_secondary_normal_min_projection: float = 0.0,
    source_secondary_line_distance_limit: float | None = None,
    source_secondary_line_hard_distance_limit: float | None = None,
    source_secondary_path_tracking: bool = True,
    source_dynamic_contact_window: bool = False,
    source_contact_active_set_stability: bool = True,
    source_contact_footprint_clipping: bool = False,
    source_internal_kinematics: str = "finite_stvk_visual",
    source_rotating_inertia: str = "finite_kinematic",
    source_max_iterations: int = 16,
    source_residual_tolerance: float = 5.0e-3,
    source_correction_tolerance: float = 1.0e-2,
    source_contact_force_increment_tolerance: float = 1.0e-2,
    source_accept_unconverged: bool = False,
    source_cutback_factor: float = 0.5,
    source_min_cutback_dt: float | None = None,
    source_checkpoint_path: Path | None = None,
    resume_source_checkpoint: bool = False,
    source_checkpoint_stride: int = 10,
    export_abaqus_vtk: bool = False,
    abaqus_vtk_manifest: Path | None = None,
    patch_prerequisite_summaries: Sequence[Path] | None = None,
) -> tuple[list[Row], Row]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = parse_gear_input(source)
    drive = str(drive_mode).lower()
    if drive not in {"closure", "source_inp"}:
        raise ValueError("drive_mode must be 'closure' or 'source_inp'")
    source_step_matched = bool(use_source_timing)
    if source_step_matched:
        duration = float(model.dynamic_duration)
        dt = float(model.dynamic_initial_dt)
    dynamic_contact_window_applied = bool(source_dynamic_contact_window) and drive == "source_inp"
    dynamic_contact_window_sweep_radius = (
        _source_dynamic_contact_window_sweep_radius(model, duration=duration) if dynamic_contact_window_applied else 0.0
    )
    effective_active_faces_per_body = int(active_faces_per_body)
    pair = build_full_active_pair(
        model,
        active_faces_per_body=effective_active_faces_per_body,
        active_patch_radius_factor=active_patch_radius_factor,
        active_patch_sweep_radius=dynamic_contact_window_sweep_radius,
    )
    mode = str(contact_mode).lower()
    if mode not in {"penalty", "hard"}:
        raise ValueError("contact_mode must be 'penalty' or 'hard'")
    if drive == "source_inp":
        if mode == "hard":
            raise ValueError("source_inp drive currently supports SFC penalty contact; use --contact-mode penalty")
        history, summary = solve_sfc_source_drive_pair(
            pair,
            young=model.young,
            poisson=model.poisson,
            density=model.density,
            pressure_stiffness=pressure_stiffness,
            duration=duration,
            dt=dt,
            gear1_angular_velocity_z=model.gear1_angular_velocity_z,
            gear2_torque_z=model.gear2_torque_z,
            hht_alpha=float(hht_alpha),
            tet4_mass_kind=str(tet4_mass_kind),
            max_iterations=max(1, int(source_max_iterations)),
            vtk_out_dir=(sfc_vtk_dir if sfc_vtk_dir is not None else out_dir / "sfc_vtk") if write_sfc_vtk else None,
            vtk_frame_stride=max(1, int(vtk_frame_stride)),
            history_frame_stride=max(1, int(history_frame_stride)),
            vtk_stem="sfc",
            vtk_include_tensors=bool(vtk_include_tensors),
            source_stress_postprocess=str(source_stress_postprocess),
            source_contact_averaging=str(source_contact_averaging),
            source_contact_kinematics=str(source_contact_kinematics),
            source_contact_normal_filter=str(source_contact_normal_filter),
            source_contact_direction=str(source_contact_direction),
            source_contact_projection=str(source_contact_projection),
            source_contact_pair_order=str(source_contact_pair_order),
            source_contact_search_radius=source_contact_search_radius,
            source_secondary_normal_min_projection=float(source_secondary_normal_min_projection),
            source_secondary_line_distance_limit=source_secondary_line_distance_limit,
            source_secondary_line_hard_distance_limit=source_secondary_line_hard_distance_limit,
            source_secondary_path_tracking=bool(source_secondary_path_tracking),
            source_contact_active_set_stability=bool(source_contact_active_set_stability),
            source_contact_footprint_clipping=bool(source_contact_footprint_clipping),
            source_internal_kinematics=str(source_internal_kinematics),
            source_rotating_inertia=str(source_rotating_inertia),
            source_residual_tolerance=float(source_residual_tolerance),
            source_correction_tolerance=float(source_correction_tolerance),
            source_contact_force_increment_tolerance=float(source_contact_force_increment_tolerance),
            source_accept_unconverged=bool(source_accept_unconverged),
            source_cutback_factor=float(source_cutback_factor),
            source_min_cutback_dt=source_min_cutback_dt,
            source_checkpoint_path=source_checkpoint_path,
            resume_source_checkpoint=bool(resume_source_checkpoint),
            source_checkpoint_stride=int(source_checkpoint_stride),
        )
    elif mode == "hard":
        history, summary = solve_sfc_cropped_pair_hard_contact(
            pair,
            young=model.young,
            poisson=model.poisson,
            density=model.density,
            pressure_stiffness=pressure_stiffness,
            duration=duration,
            dt=dt,
            target_overclosure=target_overclosure,
            rotation_rate_z=rotation_rate_z,
            max_iterations=hard_max_iterations,
            hard_enforcement="abaqus_standard_penalty",
            constraint_averaging="slave_face",
            hht_alpha=float(hht_alpha),
            linear_solver="sparse",
        )
    else:
        history, summary = solve_sfc_cropped_pair(
            pair,
            young=model.young,
            poisson=model.poisson,
            density=model.density,
            pressure_stiffness=pressure_stiffness,
            duration=duration,
            dt=dt,
            target_overclosure=target_overclosure,
            rotation_rate_z=rotation_rate_z,
            hht_alpha=float(hht_alpha),
            penalty_solver="modified_newton",
            material_linearization="reference_linear",
            vtk_out_dir=(sfc_vtk_dir if sfc_vtk_dir is not None else out_dir / "sfc_vtk") if write_sfc_vtk else None,
            vtk_frame_stride=max(1, int(vtk_frame_stride)),
            vtk_stem="sfc",
            vtk_include_tensors=bool(vtk_include_tensors),
        )
    if history:
        summary["final_max_displacement_norm"] = float(history[-1].get("max_displacement_norm", 0.0))
        summary["final_p95_von_mises"] = float(history[-1].get("p95_von_mises", 0.0))
        summary["final_p95_equivalent_elastic_strain"] = float(history[-1].get("p95_equivalent_elastic_strain", 0.0))
        summary["final_p95_von_mises_nodeavg"] = float(history[-1].get("p95_von_mises_nodeavg", 0.0))
        summary["final_p95_equivalent_elastic_strain_nodeavg"] = float(
            history[-1].get("p95_equivalent_elastic_strain_nodeavg", 0.0)
        )
    summary["active_patch_radius_factor"] = float(active_patch_radius_factor)
    summary["requested_active_faces_per_body"] = int(active_faces_per_body)
    summary["effective_active_faces_per_body"] = int(effective_active_faces_per_body)
    summary["source_dynamic_contact_window"] = int(bool(source_dynamic_contact_window))
    summary["source_dynamic_contact_window_applied"] = int(bool(dynamic_contact_window_applied))
    summary["source_dynamic_contact_window_sweep_radius"] = float(dynamic_contact_window_sweep_radius)
    summary["drive_mode"] = drive
    summary["sfc_match_source_step"] = bool(source_step_matched)
    summary["sfc_duration"] = float(duration)
    summary["sfc_dt"] = float(dt)
    summary["sfc_increment_count"] = int(round(float(duration) / float(dt))) if float(dt) > 0.0 else 0
    summary["source_dynamic_initial_dt"] = float(model.dynamic_initial_dt)
    summary["source_dynamic_duration"] = float(model.dynamic_duration)
    summary["source_dynamic_min_dt"] = float(model.dynamic_min_dt)
    summary["source_dynamic_max_dt"] = float(model.dynamic_max_dt)
    summary["source_contact_pressure_overclosure"] = str(model.contact_pressure_overclosure)
    summary["rotation_rate_z_rad_per_s"] = float(rotation_rate_z)
    summary["final_rotation_z_rad"] = float(rotation_rate_z) * float(duration)
    if drive == "source_inp":
        summary["rotation_rate_z_rad_per_s"] = float(model.gear1_angular_velocity_z)
        summary["final_rotation_z_rad"] = float(model.gear1_angular_velocity_z) * float(duration)
        summary["gear2_torque_z"] = float(model.gear2_torque_z)
    history_path = out_dir / "sfc_full_gear_lagrangian_sdf_history.csv"
    _write_csv(history_path, history)
    source_increment_trial_rows = summary.pop("_source_increment_trial_rows", None)
    if isinstance(source_increment_trial_rows, list):
        trial_path = out_dir / "sfc_source_increment_trials.csv"
        _write_csv(trial_path, source_increment_trial_rows)
        summary["source_increment_trials"] = str(trial_path)
        trial_gate = source_increment_trial_gate_metrics(history, source_increment_trial_rows)
        trial_gate_path = out_dir / "sfc_source_increment_trial_gate.csv"
        _write_csv(trial_gate_path, [trial_gate])
        summary.update(trial_gate)
        summary["source_increment_trial_gate"] = str(trial_gate_path)
    contact_total_priority_path = out_dir / "sfc_contact_total_priority_metrics.csv"
    write_contact_total_priority_csv(history, contact_total_priority_path)
    summary["contact_total_priority_metrics"] = str(contact_total_priority_path)
    contact_total_gate = contact_total_priority_gate_metrics(_read_csv_rows(contact_total_priority_path))
    contact_total_gate_path = out_dir / "sfc_contact_total_gate.csv"
    _write_csv(contact_total_gate_path, [contact_total_gate])
    summary.update(contact_total_gate)
    summary["contact_total_gate"] = str(contact_total_gate_path)
    source_convergence_gate = source_convergence_gate_metrics(summary, history)
    source_convergence_gate_path = out_dir / "sfc_source_convergence_gate.csv"
    _write_csv(source_convergence_gate_path, [source_convergence_gate])
    summary.update(source_convergence_gate)
    summary["source_convergence_gate"] = str(source_convergence_gate_path)
    source_line_search_gate = source_active_set_line_search_gate_metrics(summary, history)
    source_line_search_gate_path = out_dir / "sfc_source_active_set_line_search_gate.csv"
    _write_csv(source_line_search_gate_path, [source_line_search_gate])
    summary.update(source_line_search_gate)
    summary["source_active_set_line_search_gate"] = str(source_line_search_gate_path)
    contact_law_gate = constraint_region_contact_law_gate_metrics(history)
    contact_law_gate_path = out_dir / "sfc_constraint_region_contact_law_gate.csv"
    _write_csv(contact_law_gate_path, [contact_law_gate])
    summary.update(contact_law_gate)
    summary["constraint_region_contact_law_gate"] = str(contact_law_gate_path)
    constraint_tangent_gate = constraint_region_tangent_gate_metrics(summary, history)
    constraint_tangent_gate_path = out_dir / "sfc_constraint_region_tangent_gate.csv"
    _write_csv(constraint_tangent_gate_path, [constraint_tangent_gate])
    summary.update(constraint_tangent_gate)
    summary["constraint_region_tangent_gate"] = str(constraint_tangent_gate_path)
    path_tracking_gate = path_tracking_gate_metrics(history)
    path_tracking_gate_path = out_dir / "sfc_path_tracking_gate.csv"
    _write_csv(path_tracking_gate_path, [path_tracking_gate])
    summary.update(path_tracking_gate)
    summary["path_tracking_gate"] = str(path_tracking_gate_path)
    patch_prerequisite_summary = load_patch_prerequisite_summary(patch_prerequisite_summaries or [])
    if patch_prerequisite_summary:
        summary.update(patch_prerequisite_summary)
    full_gear_entry_gate = full_gear_entry_gate_metrics(summary)
    full_gear_entry_gate_path = out_dir / "sfc_full_gear_entry_gate.csv"
    _write_csv(full_gear_entry_gate_path, [full_gear_entry_gate])
    summary.update(full_gear_entry_gate)
    summary["full_gear_entry_gate"] = str(full_gear_entry_gate_path)
    deck_path = out_dir / "abaqus_full_gear_alignment.inp"
    _write_abaqus_alignment_deck(
        deck_path,
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=pressure_stiffness,
        duration=duration,
        dt=dt,
        target_overclosure=target_overclosure,
        rotation_rate_z=rotation_rate_z,
        contact_mode=mode,
    )
    abaqus_row: Row | None = None
    error_rows: list[Row] | None = None
    figure_path: Path | None = None
    if run_abaqus:
        abaqus_metrics, abaqus_row = run_abaqus_alignment(
            deck_path,
            out_dir,
            pair=pair,
            young=model.young,
            poisson=model.poisson,
            abaqus_command=abaqus_command,
        )
        summary.update(abaqus_row)
        _write_csv(out_dir / "abaqus_runtime.csv", [abaqus_row])
        error_rows = compare_histories(history_path, abaqus_metrics, out_dir / "sfc_vs_abaqus_alignment_errors.csv")
        figure_path = out_dir / "sfc_vs_abaqus_alignment_curves.png"
        plot_alignment_curves(history_path, abaqus_metrics, error_rows, figure_path)
        summary["alignment_errors"] = str(out_dir / "sfc_vs_abaqus_alignment_errors.csv")
        summary["alignment_figure"] = str(figure_path)
    if export_abaqus_vtk:
        summary.update(
            export_abaqus_vtk_frames(
                out_dir=out_dir,
                abaqus_command=abaqus_command,
                frame_stride=max(1, int(vtk_frame_stride)),
                include_tensors=bool(vtk_include_tensors),
                young=float(model.young),
                poisson=float(model.poisson),
            )
        )
    if abaqus_vtk_manifest is not None:
        external_manifest = Path(abaqus_vtk_manifest)
        summary["abaqus_vtk_manifest"] = str(external_manifest)
        summary["abaqus_vtk_pvd"] = str(external_manifest.with_name("abaqus.pvd"))
        summary["abaqus_vtk_frame_count"] = int(len(_read_csv_rows(external_manifest)))
    if summary.get("sfc_vtk_manifest") and summary.get("abaqus_vtk_manifest"):
        animation_errors = out_dir / "sfc_vs_abaqus_vtk_metric_errors.csv"
        animation_figure = out_dir / "sfc_vs_abaqus_vtk_metric_curves.png"
        compare_animation_manifests(
            sfc_manifest=Path(str(summary["sfc_vtk_manifest"])),
            abaqus_manifest=Path(str(summary["abaqus_vtk_manifest"])),
            out_csv=animation_errors,
            out_png=animation_figure,
        )
        summary["animation_metric_errors"] = str(animation_errors)
        summary["animation_metric_figure"] = str(animation_figure)
    if summary.get("abaqus_vtk_manifest"):
        history_errors = out_dir / "sfc_vs_abaqus_history_metric_errors.csv"
        compare_sfc_history_to_abaqus_manifest(
            sfc_history=history_path,
            abaqus_manifest=Path(str(summary["abaqus_vtk_manifest"])),
            out_csv=history_errors,
        )
        summary["history_metric_errors"] = str(history_errors)
    if summary.get("sfc_vtk_manifest") or summary.get("abaqus_vtk_manifest"):
        color_ranges = write_animation_color_ranges(
            out_dir,
            sfc_manifest=Path(str(summary["sfc_vtk_manifest"])) if summary.get("sfc_vtk_manifest") else None,
            abaqus_manifest=Path(str(summary["abaqus_vtk_manifest"])) if summary.get("abaqus_vtk_manifest") else None,
        )
        summary["animation_color_ranges"] = str(color_ranges)
        summary["paraview_animation_setup"] = str(
            write_paraview_animation_setup(
                out_dir,
                color_ranges=color_ranges,
                sfc_pvd=Path(str(summary["sfc_vtk_pvd"])) if summary.get("sfc_vtk_pvd") else None,
                abaqus_pvd=Path(str(summary["abaqus_vtk_pvd"])) if summary.get("abaqus_vtk_pvd") else None,
            )
        )
    evidence_ladder_path = out_dir / "sfc_full_gear_evidence_ladder.csv"
    _write_csv(evidence_ladder_path, full_gear_evidence_ladder_rows(summary))
    summary["full_gear_evidence_ladder"] = str(evidence_ladder_path)
    write_full_summary(out_dir / "full_gear_lagrangian_sdf_summary.md", summary, history_path, abaqus_row=abaqus_row)
    _write_csv(out_dir / "full_gear_lagrangian_sdf_summary.csv", [summary])
    return history, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--active-faces-per-body", type=int, default=16)
    parser.add_argument("--active-patch-radius-factor", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=2.5e-4)
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--overclosure", type=float, default=1.0e-5)
    parser.add_argument(
        "--rotation-rate-z",
        "--rotation-rate-z-rad-s",
        dest="rotation_rate_z",
        type=float,
        default=0.0,
        help="Prescribed G1 RP angular velocity about z in radians per second.",
    )
    parser.add_argument("--pressure-stiffness", type=float, default=5.0e9)
    parser.add_argument("--contact-mode", choices=("penalty", "hard"), default="penalty")
    parser.add_argument("--drive-mode", choices=("closure", "source_inp"), default="closure")
    parser.add_argument("--hht-alpha", type=float, default=ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA)
    parser.add_argument("--tet4-mass-kind", choices=("consistent", "calculix"), default="consistent")
    parser.add_argument("--history-frame-stride", type=int, default=2)
    parser.add_argument(
        "--use-source-timing",
        "--match-source-step",
        dest="use_source_timing",
        action="store_true",
        help="Use the *Dynamic step time and initial increment parsed from the source inp deck.",
    )
    parser.add_argument("--hard-max-iterations", type=int, default=4)
    parser.add_argument("--run-abaqus", action="store_true")
    parser.add_argument("--abaqus-command", type=str, default=None)
    parser.add_argument("--write-sfc-vtk", action="store_true")
    parser.add_argument("--sfc-vtk-dir", type=Path, default=None)
    parser.add_argument("--vtk-frame-stride", type=int, default=2)
    parser.add_argument("--vtk-scalars-only", action="store_true")
    parser.add_argument(
        "--source-stress-postprocess",
        choices=("linear_corotated", "finite_stvk_visual"),
        default="linear_corotated",
        help="Source-drive SFC stress/strain output mode; does not alter the solve.",
    )
    parser.add_argument(
        "--source-contact-averaging",
        choices=(
            "none",
            "slave_face",
            "slave_node",
            "slave_node_region",
            "slave_node_point",
            "surface_patch",
            "slave_face_constraint",
            "slave_node_constraint",
            "slave_node_region_constraint",
            "surface_patch_constraint",
            "slave_face_area_average",
            "slave_node_area_average",
            "slave_node_region_area_average",
            "surface_patch_area_average",
            "slave_face_participation",
            "slave_node_participation",
            "slave_node_region_participation",
            "surface_patch_participation",
            "slave_face_signed_participation",
            "slave_node_signed_participation",
            "slave_node_region_signed_participation",
            "surface_patch_signed_participation",
        ),
        default="slave_node_region_constraint",
        help="Optional source-drive contact constraint averaging for Abaqus-style surface-to-surface penalty diagnostics.",
    )
    parser.add_argument(
        "--source-contact-kinematics",
        choices=("linearized_mpc", "finite_rp_corotated"),
        default="finite_rp_corotated",
        help="Kinematic map used to evaluate source-drive contact geometry.",
    )
    parser.add_argument(
        "--source-contact-normal-filter",
        choices=("none", "opposing", "opposing_search"),
        default="opposing",
        help="Optional source-drive master/slave normal compatibility filter.",
    )
    parser.add_argument(
        "--source-contact-direction",
        choices=("master", "secondary_average"),
        default="secondary_average",
        help="Normal direction used by source-drive surface-to-surface contact constraints.",
    )
    parser.add_argument(
        "--source-contact-projection",
        choices=("closest_feature", "secondary_plane", "secondary_line"),
        default="secondary_line",
        help="Gap projection used with secondary-average contact direction.",
    )
    parser.add_argument(
        "--source-contact-pair-order",
        choices=("gear2_slave", "gear1_slave", "symmetric_two_pass"),
        default="gear2_slave",
        help="Source-drive contact pair orientation diagnostic.",
    )
    parser.add_argument(
        "--source-contact-search-radius",
        type=float,
        default=None,
        help="Optional source-drive broad-phase radius; closest-feature projection still determines the final gap.",
    )
    parser.add_argument(
        "--source-secondary-normal-min-projection",
        type=float,
        default=0.0,
        help=(
            "Minimum opposing normal projection for secondary-normal line constraints. "
            "Values above zero reject near-tangential line intersections that do not "
            "form a well-conditioned Abaqus-style surface-to-surface constraint."
        ),
    )
    parser.add_argument(
        "--source-secondary-line-distance-limit",
        type=float,
        default=None,
        help=(
            "Optional accepted signed distance for source secondary-line projection. "
            "A negative value disables this final line-distance gate."
        ),
    )
    parser.add_argument(
        "--source-secondary-line-hard-distance-limit",
        type=float,
        default=None,
        help=(
            "Optional hard maximum distance for secondary-normal line constraints. "
            "A negative value disables the hard finite-sliding tracking tube."
        ),
    )
    parser.add_argument(
        "--source-secondary-path-tracking",
        dest="source_secondary_path_tracking",
        action="store_true",
        default=True,
        help=(
            "Use previous accepted secondary-normal anchor faces as path-tracking "
            "hints for Abaqus-style finite-sliding surface-to-surface contact."
        ),
    )
    parser.add_argument(
        "--no-source-secondary-path-tracking",
        dest="source_secondary_path_tracking",
        action="store_false",
        help="Disable secondary-normal path-tracking hints for diagnostics.",
    )
    parser.add_argument(
        "--source-dynamic-contact-window",
        action="store_true",
        help=(
            "For source-drive gear validation, expand the initial active patch "
            "by the source RP rotation sweep so the current broad phase and "
            "path tracking, rather than a static initial crop, determine the "
            "moving contact window."
        ),
    )
    parser.add_argument(
        "--source-contact-active-set-stability",
        dest="source_contact_active_set_stability",
        action="store_true",
        default=True,
        help=(
            "Require one stable active contact signature before accepting a "
            "source-drive nonlinear iteration."
        ),
    )
    parser.add_argument(
        "--no-source-contact-active-set-stability",
        dest="source_contact_active_set_stability",
        action="store_false",
        help="Disable the Abaqus-style source-drive active contact status stability gate for diagnostics.",
    )
    parser.add_argument(
        "--source-contact-footprint-clipping",
        action="store_true",
        help="Clip slave triangle contact support to the projected master footprint before pressure integration.",
    )
    parser.add_argument(
        "--source-internal-kinematics",
        choices=("linearized_mpc", "corotated_rp", "finite_stvk_visual"),
        default="finite_stvk_visual",
        help="Internal elastic residual map for source-drive large RP rotations.",
    )
    parser.add_argument(
        "--source-rotating-inertia",
        choices=("none", "centripetal", "finite_kinematic"),
        default="finite_kinematic",
        help="Optional finite-RP inertia residual for source-drive dynamics.",
    )
    parser.add_argument(
        "--source-max-iterations",
        type=int,
        default=16,
        help="Maximum nonlinear iterations per fixed source-drive time increment; 16 matches Abaqus/Standard's common equilibrium iteration cap.",
    )
    parser.add_argument(
        "--source-residual-tolerance",
        type=float,
        default=5.0e-3,
        help="Abaqus-style normalized residual gate for source-drive equilibrium iterations.",
    )
    parser.add_argument(
        "--source-correction-tolerance",
        type=float,
        default=1.0e-2,
        help="Abaqus-style normalized displacement-correction gate for source-drive equilibrium iterations.",
    )
    parser.add_argument(
        "--source-contact-force-increment-tolerance",
        type=float,
        default=1.0e-2,
        help="Abaqus-style normalized contact-force increment gate for source-drive equilibrium iterations.",
    )
    parser.add_argument(
        "--source-accept-unconverged",
        action="store_true",
        help="Diagnostic compatibility mode: accept source-drive increments that hit the iteration limit.",
    )
    parser.add_argument(
        "--source-cutback-factor",
        type=float,
        default=0.5,
        help="Trial time-increment reduction factor reported when source-drive convergence gates request a cutback.",
    )
    parser.add_argument(
        "--source-min-cutback-dt",
        type=float,
        default=None,
        help="Minimum source-drive cutback increment used for cutback diagnostics.",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=None,
        help="Optional source-drive checkpoint path for fixed-step continuation.",
    )
    parser.add_argument(
        "--resume-source-checkpoint",
        action="store_true",
        help="Resume source-drive SFC solve from --source-checkpoint.",
    )
    parser.add_argument("--source-checkpoint-stride", type=int, default=10)
    parser.add_argument("--export-abaqus-vtk", action="store_true")
    parser.add_argument(
        "--abaqus-vtk-manifest",
        type=Path,
        default=None,
        help="Existing Abaqus VTK manifest to compare against the generated SFC VTK field curves.",
    )
    parser.add_argument(
        "--patch-prerequisite-summary",
        action="append",
        type=Path,
        default=[],
        help=(
            "Patch-ladder summary CSV produced by tooth/cropped validations. "
            "May be passed multiple times; only recognized prerequisite gate "
            "columns are forwarded to the full-gear entry gate."
        ),
    )
    args = parser.parse_args(argv)
    history, summary = run_full_gear(
        source=args.source,
        out_dir=args.out_dir,
        active_faces_per_body=int(args.active_faces_per_body),
        active_patch_radius_factor=float(args.active_patch_radius_factor),
        duration=float(args.duration),
        dt=float(args.dt),
        target_overclosure=float(args.overclosure),
        rotation_rate_z=float(args.rotation_rate_z),
        pressure_stiffness=float(args.pressure_stiffness),
        hard_max_iterations=int(args.hard_max_iterations),
        contact_mode=str(args.contact_mode),
        run_abaqus=bool(args.run_abaqus),
        drive_mode=str(args.drive_mode),
        hht_alpha=float(args.hht_alpha),
        tet4_mass_kind=str(args.tet4_mass_kind),
        history_frame_stride=int(args.history_frame_stride),
        use_source_timing=bool(args.use_source_timing),
        abaqus_command=args.abaqus_command,
        write_sfc_vtk=bool(args.write_sfc_vtk),
        sfc_vtk_dir=args.sfc_vtk_dir,
        vtk_frame_stride=int(args.vtk_frame_stride),
        vtk_include_tensors=not bool(args.vtk_scalars_only),
        source_stress_postprocess=str(args.source_stress_postprocess),
        source_contact_averaging=str(args.source_contact_averaging),
        source_contact_kinematics=str(args.source_contact_kinematics),
        source_contact_normal_filter=str(args.source_contact_normal_filter),
        source_contact_direction=str(args.source_contact_direction),
        source_contact_projection=str(args.source_contact_projection),
        source_contact_pair_order=str(args.source_contact_pair_order),
        source_contact_search_radius=args.source_contact_search_radius,
        source_secondary_normal_min_projection=float(args.source_secondary_normal_min_projection),
        source_secondary_line_distance_limit=args.source_secondary_line_distance_limit,
        source_secondary_line_hard_distance_limit=args.source_secondary_line_hard_distance_limit,
        source_secondary_path_tracking=bool(args.source_secondary_path_tracking),
        source_dynamic_contact_window=bool(args.source_dynamic_contact_window),
        source_contact_active_set_stability=bool(args.source_contact_active_set_stability),
        source_contact_footprint_clipping=bool(args.source_contact_footprint_clipping),
        source_internal_kinematics=str(args.source_internal_kinematics),
        source_rotating_inertia=str(args.source_rotating_inertia),
        source_max_iterations=int(args.source_max_iterations),
        source_residual_tolerance=float(args.source_residual_tolerance),
        source_correction_tolerance=float(args.source_correction_tolerance),
        source_contact_force_increment_tolerance=float(args.source_contact_force_increment_tolerance),
        source_accept_unconverged=bool(args.source_accept_unconverged),
        source_cutback_factor=float(args.source_cutback_factor),
        source_min_cutback_dt=args.source_min_cutback_dt,
        source_checkpoint_path=args.source_checkpoint,
        resume_source_checkpoint=bool(args.resume_source_checkpoint),
        source_checkpoint_stride=int(args.source_checkpoint_stride),
        export_abaqus_vtk=bool(args.export_abaqus_vtk),
        abaqus_vtk_manifest=args.abaqus_vtk_manifest,
        patch_prerequisite_summaries=args.patch_prerequisite_summary,
    )
    print((args.out_dir / "full_gear_lagrangian_sdf_summary.md").read_text(encoding="utf-8"))
    return 0 if history and summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
