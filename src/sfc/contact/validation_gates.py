"""Validation gates for contact-result comparison order.

These helpers encode comparison-order invariants used by validation scripts:
region-level totals must be checked before nodal CPRESS/COPEN clouds.  They do
not depend on Abaqus files or external solver output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

Row = Mapping[str, Any]

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

CONTACT_TOTAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "contact_region_normal_force",
    "contact_region_virtual_work",
    "contact_region_energy",
    "contact_active_area",
    "active_contact_region_count",
)

CONTACT_TOTAL_PATH_COLUMNS: tuple[str, ...] = (
    "contact_path_cache_hit_fraction",
    "contact_path_cache_match_fraction",
    "contact_master_face_switch_fraction",
    "contact_master_face_topological_continuity_fraction",
    "contact_master_face_invalid_jump_fraction",
    "contact_master_barycentric_drift_max",
)

CONTACT_PATH_TRACKING_REQUIRED_COLUMNS: tuple[str, ...] = (
    *CONTACT_TOTAL_PATH_COLUMNS,
    "contact_active_region_jaccard",
    "contact_active_region_persistence_fraction",
)

CONSTRAINT_REGION_CONTACT_LAW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "contact_constraint_law_source",
    "contact_constraint_open_closed_source",
    "contact_constraint_active_status_source",
    "contact_constraint_normal_source",
    "contact_constraint_region_area_source",
    "contact_constraint_force_distribution",
    "contact_constraint_secondary_node_regions_present",
    "contact_constraint_master_payload_present",
    "contact_constraint_area_positive",
    "contact_constraint_independent_quadrature_penalty_disabled",
)

CONSTRAINT_REGION_TANGENT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "contact_tangent_source",
    "contact_tangent_gap_jacobian_source",
    "contact_tangent_pressure_derivative",
    "contact_tangent_pressure_scale_filter",
    "contact_tangent_pressure_positive_scale_count",
    "contact_tangent_slave_gap_derivative_source",
    "contact_tangent_master_gap_derivative_source",
    "contact_tangent_slave_gap_derivative_nnz",
    "contact_tangent_master_gap_derivative_nnz",
    "contact_tangent_slave_master_gap_derivative_present",
    "contact_tangent_sign_convention",
    "contact_tangent_fixed_active_set",
    "contact_tangent_active_region_count",
    "contact_tangent_scale_sum",
)


def contact_total_priority_gate_metrics(priority_rows: Sequence[Row]) -> dict[str, float | int | str]:
    """Gate total contact quantities before nodal CPRESS/COPEN checks."""

    active_rows: list[Row] = []
    for row in priority_rows:
        active_regions = _row_int_flag(row, "active_contact_region_count", default=0)
        active_area = _finite_row_float(row, "contact_active_area") or 0.0
        normal_force = abs(_finite_row_float(row, "contact_region_normal_force", "normal_force") or 0.0)
        if active_regions > 0 or active_area > 0.0 or normal_force > 0.0:
            active_rows.append(row)

    has_rows = len(priority_rows) > 0
    nodal_deferred = all(_row_int_flag(row, "nodal_cpress_deferred", default=0) == 1 for row in priority_rows)
    no_nodal_columns = all(
        not _row_has_value(row, column)
        for row in priority_rows
        for column in NODAL_CONTACT_DIAGNOSTIC_COLUMNS
    )
    total_columns_present = True
    path_columns_present = True
    secondary_pressure_from_region = True
    legacy_pressure_alias_from_secondary = True
    force_consistent = True
    max_force_mismatch = 0.0
    for row in active_rows:
        total_columns_present = total_columns_present and all(
            _row_has_value(row, key) for key in CONTACT_TOTAL_REQUIRED_COLUMNS
        )
        path_columns_present = path_columns_present and all(_row_has_value(row, key) for key in CONTACT_TOTAL_PATH_COLUMNS)
        secondary_pressure_from_region = (
            secondary_pressure_from_region
            and str(row.get("contact_secondary_pressure_recovery_source", "")) == "constraint_region"
        )
        legacy_pressure_alias_from_secondary = (
            legacy_pressure_alias_from_secondary
            and str(row.get("contact_pressure_recovery_source", "")) == "constraint_region"
            and str(row.get("contact_legacy_pressure_alias_source", "")) == "secondary_constraint_region"
            and _row_int_flag(row, "contact_legacy_pressure_alias_matches_secondary", default=0) == 1
        )
        normal_force = _finite_row_float(row, "normal_force")
        region_force = _finite_row_float(row, "contact_region_normal_force")
        if normal_force is not None and region_force is not None:
            mismatch = abs(float(normal_force) - float(region_force))
            scale = max(1.0, abs(float(normal_force)), abs(float(region_force)))
            max_force_mismatch = max(max_force_mismatch, mismatch / scale)
            force_consistent = force_consistent and mismatch <= 1.0e-6 * scale

    active_contact_present = len(active_rows) > 0
    active_total_gate = (not active_contact_present) or (
        total_columns_present
        and path_columns_present
        and secondary_pressure_from_region
        and legacy_pressure_alias_from_secondary
        and force_consistent
    )
    gate_passed = int(has_rows and nodal_deferred and no_nodal_columns and active_total_gate)
    return {
        "contact_total_gate_passed": gate_passed,
        "comparison_stage": "region_totals_before_nodal_cpress",
        "contact_total_row_count": int(len(priority_rows)),
        "contact_total_active_row_count": int(len(active_rows)),
        "contact_total_active_contact_present": int(active_contact_present),
        "contact_total_nodal_cpress_deferred": int(nodal_deferred),
        "contact_total_no_nodal_priority_columns": int(no_nodal_columns),
        "contact_total_required_columns_present": int(total_columns_present),
        "contact_total_path_columns_present": int(path_columns_present),
        "contact_total_secondary_pressure_recovery_from_region": int(secondary_pressure_from_region),
        "contact_total_legacy_pressure_alias_from_secondary": int(legacy_pressure_alias_from_secondary),
        "contact_total_force_consistency_passed": int(force_consistent),
        "contact_total_force_relative_mismatch_max": float(max_force_mismatch),
    }


def contact_path_tracking_gate_metrics(
    history_rows: Sequence[Row],
    *,
    min_cache_hit_fraction: float = 0.95,
    min_cache_match_fraction: float = 0.0,
    min_active_region_jaccard: float = 0.25,
    min_active_region_persistence: float = 0.25,
    min_topological_continuity_fraction: float = 0.95,
    max_master_face_switch_fraction: float = 0.85,
    max_master_face_invalid_jump_fraction: float = 0.0,
    max_master_barycentric_drift: float = 1.50,
) -> dict[str, float | int | str]:
    """Gate accepted-state path tracking before nodal pressure/cloud checks.

    The check is deliberately region-level.  Active accepted rows must carry
    cache-hit/cache-match rates, master-face switch and topological continuity,
    active-region persistence, and barycentric drift diagnostics.  A single
    active row is not enough to prove temporal continuity; it is reported as
    not yet observable so the full-gear ladder cannot use a one-frame contact
    event as evidence for smooth sliding path tracking.
    """

    active_rows = [row for row in history_rows if _row_has_active_contact(row)]
    active_contact_present = len(active_rows) > 0
    tracking_rows = active_rows[1:] if len(active_rows) > 1 else []
    required_columns = CONTACT_PATH_TRACKING_REQUIRED_COLUMNS
    columns_present = all(_row_has_value(row, key) for row in tracking_rows for key in required_columns)
    continuity_observable = len(tracking_rows) > 0
    if tracking_rows and columns_present:
        cache_hit_min = min(_finite_row_float(row, "contact_path_cache_hit_fraction") or 0.0 for row in tracking_rows)
        cache_match_min = min(_finite_row_float(row, "contact_path_cache_match_fraction") or 0.0 for row in tracking_rows)
        face_switch_max = max(
            _finite_row_float(row, "contact_master_face_switch_fraction") or 0.0 for row in tracking_rows
        )
        topological_continuity_min = min(
            _finite_row_float(row, "contact_master_face_topological_continuity_fraction") or 0.0
            for row in tracking_rows
        )
        invalid_jump_max = max(
            _finite_row_float(row, "contact_master_face_invalid_jump_fraction") or 0.0
            for row in tracking_rows
        )
        active_jaccard_min = min(_finite_row_float(row, "contact_active_region_jaccard") or 0.0 for row in tracking_rows)
        active_persistence_min = min(
            _finite_row_float(row, "contact_active_region_persistence_fraction") or 0.0
            for row in tracking_rows
        )
        barycentric_drift_max = max(
            _finite_row_float(row, "contact_master_barycentric_drift_max") or 0.0 for row in tracking_rows
        )
    else:
        cache_hit_min = 1.0 if not active_contact_present else 0.0
        cache_match_min = 1.0 if not active_contact_present else 0.0
        face_switch_max = 0.0
        topological_continuity_min = 1.0 if not active_contact_present else 0.0
        invalid_jump_max = 0.0
        active_jaccard_min = 1.0 if not active_contact_present else 0.0
        active_persistence_min = 1.0 if not active_contact_present else 0.0
        barycentric_drift_max = 0.0
    cache_hit_ok = cache_hit_min >= float(min_cache_hit_fraction)
    cache_match_ok = cache_match_min >= float(min_cache_match_fraction)
    face_switch_ok = face_switch_max <= float(max_master_face_switch_fraction)
    topological_ok = topological_continuity_min >= float(min_topological_continuity_fraction)
    invalid_jump_ok = invalid_jump_max <= float(max_master_face_invalid_jump_fraction)
    active_jaccard_ok = active_jaccard_min >= float(min_active_region_jaccard)
    active_persistence_ok = active_persistence_min >= float(min_active_region_persistence)
    barycentric_ok = barycentric_drift_max <= float(max_master_barycentric_drift)
    if not active_contact_present:
        gate_passed = 1
    elif not continuity_observable:
        gate_passed = 0
    else:
        gate_passed = int(
            columns_present
            and cache_hit_ok
            and cache_match_ok
            and face_switch_ok
            and topological_ok
            and invalid_jump_ok
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
        "path_tracking_topological_continuity_gate_passed": int(topological_ok),
        "path_tracking_invalid_jump_gate_passed": int(invalid_jump_ok),
        "path_tracking_active_region_jaccard_gate_passed": int(active_jaccard_ok),
        "path_tracking_active_region_persistence_gate_passed": int(active_persistence_ok),
        "path_tracking_barycentric_drift_gate_passed": int(barycentric_ok),
        "path_tracking_cache_hit_fraction_min_after_first_active": float(cache_hit_min),
        "path_tracking_cache_match_fraction_min_after_first_active": float(cache_match_min),
        "path_tracking_master_face_switch_fraction_max_after_first_active": float(face_switch_max),
        "path_tracking_master_face_topological_continuity_min_after_first_active": float(
            topological_continuity_min
        ),
        "path_tracking_master_face_invalid_jump_fraction_max_after_first_active": float(invalid_jump_max),
        "path_tracking_active_region_jaccard_min_after_first_active": float(active_jaccard_min),
        "path_tracking_active_region_persistence_min_after_first_active": float(active_persistence_min),
        "path_tracking_master_barycentric_drift_max_after_first_active": float(barycentric_drift_max),
        "path_tracking_min_cache_hit_threshold": float(min_cache_hit_fraction),
        "path_tracking_min_cache_match_threshold": float(min_cache_match_fraction),
        "path_tracking_min_active_region_jaccard_threshold": float(min_active_region_jaccard),
        "path_tracking_min_active_region_persistence_threshold": float(min_active_region_persistence),
        "path_tracking_min_topological_continuity_threshold": float(min_topological_continuity_fraction),
        "path_tracking_max_master_face_switch_threshold": float(max_master_face_switch_fraction),
        "path_tracking_max_master_face_invalid_jump_threshold": float(max_master_face_invalid_jump_fraction),
        "path_tracking_max_master_barycentric_drift_threshold": float(max_master_barycentric_drift),
    }


def constraint_region_contact_law_gate_metrics(history_rows: Sequence[Row]) -> dict[str, float | int | str]:
    """Gate that active accepted contact uses secondary constraint regions.

    Active rows must prove the surface-to-surface contact law is assembled from
    slave/secondary constraint regions: signed area-average region gaps decide
    open/closed status, area-average region normals define the contact
    direction, region tributary area scales the pressure-overclosure law, and
    slave shape-function plus master closest-feature payload distribute equal
    and opposite forces.  Rows that still represent each quadrature point as an
    independent node-to-surface penalty are rejected.
    """

    active_rows = [row for row in history_rows if _row_has_active_contact(row)]
    active_contact_present = len(active_rows) > 0
    row_columns_ok = True
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
        row_columns_ok = row_columns_ok and all(
            _row_has_value(row, key) for key in CONSTRAINT_REGION_CONTACT_LAW_REQUIRED_COLUMNS
        )
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
        row_columns_ok
        and row_law_ok
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
        "constraint_region_contact_law_required_columns_present": int(row_columns_ok),
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


def constraint_region_tangent_gate_metrics(
    summary: Row,
    history_rows: Sequence[Row],
    *,
    solver_prefix: str = "source",
) -> dict[str, float | int | str]:
    """Gate fixed-active-set consistent tangent evidence for region contact.

    Active accepted rows must use the constraint-region gap Jacobian, include
    slave and master payload derivatives with the correct signs, use the
    linear pressure-overclosure derivative on the fixed active set, and carry a
    source/solver tangent solve that actually exercised active contact rows.
    Optional FD evidence is accepted only when checked, active-set-stable, and
    passing.
    """

    label = str(solver_prefix).strip("_") or "source"
    active_rows = [row for row in history_rows if _row_has_active_contact(row)]
    active_contact_present = len(active_rows) > 0
    source_solve_count = _row_int_flag(summary, f"{label}_constraint_region_tangent_solve_count", default=0)
    source_active_rows = _row_int_flag(summary, f"{label}_constraint_region_tangent_active_rows_sum", default=0)
    source_j_nnz = _row_int_flag(summary, f"{label}_constraint_region_tangent_j_nnz_sum", default=0)
    source_scale_sum = _finite_row_float(summary, f"{label}_constraint_region_tangent_scale_sum") or 0.0
    row_columns_ok = True
    row_source_ok = True
    row_jacobian_ok = True
    row_derivative_ok = True
    row_pressure_filter_ok = True
    row_slave_gap_derivative_ok = True
    row_master_gap_derivative_ok = True
    row_slave_master_derivative_ok = True
    row_sign_ok = True
    row_fixed_active_ok = True
    row_active_count_ok = True
    row_scale_ok = True
    row_solve_ok = True
    row_fd_present = False
    row_fd_checked_ok = True
    row_fd_active_set_stable_ok = True
    row_fd_passed_ok = True
    max_active_count_mismatch = 0
    max_fd_abs_error = 0.0
    max_fd_rel_error = 0.0
    for row in active_rows:
        row_columns_ok = row_columns_ok and all(
            _row_has_value(row, key) for key in CONSTRAINT_REGION_TANGENT_REQUIRED_COLUMNS
        )
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
        row_pressure_filter_ok = (
            row_pressure_filter_ok
            and str(row.get("contact_tangent_pressure_scale_filter", "")) == "gap_negative_and_region_area_positive"
            and _row_int_flag(row, "contact_tangent_pressure_positive_scale_count", default=0) > 0
        )
        row_slave_gap_derivative_ok = (
            row_slave_gap_derivative_ok
            and str(row.get("contact_tangent_slave_gap_derivative_source", ""))
            == "secondary_region_shape_weights_times_normal"
            and _row_int_flag(row, "contact_tangent_slave_gap_derivative_nnz", default=0) > 0
        )
        row_master_gap_derivative_ok = (
            row_master_gap_derivative_ok
            and str(row.get("contact_tangent_master_gap_derivative_source", ""))
            == "master_payload_weights_times_negative_normal"
            and _row_int_flag(row, "contact_tangent_master_gap_derivative_nnz", default=0) > 0
        )
        row_slave_master_derivative_ok = (
            row_slave_master_derivative_ok
            and _row_int_flag(row, "contact_tangent_slave_master_gap_derivative_present", default=0) == 1
        )
        row_sign_ok = row_sign_ok and str(row.get("contact_tangent_sign_convention", "")) == "d(-contact_force)/du"
        row_fixed_active_ok = row_fixed_active_ok and _row_int_flag(row, "contact_tangent_fixed_active_set") == 1
        row_active_count_ok = row_active_count_ok and tangent_active == active_regions
        row_scale_ok = row_scale_ok and (_finite_row_float(row, "contact_tangent_scale_sum") or 0.0) > 0.0
        row_solve_ok = row_solve_ok and _row_int_flag(row, f"{label}_constraint_region_tangent_solve_count") > 0
        if _row_has_value(row, "contact_tangent_fd_checked"):
            row_fd_present = True
            row_fd_checked_ok = row_fd_checked_ok and _row_int_flag(row, "contact_tangent_fd_checked") == 1
            row_fd_active_set_stable_ok = (
                row_fd_active_set_stable_ok
                and _row_int_flag(row, "contact_tangent_fd_active_set_stable") == 1
            )
            row_fd_passed_ok = row_fd_passed_ok and _row_int_flag(row, "contact_tangent_fd_passed") == 1
            max_fd_abs_error = max(max_fd_abs_error, _finite_row_float(row, "contact_tangent_fd_error_abs") or 0.0)
            max_fd_rel_error = max(max_fd_rel_error, _finite_row_float(row, "contact_tangent_fd_error_rel") or 0.0)
    summary_tangent_used = (not active_contact_present) or (
        source_solve_count > 0 and source_active_rows > 0 and source_j_nnz > 0 and source_scale_sum > 0.0
    )
    row_fd_ok = (not row_fd_present) or (row_fd_checked_ok and row_fd_active_set_stable_ok and row_fd_passed_ok)
    rows_ok = (
        row_columns_ok
        and row_source_ok
        and row_jacobian_ok
        and row_derivative_ok
        and row_pressure_filter_ok
        and row_slave_gap_derivative_ok
        and row_master_gap_derivative_ok
        and row_slave_master_derivative_ok
        and row_sign_ok
        and row_fixed_active_ok
        and row_active_count_ok
        and row_scale_ok
        and row_solve_ok
        and row_fd_ok
    )
    gate_passed = int((not active_contact_present) or (bool(summary_tangent_used) and bool(rows_ok)))
    return {
        "constraint_region_tangent_gate_passed": gate_passed,
        "comparison_stage": "constraint_region_tangent_before_clouds",
        "constraint_region_tangent_active_contact_present": int(active_contact_present),
        "constraint_region_tangent_active_history_row_count": int(len(active_rows)),
        "constraint_region_tangent_solver_prefix": label,
        "constraint_region_tangent_required_columns_present": int(row_columns_ok),
        "constraint_region_tangent_summary_used": int(summary_tangent_used),
        "constraint_region_tangent_row_source_ok": int(row_source_ok),
        "constraint_region_tangent_row_jacobian_ok": int(row_jacobian_ok),
        "constraint_region_tangent_row_pressure_derivative_ok": int(row_derivative_ok),
        "constraint_region_tangent_row_pressure_filter_ok": int(row_pressure_filter_ok),
        "constraint_region_tangent_row_slave_gap_derivative_ok": int(row_slave_gap_derivative_ok),
        "constraint_region_tangent_row_master_gap_derivative_ok": int(row_master_gap_derivative_ok),
        "constraint_region_tangent_row_slave_master_derivative_ok": int(row_slave_master_derivative_ok),
        "constraint_region_tangent_row_sign_convention_ok": int(row_sign_ok),
        "constraint_region_tangent_row_fixed_active_set_ok": int(row_fixed_active_ok),
        "constraint_region_tangent_row_active_count_ok": int(row_active_count_ok),
        "constraint_region_tangent_row_scale_ok": int(row_scale_ok),
        "constraint_region_tangent_row_solve_ok": int(row_solve_ok),
        "constraint_region_tangent_row_fd_present": int(row_fd_present),
        "constraint_region_tangent_row_fd_checked_ok": int(row_fd_checked_ok),
        "constraint_region_tangent_row_fd_active_set_stable_ok": int(row_fd_active_set_stable_ok),
        "constraint_region_tangent_row_fd_passed_ok": int(row_fd_passed_ok),
        "constraint_region_tangent_max_active_count_mismatch": int(max_active_count_mismatch),
        "constraint_region_tangent_fd_error_abs_max": float(max_fd_abs_error),
        "constraint_region_tangent_fd_error_rel_max": float(max_fd_rel_error),
        "constraint_region_tangent_source_solve_count": int(source_solve_count),
        "constraint_region_tangent_source_active_rows_sum": int(source_active_rows),
        "constraint_region_tangent_source_j_nnz_sum": int(source_j_nnz),
        "constraint_region_tangent_source_scale_sum": float(source_scale_sum),
    }


def _row_has_value(row: Row, key: str) -> bool:
    value = row.get(key)
    return value not in ("", None)


def _finite_row_float(row: Row, *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value in ("", None):
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if result == result and abs(result) != float("inf"):
            return result
    return None


def _row_int_flag(row: Row, key: str, *, default: int = 0) -> int:
    value = row.get(key, default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _row_has_active_contact(row: Row) -> bool:
    active_regions = _row_int_flag(row, "active_contact_region_count", default=0)
    active_area = _finite_row_float(row, "contact_active_area") or 0.0
    normal_force = abs(_finite_row_float(row, "contact_region_normal_force", "normal_force") or 0.0)
    return bool(active_regions > 0 or active_area > 0.0 or normal_force > 0.0)
