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
