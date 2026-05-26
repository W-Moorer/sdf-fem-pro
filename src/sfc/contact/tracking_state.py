"""Accepted-state contact tracking cache utilities.

Path tracking is an acceleration and continuity mechanism.  Trial contact
queries during Newton iterations, line searches, or cutbacks must not mutate
the accepted tracking state.  This module provides small generic helpers that
snapshot and restore known contact-cache fields, and commit accepted payloads
only after an increment is accepted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


def snapshot_contact_tracking_state(contact_geometries: Iterable[Any]) -> list[dict[str, Any]]:
    """Capture path-tracking caches for rollback around trial queries."""

    states: list[dict[str, Any]] = []
    for geometry in contact_geometries:
        oracle = getattr(geometry, "_oracle", None)
        states.append(
            {
                "geometry": geometry,
                "secondary_face_cache": _copy_optional_array(getattr(geometry, "_secondary_face_cache", None)),
                "secondary_barycentric_cache": _copy_optional_array(
                    getattr(geometry, "_secondary_barycentric_cache", None)
                ),
                "secondary_region_face_cache": (
                    dict(getattr(geometry, "_secondary_region_face_cache"))
                    if hasattr(geometry, "_secondary_region_face_cache")
                    else None
                ),
                "secondary_region_barycentric_cache": _copy_array_dict(
                    getattr(geometry, "_secondary_region_barycentric_cache", None)
                )
                if hasattr(geometry, "_secondary_region_barycentric_cache")
                else None,
                "secondary_master_weight_cache": _copy_optional_array(
                    getattr(geometry, "_secondary_master_weight_cache", None)
                ),
                "oracle": oracle,
                "oracle_patch_cache": dict(getattr(oracle, "_patch_cache", {})) if oracle is not None else None,
                "oracle_barycentric_cache": _copy_array_dict(getattr(oracle, "_barycentric_cache", None))
                if oracle is not None
                else None,
            }
        )
    return states


def restore_contact_tracking_state(states: Iterable[Mapping[str, Any]]) -> None:
    """Restore caches captured by :func:`snapshot_contact_tracking_state`."""

    for state_row in states:
        geometry = state_row["geometry"]
        _restore_optional_array_attribute(geometry, "_secondary_face_cache", state_row.get("secondary_face_cache"), np.int64)
        _restore_optional_array_attribute(
            geometry,
            "_secondary_barycentric_cache",
            state_row.get("secondary_barycentric_cache"),
            float,
        )
        _restore_optional_dict_attribute(geometry, "_secondary_region_face_cache", state_row.get("secondary_region_face_cache"))
        _restore_optional_array_dict_attribute(
            geometry,
            "_secondary_region_barycentric_cache",
            state_row.get("secondary_region_barycentric_cache"),
        )
        _restore_optional_array_attribute(
            geometry,
            "_secondary_master_weight_cache",
            state_row.get("secondary_master_weight_cache"),
            float,
        )
        oracle = state_row.get("oracle")
        if oracle is not None:
            setattr(oracle, "_patch_cache", dict(state_row.get("oracle_patch_cache") or {}))
            setattr(oracle, "_barycentric_cache", _copy_array_dict(state_row.get("oracle_barycentric_cache")) or {})


def run_contact_tracking_trial(contact_geometries: Iterable[Any], query: Callable[[], T]) -> T:
    """Run ``query`` and roll back any path-tracking cache mutations."""

    geometries = list(contact_geometries)
    state = snapshot_contact_tracking_state(geometries)
    try:
        return query()
    finally:
        restore_contact_tracking_state(state)


def commit_accepted_contact_tracking_from_sample_arrays(
    contact_geometries: Iterable[Any],
    sample_arrays: Mapping[str, np.ndarray] | Sequence[Mapping[str, np.ndarray] | None] | None,
) -> int:
    """Commit accepted closest-feature payloads without re-running projection.

    If one contact geometry is provided, ``sample_arrays`` is committed to that
    geometry.  If several geometries are provided, pass one array dictionary per
    geometry; an unpartitioned combined array is intentionally ignored because
    committing it would mix tracking caches from different contact pairs.
    """

    geometries = list(contact_geometries)
    if sample_arrays is None or not geometries:
        return 0

    if isinstance(sample_arrays, Mapping):
        if len(geometries) != 1:
            return 0
        return _commit_one_geometry_tracking(geometries[0], sample_arrays)

    if len(sample_arrays) != len(geometries):
        return 0
    committed = 0
    for geometry, arrays in zip(geometries, sample_arrays, strict=True):
        if arrays is not None:
            committed += _commit_one_geometry_tracking(geometry, arrays)
    return int(committed)


def accepted_contact_response_can_reuse(*, converged: bool, sample_arrays_present: bool) -> bool:
    """Return whether the final trial contact response may become accepted state."""

    return bool(converged and sample_arrays_present)


def accepted_contact_response_coverage_gate_metrics(
    summary: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str = "source",
) -> dict[str, Any]:
    """Gate accepted increments against accepted-state response provenance.

    Every accepted increment must be covered by either a final-trial response
    reuse or an accepted-state requery.  History rows may be frame-strided, but
    each emitted accepted row must still expose a binary reuse flag so outputs
    cannot silently come from an unproven trial response.
    """

    label = str(prefix).strip("_")
    accepted_count = _mapping_int_flag(summary, f"{label}_accepted_increment_count", default=len(history_rows))
    reuse_count = _mapping_int_flag(summary, f"{label}_accepted_contact_response_reuse_count", default=0)
    requery_count = _mapping_int_flag(summary, f"{label}_accepted_contact_response_requery_count", default=0)
    total_count = int(reuse_count + requery_count)
    reuse_key = f"{label}_accepted_contact_response_reused"
    accepted_key = f"{label}_increment_accepted"
    accepted_rows = [row for row in history_rows if _mapping_int_flag(row, accepted_key, default=1) == 1]
    history_reuse_flag_present = all(reuse_key in row for row in accepted_rows)
    history_reuse_flag_binary = all(_mapping_int_flag(row, reuse_key, default=-1) in (0, 1) for row in accepted_rows)
    count_matches_accepted = total_count == int(accepted_count) or (total_count == 0 and int(accepted_count) == 0)
    nonzero_when_accepted = int(accepted_count) == 0 or total_count > 0
    history_row_count_ok = len(accepted_rows) <= int(accepted_count)
    gate_passed = int(
        bool(count_matches_accepted)
        and bool(nonzero_when_accepted)
        and bool(history_row_count_ok)
        and bool(history_reuse_flag_present)
        and bool(history_reuse_flag_binary)
    )
    return {
        f"{label}_accepted_contact_response_gate_passed": int(gate_passed),
        f"{label}_accepted_contact_response_count_matches_accepted": int(count_matches_accepted),
        f"{label}_accepted_contact_response_nonzero_when_accepted": int(nonzero_when_accepted),
        f"{label}_accepted_contact_response_history_reuse_flag_present": int(history_reuse_flag_present),
        f"{label}_accepted_contact_response_history_reuse_flag_binary": int(history_reuse_flag_binary),
        f"{label}_accepted_contact_response_history_row_count_ok": int(history_row_count_ok),
        f"{label}_accepted_contact_response_accepted_increment_count": int(accepted_count),
        f"{label}_accepted_contact_response_reuse_count": int(reuse_count),
        f"{label}_accepted_contact_response_requery_count": int(requery_count),
        f"{label}_accepted_contact_response_total_count": int(total_count),
        f"{label}_accepted_contact_response_history_accepted_row_count": int(len(accepted_rows)),
    }


def _commit_one_geometry_tracking(geometry: Any, sample_arrays: Mapping[str, np.ndarray]) -> int:
    commit = getattr(geometry, "commit_secondary_tracking_from_sample_arrays", None)
    if not callable(commit):
        return 0
    return int(commit(dict(sample_arrays)))


def _copy_optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value).copy()


def _copy_array_dict(value: Any) -> dict[Any, np.ndarray] | None:
    if value is None:
        return None
    return {key: np.asarray(array, dtype=float).copy() for key, array in dict(value).items()}


def _restore_optional_array_attribute(obj: Any, name: str, value: Any, dtype: Any) -> None:
    if value is None and not hasattr(obj, name):
        return
    try:
        setattr(obj, name, None if value is None else np.asarray(value, dtype=dtype).copy())
    except AttributeError:
        if value is None:
            return
        raise


def _restore_optional_dict_attribute(obj: Any, name: str, value: Any) -> None:
    if value is None and not hasattr(obj, name):
        return
    try:
        setattr(obj, name, dict(value or {}))
    except AttributeError:
        if value is None:
            return
        raise


def _restore_optional_array_dict_attribute(obj: Any, name: str, value: Any) -> None:
    if value is None and not hasattr(obj, name):
        return
    try:
        setattr(obj, name, _copy_array_dict(value) or {})
    except AttributeError:
        if value is None:
            return
        raise


def _mapping_int_flag(row: Mapping[str, Any], key: str, *, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return int(default)
