"""Constraint-region contact aggregation utilities.

The routines in this module convert raw surface quadrature contact samples into
Abaqus-style secondary/slave constraint regions.  They operate on the batched
``sample_arrays`` dictionaries emitted by the Lagrangian SDF contact geometry:
each output row represents one secondary constraint region, not one independent
node-to-surface penalty sample.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def contact_averaging_modes(mode: str) -> tuple[str, str]:
    """Return ``(base_mode, overclosure_mode)`` for a contact averaging label."""

    averaging = str(mode).lower()
    signed_participation_mode = averaging.endswith("_signed_participation")
    signed_mode = averaging.endswith("_constraint")
    area_average_mode = averaging.endswith("_area_average")
    participation_mode = averaging.endswith("_participation")
    if signed_participation_mode:
        participation_mode = False
    if sum(int(flag) for flag in (signed_mode, area_average_mode, participation_mode, signed_participation_mode)) > 1:
        raise ValueError("contact averaging mode cannot combine suffixes")
    if signed_participation_mode:
        base_mode = averaging[: -len("_signed_participation")]
        overclosure_mode = "signed_status_linear_penalty_participation"
    elif signed_mode:
        base_mode = averaging[: -len("_constraint")]
        overclosure_mode = "signed_average"
    elif area_average_mode:
        base_mode = averaging[: -len("_area_average")]
        overclosure_mode = "positive_integral_area_average"
    elif participation_mode:
        base_mode = averaging[: -len("_participation")]
        overclosure_mode = "linear_penalty_participation"
    else:
        base_mode = averaging
        overclosure_mode = "positive_integral"
    if base_mode not in {"none", "slave_face", "slave_node", "slave_node_region", "surface_patch"}:
        raise ValueError(
            "contact averaging must be 'none', 'slave_face', 'slave_node', "
            "'slave_node_region', 'surface_patch', or a supported suffix mode"
        )
    return base_mode, overclosure_mode


def aggregate_contact_sample_arrays(sample_arrays: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray] | None:
    """Aggregate raw contact arrays into constraint-region rows.

    ``slave_node_region_constraint`` is the Abaqus-aligned default: each slave
    node receives a tributary region from the surrounding quadrature samples,
    the signed area-average gap decides open/closed status, the region normal
    is area-averaged, and the master closest-feature payload is interpolated
    with the same regional kinematic weights.
    """

    base_mode, overclosure_mode = contact_averaging_modes(mode)
    if base_mode == "none":
        return sample_arrays
    if base_mode in {"surface_patch", "slave_node"}:
        return None
    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    if gaps.size == 0:
        return _empty_like_contact_arrays(sample_arrays)
    sample_nodes = np.asarray(sample_arrays["sample_node_ids"], dtype=np.int64)
    sample_weights = np.asarray(sample_arrays["sample_weights"], dtype=float)
    normals = np.asarray(sample_arrays["normals"], dtype=float).reshape((-1, 3))
    norms = np.maximum(np.linalg.norm(normals, axis=1), 1.0e-30)
    normals = normals / norms[:, None]
    areas = np.asarray(sample_arrays["areas"], dtype=float).reshape(-1)
    master_nodes = np.asarray(sample_arrays["master_node_ids"], dtype=np.int64)
    master_weights = np.asarray(sample_arrays["master_weights"], dtype=float)
    master_barycentric = _optional_2d(sample_arrays.get("master_barycentric"))
    master_face_ids = _optional_1d(sample_arrays.get("master_face_ids"), dtype=np.int64)
    secondary_cache_indices = _optional_1d(sample_arrays.get("secondary_cache_indices"), dtype=np.int64)
    tracking_cache_hits = _optional_1d(sample_arrays.get("tracking_cache_hits"), dtype=bool)
    tracking_cache_matches = _optional_1d(sample_arrays.get("tracking_cache_matches"), dtype=bool)
    tracking_barycentric_distances = _optional_1d(
        sample_arrays.get("tracking_barycentric_distances"),
        dtype=float,
    )

    if base_mode == "slave_face":
        groups: dict[tuple[int, ...], list[int]] = {}
        order: list[tuple[int, ...]] = []
        expanded = np.arange(gaps.size, dtype=np.int64)
        secondary_node_ids = np.full(gaps.size, -1, dtype=np.int64)
        expanded_sample_nodes = sample_nodes
        expanded_sample_weights = sample_weights
        expanded_gaps = gaps
        expanded_normals = normals
        expanded_areas = areas
        for idx, nodes in enumerate(sample_nodes):
            key = tuple(int(v) for v in nodes)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(int(idx))
        group_indices = [groups[key] for key in order]
    elif base_mode == "slave_node_region":
        (
            expanded,
            expanded_sample_nodes,
            expanded_sample_weights,
            expanded_gaps,
            expanded_normals,
            expanded_areas,
            secondary_node_ids,
            group_indices,
        ) = _expand_to_slave_node_regions(
            sample_nodes,
            sample_weights,
            gaps,
            normals,
            areas,
        )
    else:
        return None

    if expanded.size == 0:
        return _empty_like_contact_arrays(sample_arrays)
    rows = [
        _aggregate_group(
            group,
            sample_nodes=expanded_sample_nodes,
            sample_weights=expanded_sample_weights,
            gaps=expanded_gaps,
            normals=expanded_normals,
            areas=expanded_areas,
            master_nodes=master_nodes[expanded],
            master_weights=master_weights[expanded],
            master_barycentric=None if master_barycentric is None else master_barycentric[expanded],
            master_face_ids=None if master_face_ids is None else master_face_ids[expanded],
            secondary_cache_indices=None if secondary_cache_indices is None else secondary_cache_indices[expanded],
            secondary_node_id=int(secondary_node_ids[int(group[0])] if secondary_node_ids.size else -1),
            tracking_cache_hits=None if tracking_cache_hits is None else tracking_cache_hits[expanded],
            tracking_cache_matches=None if tracking_cache_matches is None else tracking_cache_matches[expanded],
            tracking_barycentric_distances=(
                None if tracking_barycentric_distances is None else tracking_barycentric_distances[expanded]
            ),
            overclosure_mode=overclosure_mode,
        )
        for group in group_indices
    ]
    return _pack_rows(rows)


def constraint_region_gap_jacobian_sparse_from_arrays(
    sample_arrays: dict[str, np.ndarray],
    *,
    n_nodes: int,
    active_only: bool = False,
) -> tuple[np.ndarray, csr_matrix]:
    """Return the fixed-payload gap Jacobian for constraint-region rows."""

    gaps = np.asarray(sample_arrays["gaps"], dtype=float).reshape(-1)
    row_ids = np.flatnonzero(gaps < 0.0).astype(np.int64) if active_only else np.arange(gaps.size, dtype=np.int64)
    n_total_dofs = 3 * int(n_nodes)
    if row_ids.size == 0:
        return row_ids, coo_matrix((0, n_total_dofs), dtype=float).tocsr()
    slave_nodes = np.asarray(sample_arrays["sample_node_ids"], dtype=np.int64)
    slave_weights = np.asarray(sample_arrays["sample_weights"], dtype=float)
    master_nodes = np.asarray(sample_arrays["master_node_ids"], dtype=np.int64)
    master_weights = np.asarray(sample_arrays["master_weights"], dtype=float)
    normals = np.asarray(sample_arrays["normals"], dtype=float).reshape((-1, 3))
    normals /= np.maximum(np.linalg.norm(normals, axis=1), 1.0e-30)[:, None]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for out_row, sample_id in enumerate(row_ids):
        normal = normals[int(sample_id)]
        for sign, nodes, weights in (
            (1.0, slave_nodes[int(sample_id)], slave_weights[int(sample_id)]),
            (-1.0, master_nodes[int(sample_id)], master_weights[int(sample_id)]),
        ):
            for node, weight in zip(nodes, weights, strict=True):
                if int(node) < 0 or int(node) >= int(n_nodes):
                    continue
                for component in range(3):
                    coeff = float(sign) * float(weight) * float(normal[component])
                    if coeff == 0.0:
                        continue
                    rows.append(out_row)
                    cols.append(3 * int(node) + component)
                    data.append(coeff)
    return row_ids, coo_matrix((data, (rows, cols)), shape=(row_ids.size, n_total_dofs)).tocsr()


def constraint_region_pressure_tangent_scales_from_arrays(
    sample_arrays: dict[str, np.ndarray],
    *,
    pressure_stiffness: float,
    equilibrium_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return active row ids and ``d(A p)/dg`` magnitudes for linear penalty."""

    gaps = np.asarray(sample_arrays["gaps"], dtype=float).reshape(-1)
    active_ids = np.flatnonzero(gaps < 0.0).astype(np.int64)
    if active_ids.size == 0:
        return active_ids, np.empty(0, dtype=float)
    areas = np.asarray(sample_arrays["areas"], dtype=float).reshape(-1)
    scales = float(equilibrium_scale) * float(pressure_stiffness) * areas[active_ids]
    positive = scales > 0.0
    return active_ids[positive], scales[positive]


def contact_region_integral_metrics_from_arrays(
    sample_arrays: dict[str, np.ndarray] | None,
    *,
    stiffness: float,
) -> dict[str, float | int]:
    """Return force, work, energy, and active-area totals for region rows."""

    if sample_arrays is None:
        return _empty_contact_region_integral_metrics()
    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    if gaps.size == 0:
        return _empty_contact_region_integral_metrics()
    areas = np.asarray(sample_arrays.get("areas", np.zeros_like(gaps)), dtype=float).reshape(-1)
    penetrations = np.maximum(-gaps, 0.0)
    pressures = float(stiffness) * penetrations
    active = penetrations > 0.0
    active_area = float(np.sum(areas[active])) if np.any(active) else 0.0
    normal_force = float(np.sum(areas[active] * pressures[active])) if np.any(active) else 0.0
    energy = float(0.5 * np.sum(areas * pressures * penetrations))
    return {
        "contact_region_count": int(gaps.size),
        "active_contact_region_count": int(np.count_nonzero(active)),
        "contact_total_region_area": float(np.sum(areas)),
        "contact_active_area": active_area,
        "contact_region_normal_force": normal_force,
        "contact_region_energy": energy,
        "contact_region_virtual_work": float(2.0 * energy),
        "contact_max_region_pressure": float(np.max(pressures[active])) if np.any(active) else 0.0,
        "contact_mean_active_region_pressure": (
            float(np.sum(areas[active] * pressures[active]) / max(active_area, 1.0e-30)) if np.any(active) else 0.0
        ),
    }


def _expand_to_slave_node_regions(
    sample_nodes: np.ndarray,
    sample_weights: np.ndarray,
    gaps: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    groups: dict[int, list[int]] = {}
    order: list[int] = []
    source_rows: list[int] = []
    expanded_weights: list[np.ndarray] = []
    expanded_areas: list[float] = []
    secondary_ids: list[int] = []
    expanded_index = 0
    for idx in range(gaps.size):
        nodes = np.asarray(sample_nodes[idx], dtype=np.int64).reshape(-1)
        weights = np.asarray(sample_weights[idx], dtype=float).reshape(-1)
        for node, weight in zip(nodes, weights, strict=True):
            tributary = float(areas[idx]) * max(float(weight), 0.0)
            if tributary <= 0.0:
                continue
            source_rows.append(int(idx))
            expanded_weights.append(weights.copy())
            expanded_areas.append(tributary)
            secondary_ids.append(int(node))
            if int(node) not in groups:
                groups[int(node)] = []
                order.append(int(node))
            groups[int(node)].append(expanded_index)
            expanded_index += 1
    source = np.asarray(source_rows, dtype=np.int64)
    if source.size == 0:
        width = sample_nodes.shape[1] if sample_nodes.ndim == 2 else 0
        return (
            source,
            np.empty((0, width), dtype=np.int64),
            np.empty((0, width), dtype=float),
            np.empty(0, dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int64),
            [],
        )
    return (
        source,
        sample_nodes[source].copy(),
        np.vstack(expanded_weights).astype(float, copy=False),
        gaps[source].copy(),
        normals[source].copy(),
        np.asarray(expanded_areas, dtype=float),
        np.asarray(secondary_ids, dtype=np.int64),
        [groups[key] for key in order],
    )


def _aggregate_group(
    rows: list[int],
    *,
    sample_nodes: np.ndarray,
    sample_weights: np.ndarray,
    gaps: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    master_nodes: np.ndarray,
    master_weights: np.ndarray,
    master_barycentric: np.ndarray | None,
    master_face_ids: np.ndarray | None,
    secondary_cache_indices: np.ndarray | None,
    secondary_node_id: int,
    tracking_cache_hits: np.ndarray | None,
    tracking_cache_matches: np.ndarray | None,
    tracking_barycentric_distances: np.ndarray | None,
    overclosure_mode: str,
) -> dict[str, np.ndarray | float | int]:
    row_ids = [int(v) for v in rows]
    row_areas = np.asarray(areas[row_ids], dtype=float)
    total_area = float(np.sum(row_areas))
    if total_area <= 0.0:
        group_weights = np.full(len(row_ids), 1.0 / float(len(row_ids)), dtype=float)
        total_area = float(len(row_ids))
    else:
        group_weights = row_areas / total_area
    row_gaps = np.asarray(gaps[row_ids], dtype=float)
    aggregate_area = total_area
    if overclosure_mode == "signed_average":
        constraint_gap = float(group_weights @ row_gaps)
        kinematic_weights = group_weights
    else:
        penetrations = np.maximum(-row_gaps, 0.0)
        penetration_integral = float(row_areas @ penetrations)
        if penetration_integral > 0.0:
            constraint_gap = -penetration_integral / max(total_area, 1.0e-30)
            kinematic_weights = row_areas * penetrations / penetration_integral
        else:
            constraint_gap = float(group_weights @ row_gaps)
            kinematic_weights = group_weights
    normal = np.sum(normals[row_ids] * group_weights[:, None], axis=0)
    normal_norm = max(float(np.linalg.norm(normal)), 1.0e-30)
    normal = normal / normal_norm
    slave_ids, slave_w = _combined_shape_weights(sample_nodes, sample_weights, row_ids, kinematic_weights)
    master_ids, master_w = _combined_shape_weights(master_nodes, master_weights, row_ids, kinematic_weights)
    representative = int(np.argmax(kinematic_weights)) if len(row_ids) else 0
    source_row = row_ids[representative]
    return {
        "sample_node_ids": slave_ids,
        "sample_weights": slave_w,
        "gap": float(constraint_gap),
        "normal": normal,
        "area": float(aggregate_area),
        "master_node_ids": master_ids,
        "master_weights": master_w,
        "master_barycentric": (
            np.asarray(master_barycentric[source_row], dtype=float).copy()
            if master_barycentric is not None and master_barycentric.shape[0] > source_row
            else np.full(3, np.nan, dtype=float)
        ),
        "master_face_id": (
            int(master_face_ids[source_row])
            if master_face_ids is not None and master_face_ids.size > source_row
            else -1
        ),
        "secondary_cache_index": (
            int(secondary_cache_indices[source_row])
            if secondary_cache_indices is not None and secondary_cache_indices.size > source_row
            else -1
        ),
        "secondary_node_id": int(secondary_node_id),
        "tracking_cache_hit": bool(
            tracking_cache_hits is not None and tracking_cache_hits.size and np.any(tracking_cache_hits[row_ids])
        ),
        "tracking_cache_match": bool(
            tracking_cache_matches is not None and tracking_cache_matches.size and np.any(tracking_cache_matches[row_ids])
        ),
        "tracking_barycentric_distance": _weighted_tracking_distance(
            tracking_barycentric_distances,
            row_ids,
            kinematic_weights,
        ),
    }


def _combined_shape_weights(
    node_rows: np.ndarray,
    weight_rows: np.ndarray,
    row_indices: list[int],
    group_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    accum: dict[int, float] = {}
    for local, row_index in enumerate(row_indices):
        for node, weight in zip(node_rows[int(row_index)], weight_rows[int(row_index)], strict=True):
            contribution = float(group_weights[int(local)]) * float(weight)
            if abs(contribution) <= 1.0e-30:
                continue
            key = int(node)
            accum[key] = accum.get(key, 0.0) + contribution
    if not accum:
        return np.asarray([0], dtype=np.int64), np.asarray([0.0], dtype=float)
    ids = np.asarray(sorted(accum), dtype=np.int64)
    weights = np.asarray([accum[int(node)] for node in ids], dtype=float)
    total = float(np.sum(weights))
    if total > 0.0:
        weights /= total
    return ids, weights


def _pack_rows(rows: list[dict[str, np.ndarray | float | int]]) -> dict[str, np.ndarray]:
    if not rows:
        return _empty_packed_rows()
    slave_width = max(int(np.asarray(row["sample_node_ids"]).size) for row in rows)
    master_width = max(int(np.asarray(row["master_node_ids"]).size) for row in rows)
    bary_width = max(int(np.asarray(row["master_barycentric"]).size) for row in rows)
    out = {
        "sample_node_ids": np.zeros((len(rows), slave_width), dtype=np.int64),
        "sample_weights": np.zeros((len(rows), slave_width), dtype=float),
        "gaps": np.zeros(len(rows), dtype=float),
        "normals": np.zeros((len(rows), 3), dtype=float),
        "areas": np.zeros(len(rows), dtype=float),
        "master_node_ids": np.zeros((len(rows), master_width), dtype=np.int64),
        "master_weights": np.zeros((len(rows), master_width), dtype=float),
        "master_barycentric": np.full((len(rows), bary_width), np.nan, dtype=float),
        "master_face_ids": np.full(len(rows), -1, dtype=np.int64),
        "secondary_cache_indices": np.full(len(rows), -1, dtype=np.int64),
        "secondary_node_ids": np.full(len(rows), -1, dtype=np.int64),
        "tracking_cache_hits": np.zeros(len(rows), dtype=bool),
        "tracking_cache_matches": np.zeros(len(rows), dtype=bool),
        "tracking_barycentric_distances": np.full(len(rows), np.nan, dtype=float),
    }
    for index, row in enumerate(rows):
        slave_ids = np.asarray(row["sample_node_ids"], dtype=np.int64).reshape(-1)
        slave_w = np.asarray(row["sample_weights"], dtype=float).reshape(-1)
        master_ids = np.asarray(row["master_node_ids"], dtype=np.int64).reshape(-1)
        master_w = np.asarray(row["master_weights"], dtype=float).reshape(-1)
        bary = np.asarray(row["master_barycentric"], dtype=float).reshape(-1)
        out["sample_node_ids"][index, : slave_ids.size] = slave_ids
        out["sample_weights"][index, : slave_w.size] = slave_w
        out["master_node_ids"][index, : master_ids.size] = master_ids
        out["master_weights"][index, : master_w.size] = master_w
        out["master_barycentric"][index, : bary.size] = bary
        out["gaps"][index] = float(row["gap"])
        out["normals"][index] = np.asarray(row["normal"], dtype=float).reshape(3)
        out["areas"][index] = float(row["area"])
        out["master_face_ids"][index] = int(row["master_face_id"])
        out["secondary_cache_indices"][index] = int(row["secondary_cache_index"])
        out["secondary_node_ids"][index] = int(row["secondary_node_id"])
        out["tracking_cache_hits"][index] = bool(row["tracking_cache_hit"])
        out["tracking_cache_matches"][index] = bool(row["tracking_cache_match"])
        out["tracking_barycentric_distances"][index] = float(row["tracking_barycentric_distance"])
    return out


def _weighted_tracking_distance(values: np.ndarray | None, rows: list[int], weights: np.ndarray) -> float:
    if values is None or values.size == 0:
        return float("nan")
    distances = np.asarray(values, dtype=float)[rows]
    finite = np.isfinite(distances)
    if not np.any(finite):
        return float("nan")
    finite_weights = np.asarray(weights, dtype=float)[finite]
    weight_sum = float(np.sum(finite_weights))
    if weight_sum > 0.0:
        return float(finite_weights @ distances[finite] / weight_sum)
    return float(np.mean(distances[finite]))


def _empty_contact_region_integral_metrics() -> dict[str, float | int]:
    return {
        "contact_region_count": 0,
        "active_contact_region_count": 0,
        "contact_total_region_area": 0.0,
        "contact_active_area": 0.0,
        "contact_region_normal_force": 0.0,
        "contact_region_energy": 0.0,
        "contact_region_virtual_work": 0.0,
        "contact_max_region_pressure": 0.0,
        "contact_mean_active_region_pressure": 0.0,
    }


def _optional_1d(value: object, *, dtype: type) -> np.ndarray | None:
    return None if value is None else np.asarray(value, dtype=dtype).reshape(-1)


def _optional_2d(value: object) -> np.ndarray | None:
    return None if value is None else np.asarray(value, dtype=float)


def _empty_packed_rows() -> dict[str, np.ndarray]:
    return {
        "sample_node_ids": np.empty((0, 0), dtype=np.int64),
        "sample_weights": np.empty((0, 0), dtype=float),
        "gaps": np.empty(0, dtype=float),
        "normals": np.empty((0, 3), dtype=float),
        "areas": np.empty(0, dtype=float),
        "master_node_ids": np.empty((0, 0), dtype=np.int64),
        "master_weights": np.empty((0, 0), dtype=float),
        "master_barycentric": np.empty((0, 3), dtype=float),
        "master_face_ids": np.empty(0, dtype=np.int64),
        "secondary_cache_indices": np.empty(0, dtype=np.int64),
        "secondary_node_ids": np.empty(0, dtype=np.int64),
        "tracking_cache_hits": np.empty(0, dtype=bool),
        "tracking_cache_matches": np.empty(0, dtype=bool),
        "tracking_barycentric_distances": np.empty(0, dtype=float),
    }


def _empty_like_contact_arrays(sample_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    _ = sample_arrays
    return _empty_packed_rows()
