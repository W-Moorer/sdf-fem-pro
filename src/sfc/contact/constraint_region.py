"""Constraint-region contact aggregation utilities.

The routines in this module convert raw surface quadrature contact samples into
Abaqus-style secondary/slave constraint regions.  They operate on the batched
``sample_arrays`` dictionaries emitted by the Lagrangian SDF contact geometry:
each output row represents one secondary constraint region, not one independent
node-to-surface penalty sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags


@dataclass(slots=True)
class ConstraintRegionContactResponse:
    """Penalty response assembled from secondary constraint-region rows."""

    force: np.ndarray
    tangent: csr_matrix
    min_gap: float
    max_penetration: float
    active_count: int
    normal_force: float
    energy: float
    virtual_work: float
    active_area: float
    metrics: dict[str, float | int | str]


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


def constraint_region_contact_tangent_sparse_from_arrays(
    sample_arrays: dict[str, np.ndarray] | None,
    *,
    n_nodes: int,
    pressure_stiffness: float,
    equilibrium_scale: float = 1.0,
) -> tuple[np.ndarray, csr_matrix, np.ndarray, csr_matrix, dict[str, float | int | str]]:
    """Return the fixed-active-set constraint-region contact tangent.

    For region gap rows ``g`` and linear penalty pressure
    ``p = k_p <-g>_+``, the active-set tangent uses the fixed closest-feature
    payload derivative

    ``K_c = J.T @ diag(equilibrium_scale * k_p * A_region) @ J``.

    The returned sparse ``J`` is the full-space derivative of region gap with
    respect to nodal coordinates; it includes positive slave coefficients and
    negative master coefficients.  Open regions and non-positive area scales
    are excluded from the active rows.
    """

    n_total_dofs = 3 * int(n_nodes)
    if sample_arrays is None:
        empty_j = coo_matrix((0, n_total_dofs), dtype=float).tocsr()
        empty_k = coo_matrix((n_total_dofs, n_total_dofs), dtype=float).tocsr()
        return np.empty(0, dtype=np.int64), empty_j, np.empty(0, dtype=float), empty_k, constraint_region_tangent_metrics_from_arrays(
            None,
            pressure_stiffness=pressure_stiffness,
            equilibrium_scale=equilibrium_scale,
        )
    active_ids, scales = constraint_region_pressure_tangent_scales_from_arrays(
        sample_arrays,
        pressure_stiffness=pressure_stiffness,
        equilibrium_scale=equilibrium_scale,
    )
    all_active_ids, active_j = constraint_region_gap_jacobian_sparse_from_arrays(
        sample_arrays,
        n_nodes=n_nodes,
        active_only=True,
    )
    if active_ids.size == 0:
        empty_j = coo_matrix((0, n_total_dofs), dtype=float).tocsr()
        empty_k = coo_matrix((n_total_dofs, n_total_dofs), dtype=float).tocsr()
        return active_ids, empty_j, scales, empty_k, constraint_region_tangent_metrics_from_arrays(
            sample_arrays,
            pressure_stiffness=pressure_stiffness,
            equilibrium_scale=equilibrium_scale,
        )
    if not np.array_equal(active_ids, all_active_ids):
        row_lookup = {int(region_id): int(row) for row, region_id in enumerate(all_active_ids)}
        keep_rows = np.asarray([row_lookup[int(region_id)] for region_id in active_ids], dtype=np.int64)
        active_j = active_j[keep_rows]
    tangent = (active_j.T @ diags(np.asarray(scales, dtype=float), format="csr") @ active_j).tocsr()
    metrics = constraint_region_tangent_metrics_from_arrays(
        sample_arrays,
        pressure_stiffness=pressure_stiffness,
        equilibrium_scale=equilibrium_scale,
    )
    metrics["contact_tangent_j_nnz"] = int(active_j.nnz)
    metrics["contact_tangent_matrix_nnz"] = int(tangent.nnz)
    return active_ids, active_j, scales, tangent, metrics


def constraint_region_reduced_gap_jacobian_sparse_from_arrays(
    sample_arrays: dict[str, np.ndarray],
    *,
    transformation: object,
    free: np.ndarray,
    active_only: bool = True,
) -> tuple[np.ndarray, csr_matrix]:
    """Return the constraint-region gap Jacobian in reduced free DOFs.

    If full nodal increments satisfy ``du_full = T dq_reduced``, the reduced
    active gap derivative is exactly ``J_full @ T[:, free]``.  This keeps RP/MPC
    kinematics, free-DOF filtering, and the constraint-region contact tangent in
    one shared formulation.
    """

    transform = transformation.tocsr()
    if transform.shape[0] % 3 != 0:
        raise ValueError("constraint-region transformation row count must be a multiple of 3")
    free_cols = np.asarray(free, dtype=np.int64).reshape(-1)
    n_nodes = int(transform.shape[0] // 3)
    row_ids, full_jacobian = constraint_region_gap_jacobian_sparse_from_arrays(
        sample_arrays,
        n_nodes=n_nodes,
        active_only=active_only,
    )
    if free_cols.size == 0:
        return row_ids, coo_matrix((full_jacobian.shape[0], 0), dtype=float).tocsr()
    reduced_jacobian = (full_jacobian @ transform[:, free_cols]).tocsr()
    return row_ids, reduced_jacobian


def active_constraint_region_tangent_data_from_arrays(
    sample_arrays: dict[str, np.ndarray],
    *,
    transformation: object,
    free: np.ndarray,
    pressure_stiffness: float,
    equilibrium_scale: float,
) -> tuple[np.ndarray, csr_matrix, np.ndarray]:
    """Return active region ids, reduced ``J``, and linear penalty scales."""

    active_ids, scales = constraint_region_pressure_tangent_scales_from_arrays(
        sample_arrays,
        pressure_stiffness=pressure_stiffness,
        equilibrium_scale=equilibrium_scale,
    )
    all_active_ids, reduced_jacobian = constraint_region_reduced_gap_jacobian_sparse_from_arrays(
        sample_arrays,
        transformation=transformation,
        free=free,
        active_only=True,
    )
    if active_ids.size == 0:
        return active_ids, coo_matrix((0, reduced_jacobian.shape[1]), dtype=float).tocsr(), scales
    if not np.array_equal(active_ids, all_active_ids):
        row_lookup = {int(region_id): int(row) for row, region_id in enumerate(all_active_ids)}
        keep_rows = np.asarray([row_lookup[int(region_id)] for region_id in active_ids], dtype=np.int64)
        reduced_jacobian = reduced_jacobian[keep_rows]
    return active_ids, reduced_jacobian, scales


def constraint_region_reduced_contact_tangent_sparse_from_arrays(
    sample_arrays: dict[str, np.ndarray],
    *,
    transformation: object,
    free: np.ndarray,
    pressure_stiffness: float,
    equilibrium_scale: float,
) -> tuple[np.ndarray, csr_matrix, np.ndarray, csr_matrix, dict[str, float | int | str]]:
    """Return reduced/free-DOF fixed-active-set contact tangent data."""

    active_ids, reduced_jacobian, scales = active_constraint_region_tangent_data_from_arrays(
        sample_arrays,
        transformation=transformation,
        free=free,
        pressure_stiffness=pressure_stiffness,
        equilibrium_scale=equilibrium_scale,
    )
    tangent = (
        reduced_jacobian.T @ diags(np.asarray(scales, dtype=float), format="csr") @ reduced_jacobian
        if active_ids.size
        else coo_matrix((reduced_jacobian.shape[1], reduced_jacobian.shape[1]), dtype=float).tocsr()
    ).tocsr()
    metrics = constraint_region_tangent_metrics_from_arrays(
        sample_arrays,
        pressure_stiffness=pressure_stiffness,
        equilibrium_scale=equilibrium_scale,
    )
    metrics["contact_tangent_j_nnz"] = int(reduced_jacobian.nnz)
    metrics["contact_tangent_matrix_nnz"] = int(tangent.nnz)
    metrics["contact_tangent_coordinate_space"] = "reduced_free"
    return active_ids, reduced_jacobian, scales, tangent, metrics


def constraint_region_penalty_response_from_arrays(
    sample_arrays: dict[str, np.ndarray] | None,
    *,
    n_nodes: int,
    pressure_stiffness: float,
    tangent_equilibrium_scale: float = 1.0,
    include_tangent: bool = True,
) -> ConstraintRegionContactResponse:
    """Assemble force, tangent, and totals from region rows.

    Each row of ``sample_arrays`` is treated as one secondary constraint
    region.  The open/closed state is determined by that row's averaged gap,
    and its pressure is distributed to slave and master nodes using the
    already-aggregated shape weights and closest-feature payload.  This is the
    core response path for Abaqus-style constraint-region contact; raw
    quadrature points should be aggregated before calling this function.
    """

    count = int(n_nodes)
    force = np.zeros((count, 3), dtype=float)
    ndofs = 3 * count
    empty_tangent = coo_matrix((ndofs, ndofs), dtype=float).tocsr()
    if sample_arrays is None:
        return ConstraintRegionContactResponse(
            force=force,
            tangent=empty_tangent,
            min_gap=0.0,
            max_penetration=0.0,
            active_count=0,
            normal_force=0.0,
            energy=0.0,
            virtual_work=0.0,
            active_area=0.0,
            metrics=_empty_contact_region_integral_metrics(),
        )

    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    if gaps.size == 0:
        return ConstraintRegionContactResponse(
            force=force,
            tangent=empty_tangent,
            min_gap=0.0,
            max_penetration=0.0,
            active_count=0,
            normal_force=0.0,
            energy=0.0,
            virtual_work=0.0,
            active_area=0.0,
            metrics=_empty_contact_region_integral_metrics(),
        )

    normals = np.asarray(sample_arrays["normals"], dtype=float).reshape((-1, 3))
    normals /= np.maximum(np.linalg.norm(normals, axis=1), 1.0e-30)[:, None]
    areas = np.asarray(sample_arrays["areas"], dtype=float).reshape(-1)
    penetrations = np.maximum(-gaps, 0.0)
    active = penetrations > 0.0
    region_forces = float(pressure_stiffness) * areas * penetrations
    if np.any(active):
        vectors = region_forces[active, None] * normals[active]
        _scatter_weighted_vectors(
            force,
            np.asarray(sample_arrays["sample_node_ids"], dtype=np.int64)[active],
            np.asarray(sample_arrays["sample_weights"], dtype=float)[active],
            vectors,
            sign=1.0,
        )
        _scatter_weighted_vectors(
            force,
            np.asarray(sample_arrays["master_node_ids"], dtype=np.int64)[active],
            np.asarray(sample_arrays["master_weights"], dtype=float)[active],
            vectors,
            sign=-1.0,
        )

    if include_tangent:
        _, _, _, tangent, tangent_metrics = constraint_region_contact_tangent_sparse_from_arrays(
            sample_arrays,
            n_nodes=count,
            pressure_stiffness=float(pressure_stiffness),
            equilibrium_scale=float(tangent_equilibrium_scale),
        )
    else:
        tangent = empty_tangent
        tangent_metrics = constraint_region_tangent_metrics_from_arrays(
            sample_arrays,
            pressure_stiffness=float(pressure_stiffness),
            equilibrium_scale=float(tangent_equilibrium_scale),
        )
        tangent_metrics["contact_tangent_included"] = 0

    metrics = contact_region_integral_metrics_from_arrays(sample_arrays, stiffness=float(pressure_stiffness))
    metrics.update(tangent_metrics)
    return ConstraintRegionContactResponse(
        force=force,
        tangent=tangent,
        min_gap=float(np.min(gaps)),
        max_penetration=float(np.max(penetrations)) if penetrations.size else 0.0,
        active_count=int(np.count_nonzero(active)),
        normal_force=float(np.sum(region_forces[active])) if np.any(active) else 0.0,
        energy=float(metrics["contact_region_energy"]),
        virtual_work=float(metrics["contact_region_virtual_work"]),
        active_area=float(metrics["contact_active_area"]),
        metrics=metrics,
    )


def constraint_region_tangent_metrics_from_arrays(
    sample_arrays: dict[str, np.ndarray] | None,
    *,
    pressure_stiffness: float,
    equilibrium_scale: float = 1.0,
) -> dict[str, float | int | str]:
    """Return scalar diagnostics for the active constraint-region tangent."""

    if sample_arrays is None:
        return {
            "contact_tangent_source": "none",
            "contact_tangent_gap_jacobian_source": "none",
            "contact_tangent_pressure_derivative": "none",
            "contact_tangent_scale_formula": "",
            "contact_tangent_fixed_active_set": 0,
            "contact_tangent_active_region_count": 0,
            "contact_tangent_active_secondary_node_count": 0,
            "contact_tangent_scale_sum": 0.0,
            "contact_tangent_scale_max": 0.0,
        }
    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    active = gaps < 0.0
    active_ids, scales = constraint_region_pressure_tangent_scales_from_arrays(
        sample_arrays,
        pressure_stiffness=pressure_stiffness,
        equilibrium_scale=equilibrium_scale,
    )
    secondary_ids = np.asarray(sample_arrays.get("secondary_node_ids", np.empty(0)), dtype=np.int64).reshape(-1)
    if secondary_ids.shape == gaps.shape and active_ids.size:
        active_secondary = secondary_ids[active_ids]
        active_secondary = active_secondary[active_secondary >= 0]
    else:
        active_secondary = np.empty(0, dtype=np.int64)
    return {
        "contact_tangent_source": "constraint_region_arrays",
        "contact_tangent_gap_jacobian_source": "constraint_region_fixed_payload",
        "contact_tangent_pressure_derivative": "linear_penalty_active_set",
        "contact_tangent_scale_formula": "equilibrium_scale*pressure_stiffness*region_area",
        "contact_tangent_fixed_active_set": 1,
        "contact_tangent_active_region_count": int(np.count_nonzero(active)),
        "contact_tangent_active_secondary_node_count": int(np.unique(active_secondary).size),
        "contact_tangent_scale_sum": float(np.sum(scales)) if scales.size else 0.0,
        "contact_tangent_scale_max": float(np.max(scales)) if scales.size else 0.0,
    }


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


def secondary_node_pressure_recovery_from_regions(
    sample_arrays: dict[str, np.ndarray] | None,
    *,
    n_nodes: int,
    pressure_stiffness: float,
) -> dict[str, dict[str, np.ndarray | float | int | str]] | None:
    """Recover Abaqus-style secondary-surface CPRESS/COPEN from regions.

    ``slave_node_region_*`` contact aggregation creates one row per secondary
    constraint region and stores the owning secondary node in
    ``secondary_node_ids``.  Abaqus-like pressure output for this path is a
    region quantity attached to that secondary node; it is not the raw
    quadrature-sample pressure scattered through every slave shape function.

    Returns ``None`` when the arrays do not carry secondary-region ownership,
    so callers can fall back to legacy sample-based aliases for non-region
    contact paths.
    """

    if sample_arrays is None or "secondary_node_ids" not in sample_arrays:
        return None
    count = int(n_nodes)
    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    secondary_ids = np.asarray(sample_arrays.get("secondary_node_ids", np.empty(0)), dtype=np.int64).reshape(-1)
    if gaps.shape != secondary_ids.shape:
        return None
    areas = np.asarray(sample_arrays.get("areas", np.zeros_like(gaps)), dtype=float).reshape(-1)
    if areas.shape != gaps.shape:
        raise ValueError("secondary pressure recovery requires one area per region gap")

    pressure = np.zeros(count, dtype=float)
    penetration = np.zeros(count, dtype=float)
    area_weight = np.zeros(count, dtype=float)
    active_hit = np.zeros(count, dtype=float)
    gap_min = np.full(count, np.inf, dtype=float)

    valid = (secondary_ids >= 0) & (secondary_ids < count)
    if np.any(valid):
        valid_ids = secondary_ids[valid]
        valid_gaps = gaps[valid]
        valid_areas = areas[valid]
        valid_penetration = np.maximum(-valid_gaps, 0.0)
        valid_pressure = float(pressure_stiffness) * valid_penetration
        valid_active = valid_penetration > 0.0
        np.add.at(pressure, valid_ids, valid_areas * valid_pressure)
        np.add.at(penetration, valid_ids, valid_areas * valid_penetration)
        np.add.at(area_weight, valid_ids, valid_areas)
        np.add.at(active_hit, valid_ids, valid_active.astype(float))
        np.minimum.at(gap_min, valid_ids, valid_gaps)

    nonzero = area_weight > 0.0
    pressure[nonzero] /= area_weight[nonzero]
    penetration[nonzero] /= area_weight[nonzero]
    active_node = (active_hit > 0.0).astype(float)
    gap = np.zeros(count, dtype=float)
    finite_gap = np.isfinite(gap_min)
    gap[finite_gap] = gap_min[finite_gap]
    active_pressure = pressure[active_node > 0.0]
    fields = {
        "contact_secondary_pressure_nodeavg": pressure,
        "contact_secondary_penetration_nodeavg": penetration,
        "contact_secondary_active_node": active_node,
        "contact_secondary_gap_min_node": gap,
        "contact_secondary_sample_area_weight": area_weight,
    }
    metrics: dict[str, float | int | str] = {
        "active_contact_secondary_node_count": int(np.count_nonzero(active_node)),
        "max_contact_secondary_pressure_nodeavg": (
            float(np.max(active_pressure)) if active_pressure.size else 0.0
        ),
        "p95_contact_secondary_pressure_nodeavg": _percentile_or_zero(active_pressure, 95.0),
        "mean_active_contact_secondary_pressure_nodeavg": (
            float(np.mean(active_pressure)) if active_pressure.size else 0.0
        ),
        "max_contact_secondary_penetration_nodeavg": (
            float(np.max(penetration)) if penetration.size else 0.0
        ),
        "min_contact_secondary_gap_node": float(np.min(gap[finite_gap])) if np.any(finite_gap) else 0.0,
        "contact_secondary_pressure_recovery_source": "constraint_region",
    }
    return {"fields": fields, "metrics": metrics}


def contact_path_tracking_metrics_from_arrays(
    sample_arrays: dict[str, np.ndarray] | None,
    previous_master_face_ids: np.ndarray | None,
    previous_master_barycentric: np.ndarray | None = None,
    *,
    active_gap_tolerance: float = 0.0,
) -> tuple[dict[str, float | int], np.ndarray | None, np.ndarray | None]:
    """Measure accepted-state master-face continuity for path tracking.

    The Lagrangian SDF surface-contact geometry carries the accepted master
    face id and representative barycentric payload for every secondary
    constraint region.  This diagnostic checks whether active regions keep a
    temporally coherent path across accepted states.  It is intentionally a
    verification metric; it does not alter contact search or projection.
    """

    empty = _empty_path_tracking_metrics()
    if sample_arrays is None or "master_face_ids" not in sample_arrays:
        return empty, None, None
    face_ids = np.asarray(sample_arrays.get("master_face_ids", np.empty(0)), dtype=np.int64).reshape(-1)
    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    if face_ids.size == 0 or gaps.size == 0:
        return empty, face_ids.copy(), None
    if face_ids.shape != gaps.shape:
        return empty, face_ids.copy(), None

    usable = (face_ids >= 0) & (gaps <= float(active_gap_tolerance))
    active_faces = face_ids[usable]
    comparable = np.zeros(face_ids.shape, dtype=bool)
    switches = np.zeros(face_ids.shape, dtype=bool)
    if previous_master_face_ids is not None:
        previous = np.asarray(previous_master_face_ids, dtype=np.int64).reshape(-1)
        if previous.shape == face_ids.shape:
            comparable = usable & (previous >= 0)
            switches = comparable & (previous != face_ids)
    comparable_count = int(np.count_nonzero(comparable))
    switch_count = int(np.count_nonzero(switches))
    metrics: dict[str, float | int] = {
        "contact_active_master_face_count": int(active_faces.size),
        "contact_master_face_unique_count": int(np.unique(active_faces).size) if active_faces.size else 0,
        "contact_master_face_tracking_comparable_count": comparable_count,
        "contact_master_face_switch_count": switch_count,
        "contact_master_face_switch_fraction": (
            float(switch_count) / float(comparable_count) if comparable_count else 0.0
        ),
    }
    cache_hits = np.asarray(
        sample_arrays.get("tracking_cache_hits", np.zeros(face_ids.shape, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    cache_matches = np.asarray(
        sample_arrays.get("tracking_cache_matches", np.zeros(face_ids.shape, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if cache_hits.shape == face_ids.shape:
        active_hits = cache_hits[usable]
        hit_count = int(np.count_nonzero(active_hits))
        metrics["contact_path_cache_hit_count"] = hit_count
        metrics["contact_path_cache_hit_fraction"] = (
            float(hit_count) / float(active_faces.size) if active_faces.size else 0.0
        )
    else:
        metrics["contact_path_cache_hit_count"] = 0
        metrics["contact_path_cache_hit_fraction"] = 0.0
    if cache_matches.shape == face_ids.shape:
        active_matches = cache_matches[usable]
        match_count = int(np.count_nonzero(active_matches))
        metrics["contact_path_cache_match_count"] = match_count
        metrics["contact_path_cache_match_fraction"] = (
            float(match_count) / float(active_faces.size) if active_faces.size else 0.0
        )
    else:
        metrics["contact_path_cache_match_count"] = 0
        metrics["contact_path_cache_match_fraction"] = 0.0

    bary = _representative_barycentric_from_arrays(sample_arrays, face_ids.size)
    current_bary = None
    if bary is not None:
        current_bary = np.asarray(bary, dtype=float).copy()
        previous_bary = None if previous_master_barycentric is None else np.asarray(previous_master_barycentric, dtype=float)
        if previous_bary is not None and previous_bary.shape == current_bary.shape and previous_master_face_ids is not None:
            previous_faces = np.asarray(previous_master_face_ids, dtype=np.int64).reshape(-1)
            if previous_faces.shape == face_ids.shape:
                finite_bary = np.all(np.isfinite(current_bary), axis=1) & np.all(np.isfinite(previous_bary), axis=1)
                bary_comparable = usable & finite_bary & (previous_faces == face_ids) & (face_ids >= 0)
                if np.any(bary_comparable):
                    drifts = np.linalg.norm(current_bary[bary_comparable] - previous_bary[bary_comparable], axis=1)
                    metrics["contact_master_barycentric_tracking_comparable_count"] = int(drifts.size)
                    metrics["contact_master_barycentric_drift_mean"] = float(np.mean(drifts))
                    metrics["contact_master_barycentric_drift_max"] = float(np.max(drifts))
                else:
                    metrics.update(_empty_barycentric_tracking_metrics())
            else:
                metrics.update(_empty_barycentric_tracking_metrics())
        else:
            metrics.update(_empty_barycentric_tracking_metrics())
    else:
        metrics.update(_empty_barycentric_tracking_metrics())
    return metrics, face_ids.copy(), current_bary


def contact_active_region_continuity_metrics_from_arrays(
    sample_arrays: dict[str, np.ndarray] | None,
    previous_active_region_ids: tuple[int, ...] | None,
    *,
    active_gap_tolerance: float = 0.0,
) -> tuple[dict[str, float | int], tuple[int, ...] | None]:
    """Measure active secondary constraint-region persistence."""

    empty = _empty_active_region_continuity_metrics()
    if sample_arrays is None:
        return empty, None
    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    if gaps.size == 0:
        return empty, tuple()
    secondary_ids = np.asarray(sample_arrays.get("secondary_node_ids", np.empty(0)), dtype=np.int64).reshape(-1)
    if secondary_ids.shape != gaps.shape or not np.any(secondary_ids >= 0):
        region_ids = np.arange(gaps.size, dtype=np.int64)
    else:
        region_ids = secondary_ids
    active_mask = (gaps <= float(active_gap_tolerance)) & (region_ids >= 0)
    current_ids = tuple(sorted({int(region_id) for region_id in region_ids[active_mask]}))
    if previous_active_region_ids is None:
        metrics = dict(empty)
        metrics["contact_active_region_continuity_current_count"] = int(len(current_ids))
        return metrics, current_ids
    previous_set = {int(region_id) for region_id in previous_active_region_ids}
    current_set = set(current_ids)
    intersection = previous_set & current_set
    union = previous_set | current_set
    previous_count = len(previous_set)
    current_count = len(current_set)
    metrics: dict[str, float | int] = {
        "contact_active_region_continuity_previous_count": int(previous_count),
        "contact_active_region_continuity_current_count": int(current_count),
        "contact_active_region_continuity_intersection_count": int(len(intersection)),
        "contact_active_region_new_count": int(len(current_set - previous_set)),
        "contact_active_region_dropped_count": int(len(previous_set - current_set)),
        "contact_active_region_persistence_fraction": (
            float(len(intersection)) / float(previous_count) if previous_count else 0.0
        ),
        "contact_active_region_jaccard": float(len(intersection)) / float(len(union)) if union else 1.0,
    }
    return metrics, current_ids


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
        "tracking_cache_hit": _representative_tracking_flag(tracking_cache_hits, source_row),
        "tracking_cache_match": _representative_tracking_flag(tracking_cache_matches, source_row),
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


def _scatter_weighted_vectors(
    force: np.ndarray,
    node_rows: np.ndarray,
    weight_rows: np.ndarray,
    vectors: np.ndarray,
    *,
    sign: float,
) -> None:
    for local in range(node_rows.shape[1] if node_rows.ndim == 2 else 0):
        nodes = node_rows[:, local]
        weights = weight_rows[:, local]
        valid = (nodes >= 0) & (nodes < force.shape[0]) & (np.abs(weights) > 0.0)
        if np.any(valid):
            np.add.at(force, nodes[valid], float(sign) * weights[valid, None] * vectors[valid])


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


def _representative_tracking_flag(values: np.ndarray | None, source_row: int) -> bool:
    if values is None or values.size == 0:
        return False
    flat = np.asarray(values, dtype=bool).reshape(-1)
    if not (0 <= int(source_row) < flat.size):
        return False
    return bool(flat[int(source_row)])


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


def _percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.percentile(flat, float(percentile)))


def _empty_path_tracking_metrics() -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "contact_active_master_face_count": 0,
        "contact_master_face_unique_count": 0,
        "contact_master_face_tracking_comparable_count": 0,
        "contact_master_face_switch_count": 0,
        "contact_master_face_switch_fraction": 0.0,
        "contact_path_cache_hit_count": 0,
        "contact_path_cache_hit_fraction": 0.0,
        "contact_path_cache_match_count": 0,
        "contact_path_cache_match_fraction": 0.0,
    }
    metrics.update(_empty_barycentric_tracking_metrics())
    return metrics


def _empty_barycentric_tracking_metrics() -> dict[str, float | int]:
    return {
        "contact_master_barycentric_tracking_comparable_count": 0,
        "contact_master_barycentric_drift_mean": 0.0,
        "contact_master_barycentric_drift_max": 0.0,
    }


def _empty_active_region_continuity_metrics() -> dict[str, float | int]:
    return {
        "contact_active_region_continuity_previous_count": 0,
        "contact_active_region_continuity_current_count": 0,
        "contact_active_region_continuity_intersection_count": 0,
        "contact_active_region_new_count": 0,
        "contact_active_region_dropped_count": 0,
        "contact_active_region_persistence_fraction": 0.0,
        "contact_active_region_jaccard": 0.0,
    }


def _representative_barycentric_from_arrays(sample_arrays: dict[str, np.ndarray], expected_rows: int) -> np.ndarray | None:
    bary_in = sample_arrays.get("master_barycentric")
    if bary_in is None:
        weights = np.asarray(sample_arrays.get("master_weights", np.empty((0, 0))), dtype=float)
        return weights if weights.ndim == 2 and weights.shape == (int(expected_rows), 3) else None
    bary = np.asarray(bary_in, dtype=float)
    return bary if bary.ndim == 2 and bary.shape[0] == int(expected_rows) else None


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
