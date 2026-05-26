from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import eye

from sfc.contact.constraint_region import (
    active_constraint_region_tangent_data_from_arrays,
    aggregate_contact_sample_arrays,
    contact_active_region_continuity_metrics_from_arrays,
    constraint_region_contact_tangent_sparse_from_arrays,
    constraint_region_gap_jacobian_sparse_from_arrays,
    constraint_region_penalty_response_from_arrays,
    constraint_region_pressure_tangent_scales_from_arrays,
    constraint_region_reduced_contact_tangent_sparse_from_arrays,
    constraint_region_reduced_gap_jacobian_sparse_from_arrays,
    constraint_region_tangent_finite_difference_metrics_from_arrays,
    contact_path_tracking_metrics_from_arrays,
    contact_region_path_tracking_metrics_from_arrays,
    contact_region_integral_metrics_from_arrays,
    secondary_node_pressure_recovery_from_regions,
)


def _raw_two_sample_arrays() -> dict[str, np.ndarray]:
    return {
        "sample_node_ids": np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        "sample_weights": np.asarray([[0.5, 0.5, 0.0], [0.25, 0.75, 0.0]], dtype=float),
        "gaps": np.asarray([-0.2, 0.1], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([2.0, 2.0], dtype=float),
        "master_node_ids": np.asarray([[10, 11, 12], [10, 11, 12]], dtype=np.int64),
        "master_weights": np.asarray([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]], dtype=float),
        "master_barycentric": np.asarray([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]], dtype=float),
        "master_face_ids": np.asarray([7, 8], dtype=np.int64),
        "secondary_cache_indices": np.asarray([3, 4], dtype=np.int64),
        "tracking_cache_hits": np.asarray([True, True], dtype=bool),
        "tracking_cache_matches": np.asarray([True, False], dtype=bool),
        "tracking_barycentric_distances": np.asarray([0.01, 0.04], dtype=float),
    }


def test_slave_node_region_constraint_uses_signed_average_gap_and_payload() -> None:
    regions = aggregate_contact_sample_arrays(_raw_two_sample_arrays(), "slave_node_region_constraint")

    assert regions is not None
    assert regions["secondary_node_ids"].tolist() == [0, 1]
    assert regions["areas"] == pytest.approx([1.5, 2.5])
    assert regions["gaps"] == pytest.approx([-0.1, -0.02])
    assert regions["normals"] == pytest.approx(np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))
    assert regions["sample_node_ids"][0].tolist() == [0, 1]
    assert regions["sample_weights"][0] == pytest.approx([5.0 / 12.0, 7.0 / 12.0])
    assert regions["sample_node_ids"][1].tolist() == [0, 1]
    assert regions["sample_weights"][1] == pytest.approx([0.35, 0.65])
    assert regions["master_weights"][0] == pytest.approx([1.0 / 6.0, 0.4, 13.0 / 30.0])
    assert regions["master_face_ids"].tolist() == [7, 8]
    assert regions["tracking_cache_hits"].tolist() == [True, True]
    assert regions["tracking_cache_matches"].tolist() == [True, False]


def test_constraint_region_totals_and_tangent_use_region_rows() -> None:
    regions = aggregate_contact_sample_arrays(_raw_two_sample_arrays(), "slave_node_region_constraint")
    assert regions is not None

    metrics = contact_region_integral_metrics_from_arrays(regions, stiffness=100.0)
    row_ids, jacobian = constraint_region_gap_jacobian_sparse_from_arrays(regions, n_nodes=13, active_only=True)
    active_ids, scales = constraint_region_pressure_tangent_scales_from_arrays(
        regions,
        pressure_stiffness=100.0,
        equilibrium_scale=0.95,
    )

    assert metrics["contact_region_count"] == 2
    assert metrics["active_contact_region_count"] == 2
    assert metrics["contact_region_normal_force"] == pytest.approx(20.0)
    assert metrics["contact_region_energy"] == pytest.approx(0.8)
    assert metrics["contact_region_virtual_work"] == pytest.approx(1.6)
    assert row_ids.tolist() == [0, 1]
    assert jacobian.shape == (2, 39)
    assert active_ids.tolist() == [0, 1]
    assert scales == pytest.approx([142.5, 237.5])


def test_constraint_region_contact_tangent_is_fixed_active_set_jtwj() -> None:
    regions = aggregate_contact_sample_arrays(_raw_two_sample_arrays(), "slave_node_region_constraint")
    assert regions is not None

    active_ids, jacobian, scales, tangent, metrics = constraint_region_contact_tangent_sparse_from_arrays(
        regions,
        n_nodes=13,
        pressure_stiffness=100.0,
        equilibrium_scale=0.95,
    )

    dense_j = jacobian.toarray()
    expected = dense_j.T @ (scales[:, None] * dense_j)
    assert active_ids.tolist() == [0, 1]
    assert scales == pytest.approx([142.5, 237.5])
    np.testing.assert_allclose(tangent.toarray(), expected)
    np.testing.assert_allclose(tangent.toarray(), tangent.toarray().T)
    assert metrics["contact_tangent_source"] == "constraint_region_arrays"
    assert metrics["contact_tangent_gap_jacobian_source"] == "constraint_region_fixed_payload"
    assert metrics["contact_tangent_pressure_derivative"] == "linear_penalty_active_set"
    assert metrics["contact_tangent_sign_convention"] == "d(-contact_force)/du"
    assert metrics["contact_tangent_fixed_active_set"] == 1
    assert metrics["contact_tangent_active_region_count"] == 2
    assert metrics["contact_tangent_active_secondary_node_count"] == 2
    assert metrics["contact_tangent_j_nnz"] == jacobian.nnz
    assert metrics["contact_tangent_matrix_nnz"] == tangent.nnz


def test_constraint_region_tangent_matches_fixed_payload_force_finite_difference() -> None:
    regions = aggregate_contact_sample_arrays(_raw_two_sample_arrays(), "slave_node_region_constraint")
    assert regions is not None
    _, gap_jacobian = constraint_region_gap_jacobian_sparse_from_arrays(regions, n_nodes=13, active_only=False)
    _, _, _, tangent, _ = constraint_region_contact_tangent_sparse_from_arrays(
        regions,
        n_nodes=13,
        pressure_stiffness=100.0,
        equilibrium_scale=1.0,
    )
    direction = np.zeros(39, dtype=float)
    direction[2] = 0.01
    direction[5] = -0.02
    direction[30] = 0.015
    direction[35] = -0.01
    eps = 1.0e-6

    def force_at(scale: float) -> np.ndarray:
        shifted = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in regions.items()}
        shifted["gaps"] = np.asarray(regions["gaps"], dtype=float) + np.asarray(gap_jacobian @ (scale * direction)).reshape(-1)
        assert np.all(np.asarray(shifted["gaps"], dtype=float) < 0.0)
        return constraint_region_penalty_response_from_arrays(
            shifted,
            n_nodes=13,
            pressure_stiffness=100.0,
            include_tangent=False,
        ).force.reshape(-1)

    finite_difference_contact_force = (force_at(eps) - force_at(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(
        finite_difference_contact_force,
        -np.asarray(tangent @ direction).reshape(-1),
        rtol=1.0e-8,
        atol=1.0e-9,
    )


def test_constraint_region_tangent_fd_metrics_gate_fixed_active_set_consistency() -> None:
    regions = aggregate_contact_sample_arrays(_raw_two_sample_arrays(), "slave_node_region_constraint")
    assert regions is not None

    metrics = constraint_region_tangent_finite_difference_metrics_from_arrays(
        regions,
        n_nodes=13,
        pressure_stiffness=100.0,
        epsilon=1.0e-7,
        relative_tolerance=1.0e-7,
        absolute_tolerance=1.0e-8,
    )

    assert metrics["contact_tangent_fd_checked"] == 1
    assert metrics["contact_tangent_fd_active_set_stable"] == 1
    assert metrics["contact_tangent_fd_passed"] == 1
    assert metrics["contact_tangent_fd_error_rel"] < 1.0e-9
    assert metrics["contact_tangent_fd_source"] == "constraint_region_fixed_payload_force_difference"


def test_constraint_region_tangent_fd_metrics_reject_active_set_crossing() -> None:
    regions = aggregate_contact_sample_arrays(_raw_two_sample_arrays(), "slave_node_region_constraint")
    assert regions is not None
    direction = np.zeros(39, dtype=float)
    direction[2] = 1.0

    metrics = constraint_region_tangent_finite_difference_metrics_from_arrays(
        regions,
        n_nodes=13,
        pressure_stiffness=100.0,
        epsilon=1.0,
        direction=direction,
    )

    assert metrics["contact_tangent_fd_checked"] == 1
    assert metrics["contact_tangent_fd_active_set_stable"] == 0
    assert metrics["contact_tangent_fd_passed"] == 0


def test_constraint_region_penalty_response_uses_region_rows_for_force_and_totals() -> None:
    regions = aggregate_contact_sample_arrays(_raw_two_sample_arrays(), "slave_node_region_constraint")
    assert regions is not None

    response = constraint_region_penalty_response_from_arrays(
        regions,
        n_nodes=13,
        pressure_stiffness=100.0,
    )

    assert response.active_count == 2
    assert response.max_penetration == pytest.approx(0.1)
    assert response.normal_force == pytest.approx(20.0)
    assert response.energy == pytest.approx(0.8)
    assert response.virtual_work == pytest.approx(1.6)
    assert response.active_area == pytest.approx(4.0)
    assert response.metrics["active_contact_region_count"] == 2
    assert response.metrics["contact_region_normal_force"] == pytest.approx(20.0)
    assert response.force.shape == (13, 3)
    assert response.tangent.shape == (39, 39)
    np.testing.assert_allclose(np.sum(response.force[:3], axis=0), [0.0, 0.0, 20.0])
    np.testing.assert_allclose(np.sum(response.force[10:13], axis=0), [0.0, 0.0, -20.0])


def test_constraint_region_penalty_response_is_not_independent_sample_penalty() -> None:
    raw = _raw_two_sample_arrays()
    independent = constraint_region_penalty_response_from_arrays(raw, n_nodes=13, pressure_stiffness=100.0)
    regions = aggregate_contact_sample_arrays(raw, "slave_node_region_constraint")
    assert regions is not None
    averaged = constraint_region_penalty_response_from_arrays(regions, n_nodes=13, pressure_stiffness=100.0)

    assert independent.normal_force == pytest.approx(40.0)
    assert averaged.normal_force == pytest.approx(20.0)
    assert averaged.active_count == 2
    assert independent.active_count == 1


def test_reduced_constraint_region_tangent_applies_transformation_and_free_dofs() -> None:
    regions = aggregate_contact_sample_arrays(_raw_two_sample_arrays(), "slave_node_region_constraint")
    assert regions is not None
    transformation = eye(39, format="csr")
    free = np.asarray([2, 5, 8, 30, 32, 35], dtype=np.int64)

    full_ids, full_j = constraint_region_gap_jacobian_sparse_from_arrays(regions, n_nodes=13, active_only=True)
    reduced_ids, reduced_j = constraint_region_reduced_gap_jacobian_sparse_from_arrays(
        regions,
        transformation=transformation,
        free=free,
    )
    tangent_ids, tangent_j, scales = active_constraint_region_tangent_data_from_arrays(
        regions,
        transformation=transformation,
        free=free,
        pressure_stiffness=100.0,
        equilibrium_scale=0.95,
    )
    active_ids, reduced_j2, scales2, tangent, metrics = constraint_region_reduced_contact_tangent_sparse_from_arrays(
        regions,
        transformation=transformation,
        free=free,
        pressure_stiffness=100.0,
        equilibrium_scale=0.95,
    )

    np.testing.assert_array_equal(reduced_ids, full_ids)
    np.testing.assert_allclose(reduced_j.toarray(), full_j[:, free].toarray())
    np.testing.assert_array_equal(tangent_ids, reduced_ids)
    np.testing.assert_allclose(tangent_j.toarray(), reduced_j.toarray())
    np.testing.assert_allclose(scales, [142.5, 237.5])
    np.testing.assert_array_equal(active_ids, tangent_ids)
    np.testing.assert_allclose(reduced_j2.toarray(), tangent_j.toarray())
    np.testing.assert_allclose(scales2, scales)
    expected = tangent_j.toarray().T @ (scales[:, None] * tangent_j.toarray())
    np.testing.assert_allclose(tangent.toarray(), expected)
    assert metrics["contact_tangent_coordinate_space"] == "reduced_free"
    assert metrics["contact_tangent_j_nnz"] == tangent_j.nnz


def test_secondary_pressure_recovery_uses_region_owner_nodes() -> None:
    regions = aggregate_contact_sample_arrays(_raw_two_sample_arrays(), "slave_node_region_constraint")
    assert regions is not None

    diagnostics = secondary_node_pressure_recovery_from_regions(
        regions,
        n_nodes=13,
        pressure_stiffness=100.0,
    )

    assert diagnostics is not None
    fields = diagnostics["fields"]
    metrics = diagnostics["metrics"]
    pressure = fields["contact_secondary_pressure_nodeavg"]
    penetration = fields["contact_secondary_penetration_nodeavg"]
    active = fields["contact_secondary_active_node"]
    np.testing.assert_allclose(pressure[:3], [10.0, 2.0, 0.0])
    np.testing.assert_allclose(penetration[:3], [0.1, 0.02, 0.0])
    np.testing.assert_allclose(active[:3], [1.0, 1.0, 0.0])
    assert metrics["active_contact_secondary_node_count"] == 2
    assert metrics["max_contact_secondary_pressure_nodeavg"] == pytest.approx(10.0)
    assert metrics["mean_active_contact_secondary_pressure_nodeavg"] == pytest.approx(6.0)
    assert metrics["contact_secondary_pressure_recovery_source"] == "constraint_region"


def test_path_tracking_metrics_report_active_face_switch_and_cache_rates() -> None:
    arrays = {
        "gaps": np.asarray([-0.10, -0.02, 0.01], dtype=float),
        "master_face_ids": np.asarray([3, 4, 5], dtype=np.int64),
        "master_barycentric": np.asarray([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]], dtype=float),
        "tracking_cache_hits": np.asarray([True, True, False], dtype=bool),
        "tracking_cache_matches": np.asarray([True, False, False], dtype=bool),
    }
    previous = np.asarray([3, 7, 8], dtype=np.int64)
    previous_bary = np.asarray([[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.2, 0.2, 0.6]], dtype=float)

    metrics, current, current_bary = contact_path_tracking_metrics_from_arrays(arrays, previous, previous_bary)

    assert current is not None
    assert current_bary is not None
    np.testing.assert_array_equal(current, [3, 4, 5])
    np.testing.assert_allclose(current_bary[0], [0.7, 0.2, 0.1])
    assert metrics["contact_active_master_face_count"] == 2
    assert metrics["contact_master_face_tracking_comparable_count"] == 2
    assert metrics["contact_master_face_switch_count"] == 1
    assert metrics["contact_master_face_switch_fraction"] == pytest.approx(0.5)
    assert metrics["contact_master_face_invalid_jump_count"] == 1
    assert metrics["contact_master_barycentric_tracking_comparable_count"] == 1
    assert metrics["contact_master_barycentric_drift_mean"] == pytest.approx(np.sqrt(0.02))
    assert metrics["contact_master_barycentric_drift_max"] == pytest.approx(np.sqrt(0.02))
    assert metrics["contact_path_cache_hit_count"] == 2
    assert metrics["contact_path_cache_hit_fraction"] == pytest.approx(1.0)
    assert metrics["contact_path_cache_match_count"] == 1
    assert metrics["contact_path_cache_match_fraction"] == pytest.approx(0.5)


def test_path_tracking_default_active_set_matches_penalty_closed_regions() -> None:
    arrays = {
        "gaps": np.asarray([-0.10, 0.0, 0.02], dtype=float),
        "master_face_ids": np.asarray([3, 4, 5], dtype=np.int64),
        "tracking_cache_hits": np.asarray([True, True, True], dtype=bool),
        "tracking_cache_matches": np.asarray([True, True, True], dtype=bool),
    }
    previous = np.asarray([3, 4, 5], dtype=np.int64)

    default_metrics, _, _ = contact_path_tracking_metrics_from_arrays(arrays, previous)
    tolerant_metrics, _, _ = contact_path_tracking_metrics_from_arrays(
        arrays,
        previous,
        active_gap_tolerance=0.02,
    )

    assert default_metrics["contact_active_master_face_count"] == 1
    assert default_metrics["contact_path_cache_hit_fraction"] == pytest.approx(1.0)
    assert tolerant_metrics["contact_active_master_face_count"] == 3


def test_region_path_tracking_aligns_by_secondary_node_not_row_order() -> None:
    previous_regions = np.asarray([10, 20], dtype=np.int64)
    previous_faces = np.asarray([3, 4], dtype=np.int64)
    previous_bary = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]], dtype=float)
    arrays = {
        "secondary_node_ids": np.asarray([20, 10], dtype=np.int64),
        "gaps": np.asarray([-0.02, -0.01], dtype=float),
        "master_face_ids": np.asarray([4, 3], dtype=np.int64),
        "master_barycentric": np.asarray([[0.2, 0.65, 0.15], [0.65, 0.25, 0.10]], dtype=float),
        "tracking_cache_hits": np.asarray([True, True], dtype=bool),
        "tracking_cache_matches": np.asarray([True, True], dtype=bool),
    }

    metrics, current_regions, current_faces, current_bary = contact_region_path_tracking_metrics_from_arrays(
        arrays,
        previous_regions,
        previous_faces,
        previous_bary,
    )

    np.testing.assert_array_equal(current_regions, [20, 10])
    np.testing.assert_array_equal(current_faces, [4, 3])
    assert current_bary is not None
    assert metrics["contact_master_face_tracking_region_aligned"] == 1
    assert metrics["contact_master_face_tracking_comparable_count"] == 2
    assert metrics["contact_master_face_switch_count"] == 0
    assert metrics["contact_master_face_switch_fraction"] == pytest.approx(0.0)
    assert metrics["contact_master_barycentric_tracking_comparable_count"] == 2


def test_region_path_tracking_reports_real_switch_after_region_alignment() -> None:
    arrays = {
        "secondary_node_ids": np.asarray([20, 10], dtype=np.int64),
        "gaps": np.asarray([-0.02, -0.01], dtype=float),
        "master_face_ids": np.asarray([8, 3], dtype=np.int64),
        "tracking_cache_hits": np.asarray([True, True], dtype=bool),
        "tracking_cache_matches": np.asarray([False, True], dtype=bool),
    }

    metrics, _, _, _ = contact_region_path_tracking_metrics_from_arrays(
        arrays,
        np.asarray([10, 20], dtype=np.int64),
        np.asarray([3, 4], dtype=np.int64),
    )

    assert metrics["contact_master_face_tracking_region_aligned"] == 1
    assert metrics["contact_master_face_tracking_comparable_count"] == 2
    assert metrics["contact_master_face_switch_count"] == 1
    assert metrics["contact_master_face_switch_fraction"] == pytest.approx(0.5)
    assert metrics["contact_master_face_invalid_jump_count"] == 1

    neighbor_metrics, _, _, _ = contact_region_path_tracking_metrics_from_arrays(
        arrays,
        np.asarray([10, 20], dtype=np.int64),
        np.asarray([3, 4], dtype=np.int64),
        master_face_neighbors=(np.asarray([3], dtype=np.int64),) * 4 + (np.asarray([4, 8], dtype=np.int64),),
    )

    assert neighbor_metrics["contact_master_face_switch_count"] == 1
    assert neighbor_metrics["contact_master_face_topological_continuity_count"] == 2
    assert neighbor_metrics["contact_master_face_topological_continuity_fraction"] == pytest.approx(1.0)
    assert neighbor_metrics["contact_master_face_invalid_jump_count"] == 0


def test_active_region_continuity_metrics_use_secondary_constraint_regions() -> None:
    arrays = {
        "gaps": np.asarray([-0.10, -0.02, 0.01, -0.03], dtype=float),
        "secondary_node_ids": np.asarray([1, 2, 3, 4], dtype=np.int64),
    }
    previous = (1, 5)

    metrics, current = contact_active_region_continuity_metrics_from_arrays(arrays, previous)

    assert current == (1, 2, 4)
    assert metrics["contact_active_region_continuity_previous_count"] == 2
    assert metrics["contact_active_region_continuity_current_count"] == 3
    assert metrics["contact_active_region_continuity_intersection_count"] == 1
    assert metrics["contact_active_region_new_count"] == 2
    assert metrics["contact_active_region_dropped_count"] == 1
    assert metrics["contact_active_region_persistence_fraction"] == pytest.approx(0.5)
    assert metrics["contact_active_region_jaccard"] == pytest.approx(0.25)


def test_active_region_continuity_default_excludes_zero_gap_regions() -> None:
    arrays = {
        "gaps": np.asarray([-0.10, 0.0, 0.01], dtype=float),
        "secondary_node_ids": np.asarray([1, 2, 3], dtype=np.int64),
    }

    default_metrics, default_current = contact_active_region_continuity_metrics_from_arrays(
        arrays,
        previous_active_region_ids=None,
    )
    tolerant_metrics, tolerant_current = contact_active_region_continuity_metrics_from_arrays(
        arrays,
        previous_active_region_ids=None,
        active_gap_tolerance=0.01,
    )

    assert default_current == (1,)
    assert default_metrics["contact_active_region_continuity_current_count"] == 1
    assert tolerant_current == (1, 2, 3)
    assert tolerant_metrics["contact_active_region_continuity_current_count"] == 3
