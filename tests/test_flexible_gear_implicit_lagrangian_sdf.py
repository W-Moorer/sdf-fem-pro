from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from scipy.sparse import identity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flexible_gear_explicit_sdf_comparison import DEFAULT_SOURCE, parse_gear_input
from validation.run_flexible_gear_implicit_lagrangian_sdf_comparison import (
    _active_reduced_gap_jacobian_sparse,
    _active_reduced_gap_jacobian_sparse_from_arrays,
    _active_constraint_region_tangent_data_from_arrays,
    _ContactAggregationWorkspace,
    _aggregate_contact_samples,
    _aggregate_contact_sample_arrays,
    _assemble_contact_arrays_force_only,
    _assemble_contact_response_force_only,
    _contact_active_signature_from_arrays,
    _contact_active_signature_from_samples,
    _contact_node_diagnostics_from_arrays,
    _contact_active_region_continuity_metrics_from_arrays,
    _contact_path_tracking_metrics_from_arrays,
    _contact_region_integral_metrics_from_arrays,
    _contact_patch_representative_length,
    _constraint_region_gap_jacobian_sparse_from_arrays,
    _constraint_region_pressure_tangent_scales_from_arrays,
    _constraint_region_tangent_metrics_from_arrays,
    _contact_samples_from_arrays,
    _default_contact_search_radius,
    _default_secondary_contact_tracking_radius,
    _default_secondary_line_hard_distance_limit,
    _default_secondary_line_distance_limit,
    _filter_contact_samples_by_normal_compatibility,
    _node_average_cell_scalar,
    _node_average_cell_tensor,
    _replace_contact_normals_with_secondary_average,
    _surface_edge_length_percentile,
    _source_drive_corotated_elastic_matrix,
    _source_drive_corotated_positions_and_elastic_displacement,
    _source_drive_centripetal_acceleration,
    _source_drive_centripetal_reduced_response,
    _source_drive_corotated_visual_state_and_internal,
    _source_drive_finite_kinematic_inertia_response,
    _source_drive_finite_visual_jacobian,
    _source_contact_active_set_is_stable,
    _source_increment_convergence_decision,
    _source_increment_cutback_candidate_dt,
    _source_increment_gate_row,
    _penalty_history_row,
    _write_csv,
    _write_abaqus_alignment_deck,
    build_cropped_pair,
    solve_sfc_cropped_pair,
    solve_sfc_cropped_pair_hard_contact,
    solve_sfc_source_drive_pair,
)
from sfc.contact.hard_contact import hard_contact_gap_jacobian_from_samples
from sfc.contact.lagrangian_surface_contact import LagrangianSDFSurfaceContactGeometry
from sfc.fem.calculix_aligned import ContactSample, MechanicsModel, MechanicsState, assemble_contact_response, stvk_internal_response
from sfc.fem.rp_mpc import RigidHubMPC, build_rigid_hub_reduced_assembly
from sfc.sdf import _cpp_projection
from sfc.sdf.material_sdf import MaterialSDF
from validation.run_flexible_gear_full_lagrangian_sdf_comparison import (
    build_full_active_pair,
    compare_sfc_history_to_abaqus_manifest,
    compare_animation_manifests,
    write_animation_color_ranges,
    write_paraview_animation_setup,
)
from validation.run_source_gear_vtk_manifest_alignment import align_manifests
from validation.run_source_gear_vtk_regional_alignment import _active_overlap_metrics


pytestmark = pytest.mark.skipif(not DEFAULT_SOURCE.exists(), reason="commercial gear input is not present")


def test_node_average_cell_scalar_matches_incident_cell_average() -> None:
    cells = np.asarray([[0, 1, 2, 3], [1, 4, 2, 5], [1, 5, 2, 6]], dtype=np.int64)
    values = np.asarray([2.0, 5.0, 11.0], dtype=float)

    averaged = _node_average_cell_scalar(values, cells, node_count=7)

    assert averaged[0] == pytest.approx(2.0)
    assert averaged[1] == pytest.approx((2.0 + 5.0 + 11.0) / 3.0)
    assert averaged[2] == pytest.approx((2.0 + 5.0 + 11.0) / 3.0)
    assert averaged[3] == pytest.approx(2.0)
    assert averaged[4] == pytest.approx(5.0)
    assert averaged[5] == pytest.approx((5.0 + 11.0) / 2.0)
    assert averaged[6] == pytest.approx(11.0)


def test_node_average_cell_tensor_averages_components_before_invariant() -> None:
    cells = np.asarray([[0, 1, 2, 3], [1, 4, 2, 5]], dtype=np.int64)
    tensors = np.zeros((2, 3, 3), dtype=float)
    tensors[0, 0, 0] = 10.0
    tensors[1, 1, 1] = 10.0

    averaged = _node_average_cell_tensor(tensors, cells, node_count=6)

    np.testing.assert_allclose(averaged[0], tensors[0])
    assert averaged[1, 0, 0] == pytest.approx(5.0)
    assert averaged[1, 1, 1] == pytest.approx(5.0)
    assert averaged[2, 0, 0] == pytest.approx(5.0)
    assert averaged[2, 1, 1] == pytest.approx(5.0)
    np.testing.assert_allclose(averaged[4], tensors[1])


def test_default_contact_search_radius_uses_surface_feature_size() -> None:
    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    pair = SimpleNamespace(
        initial_patch_gap=1.0e-8,
        gear1=SimpleNamespace(nodes=nodes, contact_faces=faces),
        gear2=SimpleNamespace(nodes=nodes.copy(), contact_faces=faces.copy()),
    )

    feature = _surface_edge_length_percentile(nodes, faces)
    radius = _default_contact_search_radius(pair, target_overclosure=0.0)

    assert feature > 0.0
    assert radius == pytest.approx(2.5 * feature)
    assert radius > 1.0


def test_secondary_contact_tracking_radius_covers_local_constraint_region() -> None:
    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    pair = SimpleNamespace(
        initial_patch_gap=2.0e-5,
        gear1=SimpleNamespace(nodes=nodes, contact_faces=faces),
        gear2=SimpleNamespace(nodes=nodes.copy(), contact_faces=faces.copy()),
    )

    closest_radius = _default_contact_search_radius(pair, target_overclosure=1.0e-5)
    tracking_radius = _default_secondary_contact_tracking_radius(pair, target_overclosure=1.0e-5)
    representative = _contact_patch_representative_length(pair)

    assert closest_radius > 1.0
    assert tracking_radius >= representative
    assert tracking_radius >= 2.0 * representative
    assert tracking_radius >= closest_radius
    assert tracking_radius == pytest.approx(closest_radius)


def test_source_force_only_contact_response_matches_full_force_response() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.2, 0.3, 0.5], dtype=float),
            gap=-0.01,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=0.25,
            stiffness=1000.0,
            master_node_ids=np.asarray([3, 4, 5], dtype=np.int64),
            master_shape_weights=np.asarray([0.4, 0.4, 0.2], dtype=float),
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([1.0 / 3.0] * 3, dtype=float),
            gap=0.02,
            normal=np.asarray([0.0, 1.0, 0.0], dtype=float),
            area=0.25,
            stiffness=1000.0,
        ),
    ]

    full = assemble_contact_response(samples, n_nodes=6)
    fast = _assemble_contact_response_force_only(samples, n_nodes=6)
    arrays = {
        "sample_node_ids": np.vstack([sample.node_ids for sample in samples]),
        "sample_weights": np.vstack([sample.shape_weights for sample in samples]),
        "gaps": np.asarray([sample.gap for sample in samples], dtype=float),
        "normals": np.vstack([sample.normal for sample in samples]),
        "areas": np.asarray([sample.area for sample in samples], dtype=float),
        "master_node_ids": np.vstack(
            [
                sample.master_node_ids if sample.master_node_ids is not None else np.asarray([0, 0, 0], dtype=np.int64)
                for sample in samples
            ]
        ),
        "master_weights": np.vstack(
            [
                sample.master_shape_weights if sample.master_shape_weights is not None else np.zeros(3, dtype=float)
                for sample in samples
            ]
        ),
    }
    batched = _assemble_contact_arrays_force_only(arrays, n_nodes=6, stiffness=1000.0)

    np.testing.assert_allclose(fast.force, full.force)
    np.testing.assert_allclose(batched.force, full.force)
    assert fast.min_gap == pytest.approx(full.min_gap)
    assert batched.min_gap == pytest.approx(full.min_gap)
    assert fast.max_penetration == pytest.approx(full.max_penetration)
    assert batched.max_penetration == pytest.approx(full.max_penetration)
    assert fast.active_count == full.active_count
    assert batched.active_count == full.active_count
    assert fast.normal_force == pytest.approx(full.normal_force)
    assert batched.normal_force == pytest.approx(full.normal_force)
    assert fast.energy == pytest.approx(full.energy)
    assert batched.energy == pytest.approx(full.energy)
    assert fast.tangent.nnz == 0
    assert batched.tangent.nnz == 0


def test_slave_face_contact_averaging_combines_tri3_samples_without_tuning() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], dtype=float),
            gap=-0.30,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=2.0,
            stiffness=10.0,
            master_node_ids=np.asarray([4, 5, 6], dtype=np.int64),
            master_shape_weights=np.asarray([1.0, 0.0, 0.0], dtype=float),
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=float),
            gap=0.00,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=2.0,
            stiffness=10.0,
            master_node_ids=np.asarray([4, 5, 6], dtype=np.int64),
            master_shape_weights=np.asarray([0.0, 1.0, 0.0], dtype=float),
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], dtype=float),
            gap=0.15,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=2.0,
            stiffness=10.0,
            master_node_ids=np.asarray([4, 5, 6], dtype=np.int64),
            master_shape_weights=np.asarray([0.0, 0.0, 1.0], dtype=float),
        ),
    ]

    aggregated = _aggregate_contact_samples(samples, "slave_face")

    assert len(aggregated) == 1
    merged = aggregated[0]
    assert merged.area == pytest.approx(6.0)
    assert merged.gap == pytest.approx(-0.30 / 3.0)
    assert np.sum(merged.shape_weights) == pytest.approx(1.0)
    assert np.sum(merged.master_shape_weights) == pytest.approx(1.0)
    np.testing.assert_allclose(merged.shape_weights, [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
    np.testing.assert_allclose(merged.master_shape_weights, [1.0, 0.0, 0.0])
    assert merged.node_ids.tolist() == [0, 1, 2]
    assert merged.master_node_ids.tolist() == [4, 5, 6]


def test_linear_penalty_participation_preserves_force_and_energy() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.7, 0.2, 0.1], dtype=float),
            gap=-0.20,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=2.0,
            stiffness=100.0,
            master_node_ids=np.asarray([4, 5, 6], dtype=np.int64),
            master_shape_weights=np.asarray([0.8, 0.1, 0.1], dtype=float),
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.2, 0.6, 0.2], dtype=float),
            gap=-0.05,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=3.0,
            stiffness=100.0,
            master_node_ids=np.asarray([4, 5, 6], dtype=np.int64),
            master_shape_weights=np.asarray([0.1, 0.8, 0.1], dtype=float),
        ),
    ]

    distributed = _assemble_contact_response_force_only(samples, n_nodes=7)
    aggregated = _aggregate_contact_samples(samples, "slave_face_participation")
    reduced = _assemble_contact_response_force_only(aggregated, n_nodes=7)

    assert len(aggregated) == 1
    assert reduced.normal_force == pytest.approx(distributed.normal_force)
    assert reduced.energy == pytest.approx(distributed.energy)
    np.testing.assert_allclose(reduced.force.sum(axis=0), distributed.force.sum(axis=0), atol=1.0e-12)
    assert aggregated[0].area < sum(sample.area for sample in samples)


def test_signed_participation_uses_region_status_before_point_pressure() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.7, 0.2, 0.1], dtype=float),
            gap=-0.10,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=1.0,
            stiffness=100.0,
            master_node_ids=np.asarray([4, 5, 6], dtype=np.int64),
            master_shape_weights=np.asarray([0.8, 0.1, 0.1], dtype=float),
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.2, 0.6, 0.2], dtype=float),
            gap=0.20,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=1.0,
            stiffness=100.0,
            master_node_ids=np.asarray([4, 5, 6], dtype=np.int64),
            master_shape_weights=np.asarray([0.1, 0.8, 0.1], dtype=float),
        ),
    ]

    pointwise = _aggregate_contact_samples(samples, "slave_face_participation")[0]
    signed_status = _aggregate_contact_samples(samples, "slave_face_signed_participation")[0]

    assert pointwise.gap < 0.0
    assert signed_status.gap == pytest.approx(0.05)


def test_signed_participation_preserves_closed_region_virtual_work() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.7, 0.2, 0.1], dtype=float),
            gap=-0.30,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=1.0,
            stiffness=100.0,
            master_node_ids=np.asarray([4, 5, 6], dtype=np.int64),
            master_shape_weights=np.asarray([0.8, 0.1, 0.1], dtype=float),
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.2, 0.6, 0.2], dtype=float),
            gap=0.10,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=1.0,
            stiffness=100.0,
            master_node_ids=np.asarray([4, 5, 6], dtype=np.int64),
            master_shape_weights=np.asarray([0.1, 0.8, 0.1], dtype=float),
        ),
    ]

    participation = _aggregate_contact_samples(samples, "slave_face_participation")
    signed_status = _aggregate_contact_samples(samples, "slave_face_signed_participation")
    participation_response = _assemble_contact_response_force_only(participation, n_nodes=7)
    signed_response = _assemble_contact_response_force_only(signed_status, n_nodes=7)

    assert signed_status[0].gap < 0.0
    assert signed_response.normal_force == pytest.approx(participation_response.normal_force)
    assert signed_response.energy == pytest.approx(participation_response.energy)


def test_array_participation_matches_object_participation() -> None:
    arrays = {
        "sample_node_ids": np.asarray([[0, 1, 2], [0, 1, 2], [1, 2, 3]], dtype=np.int64),
        "sample_weights": np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], dtype=float),
        "gaps": np.asarray([-0.20, -0.05, 0.03], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([2.0, 3.0, 1.0], dtype=float),
        "master_node_ids": np.asarray([[4, 5, 6], [4, 5, 6], [5, 6, 7]], dtype=np.int64),
        "master_weights": np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]], dtype=float),
    }
    object_samples = _aggregate_contact_samples(
        _contact_samples_from_arrays(arrays, stiffness=100.0),
        "slave_node_region_participation",
    )

    array_samples = _aggregate_contact_sample_arrays(arrays, "slave_node_region_participation")

    assert array_samples is not None
    object_response = _assemble_contact_response_force_only(object_samples, n_nodes=8)
    array_response = _assemble_contact_arrays_force_only(array_samples, n_nodes=8, stiffness=100.0)
    np.testing.assert_allclose(array_response.force, object_response.force)
    assert array_response.normal_force == pytest.approx(object_response.normal_force)
    assert array_response.energy == pytest.approx(object_response.energy)
    assert _contact_active_signature_from_arrays(array_samples) == _contact_active_signature_from_samples(object_samples)


def test_array_constraint_region_carries_master_face_ids() -> None:
    arrays = {
        "sample_node_ids": np.asarray([[0, 1, 2], [0, 1, 2], [1, 2, 3]], dtype=np.int64),
        "sample_weights": np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], dtype=float),
        "gaps": np.asarray([-0.20, -0.05, 0.03], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([2.0, 3.0, 1.0], dtype=float),
        "master_node_ids": np.asarray([[4, 5, 6], [4, 5, 6], [5, 6, 7]], dtype=np.int64),
        "master_weights": np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]], dtype=float),
        "master_barycentric": np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]], dtype=float),
        "master_face_ids": np.asarray([10, 11, 12], dtype=np.int64),
    }

    aggregated = _aggregate_contact_sample_arrays(arrays, "slave_node_region_participation")

    assert aggregated is not None
    assert "master_face_ids" in aggregated
    assert np.asarray(aggregated["master_face_ids"], dtype=np.int64).size == np.asarray(aggregated["gaps"]).size
    assert set(np.asarray(aggregated["master_face_ids"], dtype=np.int64).tolist()).issubset({10, 11, 12})
    assert "master_barycentric" in aggregated
    assert np.asarray(aggregated["master_barycentric"], dtype=float).shape == (np.asarray(aggregated["gaps"]).size, 3)
    assert "secondary_node_ids" in aggregated
    assert set(np.asarray(aggregated["secondary_node_ids"], dtype=np.int64).tolist()).issubset({0, 1, 2, 3})


def test_contact_pressure_recovery_exposes_secondary_surface_aliases() -> None:
    arrays = {
        "sample_node_ids": np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        "sample_weights": np.asarray([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]], dtype=float),
        "gaps": np.asarray([-0.10, 0.05], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([2.0, 2.0], dtype=float),
        "master_node_ids": np.asarray([[4, 5, 6], [5, 6, 7]], dtype=np.int64),
        "master_weights": np.asarray([[0.3, 0.4, 0.3], [0.2, 0.2, 0.6]], dtype=float),
        "master_face_ids": np.asarray([20, 21], dtype=np.int64),
    }

    diagnostics = _contact_node_diagnostics_from_arrays(arrays, n_nodes=8, stiffness=100.0)

    fields = diagnostics["fields"]
    metrics = diagnostics["metrics"]
    np.testing.assert_allclose(fields["contact_secondary_pressure_nodeavg"], fields["contact_slave_pressure_nodeavg"])
    np.testing.assert_allclose(fields["contact_secondary_penetration_nodeavg"], fields["contact_slave_penetration_nodeavg"])
    np.testing.assert_allclose(fields["contact_pressure_nodeavg"], fields["contact_secondary_pressure_nodeavg"])
    np.testing.assert_allclose(fields["contact_penetration_nodeavg"], fields["contact_secondary_penetration_nodeavg"])
    np.testing.assert_allclose(fields["contact_active_node"], fields["contact_secondary_active_node"])
    assert metrics["active_contact_secondary_node_count"] == metrics["active_contact_slave_node_count"]
    assert metrics["max_contact_secondary_pressure_nodeavg"] == pytest.approx(metrics["max_contact_slave_pressure_nodeavg"])
    assert metrics["active_contact_node_count"] == metrics["active_contact_secondary_node_count"]
    assert metrics["max_contact_pressure_nodeavg"] == pytest.approx(metrics["max_contact_secondary_pressure_nodeavg"])


def test_contact_pressure_recovery_uses_secondary_region_nodes_when_available() -> None:
    arrays = {
        "sample_node_ids": np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        "sample_weights": np.asarray([[0.2, 0.6, 0.2], [0.2, 0.6, 0.2]], dtype=float),
        "gaps": np.asarray([-0.10, 0.05], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([2.0, 2.0], dtype=float),
        "master_node_ids": np.asarray([[4, 5, 6], [4, 5, 6]], dtype=np.int64),
        "master_weights": np.asarray([[0.3, 0.4, 0.3], [0.3, 0.4, 0.3]], dtype=float),
        "master_face_ids": np.asarray([20, 20], dtype=np.int64),
        "secondary_node_ids": np.asarray([1, 2], dtype=np.int64),
    }

    diagnostics = _contact_node_diagnostics_from_arrays(arrays, n_nodes=7, stiffness=100.0)

    secondary_pressure = diagnostics["fields"]["contact_secondary_pressure_nodeavg"]
    assert secondary_pressure[0] == pytest.approx(0.0)
    assert secondary_pressure[1] == pytest.approx(10.0)
    assert secondary_pressure[2] == pytest.approx(0.0)
    assert diagnostics["metrics"]["active_contact_secondary_node_count"] == 1
    np.testing.assert_allclose(diagnostics["fields"]["contact_pressure_nodeavg"], secondary_pressure)
    assert diagnostics["fields"]["contact_slave_pressure_nodeavg"][0] > 0.0
    assert diagnostics["fields"]["contact_master_pressure_nodeavg"][4] > 0.0
    assert diagnostics["metrics"]["active_contact_node_count"] == 1
    assert diagnostics["metrics"]["max_contact_pressure_nodeavg"] == pytest.approx(10.0)


def test_contact_region_integral_metrics_report_force_work_energy_area() -> None:
    arrays = {
        "gaps": np.asarray([-0.10, -0.20, 0.05], dtype=float),
        "areas": np.asarray([2.0, 3.0, 5.0], dtype=float),
    }

    metrics = _contact_region_integral_metrics_from_arrays(arrays, stiffness=100.0)

    assert metrics["contact_region_count"] == 3
    assert metrics["active_contact_region_count"] == 2
    assert metrics["contact_active_area"] == pytest.approx(5.0)
    assert metrics["contact_region_normal_force"] == pytest.approx(2.0 * 10.0 + 3.0 * 20.0)
    assert metrics["contact_region_energy"] == pytest.approx(0.5 * 100.0 * (2.0 * 0.1 * 0.1 + 3.0 * 0.2 * 0.2))
    assert metrics["contact_region_virtual_work"] == pytest.approx(2.0 * metrics["contact_region_energy"])


def test_contact_path_tracking_reports_master_face_switch_fraction() -> None:
    arrays = {
        "gaps": np.asarray([-0.10, -0.02, 0.01], dtype=float),
        "master_face_ids": np.asarray([3, 4, 5], dtype=np.int64),
        "master_barycentric": np.asarray([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]], dtype=float),
        "tracking_cache_hits": np.asarray([True, True, False], dtype=bool),
        "tracking_cache_matches": np.asarray([True, False, False], dtype=bool),
    }
    previous = np.asarray([3, 7, 8], dtype=np.int64)
    previous_bary = np.asarray([[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.2, 0.2, 0.6]], dtype=float)

    metrics, current, current_bary = _contact_path_tracking_metrics_from_arrays(arrays, previous, previous_bary)

    assert current is not None
    assert current_bary is not None
    np.testing.assert_array_equal(current, [3, 4, 5])
    np.testing.assert_allclose(current_bary[0], [0.7, 0.2, 0.1])
    assert metrics["contact_active_master_face_count"] == 2
    assert metrics["contact_master_face_tracking_comparable_count"] == 2
    assert metrics["contact_master_face_switch_count"] == 1
    assert metrics["contact_master_face_switch_fraction"] == pytest.approx(0.5)
    assert metrics["contact_master_barycentric_tracking_comparable_count"] == 1
    assert metrics["contact_master_barycentric_drift_mean"] == pytest.approx(np.sqrt(0.02))
    assert metrics["contact_master_barycentric_drift_max"] == pytest.approx(np.sqrt(0.02))
    assert metrics["contact_path_cache_hit_count"] == 2
    assert metrics["contact_path_cache_hit_fraction"] == pytest.approx(1.0)
    assert metrics["contact_path_cache_match_count"] == 1
    assert metrics["contact_path_cache_match_fraction"] == pytest.approx(0.5)


def test_contact_active_region_continuity_uses_secondary_constraint_regions() -> None:
    arrays = {
        "gaps": np.asarray([-0.10, -0.02, 0.01, -0.03], dtype=float),
        "secondary_node_ids": np.asarray([1, 2, 3, 4], dtype=np.int64),
    }
    previous = (1, 5)

    metrics, current = _contact_active_region_continuity_metrics_from_arrays(arrays, previous)

    assert current == (1, 2, 4)
    assert metrics["contact_active_region_continuity_previous_count"] == 2
    assert metrics["contact_active_region_continuity_current_count"] == 3
    assert metrics["contact_active_region_continuity_intersection_count"] == 1
    assert metrics["contact_active_region_new_count"] == 2
    assert metrics["contact_active_region_dropped_count"] == 1
    assert metrics["contact_active_region_persistence_fraction"] == pytest.approx(0.5)
    assert metrics["contact_active_region_jaccard"] == pytest.approx(0.25)


def test_penalty_history_reports_contact_energy_and_virtual_work() -> None:
    model = SimpleNamespace(
        X=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        elements=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        E=100.0,
        nu=0.25,
    )
    state = SimpleNamespace(x=model.X + 0.01, v=np.zeros_like(model.X), a=np.zeros_like(model.X), time=0.0)
    internal = SimpleNamespace(
        force=np.zeros((4, 3), dtype=float),
        strain=np.zeros((1, 3, 3), dtype=float),
        stress=np.zeros((1, 3, 3), dtype=float),
        von_mises=np.zeros(1, dtype=float),
        strain_energy=0.25,
    )
    contact = SimpleNamespace(
        active_count=1,
        min_gap=-0.01,
        normal_force=2.0,
        energy=0.5,
        force=np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, -0.5, 0.0]]),
    )

    row = _penalty_history_row(
        time_value=0.1,
        closure=0.0,
        rotation=0.0,
        state=state,
        model=model,
        internal=internal,
        contact_response=contact,
        young=100.0,
        poisson=0.25,
        newton_iterations=2,
        residual_norm=1.0e-6,
        solver="unit",
    )

    assert row["contact_energy"] == pytest.approx(0.5)
    assert row["contact_virtual_work"] == pytest.approx(1.0)
    assert row["contact_resultant_force_norm"] == pytest.approx(0.0)


def test_array_region_workspace_reuses_topology_without_freezing_payload() -> None:
    arrays = {
        "sample_node_ids": np.asarray([[0, 1, 2], [0, 1, 2], [1, 2, 3]], dtype=np.int64),
        "sample_weights": np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], dtype=float),
        "gaps": np.asarray([-0.20, -0.05, 0.03], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([2.0, 3.0, 1.0], dtype=float),
        "master_node_ids": np.asarray([[4, 5, 6], [4, 5, 6], [5, 6, 7]], dtype=np.int64),
        "master_weights": np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]], dtype=float),
    }
    changed_payload = {key: np.asarray(value).copy() for key, value in arrays.items()}
    changed_payload["gaps"] = np.asarray([-0.10, -0.02, -0.04], dtype=float)
    changed_payload["master_weights"] = np.asarray(
        [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.4, 0.5]],
        dtype=float,
    )
    workspace = _ContactAggregationWorkspace()

    first = _aggregate_contact_sample_arrays(arrays, "slave_node_region_participation", workspace=workspace)
    second = _aggregate_contact_sample_arrays(changed_payload, "slave_node_region_participation", workspace=workspace)
    reference = _aggregate_contact_sample_arrays(changed_payload, "slave_node_region_participation")

    assert first is not None
    assert second is not None
    assert reference is not None
    assert workspace.misses == 1
    assert workspace.hits == 1
    for key in (
        "sample_node_ids",
        "sample_weights",
        "gaps",
        "normals",
        "areas",
        "master_node_ids",
        "master_weights",
        "master_barycentric",
        "master_face_ids",
        "secondary_node_ids",
        "tracking_cache_hits",
        "tracking_cache_matches",
        "tracking_barycentric_distances",
    ):
        np.testing.assert_allclose(np.asarray(second[key]), np.asarray(reference[key]), equal_nan=True)


def test_array_signed_participation_matches_object_signed_participation() -> None:
    arrays = {
        "sample_node_ids": np.asarray([[0, 1, 2], [0, 1, 2], [1, 2, 3]], dtype=np.int64),
        "sample_weights": np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], dtype=float),
        "gaps": np.asarray([-0.20, 0.05, 0.03], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([2.0, 3.0, 1.0], dtype=float),
        "master_node_ids": np.asarray([[4, 5, 6], [4, 5, 6], [5, 6, 7]], dtype=np.int64),
        "master_weights": np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]], dtype=float),
    }
    object_samples = _aggregate_contact_samples(
        _contact_samples_from_arrays(arrays, stiffness=100.0),
        "slave_node_region_signed_participation",
    )

    array_samples = _aggregate_contact_sample_arrays(arrays, "slave_node_region_signed_participation")

    assert array_samples is not None
    object_response = _assemble_contact_response_force_only(object_samples, n_nodes=8)
    array_response = _assemble_contact_arrays_force_only(array_samples, n_nodes=8, stiffness=100.0)
    np.testing.assert_allclose(array_response.force, object_response.force)
    assert array_response.normal_force == pytest.approx(object_response.normal_force)
    assert array_response.energy == pytest.approx(object_response.energy)


def test_array_constraint_region_pressure_uses_average_gap() -> None:
    arrays = {
        "sample_node_ids": np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        "sample_weights": np.asarray([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]], dtype=float),
        "gaps": np.asarray([-0.10, 0.20], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([1.0, 1.0], dtype=float),
        "master_node_ids": np.asarray([[4, 5, 6], [4, 5, 6]], dtype=np.int64),
        "master_weights": np.asarray([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], dtype=float),
    }

    pointwise = _aggregate_contact_sample_arrays(arrays, "slave_node_region_participation")
    region = _aggregate_contact_sample_arrays(arrays, "slave_node_region_constraint")

    assert pointwise is not None
    assert region is not None
    assert np.any(np.asarray(pointwise["gaps"], dtype=float) < 0.0)
    assert np.all(np.asarray(region["gaps"], dtype=float) > 0.0)

    pointwise_response = _assemble_contact_arrays_force_only(pointwise, n_nodes=7, stiffness=100.0)
    region_response = _assemble_contact_arrays_force_only(region, n_nodes=7, stiffness=100.0)

    assert pointwise_response.normal_force > 0.0
    assert region_response.active_count == 0
    assert region_response.normal_force == pytest.approx(0.0)
    assert region_response.energy == pytest.approx(0.0)


def test_array_constraint_region_closed_force_from_region_gap() -> None:
    arrays = {
        "sample_node_ids": np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        "sample_weights": np.asarray([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]], dtype=float),
        "gaps": np.asarray([-0.30, 0.10], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([1.0, 1.0], dtype=float),
        "master_node_ids": np.asarray([[4, 5, 6], [4, 5, 6]], dtype=np.int64),
        "master_weights": np.asarray([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], dtype=float),
    }

    region = _aggregate_contact_sample_arrays(arrays, "slave_node_region_constraint")

    assert region is not None
    np.testing.assert_allclose(np.asarray(region["gaps"], dtype=float), [-0.10, -0.10, -0.10])
    response = _assemble_contact_arrays_force_only(region, n_nodes=7, stiffness=100.0)

    assert response.active_count == 3
    assert response.normal_force == pytest.approx(20.0)
    assert response.energy == pytest.approx(1.0)


def test_contact_active_signature_detects_status_changes() -> None:
    arrays = {
        "sample_node_ids": np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        "sample_weights": np.asarray([[0.3, 0.3, 0.4], [0.2, 0.5, 0.3]], dtype=float),
        "gaps": np.asarray([-0.01, 0.02], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([0.5, 0.5], dtype=float),
        "master_node_ids": np.asarray([[4, 5, 6], [5, 6, 7]], dtype=np.int64),
        "master_weights": np.asarray([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]], dtype=float),
    }
    samples = _contact_samples_from_arrays(arrays, stiffness=10.0)

    signature = _contact_active_signature_from_arrays(arrays)
    sample_signature = _contact_active_signature_from_samples(samples)

    assert signature == sample_signature
    assert len(signature) == 1
    arrays["gaps"][1] = -0.01
    assert _contact_active_signature_from_arrays(arrays) != signature


def test_source_contact_active_set_stability_requires_repeated_signature() -> None:
    signature = ((1, 2, 3, -1, 4, 5, 6),)

    assert _source_contact_active_set_is_stable(signature, None, require_stability=False)
    assert not _source_contact_active_set_is_stable(signature, None, require_stability=True)
    assert _source_contact_active_set_is_stable(signature, signature, require_stability=True)
    assert not _source_contact_active_set_is_stable(signature, tuple(), require_stability=True)


def test_source_increment_decision_requires_all_abaqus_style_gates() -> None:
    decision = _source_increment_convergence_decision(
        residual_converged=True,
        correction_converged=True,
        contact_force_increment_converged=True,
        active_set_stable=True,
        iteration_count=2,
        max_iterations=16,
        accept_unconverged=False,
    )

    assert decision.converged
    assert decision.accepted
    assert not decision.cutback_required
    assert decision.reason == "residual_correction_contact_force_active_set"

    limited = _source_increment_convergence_decision(
        residual_converged=True,
        correction_converged=True,
        contact_force_increment_converged=False,
        active_set_stable=True,
        iteration_count=16,
        max_iterations=16,
        accept_unconverged=False,
    )

    assert not limited.converged
    assert not limited.accepted
    assert limited.iteration_limited
    assert limited.cutback_required
    assert limited.reason == "contact_force_increment"


def test_source_increment_decision_marks_diagnostic_unconverged_acceptance() -> None:
    decision = _source_increment_convergence_decision(
        residual_converged=False,
        correction_converged=True,
        contact_force_increment_converged=True,
        active_set_stable=True,
        iteration_count=16,
        max_iterations=16,
        accept_unconverged=True,
    )

    assert not decision.converged
    assert decision.accepted
    assert decision.iteration_limited
    assert not decision.cutback_required
    assert decision.reason == "residual"


def test_source_increment_cutback_candidate_dt_respects_floor() -> None:
    assert _source_increment_cutback_candidate_dt(1.0e-3, min_dt=1.0e-6, cutback_factor=0.5) == pytest.approx(5.0e-4)
    assert _source_increment_cutback_candidate_dt(1.0e-6, min_dt=1.0e-6, cutback_factor=0.5) is None
    assert _source_increment_cutback_candidate_dt(1.5e-6, min_dt=1.0e-6, cutback_factor=0.5) == pytest.approx(1.0e-6)


def test_source_increment_gate_row_records_all_acceptance_gates() -> None:
    decision = _source_increment_convergence_decision(
        residual_converged=True,
        correction_converged=False,
        contact_force_increment_converged=True,
        active_set_stable=True,
        iteration_count=16,
        max_iterations=16,
        accept_unconverged=False,
    )

    row = _source_increment_gate_row(
        residual_converged=True,
        correction_converged=False,
        contact_force_increment_converged=True,
        active_set_stable=True,
        normalized_residual=1.0e-4,
        normalized_correction=2.0e-2,
        normalized_contact_force_increment=3.0e-4,
        contact_force_increment_norm=4.0,
        decision=decision,
        cutback_candidate_dt=5.0e-6,
    )

    assert row["source_residual_converged"] == 1
    assert row["source_correction_converged"] == 0
    assert row["source_contact_force_increment_converged"] == 1
    assert row["source_active_set_stable"] == 1
    assert row["source_increment_converged"] == 0
    assert row["source_increment_accepted"] == 0
    assert row["source_increment_cutback_required"] == 1
    assert row["source_step_convergence_reason"] == "correction"
    assert row["source_increment_cutback_candidate_dt"] == pytest.approx(5.0e-6)


def test_active_overlap_metrics_reports_precision_and_recall() -> None:
    sfc = np.asarray([True, True, False, True, False])
    abaqus = np.asarray([True, False, True, True, False])

    metrics = _active_overlap_metrics(sfc, abaqus)

    assert metrics["active_intersection_count"] == 2
    assert metrics["active_sfc_only_count"] == 1
    assert metrics["active_abaqus_only_count"] == 1
    assert metrics["active_precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["active_recall"] == pytest.approx(2.0 / 3.0)


def test_compare_sfc_history_to_abaqus_manifest_outputs_metric_errors(tmp_path: Path) -> None:
    history = tmp_path / "sfc_history.csv"
    manifest = tmp_path / "abaqus_manifest.csv"
    out = tmp_path / "errors.csv"
    _write_csv(
        history,
        [
            {
                "time": 0.0,
                "max_displacement_norm": 0.0,
                "p95_von_mises_nodeavg": 2.0,
                "p95_equivalent_elastic_strain_nodeavg": 0.1,
                "active_contact_node_count": 1,
            },
            {
                "time": 1.0,
                "max_displacement_norm": 3.0,
                "p95_von_mises_nodeavg": 5.0,
                "p95_equivalent_elastic_strain_nodeavg": 0.2,
                "active_contact_node_count": 2,
            },
        ],
    )
    _write_csv(
        manifest,
        [
            {
                "time": 0.0,
                "max_displacement_magnitude": 0.0,
                "p95_von_mises_nodeavg": 2.0,
                "p95_equivalent_elastic_strain_nodeavg": 0.1,
                "active_contact_node_count": 1,
            },
            {
                "time": 1.0,
                "max_displacement_magnitude": 4.0,
                "p95_von_mises_nodeavg": 10.0,
                "p95_equivalent_elastic_strain_nodeavg": 0.4,
                "active_contact_node_count": 3,
            },
        ],
    )

    rows = compare_sfc_history_to_abaqus_manifest(sfc_history=history, abaqus_manifest=manifest, out_csv=out)

    assert out.exists()
    assert rows[-1]["max_displacement_norm_rel_error"] == pytest.approx(0.25)
    assert rows[-1]["p95_von_mises_nodeavg_rel_error"] == pytest.approx(0.5)
    assert rows[-1]["active_contact_node_count_abs_error"] == pytest.approx(1.0)


def test_active_reduced_gap_jacobian_matches_full_projection() -> None:
    reference_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    hub = RigidHubMPC(np.asarray([0, 1], dtype=np.int64), reference_nodes, np.zeros(3, dtype=float))
    assembly = build_rigid_hub_reduced_assembly(reference_nodes, [hub], include_free_nodes=True)
    sample = ContactSample(
        node_ids=np.asarray([0, 2, 3], dtype=np.int64),
        shape_weights=np.asarray([0.2, 0.3, 0.5], dtype=float),
        gap=-0.01,
        normal=np.asarray([0.2, 0.3, 0.4], dtype=float),
        area=0.25,
        stiffness=1000.0,
        master_node_ids=np.asarray([1], dtype=np.int64),
        master_shape_weights=np.asarray([1.0], dtype=float),
    )
    free = np.arange(assembly.n_reduced_dofs, dtype=np.int64)

    _gaps, full_jacobian = hard_contact_gap_jacobian_from_samples([sample], n_total_dofs=3 * reference_nodes.shape[0])
    projected = (full_jacobian @ assembly.transformation)[:, free]
    direct = _active_reduced_gap_jacobian_sparse([sample], transformation=assembly.transformation, free=free)
    arrays = {
        "sample_node_ids": sample.node_ids.reshape(1, 3),
        "sample_weights": sample.shape_weights.reshape(1, 3),
        "gaps": np.asarray([sample.gap], dtype=float),
        "normals": sample.normal.reshape(1, 3),
        "areas": np.asarray([sample.area], dtype=float),
        "master_node_ids": np.asarray([[1, 1, 1]], dtype=np.int64),
        "master_weights": np.asarray([[1.0, 0.0, 0.0]], dtype=float),
    }
    direct_arrays = _active_reduced_gap_jacobian_sparse_from_arrays(arrays, transformation=assembly.transformation, free=free)

    np.testing.assert_allclose(direct.toarray(), np.asarray(projected), atol=1.0e-14)
    np.testing.assert_allclose(direct_arrays.toarray(), np.asarray(projected), atol=1.0e-14)


def test_constraint_region_tangent_matches_fixed_active_force_difference() -> None:
    arrays = {
        "sample_node_ids": np.asarray([[0, 1, 2], [1, 2, 4]], dtype=np.int64),
        "sample_weights": np.asarray([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]], dtype=float),
        "gaps": np.asarray([-0.10, -0.20], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([2.0, 3.0], dtype=float),
        "master_node_ids": np.asarray([[3, 5, 5], [3, 5, 5]], dtype=np.int64),
        "master_weights": np.asarray([[0.8, 0.2, 0.0], [0.4, 0.6, 0.0]], dtype=float),
        "master_face_ids": np.asarray([11, 12], dtype=np.int64),
        "secondary_node_ids": np.asarray([0, 1], dtype=np.int64),
    }
    n_nodes = 6
    n_dofs = 3 * n_nodes
    transformation = identity(n_dofs, format="csr")
    free = np.arange(n_dofs, dtype=np.int64)

    active_ids, j_free, tangent_scale = _active_constraint_region_tangent_data_from_arrays(
        arrays,
        transformation=transformation,
        free=free,
        pressure_stiffness=100.0,
        equilibrium_scale=1.0,
    )
    du = np.zeros(n_dofs, dtype=float)
    du[2::3] = np.asarray([1.0, -2.0, 1.5, -1.0, 0.5, -0.25], dtype=float) * 1.0e-4
    perturbed = {key: np.asarray(value).copy() for key, value in arrays.items()}
    perturbed["gaps"] = np.asarray(arrays["gaps"], dtype=float).copy()
    perturbed["gaps"][active_ids] += np.asarray(j_free @ du, dtype=float)

    base = _assemble_contact_arrays_force_only(arrays, n_nodes=n_nodes, stiffness=100.0)
    shifted = _assemble_contact_arrays_force_only(perturbed, n_nodes=n_nodes, stiffness=100.0)
    actual = shifted.force.reshape(-1) - base.force.reshape(-1)
    predicted = -np.asarray(j_free.T @ (tangent_scale * np.asarray(j_free @ du, dtype=float)), dtype=float).reshape(-1)

    np.testing.assert_allclose(actual, predicted, rtol=1.0e-11, atol=1.0e-11)


def test_constraint_region_gap_jacobian_matches_slave_master_fd() -> None:
    normal = np.asarray([0.0, 3.0, 4.0], dtype=float)
    arrays = {
        "sample_node_ids": np.asarray([[0, 1, 2], [1, 2, 4]], dtype=np.int64),
        "sample_weights": np.asarray([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]], dtype=float),
        "gaps": np.asarray([-0.10, 0.20], dtype=float),
        "normals": np.asarray([normal, normal], dtype=float),
        "areas": np.asarray([2.0, 3.0], dtype=float),
        "master_node_ids": np.asarray([[3, 5, 5], [3, 5, 5]], dtype=np.int64),
        "master_weights": np.asarray([[0.8, 0.2, 0.0], [0.4, 0.6, 0.0]], dtype=float),
    }
    n_nodes = 6
    row_ids, jacobian = _constraint_region_gap_jacobian_sparse_from_arrays(arrays, n_nodes=n_nodes)
    active_ids, active_jacobian = _constraint_region_gap_jacobian_sparse_from_arrays(
        arrays,
        n_nodes=n_nodes,
        active_only=True,
    )
    direction = np.asarray(
        [
            [0.01, -0.02, 0.03],
            [-0.04, 0.05, -0.06],
            [0.07, -0.08, 0.09],
            [-0.01, 0.02, -0.03],
            [0.04, -0.05, 0.06],
            [-0.07, 0.08, -0.09],
        ],
        dtype=float,
    )
    unit_normal = normal / np.linalg.norm(normal)
    base_x = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.5],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, -0.5],
        ],
        dtype=float,
    )

    def region_gap_from_positions(x: np.ndarray) -> np.ndarray:
        values = []
        for row in range(2):
            slave_point = sum(
                float(weight) * np.asarray(x[int(node)], dtype=float)
                for node, weight in zip(arrays["sample_node_ids"][row], arrays["sample_weights"][row], strict=True)
            )
            master_point = sum(
                float(weight) * np.asarray(x[int(node)], dtype=float)
                for node, weight in zip(arrays["master_node_ids"][row], arrays["master_weights"][row], strict=True)
            )
            values.append(float(unit_normal @ (slave_point - master_point)))
        return np.asarray(values, dtype=float)

    eps = 1.0e-6
    expected = (region_gap_from_positions(base_x + eps * direction) - region_gap_from_positions(base_x - eps * direction)) / (2.0 * eps)

    np.testing.assert_array_equal(row_ids, [0, 1])
    np.testing.assert_array_equal(active_ids, [0])
    np.testing.assert_allclose(np.asarray(jacobian @ direction.reshape(-1)).reshape(-1), expected, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(active_jacobian @ direction.reshape(-1)).reshape(-1), expected[:1], atol=1.0e-14)


def test_constraint_region_pressure_tangent_filters_open_and_zero_area_rows() -> None:
    arrays = {
        "gaps": np.asarray([-0.10, 0.20, -0.30], dtype=float),
        "areas": np.asarray([2.0, 5.0, 0.0], dtype=float),
        "sample_node_ids": np.asarray([[0], [1], [2]], dtype=np.int64),
        "sample_weights": np.ones((3, 1), dtype=float),
        "master_node_ids": np.asarray([[3], [3], [3]], dtype=np.int64),
        "master_weights": np.ones((3, 1), dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=float),
    }

    active_ids, scales = _constraint_region_pressure_tangent_scales_from_arrays(
        arrays,
        pressure_stiffness=100.0,
        equilibrium_scale=2.0,
    )
    tangent_ids, _j_free, tangent_scales = _active_constraint_region_tangent_data_from_arrays(
        arrays,
        transformation=identity(12, format="csr"),
        free=np.arange(12, dtype=np.int64),
        pressure_stiffness=100.0,
        equilibrium_scale=2.0,
    )

    np.testing.assert_array_equal(active_ids, [0])
    np.testing.assert_allclose(scales, [400.0])
    np.testing.assert_array_equal(tangent_ids, [0])
    np.testing.assert_allclose(tangent_scales, [400.0])


def test_constraint_region_tangent_metrics_use_region_rows() -> None:
    arrays = {
        "gaps": np.asarray([-0.10, -0.20, 0.02], dtype=float),
        "areas": np.asarray([2.0, 3.0, 5.0], dtype=float),
        "secondary_node_ids": np.asarray([4, 4, 5], dtype=np.int64),
    }

    metrics = _constraint_region_tangent_metrics_from_arrays(
        arrays,
        pressure_stiffness=100.0,
        equilibrium_scale=2.0,
    )

    assert metrics["contact_tangent_active_region_count"] == 2
    assert metrics["contact_tangent_active_secondary_node_count"] == 1
    assert metrics["contact_tangent_scale_sum"] == pytest.approx(2.0 * 100.0 * (2.0 + 3.0))
    assert metrics["contact_tangent_scale_max"] == pytest.approx(2.0 * 100.0 * 3.0)


def test_cropped_gear_pair_is_small_and_has_supports() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=4, expansion_rings=0)

    assert pair.gear1.nodes.shape[0] < model.gear1.nodes.shape[0]
    assert pair.gear2.nodes.shape[0] < model.gear2.nodes.shape[0]
    assert pair.gear1.contact_faces.shape[0] == 4
    assert pair.gear2.contact_faces.shape[0] == 4
    assert pair.gear1.support_nodes.size >= 4
    assert pair.gear2.support_nodes.size >= 4
    assert pair.initial_patch_gap >= 0.0


def test_full_active_gear_pair_keeps_complete_volume_mesh() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_full_active_pair(model, active_faces_per_body=4)
    expanded = build_full_active_pair(model, active_faces_per_body=4, active_patch_radius_factor=2.0)

    assert pair.gear1.nodes.shape == model.gear1.nodes.shape
    assert pair.gear2.nodes.shape == model.gear2.nodes.shape
    assert pair.gear1.elements.shape == model.gear1.elements.shape
    assert pair.gear2.elements.shape == model.gear2.elements.shape
    assert pair.gear1.contact_faces.shape[0] == 4
    assert pair.gear2.contact_faces.shape[0] == 4
    assert pair.gear1.support_nodes.size == len(set(model.gear1_hub_labels))
    assert pair.gear2.support_nodes.size == len(set(model.gear2_hub_labels))
    assert pair.initial_patch_gap >= 0.0
    assert expanded.gear1.contact_faces.shape[0] >= pair.gear1.contact_faces.shape[0]
    assert expanded.gear2.contact_faces.shape[0] >= pair.gear2.contact_faces.shape[0]


def test_full_active_gear_pair_can_use_all_source_contact_faces() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_full_active_pair(model, active_faces_per_body=0)

    assert pair.gear1.nodes.shape == model.gear1.nodes.shape
    assert pair.gear2.nodes.shape == model.gear2.nodes.shape
    assert pair.gear1.contact_faces.shape[0] == model.gear1_contact_faces.shape[0]
    assert pair.gear2.contact_faces.shape[0] == model.gear2_contact_faces.shape[0]
    assert pair.gear1.surface_entries == model.gear1_surface_entries
    assert pair.gear2.surface_entries == model.gear2_surface_entries


def test_cropped_gear_sfc_lagrangian_sdf_path_runs_one_implicit_step() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
    )

    assert len(history) == 1
    assert summary["status"] == "completed"
    assert summary["nodes"] > 0
    assert summary["elements"] > 0
    assert history[-1]["newton_iterations"] >= 1


def test_cropped_gear_modified_newton_penalty_path_runs_one_implicit_step() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        penalty_solver="modified_newton",
        material_linearization="reference_linear",
    )

    assert len(history) == 1
    assert summary["status"] == "completed"
    assert summary["contact_mode"] == "penalty"
    assert summary["penalty_solver"] == "modified_newton"
    assert summary["material_linearization"] == "reference_linear"
    assert int(summary["penalty_fallback_count"]) >= 0
    assert float(summary["timing_base_tangent_seconds"]) >= 0.0
    assert history[-1]["penalty_solver"] in {"modified_newton", "full_newton_fallback"}
    assert history[-1]["newton_iterations"] >= 1


def test_cropped_gear_modified_newton_writes_sfc_vtk_frames(tmp_path: Path) -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)
    vtk_dir = tmp_path / "sfc_vtk"

    _history, summary = solve_sfc_cropped_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        penalty_solver="modified_newton",
        material_linearization="reference_linear",
        vtk_out_dir=vtk_dir,
        vtk_frame_stride=1,
    )

    pvd = Path(str(summary["sfc_vtk_pvd"]))
    manifest = Path(str(summary["sfc_vtk_manifest"]))
    first = vtk_dir / "sfc_0000.vtk"
    second = vtk_dir / "sfc_0001.vtk"
    assert pvd.exists()
    assert manifest.exists()
    assert first.exists()
    assert second.exists()
    assert int(summary["sfc_vtk_frame_count"]) == 2
    assert pvd.read_text(encoding="utf-8").count("<DataSet") == 2
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "max_displacement_magnitude" in manifest_text
    assert "max_equivalent_elastic_strain_nodeavg" in manifest_text
    assert "p95_von_mises_nodeavg" in manifest_text
    assert "p95_equivalent_elastic_strain_nodeavg" in manifest_text
    assert "max_displacement_magnitude_object1" in manifest_text
    text = second.read_text(encoding="ascii")
    assert "DATASET UNSTRUCTURED_GRID" in text
    assert "VECTORS U float" in text
    assert "SCALARS von_mises_nodeavg float 1" in text
    assert "SCALARS strain_norm_nodeavg float 1" in text
    assert "SCALARS equivalent_elastic_strain_nodeavg float 1" in text
    assert "SCALARS contact_pressure_nodeavg float 1" in text
    assert "SCALARS contact_active_node float 1" in text
    assert "SCALARS von_mises float 1" in text
    assert "TENSORS S float" in text


def test_cropped_gear_source_drive_path_advances_rp_rotation(tmp_path: Path) -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)
    vtk_dir = tmp_path / "source_sfc_vtk"

    history, summary = solve_sfc_source_drive_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-5,
        dt=1.0e-5,
        gear1_angular_velocity_z=model.gear1_angular_velocity_z,
        gear2_torque_z=model.gear2_torque_z,
        vtk_out_dir=vtk_dir,
        vtk_frame_stride=1,
    )

    assert len(history) == 1
    assert summary["status"] == "completed"
    assert summary["contact_mode"] == "source_penalty"
    assert float(history[-1]["rp1_rotation_z"]) == pytest.approx(model.gear1_angular_velocity_z * 1.0e-5)
    assert float(history[-1]["gear2_torque_z"]) == pytest.approx(model.gear2_torque_z)
    assert int(summary["reduced_dofs"]) > 0
    assert float(summary["timing_source_base_lu_seconds"]) >= 0.0
    assert int(summary["source_sparse_cg_base_lu_preconditioner"]) >= 0
    assert int(summary["source_unconverged_accepted_count"]) >= 0
    assert int(summary["source_cutback_required_count"]) >= 0
    assert summary["source_rotation_unit"] == "radian"
    assert "source_increment_accepted" in history[-1]
    assert "source_increment_cutback_required" in history[-1]
    assert "source_increment_cutback_candidate_dt" in history[-1]
    assert "source_residual_converged" in history[-1]
    assert "source_correction_converged" in history[-1]
    assert "source_contact_force_increment_converged" in history[-1]
    assert "source_active_set_stable" in history[-1]
    assert "source_increment_converged" in history[-1]
    assert "contact_active_region_continuity_current_count" in history[-1]
    assert "contact_active_region_jaccard" in history[-1]
    manifest = Path(str(summary["sfc_vtk_manifest"]))
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "linear_corotated" in text
    assert "radian" in text
    assert "active_contact_node_count" in text
    assert "active_contact_slave_node_count" in text
    assert "active_contact_master_node_count" in text
    assert "max_contact_pressure_nodeavg" in text
    assert "source_residual_converged" in text
    assert "source_contact_force_increment_converged" in text
    vtk_text = (vtk_dir / "sfc_0001.vtk").read_text(encoding="ascii")
    assert "SCALARS contact_pressure_nodeavg float 1" in vtk_text
    assert "SCALARS contact_secondary_pressure_nodeavg float 1" in vtk_text
    assert "SCALARS contact_secondary_penetration_nodeavg float 1" in vtk_text
    assert "SCALARS contact_secondary_active_node float 1" in vtk_text
    assert "SCALARS contact_slave_pressure_nodeavg float 1" in vtk_text
    assert "SCALARS contact_master_pressure_nodeavg float 1" in vtk_text
    assert "SCALARS contact_penetration_nodeavg float 1" in vtk_text
    assert "SCALARS contact_gap_min_node float 1" in vtk_text


def test_source_drive_checkpoint_resume_matches_continuous_short_run(tmp_path: Path) -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)
    checkpoint = tmp_path / "source_drive_checkpoint.npz"

    continuous_history, _continuous_summary = solve_sfc_source_drive_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=2.0e-5,
        dt=1.0e-5,
        gear1_angular_velocity_z=model.gear1_angular_velocity_z,
        gear2_torque_z=model.gear2_torque_z,
        history_frame_stride=1,
    )
    first_history, first_summary = solve_sfc_source_drive_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-5,
        dt=1.0e-5,
        gear1_angular_velocity_z=model.gear1_angular_velocity_z,
        gear2_torque_z=model.gear2_torque_z,
        history_frame_stride=1,
        source_checkpoint_path=checkpoint,
        source_checkpoint_stride=1,
    )
    resumed_history, resumed_summary = solve_sfc_source_drive_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=2.0e-5,
        dt=1.0e-5,
        gear1_angular_velocity_z=model.gear1_angular_velocity_z,
        gear2_torque_z=model.gear2_torque_z,
        history_frame_stride=1,
        source_checkpoint_path=checkpoint,
        resume_source_checkpoint=True,
        source_checkpoint_stride=1,
    )

    assert checkpoint.exists()
    assert len(first_history) == 1
    assert len(resumed_history) == 2
    assert int(first_summary["source_resume_from_step"]) == 0
    assert int(resumed_summary["source_resume_from_step"]) == 1
    for key in ("rp1_rotation_z", "rp2_rotation_z", "max_displacement_norm", "min_gap", "normal_force"):
        assert float(resumed_history[-1][key]) == pytest.approx(float(continuous_history[-1][key]), rel=1.0e-10, abs=1.0e-12)


def test_source_drive_corotated_visual_postprocess_removes_rigid_rotation_stress() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=1.0)
    reference_internal = stvk_internal_response(model, model.X, assemble_tangent=True)
    theta = 0.7
    small_rigid = np.cross(np.asarray([[0.0, 0.0, theta]], dtype=float), X)
    state = MechanicsState(
        X + small_rigid,
        np.zeros_like(X),
        np.zeros_like(X),
        time=0.0,
    )

    visual_state, internal = _source_drive_corotated_visual_state_and_internal(
        model=model,
        state=state,
        reference_tangent=reference_internal.tangent,
        body_node_slices=(slice(0, 4), slice(4, 4)),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta, 0.0),
    )

    expected_node1 = np.asarray([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    np.testing.assert_allclose(visual_state.x[1], expected_node1, atol=1.0e-14)
    assert float(np.max(internal.von_mises)) <= 1.0e-10


def test_source_drive_corotated_contact_positions_use_finite_rp_rotation() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=1.0)
    theta = 1.9
    x_small = X + np.cross(np.asarray([[0.0, 0.0, theta]], dtype=float), X)

    x_contact, u_elastic = _source_drive_corotated_positions_and_elastic_displacement(
        model=model,
        x_raw=x_small,
        body_node_slices=(slice(0, 4), slice(4, 4)),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta, 0.0),
    )

    np.testing.assert_allclose(u_elastic, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(x_contact[1], [np.cos(theta), np.sin(theta), 0.0], atol=1.0e-14)


def test_contact_normal_compatibility_filter_keeps_opposing_surfaces() -> None:
    x = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    sample = ContactSample(
        node_ids=np.asarray([0, 1, 2], dtype=np.int64),
        shape_weights=np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        gap=-1.0e-3,
        normal=np.asarray([0.0, 0.0, -1.0]),
        area=1.0,
        stiffness=1.0,
    )
    same_side = ContactSample(
        node_ids=sample.node_ids,
        shape_weights=sample.shape_weights,
        gap=-1.0e-3,
        normal=np.asarray([0.0, 0.0, 1.0]),
        area=1.0,
        stiffness=1.0,
    )

    filtered = _filter_contact_samples_by_normal_compatibility([sample, same_side], x, mode="opposing")

    assert len(filtered) == 1
    assert filtered[0] is sample


def test_secondary_average_contact_direction_aligns_slave_normal_to_master_sign() -> None:
    x = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    sample = ContactSample(
        node_ids=np.asarray([0, 1, 2], dtype=np.int64),
        shape_weights=np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        gap=-1.0e-3,
        normal=np.asarray([0.0, 0.0, -1.0]),
        area=1.0,
        stiffness=1.0,
    )

    replaced = _replace_contact_normals_with_secondary_average([sample], x)

    assert len(replaced) == 1
    np.testing.assert_allclose(replaced[0].normal, [0.0, 0.0, -1.0])
    assert replaced[0].gap == pytest.approx(sample.gap)


def test_secondary_average_contact_direction_projects_master_gap() -> None:
    x = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, np.sqrt(0.5), np.sqrt(0.5)],
        ],
        dtype=float,
    )
    sample = ContactSample(
        node_ids=np.asarray([0, 1, 2], dtype=np.int64),
        shape_weights=np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        gap=-1.0e-3,
        normal=np.asarray([0.0, 0.0, 1.0]),
        area=1.0,
        stiffness=1.0,
    )

    replaced = _replace_contact_normals_with_secondary_average([sample], x, project_gap_to_secondary_plane=True)

    assert replaced[0].normal @ sample.normal > 0.0
    assert replaced[0].gap == pytest.approx(sample.gap / np.sqrt(0.5))


def test_secondary_normal_projection_samples_use_line_intersection_gap() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.1, 0.1, -0.2],
            [0.9, 0.1, -0.2],
            [0.1, 0.9, -0.2],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 2, 1]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        compiled_batch_projection=False,
    )

    sample = list(contact.secondary_normal_projection_samples(np.vstack([master_nodes, slave_nodes])))[0]

    assert sample.gap == pytest.approx(-0.2)
    np.testing.assert_allclose(sample.normal, [0.0, 0.0, 1.0])
    assert np.sum(sample.master_shape_weights) == pytest.approx(1.0)


def test_secondary_normal_projection_preserves_closed_line_intersection_past_limit() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.1, 0.1, -0.2],
            [0.9, 0.1, -0.2],
            [0.1, 0.9, -0.2],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 2, 1]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        compiled_batch_projection=False,
        secondary_line_distance_limit=0.1,
    )

    sample = list(contact.secondary_normal_projection_samples(np.vstack([master_nodes, slave_nodes])))[0]

    assert sample.gap == pytest.approx(-0.2)


def test_secondary_normal_projection_preserves_closed_line_intersection_past_hard_limit() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.1, 0.1, -0.2],
            [0.9, 0.1, -0.2],
            [0.1, 0.9, -0.2],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 2, 1]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        compiled_batch_projection=False,
        secondary_line_distance_limit=0.05,
        secondary_line_hard_distance_limit=0.1,
    )

    sample = list(contact.secondary_normal_projection_samples(np.vstack([master_nodes, slave_nodes])))[0]

    assert sample.gap == pytest.approx(-0.2)


def test_secondary_line_distance_limit_is_mesh_derived() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    limit = _default_secondary_line_distance_limit(pair, target_overclosure=1.0e-5)
    hard_limit = _default_secondary_line_hard_distance_limit(pair, target_overclosure=1.0e-5)

    assert limit <= _default_contact_search_radius(pair, target_overclosure=1.0e-5)
    assert limit >= 1.0e-4
    assert limit >= 2.5e-5
    assert hard_limit == pytest.approx(2.0 * limit)


def test_secondary_normal_projection_does_not_overclose_open_closest_feature() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.05],
            [0.0, 1.0, -0.05],
            [1.0, 0.0, -0.05],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.15, 0.15, -0.1],
            [0.15, 0.25, -0.1],
            [0.25, 0.15, -0.1],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 2, 1]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        compiled_batch_projection=False,
        secondary_line_distance_limit=0.01,
        secondary_line_hard_distance_limit=0.01,
    )

    sample = list(contact.secondary_normal_projection_samples(np.vstack([master_nodes, slave_nodes])))[0]

    assert sample.gap > 0.0


def test_secondary_tracking_candidates_follow_master_face_adjacency() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.2, 0.2, -0.1],
            [0.3, 0.2, -0.1],
            [0.2, 0.3, -0.1],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 2, 1]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 2, 1], [1, 2, 3]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=0.0,
        compiled_batch_projection=True,
        secondary_tracking_rings=1,
    )
    merged = contact._merge_secondary_tracking_candidates([[0]], np.asarray([0], dtype=np.int64))
    assert merged == [[0, 1]]

    contact._ensure_secondary_face_cache()[0] = 1
    released = contact._merge_secondary_tracking_candidates([[]], np.asarray([0], dtype=np.int64))

    assert released == [[]]
    assert list(contact.secondary_normal_projection_samples(np.vstack([master_nodes, slave_nodes])))


def test_secondary_normal_projection_sample_arrays_match_scalar_when_cpp_available() -> None:
    if not _cpp_projection.secondary_normal_indexed_faces_available():
        pytest.skip("C++ secondary-normal indexed projection backend is unavailable")
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.1, 0.1, -0.2],
            [0.9, 0.1, -0.2],
            [0.1, 0.9, -0.2],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 2, 1]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        compiled_batch_projection=True,
    )
    x = np.vstack([master_nodes, slave_nodes])

    scalar = list(contact.secondary_normal_projection_samples(x))[0]
    arrays = contact.secondary_normal_projection_sample_arrays(x)

    assert arrays is not None
    assert arrays["gaps"].shape == (1,)
    assert arrays["gaps"][0] == pytest.approx(scalar.gap)
    np.testing.assert_allclose(arrays["normals"][0], scalar.normal)
    np.testing.assert_allclose(arrays["master_weights"][0], scalar.master_shape_weights)
    np.testing.assert_allclose(arrays["master_barycentric"][0], scalar.master_shape_weights)
    assert "tracking_barycentric_distances" in arrays


def test_secondary_normal_projection_sample_arrays_reject_near_tangent_constraint() -> None:
    if not _cpp_projection.secondary_normal_indexed_faces_available():
        pytest.skip("C++ secondary-normal indexed projection backend is unavailable")
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, -0.6],
            [0.15, 0.0, -0.6],
            [0.0, 0.3, -0.6],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 2, 1]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        compiled_batch_projection=True,
    )
    x = np.vstack([master_nodes, slave_nodes])

    accepted = contact.secondary_normal_projection_sample_arrays(x, dot_threshold=0.0)
    rejected = contact.secondary_normal_projection_sample_arrays(x, dot_threshold=-0.25)

    assert accepted is not None
    assert accepted["gaps"].size == 1
    assert accepted["gaps"][0] < 0.0
    assert rejected is not None
    assert rejected["gaps"].size == 0


def test_secondary_path_tracking_keeps_previous_anchor_face() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.009],
            [1.0, 0.0, -0.009],
            [0.0, 1.0, -0.009],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.10, 0.10, -0.010],
            [0.30, 0.10, -0.010],
            [0.10, 0.30, -0.010],
        ],
        dtype=float,
    )
    master_faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    slave_faces = np.asarray([[0, 2, 1]], dtype=np.int64)

    instantaneous = LagrangianSDFSurfaceContactGeometry(
        slave_faces,
        MaterialSDF.from_triangle_surface(master_nodes, master_faces),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
    )
    tracked = LagrangianSDFSurfaceContactGeometry(
        slave_faces,
        MaterialSDF.from_triangle_surface(master_nodes, master_faces),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        secondary_path_tracking=True,
        secondary_tracking_rings=1,
    )
    tracked._update_secondary_tracking_cache(
        cache_index=0,
        face_id=0,
        barycentric=np.asarray([0.70, 0.15, 0.15], dtype=float),
    )
    x = np.vstack([master_nodes, slave_nodes])

    instantaneous_sample = list(instantaneous.secondary_normal_projection_samples(x))[0]
    tracked_sample = list(tracked.secondary_normal_projection_samples(x))[0]

    assert instantaneous_sample.master_node_ids.tolist() == [3, 4, 5]
    assert tracked_sample.master_node_ids.tolist() == [0, 1, 2]
    assert instantaneous_sample.gap == pytest.approx(-0.001)
    assert tracked_sample.gap == pytest.approx(-0.010)


def test_secondary_path_tracking_batch_keeps_previous_anchor_face() -> None:
    if not _cpp_projection.secondary_normal_indexed_faces_available():
        pytest.skip("C++ secondary-normal indexed projection backend is unavailable")
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.009],
            [1.0, 0.0, -0.009],
            [0.0, 1.0, -0.009],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.10, 0.10, -0.010],
            [0.30, 0.10, -0.010],
            [0.10, 0.30, -0.010],
        ],
        dtype=float,
    )
    master_faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 2, 1]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, master_faces),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        compiled_batch_projection=True,
        secondary_path_tracking=True,
        secondary_tracking_rings=1,
    )
    contact._update_secondary_tracking_cache(
        cache_index=0,
        face_id=0,
        barycentric=np.asarray([0.70, 0.15, 0.15], dtype=float),
    )

    arrays = contact.secondary_normal_projection_sample_arrays(np.vstack([master_nodes, slave_nodes]))

    assert arrays is not None
    assert arrays["master_node_ids"][0].tolist() == [0, 1, 2]
    assert arrays["gaps"][0] == pytest.approx(-0.010)


def test_secondary_normal_projection_sample_arrays_preserve_closed_line_intersection_past_limit() -> None:
    if not _cpp_projection.secondary_normal_indexed_faces_available():
        pytest.skip("C++ secondary-normal indexed projection backend is unavailable")
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.1, 0.1, -0.2],
            [0.9, 0.1, -0.2],
            [0.1, 0.9, -0.2],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        compiled_batch_projection=True,
        secondary_line_distance_limit=0.1,
    )

    arrays = contact.secondary_normal_projection_sample_arrays(np.vstack([master_nodes, slave_nodes]))

    assert arrays is not None
    assert arrays["gaps"].shape == (1,)
    assert arrays["gaps"][0] == pytest.approx(-0.2)


def test_secondary_normal_projection_sample_arrays_preserve_closed_line_past_hard_limit() -> None:
    if not _cpp_projection.secondary_normal_indexed_faces_available():
        pytest.skip("C++ secondary-normal indexed projection backend is unavailable")
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.1, 0.1, -0.2],
            [0.9, 0.1, -0.2],
            [0.1, 0.9, -0.2],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        compiled_batch_projection=True,
        secondary_line_distance_limit=0.05,
        secondary_line_hard_distance_limit=0.1,
    )

    arrays = contact.secondary_normal_projection_sample_arrays(np.vstack([master_nodes, slave_nodes]))

    assert arrays is not None
    assert arrays["gaps"].shape == (1,)
    assert arrays["gaps"][0] == pytest.approx(-0.2)


def test_secondary_normal_projection_sample_arrays_do_not_overclose_open_closest_feature() -> None:
    if not _cpp_projection.secondary_normal_indexed_faces_available():
        pytest.skip("C++ secondary-normal indexed projection backend is unavailable")
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.05],
            [0.0, 1.0, -0.05],
            [1.0, 0.0, -0.05],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.15, 0.15, -0.1],
            [0.15, 0.25, -0.1],
            [0.25, 0.15, -0.1],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 2, 1]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        compiled_batch_projection=True,
        secondary_line_distance_limit=0.01,
        secondary_line_hard_distance_limit=0.01,
    )

    arrays = contact.secondary_normal_projection_sample_arrays(np.vstack([master_nodes, slave_nodes]))

    assert arrays is not None
    assert arrays["gaps"].shape == (1,)
    assert arrays["gaps"][0] > 0.0


def test_contact_samples_from_arrays_preserves_payload() -> None:
    arrays = {
        "sample_node_ids": np.asarray([[1, 2, 3]], dtype=np.int64),
        "sample_weights": np.asarray([[0.2, 0.3, 0.5]], dtype=float),
        "gaps": np.asarray([-0.01], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([0.25], dtype=float),
        "master_node_ids": np.asarray([[4, 5, 6]], dtype=np.int64),
        "master_weights": np.asarray([[0.1, 0.2, 0.7]], dtype=float),
    }

    samples = _contact_samples_from_arrays(arrays, stiffness=12.0)

    assert len(samples) == 1
    assert samples[0].gap == pytest.approx(-0.01)
    assert samples[0].area == pytest.approx(0.25)
    assert samples[0].stiffness == pytest.approx(12.0)
    np.testing.assert_allclose(samples[0].shape_weights, [0.2, 0.3, 0.5])
    np.testing.assert_allclose(samples[0].master_shape_weights, [0.1, 0.2, 0.7])


def test_slave_face_averaging_groups_by_nodes_after_filtering() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.5, 0.5, 0.0]),
            gap=-1.0e-3,
            normal=np.asarray([0.0, 0.0, 1.0]),
            area=0.2,
            stiffness=10.0,
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.0, 0.5, 0.5]),
            gap=-2.0e-3,
            normal=np.asarray([0.0, 0.0, 1.0]),
            area=0.3,
            stiffness=10.0,
        ),
    ]

    aggregated = _aggregate_contact_samples(samples, "slave_face")

    assert len(aggregated) == 1
    assert aggregated[0].area == pytest.approx(0.5)
    assert aggregated[0].gap == pytest.approx((-1.0e-3 * 0.2 - 2.0e-3 * 0.3) / 0.5)
    np.testing.assert_allclose(aggregated[0].shape_weights, [0.125, 0.5, 0.375])


def test_slave_node_averaging_uses_tributary_area() -> None:
    sample = ContactSample(
        node_ids=np.asarray([0, 1, 2], dtype=np.int64),
        shape_weights=np.asarray([0.5, 0.3, 0.2]),
        gap=-1.0e-3,
        normal=np.asarray([0.0, 0.0, 1.0]),
        area=2.0,
        stiffness=10.0,
    )

    aggregated = _aggregate_contact_samples([sample], "slave_node")

    assert len(aggregated) == 3
    assert sum(float(item.area) for item in aggregated) == pytest.approx(2.0)
    assert {tuple(np.asarray(item.shape_weights, dtype=float)) for item in aggregated} == {
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    }


def test_slave_node_region_averaging_keeps_surface_shape_support() -> None:
    sample = ContactSample(
        node_ids=np.asarray([0, 1, 2], dtype=np.int64),
        shape_weights=np.asarray([0.5, 0.3, 0.2]),
        gap=-1.0e-3,
        normal=np.asarray([0.0, 0.0, 1.0]),
        area=2.0,
        stiffness=10.0,
        master_node_ids=np.asarray([3, 4, 5], dtype=np.int64),
        master_shape_weights=np.asarray([0.2, 0.5, 0.3]),
    )

    aggregated = _aggregate_contact_samples([sample], "slave_node_region")

    assert len(aggregated) == 3
    assert sum(float(item.area) for item in aggregated) == pytest.approx(2.0)
    for item in aggregated:
        np.testing.assert_allclose(item.shape_weights, sample.shape_weights)
        np.testing.assert_allclose(item.master_shape_weights, sample.master_shape_weights)


def test_contact_averaging_preserves_positive_overclosure_integral() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([1.0, 0.0, 0.0], dtype=float),
            gap=-0.4,
            normal=np.asarray([1.0, 0.0, 0.0], dtype=float),
            area=1.0,
            stiffness=10.0,
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.0, 1.0, 0.0], dtype=float),
            gap=0.2,
            normal=np.asarray([0.0, 1.0, 0.0], dtype=float),
            area=3.0,
            stiffness=10.0,
        ),
    ]

    merged = _aggregate_contact_samples(samples, "slave_face")[0]

    assert merged.area == pytest.approx(4.0)
    assert merged.gap == pytest.approx(-0.1)
    np.testing.assert_allclose(merged.normal, [1.0, 0.0, 0.0], atol=1.0e-14)


def test_constraint_region_averaging_uses_signed_gap_before_activation() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.5, 0.3, 0.2], dtype=float),
            gap=-1.0e-3,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=1.0,
            stiffness=10.0,
            master_node_ids=np.asarray([3, 4, 5], dtype=np.int64),
            master_shape_weights=np.asarray([0.2, 0.3, 0.5], dtype=float),
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.5, 0.3, 0.2], dtype=float),
            gap=3.0e-3,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=1.0,
            stiffness=10.0,
            master_node_ids=np.asarray([3, 4, 5], dtype=np.int64),
            master_shape_weights=np.asarray([0.2, 0.3, 0.5], dtype=float),
        ),
    ]

    positive_integral = _aggregate_contact_samples(samples, "slave_node_region")
    signed_constraint = _aggregate_contact_samples(samples, "slave_node_region_constraint")

    assert len(positive_integral) == len(signed_constraint) == 3
    assert any(sample.gap < 0.0 for sample in positive_integral)
    assert all(sample.gap > 0.0 for sample in signed_constraint)
    for sample in signed_constraint:
        np.testing.assert_allclose(sample.shape_weights, [0.5, 0.3, 0.2])


def test_area_average_constraint_keeps_integral_gap_but_not_pressure_weighted_support() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([1.0, 0.0, 0.0], dtype=float),
            gap=-2.0e-3,
            normal=np.asarray([1.0, 0.0, 0.0], dtype=float),
            area=1.0,
            stiffness=10.0,
            master_node_ids=np.asarray([3, 4, 5], dtype=np.int64),
            master_shape_weights=np.asarray([1.0, 0.0, 0.0], dtype=float),
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.0, 1.0, 0.0], dtype=float),
            gap=-1.0e-3,
            normal=np.asarray([0.0, 1.0, 0.0], dtype=float),
            area=1.0,
            stiffness=10.0,
            master_node_ids=np.asarray([3, 4, 5], dtype=np.int64),
            master_shape_weights=np.asarray([0.0, 1.0, 0.0], dtype=float),
        ),
    ]

    pressure_weighted = _aggregate_contact_samples(samples, "slave_face")[0]
    area_average = _aggregate_contact_samples(samples, "slave_face_area_average")[0]

    assert pressure_weighted.gap == pytest.approx(area_average.gap)
    np.testing.assert_allclose(pressure_weighted.shape_weights, [2.0 / 3.0, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(area_average.shape_weights, [0.5, 0.5, 0.0])


def test_lagrangian_surface_contact_nodal_samples_use_slave_corner_gaps() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, -0.30],
            [1.0, 0.0, -0.10],
            [0.0, 1.0, 0.20],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=3,
        master_node_offset=0,
        compiled_batch_projection=False,
    )
    samples = list(contact.nodal_samples(np.vstack([master_nodes, slave_nodes])))

    assert len(samples) == 3
    np.testing.assert_allclose([sample.gap for sample in samples], [-0.30, -0.10, 0.20], atol=1.0e-12)
    assert [int(np.asarray(sample.node_ids)[int(np.argmax(sample.shape_weights))]) for sample in samples] == [3, 4, 5]
    assert sum(float(sample.area) for sample in samples) == pytest.approx(
        0.5 * float(np.linalg.norm(np.cross(slave_nodes[1] - slave_nodes[0], slave_nodes[2] - slave_nodes[0])))
    )


def test_normal_compatible_samples_choose_nearest_opposing_candidate() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.1],
            [0.0, 1.0, -0.1],
            [1.0, 0.0, -0.1],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, 0.01],
            [1.0, 0.0, 0.01],
            [0.0, 1.0, 0.01],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, faces),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=0.2,
        compiled_batch_projection=False,
    )

    regular = list(contact.samples(np.vstack([master_nodes, slave_nodes])))[0]
    compatible = list(contact.normal_compatible_samples(np.vstack([master_nodes, slave_nodes])))[0]

    assert regular.master_node_ids.tolist() == [0, 1, 2]
    assert compatible.master_node_ids.tolist() == [3, 4, 5]
    assert regular.gap == pytest.approx(0.01)
    assert compatible.gap == pytest.approx(-0.11)


def test_normal_compatible_sample_arrays_match_sample_payload_when_cpp_available() -> None:
    if not _cpp_projection.normal_compatible_indexed_faces_available():
        pytest.skip("C++ normal-compatible indexed projection backend is unavailable")
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.1],
            [0.0, 1.0, -0.1],
            [1.0, 0.0, -0.1],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, 0.01],
            [1.0, 0.0, 0.01],
            [0.0, 1.0, 0.01],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=0.2,
        compiled_batch_projection=True,
    )

    arrays = contact.normal_compatible_sample_arrays(np.vstack([master_nodes, slave_nodes]))

    assert arrays is not None
    assert arrays["gaps"].shape == (1,)
    assert arrays["master_node_ids"].tolist() == [[3, 4, 5]]
    assert arrays["gaps"][0] == pytest.approx(-0.11)


def test_normal_compatible_sample_arrays_drop_incompatible_cpp_candidates() -> None:
    if not _cpp_projection.normal_compatible_indexed_faces_available():
        pytest.skip("C++ normal-compatible indexed projection backend is unavailable")
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, 0.01],
            [1.0, 0.0, 0.01],
            [0.0, 1.0, 0.01],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=0.2,
        compiled_batch_projection=True,
    )

    arrays = contact.normal_compatible_sample_arrays(np.vstack([master_nodes, slave_nodes]))

    assert arrays is not None
    assert arrays["gaps"].shape == (0,)
    assert arrays["master_node_ids"].shape == (0, 3)


def test_source_drive_corotated_elastic_matrix_removes_body_z_rotation() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.3, 0.4],
        ],
        dtype=float,
    )
    hub = RigidHubMPC(np.asarray([0, 1, 2, 3], dtype=np.int64), X, np.zeros(3))
    assembly = build_rigid_hub_reduced_assembly(X, [hub], include_free_nodes=False)
    elastic = _source_drive_corotated_elastic_matrix(
        assembly=assembly,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
    )
    q = np.zeros(assembly.n_reduced_dofs, dtype=float)
    q[assembly.hub_slice(0).start + 5] = 1.7

    np.testing.assert_allclose(np.asarray(elastic @ q).reshape((-1, 3)), 0.0, atol=1.0e-14)


def test_source_drive_finite_visual_jacobian_matches_finite_difference() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.3, 0.4],
        ],
        dtype=float,
    )
    hub = RigidHubMPC(np.asarray([0, 1, 2, 3], dtype=np.int64), X, np.zeros(3))
    assembly = build_rigid_hub_reduced_assembly(X, [hub], include_free_nodes=False)
    q = np.zeros(assembly.n_reduced_dofs, dtype=float)
    q[assembly.hub_slice(0).start + 0] = 0.03
    q[assembly.hub_slice(0).start + 1] = -0.02
    q[assembly.hub_slice(0).start + 5] = 0.73

    def visual_positions(value: np.ndarray) -> np.ndarray:
        x_raw = X + assembly.expand_displacements(value)
        return _source_drive_corotated_positions_and_elastic_displacement(
            model=MechanicsModel.from_tet4_mesh(
                X,
                np.asarray([[0, 1, 2, 3]], dtype=np.int64),
                E=1.0e3,
                nu=0.25,
                density=1.0,
            ),
            x_raw=x_raw,
            body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
            body_reference_points=(np.zeros(3), np.zeros(3)),
            body_rotation_z=(float(value[assembly.hub_slice(0).start + 5]), 0.0),
        )[0]

    jacobian = _source_drive_finite_visual_jacobian(
        assembly=assembly,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(float(q[assembly.hub_slice(0).start + 5]), 0.0),
    )
    eps = 1.0e-7
    for column in range(assembly.n_reduced_dofs):
        perturb = np.zeros_like(q)
        perturb[column] = eps
        fd = (visual_positions(q + perturb) - visual_positions(q - perturb)).reshape(-1) / (2.0 * eps)
        actual = np.asarray(jacobian[:, column].todense()).reshape(-1)
        np.testing.assert_allclose(actual, fd, rtol=5.0e-7, atol=5.0e-8)


def test_source_drive_finite_kinematic_inertia_reduces_to_linear_mass_at_zero_rotation() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=2.0, mass_kind="consistent")
    hub = RigidHubMPC(np.asarray([0, 1, 2, 3], dtype=np.int64), X, np.zeros(3))
    assembly = build_rigid_hub_reduced_assembly(X, [hub], include_free_nodes=False)
    acceleration = np.linspace(0.1, 0.6, assembly.n_reduced_dofs)

    reduced, mass_tangent = _source_drive_finite_kinematic_inertia_response(
        assembly=assembly,
        mass_matrix=model.mass_matrix,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(0.0, 0.0),
        body_angular_velocity_z=(0.0, 0.0),
        reduced_acceleration=acceleration,
    )
    expected_mass = assembly.reduce_matrix(model.mass_matrix).tocsr()

    np.testing.assert_allclose(reduced, np.asarray(expected_mass @ acceleration).reshape(-1), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(mass_tangent.toarray(), expected_mass.toarray(), rtol=1.0e-12, atol=1.0e-12)


def test_source_drive_finite_kinematic_inertia_mass_tangent_matches_acceleration_fd() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.1, 0.2, 0.9],
        ],
        dtype=float,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=2.0, mass_kind="consistent")
    hub = RigidHubMPC(np.asarray([0, 1, 2, 3], dtype=np.int64), X, np.zeros(3))
    assembly = build_rigid_hub_reduced_assembly(X, [hub], include_free_nodes=False)
    acceleration = np.linspace(-0.2, 0.35, assembly.n_reduced_dofs)
    theta = 0.61
    omega = 7.5
    reduced, mass_tangent = _source_drive_finite_kinematic_inertia_response(
        assembly=assembly,
        mass_matrix=model.mass_matrix,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta, 0.0),
        body_angular_velocity_z=(omega, 0.0),
        reduced_acceleration=acceleration,
    )
    eps = 1.0e-7
    for column in range(assembly.n_reduced_dofs):
        perturb = np.zeros_like(acceleration)
        perturb[column] = eps
        plus, _ = _source_drive_finite_kinematic_inertia_response(
            assembly=assembly,
            mass_matrix=model.mass_matrix,
            reference_nodes=X,
            body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
            body_reference_points=(np.zeros(3), np.zeros(3)),
            body_rotation_z=(theta, 0.0),
            body_angular_velocity_z=(omega, 0.0),
            reduced_acceleration=acceleration + perturb,
            include_mass_tangent=False,
        )
        minus, _ = _source_drive_finite_kinematic_inertia_response(
            assembly=assembly,
            mass_matrix=model.mass_matrix,
            reference_nodes=X,
            body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
            body_reference_points=(np.zeros(3), np.zeros(3)),
            body_rotation_z=(theta, 0.0),
            body_angular_velocity_z=(omega, 0.0),
            reduced_acceleration=acceleration - perturb,
            include_mass_tangent=False,
        )
        fd = (plus - minus) / (2.0 * eps)
        np.testing.assert_allclose(fd, np.asarray(mass_tangent[:, column].todense()).reshape(-1), rtol=1.0e-7, atol=1.0e-8)
    assert float(np.linalg.norm(reduced)) > 0.0


def test_source_drive_centripetal_acceleration_points_inward() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=float,
    )

    acc = _source_drive_centripetal_acceleration(
        reference_nodes=X,
        body_node_slices=(slice(0, 3), slice(3, 3)),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(0.0, 0.0),
        body_angular_velocity_z=(4.0, 0.0),
    )

    np.testing.assert_allclose(acc[0], [0.0, 0.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(acc[1], [-32.0, 0.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(acc[2], [0.0, -48.0, 0.0], atol=1.0e-14)


def test_source_drive_centripetal_reduced_tangent_matches_finite_difference() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 0.0, 0.9],
        ],
        dtype=float,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=2.0)
    hub = RigidHubMPC(np.asarray([0, 1, 2, 3], dtype=np.int64), X, np.zeros(3))
    assembly = build_rigid_hub_reduced_assembly(X, [hub], include_free_nodes=False)
    theta = 0.37
    omega = 8.0
    domega_dtheta = 125.0
    reduced, tangent = _source_drive_centripetal_reduced_response(
        assembly=assembly,
        mass_matrix=model.mass_matrix,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta, 0.0),
        body_angular_velocity_z=(omega, 0.0),
        velocity_sensitivity_z=(domega_dtheta, 0.0),
        include_tangent=True,
    )
    eps = 1.0e-7
    plus, _ = _source_drive_centripetal_reduced_response(
        assembly=assembly,
        mass_matrix=model.mass_matrix,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta + eps, 0.0),
        body_angular_velocity_z=(omega + domega_dtheta * eps, 0.0),
        include_tangent=False,
    )
    minus, _ = _source_drive_centripetal_reduced_response(
        assembly=assembly,
        mass_matrix=model.mass_matrix,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta - eps, 0.0),
        body_angular_velocity_z=(omega - domega_dtheta * eps, 0.0),
        include_tangent=False,
    )
    fd_column = (plus - minus) / (2.0 * eps)
    tangent_column = np.asarray(tangent[:, assembly.hub_slice(0).start + 5].toarray()).reshape(-1)

    assert np.linalg.norm(reduced) > 0.0
    np.testing.assert_allclose(tangent_column, fd_column, rtol=1.0e-7, atol=1.0e-7)


def test_source_drive_visual_postprocess_is_objective_for_large_rotation() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=1.0)
    reference_internal = stvk_internal_response(model, model.X, assemble_tangent=True)
    theta = 2.4
    small_rigid = np.cross(np.asarray([[0.0, 0.0, theta]], dtype=float), X)
    state = MechanicsState(
        X + small_rigid,
        np.zeros_like(X),
        np.zeros_like(X),
        time=0.0,
    )

    visual_state, internal = _source_drive_corotated_visual_state_and_internal(
        model=model,
        state=state,
        reference_tangent=reference_internal.tangent,
        body_node_slices=(slice(0, 4), slice(4, 4)),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta, 0.0),
        stress_postprocess="finite_stvk_visual",
    )

    expected_node1 = np.asarray([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    np.testing.assert_allclose(visual_state.x[1], expected_node1, atol=1.0e-14)
    assert float(np.max(internal.von_mises)) <= 1.0e-9


def test_source_drive_visual_postprocess_rejects_unknown_mode() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=1.0)
    reference_internal = stvk_internal_response(model, model.X, assemble_tangent=True)
    state = MechanicsState(X.copy(), np.zeros_like(X), np.zeros_like(X), time=0.0)

    with pytest.raises(ValueError, match="stress_postprocess"):
        _source_drive_corotated_visual_state_and_internal(
            model=model,
            state=state,
            reference_tangent=reference_internal.tangent,
            body_node_slices=(slice(0, 4), slice(4, 4)),
            body_reference_points=(np.zeros(3), np.zeros(3)),
            body_rotation_z=(0.0, 0.0),
            stress_postprocess="bad_mode",
        )


def test_cropped_gear_abaqus_deck_prescribes_matching_rp_motion(tmp_path) -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)
    path = tmp_path / "cropped_gear_alignment.inp"

    _write_abaqus_alignment_deck(
        path,
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=2.0e-3,
        dt=1.0e-3,
        target_overclosure=3.0e-4,
        rotation_rate_z=2.0,
    )

    text = path.read_text(encoding="ascii")
    assert "G2_RP, 1, 6" in text
    assert "G1_RP, 4, 5" in text
    assert "*Boundary, amplitude=RAMP" in text
    assert "*Dynamic, application=MODERATE DISSIPATION" in text
    assert "** Rotational boundary values are radians." in text
    assert "G1_RP, 6, 6, 4.000000000000e-03" in text
    assert "S, E, LE" in text
    assert "CPRESS" not in text
    assert "COPEN" not in text


def test_animation_color_ranges_use_global_sfc_and_abaqus_limits(tmp_path: Path) -> None:
    sfc_manifest = tmp_path / "sfc_manifest.csv"
    abaqus_manifest = tmp_path / "abaqus_manifest.csv"
    sfc_manifest.write_text(
        "frame,time,vtk_file,node_count,element_count,max_displacement_magnitude,max_von_mises,max_strain_norm,max_equivalent_elastic_strain,max_von_mises_nodeavg,max_strain_norm_nodeavg,max_equivalent_elastic_strain_nodeavg\n"
        "0,0,sfc_0000.vtk,1,1,0.1,5.0,0.02,0.002,4.0,0.018,0.0018\n"
        "1,1,sfc_0001.vtk,1,1,0.2,7.0,0.03,0.003,6.0,0.028,0.0028\n",
        encoding="utf-8",
    )
    abaqus_manifest.write_text(
        "frame,source_frame,time,vtk_file,node_count,element_count,max_displacement_magnitude,max_von_mises,max_le_norm,max_equivalent_elastic_strain,max_von_mises_nodeavg,max_le_norm_nodeavg,max_equivalent_elastic_strain_nodeavg\n"
        "0,0,0,abaqus_0000.vtk,1,1,0.15,6.0,0.01,0.001,5.0,0.009,0.0009\n"
        "1,1,1,abaqus_0001.vtk,1,1,0.25,8.0,0.04,0.004,7.0,0.038,0.0038\n",
        encoding="utf-8",
    )

    ranges = write_animation_color_ranges(tmp_path, sfc_manifest=sfc_manifest, abaqus_manifest=abaqus_manifest)
    text = ranges.read_text(encoding="utf-8")

    assert "displacement_magnitude" in text
    assert "von_mises" in text
    assert "strain_norm" in text
    assert "von_mises_nodeavg" in text
    assert "strain_norm_nodeavg" in text
    assert "equivalent_elastic_strain_nodeavg" in text
    assert "0.25" in text
    assert "8.0" in text
    assert "0.04" in text
    assert "7.0" in text
    assert "0.038" in text


def test_paraview_animation_setup_uses_nodeavg_fixed_ranges(tmp_path: Path) -> None:
    ranges = tmp_path / "animation_fixed_color_ranges.csv"
    ranges.write_text(
        "field,recommended_min,recommended_max,sfc_column,abaqus_column,paraview_note\n"
        "displacement_magnitude,0.0,0.25,max_displacement_magnitude,max_displacement_magnitude,note\n"
        "von_mises_nodeavg,0.0,8.0,max_von_mises_nodeavg,max_von_mises_nodeavg,note\n"
        "equivalent_elastic_strain_nodeavg,0.0,0.0038,max_equivalent_elastic_strain_nodeavg,max_equivalent_elastic_strain_nodeavg,note\n",
        encoding="utf-8",
    )
    sfc_pvd = tmp_path / "sfc_vtk" / "sfc.pvd"
    abaqus_pvd = tmp_path / "abaqus_vtk" / "abaqus.pvd"

    script = write_paraview_animation_setup(
        tmp_path,
        color_ranges=ranges,
        sfc_pvd=sfc_pvd,
        abaqus_pvd=abaqus_pvd,
    )
    text = script.read_text(encoding="utf-8")

    assert "DEFAULT_FIELD = \"von_mises_nodeavg\"" in text
    assert "equivalent_elastic_strain_nodeavg" in text
    assert "RescaleTransferFunction(float(vmin), float(vmax))" in text
    assert "sfc_vtk/sfc.pvd" in text
    assert "abaqus_vtk/abaqus.pvd" in text
    assert str(tmp_path.resolve()) not in text


def test_compare_animation_manifests_writes_curve_inputs(tmp_path: Path) -> None:
    sfc_manifest = tmp_path / "sfc_manifest.csv"
    abaqus_manifest = tmp_path / "abaqus_manifest.csv"
    sfc_manifest.write_text(
        "frame,time,vtk_file,node_count,element_count,max_displacement_magnitude,max_von_mises,max_strain_norm,max_equivalent_elastic_strain,max_von_mises_nodeavg,p95_von_mises_nodeavg,max_strain_norm_nodeavg,max_equivalent_elastic_strain_nodeavg,p95_equivalent_elastic_strain_nodeavg,max_displacement_magnitude_object1,max_displacement_magnitude_object2,rp2_rotation_z_rad,rp2_angular_velocity_z_rad_per_s\n"
        "0,0,sfc_0000.vtk,1,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
        "1,0.5,sfc_0001.vtk,1,1,0.2,7.0,0.03,0.003,6.0,5.0,0.028,0.0028,0.0025,0.11,0.20,0.06,0.8\n",
        encoding="utf-8",
    )
    abaqus_manifest.write_text(
        "frame,source_frame,time,vtk_file,node_count,element_count,max_displacement_magnitude,max_von_mises,max_le_norm,max_equivalent_elastic_strain,max_von_mises_nodeavg,p95_von_mises_nodeavg,max_le_norm_nodeavg,max_equivalent_elastic_strain_nodeavg,p95_equivalent_elastic_strain_nodeavg,max_displacement_magnitude_object1,max_displacement_magnitude_object2,rp2_rotation_z_rad,rp2_angular_velocity_z_rad_per_s\n"
        "0,0,0,abaqus_0000.vtk,1,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
        "1,1,1,abaqus_0001.vtk,1,1,0.4,10.0,0.08,0.006,8.0,6.0,0.04,0.0056,0.0030,0.10,0.30,0.10,1.2\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "errors.csv"
    out_png = tmp_path / "curves.png"

    rows = compare_animation_manifests(
        sfc_manifest=sfc_manifest,
        abaqus_manifest=abaqus_manifest,
        out_csv=out_csv,
        out_png=out_png,
    )

    assert out_csv.exists()
    assert out_png.exists()
    assert len(rows) == 2
    assert float(rows[-1]["abaqus_max_displacement_magnitude"]) == pytest.approx(0.2)
    assert float(rows[-1]["max_von_mises_nodeavg_rel_error"]) == pytest.approx(0.5)
    assert float(rows[-1]["p95_von_mises_nodeavg_rel_error"]) == pytest.approx(2.0 / 3.0)
    assert "p95_equivalent_elastic_strain_nodeavg_rel_error" in rows[-1]
    assert "max_equivalent_elastic_strain_nodeavg_rel_error" in rows[-1]
    assert "max_strain_norm_nodeavg_rel_error" not in rows[-1]
    assert "max_displacement_magnitude_object1_rel_error" in rows[-1]
    assert "max_displacement_magnitude_object2_rel_error" in rows[-1]
    assert "rp2_rotation_z_rad_rel_error" in rows[-1]
    assert "rp2_angular_velocity_z_rad_per_s_rel_error" in rows[-1]


def test_source_gear_manifest_alignment_entrypoint_writes_artifacts(tmp_path: Path) -> None:
    sfc_dir = tmp_path / "sfc_vtk"
    abaqus_dir = tmp_path / "abaqus_vtk"
    sfc_dir.mkdir()
    abaqus_dir.mkdir()
    sfc_manifest = sfc_dir / "sfc_manifest.csv"
    abaqus_manifest = abaqus_dir / "abaqus_manifest.csv"
    sfc_manifest.write_text(
        "frame,time,vtk_file,node_count,element_count,max_displacement_magnitude,max_von_mises,max_strain_norm,max_equivalent_elastic_strain,max_von_mises_nodeavg,p95_von_mises_nodeavg,max_strain_norm_nodeavg,max_equivalent_elastic_strain_nodeavg,p95_equivalent_elastic_strain_nodeavg\n"
        "0,0,sfc_0000.vtk,1,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
        "1,1,sfc_0001.vtk,1,1,0.2,7.0,0.03,0.003,6.0,5.0,0.028,0.0028,0.0025\n",
        encoding="utf-8",
    )
    abaqus_manifest.write_text(
        "frame,source_frame,time,vtk_file,node_count,element_count,max_displacement_magnitude,max_von_mises,max_le_norm,max_equivalent_elastic_strain,max_von_mises_nodeavg,p95_von_mises_nodeavg,max_le_norm_nodeavg,max_equivalent_elastic_strain_nodeavg,p95_equivalent_elastic_strain_nodeavg\n"
        "0,0,0,abaqus_0000.vtk,1,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
        "1,1,1,abaqus_0001.vtk,1,1,0.25,8.0,0.04,0.004,7.0,6.0,0.038,0.0038,0.003\n",
        encoding="utf-8",
    )

    summary = align_manifests(
        sfc_manifest=sfc_manifest,
        abaqus_manifest=abaqus_manifest,
        out_dir=tmp_path / "comparison",
    )

    assert int(summary["frame_count"]) == 2
    assert Path(str(summary["summary"])).exists()
    assert Path(str(summary["errors"])).exists()
    assert Path(str(summary["curves"])).exists()
    assert Path(str(summary["ranges"])).exists()
    script = Path(str(summary["paraview"]))
    assert script.exists()
    script_text = script.read_text(encoding="utf-8")
    assert "../sfc_vtk/sfc.pvd" in script_text
    assert "../abaqus_vtk/abaqus.pvd" in script_text


def test_cropped_gear_hard_contact_path_runs_one_implicit_step() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        max_iterations=4,
        automatic_increment=False,
    )

    assert len(history) == 1
    assert summary["status"] == "completed"
    assert summary["contact_mode"] == "hard"
    assert summary["hard_enforcement"] == "abaqus_standard_penalty"
    assert float(summary["hht_alpha"]) < 0.0
    assert summary["nodes"] > 0
    assert summary["elements"] > 0
    assert int(history[-1]["hard_contact_samples"]) > 0
    assert int(history[-1]["hard_active_set_converged"]) == 1
    assert "timing_internal_tangent_seconds" in summary
    assert float(summary["timing_internal_tangent_seconds"]) >= 0.0
    assert "timing_hard_contact_solve_seconds" in summary
    assert float(summary["timing_hard_contact_solve_seconds"]) >= 0.0
    assert "rp_force_norm" in history[-1]
    assert float(history[-1]["rp_force_norm"]) >= 0.0
    assert "opposing_rp_force_norm" in history[-1]
    assert float(history[-1]["opposing_rp_force_norm"]) >= 0.0
    assert history[-1]["rp_reaction_definition"] == "static_physical_constraint_residual"
    assert "rp1_hht_rp_force_norm" in history[-1]
    assert "rp1_dynamic_rp_force_norm" in history[-1]
    assert "rp1_inertia_rp_force_norm" in history[-1]
    assert "rp1_contact_rp_force_norm" in history[-1]
    assert "contact_multiplier_sum" in history[-1]
    assert "p95_equivalent_elastic_strain" in history[-1]
    assert float(history[-1]["p95_equivalent_elastic_strain"]) >= 0.0
    assert "p95_von_mises_nodeavg" in history[-1]
    assert "p95_equivalent_elastic_strain_nodeavg" in history[-1]
    assert float(history[-1]["p95_von_mises_nodeavg"]) >= 0.0
    assert float(history[-1]["p95_equivalent_elastic_strain_nodeavg"]) >= 0.0


def test_cropped_gear_hard_contact_supports_surface_patch_constraint_averaging() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        max_iterations=4,
        constraint_averaging="surface_patch",
        automatic_increment=False,
    )

    assert len(history) == 1
    assert summary["constraint_averaging"] == "surface_patch"
    assert int(history[-1]["hard_constraints"]) <= int(history[-1]["hard_contact_samples"])
    assert int(history[-1]["hard_active_set_converged"]) == 1


def test_cropped_gear_hard_contact_supports_element_pressure_smoothing() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        max_iterations=4,
        hard_enforcement="element_pressure_smoothing",
        pressure_smoothing_factor=4.0,
        automatic_increment=False,
    )

    assert len(history) == 1
    assert summary["hard_enforcement"] == "element_pressure_smoothing"
    assert float(summary["effective_hard_pressure_stiffness"]) > 5.0e9
    assert float(summary["contact_patch_min_edge_length"]) > 0.0


def test_cropped_gear_hard_contact_supports_abaqus_standard_penalty() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        max_iterations=4,
        hard_enforcement="abaqus_standard_penalty",
        automatic_increment=False,
    )

    assert len(history) == 1
    assert summary["hard_enforcement"] == "abaqus_standard_penalty"
    assert float(summary["effective_hard_pressure_stiffness"]) > 5.0e9
    assert int(summary["accepted_increment_count"]) == 1


def test_cropped_gear_abaqus_deck_can_use_hard_contact(tmp_path) -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)
    path = tmp_path / "cropped_gear_hard_alignment.inp"

    _write_abaqus_alignment_deck(
        path,
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=2.0e-3,
        dt=1.0e-3,
        target_overclosure=3.0e-4,
        rotation_rate_z=2.0,
        contact_mode="hard",
    )

    text = path.read_text(encoding="ascii")
    assert "*Surface Interaction, name=SFC_HARD" in text
    assert "*Surface Behavior, pressure-overclosure=HARD" in text
    assert "interaction=SFC_HARD, type=SURFACE TO SURFACE" in text
    assert "pressure-overclosure=LINEAR" not in text
