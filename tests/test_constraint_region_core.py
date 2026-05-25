from __future__ import annotations

import numpy as np
import pytest

from sfc.contact.constraint_region import (
    aggregate_contact_sample_arrays,
    constraint_region_gap_jacobian_sparse_from_arrays,
    constraint_region_pressure_tangent_scales_from_arrays,
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
