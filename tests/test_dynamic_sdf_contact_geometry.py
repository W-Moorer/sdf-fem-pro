from __future__ import annotations

import numpy as np
import pytest

from sfc.contact import DynamicSurfaceSDFContactGeometry
from sfc.fem.calculix_aligned import assemble_contact_response


def test_dynamic_sdf_contact_geometry_uses_slave_face_quadrature() -> None:
    slave_x = np.asarray(
        [
            [0.0, 0.0, -0.1],
            [1.0, 0.0, -0.1],
            [0.0, 1.0, -0.1],
        ],
        dtype=float,
    )
    slave_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    master_x = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    master_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    geometry = DynamicSurfaceSDFContactGeometry(
        slave_faces,
        master_x,
        master_faces,
        candidate_provider=lambda point: np.asarray([0], dtype=np.int64),
        stiffness=100.0,
    )

    samples = list(geometry.samples(slave_x))
    response = assemble_contact_response(samples, n_nodes=3)

    assert len(samples) == 3
    assert all(sample.gap == pytest.approx(-0.1) for sample in samples)
    assert all(np.allclose(sample.normal, [0.0, 0.0, 1.0]) for sample in samples)
    assert sum(sample.area for sample in samples) == pytest.approx(0.5)
    assert response.normal_force == pytest.approx(100.0 * 0.5 * 0.1)
    assert np.sum(response.force[:, 2]) == pytest.approx(response.normal_force)


def test_dynamic_sdf_contact_geometry_requires_broad_phase_candidates() -> None:
    slave_x = np.asarray(
        [
            [0.0, 0.0, -0.1],
            [1.0, 0.0, -0.1],
            [0.0, 1.0, -0.1],
        ],
        dtype=float,
    )
    face = np.asarray([[0, 1, 2]], dtype=np.int64)
    geometry = DynamicSurfaceSDFContactGeometry(
        face,
        slave_x.copy(),
        face,
        candidate_provider=lambda point: np.empty(0, dtype=np.int64),
        stiffness=100.0,
    )

    with pytest.raises(ValueError, match="candidate_face_ids"):
        list(geometry.samples(slave_x))
