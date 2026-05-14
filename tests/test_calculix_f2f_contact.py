from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.fem.calculix_aligned import PlaneContactGeometry, assemble_contact_response
from validation.calculix_f2f_contact import (
    C3D4_FACE_CENTROID_WEIGHTS,
    CalculixC3D4FaceToFacePlaneContactGeometry,
    CalculixC3D4FaceToFaceSDFContactGeometry,
)


def test_calculix_c3d4_f2f_uses_one_centroid_sample_per_face() -> None:
    x = np.asarray([[0.0, 0.0, -0.1], [1.0, 0.0, -0.1], [0.0, 1.0, -0.1]], dtype=float)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    geometry = CalculixC3D4FaceToFacePlaneContactGeometry(faces, plane_z=0.0, stiffness=100.0)

    samples = list(geometry.samples(x))
    springs = geometry.contact_springs(x)

    assert len(samples) == 1
    assert len(springs) == 1
    assert springs[0].active
    assert samples[0].shape_weights == pytest.approx(C3D4_FACE_CENTROID_WEIGHTS)
    assert samples[0].area == pytest.approx(0.5)
    assert samples[0].gap == pytest.approx(-0.1)

    response = assemble_contact_response(samples, n_nodes=3)
    assert response.active_count == 1
    assert response.normal_force == pytest.approx(100.0 * 0.5 * 0.1)
    assert response.energy == pytest.approx(0.5 * 100.0 * 0.5 * 0.1 * 0.1)
    assert np.sum(response.force[:, 2]) == pytest.approx(response.normal_force)


def test_calculix_c3d4_f2f_differs_from_three_point_plane_contact() -> None:
    x = np.asarray([[0.0, 0.0, -0.1], [1.0, 0.0, -0.1], [0.0, 1.0, -0.1]], dtype=float)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)

    strict = CalculixC3D4FaceToFacePlaneContactGeometry(faces, plane_z=0.0, stiffness=100.0)
    three_point = PlaneContactGeometry(faces, plane_z=0.0, stiffness=100.0)

    assert len(list(strict.samples(x))) == 1
    assert len(list(three_point.samples(x))) == 3
    assert assemble_contact_response(strict.samples(x), 3).normal_force == pytest.approx(
        assemble_contact_response(three_point.samples(x), 3).normal_force
    )


def test_calculix_c3d4_f2f_dynamic_sdf_plane_matches_analytic_plane() -> None:
    slave_x = np.asarray([[0.0, 0.0, -0.1], [1.0, 0.0, -0.1], [0.0, 1.0, -0.1]], dtype=float)
    slave_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    master_x = np.asarray(
        [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]],
        dtype=float,
    )
    master_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    candidates = lambda _point: np.asarray([0, 1], dtype=np.int64)

    plane = CalculixC3D4FaceToFacePlaneContactGeometry(slave_faces, plane_z=0.0, stiffness=100.0)
    sdf = CalculixC3D4FaceToFaceSDFContactGeometry(slave_faces, master_x, master_faces, candidates, stiffness=100.0)

    plane_sample = list(plane.samples(slave_x))[0]
    sdf_sample = list(sdf.samples(slave_x))[0]
    assert sdf_sample.gap == pytest.approx(plane_sample.gap)
    assert sdf_sample.area == pytest.approx(plane_sample.area)
    assert sdf_sample.normal == pytest.approx(plane_sample.normal)

    plane_response = assemble_contact_response(plane.samples(slave_x), 3)
    sdf_response = assemble_contact_response(sdf.samples(slave_x), 3)
    assert sdf_response.normal_force == pytest.approx(plane_response.normal_force)
    assert sdf_response.energy == pytest.approx(plane_response.energy)
