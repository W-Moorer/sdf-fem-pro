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
    CalculixContactConvergenceHeuristic,
    CalculixC3D4FaceToFacePlaneContactGeometry,
    CalculixC3D4FaceToFaceSDFContactGeometry,
    CalculixF2FContactLifecycle,
    CalculixF2FContactSpring,
    PersistentCalculixC3D4FaceToFacePlaneContactGeometry,
    assemble_deformable_f2f_contact_response,
    calculix_equivalent_contact_element_count,
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


def test_persistent_lifecycle_generates_persists_and_releases_springs() -> None:
    lifecycle = CalculixF2FContactLifecycle(release_tolerance=0.02)
    penetrating = _spring(clearance=-0.01)
    near_open = _spring(clearance=0.01)
    far_open = _spring(clearance=0.03)

    active = lifecycle.update([penetrating])
    assert len(active) == 1
    assert lifecycle.events[0].status == "generated"

    active = lifecycle.update([near_open])
    assert len(active) == 1
    assert lifecycle.events[0].status == "persisted"
    assert lifecycle.generated_count == 1
    assert lifecycle.penetrating_count == 0

    active = lifecycle.update([far_open])
    assert active == []
    assert lifecycle.events[0].status == "released"
    assert lifecycle.generated_count == 0


def test_lifecycle_snapshot_restore_rolls_back_trial_contact_state() -> None:
    lifecycle = CalculixF2FContactLifecycle(release_tolerance=0.02)
    lifecycle.update([_spring(clearance=-0.01)])
    snapshot = lifecycle.snapshot()

    lifecycle.update([_spring(clearance=0.03)])
    assert lifecycle.generated_count == 0

    lifecycle.restore(snapshot)
    assert lifecycle.generated_count == 1
    assert lifecycle.events[0].status == "generated"


def test_persistent_plane_geometry_reports_generated_count_separately() -> None:
    x = np.asarray([[0.0, 0.0, -0.1], [1.0, 0.0, -0.1], [0.0, 1.0, -0.1]], dtype=float)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    geometry = PersistentCalculixC3D4FaceToFacePlaneContactGeometry(faces, plane_z=0.0, stiffness=100.0)

    response = assemble_contact_response(geometry.samples(x), n_nodes=3)

    assert geometry.generated_contact_count == 1
    assert response.active_count == 1
    assert response.normal_force == pytest.approx(5.0)


def test_calculix_equivalent_cnum_counts_slave_and_master_nodes() -> None:
    x = np.asarray([[0.0, 0.0, -0.1], [1.0, 0.0, -0.1], [0.0, 1.0, -0.1]], dtype=float)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    geometry = PersistentCalculixC3D4FaceToFacePlaneContactGeometry(faces, plane_z=0.0, stiffness=100.0)

    _ = list(geometry.samples(x))

    assert geometry.generated_contact_count == 1
    assert geometry.calculix_contact_element_count == 7
    assert calculix_equivalent_contact_element_count(geometry) == 7


def test_calculix_c3d4_f2f_area_uses_current_face_area() -> None:
    x = np.asarray([[0.0, 0.0, -0.1], [2.0, 0.0, -0.1], [0.0, 1.0, -0.1]], dtype=float)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    geometry = CalculixC3D4FaceToFacePlaneContactGeometry(faces, plane_z=0.0, stiffness=100.0)

    spring = geometry.contact_springs(x)[0]

    assert spring.spring_area == pytest.approx(1.0)
    assert assemble_contact_response(geometry.samples(x), 3).normal_force == pytest.approx(10.0)


def test_convergence_heuristic_recommends_cutback_on_oscillation() -> None:
    heuristic = CalculixContactConvergenceHeuristic()
    first = heuristic.update(iteration=1, active_count=2, residual_norm=10.0)
    second = heuristic.update(iteration=2, active_count=0, residual_norm=9.0)
    third = heuristic.update(iteration=3, active_count=2, residual_norm=9.5)

    assert not first.recommended_cutback
    assert not second.recommended_cutback
    assert third.recommended_cutback
    assert third.reason == "active_set_oscillation"


def test_deformable_f2f_master_slave_response_is_action_reaction_balanced() -> None:
    spring = _spring(
        clearance=-0.1,
        master_nodes=np.asarray([0, 1, 2], dtype=np.int64),
        master_weights=np.asarray([0.2, 0.3, 0.5], dtype=float),
    )

    response = assemble_deformable_f2f_contact_response(
        [spring],
        stiffness=100.0,
        n_slave_nodes=3,
        n_master_nodes=3,
    )

    assert response.active_count == 1
    assert response.generated_count == 1
    assert response.normal_force == pytest.approx(5.0)
    slave_force = response.force[:9].reshape(3, 3).sum(axis=0)
    master_force = response.force[9:].reshape(3, 3).sum(axis=0)
    assert slave_force == pytest.approx(np.asarray([0.0, 0.0, 5.0]))
    assert master_force == pytest.approx(np.asarray([0.0, 0.0, -5.0]))
    assert slave_force + master_force == pytest.approx(np.zeros(3))
    assert response.stiffness.shape == (18, 18)


def _spring(
    *,
    clearance: float,
    master_nodes: np.ndarray | None = None,
    master_weights: np.ndarray | None = None,
) -> CalculixF2FContactSpring:
    return CalculixF2FContactSpring(
        slave_face_index=0,
        slave_nodes=np.asarray([0, 1, 2], dtype=np.int64),
        slave_weights=np.full(3, 1.0 / 3.0, dtype=float),
        master_face_index=-1 if master_nodes is None else 0,
        master_nodes=np.zeros(0, dtype=np.int64) if master_nodes is None else master_nodes,
        master_weights=np.zeros(0, dtype=float) if master_weights is None else master_weights,
        normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
        spring_area=0.5,
        clearance=float(clearance),
    )
