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
    PersistentCalculixC3D4FaceToFaceSDFContactGeometry,
    assemble_deformable_f2f_contact_response,
    calculix_equivalent_contact_element_count,
    calculix_contact_lifecycle_decision,
    calculix_hard_linear_spring_law,
    calculix_static_clearance_ramp,
    calculix_static_iloop_generation_decision,
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


def test_lifecycle_decision_reports_generated_persisted_cutback_and_released_states() -> None:
    inactive = calculix_contact_lifecycle_decision(clearance=0.03, spring_area=0.25, was_generated=False)
    generated = calculix_contact_lifecycle_decision(clearance=-0.01, spring_area=0.25, was_generated=False)
    persisted = calculix_contact_lifecycle_decision(
        clearance=0.01,
        spring_area=0.25,
        was_generated=True,
        release_tolerance=0.02,
    )
    cutback = calculix_contact_lifecycle_decision(
        clearance=0.03,
        spring_area=0.25,
        was_generated=True,
        release_tolerance=0.02,
        cutback=True,
    )
    released = calculix_contact_lifecycle_decision(
        clearance=0.03,
        spring_area=0.25,
        was_generated=True,
        release_tolerance=0.02,
    )
    reactivated = calculix_contact_lifecycle_decision(
        clearance=-0.01,
        spring_area=0.25,
        was_generated=False,
        was_ever_generated=True,
    )
    no_master = calculix_contact_lifecycle_decision(
        clearance=-0.01,
        spring_area=0.25,
        was_generated=False,
        candidate_valid=False,
    )
    lost = calculix_contact_lifecycle_decision(
        clearance=-0.01,
        spring_area=0.25,
        was_generated=True,
        candidate_valid=False,
    )
    lost_cutback = calculix_contact_lifecycle_decision(
        clearance=-0.01,
        spring_area=0.25,
        was_generated=True,
        cutback=True,
        candidate_valid=False,
    )

    assert inactive.status == "inactive"
    assert not inactive.generated
    assert generated.status == "generated"
    assert generated.generated
    assert generated.force_active
    assert persisted.status == "persisted"
    assert persisted.generated
    assert not persisted.force_active
    assert cutback.status == "cutback_persisted"
    assert cutback.generated
    assert released.status == "released"
    assert not released.generated
    assert reactivated.status == "reactivated"
    assert reactivated.generated
    assert no_master.status == "no_master_candidate"
    assert not no_master.generated
    assert not no_master.candidate_valid
    assert lost.status == "lost_candidate_released"
    assert not lost.generated
    assert lost_cutback.status == "lost_candidate_cutback_persisted"
    assert lost_cutback.generated
    assert lost_cutback.cutback_persisted


def test_lifecycle_decision_can_generate_positive_clearance_contact_type() -> None:
    generated_open = calculix_contact_lifecycle_decision(
        clearance=0.03,
        spring_area=0.25,
        was_generated=False,
        allow_positive_clearance_generation=True,
    )
    persisted_open = calculix_contact_lifecycle_decision(
        clearance=0.04,
        spring_area=0.25,
        was_generated=True,
        release_tolerance=0.0,
        allow_positive_clearance_generation=True,
    )
    reactivated_open = calculix_contact_lifecycle_decision(
        clearance=0.02,
        spring_area=0.25,
        was_generated=False,
        was_ever_generated=True,
        allow_positive_clearance_generation=True,
    )

    assert generated_open.status == "generated_positive_clearance"
    assert generated_open.generated
    assert generated_open.positive_clearance_allowed
    assert not generated_open.force_active
    assert persisted_open.status == "persisted_positive_clearance"
    assert persisted_open.generated
    assert not persisted_open.force_active
    assert reactivated_open.status == "reactivated_positive_clearance"
    assert reactivated_open.generated


def test_lifecycle_can_keep_positive_clearance_springs_without_force() -> None:
    lifecycle = CalculixF2FContactLifecycle(allow_positive_clearance_generation=True)

    active = lifecycle.update([_spring(clearance=0.02)])

    assert len(active) == 1
    assert lifecycle.generated_count == 1
    assert lifecycle.penetrating_count == 0
    assert lifecycle.events[0].status == "generated_positive_clearance"
    assert lifecycle.events[0].positive_clearance_allowed
    assert not lifecycle.events[0].force_active


def test_persistent_lifecycle_reports_reactivation_after_release() -> None:
    lifecycle = CalculixF2FContactLifecycle(release_tolerance=0.0)

    lifecycle.update([_spring(clearance=-0.01)])
    assert lifecycle.events[0].status == "generated"

    lifecycle.update([_spring(clearance=0.03)])
    assert lifecycle.events[0].status == "released"
    assert lifecycle.generated_count == 0

    lifecycle.update([_spring(clearance=-0.02)])
    assert lifecycle.events[0].status == "reactivated"
    assert lifecycle.generated_count == 1


def test_lifecycle_can_keep_lost_candidate_during_cutback() -> None:
    lifecycle = CalculixF2FContactLifecycle()
    lifecycle.update([_spring(clearance=-0.01)])

    released = lifecycle.update([])
    assert released == []
    assert lifecycle.events[0].status == "lost_candidate_released"

    lifecycle.update([_spring(clearance=-0.01)])
    kept = lifecycle.update([], cutback=True)
    assert len(kept) == 1
    assert lifecycle.events[0].status == "lost_candidate_cutback_persisted"
    assert lifecycle.events[0].cutback_persisted


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


def test_persistent_plane_geometry_can_generate_positive_clearance_without_force() -> None:
    x = np.asarray([[0.0, 0.0, 0.1], [1.0, 0.0, 0.1], [0.0, 1.0, 0.1]], dtype=float)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    geometry = PersistentCalculixC3D4FaceToFacePlaneContactGeometry(
        faces,
        plane_z=0.0,
        stiffness=100.0,
        allow_positive_clearance_generation=True,
    )

    response = assemble_contact_response(geometry.samples(x), n_nodes=3)

    assert geometry.generated_contact_count == 1
    assert response.active_count == 0
    assert response.normal_force == pytest.approx(0.0)
    assert geometry.lifecycle is not None
    assert geometry.lifecycle.events[0].status == "generated_positive_clearance"


def test_persistent_sdf_geometry_reuses_stored_master_projection() -> None:
    slave_x = np.asarray([[0.0, 0.0, -0.1], [1.0, 0.0, -0.1], [0.0, 1.0, -0.1]], dtype=float)
    slave_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    master_x = np.asarray(
        [
            [-2.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [-2.0, -2.0, -0.5],
            [2.0, -2.0, -0.5],
            [2.0, 2.0, -0.5],
        ],
        dtype=float,
    )
    master_faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    candidate = {"face": 0}

    def candidates(_point: np.ndarray) -> np.ndarray:
        return np.asarray([candidate["face"]], dtype=np.int64)

    geometry = PersistentCalculixC3D4FaceToFaceSDFContactGeometry(
        slave_faces,
        master_x,
        master_faces,
        candidates,
        stiffness=100.0,
    )

    first = geometry.contact_springs(slave_x)[0]
    candidate["face"] = 1
    second = geometry.contact_springs(slave_x)[0]

    assert first.master_face_index == 0
    assert second.master_face_index == 0
    assert np.array_equal(second.master_nodes, first.master_nodes)
    assert second.master_weights == pytest.approx(first.master_weights)
    assert second.normal == pytest.approx(first.normal)
    assert second.clearance == pytest.approx(first.clearance)


def test_sdf_geometry_skips_empty_candidate_queries_like_no_master_face() -> None:
    slave_x = np.asarray([[0.0, 0.0, -0.1], [1.0, 0.0, -0.1], [0.0, 1.0, -0.1]], dtype=float)
    slave_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    master_x = np.asarray([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0]], dtype=float)
    master_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    geometry = CalculixC3D4FaceToFaceSDFContactGeometry(
        slave_faces,
        master_x,
        master_faces,
        lambda _point: np.zeros(0, dtype=np.int64),
        stiffness=100.0,
    )

    assert geometry.contact_springs(slave_x) == []
    assert list(geometry.samples(slave_x)) == []


def test_persistent_sdf_geometry_keeps_lost_candidate_on_cutback() -> None:
    slave_x = np.asarray([[0.0, 0.0, -0.1], [1.0, 0.0, -0.1], [0.0, 1.0, -0.1]], dtype=float)
    slave_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    master_x = np.asarray([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0]], dtype=float)
    master_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    candidate = {"active": True}

    def candidates(_point: np.ndarray) -> np.ndarray:
        return np.asarray([0], dtype=np.int64) if candidate["active"] else np.zeros(0, dtype=np.int64)

    geometry = PersistentCalculixC3D4FaceToFaceSDFContactGeometry(
        slave_faces,
        master_x,
        master_faces,
        candidates,
        stiffness=100.0,
    )
    first = geometry.contact_springs(slave_x)
    assert len(first) == 1

    candidate["active"] = False
    geometry.set_cutback_retry(True)
    kept = geometry.contact_springs(slave_x)

    assert len(kept) == 1
    assert geometry.lifecycle is not None
    assert geometry.lifecycle.events[0].status == "lost_candidate_cutback_persisted"


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


def test_calculix_hard_linear_spring_law_has_no_tension_and_area_weighted_compression() -> None:
    open_law = calculix_hard_linear_spring_law(clearance=0.01, spring_area=0.5, pressure_stiffness=100.0)
    closed_law = calculix_hard_linear_spring_law(clearance=-0.02, spring_area=0.5, pressure_stiffness=100.0)
    scaled_law = calculix_hard_linear_spring_law(
        clearance=-0.02,
        spring_area=0.5,
        pressure_stiffness=100.0,
        kscale=2.0,
    )

    assert not open_law.active
    assert open_law.force_magnitude == pytest.approx(0.0)
    assert closed_law.active
    assert closed_law.penetration == pytest.approx(0.02)
    assert closed_law.tangent_scale == pytest.approx(50.0)
    assert closed_law.force_magnitude == pytest.approx(1.0)
    assert closed_law.energy == pytest.approx(0.5 * 50.0 * 0.02 * 0.02)
    assert scaled_law.tangent_scale == pytest.approx(25.0)
    assert scaled_law.force_magnitude == pytest.approx(0.5)


def test_static_clearance_ramp_stores_initial_overclosure_offset() -> None:
    ramp = calculix_static_clearance_ramp(
        clearance=-0.02,
        theta=0.2,
        reltime=0.5,
        pressure_stiffness=1000.0,
        initialize=True,
        initial_adjustment_allowed=True,
    )

    assert ramp.springarea_offset == pytest.approx(-0.025)
    assert ramp.adjusted_clearance == pytest.approx(-0.0075)
    assert ramp.initialized_offset
    assert not ramp.small_gap_closed


def test_static_clearance_ramp_closes_small_positive_initial_gap() -> None:
    ramp = calculix_static_clearance_ramp(
        clearance=0.0005,
        theta=0.2,
        reltime=0.0,
        pressure_stiffness=1000.0,
        initialize=True,
        initial_adjustment_allowed=True,
    )

    assert ramp.adjusted_clearance == pytest.approx(0.0)
    assert ramp.springarea_offset == pytest.approx(0.0)
    assert not ramp.initialized_offset
    assert ramp.small_gap_closed


def test_static_clearance_ramp_reuses_previous_offset_after_initialization() -> None:
    ramp = calculix_static_clearance_ramp(
        clearance=0.0,
        theta=0.2,
        reltime=0.25,
        pressure_stiffness=1000.0,
        previous_springarea_offset=-0.025,
    )

    assert ramp.adjusted_clearance == pytest.approx(0.01875)
    assert ramp.springarea_offset == pytest.approx(-0.025)
    assert not ramp.initialized_offset


def test_static_iloop_generation_uses_penetration_without_cutback() -> None:
    open_decision = calculix_static_iloop_generation_decision(clearance=0.01)
    penetrating = calculix_static_iloop_generation_decision(clearance=-0.01)

    assert open_decision.status == "released_positive_clearance"
    assert not open_decision.generated
    assert not open_decision.force_active
    assert penetrating.status == "generated_by_penetration"
    assert penetrating.generated
    assert penetrating.force_active
    assert penetrating.increments_iact


def test_static_iloop_generation_counts_previous_increment_contact() -> None:
    decision = calculix_static_iloop_generation_decision(
        clearance=-0.01,
        step=2,
        increment=1,
        iteration=0,
        n_contact_material_terms=7,
        tangential_regularization=0.1,
        previous_state_norm=1.0e-12,
    )

    assert decision.generated
    assert decision.increments_iprev
    assert decision.previous_contact_present


def test_static_iloop_cutback_keeps_previous_positive_clearance_contact() -> None:
    kept = calculix_static_iloop_generation_decision(
        clearance=0.02,
        cutback=True,
        n_contact_material_terms=7,
        tangential_regularization=0.1,
        previous_state_norm=1.0e-12,
    )
    released = calculix_static_iloop_generation_decision(
        clearance=0.02,
        cutback=True,
        n_contact_material_terms=7,
        tangential_regularization=0.1,
        previous_state_norm=0.0,
    )

    assert kept.status == "cutback_previous_or_penetration_generated"
    assert kept.generated
    assert not kept.force_active
    assert kept.increments_iact
    assert released.status == "cutback_released_positive_without_previous"
    assert not released.generated


def test_static_iloop2_uses_previous_state_only() -> None:
    kept = calculix_static_iloop_generation_decision(
        clearance=0.02,
        iloop=2,
        previous_state_norm=1.0e-12,
    )
    released = calculix_static_iloop_generation_decision(
        clearance=-0.02,
        iloop=2,
        previous_state_norm=0.0,
    )

    assert kept.status == "iloop2_previous_generated"
    assert kept.generated
    assert not kept.force_active
    assert released.status == "iloop2_released_without_previous"
    assert not released.generated


def test_static_iloop_generation_models_no_master_aleatoric_and_contact_type_branches() -> None:
    no_master = calculix_static_iloop_generation_decision(clearance=-0.01, master_face_detected=False)
    aleatoric = calculix_static_iloop_generation_decision(
        clearance=-0.01,
        aleatoric_enabled=True,
        aleatoric_harvest=0.95,
        aleatoric_probability=0.1,
    )
    contact_type = calculix_static_iloop_generation_decision(
        clearance=0.03,
        contact_type_allows_positive_clearance=True,
    )

    assert no_master.status == "no_master_candidate"
    assert not no_master.generated
    assert aleatoric.status == "aleatoric_removed"
    assert aleatoric.aleatoric_removed
    assert not aleatoric.generated
    assert contact_type.status == "contact_type_generated"
    assert contact_type.generated
    assert not contact_type.force_active


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
