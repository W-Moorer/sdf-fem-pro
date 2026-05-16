import numpy as np

from sfc.contact import (
    PenaltyContactBackend,
    SurfaceSample,
    contact_constraint_from_sample,
    penalty_contact_response,
)


MASTER_TRIANGLE = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
MASTER_FACES = np.array([[0, 1, 2]], dtype=np.int64)
SAMPLE = SurfaceSample(
    node_ids=np.array([0], dtype=np.int64),
    weights=np.array([1.0]),
    candidate_face_ids=np.array([0], dtype=np.int64),
)


def test_penalty_contact_backend_matches_direct_penalty_response() -> None:
    constraint = contact_constraint_from_sample(
        np.array([[0.25, 0.25, -0.1]]),
        SAMPLE,
        MASTER_TRIANGLE,
        MASTER_FACES,
    )
    backend = PenaltyContactBackend(stiffness=100.0)

    result = backend.assemble_response([constraint], n_total_dofs=12, master_dof_offset=3)
    expected_force, expected_stiffness = penalty_contact_response(
        [constraint],
        stiffness=100.0,
        n_total_dofs=12,
        master_dof_offset=3,
    )

    assert len(result.constraints) == 1
    assert np.isclose(result.constraints[0].g, constraint.g)
    assert np.allclose(result.force, expected_force)
    assert np.allclose(result.stiffness.toarray(), expected_stiffness.toarray())


def test_penalty_contact_backend_evaluates_samples_with_dynamic_sdf() -> None:
    backend = PenaltyContactBackend(stiffness=100.0)

    result = backend.evaluate(
        np.array([[0.25, 0.25, -0.1]]),
        [SAMPLE],
        MASTER_TRIANGLE,
        MASTER_FACES,
        n_total_dofs=12,
        master_dof_offset=3,
    )

    assert len(result.constraints) == 1
    assert result.constraints[0].g < 0.0
    assert result.force[2] > 0.0
