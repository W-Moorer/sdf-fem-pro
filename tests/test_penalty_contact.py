import numpy as np

from sfc.contact import (
    SurfaceSample,
    contact_constraint_from_sample,
    contact_jacobian_row,
    penalty_contact_response,
)
from sfc.mesh import extract_boundary_faces
from sfc.mesh.topology import VolumeMesh


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


def _constraint(slave_point: np.ndarray):
    return contact_constraint_from_sample(
        np.asarray(slave_point, dtype=float).reshape(1, 3),
        SAMPLE,
        MASTER_TRIANGLE,
        MASTER_FACES,
    )


def test_inactive_contact_produces_zero_force() -> None:
    constraint = _constraint(np.array([0.25, 0.25, 0.1]))

    force, K = penalty_contact_response(
        [constraint],
        stiffness=100.0,
        n_total_dofs=12,
        slave_dof_offset=0,
        master_dof_offset=3,
    )

    assert constraint.g > 0.0
    assert np.allclose(force, 0.0)
    assert K.nnz == 0


def test_penetrating_point_produces_force_along_normal() -> None:
    constraint = _constraint(np.array([0.25, 0.25, -0.1]))

    force, _ = penalty_contact_response(
        [constraint],
        stiffness=100.0,
        n_total_dofs=12,
        slave_dof_offset=0,
        master_dof_offset=3,
    )

    slave_force = force[:3]
    expected_lambda = 100.0 * 0.1

    assert constraint.g < 0.0
    assert np.allclose(slave_force, expected_lambda * constraint.n)


def test_action_reaction_balance_between_slave_and_master() -> None:
    constraint = _constraint(np.array([0.25, 0.25, -0.1]))

    force, _ = penalty_contact_response(
        [constraint],
        stiffness=100.0,
        n_total_dofs=12,
        slave_dof_offset=0,
        master_dof_offset=3,
    )

    slave_resultant = force[:3]
    master_resultant = force[3:].reshape(3, 3).sum(axis=0)

    assert np.allclose(slave_resultant + master_resultant, 0.0)


def test_contact_gap_jacobian_matches_finite_difference() -> None:
    slave = np.array([[0.25, 0.25, -0.1]])
    master = MASTER_TRIANGLE.copy()
    constraint = contact_constraint_from_sample(slave, SAMPLE, master, MASTER_FACES)
    J = contact_jacobian_row(
        constraint,
        n_total_dofs=12,
        slave_dof_offset=0,
        master_dof_offset=3,
    ).toarray()[0]

    q0 = np.concatenate([slave.ravel(), master.ravel()])

    def gap_from_q(q: np.ndarray) -> float:
        slave_q = q[:3].reshape(1, 3)
        master_q = q[3:].reshape(3, 3)
        return contact_constraint_from_sample(slave_q, SAMPLE, master_q, MASTER_FACES).g

    eps = 1e-6
    grad_fd = np.zeros_like(q0)
    for i in range(q0.size):
        dq = np.zeros_like(q0)
        dq[i] = eps
        grad_fd[i] = (gap_from_q(q0 + dq) - gap_from_q(q0 - dq)) / (2.0 * eps)

    assert np.allclose(grad_fd, J, atol=1e-6)


def _finite_difference_gap_jacobian(
    slave: np.ndarray,
    sample: SurfaceSample,
    master: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    constraint = contact_constraint_from_sample(slave, sample, master, faces)
    slave_dofs = slave.size
    n_total_dofs = slave_dofs + master.size
    J = contact_jacobian_row(
        constraint,
        n_total_dofs=n_total_dofs,
        slave_dof_offset=0,
        master_dof_offset=slave_dofs,
    ).toarray()[0]
    q0 = np.concatenate([slave.ravel(), master.ravel()])

    def gap_from_q(q: np.ndarray) -> float:
        slave_q = q[:slave_dofs].reshape(slave.shape)
        master_q = q[slave_dofs:].reshape(master.shape)
        return contact_constraint_from_sample(slave_q, sample, master_q, faces).g

    eps = 1e-6
    grad_fd = np.zeros_like(q0)
    for i in range(q0.size):
        dq = np.zeros_like(q0)
        dq[i] = eps
        grad_fd[i] = (gap_from_q(q0 + dq) - gap_from_q(q0 - dq)) / (2.0 * eps)
    return J, grad_fd


def test_contact_gap_jacobian_matches_finite_difference_for_rotated_triangle() -> None:
    master = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.3],
            [0.1, 0.9, 0.4],
        ]
    )
    face = np.array([[0, 1, 2]], dtype=np.int64)
    normal = np.cross(master[1] - master[0], master[2] - master[0])
    normal /= np.linalg.norm(normal)
    projected = 0.2 * master[0] + 0.3 * master[1] + 0.5 * master[2]
    slave = (projected - 0.1 * normal).reshape(1, 3)
    sample = SurfaceSample(
        node_ids=np.array([0], dtype=np.int64),
        weights=np.array([1.0]),
        candidate_face_ids=np.array([0], dtype=np.int64),
    )

    J, grad_fd = _finite_difference_gap_jacobian(slave, sample, master, face)

    assert np.allclose(grad_fd, J, atol=1e-6)


def test_contact_gap_jacobian_matches_finite_difference_for_multi_node_slave_sample() -> None:
    master = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.3],
            [0.1, 0.9, 0.4],
        ]
    )
    face = np.array([[0, 1, 2]], dtype=np.int64)
    normal = np.cross(master[1] - master[0], master[2] - master[0])
    normal /= np.linalg.norm(normal)
    projected = 0.25 * master[0] + 0.25 * master[1] + 0.5 * master[2]
    sample_point = projected - 0.08 * normal
    tangent = master[1] - master[0]
    tangent /= np.linalg.norm(tangent)
    slave = np.vstack((sample_point - 0.03 * tangent, sample_point + 0.03 * tangent))
    sample = SurfaceSample(
        node_ids=np.array([0, 1], dtype=np.int64),
        weights=np.array([0.5, 0.5]),
        candidate_face_ids=np.array([0], dtype=np.int64),
    )

    J, grad_fd = _finite_difference_gap_jacobian(slave, sample, master, face)

    assert np.allclose(grad_fd, J, atol=1e-6)


def test_contact_force_remains_repulsive_after_reversed_master_tet_input() -> None:
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    master = VolumeMesh(X=X, elements=np.array([[0, 2, 1, 3]], dtype=np.int64))
    faces, _ = extract_boundary_faces(master.elements, master.X)
    bottom_face_id = int(
        np.flatnonzero(
            [
                set(map(int, face.tolist())) == {0, 1, 2}
                for face in faces
            ]
        )[0]
    )
    slave = np.array([[0.25, 0.25, 0.05]])
    sample = SurfaceSample(
        node_ids=np.array([0], dtype=np.int64),
        weights=np.array([1.0]),
        candidate_face_ids=np.array([bottom_face_id], dtype=np.int64),
    )
    constraint = contact_constraint_from_sample(slave, sample, master.X, faces)

    force, _ = penalty_contact_response(
        [constraint],
        stiffness=100.0,
        n_total_dofs=3 + 3 * master.X.shape[0],
        slave_dof_offset=0,
        master_dof_offset=3,
    )

    assert constraint.g < 0.0
    assert force[2] < 0.0
