import numpy as np

from sfc.fem import (
    DeformableBody,
    assemble_gravity_force,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    fixed_dofs_from_node_set,
    newmark_beta_step,
)
from sfc.mesh import VolumeMesh


def _two_tet_body() -> DeformableBody:
    mesh = VolumeMesh(
        X=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        ),
        elements=np.array(
            [
                [0, 1, 2, 3],
                [0, 2, 1, 4],
            ],
            dtype=np.int64,
        ),
        node_sets={"fixed": np.array([0], dtype=np.int64)},
    )
    return DeformableBody(mesh=mesh, material={"E": 1000.0, "nu": 0.25}, density=2.5)


def test_fixed_dofs_remain_fixed() -> None:
    body = _two_tet_body()
    fixed = fixed_dofs_from_node_set(body.mesh.node_sets["fixed"], "xyz")
    body.u[fixed] = np.array([0.1, -0.2, 0.3])
    body.v[fixed] = np.array([1.0, 2.0, 3.0])
    body.a[fixed] = np.array([-1.0, -2.0, -3.0])

    M = assemble_mass_matrix(body)
    K = assemble_stiffness_matrix(body)
    f = assemble_gravity_force(body)

    u_new, v_new, a_new = newmark_beta_step(
        M,
        None,
        K,
        body.u,
        body.v,
        body.a,
        f,
        dt=0.01,
        fixed_dofs=fixed,
    )

    assert np.allclose(u_new[fixed], [0.1, -0.2, 0.3])
    assert np.allclose(v_new[fixed], 0.0)
    assert np.allclose(a_new[fixed], 0.0)


def test_zero_force_and_zero_initial_velocity_produces_no_motion() -> None:
    body = _two_tet_body()
    M = assemble_mass_matrix(body)
    K = assemble_stiffness_matrix(body)
    zero = np.zeros(body.n_dofs)

    u_new, v_new, a_new = newmark_beta_step(
        M,
        None,
        K,
        body.u,
        body.v,
        body.a,
        zero,
        dt=0.01,
    )

    assert np.allclose(u_new, 0.0)
    assert np.allclose(v_new, 0.0)
    assert np.allclose(a_new, 0.0)


def test_gravity_causes_downward_displacement_in_unconstrained_nodes() -> None:
    body = _two_tet_body()
    M = assemble_mass_matrix(body)
    K = assemble_stiffness_matrix(body)
    gravity = np.array([0.0, 0.0, -9.81])
    f = assemble_gravity_force(body, gravity)

    u_new, _, _ = newmark_beta_step(
        M,
        None,
        K,
        body.u,
        body.v,
        body.a,
        f,
        dt=0.01,
    )

    assert np.all(u_new[2::3] < 0.0)
