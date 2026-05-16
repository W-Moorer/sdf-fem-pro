import numpy as np

from sfc.analysis import (
    LinearNewmarkAnalysis,
    LinearStaticAnalysis,
    available_analysis_backends,
    get_analysis_backend,
)
from sfc.fem import (
    DeformableBody,
    assemble_gravity_force,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    fixed_dofs_from_node_set,
    newmark_beta_step,
)
from sfc.fem.constraints import free_dofs
from sfc.mesh import VolumeMesh


def _single_tet_body() -> DeformableBody:
    mesh = VolumeMesh(
        X=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        elements=np.array([[0, 1, 2, 3]]),
        element_type="C3D4",
    )
    return DeformableBody(mesh=mesh, material={"E": 1000.0, "nu": 0.25}, density=2.0)


def test_linear_analysis_backends_are_registered() -> None:
    assert available_analysis_backends() == ("linear_static", "linear_newmark")
    assert get_analysis_backend("linear_static").name == "linear_static"
    assert get_analysis_backend("linear_newmark").name == "linear_newmark"


def test_linear_static_backend_solves_free_dof_equilibrium() -> None:
    body = _single_tet_body()
    fixed = fixed_dofs_from_node_set([0, 1, 2], "xyz")
    force = np.zeros(body.n_dofs)
    force[3 * 3 + 2] = -1.0

    result = LinearStaticAnalysis().solve(body, force=force, fixed_dofs=fixed)

    residual = result.stiffness @ result.u - result.force
    free = free_dofs(body.n_dofs, fixed)
    assert np.allclose(result.u[fixed], 0.0)
    assert np.allclose(residual[free], 0.0, atol=1.0e-10)
    assert np.allclose(result.free_dofs, free)


def test_linear_newmark_backend_matches_existing_integrator() -> None:
    body = _single_tet_body()
    fixed = fixed_dofs_from_node_set([0], "xyz")
    gravity = np.array([0.0, 0.0, -9.81])
    backend = LinearNewmarkAnalysis(beta=0.25, gamma=0.5)

    result = backend.step(body, dt=1.0e-3, gravity=gravity, fixed_dofs=fixed)

    M = assemble_mass_matrix(body)
    K = assemble_stiffness_matrix(body)
    C = np.zeros((body.n_dofs, body.n_dofs))
    f = assemble_gravity_force(body, gravity)
    expected = newmark_beta_step(M, C, K, body.u, body.v, body.a, f, dt=1.0e-3, fixed_dofs=fixed)

    assert np.allclose(result.u, expected[0])
    assert np.allclose(result.v, expected[1])
    assert np.allclose(result.a, expected[2])


def test_linear_newmark_backend_can_update_body_state() -> None:
    body = _single_tet_body()
    result = LinearNewmarkAnalysis().step(body, dt=1.0e-3, force=np.ones(body.n_dofs), update_body=True)

    assert np.allclose(body.u, result.u)
    assert np.allclose(body.v, result.v)
    assert np.allclose(body.a, result.a)
