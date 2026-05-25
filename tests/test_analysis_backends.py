import numpy as np

from sfc.analysis import (
    LinearNewmarkAnalysis,
    LinearStaticAnalysis,
    NonlinearHHTAnalysis,
    NonlinearStaticNewtonAnalysis,
    available_analysis_backends,
    get_analysis_backend,
)
from sfc.fem import (
    DeformableBody,
    MechanicsModel,
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


def _single_tet_mechanics_model() -> MechanicsModel:
    body = _single_tet_body()
    return MechanicsModel.from_tet4_mesh(
        body.mesh.X,
        body.mesh.elements,
        E=1000.0,
        nu=0.25,
        density=2.0,
    )


def test_linear_analysis_backends_are_registered() -> None:
    assert set(available_analysis_backends()) >= {
        "linear_static",
        "linear_newmark",
        "nonlinear_static_newton",
        "nonlinear_hht",
    }
    assert get_analysis_backend("linear_static").name == "linear_static"
    assert get_analysis_backend("linear_newmark").name == "linear_newmark"
    assert get_analysis_backend("nonlinear_static_newton").name == "nonlinear_static_newton"
    assert get_analysis_backend("nonlinear_hht").name == "nonlinear_hht"


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


def test_nonlinear_static_backend_keeps_fixed_reference_state() -> None:
    model = _single_tet_mechanics_model()
    fixed = np.arange(model.n_dofs, dtype=np.int64)

    result = NonlinearStaticNewtonAnalysis().solve(model, fixed_dofs=fixed)

    assert result.converged
    assert result.iterations == 0
    assert np.allclose(result.x, model.X)
    assert np.allclose(result.u, 0.0)
    assert result.free_dofs.size == 0


def test_nonlinear_hht_backend_zero_load_has_no_motion() -> None:
    model = _single_tet_mechanics_model()
    backend = NonlinearHHTAnalysis(alpha=0.0)
    state, previous = backend.initial_state(model, gravity=0.0, dt=1.0e-3)

    result = backend.step(model, state, previous, dt=1.0e-3, gravity=0.0)

    assert np.allclose(result.state.x, model.X)
    assert np.allclose(result.state.v, 0.0)
    assert np.allclose(result.state.a, 0.0)
    assert np.allclose(result.previous_static_residual, 0.0, atol=1.0e-10)


def test_nonlinear_hht_backend_exposes_adaptive_increment_counts() -> None:
    model = _single_tet_mechanics_model()
    backend = NonlinearHHTAnalysis(
        alpha=0.0,
        adaptive_increments=True,
        acceptance_policy="relative_correction",
        min_dt=2.5e-4,
    )
    state, previous = backend.initial_state(model, gravity=0.0, dt=1.0e-3)

    result = backend.step(model, state, previous, dt=1.0e-3, gravity=0.0)

    assert np.allclose(result.state.x, model.X)
    assert result.accepted_increment_count >= 1
    assert result.cutback_count == 0
    assert result.attempted_increment_count >= result.accepted_increment_count
    assert result.min_accepted_increment > 0.0
    assert result.max_accepted_increment >= result.min_accepted_increment
