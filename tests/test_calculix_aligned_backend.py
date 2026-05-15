from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from sfc.fem.calculix_aligned import (
    ContactSample,
    MechanicsModel,
    PlaneContactGeometry,
    assemble_contact_response,
    hht_newmark_parameters,
    hht_step,
    initial_state,
    static_force_state,
    static_residual_and_tangent,
    stvk_internal_response,
)


class EmptyContactGeometry:
    def samples(self, x_current: np.ndarray) -> Iterable[ContactSample]:
        return []


def _unit_tet_model() -> MechanicsModel:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    return MechanicsModel.from_tet4_mesh(X, elements, E=1000.0, nu=0.3, density=2.0)


def test_hht_parameters_match_calculix_alpha_convention() -> None:
    beta, gamma = hht_newmark_parameters(-0.05)

    assert beta == pytest.approx(0.25 * 1.05**2)
    assert gamma == pytest.approx(0.55)


def test_consistent_mass_has_correct_total_translational_mass() -> None:
    model = _unit_tet_model()
    total_mass = model.density * np.sum(model.volumes)
    ones_x = np.zeros(model.n_dofs)
    ones_x[0::3] = 1.0

    assert ones_x @ (model.mass_matrix @ ones_x) == pytest.approx(total_mass)


def test_stvk_tangent_matches_directional_finite_difference() -> None:
    model = _unit_tet_model()
    x = model.X.copy()
    x[:, 0] += 0.04 * model.X[:, 2]
    x[:, 2] += 0.02 * model.X[:, 0]
    direction = np.linspace(-0.2, 0.3, model.n_dofs).reshape((-1, 3))
    direction /= np.linalg.norm(direction)

    response = stvk_internal_response(model, x, assemble_tangent=True)
    h = 1.0e-6
    plus = stvk_internal_response(model, x + h * direction, assemble_tangent=False).force.reshape(-1)
    minus = stvk_internal_response(model, x - h * direction, assemble_tangent=False).force.reshape(-1)
    finite_difference = (plus - minus) / (2.0 * h)

    assert response.tangent @ direction.reshape(-1) == pytest.approx(finite_difference, rel=1.0e-5, abs=1.0e-6)
    assert np.linalg.norm(response.material_tangent.data) > 0.0
    assert np.linalg.norm(response.geometric_tangent.data) > 0.0


def test_plane_contact_response_uses_area_weighted_pressure_overclosure() -> None:
    x = np.asarray(
        [
            [0.0, 0.0, -0.1],
            [1.0, 0.0, -0.1],
            [0.0, 1.0, -0.1],
        ],
        dtype=float,
    )
    geometry = PlaneContactGeometry(np.asarray([[0, 1, 2]], dtype=np.int64), plane_z=0.0, stiffness=100.0)
    response = assemble_contact_response(geometry.samples(x), n_nodes=3)

    assert response.max_penetration == pytest.approx(0.1)
    assert response.normal_force == pytest.approx(100.0 * 0.5 * 0.1)
    assert np.sum(response.force[:, 2]) == pytest.approx(response.normal_force)
    assert response.energy == pytest.approx(0.5 * 100.0 * 0.5 * 0.1**2)
    assert response.tangent.nnz > 0


def test_hht_step_keeps_contact_geometry_swappable() -> None:
    model = _unit_tet_model()
    state, previous = initial_state(model, EmptyContactGeometry(), gravity=9.81)
    next_state, next_previous, diagnostics = hht_step(
        model,
        state,
        previous,
        EmptyContactGeometry(),
        dt=0.001,
        gravity=9.81,
        alpha=0.0,
    )

    assert np.mean(next_state.x[:, 2]) < np.mean(state.x[:, 2])
    assert next_previous.shape == previous.shape
    assert diagnostics.newton_iterations > 0
    assert diagnostics.newton_acceptance_policy == "relative_correction"
    assert diagnostics.newton_acceptance_reason
    assert diagnostics.newton_acceptance_metrics
    last = diagnostics.newton_acceptance_metrics[-1]
    assert last.qam > 0.0
    assert last.uam >= 0.0
    assert last.reason
    residual, _ = static_residual_and_tangent(model, next_state.x, EmptyContactGeometry(), gravity=9.81)
    assert residual.shape == previous.shape


def test_initial_state_returns_calculix_fextini_minus_fini_history() -> None:
    model = _unit_tet_model()
    contact = EmptyContactGeometry()

    state, previous = initial_state(model, contact, gravity=9.81)
    static_state = static_force_state(model, state.x, contact, gravity=9.81)

    assert previous == pytest.approx(static_state.calculix_rhs_balance)
    assert previous == pytest.approx(-static_state.residual)


def test_hht_step_saves_accepted_contact_internal_force_history() -> None:
    model = _unit_tet_model()
    contact = PlaneContactGeometry(np.asarray([[0, 2, 1]], dtype=np.int64), plane_z=0.05, stiffness=100.0)
    state, previous = initial_state(model, contact, gravity=9.81)

    next_state, next_previous, diagnostics = hht_step(
        model,
        state,
        previous,
        contact,
        dt=0.001,
        gravity=9.81,
        alpha=-0.05,
    )
    static_state = static_force_state(model, next_state.x, contact, gravity=9.81)

    assert diagnostics.contact.normal_force > 0.0
    assert np.linalg.norm(static_state.contact_internal_force) > 0.0
    assert next_previous == pytest.approx(static_state.calculix_rhs_balance)
    assert next_previous == pytest.approx(-static_state.residual)


def test_hht_step_supports_clean_room_calculix_multicriteria_acceptance() -> None:
    model = _unit_tet_model()
    state, previous = initial_state(model, EmptyContactGeometry(), gravity=9.81)
    _, _, diagnostics = hht_step(
        model,
        state,
        previous,
        EmptyContactGeometry(),
        dt=0.001,
        gravity=9.81,
        alpha=-0.05,
        max_iterations=8,
        acceptance_policy="calculix_multicriteria",
    )

    assert diagnostics.newton_acceptance_policy == "calculix_multicriteria"
    assert diagnostics.newton_acceptance_metrics
    assert diagnostics.newton_iterations <= 8
    assert diagnostics.newton_acceptance_metrics[-1].ram >= 0.0
    assert diagnostics.newton_acceptance_metrics[-1].cam >= 0.0
