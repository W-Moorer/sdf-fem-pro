from __future__ import annotations

import numpy as np

from sfc.fem.calculix_aligned import (
    ContactSample,
    MechanicsModel,
    assemble_contact_response,
    hht_step,
    initial_state,
    stvk_internal_response,
)
from sfc.fem.hex8 import C3D8_NATURAL_NODE_COORDS, hex8_volume


def _unit_hex() -> np.ndarray:
    return 0.5 * (C3D8_NATURAL_NODE_COORDS + 1.0)


class _NoContact:
    def samples(self, x_current: np.ndarray):
        _ = x_current
        return ()


def test_hex8_nonlinear_model_volume_and_mass_are_conserved() -> None:
    X = _unit_hex()
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    model = MechanicsModel.from_hex8_mesh(X, elements, E=1000.0, nu=0.25, density=2.5)

    assert model.element_type == "c3d8"
    assert np.isclose(model.volumes[0], hex8_volume(X))
    assert np.isclose(np.sum(model.mass_matrix[0::3, 0::3]), 2.5 * hex8_volume(X))


def test_hex8_stvk_affine_uniaxial_stress_is_positive() -> None:
    X = _unit_hex()
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    model = MechanicsModel.from_hex8_mesh(X, elements, E=1000.0, nu=0.25, density=1.0)
    x = X.copy()
    x[:, 0] = 1.02 * X[:, 0]

    response = stvk_internal_response(model, x, assemble_tangent=True)

    assert response.strain_energy > 0.0
    assert response.von_mises[0] > 0.0
    assert response.material_tangent.shape == (24, 24)
    assert response.geometric_tangent.shape == (24, 24)


def test_hex8_hht_zero_load_no_motion() -> None:
    X = _unit_hex()
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    model = MechanicsModel.from_hex8_mesh(X, elements, E=1000.0, nu=0.25, density=1.0)
    contact = _NoContact()
    state, previous = initial_state(model, contact, gravity=0.0, dt=0.01, alpha=0.0)

    next_state, _next_previous, diagnostics = hht_step(
        model,
        state,
        previous,
        contact,
        dt=0.01,
        gravity=0.0,
        alpha=0.0,
        max_iterations=4,
    )

    assert np.allclose(next_state.x, X)
    assert np.allclose(next_state.v, 0.0)
    assert np.allclose(next_state.a, 0.0)
    assert diagnostics.contact.active_count == 0


def test_contact_response_applies_master_slave_action_reaction() -> None:
    sample = ContactSample(
        node_ids=np.array([0], dtype=np.int64),
        shape_weights=np.array([1.0]),
        master_node_ids=np.array([1], dtype=np.int64),
        master_shape_weights=np.array([1.0]),
        gap=-0.01,
        normal=np.array([0.0, 0.0, 1.0]),
        area=2.0,
        stiffness=100.0,
    )

    response = assemble_contact_response([sample], n_nodes=2)
    force = response.force.reshape((2, 3))

    assert np.allclose(force[0], np.array([0.0, 0.0, 2.0]))
    assert np.allclose(force[1], np.array([0.0, 0.0, -2.0]))
    assert np.allclose(np.sum(force, axis=0), 0.0)
