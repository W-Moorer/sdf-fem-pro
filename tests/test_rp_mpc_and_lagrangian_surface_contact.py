from __future__ import annotations

import numpy as np
import pytest

from sfc.contact.lagrangian_surface_contact import LagrangianSDFSurfaceContactGeometry
from sfc.fem.calculix_aligned import MechanicsModel, assemble_contact_response
from sfc.fem.implicit_dirichlet import hht_step_dirichlet, initial_state_dirichlet
from scipy.sparse import diags

from sfc.fem.rp_mpc import (
    FiniteRotationRigidHubMPC,
    RigidHubMPC,
    constant_torque_rotation_history,
    merge_dirichlet_conditions,
    reduced_hub_rotational_inertia,
    rotation_matrix_from_vector,
)
from sfc.sdf.material_sdf import MaterialSDF


class EmptyContact:
    def samples(self, x_current: np.ndarray):
        return []


def test_rigid_hub_mpc_matches_z_axis_cross_product() -> None:
    nodes = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    hub = RigidHubMPC(np.asarray([0, 1], dtype=np.int64), nodes, np.zeros(3))

    disp = hub.nodal_displacements(rotation=(0.0, 0.0, 0.1))

    assert disp[0] == pytest.approx([0.0, 0.1, 0.0])
    assert disp[1] == pytest.approx([-0.2, 0.0, 0.0])


def test_finite_rotation_hub_mpc_rotates_reference_lever_exactly() -> None:
    nodes = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    hub = FiniteRotationRigidHubMPC(np.asarray([0, 1], dtype=np.int64), nodes, np.zeros(3))

    current = hub.current_positions(rotation=(0.0, 0.0, 0.5 * np.pi))

    assert current[0] == pytest.approx([0.0, 1.0, 0.0], abs=1.0e-14)
    assert current[1] == pytest.approx([-2.0, 0.0, 0.0], abs=1.0e-14)


def test_finite_rotation_matches_small_rotation_to_first_order() -> None:
    nodes = np.asarray([[1.0, -0.2, 0.0], [0.0, 2.0, 0.3]], dtype=float)
    ids = np.asarray([0, 1], dtype=np.int64)
    theta = np.asarray([2.0e-7, -3.0e-7, 5.0e-7], dtype=float)
    small = RigidHubMPC(ids, nodes, np.zeros(3)).nodal_displacements(rotation=theta)
    finite = FiniteRotationRigidHubMPC(ids, nodes, np.zeros(3)).nodal_displacements(rotation=theta)

    assert finite == pytest.approx(small, abs=1.0e-12)


def test_rotation_matrix_from_vector_is_orthonormal() -> None:
    R = rotation_matrix_from_vector((0.2, -0.1, 0.3))

    assert R.T @ R == pytest.approx(np.eye(3), abs=1.0e-14)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_reduced_hub_rotational_inertia_matches_point_mass_formula() -> None:
    nodes = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)
    masses = np.repeat(np.asarray([2.0, 3.0], dtype=float), 3)
    inertia = reduced_hub_rotational_inertia(np.asarray([0, 1]), nodes, np.zeros(3), diags(masses))

    assert inertia[2, 2] == pytest.approx(2.0 * 1.0**2 + 3.0 * 2.0**2)
    assert inertia[0, 0] == pytest.approx(3.0 * 2.0**2)
    assert inertia[1, 1] == pytest.approx(2.0 * 1.0**2)


def test_constant_torque_rotation_history_integrates_free_rp_dof() -> None:
    times, rotations, velocities = constant_torque_rotation_history(
        np.diag([2.0, 3.0, 4.0]),
        (0.0, 0.0, 8.0),
        duration=0.1,
        dt=0.05,
    )

    assert times == pytest.approx([0.0, 0.05, 0.1])
    assert rotations[:, 2] == pytest.approx([0.0, 0.0025, 0.01])
    assert velocities[:, 2] == pytest.approx([0.0, 0.1, 0.2])


def test_constant_torque_rotation_history_respects_active_axes() -> None:
    inertia = np.asarray([[2.0, 0.0, 0.5], [0.0, 3.0, 0.0], [0.5, 0.0, 4.0]], dtype=float)
    _times, rotations, velocities = constant_torque_rotation_history(
        inertia,
        (0.0, 0.0, 8.0),
        duration=0.1,
        dt=0.1,
        active_axes=(2,),
    )

    assert rotations[-1] == pytest.approx([0.0, 0.0, 0.01])
    assert velocities[-1] == pytest.approx([0.0, 0.0, 0.2])


def test_merge_dirichlet_conditions_rejects_conflicts() -> None:
    dofs, values = merge_dirichlet_conditions(
        (np.asarray([0, 2]), np.asarray([1.0, 3.0])),
        (np.asarray([2, 5]), np.asarray([3.0, 6.0])),
    )
    assert dofs == pytest.approx([0, 2, 5])
    assert values == pytest.approx([1.0, 3.0, 6.0])
    with pytest.raises(ValueError, match="conflicting"):
        merge_dirichlet_conditions((np.asarray([0]), np.asarray([1.0])), (np.asarray([0]), np.asarray([2.0])))


def test_hht_step_dirichlet_enforces_prescribed_displacement() -> None:
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
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1000.0, nu=0.3, density=1.0)
    fixed = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    initial_values = np.zeros_like(fixed, dtype=float)
    contact = EmptyContact()
    state, previous = initial_state_dirichlet(model, contact, gravity=0.0, fixed_dofs=fixed, fixed_values=initial_values)

    next_values = np.asarray([0.01, 0.0, 0.0, 0.01, 0.0, 0.0], dtype=float)
    next_state, _previous, diagnostics = hht_step_dirichlet(
        model,
        state,
        previous,
        contact,
        dt=1.0e-3,
        gravity=0.0,
        fixed_dofs=fixed,
        fixed_values=next_values,
        alpha=0.0,
    )

    assert next_state.x.reshape(-1)[fixed] - model.X.reshape(-1)[fixed] == pytest.approx(next_values)
    assert diagnostics.newton_iterations >= 1


def test_lagrangian_sdf_surface_contact_provides_master_payload_tangent() -> None:
    slave_nodes = np.asarray([[0.0, 0.0, -0.05], [1.0, 0.0, -0.05], [0.0, 1.0, -0.05]], dtype=float)
    master_nodes = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    x_current = np.vstack([slave_nodes, master_nodes])
    slave_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    master_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    material = MaterialSDF.from_triangle_surface(master_nodes, master_faces)
    contact = LagrangianSDFSurfaceContactGeometry(
        slave_faces,
        material,
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=3,
        quadrature="centroid",
    )

    samples = list(contact.samples(x_current))
    response = assemble_contact_response(samples, n_nodes=6)

    assert len(samples) == 1
    assert samples[0].gap < 0.0
    assert samples[0].master_node_ids is not None
    assert np.sum(response.force[:3, 2]) > 0.0
    assert np.sum(response.force[3:, 2]) < 0.0
    assert response.tangent.nnz > 0
