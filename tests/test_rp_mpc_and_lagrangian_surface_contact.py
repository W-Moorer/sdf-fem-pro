from __future__ import annotations

import numpy as np
import pytest

from sfc.contact.constraint_region import aggregate_contact_sample_arrays
from sfc.contact.lagrangian_surface_contact import (
    LagrangianQ4ClosestFeatureOracle,
    LagrangianSDFQ4MasterSurfaceContactGeometry,
    LagrangianSDFQuadrilateralSurfaceContactGeometry,
    LagrangianSDFSurfaceContactGeometry,
)
from sfc.fem.calculix_aligned import MechanicsModel, assemble_contact_response
from sfc.fem.implicit_dirichlet import hht_step_dirichlet, initial_state_dirichlet
from scipy.sparse import diags

from sfc.fem.rp_mpc import (
    FiniteRotationRigidHubMPC,
    RigidHubMPC,
    build_rigid_hub_reduced_assembly,
    constant_angular_velocity_rotation_history,
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


def test_constant_angular_velocity_rotation_history_prescribes_rp_motion() -> None:
    times, rotations, velocities = constant_angular_velocity_rotation_history(
        (0.0, 0.0, 3.0),
        duration=0.2,
        dt=0.1,
        initial_rotation=(0.0, 0.0, 0.5),
    )

    assert times == pytest.approx([0.0, 0.1, 0.2])
    assert rotations[:, 2] == pytest.approx([0.5, 0.8, 1.1])
    assert velocities[:, 2] == pytest.approx([3.0, 3.0, 3.0])


def test_rigid_hub_reduced_assembly_expands_hub_and_free_nodes() -> None:
    nodes = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.3, 0.4, 0.5]], dtype=float)
    hub = RigidHubMPC(np.asarray([0, 1], dtype=np.int64), nodes, np.zeros(3))
    assembly = build_rigid_hub_reduced_assembly(nodes, [hub])
    q = np.zeros(assembly.n_reduced_dofs, dtype=float)
    q[:3] = [0.1, 0.2, 0.3]
    q[assembly.hub_slice(0)] = [0.01, -0.02, 0.03, 0.0, 0.0, 0.1]

    expanded = assembly.expand_displacements(q)
    expected_hub = hub.nodal_displacements(translation=(0.01, -0.02, 0.03), rotation=(0.0, 0.0, 0.1))

    assert assembly.free_node_ids == pytest.approx([2])
    assert expanded[0:2] == pytest.approx(expected_hub)
    assert expanded[2] == pytest.approx([0.1, 0.2, 0.3])


def test_rigid_hub_reduced_assembly_projects_mass_and_forces_to_rp_dofs() -> None:
    nodes = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)
    masses = np.repeat(np.asarray([2.0, 3.0], dtype=float), 3)
    hub = RigidHubMPC(np.asarray([0, 1], dtype=np.int64), nodes, np.zeros(3))
    assembly = build_rigid_hub_reduced_assembly(nodes, [hub], include_free_nodes=False)
    reduced_mass = assembly.reduce_matrix(diags(masses)).toarray()
    inertia = reduced_hub_rotational_inertia(np.asarray([0, 1]), nodes, np.zeros(3), diags(masses))
    nodal_force = np.asarray([[0.0, 2.0, 0.0], [-3.0, 0.0, 0.0]], dtype=float)
    reduced_force = assembly.reduce_vector(nodal_force)

    assert reduced_mass[:3, :3] == pytest.approx(5.0 * np.eye(3))
    assert reduced_mass[3:, 3:] == pytest.approx(inertia)
    assert reduced_force[:3] == pytest.approx([-3.0, 2.0, 0.0])
    assert reduced_force[5] == pytest.approx(1.0 * 2.0 + 2.0 * 3.0)


def test_rigid_hub_generalized_force_matches_virtual_work_projection() -> None:
    nodes = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=float)
    hub = RigidHubMPC(np.asarray([0, 1], dtype=np.int64), nodes, np.zeros(3))
    full_force = np.asarray([[0.0, 2.0, 0.0], [-3.0, 0.0, 1.0], [9.0, 9.0, 9.0]], dtype=float)
    generalized = hub.generalized_force_from_nodal_forces(full_force)

    assert generalized[:3] == pytest.approx([-3.0, 2.0, 1.0])
    assert generalized[3:] == pytest.approx([2.0, 0.0, 8.0])
    virtual_rp = np.asarray([0.2, -0.1, 0.3, 0.0, 0.0, 0.4], dtype=float)
    virtual_nodes = hub.nodal_displacements(translation=virtual_rp[:3], rotation=virtual_rp[3:])

    assert float(generalized @ virtual_rp) == pytest.approx(float(np.sum(full_force[hub.node_ids] * virtual_nodes)))


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


def test_lagrangian_sdf_surface_contact_compiled_batch_matches_scalar_oracle() -> None:
    slave_nodes = np.asarray([[0.0, 0.0, -0.05], [1.0, 0.0, -0.05], [0.0, 1.0, -0.05]], dtype=float)
    master_nodes = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    x_current = np.vstack([slave_nodes, master_nodes])
    slave_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    master_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    material = MaterialSDF.from_triangle_surface(master_nodes, master_faces)
    scalar = LagrangianSDFSurfaceContactGeometry(
        slave_faces,
        material,
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=3,
        quadrature="tri3",
    )
    compiled = LagrangianSDFSurfaceContactGeometry(
        slave_faces,
        material,
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=3,
        quadrature="tri3",
        compiled_batch_projection=True,
    )

    scalar_samples = list(scalar.samples(x_current))
    compiled_samples = list(compiled.samples(x_current))

    assert len(compiled_samples) == len(scalar_samples)
    for got, expected in zip(compiled_samples, scalar_samples, strict=True):
        assert got.gap == pytest.approx(expected.gap)
        np.testing.assert_allclose(got.normal, expected.normal, atol=1.0e-14)
        np.testing.assert_array_equal(got.master_node_ids, expected.master_node_ids)
        np.testing.assert_allclose(got.master_shape_weights, expected.master_shape_weights, atol=1.0e-14)


def test_lagrangian_sdf_surface_contact_vectorized_sampling_matches_scalar_order() -> None:
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, -0.05],
            [1.0, 0.0, -0.05],
            [0.0, 1.0, -0.05],
            [1.0, 1.0, -0.05],
        ],
        dtype=float,
    )
    master_nodes = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    x_current = np.vstack([slave_nodes, master_nodes])
    slave_faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    master_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    material = MaterialSDF.from_triangle_surface(master_nodes, master_faces)
    scalar = LagrangianSDFSurfaceContactGeometry(
        slave_faces,
        material,
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=4,
        quadrature="tri3",
    )
    compiled = LagrangianSDFSurfaceContactGeometry(
        slave_faces,
        material,
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=4,
        quadrature="tri3",
        compiled_batch_projection=True,
    )

    arrays = compiled.sample_arrays(x_current)
    scalar_samples = list(scalar.samples(x_current))

    if arrays is None:
        pytest.skip("compiled projection backend is not available")
    assert arrays is not None
    assert arrays["sample_node_ids"].shape == (6, 3)
    for index, sample in enumerate(scalar_samples):
        np.testing.assert_array_equal(arrays["sample_node_ids"][index], sample.node_ids)
        np.testing.assert_allclose(arrays["sample_weights"][index], sample.shape_weights)
        assert float(arrays["areas"][index]) == pytest.approx(sample.area)


def test_lagrangian_sdf_surface_contact_skips_faces_outside_conservative_tube() -> None:
    slave_nodes = np.asarray([[10.0, 0.0, -0.05], [11.0, 0.0, -0.05], [10.0, 1.0, -0.05]], dtype=float)
    master_nodes = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    x_current = np.vstack([slave_nodes, master_nodes])
    material = MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64))
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        material,
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=3,
        quadrature="tri3",
        search_radius=0.1,
    )

    assert list(contact.samples(x_current)) == []


def test_lagrangian_sdf_surface_contact_compiled_culls_out_of_tube_quadrature_points() -> None:
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, -0.02],
            [1.2, 0.0, -0.02],
            [0.0, 1.2, -0.02],
        ],
        dtype=float,
    )
    master_nodes = np.asarray([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0]], dtype=float)
    x_current = np.vstack([slave_nodes, master_nodes])
    material = MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64))
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        material,
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=3,
        quadrature="tri3",
        search_radius=0.25,
        compiled_batch_projection=True,
    )

    arrays = contact.sample_arrays(x_current)

    if arrays is None:
        pytest.skip("compiled projection backend is not available")
    assert 0 < arrays["gaps"].shape[0] < 3
    assert arrays["sample_node_ids"].shape[0] == arrays["gaps"].shape[0]
    assert arrays["areas"].shape[0] == arrays["gaps"].shape[0]


def test_lagrangian_sdf_quadrilateral_surface_contact_uses_q4_weights() -> None:
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, -0.05],
            [1.0, 0.0, -0.05],
            [1.0, 1.0, -0.05],
            [0.0, 1.0, -0.05],
        ],
        dtype=float,
    )
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    x_current = np.vstack([slave_nodes, master_nodes])
    material = MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64))
    contact = LagrangianSDFQuadrilateralSurfaceContactGeometry(
        np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        material,
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=4,
        quadrature_order=2,
    )

    samples = list(contact.samples(x_current))

    assert len(samples) == 4
    assert sum(sample.area for sample in samples) == pytest.approx(1.0)
    assert all(sample.node_ids.shape == (4,) for sample in samples)
    assert all(sample.shape_weights.sum() == pytest.approx(1.0) for sample in samples)
    assert all(sample.gap == pytest.approx(-0.05) for sample in samples)
    assert all(sample.master_node_ids is not None for sample in samples)


def test_lagrangian_sdf_triangle_contact_can_clip_to_master_footprint() -> None:
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, -0.05],
            [1.0, 0.0, -0.05],
            [0.0, 1.0, -0.05],
        ],
        dtype=float,
    )
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=float,
    )
    x_current = np.vstack([slave_nodes, master_nodes])
    material = MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64))
    unclipped = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        material,
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=3,
        quadrature="tri3",
        search_radius=1.0,
    )
    clipped = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        material,
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=3,
        quadrature="tri3",
        search_radius=1.0,
        clip_to_master_footprint=True,
    )

    unclipped_samples = list(unclipped.samples(x_current))
    clipped_samples = list(clipped.samples(x_current))

    assert sum(sample.area for sample in unclipped_samples) == pytest.approx(0.5)
    assert sum(sample.area for sample in clipped_samples) == pytest.approx(0.125)
    assert sum(sample.area * max(-sample.gap, 0.0) for sample in clipped_samples) == pytest.approx(0.00625)


def test_lagrangian_sdf_triangle_footprint_clipping_uses_single_closest_owner() -> None:
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, -0.05],
            [1.0, 0.0, -0.05],
            [0.0, 1.0, -0.05],
        ],
        dtype=float,
    )
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=float,
    )
    x_current = np.vstack([slave_nodes, master_nodes])
    material = MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64))
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        material,
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=3,
        quadrature="tri3",
        search_radius=1.0,
        clip_to_master_footprint=True,
    )

    samples = list(contact.samples(x_current))

    assert sum(sample.area for sample in samples) == pytest.approx(0.125)


def test_lagrangian_q4_master_oracle_returns_q4_payload() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    oracle = LagrangianQ4ClosestFeatureOracle(
        master_nodes,
        np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        master_nodes,
    )

    payload = oracle.query(np.asarray([0.5, 0.5, -0.05], dtype=float))

    assert payload.gap == pytest.approx(0.05)
    assert payload.normal == pytest.approx([0.0, 0.0, -1.0])
    assert payload.master_node_ids == pytest.approx([0, 1, 2, 3])
    assert payload.master_weights == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_lagrangian_sdf_q4_master_surface_contact_avoids_triangle_payload_split() -> None:
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, -0.05],
            [0.0, 1.0, -0.05],
            [1.0, 1.0, -0.05],
            [1.0, 0.0, -0.05],
        ],
        dtype=float,
    )
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    x_current = np.vstack([slave_nodes, master_nodes])
    contact = LagrangianSDFQ4MasterSurfaceContactGeometry(
        np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=4,
        quadrature_order=1,
    )

    samples = list(contact.samples(x_current))

    assert len(samples) == 1
    assert samples[0].gap == pytest.approx(0.05)
    assert samples[0].normal == pytest.approx([0.0, 0.0, -1.0])
    assert samples[0].master_node_ids == pytest.approx([4, 5, 6, 7])
    assert samples[0].master_shape_weights == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_q4_path_tracking_cache_updates_only_after_accepted_commit() -> None:
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, -0.05],
            [0.0, 1.0, -0.05],
            [1.0, 1.0, -0.05],
            [1.0, 0.0, -0.05],
        ],
        dtype=float,
    )
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    x_current = np.vstack([slave_nodes, master_nodes])
    contact = LagrangianSDFQ4MasterSurfaceContactGeometry(
        np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=4,
        quadrature_order=1,
        secondary_path_tracking=True,
    )

    trial_arrays = contact.sample_arrays(x_current)

    assert trial_arrays["master_face_ids"].tolist() == [0]
    assert trial_arrays["tracking_cache_hits"].tolist() == [False]
    assert contact._secondary_face_cache is not None
    assert contact._secondary_master_weight_cache is not None
    np.testing.assert_array_equal(contact._secondary_face_cache, [-1])
    assert np.isnan(contact._secondary_master_weight_cache).all()

    assert contact.commit_secondary_tracking_from_sample_arrays(trial_arrays) == 1
    np.testing.assert_array_equal(contact._secondary_face_cache, [0])
    np.testing.assert_allclose(contact._secondary_master_weight_cache[0], [0.25, 0.25, 0.25, 0.25])

    accepted_arrays = contact.sample_arrays(x_current)

    assert accepted_arrays["tracking_cache_hits"].tolist() == [True]
    assert accepted_arrays["tracking_cache_matches"].tolist() == [True]
    np.testing.assert_array_equal(contact._secondary_face_cache, [0])


def test_q4_path_tracking_prefers_accepted_face_and_adjacent_master_patch() -> None:
    slave_nodes = np.asarray(
        [
            [1.1, 0.1, -0.05],
            [1.1, 0.9, -0.05],
            [1.9, 0.9, -0.05],
            [1.9, 0.1, -0.05],
        ],
        dtype=float,
    )
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    x_current = np.vstack([slave_nodes, master_nodes])
    contact = LagrangianSDFQ4MasterSurfaceContactGeometry(
        np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        np.asarray([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=np.int64),
        master_nodes,
        pressure_stiffness=100.0,
        master_node_offset=4,
        quadrature_order=1,
        secondary_path_tracking=True,
        secondary_tracking_rings=1,
    )
    contact._ensure_secondary_face_cache()[0] = 0
    contact._ensure_secondary_master_weight_cache()[0] = [0.25, 0.25, 0.25, 0.25]

    assert contact._preferred_secondary_tracking_faces(0) == [0, 1]

    arrays = contact.sample_arrays(x_current)

    assert arrays["tracking_cache_hits"].tolist() == [True]
    assert arrays["tracking_cache_matches"].tolist() == [False]
    assert arrays["master_face_ids"].tolist() == [1]
    np.testing.assert_array_equal(contact._secondary_face_cache, [0])


def test_tri_batch_path_tracking_cache_updates_only_after_accepted_commit() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.1, 0.1, -0.05],
            [0.3, 0.1, -0.05],
            [0.1, 0.3, -0.05],
        ],
        dtype=float,
    )
    x_current = np.vstack([master_nodes, slave_nodes])
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 2, 1]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=100.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        compiled_batch_projection=True,
        secondary_path_tracking=True,
    )

    trial_arrays = contact.sample_arrays(x_current)
    if trial_arrays is None:
        pytest.skip("compiled triangle batch projection backend is unavailable")

    assert trial_arrays["master_face_ids"].tolist() == [0]
    assert trial_arrays["tracking_cache_hits"].tolist() == [False]
    np.testing.assert_array_equal(contact._ensure_secondary_face_cache(), [-1])
    assert np.isnan(contact._ensure_secondary_barycentric_cache()).all()

    assert contact.commit_secondary_tracking_from_sample_arrays(trial_arrays) == 1
    np.testing.assert_array_equal(contact._ensure_secondary_face_cache(), [0])
    np.testing.assert_allclose(np.sum(contact._ensure_secondary_barycentric_cache()[0]), 1.0)

    accepted_arrays = contact.sample_arrays(x_current)
    assert accepted_arrays is not None
    assert accepted_arrays["tracking_cache_hits"].tolist() == [True]
    assert accepted_arrays["tracking_cache_matches"].tolist() == [True]
    np.testing.assert_array_equal(contact._ensure_secondary_face_cache(), [0])


def test_secondary_normal_projection_path_tracking_cache_updates_only_after_accepted_commit() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.1, 0.1, -0.05],
            [0.3, 0.1, -0.05],
            [0.1, 0.3, -0.05],
        ],
        dtype=float,
    )
    x_current = np.vstack([master_nodes, slave_nodes])
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 2, 1]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=100.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        compiled_batch_projection=True,
        secondary_path_tracking=True,
    )

    trial_arrays = contact.secondary_normal_projection_sample_arrays(x_current)
    if trial_arrays is None:
        pytest.skip("compiled secondary-normal projection backend is unavailable")

    assert trial_arrays["master_face_ids"].tolist() == [0]
    assert trial_arrays["tracking_cache_hits"].tolist() == [False]
    np.testing.assert_array_equal(contact._ensure_secondary_face_cache(), [-1])
    assert np.isnan(contact._ensure_secondary_barycentric_cache()).all()

    assert contact.commit_secondary_tracking_from_sample_arrays(trial_arrays) == 1
    np.testing.assert_array_equal(contact._ensure_secondary_face_cache(), [0])
    np.testing.assert_allclose(np.sum(contact._ensure_secondary_barycentric_cache()[0]), 1.0)

    accepted_arrays = contact.secondary_normal_projection_sample_arrays(x_current)
    assert accepted_arrays is not None
    assert accepted_arrays["tracking_cache_hits"].tolist() == [True]
    assert accepted_arrays["tracking_cache_matches"].tolist() == [True]
    np.testing.assert_array_equal(contact._ensure_secondary_face_cache(), [0])


def test_secondary_region_tracking_commit_and_query_hint_are_region_based() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.2, 0.2, -0.05],
            [0.4, 0.2, -0.05],
            [0.2, 0.4, -0.05],
        ],
        dtype=float,
    )
    x_current = np.vstack([master_nodes, slave_nodes])
    material = MaterialSDF.from_triangle_surface(
        master_nodes,
        np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 2, 1]], dtype=np.int64),
        material,
        master_nodes,
        pressure_stiffness=100.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=1.0,
        secondary_path_tracking=True,
    )
    raw = {
        "sample_node_ids": np.asarray([[3, 4, 5]], dtype=np.int64),
        "sample_weights": np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        "gaps": np.asarray([-0.05], dtype=float),
        "normals": np.asarray([[0.0, 0.0, 1.0]], dtype=float),
        "areas": np.asarray([0.5], dtype=float),
        "master_node_ids": np.asarray([[0, 1, 2]], dtype=np.int64),
        "master_weights": np.asarray([[0.2, 0.3, 0.5]], dtype=float),
        "master_barycentric": np.asarray([[0.2, 0.3, 0.5]], dtype=float),
        "master_face_ids": np.asarray([1], dtype=np.int64),
    }
    regions = aggregate_contact_sample_arrays(raw, "slave_node_region_constraint")
    assert regions is not None

    committed = contact.commit_secondary_tracking_from_sample_arrays(regions)

    assert committed == 1
    assert contact._secondary_region_face_cache == {3: 1}
    assert contact._secondary_face_cache is None

    payload = contact._secondary_projection_query(
        x_current[3],
        np.asarray([0.0, 0.0, -1.0], dtype=float),
        master_nodes,
        material.boundary_faces,
        None,
        0.0,
        dot_threshold=0.0,
        cache_key=(0, "secondary", 0),
        region_nodes=np.asarray([3, 4, 5], dtype=np.int64),
        region_weights=np.asarray([1.0, 0.0, 0.0], dtype=float),
    )

    assert int(payload["master_face_id"]) == 1
    assert contact._secondary_region_hint_faces(np.asarray([3, 4, 5]), np.asarray([1.0, 0.0, 0.0])) == [1]
