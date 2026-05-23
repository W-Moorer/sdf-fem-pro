from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flexible_gear_explicit_sdf_comparison import DEFAULT_SOURCE, parse_gear_input
from validation.run_flexible_gear_implicit_lagrangian_sdf_comparison import (
    _active_reduced_gap_jacobian_sparse,
    _active_reduced_gap_jacobian_sparse_from_arrays,
    _aggregate_contact_samples,
    _assemble_contact_arrays_force_only,
    _assemble_contact_response_force_only,
    _default_contact_search_radius,
    _filter_contact_samples_by_normal_compatibility,
    _node_average_cell_scalar,
    _node_average_cell_tensor,
    _surface_edge_length_percentile,
    _source_drive_corotated_elastic_matrix,
    _source_drive_corotated_positions_and_elastic_displacement,
    _source_drive_centripetal_acceleration,
    _source_drive_centripetal_reduced_response,
    _source_drive_corotated_visual_state_and_internal,
    _source_drive_finite_kinematic_inertia_response,
    _source_drive_finite_visual_jacobian,
    _write_abaqus_alignment_deck,
    build_cropped_pair,
    solve_sfc_cropped_pair,
    solve_sfc_cropped_pair_hard_contact,
    solve_sfc_source_drive_pair,
)
from sfc.contact.hard_contact import hard_contact_gap_jacobian_from_samples
from sfc.contact.lagrangian_surface_contact import LagrangianSDFSurfaceContactGeometry
from sfc.fem.calculix_aligned import ContactSample, MechanicsModel, MechanicsState, assemble_contact_response, stvk_internal_response
from sfc.fem.rp_mpc import RigidHubMPC, build_rigid_hub_reduced_assembly
from sfc.sdf import _cpp_projection
from sfc.sdf.material_sdf import MaterialSDF
from validation.run_flexible_gear_full_lagrangian_sdf_comparison import (
    build_full_active_pair,
    compare_animation_manifests,
    write_animation_color_ranges,
    write_paraview_animation_setup,
)
from validation.run_source_gear_vtk_manifest_alignment import align_manifests


pytestmark = pytest.mark.skipif(not DEFAULT_SOURCE.exists(), reason="commercial gear input is not present")


def test_node_average_cell_scalar_matches_incident_cell_average() -> None:
    cells = np.asarray([[0, 1, 2, 3], [1, 4, 2, 5], [1, 5, 2, 6]], dtype=np.int64)
    values = np.asarray([2.0, 5.0, 11.0], dtype=float)

    averaged = _node_average_cell_scalar(values, cells, node_count=7)

    assert averaged[0] == pytest.approx(2.0)
    assert averaged[1] == pytest.approx((2.0 + 5.0 + 11.0) / 3.0)
    assert averaged[2] == pytest.approx((2.0 + 5.0 + 11.0) / 3.0)
    assert averaged[3] == pytest.approx(2.0)
    assert averaged[4] == pytest.approx(5.0)
    assert averaged[5] == pytest.approx((5.0 + 11.0) / 2.0)
    assert averaged[6] == pytest.approx(11.0)


def test_node_average_cell_tensor_averages_components_before_invariant() -> None:
    cells = np.asarray([[0, 1, 2, 3], [1, 4, 2, 5]], dtype=np.int64)
    tensors = np.zeros((2, 3, 3), dtype=float)
    tensors[0, 0, 0] = 10.0
    tensors[1, 1, 1] = 10.0

    averaged = _node_average_cell_tensor(tensors, cells, node_count=6)

    np.testing.assert_allclose(averaged[0], tensors[0])
    assert averaged[1, 0, 0] == pytest.approx(5.0)
    assert averaged[1, 1, 1] == pytest.approx(5.0)
    assert averaged[2, 0, 0] == pytest.approx(5.0)
    assert averaged[2, 1, 1] == pytest.approx(5.0)
    np.testing.assert_allclose(averaged[4], tensors[1])


def test_default_contact_search_radius_uses_surface_feature_size() -> None:
    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    pair = SimpleNamespace(
        initial_patch_gap=1.0e-8,
        gear1=SimpleNamespace(nodes=nodes, contact_faces=faces),
        gear2=SimpleNamespace(nodes=nodes.copy(), contact_faces=faces.copy()),
    )

    feature = _surface_edge_length_percentile(nodes, faces)
    radius = _default_contact_search_radius(pair, target_overclosure=0.0)

    assert feature > 0.0
    assert radius == pytest.approx(2.5 * feature)
    assert radius > 1.0


def test_source_force_only_contact_response_matches_full_force_response() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.2, 0.3, 0.5], dtype=float),
            gap=-0.01,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=0.25,
            stiffness=1000.0,
            master_node_ids=np.asarray([3, 4, 5], dtype=np.int64),
            master_shape_weights=np.asarray([0.4, 0.4, 0.2], dtype=float),
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([1.0 / 3.0] * 3, dtype=float),
            gap=0.02,
            normal=np.asarray([0.0, 1.0, 0.0], dtype=float),
            area=0.25,
            stiffness=1000.0,
        ),
    ]

    full = assemble_contact_response(samples, n_nodes=6)
    fast = _assemble_contact_response_force_only(samples, n_nodes=6)
    arrays = {
        "sample_node_ids": np.vstack([sample.node_ids for sample in samples]),
        "sample_weights": np.vstack([sample.shape_weights for sample in samples]),
        "gaps": np.asarray([sample.gap for sample in samples], dtype=float),
        "normals": np.vstack([sample.normal for sample in samples]),
        "areas": np.asarray([sample.area for sample in samples], dtype=float),
        "master_node_ids": np.vstack(
            [
                sample.master_node_ids if sample.master_node_ids is not None else np.asarray([0, 0, 0], dtype=np.int64)
                for sample in samples
            ]
        ),
        "master_weights": np.vstack(
            [
                sample.master_shape_weights if sample.master_shape_weights is not None else np.zeros(3, dtype=float)
                for sample in samples
            ]
        ),
    }
    batched = _assemble_contact_arrays_force_only(arrays, n_nodes=6, stiffness=1000.0)

    np.testing.assert_allclose(fast.force, full.force)
    np.testing.assert_allclose(batched.force, full.force)
    assert fast.min_gap == pytest.approx(full.min_gap)
    assert batched.min_gap == pytest.approx(full.min_gap)
    assert fast.max_penetration == pytest.approx(full.max_penetration)
    assert batched.max_penetration == pytest.approx(full.max_penetration)
    assert fast.active_count == full.active_count
    assert batched.active_count == full.active_count
    assert fast.normal_force == pytest.approx(full.normal_force)
    assert batched.normal_force == pytest.approx(full.normal_force)
    assert fast.energy == pytest.approx(full.energy)
    assert batched.energy == pytest.approx(full.energy)
    assert fast.tangent.nnz == 0
    assert batched.tangent.nnz == 0


def test_slave_face_contact_averaging_combines_tri3_samples_without_tuning() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], dtype=float),
            gap=-0.30,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=2.0,
            stiffness=10.0,
            master_node_ids=np.asarray([4, 5, 6], dtype=np.int64),
            master_shape_weights=np.asarray([1.0, 0.0, 0.0], dtype=float),
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=float),
            gap=0.00,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=2.0,
            stiffness=10.0,
            master_node_ids=np.asarray([4, 5, 6], dtype=np.int64),
            master_shape_weights=np.asarray([0.0, 1.0, 0.0], dtype=float),
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], dtype=float),
            gap=0.15,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=2.0,
            stiffness=10.0,
            master_node_ids=np.asarray([4, 5, 6], dtype=np.int64),
            master_shape_weights=np.asarray([0.0, 0.0, 1.0], dtype=float),
        ),
    ]

    aggregated = _aggregate_contact_samples(samples, "slave_face")

    assert len(aggregated) == 1
    merged = aggregated[0]
    assert merged.area == pytest.approx(6.0)
    assert merged.gap == pytest.approx(-0.30 / 3.0)
    assert np.sum(merged.shape_weights) == pytest.approx(1.0)
    assert np.sum(merged.master_shape_weights) == pytest.approx(1.0)
    np.testing.assert_allclose(merged.shape_weights, [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
    np.testing.assert_allclose(merged.master_shape_weights, [1.0, 0.0, 0.0])
    assert merged.node_ids.tolist() == [0, 1, 2]
    assert merged.master_node_ids.tolist() == [4, 5, 6]


def test_active_reduced_gap_jacobian_matches_full_projection() -> None:
    reference_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    hub = RigidHubMPC(np.asarray([0, 1], dtype=np.int64), reference_nodes, np.zeros(3, dtype=float))
    assembly = build_rigid_hub_reduced_assembly(reference_nodes, [hub], include_free_nodes=True)
    sample = ContactSample(
        node_ids=np.asarray([0, 2, 3], dtype=np.int64),
        shape_weights=np.asarray([0.2, 0.3, 0.5], dtype=float),
        gap=-0.01,
        normal=np.asarray([0.2, 0.3, 0.4], dtype=float),
        area=0.25,
        stiffness=1000.0,
        master_node_ids=np.asarray([1], dtype=np.int64),
        master_shape_weights=np.asarray([1.0], dtype=float),
    )
    free = np.arange(assembly.n_reduced_dofs, dtype=np.int64)

    _gaps, full_jacobian = hard_contact_gap_jacobian_from_samples([sample], n_total_dofs=3 * reference_nodes.shape[0])
    projected = (full_jacobian @ assembly.transformation)[:, free]
    direct = _active_reduced_gap_jacobian_sparse([sample], transformation=assembly.transformation, free=free)
    arrays = {
        "sample_node_ids": sample.node_ids.reshape(1, 3),
        "sample_weights": sample.shape_weights.reshape(1, 3),
        "gaps": np.asarray([sample.gap], dtype=float),
        "normals": sample.normal.reshape(1, 3),
        "areas": np.asarray([sample.area], dtype=float),
        "master_node_ids": np.asarray([[1, 1, 1]], dtype=np.int64),
        "master_weights": np.asarray([[1.0, 0.0, 0.0]], dtype=float),
    }
    direct_arrays = _active_reduced_gap_jacobian_sparse_from_arrays(arrays, transformation=assembly.transformation, free=free)

    np.testing.assert_allclose(direct.toarray(), np.asarray(projected), atol=1.0e-14)
    np.testing.assert_allclose(direct_arrays.toarray(), np.asarray(projected), atol=1.0e-14)


def test_cropped_gear_pair_is_small_and_has_supports() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=4, expansion_rings=0)

    assert pair.gear1.nodes.shape[0] < model.gear1.nodes.shape[0]
    assert pair.gear2.nodes.shape[0] < model.gear2.nodes.shape[0]
    assert pair.gear1.contact_faces.shape[0] == 4
    assert pair.gear2.contact_faces.shape[0] == 4
    assert pair.gear1.support_nodes.size >= 4
    assert pair.gear2.support_nodes.size >= 4
    assert pair.initial_patch_gap >= 0.0


def test_full_active_gear_pair_keeps_complete_volume_mesh() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_full_active_pair(model, active_faces_per_body=4)
    expanded = build_full_active_pair(model, active_faces_per_body=4, active_patch_radius_factor=2.0)

    assert pair.gear1.nodes.shape == model.gear1.nodes.shape
    assert pair.gear2.nodes.shape == model.gear2.nodes.shape
    assert pair.gear1.elements.shape == model.gear1.elements.shape
    assert pair.gear2.elements.shape == model.gear2.elements.shape
    assert pair.gear1.contact_faces.shape[0] == 4
    assert pair.gear2.contact_faces.shape[0] == 4
    assert pair.gear1.support_nodes.size == len(set(model.gear1_hub_labels))
    assert pair.gear2.support_nodes.size == len(set(model.gear2_hub_labels))
    assert pair.initial_patch_gap >= 0.0
    assert expanded.gear1.contact_faces.shape[0] >= pair.gear1.contact_faces.shape[0]
    assert expanded.gear2.contact_faces.shape[0] >= pair.gear2.contact_faces.shape[0]


def test_full_active_gear_pair_can_use_all_source_contact_faces() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_full_active_pair(model, active_faces_per_body=0)

    assert pair.gear1.nodes.shape == model.gear1.nodes.shape
    assert pair.gear2.nodes.shape == model.gear2.nodes.shape
    assert pair.gear1.contact_faces.shape[0] == model.gear1_contact_faces.shape[0]
    assert pair.gear2.contact_faces.shape[0] == model.gear2_contact_faces.shape[0]
    assert pair.gear1.surface_entries == model.gear1_surface_entries
    assert pair.gear2.surface_entries == model.gear2_surface_entries


def test_cropped_gear_sfc_lagrangian_sdf_path_runs_one_implicit_step() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
    )

    assert len(history) == 1
    assert summary["status"] == "completed"
    assert summary["nodes"] > 0
    assert summary["elements"] > 0
    assert history[-1]["newton_iterations"] >= 1


def test_cropped_gear_modified_newton_penalty_path_runs_one_implicit_step() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        penalty_solver="modified_newton",
        material_linearization="reference_linear",
    )

    assert len(history) == 1
    assert summary["status"] == "completed"
    assert summary["contact_mode"] == "penalty"
    assert summary["penalty_solver"] == "modified_newton"
    assert summary["material_linearization"] == "reference_linear"
    assert int(summary["penalty_fallback_count"]) >= 0
    assert float(summary["timing_base_tangent_seconds"]) >= 0.0
    assert history[-1]["penalty_solver"] in {"modified_newton", "full_newton_fallback"}
    assert history[-1]["newton_iterations"] >= 1


def test_cropped_gear_modified_newton_writes_sfc_vtk_frames(tmp_path: Path) -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)
    vtk_dir = tmp_path / "sfc_vtk"

    _history, summary = solve_sfc_cropped_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        penalty_solver="modified_newton",
        material_linearization="reference_linear",
        vtk_out_dir=vtk_dir,
        vtk_frame_stride=1,
    )

    pvd = Path(str(summary["sfc_vtk_pvd"]))
    manifest = Path(str(summary["sfc_vtk_manifest"]))
    first = vtk_dir / "sfc_0000.vtk"
    second = vtk_dir / "sfc_0001.vtk"
    assert pvd.exists()
    assert manifest.exists()
    assert first.exists()
    assert second.exists()
    assert int(summary["sfc_vtk_frame_count"]) == 2
    assert pvd.read_text(encoding="utf-8").count("<DataSet") == 2
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "max_displacement_magnitude" in manifest_text
    assert "max_equivalent_elastic_strain_nodeavg" in manifest_text
    assert "p95_von_mises_nodeavg" in manifest_text
    assert "p95_equivalent_elastic_strain_nodeavg" in manifest_text
    assert "max_displacement_magnitude_object1" in manifest_text
    text = second.read_text(encoding="ascii")
    assert "DATASET UNSTRUCTURED_GRID" in text
    assert "VECTORS U float" in text
    assert "SCALARS von_mises_nodeavg float 1" in text
    assert "SCALARS strain_norm_nodeavg float 1" in text
    assert "SCALARS equivalent_elastic_strain_nodeavg float 1" in text
    assert "SCALARS contact_pressure_nodeavg float 1" in text
    assert "SCALARS contact_active_node float 1" in text
    assert "SCALARS von_mises float 1" in text
    assert "TENSORS S float" in text


def test_cropped_gear_source_drive_path_advances_rp_rotation(tmp_path: Path) -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)
    vtk_dir = tmp_path / "source_sfc_vtk"

    history, summary = solve_sfc_source_drive_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-5,
        dt=1.0e-5,
        gear1_angular_velocity_z=model.gear1_angular_velocity_z,
        gear2_torque_z=model.gear2_torque_z,
        vtk_out_dir=vtk_dir,
        vtk_frame_stride=1,
    )

    assert len(history) == 1
    assert summary["status"] == "completed"
    assert summary["contact_mode"] == "source_penalty"
    assert float(history[-1]["rp1_rotation_z"]) == pytest.approx(model.gear1_angular_velocity_z * 1.0e-5)
    assert float(history[-1]["gear2_torque_z"]) == pytest.approx(model.gear2_torque_z)
    assert int(summary["reduced_dofs"]) > 0
    assert float(summary["timing_source_base_lu_seconds"]) >= 0.0
    assert int(summary["source_sparse_cg_base_lu_preconditioner"]) >= 0
    assert summary["source_rotation_unit"] == "radian"
    manifest = Path(str(summary["sfc_vtk_manifest"]))
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "linear_corotated" in text
    assert "radian" in text
    assert "active_contact_node_count" in text
    assert "active_contact_slave_node_count" in text
    assert "active_contact_master_node_count" in text
    assert "max_contact_pressure_nodeavg" in text
    vtk_text = (vtk_dir / "sfc_0001.vtk").read_text(encoding="ascii")
    assert "SCALARS contact_pressure_nodeavg float 1" in vtk_text
    assert "SCALARS contact_slave_pressure_nodeavg float 1" in vtk_text
    assert "SCALARS contact_master_pressure_nodeavg float 1" in vtk_text
    assert "SCALARS contact_penetration_nodeavg float 1" in vtk_text
    assert "SCALARS contact_gap_min_node float 1" in vtk_text


def test_source_drive_checkpoint_resume_matches_continuous_short_run(tmp_path: Path) -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)
    checkpoint = tmp_path / "source_drive_checkpoint.npz"

    continuous_history, _continuous_summary = solve_sfc_source_drive_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=2.0e-5,
        dt=1.0e-5,
        gear1_angular_velocity_z=model.gear1_angular_velocity_z,
        gear2_torque_z=model.gear2_torque_z,
        history_frame_stride=1,
    )
    first_history, first_summary = solve_sfc_source_drive_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-5,
        dt=1.0e-5,
        gear1_angular_velocity_z=model.gear1_angular_velocity_z,
        gear2_torque_z=model.gear2_torque_z,
        history_frame_stride=1,
        source_checkpoint_path=checkpoint,
        source_checkpoint_stride=1,
    )
    resumed_history, resumed_summary = solve_sfc_source_drive_pair(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=2.0e-5,
        dt=1.0e-5,
        gear1_angular_velocity_z=model.gear1_angular_velocity_z,
        gear2_torque_z=model.gear2_torque_z,
        history_frame_stride=1,
        source_checkpoint_path=checkpoint,
        resume_source_checkpoint=True,
        source_checkpoint_stride=1,
    )

    assert checkpoint.exists()
    assert len(first_history) == 1
    assert len(resumed_history) == 2
    assert int(first_summary["source_resume_from_step"]) == 0
    assert int(resumed_summary["source_resume_from_step"]) == 1
    for key in ("rp1_rotation_z", "rp2_rotation_z", "max_displacement_norm", "min_gap", "normal_force"):
        assert float(resumed_history[-1][key]) == pytest.approx(float(continuous_history[-1][key]), rel=1.0e-10, abs=1.0e-12)


def test_source_drive_corotated_visual_postprocess_removes_rigid_rotation_stress() -> None:
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
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=1.0)
    reference_internal = stvk_internal_response(model, model.X, assemble_tangent=True)
    theta = 0.7
    small_rigid = np.cross(np.asarray([[0.0, 0.0, theta]], dtype=float), X)
    state = MechanicsState(
        X + small_rigid,
        np.zeros_like(X),
        np.zeros_like(X),
        time=0.0,
    )

    visual_state, internal = _source_drive_corotated_visual_state_and_internal(
        model=model,
        state=state,
        reference_tangent=reference_internal.tangent,
        body_node_slices=(slice(0, 4), slice(4, 4)),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta, 0.0),
    )

    expected_node1 = np.asarray([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    np.testing.assert_allclose(visual_state.x[1], expected_node1, atol=1.0e-14)
    assert float(np.max(internal.von_mises)) <= 1.0e-10


def test_source_drive_corotated_contact_positions_use_finite_rp_rotation() -> None:
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
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=1.0)
    theta = 1.9
    x_small = X + np.cross(np.asarray([[0.0, 0.0, theta]], dtype=float), X)

    x_contact, u_elastic = _source_drive_corotated_positions_and_elastic_displacement(
        model=model,
        x_raw=x_small,
        body_node_slices=(slice(0, 4), slice(4, 4)),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta, 0.0),
    )

    np.testing.assert_allclose(u_elastic, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(x_contact[1], [np.cos(theta), np.sin(theta), 0.0], atol=1.0e-14)


def test_contact_normal_compatibility_filter_keeps_opposing_surfaces() -> None:
    x = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    sample = ContactSample(
        node_ids=np.asarray([0, 1, 2], dtype=np.int64),
        shape_weights=np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        gap=-1.0e-3,
        normal=np.asarray([0.0, 0.0, -1.0]),
        area=1.0,
        stiffness=1.0,
    )
    same_side = ContactSample(
        node_ids=sample.node_ids,
        shape_weights=sample.shape_weights,
        gap=-1.0e-3,
        normal=np.asarray([0.0, 0.0, 1.0]),
        area=1.0,
        stiffness=1.0,
    )

    filtered = _filter_contact_samples_by_normal_compatibility([sample, same_side], x, mode="opposing")

    assert len(filtered) == 1
    assert filtered[0] is sample


def test_slave_face_averaging_groups_by_nodes_after_filtering() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.5, 0.5, 0.0]),
            gap=-1.0e-3,
            normal=np.asarray([0.0, 0.0, 1.0]),
            area=0.2,
            stiffness=10.0,
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.0, 0.5, 0.5]),
            gap=-2.0e-3,
            normal=np.asarray([0.0, 0.0, 1.0]),
            area=0.3,
            stiffness=10.0,
        ),
    ]

    aggregated = _aggregate_contact_samples(samples, "slave_face")

    assert len(aggregated) == 1
    assert aggregated[0].area == pytest.approx(0.5)
    assert aggregated[0].gap == pytest.approx((-1.0e-3 * 0.2 - 2.0e-3 * 0.3) / 0.5)
    np.testing.assert_allclose(aggregated[0].shape_weights, [0.125, 0.5, 0.375])


def test_slave_node_averaging_uses_tributary_area() -> None:
    sample = ContactSample(
        node_ids=np.asarray([0, 1, 2], dtype=np.int64),
        shape_weights=np.asarray([0.5, 0.3, 0.2]),
        gap=-1.0e-3,
        normal=np.asarray([0.0, 0.0, 1.0]),
        area=2.0,
        stiffness=10.0,
    )

    aggregated = _aggregate_contact_samples([sample], "slave_node")

    assert len(aggregated) == 3
    assert sum(float(item.area) for item in aggregated) == pytest.approx(2.0)
    assert {tuple(np.asarray(item.shape_weights, dtype=float)) for item in aggregated} == {
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    }


def test_contact_averaging_preserves_positive_overclosure_integral() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([1.0, 0.0, 0.0], dtype=float),
            gap=-0.4,
            normal=np.asarray([1.0, 0.0, 0.0], dtype=float),
            area=1.0,
            stiffness=10.0,
        ),
        ContactSample(
            node_ids=np.asarray([0, 1, 2], dtype=np.int64),
            shape_weights=np.asarray([0.0, 1.0, 0.0], dtype=float),
            gap=0.2,
            normal=np.asarray([0.0, 1.0, 0.0], dtype=float),
            area=3.0,
            stiffness=10.0,
        ),
    ]

    merged = _aggregate_contact_samples(samples, "slave_face")[0]

    assert merged.area == pytest.approx(4.0)
    assert merged.gap == pytest.approx(-0.1)
    np.testing.assert_allclose(merged.normal, [1.0, 0.0, 0.0], atol=1.0e-14)


def test_lagrangian_surface_contact_nodal_samples_use_slave_corner_gaps() -> None:
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
            [0.0, 0.0, -0.30],
            [1.0, 0.0, -0.10],
            [0.0, 1.0, 0.20],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=3,
        master_node_offset=0,
        compiled_batch_projection=False,
    )
    samples = list(contact.nodal_samples(np.vstack([master_nodes, slave_nodes])))

    assert len(samples) == 3
    np.testing.assert_allclose([sample.gap for sample in samples], [-0.30, -0.10, 0.20], atol=1.0e-12)
    assert [int(np.asarray(sample.node_ids)[int(np.argmax(sample.shape_weights))]) for sample in samples] == [3, 4, 5]
    assert sum(float(sample.area) for sample in samples) == pytest.approx(
        0.5 * float(np.linalg.norm(np.cross(slave_nodes[1] - slave_nodes[0], slave_nodes[2] - slave_nodes[0])))
    )


def test_normal_compatible_samples_choose_nearest_opposing_candidate() -> None:
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.1],
            [0.0, 1.0, -0.1],
            [1.0, 0.0, -0.1],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, 0.01],
            [1.0, 0.0, 0.01],
            [0.0, 1.0, 0.01],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, faces),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=0.2,
        compiled_batch_projection=False,
    )

    regular = list(contact.samples(np.vstack([master_nodes, slave_nodes])))[0]
    compatible = list(contact.normal_compatible_samples(np.vstack([master_nodes, slave_nodes])))[0]

    assert regular.master_node_ids.tolist() == [0, 1, 2]
    assert compatible.master_node_ids.tolist() == [3, 4, 5]
    assert regular.gap == pytest.approx(0.01)
    assert compatible.gap == pytest.approx(-0.11)


def test_normal_compatible_sample_arrays_match_sample_payload_when_cpp_available() -> None:
    if not _cpp_projection.normal_compatible_indexed_faces_available():
        pytest.skip("C++ normal-compatible indexed projection backend is unavailable")
    master_nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.1],
            [0.0, 1.0, -0.1],
            [1.0, 0.0, -0.1],
        ],
        dtype=float,
    )
    slave_nodes = np.asarray(
        [
            [0.0, 0.0, 0.01],
            [1.0, 0.0, 0.01],
            [0.0, 1.0, 0.01],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=0.2,
        compiled_batch_projection=True,
    )

    arrays = contact.normal_compatible_sample_arrays(np.vstack([master_nodes, slave_nodes]))

    assert arrays is not None
    assert arrays["gaps"].shape == (1,)
    assert arrays["master_node_ids"].tolist() == [[3, 4, 5]]
    assert arrays["gaps"][0] == pytest.approx(-0.11)


def test_normal_compatible_sample_arrays_drop_incompatible_cpp_candidates() -> None:
    if not _cpp_projection.normal_compatible_indexed_faces_available():
        pytest.skip("C++ normal-compatible indexed projection backend is unavailable")
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
            [0.0, 0.0, 0.01],
            [1.0, 0.0, 0.01],
            [0.0, 1.0, 0.01],
        ],
        dtype=float,
    )
    contact = LagrangianSDFSurfaceContactGeometry(
        np.asarray([[0, 1, 2]], dtype=np.int64),
        MaterialSDF.from_triangle_surface(master_nodes, np.asarray([[0, 1, 2]], dtype=np.int64)),
        master_nodes,
        pressure_stiffness=10.0,
        slave_node_offset=master_nodes.shape[0],
        master_node_offset=0,
        quadrature="centroid",
        search_radius=0.2,
        compiled_batch_projection=True,
    )

    arrays = contact.normal_compatible_sample_arrays(np.vstack([master_nodes, slave_nodes]))

    assert arrays is not None
    assert arrays["gaps"].shape == (0,)
    assert arrays["master_node_ids"].shape == (0, 3)


def test_source_drive_corotated_elastic_matrix_removes_body_z_rotation() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.3, 0.4],
        ],
        dtype=float,
    )
    hub = RigidHubMPC(np.asarray([0, 1, 2, 3], dtype=np.int64), X, np.zeros(3))
    assembly = build_rigid_hub_reduced_assembly(X, [hub], include_free_nodes=False)
    elastic = _source_drive_corotated_elastic_matrix(
        assembly=assembly,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
    )
    q = np.zeros(assembly.n_reduced_dofs, dtype=float)
    q[assembly.hub_slice(0).start + 5] = 1.7

    np.testing.assert_allclose(np.asarray(elastic @ q).reshape((-1, 3)), 0.0, atol=1.0e-14)


def test_source_drive_finite_visual_jacobian_matches_finite_difference() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.3, 0.4],
        ],
        dtype=float,
    )
    hub = RigidHubMPC(np.asarray([0, 1, 2, 3], dtype=np.int64), X, np.zeros(3))
    assembly = build_rigid_hub_reduced_assembly(X, [hub], include_free_nodes=False)
    q = np.zeros(assembly.n_reduced_dofs, dtype=float)
    q[assembly.hub_slice(0).start + 0] = 0.03
    q[assembly.hub_slice(0).start + 1] = -0.02
    q[assembly.hub_slice(0).start + 5] = 0.73

    def visual_positions(value: np.ndarray) -> np.ndarray:
        x_raw = X + assembly.expand_displacements(value)
        return _source_drive_corotated_positions_and_elastic_displacement(
            model=MechanicsModel.from_tet4_mesh(
                X,
                np.asarray([[0, 1, 2, 3]], dtype=np.int64),
                E=1.0e3,
                nu=0.25,
                density=1.0,
            ),
            x_raw=x_raw,
            body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
            body_reference_points=(np.zeros(3), np.zeros(3)),
            body_rotation_z=(float(value[assembly.hub_slice(0).start + 5]), 0.0),
        )[0]

    jacobian = _source_drive_finite_visual_jacobian(
        assembly=assembly,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(float(q[assembly.hub_slice(0).start + 5]), 0.0),
    )
    eps = 1.0e-7
    for column in range(assembly.n_reduced_dofs):
        perturb = np.zeros_like(q)
        perturb[column] = eps
        fd = (visual_positions(q + perturb) - visual_positions(q - perturb)).reshape(-1) / (2.0 * eps)
        actual = np.asarray(jacobian[:, column].todense()).reshape(-1)
        np.testing.assert_allclose(actual, fd, rtol=5.0e-7, atol=5.0e-8)


def test_source_drive_finite_kinematic_inertia_reduces_to_linear_mass_at_zero_rotation() -> None:
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
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=2.0, mass_kind="consistent")
    hub = RigidHubMPC(np.asarray([0, 1, 2, 3], dtype=np.int64), X, np.zeros(3))
    assembly = build_rigid_hub_reduced_assembly(X, [hub], include_free_nodes=False)
    acceleration = np.linspace(0.1, 0.6, assembly.n_reduced_dofs)

    reduced, mass_tangent = _source_drive_finite_kinematic_inertia_response(
        assembly=assembly,
        mass_matrix=model.mass_matrix,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(0.0, 0.0),
        body_angular_velocity_z=(0.0, 0.0),
        reduced_acceleration=acceleration,
    )
    expected_mass = assembly.reduce_matrix(model.mass_matrix).tocsr()

    np.testing.assert_allclose(reduced, np.asarray(expected_mass @ acceleration).reshape(-1), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(mass_tangent.toarray(), expected_mass.toarray(), rtol=1.0e-12, atol=1.0e-12)


def test_source_drive_finite_kinematic_inertia_mass_tangent_matches_acceleration_fd() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.1, 0.2, 0.9],
        ],
        dtype=float,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=2.0, mass_kind="consistent")
    hub = RigidHubMPC(np.asarray([0, 1, 2, 3], dtype=np.int64), X, np.zeros(3))
    assembly = build_rigid_hub_reduced_assembly(X, [hub], include_free_nodes=False)
    acceleration = np.linspace(-0.2, 0.35, assembly.n_reduced_dofs)
    theta = 0.61
    omega = 7.5
    reduced, mass_tangent = _source_drive_finite_kinematic_inertia_response(
        assembly=assembly,
        mass_matrix=model.mass_matrix,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta, 0.0),
        body_angular_velocity_z=(omega, 0.0),
        reduced_acceleration=acceleration,
    )
    eps = 1.0e-7
    for column in range(assembly.n_reduced_dofs):
        perturb = np.zeros_like(acceleration)
        perturb[column] = eps
        plus, _ = _source_drive_finite_kinematic_inertia_response(
            assembly=assembly,
            mass_matrix=model.mass_matrix,
            reference_nodes=X,
            body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
            body_reference_points=(np.zeros(3), np.zeros(3)),
            body_rotation_z=(theta, 0.0),
            body_angular_velocity_z=(omega, 0.0),
            reduced_acceleration=acceleration + perturb,
            include_mass_tangent=False,
        )
        minus, _ = _source_drive_finite_kinematic_inertia_response(
            assembly=assembly,
            mass_matrix=model.mass_matrix,
            reference_nodes=X,
            body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
            body_reference_points=(np.zeros(3), np.zeros(3)),
            body_rotation_z=(theta, 0.0),
            body_angular_velocity_z=(omega, 0.0),
            reduced_acceleration=acceleration - perturb,
            include_mass_tangent=False,
        )
        fd = (plus - minus) / (2.0 * eps)
        np.testing.assert_allclose(fd, np.asarray(mass_tangent[:, column].todense()).reshape(-1), rtol=1.0e-7, atol=1.0e-8)
    assert float(np.linalg.norm(reduced)) > 0.0


def test_source_drive_centripetal_acceleration_points_inward() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=float,
    )

    acc = _source_drive_centripetal_acceleration(
        reference_nodes=X,
        body_node_slices=(slice(0, 3), slice(3, 3)),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(0.0, 0.0),
        body_angular_velocity_z=(4.0, 0.0),
    )

    np.testing.assert_allclose(acc[0], [0.0, 0.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(acc[1], [-32.0, 0.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(acc[2], [0.0, -48.0, 0.0], atol=1.0e-14)


def test_source_drive_centripetal_reduced_tangent_matches_finite_difference() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 0.0, 0.9],
        ],
        dtype=float,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=2.0)
    hub = RigidHubMPC(np.asarray([0, 1, 2, 3], dtype=np.int64), X, np.zeros(3))
    assembly = build_rigid_hub_reduced_assembly(X, [hub], include_free_nodes=False)
    theta = 0.37
    omega = 8.0
    domega_dtheta = 125.0
    reduced, tangent = _source_drive_centripetal_reduced_response(
        assembly=assembly,
        mass_matrix=model.mass_matrix,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta, 0.0),
        body_angular_velocity_z=(omega, 0.0),
        velocity_sensitivity_z=(domega_dtheta, 0.0),
        include_tangent=True,
    )
    eps = 1.0e-7
    plus, _ = _source_drive_centripetal_reduced_response(
        assembly=assembly,
        mass_matrix=model.mass_matrix,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta + eps, 0.0),
        body_angular_velocity_z=(omega + domega_dtheta * eps, 0.0),
        include_tangent=False,
    )
    minus, _ = _source_drive_centripetal_reduced_response(
        assembly=assembly,
        mass_matrix=model.mass_matrix,
        reference_nodes=X,
        body_node_slices=(slice(0, X.shape[0]), slice(X.shape[0], X.shape[0])),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta - eps, 0.0),
        body_angular_velocity_z=(omega - domega_dtheta * eps, 0.0),
        include_tangent=False,
    )
    fd_column = (plus - minus) / (2.0 * eps)
    tangent_column = np.asarray(tangent[:, assembly.hub_slice(0).start + 5].toarray()).reshape(-1)

    assert np.linalg.norm(reduced) > 0.0
    np.testing.assert_allclose(tangent_column, fd_column, rtol=1.0e-7, atol=1.0e-7)


def test_source_drive_visual_postprocess_is_objective_for_large_rotation() -> None:
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
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=1.0)
    reference_internal = stvk_internal_response(model, model.X, assemble_tangent=True)
    theta = 2.4
    small_rigid = np.cross(np.asarray([[0.0, 0.0, theta]], dtype=float), X)
    state = MechanicsState(
        X + small_rigid,
        np.zeros_like(X),
        np.zeros_like(X),
        time=0.0,
    )

    visual_state, internal = _source_drive_corotated_visual_state_and_internal(
        model=model,
        state=state,
        reference_tangent=reference_internal.tangent,
        body_node_slices=(slice(0, 4), slice(4, 4)),
        body_reference_points=(np.zeros(3), np.zeros(3)),
        body_rotation_z=(theta, 0.0),
        stress_postprocess="finite_stvk_visual",
    )

    expected_node1 = np.asarray([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    np.testing.assert_allclose(visual_state.x[1], expected_node1, atol=1.0e-14)
    assert float(np.max(internal.von_mises)) <= 1.0e-9


def test_source_drive_visual_postprocess_rejects_unknown_mode() -> None:
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
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1.0e3, nu=0.25, density=1.0)
    reference_internal = stvk_internal_response(model, model.X, assemble_tangent=True)
    state = MechanicsState(X.copy(), np.zeros_like(X), np.zeros_like(X), time=0.0)

    with pytest.raises(ValueError, match="stress_postprocess"):
        _source_drive_corotated_visual_state_and_internal(
            model=model,
            state=state,
            reference_tangent=reference_internal.tangent,
            body_node_slices=(slice(0, 4), slice(4, 4)),
            body_reference_points=(np.zeros(3), np.zeros(3)),
            body_rotation_z=(0.0, 0.0),
            stress_postprocess="bad_mode",
        )


def test_cropped_gear_abaqus_deck_prescribes_matching_rp_motion(tmp_path) -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)
    path = tmp_path / "cropped_gear_alignment.inp"

    _write_abaqus_alignment_deck(
        path,
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=2.0e-3,
        dt=1.0e-3,
        target_overclosure=3.0e-4,
        rotation_rate_z=2.0,
    )

    text = path.read_text(encoding="ascii")
    assert "G2_RP, 1, 6" in text
    assert "G1_RP, 4, 5" in text
    assert "*Boundary, amplitude=RAMP" in text
    assert "*Dynamic, application=MODERATE DISSIPATION" in text
    assert "** Rotational boundary values are radians." in text
    assert "G1_RP, 6, 6, 4.000000000000e-03" in text
    assert "S, E, LE" in text
    assert "CPRESS" not in text
    assert "COPEN" not in text


def test_animation_color_ranges_use_global_sfc_and_abaqus_limits(tmp_path: Path) -> None:
    sfc_manifest = tmp_path / "sfc_manifest.csv"
    abaqus_manifest = tmp_path / "abaqus_manifest.csv"
    sfc_manifest.write_text(
        "frame,time,vtk_file,node_count,element_count,max_displacement_magnitude,max_von_mises,max_strain_norm,max_equivalent_elastic_strain,max_von_mises_nodeavg,max_strain_norm_nodeavg,max_equivalent_elastic_strain_nodeavg\n"
        "0,0,sfc_0000.vtk,1,1,0.1,5.0,0.02,0.002,4.0,0.018,0.0018\n"
        "1,1,sfc_0001.vtk,1,1,0.2,7.0,0.03,0.003,6.0,0.028,0.0028\n",
        encoding="utf-8",
    )
    abaqus_manifest.write_text(
        "frame,source_frame,time,vtk_file,node_count,element_count,max_displacement_magnitude,max_von_mises,max_le_norm,max_equivalent_elastic_strain,max_von_mises_nodeavg,max_le_norm_nodeavg,max_equivalent_elastic_strain_nodeavg\n"
        "0,0,0,abaqus_0000.vtk,1,1,0.15,6.0,0.01,0.001,5.0,0.009,0.0009\n"
        "1,1,1,abaqus_0001.vtk,1,1,0.25,8.0,0.04,0.004,7.0,0.038,0.0038\n",
        encoding="utf-8",
    )

    ranges = write_animation_color_ranges(tmp_path, sfc_manifest=sfc_manifest, abaqus_manifest=abaqus_manifest)
    text = ranges.read_text(encoding="utf-8")

    assert "displacement_magnitude" in text
    assert "von_mises" in text
    assert "strain_norm" in text
    assert "von_mises_nodeavg" in text
    assert "strain_norm_nodeavg" in text
    assert "equivalent_elastic_strain_nodeavg" in text
    assert "0.25" in text
    assert "8.0" in text
    assert "0.04" in text
    assert "7.0" in text
    assert "0.038" in text


def test_paraview_animation_setup_uses_nodeavg_fixed_ranges(tmp_path: Path) -> None:
    ranges = tmp_path / "animation_fixed_color_ranges.csv"
    ranges.write_text(
        "field,recommended_min,recommended_max,sfc_column,abaqus_column,paraview_note\n"
        "displacement_magnitude,0.0,0.25,max_displacement_magnitude,max_displacement_magnitude,note\n"
        "von_mises_nodeavg,0.0,8.0,max_von_mises_nodeavg,max_von_mises_nodeavg,note\n"
        "equivalent_elastic_strain_nodeavg,0.0,0.0038,max_equivalent_elastic_strain_nodeavg,max_equivalent_elastic_strain_nodeavg,note\n",
        encoding="utf-8",
    )
    sfc_pvd = tmp_path / "sfc_vtk" / "sfc.pvd"
    abaqus_pvd = tmp_path / "abaqus_vtk" / "abaqus.pvd"

    script = write_paraview_animation_setup(
        tmp_path,
        color_ranges=ranges,
        sfc_pvd=sfc_pvd,
        abaqus_pvd=abaqus_pvd,
    )
    text = script.read_text(encoding="utf-8")

    assert "DEFAULT_FIELD = \"von_mises_nodeavg\"" in text
    assert "equivalent_elastic_strain_nodeavg" in text
    assert "RescaleTransferFunction(float(vmin), float(vmax))" in text
    assert "sfc_vtk/sfc.pvd" in text
    assert "abaqus_vtk/abaqus.pvd" in text
    assert str(tmp_path.resolve()) not in text


def test_compare_animation_manifests_writes_curve_inputs(tmp_path: Path) -> None:
    sfc_manifest = tmp_path / "sfc_manifest.csv"
    abaqus_manifest = tmp_path / "abaqus_manifest.csv"
    sfc_manifest.write_text(
        "frame,time,vtk_file,node_count,element_count,max_displacement_magnitude,max_von_mises,max_strain_norm,max_equivalent_elastic_strain,max_von_mises_nodeavg,p95_von_mises_nodeavg,max_strain_norm_nodeavg,max_equivalent_elastic_strain_nodeavg,p95_equivalent_elastic_strain_nodeavg,max_displacement_magnitude_object1,max_displacement_magnitude_object2,rp2_rotation_z_rad,rp2_angular_velocity_z_rad_per_s\n"
        "0,0,sfc_0000.vtk,1,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
        "1,0.5,sfc_0001.vtk,1,1,0.2,7.0,0.03,0.003,6.0,5.0,0.028,0.0028,0.0025,0.11,0.20,0.06,0.8\n",
        encoding="utf-8",
    )
    abaqus_manifest.write_text(
        "frame,source_frame,time,vtk_file,node_count,element_count,max_displacement_magnitude,max_von_mises,max_le_norm,max_equivalent_elastic_strain,max_von_mises_nodeavg,p95_von_mises_nodeavg,max_le_norm_nodeavg,max_equivalent_elastic_strain_nodeavg,p95_equivalent_elastic_strain_nodeavg,max_displacement_magnitude_object1,max_displacement_magnitude_object2,rp2_rotation_z_rad,rp2_angular_velocity_z_rad_per_s\n"
        "0,0,0,abaqus_0000.vtk,1,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
        "1,1,1,abaqus_0001.vtk,1,1,0.4,10.0,0.08,0.006,8.0,6.0,0.04,0.0056,0.0030,0.10,0.30,0.10,1.2\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "errors.csv"
    out_png = tmp_path / "curves.png"

    rows = compare_animation_manifests(
        sfc_manifest=sfc_manifest,
        abaqus_manifest=abaqus_manifest,
        out_csv=out_csv,
        out_png=out_png,
    )

    assert out_csv.exists()
    assert out_png.exists()
    assert len(rows) == 2
    assert float(rows[-1]["abaqus_max_displacement_magnitude"]) == pytest.approx(0.2)
    assert float(rows[-1]["max_von_mises_nodeavg_rel_error"]) == pytest.approx(0.5)
    assert float(rows[-1]["p95_von_mises_nodeavg_rel_error"]) == pytest.approx(2.0 / 3.0)
    assert "p95_equivalent_elastic_strain_nodeavg_rel_error" in rows[-1]
    assert "max_equivalent_elastic_strain_nodeavg_rel_error" in rows[-1]
    assert "max_strain_norm_nodeavg_rel_error" not in rows[-1]
    assert "max_displacement_magnitude_object1_rel_error" in rows[-1]
    assert "max_displacement_magnitude_object2_rel_error" in rows[-1]
    assert "rp2_rotation_z_rad_rel_error" in rows[-1]
    assert "rp2_angular_velocity_z_rad_per_s_rel_error" in rows[-1]


def test_source_gear_manifest_alignment_entrypoint_writes_artifacts(tmp_path: Path) -> None:
    sfc_dir = tmp_path / "sfc_vtk"
    abaqus_dir = tmp_path / "abaqus_vtk"
    sfc_dir.mkdir()
    abaqus_dir.mkdir()
    sfc_manifest = sfc_dir / "sfc_manifest.csv"
    abaqus_manifest = abaqus_dir / "abaqus_manifest.csv"
    sfc_manifest.write_text(
        "frame,time,vtk_file,node_count,element_count,max_displacement_magnitude,max_von_mises,max_strain_norm,max_equivalent_elastic_strain,max_von_mises_nodeavg,p95_von_mises_nodeavg,max_strain_norm_nodeavg,max_equivalent_elastic_strain_nodeavg,p95_equivalent_elastic_strain_nodeavg\n"
        "0,0,sfc_0000.vtk,1,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
        "1,1,sfc_0001.vtk,1,1,0.2,7.0,0.03,0.003,6.0,5.0,0.028,0.0028,0.0025\n",
        encoding="utf-8",
    )
    abaqus_manifest.write_text(
        "frame,source_frame,time,vtk_file,node_count,element_count,max_displacement_magnitude,max_von_mises,max_le_norm,max_equivalent_elastic_strain,max_von_mises_nodeavg,p95_von_mises_nodeavg,max_le_norm_nodeavg,max_equivalent_elastic_strain_nodeavg,p95_equivalent_elastic_strain_nodeavg\n"
        "0,0,0,abaqus_0000.vtk,1,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
        "1,1,1,abaqus_0001.vtk,1,1,0.25,8.0,0.04,0.004,7.0,6.0,0.038,0.0038,0.003\n",
        encoding="utf-8",
    )

    summary = align_manifests(
        sfc_manifest=sfc_manifest,
        abaqus_manifest=abaqus_manifest,
        out_dir=tmp_path / "comparison",
    )

    assert int(summary["frame_count"]) == 2
    assert Path(str(summary["summary"])).exists()
    assert Path(str(summary["errors"])).exists()
    assert Path(str(summary["curves"])).exists()
    assert Path(str(summary["ranges"])).exists()
    script = Path(str(summary["paraview"]))
    assert script.exists()
    script_text = script.read_text(encoding="utf-8")
    assert "../sfc_vtk/sfc.pvd" in script_text
    assert "../abaqus_vtk/abaqus.pvd" in script_text


def test_cropped_gear_hard_contact_path_runs_one_implicit_step() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        max_iterations=4,
        automatic_increment=False,
    )

    assert len(history) == 1
    assert summary["status"] == "completed"
    assert summary["contact_mode"] == "hard"
    assert summary["hard_enforcement"] == "abaqus_standard_penalty"
    assert float(summary["hht_alpha"]) < 0.0
    assert summary["nodes"] > 0
    assert summary["elements"] > 0
    assert int(history[-1]["hard_contact_samples"]) > 0
    assert int(history[-1]["hard_active_set_converged"]) == 1
    assert "timing_internal_tangent_seconds" in summary
    assert float(summary["timing_internal_tangent_seconds"]) >= 0.0
    assert "timing_hard_contact_solve_seconds" in summary
    assert float(summary["timing_hard_contact_solve_seconds"]) >= 0.0
    assert "rp_force_norm" in history[-1]
    assert float(history[-1]["rp_force_norm"]) >= 0.0
    assert "opposing_rp_force_norm" in history[-1]
    assert float(history[-1]["opposing_rp_force_norm"]) >= 0.0
    assert history[-1]["rp_reaction_definition"] == "static_physical_constraint_residual"
    assert "rp1_hht_rp_force_norm" in history[-1]
    assert "rp1_dynamic_rp_force_norm" in history[-1]
    assert "rp1_inertia_rp_force_norm" in history[-1]
    assert "rp1_contact_rp_force_norm" in history[-1]
    assert "contact_multiplier_sum" in history[-1]
    assert "p95_equivalent_elastic_strain" in history[-1]
    assert float(history[-1]["p95_equivalent_elastic_strain"]) >= 0.0
    assert "p95_von_mises_nodeavg" in history[-1]
    assert "p95_equivalent_elastic_strain_nodeavg" in history[-1]
    assert float(history[-1]["p95_von_mises_nodeavg"]) >= 0.0
    assert float(history[-1]["p95_equivalent_elastic_strain_nodeavg"]) >= 0.0


def test_cropped_gear_hard_contact_supports_surface_patch_constraint_averaging() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        max_iterations=4,
        constraint_averaging="surface_patch",
        automatic_increment=False,
    )

    assert len(history) == 1
    assert summary["constraint_averaging"] == "surface_patch"
    assert int(history[-1]["hard_constraints"]) <= int(history[-1]["hard_contact_samples"])
    assert int(history[-1]["hard_active_set_converged"]) == 1


def test_cropped_gear_hard_contact_supports_element_pressure_smoothing() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        max_iterations=4,
        hard_enforcement="element_pressure_smoothing",
        pressure_smoothing_factor=4.0,
        automatic_increment=False,
    )

    assert len(history) == 1
    assert summary["hard_enforcement"] == "element_pressure_smoothing"
    assert float(summary["effective_hard_pressure_stiffness"]) > 5.0e9
    assert float(summary["contact_patch_min_edge_length"]) > 0.0


def test_cropped_gear_hard_contact_supports_abaqus_standard_penalty() -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)

    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=1.0e-3,
        dt=1.0e-3,
        target_overclosure=1.0e-5,
        rotation_rate_z=0.0,
        max_iterations=4,
        hard_enforcement="abaqus_standard_penalty",
        automatic_increment=False,
    )

    assert len(history) == 1
    assert summary["hard_enforcement"] == "abaqus_standard_penalty"
    assert float(summary["effective_hard_pressure_stiffness"]) > 5.0e9
    assert int(summary["accepted_increment_count"]) == 1


def test_cropped_gear_abaqus_deck_can_use_hard_contact(tmp_path) -> None:
    model = parse_gear_input(DEFAULT_SOURCE)
    pair = build_cropped_pair(model, faces_per_body=3, expansion_rings=0)
    path = tmp_path / "cropped_gear_hard_alignment.inp"

    _write_abaqus_alignment_deck(
        path,
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=5.0e9,
        duration=2.0e-3,
        dt=1.0e-3,
        target_overclosure=3.0e-4,
        rotation_rate_z=2.0,
        contact_mode="hard",
    )

    text = path.read_text(encoding="ascii")
    assert "*Surface Interaction, name=SFC_HARD" in text
    assert "*Surface Behavior, pressure-overclosure=HARD" in text
    assert "interaction=SFC_HARD, type=SURFACE TO SURFACE" in text
    assert "pressure-overclosure=LINEAR" not in text
