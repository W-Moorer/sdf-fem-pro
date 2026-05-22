"""Cropped gear implicit TET4 + Lagrangian-SDF contact alignment runner.

This validation-only script builds a small contact patch from the supplied
flexible gear Abaqus input deck, then solves the cropped two-body model with
SFC's internal TET4 HHT dynamics and Lagrangian-SDF surface-contact tangent.

The Abaqus deck is used only as mesh/model input and as an optional external
reference deck writer.  The SFC solve does not read ODB, XLSX, or Abaqus
assembled matrices.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, cg, splu
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact.hard_contact import (  # noqa: E402
    hard_contact_gap_jacobian_from_samples,
    solve_linear_hard_contact_with_dirichlet,
    solve_linear_hard_contact_with_dirichlet_sparse,
)
from sfc.contact.lagrangian_surface_contact import LagrangianSDFSurfaceContactGeometry  # noqa: E402
from sfc.fem.calculix_aligned import (  # noqa: E402
    ContactResponse,
    InternalResponse,
    MechanicsModel,
    MechanicsState,
    assemble_contact_response,
    stvk_internal_response,
)
from sfc.fem.calculix_aligned import (  # noqa: E402
    ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
    calculix_dynamic_predictor,
    hht_newmark_parameters,
)
from sfc.fem.constraints import free_dofs, project_fixed_dofs  # noqa: E402
from sfc.fem.implicit_dirichlet import hht_step_dirichlet, initial_state_dirichlet  # noqa: E402
from sfc.fem.rp_mpc import RigidHubMPC, build_rigid_hub_reduced_assembly, merge_dirichlet_conditions  # noqa: E402
from sfc.sdf.material_sdf import MaterialSDF  # noqa: E402
from validation.run_flexible_gear_explicit_sdf_comparison import (  # noqa: E402
    DEFAULT_SOURCE,
    GearInputModel,
    GearMesh,
    parse_gear_input,
)
from validation.vtk_frame_series import write_pvd  # noqa: E402

Row = dict[str, Any]

DEFAULT_OUT_DIR = ROOT / "results" / "flexible_gear_implicit_lagrangian_sdf"
DEFAULT_PRESSURE_STIFFNESS = 5.0e9


def _assemble_contact_response_force_only(samples: Iterable[Any], n_nodes: int) -> ContactResponse:
    """Assemble penalty contact force/diagnostics without the tangent blocks."""

    force = np.zeros((int(n_nodes), 3), dtype=float)
    min_gap = np.inf
    max_penetration = 0.0
    active_count = 0
    normal_force = 0.0
    energy = 0.0
    for sample in samples:
        gap = float(sample.gap)
        min_gap = min(min_gap, gap)
        penetration = max(-gap, 0.0)
        if penetration <= 0.0:
            continue
        node_ids = np.asarray(sample.node_ids, dtype=np.int64)
        weights = np.asarray(sample.shape_weights, dtype=float)
        normal = np.asarray(sample.normal, dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-30)
        area = float(sample.area)
        stiffness = float(sample.stiffness)
        lam = stiffness * area * penetration
        master_ids = (
            np.asarray(sample.master_node_ids, dtype=np.int64)
            if sample.master_node_ids is not None
            else np.empty((0,), dtype=np.int64)
        )
        master_weights = (
            np.asarray(sample.master_shape_weights, dtype=float)
            if sample.master_shape_weights is not None
            else np.empty((0,), dtype=float)
        )
        if master_ids.size != master_weights.size:
            raise ValueError("master_node_ids and master_shape_weights must have the same length")
        active_count += 1
        normal_force += lam
        max_penetration = max(max_penetration, penetration)
        energy += 0.5 * stiffness * area * penetration * penetration
        for local, node in enumerate(node_ids):
            force[int(node)] += weights[local] * lam * normal
        for local, node in enumerate(master_ids):
            force[int(node)] -= master_weights[local] * lam * normal
    if not np.isfinite(min_gap):
        min_gap = 0.0
    tangent = csr_matrix((3 * int(n_nodes), 3 * int(n_nodes)), dtype=float)
    return ContactResponse(force, tangent, float(min_gap), float(max_penetration), active_count, float(normal_force), float(energy))


@dataclass(frozen=True, slots=True)
class CroppedGearPatch:
    name: str
    nodes: np.ndarray
    elements: np.ndarray
    contact_faces: np.ndarray
    support_nodes: np.ndarray
    surface_entries: tuple[tuple[int, str], ...]
    element_labels: np.ndarray
    rp: np.ndarray


@dataclass(frozen=True, slots=True)
class CroppedGearPair:
    gear1: CroppedGearPatch
    gear2: CroppedGearPatch
    drive_direction: np.ndarray
    initial_patch_gap: float


def _write_csv(path: Path, rows: list[Row], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _elastic_strain_norm_from_stress(stress: np.ndarray, *, young: float, poisson: float) -> np.ndarray:
    """Return small elastic strain norms recovered from Cauchy stress.

    Abaqus `LE` is the strain output used by the external reference deck.  The
    SFC material model stores Green strain for mechanics, so for validation
    diagnostics we recover the corresponding linear-elastic strain from the
    Cauchy stress before comparing to Abaqus `LE`.
    """

    values = np.asarray(stress, dtype=float)
    if values.size == 0:
        return np.empty(0, dtype=float)
    identity = np.eye(3)
    out = np.empty(values.shape[0], dtype=float)
    for idx, sigma in enumerate(values):
        trace = float(np.trace(sigma))
        eps = ((1.0 + float(poisson)) / float(young)) * sigma - (float(poisson) / float(young)) * trace * identity
        out[idx] = float(np.linalg.norm(eps))
    return out


def _equivalent_elastic_strain_from_mises(von_mises_values: np.ndarray, *, young: float, poisson: float) -> np.ndarray:
    vm = np.asarray(von_mises_values, dtype=float).reshape(-1)
    shear = float(young) / (2.0 * (1.0 + float(poisson)))
    return vm / max(3.0 * shear, 1.0e-30)


def _linear_reference_internal_response(
    model: MechanicsModel,
    x_current: np.ndarray,
    reference_tangent: Any,
) -> InternalResponse:
    """Return a fast small-strain TET4 response for linear-elastic decks."""

    displacement = np.asarray(x_current, dtype=float) - np.asarray(model.X, dtype=float)
    element_u = displacement[np.asarray(model.elements, dtype=np.int64)]
    grad_u = np.einsum("eai,eaj->eij", element_u, model.shape_grads, optimize=True)
    strain = 0.5 * (grad_u + np.swapaxes(grad_u, 1, 2))
    lam, mu = _lame_parameters_local(model.E, model.nu)
    identity = np.eye(3, dtype=float)
    trace = np.trace(strain, axis1=1, axis2=2)
    stress = lam * trace[:, None, None] * identity + 2.0 * mu * strain
    vm = _von_mises_local(stress)
    u_flat = displacement.reshape(-1)
    force = np.asarray(reference_tangent @ u_flat, dtype=float).reshape((-1, 3))
    energy = 0.5 * float(u_flat @ np.asarray(force, dtype=float).reshape(-1))
    empty = csr_matrix((model.n_dofs, model.n_dofs), dtype=float)
    return InternalResponse(force, strain, stress, vm, energy, empty, empty)


def _write_sfc_tet4_vtk_frame(
    path: Path,
    *,
    model: MechanicsModel,
    state: MechanicsState,
    internal: InternalResponse,
    element_object_ids: np.ndarray,
    time_value: float,
    frame_index: int,
    include_tensors: bool = True,
) -> Row:
    """Write one SFC TET4 state as a legacy VTK unstructured grid."""

    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(state.x, dtype=float)
    reference = np.asarray(model.X, dtype=float)
    displacement = points - reference
    displacement_norm = np.linalg.norm(displacement, axis=1)
    velocity = np.asarray(state.v, dtype=float)
    elements = np.asarray(model.elements, dtype=np.int64)
    object_ids = np.asarray(element_object_ids, dtype=np.int64).reshape(-1)
    if object_ids.shape != (elements.shape[0],):
        raise ValueError("element_object_ids must match element count")
    stress = _tensor_field_or_zeros(internal.stress, elements.shape[0])
    strain = _tensor_field_or_zeros(internal.strain, elements.shape[0])
    von_mises = np.asarray(internal.von_mises, dtype=float).reshape(-1)
    if von_mises.shape != (elements.shape[0],):
        von_mises = _von_mises_local(stress)
    strain_norm = np.linalg.norm(strain, axis=(1, 2))
    equivalent_strain = _equivalent_elastic_strain_from_mises(von_mises, young=model.E, poisson=model.nu)
    von_mises_nodeavg = _node_average_cell_scalar(von_mises, elements, points.shape[0])
    strain_norm_nodeavg = _node_average_cell_scalar(strain_norm, elements, points.shape[0])
    equivalent_strain_nodeavg = _node_average_cell_scalar(equivalent_strain, elements, points.shape[0])
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"SFC Lagrangian-SDF full gear frame {frame_index} time={float(time_value):.12g}\n")
        handle.write("ASCII\n")
        handle.write("DATASET UNSTRUCTURED_GRID\n")
        handle.write(f"POINTS {points.shape[0]} float\n")
        for x, y, z in points:
            handle.write(f"{x:.9e} {y:.9e} {z:.9e}\n")
        handle.write(f"CELLS {elements.shape[0]} {elements.shape[0] * 5}\n")
        for element in elements:
            handle.write("4 " + " ".join(str(int(node)) for node in element) + "\n")
        handle.write(f"CELL_TYPES {elements.shape[0]}\n")
        handle.write(("10\n" * elements.shape[0]))
        handle.write(f"POINT_DATA {points.shape[0]}\n")
        handle.write("VECTORS U float\n")
        for ux, uy, uz in displacement:
            handle.write(f"{ux:.9e} {uy:.9e} {uz:.9e}\n")
        handle.write("VECTORS V float\n")
        for vx, vy, vz in velocity:
            handle.write(f"{vx:.9e} {vy:.9e} {vz:.9e}\n")
        handle.write("SCALARS displacement_magnitude float 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for value in displacement_norm:
            handle.write(f"{float(value):.9e}\n")
        for name, values in (
            ("von_mises_nodeavg", von_mises_nodeavg),
            ("strain_norm_nodeavg", strain_norm_nodeavg),
            ("equivalent_elastic_strain_nodeavg", equivalent_strain_nodeavg),
        ):
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in values:
                handle.write(f"{float(value):.9e}\n")
        handle.write(f"CELL_DATA {elements.shape[0]}\n")
        handle.write("SCALARS object_id int 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for value in object_ids:
            handle.write(f"{int(value)}\n")
        for name, values in (
            ("von_mises", von_mises),
            ("strain_norm", strain_norm),
            ("equivalent_elastic_strain", equivalent_strain),
        ):
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in values:
                handle.write(f"{float(value):.9e}\n")
        if include_tensors:
            for name, tensor_values in (("E", strain), ("LE", strain), ("S", stress)):
                handle.write(f"TENSORS {name} float\n")
                for tensor in tensor_values:
                    for row in tensor:
                        handle.write(f"{row[0]:.9e} {row[1]:.9e} {row[2]:.9e}\n")
    return {
        "frame": int(frame_index),
        "time": float(time_value),
        "vtk_file": path.name,
        "node_count": int(points.shape[0]),
        "element_count": int(elements.shape[0]),
        "max_displacement_magnitude": float(np.max(displacement_norm)) if displacement_norm.size else 0.0,
        "max_von_mises": float(np.max(von_mises)) if von_mises.size else 0.0,
        "max_strain_norm": float(np.max(strain_norm)) if strain_norm.size else 0.0,
        "max_von_mises_nodeavg": float(np.max(von_mises_nodeavg)) if von_mises_nodeavg.size else 0.0,
        "max_strain_norm_nodeavg": float(np.max(strain_norm_nodeavg)) if strain_norm_nodeavg.size else 0.0,
    }


def _node_average_cell_scalar(cell_values: np.ndarray, cells: np.ndarray, node_count: int) -> np.ndarray:
    """Return incident-cell averaged nodal values for visualization only."""

    values = np.asarray(cell_values, dtype=float).reshape(-1)
    conn = np.asarray(cells, dtype=np.int64)
    out = np.zeros(int(node_count), dtype=float)
    counts = np.zeros(int(node_count), dtype=float)
    if values.shape[0] != conn.shape[0]:
        return out
    for cell_value, cell in zip(values, conn, strict=True):
        out[cell] += float(cell_value)
        counts[cell] += 1.0
    mask = counts > 0.0
    out[mask] /= counts[mask]
    return out


def _tensor_field_or_zeros(values: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape == (int(count), 3, 3):
        return array
    return np.zeros((int(count), 3, 3), dtype=float)


def _lame_parameters_local(young: float, poisson: float) -> tuple[float, float]:
    e = float(young)
    nu = float(poisson)
    mu = e / (2.0 * (1.0 + nu))
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def _von_mises_local(stress: np.ndarray) -> np.ndarray:
    values = np.asarray(stress, dtype=float)
    sx = values[:, 0, 0]
    sy = values[:, 1, 1]
    sz = values[:, 2, 2]
    txy = values[:, 0, 1]
    tyz = values[:, 1, 2]
    txz = values[:, 0, 2]
    mises = 0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy * txy + tyz * tyz + txz * txz)
    return np.sqrt(np.maximum(mises, 0.0))


def _face_centroids(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.mean(nodes[np.asarray(faces, dtype=np.int64)], axis=1)


def _triangle_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = nodes[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    normals[norms > 0.0] /= norms[norms > 0.0, None]
    return normals


def _orient_faces_toward(nodes: np.ndarray, faces: np.ndarray, direction: np.ndarray) -> np.ndarray:
    out = np.asarray(faces, dtype=np.int64).copy()
    normals = _triangle_normals(nodes, out)
    d = np.asarray(direction, dtype=float)
    d /= max(float(np.linalg.norm(d)), 1.0e-30)
    flip = normals @ d < 0.0
    out[flip, 1], out[flip, 2] = out[flip, 2].copy(), out[flip, 1].copy()
    return out


def _select_patch_indices(model: GearInputModel, *, faces_per_body: int) -> tuple[np.ndarray, np.ndarray, float]:
    centroids1 = _face_centroids(model.gear1.nodes, model.gear1_contact_faces)
    centroids2 = _face_centroids(model.gear2.nodes, model.gear2_contact_faces)
    tree2 = cKDTree(centroids2)
    distances, ids2 = tree2.query(centroids1, k=1)
    seed1 = int(np.argmin(distances))
    seed2 = int(ids2[seed1])
    n = max(1, int(faces_per_body))
    tree1 = cKDTree(centroids1)
    ids1 = np.asarray(tree1.query(centroids1[seed1], k=min(n, centroids1.shape[0]))[1], dtype=np.int64).ravel()
    ids2 = np.asarray(tree2.query(centroids2[seed2], k=min(n, centroids2.shape[0]))[1], dtype=np.int64).ravel()
    return ids1, ids2, float(distances[seed1])


def _crop_patch(
    *,
    name: str,
    mesh: GearMesh,
    contact_faces: np.ndarray,
    surface_entries: tuple[tuple[int, str], ...],
    selected_surface_ids: np.ndarray,
    support_direction: np.ndarray,
    support_high_side: bool,
    rp: np.ndarray,
    expansion_rings: int,
) -> CroppedGearPatch:
    selected_surface_ids = np.asarray(selected_surface_ids, dtype=np.int64).ravel()
    if selected_surface_ids.size == 0:
        raise ValueError("selected_surface_ids must not be empty")
    node_set = set(int(v) for v in contact_faces[selected_surface_ids].ravel())
    for _ in range(max(0, int(expansion_rings))):
        mask = np.asarray([any(int(node) in node_set for node in element) for element in mesh.elements], dtype=bool)
        for element in mesh.elements[mask]:
            node_set.update(int(v) for v in element)
    element_mask = np.asarray([any(int(node) in node_set for node in element) for element in mesh.elements], dtype=bool)
    selected_elements = mesh.elements[element_mask]
    selected_element_labels = mesh.element_labels[element_mask]
    node_ids = np.asarray(sorted(set(int(v) for v in selected_elements.ravel())), dtype=np.int64)
    old_to_new = {int(old): idx for idx, old in enumerate(node_ids)}
    label_to_new_element = {int(label): idx + 1 for idx, label in enumerate(selected_element_labels)}
    nodes = mesh.nodes[node_ids]
    elements = np.asarray([[old_to_new[int(v)] for v in element] for element in selected_elements], dtype=np.int64)
    selected_faces_old = contact_faces[selected_surface_ids]
    contact_faces_new = np.asarray([[old_to_new[int(v)] for v in face] for face in selected_faces_old], dtype=np.int64)
    entries = tuple(
        (label_to_new_element[int(surface_entries[int(idx)][0])], surface_entries[int(idx)][1])
        for idx in selected_surface_ids
        if int(surface_entries[int(idx)][0]) in label_to_new_element
    )
    if not entries:
        entries = tuple((idx + 1, "S1") for idx in range(min(contact_faces_new.shape[0], elements.shape[0])))
    face_nodes = np.unique(contact_faces_new.ravel())
    center = np.mean(nodes[face_nodes], axis=0)
    direction = np.asarray(support_direction, dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-30)
    scores = (nodes - center[None, :]) @ direction
    pct = 82.0 if support_high_side else 18.0
    threshold = float(np.percentile(scores, pct))
    support = np.nonzero(scores >= threshold if support_high_side else scores <= threshold)[0]
    if support.size < 4:
        support = np.argsort(scores)[-4:] if support_high_side else np.argsort(scores)[:4]
    return CroppedGearPatch(
        name=name,
        nodes=nodes,
        elements=elements,
        contact_faces=contact_faces_new,
        support_nodes=np.asarray(np.unique(support), dtype=np.int64),
        surface_entries=entries,
        element_labels=np.arange(1, elements.shape[0] + 1, dtype=np.int64),
        rp=np.asarray(rp, dtype=float),
    )


def build_cropped_pair(model: GearInputModel, *, faces_per_body: int, expansion_rings: int) -> CroppedGearPair:
    ids1, ids2, initial_gap = _select_patch_indices(model, faces_per_body=faces_per_body)
    c1 = np.mean(_face_centroids(model.gear1.nodes, model.gear1_contact_faces[ids1]), axis=0)
    c2 = np.mean(_face_centroids(model.gear2.nodes, model.gear2_contact_faces[ids2]), axis=0)
    drive = c2 - c1
    drive /= max(float(np.linalg.norm(drive)), 1.0e-30)
    gear1 = _crop_patch(
        name="gear1",
        mesh=model.gear1,
        contact_faces=model.gear1_contact_faces,
        surface_entries=model.gear1_surface_entries,
        selected_surface_ids=ids1,
        support_direction=drive,
        support_high_side=False,
        rp=model.rp1,
        expansion_rings=expansion_rings,
    )
    gear2 = _crop_patch(
        name="gear2",
        mesh=model.gear2,
        contact_faces=model.gear2_contact_faces,
        surface_entries=model.gear2_surface_entries,
        selected_surface_ids=ids2,
        support_direction=drive,
        support_high_side=True,
        rp=model.rp2,
        expansion_rings=expansion_rings,
    )
    gear1_faces = _orient_faces_toward(gear1.nodes, gear1.contact_faces, drive)
    gear2_faces = _orient_faces_toward(gear2.nodes, gear2.contact_faces, -drive)
    gear1 = CroppedGearPatch(gear1.name, gear1.nodes, gear1.elements, gear1_faces, gear1.support_nodes, gear1.surface_entries, gear1.element_labels, gear1.rp)
    gear2 = CroppedGearPatch(gear2.name, gear2.nodes, gear2.elements, gear2_faces, gear2.support_nodes, gear2.surface_entries, gear2.element_labels, gear2.rp)
    return CroppedGearPair(gear1, gear2, drive, initial_gap)


def _fixed_conditions_for_pair(
    pair: CroppedGearPair,
    *,
    closure: float,
    rotation_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    hub1 = RigidHubMPC(pair.gear1.support_nodes, pair.gear1.nodes, pair.gear1.rp)
    hub2 = RigidHubMPC(pair.gear2.support_nodes, pair.gear2.nodes, pair.gear2.rp)
    d1, v1 = hub1.dirichlet_dofs_and_values(
        translation=closure * pair.drive_direction,
        rotation=(0.0, 0.0, rotation_z),
    )
    d2, v2 = hub2.dirichlet_dofs_and_values()
    d2 = d2 + 3 * pair.gear1.nodes.shape[0]
    return merge_dirichlet_conditions((d1, v1), (d2, v2))


def _penalty_rhs_balance(model: MechanicsModel, internal: Any, contact_response: Any) -> np.ndarray:
    """Return external-minus-internal balance for the zero-gravity gear cases."""

    _ = model
    return -np.asarray(internal.force, dtype=float).reshape(-1) + np.asarray(contact_response.force, dtype=float).reshape(-1)


def _penalty_history_row(
    *,
    time_value: float,
    closure: float,
    rotation: float,
    state: MechanicsState,
    model: MechanicsModel,
    internal: Any,
    contact_response: Any,
    young: float,
    poisson: float,
    newton_iterations: int,
    residual_norm: float,
    solver: str,
) -> Row:
    disp = state.x - model.X
    strain_norm = np.linalg.norm(internal.strain, axis=(1, 2)) if internal.strain.size else np.empty(0, dtype=float)
    elastic_strain_norm = _elastic_strain_norm_from_stress(internal.stress, young=young, poisson=poisson)
    equivalent_elastic_strain = _equivalent_elastic_strain_from_mises(internal.von_mises, young=young, poisson=poisson)
    return {
        "time": float(time_value),
        "closure": float(closure),
        "rotation_z": float(rotation),
        "active_contact_samples": int(contact_response.active_count),
        "min_gap": float(contact_response.min_gap),
        "normal_force": float(contact_response.normal_force),
        "max_displacement_norm": float(np.max(np.linalg.norm(disp, axis=1))),
        "p95_von_mises": float(np.percentile(internal.von_mises, 95.0)) if internal.von_mises.size else 0.0,
        "max_von_mises": float(np.max(internal.von_mises)) if internal.von_mises.size else 0.0,
        "p95_strain_norm": float(np.percentile(strain_norm, 95.0)) if strain_norm.size else 0.0,
        "max_strain_norm": float(np.max(strain_norm)) if strain_norm.size else 0.0,
        "p95_elastic_strain_norm": float(np.percentile(elastic_strain_norm, 95.0)) if elastic_strain_norm.size else 0.0,
        "max_elastic_strain_norm": float(np.max(elastic_strain_norm)) if elastic_strain_norm.size else 0.0,
        "p95_equivalent_elastic_strain": float(np.percentile(equivalent_elastic_strain, 95.0)) if equivalent_elastic_strain.size else 0.0,
        "max_equivalent_elastic_strain": float(np.max(equivalent_elastic_strain)) if equivalent_elastic_strain.size else 0.0,
        "strain_energy": float(internal.strain_energy),
        "newton_iterations": int(newton_iterations),
        "newton_residual_norm": float(residual_norm),
        "penalty_solver": str(solver),
    }


def _solve_penalty_low_rank_correction(
    *,
    base_matrix_free: Any,
    base_lu_cache: list[Any],
    residual_free: np.ndarray,
    samples: list[Any],
    free: np.ndarray,
    n_total_dofs: int,
    equilibrium_scale: float,
    tolerance: float,
    use_cg: bool = False,
) -> np.ndarray:
    """Solve a Newton correction with the active penalty tangent as low rank."""

    rhs = -np.asarray(residual_free, dtype=float).reshape(-1)
    if rhs.size == 0:
        return np.empty(0, dtype=float)
    active_samples = [sample for sample in samples if float(sample.gap) < 0.0]
    j_free = np.empty((0, rhs.size), dtype=float)
    tangent_scale = np.asarray(
        [float(equilibrium_scale) * float(sample.stiffness) * float(sample.area) for sample in active_samples],
        dtype=float,
    )
    if active_samples:
        _gaps, gap_jacobian = hard_contact_gap_jacobian_from_samples(active_samples, n_total_dofs=int(n_total_dofs))
        j_free = np.asarray(gap_jacobian[:, np.asarray(free, dtype=np.int64)], dtype=float)
        positive = tangent_scale > 0.0
        j_free = j_free[positive]
        tangent_scale = tangent_scale[positive]
    if bool(use_cg):
        cg_solution = _solve_penalty_cg_correction(
            base_matrix_free=base_matrix_free,
            rhs=rhs,
            j_free=j_free,
            tangent_scale=tangent_scale,
            tolerance=tolerance,
        )
        if cg_solution is not None:
            return cg_solution
    if base_lu_cache[0] is None:
        base_lu_cache[0] = splu(base_matrix_free.tocsc())
    base_lu = base_lu_cache[0]
    base_solution = np.asarray(base_lu.solve(rhs), dtype=float).reshape(-1)
    if not active_samples or j_free.size == 0:
        return base_solution
    influence = np.asarray(base_lu.solve(j_free.T), dtype=float)
    schur = np.diag(1.0 / tangent_scale) + j_free @ influence
    try:
        contact_coeff = np.linalg.solve(schur, j_free @ base_solution)
    except np.linalg.LinAlgError:
        contact_coeff, *_ = np.linalg.lstsq(schur, j_free @ base_solution, rcond=None)
    return base_solution - influence @ contact_coeff


def _solve_reduced_penalty_low_rank_correction(
    *,
    base_matrix_free: Any,
    base_lu_cache: list[Any],
    residual_free: np.ndarray,
    samples: list[Any],
    transformation: Any,
    free: np.ndarray,
    n_total_dofs: int,
    equilibrium_scale: float,
) -> np.ndarray:
    """Solve a reduced-coordinate Newton correction with active contact tangent."""

    rhs = -np.asarray(residual_free, dtype=float).reshape(-1)
    if rhs.size == 0:
        return np.empty(0, dtype=float)
    active_samples = [sample for sample in samples if float(sample.gap) < 0.0]
    j_free = np.empty((0, rhs.size), dtype=float)
    tangent_scale = np.asarray(
        [float(equilibrium_scale) * float(sample.stiffness) * float(sample.area) for sample in active_samples],
        dtype=float,
    )
    if active_samples:
        _gaps, gap_jacobian = hard_contact_gap_jacobian_from_samples(active_samples, n_total_dofs=int(n_total_dofs))
        j_reduced = gap_jacobian @ transformation
        j_free = np.asarray(j_reduced[:, np.asarray(free, dtype=np.int64)], dtype=float)
        positive = tangent_scale > 0.0
        j_free = j_free[positive]
        tangent_scale = tangent_scale[positive]
    if base_lu_cache[0] is None:
        base_lu_cache[0] = splu(base_matrix_free.tocsc())
    base_lu = base_lu_cache[0]
    base_solution = np.asarray(base_lu.solve(rhs), dtype=float).reshape(-1)
    if not active_samples or j_free.size == 0:
        return base_solution
    influence = np.asarray(base_lu.solve(j_free.T), dtype=float)
    schur = np.diag(1.0 / tangent_scale) + j_free @ influence
    try:
        contact_coeff = np.linalg.solve(schur, j_free @ base_solution)
    except np.linalg.LinAlgError:
        contact_coeff, *_ = np.linalg.lstsq(schur, j_free @ base_solution, rcond=None)
    return base_solution - influence @ contact_coeff


def _active_gap_jacobian_sparse(samples: list[Any], *, n_total_dofs: int) -> Any:
    """Build a sparse gap Jacobian for active contact samples only."""

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    active_row = 0
    for sample in samples:
        if float(sample.gap) >= 0.0:
            continue
        normal = np.asarray(sample.normal, dtype=float).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 0.0:
            raise ValueError("sample normal must be nonzero")
        normal = normal / normal_norm
        for sign, node_attr, weight_attr in (
            (1.0, "node_ids", "shape_weights"),
            (-1.0, "master_node_ids", "master_shape_weights"),
        ):
            node_values = getattr(sample, node_attr, None)
            weight_values = getattr(sample, weight_attr, None)
            if node_values is None or weight_values is None:
                continue
            nodes = np.asarray(node_values, dtype=np.int64).reshape(-1)
            weights = np.asarray(weight_values, dtype=float).reshape(-1)
            if nodes.shape != weights.shape:
                raise ValueError("contact sample nodes and weights must match")
            for node, weight in zip(nodes, weights, strict=True):
                base = 3 * int(node)
                for component in range(3):
                    value = float(sign) * float(weight) * float(normal[component])
                    if value != 0.0:
                        rows.append(active_row)
                        cols.append(base + component)
                        data.append(value)
        active_row += 1
    return coo_matrix((data, (rows, cols)), shape=(active_row, int(n_total_dofs))).tocsr()


def _active_reduced_gap_jacobian_sparse(
    samples: list[Any],
    *,
    transformation: Any,
    free: np.ndarray,
) -> Any:
    """Build active contact gap Jacobian directly in reduced free DOFs."""

    T = transformation.tocsr()
    free_cols = np.asarray(free, dtype=np.int64).reshape(-1)
    reduced_to_free = np.full(T.shape[1], -1, dtype=np.int64)
    reduced_to_free[free_cols] = np.arange(free_cols.size, dtype=np.int64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    active_row = 0
    for sample in samples:
        if float(sample.gap) >= 0.0:
            continue
        normal = np.asarray(sample.normal, dtype=float).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 0.0:
            raise ValueError("sample normal must be nonzero")
        normal = normal / normal_norm
        for sign, node_attr, weight_attr in (
            (1.0, "node_ids", "shape_weights"),
            (-1.0, "master_node_ids", "master_shape_weights"),
        ):
            node_values = getattr(sample, node_attr, None)
            weight_values = getattr(sample, weight_attr, None)
            if node_values is None or weight_values is None:
                continue
            nodes = np.asarray(node_values, dtype=np.int64).reshape(-1)
            weights = np.asarray(weight_values, dtype=float).reshape(-1)
            if nodes.shape != weights.shape:
                raise ValueError("contact sample nodes and weights must match")
            for node, weight in zip(nodes, weights, strict=True):
                for component in range(3):
                    coeff = float(sign) * float(weight) * float(normal[component])
                    if coeff == 0.0:
                        continue
                    full_row = 3 * int(node) + component
                    start = int(T.indptr[full_row])
                    stop = int(T.indptr[full_row + 1])
                    for ptr in range(start, stop):
                        free_col = int(reduced_to_free[int(T.indices[ptr])])
                        if free_col >= 0:
                            rows.append(active_row)
                            cols.append(free_col)
                            data.append(coeff * float(T.data[ptr]))
        active_row += 1
    return coo_matrix((data, (rows, cols)), shape=(active_row, free_cols.size)).tocsr()


def _solve_penalty_cg_correction_sparse(
    *,
    base_matrix_free: Any,
    rhs: np.ndarray,
    j_free: Any,
    tangent_scale: np.ndarray,
    tolerance: float,
) -> np.ndarray | None:
    """Solve ``(K + J.T W J) x = rhs`` with sparse matrix-free CG."""

    b = np.asarray(rhs, dtype=float).reshape(-1)
    j = j_free.tocsr()
    scale = np.asarray(tangent_scale, dtype=float).reshape(-1)
    if j.shape[0] != scale.shape[0] or j.shape[1] != b.size:
        return None
    base = base_matrix_free.tocsr()

    def matvec(value: np.ndarray) -> np.ndarray:
        vec = np.asarray(value, dtype=float).reshape(-1)
        out = np.asarray(base @ vec, dtype=float)
        if j.shape[0]:
            out += np.asarray(j.T @ (scale * np.asarray(j @ vec, dtype=float)), dtype=float)
        return out

    diag = np.asarray(base.diagonal(), dtype=float).reshape(-1)
    if j.shape[0]:
        contact_diag = np.asarray(j.multiply(j).T @ scale, dtype=float).reshape(-1)
        diag = diag + contact_diag
    safe_diag = np.where(np.abs(diag) > 1.0e-30, diag, 1.0)
    preconditioner = LinearOperator((b.size, b.size), matvec=lambda value: np.asarray(value, dtype=float) / safe_diag, dtype=float)
    operator = LinearOperator((b.size, b.size), matvec=matvec, dtype=float)
    rtol = min(1.0e-8, max(1.0e-11, float(tolerance) * 10.0))
    atol = max(float(tolerance) * max(1.0, float(np.linalg.norm(b))) * 1.0e-2, 1.0e-10)
    try:
        solution, info = cg(
            operator,
            b,
            rtol=rtol,
            atol=atol,
            maxiter=max(200, min(2000, b.size // 20)),
            M=preconditioner,
        )
    except Exception:
        return None
    if info != 0:
        return None
    residual = matvec(solution) - b
    allowed = atol + rtol * max(1.0, float(np.linalg.norm(b)))
    if float(np.linalg.norm(residual)) > max(50.0 * allowed, 1.0e-7):
        return None
    return np.asarray(solution, dtype=float).reshape(-1)


def _solve_reduced_penalty_sparse_cg_correction(
    *,
    base_matrix_free: Any,
    residual_free: np.ndarray,
    samples: list[Any],
    transformation: Any,
    free: np.ndarray,
    n_total_dofs: int,
    equilibrium_scale: float,
    tolerance: float,
) -> np.ndarray | None:
    """Solve the reduced contact tangent with sparse matrix-free CG."""

    rhs = -np.asarray(residual_free, dtype=float).reshape(-1)
    if rhs.size == 0:
        return np.empty(0, dtype=float)
    active_samples = [sample for sample in samples if float(sample.gap) < 0.0]
    if not active_samples:
        try:
            return np.asarray(cg(base_matrix_free.tocsr(), rhs, rtol=1.0e-10, atol=1.0e-12, maxiter=1000)[0], dtype=float)
        except Exception:
            return None
    tangent_scale = np.asarray(
        [float(equilibrium_scale) * float(sample.stiffness) * float(sample.area) for sample in active_samples],
        dtype=float,
    )
    positive = tangent_scale > 0.0
    if not np.any(positive):
        return None
    _ = int(n_total_dofs)  # kept for API parity with the dense fallback.
    j_free = _active_reduced_gap_jacobian_sparse(active_samples, transformation=transformation, free=free)
    j_free = j_free[positive]
    tangent_scale = tangent_scale[positive]
    return _solve_penalty_cg_correction_sparse(
        base_matrix_free=base_matrix_free,
        rhs=rhs,
        j_free=j_free,
        tangent_scale=tangent_scale,
        tolerance=tolerance,
    )


def _solve_penalty_cg_correction(
    *,
    base_matrix_free: Any,
    rhs: np.ndarray,
    j_free: np.ndarray,
    tangent_scale: np.ndarray,
    tolerance: float,
) -> np.ndarray | None:
    """Solve the active penalty tangent system by matrix-free CG."""

    b = np.asarray(rhs, dtype=float).reshape(-1)
    j = np.asarray(j_free, dtype=float)
    scale = np.asarray(tangent_scale, dtype=float).reshape(-1)
    if j.shape[0] != scale.shape[0] or (j.ndim == 2 and j.shape[1] != b.size):
        return None
    base = base_matrix_free.tocsr()

    def matvec(value: np.ndarray) -> np.ndarray:
        vec = np.asarray(value, dtype=float).reshape(-1)
        out = np.asarray(base @ vec, dtype=float)
        if j.size:
            out += np.asarray(j.T @ (scale * (j @ vec)), dtype=float)
        return out

    diag = np.asarray(base.diagonal(), dtype=float).reshape(-1)
    if j.size:
        diag = diag + np.sum((j * j) * scale[:, None], axis=0)
    safe_diag = np.where(np.abs(diag) > 1.0e-30, diag, 1.0)
    preconditioner = LinearOperator((b.size, b.size), matvec=lambda value: np.asarray(value, dtype=float) / safe_diag, dtype=float)
    operator = LinearOperator((b.size, b.size), matvec=matvec, dtype=float)
    rtol = min(1.0e-8, max(1.0e-11, float(tolerance) * 10.0))
    atol = max(float(tolerance) * max(1.0, float(np.linalg.norm(b))) * 1.0e-2, 1.0e-10)
    try:
        solution, info = cg(
            operator,
            b,
            rtol=rtol,
            atol=atol,
            maxiter=max(200, min(2000, b.size // 20)),
            M=preconditioner,
        )
    except Exception:
        return None
    if info != 0:
        return None
    residual = matvec(solution) - b
    allowed = atol + rtol * max(1.0, float(np.linalg.norm(b)))
    if float(np.linalg.norm(residual)) > max(50.0 * allowed, 1.0e-7):
        return None
    return np.asarray(solution, dtype=float).reshape(-1)


def _solve_sfc_cropped_pair_penalty_modified_newton(
    pair: CroppedGearPair,
    *,
    young: float,
    poisson: float,
    density: float,
    pressure_stiffness: float,
    duration: float,
    dt: float,
    target_overclosure: float,
    rotation_rate_z: float,
    hht_alpha: float,
    max_iterations: int,
    tolerance: float,
    material_linearization: str,
    vtk_out_dir: Path | None = None,
    vtk_frame_stride: int = 1,
    vtk_stem: str = "sfc",
    vtk_include_tensors: bool = True,
) -> tuple[list[Row], Row]:
    """Solve linear penalty contact with exact residuals and a reused base tangent.

    The contact law and gap query are unchanged.  Only the Newton linearization
    is modified: each increment factors the HHT mass-plus-material tangent once
    and uses it for residual-correcting iterations.  If the correction does not
    converge, the increment falls back to the full tangent path.
    """

    n1 = pair.gear1.nodes.shape[0]
    X = np.vstack([pair.gear1.nodes, pair.gear2.nodes])
    elements = np.vstack([pair.gear1.elements, pair.gear2.elements + n1])
    model = MechanicsModel.from_tet4_mesh(X, elements, E=young, nu=poisson, density=density)
    master = MaterialSDF.from_triangle_surface(pair.gear2.nodes, pair.gear2.contact_faces)
    contact = LagrangianSDFSurfaceContactGeometry(
        pair.gear1.contact_faces,
        master,
        pair.gear2.nodes,
        pressure_stiffness=pressure_stiffness,
        slave_node_offset=0,
        master_node_offset=n1,
        quadrature="tri3",
        search_radius=max(2.5 * (pair.initial_patch_gap + target_overclosure), 1.0e-4),
    )
    fixed0, values0 = _fixed_conditions_for_pair(pair, closure=0.0, rotation_z=0.0)
    state, previous = initial_state_dirichlet(model, contact, gravity=0.0, fixed_dofs=fixed0, fixed_values=values0)
    rows: list[Row] = []
    steps = max(1, int(round(float(duration) / float(dt))))
    beta, gamma = hht_newmark_parameters(float(hht_alpha))
    scale = 1.0 + float(hht_alpha)
    material_mode = str(material_linearization).lower()
    if material_mode not in {"stvk", "reference_linear"}:
        raise ValueError("material_linearization must be 'stvk' or 'reference_linear'")
    reference_tangent = None
    reference_base_free = None
    reference_lu_cache: list[Any] | None = None
    if material_mode == "reference_linear":
        reference_internal = stvk_internal_response(model, model.X, assemble_tangent=True)
        reference_tangent = reference_internal.tangent.tocsr()
    vtk_enabled = vtk_out_dir is not None and int(vtk_frame_stride) > 0
    vtk_dir = Path(vtk_out_dir) if vtk_out_dir is not None else None
    vtk_datasets: list[tuple[float, Path]] = []
    vtk_manifest_rows: list[Row] = []
    element_object_ids = np.concatenate(
        [
            np.ones(pair.gear1.elements.shape[0], dtype=np.int64),
            np.full(pair.gear2.elements.shape[0], 2, dtype=np.int64),
        ]
    )
    if vtk_enabled and vtk_dir is not None:
        vtk_dir.mkdir(parents=True, exist_ok=True)
        for stale in vtk_dir.glob(f"{vtk_stem}_*.vtk"):
            stale.unlink()
        for stale in (vtk_dir / f"{vtk_stem}.pvd", vtk_dir / f"{vtk_stem}_manifest.csv"):
            if stale.exists():
                stale.unlink()
        if material_mode == "reference_linear" and reference_tangent is not None:
            initial_internal = _linear_reference_internal_response(model, state.x, reference_tangent)
        else:
            initial_internal = stvk_internal_response(model, state.x, assemble_tangent=False)
        frame_path = vtk_dir / f"{vtk_stem}_{0:04d}.vtk"
        vtk_manifest_rows.append(
            _write_sfc_tet4_vtk_frame(
                frame_path,
                model=model,
                state=state,
                internal=initial_internal,
                element_object_ids=element_object_ids,
                time_value=0.0,
                frame_index=0,
                include_tensors=vtk_include_tensors,
            )
        )
        vtk_datasets.append((0.0, frame_path))
    start = time.perf_counter()
    timing_base_tangent = 0.0
    timing_base_factor = 0.0
    timing_residual = 0.0
    timing_linear_solve = 0.0
    timing_contact_low_rank = 0.0
    timing_fallback = 0.0
    fallback_count = 0
    vtk_frame_index = 1
    for step in range(1, steps + 1):
        t = step * float(dt)
        ramp = t / max(float(duration), float(dt))
        closure = ramp * (pair.initial_patch_gap + float(target_overclosure))
        rotation = float(rotation_rate_z) * t
        fixed, values = _fixed_conditions_for_pair(pair, closure=closure, rotation_z=rotation)
        free = free_dofs(model.n_dofs, fixed)
        c0 = 1.0 / (beta * float(dt) * float(dt))
        u = (state.x - model.X).reshape(-1)
        v = state.v.reshape(-1)
        a = state.a.reshape(-1)
        u_pred, v_pred, _accold_after_prediction = calculix_dynamic_predictor(u, v, a, dt=float(dt), beta=beta, gamma=gamma)
        u_guess = project_fixed_dofs(u_pred, fixed, values)
        if material_mode == "reference_linear":
            if reference_tangent is None:
                raise RuntimeError("reference tangent was not initialized")
            if reference_base_free is None:
                t_section = time.perf_counter()
                base_matrix = (model.mass_matrix * c0 + reference_tangent * scale).tocsr()
                reference_base_free = base_matrix[free[:, None], free].tocsr() if free.size else None
                reference_lu_cache = [None]
                timing_base_tangent += time.perf_counter() - t_section
            base_matrix_free = reference_base_free
            base_lu_cache = reference_lu_cache if reference_lu_cache is not None else [None]
        else:
            t_section = time.perf_counter()
            base_internal = stvk_internal_response(model, model.X + u_guess.reshape((-1, 3)), assemble_tangent=True)
            base_matrix = (model.mass_matrix * c0 + base_internal.tangent * scale).tocsr()
            timing_base_tangent += time.perf_counter() - t_section
            t_section = time.perf_counter()
            base_matrix_free = base_matrix[free[:, None], free].tocsr() if free.size else None
            base_lu_cache = [None]
            timing_base_factor += time.perf_counter() - t_section
        converged = False
        residual_norm = np.inf
        iteration_count = 0
        if material_mode == "reference_linear" and reference_tangent is not None:
            last_internal = _linear_reference_internal_response(model, model.X + u_guess.reshape((-1, 3)), reference_tangent)
        else:
            last_internal = stvk_internal_response(model, model.X + u_guess.reshape((-1, 3)), assemble_tangent=False)
        last_contact = assemble_contact_response(contact.samples(model.X + u_guess.reshape((-1, 3))), model.n_nodes)
        for iteration in range(1, max(1, int(max_iterations)) + 1):
            u_guess = project_fixed_dofs(u_guess, fixed, values)
            x_guess = model.X + u_guess.reshape((-1, 3))
            t_section = time.perf_counter()
            if material_mode == "reference_linear":
                if reference_tangent is None:
                    raise RuntimeError("reference tangent was not initialized")
                last_internal = _linear_reference_internal_response(model, x_guess, reference_tangent)
                internal_force = np.asarray(last_internal.force, dtype=float).reshape(-1)
            else:
                last_internal = stvk_internal_response(model, x_guess, assemble_tangent=False)
                internal_force = np.asarray(last_internal.force, dtype=float).reshape(-1)
            samples = list(contact.samples(x_guess))
            last_contact = assemble_contact_response(samples, model.n_nodes)
            rhs_balance = -internal_force + np.asarray(last_contact.force, dtype=float).reshape(-1)
            a_guess = c0 * (u_guess - u_pred)
            dynamic_residual = np.asarray(model.mass_matrix @ a_guess, dtype=float) - scale * rhs_balance + float(hht_alpha) * previous
            residual_norm = float(np.linalg.norm(dynamic_residual[free])) if free.size else 0.0
            timing_residual += time.perf_counter() - t_section
            t_section = time.perf_counter()
            correction_free = (
                _solve_penalty_low_rank_correction(
                    base_matrix_free=base_matrix_free,
                    base_lu_cache=base_lu_cache,
                    residual_free=dynamic_residual[free],
                    samples=samples,
                    free=free,
                    n_total_dofs=model.n_dofs,
                    equilibrium_scale=scale,
                    tolerance=float(tolerance),
                )
                if free.size
                else np.empty(0, dtype=float)
            )
            timing_linear_solve += time.perf_counter() - t_section
            timing_contact_low_rank += time.perf_counter() - t_section
            u_guess[free] += correction_free
            iteration_count = iteration
            correction_norm = float(np.linalg.norm(correction_free))
            displacement_scale = max(1.0, float(np.linalg.norm(u_guess[free])) if free.size else 1.0)
            if correction_norm <= float(tolerance) * displacement_scale:
                converged = True
                break
        if not converged:
            fallback_count += 1
            t_section = time.perf_counter()
            state, previous, diagnostics = hht_step_dirichlet(
                model,
                state,
                previous,
                contact,
                dt=float(dt),
                gravity=0.0,
                fixed_dofs=fixed,
                fixed_values=values,
                alpha=float(hht_alpha),
                max_iterations=max_iterations,
                tolerance=float(tolerance),
                acceptance_policy="relative_correction",
            )
            timing_fallback += time.perf_counter() - t_section
            internal = diagnostics.internal
            contact_response = diagnostics.contact
            row = _penalty_history_row(
                time_value=t,
                closure=closure,
                rotation=rotation,
                state=state,
                model=model,
                internal=internal,
                contact_response=contact_response,
                young=young,
                poisson=poisson,
                newton_iterations=diagnostics.newton_iterations,
                residual_norm=diagnostics.newton_residual_norm,
                solver="full_newton_fallback",
            )
            rows.append(row)
            if vtk_enabled and vtk_dir is not None and (step % int(vtk_frame_stride) == 0 or step == steps):
                frame_path = vtk_dir / f"{vtk_stem}_{vtk_frame_index:04d}.vtk"
                vtk_manifest_rows.append(
                    _write_sfc_tet4_vtk_frame(
                        frame_path,
                        model=model,
                        state=state,
                        internal=internal,
                        element_object_ids=element_object_ids,
                        time_value=t,
                        frame_index=vtk_frame_index,
                        include_tensors=vtk_include_tensors,
                    )
                )
                vtk_datasets.append((float(t), frame_path))
                vtk_frame_index += 1
            continue
        u_new = project_fixed_dofs(u_guess, fixed, values)
        a_new = c0 * (u_new - u_pred)
        v_new = v_pred + gamma * float(dt) * a_new
        state = MechanicsState(
            model.X + u_new.reshape((-1, 3)),
            v_new.reshape((-1, 3)),
            a_new.reshape((-1, 3)),
            time=state.time + float(dt),
        )
        if material_mode == "reference_linear":
            if reference_tangent is None:
                raise RuntimeError("reference tangent was not initialized")
            last_internal = _linear_reference_internal_response(model, state.x, reference_tangent)
        else:
            last_internal = stvk_internal_response(model, state.x, assemble_tangent=False)
        last_contact = assemble_contact_response(contact.samples(state.x), model.n_nodes)
        if material_mode == "reference_linear":
            if reference_tangent is None:
                raise RuntimeError("reference tangent was not initialized")
            accepted_internal_force = np.asarray(reference_tangent @ u_new, dtype=float)
            previous = -accepted_internal_force + np.asarray(last_contact.force, dtype=float).reshape(-1)
        else:
            previous = _penalty_rhs_balance(model, last_internal, last_contact)
        rows.append(
            _penalty_history_row(
                time_value=t,
                closure=closure,
                rotation=rotation,
                state=state,
                model=model,
                internal=last_internal,
                contact_response=last_contact,
                young=young,
                poisson=poisson,
                newton_iterations=iteration_count,
                residual_norm=residual_norm,
                solver="modified_newton",
            )
        )
        if vtk_enabled and vtk_dir is not None and (step % int(vtk_frame_stride) == 0 or step == steps):
            frame_path = vtk_dir / f"{vtk_stem}_{vtk_frame_index:04d}.vtk"
            vtk_manifest_rows.append(
                _write_sfc_tet4_vtk_frame(
                    frame_path,
                    model=model,
                    state=state,
                    internal=last_internal,
                element_object_ids=element_object_ids,
                time_value=t,
                frame_index=vtk_frame_index,
                include_tensors=vtk_include_tensors,
            )
        )
            vtk_datasets.append((float(t), frame_path))
            vtk_frame_index += 1
    wall = time.perf_counter() - start
    summary = {
        "sfc_wall_seconds": wall,
        "nodes": int(X.shape[0]),
        "elements": int(elements.shape[0]),
        "gear1_contact_faces": int(pair.gear1.contact_faces.shape[0]),
        "gear2_contact_faces": int(pair.gear2.contact_faces.shape[0]),
        "gear1_support_nodes": int(pair.gear1.support_nodes.size),
        "gear2_support_nodes": int(pair.gear2.support_nodes.size),
        "initial_patch_gap": float(pair.initial_patch_gap),
        "target_overclosure": float(target_overclosure),
        "final_active_contact_samples": int(rows[-1]["active_contact_samples"]) if rows else 0,
        "final_min_gap": float(rows[-1]["min_gap"]) if rows else 0.0,
        "final_normal_force": float(rows[-1]["normal_force"]) if rows else 0.0,
        "hht_alpha": float(hht_alpha),
        "contact_mode": "penalty",
        "penalty_solver": "modified_newton",
        "material_linearization": material_mode,
        "penalty_fallback_count": int(fallback_count),
        "timing_base_tangent_seconds": float(timing_base_tangent),
        "timing_base_factor_seconds": float(timing_base_factor),
        "timing_penalty_residual_seconds": float(timing_residual),
        "timing_penalty_linear_solve_seconds": float(timing_linear_solve),
        "timing_penalty_low_rank_seconds": float(timing_contact_low_rank),
        "timing_penalty_fallback_seconds": float(timing_fallback),
        "status": "completed",
    }
    if vtk_enabled and vtk_dir is not None:
        pvd_path = vtk_dir / f"{vtk_stem}.pvd"
        manifest_path = vtk_dir / f"{vtk_stem}_manifest.csv"
        write_pvd(pvd_path, vtk_datasets)
        _write_csv(manifest_path, vtk_manifest_rows)
        summary.update(
            {
                "sfc_vtk_pvd": str(pvd_path),
                "sfc_vtk_manifest": str(manifest_path),
                "sfc_vtk_frame_count": int(len(vtk_datasets)),
                "sfc_vtk_frame_stride": int(vtk_frame_stride),
                "sfc_vtk_include_tensors": bool(vtk_include_tensors),
            }
        )
    return rows, summary


def _source_drive_fixed_reduced_dofs(assembly: Any, *, time_value: float, omega_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return reduced RP constraints matching the source Abaqus gear-contact deck."""

    hub1 = assembly.hub_slice(0)
    hub2 = assembly.hub_slice(1)
    fixed = [
        hub1.start + 0,
        hub1.start + 1,
        hub1.start + 2,
        hub1.start + 3,
        hub1.start + 4,
        hub1.start + 5,
        hub2.start + 0,
        hub2.start + 1,
        hub2.start + 2,
        hub2.start + 3,
        hub2.start + 4,
    ]
    values = [0.0] * len(fixed)
    values[5] = float(omega_z) * float(time_value)
    return np.asarray(fixed, dtype=np.int64), np.asarray(values, dtype=float)


def _project_reduced_fixed(q: np.ndarray, fixed: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=float).reshape(-1).copy()
    if fixed.size:
        out[np.asarray(fixed, dtype=np.int64)] = np.asarray(values, dtype=float)
    return out


def solve_sfc_source_drive_pair(
    pair: CroppedGearPair,
    *,
    young: float,
    poisson: float,
    density: float,
    pressure_stiffness: float,
    duration: float,
    dt: float,
    gear1_angular_velocity_z: float,
    gear2_torque_z: float,
    hht_alpha: float = ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
    max_iterations: int = 8,
    tolerance: float = 1.0e-9,
    vtk_out_dir: Path | None = None,
    vtk_frame_stride: int = 1,
    vtk_stem: str = "sfc",
    vtk_include_tensors: bool = True,
) -> tuple[list[Row], Row]:
    """Solve the source ``gear_contact.inp`` RP-drive case with SFC contact.

    The model keeps both flexible gear volume meshes.  Gear 1 follows the
    Abaqus source velocity boundary on RP dof 6, gear 2 keeps RP dofs 1--5
    fixed and receives the source moment on RP dof 6.  Contact uses gear 2 as
    slave and gear 1 as master, matching the source contact-pair order.
    """

    n1 = pair.gear1.nodes.shape[0]
    X = np.vstack([pair.gear1.nodes, pair.gear2.nodes])
    elements = np.vstack([pair.gear1.elements, pair.gear2.elements + n1])
    model = MechanicsModel.from_tet4_mesh(X, elements, E=young, nu=poisson, density=density)
    hub1 = RigidHubMPC(pair.gear1.support_nodes, X, pair.gear1.rp)
    hub2 = RigidHubMPC(pair.gear2.support_nodes + n1, X, pair.gear2.rp)
    assembly = build_rigid_hub_reduced_assembly(X, [hub1, hub2], include_free_nodes=True)
    T = assembly.transformation.tocsr()
    master = MaterialSDF.from_triangle_surface(pair.gear1.nodes, pair.gear1.contact_faces)
    contact = LagrangianSDFSurfaceContactGeometry(
        pair.gear2.contact_faces,
        master,
        pair.gear1.nodes,
        pressure_stiffness=pressure_stiffness,
        slave_node_offset=n1,
        master_node_offset=0,
        quadrature="tri3",
        search_radius=max(2.5 * (pair.initial_patch_gap + 1.0e-5), 1.0e-4),
        compiled_batch_projection=True,
    )
    beta, gamma = hht_newmark_parameters(float(hht_alpha))
    scale = 1.0 + float(hht_alpha)
    steps = max(1, int(round(float(duration) / float(dt))))
    if not np.isclose(steps * float(dt), float(duration), rtol=1.0e-9, atol=1.0e-14):
        raise ValueError("duration must be an integer multiple of dt for source-drive SFC")
    reference_internal = stvk_internal_response(model, model.X, assemble_tangent=True)
    K_full = reference_internal.tangent.tocsr()
    K_red = assembly.reduce_matrix(K_full).tocsr()
    M_red = assembly.reduce_matrix(model.mass_matrix).tocsr()
    external = np.zeros(assembly.n_reduced_dofs, dtype=float)
    hub2_slice = assembly.hub_slice(1)
    external[hub2_slice.start + 5] = float(gear2_torque_z)
    q = np.zeros(assembly.n_reduced_dofs, dtype=float)
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    v[assembly.hub_slice(0).start + 5] = float(gear1_angular_velocity_z)
    fixed0, values0 = _source_drive_fixed_reduced_dofs(assembly, time_value=0.0, omega_z=gear1_angular_velocity_z)
    q = _project_reduced_fixed(q, fixed0, values0)
    x0 = model.X + assembly.expand_displacements(q)
    samples0 = list(contact.samples(x0))
    contact0 = _assemble_contact_response_force_only(samples0, model.n_nodes)
    previous = external - np.asarray(K_red @ q, dtype=float).reshape(-1) + assembly.reduce_vector(contact0.force)
    free0 = free_dofs(assembly.n_reduced_dofs, fixed0)
    if free0.size and float(np.linalg.norm(previous[free0])) > 0.0:
        try:
            a[free0] = np.asarray(splu(M_red[free0[:, None], free0].tocsc()).solve(previous[free0]), dtype=float)
        except Exception:
            a[free0] = np.asarray(cg(M_red[free0[:, None], free0].tocsr(), previous[free0], atol=1.0e-12, rtol=1.0e-10)[0], dtype=float)
    state = MechanicsState(x0, assembly.expand_displacements(v), assembly.expand_displacements(a), time=0.0)
    element_object_ids = np.concatenate(
        [
            np.ones(pair.gear1.elements.shape[0], dtype=np.int64),
            np.full(pair.gear2.elements.shape[0], 2, dtype=np.int64),
        ]
    )
    vtk_enabled = vtk_out_dir is not None and int(vtk_frame_stride) > 0
    vtk_dir = Path(vtk_out_dir) if vtk_out_dir is not None else None
    vtk_datasets: list[tuple[float, Path]] = []
    vtk_manifest_rows: list[Row] = []
    if vtk_enabled and vtk_dir is not None:
        vtk_dir.mkdir(parents=True, exist_ok=True)
        for stale in vtk_dir.glob(f"{vtk_stem}_*.vtk"):
            stale.unlink()
        for stale in (vtk_dir / f"{vtk_stem}.pvd", vtk_dir / f"{vtk_stem}_manifest.csv"):
            if stale.exists():
                stale.unlink()
        initial_internal = _linear_reference_internal_response(model, state.x, K_full)
        frame_path = vtk_dir / f"{vtk_stem}_{0:04d}.vtk"
        vtk_manifest_rows.append(
            _write_sfc_tet4_vtk_frame(
                frame_path,
                model=model,
                state=state,
                internal=initial_internal,
                element_object_ids=element_object_ids,
                time_value=0.0,
                frame_index=0,
                include_tensors=vtk_include_tensors,
            )
        )
        vtk_datasets.append((0.0, frame_path))
    c0 = 1.0 / (beta * float(dt) * float(dt))
    base_matrix = (M_red * c0 + K_red * scale).tocsr()
    base_free = None
    base_lu_cache: list[Any] = [None]
    start = time.perf_counter()
    rows: list[Row] = []
    vtk_frame_index = 1
    timing_residual = 0.0
    timing_linear = 0.0
    source_sparse_cg_count = 0
    source_direct_fallback_count = 0
    for step in range(1, steps + 1):
        t = step * float(dt)
        fixed, values = _source_drive_fixed_reduced_dofs(assembly, time_value=t, omega_z=gear1_angular_velocity_z)
        free = free_dofs(assembly.n_reduced_dofs, fixed)
        if base_free is None:
            base_free = base_matrix[free[:, None], free].tocsr() if free.size else None
        q_pred, v_pred, _ = calculix_dynamic_predictor(q, v, a, dt=float(dt), beta=beta, gamma=gamma)
        q_guess = _project_reduced_fixed(q_pred, fixed, values)
        residual_norm = np.inf
        iteration_count = 0
        last_contact = contact0
        contact_matches_q_guess = False
        for iteration in range(1, max(1, int(max_iterations)) + 1):
            q_guess = _project_reduced_fixed(q_guess, fixed, values)
            x_guess = model.X + assembly.expand_displacements(q_guess)
            t_section = time.perf_counter()
            samples = list(contact.samples(x_guess))
            last_contact = _assemble_contact_response_force_only(samples, model.n_nodes)
            contact_matches_q_guess = True
            internal_red = np.asarray(K_red @ q_guess, dtype=float).reshape(-1)
            contact_red = assembly.reduce_vector(last_contact.force)
            rhs_balance = external - internal_red + contact_red
            a_guess = c0 * (q_guess - q_pred)
            residual = np.asarray(M_red @ a_guess, dtype=float).reshape(-1) - scale * rhs_balance + float(hht_alpha) * previous
            residual_norm = float(np.linalg.norm(residual[free])) if free.size else 0.0
            timing_residual += time.perf_counter() - t_section
            t_section = time.perf_counter()
            correction_free = np.empty(0, dtype=float)
            if free.size:
                correction_free = _solve_reduced_penalty_sparse_cg_correction(
                    base_matrix_free=base_free,
                    residual_free=residual[free],
                    samples=samples,
                    transformation=T,
                    free=free,
                    n_total_dofs=model.n_dofs,
                    equilibrium_scale=scale,
                    tolerance=float(tolerance),
                )
                if correction_free is None:
                    source_direct_fallback_count += 1
                    correction_free = _solve_reduced_penalty_low_rank_correction(
                        base_matrix_free=base_free,
                        base_lu_cache=base_lu_cache,
                        residual_free=residual[free],
                        samples=samples,
                        transformation=T,
                        free=free,
                        n_total_dofs=model.n_dofs,
                        equilibrium_scale=scale,
                    )
                else:
                    source_sparse_cg_count += 1
            timing_linear += time.perf_counter() - t_section
            iteration_count = iteration
            if float(np.linalg.norm(correction_free)) <= float(tolerance) * max(1.0, float(np.linalg.norm(q_guess[free])) if free.size else 1.0):
                break
            q_guess[free] += correction_free
            contact_matches_q_guess = False
        q_new = _project_reduced_fixed(q_guess, fixed, values)
        a_new = c0 * (q_new - q_pred)
        v_new = v_pred + gamma * float(dt) * a_new
        x_new = model.X + assembly.expand_displacements(q_new)
        state = MechanicsState(x_new, assembly.expand_displacements(v_new), assembly.expand_displacements(a_new), time=t)
        last_internal = _linear_reference_internal_response(model, state.x, K_full)
        if not contact_matches_q_guess:
            last_contact = _assemble_contact_response_force_only(contact.samples(state.x), model.n_nodes)
        previous = external - np.asarray(K_red @ q_new, dtype=float).reshape(-1) + assembly.reduce_vector(last_contact.force)
        row = _penalty_history_row(
            time_value=t,
            closure=0.0,
            rotation=float(gear1_angular_velocity_z) * t,
            state=state,
            model=model,
            internal=last_internal,
            contact_response=last_contact,
            young=young,
            poisson=poisson,
            newton_iterations=iteration_count,
            residual_norm=residual_norm,
            solver="source_reduced_rp_modified_newton",
        )
        row.update(
            {
                "rp1_rotation_z": float(q_new[assembly.hub_slice(0).start + 5]),
                "rp1_angular_velocity_z": float(v_new[assembly.hub_slice(0).start + 5]),
                "rp2_rotation_z": float(q_new[hub2_slice.start + 5]),
                "rp2_angular_velocity_z": float(v_new[hub2_slice.start + 5]),
                "gear2_torque_z": float(gear2_torque_z),
            }
        )
        rows.append(row)
        if vtk_enabled and vtk_dir is not None and (step % int(vtk_frame_stride) == 0 or step == steps):
            frame_path = vtk_dir / f"{vtk_stem}_{vtk_frame_index:04d}.vtk"
            vtk_manifest_rows.append(
                _write_sfc_tet4_vtk_frame(
                    frame_path,
                    model=model,
                    state=state,
                    internal=last_internal,
                    element_object_ids=element_object_ids,
                    time_value=t,
                    frame_index=vtk_frame_index,
                    include_tensors=vtk_include_tensors,
                )
            )
            vtk_datasets.append((float(t), frame_path))
            vtk_frame_index += 1
        q, v, a = q_new, v_new, a_new
    wall = time.perf_counter() - start
    summary = {
        "sfc_wall_seconds": float(wall),
        "nodes": int(X.shape[0]),
        "elements": int(elements.shape[0]),
        "gear1_contact_faces": int(pair.gear1.contact_faces.shape[0]),
        "gear2_contact_faces": int(pair.gear2.contact_faces.shape[0]),
        "gear1_support_nodes": int(pair.gear1.support_nodes.size),
        "gear2_support_nodes": int(pair.gear2.support_nodes.size),
        "initial_patch_gap": float(pair.initial_patch_gap),
        "final_active_contact_samples": int(rows[-1]["active_contact_samples"]) if rows else 0,
        "final_min_gap": float(rows[-1]["min_gap"]) if rows else 0.0,
        "final_normal_force": float(rows[-1]["normal_force"]) if rows else 0.0,
        "hht_alpha": float(hht_alpha),
        "contact_mode": "source_penalty",
        "penalty_solver": "source_reduced_rp_modified_newton",
        "material_linearization": "reference_linear",
        "source_gear1_angular_velocity_z_rad_per_s": float(gear1_angular_velocity_z),
        "source_gear2_torque_z": float(gear2_torque_z),
        "reduced_dofs": int(assembly.n_reduced_dofs),
        "timing_source_residual_seconds": float(timing_residual),
        "timing_source_linear_seconds": float(timing_linear),
        "source_sparse_cg_count": int(source_sparse_cg_count),
        "source_direct_fallback_count": int(source_direct_fallback_count),
        "status": "completed",
    }
    if vtk_enabled and vtk_dir is not None:
        pvd_path = vtk_dir / f"{vtk_stem}.pvd"
        manifest_path = vtk_dir / f"{vtk_stem}_manifest.csv"
        write_pvd(pvd_path, vtk_datasets)
        _write_csv(manifest_path, vtk_manifest_rows)
        summary.update(
            {
                "sfc_vtk_pvd": str(pvd_path),
                "sfc_vtk_manifest": str(manifest_path),
                "sfc_vtk_frame_count": int(len(vtk_datasets)),
                "sfc_vtk_frame_stride": int(vtk_frame_stride),
                "sfc_vtk_include_tensors": bool(vtk_include_tensors),
            }
        )
    return rows, summary


def solve_sfc_cropped_pair(
    pair: CroppedGearPair,
    *,
    young: float,
    poisson: float,
    density: float,
    pressure_stiffness: float,
    duration: float,
    dt: float,
    target_overclosure: float,
    rotation_rate_z: float,
    hht_alpha: float = -0.05,
    penalty_solver: str = "full_newton",
    material_linearization: str = "stvk",
    vtk_out_dir: Path | None = None,
    vtk_frame_stride: int = 1,
    vtk_stem: str = "sfc",
    vtk_include_tensors: bool = True,
) -> tuple[list[Row], Row]:
    solver = str(penalty_solver).lower()
    if solver not in {"full_newton", "modified_newton"}:
        raise ValueError("penalty_solver must be 'full_newton' or 'modified_newton'")
    if solver == "modified_newton":
        return _solve_sfc_cropped_pair_penalty_modified_newton(
            pair,
            young=young,
            poisson=poisson,
            density=density,
            pressure_stiffness=pressure_stiffness,
            duration=duration,
            dt=dt,
            target_overclosure=target_overclosure,
            rotation_rate_z=rotation_rate_z,
            hht_alpha=hht_alpha,
            max_iterations=12,
            tolerance=1.0e-9,
            material_linearization=material_linearization,
            vtk_out_dir=vtk_out_dir,
            vtk_frame_stride=vtk_frame_stride,
            vtk_stem=vtk_stem,
            vtk_include_tensors=vtk_include_tensors,
        )
    n1 = pair.gear1.nodes.shape[0]
    X = np.vstack([pair.gear1.nodes, pair.gear2.nodes])
    elements = np.vstack([pair.gear1.elements, pair.gear2.elements + n1])
    model = MechanicsModel.from_tet4_mesh(X, elements, E=young, nu=poisson, density=density)
    master = MaterialSDF.from_triangle_surface(pair.gear2.nodes, pair.gear2.contact_faces)
    contact = LagrangianSDFSurfaceContactGeometry(
        pair.gear1.contact_faces,
        master,
        pair.gear2.nodes,
        pressure_stiffness=pressure_stiffness,
        slave_node_offset=0,
        master_node_offset=n1,
        quadrature="tri3",
        search_radius=max(2.5 * (pair.initial_patch_gap + target_overclosure), 1.0e-4),
    )
    fixed0, values0 = _fixed_conditions_for_pair(pair, closure=0.0, rotation_z=0.0)
    state, previous = initial_state_dirichlet(model, contact, gravity=0.0, fixed_dofs=fixed0, fixed_values=values0)
    rows: list[Row] = []
    steps = max(1, int(round(float(duration) / float(dt))))
    start = time.perf_counter()
    for step in range(1, steps + 1):
        t = step * float(dt)
        ramp = t / max(float(duration), float(dt))
        closure = ramp * (pair.initial_patch_gap + float(target_overclosure))
        rotation = float(rotation_rate_z) * t
        fixed, values = _fixed_conditions_for_pair(pair, closure=closure, rotation_z=rotation)
        state, previous, diagnostics = hht_step_dirichlet(
            model,
            state,
            previous,
            contact,
            dt=float(dt),
            gravity=0.0,
            fixed_dofs=fixed,
            fixed_values=values,
            alpha=float(hht_alpha),
            max_iterations=10,
            tolerance=1.0e-9,
            acceptance_policy="relative_correction",
        )
        internal = diagnostics.internal
        disp = state.x - model.X
        strain_norm = np.linalg.norm(internal.strain, axis=(1, 2)) if internal.strain.size else np.empty(0, dtype=float)
        elastic_strain_norm = _elastic_strain_norm_from_stress(internal.stress, young=young, poisson=poisson)
        equivalent_elastic_strain = _equivalent_elastic_strain_from_mises(internal.von_mises, young=young, poisson=poisson)
        rows.append(
            {
                "time": t,
                "closure": closure,
                "rotation_z": rotation,
                "active_contact_samples": diagnostics.contact.active_count,
                "min_gap": diagnostics.contact.min_gap,
                "normal_force": diagnostics.contact.normal_force,
                "max_displacement_norm": float(np.max(np.linalg.norm(disp, axis=1))),
                "p95_von_mises": float(np.percentile(internal.von_mises, 95.0)) if internal.von_mises.size else 0.0,
                "max_von_mises": float(np.max(internal.von_mises)) if internal.von_mises.size else 0.0,
                "p95_strain_norm": float(np.percentile(strain_norm, 95.0)) if strain_norm.size else 0.0,
                "max_strain_norm": float(np.max(strain_norm)) if strain_norm.size else 0.0,
                "p95_elastic_strain_norm": float(np.percentile(elastic_strain_norm, 95.0)) if elastic_strain_norm.size else 0.0,
                "max_elastic_strain_norm": float(np.max(elastic_strain_norm)) if elastic_strain_norm.size else 0.0,
                "p95_equivalent_elastic_strain": float(np.percentile(equivalent_elastic_strain, 95.0)) if equivalent_elastic_strain.size else 0.0,
                "max_equivalent_elastic_strain": float(np.max(equivalent_elastic_strain)) if equivalent_elastic_strain.size else 0.0,
                "strain_energy": internal.strain_energy,
                "newton_iterations": diagnostics.newton_iterations,
            }
        )
    wall = time.perf_counter() - start
    summary = {
        "sfc_wall_seconds": wall,
        "nodes": int(X.shape[0]),
        "elements": int(elements.shape[0]),
        "gear1_contact_faces": int(pair.gear1.contact_faces.shape[0]),
        "gear2_contact_faces": int(pair.gear2.contact_faces.shape[0]),
        "gear1_support_nodes": int(pair.gear1.support_nodes.size),
        "gear2_support_nodes": int(pair.gear2.support_nodes.size),
        "initial_patch_gap": float(pair.initial_patch_gap),
        "target_overclosure": float(target_overclosure),
        "final_active_contact_samples": int(rows[-1]["active_contact_samples"]) if rows else 0,
        "final_min_gap": float(rows[-1]["min_gap"]) if rows else 0.0,
        "final_normal_force": float(rows[-1]["normal_force"]) if rows else 0.0,
        "hht_alpha": float(hht_alpha),
        "contact_mode": "penalty",
        "status": "completed",
    }
    return rows, summary


def _cropped_pair_model_and_contact(
    pair: CroppedGearPair,
    *,
    young: float,
    poisson: float,
    density: float,
    pressure_stiffness: float,
    target_overclosure: float,
) -> tuple[MechanicsModel, LagrangianSDFSurfaceContactGeometry]:
    n1 = pair.gear1.nodes.shape[0]
    X = np.vstack([pair.gear1.nodes, pair.gear2.nodes])
    elements = np.vstack([pair.gear1.elements, pair.gear2.elements + n1])
    model = MechanicsModel.from_tet4_mesh(X, elements, E=young, nu=poisson, density=density)
    master = MaterialSDF.from_triangle_surface(pair.gear2.nodes, pair.gear2.contact_faces)
    contact = LagrangianSDFSurfaceContactGeometry(
        pair.gear1.contact_faces,
        master,
        pair.gear2.nodes,
        pressure_stiffness=pressure_stiffness,
        slave_node_offset=0,
        master_node_offset=n1,
        quadrature="tri3",
        search_radius=max(2.5 * (pair.initial_patch_gap + target_overclosure), 1.0e-4),
    )
    return model, contact


def _hard_contact_linearized_gap_jacobian(
    contact: LagrangianSDFSurfaceContactGeometry,
    x_linearization: np.ndarray,
    u_linearization: np.ndarray,
    *,
    n_total_dofs: int,
    constraint_averaging: str = "slave_face",
) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray]:
    samples = list(contact.samples(x_linearization))
    gap_at_linearization, gap_jacobian = hard_contact_gap_jacobian_from_samples(samples, n_total_dofs=int(n_total_dofs))
    areas = np.asarray([float(sample.area) for sample in samples], dtype=float)
    averaging = str(constraint_averaging).lower()
    if averaging not in {"none", "slave_face", "surface_patch"}:
        raise ValueError("constraint_averaging must be 'none', 'slave_face', or 'surface_patch'")
    if averaging == "surface_patch" and gap_at_linearization.size:
        groups = _sample_connected_components(samples)
        gap_at_linearization, gap_jacobian, areas = _aggregate_constraint_groups(
            gap_at_linearization,
            gap_jacobian,
            areas,
            groups,
        )
    elif averaging == "slave_face" and gap_at_linearization.size >= 3 and gap_at_linearization.size % 3 == 0:
        groups = [np.arange(start, start + 3, dtype=np.int64) for start in range(0, gap_at_linearization.size, 3)]
        gap_at_linearization, gap_jacobian, areas = _aggregate_constraint_groups(
            gap_at_linearization,
            gap_jacobian,
            areas,
            groups,
        )
    gap_offset = gap_at_linearization - gap_jacobian @ np.asarray(u_linearization, dtype=float).reshape(-1)
    return samples, gap_offset, gap_jacobian, areas


def _aggregate_constraint_groups(
    gaps: np.ndarray,
    rows: np.ndarray,
    areas: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aggregated_gaps: list[float] = []
    aggregated_rows: list[np.ndarray] = []
    aggregated_areas: list[float] = []
    for group in groups:
        ids = np.asarray(group, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            continue
        group_area = areas[ids]
        total_area = float(np.sum(group_area))
        if total_area <= 0.0:
            weights = np.full(ids.size, 1.0 / float(ids.size), dtype=float)
        else:
            weights = group_area / total_area
        aggregated_gaps.append(float(weights @ gaps[ids]))
        aggregated_rows.append(weights @ rows[ids])
        aggregated_areas.append(total_area)
    if not aggregated_gaps:
        return gaps, rows, areas
    return np.asarray(aggregated_gaps, dtype=float), np.vstack(aggregated_rows), np.asarray(aggregated_areas, dtype=float)


def _sample_connected_components(samples: list[Any]) -> list[np.ndarray]:
    """Return connected sample groups using shared slave nodes."""

    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    by_node: dict[int, int] = {}
    for sample_id, sample in enumerate(samples):
        for node in np.asarray(sample.node_ids, dtype=np.int64).reshape(-1):
            key = int(node)
            if key in by_node:
                union(sample_id, by_node[key])
            else:
                by_node[key] = sample_id
    components: dict[int, list[int]] = {}
    for sample_id in range(len(samples)):
        components.setdefault(find(sample_id), []).append(sample_id)
    return [np.asarray(ids, dtype=np.int64) for ids in components.values()]


def _contact_patch_min_edge_length(pair: CroppedGearPair) -> float:
    values: list[float] = []
    for nodes, faces in ((pair.gear1.nodes, pair.gear1.contact_faces), (pair.gear2.nodes, pair.gear2.contact_faces)):
        for face in np.asarray(faces, dtype=np.int64):
            tri = np.asarray(nodes, dtype=float)[face]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                length = float(np.linalg.norm(tri[a] - tri[b]))
                if length > 0.0:
                    values.append(length)
    if not values:
        raise ValueError("contact patch has no positive edge length")
    return float(min(values))


def _contact_patch_representative_length(pair: CroppedGearPair) -> float:
    """Return an Abaqus-style representative contact facet length.

    The hard-contact penalty fallback should scale with the underlying element
    stiffness instead of a case-fitted factor.  A robust representative length
    is the area-weighted mean edge length over the two contact patches; it is
    insensitive to isolated short sliver edges but still follows mesh
    refinement.
    """

    weighted_lengths: list[float] = []
    weights: list[float] = []
    for nodes, faces in ((pair.gear1.nodes, pair.gear1.contact_faces), (pair.gear2.nodes, pair.gear2.contact_faces)):
        for face in np.asarray(faces, dtype=np.int64):
            tri = np.asarray(nodes, dtype=float)[face]
            area = 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
            if area <= 0.0:
                continue
            edges = [
                float(np.linalg.norm(tri[0] - tri[1])),
                float(np.linalg.norm(tri[1] - tri[2])),
                float(np.linalg.norm(tri[2] - tri[0])),
            ]
            positive = [value for value in edges if value > 0.0]
            if not positive:
                continue
            weighted_lengths.append(float(np.mean(positive)))
            weights.append(area)
    if not weighted_lengths:
        raise ValueError("contact patch has no positive representative length")
    return float(np.average(np.asarray(weighted_lengths, dtype=float), weights=np.asarray(weights, dtype=float)))


def _effective_hard_pressure_stiffness(
    *,
    pair: CroppedGearPair,
    young: float,
    poisson: float,
    pressure_stiffness: float,
    hard_enforcement: str,
    pressure_smoothing_factor: float,
) -> float:
    mode = str(hard_enforcement).lower()
    if mode == "element_pressure_smoothing":
        return float(pressure_smoothing_factor) * float(young) / _contact_patch_min_edge_length(pair)
    if mode == "abaqus_standard_penalty":
        # Abaqus/Standard's finite-sliding hard contact uses a penalty fallback
        # tied to representative underlying element stiffness.  We use the
        # same dimensional law, k_p = 10 E / h_rep, with h_rep computed from
        # the current contact facet set rather than from a fitted parameter.
        _ = float(poisson)  # kept in the signature for future material variants
        return 10.0 * float(young) / _contact_patch_representative_length(pair)
    return float(pressure_stiffness)


def _equivalent_hub_reaction(
    residual: np.ndarray,
    *,
    node_ids: np.ndarray,
    nodes_reference: np.ndarray,
    reference_point: np.ndarray,
    drive_direction: np.ndarray,
    current_nodes: np.ndarray | None = None,
) -> dict[str, float]:
    """Project constrained nodal residuals to the equivalent hub/RP reaction."""

    hub = RigidHubMPC(np.asarray(node_ids, dtype=np.int64), np.asarray(nodes_reference, dtype=float), np.asarray(reference_point, dtype=float))
    generalized = hub.generalized_force_from_nodal_forces(
        np.asarray(residual, dtype=float).reshape((-1, 3)),
        current_nodes=current_nodes,
    )
    force = generalized[:3]
    moment = generalized[3:]
    drive = np.asarray(drive_direction, dtype=float)
    drive_norm = float(np.linalg.norm(drive))
    if drive_norm > 0.0:
        drive = drive / drive_norm
    return {
        "rp_force_x": float(force[0]),
        "rp_force_y": float(force[1]),
        "rp_force_z": float(force[2]),
        "rp_force_drive": float(force @ drive),
        "rp_force_norm": float(np.linalg.norm(force)),
        "rp_moment_x": float(moment[0]),
        "rp_moment_y": float(moment[1]),
        "rp_moment_z": float(moment[2]),
        "rp_moment_norm": float(np.linalg.norm(moment)),
    }


def _body_vector_slice(vector: np.ndarray, *, n1_dofs: int, body: int) -> np.ndarray:
    values = np.asarray(vector, dtype=float).reshape(-1)
    if body == 1:
        return values[:n1_dofs]
    if body == 2:
        return values[n1_dofs:]
    raise ValueError("body must be 1 or 2")


def _equivalent_pair_reactions(
    residual: np.ndarray,
    *,
    pair: CroppedGearPair,
    state_x: np.ndarray,
    n1_dofs: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return gear1/gear2 RP generalized reactions for one residual vector."""

    return (
        _equivalent_hub_reaction(
            _body_vector_slice(residual, n1_dofs=n1_dofs, body=1),
            node_ids=pair.gear1.support_nodes,
            nodes_reference=pair.gear1.nodes,
            reference_point=pair.gear1.rp,
            drive_direction=pair.drive_direction,
            current_nodes=state_x[: pair.gear1.nodes.shape[0]],
        ),
        _equivalent_hub_reaction(
            _body_vector_slice(residual, n1_dofs=n1_dofs, body=2),
            node_ids=pair.gear2.support_nodes,
            nodes_reference=pair.gear2.nodes,
            reference_point=pair.gear2.rp,
            drive_direction=pair.drive_direction,
            current_nodes=state_x[pair.gear1.nodes.shape[0] :],
        ),
    )


def solve_sfc_cropped_pair_hard_contact(
    pair: CroppedGearPair,
    *,
    young: float,
    poisson: float,
    density: float,
    pressure_stiffness: float,
    duration: float,
    dt: float,
    target_overclosure: float,
    rotation_rate_z: float,
    max_iterations: int = 6,
    tolerance: float = 1.0e-9,
    hard_enforcement: str = "abaqus_standard_penalty",
    constraint_averaging: str = "slave_face",
    pressure_smoothing_factor: float = 4.0,
    hht_alpha: float = ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
    automatic_increment: bool = True,
    cutback_factor: float = 0.5,
    linear_solver: str = "dense",
) -> tuple[list[Row], Row]:
    """Solve the cropped gear pair with implicit Newmark + HARD contact KKT.

    This is the dynamic counterpart of the flat/tilted HARD-contact alignment
    cases. Each step linearizes the current Lagrangian-SDF closest-feature
    payloads around the current iterate and solves the effective Newmark system
    together with the hard normal-contact active set. The cropped gear mesh is
    TET4, so the payload is triangular; the Q4 payload layer remains the C3D8
    specialization validated by the tooth-patch runner.
    """

    enforcement = str(hard_enforcement).lower()
    if enforcement not in {"exact", "pressure_compliance", "element_pressure_smoothing", "abaqus_standard_penalty"}:
        raise ValueError(
            "hard_enforcement must be 'exact', 'pressure_compliance', 'element_pressure_smoothing', or 'abaqus_standard_penalty'"
        )
    solver_kind = str(linear_solver).lower()
    if solver_kind not in {"dense", "sparse", "iterative", "auto"}:
        raise ValueError("linear_solver must be 'dense', 'sparse', 'iterative', or 'auto'")
    effective_pressure_stiffness = _effective_hard_pressure_stiffness(
        pair=pair,
        young=young,
        poisson=poisson,
        pressure_stiffness=pressure_stiffness,
        hard_enforcement=enforcement,
        pressure_smoothing_factor=pressure_smoothing_factor,
    )
    model, contact = _cropped_pair_model_and_contact(
        pair,
        young=young,
        poisson=poisson,
        density=density,
        pressure_stiffness=pressure_stiffness,
        target_overclosure=target_overclosure,
    )
    fixed0, values0 = _fixed_conditions_for_pair(pair, closure=0.0, rotation_z=0.0)
    u0 = project_fixed_dofs(np.zeros(model.n_dofs, dtype=float), fixed0, values0)
    state_x = model.X + u0.reshape((-1, 3))
    velocity = np.zeros_like(model.X)
    acceleration = np.zeros_like(model.X)
    previous_rhs_balance = np.zeros(model.n_dofs, dtype=float)
    rows: list[Row] = []
    beta, gamma = hht_newmark_parameters(float(hht_alpha))
    max_dt = float(dt)
    min_dt = min(max_dt * 1.0e-4, 1.0e-8)
    if max_dt <= 0.0:
        raise ValueError("dt must be positive")
    if not (0.0 < float(cutback_factor) < 1.0):
        raise ValueError("cutback_factor must lie in (0, 1)")
    start = time.perf_counter()
    timing_internal_tangent = 0.0
    timing_contact_linearization = 0.0
    timing_effective_system = 0.0
    timing_hard_contact_solve = 0.0
    timing_accepted_internal = 0.0
    timing_accepted_contact = 0.0
    timing_reaction_diagnostics = 0.0
    last_iterations = 0
    last_converged = True
    cutback_count = 0
    increment_count = 0
    t = 0.0
    h_trial = max_dt
    while t < float(duration) - 1.0e-15:
        h = min(float(h_trial), float(duration) - t)
        t_candidate = t + h
        ramp = t_candidate / max(float(duration), float(dt))
        closure = ramp * (pair.initial_patch_gap + float(target_overclosure))
        rotation = float(rotation_rate_z) * t_candidate
        fixed, values = _fixed_conditions_for_pair(pair, closure=closure, rotation_z=rotation)
        u_n = (state_x - model.X).reshape(-1)
        v_n = velocity.reshape(-1)
        a_n = acceleration.reshape(-1)
        c0 = 1.0 / (beta * h * h)
        equilibrium_scale = 1.0 + float(hht_alpha)
        u_pred, v_pred, _accold_after_prediction = calculix_dynamic_predictor(u_n, v_n, a_n, dt=h, beta=beta, gamma=gamma)
        u_guess = project_fixed_dofs(u_pred.copy(), fixed, values)
        solution = None
        final_samples: list[Any] = []
        final_gap_jacobian = np.empty((0, model.n_dofs), dtype=float)
        converged = False
        iteration_count = 0
        previous_active: np.ndarray | None = None
        active_stable_count = 0
        for iteration in range(1, max(1, int(max_iterations)) + 1):
            x_guess = model.X + u_guess.reshape((-1, 3))
            t_section = time.perf_counter()
            internal = stvk_internal_response(model, x_guess, assemble_tangent=True)
            timing_internal_tangent += time.perf_counter() - t_section
            t_section = time.perf_counter()
            final_samples, gap_offset, gap_jacobian, constraint_areas = _hard_contact_linearized_gap_jacobian(
                contact,
                x_guess,
                u_guess,
                n_total_dofs=model.n_dofs,
                constraint_averaging=constraint_averaging,
            )
            timing_contact_linearization += time.perf_counter() - t_section
            final_gap_jacobian = gap_jacobian
            compliance = (
                None
                if enforcement == "exact"
                else 1.0 / (float(effective_pressure_stiffness) * np.maximum(constraint_areas, 1.0e-30))
            )
            t_section = time.perf_counter()
            tangent = internal.tangent.tocsr()
            effective_stiffness = (model.mass_matrix * c0 + tangent * equilibrium_scale).tocsr()
            effective_force = np.asarray(
                model.mass_matrix @ (c0 * u_pred)
                - equilibrium_scale * internal.force.reshape(-1)
                + equilibrium_scale * (tangent @ u_guess)
                - float(hht_alpha) * previous_rhs_balance,
                dtype=float,
            )
            timing_effective_system += time.perf_counter() - t_section
            use_sparse_hard_contact = solver_kind in {"sparse", "iterative"} or (solver_kind == "auto" and model.n_dofs > 12000)
            solve_hard_contact = solve_linear_hard_contact_with_dirichlet_sparse if use_sparse_hard_contact else solve_linear_hard_contact_with_dirichlet
            sparse_solver_mode = "primal_cg" if solver_kind == "iterative" else "direct"
            t_section = time.perf_counter()
            if use_sparse_hard_contact:
                solution = solve_hard_contact(
                    effective_stiffness,
                    effective_force,
                    gap_offset,
                    gap_jacobian,
                    fixed_dofs=fixed,
                    fixed_values=values,
                    equilibrium_jacobian_scale=equilibrium_scale,
                    normal_compliance=compliance,
                    tolerance=float(tolerance),
                    max_iterations=30,
                    linear_solver=sparse_solver_mode,
                    iterative_tolerance=1.0e-10,
                    iterative_max_iterations=600,
                    iterative_fallback_to_direct=True,
                )
            else:
                solution = solve_hard_contact(
                    effective_stiffness,
                    effective_force,
                    gap_offset,
                    gap_jacobian,
                    fixed_dofs=fixed,
                    fixed_values=values,
                    equilibrium_jacobian_scale=equilibrium_scale,
                    normal_compliance=compliance,
                    tolerance=float(tolerance),
                    max_iterations=30,
                )
            timing_hard_contact_solve += time.perf_counter() - t_section
            correction_norm = float(np.linalg.norm(solution.displacement - u_guess))
            displacement_scale = max(1.0, float(np.linalg.norm(solution.displacement)))
            u_guess = solution.displacement.copy()
            iteration_count = iteration
            active_now = np.asarray(solution.active, dtype=bool)
            if previous_active is not None and np.array_equal(active_now, previous_active):
                active_stable_count += 1
            else:
                active_stable_count = 0
            previous_active = active_now.copy()
            strict_correction = correction_norm <= float(tolerance) * displacement_scale
            abaqus_style_contact_accept = bool(solution.converged) and active_stable_count >= 1 and correction_norm <= 1.0e-4 * displacement_scale
            if bool(solution.converged) and (strict_correction or abaqus_style_contact_accept):
                converged = True
                break
        if solution is None:
            raise RuntimeError("hard contact solve did not run")
        if (not converged) and bool(automatic_increment) and h > min_dt * (1.0 + 1.0e-12):
            h_trial = max(min_dt, h * float(cutback_factor))
            cutback_count += 1
            continue
        previous_rhs_before_step = previous_rhs_balance.copy()
        u_new = solution.displacement
        a_new = c0 * (u_new - u_pred)
        v_new = v_pred + gamma * h * a_new
        state_x = model.X + u_new.reshape((-1, 3))
        velocity = v_new.reshape((-1, 3))
        acceleration = a_new.reshape((-1, 3))
        t_section = time.perf_counter()
        internal = stvk_internal_response(model, state_x, assemble_tangent=False)
        timing_accepted_internal += time.perf_counter() - t_section
        t_section = time.perf_counter()
        raw_gaps = np.asarray([float(sample.gap) for sample in contact.samples(state_x)], dtype=float)
        timing_accepted_contact += time.perf_counter() - t_section
        active = np.asarray(solution.active, dtype=bool)
        multipliers = np.asarray(solution.multipliers, dtype=float)
        active_force = float(np.sum(multipliers[active])) if multipliers.size else 0.0
        contact_pressure = np.zeros_like(multipliers)
        if multipliers.size and constraint_areas.size == multipliers.size:
            contact_pressure = multipliers / np.maximum(constraint_areas, 1.0e-30)
        max_contact_pressure = float(np.max(contact_pressure[active])) if multipliers.size and np.any(active) else 0.0
        mass_acceleration = np.asarray(model.mass_matrix @ a_new, dtype=float)
        contact_balance = np.zeros(model.n_dofs, dtype=float)
        if final_gap_jacobian.size and multipliers.size:
            contact_balance = np.asarray(final_gap_jacobian.T @ multipliers, dtype=float)
        static_balance = np.asarray(internal.force.reshape(-1) - contact_balance, dtype=float)
        current_rhs_balance = -static_balance
        dynamic_balance = np.asarray(mass_acceleration + static_balance, dtype=float)
        hht_balance = np.asarray(
            mass_acceleration - equilibrium_scale * current_rhs_balance + float(hht_alpha) * previous_rhs_before_step,
            dtype=float,
        )
        t_section = time.perf_counter()
        n1_dofs = 3 * pair.gear1.nodes.shape[0]
        hub1_hht_reaction, hub2_hht_reaction = _equivalent_pair_reactions(
            hht_balance,
            pair=pair,
            state_x=state_x,
            n1_dofs=n1_dofs,
        )
        hub1_dynamic_reaction, hub2_dynamic_reaction = _equivalent_pair_reactions(
            dynamic_balance,
            pair=pair,
            state_x=state_x,
            n1_dofs=n1_dofs,
        )
        hub1_static_reaction, hub2_static_reaction = _equivalent_pair_reactions(
            static_balance,
            pair=pair,
            state_x=state_x,
            n1_dofs=n1_dofs,
        )
        hub1_inertia_reaction, hub2_inertia_reaction = _equivalent_pair_reactions(
            mass_acceleration,
            pair=pair,
            state_x=state_x,
            n1_dofs=n1_dofs,
        )
        hub1_contact_reaction, hub2_contact_reaction = _equivalent_pair_reactions(
            contact_balance,
            pair=pair,
            state_x=state_x,
            n1_dofs=n1_dofs,
        )
        timing_reaction_diagnostics += time.perf_counter() - t_section
        previous_rhs_balance = current_rhs_balance
        disp = state_x - model.X
        strain_norm = np.linalg.norm(internal.strain, axis=(1, 2)) if internal.strain.size else np.empty(0, dtype=float)
        elastic_strain_norm = _elastic_strain_norm_from_stress(internal.stress, young=young, poisson=poisson)
        equivalent_elastic_strain = _equivalent_elastic_strain_from_mises(
            internal.von_mises,
            young=young,
            poisson=poisson,
        )
        row = {
            "time": t_candidate,
            "dt": h,
            "closure": closure,
            "rotation_z": rotation,
            "active_contact_samples": int(np.count_nonzero(active)),
            "min_gap": float(np.min(raw_gaps)) if raw_gaps.size else 0.0,
            "linearized_min_gap": float(np.min(solution.gaps)) if solution.gaps.size else 0.0,
            "normal_force": active_force,
            "contact_multiplier_sum": active_force,
            "max_contact_pressure": max_contact_pressure,
            "max_displacement_norm": float(np.max(np.linalg.norm(disp, axis=1))),
            "p95_von_mises": float(np.percentile(internal.von_mises, 95.0)) if internal.von_mises.size else 0.0,
            "max_von_mises": float(np.max(internal.von_mises)) if internal.von_mises.size else 0.0,
            "p95_strain_norm": float(np.percentile(strain_norm, 95.0)) if strain_norm.size else 0.0,
            "max_strain_norm": float(np.max(strain_norm)) if strain_norm.size else 0.0,
            "p95_elastic_strain_norm": float(np.percentile(elastic_strain_norm, 95.0)) if elastic_strain_norm.size else 0.0,
            "max_elastic_strain_norm": float(np.max(elastic_strain_norm)) if elastic_strain_norm.size else 0.0,
            "p95_equivalent_elastic_strain": float(np.percentile(equivalent_elastic_strain, 95.0))
            if equivalent_elastic_strain.size
            else 0.0,
            "max_equivalent_elastic_strain": float(np.max(equivalent_elastic_strain)) if equivalent_elastic_strain.size else 0.0,
            "strain_energy": internal.strain_energy,
            "newton_iterations": int(iteration_count),
            "hard_active_set_converged": int(bool(solution.converged)),
            "hard_outer_converged": int(bool(converged)),
            "automatic_increment_cutbacks": int(cutback_count),
            "hard_contact_samples": int(len(final_samples)),
            "hard_constraints": int(solution.gaps.shape[0]),
            "hht_alpha": float(hht_alpha),
            "hht_beta": float(beta),
            "hht_gamma": float(gamma),
        }
        row.update({f"rp1_{key}": value for key, value in hub1_hht_reaction.items()})
        row.update({f"rp2_{key}": value for key, value in hub2_hht_reaction.items()})
        row.update({f"rp1_hht_{key}": value for key, value in hub1_hht_reaction.items()})
        row.update({f"rp2_hht_{key}": value for key, value in hub2_hht_reaction.items()})
        row.update({f"rp1_dynamic_{key}": value for key, value in hub1_dynamic_reaction.items()})
        row.update({f"rp2_dynamic_{key}": value for key, value in hub2_dynamic_reaction.items()})
        row.update({f"rp1_static_{key}": value for key, value in hub1_static_reaction.items()})
        row.update({f"rp2_static_{key}": value for key, value in hub2_static_reaction.items()})
        row.update({f"rp1_inertia_{key}": value for key, value in hub1_inertia_reaction.items()})
        row.update({f"rp2_inertia_{key}": value for key, value in hub2_inertia_reaction.items()})
        row.update({f"rp1_contact_{key}": value for key, value in hub1_contact_reaction.items()})
        row.update({f"rp2_contact_{key}": value for key, value in hub2_contact_reaction.items()})
        row["rp_reaction_definition"] = "static_physical_constraint_residual"
        row["rp_force_drive"] = float(hub1_static_reaction["rp_force_drive"])
        row["rp_force_norm"] = float(hub1_static_reaction["rp_force_norm"])
        row["opposing_rp_force_norm"] = float(hub2_static_reaction["rp_force_norm"])
        row["rp_static_force_norm"] = float(hub1_static_reaction["rp_force_norm"])
        row["opposing_rp_static_force_norm"] = float(hub2_static_reaction["rp_force_norm"])
        row["rp_dynamic_force_norm"] = float(hub1_dynamic_reaction["rp_force_norm"])
        row["opposing_rp_dynamic_force_norm"] = float(hub2_dynamic_reaction["rp_force_norm"])
        row["rp_inertia_force_norm"] = float(hub1_inertia_reaction["rp_force_norm"])
        row["opposing_rp_inertia_force_norm"] = float(hub2_inertia_reaction["rp_force_norm"])
        row["rp_contact_force_norm"] = float(hub1_contact_reaction["rp_force_norm"])
        row["opposing_rp_contact_force_norm"] = float(hub2_contact_reaction["rp_force_norm"])
        rows.append(row)
        last_iterations = iteration_count
        last_converged = bool(converged)
        increment_count += 1
        t = t_candidate
        h_trial = max_dt if (not automatic_increment) else min(max_dt, max(h, min_dt) * (1.25 if converged and iteration_count <= 3 else 1.0))
    wall = time.perf_counter() - start
    summary = {
        "sfc_wall_seconds": wall,
        "nodes": int(model.X.shape[0]),
        "elements": int(model.elements.shape[0]),
        "gear1_contact_faces": int(pair.gear1.contact_faces.shape[0]),
        "gear2_contact_faces": int(pair.gear2.contact_faces.shape[0]),
        "gear1_support_nodes": int(pair.gear1.support_nodes.size),
        "gear2_support_nodes": int(pair.gear2.support_nodes.size),
        "initial_patch_gap": float(pair.initial_patch_gap),
        "target_overclosure": float(target_overclosure),
        "final_active_contact_samples": int(rows[-1]["active_contact_samples"]) if rows else 0,
        "final_min_gap": float(rows[-1]["min_gap"]) if rows else 0.0,
        "final_normal_force": float(rows[-1]["normal_force"]) if rows else 0.0,
        "final_rp_force_norm": float(rows[-1].get("rp_force_norm", 0.0)) if rows else 0.0,
        "final_opposing_rp_force_norm": float(rows[-1].get("opposing_rp_force_norm", 0.0)) if rows else 0.0,
        "final_hard_outer_iterations": int(last_iterations),
        "final_hard_outer_converged": int(last_converged),
        "accepted_increment_count": int(increment_count),
        "automatic_increment": str(bool(automatic_increment)).lower(),
        "cutback_count": int(cutback_count),
        "hht_alpha": float(hht_alpha),
        "hht_beta": float(beta),
        "hht_gamma": float(gamma),
        "rp_reaction_definition": "static_physical_constraint_residual",
        "contact_mode": "hard",
        "hard_enforcement": enforcement,
        "constraint_averaging": str(constraint_averaging),
        "linear_solver": solver_kind,
        "effective_hard_pressure_stiffness": float(effective_pressure_stiffness),
        "pressure_smoothing_factor": float(pressure_smoothing_factor),
        "contact_patch_min_edge_length": _contact_patch_min_edge_length(pair),
        "timing_internal_tangent_seconds": float(timing_internal_tangent),
        "timing_contact_linearization_seconds": float(timing_contact_linearization),
        "timing_effective_system_seconds": float(timing_effective_system),
        "timing_hard_contact_solve_seconds": float(timing_hard_contact_solve),
        "timing_accepted_internal_seconds": float(timing_accepted_internal),
        "timing_accepted_contact_seconds": float(timing_accepted_contact),
        "timing_reaction_diagnostics_seconds": float(timing_reaction_diagnostics),
        "status": "completed",
    }
    return rows, summary


def _write_abaqus_alignment_deck(
    path: Path,
    pair: CroppedGearPair,
    *,
    young: float,
    poisson: float,
    density: float,
    pressure_stiffness: float,
    duration: float,
    dt: float,
    target_overclosure: float,
    rotation_rate_z: float,
    contact_mode: str = "penalty",
) -> None:
    """Write a compact Abaqus deck with matching cropped meshes and RP MPCs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = str(contact_mode).lower()
    if mode not in {"penalty", "hard"}:
        raise ValueError("contact_mode must be 'penalty' or 'hard'")
    if mode == "hard":
        contact_lines = [
            "*Surface Interaction, name=SFC_HARD",
            "*Surface Behavior, pressure-overclosure=HARD",
            "*Friction",
            "0.",
            "*Contact Pair, interaction=SFC_HARD, type=SURFACE TO SURFACE",
            "G1_SURFACE, G2_SURFACE",
        ]
    else:
        contact_lines = [
            "*Surface Interaction, name=SFC_LINEAR",
            "1.",
            "*Friction",
            "0.",
            "*Surface Behavior, pressure-overclosure=LINEAR",
            f"{pressure_stiffness:.12e}",
            "*Contact Pair, interaction=SFC_LINEAR, type=SURFACE TO SURFACE",
            "G1_SURFACE, G2_SURFACE",
        ]
    lines: list[str] = [
        "*Heading",
        "Cropped gear RP-MPC contact alignment deck generated by SFC validation.",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
    ]
    for patch in (pair.gear1, pair.gear2):
        lines.extend([f"*Part, name={patch.name.upper()}", "*Node"])
        for idx, xyz in enumerate(patch.nodes, start=1):
            lines.append(f"{idx}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
        lines.append("*Element, type=C3D4")
        for idx, element in enumerate(patch.elements, start=1):
            labels = ", ".join(str(int(v) + 1) for v in element)
            lines.append(f"{idx}, {labels}")
        lines.append("*Elset, elset=ALL_ELEMENTS, generate")
        lines.append(f"1, {patch.elements.shape[0]}, 1")
        lines.append("*Solid Section, elset=ALL_ELEMENTS, material=STEEL")
        lines.append(",")
        lines.append("*End Part")
    lines.extend(["*Assembly, name=ASSEMBLY"])
    for patch in (pair.gear1, pair.gear2):
        lines.extend([f"*Instance, name={patch.name.upper()}-1, part={patch.name.upper()}", "*End Instance"])
    lines.extend(["*Node", f"1000001, {pair.gear1.rp[0]:.12e}, {pair.gear1.rp[1]:.12e}, {pair.gear1.rp[2]:.12e}", f"1000002, {pair.gear2.rp[0]:.12e}, {pair.gear2.rp[1]:.12e}, {pair.gear2.rp[2]:.12e}"])
    _append_assembly_nset(lines, "G1_HUB", "GEAR1-1", pair.gear1.support_nodes + 1)
    _append_assembly_nset(lines, "G2_HUB", "GEAR2-1", pair.gear2.support_nodes + 1)
    lines.extend(["*Nset, nset=G1_RP", "1000001", "*Nset, nset=G2_RP", "1000002"])
    _append_surface(lines, "G1_SURFACE", "GEAR1-1", pair.gear1.surface_entries)
    _append_surface(lines, "G2_SURFACE", "GEAR2-1", pair.gear2.surface_entries)
    lines.extend(["*MPC", "BEAM, G1_HUB, G1_RP", "*MPC", "BEAM, G2_HUB, G2_RP", "*End Assembly"])
    closure = float(pair.initial_patch_gap) + float(target_overclosure)
    final_translation = closure * pair.drive_direction
    final_rotation_z = float(rotation_rate_z) * float(duration)
    lines.extend(
        [
            "*Material, name=STEEL",
            "*Density",
            f"{density:.12e}",
            "*Elastic",
            f"{young:.12e}, {poisson:.12e}",
        ]
    )
    lines.extend(contact_lines)
    lines.extend(
        [
            "*Amplitude, name=RAMP",
            f"0., 0., {duration:.12e}, 1.",
            "*Step, name=alignment, nlgeom=YES, inc=2000",
            "*Dynamic, application=MODERATE DISSIPATION",
            f"{dt:.12e}, {duration:.12e}, {min(float(dt) * 1.0e-4, 1.0e-8):.12e}, {dt:.12e}",
            f"** Rotational boundary values are radians. Final G1_RP UR6 = {final_rotation_z:.12e} rad.",
            "*Boundary",
            "G2_RP, 1, 6",
            "G1_RP, 4, 5",
            "*Boundary, amplitude=RAMP",
            f"G1_RP, 1, 1, {final_translation[0]:.12e}",
            f"G1_RP, 2, 2, {final_translation[1]:.12e}",
            f"G1_RP, 3, 3, {final_translation[2]:.12e}",
            f"G1_RP, 6, 6, {final_rotation_z:.12e}",
            "*Output, field, frequency=1",
            "*Node Output",
            "U, RF",
            "*Element Output",
            "S, E, LE",
            "*End Step",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _append_assembly_nset(lines: list[str], name: str, instance: str, node_labels: np.ndarray) -> None:
    lines.append(f"*Nset, nset={name}, instance={instance}")
    labels = [int(v) for v in np.asarray(node_labels, dtype=np.int64).ravel()]
    for start in range(0, len(labels), 16):
        lines.append(", ".join(str(v) for v in labels[start : start + 16]))


def _append_surface(lines: list[str], name: str, instance: str, entries: tuple[tuple[int, str], ...]) -> None:
    by_side: dict[str, list[int]] = {}
    for element_label, side in entries:
        by_side.setdefault(side.upper(), []).append(int(element_label))
    for side, labels in sorted(by_side.items()):
        set_name = f"{name}_{side}"
        lines.append(f"*Elset, elset={set_name}, instance={instance}")
        for start in range(0, len(labels), 16):
            lines.append(", ".join(str(v) for v in labels[start : start + 16]))
    lines.append(f"*Surface, type=ELEMENT, name={name}")
    for side in sorted(by_side):
        lines.append(f"{name}_{side}, {side}")


def _resolve_abaqus_command(command: str | None) -> str:
    if command:
        return command
    for candidate in ("abaqus", "abq2024"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Abaqus command not found; pass --abaqus-command")


def _run_command(command: list[str], *, cwd: Path, log_path: Path) -> float:
    start = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    log_path.write_text(result.stdout, encoding="utf-8", errors="ignore")
    print(result.stdout)
    failed_text = result.stdout.lower()
    if result.returncode != 0 or "exited with errors" in failed_text or "fatal errors" in failed_text:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}")
    return elapsed


def _abaqus_wallclock_seconds(sta_path: Path) -> float | None:
    if not sta_path.exists():
        return None
    text = sta_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"WALLCLOCK TIME[^\d]*([0-9.]+)", text, flags=re.IGNORECASE)
    if matches:
        return float(matches[-1])
    matches = re.findall(r"TOTAL JOB TIME[^\d]*([0-9.]+)", text, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else None


def _abaqus_metrics_export_script() -> str:
    return r'''
from __future__ import print_function

import csv
import math
import sys

from odbAccess import openOdb


def norm3(data):
    return math.sqrt(float(data[0]) ** 2 + float(data[1]) ** 2 + float(data[2]) ** 2)


def parse_int_set(text):
    if not text:
        return set()
    return set(int(part) for part in text.split(",") if part)


def parse_vec3(text):
    values = [float(part) for part in text.split(",")]
    if len(values) != 3:
        raise RuntimeError("expected three comma-separated coordinates")
    return values


def dot3(a, b):
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def cross3(a, b):
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def add3(a, b):
    return [float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2])]


def tensor_norm(data):
    values = [float(v) for v in data]
    if len(values) < 6:
        return 0.0
    return math.sqrt(values[0] ** 2 + values[1] ** 2 + values[2] ** 2 + 2.0 * (values[3] ** 2 + values[4] ** 2 + values[5] ** 2))


def von_mises(data):
    values = [float(v) for v in data]
    if len(values) < 6:
        return 0.0
    s11, s22, s33, s12, s13, s23 = values[:6]
    return math.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * (s12 * s12 + s13 * s13 + s23 * s23))


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct / 100.0))
    return float(ordered[index])


def append_scalar_field(values, target):
    for value in values:
        data = value.data
        try:
            target.append(float(data))
        except Exception:
            target.append(float(data[0]))


def equivalent_elastic_strain_from_mises(mises, young, poisson):
    shear = float(young) / (2.0 * (1.0 + float(poisson)))
    return float(mises) / max(3.0 * shear, 1.0e-30)


odb = openOdb(path=sys.argv[1], readOnly=True)
out_path = sys.argv[2]
drive = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]
young = float(sys.argv[6])
poisson = float(sys.argv[7])
g1_hub_labels = parse_int_set(sys.argv[8])
g2_hub_labels = parse_int_set(sys.argv[9])
g1_rp = parse_vec3(sys.argv[10])
g2_rp = parse_vec3(sys.argv[11])
try:
    assembly = odb.rootAssembly
    instance_coords = {}
    for name, inst in assembly.instances.items():
        instance_coords[name.upper()] = dict((int(node.label), [float(v) for v in node.coordinates[:3]]) for node in inst.nodes)
    step = odb.steps[list(odb.steps.keys())[0]]
    with open(out_path, "w", newline="") as handle:
        fieldnames = [
            "time",
            "max_displacement_norm",
            "p95_von_mises",
            "max_von_mises",
            "p95_strain_norm",
            "max_strain_norm",
            "p95_log_strain_norm",
            "max_log_strain_norm",
            "p95_equivalent_elastic_strain",
            "max_equivalent_elastic_strain",
            "min_copen",
            "max_cpressure",
            "rp_force_drive",
            "rp_force_norm",
            "rp2_force_drive",
            "rp2_force_norm",
            "hub_virtual_force_drive",
            "hub_virtual_force_norm",
            "hub2_virtual_force_drive",
            "hub2_virtual_force_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame in step.frames:
            u_norm = []
            vm = []
            strain = []
            log_strain = []
            copen = []
            cpressure = []
            rp_force_drive = 0.0
            rp_force_norm = 0.0
            rp2_force_drive = 0.0
            rp2_force_norm = 0.0
            hub_force = [0.0, 0.0, 0.0]
            hub_moment = [0.0, 0.0, 0.0]
            hub2_force = [0.0, 0.0, 0.0]
            hub2_moment = [0.0, 0.0, 0.0]
            if "U" in frame.fieldOutputs:
                for value in frame.fieldOutputs["U"].values:
                    u_norm.append(norm3(value.data))
            if "S" in frame.fieldOutputs:
                for value in frame.fieldOutputs["S"].values:
                    try:
                        vm.append(float(value.mises))
                    except Exception:
                        vm.append(von_mises(value.data))
            if "E" in frame.fieldOutputs:
                for value in frame.fieldOutputs["E"].values:
                    strain.append(tensor_norm(value.data))
            if "LE" in frame.fieldOutputs:
                for value in frame.fieldOutputs["LE"].values:
                    log_strain.append(tensor_norm(value.data))
            if not strain:
                strain = list(log_strain)
            equivalent_strain = [equivalent_elastic_strain_from_mises(value, young, poisson) for value in vm]
            if "RF" in frame.fieldOutputs:
                for value in frame.fieldOutputs["RF"].values:
                    inst_name = ""
                    try:
                        inst_name = value.instance.name.upper()
                    except Exception:
                        inst_name = ""
                    if int(value.nodeLabel) == 1000001:
                        data = [float(v) for v in value.data[:3]]
                        rp_force_drive += data[0] * drive[0] + data[1] * drive[1] + data[2] * drive[2]
                        rp_force_norm = norm3(data)
                    if int(value.nodeLabel) == 1000002:
                        data = [float(v) for v in value.data[:3]]
                        rp2_force_drive += data[0] * drive[0] + data[1] * drive[1] + data[2] * drive[2]
                        rp2_force_norm = norm3(data)
                    if inst_name == "GEAR1-1" and int(value.nodeLabel) in g1_hub_labels:
                        data = [float(v) for v in value.data[:3]]
                        coords = instance_coords.get(inst_name, {}).get(int(value.nodeLabel), g1_rp)
                        lever = [coords[0] - g1_rp[0], coords[1] - g1_rp[1], coords[2] - g1_rp[2]]
                        hub_force = add3(hub_force, data)
                        hub_moment = add3(hub_moment, cross3(lever, data))
                    if inst_name == "GEAR2-1" and int(value.nodeLabel) in g2_hub_labels:
                        data = [float(v) for v in value.data[:3]]
                        coords = instance_coords.get(inst_name, {}).get(int(value.nodeLabel), g2_rp)
                        lever = [coords[0] - g2_rp[0], coords[1] - g2_rp[1], coords[2] - g2_rp[2]]
                        hub2_force = add3(hub2_force, data)
                        hub2_moment = add3(hub2_moment, cross3(lever, data))
            for name, output in frame.fieldOutputs.items():
                upper = name.upper()
                if "COPEN" in upper:
                    append_scalar_field(output.values, copen)
                if "CPRESS" in upper:
                    append_scalar_field(output.values, cpressure)
            writer.writerow({
                "time": float(frame.frameValue),
                "max_displacement_norm": max(u_norm) if u_norm else 0.0,
                "p95_von_mises": percentile(vm, 95.0),
                "max_von_mises": max(vm) if vm else 0.0,
                "p95_strain_norm": percentile(strain, 95.0),
                "max_strain_norm": max(strain) if strain else 0.0,
                "p95_log_strain_norm": percentile(log_strain, 95.0),
                "max_log_strain_norm": max(log_strain) if log_strain else 0.0,
                "p95_equivalent_elastic_strain": percentile(equivalent_strain, 95.0),
                "max_equivalent_elastic_strain": max(equivalent_strain) if equivalent_strain else 0.0,
                "min_copen": min(copen) if copen else 0.0,
                "max_cpressure": max(cpressure) if cpressure else 0.0,
                "rp_force_drive": rp_force_drive,
                "rp_force_norm": rp_force_norm,
                "rp2_force_drive": rp2_force_drive,
                "rp2_force_norm": rp2_force_norm,
                "hub_virtual_force_drive": dot3(hub_force, drive),
                "hub_virtual_force_norm": norm3(hub_force),
                "hub2_virtual_force_drive": dot3(hub2_force, drive),
                "hub2_virtual_force_norm": norm3(hub2_force),
            })
finally:
    odb.close()
'''


def run_abaqus_alignment(
    deck_path: Path,
    out_dir: Path,
    pair: CroppedGearPair,
    *,
    abaqus_command: str | None,
    young: float,
    poisson: float,
) -> tuple[Path, Row]:
    run_dir = out_dir / "abaqus_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    job_name = "cropped_gear_alignment"
    inp_path = run_dir / f"{job_name}.inp"
    for old in run_dir.glob(f"{job_name}.*"):
        old.unlink()
    inp_path.write_text(deck_path.read_text(encoding="ascii"), encoding="ascii")
    command = _resolve_abaqus_command(abaqus_command)
    analysis_wall = _run_command(
        [command, f"job={job_name}", f"input={inp_path.name}", "interactive"],
        cwd=run_dir,
        log_path=out_dir / "abaqus_analysis_stdout.log",
    )
    odb = run_dir / f"{job_name}.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    export_script = run_dir / "export_cropped_gear_metrics.py"
    export_script.write_text(_abaqus_metrics_export_script(), encoding="ascii")
    metrics = out_dir / "abaqus_cropped_gear_metrics.csv"
    export_wall = _run_command(
        [
            command,
            "python",
            str(export_script.resolve()),
            str(odb.resolve()),
            str(metrics.resolve()),
            f"{pair.drive_direction[0]:.16e}",
            f"{pair.drive_direction[1]:.16e}",
            f"{pair.drive_direction[2]:.16e}",
            f"{float(young):.16e}",
            f"{float(poisson):.16e}",
            ",".join(str(int(v) + 1) for v in pair.gear1.support_nodes),
            ",".join(str(int(v) + 1) for v in pair.gear2.support_nodes),
            ",".join(f"{float(v):.16e}" for v in pair.gear1.rp),
            ",".join(f"{float(v):.16e}" for v in pair.gear2.rp),
        ],
        cwd=run_dir,
        log_path=out_dir / "abaqus_export_stdout.log",
    )
    row: Row = {
        "abaqus_analysis_wall_seconds": analysis_wall,
        "abaqus_export_wall_seconds": export_wall,
        "abaqus_reported_wall_seconds": _abaqus_wallclock_seconds(run_dir / f"{job_name}.sta") or 0.0,
        "abaqus_metrics": str(metrics),
        "status": "completed",
    }
    return metrics, row


def _read_csv_rows(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_float_column(rows: list[Row], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0) or 0.0) for row in rows], dtype=float)


def _relative_error(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(b)), 1.0e-12)


def _metric_available(rows: list[Row], key: str) -> bool:
    return bool(rows) and key in rows[0]


def _preferred_strain_metric(sfc_rows: list[Row], abaqus_rows: list[Row]) -> tuple[str, str, str]:
    """Return the primary strain scalar used for Abaqus/SFC alignment.

    We keep raw tensor strain norms in the CSV, but the paper-facing scalar is
    the equivalent elastic strain recovered from von Mises stress.  This avoids
    mixing SFC Green strain diagnostics with Abaqus element output conventions.
    """

    if _metric_available(sfc_rows, "p95_equivalent_elastic_strain") and _metric_available(
        abaqus_rows,
        "p95_equivalent_elastic_strain",
    ):
        return "p95_equivalent_elastic_strain", "p95_equivalent_elastic_strain", "p95 equivalent elastic strain"
    return "p95_strain_norm", "p95_strain_norm", "p95 strain norm"


def _preferred_sfc_reaction_keys(sfc_rows: list[Row]) -> tuple[str, str, str]:
    """Return SFC RP reaction keys aligned with Abaqus/Standard RF output."""

    if sfc_rows and "rp1_hht_rp_force_norm" in sfc_rows[0] and "rp2_hht_rp_force_norm" in sfc_rows[0]:
        if np.any(np.abs(_as_float_column(sfc_rows, "rp1_hht_rp_force_norm")) > 0.0):
            return "rp1_hht_rp_force_norm", "rp2_hht_rp_force_norm", "HHT RP reaction norm"
    if sfc_rows and "rp_dynamic_force_norm" in sfc_rows[0] and "opposing_rp_dynamic_force_norm" in sfc_rows[0]:
        if np.any(np.abs(_as_float_column(sfc_rows, "rp_dynamic_force_norm")) > 0.0):
            return "rp_dynamic_force_norm", "opposing_rp_dynamic_force_norm", "dynamic RP reaction norm"
    force_key = "rp_force_norm" if sfc_rows and "rp_force_norm" in sfc_rows[0] else "normal_force"
    opposing_force_key = "opposing_rp_force_norm" if sfc_rows and "opposing_rp_force_norm" in sfc_rows[0] else force_key
    return force_key, opposing_force_key, "RP/contact reaction norm"


def compare_histories(sfc_history: Path, abaqus_history: Path, out_path: Path) -> list[Row]:
    sfc_rows = _read_csv_rows(sfc_history)
    abaqus_rows = _read_csv_rows(abaqus_history)
    t_sfc = _as_float_column(sfc_rows, "time")
    t_abaqus = _as_float_column(abaqus_rows, "time")
    force_key, opposing_force_key, _force_title = _preferred_sfc_reaction_keys(sfc_rows)
    abaqus_force_key = (
        "hub_virtual_force_norm"
        if abaqus_rows
        and "hub_virtual_force_norm" in abaqus_rows[0]
        and np.any(np.abs(_as_float_column(abaqus_rows, "hub_virtual_force_norm")) > 0.0)
        else "rp_force_norm"
    )
    abaqus_opposing_force_key = (
        "hub2_virtual_force_norm"
        if abaqus_rows
        and "hub2_virtual_force_norm" in abaqus_rows[0]
        and np.any(np.abs(_as_float_column(abaqus_rows, "hub2_virtual_force_norm")) > 0.0)
        else "rp2_force_norm"
    )
    strain_key, abaqus_strain_key, _strain_title = _preferred_strain_metric(sfc_rows, abaqus_rows)
    metrics = [
        ("max_displacement_norm", "max_displacement_norm"),
        ("p95_von_mises", "p95_von_mises"),
        ("max_von_mises", "max_von_mises"),
        (strain_key, abaqus_strain_key),
        ("max_strain_norm", "max_strain_norm"),
        (force_key, abaqus_force_key),
        (opposing_force_key, abaqus_opposing_force_key),
    ]
    if (
        _metric_available(sfc_rows, "max_contact_pressure")
        and _metric_available(abaqus_rows, "max_cpressure")
        and np.any(np.abs(_as_float_column(abaqus_rows, "max_cpressure")) > 0.0)
    ):
        metrics.append(("max_contact_pressure", "max_cpressure"))
    if t_abaqus.size == 0:
        raise RuntimeError(f"Abaqus metrics file has no frames: {abaqus_history}")
    rows: list[Row] = []
    for i, t in enumerate(t_sfc):
        row: Row = {"time": float(t)}
        for sfc_key, abaqus_key in metrics:
            sfc_value = float(sfc_rows[i].get(sfc_key, 0.0) or 0.0)
            abaqus_values = _as_float_column(abaqus_rows, abaqus_key)
            abaqus_value = float(np.interp(t, t_abaqus, abaqus_values))
            row[f"sfc_{sfc_key}"] = sfc_value
            row[f"abaqus_{abaqus_key}"] = abaqus_value
            row[f"{sfc_key}_abs_error"] = abs(sfc_value - abaqus_value)
            row[f"{sfc_key}_rel_error"] = _relative_error(sfc_value, abaqus_value)
        rows.append(row)
    _write_csv(out_path, rows)
    return rows


def plot_alignment_curves(sfc_history: Path, abaqus_history: Path, error_rows: list[Row], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    sfc_rows = _read_csv_rows(sfc_history)
    abaqus_rows = _read_csv_rows(abaqus_history)
    t_sfc = _as_float_column(sfc_rows, "time")
    t_abq = _as_float_column(abaqus_rows, "time")
    _force_key, opposing_force_key, force_title = _preferred_sfc_reaction_keys(sfc_rows)
    abaqus_opposing_force_key = (
        "hub2_virtual_force_norm"
        if abaqus_rows
        and "hub2_virtual_force_norm" in abaqus_rows[0]
        and np.any(np.abs(_as_float_column(abaqus_rows, "hub2_virtual_force_norm")) > 0.0)
        else "rp2_force_norm"
    )
    strain_key, abaqus_strain_key, strain_title = _preferred_strain_metric(sfc_rows, abaqus_rows)
    panels = [
        ("max_displacement_norm", "max_displacement_norm", "max displacement norm"),
        ("p95_von_mises", "p95_von_mises", "p95 von Mises stress"),
        (strain_key, abaqus_strain_key, strain_title),
        (opposing_force_key, abaqus_opposing_force_key, f"fixed {force_title}"),
    ]

    def with_initial_zero(time_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if time_values.size == 1 and float(time_values[0]) > 0.0:
            return np.concatenate(([0.0], time_values)), np.concatenate(([0.0], values))
        return time_values, values

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for ax, (sfc_key, abq_key, title) in zip(axes.ravel(), panels, strict=True):
        sfc_y = _as_float_column(sfc_rows, sfc_key)
        abq_y = _as_float_column(abaqus_rows, abq_key)
        t_sfc_plot, sfc_y_plot = with_initial_zero(t_sfc, sfc_y)
        t_abq_plot, abq_y_plot = with_initial_zero(t_abq, abq_y)
        rel_key = f"{sfc_key}_rel_error"
        final_error = float(error_rows[-1].get(rel_key, 0.0)) if error_rows else 0.0
        ax.plot(
            t_abq_plot,
            abq_y_plot,
            color="#1f77b4",
            linewidth=1.8,
            marker="o",
            markersize=3.6,
            label="Abaqus/Standard",
            zorder=2,
        )
        ax.plot(
            t_sfc_plot,
            sfc_y_plot,
            color="#d95f02",
            linewidth=1.8,
            linestyle="--",
            marker="s",
            markersize=3.8,
            markerfacecolor="white",
            label=f"SFC ({final_error * 100:.2f}% final err.)",
            zorder=3,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (s)", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.legend(loc="best", fontsize=8, frameon=False)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_summary(
    path: Path,
    summary: Row,
    history_path: Path,
    deck_path: Path,
    *,
    abaqus_row: Row | None = None,
    error_rows: list[Row] | None = None,
    figure_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = [
        "# Cropped Gear Lagrangian-SDF Implicit Alignment",
        "",
        "This runner uses SFC internal TET4 HHT dynamics, Abaqus-style RP/hub MPC kinematics in eliminated Dirichlet form, and Lagrangian-SDF surface contact tangent.",
        "",
        f"- contact mode: {summary.get('contact_mode', 'penalty')}",
        f"- hard enforcement: {summary.get('hard_enforcement', '')}",
        f"- constraint averaging: {summary.get('constraint_averaging', '')}",
        f"- linear solver: {summary.get('linear_solver', 'dense')}",
        f"- effective hard pressure stiffness: {float(summary.get('effective_hard_pressure_stiffness', 0.0)):.6e}",
        f"- pressure smoothing factor: {float(summary.get('pressure_smoothing_factor', 0.0)):.6e}",
        f"- HHT alpha/beta/gamma: {float(summary.get('hht_alpha', 0.0)):.6e} / {float(summary.get('hht_beta', 0.0)):.6e} / {float(summary.get('hht_gamma', 0.0)):.6e}",
        f"- RP reaction definition: {summary.get('rp_reaction_definition', 'static_physical_constraint_residual')}",
        f"- automatic increment: {summary.get('automatic_increment', '')}",
        f"- cutback count: {int(summary.get('cutback_count', 0))}",
        f"- SFC wall time: {summary['sfc_wall_seconds']:.6f} s",
        f"- timing internal+tangent: {float(summary.get('timing_internal_tangent_seconds', 0.0)):.6f} s",
        f"- timing effective system: {float(summary.get('timing_effective_system_seconds', 0.0)):.6f} s",
        f"- timing hard-contact solve: {float(summary.get('timing_hard_contact_solve_seconds', 0.0)):.6f} s",
        f"- timing contact linearization: {float(summary.get('timing_contact_linearization_seconds', 0.0)):.6f} s",
        f"- timing accepted-state diagnostics: {float(summary.get('timing_accepted_internal_seconds', 0.0)) + float(summary.get('timing_accepted_contact_seconds', 0.0)) + float(summary.get('timing_reaction_diagnostics_seconds', 0.0)):.6f} s",
        f"- nodes/elements: {summary['nodes']} / {summary['elements']}",
        f"- contact faces: {summary['gear1_contact_faces']} / {summary['gear2_contact_faces']}",
        f"- support nodes: {summary['gear1_support_nodes']} / {summary['gear2_support_nodes']}",
        f"- initial patch gap: {summary['initial_patch_gap']:.6e}",
        f"- target overclosure: {summary['target_overclosure']:.6e}",
        f"- final active samples: {summary['final_active_contact_samples']}",
        f"- final min gap: {summary['final_min_gap']:.6e}",
        f"- final normal force: {summary['final_normal_force']:.6e}",
        f"- final RP reaction norm: {summary.get('final_rp_force_norm', 0.0):.6e}",
        f"- final opposing RP reaction norm: {summary.get('final_opposing_rp_force_norm', 0.0):.6e}",
        f"- final hard outer iterations: {summary.get('final_hard_outer_iterations', '')}",
        f"- final hard outer converged: {summary.get('final_hard_outer_converged', '')}",
        f"- history CSV: `{history_path.name}`",
        f"- optional Abaqus alignment deck: `{deck_path.name}`",
    ]
    if abaqus_row is not None:
        text.extend(
            [
                "",
                "## Abaqus Native-Contact Alignment",
                "",
                f"- Abaqus analysis wall time: {float(abaqus_row.get('abaqus_analysis_wall_seconds', 0.0)):.6f} s",
                f"- Abaqus reported wall time: {float(abaqus_row.get('abaqus_reported_wall_seconds', 0.0)):.6f} s",
                f"- Abaqus export wall time: {float(abaqus_row.get('abaqus_export_wall_seconds', 0.0)):.6f} s",
            ]
        )
    if error_rows:
        final = error_rows[-1]
        strain_error_key = (
            "p95_equivalent_elastic_strain_rel_error"
            if "p95_equivalent_elastic_strain_rel_error" in final
            else "p95_strain_norm_rel_error"
        )
        strain_label = "p95 equivalent elastic strain" if strain_error_key.startswith("p95_equivalent") else "p95 strain norm"
        text.extend(
            [
                "",
                "## Final-Time Alignment Errors",
                "",
                f"- max displacement norm rel. error: {100.0 * float(final.get('max_displacement_norm_rel_error', 0.0)):.3f}%",
                f"- p95 von Mises rel. error: {100.0 * float(final.get('p95_von_mises_rel_error', 0.0)):.3f}%",
                f"- {strain_label} rel. error: {100.0 * float(final.get(strain_error_key, 0.0)):.3f}%",
                f"- RP/contact reaction rel. error: {100.0 * float(final.get('rp_force_norm_rel_error', final.get('normal_force_rel_error', 0.0))):.3f}%",
                f"- fixed-RP reaction rel. error: {100.0 * float(final.get('opposing_rp_force_norm_rel_error', 0.0)):.3f}%",
            ]
        )
    if figure_path is not None:
        text.extend(["", f"- alignment figure: `{figure_path.name}`"])
    text.extend(["", "Abaqus is not used by the SFC solve path. The deck is emitted only for external native-contact comparison."])
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--faces-per-body", type=int, default=10)
    parser.add_argument("--expansion-rings", type=int, default=1)
    parser.add_argument("--duration", type=float, default=2.0e-3)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--overclosure", type=float, default=3.0e-4)
    parser.add_argument(
        "--rotation-rate-z",
        "--rotation-rate-z-rad-s",
        dest="rotation_rate_z",
        type=float,
        default=2.0,
        help="Prescribed G1 RP angular velocity about z in radians per second.",
    )
    parser.add_argument("--pressure-stiffness", type=float, default=DEFAULT_PRESSURE_STIFFNESS)
    parser.add_argument("--contact-mode", choices=("penalty", "hard"), default="penalty")
    parser.add_argument(
        "--hard-enforcement",
        choices=("exact", "pressure_compliance", "element_pressure_smoothing", "abaqus_standard_penalty"),
        default="abaqus_standard_penalty",
    )
    parser.add_argument("--pressure-smoothing-factor", type=float, default=4.0)
    parser.add_argument("--hht-alpha", type=float, default=ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA)
    parser.add_argument("--no-automatic-increment", action="store_true")
    parser.add_argument("--cutback-factor", type=float, default=0.5)
    parser.add_argument("--constraint-averaging", choices=("none", "slave_face", "surface_patch"), default="slave_face")
    parser.add_argument("--hard-max-iterations", type=int, default=6)
    parser.add_argument("--linear-solver", choices=("dense", "sparse", "auto"), default="dense")
    parser.add_argument("--run-abaqus", action="store_true", help="Run the generated Abaqus native-contact deck and compare curves.")
    parser.add_argument("--abaqus-command", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Use the default small cropped patch settings.")
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    source = args.source
    if not source.exists():
        raise FileNotFoundError(source)
    model = parse_gear_input(source)
    pair = build_cropped_pair(model, faces_per_body=int(args.faces_per_body), expansion_rings=int(args.expansion_rings))
    if str(args.contact_mode).lower() == "hard":
        history, summary = solve_sfc_cropped_pair_hard_contact(
            pair,
            young=model.young,
            poisson=model.poisson,
            density=model.density,
            pressure_stiffness=float(args.pressure_stiffness),
            duration=float(args.duration),
            dt=float(args.dt),
            target_overclosure=float(args.overclosure),
            rotation_rate_z=float(args.rotation_rate_z),
            max_iterations=int(args.hard_max_iterations),
            hard_enforcement=str(args.hard_enforcement),
            constraint_averaging=str(args.constraint_averaging),
            pressure_smoothing_factor=float(args.pressure_smoothing_factor),
            hht_alpha=float(args.hht_alpha),
            automatic_increment=not bool(args.no_automatic_increment),
            cutback_factor=float(args.cutback_factor),
            linear_solver=str(args.linear_solver),
        )
    else:
        history, summary = solve_sfc_cropped_pair(
            pair,
            young=model.young,
            poisson=model.poisson,
            density=model.density,
            pressure_stiffness=float(args.pressure_stiffness),
            duration=float(args.duration),
            dt=float(args.dt),
            target_overclosure=float(args.overclosure),
            rotation_rate_z=float(args.rotation_rate_z),
        )
    history_path = out_dir / "sfc_cropped_gear_lagrangian_sdf_history.csv"
    summary_path = out_dir / "cropped_gear_lagrangian_sdf_summary.csv"
    deck_path = out_dir / "abaqus_cropped_gear_alignment.inp"
    report_path = out_dir / "cropped_gear_lagrangian_sdf_summary.md"
    _write_csv(history_path, history)
    _write_csv(summary_path, [summary])
    _write_abaqus_alignment_deck(
        deck_path,
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=float(args.pressure_stiffness),
        duration=float(args.duration),
        dt=float(args.dt),
        target_overclosure=float(args.overclosure),
        rotation_rate_z=float(args.rotation_rate_z),
        contact_mode=str(args.contact_mode),
    )
    abaqus_row: Row | None = None
    error_rows: list[Row] | None = None
    figure_path: Path | None = None
    if args.run_abaqus:
        abaqus_metrics, abaqus_row = run_abaqus_alignment(
            deck_path,
            out_dir,
            pair,
            abaqus_command=args.abaqus_command,
            young=model.young,
            poisson=model.poisson,
        )
        _write_csv(out_dir / "abaqus_runtime.csv", [abaqus_row])
        error_rows = compare_histories(history_path, abaqus_metrics, out_dir / "sfc_vs_abaqus_alignment_errors.csv")
        figure_path = out_dir / "sfc_vs_abaqus_alignment_curves.png"
        plot_alignment_curves(history_path, abaqus_metrics, error_rows, figure_path)
    write_summary(report_path, summary, history_path, deck_path, abaqus_row=abaqus_row, error_rows=error_rows, figure_path=figure_path)
    print(report_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
