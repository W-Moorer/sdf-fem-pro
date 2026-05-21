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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
)
from sfc.contact.lagrangian_surface_contact import LagrangianSDFSurfaceContactGeometry  # noqa: E402
from sfc.fem.calculix_aligned import MechanicsModel, stvk_internal_response  # noqa: E402
from sfc.fem.calculix_aligned import calculix_dynamic_predictor, hht_newmark_parameters  # noqa: E402
from sfc.fem.constraints import project_fixed_dofs  # noqa: E402
from sfc.fem.implicit_dirichlet import hht_step_dirichlet, initial_state_dirichlet  # noqa: E402
from sfc.fem.rp_mpc import RigidHubMPC, merge_dirichlet_conditions  # noqa: E402
from sfc.sdf.material_sdf import MaterialSDF  # noqa: E402
from validation.run_flexible_gear_explicit_sdf_comparison import (  # noqa: E402
    DEFAULT_SOURCE,
    GearInputModel,
    GearMesh,
    parse_gear_input,
)

Row = dict[str, Any]

DEFAULT_OUT_DIR = ROOT / "results" / "flexible_gear_implicit_lagrangian_sdf"
DEFAULT_PRESSURE_STIFFNESS = 5.0e9


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
) -> tuple[list[Row], Row]:
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
            alpha=-0.05,
            max_iterations=10,
            tolerance=1.0e-9,
            acceptance_policy="relative_correction",
        )
        internal = diagnostics.internal
        disp = state.x - model.X
        strain_norm = np.linalg.norm(internal.strain, axis=(1, 2)) if internal.strain.size else np.empty(0, dtype=float)
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
    if averaging not in {"none", "slave_face"}:
        raise ValueError("constraint_averaging must be 'none' or 'slave_face'")
    if averaging == "slave_face" and gap_at_linearization.size >= 3 and gap_at_linearization.size % 3 == 0:
        aggregated_gaps: list[float] = []
        aggregated_rows: list[np.ndarray] = []
        aggregated_areas: list[float] = []
        for start in range(0, gap_at_linearization.size, 3):
            stop = start + 3
            group_area = areas[start:stop]
            total_area = float(np.sum(group_area))
            if total_area <= 0.0:
                weights = np.full(3, 1.0 / 3.0, dtype=float)
            else:
                weights = group_area / total_area
            aggregated_gaps.append(float(weights @ gap_at_linearization[start:stop]))
            aggregated_rows.append(weights @ gap_jacobian[start:stop])
            aggregated_areas.append(total_area)
        gap_at_linearization = np.asarray(aggregated_gaps, dtype=float)
        gap_jacobian = np.vstack(aggregated_rows)
        areas = np.asarray(aggregated_areas, dtype=float)
    gap_offset = gap_at_linearization - gap_jacobian @ np.asarray(u_linearization, dtype=float).reshape(-1)
    return samples, gap_offset, gap_jacobian, areas


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
    hard_enforcement: str = "exact",
    constraint_averaging: str = "slave_face",
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
    if enforcement not in {"exact", "pressure_compliance"}:
        raise ValueError("hard_enforcement must be 'exact' or 'pressure_compliance'")
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
    rows: list[Row] = []
    steps = max(1, int(round(float(duration) / float(dt))))
    beta, gamma = hht_newmark_parameters(0.0)
    c0 = 1.0 / (beta * float(dt) * float(dt))
    start = time.perf_counter()
    last_iterations = 0
    last_converged = True
    for step in range(1, steps + 1):
        t = step * float(dt)
        ramp = t / max(float(duration), float(dt))
        closure = ramp * (pair.initial_patch_gap + float(target_overclosure))
        rotation = float(rotation_rate_z) * t
        fixed, values = _fixed_conditions_for_pair(pair, closure=closure, rotation_z=rotation)
        u_n = (state_x - model.X).reshape(-1)
        v_n = velocity.reshape(-1)
        a_n = acceleration.reshape(-1)
        u_pred, v_pred, _accold_after_prediction = calculix_dynamic_predictor(u_n, v_n, a_n, dt=float(dt), beta=beta, gamma=gamma)
        u_guess = project_fixed_dofs(u_pred.copy(), fixed, values)
        solution = None
        final_samples: list[Any] = []
        converged = False
        iteration_count = 0
        for iteration in range(1, max(1, int(max_iterations)) + 1):
            x_guess = model.X + u_guess.reshape((-1, 3))
            internal = stvk_internal_response(model, x_guess, assemble_tangent=True)
            final_samples, gap_offset, gap_jacobian, constraint_areas = _hard_contact_linearized_gap_jacobian(
                contact,
                x_guess,
                u_guess,
                n_total_dofs=model.n_dofs,
                constraint_averaging=constraint_averaging,
            )
            compliance = (
                None
                if enforcement == "exact"
                else 1.0 / (float(pressure_stiffness) * np.maximum(constraint_areas, 1.0e-30))
            )
            tangent = internal.tangent.tocsr()
            effective_stiffness = (tangent + model.mass_matrix * c0).tocsr()
            effective_force = np.asarray(model.mass_matrix @ (c0 * u_pred) - internal.force.reshape(-1) + tangent @ u_guess, dtype=float)
            solution = solve_linear_hard_contact_with_dirichlet(
                effective_stiffness,
                effective_force,
                gap_offset,
                gap_jacobian,
                fixed_dofs=fixed,
                fixed_values=values,
                normal_compliance=compliance,
                tolerance=float(tolerance),
                max_iterations=30,
            )
            correction_norm = float(np.linalg.norm(solution.displacement - u_guess))
            displacement_scale = max(1.0, float(np.linalg.norm(solution.displacement)))
            u_guess = solution.displacement.copy()
            iteration_count = iteration
            if correction_norm <= float(tolerance) * displacement_scale and bool(solution.converged):
                converged = True
                break
        if solution is None:
            raise RuntimeError("hard contact solve did not run")
        u_new = solution.displacement
        a_new = c0 * (u_new - u_pred)
        v_new = v_pred + gamma * float(dt) * a_new
        state_x = model.X + u_new.reshape((-1, 3))
        velocity = v_new.reshape((-1, 3))
        acceleration = a_new.reshape((-1, 3))
        internal = stvk_internal_response(model, state_x, assemble_tangent=False)
        raw_gaps = np.asarray([float(sample.gap) for sample in contact.samples(state_x)], dtype=float)
        active = np.asarray(solution.active, dtype=bool)
        multipliers = np.asarray(solution.multipliers, dtype=float)
        active_force = float(np.sum(multipliers[active])) if multipliers.size else 0.0
        disp = state_x - model.X
        strain_norm = np.linalg.norm(internal.strain, axis=(1, 2)) if internal.strain.size else np.empty(0, dtype=float)
        rows.append(
            {
                "time": t,
                "closure": closure,
                "rotation_z": rotation,
                "active_contact_samples": int(np.count_nonzero(active)),
                "min_gap": float(np.min(raw_gaps)) if raw_gaps.size else 0.0,
                "linearized_min_gap": float(np.min(solution.gaps)) if solution.gaps.size else 0.0,
                "normal_force": active_force,
                "max_displacement_norm": float(np.max(np.linalg.norm(disp, axis=1))),
                "p95_von_mises": float(np.percentile(internal.von_mises, 95.0)) if internal.von_mises.size else 0.0,
                "max_von_mises": float(np.max(internal.von_mises)) if internal.von_mises.size else 0.0,
                "p95_strain_norm": float(np.percentile(strain_norm, 95.0)) if strain_norm.size else 0.0,
                "max_strain_norm": float(np.max(strain_norm)) if strain_norm.size else 0.0,
                "strain_energy": internal.strain_energy,
                "newton_iterations": int(iteration_count),
                "hard_active_set_converged": int(bool(solution.converged)),
                "hard_outer_converged": int(bool(converged)),
                "hard_contact_samples": int(len(final_samples)),
                "hard_constraints": int(solution.gaps.shape[0]),
            }
        )
        last_iterations = iteration_count
        last_converged = bool(converged)
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
        "final_hard_outer_iterations": int(last_iterations),
        "final_hard_outer_converged": int(last_converged),
        "contact_mode": "hard",
        "hard_enforcement": enforcement,
        "constraint_averaging": str(constraint_averaging),
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
            "*Dynamic",
            f"{dt:.12e}, {duration:.12e}, {min(float(dt) * 1.0e-4, 1.0e-8):.12e}, {dt:.12e}",
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
            "S, LE",
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


odb = openOdb(path=sys.argv[1], readOnly=True)
out_path = sys.argv[2]
drive = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]
try:
    step = odb.steps[list(odb.steps.keys())[0]]
    with open(out_path, "w", newline="") as handle:
        fieldnames = [
            "time",
            "max_displacement_norm",
            "p95_von_mises",
            "max_von_mises",
            "p95_strain_norm",
            "max_strain_norm",
            "min_copen",
            "max_cpressure",
            "rp_force_drive",
            "rp_force_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame in step.frames:
            u_norm = []
            vm = []
            strain = []
            copen = []
            cpressure = []
            rp_force_drive = 0.0
            rp_force_norm = 0.0
            if "U" in frame.fieldOutputs:
                for value in frame.fieldOutputs["U"].values:
                    u_norm.append(norm3(value.data))
            if "S" in frame.fieldOutputs:
                for value in frame.fieldOutputs["S"].values:
                    try:
                        vm.append(float(value.mises))
                    except Exception:
                        vm.append(von_mises(value.data))
            if "LE" in frame.fieldOutputs:
                for value in frame.fieldOutputs["LE"].values:
                    strain.append(tensor_norm(value.data))
            if "RF" in frame.fieldOutputs:
                for value in frame.fieldOutputs["RF"].values:
                    if int(value.nodeLabel) == 1000001:
                        data = [float(v) for v in value.data[:3]]
                        rp_force_drive += data[0] * drive[0] + data[1] * drive[1] + data[2] * drive[2]
                        rp_force_norm = norm3(data)
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
                "min_copen": min(copen) if copen else 0.0,
                "max_cpressure": max(cpressure) if cpressure else 0.0,
                "rp_force_drive": rp_force_drive,
                "rp_force_norm": rp_force_norm,
            })
finally:
    odb.close()
'''


def run_abaqus_alignment(deck_path: Path, out_dir: Path, pair: CroppedGearPair, *, abaqus_command: str | None) -> tuple[Path, Row]:
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


def compare_histories(sfc_history: Path, abaqus_history: Path, out_path: Path) -> list[Row]:
    sfc_rows = _read_csv_rows(sfc_history)
    abaqus_rows = _read_csv_rows(abaqus_history)
    t_sfc = _as_float_column(sfc_rows, "time")
    t_abaqus = _as_float_column(abaqus_rows, "time")
    metrics = [
        ("max_displacement_norm", "max_displacement_norm"),
        ("p95_von_mises", "p95_von_mises"),
        ("max_von_mises", "max_von_mises"),
        ("p95_strain_norm", "p95_strain_norm"),
        ("max_strain_norm", "max_strain_norm"),
        ("normal_force", "rp_force_norm"),
    ]
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
    panels = [
        ("max_displacement_norm", "max_displacement_norm", "max displacement norm"),
        ("p95_von_mises", "p95_von_mises", "p95 von Mises stress"),
        ("p95_strain_norm", "p95_strain_norm", "p95 strain norm"),
        ("normal_force", "rp_force_norm", "normal/contact force"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for ax, (sfc_key, abq_key, title) in zip(axes.ravel(), panels, strict=True):
        sfc_y = _as_float_column(sfc_rows, sfc_key)
        abq_y = _as_float_column(abaqus_rows, abq_key)
        rel_key = f"{sfc_key}_rel_error"
        final_error = float(error_rows[-1].get(rel_key, 0.0)) if error_rows else 0.0
        ax.plot(t_abq, abq_y, color="#1f77b4", linewidth=1.8, label="Abaqus/Standard")
        ax.plot(t_sfc, sfc_y, color="#ff7f0e", linewidth=1.8, linestyle="--", label=f"SFC ({final_error * 100:.2f}% final err.)")
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
        f"- SFC wall time: {summary['sfc_wall_seconds']:.6f} s",
        f"- nodes/elements: {summary['nodes']} / {summary['elements']}",
        f"- contact faces: {summary['gear1_contact_faces']} / {summary['gear2_contact_faces']}",
        f"- support nodes: {summary['gear1_support_nodes']} / {summary['gear2_support_nodes']}",
        f"- initial patch gap: {summary['initial_patch_gap']:.6e}",
        f"- target overclosure: {summary['target_overclosure']:.6e}",
        f"- final active samples: {summary['final_active_contact_samples']}",
        f"- final min gap: {summary['final_min_gap']:.6e}",
        f"- final normal force: {summary['final_normal_force']:.6e}",
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
        text.extend(
            [
                "",
                "## Final-Time Alignment Errors",
                "",
                f"- max displacement norm rel. error: {100.0 * float(final.get('max_displacement_norm_rel_error', 0.0)):.3f}%",
                f"- p95 von Mises rel. error: {100.0 * float(final.get('p95_von_mises_rel_error', 0.0)):.3f}%",
                f"- p95 strain norm rel. error: {100.0 * float(final.get('p95_strain_norm_rel_error', 0.0)):.3f}%",
                f"- contact force rel. error: {100.0 * float(final.get('normal_force_rel_error', 0.0)):.3f}%",
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
    parser.add_argument("--rotation-rate-z", type=float, default=2.0)
    parser.add_argument("--pressure-stiffness", type=float, default=DEFAULT_PRESSURE_STIFFNESS)
    parser.add_argument("--contact-mode", choices=("penalty", "hard"), default="penalty")
    parser.add_argument("--hard-enforcement", choices=("exact", "pressure_compliance"), default="exact")
    parser.add_argument("--constraint-averaging", choices=("none", "slave_face"), default="slave_face")
    parser.add_argument("--hard-max-iterations", type=int, default=6)
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
        abaqus_metrics, abaqus_row = run_abaqus_alignment(deck_path, out_dir, pair, abaqus_command=args.abaqus_command)
        _write_csv(out_dir / "abaqus_runtime.csv", [abaqus_row])
        error_rows = compare_histories(history_path, abaqus_metrics, out_dir / "sfc_vs_abaqus_alignment_errors.csv")
        figure_path = out_dir / "sfc_vs_abaqus_alignment_curves.png"
        plot_alignment_curves(history_path, abaqus_metrics, error_rows, figure_path)
    write_summary(report_path, summary, history_path, deck_path, abaqus_row=abaqus_row, error_rows=error_rows, figure_path=figure_path)
    print(report_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
