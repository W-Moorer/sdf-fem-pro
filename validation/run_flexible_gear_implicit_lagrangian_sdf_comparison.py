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

from sfc.contact.lagrangian_surface_contact import LagrangianSDFSurfaceContactGeometry  # noqa: E402
from sfc.fem.calculix_aligned import MechanicsModel, stvk_internal_response  # noqa: E402
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


def _write_abaqus_alignment_deck(path: Path, pair: CroppedGearPair, *, young: float, poisson: float, density: float, pressure_stiffness: float, duration: float, dt: float) -> None:
    """Write a compact Abaqus deck with matching cropped meshes and RP MPCs."""

    path.parent.mkdir(parents=True, exist_ok=True)
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
    lines.extend(
        [
            "*Material, name=STEEL",
            "*Density",
            f"{density:.12e}",
            "*Elastic",
            f"{young:.12e}, {poisson:.12e}",
            "*Surface Interaction, name=SFC_LINEAR",
            "1.",
            "*Friction",
            "0.",
            "*Surface Behavior, pressure-overclosure=LINEAR",
            f"{pressure_stiffness:.12e}",
            "*Contact Pair, interaction=SFC_LINEAR, type=SURFACE TO SURFACE",
            "G1_SURFACE, G2_SURFACE",
            "*Amplitude, name=RAMP",
            f"0., 0., {duration:.12e}, 1.",
            "*Step, name=alignment, nlgeom=YES, inc=2000",
            "*Dynamic",
            f"{dt:.12e}, {duration:.12e}, {dt:.12e}, {dt:.12e}",
            "*Boundary",
            "G2_RP, 1, 6",
            "*Output, field, frequency=1",
            "*Node Output",
            "U",
            "*Element Output",
            "S, LE",
            "*Contact Output",
            "CPRESS, COPEN",
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


def write_summary(path: Path, summary: Row, history_path: Path, deck_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = [
        "# Cropped Gear Lagrangian-SDF Implicit Alignment",
        "",
        "This runner uses SFC internal TET4 HHT dynamics, Abaqus-style RP/hub MPC kinematics in eliminated Dirichlet form, and Lagrangian-SDF surface contact tangent.",
        "",
        f"- SFC wall time: {summary['sfc_wall_seconds']:.6f} s",
        f"- nodes/elements: {summary['nodes']} / {summary['elements']}",
        f"- contact faces: {summary['gear1_contact_faces']} / {summary['gear2_contact_faces']}",
        f"- support nodes: {summary['gear1_support_nodes']} / {summary['gear2_support_nodes']}",
        f"- initial patch gap: {summary['initial_patch_gap']:.6e}",
        f"- target overclosure: {summary['target_overclosure']:.6e}",
        f"- final active samples: {summary['final_active_contact_samples']}",
        f"- final min gap: {summary['final_min_gap']:.6e}",
        f"- final normal force: {summary['final_normal_force']:.6e}",
        f"- history CSV: `{history_path.name}`",
        f"- optional Abaqus alignment deck: `{deck_path.name}`",
        "",
        "Abaqus is not used by the SFC solve path. The deck is emitted only for external native-contact comparison.",
    ]
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
    parser.add_argument("--quick", action="store_true", help="Use the default small cropped patch settings.")
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    source = args.source
    if not source.exists():
        raise FileNotFoundError(source)
    model = parse_gear_input(source)
    pair = build_cropped_pair(model, faces_per_body=int(args.faces_per_body), expansion_rings=int(args.expansion_rings))
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
    )
    write_summary(report_path, summary, history_path, deck_path)
    print(report_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
