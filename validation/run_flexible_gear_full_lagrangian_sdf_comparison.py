"""Full flexible-gear implicit Lagrangian-SDF contact runner.

This validation runner keeps the complete two-gear TET4 volume meshes and
selects only the currently relevant contact surface patch for the Lagrangian
SDF contact constraints.  The active-patch selection is a broad-phase step; the
FEM solve, RP-MPC kinematics, contact gap/Jacobian, and reaction diagnostics
remain internal to SFC.
"""

from __future__ import annotations

import argparse
import csv
import sys
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

from validation.run_flexible_gear_explicit_sdf_comparison import DEFAULT_SOURCE, GearInputModel, GearMesh, parse_gear_input  # noqa: E402
from validation.run_flexible_gear_implicit_lagrangian_sdf_comparison import (  # noqa: E402
    ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
    CroppedGearPair,
    CroppedGearPatch,
    _face_centroids,
    _orient_faces_toward,
    _write_abaqus_alignment_deck,
    _write_csv,
    run_abaqus_alignment,
    solve_sfc_cropped_pair_hard_contact,
)

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "flexible_gear_full_lagrangian_sdf"


def _selected_surface_entries_for_full_mesh(
    mesh: GearMesh,
    entries: tuple[tuple[int, str], ...],
    selected_surface_ids: np.ndarray,
) -> tuple[tuple[int, str], ...]:
    """Map selected original Abaqus surface entries to emitted full-mesh labels."""

    out: list[tuple[int, str]] = []
    for sid in np.asarray(selected_surface_ids, dtype=np.int64).reshape(-1):
        element_label, side = entries[int(sid)]
        new_element_label = int(mesh.element_label_to_index[int(element_label)]) + 1
        out.append((new_element_label, str(side).upper()))
    return tuple(out)


def _hub_indices(mesh: GearMesh, labels: tuple[int, ...]) -> np.ndarray:
    ids = [mesh.label_to_index[int(label)] for label in labels if int(label) in mesh.label_to_index]
    if not ids:
        raise ValueError("hub node set did not map to this gear mesh")
    return np.asarray(sorted(set(ids)), dtype=np.int64)


def _nearest_active_surface_ids(model: GearInputModel, *, active_faces_per_body: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    centroids1 = _face_centroids(model.gear1.nodes, model.gear1_contact_faces)
    centroids2 = _face_centroids(model.gear2.nodes, model.gear2_contact_faces)
    tree2 = cKDTree(centroids2)
    distances, ids2 = tree2.query(centroids1, k=1)
    seed1 = int(np.argmin(distances))
    seed2 = int(ids2[seed1])
    n = max(1, int(active_faces_per_body))
    tree1 = cKDTree(centroids1)
    ids1 = np.asarray(tree1.query(centroids1[seed1], k=min(n, centroids1.shape[0]))[1], dtype=np.int64).reshape(-1)
    ids2_active = np.asarray(tree2.query(centroids2[seed2], k=min(n, centroids2.shape[0]))[1], dtype=np.int64).reshape(-1)
    c1 = np.mean(centroids1[ids1], axis=0)
    c2 = np.mean(centroids2[ids2_active], axis=0)
    drive = c2 - c1
    drive /= max(float(np.linalg.norm(drive)), 1.0e-30)
    return ids1, ids2_active, float(distances[seed1]), drive


def build_full_active_pair(model: GearInputModel, *, active_faces_per_body: int) -> CroppedGearPair:
    """Return a complete two-gear volume model with active contact surfaces."""

    ids1, ids2, initial_gap, drive = _nearest_active_surface_ids(model, active_faces_per_body=active_faces_per_body)
    gear1_faces = _orient_faces_toward(model.gear1.nodes, model.gear1_contact_faces[ids1], drive)
    gear2_faces = _orient_faces_toward(model.gear2.nodes, model.gear2_contact_faces[ids2], -drive)
    gear1 = CroppedGearPatch(
        name="gear1",
        nodes=model.gear1.nodes,
        elements=model.gear1.elements,
        contact_faces=gear1_faces,
        support_nodes=_hub_indices(model.gear1, model.gear1_hub_labels),
        surface_entries=_selected_surface_entries_for_full_mesh(model.gear1, model.gear1_surface_entries, ids1),
        element_labels=np.arange(1, model.gear1.elements.shape[0] + 1, dtype=np.int64),
        rp=model.rp1,
    )
    gear2 = CroppedGearPatch(
        name="gear2",
        nodes=model.gear2.nodes,
        elements=model.gear2.elements,
        contact_faces=gear2_faces,
        support_nodes=_hub_indices(model.gear2, model.gear2_hub_labels),
        surface_entries=_selected_surface_entries_for_full_mesh(model.gear2, model.gear2_surface_entries, ids2),
        element_labels=np.arange(1, model.gear2.elements.shape[0] + 1, dtype=np.int64),
        rp=model.rp2,
    )
    return CroppedGearPair(gear1, gear2, drive, initial_gap)


def write_full_summary(path: Path, summary: Row, history_path: Path, *, abaqus_row: Row | None = None) -> None:
    lines = [
        "# Full Gear Lagrangian-SDF Alignment",
        "",
        "This run keeps the full gear volume meshes and uses an active contact surface patch.",
        "",
        f"- nodes/elements: {summary['nodes']} / {summary['elements']}",
        f"- active contact faces: {summary['gear1_contact_faces']} / {summary['gear2_contact_faces']}",
        f"- support nodes: {summary['gear1_support_nodes']} / {summary['gear2_support_nodes']}",
        f"- linear solver: {summary.get('linear_solver', 'sparse')}",
        f"- RP reaction definition: {summary.get('rp_reaction_definition', '')}",
        f"- SFC wall time: {float(summary['sfc_wall_seconds']):.6f} s",
        f"- timing internal+tangent: {float(summary.get('timing_internal_tangent_seconds', 0.0)):.6f} s",
        f"- timing effective system: {float(summary.get('timing_effective_system_seconds', 0.0)):.6f} s",
        f"- timing hard-contact solve: {float(summary.get('timing_hard_contact_solve_seconds', 0.0)):.6f} s",
        f"- timing contact linearization: {float(summary.get('timing_contact_linearization_seconds', 0.0)):.6f} s",
        f"- final max displacement norm: {float(summary.get('final_max_displacement_norm', 0.0)):.6e}",
        f"- final p95 von Mises: {float(summary.get('final_p95_von_mises', 0.0)):.6e}",
        f"- final p95 equivalent elastic strain: {float(summary.get('final_p95_equivalent_elastic_strain', 0.0)):.6e}",
        f"- final RP reaction norm: {float(summary.get('final_rp_force_norm', 0.0)):.6e}",
        f"- final min gap: {float(summary.get('final_min_gap', 0.0)):.6e}",
        f"- history CSV: `{history_path.name}`",
    ]
    if abaqus_row is not None:
        lines.extend(
            [
                "",
                "## Abaqus Native-Contact Alignment",
                "",
                f"- Abaqus analysis wall time: {float(abaqus_row.get('abaqus_analysis_wall_seconds', 0.0)):.6f} s",
                f"- Abaqus export wall time: {float(abaqus_row.get('abaqus_export_wall_seconds', 0.0)):.6f} s",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full_gear(
    *,
    source: Path,
    out_dir: Path,
    active_faces_per_body: int,
    duration: float,
    dt: float,
    target_overclosure: float,
    rotation_rate_z: float,
    pressure_stiffness: float,
    hard_max_iterations: int,
    run_abaqus: bool,
    abaqus_command: str | None = None,
) -> tuple[list[Row], Row]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = parse_gear_input(source)
    pair = build_full_active_pair(model, active_faces_per_body=active_faces_per_body)
    history, summary = solve_sfc_cropped_pair_hard_contact(
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=pressure_stiffness,
        duration=duration,
        dt=dt,
        target_overclosure=target_overclosure,
        rotation_rate_z=rotation_rate_z,
        max_iterations=hard_max_iterations,
        hard_enforcement="abaqus_standard_penalty",
        constraint_averaging="slave_face",
        hht_alpha=ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
        linear_solver="sparse",
    )
    if history:
        summary["final_max_displacement_norm"] = float(history[-1].get("max_displacement_norm", 0.0))
        summary["final_p95_von_mises"] = float(history[-1].get("p95_von_mises", 0.0))
        summary["final_p95_equivalent_elastic_strain"] = float(history[-1].get("p95_equivalent_elastic_strain", 0.0))
    history_path = out_dir / "sfc_full_gear_lagrangian_sdf_history.csv"
    _write_csv(history_path, history)
    deck_path = out_dir / "abaqus_full_gear_alignment.inp"
    _write_abaqus_alignment_deck(
        deck_path,
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=pressure_stiffness,
        duration=duration,
        dt=dt,
        target_overclosure=target_overclosure,
        rotation_rate_z=rotation_rate_z,
        contact_mode="hard",
    )
    abaqus_row: Row | None = None
    if run_abaqus:
        abaqus_metrics = run_abaqus_alignment(
            deck_path,
            out_dir,
            pair=pair,
            young=model.young,
            poisson=model.poisson,
            abaqus_command=abaqus_command,
        )
        abaqus_row = dict(abaqus_metrics)
        summary.update(abaqus_row)
    write_full_summary(out_dir / "full_gear_lagrangian_sdf_summary.md", summary, history_path, abaqus_row=abaqus_row)
    _write_csv(out_dir / "full_gear_lagrangian_sdf_summary.csv", [summary])
    return history, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--active-faces-per-body", type=int, default=16)
    parser.add_argument("--duration", type=float, default=2.5e-4)
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--overclosure", type=float, default=1.0e-5)
    parser.add_argument("--rotation-rate-z", type=float, default=0.0)
    parser.add_argument("--pressure-stiffness", type=float, default=5.0e9)
    parser.add_argument("--hard-max-iterations", type=int, default=4)
    parser.add_argument("--run-abaqus", action="store_true")
    parser.add_argument("--abaqus-command", type=str, default=None)
    args = parser.parse_args(argv)
    history, summary = run_full_gear(
        source=args.source,
        out_dir=args.out_dir,
        active_faces_per_body=int(args.active_faces_per_body),
        duration=float(args.duration),
        dt=float(args.dt),
        target_overclosure=float(args.overclosure),
        rotation_rate_z=float(args.rotation_rate_z),
        pressure_stiffness=float(args.pressure_stiffness),
        hard_max_iterations=int(args.hard_max_iterations),
        run_abaqus=bool(args.run_abaqus),
        abaqus_command=args.abaqus_command,
    )
    print((args.out_dir / "full_gear_lagrangian_sdf_summary.md").read_text(encoding="utf-8"))
    return 0 if history and summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
