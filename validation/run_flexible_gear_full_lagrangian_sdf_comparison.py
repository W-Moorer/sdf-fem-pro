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
    _resolve_abaqus_command,
    _run_command,
    _write_abaqus_alignment_deck,
    _write_csv,
    run_abaqus_alignment,
    compare_histories,
    plot_alignment_curves,
    solve_sfc_cropped_pair,
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


def _nearest_ids(tree: cKDTree, point: np.ndarray, *, k: int, count: int) -> np.ndarray:
    query_count = min(max(1, int(k)), max(1, int(count)))
    return np.asarray(tree.query(point, k=query_count)[1], dtype=np.int64).reshape(-1)


def _radius_expanded_ids(centroids: np.ndarray, seed: np.ndarray, base_ids: np.ndarray, *, radius_factor: float) -> np.ndarray:
    ids = np.asarray(base_ids, dtype=np.int64).reshape(-1)
    if float(radius_factor) <= 1.0 or ids.size == 0:
        return np.unique(ids)
    offsets = np.linalg.norm(centroids[ids] - seed, axis=1)
    base_radius = float(np.max(offsets)) if offsets.size else 0.0
    if base_radius <= 0.0 and centroids.shape[0] > 1:
        distances = np.linalg.norm(centroids - seed, axis=1)
        positive = distances[distances > 0.0]
        base_radius = float(np.min(positive)) if positive.size else 0.0
    radius = max(base_radius * float(radius_factor), base_radius)
    expanded = np.flatnonzero(np.linalg.norm(centroids - seed, axis=1) <= radius + 1.0e-12)
    return np.unique(np.concatenate([ids, expanded.astype(np.int64)]))


def _nearest_active_surface_ids(
    model: GearInputModel,
    *,
    active_faces_per_body: int,
    active_patch_radius_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    centroids1 = _face_centroids(model.gear1.nodes, model.gear1_contact_faces)
    centroids2 = _face_centroids(model.gear2.nodes, model.gear2_contact_faces)
    tree2 = cKDTree(centroids2)
    distances, ids2 = tree2.query(centroids1, k=1)
    seed1 = int(np.argmin(distances))
    seed2 = int(ids2[seed1])
    n = max(1, int(active_faces_per_body))
    tree1 = cKDTree(centroids1)
    ids1 = _nearest_ids(tree1, centroids1[seed1], k=n, count=centroids1.shape[0])
    ids2_active = _nearest_ids(tree2, centroids2[seed2], k=n, count=centroids2.shape[0])
    ids1 = _radius_expanded_ids(centroids1, centroids1[seed1], ids1, radius_factor=active_patch_radius_factor)
    ids2_active = _radius_expanded_ids(centroids2, centroids2[seed2], ids2_active, radius_factor=active_patch_radius_factor)
    if float(active_patch_radius_factor) > 1.0:
        opposite_count = min(4, centroids2.shape[0])
        mapped2 = np.asarray(tree2.query(centroids1[ids1], k=opposite_count)[1], dtype=np.int64).reshape(-1)
        ids2_active = np.unique(np.concatenate([ids2_active, mapped2]))
        opposite_count = min(4, centroids1.shape[0])
        mapped1 = np.asarray(tree1.query(centroids2[ids2_active], k=opposite_count)[1], dtype=np.int64).reshape(-1)
        ids1 = np.unique(np.concatenate([ids1, mapped1]))
    c1 = np.mean(centroids1[ids1], axis=0)
    c2 = np.mean(centroids2[ids2_active], axis=0)
    drive = c2 - c1
    drive /= max(float(np.linalg.norm(drive)), 1.0e-30)
    return ids1, ids2_active, float(distances[seed1]), drive


def build_full_active_pair(
    model: GearInputModel,
    *,
    active_faces_per_body: int,
    active_patch_radius_factor: float = 1.0,
) -> CroppedGearPair:
    """Return a complete two-gear volume model with active contact surfaces."""

    ids1, ids2, initial_gap, drive = _nearest_active_surface_ids(
        model,
        active_faces_per_body=active_faces_per_body,
        active_patch_radius_factor=active_patch_radius_factor,
    )
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
        f"- active patch radius factor: {float(summary.get('active_patch_radius_factor', 1.0)):.3f}",
        f"- support nodes: {summary['gear1_support_nodes']} / {summary['gear2_support_nodes']}",
        f"- linear solver: {summary.get('linear_solver', 'sparse')}",
        f"- contact mode: {summary.get('contact_mode', '')}",
        f"- penalty solver: {summary.get('penalty_solver', '')}",
        f"- material linearization: {summary.get('material_linearization', '')}",
        f"- RP reaction definition: {summary.get('rp_reaction_definition', '')}",
        f"- SFC wall time: {float(summary['sfc_wall_seconds']):.6f} s",
        f"- timing internal+tangent: {float(summary.get('timing_internal_tangent_seconds', 0.0)):.6f} s",
        f"- timing effective system: {float(summary.get('timing_effective_system_seconds', 0.0)):.6f} s",
        f"- timing hard-contact solve: {float(summary.get('timing_hard_contact_solve_seconds', 0.0)):.6f} s",
        f"- timing contact linearization: {float(summary.get('timing_contact_linearization_seconds', 0.0)):.6f} s",
        f"- timing penalty base tangent: {float(summary.get('timing_base_tangent_seconds', 0.0)):.6f} s",
        f"- timing penalty residual: {float(summary.get('timing_penalty_residual_seconds', 0.0)):.6f} s",
        f"- timing penalty low-rank solve: {float(summary.get('timing_penalty_low_rank_seconds', 0.0)):.6f} s",
        f"- final max displacement norm: {float(summary.get('final_max_displacement_norm', 0.0)):.6e}",
        f"- final p95 von Mises: {float(summary.get('final_p95_von_mises', 0.0)):.6e}",
        f"- final p95 equivalent elastic strain: {float(summary.get('final_p95_equivalent_elastic_strain', 0.0)):.6e}",
        f"- final RP reaction norm: {float(summary.get('final_rp_force_norm', 0.0)):.6e}",
        f"- final min gap: {float(summary.get('final_min_gap', 0.0)):.6e}",
        f"- history CSV: `{history_path.name}`",
    ]
    if summary.get("sfc_vtk_pvd"):
        lines.extend(
            [
                f"- SFC VTK PVD: `{summary.get('sfc_vtk_pvd')}`",
                f"- SFC VTK frame count: {int(summary.get('sfc_vtk_frame_count', 0))}",
            ]
        )
    if abaqus_row is not None:
        lines.extend(
            [
                "",
                "## Abaqus Native-Contact Alignment",
                "",
                f"- Abaqus analysis wall time: {float(abaqus_row.get('abaqus_analysis_wall_seconds', 0.0)):.6f} s",
                f"- Abaqus export wall time: {float(abaqus_row.get('abaqus_export_wall_seconds', 0.0)):.6f} s",
                f"- alignment errors CSV: `{Path(str(summary.get('alignment_errors', ''))).name}`",
                f"- alignment figure: `{Path(str(summary.get('alignment_figure', ''))).name}`",
            ]
        )
    if summary.get("abaqus_vtk_pvd"):
        lines.extend(
            [
                "",
                "## VTK Animation",
                "",
                f"- Abaqus VTK PVD: `{summary.get('abaqus_vtk_pvd')}`",
                f"- Abaqus VTK frame count: {int(summary.get('abaqus_vtk_frame_count', 0))}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_abaqus_vtk_frames(
    *,
    out_dir: Path,
    abaqus_command: str | None,
    frame_stride: int,
    include_tensors: bool,
) -> Row:
    """Export the existing full-gear Abaqus ODB to a strided VTK/PVD series."""

    run_dir = out_dir / "abaqus_run"
    odb = run_dir / "cropped_gear_alignment.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    vtk_dir = out_dir / "abaqus_vtk"
    command = _resolve_abaqus_command(abaqus_command)
    script = ROOT / "validation" / "abaqus_odb_to_vtk.py"
    wall = _run_command(
        [
            command,
            "python",
            str(script.resolve()),
            "--odb",
            str(odb.resolve()),
            "--out-dir",
            str(vtk_dir.resolve()),
            "--stem",
            "abaqus",
            "--frame-stride",
            str(max(1, int(frame_stride))),
            "--include-tensors" if include_tensors else "--scalars-only",
        ],
        cwd=run_dir,
        log_path=out_dir / "abaqus_vtk_export_stdout.log",
    )
    manifest = vtk_dir / "abaqus_manifest.csv"
    frame_count = 0
    if manifest.exists():
        with manifest.open("r", newline="", encoding="ascii") as handle:
            frame_count = max(0, sum(1 for _ in handle) - 1)
    return {
        "abaqus_vtk_export_wall_seconds": float(wall),
        "abaqus_vtk_pvd": str(vtk_dir / "abaqus.pvd"),
        "abaqus_vtk_manifest": str(manifest),
        "abaqus_vtk_frame_count": int(frame_count),
        "abaqus_vtk_frame_stride": int(frame_stride),
        "abaqus_vtk_include_tensors": bool(include_tensors),
    }


def run_full_gear(
    *,
    source: Path,
    out_dir: Path,
    active_faces_per_body: int,
    active_patch_radius_factor: float,
    duration: float,
    dt: float,
    target_overclosure: float,
    rotation_rate_z: float,
    pressure_stiffness: float,
    hard_max_iterations: int,
    contact_mode: str,
    run_abaqus: bool,
    abaqus_command: str | None = None,
    write_sfc_vtk: bool = False,
    sfc_vtk_dir: Path | None = None,
    vtk_frame_stride: int = 1,
    vtk_include_tensors: bool = True,
    export_abaqus_vtk: bool = False,
) -> tuple[list[Row], Row]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = parse_gear_input(source)
    pair = build_full_active_pair(
        model,
        active_faces_per_body=active_faces_per_body,
        active_patch_radius_factor=active_patch_radius_factor,
    )
    mode = str(contact_mode).lower()
    if mode not in {"penalty", "hard"}:
        raise ValueError("contact_mode must be 'penalty' or 'hard'")
    if mode == "hard":
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
    else:
        history, summary = solve_sfc_cropped_pair(
            pair,
            young=model.young,
            poisson=model.poisson,
            density=model.density,
            pressure_stiffness=pressure_stiffness,
            duration=duration,
            dt=dt,
            target_overclosure=target_overclosure,
            rotation_rate_z=rotation_rate_z,
            hht_alpha=ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
            penalty_solver="modified_newton",
            material_linearization="reference_linear",
            vtk_out_dir=(sfc_vtk_dir if sfc_vtk_dir is not None else out_dir / "sfc_vtk") if write_sfc_vtk else None,
            vtk_frame_stride=max(1, int(vtk_frame_stride)),
            vtk_stem="sfc",
            vtk_include_tensors=bool(vtk_include_tensors),
        )
    if history:
        summary["final_max_displacement_norm"] = float(history[-1].get("max_displacement_norm", 0.0))
        summary["final_p95_von_mises"] = float(history[-1].get("p95_von_mises", 0.0))
        summary["final_p95_equivalent_elastic_strain"] = float(history[-1].get("p95_equivalent_elastic_strain", 0.0))
    summary["active_patch_radius_factor"] = float(active_patch_radius_factor)
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
        contact_mode=mode,
    )
    abaqus_row: Row | None = None
    error_rows: list[Row] | None = None
    figure_path: Path | None = None
    if run_abaqus:
        abaqus_metrics, abaqus_row = run_abaqus_alignment(
            deck_path,
            out_dir,
            pair=pair,
            young=model.young,
            poisson=model.poisson,
            abaqus_command=abaqus_command,
        )
        summary.update(abaqus_row)
        _write_csv(out_dir / "abaqus_runtime.csv", [abaqus_row])
        error_rows = compare_histories(history_path, abaqus_metrics, out_dir / "sfc_vs_abaqus_alignment_errors.csv")
        figure_path = out_dir / "sfc_vs_abaqus_alignment_curves.png"
        plot_alignment_curves(history_path, abaqus_metrics, error_rows, figure_path)
        summary["alignment_errors"] = str(out_dir / "sfc_vs_abaqus_alignment_errors.csv")
        summary["alignment_figure"] = str(figure_path)
    if export_abaqus_vtk:
        summary.update(
            export_abaqus_vtk_frames(
                out_dir=out_dir,
                abaqus_command=abaqus_command,
                frame_stride=max(1, int(vtk_frame_stride)),
                include_tensors=bool(vtk_include_tensors),
            )
        )
    write_full_summary(out_dir / "full_gear_lagrangian_sdf_summary.md", summary, history_path, abaqus_row=abaqus_row)
    _write_csv(out_dir / "full_gear_lagrangian_sdf_summary.csv", [summary])
    return history, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--active-faces-per-body", type=int, default=16)
    parser.add_argument("--active-patch-radius-factor", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=2.5e-4)
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--overclosure", type=float, default=1.0e-5)
    parser.add_argument("--rotation-rate-z", type=float, default=0.0)
    parser.add_argument("--pressure-stiffness", type=float, default=5.0e9)
    parser.add_argument("--contact-mode", choices=("penalty", "hard"), default="hard")
    parser.add_argument("--hard-max-iterations", type=int, default=4)
    parser.add_argument("--run-abaqus", action="store_true")
    parser.add_argument("--abaqus-command", type=str, default=None)
    parser.add_argument("--write-sfc-vtk", action="store_true")
    parser.add_argument("--sfc-vtk-dir", type=Path, default=None)
    parser.add_argument("--vtk-frame-stride", type=int, default=1)
    parser.add_argument("--vtk-scalars-only", action="store_true")
    parser.add_argument("--export-abaqus-vtk", action="store_true")
    args = parser.parse_args(argv)
    history, summary = run_full_gear(
        source=args.source,
        out_dir=args.out_dir,
        active_faces_per_body=int(args.active_faces_per_body),
        active_patch_radius_factor=float(args.active_patch_radius_factor),
        duration=float(args.duration),
        dt=float(args.dt),
        target_overclosure=float(args.overclosure),
        rotation_rate_z=float(args.rotation_rate_z),
        pressure_stiffness=float(args.pressure_stiffness),
        hard_max_iterations=int(args.hard_max_iterations),
        contact_mode=str(args.contact_mode),
        run_abaqus=bool(args.run_abaqus),
        abaqus_command=args.abaqus_command,
        write_sfc_vtk=bool(args.write_sfc_vtk),
        sfc_vtk_dir=args.sfc_vtk_dir,
        vtk_frame_stride=int(args.vtk_frame_stride),
        vtk_include_tensors=not bool(args.vtk_scalars_only),
        export_abaqus_vtk=bool(args.export_abaqus_vtk),
    )
    print((args.out_dir / "full_gear_lagrangian_sdf_summary.md").read_text(encoding="utf-8"))
    return 0 if history and summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
