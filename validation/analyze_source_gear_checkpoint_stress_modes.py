"""Diagnose source-gear stress/strain semantics from an accepted SFC checkpoint.

This script does not change the solve.  It reloads an accepted reduced state
from ``solve_sfc_source_drive_pair`` and recomputes stress/strain metrics under
several kinematic output conventions, then compares those metrics with an
external Abaqus VTK manifest at the same time.

The purpose is to separate three effects:

1. raw small-strain output on the linearized RP-MPC displacement,
2. the current SFC corotated elastic-residual output,
3. finite-deformation StVK output on the finite-RP visual configuration.

Abaqus data are used only as an external validation reference.  No Abaqus
assembled matrices, ODB files, or spreadsheet exports are read.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.fem.calculix_aligned import InternalResponse, MechanicsModel, MechanicsState, stvk_internal_response  # noqa: E402
from sfc.fem.rp_mpc import RigidHubMPC, build_rigid_hub_reduced_assembly  # noqa: E402
from validation.run_flexible_gear_explicit_sdf_comparison import DEFAULT_SOURCE, parse_gear_input  # noqa: E402
from validation.run_flexible_gear_full_lagrangian_sdf_comparison import build_full_active_pair  # noqa: E402
from validation.run_flexible_gear_implicit_lagrangian_sdf_comparison import (  # noqa: E402
    _equivalent_elastic_strain_from_mises,
    _lame_parameters_local,
    _load_source_drive_checkpoint,
    _node_averaged_internal_metric_row,
    _source_drive_corotated_positions_and_elastic_displacement,
    _von_mises_local,
    _write_csv,
)

Row = dict[str, Any]


DEFAULT_CHECKPOINT = ROOT / "results" / "source_gear_penalty_contact_history_only_0030" / "source_checkpoint.npz"
DEFAULT_ABAQUS_MANIFEST = (
    ROOT
    / "results"
    / "source_gear_abaqus_penalty_full_stride2_match_step_0030"
    / "abaqus_vtk"
    / "abaqus_manifest.csv"
)


def _read_csv_rows(path: Path) -> list[Row]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_float_column(rows: list[Row], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0) or 0.0) for row in rows], dtype=float)


def _interpolate_manifest_value(rows: list[Row], time_value: float, key: str) -> float:
    if not rows:
        raise ValueError("Abaqus manifest has no rows")
    times = _as_float_column(rows, "time")
    values = _as_float_column(rows, key)
    return float(np.interp(float(time_value), times, values))


def _relative_error(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1.0e-12)


def _small_strain_internal_from_displacement(model: MechanicsModel, displacement: np.ndarray) -> InternalResponse:
    """Return small-strain linear-elastic stress metrics without assembling force."""

    disp = np.asarray(displacement, dtype=float)
    elements = np.asarray(model.elements, dtype=np.int64)
    element_u = disp[elements]
    grad_u = np.einsum("eai,eaj->eij", element_u, model.shape_grads, optimize=True)
    strain = 0.5 * (grad_u + np.swapaxes(grad_u, 1, 2))
    lam, mu = _lame_parameters_local(model.E, model.nu)
    identity = np.eye(3, dtype=float)
    trace = np.trace(strain, axis1=1, axis2=2)
    stress = lam * trace[:, None, None] * identity + 2.0 * mu * strain
    von_mises = _von_mises_local(stress)
    zeros_force = np.zeros_like(np.asarray(model.X, dtype=float))
    empty = csr_matrix((model.n_dofs, model.n_dofs), dtype=float)
    return InternalResponse(zeros_force, strain, stress, von_mises, 0.0, empty, empty)


def _metric_row(
    *,
    mode: str,
    model: MechanicsModel,
    state: MechanicsState,
    internal: InternalResponse,
    time_value: float,
) -> Row:
    disp = np.asarray(state.x, dtype=float) - np.asarray(model.X, dtype=float)
    strain_norm = np.linalg.norm(internal.strain, axis=(1, 2)) if internal.strain.size else np.empty(0, dtype=float)
    equivalent = _equivalent_elastic_strain_from_mises(internal.von_mises, young=model.E, poisson=model.nu)
    row: Row = {
        "mode": str(mode),
        "time": float(time_value),
        "max_displacement_norm": float(np.max(np.linalg.norm(disp, axis=1))) if disp.size else 0.0,
        "p95_von_mises": float(np.percentile(internal.von_mises, 95.0)) if internal.von_mises.size else 0.0,
        "max_von_mises": float(np.max(internal.von_mises)) if internal.von_mises.size else 0.0,
        "p95_strain_norm": float(np.percentile(strain_norm, 95.0)) if strain_norm.size else 0.0,
        "max_strain_norm": float(np.max(strain_norm)) if strain_norm.size else 0.0,
        "p95_equivalent_elastic_strain": float(np.percentile(equivalent, 95.0)) if equivalent.size else 0.0,
        "max_equivalent_elastic_strain": float(np.max(equivalent)) if equivalent.size else 0.0,
    }
    row.update(_node_averaged_internal_metric_row(model, internal))
    return row


def analyze_checkpoint_stress_modes(
    *,
    source: Path,
    checkpoint_path: Path,
    abaqus_manifest: Path,
    out_dir: Path,
    active_faces_per_body: int,
    active_patch_radius_factor: float,
    duration: float | None,
) -> list[Row]:
    """Write stress-mode comparison CSV/MD files for one accepted checkpoint."""

    out_dir.mkdir(parents=True, exist_ok=True)
    source_model = parse_gear_input(source)
    pair = build_full_active_pair(
        source_model,
        active_faces_per_body=int(active_faces_per_body),
        active_patch_radius_factor=float(active_patch_radius_factor),
    )
    n1 = pair.gear1.nodes.shape[0]
    X = np.vstack([pair.gear1.nodes, pair.gear2.nodes])
    elements = np.vstack([pair.gear1.elements, pair.gear2.elements + n1])
    model = MechanicsModel.from_tet4_mesh(
        X,
        elements,
        E=source_model.young,
        nu=source_model.poisson,
        density=source_model.density,
        mass_kind="consistent",
    )
    hub1 = RigidHubMPC(pair.gear1.support_nodes, X, pair.gear1.rp)
    hub2 = RigidHubMPC(pair.gear2.support_nodes + n1, X, pair.gear2.rp)
    assembly = build_rigid_hub_reduced_assembly(X, [hub1, hub2], include_free_nodes=True)
    checkpoint = _load_source_drive_checkpoint(checkpoint_path)
    q = np.asarray(checkpoint["q"], dtype=float)
    v = np.asarray(checkpoint["v"], dtype=float)
    a = np.asarray(checkpoint["a"], dtype=float)
    if duration is None:
        time_value = float(checkpoint["step"]) * float(source_model.dynamic_initial_dt)
    else:
        time_value = float(duration)
    state = MechanicsState(
        model.X + assembly.expand_displacements(q),
        assembly.expand_displacements(v),
        assembly.expand_displacements(a),
        time=float(time_value),
    )
    body_node_slices = (slice(0, n1), slice(n1, X.shape[0]))
    body_reference_points = (np.asarray(pair.gear1.rp, dtype=float), np.asarray(pair.gear2.rp, dtype=float))
    theta = (
        float(q[assembly.hub_slice(0).start + 5]),
        float(q[assembly.hub_slice(1).start + 5]),
    )
    x_visual, u_elastic = _source_drive_corotated_positions_and_elastic_displacement(
        model=model,
        x_raw=state.x,
        body_node_slices=body_node_slices,
        body_reference_points=body_reference_points,
        body_rotation_z=theta,
    )
    visual_state = MechanicsState(x_visual, state.v.copy(), state.a.copy(), time=float(time_value))
    modes: list[tuple[str, MechanicsState, InternalResponse]] = [
        (
            "raw_small_strain_linearized_mpc",
            state,
            _small_strain_internal_from_displacement(model, state.x - model.X),
        ),
        (
            "linear_corotated_elastic_residual",
            visual_state,
            _small_strain_internal_from_displacement(model, u_elastic),
        ),
        (
            "finite_stvk_visual_configuration",
            visual_state,
            stvk_internal_response(model, visual_state.x, assemble_tangent=False),
        ),
    ]
    rows = [_metric_row(mode=name, model=model, state=mode_state, internal=internal, time_value=time_value) for name, mode_state, internal in modes]
    abaqus_rows = _read_csv_rows(abaqus_manifest)
    reference_map = {
        "max_displacement_norm": "max_displacement_magnitude",
        "p95_von_mises_nodeavg": "p95_von_mises_nodeavg",
        "p95_equivalent_elastic_strain_nodeavg": "p95_equivalent_elastic_strain_nodeavg",
        "max_von_mises_nodeavg": "max_von_mises_nodeavg",
        "max_equivalent_elastic_strain_nodeavg": "max_equivalent_elastic_strain_nodeavg",
    }
    for row in rows:
        for sfc_key, abaqus_key in reference_map.items():
            reference = _interpolate_manifest_value(abaqus_rows, time_value, abaqus_key)
            value = float(row.get(sfc_key, 0.0) or 0.0)
            row[f"abaqus_{abaqus_key}"] = reference
            row[f"{sfc_key}_rel_error"] = _relative_error(value, reference)
        row["checkpoint_step"] = int(checkpoint["step"])
        row["rp1_rotation_z_rad"] = float(theta[0])
        row["rp2_rotation_z_rad"] = float(theta[1])
        row["source"] = str(source)
        row["checkpoint"] = str(checkpoint_path)
        row["abaqus_manifest"] = str(abaqus_manifest)
    csv_path = out_dir / "source_gear_checkpoint_stress_modes.csv"
    _write_csv(csv_path, rows)
    best = min(rows, key=lambda item: float(item.get("p95_von_mises_nodeavg_rel_error", np.inf)))
    md_lines = [
        "# Source Gear Checkpoint Stress-Mode Diagnostic",
        "",
        f"- source: `{source}`",
        f"- checkpoint: `{checkpoint_path}`",
        f"- checkpoint step/time: {int(checkpoint['step'])} / {float(time_value):.9e} s",
        f"- Abaqus manifest: `{abaqus_manifest}`",
        f"- output CSV: `{csv_path}`",
        f"- best p95 node-averaged von Mises mode: `{best['mode']}`",
        "",
        "| mode | max disp err | p95 VM nodeavg err | p95 equiv strain nodeavg err |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| "
            f"{row['mode']} | "
            f"{100.0 * float(row['max_displacement_norm_rel_error']):.3f}% | "
            f"{100.0 * float(row['p95_von_mises_nodeavg_rel_error']):.3f}% | "
            f"{100.0 * float(row['p95_equivalent_elastic_strain_nodeavg_rel_error']):.3f}% |"
        )
    md_lines.extend(
        [
            "",
            "Interpretation:",
            "- If `finite_stvk_visual_configuration` closes the stress gap without changing displacement, the main issue is output kinematics/postprocessing.",
            "- If it remains far from Abaqus, the accepted SFC trajectory itself is missing Abaqus `nlgeom=YES` finite-rotation internal-force effects, so the next fix must enter the solve residual/tangent rather than only the visualization path.",
        ]
    )
    (out_dir / "source_gear_checkpoint_stress_modes.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--abaqus-manifest", type=Path, default=DEFAULT_ABAQUS_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "source_gear_checkpoint_stress_modes")
    parser.add_argument("--active-faces-per-body", type=int, default=0)
    parser.add_argument("--active-patch-radius-factor", type=float, default=1.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional physical time for the checkpoint. Defaults to checkpoint_step * source initial dt.",
    )
    args = parser.parse_args(argv)
    rows = analyze_checkpoint_stress_modes(
        source=args.source,
        checkpoint_path=args.checkpoint,
        abaqus_manifest=args.abaqus_manifest,
        out_dir=args.out_dir,
        active_faces_per_body=int(args.active_faces_per_body),
        active_patch_radius_factor=float(args.active_patch_radius_factor),
        duration=args.duration,
    )
    md_path = args.out_dir / "source_gear_checkpoint_stress_modes.md"
    print(md_path.read_text(encoding="utf-8"))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
