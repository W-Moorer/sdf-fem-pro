"""Validation suite for the clean-room CalculiX-aligned mechanics backend.

This runner focuses on backend calculations rather than contact detection.  It
checks mass assembly, StVK stress/energy, tangent consistency, HHT/Newmark
dynamics, and no-contact Newton equilibrium.  Optional CalculiX comparison is
kept as external validation evidence only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.fem.calculix_aligned import (  # noqa: E402
    ContactSample,
    MechanicsModel,
    hht_step,
    initial_state,
    stvk_internal_response,
)
from validation.run_geometric_nonlinear_acceptance import (  # noqa: E402
    _cantilever_model,
    _fixed_dofs,
    _force_vector,
    run_calculix_static_cantilever,
)
from validation.run_calculix_drop_impact_comparison import _structured_block_mesh  # noqa: E402

Row = dict[str, Any]


class EmptyContactGeometry:
    """No-contact geometry for backend-only dynamics."""

    def samples(self, x_current: np.ndarray) -> Iterable[ContactSample]:
        return []


def run_validation(out_dir: Path, *, quick: bool = False, run_calculix: bool = False) -> dict[str, Path]:
    """Run backend validation and write CSV/Markdown artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    resolutions = [1] if quick else [1, 2, 3]
    mass_rows = mass_matrix_checks(resolutions)
    patch_rows = stvk_patch_checks(resolutions)
    tangent_rows = tangent_finite_difference_checks()
    freefall_rows = freefall_dynamics_checks(quick=quick)
    cantilever_rows, newton_rows, command_rows = cantilever_newton_checks(resolutions, out_dir, run_calculix=run_calculix)
    claim_rows = claim_rows_from(mass_rows, patch_rows, tangent_rows, freefall_rows, cantilever_rows, newton_rows)
    outputs = {
        "mass": out_dir / "backend_mass_matrix.csv",
        "patch": out_dir / "backend_stvk_patch.csv",
        "tangent": out_dir / "backend_tangent_fd.csv",
        "freefall": out_dir / "backend_freefall_dynamics.csv",
        "cantilever": out_dir / "backend_cantilever_newton.csv",
        "newton": out_dir / "backend_newton_iterations.csv",
        "commands": out_dir / "backend_external_commands.csv",
        "claims": out_dir / "backend_alignment_claims.csv",
        "summary": out_dir / "backend_alignment_summary.md",
    }
    _write_csv(outputs["mass"], mass_rows)
    _write_csv(outputs["patch"], patch_rows)
    _write_csv(outputs["tangent"], tangent_rows)
    _write_csv(outputs["freefall"], freefall_rows)
    _write_csv(outputs["cantilever"], cantilever_rows)
    _write_csv(outputs["newton"], newton_rows)
    _write_csv(outputs["commands"], command_rows)
    _write_csv(outputs["claims"], claim_rows)
    _write_markdown(outputs["summary"], claim_rows, mass_rows, patch_rows, tangent_rows, freefall_rows, cantilever_rows, quick=quick)
    return outputs


def mass_matrix_checks(resolutions: list[int]) -> list[Row]:
    rows: list[Row] = []
    for resolution in resolutions:
        X, elements = _structured_block_mesh(resolution, size=(1.0, 0.8, 0.6), bottom_z=0.0)
        model = MechanicsModel.from_tet4_mesh(X, elements, E=1000.0, nu=0.3, density=2.5)
        total_mass = float(model.density * np.sum(model.volumes))
        symmetry_error = float(np.linalg.norm((model.mass_matrix - model.mass_matrix.T).data)) if model.mass_matrix.nnz else 0.0
        component_errors = []
        for component in range(3):
            ones = np.zeros(model.n_dofs)
            ones[component::3] = 1.0
            component_errors.append(abs(float(ones @ (model.mass_matrix @ ones)) - total_mass))
        rows.append(
            {
                "case": "calculix_c3d4_one_point_mass",
                "resolution": resolution,
                "nodes": model.n_nodes,
                "elements": model.elements.shape[0],
                "total_mass_expected": total_mass,
                "mass_component_max_abs_error": max(component_errors),
                "mass_symmetry_error": symmetry_error,
                "status": "passed" if max(component_errors) <= 1.0e-12 and symmetry_error <= 1.0e-12 else "failed",
            }
        )
    return rows


def stvk_patch_checks(resolutions: list[int]) -> list[Row]:
    E = 1000.0
    nu = 0.3
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    F = np.asarray([[1.10, 0.06, 0.02], [0.03, 0.95, -0.01], [0.00, 0.04, 1.07]], dtype=float)
    green = 0.5 * (F.T @ F - np.eye(3))
    second_piola = lam * float(np.trace(green)) * np.eye(3) + 2.0 * mu * green
    first_piola = F @ second_piola
    cauchy = (first_piola @ F.T) / float(np.linalg.det(F))
    rows: list[Row] = []
    for resolution in resolutions:
        X, elements = _structured_block_mesh(resolution, size=(1.0, 1.0, 1.0), bottom_z=0.0)
        model = MechanicsModel.from_tet4_mesh(X, elements, E=E, nu=nu, density=1.0)
        x = X @ F.T
        response = stvk_internal_response(model, x, assemble_tangent=True)
        analytic_energy = 0.5 * float(np.sum(green * second_piola)) * float(np.sum(model.volumes))
        energy_rel_error = abs(response.strain_energy - analytic_energy) / max(abs(analytic_energy), 1.0e-30)
        strain_error = float(np.max(np.linalg.norm(response.strain - green, axis=(1, 2))))
        stress_error = float(np.max(np.linalg.norm(response.stress - cauchy, axis=(1, 2))))
        rows.append(
            {
                "case": "affine_stvk_patch",
                "resolution": resolution,
                "nodes": model.n_nodes,
                "elements": model.elements.shape[0],
                "energy_rel_error": energy_rel_error,
                "max_green_strain_error": strain_error,
                "max_cauchy_stress_error": stress_error,
                "material_tangent_norm": float(np.linalg.norm(response.material_tangent.data)),
                "geometric_tangent_norm": float(np.linalg.norm(response.geometric_tangent.data)),
                "status": "passed" if energy_rel_error <= 1.0e-12 and strain_error <= 1.0e-12 and stress_error <= 1.0e-10 else "failed",
            }
        )
    return rows


def tangent_finite_difference_checks() -> list[Row]:
    X, elements = _structured_block_mesh(1, size=(1.0, 0.9, 0.7), bottom_z=0.0)
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1000.0, nu=0.3, density=1.0)
    x = X.copy()
    x[:, 0] += 0.04 * X[:, 2]
    x[:, 1] += 0.03 * X[:, 0]
    x[:, 2] += 0.02 * X[:, 1]
    direction = np.linspace(-0.4, 0.5, model.n_dofs).reshape((-1, 3))
    direction /= np.linalg.norm(direction)
    response = stvk_internal_response(model, x, assemble_tangent=True)
    h = 1.0e-6
    plus = stvk_internal_response(model, x + h * direction, assemble_tangent=False).force.reshape(-1)
    minus = stvk_internal_response(model, x - h * direction, assemble_tangent=False).force.reshape(-1)
    fd = (plus - minus) / (2.0 * h)
    analytic = response.tangent @ direction.reshape(-1)
    abs_error = float(np.linalg.norm(analytic - fd))
    rel_error = abs_error / max(float(np.linalg.norm(fd)), 1.0e-30)
    return [
        {
            "case": "stvk_tangent_finite_difference",
            "direction_norm": float(np.linalg.norm(direction)),
            "finite_difference_norm": float(np.linalg.norm(fd)),
            "tangent_action_norm": float(np.linalg.norm(analytic)),
            "abs_error": abs_error,
            "rel_error": rel_error,
            "status": "passed" if rel_error <= 1.0e-7 else "failed",
        }
    ]


def freefall_dynamics_checks(*, quick: bool) -> list[Row]:
    X, elements = _structured_block_mesh(1, size=(0.5, 0.4, 0.3), bottom_z=1.0)
    model = MechanicsModel.from_tet4_mesh(X, elements, E=1000.0, nu=0.3, density=2.0)
    contact = EmptyContactGeometry()
    gravity = 9.81
    v0 = -0.2
    dt_values = [0.004] if quick else [0.004, 0.002, 0.001]
    rows: list[Row] = []
    for dt in dt_values:
        state, previous = initial_state(
            model,
            contact,
            gravity=gravity,
            initial_velocity=(0.0, 0.0, v0),
            dt=dt,
            alpha=0.0,
        )
        z0 = float(np.mean(state.x[:, 2]))
        energy_values = []
        max_z_error = 0.0
        max_v_error = 0.0
        max_internal = 0.0
        steps = int(round(0.04 / dt))
        for step in range(steps + 1):
            t = step * dt
            z_expected = z0 + v0 * t - 0.5 * gravity * t * t
            v_expected = v0 - gravity * t
            z_actual = float(np.mean(state.x[:, 2]))
            v_actual = float(np.mean(state.v[:, 2]))
            response = stvk_internal_response(model, state.x, assemble_tangent=False)
            diagnostics_energy = 0.5 * float(state.v.reshape(-1) @ (model.mass_matrix @ state.v.reshape(-1)))
            diagnostics_energy += float(model.density * np.sum(model.volumes) * gravity * z_actual)
            energy_values.append(diagnostics_energy)
            max_z_error = max(max_z_error, abs(z_actual - z_expected))
            max_v_error = max(max_v_error, abs(v_actual - v_expected))
            max_internal = max(max_internal, response.strain_energy)
            if step == steps:
                break
            state, previous, _ = hht_step(model, state, previous, contact, dt=dt, gravity=gravity, alpha=0.0)
        energy_range = max(energy_values) - min(energy_values)
        rows.append(
            {
                "case": "no_contact_freefall",
                "dt": dt,
                "steps": steps,
                "max_z_cm_abs_error": max_z_error,
                "max_v_cm_abs_error": max_v_error,
                "max_internal_energy": max_internal,
                "mechanical_energy_range": energy_range,
                "status": "passed"
                if max_z_error <= 1.0e-11 and max_v_error <= 1.0e-11 and max_internal <= 1.0e-20 and energy_range <= 1.0e-10
                else "failed",
            }
        )
    return rows


def cantilever_newton_checks(resolutions: list[int], out_dir: Path, *, run_calculix: bool) -> tuple[list[Row], list[Row], list[Row]]:
    rows: list[Row] = []
    newton_rows: list[Row] = []
    command_rows: list[Row] = []
    for resolution in resolutions:
        model = _cantilever_model(resolution)
        sfc_row, iteration_rows = solve_backend_static_cantilever(model)
        newton_rows.extend(iteration_rows)
        if run_calculix:
            calc_row, command_row = run_calculix_static_cantilever(model, out_dir)
        else:
            calc_row = {"completed": "false", "tip_displacement_z": "", "reaction_z": "", "max_von_mises": ""}
            command_row = {"case": "backend_cantilever", "resolution": resolution, "completed": "false", "return_code": "skipped", "command": ""}
        command_rows.append(command_row)
        rows.append(_cantilever_comparison_row(sfc_row, calc_row))
    return rows, newton_rows, command_rows


def solve_backend_static_cantilever(model: Any, *, max_iterations: int = 25, tolerance: float = 1.0e-10) -> tuple[Row, list[Row]]:
    mechanics = MechanicsModel.from_tet4_mesh(model.X, model.elements, E=model.E, nu=model.nu, density=1.0)
    u = np.zeros(mechanics.n_dofs, dtype=float)
    fext = _force_vector(model)
    fixed = _fixed_dofs(model.fixed_nodes)
    free = np.setdiff1d(np.arange(mechanics.n_dofs), fixed)
    iteration_rows: list[Row] = []
    response = stvk_internal_response(mechanics, mechanics.X, assemble_tangent=True)
    residual = response.force.reshape(-1) - fext
    for iteration in range(max_iterations):
        x = mechanics.X + u.reshape((-1, 3))
        response = stvk_internal_response(mechanics, x, assemble_tangent=True)
        residual = response.force.reshape(-1) - fext
        residual_free = residual[free]
        residual_norm = float(np.linalg.norm(residual_free))
        correction = np.asarray(spsolve(response.tangent[free[:, None], free].tocsc(), -residual_free), dtype=float)
        correction_norm = float(np.linalg.norm(correction))
        u[free] += correction
        iteration_rows.append(
            {
                "case": "backend_static_cantilever",
                "resolution": model.resolution,
                "iteration": iteration + 1,
                "residual_norm": residual_norm,
                "correction_norm": correction_norm,
                "material_tangent_norm": float(np.linalg.norm(response.material_tangent.data)),
                "geometric_tangent_norm": float(np.linalg.norm(response.geometric_tangent.data)) if response.geometric_tangent.nnz else 0.0,
            }
        )
        if correction_norm <= tolerance * max(1.0, float(np.linalg.norm(u[free]))):
            break
    x_final = mechanics.X + u.reshape((-1, 3))
    response = stvk_internal_response(mechanics, x_final, assemble_tangent=True)
    residual = response.force.reshape(-1) - fext
    row = {
        "case": "backend_static_cantilever",
        "resolution": model.resolution,
        "nodes": mechanics.n_nodes,
        "elements": mechanics.elements.shape[0],
        "tip_displacement_z": float(np.mean(u.reshape((-1, 3))[model.tip_nodes, 2])),
        "fixed_reaction_z": float(np.sum(residual[3 * model.fixed_nodes + 2])),
        "strain_energy": response.strain_energy,
        "external_work_estimate": 0.5 * abs(float(model.total_tip_load_z) * float(np.mean(u.reshape((-1, 3))[model.tip_nodes, 2]))),
        "max_von_mises": float(np.max(response.von_mises)),
        "newton_iterations": len(iteration_rows),
        "final_residual_norm": float(np.linalg.norm(residual[free])),
        "status": "passed" if float(np.linalg.norm(residual[free])) <= 1.0e-8 else "failed",
    }
    return row, iteration_rows


def _cantilever_comparison_row(sfc: Row, calc: Row) -> Row:
    completed = str(calc.get("completed", "false")).lower() == "true"
    row = {
        "case": "backend_static_cantilever",
        "resolution": sfc["resolution"],
        "sfc_status": sfc["status"],
        "sfc_tip_displacement_z": sfc["tip_displacement_z"],
        "sfc_fixed_reaction_z": sfc["fixed_reaction_z"],
        "sfc_strain_energy": sfc["strain_energy"],
        "sfc_max_von_mises": sfc["max_von_mises"],
        "calculix_completed": str(completed).lower(),
        "calculix_tip_displacement_z": calc.get("tip_displacement_z", ""),
        "tip_displacement_rel_error": "",
        "calculix_reaction_z": calc.get("reaction_z", ""),
        "reaction_rel_error": "",
        "calculix_max_von_mises": calc.get("max_von_mises", ""),
        "stress_rel_error": "",
    }
    if completed:
        row["tip_displacement_rel_error"] = _relative_error(float(row["sfc_tip_displacement_z"]), float(row["calculix_tip_displacement_z"]))
        row["reaction_rel_error"] = _relative_error(float(row["sfc_fixed_reaction_z"]), float(row["calculix_reaction_z"]))
        row["stress_rel_error"] = _relative_error(float(row["sfc_max_von_mises"]), float(row["calculix_max_von_mises"]))
    return row


def claim_rows_from(
    mass_rows: list[Row],
    patch_rows: list[Row],
    tangent_rows: list[Row],
    freefall_rows: list[Row],
    cantilever_rows: list[Row],
    newton_rows: list[Row],
) -> list[Row]:
    return [
        {
            "claim": "consistent_mass_backend_check",
            "supported": str(all(row["status"] == "passed" for row in mass_rows)).lower(),
            "evidence_csv": "backend_mass_matrix.csv",
            "details": "CalculiX C3D4 one-point mass is symmetric and integrates total translational mass",
        },
        {
            "claim": "stvk_affine_patch_backend_check",
            "supported": str(all(row["status"] == "passed" for row in patch_rows)).lower(),
            "evidence_csv": "backend_stvk_patch.csv",
            "details": "Green strain, Cauchy stress, and strain energy match analytic affine StVK reference",
        },
        {
            "claim": "stvk_tangent_finite_difference_backend_check",
            "supported": str(all(row["status"] == "passed" for row in tangent_rows)).lower(),
            "evidence_csv": "backend_tangent_fd.csv",
            "details": "material plus geometric tangent action matches central finite difference",
        },
        {
            "claim": "hht_newmark_freefall_backend_check",
            "supported": str(all(row["status"] == "passed" for row in freefall_rows)).lower(),
            "evidence_csv": "backend_freefall_dynamics.csv",
            "details": "alpha=0 Newmark reproduces analytic constant-acceleration freefall without internal strain",
        },
        {
            "claim": "no_contact_newton_cantilever_backend_check",
            "supported": str(all(row["sfc_status"] == "passed" for row in cantilever_rows) and bool(newton_rows)).lower(),
            "evidence_csv": "backend_cantilever_newton.csv",
            "details": "no-contact cantilever Newton solve converges using backend StVK tangent",
        },
    ]


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    claim_rows: list[Row],
    mass_rows: list[Row],
    patch_rows: list[Row],
    tangent_rows: list[Row],
    freefall_rows: list[Row],
    cantilever_rows: list[Row],
    *,
    quick: bool,
) -> None:
    command = "python validation/run_calculix_aligned_backend_validation.py"
    if quick:
        command += " --quick"
    command += f" --skip-calculix --out-dir {path.parent.as_posix()}"
    lines = [
        "# CalculiX-Aligned Backend Validation",
        "",
        "This validation checks the clean-room mechanics backend independently of contact detection.",
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Claim Gates",
        "",
        "| Claim | Supported | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in claim_rows:
        lines.append(f"| {row['claim']} | {row['supported']} | `{row['evidence_csv']}` |")
    lines.extend(
        [
            "",
            "## Row Counts",
            "",
            f"- mass rows: `{len(mass_rows)}`",
            f"- StVK patch rows: `{len(patch_rows)}`",
            f"- tangent finite-difference rows: `{len(tangent_rows)}`",
            f"- freefall rows: `{len(freefall_rows)}`",
            f"- cantilever rows: `{len(cantilever_rows)}`",
            "",
            "## Limitations",
            "",
            "- This is a clean-room backend validation, not a source-level clone of CalculiX.",
            "- External CalculiX cantilever comparison is optional and skipped by default in CI-style runs.",
            "- Contact detection and SDF query validation are covered by separate runners.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0e-30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-calculix", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "calculix_aligned_backend_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(args.out_dir, quick=bool(args.quick), run_calculix=not bool(args.skip_calculix))
    print("CalculiX-aligned backend validation complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
