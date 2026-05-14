"""Geometric nonlinear acceptance checks for the validation solver.

This runner is intentionally validation-only.  It verifies the current
geometric nonlinear diagnostic path before using it for contact evidence:

1. finite-deformation StVK patch test against an analytic affine reference;
2. no-contact large-displacement cantilever, with optional CalculiX comparison;
3. Newton residual and iteration tables;
4. mesh-resolution and time-step convergence diagnostics;
5. block-plane contact only after no-contact gates are recorded.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_calculix_drop_impact_comparison import (  # noqa: E402
    _calculix_version,
    _wsl_path,
    calculix_available,
)
from validation.run_geometric_nonlinear_vtk import (  # noqa: E402
    _internal_response_with_tangent,
    _lumped_mass,
    _structured_block_mesh,
    _tet_reference_data,
    _von_mises,
    run_simulation,
)

Row = dict[str, Any]


@dataclass(slots=True)
class CantileverModel:
    """Shared SFC/CalculiX no-contact cantilever model."""

    resolution: int
    X: np.ndarray
    elements: np.ndarray
    fixed_nodes: np.ndarray
    tip_nodes: np.ndarray
    E: float
    nu: float
    total_tip_load_z: float


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cantilever_model(resolution: int, *, E: float = 1000.0, nu: float = 0.3, load_z: float = -0.2) -> CantileverModel:
    X, elements = _structured_block_mesh(resolution, size=(1.0, 0.2, 0.2), bottom_z=-0.1)
    X = X.copy()
    X[:, 0] += 0.5
    fixed = np.flatnonzero(np.isclose(X[:, 0], np.min(X[:, 0])))
    tip = np.flatnonzero(np.isclose(X[:, 0], np.max(X[:, 0])))
    return CantileverModel(resolution, X, elements, fixed, tip, E, nu, load_z)


def _fixed_dofs(nodes: np.ndarray) -> np.ndarray:
    return np.asarray([3 * int(node) + comp for node in nodes for comp in range(3)], dtype=np.int64)


def _force_vector(model: CantileverModel) -> np.ndarray:
    f = np.zeros(3 * model.X.shape[0], dtype=float)
    per_node = model.total_tip_load_z / len(model.tip_nodes)
    for node in model.tip_nodes:
        f[3 * int(node) + 2] += per_node
    return f


def solve_sfc_static_cantilever(
    model: CantileverModel,
    *,
    max_iterations: int = 25,
    tolerance: float = 1.0e-10,
) -> tuple[Row, list[Row], np.ndarray]:
    """Solve a no-contact nonlinear cantilever with analytic StVK tangent."""

    volumes, grads = _tet_reference_data(model.X, model.elements)
    u = np.zeros(3 * model.X.shape[0], dtype=float)
    fext = _force_vector(model)
    fixed = _fixed_dofs(model.fixed_nodes)
    all_dofs = np.arange(u.size)
    free = np.setdiff1d(all_dofs, fixed)
    iteration_rows: list[Row] = []
    residual = np.zeros_like(u)
    strain_energy = 0.0
    stresses = np.zeros((model.elements.shape[0], 3, 3), dtype=float)
    vm = np.zeros(model.elements.shape[0], dtype=float)

    for iteration in range(max_iterations):
        x = model.X + u.reshape((-1, 3))
        fint, _, stresses, vm, strain_energy, material, geometric, tangent = _internal_response_with_tangent(
            model.X,
            model.elements,
            volumes,
            grads,
            x,
            E=model.E,
            nu=model.nu,
            assemble_tangent=True,
        )
        residual = fint.reshape(-1) - fext
        residual_free = residual[free]
        residual_norm = float(np.linalg.norm(residual_free))
        Kff = tangent[free[:, None], free].tocsr()
        correction = np.asarray(spsolve(Kff, -residual_free), dtype=float)
        correction_norm = float(np.linalg.norm(correction))
        u[free] += correction
        iteration_rows.append(
            {
                "case": "no_contact_cantilever",
                "resolution": model.resolution,
                "iteration": iteration + 1,
                "residual_norm": residual_norm,
                "correction_norm": correction_norm,
                "material_tangent_norm": float(np.linalg.norm(material.data)) if material.nnz else 0.0,
                "geometric_tangent_norm": float(np.linalg.norm(geometric.data)) if geometric.nnz else 0.0,
            }
        )
        if correction_norm <= tolerance * max(1.0, float(np.linalg.norm(u[free]))):
            break

    x = model.X + u.reshape((-1, 3))
    tip_uz = float(np.mean(u.reshape((-1, 3))[model.tip_nodes, 2]))
    reaction_z = float(np.sum(residual[3 * model.fixed_nodes + 2]))
    result = {
        "case": "no_contact_cantilever",
        "resolution": model.resolution,
        "nodes": model.X.shape[0],
        "elements": model.elements.shape[0],
        "tip_displacement_z": tip_uz,
        "reaction_z": reaction_z,
        "strain_energy": float(strain_energy),
        "external_work_estimate": 0.5 * abs(model.total_tip_load_z * tip_uz),
        "max_von_mises": float(np.max(vm)),
        "mean_von_mises": float(np.mean(vm)),
        "newton_iterations": len(iteration_rows),
        "final_residual_norm": float(iteration_rows[-1]["residual_norm"]),
        "final_correction_norm": float(iteration_rows[-1]["correction_norm"]),
    }
    return result, iteration_rows, x


def stvk_patch_test(resolutions: list[int], *, E: float = 1000.0, nu: float = 0.3) -> list[Row]:
    """Run affine finite-deformation StVK patch checks."""

    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    F = np.asarray([[1.12, 0.08, 0.0], [0.02, 0.96, 0.03], [0.0, -0.01, 1.05]], dtype=float)
    green = 0.5 * (F.T @ F - np.eye(3))
    second_piola = lam * float(np.trace(green)) * np.eye(3) + 2.0 * mu * green
    first_piola = F @ second_piola
    cauchy = (first_piola @ F.T) / np.linalg.det(F)
    rows: list[Row] = []
    for resolution in resolutions:
        X, elements = _structured_block_mesh(resolution, size=(1.0, 1.0, 1.0), bottom_z=0.0)
        volumes, grads = _tet_reference_data(X, elements)
        x = X @ F.T
        _, strain, stress, vm, energy, *_ = _internal_response_with_tangent(
            X,
            elements,
            volumes,
            grads,
            x,
            E=E,
            nu=nu,
            assemble_tangent=True,
        )
        volume = float(np.sum(volumes))
        analytic_energy = 0.5 * float(np.sum(green * second_piola)) * volume
        rows.append(
            {
                "case": "finite_deformation_stvk_patch",
                "resolution": resolution,
                "nodes": X.shape[0],
                "elements": elements.shape[0],
                "numeric_energy": float(energy),
                "analytic_energy": analytic_energy,
                "energy_rel_error": abs(float(energy) - analytic_energy) / max(abs(analytic_energy), 1.0e-30),
                "max_green_strain_error": float(np.max(np.linalg.norm(strain - green, axis=(1, 2)))),
                "max_cauchy_stress_error": float(np.max(np.linalg.norm(stress - cauchy, axis=(1, 2)))),
                "max_von_mises": float(np.max(vm)),
                "passed": str(
                    abs(float(energy) - analytic_energy) / max(abs(analytic_energy), 1.0e-30) < 1.0e-10
                    and float(np.max(np.linalg.norm(strain - green, axis=(1, 2)))) < 1.0e-10
                ).lower(),
            }
        )
    return rows


def write_calculix_cantilever_input(model: CantileverModel, path: Path) -> None:
    """Write no-contact large-displacement cantilever input for CalculiX."""

    node_ids = np.arange(1, model.X.shape[0] + 1)
    lines = ["** Generated by validation/run_geometric_nonlinear_acceptance.py", "*NODE,NSET=Nall"]
    for node_id, point in zip(node_ids, model.X, strict=True):
        lines.append(f"{int(node_id)}, {point[0]:.12e}, {point[1]:.12e}, {point[2]:.12e}")
    lines.append("*ELEMENT,TYPE=C3D4,ELSET=Eall")
    for element_id, tet in enumerate(model.elements, start=1):
        ids = [int(node_ids[int(i)]) for i in tet]
        lines.append(f"{element_id}, {ids[0]}, {ids[1]}, {ids[2]}, {ids[3]}")
    fixed = node_ids[model.fixed_nodes]
    tip = node_ids[model.tip_nodes]
    lines.extend(["*NSET,NSET=Nfixed", _format_id_list(fixed), "*NSET,NSET=Ntip", _format_id_list(tip)])
    lines.extend(
        [
            "*MATERIAL,NAME=mat",
            "*ELASTIC",
            f"{model.E:.12g}, {model.nu:.12g}",
            "*SOLID SECTION,ELSET=Eall,MATERIAL=mat",
            "*BOUNDARY",
            "Nfixed,1,3",
            "*STEP,NLGEOM,INC=1000",
            "*STATIC",
            "0.1, 1.0, 1.0e-8, 0.1",
            "*CLOAD",
        ]
    )
    per_node = model.total_tip_load_z / len(tip)
    for node_id in tip:
        lines.append(f"{int(node_id)}, 3, {per_node:.12e}")
    lines.extend(
        [
            "*NODE PRINT,NSET=Nall,FREQUENCY=1",
            "U",
            "*NODE PRINT,NSET=Nfixed,TOTALS=ONLY,GLOBAL=YES,FREQUENCY=1",
            "RF",
            "*EL PRINT,ELSET=Eall,FREQUENCY=1",
            "S",
            "E",
            "*END STEP",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_id_list(values: np.ndarray, width: int = 12) -> str:
    return ", ".join(str(int(value)) for value in values[:width]) + ("," if len(values) <= width else ",\n" + _format_id_list(values[width:], width))


def run_calculix_static_cantilever(model: CantileverModel, out_dir: Path, *, timeout: int = 180) -> tuple[Row, Row]:
    """Run CalculiX static no-contact cantilever and parse final metrics."""

    if not calculix_available():
        return (
            {
                "case": "no_contact_cantilever",
                "resolution": model.resolution,
                "calculix_available": "false",
                "completed": "false",
                "tip_displacement_z": "",
                "reaction_z": "",
                "strain_energy_estimate": "",
                "max_von_mises": "",
                "mean_von_mises": "",
            },
            {"command": "", "completed": "false", "return_code": "not_available"},
        )
    run_dir = out_dir / "calculix_runs" / f"cantilever_r{model.resolution}"
    run_dir.mkdir(parents=True, exist_ok=True)
    case = f"cantilever_r{model.resolution}"
    inp = run_dir / f"{case}.inp"
    write_calculix_cantilever_input(model, inp)
    command = f"cd {_wsl_path(run_dir)} && ccx {case}"
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
    dat = run_dir / f"{case}.dat"
    command_row = {
        "case": "no_contact_cantilever",
        "resolution": model.resolution,
        "external_solver": "CalculiX",
        "external_solver_version": _calculix_version(),
        "command": f"wsl --exec bash -lc \"{command}\"",
        "input_file": str(inp.relative_to(out_dir)),
        "dat_file": str(dat.relative_to(out_dir)) if dat.exists() else "",
        "stdout_log": str((run_dir / "calculix_stdout.log").relative_to(out_dir)),
        "stderr_log": str((run_dir / "calculix_stderr.log").relative_to(out_dir)),
        "return_code": proc.returncode,
        "completed": str(proc.returncode == 0 and dat.exists()).lower(),
    }
    if proc.returncode != 0 or not dat.exists():
        return (
            {
                "case": "no_contact_cantilever",
                "resolution": model.resolution,
                "calculix_available": "true",
                "completed": "false",
                "tip_displacement_z": "",
                "reaction_z": "",
                "strain_energy_estimate": "",
                "max_von_mises": "",
                "mean_von_mises": "",
            },
            command_row,
        )
    displacements = _parse_calculix_dat_displacements_strict(dat, np.arange(1, model.X.shape[0] + 1))
    final_time = max(displacements)
    final_u = displacements[final_time]
    rf_z = _parse_final_total_rf_z(dat)
    stresses = _parse_final_stress_voigt(dat, model.elements.shape[0])
    vm = _voigt_von_mises(stresses)
    tip_uz = float(np.mean(final_u[model.tip_nodes, 2]))
    return (
        {
            "case": "no_contact_cantilever",
            "resolution": model.resolution,
            "calculix_available": "true",
            "completed": "true",
            "tip_displacement_z": tip_uz,
            "reaction_z": rf_z,
            "strain_energy_estimate": 0.5 * abs(model.total_tip_load_z * tip_uz),
            "max_von_mises": float(np.max(vm)),
            "mean_von_mises": float(np.mean(vm)),
        },
        command_row,
    )


def _parse_final_total_rf_z(path: Path) -> float:
    header = re.compile(r"total force .* time\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE)
    vector = re.compile(r"^\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)")
    pending = False
    final = 0.0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if header.search(line):
            pending = True
            continue
        if pending:
            match = vector.match(line)
            if match:
                final = float(match.group(3))
                pending = False
    return final


def _parse_calculix_dat_displacements_strict(path: Path, node_ids: np.ndarray) -> dict[float, np.ndarray]:
    """Parse only CalculiX displacement tables without consuming stress rows."""

    id_to_row = {int(node_id): i for i, node_id in enumerate(node_ids)}
    blocks: dict[float, np.ndarray] = {}
    current_time: float | None = None
    current = np.zeros((len(node_ids), 3), dtype=float)
    seen = 0
    in_block = False
    number = r"([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    header = re.compile(r"displacements.*time\s+" + number, re.IGNORECASE)
    row_pattern = re.compile(r"^\s*(\d+)\s+" + number + r"\s+" + number + r"\s+" + number + r"\s*$")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = header.search(line)
        if match:
            if current_time is not None and seen:
                blocks[current_time] = current.copy()
            current_time = float(match.group(1))
            current = np.zeros((len(node_ids), 3), dtype=float)
            seen = 0
            in_block = True
            continue
        if not in_block or current_time is None:
            continue
        row = row_pattern.match(line)
        if row is None:
            if seen and line.strip() == "":
                blocks[current_time] = current.copy()
                current_time = None
                in_block = False
            continue
        node_id = int(row.group(1))
        if node_id in id_to_row:
            current[id_to_row[node_id], :] = [float(row.group(2)), float(row.group(3)), float(row.group(4))]
            seen += 1
    if current_time is not None and seen:
        blocks[current_time] = current.copy()
    if not blocks:
        raise RuntimeError(f"no displacement blocks parsed from {path}")
    return dict(sorted(blocks.items()))


def _parse_final_stress_voigt(path: Path, n_elements: int) -> np.ndarray:
    header = re.compile(r"stresses .* time\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE)
    row_pattern = re.compile(
        r"^\s*(\d+)\s+\d+\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+"
        r"([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    )
    current: dict[int, list[float]] = {}
    final = np.zeros((n_elements, 6), dtype=float)
    in_block = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if header.search(line):
            if current:
                for eid, values in current.items():
                    if 1 <= eid <= n_elements:
                        final[eid - 1] = values
            current = {}
            in_block = True
            continue
        if not in_block:
            continue
        match = row_pattern.match(line)
        if match:
            current[int(match.group(1))] = [float(match.group(i)) for i in range(2, 8)]
        elif current and line.strip() == "":
            in_block = False
    if current:
        for eid, values in current.items():
            if 1 <= eid <= n_elements:
                final[eid - 1] = values
    return final


def _voigt_von_mises(stresses: np.ndarray) -> np.ndarray:
    out = np.zeros(stresses.shape[0], dtype=float)
    for i, row in enumerate(stresses):
        sxx, syy, szz, sxy, sxz, syz = row
        tensor = np.asarray([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]], dtype=float)
        out[i] = _von_mises(tensor)
    return out


def compare_cantilever(
    resolutions: list[int],
    out_dir: Path,
    *,
    run_external: bool,
) -> tuple[list[Row], list[Row], list[Row], list[Row]]:
    """Run SFC cantilever convergence and optional CalculiX comparison."""

    comparison_rows: list[Row] = []
    mesh_rows: list[Row] = []
    newton_rows: list[Row] = []
    command_rows: list[Row] = []
    previous_tip: float | None = None
    previous_energy: float | None = None
    for resolution in resolutions:
        model = _cantilever_model(resolution)
        sfc, iterations, _ = solve_sfc_static_cantilever(model)
        newton_rows.extend(iterations)
        mesh_rows.append(
            {
                "case": "no_contact_cantilever",
                "resolution": resolution,
                "nodes": sfc["nodes"],
                "elements": sfc["elements"],
                "tip_displacement_z": sfc["tip_displacement_z"],
                "strain_energy": sfc["strain_energy"],
                "max_von_mises": sfc["max_von_mises"],
                "tip_change_from_previous": "" if previous_tip is None else abs(float(sfc["tip_displacement_z"]) - previous_tip),
                "energy_change_from_previous": "" if previous_energy is None else abs(float(sfc["strain_energy"]) - previous_energy),
            }
        )
        previous_tip = float(sfc["tip_displacement_z"])
        previous_energy = float(sfc["strain_energy"])
        if run_external:
            calc, command = run_calculix_static_cantilever(model, out_dir)
        else:
            calc = {
                "case": "no_contact_cantilever",
                "resolution": resolution,
                "calculix_available": str(calculix_available()).lower(),
                "completed": "false",
                "tip_displacement_z": "",
                "reaction_z": "",
                "strain_energy_estimate": "",
                "max_von_mises": "",
                "mean_von_mises": "",
            }
            command = {"case": "no_contact_cantilever", "resolution": resolution, "command": "", "completed": "false", "return_code": "skipped"}
        command_rows.append(command)
        comparison_rows.append(_comparison_row(sfc, calc))
    return comparison_rows, mesh_rows, newton_rows, command_rows


def _comparison_row(sfc: Row, calc: Row) -> Row:
    completed = str(calc.get("completed", "false")).lower() == "true"
    row = {
        "case": "no_contact_cantilever",
        "resolution": sfc["resolution"],
        "sfc_tip_displacement_z": sfc["tip_displacement_z"],
        "calculix_tip_displacement_z": calc.get("tip_displacement_z", ""),
        "tip_displacement_rel_error": "",
        "sfc_reaction_z": sfc["reaction_z"],
        "calculix_reaction_z": calc.get("reaction_z", ""),
        "reaction_rel_error": "",
        "sfc_strain_energy": sfc["strain_energy"],
        "calculix_strain_energy_estimate": calc.get("strain_energy_estimate", ""),
        "energy_rel_error": "",
        "sfc_max_von_mises": sfc["max_von_mises"],
        "calculix_max_von_mises": calc.get("max_von_mises", ""),
        "stress_rel_error": "",
        "calculix_completed": str(completed).lower(),
        "comparison_status": "external_unavailable_or_skipped",
    }
    if completed:
        row["tip_displacement_rel_error"] = _relative_error(float(sfc["tip_displacement_z"]), float(calc["tip_displacement_z"]))
        row["reaction_rel_error"] = _relative_error(float(sfc["reaction_z"]), float(calc["reaction_z"]))
        row["energy_rel_error"] = _relative_error(float(sfc["strain_energy"]), float(calc["strain_energy_estimate"]))
        row["stress_rel_error"] = _relative_error(float(sfc["max_von_mises"]), float(calc["max_von_mises"]))
        row["comparison_status"] = "computed"
    return row


def _relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0e-30)


def timestep_convergence(out_dir: Path, *, quick: bool) -> list[Row]:
    """Run block-plane contact with dt, dt/2, dt/4 after no-contact checks."""

    dts = [0.004, 0.002, 0.001] if quick else [0.002, 0.001, 0.0005]
    duration = 0.02 if quick else 0.08
    rows: list[Row] = []
    previous_final_z: float | None = None
    previous_energy: float | None = None
    for dt in dts:
        case_dir = out_dir / "timestep_runs" / f"block_contact_dt_{dt:g}".replace(".", "p")
        outputs = run_simulation(
            case_dir,
            case="block_drop",
            integrator="implicit_hht",
            hht_alpha=0.0,
            resolution=1,
            duration=duration,
            dt=dt,
            frame_stride=max(1, int(np.ceil(duration / dt))),
            initial_gap=0.0,
            contact_stiffness=5000.0,
        )
        with outputs["history"].open(newline="", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        final = history[-1]
        energies = [float(row["total_energy"]) for row in history]
        rows.append(
            {
                "case": "block_plane_contact_timestep",
                "dt": dt,
                "duration": duration,
                "final_z_cm": final["z_cm"],
                "min_gap": min(float(row["min_gap"]) for row in history),
                "max_penetration": max(float(row["max_penetration"]) for row in history),
                "max_newton_iterations": max(int(row["newton_iterations"]) for row in history),
                "max_newton_residual_norm": max(float(row["newton_residual_norm"]) for row in history),
                "energy_drift_range": max(energies) - min(energies),
                "final_z_change_from_previous": "" if previous_final_z is None else abs(float(final["z_cm"]) - previous_final_z),
                "energy_range_change_from_previous": "" if previous_energy is None else abs((max(energies) - min(energies)) - previous_energy),
            }
        )
        previous_final_z = float(final["z_cm"])
        previous_energy = max(energies) - min(energies)
    return rows


def block_plane_contact_acceptance(out_dir: Path, *, quick: bool) -> list[Row]:
    """Run a short contact acceptance case after no-contact checks."""

    duration = 0.02 if quick else 0.08
    dt = 0.001 if quick else 0.0005
    outputs = run_simulation(
        out_dir / "block_plane_contact_acceptance",
        case="block_drop",
        integrator="implicit_hht",
        hht_alpha=0.0,
        resolution=1,
        duration=duration,
        dt=dt,
        frame_stride=max(1, int(np.ceil(duration / dt))),
        initial_gap=0.0,
        contact_stiffness=5000.0,
    )
    with outputs["history"].open(newline="", encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    return [
        {
            "case": "block_plane_contact_acceptance",
            "duration": duration,
            "dt": dt,
            "initial_gap": history[0]["min_gap"],
            "min_gap": min(float(row["min_gap"]) for row in history),
            "max_penetration": max(float(row["max_penetration"]) for row in history),
            "max_contact_tangent_norm": max(float(row["contact_tangent_norm"]) for row in history),
            "max_newton_iterations": max(int(row["newton_iterations"]) for row in history),
            "max_newton_residual_norm": max(float(row["newton_residual_norm"]) for row in history),
            "vtk_frames_csv": str(outputs["frames"].relative_to(out_dir)),
            "status": "passed" if any(int(row["active_contact_count"]) > 0 for row in history) else "failed_no_contact_activation",
        }
    ]


def run_acceptance(out_dir: Path, *, quick: bool = False, run_calculix: bool = True) -> dict[str, Path]:
    """Run all geometric nonlinear acceptance checks and write outputs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    resolutions = [1, 2, 3]
    if quick:
        resolutions = [1, 2, 3]
    patch_rows = stvk_patch_test(resolutions)
    comparison_rows, mesh_rows, newton_rows, command_rows = compare_cantilever(
        resolutions if not quick else [1],
        out_dir,
        run_external=run_calculix,
    )
    # Always keep a three-resolution SFC mesh trend, even if external CalculiX
    # is only run for the smallest quick case.
    if quick:
        _, mesh_rows, extra_newton, _ = compare_cantilever(resolutions, out_dir, run_external=False)
        newton_rows.extend(extra_newton)
    timestep_rows = timestep_convergence(out_dir, quick=quick)
    contact_rows = block_plane_contact_acceptance(out_dir, quick=quick)
    claim_rows = _claim_rows(patch_rows, comparison_rows, mesh_rows, newton_rows, timestep_rows, contact_rows)

    outputs = {
        "patch": out_dir / "stvk_patch.csv",
        "cantilever": out_dir / "cantilever_calculix_comparison.csv",
        "newton": out_dir / "newton_convergence.csv",
        "mesh": out_dir / "mesh_convergence.csv",
        "timestep": out_dir / "timestep_convergence.csv",
        "contact": out_dir / "block_plane_contact_acceptance.csv",
        "commands": out_dir / "external_solver_commands.csv",
        "claims": out_dir / "geometric_nonlinear_acceptance_claims.csv",
        "summary": out_dir / "geometric_nonlinear_acceptance_summary.md",
    }
    _write_csv(outputs["patch"], patch_rows)
    _write_csv(outputs["cantilever"], comparison_rows)
    _write_csv(outputs["newton"], newton_rows)
    _write_csv(outputs["mesh"], mesh_rows)
    _write_csv(outputs["timestep"], timestep_rows)
    _write_csv(outputs["contact"], contact_rows)
    _write_csv(outputs["commands"], command_rows)
    _write_csv(outputs["claims"], claim_rows)
    _write_markdown(outputs["summary"], patch_rows, comparison_rows, mesh_rows, newton_rows, timestep_rows, contact_rows, claim_rows)
    return outputs


def _claim_rows(
    patch_rows: list[Row],
    comparison_rows: list[Row],
    mesh_rows: list[Row],
    newton_rows: list[Row],
    timestep_rows: list[Row],
    contact_rows: list[Row],
) -> list[Row]:
    patch_pass = all(row["passed"] == "true" for row in patch_rows)
    has_three_meshes = len({int(row["resolution"]) for row in mesh_rows}) >= 3
    external_done = any(row["calculix_completed"] == "true" for row in comparison_rows)
    final_by_resolution: dict[int, Row] = {}
    for row in newton_rows:
        final_by_resolution[int(row["resolution"])] = row
    newton_ok = bool(final_by_resolution) and all(float(row["residual_norm"]) < 1.0e-5 for row in final_by_resolution.values())
    timestep_has_three = len(timestep_rows) >= 3
    contact_ok = bool(contact_rows and contact_rows[0]["status"] == "passed")
    no_contact_gate = patch_pass and has_three_meshes and newton_ok
    return [
        {
            "claim": "analytic_finite_deformation_stvk_patch",
            "supported": str(patch_pass).lower(),
            "evidence_csv": "stvk_patch.csv",
            "details": "affine Green-Lagrange strain, Cauchy stress, and energy",
        },
        {
            "claim": "no_contact_large_displacement_calculix_comparison",
            "supported": str(external_done).lower(),
            "evidence_csv": "cantilever_calculix_comparison.csv",
            "details": "tip displacement, fixed reaction, energy estimate, and stress are compared when CalculiX completes",
        },
        {
            "claim": "mesh_convergence_trend_available",
            "supported": str(has_three_meshes).lower(),
            "evidence_csv": "mesh_convergence.csv",
            "details": "trend only; no theoretical convergence order is claimed",
        },
        {
            "claim": "newton_convergence_table_available",
            "supported": str(bool(newton_rows)).lower(),
            "evidence_csv": "newton_convergence.csv",
            "details": "per-iteration residual and correction norms",
        },
        {
            "claim": "timestep_convergence_trend_available",
            "supported": str(timestep_has_three).lower(),
            "evidence_csv": "timestep_convergence.csv",
            "details": "dt, dt/2, dt/4 block-plane contact trend",
        },
        {
            "claim": "contact_checked_after_no_contact_gates",
            "supported": str(no_contact_gate and contact_ok).lower(),
            "evidence_csv": "block_plane_contact_acceptance.csv",
            "details": "contact rows are generated after patch/cantilever/no-contact Newton evidence",
        },
    ]


def _write_markdown(
    path: Path,
    patch_rows: list[Row],
    comparison_rows: list[Row],
    mesh_rows: list[Row],
    newton_rows: list[Row],
    timestep_rows: list[Row],
    contact_rows: list[Row],
    claim_rows: list[Row],
) -> None:
    lines = [
        "# Geometric Nonlinear Acceptance Handoff",
        "",
        "This validation package checks geometric nonlinearity before contact claims.",
        "",
        "## Outputs",
        "",
        "- `stvk_patch.csv`",
        "- `cantilever_calculix_comparison.csv`",
        "- `newton_convergence.csv`",
        "- `mesh_convergence.csv`",
        "- `timestep_convergence.csv`",
        "- `block_plane_contact_acceptance.csv`",
        "- `geometric_nonlinear_acceptance_claims.csv`",
        "",
        "## Status",
        "",
        f"- StVK patch rows: `{len(patch_rows)}`",
        f"- Cantilever comparison rows: `{len(comparison_rows)}`",
        f"- Newton rows: `{len(newton_rows)}`",
        f"- Mesh trend rows: `{len(mesh_rows)}`",
        f"- Time-step trend rows: `{len(timestep_rows)}`",
        f"- Contact acceptance rows: `{len(contact_rows)}`",
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
            "## Limitations",
            "",
            "- The material is St. Venant-Kirchhoff, not a general hyperelastic model.",
            "- CalculiX comparison is scoped to a no-contact cantilever and may not be numerically identical because CalculiX uses its own nonlinear strain/stress update details.",
            "- Mesh/time-step tables report trends only; they do not claim theoretical convergence order.",
            "- Block-plane contact is checked only after no-contact gates are generated.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "geometric_nonlinear_acceptance")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-calculix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_acceptance(args.out_dir, quick=args.quick, run_calculix=not args.skip_calculix)
    print("Geometric nonlinear acceptance complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
