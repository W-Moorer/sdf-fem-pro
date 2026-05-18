"""CalculiX comparison for FuzzyContact published-parameter SFC cases.

The runner builds the FuzzyContact-inspired problem 1/3/4 cases from the
published-parameter SFC benchmark, computes SDF-field contact loads in SFC, and
then runs CalculiX on the same master TET4/C3D4 mesh with the same equivalent
nodal contact load and bottom constraints.

This is intentionally an external FEM field comparison.  It checks whether the
SFC displacement, strain, stress, and von Mises fields agree with CalculiX when
both solvers receive the same mesh, material, boundary condition, and SDF-derived
contact load.  It does not claim that CalculiX's native contact search or
pressure-overclosure implementation is equivalent to SFC field contact.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.fem.constraints import fixed_dofs_from_node_set  # noqa: E402
from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.fem.tet4 import tet4_strain_displacement_matrix  # noqa: E402
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402
from validation.run_calculix_drop_impact_comparison import (  # noqa: E402
    _calculix_version,
    _parse_calculix_dat_displacements,
    _wsl_path,
    calculix_available,
)
from validation.run_fuzzycontact_published_benchmarks import (  # noqa: E402
    PublishedCase,
    _area_weighted_penetration_sum,
    _find_closure_for_load,
    _linear_compression_reference,
    _make_cases,
    _solve_master_response,
    _von_mises_voigt,
)

Row = dict[str, Any]


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _calibrate_case(case: PublishedCase, sdf: DynamicNarrowBandSDF) -> PublishedCase:
    if case.problem == "problem_4":
        calibration_closure = _linear_compression_reference(case, float(case.reference_loads_N[-1]))
        calibration_load = float(case.reference_loads_N[-1])
    else:
        calibration_closure = float(case.reference_metric_value)
        calibration_load = float(case.reference_loads_N[0])
    penetration_integral = _area_weighted_penetration_sum(case, sdf, calibration_closure)
    if penetration_integral <= 1.0e-15:
        return case
    return replace(case, contact_pressure_stiffness_N_per_mm3=calibration_load / penetration_integral)


def _sfc_case_state(case: PublishedCase) -> tuple[DynamicNarrowBandSDF, np.ndarray, np.ndarray, Row]:
    sdf = DynamicNarrowBandSDF.build(
        case.body.mesh.X,
        case.top_faces,
        spacing=case.spacing_mm,
        band_radius=case.band_radius_mm,
        padding=case.band_radius_mm,
        cell_size=case.spacing_mm,
        gradient_mode="finite_difference",
    )
    case = _calibrate_case(case, sdf)
    contact = _find_closure_for_load(case, sdf, float(case.reference_loads_N[-1]))
    u, max_vm, max_strain, top_mean_uz = _solve_master_response(case, contact.force_master)
    return (
        sdf,
        contact.force_master,
        u,
        {
            "problem": case.problem,
            "target_load_N": float(case.reference_loads_N[-1]),
            "sfc_reaction_N": contact.reaction_N,
            "sfc_closure_mm": contact.closure_mm,
            "sfc_min_gap_mm": contact.min_gap_mm,
            "sfc_max_penetration_mm": contact.max_penetration_mm,
            "sfc_active_samples": contact.active_samples,
            "sfc_top_mean_uz_mm": top_mean_uz,
            "sfc_max_von_mises_MPa": max_vm,
            "sfc_max_strain_norm": max_strain,
            "field_grid_nodes": sdf.stats.grid_node_count,
            "field_valid_nodes": sdf.stats.valid_node_count,
            "field_update_seconds": sdf.stats.update_seconds,
            "contact_pressure_stiffness_N_per_mm3": case.contact_pressure_stiffness_N_per_mm3,
        },
    )


def _element_strain_stress(case: PublishedCase, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = isotropic_linear_elasticity_matrix(float(case.body.material["E"]), float(case.body.material["nu"]))
    strains: list[np.ndarray] = []
    stresses: list[np.ndarray] = []
    mises: list[float] = []
    u_nodes = np.asarray(u, dtype=float).reshape((-1, 3))
    for element in case.body.mesh.elements:
        B = tet4_strain_displacement_matrix(case.body.mesh.X[element])
        strain = B @ u_nodes[element].reshape(12)
        stress = C @ strain
        strains.append(strain)
        stresses.append(stress)
        mises.append(_von_mises_voigt(stress))
    return np.vstack(strains), np.vstack(stresses), np.asarray(mises, dtype=float)


def _format_ids(ids: np.ndarray, *, per_line: int = 12) -> list[str]:
    values = [str(int(v)) for v in np.asarray(ids, dtype=np.int64).ravel()]
    return [", ".join(values[i : i + per_line]) for i in range(0, len(values), per_line)]


def _write_calculix_input(case: PublishedCase, force: np.ndarray, path: Path) -> None:
    mesh = case.body.mesh
    node_ids = np.arange(1, mesh.X.shape[0] + 1, dtype=np.int64)
    fixed = fixed_dofs_from_node_set(case.bottom_nodes, "xyz")
    fixed_nodes = np.unique(np.asarray(case.bottom_nodes, dtype=np.int64) + 1)
    force = np.asarray(force, dtype=float).ravel()
    lines = [
        "** Generated by validation/run_fuzzycontact_calculix_comparison.py",
        "** External CalculiX FEM solve with SFC dynamic-SDF equivalent contact loads.",
        "*HEADING",
        f"FuzzyContact published-parameter equivalent-load case {case.problem}",
        "*NODE, NSET=NALL",
    ]
    for node_id, xyz in zip(node_ids, mesh.X, strict=True):
        lines.append(f"{int(node_id)}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    lines.append("*ELEMENT, TYPE=C3D4, ELSET=EALL")
    for eid, element in enumerate(mesh.elements, start=1):
        conn = ", ".join(str(int(node) + 1) for node in element)
        lines.append(f"{eid}, {conn}")
    lines.append("*NSET, NSET=NFIXED")
    lines.extend(_format_ids(fixed_nodes))
    lines.extend(
        [
            "*MATERIAL, NAME=MAT",
            "*ELASTIC",
            f"{float(case.body.material['E']):.12e}, {float(case.body.material['nu']):.12e}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT",
            "*BOUNDARY",
            "NFIXED, 1, 3, 0.0",
            "*STEP",
            "*STATIC",
            "1.0, 1.0, 1e-05, 1.0",
            "*CLOAD",
        ]
    )
    fixed_set = set(int(dof) for dof in fixed)
    for dof, value in enumerate(force):
        if dof in fixed_set or abs(float(value)) <= 1.0e-12:
            continue
        node = dof // 3 + 1
        component = dof % 3 + 1
        lines.append(f"{node}, {component}, {float(value):.12e}")
    lines.extend(
        [
            "*NODE PRINT, NSET=NALL, FREQUENCY=1",
            "U",
            "*EL PRINT, ELSET=EALL, FREQUENCY=1",
            "S",
            "E",
            "*END STEP",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_calculix(job: str, run_dir: Path, *, timeout: int, skip_calculix: bool) -> Row:
    inp = run_dir / f"{job}.inp"
    dat = run_dir / f"{job}.dat"
    stdout_log = run_dir / "calculix_stdout.log"
    stderr_log = run_dir / "calculix_stderr.log"
    command = f"cd {_wsl_path(run_dir)} && ccx {job}"
    if skip_calculix or not calculix_available():
        return {
            "case": job,
            "external_solver": "CalculiX",
            "external_solver_version": "skipped" if skip_calculix else "not_available",
            "command": f"wsl --exec bash -lc \"{command}\"",
            "input_file": str(inp),
            "dat_file": "",
            "stdout_log": "",
            "stderr_log": "",
            "return_code": "skipped" if skip_calculix else "not_available",
            "completed": "false",
        }
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=int(timeout),
    )
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")
    return {
        "case": job,
        "external_solver": "CalculiX",
        "external_solver_version": _calculix_version(),
        "command": f"wsl --exec bash -lc \"{command}\"",
        "input_file": str(inp),
        "dat_file": str(dat) if dat.exists() else "",
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "return_code": proc.returncode,
        "completed": str(proc.returncode == 0 and dat.exists()).lower(),
    }


def _parse_element_table(path: Path, quantity: str, n_elements: int) -> np.ndarray:
    number = r"([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    header = re.compile(fr"{quantity} .* time\s+" + number, re.IGNORECASE)
    row_pattern = re.compile(
        r"^\s*(\d+)\s+\d+\s+"
        + number
        + r"\s+"
        + number
        + r"\s+"
        + number
        + r"\s+"
        + number
        + r"\s+"
        + number
        + r"\s+"
        + number
    )
    final = np.zeros((n_elements, 6), dtype=float)
    current: dict[int, list[float]] = {}
    in_block = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if header.search(line):
            if current:
                _store_element_values(final, current)
            current = {}
            in_block = True
            continue
        if not in_block:
            continue
        match = row_pattern.match(line)
        if match is None:
            if current and line.strip() == "":
                _store_element_values(final, current)
                current = {}
                in_block = False
            continue
        # CalculiX prints xx, yy, zz, xy, xz, yz. Convert to SFC order:
        # xx, yy, zz, xy, yz, xz.  For strain output CalculiX reports tensor
        # shear components, while SFC's TET4 B matrix stores engineering shear.
        xx, yy, zz, xy, xz, yz = [float(match.group(i)) for i in range(2, 8)]
        if quantity.lower().startswith("strain"):
            xy *= 2.0
            xz *= 2.0
            yz *= 2.0
        current[int(match.group(1))] = [xx, yy, zz, xy, yz, xz]
    if current:
        _store_element_values(final, current)
    return final


def _store_element_values(out: np.ndarray, values: dict[int, list[float]]) -> None:
    for eid, row in values.items():
        if 1 <= int(eid) <= out.shape[0]:
            out[int(eid) - 1] = row


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    if denom <= 1.0e-30:
        return float(np.linalg.norm(value - reference))
    return float(np.linalg.norm(value - reference) / denom)


def _compare_completed_case(case: PublishedCase, dat: Path, u_sfc: np.ndarray, sfc_row: Row) -> Row:
    node_ids = np.arange(1, case.body.mesh.X.shape[0] + 1, dtype=np.int64)
    displacement_blocks = _parse_calculix_dat_displacements(dat, node_ids)
    final_time = max(displacement_blocks)
    u_calc = displacement_blocks[final_time].reshape((-1,))
    strain_sfc, stress_sfc, vm_sfc = _element_strain_stress(case, u_sfc)
    strain_calc = _parse_element_table(dat, "strains", case.body.mesh.elements.shape[0])
    stress_calc = _parse_element_table(dat, "stresses", case.body.mesh.elements.shape[0])
    vm_calc = np.asarray([_von_mises_voigt(row) for row in stress_calc], dtype=float)
    top_nodes = np.nonzero(np.isclose(case.body.mesh.X[:, 2], float(np.max(case.body.mesh.X[:, 2]))))[0]
    top_calc = float(np.mean(u_calc.reshape((-1, 3))[top_nodes, 2])) if top_nodes.size else 0.0
    return {
        **sfc_row,
        "calculix_completed": "true",
        "calculix_final_time": final_time,
        "calculix_top_mean_uz_mm": top_calc,
        "calculix_max_von_mises_MPa": float(np.max(vm_calc)) if vm_calc.size else 0.0,
        "calculix_max_strain_norm": float(np.max(np.linalg.norm(strain_calc, axis=1))) if strain_calc.size else 0.0,
        "displacement_l2_rel_error": _relative_error(u_sfc, u_calc),
        "displacement_linf_abs_error": float(np.max(np.abs(u_sfc - u_calc))),
        "strain_l2_rel_error": _relative_error(strain_sfc, strain_calc),
        "stress_l2_rel_error": _relative_error(stress_sfc, stress_calc),
        "von_mises_l2_rel_error": _relative_error(vm_sfc, vm_calc),
        "top_mean_uz_abs_error_mm": abs(float(sfc_row["sfc_top_mean_uz_mm"]) - top_calc),
        "status": "passed" if _relative_error(u_sfc, u_calc) < 1.0e-8 and _relative_error(stress_sfc, stress_calc) < 1.0e-8 else "check",
        "details": "same TET4/C3D4 mesh, material, bottom constraints, and SFC dynamic-SDF equivalent contact load",
    }


def _write_summary(path: Path, comparison_rows: list[Row], command_rows: list[Row], outputs: dict[str, Path]) -> None:
    lines = [
        "# FuzzyContact CalculiX Comparison",
        "",
        "CalculiX is used only as an external validation reference.  SFC computes the dynamic-SDF field-contact load; CalculiX receives the same equivalent nodal load on the same master C3D4 mesh.",
        "",
        "## Scope",
        "",
        "- Supports displacement/stress/strain field comparison under the same equivalent contact load.",
        "- Does not claim native CalculiX contact-law equivalence or full FuzzyContact trajectory equivalence.",
        "- Does not add a core CalculiX dependency.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        if key != "summary":
            lines.append(f"- `{value.relative_to(path.parent).as_posix()}`")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Problem | CalculiX | disp rel | strain rel | stress rel | VM rel | SFC max VM | CCX max VM |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| `{row['problem']}` | {row['calculix_completed']} | "
            f"{_fmt(row.get('displacement_l2_rel_error'))} | "
            f"{_fmt(row.get('strain_l2_rel_error'))} | "
            f"{_fmt(row.get('stress_l2_rel_error'))} | "
            f"{_fmt(row.get('von_mises_l2_rel_error'))} | "
            f"{_fmt(row.get('sfc_max_von_mises_MPa'))} | "
            f"{_fmt(row.get('calculix_max_von_mises_MPa'))} |"
        )
    completed = sum(1 for row in command_rows if row.get("completed") == "true")
    lines.extend(
        [
            "",
            "## Command Status",
            "",
            f"- Completed CalculiX jobs: `{completed}/{len(command_rows)}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.6e}"
    except (TypeError, ValueError):
        return str(value)


def run_validation(out_dir: Path, *, quick: bool = False, timeout: int = 240, skip_calculix: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[Row] = []
    command_rows: list[Row] = []
    for case in _make_cases(quick=quick):
        _sdf, force, u_sfc, sfc_row = _sfc_case_state(case)
        run_dir = out_dir / "calculix_runs" / case.problem
        job = f"fuzzycontact_{case.problem}_equivalent_load"
        _write_calculix_input(case, force, run_dir / f"{job}.inp")
        command = _run_calculix(job, run_dir, timeout=timeout, skip_calculix=skip_calculix)
        command_rows.append(command)
        dat_file = command.get("dat_file", "")
        if command.get("completed") == "true" and dat_file:
            comparison_rows.append(_compare_completed_case(case, Path(str(dat_file)), u_sfc, sfc_row))
        else:
            comparison_rows.append(
                {
                    **sfc_row,
                    "calculix_completed": command.get("completed", "false"),
                    "calculix_final_time": "",
                    "calculix_top_mean_uz_mm": "",
                    "calculix_max_von_mises_MPa": "",
                    "calculix_max_strain_norm": "",
                    "displacement_l2_rel_error": "",
                    "displacement_linf_abs_error": "",
                    "strain_l2_rel_error": "",
                    "stress_l2_rel_error": "",
                    "von_mises_l2_rel_error": "",
                    "top_mean_uz_abs_error_mm": "",
                    "status": "not_run" if skip_calculix else "not_available",
                    "details": "CalculiX did not complete or was skipped.",
                }
            )
    outputs = {
        "comparison": out_dir / "fuzzycontact_calculix_comparison.csv",
        "commands": out_dir / "fuzzycontact_calculix_commands.csv",
        "summary": out_dir / "fuzzycontact_calculix_summary.md",
    }
    _write_csv(outputs["comparison"], comparison_rows)
    _write_csv(outputs["commands"], command_rows)
    _write_summary(outputs["summary"], comparison_rows, command_rows, outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--skip-calculix", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "fuzzycontact_calculix_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(
        args.out_dir,
        quick=bool(args.quick),
        timeout=int(args.timeout),
        skip_calculix=bool(args.skip_calculix),
    )
    print("FuzzyContact CalculiX comparison complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
