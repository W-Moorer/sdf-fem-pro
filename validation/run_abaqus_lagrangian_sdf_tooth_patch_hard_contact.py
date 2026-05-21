"""Abaqus HARD contact vs SFC Lagrangian-SDF on a tilted tooth patch.

This validation is the next alignment layer after the flat two-flexible-block
case.  It keeps the model small but makes the contact interface non-axis-
aligned, so SFC must use Q4 slave quadrature, Q4 master closest-feature
payloads, and normal-direction boundary/reaction projections consistently.

Abaqus is used only as an external reference.  The SFC path assembles C3D8
linear-elastic stiffness internally and solves the Lagrangian-SDF HARD-contact
KKT problem without importing Abaqus into the core package.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import block_diag

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sfc.contact.hard_contact import (  # noqa: E402
    hard_contact_gap_jacobian_from_samples,
    hard_contact_pressure_compliance_from_samples,
    solve_linear_hard_contact_with_dirichlet,
)
from sfc.contact.lagrangian_surface_contact import LagrangianSDFQ4MasterSurfaceContactGeometry  # noqa: E402
from sfc.fem import DeformableBody  # noqa: E402
from sfc.fem.assembler import assemble_stiffness_matrix  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "abaqus_lagrangian_sdf_tooth_patch_hard_contact"


@dataclass(frozen=True, slots=True)
class ToothPatchHardContactCase:
    """Two flexible C3D8 skew blocks pressed through a tilted tooth-like patch."""

    length: float = 1.0
    width: float = 0.75
    thickness: float = 0.55
    origin_z: float = 0.70
    slope_x: float = 0.28
    slope_y: float = -0.16
    initial_gap: float = 0.018
    closure: float = 0.048
    young_modulus: float = 12000.0
    poisson_ratio: float = 0.30
    density: float = 1.0
    quadrature_order: int = 2
    contact_enforcement: str = "pressure_compliance"
    pressure_smoothing_stiffness_factor: float = 4.5


def build_case() -> ToothPatchHardContactCase:
    return ToothPatchHardContactCase()


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tooth_patch_basis(case: ToothPatchHardContactCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1 = np.asarray([case.length, 0.0, case.slope_x * case.length], dtype=float)
    t2 = np.asarray([0.0, case.width, case.slope_y * case.width], dtype=float)
    normal = np.cross(t1, t2)
    normal /= float(np.linalg.norm(normal))
    return t1, t2, normal


def make_geometry(case: ToothPatchHardContactCase) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t1, t2, normal = tooth_patch_basis(case)
    p0 = np.asarray([0.0, 0.0, case.origin_z], dtype=float)
    top = np.asarray([p0, p0 + t1, p0 + t1 + t2, p0 + t2], dtype=float)
    lower = np.vstack((top - case.thickness * normal, top))
    upper_bottom = top + case.initial_gap * normal
    upper = np.vstack((upper_bottom, upper_bottom + case.thickness * normal))
    elements = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    return lower, upper, elements, normal


def write_abaqus_deck(case: ToothPatchHardContactCase, path: Path) -> None:
    lower, upper, elements, normal = make_geometry(case)
    closure_vec = -float(case.closure) * normal
    lines: list[str] = [
        "*Heading",
        "Tilted tooth-patch two-flexible-body HARD contact alignment",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
    ]
    for name, nodes in (("LOWER", lower), ("UPPER", upper)):
        lines.extend([f"*Part, name={name}", "*Node"])
        for label, xyz in enumerate(nodes, start=1):
            lines.append(f"{label}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
        mat = "LOWER_MAT" if name == "LOWER" else "UPPER_MAT"
        lines.extend(
            [
                "*Element, type=C3D8, elset=EALL",
                "1, " + ", ".join(str(int(node) + 1) for node in elements[0]),
                f"*Solid Section, elset=EALL, material={mat}",
                ",",
                "*End Part",
            ]
        )
    lines.extend(
        [
            "*Assembly, name=ASSEMBLY",
            "*Instance, name=LOWER-1, part=LOWER",
            "*End Instance",
            "*Instance, name=UPPER-1, part=UPPER",
            "*End Instance",
            "*Elset, elset=LOWER_TOP_ASM, instance=LOWER-1",
            "1",
            "*Elset, elset=UPPER_BOTTOM_ASM, instance=UPPER-1",
            "1",
            "*Nset, nset=LOWER_BOTTOM_ASM, instance=LOWER-1",
            "1, 2, 3, 4",
            "*Nset, nset=LOWER_TOP_ASM, instance=LOWER-1",
            "5, 6, 7, 8",
            "*Nset, nset=UPPER_BOTTOM_ASM, instance=UPPER-1",
            "1, 2, 3, 4",
            "*Nset, nset=UPPER_TOP_ASM, instance=UPPER-1",
            "5, 6, 7, 8",
            "*Surface, type=ELEMENT, name=LOWER_TOP_SURF",
            "LOWER_TOP_ASM, S2",
            "*Surface, type=ELEMENT, name=UPPER_BOTTOM_SURF",
            "UPPER_BOTTOM_ASM, S1",
            "*End Assembly",
            "*Material, name=LOWER_MAT",
            "*Density",
            f"{case.density:.12e}",
            "*Elastic",
            f"{case.young_modulus:.12e}, {case.poisson_ratio:.12e}",
            "*Material, name=UPPER_MAT",
            "*Density",
            f"{case.density:.12e}",
            "*Elastic",
            f"{case.young_modulus:.12e}, {case.poisson_ratio:.12e}",
            "*Surface Interaction, name=HARD_FRICTIONLESS",
            "*Surface Behavior, pressure-overclosure=HARD",
            "*Friction",
            "0.",
            "*Contact Pair, interaction=HARD_FRICTIONLESS, type=SURFACE TO SURFACE",
            "LOWER_TOP_SURF, UPPER_BOTTOM_SURF",
            "*Step, name=TOOTH_PATCH_HARD_PRESS, nlgeom=YES, inc=200",
            "*Static",
            "1., 1., 1e-08, 1.",
            "*Boundary",
            "LOWER_BOTTOM_ASM, 1, 3, 0.",
            f"UPPER_TOP_ASM, 1, 1, {closure_vec[0]:.12e}",
            f"UPPER_TOP_ASM, 2, 2, {closure_vec[1]:.12e}",
            f"UPPER_TOP_ASM, 3, 3, {closure_vec[2]:.12e}",
            "*Output, field, frequency=1",
            "*Node Output",
            "U, RF",
            "*Contact Output",
            "CSTRESS, CDISP",
            "*End Step",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def _resolve_abaqus_command(command: str | None) -> str:
    if command:
        return command
    for candidate in ("abaqus", "abq2024"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Abaqus command not found; pass --abaqus-command")


def _run_command(command: list[str], *, cwd: Path, log_path: Path, timeout: int) -> float:
    start = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=timeout)
    elapsed = time.perf_counter() - start
    log_path.write_text(result.stdout, encoding="utf-8", errors="ignore")
    print(result.stdout)
    text = result.stdout.lower()
    if result.returncode != 0 or "exited with errors" in text or "fatal errors" in text or "traceback" in text:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}")
    return elapsed


def _abaqus_export_script() -> str:
    return r'''
from __future__ import print_function

import csv
import sys

from odbAccess import openOdb

odb = openOdb(path=sys.argv[1], readOnly=True)
out_path = sys.argv[2]
initial_gap = float(sys.argv[3])
nx = float(sys.argv[4])
ny = float(sys.argv[5])
nz = float(sys.argv[6])
normal = (nx, ny, nz)
try:
    step = odb.steps[list(odb.steps.keys())[-1]]
    frame = step.frames[-1]
    u_values = {}
    rf_values = {}
    if "U" in frame.fieldOutputs:
        for value in frame.fieldOutputs["U"].values:
            u_values[(value.instance.name.upper(), int(value.nodeLabel))] = [float(v) for v in value.data]
    if "RF" in frame.fieldOutputs:
        for value in frame.fieldOutputs["RF"].values:
            rf_values[(value.instance.name.upper(), int(value.nodeLabel))] = [float(v) for v in value.data]

    def dotn(values):
        return values[0] * normal[0] + values[1] * normal[1] + values[2] * normal[2]

    def mean_un(instance, labels):
        values = [dotn(u_values.get((instance, label), [0.0, 0.0, 0.0])) for label in labels]
        return sum(values) / float(len(values))

    def sum_rfn(instance, labels):
        return sum(dotn(rf_values.get((instance, label), [0.0, 0.0, 0.0])) for label in labels)

    lower_top_un = mean_un("LOWER-1", (5, 6, 7, 8))
    upper_bottom_un = mean_un("UPPER-1", (1, 2, 3, 4))
    upper_top_un = mean_un("UPPER-1", (5, 6, 7, 8))
    upper_top_rfn = sum_rfn("UPPER-1", (5, 6, 7, 8))
    lower_bottom_rfn = sum_rfn("LOWER-1", (1, 2, 3, 4))
    mean_gap = initial_gap + upper_bottom_un - lower_top_un
    copen_values = []
    cpress_values = []
    for key in frame.fieldOutputs.keys():
        if key.strip().startswith("COPEN"):
            field = frame.fieldOutputs[key]
            for idx in range(len(field.values)):
                copen_values.append(float(field.values[idx].data))
        if key.strip().startswith("CPRESS"):
            field = frame.fieldOutputs[key]
            for idx in range(len(field.values)):
                cpress_values.append(float(field.values[idx].data))
    copen_mean = sum(copen_values) / float(len(copen_values)) if copen_values else mean_gap
    cpress_mean = sum(cpress_values) / float(len(cpress_values)) if cpress_values else 0.0
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "upper_top_un_mean",
                "upper_bottom_un_mean",
                "lower_top_un_mean",
                "mean_gap",
                "upper_top_rfn_sum",
                "lower_bottom_rfn_sum",
                "copen_mean",
                "cpress_mean",
                "copen_count",
                "cpress_count",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "upper_top_un_mean": upper_top_un,
                "upper_bottom_un_mean": upper_bottom_un,
                "lower_top_un_mean": lower_top_un,
                "mean_gap": mean_gap,
                "upper_top_rfn_sum": upper_top_rfn,
                "lower_bottom_rfn_sum": lower_bottom_rfn,
                "copen_mean": copen_mean,
                "cpress_mean": cpress_mean,
                "copen_count": len(copen_values),
                "cpress_count": len(cpress_values),
            }
        )
finally:
    odb.close()
'''


def run_abaqus(case: ToothPatchHardContactCase, out_dir: Path, *, abaqus_command: str | None, timeout: int) -> tuple[Path, Row]:
    run_dir = out_dir / "abaqus_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    job = "lagrangian_sdf_tooth_patch_hard_contact"
    for old in run_dir.glob(f"{job}.*"):
        old.unlink()
    deck = run_dir / f"{job}.inp"
    write_abaqus_deck(case, deck)
    command = _resolve_abaqus_command(abaqus_command)
    analysis_wall = _run_command([command, f"job={job}", f"input={deck.name}", "interactive"], cwd=run_dir, log_path=out_dir / "abaqus_stdout.log", timeout=timeout)
    odb = run_dir / f"{job}.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    _lower, _upper, _elements, normal = make_geometry(case)
    script = run_dir / "export_tooth_patch_hard_contact.py"
    script.write_text(_abaqus_export_script(), encoding="ascii")
    metrics = out_dir / "abaqus_tooth_patch_hard_contact.csv"
    export_wall = _run_command(
        [
            command,
            "python",
            str(script.resolve()),
            str(odb.resolve()),
            str(metrics.resolve()),
            f"{case.initial_gap:.17g}",
            f"{normal[0]:.17g}",
            f"{normal[1]:.17g}",
            f"{normal[2]:.17g}",
        ],
        cwd=run_dir,
        log_path=out_dir / "abaqus_export_stdout.log",
        timeout=timeout,
    )
    return metrics, {"solver": "abaqus_surface_to_surface_hard_tooth_patch", "analysis_wall_seconds": analysis_wall, "export_wall_seconds": export_wall}


def _fixed_dofs(case: ToothPatchHardContactCase, *, upper_offset: int) -> tuple[np.ndarray, np.ndarray]:
    _lower, _upper, _elements, normal = make_geometry(case)
    closure_vec = -float(case.closure) * normal
    lower_bottom = np.asarray([0, 1, 2, 3], dtype=np.int64)
    upper_top = np.asarray([4, 5, 6, 7], dtype=np.int64)
    dofs: list[int] = []
    values: list[float] = []
    for node in lower_bottom:
        for axis in range(3):
            dofs.append(3 * int(node) + axis)
            values.append(0.0)
    for node in upper_top:
        for axis in range(3):
            dofs.append(int(upper_offset) + 3 * int(node) + axis)
            values.append(float(closure_vec[axis]))
    return np.asarray(dofs, dtype=np.int64), np.asarray(values, dtype=float)


def _pressure_enforcement_stiffness(case: ToothPatchHardContactCase) -> float:
    return float(case.pressure_smoothing_stiffness_factor) * float(case.young_modulus) / max(float(case.thickness), 1.0e-30)


def _normal_compliance(case: ToothPatchHardContactCase, samples: list[Any]) -> np.ndarray | None:
    mode = str(case.contact_enforcement).lower()
    if mode == "exact":
        return None
    if mode == "pressure_compliance":
        return hard_contact_pressure_compliance_from_samples(samples, pressure_stiffness=_pressure_enforcement_stiffness(case))
    raise ValueError("contact_enforcement must be 'exact' or 'pressure_compliance'")


def _sfc_contact_samples(case: ToothPatchHardContactCase, lower: np.ndarray, upper: np.ndarray) -> tuple[list[Any], np.ndarray, np.ndarray]:
    lower_top_quad = np.asarray([[4, 5, 6, 7]], dtype=np.int64)
    upper_bottom_quad = np.asarray([[0, 3, 2, 1]], dtype=np.int64)
    search_radius = max(case.initial_gap, case.closure) + 0.5 * case.thickness
    contact = LagrangianSDFQ4MasterSurfaceContactGeometry(
        lower_top_quad,
        upper_bottom_quad,
        upper,
        pressure_stiffness=1.0,
        master_node_offset=lower.shape[0],
        quadrature_order=int(case.quadrature_order),
        search_radius=search_radius,
    )
    x_reference = np.vstack((lower, upper))
    samples = list(contact.samples(x_reference))
    g0, J = hard_contact_gap_jacobian_from_samples(samples, n_total_dofs=3 * x_reference.shape[0])
    return samples, g0, J


def run_sfc(case: ToothPatchHardContactCase, out_path: Path, *, diagnostics_path: Path | None = None) -> tuple[Path, Row]:
    lower, upper, elements, normal = make_geometry(case)
    lower_body = DeformableBody(
        mesh=VolumeMesh(lower, elements, element_type="C3D8"),
        material={"E": case.young_modulus, "nu": case.poisson_ratio},
        density=case.density,
    )
    upper_body = DeformableBody(
        mesh=VolumeMesh(upper, elements, element_type="C3D8"),
        material={"E": case.young_modulus, "nu": case.poisson_ratio},
        density=case.density,
    )
    lower_dofs = lower_body.n_dofs
    K = block_diag((assemble_stiffness_matrix(lower_body).tocsr(), assemble_stiffness_matrix(upper_body).tocsr()), format="csr")
    f = np.zeros(K.shape[0], dtype=float)
    samples, g0, J = _sfc_contact_samples(case, lower, upper)
    normal_compliance = _normal_compliance(case, samples)
    fixed, values = _fixed_dofs(case, upper_offset=lower_dofs)
    start = time.perf_counter()
    solution = solve_linear_hard_contact_with_dirichlet(
        K,
        f,
        g0,
        J,
        fixed_dofs=fixed,
        fixed_values=values,
        normal_compliance=normal_compliance,
        tolerance=1.0e-9,
        max_iterations=30,
    )
    wall = time.perf_counter() - start
    u = solution.displacement.reshape((-1, 3))
    lower_u = u[: lower.shape[0]]
    upper_u = u[lower.shape[0] :]
    residual = np.asarray(K @ solution.displacement, dtype=float).ravel() - J.T @ solution.multipliers - f
    upper_top = np.asarray([4, 5, 6, 7], dtype=np.int64)
    upper_bottom = np.asarray([0, 1, 2, 3], dtype=np.int64)
    lower_top = np.asarray([4, 5, 6, 7], dtype=np.int64)
    lower_bottom = np.asarray([0, 1, 2, 3], dtype=np.int64)
    upper_top_dofs = np.asarray([lower_dofs + 3 * int(node) + axis for node in upper_top for axis in range(3)], dtype=np.int64)
    lower_bottom_dofs = np.asarray([3 * int(node) + axis for node in lower_bottom for axis in range(3)], dtype=np.int64)
    upper_top_rf = residual[upper_top_dofs].reshape((-1, 3))
    lower_bottom_rf = residual[lower_bottom_dofs].reshape((-1, 3))
    compliance = np.zeros(solution.gaps.shape[0], dtype=float) if normal_compliance is None else np.asarray(normal_compliance, dtype=float)
    enforcement_gaps = solution.gaps + compliance * solution.multipliers
    active_area = float(sum(float(sample.area) for sample, active in zip(samples, solution.active, strict=True) if bool(active)))
    active_multiplier_sum = float(np.sum(solution.multipliers[solution.active])) if solution.multipliers.size else 0.0
    upper_bottom_un = float(np.mean(upper_u[upper_bottom] @ normal))
    lower_top_un = float(np.mean(lower_u[lower_top] @ normal))
    row = {
        "contact_enforcement": str(case.contact_enforcement),
        "pressure_smoothing_stiffness": _pressure_enforcement_stiffness(case) if str(case.contact_enforcement).lower() == "pressure_compliance" else "",
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "normal_z": float(normal[2]),
        "upper_top_un_mean": float(np.mean(upper_u[upper_top] @ normal)),
        "upper_bottom_un_mean": upper_bottom_un,
        "lower_top_un_mean": lower_top_un,
        "mean_gap": float(case.initial_gap + upper_bottom_un - lower_top_un),
        "sample_mean_gap": float(np.mean(solution.gaps)) if solution.gaps.size else 0.0,
        "sample_min_gap": float(np.min(solution.gaps)) if solution.gaps.size else 0.0,
        "sample_max_gap": float(np.max(solution.gaps)) if solution.gaps.size else 0.0,
        "enforcement_gap_max_abs": float(np.max(np.abs(enforcement_gaps))) if enforcement_gaps.size else 0.0,
        "copen_mean": float(np.mean(solution.gaps)) if solution.gaps.size else 0.0,
        "cpress_mean": active_multiplier_sum / max(active_area, 1.0e-30),
        "upper_top_rfn_sum": float(np.sum(upper_top_rf @ normal)),
        "lower_bottom_rfn_sum": float(np.sum(lower_bottom_rf @ normal)),
        "active_constraints": int(np.count_nonzero(solution.active)),
        "constraints": int(solution.gaps.shape[0]),
        "iterations": int(solution.iterations),
        "converged": int(bool(solution.converged)),
        "quadrature_samples": int(len(samples)),
    }
    _write_csv(out_path, [row])
    if diagnostics_path is not None:
        _write_contact_diagnostics(diagnostics_path, samples, solution, J, normal, normal_compliance=normal_compliance)
    return out_path, {"solver": "sfc_lagrangian_sdf_q4_tooth_patch_hard", "analysis_wall_seconds": wall}


def _format_array(values: np.ndarray) -> str:
    arr = np.asarray(values).reshape(-1)
    return ";".join(f"{float(v):.12e}" for v in arr)


def _format_int_array(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    return ";".join(str(int(v)) for v in arr)


def _write_contact_diagnostics(
    path: Path,
    samples: list[Any],
    solution: Any,
    gap_jacobian: np.ndarray,
    normal: np.ndarray,
    *,
    normal_compliance: np.ndarray | None = None,
) -> None:
    rows: list[Row] = []
    compliance = np.zeros(len(samples), dtype=float) if normal_compliance is None else np.asarray(normal_compliance, dtype=float).reshape(-1)
    for idx, sample in enumerate(samples):
        row = np.asarray(gap_jacobian[int(idx)], dtype=float)
        multiplier = float(solution.multipliers[int(idx)])
        c = float(compliance[int(idx)])
        contact_force_row = -multiplier * row
        slave_nodes = np.asarray(sample.node_ids, dtype=np.int64).reshape(-1)
        master_nodes = np.asarray(sample.master_node_ids, dtype=np.int64).reshape(-1)
        slave_force = np.asarray([sum(float(contact_force_row[3 * int(node) + axis]) for node in slave_nodes) for axis in range(3)], dtype=float)
        master_force = np.asarray([sum(float(contact_force_row[3 * int(node) + axis]) for node in master_nodes) for axis in range(3)], dtype=float)
        rows.append(
            {
                "sample_id": int(idx),
                "active": int(bool(solution.active[int(idx)])),
                "initial_gap": float(sample.gap),
                "solved_gap": float(solution.gaps[int(idx)]),
                "normal_compliance": c,
                "enforcement_gap": float(solution.gaps[int(idx)] + c * multiplier),
                "multiplier": multiplier,
                "area": float(sample.area),
                "pressure_like_multiplier_over_area": multiplier / max(float(sample.area), 1.0e-30),
                "normal_x": float(sample.normal[0]),
                "normal_y": float(sample.normal[1]),
                "normal_z": float(sample.normal[2]),
                "slave_node_ids": _format_int_array(slave_nodes),
                "slave_weights": _format_array(np.asarray(sample.shape_weights, dtype=float)),
                "master_node_ids": _format_int_array(master_nodes),
                "master_weights": _format_array(np.asarray(sample.master_shape_weights, dtype=float)),
                "slave_force_normal": float(slave_force @ normal),
                "master_force_normal": float(master_force @ normal),
            }
        )
    _write_csv(path, rows)


def _read_first(path: Path) -> Row:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows[0]


def compare(abaqus_path: Path, sfc_path: Path, out_path: Path) -> list[Row]:
    abaqus = _read_first(abaqus_path)
    sfc = _read_first(sfc_path)
    metrics = [
        "upper_top_un_mean",
        "upper_bottom_un_mean",
        "lower_top_un_mean",
        "mean_gap",
        "copen_mean",
        "cpress_mean",
        "upper_top_rfn_sum",
        "lower_bottom_rfn_sum",
    ]
    rows: list[Row] = []
    for metric in metrics:
        a = float(abaqus.get(metric, 0.0))
        s = float(sfc.get(metric, 0.0))
        denom = max(abs(a), 1.0e-12)
        rows.append({"metric": metric, "abaqus": a, "sfc": s, "abs_error": abs(s - a), "rel_error": abs(s - a) / denom})
    _write_csv(out_path, rows)
    return rows


def write_summary(path: Path, rows: list[Row], abaqus_runtime: Row, sfc_runtime: Row) -> None:
    by_metric = {str(row["metric"]): row for row in rows}
    lines = [
        "# Lagrangian-SDF Tilted Tooth-Patch HARD Contact Alignment",
        "",
        "Two flexible C3D8 skew blocks are pressed together through a tilted Q4 contact patch.",
        "Abaqus uses frictionless `SURFACE TO SURFACE` with `pressure-overclosure=HARD`; SFC uses internal C3D8 stiffness, Q4 slave quadrature, Q4 master closest-feature payloads, and pressure-compliance hard-contact enforcement.",
        "",
        f"- Abaqus analysis wall time: {float(abaqus_runtime.get('analysis_wall_seconds', 0.0)):.6f} s",
        f"- SFC analysis wall time: {float(sfc_runtime.get('analysis_wall_seconds', 0.0)):.6e} s",
        f"- upper top normal-displacement relative error: {100.0 * float(by_metric['upper_top_un_mean']['rel_error']):.6f}%",
        f"- lower top normal-displacement relative error: {100.0 * float(by_metric['lower_top_un_mean']['rel_error']):.6f}%",
        f"- upper top normal-reaction relative error: {100.0 * float(by_metric['upper_top_rfn_sum']['rel_error']):.6f}%",
        f"- COPEN mean absolute error: {float(by_metric['copen_mean']['abs_error']):.6e}",
        f"- CPRESS mean relative error: {100.0 * float(by_metric['cpress_mean']['rel_error']):.6f}%",
        f"- mean hard-contact gap abs. error: {float(by_metric['mean_gap']['abs_error']):.6e}",
        "",
        "This case checks the same HARD-contact algebra on a non-axis-aligned tooth-like patch before transferring the implementation to gear-tooth contact.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--abaqus-command", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--contact-enforcement", choices=("exact", "pressure_compliance"), default="pressure_compliance")
    parser.add_argument("--pressure-smoothing-stiffness-factor", type=float, default=4.5)
    args = parser.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case = replace(
        build_case(),
        contact_enforcement=str(args.contact_enforcement),
        pressure_smoothing_stiffness_factor=float(args.pressure_smoothing_stiffness_factor),
    )
    abaqus_path, abaqus_runtime = run_abaqus(case, out_dir, abaqus_command=args.abaqus_command, timeout=int(args.timeout))
    sfc_path, sfc_runtime = run_sfc(case, out_dir / "sfc_lagrangian_sdf_tooth_patch_hard_contact.csv", diagnostics_path=out_dir / "sfc_contact_quadrature_diagnostics.csv")
    rows = compare(abaqus_path, sfc_path, out_dir / "abaqus_vs_sfc_lagrangian_sdf_tooth_patch_hard_contact_errors.csv")
    _write_csv(out_dir / "solver_runtime.csv", [abaqus_runtime, sfc_runtime])
    summary = out_dir / "lagrangian_sdf_tooth_patch_hard_contact_summary.md"
    write_summary(summary, rows, abaqus_runtime, sfc_runtime)
    print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
