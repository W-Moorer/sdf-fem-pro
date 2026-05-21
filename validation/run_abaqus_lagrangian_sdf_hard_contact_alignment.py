"""Abaqus surface-to-surface HARD contact vs SFC Lagrangian-SDF HARD contact.

The case is the first flexible-body alignment layer before gear-tooth contact:
two C3D8 flexible blocks are pressed together in the normal direction. Abaqus
uses frictionless ``SURFACE TO SURFACE`` contact with
``pressure-overclosure=HARD``. SFC assembles both C3D8 stiffness matrices
internally and solves the corresponding Lagrangian-SDF surface quadrature hard
contact KKT problem.

Abaqus is an external validation reference only; no Abaqus module is imported
by the core SFC package.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact.hard_contact import hard_contact_gap_jacobian_from_samples, solve_linear_hard_contact_with_dirichlet  # noqa: E402
from sfc.contact.lagrangian_surface_contact import LagrangianSDFQ4MasterSurfaceContactGeometry, LagrangianSDFQuadrilateralSurfaceContactGeometry  # noqa: E402
from sfc.fem import DeformableBody  # noqa: E402
from sfc.fem.assembler import assemble_stiffness_matrix  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.sdf.material_sdf import MaterialSDF  # noqa: E402

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "abaqus_lagrangian_sdf_hard_contact_alignment"


@dataclass(frozen=True, slots=True)
class FlexibleBlockHardContactCase:
    """Two flexible C3D8 blocks pressed in normal HARD contact."""

    size: float = 1.0
    height: float = 1.0
    initial_gap: float = 0.02
    closure: float = 0.05
    young_modulus: float = 10000.0
    poisson_ratio: float = 0.30
    density: float = 1.0
    quadrature_order: int = 2
    master_payload_mode: str = "q4"


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


def build_case() -> FlexibleBlockHardContactCase:
    return FlexibleBlockHardContactCase()


def _block_nodes(*, z0: float, size: float, height: float) -> np.ndarray:
    s = float(size)
    h = float(height)
    return np.asarray(
        [
            [0.0, 0.0, z0],
            [s, 0.0, z0],
            [s, s, z0],
            [0.0, s, z0],
            [0.0, 0.0, z0 + h],
            [s, 0.0, z0 + h],
            [s, s, z0 + h],
            [0.0, s, z0 + h],
        ],
        dtype=float,
    )


def make_geometry(case: FlexibleBlockHardContactCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = _block_nodes(z0=0.0, size=case.size, height=case.height)
    upper = _block_nodes(z0=case.height + case.initial_gap, size=case.size, height=case.height)
    elements = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    return lower, upper, elements


def write_abaqus_deck(case: FlexibleBlockHardContactCase, path: Path) -> None:
    """Write the external Abaqus reference deck."""

    lower, upper, elements = make_geometry(case)
    lines: list[str] = [
        "*Heading",
        "Two-flexible-block surface-to-surface HARD contact alignment",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
    ]
    for name, nodes in (("LOWER", lower), ("UPPER", upper)):
        lines.extend([f"*Part, name={name}", "*Node"])
        for label, xyz in enumerate(nodes, start=1):
            lines.append(f"{label}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
        lines.append("*Element, type=C3D8, elset=EALL")
        for element_id, element in enumerate(elements, start=1):
            labels = [int(node) + 1 for node in element]
            lines.append(f"{element_id}, " + ", ".join(str(label) for label in labels))
        mat = "LOWER_MAT" if name == "LOWER" else "UPPER_MAT"
        lines.extend([f"*Solid Section, elset=EALL, material={mat}", ",", "*End Part"])
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
            "*Step, name=FLEX_BLOCK_HARD_PRESS, nlgeom=YES, inc=200",
            "*Static",
            "1., 1., 1e-08, 1.",
            "*Boundary",
            "LOWER_BOTTOM_ASM, 1, 3, 0.",
            "UPPER_TOP_ASM, 1, 2, 0.",
            f"UPPER_TOP_ASM, 3, 3, {-case.closure:.12e}",
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
closure = float(sys.argv[4])
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

    def mean_u3(instance, labels):
        values = [u_values.get((instance, label), [0.0, 0.0, 0.0])[2] for label in labels]
        return sum(values) / float(len(values))

    lower_top_u3 = mean_u3("LOWER-1", (5, 6, 7, 8))
    upper_bottom_u3 = mean_u3("UPPER-1", (1, 2, 3, 4))
    upper_top_u3 = mean_u3("UPPER-1", (5, 6, 7, 8))
    upper_top_rf3 = sum(rf_values.get(("UPPER-1", label), [0.0, 0.0, 0.0])[2] for label in (5, 6, 7, 8))
    lower_bottom_rf3 = sum(rf_values.get(("LOWER-1", label), [0.0, 0.0, 0.0])[2] for label in (1, 2, 3, 4))
    mean_gap = initial_gap + upper_bottom_u3 - lower_top_u3
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "upper_top_u3_mean",
                "upper_bottom_u3_mean",
                "lower_top_u3_mean",
                "mean_gap",
                "upper_top_rf3_sum",
                "lower_bottom_rf3_sum",
                "closure",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "upper_top_u3_mean": upper_top_u3,
                "upper_bottom_u3_mean": upper_bottom_u3,
                "lower_top_u3_mean": lower_top_u3,
                "mean_gap": mean_gap,
                "upper_top_rf3_sum": upper_top_rf3,
                "lower_bottom_rf3_sum": lower_bottom_rf3,
                "closure": closure,
            }
        )
finally:
    odb.close()
'''


def run_abaqus(case: FlexibleBlockHardContactCase, out_dir: Path, *, abaqus_command: str | None, timeout: int) -> tuple[Path, Row]:
    run_dir = out_dir / "abaqus_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    job = "lagrangian_sdf_hard_contact_blocks"
    for old in run_dir.glob(f"{job}.*"):
        old.unlink()
    deck = run_dir / f"{job}.inp"
    write_abaqus_deck(case, deck)
    command = _resolve_abaqus_command(abaqus_command)
    analysis_wall = _run_command([command, f"job={job}", f"input={deck.name}", "interactive"], cwd=run_dir, log_path=out_dir / "abaqus_stdout.log", timeout=timeout)
    odb = run_dir / f"{job}.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    script = run_dir / "export_flexible_block_hard_contact.py"
    script.write_text(_abaqus_export_script(), encoding="ascii")
    metrics = out_dir / "abaqus_flexible_block_hard_contact.csv"
    export_wall = _run_command(
        [command, "python", str(script.resolve()), str(odb.resolve()), str(metrics.resolve()), f"{case.initial_gap:.17g}", f"{case.closure:.17g}"],
        cwd=run_dir,
        log_path=out_dir / "abaqus_export_stdout.log",
        timeout=timeout,
    )
    return metrics, {"solver": "abaqus_surface_to_surface_hard", "analysis_wall_seconds": analysis_wall, "export_wall_seconds": export_wall}


def _fixed_dofs(case: FlexibleBlockHardContactCase, *, upper_offset: int) -> tuple[np.ndarray, np.ndarray]:
    lower_bottom = np.asarray([0, 1, 2, 3], dtype=np.int64)
    upper_top = np.asarray([4, 5, 6, 7], dtype=np.int64)
    dofs: list[int] = []
    values: list[float] = []
    for node in lower_bottom:
        for axis in range(3):
            dofs.append(3 * int(node) + axis)
            values.append(0.0)
    for node in upper_top:
        dofs.append(int(upper_offset) + 3 * int(node) + 0)
        values.append(0.0)
        dofs.append(int(upper_offset) + 3 * int(node) + 1)
        values.append(0.0)
        dofs.append(int(upper_offset) + 3 * int(node) + 2)
        values.append(-float(case.closure))
    return np.asarray(dofs, dtype=np.int64), np.asarray(values, dtype=float)


def _sfc_contact_samples(case: FlexibleBlockHardContactCase, lower: np.ndarray, upper: np.ndarray) -> tuple[list[Any], np.ndarray, np.ndarray]:
    lower_top_quad = np.asarray([[4, 5, 6, 7]], dtype=np.int64)
    mode = str(case.master_payload_mode).lower()
    search_radius = max(case.initial_gap, case.closure) + case.height
    if mode == "q4":
        upper_bottom_quads = np.asarray([[0, 3, 2, 1]], dtype=np.int64)
        contact = LagrangianSDFQ4MasterSurfaceContactGeometry(
            lower_top_quad,
            upper_bottom_quads,
            upper,
            pressure_stiffness=1.0,
            master_node_offset=lower.shape[0],
            quadrature_order=int(case.quadrature_order),
            search_radius=search_radius,
        )
    elif mode == "triangulated":
        upper_bottom_faces = np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
        material = MaterialSDF.from_triangle_surface(upper, upper_bottom_faces, band_radius=max(case.initial_gap, case.closure))
        contact = LagrangianSDFQuadrilateralSurfaceContactGeometry(
            lower_top_quad,
            material,
            upper,
            pressure_stiffness=1.0,
            master_node_offset=lower.shape[0],
            quadrature_order=int(case.quadrature_order),
            search_radius=search_radius,
        )
    else:
        raise ValueError("master_payload_mode must be 'q4' or 'triangulated'")
    x_reference = np.vstack((lower, upper))
    samples = list(contact.samples(x_reference))
    g0, J = hard_contact_gap_jacobian_from_samples(samples, n_total_dofs=3 * x_reference.shape[0])
    return samples, g0, J


def run_sfc(case: FlexibleBlockHardContactCase, out_path: Path, *, diagnostics_path: Path | None = None) -> tuple[Path, Row]:
    lower, upper, elements = make_geometry(case)
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
    fixed, values = _fixed_dofs(case, upper_offset=lower_dofs)
    start = time.perf_counter()
    solution = solve_linear_hard_contact_with_dirichlet(
        K,
        f,
        g0,
        J,
        fixed_dofs=fixed,
        fixed_values=values,
        tolerance=1.0e-9,
        max_iterations=30,
    )
    wall = time.perf_counter() - start
    u = solution.displacement.reshape((-1, 3))
    lower_u = u[: lower.shape[0]]
    upper_u = u[lower.shape[0] :]
    residual = np.asarray(K @ solution.displacement, dtype=float).ravel() - J.T @ solution.multipliers - f
    upper_top = np.asarray([4, 5, 6, 7], dtype=np.int64)
    lower_bottom = np.asarray([0, 1, 2, 3], dtype=np.int64)
    upper_top_z_dofs = lower_dofs + 3 * upper_top + 2
    lower_bottom_z_dofs = 3 * lower_bottom + 2
    row = {
        "master_payload_mode": str(case.master_payload_mode),
        "upper_top_u3_mean": float(np.mean(upper_u[upper_top, 2])),
        "upper_bottom_u3_mean": float(np.mean(upper_u[[0, 1, 2, 3], 2])),
        "lower_top_u3_mean": float(np.mean(lower_u[[4, 5, 6, 7], 2])),
        "mean_gap": float(case.initial_gap + np.mean(upper_u[[0, 1, 2, 3], 2]) - np.mean(lower_u[[4, 5, 6, 7], 2])),
        "sample_mean_gap": float(np.mean(solution.gaps)) if solution.gaps.size else 0.0,
        "sample_min_gap": float(np.min(solution.gaps)) if solution.gaps.size else 0.0,
        "sample_max_gap": float(np.max(solution.gaps)) if solution.gaps.size else 0.0,
        "upper_top_rf3_sum": float(np.sum(residual[upper_top_z_dofs])),
        "lower_bottom_rf3_sum": float(np.sum(residual[lower_bottom_z_dofs])),
        "active_constraints": int(np.count_nonzero(solution.active)),
        "constraints": int(solution.gaps.shape[0]),
        "iterations": int(solution.iterations),
        "converged": int(bool(solution.converged)),
        "quadrature_samples": int(len(samples)),
    }
    _write_csv(out_path, [row])
    if diagnostics_path is not None:
        _write_contact_diagnostics(diagnostics_path, samples, solution, J)
    return out_path, {"solver": "sfc_lagrangian_sdf_surface_hard", "analysis_wall_seconds": wall}


def _format_array(values: np.ndarray) -> str:
    arr = np.asarray(values).reshape(-1)
    return ";".join(f"{float(v):.12e}" for v in arr)


def _format_int_array(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    return ";".join(str(int(v)) for v in arr)


def _write_contact_diagnostics(path: Path, samples: list[Any], solution: Any, gap_jacobian: np.ndarray) -> None:
    rows: list[Row] = []
    for idx, sample in enumerate(samples):
        row = np.asarray(gap_jacobian[int(idx)], dtype=float)
        multiplier = float(solution.multipliers[int(idx)])
        contact_force_row = -multiplier * row
        slave_nodes = np.asarray(sample.node_ids, dtype=np.int64).reshape(-1)
        master_nodes = np.asarray(sample.master_node_ids, dtype=np.int64).reshape(-1)
        slave_z = sum(float(contact_force_row[3 * int(node) + 2]) for node in slave_nodes)
        master_z = sum(float(contact_force_row[3 * int(node) + 2]) for node in master_nodes)
        rows.append(
            {
                "sample_id": int(idx),
                "active": int(bool(solution.active[int(idx)])),
                "initial_gap": float(sample.gap),
                "solved_gap": float(solution.gaps[int(idx)]),
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
                "slave_force_z_from_row": float(slave_z),
                "master_force_z_from_row": float(master_z),
            }
        )
    _write_csv(path, rows)


def run_sfc_convergence_diagnostics(case: FlexibleBlockHardContactCase, abaqus_path: Path, out_dir: Path) -> list[Row]:
    abaqus = _read_first(abaqus_path)
    rows: list[Row] = []
    for mode in ("triangulated", "q4"):
        for order in (1, 2, 3):
            candidate = replace(case, master_payload_mode=mode, quadrature_order=order)
            result_path, runtime = run_sfc(candidate, out_dir / f"sfc_{mode}_q{order}.csv")
            row = _read_first(result_path)
            rf_a = float(abaqus["upper_top_rf3_sum"])
            rf_s = float(row["upper_top_rf3_sum"])
            gap_a = float(abaqus["mean_gap"])
            gap_s = float(row["mean_gap"])
            rows.append(
                {
                    "master_payload_mode": mode,
                    "quadrature_order": int(order),
                    "constraints": int(row["constraints"]),
                    "active_constraints": int(row["active_constraints"]),
                    "upper_top_rf3_abaqus": rf_a,
                    "upper_top_rf3_sfc": rf_s,
                    "upper_top_rf3_rel_error": abs(rf_s - rf_a) / max(abs(rf_a), 1.0e-12),
                    "mean_gap_abaqus": gap_a,
                    "mean_gap_sfc": gap_s,
                    "mean_gap_abs_error": abs(gap_s - gap_a),
                    "sample_min_gap": float(row["sample_min_gap"]),
                    "sample_max_gap": float(row["sample_max_gap"]),
                    "sfc_wall_seconds": float(runtime["analysis_wall_seconds"]),
                }
            )
    _write_csv(out_dir / "sfc_master_payload_convergence.csv", rows)
    return rows


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
        "upper_top_u3_mean",
        "upper_bottom_u3_mean",
        "lower_top_u3_mean",
        "mean_gap",
        "upper_top_rf3_sum",
        "lower_bottom_rf3_sum",
    ]
    rows: list[Row] = []
    for metric in metrics:
        a = float(abaqus.get(metric, 0.0))
        s = float(sfc.get(metric, 0.0))
        denom = max(abs(a), 1.0e-12)
        rows.append(
            {
                "metric": metric,
                "abaqus": a,
                "sfc": s,
                "abs_error": abs(s - a),
                "rel_error": abs(s - a) / denom,
            }
        )
    _write_csv(out_path, rows)
    return rows


def write_summary(path: Path, rows: list[Row], abaqus_runtime: Row, sfc_runtime: Row) -> None:
    by_metric = {str(row["metric"]): row for row in rows}
    lines = [
        "# Lagrangian-SDF Surface-to-Surface HARD Contact Alignment",
        "",
        "Two flexible C3D8 blocks are pressed together in the normal direction.",
        "Abaqus uses frictionless `SURFACE TO SURFACE` with `pressure-overclosure=HARD`; SFC uses internal C3D8 stiffness, Q4 slave quadrature, and Q4 master closest-feature payload hard contact.",
        "",
        f"- Abaqus analysis wall time: {float(abaqus_runtime.get('analysis_wall_seconds', 0.0)):.6f} s",
        f"- SFC analysis wall time: {float(sfc_runtime.get('analysis_wall_seconds', 0.0)):.6e} s",
        f"- upper top U3 relative error: {100.0 * float(by_metric['upper_top_u3_mean']['rel_error']):.6f}%",
        f"- lower top U3 relative error: {100.0 * float(by_metric['lower_top_u3_mean']['rel_error']):.6f}%",
        f"- upper top RF3 relative error: {100.0 * float(by_metric['upper_top_rf3_sum']['rel_error']):.6f}%",
        f"- mean hard-contact gap abs. error: {float(by_metric['mean_gap']['abs_error']):.6e}",
        "- Diagnostic outputs: `sfc_contact_quadrature_diagnostics.csv` and `sfc_master_payload_convergence.csv`.",
        "",
        "This validates the HARD normal-contact algebra and flexible-body boundary semantics before applying the same layer to gear-tooth contact.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--abaqus-command", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--master-payload-mode", choices=("q4", "triangulated"), default="q4")
    args = parser.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case = replace(build_case(), master_payload_mode=str(args.master_payload_mode))
    abaqus_path, abaqus_runtime = run_abaqus(case, out_dir, abaqus_command=args.abaqus_command, timeout=int(args.timeout))
    sfc_path, sfc_runtime = run_sfc(
        case,
        out_dir / "sfc_lagrangian_sdf_hard_contact.csv",
        diagnostics_path=out_dir / "sfc_contact_quadrature_diagnostics.csv",
    )
    rows = compare(abaqus_path, sfc_path, out_dir / "abaqus_vs_sfc_lagrangian_sdf_hard_contact_errors.csv")
    run_sfc_convergence_diagnostics(case, abaqus_path, out_dir)
    _write_csv(out_dir / "solver_runtime.csv", [abaqus_runtime, sfc_runtime])
    summary = out_dir / "lagrangian_sdf_hard_contact_summary.md"
    write_summary(summary, rows, abaqus_runtime, sfc_runtime)
    print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
