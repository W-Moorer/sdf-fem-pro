"""Abaqus LINEAR pressure-overclosure vs SFC Lagrangian-SDF penalty contact.

This validation isolates contact enforcement from bulk deformation.  Two C3D8
blocks are used only to define matching element surfaces: the lower block is
fully fixed and the upper block is prescribed as a rigid vertical translation.
For a linear pressure-overclosure law, the expected contact force is therefore

``F_n = k_p A max(closure - initial_gap, 0)``.

Abaqus remains an external validation reference.  The SFC path computes the
same surface-to-surface contact with the Lagrangian-SDF Q4 payload and internal
penalty force integration, without importing Abaqus data into ``src/sfc``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sfc.contact.lagrangian_surface_contact import LagrangianSDFQ4MasterSurfaceContactGeometry  # noqa: E402

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "abaqus_linear_penalty_normal_indentation"


@dataclass(frozen=True, slots=True)
class LinearPenaltyIndentationCase:
    """Rigidly prescribed two-block normal indentation for LINEAR penalty law."""

    size: float = 1.0
    upper_size: float | None = None
    upper_offset: tuple[float, float] = (0.0, 0.0)
    height: float = 1.0
    initial_gap: float = 0.02
    pressure_stiffness: float = 5.0e9
    closures: tuple[float, ...] = (0.0, 0.01, 0.02, 0.025, 0.03)
    young_modulus: float = 1.0e7
    poisson_ratio: float = 0.30
    density: float = 1.0
    quadrature_order: int = 2


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


def build_case() -> LinearPenaltyIndentationCase:
    return LinearPenaltyIndentationCase()


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


def _translated(nodes: np.ndarray, offset: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(nodes, dtype=float) + np.asarray(offset, dtype=float)


def make_geometry(case: LinearPenaltyIndentationCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = _block_nodes(z0=0.0, size=case.size, height=case.height)
    upper_size = float(case.size if case.upper_size is None else case.upper_size)
    upper = _block_nodes(z0=case.height + case.initial_gap, size=upper_size, height=case.height)
    upper = _translated(upper, (float(case.upper_offset[0]), float(case.upper_offset[1]), 0.0))
    elements = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    return lower, upper, elements


def overlap_area(case: LinearPenaltyIndentationCase) -> float:
    upper_size = float(case.size if case.upper_size is None else case.upper_size)
    ox, oy = float(case.upper_offset[0]), float(case.upper_offset[1])
    x_overlap = max(0.0, min(float(case.size), ox + upper_size) - max(0.0, ox))
    y_overlap = max(0.0, min(float(case.size), oy + upper_size) - max(0.0, oy))
    return x_overlap * y_overlap


def write_abaqus_deck(case: LinearPenaltyIndentationCase, path: Path) -> None:
    """Write a multi-step Abaqus LINEAR pressure-overclosure deck."""

    lower, upper, elements = make_geometry(case)
    lines: list[str] = [
        "*Heading",
        "LINEAR pressure-overclosure normal indentation calibration",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
    ]
    for name, nodes in (("LOWER", lower), ("UPPER", upper)):
        lines.extend([f"*Part, name={name}", "*Node"])
        for label, xyz in enumerate(nodes, start=1):
            lines.append(f"{label}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
        lines.append("*Element, type=C3D8, elset=EALL")
        labels = [int(node) + 1 for node in elements[0]]
        lines.append("1, " + ", ".join(str(label) for label in labels))
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
            "*Nset, nset=LOWER_ALL_ASM, instance=LOWER-1",
            "1, 2, 3, 4, 5, 6, 7, 8",
            "*Nset, nset=UPPER_ALL_ASM, instance=UPPER-1",
            "1, 2, 3, 4, 5, 6, 7, 8",
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
            "*Surface Interaction, name=LINEAR_FRICTIONLESS",
            "*Surface Behavior, pressure-overclosure=LINEAR",
            f"{case.pressure_stiffness:.12e}",
            "*Friction",
            "0.",
            "*Contact Pair, interaction=LINEAR_FRICTIONLESS, type=SURFACE TO SURFACE",
            "LOWER_TOP_SURF, UPPER_BOTTOM_SURF",
        ]
    )
    for index, closure in enumerate(case.closures):
        lines.extend(
            [
                f"*Step, name=INDENT_{index:02d}, nlgeom=NO, inc=50",
                "*Static",
                "1., 1., 1e-08, 1.",
                "*Boundary, OP=NEW",
                "LOWER_ALL_ASM, 1, 3, 0.",
                "UPPER_ALL_ASM, 1, 2, 0.",
                f"UPPER_ALL_ASM, 3, 3, {-float(closure):.12e}",
                "*Output, field, frequency=1",
                "*Node Output",
                "U, RF",
                "*Contact Output",
                "CSTRESS, CDISP",
                "*End Step",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


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
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout,
    )
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
closures = [float(value) for value in sys.argv[3].split(",") if value]
initial_gap = float(sys.argv[4])
area = float(sys.argv[5])
pressure_stiffness = float(sys.argv[6])
try:
    rows = []
    for index, step_name in enumerate(odb.steps.keys()):
        step = odb.steps[step_name]
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
        closure = closures[index] if index < len(closures) else -mean_u3("UPPER-1", (1, 2, 3, 4, 5, 6, 7, 8))
        mean_gap = initial_gap + upper_bottom_u3 - lower_top_u3
        penetration = max(-mean_gap, 0.0)
        upper_rf3 = sum(rf_values.get(("UPPER-1", label), [0.0, 0.0, 0.0])[2] for label in (1, 2, 3, 4, 5, 6, 7, 8))
        lower_rf3 = sum(rf_values.get(("LOWER-1", label), [0.0, 0.0, 0.0])[2] for label in (1, 2, 3, 4, 5, 6, 7, 8))
        copen_values = []
        cpress_values = []
        for key in frame.fieldOutputs.keys():
            if key.strip().startswith("COPEN"):
                for value in frame.fieldOutputs[key].values:
                    copen_values.append(float(value.data))
            if key.strip().startswith("CPRESS"):
                for value in frame.fieldOutputs[key].values:
                    cpress_values.append(float(value.data))
        copen_mean = sum(copen_values) / float(len(copen_values)) if copen_values else mean_gap
        cpress_mean = sum(cpress_values) / float(len(cpress_values)) if cpress_values else 0.0
        expected_pressure = pressure_stiffness * penetration
        expected_force = expected_pressure * area
        rows.append(
            {
                "step": index,
                "step_name": step_name,
                "closure": closure,
                "mean_gap": mean_gap,
                "penetration": penetration,
                "upper_rf3_sum": upper_rf3,
                "lower_rf3_sum": lower_rf3,
                "normal_force_magnitude": abs(upper_rf3),
                "copen_mean": copen_mean,
                "cpress_mean": cpress_mean,
                "copen_count": len(copen_values),
                "cpress_count": len(cpress_values),
                "expected_pressure": expected_pressure,
                "expected_force": expected_force,
            }
        )
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
finally:
    odb.close()
'''


def run_abaqus(case: LinearPenaltyIndentationCase, out_dir: Path, *, abaqus_command: str | None, timeout: int) -> tuple[Path, Row]:
    run_dir = out_dir / "abaqus_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    job = "linear_penalty_normal_indentation"
    for old in run_dir.glob(f"{job}.*"):
        old.unlink()
    deck = run_dir / f"{job}.inp"
    write_abaqus_deck(case, deck)
    command = _resolve_abaqus_command(abaqus_command)
    analysis_wall = _run_command(
        [command, f"job={job}", f"input={deck.name}", "interactive"],
        cwd=run_dir,
        log_path=out_dir / "abaqus_stdout.log",
        timeout=timeout,
    )
    odb = run_dir / f"{job}.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    script = run_dir / "export_linear_penalty_normal_indentation.py"
    script.write_text(_abaqus_export_script(), encoding="ascii")
    metrics = out_dir / "abaqus_linear_penalty_normal_indentation.csv"
    export_wall = _run_command(
        [
            command,
            "python",
            str(script.resolve()),
            str(odb.resolve()),
            str(metrics.resolve()),
            ",".join(f"{float(v):.17g}" for v in case.closures),
            f"{case.initial_gap:.17g}",
            f"{overlap_area(case):.17g}",
            f"{case.pressure_stiffness:.17g}",
        ],
        cwd=run_dir,
        log_path=out_dir / "abaqus_export_stdout.log",
        timeout=timeout,
    )
    return metrics, {
        "solver": "abaqus_surface_to_surface_linear_penalty",
        "analysis_wall_seconds": analysis_wall,
        "export_wall_seconds": export_wall,
    }


def run_sfc(case: LinearPenaltyIndentationCase, out_path: Path) -> tuple[Path, Row]:
    lower, upper0, _elements = make_geometry(case)
    lower_top_quad = np.asarray([[4, 5, 6, 7]], dtype=np.int64)
    upper_bottom_quad = np.asarray([[0, 3, 2, 1]], dtype=np.int64)
    rows: list[Row] = []
    start = time.perf_counter()
    for step, closure in enumerate(case.closures):
        upper = upper0.copy()
        upper[:, 2] -= float(closure)
        x_current = np.vstack((lower, upper))
        contact = LagrangianSDFQ4MasterSurfaceContactGeometry(
            lower_top_quad,
            upper_bottom_quad,
            upper,
            pressure_stiffness=float(case.pressure_stiffness),
            master_node_offset=lower.shape[0],
            quadrature_order=int(case.quadrature_order),
            search_radius=max(float(case.height), float(case.initial_gap), float(closure)) + 1.0e-12,
            clip_to_master_footprint=True,
        )
        samples = list(contact.samples(x_current))
        force = np.zeros((x_current.shape[0], 3), dtype=float)
        normal_force = 0.0
        area = 0.0
        weighted_gap = 0.0
        active_count = 0
        for sample in samples:
            gap = float(sample.gap)
            sample_area = float(sample.area)
            area += sample_area
            weighted_gap += sample_area * gap
            penetration = max(-gap, 0.0)
            if penetration <= 0.0:
                continue
            active_count += 1
            lam = float(case.pressure_stiffness) * sample_area * penetration
            normal = np.asarray(sample.normal, dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1.0e-30)
            normal_force += lam
            for node, weight in zip(sample.node_ids, sample.shape_weights, strict=True):
                force[int(node)] += float(weight) * lam * normal
            if sample.master_node_ids is not None and sample.master_shape_weights is not None:
                for node, weight in zip(sample.master_node_ids, sample.master_shape_weights, strict=True):
                    force[int(node)] -= float(weight) * lam * normal
        mean_gap = weighted_gap / max(area, 1.0e-30)
        penetration = max(-mean_gap, 0.0)
        expected_pressure = float(case.pressure_stiffness) * penetration
        expected_force = expected_pressure * overlap_area(case)
        rows.append(
            {
                "step": int(step),
                "closure": float(closure),
                "mean_gap": float(mean_gap),
                "penetration": float(penetration),
                "normal_force_magnitude": float(normal_force),
                "upper_rf3_sum": float(np.sum(force[lower.shape[0] :, 2])),
                "lower_rf3_sum": float(np.sum(force[: lower.shape[0], 2])),
                "copen_mean": float(mean_gap),
                "cpress_mean": float(normal_force / max(area, 1.0e-30)),
                "expected_pressure": float(expected_pressure),
                "expected_force": float(expected_force),
                "overlap_area": float(overlap_area(case)),
                "active_constraints": int(active_count),
                "constraints": int(len(samples)),
            }
        )
    wall = time.perf_counter() - start
    _write_csv(out_path, rows)
    return out_path, {"solver": "sfc_lagrangian_sdf_linear_penalty", "analysis_wall_seconds": wall}


def _read_rows(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rel_error(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(b)), 1.0)


def compare(abaqus_path: Path, sfc_path: Path, out_path: Path) -> list[Row]:
    abaqus_rows = _read_rows(abaqus_path)
    sfc_rows = _read_rows(sfc_path)
    sfc_by_step = {int(row["step"]): row for row in sfc_rows}
    rows: list[Row] = []
    for abaqus in abaqus_rows:
        step = int(abaqus["step"])
        sfc = sfc_by_step[step]
        for metric in ("mean_gap", "penetration", "normal_force_magnitude", "cpress_mean"):
            rows.append(
                {
                    "step": step,
                    "closure": float(abaqus["closure"]),
                    "metric": metric,
                    "abaqus": float(abaqus[metric]),
                    "sfc": float(sfc[metric]),
                    "abs_error": abs(float(sfc[metric]) - float(abaqus[metric])),
                    "rel_error": _rel_error(float(sfc[metric]), float(abaqus[metric])),
                }
            )
    _write_csv(out_path, rows)
    return rows


def write_summary(path: Path, *, case: LinearPenaltyIndentationCase, sfc_runtime: Row, abaqus_runtime: Row | None, errors: list[Row] | None) -> None:
    lines = [
        "# LINEAR Penalty Normal Indentation Alignment",
        "",
        "This case isolates normal pressure-overclosure enforcement from bulk deformation.",
        "",
        f"- pressure stiffness: `{case.pressure_stiffness:.6e}`",
        f"- initial gap: `{case.initial_gap:.6e}`",
        f"- lower area: `{case.size * case.size:.6e}`",
        f"- upper area: `{float(case.size if case.upper_size is None else case.upper_size) ** 2:.6e}`",
        f"- overlap area: `{overlap_area(case):.6e}`",
        f"- upper offset: `({float(case.upper_offset[0]):.6e}, {float(case.upper_offset[1]):.6e})`",
        f"- closures: `{', '.join(f'{float(v):.6e}' for v in case.closures)}`",
        f"- SFC wall time: `{float(sfc_runtime['analysis_wall_seconds']):.6f} s`",
    ]
    if abaqus_runtime is not None:
        lines.append(f"- Abaqus analysis wall time: `{float(abaqus_runtime['analysis_wall_seconds']):.6f} s`")
        lines.append(f"- Abaqus export wall time: `{float(abaqus_runtime['export_wall_seconds']):.6f} s`")
    if errors:
        force_errors = [float(row["rel_error"]) for row in errors if row["metric"] == "normal_force_magnitude"]
        pressure_errors = [float(row["rel_error"]) for row in errors if row["metric"] == "cpress_mean"]
        lines.extend(
            [
                "",
                "## Alignment",
                "",
                f"- max RF/normal-force relative error: `{100.0 * max(force_errors, default=0.0):.6f}%`",
                f"- max CPRESS output-mean relative difference: `{100.0 * max(pressure_errors, default=0.0):.6f}%`",
                "",
                "CPRESS is an Abaqus contact-output field and may include nodal or constraint-region averaging; RF is the primary equilibrium check for this calibration.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-abaqus", action="store_true")
    parser.add_argument("--abaqus-command", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--pressure-stiffness", type=float, default=5.0e9)
    parser.add_argument("--initial-gap", type=float, default=0.02)
    parser.add_argument("--closures", type=str, default="0,0.01,0.02,0.025,0.03")
    parser.add_argument("--upper-size", type=float, default=None)
    parser.add_argument("--upper-offset-x", type=float, default=0.0)
    parser.add_argument("--upper-offset-y", type=float, default=0.0)
    args = parser.parse_args(argv)
    closures = tuple(float(part) for part in str(args.closures).split(",") if part.strip())
    case = LinearPenaltyIndentationCase(
        pressure_stiffness=float(args.pressure_stiffness),
        initial_gap=float(args.initial_gap),
        closures=closures,
        upper_size=args.upper_size,
        upper_offset=(float(args.upper_offset_x), float(args.upper_offset_y)),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_abaqus_deck(case, args.out_dir / "linear_penalty_normal_indentation.inp")
    sfc_path, sfc_runtime = run_sfc(case, args.out_dir / "sfc_linear_penalty_normal_indentation.csv")
    abaqus_runtime: Row | None = None
    errors: list[Row] | None = None
    if bool(args.run_abaqus):
        abaqus_path, abaqus_runtime = run_abaqus(case, args.out_dir, abaqus_command=args.abaqus_command, timeout=int(args.timeout))
        errors = compare(abaqus_path, sfc_path, args.out_dir / "abaqus_vs_sfc_linear_penalty_normal_indentation_errors.csv")
    write_summary(
        args.out_dir / "linear_penalty_normal_indentation_summary.md",
        case=case,
        sfc_runtime=sfc_runtime,
        abaqus_runtime=abaqus_runtime,
        errors=errors,
    )
    print((args.out_dir / "linear_penalty_normal_indentation_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
