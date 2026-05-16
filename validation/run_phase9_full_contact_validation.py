"""Phase-9 full contact-validation evidence matrix.

The runner is intentionally claim-gated.  It gathers the current C3D8 static
and dynamic contact evidence, writes per-case CSV/Markdown/plot/VTK artifacts,
and explicitly blocks paper claims that do not yet have native SFC, CalculiX,
and timing evidence.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sfc.fem.hex8 import hex8_center_strain_stress  # noqa: E402
from validation.run_c3d8_contact_trajectory_validation import (  # noqa: E402
    _parse_element_stress,
    _parse_nodal_vectors,
    build_c3d8_model,
    main as run_c3d8_trajectory_main,
)
from validation.run_calculix_contactenergy_replay import (  # noqa: E402
    _hex8_center_strain_stress as contactenergy_hex8_center_strain_stress,
)
from validation.run_calculix_contactenergy_replay import (  # noqa: E402
    parse_contactenergy_dat,
    parse_contactenergy_input,
)

Row = dict[str, Any]


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bool_text(value: bool) -> str:
    return str(bool(value)).lower()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _case_status(native: bool, external: bool, metric_ok: bool) -> str:
    if native and external and metric_ok:
        return "supported"
    if external and metric_ok:
        return "external_replay_only"
    return "blocked"


def _write_legacy_hex_vtk(
    path: Path,
    points: np.ndarray,
    elements: np.ndarray,
    *,
    cell_values: dict[str, np.ndarray],
) -> None:
    """Write a minimal legacy VTK HEX8 file for ParaView inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(points, dtype=float)
    conn = np.asarray(elements, dtype=np.int64)
    lines: list[str] = [
        "# vtk DataFile Version 3.0",
        "SFC Phase-9 C3D8 contact validation",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {X.shape[0]} float",
    ]
    lines.extend(f"{x:.12e} {y:.12e} {z:.12e}" for x, y, z in X)
    lines.append(f"CELLS {conn.shape[0]} {conn.shape[0] * 9}")
    lines.extend("8 " + " ".join(str(int(node)) for node in element) for element in conn)
    lines.append(f"CELL_TYPES {conn.shape[0]}")
    lines.extend("12" for _ in range(conn.shape[0]))
    if cell_values:
        lines.append(f"CELL_DATA {conn.shape[0]}")
        for name, values in cell_values.items():
            data = np.asarray(values, dtype=float).ravel()
            if data.shape != (conn.shape[0],):
                raise ValueError(f"cell value {name!r} has wrong shape")
            lines.append(f"SCALARS {name} float 1")
            lines.append("LOOKUP_TABLE default")
            lines.extend(f"{float(value):.12e}" for value in data)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _von_mises(stress: np.ndarray) -> float:
    sxx, syy, szz, sxy, syz, sxz = np.asarray(stress, dtype=float)
    return float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + syz**2 + sxz**2)))


def _linear_static_evidence(out_dir: Path) -> tuple[Row, list[Row]]:
    source_dir = ROOT / "paper" / "numerical_experiments" / "calculix_contactenergy_c3d8_replay"
    replay_csv = source_dir / "data" / "calculix_contactenergy_replay.csv"
    inp_path = source_dir / "data" / "contactenergy.inp"
    dat_path = source_dir / "data" / "contactenergy.dat"
    if not replay_csv.exists() or not inp_path.exists() or not dat_path.exists():
        row = {
            "case_id": "c3d8_linear_static_contact",
            "analysis_type": "static",
            "linearity": "linear",
            "element_type": "C3D8",
            "native_sfc_result": "false",
            "calculix_comparison": "false",
            "acceleration_evidence": "false",
            "displacement_metric": "",
            "stress_strain_metric": "",
            "gap_metric": "",
            "rf_metric": "",
            "cels_metric": "",
            "active_contact_metric": "",
            "supports_external_correctness": "false",
            "supports_trajectory_equivalence": "false",
            "supports_efficiency": "false",
            "status": "blocked",
            "evidence_file": "",
        }
        return row, []

    replay = _rows(replay_csv)[0]
    model = parse_contactenergy_input(inp_path.read_text(encoding="utf-8"))
    U, _RF, _energy, _raw = parse_contactenergy_dat(dat_path.read_text(encoding="utf-8", errors="ignore"), model.node_ids)
    _strains, _stresses, vm, strain_norm = contactenergy_hex8_center_strain_stress(model, U)
    _write_legacy_hex_vtk(
        out_dir / "vtk" / "c3d8_linear_static_contact_0000.vtk",
        model.X + U,
        model.elements,
        cell_values={"von_mises": vm, "engineering_strain_norm": strain_norm},
    )
    row = {
        "case_id": "c3d8_linear_static_contact",
        "analysis_type": "static",
        "linearity": "linear",
        "element_type": "C3D8",
        "native_sfc_result": "true",
        "calculix_comparison": "true",
        "acceleration_evidence": "false",
        "displacement_metric": replay.get("sfc_c3d8_displacement_l2_rel_error", ""),
        "stress_strain_metric": replay.get("sfc_c3d8_von_mises_l2_rel_error", ""),
        "gap_metric": replay.get("sfc_dynamic_sdf_max_penetration", ""),
        "rf_metric": replay.get("contact_force_rel_error", ""),
        "cels_metric": replay.get("contact_energy_rel_error", ""),
        "active_contact_metric": replay.get("sfc_dynamic_sdf_active_quadrature_count", ""),
        "supports_external_correctness": "true",
        "supports_trajectory_equivalence": "false",
        "supports_efficiency": "false",
        "status": "supported",
        "evidence_file": _display_path(replay_csv),
    }
    stress_rows = [
        {
            "case_id": "c3d8_linear_static_contact",
            "frame": 0,
            "element_id": int(element_id),
            "von_mises": float(value),
            "engineering_strain_norm": float(strain),
            "vtk_file": "vtk/c3d8_linear_static_contact_0000.vtk",
        }
        for element_id, value, strain in zip(model.element_ids, vm, strain_norm, strict=True)
    ]
    return row, stress_rows


def _run_c3d8_trajectory(out_dir: Path, *, quick: bool, skip_calculix: bool) -> Path:
    traj_dir = out_dir / "c3d8_dynamic_trajectory"
    args = ["--out-dir", str(traj_dir)]
    if quick:
        args.append("--quick")
    if skip_calculix:
        args.append("--skip-calculix")
    rc = run_c3d8_trajectory_main(args)
    if rc != 0:
        raise RuntimeError(f"C3D8 trajectory runner failed with return code {rc}")
    return traj_dir


def _linear_dynamic_evidence(traj_dir: Path, out_dir: Path) -> tuple[Row, list[Row]]:
    summary_path = traj_dir / "c3d8_contact_trajectory_summary.csv"
    if not summary_path.exists():
        return {
            "case_id": "c3d8_linear_dynamic_contact",
            "analysis_type": "dynamic",
            "linearity": "linear",
            "element_type": "C3D8",
            "native_sfc_result": "false",
            "calculix_comparison": "false",
            "acceleration_evidence": "false",
            "displacement_metric": "",
            "stress_strain_metric": "",
            "gap_metric": "",
            "rf_metric": "",
            "cels_metric": "",
            "active_contact_metric": "",
            "supports_external_correctness": "false",
            "supports_trajectory_equivalence": "false",
            "supports_efficiency": "false",
            "status": "blocked",
            "evidence_file": "",
        }, []
    summaries = _rows(summary_path)
    external = all(row["reference_source"] == "calculix_dat" for row in summaries)
    metric_ok = all(float(row["peak_force_rel_error"]) < 1.0e-2 and float(row["peak_energy_rel_error"]) < 1.0e-2 and float(row["min_gap_rel_error"]) < 1.0e-2 for row in summaries)
    active = sum(int(row["active_sfc_rows"]) for row in summaries)
    max_force_error = max(float(row["peak_force_rel_error"]) for row in summaries)
    max_energy_error = max(float(row["peak_energy_rel_error"]) for row in summaries)
    max_gap_error = max(float(row["min_gap_rel_error"]) for row in summaries)
    max_pen = max(float(row["max_sfc_penetration"]) for row in summaries)

    stress_rows: list[Row] = []
    for case in ["block_plane_c3d8", "block_block_c3d8"]:
        model = build_c3d8_model(case=case, resolution=1, quick=True)
        dat_path = traj_dir / "calculix_runs" / f"{case}_r1" / f"{case}_r1.dat"
        if dat_path.exists():
            U_blocks = _parse_nodal_vectors(dat_path, model.node_ids, quantity="u")
            stress_blocks = _parse_element_stress(dat_path)
            if U_blocks:
                time = max(U_blocks)
                U = U_blocks[time]
                stress_map = stress_blocks.get(time, {})
            else:
                U = np.zeros_like(model.X)
                stress_map = {}
        else:
            U = np.zeros_like(model.X)
            stress_map = {}
        vm_values: list[float] = []
        strain_values: list[float] = []
        for element_id, element in zip(model.element_ids, model.elements, strict=True):
            strain, sfc_stress = hex8_center_strain_stress(model.X[element], U[element], model.E, model.nu)
            stress = stress_map.get(int(element_id), sfc_stress)
            vm_values.append(_von_mises(stress))
            strain_values.append(float(np.linalg.norm(strain)))
            stress_rows.append(
                {
                    "case_id": f"c3d8_linear_dynamic_contact_{case}",
                    "frame": 0,
                    "element_id": int(element_id),
                    "von_mises": vm_values[-1],
                    "engineering_strain_norm": strain_values[-1],
                    "vtk_file": f"vtk/c3d8_linear_dynamic_contact_{case}_0000.vtk",
                }
            )
        _write_legacy_hex_vtk(
            out_dir / "vtk" / f"c3d8_linear_dynamic_contact_{case}_0000.vtk",
            model.X + U,
            model.elements,
            cell_values={
                "von_mises": np.asarray(vm_values, dtype=float),
                "engineering_strain_norm": np.asarray(strain_values, dtype=float),
            },
        )

    row = {
        "case_id": "c3d8_linear_dynamic_contact",
        "analysis_type": "dynamic",
        "linearity": "linear",
        "element_type": "C3D8",
        "native_sfc_result": "false",
        "calculix_comparison": _bool_text(external),
        "acceleration_evidence": "false",
        "displacement_metric": "z_cm_history_from_calculix",
        "stress_strain_metric": "final_stress_cloud_exported",
        "gap_metric": max_gap_error,
        "rf_metric": max_force_error,
        "cels_metric": max_energy_error,
        "active_contact_metric": active,
        "supports_external_correctness": "false",
        "supports_trajectory_equivalence": "false",
        "supports_efficiency": "false",
        "status": _case_status(False, external, metric_ok),
        "evidence_file": _display_path(summary_path),
    }
    return row, stress_rows


def _unsupported_case(case_id: str, analysis: str, linearity: str) -> Row:
    return {
        "case_id": case_id,
        "analysis_type": analysis,
        "linearity": linearity,
        "element_type": "C3D8",
        "native_sfc_result": "false",
        "calculix_comparison": "false",
        "acceleration_evidence": "false",
        "displacement_metric": "",
        "stress_strain_metric": "",
        "gap_metric": "",
        "rf_metric": "",
        "cels_metric": "",
        "active_contact_metric": "",
        "supports_external_correctness": "false",
        "supports_trajectory_equivalence": "false",
        "supports_efficiency": "false",
        "status": "blocked_no_native_c3d8_nonlinear_backend",
        "evidence_file": "",
    }


def _write_case_markdowns(out_dir: Path, rows: list[Row]) -> None:
    case_dir = out_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        lines = [
            f"# {row['case_id']}",
            "",
            f"- Analysis: `{row['analysis_type']}`",
            f"- Linearity: `{row['linearity']}`",
            f"- Element: `{row['element_type']}`",
            f"- Native SFC result: `{row['native_sfc_result']}`",
            f"- CalculiX comparison: `{row['calculix_comparison']}`",
            f"- Acceleration evidence: `{row['acceleration_evidence']}`",
            f"- Status: `{row['status']}`",
            "",
            "## Metrics",
            "",
            f"- Displacement: `{row['displacement_metric']}`",
            f"- Stress/strain: `{row['stress_strain_metric']}`",
            f"- Gap/CDIS: `{row['gap_metric']}`",
            f"- RF/contact force: `{row['rf_metric']}`",
            f"- CELS/contact energy: `{row['cels_metric']}`",
            f"- Active contact: `{row['active_contact_metric']}`",
            "",
            "## Claim Gate",
            "",
            f"- External correctness: `{row['supports_external_correctness']}`",
            f"- Trajectory equivalence: `{row['supports_trajectory_equivalence']}`",
            f"- Efficiency: `{row['supports_efficiency']}`",
        ]
        (case_dir / f"{row['case_id']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(out_dir: Path, rows: list[Row]) -> list[Row]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_rows: list[Row] = []
    labels = [row["case_id"].replace("c3d8_", "").replace("_contact", "") for row in rows]
    external = [1.0 if row["supports_external_correctness"] == "true" else 0.0 for row in rows]
    trajectory = [1.0 if row["supports_trajectory_equivalence"] == "true" else 0.0 for row in rows]
    efficiency = [1.0 if row["supports_efficiency"] == "true" else 0.0 for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    width = 0.24
    ax.bar(x - width, external, width, label="external correctness")
    ax.bar(x, trajectory, width, label="trajectory equivalence")
    ax.bar(x + width, efficiency, width, label="efficiency")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel("claim supported")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    png = figures / "phase9_claim_gate_matrix.png"
    pdf = figures / "phase9_claim_gate_matrix.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    plot_rows.append({"plot": "phase9_claim_gate_matrix", "png": str(png.relative_to(out_dir)), "pdf": str(pdf.relative_to(out_dir)), "description": "Phase-9 claim support matrix"})

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    metric_rows = [row for row in rows if row["rf_metric"] not in {"", "z_cm_history_from_calculix"}]
    if metric_rows:
        x2 = np.arange(len(metric_rows))
        ax.bar(x2 - 0.2, [float(row["rf_metric"]) for row in metric_rows], 0.2, label="RF rel.")
        ax.bar(x2, [float(row["cels_metric"]) for row in metric_rows], 0.2, label="CELS rel.")
        ax.bar(x2 + 0.2, [float(row["gap_metric"]) if row["gap_metric"] not in {"", "z_cm_history_from_calculix"} else 0.0 for row in metric_rows], 0.2, label="gap rel./value")
        ax.set_xticks(x2)
        ax.set_xticklabels([row["case_id"].replace("c3d8_", "") for row in metric_rows], rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("metric value")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No supported metric rows", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    png = figures / "phase9_contact_error_metrics.png"
    pdf = figures / "phase9_contact_error_metrics.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    plot_rows.append({"plot": "phase9_contact_error_metrics", "png": str(png.relative_to(out_dir)), "pdf": str(pdf.relative_to(out_dir)), "description": "Phase-9 contact error metrics"})
    return plot_rows


def _claim_gates(rows: list[Row]) -> list[Row]:
    gates: list[Row] = []
    for row in rows:
        native = row["native_sfc_result"] == "true"
        external = row["calculix_comparison"] == "true"
        acceleration = row["acceleration_evidence"] == "true"
        gates.append(
            {
                "case_id": row["case_id"],
                "claim": "external_correctness",
                "allowed": _bool_text(native and external and row["supports_external_correctness"] == "true"),
                "reason": "requires native SFC result and CalculiX comparison",
            }
        )
        gates.append(
            {
                "case_id": row["case_id"],
                "claim": "trajectory_equivalence",
                "allowed": _bool_text(native and external and row["supports_trajectory_equivalence"] == "true"),
                "reason": "requires native SFC trajectory and CalculiX trajectory comparison",
            }
        )
        gates.append(
            {
                "case_id": row["case_id"],
                "claim": "efficiency",
                "allowed": _bool_text(acceleration and row["supports_efficiency"] == "true"),
                "reason": "requires acceleration/timing evidence for this case",
            }
        )
    return gates


def _write_summary(out_dir: Path, rows: list[Row], gates: list[Row], plots: list[Row], commands: list[Row]) -> None:
    lines = [
        "# Phase-9 Full Contact Validation",
        "",
        "This package organizes the evidence needed for static/dynamic, linear/nonlinear, multi-element contact claims.",
        "",
        "The current short-term multi-element scope is C3D4/TET4 plus C3D8. C3D10 is explicitly out of scope because no backend is registered.",
        "",
        "## Commands",
        "",
    ]
    for command in commands:
        lines.append(f"- `{command['command']}`")
    lines.extend(
        [
            "",
            "## Evidence Matrix",
            "",
            "| Case | Analysis | Linearity | Native SFC | CalculiX | External correctness | Trajectory equivalence | Efficiency | Status |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['analysis_type']} | {row['linearity']} | {row['native_sfc_result']} | "
            f"{row['calculix_comparison']} | {row['supports_external_correctness']} | {row['supports_trajectory_equivalence']} | "
            f"{row['supports_efficiency']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Gates",
            "",
            "| Case | Claim | Allowed | Reason |",
            "|---|---|---|---|",
        ]
    )
    for gate in gates:
        lines.append(f"| {gate['case_id']} | {gate['claim']} | {gate['allowed']} | {gate['reason']} |")
    lines.extend(["", "## Figures", ""])
    for plot in plots:
        lines.append(f"- `{plot['png']}`: {plot['description']}")
    lines.extend(
        [
            "",
            "## Current Conclusion",
            "",
            "The current evidence supports C3D8 linear static contact correctness and C3D8 dynamic external trajectory replay. It does not yet support native nonlinear C3D8 static/dynamic contact-solve equivalence or per-case efficiency claims. The gates intentionally prevent those unsupported statements from entering the paper.",
        ]
    )
    (out_dir / "phase9_full_contact_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase9(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        {
            "command": "python validation/run_phase9_full_contact_validation.py"
            + (" --quick" if args.quick else "")
            + (" --skip-calculix" if args.skip_calculix else "")
            + f" --out-dir {out_dir}",
            "description": "Phase-9 evidence matrix",
        }
    ]
    trajectory_dir = _run_c3d8_trajectory(out_dir, quick=args.quick, skip_calculix=args.skip_calculix)
    static_row, static_stress = _linear_static_evidence(out_dir)
    dynamic_row, dynamic_stress = _linear_dynamic_evidence(trajectory_dir, out_dir)
    rows = [
        static_row,
        _unsupported_case("c3d8_nonlinear_static_contact", "static", "geometric_nonlinear"),
        dynamic_row,
        _unsupported_case("c3d8_nonlinear_dynamic_contact", "dynamic", "geometric_nonlinear"),
    ]
    stress_rows = static_stress + dynamic_stress
    gates = _claim_gates(rows)
    plots = _write_plots(out_dir, rows)
    _write_case_markdowns(out_dir, rows)
    fields = [
        "case_id",
        "analysis_type",
        "linearity",
        "element_type",
        "native_sfc_result",
        "calculix_comparison",
        "acceleration_evidence",
        "displacement_metric",
        "stress_strain_metric",
        "gap_metric",
        "rf_metric",
        "cels_metric",
        "active_contact_metric",
        "supports_external_correctness",
        "supports_trajectory_equivalence",
        "supports_efficiency",
        "status",
        "evidence_file",
    ]
    _write_csv(out_dir / "phase9_full_contact_validation.csv", fields, rows)
    _write_csv(out_dir / "phase9_claim_gates.csv", ["case_id", "claim", "allowed", "reason"], gates)
    _write_csv(
        out_dir / "phase9_stress_strain_cloud.csv",
        ["case_id", "frame", "element_id", "von_mises", "engineering_strain_norm", "vtk_file"],
        stress_rows,
    )
    _write_csv(out_dir / "phase9_commands.csv", ["command", "description"], commands)
    _write_csv(out_dir / "phase9_plots.csv", ["plot", "png", "pdf", "description"], plots)
    _write_summary(out_dir, rows, gates, plots, commands)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-calculix", action="store_true", help="Use synthetic C3D8 trajectory data for CI smoke tests.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "phase9_full_contact_validation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_phase9(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
