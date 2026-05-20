"""Abaqus/Explicit linear-penalty sphere-drop time-step convergence.

This validation-only runner builds the same flexible C3D4 sphere and rigid
plane used by the commercial comparison, but runs Abaqus/Explicit with fixed
user time increments, no bulk viscosity, frictionless linear penalty contact,
and a one-second duration.  It then exports Abaqus' own ODB fields to VTK and
compares the resulting Abaqus curves across time steps.

The SFC core package is not imported by this runner.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from validation.run_abaqus_flexible_body_rigid_plane import (  # noqa: E402
    CommonConfig,
    SphereConfig,
    _chunked_lines,
    _format_element,
    _format_node,
    _mesh_counts,
    _plane_part_lines,
    _resolve_abaqus_command,
    _run_command,
    _sphere_mesh,
)
from validation.run_abaqus_sphere_drop_short_validation import (  # noqa: E402
    _abaqus_history_from_vtk,
    _configure_plot_style,
    _write_csv,
    parse_sphere_drop_inp,
)

Row = dict[str, Any]

DEFAULT_OUT_DIR = ROOT / "commercial_software_comparison" / "abaqus_explicit_linear_timestep_convergence"
DEFAULT_DT_VALUES = (1.0e-3, 1.0e-4, 1.0e-5)
DEFAULT_DURATION = 1.0
DEFAULT_OUTPUT_INTERVAL = 1.0e-3
LINEAR_PENALTY_STIFFNESS = 5.0e9
ENERGY_VARIABLES = ("ALLKE", "ALLIE", "ALLSE", "ALLVD", "ALLWK", "ETOTAL")


@dataclass(frozen=True, slots=True)
class ExplicitLinearConfig:
    """Input settings for one Abaqus/Explicit fixed-step run."""

    duration: float = DEFAULT_DURATION
    output_interval: float = DEFAULT_OUTPUT_INTERVAL
    fixed_dt: float = 1.0e-4
    density: float = 1200.0
    young: float = 5.0e7
    poisson: float = 0.30
    gravity: float = 9.81
    initial_gap: float = 0.02
    plane_half_width: float = 0.30
    contact_stiffness: float = LINEAR_PENALTY_STIFFNESS

    def common(self) -> CommonConfig:
        return CommonConfig(
            density=float(self.density),
            young=float(self.young),
            poisson=float(self.poisson),
            gravity=float(self.gravity),
            duration=float(self.duration),
            output_interval=float(self.output_interval),
            initial_gap=float(self.initial_gap),
            plane_half_width=float(self.plane_half_width),
        )


def _material_lines(cfg: ExplicitLinearConfig) -> list[str]:
    return [
        "*Material, name=BODY_MAT",
        "*Density",
        f"{cfg.density:.12e}",
        "*Elastic",
        f"{cfg.young:.12e}, {cfg.poisson:.12e}",
        "*Surface Interaction, name=FRICTIONLESS_LINEAR_PENALTY",
        "*Surface Behavior, pressure-overclosure=LINEAR",
        f"{cfg.contact_stiffness:.12e}",
        "*Friction",
        "0.",
    ]


def build_input_text(cfg: ExplicitLinearConfig) -> str:
    """Build one Abaqus/Explicit fixed-step, linear-contact sphere input."""

    common = cfg.common()
    nodes, elements, all_nodes = _sphere_mesh(SphereConfig(), common)
    lines: list[str] = [
        "*Heading",
        "Abaqus Explicit fixed-dt linear-penalty sphere free fall onto rigid plane",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
    ]
    lines.extend(_material_lines(cfg))
    lines.extend(["*Part, name=FLEX_SPHERE", "*Node"])
    lines.extend(_format_node(label, xyz) for label, xyz in enumerate(nodes, start=1))
    lines.append("*Element, type=C3D4, elset=SPHERE_EALL")
    lines.extend(_format_element(label, conn) for label, conn in enumerate(elements, start=1))
    lines.extend(
        [
            f"*Elset, elset=SPHERE_EALL, generate",
            f"1, {len(elements)}, 1",
            "*Solid Section, elset=SPHERE_EALL, material=BODY_MAT",
            ",",
            "*End Part",
        ]
    )
    lines.extend(_plane_part_lines(common))
    lines.extend(
        [
            "*Assembly, name=ASSEMBLY",
            "*Instance, name=SPHERE-1, part=FLEX_SPHERE",
            "*End Instance",
            "*Instance, name=PLANE-1, part=RIGID_PLANE",
            "*End Instance",
            "*Nset, nset=BODY_NODES, instance=SPHERE-1",
        ]
    )
    lines.extend(_chunked_lines(all_nodes))
    lines.extend(
        [
            "*End Assembly",
            "*Boundary",
            "PLANE-1.PLANE_RP, ENCASTRE",
            "*Step, name=EXPLICIT_LINEAR_FIXED_DT, nlgeom=YES",
            "*Dynamic, Explicit, DIRECT USER CONTROL",
            f"{cfg.fixed_dt:.12e}, {cfg.duration:.12e}",
            "*Bulk Viscosity",
            "0., 0.",
            "*Dload",
            f"SPHERE-1.SPHERE_EALL, GRAV, {cfg.gravity:.12e}, 0., 0., -1.",
            "*Contact",
            "*Contact Inclusions, ALL EXTERIOR",
            "*Contact Property Assignment",
            ", , FRICTIONLESS_LINEAR_PENALTY",
            f"*Output, field, time interval={cfg.output_interval:.12e}",
            "*Node Output",
            "U, V",
            "*Element Output, elset=SPHERE-1.SPHERE_EALL, directions=YES",
            "S, LE",
            f"*Output, history, time interval={cfg.output_interval:.12e}",
            "*Energy Output",
            "ALLKE, ALLIE, ALLSE, ALLVD, ALLWK, ETOTAL",
            "*End Step",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _case_name(fixed_dt: float) -> str:
    mantissa, exponent = f"{fixed_dt:.12e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"dt_{mantissa.replace('.', 'p')}e{int(exponent):+03d}".replace("+", "").replace("-", "m")


def _dt_label(rows: list[Row]) -> str:
    return rf"$\Delta t={float(rows[0]['fixed_dt']):.1e}$"


def _analysis_status_from_sta(sta_path: Path) -> tuple[str, str]:
    """Return Abaqus analysis status from the status file."""

    if not sta_path.exists():
        return "unknown", f"status file not found: {sta_path}"
    text = sta_path.read_text(encoding="utf-8", errors="ignore")
    if "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in text:
        return "completed", ""
    if "THE ANALYSIS HAS NOT BEEN COMPLETED" in text:
        details: list[str] = []
        if "Excessive distortion" in text:
            details.append("excessive element distortion")
        if "deformation speed to wave speed exceeds" in text:
            details.append("deformation speed exceeds wave speed")
        if "deep penetration" in text:
            details.append("deep contact penetration")
        return "analysis_failed", "; ".join(details) or "analysis did not complete"
    if "***ERROR" in text:
        return "analysis_failed", "Abaqus status file contains errors"
    return "unknown", "Abaqus completion marker not found"


def _run_case(
    cfg: ExplicitLinearConfig,
    out_root: Path,
    *,
    abaqus_command: str | None,
    convert_odb: bool,
    extract_energy: bool,
) -> dict[str, Path | float | int | str]:
    case_name = _case_name(cfg.fixed_dt)
    case_dir = (out_root / case_name).resolve()
    run_dir = case_dir / "abaqus_run"
    vtk_dir = case_dir / "vtk"
    run_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir.mkdir(parents=True, exist_ok=True)
    job_name = f"sphere_drop_explicit_linear_{case_name}"
    for stale in run_dir.glob(f"{job_name}*"):
        if stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink()
    for stale_vtk in vtk_dir.glob("frame_*.vtk"):
        stale_vtk.unlink()
    for stale_index in (vtk_dir / "frame.pvd", vtk_dir / "frame_manifest.csv"):
        stale_index.unlink(missing_ok=True)

    inp_path = run_dir / f"{job_name}.inp"
    inp_path.write_text(build_input_text(cfg), encoding="ascii")
    command = _resolve_abaqus_command(abaqus_command)
    total_start = time.perf_counter()
    analysis_wall = _run_command(
        [command, f"job={job_name}", f"input={inp_path.name}", "interactive", "ask_delete=OFF"],
        cwd=run_dir,
        log_path=case_dir / "abaqus_analysis_stdout.log",
    )
    odb_path = run_dir / f"{job_name}.odb"
    if not odb_path.exists():
        raise RuntimeError(f"Abaqus completed without producing {odb_path}")
    status, error_message = _analysis_status_from_sta(run_dir / f"{job_name}.sta")

    export_wall = 0.0
    if convert_odb and status == "completed":
        converter = Path(__file__).with_name("abaqus_odb_to_vtk.py").resolve()
        export_wall = _run_command(
            [
                command,
                "python",
                str(converter),
                "--odb",
                str(odb_path.resolve()),
                "--out-dir",
                str(vtk_dir.resolve()),
                "--stem",
                "frame",
            ],
            cwd=run_dir,
            log_path=case_dir / "abaqus_odb_to_vtk_stdout.log",
        )
    energy_csv = case_dir / "abaqus_energy_history.csv"
    energy_wall = 0.0
    if extract_energy and status == "completed":
        extractor = Path(__file__).with_name("abaqus_extract_history_energy.py").resolve()
        energy_wall = _run_command(
            [
                command,
                "python",
                str(extractor),
                "--odb",
                str(odb_path.resolve()),
                "--out-csv",
                str(energy_csv.resolve()),
                "--variables",
                *ENERGY_VARIABLES,
            ],
            cwd=run_dir,
            log_path=case_dir / "abaqus_energy_history_stdout.log",
        )
    total_wall = time.perf_counter() - total_start
    node_count, element_count = _mesh_counts("sphere_drop", cfg.common())
    return {
        "case": case_name,
        "fixed_dt": float(cfg.fixed_dt),
        "duration": float(cfg.duration),
        "output_interval": float(cfg.output_interval),
        "input_file": str(inp_path),
        "odb_file": str(odb_path),
        "vtk_dir": str(vtk_dir),
        "analysis_wall_seconds": float(analysis_wall),
        "odb_to_vtk_wall_seconds": float(export_wall),
        "energy_extract_wall_seconds": float(energy_wall),
        "total_wall_seconds": float(total_wall),
        "node_count": int(node_count),
        "element_count": int(element_count),
        "contact_stiffness": float(cfg.contact_stiffness),
        "bulk_viscosity_linear": 0.0,
        "bulk_viscosity_quadratic": 0.0,
        "status": status,
        "error_message": error_message,
        "energy_history_csv": str(energy_csv),
    }


def _failed_case_row(cfg: ExplicitLinearConfig, out_root: Path, exc: BaseException) -> Row:
    case_name = _case_name(cfg.fixed_dt)
    case_dir = (out_root / case_name).resolve()
    job_name = f"sphere_drop_explicit_linear_{case_name}"
    node_count, element_count = _mesh_counts("sphere_drop", cfg.common())
    return {
        "case": case_name,
        "fixed_dt": float(cfg.fixed_dt),
        "duration": float(cfg.duration),
        "output_interval": float(cfg.output_interval),
        "input_file": str(case_dir / "abaqus_run" / f"{job_name}.inp"),
        "odb_file": str(case_dir / "abaqus_run" / f"{job_name}.odb"),
        "vtk_dir": str(case_dir / "vtk"),
        "analysis_wall_seconds": "",
        "odb_to_vtk_wall_seconds": "",
        "energy_extract_wall_seconds": "",
        "total_wall_seconds": "",
        "node_count": int(node_count),
        "element_count": int(element_count),
        "contact_stiffness": float(cfg.contact_stiffness),
        "bulk_viscosity_linear": 0.0,
        "bulk_viscosity_quadratic": 0.0,
        "status": "command_failed",
        "error_message": str(exc),
        "energy_history_csv": str(case_dir / "abaqus_energy_history.csv"),
    }


def _history_for_case(row: Row) -> list[Row]:
    model = parse_sphere_drop_inp(Path(str(row["input_file"])))
    history = _abaqus_history_from_vtk(model, Path(str(row["vtk_dir"])), duration=float(row["duration"]))
    for item in history:
        item["case"] = row["case"]
        item["fixed_dt"] = float(row["fixed_dt"])
    return history


def _energy_history_for_case(row: Row) -> list[Row]:
    path = Path(str(row["energy_history_csv"]))
    if not path.exists():
        return []
    rows = _read_csv(path)
    for item in rows:
        item["case"] = row["case"]
        item["fixed_dt"] = float(row["fixed_dt"])
    return rows


def _finite_energy_values(rows: list[Row], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        values.append(float(value))
    return np.asarray(values, dtype=float)


def _energy_audit_rows(energy_histories: dict[str, list[Row]]) -> list[Row]:
    audit: list[Row] = []
    for case, rows in sorted(energy_histories.items()):
        if not rows:
            continue
        etotal = _finite_energy_values(rows, "ETOTAL")
        allke = _finite_energy_values(rows, "ALLKE")
        allie = _finite_energy_values(rows, "ALLIE")
        allse = _finite_energy_values(rows, "ALLSE")
        allvd = _finite_energy_values(rows, "ALLVD")
        allwk = _finite_energy_values(rows, "ALLWK")
        scale_candidates = [
            float(np.max(np.abs(values))) for values in (etotal, allke, allie, allse, allwk) if values.size
        ]
        scale = max(scale_candidates + [1.0e-30])
        etotal_initial = float(etotal[0]) if etotal.size else 0.0
        etotal_final = float(etotal[-1]) if etotal.size else 0.0
        etotal_drift = etotal - etotal_initial if etotal.size else np.zeros(1)
        audit.append(
            {
                "case": case,
                "fixed_dt": float(rows[0]["fixed_dt"]),
                "samples": len(rows),
                "etotal_initial": etotal_initial,
                "etotal_final": etotal_final,
                "etotal_final_minus_initial": etotal_final - etotal_initial,
                "max_abs_etotal_drift": float(np.max(np.abs(etotal_drift))),
                "max_abs_etotal_drift_over_energy_scale": float(np.max(np.abs(etotal_drift)) / scale),
                "allke_max": float(np.max(allke)) if allke.size else 0.0,
                "allie_max": float(np.max(allie)) if allie.size else 0.0,
                "allse_max": float(np.max(allse)) if allse.size else 0.0,
                "allvd_final": float(allvd[-1]) if allvd.size else 0.0,
                "allvd_max_abs": float(np.max(np.abs(allvd))) if allvd.size else 0.0,
                "allwk_final": float(allwk[-1]) if allwk.size else 0.0,
            }
        )
    return audit


def _interp(rows: list[Row], key: str, times: np.ndarray) -> np.ndarray:
    row_times = np.asarray([float(row["time"]) for row in rows], dtype=float)
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return np.interp(times, row_times, values)


def _convergence_rows(histories: dict[str, list[Row]], *, reference_case: str) -> list[Row]:
    ref = histories[reference_case]
    times = np.asarray([float(row["time"]) for row in ref], dtype=float)
    ref_z = np.asarray([float(row["z_cm"]) for row in ref], dtype=float)
    ref_gap = np.asarray([float(row["min_gap"]) for row in ref], dtype=float)
    rows: list[Row] = []
    for case, history in histories.items():
        z = _interp(history, "z_cm", times)
        gap = _interp(history, "min_gap", times)
        z_error = z - ref_z
        gap_error = gap - ref_gap
        rows.append(
            {
                "case": case,
                "reference_case": reference_case,
                "max_z_cm_abs_error_vs_reference": float(np.max(np.abs(z_error))),
                "rms_z_cm_abs_error_vs_reference": float(np.sqrt(np.mean(z_error * z_error))),
                "max_min_gap_abs_error_vs_reference": float(np.max(np.abs(gap_error))),
                "rms_min_gap_abs_error_vs_reference": float(np.sqrt(np.mean(gap_error * gap_error))),
                "z_cm_at_end": float(z[-1]),
                "min_gap_at_end": float(gap[-1]),
            }
        )
    return rows


def _plot_overlay(path: Path, histories: dict[str, list[Row]], *, key: str, ylabel: str, title: str) -> None:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(4.7, 2.2), constrained_layout=True)
    colors = ["#1f4e79", "#c0504d", "#548235", "#8064a2"]
    for index, (case, rows) in enumerate(sorted(histories.items())):
        times = np.asarray([float(row["time"]) for row in rows], dtype=float)
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        ax.plot(times, values, linewidth=1.3, color=colors[index % len(colors)], label=_dt_label(rows))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#d9d9d9", linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_energy_terms(path: Path, energy_histories: dict[str, list[Row]]) -> None:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(5.8, 2.35), constrained_layout=True)
    colors = {
        "ALLKE": "#1f4e79",
        "ALLIE": "#c0504d",
        "ALLSE": "#548235",
        "ALLVD": "#8064a2",
        "ETOTAL": "#000000",
    }
    for case, rows in sorted(energy_histories.items()):
        if not rows:
            continue
        times = np.asarray([float(row["time"]) for row in rows], dtype=float)
        suffix = _dt_label(rows)
        for variable, color in colors.items():
            values = _finite_energy_values(rows, variable)
            if values.size != times.size:
                continue
            linestyle = "-" if variable == "ETOTAL" else "--"
            linewidth = 1.4 if variable == "ETOTAL" else 1.0
            ax.plot(times, values, color=color, linestyle=linestyle, linewidth=linewidth, label=f"{variable}, {suffix}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    ax.set_title("Abaqus ODB energy history")
    ax.grid(True, color="#d9d9d9", linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=6)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _write_summary(
    path: Path,
    *,
    case_rows: list[Row],
    convergence: list[Row],
    energy_audit: list[Row],
    outputs: dict[str, Path],
) -> None:
    lines = [
        "# Abaqus Explicit Linear Contact Time-Step Convergence",
        "",
        "This external validation runs Abaqus/Explicit only.  The sphere-drop model uses frictionless linear penalty contact, fixed user time increments, and zero bulk viscosity.",
        "",
        "## Cases",
        "",
        "| Case | Fixed dt | Status | Analysis wall (s) | Export wall (s) | Energy extract wall (s) |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in case_rows:
        analysis = row.get("analysis_wall_seconds", "")
        export = row.get("odb_to_vtk_wall_seconds", "")
        energy = row.get("energy_extract_wall_seconds", "")
        analysis_text = "" if analysis == "" else f"{float(analysis):.3f}"
        export_text = "" if export == "" else f"{float(export):.3f}"
        energy_text = "" if energy == "" else f"{float(energy):.3f}"
        lines.append(
            f"| `{row['case']}` | {float(row['fixed_dt']):.1e} | {row.get('status', '')} | {analysis_text} | {export_text} | {energy_text} |"
        )
    failures = [row for row in case_rows if row.get("status") != "completed"]
    if failures:
        lines.extend(["", "## Failed Or Incomplete Cases", ""])
        for row in failures:
            lines.append(f"- `{row['case']}`: {row.get('error_message', '')}")
    lines.extend(["", "## Convergence Against Finest Step", "", "| Case | max z error | RMS z error | max gap error | RMS gap error |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in convergence:
        lines.append(
            "| `{case}` | {maxz:.6e} | {rmsz:.6e} | {maxg:.6e} | {rmsg:.6e} |".format(
                case=row["case"],
                maxz=float(row["max_z_cm_abs_error_vs_reference"]),
                rmsz=float(row["rms_z_cm_abs_error_vs_reference"]),
                maxg=float(row["max_min_gap_abs_error_vs_reference"]),
                rmsg=float(row["rms_min_gap_abs_error_vs_reference"]),
            )
        )
    if energy_audit:
        lines.extend(
            [
                "",
                "## ODB Energy Audit",
                "",
                "| Case | max abs ETOTAL drift | normalized drift | ALLVD final | ALLVD max | ALLWK final |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in energy_audit:
            lines.append(
                "| `{case}` | {drift:.6e} | {norm:.6e} | {allvd_final:.6e} | {allvd_max:.6e} | {allwk_final:.6e} |".format(
                    case=row["case"],
                    drift=float(row["max_abs_etotal_drift"]),
                    norm=float(row["max_abs_etotal_drift_over_energy_scale"]),
                    allvd_final=float(row["allvd_final"]),
                    allvd_max=float(row["allvd_max_abs"]),
                    allwk_final=float(row["allwk_final"]),
                )
            )
    lines.extend(["", "## Outputs", ""])
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value.name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_convergence(
    out_dir: Path,
    *,
    dt_values: tuple[float, ...] = DEFAULT_DT_VALUES,
    duration: float = DEFAULT_DURATION,
    output_interval: float = DEFAULT_OUTPUT_INTERVAL,
    abaqus_command: str | None = None,
    convert_odb: bool = True,
    extract_energy: bool = True,
) -> dict[str, Path]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[Row] = []
    histories: dict[str, list[Row]] = {}
    energy_histories: dict[str, list[Row]] = {}
    for fixed_dt in dt_values:
        cfg = ExplicitLinearConfig(duration=float(duration), output_interval=float(output_interval), fixed_dt=float(fixed_dt))
        try:
            row = _run_case(cfg, out_dir, abaqus_command=abaqus_command, convert_odb=convert_odb, extract_energy=extract_energy)
        except Exception as exc:  # Keep the rest of the time-step sweep running.
            row = _failed_case_row(cfg, out_dir, exc)
            print(f"[warning] {row['case']} failed: {exc}")
        case_rows.append(row)
        if convert_odb and row.get("status") == "completed":
            try:
                histories[str(row["case"])] = _history_for_case(row)
            except Exception as exc:
                row["status"] = "history_failed"
                row["error_message"] = str(exc)
                print(f"[warning] history extraction failed for {row['case']}: {exc}")
        if extract_energy and row.get("status") == "completed":
            energy_rows = _energy_history_for_case(row)
            if energy_rows:
                energy_histories[str(row["case"])] = energy_rows
    outputs = {
        "case_metrics": out_dir / "abaqus_explicit_linear_case_metrics.csv",
        "history": out_dir / "abaqus_explicit_linear_time_history.csv",
        "energy_history": out_dir / "abaqus_explicit_linear_energy_history.csv",
        "convergence": out_dir / "abaqus_explicit_linear_convergence.csv",
        "energy_audit": out_dir / "abaqus_explicit_linear_energy_audit.csv",
        "z_curve": out_dir / "abaqus_explicit_linear_z_cm.png",
        "gap_curve": out_dir / "abaqus_explicit_linear_min_gap.png",
        "force_curve": out_dir / "abaqus_explicit_linear_normal_force.png",
        "energy_curve": out_dir / "abaqus_explicit_linear_energy_terms.png",
        "summary": out_dir / "abaqus_explicit_linear_timestep_convergence_summary.md",
    }
    _write_csv(outputs["case_metrics"], case_rows)
    all_history = [row for rows in histories.values() for row in rows]
    _write_csv(outputs["history"], all_history)
    all_energy_history = [row for rows in energy_histories.values() for row in rows]
    _write_csv(outputs["energy_history"], all_energy_history)
    convergence: list[Row] = []
    if histories:
        fixed_by_case = {str(row["case"]): float(row["fixed_dt"]) for row in case_rows if str(row["case"]) in histories}
        reference_case = min(histories, key=lambda case: fixed_by_case[case])
        convergence = _convergence_rows(histories, reference_case=reference_case)
        _write_csv(outputs["convergence"], convergence)
        _plot_overlay(outputs["z_curve"], histories, key="z_cm", ylabel="Mass-center z (m)", title="Abaqus Explicit displacement convergence")
        _plot_overlay(outputs["gap_curve"], histories, key="min_gap", ylabel="Minimum gap (m)", title="Abaqus Explicit gap convergence")
        _plot_overlay(outputs["force_curve"], histories, key="normal_force_z", ylabel="Normal force proxy (N)", title="Abaqus Explicit normal force convergence")
    else:
        _write_csv(outputs["convergence"], convergence)
    energy_audit = _energy_audit_rows(energy_histories)
    _write_csv(outputs["energy_audit"], energy_audit)
    if energy_histories:
        _plot_energy_terms(outputs["energy_curve"], energy_histories)
    _write_summary(outputs["summary"], case_rows=case_rows, convergence=convergence, energy_audit=energy_audit, outputs=outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--output-interval", type=float, default=DEFAULT_OUTPUT_INTERVAL)
    parser.add_argument("--dt-values", type=float, nargs="+", default=list(DEFAULT_DT_VALUES))
    parser.add_argument("--abaqus-command", type=str, default=None)
    parser.add_argument("--skip-odb-conversion", action="store_true")
    parser.add_argument("--skip-energy-extraction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_convergence(
        args.out_dir,
        dt_values=tuple(float(value) for value in args.dt_values),
        duration=float(args.duration),
        output_interval=float(args.output_interval),
        abaqus_command=args.abaqus_command,
        convert_odb=not bool(args.skip_odb_conversion),
        extract_energy=not bool(args.skip_energy_extraction),
    )
    print("Abaqus Explicit linear-contact time-step convergence complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
