"""Analytic no-contact dynamics benchmarks for the SFC Newmark integrator."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sfc.fem.integrator import newmark_beta_step  # noqa: E402

Row = dict[str, Any]


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: list[Row]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def _time_grid(total_time: float, dt: float) -> np.ndarray:
    return np.arange(0.0, float(total_time) + 0.5 * float(dt), float(dt))


def free_fall_case(*, total_time: float, dt: float, gravity: float = 9.81, v0: float = -2.0) -> list[Row]:
    """Return Newmark-vs-analytic data for constant-acceleration free fall."""

    M = np.asarray([[1.0]])
    C = np.asarray([[0.0]])
    K = np.asarray([[0.0]])
    f = np.asarray([-float(gravity)])
    u = np.asarray([0.0])
    v = np.asarray([float(v0)])
    a = np.asarray([-float(gravity)])
    rows: list[Row] = []
    for i, time in enumerate(_time_grid(total_time, dt)):
        t = float(time)
        analytic_u = v0 * t - 0.5 * gravity * t * t
        analytic_v = v0 - gravity * t
        kinetic = 0.5 * float(v[0] * v[0])
        potential = gravity * float(u[0])
        total_energy = kinetic + potential
        rows.append(
            {
                "case": "constant_acceleration_free_fall",
                "time": t,
                "numerical_u": float(u[0]),
                "analytic_u": analytic_u,
                "displacement_error": float(u[0] - analytic_u),
                "numerical_v": float(v[0]),
                "analytic_v": analytic_v,
                "velocity_error": float(v[0] - analytic_v),
                "kinetic_energy": kinetic,
                "potential_energy": potential,
                "total_energy": total_energy,
            }
        )
        if i == len(_time_grid(total_time, dt)) - 1:
            break
        u, v, a = newmark_beta_step(M, C, K, u, v, a, f, dt=dt, beta=0.25, gamma=0.5)
    return rows


def harmonic_oscillator_case(*, total_time: float, dt: float, omega: float = 2.0 * np.pi) -> list[Row]:
    """Return Newmark-vs-analytic data for an undamped harmonic oscillator."""

    M = np.asarray([[1.0]])
    C = np.asarray([[0.0]])
    K = np.asarray([[omega * omega]])
    f = np.asarray([0.0])
    u = np.asarray([1.0])
    v = np.asarray([0.0])
    a = np.asarray([-omega * omega])
    rows: list[Row] = []
    for i, time in enumerate(_time_grid(total_time, dt)):
        t = float(time)
        analytic_u = float(np.cos(omega * t))
        analytic_v = float(-omega * np.sin(omega * t))
        kinetic = 0.5 * float(v[0] * v[0])
        strain = 0.5 * omega * omega * float(u[0] * u[0])
        total_energy = kinetic + strain
        rows.append(
            {
                "case": "undamped_harmonic_oscillator",
                "time": t,
                "numerical_u": float(u[0]),
                "analytic_u": analytic_u,
                "displacement_error": float(u[0] - analytic_u),
                "numerical_v": float(v[0]),
                "analytic_v": analytic_v,
                "velocity_error": float(v[0] - analytic_v),
                "kinetic_energy": kinetic,
                "potential_energy": strain,
                "total_energy": total_energy,
            }
        )
        if i == len(_time_grid(total_time, dt)) - 1:
            break
        u, v, a = newmark_beta_step(M, C, K, u, v, a, f, dt=dt, beta=0.25, gamma=0.5)
    return rows


def summarize_metrics(rows: list[Row]) -> list[Row]:
    """Return pass/fail metrics for the analytic benchmark rows."""

    metrics: list[Row] = []
    for case in sorted({str(row["case"]) for row in rows}):
        case_rows = [row for row in rows if row["case"] == case]
        disp_error = max(abs(float(row["displacement_error"])) for row in case_rows)
        vel_error = max(abs(float(row["velocity_error"])) for row in case_rows)
        energies = np.asarray([float(row["total_energy"]) for row in case_rows], dtype=float)
        energy_drift = float(np.max(np.abs(energies - energies[0])) / max(abs(float(energies[0])), 1.0e-30))
        if case == "constant_acceleration_free_fall":
            disp_tol = 1.0e-10
            vel_tol = 1.0e-10
            energy_tol = 1.0e-9
        else:
            disp_tol = 2.0e-3
            vel_tol = 2.0e-2
            energy_tol = 1.0e-10
        metrics.extend(
            [
                {
                    "case": case,
                    "metric": "max_displacement_abs_error",
                    "value": disp_error,
                    "tolerance": disp_tol,
                    "status": "pass" if disp_error <= disp_tol else "fail",
                },
                {
                    "case": case,
                    "metric": "max_velocity_abs_error",
                    "value": vel_error,
                    "tolerance": vel_tol,
                    "status": "pass" if vel_error <= vel_tol else "fail",
                },
                {
                    "case": case,
                    "metric": "relative_total_energy_drift",
                    "value": energy_drift,
                    "tolerance": energy_tol,
                    "status": "pass" if energy_drift <= energy_tol else "fail",
                },
            ]
        )
    return metrics


def write_plots(out_dir: Path, rows: list[Row]) -> list[Row]:
    """Write diagnostic plots for displacement and total energy."""

    plots: list[Row] = []
    for case in sorted({str(row["case"]) for row in rows}):
        case_rows = [row for row in rows if row["case"] == case]
        times = [float(row["time"]) for row in case_rows]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(times, [float(row["numerical_u"]) for row in case_rows], label="Newmark")
        ax.plot(times, [float(row["analytic_u"]) for row in case_rows], linestyle="--", label="analytic")
        ax.set_xlabel("time")
        ax.set_ylabel("displacement")
        ax.set_title(case)
        ax.grid(True, color="0.9", linewidth=0.5)
        ax.legend()
        fig.tight_layout()
        disp_png = out_dir / f"{case}_displacement.png"
        disp_pdf = out_dir / f"{case}_displacement.pdf"
        fig.savefig(disp_png, dpi=180)
        fig.savefig(disp_pdf)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(times, [float(row["total_energy"]) for row in case_rows], label="total")
        ax.set_xlabel("time")
        ax.set_ylabel("total mechanical energy")
        ax.set_title(case)
        ax.grid(True, color="0.9", linewidth=0.5)
        ax.legend()
        fig.tight_layout()
        energy_png = out_dir / f"{case}_energy.png"
        energy_pdf = out_dir / f"{case}_energy.pdf"
        fig.savefig(energy_png, dpi=180)
        fig.savefig(energy_pdf)
        plt.close(fig)

        plots.extend(
            [
                {"case": case, "plot": "displacement", "png": disp_png.name, "pdf": disp_pdf.name, "status": "ok"},
                {"case": case, "plot": "energy", "png": energy_png.name, "pdf": energy_pdf.name, "status": "ok"},
            ]
        )
    return plots


def write_markdown(path: Path, *, command: str, metrics: list[Row], plots: list[Row]) -> None:
    all_pass = all(row["status"] == "pass" for row in metrics)
    lines = [
        "# Dynamic Integrator Analytic Benchmark",
        "",
        "This benchmark verifies the no-contact dynamics path before using any contact comparison. It checks constant-acceleration free fall and an undamped single-degree-of-freedom oscillator against analytic solutions.",
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Claim Gate",
        "",
        "| Claim | Status | Evidence |",
        "| --- | --- | --- |",
        f"| no-contact Newmark dynamics matches analytic baselines | {'supported' if all_pass else 'not_supported'} | dynamic_integrator_metrics.csv |",
        "<!-- evidence csv=dynamic_integrator_metrics.csv field=value -->",
        "",
        "## Metrics",
        "",
        "| Case | Metric | Value | Tolerance | Status |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in metrics:
        lines.append(f"| {row['case']} | {row['metric']} | {row['value']} | {row['tolerance']} | {row['status']} |")
    lines.extend(["", "## Plots", ""])
    for row in plots:
        lines.append(f"- `{row['png']}` and `{row['pdf']}`")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This benchmark does not validate contact. It is an isolated check of the linear no-contact time integration behavior.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(out_dir: Path, *, quick: bool) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_time = 0.5 if quick else 2.0
    dt = 1.0e-3 if quick else 5.0e-4
    rows = [
        *free_fall_case(total_time=total_time, dt=dt),
        *harmonic_oscillator_case(total_time=total_time, dt=dt),
    ]
    metrics = summarize_metrics(rows)
    plots = write_plots(out_dir, rows)

    timeseries_csv = out_dir / "dynamic_integrator_timeseries.csv"
    metrics_csv = out_dir / "dynamic_integrator_metrics.csv"
    plots_csv = out_dir / "dynamic_integrator_plots.csv"
    summary_md = out_dir / "dynamic_integrator_summary.md"
    command = f"python validation/run_dynamic_integrator_benchmark.py {'--quick ' if quick else ''}--out-dir {out_dir.as_posix()}"

    _write_csv(timeseries_csv, _fieldnames(rows), rows)
    _write_csv(metrics_csv, _fieldnames(metrics), metrics)
    _write_csv(plots_csv, _fieldnames(plots), plots)
    write_markdown(summary_md, command=command, metrics=metrics, plots=plots)
    return {
        "timeseries": timeseries_csv,
        "metrics": metrics_csv,
        "plots": plots_csv,
        "summary": summary_md,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a short deterministic benchmark.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "dynamic_integrator_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_benchmark(args.out_dir, quick=bool(args.quick))
    print("Dynamic integrator benchmark complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
