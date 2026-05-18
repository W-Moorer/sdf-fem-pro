"""Run backend ablation for the independent SFC native-contact path.

This script does not use CalculiX as a solver.  It compares SFC contact
integration/backend variants on the same benchmark-style model so the paper can
separate accuracy evidence from engineering-cost evidence:

* node-to-surface centroid samples;
* reference surface-to-surface quadrature;
* vectorized surface-to-surface quadrature;
* vectorized surface-to-surface with forced spatial-hash grouped field build.
"""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_fuzzycontact_calculix_native_contact_trajectory import (  # noqa: E402
    _configure_times_fonts,
    run_validation as run_native_contact_validation,
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


def _read_csv(path: Path) -> list[Row]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: Row, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return float(value)


def _plot_ablation(out_dir: Path, rows: list[Row]) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    _configure_times_fonts(matplotlib)
    import matplotlib.pyplot as plt

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    label_map = {
        "node_to_surface": "node-to-surface\ncentroid",
        "surface_to_surface_reference": "reference\nsurface-to-surface",
        "surface_to_surface_vectorized": "vectorized\nsurface-to-surface",
        "surface_to_surface_vectorized_grouped_projection": "vectorized + batch\nprojection build",
    }
    labels = [label_map.get(str(row["backend_label"]), str(row["backend_label"])) for row in rows]
    x = np.arange(len(labels), dtype=float)

    field = np.asarray([_float(row, "sfc_field_update_total_seconds") for row in rows], dtype=float)
    query = np.asarray([_float(row, "sfc_query_total_seconds") for row in rows], dtype=float)
    solve = np.asarray([_float(row, "sfc_linear_solve_total_seconds") for row in rows], dtype=float)
    other = np.asarray([_float(row, "sfc_other_total_seconds") for row in rows], dtype=float)

    colors = {
        "field update": "#4c78a8",
        "query/contact": "#72b7b2",
        "linear solve": "#f58518",
        "other": "#b279a2",
    }

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    bottom = np.zeros_like(x)
    for values, label in [
        (field, "field update"),
        (query, "query/contact"),
        (solve, "linear solve"),
        (other, "other"),
    ]:
        ax.bar(x, values, bottom=bottom, label=label, color=colors[label])
        bottom += values
    ax.set_yscale("log")
    ax.set_ylabel("SFC wall time (s)")
    ax.set_xlabel("Backend")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.grid(True, axis="y", which="both", alpha=0.30)
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    png = figures / "backend_ablation_runtime.png"
    pdf = figures / "backend_ablation_runtime.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    outputs["runtime_png"] = png
    outputs["runtime_pdf"] = pdf

    disp = [_float(row, "endpoint_displacement_norm_rel_diff") for row in rows]
    vm = [_float(row, "endpoint_von_mises_rel_diff") for row in rows]
    strain = [_float(row, "endpoint_strain_norm_rel_diff") for row in rows]
    min_gap = [abs(_float(row, "endpoint_min_gap_abs_diff")) for row in rows]
    width = 0.20
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for offset, values, label in [
        (-1.5, disp, "|u| norm"),
        (-0.5, vm, "VM max"),
        (0.5, strain, "strain max"),
        (1.5, min_gap, "min gap"),
    ]:
        ax.bar(x + offset * width, values, width, label=label)
    ax.set_yscale("symlog", linthresh=1.0e-14)
    ax.set_ylabel("Endpoint difference vs reference surface quadrature")
    ax.set_xlabel("Backend")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.grid(True, axis="y", which="both", alpha=0.30)
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    png = figures / "backend_ablation_endpoint_metrics.png"
    pdf = figures / "backend_ablation_endpoint_metrics.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    outputs["endpoint_png"] = png
    outputs["endpoint_pdf"] = pdf

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    bottom = np.zeros_like(x)
    for values, label in [
        (field, "field update"),
        (query, "query/contact"),
        (solve, "linear solve"),
        (other, "other"),
    ]:
        axes[0].bar(x, values, bottom=bottom, label=label, color=colors[label])
        bottom += values
    axes[0].set_yscale("log")
    axes[0].set_ylabel("SFC wall time (s)")
    axes[0].set_title("(a) Runtime decomposition", loc="left", fontsize=10, fontweight="bold")
    axes[0].grid(True, axis="y", which="both", alpha=0.30)

    for offset, values, label in [
        (-1.5, disp, "|u| norm"),
        (-0.5, vm, "VM max"),
        (0.5, strain, "strain max"),
        (1.5, min_gap, "min gap"),
    ]:
        axes[1].bar(x + offset * width, values, width, label=label)
    axes[1].set_yscale("symlog", linthresh=1.0e-14)
    axes[1].set_ylabel("Endpoint difference")
    axes[1].set_title("(b) Difference from reference surface-to-surface", loc="left", fontsize=10, fontweight="bold")
    axes[1].grid(True, axis="y", which="both", alpha=0.30)
    for ax in axes:
        ax.set_xlabel("Backend")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    fig.legend(
        handles0 + handles1,
        labels0 + labels1,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    png = figures / "backend_ablation_overview.png"
    pdf = figures / "backend_ablation_overview.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    outputs["overview_png"] = png
    outputs["overview_pdf"] = pdf
    return outputs


def _write_summary(path: Path, rows: list[Row], outputs: dict[str, Path]) -> None:
    lines = [
        "# Native Contact Backend Ablation",
        "",
        "This ablation separates SFC backend engineering choices from the independent CalculiX accuracy comparison.",
        "CalculiX is intentionally skipped here; the reference for endpoint differences is the non-vectorized surface-to-surface quadrature backend.",
        "",
        "## Backends",
        "",
        "- `node_to_surface`: preserved centroid/sample backend.",
        "- `surface_to_surface_reference`: scalar reference surface quadrature backend.",
        "- `surface_to_surface_vectorized`: vectorized quadrature and force assembly.",
        "- `surface_to_surface_vectorized_grouped_projection`: vectorized quadrature with forced spatial-hash grouped SDF field build.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        if key != "summary":
            lines.append(f"- `{value.relative_to(path.parent).as_posix()}`")
    lines.extend(["", "## Timing Rows", "", "| Backend | SFC total s | Field update s | Query s | Linear solve s | Endpoint disp diff |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(
            f"| `{row['backend_label']}` | {float(row['sfc_total_wall_seconds']):.6e} | "
            f"{float(row['sfc_field_update_total_seconds']):.6e} | {float(row['sfc_query_total_seconds']):.6e} | "
            f"{float(row['sfc_linear_solve_total_seconds']):.6e} | {float(row['endpoint_displacement_norm_rel_diff']):.6e} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Supports the paper statement that the backend contains both node-to-surface and surface-to-surface paths.",
            "- Supports an engineering-cost ablation among preserved/reference/vectorized/batch field-build implementations.",
            "- Does not replace the CalculiX native-contact comparison or SDF field accuracy/Jacobian claim gates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    out_dir: Path,
    *,
    quick: bool = False,
    case_name: str = "problem_1",
    quadrature_order: int = 7,
    sdf_spacing_scale: float = 0.25,
    sfc_max_iterations: int = 4,
    closure_steps: int = 11,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("node_to_surface", "node_to_surface", 5_000_000),
        ("surface_to_surface_reference", "surface_to_surface", 5_000_000),
        ("surface_to_surface_vectorized", "surface_to_surface_vectorized", 5_000_000),
        ("surface_to_surface_vectorized_grouped_projection", "surface_to_surface_vectorized", 1),
    ]
    rows: list[Row] = []
    raw_dirs: dict[str, Path] = {}
    for label, backend, threshold in configs:
        run_dir = out_dir / "runs" / label
        raw_dirs[label] = run_dir
        outputs = run_native_contact_validation(
            run_dir,
            quick=quick,
            timeout=1,
            skip_calculix=True,
            sfc_backend=backend,
            quadrature_order=quadrature_order,
            sdf_spacing_scale=sdf_spacing_scale,
            sfc_relaxation=0.55,
            sfc_max_iterations=sfc_max_iterations,
            batch_projection_threshold=threshold,
            case_names=(case_name,),
            closure_steps=closure_steps,
            preserve_quick_closure_steps=True,
        )
        timing = _read_csv(outputs["timing"])[0]
        history = _read_csv(outputs["sfc_history"])
        endpoint = history[-1]
        rows.append(
            {
                "backend_label": label,
                "problem": case_name,
                "quick": str(quick).lower(),
                "closure_steps": int(closure_steps),
                "sfc_backend": backend,
                "batch_projection_threshold": threshold,
                "sfc_total_wall_seconds": _float(timing, "sfc_total_wall_seconds"),
                "sfc_field_update_total_seconds": _float(timing, "sfc_field_update_total_seconds"),
                "sfc_query_total_seconds": _float(timing, "sfc_query_total_seconds"),
                "sfc_linear_solve_total_seconds": _float(timing, "sfc_linear_solve_total_seconds"),
                "sfc_other_total_seconds": _float(timing, "sfc_other_total_seconds"),
                "sfc_contact_evaluations": int(float(timing.get("sfc_contact_evaluations", 0))),
                "endpoint_displacement_l2_norm": _float(endpoint, "sfc_displacement_l2_norm"),
                "endpoint_max_von_mises": _float(endpoint, "sfc_max_von_mises_MPa"),
                "endpoint_max_strain_norm": _float(endpoint, "sfc_max_strain_norm"),
                "endpoint_min_gap": _float(endpoint, "sfc_min_gap_mm"),
                "endpoint_active_samples": int(float(endpoint.get("sfc_active_samples", 0))),
                "endpoint_quadrature_points": int(float(endpoint.get("sfc_quadrature_points", 0))),
                "raw_run_dir": str(run_dir),
            }
        )

    reference = next(row for row in rows if row["backend_label"] == "surface_to_surface_reference")
    for row in rows:
        row["endpoint_displacement_norm_rel_diff"] = abs(
            float(row["endpoint_displacement_l2_norm"]) - float(reference["endpoint_displacement_l2_norm"])
        ) / max(abs(float(reference["endpoint_displacement_l2_norm"])), 1.0e-30)
        row["endpoint_von_mises_rel_diff"] = abs(
            float(row["endpoint_max_von_mises"]) - float(reference["endpoint_max_von_mises"])
        ) / max(abs(float(reference["endpoint_max_von_mises"])), 1.0e-30)
        row["endpoint_strain_norm_rel_diff"] = abs(
            float(row["endpoint_max_strain_norm"]) - float(reference["endpoint_max_strain_norm"])
        ) / max(abs(float(reference["endpoint_max_strain_norm"])), 1.0e-30)
        row["endpoint_min_gap_abs_diff"] = abs(float(row["endpoint_min_gap"]) - float(reference["endpoint_min_gap"]))

    outputs = {
        "ablation": out_dir / "backend_ablation.csv",
        "summary": out_dir / "backend_ablation_summary.md",
    }
    _write_csv(outputs["ablation"], rows)
    outputs.update(_plot_ablation(out_dir, rows))
    _write_summary(outputs["summary"], rows, outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--case", choices=("problem_1", "problem_3", "problem_4"), default="problem_1")
    parser.add_argument("--quadrature-order", type=int, choices=(1, 3, 7), default=7)
    parser.add_argument("--sdf-spacing-scale", type=float, default=0.25)
    parser.add_argument("--sfc-max-iterations", type=int, default=4)
    parser.add_argument("--closure-steps", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "native_contact_backend_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(
        args.out_dir,
        quick=bool(args.quick),
        case_name=str(args.case),
        quadrature_order=int(args.quadrature_order),
        sdf_spacing_scale=float(args.sdf_spacing_scale),
        sfc_max_iterations=int(args.sfc_max_iterations),
        closure_steps=int(args.closure_steps),
    )
    print("Native contact backend ablation complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
