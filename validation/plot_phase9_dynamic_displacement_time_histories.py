"""Plot Phase-9 dynamic displacement histories with physical time on x-axis."""

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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt  # noqa: E402

from validation.run_c3d8_contact_trajectory_validation import (  # noqa: E402
    _center_z,
    _parse_nodal_vectors,
    build_c3d8_model,
)

Row = dict[str, Any]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[Row]) -> None:
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


def _case_error_map(phase9_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for filename in ("native_c3d8_linear_dynamic_comparison.csv", "native_c3d8_nonlinear_dynamic_comparison.csv"):
        for row in _read_rows(phase9_dir / filename):
            if row.get("z_cm_l2_rel_error", ""):
                out[str(row["case_id"])] = float(row["z_cm_l2_rel_error"])
    return out


def _linear_calculix_reference_z(phase9_dir: Path, case: str) -> list[tuple[float, float]]:
    dat = (
        phase9_dir
        / "c3d8_linear_dynamic_calculix"
        / f"linear_dynamic_{case}_r1"
        / f"linear_dynamic_{case}_r1.dat"
    )
    if not dat.exists():
        return []
    model = build_c3d8_model(case=case, resolution=1, quick=True)
    blocks = _parse_nodal_vectors(dat, model.node_ids, quantity="u")
    return [(float(time), _center_z(model, U)) for time, U in sorted(blocks.items())]


def _trajectory_reference_z(phase9_dir: Path, case: str) -> list[tuple[float, float]]:
    rows = _read_rows(phase9_dir / "c3d8_dynamic_trajectory" / "c3d8_contact_trajectory.csv")
    values = [
        (float(row["time"]), float(row["z_cm"]))
        for row in rows
        if row.get("case") == case and row.get("resolution") == "1" and row.get("z_cm", "") != ""
    ]
    return sorted(values)


def _native_z_history(phase9_dir: Path, filename: str) -> list[tuple[float, float]]:
    rows = _read_rows(phase9_dir / filename)
    return [(float(row["time"]), float(row["z_cm"])) for row in rows if row.get("time", "") and row.get("z_cm", "")]


def _displacement_from_z(history: list[tuple[float, float]], z0: float) -> tuple[np.ndarray, np.ndarray]:
    if not history:
        return np.empty(0), np.empty(0)
    times = np.asarray([time for time, _z in history], dtype=float)
    disp = np.asarray([z - z0 for _time, z in history], dtype=float)
    return times, disp


def generate_dynamic_time_history_plot(phase9_dir: Path) -> dict[str, Path]:
    """Generate dynamic SFC/CalculiX displacement-time curves from Phase-9 CSVs."""
    phase9_dir = Path(phase9_dir)
    figures = phase9_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    errors = _case_error_map(phase9_dir)
    cases = [
        (
            "c3d8_linear_dynamic_block_plane_contact",
            "Linear block-plane",
            "block_plane_c3d8",
            "native_c3d8_linear_dynamic_block_plane.csv",
            "linear",
        ),
        (
            "c3d8_linear_dynamic_block_block_contact",
            "Linear block-block",
            "block_block_c3d8",
            "native_c3d8_linear_dynamic_block_block.csv",
            "linear",
        ),
        (
            "c3d8_nonlinear_dynamic_block_plane_contact",
            "Nonlinear block-plane",
            "block_plane_c3d8",
            "native_c3d8_nonlinear_dynamic_block_plane.csv",
            "nonlinear",
        ),
        (
            "c3d8_nonlinear_dynamic_block_block_contact",
            "Nonlinear block-block",
            "block_block_c3d8",
            "native_c3d8_nonlinear_dynamic_block_block.csv",
            "nonlinear",
        ),
    ]
    available: list[tuple[str, str, str, list[tuple[float, float]], list[tuple[float, float]]]] = []
    curve_rows: list[Row] = []
    for case_id, title, model_case, native_file, mode in cases:
        native = _native_z_history(phase9_dir, native_file)
        if not native:
            continue
        reference = _linear_calculix_reference_z(phase9_dir, model_case) if mode == "linear" else []
        if not reference:
            reference = _trajectory_reference_z(phase9_dir, model_case)
        if not reference:
            continue
        available.append((case_id, title, model_case, native, reference))
        z0 = native[0][1]
        native_by_time = {round(time, 12): z for time, z in native}
        ref_by_time = {round(time, 12): z for time, z in reference}
        for key in sorted(set(native_by_time).intersection(ref_by_time)):
            curve_rows.append(
                {
                    "case_id": case_id,
                    "time_seconds": key,
                    "sfc_z_displacement_mm": native_by_time[key] - z0,
                    "calculix_z_displacement_mm": ref_by_time[key] - z0,
                    "z_cm_l2_rel_error": errors.get(case_id, ""),
                }
            )
    if not available:
        raise FileNotFoundError(f"no dynamic time-history data found under {phase9_dir}")

    ncols = 2
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.4, 3.1 * nrows), squeeze=False)
    for ax in axes.ravel()[len(available) :]:
        ax.set_axis_off()
    for ax, (case_id, title, _model_case, native, reference) in zip(axes.ravel(), available, strict=False):
        z0 = native[0][1]
        t_native, u_native = _displacement_from_z(native, z0)
        t_ref, u_ref = _displacement_from_z(reference, z0)
        err = errors.get(case_id)
        err_label = "" if err is None else f" (z err. {100.0 * err:.2f}%)"
        ax.plot(t_ref, u_ref, color="black", linewidth=1.7, label="CalculiX reference")
        ax.plot(
            t_native,
            u_native,
            color="#4c78a8",
            linewidth=1.7,
            linestyle="--",
            label=f"SFC dynamic SDF{err_label}",
        )
        ax.set_title(title, pad=28)
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.28)
        ax.legend(
            fontsize=7.4,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            frameon=False,
            ncol=1,
            handlelength=2.0,
        )
    axes[0, 0].set_ylabel(r"$z$ displacement (mm)")
    if nrows > 1:
        axes[1, 0].set_ylabel(r"$z$ displacement (mm)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=1.5, h_pad=2.2)
    png = figures / "dynamic_displacement_time_histories.png"
    pdf = figures / "dynamic_displacement_time_histories.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    curves_csv = phase9_dir / "dynamic_displacement_time_histories.csv"
    _write_rows(curves_csv, curve_rows)
    return {"png": png, "pdf": pdf, "csv": curves_csv}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase9-dir",
        type=Path,
        default=ROOT / "paper" / "numerical_experiments" / "phase9_full_contact_validation",
    )
    return parser.parse_args()


def main() -> int:
    outputs = generate_dynamic_time_history_plot(parse_args().phase9_dir)
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
