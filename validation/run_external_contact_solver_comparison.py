"""External contact-solver comparison using SfePy.

This validation runner calls SfePy's built-in two-body penalty contact example
as an external open-source contact solver, then re-evaluates the final deformed
contact state with the SFC true dynamic narrow-band SDF field query.

The comparison is intentionally scoped: SfePy performs the nonlinear contact
solve, while SFC builds a current-space dynamic SDF field from the SfePy final
master surface and evaluates gap signs and magnitudes by interpolation. This is
external contact-solver evidence for the contact geometry layer, not a claim
that the current SFC package implements a full nonlinear contact equilibrium
solver.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
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

from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402
from validation.run_phase4_paper_validation import _format_float  # noqa: E402

Row = dict[str, Any]


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(command: list[str], *, cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=timeout)


def _sfepy_available() -> bool:
    if shutil.which("sfepy-run") is None or shutil.which("py") is None:
        return False
    try:
        _run(["py", "-3.12", "-c", "import sfepy"], cwd=ROOT, timeout=20)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _sfepy_metadata() -> tuple[str, Path]:
    code = (
        "import sfepy, pathlib; "
        "print(sfepy.__version__); "
        "print(pathlib.Path(sfepy.__file__).resolve().parent / "
        "'examples' / 'linear_elasticity' / 'two_bodies_contact.py')"
    )
    proc = _run(["py", "-3.12", "-c", code], cwd=ROOT, timeout=30)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError("failed to discover SfePy version and example path")
    return lines[0], Path(lines[1])


def _run_sfepy_case(example_path: Path, run_dir: Path, approach: float) -> tuple[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    define = (
        f"output_dir='{run_dir.as_posix()}', "
        "shape0=(2,2,2), shape1=(2,2,2), "
        "shift10=(0.0,0.0,1e-4), "
        f"shift11=(0.0,0.0,{-float(approach):.16g}), "
        "n_step=2, contact='builtin', verbose=False"
    )
    proc = _run(["sfepy-run", str(example_path), "-d", define], cwd=ROOT, timeout=120)
    return proc.stdout, proc.stderr


def _triangulated_lower_contact_surface(coors: np.ndarray, conn: np.ndarray, mat_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower_cells = conn[mat_id == 0]
    lower_nodes = np.unique(lower_cells.ravel())
    top_z = float(np.max(coors[lower_nodes, 2]))
    triangles: list[list[int]] = []
    for cell in lower_cells:
        top = np.asarray([cell[4], cell[5], cell[6], cell[7]], dtype=np.int64)
        if np.all(np.isclose(coors[top, 2], top_z)):
            triangles.append([int(top[0]), int(top[1]), int(top[2])])
            triangles.append([int(top[0]), int(top[2]), int(top[3])])
    return np.asarray(triangles, dtype=np.int64), lower_nodes


def _upper_contact_centroids(coors: np.ndarray, conn: np.ndarray, mat_id: np.ndarray, current: np.ndarray) -> np.ndarray:
    upper_cells = conn[mat_id == 1]
    upper_nodes = np.unique(upper_cells.ravel())
    bottom_z = float(np.min(coors[upper_nodes, 2]))
    centroids: list[np.ndarray] = []
    for cell in upper_cells:
        bottom = np.asarray([cell[0], cell[1], cell[2], cell[3]], dtype=np.int64)
        if np.all(np.isclose(coors[bottom, 2], bottom_z)):
            centroids.append(np.mean(current[bottom], axis=0))
    return np.asarray(centroids, dtype=float)


def _evaluate_sfc_gaps_on_sfepy_state(h5_path: Path) -> dict[str, float | int | str]:
    with h5py.File(h5_path, "r") as f:
        coors = np.asarray(f["mesh/coors"][...], dtype=float)
        conn = np.asarray(f["mesh/group0/conn"][...], dtype=np.int64)
        mat_id = np.asarray(f["mesh/group0/mat_id"][...], dtype=np.int64)
        last_step = int(np.asarray(f["last_step"][...]).ravel()[0])
        step_name = f"step{last_step}"
        u = np.asarray(f[f"{step_name}/__u/data"][...], dtype=float)
        sfepy_gap = np.asarray(f[f"{step_name}/__gap/data"][...], dtype=float).ravel()

    current = coors + u
    lower_faces, _ = _triangulated_lower_contact_surface(coors, conn, mat_id)
    query_points = _upper_contact_centroids(coors, conn, mat_id, current)
    spacing = 0.08
    band_radius = 0.30
    sdf = DynamicNarrowBandSDF.build_required_points(
        current,
        lower_faces,
        query_points,
        spacing=spacing,
        band_radius=band_radius,
        padding=band_radius,
        cell_size=0.16,
    )
    sfc_gaps = np.asarray([sdf.query_phi(point) for point in query_points], dtype=float)
    gradient_norms = np.asarray([np.linalg.norm(sdf.query_gradient(point)) for point in query_points], dtype=float)
    return {
        "sfepy_gap_min": float(np.min(sfepy_gap)) if sfepy_gap.size else 0.0,
        "sfepy_gap_mean": float(np.mean(sfepy_gap)) if sfepy_gap.size else 0.0,
        "sfepy_active_count": int(np.count_nonzero(sfepy_gap < 0.0)),
        "sfc_replay_gap_min": float(np.min(sfc_gaps)) if sfc_gaps.size else 0.0,
        "sfc_replay_gap_mean": float(np.mean(sfc_gaps)) if sfc_gaps.size else 0.0,
        "sfc_replay_active_count": int(np.count_nonzero(sfc_gaps < 0.0)),
        "sfc_query_count": int(sfc_gaps.size),
        "gap_mean_abs_difference": float(abs((np.mean(sfc_gaps) if sfc_gaps.size else 0.0) - (np.mean(sfepy_gap) if sfepy_gap.size else 0.0))),
        "displacement_l2_norm": float(np.linalg.norm(u)),
        "field_population_mode": str(sdf.grid.metadata.get("population_mode", "full")),
        "field_valid_node_count": int(sdf.stats.valid_node_count),
        "field_update_seconds": float(sdf.stats.update_seconds),
        "field_gradient_norm_min": float(np.min(gradient_norms)) if gradient_norms.size else 0.0,
        "field_gradient_norm_max": float(np.max(gradient_norms)) if gradient_norms.size else 0.0,
    }


def external_contact_solver_comparison(out_dir: Path, approaches: list[float]) -> tuple[list[Row], list[Row]]:
    if not _sfepy_available():
        raise SystemExit("SfePy external contact solver is not available. Install SfePy or provide sfepy-run.")

    version, example_path = _sfepy_metadata()
    rows: list[Row] = []
    command_rows: list[Row] = []
    for approach in approaches:
        case_dir = out_dir / "sfepy_runs" / f"approach_{float(approach):.5f}".replace(".", "p")
        stdout, stderr = _run_sfepy_case(example_path, case_dir, float(approach))
        h5_path = case_dir / "two_bodies.h5"
        metrics = _evaluate_sfc_gaps_on_sfepy_state(h5_path)
        sfepy_active = int(metrics["sfepy_active_count"]) > 0
        sfc_active = int(metrics["sfc_replay_active_count"]) > 0
        sign_agreement = sfepy_active == sfc_active
        gap_diff = float(metrics["gap_mean_abs_difference"])
        status = "ok" if sign_agreement and gap_diff < 1.5e-2 else "check"
        rows.append(
            {
                "case": "sfepy_two_bodies_contact_replay",
                "external_solver": "SfePy",
                "external_solver_version": version,
                "approach": float(approach),
                "sfepy_gap_min": metrics["sfepy_gap_min"],
                "sfepy_gap_mean": metrics["sfepy_gap_mean"],
                "sfepy_active_count": metrics["sfepy_active_count"],
                "sfc_replay_gap_min": metrics["sfc_replay_gap_min"],
                "sfc_replay_gap_mean": metrics["sfc_replay_gap_mean"],
                "sfc_replay_active_count": metrics["sfc_replay_active_count"],
                "sfc_query_count": metrics["sfc_query_count"],
                "gap_mean_abs_difference": gap_diff,
                "displacement_l2_norm": metrics["displacement_l2_norm"],
                "field_population_mode": metrics["field_population_mode"],
                "field_valid_node_count": metrics["field_valid_node_count"],
                "field_update_seconds": metrics["field_update_seconds"],
                "field_gradient_norm_min": metrics["field_gradient_norm_min"],
                "field_gradient_norm_max": metrics["field_gradient_norm_max"],
                "contact_state_agreement": str(sign_agreement).lower(),
                "status": status,
                "details": "SfePy performs nonlinear two-body penalty contact solve; SFC replays final geometry with true dynamic narrow-band SDF interpolation queries",
            }
        )
        command_rows.append(
            {
                "approach": float(approach),
                "command": f"sfepy-run {example_path.name} -d \"output_dir='{case_dir.as_posix()}', ...\"",
                "stdout_tail": "\n".join(stdout.splitlines()[-8:]),
                "stderr_tail": "\n".join(stderr.splitlines()[-8:]),
                "h5_output": str(h5_path.relative_to(out_dir)),
                "log_output": str((case_dir / "output_log.txt").relative_to(out_dir)),
            }
        )
    return rows, command_rows


def claim_rows(rows: list[Row]) -> list[Row]:
    max_gap_diff = max(float(row["gap_mean_abs_difference"]) for row in rows)
    all_signs = all(row["contact_state_agreement"] == "true" for row in rows)
    active_rows = [row for row in rows if float(row["approach"]) > 0.0]
    active_supported = any(int(row["sfepy_active_count"]) > 0 and int(row["sfc_replay_active_count"]) > 0 for row in active_rows)
    return [
        {
            "claim_id": "external_contact_solver_state_agreement",
            "claim_text": "SFC true dynamic SDF field queries agree in contact activation with SfePy external two-body contact solve final states.",
            "evidence_csv": "external_contact_solver_comparison.csv",
            "evidence_field": "contact_state_agreement",
            "gate_value": str(all_signs).lower(),
            "claim_status": "supported" if all_signs and active_supported else "not_supported",
            "details": "activation sign agreement on SfePy final deformed contact states",
        },
        {
            "claim_id": "external_contact_gap_scale_check",
            "claim_text": "SFC replayed gap magnitude remains within the configured tolerance of SfePy exported contact gap averages.",
            "evidence_csv": "external_contact_solver_comparison.csv",
            "evidence_field": "gap_mean_abs_difference",
            "gate_value": max_gap_diff,
            "claim_status": "supported" if max_gap_diff < 1.5e-2 else "not_supported",
            "details": "SfePy gap is a contact-term surface average; SFC gap is a centroid true-field interpolation replay",
        },
    ]


def write_plots(out_dir: Path, rows: list[Row]) -> list[Row]:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = [float(row["approach"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(x, [float(row["sfepy_gap_mean"]) for row in rows], marker="o", label="SfePy exported gap")
    ax.plot(x, [float(row["sfc_replay_gap_mean"]) for row in rows], marker="s", label="SFC replay gap")
    ax.axhline(0.0, color="0.35", linewidth=1.0)
    ax.set_xlabel("prescribed top approach")
    ax.set_ylabel("mean contact gap")
    ax.legend()
    fig.tight_layout()
    png = out_dir / "external_contact_gap_comparison.png"
    pdf = out_dir / "external_contact_gap_comparison.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(x, [int(row["sfepy_active_count"]) for row in rows], marker="o", label="SfePy active facets")
    ax.plot(x, [int(row["sfc_replay_active_count"]) for row in rows], marker="s", label="SFC active queries")
    ax.set_xlabel("prescribed top approach")
    ax.set_ylabel("active contact count")
    ax.legend()
    fig.tight_layout()
    png2 = out_dir / "external_contact_active_count.png"
    pdf2 = out_dir / "external_contact_active_count.pdf"
    fig.savefig(png2, dpi=180)
    fig.savefig(pdf2)
    plt.close(fig)

    return [
        {"plot": "external_contact_gap_comparison", "png": png.name, "pdf": pdf.name, "status": "ok"},
        {"plot": "external_contact_active_count", "png": png2.name, "pdf": pdf2.name, "status": "ok"},
    ]


def write_markdown(path: Path, rows: list[Row], claims: list[Row], plots: list[Row]) -> None:
    lines = [
        "# External Contact Solver Comparison Summary",
        "",
        "This validation uses SfePy as an external open-source nonlinear penalty contact solver. SfePy solves its built-in two-body contact problem; SFC then rebuilds a true current-space dynamic narrow-band SDF field from the final master surface and replays the deformed geometry by field interpolation.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python validation/run_external_contact_solver_comparison.py --out-dir results/external_contact_solver",
        "```",
        "",
        "Quick mode:",
        "",
        "```bash",
        "python validation/run_external_contact_solver_comparison.py --quick --out-dir results/external_contact_solver",
        "```",
        "",
        "## Claims",
        "",
        "| Claim | Status | Evidence | Gate value |",
        "| --- | --- | --- | ---: |",
    ]
    for claim in claims:
        lines.append(f"| {claim['claim_id']} | {claim['claim_status']} | {claim['evidence_csv']}::{claim['evidence_field']} | {claim['gate_value']} |")
        lines.append(f"<!-- evidence csv={claim['evidence_csv']} field={claim['evidence_field']} -->")

    lines.extend(["", "## Gap Replay Metrics", "", "| Approach | SfePy mean gap | SFC replay mean gap | Abs. diff. | SfePy active | SFC active | Status |", "| ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in rows:
        lines.append(
            f"| {_format_float(row['approach'])} | {_format_float(row['sfepy_gap_mean'])} | {_format_float(row['sfc_replay_gap_mean'])} | {_format_float(row['gap_mean_abs_difference'])} | {row['sfepy_active_count']} | {row['sfc_replay_active_count']} | {row['status']} |"
        )

    lines.extend(["", "## Plots", ""])
    for plot in plots:
        lines.append(f"- `{plot['png']}` and `{plot['pdf']}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- SfePy provides the external nonlinear two-body penalty contact solve.",
            "- SFC is evaluated on the final SfePy deformed geometry, so the comparison checks true-field gap consistency against an external contact state.",
            "- The compared gap values are not expected to be bitwise identical: SfePy reports contact-term surface averages, while SFC reports centroid dynamic-SDF interpolation gaps.",
            "- This evidence does not claim that SFC currently implements a full nonlinear contact equilibrium solver.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use reduced deterministic approaches for CI.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "external_contact_solver")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    approaches = [0.0, 0.06, 0.10] if args.quick else [0.0, 0.04, 0.06, 0.08, 0.10]
    rows, command_rows = external_contact_solver_comparison(args.out_dir, approaches)
    claims = claim_rows(rows)
    plots = write_plots(args.out_dir, rows)
    _write_csv(
        args.out_dir / "external_contact_solver_comparison.csv",
        [
            "case",
            "external_solver",
            "external_solver_version",
            "approach",
            "sfepy_gap_min",
            "sfepy_gap_mean",
            "sfepy_active_count",
            "sfc_replay_gap_min",
            "sfc_replay_gap_mean",
            "sfc_replay_active_count",
            "sfc_query_count",
            "gap_mean_abs_difference",
            "displacement_l2_norm",
            "field_population_mode",
            "field_valid_node_count",
            "field_update_seconds",
            "field_gradient_norm_min",
            "field_gradient_norm_max",
            "contact_state_agreement",
            "status",
            "details",
        ],
        rows,
    )
    _write_csv(
        args.out_dir / "external_contact_solver_commands.csv",
        ["approach", "command", "stdout_tail", "stderr_tail", "h5_output", "log_output"],
        command_rows,
    )
    _write_csv(
        args.out_dir / "external_contact_solver_claims.csv",
        ["claim_id", "claim_text", "evidence_csv", "evidence_field", "gate_value", "claim_status", "details"],
        claims,
    )
    _write_csv(args.out_dir / "external_contact_solver_plots.csv", ["plot", "png", "pdf", "status"], plots)
    write_markdown(args.out_dir / "external_contact_solver_summary.md", rows, claims, plots)
    print(f"Wrote external contact solver comparison outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
