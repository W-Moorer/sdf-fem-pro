"""Compare rigid-plane contact and dynamic-SDF contact queries.

This validation isolates the project-specific contact query. Both histories
use the same clean-room CalculiX-aligned mechanics backend. The only intended
change is the contact geometry provider:

- `plane`: analytic rigid-plane gap.
- `dynamic_sdf_plane`: the same rigid plane represented as current surface
  triangles and queried through the dynamic FEM-induced SDF API.
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

from validation.run_geometric_nonlinear_contact_validation import (  # noqa: E402
    _contact_model,
    _first_contact_time,
    run_sfc_geometric_contact_history,
)

Row = dict[str, Any]

COMPARE_FIELDS = [
    "z_cm",
    "v_cm_z",
    "min_gap",
    "quadrature_min_gap",
    "max_penetration",
    "active_contact_count",
    "normal_force_proxy",
    "contact_energy_proxy",
    "total_mechanical_energy_proxy",
]


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_equivalence(out_dir: Path, *, quick: bool = False) -> dict[str, Path]:
    """Run contact-query equivalence validation and write artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    duration = 0.12 if quick else 0.14
    dt = 0.004 if quick else 0.002
    resolutions = [1] if quick else [1, 2, 3]
    history_rows: list[Row] = []
    metric_rows: list[Row] = []
    for resolution in resolutions:
        model = _contact_model(resolution=resolution, duration=duration, dt=dt)
        plane_rows, _, _ = run_sfc_geometric_contact_history(
            model,
            contact_mode="plane",
            source="sfc_plane_contact",
        )
        sdf_rows, _, _ = run_sfc_geometric_contact_history(
            model,
            contact_mode="dynamic_sdf_plane",
            source="sfc_dynamic_sdf_plane_contact",
        )
        history_rows.extend(plane_rows)
        history_rows.extend(sdf_rows)
        metric_rows.append(_compare_histories(model.case, resolution, dt, plane_rows, sdf_rows))

    claim_supported = all(row["status"] == "passed" for row in metric_rows)
    claim_rows = [
        {
            "claim": "same_backend_dynamic_sdf_query_matches_analytic_plane",
            "supported": str(claim_supported).lower(),
            "evidence_csv": "contact_query_equivalence_metrics.csv",
            "details": "Both histories use identical mechanics; only contact geometry provider changes.",
        }
    ]
    outputs = {
        "history": out_dir / "contact_query_equivalence_history.csv",
        "metrics": out_dir / "contact_query_equivalence_metrics.csv",
        "claims": out_dir / "contact_query_equivalence_claims.csv",
        "summary": out_dir / "contact_query_equivalence_summary.md",
    }
    _write_csv(outputs["history"], history_rows)
    _write_csv(outputs["metrics"], metric_rows)
    _write_csv(outputs["claims"], claim_rows)
    _write_markdown(outputs["summary"], metric_rows, claim_rows, quick=quick)
    return outputs


def _compare_histories(case: str, resolution: int, dt: float, plane_rows: list[Row], sdf_rows: list[Row]) -> Row:
    row: Row = {
        "case": case,
        "resolution": resolution,
        "dt": dt,
        "contact_activation_time_plane": _first_contact_time(plane_rows),
        "contact_activation_time_dynamic_sdf": _first_contact_time(sdf_rows),
    }
    activation_plane = row["contact_activation_time_plane"]
    activation_sdf = row["contact_activation_time_dynamic_sdf"]
    row["contact_activation_abs_error"] = (
        ""
        if activation_plane is None or activation_sdf is None
        else abs(float(activation_sdf) - float(activation_plane))
    )
    max_abs = 0.0
    max_rel = 0.0
    for field in COMPARE_FIELDS:
        abs_error, rel_error = _series_errors(plane_rows, sdf_rows, field)
        row[f"{field}_max_abs_error"] = abs_error
        row[f"{field}_max_rel_error"] = rel_error
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
    row["max_field_abs_error"] = max_abs
    row["max_field_rel_error"] = max_rel
    row["status"] = "passed" if max_abs <= 1.0e-8 and max_rel <= 1.0e-8 else "failed"
    return row


def _series_errors(reference_rows: list[Row], candidate_rows: list[Row], field: str) -> tuple[float, float]:
    ref_times = np.asarray([float(row["time"]) for row in reference_rows], dtype=float)
    cand_times = np.asarray([float(row["time"]) for row in candidate_rows], dtype=float)
    ref_values = np.asarray([float(row[field]) for row in reference_rows], dtype=float)
    cand_values = np.interp(ref_times, cand_times, np.asarray([float(row[field]) for row in candidate_rows], dtype=float))
    diff = cand_values - ref_values
    abs_error = float(np.max(np.abs(diff))) if diff.size else 0.0
    rel_error = abs_error / max(float(np.max(np.abs(ref_values))) if ref_values.size else 0.0, 1.0e-30)
    return abs_error, rel_error


def _write_markdown(path: Path, metric_rows: list[Row], claim_rows: list[Row], *, quick: bool) -> None:
    command = "python validation/run_contact_query_equivalence.py"
    if quick:
        command += " --quick"
    command += f" --out-dir {path.parent.as_posix()}"
    lines = [
        "# Contact Query Equivalence",
        "",
        "This validation runs the same clean-room CalculiX-aligned mechanics backend with two contact geometry providers.",
        "The analytic rigid-plane provider is compared against the dynamic FEM-induced SDF provider using the same rigid plane represented by current-surface triangles.",
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Claim Gates",
        "",
        "| Claim | Supported | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in claim_rows:
        lines.append(f"| {row['claim']} | {row['supported']} | `{row['evidence_csv']}` |")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Resolution | Status | activation abs. err. | max field abs. err. | max field rel. err. |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in metric_rows:
        lines.append(
            f"| {row['resolution']} | {row['status']} | {_fmt(row['contact_activation_abs_error'])} | "
            f"{_fmt(row['max_field_abs_error'])} | {_fmt(row['max_field_rel_error'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A passing row means the dynamic-SDF contact geometry reproduces the analytic plane-contact history when the master surface is exactly the same plane and the mechanics backend is unchanged.",
            "This does not validate arbitrary curved or deformable master surfaces; it isolates the contact-query substitution in a controlled case.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.6e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "contact_query_equivalence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_equivalence(args.out_dir, quick=bool(args.quick))
    print("Contact query equivalence validation complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
