"""Diagnose SFC vs Abaqus contact-status alignment for the source gear case.

The script is intentionally a validation/postprocessing tool.  It reads SFC
history CSV output and Abaqus validation outputs, but it is not imported by the
core solver.  Its purpose is to localize mismatches in Abaqus-style
surface-to-surface contact status before changing solver theory.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SFC_HISTORY = ROOT / "results" / "source_gear_abq_dt_current_settings_00040" / "sfc_full_gear_lagrangian_sdf_history.csv"
DEFAULT_ABAQUS_MANIFEST = (
    ROOT
    / "results"
    / "source_gear_abaqus_penalty_full_stride2_match_step_0004"
    / "abaqus_vtk_contact_full_check"
    / "abaqus_contact_full_manifest.csv"
)
DEFAULT_ABAQUS_STA = (
    ROOT
    / "results"
    / "source_gear_abaqus_penalty_full_stride2_match_step_0004"
    / "abaqus_run"
    / "gear_contact_source_penalty.sta"
)
DEFAULT_OUT_DIR = ROOT / "results" / "source_gear_contact_status_diagnostics"


Row = dict[str, Any]


def _read_csv(path: Path) -> list[Row]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: Row, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _relative_error(value: float, reference: float, *, reference_floor: float = 1.0e-12) -> float | str:
    """Return a relative error only when the reference magnitude is meaningful."""

    ref = abs(float(reference))
    val = abs(float(value))
    floor = float(reference_floor)
    if ref < floor:
        return 0.0 if val < floor else ""
    return abs(float(value) - float(reference)) / ref


def parse_abaqus_sta_increments(path: Path) -> list[Row]:
    """Parse Abaqus/Standard ``.sta`` increment rows."""

    increments: list[Row] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split()
        if len(parts) < 9:
            continue
        try:
            step = int(parts[0])
            inc = int(parts[1])
            attempt = int(parts[2])
            severe = int(parts[3])
            equilibrium = int(parts[4])
            total_iters = int(parts[5])
            total_time = float(parts[6])
            step_time = float(parts[7])
            increment_size = float(parts[8])
        except ValueError:
            continue
        increments.append(
            {
                "step": step,
                "increment": inc,
                "attempt": attempt,
                "severe_discontinuity_iterations": severe,
                "equilibrium_iterations": equilibrium,
                "total_iterations": total_iters,
                "total_time": total_time,
                "step_time": step_time,
                "increment_size": increment_size,
            }
        )
    return increments


_MSG_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def _parse_msg_float(text: str) -> float:
    """Parse an Abaqus numeric field while preserving signed contact errors."""

    match = re.search(_MSG_FLOAT, text)
    if match is None:
        return 0.0
    return float(match.group(0))


def parse_abaqus_msg_contact_convergence(path: Path) -> list[Row]:
    """Parse Abaqus/Standard contact convergence diagnostics from ``.msg``.

    Abaqus reports surface-contact convergence separately from the global force
    residual: contact status changes appear as severe-discontinuity iterations,
    and penetration/contact-force compatibility is then checked in subsequent
    equilibrium iterations.  The parser keeps only per-increment scalar metrics
    so validation scripts can compare SFC against the same acceptance semantics
    without importing Abaqus into the core solver.
    """

    rows: list[Row] = []
    current: Row | None = None
    current_kind = ""
    current_iter = 0

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        severe_pen = current.get("severe_max_abs_penetration_error", "")
        final_pen = current.get("final_abs_penetration_error", "")
        severe_force = current.get("severe_max_abs_contact_force_error", "")
        final_force = current.get("final_abs_contact_force_error", "")
        if severe_pen != "" and final_pen != "":
            current["penetration_error_reduction"] = float(final_pen) / max(float(severe_pen), 1.0e-30)
        if severe_force != "" and final_force != "":
            current["contact_force_error_reduction"] = float(final_force) / max(float(severe_force), 1.0e-30)
        rows.append(current)
        current = None

    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        inc_match = re.match(r"INCREMENT\s+(\d+)\s+STARTS\.", line)
        if inc_match:
            finish()
            current = {
                "increment": int(inc_match.group(1)),
                "severe_discontinuity_iterations_msg": 0,
                "equilibrium_iterations_msg": 0,
                "open_to_closed_points": 0,
                "closed_to_open_points": 0,
                "contact_converged_checks": 0,
                "severe_max_abs_penetration_error": "",
                "severe_max_abs_contact_force_error": "",
                "final_abs_penetration_error": "",
                "final_abs_contact_force_error": "",
            }
            current_kind = ""
            current_iter = 0
            continue
        if current is None:
            continue
        match = re.match(r"(\d+)\s+POINTS?\s+CHANGED\s+FROM\s+OPEN\s+TO\s+CLOSED", line)
        if match:
            current["open_to_closed_points"] = int(current.get("open_to_closed_points", 0)) + int(match.group(1))
            continue
        match = re.match(r"(\d+)\s+POINTS?\s+CHANGED\s+FROM\s+CLOSED\s+TO\s+OPEN", line)
        if match:
            current["closed_to_open_points"] = int(current.get("closed_to_open_points", 0)) + int(match.group(1))
            continue
        match = re.match(r"CONVERGENCE CHECKS FOR (SEVERE DISCONTINUITY|EQUILIBRIUM) ITERATION\s+(\d+)", line)
        if match:
            current_kind = "severe" if match.group(1).startswith("SEVERE") else "equilibrium"
            current_iter = int(match.group(2))
            if current_kind == "severe":
                current["severe_discontinuity_iterations_msg"] = max(
                    int(current.get("severe_discontinuity_iterations_msg", 0)), current_iter
                )
            else:
                current["equilibrium_iterations_msg"] = max(int(current.get("equilibrium_iterations_msg", 0)), current_iter)
            continue
        if line.startswith("MAX. PENETRATION ERROR"):
            value = _parse_msg_float(line.replace("MAX. PENETRATION ERROR", "", 1))
            abs_value = abs(float(value))
            if current_kind == "severe":
                prior = current.get("severe_max_abs_penetration_error", "")
                current["severe_max_abs_penetration_error"] = max(float(prior), abs_value) if prior != "" else abs_value
            elif current_kind == "equilibrium":
                current["final_abs_penetration_error"] = abs_value
            continue
        if line.startswith("MAX. CONTACT FORCE ERROR"):
            value = _parse_msg_float(line.replace("MAX. CONTACT FORCE ERROR", "", 1))
            abs_value = abs(float(value))
            if current_kind == "severe":
                prior = current.get("severe_max_abs_contact_force_error", "")
                current["severe_max_abs_contact_force_error"] = max(float(prior), abs_value) if prior != "" else abs_value
            elif current_kind == "equilibrium":
                current["final_abs_contact_force_error"] = abs_value
            continue
        if "THE CONTACT CONSTRAINTS HAVE CONVERGED" in line:
            current["contact_converged_checks"] = int(current.get("contact_converged_checks", 0)) + 1
            continue
        match = re.match(r"TIME INCREMENT COMPLETED\s+(" + _MSG_FLOAT + r").*TOTAL TIME COMPLETED\s+(" + _MSG_FLOAT + r")", line)
        if match:
            current["increment_size_msg"] = float(match.group(1))
            current["total_time_msg"] = float(match.group(2))
            continue
        match = re.match(r"TIME INCREMENT COMPLETED\s+(" + _MSG_FLOAT + r")", line)
        if match:
            current["increment_size_msg"] = float(match.group(1))
            continue
        match = re.match(r"STEP TIME COMPLETED\s+(" + _MSG_FLOAT + r").*TOTAL TIME COMPLETED\s+(" + _MSG_FLOAT + r")", line)
        if match:
            current["step_time_msg"] = float(match.group(1))
            current["total_time_msg"] = float(match.group(2))
            continue
        match = re.match(r"ITERATION SUMMARY FOR THE INCREMENT:\s+(\d+)\s+TOTAL ITERATIONS", line)
        if match:
            current["total_iterations_msg"] = int(match.group(1))
            continue
    finish()
    return rows


def _nearest_by_time(rows: list[Row], time_value: float) -> Row:
    if not rows:
        return {}
    return min(rows, key=lambda row: abs(_float(row, "time") - float(time_value)))


def _nearest_increment(increments: list[Row], time_value: float) -> Row:
    if not increments:
        return {}
    return min(increments, key=lambda row: abs(float(row["total_time"]) - float(time_value)))


def _nearest_msg_increment(msg_rows: list[Row], time_value: float) -> Row:
    values = [row for row in msg_rows if row.get("total_time_msg", "") != ""]
    if not values:
        return {}
    return min(values, key=lambda row: abs(float(row["total_time_msg"]) - float(time_value)))


def build_contact_status_diagnostics(
    *,
    sfc_history: Path,
    abaqus_manifest: Path,
    abaqus_sta: Path | None = None,
) -> list[Row]:
    """Return per-output-frame contact-status mismatch diagnostics."""

    sfc_rows = _read_csv(Path(sfc_history))
    abaqus_rows = _read_csv(Path(abaqus_manifest))
    increments = parse_abaqus_sta_increments(Path(abaqus_sta)) if abaqus_sta is not None and Path(abaqus_sta).exists() else []
    msg_path = Path(abaqus_sta).with_suffix(".msg") if abaqus_sta is not None else None
    msg_rows = parse_abaqus_msg_contact_convergence(msg_path) if msg_path is not None and msg_path.exists() else []
    diagnostics: list[Row] = []
    for sfc in sfc_rows:
        time_value = _float(sfc, "time")
        abaqus = _nearest_by_time(abaqus_rows, time_value)
        inc = _nearest_increment(increments, time_value)
        msg = _nearest_msg_increment(msg_rows, time_value)
        sfc_active = _float(sfc, "active_contact_node_count")
        abaqus_active = _float(abaqus, "active_contact_node_count")
        sfc_p95_stress = _float(sfc, "p95_von_mises_nodeavg")
        abaqus_p95_stress = _float(abaqus, "p95_von_mises_nodeavg")
        sfc_p95_strain = _float(sfc, "p95_equivalent_elastic_strain_nodeavg")
        abaqus_p95_strain = _float(abaqus, "p95_equivalent_elastic_strain_nodeavg")
        active_delta = sfc_active - abaqus_active
        sfc_pressure = _float(sfc, "max_contact_pressure_nodeavg")
        abaqus_pressure = _float(abaqus, "max_contact_pressure_nodeavg")
        sfc_mean_pressure = _float(sfc, "mean_active_contact_pressure_nodeavg")
        abaqus_mean_pressure = _float(abaqus, "mean_active_contact_pressure_nodeavg")
        sfc_min_gap = _float(sfc, "min_contact_gap_node", _float(sfc, "min_gap"))
        abaqus_min_gap = _float(abaqus, "min_contact_gap_node")
        row = {
            "time": time_value,
            "abaqus_nearest_time": _float(abaqus, "time"),
            "abaqus_increment_size": inc.get("increment_size", ""),
            "abaqus_total_iterations": inc.get("total_iterations", ""),
            "abaqus_severe_discontinuity_iterations": inc.get("severe_discontinuity_iterations", ""),
            "abaqus_msg_open_to_closed_points": msg.get("open_to_closed_points", ""),
            "abaqus_msg_closed_to_open_points": msg.get("closed_to_open_points", ""),
            "abaqus_msg_contact_converged_checks": msg.get("contact_converged_checks", ""),
            "abaqus_msg_final_abs_penetration_error": msg.get("final_abs_penetration_error", ""),
            "abaqus_msg_final_abs_contact_force_error": msg.get("final_abs_contact_force_error", ""),
            "abaqus_msg_penetration_error_reduction": msg.get("penetration_error_reduction", ""),
            "abaqus_msg_contact_force_error_reduction": msg.get("contact_force_error_reduction", ""),
            "sfc_newton_iterations": _float(sfc, "newton_iterations"),
            "sfc_newton_residual_norm": _float(sfc, "newton_residual_norm"),
            "sfc_active_contact_node_count": sfc_active,
            "abaqus_active_contact_node_count": abaqus_active,
            "active_contact_node_count_delta": active_delta,
            "active_contact_node_count_rel_delta": abs(active_delta) / max(abs(abaqus_active), 1.0),
            "sfc_max_contact_pressure_nodeavg": sfc_pressure,
            "abaqus_max_contact_pressure_nodeavg": abaqus_pressure,
            "max_contact_pressure_nodeavg_abs_error": abs(sfc_pressure - abaqus_pressure),
            "max_contact_pressure_nodeavg_rel_error": _relative_error(
                sfc_pressure,
                abaqus_pressure,
                reference_floor=1.0e-9,
            ),
            "sfc_mean_active_contact_pressure_nodeavg": sfc_mean_pressure,
            "abaqus_mean_active_contact_pressure_nodeavg": abaqus_mean_pressure,
            "mean_active_contact_pressure_nodeavg_abs_error": abs(sfc_mean_pressure - abaqus_mean_pressure),
            "mean_active_contact_pressure_nodeavg_rel_error": _relative_error(
                sfc_mean_pressure,
                abaqus_mean_pressure,
                reference_floor=1.0e-9,
            ),
            "sfc_min_contact_gap_node": sfc_min_gap,
            "abaqus_min_contact_gap_node": abaqus_min_gap,
            "min_contact_gap_node_abs_error": abs(sfc_min_gap - abaqus_min_gap),
            "min_contact_gap_node_rel_error": _relative_error(sfc_min_gap, abaqus_min_gap, reference_floor=1.0e-12),
            "sfc_p95_von_mises_nodeavg": sfc_p95_stress,
            "abaqus_p95_von_mises_nodeavg": abaqus_p95_stress,
            "p95_von_mises_nodeavg_rel_error": abs(sfc_p95_stress - abaqus_p95_stress) / max(abs(abaqus_p95_stress), 1.0e-30),
            "sfc_p95_equivalent_elastic_strain_nodeavg": sfc_p95_strain,
            "abaqus_p95_equivalent_elastic_strain_nodeavg": abaqus_p95_strain,
            "p95_equivalent_elastic_strain_nodeavg_rel_error": abs(sfc_p95_strain - abaqus_p95_strain) / max(abs(abaqus_p95_strain), 1.0e-30),
            "release_stage_mismatch": int(abaqus_active <= 50.0 and sfc_active > 100.0 and abaqus_pressure <= 1.0e-12),
        }
        diagnostics.append(row)
    return diagnostics


def _max_row(rows: list[Row], key: str) -> Row | None:
    values = [row for row in rows if row.get(key, "") != ""]
    if not values:
        return None
    return max(values, key=lambda row: _float(row, key))


def write_summary(path: Path, rows: list[Row], *, sfc_history: Path, abaqus_manifest: Path, abaqus_sta: Path | None) -> None:
    p95_row = _max_row(rows, "p95_von_mises_nodeavg_rel_error")
    active_row = _max_row(rows, "active_contact_node_count_rel_delta")
    max_pressure_row = _max_row(rows, "max_contact_pressure_nodeavg_rel_error")
    mean_pressure_row = _max_row(rows, "mean_active_contact_pressure_nodeavg_rel_error")
    min_gap_row = _max_row(rows, "min_contact_gap_node_rel_error")
    release_rows = [row for row in rows if int(row.get("release_stage_mismatch", 0))]
    increments = parse_abaqus_sta_increments(abaqus_sta) if abaqus_sta is not None and abaqus_sta.exists() else []
    msg_path = Path(abaqus_sta).with_suffix(".msg") if abaqus_sta is not None else None
    msg_rows = parse_abaqus_msg_contact_convergence(msg_path) if msg_path is not None and msg_path.exists() else []
    increment_sizes = sorted({float(row["increment_size"]) for row in increments}) if increments else []
    severe_rows = [row for row in msg_rows if int(row.get("severe_discontinuity_iterations_msg", 0)) > 0]
    converged_contact_rows = [row for row in msg_rows if int(row.get("contact_converged_checks", 0)) > 0]
    max_final_pen = _max_row(
        [row for row in msg_rows if row.get("final_abs_penetration_error", "") != ""],
        "final_abs_penetration_error",
    )
    max_final_force = _max_row(
        [row for row in msg_rows if row.get("final_abs_contact_force_error", "") != ""],
        "final_abs_contact_force_error",
    )
    lines = [
        "# Source Gear Contact-Status Diagnostics",
        "",
        "This diagnostic compares SFC contact status with Abaqus validation outputs before changing the contact formulation.",
        "",
        f"- SFC history: `{sfc_history}`",
        f"- Abaqus manifest: `{abaqus_manifest}`",
        f"- Abaqus status file: `{abaqus_sta}`" if abaqus_sta is not None else "- Abaqus status file: not provided",
        f"- Abaqus message file: `{msg_path}`" if msg_path is not None else "- Abaqus message file: not provided",
        f"- Abaqus increment sizes parsed from `.sta`: `{increment_sizes}`" if increment_sizes else "- Abaqus increment sizes parsed from `.sta`: not available",
    ]
    if msg_rows:
        lines.extend(
            [
                f"- Abaqus `.msg` increments parsed: `{len(msg_rows)}`",
                f"- increments with severe contact-status iterations: `{len(severe_rows)}`",
                f"- increments with explicit contact compatibility convergence: `{len(converged_contact_rows)}`",
            ]
        )
        if max_final_pen is not None:
            lines.append(
                f"- max final Abaqus contact penetration compatibility error: `{_float(max_final_pen, 'final_abs_penetration_error'):.6e}`"
            )
        if max_final_force is not None:
            lines.append(
                f"- max final Abaqus contact force compatibility error: `{_float(max_final_force, 'final_abs_contact_force_error'):.6e}`"
            )
    if p95_row is not None:
        lines.append(
            f"- max p95 stress/strain relative error: `{100.0 * _float(p95_row, 'p95_von_mises_nodeavg_rel_error'):.3f}%` at `t={_float(p95_row, 'time'):.8g}`"
        )
    if active_row is not None:
        lines.append(
            f"- max active-count relative delta: `{100.0 * _float(active_row, 'active_contact_node_count_rel_delta'):.3f}%` at `t={_float(active_row, 'time'):.8g}`"
        )
    if max_pressure_row is not None:
        lines.append(
            f"- max peak-pressure relative error: `{100.0 * _float(max_pressure_row, 'max_contact_pressure_nodeavg_rel_error'):.3f}%` at `t={_float(max_pressure_row, 'time'):.8g}`"
        )
    if mean_pressure_row is not None:
        lines.append(
            f"- max active-mean-pressure relative error: `{100.0 * _float(mean_pressure_row, 'mean_active_contact_pressure_nodeavg_rel_error'):.3f}%` at `t={_float(mean_pressure_row, 'time'):.8g}`"
        )
    if min_gap_row is not None:
        lines.append(
            f"- max minimum-gap relative error: `{100.0 * _float(min_gap_row, 'min_contact_gap_node_rel_error'):.3f}%` at `t={_float(min_gap_row, 'time'):.8g}`"
        )
    lines.extend(
        [
            f"- release-stage mismatch frames: `{len(release_rows)}`",
            "",
            "## Interpretation",
            "",
            "A release-stage mismatch means Abaqus reports no or near-zero active contact and zero peak CPRESS, while SFC still carries a large active set. This isolates the remaining error to Abaqus-style finite-sliding contact-status and pressure-overclosure enforcement semantics, not to RP kinematics or the SDF distance query alone.",
            "",
            "The next solver change should therefore be validated by reducing the release-stage mismatch count and the p95 stress/strain error together.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfc-history", type=Path, default=DEFAULT_SFC_HISTORY)
    parser.add_argument("--abaqus-manifest", type=Path, default=DEFAULT_ABAQUS_MANIFEST)
    parser.add_argument("--abaqus-sta", type=Path, default=DEFAULT_ABAQUS_STA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    rows = build_contact_status_diagnostics(
        sfc_history=args.sfc_history,
        abaqus_manifest=args.abaqus_manifest,
        abaqus_sta=args.abaqus_sta,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "source_gear_contact_status_diagnostics.csv"
    md_path = args.out_dir / "source_gear_contact_status_diagnostics.md"
    _write_csv(csv_path, rows)
    write_summary(md_path, rows, sfc_history=args.sfc_history, abaqus_manifest=args.abaqus_manifest, abaqus_sta=args.abaqus_sta)
    print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
