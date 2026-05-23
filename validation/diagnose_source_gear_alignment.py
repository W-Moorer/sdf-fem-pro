"""Diagnose source-gear SFC/Abaqus alignment failure modes.

This post-processing utility reads existing SFC/Abaqus VTK manifests and an
existing regional VTK-error CSV. It does not run Abaqus, read ODB files, or
import Abaqus into the SFC core solver.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np


Row = dict[str, Any]


def _read_csv(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _rel_error(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(b)), 1.0e-30)


def equivalent_rotation_angle_near(value: float, reference: float) -> float:
    """Return the ``2*pi``-equivalent angle nearest to ``reference``.

    Abaqus finite-rotation ``UR`` output can change branches for large rotations,
    while the SFC reduced coordinate is an accumulated angular displacement.  A
    direct raw subtraction can therefore report a false multi-turn error.  This
    helper preserves the physical rotation but chooses the branch comparable to
    the reference output.
    """

    period = 2.0 * np.pi
    turns = round((float(value) - float(reference)) / period)
    return float(value) - float(turns) * period


def _first_float(row: Row, *columns: str) -> float:
    for column in columns:
        if column in row and row[column] not in ("", None):
            return _as_float(row[column])
    return 0.0


def _paired_rows(sfc_rows: list[Row], abaqus_rows: list[Row], *, time_tolerance: float) -> list[tuple[Row, Row]]:
    unused = set(range(len(abaqus_rows)))
    pairs: list[tuple[Row, Row]] = []
    for sfc_row in sfc_rows:
        if not unused:
            break
        sfc_time = _as_float(sfc_row.get("time"))
        nearest = min(unused, key=lambda index: abs(_as_float(abaqus_rows[index].get("time")) - sfc_time))
        diff = abs(_as_float(abaqus_rows[nearest].get("time")) - sfc_time)
        if diff <= float(time_tolerance):
            pairs.append((sfc_row, abaqus_rows[nearest]))
            unused.remove(nearest)
    if not pairs:
        raise ValueError("no SFC/Abaqus manifest rows can be paired within the requested time tolerance")
    return pairs


def _regional_lookup(rows: list[Row]) -> dict[tuple[float, str, str], Row]:
    lookup: dict[tuple[float, str, str], Row] = {}
    for row in rows:
        time_value = round(_as_float(row.get("sfc_time")), 14)
        region = str(row.get("region", ""))
        metric = str(row.get("metric", ""))
        lookup[(time_value, region, metric)] = row
    return lookup


def _regional_error(
    lookup: dict[tuple[float, str, str], Row],
    *,
    time_value: float,
    region: str,
    metric: str,
) -> float:
    row = lookup.get((round(float(time_value), 14), region, metric))
    if row is None:
        return 0.0
    return _as_float(row.get("p95_rel_error"))


def build_diagnostic_rows(
    *,
    sfc_manifest: Path,
    abaqus_manifest: Path,
    regional_errors: Path,
    time_tolerance: float = 1.0e-8,
) -> list[Row]:
    """Return paired RP/contact/stress diagnostic rows."""

    sfc_rows = _read_csv(sfc_manifest)
    abaqus_rows = _read_csv(abaqus_manifest)
    regional_lookup = _regional_lookup(_read_csv(regional_errors))
    rows: list[Row] = []
    for pair_index, (sfc, abaqus) in enumerate(_paired_rows(sfc_rows, abaqus_rows, time_tolerance=time_tolerance)):
        sfc_time = _as_float(sfc.get("time"))
        abaqus_time = _as_float(abaqus.get("time"))
        sfc_active = int(round(_first_float(sfc, "active_contact_node_count")))
        abaqus_active = int(round(_first_float(abaqus, "active_contact_node_count")))
        sfc_pressure_max = _first_float(sfc, "max_contact_pressure_nodeavg", "max_contact_pressure")
        abaqus_pressure_max = _first_float(abaqus, "max_contact_pressure_nodeavg", "max_cpress")
        sfc_pressure_p95 = _first_float(sfc, "p95_contact_pressure_nodeavg", "p95_contact_pressure")
        abaqus_pressure_p95 = _first_float(abaqus, "p95_contact_pressure_nodeavg", "p95_cpress")
        rp1_sfc_rotation = _first_float(sfc, "rp1_rotation_z_rad", "rp1_rotation_z")
        rp1_abaqus_rotation = _first_float(abaqus, "rp1_rotation_z_rad", "rp1_rotation_z")
        rp2_sfc_rotation = _first_float(sfc, "rp2_rotation_z_rad", "rp2_rotation_z")
        rp2_abaqus_rotation = _first_float(abaqus, "rp2_rotation_z_rad", "rp2_rotation_z")
        rp1_sfc_rotation_equiv = equivalent_rotation_angle_near(rp1_sfc_rotation, rp1_abaqus_rotation)
        rp2_sfc_rotation_equiv = equivalent_rotation_angle_near(rp2_sfc_rotation, rp2_abaqus_rotation)
        rp1_sfc_velocity = _first_float(sfc, "rp1_angular_velocity_z_rad_per_s", "rp1_angular_velocity_z")
        rp1_abaqus_velocity = _first_float(abaqus, "rp1_angular_velocity_z_rad_per_s", "rp1_angular_velocity_z")
        rp2_sfc_velocity = _first_float(sfc, "rp2_angular_velocity_z_rad_per_s", "rp2_angular_velocity_z")
        rp2_abaqus_velocity = _first_float(abaqus, "rp2_angular_velocity_z_rad_per_s", "rp2_angular_velocity_z")
        full_disp_error = _regional_error(
            regional_lookup,
            time_value=sfc_time,
            region="full",
            metric="displacement_magnitude",
        )
        full_stress_error = _regional_error(
            regional_lookup,
            time_value=sfc_time,
            region="full",
            metric="von_mises_nodeavg",
        )
        full_strain_error = _regional_error(
            regional_lookup,
            time_value=sfc_time,
            region="full",
            metric="strain_norm_nodeavg",
        )
        full_equiv_strain_error = _regional_error(
            regional_lookup,
            time_value=sfc_time,
            region="full",
            metric="equivalent_elastic_strain_nodeavg",
        )
        active_stress_error = _regional_error(
            regional_lookup,
            time_value=sfc_time,
            region="active_union",
            metric="von_mises_nodeavg",
        )
        abaqus_active_stress_error = _regional_error(
            regional_lookup,
            time_value=sfc_time,
            region="abaqus_active",
            metric="von_mises_nodeavg",
        )
        rows.append(
            {
                "pair_index": int(pair_index),
                "time": float(sfc_time),
                "abaqus_time": float(abaqus_time),
                "time_difference": float(sfc_time - abaqus_time),
                "sfc_active_contact_nodes": int(sfc_active),
                "abaqus_active_contact_nodes": int(abaqus_active),
                "active_contact_node_abs_diff": int(abs(sfc_active - abaqus_active)),
                "active_contact_node_rel_diff": abs(sfc_active - abaqus_active) / max(float(abaqus_active), 1.0),
                "sfc_max_contact_pressure": float(sfc_pressure_max),
                "abaqus_max_contact_pressure": float(abaqus_pressure_max),
                "max_contact_pressure_rel_error": _rel_error(sfc_pressure_max, abaqus_pressure_max)
                if abaqus_pressure_max != 0.0
                else float(sfc_pressure_max != 0.0),
                "sfc_p95_contact_pressure": float(sfc_pressure_p95),
                "abaqus_p95_contact_pressure": float(abaqus_pressure_p95),
                "p95_contact_pressure_rel_error": _rel_error(sfc_pressure_p95, abaqus_pressure_p95)
                if abaqus_pressure_p95 != 0.0
                else float(sfc_pressure_p95 != 0.0),
                "sfc_rp1_rotation_z_rad": float(rp1_sfc_rotation),
                "sfc_rp1_rotation_z_rad_equivalent_to_abaqus": float(rp1_sfc_rotation_equiv),
                "abaqus_rp1_rotation_z_rad": float(rp1_abaqus_rotation),
                "rp1_rotation_z_rel_error": _rel_error(rp1_sfc_rotation, rp1_abaqus_rotation),
                "rp1_rotation_z_equivalent_abs_error": abs(rp1_sfc_rotation_equiv - rp1_abaqus_rotation),
                "rp1_rotation_z_equivalent_rel_error": _rel_error(rp1_sfc_rotation_equiv, rp1_abaqus_rotation),
                "sfc_rp2_rotation_z_rad": float(rp2_sfc_rotation),
                "sfc_rp2_rotation_z_rad_equivalent_to_abaqus": float(rp2_sfc_rotation_equiv),
                "abaqus_rp2_rotation_z_rad": float(rp2_abaqus_rotation),
                "rp2_rotation_z_rel_error": _rel_error(rp2_sfc_rotation, rp2_abaqus_rotation),
                "rp2_rotation_z_equivalent_abs_error": abs(rp2_sfc_rotation_equiv - rp2_abaqus_rotation),
                "rp2_rotation_z_equivalent_rel_error": _rel_error(rp2_sfc_rotation_equiv, rp2_abaqus_rotation),
                "sfc_rp1_angular_velocity_z_rad_per_s": float(rp1_sfc_velocity),
                "abaqus_rp1_angular_velocity_z_rad_per_s": float(rp1_abaqus_velocity),
                "rp1_angular_velocity_z_rel_error": _rel_error(rp1_sfc_velocity, rp1_abaqus_velocity),
                "sfc_rp2_angular_velocity_z_rad_per_s": float(rp2_sfc_velocity),
                "abaqus_rp2_angular_velocity_z_rad_per_s": float(rp2_abaqus_velocity),
                "rp2_angular_velocity_z_rel_error": _rel_error(rp2_sfc_velocity, rp2_abaqus_velocity),
                "full_displacement_p95_rel_error": float(full_disp_error),
                "full_von_mises_p95_rel_error": float(full_stress_error),
                "full_strain_norm_p95_rel_error": float(full_strain_error),
                "full_equivalent_elastic_strain_p95_rel_error": float(full_equiv_strain_error),
                "active_union_von_mises_p95_rel_error": float(active_stress_error),
                "abaqus_active_von_mises_p95_rel_error": float(abaqus_active_stress_error),
            }
        )
    return rows


def _max_row(rows: list[Row], column: str) -> Row:
    return max(rows, key=lambda row: _as_float(row.get(column)), default={})


def summarize_diagnostics(rows: list[Row]) -> Row:
    """Summarize dominant source-gear alignment failures."""

    latest = rows[-1] if rows else {}
    worst_stress = _max_row(rows, "full_von_mises_p95_rel_error")
    worst_active = _max_row(rows, "active_union_von_mises_p95_rel_error")
    worst_rp2_rotation = _max_row(rows, "rp2_rotation_z_rel_error")
    worst_rp2_rotation_equiv = _max_row(rows, "rp2_rotation_z_equivalent_abs_error")
    worst_pressure = _max_row(rows, "p95_contact_pressure_rel_error")
    latest_displacement_ok = _as_float(latest.get("full_displacement_p95_rel_error")) <= 0.10
    latest_stress_ok = _as_float(latest.get("full_von_mises_p95_rel_error")) <= 0.10
    latest_strain_ok = _as_float(latest.get("full_strain_norm_p95_rel_error")) <= 0.10
    latest_contact_mismatch = int(latest.get("active_contact_node_abs_diff", 0) or 0)
    return {
        "latest_time": _as_float(latest.get("time")),
        "latest_full_displacement_p95_rel_error": _as_float(latest.get("full_displacement_p95_rel_error")),
        "latest_full_von_mises_p95_rel_error": _as_float(latest.get("full_von_mises_p95_rel_error")),
        "latest_full_strain_norm_p95_rel_error": _as_float(latest.get("full_strain_norm_p95_rel_error")),
        "latest_full_equivalent_elastic_strain_p95_rel_error": _as_float(
            latest.get("full_equivalent_elastic_strain_p95_rel_error")
        ),
        "latest_active_contact_node_abs_diff": int(latest_contact_mismatch),
        "latest_sfc_active_contact_nodes": int(latest.get("sfc_active_contact_nodes", 0) or 0),
        "latest_abaqus_active_contact_nodes": int(latest.get("abaqus_active_contact_nodes", 0) or 0),
        "latest_rp2_rotation_z_rel_error": _as_float(latest.get("rp2_rotation_z_rel_error")),
        "latest_rp2_rotation_z_equivalent_abs_error": _as_float(latest.get("rp2_rotation_z_equivalent_abs_error")),
        "latest_rp2_rotation_z_equivalent_rel_error": _as_float(latest.get("rp2_rotation_z_equivalent_rel_error")),
        "latest_rp2_angular_velocity_z_rel_error": _as_float(latest.get("rp2_angular_velocity_z_rel_error")),
        "worst_full_von_mises_p95_rel_error": _as_float(worst_stress.get("full_von_mises_p95_rel_error")),
        "worst_full_von_mises_time": _as_float(worst_stress.get("time")),
        "worst_active_union_von_mises_p95_rel_error": _as_float(
            worst_active.get("active_union_von_mises_p95_rel_error")
        ),
        "worst_active_union_von_mises_time": _as_float(worst_active.get("time")),
        "worst_rp2_rotation_z_rel_error": _as_float(worst_rp2_rotation.get("rp2_rotation_z_rel_error")),
        "worst_rp2_rotation_z_error_time": _as_float(worst_rp2_rotation.get("time")),
        "worst_rp2_rotation_z_equivalent_abs_error": _as_float(
            worst_rp2_rotation_equiv.get("rp2_rotation_z_equivalent_abs_error")
        ),
        "worst_rp2_rotation_z_equivalent_abs_error_time": _as_float(worst_rp2_rotation_equiv.get("time")),
        "worst_p95_contact_pressure_rel_error": _as_float(worst_pressure.get("p95_contact_pressure_rel_error")),
        "worst_p95_contact_pressure_error_time": _as_float(worst_pressure.get("time")),
        "latest_displacement_pass_10pct": int(latest_displacement_ok),
        "latest_von_mises_pass_10pct": int(latest_stress_ok),
        "latest_strain_norm_pass_10pct": int(latest_strain_ok),
        "overall_latest_gate_pass": int(latest_displacement_ok and latest_stress_ok and latest_strain_ok),
    }


def write_summary(path: Path, summary: Row, rows: list[Row], *, csv_path: Path) -> None:
    latest = rows[-1] if rows else {}
    lines = [
        "# Source Gear Alignment Diagnosis",
        "",
        "该诊断只读取已有 SFC/Abaqus manifest 与 regional error CSV，不运行 Abaqus，不读取 ODB，也不进入核心求解器。",
        "",
        "## Latest Frame",
        "",
        f"- time: `{_as_float(latest.get('time')):.12g} s`",
        f"- full displacement p95 rel. error: `{100.0 * _as_float(latest.get('full_displacement_p95_rel_error')):.3f}%`",
        f"- full von Mises p95 rel. error: `{100.0 * _as_float(latest.get('full_von_mises_p95_rel_error')):.3f}%`",
        f"- full strain-norm p95 rel. error: `{100.0 * _as_float(latest.get('full_strain_norm_p95_rel_error')):.3f}%`",
        f"- active contact nodes: SFC `{int(latest.get('sfc_active_contact_nodes', 0) or 0)}`, Abaqus `{int(latest.get('abaqus_active_contact_nodes', 0) or 0)}`",
        f"- RP2 rotation rel. error: `{100.0 * _as_float(latest.get('rp2_rotation_z_rel_error')):.3f}%`",
        f"- RP2 equivalent-branch rotation abs. error: `{_as_float(latest.get('rp2_rotation_z_equivalent_abs_error')):.6g} rad`",
        f"- RP2 equivalent-branch rotation rel. error: `{100.0 * _as_float(latest.get('rp2_rotation_z_equivalent_rel_error')):.3f}%`",
        f"- RP2 angular velocity rel. error: `{100.0 * _as_float(latest.get('rp2_angular_velocity_z_rel_error')):.3f}%`",
        "",
        "## Worst Errors",
        "",
        f"- worst full von Mises p95 rel. error: `{100.0 * _as_float(summary.get('worst_full_von_mises_p95_rel_error')):.3f}%` at `{_as_float(summary.get('worst_full_von_mises_time')):.12g} s`",
        f"- worst active-union von Mises p95 rel. error: `{100.0 * _as_float(summary.get('worst_active_union_von_mises_p95_rel_error')):.3f}%` at `{_as_float(summary.get('worst_active_union_von_mises_time')):.12g} s`",
        f"- worst RP2 rotation rel. error: `{100.0 * _as_float(summary.get('worst_rp2_rotation_z_rel_error')):.3f}%` at `{_as_float(summary.get('worst_rp2_rotation_z_error_time')):.12g} s`",
        f"- worst RP2 equivalent-branch rotation abs. error: `{_as_float(summary.get('worst_rp2_rotation_z_equivalent_abs_error')):.6g} rad` at `{_as_float(summary.get('worst_rp2_rotation_z_equivalent_abs_error_time')):.12g} s`",
        f"- worst p95 contact-pressure rel. error: `{100.0 * _as_float(summary.get('worst_p95_contact_pressure_rel_error')):.3f}%` at `{_as_float(summary.get('worst_p95_contact_pressure_error_time')):.12g} s`",
        "",
        "## Diagnosis",
        "",
    ]
    if _as_float(latest.get("full_displacement_p95_rel_error")) <= 0.10 and _as_float(
        latest.get("full_von_mises_p95_rel_error")
    ) > 0.10:
        lines.append(
            "- 位移已经被强 RP/MPC 驱动约束到较低误差，但应力仍未过 10% 门槛；位移曲线不能作为应力/应变正确性的充分证据。"
        )
    if int(latest.get("active_contact_node_abs_diff", 0) or 0) > 0:
        lines.append(
            "- 末帧 active contact 节点数不一致，说明 contact status/release 与 pressure history 尚未对齐；这会直接改变后续应力波和齿根应力趋势。"
        )
    if _as_float(latest.get("rp2_rotation_z_equivalent_rel_error")) > 0.10:
        lines.append(
            "- 按有限转动等效分支折返后，受转矩驱动的 RP2 转角误差仍超过 10%；需要优先检查 Abaqus *MPC, BEAM 的有限转动/等效虚功反力口径，而不是调接触刚度。"
        )
    if not lines[-1].startswith("-"):
        lines.append("- 当前未发现单一主导项，建议继续查看逐帧 CSV。")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- latest displacement <= 10%: `{bool(int(summary.get('latest_displacement_pass_10pct', 0) or 0))}`",
            f"- latest von Mises <= 10%: `{bool(int(summary.get('latest_von_mises_pass_10pct', 0) or 0))}`",
            f"- latest strain norm <= 10%: `{bool(int(summary.get('latest_strain_norm_pass_10pct', 0) or 0))}`",
            f"- latest combined gate pass: `{bool(int(summary.get('overall_latest_gate_pass', 0) or 0))}`",
            "",
            f"- diagnostic CSV: `{csv_path.name}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostics(
    *,
    sfc_manifest: Path,
    abaqus_manifest: Path,
    regional_errors: Path,
    out_dir: Path,
    time_tolerance: float = 1.0e-8,
) -> dict[str, Path]:
    rows = build_diagnostic_rows(
        sfc_manifest=sfc_manifest,
        abaqus_manifest=abaqus_manifest,
        regional_errors=regional_errors,
        time_tolerance=float(time_tolerance),
    )
    summary = summarize_diagnostics(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_csv = out_dir / "source_gear_alignment_diagnostics.csv"
    summary_csv = out_dir / "source_gear_alignment_diagnostic_summary.csv"
    summary_md = out_dir / "source_gear_alignment_diagnostic_summary.md"
    _write_csv(diagnostics_csv, rows)
    _write_csv(summary_csv, [summary])
    write_summary(summary_md, summary, rows, csv_path=diagnostics_csv)
    return {"diagnostics_csv": diagnostics_csv, "summary_csv": summary_csv, "summary_md": summary_md}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sfc-manifest",
        type=Path,
        default=Path("results/source_gear_penalty_contact_0030_sfc_alpha0/sfc_vtk/sfc_manifest.csv"),
    )
    parser.add_argument(
        "--abaqus-manifest",
        type=Path,
        default=Path("results/source_gear_abaqus_penalty_full_stride2_match_step_0030/abaqus_vtk/abaqus_manifest.csv"),
    )
    parser.add_argument(
        "--regional-errors",
        type=Path,
        default=Path(
            "results/source_gear_penalty_contact_0030_alpha0_sequence_alignment/"
            "source_gear_regional_vtk_errors.csv"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/source_gear_alignment_diagnostics"))
    parser.add_argument("--time-tolerance", type=float, default=1.0e-8)
    args = parser.parse_args(argv)
    outputs = run_diagnostics(
        sfc_manifest=args.sfc_manifest,
        abaqus_manifest=args.abaqus_manifest,
        regional_errors=args.regional_errors,
        out_dir=args.out_dir,
        time_tolerance=float(args.time_tolerance),
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
