"""Audit current source-gear SFC/Abaqus alignment evidence.

This script is intentionally post-processing only. It reads the source Abaqus
input deck, existing VTK manifests, and existing alignment CSVs; it does not run
SFC, Abaqus, or any ODB reader.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Row = dict[str, Any]


@dataclass(frozen=True)
class SourceDeckSettings:
    nlgeom: str
    dynamic_initial_dt: float
    dynamic_total_time: float
    dynamic_min_dt: float
    dynamic_max_dt: float
    gear1_angular_velocity_z: float
    gear2_torque_z: float
    contact_behavior: str
    contact_pair: str
    mpc_beam_count: int


def _read_csv(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _numeric_csv_line(line: str) -> list[float]:
    values: list[float] = []
    for token in line.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def parse_source_deck(path: Path) -> SourceDeckSettings:
    """Extract the source gear settings required by the active validation goal."""

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    nlgeom = ""
    dynamic = [0.0, 0.0, 0.0, 0.0]
    gear1_omega = 0.0
    gear2_torque = 0.0
    contact_behavior = ""
    contact_pair = ""
    mpc_beam_count = 0

    for index, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("*step"):
            match = re.search(r"nlgeom\s*=\s*([^,\s]+)", stripped, flags=re.IGNORECASE)
            nlgeom = match.group(1) if match else ""
        elif lower.startswith("*dynamic") and index + 1 < len(lines):
            values = _numeric_csv_line(lines[index + 1])
            if len(values) >= 4:
                dynamic = values[:4]
        elif lower.startswith("*surface behavior"):
            contact_behavior = stripped
        elif lower.startswith("*contact pair"):
            contact_pair = stripped
        elif lower.startswith("*mpc"):
            if index + 1 < len(lines) and lines[index + 1].strip().upper().startswith("BEAM"):
                mpc_beam_count += 1
        elif lower.startswith("*boundary") and "velocity" in lower:
            probe = index + 1
            while probe < len(lines) and not lines[probe].lstrip().startswith("*"):
                parts = [part.strip() for part in lines[probe].split(",")]
                if len(parts) >= 4 and parts[1] == "6" and parts[2] == "6":
                    gear1_omega = float(parts[3])
                probe += 1
        elif lower.startswith("*cload"):
            probe = index + 1
            while probe < len(lines) and not lines[probe].lstrip().startswith("*"):
                parts = [part.strip() for part in lines[probe].split(",")]
                if len(parts) >= 3 and parts[1] == "6":
                    gear2_torque = float(parts[2])
                probe += 1

    return SourceDeckSettings(
        nlgeom=nlgeom,
        dynamic_initial_dt=float(dynamic[0]),
        dynamic_total_time=float(dynamic[1]),
        dynamic_min_dt=float(dynamic[2]),
        dynamic_max_dt=float(dynamic[3]),
        gear1_angular_velocity_z=float(gear1_omega),
        gear2_torque_z=float(gear2_torque),
        contact_behavior=contact_behavior,
        contact_pair=contact_pair,
        mpc_beam_count=int(mpc_beam_count),
    )


def summarize_manifest(path: Path) -> Row:
    rows = _read_csv(path) if path.exists() else []
    times = [_as_float(row.get("time")) for row in rows]
    diffs = [b - a for a, b in zip(times, times[1:]) if b >= a]
    return {
        "path": str(path),
        "exists": int(path.exists()),
        "frame_count": len(rows),
        "start_time": min(times) if times else 0.0,
        "end_time": max(times) if times else 0.0,
        "median_frame_dt": sorted(diffs)[len(diffs) // 2] if diffs else 0.0,
        "has_contact_pressure": int(any("contact_pressure" in key for row in rows[:1] for key in row)),
        "rotation_unit": rows[-1].get("rotation_unit", "") if rows else "",
        "stress_strain_postprocess": rows[-1].get("stress_strain_postprocess", "") if rows else "",
    }


def summarize_alignment(path: Path) -> Row:
    rows = _read_csv(path) if path.exists() else []
    full = [row for row in rows if row.get("region") == "full"]
    summary: Row = {
        "path": str(path),
        "exists": int(path.exists()),
        "paired_frame_count": len({round(_as_float(row.get("sfc_time")), 15) for row in rows}),
        "end_time": max((_as_float(row.get("sfc_time")) for row in rows), default=0.0),
    }
    for metric in ("displacement_magnitude", "von_mises_nodeavg", "equivalent_elastic_strain_nodeavg", "strain_norm_nodeavg"):
        metric_rows = [row for row in full if row.get("metric") == metric]
        worst = max(metric_rows, key=lambda row: _as_float(row.get("p95_rel_error")), default=None)
        latest = max(metric_rows, key=lambda row: _as_float(row.get("sfc_time")), default=None)
        summary[f"{metric}_max_p95_rel_error"] = _as_float(worst.get("p95_rel_error")) if worst else 0.0
        summary[f"{metric}_max_p95_error_time"] = _as_float(worst.get("sfc_time")) if worst else 0.0
        summary[f"{metric}_latest_p95_rel_error"] = _as_float(latest.get("p95_rel_error")) if latest else 0.0
    return summary


def parse_abaqus_sta_completed_time(path: Path) -> Row:
    """Return the last completed Abaqus increment time from a ``.sta`` file."""

    if not path.exists():
        return {"path": str(path), "exists": 0, "completed_time": 0.0, "completed_increment": 0, "completed": 0}
    completed_time = 0.0
    completed_increment = 0
    completed = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in line.upper():
                completed = 1
            parts = line.split()
            if len(parts) >= 8 and parts[0].isdigit() and parts[1].isdigit():
                try:
                    completed_increment = int(parts[1])
                    completed_time = float(parts[6])
                except ValueError:
                    continue
    return {
        "path": str(path),
        "exists": 1,
        "completed_time": float(completed_time),
        "completed_increment": int(completed_increment),
        "completed": int(completed),
    }


def _status(ok: bool) -> str:
    return "PASS" if ok else "MISSING"


def build_audit_rows(deck: SourceDeckSettings, sfc: Row, abaqus: Row, alignment: Row, sta: Row | None = None) -> list[Row]:
    target_duration = deck.dynamic_total_time
    target_dt = deck.dynamic_initial_dt
    sta = sta or {}
    rows: list[Row] = []
    rows.append(
        {
            "requirement": "source_deck_settings_parsed",
            "status": _status(deck.dynamic_total_time > 0.0 and deck.dynamic_initial_dt > 0.0),
            "evidence": f"dt={deck.dynamic_initial_dt:g}, total={deck.dynamic_total_time:g}, nlgeom={deck.nlgeom}",
        }
    )
    rows.append(
        {
            "requirement": "source_drive_units_radian",
            "status": _status(abs(deck.gear1_angular_velocity_z - 52.36) < 1.0e-12 and sfc.get("rotation_unit") == "radian"),
            "evidence": f"deck omega_z={deck.gear1_angular_velocity_z:g}, sfc rotation_unit={sfc.get('rotation_unit', '')}",
        }
    )
    rows.append(
        {
            "requirement": "sfc_vtk_stress_strain_frames_exist",
            "status": _status(int(sfc.get("frame_count", 0)) > 0 and sfc.get("stress_strain_postprocess") != ""),
            "evidence": f"frames={sfc.get('frame_count')}, postprocess={sfc.get('stress_strain_postprocess')}",
        }
    )
    rows.append(
        {
            "requirement": "abaqus_vtk_contact_frames_exist",
            "status": _status(int(abaqus.get("frame_count", 0)) > 0 and int(abaqus.get("has_contact_pressure", 0)) == 1),
            "evidence": f"frames={abaqus.get('frame_count')}, contact_pressure_field={abaqus.get('has_contact_pressure')}",
        }
    )
    rows.append(
        {
            "requirement": "abaqus_reference_duration_matches_source",
            "status": _status(_as_float(sta.get("completed_time")) >= target_duration - 1.0e-14),
            "evidence": (
                f"sta_completed_time={_as_float(sta.get('completed_time')):g}, "
                f"target={target_duration:g}, increment={int(sta.get('completed_increment', 0) or 0)}"
            ),
        }
    )
    rows.append(
        {
            "requirement": "paired_displacement_stress_strain_curves_exist",
            "status": _status(int(alignment.get("paired_frame_count", 0)) > 1),
            "evidence": f"paired_frames={alignment.get('paired_frame_count')}, end_time={alignment.get('end_time'):g}",
        }
    )
    rows.append(
        {
            "requirement": "full_source_duration_covered",
            "status": _status(_as_float(alignment.get("end_time")) >= target_duration - 1.0e-14),
            "evidence": f"covered={_as_float(alignment.get('end_time')):g}, target={target_duration:g}",
        }
    )
    rows.append(
        {
            "requirement": "output_frame_stride_documented",
            "status": _status(_as_float(sfc.get("median_frame_dt")) >= target_dt),
            "evidence": f"sfc_frame_dt={_as_float(sfc.get('median_frame_dt')):g}, solver_dt_target={target_dt:g}",
        }
    )
    rows.append(
        {
            "requirement": "stress_strain_alignment_currently_acceptable",
            "status": _status(_as_float(alignment.get("von_mises_nodeavg_max_p95_rel_error")) < 0.10),
            "evidence": (
                "max_p95_von_mises_error="
                f"{100.0 * _as_float(alignment.get('von_mises_nodeavg_max_p95_rel_error')):.2f}%"
            ),
        }
    )
    return rows


def write_audit_report(
    *,
    deck: SourceDeckSettings,
    sfc: Row,
    abaqus: Row,
    alignment: Row,
    sta: Row,
    audit_rows: list[Row],
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "source_gear_goal_status.csv"
    md_path = out_dir / "source_gear_goal_status.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["requirement", "status", "evidence"])
        writer.writeheader()
        writer.writerows(audit_rows)
    lines = [
        "# 源齿轮目标状态核验",
        "",
        "该报告只读取已有 deck、manifest 和误差 CSV，不运行 Abaqus、SFC 或 ODB 解析。",
        "",
        "## 源 deck",
        "",
        f"- `nlgeom`: `{deck.nlgeom}`",
        f"- `*Dynamic`: initial dt `{deck.dynamic_initial_dt:g}`, total `{deck.dynamic_total_time:g}`, min `{deck.dynamic_min_dt:g}`, max `{deck.dynamic_max_dt:g}`",
        f"- Gear 1 RP dof 6 angular velocity: `{deck.gear1_angular_velocity_z:g}` rad/s",
        f"- Gear 2 RP dof 6 torque: `{deck.gear2_torque_z:g}`",
        f"- Contact behavior: `{deck.contact_behavior}`",
        f"- Contact pair: `{deck.contact_pair}`",
        f"- BEAM MPC count: `{deck.mpc_beam_count}`",
        "",
        "## 当前输出覆盖",
        "",
        f"- SFC frames: `{sfc.get('frame_count')}`, end time `{_as_float(sfc.get('end_time')):g}`, frame dt `{_as_float(sfc.get('median_frame_dt')):g}`",
        f"- Abaqus frames: `{abaqus.get('frame_count')}`, end time `{_as_float(abaqus.get('end_time')):g}`, frame dt `{_as_float(abaqus.get('median_frame_dt')):g}`",
        f"- Abaqus STA completed time: `{_as_float(sta.get('completed_time')):g}`, completed increment `{int(sta.get('completed_increment', 0) or 0)}`",
        f"- Paired curve frames: `{alignment.get('paired_frame_count')}`, end time `{_as_float(alignment.get('end_time')):g}`",
        "",
        "## 当前最大 p95 误差",
        "",
        f"- Displacement: `{100.0 * _as_float(alignment.get('displacement_magnitude_max_p95_rel_error')):.2f}%`",
        f"- von Mises: `{100.0 * _as_float(alignment.get('von_mises_nodeavg_max_p95_rel_error')):.2f}%`",
        f"- Equivalent elastic strain: `{100.0 * _as_float(alignment.get('equivalent_elastic_strain_nodeavg_max_p95_rel_error')):.2f}%`",
        f"- Strain norm: `{100.0 * _as_float(alignment.get('strain_norm_nodeavg_max_p95_rel_error')):.2f}%`",
        "",
        "## 要求核验",
        "",
        "| Requirement | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in audit_rows:
        lines.append(f"| `{row['requirement']}` | `{row['status']}` | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "当前证据证明：源 deck 设置已被读取，SFC/Abaqus 均已有 VTK 场输出，且已有位移/应力/应变曲线对齐结果。",
            "",
            "当前证据尚不能证明：完整 `0.05 s`、`dt=1e-5` 目标时长已经完成，也不能证明应力/应变趋势已与 Abaqus 对齐。下一步应继续延长同一工况窗口，并优先对齐接触压力历史、active set、surface-to-surface 约束平均和应力/应变输出口径。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "summary": md_path}


def run_audit(
    *,
    inp: Path,
    sfc_manifest: Path,
    abaqus_manifest: Path,
    abaqus_sta: Path,
    alignment_csv: Path,
    out_dir: Path,
) -> dict[str, Path]:
    deck = parse_source_deck(inp)
    sfc = summarize_manifest(sfc_manifest)
    abaqus = summarize_manifest(abaqus_manifest)
    sta = parse_abaqus_sta_completed_time(abaqus_sta)
    alignment = summarize_alignment(alignment_csv)
    audit_rows = build_audit_rows(deck, sfc, abaqus, alignment, sta)
    return write_audit_report(
        deck=deck,
        sfc=sfc,
        abaqus=abaqus,
        alignment=alignment,
        sta=sta,
        audit_rows=audit_rows,
        out_dir=out_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inp",
        type=Path,
        default=Path("commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp"),
    )
    parser.add_argument(
        "--sfc-manifest",
        type=Path,
        default=Path("results/source_gear_penalty_contact_incremental_check_sfc/sfc_vtk/sfc_manifest.csv"),
    )
    parser.add_argument(
        "--abaqus-manifest",
        type=Path,
        default=Path(
            "results/source_gear_abaqus_penalty_full_stride2_match_step_0020/"
            "abaqus_vtk_contact_0010_0020/abaqus_manifest.csv"
        ),
    )
    parser.add_argument(
        "--abaqus-sta",
        type=Path,
        default=Path(
            "results/source_gear_abaqus_penalty_full_stride2_match_step_0020/"
            "abaqus_run/gear_contact_source_penalty.sta"
        ),
    )
    parser.add_argument(
        "--alignment-csv",
        type=Path,
        default=Path(
            "results/source_gear_penalty_contact_0000_0020_merged_alignment/"
            "source_gear_regional_merged_errors.csv"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/source_gear_goal_status"))
    args = parser.parse_args(argv)
    outputs = run_audit(
        inp=args.inp,
        sfc_manifest=args.sfc_manifest,
        abaqus_manifest=args.abaqus_manifest,
        abaqus_sta=args.abaqus_sta,
        alignment_csv=args.alignment_csv,
        out_dir=args.out_dir,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
