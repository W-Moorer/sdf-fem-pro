"""CalculiX comparison for the same implicit sphere-drop condition.

This validation-only runner starts from the Abaqus/Standard sphere-drop input
deck used by the commercial comparison, writes an equivalent CalculiX C3D4/S4
frictionless linear-penalty contact deck, runs ``ccx`` through WSL, and plots
the resulting open-source native-contact trajectory against the existing SFC
Lagrangian-SDF and Abaqus reference curves.

The core SFC package is not coupled to CalculiX or Abaqus.  Both solvers are
used here only as external validation references.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter
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

from validation.run_abaqus_sphere_drop_full_validation import (  # noqa: E402
    _abaqus_history_from_vtk,
)
from validation.run_abaqus_sphere_drop_short_validation import (  # noqa: E402
    DEFAULT_VTK_DIR,
    AbaqusSphereDropModel,
    _configure_plot_style,
    parse_sphere_drop_inp,
)
from validation.run_calculix_drop_impact_comparison import (  # noqa: E402
    CALCULIX_MASTER_SURFACE_OFFSET,
    DropModel,
    _boundary_face_data,
    _fieldnames,
    _floor,
    _parse_calculix_dat_contact_totals,
    _parse_calculix_dat_displacements,
    _write_csv,
    run_calculix,
)

DEFAULT_IMPLICIT_CASE_DIR = ROOT / "commercial_software_comparison" / "abaqus_flexible_body_rigid_plane_implicit" / "sphere_drop"
DEFAULT_IMPLICIT_INP = DEFAULT_IMPLICIT_CASE_DIR / "abaqus_run" / "sphere_drop_implicit.inp"
DEFAULT_IMPLICIT_VTK_DIR = DEFAULT_IMPLICIT_CASE_DIR / "vtk"
DEFAULT_SFC_HISTORY = ROOT / "results" / "abaqus_sphere_drop_implicit_hht_surface_aligned" / "abaqus_sphere_drop_full_sfc_history.csv"

Row = dict[str, Any]


def build_calculix_model_from_abaqus(
    source: AbaqusSphereDropModel,
    *,
    duration: float,
    dt: float,
    hht_alpha: float,
    output_frequency: int = 1,
    direct_dynamic: bool = False,
    explicit_dynamic: bool = False,
    friction_coefficient: float | None = None,
) -> DropModel:
    """Return a CalculiX model using the same C3D4 body and physical settings."""

    node_ids = np.arange(1, source.nodes.shape[0] + 1, dtype=np.int64)
    surface_faces, slave_face_refs, surface_indices, surface_areas = _boundary_face_data(source.nodes, source.elements)
    plane_xy = np.asarray(source.plane_nodes[:4, :2], dtype=float)
    floor_half_width = float(np.max(np.abs(plane_xy))) if plane_xy.size else 0.3
    # CalculiX S4 SPOS is offset by half the shell thickness used by the shared
    # writer.  Shifting the shell mid-surface down keeps the master contact
    # surface at the Abaqus rigid-plane z coordinate.
    floor_mid_z = float(source.plane_z - CALCULIX_MASTER_SURFACE_OFFSET)
    floor_nodes, floor_elements = _floor(floor_half_width, floor_mid_z, start_id=int(node_ids[-1]) + 1)
    if source.contact_penalty_normal_stiffness is None:
        raise ValueError("Abaqus input deck does not provide a linear penalty contact stiffness")
    return DropModel(
        case="abaqus_implicit_sphere_drop",
        resolution=0,
        node_ids=node_ids,
        nodes=np.asarray(source.nodes, dtype=float),
        tet_elements=np.asarray(source.elements, dtype=np.int64),
        surface_faces=surface_faces,
        slave_face_refs=slave_face_refs,
        surface_node_ids=node_ids[surface_indices],
        surface_node_areas=surface_areas,
        floor_nodes=floor_nodes,
        floor_element_ids=floor_elements,
        floor_z=floor_mid_z,
        E=float(source.young),
        nu=float(source.poisson),
        density=float(source.density),
        gravity=float(source.gravity),
        initial_velocity_z=0.0,
        contact_stiffness=float(source.contact_penalty_normal_stiffness),
        total_time=float(duration),
        dt=float(dt),
        output_frequency=max(1, int(output_frequency)),
        direct_dynamic=bool(direct_dynamic),
        sfc_substeps=1,
        contact_smoothing_epsilon=0.0,
        hht_alpha=float(hht_alpha),
        model_source="abaqus_implicit_sphere_drop_inp",
        explicit_dynamic=bool(explicit_dynamic),
        friction_coefficient=friction_coefficient,
    )


def _read_rows(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_rows(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    _write_csv(path, _fieldnames(rows), rows)


def _filter_duration(rows: list[Row], duration: float) -> list[Row]:
    return [row for row in rows if float(row["time"]) <= float(duration) + 1.0e-12]


def _calculix_history_from_displacements(
    model: DropModel,
    blocks: dict[float, np.ndarray],
    totals: dict[float, dict[str, float | int]],
) -> list[Row]:
    rows: list[Row] = []
    times = [0.0] + [float(t) for t in blocks]
    displacements = [np.zeros_like(model.nodes)] + [blocks[t] for t in blocks]
    for time_value, u_nodes in zip(times, displacements, strict=True):
        current = model.nodes + u_nodes
        centers = np.mean(current[model.tet_elements, 2], axis=1)
        # The Abaqus/CalculiX input uses uniform density, so volume weighting is
        # the physically consistent center-of-mass reduction.
        volumes = []
        for tet in model.tet_elements:
            X = model.nodes[tet]
            volumes.append(abs(float(np.linalg.det(np.stack((X[1] - X[0], X[2] - X[0], X[3] - X[0]), axis=1)))) / 6.0)
        weights = np.asarray(volumes, dtype=float)
        z_cm = float(np.dot(weights, centers) / np.sum(weights))
        face_points = np.mean(current[model.surface_faces], axis=1)
        gaps = face_points[:, 2] - float(model.floor_z + CALCULIX_MASTER_SURFACE_OFFSET)
        penetration = np.maximum(-gaps, 0.0)
        total = totals.get(float(time_value), {})
        force = total.get("normal_force_from_rf", "")
        contact_energy = total.get("calculix_contact_energy", "")
        rows.append(
            {
                "source": "calculix_native_ccx",
                "step": int(round(float(time_value) / model.dt)),
                "time": float(time_value),
                "z_cm": z_cm,
                "min_gap": float(np.min(gaps)),
                "active_contact_count": int(total.get("calculix_contact_count", int(np.count_nonzero(penetration > 0.0)))),
                "max_penetration": float(np.max(penetration)) if penetration.size else 0.0,
                "normal_force_z": force,
                "contact_energy": contact_energy,
                "contact_path": "CalculiX native surface-to-surface linear penalty contact",
            }
        )
    return rows


def _series(rows: list[Row], key: str) -> tuple[np.ndarray, np.ndarray]:
    valid = [row for row in rows if key in row and row[key] != ""]
    return (
        np.asarray([float(row["time"]) for row in valid], dtype=float),
        np.asarray([float(row[key]) for row in valid], dtype=float),
    )


def _first_contact(rows: list[Row]) -> float | str:
    active = [
        float(row["time"])
        for row in rows
        if int(float(row.get("active_contact_count", 0) or 0)) > 0 or float(row.get("max_penetration", 0.0) or 0.0) > 0.0
    ]
    return "" if not active else min(active)


def _comparison_metrics(source_rows: dict[str, list[Row]], reference_name: str = "abaqus_native_vtk") -> list[Row]:
    ref = source_rows[reference_name]
    ref_t, ref_z = _series(ref, "z_cm")
    ref_gap_t, ref_gap = _series(ref, "min_gap")
    rows: list[Row] = []
    for source, rows_for_source in source_rows.items():
        t, z = _series(rows_for_source, "z_cm")
        if source == reference_name or t.size == 0:
            continue
        common = t[(t >= ref_t[0]) & (t <= ref_t[-1])]
        z_source = np.interp(common, t, z)
        z_ref = np.interp(common, ref_t, ref_z)
        z_err = z_source - z_ref
        gt, gap = _series(rows_for_source, "min_gap")
        if gt.size and ref_gap_t.size:
            gap_source = np.interp(common, gt, gap)
            gap_ref = np.interp(common, ref_gap_t, ref_gap)
            gap_err = gap_source - gap_ref
            max_gap_error: float | str = float(np.max(np.abs(gap_err)))
            rms_gap_error: float | str = float(np.sqrt(np.mean(gap_err**2)))
        else:
            max_gap_error = ""
            rms_gap_error = ""
        rows.extend(
            [
                {
                    "source": source,
                    "metric": "max_z_cm_abs_error_vs_abaqus",
                    "value": float(np.max(np.abs(z_err))),
                    "status": "reported",
                },
                {
                    "source": source,
                    "metric": "rms_z_cm_abs_error_vs_abaqus",
                    "value": float(np.sqrt(np.mean(z_err**2))),
                    "status": "reported",
                },
                {
                    "source": source,
                    "metric": "max_min_gap_abs_error_vs_abaqus",
                    "value": max_gap_error,
                    "status": "reported" if max_gap_error != "" else "not_available",
                },
                {
                    "source": source,
                    "metric": "rms_min_gap_abs_error_vs_abaqus",
                    "value": rms_gap_error,
                    "status": "reported" if rms_gap_error != "" else "not_available",
                },
                {
                    "source": source,
                    "metric": "first_contact_time",
                    "value": _first_contact(rows_for_source),
                    "status": "reported",
                },
            ]
        )
    rows.append({"source": reference_name, "metric": "first_contact_time", "value": _first_contact(ref), "status": "reported"})
    return rows


def _plot_overlay(path: Path, source_rows: dict[str, list[Row]], *, key: str, ylabel: str, title: str) -> None:
    _configure_plot_style()
    styles = {
        "abaqus_native_vtk": ("#1f4e79", "-", "Abaqus/Standard"),
        "sfc_lagrangian_sdf_full": ("#c0504d", "--", "SFC Lagrangian SDF"),
        "calculix_native_ccx": ("#548235", "-.", "CalculiX/ccx"),
    }
    fig, ax = plt.subplots(figsize=(3.35, 2.2), constrained_layout=True)
    for source, rows in source_rows.items():
        t, y = _series(rows, key)
        if t.size == 0:
            continue
        color, linestyle, label = styles.get(source, ("#666666", "-", source))
        ax.plot(t, y, color=color, linestyle=linestyle, linewidth=1.45, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#d9d9d9", linewidth=0.5)
    ax.legend(loc="best", frameon=True)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _write_summary(path: Path, *, metrics: list[Row], outputs: dict[str, Path], calculix_wall: float, duration: float) -> None:
    lines = [
        "# CalculiX Same-Condition Sphere Drop Comparison",
        "",
        "本验证从同一个 Abaqus/Standard implicit sphere-drop `.inp` 出发，生成并运行 CalculiX 原生接触模型，",
        "然后把 CalculiX 曲线与现有 SFC Lagrangian SDF 和 Abaqus/Standard 参考曲线叠加。",
        "",
        "## Scope",
        "",
        "- CalculiX 是开源外部参考求解器，不进入 SFC core。",
        "- 几何、C3D4 网格、材料、重力、线性 penalty 接触刚度、输出时间步和刚性平面位置来自同一个 `.inp`。",
        "- 默认使用 CalculiX 自动增量，因为该 Abaqus/Standard 工况也允许接触收敛切步；`--direct-dynamic` 保留为固定步长诊断选项。",
        "- 本次输出用于观察同工况下开源原生接触轨迹；不是证明 SFC 与 CalculiX 源码级等价。",
        "",
        "## Run",
        "",
        f"- Compared duration: `{duration:.6g} s`.",
        f"- CalculiX native solve wall time: `{calculix_wall:.6f} s`.",
        "",
        "## Metrics",
        "",
        "| Source | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    for row in metrics:
        lines.append(f"| `{row['source']}` | `{row['metric']}` | {row['value']} |")
    lines.extend(["", "## Outputs", ""])
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value.name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    out_dir: Path,
    *,
    inp: Path = DEFAULT_IMPLICIT_INP,
    vtk_dir: Path = DEFAULT_IMPLICIT_VTK_DIR,
    sfc_history: Path = DEFAULT_SFC_HISTORY,
    duration: float = 0.12,
    dt: float = 0.001,
    hht_alpha: float = -0.3,
    output_frequency: int = 1,
    calculix_timeout_seconds: int = 900,
    direct_dynamic: bool = False,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = parse_sphere_drop_inp(inp)
    duration_value = min(float(duration), float(source.abaqus_duration))
    model = build_calculix_model_from_abaqus(
        source,
        duration=duration_value,
        dt=float(dt),
        hht_alpha=float(hht_alpha),
        output_frequency=int(output_frequency),
        direct_dynamic=bool(direct_dynamic),
    )

    start = perf_counter()
    dat_path, command_row = run_calculix(
        model,
        out_dir,
        case_name="same_condition_sphere_drop_calculix",
        timeout_seconds=int(calculix_timeout_seconds),
    )
    calculix_wall = perf_counter() - start
    blocks = _parse_calculix_dat_displacements(dat_path, model.node_ids)
    totals = _parse_calculix_dat_contact_totals(dat_path)
    calculix_rows = _filter_duration(_calculix_history_from_displacements(model, blocks, totals), duration_value)
    abaqus_rows = _filter_duration(_abaqus_history_from_vtk(source, vtk_dir, duration=duration_value), duration_value)
    sfc_rows = _filter_duration(_read_rows(sfc_history), duration_value)
    source_rows = {
        "abaqus_native_vtk": abaqus_rows,
        "sfc_lagrangian_sdf_full": sfc_rows,
        "calculix_native_ccx": calculix_rows,
    }
    metrics = _comparison_metrics(source_rows)
    command_row["calculix_wall_seconds_measured_by_runner"] = float(calculix_wall)

    outputs = {
        "calculix_history": out_dir / "same_condition_calculix_history.csv",
        "abaqus_history": out_dir / "same_condition_abaqus_history.csv",
        "sfc_history": out_dir / "same_condition_sfc_history.csv",
        "metrics": out_dir / "same_condition_open_source_metrics.csv",
        "command": out_dir / "same_condition_calculix_command.csv",
        "z_cm_curve": out_dir / "same_condition_z_cm.png",
        "gap_curve": out_dir / "same_condition_min_gap.png",
        "force_curve": out_dir / "same_condition_normal_force.png",
        "summary": out_dir / "same_condition_calculix_summary_cn.md",
    }
    _write_rows(outputs["calculix_history"], calculix_rows)
    _write_rows(outputs["abaqus_history"], abaqus_rows)
    _write_rows(outputs["sfc_history"], sfc_rows)
    _write_rows(outputs["metrics"], metrics)
    _write_rows(outputs["command"], [command_row])
    _plot_overlay(outputs["z_cm_curve"], source_rows, key="z_cm", ylabel="Mass-center z (m)", title="Same-condition displacement trajectory")
    _plot_overlay(outputs["gap_curve"], source_rows, key="min_gap", ylabel="Minimum gap (m)", title="Same-condition minimum gap")
    _plot_overlay(outputs["force_curve"], source_rows, key="normal_force_z", ylabel="Normal contact force (N)", title="Same-condition normal force")
    _write_summary(outputs["summary"], metrics=metrics, outputs=outputs, calculix_wall=calculix_wall, duration=duration_value)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inp", type=Path, default=DEFAULT_IMPLICIT_INP)
    parser.add_argument("--vtk-dir", type=Path, default=DEFAULT_IMPLICIT_VTK_DIR)
    parser.add_argument("--sfc-history", type=Path, default=DEFAULT_SFC_HISTORY)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "calculix_same_condition_sphere_drop")
    parser.add_argument("--duration", type=float, default=0.12)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--hht-alpha", type=float, default=-0.3)
    parser.add_argument("--output-frequency", type=int, default=1)
    parser.add_argument("--calculix-timeout-seconds", type=int, default=900)
    parser.add_argument("--direct-dynamic", action="store_true", help="Use fixed direct CalculiX increments; default allows automatic cutbacks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(
        args.out_dir,
        inp=args.inp,
        vtk_dir=args.vtk_dir,
        sfc_history=args.sfc_history,
        duration=float(args.duration),
        dt=float(args.dt),
        hht_alpha=float(args.hht_alpha),
        output_frequency=int(args.output_frequency),
        calculix_timeout_seconds=int(args.calculix_timeout_seconds),
        direct_dynamic=bool(args.direct_dynamic),
    )
    print("CalculiX same-condition sphere-drop comparison complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
