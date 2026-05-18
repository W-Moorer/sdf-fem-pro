"""Export dynamic SFC/CalculiX C3D8 trajectories as numbered VTK frame series.

The script is visualization-only. It does not change the contact method or any
solver state. Existing CalculiX ``.dat`` files are parsed when available; SFC
native frames are regenerated from the same deterministic C3D8 models.
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

from validation.run_c3d8_contact_trajectory_validation import (  # noqa: E402
    _parse_element_stress,
    _parse_nodal_vectors,
    build_c3d8_model,
)
from validation.run_phase9_full_contact_validation import (  # noqa: E402
    _native_c3d8_linear_dynamic_history,
    _phase9_hex_cell_fields,
)
from validation.vtk_frame_series import write_hex_frame_series, write_legacy_hex_vtk_frame, write_pvd  # noqa: E402

Row = dict[str, Any]


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _calculix_dat_path(source_dir: Path, case: str, resolution: int) -> Path:
    return source_dir / "c3d8_dynamic_trajectory" / "calculix_runs" / f"{case}_r{resolution}" / f"{case}_r{resolution}.dat"


def _select_by_stride(frames: list[tuple[float, np.ndarray]], frame_stride: int) -> list[tuple[float, np.ndarray]]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    selected = [(time_value, values) for index, (time_value, values) in enumerate(frames) if index % frame_stride == 0]
    if frames and selected[-1][0] != frames[-1][0]:
        selected.append(frames[-1])
    return selected


def _with_initial_zero_frame(frames: list[tuple[float, np.ndarray]], *, node_count: int) -> list[tuple[float, np.ndarray]]:
    if not frames:
        return [(0.0, np.zeros((node_count, 3), dtype=float))]
    first_time, first_u = frames[0]
    if abs(float(first_time)) <= 1.0e-14:
        return frames
    return [(0.0, np.zeros_like(first_u, dtype=float)), *frames]


def _interpolate_displacement_frames(
    frames: list[tuple[float, np.ndarray]],
    target_times: list[float],
) -> list[tuple[float, np.ndarray]]:
    if not frames:
        raise ValueError("cannot interpolate an empty displacement history")
    source_times = np.asarray([float(time_value) for time_value, _U in frames], dtype=float)
    source_values = [np.asarray(U, dtype=float) for _time, U in frames]
    interpolated: list[tuple[float, np.ndarray]] = []
    for target_time in target_times:
        t = float(target_time)
        if t <= source_times[0] + 1.0e-13:
            interpolated.append((t, source_values[0].copy()))
            continue
        if t >= source_times[-1] - 1.0e-13:
            interpolated.append((t, source_values[-1].copy()))
            continue
        right = int(np.searchsorted(source_times, t, side="left"))
        left = right - 1
        t0 = float(source_times[left])
        t1 = float(source_times[right])
        alpha = 0.0 if abs(t1 - t0) <= 1.0e-15 else (t - t0) / (t1 - t0)
        U = (1.0 - alpha) * source_values[left] + alpha * source_values[right]
        interpolated.append((t, np.asarray(U, dtype=float)))
    return interpolated


def _interpolate_stress_maps(
    stress_maps: dict[float, dict[int, np.ndarray]],
    target_times: list[float],
    element_ids: np.ndarray,
) -> dict[float, dict[int, np.ndarray]]:
    if not stress_maps:
        return {float(time_value): {} for time_value in target_times}
    source_times = np.asarray(sorted(float(time_value) for time_value in stress_maps), dtype=float)
    result: dict[float, dict[int, np.ndarray]] = {}
    for target_time in target_times:
        t = float(target_time)
        if t <= source_times[0] + 1.0e-13:
            if abs(t) <= 1.0e-13 and source_times[0] > 1.0e-13:
                result[t] = {int(element_id): np.zeros(6, dtype=float) for element_id in element_ids}
            else:
                result[t] = {int(k): np.asarray(v, dtype=float).copy() for k, v in stress_maps[float(source_times[0])].items()}
            continue
        if t >= source_times[-1] - 1.0e-13:
            result[t] = {int(k): np.asarray(v, dtype=float).copy() for k, v in stress_maps[float(source_times[-1])].items()}
            continue
        right = int(np.searchsorted(source_times, t, side="left"))
        left = right - 1
        t0 = float(source_times[left])
        t1 = float(source_times[right])
        alpha = 0.0 if abs(t1 - t0) <= 1.0e-15 else (t - t0) / (t1 - t0)
        left_map = stress_maps.get(t0, {})
        right_map = stress_maps.get(t1, {})
        merged: dict[int, np.ndarray] = {}
        for element_id in element_ids:
            key = int(element_id)
            if key in left_map and key in right_map:
                merged[key] = (1.0 - alpha) * np.asarray(left_map[key], dtype=float) + alpha * np.asarray(right_map[key], dtype=float)
        result[t] = merged
    return result


def _export_sfc_case(out_dir: Path, *, case: str, resolution: int, quick: bool, frame_stride: int) -> Row:
    model = build_c3d8_model(case=case, resolution=resolution, quick=quick)
    _rows, _u_final, _vm_final, frames = _native_c3d8_linear_dynamic_history(model)
    stem = f"sfc_{case}_r{resolution}"
    outputs = write_hex_frame_series(
        out_dir / case / "sfc",
        stem=stem,
        reference_points=model.X,
        elements=model.elements,
        frames=frames,
        frame_stride=frame_stride,
        cell_scalar_fn=lambda _time, U, model=model: _phase9_hex_cell_fields(model, U),
        source_label=f"SFC native dynamic {case}",
    )
    return {
        "case": case,
        "solver": "sfc",
        "resolution": resolution,
        "source": "native_sfc_rerun",
        "frame_count": int(outputs["frame_count"]),
        "pvd": _display_path(Path(outputs["pvd"])),
        "first_frame": _display_path(Path(outputs["first_frame"])),
    }


def _load_calculix_frames(
    out_dir: Path,
    *,
    source_dir: Path,
    case: str,
    resolution: int,
    quick: bool,
    frame_stride: int | None = None,
) -> tuple[Any, Path, list[tuple[float, np.ndarray]], dict[float, dict[int, np.ndarray]]] | Row:
    model = build_c3d8_model(case=case, resolution=resolution, quick=quick)
    dat_path = _calculix_dat_path(source_dir, case, resolution)
    if not dat_path.exists():
        return {
            "case": case,
            "solver": "calculix",
            "resolution": resolution,
            "source": _display_path(dat_path),
            "frame_count": 0,
            "pvd": "",
            "first_frame": "",
            "status": "missing_dat",
        }
    displacements = _parse_nodal_vectors(dat_path, model.node_ids, quantity="u")
    stresses = _parse_element_stress(dat_path)
    frames = [(float(time_value), U) for time_value, U in sorted(displacements.items())]
    return model, dat_path, frames, stresses


def _export_calculix_case(
    out_dir: Path,
    *,
    source_dir: Path,
    case: str,
    resolution: int,
    quick: bool,
    frame_stride: int,
) -> Row:
    loaded = _load_calculix_frames(
        out_dir,
        source_dir=source_dir,
        case=case,
        resolution=resolution,
        quick=quick,
    )
    if isinstance(loaded, dict):
        return loaded
    model, dat_path, frames, stresses = loaded
    stem = f"calculix_{case}_r{resolution}"
    outputs = write_hex_frame_series(
        out_dir / case / "calculix",
        stem=stem,
        reference_points=model.X,
        elements=model.elements,
        frames=frames,
        frame_stride=frame_stride,
        cell_scalar_fn=lambda time_value, U, model=model, stresses=stresses: _phase9_hex_cell_fields(
            model,
            U,
            stresses.get(float(time_value), {}),
        ),
        source_label=f"CalculiX dynamic {case}",
    )
    return {
        "case": case,
        "solver": "calculix",
        "resolution": resolution,
        "source": _display_path(dat_path),
        "frame_count": int(outputs["frame_count"]),
        "pvd": _display_path(Path(outputs["pvd"])),
        "first_frame": _display_path(Path(outputs["first_frame"])),
        "status": "exported",
    }


def _export_common_time_case(
    out_dir: Path,
    *,
    source_dir: Path,
    case: str,
    resolution: int,
    quick: bool,
    frame_stride: int,
) -> list[Row]:
    model = build_c3d8_model(case=case, resolution=resolution, quick=quick)
    _rows, _u_final, _vm_final, sfc_frames = _native_c3d8_linear_dynamic_history(model)
    loaded = _load_calculix_frames(
        out_dir,
        source_dir=source_dir,
        case=case,
        resolution=resolution,
        quick=quick,
    )
    if isinstance(loaded, dict):
        return [
            {
                "case": case,
                "solver": "common_time",
                "resolution": resolution,
                "source": loaded.get("source", ""),
                "frame_count": 0,
                "pvd": "",
                "first_frame": "",
                "status": "missing_calculix_dat",
            }
        ]
    _calc_model, dat_path, raw_calculix_frames, raw_calculix_stresses = loaded
    common_sfc_frames = _select_by_stride(sfc_frames, frame_stride)
    target_times = [float(time_value) for time_value, _U in common_sfc_frames]
    calculix_frames = _interpolate_displacement_frames(
        _with_initial_zero_frame(raw_calculix_frames, node_count=model.X.shape[0]),
        target_times,
    )
    calculix_stresses = _interpolate_stress_maps(raw_calculix_stresses, target_times, model.element_ids)

    common_root = out_dir / case / "common_time_frames"
    sfc_outputs = write_hex_frame_series(
        common_root / "sfc",
        stem=f"common_sfc_{case}_r{resolution}",
        reference_points=model.X,
        elements=model.elements,
        frames=common_sfc_frames,
        frame_stride=1,
        cell_scalar_fn=lambda _time, U, model=model: _phase9_hex_cell_fields(model, U),
        source_label=f"SFC common-time dynamic {case}",
    )
    calc_outputs = write_hex_frame_series(
        common_root / "calculix",
        stem=f"common_calculix_{case}_r{resolution}",
        reference_points=model.X,
        elements=model.elements,
        frames=calculix_frames,
        frame_stride=1,
        cell_scalar_fn=lambda time_value, U, model=model, stresses=calculix_stresses: _phase9_hex_cell_fields(
            model,
            U,
            stresses.get(float(time_value), {}),
        ),
        source_label=f"CalculiX common-time dynamic {case}",
    )

    error_dir = common_root / "error"
    error_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[tuple[float, Path]] = []
    for frame_index, ((time_value, U_sfc), (_calc_time, U_calc)) in enumerate(zip(common_sfc_frames, calculix_frames, strict=True)):
        t = float(time_value)
        sfc_fields = _phase9_hex_cell_fields(model, U_sfc)
        calc_fields = _phase9_hex_cell_fields(model, U_calc, calculix_stresses.get(t, {}))
        point_error = U_sfc - U_calc
        cell_error = {
            "cell_displacement_error_magnitude": np.asarray(
                [float(np.mean(np.linalg.norm(point_error[element], axis=1))) for element in model.elements],
                dtype=float,
            ),
            "von_mises_abs_error": np.abs(np.asarray(sfc_fields["von_mises"]) - np.asarray(calc_fields["von_mises"])),
            "engineering_strain_norm_abs_error": np.abs(
                np.asarray(sfc_fields["engineering_strain_norm"]) - np.asarray(calc_fields["engineering_strain_norm"])
            ),
        }
        frame_path = error_dir / f"common_error_{case}_r{resolution}_{frame_index:04d}.vtk"
        write_legacy_hex_vtk_frame(
            frame_path,
            model.X + U_sfc,
            model.elements,
            point_vectors={
                "sfc_displacement": U_sfc,
                "calculix_displacement": U_calc,
                "displacement_error": point_error,
            },
            point_scalars={
                "sfc_displacement_magnitude": np.linalg.norm(U_sfc, axis=1),
                "calculix_displacement_magnitude": np.linalg.norm(U_calc, axis=1),
                "displacement_error_magnitude": np.linalg.norm(point_error, axis=1),
            },
            cell_scalars=cell_error,
            title=f"SFC minus CalculiX common-time error {case} frame {frame_index} t={t:.12e}",
        )
        datasets.append((t, frame_path))
    error_pvd = error_dir / f"common_error_{case}_r{resolution}.pvd"
    write_pvd(error_pvd, datasets)

    common_count = len(common_sfc_frames)
    return [
        {
            "case": case,
            "solver": "sfc_common_time",
            "resolution": resolution,
            "source": "native_sfc_rerun_resampled_to_sfc_times",
            "frame_count": common_count,
            "pvd": _display_path(Path(sfc_outputs["pvd"])),
            "first_frame": _display_path(Path(sfc_outputs["first_frame"])),
            "status": "exported",
        },
        {
            "case": case,
            "solver": "calculix_common_time",
            "resolution": resolution,
            "source": _display_path(dat_path),
            "frame_count": int(calc_outputs["frame_count"]),
            "pvd": _display_path(Path(calc_outputs["pvd"])),
            "first_frame": _display_path(Path(calc_outputs["first_frame"])),
            "status": "interpolated_to_sfc_times",
        },
        {
            "case": case,
            "solver": "sfc_minus_calculix_common_error",
            "resolution": resolution,
            "source": "common_time_sfc_minus_interpolated_calculix",
            "frame_count": common_count,
            "pvd": _display_path(error_pvd),
            "first_frame": _display_path(datasets[0][1]) if datasets else "",
            "status": "exported",
        },
    ]


def run_export(
    out_dir: Path,
    *,
    source_dir: Path,
    cases: list[str],
    resolution: int,
    quick: bool,
    frame_stride: int,
    common_time_frames: bool,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    for case in cases:
        rows.append(_export_sfc_case(out_dir, case=case, resolution=resolution, quick=quick, frame_stride=frame_stride))
        rows.append(
            _export_calculix_case(
                out_dir,
                source_dir=source_dir,
                case=case,
                resolution=resolution,
                quick=quick,
                frame_stride=frame_stride,
            )
        )
        if common_time_frames:
            rows.extend(
                _export_common_time_case(
                    out_dir,
                    source_dir=source_dir,
                    case=case,
                    resolution=resolution,
                    quick=quick,
                    frame_stride=frame_stride,
                )
            )
    manifest = out_dir / "dynamic_vtk_frame_manifest.csv"
    _write_csv(manifest, rows)
    summary = out_dir / "dynamic_vtk_frame_summary.md"
    lines = [
        "# Dynamic VTK Frame Export",
        "",
        "This directory contains continuously numbered legacy VTK frame sequences and PVD time indexes for ParaView animation.",
        "",
        "`sfc` and `calculix` rows preserve each solver's raw output sampling. `*_common_time` rows are resampled to the SFC time grid for side-by-side animation, and `sfc_minus_calculix_common_error` stores pointwise displacement-error fields on the same frame count.",
        "",
        "| case | solver | frames | status | pvd |",
        "| --- | --- | ---: | --- | --- |",
    ]
    lines.extend(
        f"| {row['case']} | {row['solver']} | {row['frame_count']} | {row.get('status', '')} | `{row.get('pvd', '')}` |"
        for row in rows
    )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"manifest": manifest, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "numerical_experiments" / "dynamic_vtk_frames")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "paper" / "numerical_experiments" / "phase9_full_contact_validation",
        help="Directory containing c3d8_dynamic_trajectory/calculix_runs.",
    )
    parser.add_argument("--case", action="append", choices=["block_plane_c3d8", "block_block_c3d8"], help="Case to export; may be repeated.")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Use quick model duration for regenerated SFC frames.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--no-common-time-frames",
        action="store_true",
        help="Disable common-time SFC/CalculiX/error frame export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = args.case or ["block_plane_c3d8", "block_block_c3d8"]
    outputs = run_export(
        args.out_dir,
        source_dir=args.source_dir,
        cases=list(cases),
        resolution=int(args.resolution),
        quick=bool(args.quick),
        frame_stride=int(args.frame_stride),
        common_time_frames=not bool(args.no_common_time_frames),
    )
    print("Dynamic VTK frame export complete.")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
