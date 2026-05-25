"""Full flexible-gear implicit Lagrangian-SDF contact runner.

This validation runner keeps the complete two-gear TET4 volume meshes and
selects only the currently relevant contact surface patch for the Lagrangian
SDF contact constraints.  The active-patch selection is a broad-phase step; the
FEM solve, RP-MPC kinematics, contact gap/Jacobian, and reaction diagnostics
remain internal to SFC.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flexible_gear_explicit_sdf_comparison import DEFAULT_SOURCE, GearInputModel, GearMesh, parse_gear_input  # noqa: E402
from validation.run_flexible_gear_implicit_lagrangian_sdf_comparison import (  # noqa: E402
    ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
    CroppedGearPair,
    CroppedGearPatch,
    _face_centroids,
    _orient_faces_toward,
    _resolve_abaqus_command,
    _run_command,
    _write_abaqus_alignment_deck,
    _write_csv,
    run_abaqus_alignment,
    compare_histories,
    plot_alignment_curves,
    solve_sfc_cropped_pair,
    solve_sfc_cropped_pair_hard_contact,
    solve_sfc_source_drive_pair,
)

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "flexible_gear_full_lagrangian_sdf"


def _read_csv_rows(path: Path) -> list[Row]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _max_column(rows: list[Row], column: str) -> float:
    values: list[float] = []
    for row in rows:
        value = row.get(column, "")
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values, default=0.0)


def _manifest_column_available(rows: list[Row], column: str) -> bool:
    return bool(rows) and column in rows[0]


def write_animation_color_ranges(
    out_dir: Path,
    *,
    sfc_manifest: Path | None = None,
    abaqus_manifest: Path | None = None,
) -> Path:
    """Write fixed ParaView color ranges for the SFC/Abaqus animation series."""

    sfc_rows = _read_csv_rows(sfc_manifest or (out_dir / "sfc_vtk" / "sfc_manifest.csv"))
    abaqus_rows = _read_csv_rows(abaqus_manifest or (out_dir / "abaqus_vtk" / "abaqus_manifest.csv"))
    range_rows: list[Row] = [
        {
            "field": "displacement_magnitude",
            "recommended_min": 0.0,
            "recommended_max": max(
                _max_column(sfc_rows, "max_displacement_magnitude"),
                _max_column(abaqus_rows, "max_displacement_magnitude"),
            ),
            "sfc_column": "max_displacement_magnitude",
            "abaqus_column": "max_displacement_magnitude",
            "paraview_note": "Use this fixed range for both SFC and Abaqus; do not rescale every timestep.",
        },
        {
            "field": "von_mises",
            "recommended_min": 0.0,
            "recommended_max": max(_max_column(sfc_rows, "max_von_mises"), _max_column(abaqus_rows, "max_von_mises")),
            "sfc_column": "max_von_mises",
            "abaqus_column": "max_von_mises",
            "paraview_note": "Fixed global range prevents stress-cloud flicker caused by per-frame color rescaling.",
        },
        {
            "field": "von_mises_nodeavg",
            "recommended_min": 0.0,
            "recommended_max": max(
                _max_column(sfc_rows, "max_von_mises_nodeavg"),
                _max_column(abaqus_rows, "max_von_mises_nodeavg"),
            ),
            "sfc_column": "max_von_mises_nodeavg",
            "abaqus_column": "max_von_mises_nodeavg",
            "paraview_note": "Use this point-data field for smooth animations while retaining cell von_mises for quantitative checks.",
        },
        {
            "field": "strain_norm",
            "recommended_min": 0.0,
            "recommended_max": max(_max_column(sfc_rows, "max_strain_norm"), _max_column(abaqus_rows, "max_le_norm")),
            "sfc_column": "max_strain_norm",
            "abaqus_column": "max_le_norm",
            "paraview_note": "SFC field is named strain_norm; Abaqus field is named logarithmic_strain_norm.",
        },
        {
            "field": "strain_norm_nodeavg / logarithmic_strain_norm_nodeavg",
            "recommended_min": 0.0,
            "recommended_max": max(
                _max_column(sfc_rows, "max_strain_norm_nodeavg"),
                _max_column(abaqus_rows, "max_le_norm_nodeavg"),
            ),
            "sfc_column": "max_strain_norm_nodeavg",
            "abaqus_column": "max_le_norm_nodeavg",
            "paraview_note": "Use the node-averaged point-data field for animation; use cell data for exact element statistics.",
        },
        {
            "field": "equivalent_elastic_strain_nodeavg",
            "recommended_min": 0.0,
            "recommended_max": max(
                _max_column(sfc_rows, "max_equivalent_elastic_strain_nodeavg"),
                _max_column(abaqus_rows, "max_equivalent_elastic_strain_nodeavg"),
            ),
            "sfc_column": "max_equivalent_elastic_strain_nodeavg",
            "abaqus_column": "max_equivalent_elastic_strain_nodeavg",
            "paraview_note": "Use for paper-facing strain curves when raw SFC and Abaqus strain tensor measures differ.",
        },
    ]
    path = out_dir / "animation_fixed_color_ranges.csv"
    _write_csv(path, range_rows)
    return path


def write_paraview_animation_setup(
    out_dir: Path,
    *,
    color_ranges: Path,
    sfc_pvd: Path | None = None,
    abaqus_pvd: Path | None = None,
) -> Path:
    """Write a ParaView helper that applies fixed stress/strain color ranges.

    The VTK/PVD files carry the raw fields only; ParaView otherwise tends to
    rescale the color transfer function per timestep.  The generated script is
    post-processing-only and makes the paper animation use the same global
    range and the same node-averaged field at every frame.
    """

    rows = _read_csv_rows(color_ranges)
    ranges: dict[str, tuple[float, float]] = {}
    for row in rows:
        field = str(row.get("field", "")).strip()
        if not field or "/" in field:
            continue
        try:
            ranges[field] = (
                float(row.get("recommended_min", 0.0) or 0.0),
                float(row.get("recommended_max", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            continue
    stable_fields = [
        "displacement_magnitude",
        "von_mises_nodeavg",
        "equivalent_elastic_strain_nodeavg",
    ]
    embedded_ranges = {
        field: ranges.get(field, (0.0, 1.0))
        for field in stable_fields
    }

    def script_relative_path(path: Path | None) -> str:
        if path is None:
            return ""
        return Path(os.path.relpath(Path(path).resolve(), out_dir.resolve())).as_posix()

    sfc_path = script_relative_path(sfc_pvd)
    abaqus_path = script_relative_path(abaqus_pvd)
    script = f'''"""ParaView setup for fixed-range SFC/Abaqus gear animations.

Run with ParaView's Python shell or pvpython from the result directory:

    pvpython {Path("paraview_fixed_range_animation.py").as_posix()}

The script intentionally uses point-data node-averaged fields for stress and
strain animation.  Cell-data fields remain in the VTK files for quantitative
element diagnostics.
"""

from pathlib import Path

from paraview.simple import *  # noqa: F401,F403

SFC_PVD = r"{sfc_path}"
ABAQUS_PVD = r"{abaqus_path}"
FIELD_RANGES = {embedded_ranges!r}
DEFAULT_FIELD = "von_mises_nodeavg"
ROOT = Path(__file__).resolve().parent


def _open_pvd(path):
    return PVDReader(FileName=str((ROOT / path).resolve())) if path else None


def _display(source, view, field_name):
    if source is None:
        return None
    shown = Show(source, view)
    ColorBy(shown, ("POINTS", field_name))
    shown.RescaleTransferFunctionToDataRange(False, True)
    shown.SetScalarBarVisibility(view, True)
    lut = GetColorTransferFunction(field_name)
    vmin, vmax = FIELD_RANGES.get(field_name, (0.0, 1.0))
    lut.RescaleTransferFunction(float(vmin), float(vmax))
    opacity = GetOpacityTransferFunction(field_name)
    opacity.RescaleTransferFunction(float(vmin), float(vmax))
    return shown


view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1400, 850]
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]

sfc = _open_pvd(SFC_PVD)
abaqus = _open_pvd(ABAQUS_PVD)
_display(sfc, view, DEFAULT_FIELD)
_display(abaqus, view, DEFAULT_FIELD)
ResetCamera(view)

print("Loaded fixed-range animation setup.")
print("Stable fields:", ", ".join(FIELD_RANGES))
print("Default field:", DEFAULT_FIELD, FIELD_RANGES.get(DEFAULT_FIELD))
print("Switch DEFAULT_FIELD to displacement_magnitude or equivalent_elastic_strain_nodeavg for other clouds.")
'''
    path = out_dir / "paraview_fixed_range_animation.py"
    path.write_text(script, encoding="utf-8")
    return path


def compare_animation_manifests(
    *,
    sfc_manifest: Path,
    abaqus_manifest: Path,
    out_csv: Path,
    out_png: Path,
) -> list[Row]:
    """Compare SFC and Abaqus VTK animation metrics on the SFC frame times.

    The VTK animation manifests are field-output summaries, not solver inputs.
    They are useful for the paper-facing displacement/stress/strain curves
    because both solvers can be compared from the exact fields that are opened
    in ParaView.
    """

    sfc_rows = _read_csv_rows(sfc_manifest)
    abaqus_rows = _read_csv_rows(abaqus_manifest)
    if not sfc_rows:
        raise RuntimeError(f"SFC VTK manifest has no frames: {sfc_manifest}")
    if not abaqus_rows:
        raise RuntimeError(f"Abaqus VTK manifest has no frames: {abaqus_manifest}")
    t_sfc = np.asarray([float(row.get("time", 0.0) or 0.0) for row in sfc_rows], dtype=float)
    t_abaqus = np.asarray([float(row.get("time", 0.0) or 0.0) for row in abaqus_rows], dtype=float)
    strain_metric = (
        ("p95_equivalent_elastic_strain_nodeavg", "p95_equivalent_elastic_strain_nodeavg", "p95 node-averaged equivalent elastic strain")
        if _manifest_column_available(sfc_rows, "p95_equivalent_elastic_strain_nodeavg")
        and _manifest_column_available(abaqus_rows, "p95_equivalent_elastic_strain_nodeavg")
        else ("max_equivalent_elastic_strain_nodeavg", "max_equivalent_elastic_strain_nodeavg", "max node-averaged equivalent elastic strain")
        if _manifest_column_available(sfc_rows, "max_equivalent_elastic_strain_nodeavg")
        and _manifest_column_available(abaqus_rows, "max_equivalent_elastic_strain_nodeavg")
        else ("max_strain_norm_nodeavg", "max_le_norm_nodeavg", "max node-averaged strain norm")
    )
    stress_metric = (
        ("p95_von_mises_nodeavg", "p95_von_mises_nodeavg", "p95 node-averaged von Mises")
        if _manifest_column_available(sfc_rows, "p95_von_mises_nodeavg")
        and _manifest_column_available(abaqus_rows, "p95_von_mises_nodeavg")
        else ("max_von_mises_nodeavg", "max_von_mises_nodeavg", "max node-averaged von Mises")
    )
    metrics = [
        ("max_displacement_magnitude", "max_displacement_magnitude", "max displacement magnitude"),
        stress_metric,
        strain_metric,
    ]
    diagnostic_metrics = list(metrics)
    for sfc_key, abaqus_key, label in (
        ("max_von_mises_nodeavg", "max_von_mises_nodeavg", "max node-averaged von Mises"),
        (
            "max_equivalent_elastic_strain_nodeavg",
            "max_equivalent_elastic_strain_nodeavg",
            "max node-averaged equivalent elastic strain",
        ),
        ("mean_von_mises_nodeavg", "mean_von_mises_nodeavg", "mean node-averaged von Mises"),
        (
            "mean_equivalent_elastic_strain_nodeavg",
            "mean_equivalent_elastic_strain_nodeavg",
            "mean node-averaged equivalent elastic strain",
        ),
    ):
        if _manifest_column_available(sfc_rows, sfc_key) and _manifest_column_available(abaqus_rows, abaqus_key):
            diagnostic_metrics.append((sfc_key, abaqus_key, label))
    for sfc_key, abaqus_key, label in (
        ("rp1_rotation_z_rad", "rp1_rotation_z_rad", "RP1 rotation about z"),
        ("rp2_rotation_z_rad", "rp2_rotation_z_rad", "RP2 rotation about z"),
        ("rp1_angular_velocity_z_rad_per_s", "rp1_angular_velocity_z_rad_per_s", "RP1 angular velocity about z"),
        ("rp2_angular_velocity_z_rad_per_s", "rp2_angular_velocity_z_rad_per_s", "RP2 angular velocity about z"),
    ):
        if _manifest_column_available(sfc_rows, sfc_key) and _manifest_column_available(abaqus_rows, abaqus_key):
            diagnostic_metrics.append((sfc_key, abaqus_key, label))
    for object_id in (1, 2):
        for sfc_key, abaqus_key, label in (
            (
                f"max_displacement_magnitude_object{object_id}",
                f"max_displacement_magnitude_object{object_id}",
                f"max displacement magnitude object {object_id}",
            ),
            (
                f"max_von_mises_nodeavg_object{object_id}",
                f"max_von_mises_nodeavg_object{object_id}",
                f"max node-averaged von Mises object {object_id}",
            ),
            (
                f"p95_von_mises_nodeavg_object{object_id}",
                f"p95_von_mises_nodeavg_object{object_id}",
                f"p95 node-averaged von Mises object {object_id}",
            ),
            (
                f"max_equivalent_elastic_strain_nodeavg_object{object_id}",
                f"max_equivalent_elastic_strain_nodeavg_object{object_id}",
                f"max node-averaged equivalent elastic strain object {object_id}",
            ),
            (
                f"p95_equivalent_elastic_strain_nodeavg_object{object_id}",
                f"p95_equivalent_elastic_strain_nodeavg_object{object_id}",
                f"p95 node-averaged equivalent elastic strain object {object_id}",
            ),
        ):
            if _manifest_column_available(sfc_rows, sfc_key) and _manifest_column_available(abaqus_rows, abaqus_key):
                diagnostic_metrics.append((sfc_key, abaqus_key, label))
    rows: list[Row] = []
    for frame, (t_value, sfc_row) in enumerate(zip(t_sfc, sfc_rows, strict=True)):
        row: Row = {"frame": int(frame), "time": float(t_value)}
        for sfc_key, abaqus_key, _label in diagnostic_metrics:
            sfc_value = float(sfc_row.get(sfc_key, 0.0) or 0.0)
            abaqus_series = np.asarray([float(item.get(abaqus_key, 0.0) or 0.0) for item in abaqus_rows], dtype=float)
            abaqus_value = float(np.interp(float(t_value), t_abaqus, abaqus_series))
            row[f"sfc_{sfc_key}"] = sfc_value
            row[f"abaqus_{abaqus_key}"] = abaqus_value
            row[f"{sfc_key}_abs_error"] = abs(sfc_value - abaqus_value)
            row[f"{sfc_key}_rel_error"] = abs(sfc_value - abaqus_value) / max(abs(abaqus_value), 1.0e-12)
        rows.append(row)
    _write_csv(out_csv, rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.75), constrained_layout=True)
    for ax, (sfc_key, abaqus_key, label) in zip(axes, metrics, strict=True):
        sfc_y = np.asarray([float(row.get(sfc_key, 0.0) or 0.0) for row in sfc_rows], dtype=float)
        abaqus_y = np.asarray([float(row.get(abaqus_key, 0.0) or 0.0) for row in abaqus_rows], dtype=float)
        interp_abaqus_y = np.asarray(
            [float(row[f"abaqus_{abaqus_key}"]) for row in rows],
            dtype=float,
        )
        rel_key = f"{sfc_key}_rel_error"
        final_error = float(rows[-1].get(rel_key, 0.0)) if rows else 0.0
        ax.plot(t_sfc, interp_abaqus_y, color="#1f77b4", linewidth=1.7, label="Abaqus/Standard")
        if t_abaqus.size <= 80:
            in_window = t_abaqus <= (float(np.max(t_sfc)) if t_sfc.size else 0.0) + 1.0e-14
            if np.any(in_window):
                ax.plot(
                    t_abaqus[in_window],
                    abaqus_y[in_window],
                    color="#1f77b4",
                    linestyle="none",
                    marker="o",
                    markersize=2.5,
                    alpha=0.75,
                )
        ax.plot(
            t_sfc,
            sfc_y,
            color="#d95f02",
            linewidth=1.7,
            linestyle="--",
            label=f"SFC ({final_error * 100:.2f}% final err.)",
        )
        if t_sfc.size > 1:
            ax.fill_between(
                t_sfc,
                np.minimum(sfc_y, interp_abaqus_y),
                np.maximum(sfc_y, interp_abaqus_y),
                color="#d95f02",
                alpha=0.10,
                linewidth=0.0,
            )
        ax.set_title(label, fontsize=9.5)
        ax.set_xlabel("time (s)", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.legend(loc="best", fontsize=7.5, frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return rows


def compare_sfc_history_to_abaqus_manifest(
    *,
    sfc_history: Path,
    abaqus_manifest: Path,
    out_csv: Path,
) -> list[Row]:
    """Compare SFC scalar history with Abaqus VTK manifest metrics by time.

    This avoids writing large SFC VTK files for every short diagnostic run while
    still checking the displacement, stress, strain, and contact fields against
    the already-exported Abaqus reference frames.
    """

    sfc_rows = _read_csv_rows(sfc_history)
    abaqus_rows = _read_csv_rows(abaqus_manifest)
    if not sfc_rows or not abaqus_rows:
        _write_csv(out_csv, [])
        return []
    t_abaqus = np.asarray([float(row.get("time", 0.0) or 0.0) for row in abaqus_rows], dtype=float)
    metric_pairs = [
        ("max_displacement_norm", "max_displacement_magnitude"),
        ("p95_von_mises_nodeavg", "p95_von_mises_nodeavg"),
        ("p95_equivalent_elastic_strain_nodeavg", "p95_equivalent_elastic_strain_nodeavg"),
        ("max_von_mises_nodeavg", "max_von_mises_nodeavg"),
        ("max_equivalent_elastic_strain_nodeavg", "max_equivalent_elastic_strain_nodeavg"),
        ("mean_von_mises_nodeavg", "mean_von_mises_nodeavg"),
        ("mean_equivalent_elastic_strain_nodeavg", "mean_equivalent_elastic_strain_nodeavg"),
        ("active_contact_node_count", "active_contact_node_count"),
        ("max_contact_pressure_nodeavg", "max_contact_pressure_nodeavg"),
        ("mean_active_contact_pressure_nodeavg", "mean_active_contact_pressure_nodeavg"),
    ]
    rows: list[Row] = []
    for frame, sfc_row in enumerate(sfc_rows):
        t_value = float(sfc_row.get("time", 0.0) or 0.0)
        row: Row = {"frame": int(frame), "time": t_value}
        for sfc_key, abaqus_key in metric_pairs:
            if sfc_key not in sfc_row or not _manifest_column_available(abaqus_rows, abaqus_key):
                continue
            try:
                sfc_value = float(sfc_row.get(sfc_key, 0.0) or 0.0)
                abaqus_series = np.asarray([float(item.get(abaqus_key, 0.0) or 0.0) for item in abaqus_rows], dtype=float)
            except (TypeError, ValueError):
                continue
            abaqus_value = float(np.interp(t_value, t_abaqus, abaqus_series))
            row[f"sfc_{sfc_key}"] = sfc_value
            row[f"abaqus_{abaqus_key}"] = abaqus_value
            row[f"{sfc_key}_abs_error"] = abs(sfc_value - abaqus_value)
            row[f"{sfc_key}_rel_error"] = abs(sfc_value - abaqus_value) / max(abs(abaqus_value), 1.0e-12)
        rows.append(row)
    _write_csv(out_csv, rows)
    return rows


def _selected_surface_entries_for_full_mesh(
    mesh: GearMesh,
    entries: tuple[tuple[int, str], ...],
    selected_surface_ids: np.ndarray,
) -> tuple[tuple[int, str], ...]:
    """Map selected original Abaqus surface entries to emitted full-mesh labels."""

    out: list[tuple[int, str]] = []
    for sid in np.asarray(selected_surface_ids, dtype=np.int64).reshape(-1):
        element_label, side = entries[int(sid)]
        new_element_label = int(mesh.element_label_to_index[int(element_label)]) + 1
        out.append((new_element_label, str(side).upper()))
    return tuple(out)


def _hub_indices(mesh: GearMesh, labels: tuple[int, ...]) -> np.ndarray:
    ids = [mesh.label_to_index[int(label)] for label in labels if int(label) in mesh.label_to_index]
    if not ids:
        raise ValueError("hub node set did not map to this gear mesh")
    return np.asarray(sorted(set(ids)), dtype=np.int64)


def _nearest_ids(tree: cKDTree, point: np.ndarray, *, k: int, count: int) -> np.ndarray:
    query_count = min(max(1, int(k)), max(1, int(count)))
    return np.asarray(tree.query(point, k=query_count)[1], dtype=np.int64).reshape(-1)


def _radius_expanded_ids(centroids: np.ndarray, seed: np.ndarray, base_ids: np.ndarray, *, radius_factor: float) -> np.ndarray:
    ids = np.asarray(base_ids, dtype=np.int64).reshape(-1)
    if float(radius_factor) <= 1.0 or ids.size == 0:
        return np.unique(ids)
    offsets = np.linalg.norm(centroids[ids] - seed, axis=1)
    base_radius = float(np.max(offsets)) if offsets.size else 0.0
    if base_radius <= 0.0 and centroids.shape[0] > 1:
        distances = np.linalg.norm(centroids - seed, axis=1)
        positive = distances[distances > 0.0]
        base_radius = float(np.min(positive)) if positive.size else 0.0
    radius = max(base_radius * float(radius_factor), base_radius)
    expanded = np.flatnonzero(np.linalg.norm(centroids - seed, axis=1) <= radius + 1.0e-12)
    return np.unique(np.concatenate([ids, expanded.astype(np.int64)]))


def _nearest_active_surface_ids(
    model: GearInputModel,
    *,
    active_faces_per_body: int,
    active_patch_radius_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    centroids1 = _face_centroids(model.gear1.nodes, model.gear1_contact_faces)
    centroids2 = _face_centroids(model.gear2.nodes, model.gear2_contact_faces)
    tree2 = cKDTree(centroids2)
    distances, ids2 = tree2.query(centroids1, k=1)
    seed1 = int(np.argmin(distances))
    seed2 = int(ids2[seed1])
    if int(active_faces_per_body) <= 0:
        ids1 = np.arange(centroids1.shape[0], dtype=np.int64)
        ids2_active = np.arange(centroids2.shape[0], dtype=np.int64)
        c1 = centroids1[seed1]
        c2 = centroids2[seed2]
        drive = c2 - c1
        drive /= max(float(np.linalg.norm(drive)), 1.0e-30)
        return ids1, ids2_active, float(distances[seed1]), drive
    n = int(active_faces_per_body)
    tree1 = cKDTree(centroids1)
    ids1 = _nearest_ids(tree1, centroids1[seed1], k=n, count=centroids1.shape[0])
    ids2_active = _nearest_ids(tree2, centroids2[seed2], k=n, count=centroids2.shape[0])
    ids1 = _radius_expanded_ids(centroids1, centroids1[seed1], ids1, radius_factor=active_patch_radius_factor)
    ids2_active = _radius_expanded_ids(centroids2, centroids2[seed2], ids2_active, radius_factor=active_patch_radius_factor)
    if float(active_patch_radius_factor) > 1.0:
        opposite_count = min(4, centroids2.shape[0])
        mapped2 = np.asarray(tree2.query(centroids1[ids1], k=opposite_count)[1], dtype=np.int64).reshape(-1)
        ids2_active = np.unique(np.concatenate([ids2_active, mapped2]))
        opposite_count = min(4, centroids1.shape[0])
        mapped1 = np.asarray(tree1.query(centroids2[ids2_active], k=opposite_count)[1], dtype=np.int64).reshape(-1)
        ids1 = np.unique(np.concatenate([ids1, mapped1]))
    c1 = np.mean(centroids1[ids1], axis=0)
    c2 = np.mean(centroids2[ids2_active], axis=0)
    drive = c2 - c1
    drive /= max(float(np.linalg.norm(drive)), 1.0e-30)
    return ids1, ids2_active, float(distances[seed1]), drive


def build_full_active_pair(
    model: GearInputModel,
    *,
    active_faces_per_body: int,
    active_patch_radius_factor: float = 1.0,
) -> CroppedGearPair:
    """Return a complete two-gear volume model with active contact surfaces."""

    ids1, ids2, initial_gap, drive = _nearest_active_surface_ids(
        model,
        active_faces_per_body=active_faces_per_body,
        active_patch_radius_factor=active_patch_radius_factor,
    )
    gear1_faces = _orient_faces_toward(model.gear1.nodes, model.gear1_contact_faces[ids1], drive)
    gear2_faces = _orient_faces_toward(model.gear2.nodes, model.gear2_contact_faces[ids2], -drive)
    gear1 = CroppedGearPatch(
        name="gear1",
        nodes=model.gear1.nodes,
        elements=model.gear1.elements,
        contact_faces=gear1_faces,
        support_nodes=_hub_indices(model.gear1, model.gear1_hub_labels),
        surface_entries=_selected_surface_entries_for_full_mesh(model.gear1, model.gear1_surface_entries, ids1),
        element_labels=np.arange(1, model.gear1.elements.shape[0] + 1, dtype=np.int64),
        rp=model.rp1,
    )
    gear2 = CroppedGearPatch(
        name="gear2",
        nodes=model.gear2.nodes,
        elements=model.gear2.elements,
        contact_faces=gear2_faces,
        support_nodes=_hub_indices(model.gear2, model.gear2_hub_labels),
        surface_entries=_selected_surface_entries_for_full_mesh(model.gear2, model.gear2_surface_entries, ids2),
        element_labels=np.arange(1, model.gear2.elements.shape[0] + 1, dtype=np.int64),
        rp=model.rp2,
    )
    return CroppedGearPair(gear1, gear2, drive, initial_gap)


def write_full_summary(path: Path, summary: Row, history_path: Path, *, abaqus_row: Row | None = None) -> None:
    lines = [
        "# Full Gear Lagrangian-SDF Alignment",
        "",
        "This run keeps the full gear volume meshes and uses an active contact surface patch.",
        "",
        f"- nodes/elements: {summary['nodes']} / {summary['elements']}",
        f"- active contact faces: {summary['gear1_contact_faces']} / {summary['gear2_contact_faces']}",
        f"- active patch radius factor: {float(summary.get('active_patch_radius_factor', 1.0)):.3f}",
        f"- support nodes: {summary['gear1_support_nodes']} / {summary['gear2_support_nodes']}",
        f"- linear solver: {summary.get('linear_solver', 'sparse')}",
        f"- contact mode: {summary.get('contact_mode', '')}",
        f"- source contact averaging: {summary.get('source_contact_averaging', '')}",
        f"- source contact kinematics: {summary.get('source_contact_kinematics', '')}",
        f"- source contact normal filter: {summary.get('source_contact_normal_filter', '')}",
        f"- source contact direction: {summary.get('source_contact_direction', '')}",
        f"- source contact projection: {summary.get('source_contact_projection', '')}",
        f"- source contact pair order: {summary.get('source_contact_pair_order', '')}",
        f"- source contact search radius: {summary.get('source_contact_search_radius', '')}",
        f"- source secondary normal min projection: {summary.get('source_secondary_normal_min_projection', '')}",
        f"- source secondary line hard distance limit: {summary.get('source_secondary_line_hard_distance_limit', '')}",
        f"- source secondary path tracking: {summary.get('source_secondary_path_tracking', '')}",
        f"- source active-set stability: {summary.get('source_contact_active_set_stability', '')}",
        f"- source max nonlinear iterations: {summary.get('source_max_iterations', '')}",
        f"- source converged steps: {summary.get('source_step_converged_count', '')}",
        f"- source iteration-limit steps: {summary.get('source_iteration_limit_reached_count', '')}",
        f"- source unstable accepted steps: {summary.get('source_unstable_accepted_count', '')}",
        f"- source contact footprint clipping: {summary.get('source_contact_footprint_clipping', '')}",
        f"- source internal kinematics: {summary.get('source_internal_kinematics', '')}",
        f"- source rotating inertia: {summary.get('source_rotating_inertia', '')}",
        f"- penalty solver: {summary.get('penalty_solver', '')}",
        f"- material linearization: {summary.get('material_linearization', '')}",
        f"- RP reaction definition: {summary.get('rp_reaction_definition', '')}",
        f"- SFC wall time: {float(summary['sfc_wall_seconds']):.6f} s",
        f"- timing internal+tangent: {float(summary.get('timing_internal_tangent_seconds', 0.0)):.6f} s",
        f"- timing effective system: {float(summary.get('timing_effective_system_seconds', 0.0)):.6f} s",
        f"- timing hard-contact solve: {float(summary.get('timing_hard_contact_solve_seconds', 0.0)):.6f} s",
        f"- timing contact linearization: {float(summary.get('timing_contact_linearization_seconds', 0.0)):.6f} s",
        f"- timing penalty base tangent: {float(summary.get('timing_base_tangent_seconds', 0.0)):.6f} s",
        f"- timing penalty residual: {float(summary.get('timing_penalty_residual_seconds', 0.0)):.6f} s",
        f"- timing penalty low-rank solve: {float(summary.get('timing_penalty_low_rank_seconds', 0.0)):.6f} s",
        f"- final max displacement norm: {float(summary.get('final_max_displacement_norm', 0.0)):.6e}",
        f"- final p95 von Mises: {float(summary.get('final_p95_von_mises', 0.0)):.6e}",
        f"- final p95 equivalent elastic strain: {float(summary.get('final_p95_equivalent_elastic_strain', 0.0)):.6e}",
        f"- final p95 node-averaged von Mises: {float(summary.get('final_p95_von_mises_nodeavg', 0.0)):.6e}",
        "- final p95 node-averaged equivalent elastic strain: "
        f"{float(summary.get('final_p95_equivalent_elastic_strain_nodeavg', 0.0)):.6e}",
        f"- final RP reaction norm: {float(summary.get('final_rp_force_norm', 0.0)):.6e}",
        f"- final min gap: {float(summary.get('final_min_gap', 0.0)):.6e}",
        f"- rotation rate about z: {float(summary.get('rotation_rate_z_rad_per_s', 0.0)):.6e} rad/s",
        f"- final rotation about z: {float(summary.get('final_rotation_z_rad', 0.0)):.6e} rad",
        f"- history CSV: `{history_path.name}`",
    ]
    if summary.get("sfc_vtk_pvd"):
        lines.extend(
            [
                f"- SFC VTK PVD: `{summary.get('sfc_vtk_pvd')}`",
                f"- SFC VTK frame count: {int(summary.get('sfc_vtk_frame_count', 0))}",
            ]
        )
    if abaqus_row is not None:
        lines.extend(
            [
                "",
                "## Abaqus Native-Contact Alignment",
                "",
                f"- Abaqus analysis wall time: {float(abaqus_row.get('abaqus_analysis_wall_seconds', 0.0)):.6f} s",
                f"- Abaqus export wall time: {float(abaqus_row.get('abaqus_export_wall_seconds', 0.0)):.6f} s",
                f"- alignment errors CSV: `{Path(str(summary.get('alignment_errors', ''))).name}`",
                f"- alignment figure: `{Path(str(summary.get('alignment_figure', ''))).name}`",
            ]
        )
    if summary.get("abaqus_vtk_pvd"):
        lines.extend(
            [
                "",
                "## VTK Animation",
                "",
                f"- Abaqus VTK PVD: `{summary.get('abaqus_vtk_pvd')}`",
                f"- Abaqus VTK frame count: {int(summary.get('abaqus_vtk_frame_count', 0))}",
            ]
        )
    if summary.get("animation_color_ranges"):
        lines.extend(
            [
                "",
                "## ParaView Color Ranges",
                "",
                f"- fixed color ranges: `{Path(str(summary.get('animation_color_ranges'))).name}`",
                f"- fixed-range setup script: `{Path(str(summary.get('paraview_animation_setup', ''))).name}`",
                "- Use fixed global ranges for stress/strain animations; per-frame auto-rescaling can look like stress flicker.",
            ]
        )
    if summary.get("animation_metric_errors"):
        lines.extend(
            [
                "",
                "## VTK Field-Curve Alignment",
                "",
                f"- VTK metric errors: `{Path(str(summary.get('animation_metric_errors'))).name}`",
                f"- VTK metric curves: `{Path(str(summary.get('animation_metric_figure'))).name}`",
                "- Curves use the same displacement/stress/strain fields written to the SFC and Abaqus VTK animations.",
            ]
        )
    if summary.get("history_metric_errors"):
        lines.extend(
            [
                "",
                "## History Field-Curve Alignment",
                "",
                f"- history metric errors: `{Path(str(summary.get('history_metric_errors'))).name}`",
                "- This lightweight check compares SFC scalar history against Abaqus VTK manifest fields without writing SFC VTK frames.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_abaqus_vtk_frames(
    *,
    out_dir: Path,
    abaqus_command: str | None,
    frame_stride: int,
    include_tensors: bool,
    young: float | None = None,
    poisson: float | None = None,
) -> Row:
    """Export the existing full-gear Abaqus ODB to a strided VTK/PVD series."""

    run_dir = out_dir / "abaqus_run"
    odb = run_dir / "cropped_gear_alignment.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    vtk_dir = out_dir / "abaqus_vtk"
    command = _resolve_abaqus_command(abaqus_command)
    script = ROOT / "validation" / "abaqus_odb_to_vtk.py"
    export_command = [
        command,
        "python",
        str(script.resolve()),
        "--odb",
        str(odb.resolve()),
        "--out-dir",
        str(vtk_dir.resolve()),
        "--stem",
        "abaqus",
        "--frame-stride",
        str(max(1, int(frame_stride))),
    ]
    if young is not None and poisson is not None:
        export_command.extend(["--young", f"{float(young):.16e}", "--poisson", f"{float(poisson):.16e}"])
    export_command.append("--include-tensors" if include_tensors else "--scalars-only")
    wall = _run_command(
        export_command,
        cwd=run_dir,
        log_path=out_dir / "abaqus_vtk_export_stdout.log",
    )
    manifest = vtk_dir / "abaqus_manifest.csv"
    frame_count = 0
    if manifest.exists():
        with manifest.open("r", newline="", encoding="ascii") as handle:
            frame_count = max(0, sum(1 for _ in handle) - 1)
    return {
        "abaqus_vtk_export_wall_seconds": float(wall),
        "abaqus_vtk_pvd": str(vtk_dir / "abaqus.pvd"),
        "abaqus_vtk_manifest": str(manifest),
        "abaqus_vtk_frame_count": int(frame_count),
        "abaqus_vtk_frame_stride": int(frame_stride),
        "abaqus_vtk_include_tensors": bool(include_tensors),
    }


def run_full_gear(
    *,
    source: Path,
    out_dir: Path,
    active_faces_per_body: int,
    active_patch_radius_factor: float,
    duration: float,
    dt: float,
    target_overclosure: float,
    rotation_rate_z: float,
    pressure_stiffness: float,
    hard_max_iterations: int,
    contact_mode: str,
    run_abaqus: bool,
    drive_mode: str = "closure",
    hht_alpha: float = ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
    tet4_mass_kind: str = "consistent",
    history_frame_stride: int = 1,
    use_source_timing: bool = False,
    abaqus_command: str | None = None,
    write_sfc_vtk: bool = False,
    sfc_vtk_dir: Path | None = None,
    vtk_frame_stride: int = 1,
    vtk_include_tensors: bool = True,
    source_stress_postprocess: str = "linear_corotated",
    source_contact_averaging: str = "none",
    source_contact_kinematics: str = "linearized_mpc",
    source_contact_normal_filter: str = "none",
    source_contact_direction: str = "master",
    source_contact_projection: str = "closest_feature",
    source_contact_pair_order: str = "gear2_slave",
    source_contact_search_radius: float | None = None,
    source_secondary_normal_min_projection: float = 0.0,
    source_secondary_line_distance_limit: float | None = None,
    source_secondary_line_hard_distance_limit: float | None = None,
    source_secondary_path_tracking: bool = False,
    source_contact_active_set_stability: bool = True,
    source_contact_footprint_clipping: bool = False,
    source_internal_kinematics: str = "linearized_mpc",
    source_rotating_inertia: str = "none",
    source_max_iterations: int = 16,
    source_checkpoint_path: Path | None = None,
    resume_source_checkpoint: bool = False,
    source_checkpoint_stride: int = 10,
    export_abaqus_vtk: bool = False,
    abaqus_vtk_manifest: Path | None = None,
) -> tuple[list[Row], Row]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = parse_gear_input(source)
    drive = str(drive_mode).lower()
    if drive not in {"closure", "source_inp"}:
        raise ValueError("drive_mode must be 'closure' or 'source_inp'")
    source_step_matched = bool(use_source_timing)
    if source_step_matched:
        duration = float(model.dynamic_duration)
        dt = float(model.dynamic_initial_dt)
    pair = build_full_active_pair(
        model,
        active_faces_per_body=active_faces_per_body,
        active_patch_radius_factor=active_patch_radius_factor,
    )
    mode = str(contact_mode).lower()
    if mode not in {"penalty", "hard"}:
        raise ValueError("contact_mode must be 'penalty' or 'hard'")
    if drive == "source_inp":
        if mode == "hard":
            raise ValueError("source_inp drive currently supports SFC penalty contact; use --contact-mode penalty")
        history, summary = solve_sfc_source_drive_pair(
            pair,
            young=model.young,
            poisson=model.poisson,
            density=model.density,
            pressure_stiffness=pressure_stiffness,
            duration=duration,
            dt=dt,
            gear1_angular_velocity_z=model.gear1_angular_velocity_z,
            gear2_torque_z=model.gear2_torque_z,
            hht_alpha=float(hht_alpha),
            tet4_mass_kind=str(tet4_mass_kind),
            max_iterations=max(1, int(source_max_iterations)),
            vtk_out_dir=(sfc_vtk_dir if sfc_vtk_dir is not None else out_dir / "sfc_vtk") if write_sfc_vtk else None,
            vtk_frame_stride=max(1, int(vtk_frame_stride)),
            history_frame_stride=max(1, int(history_frame_stride)),
            vtk_stem="sfc",
            vtk_include_tensors=bool(vtk_include_tensors),
            source_stress_postprocess=str(source_stress_postprocess),
            source_contact_averaging=str(source_contact_averaging),
            source_contact_kinematics=str(source_contact_kinematics),
            source_contact_normal_filter=str(source_contact_normal_filter),
            source_contact_direction=str(source_contact_direction),
            source_contact_projection=str(source_contact_projection),
            source_contact_pair_order=str(source_contact_pair_order),
            source_contact_search_radius=source_contact_search_radius,
            source_secondary_normal_min_projection=float(source_secondary_normal_min_projection),
            source_secondary_line_distance_limit=source_secondary_line_distance_limit,
            source_secondary_line_hard_distance_limit=source_secondary_line_hard_distance_limit,
            source_secondary_path_tracking=bool(source_secondary_path_tracking),
            source_contact_active_set_stability=bool(source_contact_active_set_stability),
            source_contact_footprint_clipping=bool(source_contact_footprint_clipping),
            source_internal_kinematics=str(source_internal_kinematics),
            source_rotating_inertia=str(source_rotating_inertia),
            source_checkpoint_path=source_checkpoint_path,
            resume_source_checkpoint=bool(resume_source_checkpoint),
            source_checkpoint_stride=int(source_checkpoint_stride),
        )
    elif mode == "hard":
        history, summary = solve_sfc_cropped_pair_hard_contact(
            pair,
            young=model.young,
            poisson=model.poisson,
            density=model.density,
            pressure_stiffness=pressure_stiffness,
            duration=duration,
            dt=dt,
            target_overclosure=target_overclosure,
            rotation_rate_z=rotation_rate_z,
            max_iterations=hard_max_iterations,
            hard_enforcement="abaqus_standard_penalty",
            constraint_averaging="slave_face",
            hht_alpha=float(hht_alpha),
            linear_solver="sparse",
        )
    else:
        history, summary = solve_sfc_cropped_pair(
            pair,
            young=model.young,
            poisson=model.poisson,
            density=model.density,
            pressure_stiffness=pressure_stiffness,
            duration=duration,
            dt=dt,
            target_overclosure=target_overclosure,
            rotation_rate_z=rotation_rate_z,
            hht_alpha=float(hht_alpha),
            penalty_solver="modified_newton",
            material_linearization="reference_linear",
            vtk_out_dir=(sfc_vtk_dir if sfc_vtk_dir is not None else out_dir / "sfc_vtk") if write_sfc_vtk else None,
            vtk_frame_stride=max(1, int(vtk_frame_stride)),
            vtk_stem="sfc",
            vtk_include_tensors=bool(vtk_include_tensors),
        )
    if history:
        summary["final_max_displacement_norm"] = float(history[-1].get("max_displacement_norm", 0.0))
        summary["final_p95_von_mises"] = float(history[-1].get("p95_von_mises", 0.0))
        summary["final_p95_equivalent_elastic_strain"] = float(history[-1].get("p95_equivalent_elastic_strain", 0.0))
        summary["final_p95_von_mises_nodeavg"] = float(history[-1].get("p95_von_mises_nodeavg", 0.0))
        summary["final_p95_equivalent_elastic_strain_nodeavg"] = float(
            history[-1].get("p95_equivalent_elastic_strain_nodeavg", 0.0)
        )
    summary["active_patch_radius_factor"] = float(active_patch_radius_factor)
    summary["drive_mode"] = drive
    summary["sfc_match_source_step"] = bool(source_step_matched)
    summary["sfc_duration"] = float(duration)
    summary["sfc_dt"] = float(dt)
    summary["sfc_increment_count"] = int(round(float(duration) / float(dt))) if float(dt) > 0.0 else 0
    summary["source_dynamic_initial_dt"] = float(model.dynamic_initial_dt)
    summary["source_dynamic_duration"] = float(model.dynamic_duration)
    summary["source_dynamic_min_dt"] = float(model.dynamic_min_dt)
    summary["source_dynamic_max_dt"] = float(model.dynamic_max_dt)
    summary["source_contact_pressure_overclosure"] = str(model.contact_pressure_overclosure)
    summary["rotation_rate_z_rad_per_s"] = float(rotation_rate_z)
    summary["final_rotation_z_rad"] = float(rotation_rate_z) * float(duration)
    if drive == "source_inp":
        summary["rotation_rate_z_rad_per_s"] = float(model.gear1_angular_velocity_z)
        summary["final_rotation_z_rad"] = float(model.gear1_angular_velocity_z) * float(duration)
        summary["gear2_torque_z"] = float(model.gear2_torque_z)
    history_path = out_dir / "sfc_full_gear_lagrangian_sdf_history.csv"
    _write_csv(history_path, history)
    deck_path = out_dir / "abaqus_full_gear_alignment.inp"
    _write_abaqus_alignment_deck(
        deck_path,
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=pressure_stiffness,
        duration=duration,
        dt=dt,
        target_overclosure=target_overclosure,
        rotation_rate_z=rotation_rate_z,
        contact_mode=mode,
    )
    abaqus_row: Row | None = None
    error_rows: list[Row] | None = None
    figure_path: Path | None = None
    if run_abaqus:
        abaqus_metrics, abaqus_row = run_abaqus_alignment(
            deck_path,
            out_dir,
            pair=pair,
            young=model.young,
            poisson=model.poisson,
            abaqus_command=abaqus_command,
        )
        summary.update(abaqus_row)
        _write_csv(out_dir / "abaqus_runtime.csv", [abaqus_row])
        error_rows = compare_histories(history_path, abaqus_metrics, out_dir / "sfc_vs_abaqus_alignment_errors.csv")
        figure_path = out_dir / "sfc_vs_abaqus_alignment_curves.png"
        plot_alignment_curves(history_path, abaqus_metrics, error_rows, figure_path)
        summary["alignment_errors"] = str(out_dir / "sfc_vs_abaqus_alignment_errors.csv")
        summary["alignment_figure"] = str(figure_path)
    if export_abaqus_vtk:
        summary.update(
            export_abaqus_vtk_frames(
                out_dir=out_dir,
                abaqus_command=abaqus_command,
                frame_stride=max(1, int(vtk_frame_stride)),
                include_tensors=bool(vtk_include_tensors),
                young=float(model.young),
                poisson=float(model.poisson),
            )
        )
    if abaqus_vtk_manifest is not None:
        external_manifest = Path(abaqus_vtk_manifest)
        summary["abaqus_vtk_manifest"] = str(external_manifest)
        summary["abaqus_vtk_pvd"] = str(external_manifest.with_name("abaqus.pvd"))
        summary["abaqus_vtk_frame_count"] = int(len(_read_csv_rows(external_manifest)))
    if summary.get("sfc_vtk_manifest") and summary.get("abaqus_vtk_manifest"):
        animation_errors = out_dir / "sfc_vs_abaqus_vtk_metric_errors.csv"
        animation_figure = out_dir / "sfc_vs_abaqus_vtk_metric_curves.png"
        compare_animation_manifests(
            sfc_manifest=Path(str(summary["sfc_vtk_manifest"])),
            abaqus_manifest=Path(str(summary["abaqus_vtk_manifest"])),
            out_csv=animation_errors,
            out_png=animation_figure,
        )
        summary["animation_metric_errors"] = str(animation_errors)
        summary["animation_metric_figure"] = str(animation_figure)
    if summary.get("abaqus_vtk_manifest"):
        history_errors = out_dir / "sfc_vs_abaqus_history_metric_errors.csv"
        compare_sfc_history_to_abaqus_manifest(
            sfc_history=history_path,
            abaqus_manifest=Path(str(summary["abaqus_vtk_manifest"])),
            out_csv=history_errors,
        )
        summary["history_metric_errors"] = str(history_errors)
    if summary.get("sfc_vtk_manifest") or summary.get("abaqus_vtk_manifest"):
        color_ranges = write_animation_color_ranges(
            out_dir,
            sfc_manifest=Path(str(summary["sfc_vtk_manifest"])) if summary.get("sfc_vtk_manifest") else None,
            abaqus_manifest=Path(str(summary["abaqus_vtk_manifest"])) if summary.get("abaqus_vtk_manifest") else None,
        )
        summary["animation_color_ranges"] = str(color_ranges)
        summary["paraview_animation_setup"] = str(
            write_paraview_animation_setup(
                out_dir,
                color_ranges=color_ranges,
                sfc_pvd=Path(str(summary["sfc_vtk_pvd"])) if summary.get("sfc_vtk_pvd") else None,
                abaqus_pvd=Path(str(summary["abaqus_vtk_pvd"])) if summary.get("abaqus_vtk_pvd") else None,
            )
        )
    write_full_summary(out_dir / "full_gear_lagrangian_sdf_summary.md", summary, history_path, abaqus_row=abaqus_row)
    _write_csv(out_dir / "full_gear_lagrangian_sdf_summary.csv", [summary])
    return history, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--active-faces-per-body", type=int, default=16)
    parser.add_argument("--active-patch-radius-factor", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=2.5e-4)
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--overclosure", type=float, default=1.0e-5)
    parser.add_argument(
        "--rotation-rate-z",
        "--rotation-rate-z-rad-s",
        dest="rotation_rate_z",
        type=float,
        default=0.0,
        help="Prescribed G1 RP angular velocity about z in radians per second.",
    )
    parser.add_argument("--pressure-stiffness", type=float, default=5.0e9)
    parser.add_argument("--contact-mode", choices=("penalty", "hard"), default="penalty")
    parser.add_argument("--drive-mode", choices=("closure", "source_inp"), default="closure")
    parser.add_argument("--hht-alpha", type=float, default=ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA)
    parser.add_argument("--tet4-mass-kind", choices=("consistent", "calculix"), default="consistent")
    parser.add_argument("--history-frame-stride", type=int, default=2)
    parser.add_argument(
        "--use-source-timing",
        "--match-source-step",
        dest="use_source_timing",
        action="store_true",
        help="Use the *Dynamic step time and initial increment parsed from the source inp deck.",
    )
    parser.add_argument("--hard-max-iterations", type=int, default=4)
    parser.add_argument("--run-abaqus", action="store_true")
    parser.add_argument("--abaqus-command", type=str, default=None)
    parser.add_argument("--write-sfc-vtk", action="store_true")
    parser.add_argument("--sfc-vtk-dir", type=Path, default=None)
    parser.add_argument("--vtk-frame-stride", type=int, default=2)
    parser.add_argument("--vtk-scalars-only", action="store_true")
    parser.add_argument(
        "--source-stress-postprocess",
        choices=("linear_corotated", "finite_stvk_visual"),
        default="linear_corotated",
        help="Source-drive SFC stress/strain output mode; does not alter the solve.",
    )
    parser.add_argument(
        "--source-contact-averaging",
        choices=(
            "none",
            "slave_face",
            "slave_node",
            "slave_node_region",
            "slave_node_point",
            "surface_patch",
            "slave_face_constraint",
            "slave_node_constraint",
            "slave_node_region_constraint",
            "surface_patch_constraint",
            "slave_face_area_average",
            "slave_node_area_average",
            "slave_node_region_area_average",
            "surface_patch_area_average",
            "slave_face_participation",
            "slave_node_participation",
            "slave_node_region_participation",
            "surface_patch_participation",
            "slave_face_signed_participation",
            "slave_node_signed_participation",
            "slave_node_region_signed_participation",
            "surface_patch_signed_participation",
        ),
        default="none",
        help="Optional source-drive contact constraint averaging for Abaqus-style surface-to-surface penalty diagnostics.",
    )
    parser.add_argument(
        "--source-contact-kinematics",
        choices=("linearized_mpc", "finite_rp_corotated"),
        default="linearized_mpc",
        help="Kinematic map used to evaluate source-drive contact geometry.",
    )
    parser.add_argument(
        "--source-contact-normal-filter",
        choices=("none", "opposing", "opposing_search"),
        default="none",
        help="Optional source-drive master/slave normal compatibility filter.",
    )
    parser.add_argument(
        "--source-contact-direction",
        choices=("master", "secondary_average"),
        default="master",
        help="Normal direction used by source-drive surface-to-surface contact constraints.",
    )
    parser.add_argument(
        "--source-contact-projection",
        choices=("closest_feature", "secondary_plane", "secondary_line"),
        default="closest_feature",
        help="Gap projection used with secondary-average contact direction.",
    )
    parser.add_argument(
        "--source-contact-pair-order",
        choices=("gear2_slave", "gear1_slave", "symmetric_two_pass"),
        default="gear2_slave",
        help="Source-drive contact pair orientation diagnostic.",
    )
    parser.add_argument(
        "--source-contact-search-radius",
        type=float,
        default=None,
        help="Optional source-drive broad-phase radius; closest-feature projection still determines the final gap.",
    )
    parser.add_argument(
        "--source-secondary-normal-min-projection",
        type=float,
        default=0.0,
        help=(
            "Minimum opposing normal projection for secondary-normal line constraints. "
            "Values above zero reject near-tangential line intersections that do not "
            "form a well-conditioned Abaqus-style surface-to-surface constraint."
        ),
    )
    parser.add_argument(
        "--source-secondary-line-distance-limit",
        type=float,
        default=None,
        help=(
            "Optional accepted signed distance for source secondary-line projection. "
            "A negative value disables this final line-distance gate."
        ),
    )
    parser.add_argument(
        "--source-secondary-line-hard-distance-limit",
        type=float,
        default=None,
        help=(
            "Optional hard maximum distance for secondary-normal line constraints. "
            "A negative value disables the hard finite-sliding tracking tube."
        ),
    )
    parser.add_argument(
        "--source-secondary-path-tracking",
        action="store_true",
        help=(
            "Use previous accepted secondary-normal anchor faces as path-tracking "
            "hints for Abaqus-style finite-sliding surface-to-surface contact."
        ),
    )
    parser.add_argument(
        "--source-contact-active-set-stability",
        dest="source_contact_active_set_stability",
        action="store_true",
        default=True,
        help=(
            "Require one stable active contact signature before accepting a "
            "source-drive nonlinear iteration."
        ),
    )
    parser.add_argument(
        "--no-source-contact-active-set-stability",
        dest="source_contact_active_set_stability",
        action="store_false",
        help="Disable the Abaqus-style source-drive active contact status stability gate for diagnostics.",
    )
    parser.add_argument(
        "--source-contact-footprint-clipping",
        action="store_true",
        help="Clip slave triangle contact support to the projected master footprint before pressure integration.",
    )
    parser.add_argument(
        "--source-internal-kinematics",
        choices=("linearized_mpc", "corotated_rp", "finite_stvk_visual"),
        default="linearized_mpc",
        help="Internal elastic residual map for source-drive large RP rotations.",
    )
    parser.add_argument(
        "--source-rotating-inertia",
        choices=("none", "centripetal", "finite_kinematic"),
        default="none",
        help="Optional finite-RP inertia residual for source-drive dynamics.",
    )
    parser.add_argument(
        "--source-max-iterations",
        type=int,
        default=16,
        help="Maximum nonlinear iterations per fixed source-drive time increment; 16 matches Abaqus/Standard's common equilibrium iteration cap.",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=None,
        help="Optional source-drive checkpoint path for fixed-step continuation.",
    )
    parser.add_argument(
        "--resume-source-checkpoint",
        action="store_true",
        help="Resume source-drive SFC solve from --source-checkpoint.",
    )
    parser.add_argument("--source-checkpoint-stride", type=int, default=10)
    parser.add_argument("--export-abaqus-vtk", action="store_true")
    parser.add_argument(
        "--abaqus-vtk-manifest",
        type=Path,
        default=None,
        help="Existing Abaqus VTK manifest to compare against the generated SFC VTK field curves.",
    )
    args = parser.parse_args(argv)
    history, summary = run_full_gear(
        source=args.source,
        out_dir=args.out_dir,
        active_faces_per_body=int(args.active_faces_per_body),
        active_patch_radius_factor=float(args.active_patch_radius_factor),
        duration=float(args.duration),
        dt=float(args.dt),
        target_overclosure=float(args.overclosure),
        rotation_rate_z=float(args.rotation_rate_z),
        pressure_stiffness=float(args.pressure_stiffness),
        hard_max_iterations=int(args.hard_max_iterations),
        contact_mode=str(args.contact_mode),
        run_abaqus=bool(args.run_abaqus),
        drive_mode=str(args.drive_mode),
        hht_alpha=float(args.hht_alpha),
        tet4_mass_kind=str(args.tet4_mass_kind),
        history_frame_stride=int(args.history_frame_stride),
        use_source_timing=bool(args.use_source_timing),
        abaqus_command=args.abaqus_command,
        write_sfc_vtk=bool(args.write_sfc_vtk),
        sfc_vtk_dir=args.sfc_vtk_dir,
        vtk_frame_stride=int(args.vtk_frame_stride),
        vtk_include_tensors=not bool(args.vtk_scalars_only),
        source_stress_postprocess=str(args.source_stress_postprocess),
        source_contact_averaging=str(args.source_contact_averaging),
        source_contact_kinematics=str(args.source_contact_kinematics),
        source_contact_normal_filter=str(args.source_contact_normal_filter),
        source_contact_direction=str(args.source_contact_direction),
        source_contact_projection=str(args.source_contact_projection),
        source_contact_pair_order=str(args.source_contact_pair_order),
        source_contact_search_radius=args.source_contact_search_radius,
        source_secondary_normal_min_projection=float(args.source_secondary_normal_min_projection),
        source_secondary_line_distance_limit=args.source_secondary_line_distance_limit,
        source_secondary_line_hard_distance_limit=args.source_secondary_line_hard_distance_limit,
        source_secondary_path_tracking=bool(args.source_secondary_path_tracking),
        source_contact_active_set_stability=bool(args.source_contact_active_set_stability),
        source_contact_footprint_clipping=bool(args.source_contact_footprint_clipping),
        source_internal_kinematics=str(args.source_internal_kinematics),
        source_rotating_inertia=str(args.source_rotating_inertia),
        source_max_iterations=int(args.source_max_iterations),
        source_checkpoint_path=args.source_checkpoint,
        resume_source_checkpoint=bool(args.resume_source_checkpoint),
        source_checkpoint_stride=int(args.source_checkpoint_stride),
        export_abaqus_vtk=bool(args.export_abaqus_vtk),
        abaqus_vtk_manifest=args.abaqus_vtk_manifest,
    )
    print((args.out_dir / "full_gear_lagrangian_sdf_summary.md").read_text(encoding="utf-8"))
    return 0 if history and summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
