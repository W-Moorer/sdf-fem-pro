from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_source_gear_vtk_regional_alignment import (
    _stress_error_context_rows,
    _unique_active_overlap_rows,
    _worst_active_overlap_rows,
    compare_vtk_manifests,
    compare_vtk_pair,
    read_vtk_point_scalars,
)


def _write_point_scalar_vtk(path: Path, *, pressure: tuple[float, float], active: tuple[float, float]) -> None:
    lines = [
        "# vtk DataFile Version 3.0",
        "test",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        "POINTS 2 float",
        "0 0 0",
        "1 0 0",
        "CELLS 0 0",
        "CELL_TYPES 0",
        "POINT_DATA 2",
        "VECTORS U float",
        "0 0 0",
        "0 0 0",
        "SCALARS displacement_magnitude float 1",
        "LOOKUP_TABLE default",
        "1.0",
        "2.0",
        "SCALARS von_mises_nodeavg float 1",
        "LOOKUP_TABLE default",
        "10.0",
        "20.0",
        "SCALARS strain_norm_nodeavg float 1",
        "LOOKUP_TABLE default",
        "0.1",
        "0.2",
        "SCALARS logarithmic_strain_norm_nodeavg float 1",
        "LOOKUP_TABLE default",
        "0.1",
        "0.2",
        "SCALARS equivalent_elastic_strain_nodeavg float 1",
        "LOOKUP_TABLE default",
        "0.01",
        "0.02",
        "SCALARS contact_pressure_nodeavg float 1",
        "LOOKUP_TABLE default",
        f"{pressure[0]}",
        f"{pressure[1]}",
        "SCALARS contact_penetration_nodeavg float 1",
        "LOOKUP_TABLE default",
        "0.0",
        "0.0",
        "SCALARS contact_active_node float 1",
        "LOOKUP_TABLE default",
        f"{active[0]}",
        f"{active[1]}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_read_vtk_point_scalars_reads_selected_point_data(tmp_path: Path) -> None:
    vtk = tmp_path / "frame.vtk"
    _write_point_scalar_vtk(vtk, pressure=(0.0, 5.0), active=(0.0, 1.0))

    frame = read_vtk_point_scalars(vtk)

    assert frame.point_count == 2
    assert frame.scalars["contact_pressure_nodeavg"].tolist() == [0.0, 5.0]
    assert frame.scalars["contact_active_node"].tolist() == [0.0, 1.0]


def test_compare_vtk_pair_reports_active_region_errors(tmp_path: Path) -> None:
    sfc = tmp_path / "sfc.vtk"
    abaqus = tmp_path / "abaqus.vtk"
    _write_point_scalar_vtk(sfc, pressure=(0.0, 0.0), active=(0.0, 0.0))
    _write_point_scalar_vtk(abaqus, pressure=(0.0, 5.0), active=(0.0, 1.0))

    rows = compare_vtk_pair(sfc_vtk=sfc, abaqus_vtk=abaqus, sfc_time=0.2, abaqus_time=0.1)
    by_region_metric = {(row["region"], row["metric"]): row for row in rows}

    active_stress = by_region_metric[("abaqus_active", "von_mises_nodeavg")]
    assert active_stress["sample_count"] == 1
    assert active_stress["abaqus_active_node_count"] == 1
    assert active_stress["sfc_active_node_count"] == 0
    assert active_stress["time_difference"] == pytest.approx(0.1)
    pressure = by_region_metric[("full", "contact_pressure_nodeavg")]
    assert pressure["field_rmse_rel"] == pytest.approx(1.0)
    assert pressure["active_jaccard"] == pytest.approx(0.0)


def test_compare_vtk_manifests_pairs_frames_by_time(tmp_path: Path) -> None:
    sfc_dir = tmp_path / "sfc"
    abaqus_dir = tmp_path / "abaqus"
    sfc_dir.mkdir()
    abaqus_dir.mkdir()
    for index, time_value in enumerate((0.0, 2.0e-5)):
        _write_point_scalar_vtk(sfc_dir / f"sfc_{index:04d}.vtk", pressure=(0.0, float(index)), active=(0.0, float(index)))
        _write_point_scalar_vtk(
            abaqus_dir / f"abaqus_{index:04d}.vtk",
            pressure=(0.0, 2.0 * float(index)),
            active=(0.0, float(index)),
        )
    (sfc_dir / "sfc_manifest.csv").write_text(
        "frame,time,vtk_file\n0,0.0,sfc_0000.vtk\n1,2e-05,sfc_0001.vtk\n",
        encoding="utf-8",
    )
    (abaqus_dir / "abaqus_manifest.csv").write_text(
        "frame,time,vtk_file\n0,0.0,abaqus_0000.vtk\n1,1.99999995e-05,abaqus_0001.vtk\n",
        encoding="utf-8",
    )

    rows = compare_vtk_manifests(
        sfc_manifest=sfc_dir / "sfc_manifest.csv",
        abaqus_manifest=abaqus_dir / "abaqus_manifest.csv",
        time_tolerance=1.0e-9,
    )

    assert len(rows) == 2 * 4 * 7
    assert {row["pair_index"] for row in rows} == {0, 1}
    assert max(abs(float(row["time_difference"])) for row in rows) < 1.0e-9


def test_active_overlap_summary_helpers_ignore_no_contact_duplicates() -> None:
    rows = [
        {
            "pair_index": 0,
            "sfc_time": 0.0,
            "region": "full",
            "metric": "von_mises_nodeavg",
            "p95_rel_error": 0.0,
            "active_union_count": 0,
            "active_jaccard": 1.0,
            "active_recall": 1.0,
            "active_precision": 1.0,
        },
        {
            "pair_index": 1,
            "sfc_time": 1.0e-4,
            "region": "full",
            "metric": "von_mises_nodeavg",
            "p95_rel_error": 0.2,
            "active_union_count": 10,
            "active_jaccard": 0.4,
            "active_recall": 0.5,
            "active_precision": 0.6,
            "sfc_active_node_count": 6,
            "abaqus_active_node_count": 8,
            "active_intersection_count": 4,
            "active_sfc_only_count": 2,
            "active_abaqus_only_count": 4,
        },
        {
            "pair_index": 1,
            "sfc_time": 1.0e-4,
            "region": "abaqus_active",
            "metric": "contact_pressure_nodeavg",
            "p95_rel_error": 0.1,
            "active_union_count": 10,
            "active_jaccard": 0.4,
            "active_recall": 0.5,
            "active_precision": 0.6,
            "sfc_active_node_count": 6,
            "abaqus_active_node_count": 8,
            "active_intersection_count": 4,
            "active_sfc_only_count": 2,
            "active_abaqus_only_count": 4,
        },
    ]

    unique = _unique_active_overlap_rows(rows)
    worst = _worst_active_overlap_rows(rows)
    context = _stress_error_context_rows(rows)

    assert len(unique) == 1
    assert unique[0]["sfc_time"] == pytest.approx(1.0e-4)
    assert worst == unique
    assert len(context) == 1
    assert context[0]["metric"] == "von_mises_nodeavg"
