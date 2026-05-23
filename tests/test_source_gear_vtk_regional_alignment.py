from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_source_gear_vtk_regional_alignment import compare_vtk_pair, read_vtk_point_scalars


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
