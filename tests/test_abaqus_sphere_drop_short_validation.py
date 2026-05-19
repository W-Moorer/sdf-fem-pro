from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_flexible_body_rigid_plane import build_input_text
from validation.run_abaqus_sphere_drop_short_validation import (
    DEFAULT_INP,
    _comparison_rows,
    _resolved_contact_stiffness,
    parse_sphere_drop_inp,
    read_legacy_vtk,
    run_sfc_lagrangian_sdf_short_history,
    run_validation,
)


def test_parse_sphere_drop_inp_extracts_mesh_material_and_gravity(tmp_path: Path) -> None:
    inp = tmp_path / "sphere_drop.inp"
    inp.write_text(build_input_text("sphere_drop"), encoding="ascii")

    model = parse_sphere_drop_inp(inp)

    assert model.nodes.shape[1] == 3
    assert model.elements.shape[1] == 4
    assert model.plane_faces.shape == (2, 3)
    assert model.density == pytest.approx(1200.0)
    assert model.young == pytest.approx(5.0e7)
    assert model.poisson == pytest.approx(0.30)
    assert model.gravity == pytest.approx(9.81)
    assert model.abaqus_duration == pytest.approx(3.0)
    assert model.output_interval == pytest.approx(0.001)
    assert model.contact_penalty_normal_stiffness is None
    assert np.min(model.nodes[:, 2]) - model.plane_z == pytest.approx(0.02)


def test_parse_implicit_dynamic_deck_extracts_total_time_and_penalty(tmp_path: Path) -> None:
    from validation.run_abaqus_flexible_body_rigid_plane_implicit import PENALTY_NORMAL_STIFFNESS, build_implicit_input_text

    inp = tmp_path / "sphere_drop_implicit.inp"
    inp.write_text(build_implicit_input_text("sphere_drop"), encoding="ascii")

    model = parse_sphere_drop_inp(inp)

    assert model.abaqus_duration == pytest.approx(3.0)
    assert model.output_interval == pytest.approx(0.001)
    assert model.contact_penalty_normal_stiffness == pytest.approx(PENALTY_NORMAL_STIFFNESS)


def test_resolved_contact_stiffness_prefers_cli_then_inp_then_fallback(tmp_path: Path) -> None:
    from validation.run_abaqus_flexible_body_rigid_plane_implicit import PENALTY_NORMAL_STIFFNESS, build_implicit_input_text

    inp = tmp_path / "sphere_drop_implicit.inp"
    inp.write_text(build_implicit_input_text("sphere_drop"), encoding="ascii")
    implicit_model = parse_sphere_drop_inp(inp)
    explicit_model = parse_sphere_drop_inp(DEFAULT_INP)

    assert _resolved_contact_stiffness(implicit_model, None, fallback=1.0e9) == pytest.approx(PENALTY_NORMAL_STIFFNESS)
    assert _resolved_contact_stiffness(implicit_model, 2.5e9, fallback=1.0e9) == pytest.approx(2.5e9)
    assert _resolved_contact_stiffness(explicit_model, None, fallback=1.0e9) == pytest.approx(1.0e9)


def test_sfc_short_history_uses_lagrangian_oracle_and_matches_freefall(tmp_path: Path) -> None:
    inp = tmp_path / "sphere_drop.inp"
    inp.write_text(build_input_text("sphere_drop"), encoding="ascii")
    model = parse_sphere_drop_inp(inp)

    rows = run_sfc_lagrangian_sdf_short_history(model, duration=0.01, dt=0.001, contact_stiffness=1.0e10)

    assert rows
    assert {row["contact_path"] for row in rows} == {"MaterialSDF+LagrangianSDFContactOracle"}
    assert all(int(row["active_contact_count"]) == 0 for row in rows)
    z0 = float(rows[0]["z_cm"])
    z_end = float(rows[-1]["z_cm"])
    expected = z0 - 0.5 * model.gravity * (0.01**2)
    assert z_end == pytest.approx(expected, abs=2.0e-8)


def test_vtk_subset_parser_reads_points_vectors_and_cell_scalars(tmp_path: Path) -> None:
    vtk = tmp_path / "frame_0000.vtk"
    vtk.write_text(
        "\n".join(
            [
                "# vtk DataFile Version 3.0",
                "demo",
                "ASCII",
                "DATASET UNSTRUCTURED_GRID",
                "POINTS 4 float",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "0 0 1",
                "CELLS 1 5",
                "4 0 1 2 3",
                "CELL_TYPES 1",
                "10",
                "POINT_DATA 4",
                "VECTORS displacement float",
                "0 0 0",
                "0 0 0",
                "0 0 0",
                "0 0 0",
                "VECTORS velocity float",
                "0 0 0",
                "0 0 0",
                "0 0 0",
                "0 0 0",
                "CELL_DATA 1",
                "SCALARS object_id int 1",
                "LOOKUP_TABLE default",
                "2",
                "SCALARS von_mises float 1",
                "LOOKUP_TABLE default",
                "3.5",
                "SCALARS logarithmic_strain_norm float 1",
                "LOOKUP_TABLE default",
                "0.25",
                "",
            ]
        ),
        encoding="ascii",
    )

    frame = read_legacy_vtk(vtk)

    assert frame.points.shape == (4, 3)
    assert frame.cell_types.tolist() == [10]
    assert "displacement" in frame.point_vectors
    assert frame.cell_scalars["object_id"].tolist() == [2.0]
    assert frame.cell_scalars["von_mises"].tolist() == [3.5]


def test_short_validation_runner_writes_outputs_with_existing_reference(tmp_path: Path) -> None:
    if not (
        ROOT
        / "commercial_software_comparison"
        / "abaqus_flexible_body_rigid_plane"
        / "sphere_drop"
        / "vtk"
        / "frame_manifest.csv"
    ).is_file():
        pytest.skip("external Abaqus VTK reference is not available")

    outputs = run_validation(tmp_path, duration=0.003, dt=0.001)

    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    with outputs["metrics"].open(newline="", encoding="utf-8") as handle:
        metrics = {row["metric"]: row for row in csv.DictReader(handle)}
    assert metrics["core_abaqus_dependency"]["value"] == "false"
    assert metrics["sfc_contact_path"]["value"] == "MaterialSDF+LagrangianSDFContactOracle"


def test_comparison_rows_report_gap_and_trajectory_errors() -> None:
    abaqus = [
        {"time": 0.0, "z_cm": 1.0, "min_gap": 0.2, "normal_force_z": 0.0, "contact_energy": 0.0, "active_contact_area": 0.0, "max_von_mises": 0.0, "p95_von_mises": 0.0, "volume_mean_von_mises": 0.0, "max_strain_norm": 0.0, "p95_strain_norm": 0.0, "volume_mean_strain_norm": 0.0},
        {"time": 1.0, "z_cm": 0.8, "min_gap": 0.0, "normal_force_z": 4.0, "contact_energy": 2.0, "active_contact_area": 0.2, "max_von_mises": 2.0, "p95_von_mises": 1.8, "volume_mean_von_mises": 1.0, "max_strain_norm": 0.1, "p95_strain_norm": 0.09, "volume_mean_strain_norm": 0.05},
    ]
    sfc = [
        {"time": 0.5, "z_cm": 0.9, "min_gap": 0.1, "normal_force_z": 2.5, "contact_energy": 0.75, "active_contact_area": 0.08, "max_von_mises": 1.0, "p95_von_mises": 0.95, "volume_mean_von_mises": 0.45, "max_strain_norm": 0.05, "p95_strain_norm": 0.047, "volume_mean_strain_norm": 0.02},
    ]

    rows = _comparison_rows(abaqus, sfc)

    assert rows[0]["z_cm_abs_error"] == pytest.approx(0.0)
    assert rows[0]["min_gap_abs_error"] == pytest.approx(0.0)
    assert rows[0]["normal_force_z_abs_error"] == pytest.approx(0.5)
    assert rows[0]["contact_energy_abs_error"] == pytest.approx(0.25)
    assert rows[0]["active_contact_area_abs_error"] == pytest.approx(0.02)
    assert rows[0]["p95_von_mises_abs_error"] == pytest.approx(0.05)
    assert rows[0]["volume_mean_von_mises_abs_error"] == pytest.approx(0.05)
    assert rows[0]["p95_strain_norm_abs_error"] == pytest.approx(0.002)
    assert rows[0]["volume_mean_strain_norm_abs_error"] == pytest.approx(0.005)
