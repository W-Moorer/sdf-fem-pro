from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_flexible_body_rigid_plane import build_input_text
from validation.run_abaqus_sphere_drop_full_validation import (
    _first_contact_time,
    run_sfc_lagrangian_sdf_full_history,
    run_validation,
)
from validation.run_abaqus_sphere_drop_short_validation import parse_sphere_drop_inp


def test_full_sfc_history_crosses_contact_with_lagrangian_oracle(tmp_path: Path) -> None:
    inp = tmp_path / "sphere_drop.inp"
    inp.write_text(build_input_text("sphere_drop"), encoding="ascii")
    model = parse_sphere_drop_inp(inp)

    rows = run_sfc_lagrangian_sdf_full_history(
        model,
        duration=0.07,
        dt=0.002,
        contact_stiffness=1.0e8,
        contact_damping=1.0e3,
        mass_damping=0.1,
        damping_start_time=0.02,
        integrator="hht",
        hht_alpha=-0.1,
        contact_integration="surface",
        quadrature_order=3,
        output_stride=1,
        max_newton_iterations=4,
    )

    assert rows
    assert {row["contact_path"] for row in rows} == {"MaterialSDF+LagrangianSDFContactOracle"}
    assert any(int(row["active_contact_count"]) > 0 for row in rows)
    assert _first_contact_time(rows) is not None
    assert all(float(row["sfc_solve_wall_seconds"]) > 0.0 for row in rows)
    assert {float(row["contact_damping"]) for row in rows} == {1000.0}
    assert {float(row["mass_damping"]) for row in rows} == {0.1}
    assert {float(row["damping_start_time"]) for row in rows} == {0.02}
    assert {row["time_integrator"] for row in rows} == {"hht"}
    assert {float(row["hht_alpha"]) for row in rows} == {-0.1}
    assert {row["contact_integration"] for row in rows} == {"surface"}
    assert all("active_contact_area" in row for row in rows)


def test_full_validation_runner_writes_outputs_with_existing_reference(tmp_path: Path) -> None:
    if not (
        ROOT
        / "commercial_software_comparison"
        / "abaqus_flexible_body_rigid_plane"
        / "sphere_drop"
        / "vtk"
        / "frame_manifest.csv"
    ).is_file():
        pytest.skip("external Abaqus VTK reference is not available")

    outputs = run_validation(
        tmp_path,
        duration=0.01,
        dt=0.002,
        contact_stiffness=1.0e8,
        contact_damping=1.0e3,
        mass_damping=0.1,
        damping_start_time=0.02,
        output_stride=1,
        max_newton_iterations=4,
    )

    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    with outputs["metrics"].open(newline="", encoding="utf-8") as handle:
        metrics = {row["metric"]: row for row in csv.DictReader(handle)}
    assert metrics["core_abaqus_dependency"]["value"] == "false"
    assert metrics["sfc_contact_path"]["value"] == "MaterialSDF+LagrangianSDFContactOracle"
    assert metrics["contact_damping"]["value"] == "1000.0"
    assert metrics["mass_damping"]["value"] == "0.1"
    assert metrics["damping_start_time"]["value"] == "0.02"
    assert "newton_failed_steps" in metrics
