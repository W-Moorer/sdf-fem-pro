from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCKED_CASE = ROOT / "paper" / "numerical_experiments" / "block_drop_dynamic_sdf_calculix_1s"
TET4_PATCH_CASE = ROOT / "paper" / "numerical_experiments" / "tet4_analytic_patch_stress_strain"
SCIKIT_FEM_CASE = ROOT / "paper" / "numerical_experiments" / "scikit_fem_cantilever_external"
CONTACTENERGY_CASE = ROOT / "paper" / "numerical_experiments" / "calculix_contactenergy_c3d8_replay"
C3D8_TRAJECTORY_CASE = ROOT / "paper" / "numerical_experiments" / "c3d8_contact_trajectory_validation"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _assert_file(path: Path) -> None:
    assert path.exists(), path
    assert path.stat().st_size > 0, path


def _assert_threshold(gate: dict[str, str], actual: object) -> None:
    operator = gate["operator"]
    expected_actual = gate["actual"]
    threshold = gate["threshold"]
    assert gate["status"] == "pass"

    if operator == "eq":
        actual_text = str(actual)
        assert actual_text == expected_actual
        assert actual_text == threshold
    elif operator == "le":
        actual_value = float(actual)
        expected_value = float(expected_actual)
        assert abs(actual_value - expected_value) <= max(1.0e-15, abs(expected_value) * 1.0e-12)
        assert actual_value <= float(threshold)
    elif operator == "ge":
        actual_value = float(actual)
        expected_value = float(expected_actual)
        assert abs(actual_value - expected_value) <= max(1.0e-15, abs(expected_value) * 1.0e-12)
        assert actual_value >= float(threshold)
    else:  # pragma: no cover - catches malformed future threshold rows
        raise AssertionError(f"unsupported operator {operator!r} for {gate['metric']}")


def test_locked_dynamic_sdf_block_drop_artifacts_exist() -> None:
    required = [
        LOCKED_CASE / "README.md",
        LOCKED_CASE / "MANIFEST.md",
        LOCKED_CASE / "locked_thresholds.csv",
        LOCKED_CASE / "runner_summary.md",
        LOCKED_CASE / "data" / "trajectory_comparison.csv",
        LOCKED_CASE / "data" / "alignment_diagnostics.csv",
        LOCKED_CASE / "data" / "claims.csv",
        LOCKED_CASE / "data" / "plots.csv",
        LOCKED_CASE / "data" / "external_solver_commands.csv",
        LOCKED_CASE / "figures" / "calculix_drop_z_cm.png",
        LOCKED_CASE / "figures" / "calculix_drop_z_cm.pdf",
        LOCKED_CASE / "figures" / "calculix_drop_min_gap.png",
        LOCKED_CASE / "figures" / "calculix_drop_force_proxy.png",
        LOCKED_CASE / "figures" / "calculix_drop_contact_energy.png",
        LOCKED_CASE / "figures" / "calculix_drop_total_energy.png",
    ]

    for path in required:
        _assert_file(path)


def test_locked_dynamic_sdf_block_drop_metrics_satisfy_thresholds() -> None:
    comparison = _rows(LOCKED_CASE / "data" / "trajectory_comparison.csv")
    thresholds = _rows(LOCKED_CASE / "locked_thresholds.csv")

    assert len(comparison) == 1
    metric_row = comparison[0]
    assert metric_row["case"] == "block_drop"
    assert metric_row["contact_mode"] == "persistent_dynamic_sdf_calculix_f2f"

    for gate in thresholds:
        metric = gate["metric"]
        actual = metric_row[metric]
        expected_actual = gate["actual"]
        operator = gate["operator"]
        threshold = gate["threshold"]
        assert gate["status"] == "pass"

        if operator == "eq":
            assert actual == expected_actual
            assert actual == threshold
        elif operator == "le":
            assert abs(float(actual) - float(expected_actual)) < 1.0e-15
            assert float(actual) <= float(threshold)
        else:  # pragma: no cover - catches malformed future threshold rows
            raise AssertionError(f"unsupported operator {operator!r} for {metric}")


def test_locked_dynamic_sdf_block_drop_claim_boundary_is_explicit() -> None:
    readme = (LOCKED_CASE / "README.md").read_text(encoding="utf-8")
    claims = {row["claim"]: row for row in _rows(LOCKED_CASE / "data" / "claims.csv")}

    assert "persistent_dynamic_sdf_calculix_f2f" in readme
    assert "does not claim arbitrary contact equivalence" in readme
    assert claims["block_plane_geometric_contact_scoped_acceptance"]["supported"] == "true"
    assert claims["block_plane_geometric_contact_calculix_comparison_available"]["supported"] == "true"


def test_locked_tet4_analytic_patch_artifacts_and_thresholds() -> None:
    required = [
        TET4_PATCH_CASE / "README.md",
        TET4_PATCH_CASE / "MANIFEST.md",
        TET4_PATCH_CASE / "locked_thresholds.csv",
        TET4_PATCH_CASE / "runner_summary.md",
        TET4_PATCH_CASE / "data" / "stress_strain_patch.csv",
        TET4_PATCH_CASE / "data" / "stress_strain_patch_cloud.csv",
        TET4_PATCH_CASE / "data" / "phase7_claims.csv",
        TET4_PATCH_CASE / "figures" / "stress_strain_error.png",
        TET4_PATCH_CASE / "figures" / "stress_strain_error.pdf",
        TET4_PATCH_CASE / "figures" / "stress_strain_patch_3d.png",
        TET4_PATCH_CASE / "figures" / "stress_strain_patch_3d.pdf",
    ]
    for path in required:
        _assert_file(path)

    rows = _rows(TET4_PATCH_CASE / "data" / "stress_strain_patch.csv")
    assert len(rows) == 4
    assert {row["case"] for row in rows} == {"analytic_uniaxial_stress_strain_patch"}
    assert {row["reference"] for row in rows} == {"analytic_affine_small_strain"}
    assert {row["status"] for row in rows} == {"ok"}

    cloud_rows = _rows(TET4_PATCH_CASE / "data" / "stress_strain_patch_cloud.csv")
    assert len(cloud_rows) >= 4
    assert {row["case"] for row in cloud_rows} == {"analytic_uniaxial_stress_strain_patch_cloud"}
    assert {row["status"] for row in cloud_rows} == {"ok"}
    assert max(float(row["von_mises_abs_error"]) for row in cloud_rows) < 1.0e-8
    assert max(float(row["engineering_strain_norm_abs_error"]) for row in cloud_rows) < 1.0e-14

    metrics: dict[str, object] = {
        "resolution_count": len(rows),
        "max_strain_l2_error": max(float(row["max_strain_l2_error"]) for row in rows),
        "rms_strain_l2_error": max(float(row["rms_strain_l2_error"]) for row in rows),
        "max_stress_l2_error": max(float(row["max_stress_l2_error"]) for row in rows),
        "rms_stress_l2_error": max(float(row["rms_stress_l2_error"]) for row in rows),
    }
    for gate in _rows(TET4_PATCH_CASE / "locked_thresholds.csv"):
        _assert_threshold(gate, metrics[gate["metric"]])

    claims = {row["claim_id"]: row for row in _rows(TET4_PATCH_CASE / "data" / "phase7_claims.csv")}
    assert claims["stress_strain_physical_correctness"]["claim_status"] == "supported"


def test_locked_scikit_fem_cantilever_artifacts_and_latest_stress_cloud_scheme() -> None:
    required = [
        SCIKIT_FEM_CASE / "README.md",
        SCIKIT_FEM_CASE / "MANIFEST.md",
        SCIKIT_FEM_CASE / "locked_thresholds.csv",
        SCIKIT_FEM_CASE / "runner_summary.md",
        SCIKIT_FEM_CASE / "data" / "external_fem_comparison.csv",
        SCIKIT_FEM_CASE / "data" / "external_fem_stress_cloud.csv",
        SCIKIT_FEM_CASE / "data" / "external_fem_plots.csv",
        SCIKIT_FEM_CASE / "figures" / "external_fem_error_trends.png",
        SCIKIT_FEM_CASE / "figures" / "external_fem_error_trends.pdf",
        SCIKIT_FEM_CASE / "figures" / "external_fem_stress_3d.png",
        SCIKIT_FEM_CASE / "figures" / "external_fem_stress_3d.pdf",
    ]
    for path in required:
        _assert_file(path)

    readme = (SCIKIT_FEM_CASE / "README.md").read_text(encoding="utf-8")
    assert "element von Mises stress is averaged to nodes" in readme
    assert "triangles are subdivided" in readme
    assert "not the older element-constant/blocky stress-coloring scheme" in readme

    rows = _rows(SCIKIT_FEM_CASE / "data" / "external_fem_comparison.csv")
    assert len(rows) == 4
    assert {row["case"] for row in rows} == {"cantilever_external_open_source_fem"}
    assert {row["external_solver"] for row in rows} == {"scikit-fem"}
    assert {row["status"] for row in rows} == {"ok"}

    metrics: dict[str, object] = {
        "resolution_count": len(rows),
        "max_displacement_l2_rel_error": max(float(row["displacement_l2_rel_error"]) for row in rows),
        "max_stress_l2_rel_error": max(float(row["stress_l2_rel_error"]) for row in rows),
        "max_von_mises_l2_rel_error": max(float(row["von_mises_l2_rel_error"]) for row in rows),
        "max_stiffness_fro_rel_error": max(float(row["stiffness_fro_rel_error"]) for row in rows),
    }
    for gate in _rows(SCIKIT_FEM_CASE / "locked_thresholds.csv"):
        _assert_threshold(gate, metrics[gate["metric"]])


def test_locked_calculix_contactenergy_c3d8_replay_artifacts_and_thresholds() -> None:
    required = [
        CONTACTENERGY_CASE / "README.md",
        CONTACTENERGY_CASE / "MANIFEST.md",
        CONTACTENERGY_CASE / "locked_thresholds.csv",
        CONTACTENERGY_CASE / "runner_summary.md",
        CONTACTENERGY_CASE / "data" / "calculix_contactenergy_replay.csv",
        CONTACTENERGY_CASE / "data" / "calculix_contactenergy_claims.csv",
        CONTACTENERGY_CASE / "data" / "calculix_contactenergy_commands.csv",
        CONTACTENERGY_CASE / "data" / "calculix_contactenergy_raw_cels.csv",
        CONTACTENERGY_CASE / "data" / "calculix_contactenergy_plots.csv",
        CONTACTENERGY_CASE / "data" / "calculix_contactenergy_stress_strain_cloud.csv",
        CONTACTENERGY_CASE / "data" / "calculix_contactenergy_error_metrics.csv",
        CONTACTENERGY_CASE / "data" / "contactenergy.inp",
        CONTACTENERGY_CASE / "data" / "contactenergy.dat",
        CONTACTENERGY_CASE / "figures" / "calculix_contactenergy_stress_strain_3d.png",
        CONTACTENERGY_CASE / "figures" / "calculix_contactenergy_stress_strain_3d.pdf",
        CONTACTENERGY_CASE / "figures" / "calculix_contactenergy_contact_pressure_3d.png",
        CONTACTENERGY_CASE / "figures" / "calculix_contactenergy_contact_pressure_3d.pdf",
        CONTACTENERGY_CASE / "figures" / "calculix_contactenergy_error_metrics.png",
        CONTACTENERGY_CASE / "figures" / "calculix_contactenergy_error_metrics.pdf",
        CONTACTENERGY_CASE / "figures" / "calculix_contactenergy_sfc_c3d8_error_3d.png",
        CONTACTENERGY_CASE / "figures" / "calculix_contactenergy_sfc_c3d8_error_3d.pdf",
        CONTACTENERGY_CASE / "logs" / "calculix_stdout.log",
    ]
    for path in required:
        _assert_file(path)
    assert (CONTACTENERGY_CASE / "logs" / "calculix_stderr.log").exists()

    readme = (CONTACTENERGY_CASE / "README.md").read_text(encoding="utf-8")
    assert "C3D8 static" in readme
    assert "contact-law/energy replay" in readme
    assert "not a TET4 trajectory-equivalence claim" in readme
    assert "triangulated" in readme
    assert "C3D8 boundary faces" in readme

    row = _rows(CONTACTENERGY_CASE / "data" / "calculix_contactenergy_replay.csv")[0]
    metrics: dict[str, object] = {
        "element_type": row["element_type"],
        "dynamic_sdf_backend": row["dynamic_sdf_backend"],
        "status": row["status"],
        "contact_force_rel_error": float(row["contact_force_rel_error"]),
        "contact_energy_rel_error": float(row["contact_energy_rel_error"]),
        "sfc_dynamic_sdf_master_triangle_count": int(row["sfc_dynamic_sdf_master_triangle_count"]),
        "sfc_dynamic_sdf_slave_quadrature_count": int(row["sfc_dynamic_sdf_slave_quadrature_count"]),
        "sfc_c3d8_displacement_l2_rel_error": float(row["sfc_c3d8_displacement_l2_rel_error"]),
        "sfc_c3d8_stress_l2_rel_error": float(row["sfc_c3d8_stress_l2_rel_error"]),
        "sfc_c3d8_von_mises_l2_rel_error": float(row["sfc_c3d8_von_mises_l2_rel_error"]),
    }
    for gate in _rows(CONTACTENERGY_CASE / "locked_thresholds.csv"):
        _assert_threshold(gate, metrics[gate["metric"]])

    claims = {row["claim"]: row for row in _rows(CONTACTENERGY_CASE / "data" / "calculix_contactenergy_claims.csv")}
    assert claims["dynamic_sdf_replays_c3d8_contact_energy"]["supported"] == "true"
    assert claims["dynamic_sdf_is_not_tet4_bound"]["supported"] == "true"

    stress_cloud = _rows(CONTACTENERGY_CASE / "data" / "calculix_contactenergy_stress_strain_cloud.csv")
    assert len(stress_cloud) == 2
    assert max(float(row["von_mises"]) for row in stress_cloud) > 0.0
    assert max(float(row["engineering_strain_norm"]) for row in stress_cloud) > 0.0
    assert max(float(row["sfc_von_mises"]) for row in stress_cloud) > 0.0

    error_rows = _rows(CONTACTENERGY_CASE / "data" / "calculix_contactenergy_error_metrics.csv")
    assert {row["quantity"] for row in error_rows} == {"normal_force_z", "contact_energy", "c3d8_displacement", "c3d8_von_mises"}
    by_quantity = {row["quantity"]: row for row in error_rows}
    assert by_quantity["normal_force_z"]["status"] == "ok"
    assert by_quantity["contact_energy"]["status"] == "ok"


def test_locked_c3d8_contact_trajectory_replay_artifacts_and_thresholds() -> None:
    required = [
        C3D8_TRAJECTORY_CASE / "README.md",
        C3D8_TRAJECTORY_CASE / "MANIFEST.md",
        C3D8_TRAJECTORY_CASE / "locked_thresholds.csv",
        C3D8_TRAJECTORY_CASE / "runner_summary.md",
        C3D8_TRAJECTORY_CASE / "data" / "c3d8_contact_trajectory.csv",
        C3D8_TRAJECTORY_CASE / "data" / "c3d8_contact_trajectory_summary.csv",
        C3D8_TRAJECTORY_CASE / "data" / "c3d8_contact_trajectory_stress_cloud.csv",
        C3D8_TRAJECTORY_CASE / "data" / "c3d8_contact_trajectory_claims.csv",
        C3D8_TRAJECTORY_CASE / "data" / "c3d8_contact_trajectory_commands.csv",
        C3D8_TRAJECTORY_CASE / "figures" / "c3d8_contact_trajectory_z_cm.png",
        C3D8_TRAJECTORY_CASE / "figures" / "c3d8_contact_trajectory_gap.png",
        C3D8_TRAJECTORY_CASE / "figures" / "c3d8_contact_trajectory_force_energy.png",
        C3D8_TRAJECTORY_CASE / "figures" / "c3d8_contact_trajectory_stress_cloud.png",
        C3D8_TRAJECTORY_CASE / "calculix_runs" / "block_plane_c3d8_r1" / "block_plane_c3d8_r1.inp",
        C3D8_TRAJECTORY_CASE / "calculix_runs" / "block_plane_c3d8_r1" / "block_plane_c3d8_r1.dat",
        C3D8_TRAJECTORY_CASE / "calculix_runs" / "block_block_c3d8_r1" / "block_block_c3d8_r1.inp",
        C3D8_TRAJECTORY_CASE / "calculix_runs" / "block_block_c3d8_r1" / "block_block_c3d8_r1.dat",
    ]
    for path in required:
        _assert_file(path)

    readme = (C3D8_TRAJECTORY_CASE / "README.md").read_text(encoding="utf-8")
    assert "CalculiX solves the trajectory" in readme
    assert "current-surface dynamic SDF" in readme
    assert "Native SFC nonlinear C3D8 dynamic trajectory equivalence" in readme

    summaries = _rows(C3D8_TRAJECTORY_CASE / "data" / "c3d8_contact_trajectory_summary.csv")
    assert {row["case"] for row in summaries} == {"block_plane_c3d8", "block_block_c3d8"}
    assert {row["reference_source"] for row in summaries} == {"calculix_dat"}
    assert all(row["status"] == "external_replay" for row in summaries)
    assert all(float(row["peak_force_rel_error"]) < 1.0e-2 for row in summaries)
    assert all(float(row["peak_energy_rel_error"]) < 1.0e-2 for row in summaries)
    assert all(float(row["min_gap_rel_error"]) < 1.0e-2 for row in summaries)
    assert all(int(row["active_sfc_rows"]) > 0 for row in summaries)

    claims = {row["claim"]: row for row in _rows(C3D8_TRAJECTORY_CASE / "data" / "c3d8_contact_trajectory_claims.csv")}
    assert claims["c3d8_dynamic_sdf_external_trajectory_replay"]["supported"] == "true"
    assert claims["native_sfc_nonlinear_c3d8_trajectory_equivalence"]["supported"] == "false"
