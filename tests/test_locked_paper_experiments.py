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
PHASE9_CASE = ROOT / "paper" / "numerical_experiments" / "phase9_full_contact_validation"
OFFICIAL_CALCULIX_CASE = ROOT / "paper" / "numerical_experiments" / "calculix_official_contact_examples"
OFFICIAL_LAW_SDF_CASE = ROOT / "paper" / "numerical_experiments" / "calculix_official_law_sdf_replay"


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


def test_locked_phase9_full_contact_validation_claim_gates() -> None:
    required = [
        PHASE9_CASE / "README.md",
        PHASE9_CASE / "phase9_full_contact_validation.csv",
        PHASE9_CASE / "phase9_claim_gates.csv",
        PHASE9_CASE / "phase9_stress_strain_cloud.csv",
        PHASE9_CASE / "phase9_c3d4_c3d8_side_by_side.csv",
        PHASE9_CASE / "phase9_solver_timing.csv",
        PHASE9_CASE / "phase9_curved_nonplanar_contact_external.csv",
        PHASE9_CASE / "phase9_full_contact_validation_summary.md",
        PHASE9_CASE / "figures" / "phase9_claim_gate_matrix.png",
        PHASE9_CASE / "figures" / "phase9_contact_error_metrics.png",
        PHASE9_CASE / "vtk" / "c3d8_linear_static_contact_0000.vtk",
        PHASE9_CASE / "vtk" / "c3d8_linear_dynamic_block_plane_contact_native_0000.vtk",
        PHASE9_CASE / "vtk" / "c3d8_linear_dynamic_block_block_contact_native_0000.vtk",
        PHASE9_CASE / "vtk" / "c3d8_nonlinear_static_contact_0000.vtk",
        PHASE9_CASE / "vtk" / "c3d8_nonlinear_dynamic_block_plane_contact_0000.vtk",
        PHASE9_CASE / "vtk" / "c3d8_nonlinear_dynamic_block_block_contact_0000.vtk",
        PHASE9_CASE / "vtk" / "c3d8_curved_nonplanar_contact_replay_0000.vtk",
        PHASE9_CASE / "native_c3d8_linear_dynamic_block_plane.csv",
        PHASE9_CASE / "native_c3d8_linear_dynamic_block_block.csv",
        PHASE9_CASE / "native_c3d8_linear_dynamic_comparison.csv",
        PHASE9_CASE / "native_c3d8_nonlinear_static_contactenergy.csv",
        PHASE9_CASE / "native_c3d8_nonlinear_dynamic_block_plane.csv",
        PHASE9_CASE / "native_c3d8_nonlinear_dynamic_block_block.csv",
        PHASE9_CASE / "native_c3d8_nonlinear_dynamic_comparison.csv",
    ]
    for path in required:
        _assert_file(path)

    rows = _rows(PHASE9_CASE / "phase9_full_contact_validation.csv")
    by_case = {row["case_id"]: row for row in rows}
    assert by_case["c3d8_linear_static_contact"]["supports_external_correctness"] == "true"
    assert by_case["c3d8_linear_dynamic_block_plane_contact"]["native_sfc_result"] == "true"
    assert by_case["c3d8_linear_dynamic_block_plane_contact"]["supports_trajectory_equivalence"] == "true"
    assert by_case["c3d8_linear_dynamic_block_block_contact"]["native_sfc_result"] == "true"
    assert by_case["c3d8_linear_dynamic_block_block_contact"]["supports_trajectory_equivalence"] == "true"
    assert by_case["c3d8_nonlinear_static_contact"]["native_sfc_result"] == "true"
    assert by_case["c3d8_nonlinear_static_contact"]["supports_external_correctness"] == "true"
    assert by_case["c3d8_nonlinear_dynamic_block_plane_contact"]["native_sfc_result"] == "true"
    assert by_case["c3d8_nonlinear_dynamic_block_plane_contact"]["supports_trajectory_equivalence"] == "true"
    assert by_case["c3d8_nonlinear_dynamic_block_block_contact"]["native_sfc_result"] == "true"
    assert by_case["c3d8_nonlinear_dynamic_block_block_contact"]["supports_trajectory_equivalence"] == "true"
    assert by_case["c3d8_curved_nonplanar_contact_replay"]["calculix_comparison"] == "true"
    assert by_case["c3d8_curved_nonplanar_contact_replay"]["supports_external_correctness"] == "false"

    gates = _rows(PHASE9_CASE / "phase9_claim_gates.csv")
    assert any(row["allowed"] == "true" for row in gates if row["claim"] == "efficiency")
    nonlinear_dynamic_trajectory = [
        row
        for row in gates
        if row["case_id"] == "c3d8_nonlinear_dynamic_block_plane_contact" and row["claim"] == "trajectory_equivalence"
    ][0]
    assert nonlinear_dynamic_trajectory["allowed"] == "true"
    linear_dynamic_trajectory = [
        row
        for row in gates
        if row["case_id"] == "c3d8_linear_dynamic_block_block_contact" and row["claim"] == "trajectory_equivalence"
    ][0]
    assert linear_dynamic_trajectory["allowed"] == "true"
    side_by_side = _rows(PHASE9_CASE / "phase9_c3d4_c3d8_side_by_side.csv")
    assert {"C3D4", "C3D8"} <= {row["element_type"] for row in side_by_side}
    timings = _rows(PHASE9_CASE / "phase9_solver_timing.csv")
    assert any(row["timing_claim_allowed"] == "true" for row in timings)


def test_locked_official_calculix_contact_example_line_is_claim_gated() -> None:
    required = [
        OFFICIAL_CALCULIX_CASE / "official_calculix_example_catalog.csv",
        OFFICIAL_CALCULIX_CASE / "official_calculix_example_runs.csv",
        OFFICIAL_CALCULIX_CASE / "official_calculix_example_claim_gates.csv",
        OFFICIAL_CALCULIX_CASE / "official_calculix_example_commands.csv",
        OFFICIAL_CALCULIX_CASE / "official_calculix_example_plots.csv",
        OFFICIAL_CALCULIX_CASE / "official_calculix_contact_examples_summary.md",
        OFFICIAL_CALCULIX_CASE / "figures" / "official_calculix_example_classification.png",
        OFFICIAL_CALCULIX_CASE / "figures" / "official_calculix_example_runtime.png",
        OFFICIAL_CALCULIX_CASE / "inputs" / "contactenergy.inp",
        OFFICIAL_CALCULIX_CASE / "inputs" / "scheibe2f2f.inp",
        OFFICIAL_CALCULIX_CASE / "inputs" / "ball.inp",
        OFFICIAL_CALCULIX_CASE / "inputs" / "contact1.inp",
        OFFICIAL_CALCULIX_CASE / "inputs" / "contact3.inp",
        OFFICIAL_CALCULIX_CASE / "inputs" / "contact6.inp",
    ]
    for path in required:
        _assert_file(path)

    catalog = {row["case_id"]: row for row in _rows(OFFICIAL_CALCULIX_CASE / "official_calculix_example_catalog.csv")}
    assert catalog["contactenergy_c3d8_static_energy"]["paper_use"] == "direct_reference_already_locked"
    assert catalog["scheibe2f2f_c3d8_nlgeom_static"]["paper_use"] == "candidate_requires_native_sfc_metric_extraction"
    assert catalog["ball_c3d8_dynamic_drop"]["paper_use"] == "candidate_requires_s8_floor_equivalent_or_filtered_comparison"
    assert catalog["contact1_c3d8_exponential_law"]["paper_use"] == "law_alignment_reference"
    assert catalog["ball_c3d8_dynamic_drop"]["element_types"] == "C3D8;S8"

    runs = {row["case_id"]: row for row in _rows(OFFICIAL_CALCULIX_CASE / "official_calculix_example_runs.csv")}
    assert all(row["run_status"] == "ok" for row in runs.values())
    assert runs["scheibe2f2f_c3d8_nlgeom_static"]["dat_nonempty"] == "false"
    assert runs["scheibe2f2f_c3d8_nlgeom_static"]["frd"]
    assert runs["ball_c3d8_dynamic_drop"]["dat_nonempty"] == "true"

    gates = _rows(OFFICIAL_CALCULIX_CASE / "official_calculix_example_claim_gates.csv")
    by_case_claim = {(row["case_id"], row["claim"]): row for row in gates}
    assert by_case_claim[("contactenergy_c3d8_static_energy", "paper_external_reference_direct")]["allowed"] == "true"
    assert by_case_claim[("contact3_c3d8_linear_law", "small_law_alignment_reference")]["allowed"] == "true"
    assert by_case_claim[("scheibe2f2f_c3d8_nlgeom_static", "recommended_next_adaptation")]["allowed"] == "true"
    assert by_case_claim[("ball_c3d8_dynamic_drop", "recommended_next_adaptation")]["allowed"] == "true"
    assert all(row["allowed"] == "false" for row in gates if row["claim"] == "full_trajectory_or_full_field_equivalence")


def test_locked_official_law_sdf_replay_outputs_deforming_sdf_metrics() -> None:
    required = [
        OFFICIAL_LAW_SDF_CASE / "official_law_sdf_replay_summary.csv",
        OFFICIAL_LAW_SDF_CASE / "official_law_sdf_replay_contact_points.csv",
        OFFICIAL_LAW_SDF_CASE / "official_law_sdf_replay_claim_gates.csv",
        OFFICIAL_LAW_SDF_CASE / "official_law_sdf_replay_commands.csv",
        OFFICIAL_LAW_SDF_CASE / "official_law_sdf_replay_plots.csv",
        OFFICIAL_LAW_SDF_CASE / "official_law_sdf_replay_summary.md",
        OFFICIAL_LAW_SDF_CASE / "figures" / "official_law_sdf_replay_errors.png",
        OFFICIAL_LAW_SDF_CASE / "figures" / "official_law_sdf_replay_force_balance.png",
        OFFICIAL_LAW_SDF_CASE / "inputs" / "contactenergy.inp",
        OFFICIAL_LAW_SDF_CASE / "inputs" / "contact1.inp",
        OFFICIAL_LAW_SDF_CASE / "inputs" / "contact3.inp",
        OFFICIAL_LAW_SDF_CASE / "inputs" / "contact6.inp",
    ]
    for path in required:
        _assert_file(path)

    summaries = {row["case_id"]: row for row in _rows(OFFICIAL_LAW_SDF_CASE / "official_law_sdf_replay_summary.csv")}
    assert summaries["contactenergy_c3d8_surface_linear"]["pressure_law"] == "LINEAR"
    assert summaries["contact1_c3d8_node_exponential"]["pressure_law"] == "EXPONENTIAL"
    assert float(summaries["contactenergy_c3d8_surface_linear"]["max_gap_abs_error"]) <= 1.0e-10
    assert float(summaries["contactenergy_c3d8_surface_linear"]["total_energy_rel_error"]) <= 1.0e-5
    assert float(summaries["contact1_c3d8_node_exponential"]["max_pressure_rel_error"]) <= 1.0e-4
    assert float(summaries["contact3_c3d8_node_linear"]["max_pressure_rel_error"]) <= 5.0e-2
    assert float(summaries["contact6_c3d8_node_stiff_linear"]["max_pressure_rel_error"]) <= 5.0e-2

    gates = _rows(OFFICIAL_LAW_SDF_CASE / "official_law_sdf_replay_claim_gates.csv")
    by_case_claim = {(row["case_id"], row["claim"]): row for row in gates}
    assert by_case_claim[("contactenergy_c3d8_surface_linear", "contact_energy_matches_calculix_cels")]["allowed"] == "true"
    assert by_case_claim[("contact1_c3d8_node_exponential", "pressure_law_matches_calculix_cstr")]["allowed"] == "true"
    assert by_case_claim[("contact1_c3d8_node_exponential", "contact_energy_matches_calculix_cels")]["allowed"] == "false"
    assert by_case_claim[("contact3_c3d8_node_linear", "linear_pressure_law_matches_calculix_cstr")]["allowed"] == "true"
    assert by_case_claim[("contact6_c3d8_node_stiff_linear", "linear_pressure_law_matches_calculix_cstr")]["allowed"] == "true"
    assert all(row["allowed"] == "false" for row in gates if row["claim"] == "native_sfc_trajectory_equivalence")

    contact_points = _rows(OFFICIAL_LAW_SDF_CASE / "official_law_sdf_replay_contact_points.csv")
    assert len(contact_points) >= 4
    assert all(row["calculix_cdis"] != "" for row in contact_points)
    assert all(row["calculix_cstr"] != "" for row in contact_points)
