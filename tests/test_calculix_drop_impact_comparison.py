from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_calculix_drop_impact_comparison.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _calculix_available() -> bool:
    try:
        from validation.run_calculix_drop_impact_comparison import calculix_available

        return calculix_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _calculix_available(), reason="CalculiX/ccx is unavailable through WSL")

EXPECTED_FILES = {
    "calculix_drop_time_history.csv",
    "calculix_drop_metrics.csv",
    "calculix_drop_commands.csv",
    "calculix_drop_plots.csv",
    "calculix_drop_metadata.csv",
    "calculix_drop_summary.md",
    "calculix_drop_z_cm.png",
    "calculix_drop_z_cm.pdf",
    "calculix_drop_min_gap.png",
    "calculix_drop_min_gap.pdf",
    "calculix_drop_force_proxy.png",
    "calculix_drop_force_proxy.pdf",
    "calculix_drop_contact_energy.png",
    "calculix_drop_contact_energy.pdf",
    "calculix_drop_total_energy.png",
    "calculix_drop_total_energy.pdf",
}


@pytest.fixture(scope="session")
def calculix_drop_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("calculix-drop-impact")
    subprocess.run(
        [sys.executable, str(RUNNER), "--quick", "--out-dir", str(out_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return out_dir


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_calculix_drop_quick_mode_writes_expected_outputs(calculix_drop_output: Path) -> None:
    produced = {path.name for path in calculix_drop_output.iterdir() if path.is_file()}

    assert EXPECTED_FILES <= produced
    assert all((calculix_drop_output / name).stat().st_size > 0 for name in EXPECTED_FILES)


def test_calculix_drop_history_contains_external_and_sfc_contact(calculix_drop_output: Path) -> None:
    rows = _rows(calculix_drop_output / "calculix_drop_time_history.csv")
    sources = {row["source"] for row in rows}
    cases = {row["case"] for row in rows}

    assert sources == {"calculix", "sfc_calculix_aligned"}
    assert {"sphere_like_drop", "block_drop"} <= cases
    assert any(int(row["active_contact_count"]) > 0 for row in rows if row["source"] == "calculix")
    assert any(int(row["active_contact_count"]) > 0 for row in rows if row["source"] == "sfc_calculix_aligned")
    assert min(float(row["min_gap"]) for row in rows if row["source"] == "sfc_calculix_aligned") < 0.0
    assert all(np.isfinite(float(row["min_gap"])) for row in rows)
    assert any(row["normal_force_source"] == "floor_rf_total" for row in rows if row["source"] == "calculix")
    assert any(row["normal_force_source"] == "smooth_penalty_tangent" for row in rows if row["source"] == "sfc_calculix_aligned")
    assert all(row["total_mechanical_energy_proxy"] != "" for row in rows if row["source"] == "sfc_calculix_aligned")


def test_calculix_drop_external_contact_claim_is_supported(calculix_drop_output: Path) -> None:
    metrics = _rows(calculix_drop_output / "calculix_drop_metrics.csv")
    by_metric = {row["metric"]: row for row in metrics}

    claims = [row for row in metrics if row["metric"] == "external_dynamic_contact_claim"]
    assert claims
    assert {row["value"] for row in metrics if row["metric"] == "external_solver"} == {"CalculiX"}
    claims_by_case = {row["case"]: row for row in claims}
    assert claims_by_case["block_drop"]["status"] == "supported"
    assert claims_by_case["sphere_like_drop"]["status"] == "supported"
    supported_errors = [
        float(row["value"])
        for row in metrics
        if row["metric"] == "first_contact_time_abs_error" and row["status"] == "supported"
    ]
    assert supported_errors
    assert max(supported_errors) <= 3.0e-3


def test_calculix_drop_nonquick_suite_has_multiple_resolutions() -> None:
    from validation.run_calculix_drop_impact_comparison import build_model_suite

    models = build_model_suite(quick=False)
    resolutions_by_case: dict[str, set[int]] = {}
    for model in models:
        resolutions_by_case.setdefault(model.case, set()).add(model.resolution)

    assert resolutions_by_case["sphere_like_drop"] == {1, 2, 3}
    assert resolutions_by_case["block_drop"] == {1, 2, 3}
    assert all(model.slave_face_refs for model in models)
    assert all(float(model.surface_node_areas.sum()) > 0.0 for model in models)


def test_sphere_like_drop_has_regular_bottom_contact_patch() -> None:
    from validation.run_calculix_drop_impact_comparison import build_drop_model

    model = build_drop_model(quick=True, case="sphere_like_drop", resolution=1)
    bottom_z = float(model.nodes[:, 2].min())
    bottom_faces = [face for face in model.surface_faces if np.allclose(model.nodes[face, 2], bottom_z)]
    near_bottom_faces = [face for face in model.surface_faces if float(model.nodes[face, 2].mean()) <= bottom_z + 0.02]

    assert "bottom_contact_patch" in model.model_source
    assert len(bottom_faces) >= 24
    assert len(near_bottom_faces) >= 48
    assert model.nodes.shape[0] >= 100
    assert model.contact_stiffness > build_drop_model(quick=True, case="block_drop", resolution=1).contact_stiffness


def test_drop_history_uses_mass_weighted_center_height() -> None:
    from validation.run_calculix_drop_impact_comparison import build_drop_model, _mass_weighted_center_z

    model = build_drop_model(quick=True, case="sphere_like_drop", resolution=1)
    current = model.nodes.copy()
    current[model.surface_indices[0], 2] += 10.0

    nodal_average = float(np.mean(current[:, 2]))
    mass_weighted = _mass_weighted_center_z(model, current)

    assert mass_weighted != pytest.approx(nodal_average)
    assert min(current[:, 2]) <= mass_weighted <= max(current[:, 2])


def test_drop_contact_uses_calculix_spos_plane_and_face_integration() -> None:
    from validation.run_calculix_drop_impact_comparison import (
        CALCULIX_MASTER_SURFACE_OFFSET,
        _calculix_aligned_plane_contact,
        _contact_plane_z,
        _surface_face_gaps_to_contact_plane,
        build_drop_model,
    )

    model = build_drop_model(quick=True, case="block_drop", resolution=1, gravity=0.0, initial_velocity_z=0.0)
    x_current = model.nodes.copy()
    gaps = _surface_face_gaps_to_contact_plane(model, x_current)

    assert _contact_plane_z(model) == pytest.approx(model.floor_z + CALCULIX_MASTER_SURFACE_OFFSET)
    assert np.min(gaps) == pytest.approx(0.045)

    x_current[:, 2] -= 0.046
    force, stiffness, min_gap, active_count, max_penetration, contact_energy = _calculix_aligned_plane_contact(model, x_current)

    assert min_gap < 0.0
    assert max_penetration > 0.0
    assert active_count > 0
    assert contact_energy > 0.0
    assert np.sum(force[2::3]) > 0.0
    assert stiffness.nnz > 0


def test_calculix_drop_input_and_sfc_mode_use_hht_direct_alignment(tmp_path: Path) -> None:
    from validation.run_calculix_drop_impact_comparison import (
        CALCULIX_DEFAULT_HHT_ALPHA,
        _hht_newmark_parameters,
        build_drop_model,
        write_calculix_input,
    )

    beta, gamma = _hht_newmark_parameters(CALCULIX_DEFAULT_HHT_ALPHA)
    assert beta == pytest.approx(0.25 * (1.0 - CALCULIX_DEFAULT_HHT_ALPHA) ** 2)
    assert gamma == pytest.approx(0.5 - CALCULIX_DEFAULT_HHT_ALPHA)

    model = build_drop_model(quick=True, case="block_drop", resolution=1)
    inp = tmp_path / "drop.inp"
    write_calculix_input(model, inp)
    text = inp.read_text(encoding="utf-8").lower()

    assert "*dynamic, direct, alpha=-0.05" in text
    assert "*element, type=c3d4" in text
    assert "*surface, name=ball, type=element" in text
    assert "*contact pair, interaction=contact,type=surface to surface" in text


def test_calculix_drop_long_diagnostic_options_are_written(tmp_path: Path) -> None:
    from validation.run_calculix_drop_impact_comparison import build_drop_model, write_calculix_input

    model = build_drop_model(
        quick=True,
        case="block_drop",
        resolution=1,
        duration=3.0,
        dt=5.0e-3,
        output_frequency=10,
        direct_dynamic=False,
    )
    inp = tmp_path / "drop_long.inp"
    write_calculix_input(model, inp)
    text = inp.read_text(encoding="utf-8").lower()

    assert model.total_time == pytest.approx(3.0)
    assert model.dt == pytest.approx(5.0e-3)
    assert model.output_frequency == 10
    assert not model.direct_dynamic
    assert "*dynamic, alpha=-0.05" in text
    assert "*dynamic, direct" not in text
    assert "0.005, 3, 5e-07, 0.005" in text
    assert "frequency=10" in text


def test_calculix_drop_gentle_reference_overrides_are_written(tmp_path: Path) -> None:
    from validation.run_calculix_drop_impact_comparison import build_model_suite, write_calculix_input

    models = build_model_suite(
        quick=True,
        cases=["block_drop"],
        duration=3.0,
        dt=2.0e-3,
        initial_velocity_z=-0.1,
        gravity=0.0,
        contact_stiffness_override=5000.0,
    )

    assert len(models) == 1
    model = models[0]
    assert model.case == "block_drop"
    assert model.total_time == pytest.approx(3.0)
    assert model.dt == pytest.approx(2.0e-3)
    assert model.initial_velocity_z == pytest.approx(-0.1)
    assert model.gravity == pytest.approx(0.0)
    assert model.contact_stiffness == pytest.approx(5000.0)

    inp = tmp_path / "gentle.inp"
    write_calculix_input(model, inp)
    text = inp.read_text(encoding="utf-8").lower()
    assert "nall, 3, -0.1" in text
    assert "elall, grav, 0, 0.0, 0.0, -1.0" in text
    assert "\n5000\n" in text


def test_calculix_drop_summary_claim_markers_have_backing_csv_fields(calculix_drop_output: Path) -> None:
    marker = re.compile(r"<!--\s*evidence\s+csv=(?P<csv>\S+)\s+field=(?P<field>\S+)\s*-->")
    text = (calculix_drop_output / "calculix_drop_summary.md").read_text(encoding="utf-8")
    matches = list(marker.finditer(text))

    assert matches
    for match in matches:
        csv_path = calculix_drop_output / match.group("csv")
        assert csv_path.is_file()
        with csv_path.open(newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames or []
        assert match.group("field") in fieldnames

    metadata = {row["key"]: row["value"] for row in _rows(calculix_drop_output / "calculix_drop_metadata.csv")}
    assert metadata["sfc_mass_matrix"] == "consistent"
    assert metadata["calculix_dynamic_keyword"] == "*DYNAMIC,DIRECT,ALPHA=-0.05"
