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
