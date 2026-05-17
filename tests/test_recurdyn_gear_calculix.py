from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_recurdyn_gear_calculix import parse_recurdyn_rmd

RMD = ROOT / "assets" / "jiandanjiaolian.rmd"
RUNNER = ROOT / "validation" / "run_recurdyn_gear_calculix.py"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_recurdyn_rmd_parser_extracts_gear_contact_model() -> None:
    model = parse_recurdyn_rmd(RMD)

    assert len(model.nodes) == 13630
    assert len(model.elements) == 64644
    assert model.rigid_surface.nodes_global.shape == (1641, 3)
    assert model.rigid_surface.patches.shape == (3278, 3)
    assert model.rigid_surface.marker_id == 12
    assert model.rigid_surface.marker_pose.part_id == 3
    np.testing.assert_allclose(model.rigid_surface.marker_pose.qp, [17.46, -1.366543853054, -0.478799197013])
    np.testing.assert_allclose(model.rigid_surface.part_pose.reuler, [0.0, 0.00349065850398866, 0.0])
    translated_only = model.rigid_surface.nodes_local + model.rigid_surface.marker_pose.qp[None, :]
    assert not np.allclose(model.rigid_surface.nodes_global, translated_only)
    assert model.flexible_surface_patches.shape == (10738, 3)
    assert len(model.hub_nodes) == 607
    assert model.material.E == 200000
    assert model.material.nu == 0.285
    assert model.contact.stiffness == 100000
    assert model.contact.order == 2


def test_recurdyn_gear_runner_quick_generates_calculix_deck_and_vtk(tmp_path: Path) -> None:
    out_dir = tmp_path / "recurdyn_gear"
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--quick",
            "--skip-calculix",
            "--drive-mode",
            "prescribed-surface",
            "--slave-surface-mode",
            "element-face",
            "--analysis",
            "preload-restart-dynamic",
            "--contact-law",
            "tabular",
            "--contact-adjust",
            "0.0",
            "--out-dir",
            str(out_dir),
            "--gap-candidate-count",
            "8",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )

    expected = [
        "jiandanjiaolian_calculix_1s.inp",
        "jiandanjiaolian_calculix_1s_restart.inp",
        "recurdyn_gear_calculix_summary.csv",
        "recurdyn_gear_calculix_metadata.csv",
        "recurdyn_gear_calculix_vtk_frames.csv",
        "recurdyn_gear_calculix_figures.csv",
        "recurdyn_gear_calculix_summary.md",
        "initial_gap_samples.csv",
        "initial_gap_summary.csv",
        "contact_law_alignment.csv",
        "contact_law_alignment_summary.csv",
        "contact_initialization_samples.csv",
        "contact_initialization_summary.csv",
        "vtk/recurdyn_gear_calculix_frame_0000.vtk",
        "figures/recurdyn_gear_surface_preview.png",
        "figures/initial_gap_histogram.png",
        "figures/contact_law_alignment.png",
        "figures/contact_initialization.png",
    ]
    for relative in expected:
        path = out_dir / relative
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    deck = (out_dir / "jiandanjiaolian_calculix_1s.inp").read_text(encoding="utf-8")
    assert "*ELEMENT, TYPE=C3D4, ELSET=GEAR22_SOLID" in deck
    assert "*ELEMENT, TYPE=S3, ELSET=GEAR21_SURF" in deck
    assert "*SURFACE, NAME=GEAR22_SLAVE, TYPE=ELEMENT" in deck
    assert "*CONTACT PAIR, INTERACTION=GEAR_CONTACT, TYPE=SURFACE TO SURFACE, ADJUST=0.0" in deck
    assert "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=TABULAR" in deck
    assert deck.count("*STEP, NLGEOM") == 1
    assert "*STATIC" in deck
    assert "*RESTART, WRITE, FREQUENCY=1" in deck
    assert "*EL PRINT, ELSET=GEAR22_SOLID" in deck
    assert "S,E" in deck
    restart_deck = (out_dir / "jiandanjiaolian_calculix_1s_restart.inp").read_text(encoding="utf-8")
    assert "*RESTART, READ, STEP=1" in restart_deck
    assert "*DYNAMIC" in restart_deck
    assert "*NODE PRINT, NSET=GEAR22_NALL" in restart_deck
    assert "V" in restart_deck

    metadata = {row["key"]: row["value"] for row in _rows(out_dir / "recurdyn_gear_calculix_metadata.csv")}
    assert metadata["flexible_nodes"] == "13630"
    assert metadata["tet4_elements"] == "64644"
    assert metadata["flexible_contact_element_faces"] == "10738"
    assert metadata["slave_surface_mode"] == "element-face"
    assert metadata["rigid_surface_marker_id"] == "12"
    assert metadata["rigid_surface_marker_part_id"] == "3"
    assert metadata["analysis"] == "preload-restart-dynamic"
    assert metadata["contact_law"] == "tabular"
    assert metadata["restart_write"] == "true"
    assert metadata["restart_inp_path"].endswith("jiandanjiaolian_calculix_1s_restart.inp")
    assert metadata["contact_adjust"] == "0.0"
    assert metadata["run_completed"] == "False"

    gap_summary = {row["sample_type"]: row for row in _rows(out_dir / "initial_gap_summary.csv")}
    assert {"node", "face_centroid", "all"} <= set(gap_summary)
    assert int(gap_summary["all"]["count"]) > 10000

    law_summary = {row["fit"]: row for row in _rows(out_dir / "contact_law_alignment_summary.csv")}
    assert {"parsed_K", "endpoint_fit", "least_squares_fit"} <= set(law_summary)

    init_summary = {row["sample_type"]: row for row in _rows(out_dir / "contact_initialization_summary.csv")}
    assert {"node", "face_centroid", "all"} <= set(init_summary)
    assert int(init_summary["all"]["active_candidate_count"]) > 0
    assert init_summary["all"]["preload_recommended"] == "true"
    assert init_summary["all"]["contact_adjust_recommended"] == "false"
    assert init_summary["all"]["recommended_initialization"] == "static_preload_then_dynamic_restart"
