from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

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
        "recurdyn_gear_calculix_summary.csv",
        "recurdyn_gear_calculix_metadata.csv",
        "recurdyn_gear_calculix_vtk_frames.csv",
        "recurdyn_gear_calculix_figures.csv",
        "recurdyn_gear_calculix_summary.md",
        "initial_gap_samples.csv",
        "initial_gap_summary.csv",
        "contact_law_alignment.csv",
        "contact_law_alignment_summary.csv",
        "vtk/recurdyn_gear_calculix_frame_0000.vtk",
        "figures/recurdyn_gear_surface_preview.png",
        "figures/initial_gap_histogram.png",
        "figures/contact_law_alignment.png",
    ]
    for relative in expected:
        path = out_dir / relative
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    deck = (out_dir / "jiandanjiaolian_calculix_1s.inp").read_text(encoding="utf-8")
    assert "*ELEMENT, TYPE=C3D4, ELSET=GEAR22_SOLID" in deck
    assert "*ELEMENT, TYPE=S3, ELSET=GEAR21_SURF" in deck
    assert "*SURFACE, NAME=GEAR22_SLAVE, TYPE=ELEMENT" in deck
    assert "*CONTACT PAIR, INTERACTION=GEAR_CONTACT, TYPE=SURFACE TO SURFACE" in deck
    assert "*EL PRINT, ELSET=GEAR22_SOLID" in deck
    assert "S,E" in deck

    metadata = {row["key"]: row["value"] for row in _rows(out_dir / "recurdyn_gear_calculix_metadata.csv")}
    assert metadata["flexible_nodes"] == "13630"
    assert metadata["tet4_elements"] == "64644"
    assert metadata["flexible_contact_element_faces"] == "10738"
    assert metadata["slave_surface_mode"] == "element-face"
    assert metadata["run_completed"] == "False"

    gap_summary = {row["sample_type"]: row for row in _rows(out_dir / "initial_gap_summary.csv")}
    assert {"node", "face_centroid", "all"} <= set(gap_summary)
    assert int(gap_summary["all"]["count"]) > 10000

    law_summary = {row["fit"]: row for row in _rows(out_dir / "contact_law_alignment_summary.csv")}
    assert {"parsed_K", "endpoint_fit", "least_squares_fit"} <= set(law_summary)
