from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sfepy_external_contact_runner_uses_true_dynamic_sdf_field_path() -> None:
    source = (ROOT / "validation" / "run_external_contact_solver_comparison.py").read_text(encoding="utf-8")

    assert "DynamicNarrowBandSDF" in source
    assert "build_required_points" in source
    assert "dynamic_surface_sdf" not in source
    assert "field interpolation" in source


def test_paper_no_longer_cites_nafems_or_simscale_rounded_punch() -> None:
    manuscript = (ROOT / "paper" / "manuscript.md").read_text(encoding="utf-8")
    main_tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    combined = manuscript + "\n" + main_tex

    assert "NAFEMS" not in combined
    assert "SimScale" not in combined
    assert "rounded-punch" not in combined
    assert "rounded punch" not in combined.lower()


def test_fuzzycontact_runner_is_optional_validation_not_core_dependency() -> None:
    source = (ROOT / "validation" / "run_fuzzycontact_vtu_reference.py").read_text(encoding="utf-8")
    core_sources = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src" / "sfc").rglob("*.py"))

    assert "data.mendeley.com/datasets/tdgkkjyr2f/3" in source
    assert "not a core dependency" in source
    assert "FuzzyContact" not in core_sources

