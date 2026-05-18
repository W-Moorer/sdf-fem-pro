from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_external_visual_validation import run_validation


def test_external_visual_validation_quick_outputs(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path / "external_visual", quick=True)

    required = [
        "metrics",
        "sfc_vtu",
        "reference_vtu",
        "error_vtu",
        "sdf_slice_vtu",
        "sdf_samples_vtu",
        "external_png",
        "external_pdf",
        "sdf_png",
        "sdf_pdf",
        "quadrature_png",
        "quadrature_pdf",
        "sdf_slice_csv",
        "sdf_samples_csv",
        "quadrature_csv",
        "quadrature_vtu",
        "summary",
    ]
    for key in required:
        path = outputs[key]
        assert path.exists(), key
        assert path.stat().st_size > 0, key

    metrics = outputs["metrics"].read_text(encoding="utf-8")
    assert "displacement_l2_rel_error" in metrics
    assert "von_mises_l2_rel_error" in metrics
    assert "pressure_l2_rel_error" in metrics

    summary = outputs["summary"].read_text(encoding="utf-8")
    assert "not used to prove the SDF acceleration claim" in summary
    assert "does not add a core CalculiX/Abaqus dependency" in summary
    assert "surface-to-surface quadrature contact visualization" in summary


def test_external_visual_validation_does_not_add_core_external_solver_dependency() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8").lower()
    dependencies_block = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]

    assert "calculix" not in dependencies_block
    assert "abaqus" not in dependencies_block
