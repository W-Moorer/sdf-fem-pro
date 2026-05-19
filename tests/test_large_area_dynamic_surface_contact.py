from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_large_area_dynamic_surface_contact import default_config, run_benchmark


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_large_area_dynamic_surface_contact_quick_outputs(tmp_path: Path) -> None:
    cfg = replace(
        default_config(quick=True),
        nx=4,
        ny=4,
        nz=1,
        driver_nx=4,
        driver_ny=4,
        total_time=0.05,
        dt=0.01,
        ramp_time=0.02,
        quadrature_order=3,
        spacing=0.6,
        band_radius=2.0,
        peak_pressure=2000.0,
        newmark_iterations=1,
    )
    outputs = run_benchmark(out_dir=tmp_path, quick=True, skip_calculix=True, cfg=cfg)
    rows = _read_rows(Path(outputs["sfc_history"]))
    assert rows
    assert max(int(row["active_samples"]) for row in rows) > 0
    assert {row["field_path"] for row in rows} == {"RequiredPointSDFWorkspace.build -> field_contact"}
    assert {row["load_type"] for row in rows} == {"top_pressure"}
    assert Path(outputs["summary"]).exists()
    assert Path(outputs["vtk_pvd"]).exists()
    assert int(outputs["vtk_frame_count"]) >= 2


def test_large_area_dynamic_surface_contact_deferred_diagnostics_outputs(tmp_path: Path) -> None:
    cfg = replace(
        default_config(quick=True),
        nx=3,
        ny=3,
        nz=1,
        driver_nx=3,
        driver_ny=3,
        total_time=0.02,
        dt=0.01,
        ramp_time=0.01,
        quadrature_order=3,
        spacing=0.8,
        band_radius=2.0,
        peak_pressure=2000.0,
        newmark_iterations=1,
        defer_diagnostics=True,
    )
    outputs = run_benchmark(out_dir=tmp_path, quick=True, skip_calculix=True, cfg=cfg)
    rows = _read_rows(Path(outputs["sfc_history"]))
    assert rows
    assert {row["accepted_response_reused"] for row in rows} == {"postprocess"}
    summary = Path(outputs["summary"]).read_text(encoding="utf-8")
    assert "Deferred diagnostics/postprocess: `True`" in summary
    assert "SFC solve-loop wall time excluding deferred diagnostics" in summary


def test_large_area_dynamic_runner_uses_true_field_contact_path() -> None:
    source = Path("validation/run_large_area_dynamic_surface_contact.py").read_text(encoding="utf-8")
    assert "DynamicNarrowBandSDF.build_required_points" in source
    assert "surface_to_surface_field_penalty_response_vectorized" in source
    assert "field_contact" in source
