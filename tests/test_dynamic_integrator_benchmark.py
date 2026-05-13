from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_dynamic_integrator_benchmark.py"


EXPECTED_FILES = {
    "dynamic_integrator_timeseries.csv",
    "dynamic_integrator_metrics.csv",
    "dynamic_integrator_plots.csv",
    "dynamic_integrator_summary.md",
    "constant_acceleration_free_fall_displacement.png",
    "constant_acceleration_free_fall_displacement.pdf",
    "constant_acceleration_free_fall_energy.png",
    "constant_acceleration_free_fall_energy.pdf",
    "undamped_harmonic_oscillator_displacement.png",
    "undamped_harmonic_oscillator_displacement.pdf",
    "undamped_harmonic_oscillator_energy.png",
    "undamped_harmonic_oscillator_energy.pdf",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_dynamic_integrator_benchmark_quick_outputs(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(RUNNER), "--quick", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    produced = {path.name for path in tmp_path.iterdir() if path.is_file()}
    assert EXPECTED_FILES <= produced
    assert all((tmp_path / name).stat().st_size > 0 for name in EXPECTED_FILES)

    metrics = _rows(tmp_path / "dynamic_integrator_metrics.csv")
    assert metrics
    assert {row["status"] for row in metrics} == {"pass"}
    by_metric = {(row["case"], row["metric"]): float(row["value"]) for row in metrics}
    assert by_metric[("constant_acceleration_free_fall", "max_displacement_abs_error")] <= 1.0e-10
    assert by_metric[("undamped_harmonic_oscillator", "relative_total_energy_drift")] <= 1.0e-10


def test_dynamic_integrator_summary_claim_markers_have_backing_csv_fields(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(RUNNER), "--quick", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    marker = re.compile(r"<!--\s*evidence\s+csv=(?P<csv>\S+)\s+field=(?P<field>\S+)\s*-->")
    text = (tmp_path / "dynamic_integrator_summary.md").read_text(encoding="utf-8")
    matches = list(marker.finditer(text))

    assert matches
    for match in matches:
        csv_path = tmp_path / match.group("csv")
        assert csv_path.is_file()
        with csv_path.open(newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames or []
        assert match.group("field") in fieldnames
