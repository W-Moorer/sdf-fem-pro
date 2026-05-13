from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_contact_benchmark import (
    build_contact_scene,
    comparison_method_specs,
    phase_benchmark_specs,
)


def test_quick_contact_benchmark_runs_in_ci(tmp_path: Path) -> None:
    script = ROOT / "benchmarks" / "run_contact_benchmark.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--quick",
            "--out-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "contact_benchmark.csv" in result.stdout
    assert (tmp_path / "contact_benchmark.csv").exists()
    assert (tmp_path / "contact_benchmark.md").exists()


def test_benchmark_comparison_methods_use_distinct_workloads() -> None:
    specs = comparison_method_specs(build_contact_scene())
    workloads = [spec.workload for spec in specs]

    assert len(workloads) == len(set(workloads))


def test_benchmark_phase_rows_are_isolated_and_complete() -> None:
    stages = {spec.stage for spec in phase_benchmark_specs(build_contact_scene())}

    assert {
        "build_update_broad_phase",
        "candidate_query_only",
        "local_projection_only",
        "dynamic_sdf_query_only",
        "contact_jacobian_assembly_only",
        "penalty_force_assembly_only",
        "full_contact_pipeline_end_to_end",
        "fem_assembly_only",
        "linear_solve_only",
    } <= stages
