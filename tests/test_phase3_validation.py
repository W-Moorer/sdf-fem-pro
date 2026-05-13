from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_phase3_validation.py"

REQUIRED_VALIDATION_CASES = {
    "tet4_stiffness_rigid_modes",
    "tet4_mass_conservation",
    "boundary_face_extraction",
    "closest_point_projection",
    "contact_gap_sign_convention",
    "contact_jacobian_finite_difference",
    "no_core_abaqus_imports",
    "uniaxial_linear_elastic_patch",
    "gravity_fixed_base_elastic_block",
    "undamped_free_vibration_energy",
    "deformable_block_against_rigid_plane",
    "deformable_deformable_contact_action_reaction",
}


@pytest.fixture(scope="session")
def phase3_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the Phase-3 validation CLI once in quick mode for artifact checks."""
    out_dir = tmp_path_factory.mktemp("phase3-validation")

    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--quick",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return out_dir


def _read_csv_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(dict(row) for row in csv.DictReader(f))
    return rows


def _first_present(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return value.strip()
    return ""


def _normalized_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def test_phase3_quick_mode_writes_csv_and_markdown(phase3_output: Path) -> None:
    csv_files = sorted(phase3_output.glob("*.csv"))
    markdown_files = sorted(phase3_output.glob("*.md"))

    assert csv_files, "Phase-3 quick validation must write at least one CSV file."
    assert markdown_files, "Phase-3 quick validation must write at least one Markdown summary."
    assert all(path.stat().st_size > 0 for path in csv_files + markdown_files)


def test_phase3_required_validation_cases_are_reported(phase3_output: Path) -> None:
    rows = _read_csv_rows(phase3_output.glob("*.csv"))
    reported_cases = {
        _normalized_id(_first_present(row, ("case_id", "case", "validation_case", "name")))
        for row in rows
    }

    assert REQUIRED_VALIDATION_CASES <= reported_cases


def test_phase3_benchmark_labels_use_distinct_workloads(phase3_output: Path) -> None:
    rows = _read_csv_rows(phase3_output.glob("*.csv"))
    benchmark_rows = [
        row
        for row in rows
        if _first_present(row, ("workload", "workload_id"))
        and _first_present(row, ("label", "benchmark_label", "method", "case_id", "case"))
    ]

    workloads_by_label: dict[str, str] = {}
    for row in benchmark_rows:
        label = _first_present(row, ("label", "benchmark_label", "method", "case_id", "case"))
        workload = _first_present(row, ("workload", "workload_id"))
        previous = workloads_by_label.setdefault(label, workload)
        assert previous == workload, f"Benchmark label {label!r} maps to multiple workloads."

    labels_by_workload: dict[str, str] = {}
    for label, workload in workloads_by_label.items():
        previous = labels_by_workload.setdefault(workload, label)
        assert previous == label, (
            f"Benchmark labels {previous!r} and {label!r} reuse workload {workload!r}."
        )


def test_phase3_paper_claims_have_evidence_files(phase3_output: Path) -> None:
    claim_pattern = re.compile(
        r"<!--\s*paper-claim\s+evidence=(?P<evidence>[^>\s]+)\s*-->",
        re.IGNORECASE,
    )

    for markdown_path in phase3_output.glob("*.md"):
        text = markdown_path.read_text(encoding="utf-8")
        for match in claim_pattern.finditer(text):
            evidence = match.group("evidence")
            evidence_path = (markdown_path.parent / evidence).resolve()
            assert evidence_path.is_relative_to(phase3_output.resolve())
            assert evidence_path.is_file(), (
                f"Paper claim in {markdown_path.name} references missing evidence {evidence!r}."
            )
