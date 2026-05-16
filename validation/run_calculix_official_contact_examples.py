"""Catalog and run selected official CalculiX contact examples.

The purpose of this runner is to keep the paper-facing external evidence line
honest.  It classifies official CalculiX examples that are relevant to the SFC
contact solver, runs them when possible, and writes claim gates that separate
directly usable references from future adaptation targets.

No data from this script is imported by the core solver.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


Row = dict[str, Any]

DEFAULT_WSL_EXAMPLES_DIR = "/usr/share/doc/calculix-ccx-test/examples/test"


@dataclass(frozen=True)
class OfficialExample:
    case_id: str
    file_name: str
    role: str
    expected_scope: str
    paper_use: str
    run_in_quick: bool = True


OFFICIAL_EXAMPLES = [
    OfficialExample(
        case_id="contactenergy_c3d8_static_energy",
        file_name="contactenergy.inp",
        role="official_static_contact_energy_reference",
        expected_scope="C3D8 surface-to-surface static contact law, RF, and CELS reference",
        paper_use="direct_reference_already_locked",
    ),
    OfficialExample(
        case_id="scheibe2f2f_c3d8_nlgeom_static",
        file_name="scheibe2f2f.inp.gz",
        role="official_nonlinear_static_surface_to_surface_candidate",
        expected_scope="C3D8 NLGEOM surface-to-surface contact with DLOAD and FRD output",
        paper_use="candidate_requires_native_sfc_metric_extraction",
    ),
    OfficialExample(
        case_id="ball_c3d8_dynamic_drop",
        file_name="ball.inp.gz",
        role="official_dynamic_drop_candidate",
        expected_scope="C3D8 ball with shell floor, NLGEOM dynamic node-to-surface contact",
        paper_use="candidate_requires_s8_floor_equivalent_or_filtered_comparison",
    ),
    OfficialExample(
        case_id="contact1_c3d8_exponential_law",
        file_name="contact1.inp",
        role="small_contact_law_alignment",
        expected_scope="small C3D8 NLGEOM node-to-surface contact with exponential overclosure",
        paper_use="law_alignment_reference",
    ),
    OfficialExample(
        case_id="contact3_c3d8_linear_law",
        file_name="contact3.inp",
        role="small_contact_law_alignment",
        expected_scope="small C3D8 NLGEOM node-to-surface contact with linear overclosure",
        paper_use="law_alignment_reference",
    ),
    OfficialExample(
        case_id="contact6_c3d8_stiff_linear_law",
        file_name="contact6.inp",
        role="small_contact_law_alignment",
        expected_scope="small C3D8 NLGEOM node-to-surface contact with stiff linear overclosure",
        paper_use="law_alignment_reference",
    ),
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _wsl_available() -> bool:
    if shutil.which("wsl") is None:
        return False
    try:
        return _run(["wsl", "--exec", "bash", "-lc", "true"], cwd=ROOT, timeout=15).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _calculix_available() -> bool:
    if not _wsl_available():
        return False
    return _run(["wsl", "--exec", "bash", "-lc", "command -v ccx >/dev/null"], cwd=ROOT, timeout=15).returncode == 0


def _wsl_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def _calculix_version() -> str:
    if not _calculix_available():
        return "unavailable"
    proc = _run(["wsl", "--exec", "bash", "-lc", "ccx -v 2>&1"], cwd=ROOT, timeout=20)
    lines = [line.strip() for line in (proc.stdout + proc.stderr).splitlines() if line.strip()]
    return lines[0] if lines else "CalculiX"


def _read_local_example(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_wsl_example(file_name: str, examples_dir: str) -> tuple[str, str]:
    source = f"{examples_dir.rstrip('/')}/{file_name}"
    command = f"test -f {_wsl_quote(source)} && " + (
        f"gzip -dc {_wsl_quote(source)}" if file_name.endswith(".gz") else f"cat {_wsl_quote(source)}"
    )
    proc = _run(["wsl", "--exec", "bash", "-lc", command], cwd=ROOT, timeout=45)
    if proc.returncode != 0:
        raise FileNotFoundError(source)
    return proc.stdout, source


def _read_example(file_name: str, examples_dir: Path | None, wsl_examples_dir: str) -> tuple[str, str]:
    if examples_dir is not None:
        path = examples_dir / file_name
        if not path.exists():
            raise FileNotFoundError(path)
        return _read_local_example(path), str(path)
    if not _wsl_available():
        raise FileNotFoundError("WSL is unavailable and no --examples-dir was provided")
    return _read_wsl_example(file_name, wsl_examples_dir)


def _keyword_name(line: str) -> str:
    return line.split(",", 1)[0].strip().upper()


def _keyword_param(line: str, name: str) -> str:
    for part in line.split(",")[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key.strip().upper() == name.upper():
            return value.strip()
    return ""


def _is_data_row(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("*") and not stripped.startswith("**")


def _unique_join(values: list[str]) -> str:
    unique = sorted({value for value in values if value})
    return ";".join(unique)


def parse_official_input(text: str) -> Row:
    element_types: list[str] = []
    contact_pair_types: list[str] = []
    step_types: list[str] = []
    pressure_laws: list[str] = []
    pressure_values: list[str] = []
    node_count = 0
    element_count = 0
    surface_count = 0
    material_count = 0
    has_nlgeom = False
    has_friction = False
    has_plastic = False
    has_hyperelastic = False
    has_dynamic = False
    has_static = False
    has_shell_or_beam = False
    last_section = ""
    expect_surface_behavior = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            upper = line.upper()
            section = _keyword_name(line)
            last_section = section
            if section == "*ELEMENT":
                element_type = _keyword_param(line, "TYPE").upper()
                element_types.append(element_type)
                if element_type.startswith(("S", "B", "CPS", "CAX")):
                    has_shell_or_beam = True
            elif section == "*CONTACT PAIR":
                contact_pair_types.append(_keyword_param(line, "TYPE").upper() or "DEFAULT")
            elif section == "*STEP":
                has_nlgeom = has_nlgeom or ("NLGEOM" in upper)
            elif section == "*STATIC":
                has_static = True
                step_types.append("STATIC")
            elif section == "*DYNAMIC":
                has_dynamic = True
                step_types.append("DYNAMIC")
            elif section == "*SURFACE":
                surface_count += 1
            elif section == "*MATERIAL":
                material_count += 1
            elif section == "*SURFACE BEHAVIOR":
                law = _keyword_param(line, "PRESSURE-OVERCLOSURE").upper() or "UNSPECIFIED"
                pressure_laws.append(law)
                expect_surface_behavior = True
            elif section == "*FRICTION":
                has_friction = True
            elif section == "*PLASTIC":
                has_plastic = True
            elif section == "*HYPERELASTIC":
                has_hyperelastic = True
            continue

        if expect_surface_behavior and _is_data_row(line):
            pressure_values.append(line)
            expect_surface_behavior = False
        if last_section == "*NODE" and _is_data_row(line):
            node_count += 1
        elif last_section == "*ELEMENT" and _is_data_row(line):
            element_count += 1

    supported_elements = {"C3D4", "C3D8"}
    element_set = {value for value in element_types if value}
    only_supported_solids = bool(element_set) and element_set <= supported_elements
    has_supported_solid = bool(element_set & supported_elements)
    adaptation_level = "direct" if only_supported_solids and not has_friction and not has_plastic and not has_hyperelastic else "adapt"
    if not has_supported_solid:
        adaptation_level = "future_backend_required"

    return {
        "node_count": node_count,
        "element_count": element_count,
        "element_types": _unique_join(element_types),
        "contact_pair_types": _unique_join(contact_pair_types),
        "step_types": _unique_join(step_types),
        "surface_count": surface_count,
        "material_count": material_count,
        "pressure_overclosure_laws": _unique_join(pressure_laws),
        "pressure_overclosure_values": " | ".join(pressure_values),
        "has_nlgeom": str(has_nlgeom).lower(),
        "has_dynamic": str(has_dynamic).lower(),
        "has_static": str(has_static).lower(),
        "has_friction": str(has_friction).lower(),
        "has_plastic": str(has_plastic).lower(),
        "has_hyperelastic": str(has_hyperelastic).lower(),
        "has_shell_or_beam": str(has_shell_or_beam).lower(),
        "has_supported_solid": str(has_supported_solid).lower(),
        "only_supported_solids": str(only_supported_solids).lower(),
        "adaptation_level": adaptation_level,
    }


def _job_name(file_name: str) -> str:
    name = file_name
    if name.endswith(".gz"):
        name = name[:-3]
    return Path(name).stem


def _parse_vector_blocks(dat_text: str, label: str) -> list[dict[int, np.ndarray]]:
    blocks: list[dict[int, np.ndarray]] = []
    lines = dat_text.splitlines()
    for index, line in enumerate(lines):
        if label.lower() not in line.lower():
            continue
        block: dict[int, np.ndarray] = {}
        for row in lines[index + 1 :]:
            parts = row.split()
            if len(parts) == 4 and parts[0].lstrip("+-").isdigit():
                block[int(parts[0])] = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
            elif block:
                break
        if block:
            blocks.append(block)
    return blocks


def _von_mises(stress: np.ndarray) -> float:
    sxx, syy, szz, sxy, sxz, syz = np.asarray(stress, dtype=float)
    return float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + sxz**2 + syz**2)))


def _parse_stress_max(dat_text: str) -> float:
    values: list[float] = []
    for raw in dat_text.splitlines():
        parts = raw.split()
        if len(parts) == 8 and parts[0].lstrip("+-").isdigit() and parts[1].lstrip("+-").isdigit():
            try:
                values.append(_von_mises(np.asarray([float(value) for value in parts[2:8]], dtype=float)))
            except ValueError:
                continue
    return max(values) if values else float("nan")


def parse_dat_metrics(dat_path: Path) -> Row:
    if not dat_path.exists() or dat_path.stat().st_size == 0:
        return {
            "dat_available": str(dat_path.exists()).lower(),
            "dat_nonempty": "false",
            "dat_time_block_count": 0,
            "max_displacement_norm": "",
            "max_force_norm": "",
            "max_von_mises_from_dat": "",
            "total_contact_spring_energy": "",
            "raw_cdis_rows": 0,
            "raw_cstr_rows": 0,
            "raw_cels_rows": 0,
        }
    text = dat_path.read_text(encoding="utf-8", errors="ignore")
    displacement_blocks = _parse_vector_blocks(text, "displacements")
    force_blocks = _parse_vector_blocks(text, "forces")
    energy_matches = re.findall(
        r"total contact spring energy for time\s+[-+0-9.Ee]+\s*\n\s*([-+0-9.Ee]+)",
        text,
        flags=re.IGNORECASE,
    )
    cdis_rows = len(re.findall(r"\b[-+]?\d+\s+[-+]?\d+\s+[-+0-9.Ee]+\s*$", text, flags=re.MULTILINE))
    max_u = ""
    if displacement_blocks:
        max_u = max(float(np.linalg.norm(value)) for value in displacement_blocks[-1].values())
    max_f = ""
    if force_blocks:
        max_f = max(float(np.linalg.norm(value)) for value in force_blocks[-1].values())
    return {
        "dat_available": "true",
        "dat_nonempty": "true",
        "dat_time_block_count": len(re.findall(r"\btime\s+[-+0-9.Ee]+", text, flags=re.IGNORECASE)),
        "max_displacement_norm": max_u,
        "max_force_norm": max_f,
        "max_von_mises_from_dat": _parse_stress_max(text),
        "total_contact_spring_energy": float(energy_matches[-1]) if energy_matches else "",
        "raw_cdis_rows": cdis_rows,
        "raw_cstr_rows": text.lower().count("contact stress"),
        "raw_cels_rows": text.lower().count("contact spring energy"),
    }


def run_calculix_case(*, case: OfficialExample, text: str, out_dir: Path, skip_run: bool, timeout: int) -> Row:
    run_dir = out_dir / "calculix_runs" / case.case_id
    run_dir.mkdir(parents=True, exist_ok=True)
    job = _job_name(case.file_name)
    inp_path = run_dir / f"{job}.inp"
    inp_path.write_text(text, encoding="utf-8")
    if skip_run:
        return {
            "case_id": case.case_id,
            "job_name": job,
            "run_status": "not_run",
            "return_code": "",
            "wall_time_seconds": "",
            "inp": str(inp_path.relative_to(out_dir)),
            "dat": "",
            "frd": "",
            "sta": "",
            "cvg": "",
            **parse_dat_metrics(run_dir / f"{job}.dat"),
        }
    if not _calculix_available():
        return {
            "case_id": case.case_id,
            "job_name": job,
            "run_status": "calculix_unavailable",
            "return_code": "",
            "wall_time_seconds": "",
            "inp": str(inp_path.relative_to(out_dir)),
            "dat": "",
            "frd": "",
            "sta": "",
            "cvg": "",
            **parse_dat_metrics(run_dir / f"{job}.dat"),
        }

    wsl_run_dir = _wsl_path(run_dir)
    command = f"cd {_wsl_quote(wsl_run_dir)} && ccx {_wsl_quote(job)}"
    start = time.perf_counter()
    proc = _run(["wsl", "--exec", "bash", "-lc", command], cwd=ROOT, timeout=timeout)
    wall = time.perf_counter() - start
    (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
    dat = run_dir / f"{job}.dat"
    frd = run_dir / f"{job}.frd"
    sta = run_dir / f"{job}.sta"
    cvg = run_dir / f"{job}.cvg"
    return {
        "case_id": case.case_id,
        "job_name": job,
        "run_status": "ok" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "wall_time_seconds": wall,
        "inp": str(inp_path.relative_to(out_dir)),
        "dat": str(dat.relative_to(out_dir)) if dat.exists() else "",
        "frd": str(frd.relative_to(out_dir)) if frd.exists() else "",
        "sta": str(sta.relative_to(out_dir)) if sta.exists() else "",
        "cvg": str(cvg.relative_to(out_dir)) if cvg.exists() else "",
        **parse_dat_metrics(dat),
    }


def _claim_gate(case: OfficialExample, metadata: Row, run: Row) -> list[Row]:
    input_available = metadata["input_available"] == "true"
    run_ok = run["run_status"] in {"ok", "not_run"}
    direct_contactenergy = case.case_id == "contactenergy_c3d8_static_energy"
    law_alignment = case.role == "small_contact_law_alignment"
    official_candidate = case.case_id in {"scheibe2f2f_c3d8_nlgeom_static", "ball_c3d8_dynamic_drop"}
    return [
        {
            "case_id": case.case_id,
            "claim": "official_calculix_example_available",
            "allowed": str(input_available).lower(),
            "reason": "official input was found and parsed",
        },
        {
            "case_id": case.case_id,
            "claim": "paper_external_reference_direct",
            "allowed": str(direct_contactenergy and input_available and run_ok).lower(),
            "reason": "direct paper use is currently limited to the locked contactenergy C3D8 law/energy reference",
        },
        {
            "case_id": case.case_id,
            "claim": "small_law_alignment_reference",
            "allowed": str(law_alignment and input_available and run_ok).lower(),
            "reason": "small C3D8 law examples are allowed for pressure-overclosure and contact-print alignment only",
        },
        {
            "case_id": case.case_id,
            "claim": "full_trajectory_or_full_field_equivalence",
            "allowed": "false",
            "reason": "requires a native SFC solve plus metric-level comparison before this official example can support full equivalence",
        },
        {
            "case_id": case.case_id,
            "claim": "recommended_next_adaptation",
            "allowed": str(official_candidate).lower(),
            "reason": "scheibe2f2f and ball are the best official engineering candidates but need dedicated native SFC comparison paths",
        },
    ]


def _write_plots(out_dir: Path, catalog_rows: list[Row], run_rows: list[Row]) -> list[Row]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = [row["case_id"].replace("_", "\n") for row in catalog_rows]
    direct = [1.0 if row["paper_use"] == "direct_reference_already_locked" else 0.0 for row in catalog_rows]
    law = [1.0 if row["role"] == "small_contact_law_alignment" else 0.0 for row in catalog_rows]
    candidate = [1.0 if "candidate" in row["paper_use"] else 0.0 for row in catalog_rows]
    x = np.arange(len(catalog_rows))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    width = 0.25
    ax.bar(x - width, direct, width, label="direct paper reference")
    ax.bar(x, law, width, label="law alignment")
    ax.bar(x + width, candidate, width, label="adaptation candidate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("classification")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    classification_png = figures / "official_calculix_example_classification.png"
    classification_pdf = figures / "official_calculix_example_classification.pdf"
    fig.savefig(classification_png, dpi=180)
    fig.savefig(classification_pdf)
    plt.close(fig)

    runtimes = [float(row["wall_time_seconds"]) if row["wall_time_seconds"] not in {"", None} else 0.0 for row in run_rows]
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    ax.bar(x, runtimes, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("ccx wall time (s)")
    ax.set_title("Official CalculiX example smoke run times")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    runtime_png = figures / "official_calculix_example_runtime.png"
    runtime_pdf = figures / "official_calculix_example_runtime.pdf"
    fig.savefig(runtime_png, dpi=180)
    fig.savefig(runtime_pdf)
    plt.close(fig)

    return [
        {
            "plot": "official_calculix_example_classification",
            "png": str(classification_png.relative_to(out_dir)),
            "pdf": str(classification_pdf.relative_to(out_dir)),
            "description": "Direct/law/candidate classification for selected official CalculiX contact examples",
        },
        {
            "plot": "official_calculix_example_runtime",
            "png": str(runtime_png.relative_to(out_dir)),
            "pdf": str(runtime_pdf.relative_to(out_dir)),
            "description": "CalculiX smoke-run wall times where examples were executed",
        },
    ]


def _write_summary(out_dir: Path, catalog_rows: list[Row], gates: list[Row], run_rows: list[Row]) -> None:
    lines = [
        "# Official CalculiX Contact Example Line",
        "",
        "This package records which official CalculiX examples are suitable for",
        "the paper-facing SFC validation path. It is intentionally conservative:",
        "an official example is not allowed to support full SFC trajectory or",
        "field equivalence until a native SFC solve and metric-level comparison",
        "exist for the same model.",
        "",
        "## Selected Examples",
        "",
        "| Case | File | Role | Elements | Contact | Step | Paper use |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in catalog_rows:
        lines.append(
            f"| {row['case_id']} | `{row['file_name']}` | {row['role']} | "
            f"{row['element_types']} | {row['contact_pair_types']} | {row['step_types']} | {row['paper_use']} |"
        )
    lines.extend(["", "## Claim Gates", "", "| Case | Claim | Allowed | Reason |", "| --- | --- | --- | --- |"])
    for gate in gates:
        lines.append(f"| {gate['case_id']} | {gate['claim']} | {gate['allowed']} | {gate['reason']} |")
    lines.extend(["", "## Run Status", "", "| Case | Status | DAT | FRD | Wall time |", "| --- | --- | --- | --- | ---: |"])
    for row in run_rows:
        lines.append(f"| {row['case_id']} | {row['run_status']} | {row['dat_nonempty']} | {bool(row['frd'])} | {row['wall_time_seconds']} |")
    lines.extend(
        [
            "",
            "## Recommended Use",
            "",
            "1. `contactenergy.inp` remains the direct C3D8 static contact-law and contact-energy reference.",
            "2. `contact1/contact3/contact6` should be used for small pressure-overclosure law alignment diagnostics.",
            "3. `scheibe2f2f.inp.gz` is the best official nonlinear static C3D8 surface-to-surface candidate, but it needs FRD field extraction and a native SFC comparison before paper claims.",
            "4. `ball.inp.gz` is the best official dynamic drop candidate, but its shell-floor/node-to-surface formulation must be matched or explicitly filtered before trajectory claims.",
        ]
    )
    (out_dir / "official_calculix_contact_examples_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_official_examples(*, out_dir: Path, examples_dir: Path | None, wsl_examples_dir: str, skip_run: bool, quick: bool, timeout: int) -> dict[str, list[Row]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = out_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    catalog_rows: list[Row] = []
    run_rows: list[Row] = []
    gate_rows: list[Row] = []
    command_rows = [
        {
            "command": "python validation/run_calculix_official_contact_examples.py"
            + (" --quick" if quick else "")
            + (" --skip-run" if skip_run else "")
            + f" --out-dir {out_dir}",
            "external_solver": "CalculiX",
            "external_solver_version": _calculix_version(),
            "notes": "validation-only official example catalog and smoke runner",
        }
    ]

    for case in OFFICIAL_EXAMPLES:
        if quick and not case.run_in_quick:
            continue
        try:
            text, source = _read_example(case.file_name, examples_dir, wsl_examples_dir)
            parsed = parse_official_input(text)
            input_available = "true"
            copied_name = case.file_name[:-3] if case.file_name.endswith(".gz") else case.file_name
            (input_dir / copied_name).write_text(text, encoding="utf-8")
        except FileNotFoundError:
            text = ""
            source = ""
            parsed = {
                "node_count": 0,
                "element_count": 0,
                "element_types": "",
                "contact_pair_types": "",
                "step_types": "",
                "surface_count": 0,
                "material_count": 0,
                "pressure_overclosure_laws": "",
                "pressure_overclosure_values": "",
                "has_nlgeom": "false",
                "has_dynamic": "false",
                "has_static": "false",
                "has_friction": "false",
                "has_plastic": "false",
                "has_hyperelastic": "false",
                "has_shell_or_beam": "false",
                "has_supported_solid": "false",
                "only_supported_solids": "false",
                "adaptation_level": "missing_input",
            }
            input_available = "false"

        catalog = {
            "case_id": case.case_id,
            "file_name": case.file_name,
            "source": source,
            "input_available": input_available,
            "role": case.role,
            "expected_scope": case.expected_scope,
            "paper_use": case.paper_use,
            **parsed,
        }
        catalog_rows.append(catalog)
        if input_available == "true":
            run_row = run_calculix_case(case=case, text=text, out_dir=out_dir, skip_run=skip_run, timeout=timeout)
        else:
            run_row = {
                "case_id": case.case_id,
                "job_name": _job_name(case.file_name),
                "run_status": "missing_input",
                "return_code": "",
                "wall_time_seconds": "",
                "inp": "",
                "dat": "",
                "frd": "",
                "sta": "",
                "cvg": "",
                **parse_dat_metrics(out_dir / "missing.dat"),
            }
        run_rows.append(run_row)
        gate_rows.extend(_claim_gate(case, catalog, run_row))

    catalog_fields = [
        "case_id",
        "file_name",
        "source",
        "input_available",
        "role",
        "expected_scope",
        "paper_use",
        "node_count",
        "element_count",
        "element_types",
        "contact_pair_types",
        "step_types",
        "surface_count",
        "material_count",
        "pressure_overclosure_laws",
        "pressure_overclosure_values",
        "has_nlgeom",
        "has_dynamic",
        "has_static",
        "has_friction",
        "has_plastic",
        "has_hyperelastic",
        "has_shell_or_beam",
        "has_supported_solid",
        "only_supported_solids",
        "adaptation_level",
    ]
    run_fields = [
        "case_id",
        "job_name",
        "run_status",
        "return_code",
        "wall_time_seconds",
        "inp",
        "dat",
        "frd",
        "sta",
        "cvg",
        "dat_available",
        "dat_nonempty",
        "dat_time_block_count",
        "max_displacement_norm",
        "max_force_norm",
        "max_von_mises_from_dat",
        "total_contact_spring_energy",
        "raw_cdis_rows",
        "raw_cstr_rows",
        "raw_cels_rows",
    ]
    _write_csv(out_dir / "official_calculix_example_catalog.csv", catalog_fields, catalog_rows)
    _write_csv(out_dir / "official_calculix_example_runs.csv", run_fields, run_rows)
    _write_csv(out_dir / "official_calculix_example_claim_gates.csv", ["case_id", "claim", "allowed", "reason"], gate_rows)
    _write_csv(out_dir / "official_calculix_example_commands.csv", ["command", "external_solver", "external_solver_version", "notes"], command_rows)
    plot_rows = _write_plots(out_dir, catalog_rows, run_rows)
    _write_csv(out_dir / "official_calculix_example_plots.csv", ["plot", "png", "pdf", "description"], plot_rows)
    _write_summary(out_dir, catalog_rows, gate_rows, run_rows)
    return {
        "catalog": catalog_rows,
        "runs": run_rows,
        "claim_gates": gate_rows,
        "commands": command_rows,
        "plots": plot_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run the selected official examples; all current selected cases are small.")
    parser.add_argument("--skip-run", action="store_true", help="Parse/copy official inputs without executing CalculiX.")
    parser.add_argument("--examples-dir", type=Path, default=None, help="Optional local directory containing the selected CalculiX example files.")
    parser.add_argument("--wsl-examples-dir", default=DEFAULT_WSL_EXAMPLES_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "calculix_official_contact_examples")
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_official_examples(
        out_dir=Path(args.out_dir),
        examples_dir=args.examples_dir,
        wsl_examples_dir=str(args.wsl_examples_dir),
        skip_run=bool(args.skip_run),
        quick=bool(args.quick),
        timeout=int(args.timeout),
    )
    print(f"Wrote official CalculiX contact example outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
