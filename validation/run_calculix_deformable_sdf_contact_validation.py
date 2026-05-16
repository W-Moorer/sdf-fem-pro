"""CalculiX deformable-contact state replay with SFC dynamic SDF.

This runner is intentionally scoped.  CalculiX solves a small two-deformable
block contact problem.  SFC then replays the final CalculiX-deformed geometry
with current-surface dynamic SDF queries from the upper block's slave surface to
the lower block's deformed master surface.

The evidence checks the dynamic-SDF geometry layer on an external deformed
contact state.  It is not a claim that SFC and CalculiX solve identical
deformable-deformable nonlinear contact trajectories.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.sdf.dynamic_surface_sdf import dynamic_surface_sdf  # noqa: E402
from validation.run_calculix_drop_impact_comparison import (  # noqa: E402
    _boundary_face_data,
    _calculix_version,
    _parse_calculix_dat_contact_totals,
    _parse_calculix_dat_displacements,
    _wsl_path,
    calculix_available,
)
from validation.run_geometric_nonlinear_contact_validation import _parse_calculix_dat_contact_elements  # noqa: E402
from validation.run_phase3_validation import structured_tet_block  # noqa: E402

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class TwoBlockModel:
    """Small two-block C3D4 contact model shared by CalculiX and SFC replay."""

    case_name: str
    node_ids: np.ndarray
    nodes: np.ndarray
    lower_elements: np.ndarray
    upper_elements: np.ndarray
    lower_top_faces: np.ndarray
    upper_bottom_faces: np.ndarray
    lower_top_refs: list[tuple[int, str]]
    upper_bottom_refs: list[tuple[int, str]]
    lower_bottom_node_ids: np.ndarray
    upper_top_node_ids: np.ndarray
    lower_node_count: int
    lower_element_count: int
    gap: float
    approach: float
    E: float
    nu: float
    contact_stiffness: float


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_id_list(ids: np.ndarray, *, per_line: int = 12) -> list[str]:
    values = [str(int(value)) for value in ids]
    return [", ".join(values[i : i + per_line]) for i in range(0, len(values), per_line)]


def _surface_refs_at_z(
    X: np.ndarray,
    surface_faces: np.ndarray,
    refs: list[tuple[int, str]],
    *,
    z: float,
    element_offset: int = 0,
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    faces: list[np.ndarray] = []
    kept_refs: list[tuple[int, str]] = []
    for face, ref in zip(surface_faces, refs, strict=True):
        if np.all(np.isclose(X[np.asarray(face, dtype=np.int64), 2], float(z))):
            faces.append(np.asarray(face, dtype=np.int64))
            kept_refs.append((int(ref[0]) + int(element_offset), str(ref[1])))
    return np.asarray(faces, dtype=np.int64), kept_refs


def _orient_faces_to_positive_z(X: np.ndarray, faces: np.ndarray) -> np.ndarray:
    oriented = np.asarray(faces, dtype=np.int64).copy()
    for i, face in enumerate(oriented):
        tri = X[face]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if float(normal[2]) < 0.0:
            oriented[i] = face[[0, 2, 1]]
    return oriented


def build_two_block_model(*, resolution: int, approach: float, gap: float = 0.035) -> TwoBlockModel:
    """Build a small stacked two-block CalculiX model."""

    lower = structured_tet_block(int(resolution), int(resolution), max(1, int(resolution) // 2), size=(1.0, 1.0, 0.25))
    upper = structured_tet_block(int(resolution), int(resolution), max(1, int(resolution) // 2), size=(1.0, 1.0, 0.25))
    lower_x = np.asarray(lower.X, dtype=float)
    upper_x = np.asarray(upper.X, dtype=float).copy()
    upper_x[:, 2] += float(lower_x[:, 2].max()) + float(gap)

    n_lower = lower_x.shape[0]
    e_lower = lower.elements.shape[0]
    nodes = np.vstack([lower_x, upper_x])
    node_ids = np.arange(1, nodes.shape[0] + 1, dtype=np.int64)
    lower_elements = np.asarray(lower.elements, dtype=np.int64)
    upper_elements = np.asarray(upper.elements, dtype=np.int64) + n_lower

    lower_faces, lower_refs, _, _ = _boundary_face_data(lower_x, lower_elements)
    upper_faces_local, upper_refs_local, _, _ = _boundary_face_data(upper_x, np.asarray(upper.elements, dtype=np.int64))
    lower_top_faces, lower_top_refs = _surface_refs_at_z(
        lower_x,
        lower_faces,
        lower_refs,
        z=float(lower_x[:, 2].max()),
        element_offset=0,
    )
    upper_bottom_faces_local, upper_bottom_refs = _surface_refs_at_z(
        upper_x,
        upper_faces_local,
        upper_refs_local,
        z=float(upper_x[:, 2].min()),
        element_offset=e_lower,
    )
    upper_bottom_faces = upper_bottom_faces_local + n_lower
    lower_top_faces = _orient_faces_to_positive_z(nodes, lower_top_faces)

    lower_bottom = np.flatnonzero(np.isclose(lower_x[:, 2], float(lower_x[:, 2].min()))) + 1
    upper_top = np.flatnonzero(np.isclose(upper_x[:, 2], float(upper_x[:, 2].max()))) + n_lower + 1
    return TwoBlockModel(
        case_name=f"two_block_deformable_r{int(resolution)}_a{float(approach):.4f}".replace(".", "p"),
        node_ids=node_ids,
        nodes=nodes,
        lower_elements=lower_elements,
        upper_elements=upper_elements,
        lower_top_faces=lower_top_faces,
        upper_bottom_faces=upper_bottom_faces,
        lower_top_refs=lower_top_refs,
        upper_bottom_refs=upper_bottom_refs,
        lower_bottom_node_ids=np.asarray(lower_bottom, dtype=np.int64),
        upper_top_node_ids=np.asarray(upper_top, dtype=np.int64),
        lower_node_count=n_lower,
        lower_element_count=e_lower,
        gap=float(gap),
        approach=float(approach),
        E=1.0e5,
        nu=0.3,
        contact_stiffness=6.0e4,
    )


def write_calculix_input(model: TwoBlockModel, path: Path) -> None:
    """Write a deterministic two-deformable-block CalculiX contact input."""

    lines: list[str] = [
        "** Generated by validation/run_calculix_deformable_sdf_contact_validation.py",
        "** External reference: CalculiX static two-deformable-block contact.",
        "*NODE, NSET=Nall",
    ]
    for node_id, coord in zip(model.node_ids, model.nodes, strict=True):
        lines.append(f"{int(node_id)}, {coord[0]:.12e}, {coord[1]:.12e}, {coord[2]:.12e}")

    lines.append("*ELEMENT, TYPE=C3D4, ELSET=ELOWER")
    for element_id, tet in enumerate(model.lower_elements, start=1):
        ids = [int(model.node_ids[int(i)]) for i in tet]
        lines.append(f"{element_id}, {ids[0]}, {ids[1]}, {ids[2]}, {ids[3]}")
    lines.append("*ELEMENT, TYPE=C3D4, ELSET=EUPPER")
    for element_id, tet in enumerate(model.upper_elements, start=model.lower_element_count + 1):
        ids = [int(model.node_ids[int(i)]) for i in tet]
        lines.append(f"{element_id}, {ids[0]}, {ids[1]}, {ids[2]}, {ids[3]}")

    lines.append("*ELSET, ELSET=EALL")
    lines.extend(_format_id_list(np.arange(1, model.lower_element_count + model.upper_elements.shape[0] + 1)))
    lines.append("*NSET, NSET=NLOWERBOTTOM")
    lines.extend(_format_id_list(model.lower_bottom_node_ids))
    lines.append("*NSET, NSET=NUPPERTOP")
    lines.extend(_format_id_list(model.upper_top_node_ids))
    lines.append("*SURFACE, NAME=LOWER_TOP, TYPE=ELEMENT")
    for element_id, label in model.lower_top_refs:
        lines.append(f"{int(element_id)}, {label}")
    lines.append("*SURFACE, NAME=UPPER_BOTTOM, TYPE=ELEMENT")
    for element_id, label in model.upper_bottom_refs:
        lines.append(f"{int(element_id)}, {label}")

    lines.extend(
        [
            "*BOUNDARY",
            "** Lateral DOFs are fixed for this scoped normal-contact replay.",
            "** This prevents an underconstrained r1 frictionless contact mesh",
            "** from turning the geometry validation into a lateral shear mode.",
            "Nall, 1, 2, 0.0",
            "NLOWERBOTTOM, 3, 3, 0.0",
            "*MATERIAL, NAME=MAT",
            "*ELASTIC",
            f"{model.E:.12g}, {model.nu:.12g}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT",
            "*SURFACE INTERACTION, NAME=CONTACT",
            "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
            f"{model.contact_stiffness:.12g}",
            "*CONTACT PAIR, INTERACTION=CONTACT, TYPE=SURFACE TO SURFACE",
            "UPPER_BOTTOM, LOWER_TOP",
            "*STEP, NLGEOM, INC=200",
            "*STATIC",
            "0.1, 1.0, 1e-5, 0.1",
            "*BOUNDARY",
            f"NUPPERTOP, 3, 3, {-model.approach:.12g}",
            "*NODE PRINT, NSET=Nall, FREQUENCY=1",
            "U",
            "*CONTACT PRINT, FREQUENCY=1",
            "CDIS",
            "CSTR",
            "CELS",
            "*CONTACT PRINT, FREQUENCY=1, TOTALS=ONLY",
            "CNUM",
            "*END STEP",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_calculix(model: TwoBlockModel, out_dir: Path, *, timeout: int) -> tuple[Path | None, Row]:
    run_dir = out_dir / "calculix_runs" / model.case_name
    run_dir.mkdir(parents=True, exist_ok=True)
    inp = run_dir / f"{model.case_name}.inp"
    write_calculix_input(model, inp)
    command = f"cd {_wsl_path(run_dir)} && ccx {model.case_name}"
    if not calculix_available():
        return None, {
            "case": model.case_name,
            "external_solver": "CalculiX",
            "external_solver_version": "not_available",
            "command": f"wsl --exec bash -lc \"{command}\"",
            "input_file": str(inp.relative_to(out_dir)),
            "dat_file": "",
            "stdout_log": "",
            "stderr_log": "",
            "return_code": "not_available",
            "completed": "false",
        }
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=int(timeout),
    )
    stdout_log = run_dir / "calculix_stdout.log"
    stderr_log = run_dir / "calculix_stderr.log"
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")
    dat = run_dir / f"{model.case_name}.dat"
    return dat if dat.exists() else None, {
        "case": model.case_name,
        "external_solver": "CalculiX",
        "external_solver_version": _calculix_version(),
        "command": f"wsl --exec bash -lc \"{command}\"",
        "input_file": str(inp.relative_to(out_dir)),
        "dat_file": str(dat.relative_to(out_dir)) if dat.exists() else "",
        "stdout_log": str(stdout_log.relative_to(out_dir)),
        "stderr_log": str(stderr_log.relative_to(out_dir)),
        "return_code": proc.returncode,
        "completed": str(proc.returncode == 0 and dat.exists()).lower(),
    }


def _slave_quadrature_points(x_current: np.ndarray, faces: np.ndarray) -> np.ndarray:
    shapes = (
        np.asarray([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]),
        np.asarray([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]),
        np.asarray([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]),
    )
    points: list[np.ndarray] = []
    for face in faces:
        tri = x_current[np.asarray(face, dtype=np.int64)]
        for shape in shapes:
            points.append(shape @ tri)
    return np.asarray(points, dtype=float)


def _sfc_replay_on_calculix_state(model: TwoBlockModel, displacement: np.ndarray) -> Row:
    current = model.nodes + displacement
    candidates = np.arange(model.lower_top_faces.shape[0], dtype=np.int64)
    points = _slave_quadrature_points(current, model.upper_bottom_faces)
    gaps = np.asarray(
        [
            dynamic_surface_sdf(point, current, model.lower_top_faces, candidates).g
            for point in points
        ],
        dtype=float,
    )
    return {
        "sfc_query_count": int(gaps.size),
        "sfc_gap_min": float(np.min(gaps)) if gaps.size else 0.0,
        "sfc_gap_mean": float(np.mean(gaps)) if gaps.size else 0.0,
        "sfc_active_count": int(np.count_nonzero(gaps < 0.0)),
        "sfc_max_penetration": float(np.max(np.maximum(-gaps, 0.0))) if gaps.size else 0.0,
    }


def _final_contact_metrics(dat: Path) -> tuple[float | None, Row]:
    per_contact = _parse_calculix_dat_contact_elements(dat)
    totals = _parse_calculix_dat_contact_totals(dat)
    times = sorted(set(per_contact) | set(totals))
    if not times:
        return None, {
            "calculix_cdis_min": "",
            "calculix_cdis_mean": "",
            "calculix_contact_row_count": 0,
            "calculix_cnum": "",
            "calculix_contact_energy": "",
        }
    final_time = float(times[-1])
    records = per_contact.get(final_time, {})
    cdis = [
        float(row["calculix_clearance_normal"])
        for row in records.values()
        if "calculix_clearance_normal" in row
    ]
    total = totals.get(final_time, {})
    return final_time, {
        "calculix_cdis_min": "" if not cdis else float(np.min(cdis)),
        "calculix_cdis_mean": "" if not cdis else float(np.mean(cdis)),
        "calculix_contact_row_count": int(len(cdis)),
        "calculix_cnum": total.get("calculix_contact_count", ""),
        "calculix_contact_energy": total.get("calculix_contact_energy", ""),
    }


def run_validation(
    out_dir: Path,
    *,
    quick: bool = False,
    resolution: int = 1,
    timeout: int = 180,
) -> dict[str, Path]:
    """Run the scoped CalculiX/SFC deformable-SDF contact replay validation."""

    out_dir.mkdir(parents=True, exist_ok=True)
    approaches = [0.02, 0.06] if quick else [0.02, 0.04, 0.06]
    comparison_rows: list[Row] = []
    command_rows: list[Row] = []
    for approach in approaches:
        model = build_two_block_model(resolution=int(resolution), approach=float(approach))
        dat, command = _run_calculix(model, out_dir, timeout=timeout)
        command_rows.append(command)
        if dat is None or command["completed"] != "true":
            comparison_rows.append(
                {
                    "case": model.case_name,
                    "resolution": int(resolution),
                    "approach": float(approach),
                    "calculix_completed": command["completed"],
                    "final_time": "",
                    "calculix_cdis_min": "",
                    "calculix_cdis_mean": "",
                    "calculix_contact_row_count": "",
                    "calculix_cnum": "",
                    "calculix_contact_energy": "",
                    "sfc_query_count": "",
                    "sfc_gap_min": "",
                    "sfc_gap_mean": "",
                    "sfc_active_count": "",
                    "sfc_max_penetration": "",
                    "contact_sign_agreement": "false",
                    "gap_min_abs_difference": "",
                    "status": "not_available",
                    "details": "CalculiX did not complete or did not produce a dat file.",
                }
            )
            continue
        displacements = _parse_calculix_dat_displacements(dat, model.node_ids)
        if not displacements:
            raise RuntimeError(f"No displacement blocks found in {dat}")
        final_disp_time = max(displacements)
        final_time, contact = _final_contact_metrics(dat)
        replay = _sfc_replay_on_calculix_state(model, displacements[final_disp_time])
        calc_min = contact["calculix_cdis_min"]
        sfc_min = replay["sfc_gap_min"]
        calc_active = calc_min != "" and float(calc_min) < 0.0
        sfc_active = float(sfc_min) < 0.0
        sign_agreement = bool(calc_active == sfc_active)
        gap_diff: float | str = "" if calc_min == "" else abs(float(calc_min) - float(sfc_min))
        status = "passed_replay_gate" if sign_agreement and replay["sfc_query_count"] > 0 else "check"
        comparison_rows.append(
            {
                "case": model.case_name,
                "resolution": int(resolution),
                "approach": float(approach),
                "calculix_completed": command["completed"],
                "final_time": final_time if final_time is not None else final_disp_time,
                **contact,
                **replay,
                "contact_sign_agreement": str(sign_agreement).lower(),
                "gap_min_abs_difference": gap_diff,
                "status": status,
            "details": (
                "CalculiX solves two-deformable-block contact; SFC replays the final deformed geometry with "
                "current lower-block dynamic SDF. The reference fixes lateral DOFs so this is a scoped normal "
                "compression check of external deformed-surface SDF gap evaluation."
            ),
        }
    )

    claims = _claim_rows(comparison_rows)
    outputs = {
        "comparison": out_dir / "calculix_deformable_sdf_contact_comparison.csv",
        "commands": out_dir / "calculix_deformable_sdf_contact_commands.csv",
        "claims": out_dir / "calculix_deformable_sdf_contact_claims.csv",
        "summary": out_dir / "calculix_deformable_sdf_contact_summary.md",
    }
    _write_csv(outputs["comparison"], _comparison_fields(), comparison_rows)
    _write_csv(outputs["commands"], _command_fields(), command_rows)
    _write_csv(outputs["claims"], _claim_fields(), claims)
    _write_markdown(outputs["summary"], comparison_rows, claims, quick=quick, resolution=resolution)
    return outputs


def _comparison_fields() -> list[str]:
    return [
        "case",
        "resolution",
        "approach",
        "calculix_completed",
        "final_time",
        "calculix_cdis_min",
        "calculix_cdis_mean",
        "calculix_contact_row_count",
        "calculix_cnum",
        "calculix_contact_energy",
        "sfc_query_count",
        "sfc_gap_min",
        "sfc_gap_mean",
        "sfc_active_count",
        "sfc_max_penetration",
        "contact_sign_agreement",
        "gap_min_abs_difference",
        "status",
        "details",
    ]


def _command_fields() -> list[str]:
    return [
        "case",
        "external_solver",
        "external_solver_version",
        "command",
        "input_file",
        "dat_file",
        "stdout_log",
        "stderr_log",
        "return_code",
        "completed",
    ]


def _claim_fields() -> list[str]:
    return ["claim_id", "claim_text", "evidence_csv", "evidence_field", "gate_value", "claim_status", "details"]


def _claim_rows(rows: list[Row]) -> list[Row]:
    completed = [row for row in rows if row["calculix_completed"] == "true"]
    sign_ok = bool(completed) and all(row["contact_sign_agreement"] == "true" for row in completed)
    active_replay = any(row.get("sfc_active_count") not in ("", None) and int(row["sfc_active_count"]) > 0 for row in completed)
    active_calculix = any(
        row.get("calculix_contact_row_count") not in ("", None, "") and int(row["calculix_contact_row_count"]) > 0
        for row in completed
    )
    max_gap_diff = max(
        (float(row["gap_min_abs_difference"]) for row in completed if row.get("gap_min_abs_difference") not in ("", None)),
        default=0.0,
    )
    gap_scale_ok = bool(completed) and max_gap_diff <= 1.0e-6
    return [
        {
            "claim_id": "calculix_two_deformable_contact_reference_completed",
            "claim_text": "CalculiX completed the scoped two-deformable-block contact reference cases.",
            "evidence_csv": "calculix_deformable_sdf_contact_comparison.csv",
            "evidence_field": "calculix_completed",
            "gate_value": f"{len(completed)}/{len(rows)}",
            "claim_status": "supported" if len(completed) == len(rows) and rows else "not_supported",
            "details": "external CalculiX execution gate",
        },
        {
            "claim_id": "sfc_dynamic_sdf_replay_matches_calculix_contact_sign",
            "claim_text": "SFC current-surface dynamic SDF replay agrees with CalculiX contact activation sign on completed two-block states.",
            "evidence_csv": "calculix_deformable_sdf_contact_comparison.csv",
            "evidence_field": "contact_sign_agreement",
            "gate_value": str(sign_ok and active_replay and active_calculix).lower(),
            "claim_status": "supported" if sign_ok and active_replay and active_calculix else "not_supported",
            "details": "CDIS and SFC gap magnitudes are reported separately; this gate checks contact/no-contact sign agreement",
        },
        {
            "claim_id": "sfc_dynamic_sdf_replay_gap_scale_reported",
            "claim_text": "In the scoped lateral-constrained normal-compression reference, SFC minimum dynamic-SDF gap matches CalculiX minimum CDIS.",
            "evidence_csv": "calculix_deformable_sdf_contact_comparison.csv",
            "evidence_field": "gap_min_abs_difference",
            "gate_value": max_gap_diff,
            "claim_status": "supported" if gap_scale_ok else "not_supported",
            "details": "strict only for this controlled normal-compression replay; it is not a full trajectory equivalence claim",
        },
    ]


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.6e}"
    except (TypeError, ValueError):
        return str(value)


def _write_markdown(path: Path, rows: list[Row], claims: list[Row], *, quick: bool, resolution: int) -> None:
    command = (
        "python validation/run_calculix_deformable_sdf_contact_validation.py "
        f"--resolution {int(resolution)} --out-dir {path.parent.as_posix()}"
    )
    if quick:
        command += " --quick"
    lines = [
        "# CalculiX Deformable SDF Contact Validation",
        "",
        "This validation uses CalculiX as an external reference for a small two-deformable-block contact problem, then replays the final deformed CalculiX geometry with SFC current-surface dynamic SDF queries.",
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Claim Gates",
        "",
        "| Claim | Status | Evidence | Gate value |",
        "| --- | --- | --- | --- |",
    ]
    for claim in claims:
        lines.append(f"| {claim['claim_id']} | {claim['claim_status']} | `{claim['evidence_csv']}::{claim['evidence_field']}` | `{claim['gate_value']}` |")
    lines.extend(
        [
            "",
            "## Comparison Rows",
            "",
            "| Approach | CalculiX completed | CalculiX min CDIS | SFC min gap | SFC active | Sign agreement | Status |",
            "| ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {_fmt(row['approach'])} | {row['calculix_completed']} | {_fmt(row['calculix_cdis_min'])} | {_fmt(row['sfc_gap_min'])} | {row['sfc_active_count']} | {row['contact_sign_agreement']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This verifies the SFC dynamic-SDF query on a CalculiX-deformed master surface state.",
            "- Lateral DOFs are fixed in the external model so the comparison isolates normal gap evaluation rather than lateral contact-mode drift.",
            "- It does not prove full native SFC two-body trajectory equivalence to CalculiX.",
            "- In this scoped normal-compression replay, the minimum CalculiX `CDIS` and SFC dynamic-SDF minimum gap are expected to agree to parser/output precision.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "calculix_deformable_sdf_contact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(args.out_dir, quick=bool(args.quick), resolution=int(args.resolution), timeout=int(args.timeout))
    print("CalculiX deformable SDF contact validation complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
