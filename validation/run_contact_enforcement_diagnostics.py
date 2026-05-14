"""Diagnose contact-enforcement differences against the CalculiX setup.

This runner does not change the core contact method.  It decomposes the
remaining block-plane differences into inspectable pieces:

- active-set definitions;
- area weighting;
- pressure-overclosure law;
- contact tangent sign/finite-difference consistency;
- CalculiX RF/CONTACT PRINT output definitions, when ccx is run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_calculix_drop_impact_comparison import (  # noqa: E402
    TRIANGLE_QUADRATURE_BARYCENTRIC,
    TRIANGLE_QUADRATURE_WEIGHTS,
    _calculix_aligned_plane_contact,
    _contact_plane_z,
    _parse_calculix_dat_contact_totals,
    _parse_calculix_dat_displacements,
    _smooth_overclosure_response,
    _surface_face_geometry,
    _surface_face_gaps_to_contact_plane,
    build_drop_model,
    run_calculix,
)

Row = dict[str, Any]


def run_diagnostics(out_dir: Path, *, quick: bool = False, run_external: bool = False) -> dict[str, Path]:
    """Run contact-enforcement diagnostics and write CSV/Markdown."""

    out_dir.mkdir(parents=True, exist_ok=True)
    model = _diagnostic_model(quick=quick)
    x_current = _penetrating_state(model)
    active_rows = active_set_rows(model, x_current)
    area_rows = area_weighting_rows(model, x_current)
    law_rows = pressure_overclosure_rows(model)
    tangent_rows = tangent_fd_rows(model, x_current)
    output_rows = calculix_output_definition_rows(model, out_dir, run_external=run_external)
    claim_rows = claim_rows_from(active_rows, area_rows, law_rows, tangent_rows, output_rows)
    outputs = {
        "active_set": out_dir / "contact_enforcement_active_set.csv",
        "area": out_dir / "contact_enforcement_area_weighting.csv",
        "pressure_law": out_dir / "contact_enforcement_pressure_law.csv",
        "tangent": out_dir / "contact_enforcement_tangent_fd.csv",
        "calculix_outputs": out_dir / "contact_enforcement_calculix_outputs.csv",
        "claims": out_dir / "contact_enforcement_claims.csv",
        "summary": out_dir / "contact_enforcement_summary.md",
    }
    _write_csv(outputs["active_set"], active_rows)
    _write_csv(outputs["area"], area_rows)
    _write_csv(outputs["pressure_law"], law_rows)
    _write_csv(outputs["tangent"], tangent_rows)
    _write_csv(outputs["calculix_outputs"], output_rows)
    _write_csv(outputs["claims"], claim_rows)
    _write_markdown(outputs["summary"], claim_rows, active_rows, area_rows, law_rows, tangent_rows, output_rows, quick=quick, run_external=run_external)
    return outputs


def _diagnostic_model(*, quick: bool) -> Any:
    return build_drop_model(
        quick=quick,
        case="block_drop",
        resolution=1,
        duration=0.12 if quick else 0.14,
        dt=0.002,
        output_frequency=1,
        initial_velocity_z=0.0,
        gravity=9.81,
        contact_stiffness_override=2.0e4,
        hht_alpha=-0.05,
    )


def _penetrating_state(model: Any) -> np.ndarray:
    x_current = model.nodes.copy()
    initial_gap = float(np.min(_surface_face_gaps_to_contact_plane(model, x_current)))
    x_current[:, 2] -= initial_gap + 0.004
    return x_current


def active_set_rows(model: Any, x_current: np.ndarray) -> list[Row]:
    face_gaps = _surface_face_gaps_to_contact_plane(model, x_current)
    face_pen = np.maximum(-face_gaps, 0.0)
    q = _quadrature_table(model, x_current)
    node_gaps = x_current[model.surface_indices, 2] - _contact_plane_z(model)
    node_pen = np.maximum(-node_gaps, 0.0)
    _, _, q_min_gap, q_active_count, q_max_penetration, _ = _calculix_aligned_plane_contact(model, x_current)
    return [
        {
            "definition": "surface_node_gap",
            "sample_count": int(model.surface_indices.size),
            "active_count": int(np.count_nonzero(node_pen > 0.0)),
            "min_gap": float(np.min(node_gaps)),
            "max_penetration": float(np.max(node_pen)),
            "details": "node-based active set; diagnostic only and not CalculiX surface contact",
        },
        {
            "definition": "face_centroid_gap",
            "sample_count": int(face_gaps.size),
            "active_count": int(np.count_nonzero(face_pen > 0.0)),
            "min_gap": float(np.min(face_gaps)),
            "max_penetration": float(np.max(face_pen)),
            "details": "one centroid per boundary triangle",
        },
        {
            "definition": "three_point_face_quadrature_gap",
            "sample_count": int(q.shape[0]),
            "active_count": int(np.count_nonzero(q["overclosure"] > 0.0)),
            "min_gap": float(np.min(q["gap"])),
            "max_penetration": float(np.max(np.maximum(q["overclosure"], 0.0))),
            "details": "three quadrature points per boundary triangle before smooth pressure law",
        },
        {
            "definition": "smooth_pressure_active_lambda",
            "sample_count": int(q.shape[0]),
            "active_count": int(q_active_count),
            "min_gap": float(q_min_gap),
            "max_penetration": float(q_max_penetration),
            "details": "active count after non-attractive smooth pressure-overclosure clipping",
        },
    ]


def area_weighting_rows(model: Any, x_current: np.ndarray) -> list[Row]:
    q = _quadrature_table(model, x_current)
    rows: list[Row] = []
    force, _, _, active_count, max_pen, energy = _calculix_aligned_plane_contact(model, x_current)
    rows.append(
        {
            "definition": "three_point_quadrature_smooth_area_weighted",
            "sample_count": int(q.shape[0]),
            "active_count": int(active_count),
            "active_area": float(np.sum(q["area"][q["overclosure"] > 0.0])),
            "normal_force": float(np.sum(force[2::3])),
            "contact_energy": float(energy),
            "max_penetration": float(max_pen),
            "details": "current SFC CalculiX-aligned diagnostic law",
        }
    )
    face_centers, face_areas = _surface_face_geometry(x_current, model.surface_faces)
    face_gap = face_centers[:, 2] - _contact_plane_z(model)
    face_pen = np.maximum(-face_gap, 0.0)
    rows.append(
        {
            "definition": "face_centroid_hard_linear_area_weighted",
            "sample_count": int(face_gap.size),
            "active_count": int(np.count_nonzero(face_pen > 0.0)),
            "active_area": float(np.sum(face_areas[face_pen > 0.0])),
            "normal_force": float(np.sum(model.contact_stiffness * face_areas * face_pen)),
            "contact_energy": float(np.sum(0.5 * model.contact_stiffness * face_areas * face_pen * face_pen)),
            "max_penetration": float(np.max(face_pen)),
            "details": "one area-weighted point per boundary triangle; diagnostic variant",
        }
    )
    q_pen = np.maximum(q["overclosure"], 0.0)
    rows.append(
        {
            "definition": "three_point_quadrature_hard_linear_area_weighted",
            "sample_count": int(q.shape[0]),
            "active_count": int(np.count_nonzero(q_pen > 0.0)),
            "active_area": float(np.sum(q["area"][q_pen > 0.0])),
            "normal_force": float(np.sum(model.contact_stiffness * q["area"] * q_pen)),
            "contact_energy": float(np.sum(0.5 * model.contact_stiffness * q["area"] * q_pen * q_pen)),
            "max_penetration": float(np.max(q_pen)),
            "details": "same quadrature as smooth law but without transition smoothing",
        }
    )
    node_gap = x_current[model.surface_indices, 2] - _contact_plane_z(model)
    node_pen = np.maximum(-node_gap, 0.0)
    rows.append(
        {
            "definition": "surface_node_hard_linear_unweighted",
            "sample_count": int(node_pen.size),
            "active_count": int(np.count_nonzero(node_pen > 0.0)),
            "active_area": "",
            "normal_force": float(np.sum(model.contact_stiffness * node_pen)),
            "contact_energy": float(np.sum(0.5 * model.contact_stiffness * node_pen * node_pen)),
            "max_penetration": float(np.max(node_pen)),
            "details": "intentionally unweighted node variant; shows why nodal contact is not comparable",
        }
    )
    return rows


def pressure_overclosure_rows(model: Any) -> list[Row]:
    eps = float(model.contact_smoothing_epsilon)
    stiffness = np.asarray([float(model.contact_stiffness)], dtype=float)
    values = np.asarray([-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 4.0], dtype=float) * eps
    rows: list[Row] = []
    for d in values:
        smooth_lam, smooth_tangent, smooth_energy = _smooth_overclosure_response(
            np.asarray([d], dtype=float),
            stiffness=stiffness,
            epsilon=eps,
        )
        hard_lam = float(model.contact_stiffness * max(d, 0.0))
        hard_tangent = float(model.contact_stiffness if d > 0.0 else 0.0)
        hard_energy = float(0.5 * model.contact_stiffness * max(d, 0.0) ** 2)
        rows.append(
            {
                "overclosure": float(d),
                "epsilon": eps,
                "smooth_lambda": float(smooth_lam[0]),
                "hard_lambda": hard_lam,
                "lambda_difference": float(smooth_lam[0]) - hard_lam,
                "smooth_tangent": float(smooth_tangent[0]),
                "hard_tangent": hard_tangent,
                "smooth_energy": float(smooth_energy[0]),
                "hard_energy": hard_energy,
                "details": "smooth non-attractive transition around zero overclosure",
            }
        )
    return rows


def tangent_fd_rows(model: Any, x_current: np.ndarray) -> list[Row]:
    force, stiffness, *_ = _calculix_aligned_plane_contact(model, x_current)
    direction = np.zeros(force.shape[0], dtype=float)
    direction[2::3] = np.linspace(-0.5, 0.5, x_current.shape[0])
    direction /= max(float(np.linalg.norm(direction)), 1.0e-30)
    h = 1.0e-6
    plus, *_ = _calculix_aligned_plane_contact(model, x_current + h * direction.reshape((-1, 3)))
    minus, *_ = _calculix_aligned_plane_contact(model, x_current - h * direction.reshape((-1, 3)))
    force_fd = (plus - minus) / (2.0 * h)
    residual_tangent_action = stiffness @ direction
    # The returned contact tangent is the residual contribution -df_contact/du.
    error = float(np.linalg.norm(residual_tangent_action + force_fd))
    scale = max(float(np.linalg.norm(force_fd)), 1.0e-30)
    return [
        {
            "case": "contact_tangent_finite_difference",
            "force_derivative_norm": float(np.linalg.norm(force_fd)),
            "residual_tangent_action_norm": float(np.linalg.norm(residual_tangent_action)),
            "abs_error": error,
            "rel_error": error / scale,
            "sign_convention": "K_contact approximates -d(f_contact)/du for residual assembly",
            "status": "passed" if error / scale <= 1.0e-7 else "failed",
        }
    ]


def calculix_output_definition_rows(model: Any, out_dir: Path, *, run_external: bool) -> list[Row]:
    if not run_external:
        return [
            {
                "source": "calculix",
                "status": "skipped",
                "time": "",
                "has_floor_rf": "false",
                "has_contact_energy": "false",
                "has_contact_count": "false",
                "details": "external CalculiX run skipped",
            }
        ]
    dat_path, command_row = run_calculix(model, out_dir, case_name="contact_enforcement_block_drop")
    totals = _parse_calculix_dat_contact_totals(dat_path)
    displacements = _parse_calculix_dat_displacements(dat_path, model.node_ids)
    rows: list[Row] = [
        {
            "source": "calculix_command",
            "status": "completed" if command_row["completed"] == "true" else "incomplete",
            "time": "",
            "has_floor_rf": "",
            "has_contact_energy": "",
            "has_contact_count": "",
            "details": command_row["command"],
        }
    ]
    for time, fields in totals.items():
        rows.append(
            {
                "source": "calculix_contact_print",
                "status": "parsed",
                "time": float(time),
                "has_floor_rf": str("normal_force_from_rf" in fields).lower(),
                "has_contact_energy": str("calculix_contact_energy" in fields).lower(),
                "has_contact_count": str("calculix_contact_count" in fields).lower(),
                "floor_rf_z": fields.get("floor_rf_z", ""),
                "normal_force_from_rf": fields.get("normal_force_from_rf", ""),
                "contact_energy": fields.get("calculix_contact_energy", ""),
                "contact_count": fields.get("calculix_contact_count", ""),
                "details": "RF is fixed-floor total reaction; CELS/CNUM are CONTACT PRINT totals, not SFC quadrature counts",
            }
        )
    rows.append(
        {
            "source": "calculix_displacement_blocks",
            "status": "parsed",
            "time": "",
            "has_floor_rf": "",
            "has_contact_energy": "",
            "has_contact_count": "",
            "displacement_block_count": len(displacements),
            "details": "node displacement blocks parsed for geometry-based postprocessing",
        }
    )
    return rows


def _quadrature_table(model: Any, x_current: np.ndarray) -> np.ndarray:
    face_nodes = x_current[model.surface_faces]
    _, areas = _surface_face_geometry(x_current, model.surface_faces)
    q_points = np.einsum("qn,fnc->fqc", TRIANGLE_QUADRATURE_BARYCENTRIC, face_nodes)
    gaps = q_points[:, :, 2] - _contact_plane_z(model)
    q_areas = areas[:, None] * TRIANGLE_QUADRATURE_WEIGHTS[None, :]
    dtype = [
        ("face_id", np.int64),
        ("q_id", np.int64),
        ("gap", float),
        ("overclosure", float),
        ("area", float),
    ]
    table = np.zeros(gaps.size, dtype=dtype)
    index = 0
    for face_id in range(gaps.shape[0]):
        for q_id in range(gaps.shape[1]):
            table[index] = (face_id, q_id, float(gaps[face_id, q_id]), float(-gaps[face_id, q_id]), float(q_areas[face_id, q_id]))
            index += 1
    return table


def claim_rows_from(active_rows: list[Row], area_rows: list[Row], law_rows: list[Row], tangent_rows: list[Row], output_rows: list[Row]) -> list[Row]:
    output_available = any(row.get("has_floor_rf") == "true" or row.get("has_contact_count") == "true" for row in output_rows)
    return [
        {
            "claim": "active_set_diagnostics_available",
            "supported": str(bool(active_rows)).lower(),
            "evidence_csv": "contact_enforcement_active_set.csv",
            "details": "node, centroid, quadrature, and smooth-pressure active definitions are listed separately",
        },
        {
            "claim": "area_weighting_diagnostics_available",
            "supported": str(bool(area_rows)).lower(),
            "evidence_csv": "contact_enforcement_area_weighting.csv",
            "details": "area-weighted quadrature is compared against centroid and unweighted nodal variants",
        },
        {
            "claim": "pressure_overclosure_law_diagnostics_available",
            "supported": str(bool(law_rows)).lower(),
            "evidence_csv": "contact_enforcement_pressure_law.csv",
            "details": "smooth and hard linear pressure-overclosure responses are tabulated around zero overclosure",
        },
        {
            "claim": "contact_tangent_finite_difference_check",
            "supported": str(all(row["status"] == "passed" for row in tangent_rows)).lower(),
            "evidence_csv": "contact_enforcement_tangent_fd.csv",
            "details": "contact tangent is checked against finite differences with residual sign convention",
        },
        {
            "claim": "calculix_contact_output_definition_available",
            "supported": str(output_available).lower(),
            "evidence_csv": "contact_enforcement_calculix_outputs.csv",
            "details": "true only when optional CalculiX run exports RF/contact totals for parsing",
        },
    ]


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    claim_rows: list[Row],
    active_rows: list[Row],
    area_rows: list[Row],
    law_rows: list[Row],
    tangent_rows: list[Row],
    output_rows: list[Row],
    *,
    quick: bool,
    run_external: bool,
) -> None:
    command = "python validation/run_contact_enforcement_diagnostics.py"
    if quick:
        command += " --quick"
    if not run_external:
        command += " --skip-calculix"
    command += f" --out-dir {path.parent.as_posix()}"
    lines = [
        "# Contact Enforcement Diagnostics",
        "",
        "This diagnostic decomposes remaining SFC/CalculiX contact differences into active-set, area-weighting, pressure-law, tangent, and output-definition terms.",
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Claim Gates",
        "",
        "| Claim | Supported | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in claim_rows:
        lines.append(f"| {row['claim']} | {row['supported']} | `{row['evidence_csv']}` |")
    lines.extend(
        [
            "",
            "## Row Counts",
            "",
            f"- active-set rows: `{len(active_rows)}`",
            f"- area-weighting rows: `{len(area_rows)}`",
            f"- pressure-law rows: `{len(law_rows)}`",
            f"- tangent rows: `{len(tangent_rows)}`",
            f"- CalculiX output rows: `{len(output_rows)}`",
            "",
            "## Interpretation",
            "",
            "- Active contact counts are not directly comparable unless their definitions match.",
            "- CalculiX `RF` is a fixed-floor total reaction; SFC quadrature force is an assembled penalty force.",
            "- CalculiX `CNUM` is a contact-element count, not the number of SFC quadrature points.",
            "- Tangent rows use the residual convention `K_contact ~= -d(f_contact)/du`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-calculix", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "contact_enforcement_diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_diagnostics(args.out_dir, quick=bool(args.quick), run_external=not bool(args.skip_calculix))
    print("Contact enforcement diagnostics complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
