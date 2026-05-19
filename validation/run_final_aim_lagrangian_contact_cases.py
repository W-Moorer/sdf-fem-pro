"""Run existing TET4/HEX8 contact cases through the final-aim oracle path.

The migrated cases mirror the small deterministic Phase-10 static/dynamic
models, but the contact query path is now:

    MaterialSDF -> LagrangianSDFContactOracle -> oracle penalty response

The reference is an analytic rigid plane with the same nodal penalty law.  This
runner does not build a current-space SDF grid and does not call the legacy
projection query path.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact import (  # noqa: E402
    LagrangianSDFContactOracle,
    SurfaceSample,
    lagrangian_oracle_penalty_response,
)
from sfc.fem import DeformableBody, assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.fem.hex8 import hex8_center_strain_stress  # noqa: E402
from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.fem.tet4 import tet4_strain_displacement_matrix  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.sdf import MaterialSDF  # noqa: E402
from validation.run_phase3_validation import structured_tet_block  # noqa: E402

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class CaseModel:
    mesh: VolumeMesh
    body: DeformableBody
    bottom_nodes: np.ndarray
    element_type: str


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _structured_hex_block(nx: int, ny: int, nz: int, *, size: tuple[float, float, float]) -> VolumeMesh:
    if min(nx, ny, nz) <= 0:
        raise ValueError("mesh resolution must be positive")
    lx, ly, lz = size
    nodes: list[tuple[float, float, float]] = []

    def node_id(i: int, j: int, k: int) -> int:
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes.append((lx * i / nx, ly * j / ny, lz * k / nz))
    elements: list[tuple[int, int, int, int, int, int, int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements.append(
                    (
                        node_id(i, j, k),
                        node_id(i + 1, j, k),
                        node_id(i + 1, j + 1, k),
                        node_id(i, j + 1, k),
                        node_id(i, j, k + 1),
                        node_id(i + 1, j, k + 1),
                        node_id(i + 1, j + 1, k + 1),
                        node_id(i, j + 1, k + 1),
                    )
                )
    return VolumeMesh(np.asarray(nodes, dtype=float), np.asarray(elements, dtype=np.int64), element_type="hex8")


def _case_model(element_type: str, *, quick: bool) -> CaseModel:
    nx = ny = 1 if quick else 2
    nz = 1
    size = (1.0, 1.0, 0.20)
    if element_type == "tet4":
        base = structured_tet_block(nx, ny, nz, size=size)
    elif element_type == "hex8":
        base = _structured_hex_block(nx, ny, nz, size=size)
    else:
        raise ValueError("element_type must be tet4 or hex8")
    X = base.X.copy()
    X[:, 2] += 0.04
    mesh = VolumeMesh(X, base.elements, element_type=element_type)
    body = DeformableBody(mesh=mesh, material={"E": 200.0, "nu": 0.30}, density=1.0)
    bottom_nodes = np.nonzero(np.isclose(mesh.X[:, 2], float(np.min(mesh.X[:, 2]))))[0]
    return CaseModel(mesh=mesh, body=body, bottom_nodes=bottom_nodes, element_type=element_type)


def _master_plane_oracle() -> LagrangianSDFContactOracle:
    X = np.asarray(
        [
            [-0.25, -0.25, 0.0],
            [1.25, -0.25, 0.0],
            [1.25, 1.25, 0.0],
            [-0.25, 1.25, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    material = MaterialSDF.from_triangle_surface(
        X,
        faces,
        phi=lambda point: float(point[2]),
        gradient=lambda _point: np.asarray([0.0, 0.0, 1.0], dtype=float),
        band_radius=0.25,
    )
    return LagrangianSDFContactOracle(material, X, search_radius=0.35, patch_cell_size=0.20)


def _bottom_node_samples(bottom_nodes: np.ndarray) -> list[SurfaceSample]:
    return [
        SurfaceSample(
            np.asarray([int(node)], dtype=np.int64),
            np.asarray([1.0], dtype=float),
            np.asarray([0], dtype=np.int64),
        )
        for node in np.asarray(bottom_nodes, dtype=np.int64)
    ]


def _oracle_contact_response(
    x_current: np.ndarray,
    samples: list[SurfaceSample],
    oracle: LagrangianSDFContactOracle,
    *,
    stiffness: float,
) -> tuple[tuple[Any, ...], np.ndarray, float]:
    start = perf_counter()
    slave_dofs = x_current.shape[0] * 3
    response = lagrangian_oracle_penalty_response(
        x_current,
        samples,
        oracle,
        pressure_stiffness=stiffness,
        n_total_dofs=slave_dofs + oracle.x_current.shape[0] * 3,
        slave_dof_offset=0,
        master_dof_offset=slave_dofs,
    )
    elapsed = perf_counter() - start
    return response.constraints, response.force[:slave_dofs], elapsed


def _analytic_plane_response(
    x_current: np.ndarray,
    samples: list[SurfaceSample],
    *,
    stiffness: float,
) -> tuple[np.ndarray, np.ndarray]:
    gaps = np.zeros(len(samples), dtype=float)
    force = np.zeros(x_current.shape[0] * 3, dtype=float)
    for row, sample in enumerate(samples):
        point = sample.point(x_current)
        gap = float(point[2])
        gaps[row] = gap
        penetration = max(-gap, 0.0)
        if penetration <= 0.0:
            continue
        for node, weight in zip(sample.node_ids, sample.weights, strict=True):
            force[int(node) * 3 + 2] += float(stiffness) * penetration * float(weight)
    return gaps, force


def _relative_norm_error(value: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    if denom <= 1.0e-30:
        return float(np.linalg.norm(value - reference))
    return float(np.linalg.norm(value - reference) / denom)


def _stress_metrics(model: CaseModel, u: np.ndarray) -> tuple[float, float]:
    E, nu = float(model.body.material["E"]), float(model.body.material["nu"])
    if model.element_type == "tet4":
        C = isotropic_linear_elasticity_matrix(E, nu)
        values: list[float] = []
        strain_norms: list[float] = []
        u_nodes = u.reshape((-1, 3))
        for element in model.mesh.elements:
            B = tet4_strain_displacement_matrix(model.mesh.X[element])
            strain = B @ u_nodes[element].reshape(12)
            stress = C @ strain
            values.append(_von_mises_voigt(stress))
            strain_norms.append(float(np.linalg.norm(strain)))
        return float(np.max(values)), float(np.max(strain_norms))
    values = []
    strain_norms = []
    u_nodes = u.reshape((-1, 3))
    for element in model.mesh.elements:
        strain, stress = hex8_center_strain_stress(model.mesh.X[element], u_nodes[element], E, nu)
        values.append(_von_mises_voigt(stress))
        strain_norms.append(float(np.linalg.norm(strain)))
    return float(np.max(values)), float(np.max(strain_norms))


def _von_mises_voigt(stress: np.ndarray) -> float:
    s = np.asarray(stress, dtype=float).ravel()
    if s.size == 6:
        sxx, syy, szz, sxy, syz, sxz = s
    elif s.shape == (3, 3):
        sxx, syy, szz = s[0, 0], s[1, 1], s[2, 2]
        sxy, syz, sxz = s[0, 1], s[1, 2], s[0, 2]
    else:
        raise ValueError("stress must be Voigt-6 or 3x3")
    return float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + syz**2 + sxz**2)))


def _static_case(element_type: str, *, quick: bool, oracle: LagrangianSDFContactOracle, stiffness: float) -> tuple[Row, list[Row]]:
    case_start = perf_counter()
    model = _case_model(element_type, quick=quick)
    z_min = float(np.min(model.mesh.X[:, 2]))
    height = float(np.max(model.mesh.X[:, 2]) - z_min)
    u = np.zeros(model.body.n_dofs, dtype=float).reshape((-1, 3))
    zeta = (model.mesh.X[:, 2] - z_min) / max(height, 1.0e-30)
    u[:, 2] = -0.060 - 0.010 * zeta
    x_current = model.mesh.X + u
    samples = _bottom_node_samples(model.bottom_nodes)
    constraints, oracle_force, query_seconds = _oracle_contact_response(x_current, samples, oracle, stiffness=stiffness)
    analytic_gaps, analytic_force = _analytic_plane_response(x_current, samples, stiffness=stiffness)
    oracle_gaps = np.asarray([constraint.g for constraint in constraints], dtype=float)
    max_vm, max_strain = _stress_metrics(model, u.ravel())
    row = _case_row(
        case_id=f"static_linear_{element_type}_lagrangian_oracle",
        regime="static",
        model=model,
        constraints=constraints,
        oracle_gaps=oracle_gaps,
        analytic_gaps=analytic_gaps,
        oracle_force=oracle_force,
        analytic_force=analytic_force,
        query_seconds=query_seconds,
        max_von_mises=max_vm,
        max_strain_norm=max_strain,
        z_cm_error=0.0,
    )
    row["case_wall_seconds"] = float(perf_counter() - case_start)
    return row, _sample_rows(row["case_id"], constraints, analytic_gaps)


def _dynamic_case(element_type: str, *, quick: bool, oracle: LagrangianSDFContactOracle, stiffness: float) -> tuple[Row, list[Row], list[Row]]:
    case_start = perf_counter()
    model_oracle = _case_model(element_type, quick=quick)
    model_ref = _case_model(element_type, quick=quick)
    samples = _bottom_node_samples(model_oracle.bottom_nodes)
    K = assemble_stiffness_matrix(model_oracle.body).tocsr()
    M = assemble_mass_matrix(model_oracle.body, kind="lumped").tocsr()
    mass_diag = np.maximum(M.diagonal(), 1.0e-12)
    steps = 70 if quick else 110
    dt = 1.0e-3
    u_oracle = np.zeros(model_oracle.body.n_dofs, dtype=float)
    v_oracle = np.zeros_like(u_oracle)
    u_ref = np.zeros(model_ref.body.n_dofs, dtype=float)
    v_ref = np.zeros_like(u_ref)
    v_oracle[2::3] = -0.85
    v_ref[2::3] = -0.85

    max_gap_error = 0.0
    max_force_error = 0.0
    max_z_cm_error = 0.0
    max_pen_oracle = 0.0
    max_pen_ref = 0.0
    peak_reaction_oracle = 0.0
    peak_reaction_ref = 0.0
    total_query_seconds = 0.0
    history_rows: list[Row] = []
    last_constraints: tuple[Any, ...] = ()
    last_analytic_gaps = np.empty(0, dtype=float)

    for step in range(steps):
        x_oracle = model_oracle.mesh.X + u_oracle.reshape((-1, 3))
        constraints, oracle_force, query_seconds = _oracle_contact_response(x_oracle, samples, oracle, stiffness=stiffness)
        total_query_seconds += query_seconds
        oracle_gaps = np.asarray([constraint.g for constraint in constraints], dtype=float)

        x_ref = model_ref.mesh.X + u_ref.reshape((-1, 3))
        analytic_gaps, analytic_force = _analytic_plane_response(x_ref, samples, stiffness=stiffness)
        max_gap_error = max(max_gap_error, float(np.max(np.abs(oracle_gaps - analytic_gaps))))
        max_force_error = max(max_force_error, _relative_norm_error(oracle_force, analytic_force))

        f_oracle = oracle_force - K @ u_oracle
        f_ref = analytic_force - K @ u_ref
        v_oracle += dt * (f_oracle / mass_diag)
        v_ref += dt * (f_ref / mass_diag)
        u_oracle += dt * v_oracle
        u_ref += dt * v_ref

        z_cm_oracle = float(np.mean((model_oracle.mesh.X + u_oracle.reshape((-1, 3)))[:, 2]))
        z_cm_ref = float(np.mean((model_ref.mesh.X + u_ref.reshape((-1, 3)))[:, 2]))
        max_z_cm_error = max(max_z_cm_error, abs(z_cm_oracle - z_cm_ref))
        max_pen_oracle = max(max_pen_oracle, max(-float(np.min(oracle_gaps)), 0.0))
        max_pen_ref = max(max_pen_ref, max(-float(np.min(analytic_gaps)), 0.0))
        peak_reaction_oracle = max(peak_reaction_oracle, float(np.sum(oracle_force[2::3])))
        peak_reaction_ref = max(peak_reaction_ref, float(np.sum(analytic_force[2::3])))
        history_rows.append(
            {
                "case_id": f"dynamic_linear_{element_type}_lagrangian_oracle",
                "step": step,
                "time": step * dt,
                "min_gap_oracle": float(np.min(oracle_gaps)),
                "min_gap_reference": float(np.min(analytic_gaps)),
                "reaction_z_oracle": float(np.sum(oracle_force[2::3])),
                "reaction_z_reference": float(np.sum(analytic_force[2::3])),
                "z_cm_oracle": z_cm_oracle,
                "z_cm_reference": z_cm_ref,
            }
        )
        last_constraints = constraints
        last_analytic_gaps = analytic_gaps

    max_vm, max_strain = _stress_metrics(model_oracle, u_oracle)
    row = _case_row(
        case_id=f"dynamic_linear_{element_type}_lagrangian_oracle",
        regime="dynamic",
        model=model_oracle,
        constraints=last_constraints,
        oracle_gaps=np.asarray([constraint.g for constraint in last_constraints], dtype=float),
        analytic_gaps=last_analytic_gaps,
        oracle_force=np.zeros(model_oracle.body.n_dofs, dtype=float),
        analytic_force=np.zeros(model_oracle.body.n_dofs, dtype=float),
        query_seconds=total_query_seconds,
        max_von_mises=max_vm,
        max_strain_norm=max_strain,
        z_cm_error=max_z_cm_error,
        max_gap_error_override=max_gap_error,
        max_force_error_override=max_force_error,
        max_penetration_oracle_override=max_pen_oracle,
        max_penetration_reference_override=max_pen_ref,
        reaction_oracle_override=peak_reaction_oracle,
        reaction_reference_override=peak_reaction_ref,
    )
    row["case_wall_seconds"] = float(perf_counter() - case_start)
    return row, _sample_rows(row["case_id"], last_constraints, last_analytic_gaps), history_rows


def _case_row(
    *,
    case_id: str,
    regime: str,
    model: CaseModel,
    constraints: tuple[Any, ...],
    oracle_gaps: np.ndarray,
    analytic_gaps: np.ndarray,
    oracle_force: np.ndarray,
    analytic_force: np.ndarray,
    query_seconds: float,
    max_von_mises: float,
    max_strain_norm: float,
    z_cm_error: float,
    max_gap_error_override: float | None = None,
    max_force_error_override: float | None = None,
    max_penetration_oracle_override: float | None = None,
    max_penetration_reference_override: float | None = None,
    reaction_oracle_override: float | None = None,
    reaction_reference_override: float | None = None,
) -> Row:
    max_gap_error = float(np.max(np.abs(oracle_gaps - analytic_gaps))) if oracle_gaps.size else 0.0
    max_force_rel = _relative_norm_error(oracle_force, analytic_force)
    max_pen_oracle = max(-float(np.min(oracle_gaps)), 0.0) if oracle_gaps.size else 0.0
    max_pen_ref = max(-float(np.min(analytic_gaps)), 0.0) if analytic_gaps.size else 0.0
    reaction_oracle = float(np.sum(oracle_force[2::3]))
    reaction_ref = float(np.sum(analytic_force[2::3]))
    if max_gap_error_override is not None:
        max_gap_error = float(max_gap_error_override)
    if max_force_error_override is not None:
        max_force_rel = float(max_force_error_override)
    if max_penetration_oracle_override is not None:
        max_pen_oracle = float(max_penetration_oracle_override)
    if max_penetration_reference_override is not None:
        max_pen_ref = float(max_penetration_reference_override)
    if reaction_oracle_override is not None:
        reaction_oracle = float(reaction_oracle_override)
    if reaction_reference_override is not None:
        reaction_ref = float(reaction_reference_override)
    status = "passed" if max_gap_error < 1.0e-10 and max_force_rel < 1.0e-10 and z_cm_error < 1.0e-10 else "failed"
    return {
        "case_id": case_id,
        "regime": regime,
        "element_type": model.element_type,
        "contact_path": "MaterialSDF+LagrangianSDFContactOracle",
        "reference": "analytic_plane_same_penalty_law",
        "node_count": model.mesh.X.shape[0],
        "element_count": model.mesh.elements.shape[0],
        "sample_count": len(constraints),
        "oracle_query_seconds_total": float(query_seconds),
        "max_gap_abs_error": max_gap_error,
        "force_l2_rel_error": max_force_rel,
        "max_penetration_oracle": max_pen_oracle,
        "max_penetration_reference": max_pen_ref,
        "reaction_z_oracle": reaction_oracle,
        "reaction_z_reference": reaction_ref,
        "z_cm_max_abs_error": float(z_cm_error),
        "max_von_mises": float(max_von_mises),
        "max_strain_norm": float(max_strain_norm),
        "status": status,
    }


def _sample_rows(case_id: str, constraints: tuple[Any, ...], analytic_gaps: np.ndarray) -> list[Row]:
    rows: list[Row] = []
    for idx, constraint in enumerate(constraints):
        rows.append(
            {
                "case_id": case_id,
                "sample_id": idx,
                "x": float(constraint.x_slave[0]),
                "y": float(constraint.x_slave[1]),
                "z": float(constraint.x_slave[2]),
                "gap_oracle": float(constraint.g),
                "gap_reference": float(analytic_gaps[idx]) if idx < analytic_gaps.size else "",
                "normal_x": float(constraint.normal[0]),
                "normal_y": float(constraint.normal[1]),
                "normal_z": float(constraint.normal[2]),
                "active": int(constraint.active),
                "face_id": int(constraint.query.face_id),
                "candidates_evaluated": int(constraint.query.candidates_evaluated),
            }
        )
    return rows


def _write_summary(path: Path, case_rows: list[Row], outputs: dict[str, Path]) -> None:
    passed = sum(1 for row in case_rows if row["status"] == "passed")
    lines = [
        "# Final-Aim Lagrangian SDF Contact Case Migration",
        "",
        "These cases migrate the existing static/dynamic TET4/HEX8 contact checks to:",
        "",
        "```text",
        "MaterialSDF -> LagrangianSDFContactOracle -> oracle penalty response",
        "```",
        "",
        "The analytic rigid plane is used only as a reference with the same nodal penalty law.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        if key != "summary":
            lines.append(f"- `{value.relative_to(path.parent).as_posix()}`")
    lines.extend(
        [
            "",
            "## Case Results",
            "",
            "| Case | Status | Wall time (s) | Max gap error | Force rel. error | Max penetration | Reaction z |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in case_rows:
        lines.append(
            f"| `{row['case_id']}` | {row['status']} | "
            f"{float(row.get('case_wall_seconds', 0.0)):.6e} | "
            f"{float(row['max_gap_abs_error']):.6e} | "
            f"{float(row['force_l2_rel_error']):.6e} | "
            f"{float(row['max_penetration_oracle']):.6e} | "
            f"{float(row['reaction_z_oracle']):.6e} |"
        )
    lines.extend(
        [
            "",
            "## Claim Gate",
            "",
            f"- Cases passed: `{passed}/{len(case_rows)}`.",
            f"- Total wall time for this runner: `{sum(float(row.get('case_wall_seconds', 0.0)) for row in case_rows):.6e}` s.",
            f"- Final-aim oracle path used in every case: `{str(all(row['contact_path'] == 'MaterialSDF+LagrangianSDFContactOracle' for row in case_rows)).lower()}`.",
            "- No current-space SDF grid is built in this runner.",
            "- The validation checks migration correctness for the existing small TET4/HEX8 cases, not broad nonlinear/friction/self-contact coverage.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(out_dir: Path, *, quick: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle = _master_plane_oracle()
    stiffness = 150.0
    case_rows: list[Row] = []
    sample_rows: list[Row] = []
    history_rows: list[Row] = []

    for element_type in ("tet4", "hex8"):
        row, samples = _static_case(element_type, quick=quick, oracle=oracle, stiffness=stiffness)
        case_rows.append(row)
        sample_rows.extend(samples)
        row, samples, history = _dynamic_case(element_type, quick=quick, oracle=oracle, stiffness=stiffness)
        case_rows.append(row)
        sample_rows.extend(samples)
        history_rows.extend(history)

    outputs = {
        "cases": out_dir / "final_aim_lagrangian_contact_cases.csv",
        "samples": out_dir / "final_aim_lagrangian_contact_samples.csv",
        "history": out_dir / "final_aim_lagrangian_contact_history.csv",
        "summary": out_dir / "final_aim_lagrangian_contact_summary.md",
    }
    _write_csv(outputs["cases"], case_rows)
    _write_csv(outputs["samples"], sample_rows)
    _write_csv(outputs["history"], history_rows)
    _write_summary(outputs["summary"], case_rows, outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "final_aim_lagrangian_contact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(args.out_dir, quick=bool(args.quick))
    print("Final-aim Lagrangian SDF contact case migration complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
