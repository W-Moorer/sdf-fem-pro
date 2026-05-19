"""Compare existing final-aim Lagrangian-SDF cases against CalculiX.

This runner is the second-stage validation after
``run_final_aim_lagrangian_contact_cases.py``.  It keeps the same small TET4
and HEX8 contact models, runs SFC through the material/Lagrangian SDF oracle,
generates matching CalculiX native contact input decks, and compares
displacement, strain, and stress histories.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sfc.contact import lagrangian_oracle_penalty_response  # noqa: E402
from sfc.fem import assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.fem.hex8 import hex8_center_strain_stress  # noqa: E402
from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.fem.tet4 import tet4_strain_displacement_matrix  # noqa: E402
from validation.run_c3d8_contact_trajectory_validation import (  # noqa: E402
    C3D8_FACE_NODES,
    FLOOR_SHELL_THICKNESS,
    QUAD_GAUSS,
    _calculix_version,
    _parse_element_stress,
    _parse_nodal_vectors,
    _quad_shape,
    _wsl_path,
    calculix_available,
)
from validation.run_final_aim_lagrangian_contact_cases import (  # noqa: E402
    _case_model,
    _master_plane_oracle,
    _write_csv,
    _von_mises_voigt,
)

Row = dict[str, Any]

TET4_FACE_NODES = {
    "S1": np.asarray([0, 2, 1], dtype=np.int64),
    "S2": np.asarray([0, 1, 3], dtype=np.int64),
    "S3": np.asarray([1, 2, 3], dtype=np.int64),
    "S4": np.asarray([0, 3, 2], dtype=np.int64),
}


@dataclass(frozen=True, slots=True)
class AlignmentModel:
    """Small final-aim model shared by SFC and CalculiX."""

    case_id: str
    regime: str
    element_type: str
    X: np.ndarray
    elements: np.ndarray
    E: float
    nu: float
    density: float
    top_nodes: np.ndarray
    bottom_surface: list[tuple[int, str]]
    node_ids: np.ndarray
    element_ids: np.ndarray
    floor_nodes: np.ndarray
    floor_element: np.ndarray
    contact_stiffness: float = 150.0
    static_closure: float = 0.07
    initial_velocity_z: float = -0.85
    dt: float = 1.0e-3
    dynamic_steps: int = 110
    static_steps: int = 11

    @property
    def total_time(self) -> float:
        return float(self.dt * self.dynamic_steps)


def _append_id_list(lines: list[str], values: np.ndarray, *, per_line: int = 12) -> None:
    ids = [str(int(value)) for value in np.asarray(values, dtype=np.int64).ravel()]
    for start in range(0, len(ids), per_line):
        lines.append(", ".join(ids[start : start + per_line]))


def _floor(half_width: float, z: float, *, start_id: int, element_id: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(
        [
            [start_id, -half_width, -half_width, z],
            [start_id + 1, half_width, -half_width, z],
            [start_id + 2, half_width, half_width, z],
            [start_id + 3, -half_width, half_width, z],
        ],
        dtype=float,
    )
    element = np.asarray([element_id, start_id, start_id + 1, start_id + 2, start_id + 3], dtype=np.int64)
    return nodes, element


def _surface_refs(X: np.ndarray, elements: np.ndarray, element_type: str) -> list[tuple[int, str]]:
    face_nodes = TET4_FACE_NODES if element_type == "tet4" else C3D8_FACE_NODES
    z_min = float(np.min(X[:, 2]))
    refs: list[tuple[int, str]] = []
    for element_index, element in enumerate(np.asarray(elements, dtype=np.int64)):
        for label, local in face_nodes.items():
            nodes = element[np.asarray(local, dtype=np.int64)]
            if np.allclose(X[nodes, 2], z_min):
                refs.append((element_index, str(label)))
    if not refs:
        raise RuntimeError(f"no bottom contact surface found for {element_type}")
    return refs


def _alignment_model(element_type: str, regime: str, *, quick: bool) -> AlignmentModel:
    base = _case_model(element_type, quick=quick)
    X = base.mesh.X.copy()
    elements = base.mesh.elements.copy()
    node_ids = np.arange(1, X.shape[0] + 1, dtype=np.int64)
    element_ids = np.arange(1, elements.shape[0] + 1, dtype=np.int64)
    top_nodes = np.nonzero(np.isclose(X[:, 2], float(np.max(X[:, 2]))))[0]
    floor_nodes, floor_element = _floor(
        1.5,
        -0.5 * FLOOR_SHELL_THICKNESS,
        start_id=int(node_ids[-1]) + 1,
        element_id=int(element_ids[-1]) + 1,
    )
    return AlignmentModel(
        case_id=f"{regime}_linear_{element_type}_lagrangian_vs_calculix",
        regime=regime,
        element_type=element_type,
        X=X,
        elements=elements,
        E=float(base.body.material["E"]),
        nu=float(base.body.material["nu"]),
        density=float(base.body.density),
        top_nodes=top_nodes,
        bottom_surface=_surface_refs(X, elements, element_type),
        node_ids=node_ids,
        element_ids=element_ids,
        floor_nodes=floor_nodes,
        floor_element=floor_element,
        dynamic_steps=70 if quick else 110,
        static_steps=6 if quick else 11,
    )


def _write_calculix_input(model: AlignmentModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    element_keyword = "C3D4" if model.element_type == "tet4" else "C3D8"
    lines: list[str] = [
        "** Generated by validation/run_final_aim_calculix_alignment.py",
        "*HEADING",
        f"Final-aim same-case CalculiX alignment: {model.case_id}",
        "*NODE, NSET=NALL",
    ]
    for node_id, xyz in zip(model.node_ids, model.X, strict=True):
        lines.append(f"{int(node_id)}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    lines.append("*NODE, NSET=NFLOOR")
    for row in model.floor_nodes:
        lines.append(f"{int(row[0])}, {row[1]:.12e}, {row[2]:.12e}, {row[3]:.12e}")
    lines.append(f"*ELEMENT, TYPE={element_keyword}, ELSET=ELALL")
    for element_id, element in zip(model.element_ids, model.elements, strict=True):
        conn = ", ".join(str(int(model.node_ids[int(idx)])) for idx in element)
        lines.append(f"{int(element_id)}, {conn}")
    lines.append("*ELEMENT, TYPE=S4, ELSET=FLOOR")
    lines.append(", ".join(str(int(value)) for value in model.floor_element))
    lines.append("*NSET, NSET=NTOP")
    _append_id_list(lines, model.node_ids[model.top_nodes])
    lines.extend(
        [
            "*MATERIAL, NAME=MAT",
            "*ELASTIC",
            f"{model.E:.12e}, {model.nu:.12e}",
            "*DENSITY",
            f"{model.density:.12e}",
            "*SOLID SECTION, ELSET=ELALL, MATERIAL=MAT",
            "*MATERIAL, NAME=FLOORMAT",
            "*ELASTIC",
            f"{model.E:.12e}, {model.nu:.12e}",
            "*SHELL SECTION, ELSET=FLOOR, MATERIAL=FLOORMAT",
            f"{FLOOR_SHELL_THICKNESS:.12e}",
            "*SURFACE, NAME=SSLAVE, TYPE=ELEMENT",
        ]
    )
    for element_index, label in model.bottom_surface:
        lines.append(f"{int(model.element_ids[int(element_index)])}, {label}")
    lines.extend(
        [
            "*SURFACE, NAME=SMASTER, TYPE=ELEMENT",
            "FLOOR, SPOS",
            "*SURFACE INTERACTION, NAME=CONTACT",
            "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
            f"{model.contact_stiffness:.12e}",
            "*CONTACT PAIR, INTERACTION=CONTACT, TYPE=SURFACE TO SURFACE",
            "SSLAVE, SMASTER",
            "*BOUNDARY",
            "NFLOOR, 1, 6, 0.0",
        ]
    )
    if model.regime == "static":
        lines.extend(
            [
                "NTOP, 1, 2, 0.0",
                "*STEP, NLGEOM, INC=10000",
                "*STATIC",
                f"{1.0 / max(model.static_steps - 1, 1):.12e}, 1.0, 1.0e-8, {1.0 / max(model.static_steps - 1, 1):.12e}",
                "*BOUNDARY",
                f"NTOP, 3, 3, {-model.static_closure:.12e}",
            ]
        )
    else:
        lines.extend(
            [
                "*INITIAL CONDITIONS, TYPE=VELOCITY",
                f"NALL, 3, {model.initial_velocity_z:.12e}",
                "*STEP, NLGEOM, INC=1000000",
                "*DYNAMIC, ALPHA=-0.05",
                f"{model.dt:.12e}, {model.total_time:.12e}, {1.0e-4 * model.dt:.12e}, {model.dt:.12e}",
            ]
        )
    lines.extend(
        [
            "*NODE PRINT, NSET=NALL, FREQUENCY=1",
            "U",
            "*EL PRINT, ELSET=ELALL, FREQUENCY=1",
            "S",
            "*CONTACT PRINT, FREQUENCY=1, TOTALS=YES",
            "CDIS, CSTR, CELS, CNUM",
            "*END STEP",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_calculix(model: AlignmentModel, out_dir: Path, *, timeout_seconds: int) -> tuple[dict[float, np.ndarray], dict[float, dict[int, np.ndarray]], Row]:
    if not calculix_available():
        return {}, {}, {"case_id": model.case_id, "solver": "calculix", "completed": "false", "command": "ccx unavailable", "wall_time_seconds": ""}
    run_dir = out_dir / "calculix_runs" / model.case_id
    run_dir.mkdir(parents=True, exist_ok=True)
    inp = run_dir / f"{model.case_id}.inp"
    _write_calculix_input(model, inp)
    command = f"cd {_wsl_path(run_dir)} && ccx {model.case_id}"
    start = time.perf_counter()
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    wall = time.perf_counter() - start
    (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
    dat = run_dir / f"{model.case_id}.dat"
    row: Row = {
        "case_id": model.case_id,
        "solver": "calculix",
        "completed": str(proc.returncode == 0 and dat.exists()).lower(),
        "return_code": proc.returncode,
        "command": f"wsl --exec bash -lc \"{command}\"",
        "input_file": str(inp.relative_to(out_dir)),
        "dat_file": str(dat.relative_to(out_dir)) if dat.exists() else "",
        "wall_time_seconds": wall,
        "external_solver_version": _calculix_version(),
    }
    if not dat.exists():
        return {}, {}, row
    return _parse_nodal_vectors(dat, model.node_ids, quantity="u"), _parse_element_stress(dat), row


def _surface_samples(model: AlignmentModel):
    from sfc.contact import SurfaceSample

    samples: list[SurfaceSample] = []
    weights: list[float] = []
    face_nodes = TET4_FACE_NODES if model.element_type == "tet4" else C3D8_FACE_NODES
    for element_index, label in model.bottom_surface:
        element = model.elements[int(element_index)]
        local = np.asarray(face_nodes[str(label)], dtype=np.int64)
        nodes = element[local]
        coords = model.X[nodes]
        if model.element_type == "tet4":
            area = 0.5 * float(np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0])))
            samples.append(
                SurfaceSample(
                    node_ids=np.asarray(nodes, dtype=np.int64),
                    weights=np.full(3, 1.0 / 3.0, dtype=float),
                    candidate_face_ids=np.asarray([0, 1], dtype=np.int64),
                )
            )
            weights.append(area)
        else:
            for xi in QUAD_GAUSS:
                for eta in QUAD_GAUSS:
                    shape, dxi, deta = _quad_shape(float(xi), float(eta))
                    jac = np.cross(dxi @ coords, deta @ coords)
                    samples.append(
                        SurfaceSample(
                            node_ids=np.asarray(nodes, dtype=np.int64),
                            weights=np.asarray(shape, dtype=float),
                            candidate_face_ids=np.asarray([0, 1], dtype=np.int64),
                        )
                    )
                    weights.append(float(np.linalg.norm(jac)))
    return samples, np.asarray(weights, dtype=float)


def _sfc_contact_response(model: AlignmentModel, x_current: np.ndarray):
    oracle = _master_plane_oracle()
    samples, sample_area_weights = _surface_samples(model)
    return lagrangian_oracle_penalty_response(
        x_current,
        samples,
        oracle,
        pressure_stiffness=model.contact_stiffness,
        n_total_dofs=3 * model.X.shape[0] + 3 * oracle.x_current.shape[0],
        sample_area_weights=sample_area_weights,
        slave_dof_offset=0,
        master_dof_offset=3 * model.X.shape[0],
    )


def _dirichlet_dofs(nodes: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    return np.asarray([3 * int(node) + int(axis) for node in nodes for axis in axes], dtype=np.int64)


def _solve_sfc_static(model: AlignmentModel) -> tuple[list[Row], dict[float, np.ndarray], Row]:
    case_start = time.perf_counter()
    base = _case_model(model.element_type, quick=model.X.shape[0] <= 8)
    K = assemble_stiffness_matrix(base.body).tocsr()
    n_dofs = 3 * model.X.shape[0]
    fixed_xy = _dirichlet_dofs(model.top_nodes, (0, 1))
    top_z = _dirichlet_dofs(model.top_nodes, (2,))
    fixed = np.unique(np.concatenate([fixed_xy, top_z]))
    free = np.setdiff1d(np.arange(n_dofs, dtype=np.int64), fixed)
    rows: list[Row] = []
    frames: dict[float, np.ndarray] = {0.0: np.zeros_like(model.X)}
    u = np.zeros(n_dofs, dtype=float)
    for step in range(model.static_steps):
        load = step / max(model.static_steps - 1, 1)
        u[top_z] = -model.static_closure * load
        for _iteration in range(20):
            x = model.X + u.reshape((-1, 3))
            contact = _sfc_contact_response(model, x)
            residual = K @ u - contact.force[:n_dofs]
            tangent = (K + contact.stiffness[:n_dofs, :n_dofs]).tocsr()
            correction = np.zeros(n_dofs, dtype=float)
            if free.size:
                correction[free] = np.asarray(spsolve(tangent[free[:, None], free].tocsc(), -residual[free]), dtype=float)
            u[free] += correction[free]
            u[fixed_xy] = 0.0
            u[top_z] = -model.static_closure * load
            if np.linalg.norm(correction[free]) <= 1.0e-10 * max(1.0, np.linalg.norm(u[free])):
                break
        U = u.reshape((-1, 3)).copy()
        frames[float(load)] = U
        rows.append(_history_row("sfc_lagrangian_sdf", model, float(load), U, contact=response_or_current(model, U)))
    return rows, frames, {
        "case_id": model.case_id,
        "solver": "sfc_lagrangian_sdf",
        "completed": "true",
        "wall_time_seconds": float(time.perf_counter() - case_start),
        "command": "internal SFC static Newton solve",
    }


def response_or_current(model: AlignmentModel, U: np.ndarray):
    return _sfc_contact_response(model, model.X + U)


def _solve_sfc_dynamic(model: AlignmentModel) -> tuple[list[Row], dict[float, np.ndarray], Row]:
    case_start = time.perf_counter()
    base = _case_model(model.element_type, quick=model.X.shape[0] <= 8)
    K = assemble_stiffness_matrix(base.body).tocsr()
    M = assemble_mass_matrix(base.body, kind="lumped").tocsr()
    n_dofs = 3 * model.X.shape[0]
    u = np.zeros(n_dofs, dtype=float)
    v = np.zeros_like(u)
    v[2::3] = model.initial_velocity_z
    beta = 0.25
    gamma = 0.5
    c0 = 1.0 / (beta * model.dt * model.dt)
    rows: list[Row] = []
    frames: dict[float, np.ndarray] = {}
    contact = response_or_current(model, u.reshape((-1, 3)))
    a = np.asarray(spsolve(M.tocsc(), contact.force[:n_dofs] - K @ u), dtype=float)
    for step in range(model.dynamic_steps + 1):
        t = step * model.dt
        U = u.reshape((-1, 3)).copy()
        contact = response_or_current(model, U)
        frames[float(t)] = U
        rows.append(_history_row("sfc_lagrangian_sdf", model, float(t), U, contact=contact))
        if step == model.dynamic_steps:
            break
        u_pred = u + model.dt * v + model.dt * model.dt * (0.5 - beta) * a
        v_pred = v + model.dt * (1.0 - gamma) * a
        u_guess = u_pred.copy()
        for _iteration in range(12):
            U_guess = u_guess.reshape((-1, 3))
            contact = response_or_current(model, U_guess)
            a_guess = c0 * (u_guess - u_pred)
            residual = M @ a_guess + K @ u_guess - contact.force[:n_dofs]
            tangent = (M * c0 + K + contact.stiffness[:n_dofs, :n_dofs]).tocsc()
            correction = np.asarray(spsolve(tangent, -residual), dtype=float)
            u_guess += correction
            if np.linalg.norm(correction) <= 1.0e-10 * max(1.0, np.linalg.norm(u_guess)):
                break
        u = u_guess
        a = c0 * (u - u_pred)
        v = v_pred + gamma * model.dt * a
    return rows, frames, {
        "case_id": model.case_id,
        "solver": "sfc_lagrangian_sdf",
        "completed": "true",
        "wall_time_seconds": float(time.perf_counter() - case_start),
        "command": "internal SFC implicit Newmark solve with Lagrangian SDF contact",
    }


def _cell_metrics(model: AlignmentModel, U: np.ndarray, stress_map: dict[int, np.ndarray] | None = None) -> tuple[float, float]:
    stress_map = stress_map or {}
    if model.element_type == "tet4":
        C = isotropic_linear_elasticity_matrix(model.E, model.nu)
        vm: list[float] = []
        strain_norm: list[float] = []
        for eid, element in zip(model.element_ids, model.elements, strict=True):
            B = tet4_strain_displacement_matrix(model.X[element])
            strain = B @ U[element].reshape(12)
            if int(eid) in stress_map:
                stress = stress_map[int(eid)]
            else:
                stress = C @ strain
            vm.append(_von_mises_voigt(stress))
            strain_norm.append(float(np.linalg.norm(strain)))
        return float(max(vm) if vm else 0.0), float(max(strain_norm) if strain_norm else 0.0)
    vm = []
    strain_norm = []
    for eid, element in zip(model.element_ids, model.elements, strict=True):
        strain, stress = hex8_center_strain_stress(model.X[element], U[element], model.E, model.nu)
        if int(eid) in stress_map:
            stress = stress_map[int(eid)]
        vm.append(_von_mises_voigt(stress))
        strain_norm.append(float(np.linalg.norm(strain)))
    return float(max(vm) if vm else 0.0), float(max(strain_norm) if strain_norm else 0.0)


def _history_row(source: str, model: AlignmentModel, time_value: float, U: np.ndarray, *, contact: Any | None = None, stress_map: dict[int, np.ndarray] | None = None) -> Row:
    max_vm, max_strain = _cell_metrics(model, U, stress_map=stress_map)
    min_gap = ""
    active = ""
    reaction = ""
    if contact is not None:
        min_gap = float(contact.min_gap)
        active = int(contact.active_count)
        reaction = float(np.sum(contact.force[: 3 * model.X.shape[0]][2::3]))
    return {
        "case_id": model.case_id,
        "source": source,
        "regime": model.regime,
        "element_type": model.element_type,
        "time_or_load": float(time_value),
        "mean_uz": float(np.mean(U[:, 2])),
        "max_abs_u": float(np.max(np.linalg.norm(U, axis=1))),
        "max_von_mises": max_vm,
        "max_strain_norm": max_strain,
        "min_gap": min_gap,
        "active_contact": active,
        "reaction_z": reaction,
    }


def _calculix_history(model: AlignmentModel, displacements: dict[float, np.ndarray], stresses: dict[float, dict[int, np.ndarray]]) -> list[Row]:
    rows = [_history_row("calculix_native_contact", model, 0.0, np.zeros_like(model.X), stress_map={})]
    for time_value, U in sorted(displacements.items()):
        rows.append(_history_row("calculix_native_contact", model, float(time_value), U, stress_map=stresses.get(float(time_value), {})))
    return rows


def _interp_series(rows: list[Row], source: str, case_id: str, key: str, times: np.ndarray) -> np.ndarray:
    selected = [row for row in rows if row["source"] == source and row["case_id"] == case_id and row.get(key, "") != ""]
    if not selected:
        return np.full(times.shape, np.nan)
    xp = np.asarray([float(row["time_or_load"]) for row in selected], dtype=float)
    fp = np.asarray([float(row[key]) for row in selected], dtype=float)
    order = np.argsort(xp)
    return np.interp(times, xp[order], fp[order])


def _error_rows(history: list[Row], models: list[AlignmentModel]) -> list[Row]:
    rows: list[Row] = []
    for model in models:
        sfc_times = np.asarray([float(row["time_or_load"]) for row in history if row["case_id"] == model.case_id and row["source"] == "sfc_lagrangian_sdf"], dtype=float)
        if sfc_times.size == 0:
            continue
        for key in ("mean_uz", "max_abs_u", "max_von_mises", "max_strain_norm"):
            sfc = _interp_series(history, "sfc_lagrangian_sdf", model.case_id, key, sfc_times)
            ccx = _interp_series(history, "calculix_native_contact", model.case_id, key, sfc_times)
            mask = np.isfinite(sfc) & np.isfinite(ccx)
            if not np.any(mask):
                status = "missing_calculix"
                l2 = ""
                max_abs = ""
                final_rel = ""
                peak_rel = ""
            else:
                diff = sfc[mask] - ccx[mask]
                denom = max(float(np.linalg.norm(ccx[mask])), 1.0e-30)
                l2 = float(np.linalg.norm(diff) / denom)
                max_abs = float(np.max(np.abs(diff)))
                final_rel = float(abs(sfc[mask][-1] - ccx[mask][-1]) / max(abs(ccx[mask][-1]), 1.0e-30))
                sfc_peak = float(np.max(np.abs(sfc[mask])))
                ccx_peak = float(np.max(np.abs(ccx[mask])))
                peak_rel = float(abs(sfc_peak - ccx_peak) / max(abs(ccx_peak), 1.0e-30))
                status = "computed"
            rows.append(
                {
                    "case_id": model.case_id,
                    "metric": key,
                    "l2_relative_error": l2,
                    "max_abs_error": max_abs,
                    "final_relative_error": final_rel,
                    "peak_relative_error": peak_rel,
                    "status": status,
                }
            )
    return rows


def _plot_curves(history: list[Row], out_dir: Path, models: list[AlignmentModel]) -> list[Row]:
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_rows: list[Row] = []
    for metric, ylabel in (
        ("mean_uz", "mean z-displacement"),
        ("max_strain_norm", "max strain norm"),
        ("max_von_mises", "max von Mises stress"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
        for ax, model in zip(axes.ravel(), models, strict=True):
            for source, style in (("calculix_native_contact", "k-"), ("sfc_lagrangian_sdf", "C0--")):
                rows = [row for row in history if row["case_id"] == model.case_id and row["source"] == source]
                if not rows:
                    continue
                x = np.asarray([float(row["time_or_load"]) for row in rows], dtype=float)
                y = np.asarray([float(row[metric]) for row in rows], dtype=float)
                ax.plot(x, y, style, linewidth=1.4, label=source.replace("_", " "))
            ax.set_title(model.case_id.replace("_lagrangian_vs_calculix", "").replace("_", " "), fontsize=9)
            ax.set_xlabel("time" if model.regime == "dynamic" else "load factor")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, loc="best")
        png = figures / f"final_aim_calculix_alignment_{metric}.png"
        pdf = figures / f"final_aim_calculix_alignment_{metric}.pdf"
        fig.savefig(png, dpi=300)
        fig.savefig(pdf)
        plt.close(fig)
        plot_rows.append({"metric": metric, "png": str(png.relative_to(out_dir)), "pdf": str(pdf.relative_to(out_dir))})
    return plot_rows


def _write_summary(path: Path, errors: list[Row], commands: list[Row], plots: list[Row]) -> None:
    calculix_commands = [row for row in commands if row.get("solver") == "calculix"]
    completed = sum(1 for row in calculix_commands if str(row.get("completed", "false")).lower() == "true")
    lines = [
        "# Final-Aim Lagrangian SDF vs CalculiX Same-Case Alignment",
        "",
        "This stage generates CalculiX native contact inputs for the same small TET4/HEX8 static and dynamic cases used by the final-aim Lagrangian-SDF runner.",
        "",
        "## CalculiX completion",
        "",
        f"- Completed CalculiX cases: `{completed}/{len(calculix_commands)}`.",
        "",
        "## Errors",
        "",
        "| Case | Metric | L2 relative error | Max absolute error | Final rel. error | Peak rel. error | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in errors:
        lines.append(
            f"| `{row['case_id']}` | {row['metric']} | {row['l2_relative_error']} | {row['max_abs_error']} | "
            f"{row.get('final_relative_error', '')} | {row.get('peak_relative_error', '')} | {row['status']} |"
        )
    lines.extend(["", "## Figures", ""])
    for row in plots:
        lines.append(f"- `{row['png']}`")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- SFC uses `MaterialSDF + LagrangianSDFContactOracle`.",
            "- CalculiX is an external native-contact reference and is not imported by the core package.",
            "- This is an alignment stage; timing superiority is evaluated only after displacement/strain/stress errors are acceptable.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_alignment(out_dir: Path, *, quick: bool = False, skip_calculix: bool = False, timeout_seconds: int = 180) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [_alignment_model(element, regime, quick=quick) for element in ("tet4", "hex8") for regime in ("static", "dynamic")]
    history: list[Row] = []
    command_rows: list[Row] = []
    for model in models:
        if model.regime == "static":
            sfc_rows, _frames, command = _solve_sfc_static(model)
        else:
            sfc_rows, _frames, command = _solve_sfc_dynamic(model)
        history.extend(sfc_rows)
        command_rows.append(command)
        if skip_calculix:
            command_rows.append({"case_id": model.case_id, "solver": "calculix", "completed": "false", "command": "skipped", "wall_time_seconds": ""})
            continue
        displacements, stresses, command = _run_calculix(model, out_dir, timeout_seconds=timeout_seconds)
        command_rows.append(command)
        if displacements:
            history.extend(_calculix_history(model, displacements, stresses))
    errors = _error_rows(history, models)
    plots = _plot_curves(history, out_dir, models)
    outputs = {
        "history": out_dir / "final_aim_calculix_alignment_history.csv",
        "errors": out_dir / "final_aim_calculix_alignment_errors.csv",
        "commands": out_dir / "final_aim_calculix_alignment_commands.csv",
        "plots": out_dir / "final_aim_calculix_alignment_plots.csv",
        "summary": out_dir / "final_aim_calculix_alignment_summary.md",
    }
    _write_csv(outputs["history"], history)
    _write_csv(outputs["errors"], errors)
    _write_csv(outputs["commands"], command_rows)
    _write_csv(outputs["plots"], plots)
    _write_summary(outputs["summary"], errors, command_rows, plots)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-calculix", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "final_aim_calculix_alignment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_alignment(args.out_dir, quick=bool(args.quick), skip_calculix=bool(args.skip_calculix), timeout_seconds=int(args.timeout_seconds))
    print("Final-aim Lagrangian SDF vs CalculiX alignment complete.")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
