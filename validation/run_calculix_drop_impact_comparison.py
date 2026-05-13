"""CalculiX drop-impact reference comparison.

This validation-only runner uses CalculiX/ccx as an external transient dynamic
contact reference.  The runner generates a small deterministic TET4
sphere-like elastic body impacting a fixed rigid plane, then compares CalculiX
time-history data against the current SFC linear TET4 Newmark/penalty-contact
prototype on the same mesh, material, gravity, initial velocity, and rigid-plane
location.

The comparison is deliberately scoped.  It is external solver evidence for a
clean transient dynamic contact scenario, not a claim that the current SFC
prototype matches all CalculiX nonlinear contact details.
"""

from __future__ import annotations

import argparse
import csv
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import factorized, spsolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sfc.fem import DeformableBody, assemble_gravity_force, assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.fem.tet4 import tet4_volume  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.mesh.topology import orient_tet4_connectivity  # noqa: E402
from validation.run_phase4_paper_validation import _format_float  # noqa: E402

Row = dict[str, Any]

DEFAULT_CASE_NAME = "sfc_calculix_ball_drop"
CONTACT_ACTIVATION_TOLERANCE = 5.0e-3


@dataclass(slots=True)
class DropModel:
    """Generated drop-impact model shared by CalculiX and SFC."""

    node_ids: np.ndarray
    nodes: np.ndarray
    tet_elements: np.ndarray
    surface_node_ids: np.ndarray
    floor_nodes: np.ndarray
    floor_element_ids: np.ndarray
    floor_z: float
    E: float
    nu: float
    density: float
    gravity: float
    initial_velocity_z: float
    contact_stiffness: float
    total_time: float
    dt: float
    model_source: str

    @property
    def id_to_index(self) -> dict[int, int]:
        return {int(node_id): i for i, node_id in enumerate(self.node_ids)}

    @property
    def surface_indices(self) -> np.ndarray:
        lookup = self.id_to_index
        return np.asarray([lookup[int(node_id)] for node_id in self.surface_node_ids], dtype=np.int64)


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=timeout)


def _wsl_available() -> bool:
    if shutil.which("wsl") is None:
        return False
    try:
        _run(["wsl", "--exec", "bash", "-lc", "true"], cwd=ROOT, timeout=15)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def calculix_available() -> bool:
    """Return whether CalculiX/ccx is available through WSL."""

    if not _wsl_available():
        return False
    try:
        _run(["wsl", "--exec", "bash", "-lc", "command -v ccx >/dev/null"], cwd=ROOT, timeout=20)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _calculix_version() -> str:
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", "ccx -v 2>&1"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "unknown"


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    parts = [part for part in resolved.parts[1:]]
    if not drive:
        return resolved.as_posix()
    return "/mnt/" + drive + "/" + "/".join(part.replace("\\", "/") for part in parts)


def build_drop_model(*, quick: bool) -> DropModel:
    """Build the shared CalculiX/SFC drop-impact model."""

    radius = 0.5
    initial_gap = 0.05
    floor_z = 0.0
    center_z = floor_z + initial_gap + radius
    node_ids = np.arange(1, 8, dtype=np.int64)
    nodes = np.asarray(
        [
            [0.0, 0.0, center_z],  # center
            [0.0, 0.0, center_z + radius],  # top
            [0.0, 0.0, center_z - radius],  # bottom
            [radius, 0.0, center_z],
            [0.0, radius, center_z],
            [-radius, 0.0, center_z],
            [0.0, -radius, center_z],
        ],
        dtype=float,
    )
    raw_tets = np.asarray(
        [
            [0, 1, 3, 4],
            [0, 1, 4, 5],
            [0, 1, 5, 6],
            [0, 1, 6, 3],
            [0, 2, 4, 3],
            [0, 2, 5, 4],
            [0, 2, 6, 5],
            [0, 2, 3, 6],
        ],
        dtype=np.int64,
    )
    tet_elements = orient_tet4_connectivity(nodes, raw_tets)
    surface_node_ids = np.asarray([2, 3, 4, 5, 6, 7], dtype=np.int64)
    floor_half_width = 1.1
    floor_nodes = np.asarray(
        [
            [101, -floor_half_width, -floor_half_width, floor_z],
            [102, floor_half_width, -floor_half_width, floor_z],
            [103, floor_half_width, floor_half_width, floor_z],
            [104, -floor_half_width, floor_half_width, floor_z],
        ],
        dtype=float,
    )
    floor_elements = np.asarray([[0, 101, 102, 103, 104]], dtype=np.int64)

    total_time = 0.08 if quick else 0.16
    dt = 0.002 if quick else 0.001
    initial_velocity_z = -2.0

    return DropModel(
        node_ids=node_ids,
        nodes=nodes,
        tet_elements=tet_elements,
        surface_node_ids=surface_node_ids,
        floor_nodes=floor_nodes,
        floor_element_ids=floor_elements,
        floor_z=floor_z,
        E=1000.0,
        nu=0.3,
        density=1.0,
        gravity=9.81,
        initial_velocity_z=initial_velocity_z,
        contact_stiffness=2.0e4,
        total_time=total_time,
        dt=dt,
        model_source="generated_octahedral_tet4_sphere",
    )


def _format_id_list(ids: np.ndarray, *, width: int = 10) -> list[str]:
    values = [str(int(value)) for value in ids]
    return [", ".join(values[i : i + width]) + "," for i in range(0, len(values), width)]


def write_calculix_input(model: DropModel, path: Path) -> None:
    """Write a deterministic CalculiX C3D4 drop-impact input file."""

    id_from_index = model.node_ids
    lines: list[str] = [
        "** Generated by validation/run_calculix_drop_impact_comparison.py",
        "** External reference: CalculiX/ccx transient dynamic node-to-surface contact.",
        "*NODE, NSET=Nall",
    ]
    for node_id, coord in zip(model.node_ids, model.nodes, strict=True):
        lines.append(f"{int(node_id)}, {coord[0]:.12e}, {coord[1]:.12e}, {coord[2]:.12e}")

    lines.append("*ELEMENT, TYPE=C3D4, ELSET=elall")
    for element_id, tet in enumerate(model.tet_elements, start=1):
        ids = [int(id_from_index[int(i)]) for i in tet]
        lines.append(f"{element_id}, {ids[0]}, {ids[1]}, {ids[2]}, {ids[3]}")

    lines.append("*node, nset=nfloor")
    for row in model.floor_nodes:
        lines.append(f"{int(row[0])}, {row[1]:.12e}, {row[2]:.12e}, {row[3]:.12e}")
    lines.append("*element, type=s4, elset=efloor")
    floor_element_start = model.tet_elements.shape[0] + 1
    for offset, row in enumerate(model.floor_element_ids):
        values = [floor_element_start + offset] + [int(value) for value in row[1:]]
        lines.append(", ".join(str(int(value)) for value in values))

    lines.append("*NSET,NSET=nsurface")
    lines.extend(_format_id_list(model.surface_node_ids))
    lines.extend(
        [
            "*time points, name=times, generate",
            f"0.0, {model.total_time:.12g}, {model.dt:.12g}",
            "*boundary",
            "nfloor, 1, 6,",
            "*material, name=gummi",
            "*elastic",
            f"{model.E:.12g}, {model.nu:.12g}",
            "*density",
            f"{model.density:.12g}",
            "*solid section, elset=elall, material=gummi",
            "*shell section, elset=efloor, material=gummi",
            "0.01",
            "*initial conditions, type=velocity",
            f"nall, 3, {model.initial_velocity_z:.12g}",
            "*surface, name=floor, type=element",
            "efloor, SPOS",
            "*surface, name=ball, type=node",
            "nsurface",
            "*contact pair, interaction=contact,TYPE=NODE TO SURFACE",
            "ball, floor",
            "*surface interaction, name=contact",
            "*surface behavior, pressure-overclosure=linear",
            f"{model.contact_stiffness:.12g}",
            "*step, nlgeom, inc=1000000",
            "*dynamic",
            f"{model.dt:.12g}, {model.total_time:.12g}, 1.0e-9, {model.dt:.12g}",
            "*dload",
            f"elall, grav, {model.gravity:.12g}, 0.0, 0.0, -1.0",
            "*node print, nset=nall, time points=times",
            "u",
            "*endstep",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_calculix(model: DropModel, out_dir: Path, *, case_name: str = DEFAULT_CASE_NAME) -> tuple[Path, Row]:
    """Run CalculiX through WSL and return the generated .dat path."""

    if not calculix_available():
        raise SystemExit("CalculiX/ccx is not available in WSL.")

    run_dir = out_dir / "calculix_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    inp_path = run_dir / f"{case_name}.inp"
    write_calculix_input(model, inp_path)

    wsl_run_dir = _wsl_path(run_dir)
    command = f"cd {wsl_run_dir} && ccx {case_name}"
    proc = _run(["wsl", "--exec", "bash", "-lc", command], cwd=ROOT, timeout=240)
    (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")

    dat_path = run_dir / f"{case_name}.dat"
    if not dat_path.is_file():
        raise RuntimeError(f"CalculiX did not produce {dat_path}")

    command_row = {
        "external_solver": "CalculiX",
        "external_solver_version": _calculix_version(),
        "command": f"wsl --exec bash -lc \"cd {wsl_run_dir} && ccx {case_name}\"",
        "input_file": str(inp_path.relative_to(out_dir)),
        "dat_file": str(dat_path.relative_to(out_dir)),
        "stdout_log": str((run_dir / "calculix_stdout.log").relative_to(out_dir)),
        "stderr_log": str((run_dir / "calculix_stderr.log").relative_to(out_dir)),
        "return_code": proc.returncode,
    }
    return dat_path, command_row


def _parse_calculix_dat_displacements(path: Path, node_ids: np.ndarray) -> dict[float, np.ndarray]:
    """Parse CalculiX node-print displacement blocks from a .dat file."""

    id_to_row = {int(node_id): i for i, node_id in enumerate(node_ids)}
    blocks: dict[float, np.ndarray] = {}
    current_time: float | None = None
    current = np.zeros((len(node_ids), 3), dtype=float)
    seen = 0
    header = re.compile(r"displacements.*time\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE)
    row_pattern = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    )

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = header.search(line)
        if match:
            if current_time is not None and seen:
                blocks[current_time] = current.copy()
            current_time = float(match.group(1))
            current = np.zeros((len(node_ids), 3), dtype=float)
            seen = 0
            continue

        if current_time is None:
            continue
        row = row_pattern.match(line)
        if row is None:
            continue
        node_id = int(row.group(1))
        if node_id not in id_to_row:
            continue
        current[id_to_row[node_id], :] = [float(row.group(2)), float(row.group(3)), float(row.group(4))]
        seen += 1

    if current_time is not None and seen:
        blocks[current_time] = current.copy()
    if not blocks:
        raise RuntimeError(f"no displacement blocks parsed from {path}")
    return dict(sorted(blocks.items()))


def _history_from_displacements(
    source: str,
    model: DropModel,
    displacement_blocks: dict[float, np.ndarray],
) -> list[Row]:
    """Convert displacement blocks into comparable drop-impact diagnostics."""

    surface = model.surface_indices
    rows: list[Row] = []
    times = [0.0] + [float(t) for t in displacement_blocks]
    displacements = [np.zeros_like(model.nodes)] + [displacement_blocks[t] for t in displacement_blocks]
    previous_z_cm: float | None = None
    previous_time: float | None = None
    for time, u_nodes in zip(times, displacements, strict=True):
        current = model.nodes + u_nodes
        z_cm = float(np.mean(current[:, 2]))
        if previous_z_cm is None:
            v_cm_z = model.initial_velocity_z
        else:
            v_cm_z = (z_cm - previous_z_cm) / max(float(time) - float(previous_time), 1.0e-30)
        gaps = current[surface, 2] - model.floor_z
        penetration = np.maximum(-gaps, 0.0)
        active_mask = penetration > 0.0
        if source == "calculix":
            active_mask = np.logical_or(active_mask, gaps <= CONTACT_ACTIVATION_TOLERANCE)
        normal_force_proxy = float(model.contact_stiffness * np.sum(penetration))
        rows.append(
            {
                "source": source,
                "time": float(time),
                "z_cm": z_cm,
                "v_cm_z": float(v_cm_z),
                "min_gap": float(np.min(gaps)),
                "max_penetration": float(np.max(penetration)),
                "active_contact_count": int(np.count_nonzero(active_mask)),
                "normal_force_proxy": normal_force_proxy,
                "kinetic_energy_proxy": "",
                "strain_energy": "",
                "contact_energy_proxy": float(0.5 * model.contact_stiffness * np.sum(penetration**2)),
                "details": "postprocessed from CalculiX displacement output; active contact uses gap tolerance"
                if source == "calculix"
                else "computed by SFC Newmark linear TET4 prototype",
            }
        )
        previous_z_cm = z_cm
        previous_time = float(time)
    return rows


def _total_mass(model: DropModel) -> float:
    return float(model.density * sum(tet4_volume(model.nodes[element]) for element in model.tet_elements))


def _fill_acceleration_force_proxy(rows: list[Row], model: DropModel) -> None:
    """Infer a normal-force proxy from center-of-mass acceleration."""

    total_mass = _total_mass(model)
    by_source: dict[str, list[Row]] = {}
    for row in rows:
        by_source.setdefault(str(row["source"]), []).append(row)

    for source_rows in by_source.values():
        source_rows.sort(key=lambda row: float(row["time"]))
        for i, row in enumerate(source_rows):
            if i == 0:
                continue
            previous = source_rows[i - 1]
            dt = float(row["time"]) - float(previous["time"])
            if dt <= 0.0:
                continue
            acceleration = (float(row["v_cm_z"]) - float(previous["v_cm_z"])) / dt
            force_proxy = total_mass * (acceleration + model.gravity)
            if str(row["source"]) == "calculix" and int(row["active_contact_count"]) > 0:
                row["normal_force_proxy"] = float(max(force_proxy, 0.0))


def _plane_contact_force(
    x_current: np.ndarray,
    surface_indices: np.ndarray,
    *,
    floor_z: float,
    stiffness: float,
) -> tuple[np.ndarray, float, int, float, float]:
    """Return nodal force for a frictionless rigid horizontal plane."""

    force = np.zeros(3 * x_current.shape[0], dtype=float)
    gaps = x_current[surface_indices, 2] - floor_z
    penetration = np.maximum(-gaps, 0.0)
    for node, depth in zip(surface_indices, penetration, strict=True):
        if depth > 0.0:
            force[3 * int(node) + 2] += stiffness * float(depth)
    return (
        force,
        float(np.min(gaps)),
        int(np.count_nonzero(penetration > 0.0)),
        float(np.max(penetration)),
        float(0.5 * stiffness * np.sum(penetration**2)),
    )


def run_sfc_drop_history(model: DropModel) -> list[Row]:
    """Run the matching SFC TET4 Newmark/penalty-contact drop model."""

    mesh = VolumeMesh(
        model.nodes,
        model.tet_elements,
        node_sets={"surface": model.surface_indices},
    )
    body = DeformableBody(mesh, {"E": model.E, "nu": model.nu}, density=model.density)
    K = assemble_stiffness_matrix(body).tocsr()
    M = assemble_mass_matrix(body, kind="lumped").tocsr()
    C = csr_matrix(K.shape, dtype=float)
    f_gravity = assemble_gravity_force(body, (0.0, 0.0, -model.gravity))

    n_dofs = body.n_dofs
    u = np.zeros(n_dofs, dtype=float)
    v = np.zeros(n_dofs, dtype=float)
    v[2::3] = model.initial_velocity_z
    x_current = model.nodes + u.reshape((-1, 3))
    f_contact, min_gap, active_count, max_penetration, contact_energy = _plane_contact_force(
        x_current,
        model.surface_indices,
        floor_z=model.floor_z,
        stiffness=model.contact_stiffness,
    )
    a = np.asarray(spsolve(M, f_gravity + f_contact - K @ u - C @ v), dtype=float)

    dt = model.dt
    beta = 0.25
    gamma = 0.5
    c0 = 1.0 / (beta * dt * dt)
    c1 = gamma / (beta * dt)
    K_eff = csc_matrix(K + c0 * M + c1 * C)
    solve_eff = factorized(K_eff)
    times = np.arange(0.0, model.total_time + 0.5 * dt, dt)
    rows: list[Row] = []

    for step, time in enumerate(times):
        x_current = model.nodes + u.reshape((-1, 3))
        f_contact, min_gap, active_count, max_penetration, contact_energy = _plane_contact_force(
            x_current,
            model.surface_indices,
            floor_z=model.floor_z,
            stiffness=model.contact_stiffness,
        )
        z_cm = float(np.mean(x_current[:, 2]))
        rows.append(
            {
                "source": "sfc",
                "time": float(time),
                "z_cm": z_cm,
                "v_cm_z": float(np.mean(v.reshape((-1, 3))[:, 2])),
                "min_gap": min_gap,
                "max_penetration": max_penetration,
                "active_contact_count": active_count,
                "normal_force_proxy": float(np.sum(f_contact[2::3])),
                "kinetic_energy_proxy": float(0.5 * v @ (M @ v)),
                "strain_energy": float(0.5 * u @ (K @ u)),
                "contact_energy_proxy": contact_energy,
                "details": "computed by SFC Newmark linear TET4 prototype",
            }
        )
        if step == len(times) - 1:
            break

        u_pred = u + dt * v + dt * dt * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a
        rhs = f_gravity + f_contact + M @ (c0 * u_pred) + C @ (c1 * u_pred - v_pred)
        u_new = np.asarray(solve_eff(rhs), dtype=float)
        a_new = c0 * (u_new - u_pred)
        v_new = v_pred + gamma * dt * a_new
        u, v, a = u_new, v_new, a_new

    return rows


def _first_contact_time(rows: list[Row]) -> float | None:
    active = [row for row in rows if int(row["active_contact_count"]) > 0 or float(row["max_penetration"]) > 0.0]
    return None if not active else float(active[0]["time"])


def comparison_metrics(calculix_rows: list[Row], sfc_rows: list[Row]) -> list[Row]:
    """Return summary metrics comparing CalculiX and SFC time histories."""

    cx_by_time = {round(float(row["time"]), 12): row for row in calculix_rows}
    sfc_by_time = {round(float(row["time"]), 12): row for row in sfc_rows}
    common_times = sorted(set(cx_by_time).intersection(sfc_by_time))
    z_errors = np.asarray([float(sfc_by_time[t]["z_cm"]) - float(cx_by_time[t]["z_cm"]) for t in common_times], dtype=float)
    gap_errors = np.asarray([float(sfc_by_time[t]["min_gap"]) - float(cx_by_time[t]["min_gap"]) for t in common_times], dtype=float)

    cx_first = _first_contact_time(calculix_rows)
    sfc_first = _first_contact_time(sfc_rows)
    first_contact_abs_error = "" if cx_first is None or sfc_first is None else abs(sfc_first - cx_first)
    both_contact = cx_first is not None and sfc_first is not None
    trajectory_rel = float(np.linalg.norm(z_errors) / max(np.linalg.norm([float(cx_by_time[t]["z_cm"]) for t in common_times]), 1.0e-30))
    gap_linf = float(np.max(np.abs(gap_errors))) if gap_errors.size else float("nan")
    status = "supported" if both_contact and first_contact_abs_error != "" and float(first_contact_abs_error) <= 3.0e-3 else "check"

    return [
        {
            "metric": "external_solver",
            "value": "CalculiX",
            "status": "evidence",
            "details": "external transient dynamic contact solver used through WSL ccx",
        },
        {
            "metric": "first_contact_time_calculix",
            "value": "" if cx_first is None else cx_first,
            "status": "evidence",
            "details": "first time with positive penetration or active contact count",
        },
        {
            "metric": "first_contact_time_sfc",
            "value": "" if sfc_first is None else sfc_first,
            "status": "evidence",
            "details": "first time with positive penetration or active contact count",
        },
        {
            "metric": "first_contact_time_abs_error",
            "value": first_contact_abs_error,
            "status": status,
            "details": "gate uses 3 ms tolerance in quick generated model",
        },
        {
            "metric": "z_cm_l2_relative_error",
            "value": trajectory_rel,
            "status": "evidence",
            "details": "center-of-mass trajectory difference over common output times",
        },
        {
            "metric": "min_gap_linf_abs_error",
            "value": gap_linf,
            "status": "evidence",
            "details": "minimum gap difference over common output times",
        },
        {
            "metric": "external_dynamic_contact_claim",
            "value": "supported" if status == "supported" else "not_supported",
            "status": "supported" if status == "supported" else "not_supported",
            "details": "supported only means both solvers activate contact at matching time scale on the generated model",
        },
    ]


def write_plots(out_dir: Path, history_rows: list[Row]) -> list[Row]:
    """Write paper-facing diagnostic plots for the external drop comparison."""

    out_dir.mkdir(parents=True, exist_ok=True)
    plots: list[Row] = []
    sources = sorted({str(row["source"]) for row in history_rows})

    def series(source: str, field: str) -> tuple[list[float], list[float]]:
        rows = [row for row in history_rows if row["source"] == source]
        return [float(row["time"]) for row in rows], [float(row[field]) for row in rows]

    for field, ylabel, name in [
        ("z_cm", "center-of-mass z", "calculix_drop_z_cm"),
        ("min_gap", "minimum gap to rigid plane", "calculix_drop_min_gap"),
        ("normal_force_proxy", "normal force proxy", "calculix_drop_force_proxy"),
        ("contact_energy_proxy", "contact energy proxy", "calculix_drop_contact_energy"),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for source in sources:
            x, y = series(source, field)
            ax.plot(x, y, marker="o", markersize=3.0, linewidth=1.2, label=source)
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, color="0.9", linewidth=0.5)
        fig.tight_layout()
        png = out_dir / f"{name}.png"
        pdf = out_dir / f"{name}.pdf"
        fig.savefig(png, dpi=180)
        fig.savefig(pdf)
        plt.close(fig)
        plots.append({"plot": name, "png": png.name, "pdf": pdf.name, "status": "ok"})

    return plots


def _fieldnames(rows: list[Row]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def write_markdown(
    path: Path,
    *,
    model: DropModel,
    command_row: Row,
    metric_rows: list[Row],
    plots: list[Row],
    quick: bool,
) -> None:
    claim = next(row for row in metric_rows if row["metric"] == "external_dynamic_contact_claim")
    lines = [
        "# CalculiX Drop-Impact Comparison",
        "",
        "This validation uses CalculiX/ccx as an external open-source transient dynamic contact reference. The generated model is a small vertical elastic ball/drop impact against a fixed rigid plane, represented by a deterministic octahedral TET4 mesh so that CalculiX and SFC use the same TET4 connectivity.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python validation/run_calculix_drop_impact_comparison.py {'--quick ' if quick else ''}--out-dir {path.parent.as_posix()}",
        "```",
        "",
        "## External Solver",
        "",
        f"- Solver: CalculiX/ccx",
        f"- Version: `{command_row['external_solver_version']}`",
        f"- Command: `{command_row['command']}`",
        f"- Model source: `{model.model_source}`",
        "",
        "## Model",
        "",
        f"- Nodes: {model.nodes.shape[0]}",
        f"- TET4 elements: {model.tet_elements.shape[0]}",
        f"- Surface nodes: {model.surface_node_ids.size}",
        f"- Total time: {_format_float(model.total_time)}",
        f"- Time step: {_format_float(model.dt)}",
        f"- Initial vertical velocity: {_format_float(model.initial_velocity_z)}",
        f"- Rigid plane z: {_format_float(model.floor_z)}",
        f"- Linear material: E={_format_float(model.E)}, nu={_format_float(model.nu)}, density={_format_float(model.density)}",
        "",
        "## Claim Gate",
        "",
        "| Claim | Status | Evidence | Gate |",
        "| --- | --- | --- | --- |",
        f"| external dynamic contact time-scale agreement | {claim['status']} | calculix_drop_metrics.csv::external_dynamic_contact_claim | {claim['value']} |",
        "<!-- evidence csv=calculix_drop_metrics.csv field=value -->",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Status | Details |",
        "| --- | ---: | --- | --- |",
    ]
    for row in metric_rows:
        lines.append(f"| {row['metric']} | {row['value']} | {row['status']} | {row['details']} |")
    lines.extend(["", "## Plots", ""])
    for plot in plots:
        lines.append(f"- `{plot['png']}` and `{plot['pdf']}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- CalculiX performs an independent transient dynamic contact solve.",
            "- SFC uses the same generated TET4 geometry with its current linear TET4 Newmark prototype and a normal penalty rigid-plane contact force.",
            "- The force curves are proxy quantities derived from the common gap and configured penalty stiffness; they are not claimed to be identical CalculiX reaction-force output.",
            "- The accepted claim is limited to contact activation and trajectory diagnostics for this generated case. It is not a validation of friction, self-contact, nonlinear FEM, or production contact algorithms.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_comparison(out_dir: Path, *, quick: bool) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_drop_model(quick=quick)
    dat_path, command_row = run_calculix(model, out_dir)
    displacement_blocks = _parse_calculix_dat_displacements(dat_path, model.node_ids)
    calculix_rows = _history_from_displacements("calculix", model, displacement_blocks)
    _fill_acceleration_force_proxy(calculix_rows, model)
    sfc_rows = run_sfc_drop_history(model)
    history_rows = sorted(calculix_rows + sfc_rows, key=lambda row: (float(row["time"]), str(row["source"])))
    metric_rows = comparison_metrics(calculix_rows, sfc_rows)
    plots = write_plots(out_dir, history_rows)

    history_csv = out_dir / "calculix_drop_time_history.csv"
    metrics_csv = out_dir / "calculix_drop_metrics.csv"
    commands_csv = out_dir / "calculix_drop_commands.csv"
    plots_csv = out_dir / "calculix_drop_plots.csv"
    metadata_csv = out_dir / "calculix_drop_metadata.csv"
    summary_md = out_dir / "calculix_drop_summary.md"

    _write_csv(history_csv, _fieldnames(history_rows), history_rows)
    _write_csv(metrics_csv, _fieldnames(metric_rows), metric_rows)
    _write_csv(commands_csv, _fieldnames([command_row]), [command_row])
    _write_csv(plots_csv, _fieldnames(plots), plots)
    _write_csv(
        metadata_csv,
        ["key", "value"],
        [
            {"key": "python", "value": sys.version.split()[0]},
            {"key": "platform", "value": platform.platform()},
            {"key": "calculix_version", "value": command_row["external_solver_version"]},
            {"key": "quick", "value": str(bool(quick)).lower()},
            {"key": "nodes", "value": model.nodes.shape[0]},
            {"key": "tet4_elements", "value": model.tet_elements.shape[0]},
            {"key": "surface_nodes", "value": model.surface_node_ids.size},
        ],
    )
    write_markdown(summary_md, model=model, command_row=command_row, metric_rows=metric_rows, plots=plots, quick=quick)
    return {
        "history": history_csv,
        "metrics": metrics_csv,
        "commands": commands_csv,
        "plots": plots_csv,
        "metadata": metadata_csv,
        "summary": summary_md,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a short deterministic case for CI and smoke checks.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "calculix_drop_impact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_comparison(args.out_dir, quick=bool(args.quick))
    print("CalculiX drop-impact comparison complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
