"""Long-time C3D8 dynamic SDF-contact validation.

This runner is a paper-facing model/visualization driver.  It does not modify
the core contact algorithm.  The default case runs a 3 s deformable
block-on-block C3D8 dynamic contact trajectory at resolution 4 and exports time
histories plus fine-grid final-state contact clouds.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter
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
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from sfc.contact.narrow_phase import SurfaceSample  # noqa: E402
from sfc.fem.hex8 import hex8_center_strain_stress  # noqa: E402
from sfc.fem.assembler import assemble_gravity_force, assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.fem.body import DeformableBody  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.contact.field_contact import field_contact_constraint_from_sample  # noqa: E402
from validation.run_c3d8_contact_trajectory_validation import (  # noqa: E402
    QUAD_GAUSS,
    _build_trajectory_master_field,
    _master_query_geometry,
    _orient_quad_outward,
    _quad_shape,
    _surface_quads,
    build_c3d8_model,
)
from validation.run_phase9_full_contact_validation import (  # noqa: E402
    _contact_geometry_for_model,
    _fixed_free,
    _linear_contact_response,
    _model_fixed_dofs,
    _von_mises,
    _write_legacy_hex_vtk,
)

Row = dict[str, Any]

_HEX_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


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


def _boundary_hex_faces(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    face_owner: dict[tuple[int, ...], tuple[np.ndarray, int]] = {}
    counts: dict[tuple[int, ...], int] = {}
    for element_id, element in enumerate(np.asarray(elements, dtype=np.int64)):
        for face_nodes in _HEX_FACES:
            face = np.asarray([element[i] for i in face_nodes], dtype=np.int64)
            key = tuple(sorted(int(node) for node in face))
            counts[key] = counts.get(key, 0) + 1
            face_owner.setdefault(key, (face, element_id))
    faces: list[np.ndarray] = []
    owners: list[int] = []
    for key, count in counts.items():
        if count == 1:
            face, owner = face_owner[key]
            faces.append(face)
            owners.append(owner)
    return np.asarray(faces, dtype=np.int64), np.asarray(owners, dtype=np.int64)


def _element_fields(model: Any, displacement: np.ndarray) -> dict[str, np.ndarray]:
    U = np.asarray(displacement, dtype=float)
    disp_mag = np.asarray([float(np.mean(np.linalg.norm(U[element], axis=1))) for element in model.elements], dtype=float)
    strain_norm: list[float] = []
    von_mises: list[float] = []
    for element in model.elements:
        strain, stress = hex8_center_strain_stress(model.X[element], U[element], model.E, model.nu)
        strain_norm.append(float(np.linalg.norm(strain)))
        von_mises.append(_von_mises(stress))
    return {
        "displacement_magnitude": disp_mag,
        "engineering_strain_norm": np.asarray(strain_norm, dtype=float),
        "von_mises": np.asarray(von_mises, dtype=float),
    }


def _long_time_linear_dynamic_history(
    model: Any,
    *,
    damping_alpha: float = 1.5,
) -> tuple[list[Row], np.ndarray, np.ndarray]:
    """Run a damped small-strain C3D8 Newmark trajectory with SDF contact."""

    mesh = VolumeMesh(model.X, model.elements, element_type="C3D8")
    body = DeformableBody(mesh=mesh, material={"E": model.E, "nu": model.nu}, density=model.density)
    K = assemble_stiffness_matrix(body).tocsr()
    M = assemble_mass_matrix(body, kind="consistent").tocsr()
    f_ext = assemble_gravity_force(body, (0.0, 0.0, -model.gravity))
    contact = _contact_geometry_for_model(model)
    n_dofs = body.n_dofs
    fixed = _model_fixed_dofs(model)
    free = _fixed_free(n_dofs, fixed)
    beta = 0.25
    gamma = 0.5
    dt = float(model.dt)
    c0 = 1.0 / (beta * dt * dt)
    cv = gamma * dt * c0
    u = np.zeros(n_dofs, dtype=float)
    v = np.zeros(n_dofs, dtype=float)
    v[3 * model.slave_node_indices + 2] = float(model.initial_velocity_z)
    if fixed.size:
        u[fixed] = 0.0
        v[fixed] = 0.0
    initial_contact = _linear_contact_response(contact, model.X + u.reshape((-1, 3)), body.n_nodes)
    initial_rhs = f_ext + initial_contact.force.reshape(-1) - K @ u - float(damping_alpha) * (M @ v)
    a = np.zeros(n_dofs, dtype=float)
    if fixed.size:
        a[free] = np.asarray(spsolve(M[free[:, None], free].tocsc(), initial_rhs[free]), dtype=float)
    else:
        a = np.asarray(spsolve(M.tocsc(), initial_rhs), dtype=float)
    rows: list[Row] = []
    snapshot_u = u.copy()
    snapshot_vm_force = -np.inf
    steps = int(np.ceil(float(model.total_time) / dt))
    for step in range(steps + 1):
        x_current = model.X + u.reshape((-1, 3))
        contact_response = _linear_contact_response(contact, x_current, body.n_nodes)
        normal_force = float(abs(contact_response.normal_force))
        if normal_force >= snapshot_vm_force:
            snapshot_vm_force = normal_force
            snapshot_u = u.copy()
        rows.append(
            {
                "case": model.case,
                "resolution": model.resolution,
                "time": float(step * dt),
                "z_cm": float(np.mean(x_current[model.slave_node_indices, 2])),
                "min_gap": float(contact_response.min_gap),
                "max_penetration": float(contact_response.max_penetration),
                "active_contact_count": int(contact_response.active_count),
                "normal_force": normal_force,
                "contact_energy": float(contact_response.energy),
                "sdf_replay_min_gap": float(contact_response.min_gap),
                "sdf_replay_normal_force": float(abs(contact_response.normal_force)),
                "newton_iterations": 0,
                "newton_residual_norm": 0.0,
            }
        )
        if step == steps:
            break
        u_pred = u + dt * v + dt * dt * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a
        if fixed.size:
            u_pred[fixed] = 0.0
            v_pred[fixed] = 0.0
        u_guess = u_pred.copy()
        residual_norm = np.inf
        iterations = 0
        for iteration in range(14):
            x_guess = model.X + u_guess.reshape((-1, 3))
            trial_contact = _linear_contact_response(contact, x_guess, body.n_nodes)
            a_guess = c0 * (u_guess - u_pred)
            v_guess = v_pred + gamma * dt * a_guess
            residual = M @ a_guess + float(damping_alpha) * (M @ v_guess) + K @ u_guess - f_ext - trial_contact.force.reshape(-1)
            residual_norm = float(np.linalg.norm(residual[free]))
            tangent = (M * (c0 + float(damping_alpha) * cv) + K + trial_contact.tangent).tocsc()
            correction = np.zeros(n_dofs, dtype=float)
            correction[free] = np.asarray(spsolve(tangent[free[:, None], free], -residual[free]), dtype=float)
            u_guess[free] += correction[free]
            if fixed.size:
                u_guess[fixed] = 0.0
            iterations = iteration + 1
            if float(np.linalg.norm(correction[free])) <= 1.0e-10 * max(1.0, float(np.linalg.norm(u_guess[free]))):
                break
        u = u_guess
        a = c0 * (u - u_pred)
        v = v_pred + gamma * dt * a
        if fixed.size:
            u[fixed] = 0.0
            v[fixed] = 0.0
            a[fixed] = 0.0
        rows[-1]["newton_iterations"] = iterations
        rows[-1]["newton_residual_norm"] = residual_norm
    U_snapshot = snapshot_u.reshape((-1, 3))
    vm_snapshot = _element_fields(model, U_snapshot)["von_mises"]
    return rows, U_snapshot, np.asarray(vm_snapshot, dtype=float)


def _contact_cell_fields(model: Any, displacement: np.ndarray) -> dict[str, np.ndarray]:
    x_current = model.X + np.asarray(displacement, dtype=float)
    query_nodes, master_faces = _master_query_geometry(model, x_current)
    master_sdf = _build_trajectory_master_field(query_nodes, master_faces)
    candidates = np.arange(master_faces.shape[0], dtype=np.int64)
    n_elements = model.elements.shape[0]
    element_index = {tuple(int(node) for node in element): idx for idx, element in enumerate(model.elements)}
    gap_sum = np.zeros(n_elements, dtype=float)
    pressure_sum = np.zeros(n_elements, dtype=float)
    active = np.zeros(n_elements, dtype=float)
    counts = np.zeros(n_elements, dtype=float)
    for quad, element in _surface_quads(model, model.slave_surface):
        face = _orient_quad_outward(x_current, quad, element)
        coords = x_current[face]
        eid = int(element_index[tuple(int(node) for node in element)])
        for xi in QUAD_GAUSS:
            for eta in QUAD_GAUSS:
                shape, _dxi, _deta = _quad_shape(xi, eta)
                sample = SurfaceSample(
                    node_ids=np.asarray(face, dtype=np.int64),
                    weights=np.asarray(shape, dtype=float),
                    candidate_face_ids=candidates,
                )
                constraint = field_contact_constraint_from_sample(x_current, sample, master_sdf)
                gap = float(constraint.g)
                pressure = model.contact_stiffness * max(-gap, 0.0)
                gap_sum[eid] += gap
                pressure_sum[eid] += pressure
                active[eid] = max(active[eid], 1.0 if pressure > 0.0 else 0.0)
                counts[eid] += 1.0
    denom = np.maximum(counts, 1.0)
    return {
        "sdf_gap": gap_sum / denom,
        "contact_pressure": pressure_sum / denom,
        "active_contact": active,
    }


def _plot_histories(out_dir: Path, rows: list[Row]) -> dict[str, Path]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    t = np.asarray([float(row["time"]) for row in rows], dtype=float)
    z0 = float(rows[0]["z_cm"]) if rows else 0.0
    z = np.asarray([float(row["z_cm"]) - z0 for row in rows], dtype=float)
    force = np.asarray([float(row["normal_force"]) for row in rows], dtype=float)
    energy = np.asarray([float(row["contact_energy"]) for row in rows], dtype=float)
    gap = np.asarray([float(row["min_gap"]) for row in rows], dtype=float)
    active = np.asarray([float(row["active_contact_count"]) for row in rows], dtype=float)
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 8.0), sharex=True)
    series = [
        (z, r"$z$ displacement (mm)"),
        (force, "normal force"),
        (energy, "contact energy"),
        (gap, "min gap (mm)"),
        (active, "active contact samples"),
    ]
    for ax, (values, ylabel) in zip(axes.ravel(), series, strict=False):
        ax.plot(t, values, color="#4c78a8", linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28)
    axes.ravel()[-1].set_axis_off()
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    png = figures / "long_time_c3d8_dynamic_histories.png"
    pdf = figures / "long_time_c3d8_dynamic_histories.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"history_png": png, "history_pdf": pdf}


def _plot_clouds(out_dir: Path, model: Any, displacement: np.ndarray, fields: dict[str, np.ndarray]) -> dict[str, Path]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    faces, owners = _boundary_hex_faces(model.elements)
    x_ref = np.asarray(model.X, dtype=float)
    x_plot = x_ref + np.asarray(displacement, dtype=float)
    panels = [
        ("displacement_magnitude", "displacement magnitude"),
        ("engineering_strain_norm", "strain norm"),
        ("von_mises", "von Mises"),
        ("sdf_gap", "SDF gap"),
        ("contact_pressure", "contact pressure"),
        ("active_contact", "active contact"),
    ]
    fig = plt.figure(figsize=(10.0, 6.8))
    for idx, (key, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        values = np.asarray(fields[key], dtype=float)
        face_values = values[owners]
        vmin = float(np.min(face_values))
        vmax = float(np.max(face_values))
        if abs(vmax - vmin) < 1.0e-14:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        collection = Poly3DCollection(
            [x_plot[face] for face in faces],
            cmap="viridis",
            norm=norm,
            linewidths=0.10,
            edgecolors=(0.1, 0.1, 0.1, 0.18),
        )
        collection.set_array(face_values)
        ax.add_collection3d(collection)
        bounds = x_plot[faces.reshape(-1)]
        mins = bounds.min(axis=0)
        maxs = bounds.max(axis=0)
        pad = np.maximum(0.05 * (maxs - mins), 1.0e-4)
        ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
        ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
        ax.set_zlim(float(mins[2] - pad[2]), float(maxs[2] + pad[2]))
        ax.set_box_aspect((maxs - mins + 2.0 * pad).clip(min=1.0e-12))
        ax.view_init(elev=22.0, azim=-48.0)
        ax.set_axis_off()
        ax.set_title(title, pad=3)
        fig.colorbar(collection, ax=ax, shrink=0.62, pad=0.02)
    fig.tight_layout()
    png = figures / "fine_grid_c3d8_contact_clouds.png"
    pdf = figures / "fine_grid_c3d8_contact_clouds.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"cloud_png": png, "cloud_pdf": pdf}


def run_validation(
    out_dir: Path,
    *,
    case: str = "block_block_c3d8",
    resolution: int = 4,
    total_time: float = 3.0,
    dt: float = 0.02,
    gravity: float = 0.02,
    initial_velocity_z: float = -0.015,
    damping_alpha: float = 1.5,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_c3d8_model(case=case, resolution=resolution, quick=False, total_time=total_time, dt=dt)
    model.gravity = float(gravity)
    model.initial_velocity_z = float(initial_velocity_z)
    start = perf_counter()
    rows, displacement, vm_final = _long_time_linear_dynamic_history(model, damping_alpha=damping_alpha)
    linearity = "linear_damped"
    wall = perf_counter() - start
    for row in rows:
        row["linearity"] = linearity
        row["element_type"] = "C3D8"
        row["wall_time_seconds"] = wall
    fields = _element_fields(model, displacement)
    fields["von_mises"] = np.asarray(vm_final, dtype=float)
    fields.update(_contact_cell_fields(model, displacement))
    outputs: dict[str, Path] = {
        "history": out_dir / "long_time_c3d8_dynamic_history.csv",
        "summary": out_dir / "long_time_c3d8_dynamic_summary.md",
        "vtk": out_dir / "vtk" / "long_time_c3d8_peak_contact_clouds.vtk",
    }
    _write_csv(outputs["history"], rows)
    _write_legacy_hex_vtk(outputs["vtk"], model.X + displacement, model.elements, cell_values=fields)
    outputs.update(_plot_histories(out_dir, rows))
    outputs.update(_plot_clouds(out_dir, model, displacement, fields))
    peak_contact_row = max(rows, key=lambda row: float(row["normal_force"]))
    summary = [
        "# Long-Time C3D8 Dynamic SDF Contact",
        "",
        f"- case: `{case}`",
        f"- element type: `C3D8`",
        f"- resolution: `{resolution}`",
        f"- total time: `{total_time}` s",
        f"- dt: `{dt}` s",
        f"- gravity: `{gravity}`",
        f"- initial z velocity: `{initial_velocity_z}`",
        f"- mass-proportional damping alpha: `{damping_alpha}`",
        f"- steps: `{len(rows) - 1}`",
        f"- linearity: `{linearity}`",
        f"- wall time: `{wall:.6f}` s",
        f"- max active contact samples: `{max(int(row['active_contact_count']) for row in rows)}`",
        f"- min gap: `{min(float(row['min_gap']) for row in rows):.6e}`",
        f"- max normal force: `{max(float(row['normal_force']) for row in rows):.6e}`",
        f"- cloud snapshot time: `{float(peak_contact_row['time']):.6f}` s",
        "- cloud snapshot rule: peak normal-force state",
        "",
        "## Outputs",
        "",
    ]
    summary.extend(f"- `{path.relative_to(out_dir).as_posix()}`" for path in outputs.values() if path != outputs["summary"])
    outputs["summary"].write_text("\n".join(summary) + "\n", encoding="utf-8")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "long_time_c3d8_dynamic")
    parser.add_argument("--case", choices=("block_plane_c3d8", "block_block_c3d8"), default="block_block_c3d8")
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--total-time", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--gravity", type=float, default=0.02)
    parser.add_argument("--initial-velocity-z", type=float, default=-0.015)
    parser.add_argument("--damping-alpha", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_validation(
        args.out_dir,
        case=str(args.case),
        resolution=int(args.resolution),
        total_time=float(args.total_time),
        dt=float(args.dt),
        gravity=float(args.gravity),
        initial_velocity_z=float(args.initial_velocity_z),
        damping_alpha=float(args.damping_alpha),
    )
    print("Long-time C3D8 dynamic validation complete.")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
