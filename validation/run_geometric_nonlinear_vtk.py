"""Generate VTK frames for a geometrically nonlinear TET4 contact diagnostic.

This validation-only script uses a total-Lagrangian St. Venant-Kirchhoff TET4
model and a conservative rigid-plane penalty contact law. It is intended for
ParaView inspection of displacement, Green-Lagrange strain, Cauchy stress, and
von Mises stress evolution. It does not change the core SFC solver.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

Row = dict[str, float | int | str]

TET4_VTK_TYPE = 10
QUAD_VTK_TYPE = 9
TRIANGLE_QUADRATURE_BARYCENTRIC = np.asarray(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ],
    dtype=float,
)
TRIANGLE_QUADRATURE_WEIGHTS = np.full(3, 1.0 / 3.0, dtype=float)


@dataclass(slots=True)
class NonlinearState:
    """State and diagnostics for one nonlinear TET4 frame."""

    time: float
    x: np.ndarray
    v: np.ndarray
    element_strain: np.ndarray
    element_stress: np.ndarray
    von_mises: np.ndarray
    min_gap: float
    max_penetration: float
    active_contact_count: int
    kinetic_energy: float
    strain_energy: float
    gravitational_energy: float
    contact_energy: float
    total_energy: float


def _structured_block_mesh(resolution: int, *, size: tuple[float, float, float], bottom_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a structured TET4 block mesh."""

    n = max(1, int(resolution))
    lx, ly, lz = size
    nodes: list[tuple[float, float, float]] = []
    node_id: dict[tuple[int, int, int], int] = {}
    for k in range(n + 1):
        z = bottom_z + lz * k / n
        for j in range(n + 1):
            y = -0.5 * ly + ly * j / n
            for i in range(n + 1):
                x = -0.5 * lx + lx * i / n
                node_id[(i, j, k)] = len(nodes)
                nodes.append((x, y, z))

    tets: list[tuple[int, int, int, int]] = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                v000 = node_id[(i, j, k)]
                v100 = node_id[(i + 1, j, k)]
                v010 = node_id[(i, j + 1, k)]
                v110 = node_id[(i + 1, j + 1, k)]
                v001 = node_id[(i, j, k + 1)]
                v101 = node_id[(i + 1, j, k + 1)]
                v011 = node_id[(i, j + 1, k + 1)]
                v111 = node_id[(i + 1, j + 1, k + 1)]
                tets.extend(
                    [
                        (v000, v100, v010, v001),
                        (v100, v110, v010, v111),
                        (v100, v010, v001, v111),
                        (v010, v001, v011, v111),
                        (v100, v001, v101, v111),
                    ]
                )
    X = np.asarray(nodes, dtype=float)
    elements = np.asarray(tets, dtype=np.int64)
    return X, _orient_tets_positive(X, elements)


def _fibonacci_sphere_directions(count: int) -> np.ndarray:
    """Return deterministic approximately uniform unit directions."""

    n = max(4, int(count))
    indices = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / n
    theta = np.pi * (3.0 - np.sqrt(5.0)) * indices
    radius_xy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack((radius_xy * np.cos(theta), radius_xy * np.sin(theta), z))


def _sphere_like_mesh(resolution: int, *, radius: float, center: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic Delaunay TET4 mesh of a sphere-like volume."""

    n = max(1, int(resolution))
    surface_count = max(24, 12 * n * n)
    center_array = np.asarray(center, dtype=float)
    axis_directions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    points: list[np.ndarray] = [center_array]
    for layer in range(1, n + 1):
        layer_radius = radius * layer / n
        layer_count = surface_count if layer == n else max(12, int(surface_count * (layer / n) ** 2))
        directions = _fibonacci_sphere_directions(layer_count)
        if layer == n:
            directions = np.vstack((axis_directions, directions))
        points.extend(center_array + layer_radius * directions)

    X = np.asarray(points, dtype=float)
    elements = Delaunay(X).simplices.astype(np.int64)
    elements = _orient_tets_positive(X, elements)
    return X, _remove_degenerate_tets(X, elements)


def _remove_degenerate_tets(X: np.ndarray, elements: np.ndarray, *, tolerance: float = 1.0e-12) -> np.ndarray:
    """Remove near-zero-volume tetrahedra from Delaunay output."""

    kept: list[np.ndarray] = []
    for element in elements:
        Dm = np.column_stack((X[element[1]] - X[element[0]], X[element[2]] - X[element[0]], X[element[3]] - X[element[0]]))
        if np.linalg.det(Dm) / 6.0 > tolerance:
            kept.append(element)
    if not kept:
        raise ValueError("sphere-like mesh generation produced no valid tetrahedra")
    return np.asarray(kept, dtype=np.int64)


def _orient_tets_positive(X: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Swap local nodes so each TET4 has positive reference volume."""

    oriented = np.asarray(elements, dtype=np.int64).copy()
    for e, element in enumerate(oriented):
        Dm = np.column_stack((X[element[1]] - X[element[0]], X[element[2]] - X[element[0]], X[element[3]] - X[element[0]]))
        if np.linalg.det(Dm) < 0.0:
            oriented[e, [1, 2]] = oriented[e, [2, 1]]
    return oriented


def _boundary_faces(elements: np.ndarray) -> np.ndarray:
    """Return unique TET4 boundary faces as node-index triples."""

    local_faces = ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3))
    records: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for element in elements:
        for local in local_faces:
            face = tuple(int(element[i]) for i in local)
            records.setdefault(tuple(sorted(face)), []).append(face)
    return np.asarray([faces[0] for faces in records.values() if len(faces) == 1], dtype=np.int64)


def _tet_reference_data(X: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return reference volumes and shape gradients for each TET4."""

    parent_grads = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    volumes = np.zeros(elements.shape[0], dtype=float)
    grads = np.zeros((elements.shape[0], 4, 3), dtype=float)
    for e, element in enumerate(elements):
        Xe = X[element]
        Dm = np.column_stack((Xe[1] - Xe[0], Xe[2] - Xe[0], Xe[3] - Xe[0]))
        det = float(np.linalg.det(Dm))
        if det <= 0.0:
            raise ValueError("TET4 reference element must have positive orientation")
        volumes[e] = det / 6.0
        grads[e] = parent_grads @ np.linalg.inv(Dm)
    return volumes, grads


def _lame_parameters(E: float, nu: float) -> tuple[float, float]:
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def _voigt_strain(tensor: np.ndarray) -> np.ndarray:
    return np.asarray([tensor[0, 0], tensor[1, 1], tensor[2, 2], 2.0 * tensor[0, 1], 2.0 * tensor[1, 2], 2.0 * tensor[0, 2]], dtype=float)


def _von_mises(stress: np.ndarray) -> float:
    s = stress
    return float(
        np.sqrt(
            0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2)
            + 3.0 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[0, 2] ** 2)
        )
    )


def _internal_response(
    X: np.ndarray,
    elements: np.ndarray,
    volumes: np.ndarray,
    grads: np.ndarray,
    x: np.ndarray,
    *,
    E: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return total-Lagrangian internal force, strain, stress, vm, and energy."""

    lam, mu = _lame_parameters(E, nu)
    force = np.zeros_like(x)
    strains = np.zeros((elements.shape[0], 3, 3), dtype=float)
    stresses = np.zeros((elements.shape[0], 3, 3), dtype=float)
    vm = np.zeros(elements.shape[0], dtype=float)
    energy = 0.0
    identity = np.eye(3)
    for e, element in enumerate(elements):
        xe = x[element]
        grad = grads[e]
        F = xe.T @ grad
        green = 0.5 * (F.T @ F - identity)
        second_piola = lam * float(np.trace(green)) * identity + 2.0 * mu * green
        first_piola = F @ second_piola
        for a in range(4):
            force[int(element[a])] += volumes[e] * (first_piola @ grad[a])
        J = max(float(np.linalg.det(F)), 1.0e-12)
        cauchy = (first_piola @ F.T) / J
        strains[e] = green
        stresses[e] = cauchy
        vm[e] = _von_mises(cauchy)
        energy_density = 0.5 * float(_voigt_strain(green) @ _voigt_stress(second_piola))
        energy += volumes[e] * energy_density
    return force, strains, stresses, vm, float(energy)


def _voigt_stress(tensor: np.ndarray) -> np.ndarray:
    return np.asarray([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[1, 2], tensor[0, 2]], dtype=float)


def _contact_response(
    x: np.ndarray,
    boundary_faces: np.ndarray,
    *,
    plane_z: float,
    stiffness: float,
) -> tuple[np.ndarray, float, int, float, float]:
    """Return conservative plane-contact force and energy."""

    force = np.zeros_like(x)
    min_gap = np.inf
    max_penetration = 0.0
    active_count = 0
    energy = 0.0
    bary = TRIANGLE_QUADRATURE_BARYCENTRIC
    q_weights = TRIANGLE_QUADRATURE_WEIGHTS
    for face in boundary_faces:
        tri = x[face]
        area = 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
        if area <= 0.0:
            continue
        for shape, weight in zip(bary, q_weights, strict=True):
            point = shape @ tri
            gap = float(point[2] - plane_z)
            min_gap = min(min_gap, gap)
            penetration = max(-gap, 0.0)
            if penetration <= 0.0:
                continue
            active_count += 1
            max_penetration = max(max_penetration, penetration)
            q_area = area * float(weight)
            normal_force = stiffness * penetration * q_area
            energy += 0.5 * stiffness * penetration * penetration * q_area
            for local, node in enumerate(face):
                force[int(node), 2] += shape[local] * normal_force
    if not np.isfinite(min_gap):
        min_gap = float(np.min(x[:, 2] - plane_z))
    return force, float(min_gap), active_count, float(max_penetration), float(energy)


def _lumped_mass(n_nodes: int, elements: np.ndarray, volumes: np.ndarray, density: float) -> np.ndarray:
    mass = np.zeros(n_nodes, dtype=float)
    for element, volume in zip(elements, volumes, strict=True):
        mass[element] += density * volume / 4.0
    return mass


def _write_vtk(
    path: Path,
    X: np.ndarray,
    x: np.ndarray,
    elements: np.ndarray,
    state: NonlinearState,
    *,
    plane_z: float | None = None,
    plane_half_width: float = 1.0,
) -> None:
    """Write a legacy VTK unstructured-grid frame."""

    displacement = x - X
    plane_points = _visual_plane_points(plane_z, plane_half_width) if plane_z is not None else np.empty((0, 3), dtype=float)
    points = np.vstack((x, plane_points)) if plane_points.size else x
    n_plane_cells = 1 if plane_points.size else 0
    n_cells = elements.shape[0] + n_plane_cells
    cell_int_count = elements.shape[0] * 5 + n_plane_cells * 5
    zero_tensor = np.zeros((3, 3), dtype=float)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"geometric nonlinear TET4 frame time={state.time:.12g}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {points.shape[0]} float\n")
        for point in points:
            f.write(f"{point[0]:.9e} {point[1]:.9e} {point[2]:.9e}\n")
        f.write(f"CELLS {n_cells} {cell_int_count}\n")
        for element in elements:
            f.write(f"4 {int(element[0])} {int(element[1])} {int(element[2])} {int(element[3])}\n")
        if n_plane_cells:
            offset = x.shape[0]
            f.write(f"4 {offset} {offset + 1} {offset + 2} {offset + 3}\n")
        f.write(f"CELL_TYPES {n_cells}\n")
        for _ in elements:
            f.write(f"{TET4_VTK_TYPE}\n")
        if n_plane_cells:
            f.write(f"{QUAD_VTK_TYPE}\n")
        f.write(f"POINT_DATA {points.shape[0]}\n")
        f.write("VECTORS displacement float\n")
        for value in displacement:
            f.write(f"{value[0]:.9e} {value[1]:.9e} {value[2]:.9e}\n")
        for _ in plane_points:
            f.write("0.000000000e+00 0.000000000e+00 0.000000000e+00\n")
        f.write("VECTORS velocity float\n")
        for value in state.v:
            f.write(f"{value[0]:.9e} {value[1]:.9e} {value[2]:.9e}\n")
        for _ in plane_points:
            f.write("0.000000000e+00 0.000000000e+00 0.000000000e+00\n")
        f.write(f"CELL_DATA {n_cells}\n")
        f.write("SCALARS object_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for _ in elements:
            f.write("1\n")
        if n_plane_cells:
            f.write("0\n")
        f.write("SCALARS von_mises float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for value in state.von_mises:
            f.write(f"{value:.9e}\n")
        if n_plane_cells:
            f.write("0.000000000e+00\n")
        f.write("SCALARS green_lagrange_strain_norm float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for strain in state.element_strain:
            f.write(f"{np.linalg.norm(strain):.9e}\n")
        if n_plane_cells:
            f.write("0.000000000e+00\n")
        f.write("TENSORS green_lagrange_strain float\n")
        for tensor in state.element_strain:
            _write_tensor(f, tensor)
        if n_plane_cells:
            _write_tensor(f, zero_tensor)
        f.write("TENSORS cauchy_stress float\n")
        for tensor in state.element_stress:
            _write_tensor(f, tensor)
        if n_plane_cells:
            _write_tensor(f, zero_tensor)


def _visual_plane_points(plane_z: float | None, half_width: float) -> np.ndarray:
    if plane_z is None:
        return np.empty((0, 3), dtype=float)
    h = float(half_width)
    return np.asarray(
        [
            [-h, -h, plane_z],
            [h, -h, plane_z],
            [h, h, plane_z],
            [-h, h, plane_z],
        ],
        dtype=float,
    )


def _write_tensor(handle, tensor: np.ndarray) -> None:  # noqa: ANN001
    for row in tensor:
        handle.write(f"{row[0]:.9e} {row[1]:.9e} {row[2]:.9e}\n")


def _compute_state(
    X: np.ndarray,
    elements: np.ndarray,
    volumes: np.ndarray,
    grads: np.ndarray,
    boundary_faces: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    mass: np.ndarray,
    *,
    E: float,
    nu: float,
    density: float,
    gravity: float,
    plane_z: float,
    contact_stiffness: float,
) -> tuple[NonlinearState, np.ndarray]:
    fint, strain, stress, vm, strain_energy = _internal_response(X, elements, volumes, grads, x, E=E, nu=nu)
    fcontact, min_gap, active_count, max_pen, contact_energy = _contact_response(
        x,
        boundary_faces,
        plane_z=plane_z,
        stiffness=contact_stiffness,
    )
    fext = np.zeros_like(x)
    fext[:, 2] = -mass * gravity
    force = fext + fcontact - fint
    acceleration = force / mass[:, None]
    kinetic = 0.5 * float(np.sum(mass[:, None] * v * v))
    gravitational = float(np.sum(mass * gravity * x[:, 2]))
    total = kinetic + strain_energy + gravitational + contact_energy
    state = NonlinearState(
        time=0.0,
        x=x.copy(),
        v=v.copy(),
        element_strain=strain,
        element_stress=stress,
        von_mises=vm,
        min_gap=min_gap,
        max_penetration=max_pen,
        active_contact_count=active_count,
        kinetic_energy=kinetic,
        strain_energy=strain_energy,
        gravitational_energy=gravitational,
        contact_energy=contact_energy,
        total_energy=total,
    )
    return state, acceleration


def run_simulation(
    out_dir: Path,
    *,
    case: str = "sphere_drop",
    resolution: int = 2,
    duration: float = 0.5,
    dt: float = 2.5e-4,
    frame_stride: int = 20,
    E: float = 1000.0,
    nu: float = 0.3,
    density: float = 1.0,
    gravity: float = 9.81,
    contact_stiffness: float = 5000.0,
    initial_gap: float = 0.08,
    initial_velocity_z: float = 0.0,
) -> dict[str, Path]:
    """Run a nonlinear contact diagnostic and write VTK frames plus CSV summary."""

    out_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir = out_dir / "vtk"
    vtk_dir.mkdir(parents=True, exist_ok=True)
    for stale in vtk_dir.glob("frame_*.vtk"):
        stale.unlink()
    plane_z = 0.0
    if case == "block_drop":
        block_size = (0.8, 0.8, 0.8)
        X, elements = _structured_block_mesh(resolution, size=block_size, bottom_z=plane_z + initial_gap)
        plane_half_width = 0.65
    elif case == "sphere_drop":
        sphere_radius = 0.25
        sphere_center = (0.0, 0.0, plane_z + sphere_radius + initial_gap)
        X, elements = _sphere_like_mesh(resolution, radius=sphere_radius, center=sphere_center)
        plane_half_width = 0.55
    else:
        raise ValueError(f"unknown nonlinear VTK case: {case!r}")
    boundary_faces = _boundary_faces(elements)
    volumes, grads = _tet_reference_data(X, elements)
    mass = _lumped_mass(X.shape[0], elements, volumes, density)
    x = X.copy()
    v = np.zeros_like(X)
    v[:, 2] = initial_velocity_z
    actual_initial_gap = float(np.min(x[:, 2] - plane_z))
    steps = int(np.ceil(duration / dt))
    frame_stride = max(1, int(frame_stride))
    rows: list[Row] = []
    frame_rows: list[Row] = []
    frame_index = 0

    state, acceleration = _compute_state(
        X,
        elements,
        volumes,
        grads,
        boundary_faces,
        x,
        v,
        mass,
        E=E,
        nu=nu,
        density=density,
        gravity=gravity,
        plane_z=plane_z,
        contact_stiffness=contact_stiffness,
    )

    for step in range(steps + 1):
        time = min(step * dt, duration)
        state.time = float(time)
        if step % frame_stride == 0 or step == steps:
            frame_path = vtk_dir / f"frame_{frame_index:04d}.vtk"
            _write_vtk(frame_path, X, x, elements, state, plane_z=plane_z, plane_half_width=plane_half_width)
            frame_rows.append({"frame": frame_index, "step": step, "time": float(time), "path": str(frame_path.relative_to(out_dir))})
            frame_index += 1
        rows.append(
            {
                "step": step,
                "case": case,
                "time": float(time),
                "z_cm": float(np.sum(mass * x[:, 2]) / np.sum(mass)),
                "min_gap": state.min_gap,
                "max_penetration": state.max_penetration,
                "active_contact_count": state.active_contact_count,
                "kinetic_energy": state.kinetic_energy,
                "strain_energy": state.strain_energy,
                "gravitational_energy": state.gravitational_energy,
                "contact_energy": state.contact_energy,
                "total_energy": state.total_energy,
                "max_von_mises": float(np.max(state.von_mises)),
                "mean_von_mises": float(np.mean(state.von_mises)),
            }
        )
        if step == steps:
            break

        v_half = v + 0.5 * dt * acceleration
        x_new = x + dt * v_half
        next_state, next_acceleration = _compute_state(
            X,
            elements,
            volumes,
            grads,
            boundary_faces,
            x_new,
            v_half,
            mass,
            E=E,
            nu=nu,
            density=density,
            gravity=gravity,
            plane_z=plane_z,
            contact_stiffness=contact_stiffness,
        )
        v_new = v_half + 0.5 * dt * next_acceleration
        next_state.v = v_new.copy()
        kinetic = 0.5 * float(np.sum(mass[:, None] * v_new * v_new))
        next_state.kinetic_energy = kinetic
        next_state.total_energy = kinetic + next_state.strain_energy + next_state.gravitational_energy + next_state.contact_energy
        x, v, acceleration, state = x_new, v_new, next_acceleration, next_state

    history_csv = out_dir / "geometric_nonlinear_history.csv"
    frames_csv = out_dir / "geometric_nonlinear_frames.csv"
    summary_md = out_dir / "geometric_nonlinear_summary.md"
    _write_csv(history_csv, rows)
    _write_csv(frames_csv, frame_rows)
    _write_summary(
        summary_md,
        rows=rows,
        frame_rows=frame_rows,
        resolution=resolution,
        duration=duration,
        dt=dt,
        frame_stride=frame_stride,
        case=case,
        E=E,
        nu=nu,
        density=density,
        gravity=gravity,
        contact_stiffness=contact_stiffness,
        initial_gap=actual_initial_gap,
        initial_velocity_z=initial_velocity_z,
    )
    return {"history": history_csv, "frames": frames_csv, "summary": summary_md, "vtk_dir": vtk_dir}


def _write_csv(path: Path, rows: list[Row]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    *,
    rows: list[Row],
    frame_rows: list[Row],
    resolution: int,
    duration: float,
    dt: float,
    frame_stride: int,
    case: str,
    E: float,
    nu: float,
    density: float,
    gravity: float,
    contact_stiffness: float,
    initial_gap: float,
    initial_velocity_z: float,
) -> None:
    e0 = float(rows[0]["total_energy"])
    energies = np.asarray([float(row["total_energy"]) for row in rows], dtype=float)
    rel_min = float((np.min(energies) - e0) / max(abs(e0), 1.0e-30))
    rel_max = float((np.max(energies) - e0) / max(abs(e0), 1.0e-30))
    lines = [
        "# Geometric Nonlinear VTK Diagnostic",
        "",
        "This validation-only run uses total-Lagrangian St. Venant-Kirchhoff TET4 kinematics and conservative penalty contact.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python validation/run_geometric_nonlinear_vtk.py --case {case} --resolution {resolution} --duration {duration:g} --dt {dt:g} --frame-stride {frame_stride} --out-dir {path.parent.as_posix()}",
        "```",
        "",
        "## Outputs",
        "",
        "- `vtk/frame_0000.vtk`, `vtk/frame_0001.vtk`, ... for ParaView animation.",
        "- `geometric_nonlinear_history.csv` for center-of-mass, contact, and energy diagnostics.",
        "- `geometric_nonlinear_frames.csv` for frame/time mapping.",
        "- the rigid contact plane is included as `object_id=0`; the deformable body is `object_id=1`.",
        "",
        "## Fields",
        "",
        "- point vector: `displacement`",
        "- point vector: `velocity`",
        "- cell scalar: `von_mises`",
        "- cell scalar: `green_lagrange_strain_norm`",
        "- cell scalar: `object_id`",
        "- cell tensor: `green_lagrange_strain`",
        "- cell tensor: `cauchy_stress`",
        "",
        "## Parameters",
        "",
        f"- case: `{case}`",
        f"- resolution: `{resolution}`",
        f"- duration: `{duration:g}`",
        f"- dt: `{dt:g}`",
        f"- frame stride: `{frame_stride}`",
        f"- E: `{E:g}`",
        f"- nu: `{nu:g}`",
        f"- density: `{density:g}`",
        f"- gravity: `{gravity:g}`",
        f"- contact stiffness: `{contact_stiffness:g}`",
        f"- initial visible gap: `{initial_gap:.6e}`",
        f"- initial z velocity: `{initial_velocity_z:g}`",
        "",
        "## Diagnostics",
        "",
        f"- frames written: `{len(frame_rows)}`",
        f"- initial minimum gap: `{float(rows[0]['min_gap']):.6e}`",
        f"- minimum gap: `{min(float(row['min_gap']) for row in rows):.6e}`",
        f"- maximum penetration: `{max(float(row['max_penetration']) for row in rows):.6e}`",
        f"- maximum von Mises stress: `{max(float(row['max_von_mises']) for row in rows):.6e}`",
        f"- relative energy drift range: `{rel_min:.6e}` to `{rel_max:.6e}`",
        "",
        "## Limitations",
        "",
        "This is a diagnostic nonlinear kinematics exporter, not yet the production SFC nonlinear implicit solver.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--case", choices=("sphere_drop", "block_drop"), default="sphere_drop")
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--E", type=float, default=1000.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--contact-stiffness", type=float, default=5000.0)
    parser.add_argument("--initial-gap", type=float, default=0.08)
    parser.add_argument("--initial-velocity-z", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = ROOT / "results" / ("drop_sphere_geometric_nonlinear_vtk" if args.case == "sphere_drop" else "geometric_nonlinear_vtk")
    outputs = run_simulation(
        out_dir,
        case=args.case,
        resolution=args.resolution,
        duration=args.duration,
        dt=args.dt,
        frame_stride=args.frame_stride,
        E=args.E,
        nu=args.nu,
        density=args.density,
        gravity=args.gravity,
        contact_stiffness=args.contact_stiffness,
        initial_gap=args.initial_gap,
        initial_velocity_z=args.initial_velocity_z,
    )
    print("Geometric nonlinear VTK diagnostic complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
