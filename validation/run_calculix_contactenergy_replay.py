"""Replay CalculiX `contactenergy.inp` with current-surface dynamic SDF.

This validation intentionally uses the official CalculiX C3D8 contact-energy
test as a contact-law/energy reference. It is not a TET4 trajectory-equivalence
case: CalculiX solves the static C3D8 contact problem, and SFC replays the final
deformed surface geometry using dynamic-SDF local projection and the same linear
pressure-overclosure law.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.mesh import extract_boundary_triangles  # noqa: E402
from sfc.sdf.dynamic_surface_sdf import dynamic_surface_sdf  # noqa: E402


Row = dict[str, Any]

WSL_CONTACTENERGY_INP_CANDIDATES = [
    "/tmp/sfc_calculix_source/test/contactenergy.inp",
    "/usr/share/doc/calculix-ccx-test/examples/test/contactenergy.inp",
]
WSL_CONTACTENERGY_DAT_REF_CANDIDATES = [
    "/tmp/sfc_calculix_source/test/contactenergy.dat.ref",
    "/usr/share/doc/calculix-ccx-test/examples/test/contactenergy.dat.ref",
]

C3D8_FACE_NODES = {
    "S1": np.array([0, 1, 2, 3], dtype=np.int64),
    "S2": np.array([4, 7, 6, 5], dtype=np.int64),
    "S3": np.array([0, 4, 5, 1], dtype=np.int64),
    "S4": np.array([1, 5, 6, 2], dtype=np.int64),
    "S5": np.array([2, 6, 7, 3], dtype=np.int64),
    "S6": np.array([3, 7, 4, 0], dtype=np.int64),
}


@dataclass(frozen=True)
class ContactEnergyModel:
    node_ids: np.ndarray
    X: np.ndarray
    element_ids: np.ndarray
    elements: np.ndarray
    elsets: dict[str, list[int]]
    nsets: dict[str, list[int]]
    surfaces: dict[str, list[tuple[str, str]]]
    boundaries: list[tuple[str, int, int, float]]
    cloads: list[tuple[int, int, float]]
    pressure_stiffness: float
    z_load_reference: float
    material_E: float
    material_nu: float


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


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def _wsl_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _first_existing_wsl_path(candidates: list[str]) -> str | None:
    if not _wsl_available():
        return None
    for candidate in candidates:
        proc = subprocess.run(
            ["wsl", "--exec", "bash", "-lc", f"test -f {_wsl_quote(candidate)}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0:
            return candidate
    return None


def _read_wsl_text(candidates: list[str]) -> tuple[str, str]:
    source = _first_existing_wsl_path(candidates)
    if source is None:
        raise RuntimeError(f"none of the WSL reference files exist: {candidates}")
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", f"cat {_wsl_quote(source)}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.stdout, source


def _calculix_available() -> bool:
    if not _wsl_available():
        return False
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", "command -v ccx >/dev/null"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return proc.returncode == 0


def _calculix_version() -> str:
    if not _calculix_available():
        return "unavailable"
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", "ccx -v 2>&1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    lines = [line.strip() for line in (proc.stdout + proc.stderr).splitlines() if line.strip()]
    return lines[0] if lines else "CalculiX"


def _parse_keyword(line: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in line.split(",")]
    name = parts[0].upper()
    params: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip().upper()] = value.strip()
    return name, params


def parse_contactenergy_input(text: str) -> ContactEnergyModel:
    nodes: dict[int, list[float]] = {}
    elements: dict[int, list[int]] = {}
    elsets: dict[str, list[int]] = {}
    nsets: dict[str, list[int]] = {}
    surfaces: dict[str, list[tuple[str, str]]] = {}
    boundaries: list[tuple[str, int, int, float]] = []
    cloads: list[tuple[int, int, float]] = []
    pressure_stiffness: float | None = None
    material_E: float | None = None
    material_nu: float | None = None

    section = ""
    params: dict[str, str] = {}
    current_surface = ""
    current_elset = ""
    current_nset = ""
    expect_surface_behavior_value = False
    expect_elastic_value = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            section, params = _parse_keyword(line)
            expect_surface_behavior_value = section == "*SURFACE BEHAVIOR"
            expect_elastic_value = section == "*ELASTIC"
            current_surface = params.get("NAME", "") if section == "*SURFACE" else ""
            current_elset = params.get("ELSET", "") if section == "*ELSET" else params.get("ELSET", "")
            current_nset = params.get("NSET", "") if section == "*NSET" else params.get("NSET", "") if section == "*NODE" else ""
            continue

        if expect_elastic_value:
            values = [value.strip() for value in line.split(",") if value.strip()]
            material_E = float(values[0])
            material_nu = float(values[1])
            expect_elastic_value = False
            continue

        if expect_surface_behavior_value:
            pressure_stiffness = float(line.split(",")[0])
            expect_surface_behavior_value = False
            continue

        values = [value.strip() for value in line.split(",") if value.strip()]
        if section == "*NODE":
            node_id = int(values[0])
            nodes[node_id] = [float(values[1]), float(values[2]), float(values[3])]
            if current_nset:
                nsets.setdefault(current_nset, []).append(node_id)
        elif section == "*ELEMENT":
            element_id = int(values[0])
            elements[element_id] = [int(value) for value in values[1:]]
            if current_elset:
                elsets.setdefault(current_elset, []).append(element_id)
        elif section == "*ELSET":
            elsets.setdefault(current_elset, []).extend(int(value) for value in values)
        elif section == "*NSET":
            nsets.setdefault(current_nset, []).extend(int(value) for value in values)
        elif section == "*SURFACE":
            surfaces.setdefault(current_surface, []).append((values[0], values[1].upper()))
        elif section == "*BOUNDARY":
            target = values[0]
            dof1 = int(values[1])
            dof2 = int(values[2]) if len(values) >= 3 else dof1
            value = float(values[3]) if len(values) >= 4 else 0.0
            boundaries.append((target, dof1, dof2, value))
        elif section == "*CLOAD":
            cloads.append((int(values[0]), int(values[1]), float(values[2])))

    if pressure_stiffness is None:
        raise ValueError("failed to parse pressure-overclosure stiffness from contactenergy input")
    if material_E is None or material_nu is None:
        raise ValueError("failed to parse elastic material from contactenergy input")

    node_ids = np.asarray(sorted(nodes), dtype=np.int64)
    node_to_index = {int(node_id): idx for idx, node_id in enumerate(node_ids)}
    X = np.asarray([nodes[int(node_id)] for node_id in node_ids], dtype=float)

    element_ids = np.asarray(sorted(elements), dtype=np.int64)
    conn = np.asarray([[node_to_index[node_id] for node_id in elements[int(element_id)]] for element_id in element_ids], dtype=np.int64)
    z_load_reference = abs(sum(value for _, dof, value in cloads if dof == 3))
    return ContactEnergyModel(
        node_ids=node_ids,
        X=X,
        element_ids=element_ids,
        elements=conn,
        elsets=elsets,
        nsets=nsets,
        surfaces=surfaces,
        boundaries=boundaries,
        cloads=cloads,
        pressure_stiffness=float(pressure_stiffness),
        z_load_reference=float(z_load_reference),
        material_E=float(material_E),
        material_nu=float(material_nu),
    )


def _parse_last_nodal_block(text: str, block_name: str) -> dict[int, np.ndarray]:
    lines = text.splitlines()
    blocks: list[dict[int, np.ndarray]] = []
    for idx, line in enumerate(lines):
        if block_name in line.lower():
            block: dict[int, np.ndarray] = {}
            for row in lines[idx + 1 :]:
                parts = row.split()
                if len(parts) == 4 and parts[0].lstrip("+-").isdigit():
                    block[int(parts[0])] = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
                elif block:
                    break
            if block:
                blocks.append(block)
    if not blocks:
        raise ValueError(f"failed to parse CalculiX {block_name} block")
    return blocks[-1]


def _parse_total_contact_energy(text: str) -> float:
    matches = re.findall(r"total contact spring energy for time\s+[-+0-9.Ee]+\s*\n\s*([-+0-9.Ee]+)", text, flags=re.IGNORECASE)
    if not matches:
        raise ValueError("failed to parse total contact spring energy")
    return float(matches[-1])


def _parse_raw_cels_rows(text: str) -> list[Row]:
    rows: list[Row] = []
    in_block = False
    for raw in text.splitlines():
        lower = raw.lower()
        if "contact spring energy" in lower and "slave element+face" in lower:
            in_block = True
            rows = []
            continue
        if in_block and "total contact spring energy" in lower:
            in_block = False
            continue
        if in_block:
            parts = raw.split()
            if len(parts) == 3 and parts[0].lstrip("+-").isdigit():
                rows.append({"slave_element": int(parts[0]), "slave_face": int(parts[1]), "cels": float(parts[2])})
    return rows


def parse_contactenergy_dat(text: str, node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, list[Row]]:
    displacement_block = _parse_last_nodal_block(text, "displacements")
    force_block = _parse_last_nodal_block(text, "forces")
    U = np.asarray([displacement_block.get(int(node_id), np.zeros(3)) for node_id in node_ids], dtype=float)
    RF = np.asarray([force_block.get(int(node_id), np.zeros(3)) for node_id in node_ids], dtype=float)
    return U, RF, _parse_total_contact_energy(text), _parse_raw_cels_rows(text)


def _surface_quads(model: ContactEnergyModel, surface_name: str) -> list[tuple[np.ndarray, np.ndarray]]:
    element_id_to_index = {int(element_id): idx for idx, element_id in enumerate(model.element_ids)}
    quads: list[tuple[np.ndarray, np.ndarray]] = []
    for target, face_name in model.surfaces[surface_name]:
        if face_name not in C3D8_FACE_NODES:
            raise ValueError(f"unsupported C3D8 face label {face_name}")
        element_ids = model.elsets.get(target, [int(target)] if target.isdigit() else [])
        if not element_ids:
            raise ValueError(f"surface target {target!r} does not resolve to elements")
        for element_id in element_ids:
            element = model.elements[element_id_to_index[int(element_id)]]
            quads.append((element[C3D8_FACE_NODES[face_name]], element))
    return quads


def _orient_quad_outward(X: np.ndarray, quad: np.ndarray, element: np.ndarray) -> np.ndarray:
    polygon = X[quad]
    normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
    if float(np.dot(normal, polygon.mean(axis=0) - X[element].mean(axis=0))) < 0.0:
        return quad[::-1]
    return quad


def _triangles_from_quads(X: np.ndarray, quads: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    triangles: list[np.ndarray] = []
    for quad, element in quads:
        oriented = _orient_quad_outward(X, quad, element)
        triangles.append(oriented[[0, 1, 2]])
        triangles.append(oriented[[0, 2, 3]])
    return np.asarray(triangles, dtype=np.int64)


def _triangle_area(triangle: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))


def replay_contactenergy_with_dynamic_sdf(model: ContactEnergyModel, U: np.ndarray) -> Row:
    X_current = model.X + U
    master_faces = _triangles_from_quads(X_current, _surface_quads(model, "Smast"))
    slave_faces = _triangles_from_quads(X_current, _surface_quads(model, "Sslav"))
    candidates = np.arange(master_faces.shape[0], dtype=np.int64)

    total_force = np.zeros(3, dtype=float)
    contact_energy = 0.0
    min_gap = np.inf
    active_count = 0
    for face in slave_faces:
        triangle = X_current[face]
        xq = triangle.mean(axis=0)
        area = _triangle_area(triangle)
        result = dynamic_surface_sdf(xq, X_current, master_faces, candidates)
        min_gap = min(min_gap, float(result.g))
        overclosure = max(-float(result.g), 0.0)
        if overclosure > 0.0:
            active_count += 1
            pressure = model.pressure_stiffness * overclosure
            total_force += pressure * area * result.n
            contact_energy += 0.5 * model.pressure_stiffness * overclosure * overclosure * area

    return {
        "sfc_dynamic_sdf_contact_force_x": float(total_force[0]),
        "sfc_dynamic_sdf_contact_force_y": float(total_force[1]),
        "sfc_dynamic_sdf_contact_force_z": float(total_force[2]),
        "sfc_dynamic_sdf_contact_force_norm": float(np.linalg.norm(total_force)),
        "sfc_dynamic_sdf_contact_energy": float(contact_energy),
        "sfc_dynamic_sdf_min_gap": float(min_gap),
        "sfc_dynamic_sdf_max_penetration": float(max(-min_gap, 0.0)),
        "sfc_dynamic_sdf_active_quadrature_count": int(active_count),
        "sfc_dynamic_sdf_master_triangle_count": int(master_faces.shape[0]),
        "sfc_dynamic_sdf_slave_quadrature_count": int(slave_faces.shape[0]),
    }


C3D8_NATURAL_NODE_COORDS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=float,
)

HEX8_CENTER_NATURAL_GRADIENTS = C3D8_NATURAL_NODE_COORDS / 8.0


def _hex8_natural_gradients(xi: float, eta: float, zeta: float) -> np.ndarray:
    gradients = np.empty((8, 3), dtype=float)
    for idx, (r, s, t) in enumerate(C3D8_NATURAL_NODE_COORDS):
        gradients[idx, 0] = 0.125 * r * (1.0 + s * eta) * (1.0 + t * zeta)
        gradients[idx, 1] = 0.125 * s * (1.0 + r * xi) * (1.0 + t * zeta)
        gradients[idx, 2] = 0.125 * t * (1.0 + r * xi) * (1.0 + s * eta)
    return gradients


def _hex8_B_matrix(gradients: np.ndarray) -> np.ndarray:
    B = np.zeros((6, 24), dtype=float)
    for local_node, (dndx, dndy, dndz) in enumerate(gradients):
        col = 3 * local_node
        B[0, col] = dndx
        B[1, col + 1] = dndy
        B[2, col + 2] = dndz
        B[3, col] = dndy
        B[3, col + 1] = dndx
        B[4, col + 1] = dndz
        B[4, col + 2] = dndy
        B[5, col] = dndz
        B[5, col + 2] = dndx
    return B


def _hex8_element_stiffness(Xe: np.ndarray, C: np.ndarray) -> np.ndarray:
    Ke = np.zeros((24, 24), dtype=float)
    gp = 1.0 / np.sqrt(3.0)
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            for zeta in (-gp, gp):
                dN_dnatural = _hex8_natural_gradients(float(xi), float(eta), float(zeta))
                jacobian = Xe.T @ dN_dnatural
                detJ = float(np.linalg.det(jacobian))
                if detJ <= 0.0:
                    raise ValueError("C3D8 element has non-positive Jacobian determinant")
                gradients = dN_dnatural @ np.linalg.inv(jacobian)
                B = _hex8_B_matrix(gradients)
                Ke += B.T @ C @ B * detJ
    return Ke


def _assemble_c3d8_stiffness(model: ContactEnergyModel) -> np.ndarray:
    C = isotropic_linear_elasticity_matrix(model.material_E, model.material_nu)
    ndofs = 3 * model.X.shape[0]
    K = np.zeros((ndofs, ndofs), dtype=float)
    for element in model.elements:
        Ke = _hex8_element_stiffness(model.X[element], C)
        dofs = np.asarray([3 * node + component for node in element for component in range(3)], dtype=np.int64)
        K[np.ix_(dofs, dofs)] += Ke
    return K


def _external_force_vector(model: ContactEnergyModel) -> np.ndarray:
    node_to_index = {int(node_id): idx for idx, node_id in enumerate(model.node_ids)}
    force = np.zeros(3 * model.X.shape[0], dtype=float)
    for node_id, dof, value in model.cloads:
        force[3 * node_to_index[int(node_id)] + int(dof) - 1] += float(value)
    return force


def _fixed_dofs_and_values(model: ContactEnergyModel) -> tuple[np.ndarray, np.ndarray]:
    node_to_index = {int(node_id): idx for idx, node_id in enumerate(model.node_ids)}
    fixed: list[int] = []
    values: list[float] = []
    for target, dof1, dof2, value in model.boundaries:
        if target.lstrip("+-").isdigit():
            node_ids = [int(target)]
        else:
            node_ids = model.nsets.get(target, [])
            if not node_ids:
                raise ValueError(f"boundary node set {target!r} is not defined")
        for node_id in node_ids:
            node_index = node_to_index[int(node_id)]
            for dof in range(int(dof1), int(dof2) + 1):
                fixed.append(3 * node_index + dof - 1)
                values.append(float(value))
    if not fixed:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=float)
    order = np.argsort(np.asarray(fixed, dtype=np.int64))
    fixed_array = np.asarray(fixed, dtype=np.int64)[order]
    value_array = np.asarray(values, dtype=float)[order]
    unique_fixed, unique_indices = np.unique(fixed_array, return_index=True)
    return unique_fixed, value_array[unique_indices]


def _contact_jacobian_rows(model: ContactEnergyModel) -> list[tuple[np.ndarray, float]]:
    X = model.X
    master_faces = _triangles_from_quads(X, _surface_quads(model, "Smast"))
    slave_quads = _surface_quads(model, "Sslav")
    candidates = np.arange(master_faces.shape[0], dtype=np.int64)
    rows: list[tuple[np.ndarray, float]] = []
    ndofs = 3 * X.shape[0]
    gp = 1.0 / np.sqrt(3.0)
    for quad, element in slave_quads:
        face = _orient_quad_outward(X, quad, element)
        coords = X[face]
        for xi in (-gp, gp):
            for eta in (-gp, gp):
                shape = np.asarray(
                    [
                        0.25 * (1.0 - xi) * (1.0 - eta),
                        0.25 * (1.0 + xi) * (1.0 - eta),
                        0.25 * (1.0 + xi) * (1.0 + eta),
                        0.25 * (1.0 - xi) * (1.0 + eta),
                    ],
                    dtype=float,
                )
                dshape_dxi = np.asarray(
                    [
                        -0.25 * (1.0 - eta),
                        0.25 * (1.0 - eta),
                        0.25 * (1.0 + eta),
                        -0.25 * (1.0 + eta),
                    ],
                    dtype=float,
                )
                dshape_deta = np.asarray(
                    [
                        -0.25 * (1.0 - xi),
                        -0.25 * (1.0 + xi),
                        0.25 * (1.0 + xi),
                        0.25 * (1.0 - xi),
                    ],
                    dtype=float,
                )
                xq = shape @ coords
                surface_jacobian = np.cross(dshape_dxi @ coords, dshape_deta @ coords)
                area_weight = float(np.linalg.norm(surface_jacobian))
                result = dynamic_surface_sdf(xq, X, master_faces, candidates)
                if float(result.g) > 1.0e-10:
                    continue
                J = np.zeros(ndofs, dtype=float)
                for node, weight in zip(face, shape, strict=True):
                    J[3 * int(node) : 3 * int(node) + 3] += float(weight) * result.n
                master_face = master_faces[int(result.face_id)]
                for node, weight in zip(master_face, result.w, strict=True):
                    J[3 * int(node) : 3 * int(node) + 3] -= float(weight) * result.n
                rows.append((J, area_weight))
    return rows


def solve_sfc_c3d8_contactenergy_static(model: ContactEnergyModel) -> tuple[np.ndarray, Row]:
    """Solve the contactenergy model with a validation-only SFC C3D8 backend."""

    K = _assemble_c3d8_stiffness(model)
    contact_rows = _contact_jacobian_rows(model)
    K_contact = np.zeros_like(K)
    for J, area in contact_rows:
        K_contact += model.pressure_stiffness * float(area) * np.outer(J, J)
    K_total = K + K_contact
    force = _external_force_vector(model)
    fixed, fixed_values = _fixed_dofs_and_values(model)
    all_dofs = np.arange(K_total.shape[0], dtype=np.int64)
    free = np.setdiff1d(all_dofs, fixed)
    u = np.zeros(K_total.shape[0], dtype=float)
    u[fixed] = fixed_values
    rhs = force[free] - K_total[np.ix_(free, fixed)] @ u[fixed]
    u[free] = np.linalg.solve(K_total[np.ix_(free, free)], rhs)
    gap_values = np.asarray([float(J @ u) for J, _ in contact_rows], dtype=float)
    contact_energy = 0.5 * sum(
        model.pressure_stiffness * float(area) * min(float(gap), 0.0) ** 2
        for gap, (_, area) in zip(gap_values, contact_rows, strict=True)
    )
    metrics = {
        "sfc_c3d8_static_active_contact_count": int(np.count_nonzero(gap_values <= 0.0)),
        "sfc_c3d8_static_min_gap": float(np.min(gap_values)) if gap_values.size else float("inf"),
        "sfc_c3d8_static_contact_energy": float(contact_energy),
        "sfc_c3d8_static_contact_spring_count": len(contact_rows),
    }
    return u.reshape((-1, 3)), metrics


def _von_mises(stress: np.ndarray) -> float:
    sxx, syy, szz, txy, tyz, txz = np.asarray(stress, dtype=float)
    return float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (txy**2 + tyz**2 + txz**2)))


def _hex8_center_strain_stress(model: ContactEnergyModel, U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover one-point C3D8 engineering strain and linear stress.

    This is a validation-only post-process of the CalculiX final displacement
    field. SFC does not solve C3D8 mechanics in this replay.
    """

    C = isotropic_linear_elasticity_matrix(model.material_E, model.material_nu)
    strains: list[np.ndarray] = []
    stresses: list[np.ndarray] = []
    von_mises_values: list[float] = []
    strain_norms: list[float] = []
    for element in model.elements:
        Xe = model.X[element]
        Ue = U[element]
        jacobian = Xe.T @ HEX8_CENTER_NATURAL_GRADIENTS
        gradients = HEX8_CENTER_NATURAL_GRADIENTS @ np.linalg.inv(jacobian)
        displacement_gradient = Ue.T @ gradients
        engineering_strain = np.asarray(
            [
                displacement_gradient[0, 0],
                displacement_gradient[1, 1],
                displacement_gradient[2, 2],
                displacement_gradient[0, 1] + displacement_gradient[1, 0],
                displacement_gradient[1, 2] + displacement_gradient[2, 1],
                displacement_gradient[0, 2] + displacement_gradient[2, 0],
            ],
            dtype=float,
        )
        stress = C @ engineering_strain
        strain_tensor = np.asarray(
            [
                [engineering_strain[0], 0.5 * engineering_strain[3], 0.5 * engineering_strain[5]],
                [0.5 * engineering_strain[3], engineering_strain[1], 0.5 * engineering_strain[4]],
                [0.5 * engineering_strain[5], 0.5 * engineering_strain[4], engineering_strain[2]],
            ],
            dtype=float,
        )
        strains.append(engineering_strain)
        stresses.append(stress)
        von_mises_values.append(_von_mises(stress))
        strain_norms.append(float(np.linalg.norm(strain_tensor)))
    return (
        np.asarray(strains, dtype=float),
        np.asarray(stresses, dtype=float),
        np.asarray(von_mises_values, dtype=float),
        np.asarray(strain_norms, dtype=float),
    )


def _nodal_average(elements: np.ndarray, values: np.ndarray, n_nodes: int) -> np.ndarray:
    sums = np.zeros(n_nodes, dtype=float)
    counts = np.zeros(n_nodes, dtype=float)
    for element, value in zip(elements, np.asarray(values, dtype=float), strict=True):
        sums[element] += float(value)
        counts[element] += 1.0
    return sums / np.maximum(counts, 1.0)


def _subdivide_triangle(vertices: np.ndarray, vertex_values: np.ndarray, *, subdivisions: int = 5) -> tuple[list[np.ndarray], list[float]]:
    n = max(1, int(subdivisions))

    def point(i: int, j: int) -> np.ndarray:
        w1 = i / n
        w2 = j / n
        w0 = 1.0 - w1 - w2
        return w0 * vertices[0] + w1 * vertices[1] + w2 * vertices[2]

    def value(i: int, j: int) -> float:
        w1 = i / n
        w2 = j / n
        w0 = 1.0 - w1 - w2
        return float(w0 * vertex_values[0] + w1 * vertex_values[1] + w2 * vertex_values[2])

    polygons: list[np.ndarray] = []
    values: list[float] = []
    for i in range(n):
        for j in range(n - i):
            tri = np.asarray([point(i, j), point(i + 1, j), point(i, j + 1)], dtype=float)
            polygons.append(tri)
            values.append((value(i, j) + value(i + 1, j) + value(i, j + 1)) / 3.0)
            if i + j < n - 1:
                tri = np.asarray([point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)], dtype=float)
                polygons.append(tri)
                values.append((value(i + 1, j) + value(i + 1, j + 1) + value(i, j + 1)) / 3.0)
    return polygons, values


def _draw_reference_wireframe(ax, X_ref: np.ndarray, triangles: np.ndarray) -> None:
    collection = Poly3DCollection(
        [X_ref[face] for face in triangles],
        facecolors=(0.72, 0.72, 0.72, 0.03),
        edgecolors=(0.08, 0.08, 0.08, 0.18),
        linewidths=0.25,
    )
    ax.add_collection3d(collection)


def _draw_3d_hex_surface(
    ax,
    X_ref: np.ndarray,
    X_plot: np.ndarray,
    triangles: np.ndarray,
    nodal_values: np.ndarray,
    *,
    norm: Normalize,
    cmap: str,
    title: str,
) -> Poly3DCollection:
    _draw_reference_wireframe(ax, X_ref, triangles)
    polygons: list[np.ndarray] = []
    values: list[float] = []
    node_values = np.asarray(nodal_values, dtype=float)
    for face in triangles:
        sub_polygons, sub_values = _subdivide_triangle(X_plot[face], node_values[face])
        polygons.extend(sub_polygons)
        values.extend(sub_values)
    collection = Poly3DCollection(
        polygons,
        cmap=cmap,
        norm=norm,
        edgecolors=(0.16, 0.16, 0.16, 0.16),
        linewidths=0.05,
    )
    collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection3d(collection)
    bounds = np.vstack([X_ref, X_plot])
    mins = np.min(bounds, axis=0)
    maxs = np.max(bounds, axis=0)
    padding = np.maximum(0.05 * (maxs - mins), 1.0e-4)
    ax.set_xlim(float(mins[0] - padding[0]), float(maxs[0] + padding[0]))
    ax.set_ylim(float(mins[1] - padding[1]), float(maxs[1] + padding[1]))
    ax.set_zlim(float(mins[2] - padding[2]), float(maxs[2] + padding[2]))
    ax.set_box_aspect((maxs - mins + 2.0 * padding).clip(min=1.0e-12))
    ax.view_init(elev=22.0, azim=-52.0)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.set_title(title, pad=0.0)
    ax.set_axis_off()
    return collection


def _contact_pressure_nodal_values(model: ContactEnergyModel, U: np.ndarray) -> np.ndarray:
    X_current = model.X + U
    master_faces = _triangles_from_quads(X_current, _surface_quads(model, "Smast"))
    slave_faces = _triangles_from_quads(X_current, _surface_quads(model, "Sslav"))
    candidates = np.arange(master_faces.shape[0], dtype=np.int64)
    sums = np.zeros(model.X.shape[0], dtype=float)
    counts = np.zeros(model.X.shape[0], dtype=float)
    for face in slave_faces:
        triangle = X_current[face]
        result = dynamic_surface_sdf(triangle.mean(axis=0), X_current, master_faces, candidates)
        pressure = model.pressure_stiffness * max(-float(result.g), 0.0)
        sums[face] += pressure
        counts[face] += 1.0
    return sums / np.maximum(counts, 1.0)


def _write_visualization_artifacts(out_dir: Path, model: ContactEnergyModel, U: np.ndarray, U_sfc: np.ndarray) -> tuple[list[Row], list[Row]]:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    strains, stresses, von_mises_values, strain_norms = _hex8_center_strain_stress(model, U)
    sfc_strains, sfc_stresses, sfc_von_mises_values, sfc_strain_norms = _hex8_center_strain_stress(model, U_sfc)
    nodal_vm = _nodal_average(model.elements, von_mises_values, model.X.shape[0])
    nodal_sfc_vm = _nodal_average(model.elements, sfc_von_mises_values, model.X.shape[0])
    nodal_vm_error = _nodal_average(model.elements, np.abs(sfc_von_mises_values - von_mises_values), model.X.shape[0])
    nodal_strain = _nodal_average(model.elements, strain_norms, model.X.shape[0])
    displacement_magnitude = np.linalg.norm(U, axis=1)
    displacement_error_magnitude = np.linalg.norm(U_sfc - U, axis=1)
    nodal_pressure = _contact_pressure_nodal_values(model, U)

    deformation_scale = 1000.0
    X_plot = model.X + deformation_scale * U
    boundary_triangles, _ = extract_boundary_triangles(model.elements, model.X, element_type="hex8")

    stress_norm = Normalize(vmin=0.0, vmax=max(float(np.max(nodal_vm)), 1.0e-30))
    strain_norm = Normalize(vmin=0.0, vmax=max(float(np.max(nodal_strain)), 1.0e-30))
    displacement_norm = Normalize(vmin=0.0, vmax=max(float(np.max(displacement_magnitude)), 1.0e-30))
    pressure_norm = Normalize(vmin=0.0, vmax=max(float(np.max(nodal_pressure)), 1.0e-30))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), subplot_kw={"projection": "3d"})
    panels = [
        (nodal_vm, "von Mises stress", stress_norm, "viridis"),
        (nodal_strain, "engineering strain norm", strain_norm, "plasma"),
        (displacement_magnitude, "displacement magnitude", displacement_norm, "cividis"),
    ]
    collections: list[Poly3DCollection] = []
    for ax, (values, title, norm, cmap) in zip(axes, panels, strict=True):
        collections.append(
            _draw_3d_hex_surface(
                ax,
                model.X,
                X_plot,
                boundary_triangles,
                values,
                norm=norm,
                cmap=cmap,
                title=title,
            )
        )
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.04, top=0.80, wspace=0.10)
    colorbar_specs = [(collections[0], "stress"), (collections[1], "strain"), (collections[2], "|U|")]
    for ax, (collection, label) in zip(axes, colorbar_specs, strict=True):
        fig.colorbar(collection, ax=ax, shrink=0.58, pad=0.02, label=label)
    fig.suptitle(f"CalculiX C3D8 contactenergy final state, displacement scale {deformation_scale:g}x", fontsize=13)
    stress_png = figures_dir / "calculix_contactenergy_stress_strain_3d.png"
    stress_pdf = figures_dir / "calculix_contactenergy_stress_strain_3d.pdf"
    fig.savefig(stress_png, dpi=180, bbox_inches="tight")
    fig.savefig(stress_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), subplot_kw={"projection": "3d"})
    pressure_collection = _draw_3d_hex_surface(
        axes[0],
        model.X,
        X_plot,
        boundary_triangles,
        nodal_pressure,
        norm=pressure_norm,
        cmap="magma",
        title="SFC dynamic-SDF pressure",
    )
    vm_collection = _draw_3d_hex_surface(
        axes[1],
        model.X,
        X_plot,
        boundary_triangles,
        nodal_vm,
        norm=stress_norm,
        cmap="viridis",
        title="CalculiX-displacement stress",
    )
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.04, top=0.80, wspace=0.08)
    fig.colorbar(pressure_collection, ax=axes[0], shrink=0.58, pad=0.02, label="pressure")
    fig.colorbar(vm_collection, ax=axes[1], shrink=0.58, pad=0.02, label="von Mises")
    fig.suptitle(f"Dynamic-SDF contact replay on deformed C3D8 boundary, {deformation_scale:g}x", fontsize=12)
    pressure_png = figures_dir / "calculix_contactenergy_contact_pressure_3d.png"
    pressure_pdf = figures_dir / "calculix_contactenergy_contact_pressure_3d.pdf"
    fig.savefig(pressure_png, dpi=180, bbox_inches="tight")
    fig.savefig(pressure_pdf, bbox_inches="tight")
    plt.close(fig)

    error_norm = Normalize(vmin=0.0, vmax=max(float(np.max(nodal_vm_error)), 1.0e-30))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), subplot_kw={"projection": "3d"})
    panels = [
        (nodal_sfc_vm, "SFC C3D8 von Mises", stress_norm, "viridis"),
        (nodal_vm, "CalculiX von Mises", stress_norm, "viridis"),
        (nodal_vm_error, "absolute stress error", error_norm, "magma"),
    ]
    collections = []
    for ax, (values, title, norm, cmap) in zip(axes, panels, strict=True):
        collections.append(
            _draw_3d_hex_surface(
                ax,
                model.X,
                X_plot,
                boundary_triangles,
                values,
                norm=norm,
                cmap=cmap,
                title=title,
            )
        )
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.04, top=0.80, wspace=0.10)
    for ax, (collection, label) in zip(axes, [(collections[0], "von Mises"), (collections[1], "von Mises"), (collections[2], "abs. error")], strict=True):
        fig.colorbar(collection, ax=ax, shrink=0.58, pad=0.02, label=label)
    fig.suptitle(f"SFC validation C3D8 static solve vs CalculiX, displacement scale {deformation_scale:g}x", fontsize=13)
    error_png = figures_dir / "calculix_contactenergy_sfc_c3d8_error_3d.png"
    error_pdf = figures_dir / "calculix_contactenergy_sfc_c3d8_error_3d.pdf"
    fig.savefig(error_png, dpi=180, bbox_inches="tight")
    fig.savefig(error_pdf, bbox_inches="tight")
    plt.close(fig)

    cloud_rows: list[Row] = []
    for element_id, strain, stress, vm, strain_norm_value, sfc_strain, sfc_stress, sfc_vm, sfc_strain_norm in zip(
        model.element_ids,
        strains,
        stresses,
        von_mises_values,
        strain_norms,
        sfc_strains,
        sfc_stresses,
        sfc_von_mises_values,
        sfc_strain_norms,
        strict=True,
    ):
        cloud_rows.append(
            {
                "element_id": int(element_id),
                "sfc_strain_xx": float(sfc_strain[0]),
                "sfc_strain_yy": float(sfc_strain[1]),
                "sfc_strain_zz": float(sfc_strain[2]),
                "sfc_strain_xy": float(sfc_strain[3]),
                "sfc_strain_yz": float(sfc_strain[4]),
                "sfc_strain_xz": float(sfc_strain[5]),
                "strain_xx": float(strain[0]),
                "strain_yy": float(strain[1]),
                "strain_zz": float(strain[2]),
                "strain_xy": float(strain[3]),
                "strain_yz": float(strain[4]),
                "strain_xz": float(strain[5]),
                "sfc_stress_xx": float(sfc_stress[0]),
                "sfc_stress_yy": float(sfc_stress[1]),
                "sfc_stress_zz": float(sfc_stress[2]),
                "sfc_stress_xy": float(sfc_stress[3]),
                "sfc_stress_yz": float(sfc_stress[4]),
                "sfc_stress_xz": float(sfc_stress[5]),
                "stress_xx": float(stress[0]),
                "stress_yy": float(stress[1]),
                "stress_zz": float(stress[2]),
                "stress_xy": float(stress[3]),
                "stress_yz": float(stress[4]),
                "stress_xz": float(stress[5]),
                "sfc_von_mises": float(sfc_vm),
                "von_mises": float(vm),
                "von_mises_abs_error": abs(float(sfc_vm) - float(vm)),
                "sfc_engineering_strain_norm": float(sfc_strain_norm),
                "engineering_strain_norm": float(strain_norm_value),
                "engineering_strain_norm_abs_error": abs(float(sfc_strain_norm) - float(strain_norm_value)),
            }
        )

    plot_rows = [
        {
            "plot": "calculix_contactenergy_stress_strain_3d",
            "png": stress_png.relative_to(out_dir).as_posix(),
            "pdf": stress_pdf.relative_to(out_dir).as_posix(),
            "status": "ok",
            "details": "C3D8 stress/strain post-processed from CalculiX final displacements; deformed geometry plotted with scale factor",
        },
        {
            "plot": "calculix_contactenergy_contact_pressure_3d",
            "png": pressure_png.relative_to(out_dir).as_posix(),
            "pdf": pressure_pdf.relative_to(out_dir).as_posix(),
            "status": "ok",
            "details": "SFC dynamic-SDF replay pressure on the current triangulated C3D8 boundary",
        },
        {
            "plot": "calculix_contactenergy_sfc_c3d8_error_3d",
            "png": error_png.relative_to(out_dir).as_posix(),
            "pdf": error_pdf.relative_to(out_dir).as_posix(),
            "status": "ok",
            "details": "validation-only SFC C3D8 static solve compared to CalculiX von Mises stress on the same boundary",
        },
    ]
    return plot_rows, cloud_rows


def _relative_vector_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)) / max(np.linalg.norm(expected), 1.0e-30))


def _c3d8_static_comparison_metrics(model: ContactEnergyModel, U_calculix: np.ndarray, U_sfc: np.ndarray) -> Row:
    strain_calc, stress_calc, vm_calc, strain_norm_calc = _hex8_center_strain_stress(model, U_calculix)
    strain_sfc, stress_sfc, vm_sfc, strain_norm_sfc = _hex8_center_strain_stress(model, U_sfc)
    return {
        "sfc_c3d8_displacement_l2_rel_error": _relative_vector_error(U_sfc.ravel(), U_calculix.ravel()),
        "sfc_c3d8_displacement_linf_abs_error": float(np.max(np.abs(U_sfc - U_calculix))),
        "sfc_c3d8_strain_l2_rel_error": _relative_vector_error(strain_sfc.ravel(), strain_calc.ravel()),
        "sfc_c3d8_stress_l2_rel_error": _relative_vector_error(stress_sfc.ravel(), stress_calc.ravel()),
        "sfc_c3d8_von_mises_l2_rel_error": _relative_vector_error(vm_sfc, vm_calc),
        "sfc_c3d8_strain_norm_l2_rel_error": _relative_vector_error(strain_norm_sfc, strain_norm_calc),
        "sfc_c3d8_max_von_mises_abs_error": float(np.max(np.abs(vm_sfc - vm_calc))),
        "sfc_c3d8_max_engineering_strain_norm_abs_error": float(np.max(np.abs(strain_norm_sfc - strain_norm_calc))),
    }


def _write_error_artifacts(out_dir: Path, metrics: Row) -> tuple[list[Row], list[Row]]:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    threshold = 1.0e-5
    error_rows = [
        {
            "quantity": "normal_force_z",
            "calculix_reference": float(metrics["calculix_force_reference_z_abs"]),
            "sfc_dynamic_sdf_replay": float(metrics["sfc_dynamic_sdf_contact_force_z"]),
            "absolute_error": abs(float(metrics["sfc_dynamic_sdf_contact_force_z"]) - float(metrics["calculix_force_reference_z_abs"])),
            "relative_error": float(metrics["contact_force_rel_error"]),
            "acceptance_threshold": threshold,
            "status": "ok" if float(metrics["contact_force_rel_error"]) <= threshold else "check",
        },
        {
            "quantity": "contact_energy",
            "calculix_reference": float(metrics["calculix_total_contact_spring_energy"]),
            "sfc_dynamic_sdf_replay": float(metrics["sfc_dynamic_sdf_contact_energy"]),
            "absolute_error": abs(float(metrics["sfc_dynamic_sdf_contact_energy"]) - float(metrics["calculix_total_contact_spring_energy"])),
            "relative_error": float(metrics["contact_energy_rel_error"]),
            "acceptance_threshold": threshold,
            "status": "ok" if float(metrics["contact_energy_rel_error"]) <= threshold else "check",
        },
        {
            "quantity": "c3d8_displacement",
            "calculix_reference": 1.0,
            "sfc_dynamic_sdf_replay": 1.0 + float(metrics["sfc_c3d8_displacement_l2_rel_error"]),
            "absolute_error": float(metrics["sfc_c3d8_displacement_l2_rel_error"]),
            "relative_error": float(metrics["sfc_c3d8_displacement_l2_rel_error"]),
            "acceptance_threshold": float("nan"),
            "status": "diagnostic",
        },
        {
            "quantity": "c3d8_von_mises",
            "calculix_reference": 1.0,
            "sfc_dynamic_sdf_replay": 1.0 + float(metrics["sfc_c3d8_von_mises_l2_rel_error"]),
            "absolute_error": float(metrics["sfc_c3d8_von_mises_l2_rel_error"]),
            "relative_error": float(metrics["sfc_c3d8_von_mises_l2_rel_error"]),
            "acceptance_threshold": float("nan"),
            "status": "diagnostic",
        },
    ]

    labels = ["normal force", "contact energy"]
    rows_for_bars = error_rows[:2]
    normalized_reference = np.ones(len(error_rows), dtype=float)
    normalized_replay = np.asarray(
        [
            float(row["sfc_dynamic_sdf_replay"]) / max(abs(float(row["calculix_reference"])), 1.0e-30)
            for row in rows_for_bars
        ],
        dtype=float,
    )
    normalized_reference = np.ones(len(rows_for_bars), dtype=float)
    relative_errors = np.asarray([float(row["relative_error"]) for row in rows_for_bars], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    axes[0].bar(x - width / 2.0, normalized_reference, width, label="CalculiX reference", color="#4c78a8")
    axes[0].bar(x + width / 2.0, normalized_replay, width, label="SFC replay", color="#f58518")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("normalized value")
    axes[0].set_ylim(0.9999999, 1.0000001)
    axes[0].grid(True, axis="y", color="0.9", linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, relative_errors, color="#54a24b")
    axes[1].axhline(threshold, color="#e45756", linestyle="--", linewidth=1.0, label="locked threshold")
    axes[1].set_xticks(x, labels)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("relative error")
    axes[1].grid(True, axis="y", color="0.9", linewidth=0.5, which="both")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("CalculiX contactenergy replay error")
    fig.tight_layout()
    png = figures_dir / "calculix_contactenergy_error_metrics.png"
    pdf = figures_dir / "calculix_contactenergy_error_metrics.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    plot_rows = [
        {
            "plot": "calculix_contactenergy_error_metrics",
            "png": png.relative_to(out_dir).as_posix(),
            "pdf": pdf.relative_to(out_dir).as_posix(),
            "status": "ok" if all(row["status"] == "ok" for row in error_rows) else "check",
            "details": "normalized CalculiX-vs-SFC replay values and log-scale relative errors for force and contact energy",
        }
    ]
    return plot_rows, error_rows


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1.0e-30)


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, metrics: Row, command_row: Row, claims: list[Row]) -> None:
    lines = [
        "# CalculiX Contactenergy C3D8 Replay",
        "",
        "This validation uses the official CalculiX `contactenergy.inp` C3D8",
        "surface-to-surface contact-energy test as an external contact-law",
        "reference. The primary SFC evidence replays the final CalculiX",
        "deformed geometry with the current-surface dynamic SDF query and the",
        "same linear pressure-overclosure law. A separate validation-only SFC",
        "C3D8 static backend solves the same input to produce direct field-error",
        "plots.",
        "",
        "## Command",
        "",
        "```bash",
        "python validation/run_calculix_contactenergy_replay.py --out-dir results/calculix_contactenergy_replay",
        "```",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| CalculiX total contact spring energy | {metrics['calculix_total_contact_spring_energy']} |",
        f"| SFC dynamic-SDF replay contact energy | {metrics['sfc_dynamic_sdf_contact_energy']} |",
        f"| contact energy relative error | {metrics['contact_energy_rel_error']} |",
        f"| CalculiX force reference magnitude | {metrics['calculix_force_reference_z_abs']} |",
        f"| SFC dynamic-SDF replay force z | {metrics['sfc_dynamic_sdf_contact_force_z']} |",
        f"| contact force relative error | {metrics['contact_force_rel_error']} |",
        f"| SFC min gap | {metrics['sfc_dynamic_sdf_min_gap']} |",
        f"| max post-processed von Mises stress | {metrics['max_postprocessed_von_mises']} |",
        f"| max post-processed engineering strain norm | {metrics['max_postprocessed_engineering_strain_norm']} |",
        f"| max CalculiX displacement magnitude | {metrics['max_calculix_displacement_magnitude']} |",
        f"| SFC validation C3D8 displacement L2 relative error | {metrics['sfc_c3d8_displacement_l2_rel_error']} |",
        f"| SFC validation C3D8 stress L2 relative error | {metrics['sfc_c3d8_stress_l2_rel_error']} |",
        f"| SFC validation C3D8 von Mises L2 relative error | {metrics['sfc_c3d8_von_mises_l2_rel_error']} |",
        "",
        "## Figures",
        "",
        "- `figures/calculix_contactenergy_stress_strain_3d.png`: C3D8",
        "  stress, strain, and displacement clouds post-processed from the final",
        "  CalculiX displacement field. The deformed shape is plotted with a",
        "  labeled magnification factor so that the small static displacement is",
        "  visible.",
        "- `figures/calculix_contactenergy_contact_pressure_3d.png`: SFC",
        "  dynamic-SDF replay contact pressure on the same current C3D8 boundary.",
        "- `figures/calculix_contactenergy_error_metrics.png`: normalized",
        "  CalculiX-vs-SFC replay values and log-scale relative errors for",
        "  normal force and contact energy.",
        "- `figures/calculix_contactenergy_sfc_c3d8_error_3d.png`: direct",
        "  validation-only SFC C3D8 static solve compared with the CalculiX",
        "  C3D8 stress field on the same boundary.",
        "",
        "## Claims",
        "",
        "| Claim | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for claim in claims:
        lines.append(f"| {claim['claim']} | {claim['supported']} | {claim['evidence_csv']}::{claim['evidence_field']} |")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- This is a C3D8 external contact-law/energy replay, not a TET4 trajectory",
            "  equivalence claim.",
            "- The validation-only SFC C3D8 static backend is included for direct",
            "  field-error plotting; it is not the main solver element type.",
            "- Dynamic SDF is evaluated on triangulated current C3D8 boundary faces.",
            "- Friction, self-contact, hard contact, and nonlinear material behavior are",
            "  outside this replay.",
            "",
            "## External Solver",
            "",
            f"- solver: {command_row['external_solver']}",
            f"- version: {command_row['external_solver_version']}",
            f"- completed: {command_row['completed']}",
            f"- command: `{command_row['command']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_contactenergy_replay(*, out_dir: Path, skip_calculix: bool, timeout: int) -> dict[str, list[Row]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "calculix_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    inp_text, inp_source = _read_wsl_text(WSL_CONTACTENERGY_INP_CANDIDATES)
    inp_path = run_dir / "contactenergy.inp"
    inp_path.write_text(inp_text, encoding="utf-8")

    command_row: Row
    if not skip_calculix and _calculix_available():
        wsl_run_dir = _wsl_path(run_dir)
        command = f"cd {_wsl_quote(wsl_run_dir)} && ccx contactenergy"
        proc = subprocess.run(
            ["wsl", "--exec", "bash", "-lc", command],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
        dat_path = run_dir / "contactenergy.dat"
        if proc.returncode != 0 or not dat_path.exists():
            raise RuntimeError("CalculiX contactenergy run failed; use --skip-calculix for reference replay")
        command_row = {
            "case": "contactenergy",
            "external_solver": "CalculiX",
            "external_solver_version": _calculix_version(),
            "command": f"wsl --exec bash -lc \"{command}\"",
            "input_source": inp_source,
            "inp": str(inp_path.relative_to(out_dir)),
            "dat": str(dat_path.relative_to(out_dir)),
            "completed": "true",
            "used_dat_ref": "false",
        }
    else:
        dat_text, dat_source = _read_wsl_text(WSL_CONTACTENERGY_DAT_REF_CANDIDATES)
        dat_path = run_dir / "contactenergy.dat.ref"
        dat_path.write_text(dat_text, encoding="utf-8")
        command_row = {
            "case": "contactenergy",
            "external_solver": "CalculiX",
            "external_solver_version": _calculix_version(),
            "command": "not run; official contactenergy.dat.ref replayed",
            "input_source": inp_source,
            "inp": str(inp_path.relative_to(out_dir)),
            "dat": str(dat_path.relative_to(out_dir)),
            "completed": "true",
            "used_dat_ref": "true",
        }

    model = parse_contactenergy_input(inp_text)
    dat_text = dat_path.read_text(encoding="utf-8")
    U, RF, calculix_contact_energy, cels_rows = parse_contactenergy_dat(dat_text, model.node_ids)
    U_sfc_c3d8, sfc_c3d8_static_metrics = solve_sfc_c3d8_contactenergy_static(model)
    replay = replay_contactenergy_with_dynamic_sdf(model, U)
    c3d8_comparison_metrics = _c3d8_static_comparison_metrics(model, U, U_sfc_c3d8)
    plot_rows, stress_strain_rows = _write_visualization_artifacts(out_dir, model, U, U_sfc_c3d8)

    force_error = _relative_error(float(replay["sfc_dynamic_sdf_contact_force_z"]), model.z_load_reference)
    energy_error = _relative_error(float(replay["sfc_dynamic_sdf_contact_energy"]), calculix_contact_energy)
    status = "ok" if force_error <= 1.0e-5 and energy_error <= 1.0e-5 else "check"
    metrics: Row = {
        "case": "calculix_contactenergy_c3d8_replay",
        "element_type": "c3d8",
        "reference_scope": "contact_law_energy_replay_not_tet4_trajectory",
        "dynamic_sdf_backend": "current_surface_triangulated_c3d8_faces",
        "pressure_stiffness": model.pressure_stiffness,
        "nodes": model.X.shape[0],
        "elements": model.elements.shape[0],
        "calculix_force_reference_z_abs": model.z_load_reference,
        "calculix_total_contact_spring_energy": calculix_contact_energy,
        "calculix_raw_cels_row_count": len(cels_rows),
        "calculix_rf_z_sum": float(np.sum(RF[:, 2])),
        "max_postprocessed_von_mises": max(float(row["von_mises"]) for row in stress_strain_rows),
        "max_postprocessed_engineering_strain_norm": max(float(row["engineering_strain_norm"]) for row in stress_strain_rows),
        "max_calculix_displacement_magnitude": float(np.max(np.linalg.norm(U, axis=1))),
        "visualization_deformation_scale": 1000.0,
        **sfc_c3d8_static_metrics,
        **c3d8_comparison_metrics,
        **replay,
        "contact_force_rel_error": force_error,
        "contact_energy_rel_error": energy_error,
        "status": status,
    }
    error_plot_rows, error_rows = _write_error_artifacts(out_dir, metrics)
    plot_rows.extend(error_plot_rows)
    claims = [
        {
            "claim": "calculix_contactenergy_c3d8_reference_available",
            "supported": "true",
            "evidence_csv": "calculix_contactenergy_replay.csv",
            "evidence_field": "status",
            "details": "official CalculiX contactenergy C3D8 reference parsed",
        },
        {
            "claim": "dynamic_sdf_replays_c3d8_contact_energy",
            "supported": str(status == "ok").lower(),
            "evidence_csv": "calculix_contactenergy_replay.csv",
            "evidence_field": "contact_energy_rel_error",
            "details": "current-surface dynamic SDF on triangulated C3D8 faces matches CalculiX total CELS",
        },
        {
            "claim": "dynamic_sdf_is_not_tet4_bound",
            "supported": "true",
            "evidence_csv": "calculix_contactenergy_replay.csv",
            "evidence_field": "dynamic_sdf_backend",
            "details": "replay uses C3D8 boundary quads triangulated into current surface triangles",
        },
    ]

    metric_fields = list(metrics.keys())
    _write_csv(out_dir / "calculix_contactenergy_replay.csv", metric_fields, [metrics])
    _write_csv(out_dir / "calculix_contactenergy_claims.csv", list(claims[0].keys()), claims)
    _write_csv(out_dir / "calculix_contactenergy_commands.csv", list(command_row.keys()), [command_row])
    _write_csv(
        out_dir / "calculix_contactenergy_raw_cels.csv",
        ["slave_element", "slave_face", "cels"],
        cels_rows,
    )
    _write_csv(out_dir / "calculix_contactenergy_plots.csv", list(plot_rows[0].keys()), plot_rows)
    _write_csv(out_dir / "calculix_contactenergy_stress_strain_cloud.csv", list(stress_strain_rows[0].keys()), stress_strain_rows)
    _write_csv(out_dir / "calculix_contactenergy_error_metrics.csv", list(error_rows[0].keys()), error_rows)
    _write_summary(out_dir / "calculix_contactenergy_summary.md", metrics, command_row, claims)
    return {
        "metrics": [metrics],
        "claims": claims,
        "commands": [command_row],
        "raw_cels": cels_rows,
        "plots": plot_rows,
        "stress_strain_cloud": stress_strain_rows,
        "error_metrics": error_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "calculix_contactenergy_replay")
    parser.add_argument("--quick", action="store_true", help="Accepted for consistency; this case is already small.")
    parser.add_argument("--skip-calculix", action="store_true", help="Replay the official contactenergy.dat.ref instead of running ccx.")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    run_contactenergy_replay(out_dir=args.out_dir, skip_calculix=bool(args.skip_calculix), timeout=int(args.timeout))
    print(f"Wrote CalculiX contactenergy replay outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
