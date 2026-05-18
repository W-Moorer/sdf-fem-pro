"""Independent native-contact comparison for contact benchmark schemes.

This runner is stricter than equivalent-load or replay comparisons.  CalculiX
receives its own native ``*CONTACT PAIR`` surface-to-surface input deck and
solves contact with CalculiX.  SFC independently builds the corresponding
contact model and solves contact with ``DynamicNarrowBandSDF + field_contact``.
The two independent solver outputs are then compared.

The geometry families are inspired by open contact benchmark modeling schemes,
but the comparison is not a validation against FuzzyContact published results.
The purpose is to test independent CalculiX-native-contact versus SFC-SDF-contact
agreement on matched local models.  If the active-set/contact-law
discretizations diverge, the CSV rows report ``failed_native_solver_agreement``.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from dataclasses import dataclass, replace
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

from sfc.contact import (  # noqa: E402
    SurfaceQuadratureCache,
    SurfaceSample,
    node_to_surface_field_penalty_response,
    surface_to_surface_field_penalty_response,
    surface_to_surface_field_penalty_response_vectorized,
    triangle_surface_quadrature_cache,
    triangle_surface_quadrature_samples,
)
from sfc.fem import assemble_stiffness_matrix  # noqa: E402
from sfc.fem.constraints import eliminate_fixed_dofs, expand_reduced_vector, fixed_dofs_from_node_set  # noqa: E402
from sfc.fem.material import isotropic_linear_elasticity_matrix  # noqa: E402
from sfc.fem.tet4 import tet4_strain_displacement_matrix  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402
from validation.run_calculix_drop_impact_comparison import (  # noqa: E402
    _boundary_face_data,
    _calculix_version,
    _parse_calculix_dat_displacements,
    _wsl_path,
    calculix_available,
)
from validation.run_fuzzycontact_published_benchmarks import (  # noqa: E402
    PublishedCase,
    _area_weighted_penetration_sum,
    _linear_compression_reference,
    _make_cases,
    _von_mises_voigt,
)

Row = dict[str, Any]


def _configure_times_fonts(matplotlib_module: Any) -> None:
    matplotlib_module.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


@dataclass(frozen=True, slots=True)
class DriverMesh:
    X: np.ndarray
    elements: np.ndarray
    contact_faces: np.ndarray
    contact_refs: list[tuple[int, str]]
    contact_area: np.ndarray


@dataclass(frozen=True, slots=True)
class NativeContactCase:
    case: PublishedCase
    driver: DriverMesh
    closures: tuple[float, ...]
    master_top_refs: list[tuple[int, str]]
    calibrated_stiffness: float
    sfc_backend: str
    quadrature_order: int
    sdf_spacing_scale: float
    sfc_relaxation: float
    sfc_max_iterations: int
    batch_projection_threshold: int
    surface_quadrature_cache: SurfaceQuadratureCache | None


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


def _top_surface_refs(mesh: VolumeMesh) -> list[tuple[int, str]]:
    faces, refs, _nodes, _areas = _boundary_face_data(mesh.X, mesh.elements)
    zmax = float(np.max(mesh.X[:, 2]))
    out: list[tuple[int, str]] = []
    for face, ref in zip(faces, refs, strict=True):
        if np.all(np.isclose(mesh.X[np.asarray(face, dtype=np.int64), 2], zmax)):
            out.append(ref)
    if not out:
        raise RuntimeError("failed to find master top surface refs")
    return out


def _disk_cells(radius: float, resolution: int) -> list[tuple[float, float, float, float]]:
    edges = np.linspace(-radius, radius, resolution + 1)
    cells: list[tuple[float, float, float, float]] = []
    for i in range(resolution):
        for j in range(resolution):
            x0, x1 = float(edges[i]), float(edges[i + 1])
            y0, y1 = float(edges[j]), float(edges[j + 1])
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            if cx * cx + cy * cy <= radius * radius:
                cells.append((x0, x1, y0, y1))
    return cells


def _add_node(nodes: list[tuple[float, float, float]], lookup: dict[tuple[float, float, float], int], xyz: tuple[float, float, float]) -> int:
    key = tuple(round(float(v), 12) for v in xyz)
    if key not in lookup:
        lookup[key] = len(nodes)
        nodes.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
    return lookup[key]


def _make_layered_driver(
    *,
    cells: list[tuple[float, float, float, float]],
    bottom_z,
    thickness: float,
) -> DriverMesh:
    nodes: list[tuple[float, float, float]] = []
    lookup: dict[tuple[float, float, float], int] = {}
    elements: list[tuple[int, int, int, int]] = []
    for x0, x1, y0, y1 in cells:
        bottom = [
            (x0, y0, float(bottom_z(x0, y0))),
            (x1, y0, float(bottom_z(x1, y0))),
            (x1, y1, float(bottom_z(x1, y1))),
            (x0, y1, float(bottom_z(x0, y1))),
        ]
        top = [(x, y, z + float(thickness)) for x, y, z in bottom]
        ids = [_add_node(nodes, lookup, xyz) for xyz in [*bottom, *top]]
        b0, b1, b2, b3, t0, t1, t2, t3 = ids
        elements.extend(
            [
                (b0, b1, b3, t0),
                (b1, b2, b3, t2),
                (b1, b3, t0, t2),
                (b3, t0, t3, t2),
                (b1, t0, t1, t2),
            ]
        )
    mesh = VolumeMesh(np.asarray(nodes, dtype=float), np.asarray(elements, dtype=np.int64), element_type="tet4")
    faces, refs, _surface_nodes, _areas = _boundary_face_data(mesh.X, mesh.elements)
    bottom_nodes = np.nonzero(np.isclose(mesh.X[:, 2], np.asarray([bottom_z(x, y) for x, y, _z in mesh.X], dtype=float)))[0]
    bottom_set = {int(v) for v in bottom_nodes}
    contact_faces: list[np.ndarray] = []
    contact_refs: list[tuple[int, str]] = []
    areas: list[float] = []
    for face, ref in zip(faces, refs, strict=True):
        if all(int(node) in bottom_set for node in face):
            f = np.asarray(face, dtype=np.int64)
            tri = mesh.X[f]
            area = 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
            if area > 1.0e-14:
                contact_faces.append(f)
                contact_refs.append(ref)
                areas.append(area)
    if not contact_faces:
        raise RuntimeError("driver has no contact faces")
    return DriverMesh(mesh.X, mesh.elements, np.asarray(contact_faces, dtype=np.int64), contact_refs, np.asarray(areas, dtype=float))


def _driver_for_case(case: PublishedCase, *, quick: bool) -> DriverMesh:
    top_z = float(np.max(case.body.mesh.X[:, 2]))
    if case.problem == "problem_1":
        radius = 1.0
        sample_radius = 0.45
        res = 8 if quick else 14

        def zfun(x: float, y: float) -> float:
            r2 = x * x + y * y
            return top_z + radius - math.sqrt(max(radius * radius - r2, 1.0e-12))

        return _make_layered_driver(cells=_disk_cells(sample_radius, res), bottom_z=zfun, thickness=0.12)
    if case.problem == "problem_3":
        half = 0.006 if quick else 0.012
        res = 8 if quick else 16

        def zfun(x: float, y: float) -> float:
            return top_z + abs(float(x))

        edges = np.linspace(-half, half, res + 1)
        cells = [(float(edges[i]), float(edges[i + 1]), float(edges[j]), float(edges[j + 1])) for i in range(res) for j in range(res)]
        return _make_layered_driver(cells=cells, bottom_z=zfun, thickness=0.15)
    if case.problem == "problem_4":
        radius = 4.75
        res = 6 if quick else 10

        def zfun(x: float, y: float) -> float:
            return top_z

        return _make_layered_driver(cells=_disk_cells(radius, res), bottom_z=zfun, thickness=0.4)
    raise ValueError(f"unsupported problem: {case.problem}")


def _calibrated_case(
    case: PublishedCase,
    driver: DriverMesh,
    *,
    sfc_backend: str = "surface_to_surface_vectorized",
    quadrature_order: int = 7,
    sdf_spacing_scale: float = 1.0,
    sfc_relaxation: float = 0.55,
    sfc_max_iterations: int = 15,
    batch_projection_threshold: int = 5_000_000,
) -> NativeContactCase:
    if not (0.0 < float(sfc_relaxation) <= 1.0):
        raise ValueError("sfc_relaxation must be in (0, 1]")
    if int(sfc_max_iterations) <= 0:
        raise ValueError("sfc_max_iterations must be positive")
    if int(batch_projection_threshold) < 0:
        raise ValueError("batch_projection_threshold must be non-negative")
    spacing = float(case.spacing_mm) * float(sdf_spacing_scale)
    sdf = DynamicNarrowBandSDF.build(
        case.body.mesh.X,
        case.top_faces,
        spacing=spacing,
        band_radius=case.band_radius_mm,
        padding=case.band_radius_mm,
        cell_size=spacing,
    )
    if case.problem == "problem_4":
        target_closure = _linear_compression_reference(case, float(case.reference_loads_N[-1]))
        target_load = float(case.reference_loads_N[-1])
        closures = tuple(float(target_closure) * factor for factor in (0.25, 0.5, 0.75, 1.0))
    else:
        target_closure = float(case.reference_metric_value)
        target_load = float(case.reference_loads_N[-1])
        closures = tuple(float(target_closure) * factor for factor in (0.25, 0.5, 0.75, 1.0))
    integral = _driver_penetration_integral(
        case,
        driver,
        sdf,
        target_closure,
        sfc_backend=sfc_backend,
        quadrature_order=quadrature_order,
    )
    stiffness = target_load / max(integral, 1.0e-15)
    cache = (
        triangle_surface_quadrature_cache(driver.contact_faces, driver.X, order=quadrature_order)
        if sfc_backend == "surface_to_surface_vectorized"
        else None
    )
    return NativeContactCase(
        case=replace(case, contact_pressure_stiffness_N_per_mm3=stiffness),
        driver=driver,
        closures=closures,
        master_top_refs=_top_surface_refs(case.body.mesh),
        calibrated_stiffness=float(stiffness),
        sfc_backend=sfc_backend,
        quadrature_order=int(quadrature_order),
        sdf_spacing_scale=float(sdf_spacing_scale),
        sfc_relaxation=float(sfc_relaxation),
        sfc_max_iterations=int(sfc_max_iterations),
        batch_projection_threshold=int(batch_projection_threshold),
        surface_quadrature_cache=cache,
    )


def _driver_x_current(driver: DriverMesh, closure: float) -> np.ndarray:
    out = driver.X.copy()
    out[:, 2] -= float(closure)
    return out


def _driver_centroid_points(driver: DriverMesh, closure: float) -> np.ndarray:
    pts = np.mean(driver.X[driver.contact_faces], axis=1)
    out = pts.copy()
    out[:, 2] -= float(closure)
    return out


def _driver_surface_quadrature_points(driver: DriverMesh, closure: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    driver_current = _driver_x_current(driver, closure)
    samples, weights = triangle_surface_quadrature_samples(driver.contact_faces, driver.X, order=order)
    points = np.vstack([sample.point(driver_current) for sample in samples]) if samples else np.empty((0, 3), dtype=float)
    return points, weights


def _driver_penetration_integral(
    case: PublishedCase,
    driver: DriverMesh,
    sdf: DynamicNarrowBandSDF,
    closure: float,
    *,
    sfc_backend: str,
    quadrature_order: int,
) -> float:
    if sfc_backend == "node_to_surface":
        points = _driver_centroid_points(driver, closure)
        weights = driver.contact_area
    elif sfc_backend in {"surface_to_surface", "surface_to_surface_vectorized"}:
        points, weights = _driver_surface_quadrature_points(driver, closure, quadrature_order)
    else:
        raise ValueError(f"unsupported SFC contact backend: {sfc_backend}")
    total = 0.0
    for point, area in zip(points, weights, strict=True):
        try:
            gap = sdf.query_phi(point)
        except ValueError:
            continue
        total += float(area) * max(-float(gap), 0.0)
    return float(total)


def _sample_list(n: int) -> list[SurfaceSample]:
    return [
        SurfaceSample(np.asarray([i], dtype=np.int64), np.asarray([1.0], dtype=float), np.asarray([0], dtype=np.int64))
        for i in range(n)
    ]


def _sfc_contact_force(native: NativeContactCase, u: np.ndarray, closure: float) -> tuple[np.ndarray, Row]:
    case = native.case
    driver = native.driver
    x_master = case.body.mesh.X + u.reshape((-1, 3))
    if native.sfc_backend == "node_to_surface":
        slave_x_current = _driver_centroid_points(driver, closure)
        samples = _sample_list(slave_x_current.shape[0])
        area_weights = driver.contact_area
        n_slave = slave_x_current.shape[0] * 3
        required_points = slave_x_current
    elif native.sfc_backend in {"surface_to_surface", "surface_to_surface_vectorized"}:
        slave_x_current = _driver_x_current(driver, closure)
        if native.surface_quadrature_cache is None:
            points, area_weights = _driver_surface_quadrature_points(driver, closure, native.quadrature_order)
        else:
            points = native.surface_quadrature_cache.points(slave_x_current)
            area_weights = native.surface_quadrature_cache.area_weights
        n_slave = slave_x_current.shape[0] * 3
        required_points = points
        samples = None
    else:
        raise ValueError(f"unsupported SFC contact backend: {native.sfc_backend}")
    sdf = DynamicNarrowBandSDF.build_required_points(
        x_master,
        case.top_faces,
        required_points,
        spacing=float(case.spacing_mm) * float(native.sdf_spacing_scale),
        band_radius=case.band_radius_mm,
        padding=case.band_radius_mm,
        cell_size=float(case.spacing_mm) * float(native.sdf_spacing_scale),
        batch_projection_threshold=int(native.batch_projection_threshold),
    )
    start = perf_counter()
    if native.sfc_backend == "node_to_surface":
        response = node_to_surface_field_penalty_response(
            slave_x_current,
            samples if samples is not None else [],
            sdf,
            pressure_stiffness=float(case.contact_pressure_stiffness_N_per_mm3),
            n_total_dofs=n_slave + case.body.n_dofs,
            sample_area_weights=area_weights,
            slave_dof_offset=0,
            master_dof_offset=n_slave,
        )
    elif native.sfc_backend == "surface_to_surface":
        response = surface_to_surface_field_penalty_response(
            slave_x_current,
            driver.contact_faces,
            sdf,
            pressure_stiffness=float(case.contact_pressure_stiffness_N_per_mm3),
            n_total_dofs=n_slave + case.body.n_dofs,
            slave_x_reference=driver.X,
            quadrature_order=native.quadrature_order,
            slave_dof_offset=0,
            master_dof_offset=n_slave,
        )
    else:
        response = surface_to_surface_field_penalty_response_vectorized(
            slave_x_current,
            driver.contact_faces,
            sdf,
            pressure_stiffness=float(case.contact_pressure_stiffness_N_per_mm3),
            n_total_dofs=n_slave + case.body.n_dofs,
            slave_x_reference=driver.X,
            quadrature_order=native.quadrature_order,
            quadrature_cache=native.surface_quadrature_cache,
            slave_dof_offset=0,
            master_dof_offset=n_slave,
        )
    query_seconds = perf_counter() - start
    return response.force[n_slave:].copy(), {
        "sfc_field_update_seconds": sdf.stats.update_seconds,
        "sfc_query_seconds": query_seconds,
        "sfc_contact_backend": native.sfc_backend,
        "sfc_quadrature_order": native.quadrature_order if native.sfc_backend.startswith("surface_to_surface") else 1,
        "sfc_sdf_spacing_scale": native.sdf_spacing_scale,
        "sfc_relaxation": native.sfc_relaxation,
        "sfc_max_iterations": native.sfc_max_iterations,
        "sfc_batch_projection_threshold": native.batch_projection_threshold,
        "sfc_projection_mode": sdf.grid.metadata.get("projection_mode", ""),
        "sfc_quadrature_points": int(response.quadrature_weights.size),
        "sfc_active_samples": response.active_count,
        "sfc_min_gap_mm": response.min_gap,
        "sfc_max_penetration_mm": response.max_penetration,
    }


def _solve_sfc_trajectory(native: NativeContactCase) -> list[Row]:
    case = native.case
    K = assemble_stiffness_matrix(case.body).tocsr()
    fixed = fixed_dofs_from_node_set(case.bottom_nodes, "xyz")
    u = np.zeros(case.body.n_dofs, dtype=float)
    rows: list[Row] = []
    for step_id, closure in enumerate(native.closures, start=1):
        step_start = perf_counter()
        max_correction = math.inf
        diagnostics: Row = {}
        field_update_total = 0.0
        query_total = 0.0
        linear_solve_total = 0.0
        contact_evaluations = 0
        for _iteration in range(int(native.sfc_max_iterations)):
            force, diagnostics = _sfc_contact_force(native, u, closure)
            contact_evaluations += 1
            field_update_total += float(diagnostics.get("sfc_field_update_seconds", 0.0))
            query_total += float(diagnostics.get("sfc_query_seconds", 0.0))
            solve_start = perf_counter()
            Kff, rhs, free = eliminate_fixed_dofs(K, force, fixed)
            reduced = np.asarray(spsolve(Kff, rhs), dtype=float) if free.size else np.empty(0, dtype=float)
            u_new = expand_reduced_vector(reduced, free, case.body.n_dofs, fixed)
            linear_solve_total += perf_counter() - solve_start
            max_correction = float(np.linalg.norm(u_new - u))
            relaxation = float(native.sfc_relaxation)
            u = relaxation * u_new + (1.0 - relaxation) * u
            if max_correction <= 1.0e-10 * max(1.0, float(np.linalg.norm(u_new))):
                u = u_new
                break
        step_wall = perf_counter() - step_start
        strain, stress, vm = _element_strain_stress(case, u)
        rows.append(
            {
                "problem": case.problem,
                "step": step_id,
                "closure_mm": float(closure),
                "sfc_displacement": u.copy(),
                "sfc_strain": strain,
                "sfc_stress": stress,
                "sfc_vm": vm,
                "sfc_displacement_l2_norm": float(np.linalg.norm(u)),
                "sfc_max_von_mises_MPa": float(np.max(vm)) if vm.size else 0.0,
                "sfc_max_strain_norm": float(np.max(np.linalg.norm(strain, axis=1))) if strain.size else 0.0,
                "sfc_max_correction_norm": max_correction,
                "sfc_step_wall_seconds": step_wall,
                "sfc_contact_evaluations": contact_evaluations,
                "sfc_field_update_total_seconds": field_update_total,
                "sfc_query_total_seconds": query_total,
                "sfc_linear_solve_total_seconds": linear_solve_total,
                "sfc_other_total_seconds": max(step_wall - field_update_total - query_total - linear_solve_total, 0.0),
                **diagnostics,
            }
        )
    return rows


def _element_strain_stress(case: PublishedCase, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = isotropic_linear_elasticity_matrix(float(case.body.material["E"]), float(case.body.material["nu"]))
    strains: list[np.ndarray] = []
    stresses: list[np.ndarray] = []
    vm: list[float] = []
    u_nodes = np.asarray(u, dtype=float).reshape((-1, 3))
    for element in case.body.mesh.elements:
        B = tet4_strain_displacement_matrix(case.body.mesh.X[element])
        strain = B @ u_nodes[element].reshape(12)
        stress = C @ strain
        strains.append(strain)
        stresses.append(stress)
        vm.append(_von_mises_voigt(stress))
    return np.vstack(strains), np.vstack(stresses), np.asarray(vm, dtype=float)


def _format_ids(ids: np.ndarray, *, per_line: int = 12) -> list[str]:
    values = [str(int(value)) for value in np.asarray(ids, dtype=np.int64).ravel()]
    return [", ".join(values[i : i + per_line]) for i in range(0, len(values), per_line)]


def _write_calculix_native_input(native: NativeContactCase, path: Path) -> None:
    case = native.case
    master = case.body.mesh
    driver = native.driver
    n_master = master.X.shape[0]
    X = np.vstack([master.X, driver.X])
    master_node_ids = np.arange(1, n_master + 1, dtype=np.int64)
    driver_node_ids = np.arange(n_master + 1, n_master + driver.X.shape[0] + 1, dtype=np.int64)
    lines = [
        "** Generated by validation/run_fuzzycontact_calculix_native_contact_trajectory.py",
        "** Native CalculiX surface-to-surface contact; no SFC equivalent-load shortcut.",
        "*HEADING",
        f"{case.problem} native contact trajectory",
        "*NODE, NSET=NALL",
    ]
    for node_id, xyz in zip(np.arange(1, X.shape[0] + 1), X, strict=True):
        lines.append(f"{int(node_id)}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
    lines.append("*ELEMENT, TYPE=C3D4, ELSET=EMASTER")
    for eid, element in enumerate(master.elements, start=1):
        conn = ", ".join(str(int(node) + 1) for node in element)
        lines.append(f"{eid}, {conn}")
    driver_e0 = master.elements.shape[0] + 1
    lines.append("*ELEMENT, TYPE=C3D4, ELSET=EDRIVER")
    for eid, element in enumerate(driver.elements, start=driver_e0):
        conn = ", ".join(str(n_master + int(node) + 1) for node in element)
        lines.append(f"{eid}, {conn}")
    lines.append("*NSET, NSET=NMASTER")
    lines.extend(_format_ids(master_node_ids))
    lines.append("*NSET, NSET=NDRIVER")
    lines.extend(_format_ids(driver_node_ids))
    lines.append("*NSET, NSET=NFIXED")
    lines.extend(_format_ids(case.bottom_nodes + 1))
    lines.extend(
        [
            "*MATERIAL, NAME=MAT",
            "*ELASTIC",
            f"{float(case.body.material['E']):.12e}, {float(case.body.material['nu']):.12e}",
            "*SOLID SECTION, ELSET=EMASTER, MATERIAL=MAT",
            "*SOLID SECTION, ELSET=EDRIVER, MATERIAL=MAT",
            "*SURFACE, NAME=SMASTER, TYPE=ELEMENT",
        ]
    )
    for eid, label in native.master_top_refs:
        lines.append(f"{int(eid)}, {label}")
    lines.append("*SURFACE, NAME=SSLAVE, TYPE=ELEMENT")
    for eid, label in native.driver.contact_refs:
        lines.append(f"{driver_e0 + int(eid) - 1}, {label}")
    lines.extend(
        [
            "*SURFACE INTERACTION, NAME=CONTACT",
            "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
            f"{native.calibrated_stiffness:.12e}",
            "*CONTACT PAIR, INTERACTION=CONTACT, TYPE=SURFACE TO SURFACE",
            "SSLAVE, SMASTER",
            "*BOUNDARY",
            "NFIXED, 1, 3, 0.0",
            "NDRIVER, 1, 2, 0.0",
        ]
    )
    for step_id, closure in enumerate(native.closures, start=1):
        lines.extend(
            [
                "*STEP, NLGEOM, INC=100",
                f"*STATIC",
                "0.25, 1.0, 1e-06, 0.25",
                "*BOUNDARY",
                f"NDRIVER, 3, 3, {-float(closure):.12e}",
                "*NODE PRINT, NSET=NMASTER, FREQUENCY=1",
                "U",
                "*EL PRINT, ELSET=EMASTER, FREQUENCY=1",
                "S",
                "E",
                "*CONTACT PRINT, FREQUENCY=1",
                "CDIS",
                "CSTR",
                "*CONTACT PRINT, TOTALS=ONLY, FREQUENCY=1",
                "CNUM",
                "*END STEP",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_calculix(job: str, run_dir: Path, *, timeout: int, skip_calculix: bool) -> Row:
    command = f"cd {_wsl_path(run_dir)} && ccx {job}"
    inp = run_dir / f"{job}.inp"
    dat = run_dir / f"{job}.dat"
    if skip_calculix or not calculix_available():
        return {
            "case": job,
            "external_solver": "CalculiX",
            "external_solver_version": "skipped" if skip_calculix else "not_available",
            "command": f"wsl --exec bash -lc \"{command}\"",
            "input_file": str(inp),
            "dat_file": "",
            "return_code": "skipped" if skip_calculix else "not_available",
            "completed": "false",
            "calculix_wall_seconds": "",
        }
    start = perf_counter()
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=int(timeout),
    )
    wall_seconds = perf_counter() - start
    (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
    return {
        "case": job,
        "external_solver": "CalculiX",
        "external_solver_version": _calculix_version(),
        "command": f"wsl --exec bash -lc \"{command}\"",
        "input_file": str(inp),
        "dat_file": str(dat) if dat.exists() else "",
        "return_code": proc.returncode,
        "completed": str(proc.returncode == 0 and dat.exists()).lower(),
        "calculix_wall_seconds": wall_seconds,
    }


def _parse_element_table(path: Path, quantity: str, n_elements: int) -> dict[float, np.ndarray]:
    number = r"([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    header = re.compile(fr"{quantity} .* time\s+" + number, re.IGNORECASE)
    row = re.compile(
        r"^\s*(\d+)\s+\d+\s+"
        + number
        + r"\s+"
        + number
        + r"\s+"
        + number
        + r"\s+"
        + number
        + r"\s+"
        + number
        + r"\s+"
        + number
    )
    out: dict[float, np.ndarray] = {}
    current_time: float | None = None
    current = np.zeros((n_elements, 6), dtype=float)
    seen = 0

    def flush() -> None:
        nonlocal current_time, current, seen
        if current_time is not None and seen:
            out[current_time] = current.copy()
        current_time = None
        current = np.zeros((n_elements, 6), dtype=float)
        seen = 0

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = header.search(line)
        if match:
            flush()
            current_time = float(match.group(1))
            current = np.zeros((n_elements, 6), dtype=float)
            seen = 0
            continue
        if current_time is None:
            continue
        m = row.match(line)
        if m is None:
            if seen and line.strip() == "":
                flush()
            continue
        eid = int(m.group(1))
        if 1 <= eid <= n_elements:
            xx, yy, zz, xy, xz, yz = [float(m.group(i)) for i in range(2, 8)]
            if quantity.lower().startswith("strain"):
                xy *= 2.0
                xz *= 2.0
                yz *= 2.0
            current[eid - 1] = [xx, yy, zz, xy, yz, xz]
            seen += 1
            continue
    flush()
    return dict(sorted(out.items()))


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    if denom <= 1.0e-30:
        return float(np.linalg.norm(value - reference))
    return float(np.linalg.norm(value - reference) / denom)


def _comparison_rows(native: NativeContactCase, dat: Path, sfc_rows: list[Row]) -> list[Row]:
    case = native.case
    node_ids = np.arange(1, case.body.mesh.X.shape[0] + 1, dtype=np.int64)
    disp_blocks = _parse_calculix_dat_displacements(dat, node_ids)
    stress_blocks = _parse_element_table(dat, "stresses", case.body.mesh.elements.shape[0])
    strain_blocks = _parse_element_table(dat, "strains", case.body.mesh.elements.shape[0])
    rows: list[Row] = []
    for row in sfc_rows:
        target_time = float(row["step"])
        available = sorted(disp_blocks)
        time = min(available, key=lambda value: abs(float(value) - target_time))
        u_calc = disp_blocks[time].reshape((-1,))
        stress_calc = stress_blocks.get(time, np.zeros_like(row["sfc_stress"]))
        strain_calc = strain_blocks.get(time, np.zeros_like(row["sfc_strain"]))
        vm_calc = np.asarray([_von_mises_voigt(v) for v in stress_calc], dtype=float)
        disp_err = _relative_error(row["sfc_displacement"], u_calc)
        strain_err = _relative_error(row["sfc_strain"], strain_calc)
        stress_err = _relative_error(row["sfc_stress"], stress_calc)
        vm_err = _relative_error(row["sfc_vm"], vm_calc)
        status = (
            "passed_native_solver_agreement"
            if max(disp_err, strain_err, stress_err, vm_err) < 5.0e-2
            else "failed_native_solver_agreement"
        )
        rows.append(
            {
                "problem": case.problem,
                "step": int(row["step"]),
                "closure_mm": float(row["closure_mm"]),
                "calculix_time": time,
                "displacement_l2_rel_error": disp_err,
                "strain_l2_rel_error": strain_err,
                "stress_l2_rel_error": stress_err,
                "von_mises_l2_rel_error": vm_err,
                "sfc_min_gap_mm": row["sfc_min_gap_mm"],
                "sfc_active_samples": row["sfc_active_samples"],
                "sfc_contact_backend": row["sfc_contact_backend"],
                "sfc_quadrature_order": row["sfc_quadrature_order"],
                "sfc_sdf_spacing_scale": row["sfc_sdf_spacing_scale"],
                "sfc_relaxation": row["sfc_relaxation"],
                "sfc_max_iterations": row["sfc_max_iterations"],
                "sfc_batch_projection_threshold": row["sfc_batch_projection_threshold"],
                "sfc_projection_mode": row["sfc_projection_mode"],
                "sfc_quadrature_points": row["sfc_quadrature_points"],
                "sfc_max_von_mises_MPa": row["sfc_max_von_mises_MPa"],
                "calculix_max_von_mises_MPa": float(np.max(vm_calc)) if vm_calc.size else 0.0,
                "sfc_max_strain_norm": row["sfc_max_strain_norm"],
                "calculix_max_strain_norm": float(np.max(np.linalg.norm(strain_calc, axis=1))) if strain_calc.size else 0.0,
                "status": status,
                "details": "CalculiX native *CONTACT PAIR versus independent SFC DynamicNarrowBandSDF+field_contact on the same closure history.",
            }
        )
    return rows


def _json_safe_rows(rows: list[Row]) -> list[Row]:
    safe: list[Row] = []
    for row in rows:
        safe.append({key: value for key, value in row.items() if not isinstance(value, np.ndarray)})
    return safe


def _write_summary(
    path: Path,
    comparison_rows: list[Row],
    command_rows: list[Row],
    timing_rows: list[Row],
    outputs: dict[str, Path],
) -> None:
    lines = [
        "# Independent CalculiX/SFC Native Contact Trajectory",
        "",
        "This is the strict independent-solver trajectory runner: CalculiX builds and solves a native `*CONTACT PAIR` model; SFC independently builds and solves the corresponding `DynamicNarrowBandSDF + field_contact` model.",
        "",
        "The cases use open benchmark-style contact modeling schemes only as geometry/loading inspiration. The comparison is not a claim of matching FuzzyContact published results.",
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
            "## Native Solver Agreement Gate",
            "",
            "| Problem | Step | disp rel | strain rel | stress rel | VM rel | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in comparison_rows:
        disp = _fmt(row.get("displacement_l2_rel_error"))
        strain = _fmt(row.get("strain_l2_rel_error"))
        stress = _fmt(row.get("stress_l2_rel_error"))
        vm = _fmt(row.get("von_mises_l2_rel_error"))
        lines.append(
            f"| `{row['problem']}` | {int(row['step'])} | "
            f"{disp} | {strain} | {stress} | {vm} | {row['status']} |"
        )
    passed = sum(1 for row in comparison_rows if row["status"] == "passed_native_solver_agreement")
    lines.extend(
        [
            "",
            f"- Passed rows: `{passed}/{len(comparison_rows)}`.",
            f"- Completed CalculiX jobs: `{sum(1 for row in command_rows if row.get('completed') == 'true')}/{len(command_rows)}`.",
            "",
            "## Timing",
            "",
            "| Problem | SFC total s | CalculiX wall s | CalculiX/SFC | SFC field update s | SFC query s | SFC linear solve s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in timing_rows:
        lines.append(
            f"| `{row['problem']}` | {_fmt(row.get('sfc_total_wall_seconds'))} | "
            f"{_fmt(row.get('calculix_wall_seconds'))} | {_fmt(row.get('calculix_to_sfc_wall_ratio'))} | "
            f"{_fmt(row.get('sfc_field_update_total_seconds'))} | {_fmt(row.get('sfc_query_total_seconds'))} | "
            f"{_fmt(row.get('sfc_linear_solve_total_seconds'))} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Supports independent CalculiX-native-contact versus SFC-SDF-contact agreement only for rows marked `passed_native_solver_agreement`.",
            "- If rows fail, the output is a negative result identifying contact-law/active-set/discretization mismatch, not an SDF field construction failure by itself.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.6e}"
    except (TypeError, ValueError):
        return str(value)


def _plot_native_contact_outputs(
    out_dir: Path,
    comparison_rows: list[Row],
    timing_rows: list[Row],
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    _configure_times_fonts(matplotlib)
    import matplotlib.pyplot as plt

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    numeric_rows = [
        row
        for row in comparison_rows
        if row.get("displacement_l2_rel_error") not in {"", None}
        and row.get("strain_l2_rel_error") not in {"", None}
        and row.get("stress_l2_rel_error") not in {"", None}
        and row.get("von_mises_l2_rel_error") not in {"", None}
    ]
    if numeric_rows:
        labels = [f"{row['problem']}-s{int(row['step'])}" for row in numeric_rows]
        x = np.arange(len(labels), dtype=float)
        width = 0.20
        metrics = [
            ("displacement_l2_rel_error", "disp."),
            ("strain_l2_rel_error", "strain"),
            ("stress_l2_rel_error", "stress"),
            ("von_mises_l2_rel_error", "VM"),
        ]
        fig, ax = plt.subplots(figsize=(max(7.0, 0.46 * len(labels)), 3.8))
        for offset, (key, label) in zip((-1.5, -0.5, 0.5, 1.5), metrics, strict=True):
            ax.bar(x + offset * width, [float(row[key]) for row in numeric_rows], width, label=label)
        ax.axhline(5.0e-2, color="black", linestyle="--", linewidth=1.0, label="5% gate")
        ax.set_yscale("log")
        ax.set_ylabel("Relative error")
        ax.set_xlabel("Native contact step")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(True, axis="y", which="both", alpha=0.30)
        ax.legend(ncol=5, fontsize=8)
        fig.tight_layout()
        png = figures / "native_contact_solver_errors.png"
        pdf = figures / "native_contact_solver_errors.pdf"
        fig.savefig(png, dpi=180)
        fig.savefig(pdf)
        plt.close(fig)
        outputs["native_contact_solver_errors_png"] = png
        outputs["native_contact_solver_errors_pdf"] = pdf

    numeric_timing = [
        row
        for row in timing_rows
        if row.get("sfc_total_wall_seconds") not in {"", None}
        and row.get("calculix_wall_seconds") not in {"", None}
    ]
    if numeric_timing:
        labels = [str(row["problem"]) for row in numeric_timing]
        x = np.arange(len(labels), dtype=float)
        field = np.asarray([float(row["sfc_field_update_total_seconds"]) for row in numeric_timing], dtype=float)
        query = np.asarray([float(row["sfc_query_total_seconds"]) for row in numeric_timing], dtype=float)
        solve = np.asarray([float(row["sfc_linear_solve_total_seconds"]) for row in numeric_timing], dtype=float)
        other = np.asarray([float(row["sfc_other_total_seconds"]) for row in numeric_timing], dtype=float)
        ccx = np.asarray([float(row["calculix_wall_seconds"]) for row in numeric_timing], dtype=float)
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        bottom = np.zeros_like(x)
        for values, label, color in [
            (field, "SFC field update", "#4c78a8"),
            (query, "SFC field query/contact", "#72b7b2"),
            (solve, "SFC linear solve", "#f58518"),
            (other, "SFC other", "#b279a2"),
        ]:
            ax.bar(x, values, bottom=bottom, label=label, color=color)
            bottom += values
        ax.scatter(x, ccx, marker="D", color="black", zorder=5, label="CalculiX wall")
        ax.set_yscale("log")
        ax.set_ylabel("Wall time (s)")
        ax.set_xlabel("Problem")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", which="both", alpha=0.30)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = figures / "native_contact_solver_timing_breakdown.png"
        pdf = figures / "native_contact_solver_timing_breakdown.pdf"
        fig.savefig(png, dpi=180)
        fig.savefig(pdf)
        plt.close(fig)
        outputs["native_contact_solver_timing_png"] = png
        outputs["native_contact_solver_timing_pdf"] = pdf

    return outputs


def run_validation(
    out_dir: Path,
    *,
    quick: bool = False,
    timeout: int = 360,
    skip_calculix: bool = False,
    sfc_backend: str = "surface_to_surface_vectorized",
    quadrature_order: int = 7,
    sdf_spacing_scale: float = 1.0,
    sfc_relaxation: float = 0.55,
    sfc_max_iterations: int = 15,
    batch_projection_threshold: int = 5_000_000,
    case_names: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    command_rows: list[Row] = []
    comparison_rows: list[Row] = []
    sfc_rows_all: list[Row] = []
    timing_rows: list[Row] = []
    selected_cases = _make_cases(quick=quick)
    if case_names:
        wanted = set(case_names)
        selected_cases = [case for case in selected_cases if case.problem in wanted]
        missing = wanted.difference({case.problem for case in selected_cases})
        if missing:
            raise ValueError(f"unknown or unavailable native-contact case(s): {sorted(missing)}")
    for case in selected_cases:
        driver = _driver_for_case(case, quick=quick)
        native = _calibrated_case(
            case,
            driver,
            sfc_backend=sfc_backend,
            quadrature_order=quadrature_order,
            sdf_spacing_scale=sdf_spacing_scale,
            sfc_relaxation=sfc_relaxation,
            sfc_max_iterations=sfc_max_iterations,
            batch_projection_threshold=batch_projection_threshold,
        )
        if quick:
            native = replace(native, closures=(native.closures[1], native.closures[-1]))
        sfc_rows = _solve_sfc_trajectory(native)
        sfc_rows_all.extend(_json_safe_rows(sfc_rows))
        run_dir = out_dir / "calculix_runs" / case.problem
        job = f"native_contact_{case.problem}"
        _write_calculix_native_input(native, run_dir / f"{job}.inp")
        command = _run_calculix(job, run_dir, timeout=timeout, skip_calculix=skip_calculix)
        command_rows.append(command)
        sfc_total = float(sum(float(row.get("sfc_step_wall_seconds", 0.0)) for row in sfc_rows))
        calculix_wall = command.get("calculix_wall_seconds", "")
        calculix_wall_value = float(calculix_wall) if calculix_wall not in {"", None} else math.nan
        timing_rows.append(
            {
                "problem": case.problem,
                "sfc_contact_backend": native.sfc_backend,
                "sfc_quadrature_order": native.quadrature_order if native.sfc_backend.startswith("surface_to_surface") else 1,
                "sfc_sdf_spacing_scale": native.sdf_spacing_scale,
                "sfc_relaxation": native.sfc_relaxation,
                "sfc_max_iterations": native.sfc_max_iterations,
                "sfc_batch_projection_threshold": native.batch_projection_threshold,
                "sfc_steps": len(sfc_rows),
                "sfc_total_wall_seconds": sfc_total,
                "sfc_field_update_total_seconds": float(
                    sum(float(row.get("sfc_field_update_total_seconds", 0.0)) for row in sfc_rows)
                ),
                "sfc_query_total_seconds": float(sum(float(row.get("sfc_query_total_seconds", 0.0)) for row in sfc_rows)),
                "sfc_linear_solve_total_seconds": float(
                    sum(float(row.get("sfc_linear_solve_total_seconds", 0.0)) for row in sfc_rows)
                ),
                "sfc_other_total_seconds": float(sum(float(row.get("sfc_other_total_seconds", 0.0)) for row in sfc_rows)),
                "sfc_contact_evaluations": int(sum(int(row.get("sfc_contact_evaluations", 0)) for row in sfc_rows)),
                "calculix_wall_seconds": "" if math.isnan(calculix_wall_value) else calculix_wall_value,
                "calculix_to_sfc_wall_ratio": ""
                if math.isnan(calculix_wall_value) or sfc_total <= 0.0
                else calculix_wall_value / sfc_total,
                "sfc_to_calculix_wall_ratio": ""
                if math.isnan(calculix_wall_value) or calculix_wall_value <= 0.0
                else sfc_total / calculix_wall_value,
            }
        )
        dat = command.get("dat_file", "")
        if command.get("completed") == "true" and dat:
            comparison_rows.extend(_comparison_rows(native, Path(str(dat)), sfc_rows))
        else:
            for row in sfc_rows:
                comparison_rows.append(
                    {
                        "problem": case.problem,
                        "step": int(row["step"]),
                        "closure_mm": float(row["closure_mm"]),
                        "calculix_time": "",
                        "displacement_l2_rel_error": "",
                        "strain_l2_rel_error": "",
                        "stress_l2_rel_error": "",
                        "von_mises_l2_rel_error": "",
                        "sfc_min_gap_mm": row["sfc_min_gap_mm"],
                        "sfc_active_samples": row["sfc_active_samples"],
                        "sfc_contact_backend": row["sfc_contact_backend"],
                        "sfc_quadrature_order": row["sfc_quadrature_order"],
                        "sfc_sdf_spacing_scale": row["sfc_sdf_spacing_scale"],
                        "sfc_relaxation": row["sfc_relaxation"],
                        "sfc_max_iterations": row["sfc_max_iterations"],
                        "sfc_batch_projection_threshold": row["sfc_batch_projection_threshold"],
                        "sfc_projection_mode": row["sfc_projection_mode"],
                        "sfc_quadrature_points": row["sfc_quadrature_points"],
                        "sfc_max_von_mises_MPa": row["sfc_max_von_mises_MPa"],
                        "calculix_max_von_mises_MPa": "",
                        "sfc_max_strain_norm": row["sfc_max_strain_norm"],
                        "calculix_max_strain_norm": "",
                        "status": "not_run" if skip_calculix else "calculix_not_completed",
                        "details": "CalculiX did not complete or was skipped.",
                    }
                )
    outputs = {
        "comparison": out_dir / "native_contact_solver_comparison.csv",
        "sfc_history": out_dir / "native_contact_solver_sfc_history.csv",
        "commands": out_dir / "native_contact_solver_commands.csv",
        "timing": out_dir / "native_contact_solver_timing.csv",
        "summary": out_dir / "native_contact_solver_summary.md",
    }
    _write_csv(outputs["comparison"], comparison_rows)
    _write_csv(outputs["sfc_history"], sfc_rows_all)
    _write_csv(outputs["commands"], command_rows)
    _write_csv(outputs["timing"], timing_rows)
    figure_outputs = _plot_native_contact_outputs(out_dir, comparison_rows, timing_rows)
    outputs.update(figure_outputs)
    _write_summary(outputs["summary"], comparison_rows, command_rows, timing_rows, outputs)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-calculix", action="store_true")
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "fuzzycontact_calculix_native_contact")
    parser.add_argument(
        "--sfc-backend",
        choices=("surface_to_surface_vectorized", "surface_to_surface", "node_to_surface"),
        default="surface_to_surface_vectorized",
        help="SFC contact integration backend. node_to_surface preserves the previous centroid/sample path.",
    )
    parser.add_argument(
        "--quadrature-order",
        type=int,
        choices=(1, 3, 7),
        default=7,
        help="Triangle surface quadrature order for the surface_to_surface backend.",
    )
    parser.add_argument(
        "--sdf-spacing-scale",
        type=float,
        default=1.0,
        help="Scale applied to the benchmark SDF grid spacing; values below 1 refine the dynamic field.",
    )
    parser.add_argument(
        "--sfc-relaxation",
        type=float,
        default=0.55,
        help="Fixed-point relaxation for the SFC contact solve. 1.0 applies the full updated displacement.",
    )
    parser.add_argument(
        "--sfc-max-iterations",
        type=int,
        default=15,
        help="Maximum SFC contact re-evaluations per load step.",
    )
    parser.add_argument(
        "--batch-projection-threshold",
        type=int,
        default=5_000_000,
        help="Use all-face batch projection below this pair count; otherwise use spatial-hash candidate grouped batches.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=("problem_1", "problem_3", "problem_4"),
        help="Limit the native-contact run to one benchmark-style problem. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation(
        args.out_dir,
        quick=bool(args.quick),
        timeout=int(args.timeout),
        skip_calculix=bool(args.skip_calculix),
        sfc_backend=str(args.sfc_backend),
        quadrature_order=int(args.quadrature_order),
        sdf_spacing_scale=float(args.sdf_spacing_scale),
        sfc_relaxation=float(args.sfc_relaxation),
        sfc_max_iterations=int(args.sfc_max_iterations),
        batch_projection_threshold=int(args.batch_projection_threshold),
        case_names=tuple(args.case) if args.case else None,
    )
    print("Independent CalculiX/SFC native contact trajectory complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
