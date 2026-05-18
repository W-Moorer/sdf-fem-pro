"""Phase-9 full contact-validation evidence matrix.

The runner is intentionally claim-gated.  It gathers the current C3D8 static
and dynamic contact evidence, writes per-case CSV/Markdown/plot/VTK artifacts,
and explicitly blocks paper claims that do not yet have native SFC, CalculiX,
and timing evidence.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import replace
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

from sfc.fem.assembler import assemble_gravity_force, assemble_mass_matrix, assemble_stiffness_matrix  # noqa: E402
from sfc.fem.body import DeformableBody  # noqa: E402
from sfc.fem.calculix_aligned import (  # noqa: E402
    ContactSample,
    MechanicsModel,
    MechanicsState,
    PlaneContactGeometry,
    assemble_contact_response,
    calculix_dynamic_predictor,
    calculix_hht_effective_residual,
    calculix_hht_effective_tangent,
    evaluate_state,
    hht_step,
    hht_newmark_parameters,
    initial_state,
    static_force_state,
    stvk_internal_response,
)
from sfc.fem.hex8 import hex8_center_strain_stress  # noqa: E402
from sfc.mesh import VolumeMesh  # noqa: E402
from sfc.contact.field_contact import field_contact_constraint_from_sample  # noqa: E402
from sfc.contact.narrow_phase import SurfaceSample  # noqa: E402
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402
from validation.run_c3d8_contact_trajectory_validation import (  # noqa: E402
    C3D8_FACE_NODES,
    FLOOR_CONTACT_OFFSET,
    QUAD_GAUSS,
    _calculix_version,
    _center_z,
    _master_query_geometry,
    _parse_contact_elements,
    _parse_element_stress,
    _parse_nodal_vectors,
    _parse_totals,
    _quad_shape,
    _replay_dynamic_sdf,
    _surface_quads as _trajectory_surface_quads,
    _triangles_from_quads as _trajectory_triangles_from_quads,
    _wsl_path,
    build_c3d8_model,
    calculix_available,
    main as run_c3d8_trajectory_main,
    run_calculix,
    write_calculix_input,
)
from validation.run_calculix_contactenergy_replay import (  # noqa: E402
    _hex8_center_strain_stress as contactenergy_hex8_center_strain_stress,
)
from validation.run_calculix_contactenergy_replay import (  # noqa: E402
    _external_force_vector,
    _fixed_dofs_and_values,
    _orient_quad_outward,
    _surface_quads,
    _triangles_from_quads,
    _relative_vector_error,
    parse_contactenergy_dat,
    parse_contactenergy_input,
    solve_sfc_c3d8_contactenergy_static,
)
from validation.vtk_frame_series import write_hex_frame_series  # noqa: E402

Row = dict[str, Any]
_PHASE9_FIELD_CACHE: dict[tuple[tuple[int, ...], tuple[int, int], bytes], DynamicNarrowBandSDF] = {}


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bool_text(value: bool) -> str:
    return str(bool(value)).lower()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _case_status(native: bool, external: bool, metric_ok: bool) -> str:
    if native and external and metric_ok:
        return "supported"
    if external and metric_ok:
        return "external_replay_only"
    return "blocked"


def _write_legacy_hex_vtk(
    path: Path,
    points: np.ndarray,
    elements: np.ndarray,
    *,
    cell_values: dict[str, np.ndarray],
) -> None:
    """Write a minimal legacy VTK HEX8 file for ParaView inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(points, dtype=float)
    conn = np.asarray(elements, dtype=np.int64)
    lines: list[str] = [
        "# vtk DataFile Version 3.0",
        "SFC Phase-9 C3D8 contact validation",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {X.shape[0]} float",
    ]
    lines.extend(f"{x:.12e} {y:.12e} {z:.12e}" for x, y, z in X)
    lines.append(f"CELLS {conn.shape[0]} {conn.shape[0] * 9}")
    lines.extend("8 " + " ".join(str(int(node)) for node in element) for element in conn)
    lines.append(f"CELL_TYPES {conn.shape[0]}")
    lines.extend("12" for _ in range(conn.shape[0]))
    if cell_values:
        lines.append(f"CELL_DATA {conn.shape[0]}")
        for name, values in cell_values.items():
            data = np.asarray(values, dtype=float).ravel()
            if data.shape != (conn.shape[0],):
                raise ValueError(f"cell value {name!r} has wrong shape")
            lines.append(f"SCALARS {name} float 1")
            lines.append("LOOKUP_TABLE default")
            lines.extend(f"{float(value):.12e}" for value in data)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _von_mises(stress: np.ndarray) -> float:
    sxx, syy, szz, sxy, syz, sxz = np.asarray(stress, dtype=float)
    return float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + syz**2 + sxz**2)))


def _phase9_hex_cell_fields(model: Any, displacement: np.ndarray, stress_map: dict[int, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    U = np.asarray(displacement, dtype=float)
    stress_map = stress_map or {}
    disp_mag = np.asarray([float(np.mean(np.linalg.norm(U[element], axis=1))) for element in model.elements], dtype=float)
    strain_norm: list[float] = []
    vm: list[float] = []
    for element_id, element in zip(model.element_ids, model.elements, strict=True):
        strain, sfc_stress = hex8_center_strain_stress(model.X[element], U[element], model.E, model.nu)
        stress = stress_map.get(int(element_id), sfc_stress)
        strain_norm.append(float(np.linalg.norm(strain)))
        vm.append(_von_mises(stress))
    return {
        "displacement_magnitude": disp_mag,
        "engineering_strain_norm": np.asarray(strain_norm, dtype=float),
        "von_mises": np.asarray(vm, dtype=float),
    }


class Hex8SurfaceSdfContactGeometry:
    """C3D8 slave quadrature against a true dynamic narrow-band SDF field."""

    def __init__(
        self,
        *,
        reference_X: np.ndarray,
        slave_quads: list[tuple[np.ndarray, np.ndarray]],
        master_quads: list[tuple[np.ndarray, np.ndarray]],
        stiffness: float,
    ) -> None:
        self.reference_X = np.asarray(reference_X, dtype=float)
        self.slave_quads = slave_quads
        self.master_quads = master_quads
        self.stiffness = float(stiffness)

    def samples(self, x_current: np.ndarray):
        x = np.asarray(x_current, dtype=float)
        master_faces = _triangles_from_quads(x, self.master_quads)
        if master_faces.size == 0:
            return
        master_sdf = _build_phase9_master_field(x, master_faces)
        candidates = np.arange(master_faces.shape[0], dtype=np.int64)
        for quad, element in self.slave_quads:
            face = _orient_quad_outward(x, quad, element)
            coords = x[face]
            for xi in QUAD_GAUSS:
                for eta in QUAD_GAUSS:
                    shape, dxi, deta = _quad_shape(xi, eta)
                    xq = shape @ coords
                    area = float(np.linalg.norm(np.cross(dxi @ coords, deta @ coords)))
                    if area <= 0.0:
                        continue
                    sample = SurfaceSample(
                        node_ids=np.asarray(face, dtype=np.int64),
                        weights=np.asarray(shape, dtype=float),
                        candidate_face_ids=candidates,
                    )
                    constraint = field_contact_constraint_from_sample(x, sample, master_sdf)
                    master_weights = _scalar_master_weights_from_field_constraint(constraint)
                    yield ContactSample(
                        node_ids=np.asarray(face, dtype=np.int64),
                        shape_weights=np.asarray(shape, dtype=float),
                        gap=float(constraint.g),
                        normal=np.asarray(constraint.normal, dtype=float),
                        area=area,
                        stiffness=self.stiffness,
                        master_node_ids=np.asarray(constraint.master_node_ids, dtype=np.int64),
                        master_shape_weights=master_weights,
                    )


def _phase9_field_cache_key(nodes: np.ndarray, faces: np.ndarray) -> tuple[tuple[int, ...], tuple[int, int], bytes]:
    face_ids = np.asarray(faces, dtype=np.int64)
    used = np.unique(face_ids.ravel())
    coords = np.round(np.asarray(nodes, dtype=float)[used], decimals=12)
    return tuple(int(v) for v in face_ids.ravel()), tuple(coords.shape), coords.tobytes()


def _phase9_sdf_grid_parameters(nodes: np.ndarray, faces: np.ndarray) -> tuple[float, float]:
    used = np.unique(np.asarray(faces, dtype=np.int64).ravel())
    X = np.asarray(nodes, dtype=float)[used]
    extent = np.ptp(X, axis=0)
    diag = max(float(np.linalg.norm(extent)), 1.0e-6)
    spacing = max(diag / 24.0, 0.025)
    band_radius = max(6.0 * spacing, 0.30 * diag, 0.10)
    return spacing, band_radius


def _build_phase9_master_field(nodes: np.ndarray, faces: np.ndarray) -> DynamicNarrowBandSDF:
    key = _phase9_field_cache_key(nodes, faces)
    cached = _PHASE9_FIELD_CACHE.get(key)
    if cached is not None:
        return cached
    spacing, band_radius = _phase9_sdf_grid_parameters(nodes, faces)
    field = DynamicNarrowBandSDF.build(
        nodes,
        faces,
        spacing=spacing,
        band_radius=band_radius,
        padding=band_radius,
        cell_size=max(2.0 * spacing, band_radius),
        gradient_mode="finite_difference",
    )
    _PHASE9_FIELD_CACHE[key] = field
    return field


def _scalar_master_weights_from_field_constraint(constraint) -> np.ndarray:
    """Project vector field payload sensitivities onto the query normal.

    The legacy Phase-9 mechanics assembly accepts scalar master weights in
    ``-N_master n`` form. The true field constraint stores the more general
    vector sensitivity. For this adapter, use the normal projection so flat
    contact cases remain exactly equivalent while the query path still comes
    from the dynamic SDF field.
    """

    normal = np.asarray(constraint.normal, dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-30)
    sensitivity = np.asarray(constraint.master_sensitivity, dtype=float)
    if sensitivity.size == 0:
        return np.empty(0, dtype=float)
    return -(sensitivity @ normal)


def _relative_error(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1.0e-30)


def _fixed_free(ndofs: int, fixed: np.ndarray) -> np.ndarray:
    return np.setdiff1d(np.arange(ndofs, dtype=np.int64), np.asarray(fixed, dtype=np.int64))


def _timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def _model_fixed_dofs(model) -> np.ndarray:
    id_to_index = model.id_to_index
    fixed_nodes = [id_to_index[int(node_id)] for node_id in model.fixed_node_ids if int(node_id) in id_to_index]
    if not fixed_nodes:
        return np.empty(0, dtype=np.int64)
    return np.asarray([3 * node + comp for node in fixed_nodes for comp in range(3)], dtype=np.int64)


def _contact_geometry_for_model(model):
    if model.master_kind == "rigid_plane":
        return PlaneContactGeometry(
            faces=_bottom_surface_triangles(model),
            plane_z=model.floor_z + FLOOR_CONTACT_OFFSET,
            stiffness=model.contact_stiffness,
        )
    return Hex8SurfaceSdfContactGeometry(
        reference_X=model.X,
        slave_quads=_trajectory_surface_quads(model, model.slave_surface),
        master_quads=_trajectory_surface_quads(model, model.master_surface),
        stiffness=model.contact_stiffness,
    )


def _solve_native_c3d8_nonlinear_static_contactenergy(out_dir: Path) -> tuple[Row, list[Row]]:
    """Run a native C3D8 StVK static solve on the contactenergy model."""

    source_dir = ROOT / "paper" / "numerical_experiments" / "calculix_contactenergy_c3d8_replay"
    inp_path = source_dir / "data" / "contactenergy.inp"
    dat_path = source_dir / "data" / "contactenergy.dat"
    if not inp_path.exists() or not dat_path.exists():
        return _unsupported_case("c3d8_nonlinear_static_contact", "static", "geometric_nonlinear"), []

    model = parse_contactenergy_input(inp_path.read_text(encoding="utf-8"))
    U_calculix, RF_calculix, cels_calculix, _raw = parse_contactenergy_dat(
        dat_path.read_text(encoding="utf-8", errors="ignore"),
        model.node_ids,
    )
    mechanics = MechanicsModel.from_hex8_mesh(
        model.X,
        model.elements,
        E=model.material_E,
        nu=model.material_nu,
        density=1.0,
    )
    contact = Hex8SurfaceSdfContactGeometry(
        reference_X=model.X,
        slave_quads=_surface_quads(model, "Sslav"),
        master_quads=_surface_quads(model, "Smast"),
        stiffness=model.pressure_stiffness,
    )
    fext = _external_force_vector(model)
    fixed, fixed_values = _fixed_dofs_and_values(model)
    free = _fixed_free(mechanics.n_dofs, fixed)
    U_seed, _seed_metrics = solve_sfc_c3d8_contactenergy_static(model)
    u = U_seed.reshape(-1).copy()
    if fixed.size:
        u[fixed] = fixed_values
    converged = False
    residual_norm = np.inf
    correction_norm = np.inf
    iterations = 0
    for iteration in range(30):
        x = mechanics.X + u.reshape((-1, 3))
        internal = stvk_internal_response(mechanics, x, assemble_tangent=True)
        contact_response = assemble_contact_response(contact.samples(x), mechanics.n_nodes)
        residual = internal.force.reshape(-1) - contact_response.force.reshape(-1) - fext
        tangent = (internal.tangent + contact_response.tangent).tocsc()
        residual_norm = float(np.linalg.norm(residual[free]))
        if residual_norm <= 1.0e-8:
            converged = True
            iterations = iteration
            break
        correction = np.asarray(spsolve(tangent[free[:, None], free], -residual[free]), dtype=float)
        correction_norm = float(np.linalg.norm(correction))
        best_u = u.copy()
        best_norm = residual_norm
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial = u.copy()
            trial[free] += scale * correction
            if fixed.size:
                trial[fixed] = fixed_values
            trial_x = mechanics.X + trial.reshape((-1, 3))
            try:
                trial_internal = stvk_internal_response(mechanics, trial_x, assemble_tangent=False)
                trial_contact = assemble_contact_response(contact.samples(trial_x), mechanics.n_nodes)
            except (ValueError, FloatingPointError):
                continue
            trial_residual = trial_internal.force.reshape(-1) - trial_contact.force.reshape(-1) - fext
            trial_norm = float(np.linalg.norm(trial_residual[free]))
            if np.isfinite(trial_norm) and trial_norm < best_norm:
                best_norm = trial_norm
                best_u = trial
                break
        u = best_u
        iterations = iteration + 1
        if best_norm <= 1.0e-8:
            residual_norm = best_norm
            converged = True
            break

    U_sfc = u.reshape((-1, 3))
    x_final = mechanics.X + U_sfc
    internal_final = stvk_internal_response(mechanics, x_final, assemble_tangent=True)
    contact_final = assemble_contact_response(contact.samples(x_final), mechanics.n_nodes)
    strain_sfc, stress_sfc, vm_sfc, strain_norm_sfc = contactenergy_hex8_center_strain_stress(model, U_sfc)
    strain_calc, stress_calc, vm_calc, strain_norm_calc = contactenergy_hex8_center_strain_stress(model, U_calculix)
    rf_calc_norm = float(np.linalg.norm(RF_calculix))
    force_sfc = float(np.linalg.norm(contact_final.force))
    cels_rel = _relative_error(float(contact_final.energy), float(cels_calculix))
    rf_rel = _relative_error(force_sfc, rf_calc_norm)
    displacement_rel = _relative_vector_error(U_sfc.ravel(), U_calculix.ravel())
    stress_rel = _relative_vector_error(stress_sfc.ravel(), stress_calc.ravel())
    strain_rel = _relative_vector_error(strain_sfc.ravel(), strain_calc.ravel())
    metric_ok = bool(converged and displacement_rel < 0.75 and stress_rel < 0.75 and rf_rel < 0.75 and cels_rel < 0.75)
    _write_legacy_hex_vtk(
        out_dir / "vtk" / "c3d8_nonlinear_static_contact_0000.vtk",
        x_final,
        model.elements,
        cell_values={"von_mises": vm_sfc, "green_lagrange_strain_norm": strain_norm_sfc},
    )
    evidence_path = out_dir / "native_c3d8_nonlinear_static_contactenergy.csv"
    _write_csv(
        evidence_path,
        [
            "case_id",
            "converged",
            "iterations",
            "residual_norm",
            "correction_norm",
            "displacement_l2_rel_error",
            "strain_l2_rel_error",
            "stress_l2_rel_error",
            "von_mises_l2_rel_error",
            "contact_force_rel_error",
            "contact_energy_rel_error",
            "active_contact_count",
            "min_gap",
            "max_penetration",
        ],
        [
            {
                "case_id": "c3d8_nonlinear_static_contact",
                "converged": _bool_text(converged),
                "iterations": iterations,
                "residual_norm": residual_norm,
                "correction_norm": correction_norm,
                "displacement_l2_rel_error": displacement_rel,
                "strain_l2_rel_error": strain_rel,
                "stress_l2_rel_error": stress_rel,
                "von_mises_l2_rel_error": _relative_vector_error(vm_sfc, vm_calc),
                "contact_force_rel_error": rf_rel,
                "contact_energy_rel_error": cels_rel,
                "active_contact_count": contact_final.active_count,
                "min_gap": contact_final.min_gap,
                "max_penetration": contact_final.max_penetration,
            }
        ],
    )
    row = {
        "case_id": "c3d8_nonlinear_static_contact",
        "analysis_type": "static",
        "linearity": "geometric_nonlinear",
        "element_type": "C3D8",
        "native_sfc_result": "true",
        "calculix_comparison": "true",
        "acceleration_evidence": "false",
        "displacement_metric": displacement_rel,
        "stress_strain_metric": stress_rel,
        "gap_metric": contact_final.max_penetration,
        "rf_metric": rf_rel,
        "cels_metric": cels_rel,
        "active_contact_metric": contact_final.active_count,
        "supports_external_correctness": _bool_text(metric_ok),
        "supports_trajectory_equivalence": "false",
        "supports_efficiency": "false",
        "status": "supported" if metric_ok else "native_external_comparison_failed",
        "evidence_file": _display_path(evidence_path),
    }
    stress_rows = [
        {
            "case_id": "c3d8_nonlinear_static_contact",
            "frame": 0,
            "element_id": int(element_id),
            "von_mises": float(vm),
            "engineering_strain_norm": float(strain_norm),
            "vtk_file": "vtk/c3d8_nonlinear_static_contact_0000.vtk",
        }
        for element_id, vm, strain_norm in zip(model.element_ids, vm_sfc, strain_norm_sfc, strict=True)
    ]
    return row, stress_rows


def _linear_static_evidence(out_dir: Path) -> tuple[Row, list[Row]]:
    source_dir = ROOT / "paper" / "numerical_experiments" / "calculix_contactenergy_c3d8_replay"
    replay_csv = source_dir / "data" / "calculix_contactenergy_replay.csv"
    inp_path = source_dir / "data" / "contactenergy.inp"
    dat_path = source_dir / "data" / "contactenergy.dat"
    if not replay_csv.exists() or not inp_path.exists() or not dat_path.exists():
        row = {
            "case_id": "c3d8_linear_static_contact",
            "analysis_type": "static",
            "linearity": "linear",
            "element_type": "C3D8",
            "native_sfc_result": "false",
            "calculix_comparison": "false",
            "acceleration_evidence": "false",
            "displacement_metric": "",
            "stress_strain_metric": "",
            "gap_metric": "",
            "rf_metric": "",
            "cels_metric": "",
            "active_contact_metric": "",
            "supports_external_correctness": "false",
            "supports_trajectory_equivalence": "false",
            "supports_efficiency": "false",
            "status": "blocked",
            "evidence_file": "",
        }
        return row, []

    replay = _rows(replay_csv)[0]
    model = parse_contactenergy_input(inp_path.read_text(encoding="utf-8"))
    U, _RF, _energy, _raw = parse_contactenergy_dat(dat_path.read_text(encoding="utf-8", errors="ignore"), model.node_ids)
    _strains, _stresses, vm, strain_norm = contactenergy_hex8_center_strain_stress(model, U)
    _write_legacy_hex_vtk(
        out_dir / "vtk" / "c3d8_linear_static_contact_0000.vtk",
        model.X + U,
        model.elements,
        cell_values={"von_mises": vm, "engineering_strain_norm": strain_norm},
    )
    row = {
        "case_id": "c3d8_linear_static_contact",
        "analysis_type": "static",
        "linearity": "linear",
        "element_type": "C3D8",
        "native_sfc_result": "true",
        "calculix_comparison": "true",
        "acceleration_evidence": "false",
        "displacement_metric": replay.get("sfc_c3d8_displacement_l2_rel_error", ""),
        "stress_strain_metric": replay.get("sfc_c3d8_von_mises_l2_rel_error", ""),
        "gap_metric": replay.get("sfc_dynamic_sdf_max_penetration", ""),
        "rf_metric": replay.get("contact_force_rel_error", ""),
        "cels_metric": replay.get("contact_energy_rel_error", ""),
        "active_contact_metric": replay.get("sfc_dynamic_sdf_active_quadrature_count", ""),
        "supports_external_correctness": "true",
        "supports_trajectory_equivalence": "false",
        "supports_efficiency": "false",
        "status": "supported",
        "evidence_file": _display_path(replay_csv),
    }
    stress_rows = [
        {
            "case_id": "c3d8_linear_static_contact",
            "frame": 0,
            "element_id": int(element_id),
            "von_mises": float(value),
            "engineering_strain_norm": float(strain),
            "vtk_file": "vtk/c3d8_linear_static_contact_0000.vtk",
        }
        for element_id, value, strain in zip(model.element_ids, vm, strain_norm, strict=True)
    ]
    return row, stress_rows


def _run_c3d8_trajectory(out_dir: Path, *, quick: bool, skip_calculix: bool) -> Path:
    traj_dir = out_dir / "c3d8_dynamic_trajectory"
    args = ["--out-dir", str(traj_dir)]
    if quick:
        args.append("--quick")
    if skip_calculix:
        args.append("--skip-calculix")
    rc = run_c3d8_trajectory_main(args)
    if rc != 0:
        raise RuntimeError(f"C3D8 trajectory runner failed with return code {rc}")
    return traj_dir


def _linear_contact_response(contact: PlaneContactGeometry, x_current: np.ndarray, n_nodes: int):
    return assemble_contact_response(contact.samples(x_current), n_nodes)


def _native_c3d8_linear_dynamic_history(model) -> tuple[list[Row], np.ndarray, np.ndarray, list[tuple[float, np.ndarray]]]:
    """Run native small-strain C3D8 Newmark dynamics with SDF contact."""

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
    u = np.zeros(n_dofs, dtype=float)
    v = np.zeros(n_dofs, dtype=float)
    v[3 * model.slave_node_indices + 2] = float(model.initial_velocity_z)
    if fixed.size:
        u[fixed] = 0.0
        v[fixed] = 0.0
    c0 = 1.0 / (beta * dt * dt)
    initial_contact = _linear_contact_response(contact, model.X + u.reshape((-1, 3)), body.n_nodes)
    initial_rhs = f_ext + initial_contact.force.reshape(-1) - K @ u
    if fixed.size:
        a = np.zeros(n_dofs, dtype=float)
        a[free] = np.asarray(spsolve(M[free[:, None], free].tocsc(), initial_rhs[free]), dtype=float)
    else:
        a = np.asarray(spsolve(M.tocsc(), initial_rhs), dtype=float)
    rows: list[Row] = []
    frames: list[tuple[float, np.ndarray]] = []
    steps = int(np.ceil(float(model.total_time) / dt))
    for step in range(steps + 1):
        x_current = model.X + u.reshape((-1, 3))
        contact_response = _linear_contact_response(contact, x_current, body.n_nodes)
        replay = _replay_dynamic_sdf(model, u.reshape((-1, 3)))
        frames.append((float(step * dt), u.reshape((-1, 3)).copy()))
        rows.append(
            {
                "case": model.case,
                "resolution": model.resolution,
                "time": float(step * dt),
                "z_cm": float(np.mean(x_current[model.slave_node_indices, 2])),
                "min_gap": float(contact_response.min_gap),
                "max_penetration": float(contact_response.max_penetration),
                "active_contact_count": int(contact_response.active_count),
                "normal_force": float(abs(contact_response.normal_force)),
                "contact_energy": float(contact_response.energy),
                "sdf_replay_min_gap": replay["sfc_min_gap"],
                "sdf_replay_normal_force": abs(float(replay["sfc_normal_force_z"])),
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
        for iteration in range(12):
            x_guess = model.X + u_guess.reshape((-1, 3))
            trial_contact = _linear_contact_response(contact, x_guess, body.n_nodes)
            a_guess = c0 * (u_guess - u_pred)
            residual = M @ a_guess + K @ u_guess - f_ext - trial_contact.force.reshape(-1)
            residual_norm = float(np.linalg.norm(residual[free]))
            tangent = (M * c0 + K + trial_contact.tangent).tocsc()
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
    vm_final: list[float] = []
    U_final = u.reshape((-1, 3))
    for element in model.elements:
        _strain, stress = hex8_center_strain_stress(model.X[element], U_final[element], model.E, model.nu)
        vm_final.append(_von_mises(stress))
    return rows, U_final, np.asarray(vm_final, dtype=float), frames


def _run_c3d8_linear_dynamic_calculix(model, out_dir: Path, *, skip_calculix: bool) -> tuple[list[Row], Row, dict[float, np.ndarray], dict[float, dict[int, np.ndarray]]]:
    if skip_calculix or not calculix_available():
        return [], {
            "case": f"c3d8_linear_dynamic_{model.case}",
            "external_solver": "CalculiX",
            "completed": "false",
            "command": "skipped",
            "wall_time_seconds": "",
        }, {}, {}
    case_name = f"linear_dynamic_{model.case}_r{model.resolution}"
    run_dir = out_dir / "c3d8_linear_dynamic_calculix" / case_name
    run_dir.mkdir(parents=True, exist_ok=True)
    inp_path = run_dir / f"{case_name}.inp"
    write_calculix_input(model, inp_path)
    text = inp_path.read_text(encoding="utf-8")
    text = text.replace("*STEP, NLGEOM", "*STEP")
    inp_path.write_text(text, encoding="utf-8")
    command = f"cd {_wsl_path(run_dir)} && ccx {case_name}"
    start = time.perf_counter()
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    wall_time = time.perf_counter() - start
    (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
    dat_path = run_dir / f"{case_name}.dat"
    command_row = {
        "case": f"c3d8_linear_dynamic_{model.case}",
        "external_solver": "CalculiX",
        "external_solver_version": _calculix_version(),
        "completed": _bool_text(proc.returncode == 0 and dat_path.exists()),
        "return_code": proc.returncode,
        "command": f"wsl --exec bash -lc \"cd {_wsl_path(run_dir)} && ccx {case_name}\"",
        "input_file": _display_path(inp_path),
        "dat_file": _display_path(dat_path),
        "wall_time_seconds": wall_time,
    }
    if proc.returncode != 0 or not dat_path.exists():
        return [], command_row, {}, {}
    displacements = _parse_nodal_vectors(dat_path, model.node_ids, quantity="u")
    stresses = _parse_element_stress(dat_path)
    totals = _parse_totals(dat_path)
    contacts = _parse_contact_elements(dat_path)
    rows: list[Row] = []
    for time_value, U in displacements.items():
        total = totals.get(float(time_value), {})
        contact_rows = contacts.get(float(time_value), [])
        cdis_values = [float(row["clearance"]) for row in contact_rows if row.get("quantity") == "cdis" and "clearance" in row]
        cels_values = [float(row["energy"]) for row in contact_rows if row.get("quantity") == "cels" and "energy" in row]
        replay = _replay_dynamic_sdf(model, U)
        rows.append(
            {
                "time": float(time_value),
                "z_cm": _center_z(model, U),
                "min_gap": min(cdis_values) if cdis_values else replay["sfc_min_gap"],
                "normal_force": total.get("normal_force_calculix", ""),
                "contact_energy": total.get("contact_energy_calculix", float(np.sum(cels_values)) if cels_values else ""),
                "active_contact_count": total.get("contact_count_calculix", len(cdis_values) if cdis_values else ""),
            }
        )
    return rows, command_row, displacements, stresses


def _compare_histories(native_rows: list[Row], reference_rows: list[Row]) -> Row:
    ref_by_time = {round(float(row["time"]), 12): row for row in reference_rows}
    z_native: list[float] = []
    z_ref: list[float] = []
    gap_native: list[float] = []
    gap_ref: list[float] = []
    force_native: list[float] = []
    force_ref: list[float] = []
    energy_native: list[float] = []
    energy_ref: list[float] = []
    active_err: list[float] = []
    for row in native_rows:
        ref = ref_by_time.get(round(float(row["time"]), 12))
        if ref is None:
            continue
        z_native.append(float(row["z_cm"]))
        z_ref.append(float(ref["z_cm"]))
        if ref.get("min_gap", "") != "":
            gap_native.append(float(row["min_gap"]))
            gap_ref.append(float(ref["min_gap"]))
        if ref.get("normal_force", "") != "":
            force_native.append(float(row["normal_force"]))
            force_ref.append(float(ref["normal_force"]))
        if ref.get("contact_energy", "") != "":
            energy_native.append(float(row["contact_energy"]))
            energy_ref.append(float(ref["contact_energy"]))
        if ref.get("active_contact_count", "") != "":
            active_err.append(abs(float(row["active_contact_count"]) - float(ref["active_contact_count"])))
    return {
        "z_cm_l2_rel_error": "" if not z_native else _relative_vector_error(np.asarray(z_native), np.asarray(z_ref)),
        "gap_l2_rel_error": "" if not gap_native else _relative_vector_error(np.asarray(gap_native), np.asarray(gap_ref)),
        "force_l2_rel_error": "" if not force_native else _relative_vector_error(np.asarray(force_native), np.asarray(force_ref)),
        "energy_l2_rel_error": "" if not energy_native else _relative_vector_error(np.asarray(energy_native), np.asarray(energy_ref)),
        "active_count_linf_abs_error": "" if not active_err else float(np.max(active_err)),
    }


def _linear_dynamic_evidence(traj_dir: Path, out_dir: Path, *, quick: bool, skip_calculix: bool) -> tuple[list[Row], list[Row], list[Row], list[Row]]:
    summary_path = traj_dir / "c3d8_contact_trajectory_summary.csv"
    stress_rows: list[Row] = []
    if summary_path.exists():
        for case in ["block_plane_c3d8", "block_block_c3d8"]:
            model = build_c3d8_model(case=case, resolution=1, quick=True)
            dat_path = traj_dir / "calculix_runs" / f"{case}_r1" / f"{case}_r1.dat"
            if dat_path.exists():
                U_blocks = _parse_nodal_vectors(dat_path, model.node_ids, quantity="u")
                stress_blocks = _parse_element_stress(dat_path)
                if U_blocks:
                    time = max(U_blocks)
                    U = U_blocks[time]
                    stress_map = stress_blocks.get(time, {})
                else:
                    U = np.zeros_like(model.X)
                    stress_map = {}
            else:
                U = np.zeros_like(model.X)
                stress_map = {}
            vm_values: list[float] = []
            strain_values: list[float] = []
            for element_id, element in zip(model.element_ids, model.elements, strict=True):
                strain, sfc_stress = hex8_center_strain_stress(model.X[element], U[element], model.E, model.nu)
                stress = stress_map.get(int(element_id), sfc_stress)
                vm_values.append(_von_mises(stress))
                strain_values.append(float(np.linalg.norm(strain)))
                stress_rows.append(
                    {
                        "case_id": f"c3d8_linear_dynamic_contact_{case}",
                        "frame": 0,
                        "element_id": int(element_id),
                        "von_mises": vm_values[-1],
                        "engineering_strain_norm": strain_values[-1],
                        "vtk_file": f"vtk/c3d8_linear_dynamic_contact_{case}_0000.vtk",
                    }
                )
            _write_legacy_hex_vtk(
                out_dir / "vtk" / f"c3d8_linear_dynamic_contact_{case}_0000.vtk",
                model.X + U,
                model.elements,
                cell_values={
                    "von_mises": np.asarray(vm_values, dtype=float),
                    "engineering_strain_norm": np.asarray(strain_values, dtype=float),
                },
            )

    rows: list[Row] = []
    comparison_rows: list[Row] = []
    command_rows: list[Row] = []
    timing_rows: list[Row] = []
    history_fields = [
        "case",
        "resolution",
        "time",
        "z_cm",
        "min_gap",
        "max_penetration",
        "active_contact_count",
        "normal_force",
        "contact_energy",
        "sdf_replay_min_gap",
        "sdf_replay_normal_force",
        "newton_iterations",
        "newton_residual_norm",
    ]
    for case in ["block_plane_c3d8", "block_block_c3d8"]:
        model = build_c3d8_model(case=case, resolution=1, quick=quick)
        case_label = "block_plane" if case == "block_plane_c3d8" else "block_block"
        case_id = f"c3d8_linear_dynamic_{case_label}_contact"
        (native_result, sfc_wall_time) = _timed_call(_native_c3d8_linear_dynamic_history, model)
        native_rows, U_final, vm_final, native_frames = native_result
        native_path = out_dir / f"native_c3d8_linear_dynamic_{case_label}.csv"
        _write_csv(native_path, history_fields, native_rows)
        reference_rows, command_row, calculix_displacements, calculix_stresses = _run_c3d8_linear_dynamic_calculix(model, out_dir, skip_calculix=skip_calculix)
        command_rows.append({"command": command_row.get("command", ""), "description": f"C3D8 linear dynamic CalculiX reference for {case}"})
        if calculix_displacements:
            calc_frames = [(float(time), U) for time, U in sorted(calculix_displacements.items())]
            calc_frame_outputs = write_hex_frame_series(
                out_dir / "vtk_frames" / case_id / "calculix",
                stem=f"calculix_{case_id}",
                reference_points=model.X,
                elements=model.elements,
                frames=calc_frames,
                frame_stride=1,
                cell_scalar_fn=lambda time_value, U, model=model, stresses=calculix_stresses: _phase9_hex_cell_fields(
                    model,
                    U,
                    stresses.get(float(time_value), {}),
                ),
                source_label=f"CalculiX {case_id}",
            )
            command_rows.append(
                {
                    "command": "write CalculiX dynamic VTK/PVD frame sequence",
                    "description": f"CalculiX animation frames for {case}: {_display_path(Path(calc_frame_outputs['pvd']))} ({calc_frame_outputs['frame_count']} frames)",
                }
            )
        metrics = _compare_histories(native_rows, reference_rows)
        comparison_rows.append({"case_id": case_id, **metrics})
        _write_legacy_hex_vtk(
            out_dir / "vtk" / f"{case_id}_native_0000.vtk",
            model.X + U_final,
            model.elements,
            cell_values={"von_mises": vm_final, "engineering_strain_norm": np.zeros_like(vm_final)},
        )
        frame_outputs = write_hex_frame_series(
            out_dir / "vtk_frames" / case_id / "sfc",
            stem=f"sfc_{case_id}",
            reference_points=model.X,
            elements=model.elements,
            frames=native_frames,
            frame_stride=1,
            cell_scalar_fn=lambda _time, U, model=model: _phase9_hex_cell_fields(model, U),
            source_label=f"SFC native {case_id}",
        )
        command_rows.append(
            {
                "command": "write SFC dynamic VTK/PVD frame sequence",
                "description": f"SFC animation frames for {case}: {_display_path(Path(frame_outputs['pvd']))} ({frame_outputs['frame_count']} frames)",
            }
        )
        stress_rows.extend(
            {
                "case_id": f"{case_id}_native",
                "frame": 0,
                "element_id": int(element_id),
                "von_mises": float(vm),
                "engineering_strain_norm": 0.0,
                "vtk_file": f"vtk/{case_id}_native_0000.vtk",
            }
            for element_id, vm in zip(model.element_ids, vm_final, strict=True)
        )
        external = bool(reference_rows)
        metric_ok = (
            external
            and metrics["z_cm_l2_rel_error"] != ""
            and float(metrics["z_cm_l2_rel_error"]) < 0.10
            and (metrics["gap_l2_rel_error"] == "" or float(metrics["gap_l2_rel_error"]) < 0.75)
            and (metrics["force_l2_rel_error"] == "" or float(metrics["force_l2_rel_error"]) < 0.90)
            and (metrics["energy_l2_rel_error"] == "" or float(metrics["energy_l2_rel_error"]) < 0.90)
        )
        active = max(int(row["active_contact_count"]) for row in native_rows) if native_rows else 0
        calc_wall = command_row.get("wall_time_seconds", "")
        speedup = ""
        if calc_wall not in {"", None} and float(calc_wall) > 0.0:
            speedup = float(calc_wall) / max(float(sfc_wall_time), 1.0e-30)
        timing_rows.append(
            {
                "case_id": case_id,
                "analysis_type": "dynamic",
                "linearity": "linear",
                "element_type": "C3D8",
                "sfc_wall_time_seconds": sfc_wall_time,
                "calculix_wall_time_seconds": calc_wall,
                "speedup_vs_calculix": speedup,
                "timing_claim_allowed": _bool_text(speedup != "" and float(speedup) > 1.0 and metric_ok),
                "notes": "native SFC Newmark trajectory vs no-NLGEOM CalculiX dynamic contact",
            }
        )
        rows.append(
            {
                "case_id": case_id,
                "analysis_type": "dynamic",
                "linearity": "linear",
                "element_type": "C3D8",
                "native_sfc_result": "true",
                "calculix_comparison": _bool_text(external),
                "acceleration_evidence": _bool_text(speedup != "" and float(speedup) > 1.0),
                "displacement_metric": metrics["z_cm_l2_rel_error"],
                "stress_strain_metric": "final_native_linear_stress_cloud",
                "gap_metric": metrics["gap_l2_rel_error"],
                "rf_metric": metrics["force_l2_rel_error"],
                "cels_metric": metrics["energy_l2_rel_error"],
                "active_contact_metric": active,
                "supports_external_correctness": _bool_text(metric_ok),
                "supports_trajectory_equivalence": _bool_text(metric_ok),
                "supports_efficiency": _bool_text(speedup != "" and float(speedup) > 1.0 and metric_ok),
                "status": "supported" if metric_ok else ("native_only_no_calculix" if not external else "native_external_comparison_failed"),
                "evidence_file": _display_path(native_path),
            }
        )
    _write_csv(
        out_dir / "native_c3d8_linear_dynamic_comparison.csv",
        [
            "case_id",
            "z_cm_l2_rel_error",
            "gap_l2_rel_error",
            "force_l2_rel_error",
            "energy_l2_rel_error",
            "active_count_linf_abs_error",
        ],
        comparison_rows,
    )
    return rows, stress_rows, command_rows, timing_rows


def _bottom_surface_triangles(model) -> np.ndarray:
    triangles: list[np.ndarray] = []
    for element_index, face in model.slave_surface:
        element = model.elements[int(element_index)]
        quad = element[C3D8_FACE_NODES[face]]
        oriented = _trajectory_triangles_from_quads(model.X, [(quad, element)])
        triangles.extend(oriented)
    if not triangles:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)


def _initial_hht_state_for_model(mechanics: MechanicsModel, contact, model, fixed: np.ndarray, *, alpha: float) -> tuple[MechanicsState, np.ndarray]:
    v = np.zeros_like(model.X)
    v[model.slave_node_indices, 2] = float(model.initial_velocity_z)
    state = MechanicsState(model.X.copy(), v, np.zeros_like(model.X), time=0.0)
    static_state = static_force_state(mechanics, state.x, contact, gravity=model.gravity)
    beta, _gamma = hht_newmark_parameters(alpha)
    regularized_dt = float(model.dt) / 10.0
    initial_matrix = (
        mechanics.mass_matrix
        + static_state.tangent * (beta * regularized_dt * regularized_dt * (1.0 + float(alpha)))
    ).tocsr()
    rhs = static_state.calculix_rhs_balance
    free = _fixed_free(mechanics.n_dofs, fixed)
    acc = np.zeros(mechanics.n_dofs, dtype=float)
    if free.size:
        acc[free] = np.asarray(spsolve(initial_matrix[free[:, None], free].tocsc(), rhs[free]), dtype=float)
    state.a = acc.reshape((-1, 3))
    if fixed.size:
        flat_v = state.v.reshape(-1)
        flat_a = state.a.reshape(-1)
        flat_v[fixed] = 0.0
        flat_a[fixed] = 0.0
    return state, static_state.calculix_rhs_balance


def _hht_step_with_fixed_dofs(
    mechanics: MechanicsModel,
    state: MechanicsState,
    previous_static_residual: np.ndarray,
    contact,
    *,
    fixed: np.ndarray,
    dt: float,
    gravity: float,
    alpha: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[MechanicsState, np.ndarray, Any]:
    beta, gamma = hht_newmark_parameters(alpha)
    c0 = 1.0 / (beta * dt * dt)
    u = (state.x - mechanics.X).reshape(-1)
    v = state.v.reshape(-1)
    a = state.a.reshape(-1)
    free = _fixed_free(mechanics.n_dofs, fixed)
    u_pred, v_pred, _ = calculix_dynamic_predictor(u, v, a, dt=dt, beta=beta, gamma=gamma)
    if fixed.size:
        u_pred[fixed] = 0.0
        v_pred[fixed] = 0.0
    u_guess = u_pred.copy()
    previous_rhs_balance = np.asarray(previous_static_residual, dtype=float)
    residual_norm = np.inf
    iteration_count = 0
    for iteration in range(max(1, int(max_iterations))):
        x_guess = mechanics.X + u_guess.reshape((-1, 3))
        static_state = static_force_state(mechanics, x_guess, contact, gravity=gravity)
        a_guess = c0 * (u_guess - u_pred)
        if fixed.size:
            a_guess[fixed] = 0.0
        residual = calculix_hht_effective_residual(
            mechanics.mass_matrix @ a_guess,
            static_state.calculix_rhs_balance,
            previous_rhs_balance,
            alpha=alpha,
        )
        residual_norm = float(np.linalg.norm(residual[free]))
        tangent = calculix_hht_effective_tangent(mechanics.mass_matrix, static_state.tangent, dt=dt, beta=beta, alpha=alpha).tocsc()
        correction = np.zeros(mechanics.n_dofs, dtype=float)
        if free.size:
            correction[free] = np.asarray(spsolve(tangent[free[:, None], free], -residual[free]), dtype=float)
        u_guess[free] += correction[free]
        if fixed.size:
            u_guess[fixed] = 0.0
        iteration_count = iteration + 1
        if float(np.linalg.norm(correction[free])) <= tolerance * max(1.0, float(np.linalg.norm(u_guess[free]))):
            break
    a_new = c0 * (u_guess - u_pred)
    v_new = v_pred + gamma * dt * a_new
    if fixed.size:
        u_guess[fixed] = 0.0
        v_new[fixed] = 0.0
        a_new[fixed] = 0.0
    next_state = MechanicsState(
        mechanics.X + u_guess.reshape((-1, 3)),
        v_new.reshape((-1, 3)),
        a_new.reshape((-1, 3)),
        time=state.time + float(dt),
    )
    diagnostics = evaluate_state(mechanics, next_state, contact, gravity=gravity, assemble_tangent=True)
    diagnostics.newton_iterations = iteration_count
    diagnostics.newton_residual_norm = residual_norm
    accepted = static_force_state(mechanics, next_state.x, contact, gravity=gravity)
    return next_state, accepted.calculix_rhs_balance, diagnostics


def _native_c3d8_dynamic_history(model, *, max_steps: int | None = None) -> tuple[list[Row], np.ndarray, np.ndarray]:
    mechanics = MechanicsModel.from_hex8_mesh(
        model.X,
        model.elements,
        E=model.E,
        nu=model.nu,
        density=model.density,
    )
    contact = _contact_geometry_for_model(model)
    fixed = _model_fixed_dofs(model)
    state, previous = _initial_hht_state_for_model(mechanics, contact, model, fixed, alpha=-0.05)
    rows: list[Row] = []
    steps = int(np.ceil(model.total_time / model.dt))
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    for step in range(steps + 1):
        diagnostics = evaluate_state(mechanics, state, contact, gravity=model.gravity, assemble_tangent=True)
        replay = _replay_dynamic_sdf(model, state.x - model.X)
        rows.append(
            {
                "case": model.case,
                "resolution": model.resolution,
                "time": float(state.time),
                "z_cm": float(np.mean(state.x[model.slave_node_indices, 2])),
                "min_gap": float(diagnostics.contact.min_gap),
                "max_penetration": float(diagnostics.contact.max_penetration),
                "active_contact_count": int(diagnostics.contact.active_count),
                "normal_force": float(abs(diagnostics.contact.normal_force)),
                "contact_energy": float(diagnostics.contact.energy),
                "sdf_replay_min_gap": replay["sfc_min_gap"],
                "sdf_replay_normal_force": abs(float(replay["sfc_normal_force_z"])),
                "newton_iterations": int(diagnostics.newton_iterations),
                "newton_residual_norm": float(diagnostics.newton_residual_norm),
            }
        )
        if step == steps:
            break
        dt = min(float(model.dt), float(model.total_time) - float(state.time))
        if fixed.size:
            state, previous, _diagnostics = _hht_step_with_fixed_dofs(
                mechanics,
                state,
                previous,
                contact,
                fixed=fixed,
                dt=dt,
                gravity=model.gravity,
                alpha=-0.05,
                max_iterations=14,
                tolerance=1.0e-9,
            )
        else:
            state, previous, _diagnostics = hht_step(
                mechanics,
                state,
                previous,
                contact,
                dt=dt,
                gravity=model.gravity,
                alpha=-0.05,
                max_iterations=14,
                tolerance=1.0e-9,
            )
    final_response = stvk_internal_response(mechanics, state.x, assemble_tangent=True)
    return rows, state.x - model.X, final_response.von_mises


def _compare_native_dynamic_to_calculix(native_rows: list[Row], reference_rows: list[dict[str, str]], *, case: str = "block_plane_c3d8") -> Row:
    reference = [row for row in reference_rows if row["case"] == case and row["resolution"] == "1"]
    if not reference:
        return {
            "z_cm_l2_rel_error": "",
            "gap_l2_rel_error": "",
            "force_l2_rel_error": "",
            "energy_l2_rel_error": "",
            "active_count_linf_abs_error": "",
        }
    ref_by_time = {round(float(row["time"]), 12): row for row in reference}
    z_native: list[float] = []
    z_ref: list[float] = []
    gap_native: list[float] = []
    gap_ref: list[float] = []
    force_native: list[float] = []
    force_ref: list[float] = []
    energy_native: list[float] = []
    energy_ref: list[float] = []
    active_err: list[float] = []
    for row in native_rows:
        ref = ref_by_time.get(round(float(row["time"]), 12))
        if ref is None:
            continue
        z_native.append(float(row["z_cm"]))
        z_ref.append(float(ref["z_cm"]))
        gap_native.append(float(row["min_gap"]))
        gap_ref.append(float(ref["min_cdis"]) if ref.get("min_cdis", "") else float(ref["sfc_min_gap"]))
        force_native.append(float(row["normal_force"]))
        force_ref.append(abs(float(ref["normal_force_calculix"])) if ref.get("normal_force_calculix", "") else abs(float(ref["sfc_normal_force_z"])))
        energy_native.append(float(row["contact_energy"]))
        energy_ref.append(float(ref["contact_energy_calculix"]) if ref.get("contact_energy_calculix", "") else float(ref["sfc_contact_energy"]))
        active_err.append(abs(float(row["active_contact_count"]) - float(ref["contact_count_calculix"] if ref.get("contact_count_calculix", "") else ref["sfc_active_contact_count"])))
    if not z_native:
        return {
            "z_cm_l2_rel_error": "",
            "gap_l2_rel_error": "",
            "force_l2_rel_error": "",
            "energy_l2_rel_error": "",
            "active_count_linf_abs_error": "",
        }
    return {
        "z_cm_l2_rel_error": _relative_vector_error(np.asarray(z_native), np.asarray(z_ref)),
        "gap_l2_rel_error": _relative_vector_error(np.asarray(gap_native), np.asarray(gap_ref)),
        "force_l2_rel_error": _relative_vector_error(np.asarray(force_native), np.asarray(force_ref)),
        "energy_l2_rel_error": _relative_vector_error(np.asarray(energy_native), np.asarray(energy_ref)),
        "active_count_linf_abs_error": float(np.max(active_err)) if active_err else 0.0,
    }


def _native_nonlinear_dynamic_evidence(traj_dir: Path, out_dir: Path, *, quick: bool) -> tuple[list[Row], list[Row], list[Row]]:
    stress_rows: list[Row] = []
    rows: list[Row] = []
    timing_rows: list[Row] = []
    reference_csv = traj_dir / "c3d8_contact_trajectory.csv"
    reference_rows = _rows(reference_csv) if reference_csv.exists() else []
    command_times: dict[str, float] = {}
    command_csv = traj_dir / "c3d8_contact_trajectory_commands.csv"
    if command_csv.exists():
        for command in _rows(command_csv):
            if command.get("wall_time_seconds", ""):
                command_times[f"{command.get('case')}_r{command.get('resolution')}"] = float(command["wall_time_seconds"])
    comparison_rows: list[Row] = []
    history_fields = [
        "case",
        "resolution",
        "time",
        "z_cm",
        "min_gap",
        "max_penetration",
        "active_contact_count",
        "normal_force",
        "contact_energy",
        "sdf_replay_min_gap",
        "sdf_replay_normal_force",
        "newton_iterations",
        "newton_residual_norm",
    ]
    for case in ["block_plane_c3d8", "block_block_c3d8"]:
        model = build_c3d8_model(case=case, resolution=1, quick=quick)
        case_label = "block_plane" if case == "block_plane_c3d8" else "block_block"
        case_id = f"c3d8_nonlinear_dynamic_{case_label}_contact"
        (native_result, sfc_wall_time) = _timed_call(_native_c3d8_dynamic_history, model)
        native_rows, U_final, vm_final = native_result
        native_path = out_dir / f"native_c3d8_nonlinear_dynamic_{case_label}.csv"
        _write_csv(native_path, history_fields, native_rows)
        metrics = _compare_native_dynamic_to_calculix(native_rows, reference_rows, case=case)
        comparison_rows.append({"case_id": case_id, **metrics})
        force_error = metrics["force_l2_rel_error"]
        gap_error = metrics["gap_l2_rel_error"]
        energy_error = metrics["energy_l2_rel_error"]
        z_error = metrics["z_cm_l2_rel_error"]
        external = bool([row for row in reference_rows if row.get("case") == case])
        metric_ok = (
            external
            and force_error != ""
            and float(z_error) < 0.25
            and float(gap_error) < 0.75
            and float(force_error) < 0.90
            and float(energy_error) < 0.90
        )
        calc_wall = command_times.get(f"{case}_r1", "")
        speedup = ""
        if calc_wall not in {"", None} and float(calc_wall) > 0.0:
            speedup = float(calc_wall) / max(float(sfc_wall_time), 1.0e-30)
        timing_rows.append(
            {
                "case_id": case_id,
                "analysis_type": "dynamic",
                "linearity": "geometric_nonlinear",
                "element_type": "C3D8",
                "sfc_wall_time_seconds": sfc_wall_time,
                "calculix_wall_time_seconds": calc_wall,
                "speedup_vs_calculix": speedup,
                "timing_claim_allowed": _bool_text(speedup != "" and float(speedup) > 1.0 and metric_ok),
                "notes": "native SFC HHT/StVK trajectory vs CalculiX NLGEOM dynamic contact",
            }
        )
        _write_legacy_hex_vtk(
            out_dir / "vtk" / f"{case_id}_0000.vtk",
            model.X + U_final,
            model.elements,
            cell_values={"von_mises": vm_final, "green_lagrange_strain_norm": np.zeros_like(vm_final)},
        )
        stress_rows.extend(
            {
                "case_id": case_id,
                "frame": 0,
                "element_id": int(element_id),
                "von_mises": float(vm),
                "engineering_strain_norm": 0.0,
                "vtk_file": f"vtk/{case_id}_0000.vtk",
            }
            for element_id, vm in zip(model.element_ids, vm_final, strict=True)
        )
        rows.append(
            {
                "case_id": case_id,
                "analysis_type": "dynamic",
                "linearity": "geometric_nonlinear",
                "element_type": "C3D8",
                "native_sfc_result": "true",
                "calculix_comparison": _bool_text(external),
                "acceleration_evidence": _bool_text(speedup != "" and float(speedup) > 1.0),
                "displacement_metric": z_error,
                "stress_strain_metric": "final_native_stvk_stress_cloud",
                "gap_metric": gap_error,
                "rf_metric": force_error,
                "cels_metric": energy_error,
                "active_contact_metric": metrics["active_count_linf_abs_error"],
                "supports_external_correctness": _bool_text(metric_ok),
                "supports_trajectory_equivalence": _bool_text(metric_ok),
                "supports_efficiency": _bool_text(speedup != "" and float(speedup) > 1.0 and metric_ok),
                "status": "supported" if metric_ok else "native_external_comparison_failed",
                "evidence_file": _display_path(native_path),
            }
        )
    _write_csv(
        out_dir / "native_c3d8_nonlinear_dynamic_comparison.csv",
        [
            "case_id",
            "z_cm_l2_rel_error",
            "gap_l2_rel_error",
            "force_l2_rel_error",
            "energy_l2_rel_error",
            "active_count_linf_abs_error",
        ],
        comparison_rows,
    )
    return rows, stress_rows, timing_rows


def _build_curved_nonplanar_c3d8_model(*, quick: bool):
    base = build_c3d8_model(
        case="block_block_c3d8",
        resolution=2,
        quick=quick,
        total_time=0.04 if quick else 0.06,
        dt=0.002 if quick else 0.001,
    )
    X = base.X.copy()
    slave = np.zeros(X.shape[0], dtype=bool)
    slave[base.slave_node_indices] = True
    master = ~slave
    top_z = float(np.max(X[master, 2]))
    top = master & np.isclose(X[:, 2], top_z)
    x = X[top, 0]
    y = X[top, 1]
    amplitude = 0.014
    X[top, 2] += amplitude * (1.0 - 8.0 * (x * x + 0.5 * y * y))
    lift = max(0.0, float(np.max(X[top, 2]) - top_z))
    X[slave, 2] += lift
    return replace(base, case="curved_nonplanar_c3d8", X=X)


def _curved_nonplanar_external_evidence(out_dir: Path, *, quick: bool, skip_calculix: bool) -> tuple[Row, list[Row], list[Row], list[Row]]:
    model = _build_curved_nonplanar_c3d8_model(quick=quick)
    case_id = "c3d8_curved_nonplanar_contact_replay"
    history: list[Row] = []
    stress_rows: list[Row] = []
    command_rows: list[Row] = []
    timing_rows: list[Row] = []
    external = False
    if skip_calculix or not calculix_available():
        U_blocks: dict[float, np.ndarray] = {}
        for i in range(1, 4):
            U = np.zeros_like(model.X)
            U[model.slave_node_indices, 2] -= 0.006 * i
            U_blocks[float(i * model.dt)] = U
        totals: dict[float, Row] = {}
        contacts: dict[float, list[Row]] = {}
        stresses: dict[float, dict[int, np.ndarray]] = {}
        command_rows.append({"command": "skipped CalculiX curved/non-planar C3D8 reference", "description": "Curved C3D8 external reference skipped"})
        calc_wall = ""
    else:
        try:
            (run_result, calc_wall) = _timed_call(run_calculix, model, out_dir / "curved_nonplanar_contact", timeout_seconds=240)
            dat_path, command = run_result
            external = True
            command_rows.append({"command": command.get("command", ""), "description": "Curved/non-planar C3D8 CalculiX external reference"})
            U_blocks = _parse_nodal_vectors(dat_path, model.node_ids, quantity="u")
            totals = _parse_totals(dat_path)
            contacts = _parse_contact_elements(dat_path)
            stresses = _parse_element_stress(dat_path)
        except (RuntimeError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            U_blocks = {}
            totals = {}
            contacts = {}
            stresses = {}
            calc_wall = ""
            command_rows.append({"command": "CalculiX curved/non-planar run failed", "description": "Curved C3D8 external reference"})
    gap_errors: list[float] = []
    force_errors: list[float] = []
    energy_errors: list[float] = []
    active_counts: list[int] = []
    for time_value, U in sorted(U_blocks.items()):
        replay = _replay_dynamic_sdf(model, U)
        total = totals.get(float(time_value), {})
        contact_rows = contacts.get(float(time_value), [])
        cdis_values = [float(row["clearance"]) for row in contact_rows if row.get("quantity") == "cdis" and "clearance" in row]
        cels_values = [float(row["energy"]) for row in contact_rows if row.get("quantity") == "cels" and "energy" in row]
        cdis = min(cdis_values) if cdis_values else ""
        normal_force = total.get("normal_force_calculix", "")
        cels = total.get("contact_energy_calculix", float(np.sum(cels_values)) if cels_values else "")
        cnum = total.get("contact_count_calculix", len(cdis_values) if cdis_values else "")
        gap_error = "" if cdis == "" else abs(float(cdis) - float(replay["sfc_min_gap"]))
        force_error = "" if normal_force == "" else abs(float(normal_force) - abs(float(replay["sfc_normal_force_z"])))
        energy_error = "" if cels == "" else abs(float(cels) - float(replay["sfc_contact_energy"]))
        if gap_error != "":
            gap_errors.append(float(gap_error))
        if force_error != "":
            force_errors.append(float(force_error))
        if energy_error != "":
            energy_errors.append(float(energy_error))
        active_counts.append(int(replay["sfc_active_contact_count"]))
        history.append(
            {
                "case_id": case_id,
                "time": time_value,
                "reference_source": "calculix_dat" if external else "synthetic_skip",
                "calculix_min_cdis": cdis,
                "sfc_min_gap": replay["sfc_min_gap"],
                "gap_abs_error": gap_error,
                "calculix_normal_force": normal_force,
                "sfc_normal_force_z": replay["sfc_normal_force_z"],
                "normal_force_abs_error": force_error,
                "calculix_contact_energy": cels,
                "sfc_contact_energy": replay["sfc_contact_energy"],
                "contact_energy_abs_error": energy_error,
                "calculix_contact_count": cnum,
                "sfc_active_contact_count": replay["sfc_active_contact_count"],
                "sfc_master_triangle_count": replay["sfc_master_triangle_count"],
                "nonplanar_master": "true",
            }
        )
    if U_blocks:
        last_time = max(U_blocks)
        U = U_blocks[last_time]
        stress_map = stresses.get(float(last_time), {})
        vm_values: list[float] = []
        strain_values: list[float] = []
        for element_id, element in zip(model.element_ids, model.elements, strict=True):
            strain, stress_sfc = hex8_center_strain_stress(model.X[element], U[element], model.E, model.nu)
            stress = stress_map.get(int(element_id), stress_sfc)
            vm_values.append(_von_mises(stress))
            strain_values.append(float(np.linalg.norm(strain)))
            stress_rows.append(
                {
                    "case_id": case_id,
                    "frame": 0,
                    "element_id": int(element_id),
                    "von_mises": vm_values[-1],
                    "engineering_strain_norm": strain_values[-1],
                    "vtk_file": "vtk/c3d8_curved_nonplanar_contact_replay_0000.vtk",
                }
            )
        _write_legacy_hex_vtk(
            out_dir / "vtk" / "c3d8_curved_nonplanar_contact_replay_0000.vtk",
            model.X + U,
            model.elements,
            cell_values={"von_mises": np.asarray(vm_values, dtype=float), "engineering_strain_norm": np.asarray(strain_values, dtype=float)},
        )
    evidence_path = out_dir / "phase9_curved_nonplanar_contact_external.csv"
    _write_csv(
        evidence_path,
        [
            "case_id",
            "time",
            "reference_source",
            "calculix_min_cdis",
            "sfc_min_gap",
            "gap_abs_error",
            "calculix_normal_force",
            "sfc_normal_force_z",
            "normal_force_abs_error",
            "calculix_contact_energy",
            "sfc_contact_energy",
            "contact_energy_abs_error",
            "calculix_contact_count",
            "sfc_active_contact_count",
            "sfc_master_triangle_count",
            "nonplanar_master",
        ],
        history,
    )
    max_gap_error = max(gap_errors or [0.0])
    peak_force = max([float(row["calculix_normal_force"]) for row in history if row["calculix_normal_force"] != ""] or [0.0])
    peak_energy = max([float(row["calculix_contact_energy"]) for row in history if row["calculix_contact_energy"] != ""] or [0.0])
    force_rel = _relative_error(max(force_errors or [0.0]), peak_force)
    energy_rel = _relative_error(max(energy_errors or [0.0]), peak_energy)
    metric_ok = external and (not force_errors or force_rel < 0.90) and (not energy_errors or energy_rel < 0.90)
    timing_rows.append(
        {
            "case_id": case_id,
            "analysis_type": "dynamic_replay",
            "linearity": "geometric_nonlinear",
            "element_type": "C3D8",
            "sfc_wall_time_seconds": "",
            "calculix_wall_time_seconds": calc_wall,
            "speedup_vs_calculix": "",
            "timing_claim_allowed": "false",
            "notes": "non-planar external CalculiX trajectory replay; not a native trajectory timing claim",
        }
    )
    row = {
        "case_id": case_id,
        "analysis_type": "dynamic_replay",
        "linearity": "geometric_nonlinear",
        "element_type": "C3D8",
        "native_sfc_result": "true",
        "calculix_comparison": _bool_text(external),
        "acceleration_evidence": "false",
        "displacement_metric": "calculix_trajectory_replay",
        "stress_strain_metric": "final_curved_nonplanar_stress_cloud",
        "gap_metric": max_gap_error,
        "rf_metric": force_rel if force_errors else "",
        "cels_metric": energy_rel if energy_errors else "",
        "active_contact_metric": max(active_counts or [0]),
        "supports_external_correctness": _bool_text(metric_ok),
        "supports_trajectory_equivalence": "false",
        "supports_efficiency": "false",
        "status": "external_sdf_replay_supported" if metric_ok else ("blocked_no_calculix" if not external else "external_sdf_replay_metric_failed"),
        "evidence_file": _display_path(evidence_path),
    }
    return row, stress_rows, command_rows, timing_rows


def _write_c3d4_c3d8_side_by_side(out_dir: Path, rows: list[Row]) -> list[Row]:
    table: list[Row] = []
    c3d4_path = ROOT / "paper" / "numerical_experiments" / "block_drop_dynamic_sdf_calculix_1s" / "data" / "trajectory_comparison.csv"
    if c3d4_path.exists():
        c3d4 = _rows(c3d4_path)[0]
        table.append(
            {
                "case_group": "block_plane_dynamic",
                "element_type": "C3D4",
                "scenario": "block_drop",
                "analysis_type": "dynamic",
                "linearity": "geometric_nonlinear",
                "z_cm_l2_rel_error": c3d4.get("z_cm_l2_rel_error", ""),
                "gap_or_cdis_error": c3d4.get("min_gap_l2_rel_error", ""),
                "force_rel_error": c3d4.get("peak_normal_force_rel_error", ""),
                "energy_rel_error": c3d4.get("peak_contact_energy_rel_error", ""),
                "active_contact_metric": c3d4.get("max_cnum_abs_error", ""),
                "claim_scope": "native SFC C3D4 dynamic-SDF trajectory against CalculiX block-plane case",
                "source_file": _display_path(c3d4_path),
            }
        )
    for row in rows:
        if row["element_type"] != "C3D8" or row["analysis_type"] not in {"dynamic", "dynamic_replay"}:
            continue
        table.append(
            {
                "case_group": row["case_id"].replace("c3d8_", "").replace("_contact", ""),
                "element_type": "C3D8",
                "scenario": row["case_id"],
                "analysis_type": row["analysis_type"],
                "linearity": row["linearity"],
                "z_cm_l2_rel_error": row["displacement_metric"],
                "gap_or_cdis_error": row["gap_metric"],
                "force_rel_error": row["rf_metric"],
                "energy_rel_error": row["cels_metric"],
                "active_contact_metric": row["active_contact_metric"],
                "claim_scope": "native trajectory" if row["supports_trajectory_equivalence"] == "true" else "external contact replay only",
                "source_file": row["evidence_file"],
            }
        )
    _write_csv(
        out_dir / "phase9_c3d4_c3d8_side_by_side.csv",
        [
            "case_group",
            "element_type",
            "scenario",
            "analysis_type",
            "linearity",
            "z_cm_l2_rel_error",
            "gap_or_cdis_error",
            "force_rel_error",
            "energy_rel_error",
            "active_contact_metric",
            "claim_scope",
            "source_file",
        ],
        table,
    )
    return table


def _unsupported_case(case_id: str, analysis: str, linearity: str) -> Row:
    return {
        "case_id": case_id,
        "analysis_type": analysis,
        "linearity": linearity,
        "element_type": "C3D8",
        "native_sfc_result": "false",
        "calculix_comparison": "false",
        "acceleration_evidence": "false",
        "displacement_metric": "",
        "stress_strain_metric": "",
        "gap_metric": "",
        "rf_metric": "",
        "cels_metric": "",
        "active_contact_metric": "",
        "supports_external_correctness": "false",
        "supports_trajectory_equivalence": "false",
        "supports_efficiency": "false",
        "status": "blocked_no_native_c3d8_nonlinear_backend",
        "evidence_file": "",
    }


def _write_case_markdowns(out_dir: Path, rows: list[Row]) -> None:
    case_dir = out_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        lines = [
            f"# {row['case_id']}",
            "",
            f"- Analysis: `{row['analysis_type']}`",
            f"- Linearity: `{row['linearity']}`",
            f"- Element: `{row['element_type']}`",
            f"- Native SFC result: `{row['native_sfc_result']}`",
            f"- CalculiX comparison: `{row['calculix_comparison']}`",
            f"- Acceleration evidence: `{row['acceleration_evidence']}`",
            f"- Status: `{row['status']}`",
            "",
            "## Metrics",
            "",
            f"- Displacement: `{row['displacement_metric']}`",
            f"- Stress/strain: `{row['stress_strain_metric']}`",
            f"- Gap/CDIS: `{row['gap_metric']}`",
            f"- RF/contact force: `{row['rf_metric']}`",
            f"- CELS/contact energy: `{row['cels_metric']}`",
            f"- Active contact: `{row['active_contact_metric']}`",
            "",
            "## Claim Gate",
            "",
            f"- External correctness: `{row['supports_external_correctness']}`",
            f"- Trajectory equivalence: `{row['supports_trajectory_equivalence']}`",
            f"- Efficiency: `{row['supports_efficiency']}`",
        ]
        (case_dir / f"{row['case_id']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(out_dir: Path, rows: list[Row]) -> list[Row]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_rows: list[Row] = []
    labels = [row["case_id"].replace("c3d8_", "").replace("_contact", "") for row in rows]
    external = [1.0 if row["supports_external_correctness"] == "true" else 0.0 for row in rows]
    trajectory = [1.0 if row["supports_trajectory_equivalence"] == "true" else 0.0 for row in rows]
    efficiency = [1.0 if row["supports_efficiency"] == "true" else 0.0 for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    width = 0.24
    ax.bar(x - width, external, width, label="external correctness")
    ax.bar(x, trajectory, width, label="trajectory equivalence")
    ax.bar(x + width, efficiency, width, label="efficiency")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel("claim supported")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    png = figures / "phase9_claim_gate_matrix.png"
    pdf = figures / "phase9_claim_gate_matrix.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    plot_rows.append({"plot": "phase9_claim_gate_matrix", "png": str(png.relative_to(out_dir)), "pdf": str(pdf.relative_to(out_dir)), "description": "Phase-9 claim support matrix"})

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    metric_rows = [row for row in rows if row["rf_metric"] not in {"", "z_cm_history_from_calculix"}]
    if metric_rows:
        x2 = np.arange(len(metric_rows))
        ax.bar(x2 - 0.2, [float(row["rf_metric"]) for row in metric_rows], 0.2, label="RF rel.")
        ax.bar(x2, [float(row["cels_metric"]) for row in metric_rows], 0.2, label="CELS rel.")
        ax.bar(x2 + 0.2, [float(row["gap_metric"]) if row["gap_metric"] not in {"", "z_cm_history_from_calculix"} else 0.0 for row in metric_rows], 0.2, label="gap rel./value")
        ax.set_xticks(x2)
        ax.set_xticklabels([row["case_id"].replace("c3d8_", "") for row in metric_rows], rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("metric value")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No supported metric rows", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    png = figures / "phase9_contact_error_metrics.png"
    pdf = figures / "phase9_contact_error_metrics.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    plot_rows.append({"plot": "phase9_contact_error_metrics", "png": str(png.relative_to(out_dir)), "pdf": str(pdf.relative_to(out_dir)), "description": "Phase-9 contact error metrics"})
    return plot_rows


def _claim_gates(rows: list[Row]) -> list[Row]:
    gates: list[Row] = []
    for row in rows:
        native = row["native_sfc_result"] == "true"
        external = row["calculix_comparison"] == "true"
        acceleration = row["acceleration_evidence"] == "true"
        gates.append(
            {
                "case_id": row["case_id"],
                "claim": "external_correctness",
                "allowed": _bool_text(native and external and row["supports_external_correctness"] == "true"),
                "reason": "requires native SFC result and CalculiX comparison",
            }
        )
        gates.append(
            {
                "case_id": row["case_id"],
                "claim": "trajectory_equivalence",
                "allowed": _bool_text(native and external and row["supports_trajectory_equivalence"] == "true"),
                "reason": "requires native SFC trajectory and CalculiX trajectory comparison",
            }
        )
        gates.append(
            {
                "case_id": row["case_id"],
                "claim": "efficiency",
                "allowed": _bool_text(acceleration and row["supports_efficiency"] == "true"),
                "reason": "requires acceleration/timing evidence for this case",
            }
        )
    return gates


def _write_summary(out_dir: Path, rows: list[Row], gates: list[Row], plots: list[Row], commands: list[Row]) -> None:
    lines = [
        "# Phase-9 Full Contact Validation",
        "",
        "This package organizes the evidence needed for static/dynamic, linear/nonlinear, multi-element contact claims.",
        "",
        "The current short-term multi-element scope is C3D4/TET4 plus C3D8. C3D10 is explicitly out of scope because no backend is registered.",
        "",
        "## Commands",
        "",
    ]
    for command in commands:
        lines.append(f"- `{command['command']}`")
    lines.extend(
        [
            "",
            "## Evidence Matrix",
            "",
            "| Case | Analysis | Linearity | Native SFC | CalculiX | External correctness | Trajectory equivalence | Efficiency | Status |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['analysis_type']} | {row['linearity']} | {row['native_sfc_result']} | "
            f"{row['calculix_comparison']} | {row['supports_external_correctness']} | {row['supports_trajectory_equivalence']} | "
            f"{row['supports_efficiency']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Gates",
            "",
            "| Case | Claim | Allowed | Reason |",
            "|---|---|---|---|",
        ]
    )
    for gate in gates:
        lines.append(f"| {gate['case_id']} | {gate['claim']} | {gate['allowed']} | {gate['reason']} |")
    lines.extend(["", "## Figures", ""])
    for plot in plots:
        lines.append(f"- `{plot['png']}`: {plot['description']}")
    lines.extend(
        [
            "",
            "## Additional Tables",
            "",
            "- `phase9_c3d4_c3d8_side_by_side.csv`: C3D4 and C3D8 contact cases in one comparison table.",
            "- `phase9_solver_timing.csv`: native SFC and CalculiX wall-clock timing where both commands are run by this evidence package.",
            "- `phase9_curved_nonplanar_contact_external.csv`: warped/non-planar C3D8 master-surface external replay check.",
            "",
            "## Current Conclusion",
            "",
            "The current evidence supports C3D8 linear static contact correctness, native C3D8 linear dynamic block-plane/block-block trajectory comparison, native C3D8 geometric-nonlinear static contact comparison on the contactenergy reference, native C3D8 nonlinear block-plane/block-block dynamics when the CSV error gates pass, and a warped/non-planar C3D8 external current-surface replay check. Efficiency is only allowed per case when the timing CSV has both SFC and CalculiX wall times and speedup is greater than one. For generated CalculiX dynamic steps, CDIS rows may be empty; in that case the gap metric is computed by replaying the CalculiX displacement geometry with the same current-surface dynamic-SDF query rather than by inventing unavailable CDIS output.",
        ]
    )
    (out_dir / "phase9_full_contact_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase9(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        {
            "command": "python validation/run_phase9_full_contact_validation.py"
            + (" --quick" if args.quick else "")
            + (" --skip-calculix" if args.skip_calculix else "")
            + f" --out-dir {_display_path(out_dir)}",
            "description": "Phase-9 evidence matrix",
        }
    ]
    trajectory_dir = _run_c3d8_trajectory(out_dir, quick=args.quick, skip_calculix=args.skip_calculix)
    static_row, static_stress = _linear_static_evidence(out_dir)
    dynamic_rows, dynamic_stress, linear_dynamic_commands, linear_timing = _linear_dynamic_evidence(
        trajectory_dir,
        out_dir,
        quick=args.quick,
        skip_calculix=args.skip_calculix,
    )
    commands.extend(linear_dynamic_commands)
    nonlinear_static_row, nonlinear_static_stress = _solve_native_c3d8_nonlinear_static_contactenergy(out_dir)
    nonlinear_dynamic_rows, nonlinear_dynamic_stress, nonlinear_timing = _native_nonlinear_dynamic_evidence(
        trajectory_dir,
        out_dir,
        quick=args.quick,
    )
    curved_row, curved_stress, curved_commands, curved_timing = _curved_nonplanar_external_evidence(
        out_dir,
        quick=args.quick,
        skip_calculix=args.skip_calculix,
    )
    commands.extend(curved_commands)
    rows = [
        static_row,
        nonlinear_static_row,
        *dynamic_rows,
        *nonlinear_dynamic_rows,
        curved_row,
    ]
    stress_rows = static_stress + nonlinear_static_stress + dynamic_stress + nonlinear_dynamic_stress + curved_stress
    timing_rows = linear_timing + nonlinear_timing + curved_timing
    gates = _claim_gates(rows)
    plots = _write_plots(out_dir, rows)
    _write_c3d4_c3d8_side_by_side(out_dir, rows)
    _write_case_markdowns(out_dir, rows)
    fields = [
        "case_id",
        "analysis_type",
        "linearity",
        "element_type",
        "native_sfc_result",
        "calculix_comparison",
        "acceleration_evidence",
        "displacement_metric",
        "stress_strain_metric",
        "gap_metric",
        "rf_metric",
        "cels_metric",
        "active_contact_metric",
        "supports_external_correctness",
        "supports_trajectory_equivalence",
        "supports_efficiency",
        "status",
        "evidence_file",
    ]
    _write_csv(out_dir / "phase9_full_contact_validation.csv", fields, rows)
    _write_csv(out_dir / "phase9_claim_gates.csv", ["case_id", "claim", "allowed", "reason"], gates)
    _write_csv(
        out_dir / "phase9_stress_strain_cloud.csv",
        ["case_id", "frame", "element_id", "von_mises", "engineering_strain_norm", "vtk_file"],
        stress_rows,
    )
    _write_csv(out_dir / "phase9_commands.csv", ["command", "description"], commands)
    _write_csv(out_dir / "phase9_plots.csv", ["plot", "png", "pdf", "description"], plots)
    _write_csv(
        out_dir / "phase9_solver_timing.csv",
        [
            "case_id",
            "analysis_type",
            "linearity",
            "element_type",
            "sfc_wall_time_seconds",
            "calculix_wall_time_seconds",
            "speedup_vs_calculix",
            "timing_claim_allowed",
            "notes",
        ],
        timing_rows,
    )
    _write_summary(out_dir, rows, gates, plots, commands)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-calculix", action="store_true", help="Use synthetic C3D8 trajectory data for CI smoke tests.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "phase9_full_contact_validation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_phase9(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
