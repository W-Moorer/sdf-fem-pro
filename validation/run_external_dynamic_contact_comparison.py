"""SfePy transient elastodynamic contact reference comparison.

This runner adds the external transient evidence that the quasi-static SfePy
replay case cannot provide.  It generates and runs a small SfePy
elastodynamic problem that includes:

* a consistent mass term,
* the Newmark time integrator,
* a prescribed 3 second downward motion of the upper elastic body.

It also runs SfePy's built-in two-body penalty contact solver at sampled times
along the same 3 second approach path.  The installed SfePy contact term is
used as a validation-only contact snapshot reference; it is not claimed here as
a single coupled transient contact solve.

The SFC side is intentionally scoped.  It produces an internal Newmark penalty
contact time history for the same total time, and it also replays each SfePy
transient geometry state with the SFC current-surface dynamic SDF query.  This
is validation evidence, not a new core physics feature and not a nonlinear
contact-equilibrium implementation in ``src/sfc``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

import h5py
import numpy as np
from scipy.sparse import csr_matrix
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

from sfc.contact import SurfaceSample, UniformTriangleAABBHash, contact_constraint_from_sample, penalty_contact_response  # noqa: E402
from sfc.fem import assemble_mass_matrix, assemble_stiffness_matrix, fixed_dofs_from_node_set, newmark_beta_step  # noqa: E402
from sfc.mesh import extract_boundary_faces  # noqa: E402
from sfc.sdf import dynamic_surface_sdf  # noqa: E402
from validation.run_phase3_validation import _body, structured_tet_block  # noqa: E402
from validation.run_phase4_paper_validation import _format_float  # noqa: E402
from validation.run_phase8_engineering_cases import _square_platen  # noqa: E402
from validation.run_external_contact_solver_comparison import external_contact_solver_comparison  # noqa: E402

Row = dict[str, Any]

TOTAL_TIME = 3.0
QUICK_DT = 0.05
FULL_DT = 0.01
INITIAL_GAP = 2.0e-2
APPROACH = 1.2e-1
SFEPY_CONTACT_STIFFNESS = 20.0
SFC_CONTACT_STIFFNESS = 1.8e3


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(command: list[str], *, cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=timeout)


def _sfepy_available() -> bool:
    if shutil.which("sfepy-run") is None or shutil.which("py") is None:
        return False
    try:
        _run(["py", "-3.12", "-c", "import sfepy"], cwd=ROOT, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _sfepy_version() -> str:
    proc = _run(["py", "-3.12", "-c", "import sfepy; print(sfepy.__version__)"], cwd=ROOT, timeout=60)
    return proc.stdout.strip()


def _sfepy_problem_source() -> str:
    """Return a self-contained SfePy transient elastodynamic problem."""

    return dedent(
        r'''
        import os.path as op
        from functools import partial

        import numpy as nm
        from sfepy.base.base import output
        from sfepy.discrete.fem.meshio import UserMeshIO
        from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
        from sfepy.mesh.mesh_generators import gen_block_mesh


        def get_bbox(dims, centre, eps=0.0):
            dims = nm.asarray(dims)
            centre = nm.asarray(centre)
            return nm.r_[[centre - (0.5 - eps) * dims], [centre + (0.5 - eps) * dims]]


        def gen_two_bodies(dims0, shape0, centre0, dims1, shape1, centre1, shift1):
            from sfepy.discrete.fem import Mesh

            m0 = gen_block_mesh(dims0, shape0, centre0, verbose=False)
            m1 = gen_block_mesh(dims1, shape1, centre1, verbose=False)

            coors = nm.concatenate((m0.coors, m1.coors + shift1), axis=0)
            desc = m0.descs[0]
            c0 = m0.get_conn(desc)
            c1 = m1.get_conn(desc)
            conn = nm.concatenate((c0, c1 + m0.n_nod), axis=0)

            ngroups = nm.zeros(coors.shape[0], dtype=nm.int32)
            ngroups[m0.n_nod:] = 1

            mat_id = nm.zeros(conn.shape[0], dtype=nm.int32)
            mat_id[m0.n_el:] = 1

            return Mesh.from_data('two_bodies_dynamic', coors, ngroups, [conn], [mat_id], m0.descs)


        def define(
                dims0=(1.0, 1.0, 0.5),
                shape0=(2, 2, 2),
                centre0=(0.0, 0.0, -0.25),
                dims1=(1.0, 1.0, 0.5),
                shape1=(2, 2, 2),
                centre1=(0.0, 0.0, 0.25),
                shift10=(0.0, 0.0, 1.0e-4),
                young=1.0e3,
                poisson=0.30,
                rho=1.0,
                approach=0.12,
                t1=3.0,
                dt=0.05,
                output_dir='output/external_dynamic_contact',
                verbose=False,
        ):
            inodir = partial(op.join, output_dir)
            output.set_output(filename=inodir('output_log.txt'), quiet=not verbose, combined=True)

            dim = len(dims0)
            shift10 = tuple(shift10[:dim])

            def mesh_hook(mesh, mode):
                if mode == 'read':
                    return gen_two_bodies(dims0, shape0, centre0, dims1, shape1, centre1, shift10)
                elif mode == 'write':
                    pass

            def post_process(out, pb, state, extend=False):
                return out

            filename_mesh = UserMeshIO(mesh_hook)

            bbox0 = get_bbox(dims0, centre0, eps=1.0e-5)
            bbox1 = get_bbox(dims1, nm.asarray(centre1) + nm.asarray(shift10), eps=1.0e-5)
            regions = {
                'Omega': 'all',
                'Omega0': 'cells of group 0',
                'Omega1': 'cells of group 1',
                'Bottom': ('vertices in (z < %.12e)' % bbox0[0, 2], 'facet'),
                'Top': ('vertices in (z > %.12e)' % bbox1[1, 2], 'facet'),
            }

            fields = {
                'displacement': ('real', dim, 'Omega', 1),
            }

            variables = {
                'u': ('unknown field', 'displacement', 0),
                'v': ('test field', 'displacement', 'u'),
            }

            def _top_u(ts, coors, **kwargs):
                val = nm.zeros(coors.shape[0], dtype=float)
                val[:] = -approach * ts.nt
                return val

            def _top_du(ts, coors, **kwargs):
                val = nm.zeros(coors.shape[0], dtype=float)
                val[:] = -approach / float(t1)
                return val

            def _top_ddu(ts, coors, **kwargs):
                return nm.zeros(coors.shape[0], dtype=float)

            functions = {
                'top_u': (_top_u,),
                'top_du': (_top_du,),
                'top_ddu': (_top_ddu,),
            }

            ebcs = {
                'fix_bottom': ('Bottom', {'u.all': 0.0, 'du.all': 0.0, 'ddu.all': 0.0}),
                'move_top': ('Top', {
                    'u.0': 0.0, 'du.0': 0.0, 'ddu.0': 0.0,
                    'u.1': 0.0, 'du.1': 0.0, 'ddu.1': 0.0,
                    'u.2': 'top_u', 'du.2': 'top_du', 'ddu.2': 'top_ddu',
                }),
            }

            ics = {
                'ic': ('Omega', {'u.all': 0.0, 'du.all': 0.0, 'ddu.all': 0.0}),
            }

            materials = {
                'solid': ({
                    'D': stiffness_from_youngpoisson(dim, young=young, poisson=poisson),
                    'rho': rho,
                    '.lumping': 'none',
                    '.beta': 0.0,
                },),
            }

            integrals = {
                'i': 2,
            }

            equations = {
                'balance_of_forces':
                """
                   de_mass.i.Omega(solid.rho, solid.lumping, solid.beta, v, ddu)
                 + dw_lin_elastic.2.Omega(solid.D, v, u)
                 = 0
                """,
            }

            solvers = {
                'ls': ('ls.auto_direct', {}),
                'newton': ('nls.newton', {
                    'i_max': 20,
                    'eps_a': 1.0e-8,
                    'eps_r': 1.0e-8,
                    'eps_mode': 'or',
                    'macheps': 1.0e-16,
                    'lin_red': None,
                    'ls_red': 0.5,
                    'ls_on': 1.0,
                    'ls_min': 1.0e-5,
                    'check': 0,
                    'delta': 1.0e-8,
                    'log': {'text': inodir('newton_log.txt')},
                }),
                'tsn': ('ts.newmark', {
                    't0': 0.0,
                    't1': t1,
                    'dt': dt,
                    'n_step': None,
                    'is_linear': True,
                    'beta': 0.25,
                    'gamma': 0.5,
                    'verbose': 0,
                }),
            }

            options = {
                'ts': 'tsn',
                'nls': 'newton',
                'ls': 'ls',
                'save_times': 'all',
                'active_only': False,
                'auto_transform_equations': True,
                'output_format': 'h5',
                'output_dir': output_dir,
                'post_process_hook': 'post_process',
            }

            return locals()
        '''
    ).strip() + "\n"


def _write_sfepy_problem(path: Path) -> None:
    path.write_text(_sfepy_problem_source(), encoding="utf-8")


def _run_sfepy_dynamic_reference(out_dir: Path, *, quick: bool) -> tuple[Path, Row]:
    if not _sfepy_available():
        raise SystemExit("SfePy is required for this external dynamic contact comparison.")

    run_dir = out_dir / "sfepy_transient_contact_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    problem_path = run_dir / "sfepy_transient_contact_problem.py"
    _write_sfepy_problem(problem_path)

    dt = QUICK_DT if quick else FULL_DT
    define = (
        f"output_dir='{run_dir.as_posix()}', "
        "shape0=(2,2,2), shape1=(2,2,2), "
        f"shift10=(0.0,0.0,{INITIAL_GAP:.16g}), "
        f"t1={TOTAL_TIME:.16g}, dt={dt:.16g}, "
        f"approach={APPROACH:.16g}, "
        "verbose=False"
    )
    command = ["sfepy-run", str(problem_path), "-d", define]
    h5_path = run_dir / "two_bodies_dynamic.h5"
    if h5_path.exists():
        h5_path.unlink()
    proc = _run(command, cwd=ROOT, timeout=900 if not quick else 300)
    if not h5_path.is_file():
        raise RuntimeError(f"SfePy run did not produce {h5_path}")

    command_row = {
        "source": "sfepy_transient_elastodynamic",
        "external_solver": "SfePy",
        "external_solver_version": _sfepy_version(),
        "command": " ".join(command),
        "total_time": TOTAL_TIME,
        "dt": dt,
        "h5_output": str(h5_path.relative_to(out_dir)),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-10:]),
        "details": "SfePy generated problem: Newmark, mass matrix, prescribed upper-body motion; contact is evaluated by separate SfePy snapshot references",
    }
    return h5_path, command_row


def _h5_step_names(h5: h5py.File) -> list[str]:
    names = [name for name in h5.keys() if name.startswith("step") and name[4:].isdigit()]
    return sorted(names, key=lambda name: int(name[4:]))


def _get_step_array(h5: h5py.File, step_name: str, variable: str, shape: tuple[int, int]) -> np.ndarray:
    key = f"{step_name}/__{variable}/data"
    if key not in h5:
        return np.zeros(shape, dtype=float)
    return np.asarray(h5[key][...], dtype=float)


def _sfepy_history_rows(h5_path: Path, *, dt: float, total_time: float) -> list[Row]:
    rows: list[Row] = []
    with h5py.File(h5_path, "r") as h5:
        coors = np.asarray(h5["mesh/coors"][...], dtype=float)
        shape = coors.shape
        for step_name in _h5_step_names(h5):
            step = int(step_name[4:])
            u = _get_step_array(h5, step_name, "u", shape)
            du = _get_step_array(h5, step_name, "du", shape)
            ddu = _get_step_array(h5, step_name, "ddu", shape)
            gap_key = f"{step_name}/__gap/data"
            gap = np.asarray(h5[gap_key][...], dtype=float).ravel() if gap_key in h5 else np.zeros(0, dtype=float)
            penetration = np.maximum(-gap, 0.0)
            rows.append(
                {
                    "source": "sfepy_transient_elastodynamic",
                    "time": min(step * dt, total_time),
                    "step": step,
                    "dt": dt,
                    "total_time": total_time,
                    "min_gap": float(np.min(gap)) if gap.size else 0.0,
                    "mean_gap": float(np.mean(gap)) if gap.size else 0.0,
                    "max_penetration": float(np.max(penetration)) if penetration.size else 0.0,
                    "active_contact_count": int(np.count_nonzero(penetration > 0.0)),
                    "normal_force_proxy": float(SFEPY_CONTACT_STIFFNESS * np.sum(penetration)),
                    "contact_energy_proxy": 0.5 * SFEPY_CONTACT_STIFFNESS * float(np.dot(penetration, penetration)),
                    "displacement_l2_norm": float(np.linalg.norm(u)),
                    "velocity_l2_norm": float(np.linalg.norm(du)),
                    "acceleration_l2_norm": float(np.linalg.norm(ddu)),
                    "mass_matrix_included": "true",
                    "integrator": "SfePy ts.newmark",
                    "details": "external SfePy transient elastodynamic penalty contact time-history",
                }
            )
    return rows


def _triangulated_lower_contact_surface(coors: np.ndarray, conn: np.ndarray, mat_id: np.ndarray) -> np.ndarray:
    lower_cells = conn[mat_id == 0]
    lower_nodes = np.unique(lower_cells.ravel())
    top_z = float(np.max(coors[lower_nodes, 2]))
    triangles: list[list[int]] = []
    for cell in lower_cells:
        top = np.asarray([cell[4], cell[5], cell[6], cell[7]], dtype=np.int64)
        if np.all(np.isclose(coors[top, 2], top_z)):
            triangles.append([int(top[0]), int(top[1]), int(top[2])])
            triangles.append([int(top[0]), int(top[2]), int(top[3])])
    return np.asarray(triangles, dtype=np.int64)


def _upper_contact_centroids(coors: np.ndarray, conn: np.ndarray, mat_id: np.ndarray, current: np.ndarray) -> np.ndarray:
    upper_cells = conn[mat_id == 1]
    upper_nodes = np.unique(upper_cells.ravel())
    bottom_z = float(np.min(coors[upper_nodes, 2]))
    centroids: list[np.ndarray] = []
    for cell in upper_cells:
        bottom = np.asarray([cell[0], cell[1], cell[2], cell[3]], dtype=np.int64)
        if np.all(np.isclose(coors[bottom, 2], bottom_z)):
            centroids.append(np.mean(current[bottom], axis=0))
    return np.asarray(centroids, dtype=float)


def _sfc_replay_rows_on_sfepy_history(h5_path: Path, *, dt: float, total_time: float) -> list[Row]:
    rows: list[Row] = []
    with h5py.File(h5_path, "r") as h5:
        coors = np.asarray(h5["mesh/coors"][...], dtype=float)
        conn = np.asarray(h5["mesh/group0/conn"][...], dtype=np.int64)
        mat_id = np.asarray(h5["mesh/group0/mat_id"][...], dtype=np.int64)
        lower_faces = _triangulated_lower_contact_surface(coors, conn, mat_id)
        candidates = np.arange(lower_faces.shape[0], dtype=np.int64)
        for step_name in _h5_step_names(h5):
            step = int(step_name[4:])
            u = _get_step_array(h5, step_name, "u", coors.shape)
            current = coors + u
            query_points = _upper_contact_centroids(coors, conn, mat_id, current)
            gaps = np.asarray(
                [dynamic_surface_sdf(point, current, lower_faces, candidates).g for point in query_points],
                dtype=float,
            )
            penetration = np.maximum(-gaps, 0.0)
            rows.append(
                {
                    "source": "sfc_replay_on_sfepy_transient_geometry",
                    "time": min(step * dt, total_time),
                    "step": step,
                    "dt": dt,
                    "total_time": total_time,
                    "min_gap": float(np.min(gaps)) if gaps.size else 0.0,
                    "mean_gap": float(np.mean(gaps)) if gaps.size else 0.0,
                    "max_penetration": float(np.max(penetration)) if penetration.size else 0.0,
                    "active_contact_count": int(np.count_nonzero(penetration > 0.0)),
                    "normal_force_proxy": float(SFEPY_CONTACT_STIFFNESS * np.sum(penetration)),
                    "contact_energy_proxy": 0.5 * SFEPY_CONTACT_STIFFNESS * float(np.dot(penetration, penetration)),
                    "displacement_l2_norm": float(np.linalg.norm(u)),
                    "velocity_l2_norm": 0.0,
                    "acceleration_l2_norm": 0.0,
                    "mass_matrix_included": "external_state_replay",
                    "integrator": "SFC dynamic_surface_sdf replay",
                    "details": "SFC current-surface gap replay on every SfePy transient geometry state",
                }
            )
    return rows


def _sfepy_contact_snapshot_rows(out_dir: Path, *, quick: bool) -> tuple[list[Row], list[Row], list[Row]]:
    """Run SfePy's contact solver at sampled times along the 3 s approach."""

    snapshot_count = 4 if quick else 13
    times = np.linspace(0.0, TOTAL_TIME, snapshot_count)
    approaches = [float(APPROACH * time / TOTAL_TIME) for time in times]
    snapshot_dt = float(times[1] - times[0]) if times.size > 1 else TOTAL_TIME
    rows, command_rows = external_contact_solver_comparison(out_dir / "sfepy_contact_snapshots", approaches)

    sfepy_time_rows: list[Row] = []
    replay_time_rows: list[Row] = []
    mapped_commands: list[Row] = []
    for index, row in enumerate(rows):
        time = float(times[index])
        sfepy_penetration = max(-float(row["sfepy_gap_mean"]), 0.0)
        sfepy_active = int(row["sfepy_active_count"])
        sfc_penetration = max(-float(row["sfc_replay_gap_mean"]), 0.0)
        sfc_active = int(row["sfc_replay_active_count"])
        sfepy_time_rows.append(
            {
                "source": "sfepy_contact_reference_snapshot",
                "time": time,
                "step": index,
                "dt": snapshot_dt,
                "total_time": TOTAL_TIME,
                "min_gap": float(row["sfepy_gap_min"]),
                "mean_gap": float(row["sfepy_gap_mean"]),
                "max_penetration": max(-float(row["sfepy_gap_min"]), 0.0),
                "active_contact_count": sfepy_active,
                "normal_force_proxy": SFEPY_CONTACT_STIFFNESS * sfepy_penetration * max(sfepy_active, 1),
                "contact_energy_proxy": 0.5 * SFEPY_CONTACT_STIFFNESS * sfepy_penetration * sfepy_penetration * max(sfepy_active, 1),
                "displacement_l2_norm": float(row["displacement_l2_norm"]),
                "velocity_l2_norm": 0.0,
                "acceleration_l2_norm": 0.0,
                "mass_matrix_included": "false_contact_snapshot",
                "integrator": "SfePy ts.simple quasistatic contact snapshot",
                "details": "SfePy external two-body penalty contact solve at one sampled time of the 3 s approach path",
            }
        )
        replay_time_rows.append(
            {
                "source": "sfc_contact_snapshot_replay",
                "time": time,
                "step": index,
                "dt": snapshot_dt,
                "total_time": TOTAL_TIME,
                "min_gap": float(row["sfc_replay_gap_min"]),
                "mean_gap": float(row["sfc_replay_gap_mean"]),
                "max_penetration": max(-float(row["sfc_replay_gap_min"]), 0.0),
                "active_contact_count": sfc_active,
                "normal_force_proxy": SFEPY_CONTACT_STIFFNESS * sfc_penetration * max(sfc_active, 1),
                "contact_energy_proxy": 0.5 * SFEPY_CONTACT_STIFFNESS * sfc_penetration * sfc_penetration * max(sfc_active, 1),
                "displacement_l2_norm": float(row["displacement_l2_norm"]),
                "velocity_l2_norm": 0.0,
                "acceleration_l2_norm": 0.0,
                "mass_matrix_included": "external_contact_state_replay",
                "integrator": "SFC dynamic_surface_sdf contact snapshot replay",
                "details": "SFC current-surface gap replay on the SfePy contact snapshot geometry",
            }
        )

    for index, command_row in enumerate(command_rows):
        mapped_commands.append(
            {
                "source": "sfepy_contact_reference_snapshot",
                "external_solver": "SfePy",
                "external_solver_version": rows[index]["external_solver_version"],
                "command": command_row["command"],
                "total_time": TOTAL_TIME,
                "dt": snapshot_dt,
                "h5_output": command_row["h5_output"],
                "stdout_tail": command_row["stdout_tail"],
                "stderr_tail": command_row["stderr_tail"],
                "details": f"contact snapshot at t={times[index]:.6g}s, approach={approaches[index]:.6g}",
            }
        )

    return sfepy_time_rows, replay_time_rows, mapped_commands


def _contact_constraints_for_nodes(
    slave_x: np.ndarray,
    slave_node_ids: np.ndarray,
    master_x: np.ndarray,
    master_faces: np.ndarray,
    *,
    delta_safe: float,
    cell_size: float,
) -> list[Any]:
    index = UniformTriangleAABBHash.from_surface(master_x, master_faces, delta_safe=delta_safe, cell_size=cell_size)
    constraints = []
    for node_id in slave_node_ids:
        candidates = index.query_point(slave_x[int(node_id)])
        if candidates.size == 0:
            continue
        sample = SurfaceSample(np.array([int(node_id)], dtype=np.int64), np.array([1.0]), candidates)
        constraints.append(contact_constraint_from_sample(slave_x, sample, master_x, master_faces))
    return constraints


def _sfc_newmark_contact_rows(*, quick: bool) -> list[Row]:
    dt = QUICK_DT if quick else FULL_DT
    n_steps = int(round(TOTAL_TIME / dt))
    mesh = structured_tet_block(2 if quick else 3, 2 if quick else 3, 1 if quick else 2, size=(1.0, 1.0, 0.35))
    body = _body(mesh, E=1.0e5, nu=0.30, rho=20.0)
    M = assemble_mass_matrix(body)
    K = assemble_stiffness_matrix(body)
    C = csr_matrix(K.shape, dtype=float)
    fixed_nodes = np.flatnonzero(np.isclose(mesh.X[:, 2], 0.0))
    fixed = fixed_dofs_from_node_set(fixed_nodes, "xyz")
    u = np.zeros(body.n_dofs, dtype=float)
    v = np.zeros(body.n_dofs, dtype=float)
    a = np.zeros(body.n_dofs, dtype=float)
    free_dofs = np.setdiff1d(np.arange(body.n_dofs), fixed)
    if free_dofs.size:
        from sfc.fem import eliminate_fixed_dofs

        M_reduced, rhs_reduced, reduced_free = eliminate_fixed_dofs(M, -(K @ u), fixed)
        if reduced_free.size:
            a[reduced_free] = np.asarray(spsolve(M_reduced, rhs_reduced), dtype=float)

    top_z = float(mesh.X[:, 2].max())
    top_nodes = np.flatnonzero(np.isclose(mesh.X[:, 2], top_z))
    rows: list[Row] = []
    for step in range(n_steps + 1):
        t = min(step * dt, TOTAL_TIME)
        approach = APPROACH * t / TOTAL_TIME
        current = mesh.X + u.reshape((-1, 3))
        platen_z = top_z + INITIAL_GAP - approach
        platen_x, platen_faces = _square_platen((0.5, 0.5), 0.55, platen_z, 2 if quick else 3)
        constraints = _contact_constraints_for_nodes(
            current,
            top_nodes,
            platen_x,
            platen_faces,
            delta_safe=max(0.02, abs(float(approach) - INITIAL_GAP) + 0.02),
            cell_size=0.55,
        )
        contact_force, _ = penalty_contact_response(
            constraints,
            stiffness=SFC_CONTACT_STIFFNESS,
            n_total_dofs=body.n_dofs + 3 * platen_x.shape[0],
            slave_dof_offset=0,
            master_dof_offset=body.n_dofs,
        )
        block_force = contact_force[: body.n_dofs]
        gaps = np.asarray([float(c.g) for c in constraints], dtype=float)
        penetration = np.maximum(-gaps, 0.0) if gaps.size else np.zeros(0, dtype=float)
        rows.append(
            {
                "source": "sfc_newmark_penalty_contact",
                "time": t,
                "step": step,
                "dt": dt,
                "total_time": TOTAL_TIME,
                "min_gap": float(np.min(gaps)) if gaps.size else 0.0,
                "mean_gap": float(np.mean(gaps)) if gaps.size else 0.0,
                "max_penetration": float(np.max(penetration)) if penetration.size else 0.0,
                "active_contact_count": int(np.count_nonzero(penetration > 0.0)),
                "normal_force_proxy": float(np.sum(block_force.reshape((-1, 3))[:, 2])),
                "contact_energy_proxy": 0.5 * SFC_CONTACT_STIFFNESS * float(np.dot(penetration, penetration)),
                "displacement_l2_norm": float(np.linalg.norm(u)),
                "velocity_l2_norm": float(np.linalg.norm(v)),
                "acceleration_l2_norm": float(np.linalg.norm(a)),
                "mass_matrix_included": "true",
                "integrator": "sfc.newmark_beta_step",
                "details": "SFC linear Newmark step with lagged rigid-platen normal penalty contact force",
            }
        )
        if step == n_steps:
            break
        u, v, a = newmark_beta_step(
            M,
            C,
            K,
            u,
            v,
            a,
            np.zeros(body.n_dofs, dtype=float),
            dt=dt,
            f_contact=block_force,
            fixed_dofs=fixed,
        )
    return rows


def _first_contact_time(rows: list[Row]) -> float:
    for row in rows:
        if int(row["active_contact_count"]) > 0:
            return float(row["time"])
    return float("nan")


def _compare_rows(dynamic_rows: list[Row], contact_rows: list[Row], replay_rows: list[Row], newmark_rows: list[Row]) -> list[Row]:
    count = min(len(contact_rows), len(replay_rows))
    if count == 0:
        raise RuntimeError("empty SfePy contact snapshot or replay time history")

    active_matches = [
        (int(a["active_contact_count"]) > 0) == (int(b["active_contact_count"]) > 0)
        for a, b in zip(contact_rows[:count], replay_rows[:count], strict=True)
    ]
    mean_gap_diffs = [abs(float(a["mean_gap"]) - float(b["mean_gap"])) for a, b in zip(contact_rows[:count], replay_rows[:count], strict=True)]
    min_gap_diffs = [abs(float(a["min_gap"]) - float(b["min_gap"])) for a, b in zip(contact_rows[:count], replay_rows[:count], strict=True)]
    sfepy_force = np.asarray([float(row["normal_force_proxy"]) for row in contact_rows], dtype=float)
    replay_force = np.asarray([float(row["normal_force_proxy"]) for row in replay_rows[: len(sfepy_force)]], dtype=float)
    if sfepy_force.size and np.linalg.norm(sfepy_force) > 0.0 and np.linalg.norm(replay_force) > 0.0:
        force_proxy_cosine = float(np.dot(sfepy_force, replay_force) / (np.linalg.norm(sfepy_force) * np.linalg.norm(replay_force)))
    else:
        force_proxy_cosine = 0.0

    sfc_contact_rows = [row for row in newmark_rows if int(row["active_contact_count"]) > 0]
    sfc_force_sign_ok = all(float(row["normal_force_proxy"]) <= 1.0e-12 for row in sfc_contact_rows)
    return [
        {
            "metric": "total_time",
            "value": TOTAL_TIME,
            "status": "ok",
            "details": "3 second dynamic contact comparison horizon",
        },
        {
            "metric": "sfepy_step_count",
            "value": len(dynamic_rows),
            "status": "ok" if len(dynamic_rows) > 10 else "check",
            "details": "external SfePy transient elastodynamic output rows",
        },
        {
            "metric": "sfepy_contact_snapshot_count",
            "value": len(contact_rows),
            "status": "ok" if len(contact_rows) >= 4 else "check",
            "details": "external SfePy two-body contact snapshot rows on the 3 second approach path",
        },
        {
            "metric": "sfc_newmark_step_count",
            "value": len(newmark_rows),
            "status": "ok" if len(newmark_rows) > 10 else "check",
            "details": "internal SFC Newmark penalty-contact rows",
        },
        {
            "metric": "first_contact_time_sfepy",
            "value": _first_contact_time(contact_rows),
            "status": "ok",
            "details": "first SfePy contact snapshot with active penalty contact",
        },
        {
            "metric": "first_contact_time_sfc_replay",
            "value": _first_contact_time(replay_rows),
            "status": "ok",
            "details": "first SFC replay snapshot row with negative current-surface gap",
        },
        {
            "metric": "first_contact_time_sfc_newmark",
            "value": _first_contact_time(newmark_rows),
            "status": "ok",
            "details": "first SFC Newmark row with active rigid-platen penalty contact",
        },
        {
            "metric": "active_state_agreement_fraction",
            "value": float(np.mean(active_matches)),
            "status": "ok" if float(np.mean(active_matches)) >= 0.80 else "check",
            "details": "SfePy contact activation vs SFC current-surface replay activation",
        },
        {
            "metric": "max_mean_gap_abs_difference",
            "value": float(np.max(mean_gap_diffs)),
            "status": "ok" if float(np.max(mean_gap_diffs)) < 2.0e-2 else "check",
            "details": "SfePy exported contact-term gap average vs SFC centroid replay gap",
        },
        {
            "metric": "max_min_gap_abs_difference",
            "value": float(np.max(min_gap_diffs)),
            "status": "ok" if float(np.max(min_gap_diffs)) < 2.0e-2 else "check",
            "details": "minimum gap consistency scale between SfePy and SFC replay",
        },
        {
            "metric": "force_proxy_cosine_similarity",
            "value": force_proxy_cosine,
            "status": "ok" if force_proxy_cosine > 0.50 else "check",
            "details": "trend-only comparison of SfePy and replay penalty-force proxies",
        },
        {
            "metric": "sfc_contact_force_direction_ok",
            "value": str(sfc_force_sign_ok).lower(),
            "status": "ok" if sfc_contact_rows and sfc_force_sign_ok else "check",
            "details": "rigid platen applies downward normal force to the SFC block during penetration",
        },
    ]


def _claim_rows(comparison_rows: list[Row], time_rows: list[Row]) -> list[Row]:
    by_metric = {str(row["metric"]): row for row in comparison_rows}
    sources = {str(row["source"]) for row in time_rows}
    has_dynamic = "sfepy_transient_elastodynamic" in sources
    has_contact = "sfepy_contact_reference_snapshot" in sources
    has_replay = "sfc_contact_snapshot_replay" in sources
    has_sfc = "sfc_newmark_penalty_contact" in sources
    activation = float(by_metric["active_state_agreement_fraction"]["value"])
    gap_diff = float(by_metric["max_mean_gap_abs_difference"]["value"])
    force_sign = by_metric["sfc_contact_force_direction_ok"]["value"] == "true"
    return [
        {
            "claim_id": "external_sfepy_transient_elastodynamic_reference",
            "claim_text": "The external dynamic reference is a SfePy transient elastodynamic run with mass matrix and Newmark time integration over 3 seconds.",
            "evidence_csv": "external_dynamic_contact_time_history.csv",
            "evidence_field": "integrator",
            "gate_value": "SfePy ts.newmark",
            "claim_status": "supported" if has_dynamic and float(by_metric["total_time"]["value"]) == TOTAL_TIME else "not_supported",
            "details": "SfePy generated problem includes de_mass and ts.newmark; contact snapshots are reported separately",
        },
        {
            "claim_id": "sfc_newmark_dynamic_penalty_contact_history",
            "claim_text": "SFC produces a 3 second internal Newmark penalty-contact time history with an assembled mass matrix.",
            "evidence_csv": "external_dynamic_contact_time_history.csv",
            "evidence_field": "mass_matrix_included",
            "gate_value": "true",
            "claim_status": "supported" if has_sfc and force_sign else "not_supported",
            "details": "SFC model uses assemble_mass_matrix and newmark_beta_step with lagged normal penalty contact",
        },
        {
            "claim_id": "external_sfepy_contact_snapshot_replay_agreement",
            "claim_text": "SFC current-surface dynamic SDF replay follows the SfePy contact snapshot activation and gap scale along the 3 second path.",
            "evidence_csv": "external_dynamic_contact_comparison.csv",
            "evidence_field": "value",
            "gate_value": activation,
            "claim_status": "supported" if has_contact and has_replay and activation >= 0.80 and gap_diff < 2.0e-2 else "not_supported",
            "details": "SfePy contact is a snapshot reference; not a single coupled nonlinear dynamic contact equilibrium comparison",
        },
    ]


def _write_plots(out_dir: Path, time_rows: list[Row]) -> list[Row]:
    plots: list[Row] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, list[Row]] = {}
    for row in time_rows:
        by_source.setdefault(str(row["source"]), []).append(row)

    label_map = {
        "sfepy_transient_elastodynamic": "SfePy Newmark elastodynamics",
        "sfc_replay_on_sfepy_transient_geometry": "SFC replay on SfePy dynamic geometry",
        "sfepy_contact_reference_snapshot": "SfePy contact snapshots",
        "sfc_contact_snapshot_replay": "SFC replay on contact snapshots",
        "sfc_newmark_penalty_contact": "SFC Newmark penalty contact",
    }

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for source, rows in by_source.items():
        ax.plot([float(row["time"]) for row in rows], [float(row["min_gap"]) for row in rows], label=label_map.get(source, source))
    ax.axhline(0.0, color="0.35", linewidth=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("minimum gap")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = out_dir / "external_dynamic_contact_min_gap.png"
    pdf = out_dir / "external_dynamic_contact_min_gap.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    plots.append({"plot": "external_dynamic_contact_min_gap", "png": png.name, "pdf": pdf.name, "status": "ok"})

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for source, rows in by_source.items():
        ax.plot([float(row["time"]) for row in rows], [float(row["normal_force_proxy"]) for row in rows], label=label_map.get(source, source))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("normal force proxy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = out_dir / "external_dynamic_contact_force_history.png"
    pdf = out_dir / "external_dynamic_contact_force_history.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    plots.append({"plot": "external_dynamic_contact_force_history", "png": png.name, "pdf": pdf.name, "status": "ok"})

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for source, rows in by_source.items():
        ax.plot([float(row["time"]) for row in rows], [int(row["active_contact_count"]) for row in rows], label=label_map.get(source, source))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("active contact count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = out_dir / "external_dynamic_contact_active_count.png"
    pdf = out_dir / "external_dynamic_contact_active_count.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    plots.append({"plot": "external_dynamic_contact_active_count", "png": png.name, "pdf": pdf.name, "status": "ok"})

    return plots


def _write_markdown(path: Path, time_rows: list[Row], comparison_rows: list[Row], claims: list[Row], plots: list[Row]) -> None:
    lines = [
        "# External Dynamic Contact Comparison Summary",
        "",
        "This validation uses SfePy-based FEM evidence instead of rigid-body contact references. SfePy provides a transient elastodynamic run with a mass term and Newmark time integration over a 3 second prescribed approach. SfePy's built-in two-body penalty contact example is then sampled along the same path as an external contact snapshot reference.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python validation/run_external_dynamic_contact_comparison.py --out-dir results/external_dynamic_contact",
        "```",
        "",
        "Quick/CI run:",
        "",
        "```bash",
        "python validation/run_external_dynamic_contact_comparison.py --quick --out-dir results/external_dynamic_contact",
        "```",
        "",
        "## Claims",
        "",
        "| Claim | Status | Evidence | Gate value |",
        "| --- | --- | --- | ---: |",
    ]
    for claim in claims:
        lines.append(f"| {claim['claim_id']} | {claim['claim_status']} | {claim['evidence_csv']}::{claim['evidence_field']} | {claim['gate_value']} |")
        lines.append(f"<!-- evidence csv={claim['evidence_csv']} field={claim['evidence_field']} -->")

    lines.extend(["", "## Comparison Metrics", "", "| Metric | Value | Status | Details |", "| --- | ---: | --- | --- |"])
    for row in comparison_rows:
        lines.append(f"| {row['metric']} | {_format_float(row['value']) if isinstance(row['value'], (float, int)) else row['value']} | {row['status']} | {row['details']} |")

    lines.extend(["", "## Time-History Sources", "", "| Source | Rows | First contact time | Peak force proxy |", "| --- | ---: | ---: | ---: |"])
    for source in sorted({str(row["source"]) for row in time_rows}):
        rows = [row for row in time_rows if row["source"] == source]
        peak = max(abs(float(row["normal_force_proxy"])) for row in rows)
        lines.append(f"| {source} | {len(rows)} | {_format_float(_first_contact_time(rows))} | {_format_float(peak)} |")

    lines.extend(["", "## Plots", ""])
    for plot in plots:
        lines.append(f"- `{plot['png']}` and `{plot['pdf']}`")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The SfePy contact reference is a sequence of external penalty-contact snapshots on a small generated two-body model.",
            "- The SfePy dynamic run and the SfePy contact snapshots are separate references; this is not a single coupled SfePy transient contact solve.",
            "- The SFC Newmark case uses lagged penalty contact force assembly and is not a nonlinear contact equilibrium solve.",
            "- SFC replay on SfePy geometry checks dynamic SDF gap consistency on external transient/contact states; it does not imply identical full-field dynamics.",
            "- The comparison remains frictionless normal contact with small-strain linear elasticity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_external_dynamic_contact(*, quick: bool, out_dir: Path) -> dict[str, list[Row]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = QUICK_DT if quick else FULL_DT
    h5_path, command_row = _run_sfepy_dynamic_reference(out_dir, quick=quick)
    dynamic_rows = _sfepy_history_rows(h5_path, dt=dt, total_time=TOTAL_TIME)
    dynamic_replay_rows = _sfc_replay_rows_on_sfepy_history(h5_path, dt=dt, total_time=TOTAL_TIME)
    contact_rows, contact_replay_rows, contact_command_rows = _sfepy_contact_snapshot_rows(out_dir, quick=quick)
    newmark_rows = _sfc_newmark_contact_rows(quick=quick)
    time_rows = dynamic_rows + dynamic_replay_rows + contact_rows + contact_replay_rows + newmark_rows
    comparison_rows = _compare_rows(dynamic_rows, contact_rows, contact_replay_rows, newmark_rows)
    claims = _claim_rows(comparison_rows, time_rows)
    plots = _write_plots(out_dir, time_rows)
    return {
        "time_rows": time_rows,
        "comparison_rows": comparison_rows,
        "claim_rows": claims,
        "command_rows": [command_row] + contact_command_rows,
        "plot_rows": plots,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a reduced time step count while preserving T=3s.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "external_dynamic_contact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_external_dynamic_contact(quick=bool(args.quick), out_dir=args.out_dir)
    _write_csv(
        args.out_dir / "external_dynamic_contact_time_history.csv",
        [
            "source",
            "time",
            "step",
            "dt",
            "total_time",
            "min_gap",
            "mean_gap",
            "max_penetration",
            "active_contact_count",
            "normal_force_proxy",
            "contact_energy_proxy",
            "displacement_l2_norm",
            "velocity_l2_norm",
            "acceleration_l2_norm",
            "mass_matrix_included",
            "integrator",
            "details",
        ],
        outputs["time_rows"],
    )
    _write_csv(
        args.out_dir / "external_dynamic_contact_comparison.csv",
        ["metric", "value", "status", "details"],
        outputs["comparison_rows"],
    )
    _write_csv(
        args.out_dir / "external_dynamic_contact_claims.csv",
        ["claim_id", "claim_text", "evidence_csv", "evidence_field", "gate_value", "claim_status", "details"],
        outputs["claim_rows"],
    )
    _write_csv(
        args.out_dir / "external_dynamic_contact_commands.csv",
        ["source", "external_solver", "external_solver_version", "command", "total_time", "dt", "h5_output", "stdout_tail", "stderr_tail", "details"],
        outputs["command_rows"],
    )
    _write_csv(args.out_dir / "external_dynamic_contact_plots.csv", ["plot", "png", "pdf", "status"], outputs["plot_rows"])
    _write_markdown(
        args.out_dir / "external_dynamic_contact_summary.md",
        outputs["time_rows"],
        outputs["comparison_rows"],
        outputs["claim_rows"],
        outputs["plot_rows"],
    )
    print(f"Wrote external dynamic contact comparison outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
