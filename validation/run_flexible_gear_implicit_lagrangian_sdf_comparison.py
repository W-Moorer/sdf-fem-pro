"""Cropped gear implicit TET4 + Lagrangian-SDF contact alignment runner.

This validation-only script builds a small contact patch from the supplied
flexible gear Abaqus input deck, then solves the cropped two-body model with
SFC's internal TET4 HHT dynamics and Lagrangian-SDF surface-contact tangent.

The Abaqus deck is used only as mesh/model input and as an optional external
reference deck writer.  The SFC solve does not read ODB, XLSX, or Abaqus
assembled matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, cg, splu
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact.hard_contact import (  # noqa: E402
    hard_contact_gap_jacobian_from_samples,
    solve_linear_hard_contact_with_dirichlet,
    solve_linear_hard_contact_with_dirichlet_sparse,
)
from sfc.contact.constraint_region import (  # noqa: E402
    active_constraint_region_tangent_data_from_arrays as _core_active_constraint_region_tangent_data_from_arrays,
    aggregate_contact_sample_arrays as _core_aggregate_contact_sample_arrays,
    contact_active_region_continuity_metrics_from_arrays as _core_contact_active_region_continuity_metrics_from_arrays,
    constraint_region_gap_jacobian_sparse_from_arrays as _core_constraint_region_gap_jacobian_sparse_from_arrays,
    constraint_region_pressure_tangent_scales_from_arrays as _core_constraint_region_pressure_tangent_scales_from_arrays,
    constraint_region_reduced_gap_jacobian_sparse_from_arrays as _core_constraint_region_reduced_gap_jacobian_sparse_from_arrays,
    constraint_region_tangent_metrics_from_arrays as _core_constraint_region_tangent_metrics_from_arrays,
    contact_path_tracking_metrics_from_arrays as _core_contact_path_tracking_metrics_from_arrays,
    contact_region_integral_metrics_from_arrays as _core_contact_region_integral_metrics_from_arrays,
    secondary_node_pressure_recovery_from_regions as _core_secondary_node_pressure_recovery_from_regions,
)
from sfc.contact.lagrangian_surface_contact import LagrangianSDFSurfaceContactGeometry  # noqa: E402
from sfc.contact.tracking_state import (  # noqa: E402
    accepted_contact_response_can_reuse as _core_accepted_contact_response_can_reuse,
    commit_accepted_contact_tracking_from_sample_arrays as _core_commit_accepted_contact_tracking_from_sample_arrays,
    restore_contact_tracking_state as _core_restore_contact_tracking_state,
    run_contact_tracking_trial as _core_run_contact_tracking_trial,
    snapshot_contact_tracking_state as _core_snapshot_contact_tracking_state,
)
from sfc.fem.calculix_aligned import (  # noqa: E402
    ContactSample,
    ContactResponse,
    InternalResponse,
    MechanicsModel,
    MechanicsState,
    assemble_contact_response,
    stvk_internal_response,
)
from sfc.fem.calculix_aligned import (  # noqa: E402
    ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
    calculix_dynamic_predictor,
    hht_newmark_parameters,
)
from sfc.fem.constraints import free_dofs, project_fixed_dofs  # noqa: E402
from sfc.fem.increment_control import (  # noqa: E402
    AutomaticIncrementEvent as SourceAutomaticIncrementEvent,
    AutomaticIncrementResult as SourceAutomaticIncrementResult,
    IncrementConvergenceDecision as SourceIncrementConvergenceDecision,
    active_set_line_search_choice as _core_active_set_line_search_choice,
    active_set_stability_after_line_search as _core_active_set_stability_after_line_search,
    contact_active_set_is_stable as _core_contact_active_set_is_stable,
    increment_convergence_decision as _core_increment_convergence_decision,
    increment_cutback_candidate_dt as _core_increment_cutback_candidate_dt,
    increment_gate_row as _core_increment_gate_row,
    run_automatic_increment_controller as _core_run_automatic_increment_controller,
)
from sfc.fem.implicit_dirichlet import hht_step_dirichlet, initial_state_dirichlet  # noqa: E402
from sfc.fem.rp_mpc import RigidHubMPC, build_rigid_hub_reduced_assembly, merge_dirichlet_conditions  # noqa: E402
from sfc.sdf.material_sdf import MaterialSDF  # noqa: E402
from validation.run_flexible_gear_explicit_sdf_comparison import (  # noqa: E402
    DEFAULT_SOURCE,
    GearInputModel,
    GearMesh,
    parse_gear_input,
)
from validation.vtk_frame_series import write_pvd  # noqa: E402

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class SFCVTKStaticBlocks:
    """Static legacy-VTK blocks for a fixed TET4 topology."""

    elements: np.ndarray
    object_ids: np.ndarray
    object_node_ids_by_object: tuple[tuple[int, np.ndarray], ...]
    cells_block: str
    cell_types_block: str
    object_id_block: str


@dataclass(slots=True)
class _ContactAggregationWorkspace:
    """Reusable topology workspace for source-drive contact-region averaging.

    The cache stores only discrete region membership and row-expansion maps.
    Current gaps, normals, areas, and closest-feature payload values are still
    recomputed from the latest geometry on every residual evaluation.
    """

    mode: str = ""
    base_mode: str = ""
    sample_nodes: np.ndarray | None = None
    positive_weights: np.ndarray | None = None
    group_indices: list[list[int]] | None = None
    group_secondary_node_ids: np.ndarray | None = None
    source_rows: np.ndarray | None = None
    source_locals: np.ndarray | None = None
    slave_node_one_hot: np.ndarray | None = None
    hits: int = 0
    misses: int = 0

    def matches(self, *, mode: str, base_mode: str, sample_nodes: np.ndarray, sample_weights: np.ndarray) -> bool:
        """Return true when the cached contact-region topology is reusable."""

        if self.mode != str(mode) or self.base_mode != str(base_mode):
            return False
        if self.sample_nodes is None or self.positive_weights is None:
            return False
        nodes = np.asarray(sample_nodes, dtype=np.int64)
        positive = np.asarray(sample_weights, dtype=float) > 0.0
        return bool(np.array_equal(self.sample_nodes, nodes) and np.array_equal(self.positive_weights, positive))

    def store(
        self,
        *,
        mode: str,
        base_mode: str,
        sample_nodes: np.ndarray,
        sample_weights: np.ndarray,
        group_indices: list[list[int]],
        group_secondary_node_ids: np.ndarray | None = None,
        source_rows: np.ndarray | None = None,
        source_locals: np.ndarray | None = None,
        slave_node_one_hot: np.ndarray | None = None,
    ) -> None:
        """Replace cached contact-region topology."""

        self.mode = str(mode)
        self.base_mode = str(base_mode)
        self.sample_nodes = np.asarray(sample_nodes, dtype=np.int64).copy()
        self.positive_weights = (np.asarray(sample_weights, dtype=float) > 0.0).copy()
        self.group_indices = [list(map(int, rows)) for rows in group_indices]
        self.group_secondary_node_ids = (
            None if group_secondary_node_ids is None else np.asarray(group_secondary_node_ids, dtype=np.int64).copy()
        )
        self.source_rows = None if source_rows is None else np.asarray(source_rows, dtype=np.int64).copy()
        self.source_locals = None if source_locals is None else np.asarray(source_locals, dtype=np.int64).copy()
        self.slave_node_one_hot = None if slave_node_one_hot is None else np.asarray(slave_node_one_hot, dtype=float).copy()

DEFAULT_OUT_DIR = ROOT / "results" / "flexible_gear_implicit_lagrangian_sdf"
DEFAULT_PRESSURE_STIFFNESS = 5.0e9


def _assemble_contact_response_force_only(samples: Iterable[Any], n_nodes: int) -> ContactResponse:
    """Assemble penalty contact response while preserving the tangent.

    The historical name is kept for callers, but the accepted-state response
    now carries the same fixed-active-set tangent used by the Newton correction.
    """

    return assemble_contact_response(list(samples), int(n_nodes))


def _assemble_contact_arrays_force_only(sample_arrays: dict[str, np.ndarray], n_nodes: int, *, stiffness: float) -> ContactResponse:
    """Assemble contact force, diagnostics, and fixed-active-set tangent."""

    gaps = np.asarray(sample_arrays["gaps"], dtype=float).reshape(-1)
    force = np.zeros((int(n_nodes), 3), dtype=float)
    if gaps.size == 0:
        tangent = csr_matrix((3 * int(n_nodes), 3 * int(n_nodes)), dtype=float)
        return ContactResponse(force, tangent, 0.0, 0.0, 0, 0.0, 0.0)
    normals = np.asarray(sample_arrays["normals"], dtype=float).reshape((-1, 3))
    normal_norm = np.maximum(np.linalg.norm(normals, axis=1), 1.0e-30)
    normals = normals / normal_norm[:, None]
    areas = np.asarray(sample_arrays["areas"], dtype=float).reshape(-1)
    penetration = np.maximum(-gaps, 0.0)
    active = penetration > 0.0
    lam = float(stiffness) * areas * penetration
    if np.any(active):
        active_vectors = lam[active, None] * normals[active]
        slave_nodes = np.asarray(sample_arrays["sample_node_ids"], dtype=np.int64)[active]
        slave_weights = np.asarray(sample_arrays["sample_weights"], dtype=float)[active]
        master_nodes = np.asarray(sample_arrays["master_node_ids"], dtype=np.int64)[active]
        master_weights = np.asarray(sample_arrays["master_weights"], dtype=float)[active]
        for local in range(slave_nodes.shape[1]):
            np.add.at(force, slave_nodes[:, local], slave_weights[:, local, None] * active_vectors)
        for local in range(master_nodes.shape[1]):
            np.add.at(force, master_nodes[:, local], -master_weights[:, local, None] * active_vectors)
    active_ids, gap_jacobian = _constraint_region_gap_jacobian_sparse_from_arrays(
        sample_arrays,
        n_nodes=int(n_nodes),
        active_only=True,
    )
    if active_ids.size:
        tangent_scales = float(stiffness) * areas[active_ids]
        positive = tangent_scales > 0.0
        if np.any(positive):
            j_active = gap_jacobian[positive]
            scaled_j = j_active.multiply(tangent_scales[positive][:, None])
            tangent = (j_active.T @ scaled_j).tocsr()
        else:
            tangent = csr_matrix((3 * int(n_nodes), 3 * int(n_nodes)), dtype=float)
    else:
        tangent = csr_matrix((3 * int(n_nodes), 3 * int(n_nodes)), dtype=float)
    return ContactResponse(
        force,
        tangent,
        float(np.min(gaps)),
        float(np.max(penetration)) if penetration.size else 0.0,
        int(np.count_nonzero(active)),
        float(np.sum(lam[active])) if np.any(active) else 0.0,
        float(0.5 * float(stiffness) * np.sum(areas * penetration * penetration)),
    )


def _contact_samples_from_arrays(sample_arrays: dict[str, np.ndarray], *, stiffness: float) -> list[ContactSample]:
    """Convert batched contact arrays to ``ContactSample`` rows.

    This keeps higher-level Abaqus-style averaging code shared between scalar
    and compiled contact-query paths while avoiding Python per-candidate
    projection loops.
    """

    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    if gaps.size == 0:
        return []
    sample_nodes = np.asarray(sample_arrays["sample_node_ids"], dtype=np.int64)
    sample_weights = np.asarray(sample_arrays["sample_weights"], dtype=float)
    normals = np.asarray(sample_arrays["normals"], dtype=float).reshape((-1, 3))
    areas = np.asarray(sample_arrays["areas"], dtype=float).reshape(-1)
    master_nodes = np.asarray(sample_arrays["master_node_ids"], dtype=np.int64)
    master_weights = np.asarray(sample_arrays["master_weights"], dtype=float)
    out: list[ContactSample] = []
    for idx in range(gaps.size):
        out.append(
            ContactSample(
                node_ids=np.asarray(sample_nodes[idx], dtype=np.int64).copy(),
                shape_weights=np.asarray(sample_weights[idx], dtype=float).copy(),
                gap=float(gaps[idx]),
                normal=np.asarray(normals[idx], dtype=float).copy(),
                area=float(areas[idx]),
                stiffness=float(stiffness),
                master_node_ids=np.asarray(master_nodes[idx], dtype=np.int64).copy(),
                master_shape_weights=np.asarray(master_weights[idx], dtype=float).copy(),
            )
        )
    return out


def _contact_active_signature_from_arrays(sample_arrays: dict[str, np.ndarray] | None) -> tuple[tuple[int, ...], ...]:
    """Return a deterministic active contact signature for convergence checks."""

    if sample_arrays is None:
        return tuple()
    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    if gaps.size == 0:
        return tuple()
    active_ids = np.flatnonzero(gaps < 0.0)
    if active_ids.size == 0:
        return tuple()
    sample_nodes = np.asarray(sample_arrays["sample_node_ids"], dtype=np.int64)
    sample_weights = np.asarray(sample_arrays["sample_weights"], dtype=float)
    master_nodes = np.asarray(sample_arrays["master_node_ids"], dtype=np.int64)
    master_weights = np.asarray(sample_arrays["master_weights"], dtype=float)
    signature: list[tuple[int, ...]] = []
    for idx in active_ids:
        slave_row = np.asarray(sample_nodes[int(idx)], dtype=np.int64).reshape(-1)
        slave_weight_row = np.asarray(sample_weights[int(idx)], dtype=float).reshape(-1)
        master_row = np.asarray(master_nodes[int(idx)], dtype=np.int64).reshape(-1)
        master_weight_row = np.asarray(master_weights[int(idx)], dtype=float).reshape(-1)
        slave = tuple(int(value) for value, weight in zip(slave_row, slave_weight_row, strict=True) if abs(float(weight)) > 1.0e-15)
        master = tuple(int(value) for value, weight in zip(master_row, master_weight_row, strict=True) if abs(float(weight)) > 1.0e-15)
        signature.append(slave + (-1,) + master)
    return tuple(sorted(signature))


def _contact_active_signature_from_samples(samples: list[ContactSample] | None) -> tuple[tuple[int, ...], ...]:
    """Return a deterministic active contact signature for sample objects."""

    if not samples:
        return tuple()
    signature: list[tuple[int, ...]] = []
    for sample in samples:
        if float(sample.gap) >= 0.0:
            continue
        slave_nodes = np.asarray(sample.node_ids, dtype=np.int64).reshape(-1)
        slave_weights = np.asarray(sample.shape_weights, dtype=float).reshape(-1)
        slave = tuple(
            int(value)
            for value, weight in zip(slave_nodes, slave_weights, strict=True)
            if abs(float(weight)) > 1.0e-15
        )
        if sample.master_node_ids is None:
            master: tuple[int, ...] = tuple()
        else:
            master_nodes = np.asarray(sample.master_node_ids, dtype=np.int64).reshape(-1)
            if sample.master_shape_weights is None:
                master = tuple(int(value) for value in master_nodes)
            else:
                master_weights = np.asarray(sample.master_shape_weights, dtype=float).reshape(-1)
                master = tuple(
                    int(value)
                    for value, weight in zip(master_nodes, master_weights, strict=True)
                    if abs(float(weight)) > 1.0e-15
                )
        signature.append(slave + (-1,) + master)
    return tuple(sorted(signature))


def _contact_path_tracking_metrics_from_arrays(
    sample_arrays: dict[str, np.ndarray] | None,
    previous_master_face_ids: np.ndarray | None,
    previous_master_barycentric: np.ndarray | None = None,
    *,
    active_gap_tolerance: float = 0.0,
) -> tuple[Row, np.ndarray | None, np.ndarray | None]:
    """Measure master-face continuity for the accepted batched contact path.

    The metric is diagnostic only: it checks whether the face ids and
    representative barycentric coordinates carried by the
    secondary-normal/path-tracked projection payload change unexpectedly
    between accepted states.  It is intentionally separate from SDF querying.
    """

    metrics, current_faces, current_bary = _core_contact_path_tracking_metrics_from_arrays(
        sample_arrays,
        previous_master_face_ids,
        previous_master_barycentric,
        active_gap_tolerance=active_gap_tolerance,
    )
    return dict(metrics), current_faces, current_bary


def _contact_active_region_continuity_metrics_from_arrays(
    sample_arrays: dict[str, np.ndarray] | None,
    previous_active_region_ids: tuple[int, ...] | None,
    *,
    active_gap_tolerance: float = 0.0,
) -> tuple[Row, tuple[int, ...] | None]:
    """Measure accepted-state active constraint-region continuity.

    This diagnostic compares secondary constraint-region ids before any nodal
    CPRESS recovery.  It answers whether the active contact patch persists
    across accepted states, which is the quantity needed before comparing
    pressure clouds.
    """

    metrics, current_ids = _core_contact_active_region_continuity_metrics_from_arrays(
        sample_arrays,
        previous_active_region_ids,
        active_gap_tolerance=active_gap_tolerance,
    )
    return dict(metrics), current_ids


def _empty_contact_region_integral_metrics() -> Row:
    return {
        "contact_region_count": 0,
        "active_contact_region_count": 0,
        "contact_total_region_area": 0.0,
        "contact_active_area": 0.0,
        "contact_region_normal_force": 0.0,
        "contact_region_energy": 0.0,
        "contact_region_virtual_work": 0.0,
        "contact_max_region_pressure": 0.0,
        "contact_mean_active_region_pressure": 0.0,
    }


def _contact_region_integral_metrics_from_arrays(
    sample_arrays: dict[str, np.ndarray] | None,
    *,
    stiffness: float,
) -> Row:
    """Return region-level force/work/energy metrics before nodal CPRESS checks."""

    return dict(_core_contact_region_integral_metrics_from_arrays(sample_arrays, stiffness=stiffness))

    if sample_arrays is None:
        return _empty_contact_region_integral_metrics()
    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    if gaps.size == 0:
        return _empty_contact_region_integral_metrics()
    areas = np.asarray(sample_arrays.get("areas", np.zeros_like(gaps)), dtype=float).reshape(-1)
    penetrations = np.maximum(-gaps, 0.0)
    pressures = float(stiffness) * penetrations
    active = penetrations > 0.0
    active_area = float(np.sum(areas[active])) if np.any(active) else 0.0
    normal_force = float(np.sum(areas[active] * pressures[active])) if np.any(active) else 0.0
    energy = float(0.5 * np.sum(areas * pressures * penetrations))
    return {
        "contact_region_count": int(gaps.size),
        "active_contact_region_count": int(np.count_nonzero(active)),
        "contact_total_region_area": float(np.sum(areas)),
        "contact_active_area": active_area,
        "contact_region_normal_force": normal_force,
        "contact_region_energy": energy,
        "contact_region_virtual_work": float(2.0 * energy),
        "contact_max_region_pressure": float(np.max(pressures[active])) if np.any(active) else 0.0,
        "contact_mean_active_region_pressure": (
            float(np.sum(areas[active] * pressures[active]) / max(active_area, 1.0e-30)) if np.any(active) else 0.0
        ),
    }


def _contact_region_integral_metrics_from_samples(samples: list[ContactSample] | None) -> Row:
    if not samples:
        return _empty_contact_region_integral_metrics()
    gaps = np.asarray([float(sample.gap) for sample in samples], dtype=float)
    areas = np.asarray([float(sample.area) for sample in samples], dtype=float)
    stiffness = np.asarray([float(sample.stiffness) for sample in samples], dtype=float)
    penetrations = np.maximum(-gaps, 0.0)
    pressures = stiffness * penetrations
    active = penetrations > 0.0
    active_area = float(np.sum(areas[active])) if np.any(active) else 0.0
    energy = float(0.5 * np.sum(areas * stiffness * penetrations * penetrations))
    return {
        "contact_region_count": int(gaps.size),
        "active_contact_region_count": int(np.count_nonzero(active)),
        "contact_total_region_area": float(np.sum(areas)),
        "contact_active_area": active_area,
        "contact_region_normal_force": float(np.sum(areas[active] * pressures[active])) if np.any(active) else 0.0,
        "contact_region_energy": energy,
        "contact_region_virtual_work": float(2.0 * energy),
        "contact_max_region_pressure": float(np.max(pressures[active])) if np.any(active) else 0.0,
        "contact_mean_active_region_pressure": (
            float(np.sum(areas[active] * pressures[active]) / max(active_area, 1.0e-30)) if np.any(active) else 0.0
        ),
    }


def _constraint_region_contact_law_metrics_from_arrays(
    sample_arrays: dict[str, np.ndarray] | None,
    raw_sample_arrays: dict[str, np.ndarray] | None,
    *,
    averaging_mode: str,
) -> Row:
    """Describe the active contact law used by accepted region rows.

    This is a provenance/claim-gate helper, not a separate contact model.  It
    records whether the accepted response is assembled from secondary
    constraint regions with signed area-average gap status, area-average
    normal direction, region area, slave shape-function support, and master
    closest-feature payload.  Those are the quantities needed before comparing
    Abaqus-style totals and pressure clouds.
    """

    mode = str(averaging_mode).lower()
    try:
        base_mode, overclosure_mode = _contact_averaging_modes(mode)
    except ValueError:
        base_mode, overclosure_mode = "invalid", "invalid"
    if sample_arrays is None:
        return {
            "contact_constraint_law_source": "none",
            "contact_constraint_region_source": "none",
            "contact_constraint_open_closed_source": "none",
            "contact_constraint_active_status_source": "none",
            "contact_constraint_normal_source": "none",
            "contact_constraint_region_area_source": "none",
            "contact_constraint_force_distribution": "none",
            "contact_constraint_raw_sample_count": (
                int(np.asarray(raw_sample_arrays.get("gaps", np.empty(0))).size)
                if raw_sample_arrays is not None
                else ""
            ),
            "contact_constraint_region_count": 0,
            "contact_constraint_active_region_count": 0,
            "contact_constraint_secondary_node_regions_present": 0,
            "contact_constraint_master_payload_present": 0,
            "contact_constraint_area_positive": 0,
            "contact_constraint_independent_quadrature_penalty_disabled": 0,
        }
    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    areas = np.asarray(sample_arrays.get("areas", np.empty(0)), dtype=float).reshape(-1)
    secondary_ids = np.asarray(sample_arrays.get("secondary_node_ids", np.empty(0)), dtype=np.int64).reshape(-1)
    master_nodes = np.asarray(sample_arrays.get("master_node_ids", np.empty((0, 0))), dtype=np.int64)
    master_weights = np.asarray(sample_arrays.get("master_weights", np.empty((0, 0))), dtype=float)
    raw_count: int | str = ""
    if raw_sample_arrays is not None:
        raw_count = int(np.asarray(raw_sample_arrays.get("gaps", np.empty(0))).size)
    is_constraint_region = base_mode == "slave_node_region" and overclosure_mode == "signed_average"
    secondary_present = bool(gaps.size == 0 or (secondary_ids.shape == gaps.shape and np.all(secondary_ids >= 0)))
    if master_nodes.ndim == 2 and master_weights.ndim == 2 and master_nodes.shape == master_weights.shape:
        weight_sum = np.sum(np.maximum(master_weights, 0.0), axis=1) if master_weights.size else np.empty(0)
        master_payload_present = bool(gaps.size == 0 or (weight_sum.shape == gaps.shape and np.all(weight_sum > 0.0)))
    else:
        master_payload_present = False
    area_positive = bool(gaps.size == 0 or (areas.shape == gaps.shape and np.all(areas > 0.0)))
    return {
        "contact_constraint_law_source": mode,
        "contact_constraint_region_source": (
            "secondary_node_constraint_region" if base_mode == "slave_node_region" else base_mode
        ),
        "contact_constraint_open_closed_source": (
            "signed_area_average_region_gap" if overclosure_mode == "signed_average" else overclosure_mode
        ),
        "contact_constraint_active_status_source": "aggregated_region_gap" if is_constraint_region else overclosure_mode,
        "contact_constraint_normal_source": (
            "area_average_region_normal" if overclosure_mode == "signed_average" else "overclosure_weighted_region_normal"
        ),
        "contact_constraint_region_area_source": (
            "slave_shape_tributary_area" if base_mode == "slave_node_region" else "sample_or_face_area"
        ),
        "contact_constraint_force_distribution": (
            "region_area_slave_shape_master_payload" if is_constraint_region else "sample_area_sample_shape_payload"
        ),
        "contact_constraint_raw_sample_count": raw_count,
        "contact_constraint_region_count": int(gaps.size),
        "contact_constraint_active_region_count": int(np.count_nonzero(gaps < 0.0)) if gaps.size else 0,
        "contact_constraint_secondary_node_regions_present": int(secondary_present),
        "contact_constraint_master_payload_present": int(master_payload_present),
        "contact_constraint_area_positive": int(area_positive),
        "contact_constraint_independent_quadrature_penalty_disabled": int(is_constraint_region),
    }


def _source_contact_active_set_is_stable(
    active_signature: tuple[tuple[int, ...], ...],
    previous_active_signature: tuple[tuple[int, ...], ...] | None,
    *,
    require_stability: bool,
) -> bool:
    """Return the source-drive contact-status stability gate.

    When requested, the first contact evaluation of an increment is not yet
    stable.  Abaqus/Standard treats open/closed changes as contact-status
    discontinuities and only accepts after the status has stopped changing
    across iterations.
    """

    return _core_contact_active_set_is_stable(
        active_signature,
        previous_active_signature,
        require_stability=require_stability,
    )


def _source_active_set_line_search_choice(
    current_signature: tuple[tuple[int, ...], ...] | None,
    candidates: Iterable[tuple[float, tuple[tuple[int, ...], ...]]],
    *,
    require_stability: bool,
) -> tuple[float, bool, int]:
    """Choose a correction fraction that preserves the current active set.

    The candidates are tried in caller-provided order, normally full step,
    half step, quarter step, and so on.  If active-set stability is not
    required, the first candidate is accepted.  If no candidate preserves the
    current signature, the first candidate is returned with ``stable=False`` so
    the outer increment controller can still cut back instead of silently
    accepting an unstable state.
    """

    return _core_active_set_line_search_choice(
        current_signature,
        candidates,
        require_stability=require_stability,
    )


def _source_active_set_stability_after_line_search(
    active_set_stable: bool,
    *,
    line_search_attempted: bool,
    line_search_stable: bool,
) -> bool:
    """Return the active-set gate used by the increment acceptance check.

    If line search is active and every candidate changes the active contact
    signature, the current Newton correction has an unresolved contact-status
    discontinuity.  Abaqus-style acceptance must treat that increment as
    active-set unstable until a later residual evaluation proves otherwise.
    """

    return _core_active_set_stability_after_line_search(
        active_set_stable,
        line_search_attempted=line_search_attempted,
        line_search_stable=line_search_stable,
    )


def _snapshot_contact_tracking_state(contact_geometries: Iterable[Any]) -> list[dict[str, Any]]:
    """Capture path-tracking caches so trial queries cannot become accepted state."""

    return _core_snapshot_contact_tracking_state(contact_geometries)


def _restore_contact_tracking_state(states: list[dict[str, Any]]) -> None:
    """Restore path-tracking caches captured by ``_snapshot_contact_tracking_state``."""

    _core_restore_contact_tracking_state(states)


def _source_increment_convergence_decision(
    *,
    residual_converged: bool,
    correction_converged: bool,
    contact_force_increment_converged: bool,
    active_set_stable: bool,
    iteration_count: int,
    max_iterations: int,
    accept_unconverged: bool,
) -> SourceIncrementConvergenceDecision:
    """Evaluate the source-drive convergence gates as one acceptance decision.

    Abaqus/Standard accepts an increment only after the nonlinear residual,
    displacement correction, contact-force increment, and contact status have
    all stabilized.  If the iteration limit is reached before those gates pass,
    the mathematically correct action is a cutback, not silently treating the
    current iterate as an accepted converged state.
    """

    return _core_increment_convergence_decision(
        residual_converged=residual_converged,
        correction_converged=correction_converged,
        contact_force_increment_converged=contact_force_increment_converged,
        active_set_stable=active_set_stable,
        iteration_count=iteration_count,
        max_iterations=max_iterations,
        accept_unconverged=accept_unconverged,
    )


def _source_increment_cutback_candidate_dt(
    current_dt: float,
    *,
    min_dt: float,
    cutback_factor: float,
) -> float | None:
    """Return the next cutback trial increment or ``None`` if at the floor."""

    return _core_increment_cutback_candidate_dt(
        current_dt,
        min_dt=min_dt,
        cutback_factor=cutback_factor,
    )


def _run_source_automatic_increment_controller(
    *,
    duration: float,
    initial_dt: float,
    min_dt: float,
    cutback_factor: float,
    trial: Callable[[float, float, int], SourceIncrementConvergenceDecision],
) -> SourceAutomaticIncrementResult:
    """Run an Abaqus-style accept/cutback scheduler for trial increments.

    The callback performs one nonlinear trial from ``start_time`` over ``dt``.
    Only trials whose convergence decision is accepted advance the accepted
    time.  Failed trials request cutback and are retried from the same accepted
    state with a smaller increment.
    """

    return _core_run_automatic_increment_controller(
        duration=duration,
        initial_dt=initial_dt,
        min_dt=min_dt,
        cutback_factor=cutback_factor,
        trial=trial,
    )


def _source_increment_gate_row(
    *,
    residual_converged: bool,
    correction_converged: bool,
    contact_force_increment_converged: bool,
    active_set_stable: bool,
    normalized_residual: float,
    normalized_correction: float,
    normalized_contact_force_increment: float,
    contact_force_increment_norm: float,
    decision: SourceIncrementConvergenceDecision,
    cutback_candidate_dt: float | None,
) -> Row:
    """Return one manifest/history row for Abaqus-style increment gates."""

    return _core_increment_gate_row(
        residual_converged=residual_converged,
        correction_converged=correction_converged,
        contact_force_increment_converged=contact_force_increment_converged,
        active_set_stable=active_set_stable,
        normalized_residual=normalized_residual,
        normalized_correction=normalized_correction,
        normalized_contact_force_increment=normalized_contact_force_increment,
        contact_force_increment_norm=contact_force_increment_norm,
        decision=decision,
        cutback_candidate_dt=cutback_candidate_dt,
        prefix="source",
    )


def _combined_shape_weights(
    node_rows: list[np.ndarray],
    weight_rows: list[np.ndarray],
    group_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine sample shape-function weights over a constraint group."""

    accum: dict[int, float] = {}
    for nodes, weights, scale in zip(node_rows, weight_rows, group_weights, strict=True):
        for node, weight in zip(np.asarray(nodes, dtype=np.int64).reshape(-1), np.asarray(weights, dtype=float).reshape(-1), strict=True):
            key = int(node)
            accum[key] = accum.get(key, 0.0) + float(scale) * float(weight)
    if not accum:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    ids = np.asarray(sorted(accum), dtype=np.int64)
    weights = np.asarray([accum[int(node)] for node in ids], dtype=float)
    total = float(np.sum(weights))
    if total > 0.0:
        weights /= total
    return ids, weights


def _contact_averaging_modes(mode: str) -> tuple[str, str]:
    """Return ``(base_mode, overclosure_mode)`` for contact averaging labels."""

    averaging = str(mode).lower()
    signed_participation_mode = averaging.endswith("_signed_participation")
    signed_mode = averaging.endswith("_constraint")
    area_average_mode = averaging.endswith("_area_average")
    participation_mode = averaging.endswith("_participation")
    if signed_participation_mode:
        participation_mode = False
    if sum(int(flag) for flag in (signed_mode, area_average_mode, participation_mode, signed_participation_mode)) > 1:
        raise ValueError("contact averaging mode cannot combine suffixes")
    if signed_participation_mode:
        base_mode = averaging[: -len("_signed_participation")]
        overclosure_mode = "signed_status_linear_penalty_participation"
    elif signed_mode:
        base_mode = averaging[: -len("_constraint")]
        overclosure_mode = "signed_average"
    elif area_average_mode:
        base_mode = averaging[: -len("_area_average")]
        overclosure_mode = "positive_integral_area_average"
    elif participation_mode:
        base_mode = averaging[: -len("_participation")]
        overclosure_mode = "linear_penalty_participation"
    else:
        base_mode = averaging
        overclosure_mode = "positive_integral"
    if base_mode not in {"none", "slave_face", "slave_node", "slave_node_region", "surface_patch"}:
        raise ValueError(
            "contact averaging must be 'none', 'slave_face', 'slave_node', 'slave_node_region', "
            "'surface_patch', or the corresponding '*_constraint'/'*_area_average'/'*_participation'/"
            "'*_signed_participation' modes"
        )
    return base_mode, overclosure_mode


def _combined_shape_weights_from_arrays(
    node_rows: np.ndarray,
    weight_rows: np.ndarray,
    row_indices: list[int],
    group_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine padded node/weight rows into one participation row."""

    accum: dict[int, float] = {}
    for local, row_index in enumerate(row_indices):
        nodes = np.asarray(node_rows[int(row_index)], dtype=np.int64).reshape(-1)
        weights = np.asarray(weight_rows[int(row_index)], dtype=float).reshape(-1)
        scale = float(group_weights[int(local)])
        for node, weight in zip(nodes, weights, strict=True):
            contribution = scale * float(weight)
            if abs(contribution) <= 1.0e-30:
                continue
            key = int(node)
            accum[key] = accum.get(key, 0.0) + contribution
    if not accum:
        return np.asarray([0], dtype=np.int64), np.asarray([0.0], dtype=float)
    ids = np.asarray(sorted(accum), dtype=np.int64)
    weights = np.asarray([accum[int(node)] for node in ids], dtype=float)
    total = float(np.sum(weights))
    if total > 0.0:
        weights /= total
    return ids, weights


def _aggregate_contact_array_group(
    row_indices: list[int],
    *,
    sample_nodes: np.ndarray,
    sample_weights: np.ndarray,
    gaps: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    master_nodes: np.ndarray,
    master_weights: np.ndarray,
    master_barycentric: np.ndarray | None,
    master_face_ids: np.ndarray | None,
    secondary_cache_indices: np.ndarray | None,
    secondary_node_id: int | None,
    tracking_cache_hits: np.ndarray | None,
    tracking_cache_matches: np.ndarray | None,
    tracking_barycentric_distances: np.ndarray | None,
    overclosure_mode: str,
) -> dict[str, np.ndarray | float]:
    """Aggregate a set of contact-array rows into one equivalent constraint."""

    if not row_indices:
        raise ValueError("row_indices must not be empty")
    rows = np.asarray(row_indices, dtype=np.int64)
    row_areas = np.maximum(np.asarray(areas[rows], dtype=float).reshape(-1), 0.0)
    total_area = float(np.sum(row_areas))
    if total_area <= 0.0:
        group_weights = np.full(rows.size, 1.0 / float(rows.size), dtype=float)
        total_area = float(rows.size)
    else:
        group_weights = row_areas / total_area
    row_gaps = np.asarray(gaps[rows], dtype=float).reshape(-1)
    aggregate_area = total_area
    mode = str(overclosure_mode).lower()
    if mode == "signed_average":
        constraint_gap = float(group_weights @ row_gaps)
        normal_weights = group_weights
        kinematic_weights = group_weights
    elif mode == "signed_status_linear_penalty_participation":
        signed_gap = float(group_weights @ row_gaps)
        if signed_gap >= 0.0:
            constraint_gap = signed_gap
            normal_weights = group_weights
            kinematic_weights = group_weights
        else:
            penetrations = np.maximum(-row_gaps, 0.0)
            penetration_integral = float(row_areas @ penetrations)
            if penetration_integral > 0.0:
                energy_integral = float(row_areas @ (penetrations * penetrations))
                if energy_integral > 0.0:
                    aggregate_area = max((penetration_integral * penetration_integral) / energy_integral, 1.0e-30)
                    constraint_gap = -penetration_integral / aggregate_area
                else:
                    aggregate_area = total_area
                    constraint_gap = signed_gap
                normal_weights = row_areas * penetrations / penetration_integral
                kinematic_weights = normal_weights
            else:
                constraint_gap = signed_gap
                normal_weights = group_weights
                kinematic_weights = group_weights
    else:
        penetrations = np.maximum(-row_gaps, 0.0)
        penetration_integral = float(row_areas @ penetrations)
        if penetration_integral > 0.0:
            if mode == "linear_penalty_participation":
                energy_integral = float(row_areas @ (penetrations * penetrations))
                if energy_integral > 0.0:
                    aggregate_area = max((penetration_integral * penetration_integral) / energy_integral, 1.0e-30)
                    constraint_gap = -penetration_integral / aggregate_area
                else:
                    aggregate_area = total_area
                    constraint_gap = -penetration_integral / max(total_area, 1.0e-30)
                normal_weights = row_areas * penetrations / penetration_integral
                kinematic_weights = normal_weights
            else:
                constraint_gap = -penetration_integral / max(total_area, 1.0e-30)
                if mode == "positive_integral_area_average":
                    normal_weights = group_weights
                    kinematic_weights = group_weights
                else:
                    normal_weights = row_areas * penetrations / penetration_integral
                    kinematic_weights = normal_weights
        else:
            constraint_gap = float(group_weights @ row_gaps)
            normal_weights = group_weights
            kinematic_weights = group_weights
    normal = np.sum(normals[rows] * normal_weights[:, None], axis=0)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0.0:
        normal = np.asarray(normals[int(rows[0])], dtype=float).copy()
        normal_norm = max(float(np.linalg.norm(normal)), 1.0e-30)
    normal = normal / normal_norm
    slave_ids, slave_w = _combined_shape_weights_from_arrays(sample_nodes, sample_weights, row_indices, kinematic_weights)
    master_ids, master_w = _combined_shape_weights_from_arrays(master_nodes, master_weights, row_indices, kinematic_weights)
    master_face_id = -1
    representative_local: int | None = None
    if master_face_ids is not None:
        row_face_ids = np.asarray(master_face_ids, dtype=np.int64).reshape(-1)
        if row_face_ids.size:
            valid_faces = row_face_ids[rows]
            valid_mask = valid_faces >= 0
            if np.any(valid_mask):
                weights_for_face = np.asarray(kinematic_weights, dtype=float).reshape(-1)
                best_local = int(np.argmax(np.where(valid_mask, weights_for_face, -np.inf)))
                representative_local = best_local
                master_face_id = int(valid_faces[best_local])
    master_bary_width = 3
    if master_barycentric is not None and representative_local is not None:
        row_bary = np.asarray(master_barycentric, dtype=float)
        if row_bary.ndim == 2 and row_bary.shape[0] >= int(np.max(rows)) + 1:
            master_bary_width = int(row_bary.shape[1])
    master_bary = np.full(master_bary_width, np.nan, dtype=float)
    if master_barycentric is not None and representative_local is not None:
        row_bary = np.asarray(master_barycentric, dtype=float)
        if row_bary.ndim == 2 and row_bary.shape[0] >= int(np.max(rows)) + 1 and row_bary.shape[1] == master_bary_width:
            master_bary = row_bary[int(rows[int(representative_local)])].copy()
    secondary_cache_index = -1
    if secondary_cache_indices is not None and representative_local is not None:
        cache_ids = np.asarray(secondary_cache_indices, dtype=np.int64).reshape(-1)
        source_row = int(rows[int(representative_local)])
        if cache_ids.size > source_row:
            secondary_cache_index = int(cache_ids[source_row])
    tracking_hit = False
    if tracking_cache_hits is not None:
        row_hits = np.asarray(tracking_cache_hits, dtype=bool).reshape(-1)
        if row_hits.size:
            tracking_hit = bool(np.any(row_hits[rows]))
    tracking_match = False
    if tracking_cache_matches is not None:
        row_matches = np.asarray(tracking_cache_matches, dtype=bool).reshape(-1)
        if row_matches.size:
            tracking_match = bool(np.any(row_matches[rows]))
    tracking_barycentric_distance = np.nan
    if tracking_barycentric_distances is not None:
        row_distances = np.asarray(tracking_barycentric_distances, dtype=float).reshape(-1)
        if row_distances.size:
            distances = row_distances[rows]
            finite = np.isfinite(distances)
            if np.any(finite):
                weights_for_distance = np.asarray(kinematic_weights, dtype=float).reshape(-1)[finite]
                weight_sum = float(np.sum(weights_for_distance))
                if weight_sum > 0.0:
                    tracking_barycentric_distance = float(weights_for_distance @ distances[finite] / weight_sum)
                else:
                    tracking_barycentric_distance = float(np.mean(distances[finite]))
    return {
        "sample_node_ids": slave_ids,
        "sample_weights": slave_w,
        "gap": float(constraint_gap),
        "normal": normal,
        "area": float(aggregate_area),
        "master_node_ids": master_ids,
        "master_weights": master_w,
        "master_barycentric": master_bary,
        "master_face_id": int(master_face_id),
        "secondary_cache_index": int(secondary_cache_index),
        "secondary_node_id": int(-1 if secondary_node_id is None else secondary_node_id),
        "tracking_cache_hit": int(tracking_hit),
        "tracking_cache_match": int(tracking_match),
        "tracking_barycentric_distance": float(tracking_barycentric_distance),
    }


def _pack_aggregated_contact_rows(rows: list[dict[str, np.ndarray | float]]) -> dict[str, np.ndarray]:
    """Pack variable-width aggregate rows into padded sample arrays."""

    if not rows:
        return {
            "sample_node_ids": np.empty((0, 0), dtype=np.int64),
            "sample_weights": np.empty((0, 0), dtype=float),
            "gaps": np.empty(0, dtype=float),
            "normals": np.empty((0, 3), dtype=float),
            "areas": np.empty(0, dtype=float),
            "master_node_ids": np.empty((0, 0), dtype=np.int64),
            "master_weights": np.empty((0, 0), dtype=float),
            "master_barycentric": np.empty((0, 3), dtype=float),
            "master_face_ids": np.empty(0, dtype=np.int64),
            "secondary_cache_indices": np.empty(0, dtype=np.int64),
            "secondary_node_ids": np.empty(0, dtype=np.int64),
            "tracking_cache_hits": np.empty(0, dtype=bool),
            "tracking_cache_matches": np.empty(0, dtype=bool),
            "tracking_barycentric_distances": np.empty(0, dtype=float),
        }
    slave_width = max(int(np.asarray(row["sample_node_ids"]).size) for row in rows)
    master_width = max(int(np.asarray(row["master_node_ids"]).size) for row in rows)
    sample_node_ids = np.zeros((len(rows), slave_width), dtype=np.int64)
    sample_weights = np.zeros((len(rows), slave_width), dtype=float)
    master_node_ids = np.zeros((len(rows), master_width), dtype=np.int64)
    master_weights = np.zeros((len(rows), master_width), dtype=float)
    gaps = np.zeros(len(rows), dtype=float)
    normals = np.zeros((len(rows), 3), dtype=float)
    areas = np.zeros(len(rows), dtype=float)
    master_face_ids = np.full(len(rows), -1, dtype=np.int64)
    master_bary_width = max(int(np.asarray(row.get("master_barycentric", np.full(3, np.nan))).size) for row in rows)
    master_barycentric = np.full((len(rows), master_bary_width), np.nan, dtype=float)
    secondary_cache_indices = np.full(len(rows), -1, dtype=np.int64)
    secondary_node_ids = np.full(len(rows), -1, dtype=np.int64)
    tracking_cache_hits = np.zeros(len(rows), dtype=bool)
    tracking_cache_matches = np.zeros(len(rows), dtype=bool)
    tracking_barycentric_distances = np.full(len(rows), np.nan, dtype=float)
    for index, row in enumerate(rows):
        slave_ids = np.asarray(row["sample_node_ids"], dtype=np.int64).reshape(-1)
        slave_w = np.asarray(row["sample_weights"], dtype=float).reshape(-1)
        master_ids = np.asarray(row["master_node_ids"], dtype=np.int64).reshape(-1)
        master_w = np.asarray(row["master_weights"], dtype=float).reshape(-1)
        sample_node_ids[index, : slave_ids.size] = slave_ids
        sample_weights[index, : slave_w.size] = slave_w
        master_node_ids[index, : master_ids.size] = master_ids
        master_weights[index, : master_w.size] = master_w
        gaps[index] = float(row["gap"])
        normals[index] = np.asarray(row["normal"], dtype=float).reshape(3)
        areas[index] = float(row["area"])
        master_face_ids[index] = int(row.get("master_face_id", -1))
        bary = np.asarray(row.get("master_barycentric", np.full(master_bary_width, np.nan)), dtype=float).reshape(-1)
        master_barycentric[index, : bary.size] = bary
        secondary_cache_indices[index] = int(row.get("secondary_cache_index", -1))
        secondary_node_ids[index] = int(row.get("secondary_node_id", -1))
        tracking_cache_hits[index] = bool(row.get("tracking_cache_hit", 0))
        tracking_cache_matches[index] = bool(row.get("tracking_cache_match", 0))
        tracking_barycentric_distances[index] = float(row.get("tracking_barycentric_distance", np.nan))
    return {
        "sample_node_ids": sample_node_ids,
        "sample_weights": sample_weights,
        "gaps": gaps,
        "normals": normals,
        "areas": areas,
        "master_node_ids": master_node_ids,
        "master_weights": master_weights,
        "master_barycentric": master_barycentric,
        "master_face_ids": master_face_ids,
        "secondary_cache_indices": secondary_cache_indices,
        "secondary_node_ids": secondary_node_ids,
        "tracking_cache_hits": tracking_cache_hits,
        "tracking_cache_matches": tracking_cache_matches,
        "tracking_barycentric_distances": tracking_barycentric_distances,
    }


def _aggregate_contact_sample_arrays(
    sample_arrays: dict[str, np.ndarray],
    mode: str,
    *,
    workspace: _ContactAggregationWorkspace | None = None,
) -> dict[str, np.ndarray] | None:
    """Aggregate contact sample arrays without object conversion.

    This keeps the C++/batched query path available for Abaqus-style
    surface-to-surface constraint-region modes.
    """

    if workspace is None and str(mode).lower() in {"none", "slave_node_region_constraint"}:
        return _core_aggregate_contact_sample_arrays(sample_arrays, mode)

    base_mode, overclosure_mode = _contact_averaging_modes(mode)
    if base_mode == "none":
        return sample_arrays
    if base_mode == "surface_patch":
        return None
    if base_mode == "slave_node_point":
        return None
    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    if gaps.size == 0:
        return sample_arrays
    sample_nodes = np.asarray(sample_arrays["sample_node_ids"], dtype=np.int64)
    sample_weights = np.asarray(sample_arrays["sample_weights"], dtype=float)
    normals = np.asarray(sample_arrays["normals"], dtype=float).reshape((-1, 3))
    normal_norm = np.maximum(np.linalg.norm(normals, axis=1), 1.0e-30)
    normals = normals / normal_norm[:, None]
    areas = np.asarray(sample_arrays["areas"], dtype=float).reshape(-1)
    master_nodes = np.asarray(sample_arrays["master_node_ids"], dtype=np.int64)
    master_weights = np.asarray(sample_arrays["master_weights"], dtype=float)
    master_barycentric_in = sample_arrays.get("master_barycentric")
    master_barycentric = (
        np.asarray(master_barycentric_in, dtype=float)
        if master_barycentric_in is not None
        else (master_weights.copy() if master_weights.ndim == 2 and master_weights.shape[1] == 3 else None)
    )
    master_face_ids_in = sample_arrays.get("master_face_ids")
    master_face_ids = (
        None
        if master_face_ids_in is None
        else np.asarray(master_face_ids_in, dtype=np.int64).reshape(-1)
    )
    secondary_cache_indices_in = sample_arrays.get("secondary_cache_indices")
    secondary_cache_indices = (
        None
        if secondary_cache_indices_in is None
        else np.asarray(secondary_cache_indices_in, dtype=np.int64).reshape(-1)
    )
    expanded_sample_nodes = sample_nodes
    expanded_sample_weights = sample_weights
    expanded_gaps = gaps
    expanded_normals = normals
    expanded_areas = areas
    expanded_master_nodes = master_nodes
    expanded_master_weights = master_weights
    expanded_master_barycentric = master_barycentric
    expanded_master_face_ids = master_face_ids
    expanded_secondary_cache_indices = secondary_cache_indices
    tracking_cache_hits_in = sample_arrays.get("tracking_cache_hits")
    tracking_cache_matches_in = sample_arrays.get("tracking_cache_matches")
    tracking_cache_hits = None if tracking_cache_hits_in is None else np.asarray(tracking_cache_hits_in, dtype=bool).reshape(-1)
    tracking_cache_matches = (
        None if tracking_cache_matches_in is None else np.asarray(tracking_cache_matches_in, dtype=bool).reshape(-1)
    )
    tracking_barycentric_distances_in = sample_arrays.get("tracking_barycentric_distances")
    tracking_barycentric_distances = (
        None
        if tracking_barycentric_distances_in is None
        else np.asarray(tracking_barycentric_distances_in, dtype=float).reshape(-1)
    )
    expanded_tracking_cache_hits = tracking_cache_hits
    expanded_tracking_cache_matches = tracking_cache_matches
    expanded_tracking_barycentric_distances = tracking_barycentric_distances
    group_indices: list[list[int]]
    group_secondary_node_ids: np.ndarray | None = None
    cache_hit = bool(
        workspace is not None
        and workspace.matches(
            mode=str(mode),
            base_mode=base_mode,
            sample_nodes=sample_nodes,
            sample_weights=sample_weights,
        )
        and workspace.group_indices is not None
    )
    if base_mode == "slave_face":
        if cache_hit:
            group_indices = [list(rows) for rows in workspace.group_indices or []]
            if workspace is not None:
                workspace.hits += 1
        else:
            groups: dict[tuple[int, ...], list[int]] = {}
            order: list[tuple[int, ...]] = []
            for idx, nodes in enumerate(sample_nodes):
                key = tuple(int(v) for v in np.asarray(nodes, dtype=np.int64).reshape(-1))
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(int(idx))
            group_indices = [groups[key] for key in order]
            group_secondary_node_ids = np.full(len(group_indices), -1, dtype=np.int64)
            if workspace is not None:
                workspace.misses += 1
                workspace.store(
                    mode=str(mode),
                    base_mode=base_mode,
                    sample_nodes=sample_nodes,
                    sample_weights=sample_weights,
                    group_indices=group_indices,
                    group_secondary_node_ids=group_secondary_node_ids,
                )
    else:
        if cache_hit:
            if (
                workspace is None
                or workspace.source_rows is None
                or workspace.source_locals is None
                or workspace.group_indices is None
                or workspace.group_secondary_node_ids is None
            ):
                cache_hit = False
            else:
                source_rows = workspace.source_rows
                source_locals = workspace.source_locals
                group_indices = [list(rows) for rows in workspace.group_indices]
                group_secondary_node_ids = np.asarray(workspace.group_secondary_node_ids, dtype=np.int64).copy()
                if source_rows.size:
                    expanded_sample_nodes = sample_nodes[source_rows]
                    if base_mode == "slave_node":
                        expanded_sample_weights = np.asarray(workspace.slave_node_one_hot, dtype=float).copy()
                    else:
                        expanded_sample_weights = sample_weights[source_rows].copy()
                    expanded_gaps = gaps[source_rows]
                    expanded_normals = normals[source_rows]
                    local_weights = np.maximum(sample_weights[source_rows, source_locals], 0.0)
                    expanded_areas = areas[source_rows] * local_weights
                    expanded_master_nodes = master_nodes[source_rows]
                    expanded_master_weights = master_weights[source_rows]
                    expanded_master_barycentric = None if master_barycentric is None else master_barycentric[source_rows]
                    expanded_master_face_ids = None if master_face_ids is None else master_face_ids[source_rows]
                    expanded_secondary_cache_indices = (
                        None if secondary_cache_indices is None else secondary_cache_indices[source_rows]
                    )
                    expanded_tracking_cache_hits = None if tracking_cache_hits is None else tracking_cache_hits[source_rows]
                    expanded_tracking_cache_matches = (
                        None if tracking_cache_matches is None else tracking_cache_matches[source_rows]
                    )
                    expanded_tracking_barycentric_distances = (
                        None if tracking_barycentric_distances is None else tracking_barycentric_distances[source_rows]
                    )
                else:
                    expanded_sample_nodes = np.empty((0, sample_nodes.shape[1]), dtype=np.int64)
                    expanded_sample_weights = np.empty((0, sample_weights.shape[1]), dtype=float)
                    expanded_gaps = np.empty(0, dtype=float)
                    expanded_normals = np.empty((0, 3), dtype=float)
                    expanded_areas = np.empty(0, dtype=float)
                    expanded_master_nodes = np.empty((0, master_nodes.shape[1]), dtype=np.int64)
                    expanded_master_weights = np.empty((0, master_weights.shape[1]), dtype=float)
                    expanded_master_barycentric = None if master_barycentric is None else np.empty((0, 3), dtype=float)
                    expanded_master_face_ids = None if master_face_ids is None else np.empty(0, dtype=np.int64)
                    expanded_secondary_cache_indices = (
                        None if secondary_cache_indices is None else np.empty(0, dtype=np.int64)
                    )
                    expanded_tracking_cache_hits = None if tracking_cache_hits is None else np.empty(0, dtype=bool)
                    expanded_tracking_cache_matches = (
                        None if tracking_cache_matches is None else np.empty(0, dtype=bool)
                    )
                    expanded_tracking_barycentric_distances = (
                        None if tracking_barycentric_distances is None else np.empty(0, dtype=float)
                    )
                workspace.hits += 1
        if not cache_hit:
            groups: dict[int, list[int]] = {}
            order: list[int] = []
            source_rows_list: list[int] = []
            source_locals_list: list[int] = []
            one_hot_rows: list[np.ndarray] = []
            expanded_index = 0
            for idx in range(gaps.size):
                nodes = np.asarray(sample_nodes[idx], dtype=np.int64).reshape(-1)
                weights = np.asarray(sample_weights[idx], dtype=float).reshape(-1)
                for local, (node, weight) in enumerate(zip(nodes, weights, strict=True)):
                    tributary = float(areas[idx]) * max(float(weight), 0.0)
                    if tributary <= 0.0:
                        continue
                    source_rows_list.append(int(idx))
                    source_locals_list.append(int(local))
                    if base_mode == "slave_node":
                        shape = np.zeros_like(weights, dtype=float)
                        shape[int(local)] = 1.0
                        one_hot_rows.append(shape)
                    key = int(node)
                    if key not in groups:
                        groups[key] = []
                        order.append(key)
                    groups[key].append(expanded_index)
                    expanded_index += 1
            source_rows = np.asarray(source_rows_list, dtype=np.int64)
            source_locals = np.asarray(source_locals_list, dtype=np.int64)
            group_indices = [groups[key] for key in order]
            group_secondary_node_ids = np.asarray(order, dtype=np.int64)
            if source_rows.size == 0:
                if workspace is not None:
                    workspace.misses += 1
                    workspace.store(
                        mode=str(mode),
                        base_mode=base_mode,
                        sample_nodes=sample_nodes,
                        sample_weights=sample_weights,
                        group_indices=[],
                        group_secondary_node_ids=np.empty(0, dtype=np.int64),
                        source_rows=source_rows,
                        source_locals=source_locals,
                        slave_node_one_hot=np.empty((0, sample_weights.shape[1]), dtype=float),
                    )
                return {
                    "sample_node_ids": np.empty((0, sample_nodes.shape[1]), dtype=np.int64),
                    "sample_weights": np.empty((0, sample_weights.shape[1]), dtype=float),
                    "gaps": np.empty(0, dtype=float),
                    "normals": np.empty((0, 3), dtype=float),
                    "areas": np.empty(0, dtype=float),
                    "master_node_ids": np.empty((0, master_nodes.shape[1]), dtype=np.int64),
                    "master_weights": np.empty((0, master_weights.shape[1]), dtype=float),
                    "master_barycentric": np.empty((0, 3), dtype=float),
                    "master_face_ids": np.empty(0, dtype=np.int64),
                    "secondary_cache_indices": np.empty(0, dtype=np.int64),
                    "secondary_node_ids": np.empty(0, dtype=np.int64),
                    "tracking_cache_hits": np.empty(0, dtype=bool),
                    "tracking_cache_matches": np.empty(0, dtype=bool),
                    "tracking_barycentric_distances": np.empty(0, dtype=float),
                }
            expanded_sample_nodes = sample_nodes[source_rows].copy()
            if base_mode == "slave_node":
                expanded_sample_weights = np.vstack(one_hot_rows).astype(float, copy=False)
            else:
                expanded_sample_weights = sample_weights[source_rows].copy()
            expanded_gaps = gaps[source_rows]
            expanded_normals = normals[source_rows]
            local_weights = np.maximum(sample_weights[source_rows, source_locals], 0.0)
            expanded_areas = areas[source_rows] * local_weights
            expanded_master_nodes = master_nodes[source_rows].copy()
            expanded_master_weights = master_weights[source_rows].copy()
            expanded_master_barycentric = None if master_barycentric is None else master_barycentric[source_rows].copy()
            expanded_master_face_ids = None if master_face_ids is None else master_face_ids[source_rows].copy()
            expanded_secondary_cache_indices = (
                None if secondary_cache_indices is None else secondary_cache_indices[source_rows].copy()
            )
            expanded_tracking_cache_hits = None if tracking_cache_hits is None else tracking_cache_hits[source_rows].copy()
            expanded_tracking_cache_matches = (
                None if tracking_cache_matches is None else tracking_cache_matches[source_rows].copy()
            )
            expanded_tracking_barycentric_distances = (
                None if tracking_barycentric_distances is None else tracking_barycentric_distances[source_rows].copy()
            )
            if workspace is not None:
                workspace.misses += 1
                workspace.store(
                    mode=str(mode),
                    base_mode=base_mode,
                    sample_nodes=sample_nodes,
                    sample_weights=sample_weights,
                    group_indices=group_indices,
                    group_secondary_node_ids=group_secondary_node_ids,
                    source_rows=source_rows,
                    source_locals=source_locals,
                    slave_node_one_hot=(
                        expanded_sample_weights if base_mode == "slave_node" else None
                    ),
                )
        if expanded_gaps.size == 0:
            return {
                "sample_node_ids": np.empty((0, sample_nodes.shape[1]), dtype=np.int64),
                "sample_weights": np.empty((0, sample_weights.shape[1]), dtype=float),
                "gaps": np.empty(0, dtype=float),
                "normals": np.empty((0, 3), dtype=float),
                "areas": np.empty(0, dtype=float),
                "master_node_ids": np.empty((0, master_nodes.shape[1]), dtype=np.int64),
                "master_weights": np.empty((0, master_weights.shape[1]), dtype=float),
                "master_barycentric": np.empty((0, 3), dtype=float),
                "master_face_ids": np.empty(0, dtype=np.int64),
                "secondary_cache_indices": np.empty(0, dtype=np.int64),
                "secondary_node_ids": np.empty(0, dtype=np.int64),
                "tracking_cache_hits": np.empty(0, dtype=bool),
                "tracking_cache_matches": np.empty(0, dtype=bool),
                "tracking_barycentric_distances": np.empty(0, dtype=float),
            }
    if group_secondary_node_ids is None:
        group_secondary_node_ids = np.full(len(group_indices), -1, dtype=np.int64)
    aggregated = [
        _aggregate_contact_array_group(
            rows,
            sample_nodes=expanded_sample_nodes,
            sample_weights=expanded_sample_weights,
            gaps=expanded_gaps,
            normals=expanded_normals,
            areas=expanded_areas,
            master_nodes=expanded_master_nodes,
            master_weights=expanded_master_weights,
            master_barycentric=expanded_master_barycentric,
            master_face_ids=expanded_master_face_ids,
            secondary_cache_indices=expanded_secondary_cache_indices,
            secondary_node_id=int(group_secondary_node_ids[int(group_index)]),
            tracking_cache_hits=expanded_tracking_cache_hits,
            tracking_cache_matches=expanded_tracking_cache_matches,
            tracking_barycentric_distances=expanded_tracking_barycentric_distances,
            overclosure_mode=overclosure_mode,
        )
        for group_index, rows in enumerate(group_indices)
    ]
    return _pack_aggregated_contact_rows(aggregated)


def _aggregate_contact_sample_group(
    samples: list[ContactSample],
    *,
    overclosure_mode: str = "positive_integral",
) -> ContactSample:
    """Area-average contact samples into one surface-to-surface constraint.

    ``positive_integral`` preserves the positive overclosure integral used by
    earlier SFC diagnostics and uses pressure-weighted normals/kinematics in
    active regions.  ``positive_integral_area_average`` keeps the same
    integrated overclosure but uses area-averaged normals and shape weights,
    matching the idea that a surface-to-surface constraint region has its own
    average normal/support rather than a pressure-peak-biased direction.
    ``signed_average`` follows Abaqus/Standard's
    surface-to-surface wording more closely: the constraint value is considered
    in an average sense over a finite secondary-surface region, so the signed
    clearance is averaged before the pressure-overclosure law decides whether
    that region is active.
    ``linear_penalty_participation`` preserves the virtual-work distribution of
    a set of linear pressure-overclosure quadrature samples: the equivalent
    constraint force is assembled from the same positive pressure weights, but
    the area and gap are chosen so that the aggregate contact force and penalty
    energy match the underlying sample set when the local normal is constant.
    ``signed_status_linear_penalty_participation`` uses the same participation
    force distribution only after the signed area-average region clearance is
    closed.  This mirrors the Abaqus surface-to-surface idea that a constraint
    region has one open/closed status instead of independent pointwise pressure
    at every quadrature sample.
    """

    if not samples:
        raise ValueError("samples must not be empty")
    mode = str(overclosure_mode).lower()
    if mode not in {
        "positive_integral",
        "positive_integral_area_average",
        "signed_average",
        "linear_penalty_participation",
        "signed_status_linear_penalty_participation",
    }:
        raise ValueError(
            "overclosure_mode must be 'positive_integral', 'positive_integral_area_average', "
            "'signed_average', 'linear_penalty_participation', or "
            "'signed_status_linear_penalty_participation'"
        )
    areas = np.asarray([max(float(sample.area), 0.0) for sample in samples], dtype=float)
    total_area = float(np.sum(areas))
    if total_area <= 0.0:
        group_weights = np.full(len(samples), 1.0 / float(len(samples)), dtype=float)
        total_area = float(len(samples))
    else:
        group_weights = areas / total_area
    gaps = np.asarray([float(sample.gap) for sample in samples], dtype=float)
    aggregate_area = total_area
    if mode == "signed_average":
        constraint_gap = float(group_weights @ gaps)
        normal_weights = group_weights
        kinematic_weights = group_weights
    elif mode == "signed_status_linear_penalty_participation":
        signed_gap = float(group_weights @ gaps)
        if signed_gap >= 0.0:
            constraint_gap = signed_gap
            normal_weights = group_weights
            kinematic_weights = group_weights
        else:
            penetrations = np.maximum(-gaps, 0.0)
            penetration_integral = float(areas @ penetrations)
            if penetration_integral > 0.0:
                energy_integral = float(areas @ (penetrations * penetrations))
                if energy_integral > 0.0:
                    aggregate_area = max((penetration_integral * penetration_integral) / energy_integral, 1.0e-30)
                    constraint_gap = -penetration_integral / aggregate_area
                else:
                    aggregate_area = total_area
                    constraint_gap = signed_gap
                normal_weights = areas * penetrations / penetration_integral
                kinematic_weights = normal_weights
            else:
                constraint_gap = signed_gap
                normal_weights = group_weights
                kinematic_weights = group_weights
    else:
        penetrations = np.maximum(-gaps, 0.0)
        penetration_integral = float(areas @ penetrations)
        if penetration_integral > 0.0:
            if mode == "linear_penalty_participation":
                energy_integral = float(areas @ (penetrations * penetrations))
                if energy_integral > 0.0:
                    aggregate_area = max((penetration_integral * penetration_integral) / energy_integral, 1.0e-30)
                    constraint_gap = -penetration_integral / aggregate_area
                else:
                    aggregate_area = total_area
                    constraint_gap = -penetration_integral / max(total_area, 1.0e-30)
                normal_weights = areas * penetrations / penetration_integral
                kinematic_weights = normal_weights
            else:
                constraint_gap = -penetration_integral / max(total_area, 1.0e-30)
            if mode == "positive_integral_area_average":
                normal_weights = group_weights
                kinematic_weights = group_weights
            elif mode == "positive_integral":
                normal_weights = areas * penetrations / penetration_integral
                kinematic_weights = normal_weights
        else:
            constraint_gap = float(group_weights @ gaps)
            normal_weights = group_weights
            kinematic_weights = group_weights
    normal = np.sum(
        np.asarray([normal_weights[idx] * np.asarray(sample.normal, dtype=float) for idx, sample in enumerate(samples)], dtype=float),
        axis=0,
    )
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0.0:
        normal = np.asarray(samples[0].normal, dtype=float).copy()
        normal_norm = max(float(np.linalg.norm(normal)), 1.0e-30)
    normal = normal / normal_norm
    slave_nodes, slave_weights = _combined_shape_weights(
        [np.asarray(sample.node_ids, dtype=np.int64) for sample in samples],
        [np.asarray(sample.shape_weights, dtype=float) for sample in samples],
        kinematic_weights,
    )
    master_node_rows: list[np.ndarray] = []
    master_weight_rows: list[np.ndarray] = []
    for sample in samples:
        if sample.master_node_ids is not None and sample.master_shape_weights is not None:
            master_node_rows.append(np.asarray(sample.master_node_ids, dtype=np.int64))
            master_weight_rows.append(np.asarray(sample.master_shape_weights, dtype=float))
    master_nodes: np.ndarray | None = None
    master_weights: np.ndarray | None = None
    if len(master_node_rows) == len(samples):
        master_nodes, master_weights = _combined_shape_weights(master_node_rows, master_weight_rows, kinematic_weights)
    return ContactSample(
        node_ids=slave_nodes,
        shape_weights=slave_weights,
        gap=float(constraint_gap),
        normal=normal,
        area=float(aggregate_area),
        stiffness=float(samples[0].stiffness),
        master_node_ids=master_nodes,
        master_shape_weights=master_weights,
    )


def _split_contact_sample_to_slave_nodes(sample: ContactSample) -> list[ContactSample]:
    """Split one quadrature contact sample into slave-node tributary regions."""

    nodes = np.asarray(sample.node_ids, dtype=np.int64).reshape(-1)
    weights = np.asarray(sample.shape_weights, dtype=float).reshape(-1)
    if nodes.size != weights.size:
        raise ValueError("sample node_ids and shape_weights must have the same length")
    out: list[ContactSample] = []
    for local, (node, weight) in enumerate(zip(nodes, weights, strict=True)):
        tributary = float(sample.area) * max(float(weight), 0.0)
        if tributary <= 0.0:
            continue
        shape = np.zeros_like(weights, dtype=float)
        shape[int(local)] = 1.0
        out.append(
            ContactSample(
                node_ids=nodes.copy(),
                shape_weights=shape,
                gap=float(sample.gap),
                normal=np.asarray(sample.normal, dtype=float).copy(),
                area=tributary,
                stiffness=float(sample.stiffness),
                master_node_ids=(
                    None
                    if sample.master_node_ids is None
                    else np.asarray(sample.master_node_ids, dtype=np.int64).copy()
                ),
                master_shape_weights=(
                    None
                    if sample.master_shape_weights is None
                    else np.asarray(sample.master_shape_weights, dtype=float).copy()
                ),
            )
        )
    return out


def _tribute_contact_sample_to_slave_node_regions(sample: ContactSample) -> list[tuple[int, ContactSample]]:
    """Assign one sample to nodal constraint regions without collapsing shape support.

    Abaqus surface-to-surface contact forms nodal constraint regions on the
    slave side, but the virtual-work contribution of such a region is still
    based on the surrounding surface interpolation.  This helper therefore uses
    each slave shape function only as the tributary area selector; the retained
    contact sample keeps the original slave shape weights.  Compared with the
    legacy ``slave_node`` split, this avoids concentrating the full regional
    pressure onto one slave node while preserving the same integrated pressure
    overclosure law.
    """

    nodes = np.asarray(sample.node_ids, dtype=np.int64).reshape(-1)
    weights = np.asarray(sample.shape_weights, dtype=float).reshape(-1)
    if nodes.size != weights.size:
        raise ValueError("sample node_ids and shape_weights must have the same length")
    out: list[tuple[int, ContactSample]] = []
    for node, weight in zip(nodes, weights, strict=True):
        tributary = float(sample.area) * max(float(weight), 0.0)
        if tributary <= 0.0:
            continue
        out.append(
            (
                int(node),
                ContactSample(
                    node_ids=nodes.copy(),
                    shape_weights=weights.copy(),
                    gap=float(sample.gap),
                    normal=np.asarray(sample.normal, dtype=float).copy(),
                    area=tributary,
                    stiffness=float(sample.stiffness),
                    master_node_ids=(
                        None
                        if sample.master_node_ids is None
                        else np.asarray(sample.master_node_ids, dtype=np.int64).copy()
                    ),
                    master_shape_weights=(
                        None
                        if sample.master_shape_weights is None
                        else np.asarray(sample.master_shape_weights, dtype=float).copy()
                    ),
                ),
            )
        )
    return out


def _aggregate_contact_samples(samples: list[ContactSample], mode: str) -> list[ContactSample]:
    """Aggregate contact samples using Abaqus-style surface constraint regions."""

    base_mode, overclosure_mode = _contact_averaging_modes(mode)
    if base_mode == "none" or not samples:
        return samples
    if base_mode == "slave_face":
        groups: dict[tuple[int, ...], list[ContactSample]] = {}
        order: list[tuple[int, ...]] = []
        for sample in samples:
            key = tuple(int(v) for v in np.asarray(sample.node_ids, dtype=np.int64).reshape(-1))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(sample)
        return [_aggregate_contact_sample_group(groups[key], overclosure_mode=overclosure_mode) for key in order]
    if base_mode == "slave_node":
        groups: dict[int, list[ContactSample]] = {}
        order: list[int] = []
        for sample in samples:
            for split in _split_contact_sample_to_slave_nodes(sample):
                active_node = int(np.asarray(split.node_ids, dtype=np.int64)[np.argmax(split.shape_weights)])
                if active_node not in groups:
                    groups[active_node] = []
                    order.append(active_node)
                groups[active_node].append(split)
        return [_aggregate_contact_sample_group(groups[key], overclosure_mode=overclosure_mode) for key in order]
    if base_mode == "slave_node_region":
        groups: dict[int, list[ContactSample]] = {}
        order: list[int] = []
        for sample in samples:
            for active_node, tributary_sample in _tribute_contact_sample_to_slave_node_regions(sample):
                if active_node not in groups:
                    groups[active_node] = []
                    order.append(active_node)
                groups[active_node].append(tributary_sample)
        return [_aggregate_contact_sample_group(groups[key], overclosure_mode=overclosure_mode) for key in order]
    groups = _sample_connected_components(samples)
    return [
        _aggregate_contact_sample_group(
            [samples[int(index)] for index in group],
            overclosure_mode=overclosure_mode,
        )
        for group in groups
    ]


def _filter_contact_samples_by_normal_compatibility(
    samples: list[ContactSample],
    x_current: np.ndarray,
    *,
    mode: str,
    dot_threshold: float = 0.0,
) -> list[ContactSample]:
    """Filter contact samples by master/slave surface normal compatibility.

    Abaqus-style surface-to-surface normal contact constrains opposing surface
    sides.  On a curved gear flank, the closest Euclidean triangle can be a
    side/back facet whose normal is not facing the slave facet; accepting such a
    sample creates artificial overclosure without changing the SDF query itself.
    The default ``opposing`` mode keeps samples with
    ``n_master dot n_slave <= 0``.  This is a candidate-compatibility test only:
    the retained samples still use the same closest-feature gap and normal.
    """

    label = str(mode).lower()
    if label == "none":
        return samples
    if label not in {"opposing"}:
        raise ValueError("normal compatibility mode must be 'none' or 'opposing'")
    X = np.asarray(x_current, dtype=float)
    filtered: list[ContactSample] = []
    for sample in samples:
        nodes = np.asarray(sample.node_ids, dtype=np.int64).reshape(-1)
        if nodes.size < 3 or np.any(nodes < 0) or int(nodes.max()) >= X.shape[0]:
            continue
        tri = X[nodes[:3]]
        slave_normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        slave_norm = float(np.linalg.norm(slave_normal))
        if slave_norm <= 0.0:
            continue
        slave_normal /= slave_norm
        master_normal = np.asarray(sample.normal, dtype=float).reshape(3)
        master_norm = float(np.linalg.norm(master_normal))
        if master_norm <= 0.0:
            continue
        master_normal /= master_norm
        if float(master_normal @ slave_normal) <= float(dot_threshold):
            filtered.append(sample)
    return filtered


def _sample_secondary_normal(sample: ContactSample, x_current: np.ndarray) -> np.ndarray | None:
    """Return the current slave-surface normal for one triangular sample."""

    X = np.asarray(x_current, dtype=float)
    nodes = np.asarray(sample.node_ids, dtype=np.int64).reshape(-1)
    if nodes.size < 3 or np.any(nodes < 0) or int(nodes.max()) >= X.shape[0]:
        return None
    tri = X[nodes[:3]]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        return None
    normal = normal / norm
    master_normal = np.asarray(sample.normal, dtype=float).reshape(3)
    master_norm = float(np.linalg.norm(master_normal))
    if master_norm > 0.0 and float(normal @ (master_normal / master_norm)) < 0.0:
        normal = -normal
    return normal


def _replace_contact_normals_with_secondary_average(
    samples: list[ContactSample],
    x_current: np.ndarray,
    *,
    project_gap_to_secondary_plane: bool = False,
) -> list[ContactSample]:
    """Use secondary-surface normals for Abaqus-style surface-to-surface contact.

    Abaqus/Standard defines the surface-to-surface contact direction from an
    average normal of the secondary-surface region.  The SDF closest-feature
    normal is still used for candidate selection and overclosure sign, but the
    force/Jacobian direction should follow the secondary side when this mode is
    requested.
    """

    replaced: list[ContactSample] = []
    for sample in samples:
        normal = _sample_secondary_normal(sample, x_current)
        if normal is None:
            replaced.append(sample)
            continue
        master_normal = np.asarray(sample.normal, dtype=float).reshape(3)
        master_norm = float(np.linalg.norm(master_normal))
        if bool(project_gap_to_secondary_plane) and master_norm > 0.0:
            # The closest-feature SDF gap is measured along the master normal.
            # Abaqus surface-to-surface constraints use the secondary-surface
            # normal direction, so convert the local plane clearance to that
            # direction by the normal projection factor.
            projection = abs(float((master_normal / master_norm) @ normal))
            gap = float(sample.gap) / max(projection, 1.0e-12)
        else:
            gap = float(sample.gap)
        replaced.append(
            ContactSample(
                node_ids=np.asarray(sample.node_ids, dtype=np.int64).copy(),
                shape_weights=np.asarray(sample.shape_weights, dtype=float).copy(),
                gap=gap,
                normal=normal,
                area=float(sample.area),
                stiffness=float(sample.stiffness),
                master_node_ids=(
                    None
                    if sample.master_node_ids is None
                    else np.asarray(sample.master_node_ids, dtype=np.int64).copy()
                ),
                master_shape_weights=(
                    None
                    if sample.master_shape_weights is None
                    else np.asarray(sample.master_shape_weights, dtype=float).copy()
                ),
            )
        )
    return replaced


def _empty_contact_node_diagnostics(n_nodes: int) -> dict[str, Any]:
    """Return zero-valued contact fields for VTK/manifest diagnostics."""

    count = int(n_nodes)
    fields = {
        "contact_pressure_nodeavg": np.zeros(count, dtype=float),
        "contact_penetration_nodeavg": np.zeros(count, dtype=float),
        "contact_active_node": np.zeros(count, dtype=float),
        "contact_gap_min_node": np.zeros(count, dtype=float),
        "contact_sample_area_weight": np.zeros(count, dtype=float),
    }
    metrics: Row = {
        "active_contact_node_count": 0,
        "max_contact_pressure_nodeavg": 0.0,
        "p95_contact_pressure_nodeavg": 0.0,
        "mean_active_contact_pressure_nodeavg": 0.0,
        "max_contact_penetration_nodeavg": 0.0,
        "min_contact_gap_node": 0.0,
    }
    for role in ("slave", "master", "secondary"):
        fields.update(
            {
                f"contact_{role}_pressure_nodeavg": np.zeros(count, dtype=float),
                f"contact_{role}_penetration_nodeavg": np.zeros(count, dtype=float),
                f"contact_{role}_active_node": np.zeros(count, dtype=float),
                f"contact_{role}_gap_min_node": np.zeros(count, dtype=float),
                f"contact_{role}_sample_area_weight": np.zeros(count, dtype=float),
            }
        )
        metrics.update(
            {
                f"active_contact_{role}_node_count": 0,
                f"max_contact_{role}_pressure_nodeavg": 0.0,
                f"p95_contact_{role}_pressure_nodeavg": 0.0,
                f"mean_active_contact_{role}_pressure_nodeavg": 0.0,
                f"max_contact_{role}_penetration_nodeavg": 0.0,
                f"min_contact_{role}_gap_node": 0.0,
            }
        )
    return {"fields": fields, "metrics": metrics}


def _prefixed_contact_node_diagnostics(diagnostics: dict[str, Any], role: str) -> dict[str, Any]:
    """Rename total contact diagnostic fields/metrics for one contact side."""

    label = str(role)
    fields_in = dict(diagnostics.get("fields", {}))
    metrics_in = dict(diagnostics.get("metrics", {}))
    field_map = {
        "contact_pressure_nodeavg": f"contact_{label}_pressure_nodeavg",
        "contact_penetration_nodeavg": f"contact_{label}_penetration_nodeavg",
        "contact_active_node": f"contact_{label}_active_node",
        "contact_gap_min_node": f"contact_{label}_gap_min_node",
        "contact_sample_area_weight": f"contact_{label}_sample_area_weight",
    }
    metric_map = {
        "active_contact_node_count": f"active_contact_{label}_node_count",
        "max_contact_pressure_nodeavg": f"max_contact_{label}_pressure_nodeavg",
        "p95_contact_pressure_nodeavg": f"p95_contact_{label}_pressure_nodeavg",
        "mean_active_contact_pressure_nodeavg": f"mean_active_contact_{label}_pressure_nodeavg",
        "max_contact_penetration_nodeavg": f"max_contact_{label}_penetration_nodeavg",
        "min_contact_gap_node": f"min_contact_{label}_gap_node",
    }
    return {
        "fields": {new: fields_in[old] for old, new in field_map.items() if old in fields_in},
        "metrics": {new: metrics_in[old] for old, new in metric_map.items() if old in metrics_in},
    }


def _secondary_contact_node_aliases(slave_diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Expose slave-side pressure recovery under Abaqus secondary-surface names.

    Abaqus reports CPRESS/COPEN on the secondary side for a contact pair.  SFC
    keeps the older ``slave`` names for backward compatibility and mirrors the
    same arrays/metrics under ``secondary`` for direct manifest comparison.
    This is a fallback for non-region contact paths; Abaqus-style
    surface-to-surface evidence must use ``constraint_region`` recovery.
    """

    fields: dict[str, Any] = {}
    for key, value in dict(slave_diagnostics.get("fields", {})).items():
        if "contact_slave_" in key:
            fields[key.replace("contact_slave_", "contact_secondary_")] = value
    metrics: Row = {}
    for key, value in dict(slave_diagnostics.get("metrics", {})).items():
        if "_contact_slave_" in key:
            metrics[key.replace("_contact_slave_", "_contact_secondary_")] = value
        elif key.startswith("active_contact_slave_"):
            metrics[key.replace("active_contact_slave_", "active_contact_secondary_")] = value
    metrics["contact_secondary_pressure_recovery_source"] = "slave_sample_alias"
    return {"fields": fields, "metrics": metrics}


def _secondary_region_contact_node_diagnostics_from_arrays(
    sample_arrays: dict[str, np.ndarray],
    n_nodes: int,
    *,
    stiffness: float,
) -> dict[str, Any] | None:
    """Recover Abaqus-style secondary-node pressure from region constraints.

    Aggregated ``slave_node_region_*`` contact arrays carry one
    ``secondary_node_id`` per region.  CPRESS/COPEN should be reported at that
    secondary node from the region-average overclosure, not redistributed over
    every node participating in the region shape weights.
    """

    return _core_secondary_node_pressure_recovery_from_regions(
        sample_arrays,
        n_nodes=int(n_nodes),
        pressure_stiffness=float(stiffness),
    )


def _promote_secondary_contact_diagnostics_to_legacy(diagnostics: dict[str, Any]) -> None:
    """Use secondary-region recovery as the legacy CPRESS/COPEN alias.

    Abaqus contact output for a contact pair is reported on the secondary
    surface.  SFC keeps ``contact_slave_*`` and ``contact_master_*`` fields for
    diagnostics, but the long-standing unqualified ``contact_pressure_nodeavg``
    compatibility fields should therefore mirror the secondary constraint
    region, not the slave+master scatter total.
    """

    fields = diagnostics.setdefault("fields", {})
    for old, new in (
        ("contact_secondary_pressure_nodeavg", "contact_pressure_nodeavg"),
        ("contact_secondary_penetration_nodeavg", "contact_penetration_nodeavg"),
        ("contact_secondary_active_node", "contact_active_node"),
        ("contact_secondary_gap_min_node", "contact_gap_min_node"),
        ("contact_secondary_sample_area_weight", "contact_sample_area_weight"),
    ):
        if old in fields:
            fields[new] = np.asarray(fields[old], dtype=float).copy()
    metrics = diagnostics.setdefault("metrics", {})
    for old, new in (
        ("active_contact_secondary_node_count", "active_contact_node_count"),
        ("max_contact_secondary_pressure_nodeavg", "max_contact_pressure_nodeavg"),
        ("p95_contact_secondary_pressure_nodeavg", "p95_contact_pressure_nodeavg"),
        ("mean_active_contact_secondary_pressure_nodeavg", "mean_active_contact_pressure_nodeavg"),
        ("max_contact_secondary_penetration_nodeavg", "max_contact_penetration_nodeavg"),
        ("min_contact_secondary_gap_node", "min_contact_gap_node"),
    ):
        if old in metrics:
            metrics[new] = metrics[old]
    if "contact_secondary_pressure_recovery_source" in metrics:
        metrics["contact_pressure_recovery_source"] = metrics["contact_secondary_pressure_recovery_source"]


def _new_contact_accumulators(n_nodes: int) -> dict[str, np.ndarray]:
    count = int(n_nodes)
    return {
        "weighted_pressure": np.zeros(count, dtype=float),
        "weighted_penetration": np.zeros(count, dtype=float),
        "area_weight": np.zeros(count, dtype=float),
        "active_hit": np.zeros(count, dtype=float),
        "gap_min": np.full(count, np.inf, dtype=float),
    }


def _finalize_contact_accumulators(accumulators: dict[str, np.ndarray], n_nodes: int) -> dict[str, Any]:
    return _finalize_contact_node_diagnostics(
        n_nodes=int(n_nodes),
        weighted_pressure=accumulators["weighted_pressure"],
        weighted_penetration=accumulators["weighted_penetration"],
        area_weight=accumulators["area_weight"],
        active_hit=accumulators["active_hit"],
        gap_min=accumulators["gap_min"],
    )


def _finalize_contact_node_diagnostics(
    *,
    n_nodes: int,
    weighted_pressure: np.ndarray,
    weighted_penetration: np.ndarray,
    area_weight: np.ndarray,
    active_hit: np.ndarray,
    gap_min: np.ndarray,
) -> dict[str, Any]:
    """Convert accumulated sample contact quantities to nodal diagnostic fields."""

    count = int(n_nodes)
    if count == 0:
        return _empty_contact_node_diagnostics(0)
    pressure = np.zeros(count, dtype=float)
    penetration = np.zeros(count, dtype=float)
    nonzero = area_weight > 0.0
    pressure[nonzero] = weighted_pressure[nonzero] / area_weight[nonzero]
    penetration[nonzero] = weighted_penetration[nonzero] / area_weight[nonzero]
    active_node = (active_hit > 0.0).astype(float)
    finite_gap = np.isfinite(gap_min)
    gap = np.zeros(count, dtype=float)
    gap[finite_gap] = gap_min[finite_gap]
    active_pressure = pressure[active_node > 0.0]
    fields = {
        "contact_pressure_nodeavg": pressure,
        "contact_penetration_nodeavg": penetration,
        "contact_active_node": active_node,
        "contact_gap_min_node": gap,
        "contact_sample_area_weight": area_weight.copy(),
    }
    metrics: Row = {
        "active_contact_node_count": int(np.count_nonzero(active_node)),
        "max_contact_pressure_nodeavg": float(np.max(active_pressure)) if active_pressure.size else 0.0,
        "p95_contact_pressure_nodeavg": _percentile_or_zero(active_pressure, 95.0),
        "mean_active_contact_pressure_nodeavg": float(np.mean(active_pressure)) if active_pressure.size else 0.0,
        "max_contact_penetration_nodeavg": float(np.max(penetration)) if penetration.size else 0.0,
        "min_contact_gap_node": float(np.min(gap[finite_gap])) if np.any(finite_gap) else 0.0,
    }
    return {"fields": fields, "metrics": metrics}


def _accumulate_contact_nodes(
    *,
    nodes: np.ndarray,
    weights: np.ndarray,
    gaps: np.ndarray,
    pressures: np.ndarray,
    penetrations: np.ndarray,
    areas: np.ndarray,
    active: np.ndarray,
    weighted_pressure: np.ndarray,
    weighted_penetration: np.ndarray,
    area_weight: np.ndarray,
    active_hit: np.ndarray,
    gap_min: np.ndarray,
) -> None:
    """Accumulate quadrature-sample contact diagnostics onto VTK nodes."""

    node_ids = np.asarray(nodes, dtype=np.int64)
    shape_weights = np.asarray(weights, dtype=float)
    if node_ids.size == 0 or shape_weights.size == 0:
        return
    if node_ids.shape != shape_weights.shape:
        raise ValueError("contact diagnostic node and weight arrays must have matching shapes")
    contrib = np.asarray(areas, dtype=float).reshape((-1, 1)) * shape_weights
    flat_nodes = node_ids.reshape(-1)
    flat_contrib = contrib.reshape(-1)
    flat_pressure = (contrib * np.asarray(pressures, dtype=float).reshape((-1, 1))).reshape(-1)
    flat_penetration = (contrib * np.asarray(penetrations, dtype=float).reshape((-1, 1))).reshape(-1)
    flat_active = (shape_weights * np.asarray(active, dtype=bool).reshape((-1, 1))).reshape(-1)
    flat_gaps = np.repeat(np.asarray(gaps, dtype=float).reshape(-1), node_ids.shape[1])
    np.add.at(weighted_pressure, flat_nodes, flat_pressure)
    np.add.at(weighted_penetration, flat_nodes, flat_penetration)
    np.add.at(area_weight, flat_nodes, flat_contrib)
    np.add.at(active_hit, flat_nodes, flat_active)
    np.minimum.at(gap_min, flat_nodes, flat_gaps)


def _contact_node_diagnostics_from_arrays(
    sample_arrays: dict[str, np.ndarray] | None,
    n_nodes: int,
    *,
    stiffness: float,
) -> dict[str, Any]:
    """Map batched contact samples to nodal pressure/active diagnostic fields."""

    if sample_arrays is None:
        return _empty_contact_node_diagnostics(n_nodes)
    gaps = np.asarray(sample_arrays.get("gaps", np.empty(0)), dtype=float).reshape(-1)
    count = int(n_nodes)
    if gaps.size == 0:
        return _empty_contact_node_diagnostics(count)
    areas = np.asarray(sample_arrays["areas"], dtype=float).reshape(-1)
    penetrations = np.maximum(-gaps, 0.0)
    pressures = float(stiffness) * penetrations
    active = penetrations > 0.0
    total = _new_contact_accumulators(count)
    slave = _new_contact_accumulators(count)
    master = _new_contact_accumulators(count)
    sample_nodes = np.asarray(sample_arrays["sample_node_ids"], dtype=np.int64)
    sample_weights = np.asarray(sample_arrays["sample_weights"], dtype=float)
    master_nodes = np.asarray(sample_arrays["master_node_ids"], dtype=np.int64)
    master_weights = np.asarray(sample_arrays["master_weights"], dtype=float)
    _accumulate_contact_nodes(
        nodes=sample_nodes,
        weights=sample_weights,
        gaps=gaps,
        pressures=pressures,
        penetrations=penetrations,
        areas=areas,
        active=active,
        **slave,
    )
    _accumulate_contact_nodes(
        nodes=master_nodes,
        weights=master_weights,
        gaps=gaps,
        pressures=pressures,
        penetrations=penetrations,
        areas=areas,
        active=active,
        **master,
    )
    for nodes, weights in ((sample_nodes, sample_weights), (master_nodes, master_weights)):
        _accumulate_contact_nodes(
            nodes=nodes,
            weights=weights,
            gaps=gaps,
            pressures=pressures,
            penetrations=penetrations,
            areas=areas,
            active=active,
            **total,
        )
    diagnostics = _finalize_contact_accumulators(total, count)
    slave_diag = _prefixed_contact_node_diagnostics(_finalize_contact_accumulators(slave, count), "slave")
    master_diag = _prefixed_contact_node_diagnostics(_finalize_contact_accumulators(master, count), "master")
    diagnostics["fields"].update(slave_diag["fields"])
    diagnostics["fields"].update(master_diag["fields"])
    diagnostics["metrics"].update(slave_diag["metrics"])
    diagnostics["metrics"].update(master_diag["metrics"])
    secondary_diag = _secondary_region_contact_node_diagnostics_from_arrays(
        sample_arrays,
        count,
        stiffness=stiffness,
    )
    if secondary_diag is None:
        secondary_diag = _secondary_contact_node_aliases(slave_diag)
    diagnostics["fields"].update(secondary_diag["fields"])
    diagnostics["metrics"].update(secondary_diag["metrics"])
    _promote_secondary_contact_diagnostics_to_legacy(diagnostics)
    return diagnostics


def _contact_node_diagnostics_from_samples(samples: Iterable[Any] | None, n_nodes: int) -> dict[str, Any]:
    """Map Python contact samples to nodal pressure/active diagnostic fields."""

    count = int(n_nodes)
    sample_list = list(samples or [])
    if not sample_list:
        return _empty_contact_node_diagnostics(count)
    total = _new_contact_accumulators(count)
    slave = _new_contact_accumulators(count)
    master = _new_contact_accumulators(count)
    for sample in sample_list:
        gap = float(sample.gap)
        penetration = max(-gap, 0.0)
        pressure = float(sample.stiffness) * penetration
        area = float(sample.area)
        active = penetration > 0.0
        slave_nodes = np.asarray(sample.node_ids, dtype=np.int64).reshape((1, -1))
        slave_weights = np.asarray(sample.shape_weights, dtype=float).reshape((1, -1))
        _accumulate_contact_nodes(
            nodes=slave_nodes,
            weights=slave_weights,
            gaps=np.asarray([gap], dtype=float),
            pressures=np.asarray([pressure], dtype=float),
            penetrations=np.asarray([penetration], dtype=float),
            areas=np.asarray([area], dtype=float),
            active=np.asarray([active], dtype=bool),
            **slave,
        )
        _accumulate_contact_nodes(
            nodes=slave_nodes,
            weights=slave_weights,
            gaps=np.asarray([gap], dtype=float),
            pressures=np.asarray([pressure], dtype=float),
            penetrations=np.asarray([penetration], dtype=float),
            areas=np.asarray([area], dtype=float),
            active=np.asarray([active], dtype=bool),
            **total,
        )
        master_ids = getattr(sample, "master_node_ids", None)
        master_weights = getattr(sample, "master_shape_weights", None)
        if master_ids is not None and master_weights is not None:
            master_nodes = np.asarray(master_ids, dtype=np.int64).reshape((1, -1))
            master_shape = np.asarray(master_weights, dtype=float).reshape((1, -1))
            _accumulate_contact_nodes(
                nodes=master_nodes,
                weights=master_shape,
                gaps=np.asarray([gap], dtype=float),
                pressures=np.asarray([pressure], dtype=float),
                penetrations=np.asarray([penetration], dtype=float),
                areas=np.asarray([area], dtype=float),
                active=np.asarray([active], dtype=bool),
                **master,
            )
            _accumulate_contact_nodes(
                nodes=master_nodes,
                weights=master_shape,
                gaps=np.asarray([gap], dtype=float),
                pressures=np.asarray([pressure], dtype=float),
                penetrations=np.asarray([penetration], dtype=float),
                areas=np.asarray([area], dtype=float),
                active=np.asarray([active], dtype=bool),
                **total,
            )
    diagnostics = _finalize_contact_accumulators(total, count)
    slave_diag = _prefixed_contact_node_diagnostics(_finalize_contact_accumulators(slave, count), "slave")
    master_diag = _prefixed_contact_node_diagnostics(_finalize_contact_accumulators(master, count), "master")
    diagnostics["fields"].update(slave_diag["fields"])
    diagnostics["fields"].update(master_diag["fields"])
    diagnostics["metrics"].update(slave_diag["metrics"])
    diagnostics["metrics"].update(master_diag["metrics"])
    secondary_diag = _secondary_contact_node_aliases(slave_diag)
    diagnostics["fields"].update(secondary_diag["fields"])
    diagnostics["metrics"].update(secondary_diag["metrics"])
    _promote_secondary_contact_diagnostics_to_legacy(diagnostics)
    return diagnostics


@dataclass(frozen=True, slots=True)
class CroppedGearPatch:
    name: str
    nodes: np.ndarray
    elements: np.ndarray
    contact_faces: np.ndarray
    support_nodes: np.ndarray
    surface_entries: tuple[tuple[int, str], ...]
    element_labels: np.ndarray
    rp: np.ndarray


@dataclass(frozen=True, slots=True)
class CroppedGearPair:
    gear1: CroppedGearPatch
    gear2: CroppedGearPatch
    drive_direction: np.ndarray
    initial_patch_gap: float


def _write_csv(path: Path, rows: list[Row], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _elastic_strain_norm_from_stress(stress: np.ndarray, *, young: float, poisson: float) -> np.ndarray:
    """Return small elastic strain norms recovered from Cauchy stress.

    Abaqus `LE` is the strain output used by the external reference deck.  The
    SFC material model stores Green strain for mechanics, so for validation
    diagnostics we recover the corresponding linear-elastic strain from the
    Cauchy stress before comparing to Abaqus `LE`.
    """

    values = np.asarray(stress, dtype=float)
    if values.size == 0:
        return np.empty(0, dtype=float)
    identity = np.eye(3)
    out = np.empty(values.shape[0], dtype=float)
    for idx, sigma in enumerate(values):
        trace = float(np.trace(sigma))
        eps = ((1.0 + float(poisson)) / float(young)) * sigma - (float(poisson) / float(young)) * trace * identity
        out[idx] = float(np.linalg.norm(eps))
    return out


def _equivalent_elastic_strain_from_mises(von_mises_values: np.ndarray, *, young: float, poisson: float) -> np.ndarray:
    vm = np.asarray(von_mises_values, dtype=float).reshape(-1)
    shear = float(young) / (2.0 * (1.0 + float(poisson)))
    return vm / max(3.0 * shear, 1.0e-30)


def _linear_reference_internal_response(
    model: MechanicsModel,
    x_current: np.ndarray,
    reference_tangent: Any,
) -> InternalResponse:
    """Return a fast small-strain TET4 response for linear-elastic decks."""

    displacement = np.asarray(x_current, dtype=float) - np.asarray(model.X, dtype=float)
    element_u = displacement[np.asarray(model.elements, dtype=np.int64)]
    grad_u = np.einsum("eai,eaj->eij", element_u, model.shape_grads, optimize=True)
    strain = 0.5 * (grad_u + np.swapaxes(grad_u, 1, 2))
    lam, mu = _lame_parameters_local(model.E, model.nu)
    identity = np.eye(3, dtype=float)
    trace = np.trace(strain, axis1=1, axis2=2)
    stress = lam * trace[:, None, None] * identity + 2.0 * mu * strain
    vm = _von_mises_local(stress)
    u_flat = displacement.reshape(-1)
    force = np.asarray(reference_tangent @ u_flat, dtype=float).reshape((-1, 3))
    energy = 0.5 * float(u_flat @ np.asarray(force, dtype=float).reshape(-1))
    empty = csr_matrix((model.n_dofs, model.n_dofs), dtype=float)
    return InternalResponse(force, strain, stress, vm, energy, empty, empty)


def _rotation_z_matrix(angle_rad: float) -> np.ndarray:
    """Return an active z-axis rotation matrix for an angle in radians."""

    c = float(np.cos(float(angle_rad)))
    s = float(np.sin(float(angle_rad)))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _source_drive_corotated_visual_state_and_internal(
    *,
    model: MechanicsModel,
    state: MechanicsState,
    reference_tangent: Any,
    body_node_slices: tuple[slice, slice],
    body_reference_points: tuple[np.ndarray, np.ndarray],
    body_rotation_z: tuple[float, float],
    stress_postprocess: str = "linear_corotated",
) -> tuple[MechanicsState, InternalResponse]:
    """Return finite-rotation visual state and objective elastic stress fields.

    The source Abaqus gear deck uses ``nlgeom=YES`` and RP rotations measured in
    radians.  A small-strain postprocess of the raw linearized BEAM-MPC
    displacement would incorrectly interpret the large rigid gear rotation as
    strain.  For visualization and field-curve metrics, remove the linearized
    rigid z-rotation from each body's reduced-coordinate displacement, place the
    body with the corresponding finite z-rotation, and compute stress/strain
    from the residual elastic field.  The default ``linear_corotated`` mode
    matches the current source-drive solve, which still uses a reference
    stiffness.  ``finite_stvk_visual`` is available as a diagnostic for the
    finite-rotation visual configuration.  Neither mode smooths, filters, or
    alters the contact solve.
    """

    x_visual, u_elastic = _source_drive_corotated_positions_and_elastic_displacement(
        model=model,
        x_raw=state.x,
        body_node_slices=body_node_slices,
        body_reference_points=body_reference_points,
        body_rotation_z=body_rotation_z,
    )
    visual_state = MechanicsState(
        x_visual,
        np.asarray(state.v, dtype=float).copy(),
        np.asarray(state.a, dtype=float).copy(),
        time=float(state.time),
    )
    mode = str(stress_postprocess).lower()
    if mode in {"linear_corotated", "corotated_body_elastic_residual", "linear"}:
        internal = _linear_reference_internal_response(model, np.asarray(model.X, dtype=float) + u_elastic, reference_tangent)
    elif mode in {"finite_stvk_visual", "stvk_visual"}:
        internal = stvk_internal_response(model, visual_state.x, assemble_tangent=False)
    else:
        raise ValueError("stress_postprocess must be 'linear_corotated' or 'finite_stvk_visual'")
    return visual_state, internal


def _source_drive_corotated_positions_and_elastic_displacement(
    *,
    model: MechanicsModel,
    x_raw: np.ndarray,
    body_node_slices: tuple[slice, slice],
    body_reference_points: tuple[np.ndarray, np.ndarray],
    body_rotation_z: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-RP-rotation positions and elastic residual displacements.

    The source gear deck uses Abaqus ``nlgeom=YES`` with BEAM-MPC RP rotations
    in radians.  The global source-drive solver still stores the RP rotations
    as reduced coordinates, so a large accumulated RP angle must not be used as
    a small-rotation displacement when evaluating contact geometry or visual
    stress fields.  This helper removes the small-rotation rigid component from
    each body, applies the finite z-rotation, and keeps the residual elastic
    displacement unchanged.  It changes only the kinematic map used by callers;
    final contact gaps are still computed by the Lagrangian-SDF closest-feature
    oracle on the resulting current surface.
    """

    X = np.asarray(model.X, dtype=float)
    x_raw_array = np.asarray(x_raw, dtype=float)
    if x_raw_array.shape != X.shape:
        raise ValueError("x_raw must have the same shape as model.X")
    u_raw = x_raw_array - X
    x_visual = x_raw_array.copy()
    u_elastic = u_raw.copy()
    theta_vectors = (
        np.asarray([0.0, 0.0, float(body_rotation_z[0])], dtype=float),
        np.asarray([0.0, 0.0, float(body_rotation_z[1])], dtype=float),
    )
    for node_slice, rp, theta_vec in zip(body_node_slices, body_reference_points, theta_vectors, strict=True):
        ids = np.arange(node_slice.start or 0, node_slice.stop, dtype=np.int64)
        if ids.size == 0:
            continue
        reference_point = np.asarray(rp, dtype=float).reshape(3)
        lever = X[ids] - reference_point[None, :]
        small_rigid = np.cross(theta_vec[None, :], lever)
        elastic = u_raw[ids] - small_rigid
        R = _rotation_z_matrix(float(theta_vec[2]))
        x_visual[ids] = reference_point[None, :] + lever @ R.T + elastic
        u_elastic[ids] = elastic
    return x_visual, u_elastic


def _source_drive_corotated_elastic_matrix(
    *,
    assembly: Any,
    reference_nodes: np.ndarray,
    body_node_slices: tuple[slice, slice],
    body_reference_points: tuple[np.ndarray, np.ndarray],
) -> csr_matrix:
    """Return the reduced map from RP/free coordinates to elastic residuals.

    The linear BEAM-MPC map ``T`` expands reduced coordinates to raw nodal
    displacements.  For a source gear step with large RP z-rotations, the
    elastic residual used by the corotated stress/contact path is

    ``u_elastic = T q - theta_z (e_z x (X - X_RP))``

    on every node of each body.  This sparse map is the tangent counterpart of
    :func:`_source_drive_corotated_positions_and_elastic_displacement` and is
    used only when explicitly requested by the source-drive runner.
    """

    nodes = np.asarray(reference_nodes, dtype=float)
    elastic = assembly.transformation.tolil(copy=True)
    for body_index, (node_slice, rp) in enumerate(zip(body_node_slices, body_reference_points, strict=True)):
        ids = np.arange(node_slice.start or 0, node_slice.stop, dtype=np.int64)
        if ids.size == 0:
            continue
        col = int(assembly.hub_slice(body_index).start + 5)
        reference_point = np.asarray(rp, dtype=float).reshape(3)
        lever = nodes[ids] - reference_point[None, :]
        small_z = np.column_stack((-lever[:, 1], lever[:, 0], np.zeros(ids.size, dtype=float)))
        for local, node in enumerate(ids):
            for axis in range(3):
                value = float(small_z[local, axis])
                if value != 0.0:
                    elastic[3 * int(node) + axis, col] = float(elastic[3 * int(node) + axis, col]) - value
    return elastic.tocsr()


def _source_drive_finite_visual_jacobian(
    *,
    assembly: Any,
    reference_nodes: np.ndarray,
    body_node_slices: tuple[slice, slice],
    body_reference_points: tuple[np.ndarray, np.ndarray],
    body_rotation_z: tuple[float, float],
) -> csr_matrix:
    """Return ``d x_visual / d q`` for the finite-RP visual map.

    The source Abaqus deck uses ``nlgeom=YES`` with RP rotations in radians.
    For finite-rotation internal-force diagnostics the reduced residual must
    be the virtual-work projection ``J_visual.T @ f_int(x_visual)`` rather
    than a projection through the small-rotation BEAM-MPC matrix.  This helper
    is the exact sparse Jacobian of
    :func:`_source_drive_corotated_positions_and_elastic_displacement` with
    respect to the reduced coordinates for the z-rotation path implemented by
    the source gear runner.
    """

    nodes = np.asarray(reference_nodes, dtype=float)
    visual = _source_drive_corotated_elastic_matrix(
        assembly=assembly,
        reference_nodes=nodes,
        body_node_slices=body_node_slices,
        body_reference_points=body_reference_points,
    ).tolil(copy=True)
    for body_index, (node_slice, rp, theta) in enumerate(
        zip(body_node_slices, body_reference_points, body_rotation_z, strict=True)
    ):
        ids = np.arange(node_slice.start or 0, node_slice.stop, dtype=np.int64)
        if ids.size == 0:
            continue
        col = int(assembly.hub_slice(body_index).start + 5)
        reference_point = np.asarray(rp, dtype=float).reshape(3)
        lever = nodes[ids] - reference_point[None, :]
        current_lever = lever @ _rotation_z_matrix(float(theta)).T
        d_current_lever = np.column_stack((-current_lever[:, 1], current_lever[:, 0], np.zeros(ids.size, dtype=float)))
        for local, node in enumerate(ids):
            for axis in range(3):
                value = float(d_current_lever[local, axis])
                if value != 0.0:
                    visual[3 * int(node) + axis, col] = float(visual[3 * int(node) + axis, col]) + value
    return visual.tocsr()


def _source_drive_finite_kinematic_inertia_response(
    *,
    assembly: Any,
    mass_matrix: Any,
    reference_nodes: np.ndarray,
    body_node_slices: tuple[slice, slice],
    body_reference_points: tuple[np.ndarray, np.ndarray],
    body_rotation_z: tuple[float, float],
    body_angular_velocity_z: tuple[float, float],
    reduced_acceleration: np.ndarray,
    include_mass_tangent: bool = True,
) -> tuple[np.ndarray, csr_matrix]:
    """Return finite-kinematic RP-MPC inertia by reduced virtual work.

    For a nonlinear RP map ``x = x(q)``, the inertial virtual work is

    ``dq.T J(q).T M [J(q) qddot + Jdot(q, qdot) qdot]``.

    The source-gear map is nonlinear only in each body's z-rotation, so the
    ``Jdot qdot`` term is the finite-rotation radial acceleration already used
    by the centripetal diagnostic.  The returned tangent is the acceleration
    part ``J.T M J``.  It is intentionally only the mass contribution; callers
    scale it by the Newmark ``c0`` factor.
    """

    jacobian = _source_drive_finite_visual_jacobian(
        assembly=assembly,
        reference_nodes=reference_nodes,
        body_node_slices=body_node_slices,
        body_reference_points=body_reference_points,
        body_rotation_z=body_rotation_z,
    ).tocsr()
    acc_red = np.asarray(reduced_acceleration, dtype=float).reshape(-1)
    nodal_acc = np.asarray(jacobian @ acc_red, dtype=float).reshape((-1, 3))
    nodal_acc += _source_drive_centripetal_acceleration(
        reference_nodes=reference_nodes,
        body_node_slices=body_node_slices,
        body_reference_points=body_reference_points,
        body_rotation_z=body_rotation_z,
        body_angular_velocity_z=body_angular_velocity_z,
    )
    full_inertia = np.asarray(mass_matrix @ nodal_acc.reshape(-1), dtype=float).reshape(-1)
    reduced = np.asarray(jacobian.T @ full_inertia, dtype=float).reshape(-1)
    if not bool(include_mass_tangent):
        empty = csr_matrix((assembly.n_reduced_dofs, assembly.n_reduced_dofs), dtype=float)
        return reduced, empty
    mass_tangent = (jacobian.T @ mass_matrix @ jacobian).tocsr()
    return reduced, mass_tangent


def _source_drive_centripetal_acceleration(
    *,
    reference_nodes: np.ndarray,
    body_node_slices: tuple[slice, slice],
    body_reference_points: tuple[np.ndarray, np.ndarray],
    body_rotation_z: tuple[float, float],
    body_angular_velocity_z: tuple[float, float],
) -> np.ndarray:
    """Return finite-RP centripetal acceleration for each body node.

    This is the rotating-frame inertial term missing from a purely linearized
    RP-MPC displacement solve.  It is needed for high-speed ``nlgeom=YES`` gear
    comparisons because Abaqus carries the finite rigid rotation in the
    inertial dynamics, while a corotated elastic residual alone only removes
    the rigid rotation from strain.
    """

    X = np.asarray(reference_nodes, dtype=float)
    acceleration = np.zeros_like(X)
    for node_slice, rp, theta, omega_z in zip(
        body_node_slices,
        body_reference_points,
        body_rotation_z,
        body_angular_velocity_z,
        strict=True,
    ):
        ids = np.arange(node_slice.start or 0, node_slice.stop, dtype=np.int64)
        if ids.size == 0:
            continue
        reference_point = np.asarray(rp, dtype=float).reshape(3)
        lever = X[ids] - reference_point[None, :]
        R = _rotation_z_matrix(float(theta))
        current_lever = lever @ R.T
        omega = np.asarray([0.0, 0.0, float(omega_z)], dtype=float)
        acceleration[ids] = np.cross(omega[None, :], np.cross(omega[None, :], current_lever))
    return acceleration


def _source_drive_centripetal_reduced_response(
    *,
    assembly: Any,
    mass_matrix: Any,
    reference_nodes: np.ndarray,
    body_node_slices: tuple[slice, slice],
    body_reference_points: tuple[np.ndarray, np.ndarray],
    body_rotation_z: tuple[float, float],
    body_angular_velocity_z: tuple[float, float],
    velocity_sensitivity_z: tuple[float, float] = (0.0, 0.0),
    include_tangent: bool = False,
) -> tuple[np.ndarray, csr_matrix]:
    """Return reduced centripetal inertia residual and optional tangent.

    ``velocity_sensitivity_z`` contains ``d omega_z / d theta_z`` for each RP
    rotation coordinate in the current Newton iteration.  In Newmark/HHT this is
    ``gamma / (beta dt)`` for an unknown rotational displacement.  The tangent
    is one-column-per-body because this diagnostic model only uses z-axis RP
    rotations from the source gear deck.
    """

    acc = _source_drive_centripetal_acceleration(
        reference_nodes=reference_nodes,
        body_node_slices=body_node_slices,
        body_reference_points=body_reference_points,
        body_rotation_z=body_rotation_z,
        body_angular_velocity_z=body_angular_velocity_z,
    )
    reduced = assembly.reduce_vector(mass_matrix @ acc.reshape(-1))
    if not include_tangent:
        return reduced, csr_matrix((assembly.n_reduced_dofs, assembly.n_reduced_dofs), dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    X = np.asarray(reference_nodes, dtype=float)
    for body_index, (node_slice, rp, theta, omega_z, domega_dtheta) in enumerate(
        zip(
            body_node_slices,
            body_reference_points,
            body_rotation_z,
            body_angular_velocity_z,
            velocity_sensitivity_z,
            strict=True,
        )
    ):
        ids = np.arange(node_slice.start or 0, node_slice.stop, dtype=np.int64)
        if ids.size == 0:
            continue
        reference_point = np.asarray(rp, dtype=float).reshape(3)
        lever = X[ids] - reference_point[None, :]
        current_lever = lever @ _rotation_z_matrix(float(theta)).T
        d_current_lever = np.column_stack((-current_lever[:, 1], current_lever[:, 0], np.zeros(ids.size, dtype=float)))
        radial_lever = current_lever.copy()
        radial_lever[:, 2] = 0.0
        omega = float(omega_z)
        sensitivity = float(domega_dtheta)
        dacc = -2.0 * omega * sensitivity * radial_lever - omega * omega * d_current_lever
        full = np.zeros_like(X)
        full[ids] = dacc
        column = assembly.reduce_vector(mass_matrix @ full.reshape(-1))
        source_col = int(assembly.hub_slice(body_index).start + 5)
        nonzero = np.flatnonzero(np.abs(column) > 0.0)
        rows.extend(int(row) for row in nonzero)
        cols.extend(source_col for _ in nonzero)
        data.extend(float(column[int(row)]) for row in nonzero)
    tangent = coo_matrix(
        (data, (rows, cols)),
        shape=(assembly.n_reduced_dofs, assembly.n_reduced_dofs),
        dtype=float,
    ).tocsr()
    return reduced, tangent


def _write_sfc_tet4_vtk_frame(
    path: Path,
    *,
    model: MechanicsModel,
    state: MechanicsState,
    internal: InternalResponse,
    element_object_ids: np.ndarray,
    time_value: float,
    frame_index: int,
    include_tensors: bool = True,
    static_blocks: SFCVTKStaticBlocks | None = None,
    contact_node_diagnostics: dict[str, Any] | None = None,
) -> Row:
    """Write one SFC TET4 state as a legacy VTK unstructured grid."""

    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(state.x, dtype=float)
    reference = np.asarray(model.X, dtype=float)
    displacement = points - reference
    displacement_norm = np.linalg.norm(displacement, axis=1)
    velocity = np.asarray(state.v, dtype=float)
    static = static_blocks if static_blocks is not None else _build_sfc_tet4_vtk_static_blocks(model, element_object_ids)
    elements = static.elements
    object_ids = static.object_ids
    stress = _tensor_field_or_zeros(internal.stress, elements.shape[0])
    strain = _tensor_field_or_zeros(internal.strain, elements.shape[0])
    von_mises = np.asarray(internal.von_mises, dtype=float).reshape(-1)
    if von_mises.shape != (elements.shape[0],):
        von_mises = _von_mises_local(stress)
    strain_norm = np.linalg.norm(strain, axis=(1, 2))
    equivalent_strain = _equivalent_elastic_strain_from_mises(von_mises, young=model.E, poisson=model.nu)
    stress_nodeavg = _node_average_cell_tensor(stress, elements, points.shape[0])
    strain_nodeavg = _node_average_cell_tensor(strain, elements, points.shape[0])
    von_mises_nodeavg = _von_mises_local(stress_nodeavg)
    strain_norm_nodeavg = np.linalg.norm(strain_nodeavg, axis=(1, 2))
    equivalent_strain_nodeavg = _equivalent_elastic_strain_from_mises(von_mises_nodeavg, young=model.E, poisson=model.nu)
    von_mises_scalaravg = _node_average_cell_scalar(von_mises, elements, points.shape[0])
    strain_norm_scalaravg = _node_average_cell_scalar(strain_norm, elements, points.shape[0])
    equivalent_strain_scalaravg = _node_average_cell_scalar(equivalent_strain, elements, points.shape[0])
    contact_diag = contact_node_diagnostics or _empty_contact_node_diagnostics(points.shape[0])
    contact_fields = dict(contact_diag.get("fields", {}))
    contact_metrics = dict(contact_diag.get("metrics", {}))
    object_metric_columns: Row = {}
    for object_id, object_nodes in static.object_node_ids_by_object:
        object_metric_columns[f"max_displacement_magnitude_object{object_id}"] = (
            float(np.max(displacement_norm[object_nodes])) if object_nodes.size else 0.0
        )
        object_metric_columns[f"max_von_mises_nodeavg_object{object_id}"] = (
            float(np.max(von_mises_nodeavg[object_nodes])) if object_nodes.size else 0.0
        )
        object_metric_columns[f"max_equivalent_elastic_strain_nodeavg_object{object_id}"] = (
            float(np.max(equivalent_strain_nodeavg[object_nodes])) if object_nodes.size else 0.0
        )
        object_metric_columns[f"p95_von_mises_nodeavg_object{object_id}"] = _percentile_or_zero(
            von_mises_nodeavg[object_nodes], 95.0
        )
        object_metric_columns[f"p95_equivalent_elastic_strain_nodeavg_object{object_id}"] = _percentile_or_zero(
            equivalent_strain_nodeavg[object_nodes], 95.0
        )
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"SFC Lagrangian-SDF full gear frame {frame_index} time={float(time_value):.12g}\n")
        handle.write("ASCII\n")
        handle.write("DATASET UNSTRUCTURED_GRID\n")
        handle.write(f"POINTS {points.shape[0]} float\n")
        for x, y, z in points:
            handle.write(f"{x:.9e} {y:.9e} {z:.9e}\n")
        handle.write(static.cells_block)
        handle.write(static.cell_types_block)
        handle.write(f"POINT_DATA {points.shape[0]}\n")
        handle.write("VECTORS U float\n")
        for ux, uy, uz in displacement:
            handle.write(f"{ux:.9e} {uy:.9e} {uz:.9e}\n")
        handle.write("VECTORS V float\n")
        for vx, vy, vz in velocity:
            handle.write(f"{vx:.9e} {vy:.9e} {vz:.9e}\n")
        handle.write("SCALARS displacement_magnitude float 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for value in displacement_norm:
            handle.write(f"{float(value):.9e}\n")
        for name, values in (
            ("von_mises_nodeavg", von_mises_nodeavg),
            ("strain_norm_nodeavg", strain_norm_nodeavg),
            ("equivalent_elastic_strain_nodeavg", equivalent_strain_nodeavg),
            ("von_mises_scalaravg", von_mises_scalaravg),
            ("strain_norm_scalaravg", strain_norm_scalaravg),
            ("equivalent_elastic_strain_scalaravg", equivalent_strain_scalaravg),
        ):
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in values:
                handle.write(f"{float(value):.9e}\n")
        for name, values in contact_fields.items():
            array = np.asarray(values, dtype=float).reshape(-1)
            if array.shape != (points.shape[0],):
                raise ValueError(f"contact diagnostic field {name!r} has incompatible shape")
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in array:
                handle.write(f"{float(value):.9e}\n")
        handle.write(static.object_id_block)
        for name, values in (
            ("von_mises", von_mises),
            ("strain_norm", strain_norm),
            ("equivalent_elastic_strain", equivalent_strain),
        ):
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in values:
                handle.write(f"{float(value):.9e}\n")
        if include_tensors:
            for name, tensor_values in (("E", strain), ("LE", strain), ("S", stress)):
                handle.write(f"TENSORS {name} float\n")
                for tensor in tensor_values:
                    for row in tensor:
                        handle.write(f"{row[0]:.9e} {row[1]:.9e} {row[2]:.9e}\n")
    row: Row = {
        "frame": int(frame_index),
        "time": float(time_value),
        "vtk_file": path.name,
        "node_count": int(points.shape[0]),
        "element_count": int(elements.shape[0]),
        "max_displacement_magnitude": float(np.max(displacement_norm)) if displacement_norm.size else 0.0,
        "max_von_mises": float(np.max(von_mises)) if von_mises.size else 0.0,
        "max_strain_norm": float(np.max(strain_norm)) if strain_norm.size else 0.0,
        "max_equivalent_elastic_strain": float(np.max(equivalent_strain)) if equivalent_strain.size else 0.0,
        "max_von_mises_nodeavg": float(np.max(von_mises_nodeavg)) if von_mises_nodeavg.size else 0.0,
        "max_strain_norm_nodeavg": float(np.max(strain_norm_nodeavg)) if strain_norm_nodeavg.size else 0.0,
        "max_equivalent_elastic_strain_nodeavg": float(np.max(equivalent_strain_nodeavg))
        if equivalent_strain_nodeavg.size
        else 0.0,
        "p95_von_mises_nodeavg": _percentile_or_zero(von_mises_nodeavg, 95.0),
        "p95_strain_norm_nodeavg": _percentile_or_zero(strain_norm_nodeavg, 95.0),
        "p95_equivalent_elastic_strain_nodeavg": _percentile_or_zero(equivalent_strain_nodeavg, 95.0),
        "mean_von_mises_nodeavg": float(np.mean(von_mises_nodeavg)) if von_mises_nodeavg.size else 0.0,
        "mean_strain_norm_nodeavg": float(np.mean(strain_norm_nodeavg)) if strain_norm_nodeavg.size else 0.0,
        "mean_equivalent_elastic_strain_nodeavg": float(np.mean(equivalent_strain_nodeavg))
        if equivalent_strain_nodeavg.size
        else 0.0,
        "max_von_mises_scalaravg": float(np.max(von_mises_scalaravg)) if von_mises_scalaravg.size else 0.0,
        "max_strain_norm_scalaravg": float(np.max(strain_norm_scalaravg)) if strain_norm_scalaravg.size else 0.0,
        "max_equivalent_elastic_strain_scalaravg": float(np.max(equivalent_strain_scalaravg))
        if equivalent_strain_scalaravg.size
        else 0.0,
        "p95_von_mises_scalaravg": _percentile_or_zero(von_mises_scalaravg, 95.0),
        "p95_strain_norm_scalaravg": _percentile_or_zero(strain_norm_scalaravg, 95.0),
        "p95_equivalent_elastic_strain_scalaravg": _percentile_or_zero(equivalent_strain_scalaravg, 95.0),
        "mean_von_mises_scalaravg": float(np.mean(von_mises_scalaravg)) if von_mises_scalaravg.size else 0.0,
        "mean_strain_norm_scalaravg": float(np.mean(strain_norm_scalaravg)) if strain_norm_scalaravg.size else 0.0,
        "mean_equivalent_elastic_strain_scalaravg": float(np.mean(equivalent_strain_scalaravg))
        if equivalent_strain_scalaravg.size
        else 0.0,
    }
    row.update(object_metric_columns)
    row.update(contact_metrics)
    return row


def _percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    """Return a finite percentile for manifest diagnostics."""

    array = np.asarray(values, dtype=float).reshape(-1)
    return float(np.percentile(array, float(percentile))) if array.size else 0.0


def _build_sfc_tet4_vtk_static_blocks(model: MechanicsModel, element_object_ids: np.ndarray) -> SFCVTKStaticBlocks:
    """Precompute fixed topology blocks for repeated SFC VTK frame writes."""

    elements = np.asarray(model.elements, dtype=np.int64)
    object_ids = np.asarray(element_object_ids, dtype=np.int64).reshape(-1)
    if object_ids.shape != (elements.shape[0],):
        raise ValueError("element_object_ids must match element count")
    object_node_ids: list[tuple[int, np.ndarray]] = []
    for object_id in sorted(int(value) for value in np.unique(object_ids)):
        elem_mask = object_ids == object_id
        if np.any(elem_mask):
            object_node_ids.append((object_id, np.unique(elements[elem_mask].reshape(-1))))
    cells_lines = [f"CELLS {elements.shape[0]} {elements.shape[0] * 5}\n"]
    cells_lines.extend("4 " + " ".join(str(int(node)) for node in element) + "\n" for element in elements)
    object_lines = [f"CELL_DATA {elements.shape[0]}\n", "SCALARS object_id int 1\n", "LOOKUP_TABLE default\n"]
    object_lines.extend(f"{int(value)}\n" for value in object_ids)
    return SFCVTKStaticBlocks(
        elements=elements,
        object_ids=object_ids,
        object_node_ids_by_object=tuple(object_node_ids),
        cells_block="".join(cells_lines),
        cell_types_block=f"CELL_TYPES {elements.shape[0]}\n" + ("10\n" * elements.shape[0]),
        object_id_block="".join(object_lines),
    )


def _node_average_cell_scalar(cell_values: np.ndarray, cells: np.ndarray, node_count: int) -> np.ndarray:
    """Return incident-cell averaged nodal values for visualization only."""

    values = np.asarray(cell_values, dtype=float).reshape(-1)
    conn = np.asarray(cells, dtype=np.int64)
    out = np.zeros(int(node_count), dtype=float)
    counts = np.zeros(int(node_count), dtype=float)
    if values.shape[0] != conn.shape[0]:
        return out
    flat_conn = conn.reshape(-1)
    np.add.at(out, flat_conn, np.repeat(values, conn.shape[1]))
    np.add.at(counts, flat_conn, 1.0)
    mask = counts > 0.0
    out[mask] /= counts[mask]
    return out


def _node_average_cell_tensor(cell_values: np.ndarray, cells: np.ndarray, node_count: int) -> np.ndarray:
    """Return incident-cell averaged nodal tensors.

    Abaqus-style nodal stress invariants are tensor averages followed by the
    invariant calculation.  This helper keeps SFC VTK/history metrics on that
    same nonlinear output-measure convention while leaving the solver state
    unchanged.
    """

    values = np.asarray(cell_values, dtype=float)
    conn = np.asarray(cells, dtype=np.int64)
    out = np.zeros((int(node_count), 3, 3), dtype=float)
    counts = np.zeros(int(node_count), dtype=float)
    if values.shape != (conn.shape[0], 3, 3):
        return out
    flat_conn = conn.reshape(-1)
    repeated = np.repeat(values, conn.shape[1], axis=0)
    np.add.at(out, flat_conn, repeated)
    np.add.at(counts, flat_conn, 1.0)
    mask = counts > 0.0
    out[mask] /= counts[mask, None, None]
    return out


def _node_averaged_internal_metric_row(model: MechanicsModel, internal: Any) -> Row:
    """Return SFC metrics using Abaqus-style node-averaged tensor invariants.

    The primary ``*_nodeavg`` metrics are tensor-component averages followed by
    invariant evaluation.  ``*_scalaravg`` diagnostics retain the old
    scalar-average convention for audits only.
    """

    elements = np.asarray(model.elements, dtype=np.int64)
    node_count = int(np.asarray(model.X, dtype=float).shape[0])
    stress = _tensor_field_or_zeros(internal.stress, elements.shape[0])
    strain = _tensor_field_or_zeros(internal.strain, elements.shape[0])
    von_mises = np.asarray(getattr(internal, "von_mises", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if von_mises.shape != (elements.shape[0],):
        von_mises = _von_mises_local(stress)
    strain_norm = np.linalg.norm(strain, axis=(1, 2))
    equivalent_strain = _equivalent_elastic_strain_from_mises(von_mises, young=model.E, poisson=model.nu)
    stress_nodeavg = _node_average_cell_tensor(stress, elements, node_count)
    strain_nodeavg = _node_average_cell_tensor(strain, elements, node_count)
    von_mises_nodeavg = _von_mises_local(stress_nodeavg)
    strain_norm_nodeavg = np.linalg.norm(strain_nodeavg, axis=(1, 2))
    equivalent_strain_nodeavg = _equivalent_elastic_strain_from_mises(von_mises_nodeavg, young=model.E, poisson=model.nu)
    von_mises_scalaravg = _node_average_cell_scalar(von_mises, elements, node_count)
    strain_norm_scalaravg = _node_average_cell_scalar(strain_norm, elements, node_count)
    equivalent_strain_scalaravg = _node_average_cell_scalar(equivalent_strain, elements, node_count)
    return {
        "max_von_mises_nodeavg": float(np.max(von_mises_nodeavg)) if von_mises_nodeavg.size else 0.0,
        "max_strain_norm_nodeavg": float(np.max(strain_norm_nodeavg)) if strain_norm_nodeavg.size else 0.0,
        "max_equivalent_elastic_strain_nodeavg": float(np.max(equivalent_strain_nodeavg))
        if equivalent_strain_nodeavg.size
        else 0.0,
        "p95_von_mises_nodeavg": _percentile_or_zero(von_mises_nodeavg, 95.0),
        "p95_strain_norm_nodeavg": _percentile_or_zero(strain_norm_nodeavg, 95.0),
        "p95_equivalent_elastic_strain_nodeavg": _percentile_or_zero(equivalent_strain_nodeavg, 95.0),
        "mean_von_mises_nodeavg": float(np.mean(von_mises_nodeavg)) if von_mises_nodeavg.size else 0.0,
        "mean_strain_norm_nodeavg": float(np.mean(strain_norm_nodeavg)) if strain_norm_nodeavg.size else 0.0,
        "mean_equivalent_elastic_strain_nodeavg": float(np.mean(equivalent_strain_nodeavg))
        if equivalent_strain_nodeavg.size
        else 0.0,
        "max_von_mises_scalaravg": float(np.max(von_mises_scalaravg)) if von_mises_scalaravg.size else 0.0,
        "p95_von_mises_scalaravg": _percentile_or_zero(von_mises_scalaravg, 95.0),
        "mean_von_mises_scalaravg": float(np.mean(von_mises_scalaravg)) if von_mises_scalaravg.size else 0.0,
        "max_strain_norm_scalaravg": float(np.max(strain_norm_scalaravg)) if strain_norm_scalaravg.size else 0.0,
        "p95_strain_norm_scalaravg": _percentile_or_zero(strain_norm_scalaravg, 95.0),
        "mean_strain_norm_scalaravg": float(np.mean(strain_norm_scalaravg)) if strain_norm_scalaravg.size else 0.0,
        "max_equivalent_elastic_strain_scalaravg": float(np.max(equivalent_strain_scalaravg))
        if equivalent_strain_scalaravg.size
        else 0.0,
        "p95_equivalent_elastic_strain_scalaravg": _percentile_or_zero(equivalent_strain_scalaravg, 95.0),
        "mean_equivalent_elastic_strain_scalaravg": float(np.mean(equivalent_strain_scalaravg))
        if equivalent_strain_scalaravg.size
        else 0.0,
    }


def _tensor_field_or_zeros(values: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape == (int(count), 3, 3):
        return array
    return np.zeros((int(count), 3, 3), dtype=float)


def _lame_parameters_local(young: float, poisson: float) -> tuple[float, float]:
    e = float(young)
    nu = float(poisson)
    mu = e / (2.0 * (1.0 + nu))
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def _von_mises_local(stress: np.ndarray) -> np.ndarray:
    values = np.asarray(stress, dtype=float)
    sx = values[:, 0, 0]
    sy = values[:, 1, 1]
    sz = values[:, 2, 2]
    txy = values[:, 0, 1]
    tyz = values[:, 1, 2]
    txz = values[:, 0, 2]
    mises = 0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy * txy + tyz * tyz + txz * txz)
    return np.sqrt(np.maximum(mises, 0.0))


def _face_centroids(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.mean(nodes[np.asarray(faces, dtype=np.int64)], axis=1)


def _triangle_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = nodes[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    normals[norms > 0.0] /= norms[norms > 0.0, None]
    return normals


def _surface_edge_length_percentile(nodes: np.ndarray, faces: np.ndarray, percentile: float = 95.0) -> float:
    """Return a robust contact-surface feature length from triangular faces."""

    points = np.asarray(nodes, dtype=float)
    tri = np.asarray(faces, dtype=np.int64)
    if tri.size == 0:
        return 0.0
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("faces must have shape (n_faces, 3)")
    edges = np.concatenate(
        [
            np.linalg.norm(points[tri[:, 1]] - points[tri[:, 0]], axis=1),
            np.linalg.norm(points[tri[:, 2]] - points[tri[:, 1]], axis=1),
            np.linalg.norm(points[tri[:, 0]] - points[tri[:, 2]], axis=1),
        ]
    )
    edges = edges[np.isfinite(edges) & (edges > 0.0)]
    return float(np.percentile(edges, float(percentile))) if edges.size else 0.0


def _default_contact_search_radius(pair: CroppedGearPair, *, target_overclosure: float = 0.0) -> float:
    """Return a conservative closest-feature broad-phase radius.

    The search radius is a candidate-pruning parameter only.  It must not be
    tied solely to the initial normal gap: curved tooth surfaces can have a
    very small closest-point gap while the relevant master facets are separated
    tangentially by roughly one surface feature length.  Using the larger of
    the normal gap envelope and a robust surface feature size prevents the
    broad phase from excluding the true closest-feature candidates without
    changing the final projection accuracy.
    """

    normal_envelope = 2.5 * max(float(pair.initial_patch_gap) + float(target_overclosure), 0.0)
    feature_length = max(
        _surface_edge_length_percentile(pair.gear1.nodes, pair.gear1.contact_faces),
        _surface_edge_length_percentile(pair.gear2.nodes, pair.gear2.contact_faces),
    )
    feature_envelope = 2.5 * feature_length
    return max(normal_envelope, feature_envelope, 1.0e-4)


def _default_secondary_contact_tracking_radius(pair: CroppedGearPair, *, target_overclosure: float = 0.0) -> float:
    """Return an Abaqus-style secondary contact tracking tube radius.

    The secondary-normal line projection uses this radius only to enumerate
    possible main-surface facets.  It must therefore be at least as conservative
    as the hard finite-sliding line-distance gate; otherwise Abaqus-active nodes
    can be lost before the line-distance and closest-feature open/closed gates
    evaluate the true contact status.  It also cannot be narrower than the
    closest-feature broad-phase tube: on curved finite-sliding interfaces, the
    secondary normal may intersect a neighboring main facet whose centroid is
    tangentially farther away than the local line-distance limit.  The final
    accepted contact still uses the mesh-derived line-distance limits and the
    closest-feature open/closed guard, so increasing this candidate radius does
    not relax contact accuracy.
    """

    normal_envelope = 2.5 * max(float(pair.initial_patch_gap) + float(target_overclosure), 0.0)
    local_region = max(normal_envelope, _contact_patch_representative_length(pair), 1.0e-4)
    hard_gate = _default_secondary_line_hard_distance_limit(pair, target_overclosure=target_overclosure)
    closest_feature_radius = _default_contact_search_radius(pair, target_overclosure=target_overclosure)
    return max(local_region, hard_gate, closest_feature_radius)


def _default_secondary_line_distance_limit(pair: CroppedGearPair, *, target_overclosure: float = 0.0) -> float:
    """Return the accepted normal-projection distance for secondary-line contact.

    Candidate tracking and final contact enforcement have different roles.  The
    broad phase may keep neighboring faces for continuity, but a
    surface-to-surface normal-line constraint should not convert an arbitrarily
    remote line intersection into pressure.  The accepted line distance is
    therefore tied to the representative local contact facet length and the
    requested overclosure scale, which is mesh-derived rather than curve-fitted.
    """

    return max(_contact_patch_representative_length(pair), 2.5 * float(target_overclosure), 1.0e-4)


def _default_secondary_line_hard_distance_limit(pair: CroppedGearPair, *, target_overclosure: float = 0.0) -> float:
    """Return the hard normal-line re-capture distance for secondary contact.

    The soft line-distance limit can be overridden by a genuinely closed
    closest-feature query to avoid releasing real overclosure.  A second,
    larger hard tube prevents that exception from accepting remote line hits
    far outside the local finite-sliding constraint region.
    """

    return 2.0 * _default_secondary_line_distance_limit(pair, target_overclosure=target_overclosure)


def _orient_faces_toward(nodes: np.ndarray, faces: np.ndarray, direction: np.ndarray) -> np.ndarray:
    out = np.asarray(faces, dtype=np.int64).copy()
    normals = _triangle_normals(nodes, out)
    d = np.asarray(direction, dtype=float)
    d /= max(float(np.linalg.norm(d)), 1.0e-30)
    flip = normals @ d < 0.0
    out[flip, 1], out[flip, 2] = out[flip, 2].copy(), out[flip, 1].copy()
    return out


def _select_patch_indices(model: GearInputModel, *, faces_per_body: int) -> tuple[np.ndarray, np.ndarray, float]:
    centroids1 = _face_centroids(model.gear1.nodes, model.gear1_contact_faces)
    centroids2 = _face_centroids(model.gear2.nodes, model.gear2_contact_faces)
    tree2 = cKDTree(centroids2)
    distances, ids2 = tree2.query(centroids1, k=1)
    seed1 = int(np.argmin(distances))
    seed2 = int(ids2[seed1])
    n = max(1, int(faces_per_body))
    tree1 = cKDTree(centroids1)
    ids1 = np.asarray(tree1.query(centroids1[seed1], k=min(n, centroids1.shape[0]))[1], dtype=np.int64).ravel()
    ids2 = np.asarray(tree2.query(centroids2[seed2], k=min(n, centroids2.shape[0]))[1], dtype=np.int64).ravel()
    return ids1, ids2, float(distances[seed1])


def _crop_patch(
    *,
    name: str,
    mesh: GearMesh,
    contact_faces: np.ndarray,
    surface_entries: tuple[tuple[int, str], ...],
    selected_surface_ids: np.ndarray,
    support_direction: np.ndarray,
    support_high_side: bool,
    rp: np.ndarray,
    expansion_rings: int,
) -> CroppedGearPatch:
    selected_surface_ids = np.asarray(selected_surface_ids, dtype=np.int64).ravel()
    if selected_surface_ids.size == 0:
        raise ValueError("selected_surface_ids must not be empty")
    node_set = set(int(v) for v in contact_faces[selected_surface_ids].ravel())
    for _ in range(max(0, int(expansion_rings))):
        mask = np.asarray([any(int(node) in node_set for node in element) for element in mesh.elements], dtype=bool)
        for element in mesh.elements[mask]:
            node_set.update(int(v) for v in element)
    element_mask = np.asarray([any(int(node) in node_set for node in element) for element in mesh.elements], dtype=bool)
    selected_elements = mesh.elements[element_mask]
    selected_element_labels = mesh.element_labels[element_mask]
    node_ids = np.asarray(sorted(set(int(v) for v in selected_elements.ravel())), dtype=np.int64)
    old_to_new = {int(old): idx for idx, old in enumerate(node_ids)}
    label_to_new_element = {int(label): idx + 1 for idx, label in enumerate(selected_element_labels)}
    nodes = mesh.nodes[node_ids]
    elements = np.asarray([[old_to_new[int(v)] for v in element] for element in selected_elements], dtype=np.int64)
    selected_faces_old = contact_faces[selected_surface_ids]
    contact_faces_new = np.asarray([[old_to_new[int(v)] for v in face] for face in selected_faces_old], dtype=np.int64)
    entries = tuple(
        (label_to_new_element[int(surface_entries[int(idx)][0])], surface_entries[int(idx)][1])
        for idx in selected_surface_ids
        if int(surface_entries[int(idx)][0]) in label_to_new_element
    )
    if not entries:
        entries = tuple((idx + 1, "S1") for idx in range(min(contact_faces_new.shape[0], elements.shape[0])))
    face_nodes = np.unique(contact_faces_new.ravel())
    center = np.mean(nodes[face_nodes], axis=0)
    direction = np.asarray(support_direction, dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-30)
    scores = (nodes - center[None, :]) @ direction
    pct = 82.0 if support_high_side else 18.0
    threshold = float(np.percentile(scores, pct))
    support = np.nonzero(scores >= threshold if support_high_side else scores <= threshold)[0]
    if support.size < 4:
        support = np.argsort(scores)[-4:] if support_high_side else np.argsort(scores)[:4]
    return CroppedGearPatch(
        name=name,
        nodes=nodes,
        elements=elements,
        contact_faces=contact_faces_new,
        support_nodes=np.asarray(np.unique(support), dtype=np.int64),
        surface_entries=entries,
        element_labels=np.arange(1, elements.shape[0] + 1, dtype=np.int64),
        rp=np.asarray(rp, dtype=float),
    )


def build_cropped_pair(model: GearInputModel, *, faces_per_body: int, expansion_rings: int) -> CroppedGearPair:
    ids1, ids2, initial_gap = _select_patch_indices(model, faces_per_body=faces_per_body)
    c1 = np.mean(_face_centroids(model.gear1.nodes, model.gear1_contact_faces[ids1]), axis=0)
    c2 = np.mean(_face_centroids(model.gear2.nodes, model.gear2_contact_faces[ids2]), axis=0)
    drive = c2 - c1
    drive /= max(float(np.linalg.norm(drive)), 1.0e-30)
    gear1 = _crop_patch(
        name="gear1",
        mesh=model.gear1,
        contact_faces=model.gear1_contact_faces,
        surface_entries=model.gear1_surface_entries,
        selected_surface_ids=ids1,
        support_direction=drive,
        support_high_side=False,
        rp=model.rp1,
        expansion_rings=expansion_rings,
    )
    gear2 = _crop_patch(
        name="gear2",
        mesh=model.gear2,
        contact_faces=model.gear2_contact_faces,
        surface_entries=model.gear2_surface_entries,
        selected_surface_ids=ids2,
        support_direction=drive,
        support_high_side=True,
        rp=model.rp2,
        expansion_rings=expansion_rings,
    )
    gear1_faces = _orient_faces_toward(gear1.nodes, gear1.contact_faces, drive)
    gear2_faces = _orient_faces_toward(gear2.nodes, gear2.contact_faces, -drive)
    gear1 = CroppedGearPatch(gear1.name, gear1.nodes, gear1.elements, gear1_faces, gear1.support_nodes, gear1.surface_entries, gear1.element_labels, gear1.rp)
    gear2 = CroppedGearPatch(gear2.name, gear2.nodes, gear2.elements, gear2_faces, gear2.support_nodes, gear2.surface_entries, gear2.element_labels, gear2.rp)
    return CroppedGearPair(gear1, gear2, drive, initial_gap)


def _fixed_conditions_for_pair(
    pair: CroppedGearPair,
    *,
    closure: float,
    rotation_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    hub1 = RigidHubMPC(pair.gear1.support_nodes, pair.gear1.nodes, pair.gear1.rp)
    hub2 = RigidHubMPC(pair.gear2.support_nodes, pair.gear2.nodes, pair.gear2.rp)
    d1, v1 = hub1.dirichlet_dofs_and_values(
        translation=closure * pair.drive_direction,
        rotation=(0.0, 0.0, rotation_z),
    )
    d2, v2 = hub2.dirichlet_dofs_and_values()
    d2 = d2 + 3 * pair.gear1.nodes.shape[0]
    return merge_dirichlet_conditions((d1, v1), (d2, v2))


def _penalty_rhs_balance(model: MechanicsModel, internal: Any, contact_response: Any) -> np.ndarray:
    """Return external-minus-internal balance for the zero-gravity gear cases."""

    _ = model
    return -np.asarray(internal.force, dtype=float).reshape(-1) + np.asarray(contact_response.force, dtype=float).reshape(-1)


def _penalty_history_row(
    *,
    time_value: float,
    closure: float,
    rotation: float,
    state: MechanicsState,
    model: MechanicsModel,
    internal: Any,
    contact_response: Any,
    young: float,
    poisson: float,
    newton_iterations: int,
    residual_norm: float,
    solver: str,
) -> Row:
    disp = state.x - model.X
    strain_norm = np.linalg.norm(internal.strain, axis=(1, 2)) if internal.strain.size else np.empty(0, dtype=float)
    elastic_strain_norm = _elastic_strain_norm_from_stress(internal.stress, young=young, poisson=poisson)
    equivalent_elastic_strain = _equivalent_elastic_strain_from_mises(internal.von_mises, young=young, poisson=poisson)
    row = {
        "time": float(time_value),
        "closure": float(closure),
        "rotation_z": float(rotation),
        "active_contact_samples": int(contact_response.active_count),
        "min_gap": float(contact_response.min_gap),
        "normal_force": float(contact_response.normal_force),
        "contact_energy": float(contact_response.energy),
        "contact_virtual_work": float(2.0 * contact_response.energy),
        "contact_resultant_force_norm": float(np.linalg.norm(np.asarray(contact_response.force, dtype=float).reshape((-1, 3)).sum(axis=0))),
        "max_displacement_norm": float(np.max(np.linalg.norm(disp, axis=1))),
        "p95_von_mises": float(np.percentile(internal.von_mises, 95.0)) if internal.von_mises.size else 0.0,
        "max_von_mises": float(np.max(internal.von_mises)) if internal.von_mises.size else 0.0,
        "p95_strain_norm": float(np.percentile(strain_norm, 95.0)) if strain_norm.size else 0.0,
        "max_strain_norm": float(np.max(strain_norm)) if strain_norm.size else 0.0,
        "p95_elastic_strain_norm": float(np.percentile(elastic_strain_norm, 95.0)) if elastic_strain_norm.size else 0.0,
        "max_elastic_strain_norm": float(np.max(elastic_strain_norm)) if elastic_strain_norm.size else 0.0,
        "p95_equivalent_elastic_strain": float(np.percentile(equivalent_elastic_strain, 95.0)) if equivalent_elastic_strain.size else 0.0,
        "max_equivalent_elastic_strain": float(np.max(equivalent_elastic_strain)) if equivalent_elastic_strain.size else 0.0,
        "strain_energy": float(internal.strain_energy),
        "newton_iterations": int(newton_iterations),
        "newton_residual_norm": float(residual_norm),
        "penalty_solver": str(solver),
    }
    row.update(_node_averaged_internal_metric_row(model, internal))
    return row


def _solve_penalty_low_rank_correction(
    *,
    base_matrix_free: Any,
    base_lu_cache: list[Any],
    residual_free: np.ndarray,
    samples: list[Any],
    free: np.ndarray,
    n_total_dofs: int,
    equilibrium_scale: float,
    tolerance: float,
    use_cg: bool = False,
) -> np.ndarray:
    """Solve a Newton correction with the active penalty tangent as low rank."""

    rhs = -np.asarray(residual_free, dtype=float).reshape(-1)
    if rhs.size == 0:
        return np.empty(0, dtype=float)
    active_samples = [sample for sample in samples if float(sample.gap) < 0.0]
    j_free = np.empty((0, rhs.size), dtype=float)
    tangent_scale = np.asarray(
        [float(equilibrium_scale) * float(sample.stiffness) * float(sample.area) for sample in active_samples],
        dtype=float,
    )
    if active_samples:
        _gaps, gap_jacobian = hard_contact_gap_jacobian_from_samples(active_samples, n_total_dofs=int(n_total_dofs))
        j_free = np.asarray(gap_jacobian[:, np.asarray(free, dtype=np.int64)], dtype=float)
        positive = tangent_scale > 0.0
        j_free = j_free[positive]
        tangent_scale = tangent_scale[positive]
    if bool(use_cg):
        cg_solution = _solve_penalty_cg_correction(
            base_matrix_free=base_matrix_free,
            rhs=rhs,
            j_free=j_free,
            tangent_scale=tangent_scale,
            tolerance=tolerance,
        )
        if cg_solution is not None:
            return cg_solution
    if base_lu_cache[0] is None:
        base_lu_cache[0] = splu(base_matrix_free.tocsc())
    base_lu = base_lu_cache[0]
    base_solution = np.asarray(base_lu.solve(rhs), dtype=float).reshape(-1)
    if not active_samples or j_free.size == 0:
        return base_solution
    influence = np.asarray(base_lu.solve(j_free.T), dtype=float)
    schur = np.diag(1.0 / tangent_scale) + j_free @ influence
    try:
        contact_coeff = np.linalg.solve(schur, j_free @ base_solution)
    except np.linalg.LinAlgError:
        contact_coeff, *_ = np.linalg.lstsq(schur, j_free @ base_solution, rcond=None)
    return base_solution - influence @ contact_coeff


def _solve_reduced_penalty_low_rank_correction(
    *,
    base_matrix_free: Any,
    base_lu_cache: list[Any],
    residual_free: np.ndarray,
    samples: list[Any],
    transformation: Any,
    free: np.ndarray,
    n_total_dofs: int,
    equilibrium_scale: float,
) -> np.ndarray:
    """Solve a reduced-coordinate Newton correction with active contact tangent."""

    rhs = -np.asarray(residual_free, dtype=float).reshape(-1)
    if rhs.size == 0:
        return np.empty(0, dtype=float)
    active_samples = [sample for sample in samples if float(sample.gap) < 0.0]
    j_free = np.empty((0, rhs.size), dtype=float)
    tangent_scale = np.asarray(
        [float(equilibrium_scale) * float(sample.stiffness) * float(sample.area) for sample in active_samples],
        dtype=float,
    )
    if active_samples:
        _gaps, gap_jacobian = hard_contact_gap_jacobian_from_samples(active_samples, n_total_dofs=int(n_total_dofs))
        j_reduced = gap_jacobian @ transformation
        j_free = np.asarray(j_reduced[:, np.asarray(free, dtype=np.int64)], dtype=float)
        positive = tangent_scale > 0.0
        j_free = j_free[positive]
        tangent_scale = tangent_scale[positive]
    if base_lu_cache[0] is None:
        base_lu_cache[0] = splu(base_matrix_free.tocsc())
    base_lu = base_lu_cache[0]
    base_solution = np.asarray(base_lu.solve(rhs), dtype=float).reshape(-1)
    if not active_samples or j_free.size == 0:
        return base_solution
    influence = np.asarray(base_lu.solve(j_free.T), dtype=float)
    schur = np.diag(1.0 / tangent_scale) + j_free @ influence
    try:
        contact_coeff = np.linalg.solve(schur, j_free @ base_solution)
    except np.linalg.LinAlgError:
        contact_coeff, *_ = np.linalg.lstsq(schur, j_free @ base_solution, rcond=None)
    return base_solution - influence @ contact_coeff


def _active_gap_jacobian_sparse(samples: list[Any], *, n_total_dofs: int) -> Any:
    """Build a sparse gap Jacobian for active contact samples only."""

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    active_row = 0
    for sample in samples:
        if float(sample.gap) >= 0.0:
            continue
        normal = np.asarray(sample.normal, dtype=float).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 0.0:
            raise ValueError("sample normal must be nonzero")
        normal = normal / normal_norm
        for sign, node_attr, weight_attr in (
            (1.0, "node_ids", "shape_weights"),
            (-1.0, "master_node_ids", "master_shape_weights"),
        ):
            node_values = getattr(sample, node_attr, None)
            weight_values = getattr(sample, weight_attr, None)
            if node_values is None or weight_values is None:
                continue
            nodes = np.asarray(node_values, dtype=np.int64).reshape(-1)
            weights = np.asarray(weight_values, dtype=float).reshape(-1)
            if nodes.shape != weights.shape:
                raise ValueError("contact sample nodes and weights must match")
            for node, weight in zip(nodes, weights, strict=True):
                base = 3 * int(node)
                for component in range(3):
                    value = float(sign) * float(weight) * float(normal[component])
                    if value != 0.0:
                        rows.append(active_row)
                        cols.append(base + component)
                        data.append(value)
        active_row += 1
    return coo_matrix((data, (rows, cols)), shape=(active_row, int(n_total_dofs))).tocsr()


def _active_reduced_gap_jacobian_sparse(
    samples: list[Any],
    *,
    transformation: Any,
    free: np.ndarray,
) -> Any:
    """Build active contact gap Jacobian directly in reduced free DOFs."""

    T = transformation.tocsr()
    free_cols = np.asarray(free, dtype=np.int64).reshape(-1)
    reduced_to_free = np.full(T.shape[1], -1, dtype=np.int64)
    reduced_to_free[free_cols] = np.arange(free_cols.size, dtype=np.int64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    active_row = 0
    for sample in samples:
        if float(sample.gap) >= 0.0:
            continue
        normal = np.asarray(sample.normal, dtype=float).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 0.0:
            raise ValueError("sample normal must be nonzero")
        normal = normal / normal_norm
        for sign, node_attr, weight_attr in (
            (1.0, "node_ids", "shape_weights"),
            (-1.0, "master_node_ids", "master_shape_weights"),
        ):
            node_values = getattr(sample, node_attr, None)
            weight_values = getattr(sample, weight_attr, None)
            if node_values is None or weight_values is None:
                continue
            nodes = np.asarray(node_values, dtype=np.int64).reshape(-1)
            weights = np.asarray(weight_values, dtype=float).reshape(-1)
            if nodes.shape != weights.shape:
                raise ValueError("contact sample nodes and weights must match")
            for node, weight in zip(nodes, weights, strict=True):
                for component in range(3):
                    coeff = float(sign) * float(weight) * float(normal[component])
                    if coeff == 0.0:
                        continue
                    full_row = 3 * int(node) + component
                    start = int(T.indptr[full_row])
                    stop = int(T.indptr[full_row + 1])
                    for ptr in range(start, stop):
                        free_col = int(reduced_to_free[int(T.indices[ptr])])
                        if free_col >= 0:
                            rows.append(active_row)
                            cols.append(free_col)
                            data.append(coeff * float(T.data[ptr]))
        active_row += 1
    return coo_matrix((data, (rows, cols)), shape=(active_row, free_cols.size)).tocsr()


def _active_reduced_gap_jacobian_sparse_from_arrays(
    sample_arrays: dict[str, np.ndarray],
    *,
    transformation: Any,
    free: np.ndarray,
) -> Any:
    """Build active reduced gap Jacobian from batched sample arrays."""

    _active_ids, reduced_jacobian = _core_constraint_region_reduced_gap_jacobian_sparse_from_arrays(
        sample_arrays,
        transformation=transformation,
        free=free,
        active_only=True,
    )
    return reduced_jacobian


def _constraint_region_gap_jacobian_sparse_from_arrays(
    sample_arrays: dict[str, np.ndarray],
    *,
    n_nodes: int,
    active_only: bool = False,
) -> tuple[np.ndarray, Any]:
    """Return the full-space gap Jacobian for constraint-region rows.

    Each row follows the fixed-payload surface-to-surface region gap

    ``dg = sum_i Ns_i n^T dxs_i - sum_a Nm_a n^T dxm_a``.

    This is the explicit derivative used by the fixed-active-set contact
    tangent.  It is separate from SDF querying: closest-feature ids,
    barycentric weights, region normals, and active status are held fixed.
    """

    return _core_constraint_region_gap_jacobian_sparse_from_arrays(
        sample_arrays,
        n_nodes=n_nodes,
        active_only=active_only,
    )

    gaps = np.asarray(sample_arrays["gaps"], dtype=float).reshape(-1)
    if active_only:
        row_ids = np.flatnonzero(gaps < 0.0).astype(np.int64)
    else:
        row_ids = np.arange(gaps.size, dtype=np.int64)
    n_total_dofs = 3 * int(n_nodes)
    if row_ids.size == 0:
        return row_ids, coo_matrix((0, n_total_dofs), dtype=float).tocsr()
    slave_nodes = np.asarray(sample_arrays["sample_node_ids"], dtype=np.int64)
    slave_weights = np.asarray(sample_arrays["sample_weights"], dtype=float)
    master_nodes = np.asarray(sample_arrays["master_node_ids"], dtype=np.int64)
    master_weights = np.asarray(sample_arrays["master_weights"], dtype=float)
    normals = np.asarray(sample_arrays["normals"], dtype=float).reshape((-1, 3))
    normal_norm = np.maximum(np.linalg.norm(normals, axis=1), 1.0e-30)
    normals = normals / normal_norm[:, None]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for out_row, sample_id in enumerate(row_ids):
        normal = normals[int(sample_id)]
        for sign, nodes, weights in (
            (1.0, slave_nodes[int(sample_id)], slave_weights[int(sample_id)]),
            (-1.0, master_nodes[int(sample_id)], master_weights[int(sample_id)]),
        ):
            for node, weight in zip(nodes, weights, strict=True):
                if int(node) < 0 or int(node) >= int(n_nodes):
                    continue
                for component in range(3):
                    coeff = float(sign) * float(weight) * float(normal[component])
                    if coeff == 0.0:
                        continue
                    rows.append(out_row)
                    cols.append(3 * int(node) + component)
                    data.append(coeff)
    return row_ids, coo_matrix((data, (rows, cols)), shape=(row_ids.size, n_total_dofs)).tocsr()


def _constraint_region_pressure_tangent_scales_from_arrays(
    sample_arrays: dict[str, np.ndarray],
    *,
    pressure_stiffness: float,
    equilibrium_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return active rows and ``d(A p)/dg`` magnitudes for linear penalty.

    With ``p = k_p <-g>_+`` and fixed active set, the residual tangent uses
    ``equilibrium_scale * k_p * A_region`` for each active region.
    Inactive/open regions have zero pressure-overclosure derivative.
    """

    return _core_constraint_region_pressure_tangent_scales_from_arrays(
        sample_arrays,
        pressure_stiffness=pressure_stiffness,
        equilibrium_scale=equilibrium_scale,
    )

    gaps = np.asarray(sample_arrays["gaps"], dtype=float).reshape(-1)
    active_ids = np.flatnonzero(gaps < 0.0).astype(np.int64)
    if active_ids.size == 0:
        return active_ids, np.empty(0, dtype=float)
    areas = np.asarray(sample_arrays["areas"], dtype=float).reshape(-1)
    scales = float(equilibrium_scale) * float(pressure_stiffness) * areas[active_ids]
    positive = scales > 0.0
    return active_ids[positive], scales[positive]


def _active_constraint_region_tangent_data_from_arrays(
    sample_arrays: dict[str, np.ndarray],
    *,
    transformation: Any,
    free: np.ndarray,
    pressure_stiffness: float,
    equilibrium_scale: float,
) -> tuple[np.ndarray, Any, np.ndarray]:
    """Return active region ids, reduced gap Jacobian, and pressure tangent scales.

    For aggregated ``slave_node_region_*`` arrays each row is one secondary
    constraint region.  With a fixed active set and fixed closest-feature
    payload, the residual contact tangent is

    ``Kc = J.T @ diag(equilibrium_scale * k_p * A_region) @ J``.

    This helper centralizes the active-set filtering so the solver, tests, and
    diagnostics all use the same Abaqus-style constraint-region tangent data.
    """

    return _core_active_constraint_region_tangent_data_from_arrays(
        sample_arrays,
        transformation=transformation,
        free=free,
        pressure_stiffness=pressure_stiffness,
        equilibrium_scale=equilibrium_scale,
    )


def _constraint_region_tangent_metrics_from_arrays(
    sample_arrays: dict[str, np.ndarray] | None,
    *,
    pressure_stiffness: float,
    equilibrium_scale: float = 1.0,
) -> Row:
    """Return scalar diagnostics for the active constraint-region tangent."""

    return dict(
        _core_constraint_region_tangent_metrics_from_arrays(
            sample_arrays,
            pressure_stiffness=pressure_stiffness,
            equilibrium_scale=equilibrium_scale,
        )
    )


def _solve_penalty_cg_correction_sparse(
    *,
    base_matrix_free: Any,
    rhs: np.ndarray,
    j_free: Any,
    tangent_scale: np.ndarray,
    tolerance: float,
    stats: dict[str, Any] | None = None,
    base_lu: Any | None = None,
) -> np.ndarray | None:
    """Solve ``(K + J.T W J) x = rhs`` with sparse matrix-free CG."""

    b = np.asarray(rhs, dtype=float).reshape(-1)
    j = j_free.tocsr()
    scale = np.asarray(tangent_scale, dtype=float).reshape(-1)
    if j.shape[0] != scale.shape[0] or j.shape[1] != b.size:
        return None
    base = base_matrix_free.tocsr()

    def matvec(value: np.ndarray) -> np.ndarray:
        vec = np.asarray(value, dtype=float).reshape(-1)
        out = np.asarray(base @ vec, dtype=float)
        if j.shape[0]:
            out += np.asarray(j.T @ (scale * np.asarray(j @ vec, dtype=float)), dtype=float)
        return out

    diag = np.asarray(base.diagonal(), dtype=float).reshape(-1)
    if j.shape[0]:
        contact_diag = np.asarray(j.multiply(j).T @ scale, dtype=float).reshape(-1)
        diag = diag + contact_diag
    safe_diag = np.where(np.abs(diag) > 1.0e-30, diag, 1.0)

    preconditioner_kind = "diagonal"
    if base_lu is not None:
        preconditioner = LinearOperator(
            (b.size, b.size),
            matvec=lambda value: np.asarray(base_lu.solve(np.asarray(value, dtype=float)), dtype=float).reshape(-1),
            dtype=float,
        )
        preconditioner_kind = "base_lu"
    else:
        preconditioner = LinearOperator((b.size, b.size), matvec=lambda value: np.asarray(value, dtype=float) / safe_diag, dtype=float)
    operator = LinearOperator((b.size, b.size), matvec=matvec, dtype=float)
    rtol = min(1.0e-8, max(1.0e-11, float(tolerance) * 10.0))
    atol = max(float(tolerance) * max(1.0, float(np.linalg.norm(b))) * 1.0e-2, 1.0e-10)
    iterations = 0

    def count_iteration(_value: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    try:
        solution, info = cg(
            operator,
            b,
            rtol=rtol,
            atol=atol,
            maxiter=max(200, min(2000, b.size // 20)),
            M=preconditioner,
            callback=count_iteration,
        )
    except Exception:
        return None
    if info != 0:
        return None
    residual = matvec(solution) - b
    allowed = atol + rtol * max(1.0, float(np.linalg.norm(b)))
    if float(np.linalg.norm(residual)) > max(50.0 * allowed, 1.0e-7):
        return None
    if stats is not None:
        stats["iterations"] = int(iterations)
        stats["preconditioner"] = preconditioner_kind
    return np.asarray(solution, dtype=float).reshape(-1)


def _solve_reduced_penalty_sparse_cg_correction(
    *,
    base_matrix_free: Any,
    residual_free: np.ndarray,
    samples: list[Any],
    transformation: Any,
    free: np.ndarray,
    n_total_dofs: int,
    equilibrium_scale: float,
    tolerance: float,
    stats: dict[str, Any] | None = None,
    base_lu: Any | None = None,
) -> np.ndarray | None:
    """Solve the reduced contact tangent with sparse matrix-free CG."""

    rhs = -np.asarray(residual_free, dtype=float).reshape(-1)
    if rhs.size == 0:
        return np.empty(0, dtype=float)
    active_samples = [sample for sample in samples if float(sample.gap) < 0.0]
    if not active_samples:
        try:
            return np.asarray(cg(base_matrix_free.tocsr(), rhs, rtol=1.0e-10, atol=1.0e-12, maxiter=1000)[0], dtype=float)
        except Exception:
            return None
    tangent_scale = np.asarray(
        [float(equilibrium_scale) * float(sample.stiffness) * float(sample.area) for sample in active_samples],
        dtype=float,
    )
    positive = tangent_scale > 0.0
    if not np.any(positive):
        return None
    _ = int(n_total_dofs)  # kept for API parity with the dense fallback.
    j_free = _active_reduced_gap_jacobian_sparse(active_samples, transformation=transformation, free=free)
    j_free = j_free[positive]
    tangent_scale = tangent_scale[positive]
    return _solve_penalty_cg_correction_sparse(
        base_matrix_free=base_matrix_free,
        rhs=rhs,
        j_free=j_free,
        tangent_scale=tangent_scale,
        tolerance=tolerance,
        stats=stats,
        base_lu=base_lu,
    )


def _solve_reduced_penalty_sparse_cg_correction_from_arrays(
    *,
    base_matrix_free: Any,
    residual_free: np.ndarray,
    sample_arrays: dict[str, np.ndarray],
    transformation: Any,
    free: np.ndarray,
    equilibrium_scale: float,
    pressure_stiffness: float,
    tolerance: float,
    stats: dict[str, Any] | None = None,
    base_lu: Any | None = None,
) -> np.ndarray | None:
    """Solve the reduced contact tangent using batched sample arrays."""

    rhs = -np.asarray(residual_free, dtype=float).reshape(-1)
    if rhs.size == 0:
        return np.empty(0, dtype=float)
    gaps = np.asarray(sample_arrays["gaps"], dtype=float).reshape(-1)
    active = gaps < 0.0
    if not np.any(active):
        try:
            return np.asarray(cg(base_matrix_free.tocsr(), rhs, rtol=1.0e-10, atol=1.0e-12, maxiter=1000)[0], dtype=float)
        except Exception:
            return None
    _active_ids, j_free, tangent_scale = _active_constraint_region_tangent_data_from_arrays(
        sample_arrays,
        transformation=transformation,
        free=free,
        pressure_stiffness=pressure_stiffness,
        equilibrium_scale=equilibrium_scale,
    )
    if tangent_scale.size == 0:
        return None
    if stats is not None:
        stats["contact_tangent_source"] = "constraint_region_arrays"
        stats["contact_tangent_active_region_count"] = int(tangent_scale.size)
        stats["contact_tangent_j_nnz"] = int(j_free.nnz)
        stats["contact_tangent_scale_sum"] = float(np.sum(tangent_scale))
        stats["contact_tangent_scale_max"] = float(np.max(tangent_scale))
    return _solve_penalty_cg_correction_sparse(
        base_matrix_free=base_matrix_free,
        rhs=rhs,
        j_free=j_free,
        tangent_scale=tangent_scale,
        tolerance=tolerance,
        stats=stats,
        base_lu=base_lu,
    )


def _solve_penalty_cg_correction(
    *,
    base_matrix_free: Any,
    rhs: np.ndarray,
    j_free: np.ndarray,
    tangent_scale: np.ndarray,
    tolerance: float,
) -> np.ndarray | None:
    """Solve the active penalty tangent system by matrix-free CG."""

    b = np.asarray(rhs, dtype=float).reshape(-1)
    j = np.asarray(j_free, dtype=float)
    scale = np.asarray(tangent_scale, dtype=float).reshape(-1)
    if j.shape[0] != scale.shape[0] or (j.ndim == 2 and j.shape[1] != b.size):
        return None
    base = base_matrix_free.tocsr()

    def matvec(value: np.ndarray) -> np.ndarray:
        vec = np.asarray(value, dtype=float).reshape(-1)
        out = np.asarray(base @ vec, dtype=float)
        if j.size:
            out += np.asarray(j.T @ (scale * (j @ vec)), dtype=float)
        return out

    diag = np.asarray(base.diagonal(), dtype=float).reshape(-1)
    if j.size:
        diag = diag + np.sum((j * j) * scale[:, None], axis=0)
    safe_diag = np.where(np.abs(diag) > 1.0e-30, diag, 1.0)
    preconditioner = LinearOperator((b.size, b.size), matvec=lambda value: np.asarray(value, dtype=float) / safe_diag, dtype=float)
    operator = LinearOperator((b.size, b.size), matvec=matvec, dtype=float)
    rtol = min(1.0e-8, max(1.0e-11, float(tolerance) * 10.0))
    atol = max(float(tolerance) * max(1.0, float(np.linalg.norm(b))) * 1.0e-2, 1.0e-10)
    try:
        solution, info = cg(
            operator,
            b,
            rtol=rtol,
            atol=atol,
            maxiter=max(200, min(2000, b.size // 20)),
            M=preconditioner,
        )
    except Exception:
        return None
    if info != 0:
        return None
    residual = matvec(solution) - b
    allowed = atol + rtol * max(1.0, float(np.linalg.norm(b)))
    if float(np.linalg.norm(residual)) > max(50.0 * allowed, 1.0e-7):
        return None
    return np.asarray(solution, dtype=float).reshape(-1)


def _solve_sfc_cropped_pair_penalty_modified_newton(
    pair: CroppedGearPair,
    *,
    young: float,
    poisson: float,
    density: float,
    pressure_stiffness: float,
    duration: float,
    dt: float,
    target_overclosure: float,
    rotation_rate_z: float,
    hht_alpha: float,
    max_iterations: int,
    tolerance: float,
    material_linearization: str,
    vtk_out_dir: Path | None = None,
    vtk_frame_stride: int = 1,
    vtk_stem: str = "sfc",
    vtk_include_tensors: bool = True,
) -> tuple[list[Row], Row]:
    """Solve linear penalty contact with exact residuals and a reused base tangent.

    The contact law and gap query are unchanged.  Only the Newton linearization
    is modified: each increment factors the HHT mass-plus-material tangent once
    and uses it for residual-correcting iterations.  If the correction does not
    converge, the increment falls back to the full tangent path.
    """

    n1 = pair.gear1.nodes.shape[0]
    X = np.vstack([pair.gear1.nodes, pair.gear2.nodes])
    elements = np.vstack([pair.gear1.elements, pair.gear2.elements + n1])
    model = MechanicsModel.from_tet4_mesh(X, elements, E=young, nu=poisson, density=density)
    master = MaterialSDF.from_triangle_surface(pair.gear2.nodes, pair.gear2.contact_faces)
    contact = LagrangianSDFSurfaceContactGeometry(
        pair.gear1.contact_faces,
        master,
        pair.gear2.nodes,
        pressure_stiffness=pressure_stiffness,
        slave_node_offset=0,
        master_node_offset=n1,
        quadrature="tri3",
        search_radius=_default_contact_search_radius(pair, target_overclosure=target_overclosure),
    )
    fixed0, values0 = _fixed_conditions_for_pair(pair, closure=0.0, rotation_z=0.0)
    state, previous = initial_state_dirichlet(model, contact, gravity=0.0, fixed_dofs=fixed0, fixed_values=values0)
    rows: list[Row] = []
    steps = max(1, int(round(float(duration) / float(dt))))
    beta, gamma = hht_newmark_parameters(float(hht_alpha))
    scale = 1.0 + float(hht_alpha)
    material_mode = str(material_linearization).lower()
    if material_mode not in {"stvk", "reference_linear"}:
        raise ValueError("material_linearization must be 'stvk' or 'reference_linear'")
    reference_tangent = None
    reference_base_free = None
    reference_lu_cache: list[Any] | None = None
    if material_mode == "reference_linear":
        reference_internal = stvk_internal_response(model, model.X, assemble_tangent=True)
        reference_tangent = reference_internal.tangent.tocsr()
    vtk_enabled = vtk_out_dir is not None and int(vtk_frame_stride) > 0
    vtk_dir = Path(vtk_out_dir) if vtk_out_dir is not None else None
    vtk_datasets: list[tuple[float, Path]] = []
    vtk_manifest_rows: list[Row] = []
    element_object_ids = np.concatenate(
        [
            np.ones(pair.gear1.elements.shape[0], dtype=np.int64),
            np.full(pair.gear2.elements.shape[0], 2, dtype=np.int64),
        ]
    )
    if vtk_enabled and vtk_dir is not None:
        vtk_dir.mkdir(parents=True, exist_ok=True)
        for stale in vtk_dir.glob(f"{vtk_stem}_*.vtk"):
            stale.unlink()
        for stale in (vtk_dir / f"{vtk_stem}.pvd", vtk_dir / f"{vtk_stem}_manifest.csv"):
            if stale.exists():
                stale.unlink()
        if material_mode == "reference_linear" and reference_tangent is not None:
            initial_internal = _linear_reference_internal_response(model, state.x, reference_tangent)
        else:
            initial_internal = stvk_internal_response(model, state.x, assemble_tangent=False)
        frame_path = vtk_dir / f"{vtk_stem}_{0:04d}.vtk"
        vtk_manifest_rows.append(
            _write_sfc_tet4_vtk_frame(
                frame_path,
                model=model,
                state=state,
                internal=initial_internal,
                element_object_ids=element_object_ids,
                time_value=0.0,
                frame_index=0,
                include_tensors=vtk_include_tensors,
            )
        )
        vtk_datasets.append((0.0, frame_path))
    start = time.perf_counter()
    timing_base_tangent = 0.0
    timing_base_factor = 0.0
    timing_residual = 0.0
    timing_linear_solve = 0.0
    timing_contact_low_rank = 0.0
    timing_fallback = 0.0
    fallback_count = 0
    vtk_frame_index = 1
    for step in range(1, steps + 1):
        t = step * float(dt)
        ramp = t / max(float(duration), float(dt))
        closure = ramp * (pair.initial_patch_gap + float(target_overclosure))
        rotation = float(rotation_rate_z) * t
        fixed, values = _fixed_conditions_for_pair(pair, closure=closure, rotation_z=rotation)
        free = free_dofs(model.n_dofs, fixed)
        c0 = 1.0 / (beta * float(dt) * float(dt))
        u = (state.x - model.X).reshape(-1)
        v = state.v.reshape(-1)
        a = state.a.reshape(-1)
        u_pred, v_pred, _accold_after_prediction = calculix_dynamic_predictor(u, v, a, dt=float(dt), beta=beta, gamma=gamma)
        u_guess = project_fixed_dofs(u_pred, fixed, values)
        if material_mode == "reference_linear":
            if reference_tangent is None:
                raise RuntimeError("reference tangent was not initialized")
            if reference_base_free is None:
                t_section = time.perf_counter()
                base_matrix = (model.mass_matrix * c0 + reference_tangent * scale).tocsr()
                reference_base_free = base_matrix[free[:, None], free].tocsr() if free.size else None
                reference_lu_cache = [None]
                timing_base_tangent += time.perf_counter() - t_section
            base_matrix_free = reference_base_free
            base_lu_cache = reference_lu_cache if reference_lu_cache is not None else [None]
        else:
            t_section = time.perf_counter()
            base_internal = stvk_internal_response(model, model.X + u_guess.reshape((-1, 3)), assemble_tangent=True)
            base_matrix = (model.mass_matrix * c0 + base_internal.tangent * scale).tocsr()
            timing_base_tangent += time.perf_counter() - t_section
            t_section = time.perf_counter()
            base_matrix_free = base_matrix[free[:, None], free].tocsr() if free.size else None
            base_lu_cache = [None]
            timing_base_factor += time.perf_counter() - t_section
        converged = False
        residual_norm = np.inf
        iteration_count = 0
        if material_mode == "reference_linear" and reference_tangent is not None:
            last_internal = _linear_reference_internal_response(model, model.X + u_guess.reshape((-1, 3)), reference_tangent)
        else:
            last_internal = stvk_internal_response(model, model.X + u_guess.reshape((-1, 3)), assemble_tangent=False)
        last_contact = assemble_contact_response(contact.samples(model.X + u_guess.reshape((-1, 3))), model.n_nodes)
        for iteration in range(1, max(1, int(max_iterations)) + 1):
            u_guess = project_fixed_dofs(u_guess, fixed, values)
            x_guess = model.X + u_guess.reshape((-1, 3))
            t_section = time.perf_counter()
            if material_mode == "reference_linear":
                if reference_tangent is None:
                    raise RuntimeError("reference tangent was not initialized")
                last_internal = _linear_reference_internal_response(model, x_guess, reference_tangent)
                internal_force = np.asarray(last_internal.force, dtype=float).reshape(-1)
            else:
                last_internal = stvk_internal_response(model, x_guess, assemble_tangent=False)
                internal_force = np.asarray(last_internal.force, dtype=float).reshape(-1)
            samples = list(contact.samples(x_guess))
            last_contact = assemble_contact_response(samples, model.n_nodes)
            rhs_balance = -internal_force + np.asarray(last_contact.force, dtype=float).reshape(-1)
            a_guess = c0 * (u_guess - u_pred)
            dynamic_residual = np.asarray(model.mass_matrix @ a_guess, dtype=float) - scale * rhs_balance + float(hht_alpha) * previous
            residual_norm = float(np.linalg.norm(dynamic_residual[free])) if free.size else 0.0
            timing_residual += time.perf_counter() - t_section
            t_section = time.perf_counter()
            correction_free = (
                _solve_penalty_low_rank_correction(
                    base_matrix_free=base_matrix_free,
                    base_lu_cache=base_lu_cache,
                    residual_free=dynamic_residual[free],
                    samples=samples,
                    free=free,
                    n_total_dofs=model.n_dofs,
                    equilibrium_scale=scale,
                    tolerance=float(tolerance),
                )
                if free.size
                else np.empty(0, dtype=float)
            )
            timing_linear_solve += time.perf_counter() - t_section
            timing_contact_low_rank += time.perf_counter() - t_section
            u_guess[free] += correction_free
            iteration_count = iteration
            correction_norm = float(np.linalg.norm(correction_free))
            displacement_scale = max(1.0, float(np.linalg.norm(u_guess[free])) if free.size else 1.0)
            if correction_norm <= float(tolerance) * displacement_scale:
                converged = True
                break
        if not converged:
            fallback_count += 1
            t_section = time.perf_counter()
            state, previous, diagnostics = hht_step_dirichlet(
                model,
                state,
                previous,
                contact,
                dt=float(dt),
                gravity=0.0,
                fixed_dofs=fixed,
                fixed_values=values,
                alpha=float(hht_alpha),
                max_iterations=max_iterations,
                tolerance=float(tolerance),
                acceptance_policy="relative_correction",
            )
            timing_fallback += time.perf_counter() - t_section
            internal = diagnostics.internal
            contact_response = diagnostics.contact
            row = _penalty_history_row(
                time_value=t,
                closure=closure,
                rotation=rotation,
                state=state,
                model=model,
                internal=internal,
                contact_response=contact_response,
                young=young,
                poisson=poisson,
                newton_iterations=diagnostics.newton_iterations,
                residual_norm=diagnostics.newton_residual_norm,
                solver="full_newton_fallback",
            )
            rows.append(row)
            if vtk_enabled and vtk_dir is not None and (step % int(vtk_frame_stride) == 0 or step == steps):
                frame_path = vtk_dir / f"{vtk_stem}_{vtk_frame_index:04d}.vtk"
                vtk_manifest_rows.append(
                    _write_sfc_tet4_vtk_frame(
                        frame_path,
                        model=model,
                        state=state,
                        internal=internal,
                        element_object_ids=element_object_ids,
                        time_value=t,
                        frame_index=vtk_frame_index,
                        include_tensors=vtk_include_tensors,
                    )
                )
                vtk_datasets.append((float(t), frame_path))
                vtk_frame_index += 1
            continue
        u_new = project_fixed_dofs(u_guess, fixed, values)
        a_new = c0 * (u_new - u_pred)
        v_new = v_pred + gamma * float(dt) * a_new
        state = MechanicsState(
            model.X + u_new.reshape((-1, 3)),
            v_new.reshape((-1, 3)),
            a_new.reshape((-1, 3)),
            time=state.time + float(dt),
        )
        if material_mode == "reference_linear":
            if reference_tangent is None:
                raise RuntimeError("reference tangent was not initialized")
            last_internal = _linear_reference_internal_response(model, state.x, reference_tangent)
        else:
            last_internal = stvk_internal_response(model, state.x, assemble_tangent=False)
        last_contact = assemble_contact_response(contact.samples(state.x), model.n_nodes)
        if material_mode == "reference_linear":
            if reference_tangent is None:
                raise RuntimeError("reference tangent was not initialized")
            accepted_internal_force = np.asarray(reference_tangent @ u_new, dtype=float)
            previous = -accepted_internal_force + np.asarray(last_contact.force, dtype=float).reshape(-1)
        else:
            previous = _penalty_rhs_balance(model, last_internal, last_contact)
        rows.append(
            _penalty_history_row(
                time_value=t,
                closure=closure,
                rotation=rotation,
                state=state,
                model=model,
                internal=last_internal,
                contact_response=last_contact,
                young=young,
                poisson=poisson,
                newton_iterations=iteration_count,
                residual_norm=residual_norm,
                solver="modified_newton",
            )
        )
        if vtk_enabled and vtk_dir is not None and (step % int(vtk_frame_stride) == 0 or step == steps):
            frame_path = vtk_dir / f"{vtk_stem}_{vtk_frame_index:04d}.vtk"
            vtk_manifest_rows.append(
                _write_sfc_tet4_vtk_frame(
                    frame_path,
                    model=model,
                    state=state,
                    internal=last_internal,
                element_object_ids=element_object_ids,
                time_value=t,
                frame_index=vtk_frame_index,
                include_tensors=vtk_include_tensors,
            )
        )
            vtk_datasets.append((float(t), frame_path))
            vtk_frame_index += 1
    wall = time.perf_counter() - start
    summary = {
        "sfc_wall_seconds": wall,
        "nodes": int(X.shape[0]),
        "elements": int(elements.shape[0]),
        "gear1_contact_faces": int(pair.gear1.contact_faces.shape[0]),
        "gear2_contact_faces": int(pair.gear2.contact_faces.shape[0]),
        "gear1_support_nodes": int(pair.gear1.support_nodes.size),
        "gear2_support_nodes": int(pair.gear2.support_nodes.size),
        "initial_patch_gap": float(pair.initial_patch_gap),
        "target_overclosure": float(target_overclosure),
        "final_active_contact_samples": int(rows[-1]["active_contact_samples"]) if rows else 0,
        "final_min_gap": float(rows[-1]["min_gap"]) if rows else 0.0,
        "final_normal_force": float(rows[-1]["normal_force"]) if rows else 0.0,
        "hht_alpha": float(hht_alpha),
        "contact_mode": "penalty",
        "penalty_solver": "modified_newton",
        "material_linearization": material_mode,
        "penalty_fallback_count": int(fallback_count),
        "timing_base_tangent_seconds": float(timing_base_tangent),
        "timing_base_factor_seconds": float(timing_base_factor),
        "timing_penalty_residual_seconds": float(timing_residual),
        "timing_penalty_linear_solve_seconds": float(timing_linear_solve),
        "timing_penalty_low_rank_seconds": float(timing_contact_low_rank),
        "timing_penalty_fallback_seconds": float(timing_fallback),
        "status": "completed",
    }
    if vtk_enabled and vtk_dir is not None:
        pvd_path = vtk_dir / f"{vtk_stem}.pvd"
        manifest_path = vtk_dir / f"{vtk_stem}_manifest.csv"
        write_pvd(pvd_path, vtk_datasets)
        _write_csv(manifest_path, vtk_manifest_rows)
        summary.update(
            {
                "sfc_vtk_pvd": str(pvd_path),
                "sfc_vtk_manifest": str(manifest_path),
                "sfc_vtk_frame_count": int(len(vtk_datasets)),
                "sfc_vtk_frame_stride": int(vtk_frame_stride),
                "sfc_vtk_include_tensors": bool(vtk_include_tensors),
            }
        )
    return rows, summary


def _source_drive_fixed_reduced_dofs(assembly: Any, *, time_value: float, omega_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return reduced RP constraints matching the source Abaqus gear-contact deck."""

    hub1 = assembly.hub_slice(0)
    hub2 = assembly.hub_slice(1)
    fixed = [
        hub1.start + 0,
        hub1.start + 1,
        hub1.start + 2,
        hub1.start + 3,
        hub1.start + 4,
        hub1.start + 5,
        hub2.start + 0,
        hub2.start + 1,
        hub2.start + 2,
        hub2.start + 3,
        hub2.start + 4,
    ]
    values = [0.0] * len(fixed)
    values[5] = float(omega_z) * float(time_value)
    return np.asarray(fixed, dtype=np.int64), np.asarray(values, dtype=float)


def _project_reduced_fixed(q: np.ndarray, fixed: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=float).reshape(-1).copy()
    if fixed.size:
        out[np.asarray(fixed, dtype=np.int64)] = np.asarray(values, dtype=float)
    return out


def _write_source_drive_checkpoint(
    path: Path,
    *,
    step: int,
    time_value: float,
    q: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    previous: np.ndarray,
    rows: list[Row],
    vtk_manifest_rows: list[Row],
    vtk_frame_index: int,
    timing_residual: float,
    timing_linear: float,
    timing_history: float,
    timing_vtk: float,
    timing_source_base_lu: float,
    source_sparse_cg_count: int,
    source_sparse_cg_iterations: int,
    source_sparse_cg_base_lu_preconditioner: int,
    source_direct_fallback_count: int,
    source_step_converged_count: int = 0,
    source_iteration_limit_reached_count: int = 0,
    source_unstable_accepted_count: int = 0,
    source_cutback_required_count: int = 0,
    source_unconverged_accepted_count: int = 0,
    source_line_search_trial_count: int = 0,
    source_line_search_reduced_count: int = 0,
    source_line_search_stable_count: int = 0,
    source_line_search_unstable_count: int = 0,
    source_accepted_contact_response_reuse_count: int = 0,
    source_accepted_contact_response_requery_count: int = 0,
    source_accepted_tracking_commit_count: int = 0,
    source_constraint_region_tangent_solve_count: int = 0,
    source_constraint_region_tangent_active_rows_sum: int = 0,
    source_constraint_region_tangent_active_rows_max: int = 0,
    source_constraint_region_tangent_j_nnz_sum: int = 0,
    source_constraint_region_tangent_scale_sum: float = 0.0,
    source_constraint_region_tangent_scale_max: float = 0.0,
) -> None:
    """Persist accepted source-drive state for exact fixed-step continuation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            version=np.asarray([1], dtype=np.int64),
            step=np.asarray([int(step)], dtype=np.int64),
            time_value=np.asarray([float(time_value)], dtype=float),
            q=np.asarray(q, dtype=float),
            v=np.asarray(v, dtype=float),
            a=np.asarray(a, dtype=float),
            previous=np.asarray(previous, dtype=float),
            rows_json=np.asarray([json.dumps(rows, default=_json_default)], dtype=object),
            vtk_manifest_json=np.asarray([json.dumps(vtk_manifest_rows, default=_json_default)], dtype=object),
            vtk_frame_index=np.asarray([int(vtk_frame_index)], dtype=np.int64),
            timing_residual=np.asarray([float(timing_residual)], dtype=float),
            timing_linear=np.asarray([float(timing_linear)], dtype=float),
            timing_history=np.asarray([float(timing_history)], dtype=float),
            timing_vtk=np.asarray([float(timing_vtk)], dtype=float),
            timing_source_base_lu=np.asarray([float(timing_source_base_lu)], dtype=float),
            source_sparse_cg_count=np.asarray([int(source_sparse_cg_count)], dtype=np.int64),
            source_sparse_cg_iterations=np.asarray([int(source_sparse_cg_iterations)], dtype=np.int64),
            source_sparse_cg_base_lu_preconditioner=np.asarray(
                [int(source_sparse_cg_base_lu_preconditioner)], dtype=np.int64
            ),
            source_direct_fallback_count=np.asarray([int(source_direct_fallback_count)], dtype=np.int64),
            source_step_converged_count=np.asarray([int(source_step_converged_count)], dtype=np.int64),
            source_iteration_limit_reached_count=np.asarray([int(source_iteration_limit_reached_count)], dtype=np.int64),
            source_unstable_accepted_count=np.asarray([int(source_unstable_accepted_count)], dtype=np.int64),
            source_cutback_required_count=np.asarray([int(source_cutback_required_count)], dtype=np.int64),
            source_unconverged_accepted_count=np.asarray([int(source_unconverged_accepted_count)], dtype=np.int64),
            source_line_search_trial_count=np.asarray([int(source_line_search_trial_count)], dtype=np.int64),
            source_line_search_reduced_count=np.asarray([int(source_line_search_reduced_count)], dtype=np.int64),
            source_line_search_stable_count=np.asarray([int(source_line_search_stable_count)], dtype=np.int64),
            source_line_search_unstable_count=np.asarray([int(source_line_search_unstable_count)], dtype=np.int64),
            source_accepted_contact_response_reuse_count=np.asarray(
                [int(source_accepted_contact_response_reuse_count)], dtype=np.int64
            ),
            source_accepted_contact_response_requery_count=np.asarray(
                [int(source_accepted_contact_response_requery_count)], dtype=np.int64
            ),
            source_accepted_tracking_commit_count=np.asarray(
                [int(source_accepted_tracking_commit_count)], dtype=np.int64
            ),
            source_constraint_region_tangent_solve_count=np.asarray(
                [int(source_constraint_region_tangent_solve_count)], dtype=np.int64
            ),
            source_constraint_region_tangent_active_rows_sum=np.asarray(
                [int(source_constraint_region_tangent_active_rows_sum)], dtype=np.int64
            ),
            source_constraint_region_tangent_active_rows_max=np.asarray(
                [int(source_constraint_region_tangent_active_rows_max)], dtype=np.int64
            ),
            source_constraint_region_tangent_j_nnz_sum=np.asarray(
                [int(source_constraint_region_tangent_j_nnz_sum)], dtype=np.int64
            ),
            source_constraint_region_tangent_scale_sum=np.asarray(
                [float(source_constraint_region_tangent_scale_sum)], dtype=float
            ),
            source_constraint_region_tangent_scale_max=np.asarray(
                [float(source_constraint_region_tangent_scale_max)], dtype=float
            ),
        )
    tmp.replace(path)


def _load_source_drive_checkpoint(path: Path) -> dict[str, Any]:
    """Load a source-drive checkpoint written after an accepted step."""

    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {
            "step": int(data["step"][0]),
            "time_value": float(data["time_value"][0]) if "time_value" in data else float("nan"),
            "q": np.asarray(data["q"], dtype=float),
            "v": np.asarray(data["v"], dtype=float),
            "a": np.asarray(data["a"], dtype=float),
            "previous": np.asarray(data["previous"], dtype=float),
            "rows": json.loads(str(data["rows_json"][0])),
            "vtk_manifest_rows": json.loads(str(data["vtk_manifest_json"][0])),
            "vtk_frame_index": int(data["vtk_frame_index"][0]),
            "timing_residual": float(data["timing_residual"][0]),
            "timing_linear": float(data["timing_linear"][0]),
            "timing_history": float(data["timing_history"][0]),
            "timing_vtk": float(data["timing_vtk"][0]),
            "timing_source_base_lu": float(data["timing_source_base_lu"][0]),
            "source_sparse_cg_count": int(data["source_sparse_cg_count"][0]),
            "source_sparse_cg_iterations": int(data["source_sparse_cg_iterations"][0]),
            "source_sparse_cg_base_lu_preconditioner": int(data["source_sparse_cg_base_lu_preconditioner"][0]),
            "source_direct_fallback_count": int(data["source_direct_fallback_count"][0]),
            "source_step_converged_count": int(data["source_step_converged_count"][0]) if "source_step_converged_count" in data else 0,
            "source_iteration_limit_reached_count": (
                int(data["source_iteration_limit_reached_count"][0]) if "source_iteration_limit_reached_count" in data else 0
            ),
            "source_unstable_accepted_count": (
                int(data["source_unstable_accepted_count"][0]) if "source_unstable_accepted_count" in data else 0
            ),
            "source_cutback_required_count": (
                int(data["source_cutback_required_count"][0]) if "source_cutback_required_count" in data else 0
            ),
            "source_unconverged_accepted_count": (
                int(data["source_unconverged_accepted_count"][0]) if "source_unconverged_accepted_count" in data else 0
            ),
            "source_line_search_trial_count": (
                int(data["source_line_search_trial_count"][0]) if "source_line_search_trial_count" in data else 0
            ),
            "source_line_search_reduced_count": (
                int(data["source_line_search_reduced_count"][0]) if "source_line_search_reduced_count" in data else 0
            ),
            "source_line_search_stable_count": (
                int(data["source_line_search_stable_count"][0]) if "source_line_search_stable_count" in data else 0
            ),
            "source_line_search_unstable_count": (
                int(data["source_line_search_unstable_count"][0]) if "source_line_search_unstable_count" in data else 0
            ),
            "source_accepted_contact_response_reuse_count": (
                int(data["source_accepted_contact_response_reuse_count"][0])
                if "source_accepted_contact_response_reuse_count" in data
                else 0
            ),
            "source_accepted_contact_response_requery_count": (
                int(data["source_accepted_contact_response_requery_count"][0])
                if "source_accepted_contact_response_requery_count" in data
                else 0
            ),
            "source_accepted_tracking_commit_count": (
                int(data["source_accepted_tracking_commit_count"][0])
                if "source_accepted_tracking_commit_count" in data
                else 0
            ),
            "source_constraint_region_tangent_solve_count": (
                int(data["source_constraint_region_tangent_solve_count"][0])
                if "source_constraint_region_tangent_solve_count" in data
                else 0
            ),
            "source_constraint_region_tangent_active_rows_sum": (
                int(data["source_constraint_region_tangent_active_rows_sum"][0])
                if "source_constraint_region_tangent_active_rows_sum" in data
                else 0
            ),
            "source_constraint_region_tangent_active_rows_max": (
                int(data["source_constraint_region_tangent_active_rows_max"][0])
                if "source_constraint_region_tangent_active_rows_max" in data
                else 0
            ),
            "source_constraint_region_tangent_j_nnz_sum": (
                int(data["source_constraint_region_tangent_j_nnz_sum"][0])
                if "source_constraint_region_tangent_j_nnz_sum" in data
                else 0
            ),
            "source_constraint_region_tangent_scale_sum": (
                float(data["source_constraint_region_tangent_scale_sum"][0])
                if "source_constraint_region_tangent_scale_sum" in data
                else 0.0
            ),
            "source_constraint_region_tangent_scale_max": (
                float(data["source_constraint_region_tangent_scale_max"][0])
                if "source_constraint_region_tangent_scale_max" in data
                else 0.0
            ),
        }


def solve_sfc_source_drive_pair(
    pair: CroppedGearPair,
    *,
    young: float,
    poisson: float,
    density: float,
    pressure_stiffness: float,
    duration: float,
    dt: float,
    gear1_angular_velocity_z: float,
    gear2_torque_z: float,
    hht_alpha: float = ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
    tet4_mass_kind: str = "consistent",
    max_iterations: int = 16,
    tolerance: float = 1.0e-9,
    vtk_out_dir: Path | None = None,
    vtk_frame_stride: int = 1,
    history_frame_stride: int = 1,
    vtk_stem: str = "sfc",
    vtk_include_tensors: bool = True,
    source_stress_postprocess: str = "linear_corotated",
    source_contact_averaging: str = "slave_node_region_constraint",
    source_contact_kinematics: str = "finite_rp_corotated",
    source_contact_normal_filter: str = "opposing",
    source_contact_direction: str = "secondary_average",
    source_contact_projection: str = "secondary_line",
    source_contact_pair_order: str = "gear2_slave",
    source_contact_search_radius: float | None = None,
    source_secondary_normal_min_projection: float = 0.0,
    source_secondary_line_distance_limit: float | None = None,
    source_secondary_line_hard_distance_limit: float | None = None,
    source_secondary_path_tracking: bool = True,
    source_contact_active_set_stability: bool = True,
    source_contact_footprint_clipping: bool = False,
    source_internal_kinematics: str = "finite_stvk_visual",
    source_rotating_inertia: str = "finite_kinematic",
    source_residual_tolerance: float = 5.0e-3,
    source_correction_tolerance: float = 1.0e-2,
    source_contact_force_increment_tolerance: float = 1.0e-2,
    source_accept_unconverged: bool = False,
    source_active_set_line_search: bool = True,
    source_active_set_line_search_min_alpha: float = 0.125,
    source_cutback_factor: float = 0.5,
    source_min_cutback_dt: float | None = None,
    source_checkpoint_path: Path | None = None,
    resume_source_checkpoint: bool = False,
    source_checkpoint_stride: int = 10,
) -> tuple[list[Row], Row]:
    """Solve the source ``gear_contact.inp`` RP-drive case with SFC contact.

    The model keeps both flexible gear volume meshes.  Gear 1 follows the
    Abaqus source velocity boundary on RP dof 6, gear 2 keeps RP dofs 1--5
    fixed and receives the source moment on RP dof 6.  Contact uses gear 2 as
    slave and gear 1 as master, matching the source contact-pair order.
    """

    n1 = pair.gear1.nodes.shape[0]
    X = np.vstack([pair.gear1.nodes, pair.gear2.nodes])
    elements = np.vstack([pair.gear1.elements, pair.gear2.elements + n1])
    model = MechanicsModel.from_tet4_mesh(X, elements, E=young, nu=poisson, density=density, mass_kind=tet4_mass_kind)
    hub1 = RigidHubMPC(pair.gear1.support_nodes, X, pair.gear1.rp)
    hub2 = RigidHubMPC(pair.gear2.support_nodes + n1, X, pair.gear2.rp)
    assembly = build_rigid_hub_reduced_assembly(X, [hub1, hub2], include_free_nodes=True)
    T = assembly.transformation.tocsr()
    contact_pair_order = str(source_contact_pair_order).lower()
    if contact_pair_order not in {"gear2_slave", "gear1_slave", "symmetric_two_pass"}:
        raise ValueError("source_contact_pair_order must be 'gear2_slave', 'gear1_slave', or 'symmetric_two_pass'")
    if source_contact_search_radius is None:
        if str(source_contact_direction).lower() == "secondary_average" and str(source_contact_projection).lower() == "secondary_line":
            contact_search_radius = _default_secondary_contact_tracking_radius(pair, target_overclosure=1.0e-5)
        else:
            contact_search_radius = _default_contact_search_radius(pair, target_overclosure=1.0e-5)
    else:
        contact_search_radius = float(source_contact_search_radius)
    if contact_search_radius < 0.0:
        raise ValueError("source_contact_search_radius must be non-negative")
    secondary_line_distance_limit = None
    secondary_line_hard_distance_limit = None
    if str(source_contact_direction).lower() == "secondary_average" and str(source_contact_projection).lower() == "secondary_line":
        if source_secondary_line_distance_limit is None:
            secondary_line_distance_limit = _default_secondary_line_distance_limit(pair, target_overclosure=1.0e-5)
        elif float(source_secondary_line_distance_limit) >= 0.0:
            secondary_line_distance_limit = float(source_secondary_line_distance_limit)
        if source_secondary_line_hard_distance_limit is None:
            secondary_line_hard_distance_limit = _default_secondary_line_hard_distance_limit(pair, target_overclosure=1.0e-5)
        elif float(source_secondary_line_hard_distance_limit) >= 0.0:
            secondary_line_hard_distance_limit = float(source_secondary_line_hard_distance_limit)
    def make_contact(order: str, stiffness: float) -> LagrangianSDFSurfaceContactGeometry:
        if order == "gear2_slave":
            master = MaterialSDF.from_triangle_surface(pair.gear1.nodes, pair.gear1.contact_faces)
            return LagrangianSDFSurfaceContactGeometry(
                pair.gear2.contact_faces,
                master,
                pair.gear1.nodes,
                pressure_stiffness=stiffness,
                slave_node_offset=n1,
                master_node_offset=0,
                quadrature="tri3",
                search_radius=contact_search_radius,
                compiled_batch_projection=True,
                clip_to_master_footprint=bool(source_contact_footprint_clipping),
                secondary_path_tracking=bool(source_secondary_path_tracking),
                secondary_line_distance_limit=secondary_line_distance_limit,
                secondary_line_hard_distance_limit=secondary_line_hard_distance_limit,
            )
        master = MaterialSDF.from_triangle_surface(pair.gear2.nodes, pair.gear2.contact_faces)
        return LagrangianSDFSurfaceContactGeometry(
            pair.gear1.contact_faces,
            master,
            pair.gear2.nodes,
            pressure_stiffness=stiffness,
            slave_node_offset=0,
            master_node_offset=n1,
            quadrature="tri3",
            search_radius=contact_search_radius,
            compiled_batch_projection=True,
            clip_to_master_footprint=bool(source_contact_footprint_clipping),
            secondary_path_tracking=bool(source_secondary_path_tracking),
            secondary_line_distance_limit=secondary_line_distance_limit,
            secondary_line_hard_distance_limit=secondary_line_hard_distance_limit,
        )

    if contact_pair_order == "symmetric_two_pass":
        contact_geometries = (
            make_contact("gear2_slave", 0.5 * float(pressure_stiffness)),
            make_contact("gear1_slave", 0.5 * float(pressure_stiffness)),
        )
    else:
        contact_geometries = (make_contact(contact_pair_order, float(pressure_stiffness)),)
    contact_averaging = str(source_contact_averaging).lower()
    valid_contact_averaging = {
        "none",
        "slave_face",
        "slave_node",
        "slave_node_region",
        "slave_node_point",
        "surface_patch",
        "slave_face_constraint",
        "slave_node_constraint",
        "slave_node_region_constraint",
        "surface_patch_constraint",
        "slave_face_area_average",
        "slave_node_area_average",
        "slave_node_region_area_average",
        "surface_patch_area_average",
        "slave_face_participation",
        "slave_node_participation",
        "slave_node_region_participation",
        "surface_patch_participation",
        "slave_face_signed_participation",
        "slave_node_signed_participation",
        "slave_node_region_signed_participation",
        "surface_patch_signed_participation",
    }
    if contact_averaging not in valid_contact_averaging:
        raise ValueError(
            "source_contact_averaging must be 'none', 'slave_face', 'slave_node', 'slave_node_region', "
            "'slave_node_point', 'surface_patch', or a supported suffix mode"
        )
    contact_kinematics = str(source_contact_kinematics).lower()
    if contact_kinematics not in {"linearized_mpc", "finite_rp_corotated"}:
        raise ValueError("source_contact_kinematics must be 'linearized_mpc' or 'finite_rp_corotated'")
    contact_normal_filter = str(source_contact_normal_filter).lower()
    if contact_normal_filter not in {"none", "opposing", "opposing_search"}:
        raise ValueError("source_contact_normal_filter must be 'none', 'opposing', or 'opposing_search'")
    contact_direction = str(source_contact_direction).lower()
    if contact_direction not in {"master", "secondary_average"}:
        raise ValueError("source_contact_direction must be 'master' or 'secondary_average'")
    contact_projection = str(source_contact_projection).lower()
    if contact_projection not in {"closest_feature", "secondary_plane", "secondary_line"}:
        raise ValueError("source_contact_projection must be 'closest_feature', 'secondary_plane', or 'secondary_line'")
    if contact_projection != "closest_feature" and contact_direction != "secondary_average":
        raise ValueError("secondary contact projection modes require source_contact_direction='secondary_average'")
    secondary_normal_min_projection = float(source_secondary_normal_min_projection)
    if not (0.0 <= secondary_normal_min_projection < 1.0):
        raise ValueError("source_secondary_normal_min_projection must be in [0, 1)")
    secondary_dot_threshold = -secondary_normal_min_projection
    internal_kinematics = str(source_internal_kinematics).lower()
    if internal_kinematics not in {"linearized_mpc", "corotated_rp", "finite_stvk_visual"}:
        raise ValueError(
            "source_internal_kinematics must be 'linearized_mpc', 'corotated_rp', or 'finite_stvk_visual'"
        )
    rotating_inertia = str(source_rotating_inertia).lower()
    if rotating_inertia not in {"none", "centripetal", "finite_kinematic"}:
        raise ValueError("source_rotating_inertia must be 'none', 'centripetal', or 'finite_kinematic'")
    source_residual_tolerance = float(source_residual_tolerance)
    source_correction_tolerance = float(source_correction_tolerance)
    source_contact_force_increment_tolerance = float(source_contact_force_increment_tolerance)
    if source_residual_tolerance <= 0.0:
        raise ValueError("source_residual_tolerance must be positive")
    if source_correction_tolerance <= 0.0:
        raise ValueError("source_correction_tolerance must be positive")
    if source_contact_force_increment_tolerance <= 0.0:
        raise ValueError("source_contact_force_increment_tolerance must be positive")
    source_active_set_line_search_min_alpha = float(source_active_set_line_search_min_alpha)
    if not (0.0 < source_active_set_line_search_min_alpha <= 1.0):
        raise ValueError("source_active_set_line_search_min_alpha must lie in (0, 1]")
    source_cutback_factor = float(source_cutback_factor)
    if not (0.0 < source_cutback_factor < 1.0):
        raise ValueError("source_cutback_factor must lie in (0, 1)")
    source_min_cutback_dt_value = (
        min(float(dt) * 1.0e-4, 1.0e-8) if source_min_cutback_dt is None else float(source_min_cutback_dt)
    )
    if source_min_cutback_dt_value <= 0.0:
        raise ValueError("source_min_cutback_dt must be positive")
    beta, gamma = hht_newmark_parameters(float(hht_alpha))
    scale = 1.0 + float(hht_alpha)
    steps = max(1, int(round(float(duration) / float(dt))))
    if not np.isclose(steps * float(dt), float(duration), rtol=1.0e-9, atol=1.0e-14):
        raise ValueError("duration must be an integer multiple of dt for source-drive SFC")
    reference_internal = stvk_internal_response(model, model.X, assemble_tangent=True)
    K_full = reference_internal.tangent.tocsr()
    K_red = assembly.reduce_matrix(K_full).tocsr()
    M_red = assembly.reduce_matrix(model.mass_matrix).tocsr()
    external = np.zeros(assembly.n_reduced_dofs, dtype=float)
    hub2_slice = assembly.hub_slice(1)
    external[hub2_slice.start + 5] = float(gear2_torque_z)
    q = np.zeros(assembly.n_reduced_dofs, dtype=float)
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    v[assembly.hub_slice(0).start + 5] = float(gear1_angular_velocity_z)
    fixed0, values0 = _source_drive_fixed_reduced_dofs(assembly, time_value=0.0, omega_z=gear1_angular_velocity_z)
    q = _project_reduced_fixed(q, fixed0, values0)
    x0 = model.X + assembly.expand_displacements(q)
    # Abaqus/Standard starts a direct-integration dynamic step from the
    # prescribed initial acceleration field, which is zero unless explicitly
    # initialized.  For a newly started step, the HHT history force is the
    # accepted balance from the end of the previous step, not the load that is
    # first applied in this step.  The source gear deck has no prior preload
    # step, so the HHT ``B_ini`` vector is zero.
    a[:] = 0.0
    previous = np.zeros_like(external)
    contact0 = ContactResponse(
        np.zeros_like(model.X),
        csr_matrix((model.n_dofs, model.n_dofs), dtype=float),
        0.0,
        0.0,
        0,
        0.0,
        0.0,
    )
    state = MechanicsState(x0, assembly.expand_displacements(v), assembly.expand_displacements(a), time=0.0)
    element_object_ids = np.concatenate(
        [
            np.ones(pair.gear1.elements.shape[0], dtype=np.int64),
            np.full(pair.gear2.elements.shape[0], 2, dtype=np.int64),
        ]
    )
    body_node_slices = (slice(0, n1), slice(n1, X.shape[0]))
    body_reference_points = (np.asarray(pair.gear1.rp, dtype=float), np.asarray(pair.gear2.rp, dtype=float))
    elastic_map = assembly.transformation
    if internal_kinematics in {"corotated_rp", "finite_stvk_visual"}:
        elastic_map = _source_drive_corotated_elastic_matrix(
            assembly=assembly,
            reference_nodes=model.X,
            body_node_slices=body_node_slices,
            body_reference_points=body_reference_points,
        )

    def contact_positions_from_reduced(q_reduced: np.ndarray) -> np.ndarray:
        x_linear = model.X + assembly.expand_displacements(q_reduced)
        if contact_kinematics == "linearized_mpc":
            return x_linear
        return _source_drive_corotated_positions_and_elastic_displacement(
            model=model,
            x_raw=x_linear,
            body_node_slices=body_node_slices,
            body_reference_points=body_reference_points,
            body_rotation_z=(
                float(q_reduced[assembly.hub_slice(0).start + 5]),
                float(q_reduced[hub2_slice.start + 5]),
            ),
        )[0]

    def internal_reduced_response_from_state(
        q_reduced: np.ndarray,
        *,
        assemble_tangent: bool = False,
    ) -> tuple[np.ndarray, csr_matrix | None]:
        if internal_kinematics != "finite_stvk_visual":
            tangent = K_solve if assemble_tangent else None
            return np.asarray(K_solve @ q_reduced, dtype=float).reshape(-1), tangent
        x_linear = model.X + assembly.expand_displacements(q_reduced)
        theta = (
            float(q_reduced[assembly.hub_slice(0).start + 5]),
            float(q_reduced[hub2_slice.start + 5]),
        )
        x_visual = _source_drive_corotated_positions_and_elastic_displacement(
            model=model,
            x_raw=x_linear,
            body_node_slices=body_node_slices,
            body_reference_points=body_reference_points,
            body_rotation_z=theta,
        )[0]
        internal = stvk_internal_response(model, x_visual, assemble_tangent=bool(assemble_tangent))
        visual_jacobian = _source_drive_finite_visual_jacobian(
            assembly=assembly,
            reference_nodes=model.X,
            body_node_slices=body_node_slices,
            body_reference_points=body_reference_points,
            body_rotation_z=theta,
        )
        reduced_force = np.asarray(visual_jacobian.T @ np.asarray(internal.force, dtype=float).reshape(-1), dtype=float).reshape(-1)
        if not assemble_tangent:
            return reduced_force, None
        reduced_tangent = (visual_jacobian.T @ internal.tangent @ visual_jacobian).tocsr()
        return reduced_force, reduced_tangent

    def internal_reduced_from_state(q_reduced: np.ndarray) -> np.ndarray:
        force, _tangent = internal_reduced_response_from_state(q_reduced, assemble_tangent=False)
        return force

    current_source_step_dt = float(dt)

    def centripetal_reduced_from_state(q_reduced: np.ndarray, v_reduced: np.ndarray) -> np.ndarray:
        if rotating_inertia == "none":
            return np.zeros(assembly.n_reduced_dofs, dtype=float)
        reduced, _tangent = _source_drive_centripetal_reduced_response(
            assembly=assembly,
            mass_matrix=model.mass_matrix,
            reference_nodes=model.X,
            body_node_slices=body_node_slices,
            body_reference_points=body_reference_points,
            body_rotation_z=(
                float(q_reduced[assembly.hub_slice(0).start + 5]),
                float(q_reduced[hub2_slice.start + 5]),
            ),
            body_angular_velocity_z=(
                float(v_reduced[assembly.hub_slice(0).start + 5]),
                float(v_reduced[hub2_slice.start + 5]),
            ),
            include_tangent=False,
        )
        return reduced

    def centripetal_reduced_and_tangent_for_iteration(q_reduced: np.ndarray, v_reduced: np.ndarray) -> tuple[np.ndarray, csr_matrix]:
        if rotating_inertia == "none":
            empty = csr_matrix((assembly.n_reduced_dofs, assembly.n_reduced_dofs), dtype=float)
            return np.zeros(assembly.n_reduced_dofs, dtype=float), empty
        if rotating_inertia == "finite_kinematic":
            empty = csr_matrix((assembly.n_reduced_dofs, assembly.n_reduced_dofs), dtype=float)
            return np.zeros(assembly.n_reduced_dofs, dtype=float), empty
        sensitivity = gamma * float(current_source_step_dt) * c0
        return _source_drive_centripetal_reduced_response(
            assembly=assembly,
            mass_matrix=model.mass_matrix,
            reference_nodes=model.X,
            body_node_slices=body_node_slices,
            body_reference_points=body_reference_points,
            body_rotation_z=(
                float(q_reduced[assembly.hub_slice(0).start + 5]),
                float(q_reduced[hub2_slice.start + 5]),
            ),
            body_angular_velocity_z=(
                float(v_reduced[assembly.hub_slice(0).start + 5]),
                float(v_reduced[hub2_slice.start + 5]),
            ),
            velocity_sensitivity_z=(float(sensitivity), float(sensitivity)),
            include_tangent=True,
        )

    def inertia_reduced_and_mass_for_iteration(
        q_reduced: np.ndarray,
        v_reduced: np.ndarray,
        a_reduced: np.ndarray,
    ) -> tuple[np.ndarray, csr_matrix]:
        if rotating_inertia != "finite_kinematic":
            empty = csr_matrix((assembly.n_reduced_dofs, assembly.n_reduced_dofs), dtype=float)
            return np.asarray(M_red @ a_reduced, dtype=float).reshape(-1), empty
        return _source_drive_finite_kinematic_inertia_response(
            assembly=assembly,
            mass_matrix=model.mass_matrix,
            reference_nodes=model.X,
            body_node_slices=body_node_slices,
            body_reference_points=body_reference_points,
            body_rotation_z=(
                float(q_reduced[assembly.hub_slice(0).start + 5]),
                float(q_reduced[hub2_slice.start + 5]),
            ),
            body_angular_velocity_z=(
                float(v_reduced[assembly.hub_slice(0).start + 5]),
                float(v_reduced[hub2_slice.start + 5]),
            ),
            reduced_acceleration=a_reduced,
            include_mass_tangent=True,
        )

    vtk_enabled = vtk_out_dir is not None and int(vtk_frame_stride) > 0
    history_stride = max(1, int(history_frame_stride))
    vtk_dir = Path(vtk_out_dir) if vtk_out_dir is not None else None
    vtk_datasets: list[tuple[float, Path]] = []
    vtk_manifest_rows: list[Row] = []
    vtk_static_blocks = _build_sfc_tet4_vtk_static_blocks(model, element_object_ids) if vtk_enabled else None
    postprocess_label = str(source_stress_postprocess)
    rows: list[Row] = []
    source_increment_trial_rows: list[Row] = []
    vtk_frame_index = 1
    timing_residual = 0.0
    timing_linear = 0.0
    timing_history = 0.0
    timing_vtk = 0.0
    source_sparse_cg_count = 0
    source_sparse_cg_iterations = 0
    source_sparse_cg_base_lu_preconditioner = 0
    source_direct_fallback_count = 0
    timing_source_base_lu = 0.0
    source_step_converged_count = 0
    source_iteration_limit_reached_count = 0
    source_unstable_accepted_count = 0
    source_cutback_required_count = 0
    source_unconverged_accepted_count = 0
    source_line_search_trial_count = 0
    source_line_search_reduced_count = 0
    source_line_search_stable_count = 0
    source_line_search_unstable_count = 0
    source_accepted_contact_response_reuse_count = 0
    source_accepted_contact_response_requery_count = 0
    source_accepted_tracking_commit_count = 0
    source_constraint_region_tangent_solve_count = 0
    source_constraint_region_tangent_active_rows_sum = 0
    source_constraint_region_tangent_active_rows_max = 0
    source_constraint_region_tangent_j_nnz_sum = 0
    source_constraint_region_tangent_scale_sum = 0.0
    source_constraint_region_tangent_scale_max = 0.0
    checkpoint_path = Path(source_checkpoint_path) if source_checkpoint_path is not None else None
    start_step = 0
    start_time = 0.0
    if bool(resume_source_checkpoint):
        if checkpoint_path is None:
            raise ValueError("resume_source_checkpoint requires source_checkpoint_path")
        checkpoint = _load_source_drive_checkpoint(checkpoint_path)
        start_step = int(checkpoint["step"])
        loaded_time = float(checkpoint.get("time_value", np.nan))
        start_time = loaded_time if np.isfinite(loaded_time) else float(start_step) * float(dt)
        if start_time > float(duration) + 1.0e-14:
            raise ValueError("checkpoint time is beyond requested duration")
        q = np.asarray(checkpoint["q"], dtype=float).copy()
        v = np.asarray(checkpoint["v"], dtype=float).copy()
        a = np.asarray(checkpoint["a"], dtype=float).copy()
        previous = np.asarray(checkpoint["previous"], dtype=float).copy()
        rows = list(checkpoint["rows"])
        vtk_manifest_rows = list(checkpoint["vtk_manifest_rows"])
        vtk_frame_index = int(checkpoint["vtk_frame_index"])
        timing_residual = float(checkpoint["timing_residual"])
        timing_linear = float(checkpoint["timing_linear"])
        timing_history = float(checkpoint["timing_history"])
        timing_vtk = float(checkpoint["timing_vtk"])
        timing_source_base_lu = float(checkpoint["timing_source_base_lu"])
        source_sparse_cg_count = int(checkpoint["source_sparse_cg_count"])
        source_sparse_cg_iterations = int(checkpoint["source_sparse_cg_iterations"])
        source_sparse_cg_base_lu_preconditioner = int(checkpoint["source_sparse_cg_base_lu_preconditioner"])
        source_direct_fallback_count = int(checkpoint["source_direct_fallback_count"])
        source_step_converged_count = int(checkpoint.get("source_step_converged_count", 0))
        source_iteration_limit_reached_count = int(checkpoint.get("source_iteration_limit_reached_count", 0))
        source_unstable_accepted_count = int(checkpoint.get("source_unstable_accepted_count", 0))
        source_cutback_required_count = int(checkpoint.get("source_cutback_required_count", 0))
        source_unconverged_accepted_count = int(checkpoint.get("source_unconverged_accepted_count", 0))
        source_line_search_trial_count = int(checkpoint.get("source_line_search_trial_count", 0))
        source_line_search_reduced_count = int(checkpoint.get("source_line_search_reduced_count", 0))
        source_line_search_stable_count = int(checkpoint.get("source_line_search_stable_count", 0))
        source_line_search_unstable_count = int(checkpoint.get("source_line_search_unstable_count", 0))
        source_accepted_contact_response_reuse_count = int(
            checkpoint.get("source_accepted_contact_response_reuse_count", 0)
        )
        source_accepted_contact_response_requery_count = int(
            checkpoint.get("source_accepted_contact_response_requery_count", 0)
        )
        source_accepted_tracking_commit_count = int(checkpoint.get("source_accepted_tracking_commit_count", 0))
        source_constraint_region_tangent_solve_count = int(
            checkpoint.get("source_constraint_region_tangent_solve_count", 0)
        )
        source_constraint_region_tangent_active_rows_sum = int(
            checkpoint.get("source_constraint_region_tangent_active_rows_sum", 0)
        )
        source_constraint_region_tangent_active_rows_max = int(
            checkpoint.get("source_constraint_region_tangent_active_rows_max", 0)
        )
        source_constraint_region_tangent_j_nnz_sum = int(checkpoint.get("source_constraint_region_tangent_j_nnz_sum", 0))
        source_constraint_region_tangent_scale_sum = float(
            checkpoint.get("source_constraint_region_tangent_scale_sum", 0.0)
        )
        source_constraint_region_tangent_scale_max = float(
            checkpoint.get("source_constraint_region_tangent_scale_max", 0.0)
        )
        state = MechanicsState(
            model.X + assembly.expand_displacements(q),
            assembly.expand_displacements(v),
            assembly.expand_displacements(a),
            time=float(start_time),
        )
        if vtk_enabled and vtk_dir is not None:
            vtk_datasets = [
                (float(row.get("time", 0.0) or 0.0), vtk_dir / str(row.get("vtk_file", "")))
                for row in vtk_manifest_rows
                if row.get("vtk_file")
            ]
    if vtk_enabled and vtk_dir is not None and not bool(resume_source_checkpoint):
        vtk_dir.mkdir(parents=True, exist_ok=True)
        for stale in vtk_dir.glob(f"{vtk_stem}_*.vtk"):
            stale.unlink()
        for stale in (vtk_dir / f"{vtk_stem}.pvd", vtk_dir / f"{vtk_stem}_manifest.csv"):
            if stale.exists():
                stale.unlink()
        visual_state, initial_internal = _source_drive_corotated_visual_state_and_internal(
            model=model,
            state=state,
            reference_tangent=K_full,
            body_node_slices=body_node_slices,
            body_reference_points=body_reference_points,
            body_rotation_z=(0.0, 0.0),
            stress_postprocess=postprocess_label,
        )
        frame_path = vtk_dir / f"{vtk_stem}_{0:04d}.vtk"
        frame_row = _write_sfc_tet4_vtk_frame(
            frame_path,
            model=model,
            state=visual_state,
            internal=initial_internal,
            element_object_ids=element_object_ids,
            time_value=0.0,
            frame_index=0,
            include_tensors=vtk_include_tensors,
            static_blocks=vtk_static_blocks,
        )
        frame_row.update(
            {
                "stress_strain_postprocess": postprocess_label,
                "rotation_unit": "radian",
                "rp1_rotation_z_rad": 0.0,
                "rp2_rotation_z_rad": 0.0,
                "rp1_angular_velocity_z_rad_per_s": float(v[assembly.hub_slice(0).start + 5]),
                "rp2_angular_velocity_z_rad_per_s": float(v[hub2_slice.start + 5]),
                "rp1_angular_acceleration_z_rad_per_s2": float(a[assembly.hub_slice(0).start + 5]),
                "rp2_angular_acceleration_z_rad_per_s2": float(a[hub2_slice.start + 5]),
            }
        )
        vtk_manifest_rows.append(frame_row)
        vtk_datasets.append((0.0, frame_path))
    c0 = 1.0 / (beta * float(dt) * float(dt))
    if internal_kinematics == "corotated_rp":
        K_solve = (elastic_map.T @ K_full @ elastic_map).tocsr()
    else:
        K_solve = K_red
    base_matrix = (M_red * c0 + K_solve * scale).tocsr()
    base_matrix_dt = float(dt)
    base_free = None
    base_lu_cache: list[Any] = [None]
    start = time.perf_counter()
    base_preconditioner_lu: Any | None = None
    contact_aggregation_workspace = _ContactAggregationWorkspace()
    previous_path_master_face_ids: np.ndarray | None = None
    previous_path_master_barycentric: np.ndarray | None = None
    previous_active_region_ids: tuple[int, ...] | None = None

    def source_sample_arrays_raw(x_contact: np.ndarray) -> dict[str, np.ndarray] | None:
        if (
            len(contact_geometries) != 1
            or bool(source_contact_footprint_clipping)
        ):
            return None
        if contact_averaging == "slave_node_point":
            return None
        if contact_normal_filter not in {"none", "opposing"}:
            return None
        if contact_direction == "secondary_average" and contact_projection == "secondary_line":
            geometry = contact_geometries[0]
            if hasattr(geometry, "secondary_normal_projection_sample_arrays"):
                arrays = geometry.secondary_normal_projection_sample_arrays(
                    x_contact,
                    dot_threshold=secondary_dot_threshold,
                )
                return arrays
            return None
        if contact_normal_filter != "none":
            return None
        return contact_geometries[0].sample_arrays(x_contact)

    def aggregate_source_sample_arrays(raw_arrays: dict[str, np.ndarray] | None) -> dict[str, np.ndarray] | None:
        if raw_arrays is not None and contact_averaging != "none":
            return _aggregate_contact_sample_arrays(
                raw_arrays,
                contact_averaging,
                workspace=contact_aggregation_workspace,
            )
        return raw_arrays

    def source_sample_arrays(x_contact: np.ndarray) -> dict[str, np.ndarray] | None:
        return aggregate_source_sample_arrays(source_sample_arrays_raw(x_contact))

    def source_samples(x_contact: np.ndarray) -> list[Any]:
        collected: list[Any] = []
        use_compatible_query = contact_normal_filter == "opposing_search" and contact_averaging != "slave_node_point"
        for geometry in contact_geometries:
            if (
                contact_direction == "secondary_average"
                and contact_projection == "secondary_line"
                and hasattr(geometry, "secondary_normal_projection_sample_arrays")
            ):
                arrays = geometry.secondary_normal_projection_sample_arrays(
                    x_contact,
                    dot_threshold=secondary_dot_threshold,
                )
                if arrays is not None:
                    collected.extend(_contact_samples_from_arrays(arrays, stiffness=pressure_stiffness))
                elif hasattr(geometry, "secondary_normal_projection_samples"):
                    collected.extend(
                        list(
                            geometry.secondary_normal_projection_samples(
                                x_contact,
                                dot_threshold=secondary_dot_threshold,
                            )
                        )
                    )
            elif use_compatible_query:
                arrays = (
                    geometry.normal_compatible_sample_arrays(x_contact)
                    if hasattr(geometry, "normal_compatible_sample_arrays")
                    else None
                )
                if arrays is not None:
                    collected.extend(_contact_samples_from_arrays(arrays, stiffness=pressure_stiffness))
                else:
                    collected.extend(list(geometry.normal_compatible_samples(x_contact)))
            elif contact_averaging == "slave_node_point":
                collected.extend(list(geometry.nodal_samples(x_contact)))
            else:
                collected.extend(list(geometry.samples(x_contact)))
        if contact_normal_filter == "opposing" and not use_compatible_query and contact_direction != "secondary_average":
            collected = _filter_contact_samples_by_normal_compatibility(
                collected,
                x_contact,
                mode=contact_normal_filter,
            )
        if contact_direction == "secondary_average" and contact_projection != "secondary_line":
            collected = _replace_contact_normals_with_secondary_average(
                collected,
                x_contact,
                project_gap_to_secondary_plane=contact_projection == "secondary_plane",
            )
        if contact_averaging != "none":
            aggregate_mode = "slave_node" if contact_averaging == "slave_node_point" else contact_averaging
            collected = _aggregate_contact_samples(collected, aggregate_mode)
        return collected

    def source_sample_arrays_trial(x_contact: np.ndarray) -> dict[str, np.ndarray] | None:
        return _core_run_contact_tracking_trial(contact_geometries, lambda: source_sample_arrays(x_contact))

    def source_sample_arrays_trial_with_raw(
        x_contact: np.ndarray,
    ) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
        def query() -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
            raw_arrays = source_sample_arrays_raw(x_contact)
            return raw_arrays, aggregate_source_sample_arrays(raw_arrays)

        return _core_run_contact_tracking_trial(contact_geometries, query)

    def commit_accepted_tracking_from_raw(raw_arrays: dict[str, np.ndarray] | None) -> int:
        return _core_commit_accepted_contact_tracking_from_sample_arrays(contact_geometries, raw_arrays)

    def source_samples_trial(x_contact: np.ndarray) -> list[Any]:
        return _core_run_contact_tracking_trial(contact_geometries, lambda: source_samples(x_contact))

    accepted_step = int(start_step)
    accepted_time = float(start_time)
    trial_dt = float(dt)
    while accepted_time < float(duration) - 1.0e-15:
        step_dt = min(float(trial_dt), float(duration) - float(accepted_time))
        step = int(accepted_step + 1)
        trial_index = int(len(source_increment_trial_rows) + 1)
        t = float(accepted_time + step_dt)
        current_source_step_dt = float(step_dt)
        c0 = 1.0 / (beta * float(step_dt) * float(step_dt))
        if not np.isclose(base_matrix_dt, float(step_dt), rtol=1.0e-14, atol=1.0e-16):
            base_matrix = (M_red * c0 + K_solve * scale).tocsr()
            base_matrix_dt = float(step_dt)
            base_free = None
            base_preconditioner_lu = None
            base_lu_cache = [None]
        fixed, values = _source_drive_fixed_reduced_dofs(assembly, time_value=t, omega_z=gear1_angular_velocity_z)
        free = free_dofs(assembly.n_reduced_dofs, fixed)
        if base_free is None:
            base_free = base_matrix[free[:, None], free].tocsr() if free.size else None
            if base_free is not None and free.size:
                t_section = time.perf_counter()
                try:
                    base_preconditioner_lu = splu(base_free.tocsc(), permc_spec="COLAMD", diag_pivot_thresh=0.0)
                except Exception:
                    base_preconditioner_lu = None
                timing_source_base_lu += time.perf_counter() - t_section
        q_pred, v_pred, _ = calculix_dynamic_predictor(q, v, a, dt=float(step_dt), beta=beta, gamma=gamma)
        q_guess = _project_reduced_fixed(q_pred, fixed, values)
        residual_norm = np.inf
        iteration_count = 0
        last_contact = contact0
        last_raw_sample_arrays: dict[str, np.ndarray] | None = None
        last_sample_arrays: dict[str, np.ndarray] | None = None
        last_samples: list[Any] | None = None
        previous_active_signature: tuple[tuple[int, ...], ...] | None = None
        active_set_stable = not bool(source_contact_active_set_stability)
        step_converged = False
        step_convergence_reason = "iteration_limit"
        contact_force_increment_norm = np.inf
        normalized_residual = np.inf
        normalized_correction = np.inf
        normalized_contact_force_increment = np.inf
        residual_converged = False
        correction_converged = False
        contact_force_increment_converged = False
        step_constraint_region_tangent_solve_count = 0
        step_constraint_region_tangent_active_rows_max = 0
        step_constraint_region_tangent_j_nnz_sum = 0
        step_constraint_region_tangent_scale_max = 0.0
        previous_iteration_contact_red: np.ndarray | None = None
        step_line_search_trial_count = 0
        step_line_search_reduced_count = 0
        step_line_search_stable_count = 0
        step_line_search_unstable_count = 0
        step_line_search_last_alpha = 1.0
        for iteration in range(1, max(1, int(max_iterations)) + 1):
            q_guess = _project_reduced_fixed(q_guess, fixed, values)
            x_guess = model.X + assembly.expand_displacements(q_guess)
            x_contact = contact_positions_from_reduced(q_guess)
            t_section = time.perf_counter()
            raw_sample_arrays, sample_arrays = source_sample_arrays_trial_with_raw(x_contact)
            if sample_arrays is None:
                samples = source_samples_trial(x_contact)
                last_contact = _assemble_contact_response_force_only(samples, model.n_nodes)
                last_samples = samples
                last_raw_sample_arrays = None
                last_sample_arrays = None
                active_signature = _contact_active_signature_from_samples(samples)
            else:
                samples = []
                last_contact = _assemble_contact_arrays_force_only(sample_arrays, model.n_nodes, stiffness=pressure_stiffness)
                last_raw_sample_arrays = raw_sample_arrays
                last_sample_arrays = sample_arrays
                last_samples = None
                active_signature = _contact_active_signature_from_arrays(sample_arrays)
            active_set_stable = _source_contact_active_set_is_stable(
                active_signature,
                previous_active_signature,
                require_stability=bool(source_contact_active_set_stability),
            )
            previous_active_signature = active_signature
            assemble_current_internal_tangent = internal_kinematics == "finite_stvk_visual"
            internal_red, internal_tangent_red = internal_reduced_response_from_state(
                q_guess,
                assemble_tangent=assemble_current_internal_tangent,
            )
            contact_red = assembly.reduce_vector(last_contact.force)
            rhs_balance = external - internal_red + contact_red
            a_guess = c0 * (q_guess - q_pred)
            iteration_velocity = v_pred + gamma * float(step_dt) * a_guess
            centripetal_red, centripetal_tangent = centripetal_reduced_and_tangent_for_iteration(q_guess, iteration_velocity)
            inertia_red, inertia_mass = inertia_reduced_and_mass_for_iteration(q_guess, iteration_velocity, a_guess)
            residual = (
                inertia_red
                + scale * centripetal_red
                - scale * rhs_balance
                + float(hht_alpha) * previous
            )
            residual_norm = float(np.linalg.norm(residual[free])) if free.size else 0.0
            timing_residual += time.perf_counter() - t_section
            t_section = time.perf_counter()
            correction_free = np.empty(0, dtype=float)
            if free.size:
                cg_stats: dict[str, Any] = {}
                current_base_free = base_free
                current_base_lu = base_preconditioner_lu
                if rotating_inertia == "finite_kinematic" and free.size:
                    stiffness_for_iteration = internal_tangent_red if internal_tangent_red is not None else K_solve
                    current_base = (inertia_mass * c0 + stiffness_for_iteration * scale).tocsr()
                    current_base_free = current_base[free[:, None], free].tocsr()
                    current_base_lu = (
                        base_preconditioner_lu
                        if base_preconditioner_lu is not None
                        and base_free is not None
                        and current_base_free.shape == base_free.shape
                        else None
                    )
                elif internal_tangent_red is not None and free.size:
                    current_base = (M_red * c0 + internal_tangent_red * scale).tocsr()
                    current_base_free = current_base[free[:, None], free].tocsr()
                    current_base_lu = (
                        base_preconditioner_lu
                        if base_preconditioner_lu is not None
                        and base_free is not None
                        and current_base_free.shape == base_free.shape
                        else None
                    )
                if rotating_inertia != "none" and base_free is not None:
                    if rotating_inertia == "centripetal":
                        current_base_free = (base_free + scale * centripetal_tangent[free[:, None], free]).tocsr()
                if sample_arrays is None:
                    correction_free = _solve_reduced_penalty_sparse_cg_correction(
                        base_matrix_free=current_base_free,
                        residual_free=residual[free],
                        samples=samples,
                        transformation=T,
                        free=free,
                        n_total_dofs=model.n_dofs,
                        equilibrium_scale=scale,
                        tolerance=float(tolerance),
                        stats=cg_stats,
                        base_lu=current_base_lu,
                    )
                else:
                    correction_free = _solve_reduced_penalty_sparse_cg_correction_from_arrays(
                        base_matrix_free=current_base_free,
                        residual_free=residual[free],
                        sample_arrays=sample_arrays,
                        transformation=T,
                        free=free,
                        equilibrium_scale=scale,
                        pressure_stiffness=pressure_stiffness,
                        tolerance=float(tolerance),
                        stats=cg_stats,
                        base_lu=current_base_lu,
                    )
                if correction_free is None:
                    source_direct_fallback_count += 1
                    if sample_arrays is not None:
                        samples = source_samples_trial(x_contact)
                    correction_free = _solve_reduced_penalty_low_rank_correction(
                        base_matrix_free=current_base_free,
                        base_lu_cache=base_lu_cache if rotating_inertia == "none" else [None],
                        residual_free=residual[free],
                        samples=samples,
                        transformation=T,
                        free=free,
                        n_total_dofs=model.n_dofs,
                        equilibrium_scale=scale,
                    )
                else:
                    source_sparse_cg_count += 1
                    source_sparse_cg_iterations += int(cg_stats.get("iterations", 0))
                    if str(cg_stats.get("preconditioner", "")) == "base_lu":
                        source_sparse_cg_base_lu_preconditioner += 1
                    if str(cg_stats.get("contact_tangent_source", "")) == "constraint_region_arrays":
                        active_rows = int(cg_stats.get("contact_tangent_active_region_count", 0))
                        j_nnz = int(cg_stats.get("contact_tangent_j_nnz", 0))
                        scale_sum_value = float(cg_stats.get("contact_tangent_scale_sum", 0.0))
                        scale_max_value = float(cg_stats.get("contact_tangent_scale_max", 0.0))
                        source_constraint_region_tangent_solve_count += 1
                        source_constraint_region_tangent_active_rows_sum += active_rows
                        source_constraint_region_tangent_active_rows_max = max(
                            source_constraint_region_tangent_active_rows_max,
                            active_rows,
                        )
                        source_constraint_region_tangent_j_nnz_sum += j_nnz
                        source_constraint_region_tangent_scale_sum += scale_sum_value
                        source_constraint_region_tangent_scale_max = max(
                            source_constraint_region_tangent_scale_max,
                            scale_max_value,
                        )
                        step_constraint_region_tangent_solve_count += 1
                        step_constraint_region_tangent_active_rows_max = max(
                            step_constraint_region_tangent_active_rows_max,
                            active_rows,
                        )
                        step_constraint_region_tangent_j_nnz_sum += j_nnz
                        step_constraint_region_tangent_scale_max = max(
                            step_constraint_region_tangent_scale_max,
                            scale_max_value,
                        )
            timing_linear += time.perf_counter() - t_section
            iteration_count = iteration
            correction_norm = float(np.linalg.norm(correction_free))
            correction_scale = max(1.0, float(np.linalg.norm(q_guess[free])) if free.size else 1.0)
            balance_scale = max(
                1.0,
                float(np.linalg.norm(rhs_balance[free])) if free.size else 1.0,
                float(np.linalg.norm(inertia_red[free])) if free.size else 1.0,
            )
            normalized_residual = residual_norm / balance_scale
            normalized_correction = correction_norm / correction_scale
            if previous_iteration_contact_red is None:
                contact_force_increment_norm = np.inf
                normalized_contact_force_increment = np.inf
                contact_force_increment_converged = last_contact.active_count == 0
            else:
                contact_force_increment_norm = float(np.linalg.norm(contact_red - previous_iteration_contact_red))
                contact_force_scale = max(
                    1.0,
                    float(np.linalg.norm(contact_red)),
                    float(np.linalg.norm(previous_iteration_contact_red)),
                )
                normalized_contact_force_increment = contact_force_increment_norm / contact_force_scale
                contact_force_increment_converged = (
                    normalized_contact_force_increment <= source_contact_force_increment_tolerance
                )
            correction_converged = normalized_correction <= source_correction_tolerance
            residual_converged = normalized_residual <= source_residual_tolerance
            iteration_decision = _source_increment_convergence_decision(
                residual_converged=residual_converged,
                correction_converged=correction_converged,
                contact_force_increment_converged=contact_force_increment_converged,
                active_set_stable=active_set_stable,
                iteration_count=iteration,
                max_iterations=max_iterations,
                accept_unconverged=source_accept_unconverged,
            )
            if iteration_decision.converged:
                step_converged = True
                step_convergence_reason = iteration_decision.reason
                break
            step_convergence_reason = iteration_decision.reason
            previous_iteration_contact_red = contact_red.copy()
            if (
                bool(source_active_set_line_search)
                and bool(source_contact_active_set_stability)
                and correction_free.size
                and active_signature is not None
            ):
                line_search_attempted = True
                alphas: list[float] = []
                alpha_value = 1.0
                while alpha_value >= source_active_set_line_search_min_alpha * (1.0 - 1.0e-12):
                    alphas.append(float(alpha_value))
                    alpha_value *= 0.5
                if alphas[-1] > source_active_set_line_search_min_alpha * (1.0 + 1.0e-12):
                    alphas.append(float(source_active_set_line_search_min_alpha))
                candidates: list[tuple[float, tuple[tuple[int, ...], ...]]] = []
                for alpha_candidate in alphas:
                    q_trial = q_guess.copy()
                    q_trial[free] += float(alpha_candidate) * correction_free
                    q_trial = _project_reduced_fixed(q_trial, fixed, values)
                    x_trial_contact = contact_positions_from_reduced(q_trial)
                    trial_arrays = source_sample_arrays_trial(x_trial_contact)
                    if trial_arrays is None:
                        trial_signature = _contact_active_signature_from_samples(source_samples_trial(x_trial_contact))
                    else:
                        trial_signature = _contact_active_signature_from_arrays(trial_arrays)
                    candidates.append((float(alpha_candidate), trial_signature))
                alpha_choice, stable_choice, line_search_trials = _source_active_set_line_search_choice(
                    active_signature,
                    candidates,
                    require_stability=True,
                )
                step_line_search_trial_count += int(line_search_trials)
                source_line_search_trial_count += int(line_search_trials)
                if stable_choice:
                    step_line_search_stable_count += 1
                    source_line_search_stable_count += 1
                else:
                    step_line_search_unstable_count += 1
                    source_line_search_unstable_count += 1
                if alpha_choice < 1.0 - 1.0e-12:
                    correction_free = float(alpha_choice) * correction_free
                    step_line_search_reduced_count += 1
                    source_line_search_reduced_count += 1
                step_line_search_last_alpha = float(alpha_choice)
                active_set_stable = _source_active_set_stability_after_line_search(
                    active_set_stable,
                    line_search_attempted=line_search_attempted,
                    line_search_stable=stable_choice,
                )
            q_guess[free] += correction_free
        increment_decision = _source_increment_convergence_decision(
            residual_converged=residual_converged,
            correction_converged=correction_converged,
            contact_force_increment_converged=contact_force_increment_converged,
            active_set_stable=active_set_stable,
            iteration_count=iteration_count,
            max_iterations=max_iterations,
            accept_unconverged=source_accept_unconverged,
        )
        step_convergence_reason = increment_decision.reason
        iteration_limited = increment_decision.iteration_limited
        cutback_candidate_dt = (
            _source_increment_cutback_candidate_dt(
                float(step_dt),
                min_dt=source_min_cutback_dt_value,
                cutback_factor=source_cutback_factor,
            )
            if increment_decision.cutback_required
            else None
        )
        increment_gate_row = _source_increment_gate_row(
            residual_converged=residual_converged,
            correction_converged=correction_converged,
            contact_force_increment_converged=contact_force_increment_converged,
            active_set_stable=active_set_stable,
            normalized_residual=normalized_residual,
            normalized_correction=normalized_correction,
            normalized_contact_force_increment=normalized_contact_force_increment,
            contact_force_increment_norm=contact_force_increment_norm,
            decision=increment_decision,
            cutback_candidate_dt=cutback_candidate_dt,
        )
        source_increment_trial_rows.append(
            {
                "source_trial_index": trial_index,
                "source_trial_candidate_step": int(step),
                "source_trial_start_time": float(accepted_time),
                "source_trial_end_time": float(t),
                "source_trial_dt": float(step_dt),
                "source_trial_accepted": int(bool(increment_decision.accepted and not increment_decision.cutback_required)),
                "source_trial_retry_required": int(bool(increment_decision.cutback_required and cutback_candidate_dt is not None)),
                **increment_gate_row,
            }
        )
        if increment_decision.cutback_required:
            source_iteration_limit_reached_count += int(bool(increment_decision.iteration_limited))
            source_cutback_required_count += 1
            if cutback_candidate_dt is None:
                raise RuntimeError(
                    "source-drive increment failed Abaqus-style convergence gates "
                    f"at step {step}: reason={step_convergence_reason}, "
                    f"normalized_residual={normalized_residual:.6e}, "
                    f"normalized_correction={normalized_correction:.6e}, "
                    f"normalized_contact_force_increment={normalized_contact_force_increment:.6e}, "
                    "minimum cutback dt reached"
                )
            trial_dt = float(cutback_candidate_dt)
            continue
        q_new = _project_reduced_fixed(q_guess, fixed, values)
        a_new = c0 * (q_new - q_pred)
        v_new = v_pred + gamma * float(step_dt) * a_new
        x_new = model.X + assembly.expand_displacements(q_new)
        state = MechanicsState(x_new, assembly.expand_displacements(v_new), assembly.expand_displacements(a_new), time=t)
        # Commit path-tracking caches only after the increment is accepted.
        accepted_contact_x = contact_positions_from_reduced(q_new)
        accepted_reused_response = _core_accepted_contact_response_can_reuse(
            converged=bool(increment_decision.converged),
            sample_arrays_present=last_sample_arrays is not None,
        )
        if accepted_reused_response:
            accepted_arrays = last_sample_arrays
            source_accepted_contact_response_reuse_count += 1
            accepted_tracking_committed = commit_accepted_tracking_from_raw(last_raw_sample_arrays)
            source_accepted_tracking_commit_count += accepted_tracking_committed
        else:
            accepted_arrays = source_sample_arrays(accepted_contact_x)
            accepted_tracking_committed = 0
            source_accepted_contact_response_requery_count += 1
        if accepted_arrays is None:
            accepted_samples = source_samples(accepted_contact_x)
            last_contact = _assemble_contact_response_force_only(accepted_samples, model.n_nodes)
            last_samples = accepted_samples
            last_sample_arrays = None
            active_set_stable = _source_contact_active_set_is_stable(
                _contact_active_signature_from_samples(accepted_samples),
                previous_active_signature,
                require_stability=bool(source_contact_active_set_stability),
            )
        else:
            if not accepted_reused_response:
                last_contact = _assemble_contact_arrays_force_only(accepted_arrays, model.n_nodes, stiffness=pressure_stiffness)
            last_sample_arrays = accepted_arrays
            last_samples = None
            active_set_stable = _source_contact_active_set_is_stable(
                _contact_active_signature_from_arrays(accepted_arrays),
                previous_active_signature,
                require_stability=bool(source_contact_active_set_stability),
            )
        source_step_converged_count += int(bool(step_converged))
        source_iteration_limit_reached_count += int(bool(increment_decision.iteration_limited))
        source_cutback_required_count += int(bool(increment_decision.cutback_required))
        source_unconverged_accepted_count += int((not bool(increment_decision.converged)) and bool(increment_decision.accepted))
        source_unstable_accepted_count += int(bool(source_contact_active_set_stability) and not bool(active_set_stable))
        path_tracking_metrics, current_path_master_face_ids, current_path_master_barycentric = _contact_path_tracking_metrics_from_arrays(
            last_sample_arrays,
            previous_path_master_face_ids,
            previous_path_master_barycentric,
        )
        if current_path_master_face_ids is not None:
            previous_path_master_face_ids = current_path_master_face_ids
        if current_path_master_barycentric is not None:
            previous_path_master_barycentric = current_path_master_barycentric
        active_region_continuity_metrics, current_active_region_ids = _contact_active_region_continuity_metrics_from_arrays(
            last_sample_arrays,
            previous_active_region_ids,
        )
        if current_active_region_ids is not None:
            previous_active_region_ids = current_active_region_ids
        contact_region_metrics = (
            _contact_region_integral_metrics_from_arrays(last_sample_arrays, stiffness=pressure_stiffness)
            if last_sample_arrays is not None
            else _contact_region_integral_metrics_from_samples(last_samples)
        )
        contact_law_metrics = _constraint_region_contact_law_metrics_from_arrays(
            last_sample_arrays,
            last_raw_sample_arrays,
            averaging_mode=contact_averaging,
        )
        contact_tangent_metrics = _constraint_region_tangent_metrics_from_arrays(
            last_sample_arrays,
            pressure_stiffness=pressure_stiffness,
            equilibrium_scale=scale,
        )
        previous = external - internal_reduced_from_state(q_new) + assembly.reduce_vector(last_contact.force)
        if rotating_inertia == "centripetal":
            previous = previous - centripetal_reduced_from_state(q_new, v_new)
        accepted_step = int(accepted_step + 1)
        accepted_time = float(t)
        is_final_step = accepted_time >= float(duration) - 1.0e-15
        history_due = (accepted_step % history_stride == 0) or is_final_step
        vtk_due = vtk_enabled and vtk_dir is not None and (accepted_step % int(vtk_frame_stride) == 0 or is_final_step)
        output_state: MechanicsState | None = None
        output_internal: InternalResponse | None = None
        contact_node_diagnostics: dict[str, Any] | None = None
        if history_due or vtk_due:
            contact_node_diagnostics = (
                _contact_node_diagnostics_from_arrays(last_sample_arrays, model.n_nodes, stiffness=pressure_stiffness)
                if last_sample_arrays is not None
                else _contact_node_diagnostics_from_samples(last_samples, model.n_nodes)
            )
        if history_due:
            t_section = time.perf_counter()
            output_state, output_internal = _source_drive_corotated_visual_state_and_internal(
                model=model,
                state=state,
                reference_tangent=K_full,
                body_node_slices=body_node_slices,
                body_reference_points=body_reference_points,
                body_rotation_z=(
                    float(q_new[assembly.hub_slice(0).start + 5]),
                    float(q_new[hub2_slice.start + 5]),
                ),
                stress_postprocess=postprocess_label,
            )
            row = _penalty_history_row(
                time_value=t,
                closure=0.0,
                rotation=float(gear1_angular_velocity_z) * t,
                state=output_state,
                model=model,
                internal=output_internal,
                contact_response=last_contact,
                young=young,
                poisson=poisson,
                newton_iterations=iteration_count,
                residual_norm=residual_norm,
                solver="source_reduced_rp_modified_newton",
            )
            row.update(
                {
                    "rp1_rotation_z": float(q_new[assembly.hub_slice(0).start + 5]),
                    "rp1_angular_velocity_z": float(v_new[assembly.hub_slice(0).start + 5]),
                    "rp1_angular_acceleration_z": float(a_new[assembly.hub_slice(0).start + 5]),
                    "rp2_rotation_z": float(q_new[hub2_slice.start + 5]),
                    "rp2_angular_velocity_z": float(v_new[hub2_slice.start + 5]),
                    "rp2_angular_acceleration_z": float(a_new[hub2_slice.start + 5]),
                    "gear2_torque_z": float(gear2_torque_z),
                    "source_step_dt": float(step_dt),
                    "source_accepted_step": int(accepted_step),
                    "source_line_search_trial_count": int(step_line_search_trial_count),
                    "source_line_search_reduced_count": int(step_line_search_reduced_count),
                    "source_line_search_stable_count": int(step_line_search_stable_count),
                    "source_line_search_unstable_count": int(step_line_search_unstable_count),
                    "source_line_search_last_alpha": float(step_line_search_last_alpha),
                    "source_accepted_contact_response_reused": int(bool(accepted_reused_response)),
                    "source_accepted_tracking_committed": int(accepted_tracking_committed),
                    "source_constraint_region_tangent_solve_count": int(step_constraint_region_tangent_solve_count),
                    "source_constraint_region_tangent_active_rows_max": int(
                        step_constraint_region_tangent_active_rows_max
                    ),
                    "source_constraint_region_tangent_j_nnz_sum": int(step_constraint_region_tangent_j_nnz_sum),
                    "source_constraint_region_tangent_scale_max": float(step_constraint_region_tangent_scale_max),
                    "contact_active_set_stable": int(bool(active_set_stable)),
                    "source_step_converged": int(bool(step_converged)),
                }
            )
            row.update(increment_gate_row)
            if contact_node_diagnostics is not None:
                row.update(contact_node_diagnostics.get("metrics", {}))
            row.update(contact_region_metrics)
            row.update(contact_law_metrics)
            row.update(contact_tangent_metrics)
            row.update(path_tracking_metrics)
            row.update(active_region_continuity_metrics)
            rows.append(row)
            timing_history += time.perf_counter() - t_section
        if vtk_due:
            t_section = time.perf_counter()
            frame_path = vtk_dir / f"{vtk_stem}_{vtk_frame_index:04d}.vtk"
            if output_state is None or output_internal is None:
                output_state, output_internal = _source_drive_corotated_visual_state_and_internal(
                    model=model,
                    state=state,
                    reference_tangent=K_full,
                    body_node_slices=body_node_slices,
                    body_reference_points=body_reference_points,
                    body_rotation_z=(
                        float(q_new[assembly.hub_slice(0).start + 5]),
                        float(q_new[hub2_slice.start + 5]),
                    ),
                    stress_postprocess=postprocess_label,
                )
            frame_row = _write_sfc_tet4_vtk_frame(
                frame_path,
                model=model,
                state=output_state,
                internal=output_internal,
                element_object_ids=element_object_ids,
                time_value=t,
                frame_index=vtk_frame_index,
                include_tensors=vtk_include_tensors,
                static_blocks=vtk_static_blocks,
                contact_node_diagnostics=contact_node_diagnostics,
            )
            frame_row.update(
                {
                    "stress_strain_postprocess": postprocess_label,
                    "rotation_unit": "radian",
                    "rp1_rotation_z_rad": float(q_new[assembly.hub_slice(0).start + 5]),
                    "rp2_rotation_z_rad": float(q_new[hub2_slice.start + 5]),
                    "rp1_angular_velocity_z_rad_per_s": float(v_new[assembly.hub_slice(0).start + 5]),
                    "rp2_angular_velocity_z_rad_per_s": float(v_new[hub2_slice.start + 5]),
                    "rp1_angular_acceleration_z_rad_per_s2": float(a_new[assembly.hub_slice(0).start + 5]),
                    "rp2_angular_acceleration_z_rad_per_s2": float(a_new[hub2_slice.start + 5]),
                    "source_step_dt": float(step_dt),
                    "source_accepted_step": int(accepted_step),
                    "source_line_search_trial_count": int(step_line_search_trial_count),
                    "source_line_search_reduced_count": int(step_line_search_reduced_count),
                    "source_line_search_stable_count": int(step_line_search_stable_count),
                    "source_line_search_unstable_count": int(step_line_search_unstable_count),
                    "source_line_search_last_alpha": float(step_line_search_last_alpha),
                    "source_accepted_contact_response_reused": int(bool(accepted_reused_response)),
                    "source_accepted_tracking_committed": int(accepted_tracking_committed),
                    "source_constraint_region_tangent_solve_count": int(step_constraint_region_tangent_solve_count),
                    "source_constraint_region_tangent_active_rows_max": int(
                        step_constraint_region_tangent_active_rows_max
                    ),
                    "source_constraint_region_tangent_j_nnz_sum": int(step_constraint_region_tangent_j_nnz_sum),
                    "source_constraint_region_tangent_scale_max": float(step_constraint_region_tangent_scale_max),
                }
            )
            frame_row.update(increment_gate_row)
            frame_row.update(path_tracking_metrics)
            frame_row.update(active_region_continuity_metrics)
            frame_row.update(contact_region_metrics)
            frame_row.update(contact_law_metrics)
            frame_row.update(contact_tangent_metrics)
            vtk_manifest_rows.append(frame_row)
            vtk_datasets.append((float(t), frame_path))
            vtk_frame_index += 1
            timing_vtk += time.perf_counter() - t_section
        q, v, a = q_new, v_new, a_new
        trial_dt = min(float(dt), max(source_min_cutback_dt_value, float(step_dt)) * (1.25 if increment_decision.converged else 1.0))
        if checkpoint_path is not None and (accepted_step % max(1, int(source_checkpoint_stride)) == 0 or is_final_step):
            _write_source_drive_checkpoint(
                checkpoint_path,
                step=accepted_step,
                time_value=accepted_time,
                q=q,
                v=v,
                a=a,
                previous=previous,
                rows=rows,
                vtk_manifest_rows=vtk_manifest_rows,
                vtk_frame_index=vtk_frame_index,
                timing_residual=timing_residual,
                timing_linear=timing_linear,
                timing_history=timing_history,
                timing_vtk=timing_vtk,
                timing_source_base_lu=timing_source_base_lu,
                source_sparse_cg_count=source_sparse_cg_count,
                source_sparse_cg_iterations=source_sparse_cg_iterations,
                source_sparse_cg_base_lu_preconditioner=source_sparse_cg_base_lu_preconditioner,
                source_direct_fallback_count=source_direct_fallback_count,
                source_step_converged_count=source_step_converged_count,
                source_iteration_limit_reached_count=source_iteration_limit_reached_count,
                source_unstable_accepted_count=source_unstable_accepted_count,
                source_cutback_required_count=source_cutback_required_count,
                source_unconverged_accepted_count=source_unconverged_accepted_count,
                source_line_search_trial_count=source_line_search_trial_count,
                source_line_search_reduced_count=source_line_search_reduced_count,
                source_line_search_stable_count=source_line_search_stable_count,
                source_line_search_unstable_count=source_line_search_unstable_count,
                source_accepted_contact_response_reuse_count=source_accepted_contact_response_reuse_count,
                source_accepted_contact_response_requery_count=source_accepted_contact_response_requery_count,
                source_accepted_tracking_commit_count=source_accepted_tracking_commit_count,
                source_constraint_region_tangent_solve_count=source_constraint_region_tangent_solve_count,
                source_constraint_region_tangent_active_rows_sum=source_constraint_region_tangent_active_rows_sum,
                source_constraint_region_tangent_active_rows_max=source_constraint_region_tangent_active_rows_max,
                source_constraint_region_tangent_j_nnz_sum=source_constraint_region_tangent_j_nnz_sum,
                source_constraint_region_tangent_scale_sum=source_constraint_region_tangent_scale_sum,
                source_constraint_region_tangent_scale_max=source_constraint_region_tangent_scale_max,
            )
    wall = time.perf_counter() - start
    summary = {
        "sfc_wall_seconds": float(wall),
        "nodes": int(X.shape[0]),
        "elements": int(elements.shape[0]),
        "gear1_contact_faces": int(pair.gear1.contact_faces.shape[0]),
        "gear2_contact_faces": int(pair.gear2.contact_faces.shape[0]),
        "gear1_support_nodes": int(pair.gear1.support_nodes.size),
        "gear2_support_nodes": int(pair.gear2.support_nodes.size),
        "initial_patch_gap": float(pair.initial_patch_gap),
        "final_active_contact_samples": int(rows[-1]["active_contact_samples"]) if rows else 0,
        "final_min_gap": float(rows[-1]["min_gap"]) if rows else 0.0,
        "final_normal_force": float(rows[-1]["normal_force"]) if rows else 0.0,
        "hht_alpha": float(hht_alpha),
        "contact_mode": "source_penalty",
        "penalty_solver": "source_reduced_rp_modified_newton",
        "material_linearization": "finite_stvk_visual" if internal_kinematics == "finite_stvk_visual" else "reference_linear",
        "tet4_mass_kind": str(tet4_mass_kind),
        "source_gear1_angular_velocity_z_rad_per_s": float(gear1_angular_velocity_z),
        "source_gear2_torque_z": float(gear2_torque_z),
        "source_rotation_unit": "radian",
        "source_stress_strain_postprocess": postprocess_label,
        "source_contact_averaging": contact_averaging,
        "source_contact_kinematics": contact_kinematics,
        "source_contact_normal_filter": contact_normal_filter,
        "source_contact_direction": contact_direction,
        "source_contact_projection": contact_projection,
        "source_contact_pair_order": contact_pair_order,
        "source_contact_search_radius": float(contact_search_radius),
        "source_secondary_normal_min_projection": float(secondary_normal_min_projection),
        "source_secondary_line_hard_distance_limit": (
            "" if secondary_line_hard_distance_limit is None else float(secondary_line_hard_distance_limit)
        ),
        "source_secondary_path_tracking": bool(source_secondary_path_tracking),
        "source_contact_active_set_stability": bool(source_contact_active_set_stability),
        "source_contact_footprint_clipping": bool(source_contact_footprint_clipping),
        "source_internal_kinematics": internal_kinematics,
        "source_rotating_inertia": rotating_inertia,
        "source_initial_acceleration": "abaqus_zero_dynamic_step",
        "source_initial_hht_history": "zero_previous_step_balance",
        "source_max_iterations": int(max(1, int(max_iterations))),
        "source_residual_tolerance": float(source_residual_tolerance),
        "source_correction_tolerance": float(source_correction_tolerance),
        "source_contact_force_increment_tolerance": float(source_contact_force_increment_tolerance),
        "source_accept_unconverged": bool(source_accept_unconverged),
        "source_active_set_line_search": bool(source_active_set_line_search),
        "source_active_set_line_search_min_alpha": float(source_active_set_line_search_min_alpha),
        "source_cutback_factor": float(source_cutback_factor),
        "source_min_cutback_dt": float(source_min_cutback_dt_value),
        "source_step_converged_count": int(source_step_converged_count),
        "source_iteration_limit_reached_count": int(source_iteration_limit_reached_count),
        "source_unstable_accepted_count": int(source_unstable_accepted_count),
        "source_cutback_required_count": int(source_cutback_required_count),
        "source_unconverged_accepted_count": int(source_unconverged_accepted_count),
        "source_accepted_increment_count": int(accepted_step),
        "source_increment_trial_count": int(len(source_increment_trial_rows)),
        "source_rejected_trial_count": int(
            sum(1 for row in source_increment_trial_rows if int(row.get("source_trial_accepted", 0)) == 0)
        ),
        "source_final_time": float(accepted_time),
        "source_line_search_trial_count": int(source_line_search_trial_count),
        "source_line_search_reduced_count": int(source_line_search_reduced_count),
        "source_line_search_stable_count": int(source_line_search_stable_count),
        "source_line_search_unstable_count": int(source_line_search_unstable_count),
        "source_accepted_contact_response_reuse_count": int(source_accepted_contact_response_reuse_count),
        "source_accepted_contact_response_requery_count": int(source_accepted_contact_response_requery_count),
        "source_accepted_tracking_commit_count": int(source_accepted_tracking_commit_count),
        "source_constraint_region_tangent_solve_count": int(source_constraint_region_tangent_solve_count),
        "source_constraint_region_tangent_active_rows_sum": int(source_constraint_region_tangent_active_rows_sum),
        "source_constraint_region_tangent_active_rows_max": int(source_constraint_region_tangent_active_rows_max),
        "source_constraint_region_tangent_j_nnz_sum": int(source_constraint_region_tangent_j_nnz_sum),
        "source_constraint_region_tangent_scale_sum": float(source_constraint_region_tangent_scale_sum),
        "source_constraint_region_tangent_scale_max": float(source_constraint_region_tangent_scale_max),
        "sfc_history_frame_stride": int(history_stride),
        "sfc_history_row_count": int(len(rows)),
        "reduced_dofs": int(assembly.n_reduced_dofs),
        "timing_source_residual_seconds": float(timing_residual),
        "timing_source_linear_seconds": float(timing_linear),
        "timing_source_base_lu_seconds": float(timing_source_base_lu),
        "timing_source_history_seconds": float(timing_history),
        "timing_source_vtk_seconds": float(timing_vtk),
        "source_sparse_cg_count": int(source_sparse_cg_count),
        "source_sparse_cg_iterations": int(source_sparse_cg_iterations),
        "source_sparse_cg_base_lu_preconditioner": int(source_sparse_cg_base_lu_preconditioner),
        "source_direct_fallback_count": int(source_direct_fallback_count),
        "source_contact_aggregation_workspace_hits": int(contact_aggregation_workspace.hits),
        "source_contact_aggregation_workspace_misses": int(contact_aggregation_workspace.misses),
        "source_resume_from_step": int(start_step),
        "source_checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
        "source_checkpoint_stride": int(max(1, int(source_checkpoint_stride))),
        "status": "completed",
        "_source_increment_trial_rows": source_increment_trial_rows,
    }
    if vtk_enabled and vtk_dir is not None:
        pvd_path = vtk_dir / f"{vtk_stem}.pvd"
        manifest_path = vtk_dir / f"{vtk_stem}_manifest.csv"
        write_pvd(pvd_path, vtk_datasets)
        _write_csv(manifest_path, vtk_manifest_rows)
        summary.update(
            {
                "sfc_vtk_pvd": str(pvd_path),
                "sfc_vtk_manifest": str(manifest_path),
                "sfc_vtk_frame_count": int(len(vtk_datasets)),
                "sfc_vtk_frame_stride": int(vtk_frame_stride),
                "sfc_vtk_include_tensors": bool(vtk_include_tensors),
            }
        )
    return rows, summary


def solve_sfc_cropped_pair(
    pair: CroppedGearPair,
    *,
    young: float,
    poisson: float,
    density: float,
    pressure_stiffness: float,
    duration: float,
    dt: float,
    target_overclosure: float,
    rotation_rate_z: float,
    hht_alpha: float = -0.05,
    penalty_solver: str = "full_newton",
    material_linearization: str = "stvk",
    vtk_out_dir: Path | None = None,
    vtk_frame_stride: int = 1,
    vtk_stem: str = "sfc",
    vtk_include_tensors: bool = True,
) -> tuple[list[Row], Row]:
    solver = str(penalty_solver).lower()
    if solver not in {"full_newton", "modified_newton"}:
        raise ValueError("penalty_solver must be 'full_newton' or 'modified_newton'")
    if solver == "modified_newton":
        return _solve_sfc_cropped_pair_penalty_modified_newton(
            pair,
            young=young,
            poisson=poisson,
            density=density,
            pressure_stiffness=pressure_stiffness,
            duration=duration,
            dt=dt,
            target_overclosure=target_overclosure,
            rotation_rate_z=rotation_rate_z,
            hht_alpha=hht_alpha,
            max_iterations=12,
            tolerance=1.0e-9,
            material_linearization=material_linearization,
            vtk_out_dir=vtk_out_dir,
            vtk_frame_stride=vtk_frame_stride,
            vtk_stem=vtk_stem,
            vtk_include_tensors=vtk_include_tensors,
        )
    n1 = pair.gear1.nodes.shape[0]
    X = np.vstack([pair.gear1.nodes, pair.gear2.nodes])
    elements = np.vstack([pair.gear1.elements, pair.gear2.elements + n1])
    model = MechanicsModel.from_tet4_mesh(X, elements, E=young, nu=poisson, density=density)
    master = MaterialSDF.from_triangle_surface(pair.gear2.nodes, pair.gear2.contact_faces)
    contact = LagrangianSDFSurfaceContactGeometry(
        pair.gear1.contact_faces,
        master,
        pair.gear2.nodes,
        pressure_stiffness=pressure_stiffness,
        slave_node_offset=0,
        master_node_offset=n1,
        quadrature="tri3",
        search_radius=_default_contact_search_radius(pair, target_overclosure=target_overclosure),
    )
    fixed0, values0 = _fixed_conditions_for_pair(pair, closure=0.0, rotation_z=0.0)
    state, previous = initial_state_dirichlet(model, contact, gravity=0.0, fixed_dofs=fixed0, fixed_values=values0)
    rows: list[Row] = []
    steps = max(1, int(round(float(duration) / float(dt))))
    start = time.perf_counter()
    for step in range(1, steps + 1):
        t = step * float(dt)
        ramp = t / max(float(duration), float(dt))
        closure = ramp * (pair.initial_patch_gap + float(target_overclosure))
        rotation = float(rotation_rate_z) * t
        fixed, values = _fixed_conditions_for_pair(pair, closure=closure, rotation_z=rotation)
        state, previous, diagnostics = hht_step_dirichlet(
            model,
            state,
            previous,
            contact,
            dt=float(dt),
            gravity=0.0,
            fixed_dofs=fixed,
            fixed_values=values,
            alpha=float(hht_alpha),
            max_iterations=10,
            tolerance=1.0e-9,
            acceptance_policy="relative_correction",
        )
        internal = diagnostics.internal
        disp = state.x - model.X
        strain_norm = np.linalg.norm(internal.strain, axis=(1, 2)) if internal.strain.size else np.empty(0, dtype=float)
        elastic_strain_norm = _elastic_strain_norm_from_stress(internal.stress, young=young, poisson=poisson)
        equivalent_elastic_strain = _equivalent_elastic_strain_from_mises(internal.von_mises, young=young, poisson=poisson)
        rows.append(
            {
                "time": t,
                "closure": closure,
                "rotation_z": rotation,
                "active_contact_samples": diagnostics.contact.active_count,
                "min_gap": diagnostics.contact.min_gap,
                "normal_force": diagnostics.contact.normal_force,
                "max_displacement_norm": float(np.max(np.linalg.norm(disp, axis=1))),
                "p95_von_mises": float(np.percentile(internal.von_mises, 95.0)) if internal.von_mises.size else 0.0,
                "max_von_mises": float(np.max(internal.von_mises)) if internal.von_mises.size else 0.0,
                "p95_strain_norm": float(np.percentile(strain_norm, 95.0)) if strain_norm.size else 0.0,
                "max_strain_norm": float(np.max(strain_norm)) if strain_norm.size else 0.0,
                "p95_elastic_strain_norm": float(np.percentile(elastic_strain_norm, 95.0)) if elastic_strain_norm.size else 0.0,
                "max_elastic_strain_norm": float(np.max(elastic_strain_norm)) if elastic_strain_norm.size else 0.0,
                "p95_equivalent_elastic_strain": float(np.percentile(equivalent_elastic_strain, 95.0)) if equivalent_elastic_strain.size else 0.0,
                "max_equivalent_elastic_strain": float(np.max(equivalent_elastic_strain)) if equivalent_elastic_strain.size else 0.0,
                "strain_energy": internal.strain_energy,
                "newton_iterations": diagnostics.newton_iterations,
            }
        )
    wall = time.perf_counter() - start
    summary = {
        "sfc_wall_seconds": wall,
        "nodes": int(X.shape[0]),
        "elements": int(elements.shape[0]),
        "gear1_contact_faces": int(pair.gear1.contact_faces.shape[0]),
        "gear2_contact_faces": int(pair.gear2.contact_faces.shape[0]),
        "gear1_support_nodes": int(pair.gear1.support_nodes.size),
        "gear2_support_nodes": int(pair.gear2.support_nodes.size),
        "initial_patch_gap": float(pair.initial_patch_gap),
        "target_overclosure": float(target_overclosure),
        "final_active_contact_samples": int(rows[-1]["active_contact_samples"]) if rows else 0,
        "final_min_gap": float(rows[-1]["min_gap"]) if rows else 0.0,
        "final_normal_force": float(rows[-1]["normal_force"]) if rows else 0.0,
        "hht_alpha": float(hht_alpha),
        "contact_mode": "penalty",
        "status": "completed",
    }
    return rows, summary


def _cropped_pair_model_and_contact(
    pair: CroppedGearPair,
    *,
    young: float,
    poisson: float,
    density: float,
    pressure_stiffness: float,
    target_overclosure: float,
) -> tuple[MechanicsModel, LagrangianSDFSurfaceContactGeometry]:
    n1 = pair.gear1.nodes.shape[0]
    X = np.vstack([pair.gear1.nodes, pair.gear2.nodes])
    elements = np.vstack([pair.gear1.elements, pair.gear2.elements + n1])
    model = MechanicsModel.from_tet4_mesh(X, elements, E=young, nu=poisson, density=density)
    master = MaterialSDF.from_triangle_surface(pair.gear2.nodes, pair.gear2.contact_faces)
    contact = LagrangianSDFSurfaceContactGeometry(
        pair.gear1.contact_faces,
        master,
        pair.gear2.nodes,
        pressure_stiffness=pressure_stiffness,
        slave_node_offset=0,
        master_node_offset=n1,
        quadrature="tri3",
        search_radius=_default_contact_search_radius(pair, target_overclosure=target_overclosure),
        compiled_batch_projection=True,
        secondary_path_tracking=True,
    )
    return model, contact


def _cropped_pair_path_tracking_contact(
    pair: CroppedGearPair,
    *,
    pressure_stiffness: float,
    target_overclosure: float,
) -> LagrangianSDFSurfaceContactGeometry:
    """Return the diagnostic secondary-normal contact path for a cropped pair.

    The hard-contact solve can still use its existing closest-feature
    linearization.  This diagnostic path samples accepted states with
    Abaqus-style secondary-normal regions so cropped-patch validation can catch
    master-face jumps before escalating to the full gear.
    """

    n1 = pair.gear1.nodes.shape[0]
    master = MaterialSDF.from_triangle_surface(pair.gear2.nodes, pair.gear2.contact_faces)
    return LagrangianSDFSurfaceContactGeometry(
        pair.gear1.contact_faces,
        master,
        pair.gear2.nodes,
        pressure_stiffness=float(pressure_stiffness),
        slave_node_offset=0,
        master_node_offset=n1,
        quadrature="tri3",
        search_radius=_default_contact_search_radius(pair, target_overclosure=target_overclosure),
        compiled_batch_projection=True,
        secondary_path_tracking=True,
    )


def _hard_contact_linearized_gap_jacobian(
    contact: LagrangianSDFSurfaceContactGeometry,
    x_linearization: np.ndarray,
    u_linearization: np.ndarray,
    *,
    n_total_dofs: int,
    constraint_averaging: str = "slave_node_region_constraint",
) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray]:
    averaging = str(constraint_averaging).lower()
    base_mode, _overclosure_mode = _contact_averaging_modes(averaging)
    n_nodes = int(n_total_dofs) // 3
    if 3 * n_nodes != int(n_total_dofs):
        raise ValueError("n_total_dofs must be divisible by 3")
    if base_mode in {"slave_node_region", "slave_node"}:
        arrays = contact.sample_arrays(x_linearization)
        if arrays is not None:
            aggregated_arrays = _aggregate_contact_sample_arrays(arrays, averaging)
            if aggregated_arrays is not None:
                row_ids, sparse_jacobian = _constraint_region_gap_jacobian_sparse_from_arrays(
                    aggregated_arrays,
                    n_nodes=n_nodes,
                )
                gap_at_linearization = np.asarray(aggregated_arrays["gaps"], dtype=float).reshape(-1)[row_ids]
                areas = np.asarray(aggregated_arrays["areas"], dtype=float).reshape(-1)[row_ids]
                gap_jacobian = np.asarray(sparse_jacobian.toarray(), dtype=float)
                gap_offset = gap_at_linearization - gap_jacobian @ np.asarray(u_linearization, dtype=float).reshape(-1)
                samples = _contact_samples_from_arrays(aggregated_arrays, stiffness=float(contact.pressure_stiffness))
                return samples, gap_offset, gap_jacobian, areas
    samples = list(contact.samples(x_linearization))
    if base_mode in {"slave_node_region", "slave_node"}:
        samples = _aggregate_contact_samples(samples, averaging)
    gap_at_linearization, gap_jacobian = hard_contact_gap_jacobian_from_samples(samples, n_total_dofs=int(n_total_dofs))
    areas = np.asarray([float(sample.area) for sample in samples], dtype=float)
    if base_mode not in {"none", "slave_face", "surface_patch", "slave_node_region", "slave_node"}:
        raise ValueError(
            "constraint_averaging must be 'none', 'slave_face', 'slave_node', 'slave_node_region', "
            "'surface_patch', or a corresponding *_constraint mode"
        )
    if base_mode == "surface_patch" and gap_at_linearization.size:
        groups = _sample_connected_components(samples)
        gap_at_linearization, gap_jacobian, areas = _aggregate_constraint_groups(
            gap_at_linearization,
            gap_jacobian,
            areas,
            groups,
        )
    elif base_mode == "slave_face" and gap_at_linearization.size >= 3 and gap_at_linearization.size % 3 == 0:
        groups = [np.arange(start, start + 3, dtype=np.int64) for start in range(0, gap_at_linearization.size, 3)]
        gap_at_linearization, gap_jacobian, areas = _aggregate_constraint_groups(
            gap_at_linearization,
            gap_jacobian,
            areas,
            groups,
        )
    gap_offset = gap_at_linearization - gap_jacobian @ np.asarray(u_linearization, dtype=float).reshape(-1)
    return samples, gap_offset, gap_jacobian, areas


def _aggregate_constraint_groups(
    gaps: np.ndarray,
    rows: np.ndarray,
    areas: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aggregated_gaps: list[float] = []
    aggregated_rows: list[np.ndarray] = []
    aggregated_areas: list[float] = []
    for group in groups:
        ids = np.asarray(group, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            continue
        group_area = areas[ids]
        total_area = float(np.sum(group_area))
        if total_area <= 0.0:
            weights = np.full(ids.size, 1.0 / float(ids.size), dtype=float)
        else:
            weights = group_area / total_area
        aggregated_gaps.append(float(weights @ gaps[ids]))
        aggregated_rows.append(weights @ rows[ids])
        aggregated_areas.append(total_area)
    if not aggregated_gaps:
        return gaps, rows, areas
    return np.asarray(aggregated_gaps, dtype=float), np.vstack(aggregated_rows), np.asarray(aggregated_areas, dtype=float)


def _sample_connected_components(samples: list[Any]) -> list[np.ndarray]:
    """Return connected sample groups using shared slave nodes."""

    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    by_node: dict[int, int] = {}
    for sample_id, sample in enumerate(samples):
        for node in np.asarray(sample.node_ids, dtype=np.int64).reshape(-1):
            key = int(node)
            if key in by_node:
                union(sample_id, by_node[key])
            else:
                by_node[key] = sample_id
    components: dict[int, list[int]] = {}
    for sample_id in range(len(samples)):
        components.setdefault(find(sample_id), []).append(sample_id)
    return [np.asarray(ids, dtype=np.int64) for ids in components.values()]


def _contact_patch_min_edge_length(pair: CroppedGearPair) -> float:
    values: list[float] = []
    for nodes, faces in ((pair.gear1.nodes, pair.gear1.contact_faces), (pair.gear2.nodes, pair.gear2.contact_faces)):
        for face in np.asarray(faces, dtype=np.int64):
            tri = np.asarray(nodes, dtype=float)[face]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                length = float(np.linalg.norm(tri[a] - tri[b]))
                if length > 0.0:
                    values.append(length)
    if not values:
        raise ValueError("contact patch has no positive edge length")
    return float(min(values))


def _contact_patch_representative_length(pair: CroppedGearPair) -> float:
    """Return an Abaqus-style representative contact facet length.

    The hard-contact penalty fallback should scale with the underlying element
    stiffness instead of a case-fitted factor.  A robust representative length
    is the area-weighted mean edge length over the two contact patches; it is
    insensitive to isolated short sliver edges but still follows mesh
    refinement.
    """

    weighted_lengths: list[float] = []
    weights: list[float] = []
    for nodes, faces in ((pair.gear1.nodes, pair.gear1.contact_faces), (pair.gear2.nodes, pair.gear2.contact_faces)):
        for face in np.asarray(faces, dtype=np.int64):
            tri = np.asarray(nodes, dtype=float)[face]
            area = 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
            if area <= 0.0:
                continue
            edges = [
                float(np.linalg.norm(tri[0] - tri[1])),
                float(np.linalg.norm(tri[1] - tri[2])),
                float(np.linalg.norm(tri[2] - tri[0])),
            ]
            positive = [value for value in edges if value > 0.0]
            if not positive:
                continue
            weighted_lengths.append(float(np.mean(positive)))
            weights.append(area)
    if not weighted_lengths:
        raise ValueError("contact patch has no positive representative length")
    return float(np.average(np.asarray(weighted_lengths, dtype=float), weights=np.asarray(weights, dtype=float)))


def _effective_hard_pressure_stiffness(
    *,
    pair: CroppedGearPair,
    young: float,
    poisson: float,
    pressure_stiffness: float,
    hard_enforcement: str,
    pressure_smoothing_factor: float,
) -> float:
    mode = str(hard_enforcement).lower()
    if mode == "element_pressure_smoothing":
        return float(pressure_smoothing_factor) * float(young) / _contact_patch_min_edge_length(pair)
    if mode == "abaqus_standard_penalty":
        # Abaqus/Standard's finite-sliding hard contact uses a penalty fallback
        # tied to representative underlying element stiffness.  We use the
        # same dimensional law, k_p = 10 E / h_rep, with h_rep computed from
        # the current contact facet set rather than from a fitted parameter.
        _ = float(poisson)  # kept in the signature for future material variants
        return 10.0 * float(young) / _contact_patch_representative_length(pair)
    return float(pressure_stiffness)


def _equivalent_hub_reaction(
    residual: np.ndarray,
    *,
    node_ids: np.ndarray,
    nodes_reference: np.ndarray,
    reference_point: np.ndarray,
    drive_direction: np.ndarray,
    current_nodes: np.ndarray | None = None,
) -> dict[str, float]:
    """Project constrained nodal residuals to the equivalent hub/RP reaction."""

    hub = RigidHubMPC(np.asarray(node_ids, dtype=np.int64), np.asarray(nodes_reference, dtype=float), np.asarray(reference_point, dtype=float))
    generalized = hub.generalized_force_from_nodal_forces(
        np.asarray(residual, dtype=float).reshape((-1, 3)),
        current_nodes=current_nodes,
    )
    force = generalized[:3]
    moment = generalized[3:]
    drive = np.asarray(drive_direction, dtype=float)
    drive_norm = float(np.linalg.norm(drive))
    if drive_norm > 0.0:
        drive = drive / drive_norm
    return {
        "rp_force_x": float(force[0]),
        "rp_force_y": float(force[1]),
        "rp_force_z": float(force[2]),
        "rp_force_drive": float(force @ drive),
        "rp_force_norm": float(np.linalg.norm(force)),
        "rp_moment_x": float(moment[0]),
        "rp_moment_y": float(moment[1]),
        "rp_moment_z": float(moment[2]),
        "rp_moment_norm": float(np.linalg.norm(moment)),
    }


def _body_vector_slice(vector: np.ndarray, *, n1_dofs: int, body: int) -> np.ndarray:
    values = np.asarray(vector, dtype=float).reshape(-1)
    if body == 1:
        return values[:n1_dofs]
    if body == 2:
        return values[n1_dofs:]
    raise ValueError("body must be 1 or 2")


def _equivalent_pair_reactions(
    residual: np.ndarray,
    *,
    pair: CroppedGearPair,
    state_x: np.ndarray,
    n1_dofs: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return gear1/gear2 RP generalized reactions for one residual vector."""

    return (
        _equivalent_hub_reaction(
            _body_vector_slice(residual, n1_dofs=n1_dofs, body=1),
            node_ids=pair.gear1.support_nodes,
            nodes_reference=pair.gear1.nodes,
            reference_point=pair.gear1.rp,
            drive_direction=pair.drive_direction,
            current_nodes=state_x[: pair.gear1.nodes.shape[0]],
        ),
        _equivalent_hub_reaction(
            _body_vector_slice(residual, n1_dofs=n1_dofs, body=2),
            node_ids=pair.gear2.support_nodes,
            nodes_reference=pair.gear2.nodes,
            reference_point=pair.gear2.rp,
            drive_direction=pair.drive_direction,
            current_nodes=state_x[pair.gear1.nodes.shape[0] :],
        ),
    )


def solve_sfc_cropped_pair_hard_contact(
    pair: CroppedGearPair,
    *,
    young: float,
    poisson: float,
    density: float,
    pressure_stiffness: float,
    duration: float,
    dt: float,
    target_overclosure: float,
    rotation_rate_z: float,
    max_iterations: int = 6,
    tolerance: float = 1.0e-9,
    hard_enforcement: str = "abaqus_standard_penalty",
    constraint_averaging: str = "slave_node_region_constraint",
    pressure_smoothing_factor: float = 4.0,
    hht_alpha: float = ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA,
    automatic_increment: bool = True,
    cutback_factor: float = 0.5,
    linear_solver: str = "dense",
) -> tuple[list[Row], Row]:
    """Solve the cropped gear pair with implicit Newmark + HARD contact KKT.

    This is the dynamic counterpart of the flat/tilted HARD-contact alignment
    cases. Each step linearizes the current Lagrangian-SDF closest-feature
    payloads around the current iterate and solves the effective Newmark system
    together with the hard normal-contact active set. The cropped gear mesh is
    TET4, so the payload is triangular; the Q4 payload layer remains the C3D8
    specialization validated by the tooth-patch runner.
    """

    enforcement = str(hard_enforcement).lower()
    if enforcement not in {"exact", "pressure_compliance", "element_pressure_smoothing", "abaqus_standard_penalty"}:
        raise ValueError(
            "hard_enforcement must be 'exact', 'pressure_compliance', 'element_pressure_smoothing', or 'abaqus_standard_penalty'"
        )
    solver_kind = str(linear_solver).lower()
    if solver_kind not in {"dense", "sparse", "iterative", "auto"}:
        raise ValueError("linear_solver must be 'dense', 'sparse', 'iterative', or 'auto'")
    effective_pressure_stiffness = _effective_hard_pressure_stiffness(
        pair=pair,
        young=young,
        poisson=poisson,
        pressure_stiffness=pressure_stiffness,
        hard_enforcement=enforcement,
        pressure_smoothing_factor=pressure_smoothing_factor,
    )
    model, contact = _cropped_pair_model_and_contact(
        pair,
        young=young,
        poisson=poisson,
        density=density,
        pressure_stiffness=pressure_stiffness,
        target_overclosure=target_overclosure,
    )
    tracking_contact = _cropped_pair_path_tracking_contact(
        pair,
        pressure_stiffness=pressure_stiffness,
        target_overclosure=target_overclosure,
    )
    fixed0, values0 = _fixed_conditions_for_pair(pair, closure=0.0, rotation_z=0.0)
    u0 = project_fixed_dofs(np.zeros(model.n_dofs, dtype=float), fixed0, values0)
    state_x = model.X + u0.reshape((-1, 3))
    velocity = np.zeros_like(model.X)
    acceleration = np.zeros_like(model.X)
    previous_rhs_balance = np.zeros(model.n_dofs, dtype=float)
    rows: list[Row] = []
    beta, gamma = hht_newmark_parameters(float(hht_alpha))
    max_dt = float(dt)
    min_dt = min(max_dt * 1.0e-4, 1.0e-8)
    if max_dt <= 0.0:
        raise ValueError("dt must be positive")
    if not (0.0 < float(cutback_factor) < 1.0):
        raise ValueError("cutback_factor must lie in (0, 1)")
    start = time.perf_counter()
    timing_internal_tangent = 0.0
    timing_contact_linearization = 0.0
    timing_effective_system = 0.0
    timing_hard_contact_solve = 0.0
    timing_accepted_internal = 0.0
    timing_accepted_contact = 0.0
    timing_reaction_diagnostics = 0.0
    last_iterations = 0
    last_converged = True
    cutback_count = 0
    increment_count = 0
    previous_path_master_face_ids: np.ndarray | None = None
    previous_path_master_barycentric: np.ndarray | None = None
    previous_active_region_ids: tuple[int, ...] | None = None
    t = 0.0
    h_trial = max_dt
    while t < float(duration) - 1.0e-15:
        h = min(float(h_trial), float(duration) - t)
        t_candidate = t + h
        ramp = t_candidate / max(float(duration), float(dt))
        closure = ramp * (pair.initial_patch_gap + float(target_overclosure))
        rotation = float(rotation_rate_z) * t_candidate
        fixed, values = _fixed_conditions_for_pair(pair, closure=closure, rotation_z=rotation)
        u_n = (state_x - model.X).reshape(-1)
        v_n = velocity.reshape(-1)
        a_n = acceleration.reshape(-1)
        c0 = 1.0 / (beta * h * h)
        equilibrium_scale = 1.0 + float(hht_alpha)
        u_pred, v_pred, _accold_after_prediction = calculix_dynamic_predictor(u_n, v_n, a_n, dt=h, beta=beta, gamma=gamma)
        u_guess = project_fixed_dofs(u_pred.copy(), fixed, values)
        solution = None
        final_samples: list[Any] = []
        final_gap_jacobian = np.empty((0, model.n_dofs), dtype=float)
        converged = False
        iteration_count = 0
        previous_active: np.ndarray | None = None
        active_stable_count = 0
        for iteration in range(1, max(1, int(max_iterations)) + 1):
            x_guess = model.X + u_guess.reshape((-1, 3))
            t_section = time.perf_counter()
            internal = stvk_internal_response(model, x_guess, assemble_tangent=True)
            timing_internal_tangent += time.perf_counter() - t_section
            t_section = time.perf_counter()
            final_samples, gap_offset, gap_jacobian, constraint_areas = _hard_contact_linearized_gap_jacobian(
                contact,
                x_guess,
                u_guess,
                n_total_dofs=model.n_dofs,
                constraint_averaging=constraint_averaging,
            )
            timing_contact_linearization += time.perf_counter() - t_section
            final_gap_jacobian = gap_jacobian
            compliance = (
                None
                if enforcement == "exact"
                else 1.0 / (float(effective_pressure_stiffness) * np.maximum(constraint_areas, 1.0e-30))
            )
            t_section = time.perf_counter()
            tangent = internal.tangent.tocsr()
            effective_stiffness = (model.mass_matrix * c0 + tangent * equilibrium_scale).tocsr()
            effective_force = np.asarray(
                model.mass_matrix @ (c0 * u_pred)
                - equilibrium_scale * internal.force.reshape(-1)
                + equilibrium_scale * (tangent @ u_guess)
                - float(hht_alpha) * previous_rhs_balance,
                dtype=float,
            )
            timing_effective_system += time.perf_counter() - t_section
            use_sparse_hard_contact = solver_kind in {"sparse", "iterative"} or (solver_kind == "auto" and model.n_dofs > 12000)
            solve_hard_contact = solve_linear_hard_contact_with_dirichlet_sparse if use_sparse_hard_contact else solve_linear_hard_contact_with_dirichlet
            sparse_solver_mode = "primal_cg" if solver_kind == "iterative" else "direct"
            t_section = time.perf_counter()
            if use_sparse_hard_contact:
                solution = solve_hard_contact(
                    effective_stiffness,
                    effective_force,
                    gap_offset,
                    gap_jacobian,
                    fixed_dofs=fixed,
                    fixed_values=values,
                    equilibrium_jacobian_scale=equilibrium_scale,
                    normal_compliance=compliance,
                    tolerance=float(tolerance),
                    max_iterations=30,
                    linear_solver=sparse_solver_mode,
                    iterative_tolerance=1.0e-10,
                    iterative_max_iterations=600,
                    iterative_fallback_to_direct=True,
                )
            else:
                solution = solve_hard_contact(
                    effective_stiffness,
                    effective_force,
                    gap_offset,
                    gap_jacobian,
                    fixed_dofs=fixed,
                    fixed_values=values,
                    equilibrium_jacobian_scale=equilibrium_scale,
                    normal_compliance=compliance,
                    tolerance=float(tolerance),
                    max_iterations=30,
                )
            timing_hard_contact_solve += time.perf_counter() - t_section
            correction_norm = float(np.linalg.norm(solution.displacement - u_guess))
            displacement_scale = max(1.0, float(np.linalg.norm(solution.displacement)))
            u_guess = solution.displacement.copy()
            iteration_count = iteration
            active_now = np.asarray(solution.active, dtype=bool)
            if previous_active is not None and np.array_equal(active_now, previous_active):
                active_stable_count += 1
            else:
                active_stable_count = 0
            previous_active = active_now.copy()
            strict_correction = correction_norm <= float(tolerance) * displacement_scale
            abaqus_style_contact_accept = bool(solution.converged) and active_stable_count >= 1 and correction_norm <= 1.0e-4 * displacement_scale
            if bool(solution.converged) and (strict_correction or abaqus_style_contact_accept):
                converged = True
                break
        if solution is None:
            raise RuntimeError("hard contact solve did not run")
        if (not converged) and bool(automatic_increment) and h > min_dt * (1.0 + 1.0e-12):
            h_trial = max(min_dt, h * float(cutback_factor))
            cutback_count += 1
            continue
        previous_rhs_before_step = previous_rhs_balance.copy()
        u_new = solution.displacement
        a_new = c0 * (u_new - u_pred)
        v_new = v_pred + gamma * h * a_new
        state_x = model.X + u_new.reshape((-1, 3))
        velocity = v_new.reshape((-1, 3))
        acceleration = a_new.reshape((-1, 3))
        t_section = time.perf_counter()
        internal = stvk_internal_response(model, state_x, assemble_tangent=False)
        timing_accepted_internal += time.perf_counter() - t_section
        t_section = time.perf_counter()
        raw_gaps = np.asarray([float(sample.gap) for sample in contact.samples(state_x)], dtype=float)
        timing_accepted_contact += time.perf_counter() - t_section
        tracking_arrays = tracking_contact.sample_arrays(state_x)
        tracking_regions = (
            None
            if tracking_arrays is None
            else _aggregate_contact_sample_arrays(tracking_arrays, "slave_node_region_constraint")
        )
        path_tracking_metrics, current_path_master_face_ids, current_path_master_barycentric = (
            _contact_path_tracking_metrics_from_arrays(
                tracking_regions,
                previous_path_master_face_ids,
                previous_path_master_barycentric,
                active_gap_tolerance=np.inf,
            )
        )
        if current_path_master_face_ids is not None:
            previous_path_master_face_ids = current_path_master_face_ids
        if current_path_master_barycentric is not None:
            previous_path_master_barycentric = current_path_master_barycentric
        active_region_metrics, current_active_region_ids = _contact_active_region_continuity_metrics_from_arrays(
            tracking_regions,
            previous_active_region_ids,
            active_gap_tolerance=np.inf,
        )
        contact_region_metrics = _contact_region_integral_metrics_from_arrays(
            tracking_regions,
            stiffness=float(effective_pressure_stiffness),
        )
        previous_active_region_ids = current_active_region_ids
        active = np.asarray(solution.active, dtype=bool)
        multipliers = np.asarray(solution.multipliers, dtype=float)
        active_force = float(np.sum(multipliers[active])) if multipliers.size else 0.0
        contact_pressure = np.zeros_like(multipliers)
        if multipliers.size and constraint_areas.size == multipliers.size:
            contact_pressure = multipliers / np.maximum(constraint_areas, 1.0e-30)
        max_contact_pressure = float(np.max(contact_pressure[active])) if multipliers.size and np.any(active) else 0.0
        mass_acceleration = np.asarray(model.mass_matrix @ a_new, dtype=float)
        contact_balance = np.zeros(model.n_dofs, dtype=float)
        if final_gap_jacobian.size and multipliers.size:
            contact_balance = np.asarray(final_gap_jacobian.T @ multipliers, dtype=float)
        static_balance = np.asarray(internal.force.reshape(-1) - contact_balance, dtype=float)
        current_rhs_balance = -static_balance
        dynamic_balance = np.asarray(mass_acceleration + static_balance, dtype=float)
        hht_balance = np.asarray(
            mass_acceleration - equilibrium_scale * current_rhs_balance + float(hht_alpha) * previous_rhs_before_step,
            dtype=float,
        )
        t_section = time.perf_counter()
        n1_dofs = 3 * pair.gear1.nodes.shape[0]
        hub1_hht_reaction, hub2_hht_reaction = _equivalent_pair_reactions(
            hht_balance,
            pair=pair,
            state_x=state_x,
            n1_dofs=n1_dofs,
        )
        hub1_dynamic_reaction, hub2_dynamic_reaction = _equivalent_pair_reactions(
            dynamic_balance,
            pair=pair,
            state_x=state_x,
            n1_dofs=n1_dofs,
        )
        hub1_static_reaction, hub2_static_reaction = _equivalent_pair_reactions(
            static_balance,
            pair=pair,
            state_x=state_x,
            n1_dofs=n1_dofs,
        )
        hub1_inertia_reaction, hub2_inertia_reaction = _equivalent_pair_reactions(
            mass_acceleration,
            pair=pair,
            state_x=state_x,
            n1_dofs=n1_dofs,
        )
        hub1_contact_reaction, hub2_contact_reaction = _equivalent_pair_reactions(
            contact_balance,
            pair=pair,
            state_x=state_x,
            n1_dofs=n1_dofs,
        )
        timing_reaction_diagnostics += time.perf_counter() - t_section
        previous_rhs_balance = current_rhs_balance
        disp = state_x - model.X
        strain_norm = np.linalg.norm(internal.strain, axis=(1, 2)) if internal.strain.size else np.empty(0, dtype=float)
        elastic_strain_norm = _elastic_strain_norm_from_stress(internal.stress, young=young, poisson=poisson)
        equivalent_elastic_strain = _equivalent_elastic_strain_from_mises(
            internal.von_mises,
            young=young,
            poisson=poisson,
        )
        row = {
            "time": t_candidate,
            "dt": h,
            "closure": closure,
            "rotation_z": rotation,
            "active_contact_samples": int(np.count_nonzero(active)),
            "min_gap": float(np.min(raw_gaps)) if raw_gaps.size else 0.0,
            "linearized_min_gap": float(np.min(solution.gaps)) if solution.gaps.size else 0.0,
            "normal_force": active_force,
            "contact_multiplier_sum": active_force,
            "max_contact_pressure": max_contact_pressure,
            "max_displacement_norm": float(np.max(np.linalg.norm(disp, axis=1))),
            "p95_von_mises": float(np.percentile(internal.von_mises, 95.0)) if internal.von_mises.size else 0.0,
            "max_von_mises": float(np.max(internal.von_mises)) if internal.von_mises.size else 0.0,
            "p95_strain_norm": float(np.percentile(strain_norm, 95.0)) if strain_norm.size else 0.0,
            "max_strain_norm": float(np.max(strain_norm)) if strain_norm.size else 0.0,
            "p95_elastic_strain_norm": float(np.percentile(elastic_strain_norm, 95.0)) if elastic_strain_norm.size else 0.0,
            "max_elastic_strain_norm": float(np.max(elastic_strain_norm)) if elastic_strain_norm.size else 0.0,
            "p95_equivalent_elastic_strain": float(np.percentile(equivalent_elastic_strain, 95.0))
            if equivalent_elastic_strain.size
            else 0.0,
            "max_equivalent_elastic_strain": float(np.max(equivalent_elastic_strain)) if equivalent_elastic_strain.size else 0.0,
            "strain_energy": internal.strain_energy,
            "newton_iterations": int(iteration_count),
            "hard_active_set_converged": int(bool(solution.converged)),
            "hard_outer_converged": int(bool(converged)),
            "automatic_increment_cutbacks": int(cutback_count),
            "hard_contact_samples": int(len(final_samples)),
            "hard_constraints": int(solution.gaps.shape[0]),
            "hht_alpha": float(hht_alpha),
            "hht_beta": float(beta),
            "hht_gamma": float(gamma),
        }
        row.update({f"rp1_{key}": value for key, value in hub1_hht_reaction.items()})
        row.update({f"rp2_{key}": value for key, value in hub2_hht_reaction.items()})
        row.update({f"rp1_hht_{key}": value for key, value in hub1_hht_reaction.items()})
        row.update({f"rp2_hht_{key}": value for key, value in hub2_hht_reaction.items()})
        row.update({f"rp1_dynamic_{key}": value for key, value in hub1_dynamic_reaction.items()})
        row.update({f"rp2_dynamic_{key}": value for key, value in hub2_dynamic_reaction.items()})
        row.update({f"rp1_static_{key}": value for key, value in hub1_static_reaction.items()})
        row.update({f"rp2_static_{key}": value for key, value in hub2_static_reaction.items()})
        row.update({f"rp1_inertia_{key}": value for key, value in hub1_inertia_reaction.items()})
        row.update({f"rp2_inertia_{key}": value for key, value in hub2_inertia_reaction.items()})
        row.update({f"rp1_contact_{key}": value for key, value in hub1_contact_reaction.items()})
        row.update({f"rp2_contact_{key}": value for key, value in hub2_contact_reaction.items()})
        row.update(path_tracking_metrics)
        row.update(active_region_metrics)
        row.update(contact_region_metrics)
        row["path_tracking_constraint_regions"] = (
            int(np.asarray(tracking_regions["gaps"], dtype=float).size) if tracking_regions is not None else 0
        )
        row.update(_node_averaged_internal_metric_row(model, internal))
        row["rp_reaction_definition"] = "static_physical_constraint_residual"
        row["rp_force_drive"] = float(hub1_static_reaction["rp_force_drive"])
        row["rp_force_norm"] = float(hub1_static_reaction["rp_force_norm"])
        row["opposing_rp_force_norm"] = float(hub2_static_reaction["rp_force_norm"])
        row["rp_static_force_norm"] = float(hub1_static_reaction["rp_force_norm"])
        row["opposing_rp_static_force_norm"] = float(hub2_static_reaction["rp_force_norm"])
        row["rp_dynamic_force_norm"] = float(hub1_dynamic_reaction["rp_force_norm"])
        row["opposing_rp_dynamic_force_norm"] = float(hub2_dynamic_reaction["rp_force_norm"])
        row["rp_inertia_force_norm"] = float(hub1_inertia_reaction["rp_force_norm"])
        row["opposing_rp_inertia_force_norm"] = float(hub2_inertia_reaction["rp_force_norm"])
        row["rp_contact_force_norm"] = float(hub1_contact_reaction["rp_force_norm"])
        row["opposing_rp_contact_force_norm"] = float(hub2_contact_reaction["rp_force_norm"])
        rows.append(row)
        last_iterations = iteration_count
        last_converged = bool(converged)
        increment_count += 1
        t = t_candidate
        h_trial = max_dt if (not automatic_increment) else min(max_dt, max(h, min_dt) * (1.25 if converged and iteration_count <= 3 else 1.0))
    wall = time.perf_counter() - start
    summary = {
        "sfc_wall_seconds": wall,
        "nodes": int(model.X.shape[0]),
        "elements": int(model.elements.shape[0]),
        "gear1_contact_faces": int(pair.gear1.contact_faces.shape[0]),
        "gear2_contact_faces": int(pair.gear2.contact_faces.shape[0]),
        "gear1_support_nodes": int(pair.gear1.support_nodes.size),
        "gear2_support_nodes": int(pair.gear2.support_nodes.size),
        "initial_patch_gap": float(pair.initial_patch_gap),
        "target_overclosure": float(target_overclosure),
        "final_active_contact_samples": int(rows[-1]["active_contact_samples"]) if rows else 0,
        "final_min_gap": float(rows[-1]["min_gap"]) if rows else 0.0,
        "final_normal_force": float(rows[-1]["normal_force"]) if rows else 0.0,
        "final_rp_force_norm": float(rows[-1].get("rp_force_norm", 0.0)) if rows else 0.0,
        "final_opposing_rp_force_norm": float(rows[-1].get("opposing_rp_force_norm", 0.0)) if rows else 0.0,
        "final_hard_outer_iterations": int(last_iterations),
        "final_hard_outer_converged": int(last_converged),
        "accepted_increment_count": int(increment_count),
        "automatic_increment": str(bool(automatic_increment)).lower(),
        "cutback_count": int(cutback_count),
        "hht_alpha": float(hht_alpha),
        "hht_beta": float(beta),
        "hht_gamma": float(gamma),
        "rp_reaction_definition": "static_physical_constraint_residual",
        "contact_mode": "hard",
        "hard_enforcement": enforcement,
        "constraint_averaging": str(constraint_averaging),
        "linear_solver": solver_kind,
        "effective_hard_pressure_stiffness": float(effective_pressure_stiffness),
        "pressure_smoothing_factor": float(pressure_smoothing_factor),
        "contact_patch_min_edge_length": _contact_patch_min_edge_length(pair),
        "timing_internal_tangent_seconds": float(timing_internal_tangent),
        "timing_contact_linearization_seconds": float(timing_contact_linearization),
        "timing_effective_system_seconds": float(timing_effective_system),
        "timing_hard_contact_solve_seconds": float(timing_hard_contact_solve),
        "timing_accepted_internal_seconds": float(timing_accepted_internal),
        "timing_accepted_contact_seconds": float(timing_accepted_contact),
        "timing_reaction_diagnostics_seconds": float(timing_reaction_diagnostics),
        "status": "completed",
    }
    summary.update(_cropped_patch_contact_gate_metrics(rows, summary, require_monotone_trend=False))
    return rows, summary


def _cropped_patch_contact_gate_metrics(
    history: list[Row],
    summary: Row,
    *,
    min_active_samples: int = 1,
    trend_ratio_floor: float = 0.80,
    require_monotone_trend: bool = False,
    min_path_cache_hit_fraction: float = 0.999,
    min_path_cache_match_fraction: float = 0.999,
    min_active_region_jaccard: float = 0.999,
    max_master_face_switch_fraction: float = 0.60,
    max_master_barycentric_drift: float = 0.75,
) -> Row:
    """Return cropped-patch contact sanity gates before full-gear escalation.

    This gate is intentionally local to the SFC cropped-patch layer.  It checks
    that the HARD-contact solve converged, that a nonzero active contact region
    exists, that pressure/stress/strain measures are finite and nonnegative,
    and that accepted-state path tracking remains continuous when a multi-step
    history is available.  When ``require_monotone_trend`` is enabled, a
    multi-step normal compression sequence must not lose pressure, stress, or
    strain trend relative to the first accepted step.
    """

    rows = list(history)
    if not rows:
        return {
            "cropped_patch_gate_passed": 0,
            "cropped_patch_gate_reason": "empty_history",
            "cropped_patch_convergence_gate_passed": 0,
            "cropped_patch_contact_response_gate_passed": 0,
            "cropped_patch_pressure_stress_gate_passed": 0,
            "cropped_patch_pressure_stress_trend_gate_passed": 0,
            "cropped_patch_path_tracking_gate_passed": 0,
            "cropped_patch_active_region_continuity_gate_passed": 0,
        }
    final = rows[-1]
    finite_keys = [
        "normal_force",
        "max_contact_pressure",
        "p95_von_mises_nodeavg",
        "p95_equivalent_elastic_strain_nodeavg",
        "min_gap",
        "linearized_min_gap",
    ]
    finite_ok = all(np.isfinite(float(row.get(key, 0.0))) for row in rows for key in finite_keys)
    convergence_ok = (
        str(summary.get("status", "")) == "completed"
        and str(summary.get("contact_mode", "")) == "hard"
        and int(summary.get("final_hard_outer_converged", 0)) == 1
        and all(int(row.get("hard_outer_converged", 0)) == 1 for row in rows)
    )
    contact_ok = (
        int(final.get("active_contact_samples", 0)) >= int(min_active_samples)
        and float(final.get("normal_force", 0.0)) >= 0.0
        and float(final.get("max_contact_pressure", 0.0)) >= 0.0
        and int(final.get("hard_constraints", 0)) >= int(min_active_samples)
    )
    pressure_stress_ok = (
        finite_ok
        and float(final.get("max_contact_pressure", 0.0)) > 0.0
        and float(final.get("p95_von_mises_nodeavg", final.get("p95_von_mises", 0.0))) > 0.0
        and float(final.get("p95_equivalent_elastic_strain_nodeavg", final.get("p95_equivalent_elastic_strain", 0.0))) > 0.0
    )
    trend_ok = True
    if bool(require_monotone_trend) and len(rows) > 1:
        first = rows[0]
        for key in ("normal_force", "max_contact_pressure", "p95_von_mises_nodeavg", "p95_equivalent_elastic_strain_nodeavg"):
            first_value = max(float(first.get(key, 0.0)), 1.0e-30)
            final_value = float(final.get(key, 0.0))
            if final_value < float(trend_ratio_floor) * first_value:
                trend_ok = False
                break
    tracking_rows = rows[1:] if len(rows) > 1 else []
    has_path_metrics = any("contact_path_cache_hit_fraction" in row for row in tracking_rows)
    path_cache_hit_min = (
        float(min(float(row.get("contact_path_cache_hit_fraction", 0.0)) for row in tracking_rows))
        if tracking_rows and has_path_metrics
        else 1.0
    )
    path_cache_match_min = (
        float(min(float(row.get("contact_path_cache_match_fraction", 0.0)) for row in tracking_rows))
        if tracking_rows and has_path_metrics
        else 1.0
    )
    face_switch_max = (
        float(max(float(row.get("contact_master_face_switch_fraction", 0.0)) for row in tracking_rows))
        if tracking_rows and has_path_metrics
        else 0.0
    )
    barycentric_drift_max = (
        float(max(float(row.get("contact_master_barycentric_drift_max", 0.0)) for row in tracking_rows))
        if tracking_rows and has_path_metrics
        else 0.0
    )
    active_jaccard_min = (
        float(min(float(row.get("contact_active_region_jaccard", 0.0)) for row in tracking_rows))
        if tracking_rows and has_path_metrics
        else 1.0
    )
    path_tracking_ok = bool(
        path_cache_hit_min >= float(min_path_cache_hit_fraction)
        and path_cache_match_min >= float(min_path_cache_match_fraction)
        and face_switch_max <= float(max_master_face_switch_fraction)
        and barycentric_drift_max <= float(max_master_barycentric_drift)
    )
    active_region_ok = bool(active_jaccard_min >= float(min_active_region_jaccard))
    passed = bool(convergence_ok and contact_ok and pressure_stress_ok and trend_ok and path_tracking_ok and active_region_ok)
    reason = "passed"
    if not convergence_ok:
        reason = "hard_contact_not_converged"
    elif not contact_ok:
        reason = "missing_active_contact_response"
    elif not pressure_stress_ok:
        reason = "invalid_pressure_stress_strain_response"
    elif not trend_ok:
        reason = "pressure_stress_trend_regressed"
    elif not path_tracking_ok:
        reason = "path_tracking_discontinuous"
    elif not active_region_ok:
        reason = "active_region_discontinuous"
    return {
        "cropped_patch_gate_passed": int(passed),
        "cropped_patch_gate_reason": reason,
        "cropped_patch_convergence_gate_passed": int(bool(convergence_ok)),
        "cropped_patch_contact_response_gate_passed": int(bool(contact_ok)),
        "cropped_patch_pressure_stress_gate_passed": int(bool(pressure_stress_ok)),
        "cropped_patch_pressure_stress_trend_gate_passed": int(bool(trend_ok)),
        "cropped_patch_path_tracking_gate_passed": int(bool(path_tracking_ok)),
        "cropped_patch_active_region_continuity_gate_passed": int(bool(active_region_ok)),
        "cropped_patch_min_active_samples_threshold": int(min_active_samples),
        "cropped_patch_trend_ratio_floor": float(trend_ratio_floor),
        "cropped_patch_path_cache_hit_fraction_min_after_first": float(path_cache_hit_min),
        "cropped_patch_path_cache_match_fraction_min_after_first": float(path_cache_match_min),
        "cropped_patch_master_face_switch_fraction_max": float(face_switch_max),
        "cropped_patch_master_barycentric_drift_max": float(barycentric_drift_max),
        "cropped_patch_active_region_jaccard_min_after_first": float(active_jaccard_min),
        "cropped_patch_min_path_cache_hit_threshold": float(min_path_cache_hit_fraction),
        "cropped_patch_min_path_cache_match_threshold": float(min_path_cache_match_fraction),
        "cropped_patch_max_master_face_switch_threshold": float(max_master_face_switch_fraction),
        "cropped_patch_max_master_barycentric_drift_threshold": float(max_master_barycentric_drift),
        "cropped_patch_min_active_region_jaccard_threshold": float(min_active_region_jaccard),
        "cropped_patch_final_max_contact_pressure": float(final.get("max_contact_pressure", 0.0)),
        "cropped_patch_final_p95_von_mises_nodeavg": float(
            final.get("p95_von_mises_nodeavg", final.get("p95_von_mises", 0.0))
        ),
        "cropped_patch_final_p95_equivalent_elastic_strain_nodeavg": float(
            final.get("p95_equivalent_elastic_strain_nodeavg", final.get("p95_equivalent_elastic_strain", 0.0))
        ),
    }


def _write_abaqus_alignment_deck(
    path: Path,
    pair: CroppedGearPair,
    *,
    young: float,
    poisson: float,
    density: float,
    pressure_stiffness: float,
    duration: float,
    dt: float,
    target_overclosure: float,
    rotation_rate_z: float,
    contact_mode: str = "penalty",
) -> None:
    """Write a compact Abaqus deck with matching cropped meshes and RP MPCs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = str(contact_mode).lower()
    if mode not in {"penalty", "hard"}:
        raise ValueError("contact_mode must be 'penalty' or 'hard'")
    if mode == "hard":
        contact_lines = [
            "*Surface Interaction, name=SFC_HARD",
            "*Surface Behavior, pressure-overclosure=HARD",
            "*Friction",
            "0.",
            "*Contact Pair, interaction=SFC_HARD, type=SURFACE TO SURFACE",
            "G1_SURFACE, G2_SURFACE",
        ]
    else:
        contact_lines = [
            "*Surface Interaction, name=SFC_LINEAR",
            "1.",
            "*Friction",
            "0.",
            "*Surface Behavior, pressure-overclosure=LINEAR",
            f"{pressure_stiffness:.12e}",
            "*Contact Pair, interaction=SFC_LINEAR, type=SURFACE TO SURFACE",
            "G1_SURFACE, G2_SURFACE",
        ]
    lines: list[str] = [
        "*Heading",
        "Cropped gear RP-MPC contact alignment deck generated by SFC validation.",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
    ]
    for patch in (pair.gear1, pair.gear2):
        lines.extend([f"*Part, name={patch.name.upper()}", "*Node"])
        for idx, xyz in enumerate(patch.nodes, start=1):
            lines.append(f"{idx}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}")
        lines.append("*Element, type=C3D4")
        for idx, element in enumerate(patch.elements, start=1):
            labels = ", ".join(str(int(v) + 1) for v in element)
            lines.append(f"{idx}, {labels}")
        lines.append("*Elset, elset=ALL_ELEMENTS, generate")
        lines.append(f"1, {patch.elements.shape[0]}, 1")
        lines.append("*Solid Section, elset=ALL_ELEMENTS, material=STEEL")
        lines.append(",")
        lines.append("*End Part")
    lines.extend(["*Assembly, name=ASSEMBLY"])
    for patch in (pair.gear1, pair.gear2):
        lines.extend([f"*Instance, name={patch.name.upper()}-1, part={patch.name.upper()}", "*End Instance"])
    lines.extend(["*Node", f"1000001, {pair.gear1.rp[0]:.12e}, {pair.gear1.rp[1]:.12e}, {pair.gear1.rp[2]:.12e}", f"1000002, {pair.gear2.rp[0]:.12e}, {pair.gear2.rp[1]:.12e}, {pair.gear2.rp[2]:.12e}"])
    _append_assembly_nset(lines, "G1_HUB", "GEAR1-1", pair.gear1.support_nodes + 1)
    _append_assembly_nset(lines, "G2_HUB", "GEAR2-1", pair.gear2.support_nodes + 1)
    lines.extend(["*Nset, nset=G1_RP", "1000001", "*Nset, nset=G2_RP", "1000002"])
    _append_surface(lines, "G1_SURFACE", "GEAR1-1", pair.gear1.surface_entries)
    _append_surface(lines, "G2_SURFACE", "GEAR2-1", pair.gear2.surface_entries)
    lines.extend(["*MPC", "BEAM, G1_HUB, G1_RP", "*MPC", "BEAM, G2_HUB, G2_RP", "*End Assembly"])
    closure = float(pair.initial_patch_gap) + float(target_overclosure)
    final_translation = closure * pair.drive_direction
    final_rotation_z = float(rotation_rate_z) * float(duration)
    lines.extend(
        [
            "*Material, name=STEEL",
            "*Density",
            f"{density:.12e}",
            "*Elastic",
            f"{young:.12e}, {poisson:.12e}",
        ]
    )
    lines.extend(contact_lines)
    lines.extend(
        [
            "*Amplitude, name=RAMP",
            f"0., 0., {duration:.12e}, 1.",
            "*Step, name=alignment, nlgeom=YES, inc=2000",
            "*Dynamic, application=MODERATE DISSIPATION",
            f"{dt:.12e}, {duration:.12e}, {min(float(dt) * 1.0e-4, 1.0e-8):.12e}, {dt:.12e}",
            f"** Rotational boundary values are radians. Final G1_RP UR6 = {final_rotation_z:.12e} rad.",
            "*Boundary",
            "G2_RP, 1, 6",
            "G1_RP, 4, 5",
            "*Boundary, amplitude=RAMP",
            f"G1_RP, 1, 1, {final_translation[0]:.12e}",
            f"G1_RP, 2, 2, {final_translation[1]:.12e}",
            f"G1_RP, 3, 3, {final_translation[2]:.12e}",
            f"G1_RP, 6, 6, {final_rotation_z:.12e}",
            "*Output, field, frequency=1",
            "*Node Output",
            "U, RF",
            "*Element Output",
            "S, E, LE",
            "*End Step",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _append_assembly_nset(lines: list[str], name: str, instance: str, node_labels: np.ndarray) -> None:
    lines.append(f"*Nset, nset={name}, instance={instance}")
    labels = [int(v) for v in np.asarray(node_labels, dtype=np.int64).ravel()]
    for start in range(0, len(labels), 16):
        lines.append(", ".join(str(v) for v in labels[start : start + 16]))


def _append_surface(lines: list[str], name: str, instance: str, entries: tuple[tuple[int, str], ...]) -> None:
    by_side: dict[str, list[int]] = {}
    for element_label, side in entries:
        by_side.setdefault(side.upper(), []).append(int(element_label))
    for side, labels in sorted(by_side.items()):
        set_name = f"{name}_{side}"
        lines.append(f"*Elset, elset={set_name}, instance={instance}")
        for start in range(0, len(labels), 16):
            lines.append(", ".join(str(v) for v in labels[start : start + 16]))
    lines.append(f"*Surface, type=ELEMENT, name={name}")
    for side in sorted(by_side):
        lines.append(f"{name}_{side}, {side}")


def _resolve_abaqus_command(command: str | None) -> str:
    if command:
        return command
    for candidate in ("abaqus", "abq2024"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Abaqus command not found; pass --abaqus-command")


def _run_command(command: list[str], *, cwd: Path, log_path: Path) -> float:
    start = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    log_path.write_text(result.stdout, encoding="utf-8", errors="ignore")
    print(result.stdout)
    failed_text = result.stdout.lower()
    if result.returncode != 0 or "exited with errors" in failed_text or "fatal errors" in failed_text:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}")
    return elapsed


def _abaqus_wallclock_seconds(sta_path: Path) -> float | None:
    if not sta_path.exists():
        return None
    text = sta_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"WALLCLOCK TIME[^\d]*([0-9.]+)", text, flags=re.IGNORECASE)
    if matches:
        return float(matches[-1])
    matches = re.findall(r"TOTAL JOB TIME[^\d]*([0-9.]+)", text, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else None


def _abaqus_metrics_export_script() -> str:
    return r'''
from __future__ import print_function

import csv
import math
import sys

from odbAccess import openOdb


def norm3(data):
    return math.sqrt(float(data[0]) ** 2 + float(data[1]) ** 2 + float(data[2]) ** 2)


def parse_int_set(text):
    if not text:
        return set()
    return set(int(part) for part in text.split(",") if part)


def parse_vec3(text):
    values = [float(part) for part in text.split(",")]
    if len(values) != 3:
        raise RuntimeError("expected three comma-separated coordinates")
    return values


def dot3(a, b):
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def cross3(a, b):
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def add3(a, b):
    return [float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2])]


def tensor_norm(data):
    values = [float(v) for v in data]
    if len(values) < 6:
        return 0.0
    return math.sqrt(values[0] ** 2 + values[1] ** 2 + values[2] ** 2 + 2.0 * (values[3] ** 2 + values[4] ** 2 + values[5] ** 2))


def von_mises(data):
    values = [float(v) for v in data]
    if len(values) < 6:
        return 0.0
    s11, s22, s33, s12, s13, s23 = values[:6]
    return math.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * (s12 * s12 + s13 * s13 + s23 * s23))


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct / 100.0))
    return float(ordered[index])


def append_scalar_field(values, target):
    for value in values:
        data = value.data
        try:
            target.append(float(data))
        except Exception:
            target.append(float(data[0]))


def equivalent_elastic_strain_from_mises(mises, young, poisson):
    shear = float(young) / (2.0 * (1.0 + float(poisson)))
    return float(mises) / max(3.0 * shear, 1.0e-30)


odb = openOdb(path=sys.argv[1], readOnly=True)
out_path = sys.argv[2]
drive = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]
young = float(sys.argv[6])
poisson = float(sys.argv[7])
g1_hub_labels = parse_int_set(sys.argv[8])
g2_hub_labels = parse_int_set(sys.argv[9])
g1_rp = parse_vec3(sys.argv[10])
g2_rp = parse_vec3(sys.argv[11])
try:
    assembly = odb.rootAssembly
    instance_coords = {}
    for name, inst in assembly.instances.items():
        instance_coords[name.upper()] = dict((int(node.label), [float(v) for v in node.coordinates[:3]]) for node in inst.nodes)
    step = odb.steps[list(odb.steps.keys())[0]]
    with open(out_path, "w", newline="") as handle:
        fieldnames = [
            "time",
            "max_displacement_norm",
            "p95_von_mises",
            "max_von_mises",
            "p95_strain_norm",
            "max_strain_norm",
            "p95_log_strain_norm",
            "max_log_strain_norm",
            "p95_equivalent_elastic_strain",
            "max_equivalent_elastic_strain",
            "min_copen",
            "max_cpressure",
            "rp_force_drive",
            "rp_force_norm",
            "rp2_force_drive",
            "rp2_force_norm",
            "hub_virtual_force_drive",
            "hub_virtual_force_norm",
            "hub2_virtual_force_drive",
            "hub2_virtual_force_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame in step.frames:
            u_norm = []
            vm = []
            strain = []
            log_strain = []
            copen = []
            cpressure = []
            rp_force_drive = 0.0
            rp_force_norm = 0.0
            rp2_force_drive = 0.0
            rp2_force_norm = 0.0
            hub_force = [0.0, 0.0, 0.0]
            hub_moment = [0.0, 0.0, 0.0]
            hub2_force = [0.0, 0.0, 0.0]
            hub2_moment = [0.0, 0.0, 0.0]
            if "U" in frame.fieldOutputs:
                for value in frame.fieldOutputs["U"].values:
                    u_norm.append(norm3(value.data))
            if "S" in frame.fieldOutputs:
                for value in frame.fieldOutputs["S"].values:
                    try:
                        vm.append(float(value.mises))
                    except Exception:
                        vm.append(von_mises(value.data))
            if "E" in frame.fieldOutputs:
                for value in frame.fieldOutputs["E"].values:
                    strain.append(tensor_norm(value.data))
            if "LE" in frame.fieldOutputs:
                for value in frame.fieldOutputs["LE"].values:
                    log_strain.append(tensor_norm(value.data))
            if not strain:
                strain = list(log_strain)
            equivalent_strain = [equivalent_elastic_strain_from_mises(value, young, poisson) for value in vm]
            if "RF" in frame.fieldOutputs:
                for value in frame.fieldOutputs["RF"].values:
                    inst_name = ""
                    try:
                        inst_name = value.instance.name.upper()
                    except Exception:
                        inst_name = ""
                    if int(value.nodeLabel) == 1000001:
                        data = [float(v) for v in value.data[:3]]
                        rp_force_drive += data[0] * drive[0] + data[1] * drive[1] + data[2] * drive[2]
                        rp_force_norm = norm3(data)
                    if int(value.nodeLabel) == 1000002:
                        data = [float(v) for v in value.data[:3]]
                        rp2_force_drive += data[0] * drive[0] + data[1] * drive[1] + data[2] * drive[2]
                        rp2_force_norm = norm3(data)
                    if inst_name == "GEAR1-1" and int(value.nodeLabel) in g1_hub_labels:
                        data = [float(v) for v in value.data[:3]]
                        coords = instance_coords.get(inst_name, {}).get(int(value.nodeLabel), g1_rp)
                        lever = [coords[0] - g1_rp[0], coords[1] - g1_rp[1], coords[2] - g1_rp[2]]
                        hub_force = add3(hub_force, data)
                        hub_moment = add3(hub_moment, cross3(lever, data))
                    if inst_name == "GEAR2-1" and int(value.nodeLabel) in g2_hub_labels:
                        data = [float(v) for v in value.data[:3]]
                        coords = instance_coords.get(inst_name, {}).get(int(value.nodeLabel), g2_rp)
                        lever = [coords[0] - g2_rp[0], coords[1] - g2_rp[1], coords[2] - g2_rp[2]]
                        hub2_force = add3(hub2_force, data)
                        hub2_moment = add3(hub2_moment, cross3(lever, data))
            for name, output in frame.fieldOutputs.items():
                upper = name.upper()
                if "COPEN" in upper:
                    append_scalar_field(output.values, copen)
                if "CPRESS" in upper:
                    append_scalar_field(output.values, cpressure)
            writer.writerow({
                "time": float(frame.frameValue),
                "max_displacement_norm": max(u_norm) if u_norm else 0.0,
                "p95_von_mises": percentile(vm, 95.0),
                "max_von_mises": max(vm) if vm else 0.0,
                "p95_strain_norm": percentile(strain, 95.0),
                "max_strain_norm": max(strain) if strain else 0.0,
                "p95_log_strain_norm": percentile(log_strain, 95.0),
                "max_log_strain_norm": max(log_strain) if log_strain else 0.0,
                "p95_equivalent_elastic_strain": percentile(equivalent_strain, 95.0),
                "max_equivalent_elastic_strain": max(equivalent_strain) if equivalent_strain else 0.0,
                "min_copen": min(copen) if copen else 0.0,
                "max_cpressure": max(cpressure) if cpressure else 0.0,
                "rp_force_drive": rp_force_drive,
                "rp_force_norm": rp_force_norm,
                "rp2_force_drive": rp2_force_drive,
                "rp2_force_norm": rp2_force_norm,
                "hub_virtual_force_drive": dot3(hub_force, drive),
                "hub_virtual_force_norm": norm3(hub_force),
                "hub2_virtual_force_drive": dot3(hub2_force, drive),
                "hub2_virtual_force_norm": norm3(hub2_force),
            })
finally:
    odb.close()
'''


def run_abaqus_alignment(
    deck_path: Path,
    out_dir: Path,
    pair: CroppedGearPair,
    *,
    abaqus_command: str | None,
    young: float,
    poisson: float,
) -> tuple[Path, Row]:
    run_dir = out_dir / "abaqus_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    job_name = "cropped_gear_alignment"
    inp_path = run_dir / f"{job_name}.inp"
    for old in run_dir.glob(f"{job_name}.*"):
        old.unlink()
    inp_path.write_text(deck_path.read_text(encoding="ascii"), encoding="ascii")
    command = _resolve_abaqus_command(abaqus_command)
    analysis_wall = _run_command(
        [command, f"job={job_name}", f"input={inp_path.name}", "interactive"],
        cwd=run_dir,
        log_path=out_dir / "abaqus_analysis_stdout.log",
    )
    odb = run_dir / f"{job_name}.odb"
    if not odb.exists():
        raise FileNotFoundError(odb)
    export_script = run_dir / "export_cropped_gear_metrics.py"
    export_script.write_text(_abaqus_metrics_export_script(), encoding="ascii")
    metrics = out_dir / "abaqus_cropped_gear_metrics.csv"
    export_wall = _run_command(
        [
            command,
            "python",
            str(export_script.resolve()),
            str(odb.resolve()),
            str(metrics.resolve()),
            f"{pair.drive_direction[0]:.16e}",
            f"{pair.drive_direction[1]:.16e}",
            f"{pair.drive_direction[2]:.16e}",
            f"{float(young):.16e}",
            f"{float(poisson):.16e}",
            ",".join(str(int(v) + 1) for v in pair.gear1.support_nodes),
            ",".join(str(int(v) + 1) for v in pair.gear2.support_nodes),
            ",".join(f"{float(v):.16e}" for v in pair.gear1.rp),
            ",".join(f"{float(v):.16e}" for v in pair.gear2.rp),
        ],
        cwd=run_dir,
        log_path=out_dir / "abaqus_export_stdout.log",
    )
    row: Row = {
        "abaqus_analysis_wall_seconds": analysis_wall,
        "abaqus_export_wall_seconds": export_wall,
        "abaqus_reported_wall_seconds": _abaqus_wallclock_seconds(run_dir / f"{job_name}.sta") or 0.0,
        "abaqus_metrics": str(metrics),
        "status": "completed",
    }
    return metrics, row


def _read_csv_rows(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_float_column(rows: list[Row], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0) or 0.0) for row in rows], dtype=float)


def _relative_error(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(b)), 1.0e-12)


def _metric_available(rows: list[Row], key: str) -> bool:
    return bool(rows) and key in rows[0]


def _preferred_strain_metric(sfc_rows: list[Row], abaqus_rows: list[Row]) -> tuple[str, str, str]:
    """Return the primary strain scalar used for Abaqus/SFC alignment.

    We keep raw tensor strain norms in the CSV, but the paper-facing scalar is
    the equivalent elastic strain recovered from von Mises stress.  This avoids
    mixing SFC Green strain diagnostics with Abaqus element output conventions.
    """

    if _metric_available(sfc_rows, "p95_equivalent_elastic_strain") and _metric_available(
        abaqus_rows,
        "p95_equivalent_elastic_strain",
    ):
        return "p95_equivalent_elastic_strain", "p95_equivalent_elastic_strain", "p95 equivalent elastic strain"
    return "p95_strain_norm", "p95_strain_norm", "p95 strain norm"


def _preferred_sfc_reaction_keys(sfc_rows: list[Row]) -> tuple[str, str, str]:
    """Return SFC RP reaction keys aligned with Abaqus/Standard RF output."""

    if sfc_rows and "rp1_hht_rp_force_norm" in sfc_rows[0] and "rp2_hht_rp_force_norm" in sfc_rows[0]:
        if np.any(np.abs(_as_float_column(sfc_rows, "rp1_hht_rp_force_norm")) > 0.0):
            return "rp1_hht_rp_force_norm", "rp2_hht_rp_force_norm", "HHT RP reaction norm"
    if sfc_rows and "rp_dynamic_force_norm" in sfc_rows[0] and "opposing_rp_dynamic_force_norm" in sfc_rows[0]:
        if np.any(np.abs(_as_float_column(sfc_rows, "rp_dynamic_force_norm")) > 0.0):
            return "rp_dynamic_force_norm", "opposing_rp_dynamic_force_norm", "dynamic RP reaction norm"
    force_key = "rp_force_norm" if sfc_rows and "rp_force_norm" in sfc_rows[0] else "normal_force"
    opposing_force_key = "opposing_rp_force_norm" if sfc_rows and "opposing_rp_force_norm" in sfc_rows[0] else force_key
    return force_key, opposing_force_key, "RP/contact reaction norm"


def compare_histories(sfc_history: Path, abaqus_history: Path, out_path: Path) -> list[Row]:
    sfc_rows = _read_csv_rows(sfc_history)
    abaqus_rows = _read_csv_rows(abaqus_history)
    t_sfc = _as_float_column(sfc_rows, "time")
    t_abaqus = _as_float_column(abaqus_rows, "time")
    force_key, opposing_force_key, _force_title = _preferred_sfc_reaction_keys(sfc_rows)
    abaqus_force_key = (
        "hub_virtual_force_norm"
        if abaqus_rows
        and "hub_virtual_force_norm" in abaqus_rows[0]
        and np.any(np.abs(_as_float_column(abaqus_rows, "hub_virtual_force_norm")) > 0.0)
        else "rp_force_norm"
    )
    abaqus_opposing_force_key = (
        "hub2_virtual_force_norm"
        if abaqus_rows
        and "hub2_virtual_force_norm" in abaqus_rows[0]
        and np.any(np.abs(_as_float_column(abaqus_rows, "hub2_virtual_force_norm")) > 0.0)
        else "rp2_force_norm"
    )
    strain_key, abaqus_strain_key, _strain_title = _preferred_strain_metric(sfc_rows, abaqus_rows)
    metrics = [
        ("max_displacement_norm", "max_displacement_norm"),
        ("p95_von_mises", "p95_von_mises"),
        ("max_von_mises", "max_von_mises"),
        (strain_key, abaqus_strain_key),
        ("max_strain_norm", "max_strain_norm"),
        (force_key, abaqus_force_key),
        (opposing_force_key, abaqus_opposing_force_key),
    ]
    if (
        _metric_available(sfc_rows, "max_contact_pressure")
        and _metric_available(abaqus_rows, "max_cpressure")
        and np.any(np.abs(_as_float_column(abaqus_rows, "max_cpressure")) > 0.0)
    ):
        metrics.append(("max_contact_pressure", "max_cpressure"))
    if t_abaqus.size == 0:
        raise RuntimeError(f"Abaqus metrics file has no frames: {abaqus_history}")
    rows: list[Row] = []
    for i, t in enumerate(t_sfc):
        row: Row = {"time": float(t)}
        for sfc_key, abaqus_key in metrics:
            sfc_value = float(sfc_rows[i].get(sfc_key, 0.0) or 0.0)
            abaqus_values = _as_float_column(abaqus_rows, abaqus_key)
            abaqus_value = float(np.interp(t, t_abaqus, abaqus_values))
            row[f"sfc_{sfc_key}"] = sfc_value
            row[f"abaqus_{abaqus_key}"] = abaqus_value
            row[f"{sfc_key}_abs_error"] = abs(sfc_value - abaqus_value)
            row[f"{sfc_key}_rel_error"] = _relative_error(sfc_value, abaqus_value)
        rows.append(row)
    _write_csv(out_path, rows)
    return rows


def plot_alignment_curves(sfc_history: Path, abaqus_history: Path, error_rows: list[Row], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    sfc_rows = _read_csv_rows(sfc_history)
    abaqus_rows = _read_csv_rows(abaqus_history)
    t_sfc = _as_float_column(sfc_rows, "time")
    t_abq = _as_float_column(abaqus_rows, "time")
    _force_key, opposing_force_key, force_title = _preferred_sfc_reaction_keys(sfc_rows)
    abaqus_opposing_force_key = (
        "hub2_virtual_force_norm"
        if abaqus_rows
        and "hub2_virtual_force_norm" in abaqus_rows[0]
        and np.any(np.abs(_as_float_column(abaqus_rows, "hub2_virtual_force_norm")) > 0.0)
        else "rp2_force_norm"
    )
    strain_key, abaqus_strain_key, strain_title = _preferred_strain_metric(sfc_rows, abaqus_rows)
    panels = [
        ("max_displacement_norm", "max_displacement_norm", "max displacement norm"),
        ("p95_von_mises", "p95_von_mises", "p95 von Mises stress"),
        (strain_key, abaqus_strain_key, strain_title),
        (opposing_force_key, abaqus_opposing_force_key, f"fixed {force_title}"),
    ]

    def with_initial_zero(time_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if time_values.size == 1 and float(time_values[0]) > 0.0:
            return np.concatenate(([0.0], time_values)), np.concatenate(([0.0], values))
        return time_values, values

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for ax, (sfc_key, abq_key, title) in zip(axes.ravel(), panels, strict=True):
        sfc_y = _as_float_column(sfc_rows, sfc_key)
        abq_y = _as_float_column(abaqus_rows, abq_key)
        t_sfc_plot, sfc_y_plot = with_initial_zero(t_sfc, sfc_y)
        t_abq_plot, abq_y_plot = with_initial_zero(t_abq, abq_y)
        rel_key = f"{sfc_key}_rel_error"
        final_error = float(error_rows[-1].get(rel_key, 0.0)) if error_rows else 0.0
        ax.plot(
            t_abq_plot,
            abq_y_plot,
            color="#1f77b4",
            linewidth=1.8,
            marker="o",
            markersize=3.6,
            label="Abaqus/Standard",
            zorder=2,
        )
        ax.plot(
            t_sfc_plot,
            sfc_y_plot,
            color="#d95f02",
            linewidth=1.8,
            linestyle="--",
            marker="s",
            markersize=3.8,
            markerfacecolor="white",
            label=f"SFC ({final_error * 100:.2f}% final err.)",
            zorder=3,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (s)", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.legend(loc="best", fontsize=8, frameon=False)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_summary(
    path: Path,
    summary: Row,
    history_path: Path,
    deck_path: Path,
    *,
    abaqus_row: Row | None = None,
    error_rows: list[Row] | None = None,
    figure_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = [
        "# Cropped Gear Lagrangian-SDF Implicit Alignment",
        "",
        "This runner uses SFC internal TET4 HHT dynamics, Abaqus-style RP/hub MPC kinematics in eliminated Dirichlet form, and Lagrangian-SDF surface contact tangent.",
        "",
        f"- contact mode: {summary.get('contact_mode', 'penalty')}",
        f"- hard enforcement: {summary.get('hard_enforcement', '')}",
        f"- constraint averaging: {summary.get('constraint_averaging', '')}",
        f"- linear solver: {summary.get('linear_solver', 'dense')}",
        f"- effective hard pressure stiffness: {float(summary.get('effective_hard_pressure_stiffness', 0.0)):.6e}",
        f"- pressure smoothing factor: {float(summary.get('pressure_smoothing_factor', 0.0)):.6e}",
        f"- HHT alpha/beta/gamma: {float(summary.get('hht_alpha', 0.0)):.6e} / {float(summary.get('hht_beta', 0.0)):.6e} / {float(summary.get('hht_gamma', 0.0)):.6e}",
        f"- RP reaction definition: {summary.get('rp_reaction_definition', 'static_physical_constraint_residual')}",
        f"- automatic increment: {summary.get('automatic_increment', '')}",
        f"- cutback count: {int(summary.get('cutback_count', 0))}",
        f"- SFC wall time: {summary['sfc_wall_seconds']:.6f} s",
        f"- timing internal+tangent: {float(summary.get('timing_internal_tangent_seconds', 0.0)):.6f} s",
        f"- timing effective system: {float(summary.get('timing_effective_system_seconds', 0.0)):.6f} s",
        f"- timing hard-contact solve: {float(summary.get('timing_hard_contact_solve_seconds', 0.0)):.6f} s",
        f"- timing contact linearization: {float(summary.get('timing_contact_linearization_seconds', 0.0)):.6f} s",
        f"- timing accepted-state diagnostics: {float(summary.get('timing_accepted_internal_seconds', 0.0)) + float(summary.get('timing_accepted_contact_seconds', 0.0)) + float(summary.get('timing_reaction_diagnostics_seconds', 0.0)):.6f} s",
        f"- nodes/elements: {summary['nodes']} / {summary['elements']}",
        f"- contact faces: {summary['gear1_contact_faces']} / {summary['gear2_contact_faces']}",
        f"- support nodes: {summary['gear1_support_nodes']} / {summary['gear2_support_nodes']}",
        f"- initial patch gap: {summary['initial_patch_gap']:.6e}",
        f"- target overclosure: {summary['target_overclosure']:.6e}",
        f"- final active samples: {summary['final_active_contact_samples']}",
        f"- final min gap: {summary['final_min_gap']:.6e}",
        f"- final normal force: {summary['final_normal_force']:.6e}",
        f"- final RP reaction norm: {summary.get('final_rp_force_norm', 0.0):.6e}",
        f"- final opposing RP reaction norm: {summary.get('final_opposing_rp_force_norm', 0.0):.6e}",
        f"- final hard outer iterations: {summary.get('final_hard_outer_iterations', '')}",
        f"- final hard outer converged: {summary.get('final_hard_outer_converged', '')}",
        f"- cropped patch contact gate: {'PASS' if int(summary.get('cropped_patch_gate_passed', 0)) else 'FAIL'} ({summary.get('cropped_patch_gate_reason', '')})",
        f"- cropped patch path tracking gate: {'PASS' if int(summary.get('cropped_patch_path_tracking_gate_passed', 0)) else 'FAIL'}",
        f"- cropped patch active-region continuity gate: {'PASS' if int(summary.get('cropped_patch_active_region_continuity_gate_passed', 0)) else 'FAIL'}",
        f"- cropped patch min path-cache hit/match: {float(summary.get('cropped_patch_path_cache_hit_fraction_min_after_first', 0.0)):.6e} / {float(summary.get('cropped_patch_path_cache_match_fraction_min_after_first', 0.0)):.6e}",
        f"- cropped patch max face switch/drift: {float(summary.get('cropped_patch_master_face_switch_fraction_max', 0.0)):.6e} / {float(summary.get('cropped_patch_master_barycentric_drift_max', 0.0)):.6e}",
        f"- cropped patch min active-region Jaccard: {float(summary.get('cropped_patch_active_region_jaccard_min_after_first', 0.0)):.6e}",
        f"- cropped patch final max pressure: {float(summary.get('cropped_patch_final_max_contact_pressure', 0.0)):.6e}",
        f"- cropped patch final p95 von Mises nodeavg: {float(summary.get('cropped_patch_final_p95_von_mises_nodeavg', 0.0)):.6e}",
        f"- cropped patch final p95 equivalent strain nodeavg: {float(summary.get('cropped_patch_final_p95_equivalent_elastic_strain_nodeavg', 0.0)):.6e}",
        f"- history CSV: `{history_path.name}`",
        f"- optional Abaqus alignment deck: `{deck_path.name}`",
    ]
    if abaqus_row is not None:
        text.extend(
            [
                "",
                "## Abaqus Native-Contact Alignment",
                "",
                f"- Abaqus analysis wall time: {float(abaqus_row.get('abaqus_analysis_wall_seconds', 0.0)):.6f} s",
                f"- Abaqus reported wall time: {float(abaqus_row.get('abaqus_reported_wall_seconds', 0.0)):.6f} s",
                f"- Abaqus export wall time: {float(abaqus_row.get('abaqus_export_wall_seconds', 0.0)):.6f} s",
            ]
        )
    if error_rows:
        final = error_rows[-1]
        strain_error_key = (
            "p95_equivalent_elastic_strain_rel_error"
            if "p95_equivalent_elastic_strain_rel_error" in final
            else "p95_strain_norm_rel_error"
        )
        strain_label = "p95 equivalent elastic strain" if strain_error_key.startswith("p95_equivalent") else "p95 strain norm"
        text.extend(
            [
                "",
                "## Final-Time Alignment Errors",
                "",
                f"- max displacement norm rel. error: {100.0 * float(final.get('max_displacement_norm_rel_error', 0.0)):.3f}%",
                f"- p95 von Mises rel. error: {100.0 * float(final.get('p95_von_mises_rel_error', 0.0)):.3f}%",
                f"- {strain_label} rel. error: {100.0 * float(final.get(strain_error_key, 0.0)):.3f}%",
                f"- RP/contact reaction rel. error: {100.0 * float(final.get('rp_force_norm_rel_error', final.get('normal_force_rel_error', 0.0))):.3f}%",
                f"- fixed-RP reaction rel. error: {100.0 * float(final.get('opposing_rp_force_norm_rel_error', 0.0)):.3f}%",
            ]
        )
    if figure_path is not None:
        text.extend(["", f"- alignment figure: `{figure_path.name}`"])
    text.extend(["", "Abaqus is not used by the SFC solve path. The deck is emitted only for external native-contact comparison."])
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--faces-per-body", type=int, default=10)
    parser.add_argument("--expansion-rings", type=int, default=1)
    parser.add_argument("--duration", type=float, default=2.0e-3)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--overclosure", type=float, default=3.0e-4)
    parser.add_argument(
        "--rotation-rate-z",
        "--rotation-rate-z-rad-s",
        dest="rotation_rate_z",
        type=float,
        default=2.0,
        help="Prescribed G1 RP angular velocity about z in radians per second.",
    )
    parser.add_argument("--pressure-stiffness", type=float, default=DEFAULT_PRESSURE_STIFFNESS)
    parser.add_argument("--contact-mode", choices=("penalty", "hard"), default="penalty")
    parser.add_argument(
        "--hard-enforcement",
        choices=("exact", "pressure_compliance", "element_pressure_smoothing", "abaqus_standard_penalty"),
        default="abaqus_standard_penalty",
    )
    parser.add_argument("--pressure-smoothing-factor", type=float, default=4.0)
    parser.add_argument("--hht-alpha", type=float, default=ABAQUS_STANDARD_MODERATE_DISSIPATION_ALPHA)
    parser.add_argument("--no-automatic-increment", action="store_true")
    parser.add_argument("--cutback-factor", type=float, default=0.5)
    parser.add_argument(
        "--constraint-averaging",
        choices=(
            "none",
            "slave_face",
            "slave_face_constraint",
            "slave_node",
            "slave_node_constraint",
            "slave_node_region",
            "slave_node_region_constraint",
            "slave_node_region_participation",
            "slave_node_region_signed_participation",
            "surface_patch",
            "surface_patch_constraint",
        ),
        default="slave_node_region_constraint",
    )
    parser.add_argument("--hard-max-iterations", type=int, default=6)
    parser.add_argument("--linear-solver", choices=("dense", "sparse", "auto"), default="dense")
    parser.add_argument("--run-abaqus", action="store_true", help="Run the generated Abaqus native-contact deck and compare curves.")
    parser.add_argument("--abaqus-command", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Use the default small cropped patch settings.")
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    source = args.source
    if not source.exists():
        raise FileNotFoundError(source)
    model = parse_gear_input(source)
    pair = build_cropped_pair(model, faces_per_body=int(args.faces_per_body), expansion_rings=int(args.expansion_rings))
    if str(args.contact_mode).lower() == "hard":
        history, summary = solve_sfc_cropped_pair_hard_contact(
            pair,
            young=model.young,
            poisson=model.poisson,
            density=model.density,
            pressure_stiffness=float(args.pressure_stiffness),
            duration=float(args.duration),
            dt=float(args.dt),
            target_overclosure=float(args.overclosure),
            rotation_rate_z=float(args.rotation_rate_z),
            max_iterations=int(args.hard_max_iterations),
            hard_enforcement=str(args.hard_enforcement),
            constraint_averaging=str(args.constraint_averaging),
            pressure_smoothing_factor=float(args.pressure_smoothing_factor),
            hht_alpha=float(args.hht_alpha),
            automatic_increment=not bool(args.no_automatic_increment),
            cutback_factor=float(args.cutback_factor),
            linear_solver=str(args.linear_solver),
        )
    else:
        history, summary = solve_sfc_cropped_pair(
            pair,
            young=model.young,
            poisson=model.poisson,
            density=model.density,
            pressure_stiffness=float(args.pressure_stiffness),
            duration=float(args.duration),
            dt=float(args.dt),
            target_overclosure=float(args.overclosure),
            rotation_rate_z=float(args.rotation_rate_z),
        )
    history_path = out_dir / "sfc_cropped_gear_lagrangian_sdf_history.csv"
    summary_path = out_dir / "cropped_gear_lagrangian_sdf_summary.csv"
    deck_path = out_dir / "abaqus_cropped_gear_alignment.inp"
    report_path = out_dir / "cropped_gear_lagrangian_sdf_summary.md"
    _write_csv(history_path, history)
    _write_csv(summary_path, [summary])
    _write_abaqus_alignment_deck(
        deck_path,
        pair,
        young=model.young,
        poisson=model.poisson,
        density=model.density,
        pressure_stiffness=float(args.pressure_stiffness),
        duration=float(args.duration),
        dt=float(args.dt),
        target_overclosure=float(args.overclosure),
        rotation_rate_z=float(args.rotation_rate_z),
        contact_mode=str(args.contact_mode),
    )
    abaqus_row: Row | None = None
    error_rows: list[Row] | None = None
    figure_path: Path | None = None
    if args.run_abaqus:
        abaqus_metrics, abaqus_row = run_abaqus_alignment(
            deck_path,
            out_dir,
            pair,
            abaqus_command=args.abaqus_command,
            young=model.young,
            poisson=model.poisson,
        )
        _write_csv(out_dir / "abaqus_runtime.csv", [abaqus_row])
        error_rows = compare_histories(history_path, abaqus_metrics, out_dir / "sfc_vs_abaqus_alignment_errors.csv")
        figure_path = out_dir / "sfc_vs_abaqus_alignment_curves.png"
        plot_alignment_curves(history_path, abaqus_metrics, error_rows, figure_path)
    write_summary(report_path, summary, history_path, deck_path, abaqus_row=abaqus_row, error_rows=error_rows, figure_path=figure_path)
    print(report_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
