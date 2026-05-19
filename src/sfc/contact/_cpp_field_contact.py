"""Optional C++ field-contact kernels."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - availability depends on local extension build.
    from sfc import _sfc_cpp
except Exception as exc:  # pragma: no cover
    _BACKEND = None
    _IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover
    _BACKEND = _sfc_cpp
    _IMPORT_ERROR = None


def is_available() -> bool:
    return _BACKEND is not None


def _require_backend() -> None:
    if _BACKEND is None:
        raise RuntimeError("C++ field-contact backend is not available") from _IMPORT_ERROR


def surface_penalty_response(
    points: np.ndarray,
    sample_node_ids: np.ndarray,
    sample_weights: np.ndarray,
    area_weights: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: np.ndarray,
    phi: np.ndarray,
    valid_mask: np.ndarray,
    closest_face_id: np.ndarray,
    barycentric_grid: np.ndarray,
    closest_normal: np.ndarray,
    boundary_faces: np.ndarray,
    pressure_stiffness: float,
    n_total_dofs: int,
    slave_dof_offset: int,
    master_dof_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    _require_backend()
    return _BACKEND.surface_penalty_response(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(sample_node_ids, dtype=np.int64),
        np.ascontiguousarray(sample_weights, dtype=np.float64),
        np.ascontiguousarray(area_weights, dtype=np.float64),
        np.ascontiguousarray(origin, dtype=np.float64),
        np.ascontiguousarray(spacing, dtype=np.float64),
        np.ascontiguousarray(shape, dtype=np.int64),
        np.ascontiguousarray(phi, dtype=np.float64),
        np.ascontiguousarray(valid_mask, dtype=bool),
        np.ascontiguousarray(closest_face_id, dtype=np.int64),
        np.ascontiguousarray(barycentric_grid, dtype=np.float64),
        np.ascontiguousarray(closest_normal, dtype=np.float64),
        np.ascontiguousarray(boundary_faces, dtype=np.int64),
        float(pressure_stiffness),
        int(n_total_dofs),
        int(slave_dof_offset),
        int(master_dof_offset),
    )


def contact_stiffness_matvec(
    vector: np.ndarray,
    scale: np.ndarray,
    slave_node_ids: np.ndarray,
    slave_weights: np.ndarray,
    gradients: np.ndarray,
    face_node_ids: np.ndarray,
    grid_weights: np.ndarray,
    barycentric: np.ndarray,
    normals: np.ndarray,
    n_total_dofs: int,
    slave_dof_offset: int,
    master_dof_offset: int,
) -> np.ndarray:
    """Apply the matrix-free contact tangent with the optional C++ backend."""

    _require_backend()
    return _BACKEND.contact_stiffness_matvec(
        np.ascontiguousarray(vector, dtype=np.float64),
        np.ascontiguousarray(scale, dtype=np.float64),
        np.ascontiguousarray(slave_node_ids, dtype=np.int64),
        np.ascontiguousarray(slave_weights, dtype=np.float64),
        np.ascontiguousarray(gradients, dtype=np.float64),
        np.ascontiguousarray(face_node_ids, dtype=np.int64),
        np.ascontiguousarray(grid_weights, dtype=np.float64),
        np.ascontiguousarray(barycentric, dtype=np.float64),
        np.ascontiguousarray(normals, dtype=np.float64),
        int(n_total_dofs),
        int(slave_dof_offset),
        int(master_dof_offset),
    )


def solve_contact_tangent_pcg(
    effective_indptr: np.ndarray,
    effective_indices: np.ndarray,
    effective_data: np.ndarray,
    rhs: np.ndarray,
    free_dofs: np.ndarray,
    scale: np.ndarray,
    slave_node_ids: np.ndarray,
    slave_weights: np.ndarray,
    gradients: np.ndarray,
    face_node_ids: np.ndarray,
    grid_weights: np.ndarray,
    barycentric: np.ndarray,
    normals: np.ndarray,
    n_total_dofs: int,
    slave_dof_offset: int,
    master_dof_offset: int,
    rtol: float,
    atol: float,
    maxiter: int,
    preconditioner: str = "block-sgs",
) -> tuple[np.ndarray, int, int, float]:
    """Solve the matrix-free contact tangent system with C++ PCG."""

    _require_backend()
    preconditioner_mode = {
        "sgs": 0,
        "block-sgs": 1,
        "block_sgs": 1,
    }.get(preconditioner)
    if preconditioner_mode is None:
        raise ValueError("preconditioner must be 'sgs' or 'block-sgs'")
    solution, info, iterations, residual_norm = _BACKEND.solve_contact_tangent_pcg(
        np.ascontiguousarray(effective_indptr, dtype=np.int64),
        np.ascontiguousarray(effective_indices, dtype=np.int64),
        np.ascontiguousarray(effective_data, dtype=np.float64),
        np.ascontiguousarray(rhs, dtype=np.float64),
        np.ascontiguousarray(free_dofs, dtype=np.int64),
        np.ascontiguousarray(scale, dtype=np.float64),
        np.ascontiguousarray(slave_node_ids, dtype=np.int64),
        np.ascontiguousarray(slave_weights, dtype=np.float64),
        np.ascontiguousarray(gradients, dtype=np.float64),
        np.ascontiguousarray(face_node_ids, dtype=np.int64),
        np.ascontiguousarray(grid_weights, dtype=np.float64),
        np.ascontiguousarray(barycentric, dtype=np.float64),
        np.ascontiguousarray(normals, dtype=np.float64),
        int(n_total_dofs),
        int(slave_dof_offset),
        int(master_dof_offset),
        float(rtol),
        float(atol),
        int(maxiter),
        int(preconditioner_mode),
    )
    return np.asarray(solution, dtype=float), int(info), int(iterations), float(residual_norm)
