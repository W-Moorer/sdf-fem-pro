"""Optional compiled field-contact assembly kernels."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - availability is tested through public wrappers.
    from numba import njit
except Exception as exc:  # pragma: no cover
    njit = None  # type: ignore[assignment]
    _IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


def is_available() -> bool:
    return njit is not None


def _require_numba() -> None:
    if njit is None:
        raise RuntimeError("Numba field-contact backend is not available") from _IMPORT_ERROR


if njit is not None:

    @njit(cache=True)
    def _surface_penalty_response_numba(
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
        n_samples = points.shape[0]
        force = np.zeros(n_total_dofs, dtype=np.float64)
        gaps = np.empty(n_samples, dtype=np.float64)

        for sample in range(n_samples):
            ux = (points[sample, 0] - origin[0]) / spacing[0]
            uy = (points[sample, 1] - origin[1]) / spacing[1]
            uz = (points[sample, 2] - origin[2]) / spacing[2]
            if ux < -1.0e-12 or uy < -1.0e-12 or uz < -1.0e-12:
                raise ValueError("query point is outside the narrow-band grid")
            if ux > float(shape[0] - 1) + 1.0e-12:
                raise ValueError("query point is outside the narrow-band grid")
            if uy > float(shape[1] - 1) + 1.0e-12:
                raise ValueError("query point is outside the narrow-band grid")
            if uz > float(shape[2] - 1) + 1.0e-12:
                raise ValueError("query point is outside the narrow-band grid")

            i0 = int(np.floor(ux))
            j0 = int(np.floor(uy))
            k0 = int(np.floor(uz))
            fx = ux - float(i0)
            fy = uy - float(j0)
            fz = uz - float(k0)
            if i0 >= shape[0] - 1:
                i0 = shape[0] - 2
                fx = 1.0
            if j0 >= shape[1] - 1:
                j0 = shape[1] - 2
                fy = 1.0
            if k0 >= shape[2] - 1:
                k0 = shape[2] - 2
                fz = 1.0
            if i0 < 0:
                i0 = 0
                fx = 0.0
            if j0 < 0:
                j0 = 0
                fy = 0.0
            if k0 < 0:
                k0 = 0
                fz = 0.0

            gap = 0.0
            grad_x = 0.0
            grad_y = 0.0
            grad_z = 0.0
            corner_weight = np.empty(8, dtype=np.float64)
            corner_i = np.empty(8, dtype=np.int64)
            corner_j = np.empty(8, dtype=np.int64)
            corner_k = np.empty(8, dtype=np.int64)
            cid = 0
            for ox in range(2):
                wx = 1.0 - fx
                dwx = -1.0 / spacing[0]
                if ox == 1:
                    wx = fx
                    dwx = 1.0 / spacing[0]
                for oy in range(2):
                    wy = 1.0 - fy
                    dwy = -1.0 / spacing[1]
                    if oy == 1:
                        wy = fy
                        dwy = 1.0 / spacing[1]
                    for oz in range(2):
                        wz = 1.0 - fz
                        dwz = -1.0 / spacing[2]
                        if oz == 1:
                            wz = fz
                            dwz = 1.0 / spacing[2]
                        ii = i0 + ox
                        jj = j0 + oy
                        kk = k0 + oz
                        if not valid_mask[ii, jj, kk]:
                            raise ValueError("query point is outside the valid narrow band")
                        face_id = closest_face_id[ii, jj, kk]
                        if face_id < 0:
                            raise ValueError("query point includes invalid grid-node payload")
                        value = phi[ii, jj, kk]
                        w = wx * wy * wz
                        corner_weight[cid] = w
                        corner_i[cid] = ii
                        corner_j[cid] = jj
                        corner_k[cid] = kk
                        gap += w * value
                        grad_x += value * dwx * wy * wz
                        grad_y += value * wx * dwy * wz
                        grad_z += value * wx * wy * dwz
                        cid += 1

            gaps[sample] = gap
            penetration = -gap
            if penetration <= 0.0:
                continue
            lam = pressure_stiffness * area_weights[sample] * penetration

            for local_node in range(sample_node_ids.shape[1]):
                node = sample_node_ids[sample, local_node]
                sample_w = sample_weights[sample, local_node]
                base = slave_dof_offset + 3 * node
                force[base] += lam * sample_w * grad_x
                force[base + 1] += lam * sample_w * grad_y
                force[base + 2] += lam * sample_w * grad_z

            for corner in range(8):
                ii = corner_i[corner]
                jj = corner_j[corner]
                kk = corner_k[corner]
                grid_w = corner_weight[corner]
                face_id = closest_face_id[ii, jj, kk]
                nx = closest_normal[ii, jj, kk, 0]
                ny = closest_normal[ii, jj, kk, 1]
                nz = closest_normal[ii, jj, kk, 2]
                for face_node in range(3):
                    node = boundary_faces[face_id, face_node]
                    b = barycentric_grid[ii, jj, kk, face_node]
                    base = master_dof_offset + 3 * node
                    scale = -lam * grid_w * b
                    force[base] += scale * nx
                    force[base + 1] += scale * ny
                    force[base + 2] += scale * nz

        return force, gaps


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
    _require_numba()
    return _surface_penalty_response_numba(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(sample_node_ids, dtype=np.int64),
        np.ascontiguousarray(sample_weights, dtype=np.float64),
        np.ascontiguousarray(area_weights, dtype=np.float64),
        np.ascontiguousarray(origin, dtype=np.float64),
        np.ascontiguousarray(spacing, dtype=np.float64),
        np.ascontiguousarray(shape, dtype=np.int64),
        phi,
        valid_mask,
        closest_face_id,
        barycentric_grid,
        closest_normal,
        np.ascontiguousarray(boundary_faces, dtype=np.int64),
        float(pressure_stiffness),
        int(n_total_dofs),
        int(slave_dof_offset),
        int(master_dof_offset),
    )

