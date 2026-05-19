"""Optional compiled projection kernels for SDF field construction."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised through public wrappers when available.
    from numba import njit, prange
except Exception as exc:  # pragma: no cover
    njit = None  # type: ignore[assignment]
    _IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


def is_available() -> bool:
    """Return whether the optional Numba backend can be used."""

    return njit is not None


def _require_numba() -> None:
    if njit is None:
        raise RuntimeError("Numba projection backend is not available") from _IMPORT_ERROR


if njit is not None:

    @njit(inline="always")
    def _project_point_triangle(
        px: float,
        py: float,
        pz: float,
        ax: float,
        ay: float,
        az: float,
        bx: float,
        by: float,
        bz: float,
        cx: float,
        cy: float,
        cz: float,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        abx = bx - ax
        aby = by - ay
        abz = bz - az
        acx = cx - ax
        acy = cy - ay
        acz = cz - az
        apx = px - ax
        apy = py - ay
        apz = pz - az
        d1 = abx * apx + aby * apy + abz * apz
        d2 = acx * apx + acy * apy + acz * apz

        qx = 0.0
        qy = 0.0
        qz = 0.0
        w0 = 0.0
        w1 = 0.0
        w2 = 0.0
        if d1 <= 0.0 and d2 <= 0.0:
            qx = ax
            qy = ay
            qz = az
            w0 = 1.0
        else:
            bpx = px - bx
            bpy = py - by
            bpz = pz - bz
            d3 = abx * bpx + aby * bpy + abz * bpz
            d4 = acx * bpx + acy * bpy + acz * bpz
            if d3 >= 0.0 and d4 <= d3:
                qx = bx
                qy = by
                qz = bz
                w1 = 1.0
            else:
                vc = d1 * d4 - d3 * d2
                if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                    denom = d1 - d3
                    v = 0.0
                    if abs(denom) > 1.0e-30:
                        v = d1 / denom
                    qx = ax + v * abx
                    qy = ay + v * aby
                    qz = az + v * abz
                    w0 = 1.0 - v
                    w1 = v
                else:
                    cpx = px - cx
                    cpy = py - cy
                    cpz = pz - cz
                    d5 = abx * cpx + aby * cpy + abz * cpz
                    d6 = acx * cpx + acy * cpy + acz * cpz
                    if d6 >= 0.0 and d5 <= d6:
                        qx = cx
                        qy = cy
                        qz = cz
                        w2 = 1.0
                    else:
                        vb = d5 * d2 - d1 * d6
                        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                            denom = d2 - d6
                            v = 0.0
                            if abs(denom) > 1.0e-30:
                                v = d2 / denom
                            qx = ax + v * acx
                            qy = ay + v * acy
                            qz = az + v * acz
                            w0 = 1.0 - v
                            w2 = v
                        else:
                            va = d3 * d6 - d5 * d4
                            if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                                denom = (d4 - d3) + (d5 - d6)
                                v = 0.0
                                if abs(denom) > 1.0e-30:
                                    v = (d4 - d3) / denom
                                qx = bx + v * (cx - bx)
                                qy = by + v * (cy - by)
                                qz = bz + v * (cz - bz)
                                w1 = 1.0 - v
                                w2 = v
                            else:
                                denom = va + vb + vc
                                v_face = 0.0
                                w_face = 0.0
                                if abs(denom) > 1.0e-30:
                                    v_face = vb / denom
                                    w_face = vc / denom
                                u_face = 1.0 - v_face - w_face
                                qx = u_face * ax + v_face * bx + w_face * cx
                                qy = u_face * ay + v_face * by + w_face * cy
                                qz = u_face * az + v_face * bz + w_face * cz
                                w0 = u_face
                                w1 = v_face
                                w2 = w_face

        dx = px - qx
        dy = py - qy
        dz = pz - qz
        dist2 = dx * dx + dy * dy + dz * dz
        nx = aby * acz - abz * acy
        ny = abz * acx - abx * acz
        nz = abx * acy - aby * acx
        nn = (nx * nx + ny * ny + nz * nz) ** 0.5
        if nn > 0.0:
            nx = nx / nn
            ny = ny / nn
            nz = nz / nn
        return dist2, qx, qy, qz, w0, w1, w2, nx, ny, nz

    @njit(cache=True, parallel=True)
    def _closest_points_all_faces_numba(
        points: np.ndarray,
        x_current: np.ndarray,
        boundary_faces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_points = points.shape[0]
        n_faces = boundary_faces.shape[0]
        gaps = np.empty(n_points, dtype=np.float64)
        normals = np.empty((n_points, 3), dtype=np.float64)
        face_ids = np.empty(n_points, dtype=np.int64)
        barycentric = np.empty((n_points, 3), dtype=np.float64)
        closest = np.empty((n_points, 3), dtype=np.float64)

        for ip in prange(n_points):
            px = points[ip, 0]
            py = points[ip, 1]
            pz = points[ip, 2]
            best_dist2 = 1.0e300
            best_face = -1
            best_px = 0.0
            best_py = 0.0
            best_pz = 0.0
            best_w0 = 0.0
            best_w1 = 0.0
            best_w2 = 0.0
            best_nx = 0.0
            best_ny = 0.0
            best_nz = 0.0

            for jf in range(n_faces):
                ia = boundary_faces[jf, 0]
                ib = boundary_faces[jf, 1]
                ic = boundary_faces[jf, 2]
                ax = x_current[ia, 0]
                ay = x_current[ia, 1]
                az = x_current[ia, 2]
                bx = x_current[ib, 0]
                by = x_current[ib, 1]
                bz = x_current[ib, 2]
                cx = x_current[ic, 0]
                cy = x_current[ic, 1]
                cz = x_current[ic, 2]

                abx = bx - ax
                aby = by - ay
                abz = bz - az
                acx = cx - ax
                acy = cy - ay
                acz = cz - az
                apx = px - ax
                apy = py - ay
                apz = pz - az
                d1 = abx * apx + aby * apy + abz * apz
                d2 = acx * apx + acy * apy + acz * apz

                qx = 0.0
                qy = 0.0
                qz = 0.0
                w0 = 0.0
                w1 = 0.0
                w2 = 0.0
                if d1 <= 0.0 and d2 <= 0.0:
                    qx = ax
                    qy = ay
                    qz = az
                    w0 = 1.0
                else:
                    bpx = px - bx
                    bpy = py - by
                    bpz = pz - bz
                    d3 = abx * bpx + aby * bpy + abz * bpz
                    d4 = acx * bpx + acy * bpy + acz * bpz
                    if d3 >= 0.0 and d4 <= d3:
                        qx = bx
                        qy = by
                        qz = bz
                        w1 = 1.0
                    else:
                        vc = d1 * d4 - d3 * d2
                        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                            denom = d1 - d3
                            v = 0.0
                            if abs(denom) > 1.0e-30:
                                v = d1 / denom
                            qx = ax + v * abx
                            qy = ay + v * aby
                            qz = az + v * abz
                            w0 = 1.0 - v
                            w1 = v
                        else:
                            cpx = px - cx
                            cpy = py - cy
                            cpz = pz - cz
                            d5 = abx * cpx + aby * cpy + abz * cpz
                            d6 = acx * cpx + acy * cpy + acz * cpz
                            if d6 >= 0.0 and d5 <= d6:
                                qx = cx
                                qy = cy
                                qz = cz
                                w2 = 1.0
                            else:
                                vb = d5 * d2 - d1 * d6
                                if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                                    denom = d2 - d6
                                    v = 0.0
                                    if abs(denom) > 1.0e-30:
                                        v = d2 / denom
                                    qx = ax + v * acx
                                    qy = ay + v * acy
                                    qz = az + v * acz
                                    w0 = 1.0 - v
                                    w2 = v
                                else:
                                    va = d3 * d6 - d5 * d4
                                    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                                        denom = (d4 - d3) + (d5 - d6)
                                        v = 0.0
                                        if abs(denom) > 1.0e-30:
                                            v = (d4 - d3) / denom
                                        qx = bx + v * (cx - bx)
                                        qy = by + v * (cy - by)
                                        qz = bz + v * (cz - bz)
                                        w1 = 1.0 - v
                                        w2 = v
                                    else:
                                        denom = va + vb + vc
                                        v_face = 0.0
                                        w_face = 0.0
                                        if abs(denom) > 1.0e-30:
                                            v_face = vb / denom
                                            w_face = vc / denom
                                        u_face = 1.0 - v_face - w_face
                                        qx = u_face * ax + v_face * bx + w_face * cx
                                        qy = u_face * ay + v_face * by + w_face * cy
                                        qz = u_face * az + v_face * bz + w_face * cz
                                        w0 = u_face
                                        w1 = v_face
                                        w2 = w_face

                dx = px - qx
                dy = py - qy
                dz = pz - qz
                dist2 = dx * dx + dy * dy + dz * dz
                if dist2 < best_dist2:
                    nx = aby * acz - abz * acy
                    ny = abz * acx - abx * acz
                    nz = abx * acy - aby * acx
                    nn = (nx * nx + ny * ny + nz * nz) ** 0.5
                    if nn <= 0.0:
                        continue
                    best_dist2 = dist2
                    best_face = jf
                    best_px = qx
                    best_py = qy
                    best_pz = qz
                    best_w0 = w0
                    best_w1 = w1
                    best_w2 = w2
                    best_nx = nx / nn
                    best_ny = ny / nn
                    best_nz = nz / nn

            dx = px - best_px
            dy = py - best_py
            dz = pz - best_pz
            dist = best_dist2 ** 0.5
            signed_plane_distance = dx * best_nx + dy * best_ny + dz * best_nz
            sign = 1.0
            if signed_plane_distance < 0.0:
                sign = -1.0
            if dist <= 1.0e-15:
                gaps[ip] = 0.0
                normals[ip, 0] = best_nx
                normals[ip, 1] = best_ny
                normals[ip, 2] = best_nz
            else:
                gaps[ip] = sign * dist
                normals[ip, 0] = sign * dx / dist
                normals[ip, 1] = sign * dy / dist
                normals[ip, 2] = sign * dz / dist
            face_ids[ip] = best_face
            barycentric[ip, 0] = best_w0
            barycentric[ip, 1] = best_w1
            barycentric[ip, 2] = best_w2
            closest[ip, 0] = best_px
            closest[ip, 1] = best_py
            closest[ip, 2] = best_pz

        return gaps, normals, face_ids, barycentric, closest


    @njit(cache=True, parallel=True)
    def _closest_points_padded_aabb_numba(
        points: np.ndarray,
        x_current: np.ndarray,
        boundary_faces: np.ndarray,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
        fallback_distance: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_points = points.shape[0]
        n_faces = boundary_faces.shape[0]
        gaps = np.empty(n_points, dtype=np.float64)
        normals = np.empty((n_points, 3), dtype=np.float64)
        face_ids = np.empty(n_points, dtype=np.int64)
        barycentric = np.empty((n_points, 3), dtype=np.float64)
        closest = np.empty((n_points, 3), dtype=np.float64)

        for ip in prange(n_points):
            px = points[ip, 0]
            py = points[ip, 1]
            pz = points[ip, 2]
            best_dist2 = 1.0e300
            best_face = -1
            best_px = 0.0
            best_py = 0.0
            best_pz = 0.0
            best_w0 = 0.0
            best_w1 = 0.0
            best_w2 = 0.0
            best_nx = 0.0
            best_ny = 0.0
            best_nz = 0.0
            fallback_dist2 = fallback_distance * fallback_distance

            for pass_id in range(2):
                for jf in range(n_faces):
                    if pass_id == 0:
                        if px < aabb_min[jf, 0] or px > aabb_max[jf, 0]:
                            continue
                        if py < aabb_min[jf, 1] or py > aabb_max[jf, 1]:
                            continue
                        if pz < aabb_min[jf, 2] or pz > aabb_max[jf, 2]:
                            continue
                    elif best_face != -1 and best_dist2 <= fallback_dist2:
                        continue

                    ia = boundary_faces[jf, 0]
                    ib = boundary_faces[jf, 1]
                    ic = boundary_faces[jf, 2]
                    dist2, qx, qy, qz, w0, w1, w2, nx, ny, nz = _project_point_triangle(
                        px,
                        py,
                        pz,
                        x_current[ia, 0],
                        x_current[ia, 1],
                        x_current[ia, 2],
                        x_current[ib, 0],
                        x_current[ib, 1],
                        x_current[ib, 2],
                        x_current[ic, 0],
                        x_current[ic, 1],
                        x_current[ic, 2],
                    )
                    if nx == 0.0 and ny == 0.0 and nz == 0.0:
                        continue
                    if dist2 < best_dist2:
                        best_dist2 = dist2
                        best_face = jf
                        best_px = qx
                        best_py = qy
                        best_pz = qz
                        best_w0 = w0
                        best_w1 = w1
                        best_w2 = w2
                        best_nx = nx
                        best_ny = ny
                        best_nz = nz

            dx = px - best_px
            dy = py - best_py
            dz = pz - best_pz
            dist = best_dist2 ** 0.5
            signed_plane_distance = dx * best_nx + dy * best_ny + dz * best_nz
            sign = 1.0
            if signed_plane_distance < 0.0:
                sign = -1.0
            if dist <= 1.0e-15:
                gaps[ip] = 0.0
                normals[ip, 0] = best_nx
                normals[ip, 1] = best_ny
                normals[ip, 2] = best_nz
            else:
                gaps[ip] = sign * dist
                normals[ip, 0] = sign * dx / dist
                normals[ip, 1] = sign * dy / dist
                normals[ip, 2] = sign * dz / dist
            face_ids[ip] = best_face
            barycentric[ip, 0] = best_w0
            barycentric[ip, 1] = best_w1
            barycentric[ip, 2] = best_w2
            closest[ip, 0] = best_px
            closest[ip, 1] = best_py
            closest[ip, 2] = best_pz

        return gaps, normals, face_ids, barycentric, closest


def closest_points_all_faces(
    points: np.ndarray,
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate exact all-face closest-feature projection with compiled loops."""

    _require_numba()
    return _closest_points_all_faces_numba(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(x_current, dtype=np.float64),
        np.ascontiguousarray(boundary_faces, dtype=np.int64),
    )


def closest_points_padded_aabb(
    points: np.ndarray,
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    fallback_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate projection using exact padded-AABB candidate screening."""

    _require_numba()
    return _closest_points_padded_aabb_numba(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(x_current, dtype=np.float64),
        np.ascontiguousarray(boundary_faces, dtype=np.int64),
        np.ascontiguousarray(aabb_min, dtype=np.float64),
        np.ascontiguousarray(aabb_max, dtype=np.float64),
        float(fallback_distance),
    )
