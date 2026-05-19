#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

struct Projection {
    double dist2;
    double qx;
    double qy;
    double qz;
    double w0;
    double w1;
    double w2;
    double nx;
    double ny;
    double nz;
};

Projection project_point_triangle(
    double px,
    double py,
    double pz,
    double ax,
    double ay,
    double az,
    double bx,
    double by,
    double bz,
    double cx,
    double cy,
    double cz
) {
    const double abx = bx - ax;
    const double aby = by - ay;
    const double abz = bz - az;
    const double acx = cx - ax;
    const double acy = cy - ay;
    const double acz = cz - az;
    const double apx = px - ax;
    const double apy = py - ay;
    const double apz = pz - az;
    const double d1 = abx * apx + aby * apy + abz * apz;
    const double d2 = acx * apx + acy * apy + acz * apz;

    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double w0 = 0.0;
    double w1 = 0.0;
    double w2 = 0.0;

    if (d1 <= 0.0 && d2 <= 0.0) {
        qx = ax;
        qy = ay;
        qz = az;
        w0 = 1.0;
    } else {
        const double bpx = px - bx;
        const double bpy = py - by;
        const double bpz = pz - bz;
        const double d3 = abx * bpx + aby * bpy + abz * bpz;
        const double d4 = acx * bpx + acy * bpy + acz * bpz;
        if (d3 >= 0.0 && d4 <= d3) {
            qx = bx;
            qy = by;
            qz = bz;
            w1 = 1.0;
        } else {
            const double vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
                const double denom = d1 - d3;
                const double v = std::abs(denom) > 1.0e-30 ? d1 / denom : 0.0;
                qx = ax + v * abx;
                qy = ay + v * aby;
                qz = az + v * abz;
                w0 = 1.0 - v;
                w1 = v;
            } else {
                const double cpx = px - cx;
                const double cpy = py - cy;
                const double cpz = pz - cz;
                const double d5 = abx * cpx + aby * cpy + abz * cpz;
                const double d6 = acx * cpx + acy * cpy + acz * cpz;
                if (d6 >= 0.0 && d5 <= d6) {
                    qx = cx;
                    qy = cy;
                    qz = cz;
                    w2 = 1.0;
                } else {
                    const double vb = d5 * d2 - d1 * d6;
                    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
                        const double denom = d2 - d6;
                        const double v = std::abs(denom) > 1.0e-30 ? d2 / denom : 0.0;
                        qx = ax + v * acx;
                        qy = ay + v * acy;
                        qz = az + v * acz;
                        w0 = 1.0 - v;
                        w2 = v;
                    } else {
                        const double va = d3 * d6 - d5 * d4;
                        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
                            const double denom = (d4 - d3) + (d5 - d6);
                            const double v = std::abs(denom) > 1.0e-30 ? (d4 - d3) / denom : 0.0;
                            qx = bx + v * (cx - bx);
                            qy = by + v * (cy - by);
                            qz = bz + v * (cz - bz);
                            w1 = 1.0 - v;
                            w2 = v;
                        } else {
                            const double denom = va + vb + vc;
                            const double v = std::abs(denom) > 1.0e-30 ? vb / denom : 0.0;
                            const double w = std::abs(denom) > 1.0e-30 ? vc / denom : 0.0;
                            const double u = 1.0 - v - w;
                            qx = u * ax + v * bx + w * cx;
                            qy = u * ay + v * by + w * cy;
                            qz = u * az + v * bz + w * cz;
                            w0 = u;
                            w1 = v;
                            w2 = w;
                        }
                    }
                }
            }
        }
    }

    const double dx = px - qx;
    const double dy = py - qy;
    const double dz = pz - qz;
    const double nx_raw = aby * acz - abz * acy;
    const double ny_raw = abz * acx - abx * acz;
    const double nz_raw = abx * acy - aby * acx;
    const double nn = std::sqrt(nx_raw * nx_raw + ny_raw * ny_raw + nz_raw * nz_raw);
    Projection out;
    out.dist2 = dx * dx + dy * dy + dz * dz;
    out.qx = qx;
    out.qy = qy;
    out.qz = qz;
    out.w0 = w0;
    out.w1 = w1;
    out.w2 = w2;
    out.nx = nn > 0.0 ? nx_raw / nn : 0.0;
    out.ny = nn > 0.0 ? ny_raw / nn : 0.0;
    out.nz = nn > 0.0 ? nz_raw / nn : 0.0;
    return out;
}

void store_best_projection(
    py::ssize_t ip,
    double px,
    double py,
    double pz,
    double best_dist2,
    std::int64_t best_face,
    double best_px,
    double best_py,
    double best_pz,
    double best_w0,
    double best_w1,
    double best_w2,
    double best_nx,
    double best_ny,
    double best_nz,
    py::detail::unchecked_mutable_reference<double, 1> gaps,
    py::detail::unchecked_mutable_reference<double, 2> normals,
    py::detail::unchecked_mutable_reference<std::int64_t, 1> face_ids,
    py::detail::unchecked_mutable_reference<double, 2> barycentric,
    py::detail::unchecked_mutable_reference<double, 2> closest
) {
    const double dx = px - best_px;
    const double dy = py - best_py;
    const double dz = pz - best_pz;
    const double dist = std::sqrt(best_dist2);
    const double signed_plane_distance = dx * best_nx + dy * best_ny + dz * best_nz;
    const double sign = signed_plane_distance < 0.0 ? -1.0 : 1.0;
    if (dist <= 1.0e-15) {
        gaps(ip) = 0.0;
        normals(ip, 0) = best_nx;
        normals(ip, 1) = best_ny;
        normals(ip, 2) = best_nz;
    } else {
        gaps(ip) = sign * dist;
        normals(ip, 0) = sign * dx / dist;
        normals(ip, 1) = sign * dy / dist;
        normals(ip, 2) = sign * dz / dist;
    }
    face_ids(ip) = best_face;
    barycentric(ip, 0) = best_w0;
    barycentric(ip, 1) = best_w1;
    barycentric(ip, 2) = best_w2;
    closest(ip, 0) = best_px;
    closest(ip, 1) = best_py;
    closest(ip, 2) = best_pz;
}

}  // namespace

py::tuple closest_points_all_faces(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_current,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> boundary_faces
) {
    const auto P = points.unchecked<2>();
    const auto X = x_current.unchecked<2>();
    const auto faces = boundary_faces.unchecked<2>();
    const py::ssize_t n_points = P.shape(0);
    const py::ssize_t n_faces = faces.shape(0);

    py::array_t<double> gaps({n_points});
    py::array_t<double> normals({n_points, py::ssize_t(3)});
    py::array_t<std::int64_t> face_ids({n_points});
    py::array_t<double> barycentric({n_points, py::ssize_t(3)});
    py::array_t<double> closest({n_points, py::ssize_t(3)});
    auto g = gaps.mutable_unchecked<1>();
    auto n = normals.mutable_unchecked<2>();
    auto fid = face_ids.mutable_unchecked<1>();
    auto w = barycentric.mutable_unchecked<2>();
    auto cp = closest.mutable_unchecked<2>();

    for (py::ssize_t ip = 0; ip < n_points; ++ip) {
        const double px = P(ip, 0);
        const double py = P(ip, 1);
        const double pz = P(ip, 2);
        double best_dist2 = std::numeric_limits<double>::infinity();
        std::int64_t best_face = -1;
        double best_px = 0.0, best_py = 0.0, best_pz = 0.0;
        double best_w0 = 0.0, best_w1 = 0.0, best_w2 = 0.0;
        double best_nx = 0.0, best_ny = 0.0, best_nz = 0.0;

        for (py::ssize_t jf = 0; jf < n_faces; ++jf) {
            const auto ia = faces(jf, 0);
            const auto ib = faces(jf, 1);
            const auto ic = faces(jf, 2);
            const Projection proj = project_point_triangle(
                px, py, pz,
                X(ia, 0), X(ia, 1), X(ia, 2),
                X(ib, 0), X(ib, 1), X(ib, 2),
                X(ic, 0), X(ic, 1), X(ic, 2)
            );
            if (proj.nx == 0.0 && proj.ny == 0.0 && proj.nz == 0.0) {
                continue;
            }
            if (proj.dist2 < best_dist2) {
                best_dist2 = proj.dist2;
                best_face = static_cast<std::int64_t>(jf);
                best_px = proj.qx;
                best_py = proj.qy;
                best_pz = proj.qz;
                best_w0 = proj.w0;
                best_w1 = proj.w1;
                best_w2 = proj.w2;
                best_nx = proj.nx;
                best_ny = proj.ny;
                best_nz = proj.nz;
            }
        }
        store_best_projection(
            ip, px, py, pz, best_dist2, best_face, best_px, best_py, best_pz,
            best_w0, best_w1, best_w2, best_nx, best_ny, best_nz,
            g, n, fid, w, cp
        );
    }
    return py::make_tuple(gaps, normals, face_ids, barycentric, closest);
}

py::tuple closest_points_padded_aabb(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_current,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> boundary_faces,
    py::array_t<double, py::array::c_style | py::array::forcecast> aabb_min,
    py::array_t<double, py::array::c_style | py::array::forcecast> aabb_max,
    double fallback_distance
) {
    const auto P = points.unchecked<2>();
    const auto X = x_current.unchecked<2>();
    const auto faces = boundary_faces.unchecked<2>();
    const auto amin = aabb_min.unchecked<2>();
    const auto amax = aabb_max.unchecked<2>();
    const py::ssize_t n_points = P.shape(0);
    const py::ssize_t n_faces = faces.shape(0);
    const double fallback_dist2 = fallback_distance * fallback_distance;

    py::array_t<double> gaps({n_points});
    py::array_t<double> normals({n_points, py::ssize_t(3)});
    py::array_t<std::int64_t> face_ids({n_points});
    py::array_t<double> barycentric({n_points, py::ssize_t(3)});
    py::array_t<double> closest({n_points, py::ssize_t(3)});
    auto g = gaps.mutable_unchecked<1>();
    auto n = normals.mutable_unchecked<2>();
    auto fid = face_ids.mutable_unchecked<1>();
    auto w = barycentric.mutable_unchecked<2>();
    auto cp = closest.mutable_unchecked<2>();

    for (py::ssize_t ip = 0; ip < n_points; ++ip) {
        const double px = P(ip, 0);
        const double py = P(ip, 1);
        const double pz = P(ip, 2);
        double best_dist2 = std::numeric_limits<double>::infinity();
        std::int64_t best_face = -1;
        double best_px = 0.0, best_py = 0.0, best_pz = 0.0;
        double best_w0 = 0.0, best_w1 = 0.0, best_w2 = 0.0;
        double best_nx = 0.0, best_ny = 0.0, best_nz = 0.0;

        for (int pass_id = 0; pass_id < 2; ++pass_id) {
            for (py::ssize_t jf = 0; jf < n_faces; ++jf) {
                if (pass_id == 0) {
                    if (px < amin(jf, 0) || px > amax(jf, 0)) {
                        continue;
                    }
                    if (py < amin(jf, 1) || py > amax(jf, 1)) {
                        continue;
                    }
                    if (pz < amin(jf, 2) || pz > amax(jf, 2)) {
                        continue;
                    }
                } else if (best_face != -1 && best_dist2 <= fallback_dist2) {
                    continue;
                }
                const auto ia = faces(jf, 0);
                const auto ib = faces(jf, 1);
                const auto ic = faces(jf, 2);
                const Projection proj = project_point_triangle(
                    px, py, pz,
                    X(ia, 0), X(ia, 1), X(ia, 2),
                    X(ib, 0), X(ib, 1), X(ib, 2),
                    X(ic, 0), X(ic, 1), X(ic, 2)
                );
                if (proj.nx == 0.0 && proj.ny == 0.0 && proj.nz == 0.0) {
                    continue;
                }
                if (proj.dist2 < best_dist2) {
                    best_dist2 = proj.dist2;
                    best_face = static_cast<std::int64_t>(jf);
                    best_px = proj.qx;
                    best_py = proj.qy;
                    best_pz = proj.qz;
                    best_w0 = proj.w0;
                    best_w1 = proj.w1;
                    best_w2 = proj.w2;
                    best_nx = proj.nx;
                    best_ny = proj.ny;
                    best_nz = proj.nz;
                }
            }
        }
        store_best_projection(
            ip, px, py, pz, best_dist2, best_face, best_px, best_py, best_pz,
            best_w0, best_w1, best_w2, best_nx, best_ny, best_nz,
            g, n, fid, w, cp
        );
    }
    return py::make_tuple(gaps, normals, face_ids, barycentric, closest);
}

py::tuple surface_penalty_response(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> sample_node_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> sample_weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> area_weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> origin,
    py::array_t<double, py::array::c_style | py::array::forcecast> spacing,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> shape,
    py::array_t<double, py::array::c_style | py::array::forcecast> phi,
    py::array_t<bool, py::array::c_style | py::array::forcecast> valid_mask,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> closest_face_id,
    py::array_t<double, py::array::c_style | py::array::forcecast> barycentric_grid,
    py::array_t<double, py::array::c_style | py::array::forcecast> closest_normal,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> boundary_faces,
    double pressure_stiffness,
    std::int64_t n_total_dofs,
    std::int64_t slave_dof_offset,
    std::int64_t master_dof_offset
) {
    const auto P = points.unchecked<2>();
    const auto snodes = sample_node_ids.unchecked<2>();
    const auto sweights = sample_weights.unchecked<2>();
    const auto aweights = area_weights.unchecked<1>();
    const auto org = origin.unchecked<1>();
    const auto h = spacing.unchecked<1>();
    const auto shp = shape.unchecked<1>();
    const auto phi_grid = phi.unchecked<3>();
    const auto valid = valid_mask.unchecked<3>();
    const auto face_id = closest_face_id.unchecked<3>();
    const auto bary = barycentric_grid.unchecked<4>();
    const auto normals = closest_normal.unchecked<4>();
    const auto faces = boundary_faces.unchecked<2>();
    const py::ssize_t n_samples = P.shape(0);

    py::array_t<double> force({static_cast<py::ssize_t>(n_total_dofs)});
    py::array_t<double> gaps({n_samples});
    auto f = force.mutable_unchecked<1>();
    auto g = gaps.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < n_total_dofs; ++i) {
        f(i) = 0.0;
    }

    for (py::ssize_t sample = 0; sample < n_samples; ++sample) {
        double ux = (P(sample, 0) - org(0)) / h(0);
        double uy = (P(sample, 1) - org(1)) / h(1);
        double uz = (P(sample, 2) - org(2)) / h(2);
        if (ux < -1.0e-12 || uy < -1.0e-12 || uz < -1.0e-12 ||
            ux > static_cast<double>(shp(0) - 1) + 1.0e-12 ||
            uy > static_cast<double>(shp(1) - 1) + 1.0e-12 ||
            uz > static_cast<double>(shp(2) - 1) + 1.0e-12) {
            throw std::runtime_error("query point is outside the narrow-band grid");
        }
        std::int64_t i0 = static_cast<std::int64_t>(std::floor(ux));
        std::int64_t j0 = static_cast<std::int64_t>(std::floor(uy));
        std::int64_t k0 = static_cast<std::int64_t>(std::floor(uz));
        double fx = ux - static_cast<double>(i0);
        double fy = uy - static_cast<double>(j0);
        double fz = uz - static_cast<double>(k0);
        if (i0 >= shp(0) - 1) {
            i0 = shp(0) - 2;
            fx = 1.0;
        }
        if (j0 >= shp(1) - 1) {
            j0 = shp(1) - 2;
            fy = 1.0;
        }
        if (k0 >= shp(2) - 1) {
            k0 = shp(2) - 2;
            fz = 1.0;
        }
        if (i0 < 0) {
            i0 = 0;
            fx = 0.0;
        }
        if (j0 < 0) {
            j0 = 0;
            fy = 0.0;
        }
        if (k0 < 0) {
            k0 = 0;
            fz = 0.0;
        }

        double gap = 0.0;
        double grad_x = 0.0;
        double grad_y = 0.0;
        double grad_z = 0.0;
        double corner_weight[8];
        std::int64_t corner_i[8];
        std::int64_t corner_j[8];
        std::int64_t corner_k[8];
        int cid = 0;
        for (int ox = 0; ox < 2; ++ox) {
            const double wx = ox == 0 ? 1.0 - fx : fx;
            const double dwx = ox == 0 ? -1.0 / h(0) : 1.0 / h(0);
            for (int oy = 0; oy < 2; ++oy) {
                const double wy = oy == 0 ? 1.0 - fy : fy;
                const double dwy = oy == 0 ? -1.0 / h(1) : 1.0 / h(1);
                for (int oz = 0; oz < 2; ++oz) {
                    const double wz = oz == 0 ? 1.0 - fz : fz;
                    const double dwz = oz == 0 ? -1.0 / h(2) : 1.0 / h(2);
                    const std::int64_t ii = i0 + ox;
                    const std::int64_t jj = j0 + oy;
                    const std::int64_t kk = k0 + oz;
                    if (!valid(ii, jj, kk)) {
                        throw std::runtime_error("query point is outside the valid narrow band");
                    }
                    if (face_id(ii, jj, kk) < 0) {
                        throw std::runtime_error("query point includes invalid grid-node payload");
                    }
                    const double value = phi_grid(ii, jj, kk);
                    const double weight = wx * wy * wz;
                    corner_weight[cid] = weight;
                    corner_i[cid] = ii;
                    corner_j[cid] = jj;
                    corner_k[cid] = kk;
                    gap += weight * value;
                    grad_x += value * dwx * wy * wz;
                    grad_y += value * wx * dwy * wz;
                    grad_z += value * wx * wy * dwz;
                    ++cid;
                }
            }
        }

        g(sample) = gap;
        const double penetration = -gap;
        if (penetration <= 0.0) {
            continue;
        }
        const double lambda = pressure_stiffness * aweights(sample) * penetration;

        for (py::ssize_t local_node = 0; local_node < snodes.shape(1); ++local_node) {
            const std::int64_t node = snodes(sample, local_node);
            const double sw = sweights(sample, local_node);
            const std::int64_t base = slave_dof_offset + 3 * node;
            f(base) += lambda * sw * grad_x;
            f(base + 1) += lambda * sw * grad_y;
            f(base + 2) += lambda * sw * grad_z;
        }
        for (int corner = 0; corner < 8; ++corner) {
            const std::int64_t ii = corner_i[corner];
            const std::int64_t jj = corner_j[corner];
            const std::int64_t kk = corner_k[corner];
            const double grid_w = corner_weight[corner];
            const std::int64_t fid = face_id(ii, jj, kk);
            const double nx = normals(ii, jj, kk, 0);
            const double ny = normals(ii, jj, kk, 1);
            const double nz = normals(ii, jj, kk, 2);
            for (int face_node = 0; face_node < 3; ++face_node) {
                const std::int64_t node = faces(fid, face_node);
                const double b = bary(ii, jj, kk, face_node);
                const std::int64_t base = master_dof_offset + 3 * node;
                const double scale = -lambda * grid_w * b;
                f(base) += scale * nx;
                f(base + 1) += scale * ny;
                f(base + 2) += scale * nz;
            }
        }
    }
    return py::make_tuple(force, gaps);
}

py::array_t<double> contact_stiffness_matvec(
    py::array_t<double, py::array::c_style | py::array::forcecast> vector,
    py::array_t<double, py::array::c_style | py::array::forcecast> scale,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> slave_node_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> slave_weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> gradients,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> face_node_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> grid_weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> barycentric,
    py::array_t<double, py::array::c_style | py::array::forcecast> normals,
    std::int64_t n_total_dofs,
    std::int64_t slave_dof_offset,
    std::int64_t master_dof_offset
) {
    const auto x = vector.unchecked<1>();
    const auto row_scale = scale.unchecked<1>();
    const auto snodes = slave_node_ids.unchecked<2>();
    const auto sweights = slave_weights.unchecked<2>();
    const auto grads = gradients.unchecked<2>();
    const auto fnodes = face_node_ids.unchecked<3>();
    const auto gweights = grid_weights.unchecked<2>();
    const auto bary = barycentric.unchecked<3>();
    const auto nrm = normals.unchecked<3>();
    const py::ssize_t n_active = row_scale.shape(0);
    if (x.shape(0) != static_cast<py::ssize_t>(n_total_dofs)) {
        throw std::runtime_error("vector size does not match n_total_dofs");
    }

    py::array_t<double> out({static_cast<py::ssize_t>(n_total_dofs)});
    auto y = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < n_total_dofs; ++i) {
        y(i) = 0.0;
    }

    for (py::ssize_t row = 0; row < n_active; ++row) {
        double jx = 0.0;
        for (py::ssize_t local_node = 0; local_node < snodes.shape(1); ++local_node) {
            const std::int64_t node = snodes(row, local_node);
            const double sw = sweights(row, local_node);
            const std::int64_t base = slave_dof_offset + 3 * node;
            jx += sw * (
                grads(row, 0) * x(base) +
                grads(row, 1) * x(base + 1) +
                grads(row, 2) * x(base + 2)
            );
        }
        for (py::ssize_t corner = 0; corner < fnodes.shape(1); ++corner) {
            const double gw = gweights(row, corner);
            for (py::ssize_t face_node = 0; face_node < fnodes.shape(2); ++face_node) {
                const std::int64_t node = fnodes(row, corner, face_node);
                const std::int64_t base = master_dof_offset + 3 * node;
                const double n_dot_x =
                    nrm(row, corner, 0) * x(base) +
                    nrm(row, corner, 1) * x(base + 1) +
                    nrm(row, corner, 2) * x(base + 2);
                jx -= gw * bary(row, corner, face_node) * n_dot_x;
            }
        }

        const double scaled = row_scale(row) * jx;
        for (py::ssize_t local_node = 0; local_node < snodes.shape(1); ++local_node) {
            const std::int64_t node = snodes(row, local_node);
            const double sw = sweights(row, local_node);
            const std::int64_t base = slave_dof_offset + 3 * node;
            y(base) += scaled * sw * grads(row, 0);
            y(base + 1) += scaled * sw * grads(row, 1);
            y(base + 2) += scaled * sw * grads(row, 2);
        }
        for (py::ssize_t corner = 0; corner < fnodes.shape(1); ++corner) {
            const double gw = gweights(row, corner);
            for (py::ssize_t face_node = 0; face_node < fnodes.shape(2); ++face_node) {
                const std::int64_t node = fnodes(row, corner, face_node);
                const std::int64_t base = master_dof_offset + 3 * node;
                const double coeff = -scaled * gw * bary(row, corner, face_node);
                y(base) += coeff * nrm(row, corner, 0);
                y(base + 1) += coeff * nrm(row, corner, 1);
                y(base + 2) += coeff * nrm(row, corner, 2);
            }
        }
    }
    return out;
}

py::tuple solve_contact_tangent_pcg(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> effective_indptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> effective_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> effective_data,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> free_dofs,
    py::array_t<double, py::array::c_style | py::array::forcecast> scale,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> slave_node_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> slave_weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> gradients,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> face_node_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> grid_weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> barycentric,
    py::array_t<double, py::array::c_style | py::array::forcecast> normals,
    std::int64_t n_total_dofs,
    std::int64_t slave_dof_offset,
    std::int64_t master_dof_offset,
    double rtol,
    double atol,
    std::int64_t maxiter,
    std::int64_t preconditioner_mode
) {
    const auto indptr = effective_indptr.unchecked<1>();
    const auto indices = effective_indices.unchecked<1>();
    const auto data = effective_data.unchecked<1>();
    const auto b = rhs.unchecked<1>();
    const auto free = free_dofs.unchecked<1>();
    const auto row_scale = scale.unchecked<1>();
    const auto snodes = slave_node_ids.unchecked<2>();
    const auto sweights = slave_weights.unchecked<2>();
    const auto grads = gradients.unchecked<2>();
    const auto fnodes = face_node_ids.unchecked<3>();
    const auto gweights = grid_weights.unchecked<2>();
    const auto bary = barycentric.unchecked<3>();
    const auto nrm = normals.unchecked<3>();
    const py::ssize_t n = b.shape(0);
    const py::ssize_t n_active = row_scale.shape(0);
    if (free.shape(0) != n) {
        throw std::runtime_error("free_dofs size must match rhs size");
    }
    if (indptr.shape(0) != n + 1) {
        throw std::runtime_error("effective CSR indptr size must be n + 1");
    }

    py::array_t<double> solution({n});
    auto x_out = solution.mutable_unchecked<1>();

    std::vector<std::int64_t> global_to_free(static_cast<std::size_t>(n_total_dofs), -1);
    for (py::ssize_t i = 0; i < n; ++i) {
        const auto gdof = free(i);
        if (gdof < 0 || gdof >= n_total_dofs) {
            throw std::runtime_error("free_dofs contains an invalid global dof");
        }
        global_to_free[static_cast<std::size_t>(gdof)] = static_cast<std::int64_t>(i);
        x_out(i) = 0.0;
    }

    std::vector<double> diag(static_cast<std::size_t>(n), 0.0);
    for (py::ssize_t i = 0; i < n; ++i) {
        for (std::int64_t ptr = indptr(i); ptr < indptr(i + 1); ++ptr) {
            if (indices(ptr) == i) {
                diag[static_cast<std::size_t>(i)] += data(ptr);
                break;
            }
        }
    }
    auto add_contact_diag = [&](std::int64_t gdof, double value) {
        if (gdof < 0 || gdof >= n_total_dofs) {
            return;
        }
        const std::int64_t local = global_to_free[static_cast<std::size_t>(gdof)];
        if (local >= 0) {
            diag[static_cast<std::size_t>(local)] += value;
        }
    };
    for (py::ssize_t row = 0; row < n_active; ++row) {
        const double s = row_scale(row);
        for (py::ssize_t local_node = 0; local_node < snodes.shape(1); ++local_node) {
            const std::int64_t node = snodes(row, local_node);
            const double sw = sweights(row, local_node);
            for (int comp = 0; comp < 3; ++comp) {
                const double entry = sw * grads(row, comp);
                add_contact_diag(slave_dof_offset + 3 * node + comp, s * entry * entry);
            }
        }
        for (py::ssize_t corner = 0; corner < fnodes.shape(1); ++corner) {
            const double gw = gweights(row, corner);
            for (py::ssize_t face_node = 0; face_node < fnodes.shape(2); ++face_node) {
                const std::int64_t node = fnodes(row, corner, face_node);
                const double bval = bary(row, corner, face_node);
                for (int comp = 0; comp < 3; ++comp) {
                    const double entry = -gw * bval * nrm(row, corner, comp);
                    add_contact_diag(master_dof_offset + 3 * node + comp, s * entry * entry);
                }
            }
        }
    }
    for (py::ssize_t i = 0; i < n; ++i) {
        if (std::abs(diag[static_cast<std::size_t>(i)]) <= 1.0e-30) {
            diag[static_cast<std::size_t>(i)] = 1.0;
        }
    }

    auto invert_3x3 = [](const double* block, double* inv) {
        double aug[3][6] = {
            {block[0], block[1], block[2], 1.0, 0.0, 0.0},
            {block[3], block[4], block[5], 0.0, 1.0, 0.0},
            {block[6], block[7], block[8], 0.0, 0.0, 1.0},
        };
        bool ok = true;
        for (int col = 0; col < 3; ++col) {
            int pivot = col;
            double pivot_abs = std::abs(aug[col][col]);
            for (int row = col + 1; row < 3; ++row) {
                const double candidate = std::abs(aug[row][col]);
                if (candidate > pivot_abs) {
                    pivot = row;
                    pivot_abs = candidate;
                }
            }
            if (pivot_abs <= 1.0e-24) {
                ok = false;
                break;
            }
            if (pivot != col) {
                for (int k = 0; k < 6; ++k) {
                    std::swap(aug[col][k], aug[pivot][k]);
                }
            }
            const double scale_value = aug[col][col];
            for (int k = 0; k < 6; ++k) {
                aug[col][k] /= scale_value;
            }
            for (int row = 0; row < 3; ++row) {
                if (row == col) {
                    continue;
                }
                const double factor = aug[row][col];
                for (int k = 0; k < 6; ++k) {
                    aug[row][k] -= factor * aug[col][k];
                }
            }
        }
        if (ok) {
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    inv[3 * row + col] = aug[row][3 + col];
                }
            }
            return;
        }
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                inv[3 * row + col] = 0.0;
            }
            const double diagonal = std::abs(block[3 * row + row]) > 1.0e-24
                ? block[3 * row + row]
                : 1.0;
            inv[3 * row + row] = 1.0 / diagonal;
        }
    };

    std::vector<std::int64_t> local_to_block(static_cast<std::size_t>(n), -1);
    std::vector<int> local_component(static_cast<std::size_t>(n), 0);
    std::vector<std::int64_t> block_comp_to_local;
    std::vector<char> block_comp_present;
    std::unordered_map<std::int64_t, std::int64_t> block_lookup;
    block_lookup.reserve(static_cast<std::size_t>(n));
    auto block_key_for_gdof = [&](std::int64_t gdof) -> std::int64_t {
        if (gdof >= master_dof_offset) {
            return master_dof_offset + 3 * ((gdof - master_dof_offset) / 3);
        }
        if (gdof >= slave_dof_offset) {
            return slave_dof_offset + 3 * ((gdof - slave_dof_offset) / 3);
        }
        return gdof;
    };
    for (py::ssize_t i = 0; i < n; ++i) {
        const std::int64_t gdof = free(i);
        const std::int64_t key = block_key_for_gdof(gdof);
        auto it = block_lookup.find(key);
        if (it == block_lookup.end()) {
            const std::int64_t block_id = static_cast<std::int64_t>(block_lookup.size());
            it = block_lookup.emplace(key, block_id).first;
            block_comp_to_local.push_back(-1);
            block_comp_to_local.push_back(-1);
            block_comp_to_local.push_back(-1);
            block_comp_present.push_back(0);
            block_comp_present.push_back(0);
            block_comp_present.push_back(0);
        }
        const std::int64_t block_id = it->second;
        int component = static_cast<int>(gdof - key);
        if (component < 0 || component > 2) {
            component = 0;
        }
        local_to_block[static_cast<std::size_t>(i)] = block_id;
        local_component[static_cast<std::size_t>(i)] = component;
        block_comp_to_local[static_cast<std::size_t>(3 * block_id + component)] = static_cast<std::int64_t>(i);
        block_comp_present[static_cast<std::size_t>(3 * block_id + component)] = 1;
    }
    const std::int64_t n_blocks = static_cast<std::int64_t>(block_lookup.size());
    std::vector<double> block_diag(static_cast<std::size_t>(9 * n_blocks), 0.0);
    for (py::ssize_t i = 0; i < n; ++i) {
        const std::int64_t rb = local_to_block[static_cast<std::size_t>(i)];
        const int rc = local_component[static_cast<std::size_t>(i)];
        for (std::int64_t ptr = indptr(i); ptr < indptr(i + 1); ++ptr) {
            const std::int64_t col = indices(ptr);
            const std::int64_t cb = local_to_block[static_cast<std::size_t>(col)];
            if (cb == rb) {
                const int cc = local_component[static_cast<std::size_t>(col)];
                block_diag[static_cast<std::size_t>(9 * rb + 3 * rc + cc)] += data(ptr);
            }
        }
    }
    std::vector<std::int64_t> row_blocks;
    std::vector<double> row_vectors;
    row_blocks.reserve(32);
    row_vectors.reserve(96);
    auto add_row_contribution = [&](std::int64_t gdof, int component, double value) {
        if (gdof < 0 || gdof >= n_total_dofs) {
            return;
        }
        const std::int64_t local = global_to_free[static_cast<std::size_t>(gdof)];
        if (local < 0) {
            return;
        }
        const std::int64_t block_id = local_to_block[static_cast<std::size_t>(local)];
        const int comp = local_component[static_cast<std::size_t>(local)];
        std::size_t idx = 0;
        for (; idx < row_blocks.size(); ++idx) {
            if (row_blocks[idx] == block_id) {
                break;
            }
        }
        if (idx == row_blocks.size()) {
            row_blocks.push_back(block_id);
            row_vectors.push_back(0.0);
            row_vectors.push_back(0.0);
            row_vectors.push_back(0.0);
        }
        row_vectors[3 * idx + static_cast<std::size_t>(comp)] += value;
    };
    for (py::ssize_t row = 0; row < n_active; ++row) {
        row_blocks.clear();
        row_vectors.clear();
        for (py::ssize_t local_node = 0; local_node < snodes.shape(1); ++local_node) {
            const std::int64_t node = snodes(row, local_node);
            const double sw = sweights(row, local_node);
            for (int comp = 0; comp < 3; ++comp) {
                add_row_contribution(
                    slave_dof_offset + 3 * node + comp,
                    comp,
                    sw * grads(row, comp)
                );
            }
        }
        for (py::ssize_t corner = 0; corner < fnodes.shape(1); ++corner) {
            const double gw = gweights(row, corner);
            for (py::ssize_t face_node = 0; face_node < fnodes.shape(2); ++face_node) {
                const std::int64_t node = fnodes(row, corner, face_node);
                const double bval = bary(row, corner, face_node);
                for (int comp = 0; comp < 3; ++comp) {
                    add_row_contribution(
                        master_dof_offset + 3 * node + comp,
                        comp,
                        -gw * bval * nrm(row, corner, comp)
                    );
                }
            }
        }
        const double s = row_scale(row);
        for (std::size_t idx = 0; idx < row_blocks.size(); ++idx) {
            const std::int64_t block_id = row_blocks[idx];
            for (int a = 0; a < 3; ++a) {
                for (int bcomp = 0; bcomp < 3; ++bcomp) {
                    block_diag[static_cast<std::size_t>(9 * block_id + 3 * a + bcomp)] +=
                        s * row_vectors[3 * idx + static_cast<std::size_t>(a)] *
                        row_vectors[3 * idx + static_cast<std::size_t>(bcomp)];
                }
            }
        }
    }
    for (std::int64_t block_id = 0; block_id < n_blocks; ++block_id) {
        for (int comp = 0; comp < 3; ++comp) {
            const std::size_t marker = static_cast<std::size_t>(3 * block_id + comp);
            if (!block_comp_present[marker]) {
                block_diag[static_cast<std::size_t>(9 * block_id + 3 * comp + comp)] = 1.0;
            }
        }
    }
    std::vector<double> block_inv(static_cast<std::size_t>(9 * n_blocks), 0.0);
    for (std::int64_t block_id = 0; block_id < n_blocks; ++block_id) {
        invert_3x3(
            &block_diag[static_cast<std::size_t>(9 * block_id)],
            &block_inv[static_cast<std::size_t>(9 * block_id)]
        );
    }

    auto dot = [](const std::vector<double>& a, const std::vector<double>& bvec) -> double {
        double out = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            out += a[i] * bvec[i];
        }
        return out;
    };

    auto apply_operator = [&](const std::vector<double>& x, std::vector<double>& y) {
        std::fill(y.begin(), y.end(), 0.0);
        for (py::ssize_t i = 0; i < n; ++i) {
            double value = 0.0;
            for (std::int64_t ptr = indptr(i); ptr < indptr(i + 1); ++ptr) {
                value += data(ptr) * x[static_cast<std::size_t>(indices(ptr))];
            }
            y[static_cast<std::size_t>(i)] = value;
        }
        for (py::ssize_t row = 0; row < n_active; ++row) {
            double jx = 0.0;
            for (py::ssize_t local_node = 0; local_node < snodes.shape(1); ++local_node) {
                const std::int64_t node = snodes(row, local_node);
                const double sw = sweights(row, local_node);
                for (int comp = 0; comp < 3; ++comp) {
                    const std::int64_t gdof = slave_dof_offset + 3 * node + comp;
                    const std::int64_t local = global_to_free[static_cast<std::size_t>(gdof)];
                    if (local >= 0) {
                        jx += sw * grads(row, comp) * x[static_cast<std::size_t>(local)];
                    }
                }
            }
            for (py::ssize_t corner = 0; corner < fnodes.shape(1); ++corner) {
                const double gw = gweights(row, corner);
                for (py::ssize_t face_node = 0; face_node < fnodes.shape(2); ++face_node) {
                    const std::int64_t node = fnodes(row, corner, face_node);
                    const double bval = bary(row, corner, face_node);
                    for (int comp = 0; comp < 3; ++comp) {
                        const std::int64_t gdof = master_dof_offset + 3 * node + comp;
                        const std::int64_t local = global_to_free[static_cast<std::size_t>(gdof)];
                        if (local >= 0) {
                            jx -= gw * bval * nrm(row, corner, comp) * x[static_cast<std::size_t>(local)];
                        }
                    }
                }
            }
            const double scaled = row_scale(row) * jx;
            for (py::ssize_t local_node = 0; local_node < snodes.shape(1); ++local_node) {
                const std::int64_t node = snodes(row, local_node);
                const double sw = sweights(row, local_node);
                for (int comp = 0; comp < 3; ++comp) {
                    const std::int64_t gdof = slave_dof_offset + 3 * node + comp;
                    const std::int64_t local = global_to_free[static_cast<std::size_t>(gdof)];
                    if (local >= 0) {
                        y[static_cast<std::size_t>(local)] += scaled * sw * grads(row, comp);
                    }
                }
            }
            for (py::ssize_t corner = 0; corner < fnodes.shape(1); ++corner) {
                const double gw = gweights(row, corner);
                for (py::ssize_t face_node = 0; face_node < fnodes.shape(2); ++face_node) {
                    const std::int64_t node = fnodes(row, corner, face_node);
                    const double bval = bary(row, corner, face_node);
                    for (int comp = 0; comp < 3; ++comp) {
                        const std::int64_t gdof = master_dof_offset + 3 * node + comp;
                        const std::int64_t local = global_to_free[static_cast<std::size_t>(gdof)];
                        if (local >= 0) {
                            y[static_cast<std::size_t>(local)] += -scaled * gw * bval * nrm(row, corner, comp);
                        }
                    }
                }
            }
        }
    };

    std::vector<double> precond_forward(static_cast<std::size_t>(n), 0.0);
    std::vector<double> precond_scaled(static_cast<std::size_t>(n), 0.0);
    std::vector<double> block_forward(static_cast<std::size_t>(n), 0.0);
    std::vector<double> block_scaled(static_cast<std::size_t>(n), 0.0);
    auto solve_block = [&](std::int64_t block_id, const double* rhs3, double* out3) {
        const double* inv = &block_inv[static_cast<std::size_t>(9 * block_id)];
        for (int row = 0; row < 3; ++row) {
            out3[row] =
                inv[3 * row + 0] * rhs3[0] +
                inv[3 * row + 1] * rhs3[1] +
                inv[3 * row + 2] * rhs3[2];
        }
    };
    auto apply_block_diag = [&](std::int64_t block_id, const double* value3, double* out3) {
        const double* block = &block_diag[static_cast<std::size_t>(9 * block_id)];
        for (int row = 0; row < 3; ++row) {
            out3[row] =
                block[3 * row + 0] * value3[0] +
                block[3 * row + 1] * value3[1] +
                block[3 * row + 2] * value3[2];
        }
    };
    auto apply_scalar_sgs_preconditioner = [&](const std::vector<double>& residual, std::vector<double>& out) {
        for (py::ssize_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (std::int64_t ptr = indptr(i); ptr < indptr(i + 1); ++ptr) {
                const std::int64_t col = indices(ptr);
                if (col < i) {
                    sum += data(ptr) * precond_forward[static_cast<std::size_t>(col)];
                }
            }
            precond_forward[static_cast<std::size_t>(i)] =
                (residual[static_cast<std::size_t>(i)] - sum) / diag[static_cast<std::size_t>(i)];
        }
        for (py::ssize_t i = 0; i < n; ++i) {
            precond_scaled[static_cast<std::size_t>(i)] =
                diag[static_cast<std::size_t>(i)] * precond_forward[static_cast<std::size_t>(i)];
            out[static_cast<std::size_t>(i)] = 0.0;
        }
        for (py::ssize_t ii = 0; ii < n; ++ii) {
            const py::ssize_t i = n - 1 - ii;
            double sum = 0.0;
            for (std::int64_t ptr = indptr(i); ptr < indptr(i + 1); ++ptr) {
                const std::int64_t col = indices(ptr);
                if (col > i) {
                    sum += data(ptr) * out[static_cast<std::size_t>(col)];
                }
            }
            out[static_cast<std::size_t>(i)] =
                (precond_scaled[static_cast<std::size_t>(i)] - sum) / diag[static_cast<std::size_t>(i)];
        }
    };
    auto apply_block_sgs_preconditioner = [&](const std::vector<double>& residual, std::vector<double>& out) {
        std::fill(block_forward.begin(), block_forward.end(), 0.0);
        std::fill(block_scaled.begin(), block_scaled.end(), 0.0);
        std::fill(out.begin(), out.end(), 0.0);
        for (std::int64_t block_id = 0; block_id < n_blocks; ++block_id) {
            double rhs3[3] = {0.0, 0.0, 0.0};
            double solved3[3] = {0.0, 0.0, 0.0};
            for (int comp = 0; comp < 3; ++comp) {
                const std::int64_t local = block_comp_to_local[static_cast<std::size_t>(3 * block_id + comp)];
                if (local < 0) {
                    continue;
                }
                double value = residual[static_cast<std::size_t>(local)];
                for (std::int64_t ptr = indptr(local); ptr < indptr(local + 1); ++ptr) {
                    const std::int64_t col = indices(ptr);
                    const std::int64_t col_block = local_to_block[static_cast<std::size_t>(col)];
                    if (col_block < block_id) {
                        value -= data(ptr) * block_forward[static_cast<std::size_t>(col)];
                    }
                }
                rhs3[comp] = value;
            }
            solve_block(block_id, rhs3, solved3);
            for (int comp = 0; comp < 3; ++comp) {
                const std::int64_t local = block_comp_to_local[static_cast<std::size_t>(3 * block_id + comp)];
                if (local >= 0) {
                    block_forward[static_cast<std::size_t>(local)] = solved3[comp];
                }
            }
        }
        for (std::int64_t block_id = 0; block_id < n_blocks; ++block_id) {
            double value3[3] = {0.0, 0.0, 0.0};
            double product3[3] = {0.0, 0.0, 0.0};
            for (int comp = 0; comp < 3; ++comp) {
                const std::int64_t local = block_comp_to_local[static_cast<std::size_t>(3 * block_id + comp)];
                if (local >= 0) {
                    value3[comp] = block_forward[static_cast<std::size_t>(local)];
                }
            }
            apply_block_diag(block_id, value3, product3);
            for (int comp = 0; comp < 3; ++comp) {
                const std::int64_t local = block_comp_to_local[static_cast<std::size_t>(3 * block_id + comp)];
                if (local >= 0) {
                    block_scaled[static_cast<std::size_t>(local)] = product3[comp];
                }
            }
        }
        for (std::int64_t ib = 0; ib < n_blocks; ++ib) {
            const std::int64_t block_id = n_blocks - 1 - ib;
            double rhs3[3] = {0.0, 0.0, 0.0};
            double solved3[3] = {0.0, 0.0, 0.0};
            for (int comp = 0; comp < 3; ++comp) {
                const std::int64_t local = block_comp_to_local[static_cast<std::size_t>(3 * block_id + comp)];
                if (local < 0) {
                    continue;
                }
                double value = block_scaled[static_cast<std::size_t>(local)];
                for (std::int64_t ptr = indptr(local); ptr < indptr(local + 1); ++ptr) {
                    const std::int64_t col = indices(ptr);
                    const std::int64_t col_block = local_to_block[static_cast<std::size_t>(col)];
                    if (col_block > block_id) {
                        value -= data(ptr) * out[static_cast<std::size_t>(col)];
                    }
                }
                rhs3[comp] = value;
            }
            solve_block(block_id, rhs3, solved3);
            for (int comp = 0; comp < 3; ++comp) {
                const std::int64_t local = block_comp_to_local[static_cast<std::size_t>(3 * block_id + comp)];
                if (local >= 0) {
                    out[static_cast<std::size_t>(local)] = solved3[comp];
                }
            }
        }
    };
    auto apply_preconditioner = [&](const std::vector<double>& residual, std::vector<double>& out) {
        if (preconditioner_mode == 1) {
            apply_block_sgs_preconditioner(residual, out);
        } else {
            apply_scalar_sgs_preconditioner(residual, out);
        }
    };

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    std::vector<double> r(static_cast<std::size_t>(n), 0.0);
    std::vector<double> z(static_cast<std::size_t>(n), 0.0);
    std::vector<double> p(static_cast<std::size_t>(n), 0.0);
    std::vector<double> ap(static_cast<std::size_t>(n), 0.0);
    double rhs_norm2 = 0.0;
    for (py::ssize_t i = 0; i < n; ++i) {
        r[static_cast<std::size_t>(i)] = b(i);
        rhs_norm2 += b(i) * b(i);
    }
    apply_preconditioner(r, z);
    for (py::ssize_t i = 0; i < n; ++i) {
        p[static_cast<std::size_t>(i)] = z[static_cast<std::size_t>(i)];
    }
    const double rhs_norm = std::sqrt(rhs_norm2);
    const double tolerance = std::max(std::abs(atol), std::abs(rtol) * rhs_norm);
    double residual_norm = rhs_norm;
    double rz_old = dot(r, z);
    std::int64_t info = 0;
    std::int64_t iterations = 0;
    if (residual_norm > tolerance && std::abs(rz_old) > 1.0e-300) {
        for (iterations = 1; iterations <= maxiter; ++iterations) {
            apply_operator(p, ap);
            const double denom = dot(p, ap);
            if (std::abs(denom) <= 1.0e-300) {
                info = 2;
                break;
            }
            const double alpha = rz_old / denom;
            double res2 = 0.0;
            for (py::ssize_t i = 0; i < n; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i);
                x[idx] += alpha * p[idx];
                r[idx] -= alpha * ap[idx];
                res2 += r[idx] * r[idx];
            }
            residual_norm = std::sqrt(res2);
            if (residual_norm <= tolerance) {
                info = 0;
                break;
            }
            apply_preconditioner(r, z);
            const double rz_new = dot(r, z);
            if (std::abs(rz_old) <= 1.0e-300) {
                info = 3;
                break;
            }
            const double beta = rz_new / rz_old;
            for (py::ssize_t i = 0; i < n; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i);
                p[idx] = z[idx] + beta * p[idx];
            }
            rz_old = rz_new;
        }
        if (iterations > maxiter && residual_norm > tolerance) {
            iterations = maxiter;
            info = 1;
        }
    }
    for (py::ssize_t i = 0; i < n; ++i) {
        x_out(i) = x[static_cast<std::size_t>(i)];
    }
    return py::make_tuple(solution, info, iterations, residual_norm);
}

PYBIND11_MODULE(_sfc_cpp, m) {
    m.doc() = "C++ fused SDF field-population and field-contact kernels";
    m.def("closest_points_all_faces", &closest_points_all_faces);
    m.def("closest_points_padded_aabb", &closest_points_padded_aabb);
    m.def("surface_penalty_response", &surface_penalty_response);
    m.def("contact_stiffness_matvec", &contact_stiffness_matvec);
    m.def("solve_contact_tangent_pcg", &solve_contact_tangent_pcg);
}
