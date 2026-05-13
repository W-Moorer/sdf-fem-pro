"""Minimal two-tetrahedron dynamic patch example."""

from __future__ import annotations

import numpy as np

from sfc.fem import (
    DeformableBody,
    assemble_gravity_force,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    fixed_dofs_from_node_set,
    newmark_beta_step,
)
from sfc.mesh import VolumeMesh


def build_body() -> DeformableBody:
    mesh = VolumeMesh(
        X=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        ),
        elements=np.array(
            [
                [0, 1, 2, 3],
                [0, 2, 1, 4],
            ],
            dtype=np.int64,
        ),
        node_sets={"fixed": np.array([0], dtype=np.int64)},
    )
    return DeformableBody(mesh=mesh, material={"E": 1.0e5, "nu": 0.3}, density=10.0)


def main() -> None:
    body = build_body()
    M = assemble_mass_matrix(body, kind="consistent")
    K = assemble_stiffness_matrix(body)
    C = None
    fixed = fixed_dofs_from_node_set(body.mesh.node_sets["fixed"], "xyz")

    dt = 1.0e-3
    f_gravity = assemble_gravity_force(body)
    for _ in range(10):
        body.u, body.v, body.a = newmark_beta_step(
            M,
            C,
            K,
            body.u,
            body.v,
            body.a,
            f_gravity,
            dt=dt,
            fixed_dofs=fixed,
        )

    print("Final z displacements:", body.u[2::3])


if __name__ == "__main__":
    main()
