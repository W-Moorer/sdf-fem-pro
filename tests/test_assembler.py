import numpy as np

from sfc.fem import (
    DeformableBody,
    assemble_gravity_force,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
)
from sfc.fem.tet4 import tet4_volume
from sfc.mesh import VolumeMesh


def _two_tet_body() -> DeformableBody:
    mesh = VolumeMesh(
        X=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        ),
        elements=np.array(
            [
                [0, 1, 2, 3],
                [0, 2, 1, 4],
            ]
        ),
    )
    return DeformableBody(mesh=mesh, material={"E": 1000.0, "nu": 0.25}, density=2.5)


def test_global_stiffness_shape_for_two_tet_mesh() -> None:
    body = _two_tet_body()

    K = assemble_stiffness_matrix(body)

    assert K.shape == (3 * body.n_nodes, 3 * body.n_nodes)


def test_global_mass_shape_for_two_tet_mesh() -> None:
    body = _two_tet_body()

    M = assemble_mass_matrix(body)

    assert M.shape == (3 * body.n_nodes, 3 * body.n_nodes)


def test_total_gravity_force_equals_total_mass_times_g() -> None:
    body = _two_tet_body()
    gravity = np.array([0.0, -9.81, 1.25])
    total_volume = sum(tet4_volume(body.mesh.X[e]) for e in body.mesh.elements)
    total_mass = body.density * total_volume

    F = assemble_gravity_force(body, gravity)
    resultant = F.reshape(-1, 3).sum(axis=0)

    assert np.allclose(resultant, total_mass * gravity)
