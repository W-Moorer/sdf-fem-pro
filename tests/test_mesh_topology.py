import numpy as np
import pytest

from sfc.mesh.topology import VolumeMesh, signed_tet4_jacobian_determinants


def test_volume_mesh_accepts_tet4_arrays() -> None:
    mesh = VolumeMesh(
        X=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        elements=np.array([[0, 1, 2, 3]]),
        node_sets={"fixed": [0, 1]},
        face_sets={"contact": [0]},
    )

    assert mesh.X.shape == (4, 3)
    assert mesh.elements.shape == (1, 4)
    assert mesh.element_type == "tet4"
    assert mesh.node_sets["fixed"].dtype == np.int64
    assert mesh.face_sets["contact"].dtype == np.int64


def test_volume_mesh_rejects_non_tet4_element_type() -> None:
    with pytest.raises(ValueError, match="tet4.*hex8"):
        VolumeMesh(
            X=np.zeros((4, 3)),
            elements=np.array([[0, 1, 2, 3]]),
            element_type="wedge6",
        )


def test_volume_mesh_accepts_hex8_for_surface_sdf_geometry() -> None:
    mesh = VolumeMesh(
        X=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ),
        elements=np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
        element_type="hex8",
    )

    assert mesh.elements.shape == (1, 8)
    assert mesh.element_type == "hex8"


def test_volume_mesh_reorders_negative_tet4_orientation() -> None:
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    mesh = VolumeMesh(X=X, elements=np.array([[0, 2, 1, 3]]))
    dets = signed_tet4_jacobian_determinants(mesh.X, mesh.elements)

    assert np.all(dets > 0.0)
    assert mesh.elements.tolist() == [[0, 1, 2, 3]]
