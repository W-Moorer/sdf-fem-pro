import numpy as np
import pytest

from sfc.fem.elements import (
    C3D8_HEX8_BACKEND,
    available_element_backends,
    canonical_fem_element_type,
    get_element_backend,
)
from sfc.fem.hex8 import hex8_mass, hex8_stiffness, hex8_volume
from sfc.fem.tet4 import tet4_mass, tet4_stiffness, tet4_volume


def _unit_tet() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _unit_hex() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )


def test_tet4_and_c3d4_resolve_to_same_backend() -> None:
    tet4_backend = get_element_backend("tet4")
    c3d4_backend = get_element_backend("C3D4")

    assert tet4_backend is c3d4_backend
    assert c3d4_backend.name == "tet4"
    assert c3d4_backend.nodes_per_element == 4
    assert canonical_fem_element_type("C3D4") == "tet4"
    assert available_element_backends() == ("tet4", "hex8")


def test_c3d4_backend_matches_existing_tet4_kernels() -> None:
    Xe = _unit_tet()
    backend = get_element_backend("c3d4")

    assert np.isclose(backend.volume(Xe), tet4_volume(Xe))
    assert np.allclose(backend.stiffness(Xe, 1000.0, 0.25), tet4_stiffness(Xe, 1000.0, 0.25))
    assert np.allclose(backend.mass(Xe, 2.5, "consistent"), tet4_mass(Xe, 2.5, kind="consistent"))
    assert np.allclose(backend.mass(Xe, 2.5, "lumped"), tet4_mass(Xe, 2.5, kind="lumped"))


def test_hex8_and_c3d8_resolve_to_same_backend() -> None:
    hex8_backend = get_element_backend("hex8")
    c3d8_backend = get_element_backend("C3D8")

    assert hex8_backend is c3d8_backend
    assert c3d8_backend is C3D8_HEX8_BACKEND
    assert c3d8_backend.name == "hex8"
    assert c3d8_backend.nodes_per_element == 8
    assert canonical_fem_element_type("C3D8") == "hex8"


def test_c3d8_backend_matches_hex8_kernels() -> None:
    Xe = _unit_hex()
    backend = get_element_backend("c3d8")

    assert np.isclose(backend.volume(Xe), hex8_volume(Xe))
    assert np.allclose(backend.stiffness(Xe, 1000.0, 0.25), hex8_stiffness(Xe, 1000.0, 0.25))
    assert np.allclose(backend.mass(Xe, 2.5, "consistent"), hex8_mass(Xe, 2.5, kind="consistent"))
    assert np.allclose(backend.mass(Xe, 2.5, "lumped"), hex8_mass(Xe, 2.5, kind="lumped"))


def test_unregistered_c3d10_backend_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="No FEM element backend registered.*c3d10"):
        get_element_backend("C3D10")
