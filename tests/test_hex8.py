import numpy as np

from sfc.fem.hex8 import (
    C3D8_NATURAL_NODE_COORDS,
    hex8_center_strain_stress,
    hex8_consistent_mass,
    hex8_lumped_mass,
    hex8_mass,
    hex8_shape_functions,
    hex8_stiffness,
    hex8_volume,
)


def _unit_hex() -> np.ndarray:
    return 0.5 * (C3D8_NATURAL_NODE_COORDS + 1.0)


def _translation_vector(value: np.ndarray) -> np.ndarray:
    return np.tile(np.asarray(value, dtype=float), 8)


def test_hex8_unit_cube_volume_equals_one() -> None:
    assert np.isclose(hex8_volume(_unit_hex()), 1.0)


def test_hex8_shape_functions_are_partition_of_unity() -> None:
    shape = hex8_shape_functions(0.2, -0.3, 0.4)

    assert np.isclose(np.sum(shape), 1.0)
    assert np.all(shape >= 0.0)


def test_hex8_stiffness_is_symmetric_and_translation_null() -> None:
    K = hex8_stiffness(_unit_hex(), 1000.0, 0.25)
    ux = _translation_vector(np.array([1.0, 0.0, 0.0]))
    uy = _translation_vector(np.array([0.0, -2.0, 0.0]))
    uz = _translation_vector(np.array([0.0, 0.0, 3.0]))

    assert np.allclose(K, K.T)
    assert abs(float(ux @ K @ ux)) < 1.0e-9
    assert abs(float(uy @ K @ uy)) < 1.0e-9
    assert abs(float(uz @ K @ uz)) < 1.0e-9


def test_hex8_total_mass_is_conserved() -> None:
    Xe = _unit_hex()
    rho = 2.5

    consistent = hex8_consistent_mass(Xe, rho)
    lumped = hex8_lumped_mass(Xe, rho)

    assert np.isclose(np.sum(consistent[0::3, 0::3]), rho * hex8_volume(Xe))
    assert np.isclose(np.sum(lumped[0::3, 0::3]), rho * hex8_volume(Xe))
    assert np.allclose(hex8_mass(Xe, rho, kind="consistent"), consistent)
    assert np.allclose(hex8_mass(Xe, rho, kind="lumped"), lumped)


def test_hex8_center_strain_stress_matches_affine_extension() -> None:
    Xe = _unit_hex()
    strain_xx = 0.01
    Ue = np.zeros_like(Xe)
    Ue[:, 0] = strain_xx * Xe[:, 0]

    strain, stress = hex8_center_strain_stress(Xe, Ue, 1000.0, 0.25)

    assert np.allclose(strain, np.array([strain_xx, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert stress[0] > 0.0
