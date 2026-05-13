import numpy as np

from sfc.fem.tet4 import (
    tet4_consistent_mass,
    tet4_lumped_mass,
    tet4_stiffness,
    tet4_volume,
)


UNIT_TET = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def test_unit_tetrahedron_volume_is_one_sixth() -> None:
    assert np.isclose(tet4_volume(UNIT_TET), 1.0 / 6.0)


def test_tet4_stiffness_is_symmetric() -> None:
    Ke = tet4_stiffness(UNIT_TET, E=1000.0, nu=0.25)

    assert Ke.shape == (12, 12)
    assert np.allclose(Ke, Ke.T)


def test_tet4_stiffness_has_zero_energy_for_rigid_translations() -> None:
    Ke = tet4_stiffness(UNIT_TET, E=1000.0, nu=0.25)

    for axis in range(3):
        u = np.zeros(12)
        u[axis::3] = 1.0
        energy = float(u @ Ke @ u)
        residual = Ke @ u

        assert abs(energy) < 1e-10
        assert np.linalg.norm(residual) < 1e-10


def test_tet4_stiffness_has_zero_energy_for_rigid_rotations() -> None:
    Ke = tet4_stiffness(UNIT_TET, E=1000.0, nu=0.25)

    for omega in np.eye(3):
        u = np.cross(np.broadcast_to(omega, UNIT_TET.shape), UNIT_TET).ravel()
        energy = float(u @ Ke @ u)
        residual = Ke @ u

        assert abs(energy) < 1e-10
        assert np.linalg.norm(residual) < 1e-10


def test_tet4_stiffness_is_positive_semidefinite() -> None:
    Ke = tet4_stiffness(UNIT_TET, E=1000.0, nu=0.25)

    eigenvalues = np.linalg.eigvalsh(Ke)

    assert np.min(eigenvalues) > -1e-10


def test_tet4_mass_total_equals_density_times_volume() -> None:
    rho = 7.5
    expected_total_mass = rho * tet4_volume(UNIT_TET)

    for mass in (
        tet4_consistent_mass(UNIT_TET, rho),
        tet4_lumped_mass(UNIT_TET, rho),
    ):
        assert mass.shape == (12, 12)
        for axis in range(3):
            rigid_translation = np.zeros(12)
            rigid_translation[axis::3] = 1.0
            total_mass = rigid_translation @ mass @ rigid_translation

            assert np.isclose(total_mass, expected_total_mass)
