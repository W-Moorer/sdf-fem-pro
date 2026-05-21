from __future__ import annotations

import numpy as np
import pytest

from sfc.contact.hard_contact import (
    hard_contact_gap_jacobian_from_samples,
    hard_contact_pressure_compliance_from_samples,
    solve_linear_hard_contact_active_set,
    solve_linear_hard_contact_from_samples,
    solve_linear_hard_contact_from_samples_with_dirichlet,
    solve_linear_hard_contact_with_dirichlet,
)
from sfc.fem.calculix_aligned import ContactSample


def test_linear_hard_contact_inactive_matches_unconstrained_solution() -> None:
    solution = solve_linear_hard_contact_active_set(
        np.asarray([[10.0]], dtype=float),
        np.asarray([2.0], dtype=float),
        np.asarray([0.1], dtype=float),
        np.asarray([[1.0]], dtype=float),
    )

    assert solution.converged
    assert solution.displacement == pytest.approx([0.2])
    assert solution.gaps == pytest.approx([0.3])
    assert solution.multipliers == pytest.approx([0.0])


def test_linear_hard_contact_active_enforces_zero_gap() -> None:
    solution = solve_linear_hard_contact_active_set(
        np.asarray([[10.0]], dtype=float),
        np.asarray([-2.0], dtype=float),
        np.asarray([0.1], dtype=float),
        np.asarray([[1.0]], dtype=float),
    )

    assert solution.converged
    assert solution.active.tolist() == [True]
    assert solution.displacement == pytest.approx([-0.1])
    assert solution.gaps == pytest.approx([0.0])
    assert solution.multipliers == pytest.approx([1.0])


def test_linear_hard_contact_releases_negative_multiplier() -> None:
    solution = solve_linear_hard_contact_active_set(
        np.asarray([[10.0]], dtype=float),
        np.asarray([2.0], dtype=float),
        np.asarray([0.1], dtype=float),
        np.asarray([[1.0]], dtype=float),
        initial_active=np.asarray([True]),
    )

    assert solution.converged
    assert solution.active.tolist() == [False]
    assert solution.multipliers == pytest.approx([0.0])


def test_hard_contact_sample_jacobian_includes_slave_and_master_weights() -> None:
    sample = ContactSample(
        node_ids=np.asarray([0, 1], dtype=np.int64),
        shape_weights=np.asarray([0.25, 0.75], dtype=float),
        gap=-0.05,
        normal=np.asarray([0.0, 0.0, 2.0], dtype=float),
        area=1.0,
        stiffness=1.0,
        master_node_ids=np.asarray([0], dtype=np.int64),
        master_shape_weights=np.asarray([1.0], dtype=float),
    )

    g0, J = hard_contact_gap_jacobian_from_samples(
        [sample],
        n_total_dofs=9,
        master_dof_offset=6,
    )

    assert g0 == pytest.approx([-0.05])
    assert J[0, 2] == pytest.approx(0.25)
    assert J[0, 5] == pytest.approx(0.75)
    assert J[0, 8] == pytest.approx(-1.0)


def test_hard_contact_from_samples_matches_one_dof_active_solution() -> None:
    sample = ContactSample(
        node_ids=np.asarray([0], dtype=np.int64),
        shape_weights=np.asarray([1.0], dtype=float),
        gap=0.1,
        normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
        area=1.0,
        stiffness=1.0,
    )
    K = np.diag([1.0, 1.0, 10.0])
    f = np.asarray([0.0, 0.0, -2.0], dtype=float)

    solution = solve_linear_hard_contact_from_samples(K, f, [sample], n_total_dofs=3)

    assert solution.converged
    assert solution.displacement == pytest.approx([0.0, 0.0, -0.1])
    assert solution.gaps == pytest.approx([0.0])
    assert solution.multipliers == pytest.approx([1.0])


def test_linear_hard_contact_with_dirichlet_eliminates_prescribed_dofs() -> None:
    solution = solve_linear_hard_contact_with_dirichlet(
        np.diag([10.0, 10.0]),
        np.asarray([-2.0, 0.0], dtype=float),
        np.asarray([0.1], dtype=float),
        np.asarray([[1.0, -1.0]], dtype=float),
        fixed_dofs=np.asarray([1], dtype=np.int64),
        fixed_values=np.asarray([0.2], dtype=float),
    )

    assert solution.converged
    assert solution.active.tolist() == [True]
    assert solution.displacement == pytest.approx([0.1, 0.2])
    assert solution.gaps == pytest.approx([0.0])
    assert solution.multipliers == pytest.approx([3.0])


def test_hard_contact_from_samples_with_dirichlet_includes_master_motion() -> None:
    sample = ContactSample(
        node_ids=np.asarray([0], dtype=np.int64),
        shape_weights=np.asarray([1.0], dtype=float),
        gap=0.1,
        normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
        area=1.0,
        stiffness=1.0,
        master_node_ids=np.asarray([0], dtype=np.int64),
        master_shape_weights=np.asarray([1.0], dtype=float),
    )
    K = np.eye(6, dtype=float) * 10.0
    f = np.asarray([0.0, 0.0, -2.0, 0.0, 0.0, 0.0], dtype=float)

    solution = solve_linear_hard_contact_from_samples_with_dirichlet(
        K,
        f,
        [sample],
        n_total_dofs=6,
        master_dof_offset=3,
        fixed_dofs=np.asarray([3, 4, 5], dtype=np.int64),
        fixed_values=np.asarray([0.0, 0.0, 0.2], dtype=float),
    )

    assert solution.converged
    assert solution.displacement[2] == pytest.approx(0.1)
    assert solution.displacement[5] == pytest.approx(0.2)
    assert solution.gaps == pytest.approx([0.0])
    assert solution.multipliers == pytest.approx([3.0])


def test_regularized_hard_contact_allows_pressure_compliance_overclosure() -> None:
    solution = solve_linear_hard_contact_active_set(
        np.asarray([[10.0]], dtype=float),
        np.asarray([-2.0], dtype=float),
        np.asarray([0.1], dtype=float),
        np.asarray([[1.0]], dtype=float),
        normal_compliance=np.asarray([0.05], dtype=float),
    )

    assert solution.converged
    assert solution.active.tolist() == [True]
    assert solution.multipliers[0] > 0.0
    assert solution.gaps[0] < 0.0
    assert solution.gaps[0] + 0.05 * solution.multipliers[0] == pytest.approx(0.0)


def test_pressure_compliance_from_samples_uses_area_and_pressure_stiffness() -> None:
    samples = [
        ContactSample(
            node_ids=np.asarray([0], dtype=np.int64),
            shape_weights=np.asarray([1.0], dtype=float),
            gap=-0.1,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=0.25,
            stiffness=100.0,
        ),
        ContactSample(
            node_ids=np.asarray([1], dtype=np.int64),
            shape_weights=np.asarray([1.0], dtype=float),
            gap=-0.1,
            normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
            area=0.5,
            stiffness=100.0,
        ),
    ]

    compliance = hard_contact_pressure_compliance_from_samples(samples, pressure_stiffness=200.0)

    assert compliance == pytest.approx([1.0 / (200.0 * 0.25), 1.0 / (200.0 * 0.5)])
