from __future__ import annotations

import pytest

from sfc.fem.increment_control import (
    active_set_line_search_choice,
    active_set_stability_after_line_search,
    contact_active_set_is_stable,
    increment_convergence_decision,
    increment_cutback_candidate_dt,
    increment_gate_row,
    run_automatic_increment_controller,
)


def test_increment_decision_requires_all_abaqus_style_gates() -> None:
    decision = increment_convergence_decision(
        residual_converged=True,
        correction_converged=True,
        contact_force_increment_converged=True,
        active_set_stable=True,
        iteration_count=2,
        max_iterations=4,
        accept_unconverged=False,
    )

    assert decision.converged
    assert decision.accepted
    assert not decision.cutback_required
    assert decision.reason == "residual_correction_contact_force_active_set"

    limited = increment_convergence_decision(
        residual_converged=True,
        correction_converged=True,
        contact_force_increment_converged=False,
        active_set_stable=True,
        iteration_count=4,
        max_iterations=4,
        accept_unconverged=False,
    )

    assert not limited.converged
    assert not limited.accepted
    assert limited.cutback_required
    assert limited.iteration_limited
    assert limited.reason == "contact_force_increment"


def test_increment_cutback_candidate_respects_floor() -> None:
    assert increment_cutback_candidate_dt(1.0e-3, min_dt=1.0e-6, cutback_factor=0.5) == pytest.approx(5.0e-4)
    assert increment_cutback_candidate_dt(1.0e-6, min_dt=1.0e-6, cutback_factor=0.5) is None
    assert increment_cutback_candidate_dt(1.5e-6, min_dt=1.0e-6, cutback_factor=0.5) == pytest.approx(1.0e-6)


def test_automatic_increment_controller_retries_failed_trial_from_same_state() -> None:
    calls: list[tuple[float, float, int]] = []

    def trial(start: float, dt: float, index: int):
        calls.append((start, dt, index))
        return increment_convergence_decision(
            residual_converged=index > 1,
            correction_converged=True,
            contact_force_increment_converged=True,
            active_set_stable=True,
            iteration_count=2,
            max_iterations=2,
            accept_unconverged=False,
        )

    result = run_automatic_increment_controller(
        duration=2.0e-3,
        initial_dt=1.0e-3,
        min_dt=1.0e-4,
        cutback_factor=0.5,
        trial=trial,
    )

    assert calls[0] == pytest.approx((0.0, 1.0e-3, 1))
    assert calls[1] == pytest.approx((0.0, 5.0e-4, 2))
    assert result.cutback_count == 1
    assert result.events[0].cutback_required
    assert result.events[0].retry_required
    assert result.accepted_times[-1] == pytest.approx(2.0e-3)
    assert result.final_time == pytest.approx(2.0e-3)


def test_increment_gate_row_uses_configurable_prefix() -> None:
    decision = increment_convergence_decision(
        residual_converged=False,
        correction_converged=True,
        contact_force_increment_converged=True,
        active_set_stable=True,
        iteration_count=3,
        max_iterations=3,
        accept_unconverged=False,
    )

    row = increment_gate_row(
        residual_converged=False,
        correction_converged=True,
        contact_force_increment_converged=True,
        active_set_stable=True,
        normalized_residual=0.2,
        normalized_correction=0.01,
        normalized_contact_force_increment=0.03,
        contact_force_increment_norm=4.0,
        decision=decision,
        cutback_candidate_dt=2.5e-4,
        prefix="contact",
    )

    assert row["contact_residual_converged"] == 0
    assert row["contact_increment_cutback_required"] == 1
    assert row["contact_increment_cutback_candidate_dt"] == pytest.approx(2.5e-4)
    assert row["contact_step_convergence_reason"] == "residual"


def test_active_set_line_search_reports_unstable_candidates() -> None:
    assert contact_active_set_is_stable(((1,),), ((1,),), require_stability=True)
    assert not contact_active_set_is_stable(((1,),), None, require_stability=True)
    alpha, stable, trials = active_set_line_search_choice(
        ((1,),),
        [(1.0, ((2,),)), (0.5, ((3,),))],
        require_stability=True,
    )
    assert alpha == pytest.approx(1.0)
    assert not stable
    assert trials == 2
    assert not active_set_stability_after_line_search(
        True,
        line_search_attempted=True,
        line_search_stable=False,
    )
