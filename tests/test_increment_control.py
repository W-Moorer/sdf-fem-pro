from __future__ import annotations

import pytest

from sfc.fem.increment_control import (
    active_set_line_search_choice,
    active_set_line_search_gate_metrics,
    active_set_stability_after_line_search,
    contact_active_set_is_stable,
    increment_convergence_decision,
    increment_cutback_candidate_dt,
    increment_gate_row,
    increment_trial_ledger_gate_metrics,
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


def test_increment_trial_ledger_gate_rejects_cutback_as_accepted_state() -> None:
    accepted_flags = {
        "hard_increment_converged": 1,
        "hard_increment_accepted": 1,
        "hard_increment_cutback_required": 0,
        "hard_residual_converged": 1,
        "hard_correction_converged": 1,
        "hard_contact_force_increment_converged": 1,
        "hard_active_set_stable": 1,
        "hard_iteration_limit_reached": 0,
    }
    history = [{"time": 5.0e-4, **accepted_flags}]
    accepted_trial = {
        "hard_trial_index": 2,
        "hard_trial_start_time": 0.0,
        "hard_trial_end_time": 5.0e-4,
        "hard_trial_accepted": 1,
        "hard_trial_retry_required": 0,
        **accepted_flags,
    }

    gate = increment_trial_ledger_gate_metrics(history, [accepted_trial], prefix="hard")

    assert int(gate["hard_trial_gate_passed"]) == 1
    assert int(gate["hard_trial_accepted_count"]) == 1

    accepted_cutback = dict(accepted_trial)
    accepted_cutback["hard_increment_cutback_required"] = 1
    failed = increment_trial_ledger_gate_metrics(history, [accepted_cutback], prefix="hard")

    assert int(failed["hard_trial_gate_passed"]) == 0
    assert int(failed["hard_trial_accepted_no_cutback"]) == 0


def test_increment_trial_ledger_gate_allows_rejected_cutback_retry_only_as_diagnostic() -> None:
    accepted_flags = {
        "source_increment_converged": 1,
        "source_increment_accepted": 1,
        "source_increment_cutback_required": 0,
        "source_residual_converged": 1,
        "source_correction_converged": 1,
        "source_contact_force_increment_converged": 1,
        "source_active_set_stable": 1,
        "source_iteration_limit_reached": 0,
    }
    history = [
        {"time": 5.0e-4, **accepted_flags},
        {"time": 1.0e-3, **accepted_flags},
    ]
    trial_rows = [
        {
            "source_trial_index": 1,
            "source_trial_start_time": 0.0,
            "source_trial_end_time": 1.0e-3,
            "source_trial_accepted": 0,
            "source_trial_retry_required": 1,
            "source_increment_converged": 0,
            "source_increment_accepted": 0,
            "source_increment_cutback_required": 1,
        },
        {
            "source_trial_index": 2,
            "source_trial_start_time": 0.0,
            "source_trial_end_time": 5.0e-4,
            "source_trial_accepted": 1,
            "source_trial_retry_required": 0,
            **accepted_flags,
        },
        {
            "source_trial_index": 3,
            "source_trial_start_time": 5.0e-4,
            "source_trial_end_time": 1.0e-3,
            "source_trial_accepted": 1,
            "source_trial_retry_required": 0,
            **accepted_flags,
        },
    ]

    gate = increment_trial_ledger_gate_metrics(history, trial_rows, prefix="source")

    assert int(gate["source_trial_gate_passed"]) == 1
    assert int(gate["source_trial_rejected_count"]) == 1
    assert int(gate["source_trial_cutback_trials_consistent"]) == 1

    bad_history = [{"time": 1.0e-3, **accepted_flags}]
    bad = increment_trial_ledger_gate_metrics(bad_history, [trial_rows[0]], prefix="source")

    assert int(bad["source_trial_gate_passed"]) == 0
    assert int(bad["source_trial_rejected_only_history_hits"]) == 1


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


def test_active_set_line_search_gate_requires_stable_accepted_rows() -> None:
    summary = {
        "source_active_set_line_search": True,
        "source_contact_active_set_stability": True,
        "source_line_search_trial_count": 2,
        "source_line_search_unstable_count": 0,
        "source_accepted_tracking_commit_count": 1,
        "final_active_contact_samples": 1,
    }
    history = [
        {
            "active_contact_region_count": 1,
            "contact_active_area": 2.0,
            "contact_region_normal_force": 3.0,
            "source_line_search_trial_count": 1,
            "source_line_search_reduced_count": 0,
            "source_line_search_stable_count": 1,
            "source_line_search_unstable_count": 0,
            "source_line_search_last_alpha": 1.0,
            "source_accepted_tracking_committed": 1,
            "source_active_set_stable": 1,
            "source_increment_accepted": 1,
            "source_increment_cutback_required": 0,
            "source_iteration_limit_reached": 0,
        }
    ]

    gate = active_set_line_search_gate_metrics(summary, history, prefix="source")

    assert int(gate["source_active_set_line_search_gate_passed"]) == 1
    assert int(gate["source_active_set_line_search_row_active_set_stable_ok"]) == 1
    assert int(gate["source_active_set_line_search_row_commit_ok"]) == 1

    unstable = [dict(history[0])]
    unstable[0]["source_active_set_stable"] = 0
    failed_unstable = active_set_line_search_gate_metrics(summary, unstable, prefix="source")

    assert int(failed_unstable["source_active_set_line_search_gate_passed"]) == 0
    assert int(failed_unstable["source_active_set_line_search_row_active_set_stable_ok"]) == 0

    cutback = [dict(history[0])]
    cutback[0]["source_increment_cutback_required"] = 1
    failed_cutback = active_set_line_search_gate_metrics(summary, cutback, prefix="source")

    assert int(failed_cutback["source_active_set_line_search_gate_passed"]) == 0
    assert int(failed_cutback["source_active_set_line_search_row_no_cutback_ok"]) == 0


def test_active_set_line_search_gate_requires_committed_tracking_for_active_contact() -> None:
    summary = {
        "hard_active_set_line_search": True,
        "hard_contact_active_set_stability": True,
        "hard_line_search_trial_count": 1,
        "hard_line_search_unstable_count": 0,
        "hard_accepted_tracking_commit_count": 0,
        "hard_final_active_contact_samples": 1,
    }
    history = [
        {
            "active_contact_region_count": 1,
            "hard_line_search_trial_count": 1,
            "hard_line_search_reduced_count": 0,
            "hard_line_search_stable_count": 1,
            "hard_line_search_unstable_count": 0,
            "hard_line_search_last_alpha": 1.0,
            "hard_accepted_tracking_committed": 0,
            "hard_active_set_stable": 1,
            "hard_increment_accepted": 1,
            "hard_increment_cutback_required": 0,
            "hard_iteration_limit_reached": 0,
        }
    ]

    gate = active_set_line_search_gate_metrics(summary, history, prefix="hard")

    assert int(gate["hard_active_set_line_search_gate_passed"]) == 0
    assert int(gate["hard_active_set_line_search_summary_commit_ok"]) == 0
    assert int(gate["hard_active_set_line_search_row_commit_ok"]) == 0
