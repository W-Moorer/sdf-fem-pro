"""Abaqus-style nonlinear increment acceptance and cutback utilities.

These helpers are intentionally independent from a specific benchmark runner.
They encode the common acceptance gates used by the current implicit contact
paths: residual convergence, displacement correction convergence, contact-force
increment convergence, and active-set stability.  A failed trial at the
iteration limit requests a cutback instead of being accepted as a solved state.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class IncrementConvergenceDecision:
    """One nonlinear increment acceptance decision."""

    converged: bool
    reason: str
    iteration_limited: bool
    cutback_required: bool
    accepted: bool


@dataclass(frozen=True, slots=True)
class AutomaticIncrementEvent:
    """One automatic-increment trial record."""

    trial_index: int
    start_time: float
    end_time: float
    dt: float
    accepted: bool
    retry_required: bool
    cutback_required: bool
    cutback_candidate_dt: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class AutomaticIncrementResult:
    """Automatic-increment controller diagnostics."""

    events: tuple[AutomaticIncrementEvent, ...]
    accepted_times: tuple[float, ...]
    cutback_count: int
    final_time: float


def contact_active_set_is_stable(
    active_signature: tuple[tuple[int, ...], ...],
    previous_active_signature: tuple[tuple[int, ...], ...] | None,
    *,
    require_stability: bool,
) -> bool:
    """Return true when contact status is stable enough for acceptance."""

    if not bool(require_stability):
        return True
    return previous_active_signature is not None and active_signature == previous_active_signature


def active_set_line_search_choice(
    current_signature: tuple[tuple[int, ...], ...] | None,
    candidates: Iterable[tuple[float, tuple[tuple[int, ...], ...]]],
    *,
    require_stability: bool,
) -> tuple[float, bool, int]:
    """Choose a correction fraction that preserves the current active set."""

    first_alpha: float | None = None
    trial_count = 0
    for alpha, signature in candidates:
        alpha_value = float(alpha)
        if first_alpha is None:
            first_alpha = alpha_value
        trial_count += 1
        if (not bool(require_stability)) or current_signature is None or signature == current_signature:
            return alpha_value, True, int(trial_count)
    return (1.0 if first_alpha is None else float(first_alpha)), False, int(trial_count)


def active_set_stability_after_line_search(
    active_set_stable: bool,
    *,
    line_search_attempted: bool,
    line_search_stable: bool,
) -> bool:
    """Combine Newton active-set stability with the line-search result."""

    return bool(active_set_stable) and (not bool(line_search_attempted) or bool(line_search_stable))


def increment_convergence_decision(
    *,
    residual_converged: bool,
    correction_converged: bool,
    contact_force_increment_converged: bool,
    active_set_stable: bool,
    iteration_count: int,
    max_iterations: int,
    accept_unconverged: bool,
) -> IncrementConvergenceDecision:
    """Evaluate residual/correction/contact/status gates for one increment."""

    converged = (
        bool(residual_converged)
        and bool(correction_converged)
        and bool(contact_force_increment_converged)
        and bool(active_set_stable)
    )
    if converged:
        return IncrementConvergenceDecision(
            converged=True,
            reason="residual_correction_contact_force_active_set",
            iteration_limited=False,
            cutback_required=False,
            accepted=True,
        )
    if not bool(residual_converged):
        reason = "residual"
    elif not bool(correction_converged):
        reason = "correction"
    elif not bool(contact_force_increment_converged):
        reason = "contact_force_increment"
    else:
        reason = "contact_active_set"
    limited = int(iteration_count) >= max(1, int(max_iterations))
    return IncrementConvergenceDecision(
        converged=False,
        reason=reason,
        iteration_limited=limited,
        cutback_required=bool(limited and not bool(accept_unconverged)),
        accepted=bool(accept_unconverged),
    )


def increment_cutback_candidate_dt(
    current_dt: float,
    *,
    min_dt: float,
    cutback_factor: float,
) -> float | None:
    """Return the next trial increment or ``None`` when already at the floor."""

    h = float(current_dt)
    h_min = float(min_dt)
    factor = float(cutback_factor)
    if h <= 0.0:
        raise ValueError("current_dt must be positive")
    if h_min <= 0.0:
        raise ValueError("min_dt must be positive")
    if not (0.0 < factor < 1.0):
        raise ValueError("cutback_factor must lie in (0, 1)")
    if h <= h_min * (1.0 + 1.0e-12):
        return None
    return max(h_min, h * factor)


def run_automatic_increment_controller(
    *,
    duration: float,
    initial_dt: float,
    min_dt: float,
    cutback_factor: float,
    trial: Callable[[float, float, int], IncrementConvergenceDecision],
) -> AutomaticIncrementResult:
    """Run an accept/cutback scheduler for nonlinear trial increments."""

    total = float(duration)
    h0 = float(initial_dt)
    h_min = float(min_dt)
    if total < 0.0:
        raise ValueError("duration must be non-negative")
    if h0 <= 0.0:
        raise ValueError("initial_dt must be positive")
    if h_min <= 0.0:
        raise ValueError("min_dt must be positive")
    if not (0.0 < float(cutback_factor) < 1.0):
        raise ValueError("cutback_factor must lie in (0, 1)")
    accepted_time = 0.0
    trial_dt = min(h0, max(total, h_min)) if total > 0.0 else h0
    trial_index = 0
    cutback_count = 0
    events: list[AutomaticIncrementEvent] = []
    accepted_times: list[float] = []
    while accepted_time < total - 1.0e-15:
        h = min(float(trial_dt), total - accepted_time)
        trial_index += 1
        decision = trial(float(accepted_time), float(h), int(trial_index))
        candidate = (
            increment_cutback_candidate_dt(
                h,
                min_dt=h_min,
                cutback_factor=float(cutback_factor),
            )
            if decision.cutback_required
            else None
        )
        retry_required = bool(decision.cutback_required and candidate is not None)
        accepted = bool(decision.accepted and not decision.cutback_required)
        events.append(
            AutomaticIncrementEvent(
                trial_index=int(trial_index),
                start_time=float(accepted_time),
                end_time=float(accepted_time + h),
                dt=float(h),
                accepted=accepted,
                retry_required=retry_required,
                cutback_required=bool(decision.cutback_required),
                cutback_candidate_dt=candidate,
                reason=str(decision.reason),
            )
        )
        if decision.cutback_required:
            if candidate is None:
                raise RuntimeError("automatic increment reached minimum dt without satisfying convergence gates")
            trial_dt = float(candidate)
            cutback_count += 1
            continue
        if not accepted:
            raise RuntimeError("automatic increment trial was neither accepted nor eligible for cutback")
        accepted_time += h
        accepted_times.append(float(accepted_time))
        trial_dt = min(h0, max(h_min, h * (1.25 if decision.converged else 1.0)))
    return AutomaticIncrementResult(
        events=tuple(events),
        accepted_times=tuple(accepted_times),
        cutback_count=int(cutback_count),
        final_time=float(accepted_time),
    )


def increment_gate_row(
    *,
    residual_converged: bool,
    correction_converged: bool,
    contact_force_increment_converged: bool,
    active_set_stable: bool,
    normalized_residual: float,
    normalized_correction: float,
    normalized_contact_force_increment: float,
    contact_force_increment_norm: float,
    decision: IncrementConvergenceDecision,
    cutback_candidate_dt: float | None,
    prefix: str = "source",
) -> dict[str, Any]:
    """Return one history/manifest row for increment acceptance gates."""

    label = str(prefix).strip("_")
    return {
        f"{label}_residual_converged": int(bool(residual_converged)),
        f"{label}_correction_converged": int(bool(correction_converged)),
        f"{label}_contact_force_increment_converged": int(bool(contact_force_increment_converged)),
        f"{label}_active_set_stable": int(bool(active_set_stable)),
        f"{label}_increment_converged": int(bool(decision.converged)),
        f"{label}_increment_accepted": int(bool(decision.accepted)),
        f"{label}_increment_cutback_required": int(bool(decision.cutback_required)),
        f"{label}_increment_cutback_candidate_dt": (
            "" if cutback_candidate_dt is None else float(cutback_candidate_dt)
        ),
        f"{label}_iteration_limit_reached": int(bool(decision.iteration_limited)),
        f"{label}_step_convergence_reason": str(decision.reason),
        f"{label}_normalized_residual": float(normalized_residual),
        f"{label}_normalized_correction": float(normalized_correction),
        f"{label}_normalized_contact_force_increment": float(normalized_contact_force_increment),
        f"{label}_contact_force_increment_norm": float(contact_force_increment_norm),
    }


def _row_int_flag(row: Mapping[str, Any], key: str, *, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _row_float_with_keys(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if value in ("", None):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return float(number)
    return None


def _match_time_sequence(
    query_times: Sequence[float],
    reference_times: Sequence[float],
    *,
    tolerance: float,
) -> tuple[int, float]:
    """Return ordered-subsequence matches and largest matched time error."""

    if not query_times:
        return 0, 0.0
    reference_index = 0
    matched = 0
    max_error = 0.0
    for query_time in query_times:
        best_index = -1
        best_error = math.inf
        for index in range(reference_index, len(reference_times)):
            error = abs(float(reference_times[index]) - float(query_time))
            if error < best_error:
                best_error = error
                best_index = index
            if error <= float(tolerance):
                break
        if best_index >= 0 and best_error <= float(tolerance):
            reference_index = best_index + 1
            matched += 1
            max_error = max(max_error, float(best_error))
    return matched, float(max_error if math.isfinite(max_error) else math.inf)


def _time_in_list(time: float, candidates: Sequence[float], *, tolerance: float) -> bool:
    return any(abs(float(candidate) - float(time)) <= float(tolerance) for candidate in candidates)


def increment_trial_ledger_gate_metrics(
    history_rows: Sequence[Mapping[str, Any]],
    trial_rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str = "source",
    tolerance: float = 1.0e-12,
    history_time_keys: Sequence[str] | None = None,
    trial_time_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Gate accepted increment history against the trial/cutback ledger.

    The accepted history may be frame-strided, so its times are checked as an
    ordered subset of accepted trial end times.  Rejected or cutback trials are
    allowed in the diagnostic ledger, but they must not appear as accepted
    output states.  Accepted trials and accepted history rows must satisfy the
    strict residual/correction/contact-force/active-set flags.
    """

    label = str(prefix).strip("_")
    trial_accepted_key = f"{label}_trial_accepted"
    trial_retry_key = f"{label}_trial_retry_required"
    if history_time_keys is None:
        history_time_keys = ("time", f"{label}_trial_end_time")
    if trial_time_keys is None:
        trial_time_keys = (f"{label}_trial_end_time", "time")

    accepted_trials = [row for row in trial_rows if _row_int_flag(row, trial_accepted_key) == 1]
    rejected_trials = [row for row in trial_rows if _row_int_flag(row, trial_accepted_key) == 0]
    accepted_state_flags = (
        (f"{label}_increment_converged", 1),
        (f"{label}_increment_accepted", 1),
        (f"{label}_increment_cutback_required", 0),
        (f"{label}_residual_converged", 1),
        (f"{label}_correction_converged", 1),
        (f"{label}_contact_force_increment_converged", 1),
        (f"{label}_active_set_stable", 1),
        (f"{label}_iteration_limit_reached", 0),
    )
    history_times = [
        time
        for time in (_row_float_with_keys(row, history_time_keys) for row in history_rows)
        if time is not None
    ]
    accepted_trial_times = [
        time
        for time in (_row_float_with_keys(row, trial_time_keys) for row in accepted_trials)
        if time is not None
    ]
    rejected_trial_times = [
        time
        for time in (_row_float_with_keys(row, trial_time_keys) for row in rejected_trials)
        if time is not None
    ]
    matched_count, max_time_error = _match_time_sequence(
        history_times,
        accepted_trial_times,
        tolerance=float(tolerance),
    )
    unmatched_history_count = max(0, len(history_times) - matched_count)
    history_all_accepted = all(_row_int_flag(row, f"{label}_increment_accepted", default=1) == 1 for row in history_rows)
    history_required_columns_present = all(
        all(key in row for key, _expected in accepted_state_flags) for row in history_rows
    )
    history_strictly_converged = all(
        all(_row_int_flag(row, key, default=expected) == expected for key, expected in accepted_state_flags)
        for row in history_rows
    )
    accepted_required_columns_present = all(
        all(key in row for key, _expected in accepted_state_flags) for row in accepted_trials
    )
    accepted_strictly_converged = all(
        all(_row_int_flag(row, key, default=expected) == expected for key, expected in accepted_state_flags)
        for row in accepted_trials
    )
    accepted_cutback_count = sum(
        1 for row in accepted_trials if _row_int_flag(row, f"{label}_increment_cutback_required", default=0) != 0
    )
    accepted_retry_count = sum(1 for row in accepted_trials if _row_int_flag(row, trial_retry_key, default=0) != 0)
    accepted_unconverged_count = sum(
        1 for row in accepted_trials if _row_int_flag(row, f"{label}_increment_converged", default=1) != 1
    )
    cutback_trials = [
        row for row in trial_rows if _row_int_flag(row, f"{label}_increment_cutback_required", default=0) == 1
    ]
    cutback_trials_rejected = all(_row_int_flag(row, trial_accepted_key, default=0) == 0 for row in cutback_trials)
    cutback_trials_retry = all(_row_int_flag(row, trial_retry_key, default=1) == 1 for row in cutback_trials)
    rejected_only_history_hits = 0
    for rejected_time in rejected_trial_times:
        if not _time_in_list(rejected_time, accepted_trial_times, tolerance=float(tolerance)) and _time_in_list(
            rejected_time,
            history_times,
            tolerance=float(tolerance),
        ):
            rejected_only_history_hits += 1
    cutback_trials_consistent = bool(cutback_trials_rejected and cutback_trials_retry)
    accepted_trials_strict = bool(
        accepted_required_columns_present
        and accepted_strictly_converged
        and accepted_cutback_count == 0
        and accepted_retry_count == 0
        and accepted_unconverged_count == 0
    )
    history_rows_strict = bool(history_required_columns_present and history_strictly_converged)
    gate_passed = int(
        unmatched_history_count == 0
        and history_all_accepted
        and bool(history_rows_strict)
        and bool(accepted_trials_strict)
        and bool(cutback_trials_consistent)
        and rejected_only_history_hits == 0
    )
    return {
        f"{label}_trial_gate_passed": gate_passed,
        f"{label}_trial_history_times_in_accepted_trials": int(unmatched_history_count == 0),
        f"{label}_trial_history_all_accepted": int(history_all_accepted),
        f"{label}_trial_history_required_columns_present": int(history_required_columns_present),
        f"{label}_trial_history_strictly_converged": int(history_strictly_converged),
        f"{label}_trial_accepted_required_columns_present": int(accepted_required_columns_present),
        f"{label}_trial_accepted_strictly_converged": int(accepted_strictly_converged),
        f"{label}_trial_accepted_no_cutback": int(accepted_cutback_count == 0),
        f"{label}_trial_accepted_no_retry": int(accepted_retry_count == 0),
        f"{label}_trial_accepted_no_unconverged": int(accepted_unconverged_count == 0),
        f"{label}_trial_cutback_trials_consistent": int(cutback_trials_consistent),
        f"{label}_trial_cutback_trials_rejected": int(cutback_trials_rejected),
        f"{label}_trial_cutback_trials_retry_required": int(cutback_trials_retry),
        f"{label}_trial_rejected_only_history_hits": int(rejected_only_history_hits),
        f"{label}_trial_accepted_count": int(len(accepted_trials)),
        f"{label}_trial_rejected_count": int(len(rejected_trials)),
        f"{label}_trial_cutback_count": int(len(cutback_trials)),
        f"{label}_trial_accepted_cutback_count": int(accepted_cutback_count),
        f"{label}_trial_accepted_retry_count": int(accepted_retry_count),
        f"{label}_trial_accepted_unconverged_count": int(accepted_unconverged_count),
        f"{label}_trial_history_row_count": int(len(history_rows)),
        f"{label}_trial_history_time_match_count": int(matched_count),
        f"{label}_trial_unmatched_history_count": int(unmatched_history_count),
        f"{label}_trial_accepted_history_time_linf": float(max_time_error),
        f"{label}_trial_last_history_time": float(history_times[-1]) if history_times else 0.0,
        f"{label}_trial_last_accepted_trial_time": float(accepted_trial_times[-1]) if accepted_trial_times else 0.0,
        f"{label}_trial_time_tolerance": float(tolerance),
    }
