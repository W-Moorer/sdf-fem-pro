from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sfc.contact.tracking_state import (
    accepted_contact_response_can_reuse,
    commit_accepted_contact_tracking_from_sample_arrays,
    restore_contact_tracking_state,
    run_contact_tracking_trial,
    snapshot_contact_tracking_state,
)


def test_tracking_snapshot_restore_includes_triangle_q4_and_oracle_caches() -> None:
    oracle = SimpleNamespace(
        _patch_cache={"region-a": 7},
        _barycentric_cache={"region-a": np.asarray([0.2, 0.3, 0.5], dtype=float)},
    )
    geometry = SimpleNamespace(
        _secondary_face_cache=np.asarray([1, 2, -1], dtype=np.int64),
        _secondary_barycentric_cache=np.asarray(
            [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [np.nan, np.nan, np.nan]],
            dtype=float,
        ),
        _secondary_region_face_cache={11: 5, 12: 6},
        _secondary_region_barycentric_cache={
            11: np.asarray([0.2, 0.3, 0.5], dtype=float),
            12: np.asarray([0.4, 0.4, 0.2], dtype=float),
        },
        _secondary_master_weight_cache=np.asarray(
            [[0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1], [np.nan, np.nan, np.nan, np.nan]],
            dtype=float,
        ),
        _oracle=oracle,
    )

    snapshot = snapshot_contact_tracking_state([geometry])
    geometry._secondary_face_cache[:] = [9, 9, 9]
    geometry._secondary_barycentric_cache[:] = 0.0
    geometry._secondary_region_face_cache[11] = 99
    geometry._secondary_region_face_cache[13] = 101
    geometry._secondary_region_barycentric_cache[11][:] = [1.0, 0.0, 0.0]
    geometry._secondary_master_weight_cache[:] = 1.0
    oracle._patch_cache["region-a"] = 99
    oracle._patch_cache["trial-only"] = 123
    oracle._barycentric_cache["region-a"][:] = [1.0, 0.0, 0.0]

    restore_contact_tracking_state(snapshot)

    np.testing.assert_array_equal(geometry._secondary_face_cache, [1, 2, -1])
    np.testing.assert_allclose(geometry._secondary_barycentric_cache[:2], [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    assert np.isnan(geometry._secondary_barycentric_cache[2]).all()
    assert geometry._secondary_region_face_cache == {11: 5, 12: 6}
    np.testing.assert_allclose(geometry._secondary_region_barycentric_cache[11], [0.2, 0.3, 0.5])
    np.testing.assert_allclose(geometry._secondary_region_barycentric_cache[12], [0.4, 0.4, 0.2])
    np.testing.assert_allclose(
        geometry._secondary_master_weight_cache[:2],
        [[0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1]],
    )
    assert np.isnan(geometry._secondary_master_weight_cache[2]).all()
    assert oracle._patch_cache == {"region-a": 7}
    np.testing.assert_allclose(oracle._barycentric_cache["region-a"], [0.2, 0.3, 0.5])


def test_tracking_trial_rolls_back_mutation_but_returns_query_result() -> None:
    geometry = SimpleNamespace(_secondary_face_cache=np.asarray([4], dtype=np.int64), _oracle=None)

    def query() -> str:
        geometry._secondary_face_cache[0] = 10
        return "trial-response"

    assert run_contact_tracking_trial([geometry], query) == "trial-response"
    np.testing.assert_array_equal(geometry._secondary_face_cache, [4])


class _SlottedFaceOnlyGeometry:
    __slots__ = ("_secondary_face_cache", "_oracle")

    def __init__(self) -> None:
        self._secondary_face_cache = np.asarray([3], dtype=np.int64)
        self._oracle = None


def test_tracking_restore_skips_missing_optional_slot_cache() -> None:
    geometry = _SlottedFaceOnlyGeometry()

    def query() -> str:
        geometry._secondary_face_cache[0] = 8
        return "ok"

    assert run_contact_tracking_trial([geometry], query) == "ok"
    np.testing.assert_array_equal(geometry._secondary_face_cache, [3])


class _CommittableGeometry:
    def __init__(self) -> None:
        self.committed: dict[str, np.ndarray] | None = None

    def commit_secondary_tracking_from_sample_arrays(self, sample_arrays: dict[str, np.ndarray]) -> int:
        self.committed = sample_arrays
        return int(np.asarray(sample_arrays["master_face_ids"]).size)


def test_accepted_commit_requires_partitioned_arrays_for_multiple_geometries() -> None:
    a = _CommittableGeometry()
    b = _CommittableGeometry()
    arrays = {"master_face_ids": np.asarray([1, 2], dtype=np.int64)}

    assert commit_accepted_contact_tracking_from_sample_arrays([a, b], arrays) == 0
    assert a.committed is None
    assert b.committed is None

    assert commit_accepted_contact_tracking_from_sample_arrays([a, b], [arrays, None]) == 2
    assert a.committed is not None
    assert b.committed is None


def test_accepted_response_reuse_requires_converged_final_trial_arrays() -> None:
    assert accepted_contact_response_can_reuse(converged=True, sample_arrays_present=True)
    assert not accepted_contact_response_can_reuse(converged=False, sample_arrays_present=True)
    assert not accepted_contact_response_can_reuse(converged=True, sample_arrays_present=False)
