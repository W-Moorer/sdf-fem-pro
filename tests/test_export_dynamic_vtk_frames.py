from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.export_dynamic_vtk_frames import (  # noqa: E402
    _interpolate_displacement_frames,
    _select_by_stride,
    _with_initial_zero_frame,
)


def test_common_time_interpolation_uses_target_times() -> None:
    frames = [
        (0.0, np.zeros((2, 3), dtype=float)),
        (0.1, np.ones((2, 3), dtype=float)),
        (0.2, 3.0 * np.ones((2, 3), dtype=float)),
    ]

    interpolated = _interpolate_displacement_frames(frames, [0.05, 0.15])

    assert [time_value for time_value, _U in interpolated] == [0.05, 0.15]
    np.testing.assert_allclose(interpolated[0][1], 0.5 * np.ones((2, 3)))
    np.testing.assert_allclose(interpolated[1][1], 2.0 * np.ones((2, 3)))


def test_common_time_helpers_add_zero_and_keep_final_stride_frame() -> None:
    frames = [
        (0.1, np.ones((1, 3), dtype=float)),
        (0.2, 2.0 * np.ones((1, 3), dtype=float)),
        (0.3, 3.0 * np.ones((1, 3), dtype=float)),
    ]

    with_zero = _with_initial_zero_frame(frames, node_count=1)
    selected = _select_by_stride(with_zero, 2)

    assert [time_value for time_value, _U in with_zero] == [0.0, 0.1, 0.2, 0.3]
    assert [time_value for time_value, _U in selected] == [0.0, 0.2, 0.3]
    np.testing.assert_allclose(with_zero[0][1], np.zeros((1, 3)))
