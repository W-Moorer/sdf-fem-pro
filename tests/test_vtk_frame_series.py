from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.vtk_frame_series import write_hex_frame_series


def test_write_hex_frame_series_numbered_vtk_and_pvd(tmp_path: Path) -> None:
    points = np.asarray(
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
    elements = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    frames = [
        (0.0, np.zeros_like(points)),
        (0.1, np.full_like(points, 0.01)),
        (0.2, np.full_like(points, 0.02)),
    ]

    outputs = write_hex_frame_series(
        tmp_path,
        stem="demo",
        reference_points=points,
        elements=elements,
        frames=frames,
        frame_stride=1,
        cell_scalar_fn=lambda _time, _u: {"value": np.asarray([1.0])},
        source_label="unit-test",
    )

    assert outputs["frame_count"] == 3
    assert (tmp_path / "demo_0000.vtk").exists()
    assert (tmp_path / "demo_0001.vtk").exists()
    assert (tmp_path / "demo_0002.vtk").exists()
    assert Path(outputs["pvd"]).read_text(encoding="utf-8").count("<DataSet") == 3
