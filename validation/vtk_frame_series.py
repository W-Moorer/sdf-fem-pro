"""VTK/PVD helpers for dynamic C3D8 frame sequences."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


def _array(values: np.ndarray, *, width: int) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim == 1 and width == 1:
        return data.reshape((-1, 1))
    if data.ndim != 2 or data.shape[1] != width:
        raise ValueError(f"expected array with width {width}, got {data.shape}")
    return data


def write_legacy_hex_vtk_frame(
    path: Path,
    points: np.ndarray,
    elements: np.ndarray,
    *,
    point_vectors: dict[str, np.ndarray] | None = None,
    point_scalars: dict[str, np.ndarray] | None = None,
    cell_scalars: dict[str, np.ndarray] | None = None,
    title: str = "SFC dynamic frame",
) -> None:
    """Write one ASCII legacy VTK HEX8 frame that ParaView can animate."""

    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(points, dtype=float)
    conn = np.asarray(elements, dtype=np.int64)
    point_vectors = point_vectors or {}
    point_scalars = point_scalars or {}
    cell_scalars = cell_scalars or {}
    lines: list[str] = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {X.shape[0]} float",
    ]
    lines.extend(f"{x:.12e} {y:.12e} {z:.12e}" for x, y, z in X)
    lines.append(f"CELLS {conn.shape[0]} {conn.shape[0] * 9}")
    lines.extend("8 " + " ".join(str(int(node)) for node in element) for element in conn)
    lines.append(f"CELL_TYPES {conn.shape[0]}")
    lines.extend("12" for _ in range(conn.shape[0]))
    if point_vectors or point_scalars:
        lines.append(f"POINT_DATA {X.shape[0]}")
        for name, values in point_vectors.items():
            data = _array(np.asarray(values, dtype=float), width=3)
            if data.shape[0] != X.shape[0]:
                raise ValueError(f"point vector {name!r} has wrong length")
            lines.append(f"VECTORS {name} float")
            lines.extend(f"{x:.12e} {y:.12e} {z:.12e}" for x, y, z in data)
        for name, values in point_scalars.items():
            data = _array(np.asarray(values, dtype=float), width=1).ravel()
            if data.shape[0] != X.shape[0]:
                raise ValueError(f"point scalar {name!r} has wrong length")
            lines.append(f"SCALARS {name} float 1")
            lines.append("LOOKUP_TABLE default")
            lines.extend(f"{float(value):.12e}" for value in data)
    if cell_scalars:
        lines.append(f"CELL_DATA {conn.shape[0]}")
        for name, values in cell_scalars.items():
            data = _array(np.asarray(values, dtype=float), width=1).ravel()
            if data.shape[0] != conn.shape[0]:
                raise ValueError(f"cell scalar {name!r} has wrong length")
            lines.append(f"SCALARS {name} float 1")
            lines.append("LOOKUP_TABLE default")
            lines.extend(f"{float(value):.12e}" for value in data)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pvd(path: Path, datasets: list[tuple[float, Path]]) -> None:
    """Write a ParaView collection file for a numbered frame sequence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for time_value, frame_path in datasets:
        try:
            rel = frame_path.resolve().relative_to(path.parent.resolve()).as_posix()
        except ValueError:
            rel = frame_path.as_posix()
        lines.append(f'    <DataSet timestep="{float(time_value):.12e}" group="" part="0" file="{escape(rel)}"/>')
    lines.extend(["  </Collection>", "</VTKFile>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hex_frame_series(
    out_dir: Path,
    *,
    stem: str,
    reference_points: np.ndarray,
    elements: np.ndarray,
    frames: list[tuple[float, np.ndarray]],
    frame_stride: int = 1,
    cell_scalar_fn=None,
    source_label: str,
) -> dict[str, Path | int]:
    """Write continuously numbered HEX8 VTK frames and a PVD time index."""

    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    X0 = np.asarray(reference_points, dtype=float)
    selected = [(time, U) for index, (time, U) in enumerate(frames) if index % frame_stride == 0]
    if frames and selected[-1][0] != frames[-1][0]:
        selected.append(frames[-1])
    datasets: list[tuple[float, Path]] = []
    for frame_index, (time_value, displacement) in enumerate(selected):
        U = np.asarray(displacement, dtype=float)
        points = X0 + U
        if cell_scalar_fn is None:
            cell_scalars = {}
        else:
            try:
                cell_scalars = cell_scalar_fn(float(time_value), U)
            except TypeError:
                cell_scalars = cell_scalar_fn(U)
        frame_path = out_dir / f"{stem}_{frame_index:04d}.vtk"
        write_legacy_hex_vtk_frame(
            frame_path,
            points,
            elements,
            point_vectors={"displacement": U},
            point_scalars={"displacement_magnitude": np.linalg.norm(U, axis=1)},
            cell_scalars=cell_scalars,
            title=f"{source_label} frame {frame_index} t={float(time_value):.12e}",
        )
        datasets.append((float(time_value), frame_path))
    pvd_path = out_dir / f"{stem}.pvd"
    write_pvd(pvd_path, datasets)
    return {
        "pvd": pvd_path,
        "first_frame": datasets[0][1] if datasets else out_dir / f"{stem}_0000.vtk",
        "frame_count": len(datasets),
    }
