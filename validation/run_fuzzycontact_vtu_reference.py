"""FuzzyContact VTU dataset inventory and 3D field-audit preparation.

The FuzzyContact dataset is a free CC BY 4.0 Mendeley dataset containing VTU
solution files for contact mechanics cases. This runner records the public
dataset metadata and, when VTU files are available locally, parses their fields
and renders a three-dimensional surface preview suitable for paper-facing visual
audit.

The dataset is optional validation data. It is not a core dependency and is not
downloaded by default because the public zip is large.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF  # noqa: E402

MENDELEY_DATASET_API = "https://data.mendeley.com/public-api/datasets/tdgkkjyr2f"
DATASET_PAGE = "https://data.mendeley.com/datasets/tdgkkjyr2f/3"

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class VTUData:
    path: Path
    points: np.ndarray
    cells: list[np.ndarray]
    cell_types: list[int]
    point_data: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class VTUGroupKey:
    problem: str
    scenario: str
    body: str


@dataclass(frozen=True, slots=True)
class ReplayPair:
    problem: str
    scenario: str
    master_body: str
    slave_body: str
    master: VTUData
    slave: VTUData
    source_files: tuple[Path, ...]


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fetch_metadata() -> dict[str, Any]:
    request = Request(
        MENDELEY_DATASET_API,
        headers={
            "Accept": "application/json",
            "User-Agent": "sdf-fem-pro-validation/1.0 (+https://data.mendeley.com/datasets/tdgkkjyr2f/3)",
        },
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310 - public metadata endpoint
        return json.loads(response.read().decode("utf-8"))


def _metadata_fetch_error_rows(exc: BaseException) -> list[Row]:
    return [
        {
            "dataset_id": "tdgkkjyr2f",
            "dataset_name": "Dataset of FuzzyContact",
            "doi": "10.17632/tdgkkjyr2f.3",
            "version": "3",
            "publish_date": "",
            "licence": "CC BY 4.0",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/",
            "file_name": "",
            "file_size_bytes": "",
            "sha256": "",
            "download_url": "",
            "dataset_page": DATASET_PAGE,
            "claim_scope": "optional open VTU reference dataset; metadata fetch failed",
            "status": "metadata_fetch_failed",
            "error": str(exc),
        }
    ]


def _metadata_rows(metadata: dict[str, Any]) -> list[Row]:
    licence = metadata.get("data_licence") or {}
    rows: list[Row] = []
    for file_info in metadata.get("files", []):
        details = file_info.get("content_details") or {}
        rows.append(
            {
                "dataset_id": metadata.get("id", ""),
                "dataset_name": metadata.get("name", ""),
                "doi": (metadata.get("doi") or {}).get("id", ""),
                "version": metadata.get("version", ""),
                "publish_date": metadata.get("publish_date", ""),
                "licence": licence.get("short_name", ""),
                "licence_url": licence.get("url", ""),
                "file_name": file_info.get("filename", ""),
                "file_size_bytes": file_info.get("size", ""),
                "sha256": details.get("sha256_hash", ""),
                "download_url": details.get("download_url", ""),
                "dataset_page": DATASET_PAGE,
                "claim_scope": "optional open VTU reference dataset; not core dependency",
            }
        )
    return rows


def _download_dataset(metadata_rows: list[Row], download_dir: Path) -> Path:
    if not metadata_rows:
        raise RuntimeError("metadata does not contain downloadable files")
    row = metadata_rows[0]
    url = str(row["download_url"])
    if not url:
        raise RuntimeError("metadata does not contain a download_url")
    download_dir.mkdir(parents=True, exist_ok=True)
    path = download_dir / str(row["file_name"])
    if not path.exists():
        request = Request(
            url,
            headers={
                "Accept": "application/zip,application/octet-stream,*/*",
                "User-Agent": "sdf-fem-pro-validation/1.0 (+https://data.mendeley.com/datasets/tdgkkjyr2f/3)",
            },
        )
        part = path.with_suffix(path.suffix + ".part")
        with urlopen(request, timeout=180) as response, part.open("wb") as f:  # noqa: S310 - explicit public dataset download
            shutil.copyfileobj(response, f)
        part.replace(path)
    return path


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


def _parse_ascii_array(data_array: ET.Element, *, dtype: type = float) -> np.ndarray:
    fmt = (data_array.attrib.get("format") or "ascii").lower()
    if fmt != "ascii":
        raise ValueError(f"only ASCII VTU DataArray is supported, got format={fmt!r}")
    text = " ".join((data_array.text or "").split())
    if not text:
        return np.empty(0, dtype=dtype)
    return np.fromstring(text, sep=" ", dtype=dtype)


def _read_vtu(path: Path) -> VTUData:
    tree = ET.parse(path)
    piece = tree.getroot().find(".//Piece")
    if piece is None:
        raise ValueError(f"{path} does not contain a VTU Piece")
    points_array = piece.find("./Points/DataArray")
    if points_array is None:
        raise ValueError(f"{path} does not contain VTU points")
    points = _parse_ascii_array(points_array).reshape((-1, 3))

    cells_parent = piece.find("./Cells")
    if cells_parent is None:
        raise ValueError(f"{path} does not contain VTU cells")
    conn_array = cells_parent.find("./DataArray[@Name='connectivity']")
    offsets_array = cells_parent.find("./DataArray[@Name='offsets']")
    types_array = cells_parent.find("./DataArray[@Name='types']")
    if conn_array is None or offsets_array is None or types_array is None:
        raise ValueError(f"{path} has incomplete VTU cell arrays")
    connectivity = _parse_ascii_array(conn_array, dtype=np.int64)
    offsets = _parse_ascii_array(offsets_array, dtype=np.int64)
    cell_types = _parse_ascii_array(types_array, dtype=np.int64).astype(int).tolist()
    cells: list[np.ndarray] = []
    start = 0
    for offset in offsets:
        cells.append(connectivity[start : int(offset)].astype(np.int64))
        start = int(offset)

    point_data: dict[str, np.ndarray] = {}
    point_data_parent = piece.find("./PointData")
    if point_data_parent is not None:
        for data_array in point_data_parent.findall("./DataArray"):
            name = data_array.attrib.get("Name")
            if not name:
                continue
            components = int(data_array.attrib.get("NumberOfComponents", "1"))
            values = _parse_ascii_array(data_array)
            if values.size == 0:
                continue
            if components > 1:
                values = values.reshape((-1, components))
            point_data[name] = values
    return VTUData(path=path, points=points, cells=cells, cell_types=cell_types, point_data=point_data)


def _real_vtu_files(source_dir: Path, max_files: int) -> list[Path]:
    """Return real VTU files, skipping AppleDouble sidecar files from the zip."""

    files = [
        path
        for path in sorted(source_dir.rglob("*.vtu"))
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    ]
    return files[: max(1, int(max_files))]


def _surface_polygons(data: VTUData) -> list[np.ndarray]:
    faces: dict[tuple[int, ...], np.ndarray] = {}
    counts: dict[tuple[int, ...], int] = {}
    for cell, cell_type in zip(data.cells, data.cell_types, strict=True):
        local_faces: list[np.ndarray]
        if cell_type == 10 and cell.size == 4:  # VTK_TETRA
            local_faces = [cell[[0, 2, 1]], cell[[0, 1, 3]], cell[[1, 2, 3]], cell[[2, 0, 3]]]
        elif cell_type == 12 and cell.size == 8:  # VTK_HEXAHEDRON
            local_faces = [
                cell[[0, 1, 2, 3]],
                cell[[4, 7, 6, 5]],
                cell[[0, 4, 5, 1]],
                cell[[1, 5, 6, 2]],
                cell[[2, 6, 7, 3]],
                cell[[3, 7, 4, 0]],
            ]
        elif cell_type == 5 and cell.size == 3:  # VTK_TRIANGLE
            local_faces = [cell]
        elif cell_type == 9 and cell.size == 4:  # VTK_QUAD
            local_faces = [cell]
        else:
            continue
        for face in local_faces:
            key = tuple(sorted(int(v) for v in face))
            faces.setdefault(key, face.copy())
            counts[key] = counts.get(key, 0) + 1
    return [faces[key] for key, count in counts.items() if count == 1 or len(faces[key]) in (3, 4)]


def _field_scalar(data: VTUData) -> tuple[str, np.ndarray]:
    for preferred in ("displacement", "u", "U", "Displacement"):
        if preferred in data.point_data:
            values = np.asarray(data.point_data[preferred], dtype=float)
            if values.ndim == 2:
                return "displacement_magnitude", np.linalg.norm(values[:, : min(3, values.shape[1])], axis=1)
    vector_like = [(name, arr) for name, arr in data.point_data.items() if np.asarray(arr).ndim == 2]
    if vector_like:
        name, arr = vector_like[0]
        values = np.asarray(arr, dtype=float)
        return f"{name}_magnitude", np.linalg.norm(values[:, : min(3, values.shape[1])], axis=1)
    if data.point_data:
        name, arr = next(iter(data.point_data.items()))
        return name, np.asarray(arr, dtype=float).reshape((-1,))
    return "zero", np.zeros(data.points.shape[0], dtype=float)


def _inventory_rows(vtu_files: list[Path]) -> tuple[list[Row], list[Row], list[VTUData]]:
    inventory: list[Row] = []
    field_rows: list[Row] = []
    parsed: list[VTUData] = []
    for path in vtu_files:
        try:
            data = _read_vtu(path)
        except Exception as exc:  # pragma: no cover - defensive report path.
            inventory.append({"file": str(path), "status": "parse_failed", "error": str(exc)})
            continue
        parsed.append(data)
        inventory.append(
            {
                "file": str(path),
                "status": "parsed",
                "points": int(data.points.shape[0]),
                "cells": len(data.cells),
                "point_fields": ";".join(data.point_data.keys()),
            }
        )
        for name, values in data.point_data.items():
            arr = np.asarray(values, dtype=float)
            flat = arr.reshape((-1,))
            field_rows.append(
                {
                    "file": str(path),
                    "field": name,
                    "components": 1 if arr.ndim == 1 else arr.shape[1],
                    "min": float(np.min(flat)),
                    "max": float(np.max(flat)),
                    "l2_norm": float(np.linalg.norm(flat)),
                }
            )
    return inventory, field_rows, parsed


def _classify_vtu_path(path: Path) -> tuple[VTUGroupKey, str] | None:
    """Classify a FuzzyContact VTU into problem/scenario/body/field kind."""

    parts = [part.lower() for part in path.parts]
    problem = next((part for part in parts if part.startswith("problem_")), "")
    if not problem:
        return None

    stem = path.stem.lower()
    tokens = stem.split("_")
    body = ""
    for candidate in ("bottom", "top", "base", "indenter"):
        if candidate in tokens:
            body = candidate
            break
    if not body:
        return None

    if "displacement" in tokens:
        kind = "displacement"
        consumed = {"displacement", body}
    elif "strain" in tokens:
        kind = "strain"
        consumed = {"strain", body}
    elif "stress" in tokens:
        kind = "stress"
        consumed = {"stress", body}
    elif "mises" in tokens or "von" in tokens:
        kind = "mises"
        consumed = {"mises", "von", body}
    else:
        return None

    scenario = "_".join(token for token in tokens if token not in consumed) or "default"
    return VTUGroupKey(problem=problem, scenario=scenario, body=body), kind


def _merge_field_files(key: VTUGroupKey, paths_by_kind: dict[str, Path]) -> VTUData:
    """Merge displacement/strain/stress VTU point fields sharing one mesh."""

    ordered_kinds = [kind for kind in ("displacement", "strain", "stress", "mises") if kind in paths_by_kind]
    if not ordered_kinds:
        raise ValueError(f"{key} has no recognized field files")
    base = _read_vtu(paths_by_kind[ordered_kinds[0]])
    merged: dict[str, np.ndarray] = {}
    for kind in ordered_kinds:
        data = _read_vtu(paths_by_kind[kind])
        if data.points.shape != base.points.shape or not np.allclose(data.points, base.points, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{paths_by_kind[kind]} does not share coordinates with {base.path}")
        for name, values in data.point_data.items():
            unique_name = name
            if unique_name in merged:
                unique_name = f"{kind}:{name}"
            merged[unique_name] = np.asarray(values, dtype=float)
    return VTUData(path=base.path, points=base.points, cells=base.cells, cell_types=base.cell_types, point_data=merged)


def _replay_pairs(vtu_files: list[Path], *, max_groups: int) -> list[ReplayPair]:
    grouped: dict[VTUGroupKey, dict[str, Path]] = defaultdict(dict)
    for path in vtu_files:
        classified = _classify_vtu_path(path)
        if classified is None:
            continue
        key, kind = classified
        grouped[key][kind] = path

    by_case: dict[tuple[str, str], dict[str, tuple[VTUData, tuple[Path, ...]]]] = defaultdict(dict)
    for key, paths_by_kind in sorted(grouped.items(), key=lambda item: (item[0].problem, item[0].scenario, item[0].body)):
        try:
            data = _merge_field_files(key, paths_by_kind)
        except Exception:
            continue
        by_case[(key.problem, key.scenario)][key.body] = (data, tuple(paths_by_kind[kind] for kind in sorted(paths_by_kind)))

    pairs: list[ReplayPair] = []
    for (problem, scenario), bodies in sorted(by_case.items()):
        if {"bottom", "top"} <= set(bodies):
            master_body, slave_body = "bottom", "top"
        elif {"base", "indenter"} <= set(bodies):
            master_body, slave_body = "base", "indenter"
        else:
            continue
        master, master_files = bodies[master_body]
        slave, slave_files = bodies[slave_body]
        pairs.append(
            ReplayPair(
                problem=problem,
                scenario=scenario,
                master_body=master_body,
                slave_body=slave_body,
                master=master,
                slave=slave,
                source_files=master_files + slave_files,
            )
        )
        if len(pairs) >= max(1, int(max_groups)):
            break
    return pairs


def _surface_centroids(data: VTUData, polygons: list[np.ndarray]) -> np.ndarray:
    return np.asarray([np.mean(data.points[poly], axis=0) for poly in polygons], dtype=float)


def _contact_axis(master: VTUData, slave: VTUData) -> tuple[int, float]:
    master_center = np.mean(master.points, axis=0)
    slave_center = np.mean(slave.points, axis=0)
    delta = slave_center - master_center
    axis = int(np.argmax(np.abs(delta)))
    direction = 1.0 if delta[axis] >= 0.0 else -1.0
    return axis, direction


def _extreme_surface_polygons(data: VTUData, *, axis: int, side: str, quantile: float = 0.88) -> list[np.ndarray]:
    polygons = _surface_polygons(data)
    if not polygons:
        return []
    centroids = _surface_centroids(data, polygons)
    values = centroids[:, axis]
    if side == "max":
        threshold = float(np.quantile(values, quantile))
        selected = [poly for poly, value in zip(polygons, values, strict=True) if value >= threshold]
    else:
        threshold = float(np.quantile(values, 1.0 - quantile))
        selected = [poly for poly, value in zip(polygons, values, strict=True) if value <= threshold]
    return selected or polygons


def _triangulate_polygons(polygons: list[np.ndarray]) -> np.ndarray:
    triangles: list[np.ndarray] = []
    for poly in polygons:
        if poly.size < 3:
            continue
        if poly.size == 3:
            triangles.append(poly.astype(np.int64))
        else:
            first = int(poly[0])
            for i in range(1, int(poly.size) - 1):
                triangles.append(np.asarray([first, int(poly[i]), int(poly[i + 1])], dtype=np.int64))
    if not triangles:
        return np.empty((0, 3), dtype=np.int64)
    return np.vstack(triangles).astype(np.int64)


def _sample_polygons(data: VTUData, polygons: list[np.ndarray], max_samples: int) -> tuple[np.ndarray, list[np.ndarray]]:
    if not polygons:
        return np.empty((0, 3), dtype=float), []
    count = min(max(1, int(max_samples)), len(polygons))
    indices = np.unique(np.linspace(0, len(polygons) - 1, count, dtype=int))
    selected = [polygons[int(i)] for i in indices]
    points = _surface_centroids(data, selected)
    return points, selected


def _preferred_point_field(data: VTUData, kind: str) -> np.ndarray | None:
    key_tokens = {
        "displacement": ("displacement",),
        "strain": ("strain",),
        "stress": ("stress",),
        "mises": ("mises", "von"),
    }[kind]
    for name, values in data.point_data.items():
        lower = name.lower()
        if any(token in lower for token in key_tokens):
            return np.asarray(values, dtype=float)
    return None


def _value_norm(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return float(np.linalg.norm(arr.reshape((-1,))))


def _field_on_polygon(data: VTUData, polygon: np.ndarray, kind: str) -> float:
    values = _preferred_point_field(data, kind)
    if values is None:
        return math.nan
    nodal = values[np.asarray(polygon, dtype=np.int64)]
    if nodal.ndim == 1:
        return float(np.mean(nodal))
    return float(np.mean(np.linalg.norm(nodal, axis=1)))


def _field_from_face_payload(data: VTUData, face_nodes: np.ndarray, barycentric: np.ndarray, kind: str) -> float:
    values = _preferred_point_field(data, kind)
    if values is None:
        return math.nan
    nodal = values[np.asarray(face_nodes, dtype=np.int64)]
    interp = np.asarray(barycentric, dtype=float) @ nodal
    return _value_norm(interp)


def _master_payload_field_value(data: VTUData, payload: Any, kind: str) -> float:
    weighted = 0.0
    total_weight = 0.0
    for weight, nodes, bary in zip(payload.grid_weights, payload.face_node_ids, payload.barycentric, strict=True):
        value = _field_from_face_payload(data, nodes, bary, kind)
        if not math.isfinite(value):
            continue
        alpha = float(weight)
        weighted += alpha * value
        total_weight += alpha
    return weighted / total_weight if total_weight > 0.0 else math.nan


def _replay_spacing(master: VTUData) -> float:
    span = np.ptp(master.points, axis=0)
    diag = float(np.linalg.norm(span))
    return max(diag / 56.0, 1.0e-5)


def _map_replay_pair(pair: ReplayPair, *, max_samples: int) -> tuple[list[Row], Row | None]:
    master_polygons = _surface_polygons(pair.master)
    slave_polygons = _surface_polygons(pair.slave)
    if not master_polygons or not slave_polygons:
        return [], None

    axis, direction = _contact_axis(pair.master, pair.slave)
    master_side = "max" if direction >= 0.0 else "min"
    slave_side = "min" if direction >= 0.0 else "max"
    master_contact_polygons = _extreme_surface_polygons(pair.master, axis=axis, side=master_side)
    slave_contact_polygons = _extreme_surface_polygons(pair.slave, axis=axis, side=slave_side)
    master_triangles = _triangulate_polygons(master_contact_polygons)
    if master_triangles.size == 0:
        return [], None

    sample_points, sample_polygons = _sample_polygons(pair.slave, slave_contact_polygons, max_samples)
    if sample_points.size == 0:
        return [], None

    spacing = _replay_spacing(pair.master)
    axis_gap = abs(
        (float(np.min(sample_points[:, axis])) if direction >= 0.0 else float(np.max(sample_points[:, axis])))
        - (float(np.max(pair.master.points[:, axis])) if direction >= 0.0 else float(np.min(pair.master.points[:, axis])))
    )
    master_diag = float(np.linalg.norm(np.ptp(pair.master.points, axis=0)))
    band_radius = max(16.0 * spacing, axis_gap + 12.0 * spacing, 0.30 * master_diag)
    t0 = perf_counter()
    field = DynamicNarrowBandSDF.build_required_points(
        pair.master.points,
        master_triangles,
        sample_points,
        spacing=spacing,
        band_radius=band_radius,
        padding=band_radius,
        cell_size=max(spacing, band_radius),
    )
    build_seconds = perf_counter() - t0

    rows: list[Row] = []
    query_seconds = 0.0
    failed = 0
    for sample_id, (point, polygon) in enumerate(zip(sample_points, sample_polygons, strict=True)):
        try:
            q0 = perf_counter()
            phi = field.query_phi(point)
            normal = field.query_gradient(point)
            payload = field.query_payload(point)
            query_seconds += perf_counter() - q0
        except ValueError:
            failed += 1
            continue
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 1.0e-14:
            normal = normal / normal_norm
        rows.append(
            {
                "group_id": f"{pair.problem}:{pair.scenario}:{pair.master_body}->{pair.slave_body}",
                "problem": pair.problem,
                "scenario": pair.scenario,
                "master_body": pair.master_body,
                "slave_body": pair.slave_body,
                "sample_id": sample_id,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "phi": float(phi),
                "normal_x": float(normal[0]),
                "normal_y": float(normal[1]),
                "normal_z": float(normal[2]),
                "slave_displacement_magnitude": _field_on_polygon(pair.slave, polygon, "displacement"),
                "slave_strain_norm": _field_on_polygon(pair.slave, polygon, "strain"),
                "slave_stress_norm": _field_on_polygon(pair.slave, polygon, "stress"),
                "slave_von_mises": _field_on_polygon(pair.slave, polygon, "mises"),
                "master_displacement_magnitude": _master_payload_field_value(pair.master, payload, "displacement"),
                "master_strain_norm": _master_payload_field_value(pair.master, payload, "strain"),
                "master_stress_norm": _master_payload_field_value(pair.master, payload, "stress"),
                "master_von_mises": _master_payload_field_value(pair.master, payload, "mises"),
                "closest_face_id_min": int(np.min(payload.face_ids)),
                "closest_face_id_max": int(np.max(payload.face_ids)),
            }
        )

    if not rows:
        summary = {
            "group_id": f"{pair.problem}:{pair.scenario}:{pair.master_body}->{pair.slave_body}",
            "problem": pair.problem,
            "scenario": pair.scenario,
            "status": "no_valid_samples",
            "sample_count": 0,
            "failed_samples": failed,
        }
        return rows, summary

    phi_values = np.asarray([row["phi"] for row in rows], dtype=float)
    summary = {
        "group_id": f"{pair.problem}:{pair.scenario}:{pair.master_body}->{pair.slave_body}",
        "problem": pair.problem,
        "scenario": pair.scenario,
        "status": "mapped",
        "sample_count": len(rows),
        "failed_samples": failed,
        "axis": axis,
        "spacing": spacing,
        "band_radius": band_radius,
        "master_contact_triangles": int(master_triangles.shape[0]),
        "slave_contact_polygons": len(slave_contact_polygons),
        "grid_node_count": field.stats.grid_node_count,
        "valid_node_count": field.stats.valid_node_count,
        "field_build_seconds": build_seconds,
        "field_query_seconds_total": query_seconds,
        "field_query_seconds_per_sample": query_seconds / max(len(rows), 1),
        "phi_min": float(np.min(phi_values)),
        "phi_mean": float(np.mean(phi_values)),
        "phi_max": float(np.max(phi_values)),
        "active_penetration_samples_phi_negative": int(np.count_nonzero(phi_values < 0.0)),
        "source_files": ";".join(str(path) for path in pair.source_files),
    }
    return rows, summary


def _plot_replay_samples(path_png: Path, path_pdf: Path, pair: ReplayPair, rows: list[Row]) -> Row:
    if not rows:
        raise ValueError("no replay rows to plot")
    master_polygons = _surface_polygons(pair.master)
    slave_polygons = _surface_polygons(pair.slave)
    sample_points = np.asarray([[row["x"], row["y"], row["z"]] for row in rows], dtype=float)
    phi = np.asarray([row["phi"] for row in rows], dtype=float)
    fig = plt.figure(figsize=(7.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    for polygons, data, color, alpha in (
        (master_polygons, pair.master, (0.35, 0.45, 0.62, 0.18), 0.18),
        (slave_polygons, pair.slave, (0.72, 0.46, 0.30, 0.16), 0.16),
    ):
        if polygons:
            shown = polygons[:: max(1, len(polygons) // 900)]
            collection = Poly3DCollection(
                [data.points[poly] for poly in shown],
                facecolors=color,
                edgecolors=(0.1, 0.1, 0.1, 0.05),
                linewidths=0.04,
                alpha=alpha,
            )
            ax.add_collection3d(collection)
    scatter = ax.scatter(
        sample_points[:, 0],
        sample_points[:, 1],
        sample_points[:, 2],
        c=phi,
        cmap="coolwarm",
        s=18,
        depthshade=False,
    )
    all_points = np.vstack((pair.master.points, pair.slave.points))
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    padding = np.maximum(0.04 * (maxs - mins), 1.0e-8)
    ax.set_xlim(float(mins[0] - padding[0]), float(maxs[0] + padding[0]))
    ax.set_ylim(float(mins[1] - padding[1]), float(maxs[1] + padding[1]))
    ax.set_zlim(float(mins[2] - padding[2]), float(maxs[2] + padding[2]))
    ax.set_box_aspect((maxs - mins + 2.0 * padding).clip(min=1.0e-12))
    ax.view_init(elev=24.0, azim=-46.0)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.set_title(f"SFC dynamic SDF replay: {pair.problem} {pair.scenario}", fontsize=10)
    ax.set_axis_off()
    fig.colorbar(scatter, ax=ax, shrink=0.62, pad=0.02, label="interpolated gap phi")
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=190, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    return {
        "plot": "fuzzycontact_sfc_replay_samples",
        "png": str(path_png),
        "pdf": str(path_pdf),
        "field": "interpolated gap phi",
        "source_file": ";".join(str(path) for path in pair.source_files),
    }


def _sfc_replay_rows(vtu_files: list[Path], out_dir: Path, *, max_groups: int, max_samples: int) -> tuple[list[Row], list[Row], list[Row]]:
    sample_rows: list[Row] = []
    summary_rows: list[Row] = []
    plot_rows: list[Row] = []
    for pair in _replay_pairs(vtu_files, max_groups=max_groups):
        rows, summary = _map_replay_pair(pair, max_samples=max_samples)
        sample_rows.extend(rows)
        if summary is not None:
            summary_rows.append(summary)
        if rows and not plot_rows:
            plot_rows.append(
                _plot_replay_samples(
                    out_dir / "figures" / "fuzzycontact_sfc_replay_samples.png",
                    out_dir / "figures" / "fuzzycontact_sfc_replay_samples.pdf",
                    pair,
                    rows,
                )
            )
    return sample_rows, summary_rows, plot_rows


def _plot_preview(path_png: Path, path_pdf: Path, data: VTUData) -> Row:
    polygons = _surface_polygons(data)
    field_name, values = _field_scalar(data)
    if not polygons:
        raise ValueError("no renderable surface polygons found")
    poly_points = [data.points[face] for face in polygons]
    poly_values = np.asarray([float(np.mean(values[face])) for face in polygons], dtype=float)
    norm = Normalize(vmin=float(np.min(poly_values)), vmax=max(float(np.max(poly_values)), float(np.min(poly_values)) + 1.0e-14))
    fig = plt.figure(figsize=(6.2, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(poly_points, cmap="viridis", norm=norm, linewidths=0.08, edgecolors=(0.1, 0.1, 0.1, 0.18))
    collection.set_array(poly_values)
    ax.add_collection3d(collection)
    mins = np.min(data.points, axis=0)
    maxs = np.max(data.points, axis=0)
    padding = np.maximum(0.06 * (maxs - mins), 1.0e-8)
    ax.set_xlim(float(mins[0] - padding[0]), float(maxs[0] + padding[0]))
    ax.set_ylim(float(mins[1] - padding[1]), float(maxs[1] + padding[1]))
    ax.set_zlim(float(mins[2] - padding[2]), float(maxs[2] + padding[2]))
    ax.set_box_aspect((maxs - mins + 2.0 * padding).clip(min=1.0e-12))
    ax.view_init(elev=24.0, azim=-48.0)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.set_title(f"FuzzyContact VTU preview: {field_name}", fontsize=10)
    ax.set_axis_off()
    fig.colorbar(collection, ax=ax, shrink=0.62, pad=0.02, label=field_name)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    return {"plot": "fuzzycontact_vtu_preview", "png": str(path_png), "pdf": str(path_pdf), "field": field_name, "source_file": str(data.path)}


def _write_summary(
    path: Path,
    metadata_rows: list[Row],
    inventory: list[Row],
    fields: list[Row],
    plots: list[Row],
    replay_summary: list[Row],
) -> None:
    dataset_row = metadata_rows[0] if metadata_rows else {}
    parsed_count = sum(1 for row in inventory if row.get("status") == "parsed")
    lines = [
        "# FuzzyContact VTU Reference Dataset Handoff",
        "",
        "This validation line prepares the free CC BY 4.0 FuzzyContact Mendeley dataset as an external VTU field reference.",
        "The data is optional validation material and is not a core solver dependency.",
        "",
        "## Dataset",
        "",
        f"- page: `{DATASET_PAGE}`",
        f"- DOI: `{dataset_row.get('doi', '')}`",
        f"- license: `{dataset_row.get('licence', '')}` ({dataset_row.get('licence_url', '')})",
        f"- advertised file: `{dataset_row.get('file_name', '')}`",
        f"- advertised size: `{dataset_row.get('file_size_bytes', '')}` bytes",
        "",
        "## Local Parse Status",
        "",
        f"- parsed VTU files: `{parsed_count}`",
        f"- inventoried fields: `{len(fields)}`",
        f"- SFC replay groups mapped: `{sum(1 for row in replay_summary if row.get('status') == 'mapped')}`",
        "",
        "## Claim Scope",
        "",
        "- Supports open dataset availability and VTU field parsing when local files are provided.",
        "- Supports 3D visual field audit when at least one VTU file is parsed.",
        "- Maps external displacement/strain/stress fields onto SFC dynamic-SDF replay samples when paired bodies are present.",
        "- SFC replay samples use interpolation-only field queries; closest-point projection remains a field-build kernel.",
        "- Does not claim full FuzzyContact solver equivalence or frictional contact equivalence.",
    ]
    if replay_summary:
        lines.extend(["", "## SFC Replay Mapping", ""])
        for row in replay_summary:
            lines.append(
                "- "
                f"`{row.get('group_id', '')}`: status `{row.get('status', '')}`, "
                f"samples `{row.get('sample_count', '')}`, "
                f"phi range [`{row.get('phi_min', '')}`, `{row.get('phi_max', '')}`], "
                f"build seconds `{row.get('field_build_seconds', '')}`"
            )
    if plots:
        lines.extend(["", "## Figures", ""])
        for plot in plots:
            lines.append(f"- `{Path(str(plot['png'])).name}` from `{plot['source_file']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_reference(
    out_dir: Path,
    *,
    source_dir: Path | None = None,
    download: bool = False,
    extract: bool = False,
    max_files: int = 4,
    max_replay_groups: int = 3,
    max_replay_samples: int = 80,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        metadata = _fetch_metadata()
        metadata_rows = _metadata_rows(metadata)
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        metadata_rows = _metadata_fetch_error_rows(exc)
    if download:
        zip_path = _download_dataset(metadata_rows, out_dir / "downloads")
        if extract:
            _extract_zip(zip_path, out_dir / "downloads" / "Dataset_for_Solutions")
            source_dir = out_dir / "downloads" / "Dataset_for_Solutions"

    vtu_files: list[Path] = []
    if source_dir is not None and source_dir.exists():
        vtu_files = _real_vtu_files(source_dir, max_files)
    inventory, field_rows, parsed = _inventory_rows(vtu_files)
    plot_rows: list[Row] = []
    if parsed:
        plot_rows.append(_plot_preview(out_dir / "figures" / "fuzzycontact_vtu_preview.png", out_dir / "figures" / "fuzzycontact_vtu_preview.pdf", parsed[0]))
    else:
        inventory.append(
            {
                "file": "",
                "status": "metadata_only_pending_vtu_source",
                "error": "provide --source-dir with extracted VTU files or use --download --extract",
            }
        )
    replay_rows: list[Row] = []
    replay_summary: list[Row] = []
    if vtu_files:
        replay_rows, replay_summary, replay_plots = _sfc_replay_rows(
            vtu_files,
            out_dir,
            max_groups=max_replay_groups,
            max_samples=max_replay_samples,
        )
        plot_rows.extend(replay_plots)

    outputs = {
        "metadata": out_dir / "fuzzycontact_dataset_metadata.csv",
        "inventory": out_dir / "fuzzycontact_vtu_inventory.csv",
        "fields": out_dir / "fuzzycontact_vtu_field_summary.csv",
        "sfc_replay_samples": out_dir / "fuzzycontact_sfc_replay_samples.csv",
        "sfc_replay_summary": out_dir / "fuzzycontact_sfc_replay_summary.csv",
        "plots": out_dir / "fuzzycontact_vtu_plots.csv",
        "summary": out_dir / "fuzzycontact_vtu_reference_summary.md",
    }
    _write_csv(outputs["metadata"], metadata_rows)
    _write_csv(outputs["inventory"], inventory)
    _write_csv(outputs["fields"], field_rows)
    _write_csv(outputs["sfc_replay_samples"], replay_rows)
    _write_csv(outputs["sfc_replay_summary"], replay_summary)
    _write_csv(outputs["plots"], plot_rows)
    _write_summary(outputs["summary"], metadata_rows, inventory, field_rows, plot_rows, replay_summary)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "fuzzycontact_vtu_reference")
    parser.add_argument("--source-dir", type=Path, default=None, help="Directory containing extracted FuzzyContact VTU files.")
    parser.add_argument("--download", action="store_true", help="Download the public dataset zip. The file is large.")
    parser.add_argument("--extract", action="store_true", help="Extract the downloaded zip and parse VTU files.")
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--max-replay-groups", type=int, default=3)
    parser.add_argument("--max-replay-samples", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_reference(
        args.out_dir,
        source_dir=args.source_dir,
        download=bool(args.download),
        extract=bool(args.extract),
        max_files=int(args.max_files),
        max_replay_groups=int(args.max_replay_groups),
        max_replay_samples=int(args.max_replay_samples),
    )
    print("FuzzyContact VTU reference preparation complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
