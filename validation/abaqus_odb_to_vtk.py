"""Export an Abaqus ODB frame sequence to legacy VTK files.

This script is intended to be executed with ``abaqus python`` because it imports
``odbAccess`` at runtime.  It is validation/post-processing code only and is not
imported by the core SFC package.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


VTK_CELL_TYPES = {
    "C3D4": 10,
    "C3D8": 12,
    "C3D8R": 12,
    "R3D3": 5,
    "R3D4": 9,
}


def _tensor_from_symmetric6(data: tuple[float, ...] | list[float]) -> tuple[tuple[float, float, float], ...]:
    """Return a full 3x3 tensor from Abaqus 3D symmetric tensor components."""

    if len(data) < 6:
        return (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    v11, v22, v33, v12, v13, v23 = (float(value) for value in data[:6])
    return (
        (v11, v12, v13),
        (v12, v22, v23),
        (v13, v23, v33),
    )


def _von_mises_from_symmetric6(data: tuple[float, ...] | list[float]) -> float:
    """Return von Mises stress from Abaqus 3D stress components."""

    if len(data) < 6:
        return 0.0
    s11, s22, s33, s12, s13, s23 = (float(value) for value in data[:6])
    return math.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
        + 3.0 * (s12 * s12 + s13 * s13 + s23 * s23)
    )


def _tensor_norm_from_symmetric6(data: tuple[float, ...] | list[float]) -> float:
    if len(data) < 6:
        return 0.0
    s11, s22, s33, s12, s13, s23 = (float(value) for value in data[:6])
    return math.sqrt(s11 * s11 + s22 * s22 + s33 * s33 + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23))


def _equivalent_elastic_strain_from_mises(mises: float, *, young: float | None, poisson: float | None) -> float:
    """Return the J2 equivalent elastic strain implied by von Mises stress."""

    if young is None or poisson is None:
        return 0.0
    shear = float(young) / (2.0 * (1.0 + float(poisson)))
    return float(mises) / max(3.0 * shear, 1.0e-30)


def _average_element_field(frame, field_name: str) -> dict[tuple[str, int], tuple[float, ...]]:  # noqa: ANN001
    """Average an element field over all values reported for each element."""

    if field_name not in frame.fieldOutputs:
        return {}
    sums: dict[tuple[str, int], list[float]] = {}
    counts: dict[tuple[str, int], int] = {}
    for value in frame.fieldOutputs[field_name].values:
        if not hasattr(value, "elementLabel") or value.elementLabel is None:
            continue
        key = (value.instance.name, int(value.elementLabel))
        data = tuple(float(component) for component in value.data)
        if key not in sums:
            sums[key] = [0.0] * len(data)
            counts[key] = 0
        for index, component in enumerate(data):
            sums[key][index] += component
        counts[key] += 1
    return {key: tuple(component / counts[key] for component in data) for key, data in sums.items()}


def _node_vector_field(frame, field_name: str) -> dict[tuple[str, int], tuple[float, float, float]]:  # noqa: ANN001
    """Return a nodal vector field keyed by instance name and node label."""

    if field_name not in frame.fieldOutputs:
        return {}
    values: dict[tuple[str, int], tuple[float, float, float]] = {}
    for value in frame.fieldOutputs[field_name].values:
        if not hasattr(value, "nodeLabel") or value.nodeLabel is None:
            continue
        if getattr(value, "instance", None) is None:
            continue
        data = tuple(float(component) for component in value.data)
        if len(data) < 3:
            continue
        values[(value.instance.name, int(value.nodeLabel))] = (data[0], data[1], data[2])
    return values


def _write_pvd(path: Path, datasets: list[tuple[float, Path]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for time_value, frame_path in datasets:
        try:
            rel_path = frame_path.resolve().relative_to(path.parent.resolve()).as_posix()
        except ValueError:
            rel_path = frame_path.as_posix()
        lines.append(f'    <DataSet timestep="{time_value:.12e}" group="" part="0" file="{escape(rel_path)}"/>')
    lines.extend(["  </Collection>", "</VTKFile>"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _object_id_from_instance(instance_name: str) -> int:
    """Return a stable visualization id from an Abaqus instance name."""

    upper = instance_name.upper()
    if "GEAR1" in upper:
        return 1
    if "GEAR2" in upper:
        return 2
    if "PLANE" in upper or "GROUND" in upper or "RIGID" in upper:
        return 0
    if "CUBE" in upper or "BLOCK" in upper or "BEAM" in upper or "FLEX" in upper:
        return 1
    if "SPHERE" in upper or "BALL" in upper:
        return 2
    return 1


def _write_vtk_frame(
    path: Path,
    *,
    title: str,
    points: list[tuple[float, float, float]],
    cells: list[list[int]],
    cell_types: list[int],
    object_ids: list[int],
    displacement: list[tuple[float, float, float]],
    velocity: list[tuple[float, float, float]],
    stress: list[tuple[float, ...]],
    strain: list[tuple[float, ...]],
    young: float | None = None,
    poisson: float | None = None,
    include_tensors: bool = True,
) -> None:
    cell_size = sum(len(cell) + 1 for cell in cells)
    von_mises = [_von_mises_from_symmetric6(value) for value in stress]
    strain_norm = [_tensor_norm_from_symmetric6(value) for value in strain]
    equivalent_strain = [_equivalent_elastic_strain_from_mises(value, young=young, poisson=poisson) for value in von_mises]
    von_mises_nodeavg = _node_average_cell_scalar(von_mises, cells, len(points))
    strain_norm_nodeavg = _node_average_cell_scalar(strain_norm, cells, len(points))
    equivalent_strain_nodeavg = _node_average_cell_scalar(equivalent_strain, cells, len(points))
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(title + "\n")
        handle.write("ASCII\n")
        handle.write("DATASET UNSTRUCTURED_GRID\n")
        handle.write(f"POINTS {len(points)} float\n")
        for x, y, z in points:
            handle.write(f"{x:.9e} {y:.9e} {z:.9e}\n")
        handle.write(f"CELLS {len(cells)} {cell_size}\n")
        for cell in cells:
            handle.write(str(len(cell)) + " " + " ".join(str(node) for node in cell) + "\n")
        handle.write(f"CELL_TYPES {len(cell_types)}\n")
        for cell_type in cell_types:
            handle.write(f"{cell_type}\n")
        handle.write(f"POINT_DATA {len(points)}\n")
        handle.write("VECTORS U float\n")
        for ux, uy, uz in displacement:
            handle.write(f"{ux:.9e} {uy:.9e} {uz:.9e}\n")
        handle.write("VECTORS V float\n")
        for vx, vy, vz in velocity:
            handle.write(f"{vx:.9e} {vy:.9e} {vz:.9e}\n")
        handle.write("SCALARS displacement_magnitude float 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for ux, uy, uz in displacement:
            handle.write(f"{math.sqrt(ux * ux + uy * uy + uz * uz):.9e}\n")
        for name, values in (
            ("von_mises_nodeavg", von_mises_nodeavg),
            ("logarithmic_strain_norm_nodeavg", strain_norm_nodeavg),
            ("equivalent_elastic_strain_nodeavg", equivalent_strain_nodeavg),
        ):
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in values:
                handle.write(f"{float(value):.9e}\n")
        handle.write(f"CELL_DATA {len(cells)}\n")
        handle.write("SCALARS object_id int 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for object_id in object_ids:
            handle.write(f"{object_id}\n")
        handle.write("SCALARS von_mises float 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for value in von_mises:
            handle.write(f"{value:.9e}\n")
        handle.write("SCALARS logarithmic_strain_norm float 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for value in strain_norm:
            handle.write(f"{value:.9e}\n")
        handle.write("SCALARS equivalent_elastic_strain float 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for value in equivalent_strain:
            handle.write(f"{value:.9e}\n")
        if include_tensors:
            handle.write("TENSORS LE float\n")
            for value in strain:
                for row in _tensor_from_symmetric6(value):
                    handle.write(f"{row[0]:.9e} {row[1]:.9e} {row[2]:.9e}\n")
            handle.write("TENSORS S float\n")
            for value in stress:
                for row in _tensor_from_symmetric6(value):
                    handle.write(f"{row[0]:.9e} {row[1]:.9e} {row[2]:.9e}\n")


def _node_average_cell_scalar(cell_values: list[float], cells: list[list[int]], node_count: int) -> list[float]:
    """Return incident-cell averaged nodal scalars for VTK visualization."""

    out = [0.0] * int(node_count)
    counts = [0] * int(node_count)
    if len(cell_values) != len(cells):
        return out
    for value, cell in zip(cell_values, cells):
        for node in cell:
            out[int(node)] += float(value)
            counts[int(node)] += 1
    for index, count in enumerate(counts):
        if count:
            out[index] /= float(count)
    return out


def _object_manifest_metrics(
    *,
    object_ids: list[int],
    cells: list[list[int]],
    displacement_norm: list[float],
    von_mises_nodeavg: list[float],
    equivalent_strain_nodeavg: list[float],
) -> dict[str, float]:
    """Return per-object manifest metrics from cell object ids."""

    out: dict[str, float] = {}
    unique_object_ids = sorted(set(int(value) for value in object_ids))
    for object_id in unique_object_ids:
        nodes: set[int] = set()
        for cell_object_id, cell in zip(object_ids, cells):
            if int(cell_object_id) == object_id:
                nodes.update(int(node) for node in cell)
        if not nodes:
            continue
        node_list = sorted(nodes)
        out[f"max_displacement_magnitude_object{object_id}"] = max(float(displacement_norm[node]) for node in node_list)
        out[f"max_von_mises_nodeavg_object{object_id}"] = max(float(von_mises_nodeavg[node]) for node in node_list)
        out[f"max_equivalent_elastic_strain_nodeavg_object{object_id}"] = max(
            float(equivalent_strain_nodeavg[node]) for node in node_list
        )
    return out


def export_odb_to_vtk(
    odb_path: Path,
    out_dir: Path,
    *,
    stem: str = "frame",
    frame_stride: int = 1,
    include_tensors: bool = True,
    young: float | None = None,
    poisson: float | None = None,
) -> dict[str, Path | int | float]:
    """Export all ODB frames in the first step to numbered VTK files."""

    from odbAccess import openOdb  # type: ignore[import-not-found]

    stride = max(1, int(frame_stride))
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{stem}_*.vtk"):
        stale.unlink()

    odb = openOdb(path=str(odb_path), readOnly=True)
    try:
        assembly = odb.rootAssembly
        point_index: dict[tuple[str, int], int] = {}
        reference_points: list[tuple[float, float, float]] = []
        for instance_name in sorted(assembly.instances.keys()):
            instance = assembly.instances[instance_name]
            for node in instance.nodes:
                point_index[(instance.name, int(node.label))] = len(reference_points)
                reference_points.append(tuple(float(value) for value in node.coordinates))

        cells: list[list[int]] = []
        cell_types: list[int] = []
        object_ids: list[int] = []
        cell_keys: list[tuple[str, int]] = []
        for instance_name in sorted(assembly.instances.keys()):
            instance = assembly.instances[instance_name]
            object_id = _object_id_from_instance(instance.name)
            for element in instance.elements:
                element_type = element.type.upper()
                if element_type not in VTK_CELL_TYPES:
                    raise ValueError(f"unsupported ODB element type {element.type!r}")
                cells.append([point_index[(instance.name, int(label))] for label in element.connectivity])
                cell_types.append(VTK_CELL_TYPES[element_type])
                object_ids.append(object_id)
                cell_keys.append((instance.name, int(element.label)))

        if not odb.steps:
            raise ValueError(f"ODB has no steps: {odb_path}")
        step = odb.steps[list(odb.steps.keys())[0]]
        datasets: list[tuple[float, Path]] = []
        manifest_rows: list[dict[str, str | int | float]] = []
        zero_vector = (0.0, 0.0, 0.0)
        zero_tensor = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        selected_frames = [
            (source_index, frame)
            for source_index, frame in enumerate(step.frames)
            if source_index % stride == 0 or source_index == len(step.frames) - 1
        ]
        for frame_index, (source_index, frame) in enumerate(selected_frames):
            displacement_by_node = _node_vector_field(frame, "U")
            velocity_by_node = _node_vector_field(frame, "V")
            stress_by_element = _average_element_field(frame, "S")
            strain_by_element = _average_element_field(frame, "LE")
            displacement: list[tuple[float, float, float]] = []
            velocity: list[tuple[float, float, float]] = []
            points: list[tuple[float, float, float]] = []
            for key, index in sorted(point_index.items(), key=lambda item: item[1]):
                reference_point = reference_points[index]
                u = displacement_by_node.get(key, zero_vector)
                v = velocity_by_node.get(key, zero_vector)
                points.append(
                    (
                        reference_point[0] + u[0],
                        reference_point[1] + u[1],
                        reference_point[2] + u[2],
                    )
                )
                displacement.append(u)
                velocity.append(v)
            stress = [stress_by_element.get(key, zero_tensor) for key in cell_keys]
            strain = [strain_by_element.get(key, zero_tensor) for key in cell_keys]
            displacement_norm = [math.sqrt(ux * ux + uy * uy + uz * uz) for ux, uy, uz in displacement]
            von_mises_values = [_von_mises_from_symmetric6(value) for value in stress]
            strain_norm_values = [_tensor_norm_from_symmetric6(value) for value in strain]
            equivalent_strain_values = [
                _equivalent_elastic_strain_from_mises(value, young=young, poisson=poisson) for value in von_mises_values
            ]
            von_mises_nodeavg = _node_average_cell_scalar(von_mises_values, cells, len(points))
            strain_norm_nodeavg = _node_average_cell_scalar(strain_norm_values, cells, len(points))
            equivalent_strain_nodeavg = _node_average_cell_scalar(equivalent_strain_values, cells, len(points))
            object_metrics = _object_manifest_metrics(
                object_ids=object_ids,
                cells=cells,
                displacement_norm=displacement_norm,
                von_mises_nodeavg=von_mises_nodeavg,
                equivalent_strain_nodeavg=equivalent_strain_nodeavg,
            )
            frame_path = out_dir / f"{stem}_{frame_index:04d}.vtk"
            _write_vtk_frame(
                frame_path,
                title=f"Abaqus ODB frame {frame_index} time={frame.frameValue:.12g}",
                points=points,
                cells=cells,
                cell_types=cell_types,
                object_ids=object_ids,
                displacement=displacement,
                velocity=velocity,
                stress=stress,
                strain=strain,
                young=young,
                poisson=poisson,
                include_tensors=include_tensors,
            )
            datasets.append((float(frame.frameValue), frame_path))
            manifest_row = {
                "frame": frame_index,
                "source_frame": source_index,
                "time": float(frame.frameValue),
                "vtk_file": frame_path.name,
                "node_count": len(points),
                "element_count": len(cells),
                "max_displacement_magnitude": max(displacement_norm, default=0.0),
                "max_von_mises": max(von_mises_values, default=0.0),
                "max_le_norm": max(strain_norm_values, default=0.0),
                "max_equivalent_elastic_strain": max(equivalent_strain_values, default=0.0),
                "max_von_mises_nodeavg": max(von_mises_nodeavg, default=0.0),
                "max_le_norm_nodeavg": max(strain_norm_nodeavg, default=0.0),
                "max_equivalent_elastic_strain_nodeavg": max(equivalent_strain_nodeavg, default=0.0),
            }
            manifest_row.update(object_metrics)
            manifest_rows.append(manifest_row)

        pvd_path = out_dir / f"{stem}.pvd"
        _write_pvd(pvd_path, datasets)
        manifest_path = out_dir / f"{stem}_manifest.csv"
        with manifest_path.open("w", encoding="ascii", newline="") as handle:
            fieldnames: list[str] = []
            for row in manifest_rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        return {
            "pvd": pvd_path,
            "manifest": manifest_path,
            "first_frame": datasets[0][1] if datasets else out_dir / f"{stem}_0000.vtk",
            "frame_count": len(datasets),
            "last_time": datasets[-1][0] if datasets else 0.0,
            "include_tensors": bool(include_tensors),
        }
    finally:
        odb.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odb", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default="frame")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--young", type=float, default=None)
    parser.add_argument("--poisson", type=float, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include-tensors", dest="include_tensors", action="store_true", default=True)
    group.add_argument("--scalars-only", dest="include_tensors", action="store_false")
    args = parser.parse_args()
    result = export_odb_to_vtk(
        args.odb,
        args.out_dir,
        stem=args.stem,
        frame_stride=args.frame_stride,
        include_tensors=bool(args.include_tensors),
        young=args.young,
        poisson=args.poisson,
    )
    print(f"VTK frames: {result['frame_count']}")
    print(f"PVD: {result['pvd']}")
    print(f"Manifest: {result['manifest']}")


if __name__ == "__main__":
    main()
