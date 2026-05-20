"""Run an Abaqus/Explicit sphere-drop cantilever reference case.

The generated files live under the repository-root
``commercial_software_comparison`` directory by default.  This is an optional
commercial-solver comparison workflow and must not be imported by the core SFC
package.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "commercial_software_comparison" / "abaqus_sphere_cantilever"
JOB_NAME = "sphere_cantilever_explicit"
WALLCLOCK_RE = re.compile(r"WALLCLOCK TIME \(SEC\)\s*=\s*([0-9.]+)")
LINEAR_PENALTY_STIFFNESS = 5.0e9


@dataclass(frozen=True)
class ModelConfig:
    beam_length: float = 0.60
    beam_width: float = 0.08
    beam_height: float = 0.04
    beam_nx: int = 60
    beam_ny: int = 8
    beam_nz: int = 4
    beam_density: float = 1200.0
    beam_young: float = 2.5e9
    beam_poisson: float = 0.30
    sphere_radius: float = 0.045
    sphere_latitudes: int = 24
    sphere_longitudes: int = 48
    sphere_center_x: float = 0.40
    sphere_center_y: float = 0.0
    sphere_gap: float = 0.004
    sphere_density: float = 3500.0
    sphere_young: float = 5.0e6
    sphere_poisson: float = 0.30
    initial_velocity_z: float = -1.20
    gravity: float = 9.81
    duration: float = 3.0
    output_interval: float = 0.001
    fixed_dt: float = 1.0e-5
    contact_stiffness: float = LINEAR_PENALTY_STIFFNESS
    contact_damping_fraction: float | None = None

    @property
    def sphere_center_z(self) -> float:
        return 0.5 * self.beam_height + self.sphere_radius + self.sphere_gap


def _chunked_lines(values: list[int], *, width: int = 16) -> list[str]:
    return [", ".join(str(value) for value in values[index : index + width]) for index in range(0, len(values), width)]


def _format_node(label: int, xyz: tuple[float, float, float]) -> str:
    return f"{label}, {xyz[0]:.12e}, {xyz[1]:.12e}, {xyz[2]:.12e}"


def _format_element(label: int, connectivity: tuple[int, ...]) -> str:
    return f"{label}, " + ", ".join(str(node) for node in connectivity)


def _beam_mesh(cfg: ModelConfig) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]], list[int]]:
    nodes: list[tuple[float, float, float]] = []
    node_id: dict[tuple[int, int, int], int] = {}
    for k in range(cfg.beam_nz + 1):
        z = -0.5 * cfg.beam_height + cfg.beam_height * k / cfg.beam_nz
        for j in range(cfg.beam_ny + 1):
            y = -0.5 * cfg.beam_width + cfg.beam_width * j / cfg.beam_ny
            for i in range(cfg.beam_nx + 1):
                x = cfg.beam_length * i / cfg.beam_nx
                node_id[(i, j, k)] = len(nodes) + 1
                nodes.append((x, y, z))

    elements: list[tuple[int, ...]] = []
    for k in range(cfg.beam_nz):
        for j in range(cfg.beam_ny):
            for i in range(cfg.beam_nx):
                n000 = node_id[(i, j, k)]
                n100 = node_id[(i + 1, j, k)]
                n110 = node_id[(i + 1, j + 1, k)]
                n010 = node_id[(i, j + 1, k)]
                n001 = node_id[(i, j, k + 1)]
                n101 = node_id[(i + 1, j, k + 1)]
                n111 = node_id[(i + 1, j + 1, k + 1)]
                n011 = node_id[(i, j + 1, k + 1)]
                elements.append((n000, n100, n110, n010, n001, n101, n111, n011))
    fixed_nodes = [node_id[(0, j, k)] for k in range(cfg.beam_nz + 1) for j in range(cfg.beam_ny + 1)]
    return nodes, elements, fixed_nodes


def _sphere_surface_nodes(cfg: ModelConfig) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    import math

    if cfg.sphere_latitudes < 3:
        raise ValueError("sphere_latitudes must be at least 3")
    if cfg.sphere_longitudes < 6:
        raise ValueError("sphere_longitudes must be at least 6")

    center = (cfg.sphere_center_x, cfg.sphere_center_y, cfg.sphere_center_z)
    nodes: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    top = len(nodes) + 1
    nodes.append((center[0], center[1], center[2] + cfg.sphere_radius))
    for latitude in range(1, cfg.sphere_latitudes):
        theta = math.pi * latitude / cfg.sphere_latitudes
        z = center[2] + cfg.sphere_radius * math.cos(theta)
        radial = cfg.sphere_radius * math.sin(theta)
        ring: list[int] = []
        for longitude in range(cfg.sphere_longitudes):
            phi = 2.0 * math.pi * longitude / cfg.sphere_longitudes
            ring.append(len(nodes) + 1)
            nodes.append((center[0] + radial * math.cos(phi), center[1] + radial * math.sin(phi), z))
        rings.append(ring)
    bottom = len(nodes) + 1
    nodes.append((center[0], center[1], center[2] - cfg.sphere_radius))

    triangles: list[tuple[int, int, int]] = []
    first_ring = rings[0]
    for j in range(cfg.sphere_longitudes):
        triangles.append((top, first_ring[(j + 1) % cfg.sphere_longitudes], first_ring[j]))
    for lower_index in range(len(rings) - 1):
        upper = rings[lower_index]
        lower = rings[lower_index + 1]
        for j in range(cfg.sphere_longitudes):
            a = upper[j]
            b = upper[(j + 1) % cfg.sphere_longitudes]
            c = lower[j]
            d = lower[(j + 1) % cfg.sphere_longitudes]
            triangles.append((a, b, d))
            triangles.append((a, d, c))
    last_ring = rings[-1]
    for j in range(cfg.sphere_longitudes):
        triangles.append((bottom, last_ring[j], last_ring[(j + 1) % cfg.sphere_longitudes]))
    return nodes, triangles


def _det3(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> float:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _sphere_mesh(cfg: ModelConfig) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]], list[int]]:
    center = (cfg.sphere_center_x, cfg.sphere_center_y, cfg.sphere_center_z)
    surface_nodes, triangles = _sphere_surface_nodes(cfg)
    nodes = [center] + surface_nodes
    center_label = 1
    elements: list[tuple[int, ...]] = []
    for tri in triangles:
        a, b, c = (tri[0] + 1, tri[1] + 1, tri[2] + 1)
        va = _sub(nodes[a - 1], nodes[center_label - 1])
        vb = _sub(nodes[b - 1], nodes[center_label - 1])
        vc = _sub(nodes[c - 1], nodes[center_label - 1])
        if _det3(va, vb, vc) < 0.0:
            b, c = c, b
        elements.append((center_label, a, b, c))
    return nodes, elements, list(range(1, len(nodes) + 1))


def build_input_text(cfg: ModelConfig) -> str:
    """Build the Abaqus input deck for the sphere-drop cantilever case."""

    beam_nodes, beam_elements, beam_fixed_nodes = _beam_mesh(cfg)
    sphere_nodes, sphere_elements, sphere_all_nodes = _sphere_mesh(cfg)
    lines: list[str] = [
        "*Heading",
        "Commercial reference: deformable sphere impact on deformable cantilever",
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
        "*Material, name=BEAM_MAT",
        "*Density",
        f"{cfg.beam_density:.12e}",
        "*Elastic",
        f"{cfg.beam_young:.12e}, {cfg.beam_poisson:.12e}",
        "*Material, name=SPHERE_MAT",
        "*Density",
        f"{cfg.sphere_density:.12e}",
        "*Elastic",
        f"{cfg.sphere_young:.12e}, {cfg.sphere_poisson:.12e}",
        "*Surface Interaction, name=FRICTIONLESS_CONTACT",
        "*Surface Behavior, pressure-overclosure=LINEAR",
        f"{cfg.contact_stiffness:.12e}",
    ]
    if cfg.contact_damping_fraction is not None:
        lines.extend(
            [
                "*Contact Damping, definition=CRITICAL DAMPING FRACTION",
                f"{float(cfg.contact_damping_fraction):.12e}",
            ]
        )
    lines.extend(["*Friction", "0.", "*Part, name=BEAM", "*Node"])
    lines.extend(_format_node(label, xyz) for label, xyz in enumerate(beam_nodes, start=1))
    lines.append("*Element, type=C3D8R, elset=BEAM_EALL")
    lines.extend(_format_element(label, conn) for label, conn in enumerate(beam_elements, start=1))
    lines.extend(["*Elset, elset=BEAM_EALL, generate", f"1, {len(beam_elements)}, 1", "*Solid Section, elset=BEAM_EALL, material=BEAM_MAT", ","])
    lines.extend(["*End Part", "*Part, name=SPHERE", "*Node"])
    lines.extend(_format_node(label, xyz) for label, xyz in enumerate(sphere_nodes, start=1))
    lines.append("*Element, type=C3D4, elset=SPHERE_EALL")
    lines.extend(_format_element(label, conn) for label, conn in enumerate(sphere_elements, start=1))
    lines.extend(
        [
            "*Elset, elset=SPHERE_EALL, generate",
            f"1, {len(sphere_elements)}, 1",
            "*Solid Section, elset=SPHERE_EALL, material=SPHERE_MAT",
            ",",
            "*End Part",
            "*Assembly, name=ASSEMBLY",
            "*Instance, name=BEAM-1, part=BEAM",
            "*End Instance",
            "*Instance, name=SPHERE-1, part=SPHERE",
            "*End Instance",
            "*Nset, nset=BEAM_FIXED, instance=BEAM-1",
        ]
    )
    lines.extend(_chunked_lines(beam_fixed_nodes))
    lines.append("*Nset, nset=SPHERE_NODES, instance=SPHERE-1")
    lines.extend(_chunked_lines(sphere_all_nodes))
    lines.extend(
        [
            "*End Assembly",
            "*Initial Conditions, type=VELOCITY",
            f"SPHERE_NODES, 3, {cfg.initial_velocity_z:.12e}",
            "*Boundary",
            "BEAM_FIXED, ENCASTRE",
            "*Step, name=DROP_IMPACT, nlgeom=YES",
            "*Dynamic, Explicit, DIRECT USER CONTROL",
            f"{cfg.fixed_dt:.12e}, {cfg.duration:.12e}",
            "*Bulk Viscosity",
            "0., 0.",
            "*Dload",
            f"BEAM-1.BEAM_EALL, GRAV, {cfg.gravity:.12e}, 0., 0., -1.",
            f"SPHERE-1.SPHERE_EALL, GRAV, {cfg.gravity:.12e}, 0., 0., -1.",
            "*Contact",
            "*Contact Inclusions, ALL EXTERIOR",
            "*Contact Property Assignment",
            ", , FRICTIONLESS_CONTACT",
            f"*Output, field, time interval={cfg.output_interval:.12e}",
            "*Node Output",
            "U, V",
            "*Element Output, directions=YES",
            "S, LE",
            f"*Output, history, time interval={cfg.output_interval:.12e}",
            "*Energy Output",
            "ALLKE, ALLIE, ALLSE, ALLVD, ALLWK, ETOTAL",
            "*End Step",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_abaqus_command(command: str | None) -> str:
    if command:
        return command
    for candidate in ("abaqus", "abq2024"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Abaqus command not found; pass --abaqus-command with the local launcher path")


def _run_command(command: list[str], *, cwd: Path, log_path: Path) -> float:
    start = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    log_path.write_text(result.stdout, encoding="utf-8")
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}")
    return elapsed


def _write_model_manifest(path: Path, cfg: ModelConfig, *, beam_elements: int, sphere_elements: int) -> None:
    rows = [
        ("beam_length", cfg.beam_length),
        ("beam_width", cfg.beam_width),
        ("beam_height", cfg.beam_height),
        ("beam_elements", beam_elements),
        ("sphere_radius", cfg.sphere_radius),
        ("sphere_center_x", cfg.sphere_center_x),
        ("sphere_center_z", cfg.sphere_center_z),
        ("sphere_elements", sphere_elements),
        ("initial_velocity_z", cfg.initial_velocity_z),
        ("gravity", cfg.gravity),
        ("duration", cfg.duration),
        ("fixed_dt", cfg.fixed_dt),
        ("output_interval", cfg.output_interval),
        ("contact_stiffness", cfg.contact_stiffness),
        ("contact_damping_fraction", "" if cfg.contact_damping_fraction is None else cfg.contact_damping_fraction),
        ("beam_young", cfg.beam_young),
        ("sphere_young", cfg.sphere_young),
    ]
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerows(rows)


def _write_runtime_metrics(path: Path, metrics: dict[str, float | int | str]) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def _abaqus_reported_wallclock_seconds(sta_path: Path) -> float | None:
    """Return Abaqus' own reported Explicit wallclock seconds from a ``.sta`` file."""

    if not sta_path.exists():
        return None
    matches = WALLCLOCK_RE.findall(sta_path.read_text(encoding="utf-8", errors="ignore"))
    if not matches:
        return None
    return float(matches[-1])


def run_case(out_dir: Path, *, abaqus_command: str | None = None, cfg: ModelConfig | None = None) -> dict[str, Path | int]:
    total_start = time.perf_counter()
    cfg = ModelConfig() if cfg is None else cfg
    out_dir = Path(out_dir).resolve()
    run_dir = out_dir / "abaqus_run"
    vtk_dir = out_dir / "vtk"
    run_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir.mkdir(parents=True, exist_ok=True)

    inp_path = run_dir / f"{JOB_NAME}.inp"
    inp_path.write_text(build_input_text(cfg), encoding="ascii")
    beam_nodes, beam_elements, _ = _beam_mesh(cfg)
    sphere_nodes, sphere_elements, _ = _sphere_mesh(cfg)
    _write_model_manifest(out_dir / "model_manifest.csv", cfg, beam_elements=len(beam_elements), sphere_elements=len(sphere_elements))

    command = _resolve_abaqus_command(abaqus_command)
    analysis_log = out_dir / "abaqus_analysis_stdout.log"
    analysis_wall_seconds = _run_command(
        [command, f"job={JOB_NAME}", f"input={inp_path.name}", "interactive", "ask_delete=OFF"],
        cwd=run_dir,
        log_path=analysis_log,
    )

    odb_path = run_dir / f"{JOB_NAME}.odb"
    if not odb_path.exists():
        raise RuntimeError(f"Abaqus completed without producing {odb_path}")

    converter = Path(__file__).with_name("abaqus_odb_to_vtk.py")
    export_log = out_dir / "abaqus_odb_to_vtk_stdout.log"
    export_wall_seconds = _run_command(
        [command, "python", str(converter), "--odb", str(odb_path), "--out-dir", str(vtk_dir), "--stem", "frame"],
        cwd=run_dir,
        log_path=export_log,
    )

    frame_count = len(sorted(vtk_dir.glob("frame_*.vtk")))
    if frame_count == 0:
        raise RuntimeError(f"ODB export produced no VTK frames in {vtk_dir}")
    total_wall_seconds = time.perf_counter() - total_start
    metrics_path = out_dir / "runtime_metrics.csv"
    abaqus_reported_wall_seconds = _abaqus_reported_wallclock_seconds(run_dir / f"{JOB_NAME}.sta")
    _write_runtime_metrics(
        metrics_path,
        {
            "abaqus_analysis_wall_seconds": f"{analysis_wall_seconds:.6f}",
            "abaqus_reported_explicit_wallclock_seconds": (
                f"{abaqus_reported_wall_seconds:.6f}" if abaqus_reported_wall_seconds is not None else ""
            ),
            "odb_to_vtk_wall_seconds": f"{export_wall_seconds:.6f}",
            "total_workflow_wall_seconds": f"{total_wall_seconds:.6f}",
            "abaqus_model_duration_seconds": f"{cfg.duration:.6f}",
            "fixed_dt_seconds": f"{cfg.fixed_dt:.12e}",
            "field_output_interval_seconds": f"{cfg.output_interval:.6f}",
            "vtk_frame_count": frame_count,
            "beam_element_count": len(beam_elements),
            "sphere_element_count": len(sphere_elements),
            "beam_young": f"{cfg.beam_young:.12e}",
            "sphere_young": f"{cfg.sphere_young:.12e}",
            "contact_stiffness": f"{cfg.contact_stiffness:.12e}",
            "contact_damping_fraction": "" if cfg.contact_damping_fraction is None else f"{float(cfg.contact_damping_fraction):.12e}",
        },
    )
    summary = out_dir / "README.md"
    summary.write_text(
        "\n".join(
            [
                "# Abaqus Sphere-Cantilever Explicit Reference",
                "",
                "Commercial-solver comparison artifact generated by a validation-only workflow.",
                "",
                f"- Input deck: `{inp_path.relative_to(out_dir).as_posix()}`",
                f"- ODB: `{odb_path.relative_to(out_dir).as_posix()}`",
                "- Requested field output: `U`, `V`, `S`, `LE`.",
                f"- ParaView collection: `{(vtk_dir / 'frame.pvd').relative_to(out_dir).as_posix()}`",
                f"- Numbered VTK frames: `{frame_count}` files named `frame_0000.vtk`, `frame_0001.vtk`, ...",
                f"- Analysis stdout log: `{analysis_log.relative_to(out_dir).as_posix()}`",
                f"- ODB export stdout log: `{export_log.relative_to(out_dir).as_posix()}`",
                f"- Runtime metrics: `{metrics_path.relative_to(out_dir).as_posix()}`",
                f"- Abaqus analysis wall time: `{analysis_wall_seconds:.3f}` s.",
                f"- Abaqus-reported Explicit wallclock time: `{abaqus_reported_wall_seconds:.3f}` s."
                if abaqus_reported_wall_seconds is not None
                else "- Abaqus-reported Explicit wallclock time: unavailable.",
                f"- ODB-to-VTK wall time: `{export_wall_seconds:.3f}` s.",
                f"- Total workflow wall time: `{total_wall_seconds:.3f}` s.",
                "",
                "Object ids in VTK cell data: `1` = cantilever beam, `2` = sphere.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "out_dir": out_dir,
        "inp": inp_path,
        "odb": odb_path,
        "pvd": vtk_dir / "frame.pvd",
        "frame_count": frame_count,
        "summary": summary,
        "runtime_metrics": metrics_path,
        "beam_node_count": len(beam_nodes),
        "sphere_node_count": len(sphere_nodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--abaqus-command", default=None)
    parser.add_argument("--duration", type=float, default=ModelConfig.duration)
    parser.add_argument("--output-interval", type=float, default=ModelConfig.output_interval)
    parser.add_argument("--fixed-dt", type=float, default=ModelConfig.fixed_dt)
    parser.add_argument("--beam-young", type=float, default=ModelConfig.beam_young)
    parser.add_argument("--sphere-young", type=float, default=ModelConfig.sphere_young)
    parser.add_argument("--contact-stiffness", type=float, default=ModelConfig.contact_stiffness)
    parser.add_argument("--contact-damping-fraction", type=float, default=None)
    args = parser.parse_args()
    cfg = ModelConfig(
        duration=float(args.duration),
        output_interval=float(args.output_interval),
        fixed_dt=float(args.fixed_dt),
        beam_young=float(args.beam_young),
        sphere_young=float(args.sphere_young),
        contact_stiffness=float(args.contact_stiffness),
        contact_damping_fraction=args.contact_damping_fraction,
    )
    result = run_case(args.out_dir, abaqus_command=args.abaqus_command, cfg=cfg)
    print(f"Output directory: {result['out_dir']}")
    print(f"Input deck: {result['inp']}")
    print(f"ODB: {result['odb']}")
    print(f"PVD: {result['pvd']}")
    print(f"VTK frame count: {result['frame_count']}")


if __name__ == "__main__":
    main()
