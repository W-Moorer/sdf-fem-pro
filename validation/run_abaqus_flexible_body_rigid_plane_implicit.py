"""Run Abaqus/Standard implicit flexible body drops onto a rigid plane.

The cases mirror ``run_abaqus_flexible_body_rigid_plane.py`` but use implicit
direct-integration dynamics.  Contact is frictionless linear penalty normal
contact, and the maximum implicit increment is constrained to ``0.001 s`` while
allowing smaller increments if contact convergence requires it.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_flexible_body_rigid_plane import (  # noqa: E402
    CASE_NAMES,
    CommonConfig,
    CubeConfig,
    SphereConfig,
    _abaqus_reported_wallclock_seconds,
    _chunked_lines,
    _cube_mesh,
    _format_element,
    _format_node,
    _mesh_counts,
    _plane_part_lines,
    _resolve_abaqus_command,
    _run_command,
    _sphere_mesh,
    _write_case_metrics,
)


DEFAULT_OUT_DIR = ROOT / "commercial_software_comparison" / "abaqus_flexible_body_rigid_plane_implicit"
PENALTY_NORMAL_STIFFNESS = 5.0e9


def _material_lines(common: CommonConfig) -> list[str]:
    return [
        "*Material, name=BODY_MAT",
        "*Density",
        f"{common.density:.12e}",
        "*Elastic",
        f"{common.young:.12e}, {common.poisson:.12e}",
        "*Surface Interaction, name=FRICTIONLESS_PENALTY_CONTACT",
        "*Surface Behavior, pressure-overclosure=LINEAR",
        f"{PENALTY_NORMAL_STIFFNESS:.12e}",
        "*Friction",
        "0.",
    ]


def build_implicit_input_text(case: str, common: CommonConfig | None = None) -> str:
    """Build an Abaqus/Standard implicit dynamic input deck."""

    common = common or CommonConfig()
    if case == "cube_drop":
        nodes, elements, all_nodes = _cube_mesh(CubeConfig(), common)
        part_name = "FLEX_CUBE"
        instance_name = "CUBE-1"
        element_type = "C3D8R"
        element_set = "CUBE_EALL"
        heading = "Commercial reference: implicit flexible cube free fall onto rigid plane"
    elif case == "sphere_drop":
        nodes, elements, all_nodes = _sphere_mesh(SphereConfig(), common)
        part_name = "FLEX_SPHERE"
        instance_name = "SPHERE-1"
        element_type = "C3D4"
        element_set = "SPHERE_EALL"
        heading = "Commercial reference: implicit flexible sphere free fall onto rigid plane"
    else:
        raise ValueError(f"unknown case {case!r}; expected one of {CASE_NAMES}")

    lines: list[str] = [
        "*Heading",
        heading,
        "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
    ]
    lines.extend(_material_lines(common))
    lines.extend([f"*Part, name={part_name}", "*Node"])
    lines.extend(_format_node(label, xyz) for label, xyz in enumerate(nodes, start=1))
    lines.append(f"*Element, type={element_type}, elset={element_set}")
    lines.extend(_format_element(label, conn) for label, conn in enumerate(elements, start=1))
    lines.extend(
        [
            f"*Elset, elset={element_set}, generate",
            f"1, {len(elements)}, 1",
            f"*Solid Section, elset={element_set}, material=BODY_MAT",
            ",",
            "*End Part",
        ]
    )
    lines.extend(_plane_part_lines(common))
    lines.extend(
        [
            "*Assembly, name=ASSEMBLY",
            f"*Instance, name={instance_name}, part={part_name}",
            "*End Instance",
            "*Instance, name=PLANE-1, part=RIGID_PLANE",
            "*End Instance",
            f"*Nset, nset=BODY_NODES, instance={instance_name}",
        ]
    )
    lines.extend(_chunked_lines(all_nodes))
    lines.extend(
        [
            "*End Assembly",
            "*Contact",
            "*Contact Inclusions, ALL EXTERIOR",
            "*Contact Property Assignment",
            ", , FRICTIONLESS_PENALTY_CONTACT",
            "*Boundary",
            "PLANE-1.PLANE_RP, ENCASTRE",
            "*Step, name=IMPLICIT_FREE_FALL_CONTACT, nlgeom=YES, inc=12000",
            "*Dynamic",
            f"{common.output_interval:.12e}, {common.duration:.12e}, 1.000000000000e-08, {common.output_interval:.12e}",
            "*Dload",
            f"{instance_name}.{element_set}, GRAV, {common.gravity:.12e}, 0., 0., -1.",
            f"*Output, field, time interval={common.output_interval:.12e}",
            "*Node Output",
            "U, V",
            f"*Element Output, elset={instance_name}.{element_set}, directions=YES",
            "S, LE",
            f"*Output, history, time interval={common.output_interval:.12e}",
            "*Energy Output",
            "ALLKE, ALLIE, ALLSE, ALLVD, ALLWK, ETOTAL",
            "*End Step",
            "",
        ]
    )
    return "\n".join(lines)


def run_case(case: str, out_root: Path, *, abaqus_command: str | None = None) -> dict[str, Path | int | float | str]:
    common = CommonConfig()
    total_start = time.perf_counter()
    case_dir = out_root / case
    run_dir = case_dir / "abaqus_run"
    vtk_dir = case_dir / "vtk"
    run_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir.mkdir(parents=True, exist_ok=True)

    job_name = f"{case}_implicit"
    for stale in run_dir.glob(f"{job_name}*"):
        if stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink()
    for stale_vtk in vtk_dir.glob("frame_*.vtk"):
        stale_vtk.unlink()
    for stale_index in (vtk_dir / "frame.pvd", vtk_dir / "frame_manifest.csv"):
        stale_index.unlink(missing_ok=True)

    inp_path = run_dir / f"{job_name}.inp"
    inp_path.write_text(build_implicit_input_text(case, common), encoding="ascii")
    node_count, element_count = _mesh_counts(case, common)

    command = _resolve_abaqus_command(abaqus_command)
    analysis_wall_seconds = _run_command(
        [command, f"job={job_name}", f"input={inp_path.name}", "interactive", "ask_delete=OFF"],
        cwd=run_dir,
        log_path=case_dir / "abaqus_analysis_stdout.log",
    )
    odb_path = run_dir / f"{job_name}.odb"
    if not odb_path.exists():
        raise RuntimeError(f"Abaqus completed without producing {odb_path}")

    converter = Path(__file__).with_name("abaqus_odb_to_vtk.py")
    export_wall_seconds = _run_command(
        [command, "python", str(converter), "--odb", str(odb_path), "--out-dir", str(vtk_dir), "--stem", "frame"],
        cwd=run_dir,
        log_path=case_dir / "abaqus_odb_to_vtk_stdout.log",
    )
    frame_count = len(sorted(vtk_dir.glob("frame_*.vtk")))
    if frame_count == 0:
        raise RuntimeError(f"ODB export produced no VTK frames in {vtk_dir}")
    total_wall_seconds = time.perf_counter() - total_start
    abaqus_reported_wall = _abaqus_reported_wallclock_seconds(run_dir / f"{job_name}.sta")
    metrics_path = case_dir / "runtime_metrics.csv"
    _write_case_metrics(
        metrics_path,
        {
            "case": case,
            "method": "Abaqus/Standard implicit dynamic",
            "abaqus_analysis_wall_seconds": f"{analysis_wall_seconds:.6f}",
            "abaqus_reported_standard_wallclock_seconds": f"{abaqus_reported_wall:.6f}" if abaqus_reported_wall is not None else "",
            "odb_to_vtk_wall_seconds": f"{export_wall_seconds:.6f}",
            "total_workflow_wall_seconds": f"{total_wall_seconds:.6f}",
            "abaqus_model_duration_seconds": f"{common.duration:.6f}",
            "initial_time_increment_seconds": f"{common.output_interval:.6f}",
            "maximum_time_increment_seconds": f"{common.output_interval:.6f}",
            "field_output_interval_seconds": f"{common.output_interval:.6f}",
            "vtk_frame_count": frame_count,
            "node_count_per_frame": node_count,
            "element_count_per_frame": element_count,
            "contact_normal_behavior": "LINEAR_PENALTY",
            "contact_penalty_normal_stiffness": f"{PENALTY_NORMAL_STIFFNESS:.6e}",
            "contact_scope": "ALL EXTERIOR",
            "contact_friction_coefficient": "0.0",
        },
    )
    (case_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Abaqus/Standard Implicit {case} Flexible-Body Rigid-Plane Reference",
                "",
                "Commercial-solver comparison artifact generated by a validation-only workflow.",
                "",
                f"- Input deck: `{inp_path.relative_to(case_dir).as_posix()}`",
                f"- ODB: `{odb_path.relative_to(case_dir).as_posix()}`",
                "- Requested field output: `U`, `V`, `S`, `LE`.",
                f"- ParaView collection: `{(vtk_dir / 'frame.pvd').relative_to(case_dir).as_posix()}`",
                f"- Numbered VTK frames: `{frame_count}` files named `frame_0000.vtk`, `frame_0001.vtk`, ...",
                f"- Runtime metrics: `{metrics_path.relative_to(case_dir).as_posix()}`",
                f"- Abaqus analysis wall time: `{analysis_wall_seconds:.3f}` s.",
                f"- Abaqus-reported Standard wallclock time: `{abaqus_reported_wall:.3f}` s." if abaqus_reported_wall is not None else "- Abaqus-reported Standard wallclock time: unavailable.",
                f"- ODB-to-VTK wall time: `{export_wall_seconds:.3f}` s.",
                f"- Total workflow wall time: `{total_wall_seconds:.3f}` s.",
                "",
                f"Contact: frictionless general contact with linear penalty normal stiffness `{PENALTY_NORMAL_STIFFNESS:.6e}`.",
                "Object ids in VTK cell data: `0` = rigid plane, `1` = cube, `2` = sphere.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "case": case,
        "case_dir": case_dir,
        "inp": inp_path,
        "odb": odb_path,
        "pvd": vtk_dir / "frame.pvd",
        "frame_count": frame_count,
        "metrics": metrics_path,
        "analysis_wall_seconds": analysis_wall_seconds,
        "export_wall_seconds": export_wall_seconds,
        "total_wall_seconds": total_wall_seconds,
    }


def _write_summary(out_root: Path, results: list[dict[str, Path | int | float | str]]) -> None:
    summary_csv = out_root / "summary.csv"
    with summary_csv.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "frame_count", "pvd", "metrics", "analysis_wall_seconds", "export_wall_seconds", "total_wall_seconds"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case": result["case"],
                    "frame_count": result["frame_count"],
                    "pvd": Path(result["pvd"]).relative_to(out_root).as_posix(),
                    "metrics": Path(result["metrics"]).relative_to(out_root).as_posix(),
                    "analysis_wall_seconds": f"{float(result['analysis_wall_seconds']):.6f}",
                    "export_wall_seconds": f"{float(result['export_wall_seconds']):.6f}",
                    "total_wall_seconds": f"{float(result['total_wall_seconds']):.6f}",
                }
            )
    (out_root / "README.md").write_text(
        "\n".join(
            [
                "# Abaqus/Standard Implicit Flexible Body Free-Fall To Rigid Plane",
                "",
                "Two Abaqus/Standard implicit dynamic commercial-solver comparison cases.",
                "",
                "- Duration: `3.0 s`.",
                "- Initial and maximum time increment: `0.001 s`; smaller automatic cutbacks are allowed.",
                "- Contact: frictionless general contact against a discrete rigid R3D4 plane.",
                f"- Normal contact: linear penalty pressure-overclosure with stiffness `{PENALTY_NORMAL_STIFFNESS:.6e}`.",
                "- Outputs: `U`, `V`, `S`, `LE`, numbered legacy VTK frames, and PVD collections.",
                f"- Summary CSV: `{summary_csv.name}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--abaqus-command", default=None)
    parser.add_argument("--case", choices=CASE_NAMES, action="append", help="Run one case; repeat to run multiple. Defaults to both.")
    args = parser.parse_args()

    cases = args.case or list(CASE_NAMES)
    results = [run_case(case, args.out_dir, abaqus_command=args.abaqus_command) for case in cases]
    _write_summary(args.out_dir, results)
    for result in results:
        print(f"{result['case']}: {result['pvd']} ({result['frame_count']} frames)")


if __name__ == "__main__":
    main()
