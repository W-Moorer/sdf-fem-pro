"""Audit Abaqus gear-contact input features against current SFC support.

The runner reads the supplied flexible-gear ``.inp`` file and writes a compact
feature-support table.  It does not run Abaqus and does not import any Abaqus
module; the file is treated as an optional external validation input.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flexible_gear_explicit_sdf_comparison import DEFAULT_SOURCE, parse_gear_input  # noqa: E402

Row = dict[str, Any]
DEFAULT_OUT_DIR = ROOT / "results" / "gear_inp_feature_audit"


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _keyword_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped.startswith("*") or stripped.startswith("**"):
            continue
        key = stripped.split(",", 1)[0].lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_feature_rows(source: Path = DEFAULT_SOURCE) -> list[Row]:
    """Return feature-support rows for the supplied gear input."""

    model = parse_gear_input(source)
    counts = _keyword_counts(source)
    total_nodes = int(model.gear1.nodes.shape[0] + model.gear2.nodes.shape[0])
    total_elements = int(model.gear1.elements.shape[0] + model.gear2.elements.shape[0])
    total_contact_faces = int(model.gear1_contact_faces.shape[0] + model.gear2_contact_faces.shape[0])
    rows: list[Row] = [
        {
            "feature": "two_flexible_gear_parts",
            "abaqus_usage": f"2 parts, {total_nodes} nodes, {total_elements} C3D4 elements",
            "sfc_status": "supported_parser_only",
            "evidence": "parse_gear_input extracts both complete meshes",
            "gap": "full-size SFC solve not yet completed",
        },
        {
            "feature": "c3d4_tet4_volume_elements",
            "abaqus_usage": f"{model.gear1.elements.shape[0]} + {model.gear2.elements.shape[0]} C3D4",
            "sfc_status": "supported",
            "evidence": "MechanicsModel.from_tet4_mesh and TET4 tests exist",
            "gap": "full gear memory/performance path still needs optimization",
        },
        {
            "feature": "linear_elastic_density_material",
            "abaqus_usage": f"E={model.young:g}, nu={model.poisson:g}, density={model.density:g}",
            "sfc_status": "supported",
            "evidence": "internal TET4 material and mass assembly",
            "gap": "",
        },
        {
            "feature": "element_surface_contact_sets",
            "abaqus_usage": f"{model.gear1_contact_faces.shape[0]} + {model.gear2_contact_faces.shape[0]} triangular faces",
            "sfc_status": "supported_parser_partial_solver",
            "evidence": "surface entries are parsed and mapped to triangle faces",
            "gap": "full-size all-face contact solve is not yet validated",
        },
        {
            "feature": "beam_mpc_hub_to_reference_point",
            "abaqus_usage": f"2 BEAM MPC constraints, hub nodes {len(model.gear1_hub_labels)} + {len(model.gear2_hub_labels)}",
            "sfc_status": "partial",
            "evidence": "small-rotation RigidHubMPC, finite-rotation kinematics, and reduced rotational inertia helpers exist",
            "gap": "not yet wired into the full flexible gear global solve",
        },
        {
            "feature": "velocity_angular_boundary",
            "abaqus_usage": f"gear1 omega_z={model.gear1_angular_velocity_z:g}",
            "sfc_status": "partial",
            "evidence": "finite-rotation kinematics can prescribe angular motion; prescribed RP angular velocity has an Abaqus-aligned small-case validation",
            "gap": "not yet wired as a full finite-rotation time-integrated RP constraint for all gear nodes in the full flexible gear solve",
        },
        {
            "feature": "moment_cload_on_rp",
            "abaqus_usage": f"gear2 torque_z={model.gear2_torque_z:g}",
            "sfc_status": "partial",
            "evidence": "value is parsed; reduced RP torque dynamics has an Abaqus-aligned small-case validation",
            "gap": "full flexible gear torque residual coupling is not yet assembled in the global nonlinear solve",
        },
        {
            "feature": "surface_to_surface_hard_contact",
            "abaqus_usage": "frictionless HARD pressure-overclosure, SURFACE TO SURFACE contact pair",
            "sfc_status": "partial",
            "evidence": "SFC has Lagrangian-SDF penalty surface contact and a hard normal active-set layer aligned to an Abaqus node-to-surface small case",
            "gap": "surface-to-surface hard contact must still be wired into the full Lagrangian-SDF gear solve",
        },
        {
            "feature": "nlgeom_implicit_dynamic_step",
            "abaqus_usage": "nlgeom=YES, *Dynamic 1e-05,0.05,1e-10,5e-05",
            "sfc_status": "partial",
            "evidence": "HHT/Newmark and StVK diagnostics exist",
            "gap": "full Abaqus nonlinear dynamic increment/acceptance equivalence is not complete for full gear",
        },
        {
            "feature": "preselect_field_history_output",
            "abaqus_usage": "field/history PRESELECT",
            "sfc_status": "partial",
            "evidence": "validation scripts export selected U/S/E/contact metrics",
            "gap": "general Abaqus-style output request layer is missing",
        },
    ]
    for row in rows:
        row["source"] = str(source)
        row["keyword_count_hint"] = "; ".join(f"{key}:{counts[key]}" for key in sorted(counts) if key in {"*mpc", "*dynamic", "*contact pair", "*surface behavior", "*boundary", "*cload"})
    return rows


def write_summary(path: Path, rows: list[Row]) -> None:
    supported = sum(1 for row in rows if str(row["sfc_status"]) == "supported")
    partial = sum(1 for row in rows if "partial" in str(row["sfc_status"]))
    missing = sum(1 for row in rows if "missing" in str(row["sfc_status"]))
    text = [
        "# Gear INP Feature Audit",
        "",
        f"- supported: {supported}",
        f"- partial: {partial}",
        f"- missing/exact-missing: {missing}",
        "",
        "| Feature | Abaqus usage | SFC status | Gap |",
        "|---|---|---|---|",
    ]
    for row in rows:
        text.append(f"| {row['feature']} | {row['abaqus_usage']} | {row['sfc_status']} | {row['gap']} |")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    rows = build_feature_rows(args.source)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gear_inp_feature_audit.csv"
    md_path = out_dir / "gear_inp_feature_audit.md"
    _write_csv(csv_path, rows)
    write_summary(md_path, rows)
    print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
