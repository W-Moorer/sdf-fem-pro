"""Validate dynamic-SDF queries on a prescribed deforming master surface.

This runner isolates the project method on a non-static master surface.  It
compares two contact-geometry providers using the same slave samples and the
same penalty assembly:

- spatial hash + dynamic FEM-induced SDF on the current master surface.
- brute-force all-face projection reference on the same current master surface.

The brute-force path is validation-only evidence.  The public SDF contact
geometry still requires candidate faces supplied by a broad phase.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfc.contact import DynamicSurfaceSDFContactGeometry, UniformTriangleAABBHash  # noqa: E402
from sfc.fem.calculix_aligned import assemble_contact_response  # noqa: E402
from sfc.sdf.dynamic_surface_sdf import dynamic_surface_sdf  # noqa: E402

Row = dict[str, Any]


def run_reference(out_dir: Path, *, quick: bool = False) -> dict[str, Path]:
    """Run deforming-master SDF reference validation and write artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    resolutions = [4] if quick else [4, 8, 12]
    deformation_names = ["stretch", "shear"] if quick else ["stretch", "shear", "bend"]
    query_rows: list[Row] = []
    force_rows: list[Row] = []
    for resolution in resolutions:
        master_reference, master_faces = _triangulated_master_surface(resolution)
        slave_reference, slave_faces = _slave_surface_samples(resolution)
        for deformation in deformation_names:
            master_current = _deform_master(master_reference, deformation)
            slave_current = _place_slave_near_master(slave_reference, deformation)
            query_rows.extend(_query_error_rows(resolution, deformation, master_current, master_faces, slave_current, slave_faces))
            force_rows.append(_force_error_row(resolution, deformation, master_current, master_faces, slave_current, slave_faces))

    claim_rows = _claim_rows(query_rows, force_rows)
    outputs = {
        "queries": out_dir / "deforming_master_sdf_queries.csv",
        "forces": out_dir / "deforming_master_sdf_forces.csv",
        "claims": out_dir / "deforming_master_sdf_claims.csv",
        "summary": out_dir / "deforming_master_sdf_summary.md",
    }
    _write_csv(outputs["queries"], query_rows)
    _write_csv(outputs["forces"], force_rows)
    _write_csv(outputs["claims"], claim_rows)
    _write_markdown(outputs["summary"], query_rows, force_rows, claim_rows, quick=quick)
    return outputs


def _triangulated_master_surface(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(resolution)
    nodes: list[tuple[float, float, float]] = []
    node_id: dict[tuple[int, int], int] = {}
    for j in range(n + 1):
        y = -0.5 + j / n
        for i in range(n + 1):
            x = -0.5 + i / n
            node_id[(i, j)] = len(nodes)
            nodes.append((x, y, 0.0))

    faces: list[tuple[int, int, int]] = []
    for j in range(n):
        for i in range(n):
            v00 = node_id[(i, j)]
            v10 = node_id[(i + 1, j)]
            v01 = node_id[(i, j + 1)]
            v11 = node_id[(i + 1, j + 1)]
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    return np.asarray(nodes, dtype=float), np.asarray(faces, dtype=np.int64)


def _slave_surface_samples(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    # Keep the slave mesh coarser than the master so the validation exercises
    # arbitrary query points, not node-to-node alignment.
    n = max(1, int(resolution) // 2)
    nodes: list[tuple[float, float, float]] = []
    node_id: dict[tuple[int, int], int] = {}
    for j in range(n + 1):
        y = -0.35 + 0.7 * j / n
        for i in range(n + 1):
            x = -0.35 + 0.7 * i / n
            node_id[(i, j)] = len(nodes)
            nodes.append((x, y, 0.0))
    faces: list[tuple[int, int, int]] = []
    for j in range(n):
        for i in range(n):
            v00 = node_id[(i, j)]
            v10 = node_id[(i + 1, j)]
            v01 = node_id[(i, j + 1)]
            v11 = node_id[(i + 1, j + 1)]
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    return np.asarray(nodes, dtype=float), np.asarray(faces, dtype=np.int64)


def _deform_master(reference: np.ndarray, deformation: str) -> np.ndarray:
    x = reference.copy()
    if deformation == "stretch":
        x[:, 0] *= 1.12
        x[:, 1] *= 0.94
        x[:, 2] = 0.04 * (x[:, 0] ** 2 - 0.25 * x[:, 1] ** 2)
    elif deformation == "shear":
        x[:, 0] += 0.18 * x[:, 1]
        x[:, 2] = 0.03 * x[:, 0] + 0.02 * x[:, 1]
    elif deformation == "bend":
        x[:, 2] = 0.08 * np.sin(np.pi * (x[:, 0] + 0.5)) * np.sin(np.pi * (x[:, 1] + 0.5))
    else:
        raise ValueError("unknown deformation")
    return x


def _place_slave_near_master(slave_reference: np.ndarray, deformation: str) -> np.ndarray:
    x = _deform_master(slave_reference, deformation)
    # Alternate slight separation and penetration across the patch.
    offsets = 0.012 * np.sin(3.0 * np.pi * (slave_reference[:, 0] + 0.5)) - 0.018
    x[:, 2] += offsets
    return x


def _query_error_rows(
    resolution: int,
    deformation: str,
    master_x: np.ndarray,
    master_faces: np.ndarray,
    slave_x: np.ndarray,
    slave_faces: np.ndarray,
) -> list[Row]:
    broad_phase = UniformTriangleAABBHash.from_surface(master_x, master_faces, delta_safe=0.15, cell_size=0.2)
    all_faces = np.arange(master_faces.shape[0], dtype=np.int64)
    rows: list[Row] = []
    sample_id = 0
    for face in slave_faces:
        tri = slave_x[face]
        for shape in (
            np.asarray([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]),
            np.asarray([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]),
            np.asarray([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]),
        ):
            point = shape @ tri
            candidates = broad_phase.query_point(point)
            reference = dynamic_surface_sdf(point, master_x, master_faces, all_faces)
            candidate = dynamic_surface_sdf(point, master_x, master_faces, candidates)
            normal_dot = float(np.clip(candidate.n @ reference.n, -1.0, 1.0))
            gap_error = abs(candidate.g - reference.g)
            angle_error = float(np.arccos(normal_dot))
            rows.append(
                {
                    "case": "deforming_master_sdf",
                    "resolution": resolution,
                    "deformation": deformation,
                    "sample_id": sample_id,
                    "candidate_count": int(candidates.size),
                    "total_faces": int(master_faces.shape[0]),
                    "gap_reference": reference.g,
                    "gap_dynamic_sdf": candidate.g,
                    "gap_abs_error": gap_error,
                    "normal_angle_error": angle_error,
                    "closest_face_reference": reference.face_id,
                    "closest_face_dynamic_sdf": candidate.face_id,
                    "status": "passed" if gap_error <= 1.0e-10 and angle_error <= 1.0e-7 else "failed",
                }
            )
            sample_id += 1
    return rows


def _force_error_row(
    resolution: int,
    deformation: str,
    master_x: np.ndarray,
    master_faces: np.ndarray,
    slave_x: np.ndarray,
    slave_faces: np.ndarray,
) -> Row:
    all_faces = np.arange(master_faces.shape[0], dtype=np.int64)
    broad_phase = UniformTriangleAABBHash.from_surface(master_x, master_faces, delta_safe=0.15, cell_size=0.2)
    broad_geometry = DynamicSurfaceSDFContactGeometry(
        slave_faces,
        master_x,
        master_faces,
        candidate_provider=broad_phase.query_point,
        stiffness=5.0e4,
    )
    reference_geometry = DynamicSurfaceSDFContactGeometry(
        slave_faces,
        master_x,
        master_faces,
        candidate_provider=lambda point: all_faces,
        stiffness=5.0e4,
    )
    broad = assemble_contact_response(broad_geometry.samples(slave_x), slave_x.shape[0])
    reference = assemble_contact_response(reference_geometry.samples(slave_x), slave_x.shape[0])
    force_error = float(np.linalg.norm(broad.force - reference.force))
    force_scale = max(float(np.linalg.norm(reference.force)), 1.0e-30)
    energy_error = abs(broad.energy - reference.energy)
    return {
        "case": "deforming_master_contact_force",
        "resolution": resolution,
        "deformation": deformation,
        "active_count_reference": reference.active_count,
        "active_count_dynamic_sdf": broad.active_count,
        "normal_force_reference": reference.normal_force,
        "normal_force_dynamic_sdf": broad.normal_force,
        "force_l2_abs_error": force_error,
        "force_l2_rel_error": force_error / force_scale,
        "contact_energy_reference": reference.energy,
        "contact_energy_dynamic_sdf": broad.energy,
        "contact_energy_abs_error": energy_error,
        "status": "passed" if force_error / force_scale <= 1.0e-10 and energy_error <= 1.0e-10 else "failed",
    }


def _claim_rows(query_rows: list[Row], force_rows: list[Row]) -> list[Row]:
    query_supported = bool(query_rows) and all(row["status"] == "passed" for row in query_rows)
    force_supported = bool(force_rows) and all(row["status"] == "passed" for row in force_rows)
    return [
        {
            "claim": "broad_phase_dynamic_sdf_matches_bruteforce_on_deforming_master",
            "supported": str(query_supported).lower(),
            "evidence_csv": "deforming_master_sdf_queries.csv",
            "details": "Gap and normal from spatial-hash candidates match all-face projection on prescribed current master surfaces.",
        },
        {
            "claim": "dynamic_sdf_contact_force_matches_bruteforce_reference",
            "supported": str(force_supported).lower(),
            "evidence_csv": "deforming_master_sdf_forces.csv",
            "details": "Penalty contact force assembled from spatial-hash dynamic SDF matches all-face reference contact samples.",
        },
    ]


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, query_rows: list[Row], force_rows: list[Row], claim_rows: list[Row], *, quick: bool) -> None:
    command = "python validation/run_deforming_master_sdf_reference.py"
    if quick:
        command += " --quick"
    command += f" --out-dir {path.parent.as_posix()}"
    max_gap = max(float(row["gap_abs_error"]) for row in query_rows) if query_rows else 0.0
    max_angle = max(float(row["normal_angle_error"]) for row in query_rows) if query_rows else 0.0
    max_force = max(float(row["force_l2_rel_error"]) for row in force_rows) if force_rows else 0.0
    lines = [
        "# Deforming Master Dynamic-SDF Reference",
        "",
        "This validation compares spatial-hash dynamic SDF queries against a brute-force all-face closest-projection reference on prescribed current master surfaces.",
        "The master surface is regenerated under stretch, shear, and bend deformations; the brute-force path is validation-only evidence.",
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Claim Gates",
        "",
        "| Claim | Supported | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in claim_rows:
        lines.append(f"| {row['claim']} | {row['supported']} | `{row['evidence_csv']}` |")
    lines.extend(
        [
            "",
            "## Error Summary",
            "",
            f"- Query rows: `{len(query_rows)}`",
            f"- Force rows: `{len(force_rows)}`",
            f"- Max gap absolute error: `{max_gap:.6e}`",
            f"- Max normal angle error: `{max_angle:.6e}`",
            f"- Max force relative error: `{max_force:.6e}`",
            "",
            "## Interpretation",
            "",
            "A passing result means that broad-phase candidate filtering did not change the local dynamic-SDF answer relative to all-face projection for the tested deforming master surfaces.",
            "It supports the contact-query side of the method; it does not claim external solver equivalence or nonlinear material validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "deforming_master_sdf_reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_reference(args.out_dir, quick=bool(args.quick))
    print("Deforming master dynamic-SDF reference validation complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
