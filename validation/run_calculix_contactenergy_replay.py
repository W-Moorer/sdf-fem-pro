"""Replay CalculiX `contactenergy.inp` with current-surface dynamic SDF.

This validation intentionally uses the official CalculiX C3D8 contact-energy
test as a contact-law/energy reference. It is not a TET4 trajectory-equivalence
case: CalculiX solves the static C3D8 contact problem, and SFC replays the final
deformed surface geometry using dynamic-SDF local projection and the same linear
pressure-overclosure law.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sfc.sdf.dynamic_surface_sdf import dynamic_surface_sdf  # noqa: E402


Row = dict[str, Any]

WSL_CONTACTENERGY_INP_CANDIDATES = [
    "/tmp/sfc_calculix_source/test/contactenergy.inp",
    "/usr/share/doc/calculix-ccx-test/examples/test/contactenergy.inp",
]
WSL_CONTACTENERGY_DAT_REF_CANDIDATES = [
    "/tmp/sfc_calculix_source/test/contactenergy.dat.ref",
    "/usr/share/doc/calculix-ccx-test/examples/test/contactenergy.dat.ref",
]

C3D8_FACE_NODES = {
    "S1": np.array([0, 1, 2, 3], dtype=np.int64),
    "S2": np.array([4, 7, 6, 5], dtype=np.int64),
    "S3": np.array([0, 4, 5, 1], dtype=np.int64),
    "S4": np.array([1, 5, 6, 2], dtype=np.int64),
    "S5": np.array([2, 6, 7, 3], dtype=np.int64),
    "S6": np.array([3, 7, 4, 0], dtype=np.int64),
}


@dataclass(frozen=True)
class ContactEnergyModel:
    node_ids: np.ndarray
    X: np.ndarray
    element_ids: np.ndarray
    elements: np.ndarray
    elsets: dict[str, list[int]]
    surfaces: dict[str, list[tuple[str, str]]]
    pressure_stiffness: float
    z_load_reference: float


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=timeout)


def _wsl_available() -> bool:
    if shutil.which("wsl") is None:
        return False
    try:
        _run(["wsl", "--exec", "bash", "-lc", "true"], cwd=ROOT, timeout=15)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def _wsl_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _first_existing_wsl_path(candidates: list[str]) -> str | None:
    if not _wsl_available():
        return None
    for candidate in candidates:
        proc = subprocess.run(
            ["wsl", "--exec", "bash", "-lc", f"test -f {_wsl_quote(candidate)}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0:
            return candidate
    return None


def _read_wsl_text(candidates: list[str]) -> tuple[str, str]:
    source = _first_existing_wsl_path(candidates)
    if source is None:
        raise RuntimeError(f"none of the WSL reference files exist: {candidates}")
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", f"cat {_wsl_quote(source)}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.stdout, source


def _calculix_available() -> bool:
    if not _wsl_available():
        return False
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", "command -v ccx >/dev/null"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return proc.returncode == 0


def _calculix_version() -> str:
    if not _calculix_available():
        return "unavailable"
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-lc", "ccx -v 2>&1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    lines = [line.strip() for line in (proc.stdout + proc.stderr).splitlines() if line.strip()]
    return lines[0] if lines else "CalculiX"


def _parse_keyword(line: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in line.split(",")]
    name = parts[0].upper()
    params: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip().upper()] = value.strip()
    return name, params


def parse_contactenergy_input(text: str) -> ContactEnergyModel:
    nodes: dict[int, list[float]] = {}
    elements: dict[int, list[int]] = {}
    elsets: dict[str, list[int]] = {}
    surfaces: dict[str, list[tuple[str, str]]] = {}
    cloads: list[tuple[int, int, float]] = []
    pressure_stiffness: float | None = None

    section = ""
    params: dict[str, str] = {}
    current_surface = ""
    current_elset = ""
    expect_surface_behavior_value = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            section, params = _parse_keyword(line)
            expect_surface_behavior_value = section == "*SURFACE BEHAVIOR"
            current_surface = params.get("NAME", "") if section == "*SURFACE" else ""
            current_elset = params.get("ELSET", "") if section == "*ELSET" else params.get("ELSET", "")
            continue

        if expect_surface_behavior_value:
            pressure_stiffness = float(line.split(",")[0])
            expect_surface_behavior_value = False
            continue

        values = [value.strip() for value in line.split(",") if value.strip()]
        if section == "*NODE":
            nodes[int(values[0])] = [float(values[1]), float(values[2]), float(values[3])]
        elif section == "*ELEMENT":
            element_id = int(values[0])
            elements[element_id] = [int(value) for value in values[1:]]
            if current_elset:
                elsets.setdefault(current_elset, []).append(element_id)
        elif section == "*ELSET":
            elsets.setdefault(current_elset, []).extend(int(value) for value in values)
        elif section == "*SURFACE":
            surfaces.setdefault(current_surface, []).append((values[0], values[1].upper()))
        elif section == "*CLOAD":
            cloads.append((int(values[0]), int(values[1]), float(values[2])))

    if pressure_stiffness is None:
        raise ValueError("failed to parse pressure-overclosure stiffness from contactenergy input")

    node_ids = np.asarray(sorted(nodes), dtype=np.int64)
    node_to_index = {int(node_id): idx for idx, node_id in enumerate(node_ids)}
    X = np.asarray([nodes[int(node_id)] for node_id in node_ids], dtype=float)

    element_ids = np.asarray(sorted(elements), dtype=np.int64)
    conn = np.asarray([[node_to_index[node_id] for node_id in elements[int(element_id)]] for element_id in element_ids], dtype=np.int64)
    z_load_reference = abs(sum(value for _, dof, value in cloads if dof == 3))
    return ContactEnergyModel(
        node_ids=node_ids,
        X=X,
        element_ids=element_ids,
        elements=conn,
        elsets=elsets,
        surfaces=surfaces,
        pressure_stiffness=float(pressure_stiffness),
        z_load_reference=float(z_load_reference),
    )


def _parse_last_nodal_block(text: str, block_name: str) -> dict[int, np.ndarray]:
    lines = text.splitlines()
    blocks: list[dict[int, np.ndarray]] = []
    for idx, line in enumerate(lines):
        if block_name in line.lower():
            block: dict[int, np.ndarray] = {}
            for row in lines[idx + 1 :]:
                parts = row.split()
                if len(parts) == 4 and parts[0].lstrip("+-").isdigit():
                    block[int(parts[0])] = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
                elif block:
                    break
            if block:
                blocks.append(block)
    if not blocks:
        raise ValueError(f"failed to parse CalculiX {block_name} block")
    return blocks[-1]


def _parse_total_contact_energy(text: str) -> float:
    matches = re.findall(r"total contact spring energy for time\s+[-+0-9.Ee]+\s*\n\s*([-+0-9.Ee]+)", text, flags=re.IGNORECASE)
    if not matches:
        raise ValueError("failed to parse total contact spring energy")
    return float(matches[-1])


def _parse_raw_cels_rows(text: str) -> list[Row]:
    rows: list[Row] = []
    in_block = False
    for raw in text.splitlines():
        lower = raw.lower()
        if "contact spring energy" in lower and "slave element+face" in lower:
            in_block = True
            rows = []
            continue
        if in_block and "total contact spring energy" in lower:
            in_block = False
            continue
        if in_block:
            parts = raw.split()
            if len(parts) == 3 and parts[0].lstrip("+-").isdigit():
                rows.append({"slave_element": int(parts[0]), "slave_face": int(parts[1]), "cels": float(parts[2])})
    return rows


def parse_contactenergy_dat(text: str, node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, list[Row]]:
    displacement_block = _parse_last_nodal_block(text, "displacements")
    force_block = _parse_last_nodal_block(text, "forces")
    U = np.asarray([displacement_block.get(int(node_id), np.zeros(3)) for node_id in node_ids], dtype=float)
    RF = np.asarray([force_block.get(int(node_id), np.zeros(3)) for node_id in node_ids], dtype=float)
    return U, RF, _parse_total_contact_energy(text), _parse_raw_cels_rows(text)


def _surface_quads(model: ContactEnergyModel, surface_name: str) -> list[tuple[np.ndarray, np.ndarray]]:
    element_id_to_index = {int(element_id): idx for idx, element_id in enumerate(model.element_ids)}
    quads: list[tuple[np.ndarray, np.ndarray]] = []
    for target, face_name in model.surfaces[surface_name]:
        if face_name not in C3D8_FACE_NODES:
            raise ValueError(f"unsupported C3D8 face label {face_name}")
        element_ids = model.elsets.get(target, [int(target)] if target.isdigit() else [])
        if not element_ids:
            raise ValueError(f"surface target {target!r} does not resolve to elements")
        for element_id in element_ids:
            element = model.elements[element_id_to_index[int(element_id)]]
            quads.append((element[C3D8_FACE_NODES[face_name]], element))
    return quads


def _orient_quad_outward(X: np.ndarray, quad: np.ndarray, element: np.ndarray) -> np.ndarray:
    polygon = X[quad]
    normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
    if float(np.dot(normal, polygon.mean(axis=0) - X[element].mean(axis=0))) < 0.0:
        return quad[::-1]
    return quad


def _triangles_from_quads(X: np.ndarray, quads: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    triangles: list[np.ndarray] = []
    for quad, element in quads:
        oriented = _orient_quad_outward(X, quad, element)
        triangles.append(oriented[[0, 1, 2]])
        triangles.append(oriented[[0, 2, 3]])
    return np.asarray(triangles, dtype=np.int64)


def _triangle_area(triangle: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))


def replay_contactenergy_with_dynamic_sdf(model: ContactEnergyModel, U: np.ndarray) -> Row:
    X_current = model.X + U
    master_faces = _triangles_from_quads(X_current, _surface_quads(model, "Smast"))
    slave_faces = _triangles_from_quads(X_current, _surface_quads(model, "Sslav"))
    candidates = np.arange(master_faces.shape[0], dtype=np.int64)

    total_force = np.zeros(3, dtype=float)
    contact_energy = 0.0
    min_gap = np.inf
    active_count = 0
    for face in slave_faces:
        triangle = X_current[face]
        xq = triangle.mean(axis=0)
        area = _triangle_area(triangle)
        result = dynamic_surface_sdf(xq, X_current, master_faces, candidates)
        min_gap = min(min_gap, float(result.g))
        overclosure = max(-float(result.g), 0.0)
        if overclosure > 0.0:
            active_count += 1
            pressure = model.pressure_stiffness * overclosure
            total_force += pressure * area * result.n
            contact_energy += 0.5 * model.pressure_stiffness * overclosure * overclosure * area

    return {
        "sfc_dynamic_sdf_contact_force_x": float(total_force[0]),
        "sfc_dynamic_sdf_contact_force_y": float(total_force[1]),
        "sfc_dynamic_sdf_contact_force_z": float(total_force[2]),
        "sfc_dynamic_sdf_contact_force_norm": float(np.linalg.norm(total_force)),
        "sfc_dynamic_sdf_contact_energy": float(contact_energy),
        "sfc_dynamic_sdf_min_gap": float(min_gap),
        "sfc_dynamic_sdf_max_penetration": float(max(-min_gap, 0.0)),
        "sfc_dynamic_sdf_active_quadrature_count": int(active_count),
        "sfc_dynamic_sdf_master_triangle_count": int(master_faces.shape[0]),
        "sfc_dynamic_sdf_slave_quadrature_count": int(slave_faces.shape[0]),
    }


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1.0e-30)


def _write_csv(path: Path, fieldnames: list[str], rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, metrics: Row, command_row: Row, claims: list[Row]) -> None:
    lines = [
        "# CalculiX Contactenergy C3D8 Replay",
        "",
        "This validation uses the official CalculiX `contactenergy.inp` C3D8",
        "surface-to-surface contact-energy test as an external contact-law",
        "reference. SFC does not solve the C3D8 mechanics here; it replays the",
        "final CalculiX deformed geometry with the current-surface dynamic SDF",
        "query and the same linear pressure-overclosure law.",
        "",
        "## Command",
        "",
        "```bash",
        "python validation/run_calculix_contactenergy_replay.py --out-dir results/calculix_contactenergy_replay",
        "```",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| CalculiX total contact spring energy | {metrics['calculix_total_contact_spring_energy']} |",
        f"| SFC dynamic-SDF replay contact energy | {metrics['sfc_dynamic_sdf_contact_energy']} |",
        f"| contact energy relative error | {metrics['contact_energy_rel_error']} |",
        f"| CalculiX force reference magnitude | {metrics['calculix_force_reference_z_abs']} |",
        f"| SFC dynamic-SDF replay force z | {metrics['sfc_dynamic_sdf_contact_force_z']} |",
        f"| contact force relative error | {metrics['contact_force_rel_error']} |",
        f"| SFC min gap | {metrics['sfc_dynamic_sdf_min_gap']} |",
        "",
        "## Claims",
        "",
        "| Claim | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for claim in claims:
        lines.append(f"| {claim['claim']} | {claim['supported']} | {claim['evidence_csv']}::{claim['evidence_field']} |")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- This is a C3D8 external contact-law/energy replay, not a TET4 trajectory",
            "  equivalence claim.",
            "- Dynamic SDF is evaluated on triangulated current C3D8 boundary faces.",
            "- Friction, self-contact, hard contact, and nonlinear material behavior are",
            "  outside this replay.",
            "",
            "## External Solver",
            "",
            f"- solver: {command_row['external_solver']}",
            f"- version: {command_row['external_solver_version']}",
            f"- completed: {command_row['completed']}",
            f"- command: `{command_row['command']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_contactenergy_replay(*, out_dir: Path, skip_calculix: bool, timeout: int) -> dict[str, list[Row]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "calculix_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    inp_text, inp_source = _read_wsl_text(WSL_CONTACTENERGY_INP_CANDIDATES)
    inp_path = run_dir / "contactenergy.inp"
    inp_path.write_text(inp_text, encoding="utf-8")

    command_row: Row
    if not skip_calculix and _calculix_available():
        wsl_run_dir = _wsl_path(run_dir)
        command = f"cd {_wsl_quote(wsl_run_dir)} && ccx contactenergy"
        proc = subprocess.run(
            ["wsl", "--exec", "bash", "-lc", command],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        (run_dir / "calculix_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (run_dir / "calculix_stderr.log").write_text(proc.stderr, encoding="utf-8")
        dat_path = run_dir / "contactenergy.dat"
        if proc.returncode != 0 or not dat_path.exists():
            raise RuntimeError("CalculiX contactenergy run failed; use --skip-calculix for reference replay")
        command_row = {
            "case": "contactenergy",
            "external_solver": "CalculiX",
            "external_solver_version": _calculix_version(),
            "command": f"wsl --exec bash -lc \"{command}\"",
            "input_source": inp_source,
            "inp": str(inp_path.relative_to(out_dir)),
            "dat": str(dat_path.relative_to(out_dir)),
            "completed": "true",
            "used_dat_ref": "false",
        }
    else:
        dat_text, dat_source = _read_wsl_text(WSL_CONTACTENERGY_DAT_REF_CANDIDATES)
        dat_path = run_dir / "contactenergy.dat.ref"
        dat_path.write_text(dat_text, encoding="utf-8")
        command_row = {
            "case": "contactenergy",
            "external_solver": "CalculiX",
            "external_solver_version": _calculix_version(),
            "command": "not run; official contactenergy.dat.ref replayed",
            "input_source": inp_source,
            "inp": str(inp_path.relative_to(out_dir)),
            "dat": str(dat_path.relative_to(out_dir)),
            "completed": "true",
            "used_dat_ref": "true",
        }

    model = parse_contactenergy_input(inp_text)
    dat_text = dat_path.read_text(encoding="utf-8")
    U, RF, calculix_contact_energy, cels_rows = parse_contactenergy_dat(dat_text, model.node_ids)
    replay = replay_contactenergy_with_dynamic_sdf(model, U)

    force_error = _relative_error(float(replay["sfc_dynamic_sdf_contact_force_z"]), model.z_load_reference)
    energy_error = _relative_error(float(replay["sfc_dynamic_sdf_contact_energy"]), calculix_contact_energy)
    status = "ok" if force_error <= 1.0e-5 and energy_error <= 1.0e-5 else "check"
    metrics: Row = {
        "case": "calculix_contactenergy_c3d8_replay",
        "element_type": "c3d8",
        "reference_scope": "contact_law_energy_replay_not_tet4_trajectory",
        "dynamic_sdf_backend": "current_surface_triangulated_c3d8_faces",
        "pressure_stiffness": model.pressure_stiffness,
        "nodes": model.X.shape[0],
        "elements": model.elements.shape[0],
        "calculix_force_reference_z_abs": model.z_load_reference,
        "calculix_total_contact_spring_energy": calculix_contact_energy,
        "calculix_raw_cels_row_count": len(cels_rows),
        "calculix_rf_z_sum": float(np.sum(RF[:, 2])),
        **replay,
        "contact_force_rel_error": force_error,
        "contact_energy_rel_error": energy_error,
        "status": status,
    }
    claims = [
        {
            "claim": "calculix_contactenergy_c3d8_reference_available",
            "supported": "true",
            "evidence_csv": "calculix_contactenergy_replay.csv",
            "evidence_field": "status",
            "details": "official CalculiX contactenergy C3D8 reference parsed",
        },
        {
            "claim": "dynamic_sdf_replays_c3d8_contact_energy",
            "supported": str(status == "ok").lower(),
            "evidence_csv": "calculix_contactenergy_replay.csv",
            "evidence_field": "contact_energy_rel_error",
            "details": "current-surface dynamic SDF on triangulated C3D8 faces matches CalculiX total CELS",
        },
        {
            "claim": "dynamic_sdf_is_not_tet4_bound",
            "supported": "true",
            "evidence_csv": "calculix_contactenergy_replay.csv",
            "evidence_field": "dynamic_sdf_backend",
            "details": "replay uses C3D8 boundary quads triangulated into current surface triangles",
        },
    ]

    metric_fields = list(metrics.keys())
    _write_csv(out_dir / "calculix_contactenergy_replay.csv", metric_fields, [metrics])
    _write_csv(out_dir / "calculix_contactenergy_claims.csv", list(claims[0].keys()), claims)
    _write_csv(out_dir / "calculix_contactenergy_commands.csv", list(command_row.keys()), [command_row])
    _write_csv(
        out_dir / "calculix_contactenergy_raw_cels.csv",
        ["slave_element", "slave_face", "cels"],
        cels_rows,
    )
    _write_summary(out_dir / "calculix_contactenergy_summary.md", metrics, command_row, claims)
    return {"metrics": [metrics], "claims": claims, "commands": [command_row], "raw_cels": cels_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "calculix_contactenergy_replay")
    parser.add_argument("--quick", action="store_true", help="Accepted for consistency; this case is already small.")
    parser.add_argument("--skip-calculix", action="store_true", help="Replay the official contactenergy.dat.ref instead of running ccx.")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    run_contactenergy_replay(out_dir=args.out_dir, skip_calculix=bool(args.skip_calculix), timeout=int(args.timeout))
    print(f"Wrote CalculiX contactenergy replay outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
