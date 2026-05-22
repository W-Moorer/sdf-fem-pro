"""Prepare a source-gear Abaqus deck with linear penalty contact.

This utility is validation/post-processing code.  It keeps the original gear
mesh, RP-MPCs, boundary conditions, torque, and implicit dynamic step from the
source deck, while changing only the contact enforcement/output cadence needed
for an apples-to-apples SFC penalty-contact comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "commercial_software_comparison"
    / "abaqus_flexible_body_gear_contact"
    / "gear_contact.inp"
)


def _keyword_name(line: str) -> str:
    stripped = line.strip()
    if not stripped.startswith("*") or stripped.startswith("**"):
        return ""
    return stripped.split(",", 1)[0].strip().lower()


def _replace_or_add_param(keyword_line: str, name: str, value: str) -> str:
    parts = [part.strip() for part in keyword_line.strip().split(",")]
    prefix = parts[0]
    lower_name = name.lower()
    kept = [prefix]
    replaced = False
    for part in parts[1:]:
        if not part:
            continue
        key = part.split("=", 1)[0].strip().lower()
        if key == lower_name:
            kept.append(f"{name}={value}")
            replaced = True
        else:
            kept.append(part)
    if not replaced:
        kept.append(f"{name}={value}")
    return ", ".join(kept)


def _replace_dynamic_data(line: str, *, dt: float | None, duration: float | None) -> str:
    values = [value.strip() for value in line.strip().split(",")]
    if len(values) < 2:
        return line
    if dt is not None:
        values[0] = f"{float(dt):.12e}"
    if duration is not None:
        values[1] = f"{float(duration):.12e}"
    return ",".join(values)


def prepare_source_penalty_deck_text(
    text: str,
    *,
    pressure_stiffness: float,
    frame_stride: int,
    dt: float | None = None,
    duration: float | None = None,
) -> str:
    """Return source deck text converted to linear penalty contact.

    Only validation-facing Abaqus input is rewritten.  Meshes, node/element sets,
    MPCs, loads, and boundary conditions are left untouched.
    """

    stride = max(1, int(frame_stride))
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    saw_surface_behavior = False
    saw_contact_pair = False
    saw_dynamic_data = False
    while i < len(lines):
        line = lines[i]
        key = _keyword_name(line)
        if key == "*surface behavior":
            out.append("*Surface Behavior, pressure-overclosure=LINEAR")
            out.append(f"{float(pressure_stiffness):.12e}")
            saw_surface_behavior = True
            i += 1
            while i < len(lines):
                candidate = lines[i].strip()
                if not candidate or candidate.startswith("*") or candidate.startswith("**"):
                    break
                i += 1
            continue
        if key == "*contact pair":
            rewritten = line
            if "mechanical constraint" not in line.lower():
                rewritten = _replace_or_add_param(rewritten, "mechanical constraint", "PENALTY")
            out.append(rewritten)
            saw_contact_pair = True
            i += 1
            continue
        if key in {"*output"} and "field" in line.lower():
            out.append(_replace_or_add_param(line, "frequency", str(stride)))
            i += 1
            continue
        if key in {"*output"} and "history" in line.lower():
            out.append(_replace_or_add_param(line, "frequency", str(stride)))
            i += 1
            continue
        if key == "*dynamic":
            out.append(line)
            i += 1
            while i < len(lines):
                data_line = lines[i]
                stripped = data_line.strip()
                if not stripped or stripped.startswith("**"):
                    out.append(data_line)
                    i += 1
                    continue
                if stripped.startswith("*"):
                    break
                out.append(_replace_dynamic_data(data_line, dt=dt, duration=duration))
                saw_dynamic_data = True
                i += 1
                break
            continue
        out.append(line)
        i += 1
    if not saw_surface_behavior:
        raise ValueError("source deck does not contain *Surface Behavior")
    if not saw_contact_pair:
        raise ValueError("source deck does not contain *Contact Pair")
    if (dt is not None or duration is not None) and not saw_dynamic_data:
        raise ValueError("requested dynamic timing override, but no *Dynamic data row was found")
    return "\n".join(out) + "\n"


def write_source_penalty_deck(
    source: Path,
    out_path: Path,
    *,
    pressure_stiffness: float,
    frame_stride: int,
    dt: float | None = None,
    duration: float | None = None,
    encoding: str = "cp936",
) -> None:
    """Write a source-derived linear-penalty Abaqus deck."""

    text = Path(source).read_text(encoding=encoding, errors="replace")
    converted = prepare_source_penalty_deck_text(
        text,
        pressure_stiffness=pressure_stiffness,
        frame_stride=frame_stride,
        dt=dt,
        duration=duration,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(converted, encoding=encoding, errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pressure-stiffness", type=float, default=5.0e9)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--dt", type=float, default=None, help="Optional override for the Abaqus *Dynamic initial increment.")
    parser.add_argument("--duration", type=float, default=None, help="Optional override for the Abaqus *Dynamic total time.")
    args = parser.parse_args(argv)
    write_source_penalty_deck(
        args.source,
        args.out,
        pressure_stiffness=float(args.pressure_stiffness),
        frame_stride=int(args.frame_stride),
        dt=args.dt,
        duration=args.duration,
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
