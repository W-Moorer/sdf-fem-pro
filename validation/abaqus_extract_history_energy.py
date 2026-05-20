"""Export Abaqus ODB whole-model energy history to CSV.

This validation helper is intended to be run with ``abaqus python``.  It is
kept outside the SFC core package and reads only ODB files produced by external
validation jobs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_VARIABLES = ("ALLKE", "ALLIE", "ALLSE", "ALLVD", "ALLWK", "ETOTAL")


def _open_odb(path: Path):
    from odbAccess import openOdb  # type: ignore[import-not-found]

    return openOdb(path=str(path), readOnly=True)


def export_energy_history(odb_path: Path, out_csv: Path, variables: tuple[str, ...] = DEFAULT_VARIABLES) -> int:
    """Export requested ODB history variables and return the number of rows."""

    odb = _open_odb(odb_path)
    rows: list[dict[str, str | float]] = []
    try:
        for step_name, step in odb.steps.items():
            by_time: dict[float, dict[str, str | float]] = {}
            for region_name, region in step.historyRegions.items():
                available = set(region.historyOutputs.keys())
                for variable in variables:
                    if variable not in available:
                        continue
                    output = region.historyOutputs[variable]
                    for time_value, value in output.data:
                        row = by_time.setdefault(float(time_value), {"step": step_name, "time": float(time_value)})
                        row[variable] = float(value)
                        row.setdefault("history_region", region_name)
            for time_value in sorted(by_time):
                rows.append(by_time[time_value])
    finally:
        odb.close()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "time", "history_region", *variables]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odb", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = export_energy_history(args.odb, args.out_csv, tuple(str(value) for value in args.variables))
    print(f"Energy history rows: {count}")
    print(f"CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
