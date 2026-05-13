"""Compare contact detection strategies on the deterministic contact scene."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_contact_benchmark import (
    ROOT,
    run_method_comparison,
    write_csv,
    write_markdown_summary,
    write_optional_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "benchmarks")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    iterations = 3 if args.quick else int(args.iterations)
    records = run_method_comparison(iterations)

    csv_path = args.out_dir / "contact_method_comparison.csv"
    md_path = args.out_dir / "contact_method_comparison.md"
    write_csv(records, csv_path)
    write_markdown_summary(records, md_path)
    if args.plot:
        write_optional_plot(records, args.out_dir / "contact_method_comparison.png")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
