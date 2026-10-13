"""Adjust OD flows and compare them with a benchmark using the public API."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from debiaspy import (
    adjust_inverse_penetration,
    debiaspy_example_data,
    measure_bias,
    validate_flow_overall,
)


def analyse(
    observed: pd.DataFrame,
    benchmark: pd.DataFrame,
    coverage: pd.DataFrame,
    weight_by: str = "origin",
) -> dict[str, pd.DataFrame]:
    """Run the same calculation for Census or caller-supplied area identifiers."""
    measured = measure_bias(coverage)
    adjusted = adjust_inverse_penetration(observed, coverage, weight_by=weight_by)
    invalid = ~np.isfinite(adjusted["flow_adj"])
    diagnostics = pd.DataFrame(
        {
            "observed_rows": [len(observed)],
            "adjusted_rows": [len(adjusted)],
            "undefined_adjusted_rows": [int(invalid.sum())],
        }
    )
    # Keep undefined values visible; do not calculate metrics on a reduced sample.
    metrics = (
        pd.DataFrame()
        if invalid.any()
        else validate_flow_overall(
            adjusted,
            benchmark,
            comparisons=["raw_vs_benchmark", "adjusted_vs_benchmark"],
            na_rm=False,
            drop_zeros=False,
        )["summary"]
    )
    return {
        "coverage": measured,
        "adjusted_od": adjusted,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--origin-column", default="origin")
    parser.add_argument("--destination-column", default="destination")
    parser.add_argument("--observed-flow-column", default="flow")
    parser.add_argument("--benchmark-flow-column", default="flow")
    parser.add_argument("--coverage-area-column", default="origin")
    parser.add_argument(
        "--weight-by", choices=["origin", "destination", "both"], default="origin"
    )
    parser.add_argument("--geography", choices=["lad", "msoa"], default="lad")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    supplied = [args.observed, args.benchmark, args.coverage]
    if any(supplied) and not all(supplied):
        parser.error("Supply --observed, --benchmark and --coverage together")
    if all(supplied):
        keys = {
            args.origin_column: "origin",
            args.destination_column: "destination",
        }
        observed = pd.read_csv(
            args.observed, dtype={key: str for key in [*keys, "mpd_source", "mpd_time"]}
        ).rename(columns={**keys, args.observed_flow_column: "flow"})
        benchmark = pd.read_csv(
            args.benchmark,
            dtype={key: str for key in [*keys, "mpd_source", "mpd_time"]},
        ).rename(columns={**keys, args.benchmark_flow_column: "flow"})
        coverage = pd.read_csv(
            args.coverage,
            dtype={
                key: str
                for key in [args.coverage_area_column, "mpd_source", "mpd_time"]
            },
        ).rename(columns={args.coverage_area_column: "origin"})
    else:
        bundle = debiaspy_example_data(n_areas=None, geography=args.geography)
        observed, benchmark, coverage = (
            bundle["mpd_od"],
            bundle["benchmark_od"],
            bundle["coverage"],
        )
    tables = analyse(observed, benchmark, coverage, args.weight_by)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    print(tables["diagnostics"].to_string(index=False))
    if tables["metrics"].empty:
        print("Metrics withheld: resolve missing or infinite adjusted flows first.")
    else:
        print(tables["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
