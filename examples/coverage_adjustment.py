"""Measure, adjust, and validate a synthetic two-area OD example."""

import pandas as pd

from debiaspy import (
    adjust_inverse_penetration,
    measure_bias,
    validate_flow_overall,
)


def main() -> None:
    coverage = pd.DataFrame(
        {"origin": ["A", "B"], "population": [100, 200], "user_count": [10, 50]}
    )
    observed = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "flow": [1, 2, 3, 4],
        }
    )
    # Constructed to illustrate origin weights of 10 and 4.
    benchmark = observed.copy()
    benchmark["flow"] = [10, 20, 12, 16]

    adjusted = adjust_inverse_penetration(observed, coverage, weight_by="origin")
    validation = validate_flow_overall(
        adjusted,
        benchmark,
        comparisons=["raw_vs_benchmark", "adjusted_vs_benchmark"],
    )
    print("Synthetic coverage")
    print(measure_bias(coverage).to_string(index=False))
    print("\nAdjusted synthetic OD flows")
    print(adjusted.to_string(index=False))
    print("\nValidation against the constructed synthetic benchmark")
    print(validation["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
