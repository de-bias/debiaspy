from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from debiaspy import (
    debiaspy_example_data,
    validate_bias_residual_structure,
    validate_flow_margins,
    validate_flow_plot_distribution,
    validate_flow_residual_structure,
)


def _flows() -> tuple[pd.DataFrame, pd.DataFrame]:
    adjusted = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "flow": [5.0, 10.0, 7.0, 9.0],
            "flow_adj": [6.0, 11.0, 8.0, 10.0],
        }
    )
    benchmark = adjusted.loc[:, ["origin", "destination"]].copy()
    benchmark["flow"] = [6.0, 12.0, 8.0, 11.0]
    return adjusted, benchmark


def test_example_geography_accepts_unambiguous_r_prefix() -> None:
    pytest.importorskip("debiaspydata")
    result = debiaspy_example_data(n_areas=2, geography="l")
    assert result["metadata"].iloc[0]["geography"] == "lad"


def test_validation_choice_arguments_accept_unambiguous_r_prefixes() -> None:
    adjusted, benchmark = _flows()
    margins = validate_flow_margins({"method": adjusted}, benchmark, role="o")
    assert set(margins) == {"adjusted_outputs", "benchmark"}
    assert set(margins["adjusted_outputs"]) == {"method"}

    structure = validate_flow_residual_structure(
        adjusted,
        benchmark,
        residual_type="adjuste",
        spatial_role="o",
        residual_aggregation="m",
    )
    assert structure["summary"].iloc[0]["spatial_role"] == "origin"

    coverage = pd.DataFrame(
        {
            "origin": ["A", "B"],
            "population": [100.0, 200.0],
            "user_count": [20.0, 60.0],
        }
    )
    bias = validate_bias_residual_structure(coverage, residual_type="coverage_score")
    assert bias["summary"].iloc[0]["residual_type"] == "coverage_score"


def test_plot_choice_prefixes_are_resolved_before_use() -> None:
    results = {
        "summary": pd.DataFrame(
            {
                "method": ["method"],
                "comparison": ["adjusted_vs_benchmark"],
                "jsd_mean": [0.1],
            }
        )
    }
    figure = validate_flow_plot_distribution(
        results,
        metric="j",
        value="mea",
        sort="a",
    )
    try:
        assert figure.axes
    finally:
        plt.close(figure)


def test_ambiguous_r_prefix_is_rejected() -> None:
    adjusted, benchmark = _flows()
    with pytest.raises(ValueError, match="residual_type"):
        validate_flow_residual_structure(
            adjusted,
            benchmark,
            residual_type="a",
        )
