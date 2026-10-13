import numpy as np
import pandas as pd
import pytest

from debiaspy import (
    validate_flow_overall,
    validate_flow_pairs,
)
from debiaspy._exceptions import DebiasSchemaError
from validation_inputs import toy_adjusted, toy_benchmark


def test_validate_flow_pairs_returns_joined_r_columns_and_signed_differences():
    actual = validate_flow_pairs(toy_adjusted(), toy_benchmark())

    assert list(actual.columns) == [
        "origin",
        "destination",
        "mpd_flow",
        "adj_flow",
        "benchmark_flow",
        "diff_mpd_benchmark",
        "diff_mpd_adj",
        "diff_adj_benchmark",
        "signed_residual_adjusted_vs_benchmark",
        "signed_residual_raw_vs_benchmark",
        "signed_residual_raw_vs_adjusted",
    ]
    first = actual.iloc[0]
    assert first["mpd_flow"] == 10.0
    assert first["adj_flow"] == 12.0
    assert first["benchmark_flow"] == 15.0
    assert first["diff_mpd_benchmark"] == -5.0
    assert first["diff_mpd_adj"] == -2.0
    assert first["diff_adj_benchmark"] == -3.0
    assert first["signed_residual_adjusted_vs_benchmark"] == 3.0
    assert first["signed_residual_raw_vs_benchmark"] == 5.0
    assert first["signed_residual_raw_vs_adjusted"] == 2.0


def test_validate_flow_overall_returns_r_like_metrics_and_joined_data():
    actual = validate_flow_overall(toy_adjusted(), toy_benchmark())

    assert isinstance(actual, dict)
    assert {"n", "rmse", "mae", "data"} <= set(actual)
    assert actual["comparison"] == "adjusted_vs_benchmark"
    assert actual["n"] == 3
    assert actual["data"].shape[0] == 3
    np.testing.assert_allclose(actual["rmse"], np.sqrt((3**2 + (-8) ** 2 + 3**2) / 3))
    np.testing.assert_allclose(actual["mae"], (3 + 8 + 3) / 3)
    assert actual["data"]["residual"].tolist() == [3.0, -8.0, 3.0]


def test_validate_flow_overall_by_source_returns_grouped_metrics():
    adjusted = toy_adjusted()
    adjusted["mpd_source"] = ["phone", "phone", "phone"]

    actual = validate_flow_overall(adjusted, toy_benchmark(), by_source=True)

    assert "by_source" in actual
    assert actual["by_source"]["mpd_source"].tolist() == ["phone"]
    assert actual["by_source"]["n"].tolist() == [3]


def test_validate_flow_overall_by_source_drops_missing_and_unused_sources():
    adjusted = toy_adjusted()
    adjusted["mpd_source"] = pd.Categorical(
        ["phone", pd.NA, "phone"], categories=["phone", "web"]
    )

    actual = validate_flow_overall(adjusted, toy_benchmark(), by_source=True)

    assert actual["by_source"]["mpd_source"].tolist() == ["phone"]
    assert actual["by_source"]["n"].tolist() == [2]


def test_validate_flow_overall_retained_na_propagates_rmse_and_mae():
    adjusted = toy_adjusted()
    adjusted.loc[1, "flow_adj"] = np.nan

    actual = validate_flow_overall(
        adjusted,
        toy_benchmark(),
        na_rm=False,
        drop_zeros=False,
    )

    assert np.isnan(actual["rmse"])
    assert np.isnan(actual["mae"])


def test_validate_flow_pairs_errors_for_missing_required_columns():
    with pytest.raises(DebiasSchemaError, match="adj_df.*origin"):
        validate_flow_pairs(toy_adjusted().drop(columns="origin"), toy_benchmark())

    with pytest.raises(DebiasSchemaError, match="benchmark_od_df.*flow"):
        validate_flow_pairs(toy_adjusted(), toy_benchmark().drop(columns="flow"))


def test_validate_flow_overall_errors_for_invalid_comparison():
    with pytest.raises(ValueError, match="comparisons"):
        validate_flow_overall(
            toy_adjusted(), toy_benchmark(), comparisons=["not_a_comparison"]
        )
