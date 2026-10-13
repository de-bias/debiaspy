import numpy as np
import pandas as pd
import pytest

from debiaspy import (
    validate_flow_distribution,
    validate_flow_pairs,
    validate_flow_residuals,
)
from debiaspy._exceptions import DebiasSchemaError


def toy_adjusted():
    return pd.DataFrame(
        {
            "origin": ["A", "A", "B"],
            "destination": ["X", "Y", "X"],
            "flow": [10.0, 20.0, 30.0],
            "flow_adj": [12.0, 18.0, 42.0],
        }
    )


def toy_benchmark():
    return pd.DataFrame(
        {
            "origin": ["A", "A", "B"],
            "destination": ["X", "Y", "X"],
            "flow": [15.0, 10.0, 45.0],
        }
    )


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


def test_validate_flow_residuals_returns_summary_data_and_top_worst():
    actual = validate_flow_residuals(toy_adjusted(), toy_benchmark(), top_n=2)

    assert set(actual) == {"summary", "data", "top_worst"}
    assert actual["summary"].shape[0] == 1
    assert actual["summary"]["n"].iloc[0] == 3
    assert {
        "residual_mpd_benchmark",
        "residual_adj_benchmark",
        "signed_residual_reduction",
        "abs_residual_reduction",
        "improvement_flag",
    } <= set(actual["data"].columns)
    assert actual["data"]["residual_mpd_benchmark"].tolist() == [5.0, -10.0, 15.0]
    assert actual["data"]["residual_adj_benchmark"].tolist() == [3.0, -8.0, 3.0]
    assert actual["data"]["improvement_flag"].tolist() == [
        "improved",
        "improved",
        "improved",
    ]
    assert actual["top_worst"]["abs_residual_adj_benchmark"].tolist() == [8.0, 3.0]


def test_validate_flow_distribution_returns_summary_and_origin_level():
    actual = validate_flow_distribution(toy_adjusted(), toy_benchmark())

    assert set(actual) == {"summary", "origin_level"}
    summary = actual["summary"]
    origin_level = actual["origin_level"]
    assert summary["comparison"].tolist() == ["adjusted_vs_benchmark"]
    assert summary["weight_by"].tolist() == ["none"]
    assert {"kl_mean", "jsd_mean"} <= set(summary.columns)
    assert np.isnan(summary["kl_weighted_mean"].iloc[0])
    assert origin_level["origin"].tolist() == ["A", "B"]
    assert origin_level["n_destinations"].tolist() == [2, 1]
    assert (origin_level["kl_origin"] >= 0).all()
    assert (origin_level["jsd_origin"] >= 0).all()


def test_validate_flow_distribution_propagates_invalid_contributions():
    adjusted = pd.DataFrame(
        {
            "origin": ["A", "A"],
            "destination": ["X", "Y"],
            "flow": [1.0, 1.0],
            "flow_adj": [-1.0, 2.0],
        }
    )
    benchmark = pd.DataFrame(
        {
            "origin": ["A", "A"],
            "destination": ["X", "Y"],
            "flow": [1.0, 1.0],
        }
    )

    with np.errstate(invalid="ignore"):
        actual = validate_flow_distribution(adjusted, benchmark)

    assert np.isnan(actual["origin_level"]["kl_origin"].iloc[0])
    assert np.isnan(actual["origin_level"]["jsd_origin"].iloc[0])
    assert actual["summary"]["n_origins_used"].iloc[0] == 0
    assert np.isnan(actual["summary"]["kl_mean"].iloc[0])


def test_validate_flow_residuals_keeps_infinity_in_sd_calculation():
    adjusted = pd.DataFrame(
        {
            "origin": ["A", "A", "A"],
            "destination": ["X", "Y", "Z"],
            "flow": [1.0, 2.0, np.inf],
            "flow_adj": [1.0, 2.0, np.inf],
        }
    )
    benchmark = pd.DataFrame(
        {
            "origin": ["A", "A", "A"],
            "destination": ["X", "Y", "Z"],
            "flow": [0.0, 0.0, 0.0],
        }
    )

    actual = validate_flow_residuals(adjusted, benchmark)

    assert bool(actual["data"]["residual_sd_mpd_benchmark"].isna().all())
    assert bool(actual["data"]["residual_sd_adj_benchmark"].isna().all())
    assert bool(actual["data"]["residual_mpd_benchmark_sd_score"].isna().all())
    assert bool(actual["data"]["residual_adj_benchmark_sd_score"].isna().all())


def test_validate_flow_residuals_accepts_nullable_numeric_missing_values():
    adjusted = pd.DataFrame(
        {
            "origin": ["A", "A"],
            "destination": ["X", "Y"],
            "flow": pd.Series([1.0, pd.NA], dtype="Float64"),
            "flow_adj": pd.Series([2.0, pd.NA], dtype="Float64"),
        }
    )
    benchmark = pd.DataFrame(
        {
            "origin": ["A", "A"],
            "destination": ["X", "Y"],
            "flow": pd.Series([3.0, pd.NA], dtype="Float64"),
        }
    )

    actual = validate_flow_residuals(adjusted, benchmark)
    data = actual["data"]
    summary = actual["summary"].iloc[0]

    np.testing.assert_allclose(data["pct_residual_mpd_benchmark"].iloc[0], 200 / 3)
    np.testing.assert_allclose(data["pct_residual_adj_benchmark"].iloc[0], 100 / 3)
    assert pd.isna(data["pct_residual_mpd_benchmark"].iloc[1])
    assert pd.isna(data["pct_residual_improvement"].iloc[1])
    assert data["moved_in_benchmark_direction"].tolist() == [True, False]
    assert data["improvement_flag"].tolist() == ["improved", "unchanged"]
    assert summary["share_improved"] == 0.5
    assert summary["share_unchanged"] == 0.5


def test_validate_flow_residuals_truncates_fractional_top_n_like_r():
    actual = validate_flow_residuals(toy_adjusted(), toy_benchmark(), top_n=1.5)

    assert len(actual["top_worst"]) == 1


def test_validate_flow_pairs_errors_for_missing_required_columns():
    with pytest.raises(DebiasSchemaError, match="adj_df.*origin"):
        validate_flow_pairs(toy_adjusted().drop(columns="origin"), toy_benchmark())

    with pytest.raises(DebiasSchemaError, match="benchmark_od_df.*flow"):
        validate_flow_pairs(toy_adjusted(), toy_benchmark().drop(columns="flow"))


@pytest.mark.parametrize("epsilon", [0, -1, np.nan, np.inf])
def test_validate_flow_distribution_errors_for_invalid_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        validate_flow_distribution(toy_adjusted(), toy_benchmark(), epsilon=epsilon)


@pytest.mark.parametrize("top_n", [0, -1, np.nan])
def test_validate_flow_residuals_errors_for_invalid_top_n(top_n):
    with pytest.raises(ValueError, match="top_n"):
        validate_flow_residuals(toy_adjusted(), toy_benchmark(), top_n=top_n)
