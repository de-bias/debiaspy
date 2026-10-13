import numpy as np
import pandas as pd
import pytest

from debiaspy import (
    validate_flow_residuals,
)
from validation_inputs import toy_adjusted, toy_benchmark


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


@pytest.mark.parametrize("top_n", [0, -1, np.nan])
def test_validate_flow_residuals_errors_for_invalid_top_n(top_n):
    with pytest.raises(ValueError, match="top_n"):
        validate_flow_residuals(toy_adjusted(), toy_benchmark(), top_n=top_n)
