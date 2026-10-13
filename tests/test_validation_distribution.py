import numpy as np
import pandas as pd
import pytest

from debiaspy import (
    validate_flow_distribution,
)
from validation_inputs import toy_adjusted, toy_benchmark


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


@pytest.mark.parametrize("epsilon", [0, -1, np.nan, np.inf])
def test_validate_flow_distribution_errors_for_invalid_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        validate_flow_distribution(toy_adjusted(), toy_benchmark(), epsilon=epsilon)
