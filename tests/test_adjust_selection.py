import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from adjustment_inputs import (
    simple_covariates,
    simple_coverage,
    simple_mpd,
)
from debiaspy import (
    adjust_selection_rate,
    adjust_selection_rate2,
)
from debiaspy._exceptions import DebiasSchemaError


def test_selection_rate_fixed_r_uses_origin_selection_weights():
    actual = adjust_selection_rate(
        simple_mpd(),
        simple_coverage(),
        covariates_df=simple_covariates(),
        covariate_col="income_norm",
        weight_by="origin",
        r_global=0.2,
    )

    expected = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "mpd_source": ["s1", "s1", "s1", "s1"],
            "flow": [10.0, 20.0, 30.0, 40.0],
            "weight_origin": [10.0, 10.0, 5.0, 5.0],
            "weight_destination": [math.nan, math.nan, math.nan, math.nan],
            "weight_missing": [False, False, False, False],
            "flow_adj": [100.0, 200.0, 150.0, 200.0],
        }
    )
    assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.attrs["r_global"] == 0.2


def test_selection_rate_rejects_conflicting_covariate_aliases():
    with pytest.raises(DebiasSchemaError, match="Use only one"):
        adjust_selection_rate(
            simple_mpd(),
            simple_coverage(),
            covariates_df=simple_covariates(),
            covariate_col="income_norm",
            income_col="rural_pct",
        )


def test_selection_rate2_fixed_k_uses_zagheni_weber_correction():
    actual = adjust_selection_rate2(
        simple_mpd(),
        simple_coverage(),
        weight_by="origin",
        k=1.0,
    )

    def cf(p: float) -> float:
        return p * (math.exp(-1.0) - 1.0) / (math.exp(-1.0 * p) - 1.0)

    weight_a = cf(0.1)
    weight_b = cf(0.25)
    expected = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "mpd_source": ["s1", "s1", "s1", "s1"],
            "flow": [10.0, 20.0, 30.0, 40.0],
            "weight_origin": [weight_a, weight_a, weight_b, weight_b],
            "weight_destination": [math.nan, math.nan, math.nan, math.nan],
            "weight_missing": [False, False, False, False],
            "flow_adj": [10 * weight_a, 20 * weight_a, 30 * weight_b, 40 * weight_b],
        }
    )
    assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.attrs["k"] == 1.0


def test_selection_rate2_retains_missing_group_strata_like_dplyr():
    mpd = pd.DataFrame(
        {
            "origin": ["A", "A"],
            "destination": ["B", "B"],
            "mpd_source": ["s", "s"],
            "age": ["young", np.nan],
            "note": ["observed", "missing-stratum"],
            "flow": [10.0, 20.0],
        }
    )
    coverage = pd.DataFrame(
        {
            "origin": ["A", "A"],
            "population": [100.0, 200.0],
            "user_count": [10.0, 100.0],
            "mpd_source": ["s", "s"],
            "age": ["young", np.nan],
        }
    )

    actual = adjust_selection_rate2(
        mpd,
        coverage,
        group_cols="age",
        k=np.float64(1.0),
        keep_cols="note",
    )

    assert bool(actual["flow_adj"].notna().all())
    assert not bool(actual["weight_missing"].any())
    assert actual.loc[1, "weight_origin"] == pytest.approx(
        0.5 * (math.exp(-1.0) - 1.0) / (math.exp(-0.5) - 1.0)
    )
    assert actual.attrs["k"] == 1.0
    assert "note" in actual.columns


def test_selection_rate2_rejects_logical_k_as_non_numeric_like_r():
    with pytest.raises(DebiasSchemaError, match="positive scalar"):
        adjust_selection_rate2(simple_mpd(), simple_coverage(), k=True)
