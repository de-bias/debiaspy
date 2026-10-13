import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from adjustment_inputs import (
    simple_benchmark,
    simple_mpd,
)
from debiaspy import (
    adjust_coefficient,
)
from debiaspy._exceptions import DebiasSchemaError


def test_coefficient_ols_through_origin_applies_global_beta():
    actual = adjust_coefficient(simple_mpd(), simple_benchmark())

    beta = sum([10 * 100, 20 * 200, 30 * 90, 40 * 120]) / sum(
        [10**2, 20**2, 30**2, 40**2]
    )
    expected = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "flow": [10.0, 20.0, 30.0, 40.0],
            "flow_adj": [10 * beta, 20 * beta, 30 * beta, 40 * beta],
            "coef_factor": [beta, beta, beta, beta],
        }
    )
    assert_frame_equal(actual, expected, check_dtype=False)
    assert math.isclose(actual.attrs["coef"], beta)
    assert list(actual.attrs["model"].columns) == [
        "mpd_source",
        "n",
        "beta",
        "intercept",
        "r_squared",
        "family",
        "level",
    ]


def test_coefficient_by_source_requires_source_in_both_inputs():
    with pytest.raises(DebiasSchemaError, match="`by_source = TRUE` requires"):
        adjust_coefficient(
            simple_mpd(),
            simple_benchmark(),
            by_source=True,
        )


def test_coefficient_intercept_uses_estimable_prediction_when_slope_is_aliased():
    mpd = pd.DataFrame(
        {
            "origin": ["A", "B", "C"],
            "destination": ["X", "Y", "Z"],
            "flow": [2.0, 2.0, 2.0],
        }
    )
    benchmark = mpd.assign(flow=[1.0, 2.0, 3.0])

    actual = adjust_coefficient(mpd, benchmark, fit_intercept=True)
    model = actual.attrs["model"].iloc[0]

    assert math.isnan(actual.attrs["coef"])
    assert math.isnan(model["beta"])
    assert model["intercept"] == pytest.approx(2.0)
    assert model["r_squared"] == pytest.approx(0.0)
    assert actual["coef_factor"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert actual["flow_adj"].tolist() == pytest.approx([2.0, 2.0, 2.0])


def test_coefficient_intercept_applies_flow_specific_documented_correction():
    mpd = pd.DataFrame(
        {
            "origin": ["A", "B", "C"],
            "destination": ["X", "Y", "Z"],
            "flow": [1.0, 2.0, 4.0],
        }
    )
    benchmark = mpd.assign(flow=[5.0, 7.0, 11.0])

    actual = adjust_coefficient(mpd, benchmark, fit_intercept=True)

    assert actual["flow_adj"].tolist() == pytest.approx([5.0, 7.0, 11.0])
    assert actual["coef_factor"].tolist() == pytest.approx([5.0, 3.5, 2.75])
    np.testing.assert_allclose(
        actual["flow_adj"], actual["flow"] * actual["coef_factor"]
    )


def test_coefficient_poisson_matches_r_all_zero_boundary_iterate():
    mpd = pd.DataFrame(
        {
            "origin": ["A", "B", "C"],
            "destination": ["X", "Y", "Z"],
            "flow": [2.0, 2.0, 2.0],
        }
    )
    benchmark = mpd.assign(flow=[0.0, 0.0, 0.0])

    actual = adjust_coefficient(mpd, benchmark, model_family="poisson")
    model = actual.attrs["model"].iloc[0]

    assert actual.attrs["coef"] == pytest.approx(3.7912802139559e-11, rel=1e-12)
    assert model["intercept"] == pytest.approx(-23.995732273554, rel=1e-12)
    assert np.isfinite(actual["flow_adj"]).all()
    assert (actual["flow_adj"] > 0).all()


def test_coefficient_poisson_emits_r_non_integer_warning_sequence():
    mpd = pd.DataFrame(
        {
            "origin": ["A", "B", "C"],
            "destination": ["X", "Y", "Z"],
            "flow": [2.0, 2.0, 2.0],
        }
    )
    benchmark = mpd.assign(flow=[0.5, 1.5, 2.5])

    with pytest.warns(UserWarning) as caught:
        actual = adjust_coefficient(mpd, benchmark, model_family="poisson")

    messages = [str(item.message) for item in caught]
    expected_pass = [
        "non-integer x = 0.500000",
        "non-integer x = 1.500000",
        "non-integer x = 2.500000",
    ]
    assert messages == expected_pass * 2
    assert actual.attrs["coef"] == pytest.approx(0.75)
    assert math.isnan(actual.attrs["model"].iloc[0]["r_squared"])


def test_coefficient_negbin_matches_r_with_audited_statsmodels_backend():
    pytest.importorskip("statsmodels")
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
    y = [0, 0, 1, 0, 3, 1, 8, 2, 12, 4, 18, 3]
    mpd = pd.DataFrame(
        {
            "origin": range(len(x)),
            "destination": range(len(x)),
            "flow": x,
        }
    )

    actual = adjust_coefficient(
        mpd,
        mpd.assign(flow=y),
        model_family="negbin",
    )

    assert actual.attrs["coef"] == pytest.approx(
        0.5695862072149359,
        abs=1e-8,
    )


def test_coefficient_zinb_fails_closed_when_backend_does_not_converge():
    pytest.importorskip("statsmodels")
    x = np.tile([1.0, 2.0, 4.0, 8.0], 5)
    y = [0, 0, 0, 8, 0, 0, 3, 7, 0, 0, 0, 10, 0, 1, 6, 8, 0, 0, 0, 12]
    mpd = pd.DataFrame(
        {
            "origin": range(len(x)),
            "destination": range(len(x)),
            "flow": x,
        }
    )

    with pytest.raises(DebiasSchemaError, match="optimizer did not converge"):
        adjust_coefficient(
            mpd,
            mpd.assign(flow=y),
            model_family="zinb",
        )


def test_coefficient_zinb_matches_r_with_audited_optimizer_tolerance():
    pytest.importorskip("statsmodels")
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
    y = [0, 0, 1, 0, 3, 1, 8, 2, 12, 4, 18, 3]
    mpd = pd.DataFrame(
        {
            "origin": range(len(x)),
            "destination": range(len(x)),
            "flow": x,
        }
    )

    actual = adjust_coefficient(
        mpd,
        mpd.assign(flow=y),
        model_family="zinb",
    )

    assert actual.attrs["coef"] == pytest.approx(
        0.5695908318247107,
        abs=1e-5,
    )


def test_coefficient_zinb_rejects_positive_only_counts_like_r():
    pytest.importorskip("statsmodels")

    with pytest.raises(DebiasSchemaError, match="minimum count is not zero"):
        adjust_coefficient(
            simple_mpd(),
            simple_benchmark(),
            model_family="zinb",
        )
