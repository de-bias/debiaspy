import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from debiaspy import adjust_coefficient, adjust_raking_ratio
from debiaspy._exceptions import DebiasSchemaError


def simple_mpd() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "mpd_source": ["s1", "s1", "s1", "s1"],
            "flow": [10.0, 20.0, 30.0, 40.0],
        }
    )


def simple_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["A", "B"],
            "population": [100.0, 200.0],
            "user_count": [10.0, 50.0],
            "mpd_source": ["s1", "s1"],
        }
    )


def simple_covariates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "area": ["A", "B"],
            "income_norm": [1.0, 0.0],
            "rural_pct": [0.25, 0.75],
        }
    )


def simple_benchmark() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "flow": [100.0, 200.0, 90.0, 120.0],
        }
    )


def test_raking_ratio_origin_targets_scales_each_origin_margin():
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [90.0, 140.0]})

    actual = adjust_raking_ratio(simple_mpd(), origin_targets=origin_targets)

    expected = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "mpd_source": ["s1", "s1", "s1", "s1"],
            "flow": [10.0, 20.0, 30.0, 40.0],
            "flow_adj": [30.0, 60.0, 60.0, 80.0],
            "weight_ipf": [3.0, 3.0, 2.0, 2.0],
        }
    )
    assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.attrs["ipf_converged"] is True
    assert actual.attrs["ipf_iterations"] == 2


def test_raking_ratio_requires_a_target_source():
    with pytest.raises(DebiasSchemaError, match="Provide at least one"):
        adjust_raking_ratio(simple_mpd())


def test_raking_ratio_benchmark_targets_match_numeric_ids_like_r():
    mpd = pd.DataFrame(
        {
            "origin": [1, 1, 2, 2],
            "destination": [1, 2, 1, 2],
            "flow": [10.0, 20.0, 30.0, 40.0],
        }
    )
    benchmark = pd.DataFrame(
        {
            "origin": [1, 1, 2, 2],
            "destination": [1, 2, 1, 2],
            "flow": [100.0, 200.0, 90.0, 120.0],
        }
    )

    actual = adjust_raking_ratio(mpd, benchmark_od_df=benchmark)
    origin_totals = actual.groupby("origin", sort=True, as_index=False).agg(
        flow_adj=("flow_adj", "sum")
    )
    destination_totals = actual.groupby("destination", sort=True, as_index=False).agg(
        flow_adj=("flow_adj", "sum")
    )

    assert actual["flow_adj"].sum() == pytest.approx(510.0)
    assert dict(
        zip(origin_totals["origin"], origin_totals["flow_adj"], strict=True)
    ) == pytest.approx({1: 300.0, 2: 210.0})
    assert dict(
        zip(
            destination_totals["destination"],
            destination_totals["flow_adj"],
            strict=True,
        )
    ) == pytest.approx({1: 190.0, 2: 320.0})
    assert actual["flow_adj"].tolist() != actual["flow"].tolist()


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


def test_raking_ratio_fails_closed_on_missing_flow_in_active_margin():
    mpd = simple_mpd()
    mpd.loc[0, "flow"] = np.nan
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [30.0, 70.0]})

    with pytest.raises(DebiasSchemaError, match="non-finite values"):
        adjust_raking_ratio(mpd, origin_targets=origin_targets)


def test_raking_ratio_fails_closed_on_impossible_structural_zero_margin():
    mpd = simple_mpd()
    mpd.loc[mpd["origin"] == "A", "flow"] = 0.0
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [30.0, 70.0]})

    with pytest.raises(DebiasSchemaError, match="zero support"):
        adjust_raking_ratio(mpd, origin_targets=origin_targets)


def test_raking_ratio_clipping_keeps_weight_and_flow_consistent():
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [90.0, 140.0]})

    actual = adjust_raking_ratio(
        simple_mpd(),
        origin_targets=origin_targets,
        clip_max=2.5,
    )

    np.testing.assert_allclose(
        actual["flow_adj"], actual["flow"] * actual["weight_ipf"]
    )
    assert actual["weight_ipf"].max() == pytest.approx(2.5)
    assert actual.attrs["ipf_converged"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0}, "tol"),
        ({"clip_min": -1}, "clip_min"),
        ({"clip_min": 2, "clip_max": 1}, "clip_min"),
    ],
)
def test_raking_ratio_validates_iteration_and_clipping_controls(kwargs, message):
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [90.0, 140.0]})

    with pytest.raises(DebiasSchemaError, match=message):
        adjust_raking_ratio(simple_mpd(), origin_targets=origin_targets, **kwargs)


def test_raking_ratio_rejects_negative_or_duplicate_targets():
    negative = pd.DataFrame({"origin": ["A", "B"], "target": [-1.0, 70.0]})
    duplicate = pd.DataFrame({"origin": ["A", "A"], "target": [30.0, 31.0]})

    with pytest.raises(DebiasSchemaError, match="finite non-negative"):
        adjust_raking_ratio(simple_mpd(), origin_targets=negative)
    with pytest.raises(DebiasSchemaError, match="at most one target"):
        adjust_raking_ratio(simple_mpd(), origin_targets=duplicate)
