import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from debiaspy import (
    adjust_coefficient,
)


def simple_mpd() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "mpd_source": ["s1", "s1", "s1", "s1"],
            "flow": [10.0, 20.0, 30.0, 40.0],
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
    with pytest.raises(ValueError, match="`by_source = TRUE` requires"):
        adjust_coefficient(
            simple_mpd(),
            simple_benchmark(),
            by_source=True,
        )


def test_coefficient_intercept_retains_missing_adjustments_when_slope_is_aliased():
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
    assert actual["coef_factor"].isna().all()
    assert actual["flow_adj"].isna().all()


def test_coefficient_intercept_applies_r_scalar_slope_and_retains_intercept():
    mpd = pd.DataFrame(
        {
            "origin": ["A", "B", "C"],
            "destination": ["X", "Y", "Z"],
            "flow": [1.0, 2.0, 4.0],
        }
    )
    benchmark = mpd.assign(flow=[5.0, 7.0, 11.0])

    actual = adjust_coefficient(mpd, benchmark, fit_intercept=True)

    assert actual["flow_adj"].tolist() == pytest.approx([2.0, 4.0, 8.0])
    assert actual["coef_factor"].tolist() == pytest.approx([2.0, 2.0, 2.0])
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


def test_coefficient_zinb_fails_closed_when_backend_does_not_converge(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "scipy.optimize.minimize",
        lambda *args, **kwargs: SimpleNamespace(x=np.zeros(3), success=False),
    )
    x = np.tile([1.0, 2.0, 4.0, 8.0], 5)
    y = [0, 0, 0, 8, 0, 0, 3, 7, 0, 0, 0, 10, 0, 1, 6, 8, 0, 0, 0, 12]
    mpd = pd.DataFrame(
        {
            "origin": range(len(x)),
            "destination": range(len(x)),
            "flow": x,
        }
    )

    with pytest.raises(ValueError, match="optimizer did not converge"):
        adjust_coefficient(
            mpd,
            mpd.assign(flow=y),
            model_family="zinb",
        )


def test_coefficient_zinb_matches_r_with_audited_optimizer_tolerance():
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

    with pytest.raises(ValueError, match="minimum count is not zero"):
        adjust_coefficient(
            simple_mpd(),
            simple_benchmark(),
            model_family="zinb",
        )


@pytest.mark.parametrize(
    "level,beta,n",
    [("od", 25 / 6, 4), ("origin", 237 / 58, 2), ("destination", 67 / 13, 2)],
)
def test_ols_aggregation_has_explicit_coefficients(level, beta, n):
    actual = adjust_coefficient(simple_mpd(), simple_benchmark(), level=level)
    assert actual.attrs["coef"] == pytest.approx(beta)
    assert actual.attrs["model"].n.iloc[0] == n
    np.testing.assert_allclose(actual.flow_adj, simple_mpd().flow * beta)


@pytest.mark.parametrize("level", ["od", "origin", "destination"])
def test_poisson_aggregation_retains_offset_scale(level):
    actual = adjust_coefficient(
        simple_mpd(), simple_benchmark(), level=level, model_family="poisson"
    )
    assert actual.attrs["coef"] == pytest.approx(5.1)
    assert actual.attrs["model"].intercept.iloc[0] == pytest.approx(math.log(5.1))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1.0])
@pytest.mark.parametrize("side", ["observed", "benchmark"])
def test_invalid_calibration_values_are_filtered_and_original_rows_retained(
    bad_value, side
):
    mpd = pd.DataFrame(
        {"origin": list("ABC"), "destination": list("BCA"), "flow": [1.0, 2.0, 3.0]}
    )
    benchmark = mpd.assign(flow=[2.0, 4.0, 6.0])
    (mpd if side == "observed" else benchmark).loc[2, "flow"] = bad_value
    actual = adjust_coefficient(mpd, benchmark)
    assert actual.attrs["coef"] == pytest.approx(2)
    assert actual.attrs["model"].n.iloc[0] == 2
    assert len(actual) == 3
    np.testing.assert_allclose(actual.flow_adj, mpd.flow * 2)


@pytest.mark.parametrize("level", ["origin", "destination"])
def test_margin_aggregation_omits_missing_values_before_fitting(level):
    mpd = simple_mpd().astype({"flow": float})
    mpd.loc[0, "flow"] = np.nan
    actual = adjust_coefficient(mpd, simple_benchmark(), level=level)
    x = [20, 70] if level == "origin" else [30, 60]
    y = [300, 210] if level == "origin" else [190, 320]
    expected = sum(a * b for a, b in zip(x, y, strict=True)) / sum(a * a for a in x)
    assert actual.attrs["coef"] == pytest.approx(expected)
    assert pd.isna(actual.flow_adj.iloc[0])


def test_source_specific_fits_keep_missing_adjustments_for_unmatched_source():
    first = simple_mpd().assign(mpd_source="a")
    second = first.assign(mpd_source="b")
    third = first.assign(mpd_source="unmatched")
    observed = pd.concat([first, second, third], ignore_index=True)
    benchmark = pd.concat(
        [first.assign(flow=first.flow * 2), second.assign(flow=second.flow * 4)],
        ignore_index=True,
    )
    actual = adjust_coefficient(observed, benchmark, by_source=True)
    np.testing.assert_allclose(actual.coef_factor.iloc[:8], [2] * 4 + [4] * 4)
    assert actual.flow_adj.iloc[8:].isna().all()
    assert actual.attrs["model"].n.tolist() == [4, 4]
    pd.testing.assert_frame_equal(actual.attrs["coef"], actual.attrs["model"])


def test_sources_are_pooled_by_od_keys_without_implicit_time_grouping():
    observed = pd.concat(
        [simple_mpd().assign(mpd_source="a"), simple_mpd().assign(mpd_source="b")],
        ignore_index=True,
    )
    benchmark = observed.assign(flow=observed.flow * 2)
    with pytest.warns(UserWarning, match="many-to-many"):
        actual = adjust_coefficient(observed, benchmark)
    assert actual.attrs["coef"] == pytest.approx(2)
    assert actual.attrs["model"].n.iloc[0] == 16
    assert len(actual) == 8


def test_duplicate_od_rows_expand_calibration_without_expanding_output():
    observed = pd.DataFrame(
        {
            "origin": ["A", "A", "B"],
            "destination": ["B", "B", "C"],
            "flow": [1.0, 2.0, 3.0],
        }
    )
    benchmark = observed.assign(flow=[2.0, 4.0, 6.0])
    with pytest.warns(UserWarning, match="many-to-many"):
        actual = adjust_coefficient(observed, benchmark)
    assert actual.attrs["model"].n.iloc[0] == 5
    assert actual.attrs["coef"] == pytest.approx(36 / 19)
    assert len(actual) == 3


@pytest.mark.parametrize("rows", [0, 1])
def test_insufficient_overlap_returns_missing_coefficients(rows):
    observed = simple_mpd()
    actual = adjust_coefficient(observed, simple_benchmark().iloc[:rows])
    assert np.isnan(actual.attrs["coef"])
    assert actual.attrs["model"].n.iloc[0] == rows
    assert actual.flow_adj.isna().all()
    assert len(actual) == len(observed)


def test_unmatched_observations_receive_the_global_coefficient():
    observed = simple_mpd()
    actual = adjust_coefficient(observed, simple_benchmark().iloc[:2])
    assert actual.attrs["coef"] == pytest.approx(10)
    np.testing.assert_allclose(actual.flow_adj, observed.flow * 10)


def test_zero_observed_flows_are_excluded_from_calibration_and_retained_in_output():
    observed = simple_mpd().assign(flow=[0.0, 2.0, 3.0, 0.0])
    actual = adjust_coefficient(observed, observed.assign(flow=[10.0, 4.0, 6.0, 20.0]))
    assert actual.attrs["model"].n.iloc[0] == 2
    np.testing.assert_allclose(actual.flow_adj, [0, 4, 6, 0])


def test_nullable_flows_and_custom_index_preserve_input_frames():
    observed = simple_mpd().astype({"flow": "Float64"})
    observed.index = [9, 3, 8, 1]
    observed.iloc[0, observed.columns.get_loc("flow")] = pd.NA
    benchmark = simple_benchmark()
    before_observed, before_benchmark = (
        observed.copy(deep=True),
        benchmark.copy(deep=True),
    )
    actual = adjust_coefficient(observed, benchmark)
    assert pd.isna(actual.flow_adj.iloc[0])
    assert actual.index.tolist() == [0, 1, 2, 3]
    pd.testing.assert_frame_equal(observed, before_observed)
    pd.testing.assert_frame_equal(benchmark, before_benchmark)


def test_custom_flow_names_keep_columns_and_overwrite_adjustment_fields():
    observed = (
        simple_mpd()
        .rename(columns={"flow": "trips"})
        .assign(note="kept", flow_adj=999.0, coef_factor=999.0)
    )
    benchmark = (
        simple_benchmark().rename(columns={"flow": "total"}).assign(mpd_source="s1")
    )
    actual = adjust_coefficient(
        observed,
        benchmark,
        flow_col_mpd="trips",
        flow_col_bench="total",
        keep_cols=["note", "note", "missing"],
    )
    assert list(actual) == [
        "origin",
        "destination",
        "mpd_source",
        "note",
        "flow",
        "flow_adj",
        "coef_factor",
    ]
    np.testing.assert_allclose(actual.flow_adj, observed.trips * (25 / 6))


@pytest.mark.parametrize("side", ["mpd_od_df", "benchmark_od_df"])
@pytest.mark.parametrize("column", ["origin", "destination", "flow"])
def test_required_columns_name_the_failing_input(side, column):
    args = {"mpd_od_df": simple_mpd(), "benchmark_od_df": simple_benchmark()}
    args[side] = args[side].drop(columns=column)
    with pytest.raises(ValueError, match=side):
        adjust_coefficient(**args)


@pytest.mark.parametrize(
    "options",
    [
        {"level": "bad"},
        {"model_family": "bad"},
        {"fit_intercept": 1},
        {"by_source": "yes"},
        {"keep_cols": [1]},
    ],
)
def test_invalid_options_raise_clear_python_errors(options):
    with pytest.raises((ValueError, TypeError)):
        adjust_coefficient(simple_mpd(), simple_benchmark(), **options)


@pytest.mark.parametrize("bad", ["text", "1"])
def test_non_numeric_flow_values_are_rejected(bad):
    observed = simple_mpd().astype({"flow": object})
    observed.loc[0, "flow"] = bad
    with pytest.raises(TypeError, match="numeric"):
        adjust_coefficient(observed, simple_benchmark())


def test_non_dataframe_inputs_are_rejected():
    with pytest.raises(TypeError, match="DataFrame"):
        adjust_coefficient([], simple_benchmark())


def test_no_overlapping_sources_are_rejected():
    with pytest.raises(ValueError, match="No overlapping"):
        adjust_coefficient(
            simple_mpd(), simple_benchmark().assign(mpd_source="other"), by_source=True
        )


def test_count_models_warn_when_intercept_option_is_ignored():
    with pytest.warns(UserWarning, match="ignored"):
        actual = adjust_coefficient(
            simple_mpd(), simple_benchmark(), model_family="poisson", fit_intercept=True
        )
    assert actual.attrs["coef"] == pytest.approx(5.1)


def test_zero_ols_slope_retains_model_summary_and_missing_adjustments():
    actual = adjust_coefficient(simple_mpd(), simple_benchmark().assign(flow=0))
    assert actual.attrs["coef"] == 0
    assert actual.coef_factor.isna().all()
    assert actual.flow_adj.isna().all()


def test_negative_binomial_all_zero_response_reports_fit_failure():
    with pytest.raises(ValueError, match="fit failed"):
        adjust_coefficient(
            simple_mpd(), simple_benchmark().assign(flow=0), model_family="negbin"
        )


def test_missing_statsmodels_error_explains_the_failed_family(monkeypatch):
    import builtins

    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("statsmodels"):
            raise ImportError("dependency unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ValueError, match="Negative-binomial coefficient fit failed"):
        adjust_coefficient(simple_mpd(), simple_benchmark(), model_family="negbin")


def test_empty_observed_table_has_an_informative_error():
    with pytest.raises(ValueError, match="at least one observation"):
        adjust_coefficient(simple_mpd().iloc[:0], simple_benchmark())


def test_zinb_log_dispersion_fits_the_r_boundary_fixture():
    x = np.tile([1.0, 2.0, 4.0, 8.0], 5)
    y = [0, 0, 0, 8, 0, 0, 3, 7, 0, 0, 0, 10, 0, 1, 6, 8, 0, 0, 0, 12]
    observed = pd.DataFrame({"origin": range(20), "destination": range(20), "flow": x})
    actual = adjust_coefficient(observed, observed.assign(flow=y), model_family="zinb")
    # pscl::zeroinfl count intercept at the pinned R reference, 17-digit output.
    expected = math.exp(0.051907322206174798)
    assert actual.attrs["coef"] == pytest.approx(expected, abs=1e-5)
    np.testing.assert_allclose(
        actual.flow_adj, actual.flow * expected, atol=8e-5, rtol=0
    )
