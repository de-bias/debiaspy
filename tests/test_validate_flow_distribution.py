"""Distribution validation against executed R fixtures and public API examples."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import debiaspy
from debiaspy import measure_bias_distribution, validate_flow_distribution

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "flow_distribution_r.json").read_text()
)


def _decode(value):
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if value == "Infinity":
        return np.inf
    if value == "-Infinity":
        return -np.inf
    return value


def _inputs(case):
    frames = [pd.DataFrame(case[key]) for key in ("adj", "benchmark")]
    for frame in frames:
        for column in frame:
            if column in ("origin", "destination", "mpd_source"):
                frame[column] = frame[column].astype(object)
            else:
                frame[column] = pd.to_numeric(frame[column]).astype(float)
    return frames


@pytest.mark.parametrize("name", CASES)
def test_matches_executed_r_reference(name):
    case = _decode(CASES[name])
    inputs = _inputs(case)
    originals = [frame.copy(deep=True) for frame in inputs]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in case:
            with pytest.raises(ValueError, match="Cannot compute distribution shares"):
                validate_flow_distribution(*inputs, **case["options"])
        else:
            result = validate_flow_distribution(*inputs, **case["options"])
            assert list(result) == list(case["expected"])
            for key, expected in case["expected"].items():
                table = pd.DataFrame(
                    [
                        [np.nan if v is None else v for v in row]
                        for row in expected["data"]
                    ],
                    columns=expected["columns"],
                )
                assert_frame_equal(
                    result[key], table, check_dtype=False, rtol=1e-12, atol=1e-12
                )
    assert (
        any("many-to-many" in str(item.message) for item in caught)
        == case["warnings"]["many"]
    )
    assert (
        any("NaNs produced" in str(item.message) for item in caught)
        == case["warnings"]["nan"]
    )
    assert all(
        "many-to-many" in str(item.message) or "NaNs produced" in str(item.message)
        for item in caught
    )
    for actual, original in zip(inputs, originals, strict=True):
        assert_frame_equal(actual, original)


@pytest.fixture
def inputs():
    return _inputs(_decode(CASES["weighted_all"]))


def test_explicit_shares_contributions_and_kl_direction():
    adjusted = pd.DataFrame(
        {"origin": ["A", "A"], "destination": ["X", "Y"], "flow_adj": [0, 3]}
    )
    benchmark = adjusted.drop(columns="flow_adj").assign(flow=[3, 0])
    result = validate_flow_distribution(
        adjusted, benchmark, epsilon=1, return_od_level=True
    )
    od = result["od_level"]
    np.testing.assert_allclose(od["reference_share"], [0.8, 0.2])
    np.testing.assert_allclose(od["comparison_share"], [0.2, 0.8])
    np.testing.assert_allclose(od["midpoint_share"], [0.5, 0.5])
    np.testing.assert_allclose(
        od["kl_contribution"], [0.8 * np.log(4), 0.2 * np.log(0.25)]
    )
    np.testing.assert_allclose(od["jsd_contribution"], [0.0963723785108787] * 2)
    assert result["summary"].loc[0, "kl_mean"] == pytest.approx(0.8317766166719344)
    assert result["summary"].loc[0, "jsd_mean"] == pytest.approx(0.1927447570217574)
    assert result["summary"].loc[0, "method"] is None
    assert np.isnan(result["summary"].loc[0, "kl_weighted_mean"])


def test_asymmetric_kl_and_measurement_use_the_same_shares_and_contributions():
    coverage = pd.DataFrame(
        {
            "origin": ["X", "Y", "Z"],
            "population": [100, 200, 500],
            "user_count": [50, 60, 70],
        }
    )
    measured = measure_bias_distribution(coverage, epsilon=0.1)
    adjusted = pd.DataFrame(
        {
            "origin": ["A"] * 3,
            "destination": coverage["origin"],
            "flow_adj": coverage["user_count"],
        }
    )
    benchmark = adjusted.drop(columns="flow_adj").assign(flow=coverage["population"])
    validated = validate_flow_distribution(
        adjusted, benchmark, epsilon=0.1, return_od_level=True
    )
    for measured_col, validated_col in [
        ("population_share", "reference_share"),
        ("user_share", "comparison_share"),
        ("kl_contribution", "kl_contribution"),
        ("jsd_contribution", "jsd_contribution"),
    ]:
        np.testing.assert_allclose(
            measured["area_level"][measured_col],
            validated["od_level"][validated_col],
            atol=1e-14,
        )
    reversed_result = validate_flow_distribution(
        adjusted.assign(flow_adj=coverage["population"]),
        benchmark.assign(flow=coverage["user_count"]),
        epsilon=0.1,
    )
    assert validated["summary"].loc[0, "kl_mean"] != pytest.approx(
        reversed_result["summary"].loc[0, "kl_mean"]
    )
    assert validated["summary"].loc[0, "jsd_mean"] == pytest.approx(
        reversed_result["summary"].loc[0, "jsd_mean"]
    )


def test_raw_vs_adjusted_weighting_still_uses_benchmark_totals(inputs):
    result = validate_flow_distribution(
        *inputs, comparisons="raw_vs_adjusted", weight_by="benchmark_origin_total"
    )
    origins = result["origin_level"]
    assert origins["origin"].tolist() == ["A", "B"]
    assert origins["bench_origin_total"].tolist() == [300, 100]
    for metric in ("kl", "jsd"):
        expected = (
            0.75 * origins.loc[0, f"{metric}_origin"]
            + 0.25 * origins.loc[1, f"{metric}_origin"]
        )
        assert result["summary"].loc[0, f"{metric}_weighted_mean"] == pytest.approx(
            expected
        )


def test_zero_reference_keeps_od_contributions_but_excludes_origin_metrics():
    adjusted = pd.DataFrame(
        {"origin": ["A", "A"], "destination": ["X", "Y"], "flow_adj": [2, 5]}
    )
    benchmark = adjusted.drop(columns="flow_adj").assign(flow=0)
    result = validate_flow_distribution(
        adjusted, benchmark, return_od_level=True, weight_by="benchmark_origin_total"
    )
    assert result["summary"].loc[0, "n_origins_used"] == 0
    assert result["origin_level"].loc[0, "zero_reference_total"]
    assert np.isnan(result["origin_level"].loc[0, "kl_origin"])
    assert np.isfinite(result["od_level"]["kl_contribution"]).all()
    assert (
        result["summary"][
            [
                "kl_mean",
                "kl_median",
                "kl_weighted_mean",
                "jsd_mean",
                "jsd_median",
                "jsd_weighted_mean",
            ]
        ]
        .isna()
        .all()
        .all()
    )


def test_source_labels_are_ignored_and_duplicates_are_not_aggregated(inputs):
    adjusted, benchmark = inputs
    repeated = pd.concat(
        [adjusted.assign(mpd_source="one"), adjusted.assign(mpd_source="two")],
        ignore_index=True,
    )
    result = validate_flow_distribution(
        repeated, benchmark.assign(mpd_source="other"), return_od_level=True
    )
    assert len(result["od_level"]) == 8
    assert "mpd_source" not in result["od_level"]
    assert result["origin_level"]["n_destinations"].tolist() == [4, 4]


@pytest.mark.parametrize("epsilon", [0, -1, np.inf, np.nan, True, "small", [1], None])
def test_invalid_epsilon(inputs, epsilon):
    with pytest.raises(ValueError, match="epsilon.*positive finite"):
        validate_flow_distribution(*inputs, epsilon=epsilon)


@pytest.mark.parametrize("weight_by", ["", "invalid", [], 3])
def test_invalid_weighting(inputs, weight_by):
    with pytest.raises(ValueError, match="weight_by"):
        validate_flow_distribution(*inputs, weight_by=weight_by)


@pytest.mark.parametrize(
    "comparisons", [[], "bad", ["all", "raw_vs_adjusted"], None, {"raw_vs_adjusted"}]
)
def test_invalid_comparisons(inputs, comparisons):
    with pytest.raises(ValueError, match="comparisons"):
        validate_flow_distribution(*inputs, comparisons=comparisons)


@pytest.mark.parametrize(
    "table,column",
    [
        (0, "origin"),
        (0, "destination"),
        (0, "flow_adj"),
        (1, "origin"),
        (1, "destination"),
        (1, "flow"),
    ],
)
def test_required_columns(inputs, table, column):
    inputs[table] = inputs[table].drop(columns=column)
    with pytest.raises(ValueError, match=column):
        validate_flow_distribution(*inputs)


@pytest.mark.parametrize("comparison", ["all", "raw_vs_benchmark", "raw_vs_adjusted"])
def test_raw_flow_is_optional_only_for_adjusted_comparison(inputs, comparison):
    adjusted, benchmark = inputs
    adjusted = adjusted.drop(columns="flow")
    assert len(validate_flow_distribution(adjusted, benchmark)["summary"]) == 1
    with pytest.raises(ValueError, match="adj_df.*flow"):
        validate_flow_distribution(adjusted, benchmark, comparisons=comparison)


def test_invalid_data_types_columns_and_controls(inputs):
    adjusted, benchmark = inputs
    with pytest.raises(TypeError, match="DataFrame"):
        validate_flow_distribution([], benchmark)
    with pytest.raises(ValueError, match="numeric"):
        validate_flow_distribution(adjusted.assign(flow_adj="bad"), benchmark)
    with pytest.raises(ValueError, match="real-valued"):
        validate_flow_distribution(adjusted.assign(flow_adj=1j), benchmark)
    with pytest.raises(ValueError, match="unique column"):
        validate_flow_distribution(
            pd.concat([adjusted, adjusted[["flow_adj"]]], axis=1), benchmark
        )
    with pytest.raises(ValueError, match="distinct"):
        validate_flow_distribution(adjusted, benchmark, flow_col_adj="flow")
    with pytest.raises(TypeError, match="method_name"):
        validate_flow_distribution(*inputs, method_name=3)
    for control in ("return_origin_level", "return_od_level"):
        with pytest.raises(TypeError, match=control):
            validate_flow_distribution(*inputs, **{control: "false"})


def test_nullable_values_nondefault_indices_and_independent_results(inputs):
    adjusted, benchmark = inputs
    adjusted = adjusted.assign(flow_adj=pd.array([20, None, 30, 70], dtype="Int64"))
    adjusted.index = [9, 3, 9, 1]
    benchmark.index = [4, 4, 7, 2]
    original_adjusted, original_benchmark = (
        adjusted.copy(deep=True),
        benchmark.copy(deep=True),
    )
    result = validate_flow_distribution(adjusted, benchmark, return_od_level=True)
    assert result["od_level"]["adjusted_flow"].tolist() == [20, 0, 30, 70]
    assert result["od_level"].index.tolist() == [0, 1, 2, 3]
    result["od_level"].loc[0, "adjusted_flow"] = -999
    assert_frame_equal(adjusted, original_adjusted)
    assert_frame_equal(benchmark, original_benchmark)


def test_empty_adjusted_without_optional_raw_column(inputs):
    adjusted, benchmark = inputs
    result = validate_flow_distribution(
        adjusted.iloc[:0].drop(columns="flow"), benchmark
    )
    assert result["origin_level"]["zero_adj_total"].all()
    assert np.isfinite(result["origin_level"]["kl_origin"]).all()


def test_public_export():
    assert "validate_flow_distribution" in debiaspy.__all__


def test_none_weighting_matches_default(inputs):
    default = validate_flow_distribution(*inputs)
    explicit_none = validate_flow_distribution(*inputs, weight_by=None)
    assert_frame_equal(default["summary"], explicit_none["summary"])


def test_invalid_totals_identify_origin_and_comparison(inputs):
    adjusted, benchmark = inputs
    adjusted.loc[adjusted["origin"] == "B", "flow_adj"] = np.inf
    with pytest.raises(
        ValueError, match="adjusted_vs_benchmark, origin 'B'.*non-finite"
    ):
        validate_flow_distribution(adjusted, benchmark)
