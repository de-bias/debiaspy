"""Multi-method contracts based on debiasR's validation helper fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from debiaspy import (
    validate_flow_distribution_methods,
    validate_flow_margins,
    validate_flow_overall_methods,
    validate_flow_prepare_output,
    validate_flow_prepare_outputs,
    validate_flow_residual_methods,
    validate_flow_residual_structure_methods,
)


@pytest.fixture
def inputs():
    raw = pd.DataFrame(
        {"origin": list("AABB"), "destination": list("XYXY"), "flow": [100, 50, 25, 75]}
    )
    benchmark = raw.assign(flow=[90, 65, 35, 70])
    methods = {
        "method_a": raw.assign(flow_adj=[92, 62, 33, 72]),
        "method_b": raw.assign(flow_adj=[96, 58, 30, 78]),
    }
    return raw, benchmark, methods


def test_prepare_multiple_methods_retains_raw_adjusted_and_input_flows(inputs):
    raw, _, methods = inputs
    prepared = validate_flow_prepare_outputs(methods, raw)
    assert list(prepared) == ["method_a", "method_b"]
    for name, result in prepared.items():
        pd.testing.assert_series_equal(result.flow, raw.flow)
        pd.testing.assert_series_equal(
            result.adjusted_input_flow, methods[name].flow, check_names=False
        )
        pd.testing.assert_series_equal(result.flow_adj, methods[name].flow_adj)
    assert "adjusted_input_flow" not in methods["method_a"]


def test_multiple_methods_forward_options_and_keep_result_names(inputs):
    _, benchmark, methods = inputs
    overall = validate_flow_overall_methods(
        methods, benchmark, comparisons="adjusted_vs_benchmark"
    )
    residuals = validate_flow_residual_methods(methods, benchmark, top_n=2)
    distributions = validate_flow_distribution_methods(
        methods, benchmark, comparisons="adjusted_vs_benchmark", return_od_level=True
    )
    assert list(overall) == list(residuals) == list(distributions) == list(methods)
    assert overall["method_a"]["method"] == "method_a"
    assert overall["method_a"]["mae"] == pytest.approx(2.25)
    assert residuals["method_b"]["summary"].method.iloc[0] == "method_b"
    assert len(residuals["method_b"]["top_worst"]) == 2
    assert distributions["method_a"]["summary"].method.iloc[0] == "method_a"
    assert len(distributions["method_b"]["od_level"]) == 4
    selected = validate_flow_residual_structure_methods(
        {**methods, "unused_bad": None},
        benchmark,
        methods=["method_b"],
        area_neighbors=pd.DataFrame({"area": ["A", "B"], "neighbor": ["B", "A"]}),
    )
    assert list(selected) == ["method_b"]
    assert selected["method_b"]["summary"].method.iloc[0] == "method_b"
    with pytest.raises(ValueError, match="Missing: missing"):
        validate_flow_residual_structure_methods(
            methods, benchmark, methods=["missing"]
        )


@pytest.mark.parametrize(
    "role, expected_raw, expected_adjusted, expected_benchmark",
    [
        ("origin", [150, 100], [154, 105], [155, 105]),
        ("destination", [125, 125], [125, 134], [125, 135]),
    ],
)
def test_margin_totals_match_r_without_modifying_inputs(
    inputs, role, expected_raw, expected_adjusted, expected_benchmark
):
    _, benchmark, methods = inputs
    original = methods["method_a"].copy(deep=True)
    result = validate_flow_margins(methods, benchmark, role=role)
    first = result["adjusted_outputs"]["method_a"]
    np.testing.assert_allclose(first.flow, expected_raw)
    np.testing.assert_allclose(first.flow_adj, expected_adjusted)
    np.testing.assert_allclose(result["benchmark"].flow, expected_benchmark)
    pd.testing.assert_frame_equal(methods["method_a"], original)


@pytest.mark.parametrize(
    "function",
    [
        validate_flow_overall_methods,
        validate_flow_residual_methods,
        validate_flow_distribution_methods,
        validate_flow_residual_structure_methods,
        validate_flow_margins,
    ],
)
def test_validation_errors_identify_the_failing_method_and_column(inputs, function):
    _, benchmark, methods = inputs
    methods["method_b"] = methods["method_b"].drop(columns="flow_adj")
    with pytest.raises(ValueError, match="Method 'method_b'.*flow_adj") as caught:
        function(methods, benchmark)
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("bad", [None, [1, 2], pd.Series([1, 2])])
def test_preparation_rejects_non_dataframe_method_values_with_context(inputs, bad):
    raw, _, methods = inputs
    methods["method_b"] = bad
    with pytest.raises(ValueError, match="Method 'method_b'.*DataFrame"):
        validate_flow_prepare_outputs(methods, raw)


@pytest.mark.parametrize(
    "bad", [{}, {"": pd.DataFrame()}, {1: pd.DataFrame()}, [pd.DataFrame()]]
)
def test_method_names_require_a_nonempty_named_mapping(inputs, bad):
    raw, _, _ = inputs
    with pytest.raises(ValueError, match="adjusted_outputs"):
        validate_flow_prepare_outputs(bad, raw)


def test_preparation_missing_columns_and_ambiguous_repeated_keys_are_contextual(inputs):
    raw, _, methods = inputs
    with pytest.raises(ValueError, match="Method 'method_b'.*flow_adj"):
        validate_flow_prepare_outputs(
            {"method_b": methods["method_b"].drop(columns="flow_adj")}, raw
        )
    repeated = pd.concat([raw.assign(mpd_source="one"), raw.assign(mpd_source="two")])
    with pytest.raises(ValueError, match="Method 'method_a'"):
        validate_flow_prepare_outputs({"method_a": methods["method_a"]}, repeated)


def test_custom_preparation_columns_and_reserved_name_collision(inputs):
    raw, _, methods = inputs
    adjusted = methods["method_a"].rename(columns={"flow_adj": "corrected"})
    result = validate_flow_prepare_output(
        adjusted,
        raw.rename(columns={"flow": "trips"}),
        flow_col_mpd="trips",
        flow_col_adj="corrected",
    )
    np.testing.assert_allclose(result.flow_adj, [92, 62, 33, 72])
    with pytest.raises(ValueError, match="Method 'method_a'.*raw_flow"):
        validate_flow_prepare_outputs(
            {"method_a": methods["method_a"].assign(raw_flow=1)}, raw
        )
