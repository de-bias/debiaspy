"""Preparation parity fixtures follow debiasR's test-validate-flow-methods.R."""

import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from debiaspy import (
    validate_flow_margins,
    validate_flow_prepare_output,
    validate_flow_prepare_outputs,
)


@pytest.fixture
def flows():
    raw = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["X", "Y", "X", "Y"],
            "flow": [100, 50, 25, 75],
        }
    )
    benchmark = raw.assign(flow=[90, 65, 35, 70])
    methods = OrderedDict(
        method_a=raw.assign(flow_adj=[92, 62, 33, 72]),
        method_b=raw.assign(flow_adj=[96, 58, 30, 78]),
    )
    return raw, benchmark, methods


def test_prepare_matches_r_fixture_and_preserves_inputs(flows):
    raw, _, methods = flows
    originals = {name: data.copy(deep=True) for name, data in methods.items()}
    raw_original = raw.copy(deep=True)
    result = validate_flow_prepare_outputs(methods, raw)
    assert list(result) == ["method_a", "method_b"]
    for name, data in result.items():
        expected = originals[name].rename(columns={"flow": "adjusted_input_flow"})
        expected["flow"] = [100, 50, 25, 75]
        assert_frame_equal(data, expected)
        assert_frame_equal(methods[name], originals[name])
    assert_frame_equal(raw, raw_original)
    result["method_a"].loc[0, "flow_adj"] = -1
    assert_frame_equal(methods["method_a"], originals["method_a"])


def test_prepare_custom_columns_left_join_order_and_metadata(flows):
    raw, _, methods = flows
    adjusted = (
        methods["method_a"]
        .drop(columns="flow")
        .rename(columns={"flow_adj": "corrected"})
        .iloc[[3, 0, 1]]
        .assign(note=["last", "first", "unmatched"])
    )
    result = validate_flow_prepare_output(
        adjusted,
        raw.iloc[[0, 3]].rename(columns={"flow": "count"}),
        flow_col_mpd="count",
        flow_col_adj="corrected",
    )
    expected = pd.DataFrame(
        {
            "origin": ["B", "A", "A"],
            "destination": ["Y", "X", "Y"],
            "flow_adj": [72, 92, 62],
            "note": ["last", "first", "unmatched"],
            "flow": [75.0, 100.0, np.nan],
        }
    )
    assert_frame_equal(result, expected)


def test_prepare_missing_keys_match_and_zero_flows_are_retained():
    raw = pd.DataFrame(
        {"origin": [None, "A"], "destination": ["X", "X"], "flow": [0, 2]}
    )
    adjusted = raw.drop(columns="flow").assign(flow_adj=[1, 3])
    result = validate_flow_prepare_output(adjusted, raw)
    assert result["flow"].tolist() == [0, 2]
    assert result["flow_adj"].tolist() == [1, 3]


def test_prepare_duplicate_keys_expand_and_warn_only_for_many_to_many(flows):
    raw, _, methods = flows
    duplicate_raw = raw.iloc[[0, 0]]
    with pytest.warns(UserWarning, match="Many-to-many"):
        result = validate_flow_prepare_output(
            methods["method_a"].iloc[[0, 0]], duplicate_raw
        )
    assert len(result) == 4
    assert result["flow"].tolist() == [100] * 4
    with warnings.catch_warnings(record=True) as caught:
        result = validate_flow_prepare_output(
            methods["method_a"].iloc[[0]], duplicate_raw
        )
    assert len(result) == 2
    assert not caught


@pytest.mark.parametrize(
    "role,expected_raw,expected_a,expected_b,expected_bench",
    [
        ("origin", [150, 100], [154, 105], [154, 108], [155, 105]),
        ("destination", [125, 125], [125, 134], [126, 136], [125, 135]),
    ],
)
def test_margins_match_fixed_r_fixture(
    flows, role, expected_raw, expected_a, expected_b, expected_bench
):
    raw, benchmark, methods = flows
    original_benchmark = benchmark.copy(deep=True)
    originals = {name: data.copy(deep=True) for name, data in methods.items()}
    prepared = validate_flow_prepare_outputs(methods, raw)
    result = validate_flow_margins(prepared, benchmark, role=role)
    assert list(result) == ["adjusted_outputs", "benchmark"]
    assert list(result["adjusted_outputs"]) == ["method_a", "method_b"]
    keys = (
        {"origin": ["A", "B"], "destination": ["origin_total"] * 2}
        if role == "origin"
        else {"origin": ["destination_total"] * 2, "destination": ["X", "Y"]}
    )
    for name, adjusted in [("method_a", expected_a), ("method_b", expected_b)]:
        expected = pd.DataFrame({**keys, "flow": expected_raw, "flow_adj": adjusted})
        assert_frame_equal(result["adjusted_outputs"][name], expected)
        assert_frame_equal(methods[name], originals[name])
    assert_frame_equal(
        result["benchmark"], pd.DataFrame({**keys, "flow": expected_bench})
    )
    assert_frame_equal(benchmark, original_benchmark)


def test_margins_custom_columns_missing_values_and_missing_area():
    data = pd.DataFrame(
        {
            "origin": ["B", "A", "A", None, "C"],
            "destination": ["X"] * 5,
            "raw": [np.nan, 0, 2, 3, np.nan],
            "adjusted": [4, np.nan, 6, 7, np.nan],
        }
    )
    benchmark = data.rename(columns={"raw": "bench"})
    result = validate_flow_margins(
        {"m": data},
        benchmark,
        flow_col_mpd="raw",
        flow_col_adj="adjusted",
        flow_col_bench="bench",
    )
    expected = pd.DataFrame(
        {
            "origin": ["A", "B", "C", np.nan],
            "destination": ["origin_total"] * 4,
            "flow": [2.0, 0.0, 0.0, 3.0],
            "flow_adj": [6.0, 4.0, 0.0, 7.0],
        }
    )
    assert_frame_equal(result["adjusted_outputs"]["m"], expected)
    assert_frame_equal(result["benchmark"], expected.drop(columns="flow_adj"))


def test_margins_do_not_align_distinct_method_supports(flows):
    _, benchmark, methods = flows
    selected = {
        "method_b": methods["method_b"].iloc[2:],
        "method_a": methods["method_a"].iloc[:2],
    }
    result = validate_flow_margins(selected, benchmark)
    assert list(result["adjusted_outputs"]) == ["method_b", "method_a"]
    assert result["adjusted_outputs"]["method_b"]["origin"].tolist() == ["B"]
    assert result["adjusted_outputs"]["method_a"]["origin"].tolist() == ["A"]
    assert result["benchmark"]["origin"].tolist() == ["A", "B"]


def test_empty_mapping_and_empty_tables(flows):
    raw, benchmark, methods = flows
    assert validate_flow_prepare_outputs({}, raw) == {}
    assert validate_flow_margins({}, benchmark)["adjusted_outputs"] == {}
    empty = methods["method_a"].iloc[:0]
    prepared = validate_flow_prepare_output(empty, raw)
    assert prepared.empty
    assert list(prepared) == [
        "origin",
        "destination",
        "adjusted_input_flow",
        "flow_adj",
        "flow",
    ]
    margins = validate_flow_margins({"m": empty}, benchmark.iloc[:0])
    assert margins["adjusted_outputs"]["m"].empty
    assert list(margins["adjusted_outputs"]["m"]) == [
        "origin",
        "destination",
        "flow",
        "flow_adj",
    ]
    assert margins["benchmark"].empty


@pytest.mark.parametrize(
    "bad,exception,match",
    [
        ([], TypeError, "mapping"),
        ({"": pd.DataFrame()}, ValueError, "non-empty strings"),
        ({1: pd.DataFrame()}, ValueError, "non-empty strings"),
        ({"broken": None}, TypeError, "broken.*DataFrame"),
    ],
)
def test_invalid_method_collections_identify_problem(flows, bad, exception, match):
    raw, benchmark, _ = flows
    with pytest.raises(exception, match=match):
        validate_flow_prepare_outputs(bad, raw)
    with pytest.raises(exception, match=match):
        validate_flow_margins(bad, benchmark)


def test_missing_columns_identify_method_and_input(flows):
    raw, benchmark, methods = flows
    methods["method_b"] = methods["method_b"].drop(columns="flow_adj")
    for call in [
        lambda: validate_flow_prepare_outputs(methods, raw),
        lambda: validate_flow_margins(methods, benchmark),
    ]:
        with pytest.raises(ValueError, match="method_b.*flow_adj"):
            call()
    with pytest.raises(ValueError, match="method_a.*mpd_df.*flow"):
        validate_flow_prepare_outputs(methods, raw.drop(columns="flow"))
    with pytest.raises(ValueError, match="benchmark_od_df.*flow"):
        validate_flow_margins(methods, benchmark.drop(columns="flow"))


@pytest.mark.parametrize(
    "extra,flow_col_adj",
    [
        ({"adjusted_input_flow": [0] * 4}, "flow_adj"),
        ({"corrected": [0] * 4}, "corrected"),
        ({"raw_flow": [0] * 4}, "flow_adj"),
    ],
)
def test_preparation_rejects_column_collisions(flows, extra, flow_col_adj):
    raw, _, methods = flows
    with pytest.raises(ValueError, match="conflicting|reserved"):
        validate_flow_prepare_output(
            methods["method_a"].assign(**extra), raw, flow_col_adj=flow_col_adj
        )


def test_invalid_columns_types_roles_and_duplicate_columns(flows):
    raw, benchmark, methods = flows
    data = methods["method_a"]
    with pytest.raises(ValueError, match="role"):
        validate_flow_margins(methods, benchmark, role="both")
    with pytest.raises(ValueError, match="bad.*numeric.*flow_adj"):
        validate_flow_margins({"bad": data.assign(flow_adj="text")}, benchmark)
    with pytest.raises(TypeError, match="Column names"):
        validate_flow_prepare_output(data, raw, flow_col_mpd=None)
    with pytest.raises(TypeError, match="adj_df.*DataFrame"):
        validate_flow_prepare_output([], raw)
    with pytest.raises(ValueError, match="separate adjusted"):
        validate_flow_prepare_output(data, raw, flow_col_adj="flow")
    duplicated = pd.concat([data, data[["flow_adj"]]], axis=1)
    with pytest.raises(ValueError, match="unique column"):
        validate_flow_prepare_output(duplicated, raw)
    with pytest.raises(ValueError, match="unique column"):
        validate_flow_margins({"bad": duplicated}, benchmark)


def test_categorical_margins_omit_unused_areas(flows):
    _, benchmark, methods = flows
    data = methods["method_a"].copy()
    data["origin"] = pd.Categorical(data["origin"], categories=["A", "B", "unused"])
    result = validate_flow_margins({"m": data}, benchmark)
    assert result["adjusted_outputs"]["m"]["origin"].tolist() == ["A", "B"]
    assert result["adjusted_outputs"]["m"]["flow_adj"].tolist() == [154, 105]
