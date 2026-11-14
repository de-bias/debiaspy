"""Numerical and input-contract checks for inverse penetration adjustment."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from debiaspy import adjust_inverse_penetration

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "inverse_penetration_r.json").read_text()
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


@pytest.fixture
def inputs():
    case = FIXTURE["basic"]
    return pd.DataFrame(case["mpd"]), pd.DataFrame(case["coverage"])


@pytest.mark.parametrize("case_name", FIXTURE)
@pytest.mark.parametrize("weight_by", ["origin", "destination", "both"])
def test_matches_executed_r_reference(case_name, weight_by):
    case = _decode(FIXTURE[case_name])
    flows = pd.DataFrame(case["mpd"])
    coverage = pd.DataFrame(case["coverage"])
    original_flows = flows.copy(deep=True)
    original_coverage = coverage.copy(deep=True)
    reference = case["expected"][weight_by]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = adjust_inverse_penetration(
            flows, coverage, flow_col=case.get("flow_col", "flow"), weight_by=weight_by
        )
    expected = pd.DataFrame(reference["data"], columns=reference["columns"])
    assert_frame_equal(result, expected, check_dtype=False, rtol=1e-14, atol=1e-14)
    assert_frame_equal(flows, original_flows)
    assert_frame_equal(coverage, original_coverage)
    assert (
        sum(issubclass(w.category, UserWarning) for w in caught)
        == reference["many_to_many"]
    )
    assert (
        sum(issubclass(w.category, RuntimeWarning) for w in caught)
        == reference["nan_warnings"]
    )
    assert len(caught) == reference["many_to_many"] + reference["nan_warnings"]


@pytest.mark.parametrize(
    "weight_by,weights,adjusted",
    [
        ("origin", [10, 10, 4, 3], [20, 0, 12, 12]),
        ("destination", [4, 3, 10, 10], [8, 0, 30, 40]),
        (
            "both",
            [np.sqrt(40), np.sqrt(30), np.sqrt(40), np.sqrt(30)],
            [2 * np.sqrt(40), 0, 3 * np.sqrt(40), 4 * np.sqrt(30)],
        ),
    ],
)
def test_explicit_weighting_calculations(inputs, weight_by, weights, adjusted):
    flows, coverage = inputs
    result = adjust_inverse_penetration(flows, coverage, weight_by=weight_by)
    np.testing.assert_allclose(result[f"weight_{weight_by}"], weights)
    np.testing.assert_allclose(result["flow_adj"], adjusted)
    assert result["flow"].tolist() == [2, 0, 3, 4]
    assert result["label"].tolist() == ["first", "zero", "third", "fourth"]


@pytest.mark.parametrize(
    "option,full",
    [
        (None, "origin"),
        ("o", "origin"),
        ("orig", "origin"),
        ("d", "destination"),
        ("b", "both"),
    ],
)
def test_default_and_r_style_unambiguous_prefixes(inputs, option, full):
    flows, coverage = inputs
    actual = adjust_inverse_penetration(flows, coverage, weight_by=option)
    expected = adjust_inverse_penetration(flows, coverage, weight_by=full)
    assert_frame_equal(actual, expected)
    if full == "origin":
        assert_frame_equal(adjust_inverse_penetration(flows, coverage), expected)


@pytest.mark.parametrize("bad", ["", "invalid", "ORIGIN", 1, [], ["origin", "both"]])
def test_invalid_weighting_option(inputs, bad):
    with pytest.raises(ValueError, match="weight_by.*origin.*destination.*both"):
        adjust_inverse_penetration(*inputs, weight_by=bad)


@pytest.mark.parametrize(
    "table,column",
    [
        (0, "origin"),
        (0, "destination"),
        (0, "flow"),
        (1, "origin"),
        (1, "population"),
        (1, "user_count"),
    ],
)
def test_missing_required_columns(inputs, table, column):
    data = list(inputs)
    data[table] = data[table].drop(columns=column)
    name = "mpd_od_df" if table == 0 else "coverage_df"
    with pytest.raises(ValueError, match=f"{name}.*{column}"):
        adjust_inverse_penetration(*data)


@pytest.mark.parametrize("table", [0, 1])
def test_requires_dataframes(inputs, table):
    data = list(inputs)
    data[table] = []
    with pytest.raises(TypeError, match="DataFrame"):
        adjust_inverse_penetration(*data)


@pytest.mark.parametrize(
    "table,column", [(0, "flow"), (1, "population"), (1, "user_count")]
)
def test_rejects_non_numeric_arithmetic_columns(inputs, table, column):
    data = [frame.copy() for frame in inputs]
    data[table][column] = "not numeric"
    with pytest.raises(ValueError, match=f"numeric.*{column}"):
        adjust_inverse_penetration(*data)


@pytest.mark.parametrize(
    "table,column", [(0, "flow"), (1, "population"), (1, "user_count")]
)
def test_rejects_complex_counts_without_silently_discarding_imaginary_parts(
    inputs, table, column
):
    data = [frame.copy() for frame in inputs]
    data[table][column] = 1 + 2j
    with pytest.raises(ValueError, match="real-valued"):
        adjust_inverse_penetration(*data)


@pytest.mark.parametrize("flow_col", [None, "", 42])
def test_requires_nonempty_flow_column_name(inputs, flow_col):
    with pytest.raises(TypeError, match="Column names"):
        adjust_inverse_penetration(*inputs, flow_col=flow_col)


def test_nullable_numerics_and_missing_users():
    flows = pd.DataFrame(
        {
            "origin": ["A", "B", "C"],
            "destination": ["B", "C", "A"],
            "flow": pd.array([2, None, 0], dtype="Int64"),
        }
    )
    coverage = pd.DataFrame(
        {
            "origin": ["A", "B", "C"],
            "population": pd.array([100, 200, None], dtype="Int64"),
            "user_count": pd.array([10, None, 20], dtype="Int64"),
        }
    )
    result = adjust_inverse_penetration(flows, coverage)
    np.testing.assert_allclose(
        result["weight_origin"], [10, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(result["flow_adj"], [20, np.nan, np.nan], equal_nan=True)
    assert result["flow"].dtype == flows["flow"].dtype


def test_zero_users_and_zero_flows_follow_r_without_runtime_warning():
    flows = pd.DataFrame(
        {"origin": ["A", "A", "B"], "destination": ["B"] * 3, "flow": [2, 0, 3]}
    )
    coverage = pd.DataFrame(
        {"origin": ["A", "B"], "population": [100, 0], "user_count": [0, 0]}
    )
    with warnings.catch_warnings(record=True) as caught:
        result = adjust_inverse_penetration(flows, coverage)
    assert not caught
    np.testing.assert_allclose(
        result["weight_origin"], [np.inf, np.inf, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        result["flow_adj"], [np.inf, np.nan, np.nan], equal_nan=True
    )


def test_negative_geometric_product_warns_but_is_not_rejected():
    flows = pd.DataFrame({"origin": ["A"], "destination": ["B"], "flow": [2]})
    coverage = pd.DataFrame(
        {"origin": ["A", "B"], "population": [-10, 20], "user_count": [1, 2]}
    )
    with pytest.warns(RuntimeWarning, match="NaNs produced"):
        result = adjust_inverse_penetration(flows, coverage, weight_by="both")
    assert result["weight_origin"].iloc[0] == -10
    assert np.isnan(result["weight_both"].iloc[0])
    assert np.isnan(result["flow_adj"].iloc[0])


@pytest.mark.parametrize("mode", ["origin", "destination", "both"])
def test_empty_flows_and_empty_coverage(inputs, mode):
    flows, coverage = inputs
    empty_result = adjust_inverse_penetration(flows.iloc[:0], coverage, weight_by=mode)
    expected_columns = list(adjust_inverse_penetration(flows, coverage, weight_by=mode))
    assert empty_result.empty
    assert list(empty_result) == expected_columns
    missing_result = adjust_inverse_penetration(
        flows, coverage.iloc[:0], weight_by=mode
    )
    assert len(missing_result) == len(flows)
    assert missing_result["flow_adj"].isna().all()


def test_generated_values_are_overwritten_without_mutating_inputs(inputs):
    flows, coverage = inputs
    flows = flows.assign(flow_adj=-999, weight_both=-999)
    original = flows.copy(deep=True)
    result = adjust_inverse_penetration(flows, coverage, weight_by="both")
    assert result["flow_adj"].tolist() != [-999] * len(flows)
    assert result["weight_both"].tolist() != [-999] * len(flows)
    assert_frame_equal(flows, original)
    result.loc[0, "label"] = "changed"
    assert_frame_equal(flows, original)


@pytest.mark.parametrize(
    "mode,column",
    [
        ("origin", "weight_origin"),
        ("destination", "weight_destination"),
        ("both", "weight_origin"),
        ("both", "weight_destination"),
    ],
)
def test_rejects_colliding_joined_weight_columns(inputs, mode, column):
    flows, coverage = inputs
    with pytest.raises(ValueError, match=f"conflicting.*{column}"):
        adjust_inverse_penetration(
            flows.assign(**{column: 1}), coverage, weight_by=mode
        )


def test_rejects_colliding_custom_flow_and_duplicate_column_names(inputs):
    flows, coverage = inputs
    with pytest.raises(ValueError, match="duplicate columns"):
        adjust_inverse_penetration(flows.assign(trips=1), coverage, flow_col="trips")
    for bad_flows, bad_coverage in [
        (pd.concat([flows, flows[["flow"]]], axis=1), coverage),
        (flows, pd.concat([coverage, coverage[["population"]]], axis=1)),
    ]:
        with pytest.raises(ValueError, match="unique column names"):
            adjust_inverse_penetration(bad_flows, bad_coverage)


def test_unmatched_duplicate_keys_do_not_emit_many_to_many_warning(inputs):
    flows, coverage = inputs
    duplicate_unmatched = pd.DataFrame(
        {"origin": ["unused", "unused"], "population": [1, 2], "user_count": [1, 1]}
    )
    with warnings.catch_warnings(record=True) as caught:
        result = adjust_inverse_penetration(
            flows, pd.concat([coverage, duplicate_unmatched])
        )
    assert not caught
    assert len(result) == len(flows)


def test_positional_output_index_and_duplicate_input_indices(inputs):
    flows, coverage = inputs
    flows.index = [7, 2, 7, 9]
    coverage.index = [4, 4, 8]
    expected = adjust_inverse_penetration(
        flows.reset_index(drop=True), coverage.reset_index(drop=True)
    )
    assert_frame_equal(adjust_inverse_penetration(flows, coverage), expected)
