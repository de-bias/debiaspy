"""Compare public flow validators with fixed results from the R reference."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import debiaspy
from debiaspy import validate_flow_overall, validate_flow_pairs

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "flow_validation_r.json").read_text()
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
    adj, bench = pd.DataFrame(case["adj"]), pd.DataFrame(case["benchmark"])
    options = case["options"]
    for table, columns in [
        (
            adj,
            [
                options.get("flow_col_adj", "flow_adj"),
                options.get("flow_col_mpd", "flow"),
            ],
        ),
        (bench, [options.get("flow_col_bench", "flow")]),
    ]:
        for column in columns:
            if column in table:
                table[column] = pd.to_numeric(table[column])
    return adj, bench


def _expected(spec):
    if not spec["columns"]:
        return pd.DataFrame()
    # JSON null encodes R's numeric NA/NaN, including wholly missing columns.
    data = [
        [np.nan if value is None else value for value in row] for row in spec["data"]
    ]
    return pd.DataFrame(data, columns=spec["columns"])


def _check_warnings(caught, expected):
    assert (
        sum(issubclass(item.category, UserWarning) for item in caught)
        == expected["many"]
    )
    assert (
        sum(issubclass(item.category, RuntimeWarning) for item in caught)
        == expected["perfect"]
    )
    assert len(caught) == sum(expected.values())


@pytest.mark.parametrize("name", CASES)
def test_overall_matches_executed_r_reference(name):
    case = _decode(CASES[name])
    adj, bench = _inputs(case)
    originals = adj.copy(deep=True), bench.copy(deep=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validate_flow_overall(
            adj, bench, method_name="parity", **case["options"]
        )
    actual_summary = result.get(
        "summary",
        pd.DataFrame(
            [
                {
                    key: value
                    for key, value in result.items()
                    if key not in ("data", "by_source")
                }
            ]
        ),
    )
    expected_summary = _expected(case["expected"]["summary"])
    relaxed_columns = ["ols_slope", "r_squared"] if name == "rank_deficient" else []
    if relaxed_columns:
        # R's uncentered QR and centered NumPy arithmetic differ at ~1e-8 when
        # y is near 1e8 but varies by only 1. Keep all other checks at 1e-12.
        assert_frame_equal(
            actual_summary[relaxed_columns],
            expected_summary[relaxed_columns],
            check_dtype=False,
            rtol=1e-12,
            atol=5e-8,
        )
    assert_frame_equal(
        actual_summary.drop(columns=relaxed_columns),
        expected_summary.drop(columns=relaxed_columns),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    for key in ("data", "by_source"):
        assert (key in result) == (key in case["expected"])
        if key in result:
            assert_frame_equal(
                result[key],
                _expected(case["expected"][key]),
                check_dtype=False,
                rtol=1e-12,
                atol=1e-12,
            )
    _check_warnings(caught, case["warnings"]["overall"])
    assert_frame_equal(adj, originals[0])
    assert_frame_equal(bench, originals[1])


@pytest.mark.parametrize(
    "name", [name for name, case in CASES.items() if "pairs" in case["expected"]]
)
def test_pairs_match_executed_r_reference(name):
    case = _decode(CASES[name])
    adj, bench = _inputs(case)
    options = {
        key: value
        for key, value in case["options"].items()
        if key.startswith("flow_col_")
    }
    originals = adj.copy(deep=True), bench.copy(deep=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validate_flow_pairs(adj, bench, **options)
    assert_frame_equal(
        result,
        _expected(case["expected"]["pairs"]),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    _check_warnings(caught, case["warnings"]["pairs"])
    assert_frame_equal(adj, originals[0])
    assert_frame_equal(bench, originals[1])


@pytest.fixture
def inputs():
    return _inputs(_decode(CASES["basic"]))


def test_explicit_metric_values_and_scalar_contract():
    adj = pd.DataFrame(
        {
            "origin": ["A", "A", "B"],
            "destination": ["B", "C", "C"],
            "flow_adj": [10, 20, 30],
        }
    )
    bench = adj.drop(columns="flow_adj").assign(flow=[12, 18, 33])
    result = validate_flow_overall(adj, bench, method_name="demo")
    assert "summary" not in result
    assert result["method"] == "demo"
    assert result["n"] == 3
    assert (
        result["sum_adj"] == 60 and result["sum_bench"] == 63 and result["sum_mpd"] == 0
    )
    assert result["mean_error"] == pytest.approx(1)
    assert result["rmse"] == pytest.approx(np.sqrt(17 / 3))
    assert result["mae"] == pytest.approx(7 / 3)
    assert result["mape"] == pytest.approx((2 / 12 + 2 / 18 + 3 / 33) / 3)
    assert result["ols_intercept"] == pytest.approx(0, abs=1e-12)
    assert result["ols_slope"] == pytest.approx(1.05)
    assert result["r_squared"] == pytest.approx(0.9423076923076923)
    assert result["pearson_r"] == pytest.approx(0.9707253433941511)
    assert result["spearman_rho"] == pytest.approx(1)


def test_comparison_direction_and_legacy_difference_signs(inputs):
    adj, bench = inputs
    pairs = validate_flow_pairs(adj, bench)
    assert pairs["diff_mpd_benchmark"].tolist() == [5, 10, -15]
    assert pairs["diff_mpd_adj"].tolist() == [10, 5, -10]
    assert pairs["diff_adj_benchmark"].tolist() == [-5, 5, -5]
    overall = validate_flow_overall(adj, bench, comparisons="all")
    expected = {
        "adjusted_vs_benchmark": [5, -5, 5],
        "raw_vs_benchmark": [-5, -10, 15],
        "raw_vs_adjusted": [-10, -5, 10],
    }
    for comparison, values in expected.items():
        assert pairs[f"signed_residual_{comparison}"].tolist() == values
        assert (
            overall["data"]
            .loc[overall["data"]["comparison"] == comparison, "residual"]
            .tolist()
            == values
        )


def test_order_deduplication_and_return_joined(inputs):
    result = validate_flow_overall(
        *inputs,
        comparisons=["raw_vs_adjusted", "raw_vs_benchmark", "raw_vs_adjusted"],
        return_joined=False,
    )
    assert list(result) == ["method", "summary"]
    assert result["method"] is None
    assert result["summary"]["comparison"].tolist() == [
        "raw_vs_adjusted",
        "raw_vs_benchmark",
    ]


def test_sources_do_not_change_od_join_keys():
    adj = pd.DataFrame(
        {
            "origin": ["A", "A"],
            "destination": ["B", "B"],
            "mpd_source": ["s1", "s2"],
            "flow": [2, 3],
            "flow_adj": [4, 5],
        }
    )
    bench = pd.DataFrame(
        {
            "origin": ["A", "A"],
            "destination": ["B", "B"],
            "mpd_source": ["s1", "s2"],
            "flow": [6, 7],
        }
    )
    with pytest.warns(UserWarning, match="many-to-many"):
        pairs = validate_flow_pairs(adj, bench)
    assert pairs["benchmark_flow"].tolist() == [6, 7, 6, 7]
    assert pairs["mpd_source"].tolist() == ["s1", "s1", "s2", "s2"]


def test_na_rm_false_uses_metric_specific_missing_value_rules():
    adj = pd.DataFrame(
        {
            "origin": ["A", "B", "C"],
            "destination": ["X"] * 3,
            "flow_adj": [1, np.nan, 3],
        }
    )
    bench = adj.drop(columns="flow_adj").assign(flow=[2, 4, 7])
    result = validate_flow_overall(adj, bench, na_rm=False, drop_zeros=False)
    assert result["n"] == 3 and result["sum_x"] == 4
    assert result["mean_error"] == pytest.approx(2.5)
    assert result["mape"] == pytest.approx((1 / 2 + 4 / 7) / 2)
    assert all(
        np.isnan(result[key])
        for key in ["rmse", "mae", "pearson_r", "spearman_rho", "ols_slope"]
    )


@pytest.mark.parametrize("comparison", ["all", "raw_vs_benchmark", "raw_vs_adjusted"])
def test_raw_flows_required_only_when_comparison_uses_them(inputs, comparison):
    adj, bench = inputs
    adj = adj.drop(columns="flow")
    assert validate_flow_overall(adj, bench)["n"] == 3
    with pytest.raises(ValueError, match="adj_df.*flow"):
        validate_flow_overall(adj, bench, comparisons=comparison)
    with pytest.raises(ValueError, match="adj_df.*flow"):
        validate_flow_pairs(adj, bench)


@pytest.mark.parametrize("validator", [validate_flow_overall, validate_flow_pairs])
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
def test_required_columns(inputs, validator, table, column):
    data = list(inputs)
    data[table] = data[table].drop(columns=column)
    with pytest.raises(ValueError, match=column):
        validator(*data)


@pytest.mark.parametrize(
    "bad", [[], None, "unknown", ["all", "raw_vs_adjusted"], [1], {"raw_vs_adjusted"}]
)
def test_invalid_comparisons(inputs, bad):
    with pytest.raises(ValueError, match="comparisons"):
        validate_flow_overall(*inputs, comparisons=bad)


@pytest.mark.parametrize(
    "option", ["drop_zeros", "na_rm", "by_source", "return_joined"]
)
def test_boolean_options_reject_ambiguous_values(inputs, option):
    with pytest.raises(TypeError, match=option):
        validate_flow_overall(*inputs, **{option: "false"})


@pytest.mark.parametrize("validator", [validate_flow_overall, validate_flow_pairs])
def test_input_types_and_collisions(inputs, validator):
    adj, bench = inputs
    with pytest.raises(TypeError, match="DataFrame"):
        validator([], bench)
    with pytest.raises(TypeError, match="DataFrame"):
        validator(adj, None)
    with pytest.raises(ValueError, match="numeric"):
        validator(adj.assign(flow_adj="text"), bench)
    with pytest.raises(ValueError, match="real-valued"):
        validator(adj.assign(flow_adj=1 + 2j), bench)
    with pytest.raises(ValueError, match="unique column"):
        validator(pd.concat([adj, adj[["flow_adj"]]], axis=1), bench)
    with pytest.raises(ValueError, match="distinct"):
        validator(adj, bench, flow_col_adj="flow")
    with pytest.raises(TypeError, match="Column names"):
        validator(adj, bench, flow_col_adj=None)


def test_nullable_values_and_nondefault_indices(inputs):
    adj, bench = inputs
    adj = adj.copy()
    bench = bench.copy()
    adj["flow_adj"] = pd.array([90, None, 50], dtype="Int64")
    adj.index = [7, 7, 2]
    bench.index = [4, 1, 4]
    pairs = validate_flow_pairs(adj, bench)
    assert pairs.index.tolist() == [0, 1, 2]
    np.testing.assert_allclose(
        pairs["signed_residual_adjusted_vs_benchmark"], [5, np.nan, 5], equal_nan=True
    )
    result = validate_flow_overall(adj, bench)
    assert result["n"] == 2
    assert result["rmse"] == 5


def test_only_requested_public_api_is_exported():
    assert "validate_flow_overall" in debiaspy.__all__
    assert "validate_flow_pairs" in debiaspy.__all__
    assert not hasattr(debiaspy, "validate_flow_benchmark")
    assert not hasattr(debiaspy, "validate_flow_all")


def test_positive_infinity_is_retained_when_only_drop_zeros_is_enabled():
    adj = pd.DataFrame(
        {"origin": ["A", "B"], "destination": ["X", "X"], "flow_adj": [np.inf, -1]}
    )
    bench = adj.drop(columns="flow_adj").assign(flow=[2, 3])
    result = validate_flow_overall(adj, bench, na_rm=False, drop_zeros=True)
    assert result["n"] == 1
    assert result["sum_x"] == np.inf
    assert result["mean_error"] == -np.inf
    assert result["rmse"] == np.inf


def test_source_groups_exclude_missing_ids_and_follow_factor_order(inputs):
    adj, bench = inputs
    adj = adj.assign(
        mpd_source=pd.Categorical(["a", None, "b"], categories=["b", "a", "unused"])
    )
    result = validate_flow_overall(
        adj,
        bench,
        by_source=True,
        comparisons=["raw_vs_benchmark", "adjusted_vs_benchmark"],
    )
    assert result["summary"]["n"].tolist() == [3, 3]
    assert result["by_source"]["mpd_source"].tolist() == ["b", "b", "a", "a"]
    assert (
        result["by_source"]["comparison"].tolist()
        == ["adjusted_vs_benchmark", "raw_vs_benchmark"] * 2
    )
    assert result["by_source"]["n"].tolist() == [1] * 4


def test_input_data_and_returned_tables_are_independent(inputs):
    adj, bench = inputs
    original_adj = adj.copy(deep=True)
    original_bench = bench.copy(deep=True)
    result = validate_flow_overall(adj, bench)
    result["data"].loc[0, "flow_adj"] = -999
    pairs = validate_flow_pairs(adj, bench)
    pairs.loc[0, "mpd_flow"] = -999
    assert_frame_equal(adj, original_adj)
    assert_frame_equal(bench, original_bench)
