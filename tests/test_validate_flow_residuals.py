"""Residual and spatial diagnostics compared with executed R references."""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from pandas.testing import assert_frame_equal

import debiaspy
from debiaspy import validate_flow_residual_structure, validate_flow_residuals
from debiaspy._r_permutation import _RPermutation
from debiaspy.validate_flows import _adjust_local_pvalues

FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "flow_residuals_r.json").read_text())
ADJUSTMENTS = json.loads((FIXTURES / "flow_residual_p_adjust_r.json").read_text())


def _decode(v):
    if isinstance(v, dict):
        return {k: _decode(i) for k, i in v.items()}
    if isinstance(v, list):
        return [_decode(i) for i in v]
    if v == "Infinity":
        return np.inf
    if v == "-Infinity":
        return -np.inf
    return v


def _frame(data):
    result = pd.DataFrame(data)
    for col in result:
        if col in (
            "origin",
            "destination",
            "area",
            "neighbor",
            "from",
            "to",
            "id",
            "mpd_source",
        ):
            result[col] = result[col].astype(object)
        else:
            result[col] = pd.to_numeric(result[col]).astype(float)
    return result


def _case(name):
    case = _decode(CASES[name])
    args = [_frame(case[k]) for k in ("adj", "benchmark")]
    kwargs = {
        k: _frame(v) if isinstance(v, dict) else v for k, v in case["options"].items()
    }
    return args, kwargs


@pytest.mark.parametrize("name", CASES)
def test_matches_executed_r_reference(name):
    case = _decode(CASES[name])
    args, kwargs = _case(name)
    func = (
        validate_flow_residuals
        if case["target"] == "residuals"
        else validate_flow_residual_structure
    )
    originals = [value.copy(deep=True) for value in args]
    optional = {
        k: v.copy(deep=True) for k, v in kwargs.items() if isinstance(v, pd.DataFrame)
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    assert list(result) == list(case["expected"])
    for key, table in case["expected"].items():
        expected = pd.DataFrame(
            [[np.nan if v is None else v for v in row] for row in table["data"]],
            columns=table["columns"],
        )
        assert_frame_equal(
            result[key].astype(object).where(result[key].notna(), np.nan),
            expected.astype(object).where(expected.notna(), np.nan),
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    assert (
        any("many-to-many" in str(w.message) for w in caught)
        == case["warnings"]["many"]
    )
    assert (
        any("multiple replacements" in str(w.message) for w in caught)
        == case["warnings"]["replacement"]
    )
    assert all(
        "many-to-many" in str(w.message) or "multiple replacements" in str(w.message)
        for w in caught
    )
    for actual, original in zip(args, originals, strict=True):
        assert_frame_equal(actual, original)
    for key, original in optional.items():
        assert_frame_equal(kwargs[key], original)


@pytest.mark.parametrize("method", ADJUSTMENTS["expected"])
def test_p_value_adjustments_match_r(method):
    np.testing.assert_allclose(
        _adjust_local_pvalues(np.asarray(ADJUSTMENTS["p"]), method),
        ADJUSTMENTS["expected"][method],
        rtol=1e-14,
    )


@pytest.mark.parametrize(
    "seed,expected",
    [
        (0, [[-1, 2, 1, 3, 0], [3, 1, 2, 0, -1], [1, 3, -1, 2, 0]]),
        (123, [[1, 0, 3, 2, -1], [1, -1, 0, 3, 2], [0, 1, -1, 2, 3]]),
        (-4, [[3, -1, 1, 2, 0], [-1, 0, 3, 1, 2], [0, 2, 1, -1, 3]]),
        (2026, [[3, -1, 2, 0, 1], [1, 2, -1, 3, 0], [0, 3, 1, -1, 2]]),
    ],
)
def test_seeded_permutations_match_r(seed, expected):
    generator = _RPermutation(seed)
    actual = [generator.permute(np.array([-1, 0, 1, 2, 3])).tolist() for _ in range(3)]
    assert actual == expected


@pytest.fixture
def inputs():
    return _case("residual_basic")[0]


def test_residual_signs_percentages_and_improvement(inputs):
    result = validate_flow_residuals(*inputs, top_n=2)
    data = result["data"]
    assert data["residual_mpd_benchmark"].tolist() == [-20, 30, 10]
    assert data["residual_adj_benchmark"].tolist() == [-10, 40, -10]
    assert data["signed_residual_reduction"].tolist() == [-10, -10, 20]
    assert data["abs_residual_reduction"].tolist() == [10, -10, 0]
    assert data["improvement_flag"].tolist() == ["improved", "worsened", "unchanged"]
    assert data["moved_in_benchmark_direction"].tolist() == [True, False, True]
    np.testing.assert_allclose(data["pct_residual_mpd_benchmark"], [-25, 37.5, 50])
    assert data["residual_mpd_over_1sd"].tolist() == [False, True, False]
    assert result["summary"].loc[0, "share_improved"] == pytest.approx(1 / 3)
    assert result["top_worst"]["abs_residual_adj_benchmark"].tolist() == [40, 10]


def test_missing_flags_follow_r_conventions():
    args, _ = _case("residual_all_missing")
    result = validate_flow_residuals(*args)
    assert result["data"]["improvement_flag"].tolist() == ["unchanged", "unchanged"]
    assert result["data"]["moved_in_benchmark_direction"].tolist() == [False, False]
    assert result["data"]["residual_adj_over_1sd"].isna().all()
    assert np.isnan(result["summary"].loc[0, "share_residual_adj_over_1sd"])


@pytest.mark.parametrize(
    "comparison,residual_type",
    [
        ("adjusted_vs_benchmark", "adjusted"),
        ("raw_vs_benchmark", "mpd"),
        ("raw_vs_adjusted", "adjustment"),
    ],
)
@pytest.mark.parametrize("role", ["origin", "destination"])
@pytest.mark.parametrize("aggregation", ["mean", "sum"])
def test_selection_aggregation_and_benchmark_relationship(
    inputs, comparison, residual_type, role, aggregation
):
    result = validate_flow_residual_structure(
        *inputs,
        comparison=comparison,
        spatial_role=role,
        residual_aggregation=aggregation,
    )
    od, areas = result["od_level"], result["area_level"]
    np.testing.assert_array_equal(
        od["selected_residual"], od[f"signed_residual_{comparison}"]
    )
    assert result["summary"].loc[0, "residual_type"] == residual_type
    assert result["flow_correlation"].loc[0, "benchmark_comparison"] == "benchmark_flow"
    for _, row in areas.iterrows():
        selected = od.loc[od[role] == row["area"], "selected_residual"]
        expected = selected.mean() if aggregation == "mean" else selected.sum()
        assert row["selected_residual"] == pytest.approx(expected)
        # These aliases intentionally follow R's sequential summarise semantics.
        assert row["mean_residual"] == pytest.approx(expected)
        assert row["sum_residual"] == pytest.approx(expected)
        assert row["mean_abs_residual"] == pytest.approx(abs(expected))


def test_global_local_moran_and_weighted_lags():
    args, kwargs = _case("lisa_weighted")
    result = validate_flow_residual_structure(*args, **kwargs)
    row = result["local_moran"].set_index("area").loc["A"]
    assert row["n_neighbors_used"] == 2
    assert row["local_weight_sum"] == 4
    assert row["neighbor_lag_mean"] == 1.5
    assert row["spatial_lag_z"] == pytest.approx(0.9185586535436918)
    assert row["local_moran_i"] == -4.5
    assert result["moran_i"].loc[0, "weight_sum"] == 10
    assert result["moran_i"].loc[0, "moran_i"] == pytest.approx(-1.05)


def test_local_moran_seed_is_reproducible_and_does_not_change_numpy_global_state():
    args, kwargs = _case("lisa_irregular")
    before = np.random.get_state()
    first = validate_flow_residual_structure(*args, **kwargs)
    second = validate_flow_residual_structure(*args, **kwargs)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert_frame_equal(first["local_moran"], second["local_moran"])
    assert first["moran_i"].loc[0, "n_links_used"] == 8


def test_local_moran_is_opt_in_and_unused_options_are_ignored(inputs):
    result = validate_flow_residual_structure(
        *inputs, local_moran_nsim=0, local_moran_p_adjust="bad"
    )
    assert "local_moran" not in result and "lisa_cluster" not in result["map_data"]
    assert np.isnan(result["moran_i"].loc[0, "moran_i"])


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, "two", [], 2**31])
def test_invalid_top_n(inputs, value):
    with pytest.raises(ValueError, match="top_n"):
        validate_flow_residuals(*inputs, top_n=value)


@pytest.mark.parametrize(
    "option,value",
    [
        ("local_moran_nsim", 0),
        ("local_moran_nsim", 1.5),
        ("local_moran_nsim", np.inf),
        ("local_moran_nsim", True),
        ("local_moran_alpha", -0.1),
        ("local_moran_alpha", 1.1),
        ("local_moran_alpha", np.nan),
        ("local_moran_p_adjust", "bad"),
        ("local_moran_p_adjust", []),
        ("local_moran_seed", np.inf),
        ("local_moran_seed", True),
        ("local_moran_seed", 2**31),
    ],
)
def test_invalid_local_moran_options(inputs, option, value):
    with pytest.raises(ValueError, match=option):
        validate_flow_residual_structure(*inputs, local_moran=True, **{option: value})


@pytest.mark.parametrize(
    "option,value",
    [
        ("local_moran", None),
        ("make_plots", "yes"),
        ("spatial_role", "both"),
        ("residual_aggregation", "median"),
        ("residual_type", "invalid"),
        ("comparison", "all"),
        ("comparison", []),
    ],
)
def test_invalid_structure_selection(inputs, option, value):
    with pytest.raises((ValueError, TypeError), match=option):
        validate_flow_residual_structure(*inputs, **{option: value})


@pytest.mark.parametrize(
    "function", [validate_flow_residuals, validate_flow_residual_structure]
)
@pytest.mark.parametrize(
    "table,column",
    [
        (0, "origin"),
        (0, "destination"),
        (0, "flow"),
        (0, "flow_adj"),
        (1, "origin"),
        (1, "destination"),
        (1, "flow"),
    ],
)
def test_required_columns(inputs, function, table, column):
    inputs[table] = inputs[table].drop(columns=column)
    with pytest.raises(ValueError, match=column):
        function(*inputs)


def test_invalid_optional_tables_and_columns(inputs):
    with pytest.raises(ValueError, match="covariate_df.*required"):
        validate_flow_residual_structure(*inputs, covariate_col="income")
    with pytest.raises(TypeError, match="DataFrame"):
        validate_flow_residual_structure(*inputs, area_neighbors=[])
    with pytest.raises(ValueError, match="area_neighbors.*contain"):
        validate_flow_residual_structure(
            *inputs, area_neighbors=pd.DataFrame({"area": ["A"]})
        )
    links = pd.DataFrame({"area": ["A"], "neighbor": ["B"], "w": ["bad"]})
    with pytest.raises(ValueError, match="weight_col"):
        validate_flow_residual_structure(
            *inputs, area_neighbors=links, weight_col="absent"
        )
    with pytest.raises(ValueError, match="numeric"):
        validate_flow_residual_structure(*inputs, area_neighbors=links, weight_col="w")
    with pytest.raises(ValueError, match="real-valued"):
        validate_flow_residual_structure(
            *inputs, area_neighbors=links.assign(w=1j), weight_col="w"
        )
    with pytest.raises(ValueError, match="distinct"):
        validate_flow_residual_structure(
            *inputs, area_neighbors=links, area_col="area", neighbor_col="area"
        )
    with pytest.raises(ValueError, match="covariate_df.*contain"):
        validate_flow_residual_structure(
            *inputs, covariate_df=pd.DataFrame({"area": ["A"]}), covariate_col="income"
        )
    with pytest.raises(ValueError, match="geometry_df.*contain"):
        validate_flow_residual_structure(
            *inputs, geometry_df=pd.DataFrame({"bad": [1]})
        )
    with pytest.raises(ValueError, match="unique"):
        validate_flow_residual_structure(
            *inputs, area_neighbors=pd.DataFrame([[1, 2]], columns=["area", "area"])
        )


def test_nullable_flows_and_nondefault_indices(inputs):
    adjusted, benchmark = inputs
    adjusted = adjusted.copy()
    adjusted["flow_adj"] = pd.array([90, None, 30], dtype="Int64")
    adjusted.index = [9, 9, 3]
    benchmark.index = [2, 2, 1]
    result = validate_flow_residuals(adjusted, benchmark)
    assert result["data"].index.tolist() == [0, 1, 2]
    np.testing.assert_allclose(
        result["data"]["residual_adj_benchmark"], [-10, np.nan, -10], equal_nan=True
    )


def test_plots_have_expected_content_and_render(tmp_path):
    args, kwargs = _case("structure_basic")
    result = validate_flow_residual_structure(
        *args, **kwargs, make_plots=True, x_col="x", y_col="y"
    )
    try:
        assert set(result["plots"]) == {
            "residual_reduction_distribution",
            "residual_vs_benchmark",
            "residual_vs_covariate",
            "residual_map",
        }
        for name, figure in result["plots"].items():
            assert isinstance(figure, Figure)
            path = tmp_path / f"{name}.png"
            figure.savefig(path)
            assert path.stat().st_size > 1000
        scatter = (
            result["plots"]["residual_vs_benchmark"]
            .axes[0]
            .collections[0]
            .get_offsets()
        )
        np.testing.assert_allclose(scatter[:, 0], result["od_level"]["benchmark_flow"])
        np.testing.assert_allclose(
            scatter[:, 1], result["od_level"]["selected_residual"]
        )
    finally:
        for figure in result["plots"].values():
            plt.close(figure)


def test_invalid_plot_coordinates_do_not_leave_figures_open(inputs):
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="Both.*required"):
        validate_flow_residual_structure(*inputs, make_plots=True, x_col="x")
    with pytest.raises(ValueError, match="must name columns"):
        validate_flow_residual_structure(*inputs, make_plots=True, x_col="x", y_col="y")
    assert plt.get_fignums() == before
    validate_flow_residual_structure(*inputs, x_col="unused")


def test_public_exports():
    assert {"validate_flow_residuals", "validate_flow_residual_structure"} <= set(
        debiaspy.__all__
    )


def test_sd_threshold_flags_identify_an_outlier_above_three_sd():
    adjusted = pd.DataFrame(
        {
            "origin": [f"A{i}" for i in range(10)],
            "destination": ["X"] * 10,
            "flow": [10] * 9 + [0],
            "flow_adj": [10] * 9 + [50],
        }
    )
    benchmark = adjusted[["origin", "destination"]].assign(flow=[10] * 9 + [100])
    result = validate_flow_residuals(adjusted, benchmark)
    for series in ("mpd", "adj"):
        assert result["data"].loc[
            9, f"abs_residual_{series}_benchmark_sd_score"
        ] == pytest.approx(np.sqrt(10))
        for threshold in (1, 2, 3):
            column = f"residual_{series}_over_{threshold}sd"
            assert result["data"][column].tolist() == [False] * 9 + [True]
            assert result["summary"].loc[0, f"share_{column}"] == pytest.approx(0.1)
