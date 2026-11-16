"""R parity and public-interface checks for adjust_selection_rate2."""

import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from debiaspy import adjust_selection_rate2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
METHOD = "selection_rate2"
CASE_INPUTS = json.loads((FIXTURES / f"{METHOD}_inputs.json").read_text())


@pytest.fixture(scope="module")
def r_reference(tmp_path_factory):
    """Run the reference on request, or load its stored output."""
    r_root = os.environ.get("DEBIASR_ROOT")
    output = FIXTURES / f"{METHOD}_r.json"
    if r_root:
        rscript = shutil.which("Rscript")
        if rscript is None:
            pytest.fail("DEBIASR_ROOT is set but Rscript is not on PATH.")
        output = tmp_path_factory.mktemp(METHOD) / output.name
        subprocess.run(
            [
                rscript,
                "--vanilla",
                str(PROJECT_ROOT / "scripts" / "generate_selection_reference.R"),
                str(Path(r_root).expanduser().resolve()),
                METHOD,
                str(output),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            timeout=60,
        )
    return json.loads(output.read_text())


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


def _frame(value):
    return pd.DataFrame(
        {
            key: item if isinstance(item, list) else [item]
            for key, item in _decode(value).items()
        }
    )


def _call_args(case):
    args = _decode(dict(case.get("args") or {}))
    for key, target in [
        ("mpd", "mpd_od_df"),
        ("coverage", "coverage_df"),
        ("covariates", "covariates_df"),
        ("benchmark", "benchmark_od_df"),
    ]:
        if key in case:
            args[target] = _frame(case[key])
    return args


@pytest.mark.parametrize(
    "name,mode",
    [
        (name, mode)
        for name, case in CASE_INPUTS.items()
        for mode in case.get("modes", ["origin", "destination", "both"])
    ],
)
def test_matches_r_reference(name, mode, r_reference):
    args = _call_args(CASE_INPUTS[name])
    original = {
        key: value.copy(deep=True)
        for key, value in args.items()
        if isinstance(value, pd.DataFrame)
    }
    reference = r_reference["cases"][name]["expected"][mode]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = adjust_selection_rate2(**args, weight_by=mode)
    expected = pd.DataFrame(reference["data"], columns=reference["columns"])
    for col in ["flow", "weight_origin", "weight_destination", "flow_adj"]:
        expected[col] = pd.to_numeric(expected[col])
    assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    parameter = reference["parameter"]
    if parameter is None:
        assert np.isnan(actual.attrs["k"])
    else:
        assert actual.attrs["k"] == pytest.approx(parameter, rel=1e-12, abs=1e-12)
    if reference["calibration"]:
        assert_frame_equal(
            actual.attrs["k_calibration"],
            pd.DataFrame(reference["calibration"]),
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    else:
        assert "k_calibration" not in actual.attrs
    assert (
        sum("many-to-many" in str(w.message) for w in caught)
        == reference["many_to_many"]
    )
    assert (
        sum("NAs introduced by coercion" in str(w.message) for w in caught)
        == reference["coercion_warnings"]
    )
    assert len(caught) == reference["many_to_many"] + reference["coercion_warnings"]
    for key, value in original.items():
        assert_frame_equal(args[key], value)


def test_reference_provenance_is_pinned(r_reference):
    assert (
        r_reference["metadata"]["reference"]
        == "6fa2709ae42de6dc525a4a46182fdfb448b14a3f"
    )
    assert r_reference["metadata"]["function_name"] == f"adjust_{METHOD}"


@pytest.fixture
def inputs():
    args = _call_args(CASE_INPUTS["legacy_fixed"])
    return args["mpd_od_df"], args["coverage_df"]


@pytest.mark.parametrize("mode", ["origin", "destination", "both"])
def test_empty_mpd_returns_schema_and_empty_coverage_leaves_missing_weights(
    inputs, mode
):
    m, c = inputs
    empty = adjust_selection_rate2(m.iloc[:0], c, weight_by=mode, k=1.0)
    expected = adjust_selection_rate2(m, c, weight_by=mode, k=1.0)
    assert empty.empty
    assert list(empty) == list(expected)
    missing = adjust_selection_rate2(m, c.iloc[:0], weight_by=mode, k=1.0)
    assert missing.flow_adj.isna().all()
    assert missing.weight_missing.all()


@pytest.mark.parametrize("table", [0, 1])
def test_requires_dataframes(inputs, table):
    args = list(inputs)
    args[table] = []
    with pytest.raises(TypeError, match="DataFrame"):
        adjust_selection_rate2(*args)


@pytest.mark.parametrize("column", ["origin", "destination", "flow", "mpd_source"])
def test_requires_mpd_columns(inputs, column):
    m, c = inputs
    with pytest.raises(ValueError, match="mpd_od_df"):
        adjust_selection_rate2(m.drop(columns=column), c)


@pytest.mark.parametrize("column", ["origin", "population", "user_count", "mpd_source"])
def test_requires_coverage_schema(inputs, column):
    m, c = inputs
    with pytest.raises(ValueError, match="coverage_df"):
        adjust_selection_rate2(m, c.drop(columns=column))


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, True, "bad", [1]])
def test_explicit_parameter_validation(inputs, bad):
    with pytest.raises(ValueError, match="k.*finite positive"):
        adjust_selection_rate2(*inputs, k=bad)


@pytest.mark.parametrize(
    "lower,upper", [(-1, 1), (2, 1), (np.nan, 1), (0, np.nan), (True, 2)]
)
def test_invalid_clip_bounds(inputs, lower, upper):
    with pytest.raises(ValueError, match="clip_min"):
        adjust_selection_rate2(*inputs, clip_min=lower, clip_max=upper)


@pytest.mark.parametrize("option", ["", "invalid", "ORIGIN", 1, []])
def test_invalid_options(inputs, option):
    with pytest.raises(ValueError, match="weight_by"):
        adjust_selection_rate2(*inputs, weight_by=option)
    with pytest.raises(ValueError, match="calibration_aggregate"):
        adjust_selection_rate2(*inputs, calibration_aggregate=option)


@pytest.mark.parametrize(
    "prefix,full",
    [(None, "origin"), ("orig", "origin"), ("d", "destination"), ("b", "both")],
)
def test_default_and_unambiguous_prefixes(inputs, prefix, full):
    actual = adjust_selection_rate2(*inputs, weight_by=prefix)
    expected = adjust_selection_rate2(*inputs, weight_by=full)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("grid", [[], [[1, 2]], "not a grid"])
def test_invalid_calibration_grid(inputs, grid):
    m, c = inputs
    with pytest.raises(ValueError, match="k_grid"):
        adjust_selection_rate2(m, c, benchmark_od_df=m, k_grid=grid)


def test_calibration_requires_valid_benchmark_and_overlap(inputs):
    m, c = inputs
    with pytest.raises(ValueError, match="no valid benchmark"):
        adjust_selection_rate2(m, c, benchmark_od_df=m.assign(flow=np.nan))
    with pytest.raises(ValueError, match="no overlap"):
        adjust_selection_rate2(
            m, c, benchmark_od_df=m.assign(origin="unmatched"), k_grid=[1]
        )
    with pytest.raises(ValueError, match="benchmark_od_df"):
        adjust_selection_rate2(m, c, benchmark_od_df=m.drop(columns="flow"))


def test_invalid_arithmetic_and_collisions_are_informative(inputs):
    m, c = inputs
    with pytest.raises(ValueError, match="numeric"):
        adjust_selection_rate2(m.assign(flow="text"), c)
    with pytest.raises(ValueError, match="real-valued"):
        adjust_selection_rate2(m.assign(flow=1 + 2j), c)
    with pytest.raises(ValueError, match="conflicting"):
        adjust_selection_rate2(m.assign(weight_origin=1), c)
    with pytest.raises(ValueError, match="unique column"):
        adjust_selection_rate2(pd.concat([m, m[["flow"]]], axis=1), c)


def test_duplicate_indices_and_nullable_flow(inputs):
    m, c = inputs
    m.index = [7, 7]
    c.index = [3, 3]
    m["flow"] = pd.array([10, None], dtype="Int64")
    original = m.copy(deep=True)
    actual = adjust_selection_rate2(m, c, k=1)
    assert actual.loc[0, "flow_adj"] > 0
    assert pd.isna(actual.loc[1, "flow_adj"])
    assert not actual.loc[1, "weight_missing"]
    assert_frame_equal(m, original)


def test_numeric_coercion_warns_and_preserves_missing_coverage(inputs):
    m, c = inputs
    c["population"] = ["100", "invalid"]
    with pytest.warns(UserWarning, match="coercion"):
        actual = adjust_selection_rate2(m, c, k=1)
    assert actual.loc[1, "weight_missing"]


@pytest.mark.parametrize("group", ["missing", "origin", "P", "U", "flow_adj"])
def test_invalid_group_columns(inputs, group):
    with pytest.raises(ValueError, match="group_cols"):
        adjust_selection_rate2(*inputs, group_cols=group)


def test_all_invalid_candidates_are_rejected(inputs):
    m, c = inputs
    with pytest.raises(ValueError, match="no valid candidates"):
        adjust_selection_rate2(m, c, benchmark_od_df=m, k_grid=[0, -1, np.nan, np.inf])
