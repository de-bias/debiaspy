import math
from importlib import import_module

import pandas as pd
import pytest

from debiaspy import adjust_all_methods


def test_inverse_penetration_uses_r_wrapper_default_and_named_override():
    observed = pd.DataFrame({"origin": ["A"], "destination": ["B"], "flow": [2]})
    coverage = pd.DataFrame(
        {"origin": ["A", "B"], "population": [10, 30], "user_count": [2, 3]}
    )
    result = adjust_all_methods(observed, coverage, methods="inverse_penetration")
    assert result.attrs["methods"] == list(result) == ["inverse_penetration"]
    assert result["inverse_penetration"]["flow_adj"].iloc[0] == pytest.approx(
        2 * math.sqrt(50)
    )
    overridden = adjust_all_methods(
        observed,
        coverage,
        methods="inverse_penetration",
        inverse_penetration_args={"weight_by": "destination"},
    )
    assert overridden["inverse_penetration"]["flow_adj"].tolist() == [20]


def test_all_requires_benchmark_before_dispatch():
    observed = pd.DataFrame(columns=["origin", "destination", "flow"])
    coverage = pd.DataFrame(columns=["origin", "population", "user_count"])
    with pytest.raises(ValueError, match="benchmark_od_df"):
        adjust_all_methods(observed, coverage)


def test_unavailable_method_is_named_after_validation(monkeypatch):
    module = import_module("debiaspy.adjust_coefficient")
    monkeypatch.delattr(module, "adjust_coefficient", raising=False)
    observed = pd.DataFrame(columns=["origin", "destination", "flow"])
    coverage = pd.DataFrame(columns=["origin", "population", "user_count"])
    with pytest.raises(NotImplementedError, match="coefficient"):
        adjust_all_methods(observed, coverage, observed, methods="coefficient")


def test_nested_overrides_preserve_siblings_and_explicit_none():
    from debiaspy.adjust_all_methods import _merge_args

    defaults = {"control": {"fit": {"max_iter": 100, "tol": 0.01}, "trace": True}}
    overrides = {"control": {"fit": {"max_iter": 20}, "trace": None}}
    assert _merge_args(defaults, overrides) == {
        "control": {"fit": {"max_iter": 20, "tol": 0.01}, "trace": None}
    }
    assert defaults["control"]["fit"]["max_iter"] == 100


def test_all_dispatches_six_methods_in_reference_order(monkeypatch):
    wrapper = import_module("debiaspy.adjust_all_methods")
    observed = pd.DataFrame(columns=["origin", "destination", "flow"])
    coverage = pd.DataFrame(columns=["origin", "population", "user_count"])
    covariates = pd.DataFrame(columns=["area", "rural_pct"])
    distances = pd.DataFrame(columns=["origin", "destination", "distance_km"])
    calls = {}

    def record(method, **kwargs):
        calls[method] = kwargs
        return observed

    monkeypatch.setattr(wrapper, "_run", record)
    result = adjust_all_methods(observed, coverage, observed, covariates, distances)
    expected = [
        "inverse_penetration",
        "selection_rate",
        "selection_rate2",
        "raking_ratio",
        "coefficient",
        "multilevel_bayes",
    ]
    assert list(calls) == list(result) == result.attrs["methods"] == expected
    assert calls["inverse_penetration"]["weight_by"] == "both"
    assert calls["selection_rate"]["calibration_aggregate"] == "origin"
    assert calls["selection_rate2"]["weight_by"] == "origin"
    assert calls["raking_ratio"]["benchmark_od_df"] is observed
    assert calls["coefficient"]["model_family"] == "ols"
    assert calls["multilevel_bayes"]["model_engine"] == "frequentist"
    assert {"mpd_source", "mpd_time"} <= set(calls["multilevel_bayes"]["mpd_od_df"])
    assert all(table is observed for table in result.values())
