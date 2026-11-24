from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import Any

import pandas as pd


class AdjustmentSet(dict):
    """Named adjustment results with the executed methods in attrs."""

    def __init__(self):
        super().__init__()
        self.attrs = {}


def _require_columns(frame, columns, name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"`{name}` must be a data frame.")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"`{name}` is missing: {', '.join(missing)}.")


def _run(method, **args):
    name = f"adjust_{method}"
    function = getattr(import_module(f"debiaspy.{name}"), name, None)
    if function is None:
        raise NotImplementedError(f"Adjustment method {method!r} is unavailable.")
    return function(**args)


VALID_METHODS = (
    "inverse_penetration",
    "selection_rate",
    "selection_rate2",
    "raking_ratio",
    "coefficient",
    "multilevel_bayes",
)


def adjust_all_methods(
    mpd_od_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame | None = None,
    covariates_df: pd.DataFrame | None = None,
    distance_df: pd.DataFrame | None = None,
    methods: str | Iterable[str] = "all",
    covariate_col: str = "rural_pct",
    multilevel_engine: str = "frequentist",
    inverse_penetration_args: dict[str, Any] | None = None,
    selection_rate_args: dict[str, Any] | None = None,
    selection_rate2_args: dict[str, Any] | None = None,
    raking_ratio_args: dict[str, Any] | None = None,
    coefficient_args: dict[str, Any] | None = None,
    multilevel_args: dict[str, Any] | None = None,
) -> AdjustmentSet:
    """Run a selected set of flow-adjustment methods on shared inputs.

    ``mpd_od_df`` requires ``origin``, ``destination``, and ``flow``;
    ``coverage_df`` requires ``origin``, ``population``, and ``user_count``.
    Benchmark, covariate, and distance tables are required only by methods that
    use them and must contain their canonical OD, ``area``, or distance keys.
    Method-specific dictionaries override the wrapper defaults.

    Returns an :class:`AdjustmentSet` mapping method names to adjusted data
    frames, with the executed names in ``result.attrs["methods"]``. Invalid
    methods, missing inputs, or unsupported multilevel contracts fail before
    the affected method runs. Parity is scoped to each method's audited cases.
    """
    engines = ("frequentist", "bayesian")
    matches = [
        engine
        for engine in engines
        if isinstance(multilevel_engine, str) and engine.startswith(multilevel_engine)
    ]
    if len(matches) != 1 or not multilevel_engine:
        raise ValueError("`multilevel_engine` must be one of: frequentist, bayesian")
    multilevel_engine = matches[0]
    selected = _normalise_methods(methods)
    _require_columns(mpd_od_df, ("origin", "destination", "flow"), "mpd_od_df")
    _require_columns(coverage_df, ("origin", "population", "user_count"), "coverage_df")

    if any(
        method in selected
        for method in [
            "selection_rate",
            "selection_rate2",
            "raking_ratio",
            "coefficient",
        ]
    ):
        if benchmark_od_df is None:
            raise ValueError("`benchmark_od_df` must be a data frame.")
        _require_columns(
            benchmark_od_df, ("origin", "destination", "flow"), "benchmark_od_df"
        )

    if any(method in selected for method in ["selection_rate", "multilevel_bayes"]):
        if covariates_df is None:
            raise ValueError("`covariates_df` must be a data frame.")
        _require_columns(covariates_df, ("area", covariate_col), "covariates_df")

    if "multilevel_bayes" in selected:
        if distance_df is None:
            raise ValueError(
                '`distance_df` is required when `methods` includes "multilevel_bayes".'
            )
        _require_columns(
            distance_df, ("origin", "destination", "distance_km"), "distance_df"
        )

    out: AdjustmentSet = AdjustmentSet()

    if "inverse_penetration" in selected:
        args = _merge_args(
            {
                "mpd_od_df": mpd_od_df,
                "coverage_df": coverage_df,
                "weight_by": "both",
            },
            inverse_penetration_args,
        )
        out["inverse_penetration"] = _run("inverse_penetration", **args)

    if "selection_rate" in selected:
        args = _merge_args(
            {
                "mpd_od_df": mpd_od_df,
                "coverage_df": coverage_df,
                "covariates_df": covariates_df,
                "covariate_col": covariate_col,
                "weight_by": "origin",
                "benchmark_od_df": benchmark_od_df,
                "calibration_aggregate": "origin",
            },
            selection_rate_args,
        )
        out["selection_rate"] = _run("selection_rate", **args)

    if "selection_rate2" in selected:
        args = _merge_args(
            {
                "mpd_od_df": mpd_od_df,
                "coverage_df": coverage_df,
                "weight_by": "origin",
                "benchmark_od_df": benchmark_od_df,
                "calibration_aggregate": "origin",
            },
            selection_rate2_args,
        )
        out["selection_rate2"] = _run("selection_rate2", **args)

    if "raking_ratio" in selected:
        args = _merge_args(
            {"mpd_od_df": mpd_od_df, "benchmark_od_df": benchmark_od_df},
            raking_ratio_args,
        )
        out["raking_ratio"] = _run("raking_ratio", **args)

    if "coefficient" in selected:
        args = _merge_args(
            {
                "mpd_od_df": mpd_od_df,
                "benchmark_od_df": benchmark_od_df,
                "model_family": "ols",
                "level": "od",
            },
            coefficient_args,
        )
        out["coefficient"] = _run("coefficient", **args)

    if "multilevel_bayes" in selected:
        default_formula = (
            "flow ~ "
            f"{covariate_col}_o + {covariate_col}_d + log_distance + "
            "bias_e_origin"
        )
        args = _merge_args(
            {
                "covariates_df": covariates_df,
                "distance_df": distance_df,
                "formula": default_formula,
                "model_engine": multilevel_engine,
                "scenario": "s1",
                "source_col": "mpd_source",
                "time_col": "mpd_time",
                "repeated_observation": "none",
                "prediction_scope": "complete_grid",
                "random_intercept": "none",
                "model_family": "poisson",
                "flow_adj_summary": "median",
            },
            multilevel_args,
        )
        ml_inputs = _prepare_multilevel_inputs(
            mpd_od_df,
            coverage_df,
            source_col=args.get("source_col"),
            time_col=args.get("time_col"),
        )
        args = {
            "mpd_od_df": ml_inputs["mpd_od_df"],
            "coverage_df": ml_inputs["coverage_df"],
            **args,
        }
        out["multilevel_bayes"] = _run("multilevel_bayes", **args)

    out.attrs["methods"] = list(out.keys())
    return out


def _normalise_methods(methods: str | Iterable[str]) -> list[str]:
    if isinstance(methods, str):
        selected = list(VALID_METHODS) if methods == "all" else [methods]
    else:
        selected = list(methods)
        if selected == ["all"]:
            selected = list(VALID_METHODS)

    bad = [method for method in selected if method not in VALID_METHODS]
    if bad:
        raise ValueError("`methods` contains unsupported value(s): " + ", ".join(bad))
    if not selected:
        raise ValueError("`methods` must select at least one adjustment method.")
    return list(dict.fromkeys(selected))


def _merge_args(
    defaults: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError(
            "Method-specific argument overrides must be supplied as named lists."
        )
    merged = defaults.copy()
    for key, value in overrides.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_args(merged[key], value)
        else:
            merged[key] = value
    return merged


def _prepare_multilevel_inputs(
    mpd_od_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    source_col: str | None,
    time_col: str | None,
) -> dict[str, pd.DataFrame]:
    mpd_out = mpd_od_df.copy()
    coverage_out = coverage_df.copy()

    if source_col is not None:
        if source_col not in mpd_out.columns:
            mpd_out[source_col] = "source_1"
        if source_col not in coverage_out.columns:
            coverage_out[source_col] = "source_1"

    if time_col is not None:
        if time_col not in mpd_out.columns:
            mpd_out[time_col] = "time_1"
        if time_col not in coverage_out.columns:
            coverage_out[time_col] = "time_1"

    return {"mpd_od_df": mpd_out, "coverage_df": coverage_out}
