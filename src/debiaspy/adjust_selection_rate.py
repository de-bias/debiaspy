"""Covariate-dependent selection-rate adjustment."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from debiaspy._selection_common import (
    _adjust,
    _area_strings,
    _bounds,
    _calibrate,
    _check_frame,
    _check_inputs,
    _choice,
    _clamp,
    _coverage_side,
    _join,
    _names,
    _numeric,
    _scalar,
)
from debiaspy.measure_bias import _require_columns, _require_string_column_names


def adjust_selection_rate(
    mpd_od_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    covariates_df: pd.DataFrame | None = None,
    covariate_col: str | None = None,
    weight_by: str | None = "origin",
    r_global: float | None = None,
    benchmark_od_df: pd.DataFrame | None = None,
    flow_col_bench: str = "flow",
    r_grid: Sequence[float] | np.ndarray | None = None,
    calibration_aggregate: str | None = "origin",
    clip_min: float = 0,
    clip_max: float = np.inf,
    keep_cols: str | Sequence[str] = (),
) -> pd.DataFrame:
    """Adjust observed OD flows using penetration and an area covariate.

    Parameters
    ----------
    mpd_od_df : pandas.DataFrame
        Observed ``flow`` by ``origin``, ``destination``, and ``mpd_source``.
    coverage_df : pandas.DataFrame
        Population and active users by area and source. Accepts legacy area
        counts or separate counts for each endpoint.
    covariates_df : pandas.DataFrame, optional
        Covariate values keyed by ``area`` and optionally ``mpd_source``.
        Use ``covariate_col`` to select the column to normalize.
    weight_by : {"origin", "destination", "both"}, default "origin"
        Use the selected endpoint or the geometric mean of both weights.
    r_global : float, optional
        Fixed parameter. If omitted, fit from ``benchmark_od_df`` over
        ``r_grid`` or use the descriptive users-to-population ratio.

    Returns
    -------
    pandas.DataFrame
        OD rows with endpoint weights, ``weight_missing``, and ``flow_adj``.
        Attributes ``r_global`` and ``r_calibration`` record the selected
        parameter and, when fitted, the candidate loss table.
    """
    schema = _check_inputs(mpd_od_df, coverage_df)
    mode = _choice(weight_by, ("origin", "destination", "both"), "weight_by")
    aggregate = _choice(
        calibration_aggregate, ("origin", "od"), "calibration_aggregate"
    )
    lower, upper = _bounds(clip_min, clip_max)
    keep = _names(keep_cols, "keep_cols")
    covariates = _covariates(covariates_df, covariate_col)
    prepared = {}
    # Without a usable covariate, I = 1 and the weight depends only on coverage.
    for side in ("origin", "destination") if schema == "new" else ("origin",):
        table = _coverage_side(coverage_df, schema, side, [], zero_users=False)
        if covariates is not None:
            keys = [side, *(["mpd_source"] if "mpd_source" in covariates else [])]
            table = _join(table, covariates.rename(columns={"area": side}), keys)
            table["I"] = table["I"].where(np.isfinite(table["I"]), 1.0)
        else:
            table["I"] = 1.0
        table["p"] = table["U"] / table["P"]
        prepared[side] = table

    def compute(r: float) -> pd.DataFrame:
        """Apply the weights for one value of r."""
        weights = {}
        for side in ("origin", "destination") if mode == "both" else (mode,):
            key = side if schema == "new" else "origin"
            table = prepared[key]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                weight = 1 / (table["I"] * table["p"] + (1 - table["I"]) * r)
            lookup = table.loc[:, [key, "mpd_source"]].rename(columns={key: side})
            lookup[f"weight_{side}"] = _clamp(weight, lower, upper).to_numpy()
            if schema == "legacy":
                # Legacy coverage uses the first weight per area and source.
                lookup = lookup.drop_duplicates([side, "mpd_source"], keep="first")
            weights[side] = lookup
        return _adjust(mpd_od_df, weights, mode, [], keep)

    diagnostic = None
    if r_global is not None:
        parameter = _scalar(r_global, "r_global", positive=False)
    elif benchmark_od_df is not None:
        parameter, diagnostic = _calibrate(
            compute,
            benchmark_od_df,
            flow_col_bench,
            np.arange(301) * 0.01 if r_grid is None else r_grid,
            aggregate,
            "r",
        )
    else:
        selected = (mode,) if schema == "new" and mode != "both" else tuple(prepared)
        numerator = sum(prepared[side]["U"].sum() for side in selected)
        denominator = sum(prepared[side]["P"].sum() for side in selected)
        with np.errstate(divide="ignore", invalid="ignore"):
            parameter = float(np.divide(numerator, denominator))
    result = compute(parameter)
    result.attrs["r_global"] = parameter
    if diagnostic is not None:
        result.attrs["r_calibration"] = diagnostic
    return result


def _covariates(frame: pd.DataFrame | None, column: str | None) -> pd.DataFrame | None:
    """Normalize the selected covariate and keep its lookup keys."""
    if frame is None:
        return None
    _check_frame(frame, "covariates_df")
    _require_columns(frame, ["area"], "covariates_df")
    if column is None:
        names = (
            "income_norm",
            "gni_pc",
            "income",
            "inc",
            "gni",
            "income_2000",
            "income_2010",
        )
        column = next((name for name in names if name in frame), None)
    else:
        _require_string_column_names(column)
        if column not in frame:
            raise ValueError(
                f"`covariate_col` = '{column}' not found in `covariates_df`."
            )
    if column is None:
        return None
    values = _numeric(frame[column], "selected covariate", warn=False)
    values = values.where(np.isfinite(values))
    if values.notna().any():
        low, high = values.min(), values.max()
        if not (low >= 0 and high <= 1):
            values = (values - low) / (high - low) if high > low else values * np.nan
    out = pd.DataFrame({"area": _area_strings(frame["area"]), "I": values})
    if "mpd_source" in frame:
        out["mpd_source"] = frame["mpd_source"]
    return out
