"""Selection Rate II adjustment."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from debiaspy._selection_common import (
    _adjust,
    _bounds,
    _calibrate,
    _check_inputs,
    _choice,
    _clamp,
    _coverage_side,
    _names,
    _scalar,
)
from debiaspy.measure_bias import _require_columns


def adjust_selection_rate2(
    mpd_od_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    weight_by: str | None = "origin",
    group_cols: str | Sequence[str] | None = None,
    k: float | None = None,
    k_grid: Sequence[float] | np.ndarray | None = None,
    benchmark_od_df: pd.DataFrame | None = None,
    flow_col_bench: str = "flow",
    calibration_aggregate: str | None = "origin",
    clip_min: float = 0,
    clip_max: float = np.inf,
    keep_cols: str | Sequence[str] = (),
) -> pd.DataFrame:
    """Adjust observed OD flows with the Selection Rate II factor.

    Parameters
    ----------
    mpd_od_df : pandas.DataFrame
        Observed ``flow`` by ``origin``, ``destination``, and ``mpd_source``.
    coverage_df : pandas.DataFrame
        Population and active users by area and source. Accepts legacy area
        counts or separate counts for each endpoint.
    weight_by : {"origin", "destination", "both"}, default "origin"
        Use the selected endpoint or the geometric mean of both factors.
    group_cols : str or sequence of str, optional
        Stratum columns present in both inputs and used for both endpoints.
    k : float, optional
        Fixed positive curvature. If omitted, fit from ``benchmark_od_df``
        over ``k_grid`` or use 1 when no benchmark is given.

    Returns
    -------
    pandas.DataFrame
        OD rows with endpoint weights, ``weight_missing``, and ``flow_adj``.
        Attributes ``k`` and ``k_calibration`` record the selected parameter
        and, when fitted, the candidate loss table.
    """
    schema = _check_inputs(mpd_od_df, coverage_df)
    mode = _choice(weight_by, ("origin", "destination", "both"), "weight_by")
    aggregate = _choice(
        calibration_aggregate, ("origin", "od"), "calibration_aggregate"
    )
    lower, upper = _bounds(clip_min, clip_max)
    keep = _names(keep_cols, "keep_cols")
    groups = _names(group_cols, "group_cols")
    reserved = {
        "origin",
        "destination",
        "flow",
        "mpd_source",
        "P",
        "U",
        "flow_adj",
        "weight_missing",
    }
    if reserved.intersection(groups):
        raise ValueError("`group_cols` must contain non-key stratum columns.")
    _require_columns(mpd_od_df, groups, "mpd_od_df group_cols")
    _require_columns(coverage_df, groups, "coverage_df group_cols")
    sides = ("origin", "destination") if schema == "new" else ("origin",)
    prepared = {}
    for side in sides:
        table = _coverage_side(coverage_df, schema, side, groups, zero_users=True)
        # Sum counts before taking their ratio, including repeated records.
        table = (
            table.groupby(
                [side, "mpd_source", *groups],
                sort=False,
                dropna=False,
                observed=True,
            )[["P", "U"]]
            .sum()
            .reset_index()
        )
        prepared[side] = table

    def compute(curvature: float) -> pd.DataFrame:
        """Apply the correction for one value of k."""
        weights = {}
        for side in ("origin", "destination") if mode == "both" else (mode,):
            key = side if schema == "new" else "origin"
            table = prepared[key]
            p = (table["U"] / table["P"]).clip(0, 1)
            # Evaluate the formula away from the endpoints, then restore them.
            safe = p.clip(1e-8, 1 - 1e-8)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                factor = (
                    safe * (np.exp(-curvature) - 1) / (np.exp(-curvature * safe) - 1)
                )
            factor = factor.mask(p.eq(1), 1.0).mask(p.eq(0), 0.0)
            lookup = table.loc[:, [key, "mpd_source", *groups]].rename(
                columns={key: side}
            )
            lookup[f"weight_{side}"] = _clamp(factor, lower, upper).to_numpy()
            weights[side] = lookup
        return _adjust(mpd_od_df, weights, mode, groups, keep)

    diagnostic = None
    if k is not None:
        parameter = _scalar(k, "k", positive=True)
    elif benchmark_od_df is not None:
        parameter, diagnostic = _calibrate(
            compute,
            benchmark_od_df,
            flow_col_bench,
            0.1 + np.arange(50) * 0.1 if k_grid is None else k_grid,
            aggregate,
            "k",
        )
    else:
        parameter = 1.0
    result = compute(parameter)
    result.attrs["k"] = parameter
    if diagnostic is not None:
        result.attrs["k_calibration"] = diagnostic
    return result
