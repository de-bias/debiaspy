"""Adjust origin-destination flows using inverse population coverage."""

import warnings

import numpy as np
import pandas as pd

from debiaspy.measure_bias import (
    _require_columns,
    _require_dataframe,
    _require_numeric_columns,
    _require_string_column_names,
)


def adjust_inverse_penetration(
    mpd_od_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    flow_col: str = "flow",
    weight_by: str | None = "origin",
) -> pd.DataFrame:
    """Scale flows by population divided by active users.

    Parameters
    ----------
    mpd_od_df:
        Flow table containing ``origin``, ``destination`` and ``flow_col``.
        Additional columns are retained. When both input tables contain
        ``mpd_source``, coverage is matched by area and source.
    coverage_df:
        Coverage table containing ``origin``, ``population`` and ``user_count``.
        The ``origin`` column also supplies area identifiers for destination
        weights. Identical rows in the selected coverage columns are deduplicated.
    flow_col:
        Numeric flow column, renamed to ``flow`` in the result.
    weight_by:
        ``"origin"`` (default), ``"destination"`` or ``"both"``. Origin and
        destination weights are ``population / user_count``. Both-sided weighting
        uses their geometric mean. Unambiguous prefixes are accepted; None uses
        the default origin weighting.

    Returns
    -------
    pandas.DataFrame
        Original flow columns plus ``weight_origin`` and/or
        ``weight_destination``, ``weight_both`` for both-sided weighting, and
        ``flow_adj``. Original row order is retained with a fresh positional
        index; ambiguous coverage can expand rows. Existing ``flow_adj`` and,
        for both-sided weighting, ``weight_both`` are overwritten.

    Raises
    ------
    TypeError
        If inputs are not DataFrames or column names are invalid.
    ValueError
        If required columns are absent, arithmetic columns are non-numeric,
        the weighting option is invalid, or generated names conflict with
        existing columns.

    Warns
    -----
    UserWarning
        If a coverage join has a many-to-many relationship.
    RuntimeWarning
        If the product of origin and destination weights is negative, producing
        a missing geometric-mean weight.

    Notes
    -----
    This method does not clip, fill or reject zero, negative, missing or infinite
    numeric inputs. Division by zero follows floating-point arithmetic: a
    positive population with zero users gives an infinite weight, while 0/0
    gives NaN. Multiplying zero flow by an infinite weight also gives NaN.
    Missing coverage leaves missing weights and adjusted flows. Missing area or
    source keys match missing keys in the coverage table. Different coverage rows
    for the same key are retained, even when they give the same weight.
    Neither input is modified.
    """
    choices = ("origin", "destination", "both")
    if weight_by is None:
        weight_by = "origin"
    matches = [
        choice
        for choice in choices
        if isinstance(weight_by, str) and weight_by and choice.startswith(weight_by)
    ]
    if len(matches) != 1:
        raise ValueError("`weight_by` must be one of 'origin', 'destination', 'both'.")
    weight_by = matches[0]
    _require_string_column_names(flow_col)
    for data, name in ((mpd_od_df, "mpd_od_df"), (coverage_df, "coverage_df")):
        _require_dataframe(data, name)
        if not data.columns.is_unique:
            raise ValueError(f"`{name}` must have unique column names.")
    _require_columns(mpd_od_df, ["origin", "destination", flow_col], "mpd_od_df")
    _require_columns(coverage_df, ["origin", "population", "user_count"], "coverage_df")
    _require_numeric_columns(mpd_od_df, [flow_col], "mpd_od_df")
    _require_numeric_columns(coverage_df, ["population", "user_count"], "coverage_df")
    for data, columns, name in (
        (mpd_od_df, [flow_col], "mpd_od_df"),
        (coverage_df, ["population", "user_count"], "coverage_df"),
    ):
        if any(pd.api.types.is_complex_dtype(data[column]) for column in columns):
            raise ValueError(f"`{name}` flow and coverage values must be real-valued.")
    if flow_col != "flow" and "flow" in mpd_od_df:
        raise ValueError(
            "Renaming `flow_col` to 'flow' would create duplicate columns."
        )

    source_keys = (
        ["mpd_source"]
        if "mpd_source" in mpd_od_df and "mpd_source" in coverage_df
        else []
    )
    coverage = coverage_df.loc[
        :, ["origin", *source_keys, "population", "user_count"]
    ].drop_duplicates()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        weights = coverage["population"].to_numpy(
            dtype=float, na_value=np.nan
        ) / coverage["user_count"].to_numpy(dtype=float, na_value=np.nan)
    out = mpd_od_df.copy(deep=True)
    sides = ("origin", "destination") if weight_by == "both" else (weight_by,)
    for side in sides:
        weight_name = f"weight_{side}"
        if weight_name in out:
            raise ValueError(
                f"`mpd_od_df` contains conflicting column `{weight_name}`."
            )
        side_weights = coverage.loc[:, ["origin", *source_keys]].rename(
            columns={"origin": side}
        )
        side_weights[weight_name] = weights
        out = _join_coverage_weights(out, side_weights, [side, *source_keys])

    flow = out[flow_col].to_numpy(dtype=float, na_value=np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if weight_by == "both":
            product = (
                out["weight_origin"].to_numpy() * out["weight_destination"].to_numpy()
            )
            if np.any(product < 0):
                warnings.warn(
                    "NaNs produced in geometric-mean weights.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            out["weight_both"] = np.sqrt(product)
        out["flow_adj"] = flow * out[f"weight_{weight_by}"].to_numpy()
    return out.rename(columns={flow_col: "flow"}).reset_index(drop=True)


def _join_coverage_weights(
    flows: pd.DataFrame, coverage: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    left_repeated = flows.duplicated(keys, keep=False)
    right_repeated = coverage.duplicated(keys, keep=False)
    if left_repeated.any() and right_repeated.any():
        # The two multiply matched rows can belong to different keys.
        left_keys = (
            flows.loc[:, keys].assign(_left_many=left_repeated).drop_duplicates()
        )
        right_keys = (
            coverage.loc[:, keys].assign(_right_many=right_repeated).drop_duplicates()
        )
        matched = left_keys.merge(right_keys, on=keys, how="inner", sort=False)
        if matched["_left_many"].any() and matched["_right_many"].any():
            warnings.warn(
                "Detected an unexpected many-to-many coverage join on "
                + ", ".join(keys)
                + ".",
                UserWarning,
                stacklevel=3,
            )
    return flows.merge(coverage, on=keys, how="left", sort=False)
