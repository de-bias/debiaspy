from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from debiaspy._adjustment_common import (
    join_with_r_warning,
    require_no_reserved_columns,
)
from debiaspy._choices import match_choice
from debiaspy._exceptions import DebiasSchemaError
from debiaspy._schemas import require_columns


def adjust_inverse_penetration(
    mpd_od_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    flow_col: str = "flow",
    weight_by: str = "origin",
) -> pd.DataFrame:
    """Scale MPD flows by inverse active-user penetration at flow endpoints.

    ``mpd_od_df`` requires ``origin``, ``destination``, and ``flow_col``;
    ``coverage_df`` requires ``origin``, ``population``, and ``user_count``.
    When both tables contain ``mpd_source``, coverage joins are source-specific.
    ``weight_by`` selects origin, destination, or geometric-mean weighting.

    Returns the MPD rows with canonical ``flow``, ``flow_adj``, and the applied
    endpoint weight columns. Arithmetic preserves R missing and infinite
    results, including zero user counts. Distinct coverage rows are joined
    as in R; conflicting mappings can expand the output rows.
    """
    valid_weight_by = ("origin", "destination", "both")
    weight_by = match_choice(weight_by, valid_weight_by, "weight_by")

    require_columns(mpd_od_df, ("origin", "destination", flow_col), "mpd_od_df")
    require_columns(coverage_df, ("origin", "population", "user_count"), "coverage_df")
    require_no_reserved_columns(
        mpd_od_df,
        ("weight_origin", "weight_destination", "weight_both"),
        "mpd_od_df",
    )
    if flow_col != "flow" and "flow" in mpd_od_df.columns:
        raise DebiasSchemaError(
            "`flow_col` cannot be renamed to `flow` because `mpd_od_df` "
            "already contains a `flow` column."
        )

    coverage_values = coverage_df.loc[:, ["origin", "population", "user_count"]].copy()
    if "mpd_source" in coverage_df.columns:
        coverage_values["mpd_source"] = coverage_df["mpd_source"]

    for column in ("population", "user_count"):
        if is_bool_dtype(coverage_values[column]):
            coverage_values[column] = coverage_values[column].astype(float)

    has_source = (
        "mpd_source" in mpd_od_df.columns and "mpd_source" in coverage_df.columns
    )
    out = mpd_od_df.copy()

    if weight_by in {"origin", "both"}:
        origin_keys = ["origin", "mpd_source"] if has_source else ["origin"]
        cov_o = coverage_values.loc[
            :, [*origin_keys, "population", "user_count"]
        ].drop_duplicates()
        cov_o["weight_origin"] = cov_o["population"] / cov_o["user_count"]
        out = join_with_r_warning(
            out,
            cov_o.loc[:, [*origin_keys, "weight_origin"]],
            on=origin_keys,
        )

    if weight_by in {"destination", "both"}:
        destination_keys = (
            ["destination", "mpd_source"] if has_source else ["destination"]
        )
        cov_d_keys = ["origin", "mpd_source"] if has_source else ["origin"]
        cov_d = (
            coverage_values.loc[:, [*cov_d_keys, "population", "user_count"]]
            .drop_duplicates()
            .rename(columns={"origin": "destination"})
        )
        cov_d["weight_destination"] = cov_d["population"] / cov_d["user_count"]
        out = join_with_r_warning(
            out,
            cov_d.loc[:, [*destination_keys, "weight_destination"]],
            on=destination_keys,
        )

    flow = out[flow_col]
    if weight_by == "origin":
        out["flow_adj"] = flow * out["weight_origin"]
    elif weight_by == "destination":
        out["flow_adj"] = flow * out["weight_destination"]
    else:
        product = out["weight_origin"] * out["weight_destination"]
        with np.errstate(invalid="ignore"):
            out["weight_both"] = np.sqrt(product)
        if bool((product < 0).any()):
            warnings.warn("NaNs produced", RuntimeWarning, stacklevel=2)
        out["flow_adj"] = flow * out["weight_both"]

    return out.rename(columns={flow_col: "flow"})
