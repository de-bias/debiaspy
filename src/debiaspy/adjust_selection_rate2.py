from __future__ import annotations

from numbers import Real
from typing import Literal, cast

import numpy as np
import pandas as pd

from debiaspy._adjustment_common import (
    clamp_weight,
    existing_columns,
    join_with_r_warning,
    match_choice,
    require_group_columns,
    require_no_reserved_columns,
    validate_clip_bounds,
)
from debiaspy._exceptions import DebiasSchemaError
from debiaspy._schemas import require_columns


def adjust_selection_rate2(
    mpd_od_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    weight_by: Literal["origin", "destination", "both"] = "origin",
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    k: float | None = None,
    k_grid: list[float] | tuple[float, ...] | np.ndarray | None = None,
    benchmark_od_df: pd.DataFrame | None = None,
    flow_col_bench: str = "flow",
    calibration_aggregate: Literal["origin", "od"] = "origin",
    clip_min: float = 0,
    clip_max: float = np.inf,
    keep_cols: str | list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """Apply the penetration-based selection-rate correction parameterized by k.

    ``mpd_od_df`` requires ``origin``, ``destination``, ``flow``, and
    ``mpd_source``. ``coverage_df`` must use the endpoint or legacy area
    population/user-count schema and contain every requested grouping column.
    Supply positive ``k`` directly or a benchmark OD table to select it from
    ``k_grid`` at origin or OD aggregation.

    Returns MPD rows with endpoint weights and ``flow_adj``; ``attrs["k"]``
    records the selected value and ``attrs["k_calibration"]`` records optional
    calibration diagnostics. Coverage rows are filtered and grouped as in R;
    missing weights produce missing adjusted flows. Calibration uses OD-only
    benchmark joins and removes missing values from its loss calculation.
    """
    weight_by = match_choice(weight_by, ("origin", "destination", "both"), "weight_by")
    calibration_aggregate = match_choice(
        calibration_aggregate, ("origin", "od"), "calibration_aggregate"
    )
    require_columns(
        mpd_od_df,
        ("origin", "destination", "flow", "mpd_source"),
        "mpd_od_df",
    )
    require_no_reserved_columns(
        mpd_od_df,
        (
            "weight_origin",
            "weight_destination",
            "weight_missing",
        ),
        "mpd_od_df",
    )
    clip_min, clip_max = validate_clip_bounds(clip_min, clip_max)
    mpd_od_df = mpd_od_df.copy()

    group_cols = require_group_columns(
        group_cols,
        {"mpd_od_df": mpd_od_df, "coverage_df": coverage_df},
        reserved=("origin", "destination", "flow", "mpd_source"),
    )
    schema = _coverage_schema(coverage_df)

    if schema == "new":
        cov_o_pre = _new_origin_coverage(coverage_df, group_cols)
        cov_d_pre = _new_destination_coverage(coverage_df, group_cols)
    else:
        base = _legacy_coverage(coverage_df, group_cols)
        cov_o_pre = base.rename(columns={"area": "origin", "p": "p_o"})
        cov_d_pre = base.rename(columns={"area": "destination", "p": "p_d"})

    def compute_with_k(
        k_value: float,
        additional_keep_cols: tuple[str, ...] = (),
    ) -> pd.DataFrame | None:
        if not np.isfinite(k_value) or k_value <= 0:
            return None
        cov_o = None
        cov_d = None
        if weight_by in {"origin", "both"}:
            cov_o = cov_o_pre.copy()
            cov_o["weight_origin"] = clamp_weight(
                _cf(cov_o["p_o"], k_value), clip_min, clip_max
            )
            cov_o = cov_o.loc[:, ["origin", "mpd_source", *group_cols, "weight_origin"]]
        if weight_by in {"destination", "both"}:
            cov_d = cov_d_pre.copy()
            cov_d["weight_destination"] = clamp_weight(
                _cf(cov_d["p_d"], k_value), clip_min, clip_max
            )
            cov_d = cov_d.loc[
                :, ["destination", "mpd_source", *group_cols, "weight_destination"]
            ]
        return _apply_weights(
            mpd_od_df,
            cov_o,
            cov_d,
            weight_by,
            group_cols,
            [*existing_columns(mpd_od_df, keep_cols), *additional_keep_cols],
        )

    if k_grid is None:
        k_grid = np.arange(0.1, 5.0000000001, 0.1)

    calibration = None
    if k is not None:
        if (
            isinstance(k, bool | np.bool_)
            or not isinstance(k, Real)
            or not np.isfinite(k)
            or k <= 0
        ):
            raise DebiasSchemaError("`k` must be a positive scalar.")
        k_used = float(k)
    elif benchmark_od_df is not None:
        k_used, calibration = _calibrate_k(
            compute_with_k,
            benchmark_od_df,
            flow_col_bench,
            k_grid,
            calibration_aggregate,
        )
    else:
        k_used = 1.0

    result = compute_with_k(k_used)
    if result is None:
        raise DebiasSchemaError("`k` must be a positive scalar.")
    result.attrs["k"] = k_used
    if calibration is not None:
        result.attrs["k_calibration"] = calibration
    return result


def _coverage_schema(coverage_df: pd.DataFrame) -> str:
    has_new = all(
        col in coverage_df.columns
        for col in [
            "origin",
            "origin_population",
            "origin_user_count",
            "destination",
            "destination_population",
            "destination_user_count",
            "mpd_source",
        ]
    )
    has_legacy = all(
        col in coverage_df.columns
        for col in ["origin", "population", "user_count", "mpd_source"]
    )
    if has_new:
        return "new"
    if has_legacy:
        return "legacy"
    raise DebiasSchemaError(
        "`coverage_df` does not match a supported schema.\n"
        "New:    {origin, origin_population, origin_user_count, "
        "destination, destination_population, destination_user_count, mpd_source}\n"
        "Legacy: {origin, population, user_count, mpd_source}"
    )


def _new_origin_coverage(
    coverage_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "origin": coverage_df["origin"].astype(str),
            "mpd_source": coverage_df["mpd_source"],
            **{col: coverage_df[col] for col in group_cols},
            "P_o": pd.to_numeric(coverage_df["origin_population"], errors="coerce"),
            "U_o": pd.to_numeric(coverage_df["origin_user_count"], errors="coerce"),
        }
    )
    out = out.drop_duplicates()
    out = out.loc[
        np.isfinite(out["P_o"])
        & (out["P_o"] > 0)
        & np.isfinite(out["U_o"])
        & (out["U_o"] >= 0)
    ].copy()
    grouped = (
        out.groupby(["origin", "mpd_source", *group_cols], sort=True, dropna=False)[
            ["P_o", "U_o"]
        ]
        .sum()
        .reset_index()
    )
    grouped["p_o"] = (grouped["U_o"] / grouped["P_o"]).clip(lower=0, upper=1)
    return grouped


def _new_destination_coverage(
    coverage_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "destination": coverage_df["destination"].astype(str),
            "mpd_source": coverage_df["mpd_source"],
            **{col: coverage_df[col] for col in group_cols},
            "P_d": pd.to_numeric(
                coverage_df["destination_population"], errors="coerce"
            ),
            "U_d": pd.to_numeric(
                coverage_df["destination_user_count"], errors="coerce"
            ),
        }
    )
    out = out.drop_duplicates()
    out = out.loc[
        np.isfinite(out["P_d"])
        & (out["P_d"] > 0)
        & np.isfinite(out["U_d"])
        & (out["U_d"] >= 0)
    ].copy()
    grouped = (
        out.groupby(
            ["destination", "mpd_source", *group_cols], sort=True, dropna=False
        )[["P_d", "U_d"]]
        .sum()
        .reset_index()
    )
    grouped["p_d"] = (grouped["U_d"] / grouped["P_d"]).clip(lower=0, upper=1)
    return grouped


def _legacy_coverage(
    coverage_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "area": coverage_df["origin"].astype(str),
            "mpd_source": coverage_df["mpd_source"],
            **{col: coverage_df[col] for col in group_cols},
            "P": pd.to_numeric(coverage_df["population"], errors="coerce"),
            "U": pd.to_numeric(coverage_df["user_count"], errors="coerce"),
        }
    )
    out = out.drop_duplicates()
    out = out.loc[
        np.isfinite(out["P"]) & (out["P"] > 0) & np.isfinite(out["U"]) & (out["U"] >= 0)
    ].copy()
    grouped = (
        out.groupby(["area", "mpd_source", *group_cols], sort=True, dropna=False)[
            ["P", "U"]
        ]
        .sum()
        .reset_index()
    )
    grouped["p"] = (grouped["U"] / grouped["P"]).clip(lower=0, upper=1)
    return grouped


def _cf(p_values: pd.Series, k_value: float) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").astype(float).clip(lower=0, upper=1)
    eps = 1e-8
    p_safe = p.clip(lower=eps, upper=1 - eps)
    out = p_safe * (np.exp(-k_value) - 1) / (np.exp(-k_value * p_safe) - 1)
    out = out.mask(p == 1, 1.0)
    out = out.mask(p == 0, 0.0)
    return out


def _apply_weights(
    mpd_od_df: pd.DataFrame,
    cov_o: pd.DataFrame | None,
    cov_d: pd.DataFrame | None,
    weight_by: str,
    group_cols: list[str],
    keep_cols: str | list[str] | tuple[str, ...],
) -> pd.DataFrame:
    out = mpd_od_df.copy()
    if cov_o is not None:
        out = out.merge(
            cov_o,
            how="left",
            on=["origin", "mpd_source", *group_cols],
            sort=False,
            validate="many_to_one",
        )
    else:
        out["weight_origin"] = np.nan
    if cov_d is not None:
        out = out.merge(
            cov_d,
            how="left",
            on=["destination", "mpd_source", *group_cols],
            sort=False,
            validate="many_to_one",
        )
    else:
        out["weight_destination"] = np.nan

    if weight_by == "origin":
        final_weight = out["weight_origin"]
    elif weight_by == "destination":
        final_weight = out["weight_destination"]
    else:
        final_weight = np.sqrt(out["weight_origin"] * out["weight_destination"])
        final_weight = final_weight.mask(
            ~np.isfinite(out["weight_origin"])
            | ~np.isfinite(out["weight_destination"]),
            np.nan,
        )
    out["weight_missing"] = ~np.isfinite(final_weight)
    out["flow_adj"] = np.where(
        np.isfinite(final_weight), out["flow"] * final_weight, np.nan
    )

    keep = existing_columns(out, keep_cols)
    columns = existing_columns(
        out,
        [
            "origin",
            "destination",
            "mpd_source",
            *group_cols,
            *keep,
            "flow",
            "weight_origin",
            "weight_destination",
            "weight_missing",
            "flow_adj",
        ],
    )
    return out.loc[:, columns].reset_index(drop=True)


def _calibrate_k(
    compute_with_k: object,
    benchmark_od_df: pd.DataFrame,
    flow_col_bench: str,
    k_grid: list[float] | tuple[float, ...] | np.ndarray,
    calibration_aggregate: str,
) -> tuple[float, pd.DataFrame]:
    """Use R's OD-only join and NA-removing calibration loss."""
    require_columns(
        benchmark_od_df,
        ("origin", "destination", flow_col_bench),
        "benchmark_od_df",
    )
    join_keys = ["origin", "destination"]
    bench = benchmark_od_df.loc[:, [*join_keys, flow_col_bench]].copy()
    bench["origin"] = bench["origin"].astype(str)
    bench["destination"] = bench["destination"].astype(str)
    bench = bench.rename(columns={flow_col_bench: "flow_bench"})
    bench["flow_bench"] = pd.to_numeric(bench["flow_bench"], errors="coerce")
    bench = bench.loc[np.isfinite(bench["flow_bench"])].copy()
    if bench.empty:
        raise DebiasSchemaError(
            "`benchmark_od_df` has no valid benchmark flows for calibration."
        )
    if len(k_grid) == 0:
        raise DebiasSchemaError("`k_grid` is empty; provide candidate values.")

    candidates: list[float] = []
    losses: list[float] = []
    for candidate in k_grid:
        candidate_value = float(candidate)
        adjusted = compute_with_k(candidate_value)
        if adjusted is None:
            continue
        joined = join_with_r_warning(
            adjusted.loc[:, [*join_keys, "flow_adj"]],
            bench,
            how="inner",
            on=join_keys,
        )
        if joined.empty:
            continue
        if calibration_aggregate == "origin":
            totals = cast(
                pd.DataFrame,
                joined.groupby("origin", sort=True, dropna=False)[
                    ["flow_adj", "flow_bench"]
                ].sum(),
            )
            loss = float((totals["flow_adj"] - totals["flow_bench"]).abs().sum())
        else:
            loss = float((joined["flow_adj"] - joined["flow_bench"]).abs().sum())
        candidates.append(candidate_value)
        losses.append(loss)

    if not candidates:
        raise DebiasSchemaError(
            "Calibration failed: no overlap between adjusted and benchmark flows."
        )
    best = int(np.argmin(losses))
    return candidates[best], pd.DataFrame({"k": candidates, "loss": losses})
