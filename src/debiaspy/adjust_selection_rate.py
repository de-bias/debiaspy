from __future__ import annotations

from numbers import Real
from typing import Literal, cast

import numpy as np
import pandas as pd

from debiaspy._adjustment_common import (
    clamp_weight,
    collapse_exact_key_duplicates,
    existing_columns,
    join_with_r_warning,
    match_choice,
    normalize01,
    require_no_reserved_columns,
    validate_clip_bounds,
)
from debiaspy._exceptions import DebiasSchemaError
from debiaspy._schemas import require_columns


def adjust_selection_rate(
    mpd_od_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    covariates_df: pd.DataFrame | None = None,
    covariate_col: str | None = None,
    weight_by: Literal["origin", "destination", "both"] = "origin",
    r_global: float | None = None,
    benchmark_od_df: pd.DataFrame | None = None,
    flow_col_bench: str = "flow",
    r_grid: list[float] | tuple[float, ...] | np.ndarray | None = None,
    calibration_aggregate: Literal["origin", "od"] = "origin",
    clip_min: float = 0,
    clip_max: float = np.inf,
    keep_cols: str | list[str] | tuple[str, ...] = (),
    **kwargs: object,
) -> pd.DataFrame:
    """Apply the covariate-dependent selection-rate flow correction.

    ``mpd_od_df`` requires ``origin``, ``destination``, ``flow``, and
    ``mpd_source``. ``coverage_df`` must use either the documented endpoint
    population/user-count schema or the legacy area schema; optional
    covariates require ``area`` and ``covariate_col``. A benchmark OD table can
    calibrate ``r_global`` over ``r_grid`` at origin or OD level.

    Returns MPD rows with endpoint weights and ``flow_adj``. The selected value
    is in ``attrs["r_global"]`` and calibration diagnostics, when used, are in
    ``attrs["r_calibration"]``. Coverage rows are filtered as in R; missing
    weights produce missing adjusted flows. Calibration uses OD-only benchmark
    joins and removes missing values from its loss calculation.
    """
    weight_by = match_choice(weight_by, ("origin", "destination", "both"), "weight_by")
    calibration_aggregate = match_choice(
        calibration_aggregate, ("origin", "od"), "calibration_aggregate"
    )

    legacy_income_col = kwargs.pop("income_col", None)
    if kwargs:
        raise DebiasSchemaError(
            "Unused argument(s): " + ", ".join(str(key) for key in kwargs)
        )
    if (
        covariate_col is not None
        and legacy_income_col is not None
        and covariate_col != legacy_income_col
    ):
        raise DebiasSchemaError(
            "Use only one of `covariate_col` or legacy `income_col`."
        )
    if covariate_col is None and legacy_income_col is not None:
        covariate_col = str(legacy_income_col)

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
    schema = _coverage_schema(coverage_df)
    covariate_values = _build_covariate_values(covariates_df, covariate_col)

    if schema == "new":
        cov_o_pre = _new_origin_coverage(coverage_df, covariate_values)
        cov_d_pre = _new_destination_coverage(coverage_df, covariate_values)
        cov_base_pre = None
    else:
        cov_base_pre = _legacy_coverage(coverage_df, covariate_values)
        cov_o_pre = None
        cov_d_pre = None

    def compute_with_r(
        r_value: float,
        additional_keep_cols: tuple[str, ...] = (),
    ) -> pd.DataFrame:
        if schema == "new":
            cov_o, cov_d = _selection_weights_new(
                cov_o_pre,
                cov_d_pre,
                weight_by,
                r_value,
                clip_min,
                clip_max,
            )
        else:
            cov_o, cov_d = _selection_weights_legacy(
                cov_base_pre,
                weight_by,
                r_value,
                clip_min,
                clip_max,
            )
        return _apply_selection_weights(
            mpd_od_df,
            cov_o,
            cov_d,
            weight_by,
            [*existing_columns(mpd_od_df, keep_cols), *additional_keep_cols],
        )

    if r_grid is None:
        r_grid = np.arange(0, 3.0000000001, 0.01)

    calibration: pd.DataFrame | None = None
    if r_global is not None:
        if (
            isinstance(r_global, (bool, np.bool_))
            or not isinstance(r_global, Real)
            or not np.isfinite(r_global)
            or r_global < 0
        ):
            raise DebiasSchemaError(
                "`r_global` must be a single finite non-negative number."
            )
        r_used = float(r_global)
    elif benchmark_od_df is not None:
        r_used, calibration = _calibrate_r(
            compute_with_r,
            benchmark_od_df,
            flow_col_bench,
            r_grid,
            calibration_aggregate,
        )
    elif schema == "new":
        assert cov_o_pre is not None and cov_d_pre is not None
        if weight_by == "origin":
            r_used = float(cov_o_pre["U_o"].sum() / cov_o_pre["P_o"].sum())
        elif weight_by == "destination":
            r_used = float(cov_d_pre["U_d"].sum() / cov_d_pre["P_d"].sum())
        else:
            numerator = cov_o_pre["U_o"].sum() + cov_d_pre["U_d"].sum()
            denominator = cov_o_pre["P_o"].sum() + cov_d_pre["P_d"].sum()
            r_used = float(numerator / denominator)
    else:
        assert cov_base_pre is not None
        r_used = float(cov_base_pre["U"].sum() / cov_base_pre["P"].sum())

    result = compute_with_r(r_used)
    result.attrs["r_global"] = r_used
    if calibration is not None:
        result.attrs["r_calibration"] = calibration
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
        "Legacy: {origin, population, user_count, mpd_source}\n"
        f"Got: {{{', '.join(coverage_df.columns)}}}"
    )


def _build_covariate_values(
    covariates_df: pd.DataFrame | None,
    covariate_col: str | None,
) -> pd.DataFrame | None:
    if covariates_df is None:
        return None
    require_columns(covariates_df, ("area",), "covariates_df")

    if covariate_col is not None:
        if covariate_col not in covariates_df.columns:
            raise DebiasSchemaError(
                f"`covariate_col` = '{covariate_col}' not found in `covariates_df`."
            )
        covariate_col_use = covariate_col
    else:
        candidates = [
            "income_norm",
            "gni_pc",
            "income",
            "inc",
            "gni",
            "income_2000",
            "income_2010",
        ]
        covariate_col_use = next(
            (
                candidate
                for candidate in candidates
                if candidate in covariates_df.columns
            ),
            None,
        )
    if covariate_col_use is None:
        return None

    out = pd.DataFrame(
        {
            "area": covariates_df["area"].astype(str),
            "covariate_norm": normalize01(covariates_df[covariate_col_use]),
        }
    )
    if "mpd_source" in covariates_df.columns:
        out["mpd_source"] = covariates_df["mpd_source"].to_numpy()
    keys = ["area"]
    if "mpd_source" in out.columns:
        keys.append("mpd_source")
    return collapse_exact_key_duplicates(
        out,
        keys=keys,
        value_columns=("covariate_norm",),
        frame_name="covariates_df",
    )


def _new_origin_coverage(
    coverage_df: pd.DataFrame,
    covariate_values: pd.DataFrame | None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "origin": coverage_df["origin"].astype(str),
            "mpd_source": coverage_df["mpd_source"],
            "P_o": pd.to_numeric(coverage_df["origin_population"], errors="coerce"),
            "U_o": pd.to_numeric(coverage_df["origin_user_count"], errors="coerce"),
        }
    )
    out = out.loc[
        np.isfinite(out["P_o"])
        & (out["P_o"] > 0)
        & np.isfinite(out["U_o"])
        & (out["U_o"] > 0)
    ].copy()
    if covariate_values is not None:
        if "mpd_source" in covariate_values.columns:
            out = out.merge(
                covariate_values,
                how="left",
                left_on=["origin", "mpd_source"],
                right_on=["area", "mpd_source"],
                sort=False,
                validate="many_to_one",
            )
        else:
            out = out.merge(
                covariate_values,
                how="left",
                left_on="origin",
                right_on="area",
                sort=False,
                validate="many_to_one",
            )
        income = pd.to_numeric(out["covariate_norm"], errors="coerce")
    else:
        income = pd.Series(1.0, index=out.index, dtype=float)
    out["Income_o"] = income.where(np.isfinite(income), 1.0)
    out["pen_o"] = out["U_o"] / out["P_o"]
    return out.loc[:, ["origin", "mpd_source", "P_o", "U_o", "Income_o", "pen_o"]]


def _new_destination_coverage(
    coverage_df: pd.DataFrame,
    covariate_values: pd.DataFrame | None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "destination": coverage_df["destination"].astype(str),
            "mpd_source": coverage_df["mpd_source"],
            "P_d": pd.to_numeric(
                coverage_df["destination_population"], errors="coerce"
            ),
            "U_d": pd.to_numeric(
                coverage_df["destination_user_count"], errors="coerce"
            ),
        }
    )
    out = out.loc[
        np.isfinite(out["P_d"])
        & (out["P_d"] > 0)
        & np.isfinite(out["U_d"])
        & (out["U_d"] > 0)
    ].copy()
    if covariate_values is not None:
        if "mpd_source" in covariate_values.columns:
            out = out.merge(
                covariate_values,
                how="left",
                left_on=["destination", "mpd_source"],
                right_on=["area", "mpd_source"],
                sort=False,
                validate="many_to_one",
            )
        else:
            out = out.merge(
                covariate_values,
                how="left",
                left_on="destination",
                right_on="area",
                sort=False,
                validate="many_to_one",
            )
        income = pd.to_numeric(out["covariate_norm"], errors="coerce")
    else:
        income = pd.Series(1.0, index=out.index, dtype=float)
    out["Income_d"] = income.where(np.isfinite(income), 1.0)
    out["pen_d"] = out["U_d"] / out["P_d"]
    return out.loc[:, ["destination", "mpd_source", "P_d", "U_d", "Income_d", "pen_d"]]


def _legacy_coverage(
    coverage_df: pd.DataFrame,
    covariate_values: pd.DataFrame | None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "area": coverage_df["origin"].astype(str),
            "mpd_source": coverage_df["mpd_source"],
            "P": pd.to_numeric(coverage_df["population"], errors="coerce"),
            "U": pd.to_numeric(coverage_df["user_count"], errors="coerce"),
        }
    )
    out = out.loc[
        np.isfinite(out["P"]) & (out["P"] > 0) & np.isfinite(out["U"]) & (out["U"] > 0)
    ].copy()
    if covariate_values is not None:
        if "mpd_source" in covariate_values.columns:
            out = out.merge(
                covariate_values,
                how="left",
                on=["area", "mpd_source"],
                sort=False,
                validate="many_to_one",
            )
        else:
            out = out.merge(
                covariate_values,
                how="left",
                on="area",
                sort=False,
                validate="many_to_one",
            )
        income = pd.to_numeric(out["covariate_norm"], errors="coerce")
    else:
        income = pd.Series(1.0, index=out.index, dtype=float)
    out["Income"] = income.where(np.isfinite(income), 1.0)
    out["pen"] = out["U"] / out["P"]
    return out.loc[:, ["area", "mpd_source", "P", "U", "Income", "pen"]]


def _selection_weights_new(
    cov_o_pre: pd.DataFrame | None,
    cov_d_pre: pd.DataFrame | None,
    weight_by: str,
    r_value: float,
    clip_min: float,
    clip_max: float,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    cov_o = None
    cov_d = None
    if weight_by in {"origin", "both"}:
        assert cov_o_pre is not None
        cov_o = cov_o_pre.copy()
        cov_o["weight_origin"] = clamp_weight(
            1
            / (cov_o["Income_o"] * cov_o["pen_o"] + (1 - cov_o["Income_o"]) * r_value),
            clip_min,
            clip_max,
        )
        cov_o = cov_o.loc[:, ["origin", "mpd_source", "weight_origin"]]
    if weight_by in {"destination", "both"}:
        assert cov_d_pre is not None
        cov_d = cov_d_pre.copy()
        cov_d["weight_destination"] = clamp_weight(
            1
            / (cov_d["Income_d"] * cov_d["pen_d"] + (1 - cov_d["Income_d"]) * r_value),
            clip_min,
            clip_max,
        )
        cov_d = cov_d.loc[:, ["destination", "mpd_source", "weight_destination"]]
    return cov_o, cov_d


def _selection_weights_legacy(
    cov_base_pre: pd.DataFrame | None,
    weight_by: str,
    r_value: float,
    clip_min: float,
    clip_max: float,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    assert cov_base_pre is not None
    cov_w = cov_base_pre.copy()
    cov_w["w_sel"] = clamp_weight(
        1 / (cov_w["Income"] * cov_w["pen"] + (1 - cov_w["Income"]) * r_value),
        clip_min,
        clip_max,
    )
    cov_o = None
    cov_d = None
    if weight_by in {"origin", "both"}:
        cov_o = (
            cov_w.loc[:, ["area", "mpd_source", "w_sel"]]
            .rename(columns={"area": "origin", "w_sel": "weight_origin"})
            .drop_duplicates(subset=["origin", "mpd_source"], keep="first")
        )
    if weight_by in {"destination", "both"}:
        cov_d = (
            cov_w.loc[:, ["area", "mpd_source", "w_sel"]]
            .rename(columns={"area": "destination", "w_sel": "weight_destination"})
            .drop_duplicates(subset=["destination", "mpd_source"], keep="first")
        )
    return cov_o, cov_d


def _apply_selection_weights(
    mpd_od_df: pd.DataFrame,
    cov_o: pd.DataFrame | None,
    cov_d: pd.DataFrame | None,
    weight_by: str,
    keep_cols: str | list[str] | tuple[str, ...],
) -> pd.DataFrame:
    out = mpd_od_df.copy()
    if cov_o is not None:
        out = out.merge(
            cov_o,
            how="left",
            on=["origin", "mpd_source"],
            sort=False,
            validate="many_to_one",
        )
    else:
        out["weight_origin"] = np.nan
    if cov_d is not None:
        out = out.merge(
            cov_d,
            how="left",
            on=["destination", "mpd_source"],
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
            *keep,
            "flow",
            "weight_origin",
            "weight_destination",
            "weight_missing",
            "flow_adj",
        ],
    )
    return out.loc[:, columns].reset_index(drop=True)


def _calibrate_r(
    compute_with_r: object,
    benchmark_od_df: pd.DataFrame,
    flow_col_bench: str,
    r_grid: list[float] | tuple[float, ...] | np.ndarray,
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
    if len(r_grid) == 0:
        raise DebiasSchemaError("`r_grid` is empty; provide candidate values.")

    candidates: list[float] = []
    losses: list[float] = []
    for candidate in r_grid:
        candidate_value = float(candidate)
        adjusted = compute_with_r(candidate_value)
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
    return candidates[best], pd.DataFrame({"r": candidates, "loss": losses})
