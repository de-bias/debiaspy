"""Overall, pair-level and spatial residual diagnostics for OD flows."""

import warnings
from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from debiaspy.flow_comparisons import (
    _SERIES_COLUMNS,
    _flow_comparison_spec,
    _normalise_flow_comparisons,
)
from debiaspy.measure_bias import (
    _compute_moran_i,
    _finite_pair_mask,
    _prepare_neighbor_links,
    _require_columns,
    _require_dataframe,
    _require_numeric_columns,
    _require_string_column_names,
    _safe_pearson,
)


def validate_flow_overall(
    adj_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame,
    flow_col_adj: str = "flow_adj",
    flow_col_bench: str = "flow",
    drop_zeros: bool = True,
    na_rm: bool = True,
    by_source: bool = False,
    return_joined: bool = True,
    method_name: str | None = None,
    comparisons: str | Sequence[str] = "adjusted_vs_benchmark",
    flow_col_mpd: str = "flow",
) -> dict[str, Any]:
    """Compare adjusted and/or raw flows against a reference series.

    Parameters
    ----------
    adj_df, benchmark_od_df:
        DataFrames with ``origin``, ``destination`` and the requested flow
        columns. Raw flows are required only for comparisons involving raw MPD.
        ``mpd_source`` is retained from ``adj_df`` if present.
    flow_col_adj, flow_col_bench, flow_col_mpd:
        Adjusted, benchmark and raw flow column names.
    drop_zeros:
        Keep only rows with both compared flows strictly positive. This also
        drops negative values, matching the reference implementation.
    na_rm:
        Remove non-finite x/y pairs before calculating each comparison.
    by_source:
        Also calculate metrics by source when ``adj_df`` has ``mpd_source``.
        Missing source IDs contribute to overall metrics but not source groups.
    return_joined:
        Include the filtered, comparison-level joined table under ``data``.
    method_name:
        Optional label stored as ``method`` in the result and metric tables.
    comparisons:
        A comparison ID, an ordered sequence of IDs, or ``"all"``. Supported
        IDs are ``adjusted_vs_benchmark``, ``raw_vs_benchmark`` and
        ``raw_vs_adjusted``. Repeated IDs are removed, preserving first order.

    Returns
    -------
    dict[str, Any]
        For one comparison, scalar metrics and comparison metadata. For multiple
        comparisons, ``method`` and a tidy ``summary`` DataFrame. Optional
        ``by_source`` and ``data`` DataFrames follow the corresponding flags.
        Empty comparison groups have n=0, zero sums and NaN fit/error metrics.

    Notes
    -----
    The first series is x; the second is y. Signed residuals are y - x, and OLS
    fits y ~ x. MAPE is a fraction, excludes zero reference flows, and is NaN
    when no denominator is usable. Constant series and singleton groups have
    undefined correlations and OLS metrics. With ``na_rm=False``, sums,
    mean_error and MAPE omit missing values, whereas RMSE, MAE and correlations
    propagate them. OLS additionally requires finite x + y.

    Both inputs are inner-joined on origin/destination only: unmatched pairs
    are excluded, missing keys match, and duplicates expand the join. Benchmark
    source labels are not join keys. Many-to-many joins emit UserWarning;
    essentially perfect OLS fits emit RuntimeWarning. Inputs are not modified.
    """
    selected = _normalise_flow_comparisons(comparisons)
    for name, value in (
        ("drop_zeros", drop_zeros),
        ("na_rm", na_rm),
        ("by_source", by_source),
        ("return_joined", return_joined),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"`{name}` must be a boolean.")
    if method_name is not None and not isinstance(method_name, str):
        raise TypeError("`method_name` must be a string or None.")
    raw_required = any(value != "adjusted_vs_benchmark" for value in selected)
    left, right = _prepare_flow_inputs(
        adj_df,
        benchmark_od_df,
        flow_col_adj,
        flow_col_mpd,
        flow_col_bench,
        raw_required,
        overall=True,
    )
    joined = _inner_join_flows(left, right)
    joined["adj_flow"] = joined["flow_adj"]
    joined["benchmark_flow"] = joined["flow_bench"]
    tables = []
    for comparison in selected:
        spec = _flow_comparison_spec(comparison)
        data = joined.copy()
        data["x_flow"] = data[_SERIES_COLUMNS[spec["x_series"]]]
        data["y_flow"] = data[_SERIES_COLUMNS[spec["y_series"]]]
        with np.errstate(invalid="ignore", over="ignore"):
            data["residual"] = _floats(data["y_flow"]) - _floats(data["x_flow"])
        for key, value in spec.items():
            data[key] = value
        keep = np.ones(len(data), dtype=bool)
        x, y = _floats(data["x_flow"]), _floats(data["y_flow"])
        if na_rm:
            keep &= np.isfinite(x) & np.isfinite(y)
        if drop_zeros:
            keep &= (x > 0) & (y > 0)
        tables.append(data.loc[keep])
    comparison_data = pd.concat(tables, ignore_index=True)
    rows = [
        _flow_overall_metric_row(data, comparison, method_name)
        for data, comparison in zip(tables, selected, strict=True)
    ]
    out = (
        rows[0].copy()
        if len(selected) == 1
        else {"method": method_name, "summary": pd.DataFrame(rows)}
    )
    if by_source and "mpd_source" in comparison_data:
        source_rows = []
        # R split orders source levels first and comparison labels second.
        for (source, comparison), data in comparison_data.groupby(
            ["mpd_source", "comparison"], sort=True, observed=True, dropna=True
        ):
            source_rows.append(
                {
                    **_flow_overall_metric_row(data, comparison, method_name),
                    "mpd_source": source,
                }
            )
        out["by_source"] = pd.DataFrame(source_rows)
    if return_joined:
        out["data"] = comparison_data
    return out


def validate_flow_pairs(
    adj_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
    flow_col_bench: str = "flow",
) -> pd.DataFrame:
    """Return matched OD pairs with raw, adjusted and benchmark differences.

    Parameters
    ----------
    adj_df, benchmark_od_df:
        DataFrames with origin/destination keys. ``adj_df`` must include raw
        and adjusted flows; the benchmark must contain its flow column.
    flow_col_mpd, flow_col_adj, flow_col_bench:
        Names of the raw, adjusted and benchmark flow columns.

    Returns
    -------
    pandas.DataFrame
        ``mpd_source`` if present in ``adj_df``, followed by origin, destination,
        ``mpd_flow``, ``adj_flow``, ``benchmark_flow``, three ``diff_*`` columns
        and three ``signed_residual_*`` columns. A fresh positional index is
        used. No zero, negative, missing or infinite flows are filtered out.

    Notes
    -----
    The join is inner on origin/destination only, including missing-key matches.
    Unmatched pairs disappear and duplicate keys can expand rows; many-to-many
    joins warn. Canonical residuals are reference minus baseline (y - x).
    The retained ``diff_*`` columns use the opposite, first-minus-second sign.
    Neither input is modified. Legacy function aliases are not provided.
    """
    left, right = _prepare_flow_inputs(
        adj_df,
        benchmark_od_df,
        flow_col_adj,
        flow_col_mpd,
        flow_col_bench,
        True,
        overall=False,
    )
    out = _inner_join_flows(left, right)
    raw, adjusted, benchmark = (
        _floats(out[col]) for col in ("mpd_flow", "adj_flow", "benchmark_flow")
    )
    with np.errstate(invalid="ignore", over="ignore"):
        out["diff_mpd_benchmark"] = raw - benchmark
        out["diff_mpd_adj"] = raw - adjusted
        out["diff_adj_benchmark"] = adjusted - benchmark
        out["signed_residual_adjusted_vs_benchmark"] = benchmark - adjusted
        out["signed_residual_raw_vs_benchmark"] = benchmark - raw
        out["signed_residual_raw_vs_adjusted"] = adjusted - raw
    return out


def _prepare_flow_inputs(
    adj_df,
    benchmark_od_df,
    flow_col_adj,
    flow_col_mpd,
    flow_col_bench,
    raw_required,
    *,
    overall,
):
    _require_string_column_names(flow_col_adj, flow_col_mpd, flow_col_bench)
    keys = ["origin", "destination"]
    for data, name in ((adj_df, "adj_df"), (benchmark_od_df, "benchmark_od_df")):
        _require_dataframe(data, name)
        if not data.columns.is_unique:
            raise ValueError(f"`{name}` must have unique column names.")
    adj_columns = [flow_col_adj] + ([flow_col_mpd] if raw_required else [])
    _require_columns(adj_df, [*keys, *adj_columns], "adj_df")
    _require_columns(benchmark_od_df, [*keys, flow_col_bench], "benchmark_od_df")
    raw_present = flow_col_mpd in adj_df
    if raw_present and flow_col_mpd == flow_col_adj:
        raise ValueError("Raw and adjusted flow columns must be distinct.")
    if any(
        col in [*keys, "mpd_source"]
        for col in [flow_col_adj, flow_col_mpd, flow_col_bench]
    ):
        raise ValueError("Flow columns must be distinct from identifier columns.")
    flow_columns = [flow_col_adj] + ([flow_col_mpd] if raw_present else [])
    for data, columns, name in (
        (adj_df, flow_columns, "adj_df"),
        (benchmark_od_df, [flow_col_bench], "benchmark_od_df"),
    ):
        _require_numeric_columns(data, columns, name)
        if any(pd.api.types.is_complex_dtype(data[column]) for column in columns):
            raise ValueError(f"`{name}` flow columns must be real-valued.")
    ids = (["mpd_source"] if "mpd_source" in adj_df else []) + keys
    if overall:
        # Keep the reference column order: adjusted before raw for overall data.
        left = adj_df.loc[:, ids + flow_columns].copy()
        rename = {flow_col_adj: "flow_adj"}
        if raw_present:
            rename[flow_col_mpd] = "mpd_flow"
        left = left.rename(columns=rename)
        if not raw_present:
            left["mpd_flow"] = np.nan
        right_name = "flow_bench"
    else:
        left = (
            adj_df.loc[:, ids + [flow_col_mpd, flow_col_adj]]
            .rename(columns={flow_col_mpd: "mpd_flow", flow_col_adj: "adj_flow"})
            .copy()
        )
        right_name = "benchmark_flow"
    right = (
        benchmark_od_df.loc[:, keys + [flow_col_bench]]
        .rename(columns={flow_col_bench: right_name})
        .copy()
    )
    return left, right


def _inner_join_flows(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = ["origin", "destination"]
    left_many, right_many = (
        left.duplicated(keys, keep=False),
        right.duplicated(keys, keep=False),
    )
    if left_many.any() and right_many.any():
        matches = (
            left[keys]
            .assign(_left_many=left_many)
            .drop_duplicates()
            .merge(
                right[keys].assign(_right_many=right_many).drop_duplicates(), on=keys
            )
        )
        if matches["_left_many"].any() and matches["_right_many"].any():
            warnings.warn(
                "Detected an unexpected many-to-many OD join.",
                UserWarning,
                stacklevel=3,
            )
    return left.merge(right, on=keys, how="inner", sort=False)


def _floats(values: pd.Series) -> np.ndarray:
    return values.to_numpy(dtype=float, na_value=np.nan)


def _mean_without_nan(values: np.ndarray) -> float:
    present = values[~np.isnan(values)]
    return float(np.mean(present)) if len(present) else np.nan


def _flow_overall_metric_row(
    data: pd.DataFrame, comparison: str, method_name: str | None
) -> dict[str, Any]:
    row = {
        "method": method_name,
        **_flow_comparison_spec(comparison),
        "n": len(data),
        **dict.fromkeys(("sum_x", "sum_y", "sum_adj", "sum_bench", "sum_mpd"), 0.0),
        **dict.fromkeys(
            (
                "mean_error",
                "pearson_r",
                "spearman_rho",
                "rmse",
                "mae",
                "mape",
                "ols_intercept",
                "ols_slope",
                "r_squared",
            ),
            np.nan,
        ),
    }
    if data.empty:
        return row
    x, y = _floats(data["x_flow"]), _floats(data["y_flow"])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        residual = y - x
        for output, column in (
            ("sum_x", "x_flow"),
            ("sum_y", "y_flow"),
            ("sum_adj", "flow_adj"),
            ("sum_bench", "flow_bench"),
            ("sum_mpd", "mpd_flow"),
        ):
            row[output] = float(np.nansum(_floats(data[column])))
        row["mean_error"] = _mean_without_nan(residual)
        row["rmse"] = float(np.sqrt(np.mean(residual**2)))
        row["mae"] = float(np.mean(np.abs(residual)))
        row["mape"] = _mean_without_nan(np.abs(residual / np.where(y == 0, np.nan, y)))
        x_valid, y_valid = x[~np.isnan(x)], y[~np.isnan(y)]
        x_sd = np.std(x_valid, ddof=1) if len(x_valid) > 1 else np.nan
        y_sd = np.std(y_valid, ddof=1) if len(y_valid) > 1 else np.nan
        can_cor = (
            len(x) >= 2
            and np.isfinite(x_sd)
            and x_sd > 0
            and np.isfinite(y_sd)
            and y_sd > 0
        )
        if can_cor:
            row["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
            xr, yr = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
            row["spearman_rho"] = float(np.corrcoef(xr, yr)[0, 1])
            if np.isfinite(x + y).all():
                row.update(_flow_ols_metrics(x, y))
    return row


def _flow_ols_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    centered = x - x.mean()
    # R lm's default QR tolerance drops a predictor collinear with the intercept.
    full_rank = np.linalg.norm(centered) >= 1e-7 * np.linalg.norm(x)
    slope = (
        float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))
        if full_rank
        else np.nan
    )
    intercept = float(y.mean() - slope * x.mean()) if full_rank else float(y.mean())
    fitted = intercept + slope * x if full_rank else np.full_like(y, intercept)
    rss = np.sum((y - fitted) ** 2)
    mss = np.sum((fitted - fitted.mean()) ** 2)
    rank = 2 if full_rank else 1
    resvar = rss / (len(x) - rank) if len(x) > rank else np.nan
    if (
        np.isfinite(resvar)
        and resvar < (np.mean(fitted) ** 2 + np.var(fitted, ddof=1)) * 1e-30
    ):
        warnings.warn(
            "Essentially perfect fit: summary may be unreliable.",
            RuntimeWarning,
            stacklevel=4,
        )
    return {
        "ols_intercept": intercept,
        "ols_slope": slope,
        "r_squared": float(mss / (mss + rss)) if full_rank else 0.0,
    }


def validate_flow_residuals(
    adj_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
    flow_col_bench: str = "flow",
    top_n: int = 10,
    method_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Return OD residuals, improvement diagnostics and the worst remaining errors.

    Parameters
    ----------
    adj_df, benchmark_od_df:
        DataFrames joined on origin/destination through ``validate_flow_pairs``.
        Adjusted inputs must contain raw and adjusted flows. Optional source
        labels are retained but are not join keys.
    flow_col_mpd, flow_col_adj, flow_col_bench:
        Raw, adjusted and benchmark flow column names.
    top_n:
        Number of worst pairs to retain; positive fractional values >= 1 are
        truncated to integers, matching the reference behaviour.
    method_name:
        Optional method label included in all outputs.

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``summary``, ``data`` and ``top_worst``. The worst rows are ordered by
        descending absolute adjusted error, then absolute raw error, with
        missing errors last and stable tie order.

    Notes
    -----
    Canonical residuals are reference minus baseline. Signed reduction is
    adjusted minus raw; absolute reduction is absolute raw benchmark error
    minus absolute adjusted benchmark error. Percentages are multiplied by
    100; zero denominators yield NaN. SD scores divide by the sample standard
    deviation without centering; constant/singleton series have missing scores.
    Missing improvements are labelled ``unchanged`` and missing direction
    comparisons are False, as in the reference. No flow rows are filtered.
    """
    if not _finite_number(top_n) or top_n < 1 or top_n >= 2**31:
        raise ValueError("`top_n` must be a single positive integer.")
    if method_name is not None and not isinstance(method_name, str):
        raise TypeError("`method_name` must be a string or None.")
    data = validate_flow_pairs(
        adj_df, benchmark_od_df, flow_col_mpd, flow_col_adj, flow_col_bench
    )
    data["method"] = method_name
    for target, source in {
        "residual_mpd_benchmark": "signed_residual_raw_vs_benchmark",
        "residual_adj_benchmark": "signed_residual_adjusted_vs_benchmark",
        "residual_mpd_adj": "signed_residual_raw_vs_adjusted",
        "benchmark_minus_mpd": "signed_residual_raw_vs_benchmark",
        "benchmark_minus_adj": "signed_residual_adjusted_vs_benchmark",
        "signed_residual_reduction": "signed_residual_raw_vs_adjusted",
    }.items():
        data[target] = data[source]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for name in ("mpd_benchmark", "adj_benchmark", "mpd_adj"):
            data[f"abs_residual_{name}"] = data[f"residual_{name}"].abs()
        benchmark = _floats(data["benchmark_flow"])
        for name in ("mpd_benchmark", "adj_benchmark"):
            data[f"pct_residual_{name}"] = (
                100
                * _floats(data[f"residual_{name}"])
                / np.where(benchmark == 0, np.nan, benchmark)
            )
        improvement = _floats(data["abs_residual_mpd_benchmark"]) - _floats(
            data["abs_residual_adj_benchmark"]
        )
        data["abs_residual_improvement"] = improvement
        data["abs_residual_reduction"] = improvement
        raw_absolute = _floats(data["abs_residual_mpd_benchmark"])
        data["pct_residual_improvement"] = (
            100 * improvement / np.where(raw_absolute == 0, np.nan, raw_absolute)
        )
        for name in ("mpd_benchmark", "adj_benchmark"):
            sd = _sample_sd(_floats(data[f"diff_{name}"]))
            data[f"residual_sd_{name}"] = sd
            score = (
                np.full(len(data), np.nan)
                if np.isnan(sd) or sd <= 0
                else _floats(data[f"residual_{name}"]) / sd
            )
            data[f"residual_{name}_sd_score"] = score
            data[f"abs_residual_{name}_sd_score"] = np.abs(score)
        for series in ("mpd", "adj"):
            score = _floats(data[f"abs_residual_{series}_benchmark_sd_score"])
            for threshold in (1, 2, 3):
                flag = pd.array(score > threshold, dtype="boolean")
                flag[np.isnan(score)] = pd.NA
                data[f"residual_{series}_over_{threshold}sd"] = flag
        raw_residual = _floats(data["benchmark_minus_mpd"])
        movement = _floats(data["signed_residual_reduction"])
        data["moved_in_benchmark_direction"] = (
            (raw_residual == 0) & (movement == 0)
        ) | (
            (raw_residual != 0)
            & (movement != 0)
            & (np.sign(raw_residual) == np.sign(movement))
        )
        data["improvement_flag"] = np.select(
            [improvement > 0, improvement < 0],
            ["improved", "worsened"],
            default="unchanged",
        )
        row = {"method": method_name, "n": len(data)}
        for statistic, column in (
            ("mean", "signed_residual_reduction"),
            ("median", "signed_residual_reduction"),
            ("mean", "abs_residual_reduction"),
            ("median", "abs_residual_reduction"),
            ("mean", "abs_residual_mpd_benchmark"),
            ("mean", "abs_residual_adj_benchmark"),
            ("median", "abs_residual_mpd_benchmark"),
            ("median", "abs_residual_adj_benchmark"),
        ):
            row[f"{statistic}_{column}"] = _residual_stat(
                _floats(data[column]), statistic
            )
        for flag in ("improved", "worsened", "unchanged"):
            row[f"share_{flag}"] = _residual_stat(
                (data["improvement_flag"] == flag).to_numpy(), "mean"
            )
        row["share_moved_in_benchmark_direction"] = _residual_stat(
            _floats(data["moved_in_benchmark_direction"]), "mean"
        )
        for series in ("mpd", "adj"):
            for threshold in (1, 2, 3):
                column = f"residual_{series}_over_{threshold}sd"
                row[f"share_{column}"] = _residual_stat(_floats(data[column]), "mean")
        row["reduction_share_residual_over_2sd"] = (
            row["share_residual_mpd_over_2sd"] - row["share_residual_adj_over_2sd"]
        )
    worst = (
        data.sort_values(
            ["abs_residual_adj_benchmark", "abs_residual_mpd_benchmark"],
            ascending=False,
            kind="stable",
        )
        .head(int(top_n))
        .reset_index(drop=True)
    )
    return {"summary": pd.DataFrame([row]), "data": data, "top_worst": worst}


def _finite_number(value):
    return (
        isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
    )


def _sample_sd(values):
    present = values[~np.isnan(values)]
    return float(np.std(present, ddof=1)) if len(present) > 1 else np.nan


def _residual_stat(values, statistic):
    present = values[~np.isnan(values)]
    if statistic == "sum":
        return float(np.sum(present))
    if not len(present):
        return np.nan
    return float(np.mean(present) if statistic == "mean" else np.median(present))


def _residual_choice(value, choices, name):
    if value is None:
        return choices[0]
    if isinstance(value, str) and value in choices:
        return value
    matches = [
        choice
        for choice in choices
        if isinstance(value, str) and value and choice.startswith(value)
    ]
    if len(matches) != 1:
        raise ValueError(f"`{name}` must be one of {', '.join(choices)}.")
    return matches[0]


def validate_flow_residual_structure(
    adj_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
    flow_col_bench: str = "flow",
    method_name: str | None = None,
    residual_type: str | None = "adjusted",
    comparison: str | Sequence[str] | None = None,
    spatial_role: str | None = "origin",
    residual_aggregation: str | None = "mean",
    area_neighbors: pd.DataFrame | None = None,
    area_col: str = "area",
    neighbor_col: str = "neighbor",
    weight_col: str | None = None,
    local_moran: bool = False,
    local_moran_nsim: int = 999,
    local_moran_alpha: float = 0.05,
    local_moran_p_adjust: str = "BH",
    local_moran_seed: int | None = None,
    covariate_df: pd.DataFrame | None = None,
    covariate_col: str | None = None,
    covariate_area_col: str = "area",
    geometry_df: pd.DataFrame | None = None,
    geometry_area_col: str = "area",
    x_col: str | None = None,
    y_col: str | None = None,
    make_plots: bool = False,
) -> dict[str, Any]:
    """Diagnose OD residual relationships and their area-level spatial structure.

    Parameters
    ----------
    adj_df, benchmark_od_df:
        Adjusted/raw and benchmark OD DataFrames, inner-joined on OD keys.
    flow_col_mpd, flow_col_adj, flow_col_bench:
        Raw, adjusted and benchmark flow columns. All three series are required.
    method_name:
        Optional label included in output tables.
    residual_type:
        Legacy shortcut: ``adjusted``, ``mpd`` or ``adjustment``.
    comparison:
        One of ``adjusted_vs_benchmark``, ``raw_vs_benchmark`` or
        ``raw_vs_adjusted``. Overrides the validated residual_type shortcut.
        Residuals always equal the second series minus the first.
    spatial_role, residual_aggregation:
        Aggregate by ``origin`` or ``destination``, using ``mean`` or ``sum``.
        Missing residuals are omitted; all-missing groups have NaN means and
        zero sums. The mean_residual, sum_residual and mean_abs_residual aliases
        describe the aggregated selected_residual, matching the reference.
        Unambiguous prefixes and None for defaults are accepted.
    area_neighbors:
        Optional directed edge table. Missing endpoints outside the area
        support are ignored. Duplicate and self links are retained.
    area_col, neighbor_col, weight_col:
        Edge endpoints and optional numeric weights. Default weights are one;
        nonpositive and non-finite weights are discarded, without row scaling.
    local_moran:
        Also return Local Moran statistics and LISA classifications.
    local_moran_nsim, local_moran_alpha, local_moran_p_adjust, local_moran_seed:
        Number of conditional permutations, significance threshold, multiple
        testing adjustment (holm, hochberg, hommel, bonferroni, BH, BY, fdr or
        none), and optional seed. Seeded permutations reproduce R's default
        Mersenne-Twister/Rejection sampling; other R RNG configurations are
        outside this contract. None uses fresh entropy, independently of NumPy
        global state. Validated only when local_moran is True.
    covariate_df, covariate_col, covariate_area_col:
        Optional area covariate, column to correlate and area key. Duplicate
        keys expand covariate rows; correlations use finite complete pairs.
    geometry_df, geometry_area_col:
        Optional table joined onto area-level map data, retaining all areas.
    x_col, y_col, make_plots:
        Return Matplotlib figures when requested. Both coordinate columns are
        needed for the optional coordinate map; coordinates are otherwise unused.

    Returns
    -------
    dict[str, Any]
        Tables ``summary``, ``flow_correlation``, ``moran_i``,
        ``covariate_correlation``, ``od_level``, ``area_level`` and ``map_data``.
        Optional ``covariate_data``, ``local_moran`` and ``plots`` are included
        when requested. Inputs are not modified.

    Notes
    -----
    Flow correlation always uses benchmark flow, even for raw_vs_adjusted.
    Global Moran uses all finite area residuals for its mean and denominator,
    including islands. Local Moran uses population variance and unnormalised
    weighted lag sums; displayed lag means divide by local weight sums.
    Conditional permutations hold the focal residual fixed and permute the
    remaining finite values. Pseudo p-values are two-sided absolute-statistic
    exceedance counts with a plus-one correction, adjusted over finite p-values.
    Missing links yield ``no neighbours``; constant residuals yield ``undefined``.
    """
    residual_type = _residual_choice(
        residual_type, ("adjusted", "mpd", "adjustment"), "residual_type"
    )
    types = dict(
        zip(
            ("adjusted_vs_benchmark", "raw_vs_benchmark", "raw_vs_adjusted"),
            ("adjusted", "mpd", "adjustment"),
            strict=True,
        )
    )
    if comparison is None:
        comparison = next(key for key, value in types.items() if value == residual_type)
    else:
        selected = _normalise_flow_comparisons(comparison)
        if len(selected) != 1:
            raise ValueError("`comparison` must select exactly one flow comparison.")
        comparison = selected[0]
    residual_type = types[comparison]
    spec = _flow_comparison_spec(comparison)
    spatial_role = _residual_choice(
        spatial_role, ("origin", "destination"), "spatial_role"
    )
    residual_aggregation = _residual_choice(
        residual_aggregation, ("mean", "sum"), "residual_aggregation"
    )
    for name, value in (("local_moran", local_moran), ("make_plots", make_plots)):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"`{name}` must be True or False.")
    if covariate_col is not None and covariate_df is None:
        raise ValueError("`covariate_df` is required when `covariate_col` is supplied.")
    if local_moran:
        _validate_local_moran_args(
            local_moran_nsim, local_moran_alpha, local_moran_p_adjust, local_moran_seed
        )
    if area_neighbors is not None:
        _validate_spatial_table(
            area_neighbors, "area_neighbors", [area_col, neighbor_col]
        )
        if area_col == neighbor_col or weight_col in (area_col, neighbor_col):
            raise ValueError("Neighbour endpoint and weight columns must be distinct.")
        if weight_col is not None and weight_col in area_neighbors:
            _require_real_column(area_neighbors, weight_col, "area_neighbors")
    residuals = validate_flow_residuals(
        adj_df,
        benchmark_od_df,
        flow_col_mpd,
        flow_col_adj,
        flow_col_bench,
        method_name=method_name,
    )
    od = residuals["data"].copy()
    od["selected_residual"] = od[f"signed_residual_{comparison}"]
    od["residual_type"] = residual_type
    for key, value in spec.items():
        od[key] = value
    metadata = {
        "method": method_name,
        "residual_type": residual_type,
        "comparison": comparison,
        "comparison_label": spec["comparison_label"],
    }
    spatial_metadata = {
        **metadata,
        "spatial_role": spatial_role,
        "residual_aggregation": residual_aggregation,
    }
    flow_correlation = {
        **metadata,
        "benchmark_comparison": "benchmark_flow",
        "n": int(
            _finite_pair_mask(od["selected_residual"], od["benchmark_flow"]).sum()
        ),
        "pearson_r": _safe_pearson(od["selected_residual"], od["benchmark_flow"]),
    }
    area_rows = []
    area_columns = [
        "area",
        *spatial_metadata,
        "n_od_pairs",
        "selected_residual",
        "mean_residual",
        "sum_residual",
        "mean_abs_residual",
        "benchmark_flow_sum",
        "mpd_flow_sum",
        "adj_flow_sum",
    ]
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        for area, group in od.groupby(
            spatial_role, sort=True, dropna=False, observed=True
        ):
            values = _floats(group["selected_residual"])
            # R summarise replaces selected_residual before deriving these aliases.
            aggregated = np.array([_residual_stat(values, residual_aggregation)])
            area_rows.append(
                {
                    "area": area,
                    **spatial_metadata,
                    "n_od_pairs": len(group),
                    "selected_residual": _residual_stat(values, residual_aggregation),
                    "mean_residual": _residual_stat(aggregated, "mean"),
                    "sum_residual": _residual_stat(aggregated, "sum"),
                    "mean_abs_residual": _residual_stat(np.abs(aggregated), "mean"),
                    **{
                        f"{name}_sum": _residual_stat(_floats(group[name]), "sum")
                        for name in ("benchmark_flow", "mpd_flow", "adj_flow")
                    },
                }
            )
    areas = pd.DataFrame(area_rows, columns=area_columns)
    # Empty tables still need numeric diagnostic columns for finite-value checks.
    for column in area_columns[1 + len(spatial_metadata) :]:
        areas[column] = pd.to_numeric(areas[column])
    moran_stats = {
        "moran_i": np.nan,
        "n_areas_used": int(np.isfinite(areas["selected_residual"]).sum()),
        "n_links_used": np.nan,
        "weight_sum": np.nan,
    }
    if area_neighbors is not None:
        moran_stats = _compute_moran_i(
            areas, area_neighbors, area_col, neighbor_col, weight_col
        )
    moran = {
        **spatial_metadata,
        **{
            key: moran_stats[key]
            for key in ("n_areas_used", "n_links_used", "weight_sum", "moran_i")
        },
    }
    cov_data = None
    cov_correlation = {
        **metadata,
        "spatial_role": spatial_role,
        "covariate": None,
        "n": np.nan,
        "pearson_r": np.nan,
    }
    if covariate_col is not None:
        _validate_spatial_table(
            covariate_df, "covariate_df", [covariate_area_col, covariate_col]
        )
        if covariate_col == covariate_area_col:
            raise ValueError("Covariate and area columns must be distinct.")
        _require_real_column(covariate_df, covariate_col, "covariate_df")
        cov = covariate_df[[covariate_area_col, covariate_col]].copy()
        cov.columns = ["area", "covariate_value"]
        # R right_join retains matched covariate order, then unmatched areas.
        cov_data = cov.assign(_cov_order=np.arange(len(cov))).merge(
            areas.assign(_area_order=np.arange(len(areas))),
            how="outer",
            on="area",
            sort=False,
        )
        cov_data = (
            cov_data.loc[cov_data["_area_order"].notna()]
            .sort_values(["_cov_order", "_area_order"], kind="stable")
            .drop(columns=["_cov_order", "_area_order"])
            .reset_index(drop=True)
        )
        cov_correlation.update(
            covariate=covariate_col,
            n=int(
                _finite_pair_mask(
                    cov_data["selected_residual"], cov_data["covariate_value"]
                ).sum()
            ),
            pearson_r=_safe_pearson(
                cov_data["selected_residual"], cov_data["covariate_value"]
            ),
        )
    map_data = areas.copy()
    if geometry_df is not None:
        _validate_spatial_table(geometry_df, "geometry_df", [geometry_area_col])
        geometry = geometry_df.rename(columns={geometry_area_col: "area"})
        if not geometry.columns.is_unique:
            raise ValueError("Renaming `geometry_area_col` would duplicate 'area'.")
        map_data = areas.merge(
            geometry, on="area", how="left", sort=False, suffixes=(".x", ".y")
        )
    local_table = None
    if local_moran:
        local_table = _compute_local_moran(
            areas,
            area_neighbors,
            area_col,
            neighbor_col,
            weight_col,
            int(local_moran_nsim),
            local_moran_alpha,
            local_moran_p_adjust,
            local_moran_seed,
        )
        local_columns = [
            "centered_residual",
            "residual_z",
            "neighbor_lag_sum",
            "neighbor_lag_mean",
            "spatial_lag_z",
            "n_neighbors_used",
            "local_weight_sum",
            "local_moran_i",
            "lisa_quadrant",
            "p_value",
            "p_adjusted",
            "significant",
            "lisa_cluster",
        ]
        map_data = map_data.merge(
            local_table[["area", *local_columns]],
            on="area",
            how="left",
            sort=False,
            suffixes=(".x", ".y"),
        )
    summary = {
        **spatial_metadata,
        "n_od_pairs": len(od),
        "n_areas": len(areas),
        "pearson_residual_benchmark_flow": flow_correlation["pearson_r"],
        "moran_i": moran["moran_i"],
        "pearson_residual_covariate": cov_correlation["pearson_r"],
    }
    out = {
        "summary": pd.DataFrame([summary]),
        "flow_correlation": pd.DataFrame([flow_correlation]),
        "moran_i": pd.DataFrame([moran]),
        "covariate_correlation": pd.DataFrame([cov_correlation]),
        "od_level": od,
        "area_level": areas,
        "map_data": map_data,
    }
    if cov_data is not None:
        out["covariate_data"] = cov_data
    if local_table is not None:
        out["local_moran"] = local_table
    if make_plots:
        out["plots"] = _make_flow_residual_plots(
            od,
            cov_data,
            covariate_col,
            map_data,
            x_col,
            y_col,
            spec["comparison_label"],
        )
    return out


def _validate_spatial_table(data, name, columns):
    _require_dataframe(data, name)
    _require_string_column_names(*columns)
    if not data.columns.is_unique:
        raise ValueError(f"`{name}` must have unique column names.")
    _require_columns(data, columns, name)


def _require_real_column(data, column, name):
    _require_numeric_columns(data, [column], name)
    if pd.api.types.is_complex_dtype(data[column]):
        raise ValueError(f"`{name}` column {column!r} must be real-valued.")


def _validate_local_moran_args(nsim, alpha, p_adjust, seed):
    if not _finite_number(nsim) or nsim < 1 or nsim != int(nsim) or nsim >= 2**31:
        raise ValueError("`local_moran_nsim` must be a positive whole number.")
    if not _finite_number(alpha) or not 0 <= alpha <= 1:
        raise ValueError("`local_moran_alpha` must be a number between 0 and 1.")
    if not isinstance(p_adjust, str) or p_adjust not in (
        "holm",
        "hochberg",
        "hommel",
        "bonferroni",
        "BH",
        "BY",
        "fdr",
        "none",
    ):
        raise ValueError(
            "`local_moran_p_adjust` must be one of holm, hochberg, hommel, "
            "bonferroni, BH, BY, fdr, none."
        )
    if seed is not None and (not _finite_number(seed) or not -(2**31) < seed < 2**31):
        raise ValueError(
            "`local_moran_seed` must be None or a single finite signed 32-bit number."
        )


def _compute_local_moran(
    areas, neighbors, area_col, neighbor_col, weight_col, nsim, alpha, p_adjust, seed
):
    base = areas[
        [
            "area",
            "method",
            "residual_type",
            "comparison",
            "comparison_label",
            "spatial_role",
            "residual_aggregation",
            "selected_residual",
        ]
    ].copy()
    columns = [
        "centered_residual",
        "residual_z",
        "neighbor_lag_sum",
        "neighbor_lag_mean",
        "spatial_lag_z",
        "n_neighbors_used",
        "local_weight_sum",
        "local_moran_i",
        "lisa_quadrant",
        "p_value",
        "p_adjusted",
        "significant",
        "lisa_cluster",
    ]
    residuals = _floats(base["selected_residual"])
    finite = np.isfinite(residuals)
    values = residuals[finite]
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        sd = _sample_sd(values)
        mean = float(np.mean(values)) if len(values) else np.nan
        denominator = float(np.sum((values - mean) ** 2))
    links = (
        None
        if neighbors is None
        else _prepare_neighbor_links(neighbors, area_col, neighbor_col, weight_col)
    )
    undefined = (
        len(values) < 2
        or not np.isfinite(sd)
        or sd <= 0
        or not np.isfinite(denominator)
        or denominator <= 0
    )
    for column in columns:
        base[column] = np.nan
    base["n_neighbors_used"] = 0
    base["significant"] = pd.array([pd.NA] * len(base), dtype="boolean")
    if links is None or undefined:
        label = "no neighbours" if links is None else "undefined"
        base["lisa_quadrant"] = label
        base["lisa_cluster"] = label
        return base
    variance = denominator / len(values)
    base.loc[finite, "centered_residual"] = residuals[finite] - mean
    base.loc[finite, "residual_z"] = (residuals[finite] - mean) / np.sqrt(variance)
    keys = base.loc[finite, ["area", "centered_residual", "residual_z"]].assign(
        _idx=np.flatnonzero(finite)
    )
    links = links.merge(
        keys[["area", "_idx"]], on="area", how="inner", sort=False
    ).merge(
        keys.rename(columns={"area": "neighbor", "_idx": "_neighbor_idx"}),
        on="neighbor",
        how="inner",
        sort=False,
    )
    for focal, group in links.groupby("_idx", sort=True):
        weights = _floats(group["weight"])
        total = np.sum(weights)
        lag_sum = np.sum(weights * _floats(group["centered_residual"]))
        base.loc[focal, "n_neighbors_used"] = len(group)
        base.loc[focal, "local_weight_sum"] = total
        base.loc[focal, "neighbor_lag_sum"] = lag_sum
        base.loc[focal, "neighbor_lag_mean"] = lag_sum / total
        base.loc[focal, "spatial_lag_z"] = (
            np.sum(weights * _floats(group["residual_z"])) / total
        )
        base.loc[focal, "local_moran_i"] = (
            base.loc[focal, "centered_residual"] * lag_sum / variance
        )
    centered = _floats(base["centered_residual"])
    observed = _floats(base["local_moran_i"])
    lag_mean = _floats(base["neighbor_lag_mean"])
    has_neighbors = base["n_neighbors_used"].to_numpy() > 0
    eligible = (
        has_neighbors
        & np.isfinite(observed)
        & np.isfinite(centered)
        & np.isfinite(lag_mean)
    )
    if eligible.any():
        from debiaspy._r_permutation import _RPermutation

        rng = _RPermutation(seed)
        for focal in np.flatnonzero(eligible):
            others = np.flatnonzero(
                np.isfinite(centered) & (np.arange(len(base)) != focal)
            )
            focal_links = links.loc[links["_idx"] == focal]
            neighbor_indices = focal_links["_neighbor_idx"].to_numpy(dtype=int)
            weights = _floats(focal_links["weight"])
            count = 0
            for _ in range(nsim):
                simulated = centered.copy()
                simulated[others] = rng.permute(centered[others])
                statistic = (
                    centered[focal]
                    * np.sum(weights * simulated[neighbor_indices])
                    / variance
                )
                count += abs(statistic) >= abs(observed[focal])
            base.loc[focal, "p_value"] = (count + 1) / (nsim + 1)
    finite_p = np.isfinite(_floats(base["p_value"]))
    base.loc[finite_p, "p_adjusted"] = _adjust_local_pvalues(
        _floats(base.loc[finite_p, "p_value"]), p_adjust
    )
    base.loc[finite_p, "significant"] = base.loc[finite_p, "p_adjusted"] <= alpha
    defined = np.isfinite(observed)
    quadrants = np.select(
        [
            ~has_neighbors,
            ~defined,
            (centered > 0) & (lag_mean > 0),
            (centered < 0) & (lag_mean < 0),
            (centered > 0) & (lag_mean < 0),
            (centered < 0) & (lag_mean > 0),
        ],
        ["no neighbours", "undefined", "high-high", "low-low", "high-low", "low-high"],
        default="undefined",
    )
    base["lisa_quadrant"] = quadrants
    significant = base["significant"].fillna(False).to_numpy(dtype=bool)
    base["lisa_cluster"] = np.select(
        [~has_neighbors, ~defined, ~significant],
        ["no neighbours", "undefined", "not significant"],
        default=quadrants,
    )
    # The regular R join path orders neighbour counts before lag diagnostics;
    # its no-neighbour/undefined early returns retain the original order.
    return base[
        [
            "area",
            "method",
            "residual_type",
            "comparison",
            "comparison_label",
            "spatial_role",
            "residual_aggregation",
            "selected_residual",
            "centered_residual",
            "residual_z",
            "n_neighbors_used",
            "local_weight_sum",
            "neighbor_lag_sum",
            "neighbor_lag_mean",
            "spatial_lag_z",
            "local_moran_i",
            "lisa_quadrant",
            "p_value",
            "p_adjusted",
            "significant",
            "lisa_cluster",
        ]
    ]


def _adjust_local_pvalues(values, method):
    """Adjust finite p-values with the named family-wise or FDR procedure."""
    n = len(values)
    if n < 2 or method == "none":
        return values.copy()
    if method == "bonferroni":
        return np.minimum(1, n * values)
    order = np.argsort(values, kind="stable")
    p = values[order]
    ranks = np.arange(1, n + 1)
    if method == "holm":
        adjusted = np.maximum.accumulate((n + 1 - ranks) * p)
    elif method == "hochberg" or (method == "hommel" and n == 2):
        adjusted = np.minimum.accumulate(((n + 1 - ranks) * p)[::-1])[::-1]
    elif method in ("BH", "BY", "fdr"):
        factor = np.sum(1 / ranks) if method == "BY" else 1
        adjusted = np.minimum.accumulate((factor * n / ranks * p)[::-1])[::-1]
    else:
        # Hommel's closed Simes procedure, evaluated over sorted p-values.
        adjusted = np.full(n, np.min(n * p / ranks))
        for size in range(n - 1, 1, -1):
            boundary = n - size + 1
            tail_bound = np.min(size * p[boundary:] / np.arange(2, size + 1))
            candidate = np.minimum(size * p[:boundary], tail_bound)
            adjusted = np.maximum(
                adjusted, np.r_[candidate, np.repeat(candidate[-1], size - 1)]
            )
        adjusted = np.maximum(adjusted, p)
    result = np.empty(n)
    result[order] = np.minimum(1, adjusted)
    return result


def _make_flow_residual_plots(
    od, covariates, covariate_col, map_data, x_col, y_col, label
):
    import matplotlib.pyplot as plt

    if x_col is not None or y_col is not None:
        if x_col is None or y_col is None:
            raise ValueError(
                "Both `x_col` and `y_col` are required for a residual map plot."
            )
        if x_col not in map_data or y_col not in map_data:
            raise ValueError("`x_col` and `y_col` must name columns in `map_data`.")
        _require_real_column(map_data, x_col, "map_data")
        _require_real_column(map_data, y_col, "map_data")
    plots = {}
    fig, ax = plt.subplots()
    values = _floats(od["abs_residual_reduction"])
    ax.hist(values[np.isfinite(values)], bins=30, color="#2B8CBE", edgecolor="white")
    ax.axvline(0, linestyle="--", color="#D95F0E")
    ax.set(
        xlabel="Absolute residual reduction",
        ylabel="OD pairs",
        title="Residual reduction distribution",
    )
    plots["residual_reduction_distribution"] = fig
    scatter_specs = [
        (
            "residual_vs_benchmark",
            od,
            "benchmark_flow",
            "Benchmark OD flow",
            f"{label} residual",
            "Residuals versus benchmark flow",
        )
    ]
    if covariates is not None:
        scatter_specs.append(
            (
                "residual_vs_covariate",
                covariates,
                "covariate_value",
                covariate_col,
                f"{label} area residual",
                "Area residuals versus covariate",
            )
        )
    for key, data, x_name, x_label, y_label, title in scatter_specs:
        fig, ax = plt.subplots()
        keep = _finite_pair_mask(data[x_name], data["selected_residual"])
        ax.scatter(
            _floats(data[x_name])[keep],
            _floats(data["selected_residual"])[keep],
            alpha=0.7,
        )
        ax.axhline(0, linestyle="--", color="#D95F0E")
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        plots[key] = fig
    if x_col is not None:
        fig, ax = plt.subplots()
        x, y, residual = (
            _floats(map_data[key]) for key in (x_col, y_col, "selected_residual")
        )
        keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(residual)
        points = ax.scatter(
            x[keep], y[keep], c=residual[keep], alpha=0.9, cmap="viridis"
        )
        fig.colorbar(points, ax=ax, label="Residual")
        ax.set(
            xlabel=x_col, ylabel=y_col, title="Area-level residual map", aspect="equal"
        )
        plots["residual_map"] = fig
    return plots
