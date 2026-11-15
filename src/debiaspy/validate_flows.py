"""Overall, pair-level and distribution diagnostics for origin-destination flows."""

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from debiaspy.distribution_metrics import (
    _distribution_jsd_contributions,
    _distribution_kl_contributions,
    _distribution_shares,
    _validate_distribution_epsilon,
)
from debiaspy.flow_comparisons import (
    _SERIES_COLUMNS,
    _flow_comparison_spec,
    _normalise_flow_comparisons,
)
from debiaspy.measure_bias import (
    _require_columns,
    _require_dataframe,
    _require_numeric_columns,
    _require_string_column_names,
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
    joined = _join_flows(left, right)
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
    out = _join_flows(left, right)
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


def _join_flows(
    left: pd.DataFrame, right: pd.DataFrame, *, how: str = "inner"
) -> pd.DataFrame:
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
    if how == "inner":
        return left.merge(right, on=keys, how="inner", sort=False)
    # pandas outer joins sort keys; restore the reference's left-first order.
    joined = left.assign(_left_order=np.arange(len(left))).merge(
        right.assign(_right_order=np.arange(len(right))),
        on=keys,
        how="outer",
        sort=False,
    )
    return (
        joined.sort_values(["_left_order", "_right_order"], kind="stable")
        .drop(columns=["_left_order", "_right_order"])
        .reset_index(drop=True)
    )


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


def validate_flow_distribution(
    adj_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame,
    flow_col_adj: str = "flow_adj",
    flow_col_mpd: str = "flow",
    flow_col_bench: str = "flow",
    epsilon: float = 1e-8,
    method_name: str | None = None,
    weight_by: str | None = "none",
    comparisons: str | Sequence[str] = "adjusted_vs_benchmark",
    return_origin_level: bool = True,
    return_od_level: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compare destination-share distributions separately for each origin.

    Parameters
    ----------
    adj_df, benchmark_od_df:
        DataFrames containing ``origin``, ``destination`` and the requested
        flow columns. Raw flows are required only for raw comparisons.
    flow_col_adj, flow_col_mpd, flow_col_bench:
        Names of the adjusted, raw MPD and benchmark flow columns.
    epsilon:
        Positive finite smoothing constant added to each flow before shares
        are calculated on the union of observed OD pairs.
    method_name:
        Optional method label, included in every output table.
    weight_by:
        ``"none"`` or ``"benchmark_origin_total"``. Unambiguous prefixes are
        accepted; None selects ``"none"``. Weighted means use benchmark totals,
        including for ``raw_vs_adjusted``. Only finite metrics and positive
        finite weights contribute. Unweighted means and medians are always returned.
    comparisons:
        A comparison ID, ordered sequence, or ``"all"``. Supported IDs are
        ``adjusted_vs_benchmark``, ``raw_vs_benchmark`` and ``raw_vs_adjusted``.
        The first series is x/comparison, the second y/reference.
    return_origin_level, return_od_level:
        Include origin-level diagnostics (default True) and OD-level shares
        and contributions (default False).

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``summary`` contains one row per comparison. Optional ``origin_level``
        contains origin totals, zero-total flags, KL and JSD; ``od_level``
        contains flows, shares and individual divergence contributions.

    Notes
    -----
    KL is directional: KL(y || x). JSD is symmetric; both use natural logs.
    A full OD join retains unmatched pairs and matches missing keys. Missing
    flows become zero before smoothing. Sources are ignored and duplicate
    keys expand the join (many-to-many joins warn); rows are not aggregated.
    OD rows follow benchmark order, then unmatched adjusted rows. Origin rows
    follow sorted origin/group order within each requested comparison.

    Origins with unsmoothed reference totals <= 0 have missing divergences and
    are excluded from summaries, but their OD contributions are retained.
    Negative flows are not clipped: nonpositive/non-finite smoothed totals
    raise ValueError; negative logarithm arguments warn and produce NaN.
    Empty union support raises ValueError. Inputs are not modified.
    """
    if weight_by is None:
        weight_by = "none"
    choices = ("none", "benchmark_origin_total")
    matches = [
        choice
        for choice in choices
        if isinstance(weight_by, str) and weight_by and choice.startswith(weight_by)
    ]
    if len(matches) != 1:
        raise ValueError("`weight_by` must be 'none' or 'benchmark_origin_total'.")
    weight_by = matches[0]
    selected = _normalise_flow_comparisons(comparisons)
    _validate_distribution_epsilon(epsilon)
    for name, value in (
        ("return_origin_level", return_origin_level),
        ("return_od_level", return_od_level),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"`{name}` must be a boolean.")
    if method_name is not None and not isinstance(method_name, str):
        raise TypeError("`method_name` must be a string or None.")
    adjusted, benchmark = _prepare_flow_inputs(
        adj_df,
        benchmark_od_df,
        flow_col_adj,
        flow_col_mpd,
        flow_col_bench,
        any(value != "adjusted_vs_benchmark" for value in selected),
        overall=True,
    )
    adjusted = adjusted.drop(columns="mpd_source", errors="ignore").rename(
        columns={"flow_adj": "adjusted_flow", "mpd_flow": "raw_flow"}
    )
    benchmark = benchmark.rename(columns={"flow_bench": "benchmark_flow"})
    support = _join_flows(benchmark, adjusted, how="outer")
    if support.empty:
        raise ValueError("Cannot compute distribution shares from empty OD support.")
    for column in ("benchmark_flow", "raw_flow", "adjusted_flow"):
        support[column] = _floats(support[column].fillna(0))
    origins, od_tables, summaries = [], [], []
    for comparison in selected:
        origin, od = _compute_flow_distribution_comparison(
            comparison, support, epsilon, method_name
        )
        origins.append(origin)
        od_tables.append(od)
        metadata = _flow_distribution_metadata(comparison, method_name)
        row = {
            **metadata,
            "n_origins": len(origin),
            "n_origins_used": int(origin["kl_origin"].notna().sum()),
            "weight_by": weight_by,
        }
        for metric in ("kl", "jsd"):
            values = _floats(origin[f"{metric}_origin"])
            present = values[~np.isnan(values)]
            row[f"{metric}_mean"] = float(present.mean()) if len(present) else np.nan
            row[f"{metric}_median"] = (
                float(np.median(present)) if len(present) else np.nan
            )
            row[f"{metric}_weighted_mean"] = (
                _weighted_mean_na(values, _floats(origin["bench_origin_total"]))
                if weight_by == "benchmark_origin_total"
                else np.nan
            )
        summaries.append(row)
    out = {"summary": pd.DataFrame(summaries)}
    if return_origin_level:
        out["origin_level"] = pd.concat(origins, ignore_index=True)
    if return_od_level:
        out["od_level"] = pd.concat(od_tables, ignore_index=True)
    return out


def _flow_distribution_metadata(comparison, method_name):
    spec = _flow_comparison_spec(comparison)
    labels = {
        "raw_mpd": "raw_mpd",
        "adjusted": "adjusted_mpd",
        "benchmark": "benchmark",
    }
    return {
        "method": method_name,
        "comparison": comparison,
        "comparison_label": spec["comparison_label"],
        "x_distribution": spec["x_series"],
        "y_distribution": spec["y_series"],
        "reference_distribution": labels[spec["y_series"]],
        "comparison_distribution": labels[spec["x_series"]],
    }


def _compute_flow_distribution_comparison(comparison, support, epsilon, method_name):
    metadata = _flow_distribution_metadata(comparison, method_name)
    columns = {
        "raw_mpd": "raw_flow",
        "adjusted": "adjusted_flow",
        "benchmark": "benchmark_flow",
    }
    origin_rows, od_groups = [], []
    for origin, group in support.groupby(
        "origin", sort=True, observed=True, dropna=False
    ):
        data = group.copy()
        data["reference_flow"] = data[columns[metadata["y_distribution"]]]
        data["comparison_flow"] = data[columns[metadata["x_distribution"]]]
        with np.errstate(invalid="ignore", over="ignore"):
            totals = {
                "bench_origin_total": float(data["benchmark_flow"].sum()),
                "raw_origin_total": float(data["raw_flow"].sum()),
                "adjusted_origin_total": float(data["adjusted_flow"].sum()),
                "adj_origin_total": float(data["adjusted_flow"].sum()),
                "reference_origin_total": float(data["reference_flow"].sum()),
                "comparison_origin_total": float(data["comparison_flow"].sum()),
            }
        flags = {
            f"zero_{name}_total": np.nan
            if np.isnan(totals[total])
            else totals[total] <= 0
            for name, total in zip(
                ("benchmark", "raw", "adjusted", "adj", "reference", "comparison"),
                totals,
                strict=True,
            )
        }
        try:
            reference_share = _distribution_shares(
                _floats(data["reference_flow"]), epsilon
            )
            comparison_share = _distribution_shares(
                _floats(data["comparison_flow"]), epsilon
            )
        except ValueError as error:
            raise ValueError(f"{comparison}, origin {origin!r}: {error}") from error
        data["reference_share"] = reference_share
        data["comparison_share"] = comparison_share
        data["midpoint_share"] = 0.5 * (reference_share + comparison_share)
        data["kl_contribution"] = _distribution_kl_contributions(
            reference_share, comparison_share
        )
        data["jsd_contribution"] = _distribution_jsd_contributions(
            reference_share, comparison_share
        )
        for name, value in {**metadata, **totals, **flags}.items():
            data[name] = value
        origin_rows.append(
            {
                **metadata,
                "origin": origin,
                "n_destinations": len(data),
                **totals,
                **flags,
                "kl_origin": np.nan
                if flags["zero_reference_total"]
                else float(data["kl_contribution"].sum(skipna=False)),
                "jsd_origin": np.nan
                if flags["zero_reference_total"]
                else float(data["jsd_contribution"].sum(skipna=False)),
            }
        )
        od_groups.append(
            data[
                [
                    *metadata,
                    "origin",
                    "destination",
                    "benchmark_flow",
                    "raw_flow",
                    "adjusted_flow",
                    "reference_flow",
                    "comparison_flow",
                    "reference_share",
                    "comparison_share",
                    "midpoint_share",
                    "kl_contribution",
                    "jsd_contribution",
                    *totals,
                    *flags,
                ]
            ]
        )
    return pd.DataFrame(origin_rows), pd.concat(od_groups).sort_index().reset_index(
        drop=True
    )


def _weighted_mean_na(values: np.ndarray, weights: np.ndarray) -> float:
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return np.nan
    return float(np.sum(values[keep] * weights[keep]) / np.sum(weights[keep]))
