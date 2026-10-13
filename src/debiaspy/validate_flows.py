from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from debiaspy._flow_joins import (
    REPEATED_OBSERVATION_KEYS,
    merge_validation_reference,
    validation_join_keys,
)
from debiaspy._numeric import r_style_simple_ols
from debiaspy._schemas import require_columns, require_scalar_bool
from debiaspy.distribution_metrics import (
    _distribution_jsd_contributions,
    _distribution_kl_contributions,
    _distribution_shares,
)

FlowComparison = Literal["adjusted_vs_benchmark", "raw_vs_benchmark", "raw_vs_adjusted"]


FLOW_COMPARISONS: tuple[FlowComparison, ...] = (
    "adjusted_vs_benchmark",
    "raw_vs_benchmark",
    "raw_vs_adjusted",
)


FLOW_COMPARISON_SPECS = pd.DataFrame(
    {
        "comparison": list(FLOW_COMPARISONS),
        "comparison_label": [
            "Adjusted vs benchmark",
            "Raw MPD vs benchmark",
            "Raw MPD vs adjusted",
        ],
        "x_series": ["adjusted", "raw_mpd", "raw_mpd"],
        "y_series": ["benchmark", "benchmark", "adjusted"],
        "signed_residual": [
            "benchmark - adjusted",
            "benchmark - raw",
            "adjusted - raw",
        ],
    }
)


def normalise_flow_comparisons(
    comparisons: str | Iterable[str],
) -> list[FlowComparison]:
    if isinstance(comparisons, str):
        if comparisons == "all":
            return list(FLOW_COMPARISONS)
        comparison_list = [comparisons]
    else:
        comparison_list = list(comparisons)
        if len(comparison_list) == 1 and comparison_list[0] == "all":
            return list(FLOW_COMPARISONS)

    invalid = [item for item in comparison_list if item not in FLOW_COMPARISONS]
    if invalid:
        raise ValueError(
            "`comparisons` must contain only 'adjusted_vs_benchmark', "
            "'raw_vs_benchmark', 'raw_vs_adjusted', or 'all'."
        )

    out: list[FlowComparison] = []
    for item in comparison_list:
        if item not in out:
            out.append(item)  # type: ignore[arg-type]
    return out


def flow_comparison_spec(comparison: str) -> pd.Series:
    normalised = normalise_flow_comparisons([comparison])
    row = FLOW_COMPARISON_SPECS.loc[
        FLOW_COMPARISON_SPECS["comparison"] == normalised[0]
    ]
    return row.iloc[0]


def flow_comparison_x(data: pd.DataFrame, comparison: str) -> pd.Series:
    if comparison == "adjusted_vs_benchmark":
        return cast(pd.Series, data["adj_flow"])
    if comparison in {"raw_vs_benchmark", "raw_vs_adjusted"}:
        return cast(pd.Series, data["mpd_flow"])
    raise ValueError("Unsupported flow comparison.")


def flow_comparison_y(data: pd.DataFrame, comparison: str) -> pd.Series:
    if comparison in {"adjusted_vs_benchmark", "raw_vs_benchmark"}:
        return cast(pd.Series, data["benchmark_flow"])
    if comparison == "raw_vs_adjusted":
        return cast(pd.Series, data["adj_flow"])
    raise ValueError("Unsupported flow comparison.")


def flow_comparison_residual(data: pd.DataFrame, comparison: str) -> pd.Series:
    return flow_comparison_y(data, comparison) - flow_comparison_x(data, comparison)


def build_flow_audit_table(
    adj_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
    flow_col_bench: str = "flow",
) -> pd.DataFrame:
    require_columns(
        adj_df, ("origin", "destination", flow_col_mpd, flow_col_adj), "adj_df"
    )
    require_columns(
        benchmark_od_df, ("origin", "destination", flow_col_bench), "benchmark_od_df"
    )

    join_keys = validation_join_keys(adj_df, benchmark_od_df)
    adj_key_columns = list(
        dict.fromkeys(
            [
                *[key for key in REPEATED_OBSERVATION_KEYS if key in adj_df.columns],
                "origin",
                "destination",
            ]
        )
    )
    adj = adj_df.loc[:, adj_key_columns].copy()
    adj["mpd_flow"] = adj_df[flow_col_mpd].to_numpy()
    adj["adj_flow"] = adj_df[flow_col_adj].to_numpy()
    bench_columns = list(
        dict.fromkeys(
            [
                *join_keys,
                *[
                    key
                    for key in REPEATED_OBSERVATION_KEYS
                    if key in benchmark_od_df.columns
                ],
                flow_col_bench,
            ]
        )
    )
    bench = benchmark_od_df.loc[:, bench_columns].rename(
        columns={flow_col_bench: "benchmark_flow"}
    )

    out = merge_validation_reference(
        adj,
        bench,
        how="inner",
        data_name="adj_df",
        reference_name="benchmark_od_df",
    )
    out["diff_mpd_benchmark"] = out["mpd_flow"] - out["benchmark_flow"]
    out["diff_mpd_adj"] = out["mpd_flow"] - out["adj_flow"]
    out["diff_adj_benchmark"] = out["adj_flow"] - out["benchmark_flow"]
    out["signed_residual_adjusted_vs_benchmark"] = (
        out["benchmark_flow"] - out["adj_flow"]
    )
    out["signed_residual_raw_vs_benchmark"] = out["benchmark_flow"] - out["mpd_flow"]
    out["signed_residual_raw_vs_adjusted"] = out["adj_flow"] - out["mpd_flow"]

    base_columns = [
        "origin",
        "destination",
        "mpd_flow",
        "adj_flow",
        "benchmark_flow",
        "diff_mpd_benchmark",
        "diff_mpd_adj",
        "diff_adj_benchmark",
        "signed_residual_adjusted_vs_benchmark",
        "signed_residual_raw_vs_benchmark",
        "signed_residual_raw_vs_adjusted",
    ]
    for key in reversed(REPEATED_OBSERVATION_KEYS):
        if key in out.columns:
            base_columns.insert(0, key)
    return out.loc[:, base_columns].reset_index(drop=True)


def validate_flow_pairs(
    adj_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
    flow_col_bench: str = "flow",
) -> pd.DataFrame:
    """Join adjusted and benchmark OD flows for pair-level validation.

    ``adj_df`` requires ``origin``, ``destination``, and the selected raw and
    adjusted flow columns; ``benchmark_od_df`` requires the OD keys and selected
    benchmark flow. Shared ``mpd_source``/``mpd_time`` columns are included in
    the observation key when present.

    Returns the inner-joined flows with raw, adjusted, benchmark, difference,
    and signed-residual columns. Missing keys or ambiguous repeated-observation
    joins raise. Parity is scoped to the audited pair-table schemas and values.
    """
    return build_flow_audit_table(
        adj_df=adj_df,
        benchmark_od_df=benchmark_od_df,
        flow_col_mpd=flow_col_mpd,
        flow_col_adj=flow_col_adj,
        flow_col_bench=flow_col_bench,
    )


DISTRIBUTION_SUMMARY_COLUMNS = [
    "method",
    "comparison",
    "comparison_label",
    "x_distribution",
    "y_distribution",
    "reference_distribution",
    "comparison_distribution",
    "n_origins",
    "n_origins_used",
    "weight_by",
    "kl_mean",
    "kl_median",
    "kl_weighted_mean",
    "jsd_mean",
    "jsd_median",
    "jsd_weighted_mean",
]


ORIGIN_LEVEL_COLUMNS = [
    "method",
    "comparison",
    "comparison_label",
    "x_distribution",
    "y_distribution",
    "reference_distribution",
    "comparison_distribution",
    "origin",
    "n_destinations",
    "bench_origin_total",
    "raw_origin_total",
    "adjusted_origin_total",
    "adj_origin_total",
    "reference_origin_total",
    "comparison_origin_total",
    "zero_benchmark_total",
    "zero_raw_total",
    "zero_adjusted_total",
    "zero_adj_total",
    "zero_reference_total",
    "zero_comparison_total",
    "kl_origin",
    "jsd_origin",
]


OD_LEVEL_COLUMNS = [
    "method",
    "comparison",
    "comparison_label",
    "x_distribution",
    "y_distribution",
    "reference_distribution",
    "comparison_distribution",
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
    "bench_origin_total",
    "raw_origin_total",
    "adjusted_origin_total",
    "adj_origin_total",
    "reference_origin_total",
    "comparison_origin_total",
    "zero_benchmark_total",
    "zero_raw_total",
    "zero_adjusted_total",
    "zero_adj_total",
    "zero_reference_total",
    "zero_comparison_total",
]


def validate_flow_distribution(
    adj_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame,
    flow_col_adj: str = "flow_adj",
    flow_col_mpd: str = "flow",
    flow_col_bench: str = "flow",
    epsilon: float = 1e-8,
    method_name: str | None = None,
    weight_by: str = "none",
    comparisons: str | list[str] = "adjusted_vs_benchmark",
    return_origin_level: bool = True,
    return_od_level: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compare origin-specific destination-flow distributions.

    ``adj_df`` requires OD keys and ``flow_col_adj``; raw comparisons also
    require ``flow_col_mpd``. The benchmark requires matching OD keys and
    ``flow_col_bench``. Shared source/time fields extend the observation key.
    ``epsilon`` smooths shares, while comparison, weighting, and detail options
    control the reported divergence tables.

    Returns a ``summary`` data frame and optional ``origin_level`` and
    ``od_level`` tables containing KL and Jensen-Shannon diagnostics. Invalid
    smoothing, columns, comparisons, or joins raise. Parity is scoped to the
    audited distribution tables and numeric tolerances.
    """
    weight_by = _match_weight_by(weight_by)
    return_origin_level = require_scalar_bool(
        return_origin_level, "return_origin_level"
    )
    return_od_level = require_scalar_bool(return_od_level, "return_od_level")
    comparison_ids = normalise_flow_comparisons(comparisons)
    _validate_distribution_epsilon(epsilon)

    raw_requested = any(
        comparison in {"raw_vs_benchmark", "raw_vs_adjusted"}
        for comparison in comparison_ids
    )
    req_adj = ["origin", "destination", flow_col_adj]
    if raw_requested:
        req_adj.append(flow_col_mpd)
    require_columns(adj_df, req_adj, "adj_df")
    require_columns(
        benchmark_od_df, ("origin", "destination", flow_col_bench), "benchmark_od_df"
    )

    join_keys = validation_join_keys(adj_df, benchmark_od_df)
    # Repeated-observation columns participate in public detail tables only
    # when they are shared join keys.  A constant source label carried solely
    # by the adjusted frame does not change OD identity and debiasR drops it.
    adj_key_columns = join_keys
    adj_tbl = adj_df.loc[:, adj_key_columns].copy()
    adj_tbl["adjusted_flow"] = adj_df[flow_col_adj].to_numpy()
    if flow_col_mpd in adj_df.columns:
        adj_tbl["raw_flow"] = adj_df[flow_col_mpd].to_numpy()
    else:
        adj_tbl["raw_flow"] = np.nan

    bench_columns = list(
        dict.fromkeys(
            [
                *join_keys,
                *[
                    key
                    for key in REPEATED_OBSERVATION_KEYS
                    if key in benchmark_od_df.columns
                ],
                flow_col_bench,
            ]
        )
    )
    bench_tbl = benchmark_od_df.loc[:, bench_columns].rename(
        columns={flow_col_bench: "benchmark_flow"}
    )
    support_tbl = merge_validation_reference(
        adj_tbl,
        bench_tbl,
        how="outer",
        data_name="adj_df",
        reference_name="benchmark_od_df",
    )
    support_tbl[["benchmark_flow", "adjusted_flow", "raw_flow"]] = support_tbl[
        ["benchmark_flow", "adjusted_flow", "raw_flow"]
    ].fillna(0)

    comparison_outputs = [
        _compute_flow_distribution_comparison(
            comparison=comparison,
            support_tbl=support_tbl,
            epsilon=epsilon,
            method_name=method_name,
        )
        for comparison in comparison_ids
    ]
    origin_level = pd.concat(
        [item["origin_level"] for item in comparison_outputs], ignore_index=True
    )

    summary_rows = []
    method = np.nan if method_name is None else method_name
    for comparison in comparison_ids:
        subset = origin_level.loc[origin_level["comparison"] == comparison]
        first = subset.iloc[0]
        summary_rows.append(
            {
                "method": method,
                "comparison": comparison,
                "comparison_label": first["comparison_label"],
                "x_distribution": first["x_distribution"],
                "y_distribution": first["y_distribution"],
                "reference_distribution": first["reference_distribution"],
                "comparison_distribution": first["comparison_distribution"],
                "n_origins": len(subset),
                "n_origins_used": int(subset["kl_origin"].notna().sum()),
                "weight_by": weight_by,
                "kl_mean": subset["kl_origin"].mean(skipna=True),
                "kl_median": subset["kl_origin"].median(skipna=True),
                "kl_weighted_mean": (
                    _weighted_mean_na(subset["kl_origin"], subset["bench_origin_total"])
                    if weight_by == "benchmark_origin_total"
                    else np.nan
                ),
                "jsd_mean": subset["jsd_origin"].mean(skipna=True),
                "jsd_median": subset["jsd_origin"].median(skipna=True),
                "jsd_weighted_mean": (
                    _weighted_mean_na(
                        subset["jsd_origin"], subset["bench_origin_total"]
                    )
                    if weight_by == "benchmark_origin_total"
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(summary_rows, columns=DISTRIBUTION_SUMMARY_COLUMNS)

    out = {"summary": summary}
    if return_origin_level:
        origin_columns = _columns_with_observation_keys(
            ORIGIN_LEVEL_COLUMNS, origin_level
        )
        out["origin_level"] = origin_level.loc[:, origin_columns].reset_index(drop=True)
    if return_od_level:
        od_level = pd.concat(
            [item["od_level"] for item in comparison_outputs], ignore_index=True
        )
        od_columns = _columns_with_observation_keys(OD_LEVEL_COLUMNS, od_level)
        out["od_level"] = od_level.loc[:, od_columns].reset_index(drop=True)
    return out


def _compute_flow_distribution_comparison(
    comparison: str,
    support_tbl: pd.DataFrame,
    epsilon: float,
    method_name: str | None,
) -> dict[str, pd.DataFrame]:
    if comparison == "raw_vs_benchmark":
        reference_col = "benchmark_flow"
        comparison_col = "raw_flow"
        reference_distribution = "benchmark"
        comparison_distribution = "raw_mpd"
    elif comparison == "adjusted_vs_benchmark":
        reference_col = "benchmark_flow"
        comparison_col = "adjusted_flow"
        reference_distribution = "benchmark"
        comparison_distribution = "adjusted_mpd"
    elif comparison == "raw_vs_adjusted":
        reference_col = "adjusted_flow"
        comparison_col = "raw_flow"
        reference_distribution = "adjusted_mpd"
        comparison_distribution = "raw_mpd"
    else:
        raise ValueError("Unsupported distribution comparison.")

    spec = flow_comparison_spec(comparison)
    od_level = support_tbl.copy()
    od_level["reference_flow"] = od_level[reference_col]
    od_level["comparison_flow"] = od_level[comparison_col]

    observation_keys = [
        key for key in REPEATED_OBSERVATION_KEYS if key in od_level.columns
    ]
    origin_group_keys = [*observation_keys, "origin"]
    origin_groups = od_level.groupby(
        origin_group_keys, sort=True, dropna=False, observed=True
    )
    for flow_column, total_column in [
        ("benchmark_flow", "bench_origin_total"),
        ("raw_flow", "raw_origin_total"),
        ("adjusted_flow", "adjusted_origin_total"),
        ("reference_flow", "reference_origin_total"),
        ("comparison_flow", "comparison_origin_total"),
    ]:
        od_level[total_column] = origin_groups[flow_column].transform("sum")
    od_level["adj_origin_total"] = od_level["adjusted_origin_total"]

    reference_smoothed = od_level["reference_flow"] + epsilon
    comparison_smoothed = od_level["comparison_flow"] + epsilon
    reference_smoothed_total = reference_smoothed.groupby(
        [od_level[key] for key in origin_group_keys],
        sort=True,
        dropna=False,
        observed=True,
    ).transform("sum")
    comparison_smoothed_total = comparison_smoothed.groupby(
        [od_level[key] for key in origin_group_keys],
        sort=True,
        dropna=False,
        observed=True,
    ).transform("sum")
    od_level["reference_share"] = _distribution_shares(
        od_level["reference_flow"], epsilon, reference_smoothed_total
    )
    od_level["comparison_share"] = _distribution_shares(
        od_level["comparison_flow"], epsilon, comparison_smoothed_total
    )
    od_level["midpoint_share"] = 0.5 * (
        od_level["reference_share"] + od_level["comparison_share"]
    )
    od_level["kl_contribution"] = _distribution_kl_contributions(
        od_level["reference_share"], od_level["comparison_share"]
    )
    od_level["jsd_contribution"] = _distribution_jsd_contributions(
        od_level["reference_share"],
        od_level["comparison_share"],
    )
    od_level["_origin_group_order"] = origin_groups.ngroup()
    od_level = (
        od_level.sort_values("_origin_group_order", kind="stable")
        .drop(columns="_origin_group_order")
        .reset_index(drop=True)
    )
    method = np.nan if method_name is None else method_name
    od_level["method"] = method
    od_level["comparison"] = comparison
    od_level["comparison_label"] = spec["comparison_label"]
    od_level["x_distribution"] = spec["x_series"]
    od_level["y_distribution"] = spec["y_series"]
    od_level["reference_distribution"] = reference_distribution
    od_level["comparison_distribution"] = comparison_distribution
    od_level["zero_benchmark_total"] = od_level["bench_origin_total"] <= 0
    od_level["zero_raw_total"] = od_level["raw_origin_total"] <= 0
    od_level["zero_adjusted_total"] = od_level["adjusted_origin_total"] <= 0
    od_level["zero_adj_total"] = od_level["zero_adjusted_total"]
    od_level["zero_reference_total"] = od_level["reference_origin_total"] <= 0
    od_level["zero_comparison_total"] = od_level["comparison_origin_total"] <= 0

    origin_rows = []
    group_cols = [
        "method",
        "comparison",
        "comparison_label",
        "x_distribution",
        "y_distribution",
        "reference_distribution",
        "comparison_distribution",
        *observation_keys,
        "origin",
    ]
    for keys, group in od_level.groupby(group_cols, sort=True, dropna=False):
        first = group.iloc[0]
        zero_reference_total = bool(first["zero_reference_total"])
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, object] = dict(zip(group_cols, key_values, strict=True))
        row.update(
            {
                "n_destinations": len(group),
                "bench_origin_total": first["bench_origin_total"],
                "raw_origin_total": first["raw_origin_total"],
                "adjusted_origin_total": first["adjusted_origin_total"],
                "adj_origin_total": first["adj_origin_total"],
                "reference_origin_total": first["reference_origin_total"],
                "comparison_origin_total": first["comparison_origin_total"],
                "zero_benchmark_total": first["zero_benchmark_total"],
                "zero_raw_total": first["zero_raw_total"],
                "zero_adjusted_total": first["zero_adjusted_total"],
                "zero_adj_total": first["zero_adj_total"],
                "zero_reference_total": first["zero_reference_total"],
                "zero_comparison_total": first["zero_comparison_total"],
                "kl_origin": (
                    np.nan
                    if zero_reference_total
                    else _sum_propagating_missing(
                        cast(pd.Series, group["kl_contribution"])
                    )
                ),
                "jsd_origin": (
                    np.nan
                    if zero_reference_total
                    else _sum_propagating_missing(
                        cast(pd.Series, group["jsd_contribution"])
                    )
                ),
            }
        )
        origin_rows.append(row)

    origin_level = pd.DataFrame(
        origin_rows,
        columns=_columns_with_observation_keys(ORIGIN_LEVEL_COLUMNS, od_level),
    )
    return {"origin_level": origin_level, "od_level": od_level}


def _columns_with_observation_keys(columns: list[str], data: pd.DataFrame) -> list[str]:
    out = list(columns)
    insertion_index = out.index("origin")
    for key in REPEATED_OBSERVATION_KEYS:
        if key in data.columns:
            out.insert(insertion_index, key)
            insertion_index += 1
    return out


def _validate_distribution_epsilon(epsilon: float) -> None:
    try:
        numeric = float(epsilon)
    except (TypeError, ValueError):
        raise ValueError("`epsilon` must be a single positive finite number.") from None
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError("`epsilon` must be a single positive finite number.")


def _weighted_mean_na(x: pd.Series, w: pd.Series) -> float:
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not bool(keep.any()):
        return np.nan
    return float(np.average(x[keep], weights=w[keep]))


def _sum_propagating_missing(x: pd.Series) -> float:
    """Match R's ``sum(x)`` rather than pandas' missing-skipping default."""
    if bool(x.isna().any()):
        return np.nan
    return float(x.sum(skipna=False))


def _match_weight_by(weight_by: str) -> str:
    choices = ("none", "benchmark_origin_total")
    matches = [choice for choice in choices if choice.startswith(weight_by)]
    if len(matches) != 1:
        raise ValueError("`weight_by` must be one of: none, benchmark_origin_total")
    return matches[0]


METRIC_COLUMNS = [
    "method",
    "comparison",
    "comparison_label",
    "x_series",
    "y_series",
    "signed_residual",
    "n",
    "sum_x",
    "sum_y",
    "sum_adj",
    "sum_bench",
    "sum_mpd",
    "mean_error",
    "pearson_r",
    "spearman_rho",
    "rmse",
    "mae",
    "mape",
    "ols_intercept",
    "ols_slope",
    "r_squared",
]


DATA_COLUMNS = [
    "data.origin",
    "data.destination",
    "data.flow_adj",
    "data.mpd_flow",
    "data.flow_bench",
    "data.adj_flow",
    "data.benchmark_flow",
    "data.x_flow",
    "data.y_flow",
    "data.residual",
    "data.comparison",
    "data.comparison_label",
    "data.x_series",
    "data.y_series",
    "data.signed_residual",
]


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
    comparisons: str | list[str] = "adjusted_vs_benchmark",
    flow_col_mpd: str = "flow",
) -> dict[str, Any]:
    """Compute aggregate accuracy metrics for selected flow comparisons.

    ``adj_df`` requires OD keys and ``flow_col_adj``; comparisons involving raw
    MPD also require ``flow_col_mpd``. ``benchmark_od_df`` requires OD keys and
    ``flow_col_bench``. Shared source/time fields form part of the join key.
    Options control comparison selection, zero/NA filtering, source summaries,
    and retention of joined rows.

    Returns an R-shaped metric dictionary; multi-comparison results include a
    ``summary`` table, while ``by_source`` and ``data`` are optional tables.
    Missing columns, invalid comparisons, or unsafe repeated-observation joins
    raise. Parity is scoped to audited metric tables and numeric tolerances.
    """
    drop_zeros = require_scalar_bool(drop_zeros, "drop_zeros")
    na_rm = require_scalar_bool(na_rm, "na_rm")
    by_source = require_scalar_bool(by_source, "by_source")
    return_joined = require_scalar_bool(return_joined, "return_joined")
    comparison_ids = normalise_flow_comparisons(comparisons)
    raw_requested = any(
        comparison in {"raw_vs_benchmark", "raw_vs_adjusted"}
        for comparison in comparison_ids
    )

    req_adj = ["origin", "destination", flow_col_adj]
    if raw_requested:
        req_adj.append(flow_col_mpd)
    require_columns(adj_df, req_adj, "adj_df")
    require_columns(
        benchmark_od_df, ("origin", "destination", flow_col_bench), "benchmark_od_df"
    )

    join_keys = validation_join_keys(adj_df, benchmark_od_df)
    adj_key_columns = [
        *[key for key in REPEATED_OBSERVATION_KEYS if key in adj_df.columns],
        "origin",
        "destination",
    ]
    joined = adj_df.loc[:, adj_key_columns].copy()
    joined["flow_adj"] = adj_df[flow_col_adj].to_numpy()
    if flow_col_mpd in adj_df.columns:
        joined["mpd_flow"] = adj_df[flow_col_mpd].to_numpy()
    else:
        joined["mpd_flow"] = np.nan

    bench_columns = list(
        dict.fromkeys(
            [
                *join_keys,
                *[
                    key
                    for key in REPEATED_OBSERVATION_KEYS
                    if key in benchmark_od_df.columns
                ],
                flow_col_bench,
            ]
        )
    )
    bench = benchmark_od_df.loc[:, bench_columns].rename(
        columns={flow_col_bench: "flow_bench"}
    )
    joined = merge_validation_reference(
        joined,
        bench,
        how="inner",
        data_name="adj_df",
        reference_name="benchmark_od_df",
    )
    joined["adj_flow"] = joined["flow_adj"]
    joined["benchmark_flow"] = joined["flow_bench"]

    comparison_frames = []
    for comparison in comparison_ids:
        spec = flow_comparison_spec(comparison)
        frame = joined.copy()
        frame["x_flow"] = flow_comparison_x(frame, comparison)
        frame["y_flow"] = flow_comparison_y(frame, comparison)
        frame["residual"] = flow_comparison_residual(frame, comparison)
        frame["comparison"] = comparison
        frame["comparison_label"] = spec["comparison_label"]
        frame["x_series"] = spec["x_series"]
        frame["y_series"] = spec["y_series"]
        frame["signed_residual"] = spec["signed_residual"]
        comparison_frames.append(frame)
    comparison_data = pd.concat(comparison_frames, ignore_index=True)

    if na_rm:
        comparison_data = comparison_data.loc[
            np.isfinite(comparison_data["x_flow"])
            & np.isfinite(comparison_data["y_flow"])
        ].reset_index(drop=True)
    if drop_zeros:
        comparison_data = comparison_data.loc[
            (comparison_data["x_flow"] > 0) & (comparison_data["y_flow"] > 0)
        ].reset_index(drop=True)

    metric_rows = [
        _flow_overall_metric_row(
            comparison_data.loc[comparison_data["comparison"] == comparison],
            comparison,
            method_name,
        )
        for comparison in comparison_ids
    ]
    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS).reset_index(drop=True)

    out: dict[str, Any]
    if len(comparison_ids) == 1:
        out = metrics.iloc[0].to_dict()
    else:
        out = {
            "method": np.nan if method_name is None else method_name,
            "summary": metrics,
        }

    if by_source and "mpd_source" in comparison_data.columns:
        source_rows = []
        group_cols = ["comparison", "mpd_source"]
        for (comparison, source), data in comparison_data.groupby(
            group_cols, sort=True, dropna=True, observed=True
        ):
            row = _flow_overall_metric_row(data, str(comparison), method_name)
            row["mpd_source"] = source
            source_rows.append(row)
        out["by_source"] = (
            pd.DataFrame(source_rows, columns=[*METRIC_COLUMNS, "mpd_source"])
            if source_rows
            else pd.DataFrame()
        )

    if return_joined:
        out["data"] = comparison_data.reset_index(drop=True)
    return out


def flatten_validate_flow_overall_result(result: dict[str, Any]) -> pd.DataFrame:
    """Serialize an R-like overall result to the legacy audit CSV shape."""
    if "summary" in result and isinstance(result["summary"], pd.DataFrame):
        metrics = result["summary"].reset_index(drop=True)
    else:
        scalar = {
            key: value
            for key, value in result.items()
            if key in METRIC_COLUMNS and not isinstance(value, pd.DataFrame)
        }
        metrics = pd.DataFrame([scalar], columns=METRIC_COLUMNS)

    data_obj = result.get("data")
    if not isinstance(data_obj, pd.DataFrame):
        return metrics.loc[:, METRIC_COLUMNS].reset_index(drop=True)

    flattened = []
    for _, metric_row in metrics.iterrows():
        metric = metric_row.to_dict()
        comparison = metric.get("comparison")
        data = data_obj.loc[data_obj["comparison"] == comparison].reset_index(drop=True)
        if data.empty:
            flattened.append(pd.DataFrame([metric]))
            continue

        rows = pd.DataFrame([metric] * len(data))
        data_prefixed = data.loc[
            :,
            [
                "origin",
                "destination",
                "flow_adj",
                "mpd_flow",
                "flow_bench",
                "adj_flow",
                "benchmark_flow",
                "x_flow",
                "y_flow",
                "residual",
                "comparison",
                "comparison_label",
                "x_series",
                "y_series",
                "signed_residual",
            ],
        ].add_prefix("data.")
        flattened.append(pd.concat([rows, data_prefixed], axis=1))

    out = pd.concat(flattened, ignore_index=True)
    for col in DATA_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out.loc[:, [*METRIC_COLUMNS, *DATA_COLUMNS]]


def _flow_overall_metric_row(
    data: pd.DataFrame, comparison: str, method_name: str | None
) -> dict[str, object]:
    spec = flow_comparison_spec(comparison)
    method = np.nan if method_name is None else method_name
    row: dict[str, object] = {
        "method": method,
        "comparison": comparison,
        "comparison_label": spec["comparison_label"],
        "x_series": spec["x_series"],
        "y_series": spec["y_series"],
        "signed_residual": spec["signed_residual"],
        "n": 0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_adj": 0.0,
        "sum_bench": 0.0,
        "sum_mpd": 0.0,
        "mean_error": np.nan,
        "pearson_r": np.nan,
        "spearman_rho": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "mape": np.nan,
        "ols_intercept": np.nan,
        "ols_slope": np.nan,
        "r_squared": np.nan,
    }
    if data.empty:
        return row

    x = _as_float_series(data, "x_flow")
    y = _as_float_series(data, "y_flow")
    residual = y - x
    x_sd = x.std(ddof=1)
    y_sd = y.std(ddof=1)
    can_cor = (
        len(x) >= 2
        and np.isfinite(x_sd)
        and x_sd > 0
        and np.isfinite(y_sd)
        and y_sd > 0
    )
    can_lm = can_cor and bool(np.isfinite(x + y).all())

    if can_lm:
        ols_intercept, ols_slope, _, r_squared = r_style_simple_ols(
            x.to_numpy(), y.to_numpy()
        )
    else:
        ols_intercept = np.nan
        ols_slope = np.nan
        r_squared = np.nan

    denom = y.where(y != 0, np.nan)
    row.update(
        {
            "n": len(x),
            "sum_x": x.sum(skipna=True),
            "sum_y": y.sum(skipna=True),
            "sum_adj": _as_float_series(data, "flow_adj").sum(skipna=True),
            "sum_bench": _as_float_series(data, "flow_bench").sum(skipna=True),
            "sum_mpd": _as_float_series(data, "mpd_flow").sum(skipna=True),
            "mean_error": residual.mean(skipna=True),
            "pearson_r": _pearson(x, y) if can_cor else np.nan,
            "spearman_rho": _spearman(x, y) if can_cor else np.nan,
            "rmse": float(np.sqrt((residual**2).mean(skipna=False))),
            "mae": residual.abs().mean(skipna=False),
            "mape": (residual / denom).abs().mean(skipna=True),
            "ols_intercept": ols_intercept,
            "ols_slope": ols_slope,
            "r_squared": r_squared,
        }
    )
    return row


def _as_float_series(data: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, data[column]).astype(float)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    return float(np.corrcoef(x.to_numpy(), y.to_numpy())[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")
    return _pearson(x_rank, y_rank)
