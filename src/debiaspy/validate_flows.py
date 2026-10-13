from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from debiaspy._choices import match_choice
from debiaspy._exceptions import DebiasSchemaError
from debiaspy._flow_joins import (
    REPEATED_OBSERVATION_KEYS,
    merge_validation_reference,
    validation_join_keys,
)
from debiaspy._numeric import global_moran_i
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


SUMMARY_COLUMNS = [
    "method",
    "n",
    "mean_signed_residual_reduction",
    "median_signed_residual_reduction",
    "mean_abs_residual_reduction",
    "median_abs_residual_reduction",
    "mean_abs_residual_mpd_benchmark",
    "mean_abs_residual_adj_benchmark",
    "median_abs_residual_mpd_benchmark",
    "median_abs_residual_adj_benchmark",
    "share_improved",
    "share_worsened",
    "share_unchanged",
    "share_moved_in_benchmark_direction",
    "share_residual_mpd_over_1sd",
    "share_residual_mpd_over_2sd",
    "share_residual_mpd_over_3sd",
    "share_residual_adj_over_1sd",
    "share_residual_adj_over_2sd",
    "share_residual_adj_over_3sd",
    "reduction_share_residual_over_2sd",
]


def validate_flow_residuals(
    adj_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
    flow_col_bench: str = "flow",
    top_n: int | float = 10,
    method_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Summarize OD residual changes from raw to adjusted flows.

    ``adj_df`` requires ``origin``, ``destination``, and the selected raw and
    adjusted flow columns; the benchmark requires matching OD keys and its flow
    column. Shared source/time fields are used as observation keys. ``top_n``
    selects the largest remaining adjusted residuals.

    Returns ``summary``, pair-level ``data``, and ``top_worst`` data frames with
    signed, absolute, percentage, and standard-deviation diagnostics. Invalid
    ``top_n``, missing schemas, or ambiguous joins raise. Parity is scoped to
    the audited residual tables and numeric tolerances.
    """
    if not _is_single_positive_number(top_n):
        raise ValueError("`top_n` must be a single positive integer.")
    top_n = int(top_n)

    audit_tbl = validate_flow_pairs(
        adj_df=adj_df,
        benchmark_od_df=benchmark_od_df,
        flow_col_mpd=flow_col_mpd,
        flow_col_adj=flow_col_adj,
        flow_col_bench=flow_col_bench,
    )

    residual_sd_mpd_benchmark = _sample_sd(
        _residuals_series(audit_tbl, "diff_mpd_benchmark")
    )
    residual_sd_adj_benchmark = _sample_sd(
        _residuals_series(audit_tbl, "diff_adj_benchmark")
    )

    residuals = audit_tbl.copy()
    residuals["method"] = np.nan if method_name is None else method_name
    residuals["residual_mpd_benchmark"] = residuals["signed_residual_raw_vs_benchmark"]
    residuals["residual_adj_benchmark"] = residuals[
        "signed_residual_adjusted_vs_benchmark"
    ]
    residuals["residual_mpd_adj"] = residuals["signed_residual_raw_vs_adjusted"]
    residuals["benchmark_minus_mpd"] = residuals["residual_mpd_benchmark"]
    residuals["benchmark_minus_adj"] = residuals["residual_adj_benchmark"]
    residuals["signed_residual_reduction"] = residuals["residual_mpd_adj"]
    residuals["abs_residual_mpd_benchmark"] = residuals["residual_mpd_benchmark"].abs()
    residuals["abs_residual_adj_benchmark"] = residuals["residual_adj_benchmark"].abs()
    residuals["abs_residual_mpd_adj"] = residuals["residual_mpd_adj"].abs()
    residuals["pct_residual_mpd_benchmark"] = _percentage_unless_zero(
        _residuals_series(residuals, "residual_mpd_benchmark"),
        _residuals_series(residuals, "benchmark_flow"),
    )
    residuals["pct_residual_adj_benchmark"] = _percentage_unless_zero(
        _residuals_series(residuals, "residual_adj_benchmark"),
        _residuals_series(residuals, "benchmark_flow"),
    )
    residuals["abs_residual_improvement"] = (
        residuals["abs_residual_mpd_benchmark"]
        - residuals["abs_residual_adj_benchmark"]
    )
    residuals["abs_residual_reduction"] = residuals["abs_residual_improvement"]
    residuals["pct_residual_improvement"] = _percentage_unless_zero(
        _residuals_series(residuals, "abs_residual_improvement"),
        _residuals_series(residuals, "abs_residual_mpd_benchmark"),
    )
    residuals["residual_sd_mpd_benchmark"] = residual_sd_mpd_benchmark
    residuals["residual_mpd_benchmark_sd_score"] = _sd_score(
        _residuals_series(residuals, "residual_mpd_benchmark"),
        residual_sd_mpd_benchmark,
    )
    residuals["abs_residual_mpd_benchmark_sd_score"] = residuals[
        "residual_mpd_benchmark_sd_score"
    ].abs()
    residuals["residual_sd_adj_benchmark"] = residual_sd_adj_benchmark
    residuals["residual_adj_benchmark_sd_score"] = _sd_score(
        _residuals_series(residuals, "residual_adj_benchmark"),
        residual_sd_adj_benchmark,
    )
    residuals["abs_residual_adj_benchmark_sd_score"] = residuals[
        "residual_adj_benchmark_sd_score"
    ].abs()
    residuals["residual_mpd_over_1sd"] = _residual_over_sd(
        _residuals_series(residuals, "abs_residual_mpd_benchmark_sd_score"), 1
    )
    residuals["residual_mpd_over_2sd"] = _residual_over_sd(
        _residuals_series(residuals, "abs_residual_mpd_benchmark_sd_score"), 2
    )
    residuals["residual_mpd_over_3sd"] = _residual_over_sd(
        _residuals_series(residuals, "abs_residual_mpd_benchmark_sd_score"), 3
    )
    residuals["residual_adj_over_1sd"] = _residual_over_sd(
        _residuals_series(residuals, "abs_residual_adj_benchmark_sd_score"), 1
    )
    residuals["residual_adj_over_2sd"] = _residual_over_sd(
        _residuals_series(residuals, "abs_residual_adj_benchmark_sd_score"), 2
    )
    residuals["residual_adj_over_3sd"] = _residual_over_sd(
        _residuals_series(residuals, "abs_residual_adj_benchmark_sd_score"), 3
    )
    residuals["moved_in_benchmark_direction"] = _moved_in_benchmark_direction(
        _residuals_series(residuals, "benchmark_minus_mpd"),
        _residuals_series(residuals, "signed_residual_reduction"),
    )
    residuals["improvement_flag"] = np.select(
        [
            (residuals["abs_residual_improvement"] > 0)
            .fillna(False)
            .to_numpy(dtype=bool),
            (residuals["abs_residual_improvement"] < 0)
            .fillna(False)
            .to_numpy(dtype=bool),
        ],
        ["improved", "worsened"],
        default="unchanged",
    )

    summary_values = {
        "method": np.nan if method_name is None else method_name,
        "n": len(residuals),
        "mean_signed_residual_reduction": residuals["signed_residual_reduction"].mean(
            skipna=True
        ),
        "median_signed_residual_reduction": residuals[
            "signed_residual_reduction"
        ].median(skipna=True),
        "mean_abs_residual_reduction": residuals["abs_residual_reduction"].mean(
            skipna=True
        ),
        "median_abs_residual_reduction": residuals["abs_residual_reduction"].median(
            skipna=True
        ),
        "mean_abs_residual_mpd_benchmark": residuals["abs_residual_mpd_benchmark"].mean(
            skipna=True
        ),
        "mean_abs_residual_adj_benchmark": residuals["abs_residual_adj_benchmark"].mean(
            skipna=True
        ),
        "median_abs_residual_mpd_benchmark": residuals[
            "abs_residual_mpd_benchmark"
        ].median(skipna=True),
        "median_abs_residual_adj_benchmark": residuals[
            "abs_residual_adj_benchmark"
        ].median(skipna=True),
        "share_improved": _mean_logical_na(
            _residuals_series(residuals, "improvement_flag") == "improved"
        ),
        "share_worsened": _mean_logical_na(
            _residuals_series(residuals, "improvement_flag") == "worsened"
        ),
        "share_unchanged": _mean_logical_na(
            _residuals_series(residuals, "improvement_flag") == "unchanged"
        ),
        "share_moved_in_benchmark_direction": _mean_logical_na(
            _residuals_series(residuals, "moved_in_benchmark_direction")
        ),
        "share_residual_mpd_over_1sd": _mean_logical_na(
            _residuals_series(residuals, "residual_mpd_over_1sd")
        ),
        "share_residual_mpd_over_2sd": _mean_logical_na(
            _residuals_series(residuals, "residual_mpd_over_2sd")
        ),
        "share_residual_mpd_over_3sd": _mean_logical_na(
            _residuals_series(residuals, "residual_mpd_over_3sd")
        ),
        "share_residual_adj_over_1sd": _mean_logical_na(
            _residuals_series(residuals, "residual_adj_over_1sd")
        ),
        "share_residual_adj_over_2sd": _mean_logical_na(
            _residuals_series(residuals, "residual_adj_over_2sd")
        ),
        "share_residual_adj_over_3sd": _mean_logical_na(
            _residuals_series(residuals, "residual_adj_over_3sd")
        ),
    }
    summary_values["reduction_share_residual_over_2sd"] = (
        summary_values["share_residual_mpd_over_2sd"]
        - summary_values["share_residual_adj_over_2sd"]
    )
    summary = pd.DataFrame([summary_values], columns=SUMMARY_COLUMNS)

    top_worst = (
        residuals.sort_values(
            ["abs_residual_adj_benchmark", "abs_residual_mpd_benchmark"],
            ascending=[False, False],
            kind="mergesort",
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return {
        "summary": summary,
        "data": residuals.reset_index(drop=True),
        "top_worst": top_worst,
    }


def _is_single_positive_number(value: object) -> bool:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    # R validates `top_n >= 1` and then calls as.integer(), which truncates
    # positive fractional values toward zero.
    return np.isfinite(numeric) and numeric >= 1


def _residuals_series(data: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, data[column])


def _sample_sd(x: pd.Series) -> float:
    values = x.to_numpy(dtype=float, na_value=np.nan)
    # stats::sd(..., na.rm = TRUE) removes NA/NaN but deliberately retains
    # infinities, which make the resulting standard deviation undefined.
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan
    with np.errstate(invalid="ignore"):
        return float(values.std(ddof=1))


def _sd_score(x: pd.Series, residual_sd: float) -> pd.Series:
    if pd.isna(residual_sd) or residual_sd <= 0:
        return pd.Series(np.nan, index=x.index)
    return x / residual_sd


def _residual_over_sd(abs_score: pd.Series, threshold: float) -> pd.Series:
    return cast(
        pd.Series,
        abs_score.apply(lambda value: np.nan if pd.isna(value) else value > threshold),
    )


def _mean_logical_na(x: pd.Series) -> float:
    non_na = x.dropna()
    if non_na.empty:
        return np.nan
    return float(non_na.astype(bool).mean())


def _moved_in_benchmark_direction(
    benchmark_minus_mpd: pd.Series, signed_residual_reduction: pd.Series
) -> pd.Series:
    benchmark = benchmark_minus_mpd.to_numpy(dtype=float, na_value=np.nan)
    reduction = signed_residual_reduction.to_numpy(dtype=float, na_value=np.nan)
    same_zero = (benchmark == 0) & (reduction == 0)
    either_zero = (benchmark == 0) | (reduction == 0)
    same_direction = np.sign(benchmark) == np.sign(reduction)
    return pd.Series(
        np.select([same_zero, either_zero, same_direction], [True, False, True], False),
        index=benchmark_minus_mpd.index,
    )


def _percentage_unless_zero(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """R-like ``if_else(denominator == 0, NA, 100 * num / denominator)``."""
    numerator_values = numerator.to_numpy(dtype=float, na_value=np.nan)
    denominator_values = denominator.to_numpy(dtype=float, na_value=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 100 * numerator_values / denominator_values
    values[denominator_values == 0] = np.nan
    return pd.Series(values, index=numerator.index, dtype=float)


def validate_flow_residual_structure(
    adj_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
    flow_col_bench: str = "flow",
    method_name: str | None = None,
    residual_type: str = "adjusted",
    comparison: str | None = None,
    spatial_role: str = "origin",
    residual_aggregation: str = "mean",
    area_neighbors: pd.DataFrame | None = None,
    area_col: str = "area",
    neighbor_col: str = "neighbor",
    weight_col: str | None = None,
    local_moran: bool = False,
    local_moran_nsim: int = 999,
    local_moran_alpha: float = 0.05,
    local_moran_p_adjust: str = "BH",
    local_moran_seed: int | float | None = None,
    covariate_df: pd.DataFrame | None = None,
    covariate_col: str | None = None,
    covariate_area_col: str = "area",
    geometry_df: pd.DataFrame | None = None,
    geometry_area_col: str = "area",
    x_col: str | None = None,
    y_col: str | None = None,
    make_plots: bool = False,
) -> dict[str, Any]:
    """Diagnose flow-residual association, spatial structure, and covariates.

    Adjusted and benchmark inputs require matching OD keys and selected flow
    columns. Optional neighbor links require area/neighbor keys; covariate and
    geometry tables require configured area keys. Options select the comparison,
    endpoint aggregation, global/local Moran diagnostics, and optional plots.

    Returns summary, flow-correlation, Moran, covariate, OD, area, and map data
    tables, with optional local-Moran, covariate-data, and plot entries. Invalid
    schemas, options, permutation controls, or plotting coordinates fail closed.
    Parity covers audited tabular diagnostics; plot pixels and seeded
    permutation streams are not claimed to be identical across engines.
    """
    residual_type = match_choice(
        residual_type, ("adjusted", "mpd", "adjustment"), "residual_type"
    )
    if comparison is None:
        comparison = {
            "adjusted": "adjusted_vs_benchmark",
            "mpd": "raw_vs_benchmark",
            "adjustment": "raw_vs_adjusted",
        }[residual_type]
    else:
        comparisons = normalise_flow_comparisons(comparison)
        if len(comparisons) != 1:
            raise ValueError("`comparison` must select exactly one flow comparison.")
        comparison = comparisons[0]
    spatial_role = match_choice(spatial_role, ("origin", "destination"), "spatial_role")
    residual_aggregation = match_choice(
        residual_aggregation, ("mean", "sum"), "residual_aggregation"
    )
    if covariate_col is not None and covariate_df is None:
        raise ValueError("`covariate_df` is required when `covariate_col` is supplied.")
    local_moran = require_scalar_bool(local_moran, "local_moran")
    make_plots = require_scalar_bool(make_plots, "make_plots")
    if local_moran:
        local_moran_seed = _validate_local_moran_args(
            local_moran_nsim,
            local_moran_alpha,
            local_moran_p_adjust,
            local_moran_seed,
        )

    comparison_spec = flow_comparison_spec(comparison)
    residual_type_label = {
        "adjusted_vs_benchmark": "adjusted",
        "raw_vs_benchmark": "mpd",
        "raw_vs_adjusted": "adjustment",
    }[comparison]
    residual_result = validate_flow_residuals(
        adj_df=adj_df,
        benchmark_od_df=benchmark_od_df,
        flow_col_mpd=flow_col_mpd,
        flow_col_adj=flow_col_adj,
        flow_col_bench=flow_col_bench,
        method_name=method_name,
    )

    residual_col = {
        "adjusted_vs_benchmark": "residual_adj_benchmark",
        "raw_vs_benchmark": "residual_mpd_benchmark",
        "raw_vs_adjusted": "residual_mpd_adj",
    }[comparison]
    od_level = residual_result["data"].copy()
    od_level["selected_residual"] = od_level[residual_col]
    od_level["residual_type"] = residual_type_label
    od_level["comparison"] = comparison
    od_level["comparison_label"] = comparison_spec["comparison_label"]
    od_level["x_series"] = comparison_spec["x_series"]
    od_level["y_series"] = comparison_spec["y_series"]
    od_level["signed_residual"] = comparison_spec["signed_residual"]

    finite_flow = np.isfinite(od_level["selected_residual"]) & np.isfinite(
        od_level["benchmark_flow"]
    )
    flow_correlation = pd.DataFrame(
        {
            "method": [_method_value(method_name)],
            "residual_type": [residual_type_label],
            "comparison": [comparison],
            "comparison_label": [comparison_spec["comparison_label"]],
            "benchmark_comparison": ["benchmark_flow"],
            "n": [int(finite_flow.sum())],
            "pearson_r": [
                safe_pearson(od_level["selected_residual"], od_level["benchmark_flow"])
            ],
        }
    )

    area_rows: list[dict[str, Any]] = []
    for area, group in od_level.assign(area=od_level[spatial_role]).groupby(
        "area", sort=True, dropna=False
    ):
        selected = pd.to_numeric(group["selected_residual"], errors="coerce")
        selected_residual = _aggregate_residual(selected, residual_aggregation)
        area_rows.append(
            {
                "area": area,
                "method": _method_value(method_name),
                "residual_type": residual_type_label,
                "comparison": comparison,
                "comparison_label": comparison_spec["comparison_label"],
                "spatial_role": spatial_role,
                "residual_aggregation": residual_aggregation,
                "n_od_pairs": len(group),
                "selected_residual": selected_residual,
                "mean_residual": selected.mean(skipna=True),
                "sum_residual": selected_residual,
                "mean_abs_residual": abs(selected_residual),
                "benchmark_flow_sum": group["benchmark_flow"].sum(skipna=True),
                "mpd_flow_sum": group["mpd_flow"].sum(skipna=True),
                "adj_flow_sum": group["adj_flow"].sum(skipna=True),
            }
        )
    area_level = pd.DataFrame(area_rows)

    moran_stats = {
        "moran_i": np.nan,
        "n_areas_used": int(np.isfinite(area_level["selected_residual"]).sum()),
        "n_links_used": np.nan,
        "weight_sum": np.nan,
    }
    if area_neighbors is not None:
        moran_stats = compute_moran_i(
            area_level, area_neighbors, area_col, neighbor_col, weight_col
        )

    moran_tbl = pd.DataFrame(
        {
            "method": [_method_value(method_name)],
            "residual_type": [residual_type_label],
            "comparison": [comparison],
            "comparison_label": [comparison_spec["comparison_label"]],
            "spatial_role": [spatial_role],
            "residual_aggregation": [residual_aggregation],
            "n_areas_used": [moran_stats["n_areas_used"]],
            "n_links_used": [moran_stats["n_links_used"]],
            "weight_sum": [moran_stats["weight_sum"]],
            "moran_i": [moran_stats["moran_i"]],
        }
    )

    covariate_data = None
    covariate_correlation = pd.DataFrame(
        {
            "method": [_method_value(method_name)],
            "residual_type": [residual_type_label],
            "comparison": [comparison],
            "comparison_label": [comparison_spec["comparison_label"]],
            "spatial_role": [spatial_role],
            "covariate": [np.nan],
            "n": [np.nan],
            "pearson_r": [np.nan],
        }
    )
    if covariate_col is not None and covariate_df is not None:
        require_columns(
            covariate_df, (covariate_area_col, covariate_col), "covariate_df"
        )
        if covariate_area_col != "area" and "area" in covariate_df.columns:
            raise DebiasSchemaError(
                "`covariate_df` already contains `area`; renaming "
                "`covariate_area_col` would create duplicate columns."
            )
        covariate_lookup = covariate_df.loc[
            :, [covariate_area_col, covariate_col]
        ].rename(
            columns={
                covariate_area_col: "area",
                covariate_col: "covariate_value",
            }
        )
        covariate_lookup = _collapse_exact_area_lookup(covariate_lookup, "covariate_df")
        covariate_data = covariate_lookup.merge(
            area_level,
            how="right",
            on="area",
            sort=False,
            validate="one_to_one",
        )
        finite_cov = np.isfinite(covariate_data["selected_residual"]) & np.isfinite(
            covariate_data["covariate_value"]
        )
        covariate_correlation = pd.DataFrame(
            {
                "method": [_method_value(method_name)],
                "residual_type": [residual_type_label],
                "comparison": [comparison],
                "comparison_label": [comparison_spec["comparison_label"]],
                "spatial_role": [spatial_role],
                "covariate": [covariate_col],
                "n": [int(finite_cov.sum())],
                "pearson_r": [
                    safe_pearson(
                        covariate_data["selected_residual"],
                        covariate_data["covariate_value"],
                    )
                ],
            }
        )

    map_data = area_level.copy()
    if geometry_df is not None:
        if geometry_area_col not in geometry_df.columns:
            raise DebiasSchemaError(
                "`geometry_area_col` must name a column in `geometry_df`."
            )
        if geometry_area_col != "area" and "area" in geometry_df.columns:
            raise DebiasSchemaError(
                "`geometry_df` already contains `area`; renaming "
                "`geometry_area_col` would create duplicate columns."
            )
        geometry_tbl = geometry_df.rename(columns={geometry_area_col: "area"})
        geometry_tbl = _collapse_exact_area_lookup(geometry_tbl, "geometry_df")
        map_data = area_level.merge(
            geometry_tbl,
            how="left",
            on="area",
            sort=False,
            validate="one_to_one",
        )

    local_moran_tbl = None
    if local_moran:
        local_moran_tbl = compute_local_moran(
            area_level=area_level,
            area_neighbors=area_neighbors,
            area_col=area_col,
            neighbor_col=neighbor_col,
            weight_col=weight_col,
            nsim=int(local_moran_nsim),
            alpha=float(local_moran_alpha),
            p_adjust=local_moran_p_adjust,
            seed=local_moran_seed,
        )
        map_data = map_data.merge(
            local_moran_tbl.loc[
                :,
                [
                    "area",
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
                ],
            ],
            how="left",
            on="area",
            sort=False,
        )

    summary_tbl = pd.DataFrame(
        {
            "method": [_method_value(method_name)],
            "residual_type": [residual_type_label],
            "comparison": [comparison],
            "comparison_label": [comparison_spec["comparison_label"]],
            "spatial_role": [spatial_role],
            "residual_aggregation": [residual_aggregation],
            "n_od_pairs": [len(od_level)],
            "n_areas": [len(area_level)],
            "pearson_residual_benchmark_flow": [flow_correlation["pearson_r"].iloc[0]],
            "moran_i": [moran_tbl["moran_i"].iloc[0]],
            "pearson_residual_covariate": [covariate_correlation["pearson_r"].iloc[0]],
        }
    )

    out: dict[str, Any] = {
        "summary": summary_tbl,
        "flow_correlation": flow_correlation,
        "moran_i": moran_tbl,
        "covariate_correlation": covariate_correlation,
        "od_level": od_level.reset_index(drop=True),
        "area_level": area_level.reset_index(drop=True),
        "map_data": map_data.reset_index(drop=True),
    }
    if covariate_data is not None:
        out["covariate_data"] = covariate_data.reset_index(drop=True)
    if local_moran_tbl is not None:
        out["local_moran"] = local_moran_tbl.reset_index(drop=True)
    if make_plots:
        from debiaspy.plot_validation import _plot_flow_residual_diagnostics

        if (x_col is None) != (y_col is None):
            raise ValueError(
                "Both `x_col` and `y_col` are required for a residual map plot."
            )
        if x_col is not None and y_col is not None:
            missing_coordinates = [
                column for column in (x_col, y_col) if column not in map_data.columns
            ]
            if missing_coordinates:
                raise ValueError("`x_col` and `y_col` must name columns in `map_data`.")
        out["plots"] = _plot_flow_residual_diagnostics(
            od_level=od_level,
            covariate_data=covariate_data,
            map_data=map_data,
            comparison_label=comparison_spec["comparison_label"],
            covariate_col=covariate_col,
            x_col=x_col,
            y_col=y_col,
        )
    return out


def safe_pearson(x: pd.Series, y: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce").astype(float)
    y_num = pd.to_numeric(y, errors="coerce").astype(float)
    keep = np.isfinite(x_num) & np.isfinite(y_num)
    if int(keep.sum()) < 2:
        return np.nan
    x_keep = x_num.loc[keep]
    y_keep = y_num.loc[keep]
    if x_keep.std(ddof=1) <= 0 or y_keep.std(ddof=1) <= 0:
        return np.nan
    return float(np.corrcoef(x_keep.to_numpy(), y_keep.to_numpy())[0, 1])


def compute_moran_i(
    area_level: pd.DataFrame,
    area_neighbors: pd.DataFrame,
    area_col: str,
    neighbor_col: str,
    weight_col: str | None,
) -> dict[str, float | int]:
    neighbors = prepare_neighbor_links(
        area_neighbors, area_col, neighbor_col, weight_col
    )
    return global_moran_i(area_level.loc[:, ["area", "selected_residual"]], neighbors)


def compute_local_moran(
    area_level: pd.DataFrame,
    area_neighbors: pd.DataFrame | None,
    area_col: str,
    neighbor_col: str,
    weight_col: str | None,
    nsim: int,
    alpha: float,
    p_adjust: str,
    seed: int | None,
) -> pd.DataFrame:
    local_base = area_level.loc[
        :,
        [
            "area",
            "method",
            "residual_type",
            "comparison",
            "comparison_label",
            "spatial_role",
            "residual_aggregation",
            "selected_residual",
        ],
    ].copy()
    if area_neighbors is None:
        out = local_base.copy()
        return _fill_local_moran_defaults(out, "no neighbours")

    neighbor_tbl = prepare_neighbor_links(
        area_neighbors, area_col, neighbor_col, weight_col
    )
    values = local_base.loc[np.isfinite(local_base["selected_residual"])].copy()
    n_areas_used = len(values)
    residual_mean = values["selected_residual"].mean()
    centered_denominator = float(
        ((values["selected_residual"] - residual_mean) ** 2).sum()
    )
    if n_areas_used < 2 or centered_denominator <= 0:
        return _fill_local_moran_defaults(local_base.copy(), "undefined")

    values["centered_residual"] = values["selected_residual"] - residual_mean
    values["residual_z"] = values["centered_residual"] / np.sqrt(
        centered_denominator / n_areas_used
    )
    links = (
        neighbor_tbl.merge(
            values.loc[:, ["area", "centered_residual", "residual_z"]],
            how="inner",
            on="area",
        )
        .rename(
            columns={
                "centered_residual": "area_centered_residual",
                "residual_z": "area_residual_z",
            }
        )
        .merge(
            values.loc[:, ["area", "centered_residual", "residual_z"]].rename(
                columns={
                    "area": "neighbor",
                    "centered_residual": "neighbor_centered_residual",
                    "residual_z": "neighbor_residual_z",
                }
            ),
            how="inner",
            on="neighbor",
        )
    )
    if links.empty:
        return _fill_local_moran_defaults(local_base.copy(), "no neighbours")

    neighbor_summary = (
        links.groupby("area", sort=True, dropna=False)
        .apply(_summarise_local_neighbors, include_groups=False)
        .reset_index()
    )
    local_tbl = local_base.merge(
        values.loc[:, ["area", "centered_residual", "residual_z"]],
        how="left",
        on="area",
    ).merge(neighbor_summary, how="left", on="area")
    local_tbl["n_neighbors_used"] = local_tbl["n_neighbors_used"].fillna(0).astype(int)
    local_tbl["local_moran_i"] = (
        local_tbl["centered_residual"] * local_tbl["neighbor_lag_sum"]
    ) / (centered_denominator / n_areas_used)

    has_neighbors = (
        (local_tbl["n_neighbors_used"] > 0)
        & np.isfinite(local_tbl["local_moran_i"])
        & np.isfinite(local_tbl["centered_residual"])
        & np.isfinite(local_tbl["neighbor_lag_mean"])
    )
    p_value = np.full(len(local_tbl), np.nan)
    if bool(has_neighbors.any()):
        rng = np.random.default_rng(seed)
        area_index = pd.Series(np.arange(len(local_tbl)), index=local_tbl["area"])
        link_index = links["area"].map(area_index).to_numpy()
        neighbor_index = links["neighbor"].map(area_index).to_numpy()
        weight = links["weight"].to_numpy(dtype=float)
        finite_idx = np.flatnonzero(np.isfinite(local_tbl["centered_residual"]))
        centered = local_tbl["centered_residual"].to_numpy(dtype=float)
        observed = local_tbl["local_moran_i"].to_numpy(dtype=float)
        for focal_idx in np.flatnonzero(has_neighbors.to_numpy()):
            other_idx = np.setdiff1d(finite_idx, np.array([focal_idx]))
            simulated = np.empty(nsim, dtype=float)
            focal_links = link_index == focal_idx
            for sim_idx in range(nsim):
                permuted = centered.copy()
                permuted[other_idx] = rng.permutation(centered[other_idx])
                neighbor_lag_sum = float(
                    np.sum(weight[focal_links] * permuted[neighbor_index[focal_links]])
                )
                simulated[sim_idx] = (
                    centered[focal_idx]
                    * neighbor_lag_sum
                    / (centered_denominator / n_areas_used)
                )
            p_value[focal_idx] = (
                np.sum(np.abs(simulated) >= abs(observed[focal_idx])) + 1
            ) / (nsim + 1)

    p_adjusted = _p_adjust(p_value, p_adjust)
    significant = pd.array(p_adjusted <= alpha, dtype="boolean")
    significant[~np.isfinite(p_adjusted)] = pd.NA
    local_tbl["lisa_quadrant"] = [
        _local_moran_quadrant(centered, lag, neighbors > 0, np.isfinite(local_i))
        for centered, lag, neighbors, local_i in zip(
            local_tbl["centered_residual"],
            local_tbl["neighbor_lag_mean"],
            local_tbl["n_neighbors_used"],
            local_tbl["local_moran_i"],
            strict=True,
        )
    ]
    local_tbl["p_value"] = p_value
    local_tbl["p_adjusted"] = p_adjusted
    local_tbl["significant"] = significant
    local_tbl["lisa_cluster"] = [
        _local_moran_cluster(quadrant, sig, neighbors > 0, np.isfinite(local_i))
        for quadrant, sig, neighbors, local_i in zip(
            local_tbl["lisa_quadrant"],
            local_tbl["significant"],
            local_tbl["n_neighbors_used"],
            local_tbl["local_moran_i"],
            strict=True,
        )
    ]
    return local_tbl


def prepare_neighbor_links(
    area_neighbors: pd.DataFrame,
    area_col: str,
    neighbor_col: str,
    weight_col: str | None,
) -> pd.DataFrame:
    require_columns(area_neighbors, (area_col, neighbor_col), "area_neighbors")
    if weight_col is not None and weight_col not in area_neighbors.columns:
        raise ValueError("`weight_col` must name a column in `area_neighbors`.")
    cols = [area_col, neighbor_col] + ([] if weight_col is None else [weight_col])
    out = area_neighbors.loc[:, cols].rename(
        columns={area_col: "area", neighbor_col: "neighbor"}
    )
    if weight_col is None:
        out["weight"] = 1.0
    else:
        out = out.rename(columns={weight_col: "weight"})
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").astype(float)
    return out.loc[np.isfinite(out["weight"]) & (out["weight"] > 0)].reset_index(
        drop=True
    )


def _method_value(method_name: str | None) -> object:
    return np.nan if method_name is None else method_name


def _collapse_exact_area_lookup(
    frame: pd.DataFrame,
    frame_name: str,
) -> pd.DataFrame:
    exact = frame.drop_duplicates(keep="first")
    if bool(exact["area"].duplicated(keep=False).any()):
        raise DebiasSchemaError(
            f"`{frame_name}` contains conflicting rows for `area`; "
            "the area join is not one-to-one."
        )
    return exact.reset_index(drop=True)


def _aggregate_residual(x: pd.Series, aggregation: str) -> float:
    if aggregation == "sum":
        return float(x.sum(skipna=True))
    return float(x.mean(skipna=True))


def _validate_local_moran_args(
    local_moran_nsim: int,
    local_moran_alpha: float,
    local_moran_p_adjust: str,
    local_moran_seed: int | float | None,
) -> int | None:
    if (
        isinstance(local_moran_nsim, (bool, np.bool_))
        or not isinstance(local_moran_nsim, (int, np.integer))
        or local_moran_nsim < 1
    ):
        raise ValueError("`local_moran_nsim` must be a positive whole number.")
    if (
        not isinstance(local_moran_alpha, int | float)
        or not np.isfinite(local_moran_alpha)
        or local_moran_alpha < 0
        or local_moran_alpha > 1
    ):
        raise ValueError("`local_moran_alpha` must be a number between 0 and 1.")
    valid_adjust = {
        "holm",
        "hochberg",
        "hommel",
        "bonferroni",
        "BH",
        "BY",
        "fdr",
        "none",
    }
    if local_moran_p_adjust not in valid_adjust:
        raise ValueError(
            "`local_moran_p_adjust` must be one of `stats::p.adjust.methods`."
        )
    if local_moran_seed is None:
        return None
    if isinstance(local_moran_seed, (bool, np.bool_)) or not isinstance(
        local_moran_seed, (int, float, np.integer, np.floating)
    ):
        raise ValueError("`local_moran_seed` must be `None` or a finite whole number.")
    numeric_seed = float(local_moran_seed)
    if (
        not np.isfinite(numeric_seed)
        or not numeric_seed.is_integer()
        or numeric_seed < 0
    ):
        raise ValueError("`local_moran_seed` must be `None` or a finite whole number.")
    return int(numeric_seed)


def _fill_local_moran_defaults(data: pd.DataFrame, cluster: str) -> pd.DataFrame:
    data["centered_residual"] = np.nan
    data["residual_z"] = np.nan
    data["neighbor_lag_sum"] = np.nan
    data["neighbor_lag_mean"] = np.nan
    data["spatial_lag_z"] = np.nan
    data["n_neighbors_used"] = 0
    data["local_weight_sum"] = np.nan
    data["local_moran_i"] = np.nan
    data["lisa_quadrant"] = cluster
    data["p_value"] = np.nan
    data["p_adjusted"] = np.nan
    data["significant"] = np.nan
    data["lisa_cluster"] = cluster
    return data


def _summarise_local_neighbors(group: pd.DataFrame) -> pd.Series:
    weight_sum = group["weight"].sum(skipna=True)
    neighbor_lag_sum = (group["weight"] * group["neighbor_centered_residual"]).sum(
        skipna=True
    )
    spatial_lag_z = (group["weight"] * group["neighbor_residual_z"]).sum(skipna=True)
    return pd.Series(
        {
            "n_neighbors_used": len(group),
            "local_weight_sum": weight_sum,
            "neighbor_lag_sum": neighbor_lag_sum,
            "neighbor_lag_mean": neighbor_lag_sum / weight_sum,
            "spatial_lag_z": spatial_lag_z / weight_sum,
        }
    )


def _p_adjust(p_values: np.ndarray, method: str) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    vals = p[finite]
    n = len(vals)
    if n == 0:
        return out
    method_lower = method.lower()
    if method_lower == "none":
        adjusted = vals
    elif method_lower == "bonferroni":
        adjusted = np.minimum(vals * n, 1.0)
    elif method_lower in {"bh", "fdr"}:
        order = np.argsort(vals)
        ranked = vals[order] * n / np.arange(1, n + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted = np.empty(n)
        adjusted[order] = np.minimum(ranked, 1.0)
    elif method_lower == "by":
        harmonic = np.sum(1 / np.arange(1, n + 1))
        order = np.argsort(vals)
        ranked = vals[order] * n * harmonic / np.arange(1, n + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted = np.empty(n)
        adjusted[order] = np.minimum(ranked, 1.0)
    elif method_lower == "holm":
        order = np.argsort(vals)
        ranked = (n - np.arange(n)) * vals[order]
        ranked = np.maximum.accumulate(ranked)
        adjusted = np.empty(n)
        adjusted[order] = np.minimum(ranked, 1.0)
    elif method_lower in {"hochberg", "hommel"}:
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError as exc:
            raise ImportError(
                f"`local_moran_p_adjust` method {method!r} requires statsmodels."
            ) from exc
        statsmodels_method = (
            "simes-hochberg" if method_lower == "hochberg" else "hommel"
        )
        adjusted = multipletests(vals, method=statsmodels_method)[1]
    else:
        raise ValueError(
            "`local_moran_p_adjust` must be one of `stats::p.adjust.methods`."
        )
    out[finite] = adjusted
    return out


def _local_moran_quadrant(
    centered_residual: float,
    neighbor_lag_mean: float,
    has_neighbors: bool,
    defined: bool,
) -> str:
    if not has_neighbors:
        return "no neighbours"
    if not defined:
        return "undefined"
    if centered_residual > 0 and neighbor_lag_mean > 0:
        return "high-high"
    if centered_residual < 0 and neighbor_lag_mean < 0:
        return "low-low"
    if centered_residual > 0 and neighbor_lag_mean < 0:
        return "high-low"
    if centered_residual < 0 and neighbor_lag_mean > 0:
        return "low-high"
    return "undefined"


def _local_moran_cluster(
    lisa_quadrant: str,
    significant: object,
    has_neighbors: bool,
    defined: bool,
) -> str:
    if not has_neighbors:
        return "no neighbours"
    if not defined:
        return "undefined"
    sig = cast(bool, significant) if isinstance(significant, bool | np.bool_) else False
    if not sig:
        return "not significant"
    return lisa_quadrant


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
