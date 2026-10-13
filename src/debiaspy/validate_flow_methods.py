from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from debiaspy._choices import match_choice
from debiaspy._exceptions import DebiasSchemaError
from debiaspy._flow_joins import (
    REPEATED_OBSERVATION_KEYS,
    merge_validation_reference,
    validation_join_keys,
)
from debiaspy._schemas import require_columns
from debiaspy.validate_flows import (
    validate_flow_distribution,
    validate_flow_overall,
    validate_flow_residual_structure,
    validate_flow_residuals,
)


def validate_flow_prepare_output(
    adj_df: pd.DataFrame,
    mpd_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
) -> pd.DataFrame:
    """Attach canonical raw MPD flow to one adjusted output.

    ``adj_df`` requires OD keys and ``flow_col_adj``; ``mpd_df`` requires the
    same OD keys and ``flow_col_mpd``. Shared source/time fields are used for
    repeated-observation joins. The returned frame has canonical raw ``flow``
    and ``flow_adj`` columns while preserving non-conflicting adjusted fields.
    Missing columns, reserved-name collisions, or ambiguous joins fail closed.
    Parity is scoped to the audited preparation schemas and row values.
    """
    require_columns(mpd_df, ("origin", "destination", flow_col_mpd), "mpd_df")
    require_columns(adj_df, ("origin", "destination", flow_col_adj), "adj_df")

    join_keys = validation_join_keys(adj_df, mpd_df)
    raw_columns = list(
        dict.fromkeys(
            [
                *join_keys,
                *[key for key in REPEATED_OBSERVATION_KEYS if key in mpd_df.columns],
                flow_col_mpd,
            ]
        )
    )
    raw_flow = mpd_df.loc[:, raw_columns].rename(columns={flow_col_mpd: "raw_flow"})
    reference_only_keys = [
        key
        for key in REPEATED_OBSERVATION_KEYS
        if key in mpd_df.columns and key not in adj_df.columns
    ]

    adj_tbl = adj_df.copy()
    if "raw_flow" in adj_tbl.columns:
        raise DebiasSchemaError(
            "`adj_df` contains reserved column `raw_flow`; it would collide "
            "with the raw MPD flow used during validation preparation."
        )
    if flow_col_adj == "flow":
        if "flow_adj" in adj_tbl.columns:
            raise DebiasSchemaError(
                "`adj_df` already contains `flow_adj`; it conflicts with "
                "`flow_col_adj='flow'`."
            )
        adj_tbl = adj_tbl.rename(columns={"flow": "flow_adj"})
    else:
        if "flow" in adj_tbl.columns:
            if "adjusted_input_flow" in adj_tbl.columns:
                raise DebiasSchemaError(
                    "`adj_df` contains both `flow` and `adjusted_input_flow`; "
                    "the preparation rename would create duplicate columns."
                )
            adj_tbl = adj_tbl.rename(columns={"flow": "adjusted_input_flow"})
        if flow_col_adj != "flow_adj":
            if "flow_adj" in adj_tbl.columns:
                raise DebiasSchemaError(
                    "`adj_df` already contains `flow_adj`; it conflicts with "
                    f"`flow_col_adj={flow_col_adj!r}`."
                )
            adj_tbl = adj_tbl.rename(columns={flow_col_adj: "flow_adj"})

    prepared = merge_validation_reference(
        adj_tbl,
        raw_flow,
        how="left",
        data_name="adj_df",
        reference_name="mpd_df",
    )
    return (
        prepared.drop(columns=reference_only_keys)
        .rename(columns={"raw_flow": "flow"})
        .reset_index(drop=True)
    )


def validate_flow_prepare_outputs(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    mpd_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
) -> dict[str, pd.DataFrame]:
    """Attach canonical raw MPD flow to each named adjusted output.

    ``adjusted_outputs`` must map non-empty method names to data frames accepted
    by :func:`validate_flow_prepare_output`; ``mpd_df`` supplies OD keys and the
    selected raw flow column. Returns a method-keyed dictionary of prepared
    frames. Unnamed collections, missing columns, name collisions, or ambiguous
    joins raise. Parity is scoped to the audited multi-method preparation cases.
    """
    method_names = _validate_flow_method_names(adjusted_outputs)
    return {
        method_name: _call_for_method(
            method_name,
            validate_flow_prepare_output,
            adj_df=adjusted_outputs[method_name],
            mpd_df=mpd_df,
            flow_col_mpd=flow_col_mpd,
            flow_col_adj=flow_col_adj,
        )
        for method_name in method_names
    }


def _validate_flow_method_names(
    adjusted_outputs: Mapping[str, pd.DataFrame],
) -> list[str]:
    if not isinstance(adjusted_outputs, Mapping) or isinstance(
        adjusted_outputs, pd.DataFrame
    ):
        raise ValueError(
            "`adjusted_outputs` must be a mapping of method names to data frames."
        )
    method_names = list(adjusted_outputs.keys())
    if not method_names or any(
        not isinstance(name, str) or name == "" for name in method_names
    ):
        raise ValueError(
            "`adjusted_outputs` must be a non-empty mapping with "
            "non-empty string method names."
        )
    return method_names


def validate_flow_overall_methods(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    benchmark_od_df: pd.DataFrame,
    comparisons: str | list[str] = "all",
    drop_zeros: bool = False,
    return_joined: bool = False,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Compute overall validation metrics for multiple adjustment outputs.

    ``adjusted_outputs`` must map method names to frames accepted by
    :func:`validate_flow_overall`; the benchmark must contain matching OD keys
    and flow. Comparison, filtering, join-retention, and forwarded options are
    applied consistently. Returns a method-keyed dictionary of overall results.
    Invalid names, schemas, comparisons, or joins raise. Parity is scoped to
    audited wrapper calls and the underlying overall-metric tolerances.
    """
    method_names = _validate_flow_method_names(adjusted_outputs)
    return {
        method_name: _call_for_method(
            method_name,
            validate_flow_overall,
            adj_df=adjusted_outputs[method_name],
            benchmark_od_df=benchmark_od_df,
            comparisons=comparisons,
            drop_zeros=drop_zeros,
            return_joined=return_joined,
            method_name=method_name,
            **kwargs,
        )
        for method_name in method_names
    }


def validate_flow_residual_methods(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    benchmark_od_df: pd.DataFrame,
    top_n: int = 10,
    **kwargs: Any,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Compute residual diagnostics for multiple adjustment outputs.

    ``adjusted_outputs`` must map method names to frames with OD keys plus raw
    and adjusted flows; the benchmark supplies matching OD keys and flow.
    ``top_n`` and forwarded column options apply to every method. Returns a
    method-keyed dictionary of residual summary/data/top-worst tables. Invalid
    names, schemas, controls, or joins raise. Parity is scoped to audited wrapper
    calls and the underlying residual-table tolerances.
    """
    method_names = _validate_flow_method_names(adjusted_outputs)
    return {
        method_name: _call_for_method(
            method_name,
            validate_flow_residuals,
            adj_df=adjusted_outputs[method_name],
            benchmark_od_df=benchmark_od_df,
            top_n=top_n,
            method_name=method_name,
            **kwargs,
        )
        for method_name in method_names
    }


def validate_flow_distribution_methods(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    benchmark_od_df: pd.DataFrame,
    comparisons: str | list[str] = "all",
    weight_by: str = "benchmark_origin_total",
    return_origin_level: bool = True,
    return_od_level: bool = False,
    **kwargs: Any,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Compute distribution diagnostics for multiple adjustment outputs.

    ``adjusted_outputs`` must map method names to OD flow frames; the benchmark
    requires matching OD keys and flow. Comparison, weighting, detail, and
    forwarded options apply to every method. Returns a method-keyed dictionary
    of divergence tables. Invalid names, schemas, smoothing, options, or joins
    raise. Parity is scoped to audited wrapper calls and the underlying
    distribution-table tolerances.
    """
    method_names = _validate_flow_method_names(adjusted_outputs)
    return {
        method_name: _call_for_method(
            method_name,
            validate_flow_distribution,
            adj_df=adjusted_outputs[method_name],
            benchmark_od_df=benchmark_od_df,
            comparisons=comparisons,
            method_name=method_name,
            weight_by=weight_by,
            return_origin_level=return_origin_level,
            return_od_level=return_od_level,
            **kwargs,
        )
        for method_name in method_names
    }


def validate_flow_residual_structure_methods(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    benchmark_od_df: pd.DataFrame,
    methods: Sequence[str] | None = None,
    comparison: str = "adjusted_vs_benchmark",
    **kwargs: Any,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Compute residual-structure diagnostics for selected methods.

    ``adjusted_outputs`` must map method names to OD flow frames accepted by
    :func:`validate_flow_residual_structure`; the benchmark supplies matching
    OD keys and flow. ``methods`` can select a subset, and remaining options are
    forwarded consistently. Returns a method-keyed diagnostic dictionary.
    Unknown methods, invalid schemas, or unsupported diagnostic options raise.
    Parity is scoped to audited wrapper calls and underlying tabular outputs.
    """
    method_names = _validate_flow_method_names(adjusted_outputs)
    selected = (
        method_names
        if methods is None
        else [methods]
        if isinstance(methods, str)
        else list(methods)
    )
    missing_methods = [method for method in selected if method not in method_names]
    if missing_methods:
        raise ValueError(
            "`methods` must name elements in `adjusted_outputs`. Missing: "
            + ", ".join(missing_methods)
        )
    return {
        method_name: _call_for_method(
            method_name,
            validate_flow_residual_structure,
            adj_df=adjusted_outputs[method_name],
            benchmark_od_df=benchmark_od_df,
            method_name=method_name,
            comparison=comparison,
            **kwargs,
        )
        for method_name in selected
    }


def validate_flow_margins(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    benchmark_od_df: pd.DataFrame,
    role: str = "origin",
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
    flow_col_bench: str = "flow",
) -> dict[str, Any]:
    """Aggregate adjusted, raw, and benchmark flows to area margins.

    Each named adjusted output requires ``origin``, ``destination``, and the
    selected raw and adjusted flow columns. The benchmark requires the same OD
    keys and its selected flow. ``role`` chooses origin or destination margins.
    Returns ``adjusted_outputs`` as method-keyed margin frames plus a
    ``benchmark`` margin frame. Invalid names, roles, or schemas raise. Parity
    is scoped to audited margin-wrapper schemas and deterministic sums.
    """
    role = match_choice(role, ("origin", "destination"), "role")
    method_names = _validate_flow_method_names(adjusted_outputs)
    require_columns(
        benchmark_od_df, ("origin", "destination", flow_col_bench), "benchmark_od_df"
    )

    def adjusted_margin(data: pd.DataFrame) -> pd.DataFrame:
        require_columns(
            data,
            ("origin", "destination", flow_col_mpd, flow_col_adj),
            "adjusted output",
        )
        if role == "origin":
            grouped = (
                data.groupby("origin", sort=True, dropna=False)
                .agg(flow=(flow_col_mpd, "sum"), flow_adj=(flow_col_adj, "sum"))
                .reset_index()
            )
            grouped["destination"] = "origin_total"
            return grouped.loc[:, ["origin", "destination", "flow", "flow_adj"]]

        grouped = (
            data.groupby("destination", sort=True, dropna=False)
            .agg(flow=(flow_col_mpd, "sum"), flow_adj=(flow_col_adj, "sum"))
            .reset_index()
        )
        grouped["origin"] = "destination_total"
        return grouped.loc[:, ["origin", "destination", "flow", "flow_adj"]]

    if role == "origin":
        benchmark = (
            benchmark_od_df.groupby("origin", sort=True, dropna=False)[flow_col_bench]
            .sum()
            .reset_index(name="flow")
        )
        benchmark["destination"] = "origin_total"
        benchmark = benchmark.loc[:, ["origin", "destination", "flow"]]
    else:
        benchmark = (
            benchmark_od_df.groupby("destination", sort=True, dropna=False)[
                flow_col_bench
            ]
            .sum()
            .reset_index(name="flow")
        )
        benchmark["origin"] = "destination_total"
        benchmark = benchmark.loc[:, ["origin", "destination", "flow"]]

    return {
        "adjusted_outputs": {
            method_name: _call_for_method(
                method_name, adjusted_margin, adj_df=adjusted_outputs[method_name]
            )
            for method_name in method_names
        },
        "benchmark": benchmark.reset_index(drop=True),
    }


def _call_for_method(
    name: str,
    function: Callable[..., Any],
    *,
    adj_df: pd.DataFrame,
    **kwargs: Any,
) -> Any:
    """Identify the failing method while preserving the underlying error type."""
    if not isinstance(adj_df, pd.DataFrame):
        raise ValueError(
            f"Method {name!r}: adjusted output must be a pandas DataFrame."
        )
    try:
        return function(adj_df, **kwargs)
    except (ValueError, TypeError, KeyError) as exc:
        raise type(exc)(f"Method {name!r}: {exc}") from exc
