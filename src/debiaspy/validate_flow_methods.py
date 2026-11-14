"""Prepare and validate named adjusted-flow outputs against a shared benchmark."""

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import pandas as pd

from debiaspy import validate_flows
from debiaspy.measure_bias import (
    _require_dataframe,
    _require_numeric_columns,
    _require_string_column_names,
)


class _MarginResult(TypedDict):
    adjusted_outputs: dict[str, pd.DataFrame]
    benchmark: pd.DataFrame


def validate_flow_prepare_output(
    adj_df: pd.DataFrame,
    mpd_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
) -> pd.DataFrame:
    """Join raw flows to an adjusted-flow table without changing either input.

    Parameters
    ----------
    adj_df:
        Adjusted flows containing ``origin``, ``destination`` and the adjusted
        flow column. Other columns are retained in their original order.
    mpd_df:
        Raw flows containing ``origin``, ``destination`` and the raw flow column.
    flow_col_mpd, flow_col_adj:
        Names of the raw and adjusted flow columns, respectively.

    Returns
    -------
    pandas.DataFrame
        A left join on ``origin`` and ``destination`` with canonical ``flow``
        and ``flow_adj`` columns. An existing ``flow`` in ``adj_df`` becomes
        ``adjusted_input_flow``; the raw flow column is appended last. The
        returned table has a fresh positional index.

    Notes
    -----
    Unmatched pairs retain missing raw flows. Missing keys match missing keys,
    and duplicate keys expand the join. Many-to-many joins emit a warning.
    Column-name collisions are rejected rather than silently suffixed.
    ``flow_col_adj="flow"`` is unsupported because ``flow`` is reserved for
    preservation as ``adjusted_input_flow``.
    """
    _require_string_column_names(flow_col_mpd, flow_col_adj)
    _check_columns(mpd_df, ["origin", "destination", flow_col_mpd], "mpd_df")
    _check_columns(adj_df, ["origin", "destination", flow_col_adj], "adj_df")
    if flow_col_adj in {"origin", "destination", "flow"}:
        raise ValueError("`flow_col_adj` must name a separate adjusted-flow column.")
    if flow_col_mpd in {"origin", "destination"}:
        raise ValueError("`flow_col_mpd` must name a separate raw-flow column.")

    rename = {flow_col_adj: "flow_adj"}
    if "flow" in adj_df.columns:
        rename["flow"] = "adjusted_input_flow"
    renamed_columns = [rename.get(column, column) for column in adj_df.columns]
    if len(set(renamed_columns)) != len(renamed_columns):
        raise ValueError("`adj_df` has conflicting columns after flow renaming.")
    if "raw_flow" in renamed_columns:
        raise ValueError("`adj_df` contains reserved column `raw_flow`.")

    adjusted = adj_df.rename(columns=rename)
    raw = mpd_df.loc[:, ["origin", "destination", flow_col_mpd]].copy()
    raw.columns = ["origin", "destination", "flow"]
    keys = ["origin", "destination"]
    # Match dplyr's warning only when a shared key is duplicated on both sides.
    repeated_adjusted = adjusted.loc[adjusted.duplicated(keys, keep=False), keys]
    repeated_raw = raw.loc[raw.duplicated(keys, keep=False), keys]
    if not repeated_adjusted.empty and not repeated_raw.empty:
        if (
            not repeated_adjusted.drop_duplicates()
            .merge(repeated_raw.drop_duplicates(), on=keys)
            .empty
        ):
            warnings.warn(
                "Many-to-many origin/destination join expands adjusted rows.",
                UserWarning,
                stacklevel=2,
            )
    return adjusted.merge(raw, on=keys, how="left", sort=False)


def validate_flow_prepare_outputs(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    mpd_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
) -> dict[str, pd.DataFrame]:
    """Prepare each named adjusted-flow table using the same raw flows.

    Parameters
    ----------
    adjusted_outputs:
        Mapping of non-empty string method IDs to adjusted-flow DataFrames.
        Mapping order and method IDs are retained. An empty mapping is allowed.
    mpd_df:
        Raw flows with origin and destination keys.
    flow_col_mpd, flow_col_adj:
        Raw and adjusted flow column names passed to
        :func:`validate_flow_prepare_output`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Prepared outputs keyed by method ID. Invalid inputs identify the method
        and retain the underlying exception as their cause.
    """
    _check_methods(adjusted_outputs)
    prepared = {}
    for method, data in adjusted_outputs.items():
        try:
            prepared[method] = validate_flow_prepare_output(
                data, mpd_df, flow_col_mpd=flow_col_mpd, flow_col_adj=flow_col_adj
            )
        except (TypeError, ValueError) as error:
            raise type(error)(f"Method {method!r}: {error}") from error
    return prepared


def validate_flow_overall_methods(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    benchmark_od_df: pd.DataFrame,
    comparisons: str | Sequence[str] = "all",
    drop_zeros: bool = False,
    return_joined: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply overall validation to each method in mapping order.

    Parameters
    ----------
    adjusted_outputs:
        Mapping of non-empty string method IDs to adjusted-flow DataFrames.
    benchmark_od_df:
        Shared benchmark passed unchanged to each validator call.
    comparisons:
        Flow comparisons passed to ``validate_flow_overall``. Defaults to all.
    drop_zeros, return_joined:
        Passed to the core validator; both default to False in this wrapper.
    **kwargs:
        Additional core-validator arguments. Do not supply ``method_name`` or
        ``adj_df``; these are derived from the mapping.

    Returns
    -------
    dict[str, Any]
        Unmodified core-validator results keyed by method ID.

    Raises
    ------
    NotImplementedError
        If ``validate_flow_overall`` is not yet available in ``validate_flows``.
        An empty mapping returns an empty dictionary without calling a validator.
    """
    return _apply_flow_validation(
        adjusted_outputs,
        benchmark_od_df,
        "validate_flow_overall",
        {
            "comparisons": comparisons,
            "drop_zeros": drop_zeros,
            "return_joined": return_joined,
            **kwargs,
        },
    )


def validate_flow_residual_methods(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    benchmark_od_df: pd.DataFrame,
    top_n: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply residual validation to each method in mapping order.

    Parameters
    ----------
    adjusted_outputs:
        Mapping of non-empty string method IDs to adjusted-flow DataFrames.
    benchmark_od_df:
        Shared benchmark passed unchanged to each validator call.
    top_n:
        Number of worst pairs requested from ``validate_flow_residuals``.
    **kwargs:
        Additional core-validator arguments. ``method_name`` and ``adj_df``
        are derived from the mapping and cannot be overridden.

    Returns
    -------
    dict[str, Any]
        Unmodified residual-validation results keyed by method ID.

    Raises
    ------
    NotImplementedError
        If ``validate_flow_residuals`` is not yet available in ``validate_flows``.
        Empty mappings return an empty dictionary without calling a validator.
    """
    return _apply_flow_validation(
        adjusted_outputs,
        benchmark_od_df,
        "validate_flow_residuals",
        {"top_n": top_n, **kwargs},
    )


def validate_flow_distribution_methods(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    benchmark_od_df: pd.DataFrame,
    comparisons: str | Sequence[str] = "all",
    weight_by: str = "benchmark_origin_total",
    return_origin_level: bool = True,
    return_od_level: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply distribution validation to each method in mapping order.

    Parameters
    ----------
    adjusted_outputs:
        Mapping of non-empty string method IDs to adjusted-flow DataFrames.
    benchmark_od_df:
        Shared benchmark passed unchanged to each validator call.
    comparisons:
        Comparisons passed to ``validate_flow_distribution``. Defaults to all.
    weight_by:
        Summary weighting, defaulting to benchmark origin totals.
    return_origin_level, return_od_level:
        Request origin-level and OD-level outputs from the core validator.
    **kwargs:
        Additional core-validator arguments, such as column names or epsilon.
        ``method_name`` and ``adj_df`` cannot be overridden.

    Returns
    -------
    dict[str, Any]
        Unmodified distribution-validation results keyed by method ID.

    Raises
    ------
    NotImplementedError
        If ``validate_flow_distribution`` is unavailable in ``validate_flows``.
        Empty mappings return an empty dictionary without calling a validator.
    """
    return _apply_flow_validation(
        adjusted_outputs,
        benchmark_od_df,
        "validate_flow_distribution",
        {
            "comparisons": comparisons,
            "weight_by": weight_by,
            "return_origin_level": return_origin_level,
            "return_od_level": return_od_level,
            **kwargs,
        },
    )


def validate_flow_residual_structure_methods(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    benchmark_od_df: pd.DataFrame,
    methods: str | Sequence[str] | None = None,
    comparison: str = "adjusted_vs_benchmark",
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate residual structure for all methods or an ordered selection.

    Parameters
    ----------
    adjusted_outputs:
        Mapping of non-empty string method IDs to adjusted-flow DataFrames.
    benchmark_od_df:
        Shared benchmark passed unchanged to each validator call.
    methods:
        One method ID or an ordered sequence of unique IDs. None selects all
        methods in mapping order; an empty sequence selects none. Unknown or
        repeated IDs raise ValueError because dictionary keys must be unique.
    comparison:
        Comparison passed to ``validate_flow_residual_structure``.
    **kwargs:
        Additional core-validator arguments, including spatial and covariate
        options. ``method_name`` and ``adj_df`` cannot be overridden.

    Returns
    -------
    dict[str, Any]
        Unmodified structure-validation results in the requested method order.

    Raises
    ------
    NotImplementedError
        If ``validate_flow_residual_structure`` is unavailable in
        ``validate_flows``. Empty selections do not call a validator.
    """
    return _apply_flow_validation(
        adjusted_outputs,
        benchmark_od_df,
        "validate_flow_residual_structure",
        {"comparison": comparison, **kwargs},
        methods=methods,
    )


def _apply_flow_validation(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    benchmark_od_df: pd.DataFrame,
    validator_name: str,
    options: dict[str, Any],
    methods: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    _check_methods(adjusted_outputs)
    if methods is None:
        selected = list(adjusted_outputs)
    elif isinstance(methods, str):
        selected = [methods]
    elif isinstance(methods, Sequence):
        selected = list(methods)
    else:
        raise TypeError("`methods` must be a method ID or an ordered sequence of IDs.")
    if any(not isinstance(method, str) or not method for method in selected):
        raise ValueError("`methods` must contain non-empty string method IDs.")
    missing = [method for method in selected if method not in adjusted_outputs]
    if missing:
        raise ValueError(
            "`methods` must name elements in `adjusted_outputs`. Missing: "
            + ", ".join(missing)
        )
    if len(set(selected)) != len(selected):
        raise ValueError("`methods` must not contain duplicate method IDs.")
    if not selected:
        return {}

    reserved = [key for key in ("method_name", "adj_df") if key in options]
    if reserved:
        raise TypeError(
            f"Method {selected[0]!r}: {', '.join(reserved)} is derived from "
            "`adjusted_outputs` and must not be supplied."
        )
    # Resolve at call time so preparation remains usable before the core ports land.
    validator = getattr(validate_flows, validator_name, None)
    if validator is None:
        raise NotImplementedError(
            f"Method {selected[0]!r}: `{validator_name}` is not yet available in "
            "debiaspy.validate_flows. Implement the single-output validator "
            "before running this multi-method validation."
        )
    results = {}
    for method in selected:
        try:
            results[method] = validator(
                adj_df=adjusted_outputs[method],
                benchmark_od_df=benchmark_od_df,
                method_name=method,
                **options,
            )
        except (TypeError, ValueError, KeyError) as error:
            raise type(error)(f"Method {method!r}: {error}") from error
    return results


def validate_flow_margins(
    adjusted_outputs: Mapping[str, pd.DataFrame],
    benchmark_od_df: pd.DataFrame,
    role: str = "origin",
    flow_col_mpd: str = "flow",
    flow_col_adj: str = "flow_adj",
    flow_col_bench: str = "flow",
) -> _MarginResult:
    """Aggregate named adjusted outputs and their benchmark to area totals.

    Parameters
    ----------
    adjusted_outputs:
        Mapping of non-empty string method IDs to DataFrames containing
        ``origin``, ``destination``, raw flows and adjusted flows.
    benchmark_od_df:
        Benchmark DataFrame with ``origin``, ``destination`` and benchmark flows.
    role:
        ``"origin"`` for outflow totals or ``"destination"`` for inflow totals.
    flow_col_mpd, flow_col_adj, flow_col_bench:
        Names of the raw, adjusted and benchmark flow columns.

    Returns
    -------
    dict
        ``adjusted_outputs`` maps method IDs to tables with columns ``origin``,
        ``destination``, ``flow``, ``flow_adj``. ``benchmark`` contains the
        corresponding benchmark table without ``flow_adj``. The unused key is
        set to ``"origin_total"`` or ``"destination_total"``.

    Notes
    -----
    Methods retain mapping order. Each table is aggregated independently, with
    sorted area keys and missing keys last. Missing flow values are omitted;
    an all-missing group sums to zero. Duplicate pairs contribute to totals.
    No input is mutated and no missing areas are filled across methods.
    """
    if role not in ("origin", "destination"):
        raise ValueError("`role` must be 'origin' or 'destination'.")
    _check_methods(adjusted_outputs)
    _require_string_column_names(flow_col_mpd, flow_col_adj, flow_col_bench)
    benchmark = _aggregate_margin(
        benchmark_od_df, role, {"flow": flow_col_bench}, "benchmark_od_df"
    )
    margins = {}
    for method, data in adjusted_outputs.items():
        margins[method] = _aggregate_margin(
            data,
            role,
            {"flow": flow_col_mpd, "flow_adj": flow_col_adj},
            f"adjusted_outputs[{method!r}]",
        )
    return {"adjusted_outputs": margins, "benchmark": benchmark}


def _check_methods(adjusted_outputs: Mapping[str, pd.DataFrame]) -> None:
    if not isinstance(adjusted_outputs, Mapping):
        raise TypeError(
            "`adjusted_outputs` must be a mapping of method IDs to DataFrames."
        )
    for method, data in adjusted_outputs.items():
        if not isinstance(method, str) or not method:
            raise ValueError("`adjusted_outputs` method IDs must be non-empty strings.")
        _require_dataframe(data, f"adjusted_outputs[{method!r}]")


def _check_columns(data: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    _require_dataframe(data, name)
    if not data.columns.is_unique:
        raise ValueError(f"`{name}` must have unique column names.")
    missing = list(dict.fromkeys(column for column in columns if column not in data))
    if missing:
        raise ValueError(f"`{name}` must contain missing columns: {', '.join(missing)}")


def _aggregate_margin(
    data: pd.DataFrame, role: str, flow_columns: dict[str, str], name: str
) -> pd.DataFrame:
    _check_columns(data, ["origin", "destination", *flow_columns.values()], name)
    _require_numeric_columns(data, list(flow_columns.values()), name)
    totals = data.groupby(role, sort=True, dropna=False, observed=True).agg(
        **{output: (source, "sum") for output, source in flow_columns.items()}
    )
    totals = totals.reset_index()
    other = "destination" if role == "origin" else "origin"
    totals[other] = f"{role}_total"
    return totals.loc[:, ["origin", "destination", *flow_columns]]
