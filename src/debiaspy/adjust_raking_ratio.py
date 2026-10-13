from __future__ import annotations

import numpy as np
import pandas as pd

from debiaspy._adjustment_common import (
    existing_columns,
    r_character_key_series,
    require_group_columns,
)
from debiaspy._exceptions import DebiasSchemaError
from debiaspy._schemas import require_columns


def adjust_raking_ratio(
    mpd_od_df: pd.DataFrame,
    origin_targets: pd.DataFrame | None = None,
    destination_targets: pd.DataFrame | None = None,
    benchmark_od_df: pd.DataFrame | None = None,
    flow_col_bench: str = "flow",
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    max_iter: int | float = 200,
    tol: float = 1e-6,
    clip_min: float = 0,
    clip_max: float = np.inf,
    keep_cols: str | list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """Rake MPD OD flows to supplied origin and destination margins.

    ``mpd_od_df`` requires ``origin``, ``destination``, and non-negative
    ``flow``. Target tables require their area key and ``target`` plus every
    ``group_cols`` field; alternatively margins are derived from a benchmark
    OD table containing the selected benchmark flow column.

    Returns canonical ``flow`` and ``flow_adj`` values with ``weight_ipf``.
    ``attrs["ipf_converged"]`` and ``attrs["ipf_iterations"]`` report the IPF
    state. Duplicate, negative, impossible, or unmatched targets and invalid
    controls fail closed. Parity covers audited IPF fixtures; clipping is an
    explicit post-fit constraint and can mark the result non-converged.
    """
    require_columns(mpd_od_df, ("origin", "destination", "flow"), "mpd_od_df")
    group_cols = require_group_columns(
        group_cols,
        {"mpd_od_df": mpd_od_df},
        reserved=("origin", "destination", "flow"),
    )
    max_iter = _validate_ipf_controls(max_iter, tol, clip_min, clip_max)
    flow_values = _validated_non_negative_counts(
        mpd_od_df["flow"],
        "mpd_od_df$flow",
    )

    if benchmark_od_df is not None:
        require_columns(
            benchmark_od_df,
            ("origin", "destination", flow_col_bench),
            "benchmark_od_df",
        )

    if benchmark_od_df is not None:
        missing_bench = [
            col for col in group_cols if col not in benchmark_od_df.columns
        ]
        if missing_bench:
            raise DebiasSchemaError(
                "`group_cols` missing in benchmark_od_df: " + ", ".join(missing_bench)
            )
        benchmark_flow = _validated_non_negative_counts(
            benchmark_od_df[flow_col_bench],
            f"benchmark_od_df${flow_col_bench}",
        )
        bench = pd.DataFrame(
            {
                "origin": r_character_key_series(benchmark_od_df["origin"]),
                "destination": r_character_key_series(benchmark_od_df["destination"]),
                "flow_bench": benchmark_flow,
            }
        )
        for col in group_cols:
            bench[col] = benchmark_od_df[col].to_numpy()

        if origin_targets is None:
            origin_targets = (
                bench.groupby(["origin", *group_cols], sort=True, dropna=False)[
                    "flow_bench"
                ]
                .sum()
                .reset_index()
                .rename(columns={"flow_bench": "target"})
            )
        if destination_targets is None:
            destination_targets = (
                bench.groupby(["destination", *group_cols], sort=True, dropna=False)[
                    "flow_bench"
                ]
                .sum()
                .reset_index()
                .rename(columns={"flow_bench": "target"})
            )

    if origin_targets is not None:
        require_columns(origin_targets, ("origin", "target"), "origin_targets")
        _require_target_groups(origin_targets, group_cols, "origin_targets")
        _validate_target_table(
            origin_targets, ["origin", *group_cols], "origin_targets"
        )
    if destination_targets is not None:
        require_columns(
            destination_targets,
            ("destination", "target"),
            "destination_targets",
        )
        _require_target_groups(destination_targets, group_cols, "destination_targets")
        _validate_target_table(
            destination_targets,
            ["destination", *group_cols],
            "destination_targets",
        )

    if origin_targets is None and destination_targets is None:
        raise DebiasSchemaError(
            "Provide at least one of `origin_targets`, "
            "`destination_targets`, or `benchmark_od_df`."
        )

    mpd_work = mpd_od_df.copy()
    mpd_work["flow"] = flow_values
    mpd_work["_debiaspy_origin_key"] = r_character_key_series(mpd_work["origin"])
    mpd_work["_debiaspy_destination_key"] = r_character_key_series(
        mpd_work["destination"]
    )
    _validate_target_support(
        origin_targets,
        mpd_work,
        group_cols,
        key_column="origin",
        work_key_column="_debiaspy_origin_key",
        frame_name="origin_targets",
    )
    _validate_target_support(
        destination_targets,
        mpd_work,
        group_cols,
        key_column="destination",
        work_key_column="_debiaspy_destination_key",
        frame_name="destination_targets",
    )

    out_parts: list[pd.DataFrame] = []
    converged: list[bool] = []
    iterations: list[int] = []

    if group_cols:
        grouped = mpd_work.groupby(group_cols, sort=True, dropna=False)
        mpd_groups = [group.copy() for _, group in grouped]
    else:
        mpd_groups = [mpd_work.copy()]

    for df_sub in mpd_groups:
        ot_sub = _subset_targets(
            origin_targets,
            df_sub,
            group_cols,
            key_column="origin",
            work_key_column="_debiaspy_origin_key",
        )
        dt_sub = _subset_targets(
            destination_targets,
            df_sub,
            group_cols,
            key_column="destination",
            work_key_column="_debiaspy_destination_key",
        )
        if ot_sub is None and dt_sub is None:
            raise DebiasSchemaError(
                "No origin or destination targets match an `mpd_od_df` group."
            )

        adjusted = _ipf_subset(df_sub, ot_sub, dt_sub, max_iter=max_iter, tol=tol)
        df_sub = df_sub.copy()
        raw_weight = pd.Series(adjusted["weight_ipf"], dtype=float)
        clipped_weight = raw_weight.clip(lower=clip_min, upper=clip_max)
        df_sub["weight_ipf"] = clipped_weight.to_numpy()
        flow = pd.to_numeric(df_sub["flow"], errors="coerce").to_numpy(dtype=float)
        raw_adjusted = np.asarray(adjusted["flow_adj"], dtype=float)
        df_sub["flow_adj"] = np.where(
            np.isfinite(clipped_weight.to_numpy()),
            flow * clipped_weight.to_numpy(),
            raw_adjusted,
        )
        finite_weight = np.isfinite(raw_weight.to_numpy())
        clipping_changed = bool(
            np.any(
                finite_weight
                & ~np.isclose(
                    raw_weight.to_numpy(),
                    clipped_weight.to_numpy(),
                    rtol=0,
                    atol=np.finfo(float).eps,
                )
            )
        )

        out_parts.append(df_sub)
        converged.append(bool(adjusted["converged"]) and not clipping_changed)
        iterations.append(int(adjusted["iterations"]))

    out = pd.concat(out_parts, ignore_index=True)
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
            "flow_adj",
            "weight_ipf",
        ],
    )
    result = out.loc[:, columns].reset_index(drop=True)
    result.attrs["ipf_converged"] = all(converged)
    result.attrs["ipf_iterations"] = max(iterations) if iterations else 0
    return result


def _require_target_groups(
    target: pd.DataFrame, group_cols: list[str], frame_name: str
) -> None:
    missing = [col for col in group_cols if col not in target.columns]
    if missing:
        raise DebiasSchemaError(
            f"`{frame_name}` missing group_cols: {', '.join(missing)}"
        )


def _validate_target_table(
    target: pd.DataFrame,
    key_columns: list[str],
    frame_name: str,
) -> None:
    canonical_keys = target.loc[:, key_columns].copy()
    canonical_keys[key_columns[0]] = r_character_key_series(
        canonical_keys[key_columns[0]]
    )
    if bool(canonical_keys.duplicated(key_columns, keep=False).any()):
        raise DebiasSchemaError(
            f"`{frame_name}` must contain at most one target per canonical "
            + "/".join(key_columns)
            + "."
        )
    if bool(
        target["target"].map(lambda value: isinstance(value, (bool, np.bool_))).any()
    ):
        raise DebiasSchemaError(
            f"`{frame_name}$target` must contain finite non-negative values or NA."
        )
    numeric = pd.to_numeric(target["target"], errors="coerce")
    supplied = target["target"].notna()
    invalid = supplied & (~np.isfinite(numeric) | (numeric < 0))
    if bool(invalid.any()):
        raise DebiasSchemaError(
            f"`{frame_name}$target` must contain finite non-negative values or NA."
        )


def _subset_targets(
    targets: pd.DataFrame | None,
    df_sub: pd.DataFrame,
    group_cols: list[str],
    key_column: str,
    work_key_column: str,
) -> pd.DataFrame | None:
    if targets is None:
        return None
    subset = targets
    if group_cols:
        key_vals = df_sub.iloc[0][group_cols]
        for col in group_cols:
            value = key_vals[col]
            subset = subset.loc[
                subset[col].isna() if pd.isna(value) else subset[col] == value
            ]
    target_keys = r_character_key_series(subset[key_column])
    subset = subset.loc[target_keys.isin(df_sub[work_key_column])]
    if subset.empty:
        return None
    return subset.copy()


def _ipf_subset(
    df_sub: pd.DataFrame,
    origin_targets: pd.DataFrame | None,
    destination_targets: pd.DataFrame | None,
    max_iter: int,
    tol: float,
) -> dict[str, object]:
    mat = pd.to_numeric(df_sub["flow"], errors="coerce").to_numpy(dtype=float).copy()
    origins = df_sub["_debiaspy_origin_key"].to_numpy()
    destinations = df_sub["_debiaspy_destination_key"].to_numpy()

    origin_levels = list(pd.unique(origins))
    destination_levels = list(pd.unique(destinations))
    rows_by_origin = {
        origin: np.where(origins == origin)[0] for origin in origin_levels
    }
    rows_by_destination = {
        destination: np.where(destinations == destination)[0]
        for destination in destination_levels
    }

    origin_target_map = _target_map(origin_targets, "origin")
    destination_target_map = _target_map(destination_targets, "destination")
    origin_target_values = [
        origin_target_map.get(origin, np.nan) for origin in origin_levels
    ]
    destination_target_values = [
        destination_target_map.get(destination, np.nan)
        for destination in destination_levels
    ]
    if not any(np.isfinite([*origin_target_values, *destination_target_values])):
        raise DebiasSchemaError(
            "No finite origin or destination target is active for an `mpd_od_df` group."
        )

    converged = False
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        max_rel_dev = 0.0

        if any(np.isfinite(origin_target_values)):
            for origin, target in zip(origin_levels, origin_target_values, strict=True):
                if not np.isfinite(target):
                    continue
                idx = rows_by_origin[origin]
                if not np.isfinite(mat[idx]).all():
                    raise DebiasSchemaError(
                        "`mpd_od_df$flow` contains non-finite values in an "
                        "active origin margin."
                    )
                current_sum = float(np.sum(mat[idx]))
                if current_sum > 0:
                    factor = float(target) / current_sum
                    mat[idx] = mat[idx] * factor
                    max_rel_dev = max(max_rel_dev, abs(factor - 1))
                elif target > tol:
                    raise DebiasSchemaError(
                        "An active origin margin has zero support but a positive "
                        "target; IPF cannot create mass in structural-zero cells."
                    )

        if any(np.isfinite(destination_target_values)):
            for destination, target in zip(
                destination_levels, destination_target_values, strict=True
            ):
                if not np.isfinite(target):
                    continue
                idx = rows_by_destination[destination]
                if not np.isfinite(mat[idx]).all():
                    raise DebiasSchemaError(
                        "`mpd_od_df$flow` contains non-finite values in an "
                        "active destination margin."
                    )
                current_sum = float(np.sum(mat[idx]))
                if current_sum > 0:
                    factor = float(target) / current_sum
                    mat[idx] = mat[idx] * factor
                    max_rel_dev = max(max_rel_dev, abs(factor - 1))
                elif target > tol:
                    raise DebiasSchemaError(
                        "An active destination margin has zero support but a "
                        "positive target; IPF cannot create mass in structural-zero "
                        "cells."
                    )

        if max_rel_dev < tol:
            converged = True
            break

    flow = pd.to_numeric(df_sub["flow"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = mat / flow
    weight[~np.isfinite(weight)] = np.nan

    return {
        "flow_adj": mat,
        "weight_ipf": weight,
        "converged": converged,
        "iterations": iteration,
    }


def _target_map(targets: pd.DataFrame | None, key_column: str) -> dict[object, float]:
    if targets is None:
        return {}
    deduped = targets.drop_duplicates(subset=[key_column], keep="first")
    values = pd.to_numeric(deduped["target"], errors="coerce")
    keys = r_character_key_series(deduped[key_column])
    return dict(zip(keys, values, strict=True))


def _support_value(value: object) -> tuple[str, object]:
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ("missing", "")
    return ("value", value)


def _validate_target_support(
    targets: pd.DataFrame | None,
    mpd_work: pd.DataFrame,
    group_cols: list[str],
    *,
    key_column: str,
    work_key_column: str,
    frame_name: str,
) -> None:
    if targets is None:
        return
    mpd_keys = {
        tuple([_support_value(key), *[_support_value(row[col]) for col in group_cols]])
        for key, (_, row) in zip(
            mpd_work[work_key_column], mpd_work.iterrows(), strict=True
        )
    }
    target_keys = r_character_key_series(targets[key_column])
    unsupported = [
        tuple([_support_value(key), *[_support_value(row[col]) for col in group_cols]])
        not in mpd_keys
        for key, (_, row) in zip(target_keys, targets.iterrows(), strict=True)
    ]
    if any(unsupported):
        raise DebiasSchemaError(
            f"`{frame_name}` contains target key(s) with no matching "
            "`mpd_od_df` support."
        )


def _validate_ipf_controls(
    max_iter: int | float,
    tol: float,
    clip_min: float,
    clip_max: float,
) -> int:
    if isinstance(max_iter, (bool, np.bool_)):
        raise DebiasSchemaError("`max_iter` must be a positive whole number.")
    try:
        max_iter_value = float(max_iter)
    except (TypeError, ValueError) as exc:
        raise DebiasSchemaError("`max_iter` must be a positive whole number.") from exc
    if (
        not np.isfinite(max_iter_value)
        or not max_iter_value.is_integer()
        or max_iter_value < 1
    ):
        raise DebiasSchemaError("`max_iter` must be a positive whole number.")
    if any(isinstance(value, (bool, np.bool_)) for value in (tol, clip_min, clip_max)):
        raise DebiasSchemaError(
            "`tol`, `clip_min`, and `clip_max` must be valid numeric bounds."
        )
    try:
        numeric = np.asarray([tol, clip_min, clip_max], dtype=float)
    except (TypeError, ValueError) as exc:
        raise DebiasSchemaError(
            "`tol`, `clip_min`, and `clip_max` must be valid numeric bounds."
        ) from exc
    if not np.isfinite(numeric[:2]).all() or np.isnan(numeric[2]):
        raise DebiasSchemaError(
            "`tol`, `clip_min`, and `clip_max` must be valid numeric bounds."
        )
    if tol <= 0:
        raise DebiasSchemaError("`tol` must be positive.")
    if clip_min < 0 or clip_max < clip_min:
        raise DebiasSchemaError(
            "`clip_min` must be non-negative and no greater than `clip_max`."
        )
    return int(max_iter_value)


def _validated_non_negative_counts(
    values: pd.Series,
    label: str,
) -> pd.Series:
    message = (
        f"`{label}` must contain finite non-negative counts; "
        "non-finite values are not allowed."
    )
    if bool(values.map(lambda value: isinstance(value, (bool, np.bool_))).any()):
        raise DebiasSchemaError(message)
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if not bool((np.isfinite(numeric) & (numeric >= 0)).all()):
        raise DebiasSchemaError(message)
    return numeric
