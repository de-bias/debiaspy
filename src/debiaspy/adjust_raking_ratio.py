"""Raking-ratio / IPF adjustment of origin-destination flows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def adjust_raking_ratio(
    mpd_od_df: pd.DataFrame,
    origin_targets: pd.DataFrame | None = None,
    destination_targets: pd.DataFrame | None = None,
    benchmark_od_df: pd.DataFrame | None = None,
    flow_col_bench: str = "flow",
    group_cols: str | Sequence[str] | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    clip_min: float = 0.0,
    clip_max: float = np.inf,
    keep_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Adjust OD flows so selected margins match target totals via IPF.

    Parameters
    ----------
    mpd_od_df:
        DataFrame with columns ``origin``, ``destination`` and ``flow``. An
        ``mpd_source`` column is carried through when present. Stratification
        columns named in ``group_cols`` may also be present.
    origin_targets:
        Optional DataFrame with ``origin``, ``target`` and any ``group_cols``.
    destination_targets:
        Optional DataFrame with ``destination``, ``target`` and any
        ``group_cols``.
    benchmark_od_df:
        Optional benchmark OD used to derive omitted target tables from
        ``flow_col_bench``. An explicit target table overrides the matching
        benchmark margin.
    flow_col_bench:
        Name of the benchmark flow column. Default ``"flow"``.
    group_cols:
        Optional stratification column name or sequence of names. IPF runs
        independently within each stratum. Missing values in these columns
        are rejected with ``ValueError``; they are not treated as a valid
        stratum key.
    max_iter:
        Maximum IPF iterations. Must be a positive integer.
    tol:
        Convergence tolerance on the largest absolute scale-factor change.
    clip_min, clip_max:
        Bounds applied to ``weight_ipf`` after IPF. Adjusted flows are then
        recomputed as ``flow_adj = flow * weight_ipf``.
    keep_cols:
        Optional extra columns from ``mpd_od_df`` to retain in the output.

    Returns
    -------
    pandas.DataFrame
        A copy with ``flow``, ``flow_adj`` and ``weight_ipf``, plus any carried
        identifier columns. Attributes ``ipf_converged`` and ``ipf_iterations``
        summarise convergence across strata.
    """
    _validate_controls(max_iter, tol, clip_min, clip_max)
    _require_dataframe(mpd_od_df, "mpd_od_df")
    _require_columns(mpd_od_df, ["origin", "destination", "flow"], "mpd_od_df")
    _require_non_negative_finite(mpd_od_df["flow"], "mpd_od_df['flow']")

    groups = _normalise_group_cols(group_cols)
    if groups:
        missing_groups = [col for col in groups if col not in mpd_od_df.columns]
        if missing_groups:
            missing = ", ".join(missing_groups)
            raise ValueError(f"`group_cols` missing in mpd_od_df: {missing}")
        _require_non_missing_group_values(mpd_od_df, groups, "mpd_od_df")

    keep = _normalise_keep_cols(keep_cols)

    origin_targets = None if origin_targets is None else origin_targets.copy()
    destination_targets = (
        None if destination_targets is None else destination_targets.copy()
    )

    if benchmark_od_df is not None:
        origin_targets, destination_targets = _derive_targets_from_benchmark(
            benchmark_od_df=benchmark_od_df,
            flow_col_bench=flow_col_bench,
            group_cols=groups,
            origin_targets=origin_targets,
            destination_targets=destination_targets,
        )

    if origin_targets is None and destination_targets is None:
        raise ValueError(
            "Provide at least one of `origin_targets`, `destination_targets`, "
            "or `benchmark_od_df`."
        )

    if origin_targets is not None:
        origin_targets = _validate_target_table(
            origin_targets,
            area_col="origin",
            group_cols=groups,
            name="origin_targets",
        )
        _require_targets_supported(
            origin_targets,
            mpd_od_df,
            area_col="origin",
            group_cols=groups,
            name="origin_targets",
        )

    if destination_targets is not None:
        destination_targets = _validate_target_table(
            destination_targets,
            area_col="destination",
            group_cols=groups,
            name="destination_targets",
        )
        _require_targets_supported(
            destination_targets,
            mpd_od_df,
            area_col="destination",
            group_cols=groups,
            name="destination_targets",
        )

    working = mpd_od_df.copy()
    working["_row_id"] = np.arange(len(working), dtype=np.intp)
    working["_origin_key"] = working["origin"].map(_as_area_key)
    working["_destination_key"] = working["destination"].map(_as_area_key)

    if origin_targets is not None:
        origin_targets = origin_targets.copy()
        origin_targets["_origin_key"] = origin_targets["origin"].map(_as_area_key)
    if destination_targets is not None:
        destination_targets = destination_targets.copy()
        destination_targets["_destination_key"] = destination_targets[
            "destination"
        ].map(_as_area_key)

    adjusted_parts: list[pd.DataFrame] = []
    converged_flags: list[bool] = []
    iteration_counts: list[int] = []

    if groups:
        grouped = working.groupby(list(groups), dropna=False, sort=False)
        for _, df_sub in grouped:
            ot_sub = _subset_targets(origin_targets, df_sub, groups, "origin")
            dt_sub = _subset_targets(destination_targets, df_sub, groups, "destination")
            part, converged, iterations = _ipf_subset(
                df_sub,
                ot_sub,
                dt_sub,
                max_iter=max_iter,
                tol=tol,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            adjusted_parts.append(part)
            converged_flags.append(converged)
            iteration_counts.append(iterations)
    else:
        part, converged, iterations = _ipf_subset(
            working,
            origin_targets,
            destination_targets,
            max_iter=max_iter,
            tol=tol,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        adjusted_parts.append(part)
        converged_flags.append(converged)
        iteration_counts.append(iterations)

    if not adjusted_parts:
        out = _empty_result(working, groups, keep)
        out.attrs["ipf_converged"] = True
        out.attrs["ipf_iterations"] = 0
        return out

    out = pd.concat(adjusted_parts, axis=0, ignore_index=True)
    out = out.sort_values("_row_id", kind="mergesort").reset_index(drop=True)
    out = _select_output_columns(out, groups, keep)
    out.attrs["ipf_converged"] = bool(all(converged_flags))
    out.attrs["ipf_iterations"] = int(max(iteration_counts)) if iteration_counts else 0
    return out


def _validate_controls(
    max_iter: Any,
    tol: Any,
    clip_min: Any,
    clip_max: Any,
) -> None:
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
        raise ValueError("`max_iter` must be a positive integer.")
    if int(max_iter) != max_iter or max_iter <= 0:
        raise ValueError("`max_iter` must be a positive integer.")

    try:
        tol_value = float(tol)
    except (TypeError, ValueError) as exc:
        raise ValueError("`tol` must be a positive finite number.") from exc
    if not np.isfinite(tol_value) or tol_value <= 0:
        raise ValueError("`tol` must be a positive finite number.")

    try:
        clip_min_value = float(clip_min)
    except (TypeError, ValueError) as exc:
        raise ValueError("`clip_min` must be a finite non-negative number.") from exc
    if not np.isfinite(clip_min_value) or clip_min_value < 0:
        raise ValueError("`clip_min` must be a finite non-negative number.")

    try:
        clip_max_value = float(clip_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "`clip_max` must be finite or +inf and at least `clip_min`."
        ) from exc
    if np.isnan(clip_max_value) or clip_max_value < clip_min_value:
        raise ValueError("`clip_max` must be finite or +inf and at least `clip_min`.")
    if np.isneginf(clip_max_value):
        raise ValueError("`clip_max` must be finite or +inf and at least `clip_min`.")


def _require_dataframe(value: object, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"`{name}` must be a pandas DataFrame.")


def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"`{name}` must contain columns: {missing_text}")


def _require_non_negative_finite(series: pd.Series, name: str) -> None:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain finite non-negative values.")
    if values.lt(0).any():
        raise ValueError(f"{name} must contain finite non-negative values.")


def _normalise_group_cols(group_cols: str | Sequence[str] | None) -> tuple[str, ...]:
    if group_cols is None:
        return ()
    if isinstance(group_cols, str):
        return (group_cols,)
    cols = tuple(group_cols)
    if any(not isinstance(col, str) or not col for col in cols):
        raise ValueError("`group_cols` must be a string or a sequence of column names.")
    return cols


def _normalise_keep_cols(keep_cols: Sequence[str]) -> tuple[str, ...]:
    if isinstance(keep_cols, str):
        return (keep_cols,)
    cols = tuple(keep_cols)
    if any(not isinstance(col, str) for col in cols):
        raise ValueError("`keep_cols` must be a sequence of column names.")
    return cols


def _as_area_key(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Area identifiers must be non-missing.")
    return str(value)


def _derive_targets_from_benchmark(
    benchmark_od_df: pd.DataFrame,
    flow_col_bench: str,
    group_cols: tuple[str, ...],
    origin_targets: pd.DataFrame | None,
    destination_targets: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    _require_dataframe(benchmark_od_df, "benchmark_od_df")
    required = ["origin", "destination", flow_col_bench]
    _require_columns(benchmark_od_df, required, "benchmark_od_df")
    if group_cols:
        missing = [col for col in group_cols if col not in benchmark_od_df.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"`group_cols` missing in benchmark_od_df: {missing_text}")
        _require_non_missing_group_values(
            benchmark_od_df, group_cols, "benchmark_od_df"
        )

    _require_non_negative_finite(
        benchmark_od_df[flow_col_bench],
        f"benchmark_od_df['{flow_col_bench}']",
    )

    bench = pd.DataFrame(
        {
            "origin": benchmark_od_df["origin"].map(_as_area_key),
            "destination": benchmark_od_df["destination"].map(_as_area_key),
            "flow_bench": pd.to_numeric(
                benchmark_od_df[flow_col_bench], errors="coerce"
            ),
        }
    )
    for col in group_cols:
        bench[col] = benchmark_od_df[col].to_numpy()

    if origin_targets is None:
        origin_group = ["origin", *group_cols]
        origin_targets = (
            bench.groupby(origin_group, dropna=False, sort=False)["flow_bench"]
            .sum()
            .reset_index(name="target")
        )

    if destination_targets is None:
        destination_group = ["destination", *group_cols]
        destination_targets = (
            bench.groupby(destination_group, dropna=False, sort=False)["flow_bench"]
            .sum()
            .reset_index(name="target")
        )

    return origin_targets, destination_targets


def _require_non_missing_group_values(
    df: pd.DataFrame,
    group_cols: tuple[str, ...],
    name: str,
) -> None:
    for col in group_cols:
        if df[col].isna().any():
            raise ValueError(
                f"`{name}` has missing values in group column `{col}`; "
                "missing stratum keys are not supported."
            )


def _validate_target_table(
    targets: pd.DataFrame,
    area_col: str,
    group_cols: tuple[str, ...],
    name: str,
) -> pd.DataFrame:
    _require_dataframe(targets, name)
    _require_columns(targets, [area_col, "target", *group_cols], name)
    _require_non_negative_finite(targets["target"], f"{name}['target']")
    if group_cols:
        _require_non_missing_group_values(targets, group_cols, name)

    # Duplicates must be assessed after area-key canonicalisation so that
    # values such as 1 and "1" are treated as the same area.
    key_frame = pd.DataFrame(
        {
            "_area_key": targets[area_col].map(_as_area_key),
            **{col: targets[col].to_numpy() for col in group_cols},
        }
    )
    if key_frame.duplicated().any():
        raise ValueError(f"`{name}` contains duplicate target keys.")
    return targets


def _require_targets_supported(
    targets: pd.DataFrame,
    mpd_od_df: pd.DataFrame,
    area_col: str,
    group_cols: tuple[str, ...],
    name: str,
) -> None:
    if targets.empty:
        return

    od_keys = pd.DataFrame(
        {
            area_col: mpd_od_df[area_col].map(_as_area_key),
            **{col: mpd_od_df[col].to_numpy() for col in group_cols},
        }
    ).drop_duplicates()
    target_keys = pd.DataFrame(
        {
            area_col: targets[area_col].map(_as_area_key),
            **{col: targets[col].to_numpy() for col in group_cols},
        }
    ).drop_duplicates()

    merged = target_keys.merge(
        od_keys,
        on=[area_col, *group_cols],
        how="left",
        indicator=True,
    )
    if (merged["_merge"] == "left_only").any():
        raise ValueError(f"`{name}` names an area or stratum absent from `mpd_od_df`.")


def _subset_targets(
    targets: pd.DataFrame | None,
    df_sub: pd.DataFrame,
    group_cols: tuple[str, ...],
    area_col: str,
) -> pd.DataFrame | None:
    if targets is None:
        return None

    subset = targets
    if group_cols:
        key_vals = df_sub.iloc[0]
        for col in group_cols:
            subset = subset.loc[subset[col] == key_vals[col]]

    area_key = f"_{area_col}_key"
    present = set(df_sub[area_key])
    subset = subset.loc[subset[area_key].isin(present)]
    if subset.empty:
        return None
    return subset


def _ipf_subset(
    df_sub: pd.DataFrame,
    origin_targets: pd.DataFrame | None,
    destination_targets: pd.DataFrame | None,
    max_iter: int,
    tol: float,
    clip_min: float,
    clip_max: float,
) -> tuple[pd.DataFrame, bool, int]:
    mat = pd.to_numeric(df_sub["flow"], errors="coerce").to_numpy(dtype=float).copy()
    origin_keys = df_sub["_origin_key"].to_numpy()
    destination_keys = df_sub["_destination_key"].to_numpy()

    o_levels = pd.unique(origin_keys)
    d_levels = pd.unique(destination_keys)
    o_index = pd.Index(o_levels).get_indexer(origin_keys)
    d_index = pd.Index(d_levels).get_indexer(destination_keys)

    rows_by_o = [np.flatnonzero(o_index == i) for i in range(len(o_levels))]
    rows_by_d = [np.flatnonzero(d_index == j) for j in range(len(d_levels))]

    ot_vec = np.full(len(o_levels), np.nan, dtype=float)
    if origin_targets is not None:
        ot_map = dict(
            zip(
                origin_targets["_origin_key"].to_numpy(),
                pd.to_numeric(origin_targets["target"], errors="coerce").to_numpy(
                    dtype=float
                ),
                strict=True,
            )
        )
        for i, key in enumerate(o_levels):
            if key in ot_map:
                ot_vec[i] = ot_map[key]

    dt_vec = np.full(len(d_levels), np.nan, dtype=float)
    if destination_targets is not None:
        dt_map = dict(
            zip(
                destination_targets["_destination_key"].to_numpy(),
                pd.to_numeric(destination_targets["target"], errors="coerce").to_numpy(
                    dtype=float
                ),
                strict=True,
            )
        )
        for j, key in enumerate(d_levels):
            if key in dt_map:
                dt_vec[j] = dt_map[key]

    for i, target_i in enumerate(ot_vec):
        if np.isfinite(target_i) and target_i > 0 and mat[rows_by_o[i]].sum() <= 0:
            raise ValueError(
                "Positive origin target has zero flow support; "
                "IPF cannot create a new OD cell."
            )
    for j, target_j in enumerate(dt_vec):
        if np.isfinite(target_j) and target_j > 0 and mat[rows_by_d[j]].sum() <= 0:
            raise ValueError(
                "Positive destination target has zero flow support; "
                "IPF cannot create a new OD cell."
            )

    converged = False
    iterations = 0
    use_origin = np.isfinite(ot_vec).any()
    use_destination = np.isfinite(dt_vec).any()

    while iterations < max_iter:
        iterations += 1
        max_rel_dev = 0.0

        if use_origin:
            for i, target_i in enumerate(ot_vec):
                if not np.isfinite(target_i):
                    continue
                idx = rows_by_o[i]
                current_sum = mat[idx].sum()
                if current_sum > 0:
                    factor = target_i / current_sum
                    mat[idx] *= factor
                    max_rel_dev = max(max_rel_dev, abs(factor - 1.0))

        if use_destination:
            for j, target_j in enumerate(dt_vec):
                if not np.isfinite(target_j):
                    continue
                idx = rows_by_d[j]
                current_sum = mat[idx].sum()
                if current_sum > 0:
                    factor = target_j / current_sum
                    mat[idx] *= factor
                    max_rel_dev = max(max_rel_dev, abs(factor - 1.0))

        if max_rel_dev < tol:
            converged = True
            break

    observed = pd.to_numeric(df_sub["flow"], errors="coerce").to_numpy(dtype=float)
    weight = np.ones(len(mat), dtype=float)
    positive = observed > 0
    weight[positive] = mat[positive] / observed[positive]
    weight[~np.isfinite(weight)] = np.nan

    clipped = np.clip(weight, clip_min, clip_max)
    finite_before = np.isfinite(weight)
    clip_changed = bool(
        np.any(finite_before & (clipped[finite_before] != weight[finite_before]))
    )
    if clip_changed:
        converged = False

    flow_adj = observed * clipped
    result = df_sub.copy()
    result["flow_adj"] = flow_adj
    result["weight_ipf"] = clipped
    return result, converged, iterations


def _select_output_columns(
    out: pd.DataFrame,
    group_cols: tuple[str, ...],
    keep_cols: tuple[str, ...],
) -> pd.DataFrame:
    structural = {
        "origin",
        "destination",
        "mpd_source",
        *group_cols,
        "flow",
        "flow_adj",
        "weight_ipf",
    }
    selected_keep = [
        col for col in keep_cols if col in out.columns and col not in structural
    ]
    ordered: list[str] = []
    for col in ("origin", "destination", "mpd_source", *group_cols):
        if col in out.columns and col not in ordered:
            ordered.append(col)
    ordered.extend(selected_keep)
    ordered.extend(["flow", "flow_adj", "weight_ipf"])
    return out.loc[:, ordered].copy()


def _empty_result(
    working: pd.DataFrame,
    group_cols: tuple[str, ...],
    keep_cols: tuple[str, ...],
) -> pd.DataFrame:
    working = working.copy()
    working["flow_adj"] = pd.Series(dtype=float)
    working["weight_ipf"] = pd.Series(dtype=float)
    return _select_output_columns(working, group_cols, keep_cols)


__all__ = ["adjust_raking_ratio"]
