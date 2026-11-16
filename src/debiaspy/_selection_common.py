"""Private tabular operations shared by the selection-rate methods."""

import warnings
from collections.abc import Callable, Sequence
from decimal import Decimal
from numbers import Real

import numpy as np
import pandas as pd

from debiaspy.measure_bias import (
    _require_columns,
    _require_dataframe,
    _require_numeric_columns,
    _require_string_column_names,
)


def _check_frame(frame: pd.DataFrame, name: str) -> None:
    _require_dataframe(frame, name)
    if not frame.columns.is_unique:
        raise ValueError(f"`{name}` must have unique column names.")


def _check_inputs(mpd: pd.DataFrame, coverage: pd.DataFrame) -> str:
    _check_frame(mpd, "mpd_od_df")
    _check_frame(coverage, "coverage_df")
    _require_columns(mpd, ["origin", "destination", "flow", "mpd_source"], "mpd_od_df")
    _require_numeric_columns(mpd, ["flow"], "mpd_od_df")
    if pd.api.types.is_complex_dtype(mpd["flow"]):
        raise ValueError("`mpd_od_df` flow must be real-valued.")
    for name in ("weight_origin", "weight_destination"):
        if name in mpd:
            raise ValueError(f"`mpd_od_df` contains conflicting column `{name}`.")
    new = [
        "origin",
        "origin_population",
        "origin_user_count",
        "destination",
        "destination_population",
        "destination_user_count",
        "mpd_source",
    ]
    if all(name in coverage for name in new):
        return "new"
    if all(
        name in coverage
        for name in ["origin", "population", "user_count", "mpd_source"]
    ):
        return "legacy"
    raise ValueError(
        "`coverage_df` does not match a supported schema. New: origin, "
        "origin_population, origin_user_count, destination, destination_population, "
        "destination_user_count, mpd_source. Legacy: origin, population, "
        "user_count, mpd_source."
    )


def _choice(value: str | None, choices: tuple[str, ...], name: str) -> str:
    if value is None:
        return choices[0]
    matches = [
        x for x in choices if isinstance(value, str) and value and x.startswith(value)
    ]
    if len(matches) != 1:
        raise ValueError(f"`{name}` must be one of: {', '.join(choices)}.")
    return matches[0]


def _names(value: str | Sequence[str] | None, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise TypeError(f"`{name}` must contain column names.") from exc
    _require_string_column_names(*values)
    return list(dict.fromkeys(values))


def _area_strings(values: pd.Series) -> pd.Series:
    """Convert identifiers without turning missing values into literal text."""

    def convert(value: object) -> object:
        if pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return "TRUE" if value else "FALSE"
        if isinstance(value, Real):
            if np.isposinf(value):
                return "Inf"
            if np.isneginf(value):
                return "-Inf"
            if value == 0:
                return "0"
            # as.character() chooses the shorter fixed/scientific representation
            # at 15 significant digits (fixed on ties), unlike Python's str().
            fixed = format(Decimal(format(value, ".15g")), "f")
            mantissa, exponent = format(value, ".14e").split("e")
            scientific = mantissa.rstrip("0").rstrip(".") + "e" + exponent
            return scientific if len(scientific) < len(fixed) else fixed
        return str(value)

    return values.map(convert).astype(object)


def _parse_numeric_text(value: str) -> float:
    text = value.strip()
    if not text:
        return np.nan
    if "_" in text:
        raise ValueError("Underscores are not supported in numeric text.")
    if text.lower().lstrip("+-").startswith("0x"):
        try:
            return float.fromhex(text)
        except OverflowError:
            return -np.inf if text.startswith("-") else np.inf
    return float(text)


def _numeric(values: pd.Series, name: str, *, warn: bool = True) -> pd.Series:
    if pd.api.types.is_complex_dtype(values):
        raise ValueError(f"`{name}` must be real-valued.")
    converted = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan, copy=True
    )
    failed = values.notna().to_numpy(dtype=bool) & np.isnan(converted)
    invalid = False
    # R accepts hexadecimal numbers and treats blank or NaN text as missing.
    for position in np.flatnonzero(failed):
        value = values.iloc[position]
        if isinstance(value, str):
            try:
                converted[position] = _parse_numeric_text(value)
                continue
            except ValueError:
                pass
        invalid = True
    if warn and invalid:
        warnings.warn(
            f"NAs introduced by coercion in `{name}`.", UserWarning, stacklevel=3
        )
    return pd.Series(converted, index=values.index)


def _coverage_side(
    coverage: pd.DataFrame,
    schema: str,
    side: str,
    groups: list[str],
    *,
    zero_users: bool,
) -> pd.DataFrame:
    key = side if schema == "new" else "origin"
    population = f"{side}_population" if schema == "new" else "population"
    users = f"{side}_user_count" if schema == "new" else "user_count"
    result = coverage.loc[:, [key, "mpd_source", *groups]].copy()
    result = result.rename(columns={key: side})
    result[side] = _area_strings(result[side])
    result["P"] = _numeric(coverage[population], population)
    result["U"] = _numeric(coverage[users], users)
    valid = np.isfinite(result["P"]) & result["P"].gt(0) & np.isfinite(result["U"])
    valid &= result["U"].ge(0) if zero_users else result["U"].gt(0)
    return result.loc[valid].copy()


def _join(
    left: pd.DataFrame, right: pd.DataFrame, keys: list[str], *, how: str = "left"
) -> pd.DataFrame:
    left_many = left.duplicated(keys, keep=False)
    right_many = right.duplicated(keys, keep=False)
    if left_many.any() and right_many.any():
        left_keys = left.loc[:, keys].drop_duplicates()
        right_keys = right.loc[:, keys].drop_duplicates()
        # dplyr's two multiply matched rows need not belong to the same key.
        repeated_left_matches = (
            left.loc[left_many, keys]
            .drop_duplicates()
            .merge(right_keys, on=keys, how="inner", sort=False)
        )
        repeated_right_matches = (
            right.loc[right_many, keys]
            .drop_duplicates()
            .merge(left_keys, on=keys, how="inner", sort=False)
        )
        if not repeated_left_matches.empty and not repeated_right_matches.empty:
            warnings.warn(
                "Detected an unexpected many-to-many relationship "
                "between the joined tables.",
                UserWarning,
                stacklevel=3,
            )
    return left.merge(right, on=keys, how=how, sort=False)


def _bounds(lower: float, upper: float) -> tuple[float, float]:
    for value in (lower, upper):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(
                "`clip_min` and `clip_max` must be non-negative numeric bounds."
            )
    if not np.isfinite(lower) or np.isnan(upper) or lower < 0 or upper < lower:
        raise ValueError(
            "`clip_min` and `clip_max` must be ordered non-negative bounds."
        )
    return float(lower), float(upper)


def _scalar(value: float, name: str, *, positive: bool) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        domain = "positive" if positive else "non-negative"
        raise ValueError(f"`{name}` must be a single finite {domain} number.")
    return float(value)


def _clamp(values: pd.Series, lower: float, upper: float) -> pd.Series:
    return values.where(np.isfinite(values) & values.gt(0)).clip(lower, upper)


def _adjust(
    mpd: pd.DataFrame,
    weights: dict[str, pd.DataFrame],
    mode: str,
    groups: list[str],
    keep: list[str],
) -> pd.DataFrame:
    out = mpd.copy(deep=True)
    for side in ("origin", "destination"):
        if side in weights:
            out = _join(out, weights[side], [side, "mpd_source", *groups])
        else:
            out[f"weight_{side}"] = np.nan
    if mode == "both":
        with np.errstate(invalid="ignore", over="ignore"):
            final = np.sqrt(out["weight_origin"] * out["weight_destination"])
        final = final.where(
            np.isfinite(out["weight_origin"]) & np.isfinite(out["weight_destination"])
        )
    else:
        final = out[f"weight_{mode}"]
    out["weight_missing"] = ~np.isfinite(final)
    with np.errstate(invalid="ignore", over="ignore"):
        out["flow_adj"] = out["flow"].to_numpy(dtype=float, na_value=np.nan) * final
    out.loc[out["weight_missing"], "flow_adj"] = np.nan
    columns = list(
        dict.fromkeys(
            [
                "origin",
                "destination",
                "mpd_source",
                *groups,
                *(name for name in keep if name in mpd),
                "flow",
                "weight_origin",
                "weight_destination",
                "weight_missing",
                "flow_adj",
            ]
        )
    )
    return out.loc[:, columns].reset_index(drop=True)


def _calibrate(
    compute: Callable[[float], pd.DataFrame],
    benchmark: pd.DataFrame,
    flow_name: str,
    grid: Sequence[float] | np.ndarray,
    aggregate: str,
    parameter: str,
) -> tuple[float, pd.DataFrame]:
    _check_frame(benchmark, "benchmark_od_df")
    _require_string_column_names(flow_name)
    _require_columns(benchmark, ["origin", "destination", flow_name], "benchmark_od_df")
    bench = benchmark.loc[:, ["origin", "destination", flow_name]].copy()
    bench.columns = ["origin", "destination", "flow_bench"]
    for side in ("origin", "destination"):
        bench[side] = _area_strings(bench[side])
    bench["flow_bench"] = _numeric(bench["flow_bench"], flow_name)
    bench = bench.loc[np.isfinite(bench["flow_bench"])]
    if bench.empty:
        raise ValueError(
            "`benchmark_od_df` has no valid benchmark flows for calibration."
        )
    try:
        candidates = np.asarray(grid, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{parameter}_grid` must be a numeric vector.") from exc
    if candidates.ndim != 1:
        raise ValueError(
            f"`{parameter}_grid` must be a one-dimensional numeric vector."
        )
    if not len(candidates):
        raise ValueError(f"`{parameter}_grid` is empty; provide candidate values.")
    if parameter == "r" and (
        not np.isfinite(candidates).all() or (candidates < 0).any()
    ):
        raise ValueError("`r_grid` must contain finite non-negative values.")
    losses = []
    used = []
    for candidate in candidates:
        if parameter == "k" and (not np.isfinite(candidate) or candidate <= 0):
            continue
        adjusted = compute(float(candidate))
        # Fit a single parameter across sources and groups by matching OD pairs.
        joined = _join(
            adjusted[["origin", "destination", "flow_adj"]],
            bench,
            ["origin", "destination"],
            how="inner",
        )
        if joined.empty:
            continue
        if aggregate == "origin":
            joined = joined.groupby("origin", sort=False, dropna=False)[
                ["flow_adj", "flow_bench"]
            ].sum()
        losses.append(float((joined["flow_adj"] - joined["flow_bench"]).abs().sum()))
        used.append(float(candidate))
    if not used:
        raise ValueError(
            "Calibration failed: no valid candidates or no overlap "
            "between adjusted and benchmark ODs."
        )
    return used[int(np.argmin(losses))], pd.DataFrame({parameter: used, "loss": losses})
