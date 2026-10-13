from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd

from debiaspy._choices import match_choice as match_choice
from debiaspy._exceptions import DebiasSchemaError


def numeric_series(values: Any) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


def column_names(columns: str | Iterable[str] | None) -> list[str]:
    """Normalize an R-style character vector of column names.

    A single string is one column name, not an iterable of characters.  Repeated
    names are removed in first-seen order, matching tidyselect's output contract.
    """
    if columns is None:
        return []
    values = [columns] if isinstance(columns, str) else list(columns)
    return list(dict.fromkeys(values))


def existing_columns(
    frame: pd.DataFrame,
    columns: str | Iterable[str] | None,
) -> list[str]:
    """Return existing columns once, preserving requested order."""
    return [column for column in column_names(columns) if column in frame.columns]


def require_no_reserved_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    frame_name: str,
) -> None:
    collisions = [column for column in column_names(columns) if column in frame.columns]
    if collisions:
        raise DebiasSchemaError(
            f"`{frame_name}` contains reserved adjustment column(s): "
            + ", ".join(collisions)
        )


def validate_clip_bounds(clip_min: Any, clip_max: Any) -> tuple[float, float]:
    """Validate non-negative scalar clipping bounds used by weight methods."""
    if isinstance(clip_min, (bool, np.bool_)) or isinstance(clip_max, (bool, np.bool_)):
        raise DebiasSchemaError(
            "`clip_min` and `clip_max` must be valid non-negative numeric bounds."
        )
    try:
        lower = float(clip_min)
        upper = float(clip_max)
    except (TypeError, ValueError) as exc:
        raise DebiasSchemaError(
            "`clip_min` and `clip_max` must be valid non-negative numeric bounds."
        ) from exc
    if not np.isfinite(lower) or np.isnan(upper) or lower < 0 or upper < lower:
        raise DebiasSchemaError(
            "`clip_min` and `clip_max` must be valid non-negative numeric bounds."
        )
    return lower, upper


def clamp_weight(values: Any, clip_min: float, clip_max: float) -> pd.Series:
    weight = numeric_series(values).astype(float)
    valid = np.isfinite(weight.to_numpy(dtype=float)) & (
        weight.to_numpy(dtype=float) > 0
    )
    weight = weight.mask(~valid, np.nan)
    return weight.clip(lower=clip_min, upper=clip_max)


def normalize01(values: Any) -> pd.Series:
    out = numeric_series(values).astype(float)
    finite = np.isfinite(out.to_numpy(dtype=float))
    out = out.mask(~finite, np.nan)
    if not finite.any():
        return pd.Series([np.nan] * len(out), index=out.index, dtype=float)

    v_min = float(np.nanmin(out.to_numpy(dtype=float)))
    v_max = float(np.nanmax(out.to_numpy(dtype=float)))
    if v_min >= 0 and v_max <= 1:
        return out
    if v_max > v_min:
        return (out - v_min) / (v_max - v_min)
    return pd.Series([np.nan] * len(out), index=out.index, dtype=float)


def require_group_columns(
    group_cols: str | Iterable[str] | None,
    frames: dict[str, pd.DataFrame],
    *,
    reserved: Iterable[str] = (),
) -> list[str]:
    columns = column_names(group_cols)
    invalid = [column for column in columns if column in set(reserved)]
    if invalid:
        raise DebiasSchemaError(
            "`group_cols` must contain only non-key stratum columns; reserved "
            "column(s): " + ", ".join(invalid)
        )
    for frame_name, frame in frames.items():
        missing = [col for col in columns if col not in frame.columns]
        if missing:
            raise DebiasSchemaError(
                f"`group_cols` missing in {frame_name}: {', '.join(missing)}"
            )
    return columns


def collapse_exact_key_duplicates(
    frame: pd.DataFrame,
    *,
    keys: Iterable[str],
    value_columns: Iterable[str],
    frame_name: str,
) -> pd.DataFrame:
    """Collapse identical mappings and reject conflicting rows for a key.

    This is intended for lookup tables. Repeating an identical mapping is
    harmless, whereas retaining two different mappings would multiply a join
    and make its result depend on row duplication.
    """
    key_columns = column_names(keys)
    values = [
        column for column in column_names(value_columns) if column not in key_columns
    ]
    selected = frame.loc[:, [*key_columns, *values]].copy()
    exact = selected.drop_duplicates(keep="first")
    if bool(exact.duplicated(subset=key_columns, keep=False).any()):
        raise DebiasSchemaError(
            f"`{frame_name}` contains conflicting rows for key "
            f"({', '.join(key_columns)}); the join is not one-to-one."
        )
    return exact.reset_index(drop=True)


def join_with_r_warning(
    data: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    on: list[str],
    how: Literal["left", "inner"] = "left",
) -> pd.DataFrame:
    """Preserve dplyr row expansion and warn for matching many-to-many keys."""
    duplicated_reference = reference.duplicated(on, keep=False)
    if bool(duplicated_reference.any()):
        repeated_data = data.loc[data.duplicated(on, keep=False), on].drop_duplicates()
        repeated_reference = reference.loc[duplicated_reference, on].drop_duplicates()
        if not repeated_data.merge(repeated_reference, on=on).empty:
            warnings.warn(
                "Detected an unexpected many-to-many relationship between "
                "the joined tables.",
                UserWarning,
                stacklevel=3,
            )
    return data.merge(reference, how=how, on=on)
