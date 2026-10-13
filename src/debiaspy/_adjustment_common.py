from __future__ import annotations

import warnings
from collections.abc import Iterable
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
import pandas as pd

from debiaspy._choices import match_choice as match_choice
from debiaspy._exceptions import DebiasSchemaError

REPEATED_OBSERVATION_KEYS = ("mpd_source", "mpd_time")


def numeric_series(values: Any) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


def numeric_array(values: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def r_character_key_series(values: Any) -> pd.Series:
    """Canonicalize identifier values using R-like numeric character forms.

    Pandas renders a floating-point identifier such as ``1.0`` as ``"1.0"``,
    while R's ``as.character(1.0)`` returns ``"1"``.  debiasR uses the latter
    conversion before raking, so equivalent integer and floating-point IDs must
    resolve to the same key without changing the values returned to the caller.
    """

    def canonical(value: Any) -> object:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return "TRUE" if bool(value) else "FALSE"
        if isinstance(value, Integral):
            return str(int(value))
        if isinstance(value, Real):
            numeric = float(value)
            if not np.isfinite(numeric):
                return str(value)
            if numeric.is_integer():
                return str(int(numeric))
            return np.format_float_positional(numeric, unique=True, trim="-")
        return str(value)

    return pd.Series(values, copy=False).map(canonical).astype("string")


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


def finite_positive_mask(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.isfinite(array) & (array > 0)


def finite_non_negative_mask(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.isfinite(array) & (array >= 0)


def validated_non_negative_numeric(values: Any, name: str) -> pd.Series:
    """Return finite non-negative numeric values, rejecting logical counts."""
    raw = pd.Series(values, copy=False)
    if bool(raw.map(lambda value: isinstance(value, (bool, np.bool_))).any()):
        raise DebiasSchemaError(
            f"`{name}` must contain finite non-negative numeric values."
        )
    numeric = pd.to_numeric(raw, errors="coerce").astype(float)
    if not bool((np.isfinite(numeric) & (numeric >= 0)).all()):
        raise DebiasSchemaError(
            f"`{name}` must contain finite non-negative numeric values."
        )
    return numeric


def validated_count_frame(
    frame: pd.DataFrame,
    *,
    positive: Iterable[str] = (),
    non_negative: Iterable[str] = (),
    frame_name: str,
) -> pd.DataFrame:
    """Copy a count lookup and validate all declared count columns."""
    out = frame.copy()
    for column in column_names([*positive, *non_negative]):
        out[column] = validated_non_negative_numeric(
            out[column], f"{frame_name}${column}"
        ).to_numpy()
    for column in column_names(positive):
        if bool((out[column] <= 0).any()):
            raise DebiasSchemaError(
                f"`{frame_name}${column}` must contain finite positive numeric values."
            )
    return out


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


def require_scalar_bool(value: Any, name: str) -> bool:
    """Validate a Python/R logical flag without accepting integer lookalikes."""
    if not isinstance(value, (bool, np.bool_)):
        raise DebiasSchemaError(f"`{name}` must be a single boolean value.")
    return bool(value)


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


def repeated_observation_join_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    base_keys: Iterable[str] = ("origin", "destination"),
    extra_keys: Iterable[str] = (),
) -> list[str]:
    """Return base keys plus shared source/time/stratum observation keys."""
    candidates = column_names([*REPEATED_OBSERVATION_KEYS, *extra_keys])
    return [
        *column_names(base_keys),
        *[key for key in candidates if key in left.columns and key in right.columns],
    ]


def require_unique_observation_rows(
    frame: pd.DataFrame,
    *,
    keys: Iterable[str],
    frame_name: str,
    counterpart: pd.DataFrame | None = None,
    counterpart_name: str | None = None,
    candidate_observation_keys: Iterable[str] = REPEATED_OBSERVATION_KEYS,
) -> None:
    """Fail when a join key does not identify one observation row."""
    key_columns = column_names(keys)
    if not bool(frame.duplicated(subset=key_columns, keep=False).any()):
        return

    missing = []
    if counterpart is not None:
        missing = [
            key
            for key in column_names(candidate_observation_keys)
            if key in frame.columns and key not in counterpart.columns
        ]
    detail = ""
    if missing:
        other = (
            "the other input" if counterpart_name is None else f"`{counterpart_name}`"
        )
        detail = f" Add matching observation key(s) to {other}: {', '.join(missing)}."
    raise DebiasSchemaError(
        f"`{frame_name}` must contain at most one row per join key "
        f"({', '.join(key_columns)}); the join would multiply observations."
        f"{detail}"
    )


class AdjustmentSet(dict[str, pd.DataFrame]):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.attrs: dict[str, Any] = {} if attrs is None else attrs


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


def model_result_attributes(metadata: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Expose common model metadata alongside engine-specific fits and diagnostics."""
    fields = (
        "backend",
        "model_engine",
        "model_family",
        "model_contract",
        "target_scale",
        "observation_model",
        "coverage_scale",
        "model_terms",
        "bias_terms",
        "stage_scope",
        "random_intercept",
        "scenario",
        "source_col",
        "time_col",
        "repeated_observation",
        "flow_adj_summary",
        "prediction_scope",
        "runtime_seconds",
        "od_audit",
        "distance_source",
    )
    return {
        **{name: metadata[name] for name in fields},
        "result_metadata": metadata,
        **extra,
    }
