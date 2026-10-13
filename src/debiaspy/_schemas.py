from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import pandas as pd

from debiaspy._exceptions import DebiasSchemaError


def require_columns(
    frame: pd.DataFrame, required: Iterable[str], frame_name: str
) -> None:
    required_columns = tuple(required)
    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        raise DebiasSchemaError(
            f"`{frame_name}` must contain: {', '.join(required_columns)}"
        )


def require_no_na(frame: pd.DataFrame, column: str, message: str) -> None:
    if bool(frame[column].isna().any()):
        raise DebiasSchemaError(message)


def require_scalar_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise DebiasSchemaError(f"`{name}` must be a single boolean value.")
    return bool(value)


def require_positive(frame: pd.DataFrame, column: str, message: str) -> None:
    numeric = _finite_numeric_column(frame, column, message)
    if bool((numeric <= 0).any()):
        raise DebiasSchemaError(message)
    frame[column] = numeric


def require_non_negative(frame: pd.DataFrame, column: str, message: str) -> None:
    numeric = _finite_numeric_column(frame, column, message)
    if bool((numeric < 0).any()):
        raise DebiasSchemaError(message)
    frame[column] = numeric


def _finite_numeric_column(
    frame: pd.DataFrame,
    column: str,
    message: str,
) -> pd.Series:
    values = frame[column]
    if not isinstance(values, pd.Series) or bool(
        values.map(lambda value: isinstance(value, (bool, np.bool_))).any()
    ):
        raise DebiasSchemaError(message)
    numeric = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    if not bool(np.isfinite(np.asarray(numeric, dtype=float)).all()):
        raise DebiasSchemaError(message)
    return numeric
