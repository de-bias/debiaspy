"""Measure population coverage bias."""

from __future__ import annotations

import warnings

import pandas as pd


def measure_bias(coverage_df: pd.DataFrame) -> pd.DataFrame:
    """Compute area-level active-user coverage score and coverage bias.

    The primary bias measure follows the `debiasR` definition:

    `coverage_bias = 1 - user_count / population`

    The returned DataFrame contains all input columns plus:

    - `coverage_score`: `user_count / population`
    - `coverage_bias`: `1 - coverage_score`
    - `bias`: backward-compatible alias of `coverage_bias`

    Parameters
    ----------
    coverage_df:
        DataFrame with one row per area and columns `population` and
        `user_count`.

    Returns
    -------
    pandas.DataFrame
        A copy of `coverage_df` with the added coverage columns.

    Raises
    ------
    TypeError
        If `coverage_df` is not a pandas DataFrame.
    ValueError
        If required columns are missing or values are invalid.
    """

    if not isinstance(coverage_df, pd.DataFrame):
        raise TypeError("`coverage_df` must be a pandas DataFrame.")

    required_cols = ["population", "user_count"]
    missing_cols = [col for col in required_cols if col not in coverage_df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"Missing required columns: {missing}")

    if coverage_df["population"].isna().any():
        raise ValueError("population contains NA values; cannot compute coverage bias.")
    if coverage_df["population"].le(0).any():
        raise ValueError("population must be positive to compute coverage bias.")
    if coverage_df["user_count"].lt(0).any():
        raise ValueError("user_count must be non-negative.")

    result = coverage_df.copy()
    result["coverage_score"] = result["user_count"] / result["population"]
    result["coverage_bias"] = 1 - result["coverage_score"]
    result["bias"] = result["coverage_bias"]

    if result["coverage_score"].gt(1).any():
        warnings.warn(
            "One or more areas have coverage_score > 1 "
            "(user_count exceeds population). Check inputs or definitions.",
            UserWarning,
            stacklevel=2,
        )

    return result


__all__ = ["measure_bias"]
