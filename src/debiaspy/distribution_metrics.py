"""Private distribution calculations shared by coverage and flow validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _distribution_shares(
    values: np.ndarray | pd.Series,
    epsilon: float,
    total: float | pd.Series | None = None,
) -> np.ndarray | pd.Series:
    """Smooth counts using either their total or supplied per-origin totals."""
    smoothed = values + epsilon
    if total is None:
        total = float(np.sum(smoothed))
    if bool(np.any(~np.isfinite(total) | (total <= 0))):
        raise ValueError(
            "Cannot compute distribution shares from non-finite or empty totals."
        )
    return smoothed / total


def _distribution_kl_contributions(
    reference: np.ndarray | pd.Series, comparison: np.ndarray | pd.Series
) -> np.ndarray | pd.Series:
    return reference * np.log(reference / comparison)


def _distribution_jsd_contributions(
    reference: np.ndarray | pd.Series, comparison: np.ndarray | pd.Series
) -> np.ndarray | pd.Series:
    midpoint = 0.5 * (reference + comparison)
    return 0.5 * reference * np.log(reference / midpoint) + (
        0.5 * comparison * np.log(comparison / midpoint)
    )
