"""Private distribution calculations shared by measurement and validation."""

import warnings
from numbers import Real

import numpy as np


def _validate_distribution_epsilon(epsilon: float) -> None:
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, Real)
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("`epsilon` must be a single positive finite number.")


def _distribution_shares(values: np.ndarray, epsilon: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        smoothed = values + epsilon
        total = np.nansum(smoothed)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "Cannot compute distribution shares from non-finite or empty totals."
        )
    return smoothed / total


def _distribution_kl_contributions(
    reference_share: np.ndarray, comparison_share: np.ndarray
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = reference_share / comparison_share
        if np.any(ratio < 0):
            warnings.warn(
                "NaNs produced in distribution logarithms.",
                RuntimeWarning,
                stacklevel=3,
            )
        return reference_share * np.log(ratio)


def _distribution_jsd_contributions(
    reference_share: np.ndarray, comparison_share: np.ndarray
) -> np.ndarray:
    midpoint = 0.5 * (reference_share + comparison_share)
    return 0.5 * _distribution_kl_contributions(reference_share, midpoint) + (
        0.5 * _distribution_kl_contributions(comparison_share, midpoint)
    )
