from __future__ import annotations

import numpy as np
import pandas as pd


def r_style_simple_ols(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tolerance: float = 1e-7,
) -> tuple[float, float, np.ndarray, float]:
    """Fit ``y ~ x`` with R's ordered QR alias tolerance.

    ``stats::lm`` treats a nearly constant predictor as aliased at its default
    ``1e-7`` QR tolerance.  NumPy's machine-precision rank rule instead produces
    enormous cancelling coefficients.  This helper keeps the intercept, marks
    the aliased slope as missing, and uses the intercept-only fitted values.
    """

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if (
        x_values.ndim != 1
        or y_values.ndim != 1
        or len(x_values) != len(y_values)
        or len(x_values) == 0
        or not np.isfinite(x_values).all()
        or not np.isfinite(y_values).all()
    ):
        raise ValueError("x and y must be equally sized, finite one-dimensional arrays")

    x_norm = float(np.linalg.norm(x_values))
    centered_norm = float(np.linalg.norm(x_values - np.mean(x_values)))
    if centered_norm <= tolerance * x_norm:
        intercept = float(np.mean(y_values))
        slope = np.nan
        predicted = np.full(len(y_values), intercept, dtype=float)
    else:
        design = np.column_stack([np.ones(len(x_values)), x_values])
        intercept, slope = np.linalg.lstsq(design, y_values, rcond=None)[0]
        intercept = float(intercept)
        slope = float(slope)
        predicted = intercept + slope * x_values

    total = float(np.sum((y_values - np.mean(y_values)) ** 2))
    residual = float(np.sum((y_values - predicted) ** 2))
    r_squared = np.nan if total == 0 else 1 - residual / total
    return intercept, slope, predicted, r_squared


def global_moran_i(
    values: pd.DataFrame, neighbors: pd.DataFrame
) -> dict[str, float | int]:
    """Compute global Moran I from finite area residuals and positive weighted links."""
    values = values.loc[np.isfinite(values["selected_residual"])]
    n = len(values)
    result = {
        "moran_i": np.nan,
        "n_areas_used": n,
        "n_links_used": 0,
        "weight_sum": np.nan,
    }
    if n < 2:
        return result
    mean = values["selected_residual"].mean()
    denominator = float(((values["selected_residual"] - mean) ** 2).sum())
    links = neighbors.merge(
        values.rename(columns={"selected_residual": "area_residual"}),
        on="area",
        how="inner",
    ).merge(
        values.rename(
            columns={"area": "neighbor", "selected_residual": "neighbor_residual"}
        ),
        on="neighbor",
        how="inner",
    )
    links = links.loc[np.isfinite(links["weight"]) & (links["weight"] > 0)]
    weight_sum = float(links["weight"].sum())
    result.update(n_links_used=len(links), weight_sum=weight_sum)
    if len(links) and weight_sum > 0 and denominator > 0:
        numerator = float(
            (
                links["weight"]
                * (links["area_residual"] - mean)
                * (links["neighbor_residual"] - mean)
            ).sum()
        )
        result["moran_i"] = float((n / weight_sum) * (numerator / denominator))
    return result
