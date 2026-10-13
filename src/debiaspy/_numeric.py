from __future__ import annotations

import numpy as np
import pandas as pd


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
