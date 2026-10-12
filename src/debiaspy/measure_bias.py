"""Measure population coverage bias and diagnose its residual structure."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from numbers import Real
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def measure_bias(coverage_df: pd.DataFrame) -> pd.DataFrame:
    """Compute area-level active-user coverage score and coverage bias.

    The primary bias measure follows the ``debiasR`` definition:
    ``coverage_bias = 1 - user_count / population``.

    Parameters
    ----------
    coverage_df:
        DataFrame with one row per area and columns ``population`` and
        ``user_count``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``coverage_df`` with ``coverage_score``, ``coverage_bias``
        and the backward-compatible ``bias`` alias added.
    """
    _require_dataframe(coverage_df, "coverage_df")

    required_cols = ["population", "user_count"]
    missing_cols = [col for col in required_cols if col not in coverage_df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"Missing required columns: {missing}")
    if coverage_df["population"].isna().any():
        raise ValueError("population contains NA values; cannot compute coverage bias.")
    _require_numeric_columns(coverage_df, required_cols, "coverage_df")
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


def measure_bias_distribution(
    coverage_df: pd.DataFrame,
    area_col: str = "origin",
    population_col: str = "population",
    user_count_col: str = "user_count",
    epsilon: float = 1e-8,
    return_area_level: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compare the spatial distributions of active users and population.

    The benchmark population shares form the reference distribution and the
    active-user shares form the comparison distribution. Lower
    Kullback-Leibler (KL) and Jensen-Shannon divergence (JSD) values indicate
    more similar distributions.

    Parameters
    ----------
    coverage_df:
        DataFrame containing one row per area.
    area_col, population_col, user_count_col:
        Names of the area, benchmark population and active-user columns.
    epsilon:
        Small positive smoothing constant added before shares are calculated.
    return_area_level:
        Whether to include area-level shares and metric contributions.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A dictionary containing a one-row ``summary`` DataFrame and, when
        requested, an ``area_level`` DataFrame.
    """
    _require_dataframe(coverage_df, "coverage_df")
    _validate_distribution_epsilon(epsilon)
    _require_string_column_names(area_col, population_col, user_count_col)

    required_cols = [area_col, population_col, user_count_col]
    _require_columns(coverage_df, required_cols, "coverage_df")
    _require_numeric_columns(
        coverage_df, [population_col, user_count_col], "coverage_df"
    )

    coverage = coverage_df.loc[:, required_cols].copy()
    coverage.columns = ["area", "population", "user_count"]

    if coverage["area"].isna().any():
        raise ValueError("`area_col` contains NA values; areas must be identifiable.")
    if coverage["area"].duplicated().any():
        raise ValueError("`coverage_df` must contain one row per area.")
    if coverage["population"].isna().any():
        raise ValueError(
            "`population_col` contains NA values; cannot compute distribution bias."
        )
    if coverage["user_count"].isna().any():
        raise ValueError(
            "`user_count_col` contains NA values; cannot compute distribution bias."
        )
    if coverage["population"].le(0).any():
        raise ValueError("`population_col` must be positive.")
    if coverage["user_count"].lt(0).any():
        raise ValueError("`user_count_col` must be non-negative.")

    area_level = measure_bias(coverage)
    population_share = _distribution_shares(
        area_level["population"].to_numpy(dtype=float), epsilon
    )
    user_share = _distribution_shares(
        area_level["user_count"].to_numpy(dtype=float), epsilon
    )
    area_level["population_share"] = population_share
    area_level["user_share"] = user_share
    area_level["share_difference_user_minus_population"] = user_share - population_share
    area_level["midpoint_share"] = 0.5 * (population_share + user_share)
    area_level["kl_contribution"] = population_share * np.log(
        population_share / user_share
    )
    area_level["jsd_contribution"] = _distribution_jsd_contributions(
        population_share, user_share
    )

    absolute_difference = area_level["share_difference_user_minus_population"].abs()
    summary = pd.DataFrame(
        {
            "comparison": ["active_users_vs_population"],
            "reference_distribution": ["population"],
            "comparison_distribution": ["active_users"],
            "n_areas": [len(area_level)],
            "total_population": [area_level["population"].sum()],
            "total_user_count": [area_level["user_count"].sum()],
            "epsilon": [float(epsilon)],
            "kl_population_user": [area_level["kl_contribution"].sum()],
            "jsd_population_user": [area_level["jsd_contribution"].sum()],
            "mean_abs_share_difference": [absolute_difference.mean()],
            "max_abs_share_difference": [absolute_difference.max()],
        }
    )

    result = {"summary": summary}
    if return_area_level:
        result["area_level"] = area_level
    return result


def validate_bias_residual_structure(
    coverage_df: pd.DataFrame,
    coverage_area_col: str = "origin",
    population_col: str = "population",
    user_count_col: str = "user_count",
    residual_type: str = "coverage_score",
    benchmark_od_df: pd.DataFrame | None = None,
    origin_col: str = "origin",
    destination_col: str = "destination",
    flow_col_bench: str = "flow",
    benchmark_flow_roles: str | Sequence[str] | None = (
        "origin",
        "destination",
    ),
    area_neighbors: pd.DataFrame | None = None,
    area_col: str = "area",
    neighbor_col: str = "neighbor",
    weight_col: str | None = None,
    covariate_df: pd.DataFrame | None = None,
    covariate_col: str | None = None,
    covariate_area_col: str = "area",
    geometry_df: pd.DataFrame | None = None,
    geometry_area_col: str = "area",
    x_col: str | None = None,
    y_col: str | None = None,
    make_plots: bool = False,
) -> dict[str, Any]:
    """Diagnose area-level residual structure in active-user coverage.

    The function estimates expected active-user counts under a constant global
    coverage rate and returns four residual definitions. Optional inputs add
    correlations with benchmark flows or an area-level covariate, global
    Moran's I from a neighbour-link table, joined geometry data and diagnostic
    Matplotlib figures.

    Parameters
    ----------
    coverage_df:
        DataFrame with one row per area.
    coverage_area_col, population_col, user_count_col:
        Column names in ``coverage_df``.
    residual_type:
        ``"coverage_score"``, ``"user_count"``,
        ``"standardized_user_count"`` or ``"population_lm"``.
    benchmark_od_df:
        Optional benchmark origin-destination flow table.
    origin_col, destination_col, flow_col_bench:
        Column names in ``benchmark_od_df``.
    benchmark_flow_roles:
        ``"origin"``, ``"destination"``, ``"both"`` or a sequence containing
        origin and/or destination.
    area_neighbors:
        Optional table of directed neighbour links for Moran's I.
    area_col, neighbor_col, weight_col:
        Column names in ``area_neighbors``. Equal weights are used when
        ``weight_col`` is omitted.
    covariate_df, covariate_col, covariate_area_col:
        Optional area-level covariate table and its column names.
    geometry_df, geometry_area_col:
        Optional area table joined to the returned ``map_data``.
    x_col, y_col:
        Coordinate columns in ``map_data`` used for the optional residual map.
    make_plots:
        Whether to include diagnostic Matplotlib figures under ``plots``.

    Returns
    -------
    dict[str, Any]
        DataFrames containing the summary, residual definitions, Moran's I,
        correlations, population-only model, area-level values and map data.
        Optional diagnostic data and figures are included when requested.
    """
    _require_dataframe(coverage_df, "coverage_df")
    valid_residual_types = {
        "coverage_score",
        "user_count",
        "standardized_user_count",
        "population_lm",
    }
    if residual_type not in valid_residual_types:
        options = ", ".join(sorted(valid_residual_types))
        raise ValueError(f"`residual_type` must be one of: {options}.")

    flow_roles = _normalise_benchmark_flow_roles(benchmark_flow_roles)
    if covariate_col is not None and covariate_df is None:
        raise ValueError("`covariate_df` is required when `covariate_col` is supplied.")

    coverage_columns = [coverage_area_col, population_col, user_count_col]
    _require_string_column_names(*coverage_columns)
    _require_columns(coverage_df, coverage_columns, "coverage_df")
    _require_numeric_columns(
        coverage_df, [population_col, user_count_col], "coverage_df"
    )

    coverage = coverage_df.loc[:, coverage_columns].copy()
    coverage.columns = ["area", "population", "user_count"]
    if coverage["area"].isna().any():
        raise ValueError(
            "`coverage_area_col` contains NA values; areas must be identifiable."
        )
    if coverage["area"].duplicated().any():
        raise ValueError(
            "`coverage_df` must contain one row per area for bias residual diagnostics."
        )
    if coverage["user_count"].isna().any():
        raise ValueError(
            "`user_count_col` contains NA values; cannot compute coverage residuals."
        )

    area_level = measure_bias(coverage)
    total_population = area_level["population"].sum()
    total_user_count = area_level["user_count"].sum()
    global_coverage_score = total_user_count / total_population
    area_level["global_coverage_score"] = global_coverage_score
    area_level["expected_user_count"] = area_level["population"] * global_coverage_score
    area_level["user_count_residual"] = (
        area_level["user_count"] - area_level["expected_user_count"]
    )
    area_level["coverage_score_residual"] = (
        area_level["coverage_score"] - global_coverage_score
    )
    area_level["standardized_user_count_residual"] = np.where(
        area_level["expected_user_count"] > 0,
        area_level["user_count_residual"] / np.sqrt(area_level["expected_user_count"]),
        np.nan,
    )

    population_lm, population_lm_expected = _fit_population_only_model(area_level)
    area_level["population_lm_expected_user_count"] = population_lm_expected
    area_level["population_lm_residual"] = (
        area_level["user_count"] - area_level["population_lm_expected_user_count"]
    )
    selected_residual_col = {
        "coverage_score": "coverage_score_residual",
        "user_count": "user_count_residual",
        "standardized_user_count": "standardized_user_count_residual",
        "population_lm": "population_lm_residual",
    }[residual_type]
    area_level["residual_type"] = residual_type
    area_level["selected_residual"] = area_level[selected_residual_col]

    benchmark_flow_data: pd.DataFrame | None = None
    benchmark_flow_correlation = pd.DataFrame(
        {
            "residual_type": pd.Series(dtype="object"),
            "benchmark_flow_role": pd.Series(dtype="object"),
            "n": pd.Series(dtype="int64"),
            "pearson_r": pd.Series(dtype="float64"),
        }
    )
    if benchmark_od_df is not None:
        _require_dataframe(benchmark_od_df, "benchmark_od_df")
        benchmark_columns = [origin_col, destination_col, flow_col_bench]
        _require_columns(benchmark_od_df, benchmark_columns, "benchmark_od_df")
        _require_numeric_columns(benchmark_od_df, [flow_col_bench], "benchmark_od_df")
        benchmark = benchmark_od_df.loc[:, benchmark_columns].copy()
        benchmark.columns = ["origin", "destination", "benchmark_flow"]
        benchmark_flow_tables: list[pd.DataFrame] = []

        for role in flow_roles:
            total_column = f"benchmark_{role}_flow_total"
            totals = (
                benchmark.groupby(role, dropna=False, sort=False)["benchmark_flow"]
                .sum()
                .rename(total_column)
                .rename_axis("area")
                .reset_index()
            )
            area_level = area_level.merge(totals, on="area", how="left", sort=False)
            area_level[total_column] = area_level[total_column].fillna(0)
            role_data = area_level.loc[
                :,
                [
                    "area",
                    "residual_type",
                    "selected_residual",
                    total_column,
                    "population",
                    "user_count",
                    "coverage_score",
                    "coverage_score_residual",
                ],
            ].copy()
            role_data.insert(2, "benchmark_flow_role", role)
            role_data = role_data.rename(columns={total_column: "benchmark_flow_total"})
            benchmark_flow_tables.append(role_data)

        if benchmark_flow_tables:
            benchmark_flow_data = pd.concat(benchmark_flow_tables, ignore_index=True)
            correlation_rows = []
            for role, role_data in benchmark_flow_data.groupby(
                "benchmark_flow_role", sort=False
            ):
                finite = _finite_pair_mask(
                    role_data["selected_residual"],
                    role_data["benchmark_flow_total"],
                )
                correlation_rows.append(
                    {
                        "residual_type": residual_type,
                        "benchmark_flow_role": role,
                        "n": int(finite.sum()),
                        "pearson_r": _safe_pearson(
                            role_data["selected_residual"],
                            role_data["benchmark_flow_total"],
                        ),
                    }
                )
            benchmark_flow_correlation = pd.DataFrame(correlation_rows)

    moran_stats: dict[str, Any] = {
        "moran_i": np.nan,
        "n_areas_used": int(np.isfinite(area_level["selected_residual"]).sum()),
        "n_links_used": pd.NA,
        "weight_sum": np.nan,
    }
    if area_neighbors is not None:
        moran_stats = _compute_moran_i(
            area_level,
            area_neighbors,
            area_col=area_col,
            neighbor_col=neighbor_col,
            weight_col=weight_col,
        )
    moran_i = pd.DataFrame(
        {
            "residual_type": [residual_type],
            "n_areas_used": [moran_stats["n_areas_used"]],
            "n_links_used": [moran_stats["n_links_used"]],
            "weight_sum": [moran_stats["weight_sum"]],
            "moran_i": [moran_stats["moran_i"]],
        }
    )

    covariate_data: pd.DataFrame | None = None
    covariate_correlation = pd.DataFrame(
        {
            "residual_type": [residual_type],
            "covariate": [None],
            "n": [pd.NA],
            "pearson_r": [np.nan],
        }
    )
    if covariate_col is not None:
        assert covariate_df is not None
        _require_dataframe(covariate_df, "covariate_df")
        covariate_columns = [covariate_area_col, covariate_col]
        _require_columns(covariate_df, covariate_columns, "covariate_df")
        _require_numeric_columns(covariate_df, [covariate_col], "covariate_df")
        covariate = covariate_df.loc[:, covariate_columns].copy()
        covariate.columns = ["area", "covariate_value"]
        covariate_data = area_level.merge(covariate, on="area", how="left", sort=False)
        finite = _finite_pair_mask(
            covariate_data["selected_residual"],
            covariate_data["covariate_value"],
        )
        covariate_correlation = pd.DataFrame(
            {
                "residual_type": [residual_type],
                "covariate": [covariate_col],
                "n": [int(finite.sum())],
                "pearson_r": [
                    _safe_pearson(
                        covariate_data["selected_residual"],
                        covariate_data["covariate_value"],
                    )
                ],
            }
        )

    map_data = area_level.copy()
    if geometry_df is not None:
        _require_dataframe(geometry_df, "geometry_df")
        if geometry_area_col not in geometry_df.columns:
            raise ValueError("`geometry_area_col` must name a column in `geometry_df`.")
        geometry = geometry_df.rename(columns={geometry_area_col: "area"}).copy()
        map_data = area_level.merge(geometry, on="area", how="left", sort=False)

    summary = pd.DataFrame(
        {
            "residual_type": [residual_type],
            "selected_residual_col": [selected_residual_col],
            "n_areas": [len(area_level)],
            "total_population": [total_population],
            "total_user_count": [total_user_count],
            "global_coverage_score": [global_coverage_score],
            "mean_coverage_score": [area_level["coverage_score"].mean()],
            "sd_coverage_score": [area_level["coverage_score"].std(ddof=1)],
            "mean_selected_residual": [area_level["selected_residual"].mean()],
            "sd_selected_residual": [area_level["selected_residual"].std(ddof=1)],
            "moran_i": [moran_i.loc[0, "moran_i"]],
            "pearson_bias_benchmark_origin_flow": [
                _first_matching_correlation(benchmark_flow_correlation, "origin")
            ],
            "pearson_bias_benchmark_destination_flow": [
                _first_matching_correlation(benchmark_flow_correlation, "destination")
            ],
            "pearson_bias_covariate": [covariate_correlation.loc[0, "pearson_r"]],
        }
    )
    residual_definitions = pd.DataFrame(
        {
            "residual": [
                "coverage_score_residual",
                "user_count_residual",
                "standardized_user_count_residual",
                "population_lm_residual",
            ],
            "definition": [
                "coverage_score - global_coverage_score",
                "user_count - expected_user_count",
                "user_count_residual / sqrt(expected_user_count)",
                "user_count - fitted(user_count ~ population)",
            ],
            "interpretation": [
                "Positive values mean higher active-user coverage than the global "
                "coverage score.",
                "Positive values mean more active users than expected under the "
                "global coverage score.",
                "Positive values mean more active users than expected after a "
                "Poisson-style count scaling.",
                "Positive values mean more active users than expected under a "
                "descriptive population-only linear trend.",
            ],
        }
    )

    result: dict[str, Any] = {
        "summary": summary,
        "residual_definitions": residual_definitions,
        "moran_i": moran_i,
        "benchmark_flow_correlation": benchmark_flow_correlation,
        "covariate_correlation": covariate_correlation,
        "population_lm": population_lm,
        "area_level": area_level,
        "map_data": map_data,
    }
    if benchmark_flow_data is not None:
        result["benchmark_flow_data"] = benchmark_flow_data
    if covariate_data is not None:
        result["covariate_data"] = covariate_data
    if make_plots:
        result["plots"] = _make_bias_residual_plots(
            area_level=area_level,
            selected_residual_col=selected_residual_col,
            benchmark_flow_data=benchmark_flow_data,
            covariate_data=covariate_data,
            covariate_col=covariate_col,
            map_data=map_data,
            x_col=x_col,
            y_col=y_col,
        )
    return result


def _require_dataframe(value: object, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"`{name}` must be a pandas DataFrame.")


def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if any(column not in df.columns for column in columns):
        required = ", ".join(columns)
        raise ValueError(f"`{name}` must contain: {required}")


def _require_numeric_columns(
    df: pd.DataFrame, columns: Sequence[str], name: str
) -> None:
    non_numeric = [
        column for column in columns if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        joined = ", ".join(non_numeric)
        raise ValueError(f"`{name}` column(s) must be numeric: {joined}")


def _require_string_column_names(*column_names: str) -> None:
    if any(not isinstance(column, str) or not column for column in column_names):
        raise TypeError("Column names must be non-empty strings.")


def _validate_distribution_epsilon(epsilon: float) -> None:
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, Real)
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("`epsilon` must be a single positive finite number.")


def _distribution_shares(values: np.ndarray, epsilon: float) -> np.ndarray:
    total = np.sum(values + epsilon)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "Cannot compute distribution shares from non-finite or empty totals."
        )
    return (values + epsilon) / total


def _distribution_jsd_contributions(
    population_share: np.ndarray, user_share: np.ndarray
) -> np.ndarray:
    midpoint_share = 0.5 * (population_share + user_share)
    return 0.5 * population_share * np.log(population_share / midpoint_share) + (
        0.5 * user_share * np.log(user_share / midpoint_share)
    )


def _normalise_benchmark_flow_roles(
    benchmark_flow_roles: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if benchmark_flow_roles is None:
        return ()
    roles = (
        (benchmark_flow_roles,)
        if isinstance(benchmark_flow_roles, str)
        else tuple(benchmark_flow_roles)
    )
    if not roles:
        return ()
    if roles == ("both",):
        return ("origin", "destination")
    if "both" in roles or any(role not in {"origin", "destination"} for role in roles):
        raise ValueError(
            "`benchmark_flow_roles` must contain only 'origin', 'destination', "
            "or 'both'."
        )
    return tuple(dict.fromkeys(roles))


def _fit_population_only_model(
    area_level: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    population = area_level["population"].to_numpy(dtype=float)
    user_count = area_level["user_count"].to_numpy(dtype=float)
    finite = np.isfinite(population) & np.isfinite(user_count)
    can_fit = finite.sum() >= 2 and np.unique(population[finite]).size >= 2
    fitted = np.full(len(area_level), np.nan)

    if can_fit:
        design = np.column_stack([np.ones(finite.sum()), population[finite]])
        coefficients, *_ = np.linalg.lstsq(design, user_count[finite], rcond=None)
        fitted[finite] = design @ coefficients
        residual = user_count[finite] - fitted[finite]
        centred = user_count[finite] - user_count[finite].mean()
        total_sum_squares = float(np.sum(centred**2))
        residual_sum_squares = float(np.sum(residual**2))
        r_squared = (
            1 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0
            else np.nan
        )
        intercept = float(coefficients[0])
        population_coefficient = float(coefficients[1])
    else:
        intercept = np.nan
        population_coefficient = np.nan
        r_squared = np.nan

    model = pd.DataFrame(
        {
            "residual_type": ["population_lm"],
            "model": ["user_count ~ population"],
            "n": [int(finite.sum())],
            "intercept": [intercept],
            "population_coefficient": [population_coefficient],
            "r_squared": [r_squared],
        }
    )
    return model, fitted


def _safe_pearson(x: pd.Series, y: pd.Series) -> float:
    x_values = _as_float_array(x)
    y_values = _as_float_array(y)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if finite.sum() < 2:
        return np.nan
    if np.std(x_values[finite], ddof=1) <= 0 or np.std(y_values[finite], ddof=1) <= 0:
        return np.nan
    return float(np.corrcoef(x_values[finite], y_values[finite])[0, 1])


def _as_float_array(values: pd.Series) -> np.ndarray:
    return values.to_numpy(dtype=float, na_value=np.nan)


def _finite_pair_mask(x: pd.Series, y: pd.Series) -> np.ndarray:
    return np.isfinite(_as_float_array(x)) & np.isfinite(_as_float_array(y))


def _prepare_neighbor_links(
    area_neighbors: pd.DataFrame,
    area_col: str,
    neighbor_col: str,
    weight_col: str | None,
) -> pd.DataFrame:
    _require_dataframe(area_neighbors, "area_neighbors")
    _require_columns(area_neighbors, [area_col, neighbor_col], "area_neighbors")
    if weight_col is not None and weight_col not in area_neighbors.columns:
        raise ValueError("`weight_col` must name a column in `area_neighbors`.")

    columns = [area_col, neighbor_col]
    if weight_col is not None:
        columns.append(weight_col)
    neighbors = area_neighbors.loc[:, columns].copy()
    if weight_col is None:
        neighbors.columns = ["area", "neighbor"]
        neighbors["weight"] = 1.0
    else:
        neighbors.columns = ["area", "neighbor", "weight"]
        _require_numeric_columns(neighbors, ["weight"], "area_neighbors")
    weight = _as_float_array(neighbors["weight"])
    neighbors["weight"] = weight
    return neighbors.loc[np.isfinite(weight) & (weight > 0)].copy()


def _compute_moran_i(
    area_level: pd.DataFrame,
    area_neighbors: pd.DataFrame,
    area_col: str,
    neighbor_col: str,
    weight_col: str | None,
) -> dict[str, float | int]:
    neighbors = _prepare_neighbor_links(
        area_neighbors, area_col, neighbor_col, weight_col
    )
    values = area_level.loc[:, ["area", "selected_residual"]].copy()
    values = values.loc[np.isfinite(values["selected_residual"])]
    n_areas_used = len(values)
    if n_areas_used < 2:
        return {
            "moran_i": np.nan,
            "n_areas_used": n_areas_used,
            "n_links_used": 0,
            "weight_sum": np.nan,
        }

    mean_residual = values["selected_residual"].mean()
    denominator = float(np.sum((values["selected_residual"] - mean_residual) ** 2))
    area_values = values.rename(columns={"selected_residual": "area_residual"})
    neighbor_values = values.rename(
        columns={"area": "neighbor", "selected_residual": "neighbor_residual"}
    )
    links = neighbors.merge(area_values, on="area", how="inner").merge(
        neighbor_values, on="neighbor", how="inner"
    )
    links = links.loc[np.isfinite(links["weight"]) & links["weight"].gt(0)]
    n_links_used = len(links)
    weight_sum = float(links["weight"].sum())
    if n_links_used == 0 or weight_sum <= 0 or denominator <= 0:
        return {
            "moran_i": np.nan,
            "n_areas_used": n_areas_used,
            "n_links_used": n_links_used,
            "weight_sum": weight_sum,
        }

    numerator = np.sum(
        links["weight"]
        * (links["area_residual"] - mean_residual)
        * (links["neighbor_residual"] - mean_residual)
    )
    return {
        "moran_i": float((n_areas_used / weight_sum) * (numerator / denominator)),
        "n_areas_used": n_areas_used,
        "n_links_used": n_links_used,
        "weight_sum": weight_sum,
    }


def _first_matching_correlation(correlations: pd.DataFrame, role: str) -> float:
    matches = correlations.loc[
        correlations["benchmark_flow_role"].eq(role), "pearson_r"
    ]
    return float(matches.iloc[0]) if not matches.empty else np.nan


def _make_bias_residual_plots(
    area_level: pd.DataFrame,
    selected_residual_col: str,
    benchmark_flow_data: pd.DataFrame | None,
    covariate_data: pd.DataFrame | None,
    covariate_col: str | None,
    map_data: pd.DataFrame,
    x_col: str | None,
    y_col: str | None,
) -> dict[str, Figure]:
    if (x_col is None) != (y_col is None):
        raise ValueError(
            "Both `x_col` and `y_col` are required for a bias residual map plot."
        )
    if x_col is not None and y_col is not None:
        _require_columns(map_data, [x_col, y_col], "map_data")
        _require_numeric_columns(map_data, [x_col, y_col], "map_data")

    plots: dict[str, Figure] = {}
    figure, axis = plt.subplots()
    axis.hist(area_level["selected_residual"].dropna(), bins=30, color="#2B8CBE")
    axis.axvline(0, linestyle="--", color="#D95F0E")
    axis.set(
        xlabel=selected_residual_col,
        ylabel="Areas",
        title="Bias residual distribution",
    )
    figure.tight_layout()
    plots["bias_residual_distribution"] = figure

    if benchmark_flow_data is not None:
        roles = list(benchmark_flow_data["benchmark_flow_role"].drop_duplicates())
        figure, axes = plt.subplots(1, len(roles), squeeze=False)
        for axis, role in zip(axes.flat, roles, strict=True):
            role_data = benchmark_flow_data.loc[
                benchmark_flow_data["benchmark_flow_role"].eq(role)
            ]
            axis.scatter(
                role_data["benchmark_flow_total"],
                role_data["selected_residual"],
                alpha=0.7,
            )
            axis.axhline(0, linestyle="--", color="#D95F0E")
            axis.set(
                xlabel="Benchmark area flow total",
                ylabel=selected_residual_col,
                title=role.capitalize(),
            )
        figure.suptitle("Bias residuals versus benchmark flow totals")
        figure.tight_layout()
        plots["bias_residual_vs_benchmark_flow"] = figure

    if covariate_data is not None:
        figure, axis = plt.subplots()
        axis.scatter(
            _as_float_array(covariate_data["covariate_value"]),
            _as_float_array(covariate_data["selected_residual"]),
            alpha=0.7,
        )
        axis.axhline(0, linestyle="--", color="#D95F0E")
        axis.set(
            xlabel=covariate_col,
            ylabel=selected_residual_col,
            title="Bias residuals versus covariate",
        )
        figure.tight_layout()
        plots["bias_residual_vs_covariate"] = figure

    if x_col is not None and y_col is not None:
        map_x = _as_float_array(map_data[x_col])
        map_y = _as_float_array(map_data[y_col])
        map_residual = _as_float_array(map_data["selected_residual"])
        finite = np.isfinite(map_x) & np.isfinite(map_y) & np.isfinite(map_residual)
        figure, axis = plt.subplots()
        points = axis.scatter(
            map_x[finite],
            map_y[finite],
            c=map_residual[finite],
            s=35,
            alpha=0.9,
        )
        axis.set(
            xlabel=x_col,
            ylabel=y_col,
            title="Area-level bias residual map",
            aspect="equal",
        )
        figure.colorbar(points, ax=axis, label="Residual")
        figure.tight_layout()
        plots["bias_residual_map"] = figure
    return plots


__all__ = [
    "measure_bias",
    "measure_bias_distribution",
    "validate_bias_residual_structure",
]
