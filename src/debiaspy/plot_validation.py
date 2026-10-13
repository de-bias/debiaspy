from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from debiaspy._choices import match_choice


def _plot_flow_residual_diagnostics(
    *,
    od_level: pd.DataFrame,
    covariate_data: pd.DataFrame | None,
    map_data: pd.DataFrame,
    comparison_label: str,
    covariate_col: str | None,
    x_col: str | None,
    y_col: str | None,
) -> dict[str, Any]:
    plots = {
        "residual_reduction_distribution": _plot_residual_histogram(
            od_level,
            value_col="abs_residual_reduction",
            x_label="Absolute residual reduction",
            y_label="OD pairs",
            title="Residual reduction distribution",
        ),
        "residual_vs_benchmark": _plot_residual_scatter(
            od_level,
            x_col="benchmark_flow",
            y_col="selected_residual",
            x_label="Benchmark OD flow",
            y_label=f"{comparison_label} residual",
            title="Residuals versus benchmark flow",
        ),
    }
    if covariate_data is not None:
        plots["residual_vs_covariate"] = _plot_residual_scatter(
            covariate_data,
            x_col="covariate_value",
            y_col="selected_residual",
            x_label=str(covariate_col),
            y_label=f"{comparison_label} area residual",
            title="Area residuals versus covariate",
        )
    if x_col is not None and y_col is not None:
        plots["residual_map"] = _plot_coordinate_residual_map(
            map_data,
            x_col=x_col,
            y_col=y_col,
            residual_col="selected_residual",
            title="Area-level residual map",
        )
    return plots


def _plot_residual_histogram(
    data: pd.DataFrame,
    *,
    value_col: str,
    x_label: str,
    y_label: str,
    title: str,
) -> Any:
    plt = _require_matplotlib()
    values = pd.to_numeric(data[value_col], errors="coerce")
    values = values.loc[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values.to_numpy(dtype=float), bins=30, color="#2B8CBE", edgecolor="white")
    ax.axvline(0, linestyle="--", color="#D95F0E", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _plot_residual_scatter(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    facet_col: str | None = None,
) -> Any:
    plt = _require_matplotlib()
    facet_values: list[object | None] = (
        list(pd.unique(data[facet_col])) if facet_col is not None else [None]
    )
    fig, axes = plt.subplots(
        1,
        len(facet_values),
        figsize=(6 * len(facet_values), 4),
        squeeze=False,
    )
    for ax, facet_value in zip(axes.flat, facet_values, strict=True):
        panel = (
            data.loc[data[facet_col] == facet_value] if facet_col is not None else data
        )
        x_values = pd.to_numeric(panel[x_col], errors="coerce")
        y_values = pd.to_numeric(panel[y_col], errors="coerce")
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        ax.scatter(x_values.loc[finite], y_values.loc[finite], alpha=0.7)
        ax.axhline(0, linestyle="--", color="#D95F0E", linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        panel_title = title
        if facet_col is not None:
            panel_title = f"{title} — {facet_value}"
        ax.set_title(panel_title)
    fig.tight_layout()
    return fig


def _plot_coordinate_residual_map(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    residual_col: str,
    title: str,
) -> Any:
    plt = _require_matplotlib()
    x_values = pd.to_numeric(data[x_col], errors="coerce")
    y_values = pd.to_numeric(data[y_col], errors="coerce")
    residuals = pd.to_numeric(data[residual_col], errors="coerce")
    finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(residuals)
    fig, ax = plt.subplots(figsize=(6, 5))
    points = ax.scatter(
        x_values.loc[finite],
        y_values.loc[finite],
        c=residuals.loc[finite],
        s=45,
        alpha=0.9,
        cmap="coolwarm",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    if bool(finite.any()):
        fig.colorbar(points, ax=ax, label="Residual")
    fig.tight_layout()
    return fig


def validate_flow_plot_metrics(
    metrics: Any,
    error_measures: Sequence[str] = ("mae", "rmse", "mape"),
    metric_cols: Sequence[str] | None = None,
    comparisons: str | Sequence[str] = "adjusted_vs_benchmark",
    methods: Sequence[str] | None = None,
    method_col: str = "method",
    method_family_col: str | None = None,
    method_labels: Mapping[str, str] | None = None,
    palette: Mapping[str, str] | Sequence[str] | None = None,
    relative_error_breaks: Sequence[float] = tuple(range(0, 101, 10)),
    sort: str = "none",
    sort_metric: str | None = None,
) -> Any:
    """Plot a heatmap of overall flow-error metrics by method.

    ``metrics`` accepts an overall-validation result, a method-keyed result, or
    a table containing ``method``, ``comparison``, and the selected metric
    columns. Options filter comparisons/methods and control labels, palette,
    relative-score breaks, and ordering. Returns a Matplotlib ``Figure``.
    Missing columns, empty selections, invalid options, or an unavailable
    plotting runtime raise. Parity covers audited plot data and visual grammar;
    pixel-identical rendering across Python and R is not claimed.
    """
    plt = _require_matplotlib()
    sort = _match_option(sort, ("none", "ascending", "descending"), "sort")
    _validate_relative_error_breaks(relative_error_breaks)
    default_measures = ("mae", "rmse", "mape")
    if metric_cols is not None and tuple(error_measures) != default_measures:
        raise ValueError("Use only one of `error_measures` or `metric_cols`.")
    metric_cols = [
        _normalise_metric_name(metric)
        for metric in (metric_cols if metric_cols is not None else error_measures)
    ]
    if not metric_cols:
        raise ValueError("`error_measures` must select at least one metric.")
    data = _as_validation_table(metrics, method_col, "summary", "metrics")
    data = _filter_methods(data, methods, method_col, method_labels)
    data = _filter_comparisons(data, comparisons)
    _require_columns(data, [method_col, *metric_cols], "metrics")
    if method_family_col is not None:
        _require_columns(data, [method_family_col], "metrics")
    sort_metric = (
        metric_cols[0] if sort_metric is None else _normalise_metric_name(sort_metric)
    )
    if sort_metric not in metric_cols:
        raise ValueError("`sort_metric` must name one of the plotted metrics.")
    data = _prepare_comparison_plot_rows(
        data,
        method_col=method_col,
        method_labels=method_labels,
        integrate_benchmark=True,
    )
    long = []
    for metric in metric_cols:
        for _, row in data.iterrows():
            long.append(
                {
                    "method": row["_method_label"],
                    "comparison": row["comparison"],
                    "comparison_display": row["_comparison_display"],
                    "comparison_title": row["_comparison_title"],
                    "metric": _metric_label(metric),
                    "value": row[metric],
                }
            )
    plot_data = pd.DataFrame(long)
    order = _comparison_method_order(
        data=data,
        value_col=sort_metric,
        sort=sort,
    )
    metric_order = list(pd.unique(plot_data["metric"]))
    plot_data["relative_value"] = np.nan
    for metric in metric_order:
        metric_rows = plot_data["metric"] == metric
        values = pd.to_numeric(plot_data.loc[metric_rows, "value"], errors="coerce")
        finite = np.isfinite(values)
        if bool(finite.any()):
            lower = float(values.loc[finite].min())
            upper = float(values.loc[finite].max())
            plot_data.loc[metric_rows, "relative_value"] = (
                100 * (values - lower) / (upper - lower) if upper > lower else 0.0
            )

    from matplotlib.colors import BoundaryNorm, ListedColormap

    breaks = np.asarray(relative_error_breaks, dtype=float)
    colors = _palette(len(breaks) - 1, palette)
    cmap = ListedColormap(colors)
    cmap = cmap.with_extremes(bad="#f2f2f2")
    norm = BoundaryNorm(breaks, cmap.N, clip=True)

    display_order = list(pd.unique(plot_data["comparison_display"]))
    fig, axes = plt.subplots(
        len(display_order),
        1,
        figsize=(
            max(5, len(metric_order) * 1.4),
            sum(
                max(
                    3.5,
                    len(
                        pd.unique(
                            plot_data.loc[
                                plot_data["comparison_display"] == display,
                                "method",
                            ]
                        )
                    )
                    * 0.6,
                )
                for display in display_order
            ),
        ),
        squeeze=False,
    )
    image = None
    for ax, display in zip(axes.flat, display_order, strict=True):
        panel = plot_data.loc[
            plot_data["comparison_display"] == display
        ].drop_duplicates(["method", "metric"], keep="first")
        panel_order = [method for method in order if method in set(panel["method"])]
        value_pivot = panel.pivot(
            index="method", columns="metric", values="value"
        ).reindex(index=panel_order, columns=metric_order)
        relative_pivot = panel.pivot(
            index="method", columns="metric", values="relative_value"
        ).reindex(index=panel_order, columns=metric_order)
        image = ax.imshow(
            relative_pivot.to_numpy(dtype=float),
            cmap=cmap,
            norm=norm,
            aspect="auto",
        )
        ax.set_xticks(range(len(metric_order)), metric_order, rotation=0)
        ax.set_yticks(range(len(panel_order)), panel_order)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if len(display_order) > 1:
            ax.set_title(str(panel["comparison_title"].iloc[0]))
        for y_index, method in enumerate(panel_order):
            for x_index, metric in enumerate(metric_order):
                value = value_pivot.loc[method, metric]
                if pd.notna(value):
                    ax.text(
                        x_index,
                        y_index,
                        _format_plot_value(value),
                        ha="center",
                        va="center",
                        color="#111827",
                        fontweight="bold",
                        fontsize=9,
                    )
    if image is not None:
        fig.colorbar(
            image,
            ax=list(axes.flat),
            label="Relative error score (%)",
        )
    fig.subplots_adjust(hspace=0.45, right=0.82)
    return fig


def validate_flow_plot_residuals(
    residuals: Any,
    residual: str = "signed",
    comparisons: str | Sequence[str] = "adjusted_vs_benchmark",
    methods: Sequence[str] | None = None,
    method_col: str = "method",
    method_labels: Mapping[str, str] | None = None,
    palette: Mapping[str, str] | Sequence[str] | None = None,
    show_points: bool = True,
    show_counts: bool = True,
    point_alpha: float = 0.25,
    point_size: float = 0.6,
    jitter_width: float = 0.12,
    violin_alpha: float = 0.5,
    max_points: int | float | None = 5000,
    y_transform: str = "pseudo_log",
    sort: str = "none",
    sort_metric: str = "median_absolute_residual",
) -> Any:
    """Plot residual distributions for one or more adjustment methods.

    ``residuals`` accepts residual-validation output, method-keyed outputs, or a
    pair-level table containing method labels and raw, adjusted, and benchmark
    flows. Options select residual/comparison type, filtering, sampling, violin
    and point styling, transforms, and ordering. Returns a Matplotlib ``Figure``.
    Missing finite residuals, invalid columns/options, or an unavailable plotting
    runtime raise. Parity covers audited plot data and visual grammar, not
    pixel-identical rendering across Python and R.
    """
    plt = _require_matplotlib()
    residual = _match_option(residual, ("signed", "absolute", "percent"), "residual")
    y_transform = _match_option(y_transform, ("pseudo_log", "identity"), "y_transform")
    sort = _match_option(sort, ("none", "ascending", "descending"), "sort")
    sort_metric = _match_option(
        sort_metric,
        ("median_absolute_residual", "mean_absolute_residual", "mean_residual"),
        "sort_metric",
    )
    _validate_bool(show_points, "show_points")
    _validate_bool(show_counts, "show_counts")
    _validate_unit_interval(point_alpha, "point_alpha", closed=True)
    _validate_non_negative(point_size, "point_size")
    _validate_non_negative(jitter_width, "jitter_width")
    _validate_unit_interval(violin_alpha, "violin_alpha", closed=True)
    if max_points is not None:
        try:
            max_points_value = float(max_points)
        except (TypeError, ValueError):
            raise ValueError(
                "`max_points` must be numeric, infinite, or `None`."
            ) from None
        if np.isnan(max_points_value):
            raise ValueError("`max_points` must be numeric, infinite, or `None`.")
    else:
        max_points_value = np.inf
    data = _residual_long(
        _filter_methods(
            _as_validation_table(residuals, method_col, "data", "residuals"),
            methods,
            method_col,
            method_labels,
        ),
        residual,
        comparisons,
        method_col,
        method_labels,
    )
    data = data.loc[np.isfinite(pd.to_numeric(data["value"], errors="coerce"))]
    if data.empty:
        raise ValueError("`residuals` must contain at least one finite residual.")
    labels = _residual_method_order(data, sort=sort, sort_metric=sort_metric)
    values = [
        data.loc[data["method_label"] == label, "value"].dropna() for label in labels
    ]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    violin = ax.violinplot(values, showmeans=False, showmedians=True)
    colors = _palette(len(labels), palette, keys=labels)
    for body, color in zip(violin["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(violin_alpha)
    if show_points:
        rng = np.random.default_rng(0)
        for idx, series in enumerate(values, start=1):
            sample = _thin_plot_series(series, max_points_value)
            x = idx + rng.uniform(-jitter_width, jitter_width, size=len(sample))
            ax.scatter(
                x,
                sample,
                alpha=point_alpha,
                s=point_size * 12,
                color=colors[idx - 1],
                linewidths=0,
            )
    if show_counts:
        for idx, series in enumerate(values, start=1):
            span = float(series.max() - series.min())
            label_y = float(series.max()) + max(abs(span) * 0.08, 1.0)
            ax.text(idx, label_y, f"n={len(series):,}", ha="center", fontweight="bold")
    ax.axhline(0, color="#D95F0E", linestyle="--", linewidth=1)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=30, ha="right")
    ax.set_ylabel(f"{residual} residual")
    ax.set_title("Flow residuals")
    if y_transform == "pseudo_log":
        ax.set_yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    return fig


def validate_flow_plot_scatter(
    residuals: Any,
    comparisons: str | Sequence[str] = "adjusted_vs_benchmark",
    methods: Sequence[str] | None = None,
    method_col: str = "method",
    method_labels: Mapping[str, str] | None = None,
    facet_ncol: int | None = None,
    difference_limits: Sequence[float] | None = None,
    difference_quantile: float = 0.95,
    white_band: float = 0.22,
    point_alpha: float = 0.45,
    point_size: float = 1.1,
    point_outline: bool = True,
    point_outline_colour: str = "#6B7280",
    point_stroke: float = 0.12,
    sort: str = "none",
    sort_metric: str = "median_absolute_residual",
) -> Any:
    """Plot flow-value scatter panels colored by signed residual.

    ``residuals`` accepts residual-validation output, method-keyed outputs, or a
    table containing method labels and raw, adjusted, and benchmark flows.
    Options filter methods/comparisons and control facets, residual color limits,
    point styling, and ordering. Returns a Matplotlib ``Figure``. Missing
    columns, invalid limits/options, or an unavailable plotting runtime raise.
    Parity covers audited plot data and visual grammar; pixel-identical rendering
    across Python and R is not claimed.
    """
    plt = _require_matplotlib()
    sort = _match_option(sort, ("none", "ascending", "descending"), "sort")
    sort_metric = _match_option(
        sort_metric,
        ("median_absolute_residual", "mean_absolute_residual", "mean_residual"),
        "sort_metric",
    )
    if facet_ncol is not None:
        if isinstance(facet_ncol, bool) or not isinstance(
            facet_ncol, (int, np.integer)
        ):
            raise ValueError("`facet_ncol` must be `None` or a positive whole number.")
        if int(facet_ncol) < 1:
            raise ValueError("`facet_ncol` must be `None` or a positive whole number.")
    _validate_unit_interval(difference_quantile, "difference_quantile", closed=True)
    if difference_limits is not None:
        limits = np.asarray(difference_limits, dtype=float)
        if (
            limits.shape != (2,)
            or not np.isfinite(limits).all()
            or limits[0] >= limits[1]
        ):
            raise ValueError(
                "`difference_limits` must be a finite numeric vector of length 2 "
                "with the lower limit first."
            )
    else:
        limits = None
    _validate_unit_interval(white_band, "white_band", closed=False)
    _validate_unit_interval(point_alpha, "point_alpha", closed=True)
    _validate_non_negative(point_size, "point_size")
    _validate_bool(point_outline, "point_outline")
    _validate_non_negative(point_stroke, "point_stroke")
    data = _residual_long(
        _filter_methods(
            _as_validation_table(residuals, method_col, "data", "residuals"),
            methods,
            method_col,
            method_labels,
        ),
        "signed",
        comparisons,
        method_col,
        method_labels,
    )
    finite_difference = pd.to_numeric(data["value"], errors="coerce")
    finite_difference = finite_difference.loc[np.isfinite(finite_difference)]
    if limits is None:
        limit = (
            float(finite_difference.abs().quantile(float(difference_quantile)))
            if not finite_difference.empty
            else 1.0
        )
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        limits = np.asarray([-limit, limit], dtype=float)

    from matplotlib.colors import ListedColormap, Normalize

    base_colors = plt.get_cmap("coolwarm")(np.linspace(0, 1, 256))
    zero_position = float(np.clip((0 - limits[0]) / (limits[1] - limits[0]), 0, 1))
    positions = np.linspace(0, 1, len(base_colors))
    base_colors[np.abs(positions - zero_position) <= white_band / 2, :3] = 1.0
    cmap = ListedColormap(base_colors)
    norm = Normalize(vmin=float(limits[0]), vmax=float(limits[1]), clip=True)

    method_order = _residual_method_order(data, sort=sort, sort_metric=sort_metric)
    comparison_order = list(pd.unique(data["comparison"]))
    groups = [
        (method, comparison)
        for method in method_order
        for comparison in comparison_order
        if not data.loc[
            (data["method_label"] == method) & (data["comparison"] == comparison)
        ].empty
    ]
    ncol = int(facet_ncol) if facet_ncol is not None else min(3, max(1, len(groups)))
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(5.2 * ncol, 4.6 * nrow),
        squeeze=False,
    )
    scatter = None
    for ax, (method, comparison) in zip(axes.flat, groups, strict=False):
        panel = data.loc[
            (data["method_label"] == method) & (data["comparison"] == comparison)
        ]
        scatter = ax.scatter(
            panel["x_flow"],
            panel["y_flow"],
            c=panel["value"],
            alpha=point_alpha,
            s=point_size * 20,
            cmap=cmap,
            norm=norm,
            edgecolors=point_outline_colour if point_outline else "none",
            linewidths=point_stroke if point_outline else 0,
        )
        finite = pd.concat([panel["x_flow"], panel["y_flow"]]).dropna()
        if not finite.empty:
            lower = float(finite.min())
            upper = float(finite.max())
            ax.plot(
                [lower, upper],
                [lower, upper],
                color="#4B5563",
                linestyle="--",
                linewidth=1,
            )
        ax.set_xlabel("X flow")
        ax.set_ylabel("Y flow")
        title = method
        if len(comparison_order) > 1:
            title = f"{method} — {_comparison_label(comparison)}"
        ax.set_title(title)
    for ax in axes.flat[len(groups) :]:
        ax.remove()
    if scatter is not None:
        fig.colorbar(scatter, ax=list(axes.flat[: len(groups)]), label="Residual")
    fig.subplots_adjust(wspace=0.3, hspace=0.35, right=0.88)
    return fig


def validate_flow_plot_residual_bands(
    residuals: Any,
    method_col: str = "method",
    method_labels: Mapping[str, str] | None = None,
    comparisons: str | Sequence[str] = "adjusted_vs_benchmark",
    methods: Sequence[str] | None = None,
    band_method: str = "sd",
    sd_reference: str = "selected",
    quantile_reference: str = "selected",
    quantile_probs: Sequence[float] = (0.5, 0.75, 0.9, 0.95),
    palette: Mapping[str, str] | Sequence[str] | None = None,
    label_min_share: float = 8,
    orientation: str = "horizontal",
    sort: str = "none",
    sort_metric: str = "severe_share",
) -> Any:
    """Plot method shares across residual-severity bands.

    ``residuals`` accepts residual-validation output, method-keyed outputs, or a
    table containing method labels and raw, adjusted, and benchmark flows.
    Options define standard-deviation or quantile bands, reference data,
    labels, orientation, colors, filtering, and ordering. Returns a Matplotlib
    ``Figure``. Missing columns, invalid probability/options, or an unavailable
    plotting runtime raise. Parity covers audited plot data and visual grammar,
    not pixel-identical rendering across Python and R.
    """
    plt = _require_matplotlib()
    band_method = _match_option(band_method, ("sd", "quantile"), "band_method")
    sd_reference = _match_option(
        sd_reference, ("selected", "adjusted", "pooled"), "sd_reference"
    )
    quantile_reference = _match_option(
        quantile_reference,
        ("selected", "adjusted", "pooled"),
        "quantile_reference",
    )
    orientation = _match_option(orientation, ("horizontal", "vertical"), "orientation")
    sort = _match_option(sort, ("none", "ascending", "descending"), "sort")
    sort_metric = _match_option(
        sort_metric, ("severe_share", "mean_band_score"), "sort_metric"
    )
    probs = np.asarray(quantile_probs, dtype=float)
    if (
        probs.ndim != 1
        or len(probs) == 0
        or not np.isfinite(probs).all()
        or bool(((probs <= 0) | (probs >= 1)).any())
        or bool((np.diff(probs) <= 0).any())
    ):
        raise ValueError(
            "`quantile_probs` must be a strictly increasing numeric vector with "
            "values between 0 and 1."
        )
    _validate_non_negative(label_min_share, "label_min_share")
    residual_data = _filter_methods(
        _as_validation_table(residuals, method_col, "data", "residuals"),
        methods,
        method_col,
        method_labels,
    )
    _require_columns(
        residual_data,
        [method_col, "mpd_flow", "adj_flow", "benchmark_flow"],
        "residuals",
    )
    signed = _residual_long(
        residual_data,
        "signed",
        comparisons,
        method_col,
        method_labels,
    )
    data = signed.copy()
    data["value"] = data["value"].abs()
    adjusted_residual = residual_data["benchmark_flow"] - residual_data["adj_flow"]
    raw_residual = residual_data["benchmark_flow"] - residual_data["mpd_flow"]
    adjustment_residual = residual_data["adj_flow"] - residual_data["mpd_flow"]
    if band_method == "sd":
        reference = {
            "selected": signed["value"],
            "adjusted": adjusted_residual,
            "pooled": pd.concat(
                [adjusted_residual, raw_residual, adjustment_residual],
                ignore_index=True,
            ),
        }[sd_reference]
        residual_sd = _sample_sd_like_r(reference)
        if not np.isfinite(residual_sd) or residual_sd <= 0:
            residual_sd = 1.0
        score = data["value"].abs() / residual_sd
        band_levels = ["Less than 2 SD", "2 to 3 SD", "3 to 4 SD", "Greater than 4 SD"]
        data["band"] = pd.cut(
            score,
            bins=[-np.inf, 2, 3, 4, np.inf],
            labels=band_levels,
            include_lowest=True,
        )
    else:
        reference = {
            "selected": signed["value"].abs(),
            "adjusted": adjusted_residual.abs(),
            "pooled": pd.concat(
                [adjusted_residual, raw_residual, adjustment_residual],
                ignore_index=True,
            ).abs(),
        }[quantile_reference]
        finite_reference = pd.to_numeric(reference, errors="coerce")
        finite_reference = finite_reference.loc[np.isfinite(finite_reference)]
        if finite_reference.empty:
            raise ValueError(
                "`residuals` must contain at least one finite residual for "
                "quantile bands."
            )
        quantile_breaks = np.quantile(finite_reference, probs)
        keep = ~pd.Series(quantile_breaks).duplicated().to_numpy()
        quantile_breaks = quantile_breaks[keep]
        kept_probs = probs[keep]
        if len(quantile_breaks) == 0:
            band_levels = ["All residuals"]
            data["band"] = "All residuals"
        else:
            band_levels = _quantile_band_labels(kept_probs)
            data["band"] = pd.cut(
                data["value"],
                bins=[-np.inf, *quantile_breaks, np.inf],
                labels=band_levels,
                include_lowest=True,
                right=True,
            )
    data = data.loc[
        np.isfinite(pd.to_numeric(data["value"], errors="coerce"))
        & data["band"].notna()
    ].copy()
    if data.empty:
        raise ValueError("`residuals` must contain at least one finite residual.")
    counts = (
        data.groupby(["method_label", "band"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=band_levels, fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    band_score = np.arange(1, len(band_levels) + 1, dtype=float)
    sort_values = (
        shares.iloc[:, -1]
        if sort_metric == "severe_share"
        else pd.Series(
            shares.to_numpy(dtype=float) @ band_score / 100,
            index=shares.index,
        )
    )
    if sort != "none":
        ordered_index = sort_values.sort_values(ascending=sort == "ascending").index
        shares = shares.reindex(ordered_index)
    colors = _palette(len(band_levels), palette, keys=band_levels)
    positions = np.arange(len(shares.index))
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(shares) * 0.55)))
    cumulative = np.zeros(len(shares))
    for index, band in enumerate(band_levels):
        values = shares[band].to_numpy(dtype=float)
        if orientation == "horizontal":
            bars = ax.barh(
                positions,
                values,
                left=cumulative,
                label=band,
                color=colors[index],
                edgecolor="white",
            )
        else:
            bars = ax.bar(
                positions,
                values,
                bottom=cumulative,
                label=band,
                color=colors[index],
                edgecolor="white",
            )
        for bar, share in zip(bars, values, strict=True):
            if share >= label_min_share:
                if orientation == "horizontal":
                    x = bar.get_x() + bar.get_width() / 2
                    y = bar.get_y() + bar.get_height() / 2
                else:
                    x = bar.get_x() + bar.get_width() / 2
                    y = bar.get_y() + bar.get_height() / 2
                ax.text(x, y, f"{share:.1f}%", ha="center", va="center", fontsize=8)
        cumulative = cumulative + values
    if orientation == "horizontal":
        ax.set_yticks(positions, shares.index)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of OD pairs")
        ax.set_ylabel("")
    else:
        ax.set_xticks(positions, shares.index, rotation=35, ha="right")
        ax.set_ylim(0, 100)
        ax.set_xlabel("")
        ax.set_ylabel("Share of OD pairs")
    ax.legend(title="Residual band", loc="best")
    fig.tight_layout()
    return fig


def validate_flow_plot_distribution(
    distribution_results: Any,
    metric: str = "jsd",
    value: str = "mean",
    comparisons: str | Sequence[str] = "adjusted_vs_benchmark",
    methods: Sequence[str] | None = None,
    method_col: str = "method",
    method_labels: Mapping[str, str] | None = None,
    sort: str = "none",
) -> Any:
    """Plot flow-distribution divergence summaries by method and comparison.

    ``distribution_results`` accepts one result, method-keyed results, or a
    summary table containing method/comparison identifiers and selected KL or
    Jensen-Shannon statistic. Options filter rows and control the statistic and
    ordering. Returns a Matplotlib ``Figure``. Missing columns, invalid options,
    or an unavailable plotting runtime raise. Parity covers audited plot data
    and visual grammar; pixel-identical rendering across Python and R is not
    claimed.
    """
    plt = _require_matplotlib()
    metric = _match_option(metric, ("jsd", "kl"), "metric")
    value = _match_option(value, ("mean", "weighted_mean", "median"), "value")
    sort = _match_option(sort, ("none", "ascending", "descending"), "sort")
    summary = _filter_methods(
        _as_validation_table(
            distribution_results, method_col, "summary", "distribution_results"
        ),
        methods,
        method_col,
        method_labels,
    )
    summary = _filter_comparisons(summary, comparisons)
    value_col = f"{metric}_{value}"
    _require_columns(
        summary, [method_col, "comparison", value_col], "distribution_results"
    )
    summary = summary.assign(
        method_label=summary[method_col].map(lambda x: _method_label(x, method_labels))
    )
    if sort != "none":
        summary = summary.sort_values(value_col, ascending=sort == "ascending")
    summary = summary.assign(
        comparison_label=summary["comparison"].map(_comparison_label)
    )
    method_order = list(pd.unique(summary["method_label"]))
    comparison_order = list(pd.unique(summary["comparison_label"]))
    pivot = summary.pivot_table(
        index="method_label",
        columns="comparison_label",
        values=value_col,
        observed=False,
    ).reindex(index=method_order, columns=comparison_order)
    fig, ax = plt.subplots(
        figsize=(
            max(4.5, len(comparison_order) * 1.8),
            max(3.5, len(method_order) * 0.6),
        )
    )
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(
        range(len(comparison_order)),
        comparison_order,
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(method_order)), method_order)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for y_index, method in enumerate(method_order):
        for x_index, comparison in enumerate(comparison_order):
            value_item = pivot.loc[method, comparison]
            if pd.notna(value_item):
                ax.text(
                    x_index,
                    y_index,
                    f"{float(value_item):.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
    fig.colorbar(image, ax=ax, label=metric.upper())
    fig.tight_layout()
    return fig


def validate_flow_plot_distribution_pairwise(
    distribution_results: Any,
    metric: str = "jsd",
    value: str = "mean",
    comparisons: str | Sequence[str] = "adjusted_vs_benchmark",
    methods: Sequence[str] | None = None,
    method_col: str = "method",
    method_labels: Mapping[str, str] | None = None,
    plot_type: str = "comparison",
    sort: str = "none",
    mirror_jsd: bool = True,
) -> Any:
    """Plot pairwise flow-distribution divergences as bars or heatmaps.

    ``distribution_results`` accepts one result, method-keyed results, or a
    summary table with method, reference/comparison distribution, and selected
    divergence columns. Options filter comparisons/methods and control statistic,
    layout, ordering, labels, and symmetric JSD mirroring. Returns a Matplotlib
    ``Figure``. Missing columns, invalid options, or an unavailable plotting
    runtime raise. Parity covers audited plot data and visual grammar, not
    pixel-identical rendering across Python and R.
    """
    plt = _require_matplotlib()
    metric = _match_option(metric, ("jsd", "kl"), "metric")
    value = _match_option(value, ("mean", "weighted_mean", "median"), "value")
    plot_type = _match_option(plot_type, ("comparison", "heatmap"), "plot_type")
    sort = _match_option(sort, ("none", "ascending", "descending"), "sort")
    _validate_bool(mirror_jsd, "mirror_jsd")
    summary = _filter_methods(
        _filter_comparisons(
            _as_validation_table(
                distribution_results, method_col, "summary", "distribution_results"
            ),
            comparisons,
        ),
        methods,
        method_col,
        method_labels,
    )
    value_col = f"{metric}_{value}"
    _require_columns(
        summary,
        [method_col, "reference_distribution", "comparison_distribution", value_col],
        "distribution_results",
    )
    summary = summary.assign(
        method_label=summary[method_col].map(lambda x: _method_label(x, method_labels))
    )
    method_order = _method_order_from_values(
        data=summary,
        value_col=value_col,
        method_col=method_col,
        method_labels=method_labels,
        sort=sort,
    )
    if plot_type == "comparison":
        comparison_order = list(pd.unique(summary["comparison"]))
        fig, ax = plt.subplots(figsize=(7, max(3.5, len(method_order) * 0.6)))
        height = 0.72 / max(1, len(comparison_order))
        y_base = np.arange(len(method_order), dtype=float)
        colors = _palette(len(comparison_order), None)
        for index, comparison in enumerate(comparison_order):
            subset = summary.loc[summary["comparison"] == comparison]
            values = [
                pd.to_numeric(
                    subset.loc[subset["method_label"] == method, value_col],
                    errors="coerce",
                ).mean()
                for method in method_order
            ]
            offset = (index - (len(comparison_order) - 1) / 2) * height
            ax.barh(
                y_base + offset,
                values,
                height=height,
                label=_comparison_label(comparison),
                color=colors[index],
                edgecolor="white",
            )
        ax.set_yticks(y_base, method_order)
        ax.set_xlabel(value_col)
        ax.set_ylabel("")
        if len(comparison_order) > 1:
            ax.legend(title="Comparison")
        fig.tight_layout()
        return fig

    panels: list[tuple[str, pd.DataFrame]] = []
    all_values: list[float] = []
    for method in method_order:
        panel = summary.loc[summary["method_label"] == method].copy()
        if metric == "jsd" and mirror_jsd:
            mirrored = panel.copy()
            mirrored[["reference_distribution", "comparison_distribution"]] = panel[
                ["comparison_distribution", "reference_distribution"]
            ].to_numpy()
            panel = pd.concat([panel, mirrored], ignore_index=True).drop_duplicates(
                ["reference_distribution", "comparison_distribution"]
            )
        distributions = sorted(
            set(panel["reference_distribution"].astype(str))
            | set(panel["comparison_distribution"].astype(str))
        )
        diagonal = pd.DataFrame(
            {
                "reference_distribution": distributions,
                "comparison_distribution": distributions,
                value_col: np.zeros(len(distributions)),
            }
        )
        panel = pd.concat([panel, diagonal], ignore_index=True).drop_duplicates(
            ["reference_distribution", "comparison_distribution"], keep="first"
        )
        pivot = panel.pivot_table(
            index="reference_distribution",
            columns="comparison_distribution",
            values=value_col,
            observed=False,
        )
        panels.append((method, pivot))
        all_values.extend(
            pd.to_numeric(panel[value_col], errors="coerce").dropna().astype(float)
        )
    ncol = min(3, max(1, len(panels)))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    vmax = max(all_values, default=1.0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    image = None
    for ax, (method, pivot) in zip(axes.flat, panels, strict=False):
        image = ax.imshow(
            pivot.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=vmax
        )
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title(method)
    for ax in axes.flat[len(panels) :]:
        ax.remove()
    if image is not None:
        fig.colorbar(image, ax=list(axes.flat[: len(panels)]), label=value_col)
    fig.subplots_adjust(wspace=0.3, hspace=0.35, right=0.88)
    return fig


def validate_flow_plot_structure(
    structure_results: Any,
    comparisons: str | Sequence[str] = "adjusted_vs_benchmark",
    methods: Sequence[str] | None = None,
    method_col: str = "method",
    method_labels: Mapping[str, str] | None = None,
    palette: Mapping[str, str] | Sequence[str] | None = None,
    near_zero_band: float | None = 0.1,
    show_value_labels: bool = True,
    value_digits: int = 2,
    sort: str = "none",
    sort_metric: str | None = None,
) -> Any:
    """Plot correlation and spatial-autocorrelation structure diagnostics.

    ``structure_results`` accepts one residual-structure result, method-keyed
    results, or a summary table with method/comparison and the three supported
    diagnostic columns. Options filter rows and control labels, near-zero band,
    palette, value precision, and ordering. Returns a Matplotlib ``Figure``.
    Missing columns, invalid options, or an unavailable plotting runtime raise.
    Parity covers audited plot data and visual grammar; pixel-identical rendering
    across Python and R is not claimed.
    """
    plt = _require_matplotlib()
    sort = _match_option(sort, ("none", "ascending", "descending"), "sort")
    if near_zero_band is not None:
        _validate_non_negative(near_zero_band, "near_zero_band")
    _validate_bool(show_value_labels, "show_value_labels")
    if isinstance(value_digits, bool) or not isinstance(
        value_digits, (int, np.integer)
    ):
        raise ValueError("`value_digits` must be a single non-negative whole number.")
    if int(value_digits) < 0:
        raise ValueError("`value_digits` must be a single non-negative whole number.")
    summary = _filter_methods(
        _filter_comparisons(
            _as_validation_table(
                structure_results, method_col, "summary", "structure_results"
            ),
            comparisons,
        ),
        methods,
        method_col,
        method_labels,
    )
    metric_cols = [
        "pearson_residual_benchmark_flow",
        "moran_i",
        "pearson_residual_covariate",
    ]
    _require_columns(summary, [method_col, *metric_cols], "structure_results")
    sort_metric = metric_cols[0] if sort_metric is None else sort_metric
    if sort_metric not in metric_cols:
        raise ValueError(
            "`sort_metric` must name one of the plotted structure metrics."
        )
    summary = _prepare_comparison_plot_rows(
        summary,
        method_col=method_col,
        method_labels=method_labels,
        integrate_benchmark=False,
    )
    plot_data = summary.melt(
        id_vars=[
            "_method_label",
            "comparison",
            "_comparison_display",
            "_comparison_title",
        ],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    plot_data = plot_data.rename(columns={"_method_label": "method_label"})
    metric_labels = {
        "pearson_residual_benchmark_flow": "Residual-flow correlation (r)",
        "moran_i": "Spatial autocorrelation (Moran's I)",
        "pearson_residual_covariate": "Residual-covariate correlation (r)",
    }
    plot_data["metric_label"] = plot_data["metric"].map(metric_labels)
    method_order = _comparison_method_order(
        data=summary,
        value_col=sort_metric,
        sort=sort,
    )
    metric_order = [metric_labels[col] for col in metric_cols]
    comparison_order = list(pd.unique(plot_data["_comparison_display"]))
    fig, axes = plt.subplots(
        len(comparison_order),
        1,
        figsize=(7, sum(max(3.5, len(method_order) * 0.7) for _ in comparison_order)),
        squeeze=False,
    )
    offsets = np.linspace(-0.18, 0.18, len(metric_order))
    colors = _palette(len(method_order), palette, keys=method_order)
    color_by_method = dict(zip(method_order, colors, strict=True))
    for ax, comparison in zip(axes.flat, comparison_order, strict=True):
        panel = plot_data.loc[
            plot_data["_comparison_display"] == comparison
        ].drop_duplicates(["method_label", "metric_label"], keep="first")
        panel_order = [
            method for method in method_order if method in set(panel["method_label"])
        ]
        value_pivot = panel.pivot(
            index="method_label", columns="metric_label", values="value"
        ).reindex(index=panel_order, columns=metric_order)
        y_base = np.arange(len(panel_order))
        ax.axvline(0, color="#4B5563", linestyle="--", linewidth=1)
        if near_zero_band is not None and near_zero_band > 0:
            ax.axvspan(
                -float(near_zero_band),
                float(near_zero_band),
                color="#E5E7EB",
                alpha=0.35,
            )
        for metric_index, metric_label in enumerate(metric_order):
            for method_index, method in enumerate(panel_order):
                value_item = value_pivot.loc[method, metric_label]
                color = color_by_method[method]
                ax.plot(
                    [0, value_item],
                    [y_base[method_index] + offsets[metric_index]] * 2,
                    color=color,
                    linewidth=0.8,
                    alpha=0.8,
                )
                ax.scatter(
                    [value_item],
                    [y_base[method_index] + offsets[metric_index]],
                    label=metric_label if method_index == 0 else None,
                    color=color,
                    s=45,
                )
                if show_value_labels and np.isfinite(value_item):
                    offset = 0.02 if value_item >= 0 else -0.02
                    ax.text(
                        value_item + offset,
                        y_base[method_index] + offsets[metric_index],
                        f"{float(value_item):.{int(value_digits)}f}",
                        ha="left" if value_item >= 0 else "right",
                        va="center",
                        fontsize=8,
                        color="#374151",
                    )
        ax.set_yticks(y_base, panel_order)
        ax.set_xlabel("Diagnostic value")
        ax.set_ylabel("")
        ax.set_xlim(-1, 1)
        if len(comparison_order) > 1:
            ax.set_title(str(panel["_comparison_title"].iloc[0]))
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def validate_flow_plot_lisa_map(
    structure_results: Any,
    boundaries: Any,
    boundary_area_col: str = "area",
    area_col: str = "area",
    comparisons: str | Sequence[str] = "adjusted_vs_benchmark",
    methods: Sequence[str] | None = None,
    method_col: str = "method",
    method_labels: Mapping[str, str] | None = None,
    cluster_col: str = "lisa_cluster",
    p_value_col: str = "p_value",
    p_value_threshold: float | None = 0.05,
    palette: Mapping[str, str] | Sequence[str] | None = None,
    missing_fill: str = "#F3F4F6",
    boundary_colour: str | None = None,
    boundary_linewidth: float = 0,
    outline_boundaries: Any | None = None,
    outline_colour: str = "white",
    outline_linewidth: float = 0.45,
    sort: str = "none",
    sort_metric: str = "significant_area_share",
) -> Any:
    """Map local Moran cluster classifications for residual diagnostics.

    ``structure_results`` must provide area-level map/local-Moran rows containing
    method, comparison, area, cluster, and p-value fields. ``boundaries`` must
    be GeoDataFrame-like with ``geometry`` and ``boundary_area_col``; optional
    outlines follow the same geometry contract. Styling, significance, filtering,
    and ordering options control panels. Returns a Matplotlib ``Figure``.
    Invalid geometry, columns, thresholds/options, or plotting dependencies
    raise. Parity covers audited map data and visual grammar, not pixel-identical
    rendering across Python and R.
    """
    plt = _require_matplotlib()
    sort = _match_option(sort, ("none", "ascending", "descending"), "sort")
    sort_metric = _match_option(
        sort_metric,
        ("significant_area_share", "mean_abs_local_moran", "mean_local_moran"),
        "sort_metric",
    )
    if not hasattr(boundaries, "plot") or "geometry" not in boundaries.columns:
        raise ValueError("`boundaries` must be a GeoDataFrame-like object.")
    if boundary_area_col not in boundaries.columns:
        raise ValueError("`boundary_area_col` must name a column in `boundaries`.")
    if outline_boundaries is not None and (
        not hasattr(outline_boundaries, "plot")
        or "geometry" not in outline_boundaries.columns
    ):
        raise ValueError("`outline_boundaries` must be a GeoDataFrame-like object.")
    boundary_crs = getattr(boundaries, "crs", None)
    outline_crs = getattr(outline_boundaries, "crs", None)
    if (
        boundary_crs is not None
        and outline_crs is not None
        and boundary_crs != outline_crs
    ):
        outline_boundaries = outline_boundaries.to_crs(boundary_crs)
    _validate_non_negative(boundary_linewidth, "boundary_linewidth")
    _validate_non_negative(outline_linewidth, "outline_linewidth")
    map_data = _filter_methods(
        _filter_comparisons(
            _as_validation_table(
                structure_results, method_col, "map_data", "structure_results"
            ),
            comparisons,
        ),
        methods,
        method_col,
        method_labels,
    )
    _require_columns(map_data, [area_col, method_col, cluster_col], "structure_results")
    if map_data.duplicated([method_col, "comparison", area_col]).any():
        raise ValueError(
            "`structure_results` must contain one row per method, comparison and area."
        )
    if p_value_threshold is not None:
        _validate_probability(p_value_threshold, "p_value_threshold")
        _require_columns(map_data, [p_value_col], "structure_results")
        if not pd.api.types.is_numeric_dtype(map_data[p_value_col]):
            raise ValueError("`p_value_col` must identify a numeric p-value column.")
    map_data = map_data.copy()
    map_data["method_label"] = map_data[method_col].map(
        lambda value: _method_label(value, method_labels)
    )
    map_data["_cluster"] = map_data[cluster_col].astype("string")
    if p_value_threshold is not None:
        p_values = pd.to_numeric(map_data[p_value_col], errors="coerce")
        special = map_data["_cluster"].isin(["no neighbours", "undefined"])
        significant = np.isfinite(p_values) & (p_values <= float(p_value_threshold))
        map_data.loc[~special & ~significant, "_cluster"] = "not significant"
    if "local_moran_i" not in map_data.columns:
        map_data["local_moran_i"] = np.nan

    if not bool(boundaries[boundary_area_col].isin(map_data[area_col]).any()):
        raise ValueError(
            "`boundaries` and `structure_results` do not share any area identifiers."
        )
    sort_values = map_data.assign(
        _sort_value=(
            pd.to_numeric(map_data["local_moran_i"], errors="coerce").abs()
            if sort_metric == "mean_abs_local_moran"
            else pd.to_numeric(map_data["local_moran_i"], errors="coerce")
            if sort_metric == "mean_local_moran"
            else ~map_data["_cluster"].isin(
                ["not significant", "no neighbours", "undefined"]
            )
        )
    )
    method_order = list(pd.unique(map_data["method_label"]))
    if sort != "none":
        method_order = (
            sort_values.groupby("method_label", sort=False)["_sort_value"]
            .mean()
            .sort_values(ascending=sort == "ascending")
            .index.tolist()
        )
    comparison_order = list(pd.unique(map_data["comparison"]))
    panels = [
        (method, comparison)
        for method in method_order
        for comparison in comparison_order
        if not map_data.loc[
            (map_data["method_label"] == method)
            & (map_data["comparison"] == comparison)
        ].empty
    ]
    ncol = min(3, max(1, len(panels)))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow), squeeze=False)
    edgecolor = "none" if boundary_colour is None else boundary_colour
    from matplotlib.patches import Patch

    default_colors = {
        "high-high": "#fa8775",
        "low-low": "#0000ff",
        "high-low": "#ffb14e",
        "low-high": "#9d02d7",
        "not significant": "#D1D5DB",
        "no neighbours": "#F3F4F6",
        "undefined": "#6B7280",
    }
    present_clusters = list(pd.unique(map_data["_cluster"].dropna().astype(str)))
    all_clusters = [
        cluster for cluster in default_colors if cluster in present_clusters
    ]
    all_clusters += [
        cluster for cluster in present_clusters if cluster not in default_colors
    ]
    cluster_colors = dict(
        zip(
            all_clusters,
            _palette(
                len(all_clusters),
                default_colors if palette is None else palette,
                keys=all_clusters,
            ),
            strict=True,
        )
    )
    for ax, (method, comparison) in zip(axes.flat, panels, strict=False):
        panel = map_data.loc[
            (map_data["method_label"] == method)
            & (map_data["comparison"] == comparison)
        ]
        joined = boundaries.merge(
            panel,
            how="left",
            left_on=boundary_area_col,
            right_on=area_col,
        )
        clusters = list(pd.unique(panel["_cluster"].dropna().astype(str)))
        colors = (
            joined["_cluster"].astype("string").map(cluster_colors).fillna(missing_fill)
        )
        joined.plot(
            color=colors.tolist(),
            edgecolor=edgecolor,
            linewidth=float(boundary_linewidth),
            ax=ax,
        )
        if outline_boundaries is not None:
            outline_boundaries.plot(
                ax=ax,
                facecolor="none",
                edgecolor=outline_colour,
                linewidth=float(outline_linewidth),
            )
        if clusters:
            ax.legend(
                handles=[
                    Patch(facecolor=cluster_colors[cluster], label=cluster)
                    for cluster in clusters
                ],
                title="LISA cluster",
                loc="lower center",
            )
        ax.set_axis_off()
        title = method
        if len(comparison_order) > 1:
            title = f"{method} — {_comparison_label(comparison)}"
        ax.set_title(title)
    for ax in axes.flat[len(panels) :]:
        ax.remove()
    fig.tight_layout()
    return fig


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Validation plotting functions require matplotlib. Install debiaspy[plot]."
        ) from exc
    return plt


def _prepare_comparison_plot_rows(
    data: pd.DataFrame,
    *,
    method_col: str,
    method_labels: Mapping[str, str] | None,
    integrate_benchmark: bool,
) -> pd.DataFrame:
    out = data.copy()
    out["_method_label"] = out[method_col].map(
        lambda value: _method_label(value, method_labels)
    )
    raw_benchmark = out["comparison"] == "raw_vs_benchmark"
    out.loc[raw_benchmark, "_method_label"] = "Unadjusted raw MPD"
    out = out.drop_duplicates(
        ["comparison", "_method_label"], keep="first"
    ).reset_index(drop=True)
    out["_comparison_display"] = out["comparison"].astype(str)
    out["_comparison_title"] = out["comparison"].map(_comparison_label)
    comparisons = set(out["comparison"].astype(str))
    if (
        integrate_benchmark
        and {
            "adjusted_vs_benchmark",
            "raw_vs_benchmark",
        }
        <= comparisons
    ):
        benchmark_rows = out["comparison"].isin(
            ["adjusted_vs_benchmark", "raw_vs_benchmark"]
        )
        out.loc[benchmark_rows, "_comparison_display"] = "benchmark_comparison"
        out.loc[benchmark_rows, "_comparison_title"] = (
            "Adjusted methods and raw MPD vs benchmark"
        )
    return out


def _comparison_method_order(
    data: pd.DataFrame,
    *,
    value_col: str,
    sort: str,
) -> list[str]:
    labels = data["_method_label"].astype(str)
    baseline_label = "Unadjusted raw MPD"
    baseline = [baseline_label] if bool((labels == baseline_label).any()) else []
    if sort == "none":
        non_baseline = [label for label in pd.unique(labels) if label != baseline_label]
        return [*non_baseline, *baseline]
    values = pd.to_numeric(data[value_col], errors="coerce")
    summary = (
        pd.DataFrame({"label": labels, "value": values})
        .loc[labels != baseline_label]
        .groupby("label", sort=False, dropna=False)["value"]
        .mean()
        .sort_values(ascending=sort == "ascending", na_position="last")
    )
    return [*[str(label) for label in summary.index], *baseline]


def _filter_methods(
    data: pd.DataFrame,
    methods: Sequence[str] | None,
    method_col: str,
    method_labels: Mapping[str, str] | None,
) -> pd.DataFrame:
    if methods is None:
        return data
    labels = data[method_col].map(lambda value: _method_label(value, method_labels))
    keep = data[method_col].astype(str).isin(methods) | labels.isin(methods)
    out = data.loc[keep].copy()
    if out.empty:
        raise ValueError("`methods` did not match any method identifiers or labels.")
    return out


def _filter_comparisons(
    data: pd.DataFrame, comparisons: str | Sequence[str]
) -> pd.DataFrame:
    if "comparison" not in data.columns:
        data = data.copy()
        data["comparison"] = "adjusted_vs_benchmark"
    selected = (
        ["adjusted_vs_benchmark", "raw_vs_benchmark", "raw_vs_adjusted"]
        if comparisons == "all"
        else ([comparisons] if isinstance(comparisons, str) else list(comparisons))
    )
    valid = {"adjusted_vs_benchmark", "raw_vs_benchmark", "raw_vs_adjusted"}
    bad = [comparison for comparison in selected if comparison not in valid]
    if bad:
        raise ValueError(
            "`comparisons` contains unsupported value(s): " + ", ".join(bad)
        )
    out = data.loc[data["comparison"].isin(selected)].copy()
    if out.empty:
        raise ValueError("`comparisons` did not match any rows in the input data.")
    return out


def _residual_long(
    residual_data: pd.DataFrame,
    residual: str,
    comparisons: str | Sequence[str],
    method_col: str,
    method_labels: Mapping[str, str] | None,
) -> pd.DataFrame:
    if residual not in {"signed", "absolute", "percent"}:
        raise ValueError("`residual` must be one of: signed, absolute, percent")
    comparison_ids = (
        ["adjusted_vs_benchmark", "raw_vs_benchmark", "raw_vs_adjusted"]
        if comparisons == "all"
        else ([comparisons] if isinstance(comparisons, str) else list(comparisons))
    )
    valid = {"adjusted_vs_benchmark", "raw_vs_benchmark", "raw_vs_adjusted"}
    bad = [comparison for comparison in comparison_ids if comparison not in valid]
    if bad:
        raise ValueError(
            "`comparisons` contains unsupported value(s): " + ", ".join(bad)
        )
    rows = []
    for comparison in comparison_ids:
        if comparison == "adjusted_vs_benchmark":
            x_col, y_col = "adj_flow", "benchmark_flow"
        elif comparison == "raw_vs_benchmark":
            x_col, y_col = "mpd_flow", "benchmark_flow"
        else:
            x_col, y_col = "mpd_flow", "adj_flow"
        _require_columns(residual_data, [method_col, x_col, y_col], "residuals")
        comparison_data = residual_data
        if comparison == "raw_vs_benchmark":
            dedupe_cols = (
                ["origin", "destination"]
                if {"origin", "destination"} <= set(residual_data.columns)
                else [x_col, y_col]
            )
            comparison_data = residual_data.drop_duplicates(dedupe_cols)
        diff = comparison_data[y_col] - comparison_data[x_col]
        if residual == "absolute":
            value = diff.abs()
        elif residual == "percent":
            value = 100 * diff / comparison_data[x_col].replace(0, np.nan)
        else:
            value = diff
        if comparison == "raw_vs_benchmark":
            method = pd.Series("raw_mpd_baseline", index=comparison_data.index)
            method_label = pd.Series("Unadjusted raw MPD", index=comparison_data.index)
        else:
            method = comparison_data[method_col]
            method_label = comparison_data[method_col].map(
                lambda x: _method_label(x, method_labels)
            )
        frame = pd.DataFrame(
            {
                "method": method,
                "method_label": method_label,
                "comparison": comparison,
                "x_flow": comparison_data[x_col],
                "y_flow": comparison_data[y_col],
                "value": value,
            }
        ).reset_index(drop=True)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _metric_label(metric: str) -> str:
    labels = {
        "mae": "MAE",
        "rmse": "RMSE",
        "mape": "MAPE",
        "mean_error": "Mean error",
    }
    return labels.get(metric, metric)


def _normalise_metric_name(metric: object) -> str:
    text = str(metric)
    aliases = {
        "MAE": "mae",
        "Mean absolute error": "mae",
        "RMSE": "rmse",
        "Root mean squared error": "rmse",
        "MAPE": "mape",
        "Mean absolute percentage error": "mape",
        "Mean error": "mean_error",
        "Pearson r": "pearson_r",
        "Spearman rho": "spearman_rho",
        "R-squared": "r_squared",
    }
    return aliases.get(text, text)


def _comparison_label(comparison: object) -> str:
    labels = {
        "adjusted_vs_benchmark": "Adjusted vs benchmark",
        "raw_vs_benchmark": "Raw MPD vs benchmark",
        "raw_vs_adjusted": "Raw MPD vs adjusted",
    }
    return labels.get(str(comparison), str(comparison))


def _format_plot_value(value: object) -> str:
    if pd.isna(value):
        return ""
    numeric = float(value)
    if abs(numeric) >= 100:
        return f"{numeric:.0f}"
    if abs(numeric) >= 10:
        return f"{numeric:.1f}"
    return f"{numeric:.3f}"


def _palette(
    count: int,
    palette: Mapping[str, str] | Sequence[str] | None,
    keys: Sequence[object] | None = None,
) -> list[str]:
    default = [
        "#ffd700",
        "#ffb14e",
        "#fa8775",
        "#ea5f94",
        "#cd34b5",
        "#9d02d7",
        "#0000ff",
    ]
    if isinstance(palette, Mapping):
        defaults = _palette(count, None)
        requested_keys = list(keys) if keys is not None else list(palette)
        return [
            palette.get(key, palette.get(str(key), defaults[index]))
            for index, key in enumerate(requested_keys[:count])
        ] + defaults[len(requested_keys[:count]) : count]
    if palette is not None:
        values = [palette] if isinstance(palette, str) else list(palette)
        if values:
            repeats = int(np.ceil(count / len(values)))
            return (values * repeats)[:count]
    repeats = int(np.ceil(count / len(default)))
    return (default * repeats)[:count]


def _match_option(value: object, choices: Sequence[str], name: str) -> str:
    return match_choice(value, choices, name)


def _validate_bool(value: object, name: str) -> None:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"`{name}` must be `True` or `False`.")


def _numeric_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"`{name}` must be numeric.")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"`{name}` must be numeric.") from None
    if not np.isfinite(numeric):
        raise ValueError(f"`{name}` must be finite.")
    return numeric


def _validate_non_negative(value: object, name: str) -> float:
    numeric = _numeric_scalar(value, name)
    if numeric < 0:
        raise ValueError(f"`{name}` must be non-negative.")
    return numeric


def _validate_unit_interval(value: object, name: str, closed: bool) -> float:
    numeric = _numeric_scalar(value, name)
    valid = 0 <= numeric <= 1 if closed else 0 <= numeric < 1
    if not valid:
        interval = "[0, 1]" if closed else "[0, 1)"
        raise ValueError(f"`{name}` must be in the interval {interval}.")
    return numeric


def _validate_probability(value: object, name: str) -> float:
    return _validate_unit_interval(value, name, closed=True)


def _validate_relative_error_breaks(breaks: Sequence[float]) -> None:
    try:
        values = np.asarray(breaks, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            "`relative_error_breaks` must be an increasing numeric vector "
            "covering 0 to 100."
        ) from None
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or bool((np.diff(values) <= 0).any())
        or values.min() > 0
        or values.max() < 100
    ):
        raise ValueError(
            "`relative_error_breaks` must be an increasing numeric vector "
            "covering 0 to 100."
        )


def _method_order_from_values(
    data: pd.DataFrame,
    value_col: str,
    method_col: str,
    method_labels: Mapping[str, str] | None,
    sort: str,
) -> list[str]:
    labels = data[method_col].map(lambda value: _method_label(value, method_labels))
    order = list(pd.unique(labels))
    if sort == "none":
        return order
    values = pd.to_numeric(data[value_col], errors="coerce")
    summary = (
        pd.DataFrame({"label": labels, "value": values})
        .groupby("label", sort=False, dropna=False)["value"]
        .mean()
    )
    return [
        str(value)
        for value in summary.sort_values(ascending=sort == "ascending").index.tolist()
    ]


def _residual_method_order(
    data: pd.DataFrame, sort: str, sort_metric: str
) -> list[str]:
    order = list(pd.unique(data["method_label"]))
    if sort == "none":
        return order
    values = pd.to_numeric(data["value"], errors="coerce")
    sort_values = values if sort_metric == "mean_residual" else values.abs()
    summary = pd.DataFrame(
        {"method_label": data["method_label"], "value": sort_values}
    ).groupby("method_label", sort=False)["value"]
    reduced = (
        summary.median()
        if sort_metric == "median_absolute_residual"
        else summary.mean()
    )
    return [
        str(value)
        for value in reduced.sort_values(ascending=sort == "ascending").index.tolist()
    ]


def _thin_plot_series(series: pd.Series, max_points: float) -> pd.Series:
    if not np.isfinite(max_points):
        return series
    count = int(max_points)
    if count <= 0:
        return series.iloc[0:0]
    if len(series) <= count:
        return series
    positions = np.unique(np.rint(np.linspace(0, len(series) - 1, count)).astype(int))
    return series.iloc[positions]


def _sample_sd_like_r(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan
    with np.errstate(invalid="ignore"):
        return float(values.std(ddof=1))


def _quantile_band_labels(probs: Sequence[float]) -> list[str]:
    labels = [f"<= P{float(probs[0]) * 100:.0f}"]
    labels.extend(
        f"P{float(lower) * 100:.0f} to P{float(upper) * 100:.0f}"
        for lower, upper in zip(probs[:-1], probs[1:], strict=True)
    )
    labels.append(f"> P{float(probs[-1]) * 100:.0f}")
    return labels


def _method_label(value: object, labels: Mapping[str, str] | None) -> str:
    if pd.isna(value):
        value = "method"
    text = str(value)
    return text if labels is None else labels.get(text, text)


def _require_columns(data: pd.DataFrame, columns: Sequence[str], arg: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"`{arg}` must contain: {', '.join(missing)}")


def _as_validation_table(
    results: Any, method_col: str, key: str, argument: str
) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        out = results.copy()
    elif isinstance(results, Mapping) and isinstance(results.get(key), pd.DataFrame):
        out = results[key].copy()
    elif (
        isinstance(results, Mapping)
        and key == "summary"
        and {"mae", "rmse", "mape"} <= set(results)
    ):
        out = pd.DataFrame(
            [{k: v for k, v in results.items() if not isinstance(v, pd.DataFrame)}]
        )
    elif isinstance(results, Mapping):
        rows = []
        for name, item in results.items():
            table = _as_validation_table(item, method_col, key, argument)
            if method_col not in table or table[method_col].isna().all():
                table[method_col] = name
            rows.append(table)
        out = pd.concat(rows, ignore_index=True)
    else:
        raise ValueError(
            f"`{argument}` must contain a data frame "
            f"or a validation result with `{key}`."
        )
    if method_col not in out:
        out[method_col] = "method"
    return out
