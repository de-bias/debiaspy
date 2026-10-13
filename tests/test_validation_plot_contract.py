# ruff: noqa: E402, I001
import inspect
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection, PolyCollection

from debiaspy import (
    validate_flow_plot_distribution,
    validate_flow_plot_distribution_pairwise,
    validate_flow_plot_lisa_map,
    validate_flow_plot_metrics,
    validate_flow_plot_residual_bands,
    validate_flow_plot_residuals,
    validate_flow_plot_scatter,
    validate_flow_plot_structure,
)


PLOT_SIGNATURES = {
    validate_flow_plot_metrics: [
        "metrics",
        "error_measures",
        "metric_cols",
        "comparisons",
        "methods",
        "method_col",
        "method_family_col",
        "method_labels",
        "palette",
        "relative_error_breaks",
        "sort",
        "sort_metric",
    ],
    validate_flow_plot_residuals: [
        "residuals",
        "residual",
        "comparisons",
        "methods",
        "method_col",
        "method_labels",
        "palette",
        "show_points",
        "show_counts",
        "point_alpha",
        "point_size",
        "jitter_width",
        "violin_alpha",
        "max_points",
        "y_transform",
        "sort",
        "sort_metric",
    ],
    validate_flow_plot_scatter: [
        "residuals",
        "comparisons",
        "methods",
        "method_col",
        "method_labels",
        "facet_ncol",
        "difference_limits",
        "difference_quantile",
        "white_band",
        "point_alpha",
        "point_size",
        "point_outline",
        "point_outline_colour",
        "point_stroke",
        "sort",
        "sort_metric",
    ],
    validate_flow_plot_residual_bands: [
        "residuals",
        "method_col",
        "method_labels",
        "comparisons",
        "methods",
        "band_method",
        "sd_reference",
        "quantile_reference",
        "quantile_probs",
        "palette",
        "label_min_share",
        "orientation",
        "sort",
        "sort_metric",
    ],
    validate_flow_plot_distribution: [
        "distribution_results",
        "metric",
        "value",
        "comparisons",
        "methods",
        "method_col",
        "method_labels",
        "sort",
    ],
    validate_flow_plot_distribution_pairwise: [
        "distribution_results",
        "metric",
        "value",
        "comparisons",
        "methods",
        "method_col",
        "method_labels",
        "plot_type",
        "sort",
        "mirror_jsd",
    ],
    validate_flow_plot_structure: [
        "structure_results",
        "comparisons",
        "methods",
        "method_col",
        "method_labels",
        "palette",
        "near_zero_band",
        "show_value_labels",
        "value_digits",
        "sort",
        "sort_metric",
    ],
    validate_flow_plot_lisa_map: [
        "structure_results",
        "boundaries",
        "boundary_area_col",
        "area_col",
        "comparisons",
        "methods",
        "method_col",
        "method_labels",
        "cluster_col",
        "p_value_col",
        "p_value_threshold",
        "palette",
        "missing_fill",
        "boundary_colour",
        "boundary_linewidth",
        "outline_boundaries",
        "outline_colour",
        "outline_linewidth",
        "sort",
        "sort_metric",
    ],
}

PLOT_DEFAULTS = {
    validate_flow_plot_metrics: {
        "error_measures": ("mae", "rmse", "mape"),
        "relative_error_breaks": tuple(range(0, 101, 10)),
        "sort": "none",
    },
    validate_flow_plot_residuals: {
        "residual": "signed",
        "show_points": True,
        "show_counts": True,
        "y_transform": "pseudo_log",
    },
    validate_flow_plot_scatter: {
        "difference_quantile": 0.95,
        "white_band": 0.22,
        "point_outline": True,
    },
    validate_flow_plot_residual_bands: {
        "band_method": "sd",
        "sd_reference": "selected",
        "orientation": "horizontal",
    },
    validate_flow_plot_distribution: {"metric": "jsd", "value": "mean"},
    validate_flow_plot_distribution_pairwise: {
        "plot_type": "comparison",
        "mirror_jsd": True,
    },
    validate_flow_plot_structure: {
        "near_zero_band": 0.1,
        "show_value_labels": True,
        "value_digits": 2,
    },
    validate_flow_plot_lisa_map: {
        "p_value_threshold": 0.05,
        "missing_fill": "#F3F4F6",
        "outline_linewidth": 0.45,
    },
}


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["m1", "m2"],
            "comparison": ["adjusted_vs_benchmark"] * 2,
            "mae": [10.0, 5.0],
            "rmse": [12.0, 8.0],
            "mape": [20.0, 10.0],
            "family": ["a", "b"],
        }
    )


def _residuals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["m1"] * 4 + ["m2"] * 4,
            "origin": ["A", "A", "B", "B"] * 2,
            "destination": ["X", "Y", "X", "Y"] * 2,
            "mpd_flow": [8.0, 10.0, 15.0, 20.0] * 2,
            "adj_flow": [9.0, 12.0, 13.0, 22.0, 8.5, 11.0, 14.0, 21.0],
            "benchmark_flow": [10.0, 11.0, 14.0, 18.0] * 2,
        }
    )


def _distribution() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["m1", "m2", "m1", "m2"],
            "comparison": [
                "adjusted_vs_benchmark",
                "adjusted_vs_benchmark",
                "raw_vs_adjusted",
                "raw_vs_adjusted",
            ],
            "reference_distribution": [
                "benchmark",
                "benchmark",
                "adjusted_mpd",
                "adjusted_mpd",
            ],
            "comparison_distribution": [
                "adjusted_mpd",
                "adjusted_mpd",
                "raw_mpd",
                "raw_mpd",
            ],
            "jsd_mean": [0.1, 0.2, 0.3, 0.4],
            "jsd_weighted_mean": [0.11, 0.21, 0.31, 0.41],
            "jsd_median": [0.09, 0.19, 0.29, 0.39],
            "kl_mean": [0.2, 0.4, 0.6, 0.8],
            "kl_weighted_mean": [0.21, 0.41, 0.61, 0.81],
            "kl_median": [0.19, 0.39, 0.59, 0.79],
        }
    )


def _structure() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["m1", "m2"],
            "comparison": ["adjusted_vs_benchmark"] * 2,
            "pearson_residual_benchmark_flow": [0.12, -0.34],
            "moran_i": [0.2, -0.1],
            "pearson_residual_covariate": [0.05, 0.4],
        }
    )


@pytest.mark.parametrize(("function", "parameters"), PLOT_SIGNATURES.items())
def test_plot_signatures_match_pinned_r_formals(function, parameters):
    signature = inspect.signature(function)

    assert list(signature.parameters) == parameters
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


@pytest.mark.parametrize(("function", "defaults"), PLOT_DEFAULTS.items())
def test_plot_defaults_preserve_pinned_r_first_choice_intent(function, defaults):
    signature = inspect.signature(function)

    for name, expected in defaults.items():
        assert signature.parameters[name].default == expected

    if function is validate_flow_plot_lisa_map:
        assert signature.parameters["boundaries"].default is inspect.Parameter.empty


def test_unknown_plot_options_raise_python_signature_error():
    with pytest.raises(TypeError, match="unexpected keyword"):
        cast(Any, validate_flow_plot_metrics)(_metrics(), unsupported_option=True)


def test_metric_sort_palette_and_break_validation_have_visible_effects():
    fig = validate_flow_plot_metrics(
        _metrics(),
        error_measures=("MAE", "RMSE"),
        method_family_col="family",
        palette=("#000000", "#ffffff"),
        relative_error_breaks=(0, 50, 100),
        sort="ascending",
        sort_metric="mae",
    )
    try:
        assert [tick.get_text() for tick in fig.axes[0].get_yticklabels()] == [
            "m2",
            "m1",
        ]
        assert len(fig.axes[0].images[0].cmap.colors) == 2
    finally:
        plt.close(fig)

    with pytest.raises(ValueError, match="relative_error_breaks"):
        validate_flow_plot_metrics(_metrics(), relative_error_breaks=(0, 80, 70))


def test_residual_point_count_transform_and_palette_options_are_honored():
    fig = validate_flow_plot_residuals(
        _residuals(),
        palette={"m1": "#112233", "m2": "#445566"},
        show_points=False,
        show_counts=False,
        violin_alpha=0.25,
        y_transform="identity",
    )
    try:
        ax = fig.axes[0]
        assert ax.get_yscale() == "linear"
        assert not any(isinstance(item, PathCollection) for item in ax.collections)
        assert not any(text.get_text().startswith("n=") for text in ax.texts)
        violin_bodies = [
            item for item in ax.collections if isinstance(item, PolyCollection)
        ]
        assert violin_bodies
        alphas = [body.get_alpha() for body in violin_bodies]
        assert all(alpha is not None for alpha in alphas)
        assert all(
            np.isclose(float(alpha), 0.25) for alpha in alphas if alpha is not None
        )
    finally:
        plt.close(fig)


def test_scatter_facets_limits_and_outline_options_are_honored():
    fig = validate_flow_plot_scatter(
        _residuals(),
        facet_ncol=1,
        difference_limits=(-3, 3),
        point_outline=True,
        point_outline_colour="#123456",
        point_stroke=0.5,
    )
    try:
        main_axes = [axis for axis in fig.axes if axis.get_title()]
        assert len(main_axes) == 2
        collection = main_axes[0].collections[0]
        assert collection.norm.vmin == -3
        assert collection.norm.vmax == 3
        assert collection.get_linewidths()[0] == pytest.approx(0.5)
    finally:
        plt.close(fig)

    with pytest.raises(ValueError, match="difference_limits"):
        validate_flow_plot_scatter(_residuals(), difference_limits=(1, -1))


def test_residual_band_quantiles_orientation_and_labels_are_honored():
    fig = validate_flow_plot_residual_bands(
        _residuals(),
        band_method="quantile",
        quantile_reference="pooled",
        quantile_probs=(0.25, 0.75),
        orientation="vertical",
        label_min_share=0,
        palette=("#111111", "#777777", "#eeeeee"),
    )
    try:
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Share of OD pairs"
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert legend_labels == ["<= P25", "P25 to P75", "> P75"]
        assert any(text.get_text().endswith("%") for text in ax.texts)
    finally:
        plt.close(fig)


def test_pairwise_jsd_mirroring_changes_heatmap_cells():
    data = _distribution().iloc[[0]].copy()
    mirrored = validate_flow_plot_distribution_pairwise(
        data, plot_type="heatmap", mirror_jsd=True
    )
    directional = validate_flow_plot_distribution_pairwise(
        data, plot_type="heatmap", mirror_jsd=False
    )
    try:
        mirrored_values = np.asarray(mirrored.axes[0].images[0].get_array())
        directional_values = np.asarray(directional.axes[0].images[0].get_array())
        assert np.isfinite(mirrored_values).sum() == 4
        assert np.isfinite(directional_values).sum() == 3
    finally:
        plt.close(mirrored)
        plt.close(directional)


def test_structure_band_labels_palette_and_sort_options_are_honored():
    fig = validate_flow_plot_structure(
        _structure(),
        near_zero_band=None,
        show_value_labels=True,
        value_digits=3,
        palette=("#111111", "#999999"),
        sort="ascending",
        sort_metric="moran_i",
    )
    try:
        ax = fig.axes[0]
        assert not ax.patches
        assert {tick.get_text() for tick in ax.get_yticklabels()} == {"m1", "m2"}
        assert any(text.get_text() == "0.200" for text in ax.texts)
    finally:
        plt.close(fig)


def test_metric_and_structure_plots_facet_without_collapsing_comparisons():
    comparisons = ["adjusted_vs_benchmark", "raw_vs_adjusted"]
    metrics = pd.DataFrame(
        {
            "method": ["m1", "m2", "m1", "m2"],
            "comparison": [
                comparisons[0],
                comparisons[0],
                comparisons[1],
                comparisons[1],
            ],
            "mae": [1.0, 2.0, 91.0, 92.0],
        }
    )
    structure = pd.DataFrame(
        {
            "method": ["m1", "m2", "m1", "m2"],
            "comparison": [
                comparisons[0],
                comparisons[0],
                comparisons[1],
                comparisons[1],
            ],
            "pearson_residual_benchmark_flow": [0.11, 0.12, 0.81, 0.82],
            "moran_i": [0.21, 0.22, 0.71, 0.72],
            "pearson_residual_covariate": [0.31, 0.32, 0.61, 0.62],
        }
    )

    metric_fig = validate_flow_plot_metrics(
        metrics,
        error_measures=("mae",),
        comparisons=comparisons,
    )
    structure_fig = validate_flow_plot_structure(
        structure,
        comparisons=comparisons,
    )
    try:
        metric_axes = [axis for axis in metric_fig.axes if axis.images]
        assert [axis.get_title() for axis in metric_axes] == [
            "Adjusted vs benchmark",
            "Raw MPD vs adjusted",
        ]
        assert {text.get_text() for text in metric_axes[0].texts} == {
            "1.000",
            "2.000",
        }
        assert {text.get_text() for text in metric_axes[1].texts} == {
            "91.0",
            "92.0",
        }

        assert [axis.get_title() for axis in structure_fig.axes] == [
            "Adjusted vs benchmark",
            "Raw MPD vs adjusted",
        ]
        first_values = {text.get_text() for text in structure_fig.axes[0].texts}
        second_values = {text.get_text() for text in structure_fig.axes[1].texts}
        assert {"0.11", "0.12", "0.21", "0.22", "0.31", "0.32"} <= first_values
        assert {"0.81", "0.82", "0.71", "0.72", "0.61", "0.62"} <= second_values
    finally:
        plt.close(metric_fig)
        plt.close(structure_fig)


def test_lisa_map_validates_significance_options_before_plotting():
    structure = pd.DataFrame(
        {
            "area": ["A"],
            "method": ["m1"],
            "comparison": ["adjusted_vs_benchmark"],
            "lisa_cluster": ["high-high"],
            "p_value": [0.01],
        }
    )
    boundaries = pd.DataFrame({"area": ["A"], "geometry": [None]})

    with pytest.raises(ValueError, match="p_value_threshold"):
        validate_flow_plot_lisa_map(
            structure, boundaries=boundaries, p_value_threshold=1.5
        )
    with pytest.raises(ValueError, match="outline_boundaries"):
        validate_flow_plot_lisa_map(
            structure, boundaries=boundaries, outline_boundaries=object()
        )


def test_distribution_plot_uses_selected_metric_comparisons_and_method_labels():
    fig = validate_flow_plot_distribution(
        _distribution(),
        metric="kl",
        value="median",
        comparisons="all",
        methods=["m2"],
        method_labels={"m2": "Second method"},
    )
    try:
        np.testing.assert_allclose(fig.axes[0].images[0].get_array(), [[0.39, 0.79]])
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
            "Second method"
        ]
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_metrics_plot_accepts_nested_single_comparison_results():
    results = {
        "A": {
            "method": None,
            "comparison": "adjusted_vs_benchmark",
            "mae": 1.0,
            "rmse": 2.0,
            "mape": 0.25,
        },
        "B": {
            "method": None,
            "comparison": "adjusted_vs_benchmark",
            "mae": 3.0,
            "rmse": 4.0,
            "mape": 0.5,
        },
    }
    fig = validate_flow_plot_metrics(results, error_measures=["mae"])
    try:
        np.testing.assert_allclose(fig.axes[0].images[0].get_array(), [[0.0], [100.0]])
        assert [text.get_text() for text in fig.axes[0].get_yticklabels()] == ["A", "B"]
        fig.canvas.draw()
    finally:
        plt.close(fig)
