"""Exercise real LISA figures and their geographic/colour contracts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from debiaspy import validate_flow_plot_lisa_map, validate_flow_residual_structure


@pytest.fixture
def boundaries():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"area": list("ABCDEZ")},
        geometry=[box(i, 0, i + 1, 1) for i in range(6)],
        crs="EPSG:27700",
    )


@pytest.fixture
def result():
    adjusted = pd.DataFrame(
        {
            "origin": list("ABCDE"),
            "destination": ["X"] * 5,
            "flow": [10] * 5,
            "flow_adj": [12, 12, 8, 8, 10],
        }
    )
    return validate_flow_residual_structure(
        adjusted,
        adjusted[["origin", "destination", "flow"]],
        method_name="one",
        area_neighbors=pd.DataFrame({"area": list("ABCD"), "neighbor": list("BADC")}),
        local_moran=True,
        local_moran_nsim=19,
        local_moran_alpha=1,
        local_moran_p_adjust="none",
        local_moran_seed=123,
    )


def test_lisa_pipeline_draws_expected_clusters_isolates_and_unmatched_polygons(
    boundaries, result, tmp_path
):
    palette = {"low-low": "blue", "high-high": "red", "no neighbours": "grey"}
    fig = validate_flow_plot_lisa_map(
        result,
        boundaries,
        p_value_threshold=None,
        palette=palette,
        missing_fill="yellow",
    )
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        assert fig.axes[0].get_title() == "one"
        colors = fig.axes[0].collections[0].get_facecolors()
        np.testing.assert_allclose(
            colors,
            [to_rgba(c) for c in ["blue", "blue", "red", "red", "grey", "yellow"]],
        )
        assert {t.get_text() for t in fig.axes[0].get_legend().get_texts()} == set(
            palette
        )
        fig.canvas.draw()
        fig.savefig(tmp_path / "lisa.png")
        assert (tmp_path / "lisa.png").stat().st_size > 1000
    finally:
        plt.close(fig)


def test_lisa_significance_filter_preserves_isolates_and_missing_fill(
    boundaries, result
):
    result["map_data"] = result["map_data"].assign(
        p_value=[0.01, 0.2, 0.01, 0.2, np.nan]
    )
    fig = validate_flow_plot_lisa_map(
        result,
        boundaries,
        palette={
            "low-low": "blue",
            "high-high": "red",
            "not significant": "white",
            "no neighbours": "grey",
        },
        missing_fill="yellow",
    )
    try:
        np.testing.assert_allclose(
            fig.axes[0].collections[0].get_facecolors(),
            [to_rgba(c) for c in ["blue", "white", "red", "white", "grey", "yellow"]],
        )
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_lisa_method_comparison_facets_subset_labels_and_outline(boundaries, result):
    first = result["map_data"]
    second = pd.concat(
        [
            first.assign(method="two"),
            first.assign(method="two", comparison="raw_vs_benchmark"),
        ],
        ignore_index=True,
    )
    first = pd.concat(
        [first, first.assign(comparison="raw_vs_benchmark")], ignore_index=True
    )
    combined = pd.concat([first, second], ignore_index=True)
    fig = validate_flow_plot_lisa_map(
        combined,
        boundaries,
        comparisons=["adjusted_vs_benchmark", "raw_vs_benchmark"],
        methods=["two"],
        method_labels={"two": "Second method"},
        p_value_threshold=None,
        outline_boundaries=boundaries,
    )
    try:
        assert len(fig.axes) == 2
        assert all("Second method" in ax.get_title() for ax in fig.axes)
        assert all(len(ax.collections) == 2 for ax in fig.axes)
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_lisa_method_order_uses_significant_area_share(boundaries, result):
    first = result["map_data"].assign(method="low", lisa_cluster="not significant")
    second = result["map_data"].assign(method="high")
    fig = validate_flow_plot_lisa_map(
        pd.concat([first, second]),
        boundaries,
        sort="descending",
        p_value_threshold=None,
    )
    try:
        assert [ax.get_title() for ax in fig.axes] == ["high", "low"]
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "change, message",
    [
        ("empty", "comparisons"),
        ("no_overlap", "share any area"),
        ("duplicate", "one row"),
    ],
)
def test_lisa_invalid_map_contracts_raise_clear_errors(
    boundaries, result, change, message
):
    data = result["map_data"]
    if change == "empty":
        data = data.iloc[:0]
    elif change == "no_overlap":
        data = data.assign(area=list("QRSTU"))
    elif change == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    else:
        data = data.drop(columns="comparison")
    with pytest.raises(ValueError, match=message):
        validate_flow_plot_lisa_map(data, boundaries, p_value_threshold=None)


def test_lisa_missing_comparison_uses_the_r_default(boundaries, result):
    fig = validate_flow_plot_lisa_map(
        result["map_data"].drop(columns="comparison"),
        boundaries,
        p_value_threshold=None,
    )
    try:
        assert len(fig.axes) == 1
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_lisa_cluster_colours_are_consistent_across_method_panels(boundaries, result):
    first = result["map_data"].assign(method="first", lisa_cluster="low-low")
    second = result["map_data"].assign(method="second", lisa_cluster="high-high")
    third = result["map_data"].assign(method="third", lisa_cluster="low-low")
    fig = validate_flow_plot_lisa_map(
        pd.concat([first, second, third]),
        boundaries,
        p_value_threshold=None,
    )
    try:
        colors = [ax.collections[0].get_facecolors()[0] for ax in fig.axes]
        np.testing.assert_allclose(colors[0], colors[2])
        assert not np.array_equal(colors[0], colors[1])
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_lisa_outline_reprojects_to_boundary_coordinates(boundaries, result):
    outline = boundaries.to_crs("EPSG:4326")
    fig = validate_flow_plot_lisa_map(
        result,
        boundaries,
        outline_boundaries=outline,
        p_value_threshold=None,
    )
    try:
        projected = outline.to_crs(boundaries.crs)
        paths = fig.axes[0].collections[1].get_paths()
        vertices = np.concatenate([path.vertices for path in paths])
        np.testing.assert_allclose(
            [
                vertices[:, 0].min(),
                vertices[:, 1].min(),
                vertices[:, 0].max(),
                vertices[:, 1].max(),
            ],
            projected.total_bounds,
            atol=1e-8,
        )
        assert outline.crs.to_epsg() == 4326
        fig.canvas.draw()
    finally:
        plt.close(fig)
