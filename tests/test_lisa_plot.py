"""Exercise real LISA figures and their geographic/colour contracts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest


@pytest.fixture
def boundaries():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"area": list("ABCDEZ")},
        geometry=[box(i, 0, i + 1, 1) for i in range(6)],
        crs="EPSG:27700",
    )
