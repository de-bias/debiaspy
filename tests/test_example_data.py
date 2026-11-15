"""Example-loader behaviour checked on small, explicit empirical-shaped tables."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from debiaspy import debiaspy_example_data


@pytest.fixture
def assets(monkeypatch):
    a, b, c = "E06000001", "E06000002", "E06000003"
    mpd = pd.DataFrame(
        {"origin": [a, a, b, c], "destination": [a, b, a, c], "flow": [10, 4, 5, 1]}
    )
    benchmark = pd.DataFrame(
        {"origin": [a, a, b, c], "destination": [a, b, a, c], "flow": [100, 25, 50, 0]}
    )
    tables = {"lad_OD_travel2work": mpd, "census_lad_OD_travel2work": benchmark}
    module = SimpleNamespace(
        list_datasets=lambda: tuple(tables),
        load_dataset=lambda name: tables[name].copy(deep=True),
    )
    monkeypatch.setitem(sys.modules, "test_example_assets", module)
    return tables


def load(**kwargs):
    return debiaspy_example_data(data_package="test_example_assets", **kwargs)


def test_core_outputs_and_r_reference_values(assets):
    result = load(n_areas=None)
    assert list(result) == [
        "mpd_od",
        "benchmark_od",
        "coverage",
        "active_users",
        "population",
        "covariates",
        "distance",
        "od_audit",
        "metadata",
        "lad_OD_travel2work",
        "census_lad_OD_travel2work",
    ]
    # Same explicit inputs and coverage values as the R example-loader test.
    coverage = result["coverage"].set_index("origin")
    assert coverage.loc["E06000001", "population"] == 125
    assert coverage.loc["E06000001", "user_count"] == 14
    assert coverage.loc["E06000002", "population"] == 50
    assert len(coverage) == 2
    assert len(result["mpd_od"]) == 3
    assert result["distance"].empty
    assert result["metadata"].loc[0, "distance_source"] == "not_available"
    assert result["metadata"].loc[0, "covariate_source"] == "derived_from_od_flows"
    assert result["covariates"]["income_norm"].tolist() == [1.0, 0.0]


@pytest.mark.parametrize("self_flows,rows", [(True, 4), (False, 2)])
def test_complete_grid_and_audit(assets, self_flows, rows):
    result = load(complete_grid=True, include_self_flows=self_flows)
    assert len(result["mpd_od"]) == rows
    assert len(result["benchmark_od"]) == rows
    assert result["od_audit"].loc[0, "strict_square_support"]
    assert result["od_audit"].loc[0, "mpd_balance_diff"] == 0
    if self_flows:
        assert result["mpd_od"]["mpd_zero_filled"].sum() == 1
        assert (
            result["mpd_od"].loc[result["mpd_od"].mpd_zero_filled, "flow"].eq(0).all()
        )


def test_nonzero_fill_and_coverage_precede_grid_completion(assets):
    result = load(complete_grid=True, fill_missing_flow=2)
    assert result["mpd_od"].flow.sum() == 21
    assert result["coverage"].user_count.sum() == 19


def test_optional_covariates_and_bng_centroids(assets):
    areas = ["E06000001", "E06000002"]
    assets["lad_covariates"] = pd.DataFrame({"area": areas, "income": [5, 8]})
    assets["lad_centroids"] = pd.DataFrame(
        {"area": areas, "easting": [0, 3000], "northing": [0, 4000]}
    )
    result = load()
    assert result["covariates"].income.tolist() == [5, 8]
    assert (
        result["metadata"].loc[0, "covariate_source"]
        == "test_example_assets::lad_covariates"
    )
    assert result["distance"].distance_km.tolist() == [0, 5, 5, 0]


def test_csv_paths_work_without_companion(assets, tmp_path):
    mpd, census = tmp_path / "mpd.csv.gz", tmp_path / "census.csv"
    assets["lad_OD_travel2work"].to_csv(mpd, index=False)
    assets["census_lad_OD_travel2work"].to_csv(census, index=False)
    result = debiaspy_example_data(
        mpd_path=mpd, census_path=census, data_package="absent_example_package"
    )
    assert len(result["coverage"]) == 2
    assert result["distance"].empty


def test_missing_companion_has_installation_guidance():
    with pytest.raises(ImportError, match="Install.*absent_example_package"):
        debiaspy_example_data(data_package="absent_example_package")


@pytest.mark.parametrize("n", [None, np.inf, 2, 25])
def test_area_selection_is_deterministic(assets, n):
    first = load(n_areas=n)
    second = load(n_areas=n)
    pd.testing.assert_frame_equal(first["mpd_od"], second["mpd_od"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_areas": 1},
        {"n_areas": True},
        {"n_areas": 2.5},
        {"n_areas": np.nan},
        {"fill_missing_flow": -1},
        {"fill_missing_flow": np.inf},
        {"geography": "invalid"},
        {"complete_grid": "yes"},
    ],
)
def test_invalid_controls(assets, kwargs):
    with pytest.raises(ValueError):
        load(**kwargs)


def test_no_overlapping_areas(assets):
    assets["census_lad_OD_travel2work"] = pd.DataFrame(
        {"origin": ["E06000009"], "destination": ["E06000009"], "flow": [10]}
    )
    with pytest.raises(ValueError, match="No overlapping"):
        load()


def test_msoa_names_and_area_filter(assets):
    for old, new in (
        ("lad_OD_travel2work", "msoa_OD_travel2work"),
        ("census_lad_OD_travel2work", "census_msoa_OD_travel2work"),
    ):
        data = assets[old].copy()
        data["origin"] = data.origin.str.replace("E060", "E020")
        data["destination"] = data.destination.str.replace("E060", "E020")
        assets[new] = data
    result = load(geography="msoa", complete_grid=True)
    assert "msoa_OD_travel2work" in result
    assert result["metadata"].loc[0, "geography"] == "msoa"
    assert result["mpd_od"].origin.str.startswith("E020").all()


def test_normalisation_duplicates_census_indicator_and_invalid_rows(assets):
    a, b = "E06000001", "E06000002"
    assets["lad_OD_travel2work"] = pd.DataFrame(
        {
            "LAD22CD_home": [a, a, b, "invalid", a, a],
            "LAD22CD_work": [b, b, a, b, a, b],
            "count": [2, 3, 4, 99, -1, np.nan],
        }
    )
    assets["census_lad_OD_travel2work"] = pd.DataFrame(
        {
            "Lower tier local authorities code": [a, b, a],
            "LTLA of workplace code": [b, a, a],
            "Place of work indicator (4 categories) code": [3, 3, 1],
            "Count": [50, 40, 999],
        }
    )
    before = {name: table.copy(deep=True) for name, table in assets.items()}
    result = load()
    assert result["mpd_od"].flow.tolist() == [5, 4]
    assert result["benchmark_od"].flow.tolist() == [50, 40]
    for name, table in assets.items():
        pd.testing.assert_frame_equal(table, before[name])


def test_area_selection_uses_flow_scores_then_area_code(assets):
    codes = ["E06000001", "E06000002", "E06000003"]
    for name, counts in (
        ("lad_OD_travel2work", [2, 3, 3]),
        ("census_lad_OD_travel2work", [20, 30, 30]),
    ):
        assets[name] = pd.DataFrame(
            {"origin": codes, "destination": codes, "flow": counts}
        )
    assert load(n_areas=2)["coverage"].origin.tolist() == codes[1:]


def test_partial_optional_assets_fall_back(assets):
    assets["lad_covariates"] = pd.DataFrame({"area": ["E06000001"], "x": [1]})
    assets["lad_centroids"] = pd.DataFrame(
        {"area": ["E06000001"], "longitude": [0], "latitude": [50]}
    )
    result = load()
    assert result["metadata"].loc[0, "covariate_source"] == "derived_from_od_flows"
    assert result["distance"].empty


def test_longitude_latitude_distance(assets):
    assets["lad_centroids"] = pd.DataFrame(
        {"area": ["E06000001", "E06000002"], "longitude": [0, 1], "latitude": [0, 0]}
    )
    result = load(include_self_flows=False)
    np.testing.assert_allclose(result["distance"].distance_km, [111.1950802335] * 2)


def test_missing_dataset_and_required_columns(assets):
    del assets["census_lad_OD_travel2work"]
    with pytest.raises(ValueError, match="requires.*census_lad"):
        load()
    assets["lad_OD_travel2work"] = assets["lad_OD_travel2work"].drop(columns="flow")
    with pytest.raises(ValueError, match="Could not find"):
        load()


@pytest.mark.parametrize(
    "name",
    [
        "lad_OD_travel2work",
        "census_lad_OD_travel2work",
        "lad_covariates",
        "lad_centroids",
    ],
)
def test_companion_datasets_must_be_dataframes(assets, monkeypatch, name):
    assets[name] = None
    monkeypatch.setattr(
        sys.modules["test_example_assets"], "load_dataset", lambda key: assets[key]
    )
    with pytest.raises(TypeError, match=f"{name}.*DataFrame"):
        load()


@pytest.mark.parametrize(
    "name",
    [
        "lad_OD_travel2work",
        "census_lad_OD_travel2work",
        "lad_covariates",
        "lad_centroids",
    ],
)
def test_companion_datasets_reject_duplicate_columns(assets, name):
    if name not in assets:
        assets[name] = pd.DataFrame({"area": ["E06000001"]})
    table = assets[name]
    assets[name] = pd.concat([table, table.iloc[:, :1]], axis=1)
    with pytest.raises(ValueError, match=f"{name}.*unique column names"):
        load()
