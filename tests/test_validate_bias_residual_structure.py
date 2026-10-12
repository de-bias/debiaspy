import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from debiaspy import validate_bias_residual_structure


@pytest.fixture
def coverage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "area_id": ["A", "B", "C", "D"],
            "population": [100, 100, 100, 100],
            "user_count": [5, 15, 25, 35],
        }
    )


def test_residual_structure_returns_deterministic_diagnostics(
    coverage: pd.DataFrame,
) -> None:
    benchmark = pd.DataFrame(
        {
            "origin": ["A", "B", "C", "D"],
            "destination": ["D", "C", "B", "A"],
            "flow": [10, 20, 30, 40],
        }
    )
    neighbors = pd.DataFrame(
        {"area": ["A", "B", "C", "D"], "neighbor": ["B", "A", "D", "C"]}
    )
    covariates = pd.DataFrame({"area": ["A", "B", "C", "D"], "cov": [4, 3, 2, 1]})

    result = validate_bias_residual_structure(
        coverage,
        coverage_area_col="area_id",
        benchmark_od_df=benchmark,
        area_neighbors=neighbors,
        covariate_df=covariates,
        covariate_col="cov",
    )

    assert {
        "summary",
        "residual_definitions",
        "moran_i",
        "benchmark_flow_correlation",
        "covariate_correlation",
        "population_lm",
        "area_level",
        "map_data",
        "benchmark_flow_data",
        "covariate_data",
    } == set(result)
    assert result["summary"].loc[0, "residual_type"] == "coverage_score"
    assert result["summary"].loc[0, "global_coverage_score"] == pytest.approx(0.2)
    np.testing.assert_allclose(result["area_level"]["expected_user_count"], 20)
    np.testing.assert_allclose(
        result["area_level"]["user_count_residual"], [-15, -5, 5, 15]
    )
    np.testing.assert_allclose(
        result["area_level"]["coverage_score_residual"],
        [-0.15, -0.05, 0.05, 0.15],
        atol=1e-12,
    )
    assert result["moran_i"].loc[0, "moran_i"] == pytest.approx(0.6)

    correlations = result["benchmark_flow_correlation"].set_index("benchmark_flow_role")
    assert correlations.loc["origin", "pearson_r"] == pytest.approx(1)
    assert correlations.loc["destination", "pearson_r"] == pytest.approx(-1)
    assert result["covariate_correlation"].loc[0, "pearson_r"] == pytest.approx(-1)


@pytest.mark.parametrize(
    ("residual_type", "column"),
    [
        ("coverage_score", "coverage_score_residual"),
        ("user_count", "user_count_residual"),
        ("standardized_user_count", "standardized_user_count_residual"),
        ("population_lm", "population_lm_residual"),
    ],
)
def test_residual_structure_selects_each_residual(
    coverage: pd.DataFrame, residual_type: str, column: str
) -> None:
    result = validate_bias_residual_structure(
        coverage,
        coverage_area_col="area_id",
        residual_type=residual_type,
    )

    assert result["summary"].loc[0, "selected_residual_col"] == column
    np.testing.assert_allclose(
        result["area_level"]["selected_residual"],
        result["area_level"][column],
    )


def test_standardized_residuals_match_debiasr(coverage: pd.DataFrame) -> None:
    result = validate_bias_residual_structure(
        coverage,
        coverage_area_col="area_id",
        residual_type="standardized_user_count",
    )

    np.testing.assert_allclose(
        result["area_level"]["selected_residual"],
        np.array([-15, -5, 5, 15]) / np.sqrt(20),
        atol=1e-12,
    )


def test_population_only_model_matches_expected_fit() -> None:
    coverage = pd.DataFrame(
        {
            "area_id": ["A", "B", "C", "D"],
            "population": [100, 200, 300, 400],
            "user_count": [15, 18, 40, 42],
        }
    )

    result = validate_bias_residual_structure(
        coverage,
        coverage_area_col="area_id",
        residual_type="population_lm",
    )

    assert result["population_lm"].loc[0, "model"] == "user_count ~ population"
    assert result["population_lm"].loc[0, "n"] == 4
    assert result["population_lm"].loc[0, "intercept"] == pytest.approx(3.0)
    assert result["population_lm"].loc[0, "population_coefficient"] == pytest.approx(
        0.103
    )
    np.testing.assert_allclose(
        result["area_level"]["population_lm_expected_user_count"],
        [13.3, 23.6, 33.9, 44.2],
        atol=1e-12,
    )


def test_residual_structure_joins_geometry_and_does_not_modify_input(
    coverage: pd.DataFrame,
) -> None:
    original = coverage.copy(deep=True)
    coordinates = pd.DataFrame(
        {"code": ["A", "B", "C", "D"], "x": [0, 1, 0, 1], "y": [1, 1, 0, 0]}
    )

    result = validate_bias_residual_structure(
        coverage,
        coverage_area_col="area_id",
        geometry_df=coordinates,
        geometry_area_col="code",
    )

    assert result["map_data"]["x"].tolist() == [0, 1, 0, 1]
    pd.testing.assert_frame_equal(coverage, original)


def test_residual_structure_returns_matplotlib_diagnostics(
    coverage: pd.DataFrame,
) -> None:
    benchmark = pd.DataFrame(
        {
            "origin": ["A", "B", "C", "D"],
            "destination": ["D", "C", "B", "A"],
            "flow": [10, 20, 30, 40],
        }
    )
    coordinates = pd.DataFrame(
        {"area": ["A", "B", "C", "D"], "x": [0, 1, 0, 1], "y": [1, 1, 0, 0]}
    )
    covariates = pd.DataFrame({"area": ["A", "B", "C", "D"], "income": [1, 4, 2, 3]})

    result = validate_bias_residual_structure(
        coverage,
        coverage_area_col="area_id",
        benchmark_od_df=benchmark,
        covariate_df=covariates,
        covariate_col="income",
        geometry_df=coordinates,
        x_col="x",
        y_col="y",
        make_plots=True,
    )

    assert set(result["plots"]) == {
        "bias_residual_distribution",
        "bias_residual_vs_benchmark_flow",
        "bias_residual_vs_covariate",
        "bias_residual_map",
    }
    assert all(isinstance(figure, Figure) for figure in result["plots"].values())
    for figure in result["plots"].values():
        plt.close(figure)


def test_residual_structure_supports_weighted_neighbors(
    coverage: pd.DataFrame,
) -> None:
    neighbors = pd.DataFrame(
        {
            "from": ["A", "B", "C", "D"],
            "to": ["B", "A", "D", "C"],
            "w": [2.0, 2.0, 0.0, -1.0],
        }
    )

    result = validate_bias_residual_structure(
        coverage,
        coverage_area_col="area_id",
        area_neighbors=neighbors,
        area_col="from",
        neighbor_col="to",
        weight_col="w",
    )

    assert result["moran_i"].loc[0, "n_links_used"] == 2
    assert result["moran_i"].loc[0, "weight_sum"] == pytest.approx(4)


def test_residual_structure_handles_missing_optional_numeric_values(
    coverage: pd.DataFrame,
) -> None:
    benchmark = pd.DataFrame(
        {
            "origin": ["A", "B", "C", "D"],
            "destination": ["D", "C", "B", "A"],
            "flow": pd.Series([10, pd.NA, 30, 40], dtype="Float64"),
        }
    )
    covariates = pd.DataFrame(
        {
            "area": ["A", "B", "C", "D"],
            "cov": pd.Series([4, pd.NA, 2, 1], dtype="Float64"),
        }
    )

    result = validate_bias_residual_structure(
        coverage,
        coverage_area_col="area_id",
        benchmark_od_df=benchmark,
        covariate_df=covariates,
        covariate_col="cov",
    )

    assert result["covariate_correlation"].loc[0, "n"] == 3
    assert np.isfinite(result["covariate_correlation"].loc[0, "pearson_r"])
    assert result["benchmark_flow_correlation"]["pearson_r"].notna().all()


def test_residual_structure_supports_custom_columns_and_one_flow_role() -> None:
    coverage = pd.DataFrame(
        {"zone": ["A", "B", "C"], "pop": [100, 200, 300], "users": [5, 20, 45]}
    )
    benchmark = pd.DataFrame(
        {"from": ["A", "B", "C"], "to": ["C", "A", "B"], "count": [3, 2, 1]}
    )

    result = validate_bias_residual_structure(
        coverage,
        coverage_area_col="zone",
        population_col="pop",
        user_count_col="users",
        benchmark_od_df=benchmark,
        origin_col="from",
        destination_col="to",
        flow_col_bench="count",
        benchmark_flow_roles="origin",
    )

    assert result["area_level"]["area"].tolist() == ["A", "B", "C"]
    assert result["benchmark_flow_correlation"]["benchmark_flow_role"].tolist() == [
        "origin"
    ]
    assert "benchmark_origin_flow_total" in result["area_level"]
    assert "benchmark_destination_flow_total" not in result["area_level"]


def test_residual_plot_requires_both_coordinate_columns(
    coverage: pd.DataFrame,
) -> None:
    coordinates = pd.DataFrame(
        {"area": ["A", "B", "C", "D"], "x": [0, 1, 0, 1], "y": [1, 1, 0, 0]}
    )

    with pytest.raises(ValueError, match="Both `x_col` and `y_col`"):
        validate_bias_residual_structure(
            coverage,
            coverage_area_col="area_id",
            geometry_df=coordinates,
            x_col="x",
            make_plots=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"residual_type": "unknown"}, "residual_type"),
        ({"covariate_col": "income"}, "covariate_df.*required"),
        ({"benchmark_flow_roles": "unknown"}, "benchmark_flow_roles"),
        (
            {"benchmark_od_df": pd.DataFrame({"origin": ["A"], "flow": [1]})},
            "benchmark_od_df.*contain",
        ),
        (
            {"geometry_df": pd.DataFrame({"wrong": ["A"]})},
            "geometry_area_col",
        ),
    ],
)
def test_residual_structure_validates_optional_inputs(
    coverage: pd.DataFrame, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_bias_residual_structure(
            coverage, coverage_area_col="area_id", **kwargs
        )


def test_residual_structure_rejects_duplicate_areas() -> None:
    coverage = pd.DataFrame(
        {"area_id": ["A", "A"], "population": [100, 100], "user_count": [5, 15]}
    )

    with pytest.raises(ValueError, match="one row per area"):
        validate_bias_residual_structure(coverage, coverage_area_col="area_id")


def test_residual_structure_requires_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate_bias_residual_structure(  # type: ignore[arg-type]
            {"origin": ["A"], "population": [100], "user_count": [5]}
        )
