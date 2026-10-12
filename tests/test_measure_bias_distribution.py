import numpy as np
import pandas as pd
import pytest

from debiaspy import measure_bias_distribution


def test_distribution_is_zero_for_proportional_shares() -> None:
    coverage = pd.DataFrame(
        {
            "origin": ["A", "B", "C"],
            "population": [100, 200, 300],
            "user_count": [10, 20, 30],
        }
    )

    result = measure_bias_distribution(coverage)

    assert set(result) == {"summary", "area_level"}
    assert result["summary"].loc[0, "n_areas"] == 3
    assert result["summary"].loc[0, "kl_population_user"] == pytest.approx(0, abs=1e-12)
    assert result["summary"].loc[0, "jsd_population_user"] == pytest.approx(
        0, abs=1e-12
    )
    np.testing.assert_allclose(
        result["area_level"]["population_share"],
        result["area_level"]["user_share"],
        atol=1e-8,
    )


def test_distribution_matches_manual_calculation_and_custom_columns() -> None:
    coverage = pd.DataFrame(
        {"area_id": ["A", "B"], "pop": [300, 100], "users": [25, 75]}
    )
    original = coverage.copy(deep=True)

    result = measure_bias_distribution(
        coverage,
        area_col="area_id",
        population_col="pop",
        user_count_col="users",
    )

    population_share = np.array([0.75, 0.25])
    user_share = np.array([0.25, 0.75])
    midpoint = 0.5 * (population_share + user_share)
    expected_kl = np.sum(population_share * np.log(population_share / user_share))
    expected_jsd = 0.5 * np.sum(
        population_share * np.log(population_share / midpoint)
    ) + 0.5 * np.sum(user_share * np.log(user_share / midpoint))

    assert result["summary"].loc[0, "kl_population_user"] == pytest.approx(
        expected_kl, rel=1e-8
    )
    assert result["summary"].loc[0, "jsd_population_user"] == pytest.approx(
        expected_jsd, rel=1e-8
    )
    np.testing.assert_allclose(
        result["area_level"]["share_difference_user_minus_population"],
        user_share - population_share,
        atol=1e-8,
    )
    pd.testing.assert_frame_equal(coverage, original)


def test_distribution_handles_zero_users_and_can_omit_area_level() -> None:
    coverage = pd.DataFrame(
        {
            "origin": ["A", "B"],
            "population": [100, 50],
            "user_count": [0, 0],
        }
    )

    result = measure_bias_distribution(coverage, return_area_level=False)

    assert set(result) == {"summary"}
    assert result["summary"].loc[0, "total_user_count"] == 0
    assert np.isfinite(result["summary"].loc[0, "kl_population_user"])
    assert np.isfinite(result["summary"].loc[0, "jsd_population_user"])


@pytest.mark.parametrize("epsilon", [0, -1, np.inf, np.nan, True, "small"])
def test_distribution_rejects_invalid_epsilon(epsilon: object) -> None:
    coverage = pd.DataFrame({"origin": ["A"], "population": [100], "user_count": [10]})

    with pytest.raises(ValueError, match="positive finite number"):
        measure_bias_distribution(coverage, epsilon=epsilon)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("coverage", "message"),
    [
        (pd.DataFrame({"origin": ["A"], "population": [100]}), "must contain"),
        (
            pd.DataFrame(
                {
                    "origin": ["A", "A"],
                    "population": [100, 50],
                    "user_count": [10, 5],
                }
            ),
            "one row per area",
        ),
        (
            pd.DataFrame({"origin": [None], "population": [100], "user_count": [10]}),
            "areas must be identifiable",
        ),
        (
            pd.DataFrame({"origin": ["A"], "population": [np.nan], "user_count": [10]}),
            "population_col.*NA",
        ),
        (
            pd.DataFrame(
                {"origin": ["A"], "population": [100], "user_count": [np.nan]}
            ),
            "user_count_col.*NA",
        ),
        (
            pd.DataFrame({"origin": ["A"], "population": [-1], "user_count": [1]}),
            "population_col.*positive",
        ),
        (
            pd.DataFrame({"origin": ["A"], "population": [1], "user_count": [-1]}),
            "user_count_col.*non-negative",
        ),
    ],
)
def test_distribution_validates_inputs(coverage: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        measure_bias_distribution(coverage)


def test_distribution_requires_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        measure_bias_distribution(  # type: ignore[arg-type]
            {"origin": ["A"], "population": [100], "user_count": [10]}
        )
