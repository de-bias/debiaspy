import pandas as pd
import pytest

from debiaspy import measure_bias


def test_measure_bias_computes_coverage_bias_and_preserves_columns() -> None:
    coverage = pd.DataFrame(
        {
            "area": ["A", "B", "C"],
            "population": [100.0, 200.0, 400.0],
            "user_count": [50.0, 25.0, 100.0],
        }
    )

    result = measure_bias(coverage)

    assert list(result.columns) == [
        "area",
        "population",
        "user_count",
        "coverage_score",
        "coverage_bias",
        "bias",
    ]
    assert len(result) == len(coverage)
    assert result["coverage_score"].tolist() == [0.5, 0.125, 0.25]
    assert result["coverage_bias"].tolist() == [0.5, 0.875, 0.75]
    assert result["bias"].equals(result["coverage_bias"])


def test_measure_bias_returns_copy_without_modifying_input() -> None:
    coverage = pd.DataFrame(
        {
            "population": [100.0],
            "user_count": [50.0],
        }
    )

    result = measure_bias(coverage)

    assert "coverage_score" not in coverage.columns
    assert result is not coverage


def test_measure_bias_warns_when_coverage_score_exceeds_one() -> None:
    coverage = pd.DataFrame(
        {
            "population": [100.0],
            "user_count": [110.0],
        }
    )

    with pytest.warns(UserWarning, match="coverage_score > 1"):
        result = measure_bias(coverage)

    assert result.loc[0, "coverage_score"] > 1
    assert result.loc[0, "coverage_bias"] < 0


def test_measure_bias_errors_on_missing_required_columns() -> None:
    missing_user = pd.DataFrame({"population": [100.0]})
    missing_population = pd.DataFrame({"user_count": [50.0]})

    with pytest.raises(ValueError, match="Missing required columns"):
        measure_bias(missing_user)

    with pytest.raises(ValueError, match="Missing required columns"):
        measure_bias(missing_population)


def test_measure_bias_errors_on_invalid_values() -> None:
    population_na = pd.DataFrame({"population": [None], "user_count": [1.0]})
    population_zero = pd.DataFrame({"population": [0.0], "user_count": [1.0]})
    user_negative = pd.DataFrame({"population": [100.0], "user_count": [-1.0]})

    with pytest.raises(ValueError, match="population contains NA values"):
        measure_bias(population_na)

    with pytest.raises(ValueError, match="population must be positive"):
        measure_bias(population_zero)

    with pytest.raises(ValueError, match="user_count must be non-negative"):
        measure_bias(user_negative)


def test_measure_bias_requires_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        measure_bias({"population": [100.0], "user_count": [50.0]})
