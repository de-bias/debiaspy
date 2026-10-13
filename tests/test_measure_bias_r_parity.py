"""Exercise coverage arithmetic also used by debiasR's measure_bias()."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from debiaspy import measure_bias

CASES = [
    pytest.param(100.0, 0.0, 0.0, id="zero-users"),
    pytest.param(100.0, np.nan, np.nan, id="missing-users"),
    pytest.param(100.0, np.inf, np.inf, id="infinite-users"),
    pytest.param(np.inf, 10.0, 0.0, id="infinite-population"),
    pytest.param(np.inf, np.inf, np.nan, id="infinite-ratio"),
]


@pytest.mark.parametrize("population, user_count, expected_score", CASES)
def test_measure_bias_preserves_boundary_arithmetic(
    population: float, user_count: float, expected_score: float
) -> None:
    coverage = pd.DataFrame(
        {"area": ["A"], "population": [population], "user_count": [user_count]}
    )
    original = coverage.copy(deep=True)

    if np.isposinf(expected_score):
        with pytest.warns(UserWarning, match="coverage_score > 1"):
            result = measure_bias(coverage)
    else:
        result = measure_bias(coverage)

    np.testing.assert_allclose(result["coverage_score"], [expected_score])
    np.testing.assert_allclose(result["coverage_bias"], [1 - expected_score])
    pd.testing.assert_series_equal(
        result["bias"], result["coverage_bias"], check_names=False
    )
    pd.testing.assert_frame_equal(coverage, original)
    pd.testing.assert_frame_equal(result[original.columns], original)


@pytest.fixture(scope="module")
def r_measurement_results(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """Run the optional R comparison when a reference checkout is supplied."""
    configured_root = os.environ.get("DEBIASR_ROOT")
    if configured_root is None or shutil.which("Rscript") is None:
        pytest.skip("Set DEBIASR_ROOT and install Rscript for the live R comparison")
    reference = Path(configured_root).expanduser().resolve()
    source = reference / "R" / "measure_bias.R"
    if not source.is_file():
        pytest.fail(f"R measurement source not found: {source}")

    output = tmp_path_factory.mktemp("r-measurement")
    script = output / "measure.R"
    script.write_text(
        """args <- commandArgs(trailingOnly = TRUE)
source(args[[1]])
coverage <- data.frame(
  area = LETTERS[1:5],
  population = c(100, 100, 100, Inf, Inf),
  user_count = c(0, NA_real_, Inf, 10, Inf)
)
result <- suppressWarnings(measure_bias(coverage))
utils::write.csv(result, args[[2]], row.names = FALSE, na = "NA")
""",
        encoding="utf-8",
    )
    target = output / "measurement.csv"
    subprocess.run(
        ["Rscript", str(script), str(source), str(target)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return pd.read_csv(target)


def test_measure_bias_matches_live_r_boundary_values(
    r_measurement_results: pd.DataFrame,
) -> None:
    coverage = pd.DataFrame(
        {
            "area": list("ABCDE"),
            "population": [100.0, 100.0, 100.0, np.inf, np.inf],
            "user_count": [0.0, np.nan, np.inf, 10.0, np.inf],
        }
    )
    with pytest.warns(UserWarning, match="coverage_score > 1"):
        result = measure_bias(coverage)

    pd.testing.assert_frame_equal(
        result, r_measurement_results, check_dtype=False, atol=1e-12, rtol=1e-12
    )
