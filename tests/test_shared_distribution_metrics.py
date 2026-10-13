"""Coverage and OD distributions share calculations without changing support."""

import numpy as np
import pandas as pd
import pytest

from debiaspy import measure_bias_distribution, validate_flow_distribution


@pytest.mark.parametrize("users", [[20, 0, 20], [0, 0, 0], [25, 50, 25]])
def test_coverage_and_flow_share_the_same_kl_jsd_and_smoothed_shares(users):
    coverage = pd.DataFrame(
        {"origin": list("ABC"), "population": [25, 50, 25], "user_count": users}
    )
    measured = measure_bias_distribution(coverage)
    adjusted = pd.DataFrame(
        {"origin": ["source"] * 3, "destination": list("ABC"), "flow_adj": users}
    )
    benchmark = adjusted[["origin", "destination"]].assign(
        flow=coverage.population.to_numpy()
    )
    validated = validate_flow_distribution(adjusted, benchmark, return_od_level=True)
    for measured_column, flow_column in [
        ("population_share", "reference_share"),
        ("user_share", "comparison_share"),
        ("kl_contribution", "kl_contribution"),
        ("jsd_contribution", "jsd_contribution"),
    ]:
        np.testing.assert_allclose(
            measured["area_level"][measured_column],
            validated["od_level"][flow_column],
            rtol=1e-12,
        )
    assert measured["summary"].jsd_population_user.iloc[0] == pytest.approx(
        validated["summary"].jsd_mean.iloc[0]
    )


def test_grouped_distribution_shares_keep_source_time_denominators():
    data = pd.DataFrame(
        {
            "origin": ["A"] * 4,
            "destination": ["X", "Y"] * 2,
            "mpd_source": ["first"] * 2 + ["second"] * 2,
            "flow_adj": [1, 3, 8, 2],
        }
    )
    benchmark = data.drop(columns="flow_adj").assign(flow=[2, 2, 5, 5])
    result = validate_flow_distribution(
        data, benchmark, epsilon=1, return_od_level=True
    )
    np.testing.assert_allclose(
        result["od_level"].comparison_share, [2 / 6, 4 / 6, 9 / 12, 3 / 12]
    )
    assert result["origin_level"].n_destinations.tolist() == [2, 2]
    assert result["origin_level"].comparison_origin_total.tolist() == [4, 10]
