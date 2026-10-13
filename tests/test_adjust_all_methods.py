import pandas as pd

from adjustment_inputs import (
    simple_benchmark,
    simple_covariates,
    simple_coverage,
    simple_mpd,
)
from debiaspy import (
    adjust_all_methods,
    adjust_coefficient,
    adjust_raking_ratio,
    adjust_selection_rate,
    adjust_selection_rate2,
)


def test_adjustment_keep_columns_are_selected_once_like_tidyselect():
    mpd = simple_mpd().assign(note=["a", "b", "c", "d"])
    benchmark = simple_benchmark().assign(mpd_source="s1")
    keep = ["origin", "note", "note", "flow"]
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [30.0, 70.0]})

    results = [
        adjust_coefficient(mpd, benchmark, keep_cols=keep),
        adjust_raking_ratio(mpd, origin_targets=origin_targets, keep_cols=keep),
        adjust_selection_rate(
            mpd,
            simple_coverage(),
            r_global=0.2,
            keep_cols=keep,
        ),
        adjust_selection_rate2(mpd, simple_coverage(), k=1.0, keep_cols=keep),
    ]

    for result in results:
        assert result.columns.is_unique
        assert result.columns.tolist().count("origin") == 1
        assert result.columns.tolist().count("note") == 1
        assert result.columns.tolist().count("flow") == 1


def test_adjust_all_methods_runs_selected_deterministic_methods():
    actual = adjust_all_methods(
        simple_mpd(),
        simple_coverage(),
        benchmark_od_df=simple_benchmark(),
        covariates_df=simple_covariates(),
        covariate_col="income_norm",
        methods=["inverse_penetration", "coefficient"],
    )

    assert list(actual) == ["inverse_penetration", "coefficient"]
    assert actual.attrs["methods"] == ["inverse_penetration", "coefficient"]
    assert "flow_adj" in actual["inverse_penetration"].columns
    assert "flow_adj" in actual["coefficient"].columns


def test_adjust_all_methods_uses_r_wrapper_method_order_not_caller_order():
    actual = adjust_all_methods(
        simple_mpd(),
        simple_coverage(),
        benchmark_od_df=simple_benchmark(),
        covariates_df=simple_covariates(),
        covariate_col="income_norm",
        methods=["coefficient", "inverse_penetration"],
    )

    assert list(actual) == ["inverse_penetration", "coefficient"]
    assert actual.attrs["methods"] == ["inverse_penetration", "coefficient"]
