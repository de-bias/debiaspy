"""Private comparison conventions shared by flow-validation functions."""

from collections.abc import Sequence

_COMPARISON_SPECS = {
    "adjusted_vs_benchmark": {
        "comparison_label": "Adjusted vs benchmark",
        "x_series": "adjusted",
        "y_series": "benchmark",
        "signed_residual": "benchmark - adjusted",
    },
    "raw_vs_benchmark": {
        "comparison_label": "Raw MPD vs benchmark",
        "x_series": "raw_mpd",
        "y_series": "benchmark",
        "signed_residual": "benchmark - raw",
    },
    "raw_vs_adjusted": {
        "comparison_label": "Raw MPD vs adjusted",
        "x_series": "raw_mpd",
        "y_series": "adjusted",
        "signed_residual": "adjusted - raw",
    },
}
_SERIES_COLUMNS = {
    "adjusted": "adj_flow",
    "raw_mpd": "mpd_flow",
    "benchmark": "benchmark_flow",
}


def _normalise_flow_comparisons(comparisons: str | Sequence[str]) -> list[str]:
    if isinstance(comparisons, str):
        comparisons = [comparisons]
    if not isinstance(comparisons, Sequence) or not comparisons:
        raise ValueError("`comparisons` must select at least one flow comparison.")
    if list(comparisons) == ["all"]:
        return list(_COMPARISON_SPECS)
    if any(
        not isinstance(value, str) or value not in _COMPARISON_SPECS
        for value in comparisons
    ):
        raise ValueError(
            "`comparisons` must contain only 'adjusted_vs_benchmark', "
            "'raw_vs_benchmark', 'raw_vs_adjusted', or 'all'."
        )
    return list(dict.fromkeys(comparisons))


def _flow_comparison_spec(comparison: str) -> dict[str, str]:
    return {"comparison": comparison, **_COMPARISON_SPECS[comparison]}
