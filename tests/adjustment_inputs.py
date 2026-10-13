import pandas as pd


def simple_mpd() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "mpd_source": ["s1", "s1", "s1", "s1"],
            "flow": [10.0, 20.0, 30.0, 40.0],
        }
    )


def simple_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["A", "B"],
            "population": [100.0, 200.0],
            "user_count": [10.0, 50.0],
            "mpd_source": ["s1", "s1"],
        }
    )


def simple_covariates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "area": ["A", "B"],
            "income_norm": [1.0, 0.0],
            "rural_pct": [0.25, 0.75],
        }
    )


def simple_benchmark() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "flow": [100.0, 200.0, 90.0, 120.0],
        }
    )
