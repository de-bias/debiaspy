import pandas as pd


def toy_adjusted():
    return pd.DataFrame(
        {
            "origin": ["A", "A", "B"],
            "destination": ["X", "Y", "X"],
            "flow": [10.0, 20.0, 30.0],
            "flow_adj": [12.0, 18.0, 42.0],
        }
    )


def toy_benchmark():
    return pd.DataFrame(
        {
            "origin": ["A", "A", "B"],
            "destination": ["X", "Y", "X"],
            "flow": [15.0, 10.0, 45.0],
        }
    )
