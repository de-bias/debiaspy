import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from adjustment_inputs import simple_mpd
from debiaspy import (
    adjust_raking_ratio,
)
from debiaspy._exceptions import DebiasSchemaError


def test_raking_ratio_origin_targets_scales_each_origin_margin():
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [90.0, 140.0]})

    actual = adjust_raking_ratio(simple_mpd(), origin_targets=origin_targets)

    expected = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "mpd_source": ["s1", "s1", "s1", "s1"],
            "flow": [10.0, 20.0, 30.0, 40.0],
            "flow_adj": [30.0, 60.0, 60.0, 80.0],
            "weight_ipf": [3.0, 3.0, 2.0, 2.0],
        }
    )
    assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.attrs["ipf_converged"] is True
    assert actual.attrs["ipf_iterations"] == 2


def test_raking_ratio_requires_a_target_source():
    with pytest.raises(DebiasSchemaError, match="Provide at least one"):
        adjust_raking_ratio(simple_mpd())


def test_raking_ratio_benchmark_targets_match_numeric_ids_like_r():
    mpd = pd.DataFrame(
        {
            "origin": [1, 1, 2, 2],
            "destination": [1, 2, 1, 2],
            "flow": [10.0, 20.0, 30.0, 40.0],
        }
    )
    benchmark = pd.DataFrame(
        {
            "origin": [1, 1, 2, 2],
            "destination": [1, 2, 1, 2],
            "flow": [100.0, 200.0, 90.0, 120.0],
        }
    )

    actual = adjust_raking_ratio(mpd, benchmark_od_df=benchmark)
    origin_totals = actual.groupby("origin", sort=True, as_index=False).agg(
        flow_adj=("flow_adj", "sum")
    )
    destination_totals = actual.groupby("destination", sort=True, as_index=False).agg(
        flow_adj=("flow_adj", "sum")
    )

    assert actual["flow_adj"].sum() == pytest.approx(510.0)
    assert dict(
        zip(origin_totals["origin"], origin_totals["flow_adj"], strict=True)
    ) == pytest.approx({1: 300.0, 2: 210.0})
    assert dict(
        zip(
            destination_totals["destination"],
            destination_totals["flow_adj"],
            strict=True,
        )
    ) == pytest.approx({1: 190.0, 2: 320.0})
    assert actual["flow_adj"].tolist() != actual["flow"].tolist()


def test_raking_ratio_fails_closed_on_missing_flow_in_active_margin():
    mpd = simple_mpd()
    mpd.loc[0, "flow"] = np.nan
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [30.0, 70.0]})

    with pytest.raises(DebiasSchemaError, match="non-finite values"):
        adjust_raking_ratio(mpd, origin_targets=origin_targets)


def test_raking_ratio_fails_closed_on_impossible_structural_zero_margin():
    mpd = simple_mpd()
    mpd.loc[mpd["origin"] == "A", "flow"] = 0.0
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [30.0, 70.0]})

    with pytest.raises(DebiasSchemaError, match="zero support"):
        adjust_raking_ratio(mpd, origin_targets=origin_targets)


def test_raking_ratio_clipping_keeps_weight_and_flow_consistent():
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [90.0, 140.0]})

    actual = adjust_raking_ratio(
        simple_mpd(),
        origin_targets=origin_targets,
        clip_max=2.5,
    )

    np.testing.assert_allclose(
        actual["flow_adj"], actual["flow"] * actual["weight_ipf"]
    )
    assert actual["weight_ipf"].max() == pytest.approx(2.5)
    assert actual.attrs["ipf_converged"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0}, "tol"),
        ({"clip_min": -1}, "clip_min"),
        ({"clip_min": 2, "clip_max": 1}, "clip_min"),
    ],
)
def test_raking_ratio_validates_iteration_and_clipping_controls(kwargs, message):
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [90.0, 140.0]})

    with pytest.raises(DebiasSchemaError, match=message):
        adjust_raking_ratio(simple_mpd(), origin_targets=origin_targets, **kwargs)


def test_raking_ratio_rejects_negative_or_duplicate_targets():
    negative = pd.DataFrame({"origin": ["A", "B"], "target": [-1.0, 70.0]})
    duplicate = pd.DataFrame({"origin": ["A", "A"], "target": [30.0, 31.0]})

    with pytest.raises(DebiasSchemaError, match="finite non-negative"):
        adjust_raking_ratio(simple_mpd(), origin_targets=negative)
    with pytest.raises(DebiasSchemaError, match="at most one target"):
        adjust_raking_ratio(simple_mpd(), origin_targets=duplicate)
