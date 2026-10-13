import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from debiaspy import adjust_inverse_penetration
from debiaspy._exceptions import DebiasSchemaError


def toy_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["A", "B"],
            "population": [100, 200],
            "user_count": [10, 50],
        }
    )


def toy_mpd(flow_col: str = "flow") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            flow_col: [1, 2, 3, 4],
        }
    )


def test_origin_weighting_applies_inverse_penetration_weights():
    actual = adjust_inverse_penetration(toy_mpd(), toy_coverage(), weight_by="origin")

    expected = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "flow": [1, 2, 3, 4],
            "weight_origin": [10.0, 10.0, 4.0, 4.0],
            "flow_adj": [10.0, 20.0, 12.0, 16.0],
        }
    )
    assert_frame_equal(actual, expected, check_dtype=False)


def test_destination_weighting_applies_destination_weights():
    actual = adjust_inverse_penetration(
        toy_mpd(), toy_coverage(), weight_by="destination"
    )

    expected = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "flow": [1, 2, 3, 4],
            "weight_destination": [10.0, 4.0, 10.0, 4.0],
            "flow_adj": [10.0, 8.0, 30.0, 16.0],
        }
    )
    assert_frame_equal(actual, expected, check_dtype=False)


def test_both_weighting_uses_geometric_mean_weight():
    actual = adjust_inverse_penetration(toy_mpd(), toy_coverage(), weight_by="both")

    expected = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "flow": [1, 2, 3, 4],
            "weight_origin": [10.0, 10.0, 4.0, 4.0],
            "weight_destination": [10.0, 4.0, 10.0, 4.0],
            "weight_both": [10.0, (10.0 * 4.0) ** 0.5, (4.0 * 10.0) ** 0.5, 4.0],
            "flow_adj": [10.0, 2 * (10.0 * 4.0) ** 0.5, 3 * (4.0 * 10.0) ** 0.5, 16.0],
        }
    )
    assert_frame_equal(actual, expected, check_dtype=False)


def test_custom_flow_column_is_renamed_to_flow():
    actual = adjust_inverse_penetration(
        toy_mpd(flow_col="trips"), toy_coverage(), flow_col="trips"
    )

    assert list(actual.columns) == [
        "origin",
        "destination",
        "flow",
        "weight_origin",
        "flow_adj",
    ]
    assert actual["flow"].tolist() == [1, 2, 3, 4]
    assert actual["flow_adj"].tolist() == [10.0, 20.0, 12.0, 16.0]


def test_custom_flow_column_rejects_existing_flow_column_collision():
    mpd = toy_mpd(flow_col="trips")
    mpd["flow"] = [100, 200, 300, 400]

    with pytest.raises(DebiasSchemaError, match=r"`flow_col` cannot be renamed"):
        adjust_inverse_penetration(mpd, toy_coverage(), flow_col="trips")


@pytest.mark.parametrize(
    ("weight_by", "expected_weight_column"),
    [
        ("o", "weight_origin"),
        ("orig", "weight_origin"),
        ("d", "weight_destination"),
        ("dest", "weight_destination"),
        ("b", "weight_both"),
        ("bo", "weight_both"),
    ],
)
def test_weight_by_accepts_r_style_unambiguous_prefixes(
    weight_by, expected_weight_column
):
    actual = adjust_inverse_penetration(toy_mpd(), toy_coverage(), weight_by=weight_by)

    assert expected_weight_column in actual.columns


def test_missing_mpd_flow_column_raises_schema_error():
    with pytest.raises(DebiasSchemaError, match=r"`mpd_od_df` must contain"):
        adjust_inverse_penetration(toy_mpd().drop(columns="flow"), toy_coverage())


def test_invalid_weight_by_raises_clear_error():
    with pytest.raises(ValueError, match=r"`weight_by` must be one of"):
        adjust_inverse_penetration(toy_mpd(), toy_coverage(), weight_by="area")


def test_ambiguous_weight_by_prefix_raises_clear_error():
    with pytest.raises(ValueError, match=r"`weight_by` must be one of"):
        adjust_inverse_penetration(toy_mpd(), toy_coverage(), weight_by="")


def test_mpd_source_is_included_when_both_frames_have_source_column():
    mpd = pd.DataFrame(
        {
            "origin": ["A", "A"],
            "destination": ["B", "B"],
            "mpd_source": ["x", "y"],
            "flow": [2, 2],
        }
    )
    coverage = pd.DataFrame(
        {
            "origin": ["A", "B", "A", "B"],
            "mpd_source": ["x", "x", "y", "y"],
            "population": [100, 200, 300, 500],
            "user_count": [10, 50, 10, 100],
        }
    )

    actual = adjust_inverse_penetration(mpd, coverage, weight_by="both")

    expected = pd.DataFrame(
        {
            "origin": ["A", "A"],
            "destination": ["B", "B"],
            "mpd_source": ["x", "y"],
            "flow": [2, 2],
            "weight_origin": [10.0, 30.0],
            "weight_destination": [4.0, 5.0],
            "weight_both": [(10.0 * 4.0) ** 0.5, (30.0 * 5.0) ** 0.5],
            "flow_adj": [2 * (10.0 * 4.0) ** 0.5, 2 * (30.0 * 5.0) ** 0.5],
        }
    )
    assert_frame_equal(actual, expected, check_dtype=False)
