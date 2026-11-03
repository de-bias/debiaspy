import math

import numpy as np
import pandas as pd
import pytest

from debiaspy import adjust_raking_ratio


def _smoke_od() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["X", "Y", "X", "Y"],
            "flow": [10.0, 20.0, 30.0, 40.0],
        }
    )


def _smoke_origin_targets() -> pd.DataFrame:
    return pd.DataFrame({"origin": ["A", "B"], "target": [60.0, 40.0]})


def _smoke_destination_targets() -> pd.DataFrame:
    return pd.DataFrame({"destination": ["X", "Y"], "target": [50.0, 50.0]})


def _expected_smoke_flow_adj() -> list[float]:
    root = 5.0 * math.sqrt(601.0)
    return [
        root - 95.0,
        155.0 - root,
        145.0 - root,
        root - 105.0,
    ]


def test_adjust_raking_ratio_matches_two_way_margins_and_closed_form() -> None:
    mpd = _smoke_od()
    origin_targets = _smoke_origin_targets()
    destination_targets = _smoke_destination_targets()

    result = adjust_raking_ratio(
        mpd_od_df=mpd,
        origin_targets=origin_targets,
        destination_targets=destination_targets,
        max_iter=200,
        tol=1e-10,
    )

    assert list(result.columns) == [
        "origin",
        "destination",
        "flow",
        "flow_adj",
        "weight_ipf",
    ]
    assert result.attrs["ipf_converged"] is True
    assert result.attrs["ipf_iterations"] > 0

    expected = _expected_smoke_flow_adj()
    np.testing.assert_allclose(result["flow_adj"], expected, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        result["weight_ipf"],
        result["flow_adj"] / result["flow"],
        rtol=0,
        atol=1e-12,
    )

    adj_origin = result.groupby("origin", sort=False)["flow_adj"].sum()
    assert adj_origin["A"] == pytest.approx(60.0, abs=1e-6)
    assert adj_origin["B"] == pytest.approx(40.0, abs=1e-6)

    adj_destination = result.groupby("destination", sort=False)["flow_adj"].sum()
    assert adj_destination["X"] == pytest.approx(50.0, abs=1e-6)
    assert adj_destination["Y"] == pytest.approx(50.0, abs=1e-6)


def test_adjust_raking_ratio_origin_only_matches_targets() -> None:
    mpd = _smoke_od()
    origin_targets = pd.DataFrame(
        {
            "origin": ["A", "B"],
            "target": [33.0, 77.0],
        }
    )

    result = adjust_raking_ratio(
        mpd_od_df=mpd,
        origin_targets=origin_targets,
        tol=1e-10,
    )

    adj_origin = result.groupby("origin", sort=False)["flow_adj"].sum()
    assert adj_origin["A"] == pytest.approx(33.0, abs=1e-8)
    assert adj_origin["B"] == pytest.approx(77.0, abs=1e-8)
    assert result.attrs["ipf_converged"] is True


def test_adjust_raking_ratio_destination_only_matches_targets() -> None:
    mpd = _smoke_od()
    destination_targets = pd.DataFrame(
        {
            "destination": ["X", "Y"],
            "target": [70.0, 30.0],
        }
    )

    result = adjust_raking_ratio(
        mpd_od_df=mpd,
        destination_targets=destination_targets,
        tol=1e-10,
    )

    adj_destination = result.groupby("destination", sort=False)["flow_adj"].sum()
    assert adj_destination["X"] == pytest.approx(70.0, abs=1e-8)
    assert adj_destination["Y"] == pytest.approx(30.0, abs=1e-8)


def test_adjust_raking_ratio_uses_benchmark_margins_with_override() -> None:
    mpd = _smoke_od()
    benchmark = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["X", "Y", "X", "Y"],
            "flow": [25.0, 35.0, 15.0, 25.0],
        }
    )
    origin_override = pd.DataFrame({"origin": ["A", "B"], "target": [70.0, 30.0]})

    result = adjust_raking_ratio(
        mpd_od_df=mpd,
        origin_targets=origin_override,
        benchmark_od_df=benchmark,
        tol=1e-10,
    )

    adj_origin = result.groupby("origin", sort=False)["flow_adj"].sum()
    assert adj_origin["A"] == pytest.approx(70.0, abs=1e-6)
    assert adj_origin["B"] == pytest.approx(30.0, abs=1e-6)

    adj_destination = result.groupby("destination", sort=False)["flow_adj"].sum()
    assert adj_destination["X"] == pytest.approx(40.0, abs=1e-6)
    assert adj_destination["Y"] == pytest.approx(60.0, abs=1e-6)


def test_adjust_raking_ratio_stratified_group_cols() -> None:
    mpd = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "destination": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "age_group": ["young"] * 4 + ["old"] * 4,
            "flow": [10.0, 20.0, 30.0, 40.0, 5.0, 15.0, 25.0, 35.0],
        }
    )
    origin_targets = pd.DataFrame(
        {
            "origin": ["A", "B", "A", "B"],
            "age_group": ["young", "young", "old", "old"],
            "target": [45.0, 55.0, 30.0, 50.0],
        }
    )

    result = adjust_raking_ratio(
        mpd_od_df=mpd,
        origin_targets=origin_targets,
        group_cols="age_group",
        tol=1e-10,
    )

    assert "age_group" in result.columns
    for group, target_scale in (("young", [45.0, 55.0]), ("old", [30.0, 50.0])):
        subset = result.loc[result["age_group"] == group]
        adj_origin = subset.groupby("origin", sort=False)["flow_adj"].sum()
        assert adj_origin["A"] == pytest.approx(target_scale[0], abs=1e-8)
        assert adj_origin["B"] == pytest.approx(target_scale[1], abs=1e-8)


def test_adjust_raking_ratio_preserves_input_and_carries_columns() -> None:
    mpd = _smoke_od().assign(mpd_source="demo", note="keep-me")
    original = mpd.copy()

    result = adjust_raking_ratio(
        mpd_od_df=mpd,
        origin_targets=_smoke_origin_targets(),
        destination_targets=_smoke_destination_targets(),
        keep_cols=("note",),
        tol=1e-10,
    )

    assert mpd.equals(original)
    assert list(result.columns) == [
        "origin",
        "destination",
        "mpd_source",
        "note",
        "flow",
        "flow_adj",
        "weight_ipf",
    ]
    assert result["mpd_source"].tolist() == ["demo"] * 4
    assert result["note"].tolist() == ["keep-me"] * 4
    assert result["origin"].tolist() == original["origin"].tolist()


def test_adjust_raking_ratio_reports_non_convergence() -> None:
    mpd = _smoke_od()

    result = adjust_raking_ratio(
        mpd_od_df=mpd,
        origin_targets=_smoke_origin_targets(),
        destination_targets=_smoke_destination_targets(),
        max_iter=1,
        tol=1e-12,
    )

    assert result.attrs["ipf_converged"] is False
    assert result.attrs["ipf_iterations"] == 1
    assert len(result) == 4


def test_adjust_raking_ratio_clipping_keeps_identity_and_marks_unconverged() -> None:
    mpd = _smoke_od()

    result = adjust_raking_ratio(
        mpd_od_df=mpd,
        origin_targets=_smoke_origin_targets(),
        destination_targets=_smoke_destination_targets(),
        clip_max=1.5,
        tol=1e-10,
    )

    np.testing.assert_allclose(
        result["flow_adj"],
        result["flow"] * result["weight_ipf"],
        rtol=0,
        atol=1e-12,
    )
    assert result["weight_ipf"].le(1.5).all()
    assert result.attrs["ipf_converged"] is False


def test_adjust_raking_ratio_errors_on_invalid_controls_and_data() -> None:
    mpd = _smoke_od()
    origin_targets = _smoke_origin_targets()

    with pytest.raises(ValueError, match="max_iter"):
        adjust_raking_ratio(mpd, origin_targets=origin_targets, max_iter=0)

    with pytest.raises(ValueError, match="max_iter"):
        adjust_raking_ratio(mpd, origin_targets=origin_targets, max_iter=True)

    with pytest.raises(ValueError, match="tol"):
        adjust_raking_ratio(mpd, origin_targets=origin_targets, tol=0)

    with pytest.raises(ValueError, match="clip_min"):
        adjust_raking_ratio(mpd, origin_targets=origin_targets, clip_min=-1)

    with pytest.raises(ValueError, match="clip_max"):
        adjust_raking_ratio(mpd, origin_targets=origin_targets, clip_min=2, clip_max=1)

    with pytest.raises(ValueError, match="mpd_od_df"):
        adjust_raking_ratio(
            mpd.drop(columns=["origin"]),
            origin_targets=origin_targets,
        )

    with pytest.raises(ValueError, match="Provide at least one"):
        adjust_raking_ratio(mpd)

    with pytest.raises(ValueError, match="finite non-negative"):
        adjust_raking_ratio(
            mpd.assign(flow=[10.0, -1.0, 30.0, 40.0]),
            origin_targets=origin_targets,
        )

    with pytest.raises(ValueError, match="duplicate target keys"):
        adjust_raking_ratio(
            mpd,
            origin_targets=pd.DataFrame({"origin": ["A", "A"], "target": [10.0, 20.0]}),
        )

    with pytest.raises(ValueError, match="absent from"):
        adjust_raking_ratio(
            mpd,
            origin_targets=pd.DataFrame({"origin": ["A", "Z"], "target": [30.0, 70.0]}),
        )

    with pytest.raises(ValueError, match="group_cols"):
        adjust_raking_ratio(
            mpd,
            origin_targets=pd.DataFrame(
                {"origin": ["A"], "age_group": ["young"], "target": [10.0]}
            ),
            group_cols="age_group",
        )


def test_adjust_raking_ratio_errors_on_structural_zero_support() -> None:
    mpd = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["X", "Y", "X", "Y"],
            "flow": [0.0, 0.0, 30.0, 40.0],
        }
    )
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [10.0, 70.0]})

    with pytest.raises(ValueError, match="zero flow support"):
        adjust_raking_ratio(mpd_od_df=mpd, origin_targets=origin_targets)


def test_adjust_raking_ratio_errors_on_canonicalised_duplicate_area_keys() -> None:
    mpd = pd.DataFrame(
        {
            "origin": ["1", "1", "2", "2"],
            "destination": ["X", "Y", "X", "Y"],
            "flow": [10.0, 20.0, 30.0, 40.0],
        }
    )
    origin_targets = pd.DataFrame(
        {
            "origin": [1, "1"],
            "target": [30.0, 40.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate target keys"):
        adjust_raking_ratio(mpd_od_df=mpd, origin_targets=origin_targets)


def test_adjust_raking_ratio_errors_on_missing_group_values() -> None:
    mpd = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["X", "Y", "X", "Y"],
            "age_group": ["young", "young", None, "old"],
            "flow": [10.0, 20.0, 30.0, 40.0],
        }
    )
    origin_targets = pd.DataFrame(
        {
            "origin": ["A", "B"],
            "age_group": ["young", "old"],
            "target": [30.0, 70.0],
        }
    )

    with pytest.raises(ValueError, match="missing values in group column"):
        adjust_raking_ratio(
            mpd_od_df=mpd,
            origin_targets=origin_targets,
            group_cols="age_group",
        )

    mpd_ok = mpd.assign(age_group=["young", "young", "old", "old"])
    targets_missing = pd.DataFrame(
        {
            "origin": ["A", "B"],
            "age_group": ["young", None],
            "target": [30.0, 70.0],
        }
    )
    with pytest.raises(ValueError, match="missing values in group column"):
        adjust_raking_ratio(
            mpd_od_df=mpd_ok,
            origin_targets=targets_missing,
            group_cols="age_group",
        )


def test_adjust_raking_ratio_zero_flow_cells_keep_identity() -> None:
    mpd = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["X", "Y", "X", "Y"],
            "flow": [10.0, 0.0, 30.0, 40.0],
        }
    )
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [20.0, 70.0]})

    result = adjust_raking_ratio(
        mpd_od_df=mpd,
        origin_targets=origin_targets,
        tol=1e-10,
    )

    zero_mask = result["destination"].eq("Y") & result["origin"].eq("A")
    zero_row = result.loc[zero_mask].iloc[0]
    assert zero_row["flow_adj"] == 0.0
    assert zero_row["weight_ipf"] == 1.0
    np.testing.assert_allclose(
        result["flow_adj"],
        result["flow"] * result["weight_ipf"],
        rtol=0,
        atol=1e-12,
    )
