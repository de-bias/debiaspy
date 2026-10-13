"""Flow-level spatial regressions from the pinned debiasR test cases."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from debiaspy import validate_flow_residual_structure


def spatial_inputs():
    adjusted = pd.DataFrame(
        {
            "origin": list("ABCDE"),
            "destination": ["X"] * 5,
            "flow": [10] * 5,
            "flow_adj": [12, 12, 8, 8, 10],
        }
    )
    benchmark = adjusted[["origin", "destination", "flow"]].copy()
    neighbors = pd.DataFrame({"area": list("ABCD"), "neighbor": list("BADC")})
    return adjusted, benchmark, neighbors


def structure(adjusted, benchmark, neighbors=None, **kwargs):
    return validate_flow_residual_structure(
        adjusted,
        benchmark,
        area_neighbors=neighbors,
        local_moran=True,
        local_moran_nsim=19,
        local_moran_alpha=1,
        local_moran_p_adjust="none",
        local_moran_seed=123,
        **kwargs,
    )


def test_global_and_local_moran_match_r_cluster_fixture():
    adjusted, benchmark, neighbors = spatial_inputs()
    before = adjusted.copy(deep=True)
    result = structure(adjusted, benchmark, neighbors)
    assert result["moran_i"].moran_i.iloc[0] == pytest.approx(1.25)
    local = result["local_moran"]
    np.testing.assert_allclose(local.selected_residual, [-2, -2, 2, 2, 0])
    np.testing.assert_allclose(local.local_moran_i, [1.25] * 4 + [np.nan])
    assert local.lisa_cluster.tolist() == [
        "low-low",
        "low-low",
        "high-high",
        "high-high",
        "no neighbours",
    ]
    assert result["map_data"].lisa_cluster.tolist() == local.lisa_cluster.tolist()
    pd.testing.assert_frame_equal(adjusted, before)
    repeated = structure(adjusted, benchmark, neighbors)
    pd.testing.assert_frame_equal(local, repeated["local_moran"])


def test_weighted_links_and_isolates_keep_r_denominator():
    adjusted, benchmark, _ = spatial_inputs()
    adjusted = adjusted.iloc[:3].assign(flow_adj=[12, 10, 8])
    benchmark = benchmark.iloc[:3]
    neighbors = pd.DataFrame(
        {
            "area": list("AABC"),
            "neighbor": list("BCAA"),
            "weight": [1, 3, 2, 4],
        }
    )
    result = structure(adjusted, benchmark, neighbors, weight_col="weight")
    a = result["local_moran"].set_index("area").loc["A"]
    assert a.n_neighbors_used == 2
    assert a.local_weight_sum == 4
    assert a.neighbor_lag_mean == pytest.approx(1.5)
    assert a.local_moran_i == pytest.approx(-4.5)


@pytest.mark.parametrize(
    "neighbors", [None, pd.DataFrame({"area": ["Q"], "neighbor": ["Z"]})]
)
def test_missing_links_are_reported_without_removing_areas(neighbors):
    adjusted, benchmark, _ = spatial_inputs()
    result = structure(adjusted, benchmark, neighbors)
    assert len(result["local_moran"]) == 5
    assert result["local_moran"].lisa_cluster.tolist() == ["no neighbours"] * 5
    assert result["local_moran"].local_moran_i.isna().all()


def test_constant_residuals_are_undefined_and_local_moran_is_opt_in():
    adjusted, benchmark, neighbors = spatial_inputs()
    adjusted = adjusted.assign(flow_adj=11)
    result = structure(adjusted, benchmark, neighbors)
    assert result["local_moran"].local_moran_i.isna().all()
    assert result["local_moran"].lisa_cluster.tolist() == ["undefined"] * 5
    assert "local_moran" not in validate_flow_residual_structure(adjusted, benchmark)


@pytest.mark.parametrize(
    "role, aggregation",
    [
        ("origin", "mean"),
        ("origin", "sum"),
        ("destination", "mean"),
        ("destination", "sum"),
    ],
)
def test_residual_signs_area_aggregation_covariates_and_geometry(role, aggregation):
    adjusted, benchmark, _ = spatial_inputs()
    adjusted = pd.concat(
        [adjusted, adjusted.assign(destination="Y")], ignore_index=True
    )
    benchmark = pd.concat(
        [benchmark, benchmark.assign(destination="Y")], ignore_index=True
    )
    keys = list("ABCDE") if role == "origin" else ["X", "Y"]
    geometry = pd.DataFrame({"area": keys, "x": np.arange(len(keys)), "y": 1})
    covariates = pd.DataFrame({"area": keys, "income": np.arange(len(keys))})
    result = structure(
        adjusted,
        benchmark,
        spatial_role=role,
        residual_aggregation=aggregation,
        covariate_df=covariates,
        covariate_col="income",
        geometry_df=geometry,
    )
    expected = np.array([-2, -2, 2, 2, 0]) if role == "origin" else np.zeros(2)
    if aggregation == "sum" and role == "origin":
        expected *= 2
    np.testing.assert_allclose(result["area_level"].selected_residual, expected)
    assert len(result["map_data"]) == len(keys)
    np.testing.assert_array_equal(
        result["map_data"].x.to_numpy(), geometry.x.to_numpy()
    )
    assert len(result["covariate_data"]) == len(keys)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"area_neighbors": pd.DataFrame({"area": ["A"]})}, "area_neighbors"),
        (
            {
                "area_neighbors": pd.DataFrame({"area": ["A"], "neighbor": ["B"]}),
                "weight_col": "missing",
            },
            "weight_col",
        ),
        ({"local_moran": True, "local_moran_nsim": 0}, "nsim"),
        ({"local_moran": True, "local_moran_alpha": 2}, "alpha"),
        ({"local_moran": True, "local_moran_p_adjust": "invalid"}, "p_adjust"),
    ],
)
def test_invalid_spatial_inputs_identify_the_argument(kwargs, message):
    adjusted, benchmark, _ = spatial_inputs()
    with pytest.raises(ValueError, match=message):
        validate_flow_residual_structure(adjusted, benchmark, **kwargs)


def test_live_r_global_local_and_weighted_moran(tmp_path: Path):
    root = os.environ.get("DEBIASR_ROOT")
    if not root or not shutil.which("Rscript"):
        pytest.skip("Set DEBIASR_ROOT with Rscript for live spatial parity")
    adjusted, benchmark, neighbors = spatial_inputs()
    adjusted.to_csv(tmp_path / "adjusted.csv", index=False)
    benchmark.to_csv(tmp_path / "benchmark.csv", index=False)
    neighbors.to_csv(tmp_path / "neighbors.csv", index=False)
    script = r"""
args <- commandArgs(TRUE)
pkgload::load_all(args[[1]], export_all=FALSE, helpers=FALSE, quiet=TRUE)
root <- args[[2]]
adj <- read.csv(file.path(root,"adjusted.csv"))
bench <- read.csv(file.path(root,"benchmark.csv"))
neighbors <- read.csv(file.path(root,"neighbors.csv"))
for (case in c("clusters", "weighted", "constant", "isolated")) {
  a <- adj; b <- bench; links <- neighbors; weight <- NULL
  if (case == "weighted") {
    a <- a[1:3, ]; a$flow_adj <- c(12,10,8); b <- b[1:3, ]
    links <- data.frame(area=c("A","A","B","C"),
      neighbor=c("B","C","A","A"), weight=c(1,3,2,4))
    weight <- "weight"
  }
  if (case == "constant") a$flow_adj <- 11
  if (case == "isolated") links <- NULL
  res <- debiasR::validate_flow_residual_structure(a,b,area_neighbors=links,
    weight_col=weight,local_moran=TRUE,local_moran_nsim=19,local_moran_alpha=1,
    local_moran_p_adjust="none",local_moran_seed=123)
  write.csv(res$local_moran,file.path(root,paste0(case,".local.csv")),row.names=FALSE)
  write.csv(res$moran_i,file.path(root,paste0(case,".global.csv")),row.names=FALSE)
}
"""
    (tmp_path / "compare.R").write_text(script)
    subprocess.run(
        ["Rscript", str(tmp_path / "compare.R"), root, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for case in ["clusters", "weighted", "constant", "isolated"]:
        a, b, links = adjusted, benchmark, neighbors
        kwargs = {}
        if case == "weighted":
            a = a.iloc[:3].assign(flow_adj=[12, 10, 8])
            b = b.iloc[:3]
            links = pd.DataFrame(
                {"area": list("AABC"), "neighbor": list("BCAA"), "weight": [1, 3, 2, 4]}
            )
            kwargs["weight_col"] = "weight"
        elif case == "constant":
            a = a.assign(flow_adj=11)
        elif case == "isolated":
            links = None
        actual = structure(a, b, links, **kwargs)
        reference = pd.read_csv(tmp_path / f"{case}.local.csv")
        for column in [
            "selected_residual",
            "residual_z",
            "spatial_lag_z",
            "local_moran_i",
            "n_neighbors_used",
        ]:
            np.testing.assert_allclose(
                actual["local_moran"][column],
                reference[column],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        assert (
            actual["local_moran"].lisa_cluster.tolist()
            == reference.lisa_cluster.tolist()
        )
        global_reference = pd.read_csv(tmp_path / f"{case}.global.csv")
        np.testing.assert_allclose(
            actual["moran_i"].moran_i, global_reference.moran_i, equal_nan=True
        )
