"""Compare reviewed input boundaries directly with the pinned R implementation."""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from debiaspy import (
    adjust_inverse_penetration,
    adjust_selection_rate,
    adjust_selection_rate2,
)


@pytest.fixture(scope="module")
def r_input_results(tmp_path_factory):
    configured_root = os.environ.get("DEBIASR_ROOT")
    if configured_root is None or shutil.which("Rscript") is None:
        pytest.skip("Set DEBIASR_ROOT and install Rscript for live R comparisons")
    reference = Path(configured_root).expanduser().resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == "6fa2709ae42de6dc525a4a46182fdfb448b14a3f"
    assert not subprocess.check_output(
        ["git", "-C", str(reference), "status", "--porcelain"], text=True
    ).strip()
    output = tmp_path_factory.mktemp("r-input-semantics")
    script = output / "compare.R"
    script.write_text(
        """args <- commandArgs(trailingOnly = TRUE)
source(file.path(args[[1]], "R", "adjust_inverse_penetration.R"))
library(dplyr)
source(file.path(args[[1]], "R", "adjust_selection_rate.R"))
source(file.path(args[[1]], "R", "adjust_selection_rate2.R"))
write_result <- function(value, name) {
  utils::write.csv(value, file.path(args[[2]], paste0(name, ".csv")), row.names = FALSE)
}
coverage <- data.frame(
  origin = LETTERS[1:7], population = c(100, 100, 100, Inf, 0, -100, NA),
  user_count = c(0, NA, Inf, 10, 0, 10, 10)
)
mpd <- data.frame(origin = LETTERS[1:8], destination = c(LETTERS[2:6], "A", "A", "A"),
                  flow = c(1, 2, Inf, 4, 0, -6, NA, 8))
for (weight_by in c("origin", "destination", "both")) {
  write_result(suppressWarnings(adjust_inverse_penetration(mpd, coverage,
    weight_by = weight_by)), paste0("inverse_", weight_by))
}
mpd$flow_adj <- -99
write_result(adjust_inverse_penetration(mpd, coverage), "inverse_overwrite")
coverage <- data.frame(origin = c("A", "A", "B"), population = c(100, 100, 200),
                       user_count = c(10, 20, 50))
mpd <- data.frame(origin = c("A", "A", "B"), destination = c("A", "B", "A"),
                  flow = c(1, 2, 3))
write_result(suppressWarnings(adjust_inverse_penetration(mpd, coverage)),
             "inverse_duplicates")
cases <- c("basic", "incomplete", "repeated", "boundary", "conflicting", "grouped")
for (case in cases) {
  coverage <- data.frame(origin = c("A", "B"), population = c(100, 200),
                         user_count = c(10, 50), mpd_source = "s")
  mpd <- data.frame(origin = c("A", "A", "B", "B"),
                    destination = c("A", "B", "A", "B"),
                    flow = c(1, 2, 3, 4), mpd_source = "s")
  benchmark <- mpd
  benchmark$flow <- c(10, 20, 12, 16)
  if (case == "incomplete") {
    coverage <- coverage[1, ]
  } else if (case == "repeated") {
    mpd <- mpd[c(1, 1), ]
    mpd$flow <- c(10, 20)
    mpd$mpd_time <- c("t1", "t2")
    benchmark <- mpd
    benchmark$flow <- c(100, 300)
  } else if (case == "boundary") {
    coverage <- data.frame(origin = LETTERS[1:5],
      population = c(0, 100, Inf, 100, 100),
      user_count = c(10, 0, 10, NA, -1), mpd_source = "s")
    mpd <- data.frame(origin = LETTERS[1:5], destination = "A",
                      flow = c(1, 2, 3, NA, -5), mpd_source = "s")
    benchmark <- mpd
    benchmark$flow <- c(10, 20, Inf, NA, -50)
  } else if (case == "conflicting") {
    extra <- coverage[1, ]
    extra$user_count <- 20
    coverage <- rbind(coverage, extra)
  }
  group_cols <- NULL
  if (case == "grouped") {
    mpd <- mpd[c(1, 1), ]
    mpd$flow <- c(10, 20)
    mpd$age <- c("young", "old")
    coverage <- coverage[c(1, 1), ]
    coverage$user_count <- c(10, 20)
    coverage$age <- c("young", "old")
    benchmark <- mpd
    benchmark$flow <- c(100, 300)
    group_cols <- "age"
  }
  for (aggregate in c("origin", "od")) {
    for (method in c("selection_rate", "selection_rate2")) {
      if (method == "selection_rate") {
        result <- suppressWarnings(adjust_selection_rate(mpd, coverage,
          benchmark_od_df = benchmark, r_grid = c(0.1, 0.2, 0.5),
          calibration_aggregate = aggregate))
        diagnostic <- attr(result, "r_calibration")
        parameter <- attr(result, "r_global")
      } else {
        result <- suppressWarnings(adjust_selection_rate2(mpd, coverage,
          benchmark_od_df = benchmark, k_grid = c(0.5, 1, 2),
          calibration_aggregate = aggregate, group_cols = group_cols))
        diagnostic <- attr(result, "k_calibration")
        parameter <- attr(result, "k")
      }
      name <- paste(method, case, aggregate, sep = "_")
      write_result(result, name)
      write_result(diagnostic, paste0(name, "_loss"))
      write_result(data.frame(parameter = parameter), paste0(name, "_parameter"))
    }
  }
}
"""
    )
    subprocess.run(
        ["Rscript", str(script), str(reference.resolve()), str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return output


@pytest.mark.parametrize("weight_by", ["origin", "destination", "both"])
def test_inverse_penetration_numeric_boundaries_match_r(r_input_results, weight_by):
    coverage = pd.DataFrame(
        {
            "origin": list("ABCDEFG"),
            "population": [100, 100, 100, np.inf, 0, -100, np.nan],
            "user_count": [0, np.nan, np.inf, 10, 0, 10, 10],
        }
    )
    mpd = pd.DataFrame(
        {
            "origin": list("ABCDEFGH"),
            "destination": list("BCDEFAAA"),
            "flow": [1, 2, np.inf, 4, 0, -6, np.nan, 8],
        }
    )
    if weight_by == "both":
        with pytest.warns(RuntimeWarning, match="NaNs produced"):
            actual = adjust_inverse_penetration(mpd, coverage, weight_by=weight_by)
    else:
        actual = adjust_inverse_penetration(mpd, coverage, weight_by=weight_by)
    expected = pd.read_csv(r_input_results / f"inverse_{weight_by}.csv")
    assert_frame_equal(actual, expected, check_dtype=False)


def test_inverse_penetration_row_expansion_matches_r(r_input_results):
    coverage = pd.DataFrame(
        {
            "origin": ["A", "A", "B"],
            "population": [100, 100, 200],
            "user_count": [10, 20, 50],
        }
    )
    mpd = pd.DataFrame(
        {"origin": ["A", "A", "B"], "destination": ["A", "B", "A"], "flow": [1, 2, 3]}
    )
    with pytest.warns(UserWarning, match="many-to-many"):
        actual = adjust_inverse_penetration(mpd, coverage)
    expected = pd.read_csv(r_input_results / "inverse_duplicates.csv")
    assert_frame_equal(actual, expected, check_dtype=False)


@pytest.mark.parametrize("method", ["selection_rate", "selection_rate2"])
@pytest.mark.parametrize(
    "case", ["basic", "incomplete", "repeated", "boundary", "conflicting", "grouped"]
)
@pytest.mark.parametrize("aggregate", ["origin", "od"])
def test_selection_calibration_matches_r(r_input_results, method, case, aggregate):
    coverage = pd.DataFrame(
        {
            "origin": ["A", "B"],
            "population": [100, 200],
            "user_count": [10, 50],
            "mpd_source": "s",
        }
    )
    mpd = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "flow": [1, 2, 3, 4],
            "mpd_source": "s",
        }
    )
    benchmark = mpd.assign(flow=[10, 20, 12, 16])
    if case == "incomplete":
        coverage = coverage.iloc[[0]]
    elif case == "repeated":
        mpd = (
            mpd.iloc[[0, 0]]
            .reset_index(drop=True)
            .assign(flow=[10, 20], mpd_time=["t1", "t2"])
        )
        benchmark = mpd.assign(flow=[100, 300])
    elif case == "boundary":
        coverage = pd.DataFrame(
            {
                "origin": list("ABCDE"),
                "population": [0, 100, np.inf, 100, 100],
                "user_count": [10, 0, 10, np.nan, -1],
                "mpd_source": "s",
            }
        )
        mpd = pd.DataFrame(
            {
                "origin": list("ABCDE"),
                "destination": "A",
                "flow": [1, 2, 3, np.nan, -5],
                "mpd_source": "s",
            }
        )
        benchmark = mpd.assign(flow=[10, 20, np.inf, np.nan, -50])
    elif case == "conflicting":
        coverage = pd.concat(
            [coverage, coverage.iloc[[0]].assign(user_count=20)], ignore_index=True
        )
    group_cols = None
    if case == "grouped":
        mpd = (
            mpd.iloc[[0, 0]]
            .reset_index(drop=True)
            .assign(flow=[10, 20], age=["young", "old"])
        )
        coverage = (
            coverage.iloc[[0, 0]]
            .reset_index(drop=True)
            .assign(user_count=[10, 20], age=["young", "old"])
        )
        benchmark = mpd.assign(flow=[100, 300])
        group_cols = "age"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if method == "selection_rate":
            result = adjust_selection_rate(
                mpd,
                coverage,
                benchmark_od_df=benchmark,
                r_grid=[0.1, 0.2, 0.5],
                calibration_aggregate=aggregate,
            )
            diagnostic = result.attrs["r_calibration"]
            parameter = result.attrs["r_global"]
        else:
            result = adjust_selection_rate2(
                mpd,
                coverage,
                benchmark_od_df=benchmark,
                k_grid=[0.5, 1, 2],
                calibration_aggregate=aggregate,
                group_cols=group_cols,
            )
            diagnostic = result.attrs["k_calibration"]
            parameter = result.attrs["k"]
    relationship_warnings = [w for w in captured if "many-to-many" in str(w.message)]
    assert len(relationship_warnings) == (3 if case in {"repeated", "grouped"} else 0)
    name = f"{method}_{case}_{aggregate}"
    assert_frame_equal(
        result,
        pd.read_csv(r_input_results / f"{name}.csv"),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )
    assert_frame_equal(
        diagnostic,
        pd.read_csv(r_input_results / f"{name}_loss.csv"),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )
    expected_parameter = pd.read_csv(r_input_results / f"{name}_parameter.csv").iloc[
        0, 0
    ]
    assert parameter == expected_parameter


def test_inverse_penetration_overwrites_existing_adjusted_flow_like_r(r_input_results):
    coverage = pd.DataFrame(
        {
            "origin": list("ABCDEFG"),
            "population": [100, 100, 100, np.inf, 0, -100, np.nan],
            "user_count": [0, np.nan, np.inf, 10, 0, 10, 10],
        }
    )
    mpd = pd.DataFrame(
        {
            "origin": list("ABCDEFGH"),
            "destination": list("BCDEFAAA"),
            "flow": [1, 2, np.inf, 4, 0, -6, np.nan, 8],
            "flow_adj": -99,
        }
    )
    actual = adjust_inverse_penetration(mpd, coverage)
    expected = pd.read_csv(r_input_results / "inverse_overwrite.csv")
    assert_frame_equal(actual, expected, check_dtype=False)
