"""Run from the issue-10-local root with R, dplyr and sibling debiasR installed."""

import subprocess

import numpy as np
import pandas as pd

from debiaspy import adjust_all_methods
from debiaspy.adjust_all_methods import _merge_args

r_code = """
suppressPackageStartupMessages(library(dplyr))
source("../debiasR/R/adjust_all_methods.R")
source("../debiasR/R/adjust_inverse_penetration.R")
x <- data.frame(origin="A", destination="B", flow=2)
coverage <- data.frame(origin=c("A", "B"), population=c(10,30), user_count=c(2,3))
merged <- .adjust_all_merge_args(
  list(control=list(fit=list(max_iter=100, tol=0.01), trace=TRUE)),
  list(control=list(fit=list(max_iter=20), trace=NULL))
)
stopifnot("trace" %in% names(merged$control), is.null(merged$control$trace))
cat(merged$control$fit$max_iter, merged$control$fit$tol, "\\n")
for (args in list(list(), list(weight_by="destination"))) {
  result <- adjust_all_methods(x, coverage, methods="inverse_penetration",
                               inverse_penetration_args=args)
  stopifnot(identical(names(result), attr(result, "methods")))
  cat(sprintf("%.17g", result$inverse_penetration$flow_adj), "\\n")
}
"""
reference = subprocess.check_output(["Rscript", "-e", r_code], text=True).split()
defaults = {"control": {"fit": {"max_iter": 100, "tol": 0.01}, "trace": True}}
overrides = {"control": {"fit": {"max_iter": 20}, "trace": None}}
control = _merge_args(defaults, overrides)["control"]
assert control["trace"] is None
observed = pd.DataFrame({"origin": ["A"], "destination": ["B"], "flow": [2]})
coverage = pd.DataFrame(
    {"origin": ["A", "B"], "population": [10, 30], "user_count": [2, 3]}
)
actual = [control["fit"]["max_iter"], control["fit"]["tol"]]
for args in ({}, {"weight_by": "destination"}):
    result = adjust_all_methods(
        observed, coverage, methods="inverse_penetration", inverse_penetration_args=args
    )
    assert result.attrs["methods"] == list(result) == ["inverse_penetration"]
    actual.append(result["inverse_penetration"]["flow_adj"].iloc[0])
np.testing.assert_allclose(actual, [float(x) for x in reference], rtol=1e-12, atol=0)
print("R/Python match: nested overrides, default flow and overridden flow.")
