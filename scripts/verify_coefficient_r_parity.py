"""Compare coefficient calibration with a local debiasR checkout."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels
from pandas.testing import assert_frame_equal

from debiaspy import adjust_coefficient

ROOT = Path(__file__).resolve().parents[1]


def comparison_cases():
    """Small deterministic examples covering the fitting and output contract."""
    observed = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["X", "Y", "X", "Y"],
            "flow": [10.0, 20.0, 30.0, 40.0],
        }
    )
    benchmark = observed.assign(flow=[100.0, 200.0, 90.0, 120.0])
    cases = []

    def add(name, mpd=observed, bench=benchmark, tolerance=1e-10, **options):
        cases.append((name, mpd, bench, options, tolerance))

    for family in ("ols", "poisson"):
        for level in ("od", "origin", "destination"):
            add(f"{family}_{level}", model_family=family, level=level)
    for level in ("od", "origin", "destination"):
        add(f"intercept_{level}", fit_intercept=True, level=level)
    for value, label in ((np.nan, "missing"), (np.inf, "infinite"), (-1.0, "negative")):
        add(f"{label}_observed", observed.assign(flow=[10.0, 20.0, 30.0, value]))
        add(
            f"{label}_benchmark",
            bench=benchmark.assign(flow=[100.0, 200.0, 90.0, value]),
        )
    for level in ("origin", "destination"):
        add(
            f"missing_margin_{level}",
            observed.assign(flow=[np.nan, 20.0, 30.0, 40.0]),
            level=level,
        )
    add("zero_observed", observed.assign(flow=[0.0, 20.0, 30.0, 40.0]))
    add("empty_benchmark", bench=benchmark.iloc[:0])
    add("empty_observed", observed.iloc[:0])
    add("one_pair", bench=benchmark.iloc[:1])
    add("unmatched_pairs", bench=benchmark.iloc[:2])
    add("zero_ols_response", bench=benchmark.assign(flow=0.0))
    add(
        "zero_poisson_response",
        bench=benchmark.assign(flow=0.0),
        model_family="poisson",
    )
    add(
        "fractional_poisson",
        bench=benchmark.assign(flow=[0.5, 1.5, 2.5, 3.5]),
        model_family="poisson",
        fit_intercept=True,
    )
    add(
        "aliased_intercept",
        observed.assign(flow=2.0),
        benchmark.assign(flow=2.0),
        fit_intercept=True,
    )
    add(
        "duplicate_od",
        pd.concat([observed, observed.iloc[:1]], ignore_index=True),
        pd.concat([benchmark, benchmark.iloc[:1]], ignore_index=True),
    )
    first = observed.assign(mpd_source="first")
    second = observed.assign(mpd_source="second")
    multi = pd.concat(
        [first, second, observed.assign(mpd_source="unmatched")], ignore_index=True
    )
    targets = pd.concat(
        [first.assign(flow=first.flow * 2), second.assign(flow=second.flow * 4)],
        ignore_index=True,
    )
    for level in ("od", "origin", "destination"):
        add(f"source_{level}", multi, targets, by_source=True, level=level)
    add("pooled_sources", multi.iloc[:8], targets)
    add(
        "missing_source",
        observed.assign(mpd_source=None),
        benchmark.assign(mpd_source=None),
        by_source=True,
    )
    add(
        "custom_columns",
        observed.rename(columns={"flow": "trips"}).assign(note="keep"),
        benchmark.rename(columns={"flow": "total"}),
        flow_col_mpd="trips",
        flow_col_bench="total",
        keep_cols=["note", "note", "absent"],
    )
    add("missing_required_column", observed.drop(columns="origin"))
    add("source_mismatch", first, second, by_source=True)
    counts = pd.DataFrame(
        {
            "origin": [f"o{i}" for i in range(12)],
            "destination": [f"d{i}" for i in range(12)],
            "flow": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14],
        }
    )
    target_counts = counts.assign(flow=[0, 0, 1, 0, 3, 1, 8, 2, 12, 4, 18, 3])
    add("negbin", counts, target_counts, tolerance=1e-7, model_family="negbin")
    add("zinb", counts, target_counts, tolerance=1e-5, model_family="zinb")
    add("zero_negbin_response", bench=benchmark.assign(flow=0.0), model_family="negbin")
    add("positive_only_zinb", model_family="zinb")
    boundary = pd.DataFrame(
        {
            "origin": [f"o{i}" for i in range(20)],
            "destination": [f"d{i}" for i in range(20)],
            "flow": np.tile([1.0, 2.0, 4.0, 8.0], 5),
        }
    )
    add(
        "zinb_boundary",
        boundary,
        boundary.assign(
            flow=[0, 0, 0, 8, 0, 0, 3, 7, 0, 0, 0, 10, 0, 1, 6, 8, 0, 0, 0, 12]
        ),
        tolerance=1e-5,
        model_family="zinb",
    )
    return cases


R_COMPARE = r"""
suppressPackageStartupMessages(library(dplyr))
library(jsonlite)
stopifnot(requireNamespace("MASS"), requireNamespace("pscl"))
args <- commandArgs(trailingOnly=TRUE)
source(file.path(args[[1]], "R", "adjust_coefficient.R"))
cases <- fromJSON(file.path(args[[2]], "cases.json"), simplifyVector=FALSE)
for (case in cases) {
  folder <- file.path(args[[2]], case$name)
  options <- case$options
  mpd <- read.csv(file.path(folder,"observed.csv"),
                  colClasses="character", na.strings="")
  bench <- read.csv(file.path(folder,"benchmark.csv"),
                    colClasses="character", na.strings="")
  observed_col <- if (is.null(options$flow_col_mpd)) "flow" else options$flow_col_mpd
  benchmark_col <- if (is.null(options$flow_col_bench)) {
    "flow"
  } else options$flow_col_bench
  if (observed_col %in% names(mpd)) {
    mpd[[observed_col]] <- as.numeric(mpd[[observed_col]])
  }
  if (benchmark_col %in% names(bench)) {
    bench[[benchmark_col]] <- as.numeric(bench[[benchmark_col]])
  }
  if (!is.null(options$keep_cols)) options$keep_cols <- unlist(options$keep_cols)
  seen <- character()
  status <- tryCatch(withCallingHandlers({
    result <- do.call(adjust_coefficient,
                      c(list(mpd_od_df=mpd,benchmark_od_df=bench),options))
    write.csv(as.data.frame(result),file.path(folder,"r_output.csv"),row.names=FALSE,na="")
    write.csv(attr(result,"model"),file.path(folder,"r_model.csv"),row.names=FALSE,na="")
    coefficient <- attr(result,"coef")
    if (!is.data.frame(coefficient)) coefficient <- data.frame(coef=coefficient)
    write.csv(coefficient,file.path(folder,"r_coef.csv"),row.names=FALSE,na="")
    list(status="ok")
  },warning=function(w) {
    seen <<- c(seen,conditionMessage(w));invokeRestart("muffleWarning")
  }),
  error=function(e) list(status="error",message=conditionMessage(e)))
  status$warnings <- as.list(seen)
  write_json(status,file.path(folder,"r_status.json"),auto_unbox=TRUE)
}
"""


def compare_table(actual, expected, tolerance):
    actual = actual.reset_index(drop=True).copy()
    for column in actual:
        if actual[column].dtype.kind == "O":
            actual[column] = actual[column].where(actual[column].notna(), np.nan)
    assert_frame_equal(actual, expected, check_dtype=False, atol=tolerance, rtol=0)
    deltas = []
    for column in actual.select_dtypes(include="number"):
        a, b = (
            actual[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
        )
        finite = np.isfinite(a) & np.isfinite(b)
        deltas.extend(np.abs(a[finite] - b[finite]))
    return max(deltas, default=0.0)


def main():
    reference = Path(os.environ["DEBIASR_ROOT"]).expanduser().resolve()
    if not (reference / "R" / "adjust_coefficient.R").is_file():
        raise FileNotFoundError("DEBIASR_ROOT must contain R/adjust_coefficient.R")
    cases = comparison_cases()
    rows, plotted = [], []
    example = None
    with tempfile.TemporaryDirectory(prefix="coefficient-parity-") as name:
        work = Path(name)
        for case, observed, benchmark, _options, _ in cases:
            folder = work / case
            folder.mkdir()
            observed.to_csv(folder / "observed.csv", index=False)
            benchmark.to_csv(folder / "benchmark.csv", index=False)
        (work / "cases.json").write_text(
            json.dumps([{"name": c[0], "options": c[3]} for c in cases])
        )
        script = work / "compare.R"
        script.write_text(R_COMPARE)
        subprocess.run(["Rscript", str(script), str(reference), str(work)], check=True)
        for case, observed, benchmark, options, tolerance in cases:
            r = json.loads((work / case / "r_status.json").read_text())
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                try:
                    result = adjust_coefficient(observed, benchmark, **options)
                    error = None
                except (ValueError, TypeError) as exc:
                    result, error = None, str(exc)
            detail, delta = "", None
            flow = pd.to_numeric(
                observed.get(
                    options.get("flow_col_mpd", "flow"), pd.Series(dtype=float)
                )
            )
            finite_flow = np.abs(flow[np.isfinite(flow)])
            flow_tolerance = tolerance * max(
                1.0, finite_flow.max() if len(finite_flow) else 1.0
            )
            if error or r["status"] == "error":
                passed = bool(error) and r["status"] == "error"
                status = "PASS (error)" if passed else "DIFFERENCE"
                detail = (
                    f"Python: {error or 'returned output'}; "
                    f"R: {r.get('message', 'returned output')}"
                )
            else:
                try:
                    expected = pd.read_csv(
                        work / case / "r_output.csv",
                        dtype={"origin": str, "destination": str},
                    )
                    delta = compare_table(result, expected, flow_tolerance)
                    if case == "ols_od":
                        example = (observed, benchmark, expected, result)
                    compare_table(
                        result.attrs["model"],
                        pd.read_csv(work / case / "r_model.csv"),
                        tolerance,
                    )
                    coef = result.attrs["coef"]
                    if not isinstance(coef, pd.DataFrame):
                        coef = pd.DataFrame({"coef": [coef]})
                    compare_table(
                        coef,
                        pd.read_csv(work / case / "r_coef.csv", skip_blank_lines=False),
                        tolerance,
                    )
                    pairs = np.column_stack([expected.flow_adj, result.flow_adj])
                    plotted.extend(pairs[np.isfinite(pairs).all(axis=1)])
                    status = "PASS"
                except AssertionError as exc:
                    status, detail = "DIFFERENCE", str(exc).replace("\n", " ")
                    print(case, str(exc))
            rows.append(
                {
                    "case": case,
                    "status": status,
                    "delta": delta,
                    "tolerance": tolerance,
                    "flow_tolerance": flow_tolerance,
                    "detail": detail,
                    "python_warnings": len(seen),
                    "r_warnings": len(r.get("warnings", [])),
                }
            )
    destination = ROOT / "docs" / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    figure, (ax, residual_ax) = plt.subplots(1, 2, figsize=(9, 4))
    points = np.array(plotted)
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], s=12, alpha=0.5)
        lo, hi = np.min(points), np.max(points)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        residual_ax.scatter(points[:, 0], points[:, 1] - points[:, 0], s=12, alpha=0.5)
    residual_ax.axhline(0, color="black", linewidth=1)
    residual_ax.set(
        xlabel="R adjusted flow", ylabel="Python - R", title="Numerical differences"
    )
    ax.set(
        xlabel="R adjusted flow",
        ylabel="Python adjusted flow",
        title="Coefficient calibration: synthetic parity cases",
    )
    figure.tight_layout()
    figure.savefig(destination / "issue-8-coefficient-r-parity.png", dpi=160)
    plt.close(figure)
    revision = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    r_version = subprocess.check_output(["Rscript", "--version"], text=True).strip()
    passed = sum(r["status"].startswith("PASS") for r in rows)
    report = [
        "# Issue #8: coefficient calibration parity",
        "",
        "R code: `adjust_coefficient.R`.",
        "Python code: `adjust_coefficient.py`.",
        "Tests: `test_adjust_coefficient.py`; `verify_coefficient_r_parity.py`.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv sync --locked",
        "DEBIASR_ROOT=/path/to/debiasR uv run python "
        "scripts/verify_coefficient_r_parity.py",
        "```",
        "",
        "## Environment",
        "",
        f"R reference: `{revision}`; `{r_version}`.",
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"pandas {pd.__version__}, statsmodels {statsmodels.__version__}, "
        f"SciPy {scipy.__version__}.",
        "",
        "",
        "## Results",
        "",
        f"**{passed}/{len(rows)} cases match**: adjusted rows, coefficient "
        "attributes, model summaries and expected errors. Coefficient/model "
        "tolerances are absolute. Flow tolerance propagates the coefficient "
        "tolerance through the largest finite observed flow (minimum scale 1). "
        "Matching missing/infinite values and output schemas are checked.",
        "",
        "| Case | Result | Max flow difference | Coef/model tolerance | "
        "Flow tolerance |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta = "-" if row["delta"] is None else f"{row['delta']:.3g}"
        report.append(
            f"| {row['case']} | {row['status']} | {delta} | "
            f"{row['tolerance']:.0e} | {row['flow_tolerance']:.3g} |"
        )
    if example is not None:
        observed, benchmark, expected, result = example
        report += [
            "",
            "## Default OLS fixture",
            "",
            "One common multiplier is fitted across these four OD pairs: "
            "beta = sum(observed * benchmark) / sum(observed squared) = 25/6.",
            "",
            "| OD | Observed | Benchmark | R adjusted | Python adjusted |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for i in range(len(observed)):
            report.append(
                f"| {observed.origin.iloc[i]}-{observed.destination.iloc[i]} | "
                f"{observed.flow.iloc[i]:g} | {benchmark.flow.iloc[i]:g} | "
                f"{expected.flow_adj.iloc[i]:.8f} | {result.flow_adj.iloc[i]:.8f} |"
            )
    report += [
        "",
        "## Figure",
        "",
        "![R/Python adjusted flows](issue-8-coefficient-r-parity.png)",
        "",
        "## Interpretation and review",
        "",
        "The fixtures are synthetic controls. They test calibration behaviour; "
        "independent empirical validation is required for accuracy claims.",
        "",
        "With `fit_intercept=True`, the R implementation records the intercept "
        "but applies the slope alone. Its documentation describes a flow-specific "
        "intercept correction. Python follows the current R outputs; this "
        "documentation discrepancy requires maintainer review.",
        "",
        "Global calibration uses the selected OD or margin keys. Repeated OD keys "
        "expand the fitting join, with a many-to-many warning when applicable; "
        "output observation rows are retained. Source-specific fitting uses "
        "`by_source=True`.",
        "",
        "NB uses statsmodels and ZINB optimises its statsmodels likelihood with "
        "SciPy BFGS in log-dispersion coordinates. Positive dispersion is preserved "
        "without a numerical floor. Coefficient/model tolerances are 1e-7 and "
        "1e-5 respectively for the audited fixtures. ZINB applies the "
        "count-component coefficient. Python uses explicit DataFrame/type "
        "validation and boolean arguments, and raises ValueError for failed "
        "count-model convergence. Backend warning text can differ.",
    ]
    differences = [r for r in rows if r["status"] == "DIFFERENCE"]
    if differences:
        report += ["", "## Differences requiring attention", ""]
        report += [f"- `{r['case']}`: {r['detail']}" for r in differences]
    (destination / "issue-8-coefficient-r-parity.md").write_text(
        "\n".join(report) + "\n"
    )
    print(
        f"{passed}/{len(rows)} parity cases matched; "
        "report: docs/reports/issue-8-coefficient-r-parity.md"
    )
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
