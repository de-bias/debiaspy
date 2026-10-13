# Reduced-form Bayesian investigation

R code: `adjust_multilevel_bayes.R`, `check_bayesian_parity.R`.
Python code: `_bayesian.py`, `adjust_multilevel_bayes.py`, `check_bayesian_parity.py`.
Tests: `test_bayesian_prediction_r_parity.py`, `test_bayesian_backend_contract.py`.

The five-area Census Poisson example has a counterfactual estimation problem,
not a demonstrated difference in the Python translation. This conclusion is
limited to the fixed-effect, reduced-form, rstanarm-compatible case investigated
here. It does not establish parity for every Bayesian family or random-effect
structure. The six retained R-reference ZIP quality failures remain unresolved.

## What was checked

The R checkout remains pinned at
`6fa2709ae42de6dc525a4a46182fdfb448b14a3f` and unmodified. Both implementations
use the same 25 fitting rows, outcome, transformed predictors and formula.
Actual resolved Normal prior locations and scales match: centred intercept
scale 2.5; slope scales `2.5 / sample_sd`. In particular the bias slope's scale
is approximately 985.16, because bias varies very little in this subset.

The check evaluates the centred joint log prior and Poisson log likelihood at
identical R coefficient draws. It then passes those same draws through the
actual R fixed-effect prediction helper and Python/Bambi prediction helper,
both for observed data and after setting bias to zero. Prior log densities
agree within approximately `1.5e-13`, log likelihoods within `5.8e-11`, and
predicted rates within `1.6e-13` relative error. This rules out differences in
these prior parameters, centring, link, fixed-effect prediction and
counterfactual zeroing as the explanation for this case.

## Why the means disagree

Observed bias ranges from 0.9644 to 0.9714; zero lies approximately 381 sample
standard deviations from its mean. Removing a fitted bias coefficient near
-30 therefore multiplies predicted flows by roughly `exp(29)`. Small
uncertainty in that coefficient becomes very large uncertainty on the flow
scale. The observed total is 21,488; even the median of R's counterfactual
total draws is approximately `1.15e17`. This is a large extrapolation as well as
an imprecise Monte Carlo estimate of its mean.

| Fit | Retained draws per backend | R total mean MCSE/mean | Python total mean MCSE/mean |
| --- | ---: | ---: | ---: |
| Seed 20260710; 2,000 iterations per chain | 4,000 | 96.46% | 33.43% |
| Seed 20260711; 8,000 iterations per chain | 16,000 | 21.55% | 33.42% |

Both runs use four chains, with half of each chain's iterations used for
warmup. The second run changes both seed and draw count, so it demonstrates
sensitivity rather than isolating the effect of increasing draws. Shared-draw
prior/prediction checks pass in both runs. Independent flow-mean differences
remain large. In the first R run, one draw supplies 74.8% of the sum used to
estimate the mean. Rank-based R-hat and effective sample size can look healthy
while missing this lack of mean precision. Posterior medians differ less
(maximum pair difference 20.1%, then 8.7%); they are a different summary and
have not been substituted for the R default.

## What changed

The posterior, priors, counterfactual, sampler and mean estimator are unchanged.
Python now records observed bias ranges and whether zero is outside them,
total-mean precision status, and the largest draw's share of the sum. When a
requested mean has total-flow MCSE exceeding 5% of its magnitude it emits a
`RuntimeWarning`. That 5% threshold is an explicit Python precision warning,
not an R default or a new Bayesian parity acceptance threshold. Its absence
does not certify model validity, individual flows or convergence. Median
summaries are not assessed by this mean-precision warning.

No clipping, stronger prior, default-summary switch, seed selection or
acceptance-threshold relaxation was used to make the comparison pass.
The model interpretation and unsupported counterfactual require maintainer
review before this example can support a claim of reliable corrected flows.
Further sampling alone has not resolved the problem in the tested runs.

Historical investigation verification: 288 tests passed, one optional sampling test skipped; Ruff,
formatting and whitespace checks passed. The new live-R prediction regression
passed, and both empirical R/Python sampling runs completed. The full Bayesian
family/random-effect matrix was not rerun.

## Reproduce the investigation

Install the package's `bayesian` extra and the R reference dependencies
(including rstanarm, pkgload and jsonlite). The default input route additionally
requires the Python data companion and its empirical assets.

```bash
python tools/check_bayesian_parity.py \
  --r-root /path/to/debiasR --output /tmp/reduced-form-check
python tools/check_bayesian_parity.py \
  --r-root /path/to/debiasR --input-dir /tmp/reduced-form-check \
  --output /tmp/reduced-form-longer --iterations 8000 --seed 20260711
DEBIASR_ROOT=/path/to/debiasR python -m pytest \
  tests/test_bayesian_prediction_r_parity.py \
  tests/test_bayesian_backend_contract.py
```

For another dataset, supply `--input-dir` containing `mpd_od.csv`,
`coverage.csv`, `covariates.csv` and `distance.csv` in the documented package
schemas. This checker deliberately fixes the model above to isolate this
issue; it is not an exhaustive Bayesian certification suite. It writes raw
fits/draws and a compact `investigation.json` to the chosen output directory.
`--reuse-fits` rechecks saved fits without rerunning sampling. The repository
retains only compact evidence in `docs/reports/bayesian_checks.json`; empirical
input files and posterior draws are not imported into the distribution.

Human review and maintainer agreement on the Bayesian backend and model-parity
strategy remain required by the contribution workflow.
