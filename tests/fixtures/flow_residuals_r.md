# Residual and spatial R references

`flow_residuals_r.json` records 22 cases executed on 15 November 2026 with
unchanged temporary copies of `debiasR/R/validate_flows.R` and
`debiasR/R/flow_comparisons.R` at commit `6fa2709`. The original checkout was
read only. The reference used R 4.5, dplyr 1.2.1 and tibble 3.3.1, with default
Mersenne-Twister and Rejection RNG settings. Python tests do not require R.

Cases include residual signs and improvement summaries, zeros, missing and
infinite flows, constant and empty inputs, stable ties and fractional top_n,
custom columns, duplicate joins, comparisons and spatial roles, both
aggregations, covariate/geometry joins, missing area keys, weighted/self/
duplicate links, islands, missing links and seeded Local Moran/LISA outputs.

Each case stores `target`, `adj`, `benchmark`, `options`, expected tables and
warning-presence flags. JSON null represents NA/NaN; `Infinity` and
`-Infinity` represent infinite numeric values. Optional tables are dictionaries
in options. To regenerate, create R data.frames with character identifiers and
numeric measurements, including correctly typed empty vectors; source the two
reference files in a temporary directory; call `validate_flow_residuals()` or
`validate_flow_residual_structure()` using the stored options. Write tables
with `write.csv(..., row.names = FALSE, na = "NA")`, preserving row/column order.
Capture many-to-many and scalar replacement warnings with `withCallingHandlers`.

`flow_residual_p_adjust_r.json` stores `stats::p.adjust()` results for all eight
method names applied to `[.01, .04, .03, .001, .5, .9, .04]`. Fixed permutations
in the pytest module come from `set.seed(seed)` followed by three consecutive
`sample(c(-1, 0, 1, 2, 3))` calls for seeds 0, 123, -4 and 2026.

The RNG adapter uses NumPy's MT19937 implementation with R's seed expansion,
16-bit rejection draws and sampling order. Algorithm references:
[R RNG.c](https://github.com/wch/r-source/blob/R-4-5-branch/src/main/RNG.c),
[R random.c](https://github.com/wch/r-source/blob/R-4-5-branch/src/main/random.c),
and [p.adjust.R](https://github.com/wch/r-source/blob/R-4-5-branch/src/library/stats/R/p.adjust.R).
Only default R >= 3.6 RNG settings are supported. Fresh-entropy calls do not
claim to share R's global state. A one-element positive sampling vector keeps
R's range-sampling behaviour; when that yields multiple values, only the first
replacement is used and a RuntimeWarning is emitted.

Tests compare numbers at relative/absolute tolerance `1e-12`, plus exact
column/row order, missingness, classifications and warning presence. Nullable
Python logicals and R NA are normalized only for the cross-language comparison.
The normal and undefined Local Moran paths intentionally preserve their
slightly different R column orders; map-data column order is fixed.
