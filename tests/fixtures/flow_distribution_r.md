# Flow-distribution R reference

`flow_distribution_r.json` stores inputs, arguments, results and conditions from
22 scenarios executed on 15 November 2026 using unchanged temporary copies of
`R/validate_flows.R`, `R/flow_comparisons.R` and `R/distribution_metrics.R` from
`debiasR` commit `6fa2709`. The reference used R 4.5, dplyr 1.2.1 and tibble
3.3.1. The original R checkout was read only; Python tests do not require R.

Scenarios cover all comparisons, ordered selection and deduplication, custom
columns, missing raw flows, union support, missing flows and OD keys, duplicate
joins, weighting, large smoothing, zeros, negative values, non-finite totals,
unused non-finite raw flows, empty inputs and output-selection flags. Four
scenarios store R errors; the others store the requested output tables.

To regenerate, decode JSON null as typed R NA and the `Infinity`/`-Infinity`
strings as Inf/-Inf. Construct input data.frames with character OD keys and
numeric flow columns (including typed empty vectors), source the three R
files in a temporary directory, and call `validate_flow_distribution()` with
each case's `options`. Save output tables with
`write.csv(..., row.names = FALSE, na = "NA")`. Capture errors with
`tryCatch` and warnings with `withCallingHandlers`; record whether warning
messages contain `many-to-many` or `NaNs produced`. Preserve original table
column and row order when converting the CSV values to JSON.

JSON null represents NA/NaN; signed infinity is encoded as a string. Tests use
relative/absolute numerical tolerance `1e-12` and compare table structure,
missingness and warning presence. Python uses native exception classes and
does not reproduce dplyr's aggregated warning counts or diagnostic formatting.
