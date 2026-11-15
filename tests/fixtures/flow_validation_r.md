# Overall and pair-level R reference

`flow_validation_r.json` contains 22 input scenarios and results generated on
15 November 2026 from unchanged temporary copies of
`debiasR/R/validate_flows.R` and `debiasR/R/flow_comparisons.R`, commit `6fa2709`.
The original checkout was read only. The reference used R 4.5, dplyr 1.2.1 and
tibble 3.3.1; Python tests do not require R at runtime.

The scenarios cover all comparisons, comparison order and deduplication, custom
flow columns, optional raw flows, source groups, missing keys and pairs,
duplicate joins, zeros, negatives, missing and infinite flows, empty and
singleton samples, constant series, tied ranks, perfect fits and a predictor
nearly collinear with the intercept.

Each case stores `adj`, `benchmark`, `options`, expected tables and warning
counts. JSON null represents missing values/NaN; `Infinity` and `-Infinity`
encode signed infinity. The expected `summary` is the R summary table or the
single-comparison scalar result converted to one row. `data`, `by_source` and
`pairs` retain the original column and row order. The no-raw scenario has no
pair-level result, because that function requires raw flows.

To regenerate, write each input table as CSV, decode non-finite values, and read
with `read.csv(..., stringsAsFactors = FALSE, na.strings = c("", "NA"))`.
Source both reference files into a temporary R session, call
`validate_flow_overall(..., method_name = "parity")` with each case's options,
and call `validate_flow_pairs()` with its flow-column options when raw data are
available. Write result tables using
`write.csv(..., row.names = FALSE, na = "NA")`. Count warnings containing
`many-to-many` and `essentially perfect fit` via `withCallingHandlers`.

Numeric comparisons use relative/absolute tolerance `1e-12`. For the
`rank_deficient` scenario only, `ols_slope` and `r_squared` use absolute tolerance
`5e-8`: one comparison regresses a response near `1e8` with changes of only 1,
so R's uncentered QR and centered NumPy calculations differ at approximately
`1e-8`. All other fields in that scenario retain the tighter tolerance.
