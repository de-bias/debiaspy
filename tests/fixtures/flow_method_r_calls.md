# R wrapper call fixture

`flow_method_r_calls.csv` records default arguments forwarded by the four
multi-method wrappers in `debiasR/R/validate_flow_methods.R`, commit `6fa2709`.
It was generated on 14 November 2026 using R 4.5, from a temporary copy of the
unchanged source. The original repository was read only.

To reproduce the capture, source that file into a separate R environment and
replace these four single-output functions in that environment with recording
functions: `validate_flow_overall`, `validate_flow_residuals`,
`validate_flow_distribution` and `validate_flow_residual_structure`.
The recording functions accept `adj_df`, `benchmark_od_df`, `method_name` and
`...`, and emit one row for each argument in `...`, with its value converted by
`as.character()`. Invoke each wrapper with a named list ordered `method_b`,
`method_a` and a shared benchmark, leaving all optional arguments at defaults.
Export the rows using `write.csv(..., row.names = FALSE)`.

The fixture verifies dispatch arguments, method IDs and defaults. It does not
contain numerical validation results. Python core validators are still pending;
recording test doubles are used solely to check wrapper orchestration.
