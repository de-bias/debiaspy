# Inverse-penetration R reference

`inverse_penetration_r.json` contains nine input scenarios, each with expected
outputs for `origin`, `destination` and `both` weighting. Expected tables and
warning counts were generated on 14 November 2026 by executing an unchanged copy
of `debiasR/R/adjust_inverse_penetration.R` from commit `6fa2709` in a temporary
directory. The original R checkout was read only. Execution used R 4.5 with
dplyr 1.2.1 and tibble 3.3.1.

The scenarios cover asymmetric weights, custom flow names, source-specific
coverage including missing source IDs, a source column present on only one
input, identical and conflicting duplicate coverage, different coverage records
with equal ratios, missing areas/counts/flows, and zero/negative/infinite values.

Each scenario stores `mpd`, `coverage`, optional `flow_col`, and `expected`.
Expected results preserve column order and row order. `many_to_many` and
`nan_warnings` record R warning counts. JSON null represents missing values or
NaN; `"Infinity"` and `"-Infinity"` encode signed infinities. Pandas-specific
dtypes are not asserted against R CSV storage types.

To regenerate, decode the input tables, write them to CSV, read them with
`read.csv(..., stringsAsFactors = FALSE, na.strings = c("", "NA"))`, source a
temporary copy of the reference function and call it for all three modes.
Use `withCallingHandlers` to count warnings containing `many-to-many` and
`NaNs produced`, and write each returned table with
`write.csv(..., row.names = FALSE, na = "NA")`. Convert those tables back to the
stored column/data arrays and encode non-finite values as above. Tests compare
the values with relative and absolute tolerance `1e-14`, and compare column
order, row order and warning counts exactly.
