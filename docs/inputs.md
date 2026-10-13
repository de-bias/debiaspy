# Inputs and outputs

Coverage tables contain an area identifier, `population` (benchmark count), and
`user_count` (observed count). OD tables contain `origin`, `destination` and a
flow-count column. Adjustment returns `flow_adj` alongside the observed flow.
Benchmark OD tables use the same geographic keys and counting unit.

Counts must describe compatible populations or flow margins and time periods.
A stock population denominator and a commuting-flow margin have different
interpretations. Choose the coverage definition required by the method and
validate the corrected flows against an appropriate benchmark.

`mpd_source`, time identifiers and method-specific grouping fields distinguish
repeated observations. Covariates use area keys; distances use OD keys and
kilometres. Geographic tables must have matching area identifiers and an
appropriate CRS. Consult each public function's docstring for its columns.

Inverse penetration preserves missing/infinite R arithmetic. Zero observed
users cannot support a finite inverse estimate. Conflicting coverage mappings
can expand join rows; many-to-many joins warn. Selection-rate calibration
follows the tested R OD/origin aggregation and missing-value rules.

The example uses synthetic counts and a constructed benchmark. Empirical
example loading uses debiaspydata, including the original Census 2021 ODWP01EW
benchmark. The loader uses the Python companion; the legacy R-loading/download route
has been removed. Explicit OD paths still require companion covariate and
centroid assets. Mobile-location data originate from Zhong Chen (2024),
[Anonymised human location data for urban mobility research](https://doi.org/10.5281/zenodo.13327082),
under CC BY 4.0. Census workplace data and area lookups originate from ONS under
the [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
These external data licences apply to the data; the package code is MIT.

## Applying the workflow to another benchmark

Supply observed OD, benchmark OD and coverage CSVs to
`examples/benchmark_od.py`. The observed/benchmark flow column names and the
origin, destination and coverage-area columns are configurable. CSV area
identifiers are read as strings to preserve leading zeros. Use comparable
geographic units, periods, observation units and coverage populations. Preserve
source/time identifiers where observations repeat. Algorithms do not require
Census or LAD identifiers.

The empirical default uses all overlapping LAD areas. Its coverage is derived
from OD margins, as in the R example loader, rather than using the companion's
stock-population coverage table. As the Census margins also inform weighting,
the resulting benchmark comparison is a worked calculation rather than an
independent estimate of out-of-sample accuracy. Supply independently justified
coverage and benchmark data when that is the intended assessment.

Validation metrics use matched OD pairs. The metric table reports their count;
interpret it alongside the observed and adjusted row counts. The worked
runner retains zero flows and withholds metrics when adjusted values are
missing or infinite. Structural missing cells are not automatically zero-filled.

For Bayesian adjustment, inspect the uncertainty of the adjusted-flow
functional as well as parameter diagnostics. Reduced-form bias removal can
extrapolate far beyond the observed coverage range. A converged coefficient
posterior can still produce a heavy-tailed, poorly estimated adjusted-flow
mean; use reported total-flow MCSE and its ratio to the mean when assessing
precision. Changing the posterior summary or model requires a scientific
choice, not a silent cleanup adjustment.
