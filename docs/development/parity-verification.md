# Package verification

Branch: `contribution/non-bayesian-parity`.
Reference main: `56c188a172dd71703f23973b88e22cf3c742d36b`.
R reference: `6fa2709ae42de6dc525a4a46182fdfb448b14a3f`.
Python companion: `19d1c5f67f5dcf70fcd765e42fa03d667df496df`.

## Results

- Current separated package: 292 tests passed with live R; Bayesian prediction/
  backend and latent test modules were excluded. All 204 original test functions
  and their assertion bodies are preserved in focused issue files.
- Eleven [issue contributions](issue-contributions.md) passed installed-wheel,
  import, Ruff, formatting and dependency checks on their documented bases.
- Previous complete installed Python 3.11 wheel on macOS ARM64: 342 tests passed with live R and
  the short Bambi/PyMC execution check enabled; three Bambi/Pandas warnings.
- Fresh article comparison: 149 of 151 passed at the existing tolerances;
  two public-export differences are explained in the method evidence table.
- Installed Census example: 42,455 observed/adjusted rows, zero undefined
  adjustments and 34,762 matched benchmark pairs.
- Ruff, formatting, whitespace and dependency checks passed.
- Earlier fresh Python 3.14 base/geographic check: 114 passed, one optional R
  comparison skipped. Quarto and non-sampling tutorial snippets were checked
  during the preceding package verification.

The [method evidence table](method-evidence.md) identifies each public method's
sources, tests and scope. The [full report and reproduction commands](../reports/README.md)
provide the Census tables, figures and reference comparisons.

## Implementation and review

All methods and original tests are retained. Adjustment, validation and portable
example tests are separated by scope and share their original fixture values.
The adjustment wrapper loads the multilevel method when that method is selected. Shared spatial calculations preserve the distinct input normalization
rules; common model attributes preserve priors, diagnostics and attribution.
The latest runtime simplification removes 91 lines and the article checker removes
103 lines. The public multilevel module has 524 lines. Scientific tolerances and
source/time identities are preserved.

[Issue-specific contribution notes](contribution-format.md) identify the proposed
scope and dependencies for #15 through #7 and grouped #4–#6. Human review covers
code, tests and documentation under the existing contribution workflow.

## Evidence limits

Census calibration uses benchmark margins; independent evaluation data are needed
for accuracy claims. Python spatial permutations reproduce within Python.
The earlier Bayesian investigation and its outstanding quality checks remain
recorded in [Bayesian investigation](bayesian-investigation.md). Bayesian acceptance,
full CmdStan sampling and the wider model/platform matrix remain pending.

The runtime accepts pandas DataFrames. A future Polars evaluation will compare
memory, runtime, joins and numerical results. Software retains the single MIT
licence; external datasets retain their attribution and licence terms.
