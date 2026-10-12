# R-to-Python API mapping

This document records where the public functions in `debiasR` should be
implemented in `debiaspy`. Contributors should check this mapping before
starting a function so that related code is placed in the same module and two
people do not implement the same behaviour independently.

The mapping describes the intended public API. Private implementation helpers
do not need to be translated one by one from R. Add a private Python helper only
when it makes the implementation clearer, avoids duplication or supports more
than one public function.

## General rules

- Keep the public R function name when it is already a clear Python name.
- Use a leading underscore for private Python helpers.
- Define public functions in the mapped module and export them from
  `src/debiaspy/__init__.py` only when they are ready for users.
- Preserve the documented meaning, defaults, output names, errors and warnings
  from `debiasR` where they make sense in Python.
- Use pandas DataFrames for tabular inputs and outputs.
- Record and explain any deliberate difference from `debiasR` in the pull
  request.
- Do not carry R-only compatibility aliases or package-checking files into the
  Python API without a clear Python use case.

## Public function mapping

| `debiasR` source | `debiaspy` module | Public Python function | Implementation status |
|---|---|---|---|
| `R/measure_bias.R` | `measure_bias.py` | `measure_bias()` | Implemented |
| `R/measure_bias.R` | `measure_bias.py` | `measure_bias_distribution()` | Implemented |
| `R/measure_bias.R` | `measure_bias.py` | `validate_bias_residual_structure()` | Implemented |
| `R/adjust_inverse_penetration.R` | `adjust_inverse_penetration.py` | `adjust_inverse_penetration()` | Planned |
| `R/adjust_selection_rate.R` | `adjust_selection_rate.py` | `adjust_selection_rate()` | Planned |
| `R/adjust_selection_rate2.R` | `adjust_selection_rate2.py` | `adjust_selection_rate2()` | Planned |
| `R/adjust_raking_ratio.R` | `adjust_raking_ratio.py` | `adjust_raking_ratio()` | Planned |
| `R/adjust_coefficient.R` | `adjust_coefficient.py` | `adjust_coefficient()` | Planned |
| `R/adjust_multilevel_bayes.R` | `adjust_multilevel_bayes.py` | `adjust_multilevel_bayes()` | Design required |
| `R/adjust_all_methods.R` | `adjust_all_methods.py` | `adjust_all_methods()` | Planned after individual methods |
| `R/example-data.R` | `example_data.py` | `debiaspy_example_data()` | Planned |
| `R/validate_flows.R` | `validate_flows.py` | `validate_flow_overall()` | Planned |
| `R/validate_flows.R` | `validate_flows.py` | `validate_flow_pairs()` | Planned |
| `R/validate_flows.R` | `validate_flows.py` | `validate_flow_distribution()` | Planned |
| `R/validate_flows.R` | `validate_flows.py` | `validate_flow_residuals()` | Planned |
| `R/validate_flows.R` | `validate_flows.py` | `validate_flow_residual_structure()` | Planned |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_prepare_output()` | Planned |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_prepare_outputs()` | Planned |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_overall_methods()` | Planned |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_residual_methods()` | Planned |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_distribution_methods()` | Planned |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_residual_structure_methods()` | Planned |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_margins()` | Planned |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_metrics()` | Planned |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_residuals()` | Planned |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_scatter()` | Planned |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_residual_bands()` | Planned |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_distribution()` | Planned |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_distribution_pairwise()` | Planned |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_structure()` | Planned |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_lisa_map()` | Planned |

`debiasR_example_data()` is renamed `debiaspy_example_data()` because the R
package name should not appear in the Python API.

## Functions that will not be carried into the Python API

`validate_flow_benchmark()` and `validate_flow_all()` are compatibility aliases
for older `debiasR` names. `debiaspy` has no earlier API to support, so users
should call `validate_flow_overall()` and `validate_flow_pairs()` directly.

## Internal support modules

### `distribution_metrics.py`

This module may contain shared private calculations used by distributional
measurement and validation, including:

- `_validate_distribution_epsilon()`
- `_distribution_shares()`
- `_distribution_jsd_contributions()`
- `_normalise_distribution_comparisons()`

### `flow_comparisons.py`

This module may contain shared private definitions for comparing adjusted, raw
and benchmark flows, including:

- `_normalise_flow_comparisons()`
- `_flow_comparison_spec()`
- `_flow_comparison_label()`
- `_flow_comparison_residual()`
- `_flow_comparison_x()`
- `_flow_comparison_y()`

These private names describe intended responsibilities, not a requirement to
translate every R helper literally. Contributors should reuse an existing
helper rather than create competing implementations.

## R files without a direct Python module requirement

- `R/debiasR-package.R` maps to package metadata in `pyproject.toml` and public
  exports in `src/debiaspy/__init__.py`.
- `R/globals.R` supports R package checks and has no direct Python equivalent.
  The current `globals.py` placeholder can be removed unless a Python-specific
  use emerges.
- `R/data-simulated.R` documents bundled R datasets. Empirical Python datasets
  belong in `debiaspydata`, while small test fixtures belong in `tests/`. The
  current `data_simulated.py` placeholder can be removed unless simulation
  functions are designed later.

## Implementation order and dependencies

1. Complete the measurement functions in `measure_bias.py`.
2. Implement each deterministic adjustment function independently.
3. Implement `debiaspy_example_data()` for a consistent example-data route.
4. Implement `adjust_all_methods()` after the individual adjustment methods.
5. Implement overall, pair-level, distributional, residual and spatial flow
   validation.
6. Implement the helpers that compare several adjustment methods.
7. Implement validation plotting after the corresponding validation outputs
   are stable.
8. Agree the Python Bayesian backend and model-parity strategy before
   implementing `adjust_multilevel_bayes()`.
9. Develop the tutorials after the relevant public interfaces are stable.

## Requirements for every implementation issue

The contributor should:

- inspect the corresponding function and documentation in `debiasR`
- inspect the corresponding R tests under `debiasR/tests/testthat`
- implement the Python behaviour in the module specified above
- use normal Python, NumPy and pandas conventions rather than translate R syntax
  mechanically
- preserve documented argument meanings, output names, errors and warnings
  where appropriate
- export completed public functions from `src/debiaspy/__init__.py`
- add focused pytest tests for numerical results, required columns, output
  structure, missing values, zero values, errors and warnings as relevant
- include fixed examples that demonstrate agreement with `debiasR` where
  possible
- run `python -m pytest` and `python -m ruff check .`
- update user-facing package documentation when behaviour becomes available
- explain any deliberate difference from `debiasR` in the pull request
- keep the pull request focused on the assigned issue
- disclose AI assistance in the pull request and confirm that the named human
  contributor reviewed the code, tests and documentation

Tutorial writing is outside the initial implementation issues. It will begin
after the relevant functions and outputs are stable.
