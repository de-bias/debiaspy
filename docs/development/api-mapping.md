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
| `R/adjust_inverse_penetration.R` | `adjust_inverse_penetration.py` | `adjust_inverse_penetration()` | Implemented on contribution branch |
| `R/adjust_selection_rate.R` | `adjust_selection_rate.py` | `adjust_selection_rate()` | Implemented on contribution branch |
| `R/adjust_selection_rate2.R` | `adjust_selection_rate2.py` | `adjust_selection_rate2()` | Implemented on contribution branch |
| `R/adjust_raking_ratio.R` | `adjust_raking_ratio.py` | `adjust_raking_ratio()` | Implemented on contribution branch |
| `R/adjust_coefficient.R` | `adjust_coefficient.py` | `adjust_coefficient()` | Implemented on contribution branch |
| `R/adjust_multilevel_bayes.R` | `adjust_multilevel_bayes.py` | `adjust_multilevel_bayes()` | Implementation proposed; backend review required |
| `R/adjust_all_methods.R` | `adjust_all_methods.py` | `adjust_all_methods()` | Implemented on contribution branch |
| `R/example-data.R` | `example_data.py` | `debiaspy_example_data()` | Implemented on contribution branch |
| `R/validate_flows.R` | `validate_flows.py` | `validate_flow_overall()` | Implemented on contribution branch |
| `R/validate_flows.R` | `validate_flows.py` | `validate_flow_pairs()` | Implemented on contribution branch |
| `R/validate_flows.R` | `validate_flows.py` | `validate_flow_distribution()` | Implemented on contribution branch |
| `R/validate_flows.R` | `validate_flows.py` | `validate_flow_residuals()` | Implemented on contribution branch |
| `R/validate_flows.R` | `validate_flows.py` | `validate_flow_residual_structure()` | Implemented on contribution branch |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_prepare_output()` | Implemented on contribution branch |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_prepare_outputs()` | Implemented on contribution branch |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_overall_methods()` | Implemented on contribution branch |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_residual_methods()` | Implemented on contribution branch |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_distribution_methods()` | Implemented on contribution branch |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_residual_structure_methods()` | Implemented on contribution branch |
| `R/validate_flow_methods.R` | `validate_flow_methods.py` | `validate_flow_margins()` | Implemented on contribution branch |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_metrics()` | Implemented on contribution branch |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_residuals()` | Implemented on contribution branch |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_scatter()` | Implemented on contribution branch |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_residual_bands()` | Implemented on contribution branch |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_distribution()` | Implemented on contribution branch |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_distribution_pairwise()` | Implemented on contribution branch |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_structure()` | Implemented on contribution branch |
| `R/plot_validation.R` | `plot_validation.py` | `validate_flow_plot_lisa_map()` | Implemented on contribution branch |

`debiaspy_example_data()` is the Python entry point corresponding to
`debiasR_example_data()`.

## Historical R aliases

`validate_flow_benchmark()` and `validate_flow_all()` are historical `debiasR`
aliases. The Python API exposes `validate_flow_overall()` and
`validate_flow_pairs()` as the corresponding entry points.

## Internal support modules

Shared calculations and input checks live in small private modules, including
`_adjustment_common.py`, `_flow_joins.py`, `_numeric.py` and `_schemas.py`.
Shared model inputs and formulas use `_multilevel_inputs.py`; frequentist
fitting uses `_frequentist.py`. Bayesian runtimes use `_bayesian.py`,
`_bayesian_latent.py` and `_latent.py`.
The Python formula interface uses `formula` and the paired mobility/bias inputs;
the deprecated R `custom_formula` argument is omitted.
The corresponding public functions remain in the mapped flat modules above.

## R files without a direct Python module requirement

- `R/debiasR-package.R` maps to package metadata in `pyproject.toml` and public
  exports in `src/debiaspy/__init__.py`.
- `R/globals.R` supports R package checks and has no direct Python equivalent.
- `R/data-simulated.R` documents bundled R datasets. Empirical Python datasets
  belong in `debiaspydata`, while small test fixtures belong in `tests/`.

See [method evidence](method-evidence.md) for the checked scope and
[contribution notes](contribution-format.md) for issue-specific review boundaries.

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
