# debiaspy

`debiaspy` is a Python package for assessing, adjusting and validating
population representation bias in digital trace data. It is designed for
spatio-temporally aggregated data that provide population counts by location and
flows between locations.

The package workflow focuses on three linked tasks:

1. measuring coverage and representativeness bias
2. adjusting biased origin-destination flows
3. validating adjusted flows against benchmark data

Mobile-phone-derived mobility data are the main motivating example, but the
same logic can apply to other location-to-location flow data with comparable
spatial and temporal aggregation and a validation target. Examples include trade
of goods, Internet traffic, supply chains and other forms of interaction between
places.

This repository currently contains the package skeleton only. The core
adjustment, validation and measurement methods have not yet been implemented.

## Installation for development

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

## Development checks

Run the test suite:

```bash
python -m pytest
```

Run linting:

```bash
python -m ruff check .
```

Format code:

```bash
python -m ruff format .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/development/workflow.md](docs/development/workflow.md) for the contributor
workflow.

## Planned module structure

The initial source files are organised around the main package tasks: measuring
bias, adjusting flows, validating outputs, plotting diagnostics and loading
example data. These files are placeholders until the Python implementation
begins:

- `adjust_all_methods.py`
- `adjust_coefficient.py`
- `adjust_inverse_penetration.py`
- `adjust_multilevel_bayes.py`
- `adjust_raking_ratio.py`
- `adjust_selection_rate.py`
- `adjust_selection_rate2.py`
- `data_simulated.py`
- `distribution_metrics.py`
- `example_data.py`
- `flow_comparisons.py`
- `globals.py`
- `measure_bias.py`
- `plot_validation.py`
- `validate_flow_methods.py`
- `validate_flows.py`

This flat layout keeps the package easy to inspect during the first stage of
development. It can be reorganised later if the Python implementation needs a
larger internal structure.

## Planned tutorials

Tutorial placeholders are stored in `docs/tutorials`. The sequence is organised
around the package workflow:

- `t02-why-this-matters.qmd`
- `t03-getting-set-up.qmd`
- `t04-measuring-coverage-bias.qmd`
- `t05-identifying-and-explaining-bias.qmd`
- `t06-adjusting-biases.qmd`
- `t07-validation.qmd`
- `t08-visualising-flows.qmd`
- `t09-advanced-bayesian-adjustment.qmd`

## Tutorial website

The tutorial placeholders are configured as a Quarto website. To preview the
site locally:

```bash
quarto preview
```

To render the static site:

```bash
quarto render
```

The rendered site is written to `_site/`, which is ignored by git.
