# debiaspy

`debiaspy` is the planned Python counterpart to
[`debiasR`](https://github.com/de-bias/debiasR), a package for bias correction
and validation of origin-destination mobility data derived from mobile phone
sources.

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

The initial module names mirror the flat file layout used in `debiasR/R`.
These files are placeholders until the Python implementation begins:

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

The R package metadata file `debiasR-package.R` is not mirrored because its role
is specific to R package documentation.

## Planned tutorials

Tutorial placeholders are stored in `docs/tutorials`. Their filenames mirror
the current `debiasR` vignette sequence, using `t` for tutorial instead of `v`
for vignette:

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
