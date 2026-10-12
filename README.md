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

The population coverage measurement API is implemented. Adjustment methods and
flow-validation methods remain under development.

## Installation for development

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

Install the empirical data companion when you need to reproduce tutorials or
examples that use mobility, Census, covariate and coverage data:

```bash
python -m pip install git+https://github.com/de-bias/debiaspydata.git
```

If `debiaspy` and `debiaspydata` are cloned side by side under the same parent
folder, install the local data package in editable mode with:

```bash
python -m pip install -e ../debiaspydata
```

## Current measurement functions

The package currently provides three functions for measuring population
coverage bias and examining its structure:

- `measure_bias()` calculates coverage scores and coverage bias for each area
- `measure_bias_distribution()` compares population and active-user shares
  across areas
- `validate_bias_residual_structure()` examines departures from a constant
  global coverage rate and can add benchmark-flow, covariate and spatial
  diagnostics

```python
import pandas as pd
from debiaspy import (
    measure_bias,
    measure_bias_distribution,
    validate_bias_residual_structure,
)

coverage = pd.DataFrame(
    {
        "area": ["A", "B", "C"],
        "population": [1000, 800, 1200],
        "user_count": [100, 40, 180],
    }
)

area_bias = measure_bias(coverage)
distribution_bias = measure_bias_distribution(coverage, area_col="area")
residual_bias = validate_bias_residual_structure(
    coverage,
    coverage_area_col="area",
)
```

`measure_bias()` returns a DataFrame. The other two functions return
dictionaries containing summary and area-level DataFrames, along with any
optional diagnostics requested by the user.

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

## Module structure

The source files are organised around the main package tasks: measuring bias,
adjusting flows, validating outputs, plotting diagnostics and loading example
data. `measure_bias.py` is implemented; most other modules remain placeholders:

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

## Acknowledging contributors

We use the [All Contributors Bot](https://allcontributors.org/) to recognise
everyone’s work—code, docs, ideas, design and more.

After your PR is merged, comment on an issue or PR:

```text
@all-contributors please add @your-username for code, doc, etc.
```

Replace `@your-username` and the contribution types as appropriate. See the
[emoji key](https://allcontributors.org/en/reference/emoji-key/) for available
contribution types.

Thank you for helping us build open, collaborative and impactful projects with
DEBIAS.

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://c-cabrera.com/"><img src="https://avatars.githubusercontent.com/u/33100192?v=4?s=100" width="100px;" alt="Carmen Cabrera"/><br /><sub><b>Carmen Cabrera</b></sub></a><br /><a href="https://github.com/de-bias/debiaspy/commits?author=carmen-cabrera" title="Documentation">📖</a> <a href="https://github.com/de-bias/debiaspy/commits?author=carmen-cabrera" title="Code">💻</a> <a href="https://github.com/de-bias/debiaspy/issues?q=author%3Acarmen-cabrera" title="Bug reports">🐛</a> <a href="#content-carmen-cabrera" title="Content">🖋</a> <a href="#design-carmen-cabrera" title="Design">🎨</a> <a href="#example-carmen-cabrera" title="Examples">💡</a> <a href="#ideas-carmen-cabrera" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-carmen-cabrera" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-carmen-cabrera" title="Maintenance">🚧</a> <a href="#platform-carmen-cabrera" title="Packaging/porting to new platform">📦</a> <a href="#projectManagement-carmen-cabrera" title="Project Management">📆</a> <a href="#research-carmen-cabrera" title="Research">🔬</a> <a href="https://github.com/de-bias/debiaspy/pulls?q=is%3Apr+reviewed-by%3Acarmen-cabrera" title="Reviewed Pull Requests">👀</a> <a href="#tool-carmen-cabrera" title="Tools">🔧</a> <a href="https://github.com/de-bias/debiaspy/commits?author=carmen-cabrera" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://franciscorowe.com"><img src="https://avatars.githubusercontent.com/u/28450210?v=4?s=100" width="100px;" alt="Francisco Rowe"/><br /><sub><b>Francisco Rowe</b></sub></a><br /><a href="https://github.com/de-bias/debiaspy/commits?author=fcorowe" title="Documentation">📖</a> <a href="https://github.com/de-bias/debiaspy/commits?author=fcorowe" title="Code">💻</a> <a href="https://github.com/de-bias/debiaspy/issues?q=author%3Afcorowe" title="Bug reports">🐛</a> <a href="#content-fcorowe" title="Content">🖋</a> <a href="#design-fcorowe" title="Design">🎨</a> <a href="#example-fcorowe" title="Examples">💡</a> <a href="#ideas-fcorowe" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-fcorowe" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-fcorowe" title="Maintenance">🚧</a> <a href="#platform-fcorowe" title="Packaging/porting to new platform">📦</a> <a href="#projectManagement-fcorowe" title="Project Management">📆</a> <a href="#research-fcorowe" title="Research">🔬</a> <a href="https://github.com/de-bias/debiaspy/pulls?q=is%3Apr+reviewed-by%3Afcorowe" title="Reviewed Pull Requests">👀</a> <a href="#tool-fcorowe" title="Tools">🔧</a> <a href="https://github.com/de-bias/debiaspy/commits?author=fcorowe" title="Tests">⚠️</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the
[all-contributors](https://github.com/all-contributors/all-contributors)
specification. Contributions of any kind are welcome.
