# debiaspy

Measure population representation bias, adjust origin-destination flows, and
validate the results against benchmark data using pandas DataFrames.
The methods follow [debiasR](https://github.com/de-bias/debiasR).

## Install and run

```bash
git switch contribution/r-parity
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python examples/coverage_adjustment.py
```

The example uses synthetic coverage and flows. It calls the public measurement,
inverse-penetration and validation functions and prints adjusted flows and
comparison metrics. Its constructed benchmark illustrates the calculation.
See [input definitions](docs/inputs.md) for the required fields and units.

## Methods

- Measurement: coverage scores, distribution bias and residual diagnostics.
- Adjustment: inverse penetration, raking, coefficients, selection rates and
  multilevel models; `adjust_all_methods()` runs a selected set.
- Validation: paired flows, summary metrics, margins, distributions, residuals,
  spatial diagnostics and plots.

`debiaspy_example_data()` reads the Python data companion. Install it to run
the original mobile-phone/Census 2021 OD workplace-flow example:

```bash
python -m pip install git+https://github.com/de-bias/debiaspydata.git@19d1c5f67f5dcf70fcd765e42fa03d667df496df
python examples/benchmark_od.py --output output/census-od
```

Use the same public functions with other compatible data:

```bash
python examples/benchmark_od.py --observed observed.csv --benchmark benchmark.csv \
  --coverage coverage.csv --output output/my-od
```

Column-name options support other schemas; area identifiers are unrestricted
in this caller-supplied route. The [input guide](docs/inputs.md) explains units,
denominators and matching requirements. A missing or infinite adjusted flow
is retained in the output, and metrics are withheld until it is resolved.
The Census loader reads named datasets through the Python companion API. Explicit OD paths still require companion
covariate and centroid assets for this Census-specific loader.

The base installation supports deterministic adjustment, fixed-effect count
models and validation. Install optional runtimes as needed:

```bash
python -m pip install -e ".[grouped]"   # frequentist random intercepts
python -m pip install -e ".[bayesian]"  # Bambi/PyMC models
python -m pip install -e ".[latent]"    # CmdStanPy latent models
python -m pip install -e ".[plot]"    # geographic plots
```

Latent Bayesian models additionally require CmdStan 2.32.2:

```bash
debiaspy-install-cmdstan --version 2.32.2 --directory .cmdstan
export CMDSTAN="$PWD/.cmdstan/cmdstan-2.32.2"
```

Random slopes, multiple/nested/crossed random effects and latent ZIP/ZINB remain
unsupported. Missing counts and zero-user inverse weights can produce missing
or infinite results, following the corresponding R arithmetic.

Bayesian methods are experimental on this integration branch. The checked
Poisson case matches R priors and shared-draw predictions, but its extrapolated
posterior means remain unreliable. Review the [Bayesian tutorial](docs/tutorials/t09-advanced-bayesian-adjustment.qmd)
and diagnostics before interpreting corrected flows.

## Development and parity

Follow the [contribution rules](CONTRIBUTING.md),
[implementation workflow](docs/development/workflow.md) and
[API mapping](docs/development/api-mapping.md). Development is through focused
branches and reviewed PRs. Tutorials use the existing Quarto website.

```bash
python -m pytest
python -m ruff check .
```

Set `DEBIASR_ROOT` to run the optional live-R regression tests. The adjustment
comparisons use debiasR commit `6fa2709ae42de6dc525a4a46182fdfb448b14a3f`.
The [checked side-by-side PDF](docs/reports/debiasr_debiaspy_side_by_side.pdf)
uses the original Census OD example and explains portable inputs and Bayesian
limits. The report source and commands are in [report reproduction](docs/reports/README.md).
The [integration review](docs/development/integration-review.md) records the
branch's checks, intentional API choices and outstanding model review.

We plan to investigate Polars for loading, joining and aggregating larger OD
and benchmark tables. The evaluation will compare runtime and memory use on
the Census example and other compatible datasets, while checking identifiers,
missing values, duplicate joins and numerical parity with the current pandas
implementation. Polars is not yet a dependency; the public API continues to
accept pandas DataFrames.

## Licence

Software is MIT-licensed; see [LICENSE](LICENSE). External datasets retain
their own licence and attribution requirements. Original contributor records
are retained below.

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
