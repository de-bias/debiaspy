# Contributing guidelines

Thank you for your interest in contributing to `debiaspy`. The package provides
Python tools for assessing and correcting population representation bias in
digital trace data. It is under active development, so contributions should
focus on careful package structure, tests, documentation and well-scoped
implementations of the planned methods.

## How to contribute

All changes to `main` should come through pull requests. Direct pushes to
`main` are not part of the project workflow, including for maintainers and
automation agents.

Carmen Cabrera (`carmen-cabrera`) and Francisco Rowe (`fcorowe`) can review and
merge accepted pull requests.

Pull requests should wait for an approving review from an eligible reviewer
before merge.

## Local setup

Clone the repository and move into it:

```bash
git clone https://github.com/de-bias/debiaspy.git
cd debiaspy
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package in editable development mode:

```bash
python -m pip install -e ".[dev]"
```

Editable mode means changes under `src/debiaspy/` are picked up without
reinstalling the package each time.

## Development workflow

Create a branch for each change:

```bash
git checkout -b codex/my-change
```

Use a short, descriptive branch name. The `codex/` prefix is a useful default
for local automation-assisted work, but human contributors can use any clear
branch name.

Make focused changes. Prefer small pull requests that implement one function,
one documentation update or one test improvement at a time.

## Running checks

Before opening a pull request, run:

```bash
python -m pytest
python -m ruff check .
```

If formatting changes are needed, run:

```bash
python -m ruff format .
```

The test suite checks package imports, numerical results, input validation and
the behaviour of implemented public functions. Each new public function should
receive similarly focused tests.

## Naming guide

`debiaspy` should mirror `debiasR` where this produces a clear Python API.
The [R-to-Python API mapping](docs/development/api-mapping.md) records the
agreed module and public-function locations. Check it before starting an
implementation.

- Keep public function names close to the `debiasR` names, for example
  `adjust_inverse_penetration`, `measure_bias` and `validate_flow_overall`.
- Keep source files aligned with the flat `debiasR/R` layout unless there is a
  clear reason to reorganise.
- Use Python file naming conventions, replacing hyphens with underscores. For
  example, `debiasR/R/example-data.R` is mirrored as `example_data.py`.
- Prefer clear pandas DataFrame inputs and outputs.

## Tests

When porting a function from `debiasR`, inspect the corresponding R tests and
add Python tests that cover:

- required input columns
- output column names
- basic numerical results
- missing data
- zero values and other edge cases
- errors and warnings

Where possible, tests should demonstrate that `debiaspy` matches the behaviour
of `debiasR`.

## Documentation

Tutorials live in `docs/tutorials/` and use Quarto (`.qmd`). Their filenames
mirror the `debiasR` vignette sequence with `t` for tutorial instead of `v` for
vignette.

When a public function becomes usable, update the relevant tutorial placeholder
or README section.

Write tutorials for applied researchers, students and policy readers. Prefer
clear explanations over internal implementation details.

## Pull requests

Before opening a PR:

- check that the branch is up to date with `main`
- run `python -m pytest`
- run `python -m ruff check .`
- update documentation if the user-facing behaviour changed
- link any related issue

Use the pull request template in `.github/pull_request_template.md`.

## Acknowledging contributors

We aim to recognise code, documentation, ideas, testing and review
contributions. After a PR is merged, contributors can ask to be acknowledged in
the project contributor records if the repository enables contributor tooling.
