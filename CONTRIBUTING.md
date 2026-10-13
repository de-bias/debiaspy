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

## Project guidance and status

Read the [Project Log and Status](docs/development/PROJECT_LOG_STATUS.md) before
starting work. It provides a concise summary of what is currently available on
`main`, the main development priorities, important decisions and known
limitations. Detailed function locations and implementation plans remain in
the [R-to-Python API mapping](docs/development/api-mapping.md), while individual
tasks and acceptance criteria remain in GitHub issues.

If you use a coding agent, ask it to read and follow [AGENTS.md](AGENTS.md)
before making changes. Some agents load this file automatically, but the named
human contributor remains responsible for checking the resulting code, tests
and documentation.

Every pull request must review the project log and update it where necessary.
Make any required update in the same branch and pull request as the work that
changes project status. Small changes that do not affect the recorded status do
not need to edit the file, but contributors must still confirm that they have
reviewed it in the pull request checklist.

## Local setup

Clone the repository and move into it:

```bash
git clone https://github.com/de-bias/debiaspy.git
cd debiaspy
```

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if it is
not already available, then prepare the development environment:

```bash
uv sync
```

This uses the shared Python version recorded in `.python-version`, creates a
local `.venv/` and installs the package and current development tools. You do
not need to activate `.venv`: use `uv run` for project commands.

Confirm that Python imports the current checkout:

```bash
uv run python -c "import debiaspy; print(debiaspy.__file__)"
```

Run `uv sync` again after pulling a change to `pyproject.toml` or `uv.lock`.
The committed lock file gives contributors consistent dependency versions;
do not commit `.venv/`.

## Development workflow

Create a branch for each change:

```bash
git switch -c issue-4-adjust-inverse-penetration
```

Use a short, descriptive branch name based on the issue or task. Where useful,
prefix it with the human contributor's name or GitHub username. Agent-specific
prefixes are not required.

Make focused changes. Prefer small pull requests that implement one function,
one documentation update or one test improvement at a time.

## Running checks

Before opening a pull request, run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

If formatting changes are needed, run:

```bash
uv run ruff format .
```

The test suite checks package imports, numerical results, input validation and
the behaviour of implemented public functions. Each new public function should
receive similarly focused tests.

## Dependencies

Add a dependency only when implemented functionality requires it, and explain
the reason in the relevant pull request. Use `uv add PACKAGE` for a package
needed by normal users and `uv add --dev PACKAGE` for a development-only tool.
Use an optional dependency group for a specialised feature only after that
group has been agreed.

Commit both `pyproject.toml` and the updated `uv.lock`. Contributors should not
edit `uv.lock` manually or choose their own dependency versions. Bayesian,
spatial and other specialised packages will be added later if and when the
corresponding functionality is implemented.

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
- run `uv sync`
- run `uv run pytest`
- run `uv run ruff check .`
- run `uv run ruff format --check .`
- update documentation if the user-facing behaviour changed
- review `docs/development/PROJECT_LOG_STATUS.md` and update it where necessary
- link any related issue

Use the pull request template in `.github/pull_request_template.md`.

## Acknowledging contributors

We aim to recognise code, documentation, ideas, testing and review
contributions. After a PR is merged, contributors can ask to be acknowledged in
the project contributor records if the repository enables contributor tooling.
