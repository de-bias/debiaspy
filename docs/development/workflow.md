# Development workflow

This note records the basic local workflow for developing `debiaspy`.

Before starting substantial work, read the
[Project Log and Status](PROJECT_LOG_STATUS.md) for the current state of `main`
and the [R-to-Python API mapping](api-mapping.md) for agreed function locations
and implementation order.

## One-time setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run
this command from the repository root:

```bash
uv sync
```

`uv` reads `.python-version`, `pyproject.toml` and `uv.lock`. It obtains the
shared Python version when necessary, creates `.venv/` and installs the package
and development tools. The environment does not need to be activated because
`uv run` uses it automatically.

Run `uv sync` after pulling changes to `pyproject.toml` or `uv.lock`. Commit
`uv.lock`, but never commit `.venv/`.

## Check that the package imports

```bash
uv run python -c "import debiaspy; print(debiaspy.__version__)"
```

Expected output:

```text
0.0.0
```

## Run tests

```bash
uv run pytest
```

## Run linting

```bash
uv run ruff check .
```

## Format code

```bash
uv run ruff format --check .
```

Run `uv run ruff format .` only when formatting changes are needed.

## Adding a dependency

Do not add packages in anticipation of possible future work. Add one only when
implemented functionality requires it and explain the reason in the pull
request.

- Use `uv add PACKAGE` for a dependency required by normal package use.
- Use `uv add --dev PACKAGE` for a development-only tool.
- Use an agreed optional dependency group for a specialised feature.

These commands update both `pyproject.toml` and `uv.lock`; commit both files.
Do not edit the generated lock file manually. Other contributors receive the
new dependency by pulling the change and running `uv sync`.

## Implementing a function

When porting a function from `debiasR`:

1. Check the [R-to-Python API mapping](api-mapping.md) and confirm that nobody
   else is working on the same function.
2. Identify the source R file under `debiasR/R`.
3. Read the shared [testing guide](../../tests/README.md) and the corresponding
   tests under `debiasR/tests/testthat`.
4. Implement the Python function in the matching flat module under
   `src/debiaspy`.
5. Add meaningful Python tests under `tests` in the same pull request. Do not
   use empty, `pass`-only or unjustifiably skipped placeholder tests.
6. Export the public function from `src/debiaspy/__init__.py` when the API is
   ready.
7. Update README or tutorial text if the function is user-facing.
8. Review `PROJECT_LOG_STATUS.md` and update it if the work changes the recorded
   project status.
9. Run `uv run pytest`, `uv run ruff check .` and
   `uv run ruff format --check .`.

Keep each change small enough that reviewers can compare the Python behaviour
against the R implementation.
