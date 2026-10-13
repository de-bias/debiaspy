# Development workflow

This note records the basic local workflow for developing `debiaspy`.

## One-time setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Check that the package imports

```bash
python -c "import debiaspy; print(debiaspy.__version__)"
```

Expected output:

```text
0.0.0
```

## Run tests

```bash
python -m pytest
```

## Run linting

```bash
python -m ruff check .
```

## Format code

```bash
python -m ruff format .
```

## Implementing a function

When porting a function from `debiasR`:

1. Check the [R-to-Python API mapping](api-mapping.md) and confirm that nobody
   else is working on the same function.
2. Identify the source R file under `debiasR/R`.
3. Read the corresponding tests under `debiasR/tests/testthat`.
4. Implement the Python function in the matching flat module under
   `src/debiaspy`.
5. Add focused Python tests under `tests`.
6. Export the public function from `src/debiaspy/__init__.py` when the API is
   ready.
7. Update README or tutorial text if the function is user-facing.
8. Run `python -m pytest` and `python -m ruff check .`.

Keep each change small enough that reviewers can compare the Python behaviour
against the R implementation.

## Optional comparison with R

The measurement edge-case tests include a live comparison with
`debiasR/R/measure_bias.R`. To run it against a local debiasR checkout:

```bash
DEBIASR_ROOT=/path/to/debiasR python -m pytest tests/test_measure_bias_r_parity.py
```

This comparison requires `Rscript`; it sources the measurement function directly.
The regular Python edge-case tests run without R. Record the debiasR commit used
when reporting a live comparison in a pull request.
