# Testing guide

This guide explains the minimum testing expectations for every `debiaspy`
contribution. It applies to human contributors and to work prepared with coding
agents.

## Core rule

Every new or changed function must be submitted with meaningful tests in the
same pull request. Tests are part of the implementation, not a later task.

Do not commit empty test files, tests containing only `pass` or tests that are
unconditionally skipped or marked as expected failures without a documented
reason. A test must exercise behaviour and make a useful assertion about the
result, warning or error.

## Where tests go

Place tests under `tests/` in a file corresponding to the source module:

```text
src/debiaspy/adjust_inverse_penetration.py
tests/test_adjust_inverse_penetration.py
```

Use descriptive test names that state the behaviour being checked, for example
`test_adjust_inverse_penetration_rejects_zero_coverage()`.

## How to decide what to test

Before writing tests:

1. Read the function's GitHub issue and acceptance criteria.
2. Read the corresponding `debiasR` function and documentation.
3. Inspect the corresponding tests under `debiasR/tests/testthat/`.
4. Identify any deliberate difference required for a clear Python interface.
5. Select the relevant cases from the checklist below.

Project leads do not need to provide a separate exhaustive test list for every
function. The contributor implementing the function is responsible for
identifying the relevant behaviour and edge cases. Raise methodological
ambiguities with Carmen Cabrera or Francisco Rowe rather than inventing an
answer in the implementation or tests.

## Standard coverage checklist

Consider the following for every new or changed public function:

- a basic example with explicit expected numerical results
- a fixed example demonstrating agreement with `debiasR`, where relevant
- the return type, keys, column names and column ordering
- required input columns and accepted input types
- missing values, zero values and empty inputs
- duplicated identifiers or rows where they affect the method
- invalid argument values and incompatible argument combinations
- expected warnings and informative errors
- preservation of the caller's input DataFrame
- deterministic behaviour, including a controlled seed when randomness is
  necessary
- method-specific behaviour such as convergence, non-convergence, model
  failure, spatial links or optional diagnostics

Not every item applies to every function. Tests should cover the applicable
behaviour without duplicating assertions that add no useful protection.

## What meaningful tests look like

A meaningful test normally contains:

- a small, understandable input
- an explicit expected result or behaviour
- a call to the public function
- one or more focused assertions

Avoid tests that only check that a result exists. Where numerical precision is
relevant, use an appropriate approximate comparison and document the expected
value. Do not reproduce the complete implementation inside the test to
calculate the expected result.

Skipping a test requires a clear technical reason, such as an optional external
resource that is unavailable in the test environment. Core deterministic tests
should not be skipped merely because the implementation is unfinished or the
test currently fails.

## Running tests and checks

Run the relevant test file while developing:

```bash
uv run pytest tests/test_module_name.py
```

Before requesting review, run the complete suite and code checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Do not weaken, remove or rewrite a valid existing test solely to make a new
implementation pass. If the expected behaviour needs to change, explain and
review that decision in the pull request.
