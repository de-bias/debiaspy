# Project log and status

Last reviewed: 13 October 2026

## Purpose

This document provides a concise snapshot of the current state of `debiaspy`.
It records what is available on `main`, the main areas of active and planned
development, important project decisions and known limitations.

This is not a replacement for the
[R-to-Python API mapping](api-mapping.md), which records the intended location
and status of individual functions, or the
[GitHub issue tracker](https://github.com/de-bias/debiaspy/issues), which holds
detailed tasks and acceptance criteria.

## Current package status

- **Development stage:** early development
- **Package version:** `0.0.0`
- **Supported Python versions:** Python 3.11, 3.12 and 3.13
- **Development environment:** `uv` with Python 3.12 as the shared local
  development version and committed dependency resolution in `uv.lock`
- **Package layout:** `src` layout with a flat set of public modules
- **Public interface:** the bias-measurement functions are implemented and
  exported; adjustment and flow-validation interfaces remain planned
- **Documentation:** the README, contribution guidance, development workflow,
  testing guide, API mapping and Quarto website structure are in place
- **Tutorials:** the planned tutorial sequence exists as placeholders and will
  be developed as the corresponding public interfaces stabilise
- **Example data:** empirical example data are intended to come from the
  separate `debiaspydata` package

## Implemented functionality

The public measurement interface currently includes:

- `measure_bias()`
- `measure_bias_distribution()`
- `validate_bias_residual_structure()`

These functions are implemented in `src/debiaspy/measure_bias.py`, exported
from `src/debiaspy/__init__.py` and covered by focused pytest tests.

The repository also contains the package scaffold, developer tooling,
contribution guidance and the initial Quarto documentation website.

## Active development

The immediate development programme is organised through GitHub issues. Work
should be claimed and discussed on the relevant issue before implementation so
that contributors do not duplicate effort.

Issue [#19](https://github.com/de-bias/debiaspy/issues/19) establishes this
project-status process and connects it to the agent, contribution and pull
request guidance.

## Planned work

The next planned areas are:

- **Deterministic adjustment methods:** inverse penetration, two selection-rate
  approaches, raking ratio and coefficient regression
  ([#4](https://github.com/de-bias/debiaspy/issues/4),
  [#5](https://github.com/de-bias/debiaspy/issues/5),
  [#6](https://github.com/de-bias/debiaspy/issues/6),
  [#7](https://github.com/de-bias/debiaspy/issues/7) and
  [#8](https://github.com/de-bias/debiaspy/issues/8))
- **Example-data access:** a consistent Python route to example data through
  `debiaspydata` ([issue #9](https://github.com/de-bias/debiaspy/issues/9))
- **Multi-method adjustment:** an orchestrator after the individual adjustment
  methods are stable ([issue #10](https://github.com/de-bias/debiaspy/issues/10))
- **Flow validation:** overall, pair-level, distributional, residual, spatial
  and multi-method validation
  ([#11](https://github.com/de-bias/debiaspy/issues/11),
  [#12](https://github.com/de-bias/debiaspy/issues/12),
  [#13](https://github.com/de-bias/debiaspy/issues/13) and
  [#14](https://github.com/de-bias/debiaspy/issues/14))
- **Visualisation:** validation plotting after validation outputs are stable
  ([issue #15](https://github.com/de-bias/debiaspy/issues/15))
- **Bayesian adjustment:** agreement on the Python backend and parity strategy
  before implementation ([issue #16](https://github.com/de-bias/debiaspy/issues/16))
- **Tutorials:** development of the Quarto tutorials after their corresponding
  public functions and outputs are stable

See the [API mapping](api-mapping.md) for the intended implementation order and
the complete function-level plan.

## Important project decisions

- `debiaspy` is a self-contained Python package. `debiasR` is used internally as
  a methodological and behavioural reference, but users should not need to
  know R or `debiasR` to use the Python package.
- Public tabular interfaces use pandas DataFrames and follow normal Python,
  NumPy, pandas and Matplotlib conventions.
- Public functions remain in a flat module layout while the package is at this
  stage of development.
- Public functions are exported from `src/debiaspy/__init__.py` only when they
  are implemented, documented and tested.
- Empirical example datasets belong in `debiaspydata`; small deterministic test
  fixtures belong in this repository under `tests/`.
- Every change to `main` is made through a branch and pull request. Direct
  pushes to `main` are not part of the project workflow.
- Contributors use `uv` for the shared local development environment. Python
  3.12 is the standard development version, while the package continues to
  support Python 3.11, 3.12 and 3.13.
- Dependencies are added only when implemented functionality requires them.
  Broad compatibility requirements belong in `pyproject.toml`; `uv.lock`
  records the exact shared resolution and is committed.
- New or changed functions and their meaningful tests are submitted together
  in the same pull request. `tests/README.md` is the shared testing guide;
  empty, `pass`-only and unjustifiably skipped placeholder tests are not used.
- Human contributors remain responsible for reviewing agent-assisted code,
  tests and documentation.

## Known limitations

- The package is not yet a stable release.
- Adjustment and flow-validation modules are currently placeholders rather than
  usable public interfaces.
- The example-data loader has not yet been implemented.
- The Bayesian backend and model-parity strategy have not yet been agreed.
- The Quarto tutorials are placeholders and do not yet provide a complete
  user-learning pathway.
- The current empirical scope and cross-language parity will need broader
  evaluation as adjustment and validation functions are implemented.

## Recent milestones

- Established the initial package structure, development tooling and Quarto
  website.
- Implemented the complete bias-measurement interface and focused tests.
- Created the `debiaspydata` companion package for empirical example data.
- Recorded the agreed R-to-Python public API mapping and implementation order.
- Added repository-level guidance for human contributors working with coding
  agents.
- Established this project log and status process through issue
  [#19](https://github.com/de-bias/debiaspy/issues/19).
- Standardised the contributor development environment with `uv` through issue
  [#21](https://github.com/de-bias/debiaspy/issues/21).
- Established the shared testing workflow through issue
  [#23](https://github.com/de-bias/debiaspy/issues/23).

## How to update this document

Every pull request must review this document and update it where necessary.
The contributor should make the update in the same branch and pull request as
the work that changes project status.

Update this document when a pull request changes implemented functionality,
the public API, dependencies, package structure, methodological decisions,
known limitations, development priorities or another statement recorded here.
A small change that does not affect the recorded project status does not need
to edit the file, but the contributor must still review it.

Write entries to describe the state that will be true on `main` after the pull
request is merged. Do not describe unmerged work as completed. Keep the summary
concise, update the review date when the file changes and link to the relevant
issue or pull request for detail.
