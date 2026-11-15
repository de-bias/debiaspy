# Project log and status

Last reviewed: 15 November 2026

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
- **Dependency security:** Pillow 13.0.0 is locked; GitHub reported zero open
  Dependabot alerts when checked on 15 November 2026
- **Package layout:** `src` layout with a flat set of public modules
- **Public interface:** the bias-measurement functions and inverse-penetration
  adjustment are implemented and exported; other adjustment methods and
  flow-validation interfaces remain planned
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

The public adjustment interface also includes `adjust_inverse_penetration()`,
implemented in `src/debiaspy/adjust_inverse_penetration.py`. It supports origin,
destination and geometric-mean weighting, with source-specific coverage
matching when both inputs contain `mpd_source`. It is publicly exported and
covered by calculation, input-validation and fixed R-reference tests.

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

- **Remaining deterministic adjustment methods:** two selection-rate
  approaches, raking ratio and coefficient regression
  ([#5](https://github.com/de-bias/debiaspy/issues/5),
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
- Carmen maintains this log after implementation PRs are merged, recording
  merge dates, PR numbers and summaries in merge order through separate
  documentation PRs. Contributors describe status changes in their PR bodies.

## Known limitations

- The package is not yet a stable release.
- Adjustment modules other than inverse penetration and the flow-validation
  modules remain placeholders on `main`.
- The example-data loader has not yet been implemented.
- The Bayesian backend and model-parity strategy have not yet been agreed.
- The Quarto tutorials are placeholders and do not yet provide a complete
  user-learning pathway.
- The current empirical scope and cross-language parity will need broader
  evaluation as adjustment and validation functions are implemented.

## Recent milestones

- **15 November 2026 — Pillow security update:** merged
  [PR #29](https://github.com/de-bias/debiaspy/pull/29) after the inverse
  penetration contribution and its project-status update. Updated the locked
  Pillow version from 12.3.0 to 13.0.0 without changing other dependency
  versions. All 116 tests, Ruff linting and formatting, the documentation render
  and a Matplotlib PNG export/Pillow read check passed on Python 3.12. GitHub
  subsequently reported zero open Dependabot alerts. No package interfaces
  changed.

- **15 November 2026 — inverse penetration:** merged Massimiliano Luca's
  implementation through [PR #27](https://github.com/de-bias/debiaspy/pull/27),
  completing [issue #4](https://github.com/de-bias/debiaspy/issues/4). Carmen
  reviewed the behaviour against R and confirmed that the full test suite,
  linting, formatting and public import checks passed. The contribution includes
  27 stored R-reference output tables across nine scenarios and three weighting
  modes. No dependencies were changed. Raking-ratio PR #25 remains unmerged and
  is next for review.

Earlier milestones:

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

Carmen Cabrera maintains this document after implementation PRs are merged.
Contributors and coding agents should review it before work and describe
relevant status changes, checks and remaining work in their PR descriptions.
Leave the log unchanged unless Carmen requests an edit.

After a merge, Carmen updates the current snapshot and adds a dated milestone
with the merged PR number and a concise summary. Record milestones in merge
order, with the newest first, and distinguish completed work from open PRs.
Update the review date and link to the relevant issue or PR for detail.

Log updates use a separate documentation branch and PR. Implementation PRs
should still include necessary README, API mapping and other documentation
changes. Small changes that do not affect recorded status do not require a new
milestone.
