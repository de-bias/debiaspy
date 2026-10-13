# Checked issue contributions

Each branch contains its implementation, required shared code, focused tests and
contribution notes. The base column identifies the prerequisite branch for a
focused comparison. Roots use organisation main at
`56c188a172dd71703f23973b88e22cf3c742d36b`.

Each wheel was installed and its complete branch test suite run on macOS ARM64
with Python 3.11, prepared dependencies and `DEBIASR_ROOT` configured for the
pinned R checkout. Imports, Ruff, formatting and `pip check` passed for all
branches. The wheel inventories contain the non-Bayesian implementations.
Counts include tests inherited from the base and should be interpreted per branch.

| Branch | Base | Installed tests | Scope |
| --- | --- | --- | --- |
| `contribution/issue-4-6` | `main` | 93 passed | Inverse penetration and both selection methods; live R calibration and input boundaries. |
| `contribution/issue-7` | `contribution/issue-4-6` | 104 passed | Raking: source/time keys, target margins and numerical boundaries. |
| `contribution/issue-8` | `contribution/issue-7` | 114 passed | Coefficient calibration: OLS/count families, warning contracts and numerical references. |
| `contribution/issue-9` | `main` | 51 passed | Named companion LAD/MSOA assets. Portable end-to-end example tests use the complete contribution package. |
| `contribution/issue-11` | `main` | 57 passed | Measurement arithmetic, shared numeric helpers, paired and overall validation; scoped joins, denominators and missing values. |
| `contribution/issue-12` | `contribution/issue-11` | 67 passed | Distribution validation: weighted shares, KL/JSD and invalid contributions. |
| `contribution/issue-13` | `contribution/issue-12` | 89 passed | Residual and spatial diagnostics: live R global/Local Moran, weighted links and isolated areas. |
| `contribution/issue-14` | `contribution/issue-13` | 107 passed | Named multi-method wrappers, margin helpers and method-specific errors. |
| `contribution/issue-15` | `contribution/issue-14` | 144 passed | Eight plot functions: figure contracts, panels, LISA classes and CRS alignment. |
| `contribution/issue-10` | `contribution/issue-8` | 117 passed | Explicitly selected deterministic-method orchestration. Default methods="all" and multilevel dispatch require the model implementation under issue 16. |

Issue 11 includes the measurement arithmetic and shared numeric prerequisites
in its commit history. Its contribution base is main; issues 12 through 15
continue to use their listed validation prerequisites.

## Reproduce a contribution check

```bash
git switch contribution/issue-4-6
python -m pip wheel . --no-deps --wheel-dir /tmp/debias-wheel
python -m pip install --force-reinstall --no-deps /tmp/debias-wheel/debiaspy-0.0.0-py3-none-any.whl
DEBIASR_ROOT=/path/to/debiasR MPLBACKEND=Agg python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m pip check
```

Use a clean virtual environment with the dependencies specified by the selected
branch and pytest/Ruff installed. The checks above used prepared dependencies;
fresh dependency resolution across every branch remains a separate platform check.
The empirical loader tests use the Python companion pinned at
`19d1c5f67f5dcf70fcd765e42fa03d667df496df`. Install it with:

```bash
python -m pip install git+https://github.com/de-bias/debiaspydata.git@19d1c5f67f5dcf70fcd765e42fa03d667df496df
```

Install `.[plot]` for the issue-15 GeoPandas map checks. R numerical regressions
use `6fa2709ae42de6dc525a4a46182fdfb448b14a3f`.

## Complete-package checks

The combined contribution branch retains all 204 original test functions and
assertion bodies. Their issue-specific files share adjustment and validation
fixtures. The portable OD example has its own tests and uses both adjustment and
validation; these integration checks remain with the combined package.
The full method evidence table distinguishes live R, retained R and Python-only
checks. Human review of each contribution follows the repository template.
