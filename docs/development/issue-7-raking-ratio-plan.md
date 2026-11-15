# Issue #7: raking-ratio implementation plan

## Scope and ownership

- **Issue:** [#7 — adjustment: implement raking ratio](https://github.com/de-bias/debiaspy/issues/7)
- **Human owner:** Antonio (`RiegelGestr`)
- **Local branch:** `antonio/issue-7-raking-ratio`
- **Target:** `src/debiaspy/adjust_raking_ratio.py`
- **Companion tests:** `tests/test_adjust_raking_ratio.py`

This is one issue, one branch and one future pull request. No direct push to `main`, commit, push or pull-request creation is authorised by this plan.

## Required behaviour

Implement `adjust_raking_ratio()` as the pandas/NumPy equivalent of `debiasR::adjust_raking_ratio()`. It adjusts observed origin-destination (OD) flows with iterative proportional fitting (IPF) so selected origin and/or destination margins match target totals.

The public signature should preserve the documented R arguments in Python form:

```python
adjust_raking_ratio(
    mpd_od_df,
    origin_targets=None,
    destination_targets=None,
    benchmark_od_df=None,
    flow_col_bench="flow",
    group_cols=None,
    max_iter=200,
    tol=1e-6,
    clip_min=0,
    clip_max=np.inf,
    keep_cols=(),
)
```

`mpd_od_df` must contain `origin`, `destination` and `flow`. Target tables use `origin` or `destination` plus `target`; benchmark margins are derived from the selected benchmark flow column. When `group_cols` is supplied, IPF runs independently in each stratum.

The returned copy must contain, in stable order where present:

```text
origin, destination, mpd_source, <group_cols>, <keep_cols>,
flow, flow_adj, weight_ipf
```

It must retain `DataFrame.attrs["ipf_converged"]` and `DataFrame.attrs["ipf_iterations"]`.

## Implementation choices

These choices satisfy the issue’s explicit requirement for clear failure modes while remaining methodologically compatible with the R reference.

| Case | Python behaviour to implement and test |
| --- | --- |
| One target direction | Scale only that direction’s margins. |
| Both target directions | Alternate origin and destination scaling until the largest scale-factor change is below `tol`, or `max_iter` is reached. |
| Benchmark input | Derive only the omitted target table(s), allowing an explicit target table to override its corresponding benchmark margin. |
| Convergence limit | Return the best IPF result with `ipf_converged=False` and the actual iteration count; do not silently claim success. |
| Invalid numeric controls | Raise `ValueError` for non-positive/non-integral `max_iter`, non-positive `tol`, or invalid clip bounds. |
| Invalid data | Raise `ValueError` for missing required columns, non-finite or negative flows/targets, duplicate target keys, or missing group columns. |
| Duplicate target keys after canonicalisation | Detect duplicates on stringified area keys (plus `group_cols`), so values such as `1` and `"1"` raise rather than silently overwrite in the IPF maps. |
| Missing stratum keys | Raise `ValueError` when any `group_cols` value is missing in the OD table, target tables or benchmark. Missing strata are not a supported key and must not be ignored. |
| Missing target support | Raise `ValueError` when an explicit target names an area/stratum absent from the input OD table. |
| Structural zeros | Raise `ValueError` if a positive active margin has zero flow support; IPF cannot create a new OD cell. |
| Clipping | Apply bounds to `weight_ipf` after IPF and recompute `flow_adj = flow * weight_ipf` so the documented identity remains true. Mark convergence false when clipping changes a finite weight. |

The R implementation clips `weight_ipf` after calculating `flow_adj`, which can make the two output columns inconsistent. The Python implementation will make the identity explicit and record this deliberate, user-visible difference in the future pull request.

## R/Python parity evidence

A reproducible, non-pytest verification lives in
`scripts/verify_raking_ratio_r_parity.py`. It requires a local `debiasR`
checkout and runs:

```bash
DEBIASR_ROOT=/path/to/debiasR uv run python scripts/verify_raking_ratio_r_parity.py
```

On 16 October 2026 this was executed against `debiasR` revision
`6fa2709ae42de6dc525a4a46182fdfb448b14a3f` with R 4.6.1 and Python 3.12.15.
The unclipped 2×2 smoke fixture matched to floating-point noise
(max absolute difference `4.97e-14`, RMSE `4.30e-14`, both runs converged in
7 iterations). Artefacts:

- `docs/reports/issue-7-raking-ratio-r-parity.md`
- `docs/reports/issue-7-raking-ratio-r-parity.png`

Technical limit: the script depends on an external `DEBIASR_ROOT` checkout and
R packages (`dplyr`, `rlang`, `tibble`). It is not part of the pytest suite and
does not install R or system dependencies.

## Narrow implementation design

Keep the implementation self-contained in `adjust_raking_ratio.py`, using only the project’s existing NumPy and pandas dependencies. It should:

1. validate public inputs without mutating caller-owned DataFrames;
2. canonicalise only the area keys needed to match R’s character-key handling;
3. derive and validate margins;
4. split input rows by `group_cols` when requested;
5. run a small private IPF helper on each subset;
6. assemble the documented output and convergence attributes; and
7. export the completed function from `src/debiaspy/__init__.py` and `__all__`.

Do not import the large `contribution/non-bayesian-parity` patch wholesale. Its raking implementation is based on the unmerged `contribution/issue-4-6` branch and introduces shared modules outside this issue’s scope. It is useful as a testable reference, not as this branch’s dependency.

No new package is currently justified: pandas and NumPy are sufficient. Do not change `pyproject.toml` or `uv.lock` unless an actual, reviewed implementation demonstrates a requirement.

## Test matrix

The Python tests should use small deterministic fixtures, not the external empirical data package. They must cover:

- a two-way 2×2 fixture whose adjusted origin and destination sums equal the supplied targets;
- origin-only and destination-only targets;
- benchmark-derived margins;
- stratified (`group_cols`) fitting;
- expected output columns, attributes and preservation of the input frame;
- tolerance and maximum-iteration controls, including a deliberate non-convergent case;
- invalid/missing columns, invalid controls, negative or non-finite values, duplicate keys and unmatched areas;
- duplicate target keys that collide only after area-key canonicalisation;
- missing values in `group_cols` (rejected, not silently skipped);
- zero-flow structural support and clipped weights; and
- a fixed numerical example cross-checked against the R test shape.

The final implementation must pass:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Work sequence for agent-assisted implementation

1. Re-read this plan, `AGENTS.md`, the issue and the current working-tree status. Do not alter unrelated files.
2. Implement the small private validation and IPF helpers plus the public function in the mapped module.
3. Add focused pytest coverage in the companion test module.
4. Update the public README from “planned” to a concise raking-ratio usage description once the API is stable.
5. Review `docs/development/PROJECT_LOG_STATUS.md`; update it only in the implementation change, describing the state that will be true after merge.
6. Run the complete test, lint and formatting checks and present their results for Antonio’s review.
7. Leave changes uncommitted until Antonio explicitly authorises a commit.

## Sources inspected

- `CONTRIBUTING.md`, `AGENTS.md`, `docs/development/PROJECT_LOG_STATUS.md`, `docs/development/api-mapping.md`, `docs/development/workflow.md` and `tests/README.md` in this checkout.
- [Issue #7](https://github.com/de-bias/debiaspy/issues/7).
- [`debiasR` source](https://github.com/de-bias/debiasR/blob/main/R/adjust_raking_ratio.R), its [`test-adjust-raking-ratio.R`](https://github.com/de-bias/debiasR/blob/main/tests/testthat/test-adjust-raking-ratio.R), and its [`smoke test`](https://github.com/de-bias/debiasR/blob/main/tests/testthat/test-adjust_raking_ratio-smoke.R).
- `contribution/non-bayesian-parity`, whose contribution index records its raking work as dependent on `contribution/issue-4-6`.

## Current setup state

On 16 October 2026, this checkout was cloned from the public repository, `uv` was installed, `uv sync` completed with the locked dependencies, and the baseline suite completed with **44 passed**. The working branch is local and has not been pushed. `PROJECT_LOG_STATUS.md` was reviewed; it requires no change during this planning-only setup step.
