# Issue 10

Selected deterministic-method orchestration. Multilevel dispatch requires the model implementation under #16.

Base: `contribution/issue-8`.
R reference: `6fa2709ae42de6dc525a4a46182fdfb448b14a3f`.

Sources and tests are included with their implementation.

```bash
python -m pip install -e ".[dev]"
DEBIASR_ROOT=/path/to/debiasR python -m pytest -q
python -m ruff check .
```

The companion loader requires the documented `debiaspydata` assets.
Plot maps require `.[plot]`.
