# Issue 13

Residual and spatial diagnostics: live R global/Local Moran, weighted links and isolated areas.

Base: `contribution/issue-12`.
R reference: `6fa2709ae42de6dc525a4a46182fdfb448b14a3f`.

Sources and tests are included with their implementation.

```bash
python -m pip install -e ".[dev]"
DEBIASR_ROOT=/path/to/debiasR python -m pytest -q
python -m ruff check .
```

The companion loader requires the documented `debiaspydata` assets.
Plot maps require `.[plot]`.
