# Issue 9

Named Python companion LAD/MSOA assets and empirical-loader tests.
The portable end-to-end example is a combined-package follow-up requiring
the adjustments under issues 4-6 and paired validation under issue 11.

Base: `origin/main`.
R reference: `6fa2709ae42de6dc525a4a46182fdfb448b14a3f`.

Sources and tests are included with their implementation.

```bash
python -m pip install -e ".[dev]"
DEBIASR_ROOT=/path/to/debiasR python -m pytest -q
python -m ruff check .
```

The companion loader requires the documented `debiaspydata` assets.
