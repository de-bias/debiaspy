# Issue 11

Paired and overall validation: scoped joins, metric denominators and missing values.

Base: `main` at `56c188a172dd71703f23973b88e22cf3c742d36b`.
Measurement arithmetic and shared numeric helpers are included in this
contribution's prerequisite commit, followed by paired and overall validation.
R reference: `6fa2709ae42de6dc525a4a46182fdfb448b14a3f`.

Sources and tests are included with their implementation.

```bash
python -m pip install -e ".[dev]"
DEBIASR_ROOT=/path/to/debiasR python -m pytest -q
python -m ruff check .
```

The companion loader requires the documented `debiaspydata` assets.
Plot maps require `.[plot]`.
