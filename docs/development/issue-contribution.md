# Issue 10

Orchestration for explicitly selected deterministic methods.
Pass the deterministic method names through `methods`. The default
`methods="all"` and multilevel dispatch require the model implementation
under issue 16.

Base: `contribution/issue-8`.
R reference: `6fa2709ae42de6dc525a4a46182fdfb448b14a3f`.

Sources and tests are included with their implementation.

```bash
python -m pip install -e ".[dev]"
DEBIASR_ROOT=/path/to/debiasR python -m pytest -q
python -m ruff check .
```

