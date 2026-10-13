# Reproduce the side-by-side comparison

The full report's authored LaTeX, numerical tables and recorded R/Python figures
are in `source/`. Build the recorded report with a TeX installation providing
`pdflatex` and the packages listed in the source preamble:

```bash
python tools/build_report.py --output output/debiasr_debiaspy_side_by_side.pdf
```

The default build includes the recorded `article_comparison.json` summary and
the current [method evidence table](../development/method-evidence.md).
The Markdown table is the source for the PDF method appendix.
The build runs twice for cross-references and the contents page. Its intermediate
files stay in a temporary directory. Recorded figures and tables retain their
original comparison values; compiling them reproduces that evidence snapshot.

## Re-run the numerical comparison

R code: `check_article_parity.R`.
Python code: `check_article_parity.py`.
Test: `check_article_parity.py`.

Install the Python package and companion described in the README. Check out
`de-bias/debiasR` at `6fa2709ae42de6dc525a4a46182fdfb448b14a3f` and install its
R dependencies, including `debiasRdata`, `pkgload`, `jsonlite`, `dplyr` and
`tibble`. The data reference used here is `de-bias/debiasRdata` at
`e652a1186c705d1edc4ff5427e7e2d5230ef7fe7`.

```bash
python tools/check_article_parity.py \
  --r-root /path/to/debiasR --output output/article-parity
```

The command creates a new run directory containing full-precision R/Python CSVs,
display tables, schema checks and `article_comparison.json`. It runs both
implementations afresh. The original 151 comparisons use their existing absolute
tolerances. It succeeds when the only differences are the two documented Python
public-export choices. Unexpected numerical or schema differences return a
failing exit status. Existing results remain available in their run directories.

The Bayesian article tables read the R vignette's precomputed example outputs;
they check file ingestion. Use `check_bayesian_parity.py` and
`check_bayesian_parity.R` for the separate prior, prediction and sampling
investigation described in [Bayesian investigation](../development/bayesian-investigation.md).
The article command covers the Census example. Public method tests and
`examples/benchmark_od.py` cover caller-supplied OD and benchmark tables.

Add a fresh comparison summary to the recorded report:

```bash
python tools/build_report.py \
  --checks output/article-parity/article-RUN/article_comparison.json \
  --output output/debiasr_debiaspy_side_by_side.pdf
```

The appended section identifies the current checks and reference commits.
Review changed scientific tables and figures in the authored source before
publishing a new evidence snapshot. Figure appearance is checked separately
by the plotting tests; the article command regenerates numerical tables.
