# R parity contribution review

`contribution/r-parity` contains all adjustment, validation and plotting methods
from the checked combined package, based on organisation main
`56c188a172dd71703f23973b88e22cf3c742d36b`. Main is unchanged.
The existing integration and measurement branches remain available.

## Checked behaviour

The installed wheel passes 342 tests on macOS ARM64 with Python 3.11, live R
comparisons and the optional short Bambi/PyMC sampling test enabled. Three
Bambi/Pandas deprecation warnings remain. Ruff, formatting and whitespace checks
pass. The earlier fresh Python 3.14 base/geographic environment passed 114
checks with one optional R comparison skipped. Both environments passed
`pip check`; the new branch wheel was rebuilt and installed for its final suite.

Fresh R/Python Census article calls pass 149 of 151 comparisons against debiasR
`6fa2709ae42de6dc525a4a46182fdfb448b14a3f`. The two differences concern the
Python public exports. All compared scientific tables pass their unchanged
absolute tolerances. Bayesian vignette tables retain their precomputed-reference
status. The companion reference is
`19d1c5f67f5dcf70fcd765e42fa03d667df496df`.

The installed synthetic and full Census LAD examples retain 42,455 observed/
adjusted rows and validate 34,762 matched benchmark pairs. The portable CSV
example preserves arbitrary identifiers, leading zeros and source/time scope;
undefined adjusted flows are retained and withhold metrics. Coverage derived
from the benchmark's OD margins makes this a calibration demonstration.
Independent evaluation data are needed to assess accuracy beyond calibration.

## Issue-ordered implementation

Commits follow GitHub's displayed order, #16 down to #7, with #4–#6 grouped.
Supporting code is introduced before dependent exports become usable. Intermediate
commits were checked for imports, pytest and Ruff as their APIs became available.
The final branch retains all original tests and adds a nested-metric plotting
regression. See [suggested contribution format](contribution-format.md).

| Issues | Implemented behaviour and main evidence |
| --- | --- |
| #4–#8 | Inverse weights, both selection-rate methods, raking and coefficient calibration; fixed R comparisons and zero/missing/input/identity checks. |
| #9 | Named Python companion assets, LAD/MSOA examples and portable caller-supplied OD inputs. |
| #10 | All-method orchestration and shared frequentist preparation/fitting. |
| #11–#12 | Paired/overall validation and shared distribution calculations, preserving scoped joins and denominators. |
| #13 | Residual/spatial diagnostics, boolean LISA classification, weighted links, isolates and constant residuals; four cases compared live with R. |
| #14–#15 | Method-specific errors, shared core validation and all eight plotting functions, including real LISA rendering and outline CRS alignment. |
| #16 | Proposed Bayesian backends, scoped prior/prediction evidence and runtime diagnostics; strategy and scientific acceptance remain pending. |

## Simplification

The additional cleanup removes 91 runtime Python lines and 103 article-checker
lines relative to `0fce961`, retaining every method and every original test.
Global Moran calculations share one implementation while measurement and flow
validation retain their distinct input rules. Common model attributes come from
the metadata already constructed by each engine, preserving fields, diagnostics,
priors and attribution. Checked frequentist and mocked Bayesian outputs match
the previous tables and scientific metadata after excluding runtime timing.

Plot extraction now handles nested single-comparison metric records through the
shared recursive table extractor. Its regression verifies both method labels
and plotted values. The article checker calls the public API directly and reads
named R datasets, removing redundant function injection and fallback machinery.
The public multilevel module is 524 lines; its original combined implementation
was 2,094 lines. Earlier cleanup removed duplicate parsers, legacy data routes,
synthetic production fallbacks, fingerprint/cache machinery and deprecated aliases.

Use `formula` or paired mobility/bias formulas in place of R's deprecated
`custom_formula`. Optional runtimes use `grouped`, `bayesian`, `latent` and `plot`
extras. `requirements.txt` resolves the authoritative `pyproject.toml` dependency
list. The Bayesian extra uses Matplotlib `<3.11` after a fresh install exposed the
[upstream ArviZ style import failure](https://github.com/arviz-devs/arviz-plots/issues/453).
All source attribution, contributor records and the single MIT licence remain.

## Reproducibility and remaining limits

The full PDF, authored LaTeX, recorded figures and comparison JSON are in
`docs/reports`. The [reproduction commands](../reports/README.md) distinguish
rebuilding the recorded report from fresh numerical comparisons. The existing
Quarto website and non-sampling tutorial snippets were checked in the preceding
integration verification. Polars remains a proposed evaluation.

The Bayesian investigation demonstrates matching priors, likelihood and
shared-draw predictions for the checked fixed-effect Poisson case. Its
extrapolated posterior means remain imprecise. Six retained R-reference ZIP
quality checks remain unresolved, and the full Bayesian sampling matrix has
not been rerun. CmdStan compilation/full latent sampling, wider platform matrices,
random slopes, multiple/nested/crossed effects and latent ZIP/ZINB remain outside
the verified scope. Python and R permutation streams differ.

AI assistance was used. Named human review and Bayesian backend agreement remain
pending. No PR has been created, no issue has been closed, and no merge into main
has been performed.
