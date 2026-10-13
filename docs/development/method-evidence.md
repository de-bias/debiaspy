# Method-by-method parity evidence

R reference: `6fa2709ae42de6dc525a4a46182fdfb448b14a3f`.
Python implementation: `ff41d43b8281494629d1a756eab34c93aa5c1cd1`;
subsequent changes update documentation and evidence presentation.

**Live R** executes the R reference during the article checker or named regression.
**Retained R** compares with previously recorded R outputs.
**Python contracts** check behaviour, schemas, errors or figures within Python.
The table covers all 31 public callable methods. `__version__` is package metadata.
Each row describes its checked cases. The 342-test suite includes all these types;
the article checker contains 151 comparisons, with 149 passes and two documented
public-export differences. Bayesian vignette tables read retained R outputs.

Article evidence: `check_article_parity.R`, `check_article_parity.py` and
`docs/reports/article_comparison.json`. Reproduction commands are in
[report reproduction](../reports/README.md). Plot tests check figure structure and
values; the report preserves selected R/Python figures for visual review.

| Method | Issue | R source | Python source | Tests | Evidence and scope |
| --- | --- | --- | --- | --- | --- |
| `measure_bias` | Measurement | `measure_bias.R` | `measure_bias.py` | test_measure_bias.py; test_measure_bias_r_parity.py | Live R: Census arithmetic and zero, missing and infinite values. Python: input and warning contracts. |
| `measure_bias_distribution` | Measurement | `measure_bias.R` | `measure_bias.py` | test_measure_bias_distribution.py; test_shared_distribution_metrics.py | Live R article: Census shares and divergence summaries. Python: smoothing, invalid totals and missing values. |
| `validate_bias_residual_structure` | Measurement | `measure_bias.R` | `measure_bias.py` | test_validate_bias_residual_structure.py | Live R article: Census residual summaries. Python: endpoint roles, weighted links and isolated areas. |
| `adjust_inverse_penetration` | #4 | `adjust_inverse_penetration.R` | `adjust_inverse_penetration.py` | test_adjust_inverse_penetration.py; test_r_adjustment_parity.py | Live R: origin/destination/both weights, zero/missing/infinite values, row expansion and overwriting adjusted flow. Article: Census outputs. |
| `adjust_selection_rate` | #5 | `adjust_selection_rate.R` | `adjust_selection_rate.py` | test_adjust_deterministic_methods.py; test_r_adjustment_parity.py | Live R: origin/OD calibration, missing coverage, repeated observations and numerical boundaries. Article: Census outputs. |
| `adjust_selection_rate2` | #6 | `adjust_selection_rate2.R` | `adjust_selection_rate2.py` | test_adjust_deterministic_methods.py; test_r_adjustment_parity.py | Live R: origin/OD calibration, missing coverage, repeated observations and numerical boundaries. Article: Census outputs. Selection-rate II also checks grouped calibration. |
| `adjust_raking_ratio` | #7 | `adjust_raking_ratio.R` | `adjust_raking_ratio.py` | test_adjust_deterministic_methods.py | Live R article: Census adjusted totals and selected rows. Python: margin conservation, source/time scope and zero-cell behaviour. |
| `adjust_coefficient` | #8 | `adjust_coefficient.R` | `adjust_coefficient.py` | test_adjust_deterministic_methods.py | Live R article: Census coefficients, totals and selected rows. Python: calibration, required columns and numerical boundaries. |
| `debiaspy_example_data` | #9 | `example-data.R` | `example_data.py` | test_example_data_companion.py; test_adjust_deterministic_methods.py | Live R article: named Census asset counts/schemas and selected values. Python: companion API, LAD/MSOA inputs and portable CSV identifiers. Companion assets required for the empirical loader. |
| `adjust_all_methods` | #10 | `adjust_all_methods.R` | `adjust_all_methods.py` | test_adjust_deterministic_methods.py; test_multilevel_frequentist_safety.py | Live R article: selected-method Census summaries. Python: selection, parameters and scoped preparation. Available engine scope follows each method. |
| `adjust_multilevel_bayes` | #16 (deferred) | `adjust_multilevel_bayes.R` | `adjust_multilevel_bayes.py` | test_multilevel_frequentist_safety.py; test_bayesian_prediction_r_parity.py; test_bayesian_backend_contract.py; test_latent_backend_contract.py | Frequentist: live R Census article and retained R model probes. Bayesian: retained investigation and backend contracts; acceptance remains pending. Checked Bayesian scope: fixed-effect Poisson investigation. |
| `validate_flow_overall` | #11 | `validate_flows.R` | `validate_flows.py` | test_validation_core.py | Live R article via method wrapper: Census metrics. Python: grouped sources, missing values and joined data. |
| `validate_flow_pairs` | #11 | `validate_flows.R` | `validate_flows.py` | test_validation_core.py | Live R article via overall/wrapper outputs: Census joined pairs. Python: signed residuals and join columns. |
| `validate_flow_distribution` | #12 | `validate_flows.R` | `validate_flows.py` | test_validation_core.py | Live R article via method wrapper: Census divergence summaries. Python: smoothing and invalid contributions. |
| `validate_flow_residuals` | #13 | `validate_flows.R` | `validate_flows.py` | test_validation_core.py | Live R article via method wrapper: Census residual summaries. Python: ranking, signs and missing/infinite values. |
| `validate_flow_residual_structure` | #13 | `validate_flows.R` | `validate_flows.py` | test_flow_residual_structure.py | Live R regression: four spatial cases, global/Local Moran, weighted links, isolates and constant residuals. Article: Census summaries. Permutations reproduce within Python. |
| `validate_flow_prepare_output` | #14 | `validate_flow_methods.R` | `validate_flow_methods.py` | test_validate_flow_methods.py | Python contracts: single output, columns and input preservation; exercised by preparation of multiple methods. |
| `validate_flow_prepare_outputs` | #14 | `validate_flow_methods.R` | `validate_flow_methods.py` | test_validate_flow_methods.py | Live R article: preparation of Census methods. Python: named mappings, reserved columns and ambiguous keys. |
| `validate_flow_overall_methods` | #14 | `validate_flow_methods.R` | `validate_flow_methods.py` | test_validate_flow_methods.py | Live R article: per-method Census metrics. Python: forwarding and failing-method context. |
| `validate_flow_residual_methods` | #14 | `validate_flow_methods.R` | `validate_flow_methods.py` | test_validate_flow_methods.py | Live R article: per-method Census residuals. Python: forwarding and failing-method context. |
| `validate_flow_distribution_methods` | #14 | `validate_flow_methods.R` | `validate_flow_methods.py` | test_validate_flow_methods.py | Live R article: per-method Census distributions. Python: forwarding and failing-method context. |
| `validate_flow_residual_structure_methods` | #14 | `validate_flow_methods.R` | `validate_flow_methods.py` | test_validate_flow_methods.py | Live R article: per-method Census spatial summaries. Python: forwarding and failing-method context. |
| `validate_flow_margins` | #14 | `validate_flow_methods.R` | `validate_flow_methods.py` | test_validate_flow_methods.py | Live R article: origin/destination Census margins. Retained R fixture totals and Python input-preservation contracts. |
| `validate_flow_plot_metrics` | #15 | `plot_validation.R` | `plot_validation.py` | test_validation_plot_contract.py | Python figure contracts: nested metrics, labels and plotted values.  |
| `validate_flow_plot_residuals` | #15 | `plot_validation.R` | `plot_validation.py` | test_validation_plot_contract.py | Python figure contracts: residual values and method/comparison panels.  |
| `validate_flow_plot_scatter` | #15 | `plot_validation.R` | `plot_validation.py` | test_validation_plot_contract.py | Python figure contracts: series, axes and labels.  |
| `validate_flow_plot_residual_bands` | #15 | `plot_validation.R` | `plot_validation.py` | test_validation_plot_contract.py | Python figure contracts: bands and summary contracts. |
| `validate_flow_plot_distribution` | #15 | `plot_validation.R` | `plot_validation.py` | test_validation_plot_contract.py | Python figure contracts: divergence series and ordering.  |
| `validate_flow_plot_distribution_pairwise` | #15 | `plot_validation.R` | `plot_validation.py` | test_validation_plot_contract.py | Python figure contracts: pairwise inputs and panels.  |
| `validate_flow_plot_structure` | #15 | `plot_validation.R` | `plot_validation.py` | test_validation_plot_contract.py | Python figure contracts: spatial summary panels.  |
| `validate_flow_plot_lisa_map` | #15 | `plot_validation.R` | `plot_validation.py` | test_lisa_plot.py; test_validation_plot_contract.py | Python figure contracts: cluster colours, significance, isolates, geometry and CRS alignment.  |

## API differences and scientific interpretation

`debiaspy_example_data` uses the Python package name. The Python public API uses
`validate_flow_overall` and `validate_flow_pairs`; R's two historical validation
aliases account for the export count difference together with the loader rename.
The formula interface uses `formula` and paired mobility/bias formulas. Supported
model combinations and input requirements are described in the API mapping and tutorials.

The Census example calibrates against benchmark margins. Independent benchmark
observations are required to assess accuracy beyond this calibration. The portable
CSV tests cover identifiers, leading zeros and source/time scope. Wider datasets
and platform combinations require additional verification.
