# Issue-specific contribution format

Branch: `contribution/non-bayesian-parity`.
The implementation history follows #16 through #7, then grouped #4–#6.
The proposed review scopes below cover the deterministic, data and validation
work. Bayesian strategy remains a separate maintainer discussion under #16.

Use the existing `.github/pull_request_template.md`. Each proposal should carry
its relevant implementation, supporting code, tests and documentation. Shared
modules belong with the first proposal that needs them. The issue-ordered commits
provide review boundaries; their dependencies determine which changes can be
applied independently.

| Issue | Suggested title | Scope and dependencies | Main tests |
| --- | --- | --- | --- |
| #15 | Implement validation plots with debiasR figure contracts | Eight plotting functions; depends on validation result schemas from #11–#14. | test_validation_plot_contract.py; test_lisa_plot.py |
| #14 | Implement multi-method validation and margin helpers | Named results, method errors and scoped margins; depends on core validation #11–#13. | test_validate_flow_methods.py |
| #13 | Implement residual and spatial validation with R comparisons | Residual signs, global/Local Moran, weighted links and isolates; uses paired validation and shared spatial arithmetic. | test_flow_residual_structure.py; test_validation_core.py |
| #12 | Implement flow-distribution validation with R parity | Shares, KL/JSD and missing-value handling; uses scoped paired flows and shared distribution calculations. | test_validation_core.py; test_shared_distribution_metrics.py |
| #11 | Implement overall and paired flow validation | Scoped joins, metric denominators and missing values; foundation for #12–#15. | test_validation_core.py |
| #10 | Implement selected-method adjustment orchestration | Parameter forwarding and named outputs; depends on selected individual adjustment methods. Existing model dispatch follows its documented engine scope. | test_adjust_deterministic_methods.py; test_multilevel_frequentist_safety.py |
| #9 | Load companion data and support portable OD inputs | Named LAD/MSOA assets, CSV identity scope and empirical examples; portable example uses adjustment and validation functions. | test_example_data_companion.py; test_adjust_deterministic_methods.py |
| #8 | Implement coefficient calibration with Census parity | Calibration, input columns and numerical boundaries; uses shared adjustment inputs. | test_adjust_deterministic_methods.py |
| #7 | Implement scoped raking with Census parity | OD margins, zero cells, source/time identities; uses shared adjustment inputs. | test_adjust_deterministic_methods.py |
| #4–#6 | Implement inverse-penetration and selection-rate adjustments | Coverage weights, both selection methods, calibration and zero/missing values; shared adjustment inputs and live R comparisons. | test_adjust_inverse_penetration.py; test_adjust_deterministic_methods.py; test_r_adjustment_parity.py |

The [method evidence table](method-evidence.md) supplies source filenames and
checked scope. Use `Refs #NUMBER` while review or acceptance criteria remain
outstanding.

## Description outline

```markdown
## Summary

State the user-facing behaviour and the implemented change.
Include one concrete input/output example where useful.

## Related issue

Refs #NUMBER.

## Checklist

Use the repository template's checklist and tick completed checks.

## Notes for reviewers

R source: FILE.R at REFERENCE_COMMIT.
Python source: FILE.py.
Tests: TEST_FILE.py, command and result.
Evidence: live R comparisons / retained R outputs / Python contracts.
Scope: tested input cases, tolerances and remaining differences.
Dependencies: shared code and prerequisite implementations.
Documentation: method evidence table and relevant full-report section.

AI assistance: used for implementation and documentation.
Human review: NAME and reviewed scope, once completed.
```

Check each proposed subset with its required imports, tests and documentation.
The complete package's results are recorded in [package verification](parity-verification.md).
