# Suggested contribution format

The branch is `contribution/r-parity`. No pull request has been created.
Its implementation commits follow GitHub's displayed issue order: #16, #15,
#14, #13, #12, #11, #10, #9, #8, #7, then grouped #4–#6. Documentation and
reproducible evidence follow in a separate commit. Shared dependencies appear
with their first dependent implementation; public exports become available
as supporting functions are introduced.

Use the existing `.github/pull_request_template.md` for a future proposal.
Prefer an issue-specific title, such as:

- `Implement flow-distribution validation with debiasR parity (#12)`
- `Add inverse-penetration and selection-rate adjustments (#4–#6)`
- `Propose Bayesian backends and document current parity limits (#16)`

For a proposal covering the complete branch, use
`Implement debiasR flow adjustments and validation with parity evidence`.
The complete branch is substantial; maintainers can review its issue commits
or request separate focused proposals.

## Description outline

```markdown
## Summary

Describe the implemented behaviour and the user-facing problem it addresses.
Give one concrete before/after example where useful.

## Related issue

Refs #NUMBER. List grouped issues when applicable.

## Checklist

- [ ] Proposed through a branch and PR under the repository workflow
- [ ] Relevant pytest checks run, with counts and skipped cases stated below
- [ ] Ruff and formatting checks pass
- [ ] User-facing documentation updated
- [ ] Numerical, schema, missing-value and zero-value behaviour tested
- [ ] Corresponding debiasR code and tests inspected

## Notes for reviewers

R source: FILE.R at REFERENCE_COMMIT.
Python source: FILE.py.
Tests: TEST_FILE.py; commands and results.

Explain deliberate API differences and unsupported options.
Link the full parity document and its reproduction instructions.
State which comparisons run R afresh and which use retained references.
For Bayesian work, distinguish execution, shared-draw prediction agreement,
posterior precision and model validity.

AI assistance was used. Named human review of code, tests and documentation
remains pending; record the contributor's name after that review occurs.
```

Current branch evidence is recorded in `integration-review.md` and the full
side-by-side report. The Bayesian strategy requires maintainer agreement;
issue #16 remains open. Future descriptions should use `Refs` while acceptance
criteria or review remain outstanding.
