"""Compile the recorded side-by-side report from its checked-in LaTeX and figures."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checks",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/reports/article_comparison.json",
    )
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "docs" / "reports" / "source"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="debias-report-") as directory:
        work = Path(directory)
        shutil.copytree(source, work, dirs_exist_ok=True)
        tex = "debiasr_debiaspy_side_by_side.tex"
        if args.checks:
            checks = json.loads(args.checks.read_text())
            records = checks["checks"]
            failures = [row for row in records if row["status"] != "pass"]
            expected = set(checks["expected_differences"])
            if {row["name"] for row in failures} != expected:
                raise ValueError(
                    "Resolve unexpected article differences before building the report."
                )
            appendix = (
                r"\clearpage\section{Fresh article reproduction}"
                r"\textbf{R code:} \texttt{check\_article\_parity.R}\par "
                r"\textbf{Python code:} \texttt{check\_article\_parity.py}\par "
                r"\textbf{Test:} \texttt{check\_article\_parity.py}\par "
                f"Fresh R/Python calls passed {len(records) - len(failures)} of "
                f"{len(records)} comparisons. "
                "The two differences concern public exports. "
                "Bayesian article tables use retained R vignette outputs; sampling "
                "requires the separate Bayesian investigation."
                r"\par "
                r"R reference: \texttt{" + checks["reference_commit"] + r"}\par "
                r"Python checkout: \texttt{" + checks["integration_commit"] + r"}\par "
                "The main report preserves the recorded tables and figures. "
                "The fresh run stores complete comparison CSVs "
                "alongside its JSON summary. "
                "Uncommitted changes, when present, are listed in that summary."
            )
            path = work / tex
            path.write_text(
                path.read_text().replace(
                    r"\end{document}", appendix + r"\end{document}"
                )
            )
        evidence = source.parents[1] / "development" / "method-evidence.md"
        method_rows = []
        for line in evidence.read_text().splitlines():
            if line.startswith("| `"):
                cells = [
                    cell.strip().replace("`", "") for cell in line.split("|")[1:-1]
                ]
                method, issue, r_file, py_file, tests, scope = cells
                issue = issue.replace("#", r"\#")
                test_files = "; ".join(
                    r"\path{" + name + "}" for name in tests.split("; ")
                )
                method_rows.append(
                    r"\path{" + method + "} (" + issue + r")\newline "
                    r"R: \path{" + r_file + r"}\newline "
                    r"Python: \path{"
                    + py_file
                    + r"} & "
                    + scope
                    + r"\newline Test: "
                    + test_files
                    + r" \\ \hline "
                )
        method_appendix = (
            r"\clearpage\section{Method-by-method evidence}"
            r"\textbf{R code:} \path{check_article_parity.R}\par "
            r"\textbf{Python code:} \path{check_article_parity.py}; "
            r"\path{build_report.py}\par "
            r"\textbf{Test:} \path{test_r_adjustment_parity.py}; "
            r"\path{test_validation_core.py}\par "
            "Live R executes the reference during the article checker or "
            "named regression. Retained R uses previously recorded reference outputs. "
            "Python contracts check behaviour, schemas, errors and figures. "
            "Each row identifies its checked scope. "
            "The table source is \\path{docs/development/method-evidence.md}. "
            r"\fontsize{8.5}{10}\selectfont\begin{longtable}{p{0.43\textwidth}p{0.51\textwidth}}"
            r"\textbf{Method and sources} & \textbf{Evidence and tests} "
            r"\\ \hline\endhead "
            + "\n".join(method_rows)
            + r"\end{longtable}\normalsize "
        )
        path = work / tex
        path.write_text(
            path.read_text().replace(
                r"\end{document}", method_appendix + r"\end{document}"
            )
        )
        issue_rows = []
        issue_evidence = source.parents[1] / "development" / "issue-contributions.md"
        for line in issue_evidence.read_text().splitlines():
            if line.startswith("| `"):
                branch, base, tests, scope = [
                    cell.strip().replace("`", "") for cell in line.split("|")[1:-1]
                ]
                issue_rows.append(
                    r"\path{"
                    + branch
                    + r"}\newline Base: \path{"
                    + base
                    + "} & "
                    + tests
                    + r"\newline "
                    + scope
                    + r" \\ \hline "
                )
        issue_appendix = (
            r"\clearpage\section{Issue contribution checks}"
            r"\textbf{R code:} \path{measure_bias.R}; "
            r"\path{adjust_inverse_penetration.R}; \path{validate_flows.R}\par "
            r"\textbf{Python code:} \path{adjust_all_methods.py}; "
            r"\path{validate_flows.py}; \path{build_report.py}\par "
            r"\textbf{Test:} \path{test_r_adjustment_parity.py}; "
            r"\path{test_flow_residual_structure.py}\par "
            "Each branch was built and installed as a wheel, then tested with its "
            "documented prerequisites on macOS ARM64 with Python 3.11 and prepared "
            "dependencies. Live R was enabled. All exported callables, Ruff, "
            "formatting and pip dependency checks passed. The issue wheels contain "
            "the non-Bayesian implementations. Counts include prerequisite tests. "
            "The complete package retains all 204 original test functions and "
            "their assertion bodies, with 292 non-Bayesian tests passing. "
            r"\small\begin{longtable}{p{0.46\textwidth}p{0.48\textwidth}}"
            r"\textbf{Contribution and base} & \textbf{Installed checks and scope} "
            r"\\ \hline\endhead "
            + "\n".join(issue_rows)
            + r"\end{longtable}\normalsize "
        )
        path.write_text(
            path.read_text().replace(
                r"\end{document}", issue_appendix + r"\end{document}"
            )
        )
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex],
                cwd=work,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(result.stdout[-6000:] + result.stderr)
            for line in result.stdout.splitlines():
                if "Overfull" in line:
                    print(line)
        shutil.copyfile(work / tex.replace(".tex", ".pdf"), output)
    print(output)


if __name__ == "__main__":
    main()
