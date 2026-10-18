#!/usr/bin/env python3
"""Compare debiaspy and debiasR adjust_raking_ratio on a fixed 2x2 fixture.

Usage:
    DEBIASR_ROOT=/path/to/debiasR uv run python scripts/verify_raking_ratio_r_parity.py

The script sources ``R/adjust_raking_ratio.R`` from ``DEBIASR_ROOT``, runs the
same deterministic OD example in Python, compares cells by origin/destination
keys, and writes:

- ``docs/reports/issue-7-raking-ratio-r-parity.md``
- ``docs/reports/issue-7-raking-ratio-r-parity.png``
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from debiaspy import __version__ as debiaspy_version
from debiaspy import adjust_raking_ratio

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "docs" / "reports"
REPORT_MD = REPORT_DIR / "issue-7-raking-ratio-r-parity.md"
REPORT_PNG = REPORT_DIR / "issue-7-raking-ratio-r-parity.png"

ORIGINS = ["A", "B"]
DESTINATIONS = ["X", "Y"]


def smoke_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mpd = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["X", "Y", "X", "Y"],
            "flow": [10.0, 20.0, 30.0, 40.0],
        }
    )
    origin_targets = pd.DataFrame({"origin": ["A", "B"], "target": [60.0, 40.0]})
    destination_targets = pd.DataFrame(
        {"destination": ["X", "Y"], "target": [50.0, 50.0]}
    )
    return mpd, origin_targets, destination_targets


def require_debiasr_root() -> Path:
    raw = os.environ.get("DEBIASR_ROOT")
    if not raw:
        raise SystemExit(
            "DEBIASR_ROOT is not set. Example:\n"
            "  DEBIASR_ROOT=/path/to/debiasR "
            "uv run python scripts/verify_raking_ratio_r_parity.py"
        )
    root = Path(raw).expanduser().resolve()
    source = root / "R" / "adjust_raking_ratio.R"
    if not source.is_file():
        raise SystemExit(
            f"Could not find R/adjust_raking_ratio.R under DEBIASR_ROOT={root}"
        )
    return root


def git_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def r_version() -> str:
    completed = subprocess.run(
        ["Rscript", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    text = (completed.stderr or completed.stdout).strip()
    return text.splitlines()[0] if text else "unavailable"


def run_r_adjust(
    debiasr_root: Path,
    mpd: pd.DataFrame,
    origin_targets: pd.DataFrame,
    destination_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, bool, int]:
    with tempfile.TemporaryDirectory(prefix="debiaspy-raking-parity-") as tmp:
        tmp_path = Path(tmp)
        mpd_path = tmp_path / "mpd.csv"
        origin_path = tmp_path / "origin_targets.csv"
        destination_path = tmp_path / "destination_targets.csv"
        result_path = tmp_path / "result.csv"
        meta_path = tmp_path / "meta.csv"

        mpd.to_csv(mpd_path, index=False)
        origin_targets.to_csv(origin_path, index=False)
        destination_targets.to_csv(destination_path, index=False)

        source = debiasr_root / "R" / "adjust_raking_ratio.R"
        r_code = f"""
suppressPackageStartupMessages({{
  library(dplyr)
  library(rlang)
  library(tibble)
}})
source({str(source)!r})
mpd <- read.csv({str(mpd_path)!r}, stringsAsFactors = FALSE)
origin_targets <- read.csv({str(origin_path)!r}, stringsAsFactors = FALSE)
destination_targets <- read.csv({str(destination_path)!r}, stringsAsFactors = FALSE)
res <- adjust_raking_ratio(
  mpd_od_df = mpd,
  origin_targets = origin_targets,
  destination_targets = destination_targets,
  max_iter = 200L,
  tol = 1e-10
)
write.csv(as.data.frame(res), {str(result_path)!r}, row.names = FALSE)
meta <- data.frame(
  ipf_converged = isTRUE(attr(res, "ipf_converged")),
  ipf_iterations = as.integer(attr(res, "ipf_iterations"))
)
write.csv(meta, {str(meta_path)!r}, row.names = FALSE)
"""
        completed = subprocess.run(
            ["Rscript", "-e", r_code],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise SystemExit(
                "Rscript failed while running adjust_raking_ratio().\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        result = pd.read_csv(result_path)
        meta = pd.read_csv(meta_path).iloc[0]
        return (
            result,
            bool(meta["ipf_converged"]),
            int(meta["ipf_iterations"]),
        )


def to_matrix(df: pd.DataFrame, value_col: str = "flow_adj") -> pd.DataFrame:
    keyed = df.copy()
    keyed["origin"] = keyed["origin"].astype(str)
    keyed["destination"] = keyed["destination"].astype(str)
    matrix = keyed.pivot_table(
        index="origin",
        columns="destination",
        values=value_col,
        aggfunc="sum",
    )
    return matrix.reindex(index=ORIGINS, columns=DESTINATIONS)


def relative_differences(python_vals: np.ndarray, r_vals: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(r_vals), 1e-12)
    rel = np.abs(python_vals - r_vals) / denom
    both_zero = (np.abs(python_vals) <= 1e-12) & (np.abs(r_vals) <= 1e-12)
    rel[both_zero] = 0.0
    return rel


def compute_metrics(
    python_df: pd.DataFrame,
    r_df: pd.DataFrame,
    python_converged: bool,
    python_iterations: int,
    r_converged: bool,
    r_iterations: int,
    origin_targets: pd.DataFrame,
    destination_targets: pd.DataFrame,
) -> dict[str, float | int | bool | str]:
    py_mat = to_matrix(python_df)
    r_mat = to_matrix(r_df)
    py_vals = py_mat.to_numpy(dtype=float)
    r_vals = r_mat.to_numpy(dtype=float)
    diff = py_vals - r_vals
    rel = relative_differences(py_vals, r_vals)

    py_origin = python_df.groupby("origin", sort=False)["flow_adj"].sum()
    r_origin = r_df.groupby(r_df["origin"].astype(str), sort=False)["flow_adj"].sum()
    py_dest = python_df.groupby("destination", sort=False)["flow_adj"].sum()
    r_dest = r_df.groupby(r_df["destination"].astype(str), sort=False)["flow_adj"].sum()

    origin_target_map = dict(
        zip(
            origin_targets["origin"].astype(str),
            origin_targets["target"],
            strict=True,
        )
    )
    dest_target_map = dict(
        zip(
            destination_targets["destination"].astype(str),
            destination_targets["target"],
            strict=True,
        )
    )

    return {
        "n_cells_python": int(len(python_df)),
        "n_cells_r": int(len(r_df)),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_rel_diff": float(np.max(rel)),
        "origin_margin_max_abs_diff_py_vs_r": float(
            np.max(
                np.abs(
                    np.array([py_origin[o] - r_origin[o] for o in ORIGINS], dtype=float)
                )
            )
        ),
        "destination_margin_max_abs_diff_py_vs_r": float(
            np.max(
                np.abs(
                    np.array(
                        [py_dest[d] - r_dest[d] for d in DESTINATIONS], dtype=float
                    )
                )
            )
        ),
        "origin_margin_max_abs_diff_py_vs_target": float(
            np.max(
                np.abs(
                    np.array(
                        [py_origin[o] - origin_target_map[o] for o in ORIGINS],
                        dtype=float,
                    )
                )
            )
        ),
        "destination_margin_max_abs_diff_py_vs_target": float(
            np.max(
                np.abs(
                    np.array(
                        [py_dest[d] - dest_target_map[d] for d in DESTINATIONS],
                        dtype=float,
                    )
                )
            )
        ),
        "python_ipf_converged": python_converged,
        "r_ipf_converged": r_converged,
        "python_ipf_iterations": python_iterations,
        "r_ipf_iterations": r_iterations,
        "python_matrix": py_mat,
        "r_matrix": r_mat,
        "diff_matrix": pd.DataFrame(diff, index=ORIGINS, columns=DESTINATIONS),
    }


def write_plot(metrics: dict[str, float | int | bool | str | pd.DataFrame]) -> None:
    r_mat = metrics["r_matrix"]
    py_mat = metrics["python_matrix"]
    diff_mat = metrics["diff_matrix"]
    assert isinstance(r_mat, pd.DataFrame)
    assert isinstance(py_mat, pd.DataFrame)
    assert isinstance(diff_mat, pd.DataFrame)

    vmax = max(
        float(np.nanmax(np.abs(r_mat.to_numpy()))),
        float(np.nanmax(np.abs(py_mat.to_numpy()))),
        1e-12,
    )
    dmax = max(float(np.nanmax(np.abs(diff_mat.to_numpy()))), 1e-12)

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    panels = (
        (axes[0], r_mat, "R flow_adj", "viridis", 0.0, vmax),
        (axes[1], py_mat, "Python flow_adj", "viridis", 0.0, vmax),
        (axes[2], diff_mat, "Python − R", "coolwarm", -dmax, dmax),
    )
    for axis, matrix, title, cmap, vmin, vmax_panel in panels:
        image = axis.imshow(
            matrix.to_numpy(dtype=float),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_panel,
            origin="upper",
        )
        axis.set_xticks(range(len(DESTINATIONS)), DESTINATIONS)
        axis.set_yticks(range(len(ORIGINS)), ORIGINS)
        axis.set_xlabel("destination")
        axis.set_ylabel("origin")
        axis.set_title(title)
        for i, origin in enumerate(ORIGINS):
            for j, destination in enumerate(DESTINATIONS):
                value = float(matrix.loc[origin, destination])
                axis.text(
                    j,
                    i,
                    f"{value:.4f}",
                    ha="center",
                    va="center",
                    color="white" if abs(value) > 0.55 * vmax_panel else "black",
                    fontsize=9,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    max_abs = metrics["max_abs_diff"]
    figure.suptitle(
        f"Issue #7 raking-ratio parity  |  max |Python−R| = {max_abs:.3e}",
        fontsize=12,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(REPORT_PNG, dpi=160)
    plt.close(figure)


def matrix_to_markdown(matrix: pd.DataFrame) -> str:
    headers = ["origin", *list(matrix.columns)]
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for origin, row in matrix.iterrows():
        values = [f"{float(row[col]):.8f}" for col in matrix.columns]
        lines.append("| " + " | ".join([str(origin), *values]) + " |")
    return "\n".join(lines)


def write_report(
    debiasr_revision: str,
    metrics: dict[str, float | int | bool | str | pd.DataFrame],
) -> None:
    command = (
        "DEBIASR_ROOT=/path/to/debiasR "
        "uv run python scripts/verify_raking_ratio_r_parity.py"
    )
    conclusion = (
        "Python and R agree on this unclipped 2×2 fixture to numerical "
        "tolerance: cell-wise differences are negligible, origin and "
        "destination margins match the shared targets, and both runs report "
        "IPF convergence."
        if (
            float(metrics["max_abs_diff"]) < 1e-6
            and float(metrics["origin_margin_max_abs_diff_py_vs_target"]) < 1e-6
            and float(metrics["destination_margin_max_abs_diff_py_vs_target"]) < 1e-6
            and bool(metrics["python_ipf_converged"])
            and bool(metrics["r_ipf_converged"])
        )
        else (
            "Python and R do not fully agree on this fixture under the "
            "recorded thresholds; inspect the metrics and difference panel."
        )
    )

    py_mat = metrics["python_matrix"]
    r_mat = metrics["r_matrix"]
    assert isinstance(py_mat, pd.DataFrame)
    assert isinstance(r_mat, pd.DataFrame)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(
        "\n".join(
            [
                "# Issue #7: R/Python raking-ratio parity",
                "",
                "## Reproduction",
                "",
                "```bash",
                command,
                "```",
                "",
                "## Environment",
                "",
                f"- Python: {sys.version.split()[0]} ({platform.platform()})",
                f"- debiaspy: {debiaspy_version}",
                f"- R: {r_version()}",
                "- DEBIASR_ROOT: `$DEBIASR_ROOT`",
                f"- debiasR revision: `{debiasr_revision}`",
                "- R source: `$DEBIASR_ROOT/R/adjust_raking_ratio.R`",
                "",
                "## Fixture",
                "",
                "Deterministic 2×2 OD table:",
                "",
                "| origin | destination | flow |",
                "| --- | --- | ---: |",
                "| A | X | 10 |",
                "| A | Y | 20 |",
                "| B | X | 30 |",
                "| B | Y | 40 |",
                "",
                "Origin targets: A=60, B=40. Destination targets: X=50, Y=50.",
                "Controls: `max_iter=200`, `tol=1e-10`, default clipping.",
                "Cells are compared by `origin`/`destination` keys, not row order.",
                "",
                "## Metrics",
                "",
                f"- Number of cells (Python / R): "
                f"{metrics['n_cells_python']} / {metrics['n_cells_r']}",
                f"- Max absolute difference (Python − R): "
                f"{metrics['max_abs_diff']:.6e}",
                f"- RMSE: {metrics['rmse']:.6e}",
                f"- Max relative difference "
                f"(|diff| / max(|R|, 1e-12); both-zero → 0): "
                f"{metrics['max_rel_diff']:.6e}",
                f"- Max |origin margin Python − R|: "
                f"{metrics['origin_margin_max_abs_diff_py_vs_r']:.6e}",
                f"- Max |destination margin Python − R|: "
                f"{metrics['destination_margin_max_abs_diff_py_vs_r']:.6e}",
                f"- Max |origin margin Python − target|: "
                f"{metrics['origin_margin_max_abs_diff_py_vs_target']:.6e}",
                f"- Max |destination margin Python − target|: "
                f"{metrics['destination_margin_max_abs_diff_py_vs_target']:.6e}",
                f"- IPF converged (Python / R): "
                f"{metrics['python_ipf_converged']} / {metrics['r_ipf_converged']}",
                f"- IPF iterations (Python / R): "
                f"{metrics['python_ipf_iterations']} / {metrics['r_ipf_iterations']}",
                "",
                "### Adjusted matrices",
                "",
                "R `flow_adj`:",
                "",
                matrix_to_markdown(r_mat),
                "",
                "Python `flow_adj`:",
                "",
                matrix_to_markdown(py_mat),
                "",
                "## Figure",
                "",
                "![R vs Python flow_adj and difference]"
                "(issue-7-raking-ratio-r-parity.png)",
                "",
                "## Conclusion",
                "",
                conclusion,
                "",
                "Deliberate Python differences outside this unclipped fixture "
                "remain: clipped weights are recomputed into `flow_adj`, "
                "clipping marks non-convergence, and invalid inputs raise "
                "explicit errors.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    debiasr_root = require_debiasr_root()
    debiasr_revision = git_revision(debiasr_root)
    mpd, origin_targets, destination_targets = smoke_fixture()

    python_result = adjust_raking_ratio(
        mpd_od_df=mpd,
        origin_targets=origin_targets,
        destination_targets=destination_targets,
        max_iter=200,
        tol=1e-10,
    )
    r_result, r_converged, r_iterations = run_r_adjust(
        debiasr_root, mpd, origin_targets, destination_targets
    )

    metrics = compute_metrics(
        python_df=python_result,
        r_df=r_result,
        python_converged=bool(python_result.attrs["ipf_converged"]),
        python_iterations=int(python_result.attrs["ipf_iterations"]),
        r_converged=r_converged,
        r_iterations=r_iterations,
        origin_targets=origin_targets,
        destination_targets=destination_targets,
    )
    write_plot(metrics)
    write_report(debiasr_revision, metrics)

    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_PNG}")
    print(f"max_abs_diff={metrics['max_abs_diff']:.6e}")
    print(f"rmse={metrics['rmse']:.6e}")
    print(f"max_rel_diff={metrics['max_rel_diff']:.6e}")


if __name__ == "__main__":
    main()
