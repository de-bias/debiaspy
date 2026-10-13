"""Check companion loading and the geography-independent worked example."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "benchmark_od.py"
    spec = importlib.util.spec_from_file_location("benchmark_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_example_uses_arbitrary_area_keys_and_keeps_inputs() -> None:
    observed = pd.DataFrame(
        {
            "origin": ["zone-1", "zone-2"],
            "destination": ["zone-2", "zone-1"],
            "flow": [2.0, 3.0],
        }
    )
    benchmark = observed.assign(flow=[20.0, 12.0])
    coverage = pd.DataFrame(
        {
            "origin": ["zone-1", "zone-2"],
            "population": [100, 200],
            "user_count": [10, 50],
        }
    )
    original = observed.copy(deep=True)
    result = _example_module().analyse(observed, benchmark, coverage)
    np.testing.assert_allclose(result["adjusted_od"].flow_adj, [20, 12])
    assert result["diagnostics"].loc[0, "undefined_adjusted_rows"] == 0
    assert not result["metrics"].empty
    pd.testing.assert_frame_equal(observed, original)


def test_portable_example_retains_undefined_rows_and_withholds_metrics() -> None:
    observed = pd.DataFrame({"origin": ["A"], "destination": ["B"], "flow": [2.0]})
    benchmark = observed.assign(flow=20.0)
    coverage = pd.DataFrame({"origin": ["A"], "population": [100], "user_count": [0]})
    result = _example_module().analyse(observed, benchmark, coverage)
    assert len(result["adjusted_od"]) == 1
    assert np.isinf(result["adjusted_od"].flow_adj.iloc[0])
    assert result["diagnostics"].loc[0, "undefined_adjusted_rows"] == 1
    assert result["metrics"].empty


def test_portable_csv_command_preserves_identifiers_and_custom_columns(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {"from": ["001", "002"], "to": ["002", "001"], "trips": [2, 3]}
    ).to_csv(tmp_path / "observed.csv", index=False)
    pd.DataFrame(
        {"from": ["001", "002"], "to": ["002", "001"], "count": [20, 12]}
    ).to_csv(tmp_path / "benchmark.csv", index=False)
    pd.DataFrame(
        {"zone": ["001", "002"], "population": [100, 200], "user_count": [10, 50]}
    ).to_csv(tmp_path / "coverage.csv", index=False)
    script = Path(__file__).resolve().parents[1] / "examples" / "benchmark_od.py"
    output = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--observed",
            str(tmp_path / "observed.csv"),
            "--benchmark",
            str(tmp_path / "benchmark.csv"),
            "--coverage",
            str(tmp_path / "coverage.csv"),
            "--origin-column",
            "from",
            "--destination-column",
            "to",
            "--observed-flow-column",
            "trips",
            "--benchmark-flow-column",
            "count",
            "--coverage-area-column",
            "zone",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = pd.read_csv(
        output / "adjusted_od.csv", dtype={"origin": str, "destination": str}
    )
    assert result.origin.tolist() == ["001", "002"]
    assert result.destination.tolist() == ["002", "001"]
    np.testing.assert_allclose(result.flow_adj, [20, 12])
