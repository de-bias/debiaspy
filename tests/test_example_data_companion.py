"""Check companion loading and the geography-independent worked example."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from debiaspy import debiaspy_example_data, example_data


@pytest.mark.parametrize("geography", ["lad", "msoa"])
def test_python_companion_loads_empirical_tables_without_r(
    geography: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    companion = pytest.importorskip("debiaspydata")

    def unexpected_r(*args, **kwargs):
        raise AssertionError("The Python companion route must not invoke R")

    monkeypatch.setattr(subprocess, "run", unexpected_r)
    result = debiaspy_example_data(n_areas=5, geography=geography)
    raw = companion.load_dataset(f"census_{geography}_OD_travel2work")
    expected = (
        raw.loc[
            raw.origin.isin(result["coverage"].origin)
            & raw.destination.isin(result["coverage"].origin)
        ]
        .sort_values(["origin", "destination"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(result["benchmark_od"], expected, check_dtype=False)
    assert result["metadata"].loc[0, "data_package"] == "debiaspydata"
    assert "Census 2021" in result["metadata"].loc[0, "benchmark_source"]
    assert not result["distance"].empty
    assert not result["covariates"].empty


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


@pytest.mark.parametrize("asset", ["covariates", "centroids"])
def test_empirical_companion_requires_supporting_assets(
    asset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MissingAssets:
        @staticmethod
        def list_datasets():
            return ()

    monkeypatch.setattr(
        example_data, "_python_data_companion", lambda package: MissingAssets
    )
    with pytest.raises(ValueError, match="asset from debiaspydata"):
        if asset == "covariates":
            example_data._load_example_covariates("debiaspydata", ["A"], "lad")
        else:
            example_data._load_example_distance("debiaspydata", ["A"], True, "lad")


def test_removed_r_companion_mode_fails_without_invoking_r(monkeypatch):
    def unexpected_r(*args, **kwargs):
        raise AssertionError("Retired R loading must never run a subprocess")

    monkeypatch.setattr(subprocess, "run", unexpected_r)
    with pytest.raises(ValueError, match="Python companion"):
        debiaspy_example_data(data_package="debiasRdata", n_areas=2)


def test_centroids_require_all_selected_areas():
    centroids = pd.DataFrame({"area": ["A"], "longitude": [0.0], "latitude": [51.0]})
    with pytest.raises(ValueError, match="every selected area"):
        example_data._build_example_centroid_distance(
            centroids, ["A", "B"], True, "test"
        )


def test_companion_dataset_api_supports_assets_in_separate_locations(monkeypatch):
    class Companion:
        def list_datasets(self):
            return ("msoa_covariates", "msoa_centroids")

        def load_dataset(self, name):
            if name == "msoa_covariates":
                return pd.DataFrame({"area": ["A", "B"], "rural_pct": [0.2, 0.8]})
            return pd.DataFrame(
                {"area": ["A", "B"], "longitude": [0.0, 0.1], "latitude": [51.0, 51.1]}
            )

        def data_path(self, name):
            raise AssertionError(
                "Loading must use named datasets, not guessed file locations"
            )

    monkeypatch.setattr(
        example_data, "_python_data_companion", lambda package: Companion()
    )
    covariates = example_data._load_example_covariates(
        "debiaspydata", ["A", "B"], "msoa"
    )
    distance = example_data._load_example_distance(
        "debiaspydata", ["A", "B"], True, "msoa"
    )
    assert covariates["covariates"].area.tolist() == ["A", "B"]
    assert len(distance) == 4
    assert np.isfinite(distance.distance_km).all()
