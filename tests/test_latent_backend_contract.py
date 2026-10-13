from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from debiaspy import _latent
from debiaspy._exceptions import DebiasSchemaError
from debiaspy._latent import fit_latent_cmdstan, prepare_latent_state
from debiaspy.stan import latent_two_level_stan_path


def _prepared_rows(
    sources: tuple[str, ...] = ("src1",),
    times: tuple[str, ...] = ("t1",),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    areas = ("A", "B")
    for origin_index, origin in enumerate(areas, start=1):
        for destination_index, destination in enumerate(areas, start=1):
            for source_index, source in enumerate(sources, start=1):
                for time_index, time in enumerate(times, start=1):
                    rows.append(
                        {
                            "origin": origin,
                            "destination": destination,
                            "mpd_source": source,
                            "mpd_time": time,
                            "flow": 3
                            + origin_index
                            + destination_index
                            + source_index
                            + time_index,
                            "mpd_observed": True,
                            "rural_pct_o": 0.2 * origin_index,
                            "rural_pct_d": 0.2 * destination_index,
                            "log_distance": np.log(
                                abs(origin_index - destination_index) + 1
                            ),
                            "bias_e_origin": 0.1 + 0.02 * origin_index,
                            "log_pop_o": np.log(100 + 10 * origin_index),
                            "log_pop_d": np.log(100 + 10 * destination_index),
                        }
                    )
    return pd.DataFrame(rows)


def _scenario(scenario: str) -> dict[str, Any]:
    repeated = {
        "s1": "none",
        "s2": "time",
        "s3": "source",
        "s4": "source_time",
    }[scenario]
    return {"scenario": scenario, "repeated_observation": repeated}


def _latent_prediction_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _prepared_rows(("src1", "src2"), ("t1", "t2"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        latent = prepare_latent_state(data, _scenario("s4"), "od_time")["data"]
    latent = latent.copy()
    latent["observation_probability"] = np.where(latent["origin"].eq("A"), 0.10, 0.12)
    latent.loc[latent.index[-1], "mpd_observed"] = False
    prediction = latent.reset_index(drop=True)
    fit = prediction.loc[prediction["mpd_observed"]].reset_index(drop=True)
    return fit, prediction


def _controls() -> dict[str, float | int]:
    return {
        "latent_coef_prior_scale": 1.0,
        "latent_bias_prior_scale": 0.5,
        "latent_intercept_prior_scale": 2.5,
        "latent_state_prior_scale": 0.75,
        "latent_source_prior_scale": 0.5,
        "latent_time_prior_scale": 0.5,
        "latent_phi_prior_rate": 1.0,
        "latent_adapt_delta": 0.95,
        "latent_max_treedepth": 12,
        "latent_rng_eta_max": 20.0,
    }


def _split_formula() -> dict[str, Any]:
    return {
        "interface": "split",
        "mobility_variables": [
            "rural_pct_o",
            "rural_pct_d",
            "log_distance",
        ],
        "bias_variables": ["bias_e_origin"],
        "source": "user",
    }


def test_packaged_stan_program_is_byte_identical_to_debiasr():
    packaged = latent_two_level_stan_path().read_bytes()

    configured_root = os.environ.get("DEBIASR_ROOT")
    if configured_root is None:
        pytest.skip("Set DEBIASR_ROOT for the optional Stan source parity check")
    debiasr_root = Path(configured_root)
    reference = debiasr_root / "inst" / "stan" / "latent_two_level.stan"
    assert reference.is_file()
    assert packaged == reference.read_bytes()


@pytest.mark.parametrize(
    ("scenario", "sources", "times", "unit", "n_latent", "weak"),
    [
        ("s1", ("src1",), ("t1",), "od", 4, True),
        ("s2", ("src1",), ("t1", "t2"), "od_time", 8, True),
        ("s3", ("src1", "src2"), ("t1",), "od", 4, False),
        ("s4", ("src1", "src2"), ("t1", "t2"), "od_time", 8, False),
    ],
)
def test_latent_state_auto_contract_across_scenarios(
    scenario,
    sources,
    times,
    unit,
    n_latent,
    weak,
):
    data = _prepared_rows(sources, times)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = prepare_latent_state(data, _scenario(scenario), "auto")

    assert result["latent_flow_unit"] == unit
    assert result["n_latent_flows"] == n_latent
    assert result["identifiability"]["weak_identification_warning"] is weak
    group_columns = ["origin", "destination"]
    if unit == "od_time":
        group_columns.append("mpd_time")
    assert (
        result["data"]
        .groupby(group_columns, observed=True)["latent_flow_id"]
        .nunique()
        .eq(1)
        .all()
    )


def test_unbalanced_s4_auto_falls_back_to_od():
    data = _prepared_rows(("src1", "src2"), ("t1", "t2"))
    data = data.loc[
        ~(data["mpd_source"].eq("src2") & data["mpd_time"].eq("t2"))
    ].reset_index(drop=True)

    with pytest.warns(UserWarning, match="selected `od`"):
        result = prepare_latent_state(data, _scenario("s4"), "auto")

    assert result["latent_flow_unit"] == "od"
    assert result["identifiability"]["min_sources_per_latent_flow"] == 2
    assert result["identifiability"]["weak_identification_warning"] is False


def test_stan_data_matches_r_dimensions_standardization_and_bases():
    fit, prediction = _latent_prediction_data()
    formula = _latent._build_latent_formula_info(
        _split_formula(),
        default_covariate_col="rural_pct",
        include_pop_terms=True,
    )
    info = _latent._build_latent_stan_data(
        fit,
        prediction,
        formula,
        _scenario("s4"),
        "poisson",
        _controls(),
    )
    data = info["data"]
    summary = info["summary"]

    assert data["N_obs"] == 15
    assert data["N_pred"] == 16
    assert data["L"] == 8
    assert data["S"] == 2
    assert data["T"] == 2
    assert data["use_time_effect"] == 1
    assert data["use_negbin"] == 0
    assert data["true_intercept_col"] == 1
    assert data["prior_coef_scale"] == 1
    assert data["prior_bias_scale"] == 0.5
    assert data["intercept_scale"] == 2.5
    assert data["prior_latent_state_scale"] == 0.75
    assert data["prior_source_scale"] == 0.5
    assert data["prior_time_scale"] == 0.5
    assert data["phi_prior_rate"] == 1
    assert data["max_rng_eta"] == 20
    assert summary["true_formula"] == ("~ rural_pct_o + rural_pct_d + log_distance")
    assert summary["observation_formula"] == "~ bias_e_origin - 1"
    assert summary["n_unobserved_latent_flows"] == 0

    true_non_intercept = [
        index
        for index, term in enumerate(summary["true_terms"])
        if term != "(Intercept)"
    ]
    np.testing.assert_allclose(
        data["X_true_obs"][:, true_non_intercept].mean(axis=0),
        0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        data["X_true_obs"][:, true_non_intercept].std(axis=0, ddof=1),
        1,
        atol=1e-12,
    )
    np.testing.assert_allclose(data["X_bias_obs"].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(data["X_bias_obs"].std(axis=0, ddof=1), 1, atol=1e-12)
    for basis_name in ("latent_basis", "source_basis", "time_basis"):
        basis = data[basis_name]
        np.testing.assert_allclose(basis.sum(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-12)


def test_latent_stan_data_rejects_fractional_flows():
    fit, prediction = _latent_prediction_data()
    fit["flow"] = fit["flow"].astype(float)
    fit.loc[0, "flow"] = 1.5
    formula = _latent._build_latent_formula_info(_split_formula(), "rural_pct", True)

    with pytest.raises(DebiasSchemaError, match="count-valued"):
        _latent._build_latent_stan_data(
            fit,
            prediction,
            formula,
            _scenario("s4"),
            "poisson",
            _controls(),
        )


def test_cmdstan_runtime_rejects_versions_that_cannot_compile_pinned_source(
    monkeypatch,
):
    fake = SimpleNamespace(
        cmdstan_path=lambda: "/tmp/cmdstan-2.33.1",
        cmdstan_version=lambda: (2, 33, 1),
    )
    monkeypatch.setattr(_latent.importlib, "import_module", lambda name: fake)

    with pytest.raises(RuntimeError, match=r"requires CmdStan 2\.32\.x"):
        _latent._load_compatible_cmdstanpy()


class _FakeFit:
    def __init__(self, stan_data: dict[str, Any]):
        self.stan_data = stan_data
        self.draw_count = 8

    def stan_variable(self, name: str) -> np.ndarray:
        widths = {
            "flow_true_pred": self.stan_data["N_pred"],
            "mu_mpd_pred": self.stan_data["N_pred"],
            "beta_true": self.stan_data["K_true"],
            "beta_bias": self.stan_data["K_bias"],
            "source_effect": self.stan_data["S"],
            "time_effect": self.stan_data["T"],
        }
        if name == "phi":
            return np.linspace(1.0, 1.7, self.draw_count)
        width = widths[name]
        values = np.arange(self.draw_count * width, dtype=float).reshape(
            self.draw_count, width
        )
        return 1.0 + values / 100

    def method_variables(self) -> dict[str, np.ndarray]:
        shape = (2, 4)
        return {
            "divergent__": np.zeros(shape),
            "treedepth__": np.full(shape, 5.0),
            "accept_stat__": np.full(shape, 0.97),
            "energy__": np.array([[1.0, 1.2, 1.1, 1.3], [1.4, 1.6, 1.7, 1.8]]),
        }

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Mean": [1.0, 1.0],
                "MCSE": [0.01, 0.01],
                "StdDev": [0.5, 0.5],
                "ESS_bulk": [800.0, 800.0],
                "ESS_tail": [750.0, 750.0],
                "R_hat": [1.001, 1.001],
            },
            index=["beta_true[1]", "flow_true_pred[1]"],
        )


class _FakeModel:
    created_path: str | None = None
    sample_kwargs: dict[str, Any] | None = None

    def __init__(self, stan_file: str):
        type(self).created_path = stan_file

    def sample(self, **kwargs):
        type(self).sample_kwargs = kwargs
        return _FakeFit(kwargs["data"])


def test_fit_latent_cmdstan_maps_r_iterations_extracts_and_diagnoses(
    monkeypatch,
    tmp_path,
):
    fake_module = SimpleNamespace(CmdStanModel=_FakeModel)
    monkeypatch.setattr(
        _latent,
        "_load_compatible_cmdstanpy",
        lambda: (fake_module, "2.32.2"),
    )
    monkeypatch.setenv("DEBIASPY_CMDSTAN_CACHE", str(tmp_path))
    fit, prediction = _latent_prediction_data()

    result = fit_latent_cmdstan(
        fit_df=fit,
        prediction_df=prediction,
        formula_info=_split_formula(),
        default_covariate_col="rural_pct",
        include_pop_terms=True,
        scenario_info=_scenario("s4"),
        model_family="poisson",
        controls=_controls(),
        iter=1000,
        chains=4,
        seed=123,
        refresh=0,
    )

    compiled_source = Path(_FakeModel.created_path or "")
    assert compiled_source.is_relative_to(tmp_path)
    assert compiled_source.read_bytes() == latent_two_level_stan_path().read_bytes()
    kwargs = _FakeModel.sample_kwargs
    assert kwargs is not None
    assert kwargs["iter_warmup"] == 500
    assert kwargs["iter_sampling"] == 500
    assert kwargs["chains"] == 4
    assert kwargs["parallel_chains"] == 4
    assert kwargs["seed"] == 123
    assert kwargs["adapt_delta"] == 0.95
    assert kwargs["max_treedepth"] == 12
    assert result["backend"] == "stan_latent"
    assert result["runtime_backend"] == "cmdstanpy"
    assert result["cmdstan_version"] == "2.32.2"
    assert (
        Path(_FakeModel.created_path).read_bytes()
        == latent_two_level_stan_path().read_bytes()
    )
    assert result["flow_true_pred_draws"].shape == (8, len(prediction))
    assert result["flow_mpd_pred_draws"].shape == (8, len(prediction))
    assert set(result["coefficients"]["layer"]) == {
        "true_flow",
        "observation_bias",
        "observation_source_effect",
        "observation_time_effect",
        "overdispersion",
    }
    convergence = result["diagnostics"]["convergence"]
    assert convergence["n_post_warmup_draws"] == 8
    assert convergence["divergences"] == 0
    assert convergence["treedepth_hits"] == 0
    assert convergence["rhat_max"] == pytest.approx(1.001)
    assert convergence["ess_bulk_min"] == pytest.approx(800)
    assert convergence["ess_tail_min"] == pytest.approx(750)
    assert convergence["mcse_over_sd_max"] == pytest.approx(0.02)


def test_cmdstan_source_cache_reuses_bytes_and_refreshes_changed_source(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEBIASPY_CMDSTAN_CACHE", str(tmp_path / "cache"))
    source = tmp_path / "model.stan"
    source.write_text("parameters {real x;} model {x ~ normal(0,1);}")
    target = _latent._materialize_cmdstan_source(source, "2.32.2")
    timestamp = target.stat().st_mtime_ns
    assert _latent._materialize_cmdstan_source(source, "2.32.2") == target
    assert target.stat().st_mtime_ns == timestamp
    source.write_text("parameters {real x;} model {x ~ normal(0,2);}")
    assert (
        _latent._materialize_cmdstan_source(source, "2.32.2").read_bytes()
        == source.read_bytes()
    )
