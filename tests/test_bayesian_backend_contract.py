# pyright: reportArgumentType=false, reportGeneralTypeIssues=false
from __future__ import annotations

import importlib
import os
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from debiaspy import _bayesian as bayes
from debiaspy import adjust_multilevel_bayes
from debiaspy._exceptions import DebiasSchemaError


class FakePosterior(dict[str, np.ndarray]):
    @property
    def data_vars(self) -> list[str]:
        return list(self)


class FakeInferenceData:
    def __init__(self) -> None:
        self.posterior = FakePosterior(
            {
                "Intercept": np.asarray([[[1.0, 2.0, 3.0, 4.0]]]).reshape(1, 4),
                "log_distance": np.asarray([[[-0.4, -0.3, -0.2, -0.1]]]).reshape(1, 4),
                "bias_e_origin": np.asarray([[[0.1, 0.2, 0.3, 0.4]]]).reshape(1, 4),
                "alpha": np.asarray([[[1.0, 1.1, 1.2, 1.3]]]).reshape(1, 4),
            }
        )
        self.sample_stats = {
            "diverging": np.zeros((1, 4), dtype=bool),
            "reached_max_treedepth": np.zeros((1, 4), dtype=bool),
            "tree_depth": np.asarray([[2, 3, 2, 3]]),
            "acceptance_rate": np.asarray([[0.91, 0.93, 0.92, 0.94]]),
        }


class FakePrior:
    def __init__(self, name: str, **kwargs: object) -> None:
        self.name = name
        self.kwargs = kwargs


class FakeModel:
    instances: list[FakeModel] = []

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        **kwargs: object,
    ) -> None:
        self.formula_text = formula
        self.fit_data = data.copy()
        self.family_name = family
        self.model_kwargs = kwargs
        self.fit_kwargs: dict[str, object] = {}
        self.prediction_calls: list[tuple[pd.DataFrame, dict[str, object]]] = []
        self.family = SimpleNamespace(likelihood=SimpleNamespace(parent="mu"))
        self.components = {
            "mu": SimpleNamespace(
                common_terms={
                    "Intercept": object(),
                    "log_distance": object(),
                    "bias_e_origin": object(),
                }
            )
        }
        self.instances.append(self)

    def fit(self, **kwargs: object) -> FakeInferenceData:
        self.fit_kwargs = kwargs
        return FakeInferenceData()

    def predict(
        self,
        inference_data: FakeInferenceData,
        **kwargs: object,
    ) -> SimpleNamespace:
        del inference_data
        data = kwargs["data"]
        assert isinstance(data, pd.DataFrame)
        self.prediction_calls.append((data.copy(), kwargs.copy()))
        scales = np.asarray([8.0, 10.0, 12.0, 14.0])[:, None]
        if bool(data["log_observation_probability"].notna().all()):
            row_scale = np.exp(data["log_observation_probability"].to_numpy(float))
        else:
            row_scale = 1 + data["bias_e_origin"].to_numpy(float)
        mu = scales * row_scale[None, :]
        posterior = FakePosterior(
            {
                "mu": mu.reshape(1, 4, len(data)),
                "unconditional_mean": (mu / 10).reshape(1, 4, len(data)),
            }
        )
        return SimpleNamespace(posterior=posterior)


class FakeBambi:
    Model = FakeModel
    Prior = FakePrior


class FakeArviz:
    @staticmethod
    def summary(
        _idata: FakeInferenceData,
        kind: str,
        round_to: str,
    ) -> pd.DataFrame:
        assert round_to == "none"
        if kind == "diagnostics":
            return pd.DataFrame(
                {
                    "mcse_mean": [0.0123456789, 0.0234567891],
                    "ess_bulk": [500.1234567, 450.2345678],
                    "ess_tail": [480.3456789, 430.4567891],
                    "r_hat": [1.001234567, 1.004345678],
                },
                index=["Intercept", "log_distance"],
            )
        assert kind == "stats"
        return pd.DataFrame(
            {"sd": [0.5, 0.4]},
            index=["Intercept", "log_distance"],
        )

    @staticmethod
    def bfmi(_idata: FakeInferenceData) -> np.ndarray:
        return np.asarray([0.71])


@pytest.fixture
def inputs() -> dict[str, pd.DataFrame]:
    mpd = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "flow": [12.0, 4.0, 5.0, 10.0],
        }
    )
    coverage = pd.DataFrame(
        {
            "origin": ["A", "B"],
            "population": [100.0, 80.0],
            "user_count": [20.0, 8.0],
        }
    )
    covariates = pd.DataFrame(
        {
            "area": ["A", "B"],
            "population": [100.0, 80.0],
            "rural_pct": [0.2, 0.8],
        }
    )
    distance = mpd.loc[:, ["origin", "destination"]].copy()
    distance["distance_km"] = [1.0, 2.0, 2.0, 1.0]
    return {
        "mpd_od_df": mpd,
        "coverage_df": coverage,
        "covariates_df": covariates,
        "distance_df": distance,
    }


@pytest.fixture(autouse=True)
def reset_fake_models() -> None:
    FakeModel.instances.clear()


def install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bayes,
        "_load_bayesian_dependencies",
        lambda: (FakeBambi, FakeArviz),
    )


def test_backend_and_family_routing_contract() -> None:
    assert bayes._resolve_logical_backend("poisson", "auto") == "rstanarm"
    assert bayes._resolve_logical_backend("negbin", "auto") == "rstanarm"
    assert bayes._resolve_logical_backend("zip", "auto") == "brms"
    assert bayes._resolve_logical_backend("zinb", "auto") == "brms"
    assert bayes._BAMBI_FAMILIES == {
        "poisson": "poisson",
        "negbin": "negativebinomial",
        "zip": "zero_inflated_poisson",
        "zinb": "zero_inflated_negativebinomial",
    }
    with pytest.raises(DebiasSchemaError, match="requires `backend = 'brms'`"):
        bayes._resolve_logical_backend("zip", "rstanarm")
    with pytest.raises(DebiasSchemaError, match="latent_two_level"):
        bayes._resolve_logical_backend("poisson", "stan_latent")


@pytest.mark.parametrize(
    ("iterations", "expected"),
    [(2, (1, 1)), (99, (49, 50)), (1000, (500, 500))],
)
def test_r_iter_maps_to_tune_and_retained_draws(
    iterations: int,
    expected: tuple[int, int],
) -> None:
    assert (
        bayes._sampling_schedule(iterations, chains=2, seed=123, refresh=0) == expected
    )


@pytest.mark.parametrize("name", ["iter", "chains", "seed", "refresh"])
def test_sampler_controls_reject_logical_values(name: str) -> None:
    values = {"iter": 100, "chains": 2, "seed": 123, "refresh": 0}
    values[name] = True
    with pytest.raises(DebiasSchemaError, match=name):
        bayes._sampling_schedule(**values)


def test_prior_contract_records_nonidentity_and_negbin_override() -> None:
    overrides, contract = bayes._prior_contract(FakeBambi, "rstanarm", "negbin")

    assert overrides["alpha"].name == "Exponential"
    assert overrides["alpha"].kwargs == {"auto_scale": False, "lam": 1.0}
    assert contract["overrides"] == {"alpha": "Exponential(lam=1)"}
    assert contract["identity_claim"] is False

    prior_data = pd.DataFrame(
        {
            "flow": [0.0, 8.0, 15.0],
            "log_observation_probability": [np.nan, np.nan, np.nan],
        }
    )
    zip_overrides, zip_contract = bayes._prior_contract(
        FakeBambi, "brms", "zip", prior_data
    )
    assert set(zip_overrides) == {"common", "Intercept", "psi"}
    assert zip_overrides["common"].name == "Flat"
    assert zip_overrides["Intercept"].name == "StudentT"
    assert zip_overrides["Intercept"].kwargs == {
        "auto_scale": False,
        "nu": 3.0,
        "mu": 2.1,
        "sigma": 2.5,
    }
    assert zip_overrides["psi"].name == "Beta"
    assert zip_contract["prediction_parameter"].startswith("mu")
    assert zip_contract["prior_parameter_identity"] is True

    negbin_overrides, _ = bayes._prior_contract(FakeBambi, "brms", "negbin", prior_data)
    assert set(negbin_overrides) == {"common", "Intercept", "alpha"}
    assert negbin_overrides["alpha"].name == "InverseGamma"
    assert negbin_overrides["alpha"].kwargs == {
        "auto_scale": False,
        "alpha": 0.4,
        "beta": 0.3,
    }


def test_brms_intercept_prior_applies_the_mean_log_offset() -> None:
    data = pd.DataFrame(
        {
            "flow": [0.0, 8.0, 15.0],
            "log_observation_probability": np.log([0.1, 0.2, 0.4]),
        }
    )

    location, scale = bayes._brms_intercept_prior(data, "true_flow")

    assert location == pytest.approx(2.1 - np.log([0.1, 0.2, 0.4]).mean())
    assert scale == 2.5


def test_rstanarm_marginal_prior_scales_use_sample_standard_deviation() -> None:
    class PriorModel:
        def __init__(self) -> None:
            self.center_predictors = True
            self.family = SimpleNamespace(likelihood=SimpleNamespace(parent="mu"))
            self.components = {
                "mu": SimpleNamespace(
                    common_terms={
                        "x": SimpleNamespace(data=np.asarray([1.0, 2.0, 3.0, 4.0]))
                    }
                )
            }
            self.priors = {}

        def set_priors(self, priors):
            self.priors = priors

    model = PriorModel()

    contract = bayes._apply_rstanarm_marginal_priors(FakeBambi, model)

    expected_slope = 2.5 / np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
    assert model.priors["x"].kwargs["sigma"] == pytest.approx(expected_slope)
    assert model.priors["Intercept"].kwargs["sigma"] == pytest.approx(2.5)
    assert model.priors["Intercept"].kwargs["auto_scale"] is False
    assert contract["rstanarm_marginal_prior_scales"] == {
        "x": pytest.approx(expected_slope),
        "Intercept_centered": 2.5,
    }
    assert contract["fixed_effect_prior_identity"] is True
    assert contract["prior_marginal_parameter_identity"] is True


def test_rstanarm_single_random_intercept_uses_decov_scale_prior() -> None:
    class PriorModel:
        def __init__(self) -> None:
            self.center_predictors = True
            self.family = SimpleNamespace(likelihood=SimpleNamespace(parent="mu"))
            self.components = {
                "mu": SimpleNamespace(
                    common_terms={},
                    group_specific_terms={"1|origin": SimpleNamespace()},
                )
            }
            self.priors = {}

        def set_priors(self, priors):
            self.priors = priors

    model = PriorModel()

    contract = bayes._apply_rstanarm_marginal_priors(FakeBambi, model)

    group_prior = model.priors["1|origin"]
    assert group_prior.name == "Normal"
    assert group_prior.kwargs["auto_scale"] is False
    assert group_prior.kwargs["mu"] == 0
    scale_prior = group_prior.kwargs["sigma"]
    assert isinstance(scale_prior, FakePrior)
    assert scale_prior.name == "Exponential"
    assert scale_prior.kwargs == {"auto_scale": False, "lam": 1.0}
    assert contract["group_specific_prior_identity"] is True
    assert contract["prior_parameter_identity"] is True


def test_rstanarm_random_slope_group_fails_closed() -> None:
    class PriorModel:
        def __init__(self) -> None:
            self.center_predictors = True
            self.family = SimpleNamespace(likelihood=SimpleNamespace(parent="mu"))
            self.components = {
                "mu": SimpleNamespace(
                    common_terms={},
                    group_specific_terms={
                        "1|origin": SimpleNamespace(),
                        "x|origin": SimpleNamespace(),
                    },
                )
            }
            self.priors = {}

        def set_priors(self, priors):
            self.priors = priors

    model = PriorModel()

    with pytest.raises(NotImplementedError, match="single random intercepts"):
        bayes._apply_rstanarm_marginal_priors(FakeBambi, model)

    assert model.priors == {}


def test_brms_single_random_intercept_uses_half_student_t_sd_prior() -> None:
    class PriorModel:
        def __init__(self) -> None:
            self.family = SimpleNamespace(likelihood=SimpleNamespace(parent="mu"))
            self.components = {
                "mu": SimpleNamespace(
                    group_specific_terms={"1|origin": SimpleNamespace()}
                )
            }
            self.priors = {}

        def set_priors(self, priors):
            self.priors = priors

    model = PriorModel()

    contract = bayes._apply_brms_group_priors(FakeBambi, model)

    group_prior = model.priors["1|origin"]
    assert group_prior.name == "Normal"
    scale_prior = group_prior.kwargs["sigma"]
    assert scale_prior.name == "HalfStudentT"
    assert scale_prior.kwargs == {
        "auto_scale": False,
        "nu": 3.0,
        "sigma": 2.5,
    }
    assert contract["prior_parameter_identity"] is True


def test_brms_random_slope_fails_closed() -> None:
    model = SimpleNamespace(
        family=SimpleNamespace(likelihood=SimpleNamespace(parent="mu")),
        components={
            "mu": SimpleNamespace(
                group_specific_terms={
                    "1|origin": SimpleNamespace(),
                    "x|origin": SimpleNamespace(),
                }
            )
        },
        set_priors=lambda priors: None,
    )

    with pytest.raises(NotImplementedError, match="single random intercepts"):
        bayes._apply_brms_group_priors(FakeBambi, model)


@pytest.mark.skipif(find_spec("bambi") is None, reason="Bambi is not installed")
def test_real_bambi_resolves_centered_and_single_random_intercept_priors() -> None:
    bmb = importlib.import_module("bambi")
    data = pd.DataFrame(
        {
            "flow": [1, 2, 3, 4, 5, 6],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "origin": ["A", "A", "A", "B", "B", "B"],
        }
    )
    model = bmb.Model(
        "flow ~ x + (1|origin)",
        data,
        family="poisson",
        auto_scale=True,
    )

    contract = bayes._apply_rstanarm_marginal_priors(bmb, model)

    component = model.components["mu"]
    assert model.center_predictors is True
    intercept_prior = component.intercept_term.prior
    assert intercept_prior.name == "Normal"
    assert intercept_prior.auto_scale is False
    assert float(intercept_prior.args["mu"]) == 0
    assert float(intercept_prior.args["sigma"]) == pytest.approx(2.5)
    slope_prior = component.common_terms["x"].prior
    expected_slope = 2.5 / np.std(component.common_terms["x"].data, ddof=1)
    assert float(slope_prior.args["sigma"]) == pytest.approx(expected_slope)
    group_prior = component.group_specific_terms["1|origin"].prior
    assert group_prior.name == "Normal"
    assert group_prior.auto_scale is False
    scale_prior = group_prior.args["sigma"]
    assert scale_prior.name == "Exponential"
    assert scale_prior.auto_scale is False
    assert float(scale_prior.args["lam"]) == pytest.approx(1.0)
    assert contract["prior_parameter_identity"] is True


@pytest.mark.skipif(find_spec("bambi") is None, reason="Bambi is not installed")
def test_real_bambi_rstanarm_random_slope_fails_closed() -> None:
    bmb = importlib.import_module("bambi")
    data = pd.DataFrame(
        {
            "flow": [1, 2, 3, 4, 5, 6],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "origin": ["A", "A", "A", "B", "B", "B"],
        }
    )
    model = bmb.Model(
        "flow ~ x + (1 + x|origin)",
        data,
        family="poisson",
        auto_scale=True,
    )

    with pytest.raises(NotImplementedError, match="single random intercepts"):
        bayes._apply_rstanarm_marginal_priors(bmb, model)


def test_nonlatent_model_contract_records_likelihood_parameter_mappings() -> None:
    poisson = bayes._nonlatent_model_contract("poisson")
    assert poisson["likelihood"] == "Poisson(mu)"
    assert poisson["link"] == "log"
    assert poisson["prediction_functional"] == "count_component_mu"
    assert poisson["negative_binomial_variance"] is None
    assert poisson["zero_inflation_parameter_mapping"] is None

    zinb = bayes._nonlatent_model_contract("zinb")
    assert zinb == {
        "likelihood": "ZeroInflatedNegativeBinomial2(mu, shape, zi)",
        "link": "log",
        "mean_parameter": "mu",
        "prediction_functional": "count_component_mu",
        "negative_binomial_variance": "mu + mu^2 / shape",
        "shape_parameter_mapping": "Bambi/PyMC alpha = brms/rstanarm shape",
        "zero_inflation_reference_parameter": "zi",
        "zero_inflation_runtime_parameter": "psi",
        "zero_inflation_parameter_mapping": "psi = 1 - zi",
        "zero_inflation_reference_semantics": "structural_zero_probability",
        "zero_inflation_runtime_semantics": "count_component_probability",
    }


def test_coefficient_summary_includes_intercept_and_canonical_auxiliaries() -> None:
    posterior = FakePosterior(
        {
            "Intercept": np.asarray([[1.0, 2.0, 3.0, 4.0]]),
            "x": np.asarray([[0.1, 0.2, 0.3, 0.4]]),
            "alpha": np.asarray([[2.0, 3.0, 4.0, 5.0]]),
            "psi": np.asarray([[0.9, 0.8, 0.7, 0.6]]),
            "unrelated_nuisance": np.asarray([[10.0, 20.0, 30.0, 40.0]]),
        }
    )
    model = SimpleNamespace(
        family=SimpleNamespace(
            name="zero_inflated_negativebinomial",
            likelihood=SimpleNamespace(parent="mu"),
        ),
        components={
            "mu": SimpleNamespace(
                intercept_term=SimpleNamespace(name="Intercept"),
                common_terms={"x": SimpleNamespace()},
            )
        },
    )

    summary = bayes._coefficient_summary(
        model,
        SimpleNamespace(posterior=posterior),
    )

    assert summary["term"].tolist() == ["(Intercept)", "x", "shape", "zi"]
    assert summary.loc[summary["term"] == "shape", "mean"].iloc[0] == 3.5
    assert summary.loc[summary["term"] == "zi", "mean"].iloc[0] == pytest.approx(0.25)


def test_diagnostics_fail_closed_without_a_finite_mcse_over_sd_ratio() -> None:
    class NonfiniteArviz:
        @staticmethod
        def summary(_idata, kind, round_to):
            assert round_to == "none"
            if kind == "diagnostics":
                return pd.DataFrame(
                    {"mcse_mean": [0.01, np.inf]},
                    index=["a", "b"],
                )
            return pd.DataFrame(
                {"sd": [0.0, 0.5]},
                index=["a", "b"],
            )

        @staticmethod
        def bfmi(_idata):
            return np.asarray([0.8, 0.8])

    diagnostics = bayes._collect_diagnostics(
        az=NonfiniteArviz,
        inference_data=SimpleNamespace(
            posterior={"beta": np.ones((2, 100))}, sample_stats={}
        ),
        backend="rstanarm",
        chains=2,
        draws=100,
        has_bias_term=True,
    )

    convergence = diagnostics["convergence"]
    assert "mcse_over_sd_max" not in convergence
    assert convergence["ebfmi_min"] == pytest.approx(0.8)


def test_diagnostics_reject_nonfinite_ebfmi_in_any_chain() -> None:
    class PartlyInvalidArviz(FakeArviz):
        @staticmethod
        def bfmi(_idata):
            return np.asarray([0.8, np.nan])

    diagnostics = bayes._collect_diagnostics(
        az=PartlyInvalidArviz,
        inference_data=SimpleNamespace(
            posterior={"beta": np.ones((2, 4))},
            sample_stats={
                "diverging": np.zeros((2, 4), dtype=bool),
                "reached_max_treedepth": np.zeros((2, 4), dtype=bool),
            },
        ),
        backend="rstanarm",
        chains=2,
        draws=4,
        has_bias_term=False,
    )

    assert "ebfmi_min" not in diagnostics["convergence"]


def test_live_fit_handle_is_safe_in_dataframe_attrs() -> None:
    handle = bayes.BayesianFit(
        model=SimpleNamespace(unpickleable=importlib),
        inference_data=object(),
        backend="rstanarm",
    )

    assert deepcopy(handle) is handle


def test_coverage_offset_uses_mobility_formula_only_and_summarizes_draws(
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, pd.DataFrame],
) -> None:
    install_fake_runtime(monkeypatch)
    actual = bayes.adjust_multilevel_bayesian(
        **inputs,
        formula_info={
            "formula": "flow ~ log_distance + bias_e_origin",
            "interface": "split",
            "source": "split_formula",
            "mobility_formula": "~ log_distance",
            "bias_formula": "~ bias_e_origin",
            "mobility_variables": ["log_distance"],
            "bias_variables": ["bias_e_origin"],
        },
        model_family="poisson",
        backend="auto",
        random_intercept="none",
        target_scale="true_flow",
        observation_model="coverage_offset",
        coverage_scale="origin",
        prediction_scope="complete_grid",
        flow_adj_summary="median",
        iter=9,
        chains=2,
        seed=42,
        include_flow_adj_draws=True,
        start_time=0.0,
    )

    model = FakeModel.instances[-1]
    assert model.formula_text == (
        "flow ~ log_distance + offset(log_observation_probability)"
    )
    assert "bias_e_origin" not in model.formula_text
    assert model.family_name == "poisson"
    assert model.model_kwargs["auto_scale"] is True
    assert model.fit_kwargs == {
        "draws": 5,
        "tune": 4,
        "chains": 2,
        "cores": 1,
        "random_seed": 42,
        "target_accept": 0.95,
        "nuts": {"max_treedepth": 15},
        "progressbar": False,
    }
    assert len(model.prediction_calls) == 2
    assert all(
        call[1]["include_group_specific"] is False for call in model.prediction_calls
    )
    assert np.allclose(model.prediction_calls[1][0]["log_observation_probability"], 0.0)

    assert np.allclose(actual["flow_adj"], 11.0)
    assert np.allclose(actual["flow_true_pred"], actual["flow_adj"])
    expected_probability = np.asarray([0.2, 0.2, 0.1, 0.1])
    assert np.allclose(actual["observation_probability"], expected_probability)
    assert np.allclose(actual["flow_mpd_pred"], 11.0 * expected_probability)
    assert np.allclose(actual["flow_adj_mean"], 11.0)
    assert np.all(actual["flow_adj_q2.5"] < actual["flow_adj_mean"])
    assert np.all(actual["flow_adj_q97.5"] > actual["flow_adj_mean"])
    assert actual.attrs["flow_adj_draws"].shape == (4, 4)
    assert actual.attrs["backend"] == "rstanarm"
    assert actual.attrs["runtime_backend"] == "bambi_pymc"
    assert actual.attrs["result_metadata"]["sampling"] == {
        "seed": 42,
        "chains": 2,
        "iter": 9,
        "warmup_per_chain": 4,
        "draws_per_chain": 5,
        "cores": 1,
        "target_accept": 0.95,
        "max_treedepth": 15,
    }
    convergence = actual.attrs["diagnostics"]["convergence"]
    assert convergence["rhat_max"] == 1.004345678
    assert convergence["ess_bulk_min"] == 450.2345678
    assert convergence["mcse_mean_max"] == 0.0234567891
    assert convergence["mcse_over_sd_max"] == pytest.approx(0.0234567891 / 0.4)
    expected_model_contract = bayes._nonlatent_model_contract("poisson")
    assert actual.attrs["model_contract"] == expected_model_contract
    assert actual.attrs["result_metadata"]["model_contract"] == expected_model_contract
    coefficients = actual.attrs["coefficients"]
    assert coefficients["term"].tolist() == [
        "(Intercept)",
        "log_distance",
        "bias_e_origin",
    ]
    assert {
        "flow_adj_mean",
        "flow_adj_median",
        "flow_adj_q2.5",
        "flow_adj_q97.5",
        "flow_mpd_pred_mean",
        "flow_mpd_pred_median",
        "flow_mpd_pred_q2.5",
        "flow_mpd_pred_q97.5",
    }.issubset(actual.columns)


def test_reduced_form_zeroes_only_declared_bias_terms(
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, pd.DataFrame],
) -> None:
    install_fake_runtime(monkeypatch)
    actual = adjust_multilevel_bayes(
        **inputs,
        mobility_formula="~ log_distance",
        bias_formula="~ bias_e_origin",
        model_family="poisson",
        random_intercept="none",
        target_scale="mpd_counterfactual",
        observation_model="reduced_form",
        iter=8,
        chains=1,
        seed=7,
    )

    model = FakeModel.instances[-1]
    assert model.formula_text == "flow ~ log_distance + bias_e_origin"
    prediction_data = model.prediction_calls[0][0]
    assert np.allclose(prediction_data["bias_e_origin"], 0.0)
    assert np.allclose(actual["flow_adj"], 11.0)
    assert actual["flow_mpd_pred"].isna().all()
    assert actual["flow_true_pred"].isna().all()
    assert actual.attrs["bias_terms"] == ["bias_e_origin"]


def test_public_dispatch_forwards_the_complete_nonlatent_contract(
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, pd.DataFrame],
) -> None:
    from debiaspy import adjust_multilevel_bayes

    install_fake_runtime(monkeypatch)
    actual = adjust_multilevel_bayes(
        **inputs,
        mobility_formula="~ log_distance",
        bias_formula="~ bias_e_origin",
        model_engine="bayesian",
        model_family="poisson",
        random_intercept="none",
        target_scale="true_flow",
        observation_model="coverage_offset",
        coverage_scale="origin",
        prediction_scope="complete_grid",
        iter=8,
        chains=1,
        seed=19,
    )

    assert np.allclose(actual["flow_adj"], 11.0)
    assert actual.attrs["backend"] == "rstanarm"
    assert actual.attrs["runtime_backend"] == "bambi_pymc"
    assert FakeModel.instances[-1].fit_kwargs["draws"] == 4
    assert FakeModel.instances[-1].fit_kwargs["tune"] == 4


@pytest.mark.parametrize(
    ("model_family", "backend", "expected_target_accept"),
    [("poisson", "auto", 0.99), ("zip", "brms", 0.999)],
)
def test_random_intercept_uses_stricter_sampler_adaptation(
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, pd.DataFrame],
    model_family: str,
    backend: str,
    expected_target_accept: float,
) -> None:
    install_fake_runtime(monkeypatch)
    actual = bayes.adjust_multilevel_bayesian(
        **inputs,
        model_family=model_family,
        backend=backend,
        random_intercept="origin",
        target_scale="mpd_counterfactual",
        observation_model="reduced_form",
        iter=8,
        chains=1,
        seed=23,
    )

    model = FakeModel.instances[-1]
    assert "(1 | origin)" in model.formula_text
    assert model.fit_kwargs["target_accept"] == expected_target_accept
    assert (
        actual.attrs["result_metadata"]["sampling"]["target_accept"]
        == expected_target_accept
    )


def test_zip_prediction_extracts_count_component_mu() -> None:
    data = pd.DataFrame(
        {
            "flow": [1.0, 2.0],
            "bias_e_origin": [0.0, 0.0],
            "log_observation_probability": [np.nan, np.nan],
        }
    )
    model = FakeModel("flow ~ bias_e_origin", data, "zero_inflated_poisson")
    draws = bayes._predict_parent_draws(model, FakeInferenceData(), data)

    assert np.allclose(draws[:, 0], [8.0, 10.0, 12.0, 14.0])
    assert not np.allclose(draws[:, 0], [0.8, 1.0, 1.2, 1.4])


def test_default_formula_adds_requested_random_intercept() -> None:
    formula, terms, bias_terms = bayes._build_formula_contract(
        formula_info={"interface": "default", "formula": None},
        default_covariate_col="rural_pct",
        include_pop_terms=True,
        scenario_terms=["mpd_time"],
        random_intercept="origin",
        target_scale="mpd_counterfactual",
        observation_model="reduced_form",
    )

    assert formula == (
        "flow ~ income_o + income_d + log_distance + bias_e_origin + "
        "log_pop_o + log_pop_d + mpd_time + (1 | origin)"
    )
    assert terms["formula_random_effects"] == ["(1 | origin)"]
    assert bias_terms == ["bias_e_origin"]


def test_bayesian_conflicting_covariates_fail_before_sampling(inputs, monkeypatch):
    def unexpected_runtime():
        raise AssertionError("Conflicting input joins must fail before sampling")

    monkeypatch.setattr(bayes, "_load_bayesian_dependencies", unexpected_runtime)
    conflicting = inputs["covariates_df"].iloc[[0]].copy()
    conflicting["population"] += 1
    args = dict(inputs)
    args["covariates_df"] = pd.concat(
        [inputs["covariates_df"], conflicting], ignore_index=True
    )
    with pytest.raises(DebiasSchemaError, match="conflicting"):
        adjust_multilevel_bayes(**args, random_intercept="none")


def test_missing_runtime_has_bayesian_install_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def missing(name: str, package: str | None = None):
        if name == "bambi":
            raise ImportError("blocked for contract test")
        return real_import(name, package)

    monkeypatch.setattr(bayes.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"debiaspy\[bayesian\]"):
        bayes._load_bayesian_dependencies()


def test_incompatible_runtime_has_bayesian_install_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(_name: str, _package: str | None = None):
        raise AttributeError("incompatible transitive runtime")

    monkeypatch.setattr(bayes.importlib, "import_module", broken)
    with pytest.raises(ImportError, match=r"debiaspy\[bayesian\]"):
        bayes._load_bayesian_dependencies()


@pytest.mark.skipif(
    os.environ.get("DEBIASPY_RUN_BAYES_SMOKE") != "1"
    or find_spec("bambi") is None
    or find_spec("arviz") is None,
    reason=("set DEBIASPY_RUN_BAYES_SMOKE=1 with the Bambi/PyMC runtime installed"),
)
def test_real_bambi_poisson_smoke(
    inputs: dict[str, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYTENSOR_FLAGS", f"base_compiledir={tmp_path / 'pytensor'}")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    actual = adjust_multilevel_bayes(
        **inputs,
        mobility_formula="~ log_distance",
        bias_formula="~ bias_e_origin",
        model_family="poisson",
        backend="auto",
        random_intercept="none",
        target_scale="mpd_counterfactual",
        observation_model="reduced_form",
        iter=10,
        chains=1,
        seed=101,
        refresh=0,
    )

    assert len(actual) == len(inputs["mpd_od_df"])
    assert np.isfinite(actual["flow_adj"]).all()
    assert actual.attrs["backend"] == "rstanarm"
    assert actual.attrs["runtime_backend"] == "bambi_pymc"
    assert actual.attrs["result_metadata"]["sampling"]["draws_per_chain"] == 5


def test_nonlatent_backend_rejects_latent_route_before_dependency_loading(
    inputs: dict[str, pd.DataFrame],
) -> None:
    with pytest.raises(NotImplementedError, match="CmdStanPy latent backend"):
        bayes.adjust_multilevel_bayesian(
            **inputs,
            observation_model="latent_two_level",
            target_scale="true_flow",
        )


def test_prediction_totals_are_preserved_across_chunks_without_storing_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def predict(model, inference_data, data):
        return np.arange(1, 5)[:, None] * data["value"].to_numpy()[None, :]

    monkeypatch.setattr(bayes, "_predict_parent_draws", predict)
    data = pd.DataFrame({"value": [1, 2, 3]})
    result = bayes._predict_summary(None, None, data, False, chunk_size=2)
    np.testing.assert_allclose(result.total_draws, [6, 12, 18, 24])
    assert result.draws is None


def test_adjusted_flow_diagnostics_report_functional_uncertainty() -> None:
    az = pytest.importorskip("arviz")
    import xarray as xr

    totals = np.random.default_rng(17).lognormal(size=4000)
    posterior = xr.Dataset({"coefficient": (("chain", "draw"), np.ones((4, 1000)))})
    result = bayes._adjusted_flow_diagnostics(
        az, SimpleNamespace(posterior=posterior), totals
    )
    assert result["status"] == "available"
    assert result["mean"] == pytest.approx(totals.mean())
    assert result["mcse_mean"] > 0
    assert result["ess_tail"] > 0
    assert result["mean_precision"] == "within_mcse_threshold"


def test_extreme_flow_draw_warns_even_with_good_rank_diagnostics() -> None:
    az = pytest.importorskip("arviz")
    import xarray as xr

    totals = np.random.default_rng(42).lognormal(size=4000)
    totals[0] = 1e12
    posterior = xr.Dataset({"coefficient": (("chain", "draw"), np.ones((4, 1000)))})
    diagnostics = bayes._adjusted_flow_diagnostics(
        az, SimpleNamespace(posterior=posterior), totals
    )
    assert diagnostics["rhat"] < 1.01
    assert diagnostics["mean_precision"] == "unstable"
    assert diagnostics["largest_draw_share_of_sum"] > 0.99
    with pytest.warns(RuntimeWarning, match="poorly determined"):
        bayes._warn_unstable_adjusted_mean(diagnostics, "mean")
    assert diagnostics["mean"] == pytest.approx(totals.mean())
    # A precision assessment of the mean must not certify or reject the median.
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        bayes._warn_unstable_adjusted_mean(diagnostics, "median")
    assert not caught


def test_adjusted_flow_diagnostics_require_multiple_chains() -> None:
    assert (
        bayes._adjusted_flow_diagnostics(None, None, np.ones(20))["status"]
        == "unavailable"
    )
