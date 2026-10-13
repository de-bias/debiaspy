from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from debiaspy import _multilevel_inputs as core
from debiaspy._adjustment_common import existing_columns, model_result_attributes
from debiaspy._exceptions import DebiasSchemaError
from debiaspy._latent import fit_latent_cmdstan, prepare_latent_state


def adjust_multilevel_latent(**kwargs: Any) -> pd.DataFrame:
    backend = str(kwargs["backend"])
    if backend not in {"auto", "stan_latent"}:
        raise DebiasSchemaError(
            "`observation_model = 'latent_two_level'` requires "
            "`backend = 'stan_latent'` or `backend = 'auto'`."
        )
    model_family = str(kwargs["model_family"])
    if model_family not in {"poisson", "negbin"}:
        raise DebiasSchemaError(
            "`observation_model = 'latent_two_level'` supports only "
            "`model_family = 'poisson'` or `model_family = 'negbin'`."
        )

    scenario_info = core._resolve_scenario(
        kwargs["mpd_od_df"],
        source_col=kwargs["source_col"],
        time_col=kwargs["time_col"],
        scenario=kwargs["scenario"],
        repeated_observation=kwargs["repeated_observation"],
    )
    od_audit = None
    if kwargs["prediction_scope"] == "complete_grid":
        od_audit = core._audit_complete_grid(
            kwargs["mpd_od_df"],
            flow_col=kwargs["flow_col"],
            scenario_cols=scenario_info["audit_cols"],
        )
        if not bool(od_audit["strict_square_support"].iloc[0]):
            raise DebiasSchemaError(
                "`prediction_scope = 'complete_grid'` requires a strict square OD grid."
            )

    prep = core._prepare_multilevel_data(
        mpd_od_df=kwargs["mpd_od_df"],
        coverage_df=kwargs["coverage_df"],
        covariates_df=kwargs["covariates_df"],
        distance_df=kwargs["distance_df"],
        flow_col=kwargs["flow_col"],
        income_col=kwargs["income_col"],
        pop_col=kwargs["pop_col"],
        distance_col=kwargs["distance_col"],
        scenario_info=scenario_info,
    )
    prediction_df = prep["model_df"].copy()
    latent_info = prepare_latent_state(
        prediction_df, scenario_info, kwargs["latent_flow_unit"]
    )
    prediction_df = latent_info["data"]
    core._validate_coverage_offset_data(prediction_df, kwargs["coverage_scale"])
    prediction_df = core._set_observation_probability(
        prediction_df, kwargs["coverage_scale"]
    )
    fit_df = prediction_df
    if (
        kwargs["prediction_scope"] == "complete_grid"
        and "mpd_observed" in fit_df.columns
    ):
        fit_df = fit_df.loc[fit_df["mpd_observed"].isin([True])].copy()
    if len(fit_df) < 2:
        raise DebiasSchemaError(
            "Insufficient rows for fitting after preprocessing. Need at least "
            "2 complete rows."
        )

    controls = {
        name: kwargs[name]
        for name in (
            "latent_coef_prior_scale",
            "latent_bias_prior_scale",
            "latent_intercept_prior_scale",
            "latent_state_prior_scale",
            "latent_source_prior_scale",
            "latent_time_prior_scale",
            "latent_phi_prior_rate",
            "latent_adapt_delta",
            "latent_max_treedepth",
            "latent_rng_eta_max",
        )
    }
    fit = fit_latent_cmdstan(
        fit_df=fit_df,
        prediction_df=prediction_df,
        formula_info=kwargs["formula_info"],
        default_covariate_col=prep["default_covariate_col"],
        include_pop_terms=prep["has_pop_terms"],
        scenario_info=scenario_info,
        model_family=model_family,
        controls=controls,
        iter=kwargs["iter"],
        chains=kwargs["chains"],
        seed=kwargs["seed"],
        refresh=kwargs["refresh"],
    )
    flow_adj_draws = np.asarray(fit["flow_true_pred_draws"], dtype=float)
    flow_mpd_draws = np.asarray(fit["flow_mpd_pred_draws"], dtype=float)
    summaries = _posterior_summaries(flow_adj_draws, flow_mpd_draws)
    flow_adj = summaries[f"flow_adj_{kwargs['flow_adj_summary']}"]
    flow_mpd = summaries[f"flow_mpd_pred_{kwargs['flow_adj_summary']}"]

    modeled = prediction_df.loc[:, ["row_id"]].copy()
    modeled["latent_flow_id"] = prediction_df["latent_flow_id"].astype(str)
    modeled["latent_flow_unit"] = prediction_df["latent_flow_unit"].astype(str)
    modeled["flow_adj"] = flow_adj
    modeled["flow_true_pred"] = flow_adj
    modeled["flow_mpd_pred"] = flow_mpd
    modeled["observation_probability"] = prediction_df[
        "observation_probability"
    ].to_numpy(dtype=float)
    modeled["log_observation_probability"] = prediction_df[
        "log_observation_probability"
    ].to_numpy(dtype=float)
    for name, values in summaries.items():
        modeled[name] = values

    base = prep["base_df"].copy()
    base["prediction_scope"] = kwargs["prediction_scope"]
    base["model_fit_status"] = np.where(
        base["row_id"].isin(fit_df["row_id"]),
        "fit",
        np.where(base["row_id"].isin(prediction_df["row_id"]), "predicted", "excluded"),
    )
    base = base.merge(modeled, how="left", on="row_id", sort=False)
    select = ["origin", "destination"]
    if "mpd_source" in base:
        select.append("mpd_source")
    if "mpd_time" in base:
        select.append("mpd_time")
    select.extend(existing_columns(base, kwargs["keep_cols"]))
    select.extend(
        [
            "mpd_observed",
            "mpd_zero_filled",
            "mpd_row_status",
            "prediction_scope",
            "model_fit_status",
            "flow",
            "flow_adj",
            "flow_mpd_pred",
            "flow_true_pred",
            "flow_adj_mean",
            "flow_adj_median",
            "flow_adj_q2.5",
            "flow_adj_q97.5",
            "flow_mpd_pred_mean",
            "flow_mpd_pred_median",
            "flow_mpd_pred_q2.5",
            "flow_mpd_pred_q97.5",
            "observation_probability",
            "latent_flow_id",
            "latent_flow_unit",
            "coverage_rate_o",
            "coverage_rate_d",
            "log_observation_probability",
            "distance_km",
            "log_distance",
            "bias_e_origin",
            "log_dist_synth",
        ]
    )
    out = base.loc[:, existing_columns(base, select)].reset_index(drop=True)

    stan_summary = fit["stan_data_summary"]
    identifiability = dict(latent_info["identifiability"])
    identifiability.update(
        {
            key: stan_summary[key]
            for key in (
                "n_unobserved_latent_flows",
                "min_observations_per_latent_flow",
                "max_observations_per_latent_flow",
            )
        }
    )
    formula = (
        f"{kwargs['formula_info'].get('formula') or stan_summary['true_formula']} "
        "+ offset(log_observation_probability)"
    )
    model_terms = {
        "formula": formula,
        "formula_source": kwargs["formula_info"].get("source", "default"),
        "formula_interface": kwargs["formula_info"].get("interface", "default"),
        "mobility_formula": kwargs["formula_info"].get("mobility_formula"),
        "bias_formula": kwargs["formula_info"].get("bias_formula"),
        "mobility_variables": kwargs["formula_info"].get("mobility_variables", []),
        "bias_variables": kwargs["formula_info"].get("bias_variables", []),
        "latent_true_terms": stan_summary["true_terms"],
        "latent_observation_terms": stan_summary["observation_terms"],
    }
    runtime = time.time() - float(kwargs["start_time"])
    metadata = {
        "backend": "stan_latent",
        "runtime_backend": "cmdstanpy",
        "cmdstan_version": fit["cmdstan_version"],
        "model_engine": "bayesian",
        "model_family": model_family,
        "stage_scope": (
            "complete_grid_prediction"
            if kwargs["prediction_scope"] == "complete_grid"
            else "observed_od_only"
        ),
        "prediction_scope": kwargs["prediction_scope"],
        "scenario": scenario_info["scenario"],
        "source_col": scenario_info["source_col"],
        "time_col": scenario_info["time_col"],
        "repeated_observation": scenario_info["repeated_observation"],
        "scenario_cols": scenario_info["scenario_cols"],
        "n_sources": scenario_info["n_sources"],
        "n_time_periods": scenario_info["n_time_periods"],
        "target_scale": "true_flow",
        "observation_model": "latent_two_level",
        "coverage_scale": kwargs["coverage_scale"],
        "offset_column": "log_observation_probability",
        "model_terms": model_terms,
        "bias_terms": kwargs["formula_info"].get("bias_variables", []),
        "random_intercept": kwargs["random_intercept"],
        "flow_adj_summary": kwargs["flow_adj_summary"],
        "distance_source": prep["distance_source"],
        "runtime_seconds": runtime,
        "n_input_rows": len(prep["base_df"]),
        "n_fit_rows": len(fit_df),
        "n_prediction_rows": len(prediction_df),
        "n_zero_filled_prediction_rows": int(
            np.asarray(prediction_df["mpd_zero_filled"].isin([True]), dtype=bool).sum()
        ),
        "latent_flow_unit": latent_info["latent_flow_unit"],
        "n_latent_flows": latent_info["n_latent_flows"],
        "latent_identifiability": identifiability,
        "latent_true_formula": stan_summary["true_formula"],
        "latent_observation_formula": stan_summary["observation_formula"],
        "latent_true_terms": stan_summary["true_terms"],
        "latent_observation_terms": stan_summary["observation_terms"],
        "latent_source_effect_layer": stan_summary["source_effect_layer"],
        "latent_time_effect_layer": stan_summary["time_effect_layer"],
        "latent_ignored_random_effect_terms": fit["latent_formula_info"][
            "ignored_random_effect_terms"
        ],
        "latent_controls": controls,
        "model_contract": {
            "likelihood": (
                "Poisson(mu)"
                if model_family == "poisson"
                else "NegativeBinomial2(mu, shape)"
            ),
            "link": "log",
            "mean_parameter": "mu",
            "prediction_functional": "latent_true_flow_mu",
            "negative_binomial_variance": (
                "mu + mu^2 / shape" if model_family == "negbin" else None
            ),
            "shape_parameter_mapping": (
                "Stan phi = brms/rstanarm shape" if model_family == "negbin" else None
            ),
            "zero_inflation_reference_parameter": None,
            "zero_inflation_runtime_parameter": None,
            "zero_inflation_parameter_mapping": None,
            "zero_inflation_reference_semantics": None,
            "zero_inflation_runtime_semantics": None,
        },
        "resolved_priors": {
            "reference_backend": "stan_latent",
            "identity_claim": True,
            "latent_coef_prior_scale": controls["latent_coef_prior_scale"],
            "latent_bias_prior_scale": controls["latent_bias_prior_scale"],
            "latent_intercept_prior_scale": controls["latent_intercept_prior_scale"],
            "latent_state_prior_scale": controls["latent_state_prior_scale"],
            "latent_source_prior_scale": controls["latent_source_prior_scale"],
            "latent_time_prior_scale": controls["latent_time_prior_scale"],
            "latent_phi_prior_rate": controls["latent_phi_prior_rate"],
        },
        "od_audit": od_audit,
    }
    notes = (
        "observation_model=latent_two_level; runtime=cmdstanpy; "
        "stan_source=latent_two_level.stan."
    )
    out.attrs.update(
        model_result_attributes(
            metadata,
            model=fit["stanfit"],
            formula=formula,
            coefficients=fit["coefficients"],
            runtime_backend="cmdstanpy",
            scenario_info=scenario_info,
            diagnostics=fit["diagnostics"],
            model_notes=notes,
            latent_flow_unit=latent_info["latent_flow_unit"],
            latent_identifiability=identifiability,
            latent_controls=controls,
            resolved_priors=metadata["resolved_priors"],
            cmdstan_version=fit["cmdstan_version"],
        )
    )
    if kwargs["include_flow_adj_draws"]:
        out.attrs["flow_adj_draws"] = flow_adj_draws
    return out


def _posterior_summaries(
    flow_adj_draws: np.ndarray, flow_mpd_draws: np.ndarray
) -> dict[str, np.ndarray]:
    if flow_adj_draws.ndim != 2 or flow_mpd_draws.shape != flow_adj_draws.shape:
        raise RuntimeError("Latent posterior prediction matrices have invalid shapes.")
    return {
        "flow_adj_mean": np.mean(flow_adj_draws, axis=0),
        "flow_adj_median": np.median(flow_adj_draws, axis=0),
        "flow_adj_q2.5": np.quantile(flow_adj_draws, 0.025, axis=0),
        "flow_adj_q97.5": np.quantile(flow_adj_draws, 0.975, axis=0),
        "flow_mpd_pred_mean": np.mean(flow_mpd_draws, axis=0),
        "flow_mpd_pred_median": np.median(flow_mpd_draws, axis=0),
        "flow_mpd_pred_q2.5": np.quantile(flow_mpd_draws, 0.025, axis=0),
        "flow_mpd_pred_q97.5": np.quantile(flow_mpd_draws, 0.975, axis=0),
    }


__all__ = ["adjust_multilevel_latent"]
