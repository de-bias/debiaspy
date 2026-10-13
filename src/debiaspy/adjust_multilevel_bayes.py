from __future__ import annotations

import time

import numpy as np
import pandas as pd

from debiaspy._adjustment_common import (
    existing_columns,
    match_choice,
    model_result_attributes,
)
from debiaspy._exceptions import DebiasSchemaError
from debiaspy._frequentist import (
    _filter_formula_model_data,
    _fit_count_model,
    _predict_linear,
    _stabilize_formula_factor_levels,
    _validate_frequentist_random_group,
)
from debiaspy._multilevel_inputs import (
    _add_random_intercept_to_formula,
    _audit_complete_grid,
    _bias_terms,
    _counterfactual_bias_data,
    _formula_terms,
    _frequentist_formula,
    _frequentist_random_effect_spec,
    _prepare_multilevel_data,
    _random_intercept_group_col,
    _resolve_formula_info,
    _resolve_scenario,
    _resolve_target_formula_info,
    _set_observation_probability,
    _validate_coverage_offset_data,
    _validate_observation_contract,
)


class _DefaultRandomIntercept(str):
    """Marker preserving R's distinction between missing and supplied defaults."""


_DEFAULT_RANDOM_INTERCEPT = _DefaultRandomIntercept("origin")


def adjust_multilevel_bayes(
    mpd_od_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    covariates_df: pd.DataFrame | None = None,
    distance_df: pd.DataFrame | None = None,
    flow_col: str = "flow",
    income_col: str | None = None,
    pop_col: str = "population",
    distance_col: str = "distance_km",
    source_col: str | None = None,
    time_col: str | None = None,
    scenario: str = "auto",
    repeated_observation: str = "auto",
    random_intercept: str = _DEFAULT_RANDOM_INTERCEPT,
    formula: str | None = None,
    mobility_formula: str | None = None,
    bias_formula: str | None = None,
    model_family: str = "poisson",
    model_engine: str = "bayesian",
    backend: str = "auto",
    flow_adj_summary: str = "mean",
    target_scale: str = "mpd_counterfactual",
    observation_model: str = "reduced_form",
    coverage_scale: str = "origin",
    latent_flow_unit: str = "auto",
    latent_coef_prior_scale: float = 1,
    latent_bias_prior_scale: float = 0.5,
    latent_intercept_prior_scale: float = 2.5,
    latent_state_prior_scale: float = 0.75,
    latent_source_prior_scale: float = 0.5,
    latent_time_prior_scale: float = 0.5,
    latent_phi_prior_rate: float = 1,
    latent_adapt_delta: float = 0.95,
    latent_max_treedepth: int = 12,
    latent_rng_eta_max: float = 20,
    prediction_scope: str = "observed",
    iter: int = 1000,
    chains: int = 2,
    seed: int = 123,
    refresh: int = 0,
    include_flow_adj_draws: bool = False,
    keep_cols: str | list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """Fit a frequentist or Bayesian multilevel flow-adjustment model.

    ``mpd_od_df`` requires OD keys and ``flow_col``; ``coverage_df`` supplies
    area population/user counts. Formula terms can additionally require
    ``area``-keyed covariates, OD distances, and source/time columns. Scenario,
    observation-model, family, engine, backend, random-effect, prediction, and
    posterior-summary options jointly define the supported model contract.

    Returns an adjusted-flow data frame with engine-specific model,
    coefficient, diagnostics, timing, and contract metadata in ``attrs``.
    Invalid combinations, unsupported formulas or random effects, unavailable
    runtimes, and fitting errors raise rather than falling back
    to a different model. Parity is limited to the frozen frequentist and
    Bayesian audit matrices, their priors, and declared numeric/statistical
    gates; posterior draws and fitted Python/R objects are not identical.

    Bayesian engines are experimental on the integration branch; backend
    strategy and the full sampling matrix require maintainer review.
    Bayesian mean summaries issue a ``RuntimeWarning`` when the total
    adjusted-flow mean's Monte Carlo standard error exceeds 5% of its mean.
    This precision warning does not change the returned estimator or establish
    full convergence when absent. Inspect ``adjusted_flow_total`` and
    ``counterfactual_support`` in the diagnostics before interpreting flows.
    """
    start_time = time.time()
    random_intercept_supplied = random_intercept is not _DEFAULT_RANDOM_INTERCEPT
    scenario = match_choice(scenario, ("auto", "s1", "s2", "s3", "s4"), "scenario")
    repeated_observation = match_choice(
        repeated_observation,
        ("auto", "none", "time", "source", "source_time"),
        "repeated_observation",
    )
    random_intercept = match_choice(
        random_intercept,
        ("origin", "destination", "od", "source", "time", "source_time", "none"),
        "random_intercept",
    )
    model_family = match_choice(
        model_family, ("poisson", "negbin", "zip", "zinb"), "model_family"
    )
    model_engine = match_choice(
        model_engine, ("bayesian", "frequentist"), "model_engine"
    )
    backend = match_choice(
        backend,
        ("auto", "rstanarm", "brms", "stan_latent", "gpboost"),
        "backend",
    )
    flow_adj_summary = match_choice(
        flow_adj_summary, ("mean", "median"), "flow_adj_summary"
    )
    target_scale = match_choice(
        target_scale, ("mpd_counterfactual", "true_flow"), "target_scale"
    )
    observation_model = match_choice(
        observation_model,
        ("reduced_form", "coverage_offset", "latent_two_level"),
        "observation_model",
    )
    coverage_scale = match_choice(
        coverage_scale, ("origin", "destination", "both"), "coverage_scale"
    )
    latent_flow_unit = match_choice(
        latent_flow_unit, ("auto", "od", "od_time"), "latent_flow_unit"
    )
    prediction_scope = match_choice(
        prediction_scope, ("observed", "complete_grid"), "prediction_scope"
    )
    _validate_observation_contract(target_scale, observation_model, model_engine)
    formula_info = _resolve_formula_info(
        formula,
        mobility_formula,
        bias_formula,
        allow_extended=model_engine == "bayesian",
    )
    fit_formula_info = (
        _resolve_target_formula_info(
            formula_info,
            target_scale=target_scale,
            observation_model=observation_model,
        )
        if model_engine == "frequentist"
        else formula_info
    )
    formula_random_effect = (
        _frequentist_random_effect_spec(str(fit_formula_info["formula"]))
        if model_engine == "frequentist" and fit_formula_info["formula"] is not None
        else None
    )
    reported_random_intercept = random_intercept
    if formula_random_effect is not None:
        random_intercept = formula_random_effect["selector"]
        reported_random_intercept = (
            reported_random_intercept if random_intercept_supplied else "none"
        )
    elif formula_info["formula"] is not None:
        # In debiasR a supplied formula is the complete model specification;
        # the selector does not append a random term that is absent from it.
        random_intercept = "none"
        reported_random_intercept = (
            reported_random_intercept if random_intercept_supplied else "none"
        )
    if model_engine == "bayesian":
        if backend == "gpboost":
            raise DebiasSchemaError(
                "`backend = 'gpboost'` is available only with "
                "`model_engine = 'frequentist'`."
            )
        if observation_model == "latent_two_level":
            from debiaspy._bayesian_latent import adjust_multilevel_latent

            bayesian_impl = adjust_multilevel_latent
        else:
            from debiaspy._bayesian import adjust_multilevel_bayesian

            bayesian_impl = adjust_multilevel_bayesian
        result = bayesian_impl(
            mpd_od_df=mpd_od_df,
            coverage_df=coverage_df,
            covariates_df=covariates_df,
            distance_df=distance_df,
            flow_col=flow_col,
            income_col=income_col,
            pop_col=pop_col,
            distance_col=distance_col,
            source_col=source_col,
            time_col=time_col,
            scenario=scenario,
            repeated_observation=repeated_observation,
            random_intercept=random_intercept,
            formula_info=formula_info,
            model_family=model_family,
            backend=backend,
            flow_adj_summary=flow_adj_summary,
            target_scale=target_scale,
            observation_model=observation_model,
            coverage_scale=coverage_scale,
            prediction_scope=prediction_scope,
            iter=iter,
            chains=chains,
            seed=seed,
            refresh=refresh,
            include_flow_adj_draws=include_flow_adj_draws,
            keep_cols=keep_cols,
            **(
                {
                    "latent_flow_unit": latent_flow_unit,
                    "latent_coef_prior_scale": latent_coef_prior_scale,
                    "latent_bias_prior_scale": latent_bias_prior_scale,
                    "latent_intercept_prior_scale": latent_intercept_prior_scale,
                    "latent_state_prior_scale": latent_state_prior_scale,
                    "latent_source_prior_scale": latent_source_prior_scale,
                    "latent_time_prior_scale": latent_time_prior_scale,
                    "latent_phi_prior_rate": latent_phi_prior_rate,
                    "latent_adapt_delta": latent_adapt_delta,
                    "latent_max_treedepth": latent_max_treedepth,
                    "latent_rng_eta_max": latent_rng_eta_max,
                }
                if observation_model == "latent_two_level"
                else {}
            ),
            start_time=start_time,
        )
        if reported_random_intercept != random_intercept:
            result.attrs["random_intercept"] = reported_random_intercept
            metadata = result.attrs.get("result_metadata")
            if isinstance(metadata, dict):
                metadata["random_intercept"] = reported_random_intercept
                metadata["effective_random_intercept"] = random_intercept
        return result
    formula_info = fit_formula_info
    if observation_model == "latent_two_level":
        raise DebiasSchemaError(
            "`observation_model = 'latent_two_level'` requires "
            "`model_engine = 'bayesian'`."
        )
    if random_intercept != "none" and backend not in {"auto", "gpboost"}:
        raise DebiasSchemaError(
            "Frequentist random-intercept fits require `backend='auto'` or "
            "`backend='gpboost'`."
        )
    if random_intercept == "none" and backend == "gpboost":
        raise DebiasSchemaError(
            "`backend='gpboost'` requires one supported random-intercept term. "
            "Use `backend='auto'` for fixed-effect frequentist models."
        )
    if model_family not in {"poisson", "negbin"}:
        raise NotImplementedError(
            "The debiaspy frequentist multilevel slice supports only Poisson "
            "and negative-binomial NB2 models; "
            f"`model_family='{model_family}'` is not translated."
        )
    scenario_info = _resolve_scenario(
        mpd_od_df,
        source_col=source_col,
        time_col=time_col,
        scenario=scenario,
        repeated_observation=repeated_observation,
    )
    od_audit = None
    if prediction_scope == "complete_grid":
        od_audit = _audit_complete_grid(
            mpd_od_df, flow_col=flow_col, scenario_cols=scenario_info["audit_cols"]
        )
        if not bool(od_audit["strict_square_support"].iloc[0]):
            raise DebiasSchemaError(
                "`prediction_scope = 'complete_grid'` requires a strict square OD grid."
            )

    prep = _prepare_multilevel_data(
        mpd_od_df=mpd_od_df,
        coverage_df=coverage_df,
        covariates_df=covariates_df,
        distance_df=distance_df,
        flow_col=flow_col,
        income_col=income_col,
        pop_col=pop_col,
        distance_col=distance_col,
        scenario_info=scenario_info,
    )
    prediction_df = prep["model_df"].copy()

    formula_terms = _formula_terms(
        prediction_df=prediction_df,
        formula_info=formula_info,
        default_covariate_col=prep["default_covariate_col"],
        include_pop_terms=prep["has_pop_terms"],
        scenario_info=scenario_info,
        random_intercept=random_intercept,
        target_scale=target_scale,
    )
    bias_terms = _bias_terms(formula_info, formula_terms)
    if target_scale == "true_flow":
        _validate_coverage_offset_data(prediction_df, coverage_scale)
        prediction_df = _set_observation_probability(prediction_df, coverage_scale)
    else:
        prediction_df["observation_probability"] = np.nan
        prediction_df["log_observation_probability"] = np.nan

    fit_formula = (
        formula_random_effect["fixed_formula"]
        if formula_random_effect is not None
        else _frequentist_formula(formula_info, formula_terms)
    )
    random_group_col = (
        formula_random_effect["group_col"]
        if formula_random_effect is not None
        else _random_intercept_group_col(random_intercept)
    )
    display_formula = (
        str(formula_info["formula"])
        if formula_info["formula"] is not None
        else _add_random_intercept_to_formula(fit_formula, random_group_col)
    )
    filter_terms = list(formula_terms)
    if random_group_col is not None:
        filter_terms.append(random_group_col)
    if target_scale == "true_flow":
        filter_terms.append("log_observation_probability")
    prediction_df = _filter_formula_model_data(
        prediction_df,
        filter_terms,
        context="prediction",
    )
    prediction_df = _stabilize_formula_factor_levels(prediction_df, formula_terms)
    fit_df = prediction_df
    if prediction_scope == "complete_grid" and "mpd_observed" in fit_df.columns:
        fit_df = fit_df.loc[fit_df["mpd_observed"]].copy()
    if len(fit_df) < 2:
        raise DebiasSchemaError(
            "Insufficient rows for fitting after preprocessing. Need at least "
            "2 complete rows."
        )

    if random_group_col is not None:
        _validate_frequentist_random_group(fit_df, random_group_col)

    fit = _fit_count_model(
        fit_df,
        fit_formula,
        model_family,
        target_scale,
        random_group_col=random_group_col,
        display_formula=display_formula,
    )
    if target_scale == "true_flow":
        mpd_lin = _predict_linear(fit, prediction_df, include_offset=True)
        true_df = prediction_df.copy()
        true_df["log_observation_probability"] = 0.0
        true_lin = _predict_linear(fit, true_df, include_offset=True)
        flow_mpd_pred = np.exp(mpd_lin)
        flow_adj = np.exp(true_lin)
        flow_true_pred = flow_adj.copy()
    else:
        cf_df = _counterfactual_bias_data(prediction_df, bias_terms)
        true_lin = _predict_linear(fit, cf_df, include_offset=False)
        flow_adj = np.exp(true_lin)
        flow_mpd_pred = np.full(len(flow_adj), np.nan)
        flow_true_pred = np.full(len(flow_adj), np.nan)

    modeled = prediction_df.loc[:, ["row_id"]].copy()
    modeled["flow_adj"] = flow_adj
    modeled["flow_mpd_pred"] = flow_mpd_pred
    modeled["flow_true_pred"] = flow_true_pred
    base_out = prep["base_df"].copy()
    base_out["prediction_scope"] = prediction_scope
    base_out["model_fit_status"] = np.where(
        base_out["row_id"].isin(fit_df["row_id"]),
        "fit",
        np.where(
            base_out["row_id"].isin(prediction_df["row_id"]), "predicted", "excluded"
        ),
    )
    base_out = base_out.merge(
        modeled,
        how="left",
        on="row_id",
        sort=False,
        validate="one_to_one",
    )
    for col in ["observation_probability", "log_observation_probability"]:
        if col in prediction_df.columns:
            base_out = base_out.merge(
                prediction_df.loc[:, ["row_id", col]],
                how="left",
                on="row_id",
                validate="one_to_one",
            )
        elif col not in base_out.columns:
            base_out[col] = np.nan

    keep = existing_columns(base_out, keep_cols)
    select_cols = ["origin", "destination"]
    if "mpd_source" in base_out.columns:
        select_cols.append("mpd_source")
    if "mpd_time" in base_out.columns:
        select_cols.append("mpd_time")
    select_cols.extend(keep)
    select_cols.extend(
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
            "observation_probability",
            "coverage_rate_o",
            "coverage_rate_d",
            "log_observation_probability",
            "distance_km",
            "log_distance",
            "bias_e_origin",
            "log_dist_synth",
        ]
    )
    select_cols = existing_columns(base_out, select_cols)
    out = base_out.loc[:, select_cols].reset_index(drop=True)

    backend_used = "frequentist"
    effect_structure = (
        "single_random_intercept" if random_group_col else "additive_fixed"
    )
    coefficients = fit["coefficients"].copy()
    result_metadata = {
        "backend": backend_used,
        "implementation_backend": fit["implementation_backend"],
        "model_engine": model_engine,
        "model_family": model_family,
        "stage_scope": (
            "complete_grid_prediction"
            if prediction_scope == "complete_grid"
            else "observed_od_only"
        ),
        "prediction_scope": prediction_scope,
        "scenario": scenario_info["scenario"],
        "source_col": scenario_info["source_col"],
        "time_col": scenario_info["time_col"],
        "repeated_observation": scenario_info["repeated_observation"],
        "scenario_cols": scenario_info["scenario_cols"],
        "n_sources": scenario_info["n_sources"],
        "n_time_periods": scenario_info["n_time_periods"],
        "target_scale": target_scale,
        "observation_model": observation_model,
        "coverage_scale": (
            coverage_scale
            if observation_model in {"coverage_offset", "latent_two_level"}
            else np.nan
        ),
        "model_terms": formula_terms,
        "bias_terms": bias_terms,
        "random_intercept": reported_random_intercept,
        "effective_random_intercept": random_intercept,
        "random_intercept_group": random_group_col,
        "random_intercept_source": (
            "formula" if formula_random_effect is not None else "selector"
        ),
        "flow_adj_summary": flow_adj_summary,
        "distance_source": prep["distance_source"],
        "runtime_seconds": time.time() - start_time,
        "n_input_rows": len(prep["base_df"]),
        "n_fit_rows": len(fit_df),
        "n_prediction_rows": len(prediction_df),
        "n_zero_filled_prediction_rows": int(
            prediction_df["mpd_zero_filled"].isin([True]).sum()
        ),
        "od_audit": od_audit,
        "model_contract": fit["model_contract"],
        "model_notes": (
            f"engine=frequentist; family={model_family}; "
            f"effects={effect_structure}; "
            f"random_intercept={reported_random_intercept}; "
            f"effective_random_intercept={random_intercept}; "
            f"scenario={scenario_info['scenario']}."
        ),
    }
    out.attrs.update(
        model_result_attributes(
            result_metadata,
            model=fit["model"],
            formula=fit["formula"],
            coefficients=coefficients,
            implementation_backend=fit["implementation_backend"],
            random_intercept_group=random_group_col,
            scenario_info=scenario_info,
            diagnostics=fit["diagnostics"],
            model_notes=result_metadata["model_notes"],
        )
    )
    if include_flow_adj_draws:
        out.attrs["flow_adj_draws"] = np.asarray([flow_adj])
    return out
