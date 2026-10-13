from __future__ import annotations

import importlib
import importlib.metadata
import re
import time
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from debiaspy import _multilevel_inputs as shared
from debiaspy._adjustment_common import existing_columns, model_result_attributes
from debiaspy._exceptions import DebiasSchemaError

_BAMBI_FAMILIES = {
    "poisson": "poisson",
    "negbin": "negativebinomial",
    "zip": "zero_inflated_poisson",
    "zinb": "zero_inflated_negativebinomial",
}
_AUXILIARY_POSTERIOR_VARIABLES = {"alpha", "psi"}
_FORMULA_FUNCTIONS = {
    "C",
    "I",
    "Q",
    "bs",
    "center",
    "exp",
    "log",
    "np",
    "offset",
    "poly",
    "scale",
    "sqrt",
}


@dataclass(eq=False, slots=True)
class BayesianFit:
    """In-memory Bambi model and posterior draws attached to an adjustment."""

    model: Any
    inference_data: Any
    backend: str
    runtime_backend: str = "bambi_pymc"

    def __deepcopy__(self, memo: dict[int, Any]) -> BayesianFit:
        """Preserve the live fit handle when pandas deep-copies frame attrs."""

        memo[id(self)] = self
        return self


@dataclass(slots=True)
class _PredictionSummary:
    mean: np.ndarray
    median: np.ndarray
    q2_5: np.ndarray
    q97_5: np.ndarray
    draws: np.ndarray | None
    total_draws: np.ndarray | None = None


def adjust_multilevel_bayesian(
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
    random_intercept: str = "origin",
    formula_info: Mapping[str, Any] | None = None,
    model_family: str = "poisson",
    backend: str = "auto",
    flow_adj_summary: str = "mean",
    target_scale: str = "mpd_counterfactual",
    observation_model: str = "reduced_form",
    coverage_scale: str = "origin",
    prediction_scope: str = "observed",
    iter: int = 1000,
    chains: int = 2,
    seed: int = 123,
    refresh: int = 0,
    include_flow_adj_draws: bool = False,
    keep_cols: str | Iterable[str] = (),
    start_time: float | None = None,
) -> pd.DataFrame:
    """Fit the non-latent debiasR Bayesian model with Bambi and PyMC.

    Receives the resolved formula contract from ``adjust_multilevel_bayes``.
    """

    started_at = time.time() if start_time is None else start_time
    _validate_choice(model_family, _BAMBI_FAMILIES, "model_family")
    _validate_choice(flow_adj_summary, {"mean", "median"}, "flow_adj_summary")
    _validate_choice(target_scale, {"mpd_counterfactual", "true_flow"}, "target_scale")
    _validate_choice(
        observation_model,
        {"reduced_form", "coverage_offset", "latent_two_level"},
        "observation_model",
    )
    _validate_choice(
        coverage_scale, {"origin", "destination", "both"}, "coverage_scale"
    )
    _validate_choice(
        prediction_scope, {"observed", "complete_grid"}, "prediction_scope"
    )
    if observation_model == "latent_two_level":
        raise NotImplementedError(
            "`observation_model='latent_two_level'` uses the CmdStanPy latent "
            "backend and is not handled by the Bambi/PyMC backend."
        )
    if target_scale == "true_flow" and observation_model != "coverage_offset":
        raise DebiasSchemaError(
            "`target_scale = 'true_flow'` requires "
            "`observation_model = 'coverage_offset'` for the non-latent backend."
        )
    if observation_model == "coverage_offset" and target_scale != "true_flow":
        raise DebiasSchemaError(
            "`observation_model = 'coverage_offset'` is supported only with "
            "`target_scale = 'true_flow'`."
        )

    logical_backend = _resolve_logical_backend(
        model_family=model_family,
        backend=backend,
        observation_model=observation_model,
    )
    tune, draws = _sampling_schedule(
        iter=iter, chains=chains, seed=seed, refresh=refresh
    )
    resolved_formula_info = dict(
        formula_info
        or shared._resolve_formula_info(None, None, None, allow_extended=True)
    )

    scenario_info = shared._resolve_scenario(
        mpd_od_df,
        source_col=source_col,
        time_col=time_col,
        scenario=scenario,
        repeated_observation=repeated_observation,
    )
    od_audit = None
    if prediction_scope == "complete_grid":
        od_audit = shared._audit_complete_grid(
            mpd_od_df,
            flow_col=flow_col,
            scenario_cols=scenario_info["audit_cols"],
        )
        if not bool(od_audit["strict_square_support"].iloc[0]):
            raise DebiasSchemaError(
                "`prediction_scope = 'complete_grid'` requires a strict square OD grid."
            )

    prep = shared._prepare_multilevel_data(
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

    scenario_terms = _scenario_fixed_effects(
        prediction_df,
        repeated_observation=scenario_info["repeated_observation"],
        random_intercept=random_intercept,
    )
    fit_formula, model_terms, bias_terms = _build_formula_contract(
        formula_info=resolved_formula_info,
        default_covariate_col=prep["default_covariate_col"],
        include_pop_terms=bool(prep["has_pop_terms"]),
        scenario_terms=scenario_terms,
        random_intercept=random_intercept,
        target_scale=target_scale,
        observation_model=observation_model,
    )

    if target_scale == "true_flow":
        shared._validate_coverage_offset_data(prediction_df, coverage_scale)
        prediction_df = shared._set_observation_probability(
            prediction_df, coverage_scale
        )
    else:
        prediction_df["observation_probability"] = np.nan
        prediction_df["log_observation_probability"] = np.nan

    prediction_df = _filter_formula_data(
        prediction_df,
        formula=fit_formula,
        context="prediction",
    )
    prediction_df = _stabilize_formula_categories(prediction_df, fit_formula)
    fit_df = prediction_df
    if prediction_scope == "complete_grid" and "mpd_observed" in fit_df.columns:
        fit_df = fit_df.loc[fit_df["mpd_observed"].isin([True])].copy()
    if len(fit_df) < 2:
        raise DebiasSchemaError(
            "Insufficient rows for fitting after preprocessing. "
            "Need at least 2 complete rows."
        )

    bmb, az = _load_bayesian_dependencies()
    prior_overrides, prior_contract = _prior_contract(
        bmb=bmb,
        logical_backend=logical_backend,
        model_family=model_family,
        data=fit_df,
        target_scale=target_scale,
    )
    model = bmb.Model(
        fit_formula,
        fit_df,
        family=_BAMBI_FAMILIES[model_family],
        dropna=False,
        auto_scale=True,
        priors=prior_overrides or None,
    )
    if logical_backend == "rstanarm":
        prior_contract.update(_apply_rstanarm_marginal_priors(bmb, model))
    elif logical_backend == "brms":
        prior_contract.update(_apply_brms_group_priors(bmb, model))
    prior_contract["python_resolved"] = _resolved_model_priors(model)
    # Random-intercept geometries require stricter NUTS adaptation to reduce the
    # risk of platform-sensitive divergences. The hierarchical ZIP geometry
    # needs a smaller step size than the other audited random-intercept fits.
    # These controls do not change the likelihood, priors, posterior target, or
    # estimand.
    if (
        logical_backend == "brms"
        and model_family == "zip"
        and random_intercept != "none"
    ):
        target_accept = 0.999
    elif random_intercept != "none":
        target_accept = 0.99
    else:
        target_accept = 0.95
    max_treedepth = 15 if logical_backend == "rstanarm" else 10
    inference_data = model.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=1,
        random_seed=seed,
        target_accept=target_accept,
        nuts={"max_treedepth": max_treedepth},
        progressbar=refresh > 0,
    )

    if target_scale == "true_flow":
        mpd_summary = _predict_summary(
            model,
            inference_data,
            prediction_df,
            include_draws=False,
        )
        true_prediction_df = prediction_df.copy()
        true_prediction_df["log_observation_probability"] = 0.0
        adjusted_summary = _predict_summary(
            model,
            inference_data,
            true_prediction_df,
            include_draws=include_flow_adj_draws,
        )
    else:
        counterfactual_df = _counterfactual_bias_data(prediction_df, bias_terms)
        adjusted_summary = _predict_summary(
            model,
            inference_data,
            counterfactual_df,
            include_draws=include_flow_adj_draws,
        )
        mpd_summary = _empty_prediction_summary(len(prediction_df))

    output = _assemble_output(
        prep=prep,
        prediction_df=prediction_df,
        fit_df=fit_df,
        adjusted_summary=adjusted_summary,
        mpd_summary=mpd_summary,
        flow_adj_summary=flow_adj_summary,
        target_scale=target_scale,
        prediction_scope=prediction_scope,
        keep_cols=keep_cols,
    )
    coefficients = _coefficient_summary(model, inference_data)
    diagnostics = _collect_diagnostics(
        az=az,
        inference_data=inference_data,
        backend=logical_backend,
        chains=chains,
        draws=draws,
        has_bias_term=bool(bias_terms),
    )
    diagnostics["adjusted_flow_total"] = _adjusted_flow_diagnostics(
        az, inference_data, adjusted_summary.total_draws
    )
    if target_scale == "mpd_counterfactual":
        diagnostics["counterfactual_support"] = {
            term: {
                "fit_min": float(fit_df[term].min()),
                "fit_max": float(fit_df[term].max()),
                "counterfactual_value": 0.0,
                "outside_fit_range": not bool(
                    fit_df[term].min() <= 0 <= fit_df[term].max()
                ),
            }
            for term in bias_terms
        }
    _warn_unstable_adjusted_mean(diagnostics["adjusted_flow_total"], flow_adj_summary)
    runtime_seconds = time.time() - started_at
    stage_scope = (
        "complete_grid_prediction"
        if prediction_scope == "complete_grid"
        else "observed_od_only"
    )
    model_notes = _model_notes(
        target_scale=target_scale,
        prediction_scope=prediction_scope,
        coverage_scale=coverage_scale,
        flow_adj_summary=flow_adj_summary,
    )
    versions = _runtime_versions()
    model_contract = _nonlatent_model_contract(model_family)
    result_metadata = {
        "backend": logical_backend,
        "runtime_backend": "bambi_pymc",
        "runtime_versions": versions,
        "resolved_priors": prior_contract,
        "model_engine": "bayesian",
        "model_family": model_family,
        "model_contract": model_contract,
        "stage_scope": stage_scope,
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
        "coverage_scale": coverage_scale if target_scale == "true_flow" else np.nan,
        "offset_column": (
            "log_observation_probability" if target_scale == "true_flow" else np.nan
        ),
        "model_terms": model_terms,
        "bias_terms": bias_terms,
        "random_intercept": random_intercept,
        "flow_adj_summary": flow_adj_summary,
        "distance_source": prep["distance_source"],
        "runtime_seconds": runtime_seconds,
        "n_input_rows": len(prep["base_df"]),
        "n_fit_rows": len(fit_df),
        "n_prediction_rows": len(prediction_df),
        "n_zero_filled_prediction_rows": int(
            prediction_df["mpd_zero_filled"].isin([True]).sum()
        ),
        "sampling": {
            "seed": seed,
            "chains": chains,
            "iter": iter,
            "warmup_per_chain": tune,
            "draws_per_chain": draws,
            "cores": 1,
            "target_accept": target_accept,
            "max_treedepth": max_treedepth,
        },
        "od_audit": od_audit,
    }
    fit_handle = BayesianFit(
        model=model,
        inference_data=inference_data,
        backend=logical_backend,
    )
    output.attrs.update(
        model_result_attributes(
            result_metadata,
            model=fit_handle,
            formula=fit_formula,
            coefficients=coefficients,
            runtime_backend="bambi_pymc",
            runtime_versions=versions,
            resolved_priors=prior_contract,
            scenario_info=scenario_info,
            diagnostics=diagnostics,
            model_notes=model_notes,
        )
    )
    if include_flow_adj_draws:
        output.attrs["flow_adj_draws"] = adjusted_summary.draws
    return output


def _validate_choice(value: str, choices: Iterable[str], argument: str) -> None:
    choices_tuple = tuple(choices)
    if value not in choices_tuple:
        raise ValueError(f"`{argument}` must be one of: {', '.join(choices_tuple)}")


def _resolve_logical_backend(
    model_family: str,
    backend: str,
    observation_model: str = "reduced_form",
) -> str:
    if observation_model == "latent_two_level":
        raise NotImplementedError(
            "The latent model is handled by the CmdStanPy backend."
        )
    _validate_choice(backend, {"auto", "rstanarm", "brms", "stan_latent"}, "backend")
    if backend == "stan_latent":
        raise DebiasSchemaError(
            "`backend = 'stan_latent'` is valid only with "
            "`observation_model = 'latent_two_level'`."
        )
    resolved = (
        "rstanarm"
        if backend == "auto" and model_family in {"poisson", "negbin"}
        else "brms"
        if backend == "auto"
        else backend
    )
    if resolved == "rstanarm" and model_family in {"zip", "zinb"}:
        raise DebiasSchemaError(
            f"`model_family = '{model_family}'` requires `backend = 'brms'` or "
            "`backend = 'auto'`."
        )
    return resolved


def _sampling_schedule(
    iter: int, chains: int, seed: int, refresh: int
) -> tuple[int, int]:
    _require_integer(iter, "iter", minimum=2)
    _require_integer(chains, "chains", minimum=1)
    _require_integer(seed, "seed", minimum=0)
    _require_integer(refresh, "refresh", minimum=0)
    tune = iter // 2
    return tune, iter - tune


def _require_integer(value: int, name: str, minimum: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise DebiasSchemaError(f"`{name}` must be a single integer.")
    if int(value) < minimum:
        raise DebiasSchemaError(f"`{name}` must be at least {minimum}.")


def _scenario_fixed_effects(
    data: pd.DataFrame,
    repeated_observation: str,
    random_intercept: str,
) -> list[str]:
    terms: list[str] = []
    if (
        repeated_observation in {"source", "source_time"}
        and random_intercept != "source"
        and "mpd_source" in data.columns
        and data["mpd_source"].dropna().astype(str).nunique() > 1
    ):
        terms.append("mpd_source")
    if (
        repeated_observation in {"time", "source_time"}
        and random_intercept != "time"
        and "mpd_time" in data.columns
        and data["mpd_time"].dropna().astype(str).nunique() > 1
    ):
        terms.append("mpd_time")
    return terms


def _build_formula_contract(
    formula_info: Mapping[str, Any],
    default_covariate_col: str | None,
    include_pop_terms: bool,
    scenario_terms: list[str],
    random_intercept: str,
    target_scale: str,
    observation_model: str,
) -> tuple[str, dict[str, Any], list[str]]:
    interface = str(formula_info.get("interface", "default"))
    user_formula = formula_info.get("formula")
    default_terms: list[str] = []
    if user_formula is None:
        if default_covariate_col is not None:
            default_terms.extend(["income_o", "income_d"])
        default_terms.append("log_distance")
        if target_scale == "mpd_counterfactual":
            default_terms.append("bias_e_origin")
        if include_pop_terms:
            default_terms.extend(["log_pop_o", "log_pop_d"])
        default_terms.extend(scenario_terms)
        random_term = _random_intercept_term(random_intercept)
        if random_term is not None:
            default_terms.append(random_term)
        formula = "flow ~ " + " + ".join(default_terms)
        bias_terms = ["bias_e_origin"] if target_scale == "mpd_counterfactual" else []
        formula_source = "default"
    elif interface == "split" and observation_model == "coverage_offset":
        mobility_text = str(formula_info["mobility_formula"])
        formula = "flow ~ " + mobility_text.split("~", 1)[1].strip()
        bias_terms = []
        formula_source = "split_formula"
        interface = "split_true_flow"
    else:
        formula = str(user_formula)
        bias_terms = list(formula_info.get("bias_variables", []))
        formula_source = str(formula_info.get("source", interface))

    if target_scale == "true_flow":
        formula = _append_formula_term(formula, "offset(log_observation_probability)")

    random_effects = re.findall(r"\([^()]*(?:\||\|\|)[^()]*\)", formula)
    model_terms = {
        "formula": formula,
        "formula_source": formula_source,
        "formula_interface": interface,
        "mobility_formula": formula_info.get("mobility_formula"),
        "bias_formula": formula_info.get("bias_formula"),
        "mobility_variables": list(formula_info.get("mobility_variables", [])),
        "bias_variables": bias_terms,
        "user_formula": user_formula is not None,
        "default_area_covariate": default_covariate_col,
        "default_fixed_effects": default_terms,
        "scenario_fixed_effects": [] if user_formula is not None else scenario_terms,
        "formula_variables": _formula_variables(formula),
        "requested_random_intercept": random_intercept,
        "random_effect_term": random_effects
        or _random_intercept_term(random_intercept),
        "formula_random_effects": random_effects,
    }
    return formula, model_terms, list(dict.fromkeys(bias_terms))


def _random_intercept_term(random_intercept: str) -> str | None:
    return {
        "origin": "(1 | origin)",
        "destination": "(1 | destination)",
        "od": "(1 | od_id)",
        "source": "(1 | mpd_source)",
        "time": "(1 | mpd_time)",
        "source_time": "(1 | mpd_source_time)",
        "none": None,
    }[random_intercept]


def _append_formula_term(formula: str, term: str) -> str:
    return f"{formula} + {term}"


def _formula_variables(formula: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", formula)
    return list(
        dict.fromkeys(
            token
            for token in tokens
            if token not in _FORMULA_FUNCTIONS
            and token != "flow"
            and not token.startswith("np.")
        )
    )


def _filter_formula_data(
    data: pd.DataFrame,
    formula: str,
    context: str,
) -> pd.DataFrame:
    variables = _formula_variables(formula)
    missing = [variable for variable in variables if variable not in data.columns]
    if missing:
        raise DebiasSchemaError(
            "`formula` references variable(s) not found in prepared model data: "
            + ", ".join(missing)
        )
    keep = np.ones(len(data), dtype=bool)
    for variable in ["flow", *variables]:
        values = data[variable]
        if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(numeric)
            if variable == "flow":
                valid &= numeric >= 0
        else:
            strings = values.astype("string")
            valid = (
                strings.notna().to_numpy()
                & strings.fillna("").str.len().gt(0).to_numpy()
            )
        keep &= valid
    dropped = int((~keep).sum())
    if dropped:
        warnings.warn(
            f"{dropped} row(s) were excluded from {context} data because `formula` "
            "variables were missing, non-finite, or invalid.",
            stacklevel=2,
        )
    return data.loc[keep].copy()


def _stabilize_formula_categories(data: pd.DataFrame, formula: str) -> pd.DataFrame:
    out = data.copy()
    for variable in _formula_variables(formula):
        if variable not in out.columns:
            continue
        values = out[variable]
        if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            levels = sorted(values.dropna().astype(str).unique())
            out[variable] = pd.Categorical(values.astype(str), categories=levels)
    return out


def _counterfactual_bias_data(
    data: pd.DataFrame, bias_terms: list[str]
) -> pd.DataFrame:
    out = data.copy()
    for term in bias_terms:
        if term not in out.columns:
            continue
        if not (
            pd.api.types.is_numeric_dtype(out[term])
            or pd.api.types.is_bool_dtype(out[term])
        ):
            raise DebiasSchemaError(
                "`bias_formula` fixed-effect variable(s) must be numeric, integer, "
                "or logical so the zero-bias counterfactual can set them to zero: "
                f"{term}"
            )
        out[term] = False if pd.api.types.is_bool_dtype(out[term]) else 0
    return out


def _load_bayesian_dependencies() -> tuple[Any, Any]:
    try:
        bmb = importlib.import_module("bambi")
        az = importlib.import_module("arviz")
    except ImportError as exc:
        raise ImportError(
            "Bayesian multilevel adjustment requires the Bambi/PyMC runtime. "
            "Install the runtime with `pip install debiaspy[bayesian]`."
        ) from exc
    except Exception as exc:
        raise ImportError(
            "The Bambi/PyMC runtime is installed but could not be initialized. "
            "Install the runtime with `pip install debiaspy[bayesian]`."
        ) from exc
    return bmb, az


def _prior_contract(
    bmb: Any,
    logical_backend: str,
    model_family: str,
    data: pd.DataFrame | None = None,
    target_scale: str = "mpd_counterfactual",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return explicit overrides and auditable prior-profile metadata."""

    overrides: dict[str, Any] = {}
    profile: dict[str, Any] = {
        "reference_backend": logical_backend,
        "common_effects": (
            "Bambi automatic auto-scaled Normal priors; analogous to the "
            "implicit rstanarm common-effect prior, not draw-identical"
            if logical_backend == "rstanarm"
            else "Flat prior, matching the brms default for population-level slopes"
        ),
        "intercept": (
            "Bambi automatic auto-scaled Normal prior"
            if logical_backend == "rstanarm"
            else "brms data-scaled Student-t(3, location, scale) default"
        ),
        "overrides": {},
        "identity_claim": False,
    }
    if logical_backend == "rstanarm" and model_family == "negbin":
        overrides["alpha"] = bmb.Prior("Exponential", auto_scale=False, lam=1.0)
        profile["overrides"] = {"alpha": "Exponential(lam=1)"}
    if logical_backend == "brms":
        if data is None:
            raise ValueError("`data` is required to resolve the brms prior contract.")
        location, scale = _brms_intercept_prior(data, target_scale)
        overrides.update(
            {
                "common": bmb.Prior("Flat", auto_scale=False),
                "Intercept": bmb.Prior(
                    "StudentT",
                    auto_scale=False,
                    nu=3.0,
                    mu=location,
                    sigma=scale,
                ),
            }
        )
        resolved = {
            "common": "Flat()",
            "Intercept": (f"StudentT(nu=3, mu={location:.17g}, sigma={scale:.17g})"),
        }
        if model_family in {"zip", "zinb"}:
            overrides["psi"] = bmb.Prior("Beta", auto_scale=False, alpha=1.0, beta=1.0)
            resolved["psi"] = "Beta(alpha=1, beta=1)"
        if model_family in {"negbin", "zinb"}:
            overrides["alpha"] = bmb.Prior(
                "InverseGamma", auto_scale=False, alpha=0.4, beta=0.3
            )
            resolved["alpha"] = "InverseGamma(alpha=0.4, beta=0.3)"
        profile["overrides"] = resolved
        profile["prior_parameter_identity"] = True
        profile["identity_claim"] = True
        profile["intercept_location"] = location
        profile["intercept_scale"] = scale
    if model_family in {"zip", "zinb"}:
        profile["prediction_parameter"] = (
            "mu (count-component intensity; not the unconditional mixture mean)"
        )
    return overrides, profile


def _brms_intercept_prior(data: pd.DataFrame, target_scale: str) -> tuple[float, float]:
    """Resolve brms' default log-link intercept prior from the fit rows."""

    flow = pd.to_numeric(data["flow"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(flow).all() or (flow < 0).any():
        raise DebiasSchemaError(
            "The brms prior contract requires finite non-negative flow values."
        )
    adjusted = np.where(flow == 0, 0.1, flow)
    linked = np.log(adjusted)
    location = float(np.round(np.median(linked), 1))
    median = float(np.median(linked))
    mad = float(1.4826 * np.median(np.abs(linked - median)))
    scale = max(2.5, float(np.round(mad, 1))) if np.isfinite(mad) else 2.5
    if target_scale == "true_flow":
        offset = pd.to_numeric(
            data["log_observation_probability"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(offset).all():
            raise DebiasSchemaError(
                "The brms prior contract requires a finite observation offset."
            )
        location -= float(np.mean(offset))
    return location, scale


def _resolved_model_priors(model: Any) -> dict[str, Any]:
    """Serialize priors from the fitted Bambi model as parameterized trees."""

    resolved: dict[str, Any] = {}
    components = getattr(model, "components", {})
    items = components.items() if hasattr(components, "items") else ()
    for component_name, component in items:
        intercept = getattr(component, "intercept_term", None)
        if intercept is not None and getattr(intercept, "prior", None) is not None:
            resolved[f"{component_name}.Intercept"] = _structured_prior(intercept.prior)
        for collection_name in ("common_terms", "group_specific_terms"):
            collection = getattr(component, collection_name, {})
            term_items = collection.items() if hasattr(collection, "items") else ()
            for term_name, term in term_items:
                prior = getattr(term, "prior", None)
                if prior is not None:
                    resolved[f"{component_name}.{term_name}"] = _structured_prior(prior)
        prior = getattr(component, "prior", None)
        if prior is not None:
            resolved[str(component_name)] = _structured_prior(prior)
    return dict(sorted(resolved.items()))


def _structured_prior(prior: Any) -> dict[str, Any]:
    """Return a JSON-safe distribution tree without trusting display strings."""

    name = getattr(prior, "name", None)
    if not isinstance(name, str) or not name:
        raise RuntimeError("A fitted Bambi prior has no distribution name.")
    raw_args = getattr(prior, "args", None)
    if raw_args is None:
        raw_args = getattr(prior, "kwargs", None)
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, Mapping):
        raise RuntimeError(f"Prior `{name}` has invalid distribution parameters.")
    parameters = {
        str(key): _structured_prior_value(value)
        for key, value in sorted(raw_args.items(), key=lambda item: str(item[0]))
        if str(key) != "auto_scale"
    }
    auto_scale = getattr(prior, "auto_scale", raw_args.get("auto_scale"))
    return {
        "distribution": name,
        "parameters": parameters,
        "auto_scale": None if auto_scale is None else bool(auto_scale),
    }


def _structured_prior_value(value: Any) -> Any:
    if getattr(value, "name", None) is not None:
        return _structured_prior(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _structured_prior_value(value.item())
        return [_structured_prior_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_structured_prior_value(item) for item in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise RuntimeError("A fitted Bambi prior contains a non-finite parameter.")
        return numeric
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    raise RuntimeError(
        "A fitted Bambi prior contains an unsupported parameter value of type "
        f"{type(value).__name__}."
    )


def _apply_rstanarm_marginal_priors(bmb: Any, model: Any) -> dict[str, Any]:
    """Apply rstanarm's centered fixed- and single-intercept priors."""

    if not hasattr(model, "set_priors"):
        return {
            "rstanarm_marginal_prior_scales": {},
            "prior_parameterization": "test_double_unavailable",
        }
    parent_name = getattr(
        getattr(getattr(model, "family", None), "likelihood", None), "parent", "mu"
    )
    components = getattr(model, "components", {})
    component = components.get(parent_name) if hasattr(components, "get") else None
    common_terms = getattr(component, "common_terms", {})
    term_items = common_terms.items() if hasattr(common_terms, "items") else ()
    priors: dict[str, Any] = {}
    marginal_scales: dict[str, Any] = {}
    for name, term in term_items:
        values = np.asarray(getattr(term, "data", np.asarray([])), dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            continue
        scales = np.std(values, axis=0, ddof=1)
        scales = np.divide(
            2.5,
            scales,
            out=np.full_like(scales, 2.5, dtype=float),
            where=np.isfinite(scales) & (scales > np.sqrt(np.finfo(float).eps)),
        )
        sigma: float | np.ndarray = float(scales[0]) if len(scales) == 1 else scales
        priors[str(name)] = bmb.Prior("Normal", auto_scale=False, mu=0.0, sigma=sigma)
        marginal_scales[str(name)] = (
            float(scales[0]) if len(scales) == 1 else scales.tolist()
        )

    centered = getattr(model, "center_predictors", None) is True
    if not centered:
        raise RuntimeError(
            "The rstanarm prior contract requires Bambi `center_predictors=True`."
        )
    priors["Intercept"] = bmb.Prior("Normal", auto_scale=False, mu=0.0, sigma=2.5)
    marginal_scales["Intercept_centered"] = 2.5

    group_terms = getattr(component, "group_specific_terms", {})
    group_items = list(group_terms.items()) if hasattr(group_terms, "items") else []
    terms_by_group: dict[str, list[tuple[str, str]]] = {}
    unsupported_group_terms: list[str] = []
    for raw_name, _term in group_items:
        name = str(raw_name)
        if "|" not in name:
            unsupported_group_terms.append(name)
            continue
        effect, group = (part.strip() for part in name.rsplit("|", maxsplit=1))
        terms_by_group.setdefault(group, []).append((name, effect))

    group_prior_contract: dict[str, str] = {}
    for grouped_terms in terms_by_group.values():
        if len(grouped_terms) == 1 and grouped_terms[0][1] == "1":
            name = grouped_terms[0][0]
            group_scale = bmb.Prior("Exponential", auto_scale=False, lam=1.0)
            priors[name] = bmb.Prior(
                "Normal", auto_scale=False, mu=0.0, sigma=group_scale
            )
            group_prior_contract[name] = "Normal(mu=0, sigma); sigma=Exponential(lam=1)"
        else:
            unsupported_group_terms.extend(name for name, _effect in grouped_terms)

    if unsupported_group_terms:
        names = ", ".join(sorted(unsupported_group_terms))
        raise NotImplementedError(
            "The rstanarm-compatible backend currently supports prior-identical "
            "group structure only for single random intercepts; unsupported "
            f"term(s): {names}."
        )

    model.set_priors(priors=priors)
    return {
        "common_effects": (
            "Normal(0, 2.5 / sample_sd(design_column)), matching rstanarm "
            "autoscaled centered-design slope priors"
        ),
        "intercept": (
            "Normal(0, 2.5) on the centered intercept; Bambi "
            "center_predictors=True matches rstanarm"
        ),
        "rstanarm_marginal_prior_scales": marginal_scales,
        "rstanarm_centered_intercept_scale": 2.5,
        "rstanarm_group_specific_priors": group_prior_contract,
        "unsupported_group_specific_terms": [],
        "fixed_effect_prior_identity": True,
        "group_specific_prior_identity": True,
        "prior_parameter_identity": True,
        "identity_claim": True,
        "prior_marginal_parameter_identity": True,
        "prior_parameterization": (
            "rstanarm and Bambi sample centered fixed effects; Bambi restores "
            "the uncentered posterior intercept after sampling"
        ),
    }


def _apply_brms_group_priors(bmb: Any, model: Any) -> dict[str, Any]:
    """Apply brms' default half-Student-t prior for random-intercept SDs."""

    if not hasattr(model, "set_priors"):
        return {
            "brms_group_specific_priors": {},
            "group_specific_prior_identity": False,
        }
    parent_name = getattr(
        getattr(getattr(model, "family", None), "likelihood", None), "parent", "mu"
    )
    components = getattr(model, "components", {})
    component = components.get(parent_name) if hasattr(components, "get") else None
    group_terms = getattr(component, "group_specific_terms", {})
    group_items = list(group_terms.items()) if hasattr(group_terms, "items") else []
    terms_by_group: dict[str, list[tuple[str, str]]] = {}
    for raw_name, _term in group_items:
        name = str(raw_name)
        if "|" not in name:
            raise NotImplementedError(
                "The brms-compatible backend cannot establish prior identity for "
                f"group-specific term `{name}`."
            )
        effect, group = (part.strip() for part in name.rsplit("|", maxsplit=1))
        terms_by_group.setdefault(group, []).append((name, effect))

    priors: dict[str, Any] = {}
    contracts: dict[str, str] = {}
    for grouped_terms in terms_by_group.values():
        if len(grouped_terms) != 1 or grouped_terms[0][1] != "1":
            names = ", ".join(name for name, _effect in grouped_terms)
            raise NotImplementedError(
                "The brms-compatible backend currently supports prior-identical "
                "group structure only for single random intercepts; unsupported "
                f"term(s): {names}."
            )
        name = grouped_terms[0][0]
        group_scale = bmb.Prior("HalfStudentT", auto_scale=False, nu=3.0, sigma=2.5)
        priors[name] = bmb.Prior("Normal", auto_scale=False, mu=0.0, sigma=group_scale)
        contracts[name] = "Normal(mu=0, sigma); sigma=HalfStudentT(nu=3, sigma=2.5)"
    if priors:
        model.set_priors(priors=priors)
    return {
        "brms_group_specific_priors": contracts,
        "group_specific_prior_identity": True,
        "prior_parameter_identity": True,
        "identity_claim": True,
    }


def _nonlatent_model_contract(model_family: str) -> dict[str, str | None]:
    """Return the machine-readable likelihood and prediction contract."""

    is_negative_binomial = model_family in {"negbin", "zinb"}
    is_zero_inflated = model_family in {"zip", "zinb"}
    likelihood = {
        "poisson": "Poisson(mu)",
        "negbin": "NegativeBinomial2(mu, shape)",
        "zip": "ZeroInflatedPoisson(mu, zi)",
        "zinb": "ZeroInflatedNegativeBinomial2(mu, shape, zi)",
    }[model_family]
    return {
        "likelihood": likelihood,
        "link": "log",
        "mean_parameter": "mu",
        "prediction_functional": "count_component_mu",
        "negative_binomial_variance": (
            "mu + mu^2 / shape" if is_negative_binomial else None
        ),
        "shape_parameter_mapping": (
            "Bambi/PyMC alpha = brms/rstanarm shape" if is_negative_binomial else None
        ),
        "zero_inflation_reference_parameter": "zi" if is_zero_inflated else None,
        "zero_inflation_runtime_parameter": "psi" if is_zero_inflated else None,
        "zero_inflation_parameter_mapping": (
            "psi = 1 - zi" if is_zero_inflated else None
        ),
        "zero_inflation_reference_semantics": (
            "structural_zero_probability" if is_zero_inflated else None
        ),
        "zero_inflation_runtime_semantics": (
            "count_component_probability" if is_zero_inflated else None
        ),
    }


def _predict_summary(
    model: Any,
    inference_data: Any,
    data: pd.DataFrame,
    include_draws: bool,
    chunk_size: int = 2048,
) -> _PredictionSummary:
    n_rows = len(data)
    means = np.empty(n_rows, dtype=float)
    medians = np.empty(n_rows, dtype=float)
    q2_5 = np.empty(n_rows, dtype=float)
    q97_5 = np.empty(n_rows, dtype=float)
    all_draws: np.ndarray | None = None
    draw_count: int | None = None
    total_draws: np.ndarray | None = None
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        draws = _predict_parent_draws(
            model,
            inference_data,
            data.iloc[start:stop].copy(),
        )
        if draw_count is None:
            draw_count = int(draws.shape[0])
            if include_draws:
                all_draws = np.empty((draws.shape[0], n_rows), dtype=float)
        elif draws.shape[0] != draw_count:
            raise RuntimeError("Bambi returned an inconsistent posterior draw count.")
        chunk_totals = np.sum(draws, axis=1)
        total_draws = (
            chunk_totals if total_draws is None else total_draws + chunk_totals
        )
        means[start:stop] = np.mean(draws, axis=0)
        medians[start:stop] = np.median(draws, axis=0)
        quantiles = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
        q2_5[start:stop] = quantiles[0]
        q97_5[start:stop] = quantiles[1]
        if all_draws is not None:
            all_draws[:, start:stop] = draws
    return _PredictionSummary(means, medians, q2_5, q97_5, all_draws, total_draws)


def _predict_parent_draws(
    model: Any,
    inference_data: Any,
    data: pd.DataFrame,
) -> np.ndarray:
    predicted = model.predict(
        inference_data,
        kind="response_params",
        data=data,
        inplace=False,
        include_group_specific=False,
    )
    if predicted is None:
        raise RuntimeError("Bambi did not return response-parameter predictions.")
    posterior = predicted.posterior
    parent = getattr(getattr(model, "family", None), "likelihood", None)
    parent_name = getattr(parent, "parent", "mu")
    candidates = [parent_name, "mu", "p", "lam"]
    variable = next((name for name in candidates if name in posterior), None)
    if variable is None:
        raise RuntimeError(
            "Could not find the Bambi likelihood parent in prediction output."
        )
    values = np.asarray(posterior[variable], dtype=float)
    if values.ndim < 2:
        raise RuntimeError("Bambi prediction output has an invalid draw shape.")
    if values.shape[-1] != len(data):
        raise RuntimeError("Bambi prediction output does not match prediction rows.")
    draws = values.reshape((-1, values.shape[-1]))
    if not bool(np.isfinite(draws).all()):
        raise RuntimeError(
            "Bambi response-parameter predictions contain non-finite draws."
        )
    return draws


def _empty_prediction_summary(n_rows: int) -> _PredictionSummary:
    empty = np.full(n_rows, np.nan, dtype=float)
    return _PredictionSummary(
        empty.copy(), empty.copy(), empty.copy(), empty.copy(), None
    )


def _assemble_output(
    prep: Mapping[str, Any],
    prediction_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    adjusted_summary: _PredictionSummary,
    mpd_summary: _PredictionSummary,
    flow_adj_summary: str,
    target_scale: str,
    prediction_scope: str,
    keep_cols: str | Iterable[str],
) -> pd.DataFrame:
    adjusted = (
        adjusted_summary.median
        if flow_adj_summary == "median"
        else adjusted_summary.mean
    )
    mpd = mpd_summary.median if flow_adj_summary == "median" else mpd_summary.mean
    modeled = prediction_df.loc[:, ["row_id"]].copy()
    modeled["flow_adj"] = adjusted
    modeled["flow_mpd_pred"] = mpd
    modeled["flow_true_pred"] = adjusted if target_scale == "true_flow" else np.nan
    modeled["flow_adj_mean"] = adjusted_summary.mean
    modeled["flow_adj_median"] = adjusted_summary.median
    modeled["flow_adj_q2.5"] = adjusted_summary.q2_5
    modeled["flow_adj_q97.5"] = adjusted_summary.q97_5
    modeled["flow_mpd_pred_mean"] = mpd_summary.mean
    modeled["flow_mpd_pred_median"] = mpd_summary.median
    modeled["flow_mpd_pred_q2.5"] = mpd_summary.q2_5
    modeled["flow_mpd_pred_q97.5"] = mpd_summary.q97_5
    for column in ("observation_probability", "log_observation_probability"):
        modeled[column] = prediction_df[column].to_numpy()

    base = prep["base_df"].copy()
    base["prediction_scope"] = prediction_scope
    base["model_fit_status"] = np.where(
        base["row_id"].isin(fit_df["row_id"]),
        "fit",
        np.where(base["row_id"].isin(prediction_df["row_id"]), "predicted", "excluded"),
    )
    for column in ("observation_probability", "log_observation_probability"):
        if column in base.columns:
            base = base.drop(columns=column)
    base = base.merge(modeled, how="left", on="row_id", sort=False)

    selected = ["origin", "destination"]
    selected.extend(
        column for column in ("mpd_source", "mpd_time") if column in base.columns
    )
    selected.extend(existing_columns(base, keep_cols))
    selected.extend(
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
            "coverage_rate_o",
            "coverage_rate_d",
            "log_observation_probability",
            "distance_km",
            "log_distance",
            "bias_e_origin",
            "log_dist_synth",
        ]
    )
    selected = existing_columns(base, selected)
    return base.loc[:, selected].reset_index(drop=True)


def _common_effect_names(model: Any, posterior: Any) -> list[str]:
    parent = getattr(getattr(model, "family", None), "likelihood", None)
    parent_name = getattr(parent, "parent", "mu")
    components = getattr(model, "components", {})
    component = components.get(parent_name) if hasattr(components, "get") else None
    common_terms = getattr(component, "common_terms", None)
    names: list[str] = []
    intercept = getattr(component, "intercept_term", None)
    if intercept is not None:
        names.append(
            str(
                getattr(intercept, "alias", None)
                or getattr(intercept, "name", None)
                or "Intercept"
            )
        )
    if isinstance(common_terms, Mapping):
        names.extend(map(str, common_terms.keys()))
    elif common_terms is not None:
        for term in common_terms:
            names.append(
                str(getattr(term, "alias", None) or getattr(term, "name", term))
            )
    available = list(getattr(posterior, "data_vars", posterior.keys()))
    names = list(dict.fromkeys(name for name in names if name in available))
    if names:
        return names
    return [
        name
        for name in available
        if name not in _AUXILIARY_POSTERIOR_VARIABLES
        and name != parent_name
        and "|" not in name
        and not name.endswith("_sigma")
    ]


def _coefficient_summary(model: Any, inference_data: Any) -> pd.DataFrame:
    posterior = inference_data.posterior
    rows: list[dict[str, Any]] = []
    for variable in _common_effect_names(model, posterior):
        value = posterior[variable]
        array = np.asarray(value, dtype=float)
        if array.ndim < 2:
            continue
        coefficient_shape = array.shape[2:]
        samples = array.reshape((array.shape[0] * array.shape[1], -1))
        labels = _coefficient_labels(variable, value, coefficient_shape)
        for index, label in enumerate(labels):
            draws = samples[:, index]
            rows.append(
                {
                    "term": "(Intercept)" if label == "Intercept" else label,
                    "mean": float(np.mean(draws)),
                    "sd": float(np.std(draws, ddof=1)) if len(draws) > 1 else np.nan,
                    "q2.5": float(np.quantile(draws, 0.025, method="linear")),
                    "q97.5": float(np.quantile(draws, 0.975, method="linear")),
                }
            )
    family_name = str(
        getattr(
            getattr(model, "family", None),
            "name",
            getattr(model, "family_name", ""),
        )
    )
    canonical_auxiliaries: list[tuple[str, str, bool]] = []
    if family_name in {"negativebinomial", "zero_inflated_negativebinomial"}:
        canonical_auxiliaries.append(("alpha", "shape", False))
    if family_name in {"zero_inflated_poisson", "zero_inflated_negativebinomial"}:
        canonical_auxiliaries.append(("psi", "zi", True))
    for variable, term, complement in canonical_auxiliaries:
        if variable not in posterior:
            continue
        values = np.asarray(posterior[variable], dtype=float)
        if values.ndim != 2:
            continue
        draws = values.reshape(-1)
        if complement:
            draws = 1.0 - draws
        rows.append(
            {
                "term": term,
                "mean": float(np.mean(draws)),
                "sd": float(np.std(draws, ddof=1)) if len(draws) > 1 else np.nan,
                "q2.5": float(np.quantile(draws, 0.025, method="linear")),
                "q97.5": float(np.quantile(draws, 0.975, method="linear")),
            }
        )
    return pd.DataFrame(rows, columns=["term", "mean", "sd", "q2.5", "q97.5"])


def _coefficient_labels(
    variable: str,
    value: Any,
    coefficient_shape: tuple[int, ...],
) -> list[str]:
    if not coefficient_shape:
        return [variable]
    dims = list(getattr(value, "dims", ()))[2:]
    coords = getattr(value, "coords", {})
    coordinate_values: list[list[str]] = []
    for dim, size in zip(dims, coefficient_shape, strict=False):
        if dim in coords:
            coordinate_values.append([str(item) for item in np.asarray(coords[dim])])
        else:
            coordinate_values.append([str(index + 1) for index in range(size)])
    if not coordinate_values:
        return [
            f"{variable}[{index + 1}]"
            for index in range(int(np.prod(coefficient_shape)))
        ]
    labels: list[str] = []
    for index in np.ndindex(coefficient_shape):
        suffix = ",".join(
            coordinate_values[axis][position] for axis, position in enumerate(index)
        )
        labels.append(f"{variable}[{suffix}]")
    return labels


def _collect_diagnostics(
    az: Any,
    inference_data: Any,
    backend: str,
    chains: int,
    draws: int,
    has_bias_term: bool,
) -> dict[str, Any]:
    actual_chains, actual_draws = _posterior_chain_draw_shape(inference_data)
    expected_shape = (chains, draws)
    actual_shape = (actual_chains, actual_draws)
    convergence: dict[str, Any] = {
        "status": "available",
        "n_chains": actual_chains,
        "draws_per_chain": actual_draws,
        "n_post_warmup_draws": actual_chains * actual_draws,
        "draw_shape_valid": int(actual_shape == expected_shape),
    }
    try:
        summary = az.summary(
            inference_data,
            kind="diagnostics",
            round_to="none",
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        summary = None
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        metric_map = {
            "r_hat": ("rhat_max", np.nanmax),
            "ess_bulk": ("ess_bulk_min", np.nanmin),
            "ess_tail": ("ess_tail_min", np.nanmin),
            "mcse_mean": ("mcse_mean_max", np.nanmax),
        }
        for column, (name, reducer) in metric_map.items():
            if column in summary.columns:
                values = pd.to_numeric(summary[column], errors="coerce").to_numpy(float)
                if values.size and bool(np.isfinite(values).all()):
                    convergence[name] = float(reducer(values))

        try:
            stats_summary = az.summary(
                inference_data,
                kind="stats",
                round_to="none",
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            stats_summary = None
        if (
            "mcse_mean" in summary.columns
            and isinstance(stats_summary, pd.DataFrame)
            and "sd" in stats_summary.columns
        ):
            aligned = pd.concat(
                [
                    pd.to_numeric(summary["mcse_mean"], errors="coerce").rename(
                        "mcse_mean"
                    ),
                    pd.to_numeric(stats_summary["sd"], errors="coerce").rename(
                        "posterior_sd"
                    ),
                ],
                axis=1,
                join="inner",
            )
            mcse = aligned["mcse_mean"].to_numpy(dtype=float)
            posterior_sd = aligned["posterior_sd"].to_numpy(dtype=float)
            ratio = np.divide(
                mcse,
                posterior_sd,
                out=np.full_like(mcse, np.nan),
                where=(
                    np.isfinite(mcse) & np.isfinite(posterior_sd) & (posterior_sd > 0)
                ),
            )
            stochastic = np.isfinite(posterior_sd) & (posterior_sd > 0)
            if stochastic.any() and bool(np.isfinite(ratio[stochastic]).all()):
                convergence["mcse_over_sd_max"] = float(np.max(ratio[stochastic]))

    sample_stats = getattr(inference_data, "sample_stats", {})
    divergences = _sample_stat_array(sample_stats, "diverging")
    sampler_shape_valid = True
    if divergences is not None and _sample_stat_shape(divergences) == actual_shape:
        divergence_count = int(np.asarray(divergences, dtype=bool).sum())
        convergence["divergences"] = divergence_count
        convergence["divergence_rate"] = divergence_count / max(
            actual_chains * actual_draws, 1
        )
    else:
        sampler_shape_valid = False
    reached_depth = _sample_stat_array(sample_stats, "reached_max_treedepth")
    if reached_depth is not None and _sample_stat_shape(reached_depth) == actual_shape:
        hit_count = int(np.asarray(reached_depth, dtype=bool).sum())
        convergence["treedepth_hits"] = hit_count
        convergence["treedepth_hit_rate"] = hit_count / max(
            actual_chains * actual_draws, 1
        )
    else:
        sampler_shape_valid = False
    convergence["sampler_shape_valid"] = int(sampler_shape_valid)
    tree_depth = _sample_stat_array(sample_stats, "tree_depth")
    if tree_depth is not None and np.asarray(tree_depth).size:
        convergence["treedepth_max"] = float(np.nanmax(tree_depth))
    acceptance = _sample_stat_array(sample_stats, "acceptance_rate")
    if acceptance is not None and np.asarray(acceptance).size:
        convergence["accept_stat_mean"] = float(np.nanmean(acceptance))
        convergence["accept_stat_min"] = float(np.nanmin(acceptance))
    try:
        bfmi = np.asarray(az.bfmi(inference_data), dtype=float)
    except (AttributeError, TypeError, ValueError):
        bfmi = np.asarray([], dtype=float)
    if (
        bfmi.ndim == 1
        and bfmi.size == actual_chains
        and bool(np.isfinite(bfmi).all())
        and bool((bfmi > 0).all())
    ):
        convergence["ebfmi_min"] = float(np.min(bfmi))
    return {
        "backend": backend,
        "runtime_backend": "bambi_pymc",
        "has_bias_term": has_bias_term,
        "convergence": convergence,
    }


def _sample_stat_array(sample_stats: Any, name: str) -> np.ndarray | None:
    try:
        if name not in sample_stats:
            return None
        return np.asarray(sample_stats[name])
    except (KeyError, TypeError):
        return None


def _adjusted_flow_diagnostics(
    az: Any, inference_data: Any, totals: np.ndarray | None
) -> dict[str, Any]:
    """Diagnose the total adjusted-flow functional independently of coefficients."""
    chains, draws = _posterior_chain_draw_shape(inference_data)
    if totals is None or chains < 2 or len(totals) != chains * draws:
        return {
            "status": "unavailable",
            "reason": "insufficient chain/draw information",
        }
    if not hasattr(az, "from_dict"):
        return {"status": "unavailable", "reason": "ArviZ conversion unavailable"}
    if not np.isfinite(totals).all():
        return {"status": "nonfinite", "reason": "total adjusted-flow draws overflow"}
    posterior = az.from_dict(
        posterior={"total_adjusted_flow": totals.reshape(chains, draws)}
    )
    summary = az.summary(posterior, round_to="none").loc["total_adjusted_flow"]
    sd = float(summary["sd"])
    mean = float(summary["mean"])
    relative_mcse = float(summary["mcse_mean"]) / abs(mean) if mean != 0 else np.nan
    return {
        "status": "available",
        "functional": "sum of adjusted flows across prediction rows",
        "mean": mean,
        "sd": sd,
        "mcse_mean": float(summary["mcse_mean"]),
        "mcse_relative_mean": relative_mcse,
        "mean_precision": (
            "unstable"
            if np.isfinite(relative_mcse) and relative_mcse > 0.05
            else "within_mcse_threshold"
            if np.isfinite(relative_mcse)
            else "unavailable"
        ),
        "relative_mcse_warning_threshold": 0.05,
        "largest_draw_share_of_sum": (
            float(np.max(totals) / np.sum(totals)) if np.sum(totals) > 0 else np.nan
        ),
        "mcse_over_sd": float(summary["mcse_mean"]) / sd if sd > 0 else 0.0,
        "ess_bulk": float(summary["ess_bulk"]),
        "ess_tail": float(summary["ess_tail"]),
        "rhat": float(summary["r_hat"]),
    }


def _warn_unstable_adjusted_mean(
    diagnostics: Mapping[str, Any], flow_adj_summary: str
) -> None:
    """Report imprecise means without changing the R-compatible estimator."""
    if flow_adj_summary != "mean" or diagnostics.get("mean_precision") != "unstable":
        return
    relative_mcse = float(diagnostics["mcse_relative_mean"])
    warnings.warn(
        "The total adjusted-flow posterior mean is poorly determined: "
        f"Monte Carlo error is {relative_mcse:.1%} of its estimated mean "
        "(warning threshold 5%). Good coefficient convergence does not establish "
        "precise counterfactual flow means. Inspect adjusted_flow_total diagnostics "
        "and counterfactual support; the returned R-compatible mean is unchanged.",
        RuntimeWarning,
        stacklevel=3,
    )


def _posterior_chain_draw_shape(inference_data: Any) -> tuple[int, int]:
    posterior = getattr(inference_data, "posterior", None)
    sizes = getattr(posterior, "sizes", None)
    if sizes is not None and "chain" in sizes and "draw" in sizes:
        return int(sizes["chain"]), int(sizes["draw"])
    variables = getattr(posterior, "data_vars", None)
    names = list(variables) if variables is not None else list(posterior or {})
    for name in names:
        values = np.asarray(posterior[name])
        if values.ndim >= 2:
            return int(values.shape[0]), int(values.shape[1])
    return 0, 0


def _sample_stat_shape(values: np.ndarray) -> tuple[int, int]:
    array = np.asarray(values)
    if array.ndim != 2:
        return (0, 0)
    return int(array.shape[0]), int(array.shape[1])


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("bambi", "pymc", "pytensor", "formulae", "arviz"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _model_notes(
    target_scale: str,
    prediction_scope: str,
    coverage_scale: str,
    flow_adj_summary: str,
) -> str:
    if target_scale == "true_flow":
        return (
            "target_scale=true_flow; flow_adj=flow_true_pred; "
            "flow_mpd_pred=posterior prediction on the observed-MPD scale; "
            f"offset=log(observation_probability[{coverage_scale}])."
        )
    if prediction_scope == "complete_grid":
        return (
            "stage=1; fit_rows=mpd_observed; prediction_rows=complete_grid; "
            "counterfactual=bias_terms_zero; "
            f"flow_adj_summary={flow_adj_summary}."
        )
    return (
        "stage=1; fit_rows=mpd_observed; prediction_rows=mpd_observed; "
        "counterfactual=bias_terms_zero; "
        f"flow_adj_summary={flow_adj_summary}."
    )
