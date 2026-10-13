"""CmdStanPy implementation of debiasR's latent two-level backend.

The module is internal so the public function can retain the R-compatible
``backend="stan_latent"`` contract while recording ``cmdstanpy`` as the runtime
backend.  CmdStanPy is imported lazily; importing :mod:`debiaspy` does not
require a local CmdStan toolchain.
"""

from __future__ import annotations

import importlib
import os
import re
import sysconfig
import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from debiaspy._exceptions import DebiasSchemaError
from debiaspy.stan import (
    CMDSTAN_COMPATIBLE_MAJOR_MINOR,
    latent_two_level_stan_path,
)

_VALID_LATENT_UNITS = {"auto", "od", "od_time"}
_VALID_LATENT_FAMILIES = {"poisson", "negbin"}
_RANDOM_EFFECT_RE = re.compile(r"\([^()]*\|[^()]*\)")
_GENERATED_QUANTITY_RE = re.compile(
    r"^(?:flow_true_pred|mu_mpd_pred|log_lik|y_rep_obs)\["
)

_CONTROL_DEFAULTS: dict[str, float | int] = {
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


def prepare_latent_state(
    data: pd.DataFrame,
    scenario_info: Mapping[str, Any],
    latent_flow_unit: str,
) -> dict[str, Any]:
    """Create source-invariant latent-flow identifiers and audit metadata.

    This ports ``.prepare_multilevel_latent_state`` from debiasR.  ``auto``
    selects OD-time states for S2 and for balanced S4 data with at least two
    observed sources per OD-time state; otherwise it selects OD states.
    """

    if latent_flow_unit not in _VALID_LATENT_UNITS:
        choices = ", ".join(sorted(_VALID_LATENT_UNITS))
        raise DebiasSchemaError(f"`latent_flow_unit` must be one of: {choices}.")
    scenario = str(scenario_info.get("scenario", "")).lower()
    if scenario not in {"s1", "s2", "s3", "s4"}:
        raise DebiasSchemaError(
            "`scenario_info['scenario']` must be one of: s1, s2, s3, s4."
        )

    out = data.copy()
    resolved_unit = latent_flow_unit
    if resolved_unit == "auto":
        balanced_s4 = scenario == "s4" and _has_replicated_sources(
            out, ("origin", "destination", "mpd_time")
        )
        if balanced_s4:
            resolved_unit = "od_time"
        elif scenario == "s4":
            warnings.warn(
                "`latent_flow_unit = 'auto'` selected `od` because at least "
                "one S4 OD-time latent state has fewer than two observed "
                "sources. Use `latent_flow_unit = 'od_time'` only for "
                "balanced source-time data, or after accepting the weaker "
                "local identification.",
                stacklevel=2,
            )
            resolved_unit = "od"
        elif scenario == "s2":
            resolved_unit = "od_time"
        else:
            resolved_unit = "od"

    id_columns = ["origin", "destination"]
    if resolved_unit == "od_time":
        id_columns.append("mpd_time")
    _require_columns(out, id_columns, "latent state data")

    id_values = _joined_identifier(out, id_columns)
    levels = sorted(pd.unique(id_values).tolist())
    out["latent_flow_unit"] = resolved_unit
    out["latent_flow_id"] = pd.Categorical(id_values, categories=levels)

    observed = _observed_mask(out)
    observed_ids = id_values[observed]
    counts = (
        pd.Series(observed_ids, dtype="string")
        .value_counts(sort=False)
        .reindex(levels, fill_value=0)
        .astype(int)
    )
    distinct_sources = _distinct_sources_by_state(out, id_values, observed)
    weak_local_replication = bool(
        scenario in {"s3", "s4"}
        and len(distinct_sources) > 0
        and (distinct_sources < 2).any()
    )
    weak_scenario = scenario in {"s1", "s2"} or weak_local_replication
    if weak_scenario:
        if weak_local_replication:
            warnings.warn(
                "`observation_model = 'latent_two_level'` has weak local "
                "identification: at least one latent state has fewer than "
                "two observed sources.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                "`observation_model = 'latent_two_level'` is weakly "
                f"identified for {scenario.upper()} without repeated source "
                "observations or strong priors.",
                stacklevel=2,
            )

    return {
        "data": out,
        "latent_flow_unit": resolved_unit,
        "n_latent_flows": len(levels),
        "identifiability": {
            "scenario": scenario,
            "repeated_observation": scenario_info.get("repeated_observation"),
            "min_observations_per_latent_flow": _series_min(counts),
            "max_observations_per_latent_flow": _series_max(counts),
            "min_sources_per_latent_flow": _series_min(distinct_sources),
            "max_sources_per_latent_flow": _series_max(distinct_sources),
            "weak_identification_warning": weak_scenario,
            "stronger_repeated_source_design": scenario in {"s3", "s4"},
        },
    }


def fit_latent_cmdstan(
    fit_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    formula_info: Mapping[str, Any],
    default_covariate_col: str | None,
    include_pop_terms: bool,
    scenario_info: Mapping[str, Any],
    model_family: str,
    controls: Mapping[str, Any],
    iter: int,
    chains: int,
    seed: int,
    refresh: int,
) -> dict[str, Any]:
    """Fit the package-owned latent model with CmdStanPy.

    RStan's ``iter`` includes warmup.  The equivalent CmdStanPy call uses half
    for warmup and the remainder for retained sampling.  The packaged model is
    byte-identical to debiasR and therefore requires CmdStan 2.32.x, the final
    release accepting its legacy array declaration syntax.
    """

    family = str(model_family).lower()
    if family not in _VALID_LATENT_FAMILIES:
        raise DebiasSchemaError(
            "`observation_model = 'latent_two_level'` supports only "
            "`model_family = 'poisson'` or `model_family = 'negbin'`."
        )
    iter_value = _positive_integer(iter, "iter", minimum=2)
    chains_value = _positive_integer(chains, "chains")
    seed_value = _positive_integer(seed, "seed")
    refresh_value = _nonnegative_integer(refresh, "refresh")
    latent_controls = _validate_controls(controls)
    cmdstanpy, cmdstan_version = _load_compatible_cmdstanpy()

    formula_contract = _build_latent_formula_info(
        formula_info=formula_info,
        default_covariate_col=default_covariate_col,
        include_pop_terms=include_pop_terms,
    )
    data_info = _build_latent_stan_data(
        fit_df=fit_df,
        prediction_df=prediction_df,
        latent_formula_info=formula_contract,
        scenario_info=scenario_info,
        model_family=family,
        controls=latent_controls,
    )

    packaged_stan_path = latent_two_level_stan_path()
    stan_path = _materialize_cmdstan_source(
        packaged_stan_path,
        cmdstan_version=cmdstan_version,
    )
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_path))
    iter_warmup = iter_value // 2
    iter_sampling = iter_value - iter_warmup
    fit = model.sample(
        data=data_info["data"],
        chains=chains_value,
        parallel_chains=chains_value,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=seed_value,
        refresh=refresh_value if refresh_value > 0 else None,
        show_progress=refresh_value > 0,
        adapt_delta=latent_controls["latent_adapt_delta"],
        max_treedepth=latent_controls["latent_max_treedepth"],
        save_warmup=False,
        sig_figs=18,
    )

    flow_true_pred_draws = _extract_draw_matrix(
        fit, "flow_true_pred", len(prediction_df)
    )
    flow_mpd_pred_draws = _extract_draw_matrix(fit, "mu_mpd_pred", len(prediction_df))
    coefficients = _coefficient_summary(fit, data_info)
    diagnostics = _collect_diagnostics(
        fit,
        data_info,
        cmdstan_version=cmdstan_version,
    )

    return {
        "stanfit": fit,
        "backend": "stan_latent",
        "runtime_backend": "cmdstanpy",
        "cmdstan_version": cmdstan_version,
        "latent_formula_info": formula_contract,
        "stan_data": data_info["data"],
        "stan_data_summary": data_info["summary"],
        "flow_true_pred_draws": flow_true_pred_draws,
        "flow_mpd_pred_draws": flow_mpd_pred_draws,
        "coefficients": coefficients,
        "diagnostics": diagnostics,
    }


def _observed_mask(data: pd.DataFrame) -> np.ndarray:
    if "mpd_observed" not in data.columns:
        return np.ones(len(data), dtype=bool)
    return data["mpd_observed"].isin([True]).to_numpy(dtype=bool)


def _joined_identifier(data: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = data.loc[:, list(columns)].astype("string")
    if values.isna().any(axis=None):
        missing = [column for column in columns if values[column].isna().any()]
        raise DebiasSchemaError(
            "Latent-state identifier columns cannot contain missing values: "
            + ", ".join(missing)
        )
    return values.agg("::".join, axis=1).to_numpy(dtype=str)


def _has_replicated_sources(
    data: pd.DataFrame,
    id_columns: Sequence[str],
) -> bool:
    if not set(id_columns).issubset(data.columns) or "mpd_source" not in data:
        return False
    observed = _observed_mask(data)
    if not observed.any():
        return False
    frame = data.loc[observed, [*id_columns, "mpd_source"]].copy()
    counts = frame.groupby(list(id_columns), sort=True, dropna=False)[
        "mpd_source"
    ].nunique(dropna=False)
    return len(counts) > 0 and bool((counts >= 2).all())


def _distinct_sources_by_state(
    data: pd.DataFrame,
    latent_ids: np.ndarray,
    observed: np.ndarray,
) -> pd.Series:
    if "mpd_source" not in data or not observed.any():
        return pd.Series(dtype=int)
    frame = pd.DataFrame(
        {
            "latent_flow_id": latent_ids[observed],
            "mpd_source": data.loc[observed, "mpd_source"].astype("string").array,
        }
    )
    return frame.groupby("latent_flow_id", sort=True)["mpd_source"].nunique(
        dropna=False
    )


def _build_latent_formula_info(
    formula_info: Mapping[str, Any],
    default_covariate_col: str | None,
    include_pop_terms: bool,
) -> dict[str, Any]:
    interface = str(formula_info.get("interface", "default"))
    if interface == "combined":
        raise DebiasSchemaError(
            "`observation_model = 'latent_two_level'` needs separate true-flow "
            "and observation-bias formulas. Use `mobility_formula` and "
            "`bias_formula`, or omit formulas to use package defaults."
        )

    if interface == "split":
        mobility_source = formula_info.get("mobility_formula")
        bias_source = formula_info.get("bias_formula")
        mobility_terms = _string_terms(formula_info.get("mobility_variables", ()))
        bias_terms = _string_terms(formula_info.get("bias_variables", ()))
        true_formula, true_random = _fixed_formula(
            mobility_source,
            fallback_terms=mobility_terms,
            drop_intercept=False,
        )
        observation_formula, bias_random = _fixed_formula(
            bias_source,
            fallback_terms=bias_terms,
            drop_intercept=True,
        )
        ignored = list(dict.fromkeys([*true_random, *bias_random]))
        ignored.extend(
            term
            for term in _string_terms(
                formula_info.get("ignored_random_effect_terms", ())
            )
            if term not in ignored
        )
        if ignored:
            warnings.warn(
                "Random-effect terms in `mobility_formula` or `bias_formula` "
                "are ignored by `observation_model = 'latent_two_level'`. "
                "The custom Stan backend uses its own latent OD/OD-time, "
                "source, and time effects.",
                stacklevel=3,
            )
        return {
            "true_formula": true_formula,
            "observation_formula": observation_formula,
            "mobility_variables": mobility_terms,
            "bias_variables": bias_terms,
            "ignored_random_effect_terms": ignored,
            "source": formula_info.get("source", "user"),
            "interface": "split",
        }

    true_terms: list[str] = []
    if default_covariate_col is not None:
        true_terms.extend(["income_o", "income_d"])
    true_terms.append("log_distance")
    if include_pop_terms:
        true_terms.extend(["log_pop_o", "log_pop_d"])
    true_formula = "~ " + " + ".join(true_terms) if true_terms else "~ 1"
    return {
        "true_formula": true_formula,
        "observation_formula": "~ 0",
        "mobility_variables": true_terms,
        "bias_variables": [],
        "ignored_random_effect_terms": [],
        "source": "default",
        "interface": "default",
    }


def _fixed_formula(
    source: Any,
    fallback_terms: Sequence[str],
    drop_intercept: bool,
) -> tuple[str, list[str]]:
    if source is None:
        rhs = " + ".join(fallback_terms) if fallback_terms else "1"
        ignored: list[str] = []
    else:
        text = str(source)
        rhs = text.split("~", 1)[1] if "~" in text else text
        ignored = [match.group(0).strip() for match in _RANDOM_EFFECT_RE.finditer(rhs)]
        rhs = _RANDOM_EFFECT_RE.sub(" ", rhs)
        rhs = _clean_formula_rhs(rhs)
    if drop_intercept:
        rhs = _drop_explicit_intercept(rhs)
        return f"~ {rhs} - 1" if rhs else "~ 0", ignored
    return f"~ {rhs or '1'}", ignored


def _clean_formula_rhs(rhs: str) -> str:
    value = re.sub(r"\s+", " ", rhs).strip()
    value = re.sub(r"^\s*\+|\+\s*$", "", value).strip()
    value = re.sub(r"\+\s*\+", "+", value)
    return value


def _drop_explicit_intercept(rhs: str) -> str:
    parts = [part.strip() for part in rhs.split("+")]
    kept = [part for part in parts if part not in {"", "0", "1", "-1"}]
    return " + ".join(kept)


def _build_latent_stan_data(
    fit_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    latent_formula_info: Mapping[str, Any],
    scenario_info: Mapping[str, Any],
    model_family: str,
    controls: Mapping[str, float | int],
) -> dict[str, Any]:
    del scenario_info  # Retained in the signature to mirror the R builder.
    if len(fit_df) < 1 or len(prediction_df) < 1:
        raise DebiasSchemaError(
            "The latent Stan backend requires at least one fit and prediction row."
        )
    _validate_count_flows(fit_df)
    required = [
        "flow",
        "latent_flow_id",
        "observation_probability",
    ]
    _require_columns(fit_df, required, "fit_df")
    _require_columns(
        prediction_df,
        ["latent_flow_id", "observation_probability"],
        "prediction_df",
    )

    true_mm = _paired_model_matrix(
        fit_df,
        prediction_df,
        str(latent_formula_info["true_formula"]),
        allow_empty=False,
    )
    bias_mm = _paired_model_matrix(
        fit_df,
        prediction_df,
        str(latent_formula_info["observation_formula"]),
        allow_empty=True,
    )
    if bias_mm["fit"].shape[1] == 0:
        bias_mm = {
            "fit": np.zeros((len(fit_df), 1), dtype=float),
            "prediction": np.zeros((len(prediction_df), 1), dtype=float),
            "terms": ["no_observation_bias_terms"],
        }
    true_mm = _standardize_paired_model_matrix(true_mm)
    bias_mm = _standardize_paired_model_matrix(bias_mm)

    latent_levels = _sorted_levels(prediction_df["latent_flow_id"])
    source_values_obs = _source_values(fit_df)
    source_values_pred = _source_values(prediction_df)
    time_values_obs = _time_values(fit_df)
    time_values_pred = _time_values(prediction_df)
    source_levels = sorted(pd.unique(source_values_pred).tolist())
    time_levels = sorted(pd.unique(time_values_pred).tolist())
    if not latent_levels or not source_levels or not time_levels:
        raise DebiasSchemaError(
            "Latent, source, and time level sets must each contain at least one level."
        )

    latent_id_obs = _one_based_ids(
        _text_values(fit_df["latent_flow_id"]), latent_levels, "latent_flow_id"
    )
    latent_id_pred = _one_based_ids(
        _text_values(prediction_df["latent_flow_id"]),
        latent_levels,
        "latent_flow_id",
    )
    source_id_obs = _one_based_ids(source_values_obs, source_levels, "mpd_source")
    source_id_pred = _one_based_ids(source_values_pred, source_levels, "mpd_source")
    time_id_obs = _one_based_ids(time_values_obs, time_levels, "mpd_time")
    time_id_pred = _one_based_ids(time_values_pred, time_levels, "mpd_time")

    q_obs = pd.to_numeric(fit_df["observation_probability"], errors="coerce").to_numpy(
        dtype=float
    )
    q_pred = pd.to_numeric(
        prediction_df["observation_probability"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(q_obs).all()
        or not np.isfinite(q_pred).all()
        or (q_obs <= 0).any()
        or (q_pred <= 0).any()
    ):
        raise DebiasSchemaError(
            "The latent Stan backend requires positive finite observation "
            "probabilities."
        )

    y = pd.to_numeric(fit_df["flow"], errors="coerce").to_numpy(dtype=float)
    median_true_scale = float(np.median(y / q_obs))
    if not np.isfinite(median_true_scale) or median_true_scale <= 0:
        median_flow = float(np.median(y))
        median_true_scale = max(median_flow, 1.0)

    true_terms = list(true_mm["terms"])
    true_intercept_col = (
        true_terms.index("(Intercept)") + 1 if "(Intercept)" in true_terms else 0
    )
    obs_count = np.bincount(latent_id_obs, minlength=len(latent_levels) + 1)[1:]
    latent_basis = _sum_to_zero_basis(len(latent_levels))
    source_basis = _sum_to_zero_basis(len(source_levels))
    time_basis = _sum_to_zero_basis(len(time_levels))

    stan_data: dict[str, Any] = {
        "N_obs": len(fit_df),
        "N_pred": len(prediction_df),
        "K_true": true_mm["fit"].shape[1],
        "K_bias": bias_mm["fit"].shape[1],
        "X_true_obs": true_mm["fit"],
        "X_true_pred": true_mm["prediction"],
        "X_bias_obs": bias_mm["fit"],
        "X_bias_pred": bias_mm["prediction"],
        "L": len(latent_levels),
        "L_basis": latent_basis.shape[1],
        "latent_basis": latent_basis,
        "latent_id_obs": latent_id_obs,
        "latent_id_pred": latent_id_pred,
        "y": np.rint(y).astype(int),
        "log_q_obs": np.log(q_obs),
        "log_q_pred": np.log(q_pred),
        "S": len(source_levels),
        "S_basis": source_basis.shape[1],
        "source_basis": source_basis,
        "source_id_obs": source_id_obs,
        "source_id_pred": source_id_pred,
        "T": len(time_levels),
        "T_basis": time_basis.shape[1],
        "time_basis": time_basis,
        "time_id_obs": time_id_obs,
        "time_id_pred": time_id_pred,
        "use_time_effect": int(len(time_levels) > 1),
        "use_negbin": int(model_family == "negbin"),
        "true_intercept_col": true_intercept_col,
        "intercept_loc": float(np.log(max(median_true_scale, 1e-8))),
        "intercept_scale": controls["latent_intercept_prior_scale"],
        "prior_coef_scale": controls["latent_coef_prior_scale"],
        "prior_bias_scale": controls["latent_bias_prior_scale"],
        "prior_latent_state_scale": controls["latent_state_prior_scale"],
        "prior_source_scale": controls["latent_source_prior_scale"],
        "prior_time_scale": controls["latent_time_prior_scale"],
        "phi_prior_rate": controls["latent_phi_prior_rate"],
        "max_rng_eta": controls["latent_rng_eta_max"],
    }
    summary = {
        "true_formula": latent_formula_info["true_formula"],
        "observation_formula": latent_formula_info["observation_formula"],
        "true_terms": true_terms,
        "observation_terms": list(bias_mm["terms"]),
        "true_standardization": true_mm["standardization"],
        "observation_standardization": bias_mm["standardization"],
        "latent_levels": latent_levels,
        "source_levels": source_levels,
        "time_levels": time_levels,
        "source_effect_layer": "observation",
        "time_effect_layer": "observation" if len(time_levels) > 1 else "none",
        "latent_controls": dict(controls),
        "n_unobserved_latent_flows": int((obs_count == 0).sum()),
        "min_observations_per_latent_flow": int(obs_count.min()),
        "max_observations_per_latent_flow": int(obs_count.max()),
    }
    return {"data": stan_data, "summary": summary}


def _paired_model_matrix(
    fit_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    formula: str,
    allow_empty: bool,
) -> dict[str, Any]:
    try:
        from patsy import EvalEnvironment, PatsyError, dmatrix
    except ImportError as exc:  # pragma: no cover - statsmodels installs patsy
        raise ImportError(
            "The latent backend requires `patsy`, which is installed with "
            "debiaspy's mandatory statsmodels dependency. Reinstall debiaspy."
        ) from exc

    combined = pd.concat([fit_df, prediction_df], ignore_index=True, sort=False)
    environment = EvalEnvironment.capture(0).with_outer_namespace(
        {"np": np, "log": np.log, "sqrt": np.sqrt, "exp": np.exp}
    )
    try:
        matrix = dmatrix(
            formula,
            data=combined,
            return_type="dataframe",
            NA_action="raise",
            eval_env=environment,
        )
    except PatsyError as exc:
        raise DebiasSchemaError(
            f"Could not construct latent model matrix for formula `{formula}`: {exc}"
        ) from exc
    if not allow_empty and matrix.shape[1] == 0:
        raise DebiasSchemaError(f"Formula produced an empty model matrix: {formula}")
    terms = [_r_style_term(str(term)) for term in matrix.columns]
    if len(terms) != len(set(terms)):
        raise DebiasSchemaError(
            f"Formula produced duplicate model-matrix columns: {formula}"
        )
    values = matrix.to_numpy(dtype=float)
    n_fit = len(fit_df)
    if values.shape[0] != len(combined):
        raise DebiasSchemaError(
            f"Formula changed the row count while building `{formula}`."
        )
    return {
        "fit": values[:n_fit],
        "prediction": values[n_fit:],
        "terms": terms,
    }


def _standardize_paired_model_matrix(
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    fit = np.asarray(matrix["fit"], dtype=float)
    prediction = np.asarray(matrix["prediction"], dtype=float)
    terms = list(matrix.get("terms", ()))
    if fit.ndim != 2 or prediction.ndim != 2 or fit.shape[1] != prediction.shape[1]:
        raise DebiasSchemaError(
            "Paired model matrices must be two-dimensional with equal columns."
        )
    if not terms:
        terms = [f"x{index}" for index in range(1, fit.shape[1] + 1)]
    if len(terms) != fit.shape[1]:
        raise DebiasSchemaError("Model-matrix term count does not match its columns.")

    center = np.zeros(fit.shape[1], dtype=float)
    scale = np.ones(fit.shape[1], dtype=float)
    standardized = np.ones(fit.shape[1], dtype=bool)
    threshold = np.sqrt(np.finfo(float).eps)
    for index, term in enumerate(terms):
        if term == "(Intercept)":
            standardized[index] = False
            continue
        values = fit[:, index]
        center[index] = float(np.nanmean(values))
        scale_value = float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
        if np.isfinite(scale_value) and scale_value > threshold:
            scale[index] = scale_value

    fit_scaled = (fit - center) / scale
    prediction_scaled = (prediction - center) / scale
    if not np.isfinite(fit_scaled).all() or not np.isfinite(prediction_scaled).all():
        raise DebiasSchemaError(
            "Latent model matrices contain non-finite values after standardization."
        )
    return {
        "fit": fit_scaled,
        "prediction": prediction_scaled,
        "terms": terms,
        "standardization": pd.DataFrame(
            {
                "term": terms,
                "center": center,
                "scale": scale,
                "standardized": standardized,
            }
        ),
    }


def _sum_to_zero_basis(n: int) -> np.ndarray:
    size = _positive_integer(n, "sum-to-zero level count")
    columns = max(size - 1, 1)
    basis = np.zeros((size, columns), dtype=float)
    if size == 1:
        return basis
    for index in range(1, size):
        denominator = np.sqrt(index * (index + 1))
        basis[:index, index - 1] = 1 / denominator
        basis[index, index - 1] = -index / denominator
    return basis


def _validate_count_flows(fit_df: pd.DataFrame) -> None:
    _require_columns(fit_df, ["flow"], "fit_df")
    y = pd.to_numeric(fit_df["flow"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise DebiasSchemaError(
            "The latent Stan backend requires finite non-negative observed flows."
        )
    if (np.abs(y - np.rint(y)) > np.sqrt(np.finfo(float).eps)).any():
        raise DebiasSchemaError(
            "The latent Stan backend requires count-valued observed flows."
        )


def _validate_controls(controls: Mapping[str, Any]) -> dict[str, float | int]:
    values: dict[str, Any] = dict(_CONTROL_DEFAULTS)
    values.update(dict(controls))
    positive_names = [
        "latent_coef_prior_scale",
        "latent_bias_prior_scale",
        "latent_intercept_prior_scale",
        "latent_state_prior_scale",
        "latent_source_prior_scale",
        "latent_time_prior_scale",
        "latent_phi_prior_rate",
        "latent_rng_eta_max",
    ]
    out: dict[str, float | int] = {}
    for name in positive_names:
        out[name] = _positive_float(values[name], name)
    adapt_delta = _positive_float(values["latent_adapt_delta"], "latent_adapt_delta")
    if adapt_delta >= 1:
        raise DebiasSchemaError(
            "`latent_adapt_delta` must be a single finite number between 0 and 1."
        )
    out["latent_adapt_delta"] = adapt_delta
    out["latent_max_treedepth"] = _positive_integer(
        values["latent_max_treedepth"], "latent_max_treedepth"
    )
    return out


def _load_compatible_cmdstanpy() -> tuple[Any, str]:
    try:
        cmdstanpy = importlib.import_module("cmdstanpy")
    except ImportError as exc:
        raise ImportError(
            "Backend 'stan_latent' requires `cmdstanpy` and CmdStan 2.32.x. "
            "Install cmdstanpy, install CmdStan 2.32.x, and set `CMDSTAN` to "
            "that installation."
        ) from exc
    try:
        cmdstanpy.cmdstan_path()
    except (AttributeError, RuntimeError, ValueError):
        configured = os.environ.get("DEBIASPY_CMDSTAN")
        candidates = [
            Path(configured).expanduser() if configured else None,
            Path.cwd() / ".cmdstan" / "cmdstan-2.32.2",
        ]
        for candidate in candidates:
            if candidate is not None and (candidate / "bin" / "stanc").is_file():
                cmdstanpy.set_cmdstan_path(str(candidate.resolve()))
                break
    try:
        cmdstanpy.cmdstan_path()
        version_value = cmdstanpy.cmdstan_version()
    except (AttributeError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Backend 'stan_latent' requires a working CmdStan 2.32.x "
            "installation. Run `python scripts/install_cmdstan.py`, or set "
            "`CMDSTAN`/`DEBIASPY_CMDSTAN` to its path."
        ) from exc
    if version_value is None:
        raise RuntimeError(
            "CmdStanPy could not determine the CmdStan version. Backend "
            "'stan_latent' requires CmdStan 2.32.x."
        )
    version_parts = tuple(int(part) for part in version_value)
    required = CMDSTAN_COMPATIBLE_MAJOR_MINOR
    if version_parts[:2] != required:
        found = ".".join(map(str, version_parts))
        raise RuntimeError(
            "The byte-identical debiasR latent Stan model requires CmdStan "
            f"{required[0]}.{required[1]}.x because its legacy array syntax "
            f"was removed in CmdStan 2.33; found CmdStan {found}. Set "
            "`CMDSTAN` to a compatible installation."
        )
    return cmdstanpy, ".".join(map(str, version_parts))


def _materialize_cmdstan_source(
    packaged_path: Path,
    cmdstan_version: str,
) -> Path:
    """Copy the immutable package resource into a writable compilation cache."""

    configured = os.environ.get("DEBIASPY_CMDSTAN_CACHE")
    if configured:
        cache_root = Path(configured).expanduser()
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        cache_root = (
            (Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache")
            / "debiaspy"
            / "stan"
        )
    target_dir = cache_root / f"cmdstan-{cmdstan_version}" / sysconfig.get_platform()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            "Backend 'stan_latent' could not create its CmdStan compilation "
            f"cache at `{target_dir}`. Set `DEBIASPY_CMDSTAN_CACHE` to a "
            "writable directory."
        ) from exc
    target = target_dir / packaged_path.name
    source = packaged_path.read_bytes()
    target_matches = target.is_file() and target.read_bytes() == source
    if target_matches:
        return target
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(source)
        temporary.replace(target)
    except OSError as exc:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Backend 'stan_latent' could not write its byte-identical Stan "
            f"source to `{target_dir}`. Set `DEBIASPY_CMDSTAN_CACHE` to a "
            "writable directory."
        ) from exc
    return target


def _extract_draw_matrix(fit: Any, variable: str, expected_columns: int) -> np.ndarray:
    try:
        draws = np.asarray(fit.stan_variable(variable), dtype=float)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not extract expected Stan generated quantity `{variable}`."
        ) from exc
    if draws.ndim == 1 and expected_columns == 1:
        draws = draws[:, None]
    if draws.ndim != 2 or draws.shape[1] != expected_columns:
        raise RuntimeError(
            f"Stan generated quantity `{variable}` has shape {draws.shape}; "
            f"expected (draws, {expected_columns})."
        )
    if not np.isfinite(draws).all():
        raise RuntimeError(
            f"Stan generated quantity `{variable}` contains non-finite draws."
        )
    return draws


def _coefficient_summary(fit: Any, data_info: Mapping[str, Any]) -> pd.DataFrame:
    arrays: list[np.ndarray] = []
    terms: list[str] = []
    for parameter in (
        "beta_true",
        "beta_bias",
        "source_effect",
        "time_effect",
        "phi",
    ):
        try:
            values = np.asarray(fit.stan_variable(parameter), dtype=float)
        except (KeyError, RuntimeError, ValueError):
            continue
        if values.ndim == 1:
            values = values[:, None]
        elif values.ndim != 2:
            raise RuntimeError(
                f"Stan parameter `{parameter}` has unsupported shape {values.shape}."
            )
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Stan parameter `{parameter}` contains non-finite draws."
            )
        arrays.append(values)
        terms.extend(
            [parameter]
            if parameter == "phi" and values.shape[1] == 1
            else [f"{parameter}[{index}]" for index in range(1, values.shape[1] + 1)]
        )
    if not arrays:
        raise RuntimeError("CmdStan fit contains no latent-model coefficient draws.")
    draws = np.column_stack(arrays)
    result = pd.DataFrame(
        {
            "term": terms,
            "mean": np.mean(draws, axis=0),
            "sd": np.std(draws, axis=0, ddof=1),
            "q2.5": np.quantile(draws, 0.025, axis=0),
            "q97.5": np.quantile(draws, 0.975, axis=0),
        }
    )
    result["layer"] = result["term"].map(_coefficient_layer)
    true_terms = list(data_info["summary"]["true_terms"])
    bias_terms = list(data_info["summary"]["observation_terms"])
    result["model_term"] = [
        _coefficient_model_term(term, layer, true_terms, bias_terms)
        for term, layer in zip(result["term"], result["layer"], strict=True)
    ]
    return result


def _collect_diagnostics(
    fit: Any,
    data_info: Mapping[str, Any],
    cmdstan_version: str,
) -> dict[str, Any]:
    method_variables = fit.method_variables()
    divergent = _method_array(method_variables, "divergent__")
    treedepth = _method_array(method_variables, "treedepth__")
    accept_stat = _method_array(method_variables, "accept_stat__")
    energy = _method_array(method_variables, "energy__")
    controls = data_info["summary"]["latent_controls"]
    max_treedepth = int(controls["latent_max_treedepth"])

    sampler_shape = _cmdstan_chain_draw_shape(divergent)
    n_draws_per_chain, n_chains = sampler_shape
    sampler_shape_valid = all(
        _cmdstan_chain_draw_shape(values) == sampler_shape
        for values in (divergent, treedepth, accept_stat, energy)
    )
    n_post_warmup_draws = int(divergent.size)
    divergences = int(np.nansum(divergent))
    treedepth_hits = int(np.nansum(treedepth >= max_treedepth))
    ebfmi = _ebfmi_by_chain(energy)

    try:
        summary = fit.summary(sig_figs=18)
    except TypeError:  # Small test doubles and older compatible CmdStanPy.
        summary = fit.summary()
    if not isinstance(summary, pd.DataFrame):
        summary = pd.DataFrame(summary)
    parameter_mask = np.array(
        [not _GENERATED_QUANTITY_RE.match(str(name)) for name in summary.index]
    )
    parameter_summary = summary.loc[parameter_mask]
    rhat = _summary_column(parameter_summary, ("R_hat", "Rhat"))
    ess_bulk = _summary_column(parameter_summary, ("ESS_bulk", "N_Eff"))
    ess_tail = _summary_column(parameter_summary, ("ESS_tail",))
    mcse = _summary_column(parameter_summary, ("MCSE",))
    posterior_sd = _summary_column(parameter_summary, ("StdDev", "SD"))
    stochastic = np.isfinite(posterior_sd) & (posterior_sd > 0)
    mcse_over_sd = np.divide(
        mcse,
        posterior_sd,
        out=np.full_like(mcse, np.nan),
        where=np.isfinite(posterior_sd) & (posterior_sd > 0),
    )

    return {
        "backend": "stan_latent",
        "runtime_backend": "cmdstanpy",
        "cmdstan_version": cmdstan_version,
        "has_bias_term": any(
            term != "no_observation_bias_terms"
            for term in data_info["summary"]["observation_terms"]
        ),
        "convergence": {
            "status": "available",
            "n_chains": n_chains,
            "draws_per_chain": n_draws_per_chain,
            "n_post_warmup_draws": n_post_warmup_draws,
            "draw_shape_valid": int(sampler_shape_valid),
            "sampler_shape_valid": int(sampler_shape_valid),
            "divergences": divergences,
            "divergence_rate": _safe_rate(divergences, n_post_warmup_draws),
            "treedepth_hits": treedepth_hits,
            "treedepth_hit_rate": _safe_rate(treedepth_hits, n_post_warmup_draws),
            "treedepth_max": _finite_max(treedepth),
            "max_treedepth": max_treedepth,
            "adapt_delta": controls["latent_adapt_delta"],
            "accept_stat_mean": _finite_mean(accept_stat),
            "accept_stat_min": _finite_min(accept_stat),
            "ebfmi_min": (
                float(np.min(ebfmi))
                if ebfmi.size == n_chains
                and bool(np.isfinite(ebfmi).all())
                and bool((ebfmi > 0).all())
                else np.nan
            ),
            "rhat_max": _strict_max_for_stochastic(rhat, stochastic),
            "n_eff_min": _strict_min_for_stochastic(ess_bulk, stochastic),
            "ess_bulk_min": _strict_min_for_stochastic(ess_bulk, stochastic),
            "ess_tail_min": _strict_min_for_stochastic(ess_tail, stochastic),
            "mcse_over_sd_max": _strict_max_for_stochastic(mcse_over_sd, stochastic),
        },
        "posterior_predictive": {
            "status": "available",
            "generated_quantity": "y_rep_obs",
            "rng_eta_max": controls["latent_rng_eta_max"],
        },
        "latent_state": data_info["summary"],
    }


def _method_array(method_variables: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in method_variables:
        raise RuntimeError(f"CmdStan sampler diagnostics are missing `{name}`.")
    values = np.asarray(method_variables[name], dtype=float)
    if values.ndim not in {1, 2}:
        raise RuntimeError(
            f"CmdStan sampler diagnostic `{name}` has unsupported shape {values.shape}."
        )
    return values


def _cmdstan_chain_draw_shape(values: np.ndarray) -> tuple[int, int]:
    """Return draws-per-chain and chains for CmdStan method-variable arrays."""

    array = np.asarray(values)
    if array.ndim == 1:
        return int(array.shape[0]), 1
    if array.ndim == 2:
        return int(array.shape[0]), int(array.shape[1])
    return (0, 0)


def _ebfmi_by_chain(energy: np.ndarray) -> np.ndarray:
    values = energy[:, None] if energy.ndim == 1 else energy
    output = np.full(values.shape[1], np.nan, dtype=float)
    for chain in range(values.shape[1]):
        chain_energy = values[:, chain]
        finite = chain_energy[np.isfinite(chain_energy)]
        if len(finite) < 3:
            continue
        variance = float(np.var(finite, ddof=1))
        if not np.isfinite(variance) or variance <= 0:
            continue
        output[chain] = float(np.mean(np.diff(finite) ** 2) / variance)
    return output


def _summary_column(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return np.array([], dtype=float)


def _coefficient_layer(term: str) -> str:
    if term.startswith("beta_true["):
        return "true_flow"
    if term.startswith("beta_bias["):
        return "observation_bias"
    if term.startswith("source_effect["):
        return "observation_source_effect"
    if term.startswith("time_effect["):
        return "observation_time_effect"
    if term == "phi":
        return "overdispersion"
    return "other"


def _coefficient_model_term(
    term: str,
    layer: str,
    true_terms: Sequence[str],
    bias_terms: Sequence[str],
) -> str:
    match = re.search(r"\[([0-9]+)\]$", term)
    if match is None:
        return term
    index = int(match.group(1)) - 1
    if layer == "true_flow" and 0 <= index < len(true_terms):
        return str(true_terms[index])
    if layer == "observation_bias" and 0 <= index < len(bias_terms):
        return str(bias_terms[index])
    return term


def _source_values(data: pd.DataFrame) -> np.ndarray:
    if "mpd_source" in data:
        return _text_values(data["mpd_source"])
    return np.repeat("source_1", len(data))


def _time_values(data: pd.DataFrame) -> np.ndarray:
    if "mpd_time" in data:
        return _text_values(data["mpd_time"])
    return np.repeat("time_1", len(data))


def _text_values(values: pd.Series) -> np.ndarray:
    text = values.astype("string")
    if text.isna().any():
        raise DebiasSchemaError(f"`{values.name}` cannot contain missing values.")
    return text.to_numpy(dtype=str)


def _sorted_levels(values: pd.Series) -> list[str]:
    return sorted(pd.unique(_text_values(values)).tolist())


def _one_based_ids(
    values: np.ndarray,
    levels: Sequence[str],
    name: str,
) -> np.ndarray:
    lookup = {level: index for index, level in enumerate(levels, start=1)}
    missing = sorted(set(values) - set(lookup))
    if missing:
        raise DebiasSchemaError(
            f"Prediction levels for `{name}` are missing fit value `{missing[0]}`."
        )
    return np.asarray([lookup[value] for value in values], dtype=int)


def _r_style_term(term: str) -> str:
    if term == "Intercept":
        return "(Intercept)"
    pattern = re.compile(r"(?:C\()?([A-Za-z_][A-Za-z0-9_]*)\)?\[(?:T\.)?([^\]]+)\]")
    return pattern.sub(lambda match: match.group(1) + match.group(2), term)


def _string_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(term) for term in value]
    return [str(value)]


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise DebiasSchemaError(
            f"`{name}` is missing required column(s): {', '.join(missing)}"
        )


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DebiasSchemaError(f"`{name}` must be a positive finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DebiasSchemaError(f"`{name}` must be a positive finite number.") from exc
    if not np.isfinite(result) or result <= 0:
        raise DebiasSchemaError(f"`{name}` must be a positive finite number.")
    return result


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise DebiasSchemaError(f"`{name}` must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DebiasSchemaError(f"`{name}` must be a positive integer.") from exc
    if not np.isfinite(numeric) or numeric != int(numeric) or numeric < minimum:
        raise DebiasSchemaError(f"`{name}` must be a positive integer.")
    return int(numeric)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise DebiasSchemaError(f"`{name}` must be a non-negative integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DebiasSchemaError(f"`{name}` must be a non-negative integer.") from exc
    if not np.isfinite(numeric) or numeric != int(numeric) or numeric < 0:
        raise DebiasSchemaError(f"`{name}` must be a non-negative integer.")
    return int(numeric)


def _series_min(values: pd.Series) -> int | None:
    return int(values.min()) if len(values) > 0 else None


def _series_max(values: pd.Series) -> int | None:
    return int(values.max()) if len(values) > 0 else None


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else np.nan


def _finite_min(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if len(finite) else np.nan


def _finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else np.nan


def _strict_min_for_stochastic(values: np.ndarray, stochastic: np.ndarray) -> float:
    if values.shape != stochastic.shape or not stochastic.any():
        return np.nan
    selected = values[stochastic]
    return float(np.min(selected)) if bool(np.isfinite(selected).all()) else np.nan


def _strict_max_for_stochastic(values: np.ndarray, stochastic: np.ndarray) -> float:
    if values.shape != stochastic.shape or not stochastic.any():
        return np.nan
    selected = values[stochastic]
    return float(np.max(selected)) if bool(np.isfinite(selected).all()) else np.nan


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else np.nan


__all__ = ["fit_latent_cmdstan", "prepare_latent_state"]
