"""Frequentist count-model fitting and prediction."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from debiaspy._exceptions import DebiasSchemaError


def _fit_count_model(
    fit_df: pd.DataFrame,
    formula: str,
    model_family: str,
    target_scale: str,
    *,
    random_group_col: str | None = None,
    display_formula: str | None = None,
) -> dict[str, Any]:
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError(
            "adjust_multilevel_bayes requires the base debiaspy dependency "
            "`statsmodels`; reinstall debiaspy because this installation is "
            "incomplete."
        ) from exc
    y, full_x, design_info = _fit_design_matrices(fit_df, formula)
    active_indices = _r_style_independent_columns(full_x.to_numpy(dtype=float))
    active_columns = [str(full_x.columns[index]) for index in active_indices]
    aliased_columns = [
        str(column) for column in full_x.columns if str(column) not in active_columns
    ]
    x = full_x.iloc[:, active_indices].copy()
    offset = (
        fit_df["log_observation_probability"].to_numpy(dtype=float)
        if target_scale == "true_flow" and "log_observation_probability" in fit_df
        else None
    )
    if random_group_col is not None:
        return _fit_gpboost_random_intercept(
            fit_df=fit_df,
            y=y,
            x=x,
            design_info=design_info,
            fixed_formula=formula,
            display_formula=display_formula or formula,
            model_family=model_family,
            offset=offset,
            group_col=random_group_col,
            aliased_columns=aliased_columns,
        )
    if model_family == "poisson":
        model = sm.GLM(y, x, family=sm.families.Poisson(), offset=offset)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in scalar divide",
                category=RuntimeWarning,
                module=r"statsmodels\.regression\._tools",
            )
            result = model.fit()
        converged = bool(result.converged)
        fixed_params = np.asarray(result.params, dtype=float)
        fixed_bse = np.asarray(result.bse, dtype=float)
        implementation_backend = "statsmodels_glm_poisson"
        dispersion: dict[str, float | str] | None = None
        model_contract = {
            "likelihood": "poisson",
            "link": "log",
            "conditional_mean": "exp(X beta + offset)",
            "conditional_variance": "mu",
            "dispersion": None,
            "estimation": "maximum_likelihood_irls",
            "rank_deficiency": "r_ordered_qr_alias_drop",
        }
    elif model_family == "negbin":
        try:
            from statsmodels.discrete.discrete_model import NegativeBinomial
        except ImportError as exc:  # pragma: no cover - part of statsmodels
            raise ImportError(
                "The frequentist negative-binomial backend requires the "
                "statsmodels discrete-model module. Reinstall debiaspy."
            ) from exc
        model = NegativeBinomial(
            y,
            x,
            loglike_method="nb2",
            offset=offset,
            check_rank=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Maximum Likelihood optimization failed to converge",
            )
            result = model.fit(
                method="newton",
                maxiter=100,
                disp=False,
            )
        mle_retvals = getattr(result, "mle_retvals", {})
        converged = bool(mle_retvals.get("converged", False))
        all_params = np.asarray(result.params, dtype=float)
        if len(all_params) != x.shape[1] + 1:
            raise DebiasSchemaError(
                "The statsmodels NB2 fit returned an unexpected parameter vector."
            )
        fixed_params = all_params[: x.shape[1]]
        alpha = float(all_params[-1])
        shape = float(1 / alpha) if np.isfinite(alpha) and alpha > 0 else np.nan
        # MASS::glm.nb reports the fixed-effect covariance conditional on its
        # estimated theta. Refit the identical NB2 mean model at the MLE alpha
        # solely to obtain that conditional covariance; retain the joint-MLE
        # coefficients, log likelihood, AIC, and dispersion from the discrete
        # fit above.
        covariance_fit = sm.GLM(
            y,
            x,
            family=sm.families.NegativeBinomial(alpha=alpha),
            offset=offset,
        ).fit()
        fixed_bse = np.asarray(covariance_fit.bse, dtype=float)
        dispersion = {
            "parameterization": "NB2",
            "alpha": alpha,
            "shape": shape,
        }
        implementation_backend = "statsmodels_discrete_negative_binomial_nb2"
        model_contract = {
            "likelihood": "negative_binomial_2",
            "link": "log",
            "conditional_mean": "exp(X beta + offset)",
            "conditional_variance": "mu + alpha * mu^2",
            "shape_relation": "shape = 1 / alpha",
            "dispersion": "estimated_by_maximum_likelihood",
            "estimation": "joint_maximum_likelihood_newton",
            "fixed_effect_covariance": "conditional_on_estimated_shape",
            "rank_deficiency": "r_ordered_qr_alias_drop",
        }
    else:  # guarded by the public dispatcher
        raise NotImplementedError(
            f"Unsupported frequentist model family: {model_family}"
        )
    if (
        not converged
        or not bool(np.isfinite(fixed_params).all())
        or not bool(np.isfinite(fixed_bse).all())
        or (model_family == "negbin" and not np.isfinite(shape))
    ):
        raise DebiasSchemaError(
            f"The statsmodels {model_family} fit did not converge to finite "
            "coefficients and dispersion parameters."
        )
    coefficient_terms = [
        "(Intercept)" if term == "Intercept" else term for term in x.columns
    ]
    coefficients = pd.DataFrame(
        {
            "term": coefficient_terms,
            "mean": fixed_params,
            "sd": fixed_bse,
        }
    )
    diagnostics: dict[str, Any] = {
        "backend": implementation_backend,
        "convergence": {"status": "available", "converged": converged},
        "aic": float(result.aic),
    }
    if dispersion is not None:
        diagnostics["dispersion"] = dispersion
    diagnostics["aliased_fixed_effects"] = aliased_columns
    return {
        "model": result,
        "coefficients": coefficients.reset_index(drop=True),
        "formula": display_formula or formula,
        "diagnostics": diagnostics,
        "implementation_backend": implementation_backend,
        "model_contract": model_contract,
        "fixed_params": fixed_params,
        "active_design_columns": active_columns,
        "terms": list(x.columns),
        "design_columns": list(x.columns),
        "design_info": design_info,
    }


def _r_style_independent_columns(
    matrix: np.ndarray, tolerance: float = 1e-7
) -> list[int]:
    """Select columns in formula order using R's QR alias tolerance."""
    selected: list[int] = []
    orthonormal_basis: list[np.ndarray] = []
    for index in range(matrix.shape[1]):
        column = np.asarray(matrix[:, index], dtype=float)
        original_norm = float(np.linalg.norm(column))
        if original_norm == 0:
            continue
        residual = column.copy()
        # Reorthogonalization prevents accumulated projection error from
        # retaining an aliased column in large OD designs.
        for _ in range(2):
            for basis_vector in orthonormal_basis:
                residual -= basis_vector * float(basis_vector @ residual)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > tolerance * original_norm:
            selected.append(index)
            orthonormal_basis.append(residual / residual_norm)
    return selected


def _fit_gpboost_random_intercept(
    *,
    fit_df: pd.DataFrame,
    y: np.ndarray,
    x: pd.DataFrame,
    design_info: Any,
    fixed_formula: str,
    display_formula: str,
    model_family: str,
    offset: np.ndarray | None,
    group_col: str,
    aliased_columns: list[str],
) -> dict[str, Any]:
    try:
        import gpboost as gpb
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Frequentist random-intercept fits require the pinned `gpboost` "
            "runtime. Install it with `pip install -e '.[grouped]'`."
        ) from exc

    matrix = x.to_numpy(dtype=float)
    if int(np.linalg.matrix_rank(matrix)) != matrix.shape[1]:
        raise DebiasSchemaError(
            "The fixed-effect design matrix is rank deficient. GPBoost does "
            "not apply lme4's column-dropping rule implicitly; supply a "
            "full-rank formula so the translated coefficient contract is "
            "unambiguous."
        )
    group_data = (
        fit_df[group_col].astype("string").astype(str).to_numpy().reshape(-1, 1)
    )
    likelihood = "poisson" if model_family == "poisson" else "negative_binomial"
    max_iterations = 1000
    relative_tolerance = 1e-8
    model = gpb.GPModel(
        group_data=group_data,
        likelihood=likelihood,
    )
    # GPBoost's implicit initialization is process-state dependent for some
    # small/boundary fits.  Pin every optimized parameter explicitly so that
    # repeating the same fit in one process starts from the same point.
    initial_coefficients = np.zeros(matrix.shape[1], dtype=float)
    if "Intercept" in x.columns:
        initial_coefficients[x.columns.get_loc("Intercept")] = float(
            np.log(max(float(np.mean(y)), np.finfo(float).tiny))
        )
    fit_params: dict[str, Any] = {
        "trace": False,
        "optimizer_cov": "lbfgs",
        "optimizer_coef": "lbfgs",
        "maxit": max_iterations,
        "delta_rel_conv": relative_tolerance,
        "init_cov_pars": np.ones(1, dtype=float),
        "init_coef": initial_coefficients,
    }
    if model_family == "negbin":
        fit_params["init_aux_pars"] = np.ones(1, dtype=float)
    try:
        model.fit(
            y=np.asarray(y, dtype=float),
            X=x,
            offset=offset,
            params=fit_params,
        )
        coef_with_se = np.asarray(
            model.get_coef(std_err=True, format_pandas=False), dtype=float
        )
        covariance_parameters = np.asarray(
            model.get_cov_pars(format_pandas=False), dtype=float
        ).reshape(-1)
        negative_log_likelihood = float(model.get_current_neg_log_likelihood())
        # GPBoost 1.6.8 exposes the optimizer iteration count only through
        # this stable internal wrapper around its public C API.
        optimizer_iterations = int(model._get_num_optim_iter())
    except Exception as exc:
        raise DebiasSchemaError(
            "The GPBoost single-random-intercept fit failed before producing "
            "finite likelihood, covariance, and coefficient diagnostics."
        ) from exc

    if coef_with_se.shape != (2, matrix.shape[1]):
        raise DebiasSchemaError(
            "GPBoost returned an unexpected fixed-effect coefficient summary."
        )
    fixed_params = coef_with_se[0]
    fixed_bse = coef_with_se[1]
    random_variance = (
        float(covariance_parameters[0]) if len(covariance_parameters) == 1 else np.nan
    )
    converged = bool(
        0 < optimizer_iterations < max_iterations
        and np.isfinite(negative_log_likelihood)
        and np.isfinite(fixed_params).all()
        and np.isfinite(fixed_bse).all()
        and np.isfinite(random_variance)
        and random_variance > 0
    )

    dispersion: dict[str, float | str] | None = None
    auxiliary_parameters: list[float] = []
    estimated_parameter_count = matrix.shape[1] + 1
    conditional_variance = "mu"
    if model_family == "negbin":
        try:
            # GPBoost 1.6.8 does not accept ``std_err`` in get_aux_pars,
            # unlike the current rendered API documentation.
            auxiliary = np.asarray(
                model.get_aux_pars(format_pandas=False), dtype=float
            ).reshape(-1)
        except Exception as exc:
            raise DebiasSchemaError(
                "GPBoost did not return the estimated NB2 shape parameter."
            ) from exc
        shape = float(auxiliary[0]) if len(auxiliary) == 1 else np.nan
        auxiliary_parameters = auxiliary.tolist()
        converged = converged and bool(np.isfinite(shape) and shape > 0)
        dispersion = {
            "parameterization": "NB2",
            "shape": shape,
            "alpha": float(1 / shape) if np.isfinite(shape) and shape > 0 else np.nan,
        }
        estimated_parameter_count += 1
        conditional_variance = "mu + mu^2 / shape"

    if not converged:
        raise DebiasSchemaError(
            "The GPBoost fit did not satisfy the fail-closed convergence "
            "contract: completion before maxit and finite positive covariance, "
            "dispersion, coefficient, and likelihood diagnostics."
        )

    coefficient_terms = [
        "(Intercept)" if term == "Intercept" else term for term in x.columns
    ]
    coefficients = pd.DataFrame(
        {
            "term": coefficient_terms,
            "mean": fixed_params,
            "sd": fixed_bse,
        }
    )
    runtime_version = str(getattr(gpb, "__version__", "unknown"))
    diagnostics: dict[str, Any] = {
        "backend": "gpboost_grouped_random_intercept",
        "runtime_version": runtime_version,
        "convergence": {
            "status": "available",
            "converged": True,
            "optimizer": "lbfgs",
            "iterations": optimizer_iterations,
            "max_iterations": max_iterations,
            "relative_tolerance": relative_tolerance,
            "initial_covariance_parameters": [1.0],
            "initial_fixed_coefficients": initial_coefficients.tolist(),
            "initial_auxiliary_parameters": ([1.0] if model_family == "negbin" else []),
        },
        "log_likelihood": -negative_log_likelihood,
        "aic": float(2 * negative_log_likelihood + 2 * estimated_parameter_count),
        "random_effect": {
            "group_column": group_col,
            "n_groups": int(pd.Series(group_data[:, 0]).nunique()),
            "variance": random_variance,
            "sd": float(np.sqrt(random_variance)),
        },
        "aliased_fixed_effects": aliased_columns,
    }
    if dispersion is not None:
        diagnostics["dispersion"] = dispersion
    model_contract = {
        "likelihood": (
            "poisson" if model_family == "poisson" else "negative_binomial_2"
        ),
        "link": "log",
        "conditional_mean": "exp(X beta + random_intercept + offset)",
        "conditional_variance": conditional_variance,
        "random_effects": "one_gaussian_grouped_random_intercept",
        "marginal_likelihood": "laplace_approximation",
        "estimation": "maximum_approximated_marginal_likelihood",
        "initialization": "explicit_deterministic_parameters",
        "counterfactual_prediction": (
            "fixed_effects_only_exp(X beta + offset); random effects excluded"
        ),
        "r_comparator": (
            "lme4::glmer" if model_family == "poisson" else "lme4::glmer.nb"
        ),
        "rank_deficiency": "r_ordered_qr_alias_drop",
    }
    model_snapshot = {
        "runtime_backend": "gpboost",
        "runtime_version": runtime_version,
        "likelihood": likelihood,
        "fixed_formula": fixed_formula,
        "group_column": group_col,
        "fixed_effect_terms": list(x.columns),
        "fixed_effect_coefficients": fixed_params.tolist(),
        "fixed_effect_standard_errors": fixed_bse.tolist(),
        "covariance_parameters": covariance_parameters.tolist(),
        "auxiliary_parameters": auxiliary_parameters,
        "negative_log_likelihood": negative_log_likelihood,
        "optimizer_iterations": optimizer_iterations,
        "optimizer_initialization": {
            "covariance_parameters": [1.0],
            "fixed_effect_coefficients": initial_coefficients.tolist(),
            "auxiliary_parameters": [1.0] if model_family == "negbin" else [],
        },
    }
    return {
        # A live GPModel owns ctypes pointers and cannot be deep-copied by
        # pandas when attrs propagate during ordinary column selection. Keep a
        # compact, pointer-free fitted-model snapshot instead.
        "model": model_snapshot,
        "coefficients": coefficients,
        "formula": display_formula,
        "fixed_formula": fixed_formula,
        "diagnostics": diagnostics,
        "implementation_backend": "gpboost_grouped_random_intercept",
        "model_contract": model_contract,
        "fixed_params": fixed_params,
        "active_design_columns": list(x.columns),
        "terms": list(x.columns),
        "design_columns": list(x.columns),
        "design_info": design_info,
    }


def _predict_linear(
    fit: dict[str, Any],
    data: pd.DataFrame,
    include_offset: bool,
) -> np.ndarray:
    x = _prediction_design_matrix(data, fit["design_info"], fit["formula"])
    x = x.loc[:, fit["active_design_columns"]]
    linear = x.to_numpy(dtype=float) @ np.asarray(fit["fixed_params"], dtype=float)
    if include_offset and "log_observation_probability" in data.columns:
        linear = linear + data["log_observation_probability"].to_numpy(dtype=float)
    return linear


def _fit_design_matrices(
    data: pd.DataFrame, formula: str
) -> tuple[np.ndarray, pd.DataFrame, Any]:
    try:
        from patsy import EvalEnvironment, PatsyError, dmatrices
    except ImportError as exc:  # pragma: no cover - statsmodels installs patsy
        raise ImportError(
            "The frequentist multilevel backend requires `patsy`, which is "
            "installed with debiaspy's mandatory statsmodels dependency. "
            "Reinstall debiaspy."
        ) from exc

    environment = EvalEnvironment.capture(0).with_outer_namespace(
        {"np": np, "log": np.log}
    )
    try:
        response, design = dmatrices(
            formula,
            data=data,
            return_type="dataframe",
            NA_action="raise",
            eval_env=environment,
        )
    except (PatsyError, ValueError, TypeError) as exc:
        raise DebiasSchemaError(
            f"Could not construct the frequentist model matrix for formula "
            f"`{formula}`: {exc}"
        ) from exc
    if len(response) != len(data) or len(design) != len(data):
        raise DebiasSchemaError(
            f"Formula changed the row count while building `{formula}`."
        )
    values = response.iloc[:, 0].to_numpy(dtype=float)
    matrix = design.reset_index(drop=True).astype(float)
    if not bool(np.isfinite(values).all()) or not bool(
        np.isfinite(matrix.to_numpy(dtype=float)).all()
    ):
        raise DebiasSchemaError(
            f"Formula `{formula}` produced non-finite response or design values."
        )
    return values, matrix, design.design_info


def _prediction_design_matrix(
    data: pd.DataFrame, design_info: Any, formula: str
) -> pd.DataFrame:
    try:
        from patsy import PatsyError, build_design_matrices
    except ImportError as exc:  # pragma: no cover - statsmodels installs patsy
        raise ImportError(
            "The frequentist multilevel backend requires `patsy`, which is "
            "installed with debiaspy's mandatory statsmodels dependency. "
            "Reinstall debiaspy."
        ) from exc

    try:
        matrix = build_design_matrices(
            [design_info],
            data=data,
            return_type="dataframe",
            NA_action="raise",
        )[0]
    except (PatsyError, ValueError, TypeError) as exc:
        raise DebiasSchemaError(
            f"Could not construct the frequentist prediction matrix for formula "
            f"`{formula}`: {exc}"
        ) from exc
    if len(matrix) != len(data):
        raise DebiasSchemaError(
            f"Formula changed the row count while predicting from `{formula}`."
        )
    out = matrix.reset_index(drop=True).astype(float)
    if not bool(np.isfinite(out.to_numpy(dtype=float)).all()):
        raise DebiasSchemaError(
            f"Formula `{formula}` produced non-finite prediction values."
        )
    return out


def _validate_frequentist_random_group(fit_df: pd.DataFrame, group_col: str) -> None:
    if group_col not in fit_df.columns:
        raise DebiasSchemaError(
            f"Random-intercept grouping column `{group_col}` is not available "
            "in prepared model data."
        )
    groups = fit_df[group_col]
    if bool(groups.isna().any()):
        raise DebiasSchemaError(
            f"Random-intercept grouping column `{group_col}` contains missing "
            "values after model-data filtering."
        )
    n_groups = int(groups.astype(str).nunique(dropna=True))
    if n_groups < 2:
        raise DebiasSchemaError(
            f"Random-intercept grouping column `{group_col}` requires at least "
            "2 distinct levels."
        )
    if group_col == "od_id":
        counts = groups.astype(str).value_counts()
        if not counts.empty and int(counts.max()) <= 1:
            warnings.warn(
                "OD random intercepts may be weakly identified when each OD "
                "pair appears once. Use with caution.",
                stacklevel=2,
            )


def _filter_formula_model_data(
    data: pd.DataFrame,
    terms: list[str],
    *,
    context: str,
) -> pd.DataFrame:
    variables = list(dict.fromkeys(["flow", *terms]))
    missing = [variable for variable in variables if variable not in data.columns]
    if missing:
        raise DebiasSchemaError(
            "`formula` references variable(s) not found in prepared model data: "
            + ", ".join(missing)
        )

    keep = np.ones(len(data), dtype=bool)
    for variable in variables:
        values = data[variable]
        if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            variable_keep = np.isfinite(numeric)
        else:
            variable_keep = values.notna().to_numpy(dtype=bool).copy()
            if pd.api.types.is_string_dtype(values) or values.dtype == object:
                nonempty = values.astype("string").str.len().fillna(0).to_numpy() > 0
                variable_keep &= nonempty
        keep &= variable_keep

    flow = pd.to_numeric(data["flow"], errors="coerce").to_numpy(dtype=float)
    keep &= np.isfinite(flow) & (flow >= 0)
    dropped = int((~keep).sum())
    if dropped:
        warnings.warn(
            f"{dropped} row(s) were excluded from {context} data because "
            "`formula` variables were missing, non-finite, or invalid.",
            stacklevel=2,
        )
    return data.loc[keep].copy()


def _stabilize_formula_factor_levels(
    data: pd.DataFrame, variables: list[str]
) -> pd.DataFrame:
    """Preserve full prediction-level factor support when fitting a subset."""
    out = data.copy()
    for variable in variables:
        if variable not in out.columns:
            continue
        values = out[variable]
        if not (
            pd.api.types.is_object_dtype(values)
            or pd.api.types.is_string_dtype(values)
            or isinstance(values.dtype, pd.CategoricalDtype)
        ):
            continue
        levels = sorted(
            {str(value) for value in values.dropna() if len(str(value)) > 0}
        )
        out[variable] = pd.Categorical(values.astype("string"), categories=levels)
    return out
