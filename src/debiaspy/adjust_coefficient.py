"""Regression-based calibration of observed flows against benchmark flows."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd


def adjust_coefficient(
    mpd_od_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_bench: str = "flow",
    model_family: Literal["ols", "poisson", "negbin", "zinb"] = "ols",
    level: Literal["od", "origin", "destination"] = "od",
    fit_intercept: bool = False,
    by_source: bool = False,
    keep_cols: str | Sequence[str] = (),
) -> pd.DataFrame:
    """Calibrate observed flows to a benchmark and apply a positive multiplier.

    Parameters
    ----------
    mpd_od_df, benchmark_od_df : pandas.DataFrame
        Observed and benchmark tables containing origin, destination and the
        selected flow columns. Calibration joins OD keys or their margins.
        Repeated OD keys expand the calibration join; output rows follow the
        observed input. Source-specific fitting requires mpd_source in both.
    flow_col_mpd, flow_col_bench : str
        Names of the observed and benchmark flow columns.
    model_family : {"ols", "poisson", "negbin", "zinb"}
        OLS fits a slope. Count models fit an intercept with log(observed flow)
        as an offset. Negative-binomial families use statsmodels.
    level : {"od", "origin", "destination"}
        Calibration uses OD pairs, origin totals or destination totals.
        Margin sums omit missing values. Usable calibration pairs have finite
        observed flow greater than zero and finite benchmark flow at least zero.
    fit_intercept : bool
        Estimate an intercept for OLS diagnostics. The fitted slope remains the
        applied multiplier. Ignored with a warning for count families.
    by_source : bool
        Estimate one coefficient for each overlapping mpd_source value.
        Observations from sources absent from the benchmark retain missing
        adjustments. A global fit pools rows by the selected calibration keys.
    keep_cols : str or sequence of str
        Additional observed columns to retain, once each, when present.

    Returns
    -------
    pandas.DataFrame
        origin, destination, optional mpd_source and selected columns, followed
        by flow, flow_adj and coef_factor. The input tables are preserved.
        attrs["coef"] stores the scalar coefficient or source-specific table;
        attrs["model"] stores n, beta, intercept, r_squared, family and level.
        Model n counts usable calibration pairs after joining and aggregation.
        Fewer than two usable calibration pairs yield missing coefficients.
        Non-positive or non-finite coefficients yield missing adjustments.

    Raises
    ------
    ValueError
        Required columns or argument values are invalid, sources do not overlap,
        or a count-model fit fails or produces invalid parameters.
    TypeError
        Inputs are not DataFrames or flow columns contain non-numeric values.

    Notes
    -----
    The OLS intercept is recorded in model diagnostics; adjusted flow is the
    original observed flow multiplied by the fitted slope. Zero-inflated NB
    uses the count-component coefficient as its multiplier.
    """
    model_family = _choice(
        model_family, ("ols", "poisson", "negbin", "zinb"), "model_family"
    )
    level = _choice(level, ("od", "origin", "destination"), "level")
    for name, flag in (("fit_intercept", fit_intercept), ("by_source", by_source)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"`{name}` must be a boolean.")
    for name, frame, flow_col in (
        ("mpd_od_df", mpd_od_df, flow_col_mpd),
        ("benchmark_od_df", benchmark_od_df, flow_col_bench),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"`{name}` must be a pandas DataFrame.")
        if not isinstance(flow_col, str):
            raise TypeError("Flow column names must be strings.")
        missing = [
            col for col in ("origin", "destination", flow_col) if col not in frame
        ]
        if missing:
            raise ValueError(f"`{name}` must contain: {', '.join(missing)}")
        if not frame.columns.is_unique:
            raise ValueError(f"`{name}` must have unique column names.")
        _numeric_flow(frame[flow_col], name)
    if mpd_od_df.empty:
        raise ValueError("`mpd_od_df` must contain at least one observation.")

    has_source = "mpd_source" in mpd_od_df and "mpd_source" in benchmark_od_df
    if by_source and not has_source:
        raise ValueError("`by_source = TRUE` requires `mpd_source` in both inputs.")
    if model_family != "ols" and fit_intercept:
        warnings.warn(
            "`fit_intercept` is ignored for count model families; "
            "using offset(log(x)) with intercept only.",
            UserWarning,
            stacklevel=2,
        )
    if isinstance(keep_cols, str):
        keep_cols = [keep_cols]
    elif not isinstance(keep_cols, Sequence):
        raise TypeError("`keep_cols` must be a column name or sequence of names.")
    if not all(isinstance(col, str) for col in keep_cols):
        raise TypeError("`keep_cols` must contain column names.")

    sources = [None]
    if by_source:
        sources = list(
            pd.unique(
                mpd_od_df.loc[
                    mpd_od_df.mpd_source.isin(benchmark_od_df.mpd_source), "mpd_source"
                ]
            )
        )
        if not sources:
            raise ValueError("No overlapping `mpd_source` values for by_source = TRUE.")
    rows = []
    for source in sources:
        mpd, benchmark = mpd_od_df, benchmark_od_df
        if by_source:
            # R equality filtering produces an empty subset for a missing source.
            mpd = mpd.loc[mpd.mpd_source.eq(source).fillna(False)]
            benchmark = benchmark.loc[benchmark.mpd_source.eq(source).fillna(False)]
        pairs = _calibration_pairs(mpd, benchmark, flow_col_mpd, flow_col_bench, level)
        rows.append(_fit_one(pairs, model_family, level, fit_intercept, source))
    model = pd.DataFrame(
        rows,
        columns=[
            "mpd_source",
            "n",
            "beta",
            "intercept",
            "r_squared",
            "family",
            "level",
        ],
    )
    out = mpd_od_df.copy()
    if by_source:
        factors = (
            out[["mpd_source"]]
            .merge(
                model[["mpd_source", "beta"]],
                on="mpd_source",
                how="left",
                sort=False,
                validate="many_to_one",
            )
            .beta.to_numpy(dtype=float)
        )
    else:
        factors = np.full(len(out), model.beta.iloc[0])
    factors = np.where(np.isfinite(factors) & (factors > 0), factors, np.nan)
    out["coef_factor"] = factors
    with np.errstate(invalid="ignore", over="ignore"):
        out["flow_adj"] = _numeric_flow(out[flow_col_mpd], "mpd_od_df") * factors
    columns = list(
        dict.fromkeys(
            [
                "origin",
                "destination",
                *(["mpd_source"] if has_source else []),
                *[col for col in keep_cols if col in out],
                flow_col_mpd,
                "flow_adj",
                "coef_factor",
            ]
        )
    )
    if flow_col_mpd != "flow" and "flow" in columns:
        raise ValueError("The selected output columns would duplicate `flow`.")
    result = (
        out.loc[:, columns]
        .rename(columns={flow_col_mpd: "flow"})
        .reset_index(drop=True)
    )
    result.attrs["coef"] = model.copy() if by_source else float(model.beta.iloc[0])
    result.attrs["model"] = model
    return result


def _choice(value: str, choices: tuple[str, ...], name: str) -> str:
    if isinstance(value, str):
        matches = [choice for choice in choices if choice.startswith(value)]
        if value in choices:
            return value
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"`{name}` must be one of: {', '.join(choices)}")


def _numeric_flow(values: pd.Series, name: str) -> np.ndarray:
    if values.dtype.kind not in "biuf" and not all(
        pd.isna(value) or isinstance(value, Real) for value in values
    ):
        raise TypeError(f"`{name}` flow values must be numeric.")
    return pd.to_numeric(values, errors="raise").to_numpy(dtype=float, na_value=np.nan)


def _calibration_pairs(
    mpd: pd.DataFrame,
    benchmark: pd.DataFrame,
    flow_col_mpd: str,
    flow_col_bench: str,
    level: str,
) -> pd.DataFrame:
    keys = ["origin", "destination"] if level == "od" else [level]
    left = mpd.loc[:, [*keys, flow_col_mpd]].copy()
    right = benchmark.loc[:, [*keys, flow_col_bench]].copy()
    left[flow_col_mpd] = _numeric_flow(left[flow_col_mpd], "mpd_od_df")
    right[flow_col_bench] = _numeric_flow(right[flow_col_bench], "benchmark_od_df")
    left = left.rename(columns={flow_col_mpd: "x"})
    right = right.rename(columns={flow_col_bench: "y"})
    if level != "od":
        left = (
            left.groupby(keys, dropna=False, sort=False, observed=True)
            .x.sum()
            .reset_index()
        )
        right = (
            right.groupby(keys, dropna=False, sort=False, observed=True)
            .y.sum()
            .reset_index()
        )
    left["_left_row"] = np.arange(len(left))
    right["_right_row"] = np.arange(len(right))
    pairs = left.merge(right, on=keys, how="inner", sort=False)
    if pairs._left_row.duplicated().any() and pairs._right_row.duplicated().any():
        warnings.warn(
            "Detected an unexpected many-to-many relationship "
            "between calibration rows.",
            UserWarning,
            stacklevel=3,
        )
    valid = np.isfinite(pairs.x) & (pairs.x > 0) & np.isfinite(pairs.y) & (pairs.y >= 0)
    return pairs.loc[valid, ["x", "y"]]


def _fit_one(
    pairs: pd.DataFrame,
    family: str,
    level: str,
    fit_intercept: bool,
    source: object,
) -> dict[str, object]:
    x, y = pairs.x.to_numpy(), pairs.y.to_numpy()
    beta = intercept = r_squared = np.nan
    if len(x) >= 2:
        if family == "ols":
            if fit_intercept:
                beta, intercept = _ols_with_intercept(x, y)
                fitted = (
                    np.full_like(y, intercept)
                    if not np.isfinite(beta)
                    else intercept + beta * x
                )
                r_squared = _lm_r_squared(y, fitted)
            else:
                beta, intercept = float(np.sum(x * y) / np.sum(x**2)), 0.0
                r_squared = _cor_squared(y, beta * x)
        else:
            fit = {
                "poisson": _fit_poisson_offset,
                "negbin": _fit_negbin_offset,
                "zinb": _fit_zinb_offset,
            }[family]
            intercept, beta = fit(x, y)
            r_squared = _cor_squared(y, beta * x)
    return {
        "mpd_source": source,
        "n": len(x),
        "beta": beta,
        "intercept": intercept,
        "r_squared": r_squared,
        "family": family,
        "level": level,
    }


def _ols_with_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # stats::lm uses a 1e-7 QR tolerance and leaves an aliased slope as NA.
    # Comparing the residualized x column with its original norm reproduces
    # that behavior for the two-column (intercept, x) model used here.
    x_norm = float(np.linalg.norm(x))
    centered_norm = float(np.linalg.norm(x - np.mean(x)))
    if centered_norm <= 1e-7 * x_norm:
        return np.nan, float(np.mean(y))

    design = np.column_stack([np.ones_like(x), x])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(beta), float(alpha)


def _fit_poisson_offset(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit R's intercept-only Poisson GLM with ``offset(log(x))``.

    The closed-form MLE is undefined when every response is zero.  R's IRLS
    implementation intentionally returns the finite boundary iterate at its
    convergence tolerance, so the iteration is reproduced instead of replacing
    that case with beta=0.
    """
    offset = np.log(x)
    mu = y + 0.1  # poisson()$initialize in R
    eta = np.log(mu)
    deviance = _poisson_deviance(y, mu)
    alpha = np.nan
    converged = False

    for _ in range(25):  # glm.control()$maxit
        working_response = (eta - offset) + (y - mu) / mu
        alpha = float(np.sum(mu * working_response) / np.sum(mu))
        eta = alpha + offset
        mu = np.exp(eta)
        next_deviance = _poisson_deviance(y, mu)
        if abs(next_deviance - deviance) / (0.1 + abs(next_deviance)) < 1e-8:
            converged = True
            break
        deviance = next_deviance

    if not converged:
        warnings.warn("glm.fit: algorithm did not converge", stacklevel=4)

    # R's dpois-based AIC evaluation traverses non-integer observations twice.
    fractional = y[np.abs(y - np.rint(y)) > 1e-7 * np.maximum(1.0, np.abs(y))]
    for _ in range(2):
        for value in fractional:
            warnings.warn(f"non-integer x = {value:.6f}", stacklevel=4)

    with np.errstate(over="ignore", invalid="ignore"):
        beta = float(np.exp(alpha))
    return alpha, beta


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    contributions = np.array(mu, copy=True)
    positive = y > 0
    contributions[positive] = y[positive] * np.log(y[positive] / mu[positive]) - (
        y[positive] - mu[positive]
    )
    return float(2 * np.sum(contributions))


def _fit_negbin_offset(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        import statsmodels.api as sm

        if not np.any(y > 0):
            raise ValueError("missing value where TRUE/FALSE needed")

        exog = np.ones((len(x), 1), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.NegativeBinomial(y, exog, offset=np.log(x)).fit(
                method="bfgs",
                maxiter=1000,
                gtol=1e-8,
                disp=0,
            )
        params = np.asarray(fit.params, dtype=float)
        converged = bool(fit.mle_retvals.get("converged", False))
        if (
            not converged
            or len(params) < 2
            or not np.isfinite(params).all()
            or params[-1] <= 0
        ):
            raise ValueError("statsmodels negative-binomial optimizer did not converge")
        alpha = float(fit.params[0])
        beta = math.exp(alpha)
        return alpha, beta
    except Exception as exc:
        raise ValueError(
            "Negative-binomial coefficient fit failed: " + str(exc)
        ) from exc


def _fit_zinb_offset(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

        if np.any(y != np.floor(y)):
            raise ValueError("invalid dependent variable, non-integer values")
        if not np.any(y > 0):
            raise ValueError("0 (non-NA) cases")
        if not np.any(y == 0):
            raise ValueError("invalid dependent variable, minimum count is not zero")

        exog = np.ones((len(x), 1), dtype=float)
        from scipy.optimize import minimize

        model = ZeroInflatedNegativeBinomialP(y, exog, exog_infl=exog, offset=np.log(x))

        def objective(parameters):
            # Optimise log-dispersion so NB2 dispersion stays positive.
            # Scale by n, as in statsmodels' likelihood optimisers.
            natural = np.array(parameters, copy=True)
            natural[-1] = np.exp(natural[-1])
            gradient = model.score(natural)
            gradient[-1] *= natural[-1]
            return -model.loglike(natural) / len(y), -gradient / len(y)

        initial = [0.0, np.log(np.sum(y) / np.sum(x)), 0.0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = minimize(
                objective,
                initial,
                jac=True,
                method="BFGS",
                options={"maxiter": 10000, "gtol": 1e-5},
            )
        params = np.array(fit.x, dtype=float)
        params[-1] = np.exp(params[-1])
        converged = bool(fit.success)
        if (
            not converged
            or len(params) < 3
            or not np.isfinite(params).all()
            or params[-1] <= 0
        ):
            raise ValueError("statsmodels ZINB optimizer did not converge")
        alpha = float(params[1])
        beta = math.exp(alpha)
        return alpha, beta
    except Exception as exc:
        raise ValueError(
            "Zero-inflated negative-binomial coefficient fit failed: " + str(exc)
        ) from exc


def _lm_r_squared(y: np.ndarray, yhat: np.ndarray) -> float:
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total == 0:
        return 0.0
    residual = float(np.sum((y - yhat) ** 2))
    return 1 - residual / total


def _cor_squared(y: np.ndarray, yhat: np.ndarray) -> float:
    # np.mean can introduce a one-ulp spread when subtracting an array whose
    # entries are bitwise identical; stats::cor treats that vector as constant.
    if (
        len(y) < 2
        or np.ptp(y) == 0
        or np.ptp(yhat) == 0
        or np.std(y) == 0
        or np.std(yhat) == 0
    ):
        return np.nan
    return float(np.corrcoef(y, yhat)[0, 1] ** 2)
