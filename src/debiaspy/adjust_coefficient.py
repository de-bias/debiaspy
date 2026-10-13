from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
import pandas as pd

from debiaspy._adjustment_common import (
    existing_columns,
    finite_non_negative_mask,
    match_choice,
    repeated_observation_join_keys,
    require_no_reserved_columns,
    require_scalar_bool,
    require_unique_observation_rows,
    validated_non_negative_numeric,
)
from debiaspy._exceptions import DebiasSchemaError
from debiaspy._schemas import require_columns


def adjust_coefficient(
    mpd_od_df: pd.DataFrame,
    benchmark_od_df: pd.DataFrame,
    flow_col_mpd: str = "flow",
    flow_col_bench: str = "flow",
    model_family: Literal["ols", "poisson", "negbin", "zinb"] = "ols",
    level: Literal["od", "origin", "destination"] = "od",
    fit_intercept: bool = False,
    by_source: bool = False,
    keep_cols: str | list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """Estimate and apply a benchmark-to-MPD flow calibration coefficient.

    Both inputs require ``origin``, ``destination``, and their selected flow
    columns; source-stratified fitting also requires ``mpd_source`` in both.
    ``model_family`` selects OLS, Poisson, negative-binomial, or zero-inflated
    negative-binomial fitting, while ``level`` selects OD or marginal fitting.

    Returns the selected MPD keys and columns with canonical ``flow``,
    ``flow_adj``, and ``coef_factor`` fields. The fitted coefficient and model
    table are stored in ``attrs["coef"]`` and ``attrs["model"]``. Invalid
    schemas and non-estimable count models fail closed. Parity covers the
    declared deterministic coefficient fixtures and numeric tolerances.
    """
    model_family = match_choice(
        model_family, ("ols", "poisson", "negbin", "zinb"), "model_family"
    )
    level = match_choice(level, ("od", "origin", "destination"), "level")
    fit_intercept = require_scalar_bool(fit_intercept, "fit_intercept")
    by_source = require_scalar_bool(by_source, "by_source")

    require_columns(mpd_od_df, ("origin", "destination", flow_col_mpd), "mpd_od_df")
    require_columns(
        benchmark_od_df,
        ("origin", "destination", flow_col_bench),
        "benchmark_od_df",
    )
    require_no_reserved_columns(
        mpd_od_df,
        ("beta", "intercept", "coef_factor", "flow_adj"),
        "mpd_od_df",
    )
    mpd_od_df = mpd_od_df.copy()
    benchmark_od_df = benchmark_od_df.copy()
    mpd_od_df[flow_col_mpd] = validated_non_negative_numeric(
        mpd_od_df[flow_col_mpd], f"mpd_od_df${flow_col_mpd}"
    ).to_numpy()
    benchmark_od_df[flow_col_bench] = validated_non_negative_numeric(
        benchmark_od_df[flow_col_bench], f"benchmark_od_df${flow_col_bench}"
    ).to_numpy()
    requested_keep = existing_columns(mpd_od_df, keep_cols)
    if flow_col_mpd != "flow" and "flow" in requested_keep:
        raise DebiasSchemaError(
            "`flow_col_mpd` cannot be renamed to `flow` because `keep_cols` "
            "also selects the existing `flow` column."
        )

    has_source = (
        "mpd_source" in mpd_od_df.columns and "mpd_source" in benchmark_od_df.columns
    )
    if by_source and not has_source:
        raise DebiasSchemaError(
            "`by_source = TRUE` requires `mpd_source` in both inputs."
        )
    if model_family != "ols" and fit_intercept:
        warnings.warn(
            "`fit_intercept` is ignored for count model families; "
            "using offset(log(x)) with intercept only.",
            stacklevel=2,
        )

    if by_source:
        sources = [
            source
            for source in pd.unique(mpd_od_df["mpd_source"])
            if source in set(pd.unique(benchmark_od_df["mpd_source"]))
        ]
        if not sources:
            raise DebiasSchemaError(
                "No overlapping `mpd_source` values for by_source = TRUE."
            )
        fits = [
            _fit_one(
                mpd_od_df.loc[mpd_od_df["mpd_source"] == source],
                benchmark_od_df.loc[benchmark_od_df["mpd_source"] == source],
                flow_col_mpd=flow_col_mpd,
                flow_col_bench=flow_col_bench,
                model_family=model_family,
                level=level,
                fit_intercept=fit_intercept,
                source_label=source,
            )
            for source in sources
        ]
        coef_tbl = pd.concat([fit["summary"] for fit in fits], ignore_index=True)
        coef_attr: float | pd.DataFrame = coef_tbl.copy()
    else:
        fit = _fit_one(
            mpd_od_df,
            benchmark_od_df,
            flow_col_mpd=flow_col_mpd,
            flow_col_bench=flow_col_bench,
            model_family=model_family,
            level=level,
            fit_intercept=fit_intercept,
            source_label=None,
        )
        coef_tbl = fit["summary"]
        coef_attr = float(coef_tbl["beta"].iloc[0])

    out = mpd_od_df.copy()
    keep = existing_columns(out, requested_keep)

    if by_source:
        coefficient_columns = ["mpd_source", "beta"]
        if model_family == "ols" and fit_intercept:
            coefficient_columns.append("intercept")
        out = out.merge(
            coef_tbl.loc[:, coefficient_columns],
            how="left",
            on="mpd_source",
            sort=False,
        )
        beta_values = pd.to_numeric(out["beta"], errors="coerce")
    else:
        beta_values = pd.Series(float(coef_tbl["beta"].iloc[0]), index=out.index)
    flow = pd.to_numeric(out[flow_col_mpd], errors="coerce")
    if model_family == "ols" and fit_intercept:
        intercept_values = (
            pd.to_numeric(out["intercept"], errors="coerce")
            if by_source
            else pd.Series(float(coef_tbl["intercept"].iloc[0]), index=out.index)
        )
        prediction = intercept_values + beta_values.fillna(0.0) * flow
        positive_flow = np.isfinite(flow) & (flow > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = prediction / flow
        valid = positive_flow & np.isfinite(prediction) & (prediction > 0)
        out["coef_factor"] = factor.where(valid, np.nan)
        out["flow_adj"] = prediction.where(valid, np.nan)
    else:
        invalid = ~np.isfinite(beta_values) | (beta_values <= 0)
        out["coef_factor"] = beta_values.mask(invalid, np.nan)
        out["flow_adj"] = np.where(
            np.isfinite(out["coef_factor"].to_numpy(dtype=float)),
            flow * out["coef_factor"],
            np.nan,
        )
    out = out.drop(columns=["beta", "intercept"], errors="ignore")

    columns = existing_columns(
        out,
        [
            "origin",
            "destination",
            *(["mpd_source"] if has_source else []),
            *keep,
            flow_col_mpd,
            "flow_adj",
            "coef_factor",
        ],
    )
    result = out.loc[:, columns].rename(columns={flow_col_mpd: "flow"})
    result = result.reset_index(drop=True)
    result.attrs["coef"] = coef_attr
    result.attrs["model"] = coef_tbl.reset_index(drop=True)
    return result


def _fit_one(
    mpd: pd.DataFrame,
    benchmark: pd.DataFrame,
    flow_col_mpd: str,
    flow_col_bench: str,
    model_family: str,
    level: str,
    fit_intercept: bool,
    source_label: object | None,
) -> dict[str, pd.DataFrame]:
    xy = _build_xy(mpd, benchmark, flow_col_mpd, flow_col_bench, level)
    x = pd.to_numeric(xy["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(xy["y"], errors="coerce").to_numpy(dtype=float)
    keep = finite_non_negative_mask(y) & np.isfinite(x) & (x > 0)
    x = x[keep]
    y = y[keep]

    if len(x) < 2:
        beta = np.nan
        alpha = np.nan
        r_squared = np.nan
        n = len(x)
    elif model_family == "ols":
        n = len(x)
        if fit_intercept:
            beta, alpha = _ols_with_intercept(x, y)
            # An aliased slope is NA in stats::coef(), but stats::fitted()
            # still contains the estimable intercept-only predictions.
            yhat = (
                np.full_like(y, alpha, dtype=float)
                if not np.isfinite(beta)
                else alpha + beta * x
            )
            r_squared = _lm_r_squared(y, yhat)
        else:
            beta = float(np.sum(x * y) / np.sum(x**2))
            alpha = 0.0
            yhat = beta * x
            r_squared = _cor_squared(y, yhat)
    elif model_family == "poisson":
        n = len(x)
        alpha, beta = _fit_poisson_offset(x, y)
        r_squared = _cor_squared(y, beta * x)
    elif model_family == "negbin":
        n = len(x)
        alpha, beta = _fit_negbin_offset(x, y)
        r_squared = _cor_squared(y, beta * x)
    elif model_family == "zinb":
        n = len(x)
        alpha, beta = _fit_zinb_offset(x, y)
        r_squared = _cor_squared(y, beta * x)
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")

    summary = pd.DataFrame(
        {
            "mpd_source": [source_label],
            "n": [n],
            "beta": [beta],
            "intercept": [alpha],
            "r_squared": [r_squared],
            "family": [model_family],
            "level": [level],
        }
    )
    return {"summary": summary}


def _build_xy(
    mpd: pd.DataFrame,
    benchmark: pd.DataFrame,
    flow_col_mpd: str,
    flow_col_bench: str,
    level: str,
) -> pd.DataFrame:
    join_keys = repeated_observation_join_keys(mpd, benchmark)
    require_unique_observation_rows(
        mpd,
        keys=join_keys,
        frame_name="mpd_od_df",
        counterpart=benchmark,
        counterpart_name="benchmark_od_df",
    )
    require_unique_observation_rows(
        benchmark,
        keys=join_keys,
        frame_name="benchmark_od_df",
        counterpart=mpd,
        counterpart_name="mpd_od_df",
    )
    repeated_keys = [key for key in join_keys if key not in {"origin", "destination"}]
    if level == "od":
        left = mpd.loc[:, [*join_keys, flow_col_mpd]].rename(
            columns={flow_col_mpd: "x"}
        )
        right = benchmark.loc[:, [*join_keys, flow_col_bench]].rename(
            columns={flow_col_bench: "y"}
        )
        return left.merge(
            right,
            how="inner",
            on=join_keys,
            sort=False,
            validate="one_to_one",
        )

    if level == "origin":
        aggregate_keys = ["origin", *repeated_keys]
        left = (
            mpd.groupby(aggregate_keys, sort=True, dropna=False)[flow_col_mpd]
            .sum()
            .reset_index(name="x")
        )
        right = (
            benchmark.groupby(aggregate_keys, sort=True, dropna=False)[flow_col_bench]
            .sum()
            .reset_index(name="y")
        )
        return left.merge(
            right,
            how="inner",
            on=aggregate_keys,
            sort=False,
            validate="one_to_one",
        )

    aggregate_keys = ["destination", *repeated_keys]
    left = (
        mpd.groupby(aggregate_keys, sort=True, dropna=False)[flow_col_mpd]
        .sum()
        .reset_index(name="x")
    )
    right = (
        benchmark.groupby(aggregate_keys, sort=True, dropna=False)[flow_col_bench]
        .sum()
        .reset_index(name="y")
    )
    return left.merge(
        right,
        how="inner",
        on=aggregate_keys,
        sort=False,
        validate="one_to_one",
    )


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
        warnings.warn("glm.fit: algorithm did not converge", stacklevel=3)

    # R's dpois-based AIC evaluation traverses non-integer observations twice.
    fractional = y[np.abs(y - np.rint(y)) > 1e-7 * np.maximum(1.0, np.abs(y))]
    for _ in range(2):
        for value in fractional:
            warnings.warn(f"non-integer x = {value:.6f}", stacklevel=3)

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
        raise DebiasSchemaError(
            "Negative-binomial coefficient fit failed. debiaspy refuses to "
            "downgrade this branch to Poisson because that would no longer "
            "match debiasR's model contract. Original error: " + str(exc)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ZeroInflatedNegativeBinomialP(
                y,
                exog,
                exog_infl=np.ones((len(x), 1), dtype=float),
                offset=np.log(x),
            ).fit(disp=0, maxiter=200)
        params = np.asarray(fit.params, dtype=float)
        converged = bool(fit.mle_retvals.get("converged", False))
        if (
            not converged
            or len(params) < 3
            or not np.isfinite(params).all()
            or params[-1] <= 0
        ):
            raise ValueError("statsmodels ZINB optimizer did not converge")
        names = list(fit.model.exog_names)
        alpha_idx = names.index("const") if "const" in names else 1
        alpha = float(params[alpha_idx])
        beta = math.exp(alpha)
        return alpha, beta
    except Exception as exc:
        raise DebiasSchemaError(
            "Zero-inflated negative-binomial coefficient fit failed. debiaspy "
            "refuses to downgrade this branch because that would no longer "
            "match debiasR's model contract. Original error: " + str(exc)
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
