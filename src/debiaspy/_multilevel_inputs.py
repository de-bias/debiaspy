"""Shared multilevel input preparation and formula resolution."""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from debiaspy._adjustment_common import (
    collapse_exact_key_duplicates,
    validated_non_negative_numeric,
)
from debiaspy._exceptions import DebiasSchemaError
from debiaspy._schemas import require_columns


def _validate_observation_contract(
    target_scale: str, observation_model: str, model_engine: str
) -> None:
    if target_scale == "true_flow" and observation_model not in {
        "coverage_offset",
        "latent_two_level",
    }:
        raise DebiasSchemaError(
            "`target_scale = 'true_flow'` requires `observation_model = "
            "'coverage_offset'` or `observation_model = 'latent_two_level'`."
        )
    if (
        observation_model in {"coverage_offset", "latent_two_level"}
        and target_scale != "true_flow"
    ):
        raise DebiasSchemaError(
            f"`observation_model = '{observation_model}'` is supported only with "
            "`target_scale = 'true_flow'`."
        )
    if observation_model == "latent_two_level" and model_engine != "bayesian":
        raise DebiasSchemaError(
            "`observation_model = 'latent_two_level'` requires "
            "`model_engine = 'bayesian'`."
        )


def _resolve_formula_info(
    formula: str | None,
    mobility_formula: str | None,
    bias_formula: str | None,
    *,
    allow_extended: bool = False,
) -> dict[str, Any]:
    split_supplied = mobility_formula is not None or bias_formula is not None
    combined_supplied = formula is not None
    if split_supplied and combined_supplied:
        raise DebiasSchemaError(
            "Use either `formula` or the split "
            "`mobility_formula` + `bias_formula` interface, not both."
        )
    if split_supplied and (mobility_formula is None or bias_formula is None):
        raise DebiasSchemaError(
            "Supply both `mobility_formula` and `bias_formula` when using the "
            "split formula interface."
        )
    source = "formula" if formula is not None else "default"
    if split_supplied:
        mobility_rhs = _one_sided_rhs(str(mobility_formula), "mobility_formula")
        bias_rhs = _one_sided_rhs(str(bias_formula), "bias_formula")
        if not allow_extended:
            _validate_supported_formula_rhs(
                mobility_rhs, "mobility_formula", allow_random_intercept=True
            )
            _validate_supported_formula_rhs(
                bias_rhs, "bias_formula", allow_random_intercept=True
            )
        mobility_terms = _formula_fixed_variables(mobility_rhs)
        bias_terms = _formula_fixed_variables(bias_rhs)
        combined = f"flow ~ {mobility_rhs} + {bias_rhs}"
        return {
            "formula": combined,
            "interface": "split",
            "source": "split_formula",
            "mobility_formula": f"~ {mobility_rhs}",
            "bias_formula": f"~ {bias_rhs}",
            "mobility_variables": mobility_terms,
            "bias_variables": bias_terms,
        }
    if formula is None:
        return {
            "formula": None,
            "interface": "default",
            "source": source,
            "mobility_formula": None,
            "bias_formula": None,
            "mobility_variables": [],
            "bias_variables": [],
        }
    formula_text = str(formula)
    if "~" not in formula_text:
        raise DebiasSchemaError("`formula` must include a response, usually `flow`.")
    response = formula_text.split("~", 1)[0].strip()
    if response != "flow":
        raise DebiasSchemaError(
            "`formula` response must be `flow`; `flow_col` is standardised to "
            "`flow` before fitting."
        )
    if not allow_extended:
        _parse_rhs_terms(formula_text, allow_random_intercept=True)
    return {
        "formula": formula_text,
        "interface": "combined",
        "source": source,
        "mobility_formula": None,
        "bias_formula": None,
        "mobility_variables": [],
        "bias_variables": ["bias_e_origin"] if "bias_e_origin" in formula_text else [],
    }


def _resolve_target_formula_info(
    formula_info: dict[str, Any],
    *,
    target_scale: str,
    observation_model: str,
) -> dict[str, Any]:
    """Apply debiasR's non-latent true-flow split-formula semantics."""
    if (
        target_scale != "true_flow"
        or observation_model == "latent_two_level"
        or formula_info.get("interface") != "split"
    ):
        return formula_info

    mobility_formula = formula_info.get("mobility_formula")
    if mobility_formula is None:  # pragma: no cover - guarded by formula resolution
        raise DebiasSchemaError(
            "The split formula interface requires `mobility_formula`."
        )
    mobility_rhs = _one_sided_rhs(str(mobility_formula), "mobility_formula")
    resolved = formula_info.copy()
    resolved["formula"] = f"flow ~ {mobility_rhs}"
    resolved["interface"] = "split_true_flow"
    resolved["bias_variables"] = []
    return resolved


def _resolve_scenario(
    mpd_od_df: pd.DataFrame,
    source_col: str | None,
    time_col: str | None,
    scenario: str,
    repeated_observation: str,
) -> dict[str, Any]:
    source_col = _resolve_optional_col(
        mpd_od_df, source_col, "mpd_source", "source_col"
    )
    time_col = _resolve_optional_col(mpd_od_df, time_col, "mpd_time", "time_col")
    source_values = _distinct_values(mpd_od_df, source_col)
    time_values = _distinct_values(mpd_od_df, time_col)
    has_source_variation = len(source_values) > 1
    has_time_variation = len(time_values) > 1
    inferred = (
        "s4"
        if has_source_variation and has_time_variation
        else "s2"
        if has_time_variation
        else "s3"
        if has_source_variation
        else "s1"
    )
    scenario_resolved = inferred if scenario == "auto" else scenario
    _validate_scenario(
        scenario_resolved,
        source_col,
        time_col,
        has_source_variation,
        has_time_variation,
    )
    repeated_resolved = (
        {
            "s1": "none",
            "s2": "time",
            "s3": "source",
            "s4": "source_time",
        }[scenario_resolved]
        if repeated_observation == "auto"
        else repeated_observation
    )
    _validate_repeated(
        repeated_resolved,
        source_col,
        time_col,
        has_source_variation,
        has_time_variation,
    )
    return {
        "scenario": scenario_resolved,
        "source_col": source_col,
        "time_col": time_col,
        "repeated_observation": repeated_resolved,
        "n_sources": len(source_values),
        "n_time_periods": len(time_values),
        "scenario_cols": [
            col
            for col, enabled in [("mpd_source", source_col), ("mpd_time", time_col)]
            if enabled
        ],
        "audit_cols": [col for col in [source_col, time_col] if col is not None],
    }


def _prepare_multilevel_data(
    mpd_od_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    covariates_df: pd.DataFrame | None,
    distance_df: pd.DataFrame | None,
    flow_col: str,
    income_col: str | None,
    pop_col: str,
    distance_col: str,
    scenario_info: dict[str, Any],
) -> dict[str, Any]:
    require_columns(mpd_od_df, ("origin", "destination", flow_col), "mpd_od_df")
    require_columns(coverage_df, ("origin", "population", "user_count"), "coverage_df")
    reserved_input_columns = {
        "row_id",
        "flow_adj",
        "flow_mpd_pred",
        "flow_true_pred",
        "observation_probability",
        "log_observation_probability",
        "model_fit_status",
        "prediction_scope",
    }
    collisions = sorted(reserved_input_columns & set(mpd_od_df.columns))
    if collisions:
        raise DebiasSchemaError(
            "`mpd_od_df` contains reserved multilevel output column(s): "
            + ", ".join(collisions)
        )
    coverage_df = coverage_df.copy()
    coverage_df["population"] = validated_non_negative_numeric(
        coverage_df["population"], "coverage_df$population"
    ).to_numpy()
    if bool((coverage_df["population"] <= 0).any()):
        raise DebiasSchemaError(
            "`coverage_df$population` must contain finite positive numeric values."
        )
    coverage_df["user_count"] = validated_non_negative_numeric(
        coverage_df["user_count"], "coverage_df$user_count"
    ).to_numpy()
    if covariates_df is None:
        areas = sorted(
            set(map(str, mpd_od_df["origin"]))
            | set(map(str, mpd_od_df["destination"]))
            | set(map(str, coverage_df["origin"]))
        )
        covariates_df = pd.DataFrame({"area": areas})
    require_columns(covariates_df, ("area",), "covariates_df")
    default_covariate_col = _resolve_default_covariate_col(
        covariates_df, income_col, pop_col
    )

    base_df = mpd_od_df.copy()
    base_df["row_id"] = np.arange(1, len(base_df) + 1)
    base_df["origin"] = base_df["origin"].astype(str)
    base_df["destination"] = base_df["destination"].astype(str)
    base_df["flow"] = validated_non_negative_numeric(
        base_df[flow_col], "mpd_od_df$flow"
    ).to_numpy()
    base_df["od_id"] = base_df["origin"] + "___" + base_df["destination"]
    source_col = scenario_info["source_col"]
    time_col = scenario_info["time_col"]
    if source_col is not None:
        base_df["mpd_source"] = mpd_od_df[source_col].astype("string")
    if time_col is not None:
        base_df["mpd_time"] = mpd_od_df[time_col].astype("string")
    if "mpd_source" in base_df.columns and "mpd_time" in base_df.columns:
        base_df["mpd_source_time"] = base_df["mpd_source"] + "___" + base_df["mpd_time"]
    base_df["mpd_observed"] = (
        base_df["mpd_observed"].isin([True])
        if "mpd_observed" in base_df.columns
        else True
    )
    base_df["mpd_zero_filled"] = (
        base_df["mpd_zero_filled"].isin([True])
        if "mpd_zero_filled" in base_df.columns
        else False
    )
    if "mpd_row_status" not in base_df.columns:
        base_df["mpd_row_status"] = np.where(
            base_df["mpd_observed"], "observed", "zero_filled"
        )

    distance_source = "input"
    if distance_df is not None:
        require_columns(
            distance_df, ("origin", "destination", distance_col), "distance_df"
        )
        dist_tbl = distance_df.loc[:, ["origin", "destination", distance_col]].rename(
            columns={distance_col: "distance_km"}
        )
        dist_tbl["origin"] = dist_tbl["origin"].astype(str)
        dist_tbl["destination"] = dist_tbl["destination"].astype(str)
        dist_tbl["distance_km"] = validated_non_negative_numeric(
            dist_tbl["distance_km"], f"distance_df${distance_col}"
        ).to_numpy()
        dist_tbl = collapse_exact_key_duplicates(
            dist_tbl,
            keys=("origin", "destination"),
            value_columns=("distance_km",),
            frame_name="distance_df",
        )
        base_df = base_df.merge(
            dist_tbl,
            how="left",
            on=["origin", "destination"],
            validate="many_to_one",
        )
    elif distance_col in base_df.columns:
        base_df["distance_km"] = validated_non_negative_numeric(
            base_df[distance_col], f"mpd_od_df${distance_col}"
        ).to_numpy()
    else:
        distance_source = "synthetic"
        warnings.warn(
            "No real distance input provided; using synthetic index-based distance. "
            "Pass `distance_df` to use real OD distances.",
            stacklevel=2,
        )
        areas = sorted(set(base_df["origin"]) | set(base_df["destination"]))
        lookup = {area: idx + 1 for idx, area in enumerate(areas)}
        base_df["distance_km"] = [
            abs(lookup[o] - lookup[d]) + 1
            for o, d in zip(base_df["origin"], base_df["destination"], strict=True)
        ]
    base_df["distance_km"] = pd.to_numeric(base_df["distance_km"], errors="coerce")
    base_df["log_distance"] = np.log(
        np.maximum(base_df["distance_km"], np.finfo(float).eps)
    )
    base_df["log_dist_synth"] = base_df["log_distance"]

    cov_origin = coverage_df.loc[:, ["origin", "population", "user_count"]].copy()
    cov_origin["origin"] = cov_origin["origin"].astype(str)
    cov_origin["population"] = pd.to_numeric(cov_origin["population"], errors="coerce")
    cov_origin["user_count"] = pd.to_numeric(cov_origin["user_count"], errors="coerce")
    for source in ("mpd_source", "mpd_time"):
        ref_col = (
            scenario_info["source_col"]
            if source == "mpd_source"
            else scenario_info["time_col"]
        )
        if ref_col is not None and ref_col in coverage_df.columns:
            cov_origin[source] = coverage_df[ref_col].astype(str)
        elif source in base_df.columns and source in coverage_df.columns:
            cov_origin[source] = coverage_df[source].astype(str)
    join_keys = [
        key
        for key in ["origin", "mpd_source", "mpd_time"]
        if key in cov_origin and key in base_df
    ]
    cov_origin = collapse_exact_key_duplicates(
        cov_origin,
        keys=join_keys,
        value_columns=("population", "user_count"),
        frame_name="coverage_df",
    )
    base_df = base_df.merge(
        cov_origin,
        how="left",
        on=join_keys,
        validate="many_to_one",
    )
    cov_destination = cov_origin.rename(
        columns={
            "origin": "destination",
            "population": "population_d_coverage",
            "user_count": "user_count_d_coverage",
        }
    )
    destination_join_keys = [
        key
        for key in ["destination", "mpd_source", "mpd_time"]
        if key in cov_destination and key in base_df
    ]
    base_df = base_df.merge(
        cov_destination,
        how="left",
        on=destination_join_keys,
        validate="many_to_one",
    )

    area_covariates = covariates_df.copy()
    area_covariates["area"] = area_covariates["area"].astype(str)
    pop_area = coverage_df.loc[:, ["origin", "population"]].rename(
        columns={"origin": "area", "population": ".coverage_population"}
    )
    pop_area["area"] = pop_area["area"].astype(str)
    pop_area[".coverage_population"] = pd.to_numeric(
        pop_area[".coverage_population"], errors="coerce"
    )
    # This table is only the area-level population fallback for covariates.
    # debiasR takes the first physical area row even when source/time-specific
    # populations vary, while the coverage-rate joins above remain fully keyed.
    pop_area = pop_area.drop_duplicates(subset="area", keep="first").reset_index(
        drop=True
    )
    area_covariates = collapse_exact_key_duplicates(
        area_covariates,
        keys=("area",),
        value_columns=tuple(
            column for column in area_covariates.columns if column != "area"
        ),
        frame_name="covariates_df",
    )
    area_covariates = area_covariates.merge(
        pop_area,
        how="left",
        on="area",
        validate="one_to_one",
    )
    pop_val = (
        pd.to_numeric(area_covariates[pop_col], errors="coerce")
        if pop_col in area_covariates.columns
        else pd.Series(np.nan, index=area_covariates.index)
    )
    area_covariates["pop_val"] = pop_val.where(
        np.isfinite(pop_val), area_covariates[".coverage_population"]
    )
    join_covariates = area_covariates.drop(columns=[".coverage_population", "pop_val"])
    origin_cov = join_covariates.rename(
        columns={col: f"{col}_o" for col in join_covariates.columns if col != "area"}
    ).rename(columns={"area": "origin"})
    dest_cov = join_covariates.rename(
        columns={col: f"{col}_d" for col in join_covariates.columns if col != "area"}
    ).rename(columns={"area": "destination"})
    base_df = base_df.merge(origin_cov, how="left", on="origin", validate="many_to_one")
    base_df = base_df.merge(
        dest_cov, how="left", on="destination", validate="many_to_one"
    )
    base_df = base_df.merge(
        area_covariates.loc[:, ["area", "pop_val"]].rename(
            columns={"area": "origin", "pop_val": "pop_o"}
        ),
        how="left",
        on="origin",
        validate="many_to_one",
    )
    base_df = base_df.merge(
        area_covariates.loc[:, ["area", "pop_val"]].rename(
            columns={"area": "destination", "pop_val": "pop_d"}
        ),
        how="left",
        on="destination",
        validate="many_to_one",
    )
    if len(base_df) != len(mpd_od_df) or bool(base_df["row_id"].duplicated().any()):
        raise DebiasSchemaError(
            "Multilevel input joins changed the MPD row identity; distance, "
            "coverage, and covariate lookups must be one-to-one per key."
        )
    base_df["pop_o"] = base_df["pop_o"].where(
        np.isfinite(base_df["pop_o"]), base_df["population"]
    )
    base_df["pop_d"] = base_df["pop_d"].where(
        np.isfinite(base_df["pop_d"]), base_df["pop_o"]
    )
    base_df["log_pop_o"] = np.log(np.maximum(base_df["pop_o"], np.finfo(float).eps))
    base_df["log_pop_d"] = np.log(np.maximum(base_df["pop_d"], np.finfo(float).eps))
    base_df["coverage_rate_o"] = base_df["user_count"] / base_df["population"]
    base_df["coverage_rate_d"] = (
        base_df["user_count_d_coverage"] / base_df["population_d_coverage"]
    )
    base_df["log_coverage_o"] = np.log(base_df["coverage_rate_o"])
    base_df["log_coverage_d"] = np.log(base_df["coverage_rate_d"])
    base_df["bias_e_origin"] = 1 - base_df["coverage_rate_o"]
    if default_covariate_col is not None:
        if f"{default_covariate_col}_o" in base_df.columns:
            base_df["income_o"] = pd.to_numeric(
                base_df[f"{default_covariate_col}_o"], errors="coerce"
            )
        if f"{default_covariate_col}_d" in base_df.columns:
            base_df["income_d"] = pd.to_numeric(
                base_df[f"{default_covariate_col}_d"], errors="coerce"
            )
    req_fit = ["flow", "log_distance", "bias_e_origin"]
    model_df = base_df.loc[
        np.isfinite(base_df["flow"])
        & (base_df["flow"] >= 0)
        & np.logical_and.reduce([np.isfinite(base_df[col]) for col in req_fit])
    ].copy()
    has_pop_terms = bool(
        np.isfinite(model_df.get("log_pop_o", pd.Series(dtype=float))).all()
        and np.isfinite(model_df.get("log_pop_d", pd.Series(dtype=float))).all()
    )
    if not has_pop_terms:
        model_df["log_pop_o"] = np.nan
        model_df["log_pop_d"] = np.nan
    dropped_n = len(base_df) - len(model_df)
    if dropped_n > 0:
        warnings.warn(
            f"{dropped_n} row(s) were excluded from model fitting due to "
            "missing/non-finite required fields.",
            stacklevel=2,
        )
    return {
        "base_df": base_df,
        "model_df": model_df,
        "default_covariate_col": default_covariate_col,
        "has_pop_terms": has_pop_terms,
        "distance_source": distance_source,
    }


def _formula_terms(
    prediction_df: pd.DataFrame,
    formula_info: dict[str, Any],
    default_covariate_col: str | None,
    include_pop_terms: bool,
    scenario_info: dict[str, Any],
    random_intercept: str,
    target_scale: str,
) -> list[str]:
    if formula_info["formula"] is not None:
        terms = _parse_rhs_terms(
            str(formula_info["formula"]), allow_random_intercept=True
        )
        missing = [term for term in terms if term not in prediction_df.columns]
        if missing:
            raise DebiasSchemaError(
                "`formula` references variable(s) not found in prepared model "
                "data: " + ", ".join(missing)
            )
    else:
        terms = []
        if default_covariate_col is not None:
            terms.extend(["income_o", "income_d"])
        terms.append("log_distance")
        if target_scale == "mpd_counterfactual":
            terms.append("bias_e_origin")
        if include_pop_terms:
            terms.extend(["log_pop_o", "log_pop_d"])
        repeated = scenario_info["repeated_observation"]
        if (
            repeated in {"source", "source_time"}
            and random_intercept != "source"
            and "mpd_source" in prediction_df.columns
            and int(scenario_info["n_sources"]) > 1
        ):
            terms.append("mpd_source")
        if (
            repeated in {"time", "source_time"}
            and random_intercept != "time"
            and "mpd_time" in prediction_df.columns
            and int(scenario_info["n_time_periods"]) > 1
        ):
            terms.append("mpd_time")
    return list(
        dict.fromkeys([term for term in terms if term in prediction_df.columns])
    )


def _frequentist_formula(formula_info: dict[str, Any], terms: list[str]) -> str:
    if formula_info["formula"] is not None:
        return str(formula_info["formula"])
    rhs = " + ".join(terms) if terms else "1"
    return f"flow ~ {rhs}"


def _random_intercept_group_col(random_intercept: str) -> str | None:
    return {
        "origin": "origin",
        "destination": "destination",
        "od": "od_id",
        "source": "mpd_source",
        "time": "mpd_time",
        "source_time": "mpd_source_time",
        "none": None,
    }[random_intercept]


def _add_random_intercept_to_formula(fixed_formula: str, group_col: str | None) -> str:
    if group_col is None:
        return fixed_formula
    return f"{fixed_formula} + (1 | {group_col})"


def _bias_terms(formula_info: dict[str, Any], terms: list[str]) -> list[str]:
    explicit = [
        term for term in formula_info.get("bias_variables", []) if term in terms
    ]
    if explicit:
        return explicit
    return ["bias_e_origin"] if "bias_e_origin" in terms else []


def _counterfactual_bias_data(
    data: pd.DataFrame, bias_terms: list[str]
) -> pd.DataFrame:
    out = data.copy()
    for term in bias_terms:
        if term in out.columns:
            out[term] = 0
    return out


def _set_observation_probability(
    data: pd.DataFrame, coverage_scale: str
) -> pd.DataFrame:
    out = data.copy()
    if coverage_scale == "origin":
        out["observation_probability"] = out["coverage_rate_o"]
    elif coverage_scale == "destination":
        out["observation_probability"] = out["coverage_rate_d"]
    else:
        out["observation_probability"] = np.sqrt(
            out["coverage_rate_o"] * out["coverage_rate_d"]
        )
    out["log_observation_probability"] = np.log(out["observation_probability"])
    return out


def _validate_coverage_offset_data(data: pd.DataFrame, coverage_scale: str) -> None:
    needed = {
        "origin": ["coverage_rate_o"],
        "destination": ["coverage_rate_d"],
        "both": ["coverage_rate_o", "coverage_rate_d"],
    }[coverage_scale]
    missing = [col for col in needed if col not in data.columns]
    if missing:
        raise DebiasSchemaError(
            "`observation_model = 'coverage_offset'` requires prepared coverage "
            "column(s): " + ", ".join(missing)
        )
    invalid = [
        col
        for col in needed
        if not bool((np.isfinite(data[col]) & (data[col] > 0)).all())
    ]
    if invalid:
        raise DebiasSchemaError(
            "`observation_model = 'coverage_offset'` requires positive finite "
            "coverage rates. Invalid column(s): " + ", ".join(invalid)
        )


def _resolve_optional_col(
    df: pd.DataFrame, requested: str | None, default: str, arg_name: str
) -> str | None:
    if requested is not None:
        if requested not in df.columns:
            raise DebiasSchemaError(f"`{arg_name}` must name a column in `mpd_od_df`.")
        return requested
    return default if default in df.columns else None


def _distinct_values(df: pd.DataFrame, col: str | None) -> list[str]:
    if col is None or col not in df.columns:
        return []
    return sorted(x for x in df[col].dropna().astype(str).unique() if x)


def _validate_scenario(
    scenario: str,
    source_col: str | None,
    time_col: str | None,
    has_source_variation: bool,
    has_time_variation: bool,
) -> None:
    if scenario == "s1" and (has_source_variation or has_time_variation):
        raise DebiasSchemaError(
            "`scenario = 's1'` requires at most one source and one time period."
        )
    if scenario == "s2" and (not has_time_variation or has_source_variation):
        raise DebiasSchemaError(
            "`scenario = 's2'` requires one source and at least two time periods."
        )
    if scenario == "s3" and (not has_source_variation or has_time_variation):
        raise DebiasSchemaError(
            "`scenario = 's3'` requires at least two sources and one time period."
        )
    if scenario == "s4" and (not has_source_variation or not has_time_variation):
        raise DebiasSchemaError(
            "`scenario = 's4'` requires at least two sources and at least "
            "two time periods."
        )
    if scenario in {"s2", "s4"} and time_col is None:
        raise DebiasSchemaError(f"`time_col` is required for scenario {scenario}.")
    if scenario in {"s3", "s4"} and source_col is None:
        raise DebiasSchemaError(f"`source_col` is required for scenario {scenario}.")


def _validate_repeated(
    repeated: str,
    source_col: str | None,
    time_col: str | None,
    has_source_variation: bool,
    has_time_variation: bool,
) -> None:
    if repeated in {"source", "source_time"} and (
        source_col is None or not has_source_variation
    ):
        raise DebiasSchemaError(
            f"`repeated_observation = '{repeated}'` requires at least two sources."
        )
    if repeated in {"time", "source_time"} and (
        time_col is None or not has_time_variation
    ):
        raise DebiasSchemaError(
            f"`repeated_observation = '{repeated}'` requires at least two time periods."
        )


def _resolve_default_covariate_col(
    covariates_df: pd.DataFrame,
    income_col: str | None,
    pop_col: str,
) -> str | None:
    if income_col is not None:
        if income_col not in covariates_df.columns:
            raise DebiasSchemaError(
                "`income_col` must name a column in `covariates_df`."
            )
        return income_col
    for candidate in ("income_norm", "rural_pct"):
        if candidate in covariates_df.columns:
            return candidate
    numeric_cols = [
        col
        for col in covariates_df.columns
        if col not in {"area", pop_col}
        and pd.api.types.is_numeric_dtype(covariates_df[col])
    ]
    return numeric_cols[0] if numeric_cols else None


def _audit_complete_grid(
    mpd_od_df: pd.DataFrame, flow_col: str, scenario_cols: list[str]
) -> pd.DataFrame:
    origins = set(mpd_od_df["origin"].astype(str))
    destinations = set(mpd_od_df["destination"].astype(str))
    area_set = origins | destinations
    key_cols = ["origin", "destination", *scenario_cols]
    has_self_flows = bool(
        (mpd_od_df["origin"].astype(str) == mpd_od_df["destination"].astype(str)).any()
    )
    expected = len(area_set) ** 2
    if not has_self_flows:
        expected = len(area_set) * max(len(area_set) - 1, 0)
    if scenario_cols:
        expected *= int(mpd_od_df.loc[:, scenario_cols].drop_duplicates().shape[0])
    duplicate_keys = int(mpd_od_df.duplicated(key_cols).sum())
    finite_non_negative = bool(
        (
            np.isfinite(pd.to_numeric(mpd_od_df[flow_col], errors="coerce"))
            & (mpd_od_df[flow_col] >= 0)
        ).all()
    )
    strict = (
        origins == destinations
        and len(mpd_od_df) == expected
        and duplicate_keys == 0
        and finite_non_negative
    )
    return pd.DataFrame(
        {
            "strict_square_support": [strict],
            "include_self_flows": [has_self_flows],
            "expected_od_rows": [expected],
            "actual_od_rows": [len(mpd_od_df)],
            "duplicate_key_rows": [duplicate_keys],
            "finite_non_negative_flows": [finite_non_negative],
        }
    )


def _one_sided_rhs(formula: str, arg_name: str) -> str:
    if formula.count("~") != 1 or formula.split("~", 1)[0].strip():
        raise DebiasSchemaError(
            f"`{arg_name}` must be a one-sided formula such as `~ bias_e_origin`."
        )
    rhs = formula.split("~", 1)[1].strip()
    if not rhs:
        raise DebiasSchemaError(f"`{arg_name}` must contain a right-hand side.")
    return rhs


def _formula_fixed_variables(rhs: str) -> list[str]:
    fixed_rhs = re.sub(r"\([^()]*\|[^()]*\)", " ", rhs)
    identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", fixed_rhs)
    functions = {
        match.group(1)
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", fixed_rhs)
    }
    ignored = {
        "flow",
        "np",
        "offset",
        "log",
        "exp",
        "sqrt",
        "I",
        "C",
        "bs",
        "poly",
        "True",
        "False",
    }
    return list(
        dict.fromkeys(
            name
            for name in identifiers
            if name not in ignored and name not in functions
        )
    )


def _parse_rhs_terms(
    formula: str, *, allow_random_intercept: bool = False
) -> list[str]:
    if "~" not in formula:
        raise DebiasSchemaError("`formula` must include a response, usually `flow`.")
    lhs, rhs = formula.split("~", 1)
    if lhs.strip() != "flow":
        raise DebiasSchemaError(
            "`formula` response must be `flow`; `flow_col` is standardised to "
            "`flow` before fitting."
        )
    _validate_supported_formula_rhs(
        rhs,
        "formula",
        allow_random_intercept=allow_random_intercept,
    )
    return _formula_fixed_variables(rhs)


def _validate_supported_formula_rhs(
    rhs: str,
    arg_name: str,
    *,
    allow_random_intercept: bool = False,
) -> None:
    random_terms = re.findall(r"\(([^()]*)\|([^()]*)\)", rhs)
    stripped = re.sub(r"\([^()]*\|[^()]*\)", " ", rhs)
    if random_terms and not allow_random_intercept:
        raise NotImplementedError(
            f"`{arg_name}` random-effect terms are not translated in the "
            "debiaspy frequentist slice; fixed-dummy substitution is refused."
        )
    if allow_random_intercept and random_terms:
        _validate_random_effect_terms(random_terms, rhs)
    if "|" in stripped:
        raise NotImplementedError(
            f"`{arg_name}` contains a random-effect structure that is not a "
            "single `(1 | group)` term."
        )
    if re.search(r"\boffset\s*\(", stripped):
        raise DebiasSchemaError(
            f"`{arg_name}` uses a user-specified offset, which is not translated "
            "in the frequentist slice. Use `observation_model='coverage_offset'` "
            "for the supported coverage offset."
        )
    if any(token in stripped for token in [":", "*", "/", "^"]):
        raise DebiasSchemaError(
            f"`{arg_name}` interactions are not yet translated in the Python "
            "multilevel scaffold."
        )
    transformed = re.sub(r"(?:np\.)?log\([^)]+\)", "x", stripped)
    unsupported = re.findall(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", transformed)
    if unsupported:
        names = ", ".join(sorted({name.strip(" (") for name in unsupported}))
        raise DebiasSchemaError(
            f"`{arg_name}` uses unsupported transform(s): {names}. The Python "
            "scaffold currently supports plain columns and log(column)."
        )


def _validate_random_effect_terms(terms: list[tuple[str, str]], formula: str) -> None:
    if "||" in formula:
        raise NotImplementedError(
            "Double-bar random-effect syntax is not translated by the "
            "frequentist GPBoost backend."
        )
    if len(terms) != 1:
        raise NotImplementedError(
            "The frequentist GPBoost backend supports exactly one random-"
            "intercept term; multiple random-effect terms are not translated."
        )
    slope, group = terms[0]
    if slope.strip() != "1":
        raise NotImplementedError(
            "The frequentist GPBoost backend supports a single random "
            "intercept only; random slopes are not translated and are not "
            "approximated with fixed dummies."
        )
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", group.strip()):
        raise NotImplementedError(
            "The frequentist GPBoost backend requires one plain grouping "
            "column in `(1 | group)`; nested or crossed grouping expressions "
            "are not translated."
        )


def _frequentist_random_effect_spec(formula: str) -> dict[str, str] | None:
    if "~" not in formula:
        return None
    lhs, rhs = formula.split("~", 1)
    terms = re.findall(r"\(([^()]*)\|([^()]*)\)", rhs)
    if not terms:
        if "|" in rhs:
            raise NotImplementedError(
                "The frequentist GPBoost backend could not parse the random-"
                "effect term. Use exactly one `(1 | group)` term."
            )
        return None
    _validate_random_effect_terms(terms, formula)
    group_col = terms[0][1].strip()
    selector_by_group = {
        "origin": "origin",
        "destination": "destination",
        "od_id": "od",
        "mpd_source": "source",
        "mpd_time": "time",
        "mpd_source_time": "source_time",
    }
    if group_col not in selector_by_group:
        raise NotImplementedError(
            "The frequentist GPBoost backend supports random intercepts for "
            "origin, destination, od_id, mpd_source, mpd_time, or "
            "mpd_source_time only."
        )
    fixed_rhs = re.sub(r"\([^()]*\|[^()]*\)", " ", rhs)
    fixed_rhs = re.sub(r"\s*\+\s*\+\s*", " + ", fixed_rhs)
    fixed_rhs = re.sub(r"^\s*\+|\+\s*$", "", fixed_rhs).strip()
    if not fixed_rhs:
        fixed_rhs = "1"
    return {
        "selector": selector_by_group[group_col],
        "group_col": group_col,
        "fixed_formula": f"{lhs.strip()} ~ {fixed_rhs}",
    }


def _clean_formula_terms(rhs: str) -> list[str]:
    rhs = re.sub(r"\([^)]*\|[^)]*\)", " ", rhs)
    rhs = rhs.replace(":", "+").replace("*", "+")
    tokens = [token.strip() for token in rhs.split("+")]
    out = []
    for token in tokens:
        token = token.strip()
        if not token or token in {"1", "0", "-1"} or token.startswith("offset("):
            continue
        token = re.sub(r"^np\.log\(([^)]+)\)$", r"\1", token)
        token = re.sub(r"^log\(([^)]+)\)$", r"\1", token)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token):
            out.append(token)
    return list(dict.fromkeys(out))
