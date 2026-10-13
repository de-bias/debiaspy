from __future__ import annotations

import argparse
import json
import os
import subprocess
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from pandas.testing import assert_frame_equal

import debiaspy
from debiaspy import (
    adjust_all_methods,
    adjust_coefficient,
    adjust_inverse_penetration,
    adjust_multilevel_bayes,
    adjust_raking_ratio,
    adjust_selection_rate,
    adjust_selection_rate2,
    debiaspy_example_data,
    measure_bias,
    measure_bias_distribution,
    validate_bias_residual_structure,
    validate_flow_distribution_methods,
    validate_flow_margins,
    validate_flow_overall_methods,
    validate_flow_prepare_outputs,
    validate_flow_residual_methods,
    validate_flow_residual_structure_methods,
)

DISPLAY_DIR: Path | None = None
RAW_DIR: Path | None = None


def schema_path_for(path: Path) -> Path:
    return path.with_name(path.name.removesuffix(".csv") + ".schema.json")


def table_schema(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [
            {
                "name": str(column),
                "logical_type": logical_schema_type(frame[column]),
                "missing_count": int(frame[column].isna().sum()),
            }
            for column in frame.columns
        ],
    }


def logical_schema_type(series: pd.Series) -> str:
    if bool(series.isna().all()):
        return "missing"
    if is_bool_dtype(series):
        return "boolean"
    if is_numeric_dtype(series):
        return "numeric"
    if is_datetime64_any_dtype(series):
        return "datetime"
    return "string"


def write_frame_csv(frame: pd.DataFrame, path: Path, float_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        na_rep="__DEBIASPY_NA__",
        float_format=float_format,
    )


def canonicalize_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame.copy()
    numeric_columns = [
        column
        for column in canonical.columns
        if is_numeric_dtype(canonical[column]) and not is_bool_dtype(canonical[column])
    ]
    for column in numeric_columns:
        values = canonical[column].copy()
        values = values.mask(values.abs() <= 1e-12, 0.0)
        canonical[column] = values
    return canonical


def flatten_overall_method_result(result: dict[str, Any]) -> pd.DataFrame:
    summary = result.get("summary")
    if not isinstance(summary, pd.DataFrame):
        from debiaspy.validate_flows import (  # noqa: PLC0415
            flatten_validate_flow_overall_result,
        )

        return flatten_validate_flow_overall_result(result)
    method = result.get("method", np.nan)
    out = summary.add_prefix("summary.").copy()
    out.insert(0, "method", method)
    return out.reset_index(drop=True)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


ARTICLE_PAGES = (
    "index",
    "v03-getting-set-up",
    "v04-measuring-coverage-bias",
    "v05-identifying-and-explaining-bias",
    "v06-adjusting-biases",
    "v07-validation",
    "v09-advanced-bayesian-adjustment",
)

ARTICLE_URLS = {
    "index": "https://de-bias.github.io/debiasR/index.html",
    "v03-getting-set-up": (
        "https://de-bias.github.io/debiasR/articles/v03-getting-set-up.html"
    ),
    "v04-measuring-coverage-bias": (
        "https://de-bias.github.io/debiasR/articles/v04-measuring-coverage-bias.html"
    ),
    "v05-identifying-and-explaining-bias": (
        "https://de-bias.github.io/debiasR/articles/"
        "v05-identifying-and-explaining-bias.html"
    ),
    "v06-adjusting-biases": (
        "https://de-bias.github.io/debiasR/articles/v06-adjusting-biases.html"
    ),
    "v07-validation": "https://de-bias.github.io/debiasR/articles/v07-validation.html",
    "v09-advanced-bayesian-adjustment": (
        "https://de-bias.github.io/debiasR/articles/"
        "v09-advanced-bayesian-adjustment.html"
    ),
}

ARTICLE_CSV_NAMES = (
    "index.public_exports.csv",
    "index.workflow_pages.csv",
    "v03-getting-set-up.data_counts.csv",
    "v03-getting-set-up.input_schemas.csv",
    "v03-getting-set-up.lad_OD_travel2work.head.csv",
    "v03-getting-set-up.census_lad_OD_travel2work.head.csv",
    "v03-getting-set-up.coverage_lad.head.csv",
    "v03-getting-set-up.lad_covariates.head.csv",
    "v03-getting-set-up.lad_centroids.head.csv",
    "v04-measuring-coverage-bias.coverage_lad.head.csv",
    "v04-measuring-coverage-bias.coverage_renamed.head.csv",
    "v04-measuring-coverage-bias.measure_bias.head.csv",
    "v04-measuring-coverage-bias.coverage_totals.csv",
    "v04-measuring-coverage-bias.lowest_bias.head.csv",
    "v04-measuring-coverage-bias.distribution.summary.csv",
    "v04-measuring-coverage-bias.distribution.top_share_diff.csv",
    "v04-measuring-coverage-bias.residual.summary.csv",
    "v04-measuring-coverage-bias.residual.definitions.csv",
    "v04-measuring-coverage-bias.residual.area_head.csv",
    "v04-measuring-coverage-bias.plot_contracts.csv",
    "v05-identifying-and-explaining-bias.coverage.head.csv",
    "v05-identifying-and-explaining-bias.covariates.head.csv",
    "v05-identifying-and-explaining-bias.bias.head.csv",
    "v05-identifying-and-explaining-bias.residual.head.csv",
    "v05-identifying-and-explaining-bias.bias_covariates.head.csv",
    "v05-identifying-and-explaining-bias.coefficient_table.csv",
    "v05-identifying-and-explaining-bias.fitted_residual.head.csv",
    "v05-identifying-and-explaining-bias.top_unexplained.csv",
    "v05-identifying-and-explaining-bias.plot_contracts.csv",
    "v06-adjusting-biases.raw_heads.OD_travel2work.csv",
    "v06-adjusting-biases.raw_heads.census_OD_travel2work.csv",
    "v06-adjusting-biases.raw_heads.coverage.csv",
    "v06-adjusting-biases.example_counts.csv",
    "v06-adjusting-biases.inverse_penetration.preview.csv",
    "v06-adjusting-biases.selection_rate.preview.csv",
    "v06-adjusting-biases.selection_rate.attrs.csv",
    "v06-adjusting-biases.selection_rate2.preview.csv",
    "v06-adjusting-biases.selection_rate2.attrs.csv",
    "v06-adjusting-biases.raking_ratio.preview.csv",
    "v06-adjusting-biases.raking_ratio.summary.csv",
    "v06-adjusting-biases.coefficient.preview.csv",
    "v06-adjusting-biases.coefficient.attrs.csv",
    "v06-adjusting-biases.bayesian_precomputed.display.csv",
    "v06-adjusting-biases.bayesian_precomputed.metadata.csv",
    "v06-adjusting-biases.bayesian_precomputed.model_terms.csv",
    "v06-adjusting-biases.adjust_all_methods.summary.csv",
    "v07-validation.example_counts.csv",
    "v07-validation.method_set.csv",
    "v07-validation.bayesian_precomputed.display.csv",
    "v07-validation.bayesian_precomputed.metadata.csv",
    "v07-validation.level1.overall_metrics.csv",
    "v07-validation.level2.origin_margin_metrics.csv",
    "v07-validation.level2.destination_margin_metrics.csv",
    "v07-validation.level3.residual_summary.csv",
    "v07-validation.level3.top_worst.csv",
    "v07-validation.level4.distribution_summary.csv",
    "v07-validation.level4.origin_head.csv",
    "v07-validation.level5.neighbor_summary.csv",
    "v07-validation.level5.structure_summary.csv",
    "v07-validation.plot_contracts.csv",
    "v09-advanced-bayesian-adjustment.data_counts.csv",
    "v09-advanced-bayesian-adjustment.mpd_od.head.csv",
    "v09-advanced-bayesian-adjustment.coverage.head.csv",
    "v09-advanced-bayesian-adjustment.covariates.head.csv",
    "v09-advanced-bayesian-adjustment.distance.head.csv",
    "v09-advanced-bayesian-adjustment.benchmark.head.csv",
    "v09-advanced-bayesian-adjustment.coverage_offset_frequentist.head.csv",
    "v09-advanced-bayesian-adjustment.reduced_form_frequentist.head.csv",
    "v09-advanced-bayesian-adjustment.bayesian_precomputed.output.csv",
    "v09-advanced-bayesian-adjustment.bayesian_precomputed.summary.csv",
    "v09-advanced-bayesian-adjustment.bayesian_precomputed.diagnostics.csv",
    "v09-advanced-bayesian-adjustment.bayesian_backend_contract.csv",
)

ARTICLE_META_JSON_NAMES = tuple(f"{page}.meta.json" for page in ARTICLE_PAGES)
ARTICLE_SCHEMA_JSON_NAMES = tuple(
    name.removesuffix(".csv") + ".schema.json" for name in ARTICLE_CSV_NAMES
)
ARTICLE_JSON_NAMES = ARTICLE_META_JSON_NAMES + ARTICLE_SCHEMA_JSON_NAMES

VALIDATION_METHOD_LABELS = {
    "unadjusted": "Unadjusted raw MPD",
    "inverse_penetration": "Inverse penetration",
    "selection_rate2": "Selection rate II",
    "coefficient": "Coefficient adjustment",
    "bayes_gravity": "Bayesian gravity",
    "bayes_gravity_education": "Bayesian gravity + education",
    "bayes_origin_pool": "Bayesian gravity + education + origin pooling",
}

ARTICLE_STABLE_DISPLAY_ROUNDING = {
    "v07-validation.level2.origin_margin_metrics.csv": {"spearman_rho": 3},
    "v07-validation.level2.destination_margin_metrics.csv": {"spearman_rho": 3},
}


def _read_r_dataset(name: str) -> pd.DataFrame:
    code = (
        "env <- new.env(); name <- " + json.dumps(name) + "; "
        'utils::data(list=name, package="debiasRdata", envir=env); '
        "write.csv(get(name, envir=env, inherits=FALSE), stdout(), row.names=FALSE)"
    )
    result = subprocess.run(
        ["Rscript", "-e", code], check=True, capture_output=True, text=True
    )
    return pd.read_csv(StringIO(result.stdout))


def write_article_outputs(repo_root: Path, out_dir: Path) -> None:

    raw = {
        name: _read_r_dataset(name)
        for name in (
            "lad_OD_travel2work",
            "census_lad_OD_travel2work",
            "coverage_lad",
            "lad_covariates",
            "lad_centroids",
        )
    }

    write_index_outputs(out_dir, debiaspy.__all__)
    write_page_meta(out_dir, "index", "public API and workflow page map")
    write_v03_outputs(out_dir, raw)
    write_page_meta(out_dir, "v03-getting-set-up", "raw debiasRdata inputs")
    write_v04_outputs(out_dir, raw["coverage_lad"])
    write_page_meta(out_dir, "v04-measuring-coverage-bias", "coverage-bias tables")
    write_v05_outputs(out_dir, raw["coverage_lad"], raw["lad_covariates"])
    write_page_meta(
        out_dir, "v05-identifying-and-explaining-bias", "residual-bias regression"
    )
    write_v06_outputs(out_dir, repo_root, raw)
    write_page_meta(out_dir, "v06-adjusting-biases", "adjustment-method outputs")
    write_v07_outputs(out_dir, repo_root)
    write_page_meta(out_dir, "v07-validation", "validation calls executed without MCMC")
    write_v09_outputs(out_dir, repo_root)
    write_page_meta(
        out_dir,
        "v09-advanced-bayesian-adjustment",
        "model_engine='frequentist' maximum-likelihood API-selector calls plus static "
        "Python backend inventory",
    )


def write_index_outputs(out_dir: Path, python_exports: list[str]) -> None:
    write_csv(
        out_dir,
        "index.public_exports.csv",
        pd.DataFrame({"function": sorted(python_exports)}),
    )
    write_csv(
        out_dir,
        "index.workflow_pages.csv",
        pd.DataFrame(
            {
                "order": list(range(1, len(ARTICLE_PAGES) + 1)),
                "page": ARTICLE_PAGES,
                "url": [ARTICLE_URLS[page] for page in ARTICLE_PAGES],
                "scope": [
                    "package overview",
                    "setup data inputs",
                    "coverage bias",
                    "explaining bias",
                    "adjusting flows",
                    "validation",
                    "advanced multilevel",
                ],
            }
        ),
    )


def write_v03_outputs(out_dir: Path, raw: dict[str, pd.DataFrame]) -> None:
    objects = (
        "lad_OD_travel2work",
        "census_lad_OD_travel2work",
        "coverage_lad",
        "lad_covariates",
        "lad_centroids",
    )
    write_csv(
        out_dir,
        "v03-getting-set-up.data_counts.csv",
        pd.DataFrame(
            {
                "object": objects,
                "rows": [len(raw[name]) for name in objects],
                "columns": [len(raw[name].columns) for name in objects],
            }
        ),
    )
    schema_rows = []
    for object_name in objects:
        frame = raw[object_name]
        for column in frame.columns:
            series = frame.loc[:, column]
            if isinstance(series, pd.DataFrame):
                raise RuntimeError(f"Duplicate column in {object_name}: {column}")
            schema_rows.append(
                {
                    "object": object_name,
                    "column": column,
                    "logical_type": logical_schema_type(series),
                    "missing_count": int(series.isna().to_numpy().sum()),
                }
            )
    write_csv(
        out_dir,
        "v03-getting-set-up.input_schemas.csv",
        pd.DataFrame(schema_rows),
    )
    for object_name in objects:
        write_csv(
            out_dir,
            f"v03-getting-set-up.{object_name}.head.csv",
            raw[object_name].head(6).reset_index(drop=True),
        )


def write_v04_outputs(out_dir: Path, coverage_lad: pd.DataFrame) -> None:
    coverage = coverage_lad.rename(columns={"code": "area"})
    bias_lad = measure_bias(coverage)
    distribution = measure_bias_distribution(
        coverage_df=coverage,
        area_col="area",
        population_col="population",
        user_count_col="user_count",
    )
    residuals = validate_bias_residual_structure(
        coverage_df=coverage,
        coverage_area_col="area",
        population_col="population",
        user_count_col="user_count",
    )
    residual_lad = residuals["area_level"].copy()
    residual_lad["baseline_coverage_bias"] = 1 - residual_lad["global_coverage_score"]
    residual_lad["residual_bias"] = (
        residual_lad["coverage_bias"] - residual_lad["baseline_coverage_bias"]
    )

    write_csv(
        out_dir,
        "v04-measuring-coverage-bias.coverage_lad.head.csv",
        coverage_lad.head(6),
    )
    write_csv(
        out_dir,
        "v04-measuring-coverage-bias.coverage_renamed.head.csv",
        coverage.head(6),
    )
    write_csv(
        out_dir,
        "v04-measuring-coverage-bias.measure_bias.head.csv",
        bias_lad.loc[
            :,
            [
                "area",
                "name",
                "population",
                "user_count",
                "coverage_score",
                "coverage_bias",
            ],
        ].head(6),
    )
    write_csv(
        out_dir,
        "v04-measuring-coverage-bias.coverage_totals.csv",
        pd.DataFrame(
            {
                "lad_count": [len(coverage)],
                "census_pop": [coverage["population"].sum()],
                "digital_trace_pop": [coverage["user_count"].sum()],
                "overall_coverage": [
                    coverage["user_count"].sum() / coverage["population"].sum()
                ],
                "overall_coverage_bias": [
                    1 - coverage["user_count"].sum() / coverage["population"].sum()
                ],
            }
        ),
    )
    write_csv(
        out_dir,
        "v04-measuring-coverage-bias.lowest_bias.head.csv",
        bias_lad.sort_values(["coverage_bias", "area"])
        .loc[:, ["area", "name", "coverage_score", "coverage_bias"]]
        .head(6),
    )
    write_csv(
        out_dir,
        "v04-measuring-coverage-bias.distribution.summary.csv",
        distribution["summary"],
    )
    write_csv(
        out_dir,
        "v04-measuring-coverage-bias.distribution.top_share_diff.csv",
        distribution["area_level"]
        .assign(
            abs_share_difference=lambda frame: frame[
                "share_difference_user_minus_population"
            ].abs()
        )
        .sort_values(["abs_share_difference", "area"], ascending=[False, True])
        .drop(columns=["abs_share_difference"])
        .head(6),
    )
    write_csv(
        out_dir,
        "v04-measuring-coverage-bias.residual.summary.csv",
        residuals["summary"],
    )
    write_csv(
        out_dir,
        "v04-measuring-coverage-bias.residual.definitions.csv",
        residuals["residual_definitions"],
    )
    write_csv(
        out_dir,
        "v04-measuring-coverage-bias.residual.area_head.csv",
        residual_lad.loc[
            :,
            [
                "area",
                "population",
                "user_count",
                "coverage_score",
                "coverage_bias",
                "expected_user_count",
                "residual_bias",
            ],
        ].head(6),
    )
    write_csv(
        out_dir,
        "v04-measuring-coverage-bias.plot_contracts.csv",
        pd.DataFrame(
            {
                "plot": [
                    "coverage_bias_histogram",
                    "population_user_count_scatter",
                    "baseline_residual_panel",
                    "residual_bias_histogram",
                ],
                "source_rows": [
                    len(bias_lad),
                    len(bias_lad),
                    len(residual_lad),
                    len(residual_lad),
                ],
                "x": ["coverage_bias", "population", "population", "residual_bias"],
                "y": ["count", "user_count", "residual_bias", "count"],
            }
        ),
    )


def write_v05_outputs(
    out_dir: Path, coverage_lad: pd.DataFrame, covariates: pd.DataFrame
) -> None:
    coverage = coverage_lad.rename(columns={"code": "area"})
    bias_lad = measure_bias(coverage)
    residuals = validate_bias_residual_structure(
        coverage_df=coverage,
        coverage_area_col="area",
        population_col="population",
        user_count_col="user_count",
    )
    residual_lad = residuals["area_level"].copy()
    residual_lad["baseline_coverage_bias"] = 1 - residual_lad["global_coverage_score"]
    residual_lad["residual_bias"] = (
        residual_lad["coverage_bias"] - residual_lad["baseline_coverage_bias"]
    )
    residual_lad = residual_lad.merge(coverage.loc[:, ["area", "name"]], on="area")
    covariate_cols = [
        "area",
        "per_age_20.29",
        "per_age_70plus",
        "per_level4",
        "per_hh_no_centralheat",
        "per_NS_SeC_L13_routine",
        "rural_pct",
    ]
    bias_covariates = residual_lad.merge(covariates.loc[:, covariate_cols], on="area")
    coefficient_table, fitted = fit_v05_ols(bias_covariates)

    write_csv(
        out_dir,
        "v05-identifying-and-explaining-bias.coverage.head.csv",
        coverage.head(6),
    )
    write_csv(
        out_dir,
        "v05-identifying-and-explaining-bias.covariates.head.csv",
        covariates.head(6),
    )
    write_csv(
        out_dir,
        "v05-identifying-and-explaining-bias.bias.head.csv",
        bias_lad.loc[
            :, ["area", "population", "user_count", "coverage_score", "coverage_bias"]
        ].head(6),
    )
    write_csv(
        out_dir,
        "v05-identifying-and-explaining-bias.residual.head.csv",
        residual_lad.loc[
            :, ["area", "name", "coverage_score", "coverage_bias", "residual_bias"]
        ].head(6),
    )
    write_csv(
        out_dir,
        "v05-identifying-and-explaining-bias.bias_covariates.head.csv",
        bias_covariates.loc[
            :,
            [
                "area",
                "residual_bias",
                "per_age_20.29",
                "per_age_70plus",
                "per_level4",
                "rural_pct",
            ],
        ].head(6),
    )
    write_csv(
        out_dir,
        "v05-identifying-and-explaining-bias.coefficient_table.csv",
        coefficient_table,
    )
    write_csv(
        out_dir,
        "v05-identifying-and-explaining-bias.fitted_residual.head.csv",
        fitted.loc[
            :, ["area", "residual_bias", "fitted_residual_bias", "unexplained_residual"]
        ].head(6),
    )
    write_csv(
        out_dir,
        "v05-identifying-and-explaining-bias.top_unexplained.csv",
        fitted.assign(abs_unexplained=lambda frame: frame["unexplained_residual"].abs())
        .sort_values(["abs_unexplained", "area"], ascending=[False, True])
        .drop(columns=["abs_unexplained"])
        .loc[
            :,
            [
                "area",
                "name",
                "residual_bias",
                "fitted_residual_bias",
                "unexplained_residual",
            ],
        ]
        .head(10),
    )
    write_csv(
        out_dir,
        "v05-identifying-and-explaining-bias.plot_contracts.csv",
        pd.DataFrame(
            {
                "plot": [
                    "residual_vs_age_20_29",
                    "residual_vs_level4",
                    "observed_vs_fitted_residual",
                ],
                "source_rows": [
                    len(bias_covariates),
                    len(bias_covariates),
                    len(fitted),
                ],
                "x": ["per_age_20.29", "per_level4", "fitted_residual_bias"],
                "y": ["residual_bias", "residual_bias", "residual_bias"],
            }
        ),
    )


def write_v06_outputs(
    out_dir: Path, repo_root: Path, raw: dict[str, pd.DataFrame]
) -> None:
    example_data = debiaspy_example_data(n_areas=25, complete_grid=True)
    mpd_od = example_data["mpd_od"]
    benchmark_od = example_data["benchmark_od"]
    coverage = example_data["coverage"]
    covariates = example_data["covariates"]
    distance = example_data["distance"]
    lookup = benchmark_od.loc[:, ["origin", "destination", "flow"]].rename(
        columns={"flow": "flow_benchmark"}
    )

    write_csv(
        out_dir,
        "v06-adjusting-biases.raw_heads.OD_travel2work.csv",
        raw["lad_OD_travel2work"].head(6),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.raw_heads.census_OD_travel2work.csv",
        raw["census_lad_OD_travel2work"].head(6),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.raw_heads.coverage.csv",
        raw["coverage_lad"].rename(columns={"code": "area"}).head(6),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.example_counts.csv",
        object_counts(
            {
                key: example_data[key]
                for key in [
                    "mpd_od",
                    "benchmark_od",
                    "coverage",
                    "covariates",
                    "distance",
                ]
            }
        ),
    )

    inverse_penetration = adjust_inverse_penetration(
        mpd_od, coverage, weight_by="origin"
    )
    selection_rate = adjust_selection_rate(
        mpd_od,
        coverage,
        covariates_df=covariates,
        covariate_col="rural_pct",
        weight_by="origin",
        benchmark_od_df=benchmark_od,
        calibration_aggregate="origin",
    )
    selection_rate2 = adjust_selection_rate2(
        mpd_od,
        coverage,
        weight_by="origin",
        benchmark_od_df=benchmark_od,
        calibration_aggregate="origin",
    )
    raking_ratio = adjust_raking_ratio(mpd_od, benchmark_od_df=benchmark_od)
    coefficient = adjust_coefficient(
        mpd_od, benchmark_od, model_family="ols", level="od"
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.inverse_penetration.preview.csv",
        adjustment_preview(inverse_penetration, lookup),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.selection_rate.preview.csv",
        adjustment_preview(selection_rate, lookup),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.selection_rate.attrs.csv",
        attr_frame(selection_rate, ["r_global"]),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.selection_rate2.preview.csv",
        adjustment_preview(selection_rate2, lookup),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.selection_rate2.attrs.csv",
        attr_frame(selection_rate2, ["k"]),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.raking_ratio.preview.csv",
        adjustment_preview(raking_ratio, lookup),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.raking_ratio.summary.csv",
        raking_summary(raking_ratio),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.coefficient.preview.csv",
        adjustment_preview(coefficient, lookup),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.coefficient.attrs.csv",
        attr_frame(coefficient, ["coef"]),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.bayesian_precomputed.display.csv",
        read_extdata(repo_root, "v06-bayesian-s1-comparison.csv"),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.bayesian_precomputed.metadata.csv",
        read_extdata(repo_root, "v06-bayesian-s1-metadata.csv"),
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.bayesian_precomputed.model_terms.csv",
        read_extdata(repo_root, "v06-bayesian-s1-model-terms.csv"),
    )

    method_results = adjust_all_methods(
        mpd_od,
        coverage,
        benchmark_od_df=benchmark_od,
        covariates_df=covariates,
        distance_df=distance,
        covariate_col="rural_pct",
        multilevel_engine="frequentist",
    )
    write_csv(
        out_dir,
        "v06-adjusting-biases.adjust_all_methods.summary.csv",
        adjustment_set_summary(method_results),
    )


def write_v07_outputs(out_dir: Path, repo_root: Path) -> None:
    example_data = debiaspy_example_data(
        n_areas=float("inf"), complete_grid=True, geography="lad"
    )
    mpd_od = example_data["mpd_od"]
    benchmark_od = example_data["benchmark_od"]
    coverage = example_data["coverage"]
    covariates = example_data["covariates"]
    distance = example_data["distance"]

    deterministic_methods = ["inverse_penetration", "selection_rate2", "coefficient"]
    deterministic_results = adjust_all_methods(
        mpd_od,
        coverage,
        benchmark_od_df=benchmark_od,
        covariates_df=covariates,
        covariate_col="per_level4",
        methods=deterministic_methods,
    )
    deterministic_results = {
        "unadjusted": mpd_od.assign(flow_adj=mpd_od["flow"]),
        **deterministic_results,
    }
    deterministic_method_results = validate_flow_prepare_outputs(
        deterministic_results, mpd_df=mpd_od
    )
    bayesian_results = read_extdata(repo_root, "v07-validation-bayesian-adjusted.csv")
    bayesian_method_results = {
        method: frame.drop(columns=["method"]).reset_index(drop=True)
        for method, frame in bayesian_results.groupby("method", sort=False)
    }
    all_method_results = {**deterministic_method_results, **bayesian_method_results}
    level1_methods = [
        "unadjusted",
        "inverse_penetration",
        "selection_rate2",
        "coefficient",
        "bayes_gravity",
        "bayes_gravity_education",
        "bayes_origin_pool",
    ]
    validation_methods = [
        "inverse_penetration",
        "coefficient",
        "bayes_gravity_education",
        "unadjusted",
    ]

    write_csv(
        out_dir,
        "v07-validation.example_counts.csv",
        object_counts(
            {
                key: example_data[key]
                for key in [
                    "mpd_od",
                    "benchmark_od",
                    "coverage",
                    "covariates",
                    "distance",
                ]
            }
        ),
    )
    write_csv(
        out_dir,
        "v07-validation.method_set.csv",
        pd.DataFrame(
            {
                "method": level1_methods,
                "label": [
                    VALIDATION_METHOD_LABELS[method] for method in level1_methods
                ],
                "source": [
                    "executed_without_mcmc_or_raw"
                    if method in deterministic_method_results
                    else "precomputed_bayesian_fixture"
                    for method in level1_methods
                ],
            }
        ),
    )
    write_csv(
        out_dir,
        "v07-validation.bayesian_precomputed.display.csv",
        read_extdata(repo_root, "v07-validation-bayesian-display.csv"),
    )
    write_csv(
        out_dir,
        "v07-validation.bayesian_precomputed.metadata.csv",
        read_extdata(repo_root, "v07-validation-bayesian-metadata.csv"),
    )

    level1 = validate_flow_overall_methods(
        {method: all_method_results[method] for method in level1_methods},
        benchmark_od,
        comparisons="adjusted_vs_benchmark",
    )
    write_csv(
        out_dir,
        "v07-validation.level1.overall_metrics.csv",
        combine_overall(level1, level1_methods),
    )

    selected = {method: all_method_results[method] for method in validation_methods}
    origin_margins = validate_flow_margins(selected, benchmark_od, role="origin")
    origin_metrics = validate_flow_overall_methods(
        origin_margins["adjusted_outputs"],
        origin_margins["benchmark"],
        comparisons="adjusted_vs_benchmark",
    )
    destination_margins = validate_flow_margins(
        selected, benchmark_od, role="destination"
    )
    destination_metrics = validate_flow_overall_methods(
        destination_margins["adjusted_outputs"],
        destination_margins["benchmark"],
        comparisons="adjusted_vs_benchmark",
    )
    write_csv(
        out_dir,
        "v07-validation.level2.origin_margin_metrics.csv",
        combine_overall(origin_metrics, validation_methods),
    )
    write_csv(
        out_dir,
        "v07-validation.level2.destination_margin_metrics.csv",
        combine_overall(destination_metrics, validation_methods),
    )

    residuals = validate_flow_residual_methods(selected, benchmark_od, top_n=12)
    write_csv(
        out_dir,
        "v07-validation.level3.residual_summary.csv",
        combine_nested(residuals, validation_methods, "summary"),
    )
    write_csv(
        out_dir,
        "v07-validation.level3.top_worst.csv",
        method_order_head(
            combine_nested(residuals, validation_methods, "top_worst"),
            validation_methods,
            5,
        ),
    )

    distribution = validate_flow_distribution_methods(
        selected, benchmark_od, comparisons="adjusted_vs_benchmark"
    )
    write_csv(
        out_dir,
        "v07-validation.level4.distribution_summary.csv",
        combine_nested(distribution, validation_methods, "summary"),
    )
    write_csv(
        out_dir,
        "v07-validation.level4.origin_head.csv",
        method_order_head(
            combine_nested(distribution, validation_methods, "origin_level"),
            validation_methods,
            5,
        ),
    )

    neighbors = nearest_neighbors(distance, covariates["area"], k=4)
    structure_covariates = covariates.loc[:, ["area", "per_level4"]]
    residual_structure = validate_flow_residual_structure_methods(
        selected,
        benchmark_od,
        comparison="adjusted_vs_benchmark",
        spatial_role="origin",
        residual_aggregation="mean",
        area_neighbors=neighbors,
        local_moran=False,
        covariate_df=structure_covariates,
        covariate_col="per_level4",
        methods=validation_methods,
    )
    write_csv(
        out_dir,
        "v07-validation.level5.neighbor_summary.csv",
        neighbor_summary(neighbors),
    )
    write_csv(
        out_dir,
        "v07-validation.level5.structure_summary.csv",
        combine_nested(residual_structure, validation_methods, "summary"),
    )
    write_csv(
        out_dir,
        "v07-validation.plot_contracts.csv",
        pd.DataFrame(
            {
                "plot": [
                    "level1_metrics",
                    "level2_origin_scatter",
                    "level2_destination_scatter",
                    "level3_flow_scatter",
                    "level3_residual_bands",
                    "level4_distribution_pairwise",
                    "level5_structure",
                ],
                "source": [
                    "level1.overall_metrics",
                    "origin_margin_residuals",
                    "destination_margin_residuals",
                    "level3.residual_summary",
                    "level3.residual_summary",
                    "level4.distribution_summary",
                    "level5.structure_summary",
                ],
                "random_sampling": [False, False, False, False, False, False, False],
            }
        ),
    )


def write_v09_outputs(out_dir: Path, repo_root: Path) -> None:
    data = debiaspy_example_data(
        n_areas=float("inf"), complete_grid=True, geography="lad"
    )
    mpd_od = data["mpd_od"].assign(mpd_time="single_period")
    coverage = data["coverage"].assign(mpd_time="single_period")
    covariates = data["covariates"]
    distance = data["distance"]
    benchmark = data["benchmark_od"]

    write_csv(
        out_dir,
        "v09-advanced-bayesian-adjustment.data_counts.csv",
        object_counts(
            {
                "mpd_od": mpd_od,
                "coverage": coverage,
                "covariates": covariates,
                "distance": distance,
                "benchmark_od": benchmark,
            }
        ),
    )
    write_csv(
        out_dir, "v09-advanced-bayesian-adjustment.mpd_od.head.csv", mpd_od.head(6)
    )
    write_csv(
        out_dir, "v09-advanced-bayesian-adjustment.coverage.head.csv", coverage.head(6)
    )
    write_csv(
        out_dir,
        "v09-advanced-bayesian-adjustment.covariates.head.csv",
        covariates.head(6),
    )
    write_csv(
        out_dir, "v09-advanced-bayesian-adjustment.distance.head.csv", distance.head(6)
    )
    write_csv(
        out_dir,
        "v09-advanced-bayesian-adjustment.benchmark.head.csv",
        benchmark.head(6),
    )

    coverage_offset = adjust_multilevel_bayes(
        mpd_od_df=mpd_od,
        coverage_df=coverage,
        covariates_df=covariates,
        distance_df=distance,
        mobility_formula=(
            "~ log_pop_o + log_pop_d + log_distance + rural_pct_o + "
            "rural_pct_d + per_level4_o + per_level4_d"
        ),
        bias_formula="~ 0",
        target_scale="true_flow",
        observation_model="coverage_offset",
        coverage_scale="origin",
        model_engine="frequentist",
        scenario="s1",
        source_col="mpd_source",
        time_col="mpd_time",
        repeated_observation="none",
        prediction_scope="complete_grid",
        random_intercept="none",
        model_family="poisson",
    )
    reduced_form = adjust_multilevel_bayes(
        mpd_od_df=mpd_od,
        coverage_df=coverage,
        covariates_df=covariates,
        distance_df=distance,
        mobility_formula=(
            "~ log_pop_o + log_pop_d + log_distance + rural_pct_o + "
            "rural_pct_d + per_level4_o + per_level4_d"
        ),
        bias_formula="~ bias_e_origin",
        target_scale="mpd_counterfactual",
        observation_model="reduced_form",
        model_engine="frequentist",
        scenario="s1",
        source_col="mpd_source",
        time_col="mpd_time",
        repeated_observation="none",
        prediction_scope="complete_grid",
        random_intercept="none",
        model_family="poisson",
    )
    write_csv(
        out_dir,
        "v09-advanced-bayesian-adjustment.coverage_offset_frequentist.head.csv",
        coverage_offset.loc[
            :,
            [
                "origin",
                "destination",
                "flow",
                "flow_adj",
                "flow_mpd_pred",
                "flow_true_pred",
                "observation_probability",
            ],
        ].head(6),
    )
    write_csv(
        out_dir,
        "v09-advanced-bayesian-adjustment.reduced_form_frequentist.head.csv",
        reduced_form.loc[
            :, ["origin", "destination", "flow", "flow_adj", "flow_mpd_pred"]
        ].head(6),
    )
    write_csv(
        out_dir,
        "v09-advanced-bayesian-adjustment.bayesian_precomputed.output.csv",
        read_extdata(repo_root, "v09-bayesian-real-data-output.csv"),
    )
    write_csv(
        out_dir,
        "v09-advanced-bayesian-adjustment.bayesian_precomputed.summary.csv",
        read_extdata(repo_root, "v09-bayesian-real-data-summary.csv"),
    )
    write_csv(
        out_dir,
        "v09-advanced-bayesian-adjustment.bayesian_precomputed.diagnostics.csv",
        read_extdata(repo_root, "v09-bayesian-real-data-diagnostics.csv"),
    )
    write_csv(
        out_dir,
        "v09-advanced-bayesian-adjustment.bayesian_backend_contract.csv",
        bayesian_backend_contract(),
    )


def fit_v05_ols(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import scipy.stats as stats  # noqa: PLC0415

    predictors = [
        "per_age_20.29",
        "per_age_70plus",
        "per_level4",
        "per_hh_no_centralheat",
        "per_NS_SeC_L13_routine",
        "rural_pct",
    ]
    x = np.column_stack([np.ones(len(data)), data[predictors].to_numpy(dtype=float)])
    y = data["residual_bias"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted_values = x @ coef
    residual = y - fitted_values
    df_resid = len(y) - x.shape[1]
    sigma2 = float((residual @ residual) / df_resid)
    cov = sigma2 * np.linalg.inv(x.T @ x)
    std_error = np.sqrt(np.diag(cov))
    statistic = coef / std_error
    p_value = 2 * stats.t.sf(np.abs(statistic), df_resid)
    terms = ["(Intercept)", *predictors]
    table = pd.DataFrame(
        {
            "term": terms,
            "estimate": coef,
            "std_error": std_error,
            "statistic": statistic,
            "p_value": p_value,
        }
    )
    fitted = data.copy()
    fitted["fitted_residual_bias"] = fitted_values
    fitted["unexplained_residual"] = residual
    return table, fitted


def adjustment_preview(
    adjusted: pd.DataFrame, lookup: pd.DataFrame, n: int = 5
) -> pd.DataFrame:
    return (
        adjusted.merge(lookup, on=["origin", "destination"], how="left", sort=False)
        .loc[
            :,
            [
                "origin",
                "destination",
                "flow",
                "flow_adj",
                "flow_benchmark",
            ],
        ]
        .rename(columns={"flow": "flow_mpd_raw", "flow_adj": "flow_mpd_adjusted"})
        .head(n)
        .reset_index(drop=True)
    )


def attr_frame(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key in keys:
        value = frame.attrs.get(key)
        if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
            value = value[0] if len(value) else np.nan
        if isinstance(value, np.generic):
            value = value.item()
        rows.append({"attribute": key, "value": value})
    return pd.DataFrame(rows)


def raking_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_total": [frame["flow"].sum()],
            "adjusted_total": [frame["flow_adj"].sum()],
            "median_weight": [frame["weight_ipf"].median()],
            "ipf_converged": [frame.attrs.get("ipf_converged")],
            "ipf_iterations": [frame.attrs.get("ipf_iterations")],
        }
    )


def adjustment_set_summary(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method, frame in results.items():
        rows.append(
            {
                "method": method,
                "engine": frame.attrs.get("model_engine", "direct_adjustment"),
                "rows": len(frame),
                "adjusted_total": frame["flow_adj"].sum(),
            }
        )
    return pd.DataFrame(rows)


def object_counts(objects: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "object": list(objects.keys()),
            "rows": [len(frame) for frame in objects.values()],
            "columns": [len(frame.columns) for frame in objects.values()],
        }
    )


def combine_overall(
    results: dict[str, dict[str, Any]], methods: list[str]
) -> pd.DataFrame:
    rows = []
    for method in methods:
        summary = results[method].get("summary")
        if isinstance(summary, pd.DataFrame):
            frame = summary.copy()
        else:
            frame = flatten_overall_method_result(results[method])
        frame["method"] = method
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def combine_nested(
    results: dict[str, dict[str, pd.DataFrame]], methods: list[str], key: str
) -> pd.DataFrame:
    frames = []
    for method in methods:
        frame = results[method][key].copy()
        frame = frame.drop(columns=["method"], errors="ignore")
        frame["method"] = method
        frame = frame.loc[
            :, ["method", *[col for col in frame.columns if col != "method"]]
        ]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def method_order_head(frame: pd.DataFrame, methods: list[str], n: int) -> pd.DataFrame:
    frames = []
    for method in methods:
        frames.append(frame.loc[frame["method"] == method].head(n))
    return pd.concat(frames, ignore_index=True)


def nearest_neighbors(distance: pd.DataFrame, areas: pd.Series, k: int) -> pd.DataFrame:
    area_list = list(map(str, areas.tolist()))
    rows = []
    candidate = distance.loc[
        distance["origin"].isin(area_list)
        & distance["destination"].isin(area_list)
        & (distance["origin"] != distance["destination"])
    ].copy()
    candidate = candidate.sort_values(["origin", "distance_km", "destination"])
    for area in area_list:
        for _, row in candidate.loc[candidate["origin"] == area].head(k).iterrows():
            rows.append({"area": area, "neighbor": row["destination"]})
    return pd.DataFrame(rows)


def neighbor_summary(neighbors: pd.DataFrame) -> pd.DataFrame:
    counts = neighbors.groupby("area", sort=True).size().reset_index()
    counts.columns = ["area", "neighbors"]
    return pd.DataFrame(
        {
            "areas": [counts["area"].nunique()],
            "links": [len(neighbors)],
            "min_neighbors": [counts["neighbors"].min()],
            "max_neighbors": [counts["neighbors"].max()],
        }
    )


def read_extdata(repo_root: Path, filename: str) -> pd.DataFrame:
    path = resolve_debiasr_root(repo_root) / "inst" / "extdata" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing debiasR extdata file: {path}")
    out = pd.read_csv(path, true_values=["TRUE"], false_values=["FALSE"])
    for column in out.columns:
        non_missing = out[column].dropna()
        if not non_missing.empty and set(non_missing.map(type)) <= {bool, np.bool_}:
            out[column] = out[column].astype("boolean")
    return out


def bayesian_backend_contract() -> pd.DataFrame:
    """Describe implemented Bayesian runtimes without executing MCMC."""

    return pd.DataFrame(
        {
            "scope": ["nonlatent", "latent_two_level"],
            "status": ["implemented", "implemented"],
            "implementation": [
                "Bambi/PyMC",
                "same debiasR Stan program via CmdStanPy",
            ],
            "model_families": [
                "poisson|negbin|zip|zinb",
                "poisson|negbin",
            ],
            "evidence_scope": [
                "Static Python backend inventory; this article harness does not "
                "execute MCMC. Controlled Bayesian parity evidence: "
                "docs/reports/bayesian_checks.json.",
                "Static Python backend inventory and pinned Stan-source identity; "
                "this article harness does not execute MCMC. Controlled Bayesian "
                "parity evidence: docs/reports/bayesian_checks.json.",
            ],
        }
    )


def write_csv(out_dir: Path, name: str, frame: pd.DataFrame) -> None:
    raw_frame = frame.reset_index(drop=True)
    display_frame = stable_article_display(raw_frame, name)
    path = out_dir / name
    write_frame_csv(
        canonicalize_display_frame(display_frame),
        path,
        "%.10g",
    )
    write_json(table_schema(raw_frame), schema_path_for(path))
    if RAW_DIR is None:
        raise RuntimeError("Article output directory has not been set.")
    write_frame_csv(raw_frame, RAW_DIR / name, "%.17g")


def stable_article_display(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    rounded_columns = ARTICLE_STABLE_DISPLAY_ROUNDING.get(name)
    if not rounded_columns:
        return frame
    display = frame.copy()
    for column, digits in rounded_columns.items():
        if column in display.columns:
            display[column] = display[column].round(digits)
    return display


def write_page_meta(out_dir: Path, page: str, scope: str) -> None:
    write_json(
        {
            "page": page,
            "source_url": ARTICLE_URLS[page],
            "scope": scope,
            "evidence": (
                "R and Python calls executed without MCMC, precomputed Bayesian "
                "file-ingestion checks, and a static Python backend inventory"
            ),
            "bayesian_execution": (
                "The article harness does not execute MCMC. Separate posterior-"
                "comparison evidence for the controlled Bayesian fixtures is "
                "recorded in docs/reports/bayesian_checks.json"
            ),
        },
        out_dir / f"{page}.meta.json",
    )


def resolve_debiasr_root(repo_root: Path) -> Path:
    configured = os.environ.get("DEBIASR_ROOT")
    if not configured:
        raise ValueError("Set DEBIASR_ROOT to the pinned R checkout.")
    path = Path(configured)
    return path.resolve()


def article_tolerance_for(name: str) -> float:
    if name == "v06-adjusting-biases.adjust_all_methods.summary.csv":
        return 1e-6
    if name == "v07-validation.level1.overall_metrics.csv":
        return 1e-5
    if name.startswith("v07-validation.level2."):
        return 2e-4
    if name == "v09-advanced-bayesian-adjustment.coverage_offset_frequentist.head.csv":
        return 1e-5
    if name == "v09-advanced-bayesian-adjustment.reduced_form_frequentist.head.csv":
        return 2.0
    if "coefficient_table" in name:
        return 1e-8
    if "frequentist" in name:
        return 1e-6
    if "validation" in name:
        return 1e-6
    return 1e-8


def main() -> int:
    global DISPLAY_DIR, RAW_DIR
    parser = argparse.ArgumentParser(
        description="Re-run the Census article comparisons against debiasR."
    )
    parser.add_argument("--r-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = args.r_root.resolve()
    os.environ["DEBIASR_ROOT"] = str(reference)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Each run gets its own directory; existing results remain available.
    import tempfile

    run = Path(tempfile.mkdtemp(prefix="article-", dir=output))
    r_display = run / "r"
    r_raw = run / "r-raw"
    DISPLAY_DIR = run / "python"
    RAW_DIR = run / "python-raw"
    for directory in [r_display, r_raw, DISPLAY_DIR, RAW_DIR]:
        directory.mkdir()
    subprocess.run(
        [
            "Rscript",
            str(Path(__file__).with_suffix(".R")),
            str(reference),
            str(r_display),
            str(r_raw),
        ],
        check=True,
    )
    write_article_outputs(Path(__file__).resolve().parents[1], DISPLAY_DIR)
    records = []
    for name in ARTICLE_CSV_NAMES:
        left = pd.read_csv(r_raw / name, keep_default_na=False, na_values=["NA"])
        right = pd.read_csv(
            RAW_DIR / name, keep_default_na=False, na_values=["__DEBIASPY_NA__"]
        )
        tolerance = article_tolerance_for(name)
        record = {
            "name": name,
            "kind": "csv",
            "status": "pass",
            "tolerance": tolerance,
            "rows_r": len(left),
            "rows_python": len(right),
            "message": "",
        }
        deltas = []
        if left.shape == right.shape and list(left.columns) == list(right.columns):
            for column in left:
                if (
                    is_numeric_dtype(left[column])
                    and is_numeric_dtype(right[column])
                    and not is_bool_dtype(left[column])
                    and not is_bool_dtype(right[column])
                ):
                    a, b = left[column].to_numpy(float), right[column].to_numpy(float)
                    finite = np.isfinite(a) & np.isfinite(b)
                    if finite.any():
                        deltas.append(float(np.max(np.abs(a[finite] - b[finite]))))
        record["max_abs_delta"] = max(deltas) if deltas else None
        try:
            assert_frame_equal(left, right, check_dtype=False, atol=tolerance, rtol=0)
        except AssertionError as exc:
            record.update(status="fail", message=str(exc).strip())
        records.append(record)
    for name in ARTICLE_JSON_NAMES:
        equal = json.loads((r_display / name).read_text()) == json.loads(
            (DISPLAY_DIR / name).read_text()
        )
        records.append(
            {
                "name": name,
                "kind": "json",
                "status": "pass" if equal else "fail",
                "message": "" if equal else "JSON values differ",
            }
        )
    expected = {"index.public_exports.csv", "index.public_exports.schema.json"}
    failures = [record for record in records if record["status"] == "fail"]
    commit = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    python_commit = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    changes = subprocess.check_output(
        [
            "git",
            "-C",
            str(Path(__file__).resolve().parents[1]),
            "status",
            "--porcelain",
        ],
        text=True,
    ).splitlines()
    write_json(
        {
            "working_tree_changes": changes,
            "reference_commit": commit,
            "integration_commit": python_commit,
            "checks": records,
            "expected_differences": sorted(expected),
            "scope": (
                "Fresh R/Python non-MCMC article calls; "
                "Bayesian extdata are reference fixtures."
            ),
        },
        run / "article_comparison.json",
    )
    print(
        f"{len(records) - len(failures)}/{len(records)} comparisons passed; "
        f"results: {run}"
    )
    for record in failures:
        print(f"{record['name']}: {record['message'].splitlines()[0]}")
    # Scientific differences fail; the two documented API differences are expected.
    return 0 if {record["name"] for record in failures} == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
