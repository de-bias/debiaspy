# pyright: reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false
import builtins
from typing import cast

import numpy as np
import pandas as pd
import pytest

from debiaspy import adjust_all_methods, adjust_multilevel_bayes
from debiaspy._exceptions import DebiasSchemaError
from debiaspy._frequentist import _fit_count_model
from debiaspy._multilevel_inputs import _random_intercept_group_col


@pytest.fixture(scope="module")
def example_data() -> dict[str, pd.DataFrame]:
    # Synthetic fixtures belong in tests, never in the public data loader.
    origins = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    destinations = list("ABC") * 3
    mpd = pd.DataFrame(
        {
            "origin": origins,
            "destination": destinations,
            "flow": [10, 20, 5, 30, 40, 10, 15, 25, 35],
            "mpd_source": "lad_mpd",
            "mpd_observed": True,
            "mpd_zero_filled": False,
            "mpd_row_status": "observed",
        }
    )
    benchmark = pd.DataFrame(
        {
            "origin": origins,
            "destination": destinations,
            "flow": [100, 180, 60, 120, 220, 80, 90, 140, 160],
        }
    )
    coverage = pd.DataFrame(
        {
            "origin": list("ABC"),
            "destination": list("ABC"),
            "population": [340, 420, 390],
            "user_count": [35, 80, 75],
            "mpd_source": "lad_mpd",
        }
    )
    covariates = pd.DataFrame(
        {
            "area": list("ABC"),
            "population": [340, 420, 390],
            "income_norm": [1 / 24, 1, 0],
        }
    )
    distance = pd.DataFrame(
        {
            "origin": origins,
            "destination": destinations,
            "distance_km": [1, 2, 3, 2, 1, 2, 3, 2, 1],
        }
    )
    return {
        "mpd_od": mpd,
        "benchmark_od": benchmark,
        "coverage": coverage,
        "covariates": covariates,
        "distance": distance,
    }


def _supported_args(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    return {
        "mpd_od_df": data["mpd_od"],
        "coverage_df": data["coverage"],
        "covariates_df": data["covariates"],
        "distance_df": data["distance"],
        "model_engine": "frequentist",
        "model_family": "poisson",
        "scenario": "s1",
        "repeated_observation": "none",
        "random_intercept": "none",
    }


def test_multilevel_collapses_exact_distance_duplicates_without_multiplying_rows(
    example_data: dict[str, pd.DataFrame],
):
    args = _supported_args(example_data)
    args["distance_df"] = pd.concat(
        [example_data["distance"], example_data["distance"].iloc[[0]]],
        ignore_index=True,
    )

    actual = adjust_multilevel_bayes(**args)

    assert len(actual) == len(example_data["mpd_od"])


def test_multilevel_rejects_conflicting_distance_coverage_and_covariate_lookups(
    example_data: dict[str, pd.DataFrame],
):
    distance_conflict = example_data["distance"].iloc[[0]].copy()
    distance_conflict["distance_km"] += 1
    coverage_conflict = example_data["coverage"].iloc[[0]].copy()
    coverage_conflict["population"] += 1
    covariate_conflict = example_data["covariates"].iloc[[0]].copy()
    covariate_column = next(
        column
        for column in covariate_conflict.columns
        if column != "area"
        and pd.api.types.is_numeric_dtype(covariate_conflict[column])
    )
    covariate_conflict[covariate_column] += 1

    conflicting_inputs = (
        (
            "distance_df",
            pd.concat([example_data["distance"], distance_conflict], ignore_index=True),
            "distance_df.*conflicting",
        ),
        (
            "coverage_df",
            pd.concat([example_data["coverage"], coverage_conflict], ignore_index=True),
            "coverage_df.*conflicting",
        ),
        (
            "covariates_df",
            pd.concat(
                [example_data["covariates"], covariate_conflict], ignore_index=True
            ),
            "covariates_df.*conflicting",
        ),
    )
    for argument, value, message in conflicting_inputs:
        args = _supported_args(example_data)
        args[argument] = value
        with pytest.raises(DebiasSchemaError, match=message):
            adjust_multilevel_bayes(**args)


@pytest.mark.parametrize(
    "reserved",
    [
        "row_id",
        "flow_adj",
        "flow_mpd_pred",
        "flow_true_pred",
        "observation_probability",
        "log_observation_probability",
        "model_fit_status",
        "prediction_scope",
    ],
)
def test_multilevel_rejects_reserved_output_columns(
    example_data: dict[str, pd.DataFrame], reserved: str
):
    args = _supported_args(example_data)
    args["mpd_od_df"] = example_data["mpd_od"].assign(**{reserved: 1})

    with pytest.raises(DebiasSchemaError, match="reserved multilevel output"):
        adjust_multilevel_bayes(**args)


@pytest.mark.parametrize(
    ("frame", "column", "value"),
    [
        ("mpd_od_df", "flow", -1.0),
        ("mpd_od_df", "flow", np.inf),
        ("coverage_df", "population", 0.0),
        ("coverage_df", "population", np.inf),
        ("coverage_df", "user_count", -1.0),
        ("distance_df", "distance_km", -1.0),
        ("distance_df", "distance_km", np.inf),
    ],
)
def test_multilevel_rejects_invalid_count_and_distance_inputs(
    example_data: dict[str, pd.DataFrame],
    frame: str,
    column: str,
    value: float,
):
    args = _supported_args(example_data)
    invalid = cast(pd.DataFrame, args[frame]).copy()
    invalid[column] = invalid[column].astype(object)
    invalid.loc[invalid.index[0], column] = value
    args[frame] = invalid

    with pytest.raises(DebiasSchemaError):
        adjust_multilevel_bayes(**args)


def test_repeated_time_coverage_may_have_time_specific_population() -> None:
    fixture = _repeated_fixture(("alpha",), ("t1", "t2"))
    fixture["coverage"].loc[fixture["coverage"]["period_id"] == "t2", "population"] += (
        10
    )

    actual = adjust_multilevel_bayes(
        fixture["mpd_od"],
        fixture["coverage"],
        fixture["covariates"],
        fixture["distance"],
        model_engine="frequentist",
        scenario="s2",
        source_col="provider_id",
        time_col="period_id",
        random_intercept="none",
    )

    assert len(actual) == len(fixture["mpd_od"])
    assert bool(np.isfinite(actual["flow_adj"]).all())


def _scenario_fixture(
    n_areas: int,
    sources: tuple[str, ...] = ("operator_a",),
    periods: tuple[str, ...] = ("2021_q1",),
) -> dict[str, pd.DataFrame]:
    """Python construction of the pinned-R multilevel probe fixture."""
    areas = [f"E020{index:05d}" for index in range(1, n_areas + 1)]
    mpd_rows = [
        (origin, destination, source, period)
        for period in periods
        for source in sources
        for destination in areas
        for origin in areas
    ]
    mpd_od = pd.DataFrame(
        mpd_rows,
        columns=["origin", "destination", "provider_id", "period_id"],
    )
    coverage_rows = [
        (origin, source, period, 1000 + 25 * area_index)
        for period in periods
        for source in sources
        for area_index, origin in enumerate(areas, start=1)
    ]
    coverage = pd.DataFrame(
        coverage_rows,
        columns=["origin", "provider_id", "period_id", "population"],
    )
    coverage["user_count"] = 0
    covariates = pd.DataFrame(
        {
            "area": areas,
            "income_norm": np.linspace(0.15, 0.85, n_areas),
            "rural_pct": np.linspace(0.75, 0.25, n_areas),
            "deprivation_score": np.linspace(1.2, 3.6, n_areas),
            "population": 1000 + 25 * np.arange(1, n_areas + 1),
        }
    )
    distance = pd.DataFrame(
        [
            (
                origin,
                destination,
                abs(areas.index(origin) - areas.index(destination)) + 1,
            )
            for destination in areas
            for origin in areas
        ],
        columns=["origin", "destination", "distance_km"],
    )
    return {
        "mpd_od": mpd_od,
        "coverage": coverage,
        "covariates": covariates,
        "distance": distance,
    }


def _fixed_nb_fixture() -> dict[str, pd.DataFrame]:
    fixture = _scenario_fixture(8)
    coverage = fixture["coverage"]
    area_index = coverage["origin"].map(
        {area: index for index, area in enumerate(coverage["origin"].unique(), 1)}
    )
    coverage_rate = 0.08 + 0.05 * area_index.to_numpy()
    coverage["user_count"] = np.rint(
        coverage["population"].to_numpy() * coverage_rate
    ).astype(int)

    mpd_od = fixture["mpd_od"]
    distance = fixture["distance"].set_index(["origin", "destination"])
    distance_values = distance["distance_km"].reindex(
        pd.MultiIndex.from_frame(mpd_od[["origin", "destination"]])
    )
    coverage_by_origin = coverage.set_index("origin")
    bias = (
        1
        - (coverage_by_origin["user_count"] / coverage_by_origin["population"])
        .reindex(mpd_od["origin"])
        .to_numpy()
    )
    eta = 3.2 - 0.4 * np.log(distance_values.to_numpy()) - 1.1 * bias
    multipliers = np.resize(
        np.array([0.25, 0.5, 0.8, 1.0, 1.3, 1.7, 2.2, 0.65]),
        len(mpd_od),
    )
    mpd_od["flow"] = np.rint(np.exp(eta) * multipliers).astype(int)
    return fixture


def _repeated_fixture(
    sources: tuple[str, ...], periods: tuple[str, ...]
) -> dict[str, pd.DataFrame]:
    fixture = _scenario_fixture(6, sources=sources, periods=periods)
    coverage = fixture["coverage"]
    areas = coverage["origin"].unique().tolist()
    area_index = (
        coverage["origin"]
        .map({area: index for index, area in enumerate(areas, 1)})
        .to_numpy()
    )
    source_index = (
        coverage["provider_id"]
        .map({source: index for index, source in enumerate(sorted(sources), 1)})
        .to_numpy()
    )
    time_index = (
        coverage["period_id"]
        .map({period: index for index, period in enumerate(sorted(periods), 1)})
        .to_numpy()
    )
    rate = (
        0.08
        + 0.055 * area_index
        + 0.025 * (source_index - 1)
        + 0.012 * (time_index - 1)
    )
    coverage["user_count"] = np.rint(coverage["population"].to_numpy() * rate).astype(
        int
    )

    mpd_od = fixture["mpd_od"]
    distance = fixture["distance"].set_index(["origin", "destination"])
    distance_values = (
        distance["distance_km"]
        .reindex(pd.MultiIndex.from_frame(mpd_od[["origin", "destination"]]))
        .to_numpy()
    )
    coverage_key = ["origin", "provider_id", "period_id"]
    keyed_coverage = coverage.set_index(coverage_key)
    mpd_keys = pd.MultiIndex.from_frame(mpd_od[coverage_key])
    bias = (
        1
        - (keyed_coverage["user_count"] / keyed_coverage["population"])
        .reindex(mpd_keys)
        .to_numpy()
    )
    eta = (
        3.1
        - 0.35 * np.log(distance_values)
        - 0.9 * bias
        + 0.22 * (mpd_od["provider_id"] == "zeta").to_numpy()
        - 0.14 * (mpd_od["period_id"] == "t2").to_numpy()
    )
    mpd_od["flow"] = np.rint(
        np.exp(eta) * np.resize(np.array([0.85, 1.05, 1.15, 0.95]), len(mpd_od))
    ).astype(int)
    return fixture


def _random_intercept_fixture(model_family: str) -> dict[str, pd.DataFrame]:
    fixture = _scenario_fixture(10)
    coverage = fixture["coverage"]
    areas = coverage["origin"].unique().tolist()
    area_index = np.arange(1, len(areas) + 1)
    coverage["user_count"] = np.rint(
        coverage["population"].to_numpy() * (0.09 + 0.038 * area_index)
    ).astype(int)

    mpd_od = fixture["mpd_od"]
    origin_index = (
        mpd_od["origin"]
        .map({area: index for index, area in enumerate(areas, 1)})
        .to_numpy()
    )
    destination_index = (
        mpd_od["destination"]
        .map({area: index for index, area in enumerate(areas, 1)})
        .to_numpy()
    )
    distance = fixture["distance"].set_index(["origin", "destination"])
    distance_values = (
        distance["distance_km"]
        .reindex(pd.MultiIndex.from_frame(mpd_od[["origin", "destination"]]))
        .to_numpy()
    )
    coverage_by_origin = coverage.set_index("origin")
    bias = (
        1
        - (coverage_by_origin["user_count"] / coverage_by_origin["population"])
        .reindex(mpd_od["origin"])
        .to_numpy()
    )
    group_effect = np.array(
        [-0.5, 0.3, -0.2, 0.45, 0.05, -0.35, 0.55, -0.1, 0.25, -0.4]
    )[origin_index - 1]
    eta = 3.0 - 0.32 * np.log(distance_values) - 0.75 * bias + group_effect
    multipliers = (
        np.array([0.9, 1.05, 1.1, 0.95, 1.02, 0.98, 1.08, 0.92, 1.03, 0.97])
        if model_family == "poisson"
        else np.array([0.25, 0.5, 0.75, 1.0, 1.3, 1.65, 2.2, 0.6, 1.15, 1.5])
    )
    multiplier_index = (3 * origin_index + 7 * destination_index) % 10
    mpd_od["flow"] = np.rint(np.exp(eta) * multipliers[multiplier_index]).astype(int)
    return fixture


def test_supported_s1_poisson_has_finite_output_and_unique_columns(
    example_data: dict[str, pd.DataFrame],
):
    actual = adjust_multilevel_bayes(
        **_supported_args(example_data),
        keep_cols=["origin", "log_distance", "log_distance", "flow"],
    )

    assert bool(np.isfinite(actual["flow_adj"]).all())
    assert actual.columns.is_unique
    assert actual.columns.tolist().count("origin") == 1
    assert actual.columns.tolist().count("log_distance") == 1
    assert actual.columns.tolist().count("flow") == 1
    assert actual.attrs["scenario"] == "s1"
    assert actual.attrs["repeated_observation"] == "none"
    assert actual.attrs["random_intercept"] == "none"


def test_rank_deficient_default_formula_matches_pinned_r_alias_policy():
    mpd = pd.DataFrame(
        {
            "origin": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "destination": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "flow": [10, 20, 5, 30, 40, 10, 15, 25, 35],
        }
    )
    coverage = pd.DataFrame(
        {
            "origin": ["A", "B", "C"],
            "population": [1000, 2000, 1500],
            "user_count": [100, 500, 300],
        }
    )
    covariates = pd.DataFrame(
        {
            "area": ["A", "B", "C"],
            "income_norm": [0.2, 0.6, 1.0],
            "rural_pct": [0.7, 0.3, 0.5],
        }
    )
    areas = ["A", "B", "C"]
    distance = pd.DataFrame(
        [
            (
                origin,
                destination,
                abs(areas.index(origin) - areas.index(destination)) + 1,
            )
            for destination in areas
            for origin in areas
        ],
        columns=["origin", "destination", "distance_km"],
    )

    actual = adjust_multilevel_bayes(
        mpd,
        coverage,
        covariates,
        distance,
        model_engine="frequentist",
        scenario="s1",
        random_intercept="none",
    )

    coefficients = actual.attrs["coefficients"]
    assert coefficients["term"].tolist() == [
        "(Intercept)",
        "income_o",
        "income_d",
        "log_distance",
        "bias_e_origin",
        "log_pop_d",
    ]
    np.testing.assert_allclose(
        coefficients["mean"].to_numpy(),
        [0.4245922, 0.6262621, -0.7306012, -0.6736359, -3.2117504, 0.7634193],
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual["flow_adj"].to_numpy(),
        [
            292.141529891335,
            232.114553612933,
            105.872188627077,
            235.288960430334,
            475.641623488299,
            178.729463953881,
            230.023931472779,
            383.078787769700,
            366.246627266549,
        ],
        rtol=2e-9,
        atol=2e-8,
    )
    assert actual.attrs["diagnostics"]["aliased_fixed_effects"] == ["log_pop_o"]


def test_adjust_all_methods_preserves_supported_default_multilevel_path(
    example_data: dict[str, pd.DataFrame],
):
    actual = adjust_all_methods(
        example_data["mpd_od"],
        example_data["coverage"],
        benchmark_od_df=example_data["benchmark_od"],
        covariates_df=example_data["covariates"],
        distance_df=example_data["distance"],
        methods=["multilevel_bayes"],
        covariate_col="income_norm",
        multilevel_engine="frequentist",
    )["multilevel_bayes"]

    assert bool(np.isfinite(actual["flow_adj"]).all())
    assert actual.attrs["model_family"] == "poisson"
    assert actual.attrs["random_intercept"] == "none"


@pytest.mark.parametrize(
    ("selector", "group_col"),
    [
        ("origin", "origin"),
        ("destination", "destination"),
        ("od", "od_id"),
        ("source", "mpd_source"),
        ("time", "mpd_time"),
        ("source_time", "mpd_source_time"),
        ("none", None),
    ],
)
def test_every_declared_random_intercept_selector_has_an_explicit_group_contract(
    selector: str, group_col: str | None
):
    assert _random_intercept_group_col(selector) == group_col


def test_additive_formula_uses_none_for_unsupplied_default_random_intercept(
    example_data: dict[str, pd.DataFrame],
):
    args = _supported_args(example_data)
    args.pop("random_intercept")

    actual = adjust_multilevel_bayes(
        **args,
        formula="flow ~ log_distance + bias_e_origin",
    )

    assert bool(np.isfinite(actual["flow_adj"]).all())
    assert actual.attrs["random_intercept"] == "none"


def test_additive_formula_preserves_explicit_default_selector_metadata(
    example_data: dict[str, pd.DataFrame],
):
    args = _supported_args(example_data)
    args.pop("random_intercept")
    formula = "flow ~ log_distance + bias_e_origin"

    omitted = adjust_multilevel_bayes(**args, formula=formula)
    explicit = adjust_multilevel_bayes(
        **args,
        formula=formula,
        random_intercept="origin",
    )

    np.testing.assert_allclose(explicit["flow_adj"], omitted["flow_adj"])
    assert omitted.attrs["random_intercept"] == "none"
    assert explicit.attrs["random_intercept"] == "origin"
    assert explicit.attrs["random_intercept_group"] is None
    assert explicit.attrs["result_metadata"]["effective_random_intercept"] == "none"


def test_random_effect_formula_is_authoritative_but_preserves_caller_metadata(
    example_data: dict[str, pd.DataFrame],
):
    args = _supported_args(example_data)
    args.pop("random_intercept")
    formula = "flow ~ log_distance + bias_e_origin + (1 | origin)"

    omitted = adjust_multilevel_bayes(**args, formula=formula, backend="gpboost")
    explicit = adjust_multilevel_bayes(
        **args,
        formula=formula,
        random_intercept="destination",
        backend="gpboost",
    )

    # The external GPBoost optimizer can differ at sub-ppm precision between
    # equivalent parallel fits. This test concerns formula authority and uses
    # the package's numerical-equivalence standard rather than byte identity.
    np.testing.assert_allclose(
        explicit["flow_adj"],
        omitted["flow_adj"],
        rtol=1e-6,
        atol=1e-8,
    )
    assert omitted.attrs["random_intercept"] == "none"
    assert explicit.attrs["random_intercept"] == "destination"
    assert omitted.attrs["random_intercept_group"] == "origin"
    assert explicit.attrs["random_intercept_group"] == "origin"


def test_log_formula_matches_pinned_r_design_and_predictions(
    example_data: dict[str, pd.DataFrame],
):
    actual = adjust_multilevel_bayes(
        **_supported_args(example_data),
        formula="flow ~ log(distance_km) + bias_e_origin",
    )

    coefficients = actual.attrs["coefficients"].set_index("term")["mean"]
    assert coefficients.index.tolist() == [
        "(Intercept)",
        "log(distance_km)",
        "bias_e_origin",
    ]
    np.testing.assert_allclose(
        coefficients.to_numpy(),
        [10.5185431, -0.6770352, -8.5894411],
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual["flow_adj"].to_numpy(),
        [
            36995.188347,
            23138.613862,
            17583.987137,
            23138.613862,
            36995.188347,
            23138.613862,
            17583.987137,
            23138.613862,
            36995.188347,
        ],
        rtol=1e-7,
        atol=1e-7,
    )
    assert actual.attrs["formula"] == ("flow ~ log(distance_km) + bias_e_origin")


def test_zero_intercept_and_split_transforms_match_pinned_r_semantics(
    example_data: dict[str, pd.DataFrame],
):
    combined = adjust_multilevel_bayes(
        **_supported_args(example_data),
        formula="flow ~ 0 + log(distance_km) + bias_e_origin",
    )
    split = adjust_multilevel_bayes(
        **_supported_args(example_data),
        mobility_formula="~ 0 + log(distance_km)",
        bias_formula="~ bias_e_origin",
    )

    coefficients = combined.attrs["coefficients"].set_index("term")["mean"]
    assert coefficients.index.tolist() == [
        "log(distance_km)",
        "bias_e_origin",
    ]
    np.testing.assert_allclose(
        coefficients.to_numpy(),
        [-0.6636593, 3.9871737],
        rtol=1e-7,
        atol=1e-7,
    )
    expected = [
        1.0,
        0.6312751,
        0.4823409,
        0.6312751,
        1.0,
        0.6312751,
        0.4823409,
        0.6312751,
        1.0,
    ]
    np.testing.assert_allclose(
        combined["flow_adj"].to_numpy(), expected, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        split["flow_adj"].to_numpy(), expected, rtol=1e-7, atol=1e-7
    )
    assert split.attrs["formula"] == ("flow ~ 0 + log(distance_km) + bias_e_origin")


def test_missing_formula_variable_rows_are_excluded_like_pinned_r(
    example_data: dict[str, pd.DataFrame],
):
    mpd = example_data["mpd_od"].copy()
    mpd["trip_covariate"] = [0.2, 0.7, np.nan, 1.1, 1.3, 1.8, 2.0, 2.4, 2.9]
    args = _supported_args(example_data)
    args["mpd_od_df"] = mpd

    with pytest.warns(UserWarning, match="1 row.*formula.*missing"):
        actual = adjust_multilevel_bayes(
            **args,
            formula=("flow ~ log(distance_km) + trip_covariate + bias_e_origin"),
        )

    assert actual.attrs["result_metadata"]["n_fit_rows"] == 8
    assert actual.attrs["result_metadata"]["n_prediction_rows"] == 8
    assert actual["model_fit_status"].tolist() == [
        "fit",
        "fit",
        "excluded",
        "fit",
        "fit",
        "fit",
        "fit",
        "fit",
        "fit",
    ]
    expected = np.array(
        [
            26268.163767,
            16938.965099,
            np.nan,
            16632.609168,
            24982.268180,
            16109.758282,
            12515.875551,
            15674.701830,
            23223.405416,
        ]
    )
    np.testing.assert_allclose(
        actual["flow_adj"].to_numpy(), expected, rtol=1e-7, atol=1e-7
    )


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        (
            "flow ~ log_distance + (1 + log_distance | origin)",
            "random slopes",
        ),
        (
            "flow ~ log_distance + (1 | origin) + (1 | destination)",
            "multiple random-effect terms",
        ),
        ("flow ~ log_distance + (1 | origin/destination)", "plain grouping"),
    ],
)
def test_non_equivalent_random_effect_structures_fail_closed(
    example_data: dict[str, pd.DataFrame],
    formula: str,
    message: str,
):
    with pytest.raises(NotImplementedError, match=message):
        adjust_multilevel_bayes(
            **_supported_args(example_data),
            formula=formula,
        )


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("flow ~ distance_km / bias_e_origin", "interactions"),
        ("flow ~ offset(log_distance)", "user-specified offset"),
        ("flow ~ sqrt(distance_km)", "unsupported transform"),
    ],
)
def test_unsupported_formula_constructs_fail_clearly(
    example_data: dict[str, pd.DataFrame], formula: str, message: str
):
    with pytest.raises(DebiasSchemaError, match=message):
        adjust_multilevel_bayes(
            **_supported_args(example_data),
            formula=formula,
        )


def test_fixed_negbin_nb2_matches_pinned_mass_glm_nb_probe():
    fixture = _fixed_nb_fixture()
    actual = adjust_multilevel_bayes(
        fixture["mpd_od"],
        fixture["coverage"],
        fixture["covariates"],
        fixture["distance"],
        model_engine="frequentist",
        model_family="negbin",
        scenario="s1",
        random_intercept="none",
        source_col="provider_id",
        time_col="period_id",
        formula="flow ~ log_distance + bias_e_origin",
    )

    coefficients = actual.attrs["coefficients"].set_index("term")
    np.testing.assert_allclose(
        coefficients["mean"].to_numpy(),
        [5.9188592, -0.5049488, -4.8964454],
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        coefficients["sd"].to_numpy(),
        [0.39769959, 0.09939745, 0.56081904],
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        actual["flow_adj"].head(12).to_numpy(),
        [
            371.9871250412361,
            262.1338830221019,
            213.6023780416201,
            184.7219110624536,
            165.0379475808964,
            150.5224697564408,
            139.2504578008620,
            130.1708273389746,
            262.1338830221019,
            371.9871250412361,
            262.1338830221019,
            213.6023780416201,
        ],
        rtol=1e-9,
        atol=1e-9,
    )
    diagnostics = actual.attrs["diagnostics"]
    assert diagnostics["backend"] == "statsmodels_discrete_negative_binomial_nb2"
    assert diagnostics["convergence"]["converged"] is True
    assert diagnostics["dispersion"]["shape"] == pytest.approx(10.49985, rel=1e-6)
    assert diagnostics["aic"] == pytest.approx(333.2225, abs=1e-4)
    assert actual.attrs["model_contract"]["conditional_variance"] == (
        "mu + alpha * mu^2"
    )


def test_fixed_negbin_coverage_offset_matches_pinned_r_probe():
    fixture = _fixed_nb_fixture()
    actual = adjust_multilevel_bayes(
        fixture["mpd_od"],
        fixture["coverage"],
        fixture["covariates"],
        fixture["distance"],
        model_engine="frequentist",
        model_family="negbin",
        scenario="s1",
        random_intercept="none",
        source_col="provider_id",
        time_col="period_id",
        formula="flow ~ log_distance",
        target_scale="true_flow",
        observation_model="coverage_offset",
        coverage_scale="origin",
    )

    np.testing.assert_allclose(
        actual.attrs["coefficients"]["mean"].to_numpy(),
        [3.7811454, -0.4687846],
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual["flow_true_pred"].head(8).to_numpy(),
        [
            43.8662560112630,
            31.6965751501449,
            26.2097854538001,
            22.9030915241735,
            20.6283354730997,
            18.9384850645162,
            17.6182030482446,
            16.5491570896823,
        ],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        actual["flow_mpd_pred"],
        actual["flow_true_pred"] * actual["observation_probability"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert actual.attrs["diagnostics"]["dispersion"]["shape"] == pytest.approx(
        14.2105, rel=1e-6
    )


@pytest.mark.parametrize("bias_formula", ["~ 0", "~ bias_e_origin"])
def test_split_coverage_offset_uses_only_the_mobility_formula(
    example_data: dict[str, pd.DataFrame], bias_formula: str
):
    shared = {
        **_supported_args(example_data),
        "target_scale": "true_flow",
        "observation_model": "coverage_offset",
        "coverage_scale": "origin",
    }
    mobility_only = adjust_multilevel_bayes(
        **shared,
        formula="flow ~ log_distance",
    )
    split = adjust_multilevel_bayes(
        **shared,
        mobility_formula="~ log_distance",
        bias_formula=bias_formula,
    )

    assert split.attrs["coefficients"]["term"].tolist() == [
        "(Intercept)",
        "log_distance",
    ]
    assert split.attrs["formula"] == "flow ~ log_distance"
    assert split.attrs["bias_terms"] == []
    pd.testing.assert_frame_equal(split, mobility_only)
    pd.testing.assert_frame_equal(
        split.attrs["coefficients"], mobility_only.attrs["coefficients"]
    )


@pytest.mark.parametrize(
    ("scenario", "sources", "periods", "repeated", "scenario_terms"),
    [
        ("s1", ("alpha",), ("t1",), "none", ()),
        ("s2", ("alpha",), ("t1", "t2"), "time", ("mpd_time",)),
        ("s3", ("alpha", "zeta"), ("t1",), "source", ("mpd_source",)),
        (
            "s4",
            ("alpha", "zeta"),
            ("t1", "t2"),
            "source_time",
            ("mpd_source", "mpd_time"),
        ),
    ],
)
def test_fixed_effect_s1_to_s4_repeated_scenarios_are_supported(
    scenario: str,
    sources: tuple[str, ...],
    periods: tuple[str, ...],
    repeated: str,
    scenario_terms: tuple[str, ...],
):
    fixture = _repeated_fixture(sources, periods)
    formula_terms = ["log_distance", "bias_e_origin", *scenario_terms]
    actual = adjust_multilevel_bayes(
        fixture["mpd_od"],
        fixture["coverage"],
        fixture["covariates"],
        fixture["distance"],
        model_engine="frequentist",
        scenario=scenario,
        source_col="provider_id",
        time_col="period_id",
        random_intercept="none",
        formula="flow ~ " + " + ".join(formula_terms),
    )

    assert actual.attrs["scenario"] == scenario
    assert actual.attrs["repeated_observation"] == repeated
    assert bool(np.isfinite(actual["flow_adj"]).all())
    assert set(scenario_terms).issubset(set(actual.attrs["model_terms"]))


def test_s4_source_time_fixed_effects_match_pinned_r_glm_probe():
    fixture = _repeated_fixture(("zeta", "alpha"), ("t2", "t1"))
    actual = adjust_multilevel_bayes(
        fixture["mpd_od"],
        fixture["coverage"],
        fixture["covariates"],
        fixture["distance"],
        model_engine="frequentist",
        scenario="s4",
        source_col="provider_id",
        time_col="period_id",
        random_intercept="none",
        formula=("flow ~ log_distance + bias_e_origin + mpd_source + mpd_time"),
    )

    coefficients = actual.attrs["coefficients"].set_index("term")["mean"]
    expected = {
        "(Intercept)": 3.0484320,
        "log_distance": -0.3376486,
        "bias_e_origin": -0.8478419,
        "mpd_source[T.zeta]": 0.2247812,
        "mpd_time[T.t2]": -0.1411181,
    }
    for term, value in expected.items():
        assert coefficients[term] == pytest.approx(value, abs=1e-7)
    np.testing.assert_allclose(
        actual["flow_adj"].head(12).to_numpy(),
        [
            22.92195276698864,
            18.13882929756805,
            15.81802599439648,
            14.35380011602466,
            13.31206296082182,
            12.51727879616191,
            18.13882929756805,
            22.92195276698864,
            18.13882929756805,
            15.81802599439648,
            14.35380011602466,
            13.31206296082182,
        ],
        rtol=1e-12,
        atol=1e-12,
    )
    assert actual.attrs["diagnostics"]["aic"] == pytest.approx(604.6526, abs=1e-4)


def test_auto_inferred_s2_structure_uses_time_fixed_effect():
    fixture = _repeated_fixture(("alpha",), ("t1", "t2"))
    actual = adjust_multilevel_bayes(
        fixture["mpd_od"],
        fixture["coverage"],
        fixture["covariates"],
        fixture["distance"],
        model_engine="frequentist",
        scenario="auto",
        repeated_observation="auto",
        source_col="provider_id",
        time_col="period_id",
        random_intercept="none",
        formula="flow ~ log_distance + bias_e_origin + mpd_time",
    )

    assert actual.attrs["scenario"] == "s2"
    assert actual.attrs["repeated_observation"] == "time"
    assert "mpd_time" in actual.attrs["model_terms"]


def test_duplicate_s1_od_observations_follow_r_glm_row_semantics(
    example_data: dict[str, pd.DataFrame],
):
    mpd = pd.concat(
        [example_data["mpd_od"], example_data["mpd_od"].iloc[[0]]],
        ignore_index=True,
    )
    args = _supported_args(example_data)
    args["mpd_od_df"] = mpd

    actual = adjust_multilevel_bayes(**args)

    assert len(actual) == len(mpd)
    assert bool(np.isfinite(actual["flow_adj"]).all())


@pytest.mark.parametrize(
    ("selector", "scenario", "sources", "periods", "expected_group"),
    [
        ("destination", "s1", ("alpha",), ("t1",), "destination"),
        ("od", "s2", ("alpha",), ("t1", "t2"), "od_id"),
        (
            "source",
            "s4",
            ("alpha", "zeta"),
            ("t1", "t2"),
            "mpd_source",
        ),
        ("time", "s4", ("alpha", "zeta"), ("t1", "t2"), "mpd_time"),
        (
            "source_time",
            "s4",
            ("alpha", "zeta"),
            ("t1", "t2"),
            "mpd_source_time",
        ),
    ],
)
def test_gpboost_executes_each_non_origin_random_intercept_selector(
    selector: str,
    scenario: str,
    sources: tuple[str, ...],
    periods: tuple[str, ...],
    expected_group: str,
):
    fixture = _repeated_fixture(sources, periods)
    actual = adjust_multilevel_bayes(
        fixture["mpd_od"],
        fixture["coverage"],
        fixture["covariates"],
        fixture["distance"],
        model_engine="frequentist",
        model_family="poisson",
        scenario=scenario,
        source_col="provider_id",
        time_col="period_id",
        random_intercept=selector,
        backend="gpboost",
    )

    assert actual.attrs["random_intercept"] == selector
    assert actual.attrs["random_intercept_group"] == expected_group
    assert actual.attrs["diagnostics"]["convergence"]["converged"] is True
    assert bool(np.isfinite(actual["flow_adj"]).all())


def test_gpboost_poisson_random_intercept_matches_pinned_lme4_probe():
    fixture = _random_intercept_fixture("poisson")
    actual = adjust_multilevel_bayes(
        fixture["mpd_od"],
        fixture["coverage"],
        fixture["covariates"],
        fixture["distance"],
        model_engine="frequentist",
        model_family="poisson",
        scenario="s1",
        formula=("flow ~ log_distance + bias_e_origin + (1 | origin)"),
        backend="gpboost",
    )

    coefficients = actual.attrs["coefficients"].set_index("term")
    np.testing.assert_allclose(
        coefficients["mean"].to_numpy(),
        [2.9875256, -0.2472166, -0.8467769],
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        coefficients["sd"].to_numpy(),
        [0.73202002, 0.05338784, 1.02984503],
        rtol=0,
        atol=4e-4,
    )
    np.testing.assert_allclose(
        actual["flow_adj"].head(12).to_numpy(),
        [
            19.83653776555039,
            16.71268609392764,
            15.11866940867395,
            14.08077759210780,
            13.32505079332479,
            12.73778614854080,
            12.26149977380589,
            11.86334120584265,
            11.52288606965135,
            11.22662602348034,
            16.71268609392764,
            19.83653776555039,
        ],
        rtol=1e-4,
        atol=1e-4,
    )
    diagnostics = actual.attrs["diagnostics"]
    assert diagnostics["runtime_version"] == "1.6.8"
    assert diagnostics["convergence"]["converged"] is True
    assert diagnostics["log_likelihood"] == pytest.approx(-214.5065, abs=1e-4)
    assert diagnostics["aic"] == pytest.approx(437.0131, abs=1e-4)
    assert diagnostics["random_effect"]["sd"] == pytest.approx(0.3341181, abs=1e-5)
    assert actual.attrs["model_contract"]["counterfactual_prediction"] == (
        "fixed_effects_only_exp(X beta + offset); random effects excluded"
    )
    assert isinstance(actual.attrs["model"], dict)
    # Ordinary pandas operations deep-copy attrs; the stored GPBoost snapshot
    # must therefore remain pointer-free.
    assert bool(np.isfinite(actual["flow_adj"]).all())


def test_gpboost_negbin_random_intercept_matches_declared_lme4_tolerance():
    fixture = _random_intercept_fixture("negbin")
    actual = adjust_multilevel_bayes(
        fixture["mpd_od"],
        fixture["coverage"],
        fixture["covariates"],
        fixture["distance"],
        model_engine="frequentist",
        model_family="negbin",
        scenario="s1",
        formula=("flow ~ log_distance + bias_e_origin + (1 | origin)"),
        backend="gpboost",
    )

    # lme4::glmer.nb and GPBoost optimize independent Laplace
    # approximations. On this pinned fixture their fixed coefficients differ
    # by at most 0.0063 and their fixed-only fitted means by at most 0.73%.
    coefficients = actual.attrs["coefficients"].set_index("term")
    np.testing.assert_allclose(
        coefficients["mean"].to_numpy(),
        [2.6321893, 0.1671216, -1.0238765],
        rtol=0,
        atol=0.0065,
    )
    np.testing.assert_allclose(
        coefficients["sd"].to_numpy(),
        [0.79312926, 0.09392764, 1.10962659],
        rtol=0,
        atol=0.001,
    )
    np.testing.assert_allclose(
        actual["flow_adj"].head(12).to_numpy(),
        [
            13.90417646273357,
            15.61183213906130,
            16.70638627675621,
            17.52921529667570,
            18.19525839879875,
            18.75819822210157,
            19.24772356281733,
            19.68208383104514,
            20.07334581636317,
            20.42992769907962,
            15.61183213906130,
            13.90417646273357,
        ],
        rtol=0.0075,
        atol=0.001,
    )
    diagnostics = actual.attrs["diagnostics"]
    assert diagnostics["log_likelihood"] == pytest.approx(-291.3304, abs=0.005)
    assert diagnostics["aic"] == pytest.approx(592.6608, abs=0.01)
    # The independent Laplace optimizers agree on the random-effect SD to
    # 0.04%; the deterministic GPBoost initialization yields 0.34120.
    assert diagnostics["random_effect"]["sd"] == pytest.approx(0.3413342, abs=2e-4)
    assert diagnostics["dispersion"]["shape"] == pytest.approx(6.44726, rel=0.009)
    assert actual.attrs["model_contract"]["conditional_variance"] == (
        "mu + mu^2 / shape"
    )


def test_gpboost_coverage_offset_prediction_excludes_random_effects(
    example_data: dict[str, pd.DataFrame],
):
    actual = adjust_multilevel_bayes(
        example_data["mpd_od"],
        example_data["coverage"],
        example_data["covariates"],
        example_data["distance"],
        model_engine="frequentist",
        model_family="poisson",
        scenario="s1",
        formula="flow ~ log_distance + (1 | origin)",
        target_scale="true_flow",
        observation_model="coverage_offset",
        coverage_scale="origin",
        backend="gpboost",
    )

    np.testing.assert_allclose(
        actual["flow_mpd_pred"].to_numpy(),
        (actual["flow_true_pred"] * actual["observation_probability"]).to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert actual.attrs["model_contract"]["counterfactual_prediction"] == (
        "fixed_effects_only_exp(X beta + offset); random effects excluded"
    )


def test_explicit_gpboost_backend_requires_a_random_intercept(
    example_data: dict[str, pd.DataFrame],
):
    with pytest.raises(
        DebiasSchemaError,
        match="gpboost.*requires one supported random-intercept term",
    ):
        adjust_multilevel_bayes(
            example_data["mpd_od"],
            example_data["coverage"],
            example_data["covariates"],
            example_data["distance"],
            model_engine="frequentist",
            formula="flow ~ log_distance + bias_e_origin",
            backend="gpboost",
        )


def test_missing_statsmodels_message_describes_base_dependency(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("statsmodels"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    fit_df = pd.DataFrame({"flow": [1.0, 2.0], "x": [0.0, 1.0]})

    with pytest.raises(ImportError) as caught:
        _fit_count_model(fit_df, "flow ~ x", "poisson", "mpd_counterfactual")

    message = str(caught.value)
    assert "base debiaspy dependency" in message
    assert "debiaspy[model]" not in message
