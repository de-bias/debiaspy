from __future__ import annotations

import importlib
import numbers
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from debiaspy._choices import match_choice
from debiaspy._exceptions import DebiasSchemaError


def debiaspy_example_data(
    n_areas: int | float | None = 25,
    data_package: str = "debiaspydata",
    mpd_path: str | Path | None = None,
    census_path: str | Path | None = None,
    include_self_flows: bool = True,
    complete_grid: bool = False,
    fill_missing_flow: float = 0,
    geography: str = "lad",
) -> dict[str, pd.DataFrame]:
    """Load and normalize example mobility and Census benchmark tables.

    The default route reads the installed debiaspydata companion. Explicit
    MPD/Census paths can override the OD counts.
    Source OD files are normalized
    to ``origin``, ``destination``, and ``flow``; ``geography`` selects LAD or
    MSOA area-code validation. Grid completion and missing-flow filling are
    explicit options.

    Returns the analysis bundle of MPD, benchmark, coverage, covariate,
    distance, and supporting tables expected by the public workflows. Missing
    sources, invalid area overlap, and invalid controls raise
    :class:`DebiasSchemaError`. Parity covers the pinned example-data sources
    and audited subset construction, not arbitrary external files.
    """
    try:
        geography = match_choice(geography, ("lad", "msoa"), "geography")
    except ValueError as exc:
        raise DebiasSchemaError(str(exc)) from exc
    for name, value in (
        ("include_self_flows", include_self_flows),
        ("complete_grid", complete_grid),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise DebiasSchemaError(f"`{name}` must be a single boolean value.")
    include_self_flows = bool(include_self_flows)
    complete_grid = bool(complete_grid)
    if n_areas is not None:
        if (
            isinstance(n_areas, (bool, np.bool_))
            or not isinstance(n_areas, numbers.Real)
            or np.isnan(n_areas)
            or np.isneginf(n_areas)
            or (np.isfinite(n_areas) and float(n_areas) != int(n_areas))
        ):
            raise DebiasSchemaError(
                "`n_areas` must be a whole number of at least 2, positive "
                "infinity, or None."
            )
        if np.isfinite(n_areas) and int(n_areas) < 2:
            raise DebiasSchemaError(
                "`n_areas` must be a whole number of at least 2, positive "
                "infinity, or None."
            )
    if (
        isinstance(fill_missing_flow, bool | np.bool_)
        or not isinstance(fill_missing_flow, numbers.Real)
        or not np.isfinite(fill_missing_flow)
        or fill_missing_flow < 0
    ):
        raise DebiasSchemaError(
            "`fill_missing_flow` must be a single finite non-negative number."
        )

    mpd_od = _normalise_mpd_travel_to_work(
        _load_example_source(
            mpd_path,
            f"{geography}_OD_travel2work",
            data_package,
        ),
        include_self_flows,
        geography,
    )
    benchmark_od = _normalise_census_travel_to_work(
        _load_example_source(
            census_path,
            f"census_{geography}_OD_travel2work",
            data_package,
        ),
        include_self_flows,
        geography,
    )
    common_areas = sorted(
        (set(mpd_od["origin"]) | set(mpd_od["destination"]))
        & (set(benchmark_od["origin"]) | set(benchmark_od["destination"]))
    )
    if not common_areas:
        raise DebiasSchemaError(
            "No overlapping area codes were found between MPD and Census OD data."
        )

    score_rows = pd.concat(
        [
            mpd_od.loc[:, ["origin", "flow"]].rename(columns={"origin": "area"}),
            mpd_od.loc[:, ["destination", "flow"]].rename(
                columns={"destination": "area"}
            ),
            benchmark_od.loc[:, ["origin", "flow"]].rename(columns={"origin": "area"}),
            benchmark_od.loc[:, ["destination", "flow"]].rename(
                columns={"destination": "area"}
            ),
        ],
        ignore_index=True,
    )
    area_scores = (
        score_rows.loc[score_rows["area"].isin(common_areas)]
        .groupby("area", sort=True, dropna=False)["flow"]
        .sum()
        .reset_index(name="score")
        .sort_values(["score", "area"], ascending=[False, True])
    )
    if n_areas is not None and np.isfinite(n_areas):
        n = int(n_areas)
        area_scores = area_scores.head(n)
    selected_areas = area_scores["area"].tolist()
    mpd_od = (
        mpd_od.loc[
            mpd_od["origin"].isin(selected_areas)
            & mpd_od["destination"].isin(selected_areas)
        ]
        .assign(mpd_source=f"locomizer_travel_to_work_{geography}")
        .loc[:, ["origin", "destination", "mpd_source", "flow"]]
        .sort_values(["origin", "destination"])
        .reset_index(drop=True)
    )
    benchmark_od = (
        benchmark_od.loc[
            benchmark_od["origin"].isin(selected_areas)
            & benchmark_od["destination"].isin(selected_areas)
        ]
        .sort_values(["origin", "destination"])
        .reset_index(drop=True)
    )
    if mpd_od.empty or benchmark_od.empty:
        raise DebiasSchemaError(
            "The selected areas do not contain overlapping MPD and Census OD flows."
        )
    coverage = _build_coverage(mpd_od, benchmark_od)
    final_areas = coverage["origin"].tolist()
    mpd_od = mpd_od.loc[
        mpd_od["origin"].isin(final_areas) & mpd_od["destination"].isin(final_areas)
    ].reset_index(drop=True)
    benchmark_od = benchmark_od.loc[
        benchmark_od["origin"].isin(final_areas)
        & benchmark_od["destination"].isin(final_areas)
    ].reset_index(drop=True)
    if complete_grid:
        completed = _complete_support(
            mpd_od, benchmark_od, final_areas, include_self_flows, fill_missing_flow
        )
        mpd_od = completed["mpd_od"]
        benchmark_od = completed["benchmark_od"]
        od_audit = completed["od_audit"]
    else:
        od_audit = _audit_support(
            mpd_od, benchmark_od, complete_grid, include_self_flows
        )
    covariate_bundle = _load_example_covariates(data_package, final_areas, geography)
    covariates = covariate_bundle["covariates"]
    covariate_source = covariate_bundle["covariate_source"]
    distance = _load_example_distance(
        data_package, final_areas, include_self_flows, geography
    )
    distance_source = str(distance["distance_source"].iloc[0])
    metadata = pd.DataFrame(
        {
            "data_package": [data_package],
            "geography": [geography],
            "mpd_source": [
                _example_metadata_source(
                    mpd_path,
                    f"Zenodo 10.5281/zenodo.13327082: {geography}_OD_travel2work",
                )
            ],
            "benchmark_source": [
                _example_metadata_source(
                    census_path,
                    (
                        f"Census 2021 ODWP01EW "
                        f"{'LTLA' if geography == 'lad' else 'MSOA'} workplace flows"
                    ),
                )
            ],
            "covariate_source": [covariate_source],
            "distance_source": [distance_source],
            "complete_grid": [complete_grid],
            "include_self_flows": [include_self_flows],
            "fill_missing_flow": [fill_missing_flow],
            "n_areas": [len(final_areas)],
            "n_mpd_od": [len(mpd_od)],
            "n_benchmark_od": [len(benchmark_od)],
            "n_expected_od": [od_audit["expected_od_rows"].iloc[0]],
            "n_mpd_zero_filled": [od_audit["n_mpd_zero_filled"].iloc[0]],
            "n_benchmark_zero_filled": [od_audit["n_benchmark_zero_filled"].iloc[0]],
            "mpd_total_flow": [od_audit["mpd_total_flow"].iloc[0]],
            "benchmark_total_flow": [od_audit["benchmark_total_flow"].iloc[0]],
            "mpd_balance_diff": [od_audit["mpd_balance_diff"].iloc[0]],
            "benchmark_balance_diff": [od_audit["benchmark_balance_diff"].iloc[0]],
        }
    )
    out: dict[str, pd.DataFrame] = {
        "mpd_od": mpd_od,
        "benchmark_od": benchmark_od,
        "coverage": coverage,
        "active_users": coverage.loc[:, ["origin", "user_count", "mpd_source"]],
        "population": coverage.loc[:, ["origin", "population"]],
        "covariates": covariates,
        "distance": distance,
        "od_audit": od_audit,
        "metadata": metadata,
    }
    if geography == "msoa":
        out["msoa_OD_travel2work"] = mpd_od
        out["census_msoa_OD_travel2work"] = benchmark_od
    else:
        out["lad_OD_travel2work"] = mpd_od
        out["census_lad_OD_travel2work"] = benchmark_od
    return out


def _load_example_source(
    path: str | Path | None, name: str, data_package: str
) -> pd.DataFrame:
    return (
        pd.read_csv(path)
        if path is not None
        else _load_companion_dataset(name, data_package)
    )


def _normalise_mpd_travel_to_work(
    data: pd.DataFrame,
    include_self_flows: bool,
    geography: str,
    validate_area_codes: bool = True,
) -> pd.DataFrame:
    origin_col = _find_col(
        data,
        [
            "origin",
            "MSOA21CD_home",
            "msoa21cd_home",
            "LAD22CD_home",
            "lad22cd_home",
            "origin_lad",
            "from",
        ],
    )
    destination_col = _find_col(
        data,
        [
            "destination",
            "MSOA21CD_work",
            "msoa21cd_work",
            "LAD22CD_work",
            "lad22cd_work",
            "destination_lad",
            "to",
        ],
    )
    flow_col = _find_col(data, ["flow", "count", "total_flow"])
    return _normalise_od_table(
        data,
        origin_col,
        destination_col,
        flow_col,
        include_self_flows,
        _example_area_pattern(geography) if validate_area_codes else None,
    )


def _normalise_census_travel_to_work(
    data: pd.DataFrame,
    include_self_flows: bool,
    geography: str,
    validate_area_codes: bool = True,
) -> pd.DataFrame:
    indicator_col = _find_col(
        data,
        [
            "Place of work indicator (4 categories) code",
            "place_of_work_indicator_4_categories_code",
            "place_of_work_indicator_code",
        ],
        required=False,
    )
    if indicator_col is not None:
        indicator = pd.to_numeric(data[indicator_col], errors="coerce")
        data = data.loc[indicator == 3].copy()
    origin_col = _find_col(
        data,
        [
            "origin",
            "Lower tier local authorities code",
            "lower_tier_local_authorities_code",
            "Middle layer Super Output Areas code",
            "middle_layer_super_output_areas_code",
            "Area of residence",
        ],
    )
    destination_col = _find_col(
        data,
        [
            "destination",
            "LTLA of workplace code",
            "ltla_of_workplace_code",
            "MSOA of workplace code",
            "msoa_of_workplace_code",
            "Area of workplace",
        ],
    )
    flow_col = _find_col(
        data,
        ["flow", "Count", "count", "All categories: Method of travel to work"],
    )
    return _normalise_od_table(
        data,
        origin_col,
        destination_col,
        flow_col,
        include_self_flows,
        _example_area_pattern(geography) if validate_area_codes else None,
    )


def _normalise_od_table(
    data: pd.DataFrame,
    origin_col: str,
    destination_col: str,
    flow_col: str,
    include_self_flows: bool,
    area_pattern: str | None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "origin": data[origin_col].astype(str),
            "destination": data[destination_col].astype(str),
            "flow": pd.to_numeric(data[flow_col], errors="coerce"),
        }
    )
    out = out.loc[
        out["origin"].notna()
        & out["destination"].notna()
        & np.isfinite(out["flow"])
        & (out["flow"] >= 0)
    ]
    if area_pattern:
        matches_origin = out["origin"].str.match(area_pattern, na=False)
        matches_destination = out["destination"].str.match(area_pattern, na=False)
        out = out.loc[matches_origin & matches_destination]
    if not include_self_flows:
        out = out.loc[out["origin"] != out["destination"]]
    return (
        out.groupby(["origin", "destination"], sort=True, dropna=False)["flow"]
        .sum()
        .reset_index()
        .loc[lambda df: df["flow"] > 0]
        .reset_index(drop=True)
    )


def _example_metadata_source(path: str | Path | None, package_source: str) -> str:
    return package_source if path is None else f"User-supplied file: {Path(path).name}"


def _python_data_companion(data_package: str = "debiaspydata"):
    try:
        return importlib.import_module(data_package)
    except ModuleNotFoundError as exc:
        if exc.name != data_package:
            raise
        raise DebiasSchemaError(
            f"Install the Python companion `{data_package}` to load the empirical "
            "example and its supporting covariate/centroid assets. "
            "R data packages and automatic downloads are not supported."
        ) from exc


def _find_col(
    data: pd.DataFrame, candidates: list[str], required: bool = True
) -> str | None:
    clean_names = [_clean_colname(col) for col in data.columns]
    for candidate in candidates:
        clean_candidate = _clean_colname(candidate)
        if clean_candidate in clean_names:
            return str(data.columns[clean_names.index(clean_candidate)])
    if required:
        raise DebiasSchemaError(
            "Could not find any of these columns: " + ", ".join(candidates)
        )
    return None


def _clean_colname(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower())


def _example_area_pattern(geography: str) -> str:
    if geography == "msoa":
        return r"^[EW][0-9]{8}$"
    return r"^(E0[6789]|W06)[0-9]{6}$"


def _load_example_covariates(
    data_package: str, areas: list[str], geography: str
) -> dict[str, object]:
    name = f"{geography}_covariates"
    covariates = _normalise_example_covariates(
        _load_companion_dataset(name, data_package), areas
    )
    if covariates.empty:
        raise DebiasSchemaError(
            f"The {data_package} covariate asset does not "
            "cover every selected example area."
        )
    return {"covariates": covariates, "covariate_source": f"{data_package}::{name}"}


def _normalise_example_covariates(data: pd.DataFrame, areas: list[str]) -> pd.DataFrame:
    area_col = _find_col(
        data,
        [
            "area",
            "lad",
            "lad21cd",
            "lad22cd",
            "ltla",
            "ltla_code",
            "msoa",
            "msoa21cd",
            "msoa_code",
            "code",
        ],
        required=False,
    )
    if area_col is None:
        return pd.DataFrame()
    area_set = sorted(map(str, areas))
    out = data.copy()
    out["area"] = out[area_col].astype(str)
    if area_col != "area":
        out = out.drop(columns=[area_col])
    out = out.loc[out["area"].isin(area_set) & out["area"].notna()]
    if out.empty:
        return pd.DataFrame()
    out = (
        out.sort_values("area", kind="stable")
        .drop_duplicates(subset="area", keep="first")
        .reset_index(drop=True)
    )
    if set(out["area"]) != set(area_set):
        return pd.DataFrame()
    out["_area_order"] = out["area"].map(
        {area: idx for idx, area in enumerate(area_set)}
    )
    return (
        out.sort_values("_area_order")
        .drop(columns=["_area_order"])
        .reset_index(drop=True)
    )


def _load_example_distance(
    data_package: str, areas: list[str], include_self_flows: bool, geography: str
) -> pd.DataFrame:
    centroids = _normalise_example_centroids(
        _load_companion_dataset(f"{geography}_centroids", data_package), areas
    )
    return _build_example_centroid_distance(
        centroids, areas, include_self_flows, f"debiasRdata_{geography}_centroids"
    )


def _normalise_example_centroids(data: pd.DataFrame, areas: list[str]) -> pd.DataFrame:
    area_col = _find_col(
        data,
        [
            "area",
            "lad",
            "lad21cd",
            "lad22cd",
            "ltla",
            "ltla_code",
            "msoa",
            "msoa21cd",
            "msoa_code",
            "code",
        ],
        False,
    )
    longitude_col = _find_col(data, ["longitude", "long", "lon", "x_longitude"], False)
    latitude_col = _find_col(data, ["latitude", "lat", "y_latitude"], False)
    easting_col = _find_col(data, ["easting", "bng_e", "x", "x_coord"], False)
    northing_col = _find_col(data, ["northing", "bng_n", "y", "y_coord"], False)
    if area_col is None or (
        (longitude_col is None or latitude_col is None)
        and (easting_col is None or northing_col is None)
    ):
        return _empty_example_centroids()
    area_set = sorted(map(str, areas))
    out = pd.DataFrame(
        {
            "area": data[area_col].astype(str),
            "longitude": _numeric_or_nan(data, longitude_col),
            "latitude": _numeric_or_nan(data, latitude_col),
            "easting": _numeric_or_nan(data, easting_col),
            "northing": _numeric_or_nan(data, northing_col),
        }
    )
    out = out.loc[out["area"].isin(area_set) & out["area"].notna()]
    if out.empty:
        return _empty_example_centroids()
    out = (
        out.sort_values("area", kind="stable")
        .drop_duplicates(subset="area", keep="first")
        .reset_index(drop=True)
    )
    if set(out["area"]) != set(area_set):
        return _empty_example_centroids()
    return out.reset_index(drop=True)


def _build_example_centroid_distance(
    centroids: pd.DataFrame,
    areas: list[str],
    include_self_flows: bool,
    distance_source: str,
) -> pd.DataFrame:
    area_list = sorted(map(str, areas))
    if centroids.empty or not set(area_list).issubset(set(centroids["area"])):
        raise DebiasSchemaError(
            "The centroid asset must cover every selected area with valid coordinates."
        )
    grid = pd.DataFrame(
        product(area_list, area_list), columns=["origin", "destination"]
    )
    if not include_self_flows:
        grid = grid.loc[grid["origin"] != grid["destination"]]
    coords = centroids.set_index("area")
    rows = []
    for _, row in grid.iterrows():
        origin = str(row["origin"])
        destination = str(row["destination"])
        o = coords.loc[origin]
        d = coords.loc[destination]
        if (
            np.isfinite(o["longitude"])
            and np.isfinite(o["latitude"])
            and np.isfinite(d["longitude"])
            and np.isfinite(d["latitude"])
        ):
            distance_km = _haversine_km(
                float(o["longitude"]),
                float(o["latitude"]),
                float(d["longitude"]),
                float(d["latitude"]),
            )
        elif (
            np.isfinite(o["easting"])
            and np.isfinite(o["northing"])
            and np.isfinite(d["easting"])
            and np.isfinite(d["northing"])
        ):
            distance_km = (
                (float(d["easting"]) - float(o["easting"])) ** 2
                + (float(d["northing"]) - float(o["northing"])) ** 2
            ) ** 0.5 / 1000
        else:
            raise DebiasSchemaError(
                "The centroid asset must cover every selected area "
                "with valid coordinates."
            )
        rows.append(
            {
                "origin": origin,
                "destination": destination,
                "distance_km": distance_km,
                "distance_source": distance_source,
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["origin", "destination"]).reset_index(drop=True)
    )


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_km = 6371.0088
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return float(2 * radius_km * np.arcsin(min(1, np.sqrt(a))))


def _numeric_or_nan(data: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None:
        return pd.Series(np.nan, index=data.index)
    return pd.Series(pd.to_numeric(data[col], errors="coerce"), index=data.index)


def _empty_example_centroids() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "area": pd.Series(dtype=str),
            "longitude": pd.Series(dtype=float),
            "latitude": pd.Series(dtype=float),
            "easting": pd.Series(dtype=float),
            "northing": pd.Series(dtype=float),
        }
    )


def _build_coverage(mpd_od: pd.DataFrame, benchmark_od: pd.DataFrame) -> pd.DataFrame:
    mpd_totals = (
        mpd_od.groupby("origin", sort=True, dropna=False)["flow"]
        .sum()
        .reset_index(name="user_count")
    )
    benchmark_totals = (
        benchmark_od.groupby("origin", sort=True, dropna=False)["flow"]
        .sum()
        .reset_index(name="population")
    )
    coverage = benchmark_totals.merge(mpd_totals, how="inner", on="origin")
    coverage = coverage.loc[
        (coverage["population"] > 0) & (coverage["user_count"] >= 0)
    ]
    coverage["mpd_source"] = (
        mpd_od["mpd_source"].iloc[0] if "mpd_source" in mpd_od.columns else "mpd"
    )
    coverage["destination"] = coverage["origin"]
    return (
        coverage.loc[
            :, ["origin", "destination", "population", "user_count", "mpd_source"]
        ]
        .sort_values("origin")
        .reset_index(drop=True)
    )


def _complete_support(
    mpd_od: pd.DataFrame,
    benchmark_od: pd.DataFrame,
    areas: list[str],
    include_self_flows: bool,
    fill_missing_flow: float,
) -> dict[str, pd.DataFrame]:
    area_list = sorted(set(map(str, areas)))
    grid = pd.DataFrame(
        product(area_list, area_list), columns=["origin", "destination"]
    )
    if not include_self_flows:
        grid = grid.loc[grid["origin"] != grid["destination"]]
    mpd_completed = grid.merge(mpd_od, how="left", on=["origin", "destination"])
    source_values = mpd_od.get("mpd_source", pd.Series(dtype=object)).dropna()
    default_source = str(source_values.iloc[0]) if not source_values.empty else "mpd"
    mpd_completed["mpd_source"] = mpd_completed["mpd_source"].fillna(default_source)
    mpd_completed["mpd_observed"] = mpd_completed["flow"].notna()
    mpd_completed["mpd_zero_filled"] = mpd_completed["flow"].isna()
    mpd_completed["mpd_row_status"] = np.where(
        mpd_completed["mpd_observed"], "observed", "zero_filled"
    )
    mpd_completed["flow"] = mpd_completed["flow"].fillna(fill_missing_flow)
    mpd_completed = mpd_completed.loc[
        :,
        [
            "origin",
            "destination",
            "mpd_source",
            "mpd_observed",
            "mpd_zero_filled",
            "mpd_row_status",
            "flow",
        ],
    ].sort_values(["origin", "destination"])
    bench_completed = grid.merge(benchmark_od, how="left", on=["origin", "destination"])
    bench_completed["benchmark_observed"] = bench_completed["flow"].notna()
    bench_completed["benchmark_zero_filled"] = bench_completed["flow"].isna()
    bench_completed["benchmark_row_status"] = np.where(
        bench_completed["benchmark_observed"], "observed", "zero_filled"
    )
    bench_completed["flow"] = bench_completed["flow"].fillna(fill_missing_flow)
    bench_completed = bench_completed.loc[
        :,
        [
            "origin",
            "destination",
            "benchmark_observed",
            "benchmark_zero_filled",
            "benchmark_row_status",
            "flow",
        ],
    ].sort_values(["origin", "destination"])
    return {
        "mpd_od": mpd_completed.reset_index(drop=True),
        "benchmark_od": bench_completed.reset_index(drop=True),
        "od_audit": _audit_support(
            mpd_completed, bench_completed, True, include_self_flows
        ),
    }


def _audit_support(
    mpd_od: pd.DataFrame,
    benchmark_od: pd.DataFrame,
    complete_grid: bool,
    include_self_flows: bool,
) -> pd.DataFrame:
    areas = sorted(
        set(mpd_od["origin"])
        | set(mpd_od["destination"])
        | set(benchmark_od["origin"])
        | set(benchmark_od["destination"])
    )
    expected = (
        len(areas) * len(areas) if include_self_flows else len(areas) * (len(areas) - 1)
    )
    mpd_origin = mpd_od.groupby("origin")["flow"].sum()
    mpd_dest = mpd_od.groupby("destination")["flow"].sum()
    bench_origin = benchmark_od.groupby("origin")["flow"].sum()
    bench_dest = benchmark_od.groupby("destination")["flow"].sum()
    mpd_origins = sorted(mpd_od["origin"].astype(str).unique())
    mpd_destinations = sorted(mpd_od["destination"].astype(str).unique())
    benchmark_origins = sorted(benchmark_od["origin"].astype(str).unique())
    benchmark_destinations = sorted(benchmark_od["destination"].astype(str).unique())
    duplicate_mpd = int(mpd_od.duplicated(["origin", "destination"]).sum())
    duplicate_benchmark = int(benchmark_od.duplicated(["origin", "destination"]).sum())
    same_area_set = (
        mpd_origins == mpd_destinations
        and benchmark_origins == benchmark_destinations
        and mpd_origins == benchmark_origins
    )
    strict_square_support = (
        same_area_set
        and len(mpd_od) == expected
        and len(benchmark_od) == expected
        and duplicate_mpd == 0
        and duplicate_benchmark == 0
        and bool((np.isfinite(mpd_od["flow"]) & (mpd_od["flow"] >= 0)).all())
        and bool(
            (np.isfinite(benchmark_od["flow"]) & (benchmark_od["flow"] >= 0)).all()
        )
    )
    return pd.DataFrame(
        {
            "complete_grid": [complete_grid],
            "include_self_flows": [include_self_flows],
            "strict_square_support": [strict_square_support],
            "same_origin_destination_area_set": [same_area_set],
            "n_areas": [len(areas)],
            "expected_od_rows": [expected],
            "n_mpd_od": [len(mpd_od)],
            "n_benchmark_od": [len(benchmark_od)],
            "n_mpd_duplicate_pairs": [duplicate_mpd],
            "n_benchmark_duplicate_pairs": [duplicate_benchmark],
            "n_mpd_zero_filled": [
                int(mpd_od.get("mpd_zero_filled", pd.Series(dtype=bool)).sum())
            ],
            "n_benchmark_zero_filled": [
                int(
                    benchmark_od.get(
                        "benchmark_zero_filled", pd.Series(dtype=bool)
                    ).sum()
                )
            ],
            "mpd_total_flow": [mpd_od["flow"].sum()],
            "benchmark_total_flow": [benchmark_od["flow"].sum()],
            "mpd_total_outflow": [mpd_origin.sum()],
            "mpd_total_inflow": [mpd_dest.sum()],
            "benchmark_total_outflow": [bench_origin.sum()],
            "benchmark_total_inflow": [bench_dest.sum()],
            "mpd_balance_diff": [(mpd_origin.sum() - mpd_dest.sum())],
            "benchmark_balance_diff": [(bench_origin.sum() - bench_dest.sum())],
        }
    )


def _load_companion_dataset(name: str, data_package: str) -> pd.DataFrame:
    companion = _python_data_companion(data_package)
    if name not in companion.list_datasets():
        raise DebiasSchemaError(
            f"The example-data route requires the {name} asset from {data_package}."
        )
    return companion.load_dataset(name)
