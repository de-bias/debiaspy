"""Investigate reduced-form Poisson parity using identical posterior draws.

Run from an installed integration package with its models extra and R/rstanarm.
Raw inputs and posterior files stay in the caller's output directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from debiaspy import _bayesian as bayes
from debiaspy import adjust_multilevel_bayes, debiaspy_example_data

TERMS = ["log_pop_o", "log_pop_d", "log_distance", "bias_e_origin"]
FORMULA = "flow ~ " + " + ".join(TERMS)
INPUTS = ["mpd_od", "coverage", "covariates", "distance"]


def read_csv(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"origin": str, "destination": str, "area": str})
    for column in ["mpd_observed", "mpd_zero_filled"]:
        if column in data:
            data[column] = data[column].astype(str).str.lower().isin(["true", "1"])
    return data


def build_model(data: pd.DataFrame) -> bmb.Model:
    model = bmb.Model(FORMULA, data, family="poisson", dropna=False, auto_scale=True)
    bayes._apply_rstanarm_marginal_priors(bmb, model)
    model.build()
    return model


def relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(actual - reference) / np.maximum(reference, 1e-300)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--reuse-fits", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    controls_path = root / "sampling_controls.json"
    if not args.reuse_fits:
        controls_path.write_text(
            json.dumps(
                {
                    "iterations": args.iterations,
                    "seed": args.seed,
                    "chains": 4,
                }
            )
            + "\n"
        )
        tables = (
            {name: read_csv(args.input_dir / f"{name}.csv") for name in INPUTS}
            if args.input_dir
            else debiaspy_example_data(n_areas=5, complete_grid=True)
        )
        for name in INPUTS:
            tables[name].to_csv(root / f"{name}.csv", index=False)
    subprocess.run(
        [
            "Rscript",
            str(Path(__file__).with_suffix(".R")),
            str(args.r_root.resolve()),
            str(root),
            str(args.iterations),
            str(args.seed),
            "reuse" if args.reuse_fits else "fresh",
        ],
        check=True,
    )
    if args.reuse_fits:
        python_data = read_csv(root / "python_model_data.csv")
        model = build_model(python_data)
        posterior = az.from_netcdf(root / "python_posterior.nc")
    else:
        result = adjust_multilevel_bayes(
            tables["mpd_od"],
            tables["coverage"],
            covariates_df=tables["covariates"],
            distance_df=tables["distance"],
            mobility_formula="~ log_pop_o + log_pop_d + log_distance",
            bias_formula="~ bias_e_origin",
            scenario="s1",
            random_intercept="none",
            model_engine="bayesian",
            model_family="poisson",
            backend="rstanarm",
            observation_model="reduced_form",
            target_scale="mpd_counterfactual",
            iter=args.iterations,
            chains=4,
            seed=args.seed,
            refresh=0,
        )
        handle = result.attrs["model"]
        model, posterior = handle.model, handle.inference_data
        python_data = model.data
        python_data.to_csv(root / "python_model_data.csv", index=False)
        posterior.to_netcdf(root / "python_posterior.nc")
    r_data = read_csv(root / "r_model_data.csv")
    # Check the actual fitting rows before injecting any shared draws.
    for column in ["origin", "destination", "flow", *TERMS]:
        if column in TERMS or column == "flow":
            np.testing.assert_allclose(python_data[column], r_data[column], rtol=1e-12)
        else:
            assert python_data[column].tolist() == r_data[column].tolist(), column
    r_draws = read_csv(root / "r_draws.csv")
    coefficients = r_draws[["(Intercept)", *TERMS]].to_numpy()
    shape = (4, len(coefficients) // 4)
    shared = az.from_dict(
        posterior={
            "Intercept": coefficients[:, 0].reshape(shape),
            **{
                term: coefficients[:, j + 1].reshape(shape)
                for j, term in enumerate(TERMS)
            },
        }
    )
    priors = json.loads((root / "r_priors.json").read_text())
    resolved = bayes._resolved_model_priors(model)
    scales = np.asarray(priors["prior"]["adjusted_scale"])
    np.testing.assert_allclose(
        [resolved[f"mu.{term}"]["parameters"]["sigma"] for term in TERMS],
        scales,
        rtol=1e-12,
    )
    assert resolved["mu.Intercept"]["parameters"] == {"mu": 0.0, "sigma": 2.5}
    assert priors["prior_intercept"]["dist"] == "normal"
    assert priors["prior_intercept"]["location"] == [0]
    assert priors["prior_intercept"]["scale"] == 2.5
    assert priors["prior"]["dist"] == "normal"
    np.testing.assert_equal(priors["prior"]["location"], np.zeros(4))
    centered = (
        coefficients[:, 0] + coefficients[:, 1:] @ r_data[TERMS].mean().to_numpy()
    )
    log_prior = norm.logpdf(centered, 0, 2.5)
    log_prior += norm.logpdf(coefficients[:, 1:], 0, scales).sum(axis=1)
    r_log_prior = read_csv(root / "r_log_prior.csv")["log_prior"].to_numpy()
    np.testing.assert_allclose(log_prior, r_log_prior, atol=1e-10, rtol=1e-12)
    checks = {}
    for mode in ["observed", "counterfactual"]:
        data = (
            python_data
            if mode == "observed"
            else bayes._counterfactual_bias_data(python_data, ["bias_e_origin"])
        )
        actual = bayes._predict_parent_draws(model, shared, data)
        reference = np.exp(read_csv(root / f"r_{mode}_log_draws.csv").to_numpy())
        error = relative_error(actual, reference)
        assert error < 1e-10, (mode, error)
        checks[mode] = {"shared_draw_max_relative_error": error}
        if mode == "observed":
            log_likelihood = poisson.logpmf(
                python_data["flow"].to_numpy()[None, :], actual
            ).sum(axis=1)
            r_log_likelihood = read_csv(root / "r_log_likelihood.csv")[
                "log_likelihood"
            ].to_numpy()
            np.testing.assert_allclose(
                log_likelihood, r_log_likelihood, atol=1e-8, rtol=1e-10
            )
            checks[mode]["log_likelihood_max_absolute_error"] = float(
                np.max(np.abs(log_likelihood - r_log_likelihood))
            )
    cf = bayes._counterfactual_bias_data(python_data, ["bias_e_origin"])
    python_flows = bayes._predict_parent_draws(model, posterior, cf)
    r_flows = np.exp(read_csv(root / "r_counterfactual_log_draws.csv").to_numpy())
    r_summary = bayes._adjusted_flow_diagnostics(az, shared, r_flows.sum(axis=1))
    python_summary = bayes._adjusted_flow_diagnostics(
        az, posterior, python_flows.sum(axis=1)
    )
    report = {
        "scope": "fixed-effect reduced-form Poisson; five-area Census by default",
        "reference_commit": subprocess.check_output(
            ["git", "-C", str(args.r_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "fresh_sampling": not args.reuse_fits,
        "sampling_controls": (
            json.loads(controls_path.read_text())
            if controls_path.exists()
            else {"status": "unrecorded in reused fit"}
        ),
        "fit_rows": len(r_data),
        "posterior_draws_per_backend": len(coefficients),
        "prior_parameters_match": True,
        "centered_log_prior_max_absolute_error": float(
            np.max(np.abs(log_prior - r_log_prior))
        ),
        "prediction_checks": checks,
        "resolved_priors": resolved,
        "bias_fit_range": [
            float(r_data.bias_e_origin.min()),
            float(r_data.bias_e_origin.max()),
        ],
        "counterfactual_bias": 0.0,
        "counterfactual_distance_in_sample_sd": float(
            abs(r_data.bias_e_origin.mean()) / r_data.bias_e_origin.std()
        ),
        "r_total": r_summary,
        "python_total": python_summary,
        "independent_mean_max_relative_difference": relative_error(
            python_flows.mean(axis=0), r_flows.mean(axis=0)
        ),
        "independent_median_max_relative_difference": relative_error(
            np.median(python_flows, axis=0), np.median(r_flows, axis=0)
        ),
        "conclusion": (
            "prior and shared-draw prediction parity pass; "
            "independent mean precision is reported separately"
        ),
    }
    (root / "investigation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
