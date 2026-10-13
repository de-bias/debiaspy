"""Compare actual R and Bambi priors/predictions without a posterior parity claim."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from debiaspy import _bayesian as bayes


def test_rstanarm_shared_draw_predictions_and_resolved_priors(tmp_path: Path) -> None:
    r_root = os.environ.get("DEBIASR_ROOT")
    if not r_root or not shutil.which("Rscript"):
        pytest.skip("Set DEBIASR_ROOT with Rscript and rstanarm for live parity")
    bmb = pytest.importorskip("bambi")
    az = pytest.importorskip("arviz")
    check = subprocess.run(
        [
            "Rscript",
            "-e",
            "quit(status=if(requireNamespace('rstanarm',quietly=TRUE)) 0 else 1)",
        ],
        capture_output=True,
    )
    if check.returncode:
        pytest.skip("rstanarm is not installed")
    terms = ["log_distance", "bias_e_origin"]
    data = pd.DataFrame(
        {
            "flow": [1, 4, 2, 8, 3, 6],
            "log_distance": [-1.0, 0.5, 1.5, 2.0, -0.5, 0.0],
            "bias_e_origin": [0.1, 0.4, 0.8, 0.7, 0.2, 0.5],
        }
    )
    data.to_csv(tmp_path / "data.csv", index=False)
    script = r"""
args <- commandArgs(TRUE)
source(file.path(args[[1]], "R", "adjust_multilevel_bayes.R"))
root <- args[[2]]
data <- read.csv(file.path(root, "data.csv"))
# Prior draws exercise the real fitted-object prediction interface. They are
# not used as evidence about convergence or posterior accuracy.
fit <- rstanarm::stan_glm(flow ~ log_distance + bias_e_origin, data,
  family = poisson(), prior_PD = TRUE, iter = 40, chains = 1, seed = 131,
  refresh = 0)
write.csv(as.matrix(fit), file.path(root, "draws.csv"), row.names = FALSE)
jsonlite::write_json(fit$prior.info, file.path(root, "priors.json"),
  auto_unbox = TRUE, digits = 17)
data$bias_e_origin <- 0
write.csv(.posterior_linpred_fixef(fit, "rstanarm", data),
  file.path(root, "log_predictions.csv"), row.names = FALSE)
"""
    (tmp_path / "compare.R").write_text(script)
    subprocess.run(
        ["Rscript", str(tmp_path / "compare.R"), r_root, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    model = bmb.Model("flow ~ log_distance + bias_e_origin", data, family="poisson")
    bayes._apply_rstanarm_marginal_priors(bmb, model)
    model.build()
    priors = json.loads((tmp_path / "priors.json").read_text())
    resolved = bayes._resolved_model_priors(model)
    np.testing.assert_allclose(
        [resolved[f"mu.{term}"]["parameters"]["sigma"] for term in terms],
        priors["prior"]["adjusted_scale"],
        rtol=1e-12,
    )
    assert resolved["mu.Intercept"]["parameters"] == {"mu": 0.0, "sigma": 2.5}
    draws = pd.read_csv(tmp_path / "draws.csv")
    posterior = az.from_dict(
        posterior={
            "Intercept": draws["(Intercept)"].to_numpy()[None, :],
            **{term: draws[term].to_numpy()[None, :] for term in terms},
        }
    )
    cf = bayes._counterfactual_bias_data(data, ["bias_e_origin"])
    actual = bayes._predict_parent_draws(model, posterior, cf)
    reference = np.exp(pd.read_csv(tmp_path / "log_predictions.csv").to_numpy())
    np.testing.assert_allclose(actual, reference, rtol=1e-11)
