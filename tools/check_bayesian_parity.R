# Reference half of the fixed-effect Poisson investigation. Never edits debiasR.
args <- commandArgs(trailingOnly = TRUE)
pkgload::load_all(args[[1]], export_all = FALSE, helpers = FALSE, quiet = TRUE)
root <- args[[2]]
iterations <- as.integer(args[[3]])
seed <- as.integer(args[[4]])
reuse <- identical(args[[5]], "reuse")
read_input <- function(name) {
  path <- file.path(root, paste0(name, ".csv"))
  columns <- names(utils::read.csv(path, nrows = 0))
  types <- rep(NA_character_, length(columns))
  types[columns %in% c("origin", "destination", "area")] <- "character"
  value <- utils::read.csv(path, colClasses = types)
  for (column in intersect(c("mpd_observed", "mpd_zero_filled"), names(value))) {
    value[[column]] <- as.logical(value[[column]])
  }
  value
}
options(mc.cores = 1)
model_path <- file.path(root, "r_model.rds")
if (reuse) {
  model <- readRDS(model_path)
} else {
  result <- debiasR::adjust_multilevel_bayes(
    mpd_od_df = read_input("mpd_od"), coverage_df = read_input("coverage"),
    covariates_df = read_input("covariates"), distance_df = read_input("distance"),
    mobility_formula = "~ log_pop_o + log_pop_d + log_distance",
    bias_formula = "~ bias_e_origin", scenario = "s1", random_intercept = "none",
    model_engine = "bayesian", model_family = "poisson", backend = "rstanarm",
    observation_model = "reduced_form", target_scale = "mpd_counterfactual",
    iter = iterations, chains = 4, seed = seed, refresh = 0
  )
  model <- attr(result, "model")
  saveRDS(model, model_path)
}
draws <- as.matrix(model)
utils::write.csv(draws, file.path(root, "r_draws.csv"), row.names = FALSE)
utils::write.csv(model$data, file.path(root, "r_model_data.csv"), row.names = FALSE)
jsonlite::write_json(model$prior.info, file.path(root, "r_priors.json"),
                     auto_unbox = TRUE, pretty = TRUE, digits = 17)
for (mode in c("observed", "counterfactual")) {
  data <- model$data
  if (mode == "counterfactual") data$bias_e_origin <- 0
  lp <- debiasR:::.posterior_linpred_fixef(model, "rstanarm", data)
  utils::write.csv(lp, file.path(root, paste0("r_", mode, "_log_draws.csv")),
                   row.names = FALSE)
  if (mode == "observed") {
    log_likelihood <- apply(lp, 1, function(eta) {
      sum(stats::dpois(model$data$flow, exp(eta), log = TRUE))
    })
    utils::write.csv(data.frame(log_likelihood = log_likelihood),
                     file.path(root, "r_log_likelihood.csv"), row.names = FALSE)
  }
}
# Evaluate the actual centered prior at the exported posterior draws.
terms <- c("log_pop_o", "log_pop_d", "log_distance", "bias_e_origin")
X <- stats::model.matrix(stats::formula(model), model$data)
centered_intercept <- draws[, "(Intercept)"] +
  as.numeric(draws[, terms] %*% colMeans(X[, terms]))
log_prior <- stats::dnorm(centered_intercept, 0, 2.5, log = TRUE)
for (j in seq_along(terms)) {
  log_prior <- log_prior + stats::dnorm(draws[, terms[[j]]],
    model$prior.info$prior$location[[j]],
    model$prior.info$prior$adjusted_scale[[j]], log = TRUE)
}
utils::write.csv(data.frame(log_prior = log_prior),
                 file.path(root, "r_log_prior.csv"), row.names = FALSE)
