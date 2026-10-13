#!/usr/bin/env Rscript

git_checked <- function(repo, args) {
  output <- suppressWarnings(
    system2("git", c("-C", repo, args), stdout = TRUE, stderr = TRUE)
  )
  status <- attr(output, "status")
  if (is.null(status)) status <- 0L
  if (!identical(status, 0L)) {
    stop(
      sprintf("git %s failed in %s: %s", paste(args, collapse = " "), repo, paste(output, collapse = "\n")),
      call. = FALSE
    )
  }
  output
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Usage: check_article_parity.R R_ROOT DISPLAY_DIR RAW_DIR")
debiasr_root <- normalizePath(args[[1]], mustWork = TRUE)
goldens_tmp <- normalizePath(args[[2]], mustWork = TRUE)
raw_goldens_articles <- normalizePath(args[[3]], mustWork = TRUE)
write_raw_goldens <- TRUE

if (!file.exists(file.path(debiasr_root, "DESCRIPTION"))) {
  stop("DEBIASR_ROOT does not point to a debiasR checkout.", call. = FALSE)
}
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required.", call. = FALSE)
if (!requireNamespace("pkgload", quietly = TRUE)) stop("pkgload is required.", call. = FALSE)
if (!requireNamespace("dplyr", quietly = TRUE)) stop("dplyr is required.", call. = FALSE)
if (!requireNamespace("tibble", quietly = TRUE)) stop("tibble is required.", call. = FALSE)
if (!requireNamespace("debiasRdata", quietly = TRUE)) stop("debiasRdata is required.", call. = FALSE)

expected_commit <- "6fa2709ae42de6dc525a4a46182fdfb448b14a3f"
debiasr_commit <- git_checked(debiasr_root, c("rev-parse", "HEAD"))[[1]]
if (!identical(debiasr_commit, expected_commit)) stop("Use debiasR commit ", expected_commit)
if (length(git_checked(debiasr_root, c("status", "--porcelain"))) > 0L) {
  stop("DEBIASR_ROOT has uncommitted changes; refusing to generate article goldens.", call. = FALSE)
}

pkgload::load_all(debiasr_root, export_all = FALSE, helpers = FALSE, quiet = TRUE)

AUDIT_SIGNIFICANT_DIGITS <- 10L
RAW_AUDIT_SIGNIFICANT_DIGITS <- 17L
DISPLAY_ZERO_EPSILON <- 1e-12

ARTICLE_PAGES <- c(
  "index",
  "v03-getting-set-up",
  "v04-measuring-coverage-bias",
  "v05-identifying-and-explaining-bias",
  "v06-adjusting-biases",
  "v07-validation",
  "v09-advanced-bayesian-adjustment"
)

ARTICLE_URLS <- list(
  index = "https://de-bias.github.io/debiasR/index.html",
  `v03-getting-set-up` = "https://de-bias.github.io/debiasR/articles/v03-getting-set-up.html",
  `v04-measuring-coverage-bias` = "https://de-bias.github.io/debiasR/articles/v04-measuring-coverage-bias.html",
  `v05-identifying-and-explaining-bias` = "https://de-bias.github.io/debiasR/articles/v05-identifying-and-explaining-bias.html",
  `v06-adjusting-biases` = "https://de-bias.github.io/debiasR/articles/v06-adjusting-biases.html",
  `v07-validation` = "https://de-bias.github.io/debiasR/articles/v07-validation.html",
  `v09-advanced-bayesian-adjustment` = "https://de-bias.github.io/debiasR/articles/v09-advanced-bayesian-adjustment.html"
)

write_csv_to_dir <- function(x, name, directory, significant_digits, canonicalize) {
  old_options <- options(digits = significant_digits)
  on.exit(options(old_options), add = TRUE)
  x <- as.data.frame(x)
  if (canonicalize) {
    numeric_columns <- vapply(x, is.numeric, logical(1))
    x[numeric_columns] <- lapply(x[numeric_columns], function(column) {
      column[abs(column) <= DISPLAY_ZERO_EPSILON] <- 0
      signif(column, significant_digits)
    })
  }
  utils::write.csv(x, file.path(directory, name), row.names = FALSE)
}

logical_schema_type <- function(x) {
  if (all(is.na(x))) return("missing")
  if (is.logical(x)) return("boolean")
  if (is.numeric(x)) return("numeric")
  if (inherits(x, c("Date", "POSIXct", "POSIXlt"))) return("datetime")
  "string"
}

write_json <- function(x, name, directory = goldens_tmp) {
  jsonlite::write_json(
    x,
    file.path(directory, name),
    auto_unbox = TRUE,
    pretty = TRUE,
    null = "null"
  )
}

write_schema_json <- function(x, name) {
  x <- as.data.frame(x)
  schema_name <- sub("\\.csv$", ".schema.json", name)
  write_json(
    list(
      row_count = unname(as.integer(nrow(x))),
      column_count = unname(as.integer(ncol(x))),
      columns = lapply(names(x), function(column_name) {
        column <- x[[column_name]]
        list(
          name = column_name,
          logical_type = logical_schema_type(column),
          missing_count = unname(as.integer(sum(is.na(column))))
        )
      })
    ),
    schema_name
  )
}

stable_article_display <- function(x, name) {
  x <- as.data.frame(x)
  stable_spearman_files <- c(
    "v07-validation.level2.origin_margin_metrics.csv",
    "v07-validation.level2.destination_margin_metrics.csv"
  )
  if (name %in% stable_spearman_files && "spearman_rho" %in% names(x)) {
    x$spearman_rho <- round(x$spearman_rho, 3)
  }
  x
}

write_csv <- function(x, name) {
  write_schema_json(x, name)
  write_csv_to_dir(stable_article_display(x, name), name, goldens_tmp, AUDIT_SIGNIFICANT_DIGITS, canonicalize = TRUE)
  if (write_raw_goldens) {
    write_csv_to_dir(x, name, raw_goldens_articles, RAW_AUDIT_SIGNIFICANT_DIGITS, canonicalize = FALSE)
  }
}

write_page_meta <- function(page, scope) {
  write_json(
    list(
      page = page,
      source_url = ARTICLE_URLS[[page]],
      scope = scope,
      evidence = "R and Python calls executed without MCMC, precomputed Bayesian file-ingestion checks, and a static Python backend inventory",
      bayesian_execution = "The article harness does not execute MCMC. Separate posterior-comparison evidence for the controlled Bayesian fixtures is recorded in docs/reports/bayesian_checks.json"
    ),
    paste0(page, ".meta.json")
  )
}

object_counts <- function(objects) {
  data.frame(
    object = names(objects),
    rows = vapply(objects, nrow, integer(1)),
    columns = vapply(objects, ncol, integer(1)),
    stringsAsFactors = FALSE
  )
}

input_schemas <- function(objects) {
  do.call(rbind, lapply(names(objects), function(object_name) {
    frame <- as.data.frame(objects[[object_name]])
    data.frame(
      object = object_name,
      column = names(frame),
      logical_type = vapply(frame, logical_schema_type, character(1)),
      missing_count = vapply(frame, function(x) sum(is.na(x)), integer(1)),
      stringsAsFactors = FALSE
    )
  }))
}

adjustment_preview <- function(adjusted, lookup, n = 5L) {
  adjusted |>
    dplyr::left_join(lookup, by = c("origin", "destination")) |>
    dplyr::select(
      origin,
      destination,
      flow_mpd_raw = flow,
      flow_mpd_adjusted = flow_adj,
      flow_benchmark
    ) |>
    dplyr::slice_head(n = n)
}

attr_frame <- function(x, keys) {
  values <- lapply(keys, function(key) {
    value <- attr(x, key, exact = TRUE)
    if (is.null(value)) NA else unname(value[[1]])
  })
  data.frame(attribute = keys, value = unlist(values), stringsAsFactors = FALSE)
}

raking_summary <- function(x) {
  data.frame(
    raw_total = sum(x$flow, na.rm = TRUE),
    adjusted_total = sum(x$flow_adj, na.rm = TRUE),
    median_weight = median(x$weight_ipf, na.rm = TRUE),
    ipf_converged = attr(x, "ipf_converged", exact = TRUE),
    ipf_iterations = attr(x, "ipf_iterations", exact = TRUE)
  )
}

adjustment_set_summary <- function(results) {
  data.frame(
    method = names(results),
    engine = vapply(results, function(x) {
      engine <- attr(x, "model_engine", exact = TRUE)
      if (is.null(engine)) "direct_adjustment" else engine
    }, character(1)),
    rows = vapply(results, nrow, integer(1)),
    adjusted_total = vapply(results, function(x) sum(x$flow_adj, na.rm = TRUE), numeric(1)),
    stringsAsFactors = FALSE
  )
}

combine_overall <- function(results, methods) {
  dplyr::bind_rows(lapply(methods, function(method) as.data.frame(results[[method]])))
}

combine_nested <- function(results, methods, key) {
  dplyr::bind_rows(lapply(methods, function(method) {
    out <- as.data.frame(results[[method]][[key]])
    out <- out[, names(out) != "method", drop = FALSE]
    cbind(method = method, out, stringsAsFactors = FALSE)
  }))
}

method_order_head <- function(frame, methods, n) {
  dplyr::bind_rows(lapply(methods, function(target_method) {
    frame |>
      dplyr::filter(.data$method == .env$target_method) |>
      dplyr::slice_head(n = n)
  }))
}

nearest_neighbors <- function(distance, areas, k) {
  area_set <- as.character(areas)
  candidate <- distance |>
    dplyr::filter(
      .data$origin %in% area_set,
      .data$destination %in% area_set,
      .data$origin != .data$destination
    ) |>
    dplyr::arrange(.data$origin, .data$distance_km, .data$destination)
  dplyr::bind_rows(lapply(area_set, function(area) {
    candidate |>
      dplyr::filter(.data$origin == area) |>
      dplyr::slice_head(n = k) |>
      dplyr::transmute(area = .data$origin, neighbor = .data$destination)
  }))
}

neighbor_summary <- function(neighbors) {
  counts <- neighbors |>
    dplyr::count(.data$area, name = "neighbors")
  data.frame(
    areas = dplyr::n_distinct(counts$area),
    links = nrow(neighbors),
    min_neighbors = min(counts$neighbors),
    max_neighbors = max(counts$neighbors)
  )
}

read_extdata <- function(filename) {
  utils::read.csv(file.path(debiasr_root, "inst", "extdata", filename), stringsAsFactors = FALSE, check.names = FALSE)
}

bayesian_backend_contract <- function() {
  data.frame(
    scope = c("nonlatent", "latent_two_level"),
    status = c("implemented", "implemented"),
    implementation = c(
      "Bambi/PyMC",
      "same debiasR Stan program via CmdStanPy"
    ),
    model_families = c(
      "poisson|negbin|zip|zinb",
      "poisson|negbin"
    ),
    evidence_scope = c(
      "Static Python backend inventory; this article harness does not execute MCMC. Controlled Bayesian parity evidence: docs/reports/bayesian_checks.json.",
      "Static Python backend inventory and pinned Stan-source identity; this article harness does not execute MCMC. Controlled Bayesian parity evidence: docs/reports/bayesian_checks.json."
    ),
    stringsAsFactors = FALSE
  )
}

coverage_lad <- debiasRdata::coverage_lad
lad_covariates <- debiasRdata::lad_covariates
lad_centroids <- debiasRdata::lad_centroids
lad_OD_travel2work <- debiasRdata::lad_OD_travel2work
census_lad_OD_travel2work <- debiasRdata::census_lad_OD_travel2work

namespace_lines <- readLines(file.path(debiasr_root, "NAMESPACE"), warn = FALSE)
exports <- sort(sub("^export\\((.*)\\)$", "\\1", grep("^export\\(", namespace_lines, value = TRUE)))
write_csv(data.frame("function" = exports, check.names = FALSE), "index.public_exports.csv")
write_csv(
  data.frame(
    order = seq_along(ARTICLE_PAGES),
    page = ARTICLE_PAGES,
    url = unname(vapply(ARTICLE_PAGES, function(page) ARTICLE_URLS[[page]], character(1))),
    scope = c(
      "package overview",
      "setup data inputs",
      "coverage bias",
      "explaining bias",
      "adjusting flows",
      "validation",
      "advanced multilevel"
    ),
    stringsAsFactors = FALSE
  ),
  "index.workflow_pages.csv"
)
write_page_meta("index", "public API and workflow page map")

v03_objects <- list(
  lad_OD_travel2work = lad_OD_travel2work,
  census_lad_OD_travel2work = census_lad_OD_travel2work,
  coverage_lad = coverage_lad,
  lad_covariates = lad_covariates,
  lad_centroids = lad_centroids
)
write_csv(object_counts(v03_objects), "v03-getting-set-up.data_counts.csv")
write_csv(input_schemas(v03_objects), "v03-getting-set-up.input_schemas.csv")
for (object_name in names(v03_objects)) {
  write_csv(utils::head(v03_objects[[object_name]], 6), paste0("v03-getting-set-up.", object_name, ".head.csv"))
}
write_page_meta("v03-getting-set-up", "raw debiasRdata inputs")

coverage <- coverage_lad |>
  dplyr::rename(area = code)
bias_lad <- debiasR::measure_bias(coverage)
distribution_bias <- debiasR::measure_bias_distribution(
  coverage_df = coverage,
  area_col = "area",
  population_col = "population",
  user_count_col = "user_count"
)
bias_residuals <- debiasR::validate_bias_residual_structure(
  coverage_df = coverage,
  coverage_area_col = "area",
  population_col = "population",
  user_count_col = "user_count"
)
residual_lad <- bias_residuals$area_level |>
  dplyr::mutate(
    baseline_coverage_bias = 1 - .data$global_coverage_score,
    residual_bias = .data$coverage_bias - .data$baseline_coverage_bias
  )
write_csv(utils::head(coverage_lad, 6), "v04-measuring-coverage-bias.coverage_lad.head.csv")
write_csv(utils::head(coverage, 6), "v04-measuring-coverage-bias.coverage_renamed.head.csv")
write_csv(
  bias_lad |>
    dplyr::select(area, name, population, user_count, coverage_score, coverage_bias) |>
    utils::head(6),
  "v04-measuring-coverage-bias.measure_bias.head.csv"
)
write_csv(
  data.frame(
    lad_count = nrow(coverage),
    census_pop = sum(coverage$population),
    digital_trace_pop = sum(coverage$user_count),
    overall_coverage = sum(coverage$user_count) / sum(coverage$population),
    overall_coverage_bias = 1 - sum(coverage$user_count) / sum(coverage$population)
  ),
  "v04-measuring-coverage-bias.coverage_totals.csv"
)
write_csv(
  bias_lad |>
    dplyr::arrange(.data$coverage_bias, .data$area) |>
    dplyr::select(area, name, coverage_score, coverage_bias) |>
    utils::head(6),
  "v04-measuring-coverage-bias.lowest_bias.head.csv"
)
write_csv(distribution_bias$summary, "v04-measuring-coverage-bias.distribution.summary.csv")
write_csv(
  distribution_bias$area_level |>
    dplyr::mutate(abs_share_difference = abs(.data$share_difference_user_minus_population)) |>
    dplyr::arrange(dplyr::desc(.data$abs_share_difference), .data$area) |>
    dplyr::select(-abs_share_difference) |>
    utils::head(6),
  "v04-measuring-coverage-bias.distribution.top_share_diff.csv"
)
write_csv(bias_residuals$summary, "v04-measuring-coverage-bias.residual.summary.csv")
write_csv(bias_residuals$residual_definitions, "v04-measuring-coverage-bias.residual.definitions.csv")
write_csv(
  residual_lad |>
    dplyr::select(area, population, user_count, coverage_score, coverage_bias, expected_user_count, residual_bias) |>
    utils::head(6),
  "v04-measuring-coverage-bias.residual.area_head.csv"
)
write_csv(
  data.frame(
    plot = c("coverage_bias_histogram", "population_user_count_scatter", "baseline_residual_panel", "residual_bias_histogram"),
    source_rows = c(nrow(bias_lad), nrow(bias_lad), nrow(residual_lad), nrow(residual_lad)),
    x = c("coverage_bias", "population", "population", "residual_bias"),
    y = c("count", "user_count", "residual_bias", "count"),
    stringsAsFactors = FALSE
  ),
  "v04-measuring-coverage-bias.plot_contracts.csv"
)
write_page_meta("v04-measuring-coverage-bias", "coverage-bias tables")

bias_lad_v05 <- debiasR::measure_bias(coverage)
bias_residuals_v05 <- debiasR::validate_bias_residual_structure(
  coverage_df = coverage,
  coverage_area_col = "area",
  population_col = "population",
  user_count_col = "user_count"
)
residual_lad_v05 <- bias_residuals_v05$area_level |>
  dplyr::mutate(
    baseline_coverage_bias = 1 - .data$global_coverage_score,
    residual_bias = .data$coverage_bias - .data$baseline_coverage_bias
  ) |>
  dplyr::left_join(coverage |> dplyr::select(area, name), by = "area")
covariate_cols <- c(
  "area",
  "per_age_20.29",
  "per_age_70plus",
  "per_level4",
  "per_hh_no_centralheat",
  "per_NS_SeC_L13_routine",
  "rural_pct"
)
bias_covariates <- residual_lad_v05 |>
  dplyr::left_join(lad_covariates |> dplyr::select(dplyr::all_of(covariate_cols)), by = "area")
bias_model <- stats::lm(
  residual_bias ~
    per_age_20.29 +
    per_age_70plus +
    per_level4 +
    per_hh_no_centralheat +
    per_NS_SeC_L13_routine +
    rural_pct,
  data = bias_covariates
)
coefficient_table <- as.data.frame(stats::coef(summary(bias_model)))
coefficient_table$term <- rownames(coefficient_table)
rownames(coefficient_table) <- NULL
names(coefficient_table) <- c("estimate", "std_error", "statistic", "p_value", "term")
coefficient_table <- coefficient_table[, c("term", "estimate", "std_error", "statistic", "p_value")]
bias_covariates <- bias_covariates |>
  dplyr::mutate(
    fitted_residual_bias = stats::fitted(bias_model),
    unexplained_residual = .data$residual_bias - .data$fitted_residual_bias
  )
write_csv(utils::head(coverage, 6), "v05-identifying-and-explaining-bias.coverage.head.csv")
write_csv(utils::head(lad_covariates, 6), "v05-identifying-and-explaining-bias.covariates.head.csv")
write_csv(
  bias_lad_v05 |>
    dplyr::select(area, population, user_count, coverage_score, coverage_bias) |>
    utils::head(6),
  "v05-identifying-and-explaining-bias.bias.head.csv"
)
write_csv(
  residual_lad_v05 |>
    dplyr::select(area, name, coverage_score, coverage_bias, residual_bias) |>
    utils::head(6),
  "v05-identifying-and-explaining-bias.residual.head.csv"
)
write_csv(
  bias_covariates |>
    dplyr::select(area, residual_bias, per_age_20.29, per_age_70plus, per_level4, rural_pct) |>
    utils::head(6),
  "v05-identifying-and-explaining-bias.bias_covariates.head.csv"
)
write_csv(coefficient_table, "v05-identifying-and-explaining-bias.coefficient_table.csv")
write_csv(
  bias_covariates |>
    dplyr::select(area, residual_bias, fitted_residual_bias, unexplained_residual) |>
    utils::head(6),
  "v05-identifying-and-explaining-bias.fitted_residual.head.csv"
)
write_csv(
  bias_covariates |>
    dplyr::mutate(abs_unexplained = abs(.data$unexplained_residual)) |>
    dplyr::arrange(dplyr::desc(.data$abs_unexplained), .data$area) |>
    dplyr::select(area, name, residual_bias, fitted_residual_bias, unexplained_residual) |>
    utils::head(10),
  "v05-identifying-and-explaining-bias.top_unexplained.csv"
)
write_csv(
  data.frame(
    plot = c("residual_vs_age_20_29", "residual_vs_level4", "observed_vs_fitted_residual"),
    source_rows = c(nrow(bias_covariates), nrow(bias_covariates), nrow(bias_covariates)),
    x = c("per_age_20.29", "per_level4", "fitted_residual_bias"),
    y = c("residual_bias", "residual_bias", "residual_bias"),
    stringsAsFactors = FALSE
  ),
  "v05-identifying-and-explaining-bias.plot_contracts.csv"
)
write_page_meta("v05-identifying-and-explaining-bias", "residual-bias regression")

example_data <- debiasR::debiasR_example_data(n_areas = 25, complete_grid = TRUE)
mpd_od <- example_data$mpd_od
benchmark_od <- example_data$benchmark_od
coverage_adjustment <- example_data$coverage
covariates_adjustment <- example_data$covariates
distance <- example_data$distance
benchmark_lookup <- benchmark_od |>
  dplyr::select(origin, destination, flow_benchmark = flow)
write_csv(utils::head(lad_OD_travel2work, 6), "v06-adjusting-biases.raw_heads.OD_travel2work.csv")
write_csv(utils::head(census_lad_OD_travel2work, 6), "v06-adjusting-biases.raw_heads.census_OD_travel2work.csv")
write_csv(utils::head(coverage, 6), "v06-adjusting-biases.raw_heads.coverage.csv")
write_csv(object_counts(example_data[c("mpd_od", "benchmark_od", "coverage", "covariates", "distance")]), "v06-adjusting-biases.example_counts.csv")
adj_inverse_penetration <- debiasR::adjust_inverse_penetration(mpd_od, coverage_adjustment, weight_by = "origin")
adj_selection_rate <- debiasR::adjust_selection_rate(
  mpd_od,
  coverage_adjustment,
  covariates_df = covariates_adjustment,
  covariate_col = "rural_pct",
  weight_by = "origin",
  benchmark_od_df = benchmark_od,
  calibration_aggregate = "origin"
)
adj_selection_rate2 <- debiasR::adjust_selection_rate2(
  mpd_od,
  coverage_adjustment,
  weight_by = "origin",
  benchmark_od_df = benchmark_od,
  calibration_aggregate = "origin"
)
adj_raking_ratio <- debiasR::adjust_raking_ratio(mpd_od, benchmark_od_df = benchmark_od)
adj_coefficient <- debiasR::adjust_coefficient(mpd_od, benchmark_od, model_family = "ols", level = "od")
write_csv(adjustment_preview(adj_inverse_penetration, benchmark_lookup), "v06-adjusting-biases.inverse_penetration.preview.csv")
write_csv(adjustment_preview(adj_selection_rate, benchmark_lookup), "v06-adjusting-biases.selection_rate.preview.csv")
write_csv(attr_frame(adj_selection_rate, "r_global"), "v06-adjusting-biases.selection_rate.attrs.csv")
write_csv(adjustment_preview(adj_selection_rate2, benchmark_lookup), "v06-adjusting-biases.selection_rate2.preview.csv")
write_csv(attr_frame(adj_selection_rate2, "k"), "v06-adjusting-biases.selection_rate2.attrs.csv")
write_csv(adjustment_preview(adj_raking_ratio, benchmark_lookup), "v06-adjusting-biases.raking_ratio.preview.csv")
write_csv(raking_summary(adj_raking_ratio), "v06-adjusting-biases.raking_ratio.summary.csv")
write_csv(adjustment_preview(adj_coefficient, benchmark_lookup), "v06-adjusting-biases.coefficient.preview.csv")
write_csv(attr_frame(adj_coefficient, "coef"), "v06-adjusting-biases.coefficient.attrs.csv")
write_csv(read_extdata("v06-bayesian-s1-comparison.csv"), "v06-adjusting-biases.bayesian_precomputed.display.csv")
write_csv(read_extdata("v06-bayesian-s1-metadata.csv"), "v06-adjusting-biases.bayesian_precomputed.metadata.csv")
write_csv(read_extdata("v06-bayesian-s1-model-terms.csv"), "v06-adjusting-biases.bayesian_precomputed.model_terms.csv")
method_results <- debiasR::adjust_all_methods(
  mpd_od_df = mpd_od,
  coverage_df = coverage_adjustment,
  benchmark_od_df = benchmark_od,
  covariates_df = covariates_adjustment,
  distance_df = distance,
  covariate_col = "rural_pct",
  multilevel_engine = "frequentist"
)
write_csv(adjustment_set_summary(method_results), "v06-adjusting-biases.adjust_all_methods.summary.csv")
write_page_meta("v06-adjusting-biases", "adjustment-method outputs")

validation_data <- debiasR::debiasR_example_data(n_areas = Inf, complete_grid = TRUE, geography = "lad")
mpd_val <- validation_data$mpd_od
benchmark_val <- validation_data$benchmark_od
coverage_val <- validation_data$coverage
covariates_val <- validation_data$covariates
distance_val <- validation_data$distance
deterministic_methods <- c("inverse_penetration", "selection_rate2", "coefficient")
deterministic_results <- debiasR::adjust_all_methods(
  mpd_od_df = mpd_val,
  coverage_df = coverage_val,
  benchmark_od_df = benchmark_val,
  covariates_df = covariates_val,
  covariate_col = "per_level4",
  methods = deterministic_methods
)
deterministic_results <- c(list(unadjusted = dplyr::mutate(mpd_val, flow_adj = flow)), deterministic_results)
deterministic_method_results <- debiasR::validate_flow_prepare_outputs(deterministic_results, mpd_df = mpd_val)
bayesian_results <- read_extdata("v07-validation-bayesian-adjusted.csv")
bayesian_method_results <- split(bayesian_results[, setdiff(names(bayesian_results), "method")], bayesian_results$method)
all_method_results <- c(deterministic_method_results, bayesian_method_results)
level1_methods <- c("unadjusted", "inverse_penetration", "selection_rate2", "coefficient", "bayes_gravity", "bayes_gravity_education", "bayes_origin_pool")
validation_methods <- c("inverse_penetration", "coefficient", "bayes_gravity_education", "unadjusted")
labels <- c(
  unadjusted = "Unadjusted raw MPD",
  inverse_penetration = "Inverse penetration",
  selection_rate2 = "Selection rate II",
  coefficient = "Coefficient adjustment",
  bayes_gravity = "Bayesian gravity",
  bayes_gravity_education = "Bayesian gravity + education",
  bayes_origin_pool = "Bayesian gravity + education + origin pooling"
)
write_csv(object_counts(validation_data[c("mpd_od", "benchmark_od", "coverage", "covariates", "distance")]), "v07-validation.example_counts.csv")
write_csv(
  data.frame(
    method = level1_methods,
    label = unname(labels[level1_methods]),
    source = ifelse(level1_methods %in% names(deterministic_method_results), "executed_without_mcmc_or_raw", "precomputed_bayesian_fixture"),
    stringsAsFactors = FALSE
  ),
  "v07-validation.method_set.csv"
)
write_csv(read_extdata("v07-validation-bayesian-display.csv"), "v07-validation.bayesian_precomputed.display.csv")
write_csv(read_extdata("v07-validation-bayesian-metadata.csv"), "v07-validation.bayesian_precomputed.metadata.csv")
overall_results <- debiasR::validate_flow_overall_methods(
  adjusted_outputs = all_method_results[level1_methods],
  benchmark_od_df = benchmark_val,
  comparisons = "adjusted_vs_benchmark"
)
write_csv(combine_overall(overall_results, level1_methods), "v07-validation.level1.overall_metrics.csv")
selected_results <- all_method_results[validation_methods]
origin_margins <- debiasR::validate_flow_margins(selected_results, benchmark_val, role = "origin")
origin_margin_metrics <- debiasR::validate_flow_overall_methods(origin_margins$adjusted_outputs, origin_margins$benchmark, comparisons = "adjusted_vs_benchmark")
destination_margins <- debiasR::validate_flow_margins(selected_results, benchmark_val, role = "destination")
destination_margin_metrics <- debiasR::validate_flow_overall_methods(destination_margins$adjusted_outputs, destination_margins$benchmark, comparisons = "adjusted_vs_benchmark")
write_csv(combine_overall(origin_margin_metrics, validation_methods), "v07-validation.level2.origin_margin_metrics.csv")
write_csv(combine_overall(destination_margin_metrics, validation_methods), "v07-validation.level2.destination_margin_metrics.csv")
flow_residuals <- debiasR::validate_flow_residual_methods(selected_results, benchmark_val, top_n = 12)
write_csv(combine_nested(flow_residuals, validation_methods, "summary"), "v07-validation.level3.residual_summary.csv")
write_csv(
  method_order_head(
    combine_nested(flow_residuals, validation_methods, "top_worst"),
    validation_methods,
    5
  ),
  "v07-validation.level3.top_worst.csv"
)
distribution_results <- debiasR::validate_flow_distribution_methods(selected_results, benchmark_val, comparisons = "adjusted_vs_benchmark")
write_csv(combine_nested(distribution_results, validation_methods, "summary"), "v07-validation.level4.distribution_summary.csv")
write_csv(
  method_order_head(
    combine_nested(distribution_results, validation_methods, "origin_level"),
    validation_methods,
    5
  ),
  "v07-validation.level4.origin_head.csv"
)
neighbors <- nearest_neighbors(distance_val, covariates_val$area, 4)
structure_covariates <- covariates_val |> dplyr::select(area, per_level4)
residual_structure <- debiasR::validate_flow_residual_structure_methods(
  adjusted_outputs = selected_results,
  benchmark_od_df = benchmark_val,
  comparison = "adjusted_vs_benchmark",
  spatial_role = "origin",
  residual_aggregation = "mean",
  area_neighbors = neighbors,
  local_moran = FALSE,
  covariate_df = structure_covariates,
  covariate_col = "per_level4",
  methods = validation_methods
)
write_csv(neighbor_summary(neighbors), "v07-validation.level5.neighbor_summary.csv")
write_csv(combine_nested(residual_structure, validation_methods, "summary"), "v07-validation.level5.structure_summary.csv")
write_csv(
  data.frame(
    plot = c("level1_metrics", "level2_origin_scatter", "level2_destination_scatter", "level3_flow_scatter", "level3_residual_bands", "level4_distribution_pairwise", "level5_structure"),
    source = c("level1.overall_metrics", "origin_margin_residuals", "destination_margin_residuals", "level3.residual_summary", "level3.residual_summary", "level4.distribution_summary", "level5.structure_summary"),
    random_sampling = FALSE,
    stringsAsFactors = FALSE
  ),
  "v07-validation.plot_contracts.csv"
)
write_page_meta("v07-validation", "validation calls executed without MCMC")

lad_data <- debiasR::debiasR_example_data(n_areas = Inf, complete_grid = TRUE, geography = "lad")
mpd_od_df <- lad_data$mpd_od
mpd_od_df$mpd_time <- "single_period"
coverage_df <- lad_data$coverage
coverage_df$mpd_time <- "single_period"
covariates_df <- lad_data$covariates
distance_df <- lad_data$distance
benchmark_od_df <- lad_data$benchmark_od
write_csv(
  object_counts(list(mpd_od = mpd_od_df, coverage = coverage_df, covariates = covariates_df, distance = distance_df, benchmark_od = benchmark_od_df)),
  "v09-advanced-bayesian-adjustment.data_counts.csv"
)
write_csv(utils::head(mpd_od_df, 6), "v09-advanced-bayesian-adjustment.mpd_od.head.csv")
write_csv(utils::head(coverage_df, 6), "v09-advanced-bayesian-adjustment.coverage.head.csv")
write_csv(utils::head(covariates_df, 6), "v09-advanced-bayesian-adjustment.covariates.head.csv")
write_csv(utils::head(distance_df, 6), "v09-advanced-bayesian-adjustment.distance.head.csv")
write_csv(utils::head(benchmark_od_df, 6), "v09-advanced-bayesian-adjustment.benchmark.head.csv")
adj_frequentist <- debiasR::adjust_multilevel_bayes(
  mpd_od_df = mpd_od_df,
  coverage_df = coverage_df,
  covariates_df = covariates_df,
  distance_df = distance_df,
  mobility_formula = ~ log_pop_o + log_pop_d + log_distance +
    rural_pct_o + rural_pct_d + per_level4_o + per_level4_d,
  bias_formula = ~ 0,
  target_scale = "true_flow",
  observation_model = "coverage_offset",
  coverage_scale = "origin",
  model_engine = "frequentist",
  scenario = "s1",
  source_col = "mpd_source",
  time_col = "mpd_time",
  repeated_observation = "none",
  prediction_scope = "complete_grid",
  random_intercept = "none",
  model_family = "poisson"
)
write_csv(
  utils::head(adj_frequentist[c("origin", "destination", "flow", "flow_adj", "flow_mpd_pred", "flow_true_pred", "observation_probability")], 6),
  "v09-advanced-bayesian-adjustment.coverage_offset_frequentist.head.csv"
)
adj_reduced_form_check <- debiasR::adjust_multilevel_bayes(
  mpd_od_df = mpd_od_df,
  coverage_df = coverage_df,
  covariates_df = covariates_df,
  distance_df = distance_df,
  mobility_formula = ~ log_pop_o + log_pop_d + log_distance +
    rural_pct_o + rural_pct_d + per_level4_o + per_level4_d,
  bias_formula = ~ bias_e_origin,
  target_scale = "mpd_counterfactual",
  observation_model = "reduced_form",
  model_engine = "frequentist",
  scenario = "s1",
  source_col = "mpd_source",
  time_col = "mpd_time",
  repeated_observation = "none",
  prediction_scope = "complete_grid",
  random_intercept = "none",
  model_family = "poisson"
)
write_csv(
  utils::head(adj_reduced_form_check[c("origin", "destination", "flow", "flow_adj", "flow_mpd_pred")], 6),
  "v09-advanced-bayesian-adjustment.reduced_form_frequentist.head.csv"
)
write_csv(read_extdata("v09-bayesian-real-data-output.csv"), "v09-advanced-bayesian-adjustment.bayesian_precomputed.output.csv")
write_csv(read_extdata("v09-bayesian-real-data-summary.csv"), "v09-advanced-bayesian-adjustment.bayesian_precomputed.summary.csv")
write_csv(read_extdata("v09-bayesian-real-data-diagnostics.csv"), "v09-advanced-bayesian-adjustment.bayesian_precomputed.diagnostics.csv")
write_csv(bayesian_backend_contract(), "v09-advanced-bayesian-adjustment.bayesian_backend_contract.csv")
write_page_meta("v09-advanced-bayesian-adjustment", "model_engine='frequentist' maximum-likelihood API-selector calls plus static Python backend inventory")


message("Wrote fresh R article outputs to ", goldens_tmp)
