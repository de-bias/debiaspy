# Generate fixed oracle tables without altering the reference checkout.
# Rscript scripts/generate_selection_reference.R /path/to/debiasR selection_rate
# Requires dplyr and jsonlite for fixture generation, not for Python tests.
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(2L, 3L) ||
    !args[[2]] %in% c("selection_rate", "selection_rate2")) {
  stop("Usage: Rscript scripts/generate_selection_reference.R R_ROOT METHOD [OUTPUT_JSON]")
}
root <- normalizePath(args[[1]], mustWork = TRUE)
method <- args[[2]]
reference <- "6fa2709ae42de6dc525a4a46182fdfb448b14a3f"
git_args <- c("--no-optional-locks", "-C", shQuote(root))
head <- system2("git", c(git_args, "rev-parse", "HEAD"), stdout = TRUE)
dirty <- system2("git", c(git_args, "status", "--porcelain"), stdout = TRUE)
if (!identical(head, reference) || length(dirty)) stop("Reference must be pinned and clean.")
suppressPackageStartupMessages(library(dplyr))
source(file.path(root, "R", paste0("adjust_", method, ".R")))
input_path <- file.path("tests", "fixtures", paste0(method, "_inputs.json"))
cases <- jsonlite::fromJSON(input_path, simplifyVector = TRUE)
parameter <- if (method == "selection_rate") "r_global" else "k"
diagnostic <- if (method == "selection_rate") "r_calibration" else "k_calibration"
for (name in names(cases)) {
  case <- cases[[name]]
  call_args <- case$args
  if (is.null(call_args)) call_args <- list()
  for (entry in c("mpd", "coverage", "covariates", "benchmark")) {
    if (!is.null(case[[entry]])) {
      target <- switch(entry, mpd = "mpd_od_df", coverage = "coverage_df",
                       covariates = "covariates_df", benchmark = "benchmark_od_df")
      call_args[[target]] <- as.data.frame(case[[entry]], stringsAsFactors = FALSE)
    }
  }
  if (!is.null(call_args$k_grid)) {
    call_args$k_grid <- as.numeric(call_args$k_grid)
  }
  modes <- if (is.null(case$modes)) c("origin", "destination", "both") else case$modes
  expected <- list()
  for (mode in modes) {
    call_args$weight_by <- mode
    messages <- character()
    result <- withCallingHandlers(
      do.call(get(paste0("adjust_", method)), call_args),
      warning = function(w) {
        messages <<- c(messages, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    expected[[mode]] <- list(
      columns = names(result), data = as.data.frame(result),
      parameter = attr(result, parameter), calibration = attr(result, diagnostic),
      many_to_many = sum(grepl("many-to-many", messages)),
      coercion_warnings = sum(grepl("NAs introduced by coercion", messages)),
      warnings = messages
    )
  }
  cases[[name]]$expected <- expected
}
output <- list(
  metadata = list(
    reference = reference, function_name = paste0("adjust_", method),
    r_version = R.version.string,
    dplyr = as.character(packageVersion("dplyr")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  cases = cases
)
output_path <- file.path("tests", "fixtures", paste0(method, "_r.json"))
if (length(args) == 3L) output_path <- args[[3]]
jsonlite::write_json(output, output_path,
  dataframe = "rows", auto_unbox = TRUE, na = "null", digits = 17, pretty = TRUE
)
