source("tests/r/reference.R")
args <- commandArgs(trailingOnly = TRUE)
input <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
read_mat <- function(x) matrix(unlist(x$data), x$rows, x$cols)
y <- read_mat(input$y)
sigma <- read_mat(input$sigma)
mu <- unlist(input$mu)
b <- read_mat(input$b)
ns <- asNamespace("lavaan")
sample_stats <- get("lav_mvn_samp_stats", ns)
loglik <- get("lav_mvn_loglik_data", ns)
scores <- get("lav_mvn_sc_mu_sigma", ns)
info <- get("lav_mvn_info_expected", ns)
dup <- get("lav_mat_dup_pre_post", ns)
implied <- get("lav_lisrel_sigma", ns)
precision <- solve(sigma)
mlist <- lapply(input[c("lambda", "theta", "psi", "beta")], read_mat)
measure <- function(name, operation) {
  for (i in seq_len(3)) operation()
  reps <- 1L
  repeat {
    start <- proc.time()[["elapsed"]]
    for (i in seq_len(reps)) checksum <- operation()
    if (proc.time()[["elapsed"]] - start >= 0.06) break
    reps <- reps * 2L
  }
  times <- numeric(7)
  for (sample in seq_len(7)) {
    start <- proc.time()[["elapsed"]]
    for (i in seq_len(reps)) checksum <- operation()
    times[sample] <- (proc.time()[["elapsed"]] - start) * 1e6 / reps
  }
  list(name = name, samples_us = times, median_us = median(times), repetitions = reps, checksum = checksum)
}
results <- list(
  measure("sample_statistics", function() sample_stats(y)$cov[1, 1]),
  measure("normal_log_likelihood", function() loglik(y, mu = mu, sigma_1 = sigma, sinv_method = "chol")),
  measure("normal_scores", function() scores(y, mu = mu, sigma_1 = sigma, sinv_method = "chol")[1, 1]),
  measure("expected_information", function() info(sigma_inv = precision)[1, 1]),
  measure("duplication_pre_post", function() dup(b)[1, 1]),
  measure("lisrel_implied_covariance", function() implied(mlist)[1, 1]))
emit(list(implementation = "R lavaan", version = expected_version, R = R.version.string,
          BLAS = extSoftVersion()[["BLAS"]], results = results))
