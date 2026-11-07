source("tests/r/reference.R")
data("HolzingerSwineford1939", package = "lavaan")
syntax <- "visual =~ x1 + x2 + x3
           textual =~ x4 + x5 + x6
           speed =~ x7 + x8 + x9"
specs <- list(
  list(name = "default", syntax = syntax, meanstructure = FALSE, std_lv = FALSE, orthogonal = FALSE),
  list(name = "means", syntax = syntax, meanstructure = TRUE, std_lv = FALSE, orthogonal = FALSE),
  list(name = "std_lv", syntax = syntax, meanstructure = FALSE, std_lv = TRUE, orthogonal = FALSE),
  list(name = "orthogonal", syntax = syntax, meanstructure = FALSE, std_lv = FALSE, orthogonal = TRUE),
  list(name = "residual_and_equality", syntax = paste0(syntax, "\nx1 ~~ x2\nx4 ~~ equal*x4\nx5 ~~ equal*x5"), meanstructure = FALSE, std_lv = FALSE, orthogonal = FALSE),
  list(name = "fixed", syntax = "visual =~ 1*x1 + 0.8*x2 + 1.2*x3\nx2 ~~ 0.5*x2", meanstructure = FALSE, std_lv = FALSE, orthogonal = FALSE)
)
cases <- lapply(specs, function(spec) {
  fit <- cfa(spec$syntax, HolzingerSwineford1939, meanstructure = spec$meanstructure, std.lv = spec$std_lv, orthogonal = spec$orthogonal, information = "expected")
  stopifnot(lavInspect(fit, "converged"))
  columns <- paste0("x", 9:1)
  y <- as.matrix(HolzingerSwineford1939[, columns])
  est <- lavInspect(fit, "est")
  implied <- lavInspect(fit, "implied")
  list(spec = spec, y = mat(y), columns = columns, covariance = mat(implied$cov),
       mean = if (!is.null(implied$mean)) as.vector(implied$mean) else NULL,
       estimated = lapply(est, mat), chi_square = fitMeasures(fit, "chisq"),
       df = fitMeasures(fit, "df"), log_likelihood = as.numeric(logLik(fit)))
})
emit(cases)
