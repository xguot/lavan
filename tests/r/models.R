source("tests/r/reference.R")
data("HolzingerSwineford1939", package = "lavaan")
data("PoliticalDemocracy", package = "lavaan")
data("Demo.growth", package = "lavaan")
hs_model <- "visual =~ x1 + x2 + x3
             textual =~ x4 + x5 + x6
             speed =~ x7 + x8 + x9"
political_model <- "ind60 =~ x1 + x2 + x3
                   dem60 =~ y1 + y2 + y3 + y4
                   dem65 =~ y5 + y6 + y7 + y8
                   dem60 ~ ind60
                   dem65 ~ ind60 + dem60
                   y1 ~~ y5
                   y2 ~~ y4 + y6
                   y3 ~~ y7
                   y4 ~~ y8
                   y6 ~~ y8"
hs_missing <- HolzingerSwineford1939
for (j in 1:9) hs_missing[seq.int(j, nrow(hs_missing), by = 13), paste0("x",j)] <- NA_real_
fits <- list(
  hs_cfa = cfa(hs_model, HolzingerSwineford1939, information = "expected"),
  hs_means = cfa(hs_model, HolzingerSwineford1939, meanstructure = TRUE, information = "expected"),
  growth = growth("i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4
                   s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4", Demo.growth, information = "expected"),
  observed_path = sem("y2 ~ y1 + x1 + x2
                       y3 ~ y2 + x1", PoliticalDemocracy, fixed.x = FALSE, information = "expected"),
  political_sem = sem(political_model, PoliticalDemocracy, information = "expected"),
  hs_missing = cfa(hs_model, hs_missing, missing = "ml", meanstructure = TRUE, information = "expected")
)
cases <- lapply(names(fits), function(name) {
  fit <- fits[[name]]
  stopifnot(lavInspect(fit, "converged"))
  start <- lavInspect(fit, "start")
  free <- lavInspect(fit, "free")
  est <- lavInspect(fit, "est")
  delta <- lavInspect(fit, "delta")
  samp <- lavInspect(fit, "sampstat")
  source_data <- switch(name, hs_cfa = HolzingerSwineford1939, hs_means = HolzingerSwineford1939,
                         growth = Demo.growth, hs_missing = hs_missing, PoliticalDemocracy)
  sample_mean <- if (is.null(samp$mean)) colMeans(source_data[, lavNames(fit, "ov"), drop = FALSE]) else samp$mean
  measures <- fitMeasures(fit, c("chisq", "df", "baseline.chisq", "baseline.df", "cfi", "tli", "rmsea", "logl", "aic", "bic", "srmr"))
  list(name = name, meanstructure = fit@Options$meanstructure,
       missing_data = if (name == "hs_missing") mat(as.matrix(source_data[, lavNames(fit,"ov")])) else NULL,
       baseline_covariates = match(lavNames(fit, "ov.x"), lavNames(fit, "ov")) - 1L,
       start = lapply(start, mat), free = lapply(free, mat), estimated = lapply(est, mat),
       covariance = mat(samp$cov), sample_mean = as.vector(sample_mean), nobs = lavInspect(fit, "nobs"),
       coefficients = as.vector(coef(fit)), parameter_names = names(coef(fit)),
       jacobian = mat(delta), vcov = mat(vcov(fit)), measures = as.list(measures))
})
emit(cases)
