source("tests/r/reference.R")
set.seed(314159)

matrix_cases <- lapply(c(1:8, 12, 20), function(n) {
  a <- matrix(rnorm(n * n), n, n)
  d <- lav("lav_mat_dup", n)
  dp <- lav("lav_mat_dup_ginv", n)
  b <- matrix(rnorm(n^4), n^2, n^2)
  x <- matrix(rnorm(n * (n + 2)), n, n + 2)
  s <- crossprod(a) + diag(n)
  half <- lapply(c(TRUE, FALSE), function(diagonal) {
    orders <- c("vech", "vechr", "vechu", "vechru")
    list(diagonal = diagonal,
         indices = lapply(orders, function(o) as.vector(lav(paste0("lav_mat_", o, "_idx"), n, diagonal)) - 1L),
         values = lapply(orders, function(o) as.vector(lav(paste0("lav_mat_", o), a, diagonal))),
         reverses = lapply(orders, function(o) mat(lav(paste0("lav_mat_", o, "_rev"), lav(paste0("lav_mat_", o), a, diagonal), diagonal))))
  })
  list(n = n, a = mat(a), b = mat(b), x = mat(x), s = mat(s),
       vec = as.vector(a), vecr = as.vector(t(a)), half = half,
       diag = lav("lav_mat_diag_idx", n) - 1L,
       diagh = lav("lav_mat_diagh_idx", n) - 1L,
       antidiag = lav("lav_mat_antidiag_idx", n) - 1L,
       duplication = mat(d), duplication_inverse = mat(dp),
       dup_pre = mat(lav("lav_mat_dup_pre", b)), dup_post = mat(lav("lav_mat_dup_post", b)),
       dup_both = mat(lav("lav_mat_dup_pre_post", b)),
       ginv_pre = mat(lav("lav_mat_dup_ginv_pre", b)), ginv_post = mat(lav("lav_mat_dup_ginv_post", b)),
       ginv_both = mat(lav("lav_mat_dup_ginv_pre_post", b)),
       commutation = mat(lav("lav_mat_com", n, n + 2)),
       com_pre = mat(lav("lav_mat_com_pre", b)), com_post = mat(lav("lav_mat_com_post", b)),
       com_both = mat(lav("lav_mat_com_pre_post", b)),
       sqrt = mat(lav("lav_mat_sym_sqrt", s)),
       inverse = mat(lav("lav_mat_sym_inverse", s, sinv_method = "chol")),
       logdet = as.numeric(determinant(s, logarithm = TRUE)$modulus),
       cor = mat(cov2cor(s)),
       cor2cov = mat(lav("lav_cor2cov", cov2cor(s), sqrt(diag(s)))),
       partial_cor = mat(lav("lav_cov2cor_partial", s, c(1L, n))),
       bdiag = mat(lav("lav_mat_bdiag", a, x)),
       trace_product = lav("lav_mat_trace", a, x, t(x)))
})

sample_cases <- lapply(c(1:7, 12), function(p) {
  n <- 41
  y <- matrix(rnorm(n * p), n, p)
  a <- matrix(rnorm(p * p), p, p)
  sigma <- crossprod(a) + diag(p)
  mu <- rnorm(p)
  weights <- runif(n, 0.2, 3)
  weights[c(2, 8)] <- 0
  lapply(c(FALSE, TRUE), function(weighted) {
    wt <- if (weighted) weights else NULL
    stats <- lav("lav_mvn_samp_stats", y, wt)
    fixed <- if (p > 2) c(1L, p) else if (p == 2) 1L else integer(0)
    variant <- lapply(list(integer(0), fixed), function(fixed) {
      list(fixed = fixed - 1L,
           log_density = as.vector(lav("lav_mvn_loglik_data", y, wt, mu, sigma, casewise = TRUE, x_idx = fixed)),
           log_likelihood = lav("lav_mvn_loglik_data", y, wt, mu, sigma, x_idx = fixed),
           scores = mat(lav("lav_mvn_sc_mu_sigma", y, wt, mu, sigma, x_idx = fixed)),
           gradient = as.vector(lav("lav_mvn_dlogl_dmu_dvechsigma", y, wt, mu, sigma, x_idx = fixed)),
           expected = mat(lav("lav_mvn_info_expected", sigma_1 = sigma, x_idx = fixed)),
           observed = mat(lav("lav_mvn_info_observed_data", y, wt, mu, sigma, x_idx = fixed)),
           inverse_expected = mat(lav("lav_mvn_inv_info_expected", sigma_1 = sigma, x_idx = fixed)),
           first_order = mat(lav("lav_mvn_info_firstorder", y, wt, mu = mu, sigma_1 = sigma, x_idx = fixed)),
           cluster = mat(lav("lav_mvn_info_firstorder", y, wt, cluster_idx = rep(1:9, length.out = n), mu = mu, sigma_1 = sigma, x_idx = fixed)))
    })
    s <- stats$cov
    attr(sigma, "inv") <- solve(sigma)
    attr(sigma, "log.det") <- as.numeric(determinant(sigma, logarithm = TRUE)$modulus)
    attr(sigma, "po") <- TRUE
    list(y = mat(y), weights = wt, mean = as.vector(stats$mean), cov = mat(s),
         nobs = if (weighted) sum(weights) else n,
         mu = as.vector(mu), sigma = mat(sigma), variants = variant,
         variance_ml = as.vector(lav("lav_mat_var_wt", y, wt, "ML")),
         variance_unbiased = as.vector(lav("lav_mat_var_wt", y, wt, "unbiased")),
         ml = lav("lav_model_objective_ml", sigma, mu, s, stats$mean, as.numeric(determinant(s, logarithm = TRUE)$modulus), TRUE),
         gls = lav("lav_model_objective_gls", sigma, mu, s, solve(s), stats$mean, TRUE),
         gls_cor = lav("lav_model_objective_gls", sigma, mu, s, solve(s), stats$mean, TRUE, TRUE),
         expected_cor = mat(lav("lav_mvn_info_expected", sigma_1 = sigma, meanstructure = FALSE, correlation = TRUE)),
         wls = lav("lav_model_objective_wls", as.vector(mu), as.vector(stats$mean), sigma),
         dwls = lav("lav_model_objective_dwls", as.vector(mu), as.vector(stats$mean), diag(sigma)),
         transformed = mat(lav("lav_mat_transform_mean_cov", y, mu, sigma)))
  })
})
sample_cases <- unlist(sample_cases, recursive = FALSE)

fit_inputs <- rbind(
  c(85.3, 24, 918.8, 36, 1, 1), c(12, 24, 918, 36, 1, 1),
  c(0, 0, 0, 0, 1, 1), c(1e-10, 0, 1e-9, 0, 1, 1),
  c(24, 24, 36, 36, 1, 1), c(300, 24, 100, 36, 1, 1),
  c(60, 40, 100, 36, 1, 1), c(0, -1, 50, 36, 1, 1),
  c(85, 24, 900, 36, 1.3, 1.5), c(85, 24, 900, 36, 1.3, 1),
  c(85, 0, 900, 36, 1.3, 1.5), c(Inf, 24, 900, 36, 1, 1),
  c(85, 24, Inf, 36, 1, 1), c(NA, 24, 900, 36, 1, 1),
  c(-5, 24, -8, 36, 1, 1), c(24, 24, 24 + 1e-10, 36, 1, 1),
  cbind(runif(150, 0, 500), sample(0:50, 150, TRUE), runif(150, 0, 1500), sample(0:80, 150, TRUE), runif(150, 0.8, 2), runif(150, 0.8, 2)))
fit_cases <- lapply(seq_len(nrow(fit_inputs)), function(i) {
  x <- fit_inputs[i, ]
  suppressWarnings(list(input = as.vector(x),
       cfi = lav("lav_fit_cfi", x[1], x[2], x[3], x[4], x[5], x[6]),
       rni = lav("lav_fit_rni", x[1], x[2], x[3], x[4], x[5], x[6]),
       tli = lav("lav_fit_tli", x[1], x[2], x[3], x[4], x[5], x[6]),
       rfi = lav("lav_fit_rfi", x[1], x[2], x[3], x[4]),
       nfi = lav("lav_fit_nfi", x[1], x[2], x[3], x[4]),
       pnfi = lav("lav_fit_pnfi", x[1], x[2], x[3], x[4]),
       ifi = lav("lav_fit_ifi", x[1], x[2], x[3], x[4]),
       rmsea = lav("lav_fit_rmsea", x[1], x[2], n = 301, g = 2, c_hat = x[5]),
       aic = lav("lav_fit_aic", -200, x[2]),
       bic = lav("lav_fit_bic", -200, x[2], 301),
       sabic = lav("lav_fit_sabic", -200, x[2], 301)))
})

misc <- list()
for (n in c(1, 3, 7)) {
  a <- matrix(0, n, n)
  a[lower.tri(a)] <- rnorm(n * (n - 1) / 2)
  misc[[length(misc) + 1L]] <- list(a = mat(a), inverse_iminus = mat(lav("lav_mat_inverse_iminus", a)))
  b <- matrix(rnorm(n^2, sd = 0.07), n, n)
  misc[[length(misc) + 1L]] <- list(a = mat(b), inverse_iminus = mat(lav("lav_mat_inverse_iminus", b)))
}
missing_cases <- lapply(c(2, 3, 5, 8), function(p) {
  y <- matrix(rnorm(40 * p), 40, p)
  y[runif(length(y)) < 0.25] <- NA_real_
  y[1, ] <- NA_real_
  a <- matrix(rnorm(p^2), p, p)
  sigma <- crossprod(a) + diag(p)
  mu <- rnorm(p)
  lapply(c(FALSE, TRUE), function(weighted) {
    wt <- if (weighted) runif(40, 0.3, 2) else NULL
    if (weighted) wt[5] <- 0
    variants <- lapply(list(integer(0), 1L), function(fixed) {
      sc <- lav("lav_mvn_mi_sc_mu_sigma", y = y, wt = wt, mu = mu, sigma_1 = sigma, x_idx = fixed)
      list(fixed = fixed - 1L, scores = mat(sc), gradient = as.vector(colSums(sc, na.rm = TRUE)),
           expected = mat(lav("lav_mvn_mi_info_expected", y = y, wt = wt, sigma_1 = sigma, x_idx = fixed)),
           observed = mat(lav("lav_mvn_mi_info_observed_data", y = y, wt = wt, mu = mu, sigma_1 = sigma, x_idx = fixed)),
           first_order = mat(lav("lav_mvn_mi_info_firstorder", y = y, wt = wt, mu = mu, sigma_1 = sigma, x_idx = fixed)),
           clustered = mat(lav("lav_mvn_mi_info_firstorder", y = y, wt = wt, cluster_idx = rep(1:7, length.out = 40), mu = mu, sigma_1 = sigma, x_idx = fixed)))
    })
    estep <- lav("lav_mvn_mi_estep", y = y, wt = wt, mu = mu, sigma_1 = sigma)
    list(y = mat(y), weights = wt, sigma = mat(sigma), mu = as.vector(mu), variants = variants,
         log_density = as.vector(lav("lav_mvn_mi_loglik_data", y = y, wt = wt, mu = mu, sigma_1 = sigma, casewise = TRUE)),
         log_likelihood = lav("lav_mvn_mi_loglik_data", y = y, wt = wt, mu = mu, sigma_1 = sigma),
         imputed = mat(lav("lav_mvn_mi_impute_pattern", y = y, mu = mu, sigma_1 = sigma)),
         t1 = as.vector(estep$T1), t2 = mat(estep$T2),
         missing_variance = as.vector(lav("lav_mat_var_wt", y, wt, "ML")))
  })
})
variance_edges <- lapply(list(NULL, c(0,1,1)), function(wt) {
  y <- matrix(c(NA,NA,NA,1,NA,NA,NA,2,4),3,3)
  list(y = mat(y), weights = wt,
       ml = as.vector(lav("lav_mat_var_wt",y,wt,"ML")),
       unbiased = as.vector(lav("lav_mat_var_wt",y,wt,"unbiased")))
})
shifted <- matrix(c(1,2,5,4,7,3,2,5,6,3,0,1),6,2) + rep(c(1e12,-1e12), each = 6)
shifted_weights <- c(0,0.2,1,2,0.5,1)
shifted_weighted <- lav("lav_mvn_samp_stats",shifted,shifted_weights)
singular <- diag(1,3)
misc[[length(misc)+1L]] <- list(a = mat(singular), inverse_iminus = mat(suppressWarnings(lav("lav_mat_inverse_iminus",singular))))
emit(list(version = expected_version, matrices = matrix_cases, samples = sample_cases, fits = fit_cases, misc = misc, missing = unlist(missing_cases, recursive = FALSE),
          variance_edges = variance_edges, shifted = list(y = mat(shifted), mean = as.vector(colMeans(shifted)), covariance = mat(lav("lav_mat_cov",shifted)), weights = shifted_weights, weighted_mean = as.vector(shifted_weighted$mean), weighted_covariance = mat(shifted_weighted$cov)),
          indefinite_sqrt = list(a = mat(diag(c(-1,0,4))), root = mat(lav("lav_mat_sym_sqrt",diag(c(-1,0,4)))))))
