The port is incomplete. `reports/coverage.json` inventories every indexed
top-level reference function/alias and every namespace export, with unsupported
as the default. `verification/components.json` supplies explicit support overrides.
Counts include aliases and wrappers; they are not a completeness score. The R
AST inventory currently indexes 1,707 definitions across 248 files and 159,455
lines, plus 236 namespace exports.

| Area | Implemented domain | Important gaps |
| --- | --- | --- |
| CFA | Complete continuous single group, normal ML; `=~`, `~~`, `~ 1`; numeric, NA and label modifiers; marker scaling, std.lv, orthogonal, means, shared labels | Full grammar/defaults, higher-order syntax, latent-observed covariance syntax, growth/SEM string entry points |
| Numeric models | Continuous LISREL, beta regressions, fixed/free entries, means, shared parameters, ML and pattern-wise FIML | Categorical delta, composites, exogenous conditioning, multigroup/multilevel semantics, nonlinear/inequality constraints |
| Inference | Chi-square statistic, df, baseline with supplied covariates, expected-information SE, CFI/TLI/RNI/RFI/NFI/PNFI/IFI, RMSEA point estimate, AIC/BIC/SABIC | P-values, CIs/close-fit tests, robust corrections/SE, observed model information, SRMR/GFI, standardized solutions, modification indices and model comparison |
| Matrices | Vectorization/reconstruction, triangular indices, duplication/pseudoinverse products, commutation, block diagonals, traces, square roots, strictly SPD solves, inverse(I-A) | General symmetric indefinite/eigen inversion and preconditioning, orthogonal complement, row reduction, inverse downdates and other helpers |
| Moments | ML divisor N, complete moments, frequency weights, covariance about a supplied mean, weighted missing-value variances, target-moment transform | Sampling-weight design inference, R coercion/recycling, pairwise sample covariances |
| Gaussian kernels | Raw/statistic/casewise likelihoods; mean/vech scores; expected, observed, inverse expected and first-order information; documented fixed-x and cluster branches | Singular covariance, eigen inverse-method branches, external x moments, all helper APIs |
| Missing data | Patterns, likelihoods, scores, information, imputation, E-step, unrestricted EM; numeric FIML with means | Conditional/fixed-x likelihoods, missing baseline covariates, EM acceleration, all-missing datasets, full option system |
| Objectives | ML, GLS with correlation correction, WLS/DWLS scalar discrepancies | Full GLS/WLS/DWLS fitting pipelines, REML, PML, MML, DLS, categorical and multilevel objectives |

Bootstrap, simulation, EFA/rotations, SAM, MIIV, categorical/multilevel estimation,
Bayesian/external backends, plotting/export, S3/S4 behavior, list/multiple
imputation workflows, prediction/factor scores, inspection compatibility, and
most parameter-table/options utilities remain unsupported. Everything omitted
from the table and inventory overrides is unsupported.

Rust rejects incompatible shapes, duplicate slots, non-finite complete data,
invalid weights, and unsupported front-end syntax. These errors differ from
some R coercion, warning, NaN and recovery behavior. NaN represents both R NA
and NaN; their payload distinction is not preserved. Matrix names and R classes
are not reproduced. The spectral square root follows R's truncation of every
negative eigenvalue, including substantively negative ones.

Regular SPD Gaussian models are the validated numerical domain. Extreme scaling,
near singularity, weak identification, singular beta systems, nonunique optima,
boundary solutions, and arbitrary adversarial floating-point inputs have not
been validated comprehensively. Expected-information inversion may return None;
a returned covariance does not establish identification. Warning parity,
Heywood-case diagnostics and R optimizer failure semantics are gaps.

Kernel comparisons use `3e-10 * max(1, abs(R))`; recorded errors are in
`reports/validation.json`. Numeric model parameter comparisons use `2e-5`,
chi-square `1e-8`, parameter covariance `3e-5`, and analytic Jacobians `3e-10`
on the same relative/absolute scale. CFA tests use `2e-7` optimizer gradient
tolerance and numeric model tests `1e-7`; the default optimizer uses `1e-7`.
These are test tolerances, not accuracy
guarantees for untested inputs. The scalar comparison count includes many zero
operator entries; it is not a count of independent statistical models.
