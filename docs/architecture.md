The crate separates model semantics from numerical algebra and inference.
`reference/lavaan` is read-only input to the port, oracle, and inventory.

| Module | Responsibility |
| --- | --- |
| `indexing` | Production integer kernels shared verbatim with Aeneas extraction |
| `matrix` | Vectorization, duplication, commutation, covariance transforms, SPD solves, spectral roots and inverse(I-A) |
| `sample` | ML moments, compensated sums, weighted variances and moment transforms |
| `mvnorm` | Reusable Gaussian likelihoods, scores and unit information |
| `missing` | Observed patterns, marginal likelihoods, scores, information, imputation, E-step and saturated EM |
| `objective` | ML, GLS, WLS and DWLS discrepancy functions |
| `model` | Numeric LISREL, free/shared parameter slots, analytic Jacobians, ML/FIML and expected-information covariance |
| `cfa` | Bounded syntax, CFA defaults, column matching and complete-data fitting |
| `optim` | Independent BFGS, Armijo search and explicit convergence status |
| `fit` | Scalar fit and information criteria |

`Lisrel` stores lambda, theta, psi, beta, nu and alpha. Implied moments are
`T = lambda (I-beta)^(-1)`, `Sigma = T psi T' + theta`, and `mu = nu + T alpha`.
A `Parameter` owns one or more `Slot`s; symmetric covariance slots write both
positions. Shared slots express equality without redundant optimization variables.
Different parameters cannot write the same slot.

Analytic directional derivatives map moment gradients and unit information to
free parameters. Inference uses `N J' I J`, keeping sample-size scaling explicit.
Failed convergence and unavailable covariance are visible in `Fit`. A returned
parameter covariance does not certify model identification.

The optimizer differs from R's nlminb. Tests compare converged statistical
outputs within documented tolerances; trajectories and iteration counts differ.
Numeric models must supply baseline observed-covariate indices explicitly:
the matrices alone do not encode that semantic distinction.

The runtime has no R dependency, unsafe Rust, external BLAS requirement, or
mutable global state. Implicit operator products and direct information formulas
reduce allocations. Python and R are development dependencies.
