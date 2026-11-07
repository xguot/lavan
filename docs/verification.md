There are three distinct levels of evidence. The port has passing examples at
each level and has no whole-program correctness proof.

| Evidence | Scope | Check |
| --- | --- | --- |
| R equivalence | Matrix, sample, Gaussian, missing-data, objective and fit fixtures; six numeric models and six CFA variants | `cargo test --locked -- --nocapture` |
| Lean/Mathlib | 14 exact integer/real/matrix theorems | `lake build` in `verification` |
| Aeneas Rust refinement | Two production usize functions, three theorems | `python3 scripts/extract_rust.py --check` |

The reference installer stamps DESCRIPTION, NAMESPACE, all R files and bundled
datasets. Tests check the current source, installed version and workspace library.
They never substitute CRAN or global lavaan. The inventory additionally records
SHA256 source digests. R runs in fresh processes.

Oracle JSON uses 17 decimal digits and Rust's round-trip float parser. A large-
location mean fixture checks exact transmission, avoiding an oracle transport
error being mistaken for numerical algorithm disagreement.

The main proof project pins Lean/Mathlib 4.24.0. Its theorems cover column-index
injectivity/bounds, symmetric half reconstruction, duplication projection and
left-inverse identities, transpose involution, covariance shift, two-sided finite
nilpotent inverses, CFI bounds, RMSEA nonnegativity/zero behavior, and a univariate
Gaussian variance-score identity. These are exact-arithmetic specifications;
they establish no `f64` error bound. The CFI proof excludes R's approximate-zero
branch, which is covered by numerical tests.

The extracted project pins Aeneas `nightly-2026.11.06-4acbb6d`, its bundled Charon,
Rust `nightly-2026-09-17`, and Lean/Mathlib 4.35.0. Its Rust crate compiles
`../../src/indexing.rs` directly, the same source used by the runtime. Regeneration
rejects changed generated Lean before checking proofs. `column_index_spec`,
`column_index_bounded` and `transpose_index_spec` prove successful checked
arithmetic, correct output and range bounds under explicit preconditions. They
do not prove surrounding loops, dimension validation, allocation, matrix routines
or compiled machine code.

Both projects include `Audit.lean` to print every theorem's axiom dependencies.
Validation permits only Lean's `propext`, `Classical.choice` and `Quot.sound`;
`sorryAx` and other axioms fail the audit. There are no admitted proofs or custom
axioms in these results. The trusted base includes Lean's kernel, Mathlib and
Aeneas integer models, the Charon/Aeneas translators and the Rust compiler.

BFGS, EM convergence, the parser/defaults, statistical asymptotics, Cholesky/eigen/
SVD implementations, conditioning, Jacobian refinement, identification and
floating-point error bounds remain unproved. Finite differences independently
check Gaussian gradients and observed information. R checks model Jacobians and
fitted outputs. Convergence is tested rather than guaranteed for arbitrary models.

Setup follows the [official Aeneas workflow](https://github.com/AeneasVerif/aeneas#usage).
Asset names and SHA256 digests are pinned in `verification/tools.json`. A clean
host and CI execution have not yet been demonstrated here.

`scripts/validate.py` runs formatting, lint, tests, inventory, proof builds,
extraction regeneration and axiom audits. A success report is written only after
all checks pass. Logs are in `target/validation`; `reports/validation.json` records
source hashes and evidence from the last run.

Benchmarks use identical serialized inputs, three warmups, calibration to at
least 60 ms and seven wall-time samples. Reports contain all samples, medians,
versions, dimensions, input/source hashes and a scalar output sentinel check.
Input parsing, startup and fixture creation are excluded. Rust uses release
optimization and no external BLAS; R's BLAS is recorded. Results concern six
kernels on one machine; no whole-model speedup is established.
