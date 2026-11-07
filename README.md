# lavan

An incremental Rust translation of the local `reference/lavaan` R package.
**The entire package has not been ported.** Current support includes a bounded
CFA front end, continuous numeric LISREL models, ML and Gaussian FIML fitting,
matrix kernels, Gaussian likelihoods and information, missing-data moments,
ML/GLS/WLS/DWLS objective functions, and selected scalar fit indices.

The reference is lavaan `0.7-3.3395`, commit
`7ec748e77ad301eecd2c37d8f16e15dae756c606`. Rust never delegates computation to R.
R is required for development equivalence tests and benchmarks.

Clone with `git clone --recurse-submodules https://github.com/xguot/lavan.git`.
For an existing checkout, run `git submodule update --init --recursive` to obtain
the pinned reference before preparing the R oracle.

```rust
use lavan::{cfa, Matrix};

fn fit(data: &Matrix) -> lavan::Result<()> {
    let result = cfa("f =~ x1 + x2 + x3 + x4", data,
                     &["x1", "x2", "x3", "x4"], Default::default())?;
    assert!(result.fit.solution.converged());
    println!("chi-square={}, df={}", result.fit.chi_square, result.fit.df);
    Ok(())
}
```

Run `cargo run --release --example cfa` for a self-contained example. Numerical
models with regressions, fixed entries, and shared parameters use `model::Model`;
`Model::fit_missing` accepts explicitly prepared `missing::MissingData`.

Install Rust, R, Python 3.11+, and R packages `MASS`, `mnormt`, `pbivnorm`,
`numDeriv`, `quadprog`, and `jsonlite`. Then run from this repository:

```sh
Rscript --vanilla scripts/prepare_reference.R
cargo test --locked -- --nocapture
cargo clippy --locked --all-targets -- -D warnings
python3 scripts/inventory.py --check
python3 scripts/benchmark.py
```

Equivalence tests load the locally installed reference separately from global R
libraries. They fail when the oracle is missing, stale, or has a different
version; there is no silent skip. Runtime matrices are column-major `f64`,
and Rust indices are zero based.

For proofs, install `elan` and `rustup`, then run:

```sh
python3 scripts/setup_verification.py
export ELAN_HOME="$PWD/.tools/elan"
python3 scripts/validate.py
```

Setup installs pinned tools in `.tools`, verifies the Aeneas archive SHA256,
and supports macOS arm64 and Linux arm64/x86_64. Only macOS arm64 has been
executed here; other platforms remain unvalidated. Initial setup downloads
substantial compiler and proof caches.

See [support and gaps](docs/support.md), [verification scope](docs/verification.md),
[architecture](docs/architecture.md), and [remaining work](docs/roadmap.md).
Machine-readable evidence lives in [validation](reports/validation.json),
[reference inventory](reports/coverage.json), and [benchmarks](reports/benchmarks.json).
Numerical comparisons, mathematical proofs, and extracted Rust proofs are
distinct evidence; none establishes whole-package equivalence.

GPL-3.0-or-later. See [NOTICE](NOTICE) for source attribution.
