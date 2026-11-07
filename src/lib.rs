//! Incremental Rust translation of the local `reference/lavaan` implementation.
//!
//! See `docs/support.md` and `docs/verification.md` for the supported domain and
//! the distinction between numerical comparisons and mathematical proofs.

pub mod cfa;
pub mod error;
pub mod fit;
pub mod indexing;
pub mod matrix;
pub mod missing;
pub mod model;
pub mod mvnorm;
pub mod objective;
pub mod optim;
pub mod sample;

pub use cfa::{CfaFit, CfaOptions, cfa};
pub use error::{Error, Result};
pub type Matrix = nalgebra::DMatrix<f64>;
pub type Vector = nalgebra::DVector<f64>;
