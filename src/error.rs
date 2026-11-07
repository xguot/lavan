use std::fmt;

/// Invalid input is reported explicitly rather than silently coerced or recycled.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Dimension(String),
    InvalidInput(String),
    NonFinite(&'static str),
    NotSymmetric,
    NotPositiveDefinite,
    Singular,
    Unsupported(String),
    Optimization(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dimension(s) => write!(f, "incompatible dimensions: {s}"),
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
            Self::NonFinite(s) => write!(f, "non-finite values in {s}"),
            Self::NotSymmetric => f.write_str("matrix must be symmetric"),
            Self::NotPositiveDefinite => f.write_str("matrix must be positive definite"),
            Self::Singular => f.write_str("matrix is singular"),
            Self::Unsupported(s) => write!(f, "unsupported functionality: {s}"),
            Self::Optimization(s) => write!(f, "optimization failed: {s}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
