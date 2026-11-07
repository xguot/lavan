//! Complete-data sample moments and frequency-weighted variances.
//! Reference: `lav_mat_cov`, `lav_mat_var_wt`, `lav_mvn_samp_stats`.

use crate::{Error, Matrix, Result, Vector, matrix};

/// Compensated summation keeps moments accurate when locations are large.
pub(crate) fn sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut total, mut correction) = (0.0_f64, 0.0_f64);
    for x in values {
        let t = total + x;
        correction += if total.abs() >= x.abs() {
            (total - t) + x
        } else {
            (x - t) + total
        };
        total = t;
    }
    total + correction
}

pub(crate) fn validate_weights(rows: usize, weights: Option<&[f64]>) -> Result<f64> {
    if rows == 0 {
        return Err(Error::InvalidInput("no observations".into()));
    }
    if let Some(w) = weights {
        if w.len() != rows {
            return Err(Error::Dimension(
                "weight count differs from observation count".into(),
            ));
        }
        if w.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(Error::InvalidInput(
                "weights must be finite and nonnegative".into(),
            ));
        }
        let n = sum(w.iter().copied());
        if n <= 0.0 || !n.is_finite() {
            return Err(Error::InvalidInput(
                "weights must have positive finite sum".into(),
            ));
        }
        Ok(n)
    } else {
        Ok(rows as f64)
    }
}

#[derive(Debug, Clone)]
pub struct SampleStats {
    mean: Vector,
    covariance: Matrix,
    nobs: f64,
}

impl SampleStats {
    /// Covariance uses the ML divisor N, as lavaan does internally.
    pub fn new(mean: Vector, covariance: Matrix, nobs: f64) -> Result<Self> {
        let p = matrix::symmetric(&covariance)?;
        if p == 0 || mean.len() != p {
            return Err(Error::Dimension(
                "mean and covariance dimensions differ or are empty".into(),
            ));
        }
        if mean.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFinite("sample mean"));
        }
        if !nobs.is_finite() || nobs <= 0.0 {
            return Err(Error::InvalidInput(
                "sample size must be positive and finite".into(),
            ));
        }
        Ok(Self {
            mean,
            covariance,
            nobs,
        })
    }

    pub fn from_data(data: &Matrix, weights: Option<&[f64]>) -> Result<Self> {
        matrix::finite(
            data,
            "data (missing values require an explicit missing-data method)",
        )?;
        if data.ncols() == 0 {
            return Err(Error::InvalidInput("no variables".into()));
        }
        let n = validate_weights(data.nrows(), weights)?;
        if weights.is_none() && data.nrows() < 2 {
            return Err(Error::InvalidInput(
                "unweighted covariance needs at least two observations".into(),
            ));
        }
        let p = data.ncols();
        let normalized = weights.map(|w| w.iter().map(|x| x / n).collect::<Vec<_>>());
        // cov.wt normalizes weights before computing its center. Unweighted
        // means use a shifted sum to retain a large common location accurately.
        let mean = Vector::from_fn(p, |j, _| {
            if let Some(w) = &normalized {
                sum((0..data.nrows()).map(|i| w[i] * data[(i, j)]))
            } else {
                let origin = data[(0, j)];
                origin + sum((0..data.nrows()).map(|i| data[(i, j)] - origin)) / n
            }
        });
        // stats::cov corrects the centering mean on a second pass. Do not
        // subtract residual_sum products from the covariance: that alternative
        // differs from R when the mean correction rounds away at large offsets.
        let center = if normalized.is_none() {
            Vector::from_fn(p, |j, _| {
                mean[j] + sum((0..data.nrows()).map(|i| data[(i, j)] - mean[j])) / n
            })
        } else {
            mean.clone()
        };
        let root_weights = normalized
            .as_ref()
            .map(|w| w.iter().map(|x| x.sqrt()).collect::<Vec<_>>());
        let mut covariance = Matrix::zeros(p, p);
        for j in 0..p {
            for k in j..p {
                let cross = sum((0..data.nrows()).map(|i| {
                    let scale = root_weights.as_ref().map_or(1.0, |w| w[i]);
                    ((data[(i, j)] - center[j]) * scale) * ((data[(i, k)] - center[k]) * scale)
                }));
                let value = if normalized.is_some() {
                    cross
                } else {
                    cross / n
                };
                covariance[(j, k)] = value;
                covariance[(k, j)] = value;
            }
        }
        Self::new(mean, covariance, n)
    }

    pub fn mean(&self) -> &Vector {
        &self.mean
    }
    pub fn covariance(&self) -> &Matrix {
        &self.covariance
    }
    pub fn nobs(&self) -> f64 {
        self.nobs
    }
    pub fn nvars(&self) -> usize {
        self.mean.len()
    }

    pub fn covariance_about(&self, mu: &Vector) -> Result<Matrix> {
        if mu.len() != self.mean.len() {
            return Err(Error::Dimension("mean length differs".into()));
        }
        if mu.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFinite("mean"));
        }
        let diff = &self.mean - mu;
        Ok(&self.covariance + &diff * diff.transpose())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum VarianceDivisor {
    MaximumLikelihood,
    UnbiasedFrequency,
}

/// Frequency-weighted column variances; NaN observations are omitted per column.
/// Degenerate columns follow R's division and omission rules, including zero
/// for an entirely missing column with the unbiased-frequency divisor.
pub fn weighted_variances(
    data: &Matrix,
    weights: Option<&[f64]>,
    divisor: VarianceDivisor,
) -> Result<Vector> {
    validate_weights(data.nrows(), weights)?;
    if data.iter().any(|x| x.is_infinite()) {
        return Err(Error::NonFinite("data"));
    }
    let weight = |i: usize| weights.map_or(1.0, |w| w[i]);
    let any_missing = data.iter().any(|x| x.is_nan());
    let mut out = Vector::zeros(data.ncols());
    for j in 0..data.ncols() {
        let indices: Vec<usize> = (0..data.nrows())
            .filter(|&i| !data[(i, j)].is_nan())
            .collect();
        let n = sum(indices.iter().map(|&i| weight(i)));
        let mean = sum(indices.iter().map(|&i| weight(i) * data[(i, j)])) / n;
        let numerator = sum(indices
            .iter()
            .map(|&i| weight(i) * (data[(i, j)] - mean).powi(2))
            .filter(|x| !any_missing || !x.is_nan()));
        let denominator = match divisor {
            VarianceDivisor::MaximumLikelihood => n,
            VarianceDivisor::UnbiasedFrequency => n - 1.0,
        };
        out[j] = numerator / denominator;
    }
    Ok(out)
}

pub fn transform_mean_covariance(
    data: &Matrix,
    target_mean: &Vector,
    target_covariance: &Matrix,
) -> Result<Matrix> {
    let stats = SampleStats::from_data(data, None)?;
    let p = stats.nvars();
    if target_mean.len() != p || target_covariance.shape() != (p, p) {
        return Err(Error::Dimension(
            "target dimensions differ from data".into(),
        ));
    }
    if target_mean.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFinite("target mean"));
    }
    let inverse = matrix::Spd::new(stats.covariance())?.inverse();
    let transform = matrix::symmetric_sqrt(&inverse)? * matrix::symmetric_sqrt(target_covariance)?;
    let mut out = data * transform;
    for j in 0..p {
        let center = sum(out.column(j).iter().copied()) / data.nrows() as f64;
        for i in 0..data.nrows() {
            out[(i, j)] += target_mean[j] - center;
        }
    }
    Ok(out)
}
