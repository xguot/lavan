//! Multivariate normal likelihoods, scores, and information.
//! References: `lav_mvnorm.R` and `lav_mvnorm_kernels.R`.

use crate::{
    Error, Matrix, Result, Vector,
    matrix::{self, Spd},
    sample::{self, SampleStats},
};

pub struct Normal {
    mean: Vector,
    covariance: Matrix,
    precision: Matrix,
    factor: Spd,
}

impl Normal {
    pub fn new(mean: Vector, covariance: Matrix) -> Result<Self> {
        if mean.len() != covariance.nrows() {
            return Err(Error::Dimension("normal mean length differs".into()));
        }
        if mean.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFinite("normal mean"));
        }
        let factor = Spd::new(&covariance)?;
        let precision = factor.inverse();
        Ok(Self {
            mean,
            covariance,
            precision,
            factor,
        })
    }

    pub fn mean(&self) -> &Vector {
        &self.mean
    }
    pub fn covariance(&self) -> &Matrix {
        &self.covariance
    }
    pub fn precision(&self) -> &Matrix {
        &self.precision
    }
    pub fn logdet(&self) -> f64 {
        self.factor.logdet()
    }

    fn check_data(&self, data: &Matrix, weights: Option<&[f64]>) -> Result<f64> {
        if data.ncols() != self.mean.len() {
            return Err(Error::Dimension("data and normal dimensions differ".into()));
        }
        matrix::finite(data, "data")?;
        sample::validate_weights(data.nrows(), weights)
    }

    pub(crate) fn mask(&self, fixed_x: &[usize]) -> Result<Vec<bool>> {
        let mut mask = vec![false; self.mean.len()];
        for &i in fixed_x {
            if i >= mask.len() {
                return Err(Error::Dimension("fixed-x index out of range".into()));
            }
            if mask[i] {
                return Err(Error::InvalidInput("duplicate fixed-x index".into()));
            }
            mask[i] = true;
        }
        Ok(mask)
    }

    pub fn marginal(&self, indices: &[usize]) -> Result<Self> {
        self.mask(indices)?;
        Self::new(
            Vector::from_iterator(indices.len(), indices.iter().map(|&i| self.mean[i])),
            Matrix::from_fn(indices.len(), indices.len(), |i, j| {
                self.covariance[(indices[i], indices[j])]
            }),
        )
    }

    /// Weighted casewise log densities. With `fixed_x`, subtract their marginal
    /// log densities, exactly as lavaan's conditional likelihood does.
    pub fn log_densities(
        &self,
        data: &Matrix,
        weights: Option<&[f64]>,
        fixed_x: &[usize],
    ) -> Result<Vector> {
        self.check_data(data, weights)?;
        let p = self.mean.len();
        let centered_t = Matrix::from_fn(p, data.nrows(), |j, i| data[(i, j)] - self.mean[j]);
        let solved = self.factor.solve(&centered_t)?;
        let constant = p as f64 * (2.0 * std::f64::consts::PI).ln() + self.logdet();
        let mut out = Vector::from_fn(data.nrows(), |i, _| {
            let distance = centered_t.column(i).dot(&solved.column(i));
            -0.5 * (constant + distance) * weights.map_or(1.0, |w| w[i])
        });
        if !fixed_x.is_empty() {
            let marginal = self.marginal(fixed_x)?;
            let x = Matrix::from_fn(data.nrows(), fixed_x.len(), |i, j| data[(i, fixed_x[j])]);
            out -= marginal.log_densities(&x, weights, &[])?;
        }
        Ok(out)
    }

    pub fn log_likelihood(
        &self,
        data: &Matrix,
        weights: Option<&[f64]>,
        fixed_x: &[usize],
    ) -> Result<f64> {
        Ok(sample::sum(
            self.log_densities(data, weights, fixed_x)?.iter().copied(),
        ))
    }

    pub fn log_likelihood_stats(&self, stats: &SampleStats, fixed_x: &[usize]) -> Result<f64> {
        if stats.nvars() != self.mean.len() {
            return Err(Error::Dimension(
                "sample and normal dimensions differ".into(),
            ));
        }
        let w = stats.covariance_about(&self.mean)?;
        let trace = self.precision.component_mul(&w).sum();
        let mut value = -0.5
            * stats.nobs()
            * (self.mean.len() as f64 * (2.0 * std::f64::consts::PI).ln() + self.logdet() + trace);
        if !fixed_x.is_empty() {
            let marginal = self.marginal(fixed_x)?;
            let x_stats = SampleStats::new(
                Vector::from_iterator(fixed_x.len(), fixed_x.iter().map(|&i| stats.mean()[i])),
                Matrix::from_fn(fixed_x.len(), fixed_x.len(), |i, j| {
                    stats.covariance()[(fixed_x[i], fixed_x[j])]
                }),
                stats.nobs(),
            )?;
            value -= marginal.log_likelihood_stats(&x_stats, &[])?;
        }
        Ok(value)
    }

    /// Casewise scores with columns `[mu, vech(Sigma)]` or only `vech(Sigma)`.
    /// Fixed-x entries follow lavaan's zeroing convention.
    pub fn scores(
        &self,
        data: &Matrix,
        weights: Option<&[f64]>,
        fixed_x: &[usize],
        meanstructure: bool,
    ) -> Result<Matrix> {
        self.check_data(data, weights)?;
        let fixed = self.mask(fixed_x)?;
        let p = self.mean.len();
        let pairs = pairs(p, true);
        let offset = if meanstructure { p } else { 0 };
        let centered = Matrix::from_fn(data.nrows(), p, |i, j| data[(i, j)] - self.mean[j]);
        let ysi = centered * &self.precision;
        let mut out = Matrix::zeros(data.nrows(), offset + pairs.len());
        for i in 0..data.nrows() {
            let weight = weights.map_or(1.0, |w| w[i]);
            if meanstructure {
                for j in 0..p {
                    if !fixed[j] {
                        out[(i, j)] = ysi[(i, j)] * weight;
                    }
                }
            }
            for (k, &(row, col)) in pairs.iter().enumerate() {
                if fixed[row] && fixed[col] {
                    continue;
                }
                let scale = if row == col { 0.5 } else { 1.0 };
                out[(i, offset + k)] =
                    (ysi[(i, row)] * ysi[(i, col)] - self.precision[(row, col)]) * scale * weight;
            }
        }
        Ok(out)
    }

    pub fn gradient(
        &self,
        data: &Matrix,
        weights: Option<&[f64]>,
        fixed_x: &[usize],
        meanstructure: bool,
    ) -> Result<Vector> {
        let sc = self.scores(data, weights, fixed_x, meanstructure)?;
        Ok(Vector::from_fn(sc.ncols(), |j, _| {
            sample::sum(sc.column(j).iter().copied())
        }))
    }

    /// Unit expected information, not multiplied by N.
    pub fn information_expected(
        &self,
        fixed_x: &[usize],
        meanstructure: bool,
        correlation: bool,
    ) -> Result<Matrix> {
        if correlation && !fixed_x.is_empty() {
            return Err(Error::Unsupported("fixed-x correlation information".into()));
        }
        let i22 = kron_dup_half(&self.precision, &self.precision, !correlation);
        let mut out = if meanstructure {
            matrix::block_diagonal(&[self.precision.clone(), i22])?
        } else {
            i22
        };
        self.zero_fixed_information(&mut out, fixed_x, meanstructure, !correlation)?;
        Ok(out)
    }

    /// Unit observed information from complete-data sample statistics.
    pub fn information_observed(
        &self,
        stats: &SampleStats,
        fixed_x: &[usize],
        meanstructure: bool,
    ) -> Result<Matrix> {
        if stats.nvars() != self.mean.len() {
            return Err(Error::Dimension(
                "sample and normal dimensions differ".into(),
            ));
        }
        let p = self.mean.len();
        let w = stats.covariance_about(&self.mean)?;
        let aaa = &self.precision * (w * 2.0 - &self.covariance) * &self.precision;
        let i22 = kron_dup_half(&self.precision, &aaa, true);
        let mut out = if meanstructure {
            let mut out = matrix::block_diagonal(&[self.precision.clone(), i22])?;
            let c = &self.precision * (stats.mean() - &self.mean);
            for (k, &(i, j)) in pairs(p, true).iter().enumerate() {
                for l in 0..p {
                    let v = if i == j {
                        c[i] * self.precision[(i, l)]
                    } else {
                        c[i] * self.precision[(j, l)] + c[j] * self.precision[(i, l)]
                    };
                    out[(p + k, l)] = v;
                    out[(l, p + k)] = v;
                }
            }
            out
        } else {
            i22
        };
        self.zero_fixed_information(&mut out, fixed_x, meanstructure, true)?;
        Ok(out)
    }

    /// Inverse expected unit information (normal-theory asymptotic covariance).
    pub fn inverse_information_expected(
        &self,
        fixed_x: &[usize],
        meanstructure: bool,
    ) -> Result<Matrix> {
        let fixed = self.mask(fixed_x)?;
        let mut residual = self.covariance.clone();
        if !fixed_x.is_empty() {
            let y: Vec<usize> = (0..fixed.len()).filter(|&i| !fixed[i]).collect();
            let b = Matrix::from_fn(y.len(), fixed_x.len(), |i, j| {
                self.covariance[(y[i], fixed_x[j])]
            });
            let c = self.marginal(fixed_x)?;
            let conditional =
                Matrix::from_fn(y.len(), y.len(), |i, j| self.covariance[(y[i], y[j])])
                    - &b * &c.precision * b.transpose();
            residual.fill(0.0);
            for i in 0..y.len() {
                for j in 0..y.len() {
                    residual[(y[i], y[j])] = conditional[(i, j)];
                }
            }
        }
        let mut i22 = covariance_of_covariances(&self.covariance)?;
        if !fixed_x.is_empty() {
            i22 -= covariance_of_covariances(&(&self.covariance - &residual))?;
        }
        if meanstructure {
            matrix::block_diagonal(&[residual, i22])
        } else {
            Ok(i22)
        }
    }

    /// Outer product of scores / N. Optional cluster sums use C/(C-1).
    pub fn information_first_order(
        &self,
        data: &Matrix,
        weights: Option<&[f64]>,
        fixed_x: &[usize],
        meanstructure: bool,
        clusters: Option<&[usize]>,
    ) -> Result<Matrix> {
        let n = self.check_data(data, weights)?;
        let mut sc = self.scores(data, weights, fixed_x, meanstructure)?;
        if let Some(cluster) = clusters {
            if cluster.len() != data.nrows() {
                return Err(Error::Dimension("cluster count differs".into()));
            }
            let mut groups = std::collections::BTreeMap::<usize, Vector>::new();
            for (i, &label) in cluster.iter().enumerate() {
                let sum = groups
                    .entry(label)
                    .or_insert_with(|| Vector::zeros(sc.ncols()));
                for j in 0..sc.ncols() {
                    sum[j] += sc[(i, j)];
                }
            }
            let c = groups.len();
            if c < 2 {
                return Err(Error::InvalidInput(
                    "cluster correction needs at least two clusters".into(),
                ));
            }
            let correction = (c as f64 / (c - 1) as f64).sqrt();
            let rows: Vec<Vector> = groups.into_values().collect();
            sc = Matrix::from_fn(c, sc.ncols(), |i, j| rows[i][j] * correction);
        }
        Ok(sc.transpose() * sc / n)
    }

    pub(crate) fn zero_fixed_information(
        &self,
        out: &mut Matrix,
        fixed_x: &[usize],
        meanstructure: bool,
        diagonal: bool,
    ) -> Result<()> {
        let fixed = self.mask(fixed_x)?;
        let offset = if meanstructure { self.mean.len() } else { 0 };
        let mut indices = Vec::new();
        if meanstructure {
            indices.extend(fixed_x.iter().copied());
        }
        for (k, &(i, j)) in pairs(self.mean.len(), diagonal).iter().enumerate() {
            if fixed[i] && fixed[j] {
                indices.push(offset + k);
            }
        }
        for i in indices {
            out.row_mut(i).fill(0.0);
            out.column_mut(i).fill(0.0);
        }
        Ok(())
    }
}

pub(crate) fn pairs(n: usize, diagonal: bool) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for j in 0..n {
        for i in j + usize::from(!diagonal)..n {
            pairs.push((i, j));
        }
    }
    pairs
}

/// Direct `0.5 D' (X ⊗ Y) D`, avoiding the O(p^4) Kronecker allocation.
pub(crate) fn kron_dup_half(x: &Matrix, y: &Matrix, diagonal: bool) -> Matrix {
    let pairs = pairs(x.nrows(), diagonal);
    Matrix::from_fn(pairs.len(), pairs.len(), |a, b| {
        let (i, j) = pairs[a];
        let (k, l) = pairs[b];
        let mut value = x[(j, l)] * y[(i, k)];
        if i != j {
            value += x[(i, l)] * y[(j, k)];
        }
        if k != l {
            value += x[(j, k)] * y[(i, l)];
        }
        if i != j && k != l {
            value += x[(i, k)] * y[(j, l)];
        }
        value * 0.5
    })
}

pub fn covariance_of_covariances(covariance: &Matrix) -> Result<Matrix> {
    matrix::symmetric(covariance)?;
    let pairs = pairs(covariance.nrows(), true);
    Ok(Matrix::from_fn(pairs.len(), pairs.len(), |a, b| {
        let (i, j) = pairs[a];
        let (k, l) = pairs[b];
        covariance[(i, k)] * covariance[(j, l)] + covariance[(i, l)] * covariance[(j, k)]
    }))
}
