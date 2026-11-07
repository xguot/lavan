//! Gaussian missing-data kernels from `lav_mvnorm_missing.R`.
//! NaN marks missing observations. Empty cases do not contribute to moments or
//! likelihoods and retain NaN casewise output, following the R implementation.

use crate::{
    Error, Matrix, Result, Vector,
    mvnorm::{self, Normal},
    sample::{self, SampleStats},
};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct Pattern {
    pub observed: Vec<usize>,
    pub cases: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct MissingData {
    data: Matrix,
    weights: Vec<f64>,
    patterns: Vec<Pattern>,
    nobs: f64,
}

#[derive(Debug, Clone)]
pub struct Saturated {
    pub statistics: SampleStats,
    pub log_likelihood: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl MissingData {
    pub fn new(data: Matrix, weights: Option<&[f64]>) -> Result<Self> {
        if data.ncols() == 0 {
            return Err(Error::InvalidInput("no variables".into()));
        }
        if data.iter().any(|x| x.is_infinite()) {
            return Err(Error::NonFinite("data"));
        }
        sample::validate_weights(data.nrows(), weights)?;
        let weights = weights.map_or_else(|| vec![1.0; data.nrows()], <[f64]>::to_vec);
        let mut groups = BTreeMap::<Vec<usize>, Vec<usize>>::new();
        for i in 0..data.nrows() {
            let observed: Vec<usize> = (0..data.ncols())
                .filter(|&j| !data[(i, j)].is_nan())
                .collect();
            if !observed.is_empty() {
                groups.entry(observed).or_default().push(i);
            }
        }
        let patterns: Vec<Pattern> = groups
            .into_iter()
            .map(|(observed, cases)| Pattern { observed, cases })
            .collect();
        let nobs = sample::sum(
            patterns
                .iter()
                .flat_map(|p| p.cases.iter().map(|&i| weights[i])),
        );
        if nobs <= 0.0 {
            return Err(Error::InvalidInput(
                "no positive-weight nonempty observations".into(),
            ));
        }
        Ok(Self {
            data,
            weights,
            patterns,
            nobs,
        })
    }

    pub fn patterns(&self) -> &[Pattern] {
        &self.patterns
    }
    pub fn nobs(&self) -> f64 {
        self.nobs
    }
    pub fn data(&self) -> &Matrix {
        &self.data
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Unrestricted Gaussian ML moments by EM. Reports iteration limits
    /// explicitly. Entirely unobserved or constant columns are rejected.
    pub fn saturated(&self, tolerance: f64, max_iterations: usize) -> Result<Saturated> {
        if !tolerance.is_finite() || tolerance <= 0.0 || max_iterations == 0 {
            return Err(Error::InvalidInput("invalid EM controls".into()));
        }
        let p = self.data.ncols();
        let mut mean = Vector::zeros(p);
        let mut covariance = Matrix::zeros(p, p);
        for j in 0..p {
            let cases: Vec<usize> = (0..self.data.nrows())
                .filter(|&i| !self.data[(i, j)].is_nan())
                .collect();
            let total = sample::sum(cases.iter().map(|&i| self.weights[i]));
            if total <= 0.0 {
                return Err(Error::InvalidInput(
                    "a variable is never observed with positive weight".into(),
                ));
            }
            mean[j] =
                sample::sum(cases.iter().map(|&i| self.weights[i] * self.data[(i, j)])) / total;
            covariance[(j, j)] = sample::sum(
                cases
                    .iter()
                    .map(|&i| self.weights[i] * (self.data[(i, j)] - mean[j]).powi(2)),
            ) / total;
        }
        let mut normal = Normal::new(mean, covariance)?;
        let mut log_likelihood = self.log_likelihood(&normal)?;
        for iteration in 1..=max_iterations {
            let (first, second) = self.expectation_step(&normal)?;
            let mean = first / self.nobs;
            let raw_covariance = second / self.nobs - &mean * mean.transpose();
            let covariance = (&raw_covariance + raw_covariance.transpose()) * 0.5;
            let new_normal = Normal::new(mean, covariance)?;
            let new_log_likelihood = self.log_likelihood(&new_normal)?;
            let mut change = 0.0_f64;
            for (&a, &b) in new_normal.mean().iter().zip(normal.mean()) {
                change = change.max((a - b).abs() / (1.0 + b.abs()));
            }
            for (&a, &b) in new_normal.covariance().iter().zip(normal.covariance()) {
                change = change.max((a - b).abs() / (1.0 + b.abs()));
            }
            let converged = change <= tolerance
                && (new_log_likelihood - log_likelihood).abs()
                    <= tolerance * (1.0 + log_likelihood.abs());
            normal = new_normal;
            log_likelihood = new_log_likelihood;
            if converged || iteration == max_iterations {
                return Ok(Saturated {
                    statistics: SampleStats::new(
                        normal.mean().clone(),
                        normal.covariance().clone(),
                        self.nobs,
                    )?,
                    log_likelihood,
                    iterations: iteration,
                    converged,
                });
            }
        }
        unreachable!()
    }

    fn validate(&self, normal: &Normal) -> Result<()> {
        if self.data.ncols() != normal.mean().len() {
            return Err(Error::Dimension(
                "missing data and normal dimensions differ".into(),
            ));
        }
        Ok(())
    }

    fn block(&self, pattern: &Pattern) -> (Matrix, Vec<f64>) {
        (
            Matrix::from_fn(pattern.cases.len(), pattern.observed.len(), |i, j| {
                self.data[(pattern.cases[i], pattern.observed[j])]
            }),
            pattern.cases.iter().map(|&i| self.weights[i]).collect(),
        )
    }

    pub fn log_densities(&self, normal: &Normal) -> Result<Vector> {
        self.validate(normal)?;
        let mut out = Vector::from_element(self.data.nrows(), f64::NAN);
        for pattern in &self.patterns {
            let marginal = normal.marginal(&pattern.observed)?;
            let (block, weights) = self.block(pattern);
            let local = marginal.log_densities(&block, None, &[])?;
            for (i, &case) in pattern.cases.iter().enumerate() {
                out[case] = local[i] * weights[i];
            }
        }
        Ok(out)
    }

    pub fn log_likelihood(&self, normal: &Normal) -> Result<f64> {
        Ok(sample::sum(
            self.log_densities(normal)?
                .iter()
                .copied()
                .filter(|x| !x.is_nan()),
        ))
    }

    /// Casewise scores retain NaN for entries involving unobserved variables.
    pub fn scores(
        &self,
        normal: &Normal,
        fixed_x: &[usize],
        meanstructure: bool,
    ) -> Result<Matrix> {
        self.validate(normal)?;
        let p = self.data.ncols();
        let fixed = normal.mask(fixed_x)?;
        let all_pairs = mvnorm::pairs(p, true);
        let offset = if meanstructure { p } else { 0 };
        let mut out = Matrix::from_element(self.data.nrows(), offset + all_pairs.len(), f64::NAN);
        for pattern in &self.patterns {
            let marginal = normal.marginal(&pattern.observed)?;
            let (block, weights) = self.block(pattern);
            let local = marginal.scores(&block, None, &[], meanstructure)?;
            let positions = self.moment_positions(pattern, meanstructure);
            for (i, &case) in pattern.cases.iter().enumerate() {
                for (j, &global) in positions.iter().enumerate() {
                    out[(case, global)] = local[(i, j)] * weights[i];
                }
            }
        }
        if meanstructure {
            for &i in fixed_x {
                out.column_mut(i).fill(0.0);
            }
        }
        for (k, &(i, j)) in all_pairs.iter().enumerate() {
            if fixed[i] && fixed[j] {
                out.column_mut(offset + k).fill(0.0);
            }
        }
        Ok(out)
    }

    pub fn gradient(
        &self,
        normal: &Normal,
        fixed_x: &[usize],
        meanstructure: bool,
    ) -> Result<Vector> {
        let scores = self.scores(normal, fixed_x, meanstructure)?;
        Ok(Vector::from_fn(scores.ncols(), |j, _| {
            sample::sum(scores.column(j).iter().copied().filter(|x| !x.is_nan()))
        }))
    }

    pub fn information_first_order(
        &self,
        normal: &Normal,
        fixed_x: &[usize],
        meanstructure: bool,
        clusters: Option<&[usize]>,
    ) -> Result<Matrix> {
        let mut sc = self.scores(normal, fixed_x, meanstructure)?;
        if let Some(cluster) = clusters {
            if cluster.len() != sc.nrows() {
                return Err(Error::Dimension("cluster count differs".into()));
            }
            let mut groups = BTreeMap::<usize, Vector>::new();
            for (i, &label) in cluster.iter().enumerate() {
                let row = groups
                    .entry(label)
                    .or_insert_with(|| Vector::zeros(sc.ncols()));
                for j in 0..sc.ncols() {
                    if !sc[(i, j)].is_nan() {
                        row[j] += sc[(i, j)];
                    }
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
        Ok(Matrix::from_fn(sc.ncols(), sc.ncols(), |j, k| {
            sample::sum(
                (0..sc.nrows())
                    .map(|i| sc[(i, j)] * sc[(i, k)])
                    .filter(|x| !x.is_nan()),
            ) / self.nobs
        }))
    }

    fn moment_positions(&self, pattern: &Pattern, meanstructure: bool) -> Vec<usize> {
        let p = self.data.ncols();
        let all_pairs = mvnorm::pairs(p, true);
        let mut positions = if meanstructure {
            pattern.observed.clone()
        } else {
            Vec::new()
        };
        let offset = if meanstructure { p } else { 0 };
        for (i, j) in mvnorm::pairs(pattern.observed.len(), true) {
            let pair = (pattern.observed[i], pattern.observed[j]);
            positions.push(offset + all_pairs.iter().position(|&x| x == pair).unwrap());
        }
        positions
    }

    /// Pattern-weighted unit expected or observed information.
    pub fn information(
        &self,
        normal: &Normal,
        fixed_x: &[usize],
        meanstructure: bool,
        observed: bool,
    ) -> Result<Matrix> {
        self.validate(normal)?;
        normal.mask(fixed_x)?;
        let p = self.data.ncols();
        let q = mvnorm::pairs(p, true).len() + if meanstructure { p } else { 0 };
        let mut out = Matrix::zeros(q, q);
        for pattern in &self.patterns {
            let marginal = normal.marginal(&pattern.observed)?;
            let (block, weights) = self.block(pattern);
            let frequency = sample::sum(weights.iter().copied());
            if frequency == 0.0 {
                continue;
            }
            let local = if observed {
                let stats = SampleStats::from_data(&block, Some(&weights))?;
                marginal.information_observed(&stats, &[], meanstructure)?
            } else {
                marginal.information_expected(&[], meanstructure, false)?
            };
            let positions = self.moment_positions(pattern, meanstructure);
            for (i, &global_i) in positions.iter().enumerate() {
                for (j, &global_j) in positions.iter().enumerate() {
                    out[(global_i, global_j)] += frequency * local[(i, j)] / self.nobs;
                }
            }
        }
        normal.zero_fixed_information(&mut out, fixed_x, meanstructure, true)?;
        Ok(out)
    }

    /// Conditional-mean imputation; entirely empty cases remain NaN.
    pub fn impute(&self, normal: &Normal) -> Result<Matrix> {
        self.validate(normal)?;
        let mut out = self.data.clone();
        let p = self.data.ncols();
        for pattern in &self.patterns {
            let missing: Vec<usize> = (0..p).filter(|i| !pattern.observed.contains(i)).collect();
            if missing.is_empty() {
                continue;
            }
            let marginal = normal.marginal(&pattern.observed)?;
            let cross = Matrix::from_fn(missing.len(), pattern.observed.len(), |i, j| {
                normal.covariance()[(missing[i], pattern.observed[j])]
            });
            let regression = cross * marginal.precision();
            for &case in &pattern.cases {
                let diff = Vector::from_iterator(
                    pattern.observed.len(),
                    pattern
                        .observed
                        .iter()
                        .map(|&i| self.data[(case, i)] - normal.mean()[i]),
                );
                let prediction = &regression * diff;
                for (i, &var) in missing.iter().enumerate() {
                    out[(case, var)] = normal.mean()[var] + prediction[i];
                }
            }
        }
        Ok(out)
    }

    /// E-step sums E[Y] and E[Y Y']; includes conditional residual uncertainty.
    pub fn expectation_step(&self, normal: &Normal) -> Result<(Vector, Matrix)> {
        self.validate(normal)?;
        let imputed = self.impute(normal)?;
        let p = self.data.ncols();
        let mut first = Vector::zeros(p);
        let mut second = Matrix::zeros(p, p);
        for pattern in &self.patterns {
            let missing: Vec<usize> = (0..p).filter(|i| !pattern.observed.contains(i)).collect();
            let frequency = sample::sum(pattern.cases.iter().map(|&i| self.weights[i]));
            for &case in &pattern.cases {
                let y = imputed.row(case).transpose();
                first += &y * self.weights[case];
                second += &y * y.transpose() * self.weights[case];
            }
            if !missing.is_empty() {
                let marginal = normal.marginal(&pattern.observed)?;
                let cross = Matrix::from_fn(missing.len(), pattern.observed.len(), |i, j| {
                    normal.covariance()[(missing[i], pattern.observed[j])]
                });
                let conditional = Matrix::from_fn(missing.len(), missing.len(), |i, j| {
                    normal.covariance()[(missing[i], missing[j])]
                }) - &cross * marginal.precision() * cross.transpose();
                for (i, &a) in missing.iter().enumerate() {
                    for (j, &b) in missing.iter().enumerate() {
                        second[(a, b)] += frequency * conditional[(i, j)];
                    }
                }
            }
        }
        Ok((first, second))
    }
}
