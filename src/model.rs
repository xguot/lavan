//! Continuous LISREL model representation, analytic derivatives, and ML fitting.
//! The moment equations follow `lav_lisrel_sigma` and `lav_lisrel_mu`.
//! This numerical API does not implement lavaan's R syntax or automatic defaults.

use crate::{
    Error, Matrix, Result, Vector, matrix,
    mvnorm::{self, Normal},
    objective::MaximumLikelihood,
    optim,
    sample::SampleStats,
};

#[derive(Debug, Clone)]
pub struct Lisrel {
    pub lambda: Matrix,
    pub theta: Matrix,
    pub psi: Matrix,
    pub beta: Matrix,
    pub nu: Vector,
    pub alpha: Vector,
}

#[derive(Debug, Clone)]
pub struct Implied {
    pub covariance: Matrix,
    pub mean: Vector,
}

impl Lisrel {
    pub fn new(lambda: Matrix, theta: Matrix, psi: Matrix) -> Result<Self> {
        let (p, q) = lambda.shape();
        let model = Self {
            lambda,
            theta,
            psi,
            beta: Matrix::zeros(q, q),
            nu: Vector::zeros(p),
            alpha: Vector::zeros(q),
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<()> {
        let (p, q) = self.lambda.shape();
        if p == 0
            || q == 0
            || self.theta.shape() != (p, p)
            || self.psi.shape() != (q, q)
            || self.beta.shape() != (q, q)
            || self.nu.len() != p
            || self.alpha.len() != q
        {
            return Err(Error::Dimension("invalid LISREL matrix dimensions".into()));
        }
        matrix::finite(&self.lambda, "lambda")?;
        matrix::finite(&self.beta, "beta")?;
        matrix::symmetric(&self.theta)?;
        matrix::symmetric(&self.psi)?;
        if self
            .nu
            .iter()
            .chain(self.alpha.iter())
            .any(|x| !x.is_finite())
        {
            return Err(Error::NonFinite("intercepts"));
        }
        Ok(())
    }

    fn transform(&self) -> Result<(Matrix, Matrix)> {
        self.validate()?;
        let h = matrix::inverse_iminus(&self.beta)?;
        let t = &self.lambda * &h;
        Ok((h, t))
    }

    pub fn implied(&self) -> Result<Implied> {
        let (_, t) = self.transform()?;
        Ok(Implied {
            covariance: &t * &self.psi * t.transpose() + &self.theta,
            mean: &self.nu + &t * &self.alpha,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Member {
    Lambda,
    Theta,
    Psi,
    Beta,
    Nu,
    Alpha,
}

/// A covariance slot writes both symmetric positions. A vector slot has col=0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Slot {
    pub member: Member,
    pub row: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub initial: f64,
    /// Multiple slots express a shared parameter / linear equality of entries.
    pub slots: Vec<Slot>,
}

#[derive(Debug, Clone)]
pub struct Model {
    base: Lisrel,
    parameters: Vec<Parameter>,
    meanstructure: bool,
    baseline_covariates: Vec<usize>,
}

fn set(model: &mut Lisrel, slot: Slot, value: f64) {
    let Slot { member, row, col } = slot;
    match member {
        Member::Lambda => model.lambda[(row, col)] = value,
        Member::Theta => {
            model.theta[(row, col)] = value;
            model.theta[(col, row)] = value;
        }
        Member::Psi => {
            model.psi[(row, col)] = value;
            model.psi[(col, row)] = value;
        }
        Member::Beta => model.beta[(row, col)] = value,
        Member::Nu => model.nu[row] = value,
        Member::Alpha => model.alpha[row] = value,
    }
}

impl Model {
    pub fn new(base: Lisrel, parameters: Vec<Parameter>, meanstructure: bool) -> Result<Self> {
        base.validate()?;
        let (p, q) = base.lambda.shape();
        let mut occupied = std::collections::BTreeSet::new();
        let mut names = std::collections::BTreeSet::new();
        for parameter in &parameters {
            if parameter.name.is_empty()
                || !names.insert(parameter.name.clone())
                || parameter.slots.is_empty()
                || !parameter.initial.is_finite()
            {
                return Err(Error::InvalidInput(
                    "parameters need unique names, finite starts, and nonempty slots".into(),
                ));
            }
            for &mut_slot in &parameter.slots {
                let mut slot = mut_slot;
                let dims = match slot.member {
                    Member::Lambda => (p, q),
                    Member::Theta => (p, p),
                    Member::Psi | Member::Beta => (q, q),
                    Member::Nu => (p, 1),
                    Member::Alpha => (q, 1),
                };
                if slot.row >= dims.0 || slot.col >= dims.1 {
                    return Err(Error::Dimension("parameter slot out of range".into()));
                }
                if !meanstructure && matches!(slot.member, Member::Nu | Member::Alpha) {
                    return Err(Error::InvalidInput(
                        "free intercepts require meanstructure".into(),
                    ));
                }
                if matches!(slot.member, Member::Theta | Member::Psi) && slot.row < slot.col {
                    std::mem::swap(&mut slot.row, &mut slot.col);
                }
                if !occupied.insert(slot) {
                    return Err(Error::InvalidInput(
                        "multiple parameters write the same slot".into(),
                    ));
                }
            }
        }
        Ok(Self {
            base,
            parameters,
            meanstructure,
            baseline_covariates: Vec::new(),
        })
    }

    /// Observed exogenous variables remain correlated in lavaan's baseline model.
    /// Supply their observed-column indices for path models; this semantic
    /// information cannot be inferred from the numeric matrices alone.
    pub fn with_baseline_covariates(mut self, indices: Vec<usize>) -> Result<Self> {
        let mut unique = std::collections::BTreeSet::new();
        for &i in &indices {
            if i >= self.base.lambda.nrows() {
                return Err(Error::Dimension(
                    "baseline covariate index out of range".into(),
                ));
            }
            if !unique.insert(i) {
                return Err(Error::InvalidInput(
                    "duplicate baseline covariate index".into(),
                ));
            }
        }
        self.baseline_covariates = indices;
        Ok(self)
    }

    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }
    pub fn meanstructure(&self) -> bool {
        self.meanstructure
    }
    pub fn initial(&self) -> Vector {
        Vector::from_iterator(
            self.parameters.len(),
            self.parameters.iter().map(|p| p.initial),
        )
    }

    pub fn matrices(&self, values: &Vector) -> Result<Lisrel> {
        if values.len() != self.parameters.len() {
            return Err(Error::Dimension("parameter count differs".into()));
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFinite("parameters"));
        }
        let mut model = self.base.clone();
        for (parameter, &value) in self.parameters.iter().zip(values) {
            for &slot in &parameter.slots {
                set(&mut model, slot, value);
            }
        }
        Ok(model)
    }

    pub fn implied(&self, values: &Vector) -> Result<Implied> {
        self.matrices(values)?.implied()
    }

    fn directions(&self, values: &Vector) -> Result<(Implied, Vec<Implied>)> {
        let matrices = self.matrices(values)?;
        let (h, t) = matrices.transform()?;
        let (p, q) = matrices.lambda.shape();
        let implied = Implied {
            covariance: &t * &matrices.psi * t.transpose() + &matrices.theta,
            mean: &matrices.nu + &t * &matrices.alpha,
        };
        let mut derivatives = Vec::with_capacity(self.parameters.len());
        for parameter in &self.parameters {
            let mut direction = Lisrel {
                lambda: Matrix::zeros(p, q),
                theta: Matrix::zeros(p, p),
                psi: Matrix::zeros(q, q),
                beta: Matrix::zeros(q, q),
                nu: Vector::zeros(p),
                alpha: Vector::zeros(q),
            };
            for &slot in &parameter.slots {
                set(&mut direction, slot, 1.0);
            }
            let dt = &direction.lambda * &h + &t * &direction.beta * &h;
            let one_sided = &dt * &matrices.psi * t.transpose();
            derivatives.push(Implied {
                covariance: &one_sided
                    + one_sided.transpose()
                    + &t * &direction.psi * t.transpose()
                    + direction.theta,
                mean: direction.nu + &dt * &matrices.alpha + &t * direction.alpha,
            });
        }
        Ok((implied, derivatives))
    }

    /// Jacobian of `[mean, vech(Sigma)]` (or only vech) with respect to free parameters.
    pub fn jacobian(&self, values: &Vector) -> Result<Matrix> {
        let (_, derivatives) = self.directions(values)?;
        let p = self.base.lambda.nrows();
        let pairs = mvnorm::pairs(p, true);
        let offset = if self.meanstructure { p } else { 0 };
        Ok(Matrix::from_fn(
            offset + pairs.len(),
            derivatives.len(),
            |i, j| {
                if i < offset {
                    derivatives[j].mean[i]
                } else {
                    let (r, c) = pairs[i - offset];
                    derivatives[j].covariance[(r, c)]
                }
            },
        ))
    }

    pub fn objective_gradient(
        &self,
        values: &Vector,
        objective: &MaximumLikelihood,
    ) -> Result<(f64, Vector)> {
        let (implied, directions) = self.directions(values)?;
        let mean = self.meanstructure.then_some(&implied.mean);
        let value = objective.evaluate(&implied.covariance, mean)?;
        if !value.is_finite() {
            return Ok((value, Vector::zeros(values.len())));
        }
        let (ds, dm) = objective.gradient(&implied.covariance, mean)?;
        let gradient = Vector::from_iterator(
            directions.len(),
            directions.iter().map(|d| {
                let mean = dm.as_ref().map_or(0.0, |g| g.dot(&d.mean));
                ds.component_mul(&d.covariance).sum() + mean
            }),
        );
        Ok((value, gradient))
    }

    pub fn fit(&self, stats: SampleStats, options: optim::Options) -> Result<Fit> {
        let objective = MaximumLikelihood::new(stats)?;
        if objective.stats().nvars() != self.base.lambda.nrows() {
            return Err(Error::Dimension(
                "model and sample variable counts differ".into(),
            ));
        }
        let solution = optim::minimize(self.initial(), options, |values| {
            self.objective_gradient(values, &objective)
        })?;
        let implied = self.implied(&solution.parameters)?;
        let log_likelihood = objective.log_likelihood(
            &implied.covariance,
            self.meanstructure.then_some(&implied.mean),
        )?;
        let p = objective.stats().nvars();
        let moments = matrix::half_len(p, true)? + if self.meanstructure { p } else { 0 };
        let df = moments as isize - self.parameters.len() as isize;
        let chi_square = objective.stats().nobs() * solution.objective;
        let mut baseline = Matrix::zeros(p, p);
        baseline.set_diagonal(&objective.stats().covariance().diagonal());
        for &i in &self.baseline_covariates {
            for &j in &self.baseline_covariates {
                baseline[(i, j)] = objective.stats().covariance()[(i, j)];
            }
        }
        let baseline_chi_square = objective.stats().nobs() * objective.evaluate(&baseline, None)?;
        let nx = self.baseline_covariates.len();
        let baseline_df = (p * (p - 1) / 2 - nx * nx.saturating_sub(1) / 2) as isize;
        let parameter_covariance = if solution.converged() && !solution.parameters.is_empty() {
            let jacobian = self.jacobian(&solution.parameters)?;
            let normal = Normal::new(implied.mean.clone(), implied.covariance.clone())?;
            let moment_info = normal.information_expected(&[], self.meanstructure, false)?;
            let information =
                jacobian.transpose() * moment_info * jacobian * objective.stats().nobs();
            matrix::Spd::new(&information)
                .ok()
                .map(|factor| factor.inverse())
        } else if solution.parameters.is_empty() {
            Some(Matrix::zeros(0, 0))
        } else {
            None
        };
        Ok(Fit {
            solution,
            implied,
            log_likelihood,
            chi_square,
            df,
            baseline_chi_square,
            baseline_df,
            nobs: objective.stats().nobs(),
            parameter_covariance,
        })
    }
    /// Pattern-wise full information ML for a continuous single-group model.
    /// Uses expected missing-data information for standard errors. Observed
    /// covariates and conditional-x likelihoods remain unsupported here.
    pub fn fit_missing(
        &self,
        data: &crate::missing::MissingData,
        options: optim::Options,
    ) -> Result<Fit> {
        if !self.meanstructure {
            return Err(Error::InvalidInput("FIML requires a mean structure".into()));
        }
        if !self.baseline_covariates.is_empty() {
            return Err(Error::Unsupported("FIML baseline covariates".into()));
        }
        if data.data().ncols() != self.base.lambda.nrows() {
            return Err(Error::Dimension(
                "model and missing-data variable counts differ".into(),
            ));
        }
        let saturated = data.saturated(1e-10, 10_000)?;
        if !saturated.converged {
            return Err(Error::Optimization(
                "saturated-model EM iteration limit".into(),
            ));
        }
        let solution = optim::minimize(self.initial(), options, |values| {
            let (implied, directions) = self.directions(values)?;
            let normal = match Normal::new(implied.mean, implied.covariance) {
                Ok(n) => n,
                Err(Error::NotPositiveDefinite) => {
                    return Ok((f64::INFINITY, Vector::zeros(values.len())));
                }
                Err(e) => return Err(e),
            };
            let value = (2.0 * (saturated.log_likelihood - data.log_likelihood(&normal)?)
                / data.nobs())
            .max(0.0);
            let score = data.gradient(&normal, &[], true)?;
            let p = normal.mean().len();
            let pairs = mvnorm::pairs(p, true);
            let gradient = Vector::from_iterator(
                directions.len(),
                directions.iter().map(|d| {
                    let mean = (0..p).map(|i| score[i] * d.mean[i]).sum::<f64>();
                    let cov = pairs
                        .iter()
                        .enumerate()
                        .map(|(k, &(i, j))| score[p + k] * d.covariance[(i, j)])
                        .sum::<f64>();
                    -2.0 * (mean + cov) / data.nobs()
                }),
            );
            Ok((value, gradient))
        })?;
        let implied = self.implied(&solution.parameters)?;
        let normal = Normal::new(implied.mean.clone(), implied.covariance.clone())?;
        let log_likelihood = data.log_likelihood(&normal)?;
        let p = implied.mean.len();
        let mut baseline_mean = Vector::zeros(p);
        let mut baseline_covariance = Matrix::zeros(p, p);
        for j in 0..p {
            let cases: Vec<usize> = (0..data.data().nrows())
                .filter(|&i| !data.data()[(i, j)].is_nan())
                .collect();
            let block = Matrix::from_fn(cases.len(), 1, |i, _| data.data()[(cases[i], j)]);
            let weights: Vec<f64> = cases.iter().map(|&i| data.weights()[i]).collect();
            let stats = SampleStats::from_data(&block, Some(&weights))?;
            baseline_mean[j] = stats.mean()[0];
            baseline_covariance[(j, j)] = stats.covariance()[(0, 0)];
        }
        let baseline = Normal::new(baseline_mean, baseline_covariance)?;
        let parameter_covariance = if solution.converged() && !solution.parameters.is_empty() {
            let jacobian = self.jacobian(&solution.parameters)?;
            let information = jacobian.transpose()
                * data.information(&normal, &[], true, false)?
                * jacobian
                * data.nobs();
            matrix::Spd::new(&information)
                .ok()
                .map(|factor| factor.inverse())
        } else if solution.parameters.is_empty() {
            Some(Matrix::zeros(0, 0))
        } else {
            None
        };
        Ok(Fit {
            chi_square: (2.0 * (saturated.log_likelihood - log_likelihood)).max(0.0),
            df: (matrix::half_len(p, true)? + p) as isize - self.parameters.len() as isize,
            baseline_chi_square: (2.0
                * (saturated.log_likelihood - data.log_likelihood(&baseline)?))
            .max(0.0),
            baseline_df: (p * (p - 1) / 2) as isize,
            nobs: data.nobs(),
            solution,
            implied,
            log_likelihood,
            parameter_covariance,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Fit {
    pub solution: optim::Solution,
    pub implied: Implied,
    pub log_likelihood: f64,
    pub chi_square: f64,
    pub df: isize,
    pub baseline_chi_square: f64,
    pub baseline_df: isize,
    pub nobs: f64,
    /// Expected-information covariance; None when not converged or not identified.
    pub parameter_covariance: Option<Matrix>,
}

impl Fit {
    pub fn comparison(&self) -> crate::fit::Comparison {
        crate::fit::Comparison::new(
            self.chi_square,
            self.df as f64,
            self.baseline_chi_square,
            self.baseline_df as f64,
        )
    }
    pub fn standard_errors(&self) -> Option<Vector> {
        self.parameter_covariance
            .as_ref()
            .map(|v| v.diagonal().map(f64::sqrt))
    }
}
