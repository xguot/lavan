//! Unconstrained BFGS with analytic gradients and an Armijo line search.
//! This is an independent Rust optimizer, not a translation of R's nlminb.

use crate::{Error, Matrix, Result, Vector};

#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub max_line_search: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 2000,
            gradient_tolerance: 1e-7,
            max_line_search: 60,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Converged,
    IterationLimit,
    LineSearchFailed,
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub parameters: Vector,
    pub objective: f64,
    pub gradient: Vector,
    pub iterations: usize,
    pub evaluations: usize,
    pub status: Status,
}

impl Solution {
    pub fn converged(&self) -> bool {
        self.status == Status::Converged
    }
}

pub fn minimize<F>(initial: Vector, options: Options, mut evaluate: F) -> Result<Solution>
where
    F: FnMut(&Vector) -> Result<(f64, Vector)>,
{
    if options.max_line_search == 0
        || options.max_iterations == 0
        || !options.gradient_tolerance.is_finite()
        || options.gradient_tolerance <= 0.0
    {
        return Err(Error::InvalidInput("invalid optimization controls".into()));
    }
    if initial.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFinite("initial parameters"));
    }
    let p = initial.len();
    let (mut f, mut gradient) = evaluate(&initial)?;
    if gradient.len() != p {
        return Err(Error::Dimension("objective gradient length differs".into()));
    }
    if !f.is_finite() || gradient.iter().any(|x| !x.is_finite()) {
        return Err(Error::Optimization(
            "initial point has non-finite objective or gradient".into(),
        ));
    }
    let mut parameters = initial;
    let mut evaluations = 1;
    let mut inverse_hessian = Matrix::identity(p, p);
    for iteration in 0..=options.max_iterations {
        if p == 0 || gradient.amax() <= options.gradient_tolerance {
            return Ok(Solution {
                parameters,
                objective: f,
                gradient,
                iterations: iteration,
                evaluations,
                status: Status::Converged,
            });
        }
        if iteration == options.max_iterations {
            break;
        }
        let mut direction = -&inverse_hessian * &gradient;
        let mut slope = gradient.dot(&direction);
        if !slope.is_finite() || slope >= 0.0 {
            inverse_hessian.fill(0.0);
            inverse_hessian.fill_diagonal(1.0);
            direction = -&gradient;
            slope = -gradient.norm_squared();
        }
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..options.max_line_search {
            let candidate = &parameters + &direction * step;
            let (new_f, new_gradient) = evaluate(&candidate)?;
            evaluations += 1;
            if new_gradient.len() != p {
                return Err(Error::Dimension("objective gradient length changed".into()));
            }
            if new_f.is_finite()
                && new_gradient.iter().all(|x| x.is_finite())
                && new_f <= f + 1e-4 * step * slope
            {
                accepted = Some((candidate, new_f, new_gradient));
                break;
            }
            step *= 0.5;
        }
        let Some((new_parameters, new_f, new_gradient)) = accepted else {
            return Ok(Solution {
                parameters,
                objective: f,
                gradient,
                iterations: iteration,
                evaluations,
                status: Status::LineSearchFailed,
            });
        };
        let s = &new_parameters - &parameters;
        let y = &new_gradient - &gradient;
        let curvature = s.dot(&y);
        if curvature > f64::EPSILON * s.norm() * y.norm() && curvature.is_finite() {
            let hy = &inverse_hessian * &y;
            inverse_hessian += (&s * s.transpose())
                * ((curvature + y.dot(&hy)) / curvature.powi(2))
                - (&hy * s.transpose() + &s * hy.transpose()) / curvature;
            // Remove asymmetry from accumulated rounding.
            inverse_hessian = (&inverse_hessian + inverse_hessian.transpose()) * 0.5;
        }
        parameters = new_parameters;
        f = new_f;
        gradient = new_gradient;
    }
    Ok(Solution {
        parameters,
        objective: f,
        gradient,
        iterations: options.max_iterations,
        evaluations,
        status: Status::IterationLimit,
    })
}
