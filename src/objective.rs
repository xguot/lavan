//! Estimation objectives from `R/lav_objective.R`.

use crate::{
    Error, Matrix, Result, Vector,
    matrix::{self, Spd},
    mvnorm::Normal,
    sample::SampleStats,
};

pub struct MaximumLikelihood {
    stats: SampleStats,
    sample_logdet: f64,
}

impl MaximumLikelihood {
    pub fn new(stats: SampleStats) -> Result<Self> {
        let sample_logdet = Spd::new(stats.covariance())?.logdet();
        Ok(Self {
            stats,
            sample_logdet,
        })
    }

    pub fn stats(&self) -> &SampleStats {
        &self.stats
    }

    /// F_ML; lavaan truncates finite negative roundoff to zero.
    /// Non-positive-definite model covariance returns +infinity.
    pub fn evaluate(&self, covariance: &Matrix, mean: Option<&Vector>) -> Result<f64> {
        let p = self.stats.nvars();
        if covariance.shape() != (p, p) {
            return Err(Error::Dimension(
                "model covariance dimensions differ".into(),
            ));
        }
        let factor = match Spd::new(covariance) {
            Ok(x) => x,
            Err(Error::NotPositiveDefinite) => return Ok(f64::INFINITY),
            Err(e) => return Err(e),
        };
        let w = if let Some(mu) = mean {
            self.stats.covariance_about(mu)?
        } else {
            self.stats.covariance().clone()
        };
        let value = factor.logdet() + factor.solve(&w)?.trace() - self.sample_logdet - p as f64;
        Ok(nonnegative(value))
    }

    /// Derivatives with respect to full Sigma and mu (before parameter mapping).
    pub fn gradient(
        &self,
        covariance: &Matrix,
        mean: Option<&Vector>,
    ) -> Result<(Matrix, Option<Vector>)> {
        if covariance.shape() != self.stats.covariance().shape() {
            return Err(Error::Dimension(
                "model covariance dimensions differ".into(),
            ));
        }
        let inverse = Spd::new(covariance)?.inverse();
        let w = if let Some(mu) = mean {
            self.stats.covariance_about(mu)?
        } else {
            self.stats.covariance().clone()
        };
        let dsigma = &inverse - &inverse * w * &inverse;
        let dmu = mean.map(|mu| &inverse * (mu - self.stats.mean()) * 2.0);
        Ok((dsigma, dmu))
    }

    pub fn log_likelihood(&self, covariance: &Matrix, mean: Option<&Vector>) -> Result<f64> {
        let mu = mean.unwrap_or(self.stats.mean());
        Normal::new(mu.clone(), covariance.clone())?.log_likelihood_stats(&self.stats, &[])
    }
}

fn nonnegative(x: f64) -> f64 {
    if x.is_finite() && x < 0.0 { 0.0 } else { x }
}

pub fn weighted_least_squares(
    observed: &Vector,
    estimated: &Vector,
    weights: &Matrix,
) -> Result<f64> {
    if observed.len() != estimated.len() || weights.shape() != (observed.len(), observed.len()) {
        return Err(Error::Dimension(
            "WLS moment and weight dimensions differ".into(),
        ));
    }
    matrix::finite(weights, "WLS weights")?;
    let diff = observed - estimated;
    Ok(nonnegative(diff.dot(&(weights * &diff))))
}

pub fn diagonally_weighted_least_squares(
    observed: &Vector,
    estimated: &Vector,
    weights: &Vector,
) -> Result<f64> {
    if observed.len() != estimated.len() || weights.len() != observed.len() {
        return Err(Error::Dimension(
            "DWLS moment and weight dimensions differ".into(),
        ));
    }
    let diff = observed - estimated;
    Ok(nonnegative(diff.component_mul(&diff).dot(weights)))
}

/// GLS discrepancy, including the optional correlation correction.
pub fn generalized_least_squares(
    stats: &SampleStats,
    covariance: &Matrix,
    mean: Option<&Vector>,
    correlation: bool,
) -> Result<f64> {
    let p = stats.nvars();
    if covariance.shape() != (p, p) {
        return Err(Error::Dimension("GLS covariance dimensions differ".into()));
    }
    matrix::symmetric(covariance)?;
    let inverse = Spd::new(stats.covariance())?.inverse();
    let tmp = &inverse * (stats.covariance() - covariance);
    let mut value = 0.5 * tmp.component_mul(&tmp.transpose()).sum();
    if correlation {
        let diagonal = tmp.diagonal();
        let tt = Matrix::identity(p, p) + stats.covariance().component_mul(&inverse);
        let solved = tt.lu().solve(&diagonal).ok_or(Error::Singular)?;
        value -= diagonal.dot(&solved);
    }
    if let Some(mu) = mean {
        if mu.len() != p {
            return Err(Error::Dimension("GLS mean dimensions differ".into()));
        }
        let diff = stats.mean() - mu;
        value += diff.dot(&(&inverse * &diff));
    }
    Ok(nonnegative(value))
}
