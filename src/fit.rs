//! Scalar fit indices translated from `lav_fit_cfi.R`, `lav_fit_rmsea.R`,
//! and `lav_fit_aic.R`. NaN represents R's missing numeric results.

/// R's `isTRUE(all.equal(value, 0))` uses an absolute sqrt(epsilon) tolerance.
fn approximately_zero(value: f64) -> bool {
    value.abs() <= f64::EPSILON.sqrt()
}
fn any_nan(values: &[f64]) -> bool {
    values.iter().any(|x| x.is_nan())
}
fn rmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Comparison {
    pub chi_square: f64,
    pub df: f64,
    pub baseline_chi_square: f64,
    pub baseline_df: f64,
    pub correction: f64,
    pub baseline_correction: f64,
}

impl Comparison {
    pub fn new(chi_square: f64, df: f64, baseline_chi_square: f64, baseline_df: f64) -> Self {
        Self {
            chi_square,
            df,
            baseline_chi_square,
            baseline_df,
            correction: 1.0,
            baseline_correction: 1.0,
        }
    }

    fn has_nan(self) -> bool {
        any_nan(&[
            self.chi_square,
            self.df,
            self.baseline_chi_square,
            self.baseline_df,
            self.correction,
            self.baseline_correction,
        ])
    }

    fn residuals(self) -> (f64, f64) {
        let robust = self.df > 0.0 && self.correction != 1.0 && self.baseline_correction != 1.0;
        if robust {
            (
                self.chi_square - self.correction * self.df,
                self.baseline_chi_square - self.baseline_correction * self.baseline_df,
            )
        } else {
            (
                self.chi_square - self.df,
                self.baseline_chi_square - self.baseline_df,
            )
        }
    }

    pub fn cfi(self) -> f64 {
        if self.has_nan() {
            return f64::NAN;
        }
        let (a, b) = self.residuals();
        let t1 = rmax(a, 0.0);
        let t2 = rmax(rmax(a, b), 0.0);
        if approximately_zero(t1) && approximately_zero(t2) {
            1.0
        } else {
            1.0 - t1 / t2
        }
    }

    pub fn rni(self) -> f64 {
        if self.has_nan() {
            return f64::NAN;
        }
        let (a, b) = self.residuals();
        if approximately_zero(b) || !a.is_finite() || !b.is_finite() {
            f64::NAN
        } else {
            1.0 - a / b
        }
    }

    pub fn tli(self) -> f64 {
        if self.has_nan() {
            return f64::NAN;
        }
        let (a, b) = self.residuals();
        let (t1, t2) = (a * self.baseline_df, b * self.df);
        if self.df > 0.0 && t2.abs() > 0.0 {
            1.0 - t1 / t2
        } else if !t1.is_finite() || !t2.is_finite() {
            f64::NAN
        } else {
            1.0
        }
    }

    pub fn rfi(self) -> f64 {
        if any_nan(&[
            self.chi_square,
            self.df,
            self.baseline_chi_square,
            self.baseline_df,
        ]) || self.df > self.baseline_df
        {
            return f64::NAN;
        }
        if self.df > 0.0 && self.baseline_df > 0.0 {
            let t2 = self.baseline_chi_square / self.baseline_df;
            let t1 = t2 - self.chi_square / self.df;
            if !t1.is_finite() || !t2.is_finite() {
                f64::NAN
            } else if t1 < 0.0 || t2 < 0.0 {
                1.0
            } else {
                t1 / t2
            }
        } else {
            1.0
        }
    }

    pub fn nfi(self) -> f64 {
        if any_nan(&[
            self.chi_square,
            self.df,
            self.baseline_chi_square,
            self.baseline_df,
        ]) || self.df > self.baseline_df
            || approximately_zero(self.baseline_chi_square)
        {
            f64::NAN
        } else if self.df > 0.0 {
            (self.baseline_chi_square - self.chi_square) / self.baseline_chi_square
        } else {
            1.0
        }
    }

    pub fn pnfi(self) -> f64 {
        if any_nan(&[
            self.chi_square,
            self.df,
            self.baseline_chi_square,
            self.baseline_df,
        ]) {
            return f64::NAN;
        }
        if self.baseline_df > 0.0 && self.baseline_chi_square > 0.0 {
            (self.df / self.baseline_df)
                * ((self.baseline_chi_square - self.chi_square) / self.baseline_chi_square)
        } else {
            f64::NAN
        }
    }

    pub fn ifi(self) -> f64 {
        if any_nan(&[
            self.chi_square,
            self.df,
            self.baseline_chi_square,
            self.baseline_df,
        ]) {
            return f64::NAN;
        }
        let (t1, t2) = (
            self.baseline_chi_square - self.chi_square,
            self.baseline_chi_square - self.df,
        );
        if !t1.is_finite() || !t2.is_finite() {
            f64::NAN
        } else if t2 < 0.0 {
            1.0
        } else if approximately_zero(t2) {
            f64::NAN
        } else {
            t1 / t2
        }
    }
}

pub fn rmsea(chi_square: f64, df: f64, nobs: f64, groups: f64, correction: f64) -> f64 {
    if df.is_nan() {
        return f64::NAN;
    }
    if df > 0.0 {
        rmax((chi_square / nobs) / df - correction / nobs, 0.0).sqrt() * groups.sqrt()
    } else {
        0.0
    }
}

pub fn rmsea_population(discrepancy: f64, df: f64) -> f64 {
    (discrepancy / df).sqrt()
}
pub fn aic(log_likelihood: f64, parameters: f64) -> f64 {
    -2.0 * log_likelihood + 2.0 * parameters
}
pub fn bic(log_likelihood: f64, parameters: f64, nobs: f64) -> f64 {
    -2.0 * log_likelihood + parameters * nobs.ln()
}
pub fn sabic(log_likelihood: f64, parameters: f64, nobs: f64) -> f64 {
    -2.0 * log_likelihood + parameters * ((nobs + 2.0) / 24.0).ln()
}
