use lavan::missing::MissingData;
use lavan::{
    Matrix, Vector,
    fit::{self, Comparison},
    matrix::{self, Triangle},
    mvnorm::Normal,
    objective::{self, MaximumLikelihood},
    sample::{SampleStats, VarianceDivisor, weighted_variances},
};
use serde_json::Value;
use std::process::Command;

fn number(x: &Value) -> f64 {
    if let Some(v) = x.as_f64() {
        return v;
    }
    match x.as_str() {
        Some("NA" | "NaN") => f64::NAN,
        Some("Inf") => f64::INFINITY,
        Some("-Inf") => f64::NEG_INFINITY,
        _ => panic!("unexpected R number: {x}"),
    }
}

fn numbers(x: &Value) -> Vec<f64> {
    if let Some(a) = x.as_array() {
        a.iter().map(number).collect()
    } else if x.is_null() {
        Vec::new()
    } else {
        vec![number(x)]
    }
}
fn indices(x: &Value) -> Vec<usize> {
    numbers(x).into_iter().map(|x| x as usize).collect()
}
fn matrix(x: &Value) -> Matrix {
    Matrix::from_column_slice(
        x["rows"].as_u64().unwrap() as usize,
        x["cols"].as_u64().unwrap() as usize,
        &numbers(&x["data"]),
    )
}
fn vector(x: &Value) -> Vector {
    Vector::from_vec(numbers(x))
}

#[derive(Default)]
struct Comparisons {
    count: usize,
    max_absolute: f64,
    max_scaled: f64,
    absolute_case: String,
    scaled_case: String,
}

impl Comparisons {
    fn close(&mut self, label: &str, actual: f64, expected: f64) {
        self.count += 1;
        if actual == expected || (actual.is_nan() && expected.is_nan()) {
            return;
        }
        let error = (actual - expected).abs();
        let scale = 1.0_f64.max(expected.abs());
        if error > self.max_absolute {
            self.max_absolute = error;
            self.absolute_case =
                format!("{label}, Rust={actual:.17e}, R={expected:.17e}, error={error:.3e}");
        }
        if error / scale > self.max_scaled {
            self.max_scaled = error / scale;
            self.scaled_case = format!(
                "{label}, Rust={actual:.17e}, R={expected:.17e}, scaled error={:.3e}",
                error / scale
            );
        }
        assert!(
            actual.is_finite() && expected.is_finite() && error <= 3e-10 * scale,
            "{label}: Rust={actual:.17e}, R={expected:.17e}, error={error:.3e}"
        );
    }

    fn slice(&mut self, label: &str, actual: &[f64], expected: &Value) {
        let expected = numbers(expected);
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (i, (&a, b)) in actual.iter().zip(expected).enumerate() {
            self.close(&format!("{label}[{i}]"), a, b);
        }
    }

    fn matrix(&mut self, label: &str, actual: &Matrix, expected: &Value) {
        assert_eq!(actual.shape(), matrix(expected).shape(), "{label} shape");
        self.slice(label, actual.as_slice(), &expected["data"]);
    }
}

/// Runs the actual local lavaan package; missing R or a stale oracle is a failure.
#[test]
fn local_lavaan_equivalence() {
    let output = Command::new("Rscript")
        .args(["--vanilla", "tests/r/oracle.R"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Rscript is required for equivalence tests");
    assert!(
        output.status.success(),
        "R oracle failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Value =
        serde_json::from_slice(&output.stdout).expect("R oracle must emit valid JSON");
    assert_eq!(oracle["version"], "0.7-3.3395");
    let mut c = Comparisons::default();
    for case in oracle["matrices"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let a = matrix(&case["a"]);
        let b = matrix(&case["b"]);
        let s = matrix(&case["s"]);
        c.slice("vec", &matrix::vec(&a), &case["vec"]);
        c.slice("vecr", &matrix::vecr(&a), &case["vecr"]);
        assert_eq!(matrix::diagonal_indices(n).unwrap(), indices(&case["diag"]));
        assert_eq!(
            matrix::half_diagonal_indices(n).unwrap(),
            indices(&case["diagh"])
        );
        assert_eq!(
            matrix::antidiagonal_indices(n).unwrap(),
            indices(&case["antidiag"])
        );
        let orders = [
            Triangle::LowerColumn,
            Triangle::LowerRow,
            Triangle::UpperColumn,
            Triangle::UpperRow,
        ];
        for half in case["half"].as_array().unwrap() {
            let diagonal = half["diagonal"].as_bool().unwrap();
            for (i, &order) in orders.iter().enumerate() {
                assert_eq!(
                    matrix::triangle_indices(n, order, diagonal).unwrap(),
                    indices(&half["indices"][i])
                );
                let v = matrix::half_vectorize(&a, order, diagonal).unwrap();
                c.slice("half vectorization", &v, &half["values"][i]);
                c.matrix(
                    "half reverse",
                    &matrix::half_reverse(&v, order, diagonal).unwrap(),
                    &half["reverses"][i],
                );
            }
        }
        c.matrix(
            "duplication",
            &matrix::duplication(n).unwrap(),
            &case["duplication"],
        );
        c.matrix(
            "duplication inverse",
            &matrix::duplication_inverse(n).unwrap(),
            &case["duplication_inverse"],
        );
        for (inverse, prefix) in [(false, "dup"), (true, "ginv")] {
            c.matrix(
                prefix,
                &matrix::duplication_pre(&b, inverse).unwrap(),
                &case[format!("{prefix}_pre")],
            );
            c.matrix(
                prefix,
                &matrix::duplication_post(&b, inverse).unwrap(),
                &case[format!("{prefix}_post")],
            );
            c.matrix(
                prefix,
                &matrix::duplication_pre_post(&b, inverse).unwrap(),
                &case[format!("{prefix}_both")],
            );
        }
        c.matrix(
            "commutation",
            &matrix::commutation(n, n + 2).unwrap(),
            &case["commutation"],
        );
        c.matrix(
            "com_pre",
            &matrix::commutation_pre(&b, n, n).unwrap(),
            &case["com_pre"],
        );
        c.matrix(
            "com_post",
            &matrix::commutation_post(&b).unwrap(),
            &case["com_post"],
        );
        c.matrix(
            "com_both",
            &matrix::commutation_pre_post(&b).unwrap(),
            &case["com_both"],
        );
        c.matrix(
            "symmetric_sqrt",
            &matrix::symmetric_sqrt(&s).unwrap(),
            &case["sqrt"],
        );
        let spd = matrix::Spd::new(&s).unwrap();
        c.matrix("inverse", &spd.inverse(), &case["inverse"]);
        c.close("logdet", spd.logdet(), number(&case["logdet"]));
        c.matrix(
            "correlation",
            &matrix::covariance_to_correlation(&s, None).unwrap(),
            &case["cor"],
        );
        c.matrix(
            "cor2cov",
            &matrix::correlation_to_covariance(
                &matrix(&case["cor"]),
                s.diagonal().map(f64::sqrt).as_slice(),
            )
            .unwrap(),
            &case["cor2cov"],
        );
        c.matrix(
            "partial correlation",
            &matrix::covariance_to_correlation(&s, Some(&[0, n - 1])).unwrap(),
            &case["partial_cor"],
        );
        let x = matrix(&case["x"]);
        c.matrix(
            "block diagonal",
            &matrix::block_diagonal(&[a.clone(), x.clone()]).unwrap(),
            &case["bdiag"],
        );
        c.close(
            "trace product",
            matrix::trace_product(&[a, x.clone(), x.transpose()]).unwrap(),
            number(&case["trace_product"]),
        );
    }
    for case in oracle["samples"].as_array().unwrap() {
        let y = matrix(&case["y"]);
        let weights = (!case["weights"].is_null()).then(|| numbers(&case["weights"]));
        let weights = weights.as_deref();
        let stats = SampleStats::from_data(&y, weights).unwrap();
        c.slice("sample mean", stats.mean().as_slice(), &case["mean"]);
        c.matrix("sample covariance", stats.covariance(), &case["cov"]);
        c.close("nobs", stats.nobs(), number(&case["nobs"]));
        let normal = Normal::new(vector(&case["mu"]), matrix(&case["sigma"])).unwrap();
        for variant in case["variants"].as_array().unwrap() {
            let fixed = indices(&variant["fixed"]);
            c.slice(
                "log density",
                normal
                    .log_densities(&y, weights, &fixed)
                    .unwrap()
                    .as_slice(),
                &variant["log_density"],
            );
            c.close(
                "log likelihood raw",
                normal.log_likelihood(&y, weights, &fixed).unwrap(),
                number(&variant["log_likelihood"]),
            );
            c.close(
                "log likelihood stats",
                normal.log_likelihood_stats(&stats, &fixed).unwrap(),
                number(&variant["log_likelihood"]),
            );
            c.matrix(
                "scores",
                &normal.scores(&y, weights, &fixed, true).unwrap(),
                &variant["scores"],
            );
            c.slice(
                "gradient",
                normal
                    .gradient(&y, weights, &fixed, true)
                    .unwrap()
                    .as_slice(),
                &variant["gradient"],
            );
            c.matrix(
                "expected information",
                &normal.information_expected(&fixed, true, false).unwrap(),
                &variant["expected"],
            );
            c.matrix(
                "observed information",
                &normal.information_observed(&stats, &fixed, true).unwrap(),
                &variant["observed"],
            );
            c.matrix(
                "inverse expected information",
                &normal.inverse_information_expected(&fixed, true).unwrap(),
                &variant["inverse_expected"],
            );
            c.matrix(
                "first-order information",
                &normal
                    .information_first_order(&y, weights, &fixed, true, None)
                    .unwrap(),
                &variant["first_order"],
            );
            let clusters: Vec<usize> = (0..y.nrows()).map(|i| i % 9).collect();
            c.matrix(
                "clustered information",
                &normal
                    .information_first_order(&y, weights, &fixed, true, Some(&clusters))
                    .unwrap(),
                &variant["cluster"],
            );
        }
        c.slice(
            "ML variances",
            weighted_variances(&y, weights, VarianceDivisor::MaximumLikelihood)
                .unwrap()
                .as_slice(),
            &case["variance_ml"],
        );
        c.slice(
            "frequency variances",
            weighted_variances(&y, weights, VarianceDivisor::UnbiasedFrequency)
                .unwrap()
                .as_slice(),
            &case["variance_unbiased"],
        );
        let ml = MaximumLikelihood::new(stats.clone()).unwrap();
        c.close(
            "ML objective",
            ml.evaluate(normal.covariance(), Some(normal.mean()))
                .unwrap(),
            number(&case["ml"]),
        );
        c.close(
            "GLS objective",
            objective::generalized_least_squares(
                &stats,
                normal.covariance(),
                Some(normal.mean()),
                false,
            )
            .unwrap(),
            number(&case["gls"]),
        );
        c.close(
            "correlation GLS",
            objective::generalized_least_squares(
                &stats,
                normal.covariance(),
                Some(normal.mean()),
                true,
            )
            .unwrap(),
            number(&case["gls_cor"]),
        );
        c.matrix(
            "correlation information",
            &normal.information_expected(&[], false, true).unwrap(),
            &case["expected_cor"],
        );
        c.close(
            "WLS objective",
            objective::weighted_least_squares(stats.mean(), normal.mean(), normal.covariance())
                .unwrap(),
            number(&case["wls"]),
        );
        c.close(
            "DWLS objective",
            objective::diagonally_weighted_least_squares(
                stats.mean(),
                normal.mean(),
                &normal.covariance().diagonal(),
            )
            .unwrap(),
            number(&case["dwls"]),
        );
        c.matrix(
            "transform moments",
            &lavan::sample::transform_mean_covariance(&y, normal.mean(), normal.covariance())
                .unwrap(),
            &case["transformed"],
        );
    }
    for case in oracle["fits"].as_array().unwrap() {
        let x = numbers(&case["input"]);
        let comparison = Comparison {
            chi_square: x[0],
            df: x[1],
            baseline_chi_square: x[2],
            baseline_df: x[3],
            correction: x[4],
            baseline_correction: x[5],
        };
        for (name, actual) in [
            ("cfi", comparison.cfi()),
            ("rni", comparison.rni()),
            ("tli", comparison.tli()),
            ("rfi", comparison.rfi()),
            ("nfi", comparison.nfi()),
            ("pnfi", comparison.pnfi()),
            ("ifi", comparison.ifi()),
            ("rmsea", fit::rmsea(x[0], x[1], 301.0, 2.0, x[4])),
            ("aic", fit::aic(-200.0, x[1])),
            ("bic", fit::bic(-200.0, x[1], 301.0)),
            ("sabic", fit::sabic(-200.0, x[1], 301.0)),
        ] {
            c.close(name, actual, number(&case[name]));
        }
    }
    for case in oracle["misc"].as_array().unwrap() {
        c.matrix(
            "inverse(I-A)",
            &matrix::inverse_iminus(&matrix(&case["a"])).unwrap(),
            &case["inverse_iminus"],
        );
    }
    for case in oracle["missing"].as_array().unwrap() {
        let weights = (!case["weights"].is_null()).then(|| numbers(&case["weights"]));
        let missing = MissingData::new(matrix(&case["y"]), weights.as_deref()).unwrap();
        let normal = Normal::new(vector(&case["mu"]), matrix(&case["sigma"])).unwrap();
        c.slice(
            "missing weighted variance",
            weighted_variances(
                missing.data(),
                weights.as_deref(),
                VarianceDivisor::MaximumLikelihood,
            )
            .unwrap()
            .as_slice(),
            &case["missing_variance"],
        );
        c.slice(
            "missing log densities",
            missing.log_densities(&normal).unwrap().as_slice(),
            &case["log_density"],
        );
        c.close(
            "missing log likelihood",
            missing.log_likelihood(&normal).unwrap(),
            number(&case["log_likelihood"]),
        );
        c.matrix(
            "conditional mean imputation",
            &missing.impute(&normal).unwrap(),
            &case["imputed"],
        );
        let (t1, t2) = missing.expectation_step(&normal).unwrap();
        c.slice("E-step first moment", t1.as_slice(), &case["t1"]);
        c.matrix("E-step second moment", &t2, &case["t2"]);
        for v in case["variants"].as_array().unwrap() {
            let fixed = indices(&v["fixed"]);
            c.matrix(
                "missing scores",
                &missing.scores(&normal, &fixed, true).unwrap(),
                &v["scores"],
            );
            c.slice(
                "missing gradient",
                missing.gradient(&normal, &fixed, true).unwrap().as_slice(),
                &v["gradient"],
            );
            c.matrix(
                "missing expected information",
                &missing.information(&normal, &fixed, true, false).unwrap(),
                &v["expected"],
            );
            c.matrix(
                "missing observed information",
                &missing.information(&normal, &fixed, true, true).unwrap(),
                &v["observed"],
            );
            c.matrix(
                "missing first order information",
                &missing
                    .information_first_order(&normal, &fixed, true, None)
                    .unwrap(),
                &v["first_order"],
            );
            let clusters: Vec<usize> = (0..missing.data().nrows()).map(|i| i % 7).collect();
            c.matrix(
                "missing clustered information",
                &missing
                    .information_first_order(&normal, &fixed, true, Some(&clusters))
                    .unwrap(),
                &v["clustered"],
            );
        }
    }
    for case in oracle["variance_edges"].as_array().unwrap() {
        let y = matrix(&case["y"]);
        let weights = (!case["weights"].is_null()).then(|| numbers(&case["weights"]));
        c.slice(
            "degenerate ML variance",
            weighted_variances(&y, weights.as_deref(), VarianceDivisor::MaximumLikelihood)
                .unwrap()
                .as_slice(),
            &case["ml"],
        );
        c.slice(
            "degenerate unbiased variance",
            weighted_variances(&y, weights.as_deref(), VarianceDivisor::UnbiasedFrequency)
                .unwrap()
                .as_slice(),
            &case["unbiased"],
        );
    }
    let shifted = &oracle["shifted"];
    let statistics = SampleStats::from_data(&matrix(&shifted["y"]), None).unwrap();
    // These chosen large locations must survive the oracle's JSON round trip.
    assert_eq!(
        statistics.mean().as_slice(),
        numbers(&shifted["mean"]).as_slice()
    );
    c.slice(
        "large location mean",
        statistics.mean().as_slice(),
        &shifted["mean"],
    );
    c.matrix(
        "large location covariance",
        statistics.covariance(),
        &shifted["covariance"],
    );
    let weighted =
        SampleStats::from_data(&matrix(&shifted["y"]), Some(&numbers(&shifted["weights"])))
            .unwrap();
    c.slice(
        "large location weighted mean",
        weighted.mean().as_slice(),
        &shifted["weighted_mean"],
    );
    c.matrix(
        "large location weighted covariance",
        weighted.covariance(),
        &shifted["weighted_covariance"],
    );
    let sqrt = &oracle["indefinite_sqrt"];
    c.matrix(
        "indefinite spectral truncation",
        &matrix::symmetric_sqrt(&matrix(&sqrt["a"])).unwrap(),
        &sqrt["root"],
    );
    println!(
        "R equivalence: {} scalar comparisons passed; max abs error {:.3e}; max scaled error {:.3e}; tolerance 3e-10 * max(1, |R|)",
        c.count, c.max_absolute, c.max_scaled
    );
    println!("Worst absolute comparison: {}", c.absolute_case);
    println!("Worst scaled comparison: {}", c.scaled_case);
}
