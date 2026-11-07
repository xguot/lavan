use lavan::{
    Matrix, Vector, fit,
    model::{Lisrel, Member, Model, Parameter, Slot},
    optim,
    sample::SampleStats,
};
use serde_json::Value;
use std::process::Command;

fn numbers(x: &Value) -> Vec<f64> {
    if let Some(x) = x.as_array() {
        x.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect()
    } else {
        vec![x.as_f64().unwrap()]
    }
}
fn matrix(x: &Value) -> Matrix {
    Matrix::from_column_slice(
        x["rows"].as_u64().unwrap() as usize,
        x["cols"].as_u64().unwrap() as usize,
        &numbers(&x["data"]),
    )
}
fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * 1.0_f64.max(expected.abs()),
        "{label}: Rust={actual:.16e} R={expected:.16e}"
    );
}
fn assert_matrix(label: &str, actual: &Matrix, expected: &Matrix, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape(), "{label}");
    for (&a, &b) in actual.iter().zip(expected.iter()) {
        assert_close(label, a, b, tolerance);
    }
}

#[test]
fn fits_and_standard_errors_match_local_lavaan() {
    let output = Command::new("Rscript")
        .args(["--vanilla", "tests/r/models.R"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).expect("model oracle JSON");
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let start = &case["start"];
        let mut base = Lisrel::new(
            matrix(&start["lambda"]),
            matrix(&start["theta"]),
            matrix(&start["psi"]),
        )
        .unwrap();
        if !start["beta"].is_null() {
            base.beta = matrix(&start["beta"]);
        }
        if !start["nu"].is_null() {
            base.nu = Vector::from_vec(numbers(&start["nu"]["data"]));
        }
        if !start["alpha"].is_null() {
            base.alpha = Vector::from_vec(numbers(&start["alpha"]["data"]));
        }
        let r_parameters = numbers(&case["coefficients"]);
        let mut parameters: Vec<Parameter> = (0..r_parameters.len())
            .map(|i| Parameter {
                name: format!("p{}", i + 1),
                initial: 0.0,
                slots: Vec::new(),
            })
            .collect();
        for (member, key) in [
            (Member::Lambda, "lambda"),
            (Member::Theta, "theta"),
            (Member::Psi, "psi"),
            (Member::Beta, "beta"),
            (Member::Nu, "nu"),
            (Member::Alpha, "alpha"),
        ] {
            if case["free"][key].is_null() {
                continue;
            }
            let free = matrix(&case["free"][key]);
            let starts = matrix(&start[key]);
            for col in 0..free.ncols() {
                for row in 0..free.nrows() {
                    let id = free[(row, col)] as usize;
                    if id == 0 || (matches!(member, Member::Theta | Member::Psi) && row < col) {
                        continue;
                    }
                    let parameter = &mut parameters[id - 1];
                    parameter.initial = starts[(row, col)];
                    parameter.slots.push(Slot { member, row, col });
                }
            }
        }
        let model = Model::new(base, parameters, case["meanstructure"].as_bool().unwrap())
            .unwrap()
            .with_baseline_covariates(
                numbers(&case["baseline_covariates"])
                    .into_iter()
                    .map(|i| i as usize)
                    .collect(),
            )
            .unwrap();
        let r_vector = Vector::from_vec(r_parameters.clone());
        assert_matrix(
            "analytic Jacobian",
            &model.jacobian(&r_vector).unwrap(),
            &matrix(&case["jacobian"]),
            3e-10,
        );
        let stats = SampleStats::new(
            Vector::from_vec(numbers(&case["sample_mean"])),
            matrix(&case["covariance"]),
            case["nobs"].as_f64().unwrap(),
        )
        .unwrap();
        let options = optim::Options {
            gradient_tolerance: 1e-7,
            ..Default::default()
        };
        let fitted = if case["missing_data"].is_null() {
            model.fit(stats, options).unwrap()
        } else {
            let missing =
                lavan::missing::MissingData::new(matrix(&case["missing_data"]), None).unwrap();
            model.fit_missing(&missing, options).unwrap()
        };
        println!(
            "{name}: {:?}, iterations={}, evaluations={}, gradient={:.3e}, chisq={:.10}",
            fitted.solution.status,
            fitted.solution.iterations,
            fitted.solution.evaluations,
            fitted.solution.gradient.amax(),
            fitted.chi_square
        );
        assert!(fitted.solution.converged(), "{name}: {:?}", fitted.solution);
        for (&a, b) in fitted.solution.parameters.iter().zip(r_parameters) {
            assert_close(name, a, b, 2e-5);
        }
        let measures = &case["measures"];
        assert_close(
            "chisq",
            fitted.chi_square,
            measures["chisq"].as_f64().unwrap(),
            1e-8,
        );
        assert_eq!(fitted.df, measures["df"].as_i64().unwrap() as isize);
        assert_close(
            "baseline chisq",
            fitted.baseline_chi_square,
            measures["baseline.chisq"].as_f64().unwrap(),
            1e-10,
        );
        assert_eq!(
            fitted.baseline_df,
            measures["baseline.df"].as_i64().unwrap() as isize
        );
        assert_close(
            "logl",
            fitted.log_likelihood,
            measures["logl"].as_f64().unwrap(),
            1e-10,
        );
        assert_close(
            "cfi",
            fitted.comparison().cfi(),
            measures["cfi"].as_f64().unwrap(),
            1e-9,
        );
        assert_close(
            "tli",
            fitted.comparison().tli(),
            measures["tli"].as_f64().unwrap(),
            1e-9,
        );
        assert_close(
            "rmsea",
            fit::rmsea(fitted.chi_square, fitted.df as f64, fitted.nobs, 1.0, 1.0),
            measures["rmsea"].as_f64().unwrap(),
            1e-9,
        );
        assert_close(
            "aic",
            fit::aic(
                fitted.log_likelihood,
                fitted.solution.parameters.len() as f64,
            ),
            measures["aic"].as_f64().unwrap(),
            1e-10,
        );
        assert_close(
            "bic",
            fit::bic(
                fitted.log_likelihood,
                fitted.solution.parameters.len() as f64,
                fitted.nobs,
            ),
            measures["bic"].as_f64().unwrap(),
            1e-10,
        );
        assert_matrix(
            "parameter vcov",
            fitted
                .parameter_covariance
                .as_ref()
                .expect("identified model covariance"),
            &matrix(&case["vcov"]),
            3e-5,
        );
    }
}
