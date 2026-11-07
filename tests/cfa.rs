use lavan::{CfaOptions, Matrix, cfa, optim};
use serde_json::Value;
use std::process::Command;

fn numbers(x: &Value) -> Vec<f64> {
    if let Some(x) = x.as_array() {
        x.iter().map(|v| v.as_f64().unwrap()).collect()
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
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol * 1.0_f64.max(b.abs()), "Rust {a} R {b}");
}

#[test]
fn cfa_frontend_matches_r_defaults_and_supported_modifiers() {
    let output = Command::new("Rscript")
        .args(["--vanilla", "tests/r/cfa.R"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).unwrap();
    for case in cases.as_array().unwrap() {
        let spec = &case["spec"];
        let columns: Vec<&str> = case["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();
        let fitted = cfa(
            spec["syntax"].as_str().unwrap(),
            &matrix(&case["y"]),
            &columns,
            CfaOptions {
                meanstructure: spec["meanstructure"].as_bool().unwrap(),
                std_lv: spec["std_lv"].as_bool().unwrap(),
                orthogonal: spec["orthogonal"].as_bool().unwrap(),
                optimizer: optim::Options {
                    gradient_tolerance: 2e-7,
                    ..Default::default()
                },
            },
        )
        .unwrap();
        println!(
            "CFA {}: {:?} after {} iterations",
            spec["name"], fitted.fit.solution.status, fitted.fit.solution.iterations
        );
        assert!(fitted.fit.solution.converged(), "{:?}", fitted.fit.solution);
        close(
            fitted.fit.chi_square,
            case["chi_square"].as_f64().unwrap(),
            1e-9,
        );
        close(
            fitted.fit.log_likelihood,
            case["log_likelihood"].as_f64().unwrap(),
            1e-10,
        );
        assert_eq!(fitted.fit.df, case["df"].as_i64().unwrap() as isize);
        for (&a, &b) in fitted
            .fit
            .implied
            .covariance
            .iter()
            .zip(matrix(&case["covariance"]).iter())
        {
            close(a, b, 2e-5);
        }
        let matrices = fitted
            .model
            .matrices(&fitted.fit.solution.parameters)
            .unwrap();
        for (name, actual) in [
            ("lambda", matrices.lambda),
            ("psi", matrices.psi),
            ("theta", matrices.theta),
        ] {
            let expected = matrix(&case["estimated"][name]);
            for (&a, &b) in actual.iter().zip(expected.iter()) {
                close(a, b, 2e-5);
            }
        }
    }
}

#[test]
fn unsupported_syntax_is_rejected_before_fitting() {
    let y = Matrix::from_row_slice(
        5,
        3,
        &[
            1.0, 2.0, 0.0, 2.0, 1.0, 3.0, 3.0, 2.0, 1.0, 4.0, 6.0, 3.0, 5.0, 3.0, 7.0,
        ],
    );
    for syntax in [
        "f =~ x1 + x2 + x3\nx1 ~ x2",
        "f =~ start(1)*x1 + x2 + x3",
        "f =~ x1 + x2 + x3\na := x1",
        "f =~ x1 + x2 + x3\nx1 | t1",
        "f =~ x1 + x1 + x2",
    ] {
        assert!(
            cfa(syntax, &y, &["x1", "x2", "x3"], Default::default()).is_err(),
            "{syntax}"
        );
    }
}
