use lavan::{Matrix, cfa};

#[test]
fn default_controls_fit_an_exact_factor_covariance() {
    let loadings = [1.0, 0.8, 1.2, 0.7];
    let data = Matrix::from_fn(400, 4, |i, j| {
        let t = std::f64::consts::TAU * i as f64 / 400.0;
        loadings[j] * t.sin() + 0.5 * ((j + 2) as f64 * t).cos()
    });
    let fitted = cfa(
        "f =~ x1 + x2 + x3 + x4",
        &data,
        &["x1", "x2", "x3", "x4"],
        Default::default(),
    )
    .unwrap();
    assert!(fitted.fit.solution.converged(), "{:?}", fitted.fit.solution);
    assert!(fitted.fit.chi_square < 1e-9);
    let matrices = fitted
        .model
        .matrices(&fitted.fit.solution.parameters)
        .unwrap();
    for (i, &loading) in loadings.iter().enumerate() {
        assert!((matrices.lambda[(i, 0)] - loading).abs() < 1e-6);
    }
}
