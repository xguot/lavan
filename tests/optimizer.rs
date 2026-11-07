use lavan::{
    Vector,
    optim::{self, Options, Status},
};

#[test]
fn bfgs_solves_a_curved_objective_and_reports_limits() {
    let rosenbrock = |x: &Vector| {
        let (a, b) = (x[0], x[1]);
        let residual = b - a * a;
        Ok((
            (1.0 - a).powi(2) + 100.0 * residual.powi(2),
            Vector::from_vec(vec![
                -2.0 * (1.0 - a) - 400.0 * a * residual,
                200.0 * residual,
            ]),
        ))
    };
    let initial = Vector::from_vec(vec![-1.2, 1.0]);
    let solution = optim::minimize(initial.clone(), Options::default(), rosenbrock).unwrap();
    assert_eq!(solution.status, Status::Converged);
    assert!((solution.parameters - Vector::from_element(2, 1.0)).amax() < 1e-7);
    let limited = optim::minimize(
        initial,
        Options {
            max_iterations: 1,
            ..Default::default()
        },
        rosenbrock,
    )
    .unwrap();
    assert_eq!(limited.status, Status::IterationLimit);
    assert!(
        optim::minimize(Vector::zeros(1), Options::default(), |_| Ok((
            f64::INFINITY,
            Vector::zeros(1)
        )))
        .is_err()
    );
    assert!(
        optim::minimize(Vector::zeros(1), Options::default(), |_| Ok((
            0.0,
            Vector::zeros(2)
        )))
        .is_err()
    );
}
