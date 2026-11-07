use lavan::{
    Error, Matrix, Vector,
    matrix::{self, Triangle},
    mvnorm::Normal,
    objective::MaximumLikelihood,
    sample::SampleStats,
};

fn close(a: &Matrix, b: &Matrix, tolerance: f64) {
    assert_eq!(a.shape(), b.shape());
    assert!((a - b).amax() < tolerance, "error {}", (a - b).amax());
}

#[test]
fn duplication_projection_and_transpose_permutation() {
    for n in 1..10 {
        let d = matrix::duplication(n).unwrap();
        let dp = matrix::duplication_inverse(n).unwrap();
        close(
            &(&dp * &d),
            &Matrix::identity(dp.nrows(), dp.nrows()),
            1e-14,
        );
        let projection = &d * &dp;
        close(&(&projection * &projection), &projection, 1e-14);
        let a = Matrix::from_fn(n, n + 2, |i, j| (i + j * n) as f64);
        let k = matrix::commutation(n, n + 2).unwrap();
        close(
            &(&k * Matrix::from_column_slice(a.len(), 1, a.as_slice())),
            &Matrix::from_column_slice(a.len(), 1, a.transpose().as_slice()),
            1e-14,
        );
        close(
            &(matrix::commutation(n + 2, n).unwrap() * k),
            &Matrix::identity(a.len(), a.len()),
            1e-14,
        );
    }
}

#[test]
fn normal_scores_and_observed_information_are_derivatives() {
    let y = Matrix::from_row_slice(4, 2, &[1.2, -0.4, -0.2, 1.8, 0.9, 2.2, 1.1, 0.1]);
    let parameters = [0.3, -0.1, 1.6, 0.2, 1.2];
    let build = |x: &[f64]| {
        Normal::new(
            Vector::from_vec(x[..2].to_vec()),
            matrix::half_reverse(&x[2..], Triangle::LowerColumn, true).unwrap(),
        )
        .unwrap()
    };
    let normal = build(&parameters);
    let gradient = normal.gradient(&y, None, &[], true).unwrap();
    let stats = SampleStats::from_data(&y, None).unwrap();
    let observed = normal.information_observed(&stats, &[], true).unwrap();
    for i in 0..parameters.len() {
        let h = 1e-5;
        let mut plus = parameters;
        plus[i] += h;
        let mut minus = parameters;
        minus[i] -= h;
        let finite_difference = (build(&plus).log_likelihood(&y, None, &[]).unwrap()
            - build(&minus).log_likelihood(&y, None, &[]).unwrap())
            / (2.0 * h);
        assert!((gradient[i] - finite_difference).abs() < 2e-8);
        let dg = (build(&plus).gradient(&y, None, &[], true).unwrap()
            - build(&minus).gradient(&y, None, &[], true).unwrap())
            / (2.0 * h);
        for j in 0..parameters.len() {
            assert!((dg[j] + 4.0 * observed[(j, i)]).abs() < 3e-8);
        }
    }
    let expected = normal.information_expected(&[], true, false).unwrap();
    let inverse = normal.inverse_information_expected(&[], true).unwrap();
    close(&(expected * inverse), &Matrix::identity(5, 5), 1e-12);
}

#[test]
fn ml_has_zero_discrepancy_at_saturated_moments() {
    let y = Matrix::from_row_slice(4, 2, &[1.0, 2.0, 2.0, 1.0, 0.0, 3.0, 4.0, 3.0]);
    let stats = SampleStats::from_data(&y, None).unwrap();
    let objective = MaximumLikelihood::new(stats.clone()).unwrap();
    assert!(
        objective
            .evaluate(stats.covariance(), Some(stats.mean()))
            .unwrap()
            < 1e-14
    );
    let (ds, dm) = objective
        .gradient(stats.covariance(), Some(stats.mean()))
        .unwrap();
    assert!(ds.amax() < 1e-14);
    assert!(dm.unwrap().amax() < 1e-14);
    let bad = Matrix::from_diagonal(&Vector::from_vec(vec![-1.0, 1.0]));
    assert_eq!(objective.evaluate(&bad, None).unwrap(), f64::INFINITY);
}

#[test]
fn malformed_and_unsupported_domains_are_explicit() {
    assert!(matrix::half_reverse(&[1.0, 2.0], Triangle::LowerColumn, true).is_err());
    assert!(matrix::duplication(0).is_err());
    assert!(matrix::triangle_indices(usize::MAX, Triangle::LowerColumn, true).is_err());
    assert!(matrix::half_len(usize::MAX, true).is_err());
    assert!(matrix::duplication_pre(&Matrix::zeros(3, 2), false).is_err());
    assert!(matrix::Spd::new(&Matrix::zeros(2, 2)).is_err());
    assert!(matches!(
        Normal::new(
            Vector::zeros(2),
            Matrix::from_row_slice(2, 2, &[1.0, 0.5, 0.1, 1.0])
        ),
        Err(Error::NotSymmetric)
    ));
    assert!(SampleStats::from_data(&Matrix::from_element(2, 1, f64::NAN), None).is_err());
    assert!(SampleStats::from_data(&Matrix::zeros(2, 1), Some(&[-1.0, 2.0])).is_err());
    assert_eq!(
        matrix::half_reverse(&[], Triangle::LowerColumn, false)
            .unwrap()
            .shape(),
        (1, 1)
    );
}
