use lavan::{Error, Matrix, cfa};

fn main() -> lavan::Result<()> {
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
    )?;
    if !fitted.fit.solution.converged() {
        return Err(Error::Optimization(format!(
            "{:?}; max gradient {}",
            fitted.fit.solution.status,
            fitted.fit.solution.gradient.amax()
        )));
    }
    println!(
        "chi-square={:.8}, df={}",
        fitted.fit.chi_square, fitted.fit.df
    );
    for (parameter, value) in fitted
        .model
        .parameters()
        .iter()
        .zip(&fitted.fit.solution.parameters)
    {
        println!("{}: {:.6}", parameter.name, value);
    }
    Ok(())
}
