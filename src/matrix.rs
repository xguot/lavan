//! Column-major matrix algebra corresponding to `R/lav_matrix.R`.
//!
//! Indices in this API are zero based. Dense duplication/commutation matrices
//! are provided for interoperability; use the implicit products for efficiency.

use crate::indexing::{column_index, transpose_index};
use crate::{Error, Matrix, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Triangle {
    LowerColumn,
    LowerRow,
    UpperColumn,
    UpperRow,
}

pub(crate) fn square(a: &Matrix) -> Result<usize> {
    if a.nrows() != a.ncols() {
        return Err(Error::Dimension("expected a square matrix".into()));
    }
    Ok(a.nrows())
}

pub(crate) fn finite(a: &Matrix, name: &'static str) -> Result<()> {
    if a.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFinite(name));
    }
    Ok(())
}

pub(crate) fn symmetric(a: &Matrix) -> Result<usize> {
    let n = square(a)?;
    finite(a, "symmetric matrix")?;
    for j in 0..n {
        for i in j + 1..n {
            let scale = 1.0_f64.max(a[(i, j)].abs()).max(a[(j, i)].abs());
            if (a[(i, j)] - a[(j, i)]).abs() > 1e-12 * scale {
                return Err(Error::NotSymmetric);
            }
        }
    }
    Ok(n)
}

fn checked_area(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .filter(|&n| n <= isize::MAX as usize / size_of::<f64>())
        .ok_or_else(|| Error::Dimension("matrix size overflows address space".into()))
}

/// Half-vector length, checked before index arithmetic or allocation.
pub fn half_len(n: usize, diagonal: bool) -> Result<usize> {
    let m = if diagonal {
        n.checked_add(1)
    } else {
        Some(n.saturating_sub(1))
    }
    .ok_or_else(|| Error::Dimension("triangle size overflow".into()))?;
    let (a, b) = if n.is_multiple_of(2) {
        (n / 2, m)
    } else {
        (n, m / 2)
    };
    a.checked_mul(b)
        .ok_or_else(|| Error::Dimension("triangle size overflow".into()))
}

/// Positions selected in a column-major vectorization, in lavaan order.
pub fn triangle_indices(n: usize, order: Triangle, diagonal: bool) -> Result<Vec<usize>> {
    checked_area(n, n)?;
    let mut out = Vec::with_capacity(half_len(n, diagonal)?);
    let skip = usize::from(!diagonal);
    match order {
        Triangle::LowerColumn => {
            for col in 0..n {
                for row in col + skip..n {
                    out.push(column_index(row, col, n));
                }
            }
        }
        Triangle::LowerRow => {
            for row in skip..n {
                for col in 0..row + 1 - skip {
                    out.push(column_index(row, col, n));
                }
            }
        }
        Triangle::UpperColumn => {
            for col in skip..n {
                for row in 0..col + 1 - skip {
                    out.push(column_index(row, col, n));
                }
            }
        }
        Triangle::UpperRow => {
            for row in 0..n {
                for col in row + skip..n {
                    out.push(column_index(row, col, n));
                }
            }
        }
    }
    Ok(out)
}

pub fn vec(a: &Matrix) -> Vec<f64> {
    a.as_slice().to_vec()
}

pub fn vecr(a: &Matrix) -> Vec<f64> {
    a.transpose().as_slice().to_vec()
}

pub fn half_vectorize(a: &Matrix, order: Triangle, diagonal: bool) -> Result<Vec<f64>> {
    let n = square(a)?;
    Ok(triangle_indices(n, order, diagonal)?
        .into_iter()
        .map(|i| a[i])
        .collect())
}

/// Reconstruct a symmetric matrix. Off-diagonal input leaves a zero diagonal.
/// An empty diagonal-free vector corresponds to a 1 by 1 zero matrix, as in R.
pub fn half_reverse(x: &[f64], order: Triangle, diagonal: bool) -> Result<Matrix> {
    let mut lo = if diagonal { 0 } else { 1 };
    let mut hi = x
        .len()
        .checked_add(2)
        .ok_or_else(|| Error::Dimension("triangle length overflow".into()))?;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if half_len(mid, diagonal).is_ok_and(|size| size < x.len()) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if half_len(lo, diagonal)? != x.len() {
        return Err(Error::Dimension("vector length is not triangular".into()));
    }
    checked_area(lo, lo)?;
    let mut out = Matrix::zeros(lo, lo);
    for (k, index) in triangle_indices(lo, order, diagonal)?
        .into_iter()
        .enumerate()
    {
        let row = index % lo;
        let col = index / lo;
        out[(row, col)] = x[k];
        out[(col, row)] = x[k];
    }
    Ok(out)
}

pub fn diagonal_indices(n: usize) -> Result<Vec<usize>> {
    checked_area(n, n)?;
    Ok((0..n).map(|i| i * n + i).collect())
}

pub fn half_diagonal_indices(n: usize) -> Result<Vec<usize>> {
    half_len(n, true)?;
    let mut index = 0;
    Ok((0..n)
        .map(|i| {
            let k = index;
            index += n - i;
            k
        })
        .collect())
}

pub fn antidiagonal_indices(n: usize) -> Result<Vec<usize>> {
    checked_area(n, n)?;
    Ok((0..n).map(|i| (i + 1) * n.saturating_sub(1)).collect())
}

fn dense_duplication_order(n: usize) -> Result<()> {
    if !(1..=255).contains(&n) {
        return Err(Error::InvalidInput(
            "duplication order must be in 1..=255 (lavaan limit)".into(),
        ));
    }
    checked_area(n * n, half_len(n, true)?)?;
    Ok(())
}

pub fn duplication(n: usize) -> Result<Matrix> {
    dense_duplication_order(n)?;
    let mut d = Matrix::zeros(n * n, half_len(n, true)?);
    for (k, i) in triangle_indices(n, Triangle::LowerColumn, true)?
        .into_iter()
        .enumerate()
    {
        let j = (i % n) * n + i / n;
        d[(i, k)] = 1.0;
        d[(j, k)] = 1.0;
    }
    Ok(d)
}

pub fn duplication_inverse(n: usize) -> Result<Matrix> {
    dense_duplication_order(n)?;
    let mut d = Matrix::zeros(half_len(n, true)?, n * n);
    for (k, i) in triangle_indices(n, Triangle::LowerColumn, true)?
        .into_iter()
        .enumerate()
    {
        let j = (i % n) * n + i / n;
        let weight = if i == j { 1.0 } else { 0.5 };
        d[(k, i)] = weight;
        d[(k, j)] = weight;
    }
    Ok(d)
}

fn square_root_order(n2: usize) -> Result<usize> {
    let n = n2.isqrt();
    if n.checked_mul(n) != Some(n2) {
        return Err(Error::Dimension(
            "axis length must be a perfect square".into(),
        ));
    }
    Ok(n)
}

/// `D' A`, or `D+ A` when `inverse` is true; no dense D is allocated.
pub fn duplication_pre(a: &Matrix, inverse: bool) -> Result<Matrix> {
    let n = square_root_order(a.nrows())?;
    let indices = triangle_indices(n, Triangle::LowerColumn, true)?;
    let mut out = Matrix::zeros(indices.len(), a.ncols());
    for (k, &i) in indices.iter().enumerate() {
        let j = (i % n) * n + i / n;
        for col in 0..a.ncols() {
            out[(k, col)] = if i == j {
                a[(i, col)]
            } else if inverse {
                (a[(i, col)] + a[(j, col)]) / 2.0
            } else {
                a[(i, col)] + a[(j, col)]
            };
        }
    }
    Ok(out)
}

/// `A D`, or `A (D+)'` when `inverse` is true.
pub fn duplication_post(a: &Matrix, inverse: bool) -> Result<Matrix> {
    Ok(duplication_pre(&a.transpose(), inverse)?.transpose())
}

pub fn duplication_pre_post(a: &Matrix, inverse: bool) -> Result<Matrix> {
    square(a)?;
    duplication_post(&duplication_pre(a, inverse)?, inverse)
}

/// Permutation gathering positions for `vec(A')` where A has m rows, n columns.
pub fn commutation_indices(m: usize, n: usize) -> Result<Vec<usize>> {
    let size = checked_area(m, n)?;
    let mut out = Vec::with_capacity(size);
    for index in 0..size {
        out.push(transpose_index(index, m, n));
    }
    Ok(out)
}

pub fn commutation(m: usize, n: usize) -> Result<Matrix> {
    if m == 0 || n == 0 {
        return Err(Error::InvalidInput(
            "commutation dimensions must be positive".into(),
        ));
    }
    let size = checked_area(m, n)?;
    checked_area(size, size)?;
    let mut out = Matrix::zeros(size, size);
    for (row, col) in commutation_indices(m, n)?.into_iter().enumerate() {
        out[(row, col)] = 1.0;
    }
    Ok(out)
}

pub fn commutation_pre(a: &Matrix, m: usize, n: usize) -> Result<Matrix> {
    let indices = commutation_indices(m, n)?;
    if a.nrows() != indices.len() {
        return Err(Error::Dimension("commutation rows differ from m*n".into()));
    }
    Ok(Matrix::from_fn(a.nrows(), a.ncols(), |i, j| {
        a[(indices[i], j)]
    }))
}

pub fn commutation_post(a: &Matrix) -> Result<Matrix> {
    let n = square_root_order(a.ncols())?;
    Ok(commutation_pre(&a.transpose(), n, n)?.transpose())
}

pub fn commutation_pre_post(a: &Matrix) -> Result<Matrix> {
    let n = square_root_order(square(a)?)?;
    commutation_post(&commutation_pre(a, n, n)?)
}

pub fn block_diagonal(blocks: &[Matrix]) -> Result<Matrix> {
    let rows = blocks
        .iter()
        .try_fold(0_usize, |n, a| n.checked_add(a.nrows()))
        .ok_or_else(|| Error::Dimension("block rows overflow".into()))?;
    let cols = blocks
        .iter()
        .try_fold(0_usize, |n, a| n.checked_add(a.ncols()))
        .ok_or_else(|| Error::Dimension("block columns overflow".into()))?;
    checked_area(rows, cols)?;
    let mut out = Matrix::zeros(rows, cols);
    let (mut r, mut c) = (0, 0);
    for a in blocks {
        out.view_mut((r, c), a.shape()).copy_from(a);
        r += a.nrows();
        c += a.ncols();
    }
    Ok(out)
}

pub fn trace_product(matrices: &[Matrix]) -> Result<f64> {
    let Some(first) = matrices.first() else {
        return Ok(f64::NAN);
    };
    if matrices.len() == 1 {
        square(first)?;
        return Ok(first.trace());
    }
    for pair in matrices.windows(2) {
        if pair[0].ncols() != pair[1].nrows() {
            return Err(Error::Dimension("nonconformable matrix product".into()));
        }
    }
    if matrices.last().unwrap().ncols() != first.nrows() {
        return Err(Error::Dimension("trace product must be square".into()));
    }
    let mut tail = matrices[1].clone();
    for a in &matrices[2..] {
        tail *= a;
    }
    Ok(first
        .iter()
        .zip(tail.transpose().iter())
        .map(|(x, y)| x * y)
        .sum())
}

pub fn kronecker(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    let rows = a
        .nrows()
        .checked_mul(b.nrows())
        .ok_or_else(|| Error::Dimension("Kronecker rows overflow".into()))?;
    let cols = a
        .ncols()
        .checked_mul(b.ncols())
        .ok_or_else(|| Error::Dimension("Kronecker columns overflow".into()))?;
    checked_area(rows, cols)?;
    Ok(Matrix::from_fn(rows, cols, |i, j| {
        a[(i / b.nrows(), j / b.ncols())] * b[(i % b.nrows(), j % b.ncols())]
    }))
}

/// Cholesky factorization for strictly positive definite matrices only.
/// Reuse this object for solves; no inverse is needed for likelihoods.
pub struct Spd {
    factor: nalgebra::linalg::Cholesky<f64, nalgebra::Dyn>,
    logdet: f64,
}

impl Spd {
    pub fn new(a: &Matrix) -> Result<Self> {
        let n = symmetric(a)?;
        if n == 0 {
            return Err(Error::InvalidInput("empty covariance matrix".into()));
        }
        let factor = a.clone().cholesky().ok_or(Error::NotPositiveDefinite)?;
        let logdet = 2.0 * factor.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
        if !logdet.is_finite() {
            return Err(Error::NotPositiveDefinite);
        }
        Ok(Self { factor, logdet })
    }

    pub fn logdet(&self) -> f64 {
        self.logdet
    }

    pub fn inverse(&self) -> Matrix {
        self.factor.inverse()
    }

    pub fn solve(&self, rhs: &Matrix) -> Result<Matrix> {
        if rhs.nrows() != self.factor.l().nrows() {
            return Err(Error::Dimension("right hand side row count differs".into()));
        }
        finite(rhs, "right hand side")?;
        Ok(self.factor.solve(rhs))
    }
}

/// Spectral square root; lavaan truncates *all* negative eigenvalues to zero.
pub fn symmetric_sqrt(a: &Matrix) -> Result<Matrix> {
    symmetric(a)?;
    if a.is_empty() {
        return Ok(a.clone());
    }
    let eig = a.clone().symmetric_eigen();
    let roots = eig.eigenvalues.map(|x| x.max(0.0).sqrt());
    Ok(&eig.eigenvectors * Matrix::from_diagonal(&roots) * eig.eigenvectors.transpose())
}

/// Moore-Penrose inverse, with the same relative cutoff as MASS::ginv.
pub fn generalized_inverse(a: &Matrix) -> Result<Matrix> {
    finite(a, "matrix")?;
    if a.is_empty() {
        return Ok(Matrix::zeros(a.ncols(), a.nrows()));
    }
    let svd = a.clone().svd(true, true);
    let cutoff = f64::EPSILON.sqrt() * svd.singular_values.max();
    svd.pseudo_inverse(cutoff).map_err(|_| Error::Singular)
}

/// Nilpotent shortcut through A^6, followed by a solve or generalized inverse.
pub fn inverse_iminus(a: &Matrix) -> Result<Matrix> {
    let n = square(a)?;
    finite(a, "regression matrix")?;
    let mut out = Matrix::identity(n, n) + a;
    let mut power = a.clone();
    for exponent in 2..=6 {
        power = &power * a;
        if power.iter().all(|&x| x == 0.0) {
            return Ok(out);
        }
        if exponent < 6 {
            out += &power;
        }
    }
    let system = Matrix::identity(n, n) - a;
    if let Some(inverse) = system.clone().try_inverse() {
        Ok(inverse)
    } else {
        generalized_inverse(&system)
    }
}

/// Scale a covariance matrix to a correlation matrix (or selected variables).
pub fn covariance_to_correlation(a: &Matrix, indices: Option<&[usize]>) -> Result<Matrix> {
    let n = symmetric(a)?;
    let mut scale = vec![1.0; n];
    let selected: Vec<usize> = match indices {
        Some(x) if !x.is_empty() => x.to_vec(),
        _ => (0..n).collect(),
    };
    for i in selected {
        if i >= n {
            return Err(Error::Dimension("correlation index out of range".into()));
        }
        if a[(i, i)] <= 0.0 {
            return Err(Error::InvalidInput(
                "correlation requires positive variances".into(),
            ));
        }
        scale[i] = a[(i, i)].sqrt();
    }
    Ok(Matrix::from_fn(n, n, |i, j| {
        a[(i, j)] / (scale[i] * scale[j])
    }))
}

pub fn correlation_to_covariance(a: &Matrix, standard_deviations: &[f64]) -> Result<Matrix> {
    let n = symmetric(a)?;
    if standard_deviations.len() != n {
        return Err(Error::Dimension("standard deviation count differs".into()));
    }
    if standard_deviations
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(Error::InvalidInput(
            "standard deviations must be finite and nonnegative".into(),
        ));
    }
    Ok(Matrix::from_fn(n, n, |i, j| {
        a[(i, j)] * standard_deviations[i] * standard_deviations[j]
    }))
}
