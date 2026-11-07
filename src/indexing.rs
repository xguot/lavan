//! Small integer kernels shared by the runtime and the Aeneas extraction crate.
//!
//! Callers must validate dimensions first. `column_index` requires row < rows,
//! col < cols, and rows*cols <= usize::MAX. `transpose_index` additionally
//! requires cols > 0 and index < rows*cols. These preconditions are proved
//! sufficient for the extracted functions in `verification/aeneas`.

#[inline]
pub fn column_index(row: usize, col: usize, rows: usize) -> usize {
    row + col * rows
}

#[inline]
pub fn transpose_index(index: usize, rows: usize, cols: usize) -> usize {
    let row = index / cols;
    let col = index % cols;
    column_index(row, col, rows)
}
