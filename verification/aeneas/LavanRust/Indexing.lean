import LavanKernels

open Aeneas Aeneas.Std Result

namespace lavan_kernels

/-- This proves the extracted production Rust function terminates successfully
    without arithmetic overflow and returns the mathematical column index. -/
@[step] theorem column_index_spec (row col rows : Usize)
    (capacity : row.val + col.val * rows.val ≤ Usize.max) :
    column_index row col rows ⦃ result =>
      result.val = row.val + col.val * rows.val ⦄ := by
  unfold column_index
  step
  step
  omega

theorem column_index_bounded (row col rows cols : Usize)
    (hr : row.val < rows.val) (hc : col.val < cols.val)
    (capacity : rows.val * cols.val ≤ Usize.max) :
    column_index row col rows ⦃ result =>
      result.val = row.val + col.val * rows.val ∧ result.val < rows.val * cols.val ⦄ := by
  have hb : row.val + col.val * rows.val < rows.val * cols.val := by
    calc
      row.val + col.val * rows.val < rows.val + col.val * rows.val := Nat.add_lt_add_right hr _
      _ = (col.val + 1) * rows.val := by ring
      _ ≤ cols.val * rows.val := Nat.mul_le_mul_right rows.val hc
      _ = rows.val * cols.val := Nat.mul_comm _ _
  step
  omega

@[step] theorem transpose_index_spec (index rows cols : Usize)
    (hcols : 0 < cols.val) (hindex : index.val < rows.val * cols.val)
    (capacity : rows.val * cols.val ≤ Usize.max) :
    transpose_index index rows cols ⦃ result =>
      result.val = index.val / cols.val + (index.val % cols.val) * rows.val ∧
      result.val < rows.val * cols.val ⦄ := by
  unfold transpose_index
  step as ⟨r, hr⟩
  step as ⟨c, hc⟩
  have hrange : r.val < rows.val := by
    rw [hr]
    exact (Nat.div_lt_iff_lt_mul hcols).mpr (by simpa [Nat.mul_comm] using hindex)
  have crange : c.val < cols.val := by
    rw [hc]
    exact Nat.mod_lt _ hcols
  have hb : r.val + c.val * rows.val < rows.val * cols.val := by
    calc
      r.val + c.val * rows.val < rows.val + c.val * rows.val := Nat.add_lt_add_right hrange _
      _ = (c.val + 1) * rows.val := by ring
      _ ≤ cols.val * rows.val := Nat.mul_le_mul_right rows.val crange
      _ = rows.val * cols.val := Nat.mul_comm _ _
  step
  constructor
  · simp_all
  · omega

end lavan_kernels
