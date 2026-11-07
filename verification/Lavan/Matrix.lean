import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Lean.Elab.Tactic.Omega
import Mathlib.Algebra.Ring.GeomSum

/-!
Exact-arithmetic specifications for the port's matrix kernels. These theorems
do not assert IEEE-754 error bounds or establish refinement of the Rust source.
-/
namespace Lavan

noncomputable section

def columnIndex (rows row col : Nat) : Nat := row + col * rows

theorem columnIndex_injective {rows r₁ r₂ c₁ c₂ : Nat}
    (hrows : 0 < rows) (h₁ : r₁ < rows) (h₂ : r₂ < rows)
    (h : columnIndex rows r₁ c₁ = columnIndex rows r₂ c₂) :
    r₁ = r₂ ∧ c₁ = c₂ := by
  unfold columnIndex at h
  have hr := congrArg (fun k => k % rows) h
  simp [Nat.add_mod, Nat.mod_eq_of_lt h₁, Nat.mod_eq_of_lt h₂] at hr
  subst r₂
  constructor
  · rfl
  · exact Nat.mul_right_cancel hrows (Nat.add_left_cancel h)

theorem columnIndex_bound {rows cols row col : Nat}
    (hr : row < rows) (hc : col < cols) :
    columnIndex rows row col < rows * cols := by
  unfold columnIndex
  calc
    row + col * rows < rows + col * rows := Nat.add_lt_add_right hr _
    _ = (col + 1) * rows := by ring
    _ ≤ cols * rows := Nat.mul_le_mul_right rows hc
    _ = rows * cols := Nat.mul_comm _ _

def lowerRead {n : Nat} (a : Matrix (Fin n) (Fin n) ℝ) (i j : Fin n) : ℝ :=
  if j ≤ i then a i j else a j i

theorem lowerRead_symmetric {n : Nat} (a : Matrix (Fin n) (Fin n) ℝ) :
    ∀ i j, lowerRead a i j = lowerRead a j i := by
  intro i j
  simp only [lowerRead]
  split_ifs with h₁ h₂ h₂
  · have : i = j := le_antisymm h₂ h₁
    subst j
    rfl
  · rfl
  · rfl
  · exact False.elim (h₂ (le_of_not_ge h₁))

theorem lowerRead_roundtrip {n : Nat} (a : Matrix (Fin n) (Fin n) ℝ)
    (hs : ∀ i j, a i j = a j i) : lowerRead a = a := by
  funext i j
  unfold lowerRead
  split_ifs
  · rfl
  · exact hs j i

def symmetrize {n : Nat} (a : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => (a i j + a j i) / 2

theorem duplication_projection_idempotent {n : Nat} (a : Matrix (Fin n) (Fin n) ℝ) :
    symmetrize (symmetrize a) = symmetrize a := by
  funext i j
  simp only [symmetrize]
  ring

theorem duplication_inverse_left {n : Nat} (a : Matrix (Fin n) (Fin n) ℝ)
    (hs : ∀ i j, a i j = a j i) : symmetrize a = a := by
  funext i j
  simp only [symmetrize, ← hs i j]
  ring

theorem commutation_involution {m n : Nat} (a : Matrix (Fin m) (Fin n) ℝ) :
    a.transpose.transpose = a := Matrix.transpose_transpose a

theorem covariance_shift_identity (s mean target x : ℝ) :
    s + (mean - target)^2 = s + (mean - x)^2 + 2*(mean-x)*(x-target) + (x-target)^2 := by
  ring

/-- Exact nilpotent branch used when A² = 0. -/
theorem inverse_iminus_square_zero {n : Nat} (a : Matrix (Fin n) (Fin n) ℝ)
    (h : a * a = 0) : (1 - a) * (1 + a) = 1 ∧ (1 + a) * (1 - a) = 1 := by
  constructor <;> simp [Matrix.mul_add, Matrix.mul_sub, Matrix.add_mul, Matrix.sub_mul, h]

/-- Every finite nilpotent branch through A^6 is an exact two-sided inverse. -/
theorem inverse_iminus_nilpotent {n k : Nat} (a : Matrix (Fin n) (Fin n) ℝ)
    (h : a ^ k = 0) :
    (∑ i ∈ Finset.range k, a ^ i) * (1 - a) = 1 ∧
    (1 - a) * (∑ i ∈ Finset.range k, a ^ i) = 1 := by
  constructor
  · simpa [h] using geom_sum_mul_neg a k
  · simpa [h] using mul_neg_geom_sum a k

end
end Lavan
