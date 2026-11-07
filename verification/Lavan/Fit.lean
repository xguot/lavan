import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp

namespace Lavan

noncomputable section

/-- Exact real core of lav_fit_cfi; R's tolerance branch is separate. -/
def cfi (target baseline : ℝ) : ℝ :=
  let a := max target 0
  let b := max (max target baseline) 0
  if b = 0 then 1 else 1 - a / b

theorem cfi_bounds (target baseline : ℝ) : 0 ≤ cfi target baseline ∧ cfi target baseline ≤ 1 := by
  unfold cfi
  dsimp
  split_ifs with h
  · constructor <;> norm_num
  · have hb0 : 0 ≤ max (max target baseline) 0 := le_max_right _ _
    have hb : 0 < max (max target baseline) 0 := lt_of_le_of_ne hb0 (Ne.symm h)
    have ha : 0 ≤ max target 0 := le_max_right _ _
    have hab : max target 0 ≤ max (max target baseline) 0 :=
      max_le (le_trans (le_max_left _ _) (le_max_left _ _)) (le_max_right _ _)
    have hratio0 : 0 ≤ max target 0 / max (max target baseline) 0 := div_nonneg ha hb0
    have hratio1 : max target 0 / max (max target baseline) 0 ≤ 1 := (div_le_one hb).mpr hab
    constructor <;> linarith

def rmsea (x df n correction : ℝ) : ℝ :=
  Real.sqrt (max ((x / n) / df - correction / n) 0)

theorem rmsea_nonnegative (x df n correction : ℝ) : 0 ≤ rmsea x df n correction :=
  Real.sqrt_nonneg _

theorem rmsea_zero_at_small_discrepancy (x df n correction : ℝ)
    (h : (x / n) / df ≤ correction / n) : rmsea x df n correction = 0 := by
  unfold rmsea
  rw [max_eq_right (sub_nonpos.mpr h), Real.sqrt_zero]

theorem gaussian_variance_score_zero_at_moment_match (s : ℝ) (hs : s ≠ 0) :
    -(1 / 2 : ℝ) * (1 / s - (1 / s) * s * (1 / s)) = 0 := by
  field_simp
  ring

end
end Lavan
