import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
Finite-scale proof of the historical symmetric estimate in the manuscript.
The analytical input is exactly the linewise remainder inequality, in both
orientations. No Hessian, auxiliary limit, or convergence of a sequence is assumed.
The implication from Holder derivatives to the remainder inequality is outside
this module and is explicitly listed for manual checking.
-/
namespace UAG.History

noncomputable section

def pairError (R : ℝ → ℝ) (a t : ℝ) : ℝ :=
  (1 + a / t) / 2 * R t + (1 - a / t) / 2 * R (-t)
    - t ^ 2 / a ^ 2 * R a

def reverseError (R : ℝ → ℝ) (a t : ℝ) : ℝ :=
  (1 + a / t) / 2 * R (-t) + (1 - a / t) / 2 * R t
    - t ^ 2 / a ^ 2 * R (-a)

def symmetric (R : ℝ → ℝ) (t : ℝ) : ℝ :=
  (R t + R (-t)) / t ^ 2

def curvature (R : ℝ → ℝ) (a b : ℝ) : ℝ :=
  2 * (a ^ 3 * R b + b ^ 3 * R (-a)) / (a ^ 2 * b ^ 2 * (a + b))

theorem errors_add (R : ℝ → ℝ) {a t : ℝ} (ha : a ≠ 0) (ht : t ≠ 0) :
    pairError R a t + reverseError R a t =
      t ^ 2 * (symmetric R t - symmetric R a) := by
  unfold pairError reverseError symmetric
  field_simp
  ring

theorem curvature_error_identity (R : ℝ → ℝ) {a b : ℝ}
    (ha : 0 < a) (hb : 0 < b) :
    (symmetric R a - curvature R a b) * ((1 + b / a) * a ^ 2) =
      2 * pairError R b a := by
  have ha0 : a ≠ 0 := ne_of_gt ha
  have hb0 : b ≠ 0 := ne_of_gt hb
  have hab0 : a + b ≠ 0 := ne_of_gt (add_pos ha hb)
  unfold symmetric curvature pairError
  field_simp
  ring

theorem rpow_factor {t r : ℝ} (ht : 0 < t) :
    t ^ r = t ^ (2 : ℕ) * t ^ (r - 2) := by
  calc
    t ^ r = t ^ ((2 : ℝ) + (r - 2)) := by congr 1; ring
    _ = t ^ (2 : ℝ) * t ^ (r - 2) := Real.rpow_add ht _ _
    _ = t ^ (2 : ℕ) * t ^ (r - 2) := by norm_num

/-- The manuscript's constant 4, derived using only three finite-scale bounds. -/
theorem symmetric_descent_ordered (R : ℝ → ℝ) {K r a b t : ℝ}
    (hK : 0 ≤ K) (hr : 2 < r) (hb : 0 < b) (hba : b ≤ a) (hat : a ≤ t)
    (hforward : |pairError R a t| ≤ K * (t ^ r + t ^ 2 * a ^ (r - 2)))
    (hreverse : |reverseError R a t| ≤ K * (t ^ r + t ^ 2 * a ^ (r - 2)))
    (hsmall : |pairError R b a| ≤ K * (a ^ r + a ^ 2 * b ^ (r - 2))) :
    (R t + R (-t)) / 2 ≤ curvature R a b * t ^ 2 / 2 + 4 * K * t ^ r := by
  have ha : 0 < a := lt_of_lt_of_le hb hba
  have ht : 0 < t := lt_of_lt_of_le ha hat
  have har : 0 ≤ a ^ (r - 2) := Real.rpow_nonneg ha.le _
  have hbr : b ^ (r - 2) ≤ a ^ (r - 2) :=
    Real.rpow_le_rpow hb.le hba (by linarith)
  have hatr : a ^ (r - 2) ≤ t ^ (r - 2) :=
    Real.rpow_le_rpow ha.le hat (by linarith)
  have ht2 : 0 < t ^ (2 : ℕ) := sq_pos_of_pos ht
  have ha2 : 0 < a ^ (2 : ℕ) := sq_pos_of_pos ha
  have hp := (abs_le.mp hforward).2
  have hm := (abs_le.mp hreverse).2
  rw [rpow_factor ht] at hp hm
  have he := errors_add R (ne_of_gt ha) (ne_of_gt ht)
  have hsym : symmetric R t ≤ symmetric R a + 2 * K * (t ^ (r - 2) + a ^ (r - 2)) := by
    apply le_of_mul_le_mul_right _ ht2
    nlinarith only [hp, hm, he]
  let den : ℝ := (1 + b / a) * a ^ 2
  have hden : a ^ 2 ≤ den := by
    have hn : 0 ≤ b / a * a ^ (2 : ℕ) := mul_nonneg (div_nonneg hb.le ha.le) ha2.le
    dsimp [den]
    nlinarith only [hn]
  have hdenpos : 0 < den := lt_of_lt_of_le ha2 hden
  have hd := (abs_le.mp hsmall).2
  rw [rpow_factor ha] at hd
  have hdBound : pairError R b a ≤ 2 * K * a ^ 2 * a ^ (r - 2) := by
    have hbmul := mul_le_mul_of_nonneg_left hbr (mul_nonneg hK ha2.le)
    nlinarith only [hd, hbmul]
  have hid := curvature_error_identity R ha hb
  change (symmetric R a - curvature R a b) * den = 2 * pairError R b a at hid
  have hscale : 4 * K * a ^ (r - 2) * a ^ 2 ≤ 4 * K * a ^ (r - 2) * den :=
    mul_le_mul_of_nonneg_left hden (by positivity)
  have hq : symmetric R a ≤ curvature R a b + 4 * K * a ^ (r - 2) := by
    apply le_of_mul_le_mul_right _ hdenpos
    nlinarith only [hdBound, hid, hscale]
  have hpow := mul_le_mul_of_nonneg_left hatr (show 0 ≤ 6 * K by positivity)
  have htBound : symmetric R t ≤ curvature R a b + 8 * K * t ^ (r - 2) := by
    nlinarith only [hsym, hq, hpow]
  have hmul := mul_le_mul_of_nonneg_right htBound ht2.le
  have hs : symmetric R t * t ^ (2 : ℕ) = R t + R (-t) := by
    unfold symmetric
    exact div_mul_cancel₀ _ (ne_of_gt ht2)
  rw [hs] at hmul
  rw [rpow_factor ht]
  nlinarith only [hmul]

def RemainderBound (R : ℝ → ℝ) (K r : ℝ) : Prop :=
  ∀ a t : ℝ, 0 < a → a ≤ t →
    |pairError R a t| ≤ K * (t ^ r + t ^ 2 * a ^ (r - 2)) ∧
    |reverseError R a t| ≤ K * (t ^ r + t ^ 2 * a ^ (r - 2))

theorem reverse_bound {R : ℝ → ℝ} {K r : ℝ} (h : RemainderBound R K r) :
    RemainderBound (fun s => R (-s)) K r := by
  intro a t ha hat
  have hh := h a t ha hat
  simpa only [pairError, reverseError, neg_neg] using And.intro hh.2 hh.1

theorem curvature_reverse (R : ℝ → ℝ) (a b : ℝ) :
    curvature (fun s => R (-s)) b a = curvature R a b := by
  unfold curvature
  simp only [neg_neg]
  congr 1 <;> ring

/-- The full historical symmetric estimate, with either ordering of a and b. -/
theorem symmetric_descent (R : ℝ → ℝ) {K r a b t : ℝ}
    (hK : 0 ≤ K) (hr : 2 < r) (ha : 0 < a) (hb : 0 < b)
    (hat : a ≤ t) (hbt : b ≤ t) (h : RemainderBound R K r) :
    (R t + R (-t)) / 2 ≤ curvature R a b * t ^ 2 / 2 + 4 * K * t ^ r := by
  rcases le_total b a with hba | hab
  · exact symmetric_descent_ordered R hK hr hb hba hat
      (h a t ha hat).1 (h a t ha hat).2 (h b a hb hba).1
  · have hh := reverse_bound h
    have H := symmetric_descent_ordered (fun s => R (-s)) hK hr ha hab hbt
      (hh b t hb hbt).1 (hh b t hb hbt).2 (hh a b ha hab).1
    rw [curvature_reverse] at H
    simpa only [neg_neg, add_comm] using H

/-- Cancellation of the constant and linear terms converts the R-bound to f. -/
theorem function_symmetric_descent (F : ℝ → ℝ) (g : ℝ) {K r a b t : ℝ}
    (hK : 0 ≤ K) (hr : 2 < r) (ha : 0 < a) (hb : 0 < b)
    (hat : a ≤ t) (hbt : b ≤ t)
    (h : RemainderBound (fun s => F s - F 0 - s * g) K r) :
    (F t + F (-t)) / 2 ≤ F 0 +
      curvature (fun s => F s - F 0 - s * g) a b * t ^ 2 / 2 + 4 * K * t ^ r := by
  have hh := symmetric_descent (fun s => F s - F 0 - s * g) hK hr ha hb hat hbt h
  nlinarith only [hh]

/-- The exact expression in eq:outer-remainder, restricted to one line. -/
def manuscriptError (F : ℝ → ℝ) (g a t : ℝ) : ℝ :=
  (1 + a / t) / 2 * F t + (1 - a / t) / 2 * F (-t) - F a -
    (t ^ 2 / a ^ 2 - 1) * (F a - F 0 - a * g)

theorem normalized_error (F : ℝ → ℝ) (g : ℝ) {a t : ℝ}
    (ha : a ≠ 0) (ht : t ≠ 0) :
    pairError (fun s => F s - F 0 - s * g) a t = manuscriptError F g a t := by
  unfold pairError manuscriptError
  field_simp
  ring

theorem normalized_reverse_error (F : ℝ → ℝ) (g : ℝ) {a t : ℝ}
    (ha : a ≠ 0) (ht : t ≠ 0) :
    reverseError (fun s => F s - F 0 - s * g) a t =
      manuscriptError (fun s => F (-s)) (-g) a t := by
  unfold reverseError manuscriptError
  simp only [neg_neg, neg_zero]
  field_simp
  ring

/-- Cubic terms cancel exactly; this also covers degree two with c3 = 0. -/
theorem cubic_cancellation (c0 c1 c2 c3 : ℝ) {a t : ℝ}
    (ha : a ≠ 0) (ht : t ≠ 0) :
    manuscriptError (fun s => c0 + c1 * s + c2 * s ^ 2 + c3 * s ^ 3) c1 a t = 0 := by
  unfold manuscriptError
  field_simp
  ring

/-- Direct correspondence with the two orientations of the paper's condition. -/
theorem symmetric_descent_from_manuscript (F : ℝ → ℝ) (g : ℝ) {K r a b t : ℝ}
    (hK : 0 ≤ K) (hr : 2 < r) (ha : 0 < a) (hb : 0 < b)
    (hat : a ≤ t) (hbt : b ≤ t)
    (h : ∀ u v : ℝ, 0 < u → u ≤ v →
      |manuscriptError F g u v| ≤ K * (v ^ r + v ^ 2 * u ^ (r - 2)) ∧
      |manuscriptError (fun s => F (-s)) (-g) u v| ≤
        K * (v ^ r + v ^ 2 * u ^ (r - 2))) :
    (F t + F (-t)) / 2 ≤ F 0 +
      curvature (fun s => F s - F 0 - s * g) a b * t ^ 2 / 2 + 4 * K * t ^ r := by
  apply function_symmetric_descent F g hK hr ha hb hat hbt
  intro u v hu huv
  have hv : 0 < v := lt_of_lt_of_le hu huv
  rw [normalized_error F g (ne_of_gt hu) (ne_of_gt hv),
      normalized_reverse_error F g (ne_of_gt hu) (ne_of_gt hv)]
  exact h u v hu huv

end
end UAG.History
