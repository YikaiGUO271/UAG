import Mathlib.Analysis.Real.Sqrt
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# UAG: the quadratic-model algebra

This file uses real coordinate vectors on an arbitrary finite index type.
`dot` and `sqDist` are the Euclidean dot product and squared distance written
as finite sums. Function values and the queried gradient are parameters:
the identities below do not need convexity or differentiability.

Manuscript mapping: `eq:nag-parameters`, `eq:inner-completion-squares`,
`eq:inner-cross-cancellation`, `eq:inner-gap-identity`, and
`eq:inner-momentum-displacement` in the frozen merged manuscript.
-/

namespace UAG.Quadratic

open scoped BigOperators

noncomputable section

def curvature (lam A : ℝ) : ℝ := 1 + lam * A / 2

def discriminant (lam ell A : ℝ) : ℝ :=
  1 + 2 * (lam + ell) * A * (2 + lam * A)

def weight (lam ell A : ℝ) : ℝ :=
  (1 + lam * A + Real.sqrt (discriminant lam ell A)) / (lam + 2 * ell)

def theta (lam A a : ℝ) : ℝ :=
  a * (2 + lam * A) /
    (A * (2 + lam * (A + a)) + a * (2 + lam * A))

theorem curvature_pos {lam A : ℝ} (hl : 0 < lam) (hA : 0 ≤ A) :
    0 < curvature lam A := by
  unfold curvature
  positivity

theorem discriminant_pos {lam ell A : ℝ}
    (hl : 0 < lam) (he : 0 < ell) (hA : 0 ≤ A) :
    0 < discriminant lam ell A := by
  unfold discriminant
  positivity

theorem weight_pos {lam ell A : ℝ}
    (hl : 0 < lam) (he : 0 < ell) (hA : 0 ≤ A) :
    0 < weight lam ell A := by
  unfold weight
  positivity

theorem root_equation_algebra (c d p s : ℝ) (hd : d ≠ 0)
    (hs : s ^ 2 = c ^ 2 + d * p) :
    d * ((c + s) / d) ^ 2 - 2 * c * ((c + s) / d) - p = 0 := by
  field_simp [hd]
  nlinarith [hs]

theorem weight_quadratic {lam ell A : ℝ}
    (hl : 0 < lam) (he : 0 < ell) (hA : 0 ≤ A) :
    (lam + 2 * ell) * (weight lam ell A) ^ 2 -
      2 * (1 + lam * A) * weight lam ell A - A * (2 + lam * A) = 0 := by
  have hd : lam + 2 * ell ≠ 0 := ne_of_gt (by positivity)
  have hs : Real.sqrt (discriminant lam ell A) ^ 2 =
      (1 + lam * A) ^ 2 + (lam + 2 * ell) * (A * (2 + lam * A)) := by
    rw [Real.sq_sqrt (le_of_lt (discriminant_pos hl he hA))]
    unfold discriminant
    ring
  exact root_equation_algebra (1 + lam * A) (lam + 2 * ell)
    (A * (2 + lam * A)) (Real.sqrt (discriminant lam ell A)) hd hs

theorem weight_relation {lam ell A : ℝ}
    (hl : 0 < lam) (he : 0 < ell) (hA : 0 ≤ A) :
    (weight lam ell A) ^ 2 =
      (lam + ell)⁻¹ * (A + weight lam ell A) *
        curvature lam (A + weight lam ell A) := by
  have hp := weight_quadratic hl he hA
  have hden : lam + ell ≠ 0 := ne_of_gt (by positivity)
  unfold curvature
  field_simp [hden]
  nlinarith [hp]

theorem positive_root_unique {lam ell A b : ℝ}
    (hl : 0 < lam) (he : 0 < ell) (hA : 0 ≤ A) (hb : 0 < b)
    (hroot : (lam + 2 * ell) * b ^ 2 -
      2 * (1 + lam * A) * b - A * (2 + lam * A) = 0) :
    b = weight lam ell A := by
  let a := weight lam ell A
  have ha : 0 < a := weight_pos hl he hA
  have hqa : (lam + 2 * ell) * a ^ 2 -
      2 * (1 + lam * A) * a - A * (2 + lam * A) = 0 :=
    weight_quadratic hl he hA
  have hfac : (b - a) * ((lam + 2 * ell) * (b + a) -
      2 * (1 + lam * A)) = 0 := by nlinarith [hroot, hqa]
  rcases mul_eq_zero.mp hfac with hab | hs
  · dsimp [a] at hab
    linarith
  · have hs' := congrArg (fun t : ℝ => t * a) hs
    have hprod : 0 < (lam + 2 * ell) * (a * b) := by positivity
    have hk : 0 ≤ A * (2 + lam * A) := by positivity
    nlinarith [hqa, hs']

theorem positive_weight_relation_unique {lam ell A b : ℝ}
    (hl : 0 < lam) (he : 0 < ell) (hA : 0 ≤ A) (hb : 0 < b)
    (hroot : b ^ 2 = (lam + ell)⁻¹ * (A + b) * curvature lam (A + b)) :
    b = weight lam ell A := by
  apply positive_root_unique hl he hA hb
  have hd : lam + ell ≠ 0 := ne_of_gt (by positivity)
  unfold curvature at hroot
  field_simp [hd] at hroot
  nlinarith [hroot]

theorem theta_den_pos {lam A a : ℝ}
    (hl : 0 < lam) (hA : 0 ≤ A) (ha : 0 < a) :
    0 < A * (2 + lam * (A + a)) + a * (2 + lam * A) := by
  positivity

theorem theta_pos {lam A a : ℝ}
    (hl : 0 < lam) (hA : 0 ≤ A) (ha : 0 < a) :
    0 < theta lam A a := by
  unfold theta
  positivity

theorem theta_le_one {lam A a : ℝ}
    (hl : 0 < lam) (hA : 0 ≤ A) (ha : 0 < a) :
    theta lam A a ≤ 1 := by
  unfold theta
  apply (div_le_one (theta_den_pos hl hA ha)).2
  have h : 0 ≤ A * (2 + lam * (A + a)) := by positivity
  linarith

theorem theta_zero_weight {lam a : ℝ} (ha : a ≠ 0) :
    theta lam 0 a = 1 := by
  unfold theta
  field_simp
  ring

def extrapolate (lam A a x z : ℝ) : ℝ := x + theta lam A a * (z - x)

/-- The cancellation follows from the actual displayed `theta` formula. -/
theorem cross_cancellation {lam A a : ℝ}
    (hl : 0 < lam) (hA : 0 ≤ A) (ha : 0 < a) (x z : ℝ) :
    A * curvature lam (A + a) * (x - extrapolate lam A a x z) +
      a * curvature lam A * (z - extrapolate lam A a x z) = 0 := by
  have hd := ne_of_gt (theta_den_pos hl hA ha)
  unfold extrapolate theta curvature
  field_simp
  ring

section Coordinates

variable {ι : Type*} [Fintype ι]

def dot (x y : ι → ℝ) : ℝ := ∑ i, x i * y i
def normSq (x : ι → ℝ) : ℝ := ∑ i, (x i) ^ 2
def sqDist (x y : ι → ℝ) : ℝ := ∑ i, (x i - y i) ^ 2

theorem normSq_nonneg (x : ι → ℝ) : 0 ≤ normSq x :=
  Finset.sum_nonneg (fun i _ => sq_nonneg (x i))

theorem sqDist_nonneg (x y : ι → ℝ) : 0 ≤ sqDist x y :=
  Finset.sum_nonneg (fun i _ => sq_nonneg (x i - y i))

@[simp] theorem sqDist_self (x : ι → ℝ) : sqDist x x = 0 := by
  simp [sqDist]

theorem sqDist_comm (x y : ι → ℝ) : sqDist x y = sqDist y x := by
  unfold sqDist
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- The initial affine correction in `eq:inner-entry-model`. -/
def initialCorrection (lam A : ℝ) (p z g : ι → ℝ) : ι → ℝ :=
  fun i => -curvature lam A * (z i - p i) - A * g i

def initialPhi (lam A Jp : ℝ) (p z g u : ι → ℝ) : ℝ :=
  A * (Jp + dot g (fun i => u i - p i) + lam / 4 * sqDist u p) +
    (1 / 2 : ℝ) * sqDist u p + dot (initialCorrection lam A p z g) (fun i => u i - p i)

theorem initial_representation (lam A Jp : ℝ) (p z g u : ι → ℝ) :
    initialPhi lam A Jp p z g u =
      A * Jp - curvature lam A / 2 * sqDist p z + curvature lam A / 2 * sqDist u z := by
  have hcoord : ∀ i,
      A * (g i * (u i - p i) + lam / 4 * (u i - p i) ^ 2) +
        (1 / 2 : ℝ) * (u i - p i) ^ 2 +
        (-curvature lam A * (z i - p i) - A * g i) * (u i - p i) =
      -curvature lam A / 2 * (p i - z i) ^ 2 +
        curvature lam A / 2 * (u i - z i) ^ 2 := by
    intro i
    unfold curvature
    ring
  have hsum := congrArg (fun F : ι → ℝ => ∑ i, F i) (funext hcoord)
  unfold initialPhi dot sqDist initialCorrection
  simp only [mul_add, mul_sub, Finset.sum_add_distrib,
    Finset.sum_sub_distrib, ← Finset.mul_sum] at hsum ⊢
  linarith [hsum]

theorem initial_anchor_value (lam A Jp : ℝ) (p z g : ι → ℝ) :
    initialPhi lam A Jp p z g p = A * Jp := by
  rw [initial_representation]
  ring

theorem initial_minimum {lam A : ℝ} (hl : 0 < lam) (hA : 0 ≤ A)
    (Jp : ℝ) (p z g : ι → ℝ) :
    initialPhi lam A Jp p z g z = A * Jp - curvature lam A / 2 * sqDist p z ∧
      ∀ u, initialPhi lam A Jp p z g z ≤ initialPhi lam A Jp p z g u := by
  have hc := curvature_pos hl hA
  constructor
  · rw [initial_representation]
    simp
  · intro u
    rw [initial_representation, initial_representation]
    simp only [sqDist_self, mul_zero, add_zero]
    exact le_add_of_nonneg_right (mul_nonneg (by positivity) (sqDist_nonneg u z))

def shiftedGradient (lam : ℝ) (z y g : ι → ℝ) : ι → ℝ :=
  fun i => g i + lam / 2 * (z i - y i)

theorem shifted_norm (lam : ℝ) (z y g : ι → ℝ) :
    normSq (shiftedGradient lam z y g) = normSq g +
      lam * dot g (fun i => z i - y i) + lam ^ 2 / 4 * sqDist z y := by
  unfold normSq shiftedGradient dot sqDist
  simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

def modelPoint (lam A a : ℝ) (z y g : ι → ℝ) : ι → ℝ :=
  fun i => z i - a / curvature lam (A + a) * shiftedGradient lam z y g i

def nextMinimum (lam A a m Jy : ℝ) (z y g : ι → ℝ) : ℝ :=
  m + a * Jy + a * dot g (fun i => z i - y i) +
    a * lam / 4 * sqDist z y -
    a ^ 2 / (2 * curvature lam (A + a)) * normSq (shiftedGradient lam z y g)

def quadraticPlus (lam A a m Jy : ℝ) (z y g u : ι → ℝ) : ℝ :=
  m + curvature lam A / 2 * sqDist u z +
    a * (Jy + dot g (fun i => u i - y i) + lam / 4 * sqDist u y)

theorem completing_square_coordinate (lam A a u z y g : ℝ)
    (hc : curvature lam (A + a) ≠ 0) :
    curvature lam A / 2 * (u - z) ^ 2 +
      a * (g * (u - y) + lam / 4 * (u - y) ^ 2) =
    a * g * (z - y) + a * lam / 4 * (z - y) ^ 2 -
      a ^ 2 / (2 * curvature lam (A + a)) * (g + lam / 2 * (z - y)) ^ 2 +
      curvature lam (A + a) / 2 *
        (u - (z - a / curvature lam (A + a) * (g + lam / 2 * (z - y)))) ^ 2 := by
  field_simp [hc]
  unfold curvature
  ring

/-- Exact quadratic representation, hence a proof of the claimed minimum. -/
theorem quadratic_representation (lam A a m Jy : ℝ) (z y g u : ι → ℝ)
    (hc : curvature lam (A + a) ≠ 0) :
    quadraticPlus lam A a m Jy z y g u =
      nextMinimum lam A a m Jy z y g +
      curvature lam (A + a) / 2 * sqDist u (modelPoint lam A a z y g) := by
  have hsum := congrArg (fun F : ι → ℝ => ∑ i, F i)
    (funext (fun i => completing_square_coordinate lam A a (u i) (z i) (y i) (g i) hc))
  unfold quadraticPlus nextMinimum dot sqDist normSq modelPoint shiftedGradient
  simp only [mul_add, mul_assoc, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum] at hsum
  simp only [mul_add]
  linarith [hsum]

theorem modelPoint_is_minimizer {lam A a : ℝ}
    (hl : 0 < lam) (hA : 0 ≤ A) (ha : 0 < a)
    (m Jy : ℝ) (z y g : ι → ℝ) :
    quadraticPlus lam A a m Jy z y g (modelPoint lam A a z y g) =
        nextMinimum lam A a m Jy z y g ∧
      ∀ u, nextMinimum lam A a m Jy z y g ≤ quadraticPlus lam A a m Jy z y g u := by
  have hc : 0 < curvature lam (A + a) := curvature_pos hl (by linarith)
  constructor
  · rw [quadratic_representation _ _ _ _ _ _ _ _ _ (ne_of_gt hc)]
    simp
  · intro u
    rw [quadratic_representation _ _ _ _ _ _ _ _ _ (ne_of_gt hc)]
    have hn := sqDist_nonneg u (modelPoint lam A a z y g)
    exact le_add_of_nonneg_right (mul_nonneg (by positivity) hn)

def extrapolateVec (lam A a : ℝ) (x z : ι → ℝ) : ι → ℝ :=
  fun i => extrapolate lam A a (x i) (z i)

theorem dot_cross_cancellation {lam A a : ℝ}
    (hl : 0 < lam) (hA : 0 ≤ A) (ha : 0 < a) (x z g : ι → ℝ) :
    A * curvature lam (A + a) *
        dot g (fun i => x i - extrapolateVec lam A a x z i) +
      a * curvature lam A *
        dot g (fun i => z i - extrapolateVec lam A a x z i) = 0 := by
  calc
    _ = ∑ i, g i * (A * curvature lam (A + a) *
          (x i - extrapolateVec lam A a x z i) +
        a * curvature lam A * (z i - extrapolateVec lam A a x z i)) := by
      unfold dot
      simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = 0 := by
      apply Finset.sum_eq_zero
      intro i _
      dsimp only [extrapolateVec]
      rw [cross_cancellation hl hA ha (x i) (z i)]
      ring

/-- Scalar elimination used after expanding the inner products. -/
theorem gap_identity_algebra (lam A a eta m Jx Jy Jxp d e s q : ℝ)
    (hc : curvature lam (A + a) ≠ 0)
    (hw : a ^ 2 = eta * (A + a) * curvature lam (A + a))
    (hcross : A * curvature lam (A + a) * d + a * curvature lam A * e = 0) :
    (A + a) * Jxp - (m + a * Jy + a * e + a * lam / 4 * s -
      a ^ 2 / (2 * curvature lam (A + a)) *
        (q + lam * e + lam ^ 2 / 4 * s)) =
      A * Jx - m - (A + a) * (Jy - Jxp - eta / 2 * q) -
        A * (Jx - Jy - d) -
        a * lam * curvature lam A / (4 * curvature lam (A + a)) * s := by
  have hwq := congrArg (fun t : ℝ => t * q) hw
  field_simp [hc]
  unfold curvature at hwq hcross ⊢
  nlinarith [hwq, hcross]

/-- The exact model-gap identity with the actual explicit coefficients. -/
theorem uag_gap_identity {lam ell A : ℝ}
    (hl : 0 < lam) (he : 0 < ell) (hA : 0 ≤ A)
    (m Jx Jy Jxp : ℝ) (x z g : ι → ℝ) :
    let a := weight lam ell A
    let y := extrapolateVec lam A a x z
    (A + a) * Jxp - nextMinimum lam A a m Jy z y g =
      A * Jx - m - (A + a) * (Jy - Jxp - (lam + ell)⁻¹ / 2 * normSq g) -
        A * (Jx - Jy - dot g (fun i => x i - y i)) -
        a * lam * curvature lam A / (4 * curvature lam (A + a)) * sqDist z y := by
  dsimp only
  have ha := weight_pos hl he hA
  have hc := curvature_pos hl (show 0 ≤ A + weight lam ell A by linarith)
  unfold nextMinimum
  rw [shifted_norm]
  exact gap_identity_algebra lam A (weight lam ell A) ((lam + ell)⁻¹)
    m Jx Jy Jxp _ _ _ _ (ne_of_gt hc) (weight_relation hl he hA)
    (dot_cross_cancellation hl hA ha x z g)

theorem model_point_coordinate_formula (lam A a z y g : ℝ)
    (hc : curvature lam (A + a) ≠ 0) :
    z - a / curvature lam (A + a) * (g + lam / 2 * (z - y)) =
      (curvature lam A * z + a * (lam / 2 * y - g)) / curvature lam (A + a) := by
  field_simp [hc]
  unfold curvature
  ring

theorem displacement_coordinate (lam A a eta x z y g : ℝ)
    (ha : a ≠ 0) (hc : curvature lam (A + a) ≠ 0)
    (hw : a ^ 2 = eta * (A + a) * curvature lam (A + a))
    (hcross : A * curvature lam (A + a) * (x - y) +
      a * curvature lam A * (z - y) = 0) :
    z - a / curvature lam (A + a) * (g + lam / 2 * (z - y)) =
      (y - eta * g) + A / a * ((y - eta * g) - x) := by
  have hwg := congrArg (fun t : ℝ => t * g) hw
  field_simp [ha, hc]
  unfold curvature at hwg hcross ⊢
  nlinarith [hwg, hcross]

/- Algorithm 2's displacement update is the minimizer update. -/
omit [Fintype ι] in
theorem uag_displacement_update {lam ell A : ℝ}
    (hl : 0 < lam) (he : 0 < ell) (hA : 0 ≤ A) (x z g : ι → ℝ) :
    let a := weight lam ell A
    let y := extrapolateVec lam A a x z
    modelPoint lam A a z y g =
      fun i => (y i - (lam + ell)⁻¹ * g i) +
        A / a * ((y i - (lam + ell)⁻¹ * g i) - x i) := by
  dsimp only
  funext i
  have ha := weight_pos hl he hA
  have hc := curvature_pos hl (show 0 ≤ A + weight lam ell A by linarith)
  exact displacement_coordinate lam A (weight lam ell A) ((lam + ell)⁻¹)
    (x i) (z i) _ (g i) (ne_of_gt ha) (ne_of_gt hc)
    (weight_relation hl he hA) (cross_cancellation hl hA ha (x i) (z i))

/-- The accepted tests imply gap decrease; no assumption `G ≥ 0` is used. -/
theorem accepted_gap_nonincrease {lam ell A : ℝ}
    (hl : 0 < lam) (he : 0 < ell) (hA : 0 ≤ A)
    (m Jx Jy Jxp : ℝ) (x z g : ι → ℝ)
    (hdescent : 0 ≤ Jy - Jxp - (lam + ell)⁻¹ / 2 * normSq g)
    (hcurvature : 0 ≤ Jx - Jy -
      dot g (fun i => x i - extrapolateVec lam A (weight lam ell A) x z i)) :
    (A + weight lam ell A) * Jxp -
      nextMinimum lam A (weight lam ell A) m Jy z
        (extrapolateVec lam A (weight lam ell A) x z) g ≤ A * Jx - m := by
  rw [uag_gap_identity hl he hA m Jx Jy Jxp x z g]
  have ha := weight_pos hl he hA
  have hc := curvature_pos hl (show 0 ≤ A + weight lam ell A by linarith)
  have hh := curvature_pos hl hA
  have hd := sqDist_nonneg z (extrapolateVec lam A (weight lam ell A) x z)
  have h1 : 0 ≤ (A + weight lam ell A) *
      (Jy - Jxp - (lam + ell)⁻¹ / 2 * normSq g) := mul_nonneg (by linarith) hdescent
  have h2 := mul_nonneg hA hcurvature
  have h3 : 0 ≤ weight lam ell A * lam * curvature lam A /
      (4 * curvature lam (A + weight lam ell A)) *
        sqDist z (extrapolateVec lam A (weight lam ell A) x z) := by positivity
  linarith

end Coordinates

/-- The other accepted-state invariant uses the anchor curvature test. -/
theorem anchor_step (A a J P r : ℝ) (ha : 0 ≤ a) (hr : 0 ≤ r)
    (hP : P ≤ A * J) : P + a * (J - r) ≤ (A + a) * J := by
  nlinarith [mul_nonneg ha hr]

theorem gap_invariant (G : ℕ → ℝ) (hstep : ∀ n, G (n + 1) ≤ G n) :
    ∀ n, G n ≤ G 0 := by
  intro n
  induction n with
  | zero => exact le_rfl
  | succ n ih => exact le_trans (hstep n) ih

theorem combined_bound (A Jx Janchor G G0 c dist P : ℝ)
    (hrepresentation : P = A * Jx - G + c / 2 * dist)
    (hanchor : P ≤ A * Janchor) (hgap : G ≤ G0) :
    A * (Jx - Janchor) + c / 2 * dist ≤ G0 := by
  linarith

section AcceptedHistory

variable {ι : Type*} [Fintype ι]

structure ModelState (ι : Type*) where
  A : ℝ
  m : ℝ
  x : ι → ℝ
  z : ι → ℝ

def modelGap (J : (ι → ℝ) → ℝ) (s : ModelState ι) : ℝ := s.A * J s.x - s.m

def anchorValue (lam : ℝ) (p : ι → ℝ) (s : ModelState ι) : ℝ :=
  s.m + curvature lam s.A / 2 * sqDist p s.z

def trialState (lam ell : ℝ) (J : (ι → ℝ) → ℝ)
    (grad : (ι → ℝ) → (ι → ℝ)) (s : ModelState ι) : ModelState ι :=
  let a := weight lam ell s.A
  let y := extrapolateVec lam s.A a s.x s.z
  let g := grad y
  let xp := fun i => y i - (lam + ell)⁻¹ * g i
  { A := s.A + a
    m := nextMinimum lam s.A a s.m (J y) s.z y g
    x := xp
    z := modelPoint lam s.A a s.z y g }

/-- Exactly the three tests in `eq:inner-commit-tests`, in coordinates. -/
def acceptedTests (lam ell : ℝ) (J : (ι → ℝ) → ℝ)
    (grad : (ι → ℝ) → (ι → ℝ)) (p : ι → ℝ) (s : ModelState ι) : Prop :=
  let a := weight lam ell s.A
  let y := extrapolateVec lam s.A a s.x s.z
  let g := grad y
  let xp := fun i => y i - (lam + ell)⁻¹ * g i
  0 ≤ J s.x - J y - dot g (fun i => s.x i - y i) ∧
    0 ≤ J p - (J y + dot g (fun i => p i - y i) + lam / 4 * sqDist p y) ∧
    0 ≤ J y - J xp - (lam + ell)⁻¹ / 2 * normSq g

theorem trial_mass_pos {lam ell : ℝ}
    (hl : 0 < lam) (he : 0 < ell) (J : (ι → ℝ) → ℝ)
    (grad : (ι → ℝ) → (ι → ℝ)) (s : ModelState ι) (hA : 0 ≤ s.A) :
    0 < (trialState lam ell J grad s).A := by
  change 0 < s.A + weight lam ell s.A
  have := weight_pos hl he hA
  linarith

theorem trial_gap_nonincrease {lam ell : ℝ}
    (hl : 0 < lam) (he : 0 < ell) (J : (ι → ℝ) → ℝ)
    (grad : (ι → ℝ) → (ι → ℝ)) (p : ι → ℝ) (s : ModelState ι)
    (hA : 0 ≤ s.A) (ht : acceptedTests lam ell J grad p s) :
    modelGap J (trialState lam ell J grad s) ≤ modelGap J s := by
  exact accepted_gap_nonincrease hl he hA s.m (J s.x)
    (J (extrapolateVec lam s.A (weight lam ell s.A) s.x s.z))
    (J (trialState lam ell J grad s).x) s.x s.z
    (grad (extrapolateVec lam s.A (weight lam ell s.A) s.x s.z)) ht.2.2 ht.1

theorem trial_anchor_invariant {lam ell : ℝ}
    (hl : 0 < lam) (he : 0 < ell) (J : (ι → ℝ) → ℝ)
    (grad : (ι → ℝ) → (ι → ℝ)) (p : ι → ℝ) (s : ModelState ι)
    (hA : 0 ≤ s.A) (ht : acceptedTests lam ell J grad p s)
    (hs : anchorValue lam p s ≤ s.A * J p) :
    anchorValue lam p (trialState lam ell J grad s) ≤
      (trialState lam ell J grad s).A * J p := by
  let a := weight lam ell s.A
  let y := extrapolateVec lam s.A a s.x s.z
  let g := grad y
  have ha : 0 < a := weight_pos hl he hA
  have hc : 0 < curvature lam (s.A + a) := curvature_pos hl (by linarith)
  have hm := quadratic_representation lam s.A a s.m (J y) s.z y g p (ne_of_gt hc)
  have htanchor : J y + dot g (fun i => p i - y i) + lam / 4 * sqDist p y ≤ J p := by
    have := ht.2.1
    change 0 ≤ J p - (J y + dot g (fun i => p i - y i) + lam / 4 * sqDist p y) at this
    linarith
  have hmul := mul_le_mul_of_nonneg_left htanchor (le_of_lt ha)
  change nextMinimum lam s.A a s.m (J y) s.z y g +
    curvature lam (s.A + a) / 2 * sqDist p (modelPoint lam s.A a s.z y g) ≤
      (s.A + a) * J p
  rw [← hm]
  unfold quadraticPlus
  change s.m + curvature lam s.A / 2 * sqDist p s.z ≤ s.A * J p at hs
  nlinarith [hs, hmul]

/-- These are the accepted NAG states, with the actual explicit update.
The caller must still connect them to Algorithm 2's control-flow trace. -/
theorem accepted_history_invariants {lam : ℝ} (hl : 0 < lam)
    (N : ℕ) (ell : ℕ → ℝ) (he : ∀ n, n < N → 0 < ell n)
    (J : (ι → ℝ) → ℝ) (grad : (ι → ℝ) → (ι → ℝ)) (p : ι → ℝ)
    (s : ℕ → ModelState ι) (hA0 : 0 ≤ (s 0).A)
    (hanchor0 : anchorValue lam p (s 0) ≤ (s 0).A * J p)
    (hstep : ∀ n, n < N → s (n + 1) = trialState lam (ell n) J grad (s n))
    (htest : ∀ n, n < N → acceptedTests lam (ell n) J grad p (s n)) :
    ∀ n, n ≤ N → 0 ≤ (s n).A ∧ modelGap J (s n) ≤ modelGap J (s 0) ∧
      anchorValue lam p (s n) ≤ (s n).A * J p ∧
      (s n).A * (J (s n).x - J p) +
        curvature lam (s n).A / 2 * sqDist p (s n).z ≤ modelGap J (s 0) := by
  intro n hn
  have hbase : ∀ n, n ≤ N → 0 ≤ (s n).A ∧ modelGap J (s n) ≤ modelGap J (s 0) ∧
      anchorValue lam p (s n) ≤ (s n).A * J p := by
    intro n
    induction n with
    | zero => intro _; exact ⟨hA0, le_rfl, hanchor0⟩
    | succ n ih =>
      intro hle
      have hlt := Nat.lt_of_succ_le hle
      have ih' := ih (Nat.le_of_succ_le hle)
      have hm := trial_mass_pos hl (he n hlt) J grad (s n) ih'.1
      have hg := trial_gap_nonincrease hl (he n hlt) J grad p (s n) ih'.1 (htest n hlt)
      have hp := trial_anchor_invariant hl (he n hlt) J grad p (s n) ih'.1 (htest n hlt) ih'.2.2
      rw [← hstep n hlt] at hm hg hp
      exact ⟨le_of_lt hm, le_trans hg ih'.2.1, hp⟩
  have hbound := hbase n hn
  refine ⟨hbound.1, hbound.2.1, hbound.2.2, ?_⟩
  exact combined_bound (s n).A (J (s n).x) (J p) (modelGap J (s n))
    (modelGap J (s 0)) (curvature lam (s n).A) (sqDist p (s n).z)
    (anchorValue lam p (s n)) (by dsimp [anchorValue, modelGap]; ring)
    hbound.2.2 hbound.2.1

end AcceptedHistory

end
end UAG.Quadratic

#print axioms UAG.Quadratic.positive_weight_relation_unique
#print axioms UAG.Quadratic.initial_minimum
#print axioms UAG.Quadratic.modelPoint_is_minimizer
#print axioms UAG.Quadratic.uag_gap_identity
#print axioms UAG.Quadratic.uag_displacement_update
#print axioms UAG.Quadratic.accepted_history_invariants
