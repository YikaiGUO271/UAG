import UAG.Accounting
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Exact real step-size ledger and the logarithmic FG bound

This module removes the integer rounding allowance from `Accounting`.
It executes the recorded doubling/halving events on a real positive scale,
proves the exact scale/exponent relation, then derives the logarithmic ledger
and the manuscript's real-valued query bounds.

The substantive analytical input is the upper bound on the final persistent
scale, `runRealScale b events ≤ 2*L`.  The numerical predicates ensuring that
bound, and the refinement from the complete UAG program to the recorded
events, are not assumed to have been formalized here.
-/

namespace UAG.ScaleBridge

open UAG.Accounting

noncomputable section

/-- Replay's temporary scale is restored; these events concern only the
persistent scale updated by the forward inner recurrences. -/
def runRealScale (b : ℝ) : List ScaleEvent → ℝ
  | [] => b
  | .double :: es => runRealScale (2 * b) es
  | .halve :: es => runRealScale (b / 2) es

theorem runRealScale_eq (es : List ScaleEvent) (b : ℝ) :
    runRealScale b es = b * (2 : ℝ) ^ scaleNet es := by
  induction es generalizing b with
  | nil => simp [runRealScale, scaleNet]
  | cons e es ih =>
    cases e with
    | double =>
      simp only [runRealScale, scaleNet, ih]
      rw [zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      norm_num
      ring
    | halve =>
      simp only [runRealScale, scaleNet, ih]
      rw [zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      norm_num
      ring

def persistentScale (b : ℝ) (cs : List CallPacket) : ℝ :=
  runRealScale b (scaleEvents cs)

theorem persistentScale_eq (b : ℝ) (cs : List CallPacket) :
    persistentScale b cs = b * (2 : ℝ) ^ finalExponent cs := by
  rw [persistentScale, runRealScale_eq, global_scale_ledger]
  rfl

theorem persistentScale_pos (b : ℝ) (cs : List CallPacket) (hb : 0 < b) :
    0 < persistentScale b cs := by
  rw [persistentScale_eq]
  exact mul_pos hb (zpow_pos (by norm_num) _)

def logarithmicAllowance (b L : ℝ) : ℝ := Real.log (2 * L / b) / Real.log 2

/-- The log conversion is exact; there is no ceiling and no added constant. -/
theorem exponent_le_log (b L : ℝ) (cs : List CallPacket)
    (hb : 0 < b) (hL : 0 < L) (hscale : persistentScale b cs ≤ 2 * L) :
    ((finalExponent cs : Int) : ℝ) ≤ logarithmicAllowance b L := by
  have hp : 0 < (2 : ℝ) ^ finalExponent cs := zpow_pos (by norm_num) _
  have hr : 0 < 2 * L / b := div_pos (mul_pos (by norm_num) hL) hb
  have hpow : (2 : ℝ) ^ finalExponent cs ≤ 2 * L / b := by
    apply (le_div_iff₀ hb).2
    rw [persistentScale_eq] at hscale
    nlinarith
  have hlog := (Real.log_le_log_iff hp hr).2 hpow
  rw [Real.log_zpow] at hlog
  exact (le_div_iff₀ (Real.log_pos (by norm_num))).2 hlog

theorem exact_real_scale_ledger (b : ℝ) (cs : List CallPacket) (hb : 0 < b) :
    (failures cs : ℝ) - (lowerings cs : ℝ) =
      Real.log (persistentScale b cs / b) / Real.log 2 := by
  have hb0 : b ≠ 0 := ne_of_gt hb
  have hratio : persistentScale b cs / b = (2 : ℝ) ^ finalExponent cs := by
    rw [persistentScale_eq]
    field_simp
  rw [hratio, Real.log_zpow]
  have hlog0 : Real.log (2 : ℝ) ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  rw [mul_div_cancel_right₀ _ hlog0]
  simp [finalExponent]

/-- Calibrated `b ≤ 2L` makes the logarithmic allowance nonnegative.
This fact is useful independently of the final persistent scale. -/
theorem logarithmicAllowance_nonneg (b L : ℝ) (hb : 0 < b) (hbL : b ≤ 2 * L) :
    0 ≤ logarithmicAllowance b L := by
  have hratio : 1 ≤ 2 * L / b := (le_div_iff₀ hb).2 (by simpa using hbL)
  exact div_nonneg (Real.log_nonneg hratio) (le_of_lt (Real.log_pos (by norm_num)))

theorem real_failure_bound (b L : ℝ) (cs : List CallPacket)
    (hb : 0 < b) (hL : 0 < L) (hscale : persistentScale b cs ≤ 2 * L) :
    (failures cs : ℝ) ≤ (successes cs : ℝ) + (cs.length : ℝ) +
      logarithmicAllowance b L := by
  have he := exponent_le_log b L cs hb hL hscale
  have hd : (lowerings cs : ℝ) ≤ (successes cs : ℝ) + (cs.length : ℝ) := by
    exact_mod_cast global_lowerings_bound cs
  simp only [finalExponent, Int.cast_sub, Int.cast_natCast] at he
  linarith

theorem real_forward_bound (b L : ℝ) (cs : List CallPacket)
    (hb : 0 < b) (hL : 0 < L) (hscale : persistentScale b cs ≤ 2 * L) :
    (forwardQueries cs : ℝ) ≤ 5 * (successes cs : ℝ) +
      5 * (cs.length : ℝ) + 2 * logarithmicAllowance b L := by
  have hf : (forwardQueries cs : ℝ) ≤ 3 * (successes cs : ℝ) +
      3 * (cs.length : ℝ) + 2 * (failures cs : ℝ) := by
    exact_mod_cast global_forward_bound cs
  have hs := real_failure_bound b L cs hb hL hscale
  linarith

/-- Exact real-valued version of eq:inner-global-fg. -/
theorem real_inner_query_bound (b L : ℝ) (cs : List CallPacket)
    (hb : 0 < b) (hL : 0 < L) (hscale : persistentScale b cs ≤ 2 * L) :
    ((forwardQueries cs + replayQueries cs : Nat) : ℝ) ≤
      10 * (successes cs : ℝ) + 10 * (cs.length : ℝ) +
        4 * logarithmicAllowance b L := by
  have hf := real_forward_bound b L cs hb hL hscale
  have hr : (replayQueries cs : ℝ) ≤ (forwardQueries cs : ℝ) := by
    exact_mod_cast global_replay_bound cs
  push_cast
  linarith

/-- Exact real-valued version of eq:all-fg for prefix-certified replay. -/
theorem real_full_query_bound (b L : ℝ) (cs : List CallPacket) (calibration : Nat)
    (hb : 0 < b) (hL : 0 < L) (hscale : persistentScale b cs ≤ 2 * L) :
    ((1 + calibration + packetQueries cs : Nat) : ℝ) ≤
      1 + (calibration : ℝ) + 11 * (successes cs : ℝ) +
        12 * (cs.length : ℝ) + 4 * logarithmicAllowance b L + (quota cs : ℝ) := by
  have hi := real_inner_query_bound b L cs hb hL hscale
  have hp : (packetQueries cs : ℝ) ≤ (forwardQueries cs : ℝ) +
      (replayQueries cs : ℝ) + 2 * (cs.length : ℝ) +
      (successes cs : ℝ) + (quota cs : ℝ) := by
    exact_mod_cast packet_cost_bound cs
  push_cast at hi ⊢
  linarith

/-- Composition with an independent operational replay cost theorem. -/
theorem real_full_query_bound_of_replay_cost (b L : ℝ) (cs : List CallPacket)
    (calibration actualReplay : Nat) (hb : 0 < b) (hL : 0 < L)
    (hscale : persistentScale b cs ≤ 2 * L)
    (hreplay : actualReplay ≤ forwardQueries cs) :
    ((1 + calibration + nonReplayQueries cs + actualReplay : Nat) : ℝ) ≤
      1 + (calibration : ℝ) + 11 * (successes cs : ℝ) +
        12 * (cs.length : ℝ) + 4 * logarithmicAllowance b L + (quota cs : ℝ) := by
  have hf := real_forward_bound b L cs hb hL hscale
  have hr : (actualReplay : ℝ) ≤ (forwardQueries cs : ℝ) := by
    exact_mod_cast hreplay
  have hn : (nonReplayQueries cs : ℝ) ≤ (forwardQueries cs : ℝ) +
      2 * (cs.length : ℝ) + (successes cs : ℝ) + (quota cs : ℝ) := by
    exact_mod_cast nonReplay_cost_bound cs
  push_cast
  linarith

end

end UAG.ScaleBridge
