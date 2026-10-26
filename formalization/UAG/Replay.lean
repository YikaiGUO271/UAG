import Std

/-!
# Deterministic replay and the cost of the first historical witness

This file formalizes the discrete replay interface of UAG Algorithm 4.  The
`step` parameter denotes a complete call of Inner, including all rejected
trials, scale changes, and cache reuses before that call returns.  Its cost is
the number of actual FG queries, not the number of arithmetic operations.

The theorem is conditional on a successful forward prefix of this *same*
deterministic `step`.  Establishing that the manuscript's real-valued Inner
procedure implements this interface remains a separate refinement obligation.
In particular, best-point early exits and caches may not simply be assumed
irrelevant to `step`; the UAG application must establish that fact on the
replayed prefix.  The existence of a negative-curvature historical witness is
also an explicit mathematical input, not an axiom proved by this module.

The lists below are ghost records used in the specification and proof.  The
`replay` function itself reads only a state and a natural-number fuel bound;
it does not read the ghost history or stored historical points.
-/

namespace UAG.Replay

universe u v w

structure Observation (Point : Type u) where
  point : Point
  queries : Nat

def historyCost {Point : Type u} : List (Observation Point) → Nat
  | [] => 0
  | o :: os => o.queries + historyCost os

theorem historyCost_append {Point : Type u}
    (xs ys : List (Observation Point)) :
    historyCost (xs ++ ys) = historyCost xs + historyCost ys := by
  induction xs with
  | nil => simp [historyCost]
  | cons x xs ih => simp [historyCost, ih, Nat.add_assoc]

/-- Different global or local exits are not mistaken for successful updates. -/
inductive ExitReason where
  | accepted
  | direction
  | needsReplay
  | stationary
  | budgetExhausted
  | assertionFailure
  deriving DecidableEq, Repr

inductive InnerResult (Core : Type u) (Point : Type v) where
  | completed (next : Core) (point : Point) (queries : Nat)
  | exit (reason : ExitReason) (queries : Nat)

inductive SearchResult (Point : Type u) where
  | exhausted (queries : Nat)
  | found (point : Point) (queries : Nat)
  | unexpected (reason : ExitReason) (queries : Nat)

def SearchResult.queries {Point : Type u} : SearchResult Point → Nat
  | .exhausted q => q
  | .found _ q => q
  | .unexpected _ q => q

def SearchResult.charge {Point : Type u} (q : Nat) : SearchResult Point → SearchResult Point
  | .exhausted r => .exhausted (q + r)
  | .found p r => .found p (q + r)
  | .unexpected e r => .unexpected e (q + r)

@[simp] theorem charge_queries {Point : Type u} (q : Nat) (r : SearchResult Point) :
    (r.charge q).queries = q + r.queries := by
  cases r <;> rfl

/-- Specification: inspect historical observations until the first witness. -/
def scan {Point : Type u} (bad : Point → Bool) :
    List (Observation Point) → SearchResult Point
  | [] => .exhausted 0
  | o :: os => if bad o.point then .found o.point o.queries
              else (scan bad os).charge o.queries

/-- Actual replay: regenerate observations by calling the deterministic step.
Fuel is a bound on successful updates, not a query budget and not stored history.
-/
def replay {Core : Type u} {Point : Type v}
    (step : Core → InnerResult Core Point) (bad : Point → Bool) :
    Nat → Core → SearchResult Point
  | 0, _ => .exhausted 0
  | n + 1, s =>
      match step s with
      | .completed s' p q =>
          if bad p then .found p q else (replay step bad n s').charge q
      | .exit reason q => .unexpected reason q

/-- Once a witness is found, supplying a larger termination bound does not
alter either that witness or the query count. Thus the fuel is not a new
algorithmic tuning parameter affecting the successful replay trajectory. -/
theorem replay_found_of_le {Core : Type u} {Point : Type v}
    {step : Core → InnerResult Core Point} {bad : Point → Bool}
    {fuel larger : Nat} {s : Core} {p : Point} {q : Nat}
    (hle : fuel ≤ larger)
    (hfound : replay step bad fuel s = .found p q) :
    replay step bad larger s = .found p q := by
  induction fuel generalizing larger s p q with
  | zero => simp [replay] at hfound
  | succ fuel ih =>
      cases larger with
      | zero => omega
      | succ larger =>
          cases hstep : step s with
          | exit reason cost => simp [replay, hstep] at hfound
          | completed next point cost =>
              cases hb : bad point with
              | true => simpa [replay, hstep, hb] using hfound
              | false =>
                  simp only [replay, hstep, hb, Bool.false_eq_true, ↓reduceIte] at hfound ⊢
                  cases hr : replay step bad fuel next with
                  | exhausted spent => simp [hr, SearchResult.charge] at hfound
                  | unexpected reason spent => simp [hr, SearchResult.charge] at hfound
                  | found witness spent =>
                      have hrec := ih (by omega : fuel ≤ larger) hr
                      rw [hrec]
                      simpa only [hr] using hfound

/-- A ghost record of consecutive successful Inner outputs, with exact costs. -/
inductive CoreHistory {Core : Type u} {Point : Type v}
    (step : Core → InnerResult Core Point) :
    Core → List (Observation Point) → Core → Prop
  | nil (s) : CoreHistory step s [] s
  | cons {s s' t p q os}
      (output : step s = .completed s' p q)
      (tail : CoreHistory step s' os t) :
      CoreHistory step s (Observation.mk p q :: os) t

/-- An auxiliary secant step can continue or halt; only continuing steps can
occur in a prefix which subsequently reaches a historical witness. -/
inductive AuxResult (Aux : Type u) where
  | continued (next : Aux) (queries : Nat)
  | halted (reason : ExitReason) (queries : Nat)

structure FullState (Core : Type u) (Aux : Type v) where
  core : Core
  auxiliary : Aux
  queries : Nat

/-- Forward execution with arbitrarily many continuing secant steps interleaved.
Auxiliary transitions can change their entire auxiliary state, and add arbitrary
nonnegative FG cost, but by construction cannot change the core recurrence.
The UAG-specific correspondence to this separation is a manual interface.
-/
inductive ForwardPrefix {Core : Type u} {Aux : Type v} {Point : Type w}
    (step : Core → InnerResult Core Point)
    (side : Core → Aux → AuxResult Aux) :
    FullState Core Aux → List (Observation Point) → FullState Core Aux → Prop
  | nil (s) : ForwardPrefix step side s [] s
  | inner {s t s' p q os}
      (output : step s.core = .completed s' p q)
      (tail : ForwardPrefix step side
        ⟨s', s.auxiliary, s.queries + q⟩ os t) :
      ForwardPrefix step side s (Observation.mk p q :: os) t
  | auxiliary {s t a q os}
      (output : side s.core s.auxiliary = .continued a q)
      (tail : ForwardPrefix step side
        ⟨s.core, a, s.queries + q⟩ os t) :
      ForwardPrefix step side s os t

/-- Erasing continuing auxiliary steps preserves the deterministic core prefix. -/
theorem ForwardPrefix.erase {Core : Type u} {Aux : Type v} {Point : Type w}
    {step : Core → InnerResult Core Point} {side : Core → Aux → AuxResult Aux}
    {s t : FullState Core Aux} {os : List (Observation Point)}
    (h : ForwardPrefix step side s os t) :
    CoreHistory step s.core os t.core := by
  induction h with
  | nil s => exact .nil s.core
  | inner output tail ih => exact .cons output ih
  | auxiliary output tail ih => exact ih

/-- Erasure never increases cost; this also charges all continuing secants. -/
theorem ForwardPrefix.cost {Core : Type u} {Aux : Type v} {Point : Type w}
    {step : Core → InnerResult Core Point} {side : Core → Aux → AuxResult Aux}
    {s t : FullState Core Aux} {os : List (Observation Point)}
    (h : ForwardPrefix step side s os t) :
    s.queries + historyCost os ≤ t.queries := by
  induction h with
  | nil s => simp [historyCost]
  | inner output tail ih =>
      simp only [historyCost] at *
      omega
  | auxiliary output tail ih =>
      dsimp only at ih
      omega

/-- Regenerating a valid successful prefix is identical to scanning its record. -/
theorem replay_eq_scan {Core : Type u} {Point : Type v}
    {step : Core → InnerResult Core Point} (bad : Point → Bool)
    {s t : Core} {os : List (Observation Point)}
    (h : CoreHistory step s os t) :
    replay step bad os.length s = scan bad os := by
  induction h with
  | nil s => rfl
  | cons output tail ih =>
      simp [replay, scan, output, ih]

theorem scan_cost_le {Point : Type u} (bad : Point → Bool)
    (os : List (Observation Point)) :
    (scan bad os).queries ≤ historyCost os := by
  induction os with
  | nil => simp [scan, SearchResult.queries, historyCost]
  | cons o os ih =>
      cases hb : bad o.point with
      | false =>
          simpa only [scan, hb, Bool.false_eq_true, ↓reduceIte,
            charge_queries, historyCost] using Nat.add_le_add_left ih o.queries
      | true => simp [scan, hb, historyCost, SearchResult.queries]

/-- An exact description of the first matching observation and its prefix cost. -/
inductive FirstMatch {Point : Type u} (bad : Point → Bool) :
    List (Observation Point) → Point → Nat → Prop
  | head {o os} (valid : bad o.point = true) :
      FirstMatch bad (o :: os) o.point o.queries
  | tail {o os p q} (skips : bad o.point = false)
      (found : FirstMatch bad os p q) :
      FirstMatch bad (o :: os) p (o.queries + q)

theorem FirstMatch.witness_valid {Point : Type u} {bad : Point → Bool}
    {os : List (Observation Point)} {p : Point} {q : Nat}
    (h : FirstMatch bad os p q) : bad p = true := by
  induction h with
  | head valid => exact valid
  | tail skips found ih => exact ih

theorem FirstMatch.cost_le {Point : Type u} {bad : Point → Bool}
    {os : List (Observation Point)} {p : Point} {q : Nat}
    (h : FirstMatch bad os p q) : q ≤ historyCost os := by
  induction h with
  | head valid => simp [historyCost]
  | tail skips found ih =>
      simp only [historyCost]
      omega

theorem scan_first_match {Point : Type u} (bad : Point → Bool)
    (os : List (Observation Point)) {p : Point} {q : Nat}
    (h : scan bad os = .found p q) : FirstMatch bad os p q := by
  induction os generalizing p q with
  | nil => simp [scan] at h
  | cons o os ih =>
      cases hb : bad o.point with
      | true =>
          simp only [scan, hb, ↓reduceIte] at h
          cases h
          exact .head hb
      | false =>
          simp only [scan, hb, Bool.false_eq_true, ↓reduceIte] at h
          cases hr : scan bad os with
          | exhausted k => simp [hr, SearchResult.charge] at h
          | unexpected e k => simp [hr, SearchResult.charge] at h
          | found p' k =>
              simp only [hr, SearchResult.charge] at h
              cases h
              exact .tail hb (ih hr)

theorem scan_finds {Point : Type u} (bad : Point → Bool)
    (os : List (Observation Point))
    (hex : ∃ o ∈ os, bad o.point = true) :
    ∃ p q, scan bad os = .found p q := by
  induction os with
  | nil => simp at hex
  | cons o os ih =>
      cases hb : bad o.point with
      | true => exact ⟨o.point, o.queries, by simp [scan, hb]⟩
      | false =>
          have htail : ∃ a ∈ os, bad a.point = true := by
            obtain ⟨a, ha, hbad⟩ := hex
            simp only [List.mem_cons] at ha
            rcases ha with ha | ha
            · subst a
              simp [hb] at hbad
            · exact ⟨a, ha, hbad⟩
          obtain ⟨p, q, hp⟩ := ih htail
          exact ⟨p, o.queries + q, by simp [scan, hb, hp, SearchResult.charge]⟩

/-- Main replay theorem.  A historical witness in a valid forward prefix is
found by regenerated computation, is the first witness, and costs no more
than that forward prefix (including its auxiliary work). -/
theorem replay_finds_within_forward_cost
    {Core : Type u} {Aux : Type v} {Point : Type w}
    {step : Core → InnerResult Core Point} {side : Core → Aux → AuxResult Aux}
    (bad : Point → Bool) {s t : FullState Core Aux}
    {os : List (Observation Point)}
    (trace : ForwardPrefix step side s os t)
    (witness : ∃ o ∈ os, bad o.point = true) :
    ∃ p q, replay step bad os.length s.core = .found p q ∧
      FirstMatch bad os p q ∧ bad p = true ∧
      q ≤ historyCost os ∧ s.queries + q ≤ t.queries := by
  obtain ⟨p, q, hpq⟩ := scan_finds bad os witness
  have hf := scan_first_match bad os hpq
  refine ⟨p, q, ?_, hf, hf.witness_valid, hf.cost_le, ?_⟩
  · rw [replay_eq_scan bad trace.erase]
    exact hpq
  · have htotal := trace.cost
    have hprefix := hf.cost_le
    omega

/-- This version uses the manuscript's F, which excludes secants.  If the
successful-prefix cost is at most F, replay also costs at most F. -/
theorem replay_finds_within_inner_budget
    {Core : Type u} {Point : Type v}
    {step : Core → InnerResult Core Point} (bad : Point → Bool)
    {s t : Core} {os : List (Observation Point)} {F : Nat}
    (trace : CoreHistory step s os t)
    (witness : ∃ o ∈ os, bad o.point = true)
    (prefix_budget : historyCost os ≤ F) :
    ∃ p q, replay step bad os.length s = .found p q ∧
      FirstMatch bad os p q ∧ q ≤ F := by
  obtain ⟨p, q, hpq⟩ := scan_finds bad os witness
  have hf := scan_first_match bad os hpq
  refine ⟨p, q, ?_, hf, Nat.le_trans hf.cost_le prefix_budget⟩
  rw [replay_eq_scan bad trace]
  exact hpq

/-- The terminal probe or other work after the successful prefix can be added
to the forward budget without affecting the replay guarantee. -/
theorem replay_finds_with_forward_suffix
    {Core : Type u} {Point : Type v}
    {step : Core → InnerResult Core Point} (bad : Point → Bool)
    {s t : Core} {os : List (Observation Point)}
    (suffix : List (Observation Point))
    (trace : CoreHistory step s os t)
    (witness : ∃ o ∈ os, bad o.point = true) :
    ∃ p q, replay step bad os.length s = .found p q ∧
      FirstMatch bad os p q ∧ q ≤ historyCost (os ++ suffix) := by
  apply replay_finds_within_inner_budget bad trace witness
  rw [historyCost_append]
  omega

/-- Every actual query before the certified replay endpoint passes C<C_start+F.
An external budget at least C_start+F also preserves that strict guard after
the algorithm takes the minimum of the two limits. -/
theorem replay_query_guard (start F spent consumed external : Nat)
    (hspent : spent ≤ F) (hconsumed : consumed < spent)
    (hexternal : start + F ≤ external) :
    start + consumed < min external (start + F) := by
  omega

/-- Restoring recurrence state cannot erase the queries spent during replay. -/
def restoreCore {Core : Type u} {Aux : Type v}
    (saved current : FullState Core Aux) : FullState Core Aux :=
  { current with core := saved.core }

@[simp] theorem restoreCore_core {Core : Type u} {Aux : Type v}
    (saved current : FullState Core Aux) :
    (restoreCore saved current).core = saved.core := rfl

@[simp] theorem restoreCore_queries {Core : Type u} {Aux : Type v}
    (saved current : FullState Core Aux) :
    (restoreCore saved current).queries = current.queries := rfl

end UAG.Replay
