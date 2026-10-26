import Std

/-!
# UAG: query accounting for finite execution prefixes

This file formalizes a counting abstraction of Algorithms 1--4 of the manuscript.
It does NOT claim that the mathematical branch predicates of UAG have all been
formalized.  Its interface is a finite `CallTrace`, whose constructors record the
order and local query allowances of an inner recurrence, including a final
partial call.  There is no theorem-specific axiom or unfinished proof.

The remaining mathematical interface is explicit: the terminal scale exponent
must be bounded by `K`; replay events must be a forward event prefix.  Neither
the final forward bound nor the final full-FG bound is a field of the interface.

Source correspondence: main.tex, eq:inner-one-call-scale-ledger,
eq:inner-global-fg, eq:inner-base-global-fg, and eq:all-fg.
-/

namespace UAG.Accounting

def sumBy {α : Type} (f : α → Nat) : List α → Nat
  | [] => 0
  | a :: as => f a + sumBy f as

def eventCost : List Nat → Nat := sumBy id

theorem eventCost_append (xs ys : List Nat) :
    eventCost (xs ++ ys) = eventCost xs + eventCost ys := by
  induction xs with
  | nil => simp [eventCost, sumBy]
  | cons a as ih => simp [eventCost, sumBy] at *; omega

theorem eventCost_replicate_one (n : Nat) :
    eventCost (List.replicate n 1) = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [List.replicate_succ, eventCost, sumBy] at *; omega

/-- A block performs zero, one, or at most `cap` actual oracle queries.
Zero allows a cached value or an early return before that query. -/
structure QueryBlock (cap : Nat) where
  queries : Nat
  withinCap : queries ≤ cap

def blockCost {cap : Nat} : List (QueryBlock cap) → Nat :=
  sumBy QueryBlock.queries

theorem blockCost_le {cap : Nat} (xs : List (QueryBlock cap)) :
    blockCost xs ≤ cap * xs.length := by
  induction xs with
  | nil => simp [blockCost, sumBy]
  | cons x xs ih =>
    have hx := x.withinCap
    simp [blockCost, sumBy, List.length_cons, Nat.mul_add] at *
    omega

theorem eventCost_map_queries {cap : Nat} (xs : List (QueryBlock cap)) :
    eventCost (xs.map QueryBlock.queries) = blockCost xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [eventCost, blockCost, sumBy] at *; omega

/-- `true` means that a successful NAG update has occurred in this recurrence.
At the initial state the nonzero gradient rules out a correction query. -/
inductive Correction : Bool → Type
  | cached {seen : Bool} : Correction seen
  | queried : Correction true

def correctionAllowance (seen : Bool) : Nat := if seen then 1 else 0

def Correction.cost {seen : Bool} : Correction seen → Nat
  | .cached => 0
  | .queried => 1

theorem Correction.cost_le {seen : Bool} (c : Correction seen) :
    c.cost ≤ correctionAllowance seen := by
  cases c <;> simp [Correction.cost, correctionAllowance]

/-- The final partial iteration may finish before a probe/search, during a
nonfailed probe, or at any point of the final NAG search.  Failed probes that
precede the exit are recorded separately. -/
inductive Exit
  | immediate
  | probe (terminalQuery : QueryBlock 1)
  | nag (failedTrials : List (QueryBlock 2)) (terminalTrial : QueryBlock 2)

def Exit.queries : Exit → Nat
  | .immediate => 0
  | .probe q => q.queries
  | .nag fs q => blockCost fs + q.queries

def Exit.failures : Exit → Nat
  | .immediate => 0
  | .probe _ => 0
  | .nag fs _ => fs.length

def Exit.lowerings : Exit → Nat
  | .immediate => 0
  | .probe _ => 0
  | .nag _ _ => 1

def Exit.events : Exit → List Nat
  | .immediate => []
  | .probe q => [q.queries]
  | .nag fs q => fs.map QueryBlock.queries ++ [q.queries]

theorem Exit.cost_events (e : Exit) : eventCost e.events = e.queries := by
  cases e with
  | immediate => rfl
  | probe q => simp [Exit.events, Exit.queries, eventCost, sumBy]
  | nag fs q =>
    simp only [Exit.events, Exit.queries, eventCost_append, eventCost_map_queries]
    simp [eventCost, sumBy]

theorem Exit.queries_le (e : Exit) : e.queries ≤ 2 + 2 * e.failures := by
  cases e with
  | immediate => decide
  | probe q => have hq := q.withinCap; simp [Exit.queries, Exit.failures]; omega
  | nag fs q =>
    have hf := blockCost_le fs
    have hq := q.withinCap
    simp [Exit.queries, Exit.failures]
    omega

theorem Exit.lowerings_le (e : Exit) : e.lowerings ≤ 1 := by
  cases e <;> simp [Exit.lowerings]

/-- A grammar of one inner recurrence, possibly stopped partway through.
Each successful constructor contains precisely one NAG search (one lowering),
an arbitrary list of failed NAG trials, and at most one post-update secant.
The `tail` begins at a state permitting one correction.  Scale changes are
counted even if no successful update follows them. -/
inductive CallTrace : Bool → Type
  | terminal {seen : Bool} (correction : Correction seen)
      (failedProbes : Nat) (exit : Exit) : CallTrace seen
  | success {seen : Bool} (correction : Correction seen)
      (failedProbes : Nat) (failedNAG : List (QueryBlock 2))
      (acceptedNAG : QueryBlock 2) (secant : QueryBlock 1)
      (tail : CallTrace true) : CallTrace seen

def CallTrace.successes {seen : Bool} : CallTrace seen → Nat
  | .terminal _ _ _ => 0
  | .success _ _ _ _ _ tail => 1 + tail.successes

def CallTrace.failures {seen : Bool} : CallTrace seen → Nat
  | .terminal _ p e => p + e.failures
  | .success _ p fs _ _ tail => p + fs.length + tail.failures

def CallTrace.lowerings {seen : Bool} : CallTrace seen → Nat
  | .terminal _ _ e => e.lowerings
  | .success _ _ _ _ _ tail => 1 + tail.lowerings

def CallTrace.queries {seen : Bool} : CallTrace seen → Nat
  | .terminal c p e => c.cost + p + e.queries
  | .success c p fs q _ tail => c.cost + p + blockCost fs + q.queries + tail.queries

def CallTrace.secantQueries {seen : Bool} : CallTrace seen → Nat
  | .terminal _ _ _ => 0
  | .success _ _ _ _ q tail => q.queries + tail.secantQueries

def CallTrace.events {seen : Bool} : CallTrace seen → List Nat
  | .terminal c p e => [c.cost] ++ List.replicate p 1 ++ e.events
  | .success c p fs q _ tail =>
      [c.cost] ++ List.replicate p 1 ++ fs.map QueryBlock.queries ++
      [q.queries] ++ tail.events

theorem CallTrace.cost_events {seen : Bool} (t : CallTrace seen) :
    eventCost t.events = t.queries := by
  induction t with
  | terminal c p e =>
    simp only [CallTrace.events, CallTrace.queries, eventCost_append,
      eventCost_replicate_one, Exit.cost_events]
    simp [eventCost, sumBy]
  | success c p fs q sc tail ih =>
    simp only [CallTrace.events, CallTrace.queries, eventCost_append,
      eventCost_replicate_one, eventCost_map_queries, ih]
    simp [eventCost, sumBy]

/-- A stronger per-recurrence query bound, proved by its control-flow grammar. -/
theorem CallTrace.queries_le {seen : Bool} (t : CallTrace seen) :
    t.queries ≤ 3 * t.successes + 2 + 2 * t.failures + correctionAllowance seen := by
  induction t with
  | terminal c p e =>
    have hc := c.cost_le
    have he := e.queries_le
    simp only [CallTrace.queries, CallTrace.successes, CallTrace.failures]
    omega
  | success c p fs q sc tail ih =>
    have hc := c.cost_le
    have hf := blockCost_le fs
    have hq := q.withinCap
    simp only [correctionAllowance, ite_true] at ih
    simp only [CallTrace.queries, CallTrace.successes, CallTrace.failures]
    omega

theorem CallTrace.forward_bound (t : CallTrace false) :
    t.queries ≤ 3 * t.successes + 3 + 2 * t.failures := by
  have ht := t.queries_le
  simp [correctionAllowance] at ht
  omega

theorem CallTrace.lowerings_le {seen : Bool} (t : CallTrace seen) :
    t.lowerings ≤ t.successes + 1 := by
  induction t with
  | terminal c p e => exact e.lowerings_le
  | success c p fs q sc tail ih =>
    simp only [CallTrace.lowerings, CallTrace.successes]
    omega

theorem CallTrace.secants_le {seen : Bool} (t : CallTrace seen) :
    t.secantQueries ≤ t.successes := by
  induction t with
  | terminal c p e => exact Nat.le_refl 0
  | success c p fs q sc tail ih =>
    have hq := sc.withinCap
    simp only [CallTrace.secantQueries, CallTrace.successes]
    omega

/-- Structural prefix relation.  This is a refinement interface: a separate
execution proof must show the actual replay produces such a prefix. -/
def IsEventPrefix (pre full : List Nat) : Prop :=
  ∃ suffix, full = pre ++ suffix

theorem prefix_cost_le {pre full : List Nat} (h : IsEventPrefix pre full) :
    eventCost pre ≤ eventCost full := by
  obtain ⟨suffix, rfl⟩ := h
  rw [eventCost_append]
  omega

/-- All noncalibration FG work associated with one recurrence.  A localized
recurrence has zero initial quota slots; an ordinary one has the prescribed
number, with zero-query slots allowed.  There is at most one replay per call. -/
structure CallPacket where
  forward : CallTrace false
  replay : List Nat
  replayPrefix : IsEventPrefix replay forward.events
  directional : QueryBlock 2
  quotaSlots : List (QueryBlock 1)

def CallPacket.queries (c : CallPacket) : Nat :=
  c.forward.queries + eventCost c.replay + c.directional.queries +
    c.forward.secantQueries + blockCost c.quotaSlots

def successes (cs : List CallPacket) : Nat := sumBy (fun c => c.forward.successes) cs
def failures (cs : List CallPacket) : Nat := sumBy (fun c => c.forward.failures) cs
def lowerings (cs : List CallPacket) : Nat := sumBy (fun c => c.forward.lowerings) cs
def forwardQueries (cs : List CallPacket) : Nat := sumBy (fun c => c.forward.queries) cs
def replayQueries (cs : List CallPacket) : Nat := sumBy (fun c => eventCost c.replay) cs
def quota (cs : List CallPacket) : Nat := sumBy (fun c => c.quotaSlots.length) cs
def packetQueries (cs : List CallPacket) : Nat := sumBy CallPacket.queries cs

/-- All call-associated work other than replay.  Keeping it separate lets a
semantic replay theorem provide its own cost, without identifying its
coarse-grained observation trace with the query blocks used here. -/
def nonReplayQueries (cs : List CallPacket) : Nat :=
  sumBy (fun c => c.forward.queries + c.directional.queries +
    c.forward.secantQueries + blockCost c.quotaSlots) cs

theorem global_forward_bound (cs : List CallPacket) :
    forwardQueries cs ≤ 3 * successes cs + 3 * cs.length + 2 * failures cs := by
  induction cs with
  | nil => simp [forwardQueries, successes, failures, sumBy]
  | cons c cs ih =>
    have hc := c.forward.forward_bound
    simp [forwardQueries, successes, failures, sumBy] at *
    omega

theorem global_lowerings_bound (cs : List CallPacket) :
    lowerings cs ≤ successes cs + cs.length := by
  induction cs with
  | nil => simp [lowerings, successes, sumBy]
  | cons c cs ih =>
    have hc := c.forward.lowerings_le
    simp [lowerings, successes, sumBy] at *
    omega

theorem global_replay_bound (cs : List CallPacket) :
    replayQueries cs ≤ forwardQueries cs := by
  induction cs with
  | nil => exact Nat.le_refl 0
  | cons c cs ih =>
    have hc := prefix_cost_le c.replayPrefix
    rw [c.forward.cost_events] at hc
    simp only [replayQueries, forwardQueries, sumBy] at *
    omega

/-- All packet additions are derived from local constructor caps. -/
theorem packet_cost_bound (cs : List CallPacket) :
    packetQueries cs ≤ forwardQueries cs + replayQueries cs +
      2 * cs.length + successes cs + quota cs := by
  induction cs with
  | nil => simp [packetQueries, forwardQueries, replayQueries, successes, quota, sumBy]
  | cons c cs ih =>
    have hd := c.directional.withinCap
    have hs := c.forward.secants_le
    have hq := blockCost_le c.quotaSlots
    simp [packetQueries, forwardQueries, replayQueries, successes, quota,
      sumBy, CallPacket.queries] at *
    omega

theorem nonReplay_cost_bound (cs : List CallPacket) :
    nonReplayQueries cs ≤ forwardQueries cs + 2 * cs.length +
      successes cs + quota cs := by
  induction cs with
  | nil => simp [nonReplayQueries, forwardQueries, successes, quota, sumBy]
  | cons c cs ih =>
    have hd := c.directional.withinCap
    have hs := c.forward.secants_le
    have hq := blockCost_le c.quotaSlots
    simp [nonReplayQueries, forwardQueries, successes, quota, sumBy] at *
    omega

/-- These are precisely the updates of the *persistent* reciprocal step size.
Replay's temporary scale is excluded because it is restored on return;
secant trials update a separate scale and do not change this exponent. -/
inductive ScaleEvent
  | double
  | halve

def scaleNet : List ScaleEvent → Int
  | [] => 0
  | .double :: es => 1 + scaleNet es
  | .halve :: es => -1 + scaleNet es

def runScale (s : Int) : List ScaleEvent → Int
  | [] => s
  | .double :: es => runScale (s + 1) es
  | .halve :: es => runScale (s - 1) es

theorem runScale_eq (es : List ScaleEvent) (s : Int) :
    runScale s es = s + scaleNet es := by
  induction es generalizing s with
  | nil => simp [runScale, scaleNet]
  | cons e es ih =>
    cases e <;> simp only [runScale, scaleNet, ih] <;> omega

theorem scaleNet_append (xs ys : List ScaleEvent) :
    scaleNet (xs ++ ys) = scaleNet xs + scaleNet ys := by
  induction xs with
  | nil => simp [scaleNet]
  | cons e es ih =>
    cases e <;> simp only [List.cons_append, scaleNet, ih] <;> omega

theorem scaleNet_doubles (n : Nat) :
    scaleNet (List.replicate n ScaleEvent.double) = (n : Int) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [List.replicate_succ, scaleNet, ih]
    omega

def Exit.scaleEvents : Exit → List ScaleEvent
  | .immediate => []
  | .probe _ => []
  | .nag fs _ => .halve :: List.replicate fs.length .double

theorem Exit.scale_ledger (e : Exit) :
    scaleNet e.scaleEvents = (e.failures : Int) - (e.lowerings : Int) := by
  cases e with
  | immediate => rfl
  | probe q => rfl
  | nag fs q =>
    simp only [Exit.scaleEvents, Exit.failures, Exit.lowerings, scaleNet,
      scaleNet_doubles]
    omega

def CallTrace.scaleEvents {seen : Bool} : CallTrace seen → List ScaleEvent
  | .terminal _ p e => List.replicate p .double ++ e.scaleEvents
  | .success _ p fs _ _ tail =>
      List.replicate p .double ++ [.halve] ++
        List.replicate fs.length .double ++ tail.scaleEvents

theorem CallTrace.scale_ledger {seen : Bool} (t : CallTrace seen) :
    scaleNet t.scaleEvents = (t.failures : Int) - (t.lowerings : Int) := by
  induction t with
  | terminal c p e =>
    simp only [CallTrace.scaleEvents, CallTrace.failures, CallTrace.lowerings,
      scaleNet_append, scaleNet_doubles, Exit.scale_ledger]
    omega
  | success c p fs q sc tail ih =>
    simp only [CallTrace.scaleEvents, CallTrace.failures, CallTrace.lowerings,
      scaleNet_append, scaleNet_doubles, scaleNet, ih]
    omega

def scaleEvents : List CallPacket → List ScaleEvent
  | [] => []
  | c :: cs => c.forward.scaleEvents ++ scaleEvents cs

theorem global_scale_ledger (cs : List CallPacket) :
    scaleNet (scaleEvents cs) = (failures cs : Int) - (lowerings cs : Int) := by
  induction cs with
  | nil => rfl
  | cons c cs ih =>
    simp only [scaleEvents, scaleNet_append, CallTrace.scale_ledger, failures,
      lowerings, sumBy] at *
    omega

/-- The dimensionless integer exponent starts at zero after calibration.
Every recorded failure doubles the scale and every search lowering halves it.
This net exponent is carried globally, never reset at recurrence boundaries. -/
def finalExponent (cs : List CallPacket) : Int :=
  (failures cs : Int) - (lowerings cs : Int)

/-- The global ledger is derived from the actual sequence of doubling and
halving events, including failed/terminal searches across every call boundary. -/
theorem finalExponent_eq_runScale (cs : List CallPacket) :
    finalExponent cs = runScale 0 (scaleEvents cs) := by
  rw [runScale_eq, global_scale_ledger]
  simp [finalExponent]

theorem scale_bound_interface (cs : List CallPacket) (K : Nat)
    (h : runScale 0 (scaleEvents cs) ≤ (K : Int)) :
    finalExponent cs ≤ (K : Int) := by
  rw [finalExponent_eq_runScale]
  exact h

theorem failure_bound_of_scale (cs : List CallPacket) (K : Nat)
    (hscale : finalExponent cs ≤ (K : Int)) :
    failures cs ≤ successes cs + cs.length + K := by
  have hd := global_lowerings_bound cs
  simp only [finalExponent] at hscale
  omega

theorem forward_bound_of_scale (cs : List CallPacket) (K : Nat)
    (hscale : finalExponent cs ≤ (K : Int)) :
    forwardQueries cs ≤ 5 * successes cs + 5 * cs.length + 2 * K := by
  have hf := global_forward_bound cs
  have hs := failure_bound_of_scale cs K hscale
  omega

/-- Discrete form of eq:inner-global-fg, including replay and partial calls. -/
theorem inner_query_bound (cs : List CallPacket) (K : Nat)
    (hscale : finalExponent cs ≤ (K : Int)) :
    forwardQueries cs + replayQueries cs ≤
      10 * successes cs + 10 * cs.length + 4 * K := by
  have hf := forward_bound_of_scale cs K hscale
  have hr := global_replay_bound cs
  omega

/-- Discrete form of eq:all-fg. `calibration` counts calibration queries after
the initial point.  The `K` to logarithm conversion and the analytic bound on
each initial secant quota are deliberately separate mathematical interfaces. -/
theorem full_query_bound (cs : List CallPacket) (K calibration : Nat)
    (hscale : finalExponent cs ≤ (K : Int)) :
    1 + calibration + packetQueries cs ≤
      1 + calibration + 11 * successes cs + 12 * cs.length + 4 * K + quota cs := by
  have hi := inner_query_bound cs K hscale
  have hp := packet_cost_bound cs
  omega

/-- Composition interface for a separately proved operational replay theorem.
`hreplay` is the precise boundary to that theorem, not a claim established by
the control-flow grammar of this file.  The remaining FG accounting is derived
here.  In particular the final FG inequality is never assumed. -/
theorem full_query_bound_of_replay_cost (cs : List CallPacket)
    (K calibration actualReplay : Nat)
    (hscale : finalExponent cs ≤ (K : Int))
    (hreplay : actualReplay ≤ forwardQueries cs) :
    1 + calibration + nonReplayQueries cs + actualReplay ≤
      1 + calibration + 11 * successes cs + 12 * cs.length + 4 * K + quota cs := by
  have hf := forward_bound_of_scale cs K hscale
  have hn := nonReplay_cost_bound cs
  omega

end UAG.Accounting
