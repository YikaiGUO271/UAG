import UAG
import Lean

open Lean Elab Command

/- Audit transitive dependencies of every UAG theorem, not just selected examples.
   `sorryAx`, added research axioms, and native-reduction axioms are rejected.
   The three permitted names are Lean's usual classical mathematical foundations.
-/
run_cmd do
  let env ← getEnv
  let allowed : List Name := [`propext, `Quot.sound, `Classical.choice]
  let mut checked : Nat := 0
  for (name, info) in env.constants.toList do
    if name.getRoot == `UAG then
      match info with
      | .axiomInfo _ => throwError "Unexpected local axiom: {name}"
      | .thmInfo _ =>
        let axioms ← collectAxioms name
        for ax in axioms do
          unless allowed.contains ax do
            throwError "Unapproved axiom {ax} in {name}"
        logInfo m!"AUDIT {name}: {axioms.toList}"
        checked := checked + 1
      | _ => pure ()
  if checked == 0 then throwError "No UAG theorems were audited"
  logInfo m!"PASS: {checked} UAG theorems; no unfinished-proof or extra axioms."
