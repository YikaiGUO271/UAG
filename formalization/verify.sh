#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
lean --version
lake build UAG
lake env lean verification/AxiomAudit.lean
printf '%s\n' 'All UAG modules and the transitive theorem-axiom audit passed.'
