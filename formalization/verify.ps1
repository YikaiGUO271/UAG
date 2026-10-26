$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & lean --version
    if ($LASTEXITCODE -ne 0) { throw 'Cannot run the pinned Lean toolchain.' }
    & lake build UAG
    if ($LASTEXITCODE -ne 0) { throw 'Lean build failed.' }
    & lake env lean verification/AxiomAudit.lean
    if ($LASTEXITCODE -ne 0) { throw 'Transitive axiom audit failed.' }
    Write-Output 'All UAG modules and the transitive theorem-axiom audit passed.'
} finally {
    Pop-Location
}
