# --- Local Operator Test Coverage Check Script (PowerShell) ---

$ErrorActionPreference = "Stop"

Write-Host "--- Local Operator Test Coverage Check Script (PowerShell) ---"

$ProjectRoot = (Get-Item -Path (Join-Path $PSScriptRoot "..")).FullName
$CoverageScript = Join-Path $ProjectRoot ".github\check_operator_coverage.py"

Write-Host "Project Root: $ProjectRoot"
Write-Host "Coverage Script: $CoverageScript"

if (-not (Test-Path $CoverageScript)) {
    Write-Error "Error: Coverage script '$CoverageScript' not found."
    exit 1
}

Write-Host "Running operator test coverage check..."
python3 "$CoverageScript"

$CoverageResult = $LASTEXITCODE
if ($CoverageResult -eq 0) {
    Write-Host "Coverage check complete. All operators seem to have test files."
} else {
    Write-Warning "Coverage check complete. Some operators may be missing test files. Check output above."
}

exit $CoverageResult
