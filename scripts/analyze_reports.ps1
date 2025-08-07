# --- Local Test Report Analysis Script (PowerShell) ---

$ErrorActionPreference = "Stop"

Write-Host "--- Local Test Report Analysis Script (PowerShell) ---"

$ProjectRoot = (Get-Item -Path (Join-Path $PSScriptRoot "..")).FullName
$ReportsDir = Join-Path $ProjectRoot "reports"
$AnalysisScript = Join-Path $ProjectRoot ".github\check_all_results.py"

Write-Host "Project Root: $ProjectRoot"
Write-Host "Reports Directory: $ReportsDir"
Write-Host "Analysis Script: $AnalysisScript"

if (-not (Test-Path $ReportsDir)) {
    Write-Error "Error: Reports directory '$ReportsDir' not found. Run the build and test script first."
    exit 1
}

if (-not (Test-Path $AnalysisScript)) {
    Write-Error "Error: Analysis script '$AnalysisScript' not found."
    exit 1
}

# Find all report-*.csv files in the reports directory
$ReportFiles = Get-ChildItem -Path $ReportsDir -Filter "report-*.csv" | ForEach-Object { $_.FullName }

if ($ReportFiles.Count -eq 0) {
    Write-Warning "No report files (report-*.csv) found in $ReportsDir."
    Write-Host "No reports to analyze. Exiting."
    exit 0 # Or exit 1 if reports are strictly expected
}

Write-Host "Found reports to analyze:"
$ReportFiles | ForEach-Object { Write-Host "  - $_" }

Write-Host "Running analysis script..."
python3 "$AnalysisScript" $ReportFiles

$AnalysisResult = $LASTEXITCODE
if ($AnalysisResult -eq 0) {
    Write-Host "Analysis complete. All tests passed or reports were clean."
} else {
    Write-Warning "Analysis complete. Some tests failed or reports had issues. Check output above."
}

exit $AnalysisResult
