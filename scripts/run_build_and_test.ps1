# --- Local Build and Test Script (PowerShell) ---

$ErrorActionPreference = "Stop"

Write-Host "--- Local Build and Test Script (PowerShell) ---"

$ProjectRoot = (Get-Item -Path (Join-Path $PSScriptRoot "..")).FullName
$BuildDir = Join-Path $ProjectRoot "build"
$ReportsDir = Join-Path $ProjectRoot "reports"

# Platform suffix for Windows
$PlatformSuffix = "windows"

$GeneratedReportBasename = "cten_test_report_${PlatformSuffix}.csv"
$GeneratedReportFile = Join-Path $BuildDir $GeneratedReportBasename
$TargetAnalysisReportFile = Join-Path $ReportsDir "report-${PlatformSuffix}.csv"

Write-Host "Project Root: $ProjectRoot"
Write-Host "Build Directory: $BuildDir"
Write-Host "Reports Directory: $ReportsDir"
Write-Host "Platform Suffix: $PlatformSuffix"

# Clean and create build directory
Write-Host "Cleaning and creating build directory..."
if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}
New-Item -ItemType Directory -Path $BuildDir | Out-Null

# Create reports directory if it doesn't exist
if (-not (Test-Path $ReportsDir)) {
    New-Item -ItemType Directory -Path $ReportsDir | Out-Null
}

Write-Host "Configuring CMake (using MinGW Makefiles)..."
cmake -B "$BuildDir" -S "$ProjectRoot"

Write-Host "Building tests (cten_tests)..."
cmake --build "$BuildDir" --target cten_tests --config Debug

Write-Host "Running cTensor tests via ctest..."
Push-Location "$BuildDir"
ctest -C Debug --output-on-failure
Pop-Location

# Fallback logic if ctest didn't produce the report or if direct execution is preferred
if (-not (Test-Path $GeneratedReportFile)) {
    Write-Host "ctest did not produce '$GeneratedReportFile' in $BuildDir, attempting direct execution..."
    $ExecutablePath = Join-Path $BuildDir "bin\cten_tests.exe" # Path for Debug config by default
    
    if (Test-Path $ExecutablePath) {
        Write-Host "Running $ExecutablePath directly..."
        Push-Location (Split-Path $ExecutablePath)
        & .\cten_tests.exe
        Pop-Location
        # After direct execution, the report should be in BUILD_DIR (as per cten_tests.c logic)
    } else {
        Write-Error "Error: cten_tests executable not found at $ExecutablePath"
        # Check if it's directly in BUILD_DIR (less common for CMake RUNTIME_OUTPUT_DIRECTORY)
        $ExecutablePathAlt = Join-Path $BuildDir "cten_tests.exe"
        if (Test-Path $ExecutablePathAlt) {
            Write-Host "Running $ExecutablePathAlt directly..."
            Push-Location $BuildDir
            & .\cten_tests.exe
            Pop-Location
        } else {
            Write-Error "Error: cten_tests executable also not found at $ExecutablePathAlt"
            exit 1
        }
    }
}

if (Test-Path $GeneratedReportFile) {
    Write-Host "Test report generated: $GeneratedReportFile"
    Write-Host "Copying to $TargetAnalysisReportFile for analysis..."
    Copy-Item -Path $GeneratedReportFile -Destination $TargetAnalysisReportFile -Force
    Write-Host "Build and test complete. Report ready for analysis."
} else {
    Write-Error "Error: Test report $GeneratedReportFile not found after build and test execution."
    exit 1
}
