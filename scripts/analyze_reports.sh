#!/bin/bash
set -e

echo "--- Local Test Report Analysis Script (Bash) ---"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="${PROJECT_ROOT}/reports"
ANALYSIS_SCRIPT="${PROJECT_ROOT}/.github/check_all_results.py"

echo "Project Root: ${PROJECT_ROOT}"
echo "Reports Directory: ${REPORTS_DIR}"
echo "Analysis Script: ${ANALYSIS_SCRIPT}"

if [ ! -d "${REPORTS_DIR}" ]; then
    echo "Error: Reports directory '${REPORTS_DIR}' not found. Run the build and test script first." >&2
    exit 1
fi

if [ ! -f "${ANALYSIS_SCRIPT}" ]; then
    echo "Error: Analysis script '${ANALYSIS_SCRIPT}' not found." >&2
    exit 1
fi

# Find all report-*.csv files in the reports directory
REPORT_FILES=($(find "${REPORTS_DIR}" -name "report-*.csv"))

if [ ${#REPORT_FILES[@]} -eq 0 ]; then
    echo "No report files (report-*.csv) found in ${REPORTS_DIR}." >&2
    # We can choose to exit with 0 if no reports mean nothing to check, or 1 if reports are expected.
    # For CI alignment, if a build ran, a report should exist.
    # However, for local use, maybe no builds ran yet. Let's allow it for now.
    echo "No reports to analyze. Exiting."
    exit 0
fi

echo "Found reports to analyze:"
for report_file in "${REPORT_FILES[@]}"; do
    echo "  - ${report_file}"
done

echo "Running analysis script..."
python3 "${ANALYSIS_SCRIPT}" "${REPORT_FILES[@]}"

ANALYSIS_RESULT=$?
if [ $ANALYSIS_RESULT -eq 0 ]; then
    echo "Analysis complete. All tests passed or reports were clean."
else
    echo "Analysis complete. Some tests failed or reports had issues. Check output above." >&2
fi

exit $ANALYSIS_RESULT