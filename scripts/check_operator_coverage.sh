#!/bin/bash
set -e

echo "--- Local Operator Test Coverage Check Script (Bash) ---"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COVERAGE_SCRIPT="${PROJECT_ROOT}/.github/check_operator_coverage.py"

echo "Project Root: ${PROJECT_ROOT}"
echo "Coverage Script: ${COVERAGE_SCRIPT}"

if [ ! -f "${COVERAGE_SCRIPT}" ]; then
    echo "Error: Coverage script '${COVERAGE_SCRIPT}' not found." >&2
    exit 1
fi

echo "Running operator test coverage check..."
python3 "${COVERAGE_SCRIPT}"

COVERAGE_RESULT=$?
if [ $COVERAGE_RESULT -eq 0 ]; then
    echo "Coverage check complete. All operators seem to have test files."
else
    echo "Coverage check complete. Some operators may be missing test files. Check output above." >&2
fi

exit $COVERAGE_RESULT