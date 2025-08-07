#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Local Build and Test Script (Bash) ---"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
REPORTS_DIR="${PROJECT_ROOT}/reports"

# Determine platform for report naming (matches CMake logic)
PLATFORM_SUFFIX="unknown"
if [[ "$(uname)" == "Linux" ]]; then
    PLATFORM_SUFFIX="linux"
elif [[ "$(uname)" == "Darwin" ]]; then
    PLATFORM_SUFFIX="macos"
fi

GENERATED_REPORT_BASENAME="cten_test_report_${PLATFORM_SUFFIX}.csv"
GENERATED_REPORT_FILE="${BUILD_DIR}/${GENERATED_REPORT_BASENAME}"
TARGET_ANALYSIS_REPORT_FILE="${REPORTS_DIR}/report-${PLATFORM_SUFFIX}.csv"

echo "Project Root: ${PROJECT_ROOT}"
echo "Build Directory: ${BUILD_DIR}"
echo "Reports Directory: ${REPORTS_DIR}"
echo "Platform Suffix: ${PLATFORM_SUFFIX}"

# Clean and create build directory
echo "Cleaning and creating build directory..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Create reports directory if it doesn't exist
mkdir -p "${REPORTS_DIR}"

echo "Configuring CMake..."
cmake -B "${BUILD_DIR}" -S "${PROJECT_ROOT}"

echo "Building tests (cten_tests)..."
cmake --build "${BUILD_DIR}" --target cten_tests --config Debug

echo "Running cTensor tests via ctest..."
cd "${BUILD_DIR}"
ctest -C Debug --output-on-failure
cd "${PROJECT_ROOT}"

# Fallback logic if ctest didn't produce the report or if direct execution is preferred
if [ ! -f "${GENERATED_REPORT_FILE}" ]; then
  echo "ctest did not produce '${GENERATED_REPORT_FILE}' in ${BUILD_DIR}, attempting direct execution..."
  EXECUTABLE_PATH="${BUILD_DIR}/bin/cten_tests"
  if [ -f "${EXECUTABLE_PATH}" ]; then
    echo "Running ${EXECUTABLE_PATH} directly..."
    cd "${BUILD_DIR}/bin"
    ./cten_tests
    cd "${PROJECT_ROOT}"
    # After direct execution, the report should be in BUILD_DIR (as per cten_tests.c logic)
  else
    echo "Error: cten_tests executable not found at ${EXECUTABLE_PATH}" >&2
    # Check if it's directly in BUILD_DIR (less common for CMake RUNTIME_OUTPUT_DIRECTORY)
    EXECUTABLE_PATH_ALT="${BUILD_DIR}/cten_tests"
    if [ -f "${EXECUTABLE_PATH_ALT}" ]; then
        echo "Running ${EXECUTABLE_PATH_ALT} directly..."
        cd "${BUILD_DIR}"
        ./cten_tests
        cd "${PROJECT_ROOT}"
    else
        echo "Error: cten_tests executable also not found at ${EXECUTABLE_PATH_ALT}" >&2
        exit 1
    fi
  fi
fi

if [ -f "${GENERATED_REPORT_FILE}" ]; then
  echo "Test report generated: ${GENERATED_REPORT_FILE}"
  echo "Copying to ${TARGET_ANALYSIS_REPORT_FILE} for analysis..."
  cp "${GENERATED_REPORT_FILE}" "${TARGET_ANALYSIS_REPORT_FILE}"
  echo "Build and test complete. Report ready for analysis."
else
  echo "Error: Test report ${GENERATED_REPORT_FILE} not found after build and test execution." >&2
  exit 1
fi