#!/bin/bash
# Remove generated report artifacts from the example run scripts.
# Usage: bash examples/clean_reports.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="${SCRIPT_DIR}/reports"
PLOTS_DIR="${SCRIPT_DIR}/plots"

for dir in "$REPORTS_DIR" "$PLOTS_DIR"; do
    if [[ -d "$dir" ]]; then
        rm -f "$dir"/*.{csv,md,png,pdf}
        echo "Cleaned: ${dir}"
    fi
done
