#!/usr/bin/env bash
# Reproduces the paper's results from the data in ./data. Run from this folder: bash run_all.sh
# Use another interpreter with: PYTHON=/path/to/python bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-python3}"

echo "== Step 1: peer-review constraint (SC3) and selection counts"
"$PYTHON" 1_selection/sc3_peer_review_filter.py
"$PYTHON" 1_selection/selection_log.py

echo "== Step 2: tables, figures and generated text for Sections 5-6 and the appendix"
"$PYTHON" 2_results/rq_tables_and_figures.py
echo "Done. Results are in ./outputs"
