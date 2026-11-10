#!/usr/bin/env bash
# 3 properties x 5 n-levels x 5 folds x 3 seeds
# Runs in DSC6020_Project_GNN. Writes one JSON per run into results/raw_runs/.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CONDA_EXE:?activate conda first}"
echo "target env: DSC6020_Project_GNN"
echo "not implemented yet -- the arm implementations land in P1 (src/b2/arms/)."
echo "when implemented this script must:"
echo "  1. write an env fingerprint per run"
echo "  2. refuse to start if splits/ does not match manifests/sha256.txt"
echo "  3. never edit an existing run record; a re-run gets a new run_id"
