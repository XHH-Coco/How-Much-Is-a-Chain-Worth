#!/usr/bin/env bash
# Download every benchmark release into data/raw/. Nothing is committed.
# Every URL below must be the publisher's own; we never re-host.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw/{polybench,polybench26,openpoly,external}

echo "== PolyBench (39 tasks, 42,169 records) =="
echo "   landing page / API: https://doi.org/10.6084/m9.figshare.30917717"
echo "   TBD: exact per-task CSV URLs are written here on D0 once resolved."
: "${POLYBENCH_BASE:?set POLYBENCH_BASE to the platform base URL}"

echo "== PolyBench26 (8 properties, ~250k datapoints) =="
: "${POLYBENCH26_BASE:?set POLYBENCH26_BASE}"

echo "== OpenPoly (741 structures / 3,985 labels) =="
: "${OPENPOLY_BASE:?set OPENPOLY_BASE}"

echo "== polyBERT embeddings (7.3 GB, separate cache) =="
echo "   run inside DSC6020_Project_Polybert; cached under data/external/polybert/"

echo
echo "Now record what you got:"
echo "  python scripts/01_build_manifests.py --hash"
