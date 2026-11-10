#!/usr/bin/env bash
# Five figures, generated from results/aggregate only.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "figures planned:"
echo "  1 Lambda heatmap (property x arm, with CI)"
echo "  2 Lambda vs tier ordering (with the dielectric bet marked)"
echo "  3 scaling curves, Lambda vs log n, with kappa and its CI"
echo "  4 OpenPoly crossover scan"
echo "  5 main table, 8 tasks x arms, with the honest gap to the leaderboard"
echo
echo "not implemented yet -- matplotlib is not installed (see environment/README.md)."
