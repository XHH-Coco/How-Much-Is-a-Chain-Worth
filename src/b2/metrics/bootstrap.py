"""Percentile bootstrap. Stdlib only, so the CI can run it before any install.

PolyBench gives us ONE official split, so uncertainty cannot come from k-fold. It
comes from a bootstrap over test residuals plus 3 model seeds (splits/README.md).
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile; q in [0, 1]."""
    if not values:
        raise ValueError("empty sample")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def bootstrap_ci(
    sample: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    n_resamples: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    if not sample:
        raise ValueError("empty sample")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    rng = random.Random(seed)
    n = len(sample)
    stats: List[float] = []
    for _ in range(n_resamples):
        draw = [sample[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(draw))
    alpha = (1.0 - level) / 2.0
    return {
        "point": float(statistic(sample)),
        "lo": percentile(stats, alpha),
        "hi": percentile(stats, 1.0 - alpha),
        "n_resamples": n_resamples,
        "level": level,
        "seed": seed,
    }


def ci_excludes_zero(ci: Dict[str, float]) -> bool:
    """True when the interval does not straddle zero -- the only case we call a finding."""
    return ci["lo"] > 0.0 or ci["hi"] < 0.0
