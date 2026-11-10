"""Lambda (chain-context recovery rate) and kappa (data-compensation rate).

    Lambda = (R2_chain - R2_unit) / (1 - R2_unit)          in [0, 1]
    kappa  = d Lambda / d log10(n)   on the OFFICIAL scaling splits

Two traps this module refuses to fall into:

1. The denominator vanishes as R2_unit -> 1. We then report None and say why,
   never a huge number that looks like a discovery.
2. Lambda is dimensionless and therefore comparable across properties -- but it is
   NOT poolable across releases (R1). Nothing here adds Lambdas from two benchmarks.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

MIN_DENOMINATOR = 1e-3


class DenominatorUnstable(ValueError):
    """R2_unit is so close to 1 that Lambda has no meaning."""


def lambda_recovery(r2_unit: float, r2_chain: float, min_denominator: float = MIN_DENOMINATOR) -> float:
    if not (0.0 <= r2_unit < 1.0):
        raise ValueError("r2_unit must be in [0, 1)")
    denominator = 1.0 - r2_unit
    if denominator < min_denominator:
        raise DenominatorUnstable(
            "R2_unit=%.6f leaves a denominator of %.2e; report delta-R2 instead" % (r2_unit, denominator)
        )
    return (r2_chain - r2_unit) / denominator


def lambda_or_none(r2_unit: float, r2_chain: float) -> Optional[float]:
    """Same as lambda_recovery but returns None instead of raising, for batch tables."""
    try:
        return lambda_recovery(r2_unit, r2_chain)
    except DenominatorUnstable:
        return None


def _ols(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two points to fit a slope")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("all x identical; slope undefined")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": slope, "intercept": intercept, "r2": r2, "n_points": n}


def kappa(ns: Sequence[int], lambdas: Sequence[Optional[float]]) -> Dict[str, float]:
    """Slope of Lambda against log10(n), on the official scaling levels only.

    Pairs where Lambda is None are dropped, and the count is reported so a reader
    can see how much of the curve survived.
    """
    if len(ns) != len(lambdas):
        raise ValueError("ns and lambdas must be the same length")
    xs, ys, dropped = [], [], 0
    for n, lam in zip(ns, lambdas):
        if lam is None:
            dropped += 1
            continue
        xs.append(math.log10(n))
        ys.append(lam)
    fit = _ols(xs, ys)
    fit["dropped"] = dropped
    fit["claim"] = "kappa ~ 0" if abs(fit["slope"]) < 1e-3 else "kappa != 0"
    return fit


def kappa_significant(fit: Dict[str, float], ci: Dict[str, float]) -> bool:
    """H2 is falsified only if kappa is significantly BELOW zero.

    Direction matters: a positive kappa would mean more data makes the gap WORSE,
    which is not what we predict, and must not be reported as support.
    """
    return ci["hi"] < 0.0
