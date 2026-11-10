"""Rank correlation, used only for H1's tier ordering."""

from __future__ import annotations

from typing import Sequence


def _rank(xs: Sequence[float]):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError("need >= 3 paired observations")
    rx, ry = _rank(xs), _rank(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    if den == 0:
        raise ValueError("constant ranks; correlation undefined")
    return num / den


def h1_supported(rho: float, threshold: float = 0.5) -> bool:
    """H1 pre-registers rho >= 0.5 with no inversion spanning two or more tiers."""
    return rho >= threshold
