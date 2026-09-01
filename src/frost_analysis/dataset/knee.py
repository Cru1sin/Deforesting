"""Global fast-to-slow knee detection for recovery temperature curves."""

from __future__ import annotations

import numpy as np


def find_global_knee(minutes: np.ndarray, temperature: np.ndarray) -> float | None:
    """Return the best continuous two-line knee over the complete curve."""
    x = np.asarray(minutes, dtype=float)
    y = np.asarray(temperature, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if len(x) < 12 or x[-1] - x[0] < 8:
        return None

    scored: list[tuple[float, float]] = []
    candidates = np.flatnonzero((x >= x[0] + 3.0) & (x <= x[-1] - 3.0))
    for position in candidates:
        knot = x[position]
        design = np.column_stack((np.ones(len(x)), x, np.maximum(0.0, x - knot)))
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        before = coefficients[1]
        after = before + coefficients[2]
        if before > 0 and after < before:
            error = float(np.square(design @ coefficients - y).sum())
            scored.append((error, float(knot)))
    return min(scored)[1] if scored else None
