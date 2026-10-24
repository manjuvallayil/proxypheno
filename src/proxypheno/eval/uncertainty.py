from __future__ import annotations

import numpy as np


def conformal_quantile(residuals, alpha: float = 0.1) -> float:
    """Split-conformal half-width for a 1 - alpha interval."""
    return float(np.quantile(np.abs(np.asarray(residuals, dtype=float)), 1 - alpha))


def conformal_intervals(pred, half_width: float):
    pred = np.asarray(pred, dtype=float)
    return pred - half_width, pred + half_width


def coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def bootstrap_mean_ci(values, n_boot: int = 2000, alpha: float = 0.05, random_state: int = 42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_state)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def bootstrap_odds_ratio_ci(
    a: int,
    n: int,
    b: int,
    m: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
):
    """Bootstrap CI for the odds ratio of a 2x2 table.

    ``a`` carriers out of ``n`` in the group; ``b`` carriers out of ``m`` in
    the rest.
    """
    rng = np.random.default_rng(random_state)
    group = np.r_[np.ones(a), np.zeros(n - a)]
    rest = np.r_[np.ones(b), np.zeros(m - b)]
    odds = []
    for _ in range(n_boot):
        ga = rng.choice(group, n, replace=True).sum()
        gr = rng.choice(rest, m, replace=True).sum()
        # Haldane correction for zero cells.
        odds.append(((ga + 0.5) / (n - ga + 0.5)) / ((gr + 0.5) / (m - gr + 0.5)))
    return float(np.quantile(odds, alpha / 2)), float(np.quantile(odds, 1 - alpha / 2))
