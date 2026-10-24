from __future__ import annotations

from ..features import conditional_residuals


def ridge_proxy_residuals(X, alpha: float = 1.0, n_splits: int = 5, random_state: int = 42):
    """Linear baseline: cross-fitted Ridge residual for each trait."""
    return conditional_residuals(X, alpha=alpha, n_splits=n_splits, random_state=random_state)
