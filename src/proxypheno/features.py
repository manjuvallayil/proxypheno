from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def conditional_residuals(
    X: pd.DataFrame, alpha: float = 1.0, n_splits: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Cross-fitted residual of each trait conditional on all other traits.

    This is the proxy-phenotype residual: how unexpected a trait is given the
    rest of the animal's phenotype profile.
    """
    values = np.asarray(X, dtype=float)
    _, p = values.shape
    residuals = np.full_like(values, np.nan)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for j in range(p):
        y = values[:, j]
        others = [k for k in range(p) if k != j]
        X_others = values[:, others]
        for train_idx, test_idx in folds.split(X_others):
            model = Ridge(alpha=alpha)
            model.fit(X_others[train_idx], y[train_idx])
            residuals[test_idx, j] = y[test_idx] - model.predict(X_others[test_idx])

    return pd.DataFrame(residuals, index=X.index, columns=X.columns)


def pca_embedding(
    X: pd.DataFrame, n_components: float = 0.95, random_state: int = 42
) -> tuple[np.ndarray, PCA]:
    model = PCA(n_components=n_components, random_state=random_state)
    scores = model.fit_transform(np.asarray(X, dtype=float))
    return scores, model
