from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


@dataclass
class NoveltyModel:
    """PCA-space Mahalanobis + IsolationForest novelty score.

    The residual phenotype space in this dataset is a single, approximately
    Gaussian continuum: density clustering finds no stable clusters. Novelty is
    therefore treated as a ranking problem, and interpretable signatures are
    derived from the dominant residual trait (see ``signature_groups``).
    """

    n_components: float | int = 0.95
    random_state: int = 42
    contamination: float | str = "auto"
    pca: PCA | None = field(default=None, init=False)
    iforest: IsolationForest | None = field(default=None, init=False)
    mean_: np.ndarray | None = field(default=None, init=False)
    cov_inv_: np.ndarray | None = field(default=None, init=False)

    def fit(self, X) -> NoveltyModel:
        values = np.asarray(X, dtype=float)
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        scores = self.pca.fit_transform(values)
        self.iforest = IsolationForest(
            random_state=self.random_state, contamination=self.contamination
        )
        self.iforest.fit(scores)
        self.mean_ = scores.mean(axis=0)
        cov = np.cov(scores, rowvar=False)
        self.cov_inv_ = np.linalg.pinv(cov)
        return self

    def score(self, X, index=None) -> pd.DataFrame:
        if self.pca is None or self.iforest is None:
            raise RuntimeError("Call fit() before score().")
        values = np.asarray(X, dtype=float)
        scores = self.pca.transform(values)
        diff = scores - self.mean_
        mahalanobis = np.sqrt(np.einsum("ij,jk,ik->i", diff, self.cov_inv_, diff))
        iso = -self.iforest.score_samples(scores)
        out = pd.DataFrame(
            {"pca_mahalanobis": mahalanobis, "iforest": iso}, index=index
        )
        out["novelty_score"] = out.rank(pct=True).mean(axis=1)
        return out

    def transform(self, X) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("Call fit() before transform().")
        return self.pca.transform(np.asarray(X, dtype=float))


def signature_groups(
    residual_z: pd.DataFrame, z_threshold: float = 3.0
) -> pd.DataFrame:
    """Assign each animal to an interpretable residual signature.

    The signature is the trait with the largest absolute residual and the
    direction of that residual (``low_`` or ``high_``). Animals below the
    threshold are ``diffuse``.
    """
    values = residual_z.to_numpy(dtype=float)
    abs_values = np.abs(values)
    dominant_idx = abs_values.argmax(axis=1)
    rows = np.arange(len(values))
    max_abs = abs_values[rows, dominant_idx]
    signed = values[rows, dominant_idx]
    traits = residual_z.columns.to_numpy()
    dominant = traits[dominant_idx]
    tail = np.where(signed < 0, "low", "high")
    signature = [
        f"{t}_{d}" if m >= z_threshold else "diffuse"
        for t, d, m in zip(tail, dominant, max_abs)
    ]
    return pd.DataFrame(
        {
            "signature": signature,
            "dominant_trait": dominant,
            "tail": tail,
            "max_abs_z": max_abs,
        },
        index=residual_z.index,
    )
