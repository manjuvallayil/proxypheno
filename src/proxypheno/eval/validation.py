from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from ..models.novelty import NoveltyModel


def split_by_id(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    ids = df["ID"].unique()
    train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=random_state)
    return df[df["ID"].isin(train_ids)].copy(), df[df["ID"].isin(test_ids)].copy()


def bootstrap_novelty_stability(
    X,
    n_boot: int = 10,
    subsample: float = 0.8,
    top_quantile: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Refit the novelty model on subsamples and compare held-out rankings.

    Returns the mean Spearman correlation of held-out novelty scores and the
    mean Jaccard overlap of the top-5% set. These are the stability checks that
    matter for a ranking-based novelty engine.
    """
    values = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    n = len(values)

    full_model = NoveltyModel().fit(values)
    full_scores = full_model.score(values)["novelty_score"].to_numpy()

    rhos: list[float] = []
    jaccards: list[float] = []
    for _ in range(n_boot):
        train = rng.choice(n, int(n * subsample), replace=False)
        holdout = np.setdiff1d(np.arange(n), train)
        if len(holdout) < 100:
            continue
        model = NoveltyModel().fit(values[train])
        boot_scores = model.score(values[holdout])["novelty_score"].to_numpy()

        rho = spearmanr(full_scores[holdout], boot_scores).statistic
        if np.isfinite(rho):
            rhos.append(float(rho))

        full_top = holdout[full_scores[holdout] >= np.quantile(full_scores[holdout], top_quantile)]
        boot_top = holdout[boot_scores >= np.quantile(boot_scores, top_quantile)]
        union = set(full_top) | set(boot_top)
        if union:
            jaccards.append(len(set(full_top) & set(boot_top)) / len(union))

    return {
        "mean_spearman": float(np.mean(rhos)) if rhos else float("nan"),
        "sd_spearman": float(np.std(rhos)) if rhos else float("nan"),
        "mean_top5_jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
        "n_boot": len(rhos),
    }
