import numpy as np
import pandas as pd

from proxypheno.features import conditional_residuals
from proxypheno.models.novelty import NoveltyModel, signature_groups


def test_conditional_residuals_shape():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    residuals = conditional_residuals(X, n_splits=3)
    assert residuals.shape == X.shape
    assert not residuals.isna().any().any()


def test_novelty_model_scores():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6))
    model = NoveltyModel(n_components=3).fit(X)
    scores = model.score(X)
    assert {"pca_mahalanobis", "iforest", "novelty_score"} <= set(scores.columns)
    assert scores["novelty_score"].between(0, 1).all()
    assert model.transform(X).shape == (500, 3)


def test_signature_groups():
    R = pd.DataFrame(np.zeros((3, 6)), columns=list("abcdef"))
    R.iloc[0, 0] = 5.0
    R.iloc[1, 1] = -4.0
    R.iloc[2, :] = 0.5
    signatures = signature_groups(R, z_threshold=3.0)
    assert signatures.loc[0, "signature"] == "high_a"
    assert signatures.loc[1, "signature"] == "low_b"
    assert signatures.loc[2, "signature"] == "diffuse"
