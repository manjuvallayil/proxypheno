from __future__ import annotations

import numpy as np


def lgbm_predict(X_train, y_train, X_test, **kwargs):
    import lightgbm as lgb

    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(kwargs)
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model.predict(X_test), model


def mlp_predict(X_train, y_train, X_test, **kwargs):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    params = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_iter": 500,
        "early_stopping": True,
        "random_state": 42,
    }
    params.update(kwargs)
    model = make_pipeline(StandardScaler(), MLPRegressor(**params))
    model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float))
    return model.predict(np.asarray(X_test, dtype=float)), model
