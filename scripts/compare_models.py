from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from proxypheno.config import ARTIFACTS_DIR, FIGURES_DIR
from proxypheno.io import load_discovery
from proxypheno.models.ml import lgbm_predict, mlp_predict
from proxypheno.qc import robust_scale, trait_columns

NOVELTY_EXCLUDE = {"adaptability_milking", "shed_temperament", "farmer_opinion"}
TEST_SIZE = 0.2
RANDOM_STATE = 42


def impute_traits(df: pd.DataFrame, traits: list[str]) -> pd.DataFrame:
    values = df[traits].astype(float)
    imputer = IterativeImputer(max_iter=5, random_state=RANDOM_STATE, initial_strategy="median")
    imputed = imputer.fit_transform(values)
    return pd.DataFrame(imputed, columns=traits, index=df.index)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    discovery = load_discovery()
    all_traits = trait_columns(discovery)
    traits = [t for t in all_traits if t not in NOVELTY_EXCLUDE]
    scaled = robust_scale(impute_traits(discovery, all_traits), all_traits)[traits]

    train_idx, test_idx = train_test_split(
        np.arange(len(scaled)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    rows = []
    for trait in traits:
        others = [t for t in traits if t != trait]
        X = scaled[others].to_numpy()
        y = scaled[trait].to_numpy()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        candidates = {
            "ridge": lambda Xtr=X_train, ytr=y_train, Xte=X_test: (
                Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte),
                None,
            ),
            "lightgbm": lambda Xtr=X_train, ytr=y_train, Xte=X_test: lgbm_predict(Xtr, ytr, Xte),
            "mlp": lambda Xtr=X_train, ytr=y_train, Xte=X_test: mlp_predict(Xtr, ytr, Xte),
        }
        for name, fit in candidates.items():
            start = time.perf_counter()
            pred, _ = fit()
            elapsed = time.perf_counter() - start
            row = {"trait": trait, "model": name, "seconds": elapsed}
            row.update(metrics(y_test, pred))
            rows.append(row)
            print(
                f"{trait:24s} {name:9s} R2={row['r2']:.4f} "
                f"RMSE={row['rmse']:.4f} ({elapsed:.1f}s)"
            )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("model")
        .agg(
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            total_seconds=("seconds", "sum"),
        )
        .reset_index()
        .sort_values("mean_r2", ascending=False)
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_animals": len(scaled),
        "n_traits": len(traits),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "summary": summary.to_dict(orient="records"),
        "per_trait": results.to_dict(orient="records"),
        "note": (
            "Single 80/20 hold-out. Ridge and the ML models use the same split. "
            "Traits are robust-scaled; R2 is on the scaled trait."
        ),
    }
    (ARTIFACTS_DIR / "model_comparison.json").write_text(json.dumps(payload, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary["model"], summary["mean_r2"], color="#1f6f5c")
    ax.set_ylabel("mean R2 across traits")
    ax.set_title("Proxy-prediction: linear baseline vs ML")
    ax.set_ylim(0, max(0.05, float(summary["mean_r2"].max()) * 1.2))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)

    print("\n" + summary.to_string(index=False))
    print(f"\nWrote {ARTIFACTS_DIR / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
