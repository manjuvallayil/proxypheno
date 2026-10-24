from __future__ import annotations

import numpy as np
import pandas as pd


def trait_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "ID" and not c.endswith("__w")]


def missingness(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    n = len(df)
    out = pd.DataFrame(
        {
            "trait": cols,
            "n_present": [int(df[c].notna().sum()) for c in cols],
            "pct_missing": [float(df[c].isna().mean() * 100) for c in cols],
        }
    )
    out["n_total"] = n
    return out.sort_values("pct_missing", ascending=False, ignore_index=True)


def summarise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            rows.append({"trait": c, "n": 0})
            continue
        rows.append(
            {
                "trait": c,
                "n": int(s.size),
                "mean": float(s.mean()),
                "sd": float(s.std()),
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "median": float(s.median()),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def impute_median(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].fillna(out[c].median())
    return out


def robust_scale(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        med = out[c].median()
        iqr = out[c].quantile(0.75) - out[c].quantile(0.25)
        if not np.isfinite(iqr) or iqr == 0:
            iqr = out[c].std()
        if not np.isfinite(iqr) or iqr == 0:
            iqr = 1.0
        out[c] = (out[c] - med) / iqr
    return out


def outlier_flags(df: pd.DataFrame, cols: list[str], z: float = 5.0) -> pd.DataFrame:
    scaled = robust_scale(df, cols)
    flags = scaled[cols].abs() > z
    flags.insert(0, "ID", df["ID"].to_numpy())
    return flags
