from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from ..config import GENS_DIR


def _open_bed(prefix: str):
    from bed_reader import open_bed

    return open_bed(str(GENS_DIR / f"{prefix}.bed"))


def read_dosages(prefix: str, snp_indices=None):
    """Return (iid, sid, dosage matrix) for a PLINK prefix."""
    bed = _open_bed(prefix)
    iid = np.asarray(bed.iid)
    sid = np.asarray(bed.sid)
    if snp_indices is None:
        snp_indices = np.arange(bed.sid_count)
    values = bed.read(index=np.s_[:, snp_indices], dtype="float32")
    return iid, sid, values


def carrier_status(prefix: str, snp_index: int = 0) -> pd.DataFrame:
    iid, sid, values = read_dosages(prefix, [snp_index])
    dosage = np.nan_to_num(values[:, 0], nan=0.0)
    return pd.DataFrame(
        {
            "ID": np.asarray(iid).astype(str),
            "snp": str(sid[snp_index]),
            "dosage": dosage,
            "carrier": dosage > 0,
            "hom_alt": dosage >= 2,
        }
    )


def region_burden(prefix: str) -> pd.DataFrame:
    """Sum of alt-allele dosages across a genotype region, per animal."""
    iid, _, values = read_dosages(prefix)
    burden = np.nansum(values, axis=1)
    return pd.DataFrame({"ID": np.asarray(iid).astype(str), "region_burden": burden})


def _fisher(a: int, n: int, b: int, m: int) -> tuple[float, float]:
    if n == 0 or m == 0:
        return float("nan"), float("nan")
    try:
        odds_ratio, p_value = fisher_exact([[a, n - a], [b, m - b]])
    except ValueError:
        return float("nan"), float("nan")
    return float(odds_ratio), float(p_value)


def _enrichment_row(
    df: pd.DataFrame, mask: np.ndarray, carrier_col: str, hom_col: str | None
) -> dict:
    n = int(mask.sum())
    m = int((~mask).sum())
    carriers = int(df.loc[mask, carrier_col].sum())
    rest_carriers = int(df.loc[~mask, carrier_col].sum())
    odds_ratio, p_value = _fisher(carriers, n, rest_carriers, m)
    row = {
        "n": n,
        "carriers": carriers,
        "carrier_rate": carriers / n if n else float("nan"),
        "rest_carriers": rest_carriers,
        "rest_n": m,
        "rest_rate": rest_carriers / m if m else float("nan"),
        "odds_ratio": odds_ratio,
        "p_value": p_value,
    }
    if hom_col is not None:
        homs = int(df.loc[mask, hom_col].sum())
        rest_homs = int(df.loc[~mask, hom_col].sum())
        hom_or, hom_p = _fisher(homs, n, rest_homs, m)
        row.update(
            {
                "hom_carriers": homs,
                "hom_rate": homs / n if n else float("nan"),
                "hom_odds_ratio": hom_or,
                "hom_p_value": hom_p,
            }
        )
    return row


def tail_enrichment(
    df: pd.DataFrame,
    trait: str,
    tails: tuple[float, ...] = (0.025, 0.975),
    carrier_col: str = "carrier",
    hom_col: str | None = "hom_alt",
) -> pd.DataFrame:
    """Carrier enrichment in the low and high tails of one residual trait."""
    rows = []
    for quantile in tails:
        tail = "low" if quantile < 0.5 else "high"
        threshold = float(df[trait].quantile(quantile))
        mask = (df[trait] <= threshold) if tail == "low" else (df[trait] >= threshold)
        row = {"trait": trait, "tail": tail, "threshold": threshold}
        row.update(_enrichment_row(df, mask.to_numpy(), carrier_col, hom_col))
        rows.append(row)
    return pd.DataFrame(rows)


def group_enrichment(
    df: pd.DataFrame,
    group_col: str,
    carrier_col: str = "carrier",
    hom_col: str | None = "hom_alt",
) -> pd.DataFrame:
    """Carrier enrichment for each named group versus the rest."""
    rows = []
    for group, index in df.groupby(group_col).groups.items():
        mask = df.index.isin(index)
        row = {"group": group}
        row.update(_enrichment_row(df, mask, carrier_col, hom_col))
        rows.append(row)
    return pd.DataFrame(rows)


def bh_fdr(p_values) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(pv)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_valid = np.empty_like(adjusted)
    q_valid[order] = np.clip(adjusted, 0, 1)
    q[valid] = q_valid
    return q
