import numpy as np
import pandas as pd

from proxypheno.models.genotypes import bh_fdr, tail_enrichment


def test_tail_enrichment_detects_enriched_tail():
    trait = np.r_[np.zeros(90), -10 * np.ones(10)]
    carrier = np.r_[np.zeros(90), np.ones(10)].astype(bool)
    df = pd.DataFrame({"trait": trait, "carrier": carrier, "hom_alt": carrier})
    out = tail_enrichment(df, "trait", tails=(0.1,))
    row = out.iloc[0]
    assert row["tail"] == "low"
    assert row["odds_ratio"] > 1
    assert row["p_value"] < 0.01


def test_bh_fdr_is_monotone():
    q = bh_fdr([0.01, 0.02, 0.5])
    assert q[0] <= q[1] <= q[2]
    assert np.all((q >= 0) & (q <= 1))
