from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    GENS_DIR,
    MUTATIONS,
    OTHER_DIR,
    PHEN1_TRAITS,
    PHEN2_TRAITS,
    PHEN4_TRAITS,
    PHENS_DIR,
)


def _read_phen(path: Path, sep: str = "\t") -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_trait_set(prefix: str, traits: dict[int, str], sep: str = "\t") -> pd.DataFrame:
    """Load a phen_* family and return one wide frame keyed on ID."""
    frames = []
    for idx, name in traits.items():
        path = PHENS_DIR / f"{prefix}.{idx}.txt"
        df = _read_phen(path, sep=sep)
        df = df.rename(columns={"yd_trait": name, "rinverse": f"{name}__w"})
        frames.append(df[["ID", name, f"{name}__w"]])
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="ID", how="outer")
    return out


def load_discovery() -> pd.DataFrame:
    """Discovery population: 3 growth traits + 16 production/conformation traits."""
    growth = load_trait_set("phen_1", PHEN1_TRAITS)
    conformation = load_trait_set("phen_4", PHEN4_TRAITS)
    return growth.merge(conformation, on="ID", how="outer")


def load_validation() -> pd.DataFrame:
    """Validation population: first-lactation production + conformation traits."""
    production = load_trait_set("phen_2", PHEN2_TRAITS)
    rename = {t: f"{t}_v" for t in PHEN2_TRAITS.values()}
    rename.update({f"{t}__w": f"{t}_v__w" for t in PHEN2_TRAITS.values()})
    production = production.rename(columns=rename)
    conformation = load_trait_set("phen_4", PHEN4_TRAITS)
    return production.merge(conformation, on="ID", how="left")


def load_calf_study() -> pd.DataFrame:
    return _read_phen(PHENS_DIR / "phen_3.1.txt", sep=r"\s+")


def load_simulated() -> pd.DataFrame:
    return _read_phen(PHENS_DIR / "phen_8.1.txt", sep=r"\s+")


def read_fam(prefix: str) -> pd.DataFrame:
    path = GENS_DIR / f"{prefix}.fam"
    cols = ["FID", "IID", "PID", "MID", "SEX", "PHENO"]
    return pd.read_csv(path, sep=r"\s+", header=None, names=cols)


def load_pedigree(name: str) -> pd.DataFrame:
    path = OTHER_DIR / name
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["animal", "sire", "dam"])
    return df


def load_gwas(name: str) -> pd.DataFrame:
    return pd.read_csv(OTHER_DIR / name, sep=r"\s+")


def mutation_bim_frame() -> pd.DataFrame:
    rows = []
    for idx, meta in MUTATIONS.items():
        rows.append(
            {
                "mutation": f"mut_{idx:02d}",
                "chr": meta["chr"],
                "pos": meta["pos"],
                "snp": meta["snp"],
                "hint": meta["hint"],
                "prefix": f"gen_2.{idx}",
            }
        )
    return pd.DataFrame(rows)
