from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from proxypheno.config import ARTIFACTS_DIR, DATA_ROOT, GENS_DIR, OTHER_DIR, PHENS_DIR
from proxypheno.io import load_calf_study, load_discovery, load_validation
from proxypheno.qc import missingness, summarise, trait_columns


def main() -> None:
    print(f"Data root: {DATA_ROOT}")
    print(f"Phens dir: {PHENS_DIR}")
    print(f"Gens dir:  {GENS_DIR} ({'present' if GENS_DIR.exists() else 'missing'})")
    print(f"Other dir: {OTHER_DIR} ({'present' if OTHER_DIR.exists() else 'missing'})")

    discovery = load_discovery()
    validation = load_validation()
    traits = trait_columns(discovery)

    print(f"\nDiscovery: {discovery.shape[0]:,} animals x {len(traits)} traits")
    print(f"Validation: {validation.shape[0]:,} animals x {len(trait_columns(validation))} traits")

    miss = missingness(discovery, traits)
    print("\nMissingness (top 5):")
    print(miss.head(5).to_string(index=False))

    summary = summarise(discovery, traits)
    print("\nTrait summary (first 5):")
    print(summary.head(5).to_string(index=False))

    calf = load_calf_study()
    by_variant = (
        calf.groupby("VARIANT")["weight"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    control = by_variant.loc[by_variant["VARIANT"] == "CONTROL", "mean"]
    if not control.empty:
        by_variant["pct_vs_control"] = (by_variant["mean"] / float(control.iloc[0]) - 1) * 100
    print("\nCalf study positive control:")
    print(by_variant.to_string(index=False))

    payload = {
        "data_root": str(DATA_ROOT),
        "discovery": {"n": len(discovery), "n_traits": len(traits)},
        "validation": {"n": len(validation), "n_traits": len(trait_columns(validation))},
        "missingness": miss.to_dict(orient="records"),
        "trait_summary": summary.to_dict(orient="records"),
        "calf_positive_control": by_variant.to_dict(orient="records"),
        "genotype_files": {
            "gen_1": (GENS_DIR / "gen_1.1.bed").exists(),
            "gen_2": (GENS_DIR / "gen_2.1.bed").exists(),
            "gen_3": (GENS_DIR / "gen_3.1.bed").exists(),
            "gen_11": (GENS_DIR / "gen_11.1.bed").exists(),
            "gen_12": (GENS_DIR / "gen_12.1.bed").exists(),
        },
        "pedigree_files": {
            "other_3": (OTHER_DIR / "other_3.txt").exists(),
            "other_4": (OTHER_DIR / "other_4.txt").exists(),
        },
    }
    out = ARTIFACTS_DIR / "audit.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
