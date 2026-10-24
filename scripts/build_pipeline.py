from __future__ import annotations

import json
import sys
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

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from proxypheno.config import ARTIFACTS_DIR, FIGURES_DIR, GENS_DIR, MUTATIONS
from proxypheno.eval.uncertainty import bootstrap_mean_ci
from proxypheno.eval.validation import bootstrap_novelty_stability
from proxypheno.features import conditional_residuals
from proxypheno.genai.precompute_briefs import precompute_briefs
from proxypheno.io import load_calf_study, load_discovery
from proxypheno.models.genotypes import (
    bh_fdr,
    carrier_status,
    group_enrichment,
    tail_enrichment,
)
from proxypheno.models.novelty import NoveltyModel, signature_groups
from proxypheno.qc import robust_scale, trait_columns

NOVELTY_EXCLUDE = {"adaptability_milking", "shed_temperament", "farmer_opinion"}
Z_THRESHOLD = 3.0
TOP_QUANTILE = 0.95
MIN_SIGNATURE_N = 30


def impute_traits(df: pd.DataFrame, traits: list[str]) -> pd.DataFrame:
    values = df[traits].astype(float)
    imputer = IterativeImputer(
        max_iter=5, random_state=42, initial_strategy="median"
    )
    imputed = imputer.fit_transform(values)
    return pd.DataFrame(imputed, columns=traits, index=df.index)


def build_signature_summary(
    profiles: pd.DataFrame, residual_z: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for signature, group in profiles.groupby("signature"):
        if signature == "diffuse" or len(group) < MIN_SIGNATURE_N:
            continue
        mean_z = residual_z.loc[group.index].mean()
        top = mean_z.reindex(mean_z.abs().sort_values(ascending=False).index).head(3)
        row = {
            "signature": signature,
            "n": len(group),
            "novelty_mean": float(group["novelty_score"].mean()),
            "max_abs_z_mean": float(group["max_abs_z"].mean()),
            "top_traits": ", ".join(top.index),
        }
        for rank, (trait, value) in enumerate(top.items(), start=1):
            row[f"top_trait_{rank}"] = trait
            row[f"top_z_{rank}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)


def positive_control() -> tuple[pd.DataFrame, dict]:
    calf = load_calf_study()
    table = (
        calf.groupby("VARIANT")["weight"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    control_mean = table.loc[table["VARIANT"] == "CONTROL", "mean"]
    if not control_mean.empty:
        table["pct_vs_control"] = (table["mean"] / float(control_mean.iloc[0]) - 1) * 100

    summary: dict = {}
    plcd4 = calf.loc[calf["VARIANT"] == "PLCD4", "weight"].to_numpy()
    control = calf.loc[calf["VARIANT"] == "CONTROL", "weight"].to_numpy()
    if len(plcd4) and len(control):
        rng = np.random.default_rng(42)
        diffs = [
            rng.choice(plcd4, len(plcd4)).mean() - rng.choice(control, len(control)).mean()
            for _ in range(2000)
        ]
        low, high = np.quantile(diffs, [0.025, 0.975])
        summary = {
            "variant": "PLCD4",
            "n_carriers": len(plcd4),
            "n_controls": len(control),
            "carrier_mean_weight": float(plcd4.mean()),
            "control_mean_weight": float(control.mean()),
            "mean_difference": float(plcd4.mean() - control.mean()),
            "difference_ci_low": float(low),
            "difference_ci_high": float(high),
            "pct_vs_control": float((plcd4.mean() / control.mean() - 1) * 100),
            "control_mean_ci": bootstrap_mean_ci(control),
        }
    return table, summary


def genotype_analysis(
    profiles: pd.DataFrame, novelty_traits: list[str]
) -> pd.DataFrame:
    tail_frames = []
    signature_frames = []
    novelty_frames = []

    z_cols = [f"{trait}__z" for trait in novelty_traits]
    base = profiles[["ID", "signature", "novelty_score", *z_cols]].copy()

    for idx, meta in MUTATIONS.items():
        prefix = f"gen_3.{idx}"
        if not (GENS_DIR / f"{prefix}.bed").exists():
            continue
        status = carrier_status(prefix)
        merged = base.merge(status[["ID", "carrier", "hom_alt"]], on="ID", how="inner")
        if merged["carrier"].nunique() < 2:
            continue

        for trait in novelty_traits:
            table = tail_enrichment(merged, f"{trait}__z")
            table["trait"] = trait
            table["analysis"] = "tail"
            table["mutation"] = f"mut_{idx:02d}"
            table["hint"] = meta["hint"]
            tail_frames.append(table)

        table = group_enrichment(merged, "signature")
        table = table[table["group"] != "diffuse"]
        table = table[table["n"] >= MIN_SIGNATURE_N]
        table["trait"] = None
        table["analysis"] = "signature"
        table["mutation"] = f"mut_{idx:02d}"
        table["hint"] = meta["hint"]
        signature_frames.append(table)

        threshold = merged["novelty_score"].quantile(TOP_QUANTILE)
        merged["novelty_group"] = np.where(
            merged["novelty_score"] >= threshold, "top5_novelty", "rest"
        )
        table = group_enrichment(merged, "novelty_group")
        table = table[table["group"] == "top5_novelty"].copy()
        table["trait"] = None
        table["analysis"] = "novelty"
        table["mutation"] = f"mut_{idx:02d}"
        table["hint"] = meta["hint"]
        novelty_frames.append(table)

    frames = tail_frames + signature_frames + novelty_frames
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["q_value"] = np.nan
    out["hom_q_value"] = np.nan
    for index in out.groupby("analysis").groups.values():
        out.loc[index, "q_value"] = bh_fdr(out.loc[index, "p_value"])
        out.loc[index, "hom_q_value"] = bh_fdr(out.loc[index, "hom_p_value"])
    return out.sort_values(["analysis", "q_value"], ignore_index=True)


def save_figures(
    profiles: pd.DataFrame,
    signature_summary: pd.DataFrame,
    control: pd.DataFrame,
    enrichment: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        profiles["pc1"],
        profiles["pc2"],
        c=profiles["novelty_score"],
        s=2,
        cmap="viridis",
        alpha=0.5,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Residual phenotype space, coloured by novelty")
    fig.colorbar(scatter, ax=ax, label="novelty score")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_novelty.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(profiles["novelty_score"], bins=60, color="#3b6ea5")
    ax.set_xlabel("novelty score")
    ax.set_ylabel("animals")
    ax.set_title("Novelty distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "novelty_hist.png", dpi=160)
    plt.close(fig)

    if not signature_summary.empty:
        top = signature_summary.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top["signature"], top["n"], color="#1f6f5c")
        ax.set_xlabel("animals")
        ax.set_title("Largest residual signatures")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "signature_counts.png", dpi=160)
        plt.close(fig)

    if not control.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(control["VARIANT"], control["mean"], color="#3b6ea5")
        ax.set_ylabel("mean weight (kg)")
        ax.set_title("Calf study: carrier weight")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "calf_positive_control.png", dpi=160)
        plt.close(fig)

    tail = enrichment[enrichment["analysis"] == "tail"] if not enrichment.empty else pd.DataFrame()
    if not tail.empty:
        tail = tail.copy()
        tail["label"] = tail["trait"] + "_" + tail["tail"]
        pivot = tail.pivot_table(
            index="mutation", columns="label", values="q_value", aggfunc="min"
        )
        values = -np.log10(pivot.clip(lower=1e-12).to_numpy())
        fig, ax = plt.subplots(figsize=(12, 5))
        image = ax.imshow(values, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
        ax.set_title("Tail enrichment, -log10(q) for carrier status")
        fig.colorbar(image, ax=ax, label="-log10(q)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "tail_enrichment.png", dpi=160)
        plt.close(fig)


def main() -> None:
    discovery = load_discovery()
    all_traits = trait_columns(discovery)
    novelty_traits = [t for t in all_traits if t not in NOVELTY_EXCLUDE]
    print(
        f"Loaded {len(discovery):,} animals, {len(all_traits)} traits; "
        f"{len(novelty_traits)} used for novelty"
    )

    imputed = impute_traits(discovery, all_traits)
    scaled = robust_scale(imputed, all_traits)

    residual = conditional_residuals(scaled[novelty_traits])
    residual_z = robust_scale(residual, novelty_traits)

    model = NoveltyModel().fit(residual_z)
    scores = model.score(residual_z, index=discovery.index)
    pca_scores = model.transform(residual_z)
    signatures = signature_groups(residual_z, z_threshold=Z_THRESHOLD)

    profiles = pd.DataFrame(
        {
            "ID": discovery["ID"].astype(str).to_numpy(),
            "novelty_score": scores["novelty_score"].to_numpy(),
            "pca_mahalanobis": scores["pca_mahalanobis"].to_numpy(),
            "iforest": scores["iforest"].to_numpy(),
            "max_abs_z": signatures["max_abs_z"].to_numpy(),
            "dominant_trait": signatures["dominant_trait"].to_numpy(),
            "signature": signatures["signature"].to_numpy(),
        }
    )
    for i in range(min(2, pca_scores.shape[1])):
        profiles[f"pc{i + 1}"] = pca_scores[:, i]
    for trait in all_traits:
        profiles[f"{trait}__z"] = scaled[trait].to_numpy()

    signature_summary = build_signature_summary(profiles, residual_z)
    print(f"Found {len(signature_summary)} signatures (n >= {MIN_SIGNATURE_N})")

    stability = bootstrap_novelty_stability(residual_z, n_boot=10, subsample=0.8)
    enrichment = genotype_analysis(profiles, novelty_traits)
    if not enrichment.empty:
        best = (
            enrichment[enrichment["analysis"] == "signature"]
            .sort_values("q_value")
            .drop_duplicates("group")
            .set_index("group")
        )
        signature_summary = signature_summary.merge(
            best[["odds_ratio", "q_value", "mutation"]].rename(
                columns={
                    "odds_ratio": "max_carrier_enrichment",
                    "q_value": "min_q_value",
                    "mutation": "top_mutation",
                }
            ),
            left_on="signature",
            right_index=True,
            how="left",
        )

    control, control_summary = positive_control()
    metrics = {
        "n_animals": len(profiles),
        "n_traits_total": len(all_traits),
        "n_traits_novelty": len(novelty_traits),
        "excluded_traits": sorted(NOVELTY_EXCLUDE),
        "n_signatures": len(signature_summary),
        "n_diffuse": int((profiles["signature"] == "diffuse").sum()),
        "signature_threshold_z": Z_THRESHOLD,
        "novelty_stability": stability,
        "top5_novelty_threshold": float(profiles["novelty_score"].quantile(TOP_QUANTILE)),
        "positive_control": control_summary,
        "data_note": (
            "Phenotypes are pre-adjusted yield deviations. Known bodyweight QTL "
            "effects are attenuated in this deposit, while the chr14 casein signal "
            "is recovered; the prospective calf study is the independent positive control."
        ),
    }

    save_figures(profiles, signature_summary, control, enrichment)

    for column in profiles.columns:
        if column not in {"ID", "signature", "dominant_trait"}:
            profiles[column] = profiles[column].astype("float32")
    profiles.to_parquet(ARTIFACTS_DIR / "animal_profiles.parquet", index=False)
    signature_summary.to_parquet(ARTIFACTS_DIR / "signature_summary.parquet", index=False)
    if not enrichment.empty:
        enrichment.to_parquet(ARTIFACTS_DIR / "carrier_enrichment.parquet", index=False)
    (ARTIFACTS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS_DIR / "positive_control.json").write_text(
        json.dumps({"table": control.to_dict(orient="records"), "summary": control_summary}, indent=2)
    )
    precompute_briefs(signature_summary, ARTIFACTS_DIR / "briefs.json", metrics)

    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
