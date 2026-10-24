from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proxypheno.config import ARTIFACTS_DIR

st.set_page_config(page_title="ProxyPheno", page_icon=None, layout="wide")


@st.cache_data
def load_artifacts():
    def parquet(name):
        path = ARTIFACTS_DIR / name
        return pd.read_parquet(path) if path.exists() else None

    def js(name):
        path = ARTIFACTS_DIR / name
        return json.loads(path.read_text()) if path.exists() else {}

    return (
        parquet("animal_profiles.parquet"),
        parquet("signature_summary.parquet"),
        parquet("carrier_enrichment.parquet"),
        js("positive_control.json"),
        js("briefs.json"),
        js("model_metrics.json"),
        js("model_comparison.json"),
    )


profiles, signatures, enrichment, control, briefs, metrics, comparison = load_artifacts()

st.title("ProxyPheno")
st.caption("An ML proxy-phenotype engine for novel trait discovery in dairy cattle")

if profiles is None:
    st.warning(
        "No artifacts found. Run `make pipeline` first, or point the app at a folder "
        "containing `animal_profiles.parquet`."
    )
    st.stop()

page = st.sidebar.radio(
    "View", ["Overview", "Animal explorer", "Signature explorer", "Method"]
)

if page == "Overview":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Animals", f"{len(profiles):,}")
    col2.metric("Traits in novelty model", metrics.get("n_traits_novelty", "n/a"))
    col3.metric("Signatures", metrics.get("n_signatures", "n/a"))
    col4.metric("Top-5% threshold", f"{metrics.get('top5_novelty_threshold', 0):.3f}")

    st.subheader("What this is")
    st.write(
        "ProxyPheno learns the joint distribution of routine dairy phenotypes, scores how "
        "unexpected each animal's profile is, and groups animals by the trait that drives "
        "their unusualness. The residual space is a single continuum, so novelty is a "
        "ranking problem; interpretable signatures replace forced clusters."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Novelty distribution")
        counts, edges = np.histogram(profiles["novelty_score"], bins=40)
        hist = pd.DataFrame(
            {"novelty": edges[:-1].round(3), "animals": counts}
        ).set_index("novelty")
        st.bar_chart(hist)

    with right:
        st.subheader("Positive control: calf study")
        if control.get("table"):
            table = pd.DataFrame(control["table"])
            st.dataframe(table, use_container_width=True, hide_index=True)
            summary = control.get("summary", {})
            if summary:
                st.metric(
                    "PLCD4 carriers vs controls",
                    f"{summary.get('pct_vs_control', 0):.1f}%",
                    help=(
                        f"{summary.get('n_carriers')} carriers vs "
                        f"{summary.get('n_controls')} controls; "
                        f"mean difference {summary.get('mean_difference', 0):.1f} kg "
                        f"(95% CI {summary.get('difference_ci_low', 0):.1f} to "
                        f"{summary.get('difference_ci_high', 0):.1f})"
                    ),
                )
        else:
            st.info("Positive-control artifact not found.")

    st.subheader("Known-locus recovery (discovery population)")
    if enrichment is not None and not enrichment.empty:
        tail_hits = enrichment[enrichment["analysis"] == "tail"].nsmallest(8, "q_value")
        st.dataframe(
            tail_hits[
                [
                    "mutation",
                    "hint",
                    "trait",
                    "tail",
                    "n",
                    "carrier_rate",
                    "rest_rate",
                    "odds_ratio",
                    "q_value",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        novelty_hits = enrichment[enrichment["analysis"] == "novelty"].nsmallest(5, "q_value")
        st.dataframe(
            novelty_hits[
                ["mutation", "hint", "n", "carrier_rate", "rest_rate", "odds_ratio", "q_value"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "The chr14 casein locus is a known milk-composition signal; recovering it "
            "validates the enrichment machinery. Enrichment is hypothesis-generating, "
            "not causal."
        )
    else:
        st.info("Enrichment artifact not found.")

    st.subheader("Largest signatures")
    if signatures is not None:
        st.dataframe(signatures.head(15), use_container_width=True, hide_index=True)

    stability = metrics.get("novelty_stability", {})
    if stability:
        st.subheader("Stability")
        st.write(
            f"Bootstrap refits ({stability.get('n_boot', 0)}x) give a mean Spearman "
            f"correlation of {stability.get('mean_spearman', float('nan')):.3f} and a "
            f"top-5% Jaccard overlap of "
            f"{stability.get('mean_top5_jaccard', float('nan')):.3f}."
        )

    if comparison.get("summary"):
        st.subheader("Baseline vs ML")
        st.dataframe(
            pd.DataFrame(comparison["summary"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(comparison.get("note", ""))

    st.caption(metrics.get("data_note", ""))

elif page == "Animal explorer":
    st.subheader("Animal explorer")
    top_ids = profiles.nlargest(500, "novelty_score")["ID"].tolist()
    animal_id = st.selectbox("Animal ID (top 500 most novel)", top_ids)
    manual_id = st.text_input("Or enter any animal ID")
    if manual_id.strip():
        animal_id = manual_id.strip()

    match = profiles[profiles["ID"] == animal_id]
    if match.empty:
        st.warning("Animal not found.")
        st.stop()
    row = match.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Novelty score", f"{row['novelty_score']:.3f}")
    col2.metric("Signature", str(row["signature"]))
    col3.metric("Max |residual z|", f"{row['max_abs_z']:.2f}")

    trait_cols = [c for c in profiles.columns if c.endswith("__z")]
    st.subheader("Trait profile (robust z-score)")
    st.bar_chart(row[trait_cols].astype(float))

    brief = briefs.get("signatures", {}).get(str(row["signature"]))
    if brief:
        st.subheader("Signature brief")
        st.json(brief)

elif page == "Signature explorer":
    st.subheader("Signature summary")
    if signatures is None:
        st.info("No signature summary found.")
        st.stop()
    st.dataframe(signatures, use_container_width=True, hide_index=True)

    st.subheader("Signature sizes")
    st.bar_chart(signatures.head(20).set_index("signature")["n"])

    choice = st.selectbox("Inspect a signature", signatures["signature"].tolist())
    brief = briefs.get("signatures", {}).get(str(choice))
    if brief:
        st.json(brief)
    if enrichment is not None:
        rows = enrichment[
            (enrichment["analysis"] == "signature") & (enrichment["group"] == choice)
        ]
        if not rows.empty:
            st.subheader("Carrier enrichment")
            st.dataframe(
                rows.sort_values("q_value")[
                    [
                        "mutation",
                        "hint",
                        "n",
                        "carrier_rate",
                        "rest_rate",
                        "odds_ratio",
                        "p_value",
                        "q_value",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

elif page == "Method":
    st.subheader("Pipeline")
    st.write(
        "1. Load 19 routine phenotypes from the Dryad deposit. "
        "2. Exclude 3 traits with >10% missingness from the novelty model; "
        "iteratively impute the rest. "
        "3. Cross-fitted Ridge residuals give each trait conditional on the others. "
        "4. PCA, Mahalanobis distance and IsolationForest produce a novelty score. "
        "5. HDBSCAN finds no stable density clusters, so animals are grouped by the "
        "dominant residual trait into low/high signatures. "
        "6. Ridge, LightGBM and an MLP are compared on the same hold-out split. "
        "7. Carrier enrichment tests the 12 prioritised mutations in trait tails, "
        "signatures and the top-5% novelty set. "
        "8. Precomputed briefs translate structured outputs into language."
    )
    st.subheader("Limitations")
    st.write(
        "Phenotypes are pre-adjusted yield deviations; known bodyweight QTL effects are "
        "attenuated in this deposit, while the chr14 casein signal is recovered. The "
        "analysis is observational and produces discovery-support hypotheses, not "
        "diagnoses or causal claims. The prospective calf study is the independent "
        "positive control."
    )
