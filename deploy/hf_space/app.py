from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="ProxyPheno", layout="wide")


@st.cache_data
def load():
    def parquet(name):
        path = ARTIFACTS / name
        return pd.read_parquet(path) if path.exists() else None

    def js(name):
        path = ARTIFACTS / name
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


profiles, signatures, enrichment, control, briefs, metrics, comparison = load()

st.title("ProxyPheno")
st.caption("An ML proxy-phenotype engine for novel trait discovery in dairy cattle")

if profiles is None:
    st.warning("Artifacts not found. Copy the output of `make pipeline` into `artifacts/`.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Animals", f"{len(profiles):,}")
col2.metric("Traits in novelty model", metrics.get("n_traits_novelty", "n/a"))
col3.metric("Signatures", metrics.get("n_signatures", "n/a"))
col4.metric("Top-5% threshold", f"{metrics.get('top5_novelty_threshold', 0):.3f}")

page = st.sidebar.radio(
    "View", ["Overview", "Animal explorer", "Signature explorer", "Method"]
)

if page == "Overview":
    st.subheader("Novelty distribution")
    counts, edges = np.histogram(profiles["novelty_score"], bins=40)
    st.bar_chart(pd.DataFrame({"novelty": edges[:-1], "animals": counts}).set_index("novelty"))

    st.subheader("Positive control: calf study")
    if control.get("table"):
        st.dataframe(pd.DataFrame(control["table"]), use_container_width=True, hide_index=True)
        summary = control.get("summary", {})
        if summary:
            st.metric("PLCD4 carriers vs controls", f"{summary.get('pct_vs_control', 0):.1f}%")
    else:
        st.info("Positive-control artifact not found.")

    st.subheader("Known-locus recovery")
    if enrichment is not None and not enrichment.empty:
        hits = enrichment[enrichment["analysis"] == "tail"].nsmallest(8, "q_value")
        st.dataframe(
            hits[["mutation", "hint", "trait", "tail", "odds_ratio", "q_value"]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("The chr14 casein locus is a known milk-composition signal.")
    else:
        st.info("Enrichment artifact not found.")

    st.subheader("Largest signatures")
    if signatures is not None:
        st.dataframe(signatures.head(15), use_container_width=True, hide_index=True)

    stability = metrics.get("novelty_stability", {})
    if stability:
        st.write(
            f"Bootstrap stability: Spearman {stability.get('mean_spearman', float('nan')):.3f}, "
            f"top-5% Jaccard {stability.get('mean_top5_jaccard', float('nan')):.3f}."
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
    animal_id = st.selectbox(
        "Animal ID (top 500 most novel)", profiles.nlargest(500, "novelty_score")["ID"].tolist()
    )
    row = profiles[profiles["ID"] == animal_id].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Novelty score", f"{row['novelty_score']:.3f}")
    col2.metric("Signature", str(row["signature"]))
    col3.metric("Max |residual z|", f"{row['max_abs_z']:.2f}")
    trait_cols = [c for c in profiles.columns if c.endswith("__z")]
    st.bar_chart(row[trait_cols].astype(float))
    brief = briefs.get("signatures", {}).get(str(row["signature"]))
    if brief:
        st.json(brief)

elif page == "Signature explorer":
    st.dataframe(signatures, use_container_width=True, hide_index=True)
    st.bar_chart(signatures.head(20).set_index("signature")["n"])
    choice = st.selectbox("Inspect a signature", signatures["signature"].tolist())
    brief = briefs.get("signatures", {}).get(str(choice))
    if brief:
        st.json(brief)

else:
    st.write(
        "Conditional Ridge residuals -> PCA/Mahalanobis + IsolationForest novelty -> "
        "dominant-trait signatures -> known-locus enrichment -> grounded briefs."
    )
    st.write(
        "Phenotypes are pre-adjusted yield deviations. The analysis is observational and "
        "produces discovery-support hypotheses, not diagnoses or causal claims."
    )
