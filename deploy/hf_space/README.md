---
title: ProxyPheno
sdk: streamlit
app_file: app.py
pinned: false
---

# ProxyPheno demo

Streamlit demo for the ProxyPheno proof of concept. It reads precomputed artifacts
(`animal_profiles.parquet`, `signature_summary.parquet`, `carrier_enrichment.parquet`,
`positive_control.json`, `briefs.json`, `model_metrics.json`) from the `artifacts/`
folder in this Space.

Copy the artifacts produced by `make pipeline` into `artifacts/` before deploying.
