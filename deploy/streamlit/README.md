# ProxyPheno Streamlit bundle

Standalone Streamlit app for the ProxyPheno proof of concept. It reads precomputed
artifacts (`animal_profiles.parquet`, `signature_summary.parquet`,
`carrier_enrichment.parquet`, `positive_control.json`, `briefs.json`,
`model_metrics.json`, `model_comparison.json`) from the `artifacts/` folder in this
directory.

Deploy options:

- **Streamlit Community Cloud:** point the app at `deploy/streamlit/app.py`; the root
  `requirements.txt` covers the runtime dependencies.
- **Render:** the root `render.yaml` defines a free web service that runs this app.

Hugging Face compute Spaces now require a paid plan; HF Static Spaces remain free but
cannot run Streamlit.
