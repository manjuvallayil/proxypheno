# ProxyPheno

An ML proxy-phenotype engine for novel trait discovery in dairy cattle.

This is a proof of concept built on the Dryad deposit behind:

> Reynolds et al. (2021). *Non-additive association analysis using proxy phenotypes
> identifies novel cattle syndromes.* Nature Genetics 53, 949-954.
> https://doi.org/10.1038/s41588-021-00872-5

## What it does

1. Loads 19 routine phenotypes for ~80k New Zealand dairy animals.
2. Computes cross-fitted conditional residuals: how unexpected each trait is given
   the rest of the animal's profile.
3. Scores novelty in PCA space (Mahalanobis + IsolationForest).
4. Groups animals by dominant residual trait into interpretable signatures
   (HDBSCAN found no density-separated clusters, so none are forced).
5. Tests signature, trait-tail and novelty enrichment for known recessive-locus carriers.
6. Compares Ridge, LightGBM and an MLP on identical splits.
7. Precomputes grounded, plain-language briefs from structured results.

## Data

The data is not committed. Download the Dryad deposit manually:

- Dataset page: https://datadryad.org/dataset/doi:10.5061/dryad.kwh70rz1z
- Required: `README.dryad.md`, `phens.tar.gz`
- Optional (genetic validation): `gens.tar.gz`, `other.tar.gz`

Place the extracted folders at either `proxypheno/data/raw/` or a sibling
`dryad/` folder. The path resolver in `src/proxypheno/config.py` handles both.

## Quick start

```bash
make setup     # create .venv and install
make audit     # schema, missingness, positive-control check
make test      # unit tests
make pipeline  # build artifacts
make compare   # Ridge vs LightGBM vs MLP comparison
make app       # launch the Streamlit demo
```

## Results

- 80,027 animals, 19 traits; 16 used for novelty after excluding three with >10% missingness.
- 22 dominant-trait signatures; 76,652 animals are diffuse.
- Novelty stability over 10 bootstrap refits: Spearman **0.988**, top-5% Jaccard **0.783**.
- Positive control (prospective calf study): PLCD4 carriers **147.1 kg vs 198.4 kg** for
  controls (**-25.9%**; 95% CI -62.3 to -39.4 kg).
- Known-locus recovery: the chr14 casein region is enriched in milk/protein tails
  (q < 1e-100) and persists within breed.
- Proxy-prediction mean R²: LightGBM **0.488**, MLP **0.481**, Ridge **0.479**
  (LightGBM best on 14/16 traits; the linear baseline is strong).

## Layout

```
src/proxypheno/    pipeline package
  config.py        paths, trait maps, mutation map
  io.py            phenotype, pedigree, GWAS loaders
  qc.py            missingness, scaling, outliers
  features.py      conditional residuals, PCA
  models/          novelty, genotypes, baseline, ML
  eval/            metrics, stability, conformal intervals
  genai/           grounded corpus and briefs
  app/             Streamlit demo
scripts/           audit, pipeline and model-comparison entry points
tests/             schema, metrics, pipeline tests
deploy/            Vercel microsite and prepared Streamlit bundle
artifacts/         precomputed demo artifacts
```

## Status

Phases 0-4 are complete: pipeline, novelty and signatures, positive controls,
GenAI-style briefs, and the baseline-vs-ML comparison. Phase 5 (deploy) is prepared
but blocked on hosting/auth: HF compute Spaces now require a paid plan, and the
Vercel CLI is not authenticated. Streamlit Community Cloud or Render are the
free-tier options for the demo.

## Limitations

Phenotypes are pre-adjusted yield deviations. The deposited bodyweight trait
attenuates the 2021 QTL effects, so the prospective calf study is the clean
positive control; the chr14 casein signal validates the enrichment machinery.
The analysis is observational and produces discovery-support hypotheses, not
diagnoses or causal claims.
