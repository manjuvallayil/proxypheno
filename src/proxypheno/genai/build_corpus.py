from __future__ import annotations

import json
from pathlib import Path

from ..config import PHEN1_TRAITS, PHEN2_TRAITS, PHEN4_TRAITS


def build_corpus(output_path: Path) -> dict:
    """Small grounded corpus used by the brief generator and the demo."""
    corpus = {
        "paper": {
            "title": (
                "Non-additive association analysis using proxy phenotypes identifies "
                "novel cattle syndromes"
            ),
            "journal": "Nature Genetics",
            "year": 2021,
            "doi": "10.1038/s41588-021-00872-5",
            "dryad": "https://datadryad.org/dataset/doi:10.5061/dryad.kwh70rz1z",
            "summary": (
                "A sequence-resolution screen of 130,725 cattle used routine proxy "
                "phenotypes (body weight, stature, body condition score) to discover six "
                "novel recessive loci. The deposit contains the genotypes, phenotypes, "
                "pedigree, GWAS summary statistics and simulated data behind the paper."
            ),
        },
        "traits": {
            "phen_1": PHEN1_TRAITS,
            "phen_2": PHEN2_TRAITS,
            "phen_4": PHEN4_TRAITS,
        },
        "guardrails": [
            "Numbers must be injected from structured model outputs, never generated.",
            "Every brief must cite the result identifiers it is based on.",
            "The engine produces discovery-support hypotheses, not diagnoses.",
            "Refuse to summarise when evidence is insufficient.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(corpus, indent=2, default=str))
    return corpus
