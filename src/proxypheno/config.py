from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_data_root() -> Path:
    """Locate the Dryad deposit.

    Order: PROXYPHENO_DATA env var -> data/raw -> sibling dryad folder.
    """
    env = os.environ.get("PROXYPHENO_DATA")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT.parent / "dryad"]
    for candidate in candidates:
        if (candidate / "phens").is_dir() or (candidate / "phen_1.1.txt").is_file():
            return candidate
    return candidates[0]


DATA_ROOT = resolve_data_root()
PHENS_DIR = DATA_ROOT / "phens" if (DATA_ROOT / "phens").is_dir() else DATA_ROOT
GENS_DIR = DATA_ROOT / "gens"
OTHER_DIR = DATA_ROOT / "other"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

for _dir in (INTERIM_DIR, PROCESSED_DIR, ARTIFACTS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# phen_1.*: discovery population growth and developmental traits
PHEN1_TRAITS = {1: "bodyweight", 2: "stature", 3: "bcs"}

# phen_2.*: validation population first-lactation production traits
PHEN2_TRAITS = {1: "milk", 2: "protein", 3: "milkfat"}

# phen_4.*: production and conformation traits
PHEN4_TRAITS = {
    1: "milk",
    2: "protein",
    3: "fat",
    4: "adaptability_milking",
    5: "shed_temperament",
    6: "farmer_opinion",
    7: "capacity",
    8: "rump_angle",
    9: "rump_width",
    10: "legs",
    11: "udder_support",
    12: "front_udder",
    13: "front_teat_placement",
    14: "rear_teat_placement",
    15: "udder_overall",
    16: "dairy_conformation",
}

DISCOVERY_TRAITS = list(PHEN1_TRAITS.values()) + list(PHEN4_TRAITS.values())
VALIDATION_TRAITS = [f"{t}_v" for t in PHEN2_TRAITS.values()] + list(PHEN4_TRAITS.values())

# gen_2.* / gen_3.* prioritised mutations. `hint` is tentative and should be
# confirmed against the paper supplement before being used in prose.
MUTATIONS = {
    1: {"chr": 2, "pos": 22836659, "snp": "rs211074296", "hint": "PLCD4"},
    2: {"chr": 2, "pos": 107313998, "snp": "rs438824358", "hint": "MTRF1"},
    3: {"chr": 5, "pos": 77632752, "snp": "Chr5.77632752", "hint": "FGD4"},
    4: {"chr": 7, "pos": 63649071, "snp": "rs447362502", "hint": ""},
    5: {"chr": 12, "pos": 11463981, "snp": "Chr12.11463981", "hint": ""},
    6: {"chr": 14, "pos": 1802265, "snp": "rs109234250", "hint": ""},
    7: {"chr": 14, "pos": 25015640, "snp": "rs109815800", "hint": ""},
    8: {"chr": 14, "pos": 26064929, "snp": "rs109686357", "hint": ""},
    9: {"chr": 19, "pos": 30114249, "snp": "Chr19.30114249", "hint": ""},
    10: {"chr": 28, "pos": 1312334, "snp": "Chr28.1312334", "hint": "GALNT2"},
    11: {"chr": 29, "pos": 44213160, "snp": "chr29_44213160", "hint": "DPF2/MUS81"},
    12: {"chr": 29, "pos": 44645469, "snp": "chr29_44645469", "hint": "DPF2/MUS81"},
}

# gen_1.* discovery regions (chromosome:start-end in Mb)
DISCOVERY_REGIONS = {
    1: "Chr2:22Mbp",
    2: "Chr2:107Mbp",
    3: "Chr5:77Mbp",
    4: "Chr7:63Mbp",
    5: "Chr12:11Mbp",
    6: "Chr14:25Mbp",
    7: "Chr19:30Mbp",
    8: "Chr28:1Mbp",
    9: "Chr29:44Mbp",
}
