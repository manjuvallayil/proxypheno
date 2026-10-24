import pytest

from proxypheno.config import PHENS_DIR
from proxypheno.io import load_calf_study, load_discovery, load_validation

pytestmark = pytest.mark.skipif(
    not (PHENS_DIR / "phen_1.1.txt").exists(), reason="Dryad phenotype data not present"
)


def test_discovery_schema():
    df = load_discovery()
    for column in ["ID", "bodyweight", "stature", "bcs", "milk", "protein", "fat"]:
        assert column in df.columns
    assert len(df) > 70_000


def test_validation_schema():
    df = load_validation()
    assert "milk_v" in df.columns
    assert len(df) > 30_000


def test_calf_study_schema():
    df = load_calf_study()
    assert {"animal", "VARIANT", "HERD", "weight"} <= set(df.columns)
    assert set(df["VARIANT"]).issuperset({"CONTROL", "PLCD4"})
