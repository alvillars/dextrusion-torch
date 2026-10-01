"""Conversion of legacy Keras DeXNets, using the real models.

The legacy models are not part of this repository (several MB each, TensorFlow format). Point
``DEXTRUSION_LEGACY_MODELS`` to a folder holding them (folders and/or zips, e.g. the original
``DeXNets`` folder) to run these tests. They use ``uv`` to run TensorFlow in an isolated
environment, like ``dextrusion convert`` does.
"""

import os
import zipfile
from pathlib import Path

import pytest
from safetensors.torch import load_file

from dextrusion.cli import main
from dextrusion.config import DeXConfig
from dextrusion.io import load_model

LEGACY = os.environ.get("DEXTRUSION_LEGACY_MODELS")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not LEGACY, reason="set DEXTRUSION_LEGACY_MODELS to run"),
]
SHIPPED = Path(__file__).parent.parent / "models"


def test_cli_convert_reproduces_shipped_models(tmp_path):
    src = Path(LEGACY) / "notum_all" / "notumAll1"
    out = tmp_path / "converted"
    assert main(["-q", "convert", str(src), "-o", str(out)]) == 0
    got = load_file(out / "model.safetensors")
    want = load_file(SHIPPED / "notum_all" / "notumAll1" / "model.safetensors")
    assert got.keys() == want.keys()
    for k in want:
        assert (got[k] == want[k]).all(), k
    assert DeXConfig.from_json(out / "config.json") == DeXConfig.from_json(
        SHIPPED / "notum_all" / "notumAll1" / "config.json")


def test_zip_conversion_and_automatic_conversion_on_load(tmp_path):
    zsrc = next(Path(LEGACY).rglob("notum_Ext0.zip"))
    out = tmp_path / "ext"
    assert main(["-q", "convert", str(zsrc), "-o", str(out)]) == 0
    assert load_model(out)[1].ncat == 2

    # a legacy folder is converted on first use and cached inside it
    legacy = tmp_path / "legacy"
    with zipfile.ZipFile(zsrc) as z:
        z.extractall(tmp_path)
    (tmp_path / "notum_Ext0").rename(legacy)
    _, cfg = load_model(legacy)
    assert cfg.ncat == 2 and (legacy / "torch" / "model.safetensors").exists()
