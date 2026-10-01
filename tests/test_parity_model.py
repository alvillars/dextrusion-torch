"""Converted DeXNets must reproduce the original Keras outputs."""

from pathlib import Path

import numpy as np
import pytest
import torch
from windows import make_windows

from dextrusion.io import load_native

ROOT = Path(__file__).parent.parent
GOLDEN = np.load(ROOT / "tests" / "fixtures" / "golden_outputs.npz")
MODELS = {p.name: p for p in sorted((ROOT / "models").glob("*/*")) if p.is_dir()}


def test_all_golden_models_are_shipped():
    assert set(GOLDEN.files) == set(MODELS)


@pytest.mark.parametrize("name", sorted(GOLDEN.files))
def test_model_matches_keras(name):
    model, config = load_native(MODELS[name])
    x = torch.from_numpy(make_windows()).permute(0, 1, 4, 2, 3).contiguous()  # B,T,1,H,W
    probs = model.predict_proba(x).numpy()
    assert probs.shape[1] == config.ncat
    np.testing.assert_allclose(probs, GOLDEN[name], atol=1e-5)


@pytest.mark.parametrize("name", ["notumAll0", "notumExtSOPDiv0"])
def test_golden_outputs_are_informative(name):
    # guards against a fixture that would pass with any weights (saturated outputs)
    assert GOLDEN[name][12:].std(axis=0).max() > 0.05
