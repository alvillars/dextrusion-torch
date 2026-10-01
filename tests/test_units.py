import numpy as np
import pytest
import torch

from dextrusion.config import DeXConfig
from dextrusion.convert.keras_weights import gru_keras_to_torch, keras_arrays_to_state_dict
from dextrusion.evaluate import compare, false_negatives, false_positives
from dextrusion.inference import Tiling, divide, scale_windows
from dextrusion.model import DeXNet

FIX = __import__("pathlib").Path(__file__).parent / "fixtures"


def test_legacy_config_parser():
    c = DeXConfig.from_legacy_cfg(FIX / "legacy_config.cfg")
    assert c.ncat == 4 and c.nb_filters == 16 and c.batch_size == 50
    assert c.catnames == ["", "_cell_death.zip", "_cell_sop.zip", "_cell_division.zip"]
    assert c.half_size == (22, 22) and c.nframes == (5, 5)
    assert c.window_shape == (10, 45, 45)
    assert c.bounds == (1, 2, 6, 6)
    assert (c.nb_epochs, c.augmentation, c.add_nothing_windows) == (40, 3, 15)


def test_config_json_roundtrip(tmp_path):
    c = DeXConfig.from_legacy_cfg(FIX / "legacy_config.cfg")
    c.to_json(tmp_path / "c.json")
    assert DeXConfig.from_json(tmp_path / "c.json") == c


def test_gru_conversion_matches_keras_equations():
    """Re-implement the Keras (reset_after=True) GRU step by hand and compare with torch."""
    rng = np.random.default_rng(0)
    nin, u = 7, 5
    kernel, rec = rng.normal(size=(nin, 3 * u)), rng.normal(size=(u, 3 * u))
    bias = rng.normal(size=(2, 3 * u))
    x = rng.normal(size=(1, 4, nin))

    def sig(v):
        return 1 / (1 + np.exp(-v))

    h = np.zeros(u)
    for t in range(4):
        mx = x[0, t] @ kernel + bias[0]
        mh = h @ rec + bias[1]
        xz, xr, xh = np.split(mx, 3)
        hz, hr, hh = np.split(mh, 3)
        z, r = sig(xz + hz), sig(xr + hr)
        h = z * h + (1 - z) * np.tanh(xh + r * hh)

    gru = torch.nn.GRU(nin, u, batch_first=True).double()
    sd = gru_keras_to_torch(kernel, rec, bias)
    gru.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()})
    out, _ = gru(torch.from_numpy(x))
    np.testing.assert_allclose(out[0, -1].detach().numpy(), h, atol=1e-10)


def test_state_dict_conversion_rejects_wrong_layout():
    with pytest.raises(ValueError):
        keras_arrays_to_state_dict({"variables/0": np.zeros((3, 3, 1, 4))})


def test_model_shapes_and_eval_determinism():
    m = DeXNet(4, 8).eval()
    x = torch.rand(3, 10, 1, 45, 45)
    assert m(x).shape == (3, 4)
    torch.testing.assert_close(m.predict_proba(x), m.predict_proba(x))
    assert torch.allclose(m.predict_proba(x).sum(1), torch.ones(3))


def test_tiling_matches_original_formulas():
    cfg = DeXConfig()
    shape = (46, 200, 180)  # padded movie
    t = Tiling.for_shape(shape, cfg, shiftz=12, shiftxy=1, dz=2, dxy=25)
    assert t.nx == int(np.ceil((180 - 45 - 1) / 25)) and t.ny == int(np.ceil((200 - 45 - 1) / 25))
    assert t.nz == int(np.ceil((46 - 10 - 12) / 2))
    # window ids enumerate x fastest, then y, then t; same as the original get_index
    nw3, nw2 = t.ny * t.nx, t.nx
    for g in (0, 1, t.nx, t.nx * t.ny, t.n_windows - 1):
        exp = (5 + 12 + int((g / nw3) % t.nz) * 2, 22 + 1 + int((g / nw2) % t.ny) * 25,
               22 + 1 + int(g % nw2) * 25)
        assert tuple(t.centres(g, g + 1)[0]) == exp
    last = t.centres(t.n_windows - 1, t.n_windows)[0]
    assert last[0] + 5 <= shape[0] and last[1] + 22 < shape[1] and last[2] + 22 < shape[2]


def test_scale_windows_per_window_minmax():
    w = np.stack([np.arange(8).reshape(2, 2, 2) * 100 + 7, np.full((2, 2, 2), 9)]).astype(np.uint16)
    out = scale_windows(w)
    assert out.dtype == np.uint8
    assert out[0].min() == 0 and out[0].max() == 255
    assert (out[1] == 0).all()  # flat window


def test_divide_averages_and_crops_padding():
    pred = np.zeros((1, 8, 4, 4), np.float16)
    npred = np.zeros((8, 4, 4), np.uint8)
    pred[0, 2:6], npred[2:6] = 1.0, 2  # accumulated 1.0 over 2 windows -> 0.5
    out = divide(pred, npred, 2, 2)
    assert out.shape == (1, 4, 4, 4) and (out == 127).all()


def test_event_matching_scores():
    truth = [(10, 50, 50), (30, 80, 80)]
    det = [(11, 52, 49), (11, 52, 49), (60, 5, 5)]  # duplicate and a far detection
    s = compare(det, truth, 15, 4)
    assert (s.tp, s.fp, s.fn) == (1, 2, 1)
    assert false_positives(det, truth, 15, 4) == [(11, 52, 49), (60, 5, 5)]
    assert false_negatives(det, truth, 15, 4) == [(30, 80, 80)]
    assert compare([], truth).precision == 0
