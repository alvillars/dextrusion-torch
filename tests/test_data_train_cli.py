import csv

import numpy as np
import pytest
import tifffile
import torch
from synthetic_movie import make_movie

from dextrusion.cli import main
from dextrusion.config import DeXConfig
from dextrusion.data.dataset import build_datasets, collate_windows
from dextrusion.io import create_roi, load_model, read_rois, write_rois
from dextrusion.train import TrainOptions, train

EVENTS = {1: [(10, 40, 45), (14, 70, 30)], 2: [(22, 90, 80)], 3: [(28, 35, 95)]}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("train")
    tifffile.imwrite(d / "m1.tif", make_movie())
    cfg = DeXConfig()
    for cat, pts in EVENTS.items():
        write_rois(d / f"m1{cfg.catnames[cat]}", [create_roi(p, cat) for p in pts], verbose=False)
    write_rois(d / "m1_nothing.zip", [create_roi((5, 20, 20))], verbose=False)
    return d


def test_roi_roundtrip(data_dir):
    assert read_rois(data_dir / "m1_cell_death.zip") == EVENTS[1]


def test_dataset_is_deterministic_and_leak_free(data_dir):
    cfg = DeXConfig()
    a_tr, a_va = build_datasets(data_dir, cfg, 0.4, naug=3, add_nothing_windows=5, seed=7)
    b_tr, b_va = build_datasets(data_dir, cfg, 0.4, naug=3, add_nothing_windows=5, seed=7)
    assert a_tr.samples == b_tr.samples and a_va.samples == b_va.samples
    c_tr, _ = build_datasets(data_dir, cfg, 0.4, naug=3, add_nothing_windows=5, seed=8)
    assert c_tr.samples != a_tr.samples

    # every window derived from one source ROI lies on one side of the split
    def sources(ds):
        out = set()
        for s in ds.samples:
            if s.cat == 0:
                continue
            near = [r for r in EVENTS[s.cat] if abs(r[0] - s.z) <= 3 and abs(r[1] - s.y) <= 9
                    and abs(r[2] - s.x) <= 9]
            assert len(near) == 1
            out.add((s.cat, near[0]))
        return out

    assert not sources(a_tr) & sources(a_va)
    assert a_tr.samples and a_va.samples
    assert all(s.cat in range(4) for s in a_tr.samples)


def test_dataset_items_and_collate(data_dir):
    cfg = DeXConfig()
    tr, va = build_datasets(data_dir, cfg, 0.3, naug=3, add_nothing_windows=5, seed=1)
    x, y = collate_windows([tr[i] for i in range(6)])
    assert x.shape == (6, 10, 1, 45, 45) and y.shape == (6,)
    assert x.min() >= 0 and x.max() <= 1 + 1e-6
    xv, _ = collate_windows([va[0], va[0]])
    torch.testing.assert_close(xv[0], xv[1])  # validation windows are not augmented
    xb, _ = collate_windows([tr[i] for i in range(6)], normalize="batch")
    assert xb.shape == x.shape


def _train(data_dir, out, seed=0, **kw):
    cfg = DeXConfig(nb_filters=4, batch_size=8)
    opts = TrainOptions(epochs=3, naug=8, add_nothing_windows=5, seed=seed, device="cpu", **kw)
    return train(data_dir, out, cfg, opts)


def test_training_saves_loadable_model_and_is_seed_deterministic(data_dir, tmp_path):
    _train(data_dir, tmp_path / "a")
    _train(data_dir, tmp_path / "b")
    with open(tmp_path / "a" / "history.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3 and float(rows[-1]["loss"]) < 5
    (ma, ca), (mb, _) = load_model(tmp_path / "a"), load_model(tmp_path / "b")
    assert ca.nb_epochs == 3 and ca.augmentation == 8
    for (k, va), vb in zip(ma.state_dict().items(), mb.state_dict().values()):
        torch.testing.assert_close(va, vb, msg=k)


def test_retrain_from_existing_model(data_dir, tmp_path):
    _train(data_dir, tmp_path / "base")
    cfg = DeXConfig(batch_size=8)
    out = train(data_dir, tmp_path / "re", cfg,
                TrainOptions(epochs=1, naug=8, add_nothing_windows=5, device="cpu"),
                init_from=tmp_path / "base")
    assert load_model(out)[1].nb_filters == 4  # geometry follows the retrained model


def test_cli_detect_and_evaluate(data_dir, tmp_path):
    model = tmp_path / "net"
    _train(data_dir, model)
    out = tmp_path / "res"
    code = main(["-q", "detect", str(data_dir / "m1.tif"), "-m", str(model), "-o", str(out),
                 "--volume-threshold", "0", "--proba-threshold", "0", "--device", "cpu",
                 "--save-proba", "--save-cleaned", "--cat", "1"])
    assert code == 0
    names = {p.name for p in out.iterdir()}
    assert {"m1_cell_death_rawproba.tif", "m1_cell_death_proba.tif", "m1_cell_death.zip"} <= names
    raw = tifffile.imread(out / "m1_cell_death_rawproba.tif")
    assert raw.shape == make_movie().shape and raw.dtype == np.uint8
    assert main(["-q", "evaluate", "events", str(out / "m1_cell_death.zip"),
                 str(data_dir / "m1_cell_death.zip")]) == 0
    assert main(["-q", "evaluate", "windows", str(data_dir), "-m", str(model),
                 "--device", "cpu"]) == 0
