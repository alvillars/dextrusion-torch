import numpy as np
import pytest
import tifffile
from synthetic_movie import make_movie

from dextrusion.cli import main, parse_oversample
from dextrusion.config import DeXConfig
from dextrusion.data.dataset import build_datasets
from dextrusion.io import create_roi, load_model, write_rois

DEATH = [(10, 40, 45), (14, 70, 30), (22, 90, 80), (28, 35, 95)]
DIVISION = [(12, 100, 100), (24, 30, 30), (16, 50, 100), (26, 80, 40)]


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    d = tmp_path_factory.mktemp("os")
    for name, seed in (("a", 3), ("b", 4)):
        tifffile.imwrite(d / f"{name}.tif", make_movie(seed=seed))
        write_rois(d / f"{name}_cell_death.zip", [create_roi(p, 1) for p in DEATH], verbose=False)
        write_rois(d / f"{name}_cell_division.zip", [create_roi(p, 3) for p in DIVISION], verbose=False)
    return d


CFG = DeXConfig()


def build(folder, **kw):
    return build_datasets(folder, CFG, val_ratio=0.4, naug=3, add_nothing_windows=0, seed=5, **kw)


def count(ds, movie):
    return sum(s.movie == movie for s in ds.samples)


def test_oversampling_multiplies_training_windows_only(folder):
    tr1, va1 = build(folder)
    tr5, va5 = build(folder, oversample={"a": 5})
    ratio = count(tr5, 0) / count(tr1, 0)
    assert 3.5 < ratio < 6.5, ratio  # about 5x (the split of copies between train / val varies a bit)
    assert 0.8 < count(tr5, 1) / count(tr1, 1) < 1.2  # the other movie is sampled once
    # the validation windows of the oversampled movie come from the first pass: unchanged
    assert [s for s in va5.samples if s.movie == 0] == [s for s in va1.samples if s.movie == 0]


def test_events_stay_on_one_side_of_the_split_when_oversampled(folder):
    tr, va = build(folder, oversample={"a": 6, "b": 3})
    events = {1: DEATH, 3: DIVISION}

    def sources(ds):
        out = set()
        for s in ds.samples:
            if s.cat == 0:
                continue
            near = [r for r in events[s.cat] if abs(r[0] - s.z) <= 3 and abs(r[1] - s.y) <= 9
                    and abs(r[2] - s.x) <= 9]
            assert len(near) == 1
            out.add((s.movie, s.cat, near[0]))
        return out

    assert sources(tr) and sources(va)
    assert not sources(tr) & sources(va)


def test_oversampling_is_deterministic_and_validated(folder):
    a, _ = build(folder, oversample={"a": 3})
    b, _ = build(folder, oversample={"a": 3})
    assert a.samples == b.samples
    with pytest.raises(ValueError, match="no movie named 'zzz'"):
        build(folder, oversample={"zzz": 2})
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="integer >= 1"):
            build(folder, oversample={"a": bad})


def test_parse_oversample():
    assert parse_oversample(None) is None
    assert parse_oversample(["kate_v2_c1_d10-12=30", "b=2"]) == {"kate_v2_c1_d10-12": 30, "b": 2}
    for bad in (["a"], ["a=0"], ["a=x"], ["=3"], ["a=2.5"]):
        with pytest.raises(SystemExit):
            parse_oversample(bad)


def test_cli_train_with_oversample(folder, tmp_path):
    out = tmp_path / "net"
    code = main(["-q", "train", str(folder), "-o", str(out), "--nb-filters", "4", "--batch-size", "8",
                 "--epochs", "1", "--naug", "3", "--add-nothing", "0", "--device", "cpu",
                 "--oversample", "a=4"])
    assert code == 0
    assert load_model(out)[1].ncat == 4
    with pytest.raises(SystemExit, match="no movie named"):
        main(["-q", "train", str(folder), "-o", str(tmp_path / "x"), "--epochs", "1", "--device", "cpu",
              "--oversample", "nope=2"])
    assert np.isfinite(float(open(out / "history.csv").read().splitlines()[1].split(",")[2]))
