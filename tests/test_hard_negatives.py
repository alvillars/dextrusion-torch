import logging

import pytest
import tifffile
from synthetic_movie import make_movie

from dextrusion.config import DeXConfig
from dextrusion.data.dataset import build_datasets
from dextrusion.io import create_roi, write_rois

DEATH = [(10, 40, 45), (14, 70, 30), (22, 90, 80), (28, 35, 95)]
NOTHING = [(6, 20, 25), (12, 100, 60), (18, 60, 100), (24, 25, 70), (30, 100, 25), (8, 55, 20)]
CFG = DeXConfig()


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    d = tmp_path_factory.mktemp("hn")
    tifffile.imwrite(d / "a.tif", make_movie(seed=3))
    write_rois(d / "a_cell_death.zip", [create_roi(p, 1) for p in DEATH], verbose=False)
    write_rois(d / "a_nothing.zip", [create_roi(p, 0) for p in NOTHING], verbose=False)
    return d


def build(folder, add_nothing):
    return build_datasets(folder, CFG, val_ratio=0.4, naug=3, add_nothing_windows=add_nothing,
                          seed=2)


def near(ds, roi):
    return [s for s in ds.samples
            if abs(s.z - roi[0]) <= 2 and abs(s.y - roi[1]) <= 5 and abs(s.x - roi[2]) <= 5]


def test_nothing_rois_become_class_zero_windows(folder):
    tr0, va0 = build(folder, 0)
    tr, va = build(folder, 5)
    # the hand-picked windows are appended after everything else: the rest is unchanged
    assert tr.samples[: len(tr0.samples)] == tr0.samples
    assert va.samples[: len(va0.samples)] == va0.samples
    extra = tr.samples[len(tr0.samples):] + va.samples[len(va0.samples):]
    assert len(extra) >= len(NOTHING) * 5 // 2  # base window + add_nothing copies per ROI
    assert all(s.cat == 0 for s in extra)
    assert any(not s.noisy for s in extra) and any(s.noisy for s in extra)


def test_each_nothing_roi_stays_on_one_side_of_the_split(folder):
    tr0, va0 = build(folder, 0)
    tr, va = build(folder, 5)
    n_tr, n_va = len(tr0.samples), len(va0.samples)
    for roi in NOTHING:
        in_tr = [s for s in near(tr, roi) if s in tr.samples[n_tr:]]
        in_va = [s for s in near(va, roi) if s in va.samples[n_va:]]
        assert not (in_tr and in_va), roi
    assert tr.samples[n_tr:] and va.samples[n_va:]


@pytest.mark.parametrize("value", [0, 1])
def test_warns_when_nothing_file_is_ignored(folder, caplog, value):
    with caplog.at_level(logging.WARNING, logger="dextrusion"):
        build(folder, value)
    assert any("a_nothing.zip is ignored" in r.message for r in caplog.records)


def test_no_warning_when_nothing_file_is_used(folder, caplog):
    with caplog.at_level(logging.WARNING, logger="dextrusion"):
        build(folder, 5)
    assert not any("ignored" in r.message for r in caplog.records)
