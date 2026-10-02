import numpy as np
import pytest
import tifffile
from synthetic_movie import make_movie

from dextrusion.cli import build_parser, parse_classes
from dextrusion.config import DeXConfig
from dextrusion.data.dataset import build_datasets
from dextrusion.io import read_rois
from dextrusion.label import (
    CLASS_KEYS,
    DEFAULT_CLASSES,
    SAVE_KEY,
    clean_points,
    load_points,
    roi_path,
    save_points,
)

SHAPE = (36, 128, 128)


def test_clean_points_rounds_clips_dedupes_and_sorts():
    pts = [[5.4, 10.6, 20.2], [5.4, 10.6, 20.2], [0.0, 3.0, 4.0], [999, -3.2, 130.7]]
    assert clean_points(pts, SHAPE) == [(0, 3, 4), (5, 11, 20), (35, 0, 127)]
    assert clean_points([], SHAPE) == []


def test_save_load_roundtrip_backup_and_empty(tmp_path):
    p = tmp_path / "m_cell_division.zip"
    assert load_points(p).shape == (0, 3)  # no file yet
    assert save_points(p, [[3, 10, 20], [7.2, 50.4, 60.6]], SHAPE) == 2
    assert not (tmp_path / "m_cell_division.bak.zip").exists()  # nothing to back up the first time
    assert read_rois(p) == [(3, 10, 20), (7, 50, 61)]
    np.testing.assert_array_equal(load_points(p), [[3, 10, 20], [7, 50, 61]])
    # overwriting keeps the previous version as .bak
    save_points(p, [[9, 9, 9]], SHAPE)
    assert read_rois(tmp_path / "m_cell_division.bak.zip") == [(3, 10, 20), (7, 50, 61)]
    assert read_rois(p) == [(9, 9, 9)]
    # all points removed -> a valid empty file, not a crash
    assert save_points(p, np.zeros((0, 3)), SHAPE) == 0
    assert read_rois(p) == []


def test_parse_classes_and_defaults():
    assert parse_classes(None) is None
    assert parse_classes(["a=_a.zip", "b=_b.zip"]) == {"a": "_a.zip", "b": "_b.zip"}
    for bad in (["a"], ["a=nozip"], ["=_a.zip"]):
        with pytest.raises(SystemExit):
            parse_classes(bad)
    assert list(DEFAULT_CLASSES) == ["division", "delamination"]
    assert len(set(CLASS_KEYS)) == len(CLASS_KEYS)
    assert SAVE_KEY not in CLASS_KEYS


def test_train_catnames_option_is_validated():
    from dextrusion.cli import _cmd_train

    parser = build_parser()
    a = parser.parse_args(["train", "x", "-o", "y", "--catnames", "", "_cell_delamination.zip",
                           "_cell_division.zip"])
    assert a.catnames == ["", "_cell_delamination.zip", "_cell_division.zip"]
    a = parser.parse_args(["train", "x", "-o", "y", "--init-from", "net", "--catnames", "",
                           "_a.zip"])
    with pytest.raises(SystemExit, match="init-from"):
        _cmd_train(a)
    for bad in (["_a.zip", "_b.zip"], [""], ["", "_a.txt"]):
        a = parser.parse_args(["train", "x", "-o", "y", "--catnames", *bad])
        with pytest.raises(SystemExit):
            _cmd_train(a)


def test_labels_feed_training_with_custom_classes(tmp_path):
    """ROI files written by the labeling code are read by the training pipeline."""
    movie = make_movie()
    tifffile.imwrite(tmp_path / "m.tif", movie)
    events = {"delamination": [(10, 40, 45), (14, 70, 30), (22, 90, 80)],
              "division": [(28, 35, 95), (18, 60, 60)]}
    for name, pts in events.items():
        save_points(roi_path(tmp_path, "m", DEFAULT_CLASSES[name]), pts, movie.shape)
    cfg = DeXConfig(ncat=3, catnames=["", "_cell_delamination.zip", "_cell_division.zip"])
    tr, va = build_datasets(tmp_path, cfg, val_ratio=0.3, naug=3, add_nothing_windows=0, seed=1)
    cats = {s.cat for s in tr.samples + va.samples}
    assert cats == {0, 1, 2}  # nothing, delamination, division
    x, y = tr[0]
    assert x.shape == (10, 1, 45, 45) and int(y) in (0, 1, 2)


# ---------------------------------------------------------------- napari part (skipped without it)
def press(viewer, key):
    """Trigger the viewer-level binding of a plain key (napari stores letters in upper case)."""
    from napari.utils.key_bindings import coerce_keybinding

    viewer.keymap[coerce_keybinding(key)](viewer)


def test_keys_do_not_clash_with_napari_shortcuts():
    pytest.importorskip("napari")
    from napari.utils.key_bindings import coerce_keybinding
    from napari.utils.shortcuts import default_shortcuts

    used = {str(coerce_keybinding(str(k))) for keys in default_shortcuts.values() for k in keys}
    for key in [*CLASS_KEYS, SAVE_KEY]:
        assert str(coerce_keybinding(key)) not in used, f"key '{key}' is used by napari"


def _viewer_and_layers(tmp_path, **kw):
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    from dextrusion.label import build_layers

    movie = make_movie()
    viewer = ViewerModel()
    layers, save_all = build_layers(viewer, movie, tmp_path, "m", dict(DEFAULT_CLASSES), **kw)
    return viewer, layers, save_all, movie


def test_napari_layers_autosave_and_resume(tmp_path):
    viewer, layers, save_all, movie = _viewer_and_layers(tmp_path)
    assert [layer.name for layer in viewer.layers] == ["movie", "division", "delamination"]
    assert layers["division"].ndim == 3

    layers["division"].add([12.0, 40.2, 50.7])  # autosave fires on the data event
    layers["delamination"].add([20, 80, 30])
    assert read_rois(roi_path(tmp_path, "m", "_cell_division.zip")) == [(12, 40, 51)]
    assert read_rois(roi_path(tmp_path, "m", "_cell_delamination.zip")) == [(20, 80, 30)]
    assert save_all() == {"division": 1, "delamination": 1}

    # a new session resumes from the files
    _, layers2, _, _ = _viewer_and_layers(tmp_path)
    np.testing.assert_array_equal(layers2["division"].data, [[12, 40, 51]])

    # removing the point rewrites an empty file
    layers2["division"].selected_data = {0}
    layers2["division"].remove_selected()
    assert read_rois(roi_path(tmp_path, "m", "_cell_division.zip")) == []


def test_napari_autosave_can_be_switched_off(tmp_path):
    _, layers, save_all, _ = _viewer_and_layers(tmp_path)
    save_all.state["autosave"] = False
    layers["division"].add([5, 5, 5])
    assert not roi_path(tmp_path, "m", "_cell_division.zip").exists()
    save_all()
    assert read_rois(roi_path(tmp_path, "m", "_cell_division.zip")) == [(5, 5, 5)]


def test_controls_keys_buttons_and_status(tmp_path, monkeypatch):
    """Drive the key bindings, buttons and status line (windowless viewer model, offscreen Qt)."""
    pytest.importorskip("napari")
    pytest.importorskip("magicgui")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    from dextrusion.label import attach_controls

    app = QApplication.instance() or QApplication([])  # noqa: F841 (must stay alive)
    viewer, layers, save_all, _ = _viewer_and_layers(tmp_path)
    container = attach_controls(viewer, layers, save_all, list(layers))
    by_text = {w.text: w for w in container if hasattr(w, "text")}
    status = container[-1]

    assert viewer.layers.selection.active is layers["division"]  # first class active at start
    assert layers["division"].mode == "add"
    assert status.value == "division: 0, delamination: 0"

    press(viewer, CLASS_KEYS[1])  # key of the second class
    assert viewer.layers.selection.active is layers["delamination"]
    assert layers["delamination"].mode == "add"
    press(viewer, CLASS_KEYS[0])
    assert viewer.layers.selection.active is layers["division"]
    by_text[f"add delamination  (key {CLASS_KEYS[1]})"].clicked.emit()  # same through the button
    assert viewer.layers.selection.active is layers["delamination"]

    layers["division"].add([4, 30, 31])
    assert status.value == "division: 1, delamination: 0"  # status follows the data

    by_text["autosave on every change"].value = False
    layers["division"].add([6, 31, 32])
    assert len(read_rois(roi_path(tmp_path, "m", "_cell_division.zip"))) == 1  # not saved yet
    press(viewer, SAVE_KEY)
    assert read_rois(roi_path(tmp_path, "m", "_cell_division.zip")) == [(4, 30, 31), (6, 31, 32)]
    assert status.value.startswith("saved ")
