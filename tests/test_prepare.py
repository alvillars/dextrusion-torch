import json

import numpy as np
import pytest
import tifffile

from dextrusion.cli import main
from dextrusion.io import create_roi, read_rois, write_rois
from dextrusion.prepare import find_roi_files, prepare, rescale_point


def blob_movie(shape=(40, 200, 300), centre=(20, 61, 177), sigma=(2.0, 4.0, 4.0)):
    t, y, x = np.indices(shape)
    g = np.exp(-sum(((c - m) / s) ** 2 / 2 for c, m, s in zip((t, y, x), centre, sigma)))
    return (g * 3000 + 50).astype(np.uint16)


def test_rescale_point_corners_and_middle():
    assert rescale_point((0, 0, 0), (40, 200, 300), (40, 100, 150)) == (0, 0, 0)
    assert rescale_point((39, 199, 299), (40, 200, 300), (40, 100, 150)) == (39, 99, 149)
    assert rescale_point((20, 100, 150), (41, 201, 301), (21, 101, 151)) == (10, 50, 75)
    assert rescale_point((5, 5, 5), (1, 1, 1), (1, 1, 1)) == (0, 0, 0)  # degenerate axis


@pytest.mark.parametrize("cell_diameter,duration", [(50, 4.5), (25, 9.0), (50, 9.0), (38, 4.5)])
def test_roi_follows_the_blob_in_the_rescaled_movie(tmp_path, cell_diameter, duration):
    movie = blob_movie()
    centre = (20, 61, 177)
    src = tmp_path / "src"
    src.mkdir()
    tifffile.imwrite(src / "m.tif", movie)
    write_rois(src / "m_cell_division.zip", [create_roi(centre, 0)], verbose=False)
    s = prepare(src / "m.tif", tmp_path / "out", cell_diameter, duration)
    scaled = tifffile.imread(tmp_path / "out" / "m.tif")
    (roi,) = read_rois(tmp_path / "out" / "m_cell_division.zip")
    peak = np.unravel_index(np.argmax(scaled), scaled.shape)
    assert list(scaled.shape) == s["shape_out"]
    assert max(abs(a - b) for a, b in zip(roi, peak)) <= 1, (roi, peak, scaled.shape)


def test_no_rescaling_when_scale_matches(tmp_path):
    movie = blob_movie()
    src = tmp_path / "src"
    src.mkdir()
    tifffile.imwrite(src / "m.tif", movie)
    write_rois(src / "m_cell_death.zip", [create_roi((20, 61, 177), 0)], verbose=False)
    s = prepare(src / "m.tif", tmp_path / "out", cell_diameter=28, extrusion_duration=4.5)  # within 30 %
    assert s["ratio_xy"] == 1.0 and s["ratio_t"] == 1.0
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "out" / "m.tif"), movie)
    assert read_rois(tmp_path / "out" / "m_cell_death.zip") == [(20, 61, 177)]


def test_roi_discovery_summary_and_safeguards(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    tifffile.imwrite(src / "m.tif", blob_movie())
    tifffile.imwrite(src / "other.tif", blob_movie())
    for name in ("m_cell_division.zip", "m_cell_delamination.zip", "m_cell_division.bak.zip",
                 "other_cell_death.zip"):
        write_rois(src / name, [create_roi((10, 50, 50), 0), create_roi((30, 100, 200), 0)], verbose=False)
    assert [p.name for p in find_roi_files(src / "m.tif", src)] == ["m_cell_delamination.zip", "m_cell_division.zip"]

    s = prepare(src / "m.tif", tmp_path / "out", cell_diameter=50)
    assert s["rois"] == {"m_cell_delamination.zip": 2, "m_cell_division.zip": 2}
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "m.prepare.json", "m.tif", "m_cell_delamination.zip", "m_cell_division.zip"]
    assert json.loads((tmp_path / "out" / "m.prepare.json").read_text())["ratio_xy"] == 0.5
    # the original files are untouched
    assert read_rois(src / "m_cell_division.zip") == [(10, 50, 50), (30, 100, 200)]
    with pytest.raises(ValueError, match="original movie"):
        prepare(src / "m.tif", src, cell_diameter=50)
    with pytest.raises(ValueError, match="T, Y, X"):
        tifffile.imwrite(src / "flat.tif", np.zeros((10, 10), np.uint8))
        prepare(src / "flat.tif", tmp_path / "out2", cell_diameter=50)


def test_prepare_cli_and_trainability(tmp_path, capsys):
    src = tmp_path / "src"
    src.mkdir()
    movie = blob_movie(shape=(40, 260, 360), centre=(20, 130, 180))
    tifffile.imwrite(src / "m.tif", movie)
    write_rois(src / "m_cell_division.zip", [create_roi((20, 130, 180), 0)], verbose=False)
    assert main(["-q", "prepare", str(src / "m.tif"), "-o", str(tmp_path / "train"),
                 "--cell-diameter", "50", "--rois", str(src / "m_cell_division.zip")]) == 0
    assert "zoom xy 0.500" in capsys.readouterr().out
    from dextrusion.config import DeXConfig
    from dextrusion.data.dataset import build_datasets

    cfg = DeXConfig(ncat=2, catnames=["", "_cell_division.zip"])
    tr, va = build_datasets(tmp_path / "train", cfg, val_ratio=0.0, naug=3, add_nothing_windows=0, seed=0)
    assert len(tr) > 0 and any(s.cat == 1 for s in tr.samples)  # the scaled ROI yields windows
    with pytest.raises(SystemExit):
        main(["-q", "prepare", str(src / "m.tif"), "-o", str(src), "--cell-diameter", "50"])
