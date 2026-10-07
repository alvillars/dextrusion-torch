import json

import numpy as np
import pytest
import tifffile

from dextrusion.cli import main
from dextrusion.io import create_roi, read_rois, write_rois
from dextrusion.reverse import reverse, reverse_point

SHAPE = (40, 60, 80)
CENTRE = (12, 30, 41)


def blob_movie(shape=SHAPE, centre=CENTRE, sigma=(2.0, 4.0, 4.0)):
    t, y, x = np.indices(shape)
    g = np.exp(-sum(((c - m) / s) ** 2 / 2 for c, m, s in zip((t, y, x), centre, sigma)))
    return (g * 3000 + 50).astype(np.uint16)


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    tifffile.imwrite(src / "m.tif", blob_movie())
    write_rois(src / "m_cell_death.zip", [create_roi(CENTRE, 0), create_roi((30, 5, 6), 0)], verbose=False)
    write_rois(src / "m_cell_division.zip", [create_roi((20, 20, 20), 0)], verbose=False)
    write_rois(src / "m_cell_sop.zip", [create_roi((21, 21, 21), 0)], verbose=False)
    write_rois(src / "m_nothing.zip", [create_roi((7, 8, 9), 0)], verbose=False)
    write_rois(src / "m_cell_death.bak.zip", [create_roi((1, 1, 1), 0)], verbose=False)
    return src


def test_reverse_point_is_an_involution():
    assert reverse_point((0, 3, 4), 40) == (39, 3, 4)
    assert reverse_point((39, 3, 4), 40) == (0, 3, 4)
    assert reverse_point(reverse_point((12, 30, 41), 40), 40) == (12, 30, 41)


def test_roi_follows_the_blob_in_the_reversed_movie(source, tmp_path):
    out = tmp_path / "out"
    s = reverse(source / "m.tif", out)
    rev = tifffile.imread(out / "m_rev.tif")
    (first, _) = read_rois(out / "m_rev_cell_delamination.zip")
    peak = np.unravel_index(np.argmax(rev), rev.shape)
    assert rev.shape == SHAPE and list(rev.shape) == s["shape"]
    assert first == (SHAPE[0] - 1 - CENTRE[0], CENTRE[1], CENTRE[2])
    assert max(abs(a - b) for a, b in zip(first, peak)) <= 1, (first, peak)
    np.testing.assert_array_equal(rev, tifffile.imread(source / "m.tif")[::-1])


def test_death_becomes_delamination_nothing_is_reversed_and_the_rest_is_dropped(source, tmp_path):
    out = tmp_path / "out"
    s = reverse(source / "m.tif", out)
    assert sorted(p.name for p in out.glob("*.zip")) == ["m_rev_cell_delamination.zip", "m_rev_nothing.zip"]
    assert s["rois"] == {"m_rev_cell_delamination.zip": 2, "m_rev_nothing.zip": 1}
    assert sorted(s["dropped"]) == ["m_cell_division.zip", "m_cell_sop.zip"]  # .bak.zip is not even listed
    assert read_rois(out / "m_rev_nothing.zip") == [(SHAPE[0] - 1 - 7, 8, 9)]
    assert json.loads((out / "m_rev.reverse.json").read_text())["death_as"] == "_cell_delamination.zip"


def test_reversing_twice_gives_back_the_movie_and_the_rois(source, tmp_path):
    once = reverse(source / "m.tif", tmp_path / "once", death_as="_cell_death.zip")
    assert "m_rev_cell_death.zip" in once["rois"]
    reverse(tmp_path / "once" / "m_rev.tif", tmp_path / "twice")
    twice = tmp_path / "twice"
    np.testing.assert_array_equal(tifffile.imread(twice / "m_rev_rev.tif"), tifffile.imread(source / "m.tif"))
    assert read_rois(twice / "m_rev_rev_cell_delamination.zip") == read_rois(source / "m_cell_death.zip")


def test_output_may_be_the_movies_own_folder_and_originals_are_untouched(source):
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    reverse(source / "m.tif", source)
    assert (source / "m_rev.tif").is_file() and (source / "m_rev_cell_delamination.zip").is_file()
    assert all((source / n).read_bytes() == b for n, b in before.items())


def test_refuses_a_bad_suffix(source, tmp_path):
    with pytest.raises(ValueError, match="ending in .zip"):
        reverse(source / "m.tif", tmp_path / "out", death_as="_cell_delamination")


def test_cli_reverse_is_accepted(source, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["reverse", str(source / "m.tif"), "-o", str(out), "--death-as", "_cell_emerging.zip"]) == 0
    assert (out / "m_rev.tif").is_file() and (out / "m_rev_cell_emerging.zip").is_file()
    assert "40 frames reversed" in capsys.readouterr().out
