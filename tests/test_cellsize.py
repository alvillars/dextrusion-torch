import argparse

import numpy as np
import pytest
import tifffile
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import cKDTree

from dextrusion.cellsize import CellSizeError, estimate_from_file, estimate_spacing
from dextrusion.cli import main, parse_diameter, resolve_cell_diameter

SHAPE = (384, 576)


def membranes(a, shape=SHAPE, seed=0, width=1.5, jitter=0.15):
    """Bright membranes between the cells of a jittered hexagonal lattice with lattice constant ``a`` px."""
    rng = np.random.default_rng(seed)
    h, w = shape
    pts = np.array([(y, x + (a / 2 if j % 2 else 0))
                    for j, y in enumerate(np.arange(-2 * a, h + 2 * a, a * np.sqrt(3) / 2))
                    for x in np.arange(-2 * a, w + 2 * a, a)])
    pts = pts + rng.normal(0, jitter * a, pts.shape)
    yy, xx = np.indices(shape)
    d, _ = cKDTree(pts).query(np.c_[yy.ravel(), xx.ravel()], k=2)
    return np.exp(-(((d[:, 1] - d[:, 0]).reshape(shape)) / width) ** 2 / 2) * 2000 + 5000


def frames(a, n=3, **kw):
    return [membranes(a, seed=k, **kw) for k in range(n)]


@pytest.mark.parametrize("a", [16, 24, 40])
def test_spacing_of_synthetic_cells(a):
    est = estimate_spacing(frames(a))
    assert est["spacing"] == pytest.approx(a, rel=0.08), est
    assert est["n_tiles"] == est["n_valid"] >= 10


def test_estimate_scales_with_the_image():
    est = estimate_spacing([zoom(f, 1.5, order=1) for f in frames(24)])
    assert est["spacing"] == pytest.approx(36, rel=0.08)


def test_blurred_half_and_zero_border_do_not_matter():
    def half_blur(f):
        g = f.copy()
        g[:, g.shape[1] // 2:] = gaussian_filter(f, 6)[:, g.shape[1] // 2:]
        return g

    def border(f):
        g = f.copy()
        g[:, -150:] = 0  # outside the image, e.g. beyond an unwrapped surface
        return g

    assert estimate_spacing([half_blur(f) for f in frames(32)])["spacing"] == pytest.approx(32, rel=0.08)
    est = estimate_spacing([border(f) for f in frames(32, n=6)])
    assert est["spacing"] == pytest.approx(32, rel=0.08)
    assert est["n_valid"] == 24  # 4 of the 6 tiles per frame: the tiles on the empty part were skipped


def test_float_movies_with_negative_values_are_handled():
    est = estimate_spacing([f - 5500 for f in frames(24)])  # valid pixels below zero (float movies)
    assert est["spacing"] == pytest.approx(24, rel=0.08)


def test_structureless_images_raise():
    rng = np.random.default_rng(0)
    with pytest.raises(CellSizeError):
        estimate_spacing([rng.normal(size=SHAPE) * 100 + 5000 for _ in range(3)])


def test_estimate_from_file_reads_a_tif(tmp_path):
    tifffile.imwrite(tmp_path / "m.tif", np.stack(frames(24, n=6)).astype(np.uint16))
    est = estimate_from_file(tmp_path / "m.tif", n_frames=4)
    assert est["spacing"] == pytest.approx(24, rel=0.08)
    assert est["frames"] == [0, 1, 3, 5] and 0.9 < est["trend"] < 1.1
    tifffile.imwrite(tmp_path / "short.tif", np.stack(frames(24, n=3)).astype(np.uint16))  # stored as colour planes
    assert estimate_from_file(tmp_path / "short.tif")["spacing"] == pytest.approx(24, rel=0.1)
    tifffile.imwrite(tmp_path / "flat.tif", np.zeros(SHAPE, np.uint16))
    with pytest.raises(ValueError, match="expected a"):
        estimate_from_file(tmp_path / "flat.tif")


def test_parse_diameter():
    assert parse_diameter("auto") == "auto" and parse_diameter("AUTO") == "auto"
    assert parse_diameter("40") == 40.0
    for bad in ("big", "-5", "0"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_diameter(bad)


@pytest.fixture
def movie40(tmp_path):
    path = tmp_path / "m.tif"
    tifffile.imwrite(path, np.stack(frames(40, n=6)).astype(np.uint16))
    return path


def test_estimate_size_cli(movie40, capsys):
    assert main(["estimate-size", str(movie40), "--frames", "4"]) == 0
    out = capsys.readouterr().out
    assert "m.tif: 3" in out or "m.tif: 4" in out  # about 40 px
    assert "detect rescales by 0.6" in out


def test_estimate_size_cli_reports_failure(tmp_path, capsys):
    rng = np.random.default_rng(0)
    tifffile.imwrite(tmp_path / "noise.tif", (rng.normal(size=(6, *SHAPE)) * 100 + 5000).astype(np.uint16))
    assert main(["estimate-size", str(tmp_path / "noise.tif")]) == 1
    assert "give --cell-diameter explicitly" in capsys.readouterr().out


def test_auto_resolves_per_movie_and_numbers_pass_through(movie40, tmp_path):
    assert resolve_cell_diameter(37.0, movie40) == 37.0
    assert resolve_cell_diameter("auto", movie40) == pytest.approx(40, rel=0.08)
    rng = np.random.default_rng(0)
    tifffile.imwrite(tmp_path / "noise.tif", (rng.normal(size=(6, *SHAPE)) * 100 + 5000).astype(np.uint16))
    with pytest.raises(SystemExit, match="noise.tif"):
        resolve_cell_diameter("auto", tmp_path / "noise.tif")


def test_prepare_accepts_auto(movie40, tmp_path):
    out = tmp_path / "out"
    assert main(["prepare", str(movie40), "-o", str(out), "--cell-diameter", "auto"]) == 0
    import json
    ratio = json.loads((out / "m.prepare.json").read_text())["ratio_xy"]
    assert ratio == pytest.approx(25 / 40, rel=0.1)


def test_detect_resolves_auto_for_each_movie(movie40, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("dextrusion.run.detect_movie", lambda movie, models, outdir, opts: seen.append(
        (movie.name, opts.cell_diameter)))
    other = tmp_path / "n.tif"
    tifffile.imwrite(other, np.stack(frames(24, n=6)).astype(np.uint16))
    assert main(["detect", str(movie40), str(other), "-m", "models", "--cell-diameter", "auto"]) == 0
    assert [n for n, _ in seen] == ["m.tif", "n.tif"]
    assert seen[0][1] == pytest.approx(40, rel=0.08) and seen[1][1] == pytest.approx(24, rel=0.08)
    seen.clear()
    assert main(["detect", str(movie40), "-m", "models", "--cell-diameter", "33"]) == 0
    assert seen == [("m.tif", 33.0)]
