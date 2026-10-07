"""Estimate the typical cell spacing of a movie from its images (no annotations needed).

Spacing is the center-to-center distance of neighbouring cells in pixels: the position of the first
clear maximum of the radial autocorrelation of a high-passed image tile. It is computed on many
tiles of a few evenly spaced frames and the **median** over tiles is the estimate. Tiles that are
partly outside the image (zero pixels, e.g. outside an unwrapped surface) are skipped, and tiles
without a clear peak (blurred, no regular cell pattern) do not vote.

Limits to keep in mind:

* The autocorrelation of the tapered tile is divided by that of the taper, so a small tile does not
  pull the peak to smaller lags.
* The high-pass scale is fixed (12 px). On crisp images (synthetic cells, the Zenodo movies) the
  estimate does not depend on it. On blurred, irregular projections it does: there is no sharp
  cell scale, and the value moves by about +-15 % with the high-pass scale. Treat it as the typical
  cell size of the movie, not as a measurement to the pixel.
* Cell size changes over time in a growing tissue; ``trend`` reports the last third over the first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


class CellSizeError(ValueError):
    """The spacing could not be estimated; give the diameter explicitly."""


def _radial_mean(img: np.ndarray, centre: int, rmax: int) -> np.ndarray:
    yy, xx = np.indices(img.shape)
    r = np.rint(np.hypot(yy - centre, xx - centre)).astype(int)
    keep = r <= rmax
    return np.bincount(r[keep], img[keep], minlength=rmax + 1) / np.maximum(
        np.bincount(r[keep], minlength=rmax + 1), 1)


def _window_autocorrelation(window: np.ndarray, pad: int) -> np.ndarray:
    ac = np.fft.fftshift(np.fft.irfft2(np.abs(np.fft.rfft2(window, s=(pad, pad))) ** 2, s=(pad, pad)))
    return ac / ac.max()


def _first_clear_maximum(profile: np.ndarray, min_prominence: float) -> float:
    """Position of the first clear maximum after the first minimum of the (3-point smoothed) profile.

    "Clear" means it rises at least ``min_prominence`` above the lowest value since the minimum, so
    the small wiggles of a noisy profile are not taken for the neighbour peak. NaN if there is none.
    """
    p = np.convolve(profile, np.ones(3) / 3, mode="same")
    i = 2
    while i < len(p) - 1 and not (p[i] < p[i - 1] and p[i] <= p[i + 1]):
        i += 1
    low = p[i]
    j = i + 1
    while j < len(p) - 1:
        low = min(low, p[j])
        if p[j] > p[j - 1] and p[j] >= p[j + 1] and p[j] - low >= min_prominence:
            break
        j += 1
    if j >= len(p) - 1:
        return float("nan")
    denom = p[j - 1] - 2 * p[j] + p[j + 1]
    shift = 0.5 * (p[j - 1] - p[j + 1]) / denom if denom != 0 else 0.0  # parabolic refinement
    return float(j + np.clip(shift, -0.5, 0.5))


class _Tiler:
    """High-pass, taper and autocorrelation of square tiles of one size (windows computed once)."""

    def __init__(self, tile: int, sigma: float, min_prominence: float):
        self.tile, self.pad, self.sigma, self.min_prominence = tile, 2 * tile, sigma, min_prominence
        taper = np.hanning(tile).astype(np.float32)
        self.window = np.outer(taper, taper)
        self.window_ac = np.maximum(_window_autocorrelation(self.window, self.pad), 0.25)

    def spacing(self, t: np.ndarray, valid: np.ndarray) -> float:
        v = valid.astype(np.float32)
        background = gaussian_filter(t * v, self.sigma) / np.maximum(gaussian_filter(v, self.sigma), 1e-3)
        hp = (t - background) * v * self.window  # invalid pixels do not leak into the mean
        power = np.abs(np.fft.rfft2(hp, s=(self.pad, self.pad))) ** 2
        ac = np.fft.fftshift(np.fft.irfft2(power, s=(self.pad, self.pad))) / self.window_ac
        profile = _radial_mean(ac, self.pad // 2, self.tile // 2)
        if not profile[2] > 0:
            return float("nan")
        return _first_clear_maximum(profile / profile[2], self.min_prominence)


def estimate_spacing(frames: list[np.ndarray], tile: int = 192, sigma: float = 12.0,
                     min_valid: float = 0.9, min_prominence: float = 0.10, min_tiles: int = 10,
                     min_fraction: float = 0.4, min_spacing: float = 8.0) -> dict:
    """Median cell spacing (px) over the tiles of a list of 2D frames; see the module docstring.

    :param frames: 2D arrays in time order (zeros are treated as outside the image)
    :param tile: tile size in pixels (reduced to fit small frames)
    :param sigma: Gaussian high-pass scale in pixels
    :param min_valid: tiles with a smaller fraction of non-zero pixels are skipped
    :param min_prominence: height of the neighbour peak above the first minimum, relative to the
        autocorrelation at a lag of 2 px
    :param min_tiles: at least this many tiles must show a peak
    :param min_fraction: and at least this fraction of the valid tiles
    :param min_spacing: a median below this (px) is pixel-scale noise, not cells
    :return: ``spacing`` (median), ``q25``, ``q75``, ``trend`` (median of the last third of the frames
        over the first third), ``n_tiles`` (with a peak), ``n_valid`` (tested), ``n_frames``
    :raises CellSizeError: too few tiles show a cell pattern, or the median is pixel-scale noise
    """
    h, w = frames[0].shape
    tile = int(min(tile, h, w)) // 2 * 2
    tiler = _Tiler(tile, sigma, min_prominence)
    per_frame, n_valid = [], 0
    for f in frames:
        f = np.asarray(f, dtype=np.float32)
        values = []
        for y in range(0, h - tile + 1, tile):
            for x in range(0, w - tile + 1, tile):
                t = f[y:y + tile, x:x + tile]
                valid = t != 0  # zero = outside the image (valid pixels of float movies can be negative)
                if valid.mean() < min_valid:
                    continue
                n_valid += 1
                values.append(tiler.spacing(t, valid))
        per_frame.append(np.array(values))
    found = np.concatenate([v[np.isfinite(v)] for v in per_frame])
    if len(found) < min_tiles or len(found) < min_fraction * n_valid:
        raise CellSizeError(f"only {len(found)} of {n_valid} tiles show a cell pattern; "
                            "give --cell-diameter explicitly")
    if np.median(found) < min_spacing:
        raise CellSizeError(f"the median spacing ({np.median(found):.1f} px) is pixel-scale noise, not cells; "
                            "give --cell-diameter explicitly")
    third = max(1, len(frames) // 3)
    early = np.concatenate([v[np.isfinite(v)] for v in per_frame[:third]])
    late = np.concatenate([v[np.isfinite(v)] for v in per_frame[-third:]])
    trend = float(np.median(late) / np.median(early)) if len(early) and len(late) else float("nan")
    return {"spacing": float(np.median(found)), "q25": float(np.percentile(found, 25)),
            "q75": float(np.percentile(found, 75)), "trend": trend, "n_tiles": len(found),
            "n_valid": n_valid, "n_frames": len(frames)}


def estimate_from_file(path: str | Path, n_frames: int = 12, tile: int = 192) -> dict:
    """Estimate the spacing of a ``(T, Y, X)`` tif from ``n_frames`` evenly spaced frames.

    One page per frame is the usual layout and only the sampled pages are read; a file whose frames
    are not separate pages (a short stack stored as colour planes) is read whole.
    """
    with tifffile.TiffFile(path) as tf:
        shape = tf.series[0].shape
        if len(shape) != 3:
            raise ValueError(f"{path}: expected a (T, Y, X) movie, got shape {shape}")
        idx = np.unique(np.linspace(0, shape[0] - 1, min(n_frames, shape[0])).astype(int))
        if len(tf.pages) == shape[0]:
            frames = [tf.pages[int(k)].asarray() for k in idx]
        else:
            stack = tf.asarray()
            frames = [stack[int(k)] for k in idx]
    out = estimate_spacing(frames, tile=tile)
    out["frames"] = idx.tolist()
    return out
