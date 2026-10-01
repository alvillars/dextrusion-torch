"""Small deterministic movie used by the end-to-end parity test (no real data is shipped)."""

import numpy as np
from scipy.ndimage import gaussian_filter


def make_movie(t: int = 36, size: int = 128, seed: int = 3) -> np.ndarray:
    """uint16 (T, Y, X): epithelium-like texture with a few shrinking and growing bright spots."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    base = gaussian_filter(rng.random((size, size)), 2.5)
    base = (base - base.min()) / np.ptp(base)
    events = [(10, 40, 45, 0), (22, 90, 80, 1), (14, 70, 30, 0), (28, 35, 95, 1)]  # t, y, x, kind
    movie = np.empty((t, size, size), np.float32)
    for i in range(t):
        f = 0.55 * base + 0.08 * rng.random((size, size))
        for t0, y0, x0, kind in events:
            dt = i - t0
            if kind == 0:  # extrusion-like: spot shrinks then vanishes
                rad = 9 - 1.4 * dt if dt >= -4 else 0
            else:  # division-like: spot grows
                rad = 3 + 0.9 * (dt + 6) if -6 <= dt < 6 else 0
            if rad > 0.5:
                f += 0.9 * np.exp(((yy - y0) ** 2 + (xx - x0) ** 2) / (-2 * rad**2))
        movie[i] = f
    movie = (movie - movie.min()) / np.ptp(movie)
    return (movie * 4000 + 100).astype(np.uint16)


def make_probamap(seed: int = 5, shape=(3, 40, 96, 96)) -> np.ndarray:
    """uint8 probability maps with touching / separate blobs, to exercise the watershed split."""
    rng = np.random.default_rng(seed)
    nc, t, h, w = shape
    tt, yy, xx = np.mgrid[:t, :h, :w]
    out = np.zeros(shape, np.float32)
    for c in range(nc):
        for _ in range(7):
            ct, cy, cx = rng.integers(4, t - 4), rng.integers(10, h - 10), rng.integers(10, w - 10)
            st, sy = rng.uniform(1.5, 3.5), rng.uniform(4, 8)
            blob = np.exp(-((tt - ct) ** 2 / (2 * st**2) + ((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sy**2)))
            out[c] = np.maximum(out[c], blob * rng.uniform(150, 255))
    return out.astype(np.uint8)
