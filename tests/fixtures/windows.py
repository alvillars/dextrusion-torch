"""Deterministic test windows shared by the golden-output generator and the parity tests.

Pure noise saturates every DeXNet to a constant output, so the set mixes noise with structured
windows (a shrinking / growing / flickering bright blob over smoothed texture) whose outputs are
not saturated and therefore actually test the weights.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

N_NOISE = 12
N_STRUCTURED = 48


def _structured(rng: np.random.Generator, kind: int) -> np.ndarray:
    t, h, w = 10, 45, 45
    yy, xx = np.mgrid[:h, :w]
    base = gaussian_filter(rng.random((h, w)), rng.uniform(1, 4))
    base = (base - base.min()) / (np.ptp(base) + 1e-9)
    out = np.empty((t, h, w), np.float32)
    for i in range(t):
        f = base * 0.6 + 0.1 * rng.random((h, w))
        r2 = (yy - 22) ** 2 + (xx - 22) ** 2
        rad = {0: 14 - 1.6 * i, 1: 4 + 1.2 * i}.get(kind, 8)
        amp = 1.0 if kind != 2 else rng.random()
        out[i] = f + 0.8 * np.exp(-r2 / (2 * max(rad, 0.5) ** 2)) * amp
    return (out - out.min()) / np.ptp(out)


def make_windows() -> np.ndarray:
    """``(N, 10, 45, 45, 1)`` float32 in [0, 1] (Keras layout)."""
    rng = np.random.default_rng(0)
    noise = rng.random((N_NOISE, 10, 45, 45), dtype=np.float32)
    rng = np.random.default_rng(1)
    structured = np.stack([_structured(rng, k % 3) for k in range(N_STRUCTURED)])
    return np.concatenate([noise, structured.astype(np.float32)])[..., None]
