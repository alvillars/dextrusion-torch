"""Sliding-window detection: movie -> per-class probability maps.

Faithful port of the original ``DeXtrusion.detect_events`` so that old DeXNets give the same
maps. Behaviours kept on purpose (all are visible in the results of already published work):

* window intensities are min-max scaled *per window*, truncated to uint8 and divided by 255;
* probabilities of a window are accumulated in a float16 map over a small block around the
  window centre (``DeXConfig.bounds``), with a uint8 counter, then averaged;
* with an ensemble of N models, model ``k`` is shifted by ``floor(k * dxy / N)`` frames and
  ``floor(k * dz / N)`` pixels (``dxy`` and ``dz`` are swapped w.r.t. what the names suggest);
  set ``legacy_ensemble_shift=False`` for the geometrically consistent shifts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom

from .config import DeXConfig
from .io import load_model, resolve_models
from .model import DeXNet

log = logging.getLogger("dextrusion")


@dataclass
class Tiling:
    """Window grid in the (padded) movie. Mirrors ``update_nwins`` / ``get_index``."""

    nz: int
    ny: int
    nx: int
    shiftz: int
    shiftxy: int
    dz: int
    dxy: int
    nframes: tuple[int, int]
    half_size: tuple[int, int]

    @classmethod
    def for_shape(cls, shape, config: DeXConfig, shiftz, shiftxy, dz, dxy) -> Tiling:
        hy, hx = config.half_size
        nx = ceil((shape[2] - hx * 2 - 1 - shiftxy) / dxy)
        ny = ceil((shape[1] - hy * 2 - 1 - shiftxy) / dxy)
        nz = ceil((shape[0] - sum(config.nframes) - shiftz) / dz)
        return cls(nz, ny, nx, shiftz, shiftxy, dz, dxy, config.nframes, config.half_size)

    @property
    def n_windows(self) -> int:
        return max(self.nz, 0) * max(self.ny, 0) * max(self.nx, 0)

    def centres(self, start: int, stop: int) -> np.ndarray:
        """``(n, 3)`` window centres ``(t, y, x)`` for window ids ``start:stop``."""
        g = np.arange(start, stop)
        t = self.nframes[0] + self.shiftz + ((g // (self.ny * self.nx)) % self.nz) * self.dz
        y = self.half_size[0] + self.shiftxy + ((g // self.nx) % self.ny) * self.dxy
        x = self.half_size[1] + self.shiftxy + (g % self.nx) * self.dxy
        return np.stack([t, y, x], axis=1)


def scale_windows(windows: np.ndarray) -> np.ndarray:
    """Per-window min-max scaling to uint8, same arithmetic (and dtype promotion) as the original."""
    axes = tuple(range(1, windows.ndim))
    mn = windows.min(axis=axes, keepdims=True)
    d = windows.max(axis=axes, keepdims=True) - mn
    den = np.where(d > 0, d, 1).astype(d.dtype)  # flat windows: (w - min) is already 0
    return np.uint8((windows - mn) / den * 255)


def extract_windows(img: np.ndarray, tiling: Tiling, start: int, stop: int) -> np.ndarray:
    nf0, nf1 = tiling.nframes
    hy, hx = tiling.half_size
    cs = tiling.centres(start, stop)
    wins = np.empty((len(cs), nf0 + nf1, 2 * hy + 1, 2 * hx + 1), dtype=img.dtype)
    for i, (t, y, x) in enumerate(cs):
        wins[i] = img[t - nf0 : t + nf1, y - hy : y + hy + 1, x - hx : x + hx + 1]
    return scale_windows(wins)


def resize_image(img: np.ndarray, ratioxy: float = 1, ratioz: float = 1) -> np.ndarray:
    return zoom(img, (ratioz, ratioxy, ratioxy))


def check_image(img: np.ndarray) -> None:
    if img.ndim != 3:
        raise ValueError(
            f"expected a gray-scale movie shaped (T, Y, X), got {img.shape} "
            "(check that it is not RGB / multichannel)"
        )


@dataclass
class ProbaMaps:
    """Result of a detection: uint8 probability maps for classes ``1..ncat-1``."""

    probamap: np.ndarray  # (ncat-1, T, Y, X) at the *rescaled* movie size
    init_shape: tuple[int, int, int]  # shape of the movie given by the user
    config: DeXConfig
    dz: int
    dxy: int

    @property
    def catnames(self) -> list[str]:
        return self.config.catnames


def _predict(model: DeXNet, windows_u8: np.ndarray, device, batch_size: int) -> np.ndarray:
    out = []
    for i in range(0, len(windows_u8), batch_size):
        chunk = windows_u8[i : i + batch_size]
        x = torch.from_numpy((chunk / 255).astype(np.float32)).unsqueeze(2).to(device)
        out.append(model.predict_proba(x).cpu().numpy())
    return np.concatenate(out)


def detect(
    img: np.ndarray,
    models: str | Path | list[str | Path],
    cell_diameter: float = 25,
    extrusion_duration: float = 4.5,
    dxy: int = 25,
    dz: int = 2,
    group_size: int = 4096,
    batch_size: int = 512,
    device: str | torch.device | None = None,
    legacy_ensemble_shift: bool = True,
) -> ProbaMaps:
    """Run DeXNet(s) on a ``(T, Y, X)`` movie and return the probability maps.

    :param models: a DeXNet dir (native or legacy Keras), a dir of DeXNets (ensemble), or a list
    :param group_size: windows extracted in memory at once (accumulation order is unaffected)
    :param batch_size: windows per forward pass
    """
    check_image(img)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    paths: list[Path] = []
    for m in models if isinstance(models, (list, tuple)) else [models]:
        paths.extend(resolve_models(m))
    loaded = [load_model(p, device) for p in paths]
    cfg0 = loaded[0][1]
    for _, c in loaded[1:]:
        if (c.ncat, c.nframes, c.half_size) != (cfg0.ncat, cfg0.nframes, cfg0.half_size):
            raise ValueError("all models of an ensemble must share ncat, nframes and half_size")

    init_shape = tuple(img.shape)
    log.info("Initial image shape: %s", init_shape)
    if abs(cell_diameter - cfg0.cell_diameter) > cfg0.cell_diameter * 0.3:
        img = resize_image(img, ratioxy=cfg0.cell_diameter / cell_diameter, ratioz=1)
    if abs(extrusion_duration - cfg0.extrusion_duration) > cfg0.extrusion_duration * 0.3:
        img = resize_image(img, ratioxy=1, ratioz=cfg0.extrusion_duration / extrusion_duration)
    scaled_shape = img.shape
    log.info("Rescaled image shape: %s", scaled_shape)

    nf0, nf1 = cfg0.nframes
    img = np.concatenate((np.repeat(img[:1], nf0, axis=0), img), axis=0)
    img = np.concatenate((img, np.repeat(img[-1:], nf1, axis=0)), axis=0)

    ncat = cfg0.ncat
    pred = np.zeros((ncat - 1,) + img.shape, dtype="float16")
    npred = np.zeros(img.shape, dtype="uint8")
    b0, b1, by, bx = cfg0.bounds

    nmod = len(loaded)
    for cmod, (model, cfg) in enumerate(loaded):
        if legacy_ensemble_shift:
            shiftz, shiftxy = floor(cmod * dxy / nmod), floor(cmod * dz / nmod)
        else:
            shiftz, shiftxy = floor(cmod * dz / nmod), floor(cmod * dxy / nmod)
        tiling = Tiling.for_shape(img.shape, cfg, shiftz, shiftxy, dz, dxy)
        if tiling.n_windows <= 0:
            raise ValueError(f"movie {init_shape} is too small for the window size of the model")
        log.info("Model %d/%d, shift (t=%d, xy=%d): %d windows", cmod + 1, nmod, shiftz, shiftxy,
                 tiling.n_windows)
        for start in range(0, tiling.n_windows, group_size):
            stop = min(start + group_size, tiling.n_windows)
            wins = extract_windows(img, tiling, start, stop)
            probs = _predict(model, wins, device, batch_size)
            cs = tiling.centres(start, stop)
            for (t, y, x), p in zip(cs, probs):
                sl = (slice(t - b0, t + b1), slice(y - by, y + by + 1), slice(x - bx, x + bx + 1))
                for c in range(ncat - 1):
                    # float16 accumulation, as in the original (numpy 1.x value-based casting)
                    pred[(c, *sl)] += np.float16(p[1 + c])
                npred[sl] += 1

    probamap = divide(pred, npred, nf0, nf1)
    return ProbaMaps(probamap, init_shape, loaded[-1][1], dz, dxy)


def divide(pred: np.ndarray, npred: np.ndarray, nf0: int, nf1: int) -> np.ndarray:
    """Average accumulated probabilities, crop the padding, convert to uint8 (0-255)."""
    npred = npred.copy()
    npred[npred == 0] = 1
    n = pred.shape[1]
    out = np.zeros((pred.shape[0], n - nf0 - nf1) + pred.shape[2:], dtype="uint8")
    for c in range(pred.shape[0]):
        out[c] = np.uint8(pred[c, nf0 : n - nf1] / npred[nf0 : n - nf1] * 255)
    return out
