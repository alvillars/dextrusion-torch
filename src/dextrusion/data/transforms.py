"""Training augmentations (MONAI transforms) on one window ``(1, T, H, W)``, values in 0-255.

Same augmentations and probabilities as the original generator:

* vertical / horizontal flips, p = 0.3 each (all samples);
* for samples that are augmented copies only (``noisy``): whole-window gamma (p .25), gamma on
  one random frame (p .3), isotropic 3D Gaussian blur (p .25), smoothed Gaussian noise (p .4),
  random square set to 0 (p .15) and random square incremented by 1 (p .15).

Intensities are in uint8 units (0-255) when these transforms run, exactly as in the original,
so the "noise" and "+1 square" perturbations are small there too.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.transforms import (
    AdjustContrast,
    Compose,
    GaussianSmooth,
    MapTransform,
    RandAdjustContrast,
    RandFlipd,
    RandomizableTransform,
)


class RandFrameGamma(RandomizableTransform):
    """Gamma in [0, gamma_max] on one random frame (simulates a dark / bright frame)."""

    def __init__(self, prob: float = 0.3, gamma_max: float = 1.5):
        RandomizableTransform.__init__(self, prob)
        self.gamma_max = gamma_max

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        self.randomize(None)
        if not self._do_transform:
            return img
        gamma = float(self.gamma_max * self.R.uniform(0, 1))
        frame = int(self.R.randint(0, img.shape[1]))
        out = img.clone()
        out[:, frame] = AdjustContrast(max(gamma, 1e-3))(img[:, frame : frame + 1])[:, 0]
        return out


class RandIsoBlur(RandomizableTransform):
    """3D Gaussian blur over (T, H, W) with one sigma in [0, sigma_max]."""

    def __init__(self, prob: float = 0.25, sigma_max: float = 1.5):
        RandomizableTransform.__init__(self, prob)
        self.sigma_max = sigma_max

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        self.randomize(None)
        if not self._do_transform:
            return img
        sigma = max(float(self.sigma_max * self.R.uniform(0, 1)), 1e-3)
        return GaussianSmooth(sigma=sigma)(img)


class RandSmoothedNoise(RandomizableTransform):
    """Add N(0, 1) noise (in intensity units) blurred with sigma in [0, sigma_max]."""

    def __init__(self, prob: float = 0.4, sigma_max: float = 1.5):
        RandomizableTransform.__init__(self, prob)
        self.sigma_max = sigma_max

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        self.randomize(None)
        if not self._do_transform:
            return img
        noise = torch.from_numpy(self.R.normal(0, 1, img.shape).astype(np.float32))
        sigma = max(float(self.sigma_max * self.R.uniform(0, 1)), 1e-3)
        return img + GaussianSmooth(sigma=sigma)(noise)


class RandSquare(RandomizableTransform):
    """Random square (side 1-11 px, same position on every frame): zeroed, or incremented by 1."""

    def __init__(self, prob: float, mode: str):
        RandomizableTransform.__init__(self, prob)
        assert mode in ("zero", "add")
        self.mode = mode

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        self.randomize(None)
        if not self._do_transform:
            return img
        _, _, h, w = img.shape
        side = int(self.R.randint(1, 12))
        iy = int(self.R.randint(0, h - 1 - side))
        ix = int(self.R.randint(0, w - 1 - side))
        out = img.clone()
        sq = out[:, :, iy : iy + side, ix : ix + side]
        if self.mode == "zero":
            sq.zero_()
        else:
            sq.add_(1)
        return out


def noise_augmentations() -> Compose:
    return Compose(
        [
            RandAdjustContrast(prob=0.25, gamma=(0.6, 1.4)),
            RandFrameGamma(prob=0.3, gamma_max=1.5),
            RandIsoBlur(prob=0.25, sigma_max=1.5),
            RandSmoothedNoise(prob=0.4, sigma_max=1.5),
            RandSquare(0.15, "zero"),
            RandSquare(0.15, "add"),
        ]
    )


class ApplyIfNoisyd(MapTransform):
    """Apply ``transform`` to ``key`` only when ``data['noisy']`` is true."""

    def __init__(self, key: str, transform: Compose):
        super().__init__(keys=[key])
        self.key, self.transform = key, transform

    def set_random_state(self, seed=None, state=None):
        self.transform.set_random_state(seed, state)
        return self

    def __call__(self, data):
        d = dict(data)
        if d["noisy"]:
            d[self.key] = self.transform(d[self.key])
        return d


def train_transforms(augment_noise: bool = True) -> Compose:
    """Flips for every training sample, plus noise augmentations for augmented copies."""
    steps = [
        RandFlipd(keys="image", prob=0.3, spatial_axis=1),
        RandFlipd(keys="image", prob=0.3, spatial_axis=2),
    ]
    if augment_noise:
        steps.append(ApplyIfNoisyd("image", noise_augmentations()))
    return Compose(steps)
