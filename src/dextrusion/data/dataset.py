"""Training windows from tif movies + ImageJ ROI files.

Replaces the original ``MovieGeneratorFromROI``. Folder layout (unchanged): for every
``movie.tif`` (T, Y, X), ROI zips named ``movie<catname>`` (``movie_cell_death.zip``...), and an
optional ``movie_nothing.zip`` with hand-picked "no event" windows (typical false positives).

Sampling follows the original: per movie, classes are balanced (a class below 0.75x the largest
one is topped up to 0.85x), every ROI gives one jittered window plus randomly re-jittered copies
up to ``naug`` times the class count, and "nothing" windows are drawn at random positions that
contain no ROI. Windows are extracted lazily from the movies instead of via temporary tif files.

Difference that matters: the train/validation split is made **per source ROI** before
augmentation, so jittered copies of one event never end up on both sides (the original shuffled
all windows, leaking validation into training). Validation windows are not augmented.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from ..config import DeXConfig
from ..io import read_rois
from .transforms import train_transforms

log = logging.getLogger("dextrusion")

NOTHING_FACTOR = 1.2  # more "nothing" windows than events (slight imbalance on purpose)


@dataclass(frozen=True)
class Sample:
    movie: int
    z: int
    y: int
    x: int
    cat: int
    noisy: bool  # augmented copy: eligible for the noise augmentations


def window_fits(shape, config: DeXConfig, z: int, y: int, x: int) -> bool:
    (nf0, nf1), (hy, hx) = config.nframes, config.half_size
    return (
        x - hx >= 0 and x + hx + 1 <= shape[2]
        and y - hy >= 0 and y + hy + 1 <= shape[1]
        and z - nf0 >= 0 and z + nf1 <= shape[0]
    )


def contains(config: DeXConfig, roi, z, y, x) -> bool:
    """Does the window centred on (z, y, x) contain the ROI point? (original bounds, kept)"""
    (nf0, nf1), (hy, hx) = config.nframes, config.half_size
    if roi[0] < z - nf0 or roi[0] > z + nf1:
        return False
    if roi[1] < y - hy or roi[1] > y + hy + 1:
        return False
    return x - hx < roi[2] < x + hx + 1


def jitter(roi, config: DeXConfig, rng: np.random.Generator):
    """ROI position with the original random shift (0.2 x half window in xy, +-2 frames)."""
    frame, y, x = roi
    hy, hx = config.half_size
    return (
        math.floor(frame + rng.uniform(-1.0, 1.0) * 2),
        math.floor(y + rng.uniform(-1.0, 1.0) * hy * 0.2),
        math.floor(x + rng.uniform(-1.0, 1.0) * hx * 0.2),
    )


def extract_window(movie: np.ndarray, config: DeXConfig, z: int, y: int, x: int) -> np.ndarray:
    (nf0, nf1), (hy, hx) = config.nframes, config.half_size
    return movie[z - nf0 : z + nf1, y - hy : y + hy + 1, x - hx : x + hx + 1]


def to_uint8(window: np.ndarray) -> np.ndarray:
    """Per-window min-max scaling to uint8 (what the network sees at inference as well)."""
    w = window.astype(np.float64)
    n = w - w.min()
    d = w.max() - w.min()
    return np.uint8((n / d if d > 0 else n) * 255)


def _sample_movie(imgname: Path, idx: int, shape, config, naug, add_nothing, val_ratio, rng,
                  balance=True, split_seed=0):
    """All (train, val) samples of one movie.

    ``rng`` drives the jitter and the random "nothing" windows. Which annotated events go to
    validation is drawn from a separate generator seeded by ``(split_seed, idx)``, so sampling the
    same movie again with another ``rng`` (oversampling) keeps every event on the same side.
    """
    split_rng = np.random.default_rng([split_seed, idx])
    train: list[Sample] = []
    val: list[Sample] = []
    base = imgname.with_suffix("")
    nz, ny, nx = shape
    (nf0, nf1), (hy, hx) = config.nframes, config.half_size

    def put(sample: Sample, is_val: bool):
        (val if is_val else train).append(sample)

    rois_by_cat: dict[int, list] = {}
    for cat in range(1, config.ncat):
        f = Path(str(base) + config.catnames[cat])
        if f.is_file():
            rois_by_cat[cat] = read_rois(f)
    counts = np.zeros(config.ncat)
    for cat, rois in rois_by_cat.items():
        counts[cat] = len(rois)
    all_rois = [r for rois in rois_by_cat.values() for r in rois]
    nmax = counts.max()

    for cat, rois in rois_by_cat.items():
        if counts[cat] < 0.75 * nmax and balance:
            counts[cat] = 0.85 * nmax
        in_val = split_rng.random(len(rois)) < val_ratio  # split decided per source ROI
        done = 0
        for i, roi in enumerate(rois):
            z, y, x = jitter(roi, config, rng)
            if window_fits(shape, config, z, y, x):
                put(Sample(idx, z, y, x, cat, False), in_val[i])
                done += 1
        target, limit, it = counts[cat] * naug, counts[cat] * naug * 2000, 0
        while done < target and it < limit:
            it += 1
            i = int(rng.integers(len(rois)))
            z, y, x = jitter(rois[i], config, rng)
            if window_fits(shape, config, z, y, x):
                put(Sample(idx, z, y, x, cat, True), in_val[i])
                done += 1

    # "nothing" windows at random positions without any ROI
    target = nmax * NOTHING_FACTOR * naug
    limit, it, done = target * 3000, 0, 0
    while done < target and it < limit:
        it += 1
        x = hx + int(rng.integers(nx - 2 * hx))
        y = hy + int(rng.integers(ny - 2 * hy))
        z = nf0 + int(rng.integers(nz - nf1))
        if not any(contains(config, r, z, y, x) for r in all_rois) and window_fits(shape, config, z, y, x):
            put(Sample(idx, z, y, x, 0, bool(rng.random() < 0.5)), rng.random() < val_ratio)
            done += 1

    # hand-picked false positives (reinforcement)
    nothing = Path(str(base) + "_nothing.zip")
    if add_nothing > 1 and nothing.is_file():
        rois = read_rois(nothing)
        in_val = split_rng.random(len(rois)) < val_ratio
        for i, roi in enumerate(rois):
            z, y, x = jitter(roi, config, rng)
            if window_fits(shape, config, z, y, x):
                put(Sample(idx, z, y, x, 0, False), in_val[i])
        if naug > 1:
            target, limit, it, done = len(rois) * add_nothing, len(rois) * add_nothing * 1000, 0, 0
            while done < target and it < limit:
                it += 1
                i = int(rng.integers(len(rois)))
                z, y, x = jitter(rois[i], config, rng)
                if window_fits(shape, config, z, y, x):
                    put(Sample(idx, z, y, x, 0, True), in_val[i])
                    done += 1
    return train, val


class WindowDataset(Dataset):
    """Items are ``(window, label)``; window is float32 ``(T, 1, H, W)`` with values in 0-255.

    Use :func:`collate_windows` to normalise (per sample, or per batch for legacy behaviour).
    ``transform`` (a MONAI ``Compose`` of dict transforms on key ``image``) is the training
    augmentation; keep it ``None`` for validation / evaluation.
    """

    def __init__(self, movies: list[np.ndarray], samples: list[Sample], config: DeXConfig,
                 transform=None):
        self.movies, self.samples, self.config, self.transform = movies, samples, config, transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        win = to_uint8(extract_window(self.movies[s.movie], self.config, s.z, s.y, s.x))
        img = torch.from_numpy(win.astype(np.float32)).unsqueeze(0)  # (1, T, H, W)
        if self.transform is not None:
            img = self.transform({"image": img, "noisy": s.noisy})["image"]
        return torch.as_tensor(img).transpose(0, 1).contiguous(), s.cat  # (T, 1, H, W)


def collate_windows(batch, normalize: str = "sample"):
    """Stack and min-max normalise to [0, 1].

    ``normalize="batch"`` reproduces the original generator, which scaled with the min and max of
    the whole batch (so a sample's input depended on its batch mates).
    """
    x = torch.stack([b[0] for b in batch]).float()
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    if normalize == "batch":
        mn, mx = x.min(), x.max()
        x = (x - mn) / (mx - mn) if mx > mn else x - mn
    else:
        flat = x.flatten(1)
        mn, mx = flat.min(1).values, flat.max(1).values
        d = (mx - mn).clamp_min(1e-12)
        x = (x - mn.view(-1, 1, 1, 1, 1)) / d.view(-1, 1, 1, 1, 1)
    return x, y


def load_movies(data_path: str | Path) -> tuple[list[Path], list[np.ndarray]]:
    data_path = Path(data_path)
    if not data_path.is_dir():
        raise FileNotFoundError(f"folder {data_path} not found")
    paths = sorted(data_path.glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"no .tif movie in {data_path}")
    return paths, [tifffile.imread(p) for p in paths]


def build_datasets(data_path, config: DeXConfig, val_ratio: float = 0.2, naug: int = 1,
                   add_nothing_windows: int = 10, augment_noise: bool = True, seed: int = 0,
                   oversample: dict[str, int] | None = None):
    """Return ``(train_dataset, val_dataset)`` for a training folder. Deterministic given ``seed``.

    :param oversample: ``{movie name (without .tif): k}``. Those movies are sampled ``k`` times
        with independent jitter and random windows, so that a small annotated movie gets a fair
        share next to big datasets. Only the training windows are repeated; the validation
        windows come from the first pass and every annotated event stays on one side.
    """
    paths, movies = load_movies(data_path)
    oversample = dict(oversample or {})
    stems = [p.stem for p in paths]
    for name, k in oversample.items():
        if name not in stems:
            raise ValueError(f"--oversample: no movie named '{name}' in {data_path} "
                             f"(movies: {', '.join(stems)})")
        if int(k) != k or k < 1:
            raise ValueError(f"--oversample {name}={k}: the factor must be an integer >= 1")
    if add_nothing_windows <= 1:
        for p in paths:
            if Path(str(p.with_suffix("")) + "_nothing.zip").is_file():
                log.warning("%s_nothing.zip is ignored because add_nothing_windows=%d "
                            "(use a value > 1 to train on these hand-picked non-events)",
                            p.stem, add_nothing_windows)
    rng = np.random.default_rng(seed)
    train: list[Sample] = []
    val: list[Sample] = []
    for i, (p, m) in enumerate(zip(paths, movies)):
        if m.ndim != 3:
            raise ValueError(f"{p}: expected a (T, Y, X) movie, got shape {m.shape}")
        k = int(oversample.get(p.stem, 1))
        for rep in range(k):
            t, v = _sample_movie(p, i, m.shape, config, naug, add_nothing_windows, val_ratio, rng,
                                 split_seed=seed)
            train += t
            if rep == 0:
                val += v
                log.info("%s: %d train / %d validation windows%s", p.name, len(t), len(v),
                         f" (x{k} oversampled)" if k > 1 else "")
    tr = train_transforms(augment_noise and naug > 1)
    return (WindowDataset(movies, train, config, tr), WindowDataset(movies, val, config, None))
