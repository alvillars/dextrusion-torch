"""High level helpers: detect on a movie file and write the results next to it."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader

from . import postprocess as pp
from .data.dataset import WindowDataset, _sample_movie, collate_windows, load_movies
from .evaluate import window_scores
from .inference import ProbaMaps, detect
from .io import create_roi, load_model, read_movie, write_movie, write_rois

log = logging.getLogger("dextrusion")


@dataclass
class DetectOptions:
    cell_diameter: float = 25
    extrusion_duration: float = 4.5
    dxy: int = 10  # spatial step of the sliding window
    dz: int = 2  # temporal step of the sliding window
    group_size: int = 4096
    batch_size: int = 512
    cat: int | None = None  # event class to export (None: all)
    volume_threshold: float = 800
    proba_threshold: float = 180
    disxy: int = 10  # peaks closer than this (pixels) are merged
    distime: int = 4  # peaks closer than this (frames) are merged
    save_rawproba: bool = True
    save_proba: bool = False
    save_cleaned: bool = False
    save_rois: bool = True
    device: str | None = None
    legacy_ensemble_shift: bool = True


def _stem(name: str) -> str:
    return Path(name).stem  # '_cell_death.zip' -> '_cell_death'


def detect_movie(movie: str | Path, models, outfolder: str | Path | None = None,
                 opts: DetectOptions | None = None) -> ProbaMaps:
    opts = opts or DetectOptions()
    movie = Path(movie)
    out = Path(outfolder) if outfolder else movie.parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    outname = out / movie.stem
    img = read_movie(movie)
    res = detect(img, models, opts.cell_diameter, opts.extrusion_duration, opts.dxy, opts.dz,
                 opts.group_size, opts.batch_size, opts.device, opts.legacy_ensemble_shift)
    cfg = res.config
    cats = range(1, cfg.ncat) if opts.cat is None else [opts.cat]
    for cat in cats:
        pm = res.probamap[cat - 1]
        tag = outname.as_posix() + _stem(cfg.catnames[cat])
        if opts.save_rawproba:
            write_movie(tag + "_rawproba.tif", pm)
        if opts.save_proba:
            write_movie(tag + "_proba.tif", pp.to_shape(pm, res.init_shape))
        if opts.save_cleaned:
            clean = pp.cleaned_probamap(pm, opts.volume_threshold, opts.proba_threshold,
                                        opts.disxy, opts.distime)
            write_movie(tag + "_cleaned_proba.tif", pp.to_shape(clean, res.init_shape))
        if opts.save_rois:
            events = pp.get_events(pm, res.init_shape, opts.volume_threshold,
                                   opts.proba_threshold, opts.disxy, opts.distime)
            write_rois(outname.as_posix() + cfg.catnames[cat],
                       [create_roi((e.t, e.y, e.x), cat=cat) for e in events])
    return res


def evaluate_windows(model_dir, data_path, val_ratio: float = 1.0, naug: int = 1, seed: int = 0,
                     batch_size: int = 64, device: str | None = None) -> dict:
    """Window-level accuracy / balanced accuracy / confusion matrix on a labelled folder.

    Windows are not augmented, so the result is deterministic (unlike the original, which scored
    randomly flipped and noised windows). With the default ``val_ratio=1`` every window is used.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(model_dir, device)
    paths, movies = load_movies(data_path)
    rng = np.random.default_rng(seed)
    samples = []
    for i, (p, m) in enumerate(zip(paths, movies)):
        tr, va = _sample_movie(p, i, m.shape, cfg, naug, 0, val_ratio, rng)
        samples += va + tr
    dl = DataLoader(WindowDataset(movies, samples, cfg), batch_size=batch_size,
                    collate_fn=collate_windows)
    ys, ps = [], []
    for x, y in dl:
        ps.append(model.predict_proba(x.to(device)).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return window_scores(np.concatenate(ys), np.concatenate(ps))
