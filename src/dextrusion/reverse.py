"""Make a time-reversed training copy of a movie and of its ROI files.

Played backwards, a cell that shrinks and leaves the layer (extrusion) is a cell that appears and
expands (an emerging cell, e.g. a delamination seen from the basal side). ``dextrusion reverse``
writes the reversed movie and moves the ``cell_death`` ROIs to a class of their own, so that the
large set of annotated extrusions can pre-train that class. Original files are never modified (outputs are named ``<movie>_rev*``, so the output folder may be
the movie's own folder or a training folder that links to it).

Only the death ROIs (and the hand-picked ``nothing`` positions) are carried over: a reversed
division is two cells merging and a reversed SOP is not a SOP, so those files are left out.
The marked frame keeps its meaning (``t -> T - 1 - t``); the training window spans 5 frames
before to 4 after it, so the reversed window is shifted by one frame with respect to an exact
mirror, well inside the +-2 frames of jitter of the sampler.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .io import create_roi, read_movie, read_rois, write_movie, write_rois
from .prepare import find_roi_files

log = logging.getLogger("dextrusion")

DEATH = "_cell_death.zip"
NOTHING = "_nothing.zip"
DEFAULT_DEATH_AS = "_cell_delamination.zip"


def reverse_point(point, n_frames: int) -> tuple[int, int, int]:
    """Map a ``(frame, y, x)`` position of a movie to the time-reversed movie."""
    frame, y, x = point
    return (n_frames - 1 - int(frame), int(y), int(x))


def reverse(movie: str | Path, out_dir: str | Path, rois_dir: str | Path | None = None,
            roi_files: list[str | Path] | None = None, death_as: str = DEFAULT_DEATH_AS) -> dict:
    """Write ``<out_dir>/<movie>_rev.tif`` and the reversed ROI files.

    ``<movie>_cell_death.zip`` becomes ``<movie>_rev<death_as>`` and ``<movie>_nothing.zip``
    becomes ``<movie>_rev_nothing.zip``; every other ROI file (division, SOP, ...) is not copied.

    :param rois_dir: where the ROI files of the movie are (default: the movie's folder)
    :param roi_files: explicit ROI files instead of searching ``rois_dir``
    :param death_as: ROI file suffix given to the reversed death ROIs
    :return: a summary (shapes, ROI counts, dropped files), also saved as ``<movie>_rev.reverse.json``
    """
    movie, out_dir = Path(movie), Path(out_dir)
    if not death_as.endswith(".zip"):
        raise ValueError(f"death_as must be a ROI file suffix ending in .zip, got {death_as!r}")
    img = read_movie(movie)
    if img.ndim != 3:
        raise ValueError(f"{movie}: expected a (T, Y, X) movie, got shape {img.shape}")
    files = [Path(f) for f in roi_files] if roi_files else find_roi_files(
        movie, Path(rois_dir) if rois_dir else movie.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{movie.stem}_rev"
    write_movie(out_dir / f"{stem}.tif", np.ascontiguousarray(img[::-1]))
    counts, dropped = {}, []
    for f in files:
        suffix = f.name[len(movie.stem):]  # '_cell_death.zip', '_nothing.zip', ...
        target = {DEATH: death_as, NOTHING: NOTHING}.get(suffix)
        if target is None:
            dropped.append(f.name)
            continue
        pts = [reverse_point(p, img.shape[0]) for p in read_rois(f)]
        write_rois(out_dir / f"{stem}{target}", [create_roi(p, cat=0) for p in pts], verbose=False)
        counts[f"{stem}{target}"] = len(pts)
    summary = {"movie": str(movie), "shape": list(img.shape), "death_as": death_as,
               "rois": counts, "dropped": dropped}
    (out_dir / f"{stem}.reverse.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("%s: reversed %d frames, ROI files %s, dropped %s", movie.name, img.shape[0],
             counts, dropped)
    return summary
