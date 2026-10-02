"""Make a training copy of a movie (and of its ROI files) at the scale a network expects.

The networks are trained on cells of about 25 px and events of about 4.5 frames. A movie whose
cells are, say, 50 px wide has to be downscaled before it can be used for training, and the
annotated ROIs have to follow. ``dextrusion prepare`` applies exactly the zoom that
``dextrusion detect`` applies to the same movie (``--cell-diameter`` / ``--extrusion-duration`` of
the movie), so training and detection see the same scale. Original files are never modified.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .inference import rescale_movie, scale_factors
from .io import create_roi, read_movie, read_rois, write_movie, write_rois

log = logging.getLogger("dextrusion")


def rescale_point(point, in_shape, out_shape) -> tuple[int, int, int]:
    """Map a ``(frame, y, x)`` position of the original movie into the zoomed movie.

    ``scipy.ndimage.zoom`` aligns the corner pixels of input and output, so a coordinate ``c``
    becomes ``c * (n_out - 1) / (n_in - 1)`` along every axis. The result is rounded and clipped.
    """
    out = []
    for c, n_in, n_out in zip(point, in_shape, out_shape):
        v = c * (n_out - 1) / (n_in - 1) if n_in > 1 else 0.0
        out.append(int(min(max(round(v), 0), n_out - 1)))
    return tuple(out)  # type: ignore[return-value]


def find_roi_files(movie: Path, rois_dir: Path) -> list[Path]:
    """``<movie name>_*.zip`` files of a folder, without the ``.bak.zip`` backups of the labeler."""
    return sorted(p for p in rois_dir.glob(f"{movie.stem}_*.zip") if not p.name.endswith(".bak.zip"))


def prepare(movie: str | Path, out_dir: str | Path, cell_diameter: float = 25,
            extrusion_duration: float = 4.5, target_diameter: float = 25,
            target_duration: float = 4.5, rois_dir: str | Path | None = None,
            roi_files: list[str | Path] | None = None) -> dict:
    """Write ``<out_dir>/<movie>.tif`` rescaled to the target scale and the scaled ROI files.

    :param cell_diameter: typical cell diameter of *this* movie (pixels)
    :param extrusion_duration: typical event duration in *this* movie (frames)
    :param target_diameter: scale of the networks (default 25 px)
    :param target_duration: scale of the networks (default 4.5 frames)
    :param rois_dir: where the ROI files of the movie are (default: the movie's folder)
    :param roi_files: explicit ROI files instead of searching ``rois_dir``
    :return: a summary (ratios, shapes, ROI counts), also saved as ``<movie>.prepare.json``
    """
    movie, out_dir = Path(movie), Path(out_dir)
    img = read_movie(movie)
    if img.ndim != 3:
        raise ValueError(f"{movie}: expected a (T, Y, X) movie, got shape {img.shape}")
    ratioxy, ratioz = scale_factors(target_diameter, cell_diameter, target_duration,
                                    extrusion_duration)
    scaled = rescale_movie(img, ratioxy, ratioz)
    files = [Path(f) for f in roi_files] if roi_files else find_roi_files(
        movie, Path(rois_dir) if rois_dir else movie.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / movie.name).resolve() == movie.resolve():
        raise ValueError("the output folder holds the original movie: choose another folder")

    write_movie(out_dir / movie.name, scaled)
    counts = {}
    for f in files:
        pts = [rescale_point(p, img.shape, scaled.shape) for p in read_rois(f)]
        write_rois(out_dir / f.name, [create_roi(p, cat=0) for p in pts], verbose=False)
        counts[f.name] = len(pts)
    summary = {"movie": str(movie), "ratio_xy": ratioxy, "ratio_t": ratioz,
               "shape_in": list(img.shape), "shape_out": list(scaled.shape),
               "cell_diameter": cell_diameter, "extrusion_duration": extrusion_duration,
               "target_diameter": target_diameter, "target_duration": target_duration,
               "rois": counts}
    (out_dir / f"{movie.stem}.prepare.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("%s: %s -> %s (ratio xy %.3f, t %.3f), %d ROI file(s)", movie.name,
             tuple(img.shape), tuple(scaled.shape), ratioxy, ratioz, len(files))
    return summary

