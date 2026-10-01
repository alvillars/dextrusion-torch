"""Model, movie and ROI input/output."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import roifile
import tifffile
import torch
from safetensors.torch import load_file, save_file

from .config import DeXConfig
from .model import DeXNet

log = logging.getLogger("dextrusion")

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
LEGACY_CONFIG_FILE = "config.cfg"
CONVERTED_SUBDIR = "torch"  # cache of auto-converted legacy models, inside the legacy dir


# ---------------------------------------------------------------------------- models
def save_model(path: str | Path, model: DeXNet, config: DeXConfig) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path / WEIGHTS_FILE))
    config.nb_filters = model.nb_filters
    config.ncat = model.ncat
    config.to_json(path / CONFIG_FILE)


def is_native_model(path: Path) -> bool:
    return (path / WEIGHTS_FILE).exists() and (path / CONFIG_FILE).exists()


def is_legacy_model(path: Path) -> bool:
    return (path / LEGACY_CONFIG_FILE).exists() and (path / "variables").is_dir()


def load_native(path: str | Path, device: str | torch.device = "cpu") -> tuple[DeXNet, DeXConfig]:
    path = Path(path)
    config = DeXConfig.from_json(path / CONFIG_FILE)
    model = DeXNet(ncat=config.ncat, nb_filters=config.nb_filters)
    model.load_state_dict(load_file(str(path / WEIGHTS_FILE)), strict=True)
    return model.to(device).eval(), config


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[DeXNet, DeXConfig]:
    """Load a native model dir, or a legacy Keras DeXNet (converted once, then cached)."""
    path = Path(path)
    if is_native_model(path):
        return load_native(path, device)
    if is_legacy_model(path):
        cached = path / CONVERTED_SUBDIR
        if not is_native_model(cached):
            from .convert.cli import convert_keras

            log.info("Legacy Keras DeXNet %s: converting to %s", path, cached)
            convert_keras(path, cached)
        return load_native(cached, device)
    raise FileNotFoundError(
        f"{path} is neither a DeXNet ({WEIGHTS_FILE} + {CONFIG_FILE}) nor a legacy Keras DeXNet"
    )


def resolve_models(path: str | Path) -> list[Path]:
    """One model dir, or every model found in the sub-directories of ``path`` (sorted)."""
    path = Path(path)
    if is_native_model(path) or is_legacy_model(path):
        return [path]
    subs = sorted(
        p for p in path.iterdir() if p.is_dir() and (is_native_model(p) or is_legacy_model(p))
    )
    if not subs:
        raise FileNotFoundError(f"no DeXNet found in {path}")
    return subs


# ----------------------------------------------------------------------------- movies
def read_movie(path: str | Path) -> np.ndarray:
    return tifffile.imread(path)


def write_movie(path: str | Path, img: np.ndarray, astime: bool = True) -> None:
    tifffile.imwrite(path, img, imagej=True, metadata={"axes": "TYX" if astime else "ZYX"})


# -------------------------------------------------------------------------------- ROIs
def read_rois(path: str | Path) -> list[tuple[int, int, int]]:
    """ImageJ point ROIs -> list of ``(frame, y, x)`` (frame is 0-based)."""
    out = []
    for roi in roifile.ImagejRoi.fromfile(str(path)):
        frame = max(roi.position, roi.z_position, roi.t_position) - 1
        out.append((frame, roi.top, roi.left))
    return out


_COLORS = {1: b"\xff\xff\x00\x00", 2: b"\xff\x00\x00\xff", 3: b"\xff\x00\xff\x00"}


def create_roi(pt: tuple[int, int, int], cat: int = 1, astime: bool = True) -> roifile.ImagejRoi:
    """Point ROI at ``pt = (frame, y, x)`` (frame 0-based)."""
    roi = roifile.ImagejRoi()
    roi.version = 227
    roi.roitype = roifile.ROI_TYPE(10)
    roi.name = str(pt[0] + 1).zfill(4) + "-" + str(pt[1]).zfill(4) + "-" + str(pt[2]).zfill(4)
    roi.n_coordinates = 1
    roi.left = int(pt[2])
    roi.top = int(pt[1])
    roi.position = pt[0] + 1
    roi.z_position = 1 if astime else pt[0] + 1
    roi.t_position = pt[0] + 1 if astime else 1
    roi.c_position = 1
    roi.integer_coordinates = np.array([[0, 0]])
    roi.stroke_width = 3
    if cat in _COLORS:
        roi.stroke_color = _COLORS[cat]
    return roi


def write_rois(path: str | Path, rois: list, verbose: bool = True) -> None:
    if verbose:
        log.info("Writing %d ROIs in %s", len(rois), path)
    roifile.roiwrite(str(path), rois, mode="w")
