"""Probability maps -> event volumes -> point events (and Fiji ROIs)."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isnan

import numpy as np
from scipy.ndimage import (
    center_of_mass,
    distance_transform_edt,
    label,
    maximum_filter,
    minimum_filter,
    zoom,
)
from scipy.ndimage import sum as ndsum
from skimage.morphology import local_maxima
from skimage.segmentation import watershed

RAW_THRESHOLD = 125  # probability (0-255) above which a pixel is "positive"


@dataclass
class Event:
    t: int
    y: int
    x: int
    volume: float
    proba: float


def rawproba_to_volumes(img: np.ndarray, threshold: int = RAW_THRESHOLD, mindxy: int = 5,
                        mindt: int = 3):
    """Split the positive part of a probability map into separate event volumes.

    Returns ``(binary, labels, label_ids, volumes, mean_probabilities)``.
    """
    binimg = np.copy(img)
    binimg[img > threshold] = 1
    binimg[img <= threshold] = 0
    # Spatial 2x2 erosion. The original passed size=(0, 2, 2); scipy skips axes with size <= 1,
    # so this is the same operation, spelled explicitly.
    binimg = minimum_filter(binimg, size=(1, 2, 2))
    distance = distance_transform_edt(binimg)
    maxs = local_maxima(distance, connectivity=16)
    maxs = maximum_filter(maxs, size=(mindt, mindxy, mindxy))

    maxlabels, _ = label(maxs > 0)
    labels = watershed(-distance, maxlabels, mask=binimg)
    ids = np.arange(1, np.max(labels) + 1)
    vols = ndsum(binimg, labels, ids)
    vals = ndsum(img, labels, ids) / vols
    return binimg, labels, ids, vols, vals


def rescale_position(z, y, x, init_shape, map_shape):
    """Map a position in the (rescaled) probability map back to the original movie."""
    ratioxy = init_shape[1] / map_shape[1]
    ratioz = init_shape[0] / map_shape[0]
    return floor(z * ratioz), floor(y * ratioxy), floor(x * ratioxy)


def get_events(probamap: np.ndarray, init_shape, volume_threshold: float = 800,
               proba_threshold: float = 180, disxy: int = 10, dist: int = 4) -> list[Event]:
    """Centroids of positive volumes bigger / more probable than the thresholds.

    :param probamap: ``(T, Y, X)`` uint8 map of one event class (at the rescaled size)
    :param init_shape: shape of the original movie, positions are expressed in it
    """
    binimg, labels, ids, vols, vals = rawproba_to_volumes(probamap, RAW_THRESHOLD, disxy, dist)
    coords = center_of_mass(binimg, labels=labels, index=ids)
    events = []
    for pt, vol, val in zip(coords, vols, vals):
        if not isnan(pt[0]) and vol > volume_threshold and val > proba_threshold:
            t, y, x = rescale_position(pt[0], pt[1], pt[2], init_shape, probamap.shape)
            events.append(Event(t, y, x, float(vol), float(val)))
    return events


def cleaned_probamap(probamap: np.ndarray, volume_threshold: float = 800,
                     proba_threshold: float = 180, disxy: int = 10, distime: int = 4) -> np.ndarray:
    """Probability map keeping only the volumes that would give an event (for visualisation)."""
    _, labels, ids, vols, vals = rawproba_to_volumes(probamap, RAW_THRESHOLD, disxy, distime)
    out = np.zeros(probamap.shape, dtype="uint8")
    for lab, vol, val in zip(ids, vols, vals):
        if not isnan(vol) and vol > volume_threshold and val > proba_threshold:
            out[labels == lab] = floor(val)
    return out


def to_shape(img: np.ndarray, shape) -> np.ndarray:
    """Resize a (T, Y, X) map to ``shape`` (same zoom as the original)."""
    if tuple(img.shape) == tuple(shape):
        return img
    return zoom(img, (shape[0] / img.shape[0], shape[1] / img.shape[1], shape[1] / img.shape[1]))
