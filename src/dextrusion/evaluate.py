"""Event-level and window-level evaluation.

Events are ``(frame, y, x)`` tuples. Event-level matching is the original greedy one-to-one
matching: a detection matches a ground-truth event within ``distance_t`` frames and
``distance_xy`` pixels, the closest (xy distance + |dt|) still-unmatched one being preferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from sklearn import metrics

Event = tuple[int, int, int]


def distance_xy(a: Event, b: Event) -> float:
    return sqrt((a[2] - b[2]) ** 2 + (a[1] - b[1]) ** 2)


def matched(roi: Event, candidates: list[Event], dxy: float, dt: float, taken: np.ndarray | None) -> bool:
    """Is ``roi`` matched by one of ``candidates``? Marks it in ``taken`` when given."""
    closest = -1
    mind = dt + dxy * 1.1
    for i, cur in enumerate(candidates):
        if abs(cur[0] - roi[0]) <= dt and distance_xy(cur, roi) <= dxy:
            if taken is None:
                return True
            if taken[i] == 0:
                mix = distance_xy(cur, roi) + abs(cur[0] - roi[0])
                if mix < mind:
                    mind, closest = mix, i
    if closest < 0:
        return False
    if taken is not None:
        taken[closest] = 1
    return True


def false_positives(detected: list[Event], truth: list[Event], dxy: float, dt: float) -> list[Event]:
    taken = np.zeros(len(truth))
    return [r for r in detected if not matched(r, truth, dxy, dt, taken)]


def false_negatives(detected: list[Event], truth: list[Event], dxy: float, dt: float) -> list[Event]:
    return [r for r in truth if not matched(r, detected, dxy, dt, None)]


@dataclass
class Score:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float


def compare(detected: list[Event], truth: list[Event], distance_xy: float = 15,
            distance_t: float = 4) -> Score:
    """True/false positives, false negatives, precision and recall (original semantics)."""
    taken = np.zeros(len(truth))
    tp = fp = 0
    for r in detected:
        if matched(r, truth, distance_xy, distance_t, taken):
            tp += 1
        else:
            fp += 1
    fn = sum(1 for r in truth if not matched(r, detected, distance_xy, distance_t, None))
    if tp + fp == 0:
        return Score(tp, fp, fn, 0.0, 0.0)
    return Score(tp, fp, fn, tp / (tp + fp), tp / (tp + fn) if tp + fn else 0.0)


def window_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": len(y_true),
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": metrics.confusion_matrix(y_true, y_pred).tolist(),
    }
