"""Command line: ``dextrusion {detect,train,evaluate,convert}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import DeXConfig


def _add_detect(sub):
    p = sub.add_parser("detect", help="detect events in movie(s)")
    p.add_argument("movies", nargs="+", type=Path, help="tif movies shaped (T, Y, X)")
    p.add_argument("-m", "--models", required=True, type=Path,
                   help="a DeXNet folder, or a folder of DeXNets (used as an ensemble)")
    p.add_argument("-o", "--outdir", type=Path, help="default: <movie folder>/results")
    p.add_argument("--cell-diameter", type=float, default=25)
    p.add_argument("--extrusion-duration", type=float, default=4.5)
    p.add_argument("--dxy", type=int, default=10, help="spatial step of the sliding window")
    p.add_argument("--dz", type=int, default=2, help="temporal step of the sliding window")
    p.add_argument("--group-size", type=int, default=4096, help="windows kept in memory at once")
    p.add_argument("--batch-size", type=int, default=512, help="windows per forward pass")
    p.add_argument("--cat", type=int, help="event class to export (default: all)")
    p.add_argument("--volume-threshold", type=float, default=800)
    p.add_argument("--proba-threshold", type=float, default=180)
    p.add_argument("--disxy", type=int, default=10)
    p.add_argument("--distime", type=int, default=4)
    p.add_argument("--save-proba", action="store_true", help="also save probability maps at movie size")
    p.add_argument("--save-cleaned", action="store_true", help="also save thresholded probability maps")
    p.add_argument("--no-rois", action="store_true")
    p.add_argument("--device", help="cpu, cuda, cuda:1... (default: cuda if available)")
    p.add_argument("--consistent-ensemble-shift", action="store_true",
                   help="use the geometrically consistent ensemble shifts instead of the original ones")


def _add_train(sub):
    p = sub.add_parser("train", help="train (or retrain with --init-from) a DeXNet")
    p.add_argument("data", type=Path, help="folder with movies.tif and ROI zips")
    p.add_argument("-o", "--out", required=True, type=Path)
    p.add_argument("--init-from", type=Path, help="DeXNet (native or legacy Keras) to fine-tune")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--nb-filters", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--ncat", type=int, default=4)
    p.add_argument("--catnames", nargs="+", metavar="SUFFIX",
                   help="ROI file suffix of every class, first one empty for 'no event', e.g. "
                        "--catnames '' _cell_delamination.zip _cell_division.zip (sets ncat; "
                        "default: '' _cell_death.zip _cell_sop.zip _cell_division.zip)")
    p.add_argument("--half-size", type=int, nargs=2, default=(22, 22))
    p.add_argument("--nframes", type=int, nargs=2, default=(5, 5))
    p.add_argument("--cell-diameter", type=float, default=25)
    p.add_argument("--extrusion-duration", type=float, default=4.5)
    p.add_argument("--naug", type=int, default=1, help="augmentation factor (1: none)")
    p.add_argument("--add-nothing", type=int, default=10)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--no-noise", action="store_true", help="disable noise augmentations")
    p.add_argument("--no-plateau", action="store_true", help="disable lr reduction on plateau")
    p.add_argument("--legacy-batch-normalization", action="store_true",
                   help="min-max scale with the whole batch (original behaviour)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device")


def _add_evaluate(sub):
    p = sub.add_parser("evaluate", help="evaluate windows or detected events")
    ev = p.add_subparsers(dest="what", required=True)
    w = ev.add_parser("windows", help="window classification scores on a training-style folder")
    w.add_argument("data", type=Path)
    w.add_argument("-m", "--model", required=True, type=Path)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--device")
    e = ev.add_parser("events", help="precision / recall of detected ROIs vs ground truth")
    e.add_argument("detected", type=Path, help="ROI zip written by `detect`")
    e.add_argument("truth", type=Path, help="ground-truth ROI zip")
    e.add_argument("--distance-xy", type=float, default=15)
    e.add_argument("--distance-t", type=float, default=4)
    e.add_argument("--write-fp", type=Path, help="save false positives as ROI zip")
    e.add_argument("--write-fn", type=Path, help="save false negatives as ROI zip")


def _add_convert(sub):
    p = sub.add_parser("convert", help="convert legacy Keras DeXNet(s) to the PyTorch format")
    p.add_argument("sources", nargs="+", type=Path,
                   help="legacy DeXNet folder(s) or .zip file(s)")
    p.add_argument("-o", "--out", type=Path,
                   help="output folder (single source only; default: <source>/torch or <zip>.torch)")


def _add_label(sub):
    p = sub.add_parser("label", help="annotate events in napari and save training ROI files "
                                     "(needs the `label` extra)")
    p.add_argument("movie", type=Path, help="tif movie shaped (T, Y, X)")
    p.add_argument("-o", "--out", required=True, type=Path,
                   help="folder for the ROI files (the movie is symlinked into it)")
    p.add_argument("--classes", nargs="+", metavar="NAME=SUFFIX",
                   help="classes to annotate (default: division=_cell_division.zip "
                        "delamination=_cell_delamination.zip)")
    p.add_argument("--point-size", type=float, default=25, help="displayed size of the points")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dextrusion", description=__doc__)
    p.add_argument("-q", "--quiet", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    _add_detect(sub)
    _add_train(sub)
    _add_evaluate(sub)
    _add_convert(sub)
    _add_label(sub)
    return p


def _cmd_detect(a) -> int:
    from .run import DetectOptions, detect_movie

    opts = DetectOptions(
        cell_diameter=a.cell_diameter, extrusion_duration=a.extrusion_duration, dxy=a.dxy,
        dz=a.dz, group_size=a.group_size, batch_size=a.batch_size, cat=a.cat,
        volume_threshold=a.volume_threshold, proba_threshold=a.proba_threshold, disxy=a.disxy,
        distime=a.distime, save_proba=a.save_proba, save_cleaned=a.save_cleaned,
        save_rois=not a.no_rois, device=a.device,
        legacy_ensemble_shift=not a.consistent_ensemble_shift)
    for movie in a.movies:
        logging.getLogger("dextrusion").info("Detecting events on %s", movie)
        detect_movie(movie, a.models, a.outdir, opts)
    return 0


def _cmd_train(a) -> int:
    from .train import TrainOptions, train

    extra = {}
    if a.catnames and a.init_from:
        raise SystemExit("--catnames cannot be combined with --init-from: a retrained network keeps "
                         "the classes of the network it starts from. Train from scratch to use "
                         "new classes.")
    if a.catnames:
        if a.catnames[0] != "" or len(a.catnames) < 2 or not all(c.endswith(".zip") for c in a.catnames[1:]):
            raise SystemExit("--catnames: the first name must be the empty string (no event) and "
                             "the others ROI file suffixes ending in .zip")
        extra = {"catnames": list(a.catnames), "ncat": len(a.catnames)}
    config = DeXConfig(**{"ncat": a.ncat, **extra}, half_size=tuple(a.half_size),
                       nframes=tuple(a.nframes), cell_diameter=a.cell_diameter,
                       extrusion_duration=a.extrusion_duration, nb_filters=a.nb_filters,
                       batch_size=a.batch_size)
    opts = TrainOptions(
        epochs=a.epochs, lr=a.lr, val_ratio=a.val_ratio, naug=a.naug,
        add_nothing_windows=a.add_nothing, augment_noise=not a.no_noise,
        plateau=not a.no_plateau, legacy_batch_normalization=a.legacy_batch_normalization,
        seed=a.seed, num_workers=a.workers, device=a.device)
    train(a.data, a.out, config, opts, init_from=a.init_from)
    return 0


def _cmd_evaluate(a) -> int:
    if a.what == "windows":
        from .run import evaluate_windows

        print(json.dumps(evaluate_windows(a.model, a.data, seed=a.seed, device=a.device), indent=2))
        return 0
    from . import evaluate as ev
    from .io import create_roi, read_rois, write_rois

    det, truth = read_rois(a.detected), read_rois(a.truth)
    s = ev.compare(det, truth, a.distance_xy, a.distance_t)
    print(json.dumps(s.__dict__, indent=2))
    if a.write_fp:
        write_rois(a.write_fp, [create_roi(r) for r in
                                ev.false_positives(det, truth, a.distance_xy, a.distance_t)])
    if a.write_fn:
        write_rois(a.write_fn, [create_roi(r) for r in
                                ev.false_negatives(det, truth, a.distance_xy, a.distance_t)])
    return 0


def _cmd_convert(a) -> int:
    from .convert.cli import convert_keras

    if a.out is not None and len(a.sources) != 1:
        raise SystemExit("--out can only be used with a single source")
    for src in a.sources:
        print(convert_keras(src, a.out))
    return 0


def parse_classes(items: list[str] | None) -> dict[str, str] | None:
    """``NAME=SUFFIX`` pairs -> dict (None keeps the defaults of the labeling tool)."""
    if not items:
        return None
    out = {}
    for item in items:
        name, sep, suffix = item.partition("=")
        if not sep or not name or not suffix.endswith(".zip"):
            raise SystemExit(f"--classes: '{item}' is not NAME=SUFFIX with a suffix ending in .zip")
        out[name] = suffix
    return out


def _cmd_label(a) -> int:
    from .label import launch

    launch(a.movie, a.out, parse_classes(a.classes), a.point_size)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s", stream=sys.stderr)
    return {"detect": _cmd_detect, "train": _cmd_train, "evaluate": _cmd_evaluate,
            "convert": _cmd_convert, "label": _cmd_label}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
