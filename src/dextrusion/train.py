"""Training / retraining of a DeXNet."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch
from monai.data import DataLoader
from monai.utils import set_determinism
from torch import nn

from .config import DeXConfig, extend_catnames
from .data.dataset import build_datasets, collate_windows
from .io import load_model, save_model
from .model import DeXNet

log = logging.getLogger("dextrusion")


@dataclass
class TrainOptions:
    epochs: int = 50
    lr: float = 0.1  # plain SGD, as in the original
    batch_size: int | None = None  # default: config.batch_size
    val_ratio: float = 0.2
    naug: int = 1
    add_nothing_windows: int = 10
    augment_noise: bool = True
    plateau: bool = True  # ReduceLROnPlateau(factor=.1, patience=10) on val loss, as in the original
    legacy_batch_normalization: bool = False  # scale with the batch-wide min/max like the original
    seed: int = 0
    num_workers: int = 0
    device: str | None = None
    freeze_cnn: bool = False  # fine-tuning: keep the per-frame CNN fixed, train GRU + head only
    oversample: dict[str, int] | None = None  # {movie name: k}: sample these movies k times


def _run_epoch(model, loader, device, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total / max(n, 1), correct / max(n, 1)


def train(data_path: str | Path, out_dir: str | Path, config: DeXConfig, opts: TrainOptions,
          init_from: str | Path | None = None, new_catnames: list[str] | None = None) -> Path:
    """Train on a folder of movies + ROI zips and save the model (safetensors + config.json).

    :param init_from: an existing DeXNet (native or legacy Keras) to fine-tune ("retrain")
    :param new_catnames: with ``init_from``, the class names of the fine-tuned network: the classes
        of the starting network, in the same order, followed by the new classes. The output layer
        is widened (existing rows kept, new rows initialised).
    """
    device = torch.device(opts.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_determinism(seed=opts.seed)
    out_dir = Path(out_dir)

    if init_from is not None:
        model, base_cfg = load_model(init_from, device)
        names = list(base_cfg.catnames)
        if new_catnames is not None:
            names = extend_catnames(base_cfg.catnames, new_catnames)
            if len(names) > base_cfg.ncat:
                log.info("Adding classes %s to %s (output layer widened %d -> %d)",
                         names[base_cfg.ncat:], init_from, base_cfg.ncat, len(names))
                model = model.with_extra_classes(len(names))
        # window geometry must follow the model being retrained; classes only grow
        config = replace(config, ncat=len(names), catnames=names,
                         half_size=base_cfg.half_size, nframes=base_cfg.nframes,
                         nb_filters=base_cfg.nb_filters, cell_diameter=base_cfg.cell_diameter,
                         extrusion_duration=base_cfg.extrusion_duration)
    else:
        if new_catnames is not None:
            raise ValueError("new_catnames only applies when fine-tuning (init_from)")
        model = DeXNet(config.ncat, config.nb_filters).to(device)
    if opts.freeze_cnn:
        model.freeze_cnn()
        log.info("CNN frozen: training the GRU and the decision head only")
    batch_size = opts.batch_size or config.batch_size
    config.nb_epochs, config.augmentation = opts.epochs, opts.naug
    config.add_nothing_windows, config.batch_size = opts.add_nothing_windows, batch_size

    train_ds, val_ds = build_datasets(
        data_path, config, opts.val_ratio, opts.naug, opts.add_nothing_windows,
        opts.augment_noise, opts.seed, opts.oversample)
    log.info("%d training / %d validation windows", len(train_ds), len(val_ds))
    if len(train_ds) < batch_size:
        raise ValueError("fewer training windows than one batch: add data or lower the batch size")
    collate = partial(collate_windows, normalize="batch" if opts.legacy_batch_normalization else "sample")
    g = torch.Generator().manual_seed(opts.seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                          num_workers=opts.num_workers, collate_fn=collate, generator=g)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=opts.num_workers, collate_fn=collate) if len(val_ds) else None

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=opts.lr)
    # Keras ReduceLROnPlateau(patience=10) reduces on the 10th stagnating epoch; torch's counter
    # needs patience=9 for the same behaviour. min_delta is absolute in Keras.
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=9, threshold=1e-4, threshold_mode="abs")
        if opts.plateau and val_dl is not None else None)
    loss_fn = nn.CrossEntropyLoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(out_dir / "history.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "lr", "loss", "acc", "val_loss", "val_acc"])
        for epoch in range(1, opts.epochs + 1):
            loss, acc = _run_epoch(model, train_dl, device, loss_fn, optimizer)
            vloss, vacc = _run_epoch(model, val_dl, device, loss_fn) if val_dl else (float("nan"),) * 2
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, lr, loss, acc, vloss, vacc])
            fh.flush()
            log.info("epoch %d/%d lr=%.4g loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f",
                     epoch, opts.epochs, lr, loss, acc, vloss, vacc)
            if scheduler is not None:
                scheduler.step(vloss)
    log.info("Training done in %.1f min", (time.time() - start) / 60)
    save_model(out_dir, model, config)
    return out_dir
