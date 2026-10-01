"""Network/detection parameters, stored next to the weights as ``config.json``.

Legacy DeXNets ship a ``config.cfg`` (``key = value`` lines); :meth:`DeXConfig.from_legacy_cfg`
reads it.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from math import floor
from pathlib import Path

DEFAULT_CATNAMES = ["", "_cell_death.zip", "_cell_sop.zip", "_cell_division.zip"]


def _tuple(value: str) -> tuple[int, ...]:
    return tuple(int(v) for v in ast.literal_eval(value.strip()))


@dataclass
class DeXConfig:
    ncat: int = 4  # number of classes, including "no event" (class 0)
    catnames: list[str] = field(default_factory=lambda: list(DEFAULT_CATNAMES))
    half_size: tuple[int, int] = (22, 22)  # window half size in (y, x) pixels
    nframes: tuple[int, int] = (5, 5)  # frames before / after the window centre
    cell_diameter: float = 25.0  # pixels, used to rescale movies spatially
    extrusion_duration: float = 4.5  # frames, used to rescale movies temporally
    nb_filters: int = 8
    batch_size: int = 30
    nb_epochs: int | None = None
    augmentation: int | None = None
    add_nothing_windows: int | None = None

    @property
    def window_shape(self) -> tuple[int, int, int]:
        return (sum(self.nframes), 2 * self.half_size[0] + 1, 2 * self.half_size[1] + 1)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Extent around a window centre that receives the window's probability.

        (frames before, frames after, y half-extent, x half-extent)
        """
        return (
            floor(self.nframes[0] * 0.2),
            floor(self.nframes[1] * 0.2) + 1,
            floor(self.half_size[0] * 0.3),
            floor(self.half_size[1] * 0.3),
        )

    # ------------------------------------------------------------------ json
    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> DeXConfig:
        d = json.loads(Path(path).read_text())
        d["half_size"] = tuple(d["half_size"])
        d["nframes"] = tuple(d["nframes"])
        return cls(**d)

    # ---------------------------------------------------------------- legacy
    @classmethod
    def from_legacy_cfg(cls, path: str | Path) -> DeXConfig:
        """Parse the ``config.cfg`` written by the original TensorFlow implementation."""
        cfg = cls()
        for line in Path(path).read_text().splitlines():
            if line.startswith("#") or "=" not in line:
                continue
            key, value = (s.strip() for s in line.split("=", 1))
            match key:
                case "nb_events_category":
                    cfg.ncat = int(value)
                case "events_category_names":
                    cfg.catnames = [str(v) for v in ast.literal_eval(value)]
                case "window_half_size":
                    cfg.half_size = _tuple(value)  # type: ignore[assignment]
                case "nb_temporal_frames":
                    cfg.nframes = _tuple(value)  # type: ignore[assignment]
                case "cell_diameter":
                    cfg.cell_diameter = float(value)
                case "extrusion_duration":
                    cfg.extrusion_duration = float(value)
                case "neuralnetwork_nb_filters":
                    cfg.nb_filters = int(value)
                case "neuralnetwork_batch_size":
                    cfg.batch_size = int(value)
                case "neuralnetwork_nb_epochs":
                    cfg.nb_epochs = int(value)
                case "datatraining_augmentation":
                    cfg.augmentation = int(value)
                case "datatraining_addnothingwindows":
                    cfg.add_nothing_windows = int(value)
                # datatraining_path is deliberately dropped (machine specific)
        return cfg
