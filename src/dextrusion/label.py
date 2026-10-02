"""Annotate events of a movie in napari and save them as training ROI files.

``dextrusion label movie.tif -o annotations/`` opens the movie with one Points layer per class.
Click to add an event (the point is stored at the current frame), select + Backspace to delete.
Everything is saved automatically to ``<out>/<movie name><suffix>`` (ImageJ point ROIs, the same
files ``dextrusion train`` reads); the previous version of a file is kept as ``<name>.bak.zip``.
Existing files are loaded, so a session can be resumed. The movie is symlinked into the output
folder so that the folder is directly a training folder.

Annotation convention (keep one, as a training window spans 5 frames before to 4 frames after the
marked frame, and the sampler jitters the position by +-2 frames): mark each event at the frame
and cell centre where it is most recognisable, always on the same landmark of the event.

The coordinates are those of the movie you annotate. If cells in this movie are much larger than
the ~25 px the networks were trained at, make a downscaled training copy of the movie and of the
ROIs before training; this tool does not rescale.
"""

from __future__ import annotations

import inspect
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .io import create_roi, read_rois, write_rois

log = logging.getLogger("dextrusion")

# class name -> ROI file suffix (the file is "<movie name><suffix>")
DEFAULT_CLASSES = {"division": "_cell_division.zip", "delamination": "_cell_delamination.zip"}
PALETTE = ["lime", "magenta", "orange", "deepskyblue", "yellow"]
# Keys to switch to a class in add mode, and to save. Plain letters unused by napari's default
# shortcuts (checked against napari 0.9.2 by tests/test_label.py; note napari lists plain letter
# keys in upper case, 'S' is the plain s key). The digit keys are taken (1 deletes selected points).
CLASS_KEYS = ["q", "w", "g", "h", "j"]
SAVE_KEY = "k"


# ----------------------------------------------------------------------------- no GUI needed
def roi_path(folder: str | Path, movie_stem: str, suffix: str) -> Path:
    return Path(folder) / f"{movie_stem}{suffix}"


def load_points(path: str | Path) -> np.ndarray:
    """Events of a ROI file as an ``(n, 3)`` float array of ``(frame, y, x)``; empty if no file."""
    path = Path(path)
    if not path.exists():
        return np.zeros((0, 3))
    return np.array(read_rois(path), dtype=float).reshape(-1, 3)


def clean_points(points, shape) -> list[tuple[int, int, int]]:
    """Round to pixel coordinates, clip to the movie, drop exact duplicates, sort by (t, y, x)."""
    pts = np.rint(np.asarray(points, dtype=float).reshape(-1, 3)).astype(int)
    pts = np.clip(pts, 0, np.array(shape[:3]) - 1)
    return sorted({(int(t), int(y), int(x)) for t, y, x in pts})


def save_points(path: str | Path, points, shape) -> int:
    """Write the events to a ROI zip, keeping the previous file as ``<name>.bak.zip``."""
    path = Path(path)
    pts = clean_points(points, shape)
    if path.exists():
        shutil.copy2(path, path.with_name(path.stem + ".bak.zip"))
    write_rois(path, [create_roi(p, cat=0) for p in pts], verbose=False)
    return len(pts)


def point_style_kwargs(color: str, width: float = 0.1) -> dict:
    """napari >= 0.5 renamed the points ``edge_*`` arguments to ``border_*``; support both."""
    import napari

    params = inspect.signature(napari.Viewer.add_points).parameters
    kind = "border" if "border_color" in params else "edge"
    return {f"{kind}_color": color, f"{kind}_width": width}


# ------------------------------------------------------------------------------- napari part
def build_layers(viewer, movie: np.ndarray, out_dir: str | Path, movie_stem: str,
                 classes: dict[str, str], point_size: float = 25, autosave: bool = True):
    """Add the movie and one Points layer per class to ``viewer``.

    Works with any napari viewer object (also the windowless ``ViewerModel``). Returns
    ``(layers, save_all)``: the ``{class name: Points layer}`` dict and a function that writes all
    classes to disk and returns ``{class name: number of events}``.
    """
    lo, hi = np.percentile(movie[:: max(1, len(movie) // 10)], (1, 99.8))
    viewer.add_image(movie, name="movie", contrast_limits=(float(lo), float(hi)))
    layers = {}
    for i, (name, suffix) in enumerate(classes.items()):
        pts = load_points(roi_path(out_dir, movie_stem, suffix))
        layers[name] = viewer.add_points(
            pts, ndim=3, name=name, size=point_size, face_color="transparent",
            **point_style_kwargs(PALETTE[i % len(PALETTE)]))

    state = {"autosave": autosave}

    def save_all() -> dict[str, int]:
        return {name: save_points(roi_path(out_dir, movie_stem, classes[name]), layer.data,
                                  movie.shape) for name, layer in layers.items()}

    def on_change(event=None):
        if state["autosave"]:
            save_all()

    for layer in layers.values():
        layer.events.data.connect(on_change)
    save_all.state = state  # lets the GUI toggle autosave
    return layers, save_all


def link_movie(movie: Path, out_dir: Path) -> None:
    """Symlink the movie into the output folder (never touches an existing file)."""
    link = out_dir / movie.name
    if not (link.exists() or link.is_symlink()):
        os.symlink(movie.resolve(), link)
        log.info("Linked %s -> %s", link, movie.resolve())


def launch(movie: str | Path, out_dir: str | Path, classes: dict[str, str] | None = None,
           point_size: float = 25) -> None:
    """Open the annotation tool and block until the window is closed."""
    make_window(movie, out_dir, classes, point_size)
    import napari

    napari.run()


def make_window(movie: str | Path, out_dir: str | Path, classes: dict[str, str] | None = None,
                point_size: float = 25, show: bool = True):
    """Build the annotation window (needs napari: ``uv sync --extra label``).

    Returns ``(viewer, layers, save_all)``. ``show=False`` builds it without displaying it (tests).
    """
    try:
        import magicgui  # noqa: F401
        import napari
    except ImportError as e:
        raise SystemExit("napari is needed for labeling: install it with "
                         "`uv sync --extra label` (or `pip install 'dextrusion[label]'`)") from e
    import tifffile

    movie, out_dir = Path(movie), Path(out_dir)
    classes = classes or DEFAULT_CLASSES
    img = tifffile.imread(movie)
    if img.ndim != 3:
        raise SystemExit(f"{movie}: expected a (T, Y, X) movie, got shape {img.shape}")
    out_dir.mkdir(parents=True, exist_ok=True)
    link_movie(movie, out_dir)

    viewer = napari.Viewer(title=f"dextrusion labels: {movie.name}", show=show)
    layers, save_all = build_layers(viewer, img, out_dir, movie.stem, classes, point_size)

    container = attach_controls(viewer, layers, save_all, list(classes))
    viewer.window.add_dock_widget(container, area="right", name="dextrusion labels")
    log.info("Annotations are saved in %s (files %s<suffix>)", out_dir, movie.stem)
    return viewer, layers, save_all


def attach_controls(viewer, layers, save_all, names):
    """Bind the class / save keys and build the control widget (buttons, autosave, status).

    Needs a viewer model only (no window), so it is testable without a display. Returns the
    magicgui container to dock; the first class is activated in add mode.
    """
    from magicgui.widgets import CheckBox, Container, Label, PushButton

    status = Label(value="")

    def counts() -> str:
        return ", ".join(f"{n}: {len(layers[n].data)}" for n in names)

    def do_save():
        n = save_all()
        status.value = f"saved {time.strftime('%H:%M:%S')} ({', '.join(f'{k}: {v}' for k, v in n.items())})"

    def refresh(event=None):
        status.value = counts()

    def activate(name):
        layer = layers[name]
        viewer.layers.selection.active = layer
        layer.mode = "add"

    if len(names) > len(CLASS_KEYS):
        raise SystemExit(f"at most {len(CLASS_KEYS)} classes are supported by the labeling tool")
    buttons = []
    for name, key in zip(names, CLASS_KEYS):
        b = PushButton(text=f"add {name}  (key {key})")
        b.clicked.connect(lambda _=None, n=name: activate(n))
        buttons.append(b)
        viewer.bind_key(key, lambda v, n=name: activate(n), overwrite=True)
    save_btn = PushButton(text=f"save now  (key {SAVE_KEY})")
    save_btn.clicked.connect(lambda _=None: do_save())
    viewer.bind_key(SAVE_KEY, lambda v: do_save(), overwrite=True)
    auto = CheckBox(value=True, text="autosave on every change")
    auto.changed.connect(lambda v: save_all.state.update(autosave=bool(v)))
    for layer in layers.values():
        layer.events.data.connect(refresh)
    refresh()
    activate(names[0])
    return Container(widgets=[*buttons, save_btn, auto, status])
