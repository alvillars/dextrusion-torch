"""Convert legacy Keras DeXNets (SavedModel dir or zip) to safetensors + config.json."""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from ..config import DeXConfig
from ..io import LEGACY_CONFIG_FILE, save_model
from .keras_weights import build_model

log = logging.getLogger("dextrusion")
EXTRACT_SCRIPT = Path(__file__).with_name("extract_tf.py")


def _extract_in_process(model_dir: Path, out: Path) -> None:
    import runpy

    ns = runpy.run_path(str(EXTRACT_SCRIPT), run_name="extract_tf")
    ns["main"](str(model_dir), str(out))


def _extract_isolated(model_dir: Path, out: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Converting a Keras DeXNet needs TensorFlow. Either install `uv` (it will run the "
            "TensorFlow step in an isolated environment) or `pip install 'dextrusion[convert]'`."
        )
    cmd = [uv, "run", "--no-project", "--quiet", "--script", str(EXTRACT_SCRIPT), str(model_dir), str(out)]
    log.info("Reading Keras weights in an isolated uv environment (first run downloads TensorFlow)")
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"TensorFlow weight extraction failed:\n{res.stderr or res.stdout}")


def extract_keras_arrays(model_dir: Path) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "weights.npz"
        if importlib.util.find_spec("tensorflow") is not None:
            _extract_in_process(model_dir, out)
        else:
            _extract_isolated(model_dir, out)
        with np.load(out) as z:
            return {k.replace("|", "/"): z[k] for k in z.files}


def convert_keras(src: str | Path, out: str | Path | None = None) -> Path:
    """Convert one legacy DeXNet (directory or ``.zip``). Returns the output directory."""
    src = Path(src)
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = src
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            cands = [p.parent for p in Path(tmp).rglob(LEGACY_CONFIG_FILE)]
            if len(cands) != 1:
                raise ValueError(f"{src}: expected exactly one model inside the zip")
            model_dir = cands[0]
            default_out = src.with_suffix(".torch")
        else:
            default_out = src / "torch"
        out = Path(out) if out is not None else default_out

        config = DeXConfig.from_legacy_cfg(model_dir / LEGACY_CONFIG_FILE)
        model = build_model(extract_keras_arrays(model_dir))
        if model.ncat != config.ncat or model.nb_filters != config.nb_filters:
            raise ValueError(
                f"config.cfg (ncat={config.ncat}, filters={config.nb_filters}) does not match the "
                f"weights (ncat={model.ncat}, filters={model.nb_filters})"
            )
        save_model(out, model, config)
    log.info("Converted %s -> %s", src, out)
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin wrapper
    from ..cli import main as cli_main

    return cli_main(["convert", *(argv if argv is not None else sys.argv[1:])])
