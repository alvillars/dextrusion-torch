# /// script
# requires-python = ">=3.10"
# dependencies = ["tensorflow-cpu", "numpy", "scipy"]
# ///
"""Generate golden Keras outputs for the parity tests.

Usage: ``uv run --script make_golden.py <DeXNets_dir> <out.npz>``

Runs every legacy DeXNet found under ``<DeXNets_dir>`` (directories and zips) on the shared
test windows (``windows.py``) and stores the softmax outputs. Needs the legacy models, so it is only run
by maintainers; the resulting npz is committed.
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from windows import make_windows


def predict(model_dir: Path, x: np.ndarray) -> np.ndarray:
    sig = tf.saved_model.load(str(model_dir)).signatures["serving_default"]
    (out,) = sig(**{sig.structured_input_signature[1].__iter__().__next__(): tf.constant(x)}).values()
    return out.numpy()


def main(root: str, out: str) -> None:
    results = {}
    x = make_windows()
    with tempfile.TemporaryDirectory() as tmp:
        for d in sorted(Path(root).rglob("config.cfg")):
            results[d.parent.name] = predict(d.parent, x)
        for z in sorted(Path(root).rglob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(Path(tmp) / z.stem)
            (cfg,) = (Path(tmp) / z.stem).rglob("config.cfg")
            results[cfg.parent.name] = predict(cfg.parent, x)
    np.savez_compressed(out, **results)
    print({k: v.shape for k, v in results.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
