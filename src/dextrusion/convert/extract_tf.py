# /// script
# requires-python = ">=3.10"
# dependencies = ["tensorflow-cpu", "numpy"]
# ///
"""TensorFlow side of the DeXNet conversion.

Reads the weights of a legacy Keras SavedModel directory with ``tf.train.load_checkpoint``
(no Keras model is built, so any TensorFlow >= 2.x works) and writes them to an ``.npz``.

Usage: ``uv run --script extract_tf.py <saved_model_dir> <out.npz>``
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"


def main(model_dir: str, out: str) -> None:
    prefix = os.path.join(model_dir, "variables", "variables")
    if not os.path.exists(prefix + ".index"):
        raise SystemExit(f"{model_dir}: no variables/variables.index, not a Keras SavedModel")
    reader = tf.train.load_checkpoint(prefix)
    arrays = {}
    for key in reader.get_variable_to_shape_map():
        if not key.endswith(SUFFIX):
            continue
        name = key[: -len(SUFFIX)]
        if ".OPTIMIZER" in name:
            continue
        if name.startswith(("variables/", "layer_with_weights-")):
            arrays[name] = reader.get_tensor(key)
    np.savez(out, **{k.replace("/", "|"): v for k, v in arrays.items()})
    print(f"extracted {len(arrays)} arrays")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
