# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["tensorflow-cpu", "numpy<2", "scipy", "scikit-image", "tifffile", "roifile"]
# ///
"""Reference detection results produced by the *original* DeXtrusion code.

Usage: ``uv run --script make_e2e_reference.py <original_repo> <DeXNets_dir> <out.npz>``

The original ``DeXtrusion.py`` is executed unchanged; only its ``Network`` class (TF 2.8 Keras
loading, unavailable on current TensorFlow) is replaced by a stub that evaluates the same
SavedModel through ``tf.saved_model``. numpy < 2 is pinned because the original float16
accumulation relies on numpy 1.x casting rules.
"""

import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import scipy.ndimage
import tensorflow as tf
import tifffile
from synthetic_movie import make_movie, make_probamap

# --- shims so the unmodified original module imports on current libraries ------------------
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
if not hasattr(scipy.ndimage, "measurements"):
    scipy.ndimage.measurements = scipy.ndimage


class Network:  # stands in for dextrusion.Network.Network
    def __init__(self, verbose=True):
        self.model = None

    def reset(self, model_path):
        self.model = tf.saved_model.load(str(model_path)).signatures["serving_default"]

    def predict_batch(self, x):
        key = next(iter(self.model.structured_input_signature[1]))
        outs = [self.model(**{key: tf.constant(x[i : i + 512].astype(np.float32))}) for i in range(0, len(x), 512)]
        return np.concatenate([next(iter(o.values())).numpy() for o in outs])


def main(repo: str, nets: str, out: str) -> None:
    sys.path.insert(0, str(Path(repo) / "src"))
    pkg = types.ModuleType("dextrusion.Network")
    pkg.Network = Network
    sys.modules["dextrusion.Network"] = pkg
    from dextrusion.DeXtrusion import DeXtrusion  # original code

    movie = make_movie()
    result = {"movie": movie}
    with tempfile.TemporaryDirectory() as tmp:
        tifffile.imwrite(Path(tmp) / "movie.tif", movie)
        for name, rel, kwargs in [
            ("single", ["notum_all/notumAll0"], {}),
            ("ensemble", ["notum_all/notumAll0", "notum_all/notumAll1"], {}),
            ("rescaled", ["notum_all/notumAll0"], {"cell_diameter": 40, "extrusion_duration": 9}),
        ]:
            dex = DeXtrusion(verbose=False)
            dex.detect_events_onmovie(
                str(Path(tmp) / "movie.tif"), models=[str(Path(nets) / r) for r in rel],
                dxy=25, dz=2, group_size=1000, outfolder=str(Path(tmp) / name), **kwargs)
            result[f"{name}_raw"] = dex.probamap
            import dextrusion.RoiUtils as ru
            for tag, vol, proba in [("", 50, 100), ("loose_", 0, 0)]:
                dex.get_rois(cat=None, volume_threshold=vol, proba_threshold=proba, disxy=10, dist=4)
                for cat in range(1, dex.ncat):
                    rois = ru.read_rois(dex.outname + dex.catnames[cat])
                    result[f"{name}_{tag}rois_{cat}"] = np.array(rois, dtype=int).reshape(-1, 3)
    # post-processing alone, on a rich synthetic probability map (shape differs from the movie,
    # which also exercises the position rescaling)
    dex = DeXtrusion(verbose=False)
    dex.ncat = 4
    dex.catnames = ["", "_cell_death.zip", "_cell_sop.zip", "_cell_division.zip"]
    dex.probamap = make_probamap()
    dex.init_shape = (80, 192, 192)
    with tempfile.TemporaryDirectory() as tmp:
        dex.outname = os.path.join(tmp, "synth")
        result["post_probamap"] = dex.probamap
        for tag, vol, proba in [("", 800, 180), ("loose_", 0, 0), ("mid_", 150, 120)]:
            dex.get_rois(cat=None, volume_threshold=vol, proba_threshold=proba, disxy=10, dist=4)
            for cat in range(1, 4):
                rois = ru.read_rois(dex.outname + dex.catnames[cat])
                result[f"post_{tag}rois_{cat}"] = np.array(rois, dtype=int).reshape(-1, 3)
        result["post_clean_1"] = dex.clean_probamap(None, cat=1, threshold=125, mindxy=10, mindt=4,
                                                    volume_threshold=150, proba_threshold=120)
    np.savez_compressed(out, **result)
    print({k: v.shape for k, v in result.items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
