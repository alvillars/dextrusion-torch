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
    # event-level scoring: original RoiUtils on seeded random detection / ground-truth sets
    import dextrusion.RoiUtils as ru

    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(14):
            n_gt = [0, 1, 5, 12, 30][i % 5]
            n_det = [0, 3, 9, 20][i % 4]
            gt = np.stack([rng.integers(0, 40, n_gt), rng.integers(0, 120, n_gt),
                           rng.integers(0, 120, n_gt)], axis=1) if n_gt else np.zeros((0, 3), int)
            # detections: jittered copies of some GT events (incl. duplicates), plus random ones
            near = (gt[rng.integers(0, max(n_gt, 1), n_det)] + rng.integers(-6, 7, (n_det, 3))
                    if n_gt else np.zeros((0, 3), int))
            far = np.stack([rng.integers(0, 40, n_det), rng.integers(0, 120, n_det),
                            rng.integers(0, 120, n_det)], axis=1)
            keep = rng.random(n_det) < 0.7
            det = np.where(keep[:, None], near if n_gt else far, far)
            det = np.clip(det, 0, None).astype(int).reshape(-1, 3)
            gt = gt.astype(int).reshape(-1, 3)
            result[f"score_{i}_det"], result[f"score_{i}_gt"] = det, gt
            dfile, gfile = os.path.join(tmp, f"d{i}.zip"), os.path.join(tmp, f"g{i}.zip")
            ru.write_rois(dfile, [ru.create_roi(tuple(map(int, r))) for r in det], verbose=False)
            ru.write_rois(gfile, [ru.create_roi(tuple(map(int, r))) for r in gt], verbose=False)
            for dxy, dt in [(15, 4), (10, 3)]:
                d, g = ru.read_rois(dfile) if n_det else [], ru.read_rois(gfile) if n_gt else []
                res = ru.compare_rois(dfile, gfile, distance_xy=dxy, distance_t=dt) if (
                    n_det and n_gt) else None
                if res is None:  # the original cannot read empty ROI files: use its core functions
                    tp, fp = ru.check_positives(d, g, dxy, dt)
                    fn = ru.check_falseneg(g, d, dxy, dt)
                    res = [tp, fp, fn, tp / (tp + fp) if tp + fp else 0,
                           tp / (tp + fn) if tp + fp and tp + fn else 0]
                result[f"score_{i}_{dxy}_{dt}"] = np.array(res, dtype=float)
                fpl = [(r.position - 1, r.top, r.left) for r in ru.get_falsepositives(d, g, dxy, dt)]
                fnl = [(r.position - 1, r.top, r.left) for r in ru.get_falsenegatives(d, g, dxy, dt)]
                result[f"score_{i}_{dxy}_{dt}_fp"] = np.array(fpl, dtype=int).reshape(-1, 3)
                result[f"score_{i}_{dxy}_{dt}_fn"] = np.array(fnl, dtype=int).reshape(-1, 3)
    np.savez_compressed(out, **result)
    print({k: v.shape for k, v in result.items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
