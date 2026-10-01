"""Map the variables of a legacy Keras DeXNet (TF 2.x SavedModel) onto :class:`DeXNet`.

The Keras model stores its weights as a flat ``variables/N`` list (convnet trainable weights,
then convnet BatchNorm moving statistics, then GRU) plus ``layer_with_weights-K`` entries for the
Dense layers. Every array is shape-checked, so a model that does not follow the DeXNet layout
fails loudly instead of loading garbage.
"""

from __future__ import annotations

import numpy as np
import torch

from ..model import DeXNet

N_STAGES = 4
GRU_UNITS = 64


def _get(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    try:
        return arrays[key]
    except KeyError as e:
        raise ValueError(f"variable '{key}' not found: not a DeXNet SavedModel?") from e


def _check(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(arr.shape)}")
    return arr


def infer_architecture(arrays: dict[str, np.ndarray]) -> tuple[int, int]:
    """Return ``(ncat, nb_filters)`` from the stored shapes."""
    first = _get(arrays, "variables/0")
    nb_filters = int(first.shape[-1])
    ncat = int(_get(arrays, "layer_with_weights-4/bias").shape[0])
    return ncat, nb_filters


def gru_keras_to_torch(
    kernel: np.ndarray, recurrent: np.ndarray, bias: np.ndarray
) -> dict[str, np.ndarray]:
    """Keras GRU (reset_after=True, gate order z,r,h) -> torch GRU (gate order r,z,n)."""
    u = recurrent.shape[0]

    def reorder(a: np.ndarray) -> np.ndarray:
        z, r, h = a[..., :u], a[..., u : 2 * u], a[..., 2 * u :]
        return np.concatenate([r, z, h], axis=-1)

    return {
        "weight_ih_l0": reorder(kernel).T,
        "weight_hh_l0": reorder(recurrent).T,
        "bias_ih_l0": reorder(bias[0]),
        "bias_hh_l0": reorder(bias[1]),
    }


def keras_arrays_to_state_dict(arrays: dict[str, np.ndarray]) -> tuple[dict, int, int]:
    """Convert raw Keras arrays to a torch ``state_dict``; returns ``(state, ncat, nb_filters)``."""
    ncat, f = infer_architecture(arrays)
    chans = [1] + [f * 2**i for i in range(N_STAGES)]
    sd: dict[str, np.ndarray] = {}

    for s in range(N_STAGES):
        cin, cout = chans[s], chans[s + 1]
        base = 6 * s
        p = f"stage{s + 1}"
        for j, (ci, name) in enumerate([(cin, "0"), (cout, "2")]):
            k = _check(f"variables/{base + 2 * j}", _get(arrays, f"variables/{base + 2 * j}"), (3, 3, ci, cout))
            b = _check(f"variables/{base + 2 * j + 1}", _get(arrays, f"variables/{base + 2 * j + 1}"), (cout,))
            sd[f"{p}.{name}.weight"] = k.transpose(3, 2, 0, 1)  # (kh,kw,in,out)->(out,in,kh,kw)
            sd[f"{p}.{name}.bias"] = b
        sd[f"{p}.4.weight"] = _check("bn gamma", _get(arrays, f"variables/{base + 4}"), (cout,))
        sd[f"{p}.4.bias"] = _check("bn beta", _get(arrays, f"variables/{base + 5}"), (cout,))
        mean = _check("bn mean", _get(arrays, f"variables/{6 * N_STAGES + 2 * s}"), (cout,))
        var = _check("bn var", _get(arrays, f"variables/{6 * N_STAGES + 2 * s + 1}"), (cout,))
        if (var < 0).any():
            raise ValueError("BatchNorm moving variance has negative values: unexpected variable order")
        sd[f"{p}.4.running_mean"] = mean
        sd[f"{p}.4.running_var"] = var
        sd[f"{p}.4.num_batches_tracked"] = np.zeros((), dtype=np.int64)

    g0 = 6 * N_STAGES + 2 * N_STAGES
    fin = chans[-1]
    gru = gru_keras_to_torch(
        _check("gru kernel", _get(arrays, f"variables/{g0}"), (fin, 3 * GRU_UNITS)),
        _check("gru recurrent", _get(arrays, f"variables/{g0 + 1}"), (GRU_UNITS, 3 * GRU_UNITS)),
        _check("gru bias", _get(arrays, f"variables/{g0 + 2}"), (2, 3 * GRU_UNITS)),
    )
    sd.update({f"gru.{k}": v for k, v in gru.items()})

    for idx, name, shape in [(2, "fc1", (GRU_UNITS, 32)), (3, "fc2", (32, 16)), (4, "out", (16, ncat))]:
        k = _check(name, _get(arrays, f"layer_with_weights-{idx}/kernel"), shape)
        b = _check(name, _get(arrays, f"layer_with_weights-{idx}/bias"), (shape[1],))
        sd[f"{name}.weight"] = k.T
        sd[f"{name}.bias"] = b
    return sd, ncat, f


def to_torch_state(sd: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}


def build_model(arrays: dict[str, np.ndarray]) -> DeXNet:
    sd, ncat, f = keras_arrays_to_state_dict(arrays)
    model = DeXNet(ncat=ncat, nb_filters=f)
    model.load_state_dict(to_torch_state(sd), strict=True)
    return model.eval()
