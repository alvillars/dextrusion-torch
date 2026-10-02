import numpy as np
import pytest
import tifffile
import torch
from synthetic_movie import make_movie

from dextrusion.cli import main
from dextrusion.config import DeXConfig, extend_catnames
from dextrusion.io import create_roi, load_model, write_rois
from dextrusion.label import save_points
from dextrusion.model import DeXNet

BASE = ["", "_cell_death.zip", "_cell_sop.zip", "_cell_division.zip"]
NEW = [*BASE, "_cell_delamination.zip"]


# ------------------------------------------------------------------------------------ model
def test_extra_classes_keep_existing_logits():
    torch.manual_seed(0)
    old = DeXNet(4, 4).eval()
    new = old.with_extra_classes(6).eval()
    x = torch.rand(5, 10, 1, 45, 45)
    assert new.ncat == 6 and new.out.weight.shape == (6, 16)
    torch.testing.assert_close(new(x)[:, :4], old(x))  # same logits for the existing classes
    # everything below the output layer is a copy
    for k, v in old.state_dict().items():
        if not k.startswith("out."):
            torch.testing.assert_close(new.state_dict()[k], v)
    assert old.ncat == 4  # original untouched
    with pytest.raises(ValueError):
        old.with_extra_classes(3)
    assert old.with_extra_classes(4).ncat == 4  # same size is allowed


def test_freeze_cnn_keeps_weights_and_batchnorm_statistics():
    torch.manual_seed(0)
    m = DeXNet(4, 4)
    m.freeze_cnn()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    for _ in range(3):
        m.train()
        assert all(not s.training for s in m._stages())  # BatchNorm stays in eval mode
        loss = torch.nn.functional.cross_entropy(m(torch.rand(6, 10, 1, 45, 45)), torch.tensor([0, 1, 2, 3, 0, 1]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = m.state_dict()
    for k in before:
        same = torch.equal(before[k], after[k])
        if k.startswith("stage"):
            assert same, f"{k} changed although the CNN is frozen"
    assert not torch.equal(before["gru.weight_ih_l0"], after["gru.weight_ih_l0"])
    assert not torch.equal(before["out.weight"], after["out.weight"])
    m.predict_proba(torch.rand(2, 10, 1, 45, 45))  # still usable, restores the training flag
    assert m.training


def test_extend_catnames():
    assert extend_catnames(BASE, NEW) == NEW
    assert extend_catnames(BASE, BASE) == BASE
    with pytest.raises(ValueError, match="start with"):
        extend_catnames(BASE, ["", "_cell_sop.zip", "_cell_death.zip"])
    with pytest.raises(ValueError, match="start with"):
        extend_catnames(BASE, ["", "_cell_delamination.zip"])
    with pytest.raises(ValueError, match="duplicate"):
        extend_catnames(BASE, [*BASE, "_cell_death.zip"])


# ------------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    d = tmp_path_factory.mktemp("ft")
    movie = make_movie()
    tifffile.imwrite(d / "m.tif", movie)
    events = {"_cell_death.zip": [(10, 40, 45), (14, 70, 30)], "_cell_sop.zip": [(22, 90, 80)],
              "_cell_division.zip": [(28, 35, 95), (18, 60, 60)]}
    for suffix, pts in events.items():
        write_rois(d / f"m{suffix}", [create_roi(p, 1) for p in pts], verbose=False)
    save_points(d / "m_cell_delamination.zip", [(12, 100, 100), (24, 30, 30), (16, 50, 100)], movie.shape)
    return d


@pytest.fixture(scope="module")
def base_net(folder, tmp_path_factory):
    out = tmp_path_factory.mktemp("base") / "net"
    code = main(["-q", "train", str(folder), "-o", str(out), "--nb-filters", "4", "--batch-size", "8",
                 "--epochs", "1", "--naug", "8", "--add-nothing", "0", "--device", "cpu"])
    assert code == 0
    return out


def test_finetune_adds_a_class(folder, base_net, tmp_path):
    out = tmp_path / "ft"
    code = main(["-q", "train", str(folder), "-o", str(out), "--init-from", str(base_net),
                 "--catnames", *NEW, "--epochs", "1", "--naug", "8", "--add-nothing", "0",
                 "--batch-size", "8", "--lr", "0.01", "--device", "cpu"])
    assert code == 0
    model, cfg = load_model(out)
    assert cfg.catnames == NEW and cfg.ncat == 5 and model.ncat == 5
    assert cfg.nb_filters == 4  # geometry follows the starting network
    _, base_cfg = load_model(base_net)
    assert base_cfg.ncat == 4  # the starting network is untouched


def test_finetune_with_frozen_cnn_leaves_cnn_unchanged(folder, base_net, tmp_path):
    out = tmp_path / "ft_frozen"
    main(["-q", "train", str(folder), "-o", str(out), "--init-from", str(base_net), "--catnames", *NEW,
          "--freeze-cnn", "--epochs", "1", "--naug", "8", "--add-nothing", "0", "--batch-size", "8",
          "--lr", "0.01", "--device", "cpu"])
    base, _ = load_model(base_net)
    ft, _ = load_model(out)
    for k, v in base.state_dict().items():
        if k.startswith("stage"):
            assert torch.equal(v, ft.state_dict()[k]), k
    assert not torch.equal(base.state_dict()["gru.weight_ih_l0"], ft.state_dict()["gru.weight_ih_l0"])


def test_finetune_rejects_reordered_classes_and_catnames_without_init(folder, base_net, tmp_path):
    with pytest.raises(SystemExit, match="start with"):
        main(["-q", "train", str(folder), "-o", str(tmp_path / "x"), "--init-from", str(base_net),
              "--catnames", "", "_cell_sop.zip", "_cell_death.zip", "--epochs", "1",
              "--device", "cpu"])
    assert not (tmp_path / "x" / "model.safetensors").exists()


def test_detect_names_the_new_class_files(folder, base_net, tmp_path):
    """A fine-tuned network writes ROI files named after its classes, including the new one."""
    out = tmp_path / "ft"
    main(["-q", "train", str(folder), "-o", str(out), "--init-from", str(base_net), "--catnames", *NEW,
          "--epochs", "1", "--naug", "8", "--add-nothing", "0", "--batch-size", "8", "--device", "cpu"])
    res = tmp_path / "res"
    main(["-q", "detect", str(folder / "m.tif"), "-m", str(out), "-o", str(res), "--device", "cpu",
          "--volume-threshold", "0", "--proba-threshold", "0"])
    names = {p.name for p in res.iterdir()}
    assert "m_cell_delamination.zip" in names and "m_cell_delamination_rawproba.tif" in names
    assert tifffile.imread(res / "m_cell_delamination_rawproba.tif").dtype == np.uint8
    assert DeXConfig.from_json(out / "config.json").catnames == NEW
