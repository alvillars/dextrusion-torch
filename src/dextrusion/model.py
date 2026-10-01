"""DeXNet: per-frame CNN followed by a GRU and a small decision head."""

from __future__ import annotations

import torch
from torch import nn


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    # Keras layout: conv, relu, conv, relu, batchnorm (momentum .95, eps 1e-3)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout, eps=1e-3, momentum=0.05),
    )


class DeXNet(nn.Module):
    """Input ``(B, T, 1, H, W)`` in [0, 1]; output class logits ``(B, ncat)``."""

    def __init__(self, ncat: int = 4, nb_filters: int = 16):
        super().__init__()
        f = nb_filters
        self.ncat = ncat
        self.nb_filters = f
        self.stage1 = _conv_block(1, f)
        self.stage2 = _conv_block(f, 2 * f)
        self.stage3 = _conv_block(2 * f, 4 * f)
        self.stage4 = _conv_block(4 * f, 8 * f)
        self.pool = nn.MaxPool2d(2)
        self.gru = nn.GRU(8 * f, 64, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, ncat)
        self.init_like_keras()

    def init_like_keras(self) -> None:
        """Keras defaults of the original model: Glorot-uniform kernels, orthogonal recurrent
        kernel, zero biases (matters when training from scratch, not when loading weights)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame latent vectors ``(B, T, 8*nb_filters)``."""
        b, t = x.shape[:2]
        x = x.reshape(b * t, *x.shape[2:])
        x = self.pool(self.stage1(x))
        x = self.pool(self.stage2(x))
        x = self.pool(self.stage3(x))
        x = self.stage4(x).amax(dim=(2, 3))
        return x.reshape(b, t, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.gru(self.features(x))
        h = self.drop(seq[:, -1])
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return self.out(h)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            return torch.softmax(self(x), dim=-1)
        finally:
            self.train(was_training)
