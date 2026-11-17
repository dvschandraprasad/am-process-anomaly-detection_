"""Small 1D convolutional autoencoder (PyTorch).

Idea: the network gets a short piece of the raw signal (one segment), squeezes
it through a narrow middle, and tries to rebuild it. It is trained on normal
segments only, so it learns to rebuild normal shapes well. A segment it cannot
rebuild well is unusual: anomaly score = reconstruction error.

Unlike the other methods it sees the raw frames along the scan path, not
summary features.

Extra "context" channels (for example how much metal is below the laser) can
be given as input. The network may use them, but it is only asked to rebuild
the measured signals.
"""
from __future__ import annotations

import numpy as np

from .base import Detector

try:  # PyTorch is only needed for this one method
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def _build_net(n_in: int, n_channels: int):
    return nn.Sequential(
        # encoder: window -> half length -> quarter length, few channels
        nn.Conv1d(n_in, 16, kernel_size=5, padding=2), nn.ReLU(),
        nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
        nn.Conv1d(16, 4, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        # decoder: back to the full length
        nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
        nn.Conv1d(4, 16, kernel_size=5, padding=2), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
        nn.Conv1d(16, n_channels, kernel_size=5, padding=2),
    )


class AutoencoderDetector(Detector):
    name = "autoencoder"
    input_kind = "windows"

    def __init__(self, epochs: int = 20, batch_size: int = 256, lr: float = 1e-3,
                 seed: int = 0, device: str | None = None, n_context: int = 0):
        if not HAS_TORCH:
            raise ImportError("The autoencoder needs PyTorch: pip install torch")
        self.epochs, self.batch_size, self.lr, self.seed = epochs, batch_size, lr, seed
        self.n_context = n_context  # the last `n_context` channels are context
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare(self, X: np.ndarray):
        # X: (segments, window, signals) -> (segments, signals, window), standardised
        X = (np.asarray(X, dtype=np.float32) - self.mean_) / self.std_
        return torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1)))

    def fit(self, X, groups=None):
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] % 4 != 0:
            raise ValueError("The autoencoder needs a window length that divides by 4")
        self.mean_ = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True)
        self.std_ = np.where(std > 0, std, 1.0).astype(np.float32)

        torch.manual_seed(self.seed)
        self.n_signals_ = X.shape[2] - self.n_context
        self.net_ = _build_net(X.shape[2], self.n_signals_).to(self.device)
        data = self._prepare(X)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(self.seed)
        self.net_.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(data), generator=gen)
            for i in range(0, len(data), self.batch_size):
                batch = data[order[i: i + self.batch_size]].to(self.device)
                loss = ((self.net_(batch) - batch[:, : self.n_signals_]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def score(self, X, groups=None):
        data = self._prepare(X)
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(data), 4096):
                batch = data[i: i + 4096].to(self.device)
                err = ((self.net_(batch) - batch[:, : self.n_signals_]) ** 2).mean(dim=(1, 2))
                out.append(err.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
