"""Fixed limits: the traditional rule-based check (the baseline).

For every feature we learn the normal mean and standard deviation, separately
for each machine setting ("regime", e.g. contour and infill). A segment's
score is how many standard deviations its worst feature is away from the mean.

  score > k  <=>  at least one feature is outside  mean +/- k * std

This is what "limit-based evaluation" means on a production line. Every other
method has to beat it.
"""
from __future__ import annotations

import numpy as np

from .base import Detector


class LimitsDetector(Detector):
    name = "limits"

    def __init__(self, columns: list[int] | None = None, per_group: bool = True,
                 fixed_k: float | None = None):
        self.columns = columns          # which feature columns get a limit
        self.per_group = per_group
        self.fixed_threshold = fixed_k  # e.g. 3.0 for the classic 3-sigma rule
        self.stats_: dict = {}

    def _select(self, X: np.ndarray) -> np.ndarray:
        return X if self.columns is None else X[:, self.columns]

    def fit(self, X, groups=None):
        X = self._select(np.asarray(X, dtype=np.float64))
        if not self.per_group or groups is None:
            groups = np.zeros(len(X), dtype=np.int64)
        self.stats_ = {}
        for g in np.unique(groups):
            rows = X[groups == g]
            std = rows.std(axis=0)
            self.stats_[g] = (rows.mean(axis=0), np.where(std > 0, std, 1.0))
        std_all = X.std(axis=0)
        self.fallback_ = (X.mean(axis=0), np.where(std_all > 0, std_all, 1.0))
        return self

    def score(self, X, groups=None):
        X = self._select(np.asarray(X, dtype=np.float64))
        if not self.per_group or groups is None:
            groups = np.zeros(len(X), dtype=np.int64)
        out = np.zeros(len(X))
        for g in np.unique(groups):
            mean, std = self.stats_.get(g, self.fallback_)  # unseen setting: use overall limits
            m = groups == g
            out[m] = np.abs((X[m] - mean) / std).max(axis=1)
        return out
