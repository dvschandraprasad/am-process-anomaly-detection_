"""What every anomaly detector in this project looks like.

All detectors work the same way:
  1. fit(...)   learn "normal" from normal training segments only
  2. score(...) give every segment a number; higher = more unusual

The alarm level (threshold) is not part of the detector. It is chosen later
on validation data, the same way for every method (see evaluate.py).
"""
from __future__ import annotations

import numpy as np


class Detector:
    name = "base"
    #: "features" = works on the feature table, "windows" = works on raw signal windows
    input_kind = "features"
    #: set for rule-based checks that come with their own fixed alarm level
    fixed_threshold: float | None = None

    def fit(self, X: np.ndarray, groups: np.ndarray | None = None) -> "Detector":
        raise NotImplementedError

    def score(self, X: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


def subsample(X: np.ndarray, max_rows: int | None, seed: int) -> np.ndarray:
    """Take at most `max_rows` random rows (keeps slow methods fast)."""
    if max_rows is None or len(X) <= max_rows:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), size=max_rows, replace=False)]
