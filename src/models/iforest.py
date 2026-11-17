"""Isolation Forest.

Idea: build many random decision trees. Unusual points are easy to cut off
from the rest, so they end up alone after only a few cuts. Few cuts = high
anomaly score. Fast, and works well on feature tables.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import Detector


class IForestDetector(Detector):
    name = "iforest"

    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1)

    def fit(self, X, groups=None):
        self.model.fit(np.asarray(X, dtype=np.float64))
        return self

    def score(self, X, groups=None):
        # sklearn gives high values to normal points, so flip the sign
        return -self.model.score_samples(np.asarray(X, dtype=np.float64))
