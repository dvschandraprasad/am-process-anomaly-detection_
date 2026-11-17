"""Local Outlier Factor and One-Class SVM.

Both need features on the same scale, so the features are standardised first
(mean 0, std 1, learned on the training data).

Local Outlier Factor (LOF): compares how crowded the area around a point is
with how crowded it is around its nearest training neighbours. A point in a
much emptier area than its neighbours is unusual.

One-Class SVM: draws a tight border around the normal training data.
Points outside the border are unusual.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .base import Detector, subsample


class LOFDetector(Detector):
    name = "lof"

    def __init__(self, n_neighbors: int = 20, max_train: int | None = 20000, seed: int = 0):
        self.max_train, self.seed = max_train, seed
        self.n_neighbors = n_neighbors

    def fit(self, X, groups=None):
        X = subsample(np.asarray(X, dtype=np.float64), self.max_train, self.seed)
        k = max(1, min(self.n_neighbors, len(X) - 1))
        self.model = make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=k, novelty=True))
        self.model.fit(X)
        return self

    def score(self, X, groups=None):
        return -self.model.score_samples(np.asarray(X, dtype=np.float64))


class OCSVMDetector(Detector):
    name = "ocsvm"

    def __init__(self, nu: float = 0.01, max_train: int | None = 5000, seed: int = 0):
        self.max_train, self.seed = max_train, seed
        self.model = make_pipeline(StandardScaler(), OneClassSVM(nu=nu, kernel="rbf", gamma="scale"))

    def fit(self, X, groups=None):
        self.model.fit(subsample(np.asarray(X, dtype=np.float64), self.max_train, self.seed))
        return self

    def score(self, X, groups=None):
        return -self.model.score_samples(np.asarray(X, dtype=np.float64))
