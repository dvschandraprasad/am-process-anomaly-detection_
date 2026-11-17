"""Anomaly detectors. `build_detector` makes one from its name."""
from __future__ import annotations

from .autoencoder import HAS_TORCH, AutoencoderDetector
from .base import Detector
from .iforest import IForestDetector
from .limits import LimitsDetector
from .lof import LOFDetector, OCSVMDetector

#: methods whose result changes with the random seed
RANDOM_METHODS = {"iforest", "lof", "ocsvm", "autoencoder"}

#: names shown in tables and plots
LABELS = {
    "limits_3sigma": "Fixed limits (3-sigma rule)",
    "limits": "Fixed limits (tuned k)",
    "limits_levels": "Fixed limits, level features only (tuned k)",
    "iforest": "Isolation Forest",
    "lof": "Local Outlier Factor",
    "ocsvm": "One-Class SVM",
    "autoencoder": "Conv. autoencoder",
}


def build_detector(name: str, seed: int = 0, params: dict | None = None,
                   limit_columns: list[int] | None = None) -> Detector:
    params = dict(params or {})
    if name == "limits":
        return LimitsDetector(columns=limit_columns, **params)
    if name == "limits_levels":
        det = LimitsDetector(columns=limit_columns, **params)
        det.name = "limits_levels"
        return det
    if name == "limits_3sigma":
        det = LimitsDetector(columns=limit_columns, fixed_k=3.0, **params)
        det.name = "limits_3sigma"
        return det
    if name == "iforest":
        return IForestDetector(seed=seed, **params)
    if name == "lof":
        return LOFDetector(seed=seed, **params)
    if name == "ocsvm":
        return OCSVMDetector(seed=seed, **params)
    if name == "autoencoder":
        return AutoencoderDetector(seed=seed, **params)
    raise ValueError(f"Unknown method: {name}")


__all__ = ["Detector", "build_detector", "HAS_TORCH", "LABELS", "RANDOM_METHODS",
           "LimitsDetector", "IForestDetector", "LOFDetector", "OCSVMDetector",
           "AutoencoderDetector"]
