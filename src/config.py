"""Settings for the pipeline.

A config is a plain nested dict read from a YAML file. Anything missing in the
YAML file is filled in from DEFAULTS below, so a config file only has to list
what is different.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "data": {
        # part id -> folder or .zip with one CSV file per layer
        "parts": {},
        # folder for fast Parquet copies of the loaded parts (None = no cache)
        "cache_dir": "data/processed",
        # [first, last] layer to load, or None for all layers
        "layers": None,
        # NIST Overhang Part X4: 250 layers of 20 micrometres
        "layer_thickness_mm": 0.02,
        # list of column names, only needed if the files do not have the
        # 40 columns described in NIST AMS 100-69
        "columns": None,
    },
    "split": {
        # "by_part": whole parts go to train / val / test (the honest way)
        # "by_layer_blocks": for when only one part is downloaded
        "mode": "by_part",
        "train": [1, 2],
        "val": [3],
        "test": [4],
        "block_layers": 10,
    },
    "segments": {
        # one segment = `window` melt pool frames in a row on one scan track
        "window": 12,
        # the first frame of each track often shows no melt pool yet
        "drop_first": 1,
        # signals that the detectors look at
        "signals": ["mp_length_100", "mp_width_100", "mp_area_100"],
        # machine settings, used only as mean values (they say which
        # "regime" a segment is in, e.g. contour or infill)
        "context": ["cmd_power", "speed"],
        # also tell the detectors how much solid metal is under the laser
        # (known from the part shape). Limits are then set per region.
        "geometry_context": True,
    },
    "labels": {
        "overhang": {
            "cell_mm": 0.25,      # x/y grid size used to check for metal below
            "max_solid_mm": 0.5,  # less solid metal below than this = overhang
            "cap_mm": 1.0,
        },
        "xct": {
            "enabled": True,
            "column": "xct_m3",   # XCT value after the 3x3x3 mean filter
            "pore_is": "low",     # pores are darker ("low") or brighter ("high")
            "threshold": "auto",  # a number, or "auto" (median -/+ k robust std)
            "k": 6.0,
            "ignore_edge": True,  # do not judge segments at the part surface
            # what counts as "at the surface" (see labels.add_xct_surface):
            # closer than this to the outside in the same layer ...
            "surface_xy_mm": 0.5,
            # ... or closer than this to a top face, a face over powder, or
            # the lowest layers of the XCT scan
            "surface_z_mm": 0.06,
        },
    },
    "inject": {
        "seed": 123,
        "events_per_1000_points": 0.5,
        "severity": 1.0,
        "types": ["spike", "drop", "drift", "noise"],
        # a segment counts as faulty if it holds at least this many fault points
        "min_fault_points": 3,
    },
    "methods": ["limits", "iforest", "lof", "ocsvm", "autoencoder"],
    "method_params": {
        "limits": {},
        "iforest": {"n_estimators": 200},
        "lof": {"n_neighbors": 20, "max_train": 20000},
        "ocsvm": {"nu": 0.01, "max_train": 5000},
        "autoencoder": {"epochs": 20, "batch_size": 256, "lr": 1e-3},
    },
    "seeds": [0, 1, 2],
    "threshold": {
        # the alarm level of every method is set so that this share of the
        # normal validation segments is flagged
        "target_frr": 0.01,
    },
    "output_dir": "results/run",
}


def deep_merge(base: dict, extra: dict) -> dict:
    """Return a copy of `base` where values from `extra` win."""
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        # "parts" is a free mapping (part id -> path); never merge it with defaults
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "parts":
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Read a YAML config file and fill in the defaults."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["data"]["parts"] = {int(k): str(v) for k, v in cfg["data"]["parts"].items()}
    return cfg
