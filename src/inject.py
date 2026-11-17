"""Add synthetic faults to normal melt pool signals.

Real defect labels are rare. Injected faults give us anomalies whose place,
kind and size we know exactly, so detection can be measured precisely.

Fault kinds (sizes are in units of the normal spread of each signal):
  spike  a few frames much too high
  drop   a short stretch much too low
  drift  a slow ramp away from normal
  noise  a stretch with much more scatter than normal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import assign_tracks

FAULT_TYPES = ("spike", "drop", "drift", "noise")

# (shortest, longest) length in frames, and size in normal spreads
_SPEC = {
    "spike": {"frames": (3, 6), "size": 5.0},
    "drop": {"frames": (6, 15), "size": 5.0},
    "drift": {"frames": (30, 60), "size": 4.0},
    "noise": {"frames": (16, 40), "size": 3.0},
}


def robust_std(values: np.ndarray) -> float:
    med = np.median(values)
    s = 1.4826 * np.median(np.abs(values - med))
    return float(s) if s > 0 else float(np.std(values))


def _regime(points: pd.DataFrame) -> np.ndarray:
    """Machine setting of every point (commanded power rounded to 5 W)."""
    if "cmd_power" in points.columns:
        return (np.round(points["cmd_power"].to_numpy() / 5.0) * 5.0).astype(np.int64)
    return np.zeros(len(points), dtype=np.int64)


def signal_spreads(points: pd.DataFrame, signals: list[str]) -> dict:
    """Normal spread (robust std) of every signal, per machine setting.

    Returns {setting: {signal: spread}} plus the key "all" for all points
    together. Compute it on TRAINING data and give it to `inject_faults`, so
    that the size of the faults does not depend on the test data.
    """
    regime = _regime(points)
    out: dict = {"all": {s: robust_std(points[s].to_numpy(dtype=np.float64)) for s in signals}}
    for r in np.unique(regime):
        m = regime == r
        out[int(r)] = {s: robust_std(points[s].to_numpy(dtype=np.float64)[m]) for s in signals}
    return out


def inject_faults(
    points: pd.DataFrame,
    signals: list[str],
    events_per_1000_points: float = 0.5,
    severity: float = 1.0,
    types: tuple[str, ...] | list[str] = FAULT_TYPES,
    seed: int = 0,
    spreads: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a copy of `points` with faults added, and a table of the faults.

    Only the columns in `signals` are changed. All signals of one fault move
    together (a bigger melt pool is longer, wider and has more area).
    New column `fault_id` is -1 for untouched points.

    `spreads` (from `signal_spreads` on the training data) sets the size unit
    of the faults. If it is None, the spread of `points` itself is used.
    A machine setting that is not in `spreads` uses its "all" entry.

    The fault table has one row per fault:
    fault_id, fault_type, part, layer, row_start, row_end (end not included).
    """
    rng = np.random.default_rng(seed)
    out = points.reset_index(drop=True).copy()
    n = len(out)
    fault_id = np.full(n, -1, dtype=np.int32)
    if n == 0:
        out["fault_id"] = fault_id
        return out, pd.DataFrame(columns=["fault_id", "fault_type", "part", "layer", "row_start", "row_end"])

    track = assign_tracks(out)
    first_rows = np.flatnonzero(np.r_[True, track[1:] != track[:-1]])
    lengths = np.diff(np.r_[first_rows, n])

    # Normal spread of every signal, per machine setting (contour / infill ...)
    if spreads is None:
        spreads = signal_spreads(out, signals)
    regime = _regime(out)
    values = {s: out[s].to_numpy(dtype=np.float64).copy() for s in signals}
    spread = {s: np.zeros(n) for s in signals}
    for r in np.unique(regime):
        m = regime == r
        known = spreads.get(int(r), spreads["all"])
        for s in signals:
            spread[s][m] = known[s]

    n_events = int(round(events_per_1000_points * n / 1000.0))
    kinds = [types[i % len(types)] for i in range(n_events)]
    rng.shuffle(kinds)

    events = []
    for kind in kinds:
        lo, hi = _SPEC[kind]["frames"]
        for _try in range(50):
            k = rng.choice(len(first_rows), p=lengths / lengths.sum())
            room = lengths[k] - 2  # keep off the first frame of the track
            if room < lo:
                continue
            length = int(rng.integers(lo, min(hi, room) + 1))
            start = int(first_rows[k] + 1 + rng.integers(0, room - length + 1))
            stop = start + length
            # leave a gap to other faults so that they do not merge
            if np.any(fault_id[max(start - 8, 0): stop + 8] >= 0):
                continue
            break
        else:
            continue  # no free place found for this fault

        size = _SPEC[kind]["size"] * severity
        if kind == "spike":
            shape = np.full(length, size)
        elif kind == "drop":
            shape = np.full(length, -size)
        elif kind == "drift":
            shape = np.linspace(0.0, size, length) * rng.choice([-1.0, 1.0])
        else:  # noise: the same random pattern for all signals
            shape = rng.normal(0.0, size, length)

        for s in signals:
            new = values[s][start:stop] + shape * spread[s][start:stop]
            values[s][start:stop] = np.maximum(new, 0.0)  # sizes cannot be negative
        fid = len(events)
        fault_id[start:stop] = fid
        events.append({
            "fault_id": fid, "fault_type": kind,
            "part": int(out["part"].iat[start]), "layer": int(out["layer"].iat[start]),
            "row_start": start, "row_end": stop,
        })

    for s in signals:
        out[s] = values[s].astype(points[s].dtype)
    out["fault_id"] = fault_id
    events_df = pd.DataFrame(
        events, columns=["fault_id", "fault_type", "part", "layer", "row_start", "row_end"]
    )
    return out, events_df
