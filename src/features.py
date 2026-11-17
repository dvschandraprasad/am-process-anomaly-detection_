"""Cut the point data into segments and compute features for each segment.

Words used here:
  * point   = one row = one melt pool camera frame
  * track   = one laser line: frames that follow each other without a time gap
  * segment = `window` frames in a row on one track (the unit we score)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STATS = ("mean", "std", "min", "max", "slope", "jitter")
TRACK_START_FEATURE = "track_start_flag"


@dataclass
class Segments:
    """Segments of one point table.

    df       one row per segment: where it is, its features, label inputs
    windows  raw signal values, shape (n_segments, window, n_signals)
    signals  names of the signals in `windows`
    feature_cols  names of the feature columns in `df`
    context_windows  known-in-advance values per frame (metal below), shape
             (n_segments, window, n_context), or None
    """
    df: pd.DataFrame
    windows: np.ndarray
    signals: list[str]
    feature_cols: list[str]
    context_windows: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.df)


def assign_tracks(points: pd.DataFrame, gap_factor: float = 2.5) -> np.ndarray:
    """Give every point a track number.

    A new track starts when the part or layer changes, or when the time step
    to the previous point is clearly longer than normal (the laser was off).
    In the NIST data frames are 100 us apart and the laser is off for about
    3000 us between tracks. One missing frame (200 us) does not start a new track.
    `points` must be sorted by part, layer and time.
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    t = points["time_us"].to_numpy(dtype=np.float64)
    layer = points["layer"].to_numpy()
    part = points["part"].to_numpy()
    dt = np.diff(t)
    same = (layer[1:] == layer[:-1]) & (part[1:] == part[:-1])
    usual = np.median(dt[same & (dt > 0)]) if np.any(same & (dt > 0)) else 1.0
    new_track = np.ones(n, dtype=bool)
    new_track[1:] = ~same | (dt > gap_factor * usual) | (dt <= 0)
    return np.cumsum(new_track) - 1


def window_starts(track: np.ndarray, window: int, drop_first: int = 0) -> np.ndarray:
    """Row numbers where a full segment starts.

    Segments do not overlap and never cross a track border. Frames left over
    at the end of a track (fewer than `window`) are not used.
    """
    n = len(track)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    is_first = np.ones(n, dtype=bool)
    is_first[1:] = track[1:] != track[:-1]
    first_row = np.maximum.accumulate(np.where(is_first, np.arange(n), 0))
    pos = np.arange(n) - first_row - drop_first
    starts = np.flatnonzero((pos >= 0) & (pos % window == 0))
    ends = starts + window - 1
    ok = ends < n
    starts, ends = starts[ok], ends[ok]
    return starts[track[ends] == track[starts]]


def window_features(windows: np.ndarray, signals: list[str]) -> pd.DataFrame:
    """Summary numbers for each segment and signal.

    mean, std, min, max: level and spread
    slope:  straight-line trend inside the segment (per frame)
    jitter: mean absolute change from one frame to the next
    """
    n, w, _ = windows.shape
    t = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    out: dict[str, np.ndarray] = {}
    for j, sig in enumerate(signals):
        v = windows[:, :, j].astype(np.float64)
        out[f"{sig}_mean"] = v.mean(axis=1)
        out[f"{sig}_std"] = v.std(axis=1)
        out[f"{sig}_min"] = v.min(axis=1)
        out[f"{sig}_max"] = v.max(axis=1)
        out[f"{sig}_slope"] = (v * t).sum(axis=1) / (t ** 2).sum() if w > 1 else np.zeros(n)
        out[f"{sig}_jitter"] = np.abs(np.diff(v, axis=1)).mean(axis=1) if w > 1 else np.zeros(n)
    return pd.DataFrame(out)


def make_segments(
    points: pd.DataFrame,
    signals: list[str],
    context: list[str] | None = None,
    window: int = 12,
    drop_first: int = 1,
    xct_column: str | None = "xct_m3",
    geometry_context: bool = False,
) -> Segments:
    """Cut a point table into segments and compute their features.

    `points` must be sorted by part, layer and time (as `load_part` returns it).
    Extra columns are summarised if they exist:
      solid_below -> mean          (how much metal is under the segment)
      is_edge     -> edge_frac     (share of points at the part surface)
      xct column  -> xct_min, xct_max
      fault_id    -> n_fault, fault_id   (injected faults)

    geometry_context=True also gives `solid_below` to the detectors as an
    input. It is known from the part shape before printing, so using it is
    fair: "normal" may then depend on how much metal is under the laser.
    """
    context = list(context or [])
    points = points.reset_index(drop=True)
    track = assign_tracks(points)
    starts = window_starts(track, window, drop_first)
    idx = starts[:, None] + np.arange(window)[None, :]

    def col(name: str) -> np.ndarray:
        return points[name].to_numpy()[idx]

    windows = np.stack([col(s) for s in signals], axis=2).astype(np.float32) \
        if len(starts) else np.zeros((0, window, len(signals)), dtype=np.float32)

    x, y = col("x"), col("y")
    df = pd.DataFrame({
        "part": points["part"].to_numpy()[starts],
        "layer": points["layer"].to_numpy()[starts],
        "z": points["z"].to_numpy()[starts],
        "x": x.mean(axis=1), "y": y.mean(axis=1),
        "x0": x[:, 0], "y0": y[:, 0], "x1": x[:, -1], "y1": y[:, -1],
        "t_start_us": points["time_us"].to_numpy()[starts],
        "row_start": starts,
        "track": track[starts],
    })

    feats = window_features(windows, signals)
    feature_cols = list(feats.columns)
    for c in context:
        feats[f"{c}_mean"] = col(c).mean(axis=1)
        feature_cols.append(f"{c}_mean")
    df = pd.concat([df, feats], axis=1)

    # "Regime": which machine setting the segment was made with
    # (for example contour at 100 W or infill at 195 W).
    if "cmd_power" in points.columns:
        df["regime"] = (np.round(np.median(col("cmd_power"), axis=1) / 5.0) * 5.0).astype(np.int32)
    else:
        df["regime"] = 0

    context_windows = None
    # The melt pool needs a few frames to grow at the start of every track.
    # Where a track starts is known from the scan plan, so it is fair context.
    is_first = np.ones(len(track), dtype=bool)
    is_first[1:] = track[1:] != track[:-1]
    first_row = np.maximum.accumulate(np.where(is_first, np.arange(len(track)), 0))
    df["track_start"] = (starts - first_row[starts] == drop_first) if len(starts) else np.zeros(0, dtype=bool)
    df[TRACK_START_FEATURE] = df["track_start"].astype(np.float64)
    feature_cols.append(TRACK_START_FEATURE)

    if "solid_below" in points.columns:
        df["solid_below"] = col("solid_below").mean(axis=1)
        if geometry_context:
            feature_cols.append("solid_below")
            context_windows = col("solid_below")[:, :, None].astype(np.float32)
    if "is_edge" in points.columns:
        df["edge_frac"] = col("is_edge").astype(np.float32).mean(axis=1)
    if xct_column and xct_column in points.columns:
        v = col(xct_column)
        df["xct_min"], df["xct_max"] = v.min(axis=1), v.max(axis=1)
    if "fault_id" in points.columns:
        f = col("fault_id")
        df["n_fault"] = (f >= 0).sum(axis=1)
        df["fault_id"] = f.max(axis=1)

    return Segments(df=df, windows=windows, signals=list(signals), feature_cols=feature_cols,
                    context_windows=context_windows)
