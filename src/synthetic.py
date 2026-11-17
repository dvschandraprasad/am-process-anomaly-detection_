"""Make small FAKE data with the same layout as the NIST registered files.

This is only for tests and for the quick start, so the code can run without
the big download. The numbers are invented by simple rules. Results on this
data say nothing about real printing.

What the fake part looks like (a small copy of the NIST overhang part):
  * a block, scanned line by line; the scan direction turns 90 degrees
    every layer
  * one side grows outwards with height (a 45 degree overhang)
  * first and last scan line of a layer use "contour" settings (100 W),
    the rest use "infill" settings (195 W)
  * the melt pool is bigger where there is little solid metal below
  * a few pores: XCT is dark there, and for some of them the melt pool
    was also odd while printing
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .load import COLUMNS

FRAME_US = 100.0  # 10 000 frames per second


@dataclass
class PartSpec:
    lx: float = 4.0               # length of the block in X [mm]
    ly: float = 2.5               # width in Y [mm]
    n_layers: int = 12
    layer_mm: float = 0.08
    overhang_start_mm: float = 0.24   # height where the overhang starts
    overhang_max_mm: float = 0.72     # how far it grows outwards
    hatch_mm: float = 0.125           # distance between scan lines
    n_pores: int = 6


def _inside(spec: PartSpec, x: np.ndarray, y: np.ndarray, z: np.ndarray | float) -> np.ndarray:
    grow = np.clip(np.asarray(z) - spec.overhang_start_mm, 0.0, spec.overhang_max_mm)
    return (x >= 0) & (x <= spec.lx + grow) & (y >= 0) & (y <= spec.ly)


def _solid_below(spec: PartSpec, x, y, z: float, cap: float = 1.0) -> np.ndarray:
    """Thickness of metal straight under each point (the truth, by geometry)."""
    solid = np.zeros(len(x))
    alive = np.ones(len(x), dtype=bool)
    for step in range(1, int(np.ceil(cap / spec.layer_mm)) + 1):
        zz = z - step * spec.layer_mm
        if zz > 1e-9:
            alive &= _inside(spec, x, y, zz)
        solid += spec.layer_mm * alive
    return np.minimum(solid, cap)


def _scan_layer(spec: PartSpec, layer: int) -> dict[str, np.ndarray]:
    """Laser positions, settings and times for one layer of one part."""
    z = layer * spec.layer_mm
    x_max = spec.lx + float(np.clip(z - spec.overhang_start_mm, 0.0, spec.overhang_max_mm))
    along_x = layer % 2 == 0
    across = spec.ly if along_x else x_max
    length = x_max if along_x else spec.ly
    lines = np.arange(spec.hatch_mm / 2, across, spec.hatch_mm)

    xs, ys, ps, vs, ts = [], [], [], [], []
    t = 0.0
    for k, c in enumerate(lines):
        contour = k == 0 or k == len(lines) - 1
        power, speed = (100.0, 900.0) if contour else (195.0, 800.0)
        step = speed * FRAME_US * 1e-6
        s = np.arange(step / 2, length, step)
        if k % 2 == 1:
            s = s[::-1]
        x, y = (s, np.full_like(s, c)) if along_x else (np.full_like(s, c), s)
        xs.append(x); ys.append(y)
        ps.append(np.full(len(s), power)); vs.append(np.full(len(s), speed))
        ts.append(t + FRAME_US * np.arange(len(s)))
        t += FRAME_US * len(s) + 600.0  # laser off while jumping to the next line
    return {
        "x": np.concatenate(xs), "y": np.concatenate(ys), "z": z,
        "cmd_power": np.concatenate(ps), "cmd_speed": np.concatenate(vs),
        "t": np.concatenate(ts),
    }


def make_part(part: int, spec: PartSpec, seed: int = 0) -> pd.DataFrame:
    """Make the point table of one fake part (same columns as `load_part`)."""
    rng = np.random.default_rng(seed * 1000 + part)
    part_gain = 1.0 + rng.normal(0.0, 0.01)       # parts are not exactly equal
    offset = np.array([-12.0 + 8.0 * (part - 1), -6.0])  # place on the build plate

    # pores: half of them in the overhang, where printing is harder
    pores = []
    z_top = spec.n_layers * spec.layer_mm
    for i in range(spec.n_pores):
        pz = rng.uniform(spec.overhang_start_mm + 0.1, z_top) if i % 2 == 0 else rng.uniform(0.1, z_top)
        grow = float(np.clip(pz - spec.overhang_start_mm, 0.0, spec.overhang_max_mm))
        px = rng.uniform(spec.lx, spec.lx + max(grow - 0.2, 0.05)) if i % 2 == 0 else rng.uniform(0.4, spec.lx - 0.4)
        py = rng.uniform(0.4, spec.ly - 0.4)
        pores.append({"p": np.array([px, py, pz]), "r": rng.uniform(0.08, 0.16),
                      "visible": rng.random() < 0.7,          # seen in the melt pool?
                      "factor": rng.choice([0.7, 1.4])})

    tables = []
    for layer in range(1, spec.n_layers + 1):
        sc = _scan_layer(spec, layer)
        x, y, z = sc["x"], sc["y"], sc["z"]
        n = len(x)
        solid = _solid_below(spec, x, y, z)
        infill = sc["cmd_power"] > 150

        # melt pool area: level by setting, bigger over powder, short dip at track start
        area = np.where(infill, 0.030, 0.012) * part_gain
        area = area * (1.0 + 0.5 * np.clip(1.0 - solid / 0.5, 0.0, 1.0))
        new_track = np.r_[True, np.diff(sc["t"]) > 1.5 * FRAME_US]
        pos = np.arange(n) - np.maximum.accumulate(np.where(new_track, np.arange(n), 0))
        area = area * np.select([pos == 1, pos == 2, pos == 3], [0.65, 0.75, 0.93], 1.0)  # grows over a few frames
        noise = lfilter([1.0], [1.0, -0.6], rng.normal(0.0, 0.05, n))  # smooth scatter
        area = area * (1.0 + noise)

        xct = rng.normal(42000.0, 1200.0, n)
        d_edge = np.minimum.reduce([x, y, spec.ly - y,
                                    spec.lx + np.clip(z - spec.overhang_start_mm, 0, spec.overhang_max_mm) - x])
        xct = np.where(d_edge < 0.05, xct - 12000.0, xct)  # surface voxels mix metal and air
        for pore in pores:
            d = np.sqrt((x - pore["p"][0]) ** 2 + (y - pore["p"][1]) ** 2 + (z - pore["p"][2]) ** 2)
            xct = np.where(d < pore["r"], rng.normal(16000.0, 2500.0, n), xct)
            if pore["visible"]:
                area = np.where(d < 1.2 * pore["r"], area * pore["factor"], area)
        area = np.maximum(area, 1e-4)
        area = np.where(pos == 0, 0.0, area)   # real data: no melt pool in the first frame

        root = np.sqrt(area)
        length = 1.6 * root * (1.0 + rng.normal(0.0, 0.02, n))
        width = 0.8 * root * (1.0 + rng.normal(0.0, 0.02, n))
        lwi = {name: np.clip(rng.normal(120.0, 12.0, n) + 30.0 * (solid < 0.5) * ("scan" in name), 0, 255)
               for name in ("powder_a", "powder_b", "powder_c", "scan_a", "scan_b", "scan_c")}

        data = {
            "part": np.full(n, part), "time_us": layer * 30e6 + (part - 1) * 2e6 + sc["t"],
            "x": x + offset[0], "y": y + offset[1],
            "cmd_power": sc["cmd_power"], "cmd_speed": sc["cmd_speed"],
            "x_real": x + offset[0] + rng.normal(0.0, 0.003, n),
            "y_real": y + offset[1] + rng.normal(0.0, 0.003, n),
            "power": sc["cmd_power"] + rng.normal(0.0, 0.5, n),
            "speed": sc["cmd_speed"] + rng.normal(0.0, 5.0, n),
        }
        for thr, a_gain in ((80, 1.25), (100, 1.0), (120, 0.8)):
            g = np.sqrt(a_gain)
            data[f"mp_length_{thr}"] = length * g
            data[f"mp_width_{thr}"] = width * g
            data[f"mp_area_{thr}"] = area * a_gain
        for name, v in lwi.items():
            data[f"lwi_{name}"] = np.round(v)
            data[f"lwi_{name}_m3"] = np.round(0.7 * v + 36.0)
            data[f"lwi_{name}_m5"] = np.round(0.5 * v + 60.0)
        xct = xct / 78000.0                    # real data: XCT values lie between 0 and 1
        data["xct"] = xct
        data["xct_m3"] = 0.8 * xct + 0.2 * 42000.0 / 78000.0
        data["xct_m5"] = 0.6 * xct + 0.4 * 42000.0 / 78000.0

        df = pd.DataFrame(data)[COLUMNS]
        df.insert(0, "layer", np.int32(layer))
        tables.append(df)

    points = pd.concat(tables, ignore_index=True)
    points["part"] = points["part"].astype(np.int32)
    points["z"] = (points["layer"] * spec.layer_mm).astype(np.float32)
    for col in points.columns:
        if col not in ("part", "layer", "time_us"):
            points[col] = points[col].astype(np.float32)
    return points


_FMT = {"part": "%d", "time_us": "%d", "cmd_power": "%.1f", "cmd_speed": "%.1f",
        "power": "%.2f", "speed": "%.1f"}


def write_part(points: pd.DataFrame, folder: str | Path, header: bool = False, gz: bool = True) -> None:
    """Write one part the way NIST stores it: one CSV file per layer, 40 columns."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fmt = []
    for c in COLUMNS:
        if c in _FMT:
            fmt.append(_FMT[c])
        elif c.startswith("lwi_"):
            fmt.append("%d")
        elif c.startswith("xct"):
            fmt.append("%.5f")
        elif "area" in c:
            fmt.append("%.5f")
        else:
            fmt.append("%.4f")
    for layer, rows in points.groupby("layer"):
        name = folder / f"layer_{int(layer):04d}.csv{'.gz' if gz else ''}"
        np.savetxt(name, rows[COLUMNS].to_numpy(dtype=np.float64), fmt=fmt, delimiter=",",
                   header=",".join(COLUMNS) if header else "", comments="")


def make_dataset(n_parts: int = 4, spec: PartSpec | None = None, seed: int = 0) -> dict[int, pd.DataFrame]:
    spec = spec or PartSpec()
    return {p: make_part(p, spec, seed) for p in range(1, n_parts + 1)}
