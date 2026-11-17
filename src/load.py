"""Read the registered NIST "Overhang Part X4" data into one table per part.

Layout of the data (NIST AMS 100-69, dataset mds2-3761):
  * one folder per part, one CSV file per layer (250 layers)
  * 40 columns per file, only numbers
  * one row = one melt pool camera frame (10 000 frames per second),
    taken at a known laser position and time

The description does not say what the files are called or whether they have a
header line, so this loader works both ways:
  * the layer number is the last whole number in the file name
  * a header line is detected and skipped
  * columns are named by POSITION (several NIST labels repeat)
"""
from __future__ import annotations

import gzip
import io
import re
import zipfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

# Column names by position, following Tables 1-3 of NIST AMS 100-69.
COLUMNS: list[str] = [
    "part",            # 1  part number (1-4)
    "time_us",         # 2  build time [microseconds]
    "x",               # 3  commanded laser position X [mm]
    "y",               # 4  commanded laser position Y [mm]
    "cmd_power",       # 5  commanded laser power [W]
    "cmd_speed",       # 6  commanded scan speed [mm/s]
    "x_real",          # 7  measured laser position X [mm]
    "y_real",          # 8  measured laser position Y [mm]
    "power",           # 9  measured laser power [W]
    "speed",           # 10 measured scan speed [mm/s]
    "mp_length_80",    # 11 melt pool length, image threshold 80 [mm]
    "mp_width_80",     # 12 melt pool width, threshold 80 [mm]
    "mp_area_80",      # 13 melt pool area, threshold 80 [mm^2]
    "mp_length_100",   # 14
    "mp_width_100",    # 15
    "mp_area_100",     # 16
    "mp_length_120",   # 17
    "mp_width_120",    # 18
    "mp_area_120",     # 19
    # layer camera ("LWI"), powder view after spreading, LED A/B/C,
    # raw value / 3x3 mean / 5x5 mean
    "lwi_powder_a", "lwi_powder_a_m3", "lwi_powder_a_m5",   # 20-22
    "lwi_powder_b", "lwi_powder_b_m3", "lwi_powder_b_m5",   # 23-25
    "lwi_powder_c", "lwi_powder_c_m3", "lwi_powder_c_m5",   # 26-28
    # layer camera, view after the laser scan
    "lwi_scan_a", "lwi_scan_a_m3", "lwi_scan_a_m5",         # 29-31
    "lwi_scan_b", "lwi_scan_b_m3", "lwi_scan_b_m5",         # 32-34
    "lwi_scan_c", "lwi_scan_c_m3", "lwi_scan_c_m5",         # 35-37
    "xct",             # 38 X-ray CT voxel value at this position
    "xct_m3",          # 39 XCT after a 3x3x3 mean filter
    "xct_m5",          # 40 XCT after a 5x5x5 mean filter
]

_CSV_SUFFIXES = (".csv", ".csv.gz", ".txt", ".txt.gz")


def _stem(name: str) -> str:
    stem = Path(name).name
    for suffix in _CSV_SUFFIXES:
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def layer_from_name(name: str) -> int | None:
    """Return the last whole number in a file name, e.g. 'Layer0125.csv' -> 125."""
    numbers = re.findall(r"\d+", _stem(name))
    return int(numbers[-1]) if numbers else None


def layers_from_names(names: list[str]) -> list[int]:
    """Work out the layer number of every file from the file names.

    Uses the last number in the name that is different for every file
    ('Part1_Layer007.csv' and 'Layer007_Part1.csv' both give 7). If the names
    hold no such number, the files are numbered 1, 2, 3 ... in natural order.
    """
    groups = [[int(g) for g in re.findall(r"\d+", _stem(n))] for n in names]
    n_groups = {len(g) for g in groups}
    if len(n_groups) == 1 and 0 not in n_groups:
        for pos in reversed(range(n_groups.pop())):
            values = [g[pos] for g in groups]
            if len(set(values)) == len(values):
                return values

    def natural(name: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

    order = sorted(range(len(names)), key=lambda i: natural(names[i]))
    layers = [0] * len(names)
    for rank, i in enumerate(order, start=1):
        layers[i] = rank
    return layers


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def sniff(first_line: str) -> tuple[str | None, bool]:
    """Look at the first line of a file. Return (separator, has_header)."""
    line = first_line.strip()
    if "," in line:
        sep: str | None = ","
    elif ";" in line:
        sep = ";"
    elif "\t" in line:
        sep = "\t"
    else:
        sep = None  # any white space
    tokens = [t.strip() for t in (line.split(sep) if sep else line.split())]
    has_header = any(t and not _is_number(t) for t in tokens)
    return sep, has_header


def list_layer_files(path: str | Path) -> list[tuple[int, str, Callable[[], io.BufferedIOBase]]]:
    """Find the layer files in a folder or a .zip file.

    Returns a list of (layer number, file name, function that opens the file),
    sorted by layer number.
    """
    path = Path(path)
    found: list[tuple[str, Callable[[], io.BufferedIOBase]]] = []

    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.is_file() and file.name.lower().endswith(_CSV_SUFFIXES):
                opener = (lambda f=file: gzip.open(f, "rb")) if file.suffix == ".gz" \
                    else (lambda f=file: open(f, "rb"))
                found.append((str(file.relative_to(path)), opener))
    elif path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist() if n.lower().endswith(_CSV_SUFFIXES))

        def zip_opener(name: str):
            def _open():
                data = zipfile.ZipFile(path).read(name)
                if name.lower().endswith(".gz"):
                    data = gzip.decompress(data)
                return io.BytesIO(data)
            return _open

        found = [(name, zip_opener(name)) for name in names]
    else:
        raise FileNotFoundError(f"{path} is not a folder or a .zip file")

    if not found:
        raise FileNotFoundError(f"No CSV files found in {path}")

    layers = layers_from_names([name for name, _ in found])
    out = [(int(l), name, opener) for l, (name, opener) in zip(layers, found)]
    out.sort(key=lambda item: item[0])
    return out


def read_layer_file(opener: Callable[[], io.BufferedIOBase], columns: list[str]) -> pd.DataFrame:
    """Read one layer file into a table with the given column names."""
    with opener() as fh:
        raw = fh.read()
    first_line = raw[:20000].split(b"\n", 1)[0].decode("utf-8", errors="replace")
    sep, has_header = sniff(first_line)
    df = pd.read_csv(
        io.BytesIO(raw),
        sep=sep if sep else r"\s+",
        header=None,
        skiprows=1 if has_header else 0,
    )
    # A trailing separator makes an extra, empty column at the end: drop it.
    # Do NOT drop empty columns in general. In the NIST files some layers
    # really have no value in some columns (NaN): no XCT in the lowest
    # layers, no layer camera values in layer 1.
    while df.shape[1] > len(columns) and df.iloc[:, -1].isna().all():
        df = df.iloc[:, :-1]
    if df.shape[1] != len(columns):
        raise ValueError(
            f"Expected {len(columns)} columns but the file has {df.shape[1]}. "
            "Run `python scripts/inspect_data.py <part folder>` to see what is in "
            "the files, then set `data.columns` in your config."
        )
    df.columns = columns
    return df


def load_part(
    path: str | Path,
    part: int | None = None,
    layers: Iterable[int] | None = None,
    columns: list[str] | None = None,
    layer_thickness_mm: float = 0.02,
) -> pd.DataFrame:
    """Load one part: all its layer files stacked into one table.

    Adds the columns `layer` (from the file name) and `z` (layer height in mm).
    `layers` can be (first, last) to load only a range of layers.
    Rows are sorted by layer and time.
    """
    columns = list(columns) if columns else COLUMNS
    files = list_layer_files(path)
    if layers is not None:
        first, last = min(layers), max(layers)
        files = [f for f in files if first <= f[0] <= last]
        if not files:
            raise ValueError(f"No layer files between {first} and {last} in {path}")

    tables = []
    for layer, _name, opener in files:
        df = read_layer_file(opener, columns)
        df.insert(0, "layer", np.int32(layer))
        tables.append(df)
    points = pd.concat(tables, ignore_index=True)

    if part is not None:
        points["part"] = int(part)
    points["part"] = points["part"].round().astype(np.int32)
    points["z"] = (points["layer"] * layer_thickness_mm).astype(np.float32)

    for col in points.columns:
        if col not in ("part", "layer", "time_us"):
            points[col] = points[col].astype(np.float32)
    points["time_us"] = points["time_us"].astype(np.float64)

    points = points.sort_values(["layer", "time_us"], kind="stable").reset_index(drop=True)
    return points


def load_part_cached(
    path: str | Path,
    part: int,
    cache_dir: str | Path | None,
    layers: Iterable[int] | None = None,
    columns: list[str] | None = None,
    layer_thickness_mm: float = 0.02,
) -> pd.DataFrame:
    """Like `load_part`, but keeps a Parquet copy so the next run is fast."""
    if cache_dir is None:
        return load_part(path, part, layers, columns, layer_thickness_mm)
    cache_dir = Path(cache_dir)
    tag = "all" if layers is None else f"{min(layers)}-{max(layers)}"
    cache_file = cache_dir / f"part{part}_layers-{tag}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    points = load_part(path, part, layers, columns, layer_thickness_mm)
    cache_dir.mkdir(parents=True, exist_ok=True)
    points.to_parquet(cache_file, index=False)
    return points
