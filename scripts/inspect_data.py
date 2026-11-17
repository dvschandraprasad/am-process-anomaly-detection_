"""Look inside a downloaded part BEFORE running the pipeline.

    python scripts/inspect_data.py data/raw/part1

It prints what the loader needs to know: how many layer files there are, what
they are called, whether they have a header line, how many columns, and the
range of every column. Use it to fill in docs/data_notes.md and to check the
settings in configs/nist_x4.yaml (layer numbers, XCT pore threshold).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load import COLUMNS, list_layer_files, sniff  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="folder or .zip of one part")
    parser.add_argument("--layer", type=int, default=None, help="layer to look at (default: the middle one)")
    args = parser.parse_args()

    files = list_layer_files(args.path)
    layers = [f[0] for f in files]
    print(f"Path:         {args.path}")
    print(f"Layer files:  {len(files)}  (layer numbers {min(layers)} to {max(layers)})")
    print(f"First files:  {[f[1] for f in files[:3]]}")
    print(f"Last file:    {files[-1][1]}")

    pick = args.layer if args.layer is not None else layers[len(layers) // 2]
    layer, name, opener = next(f for f in files if f[0] == pick)
    with opener() as fh:
        raw = fh.read()
    first_line = raw[:20000].split(b"\n", 1)[0].decode("utf-8", errors="replace")
    sep, has_header = sniff(first_line)
    print(f"\nFile looked at: {name}  (layer {layer}, {len(raw) / 1e6:.1f} MB)")
    print(f"Separator:    {sep!r} (None = white space)")
    print(f"Header line:  {'yes' if has_header else 'no'}")
    if has_header:
        print(f"Header text:  {first_line[:300]}")

    import io
    df = pd.read_csv(io.BytesIO(raw), sep=sep if sep else r"\s+", header=None, skiprows=1 if has_header else 0)
    df = df.dropna(axis=1, how="all")
    print(f"Rows:         {len(df):,}")
    print(f"Columns:      {df.shape[1]}  (the loader expects {len(COLUMNS)})")

    names = COLUMNS if df.shape[1] == len(COLUMNS) else [f"col{i + 1}" for i in range(df.shape[1])]
    stats = pd.DataFrame({
        "name": names,
        "min": df.min().to_numpy(), "median": df.median().to_numpy(), "max": df.max().to_numpy(),
        "empty %": (df.isna().mean() * 100).round(2).to_numpy(),
    }, index=range(1, df.shape[1] + 1))
    pd.set_option("display.width", 160, "display.max_rows", 100, "display.float_format", "{:.5g}".format)
    print("\nColumn overview:")
    print(stats)

    if df.shape[1] == len(COLUMNS):
        df.columns = COLUMNS
        dt = np.diff(df["time_us"].to_numpy())
        print(f"\nTime step between rows [us]: median {np.median(dt):.1f}, "
              f"share of gaps (> 1.5 x median): {np.mean(dt > 1.5 * np.median(dt)):.1%}")
        print(f"Commanded laser powers [W]: {sorted(df['cmd_power'].round().unique().tolist())[:12]}")
        q = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
        print("XCT value (3x3x3 mean) at quantiles "
              + ", ".join(f"{p:.1%}: {v:.5g}" for p, v in zip(q, df["xct_m3"].quantile(q))))
        print("  -> if pores are DARK the low end has a long tail; if BRIGHT the high end has. "
              "Set labels.xct.pore_is to match.")
        first = np.r_[True, dt > 2.5 * np.median(dt)]
        print(f"Tracks in this layer: {int(first.sum())}; melt pool area is 0 in "
              f"{np.mean(df['mp_area_100'][first] == 0):.0%} of first frames and "
              f"{np.mean(df['mp_area_100'][~first] == 0):.2%} of other frames")
    else:
        print("\nThe column count differs from the NIST description. "
              "Set `data.columns` in your config to a list of names for these columns.")


if __name__ == "__main__":
    main()
