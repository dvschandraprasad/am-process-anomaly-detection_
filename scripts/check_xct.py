"""Look at the XCT values of whole parts: are there pores, and where?

    python scripts/check_xct.py --config configs/nist_x4.yaml

For every part it prints how many points fall below the automatic pore
threshold (median - k robust std), and where those points are:
  * with the old surface rule (`is_edge`: only cells next to an empty cell in
    the same layer are left out), and
  * with the full surface rule (`xct_surface`: also top faces, faces over
    powder and the lowest layers of the XCT scan, see labels.add_xct_surface).
It also saves a histogram of the XCT values to <output_dir>/figures/xct_histogram.png.
Numbers from this script are in docs/data_notes.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.labels import add_geometry, add_xct_surface, xct_threshold  # noqa: E402
from src.load import load_part_cached  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data, ov, xc = cfg["data"], cfg["labels"]["overhang"], cfg["labels"]["xct"]
    col, k = xc["column"], xc["k"]

    hist = {}
    for part, path in sorted(data["parts"].items()):
        pts = load_part_cached(path, part, data["cache_dir"], data["layers"], data["columns"],
                               data["layer_thickness_mm"])
        pts = add_geometry(pts, data["layer_thickness_mm"], ov["cell_mm"], ov["cap_mm"])
        pts = add_xct_surface(pts, col, data["layer_thickness_mm"], ov["cell_mm"],
                              xc["surface_xy_mm"], xc["surface_z_mm"])
        v = pts[col].to_numpy(dtype=np.float64)
        has = np.isfinite(v)
        first = int(pts.loc[has, "layer"].min())
        print(f"\npart {part}: {len(pts):,} points, {has.sum():,} with an XCT value "
              f"(layers {first} to {int(pts.loc[has, 'layer'].max())})")

        for name, surface in (("old rule (is_edge)", pts["is_edge"].to_numpy()),
                              ("full rule (xct_surface)", pts["xct_surface"].to_numpy())):
            inner = has & ~surface
            thr = xct_threshold(v[inner], xc["pore_is"], k)
            low = inner & ((v < thr) if xc["pore_is"] == "low" else (v > thr))
            q = np.quantile(v[inner], [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0])
            print(f"  {name}: {inner.sum():,} inner points, median {q[3]:.4f}, "
                  f"threshold {thr:.4f}, below it: {low.sum():,} points")
            print("    quantiles 0 / 0.1 % / 1 % / 50 % / 99 % / 99.9 % / 100 %: "
                  + " / ".join(f"{x:.4f}" for x in q))
            if low.any():
                layer = pts["layer"].to_numpy()[low]
                bottom = layer <= first + 2
                top = layer >= pts["layer"].max() - 1
                print(f"    of these: {bottom.sum():,} in the lowest 3 XCT layers, {top.sum():,} in the "
                      f"top 2 layers, {(~bottom & ~top & pts['xct_surface'].to_numpy()[low]).sum():,} "
                      f"elsewhere at a surface (full rule), "
                      f"{(~pts['xct_surface'].to_numpy()[low]).sum():,} away from any surface")
        hist[part] = (v[has & ~pts["is_edge"].to_numpy()], v[has & ~pts["xct_surface"].to_numpy()])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(hist), figsize=(3.6 * len(hist), 3.2), sharey=True, squeeze=False)
    bins = np.linspace(0.15, 0.70, 111)
    for ax, (part, (old, new)) in zip(axes[0], hist.items()):
        ax.hist(old, bins=bins, color="#999999", label="old rule: side cells left out")
        ax.hist(new, bins=bins, color="#1f77b4", label="full rule: all surfaces left out")
        thr = xct_threshold(new, xc["pore_is"], k)
        ax.axvline(thr, color="#d62728", lw=1, ls="--", label=f"auto threshold, full rule ({thr:.3f})")
        ax.set_yscale("log")
        ax.set_title(f"Part {part}")
        ax.set_xlabel(f"XCT value ({col})")
    axes[0][0].set_ylabel("points (log scale)")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle("XCT values inside the part. Low values = dark = air.", fontsize=10)
    fig.tight_layout()
    out = Path(cfg["output_dir"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "xct_histogram.png", dpi=130)
    print(f"\nHistogram saved to {out / 'xct_histogram.png'}")


if __name__ == "__main__":
    main()
