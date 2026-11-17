"""Look at the XCT values of whole parts: are there pores, and where?

    python scripts/check_xct.py --config configs/nist_x4.yaml

For every part it prints how many points fall below the automatic pore
threshold (median - k robust std), and where those points are:
  * with the old surface rule (`is_edge`: only cells next to an empty cell in
    the same layer are left out), and
  * with the full surface rule (`xct_surface`: also top faces, faces over
    powder and the lowest layers of the XCT scan, see labels.add_xct_surface).
It also prints how close the inner points come to the outer faces of the part,
and saves a histogram of the XCT values to <output_dir>/figures/xct_histogram.png.
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

VOXEL_MM = 0.03   # xct_m3: 3 x 3 x 3 mean of 10 um voxels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data, ov, xc = cfg["data"], cfg["labels"]["overhang"], cfg["labels"]["xct"]
    col, k = xc["column"], xc["k"]

    hist = {}
    split = cfg["split"]
    train_parts = set(split["train"]) if split["mode"] == "by_part" else set(data["parts"])
    train_inner = []
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
                marked = pts["xct_surface"].to_numpy()[low]
                print(f"    of these: {bottom.sum():,} in the lowest 3 XCT layers, {top.sum():,} in the "
                      f"top 2 layers, {(~bottom & ~top & marked).sum():,} elsewhere and marked by the "
                      f"full rule, {(~marked).sum():,} not marked by the full rule "
                      "(the full rule was made after looking at these points, so this is not a "
                      "separate check)")
        # How far inner points stay from the outer faces: per layer and per 0.1 mm row,
        # distance to the first and last point with an XCT value in that row.
        h = pts.loc[has, ["layer", "x", "y", "xct_surface"]]
        gaps = []
        for ax, other in (("x", "y"), ("y", "x")):
            row = [h["layer"], np.floor(h[other] / 0.1)]
            lo, hi = h.groupby(row)[ax].transform("min"), h.groupby(row)[ax].transform("max")
            inner_pts = ~h["xct_surface"]
            gaps += [f"smallest {ax} {(h[ax] - lo)[inner_pts].min():.2f}",
                     f"largest {ax} {(hi - h[ax])[inner_pts].min():.2f}"]
        print("  full rule: closest inner point to the outer faces [mm]: " + ", ".join(gaps))
        # How much of each layer the XCT values cover: each frame reads XCT at one spot,
        # and xct_m3 is a 3 x 3 x 3 mean of 10 um voxels, so about 0.03 x 0.03 mm.
        cells = pts.loc[has, ["layer"]].assign(
            ix=np.floor((pts.loc[has, "x"] - pts["x"].min()) / ov["cell_mm"]).astype(int),
            iy=np.floor((pts.loc[has, "y"] - pts["y"].min()) / ov["cell_mm"]).astype(int))
        area = cells.drop_duplicates().groupby("layer").size() * ov["cell_mm"] ** 2
        seen = cells.groupby("layer").size() * VOXEL_MM ** 2
        cover = (seen / area).median()
        print(f"  share of the layer area where XCT is read (frames x {VOXEL_MM} x {VOXEL_MM} mm "
              f"/ area of {ov['cell_mm']} mm cells with XCT), middle value over layers: {cover:.1%}")
        hist[part] = (v[has & ~pts["is_edge"].to_numpy()], v[has & ~pts["xct_surface"].to_numpy()])
        if part in train_parts:
            train_inner.append(v[has & ~pts["xct_surface"].to_numpy()])

    # the threshold the pipeline uses: from the inner points of the training parts
    pipe_thr = xct_threshold(np.concatenate(train_inner), xc["pore_is"], k) if train_inner else None
    if pipe_thr is not None:
        print(f"\nThreshold the pipeline uses (inner points of parts {sorted(train_parts)}): {pipe_thr:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(hist), figsize=(3.6 * len(hist), 3.2), sharey=True, squeeze=False)
    bins = np.linspace(0.15, 0.70, 111)
    for ax, (part, (old, new)) in zip(axes[0], hist.items()):
        ax.hist(old, bins=bins, color="#999999", label="old rule: side cells left out")
        ax.hist(new, bins=bins, color="#1f77b4", label="full rule: all surfaces left out")
        if pipe_thr is not None:
            ax.axvline(pipe_thr, color="#d62728", lw=1, ls="--",
                       label=f"pipeline threshold ({pipe_thr:.3f})")
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
