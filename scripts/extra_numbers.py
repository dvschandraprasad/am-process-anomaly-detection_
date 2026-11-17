"""Numbers in the docs that the pipeline itself does not write.

    python scripts/extra_numbers.py --config configs/nist_x4.yaml

Run it after `run_pipeline.py` with the same config. It writes
<output_dir>/extra_numbers.md with two parts:
  * data:    rows, tracks, missing XCT layers, melt pool size per part, in
             layer 1 and over the overhang (from the loaded parts)
  * results: alarms on the clean test part, alarms in layer 1, and the rank
             correlation with XCT inside each layer (from scores_test.parquet
             and traceability.csv of the run)
docs/data_notes.md and docs/report.md point to this file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.features import assign_tracks  # noqa: E402
from src.labels import add_geometry  # noqa: E402
from src.load import load_part_cached  # noqa: E402
from src.models import LABELS  # noqa: E402

INFILL_W, CONTOUR_W = 195.0, 100.0


def _median(values: pd.Series) -> float:
    return float(values.median()) if len(values) else float("nan")


def data_numbers(cfg: dict) -> list[str]:
    """Numbers about the loaded parts (one part in memory at a time)."""
    data, ov, sg = cfg["data"], cfg["labels"]["overhang"], cfg["segments"]
    area = [s for s in sg["signals"] if s.startswith("mp_area")]
    area = area[0] if area else sg["signals"][-1]
    xct = cfg["labels"]["xct"]["column"]
    min_track = sg["window"] + sg["drop_first"]
    rows = []
    for part, path in sorted(data["parts"].items()):
        p = load_part_cached(path, part, data["cache_dir"], data["layers"], data["columns"],
                             data["layer_thickness_mm"])
        p = add_geometry(p, data["layer_thickness_mm"], ov["cell_mm"], ov["cap_mm"])
        track = assign_tracks(p)
        track_len = np.bincount(track)
        per_layer = p.groupby("layer").size()
        has_xct = p.groupby("layer")[xct].apply(lambda v: np.isfinite(v).any()) if xct in p \
            else pd.Series(False, index=per_layer.index)
        no_xct = per_layer.index[~has_xct.to_numpy()]

        melt = p[area] > 0
        infill = melt & np.isclose(p["cmd_power"], INFILL_W)
        contour = melt & np.isclose(p["cmd_power"], CONTOUR_W)
        overhang = p["solid_below"] < ov["max_solid_mm"]
        powder = p["solid_below"] < 0.1
        upper = p["layer"] > 50
        rows.append({
            "part": part,
            "rows": len(p),
            "rows per layer (min / median / max)":
                f"{per_layer.min()} / {per_layer.median():.0f} / {per_layer.max()}",
            "layers without XCT": (f"{no_xct.min()}-{no_xct.max()} ({len(no_xct)})" if len(no_xct) else "none"),
            "tracks": len(track_len),
            f"frames in tracks < {min_track} frames": f"{track_len[track_len < min_track].sum() / len(p):.1%}",
            f"infill area, median [mm²]": f"{_median(p.loc[infill, area]):.4f}",
            f"contour area, median [mm²]": f"{_median(p.loc[contour, area]):.4f}",
            "infill area, layer 1": f"{_median(p.loc[infill & (p['layer'] == 1), area]):.4f}",
            "infill area, layers > 50": f"{_median(p.loc[infill & upper, area]):.4f}",
            f"frames in overhang region (< {ov['max_solid_mm']} mm metal below)": f"{overhang.mean():.1%}",
            "overhang layers": (f"{p.loc[overhang, 'layer'].min()}-{p.loc[overhang, 'layer'].max()}"
                                if overhang.any() else "none"),
            "infill area over overhang / elsewhere": (
                f"{_median(p.loc[infill & overhang, area]) / _median(p.loc[infill & ~overhang, area]) - 1:+.1%}"),
            "infill area, < 0.1 mm metal below": f"{_median(p.loc[infill & powder, area]):.4f}",
        })
        del p
    table = pd.DataFrame(rows).set_index("part").T
    lines = ["## Data (loaded parts)", "",
             f"Melt pool area = `{area}`; only frames with a melt pool (area > 0). "
             f"Infill = commanded {INFILL_W:.0f} W, contour = {CONTOUR_W:.0f} W. Middle values (medians).", "",
             "| | " + " | ".join(f"Part {c}" for c in table.columns) + " |",
             "|---|" + "---:|" * len(table.columns)]
    lines += [f"| {name} | " + " | ".join(str(v) for v in row) + " |" for name, row in table.iterrows()]
    return lines + [""]


def result_numbers(out_dir: Path) -> list[str]:
    """Numbers about the alarms on the clean test part (first seed of each method)."""
    s = pd.read_parquet(out_dir / "scores_test.parquet")
    trace = pd.read_csv(out_dir / "traceability.csv")
    methods = [c[len("flag_"):] for c in s.columns if c.startswith("flag_")]
    tuned = [m for m in methods if m not in ("limits_3sigma", "limits_levels")]
    lines = ["## Alarms on the clean test part (no injected faults)", "",
             "Methods with randomness: first seed only (the scores in `scores_test.parquet`).", ""]

    flags = s[[f"flag_{m}" for m in tuned]].to_numpy()
    n_methods = flags.sum(axis=1)
    hit = n_methods > 0
    counts = ", ".join(f"by exactly {k}: {int((n_methods == k).sum())}" for k in range(1, len(tuned) + 1))
    lines += [f"- Segments: {len(s)}. Flagged by at least one tuned method: {int(hit.sum())} "
              f"(= rows in traceability.csv: {len(trace)}).",
              f"- {counts}."]

    first = s["layer"] == 1
    if first.any():
        lines += [f"- Layer 1: {int(first.sum())} segments ({first.mean():.2%} of all), "
                  f"{int((hit & first).sum())} of the {int(hit.sum())} flagged ones.", ""]
    # share of segments flagged, by height in the part
    bands = {"layer 1": s["layer"] == 1, "layers 2-20": s["layer"].between(2, 20),
             "layers 21 and up": s["layer"] >= 21}
    bands = {k: v for k, v in bands.items() if v.any()}
    lines += ["Share of segments flagged, by height in the part "
              "(" + ", ".join(f"{k}: {int(v.sum())} segments" for k, v in bands.items()) + "):", "",
              "| Method | " + " | ".join(bands) + " |", "|---|" + "---:|" * len(bands)]
    for m in methods:
        f = s[f"flag_{m}"]
        lines.append(f"| {LABELS.get(m, m)} | " + " | ".join(f"{f[v].mean():.1%}" for v in bands.values()) + " |")
    lines.append("")
    # layer 1 and the layer with the most alarms above it (what the layer heatmaps show)
    above = s[~first]
    lines += ["Flagged segments in layer 1, and in the layer with the most alarms above layer 1:", "",
              "| Method | Layer 1 | Layer with the most alarms above layer 1 |", "|---|---:|---:|"]
    for m in methods:
        in_first = (f"{int(s.loc[first, f'flag_{m}'].sum())} of {int(first.sum())}" if first.any() else "-")
        if len(above):
            per = above.groupby("layer")[f"flag_{m}"].agg(["sum", "size"])
            top = per["sum"].idxmax()
            in_top = f"layer {int(top)}: {int(per.loc[top, 'sum'])} of {int(per.loc[top, 'size'])}"
        else:
            in_top = "-"
        lines.append(f"| {LABELS.get(m, m)} | {in_first} | {in_top} |")
    lines.append("")

    if "xct_valid" in s and s["xct_valid"].sum() > 10:
        v = s[s["xct_valid"]].copy()
        lines += ["## Rank correlation with XCT, overall and inside each layer", "",
                  f"{len(v)} segments away from the surface. \"Inside each layer\": score and XCT "
                  "value are first turned into ranks within their layer, then correlated. "
                  "This removes slow changes up the part.", "",
                  "| Method | Overall | Inside each layer |", "|---|---:|---:|"]
        xr = v.groupby("layer")["xct_min"].rank(pct=True)
        for m in tuned:
            overall = spearmanr(v[f"score_{m}"], v["xct_min"]).statistic
            sr = v.groupby("layer")[f"score_{m}"].rank(pct=True)
            within = spearmanr(sr, xr).statistic
            lines.append(f"| {LABELS.get(m, m)} | {overall:+.3f} | {within:+.3f} |")
        lines.append("")

    if len(trace):
        top = trace[trace["n_methods"] == trace["n_methods"].max()]
        layers = list(dict.fromkeys(top["layer"].astype(int)))
        lines += ["## Top of traceability.csv", "",
                  f"- {len(top)} segments are flagged by all {int(trace['n_methods'].max())} tuned methods, "
                  f"in {len(layers)} layers. First layers in the list: "
                  + ", ".join(str(x) for x in layers[:8]) + ".",
                  f"- Of these, over the overhang region: {int(top['is_overhang'].sum())}." if "is_overhang" in top
                  else "", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    if not (out_dir / "scores_test.parquet").exists():
        sys.exit(f"{out_dir / 'scores_test.parquet'} not found. Run run_pipeline.py with this config first.")
    lines = ["# Extra numbers", "", f"Written by `scripts/extra_numbers.py --config {args.config}`.", ""]
    lines += data_numbers(cfg) + result_numbers(out_dir)
    text = "\n".join(lines)
    (out_dir / "extra_numbers.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWritten to {out_dir / 'extra_numbers.md'}")


if __name__ == "__main__":
    main()
