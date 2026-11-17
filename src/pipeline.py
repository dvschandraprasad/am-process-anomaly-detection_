"""The whole experiment, from raw files to tables and figures.

Steps:
  1. load the parts and work out the geometry (metal below, surface)
  2. split into train / validation / test (by part, never randomly)
  3. cut into segments, compute features and labels
  4. make a copy of the test data with injected faults
  5. for every method: learn "normal" on train, set the alarm level on
     validation, score the test data
  6. write tables, figures and the traceability table
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import plots
from .evaluate import (detection_metrics, event_detection, pr_curve, ranking_metrics,
                       summarise, threshold_for_frr, to_markdown)
from .features import Segments, make_segments
from .inject import inject_faults
from .labels import add_geometry, add_xct_surface, label_segments, xct_threshold
from .load import load_part_cached
from .models import HAS_TORCH, LABELS, RANDOM_METHODS, build_detector

SPLIT_PATTERN = ("train", "train", "train", "val", "test")


def log(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


# ------------------------------------------------------------------- data --

def prepare_points(cfg: dict) -> dict[int, pd.DataFrame]:
    """Load every part named in the config and add the geometry columns."""
    data, lab, xc = cfg["data"], cfg["labels"]["overhang"], cfg["labels"]["xct"]
    parts = {}
    for part, path in sorted(data["parts"].items()):
        points = load_part_cached(path, part, data["cache_dir"], data["layers"],
                                  data["columns"], data["layer_thickness_mm"])
        points = add_geometry(points, data["layer_thickness_mm"], lab["cell_mm"], lab["cap_mm"])
        if xc["enabled"] and xc["ignore_edge"] and xc["column"] in points:
            points = add_xct_surface(points, xc["column"], data["layer_thickness_mm"], lab["cell_mm"],
                                     xc["surface_xy_mm"], xc["surface_z_mm"])
        parts[part] = points
        log(f"part {part}: {len(points):,} points, layers {points['layer'].min()}-{points['layer'].max()}")
    return parts


def split_points(parts: dict[int, pd.DataFrame], cfg: dict) -> dict[str, pd.DataFrame]:
    """Split into train / val / test point tables.

    by_part: whole parts go to one set. Nothing from a test part is ever seen
        in training.
    by_layer_blocks: for when you only have one part. Blocks of layers go
        to train, train, train, val, test, and so on up the part. A layer is
        never shared between sets, but neighbouring layers are alike, so this
        split is easier than by_part. Say so when you report results.
    """
    sp = cfg["split"]
    sets: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    if sp["mode"] == "by_part":
        for role in sets:
            for part in sp[role]:
                if part not in parts:
                    raise KeyError(f"split.{role} names part {part}, but data.parts does not have it")
                sets[role].append(parts[part])
    elif sp["mode"] == "by_layer_blocks":
        for part in sorted(parts):
            points = parts[part]
            block = (points["layer"] - points["layer"].min()) // sp["block_layers"]
            role = np.array(SPLIT_PATTERN)[(block % len(SPLIT_PATTERN)).to_numpy()]
            for name in sets:
                sets[name].append(points[role == name])
    else:
        raise ValueError(f"Unknown split.mode: {sp['mode']}")
    out = {}
    for role, tables in sets.items():
        if not tables or sum(len(t) for t in tables) == 0:
            raise ValueError(f"The {role} set is empty. Check the `split` settings.")
        out[role] = pd.concat(tables, ignore_index=True)
    return out


def segment(points: pd.DataFrame, cfg: dict, xct_thr: float | None) -> Segments:
    sg, lab = cfg["segments"], cfg["labels"]
    seg = make_segments(points, sg["signals"], sg["context"], sg["window"], sg["drop_first"],
                        lab["xct"]["column"], sg["geometry_context"])
    seg.df = label_segments(seg.df, lab["overhang"]["max_solid_mm"], xct_thr, lab["xct"]["pore_is"],
                            lab["xct"]["ignore_edge"], cfg["inject"]["min_fault_points"])
    return seg


def _inputs(seg: Segments, kind: str) -> np.ndarray:
    if kind == "windows":
        if seg.context_windows is None:
            return seg.windows
        return np.concatenate([seg.windows, seg.context_windows], axis=2)
    return seg.df[seg.feature_cols].to_numpy(dtype=np.float64)


def _groups(seg: Segments, per_region: bool) -> np.ndarray:
    """Group number for the fixed limits.

    Limits are set per machine setting and for track starts separately, and
    per region (overhang or not) if wanted.
    """
    group = seg.df["regime"].to_numpy().astype(np.int64) * 2 + seg.df["track_start"].to_numpy().astype(np.int64)
    if per_region:
        group = group * 2 + seg.df["is_overhang"].to_numpy().astype(np.int64)
    return group


# -------------------------------------------------------------------- run --

def run(cfg: dict, make_figures: bool = True) -> dict:
    """Run the experiment described by `cfg`. Returns the main result tables."""
    out_dir = Path(cfg["output_dir"])
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    sg, xc = cfg["segments"], cfg["labels"]["xct"]

    parts = prepare_points(cfg)
    sets = split_points(parts, cfg)

    # XCT alarm level, from training data only
    xct_thr = None
    if xc["enabled"] and xc["column"] in sets["train"].columns:
        if xc["threshold"] == "auto":
            surface = "xct_surface" if "xct_surface" in sets["train"] else "is_edge"
            inner = sets["train"].loc[~sets["train"][surface], xc["column"]].to_numpy()
            xct_thr = xct_threshold(inner, xc["pore_is"], xc["k"])
        else:
            xct_thr = float(xc["threshold"])
        log(f"XCT pore threshold ({xc['column']}, pore is {xc['pore_is']}): {xct_thr:.5g}")

    seg_train = segment(sets["train"], cfg, xct_thr)
    seg_val = segment(sets["val"], cfg, xct_thr)
    seg_test = segment(sets["test"], cfg, xct_thr)
    inj = cfg["inject"]
    inj_points, events = inject_faults(sets["test"], sg["signals"], inj["events_per_1000_points"],
                                       inj["severity"], inj["types"], inj["seed"])
    seg_inj = segment(inj_points, cfg, xct_thr)
    seg_inj.df = seg_inj.df.merge(events[["fault_id", "fault_type"]], on="fault_id", how="left")

    train_ok = ~seg_train.df["xct_defect"].to_numpy()   # learn from normal segments only
    val_ok = ~seg_val.df["xct_defect"].to_numpy()
    n_test_layers = seg_test.df.groupby("part")["layer"].nunique().sum()
    log(f"segments: train {len(seg_train):,} ({train_ok.sum():,} normal), val {len(seg_val):,}, "
        f"test {len(seg_test):,}; injected faults: {len(events)} "
        f"({int(seg_inj.df['is_injected'].sum())} faulty segments)")

    # Limits only look at simple level features of the melt pool signals.
    limit_cols = [i for i, c in enumerate(seg_train.feature_cols)
                  if c.endswith(("_mean", "_min", "_max")) and c.rsplit("_", 1)[0] in sg["signals"]]

    methods = list(cfg["methods"])
    if "autoencoder" in methods and not HAS_TORCH:
        log("PyTorch is not installed: skipping the autoencoder")
        methods.remove("autoencoder")
    if "limits" in methods:
        methods.insert(0, "limits_3sigma")

    inj_df, test_df = seg_inj.df, seg_test.df
    per_region = bool(sg["geometry_context"])
    inj_mask = ~inj_df["xct_defect"].to_numpy()          # real pores are not "normal"
    y_inj = inj_df["is_injected"].to_numpy()
    xct_mask = test_df["xct_valid"].to_numpy()
    y_xct = test_df["xct_defect"].to_numpy()
    overhang_inj = inj_df["is_overhang"].to_numpy()

    # XCT value on the "pore side": low values if pores are dark
    xct_side = None
    if xct_mask.sum() > 10 and "xct_min" in test_df:
        xct_side = (test_df["xct_min"] if xc["pore_is"] == "low" else -test_df["xct_max"]).to_numpy()

    rows_inj, rows_xct, rows_fault, rows_rank = [], [], [], []
    scores_test = test_df.copy()
    scores_inj = inj_df.copy()
    curves, thresholds, centers = {}, {}, {}

    for method in methods:
        base = method.replace("_3sigma", "")
        seeds = cfg["seeds"] if base in RANDOM_METHODS else cfg["seeds"][:1]
        for seed in seeds:
            params = dict(cfg["method_params"].get(base) or {})
            if base == "autoencoder" and seg_train.context_windows is not None:
                params["n_context"] = seg_train.context_windows.shape[2]
            det = build_detector(method, seed, params,
                                 limit_columns=limit_cols if base == "limits" else None)
            kind = det.input_kind
            t0 = time.perf_counter()
            det.fit(_inputs(seg_train, kind)[train_ok], _groups(seg_train, per_region)[train_ok])
            fit_s = time.perf_counter() - t0

            s_val = det.score(_inputs(seg_val, kind), _groups(seg_val, per_region))
            t0 = time.perf_counter()
            s_test = det.score(_inputs(seg_test, kind), _groups(seg_test, per_region))
            score_ms = 1000.0 * (time.perf_counter() - t0) / max(n_test_layers, 1)
            s_inj = det.score(_inputs(seg_inj, kind), _groups(seg_inj, per_region))

            thr = det.fixed_threshold if det.fixed_threshold is not None \
                else threshold_for_frr(s_val[val_ok], cfg["threshold"]["target_frr"])

            # --- injected faults
            flagged = s_inj > thr
            normal = inj_mask & ~y_inj
            row = {"method": method, "seed": seed, "threshold": thr,
                   **detection_metrics(y_inj[inj_mask], s_inj[inj_mask], thr),
                   **ranking_metrics(y_inj[inj_mask], s_inj[inj_mask]),
                   "frr_bulk": float(np.mean(flagged[normal & ~overhang_inj])) if (normal & ~overhang_inj).any() else np.nan,
                   "frr_overhang": float(np.mean(flagged[normal & overhang_inj])) if (normal & overhang_inj).any() else np.nan,
                   "fit_s": fit_s, "score_ms_per_layer": score_ms}
            rows_inj.append(row)
            if len(events):
                ft = event_detection(inj_df, flagged, events)
                rows_fault.append(ft.assign(method=method, seed=seed))

            # --- real pores seen by XCT (no injected faults here)
            if xct_mask.any() and y_xct[xct_mask].any():
                rows_xct.append({"method": method, "seed": seed,
                                 **detection_metrics(y_xct[xct_mask], s_test[xct_mask], thr),
                                 **ranking_metrics(y_xct[xct_mask], s_test[xct_mask])})

            if xct_side is not None:
                rows_rank.append({"method": method, "seed": seed,
                                  "xct_rank_corr": float(spearmanr(s_test[xct_mask], xct_side[xct_mask]).statistic)})

            if seed == seeds[0]:   # keep the scores of the first seed for figures
                thresholds[method] = float(thr)
                centers[method] = float(np.median(s_val[val_ok]))
                scores_test[f"score_{method}"] = s_test
                scores_test[f"flag_{method}"] = s_test > thr
                scores_inj[f"score_{method}"] = s_inj
                scores_inj[f"flag_{method}"] = flagged
                if y_inj[inj_mask].any():
                    curves[method] = pr_curve(y_inj[inj_mask], s_inj[inj_mask])
        log(f"{method}: done")

    # ---------------------------------------------------------------- tables
    metrics_inj = pd.DataFrame(rows_inj)
    m_cols = ["precision", "recall", "f1", "frr", "frr_bulk", "frr_overhang", "pr_auc", "roc_auc",
              "frr_at_90_recall", "fit_s", "score_ms_per_layer"]
    summary_inj = summarise(metrics_inj, ["method"], m_cols)
    metrics_inj.to_csv(out_dir / "metrics_injected.csv", index=False)
    summary_inj.to_csv(out_dir / "summary_injected.csv", index=False)

    md = ["# Results", "",
          f"Alarm level of every method: set on validation data so that "
          f"{cfg['threshold']['target_frr']:.1%} of normal segments are flagged "
          "(except the 3-sigma rule, which is fixed).", "",
          "## Injected faults (test set)", "",
          to_markdown(summary_inj,
                      {"precision": "Precision", "recall": "Recall", "f1": "F1",
                       "frr": "False rejection rate", "pr_auc": "PR-AUC",
                       "frr_at_90_recall": "FRR at 90 % recall",
                       "score_ms_per_layer": "Run time [ms/layer]"},
                      labels=LABELS, percent=("frr", "frr_at_90_recall")), ""]

    fault_types = pd.DataFrame()
    if rows_fault:
        fault_types = pd.concat(rows_fault, ignore_index=True)
        fault_types = fault_types.groupby(["method", "fault_type"], sort=False).agg(
            detection_rate=("detection_rate", "mean"), n_faults=("n_faults", "first")).reset_index()
        fault_types.to_csv(out_dir / "fault_types.csv", index=False)
        wide = fault_types.pivot_table(index="method", columns="fault_type", values="detection_rate", sort=False)
        md += ["## Share of injected faults caught, by kind", "",
               "| Method | " + " | ".join(wide.columns) + " |", "|---|" + "---:|" * len(wide.columns)]
        md += [f"| {LABELS.get(m, m)} | " + " | ".join(f"{v:.0%}" for v in r) + " |" for m, r in
               zip(wide.index, wide.to_numpy())]
        md.append("")

    md += ["## False rejections by region", "",
           to_markdown(summary_inj, {"frr_bulk": "Solid metal below", "frr_overhang": "Overhang region"},
                       labels=LABELS, percent=("frr_bulk", "frr_overhang")), ""]

    summary_xct = pd.DataFrame()
    if rows_xct:
        metrics_xct = pd.DataFrame(rows_xct)
        summary_xct = summarise(metrics_xct, ["method"],
                                ["precision", "recall", "f1", "frr", "pr_auc", "roc_auc"])
        metrics_xct.to_csv(out_dir / "metrics_xct.csv", index=False)
        summary_xct.to_csv(out_dir / "summary_xct.csv", index=False)
        n_pos, n_all = int(y_xct[xct_mask].sum()), int(xct_mask.sum())
        md += [f"## Pores seen by X-ray CT (test set, no injected faults)", "",
               f"{n_pos} of {n_all} segments contain a pore ({n_pos / n_all:.2%}). "
               "A method that guesses at random would get a PR-AUC of about that share.", "",
               to_markdown(summary_xct, {"precision": "Precision", "recall": "Recall",
                                         "frr": "False rejection rate", "pr_auc": "PR-AUC",
                                         "roc_auc": "ROC-AUC"},
                           labels=LABELS, percent=("frr",)), ""]
    else:
        md += ["## Pores seen by X-ray CT", "",
               "No segment in the test set is beyond the XCT pore threshold, so precision and recall "
               "against XCT cannot be given.", ""]

    xct_rank = pd.DataFrame()
    if rows_rank:
        rank_rows = pd.DataFrame(rows_rank)
        rank_rows.to_csv(out_dir / "metrics_xct_rank.csv", index=False)
        xct_rank = summarise(rank_rows[rank_rows["method"] != "limits_3sigma"], ["method"], ["xct_rank_corr"])
        md += ["## Anomaly score against XCT value (test set, no injected faults)", "",
               "Spearman rank correlation between the anomaly score of a segment and its XCT value "
               f"({int(xct_mask.sum())} segments away from the part surface). "
               "It needs no pore threshold. Negative means: more unusual segments tend to sit "
               "where the XCT value is closer to the pore side. Near 0 means no link.", "",
               to_markdown(xct_rank, {"xct_rank_corr": "Rank correlation"}, labels=LABELS), ""]
    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------------------------------------------------- traceability
    # "Alarm ratio" puts the scores of all methods on one scale:
    # 0 = a typical normal segment, 1 = exactly at the alarm level.
    def alarm_ratio(scores, method):
        span = thresholds[method] - centers[method]
        return (np.asarray(scores) - centers[method]) / (span if span != 0 else 1.0)

    tuned = [m for m in methods if m != "limits_3sigma"]
    flags = scores_test[[f"flag_{m}" for m in tuned]].to_numpy()
    ratios = np.column_stack([alarm_ratio(scores_test[f"score_{m}"], m) for m in tuned])
    hit = flags.any(axis=1)
    trace = scores_test.loc[hit, ["part", "layer", "x", "y", "z", "t_start_us", "regime"]].copy()
    for col in ("solid_below", "is_overhang", "xct_min", "xct_defect"):
        if col in scores_test:
            trace[col] = scores_test.loc[hit, col]
    trace["n_methods"] = flags[hit].sum(axis=1)
    trace["flagged_by"] = [", ".join(m for m, f in zip(tuned, row) if f) for row in flags[hit]]
    for j, m in enumerate(tuned):
        trace[f"alarm_ratio_{m}"] = np.round(ratios[hit, j], 2)
    trace["max_alarm_ratio"] = np.round(ratios[hit].max(axis=1), 2)
    trace = trace.sort_values(["n_methods", "max_alarm_ratio"], ascending=False)
    trace.round({"x": 3, "y": 3, "z": 3, "solid_below": 3}).to_csv(out_dir / "traceability.csv", index=False)

    scores_test.to_parquet(out_dir / "scores_test.parquet", index=False)
    scores_inj.to_parquet(out_dir / "scores_test_injected.parquet", index=False)
    events.to_csv(out_dir / "injected_faults.csv", index=False)
    info = {"xct_threshold": xct_thr, "thresholds": thresholds, "typical_normal_score": centers,
            "methods": methods,
            "n_segments": {"train": len(seg_train), "train_normal": int(train_ok.sum()),
                           "val": len(seg_val), "test": len(seg_test)},
            "n_injected_faults": len(events), "config": cfg}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")

    # ---------------------------------------------------------------- figures
    if make_figures and len(summary_inj):
        fig_dir = out_dir / "figures"
        ranked = summary_inj[summary_inj["method"] != "limits_3sigma"].sort_values("pr_auc", ascending=False)
        best = ranked["method"].iloc[0]
        pr_auc = dict(zip(summary_inj["method"], summary_inj["pr_auc"]))

        if curves:
            fig = plots.plot_pr_curves({m: curves[m] for m in tuned if m in curves}, pr_auc,
                                       float(y_inj[inj_mask].mean()),
                                       "Injected faults: precision-recall curves (test set)")
            fig.savefig(fig_dir / "pr_curves.png"); plots.plt.close(fig)

        # Show the layer with the most alarms; prefer layers that also have an XCT pore.
        # Layer 1 (the first layer on the build plate) is special: its melt pool is
        # bigger, so it often has the most alarms. It gets its own figure, and the
        # main figure shows the layer with the most alarms above it.
        per_layer = scores_test.groupby(["part", "layer"]).agg(
            alarms=(f"flag_{best}", "sum"), pores=("xct_defect", "sum"))
        per_layer["rank"] = per_layer["alarms"] + 1000 * ((per_layer["pores"] > 0) & (per_layer["alarms"] > 0))
        is_first = per_layer.index.get_level_values("layer") == 1
        pick = {"layer_heatmap.png": per_layer[~is_first] if (~is_first).any() else per_layer}
        if is_first.any() and (~is_first).any():
            pick["layer_heatmap_layer1.png"] = per_layer[is_first]
        for name, table in pick.items():
            part_id, layer_id = table["rank"].idxmax()
            fig = plots.plot_layer_heatmap(scores_test, int(layer_id), int(part_id),
                                           alarm_ratio(scores_test[f"score_{best}"], best),
                                           scores_test[f"flag_{best}"], best)
            fig.savefig(fig_dir / name); plots.plt.close(fig)

        if len(events):
            fig = plots.plot_signal_examples(inj_points, scores_inj, scores_inj[f"flag_{best}"], events,
                                             sg["signals"][-1], sg["window"], best)
            fig.savefig(fig_dir / "signal_examples.png"); plots.plt.close(fig)
            fig = plots.plot_fault_types(fault_types, tuned)
            fig.savefig(fig_dir / "fault_types.png"); plots.plt.close(fig)

        if summary_inj["frr_overhang"].notna().any():
            fig = plots.plot_frr_by_region(summary_inj, tuned)
            fig.savefig(fig_dir / "frr_by_region.png"); plots.plt.close(fig)

    log(f"results written to {out_dir}")
    return {"summary_injected": summary_inj, "summary_xct": summary_xct, "xct_rank": xct_rank,
            "fault_types": fault_types,
            "traceability": trace, "thresholds": thresholds, "events": events}
