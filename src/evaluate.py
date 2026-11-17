"""Measure how well a detector works.

Words:
  * positive = a segment that really is anomalous (by the label in use)
  * flagged  = a segment the detector raised an alarm for (score > threshold)

  precision  of all flagged segments, how many are really anomalous
  recall     of all anomalous segments, how many were flagged
  F1         one number that mixes precision and recall
  false rejection rate (FRR)
             of all NORMAL segments, how many were flagged by mistake.
             On a production line every false rejection costs time or scrap.
  PR-AUC     area under the precision-recall curve. It needs no threshold
             and stays honest when anomalies are rare.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def threshold_for_frr(normal_scores: np.ndarray, target_frr: float) -> float:
    """Alarm level that flags `target_frr` of the given normal segments.

    Use VALIDATION data here, never test data.
    """
    return float(np.quantile(np.asarray(normal_scores, dtype=np.float64), 1.0 - target_frr))


def detection_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Precision, recall, F1 and false rejection rate at one threshold."""
    y = np.asarray(y_true, dtype=bool)
    flagged = np.asarray(scores) > threshold
    tp = int(np.sum(flagged & y))
    fp = int(np.sum(flagged & ~y))
    fn = int(np.sum(~flagged & y))
    n_neg = int(np.sum(~y))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else (0.0 if (tp + fn) else float("nan"))
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "frr": fp / n_neg if n_neg else float("nan"),
        "n_pos": int(y.sum()), "n_neg": n_neg, "n_flagged": int(flagged.sum()),
    }


def frr_at_recall(y_true: np.ndarray, scores: np.ndarray, recall: float = 0.9) -> float:
    """False rejection rate you must accept to catch `recall` of the anomalies."""
    y = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    thr = np.quantile(scores[y], 1.0 - recall)
    return float(np.mean(scores[~y] >= thr))


def recall_at_frr(y_true: np.ndarray, scores: np.ndarray, frr: float) -> float:
    """Recall when the alarm level is set so that `frr` of the normal segments here are flagged.

    The alarm level comes from the SAME data that is scored, so this is only
    for comparing methods at one common false rejection rate, never for
    reporting how a method would work on a new part.
    """
    y = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    thr = np.quantile(scores[~y], 1.0 - frr)
    return float(np.mean(scores[y] > thr))


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Metrics that do not need a threshold."""
    y = np.asarray(y_true, dtype=bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "frr_at_90_recall": float("nan")}
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "frr_at_90_recall": frr_at_recall(y, scores, 0.9),
    }


def pr_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points of the precision-recall curve: (recall, precision)."""
    precision, recall, _ = precision_recall_curve(np.asarray(y_true, dtype=bool), scores)
    return recall, precision


def event_detection(seg_df: pd.DataFrame, flagged: np.ndarray, events: pd.DataFrame) -> pd.DataFrame:
    """Share of injected faults that were caught, per fault kind.

    A fault counts as caught if at least one of its labelled segments was
    flagged. Faults that did not end up in any labelled segment (for example
    in the unused tail of a track) are left out.
    """
    df = seg_df.assign(flagged=np.asarray(flagged, dtype=bool))
    df = df[df["is_injected"]]
    caught = df.groupby("fault_id")["flagged"].any()
    ev = events.set_index("fault_id").join(caught.rename("caught"), how="inner")
    out = ev.groupby("fault_type")["caught"].agg(n_faults="size", n_caught="sum").reset_index()
    out["detection_rate"] = out["n_caught"] / out["n_faults"]
    return out


def summarise(rows: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean and standard deviation over random seeds."""
    grouped = rows.groupby(by, sort=False)[metrics]
    mean = grouped.mean()
    std = grouped.std(ddof=0).add_suffix("_std")
    out = pd.concat([mean, std], axis=1)
    out["n_seeds"] = grouped.size()
    return out.reset_index()


def to_markdown(summary: pd.DataFrame, metrics: dict[str, str], label_col: str = "method",
                labels: dict[str, str] | None = None, percent: tuple[str, ...] = (),
                rows: list[str] | None = None) -> str:
    """Make a Markdown table: one row per method, 'mean ± std' per metric.

    `rows`: only these methods (default: all).
    """
    labels = labels or {}
    head = "| Method | " + " | ".join(metrics.values()) + " |"
    sep = "|---|" + "---:|" * len(metrics)
    lines = [head, sep]
    if rows is not None:
        summary = summary[summary[label_col].isin(rows)]
    for _, row in summary.iterrows():
        cells = []
        for m in metrics:
            mean, std = row[m], row.get(f"{m}_std", 0.0)
            if pd.isna(mean):
                cells.append("n/a")
                continue
            scale, unit, digits = (100.0, " %", 1) if m in percent else (1.0, "", 3)
            text = f"{mean * scale:.{digits}f}"
            if row.get("n_seeds", 1) > 1:
                text += f" ± {std * scale:.{digits}f}"
            cells.append(text + unit)
        lines.append(f"| {labels.get(row[label_col], row[label_col])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
