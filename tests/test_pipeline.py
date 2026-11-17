"""End-to-end checks: the whole experiment on tiny fake data."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.pipeline import prepare_points, run, split_points

REPO = Path(__file__).resolve().parents[1]


def small_config(data_dir, out_dir, **extra):
    cfg = {
        "data": {"parts": {p: str(data_dir / f"part{p}") for p in (1, 2, 3)},
                 "cache_dir": None, "layer_thickness_mm": 0.1},
        "split": {"mode": "by_part", "train": [1], "val": [2], "test": [3]},
        "inject": {"events_per_1000_points": 3.0},
        "methods": ["limits", "iforest", "lof"],
        "seeds": [0, 1],
        "output_dir": str(out_dir),
    }
    cfg.update(extra)
    return load_config(None, cfg)


def test_split_by_part_keeps_parts_apart(data_dir, tmp_path):
    cfg = small_config(data_dir, tmp_path)
    sets = split_points(prepare_points(cfg), cfg)
    assert set(sets["train"]["part"]) == {1}
    assert set(sets["val"]["part"]) == {2}
    assert set(sets["test"]["part"]) == {3}


def test_split_by_layer_blocks_shares_no_layer(data_dir, tmp_path):
    cfg = small_config(data_dir, tmp_path, split={"mode": "by_layer_blocks", "block_layers": 1})
    cfg["data"]["parts"] = {1: cfg["data"]["parts"][1]}
    sets = split_points(prepare_points(cfg), cfg)
    layers = {k: set(v["layer"]) for k, v in sets.items()}
    assert layers["train"] == {1, 2, 3, 6, 7, 8} and layers["val"] == {4} and layers["test"] == {5}


def test_full_run_writes_tables_figures_and_traceability(data_dir, tmp_path):
    out = tmp_path / "out"
    res = run(small_config(data_dir, out))

    summary = res["summary_injected"].set_index("method")
    assert list(summary.index) == ["limits_3sigma", "limits", "limits_levels", "iforest", "lof"]
    assert summary.loc["iforest", "n_seeds"] == 2 and summary.loc["limits", "n_seeds"] == 1
    for col in ("precision", "recall", "frr", "pr_auc"):
        assert summary[col].between(0, 1).all()
    # the fixed limits must clearly beat random guessing on injected faults
    base_rate = pd.read_parquet(out / "scores_test_injected.parquet")["is_injected"].mean()
    assert summary.loc["limits", "pr_auc"] > 3 * base_rate

    for name in ("results.md", "metrics_injected.csv", "fault_types.csv", "traceability.csv",
                 "run_info.json", "figures/pr_curves.png", "figures/layer_heatmap.png",
                 "figures/signal_examples.png", "figures/fault_types.png",
                 "figures/layer_heatmap_layer1.png"):
        assert (out / name).exists(), name

    trace = pd.read_csv(out / "traceability.csv")
    assert {"part", "layer", "x", "y", "z", "flagged_by", "n_methods"} <= set(trace.columns)
    assert (trace["part"] == 3).all()            # only the test part is reported
    assert (trace["n_methods"] >= 1).all()


def test_thresholds_come_from_validation_not_test(data_dir, tmp_path):
    """Changing the test data must not change any alarm level (no leakage)."""
    a = run(small_config(data_dir, tmp_path / "a", methods=["limits", "iforest"], seeds=[0]),
            make_figures=False)
    cfg = small_config(data_dir, tmp_path / "b", methods=["limits", "iforest"], seeds=[0])
    cfg["inject"]["severity"] = 3.0              # much bigger faults in the test set
    cfg["inject"]["seed"] = 99
    b = run(cfg, make_figures=False)
    assert a["thresholds"] == b["thresholds"]
    # ... while the results do change
    assert not np.isclose(a["summary_injected"]["recall"].iloc[1], b["summary_injected"]["recall"].iloc[1])


def test_sample_data_in_the_repo_runs(tmp_path):
    if not (REPO / "data" / "sample" / "part1").exists():
        pytest.skip("sample data not generated")
    cfg = load_config(REPO / "configs" / "sample.yaml")
    cfg["data"]["parts"] = {p: str(REPO / path) for p, path in cfg["data"]["parts"].items()}
    cfg.update(output_dir=str(tmp_path / "sample"), methods=["limits", "iforest"], seeds=[0])
    res = run(cfg, make_figures=False)
    assert len(res["summary_injected"]) == 4   # 3-sigma, tuned k, level features only, iforest


def test_extra_numbers_script_reads_a_run(data_dir, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("extra_numbers", REPO / "scripts" / "extra_numbers.py")
    extra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extra)

    cfg = small_config(data_dir, tmp_path / "out", methods=["limits", "iforest"], seeds=[0])
    res = run(cfg, make_figures=False)
    data = "\n".join(extra.data_numbers(cfg))
    assert "| tracks |" in data and "Part 3" in data
    text = "\n".join(extra.result_numbers(tmp_path / "out"))
    assert f"rows in traceability.csv: {len(res['traceability'])}" in text
    assert "Inside each layer" in text
