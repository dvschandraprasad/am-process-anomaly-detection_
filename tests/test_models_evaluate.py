import numpy as np
import pandas as pd
import pytest

from src.evaluate import (detection_metrics, event_detection, frr_at_recall, ranking_metrics,
                          summarise, threshold_for_frr, to_markdown)
from src.models import HAS_TORCH, build_detector


@pytest.fixture(scope="module")
def blobs():
    """Normal data around zero, and a few points far away."""
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, (600, 6))
    normal = rng.normal(0, 1, (200, 6))
    odd = rng.normal(0, 1, (20, 6)) + 8.0
    return train, normal, odd


@pytest.mark.parametrize("name", ["limits", "limits_3sigma", "iforest", "lof", "ocsvm"])
def test_detectors_score_odd_points_higher(blobs, name):
    train, normal, odd = blobs
    det = build_detector(name, seed=0).fit(train)
    assert det.score(odd).min() > np.quantile(det.score(normal), 0.99)


def test_limits_score_is_the_worst_z_value():
    train = np.array([[0.0, 10.0], [2.0, 14.0]])          # means 1 and 12, stds 1 and 2
    det = build_detector("limits").fit(train)
    assert np.allclose(det.score(np.array([[1.0, 12.0], [4.0, 12.0], [1.0, 6.0]])), [0.0, 3.0, 3.0])
    assert build_detector("limits_3sigma").fixed_threshold == 3.0


def test_limits_per_group_use_their_own_normal():
    rng = np.random.default_rng(1)
    low, high = rng.normal(1, 0.1, (300, 1)), rng.normal(5, 0.1, (300, 1))
    X, groups = np.vstack([low, high]), np.r_[np.zeros(300), np.ones(300)].astype(int)
    det = build_detector("limits").fit(X, groups)
    # the value 5 is normal for group 1 but far off for group 0
    assert det.score(np.array([[5.0]]), np.array([1]))[0] < 2
    assert det.score(np.array([[5.0]]), np.array([0]))[0] > 20


def test_iforest_changes_with_seed_and_repeats_with_same_seed(blobs):
    train, normal, _ = blobs
    a = build_detector("iforest", seed=0).fit(train).score(normal)
    b = build_detector("iforest", seed=0).fit(train).score(normal)
    c = build_detector("iforest", seed=1).fit(train).score(normal)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_autoencoder_flags_broken_windows():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 16)

    def waves(n):
        return (np.sin(2 * np.pi * (t[None, :] + rng.random((n, 1))))[:, :, None]
                + rng.normal(0, 0.05, (n, 16, 1))).astype(np.float32)

    det = build_detector("autoencoder", seed=0, params={"epochs": 30, "batch_size": 64}).fit(waves(800))
    normal = waves(100)
    broken = normal.copy()
    broken[:, 5:9, 0] += 3.0
    assert np.median(det.score(broken)) > np.quantile(det.score(normal), 0.99)
    # context channels are read but not rebuilt
    with_context = np.concatenate([waves(200), np.ones((200, 16, 1), dtype=np.float32)], axis=2)
    det2 = build_detector("autoencoder", seed=0, params={"epochs": 1, "n_context": 1}).fit(with_context)
    assert det2.score(with_context).shape == (200,)


def test_detection_metrics_by_hand():
    y = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1])
    m = detection_metrics(y, s, threshold=0.5)        # flags items 0 and 2
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["frr"] == 0.25                           # 1 of 4 normal items flagged
    assert (m["n_pos"], m["n_neg"], m["n_flagged"]) == (2, 4, 2)


def test_metrics_without_positives_do_not_crash():
    m = detection_metrics(np.zeros(4, dtype=bool), np.array([0.1, 0.2, 0.3, 0.9]), 0.5)
    assert np.isnan(m["recall"]) and m["frr"] == 0.25
    assert np.isnan(ranking_metrics(np.zeros(4, dtype=bool), np.arange(4.0))["pr_auc"])


def test_threshold_gives_the_wanted_false_rejection_rate():
    scores = np.random.default_rng(0).normal(size=20000)
    thr = threshold_for_frr(scores, 0.02)
    assert abs(np.mean(scores > thr) - 0.02) < 0.002


def test_perfect_and_useless_scores():
    y = np.r_[np.ones(10), np.zeros(90)].astype(bool)
    perfect = ranking_metrics(y, y.astype(float))
    assert perfect["pr_auc"] == 1.0 and perfect["frr_at_90_recall"] == 0.0
    assert frr_at_recall(y, np.zeros(100), 0.9) == 1.0   # must flag everything


def test_event_detection_counts_caught_faults():
    seg = pd.DataFrame({"is_injected": [True, True, True, False], "fault_id": [0, 0, 1, -1]})
    events = pd.DataFrame({"fault_id": [0, 1, 2], "fault_type": ["spike", "spike", "drift"]})
    out = event_detection(seg, np.array([False, True, False, True]), events).set_index("fault_type")
    # fault 0 caught (one of its two segments), fault 1 missed, fault 2 has no segment
    assert out.loc["spike", "n_faults"] == 2 and out.loc["spike", "detection_rate"] == 0.5
    assert "drift" not in out.index


def test_summary_and_markdown():
    rows = pd.DataFrame({"method": ["iforest", "iforest", "limits"], "frr": [0.01, 0.03, 0.02]})
    summary = summarise(rows, ["method"], ["frr"])
    line = summary.set_index("method").loc["iforest"]
    assert np.isclose(line["frr"], 0.02) and np.isclose(line["frr_std"], 0.01) and line["n_seeds"] == 2
    text = to_markdown(summary, {"frr": "FRR"}, percent=("frr",))
    assert "2.0 ± 1.0 %" in text and "| limits | 2.0 % |" in text
