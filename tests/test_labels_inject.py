import numpy as np
import pandas as pd

from src.features import make_segments
from src.inject import FAULT_TYPES, inject_faults
from src.labels import add_geometry, label_segments, xct_threshold

SIGNALS = ["mp_length_100", "mp_width_100", "mp_area_100"]


def grid_layer(layer, x_max, step=0.1):
    xs = np.arange(0.05, x_max, step)
    ys = np.arange(0.05, 1.0, step)
    xx, yy = np.meshgrid(xs, ys)
    n = xx.size
    return pd.DataFrame({"part": 1, "layer": layer, "x": xx.ravel(), "y": yy.ravel(),
                         "time_us": layer * 1e6 + 100.0 * np.arange(n)})


def test_solid_below_sees_the_overhang():
    # layers 1-3 are 1 mm long, layers 4-6 are 2 mm long: x > 1 hangs over powder
    points = pd.concat([grid_layer(l, 1.0 if l <= 3 else 2.0) for l in range(1, 7)], ignore_index=True)
    out = add_geometry(points, layer_thickness_mm=0.1, cell_mm=0.25, cap_mm=1.0)
    over = out["x"] > 1.05

    # first loaded layer: assume solid metal below
    assert (out.loc[out["layer"] == 1, "solid_below"] == 1.0).all()
    # first layer of the overhang: nothing below; two layers later: 0.2 mm below
    assert (out.loc[(out["layer"] == 4) & over, "solid_below"] == 0.0).all()
    assert np.allclose(out.loc[(out["layer"] == 6) & over, "solid_below"], 0.2)
    # the old part keeps full support
    assert (out.loc[(out["layer"] == 6) & (out["x"] < 0.7), "solid_below"] == 1.0).all()
    # surface cells are marked, the middle is not
    mid = (out["x"].between(0.3, 0.7)) & (out["y"].between(0.3, 0.7))
    assert not out.loc[mid, "is_edge"].any()
    assert out.loc[out["x"] < 0.2, "is_edge"].all()


def test_xct_threshold_sits_between_metal_and_pores():
    rng = np.random.default_rng(0)
    metal = rng.normal(0.54, 0.011, 5000)     # real NIST values lie between 0 and 1
    assert 0.40 < xct_threshold(metal, "low", k=6.0) < 0.50
    assert 0.58 < xct_threshold(metal, "high", k=6.0) < 0.68


def test_label_segments():
    df = pd.DataFrame({"solid_below": [1.0, 0.1], "xct_min": [40000.0, 15000.0],
                       "xct_max": [42000.0, 41000.0], "edge_frac": [0.0, 0.0], "n_fault": [0, 5]})
    out = label_segments(df, overhang_max_solid_mm=0.5, xct_thr=30000.0, min_fault_points=3)
    assert out["is_overhang"].tolist() == [False, True]
    assert out["xct_defect"].tolist() == [False, True]
    assert out["is_injected"].tolist() == [False, True]
    # surface segments get no XCT label
    df["edge_frac"] = [0.0, 0.9]
    out = label_segments(df, xct_thr=30000.0)
    assert out["xct_valid"].tolist() == [True, False]
    assert not out["xct_defect"].any()


def test_injection_changes_only_fault_points(dataset):
    points = dataset[1]
    out, events = inject_faults(points, SIGNALS, events_per_1000_points=3.0, seed=1)
    assert len(events) >= 8
    assert set(events["fault_type"]) == set(FAULT_TYPES)
    hit = out["fault_id"].to_numpy() >= 0
    assert hit.sum() == (events["row_end"] - events["row_start"]).sum()
    for s in SIGNALS:
        np.testing.assert_array_equal(out.loc[~hit, s], points.loc[~hit, s])
        assert (out[s] >= 0).all()
    # nothing else is touched
    for col in ("x", "y", "time_us", "power", "xct"):
        np.testing.assert_array_equal(out[col], points[col])
    # every fault stays on one track of one layer
    for e in events.itertuples():
        rows = out.iloc[e.row_start:e.row_end]
        assert rows["layer"].nunique() == 1
        assert (np.diff(rows["time_us"]) < 150).all()


def test_injection_is_repeatable_and_seeded(dataset):
    a, ev_a = inject_faults(dataset[1], SIGNALS, 2.0, seed=5)
    b, ev_b = inject_faults(dataset[1], SIGNALS, 2.0, seed=5)
    c, _ = inject_faults(dataset[1], SIGNALS, 2.0, seed=6)
    pd.testing.assert_frame_equal(ev_a, ev_b)
    np.testing.assert_array_equal(a["mp_area_100"], b["mp_area_100"])
    assert not np.array_equal(a["mp_area_100"], c["mp_area_100"])


def test_spike_goes_up_and_drop_goes_down(dataset):
    points = dataset[1]
    out, events = inject_faults(points, ["mp_area_100"], 3.0, types=["spike", "drop"], seed=2)
    for e in events.itertuples():
        diff = (out["mp_area_100"] - points["mp_area_100"]).iloc[e.row_start:e.row_end]
        assert (diff > 0).all() if e.fault_type == "spike" else (diff < 0).all()


def test_injected_faults_reach_the_segment_table(dataset):
    out, events = inject_faults(dataset[1], SIGNALS, 3.0, seed=1)
    seg = make_segments(out, SIGNALS, window=16)
    df = label_segments(seg.df, min_fault_points=3)
    assert df["is_injected"].sum() >= len(events) * 0.5
    assert set(df.loc[df["is_injected"], "fault_id"]) <= set(events["fault_id"])
