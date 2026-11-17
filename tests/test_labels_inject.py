import numpy as np
import pandas as pd
import pytest

from src.features import make_segments
from src.inject import FAULT_TYPES, inject_faults
from src.labels import add_geometry, add_xct_surface, label_segments, xct_threshold

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


def test_xct_surface_marks_sides_top_and_start_of_scan():
    # a 2 x 1 mm block, 10 layers of 0.02 mm; the XCT scan starts at layer 3
    points = pd.concat([grid_layer(l, 2.0, step=0.05) for l in range(1, 11)], ignore_index=True)
    points["xct_m3"] = np.where(points["layer"] >= 3, 0.54, np.nan)
    out = add_xct_surface(points, "xct_m3", layer_thickness_mm=0.02, cell_mm=0.1,
                          xy_mm=0.2, z_mm=0.04)
    s = out["xct_surface"]
    middle = out["x"].between(0.45, 1.55) & out["y"].between(0.35, 0.65)
    assert s[out["layer"] <= 2].all()                      # no XCT value
    assert s[out["layer"].isin([3, 4])].all()              # just above the start of the scan
    assert s[out["layer"].isin([9, 10])].all()             # just below the top face
    assert not s[middle & out["layer"].between(5, 8)].any()
    assert s[out["layer"].between(5, 8) & (out["x"] < 0.15)].all()   # side face


def block_layer(layer, x0, x1, y0, y1, step=0.05, hole=None):
    """Points on a regular grid over [x0, x1] x [y0, y1], optionally with a square hole."""
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    xx, yy = (a.ravel() for a in np.meshgrid(xs, ys))
    if hole is not None:
        hx0, hx1, hy0, hy1 = hole
        keep = ~((xx > hx0) & (xx < hx1) & (yy > hy0) & (yy < hy1))
        xx, yy = xx[keep], yy[keep]
    return pd.DataFrame({"part": 1, "layer": layer, "x": xx, "y": yy,
                         "time_us": layer * 1e6 + 100.0 * np.arange(xx.size), "xct_m3": 0.54})


@pytest.mark.parametrize("x0, x1, y0, y1", [
    (0.0, 2.0, 0.0, 1.0),       # size a whole number of cells: max faces on a cell border
    (0.0, 2.05, 0.0, 1.05),     # max faces in the middle of a cell
    (0.03, 1.98, 0.07, 1.12),   # shifted
])
def test_xct_surface_side_band_is_the_same_on_every_face(x0, x1, y0, y1):
    # 9 layers, look at the middle one (far from the top and bottom)
    points = pd.concat([block_layer(l, x0, x1, y0, y1) for l in range(1, 10)], ignore_index=True)
    xy, cell = 0.2, 0.1
    out = add_xct_surface(points, "xct_m3", layer_thickness_mm=0.02, cell_mm=cell, xy_mm=xy, z_mm=0.04)
    mid = out[out["layer"] == 5]
    x, y = mid["x"], mid["y"]
    dist = np.minimum.reduce([x - x.min(), x.max() - x, y - y.min(), y.max() - y])
    s = mid["xct_surface"]
    # every point closer than xy to ANY face (low or high x / y) is marked ...
    assert s[dist < xy - 1e-6].all()
    # ... and points more than one cell further in are not
    assert not s[dist > xy + cell + 1e-6].any()
    # both high faces are really covered (they were not with the old rule)
    assert s[x > x.max() - xy + 1e-6].all() and s[y > y.max() - xy + 1e-6].all()


def test_xct_surface_marks_a_hole_and_a_face_over_powder():
    # layers 1-6: 1 x 1 mm; layers 7-12: 2 x 1 mm (x > 1 hangs over powder),
    # with a 0.4 x 0.4 mm hole in the middle of the new half
    hole = (1.3, 1.7, 0.3, 0.7)
    points = pd.concat([block_layer(l, 0.0, 1.0 if l <= 6 else 2.0, 0.0, 1.0,
                                    hole=None if l <= 6 else hole) for l in range(1, 13)],
                       ignore_index=True)
    out = add_xct_surface(points, "xct_m3", layer_thickness_mm=0.02, cell_mm=0.1, xy_mm=0.2, z_mm=0.04)
    s, x, y, layer = out["xct_surface"], out["x"], out["y"], out["layer"]
    # face over powder: the first 2 layers (0.04 mm) of the new half are marked
    assert s[layer.isin([7, 8]) & (x > 1.1)].all()
    # around the hole (closer than 0.2 mm to its edge)
    near_hole = layer.between(9, 10) & x.between(1.15, 1.85) & y.between(0.15, 0.85)
    assert s[near_hole].all()
    # in layers 9-10, points far from every face are not marked
    far = layer.between(9, 10) & x.between(0.35, 0.95) & y.between(0.35, 0.65)
    assert far.any() and not s[far].any()


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
    # segments without an XCT value (NaN) get no XCT label either
    df["edge_frac"] = [0.0, 0.0]
    df["xct_min"] = [np.nan, 15000.0]
    df["xct_max"] = [np.nan, 41000.0]
    out = label_segments(df, xct_thr=30000.0)
    assert out["xct_valid"].tolist() == [False, True]
    assert out["xct_defect"].tolist() == [False, True]


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
