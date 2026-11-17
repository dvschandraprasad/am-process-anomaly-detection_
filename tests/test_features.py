import numpy as np
import pandas as pd

from src.features import assign_tracks, make_segments, window_features, window_starts


def toy_points(track_lengths, layer=1, part=1, gap_us=600.0):
    """Points on straight tracks, 100 us apart, with a longer gap between tracks."""
    t, rows = 0.0, []
    for k, n in enumerate(track_lengths):
        for i in range(n):
            rows.append({"part": part, "layer": layer, "z": 0.02 * layer, "time_us": t,
                         "x": 0.08 * i, "y": 0.1 * k, "cmd_power": 195.0, "speed": 800.0,
                         "sig": float(i)})
            t += 100.0
        t += gap_us
    return pd.DataFrame(rows)


def test_tracks_split_at_time_gaps_and_layer_changes():
    a = toy_points([5, 7], layer=1)
    b = toy_points([4], layer=2)
    b["time_us"] += a["time_us"].max() + 100.0   # no time gap, but a new layer
    track = assign_tracks(pd.concat([a, b], ignore_index=True))
    assert np.bincount(track).tolist() == [5, 7, 4]


def test_one_missing_frame_does_not_split_a_track():
    points = toy_points([10])
    points = points.drop(index=4).reset_index(drop=True)     # a 200 us step inside the track
    assert assign_tracks(points).max() == 0


def test_windows_do_not_cross_tracks_or_overlap():
    points = toy_points([10, 9, 3])
    track = assign_tracks(points)
    starts = window_starts(track, window=4, drop_first=1)
    # track 1: rows 1-4 and 5-8 (row 9 left over); track 2: rows 11-14, 15-18; track 3: too short
    assert starts.tolist() == [1, 5, 11, 15]
    for s in starts:
        assert track[s] == track[s + 3]


def test_window_features_on_known_numbers():
    w = np.array([[0.0, 1.0, 2.0, 3.0]])[:, :, None]   # one segment, one signal
    f = window_features(w, ["sig"]).iloc[0]
    assert f["sig_mean"] == 1.5
    assert f["sig_min"] == 0.0 and f["sig_max"] == 3.0
    assert np.isclose(f["sig_std"], np.std([0, 1, 2, 3]))
    assert np.isclose(f["sig_slope"], 1.0)    # rises by 1 per frame
    assert np.isclose(f["sig_jitter"], 1.0)


def test_make_segments_shapes_and_positions():
    points = toy_points([10, 9, 3])
    seg = make_segments(points, ["sig"], ["cmd_power"], window=4, drop_first=1, xct_column=None)
    assert len(seg) == 4
    assert seg.windows.shape == (4, 4, 1)
    assert seg.feature_cols[-2:] == ["cmd_power_mean", "track_start_flag"]
    # the first segment of each track is marked (tracks have 10 and 9 frames)
    assert seg.df["track_start"].tolist() == [True, False, True, False]
    first = seg.df.iloc[0]
    assert first["row_start"] == 1
    assert np.isclose(first["x"], np.mean([0.08, 0.16, 0.24, 0.32]))
    assert first["regime"] == 195
    # the raw windows hold the signal values of the right rows
    assert seg.windows[0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_segments_of_real_shaped_data(dataset):
    seg = make_segments(dataset[1], ["mp_length_100", "mp_width_100", "mp_area_100"],
                        ["cmd_power", "speed"], window=16)
    assert len(seg) > 50
    assert not seg.df[seg.feature_cols].isna().any().any()
    assert set(seg.df["regime"].unique()) == {100, 195}
