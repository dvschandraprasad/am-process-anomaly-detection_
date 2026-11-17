"""Labels: what counts as "not normal".

Three kinds of labels are used, and they mean different things:
  * geometry (overhang): the laser is melting over loose powder instead of
    solid metal. This region behaves differently, but it is not a defect.
  * XCT: the X-ray CT scan after printing shows a pore at this position.
    This is the closest thing to a real defect label.
  * injected: a synthetic fault was added to the signal (see inject.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- geometry --

def add_geometry(
    points: pd.DataFrame,
    layer_thickness_mm: float,
    cell_mm: float = 0.25,
    cap_mm: float = 1.0,
) -> pd.DataFrame:
    """Add `solid_below` and `is_edge` to a point table.

    solid_below [mm]: how much printed metal lies directly under the point,
        counted in layers, up to `cap_mm`. It is worked out from the data
        itself: an x/y cell is "metal" in a layer if the laser was there.
        Under the first loaded layer we assume solid metal (the build plate,
        or layers that were not loaded).
    is_edge: the point lies in a cell at the outer surface of the part
        (at least one neighbour cell has no metal in this layer).
    """
    points = points.copy()
    solid = np.zeros(len(points), dtype=np.float32)
    edge = np.zeros(len(points), dtype=bool)

    for _, part_rows in points.groupby("part", sort=False).indices.items():
        sub = points.iloc[part_rows]
        x, y = sub["x"].to_numpy(), sub["y"].to_numpy()
        layer = sub["layer"].to_numpy()
        ix = np.floor((x - x.min()) / cell_mm).astype(np.int64)
        iy = np.floor((y - y.min()) / cell_mm).astype(np.int64)
        nx, ny = ix.max() + 1, iy.max() + 1

        layers = np.unique(layer)
        cap_layers = cap_mm / layer_thickness_mm
        run = np.full((nx, ny), cap_layers, dtype=np.float64)  # layers of metal below
        prev_occ = None
        prev_layer = None
        for lay in layers:
            m = layer == lay
            occ = np.zeros((nx, ny), dtype=bool)
            occ[ix[m], iy[m]] = True
            if prev_occ is not None:
                run = np.where(prev_occ, run + (lay - prev_layer), 0.0)
                run = np.minimum(run, cap_layers)
            solid[part_rows[m]] = (run[ix[m], iy[m]] * layer_thickness_mm).astype(np.float32)

            padded = np.pad(occ, 1, constant_values=False)
            all_neighbours = np.ones((nx, ny), dtype=bool)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    all_neighbours &= padded[1 + dx: 1 + dx + nx, 1 + dy: 1 + dy + ny]
            edge[part_rows[m]] = ~all_neighbours[ix[m], iy[m]]

            prev_occ, prev_layer = occ, lay

    points["solid_below"] = np.minimum(solid, cap_mm)
    points["is_edge"] = edge
    return points


# --------------------------------------------------------------------- XCT --

def add_xct_surface(
    points: pd.DataFrame,
    xct_column: str,
    layer_thickness_mm: float,
    cell_mm: float = 0.25,
    xy_mm: float = 0.5,
    z_mm: float = 0.06,
) -> pd.DataFrame:
    """Add `xct_surface`: points where the XCT value is no good as a pore label.

    Near any surface of the part an XCT voxel holds metal and air (or powder),
    so its value is low and looks like a pore. In the real NIST data almost
    all low XCT values sit at such places (see docs/data_notes.md). A point is
    marked if
      * side:   a cell within `xy_mm` in the same layer has no XCT value
                (outside of the part, or a hole)
      * top / bottom: a cell next to it has no XCT value in one of the layers
                within `z_mm` above or below it (top face, face over powder,
                lowest layers of the XCT scan)
      * it has no XCT value itself.
    Above the last loaded layer counts as air (so the last layers of a partial
    load are marked too). Below the first loaded layer counts as metal, like
    in `add_geometry`.
    """
    points = points.copy()
    surface = np.ones(len(points), dtype=bool)
    if xct_column not in points:
        points["xct_surface"] = surface
        return points
    r_xy = max(1, int(round(xy_mm / cell_mm)))
    n_z = max(1, int(round(z_mm / layer_thickness_mm)))

    def all_cells(occ: np.ndarray, r: int) -> np.ndarray:
        """True where every cell in the (2r+1) x (2r+1) square around has a value."""
        _, nx, ny = occ.shape
        padded = np.pad(occ, ((0, 0), (r, r), (r, r)), constant_values=False)
        out = np.ones_like(occ)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                out &= padded[:, r + dx: r + dx + nx, r + dy: r + dy + ny]
        return out

    for _, rows in points.groupby("part", sort=False).indices.items():
        sub = points.iloc[rows]
        x, y = sub["x"].to_numpy(), sub["y"].to_numpy()
        ix = np.floor((x - x.min()) / cell_mm).astype(np.int64)
        iy = np.floor((y - y.min()) / cell_mm).astype(np.int64)
        layers, pos = np.unique(sub["layer"].to_numpy(), return_inverse=True)
        has = np.isfinite(sub[xct_column].to_numpy())
        grid = np.zeros((len(layers), ix.max() + 1, iy.max() + 1), dtype=bool)
        grid[pos[has], ix[has], iy[has]] = True

        side = ~all_cells(grid, r_xy)[pos, ix, iy]
        near = all_cells(grid, 1)
        vertical = np.zeros(len(rows), dtype=bool)
        for k in range(1, n_z + 1):
            up = pos + k
            ok = up < len(layers)
            vertical |= ~ok  # nothing loaded above: count as air
            vertical[ok] |= ~near[up[ok], ix[ok], iy[ok]]
            down = pos - k
            ok = down >= 0   # nothing loaded below: count as metal
            vertical[ok] |= ~near[down[ok], ix[ok], iy[ok]]
        surface[rows] = side | vertical | ~has

    points["xct_surface"] = surface
    return points


def xct_threshold(values: np.ndarray, pore_is: str = "low", k: float = 6.0) -> float:
    """Pick the XCT value beyond which a voxel counts as a pore.

    Solid metal gives a narrow range of XCT values. We take the median and a
    robust spread (MAD), and call everything more than `k` spreads away on the
    pore side a pore. CHECK THIS against the XCT histogram of the real data.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    med = np.median(values)
    spread = 1.4826 * np.median(np.abs(values - med))
    spread = spread if spread > 0 else max(values.std(), 1e-9)
    return float(med - k * spread) if pore_is == "low" else float(med + k * spread)


def label_segments(
    seg_df: pd.DataFrame,
    overhang_max_solid_mm: float = 0.5,
    xct_thr: float | None = None,
    pore_is: str = "low",
    ignore_edge: bool = True,
    min_fault_points: int = 3,
) -> pd.DataFrame:
    """Add the label columns to a segment table.

    is_overhang   little solid metal below the segment
    xct_defect    XCT shows a pore inside the segment
    xct_valid     False for segments at the part surface (XCT there mixes
                  metal and air, so it cannot be trusted as a pore label)
                  and for segments without an XCT value. Uses
                  `xct_surface_frac` if it is there, else `edge_frac`.
    is_injected   the segment holds an injected fault
    """
    df = seg_df.copy()
    if "solid_below" in df:
        df["is_overhang"] = df["solid_below"] < overhang_max_solid_mm
    else:
        df["is_overhang"] = False

    if xct_thr is not None and "xct_min" in df:
        if pore_is == "low":
            df["xct_defect"] = df["xct_min"] < xct_thr
        else:
            df["xct_defect"] = df["xct_max"] > xct_thr
        if ignore_edge and "xct_surface_frac" in df:
            df["xct_valid"] = df["xct_surface_frac"] < 0.5
        elif ignore_edge and "edge_frac" in df:
            df["xct_valid"] = df["edge_frac"] < 0.5
        else:
            df["xct_valid"] = True
        # no XCT value here (in the NIST files the lowest ~20 layers have none)
        df["xct_valid"] &= np.isfinite(df["xct_min"]) & np.isfinite(df["xct_max"])
        df["xct_defect"] = df["xct_defect"] & df["xct_valid"]
    else:
        df["xct_defect"] = False
        df["xct_valid"] = False

    if "n_fault" in df:
        df["is_injected"] = df["n_fault"] >= min_fault_points
    else:
        df["is_injected"] = False
    return df
