# Data notes

Dataset: **NIST "Overhang Part X4", fully registered** (ID `mds2-3761`).
Description: NIST AMS 100-69, https://doi.org/10.6028/NIST.AMS.100-69

This file has two parts: what the description says, and what still has to be
checked on the real files.

## What the description says

- Four parts. One folder per part, **250 CSV files per part, one per layer**.
- **40 columns**, only numbers.
- **One row = one melt pool camera frame.** The camera takes 10 000 frames per
  second, only while the laser is on. Rows without an image were removed.
- Rows are in time order. A time gap between two rows means the laser was off
  (a new scan track starts).
- Positions are in the machine coordinate system, in mm.
- Part size about 9 × 5 × 5 mm, 250 layers of 20 µm.
- The scan direction turns 90° from layer to layer.
- The first image of each track often shows no melt pool.

### Columns (by position)

| # | Name in this repo | Unit | Meaning |
|---|---|---|---|
| 1 | `part` | – | Part number 1–4 |
| 2 | `time_us` | µs | Build time |
| 3, 4 | `x`, `y` | mm | Commanded laser position |
| 5 | `cmd_power` | W | Commanded laser power |
| 6 | `cmd_speed` | mm/s | Commanded scan speed |
| 7, 8 | `x_real`, `y_real` | mm | Measured laser position |
| 9 | `power` | W | Measured laser power |
| 10 | `speed` | mm/s | Measured scan speed (smoothed) |
| 11–13 | `mp_length_80`, `mp_width_80`, `mp_area_80` | mm, mm, mm² | Melt pool size, image threshold 80 |
| 14–16 | `mp_length_100`, `mp_width_100`, `mp_area_100` | mm, mm, mm² | Melt pool size, threshold 100 (**used by default**) |
| 17–19 | `mp_length_120`, `mp_width_120`, `mp_area_120` | mm, mm, mm² | Melt pool size, threshold 120 |
| 20–28 | `lwi_powder_{a,b,c}`, `…_m3`, `…_m5` | – | Layer camera after powder spreading, lights A/B/C: raw, 3×3 mean, 5×5 mean |
| 29–37 | `lwi_scan_{a,b,c}`, `…_m3`, `…_m5` | – | Layer camera after the laser scan, same order |
| 38 | `xct` | – | X-ray CT voxel value at this position |
| 39 | `xct_m3` | – | XCT after 3×3×3 mean filter (**used for pore labels**) |
| 40 | `xct_m5` | – | XCT after 5×5×5 mean filter |

Columns are named by position because several labels in the NIST tables repeat.

### How the numbers were made (short)

- **Melt pool size:** each camera image is turned black/white at a threshold
  (80, 100 or 120). Area = biggest white blob. Length and width = long and
  short axis of an ellipse fitted to it. These are *apparent* sizes from
  brightness, not true temperatures.
- **Layer camera:** each point takes the nearest pixel of the layer image.
  Lining up can be off by about 2 pixels; the mean-filtered values cope with that.
- **XCT:** measured as 16-bit values, resampled to 10 µm voxels, lined up with
  the build using reference surfaces. Position uncertainty about 1.2 voxels.
  In the files the values are scaled to lie between 0 and 1.

## What the real files show

Seen in **part 1, layers 120 to 140** (21 files, 16 November 2026). Check the
points marked "to check" again when you have the full download.

| Point | What was seen |
|---|---|
| File names | `L0120.csv` … `L0140.csv`. The layer number is read correctly. |
| Header line | No |
| Columns | 40, separated by commas. No empty values. |
| Rows per layer | 3 723 to 3 968 (about 0.9 MB per file) |
| Time step | 100 µs between frames. About 3 000 µs between tracks. A few single frames are missing (200 µs steps). |
| Time column | Starts again in every layer (about 77 500 µs at the first row). It is time inside the layer, not since the start of the build. |
| Laser settings | Two: 100 W at about 900 mm/s (contour) and 195 W at about 800 mm/s (infill). About 10 % of frames are contour. |
| Measured power | About 177 to 186 W when 195 W is commanded. Near 0 W in the first frame of a track. |
| Measured speed | Noisy (standard deviation about 160 mm/s). |
| Position | x from −6 to 3 mm, y from 11 to 16 mm (machine coordinates). |
| Shape in these layers | Not a full rectangle: two thin walls (0.5 mm) on the left side, and a step on the right side. |
| Track length | From 1 to 113 frames. Many tracks across the thin walls have only 7 frames. |
| XCT values | **Between 0 and 1** (not 16-bit whole numbers). Middle value 0.537 inside the part. |
| XCT: dark or bright pores | Points at the part surface (metal mixed with air) go down to 0.34. So air is **dark**, and pores should be low values. `pore_is: low` is right. |
| Layer camera values | 0 to 255. Light A in the powder view is close to its top value (middle value 247), so it shows little. |

### What "normal" looks like (layers 120 to 140)

- **Infill (195 W):** melt pool area about 0.021 mm² (std 0.005), length about
  0.24 mm, width about 0.11 mm.
- **Contour (100 W):** area about 0.011 mm² (std 0.003), length about 0.15 mm,
  width about 0.09 mm.
- **Start of a track:** the first frame has **no melt pool** (all sizes are 0).
  The area then grows over about 3 frames (0.015, 0.017, 0.022 mm²) before it
  is steady. Zeros appear only in first frames.
- **XCT inside the part:** values lie in a narrow, even band (1 % of points
  below 0.498, 1 % above 0.575). There is no clear group of dark pore values
  in these layers.

## Still to check on the full download

- [ ] Size of each zip file: …
- [ ] Are all 250 layers of all 4 parts there, with the same file names: …
- [ ] **XCT pore threshold.** The pipeline uses median − 6 robust std by
      default (0.471 in layers 120 to 140, where almost nothing is below it).
      Look at the XCT histogram of the whole part in
      `notebooks/01_explore.ipynb`. Are there clear dark pores anywhere?
      Chosen value and why: …
- [ ] How many pore segments per part (if very few, XCT results are only a rough check): …
- [ ] Where the four parts sit on the build plate (x/y ranges): …
- [ ] Do the four parts have the same typical melt pool size: …
- [ ] Over the overhang, with all layers loaded: how much bigger is the melt pool: …
