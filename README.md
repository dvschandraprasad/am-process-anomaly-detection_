# Anomaly detection in metal additive manufacturing process data

Learn what "normal" looks like in in-situ process monitoring data from metal
3D printing (laser powder bed fusion). Then find, compare and show the places
that are **not** normal, and check them against X-ray CT.

![tests](https://github.com/dvschandraprasad/am-process-anomaly-detection/actions/workflows/tests.yml/badge.svg)

> **In short.** On public NIST melt pool data (4 parts: train on 2, set the
> alarm level on 1, test on 1), Local Outlier Factor and One-Class SVM catch
> somewhat more **synthetic** faults than fixed limits set per region (recall
> 0.67 and 0.66 against 0.61 when all flag 1 % of normal segments), and
> about the same when 90 % of faults must be caught. Real defects could not
> be tested: at the points where XCT is read, no pore shows up away from the
> surface. All numbers come from one test part.

> **Status.** The full run on all four real NIST parts is done (17 November
> 2026): train on parts 1 and 2, set the alarm level on part 3, test on
> part 4. See [Results](#results). The short report is in
> [`docs/report.md`](docs/report.md).

## The question

On a production line, parts are often checked with fixed limits: a value
outside `mean ± k·std` is rejected. That is simple, but it rejects good parts
when "normal" is different in different places of the part.

This project asks: **can a learned model of normal behaviour find real
problems with fewer false rejections than fixed limits?**

## Data

**NIST "Overhang Part X4", fully registered dataset (2025), ID `mds2-3761`.**

- Four nominally identical nickel alloy 625 parts, each 250 layers of 20 µm.
- Each part has a 45° overhang and a round hole. There the laser melts over
  loose powder instead of solid metal, so the process behaves differently.
- A high-speed camera looked at the melt pool through the laser path at
  10 000 frames per second. NIST turned each frame into numbers: melt pool
  length, width and area.
- Every frame is lined up with the laser position, the laser power, the layer
  camera image and the **X-ray CT value** of the finished part at that spot.
- One folder per part, one CSV file per layer, 40 columns, one row per frame.

Column list and open questions: [`docs/data_notes.md`](docs/data_notes.md).

## Method

| Step | What happens | Code |
|---|---|---|
| Load | Read the layer files into one table per part | `src/load.py` |
| Geometry | For every point: how much solid metal is below it? Is it at the surface? | `src/labels.py` |
| Segments | Cut each laser track into pieces of 12 frames (1.2 ms, about 1 mm). A segment is the unit that gets a score. | `src/features.py` |
| Features | Per segment and signal: mean, std, min, max, slope, jitter | `src/features.py` |
| Split | Whole parts go to train / validation / test. Never a random split. | `src/pipeline.py` |
| Labels | (1) injected faults, (2) pores seen by X-ray CT, (3) overhang region | `src/inject.py`, `src/labels.py` |
| Detect | Learn "normal" from normal training segments only, then score the test part | `src/models/` |
| Evaluate | Precision, recall, false rejection rate, PR-AUC, run time | `src/evaluate.py` |
| Show | Layer heatmap, PR curves, signal plots, traceability table | `src/plots.py` |

### Methods compared

| Method | Kind | Idea |
|---|---|---|
| Fixed limits, 3-sigma rule | Rule (baseline) | Flag if any of the 18 melt pool features (6 numbers × 3 signals) is outside mean ± 3 std. Separate limits per laser setting, track start and region. No tuning. |
| Fixed limits, tuned k | Rule (baseline) | Same, but k is set on validation data like for the other methods |
| Fixed limits, level features only | Rule (extra baseline) | Like tuned k, but only mean, min and max of each signal (9 features). Shown to see how much the choice of features matters. |
| Isolation Forest | Classic ML | Unusual points are easy to separate with random cuts |
| Local Outlier Factor | Classic ML | Unusual points sit in emptier areas than their neighbours |
| One-Class SVM | Classic ML | Draw a tight border around normal data |
| 1D conv. autoencoder | Deep learning | Learn to rebuild normal signal pieces; a bad rebuild = unusual |

### Rules for a fair comparison

- All methods use the same train / validation / test parts.
- Training uses **normal segments only** (segments with an XCT pore are
  removed; in the full NIST run there were none).
- The alarm level of each method is set on the **validation part** so that 1 %
  of normal validation segments are flagged.
- **No model, scaler, alarm level or fault size is fitted on the test part.**
  Tests check that changing the test part changes no alarm level
  (`tests/test_pipeline.py`).
- Known exception: the XCT surface rule (which points count as "near a
  surface") was chosen after looking at the XCT values of all four parts,
  including the test part. It only touches the XCT labels.
- The figures show the method with the highest PR-AUC on the test part. This
  only picks what to show.
- Methods with randomness run with 3 seeds. The ± in the tables is the
  standard deviation over these model seeds. The injected faults are placed
  with one fixed seed, so the ± does not include the effect of where the
  faults land.

### The three kinds of labels

| Label | Meaning | Use |
|---|---|---|
| Injected faults | Spikes, drops, drifts and noise bursts added to normal test signals. Place and size are known exactly. | Main numbers |
| XCT pores | The X-ray CT value at the segment says "pore". Points near any surface of the part are left out (closer than 0.25 mm to a side face, on some faces up to 0.5 mm, or closer than 0.06 mm to a top or bottom face), because there XCT mixes metal and air. | Check against real defects |
| Overhang region | Less than 0.5 mm solid metal below the segment | Shows where false rejections come from. **Not** a defect. |

## Quick start (fake sample data, about 30 seconds)

```bash
git clone https://github.com/dvschandraprasad/am-process-anomaly-detection.git
cd am-process-anomaly-detection
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch, as in CI
pip install -r requirements.txt

pytest -q                                                   # run the tests
python scripts/run_pipeline.py --config configs/sample.yaml # run everything
```

Output goes to `results/sample/`: `results.md` (tables), `figures/`,
`traceability.csv`.

The sample in `data/sample/` is **invented data** with the same file layout
as the NIST files. It only shows that the code runs.

## Run on the real NIST data

```bash
# 1. Download one part first (large file) and look inside
python scripts/download_data.py --parts 1
python scripts/inspect_data.py data/raw/part1

# 2. First run with one part (split by blocks of layers)
python scripts/run_pipeline.py --config configs/nist_x4_one_part.yaml

# 3. Download the rest, look at the XCT values, and do the real run (split by part)
python scripts/download_data.py --parts 2 3 4
python scripts/check_xct.py --config configs/nist_x4.yaml
python scripts/run_pipeline.py --config configs/nist_x4.yaml

# 4. Numbers in the docs that the pipeline does not write (after step 3)
python scripts/extra_numbers.py --config configs/nist_x4.yaml
```

Before step 2, read the output of `inspect_data.py` and check the open points
in [`docs/data_notes.md`](docs/data_notes.md), most of all the XCT pore
threshold.

The NIST zip files unpack one folder deeper (`data/raw/part1/part01/L0001.csv`).
The loader searches all sub-folders, so `data/raw/part1` in the config works.

The full run takes about 3 to 4 minutes on a machine with 2 CPU cores and
needs about 3 GB of memory. Reading the CSV files the first time adds about
20 seconds; after that a Parquet copy in `data/processed/` is used.

Then look at the data yourself in `notebooks/01_explore.ipynb` and at the
results in `notebooks/02_results.ipynb`.

## Results

**Full run on the real NIST data** (`configs/nist_x4.yaml`, 17 November 2026).

| | |
|---|---|
| Train | parts 1 and 2: 176 756 segments |
| Validation (sets the alarm level) | part 3: 88 378 segments |
| Test | part 4: 88 378 segments |
| Injected faults in the test part | 594 faults: 1 362 faulty and 86 881 normal segments (1.5 % faulty). 135 segments with only 1 or 2 fault frames are left out. |
| Seeds | 3 for methods with randomness; ± is the std over these model seeds |

The alarm level of every method is set on part 3 so that 1 % of its normal
segments are flagged. Only the 3-sigma rule keeps its fixed level.

### Injected faults (test part 4)

| Method | Precision | Recall | F1 | False rejection rate | PR-AUC | FRR at 90 % recall | Run time [ms/layer] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed limits (3-sigma rule) | 0.106 | 0.830 | 0.188 | 10.9 % | 0.584 | 28.5 % | 0.092 |
| Fixed limits (tuned k) | 0.423 | 0.653 | 0.513 | 1.4 % | 0.584 | 28.5 % | 0.079 |
| Fixed limits, level features only (tuned k) | 0.370 | 0.570 | 0.449 | 1.5 % | 0.370 | 35.9 % | 0.057 |
| Isolation Forest | 0.415 ± 0.017 | 0.510 ± 0.034 | 0.458 ± 0.024 | 1.1 ± 0.0 % | 0.424 ± 0.046 | 34.8 ± 1.2 % | 2.283 ± 0.101 |
| Local Outlier Factor | 0.359 ± 0.012 | 0.701 ± 0.008 | 0.475 ± 0.010 | 2.0 ± 0.1 % | 0.667 ± 0.004 | 26.8 ± 0.9 % | 11.487 ± 0.594 |
| One-Class SVM | 0.419 ± 0.013 | 0.697 ± 0.008 | 0.523 ± 0.010 | 1.5 ± 0.1 % | 0.654 ± 0.009 | 31.1 ± 2.4 % | 2.275 ± 0.095 |
| Conv. autoencoder | 0.318 ± 0.020 | 0.322 ± 0.026 | 0.320 ± 0.023 | 1.1 ± 0.0 % | 0.304 ± 0.021 | 59.6 ± 1.3 % | 0.403 ± 0.042 |

Precision looks low for every method because faulty segments are rare
(1.5 %). A method that guesses at random gets a PR-AUC of about 0.015.
Run time is the time to score the test part, per layer, on 2 CPU cores.
It changed clearly between full runs (for example the autoencoder took
0.38 to 0.48 ms per layer), so only the order of size counts.

**Recall at the same false rejection rate** (for comparison only). With the
alarm level from part 3, the methods end up at different false rejection
rates on part 4 (1.1 to 2.0 %). To compare them at one point, here the
alarm level is set **on the test part** so that every method flags exactly
1 % or 2 % of its normal segments. This is not how a method would do on a
new part.

| Method | Recall at 1 % FRR | Recall at 2 % FRR |
|---|---:|---:|
| Fixed limits (tuned k) | 0.614 | 0.684 |
| Fixed limits, level features only (tuned k) | 0.491 | 0.626 |
| Isolation Forest | 0.490 ± 0.039 | 0.580 ± 0.010 |
| Local Outlier Factor | 0.665 ± 0.003 | 0.702 ± 0.007 |
| One-Class SVM | 0.663 ± 0.004 | 0.717 ± 0.011 |
| Conv. autoencoder | 0.315 ± 0.025 | 0.387 ± 0.025 |

![Precision-recall curves, full run](docs/images/full_pr_curves.png)

**Share of injected faults caught, by kind** (over the 576 of 594 faults
that have at least one labelled segment: 148 drifts, 143 drops, 148 noise
bursts, 137 spikes)

| Method | drift | drop | noise | spike |
|---|---:|---:|---:|---:|
| Fixed limits (3-sigma rule) | 99 % | 100 % | 100 % | 100 % |
| Fixed limits (tuned k) | 86 % | 98 % | 100 % | 97 % |
| Fixed limits, level features only (tuned k) | 78 % | 95 % | 97 % | 77 % |
| Isolation Forest | 37 % | 87 % | 100 % | 89 % |
| Local Outlier Factor | 87 % | 100 % | 100 % | 98 % |
| One-Class SVM | 92 % | 100 % | 100 % | 99 % |
| Conv. autoencoder | 8 % | 38 % | 100 % | 20 % |

**False rejections by region** (normal test segments: 83 497 with solid
metal below, 3 384 in the overhang region)

| Method | Solid metal below | Overhang region |
|---|---:|---:|
| Fixed limits (3-sigma rule) | 11.0 % | 10.3 % |
| Fixed limits (tuned k) | 1.4 % | 1.4 % |
| Fixed limits, level features only (tuned k) | 1.5 % | 1.0 % |
| Isolation Forest | 0.4 ± 0.0 % | 19.4 ± 0.9 % |
| Local Outlier Factor | 1.9 ± 0.1 % | 4.6 ± 0.1 % |
| One-Class SVM | 1.2 ± 0.1 % | 9.9 ± 0.2 % |
| Conv. autoencoder | 1.0 ± 0.0 % | 2.0 ± 0.2 % |

![False rejections by region, full run](docs/images/full_frr_by_region.png)

### X-ray CT

**At the points where XCT is read, and away from the surface, no XCT value
is below the pore threshold** (0.479), in any of the four parts. Low XCT
values do exist, but all of them sit in the surface band that is left out:
the lowest layers of the XCT scan, the top layers, top faces and side faces
([details](docs/data_notes.md#xct-on-the-full-parts)). So precision and
recall against real pores **cannot be given**.

This does **not** mean the parts have no pores. The check cannot see:

- **small pores.** Each value is a 3 × 3 × 3 mean of 10 µm voxels, and the
  threshold is 6 robust standard deviations below the middle value. Roughly
  a fifth of the 27 voxels must be air before a value drops below it (a
  rough estimate from the values at the surface).
- **pores between the sampled spots.** XCT is read only where a camera
  frame is: about 12 % of each layer's area.
- **pores near the surface**, where they are common in this kind of
  process. About 45 % of the sampled points are left out as "near a
  surface".

The surface rule was chosen after looking at the XCT values of all four
parts, including test part 4.

![XCT values per part](docs/images/full_xct_histogram.png)

What can be measured is the rank correlation between the anomaly score and
the XCT value, on the 47 838 test segments away from the surface:

| Method | Rank correlation |
|---|---:|
| Fixed limits (tuned k) | −0.100 |
| Fixed limits, level features only (tuned k) | −0.143 |
| Isolation Forest | −0.155 ± 0.002 |
| Local Outlier Factor | −0.091 ± 0.004 |
| One-Class SVM | −0.000 ± 0.007 |
| Conv. autoencoder | −0.048 ± 0.001 |

The link is weak. Most of it comes from slow changes up the part, not from
single spots (see [the report](docs/report.md#5-what-the-results-mean)).

### What the results say

- **Fixed limits with all features are a strong baseline.** With the alarm
  level from part 3, tuned fixed limits catch 65 % of the faulty segments
  at 1.4 % false rejections.
- **Local Outlier Factor and One-Class SVM are somewhat better at low alarm
  rates.** When every method flags exactly 1 % of normal test segments,
  they catch 67 % and 66 %, against 61 % for the fixed limits; at 2 %:
  70 % and 72 % against 68 %. PR-AUC: 0.67 and 0.65 against 0.58.
- **At high recall they are about equal.** To catch 90 % of the faulty
  segments, Local Outlier Factor must flag 26.8 % of normal segments, the
  fixed limits 28.5 % and One-Class SVM 31.1 %.
- **The features matter as much as the method.** Fixed limits on mean, min
  and max only reach PR-AUC 0.37 and catch 49 % at 1 %.
- Isolation Forest (PR-AUC 0.42) and the autoencoder (0.30) are weaker than
  the fixed limits here. Slow drifts are the hardest kind for them
  (37 % and 8 % caught).
- The 3-sigma rule catches 83 % of the faulty segments, but it flags 10.9 %
  of the normal ones.
- The alarm level was set for 1 % on part 3. On part 4 the tuned methods
  flag 1.1 to 2.0 % of normal segments. So an alarm level carries over to
  a new part only roughly.
- Isolation Forest and One-Class SVM give many false rejections in the
  overhang region (19.4 % and 9.9 %). The tuned fixed limits, which are set
  per region, give 1.4 % there.
- Many alarms on the clean test part are in layer 1, the first layer on the
  build plate (271 of 3 828 flagged segments, from only 469 of 88 378
  segments). Its melt pool is bigger than in the layers above, and bigger
  than in layer 1 of the training parts. Layers 2 to 20 also get more
  alarms than the rest for most methods
  ([numbers](docs/report.md#4-results)).

The heatmap below shows layer 9, the layer with the most alarms above
layer 1 (Local Outlier Factor, 101 of 504 segments flagged). The tracks run
along y, from y ≈ −11 to −16 mm, and have 5 segments each. The rows of
alarms at y ≈ −14.4 and −15.4 mm are the last two segments of the tracks:
they hold 61 of the 101 alarms. Layer 1 is in
[`full_layer_heatmap_layer1.png`](docs/images/full_layer_heatmap_layer1.png).

![Layer 9 of the test part, Local Outlier Factor](docs/images/full_layer_heatmap.png)

### What the results do not show

- **One test part.** Every number comes from part 4 only. There is no
  spread over test parts.
- **Injected faults are synthetic.** They show that a method reacts to that
  kind of signal change. They are not real defects.
- **Nothing about real pores.** XCT gives no pore labels here (see above),
  so finding real defects is not tested.
- The overhang region is "different", not "defective".

**Earlier small check** (part 1, layers 120 to 140, split by blocks of
layers, made before the full run): see
[`docs/first_real_data_check.md`](docs/first_real_data_check.md).

Every flagged segment is listed in `traceability.csv` with part, layer,
x / y / z, time, the methods that flagged it and the XCT value (empty when
the segment is at a surface), so someone can go and look at that exact spot.

## What this project can and cannot claim

- It **can** compare methods on public research data from one machine, one
  material and one part shape.
- Injected faults are **synthetic**. Good numbers on them show that a method
  reacts to that kind of signal change. They do not prove it finds real defects.
- XCT labels are real, but lining up XCT with the laser position has an
  uncertainty of about one voxel (10 µm), and the pore threshold is our own
  choice. XCT is read only at the camera frames (about 12 % of each layer),
  and points near a surface are left out. At the remaining points no value
  is below the threshold, so small pores, pores between the sampled spots
  and pores near the surface cannot be seen.
- The overhang region is "different", not "defective".
- 10.7 % of all frames are in no segment and get no score: tracks shorter
  than 13 frames (short scan lines across thin walls, 3.2 % of all frames),
  the first frame of every longer track (1.6 %), and the frames left over at
  the end of a track when fewer than 12 remain (5.8 %).
- It does **not** process the raw melt pool video (NIST did that).
- It does **not** run in real time or control a printer.
- Results do **not** carry over to other machines, materials or factories
  without new data.

## Repository layout

```
├── configs/            settings for the sample run and the NIST runs
├── data/
│   ├── raw/            NIST downloads (not in Git)
│   └── sample/         small fake data for tests and the quick start
├── docs/
│   ├── data_notes.md   what is in the data, and what to check
│   ├── first_real_data_check.md   first run on 21 real layers
│   ├── images/         figures shown in the docs (full_* = full run)
│   └── report.md       short report with results and recommendations
├── notebooks/          01_explore, 02_results
├── scripts/            download_data, inspect_data, check_xct, run_pipeline,
│                       extra_numbers, make_sample_data
├── src/
│   ├── load.py         read the NIST files
│   ├── features.py     tracks, segments, features
│   ├── labels.py       geometry and XCT labels
│   ├── inject.py       synthetic faults
│   ├── models/         limits, iforest, lof (+ ocsvm), autoencoder
│   ├── evaluate.py     metrics
│   ├── plots.py        figures
│   ├── pipeline.py     the whole experiment
│   └── synthetic.py    makes the fake sample data
└── tests/              pytest, run in GitHub Actions
```

## Sources

- Z. Yang, Y. Lu, H. Yeung, B. Lane, N. Van Handel (2025). *A Fully Registered
  In-Situ and Ex-Situ Dataset for Metal Powder Bed Fusion Additive
  Manufacturing.* NIST AMS 100-69. https://doi.org/10.6028/NIST.AMS.100-69
  (data: NIST dataset `mds2-3761`)
- B. Lane, H. Yeung (2020). *Process Monitoring Dataset from the Additive
  Manufacturing Metrology Testbed (AMMT): Overhang Part X4.* J. Res. NIST 125.
  https://doi.org/10.6028/jres.125.027
- NIST AMMT process monitoring datasets: https://www.nist.gov/node/1560441

## License

Code: MIT. The NIST data has its own terms; see the NIST pages above.

Developed with AI assistance (Claude).
