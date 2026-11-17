# First check on real NIST data

**Date:** 16 November 2026
**Data:** NIST Overhang Part X4, registered dataset `mds2-3761`, **part 1, layers 120 to 140** (21 layer files, `L0120.csv` to `L0140.csv`, 81 001 rows).
**Config:** [`configs/nist_x4_layers_120_140.yaml`](../configs/nist_x4_layers_120_140.yaml)

This run answers one question: **does the code work on the real files?** Yes.
It is **not** the final result. It uses 21 of 1000 layer files, from one part.
The full run on all four parts is in the [README](../README.md#results) and
the [report](report.md).

> **Note (17 November 2026).** This check was first made with the code of
> 16 November. Later changes move its numbers: frames with the same time
> stamp no longer start a new track, XCT labels leave out all surfaces, the
> fixed limits now look at all six numbers of each signal (the old version
> is kept as "level features only"), the commanded scan speed replaces the
> measured one as context, fault sizes come from the training layers, and
> segments with only 1 or 2 fault frames are left out. The numbers and
> figures on this page are from a new run with the current code
> (`results/nist_x4_layers_120_140/`).

## How to repeat it

```bash
python scripts/download_data.py --parts 1
python scripts/run_pipeline.py --config configs/nist_x4_layers_120_140.yaml
```

## Setup

| | |
|---|---|
| Split | Blocks of 2 layers: train, train, train, validation, test, and so on |
| Train layers | 120–125, 130–135, 140 (13 layers, 3 572 segments) |
| Validation layers | 126, 127, 136, 137 (1 098 segments) |
| Test layers | 128, 129, 138, 139 (1 098 segments) |
| Segment | 12 camera frames in a row on one track (1.2 ms) |
| Injected faults | 47 faults in the test layers: 99 faulty and 988 normal segments; 11 segments with only 1 or 2 fault frames are left out |
| Alarm level | Set on the validation layers: 1 % of normal segments flagged |
| Seeds | 3 for methods with randomness; tables show mean ± std |

## Results: injected faults (test layers)

| Method | Precision | Recall | F1 | False rejection rate | PR-AUC | FRR at 90 % recall |
|---|---:|---:|---:|---:|---:|---:|
| Fixed limits (3-sigma rule) | 0.542 | 0.848 | 0.661 | 7.2 % | 0.804 | 12.0 % |
| Fixed limits (tuned k) | 0.883 | 0.687 | 0.773 | 0.9 % | 0.804 | 12.0 % |
| Fixed limits, level features only (tuned k) | 0.891 | 0.576 | 0.699 | 0.7 % | 0.782 | 13.6 % |
| Isolation Forest | 0.839 ± 0.020 | 0.545 ± 0.033 | 0.661 ± 0.030 | 1.0 ± 0.1 % | 0.710 ± 0.015 | 17.1 ± 0.6 % |
| Local Outlier Factor | 0.933 | 0.707 | 0.805 | 0.5 % | 0.857 | 12.2 % |
| One-Class SVM | 0.867 | 0.727 | 0.791 | 1.1 % | 0.831 | 25.0 % |
| Conv. autoencoder | 0.782 ± 0.041 | 0.475 ± 0.046 | 0.590 ± 0.046 | 1.3 ± 0.2 % | 0.663 ± 0.039 | 48.8 ± 4.4 % |

Local Outlier Factor and One-Class SVM gave the same result with all 3
seeds, so no ± is shown for them.

Recall when every method flags exactly 1 % of the normal test segments
(alarm level set on the test layers, for comparison only): tuned fixed
limits 0.717, level features only 0.747, Isolation Forest 0.532 ± 0.067,
Local Outlier Factor 0.758, One-Class SVM 0.717, autoencoder 0.465 ± 0.058.

![Precision-recall curves](images/real_layers_120_140_pr_curves.png)

### Share of injected faults caught, by kind

Over the 46 of 47 faults that have at least one labelled segment.

| Method | drift | drop | noise | spike |
|---|---:|---:|---:|---:|
| Fixed limits (3-sigma rule) | 100 % | 100 % | 100 % | 100 % |
| Fixed limits (tuned k) | 75 % | 100 % | 100 % | 100 % |
| Fixed limits, level features only (tuned k) | 67 % | 100 % | 91 % | 55 % |
| Isolation Forest | 31 % | 72 % | 100 % | 82 % |
| Local Outlier Factor | 75 % | 100 % | 100 % | 82 % |
| One-Class SVM | 83 % | 100 % | 100 % | 91 % |
| Conv. autoencoder | 28 % | 53 % | 100 % | 61 % |

![Signal around one caught injected fault of each kind](images/real_layers_120_140_signal_examples.png)

One caught fault of each kind, picked on purpose to show what the faults
look like. The black line is the real melt pool area. The dips to zero are
real: the first camera frame of every track shows no melt pool.

### False rejections by region

| Method | Solid metal below | Overhang region |
|---|---:|---:|
| Fixed limits (3-sigma rule) | 6.6 % | 41.2 % |
| Fixed limits (tuned k) | 0.5 % | 23.5 % |
| Fixed limits, level features only (tuned k) | 0.4 % | 17.6 % |
| Isolation Forest | 0.2 ± 0.1 % | 47.1 ± 0.0 % |
| Local Outlier Factor | 0.1 % | 23.5 % |
| One-Class SVM | 0.4 % | 41.2 % |
| Conv. autoencoder | 1.2 ± 0.2 % | 9.8 ± 2.8 % |

The overhang column rests on only 17 normal test segments, so these
percentages are rough.

## Results: clean test layers and X-ray CT

On the test layers without injected faults, 28 of 1 098 segments were flagged
by at least one of the five main methods (15 by one method, 4 by two, 6 by
three, 3 by four). 39 % of them lie in the overhang region.

![Layer 128](images/real_layers_120_140_layer_heatmap.png)

Layer 128, Local Outlier Factor. The three alarms sit on the edge at
y ≈ 13.5 mm, where this layer reaches out over powder.

**XCT pores.** With the default threshold (0.484), no test segment away from
the surface is below it. The lowest XCT value in such a segment is 0.513.
So precision and recall against XCT cannot be given for these layers.

**Anomaly score against XCT value** (Spearman rank correlation, 333 test
segments away from the surface; test layers 138 and 139 are close to the
last loaded layer, which counts as a top face, so they are left out):

| Method | Rank correlation |
|---|---:|
| Fixed limits (tuned k) | 0.005 |
| Fixed limits, level features only (tuned k) | 0.009 |
| Isolation Forest | −0.016 ± 0.008 |
| Local Outlier Factor | 0.001 |
| One-Class SVM | −0.006 |
| Conv. autoencoder | 0.011 ± 0.006 |

All values are close to zero. In these layers there is no visible link between
how unusual the melt pool is and the XCT value.

## What this shows

- The loader reads the real files with no changes: 40 columns, no header,
  layer number from the file name.
- The alarm level carries over from validation to test layers: every tuned
  method flags 0.5 to 1.3 % of normal test segments (target 1 %).
- The 3-sigma rule catches the most faults, but it flags 7.2 % of normal
  segments. At about 1 % false rejections, Local Outlier Factor finds 71 % of
  the faulty segments and tuned fixed limits find 69 %.
- Slow drifts are the hardest kind for most methods.
- The autoencoder is the weakest here. It had only 3 572 training segments.
- False rejections are much more common in the overhang region than over
  solid metal, for every method.

## What this does not show

- **It is a small piece of one part.** Numbers will change on the full data.
- **The split is by layer blocks, not by part.** Neighbouring layers look
  alike, so this is an easier test than testing on an unseen part.
- **Injected faults are synthetic.** They are not real defects.
- **The overhang region is under-counted here.** Below the first loaded layer
  (120) the code assumes solid metal. With all 250 layers loaded this is
  worked out properly.
- **Nothing can be said about real pores** from these 21 layers.
