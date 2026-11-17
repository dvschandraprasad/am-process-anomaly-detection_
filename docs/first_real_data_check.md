# First check on real NIST data

**Date:** 16 November 2026
**Data:** NIST Overhang Part X4, registered dataset `mds2-3761`, **part 1, layers 120 to 140** (21 layer files, `L0120.csv` to `L0140.csv`, 81 001 rows).
**Config:** [`configs/nist_x4_layers_120_140.yaml`](../configs/nist_x4_layers_120_140.yaml)

This run answers one question: **does the code work on the real files?** Yes.
It is **not** the final result. It uses 21 of 1000 layer files, from one part.
The full run on all four parts is in the [README](../README.md#results) and
the [report](report.md).

> **Note (17 November 2026).** This check was first made with the code of
> 16 November. Two later fixes change its numbers a little: frames with the
> same time stamp no longer start a new track, and XCT labels now leave out
> all surfaces (not only side cells). The numbers and figures on this page
> are from a new run with the current code
> (`results/nist_x4_layers_120_140/`). The findings stay the same.

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
| Injected faults | 47 faults in the test layers, 99 faulty segments (9 % of test segments) |
| Alarm level | Set on the validation layers: 1 % of normal segments flagged |
| Seeds | 3 for methods with randomness; tables show mean ± std |

## Results: injected faults (test layers)

| Method | Precision | Recall | F1 | False rejection rate | PR-AUC | FRR at 90 % recall |
|---|---:|---:|---:|---:|---:|---:|
| Fixed limits (3-sigma rule) | 0.615 | 0.838 | 0.709 | 5.2 % | 0.757 | 11.1 % |
| Fixed limits (tuned k) | 0.857 | 0.606 | 0.710 | 1.0 % | 0.757 | 11.1 % |
| Isolation Forest | 0.829 ± 0.003 | 0.586 ± 0.030 | 0.686 ± 0.019 | 1.2 ± 0.1 % | 0.736 ± 0.019 | 17.6 ± 1.5 % |
| Local Outlier Factor | 0.878 | 0.727 | 0.796 | 1.0 % | 0.837 | 9.2 % |
| One-Class SVM | 0.829 | 0.687 | 0.751 | 1.4 % | 0.814 | 20.8 % |
| Conv. autoencoder | 0.736 ± 0.050 | 0.498 ± 0.037 | 0.594 ± 0.042 | 1.8 ± 0.3 % | 0.652 ± 0.043 | 49.1 ± 4.2 % |

Local Outlier Factor and One-Class SVM gave the same result with all 3
seeds, so no ± is shown for them.

![Precision-recall curves](images/real_layers_120_140_pr_curves.png)

### Share of injected faults caught, by kind

| Method | drift | drop | noise | spike |
|---|---:|---:|---:|---:|
| Fixed limits (3-sigma rule) | 100 % | 100 % | 100 % | 100 % |
| Fixed limits (tuned k) | 75 % | 100 % | 91 % | 55 % |
| Isolation Forest | 42 % | 81 % | 100 % | 85 % |
| Local Outlier Factor | 75 % | 100 % | 100 % | 91 % |
| One-Class SVM | 75 % | 100 % | 100 % | 91 % |
| Conv. autoencoder | 28 % | 64 % | 100 % | 67 % |

![Signal around one injected fault of each kind](images/real_layers_120_140_signal_examples.png)

The black line is the real melt pool area. The dips to zero are real: the
first camera frame of every track shows no melt pool.

### False rejections by region

| Method | Solid metal below | Overhang region |
|---|---:|---:|
| Fixed limits (3-sigma rule) | 4.8 % | 29.4 % |
| Fixed limits (tuned k) | 0.7 % | 17.6 % |
| Isolation Forest | 0.4 ± 0.0 % | 47.1 ± 4.8 % |
| Local Outlier Factor | 0.6 % | 23.5 % |
| One-Class SVM | 0.8 % | 35.3 % |
| Conv. autoencoder | 1.6 ± 0.4 % | 9.8 ± 2.8 % |

The overhang column rests on only 17 normal test segments, so these
percentages are rough.

## Results: clean test layers and X-ray CT

On the test layers without injected faults, 29 of 1 098 segments were flagged
by at least one tuned method (17 by one method, 4 by two, 6 by three, 2 by
four). 41 % of them lie in the overhang region.

![Layer 128](images/real_layers_120_140_layer_heatmap.png)

Layer 128, Local Outlier Factor. The three alarms sit on the edge at
y ≈ 13.5 mm, where this layer reaches out over powder.

**XCT pores.** With the default threshold (0.484), no test segment contains a
pore. The lowest XCT value in a test segment away from the surface is 0.513.
So precision and recall against XCT cannot be given for these layers.

**Anomaly score against XCT value** (Spearman rank correlation, 333 test
segments away from the surface; test layers 138 and 139 are close to the
last loaded layer, which counts as a top face, so they are left out):

| Method | Rank correlation |
|---|---:|
| Fixed limits (tuned k) | 0.009 |
| Isolation Forest | −0.035 ± 0.007 |
| Local Outlier Factor | −0.021 |
| One-Class SVM | −0.007 |
| Conv. autoencoder | 0.011 ± 0.006 |

All values are close to zero. In these layers there is no visible link between
how unusual the melt pool is and the XCT value.

## What this shows

- The loader reads the real files with no changes: 40 columns, no header,
  layer number from the file name.
- The alarm level carries over from validation to test layers: every tuned
  method flags 1.0 to 1.8 % of normal test segments (target 1 %).
- The 3-sigma rule catches the most faults, but it flags 5 % of normal
  segments. At about 1 % false rejections, Local Outlier Factor finds 73 % of
  the faulty segments and tuned fixed limits find 61 %.
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
