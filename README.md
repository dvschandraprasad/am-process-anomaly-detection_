# Anomaly detection in metal additive manufacturing process data

Learn what "normal" looks like in in-situ process monitoring data from metal
3D printing (laser powder bed fusion). Then find, compare and show the places
that are **not** normal, and check them against X-ray CT.

![tests](https://github.com/dvschandraprasad/am-process-anomaly-detection/actions/workflows/tests.yml/badge.svg)

> **Status.** The code runs on the real NIST files. So far it has been run on
> a small piece of the real data only: part 1, layers 120 to 140
> ([first real data check](docs/first_real_data_check.md)). The full run on
> all four parts is still to do. See [Results](#results).

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
| Fixed limits, 3-sigma rule | Rule (baseline) | Flag if any feature is outside mean ± 3 std. No tuning. |
| Fixed limits, tuned k | Rule (baseline) | Same, but k is set on validation data like for the other methods |
| Isolation Forest | Classic ML | Unusual points are easy to separate with random cuts |
| Local Outlier Factor | Classic ML | Unusual points sit in emptier areas than their neighbours |
| One-Class SVM | Classic ML | Draw a tight border around normal data |
| 1D conv. autoencoder | Deep learning | Learn to rebuild normal signal pieces; a bad rebuild = unusual |

### Rules for a fair comparison

- All methods use the same train / validation / test parts.
- Training uses **normal segments only** (segments with an XCT pore are removed).
- The alarm level of each method is set on the **validation part** so that 1 %
  of normal validation segments are flagged. The test part is never used for
  tuning. A test checks this (`tests/test_pipeline.py`).
- Methods with randomness run with 3 seeds; tables show mean ± std.

### The three kinds of labels

| Label | Meaning | Use |
|---|---|---|
| Injected faults | Spikes, drops, drifts and noise bursts added to normal test signals. Place and size are known exactly. | Main numbers |
| XCT pores | The X-ray CT value at the segment says "pore" | Check against real defects |
| Overhang region | Less than 0.5 mm solid metal below the segment | Shows where false rejections come from. **Not** a defect. |

## Quick start (fake sample data, about 15 seconds)

```bash
git clone https://github.com/dvschandraprasad/am-process-anomaly-detection.git
cd am-process-anomaly-detection
python -m venv .venv && source .venv/bin/activate
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

# 3. Download the rest and do the real run (split by part)
python scripts/download_data.py --parts 2 3 4
python scripts/run_pipeline.py --config configs/nist_x4.yaml
```

Before step 2, read the output of `inspect_data.py` and check the open points
in [`docs/data_notes.md`](docs/data_notes.md), most of all the XCT pore
threshold.

Then look at the data yourself in `notebooks/01_explore.ipynb` and at the
results in `notebooks/02_results.ipynb`.

## Results

**First check on real data** (part 1, layers 120 to 140, split by blocks of
layers): see [`docs/first_real_data_check.md`](docs/first_real_data_check.md).
It shows that the code works on the real files. It is too small to be the
final result.

**To do: run on all four parts and fill in this section.**

1. Run `configs/nist_x4.yaml` (see above).
2. Paste the tables from `results/nist_x4/results.md` here.
3. Copy `pr_curves.png`, `layer_heatmap.png` and `frr_by_region.png` from
   `results/nist_x4/figures/` to `docs/images/` and show them here.
4. Write 3 to 5 sentences: which method had the lowest false rejection rate,
   which faults were missed, and how well the alarms match the XCT pores.

Every flagged segment is listed in `traceability.csv` with part, layer,
x / y / z, time, the methods that flagged it and the XCT value, so someone can
go and look at that exact spot.

## What this project can and cannot claim

- It **can** compare methods on public research data from one machine, one
  material and one part shape.
- Injected faults are **synthetic**. Good numbers on them show that a method
  reacts to that kind of signal change. They do not prove it finds real defects.
- XCT labels are real, but pores are rare and tiny, and lining up XCT with
  the laser position has an uncertainty of about one voxel (10 µm). The pore
  threshold is our own choice.
- The overhang region is "different", not "defective".
- Tracks shorter than 13 frames get no score. In the real data these are the
  short scan lines across thin walls (about 6 % of all frames in layers 120 to 140).
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
│   └── report.md       short report with recommendations
├── notebooks/          01_explore, 02_results
├── scripts/            download_data, inspect_data, run_pipeline, make_sample_data
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
