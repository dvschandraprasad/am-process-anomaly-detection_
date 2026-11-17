# Anomaly detection in melt pool monitoring data: short report

> **Draft.** Sections 1–3 and 6 describe what the code does. Sections 4, 5
> and 7 need the results of the run on the NIST data. Everything marked
> **TODO** must be filled in or checked before this report is shared.

## 1. Goal

On a production line, measurements are often checked against fixed limits.
This works when "normal" is the same everywhere. In laser powder bed fusion it
is not: the melt pool behaves differently over solid metal and over loose
powder (overhangs), and differently for contour and infill scans.

Question: **can a model that learns normal behaviour find real problems with
fewer false rejections than fixed limits?**

## 2. Data

NIST "Overhang Part X4", fully registered dataset (`mds2-3761`, NIST AMS 100-69).

- 4 nominally identical parts in nickel alloy 625, 250 layers of 20 µm each.
- A high-speed camera on the laser path recorded the melt pool at 10 000
  frames per second. NIST extracted melt pool length, width and area.
- Each frame is registered with laser position and power, the layer camera
  and the X-ray CT (XCT) value of the finished part.

Signals used: melt pool length, width and area (image threshold 100).

TODO: number of rows and segments per part, after the run (`run_info.json`).

## 3. Method

**Unit of analysis.** Each laser track is cut into segments of 12 frames
(1.2 ms, about 1 mm of travel). Each segment gets one anomaly score.

**Features.** Per segment and signal: mean, standard deviation, minimum,
maximum, slope and jitter (mean change from frame to frame). Plus three things
known before printing: the laser setting, whether the segment is the start of
a track (the melt pool needs a few frames to grow), and how much solid metal
is below the segment.

**Split.** Parts 1–2 for training, part 3 for validation, part 4 for testing.
Whole parts are kept together, so the test part is truly unseen.

**Training.** Only normal segments (no XCT pore) are used for training.

**Methods.** Fixed limits (3-sigma rule, and with tuned k), Isolation Forest,
Local Outlier Factor, One-Class SVM, and a small 1D convolutional autoencoder
on the raw frames.

**Alarm level.** For every method the alarm level is set on the validation
part so that 1 % of normal segments are flagged. Only the 3-sigma rule keeps
its fixed level. The test part is never used for tuning.

**Labels.**
1. *Injected faults* in the test part: spikes, drops, drifts and noise bursts
   with known place and size.
2. *XCT pores:* segments where the XCT value is beyond the pore threshold.
   Segments at the part surface are left out, because XCT there mixes metal
   and air.
3. *Overhang region:* less than 0.5 mm solid metal below. This is not a
   defect; it is used to see where false rejections come from.

**Metrics.** Precision, recall, F1, false rejection rate (normal segments
flagged ÷ all normal segments), PR-AUC, share of faults caught per fault kind,
and run time per layer.

## 4. Results

TODO: paste the tables from `results/nist_x4/results.md`.

TODO: add `pr_curves.png`, `layer_heatmap.png`, `frr_by_region.png`.

## 5. What the results mean

TODO: answer these with numbers from section 4.

- Which method has the lowest false rejection rate on the test part? Is it
  close to the 1 % it was set to on the validation part? (If not, the parts
  differ, and an alarm level does not carry over from part to part.)
- How much better than the 3-sigma rule is the best method, at the same recall?
- Which fault kinds are caught, and which are missed? (Slow drifts are usually
  the hardest for methods that look at short segments.)
- Where do the false rejections sit: in the bulk or in the overhang region?
- Do the alarms on the clean test part match the XCT pores better than chance?
  Look at PR-AUC next to the share of pore segments.
- Open `traceability.csv` and look at the top 10 rows on the layer heatmap.
  Do they make physical sense?

## 6. Limits of this work

- One machine, one material, one part shape, four parts.
- Injected faults are synthetic. They test if a method reacts to a kind of
  signal change, not if it finds real defects.
- XCT pore labels depend on a threshold that we chose. Pores are rare, so the
  XCT numbers have wide uncertainty.
- Melt pool sizes come from image brightness and are affected by the camera
  settings and motion blur.
- Segments are short (1.2 ms). Problems that build up over many tracks or
  layers are not looked at.
- Tracks shorter than 13 frames (short scan lines across thin walls) get no score.
- Nothing here runs in real time.

## 7. Recommendations

TODO: keep only the points that your results support, and add numbers.

- **Set limits per region and per laser setting**, not one limit for the whole
  part. Check in the results how much this lowers false rejections.
- **Choose the alarm level from a false rejection budget** (for example 1 %)
  on parts known to be good, and re-check it on new parts.
- **Keep the fixed limits as a baseline next to any learned model.** A learned
  model should only replace them where it is clearly better at the same false
  rejection rate.
- **Keep every alarm traceable** to part, layer and position, so that it can
  be checked against inspection (here: XCT).
- **Before using this in production:** collect labelled data from the real
  process, test on more parts and shapes, watch for drift of "normal" over
  time, and measure the run time on the real data rate.
