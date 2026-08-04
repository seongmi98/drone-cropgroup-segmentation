# Methodology

## Problem setting

Semantic segmentation of four upland crops (cabbage, radish, onion, garlic)
plus background from UAV RGB imagery, under two field realities:

1. **Regional separation of crops.** Cabbage/radish and onion/garlic are grown
   in different regions, so a single all-in-one model is not the most realistic
   deployment configuration.
2. **Altitude-dependent resolution.** Imagery is captured across a range of
   flight altitudes; ground sampling distance changes substantially between
   low- and high-altitude flights.

## Model configurations

- **Baseline (unified):** one 5-class model over the full altitude range.
- **Crop-group specialized:** separate models for **cabbage·radish** and
  **onion·garlic** (reflecting regional separation).
- **Altitude-stratified:** for each crop group, models trained on low / mid /
  high altitude bands versus a full-altitude model.

## Negative-sample background labeling

For a crop-group model, non-target crops are relabeled as **background** rather
than kept as separate classes, and a controllable fraction of background-only
(negative) images is added to training. This suppresses false positives on
imagery containing only non-target crops — the behavior desired for
compliance-oriented monitoring. The background ratio is chosen by ablation.

## Architecture comparison

Mask2Former (Swin-T) is compared against DeepLabV3+ and SegNeXt as reference
baselines in the 5-class, full-altitude setting.

## Evaluation protocol

- **k-fold cross-validation**, reported as fold-wise mean ± standard deviation.
- **Realistic generalization splits** (separating train/test material by time
  and/or region) rather than a purely random split.
- **Metrics:** mIoU, mACC, per-class IoU, plus false-positive analysis
  (predicted-target-pixel-ratio, FP@1%, FP@5%) for the negative-sampling study.

## Results

Quantitative results are **withheld pending publication** of the accompanying
manuscript. See [`../results/README.md`](../results/README.md) for the list of
experiments and metrics that were measured.
