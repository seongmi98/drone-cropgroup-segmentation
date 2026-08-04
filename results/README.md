# Results

This repository is shared as a **code / methodology portfolio**. The accompanying
manuscript is not yet published, so **quantitative results are withheld** here.

Below is *what* was measured (the experimental design and metrics), without the values.

## Experiments

| Experiment | Question |
|-----------|----------|
| Architecture comparison | Mask2Former vs SegNeXt vs DeepLabV3+ (5-class, full altitude) |
| Background-ratio ablation | How much negative (background-only) data to add for crop-group models |
| False-positive evaluation | Effect of negative sampling on hallucinated target-crop predictions |
| Altitude-stratified experiment | Full-altitude vs low / mid / high specialized models per crop group |

## Metrics

- **mIoU / mACC** — mean intersection-over-union and mean accuracy.
- **Per-class IoU** — background + each crop.
- **predicted-target-pixel-ratio, FP@1%, FP@5%** — false-positive rate on images
  containing no target crop (lower is better).

All experiments use **k-fold cross-validation**, reported as fold-wise mean ± std.

> Numerical tables and result figures will be added once the paper is published.
