# Multi-Class Upland Crop Semantic Segmentation from Drone RGB Imagery

Semantic segmentation of multiple upland field crops from UAV (drone) RGB imagery using
**Mask2Former (Swin-T backbone)**. The work targets a practical field-monitoring scenario in
which crop classes are geographically separated and image resolution varies with flight
altitude.

> **⚠️ Notice — data & results not included**
> This repository accompanies a manuscript that is **currently under review / not yet published**.
> - **The dataset is NOT public and is not distributed here.** Do not request or redistribute it.
> - **No quantitative results, trained weights, or checkpoints are included** in this repository.
> - Only source code and a high-level description of the approach are provided.
> Quantitative results and dataset details will follow the paper's publication and its data-release policy.

---

## Overview

Reliable, automated crop mapping from drone imagery can support agricultural monitoring tasks
such as compliance inspection for direct-payment programs. This project studies how to build
practical segmentation models for four upland crops — **cabbage, radish, onion, and garlic**
(plus background) — under two realities of field deployment:

1. **Crops are regionally separated.** In practice, cabbage/radish and onion/garlic are grown in
   different regions, so a single all-in-one model is not the most realistic configuration.
2. **Resolution depends on flight altitude.** Drone imagery is captured across a range of
   altitudes, and ground sampling distance changes substantially between low- and high-altitude
   flights.

The study therefore compares a single unified model against **crop-group–specialized models**,
and further examines the effect of **altitude-stratified training**.

## Task

- **Type:** Semantic segmentation (per-pixel classification)
- **Classes:** Background + 4 crops (Cabbage, Radish, Onion, Garlic)
- **Input:** UAV RGB orthoimagery / tiles captured at multiple altitudes
- **Primary model:** Mask2Former with a Swin-Tiny backbone (Detectron2)

## Methodology

### Model configurations

The experiments compare model scoping strategies rather than a single fixed model:

- **Baseline (unified):** one 5-class model (background + all four crops) trained over the full
  altitude range.
- **Crop-group specialized:** separate models for the two crop groups
  (**cabbage·radish** and **onion·garlic**), reflecting the regional separation of crops in the field.
- **Altitude-stratified variants:** for each crop group, models trained on specific altitude
  bands (low / mid / high) versus a full-altitude model, to study whether altitude specialization
  helps.

### Negative-sample background labeling

When training a crop-group–specialized model, crops that do not belong to that group are
**relabeled as background** rather than left as separate classes. This mirrors real deployment,
where non-target crops from neighboring fields may appear in an image, and encourages
conservative behavior (fewer false positives) — a desirable property for compliance-oriented
monitoring. The proportion of background-only (negative) training images is treated as a tunable
factor and studied via ablation.

### Architecture comparison

Alongside Mask2Former, conventional segmentation baselines (e.g., DeepLabV3+ and SegNeXt) are
included as reference points for the 5-class full-altitude setting.

### Evaluation protocol

- **Cross-validation:** k-fold cross-validation is used, and results are reported as fold-wise
  statistics (mean ± standard deviation) rather than a single run.
- **Realistic splits:** evaluation is designed to reflect operational generalization
  (e.g., separating training and test material by time and/or region) rather than a purely random
  split.
- **Metrics:** standard semantic-segmentation metrics — mean IoU (mIoU), mean accuracy (mACC),
  per-class IoU — plus a false-positive–oriented analysis for the negative-sampling study.

## Repository structure

> Scripts are organized by pipeline stage. Paths, dataset roots, and run commands are environment
> specific; see `DEV_NOTES.md` for the raw command history used during development.

### Data preparation (COCO / panoptic / semantic)
- `dataset_converter_instance2panoptic.py`, `dataset_convert_instancejson2panopticjson.py`,
  `dataset_prepare_coco_semantic_annos_from_panoptic_annos.py` — convert instance annotations
  into panoptic and then semantic (COCO) formats.
- `filter_coco_drop_classes.py`, `remap_semseg_keep4.py`, `remap_coco_category_ids.py` —
  select/remap classes and enforce background = 0.
- `split.py`, `copy_split_images.py`, `undersample.py` — train/val/test splitting, image
  organization, and class-balanced undersampling.
- `dataset_make_master_5fold.py` — build the master k-fold split.
- `고도기반json나누기.py` / `json_고도별통계.py` — altitude-based JSON partitioning and statistics.
- `analyze_class_distribution.py`, `check_labels.py`, `dataset_visualizer.py`, `json_image.py` —
  dataset inspection, label checks, and annotation overlay visualization.

### Training
- `train_net.py` — main Mask2Former training entry point (Detectron2).
- Altitude- and seed-specific training drivers (multi-seed / per-altitude variants).

### Evaluation & inference
- `evaluation_test_semantic.py`, `evaluation_test_semantic2.py`, `eval_mask2former.py` —
  semantic-segmentation evaluation.
- `predict.py`, `demo.py` — single-image / batch inference and prediction visualization.

### Ensembling (exploratory)
- `softmax_averaging*.py`, `meta_ensemble*.py`, `stacking.py` — softmax averaging and
  meta/stacking ensemble experiments.

## Environment

- **Framework:** [Mask2Former](https://github.com/facebookresearch/Mask2Former) on
  [Detectron2](https://github.com/facebookresearch/detectron2)
- **Backbone:** Swin-Tiny
- **Config base:** `configs/.../swin/maskformer2_swin_tiny_bs16_90k.yaml`

Follow the official Mask2Former / Detectron2 installation instructions, then place datasets and
configs according to your environment. Development command history is preserved in `DEV_NOTES.md`.

## Citation

A citation entry will be added here once the accompanying manuscript is published.

## License / usage

Source code in this repository is shared for research reference. The dataset and any model
weights are **not** included and are **not** to be redistributed.
