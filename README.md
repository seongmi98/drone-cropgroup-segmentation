# Drone Crop-Group Semantic Segmentation

Multi-class upland-crop semantic segmentation from UAV (drone) RGB imagery using
**Mask2Former (Swin-T)**. The study compares a single unified model against
**crop-group–specialized models** and examines **altitude-stratified training**,
targeting a practical field-monitoring scenario in which crop classes are
geographically separated and image resolution varies with flight altitude.

> **⚠️ Code / methodology portfolio — data & results withheld**
> This repository is shared as a **code and methodology showcase**. The accompanying
> manuscript is **not yet published**.
> - **The dataset is NOT public and is not distributed here.** Do not request or redistribute it.
> - **No trained weights or checkpoints are included.**
> - **Quantitative results and result figures are withheld** pending publication
>   (see `results/README.md` and `figures/README.md` for what was measured).

---

## Overview

Reliable, automated crop mapping from drone imagery can support agricultural monitoring
(e.g., compliance inspection for direct-payment programs). This project segments four
upland crops — **cabbage, radish, onion, garlic** (plus background) — under two field
realities: crops are grown in different regions, and image resolution depends on flight
altitude. It therefore compares a unified model with crop-group–specialized models, and
studies altitude specialization.

See [`docs/methodology.md`](docs/methodology.md) for the full method description.

## Repository structure

```
drone-cropgroup-segmentation/
├── README.md
├── LICENSE                       # MIT (code only; data/weights excluded)
├── requirements.txt              # torch + detectron2/mask2former (from source)
├── .gitignore                    # excludes data, weights, outputs
├── configs/                      # model training configs
│   ├── mask2former_swint.yaml
│   ├── segnext.yaml              # placeholder (MMSegmentation baseline)
│   └── deeplabv3plus.yaml
├── datasets/                     # data PREPARATION scripts only (no real data)
│   ├── README.md                 # full pipeline order
│   ├── make_master_5fold.py      # build master k-fold split
│   ├── remap_5to3class.py        # class remap / keep target crops, bg=0
│   ├── background_only_sampling.py  # negative-sample background labeling (template)
│   └── preprocessing/            # raw COCO -> panoptic -> semantic + QC
│       ├── filter_coco_classes.py
│       ├── merge_coco_json.py
│       ├── undersample.py
│       ├── instance_to_panoptic.py
│       ├── instance_json_to_panoptic_json.py
│       ├── panoptic_to_semantic.py
│       ├── visualize_annotations.py
│       └── analyze_class_distribution.py
├── tools/
│   ├── train_net.py              # Mask2Former training (Detectron2)
│   ├── eval_semantic.py          # semantic-segmentation evaluation
│   └── fp_evaluation.py          # predicted-target-pixel-ratio, FP@1/5% (template)
├── results/                      # experiment/metric description (values withheld)
│   └── README.md
├── figures/                      # figures withheld pending publication
│   └── README.md
└── docs/
    └── methodology.md
```

## Method at a glance

- **Task:** per-pixel semantic segmentation — background + cabbage, radish, onion, garlic.
- **Model:** Mask2Former with a Swin-Tiny backbone (Detectron2). DeepLabV3+ and SegNeXt
  serve as reference baselines.
- **Configurations:** unified 5-class model vs. crop-group–specialized models
  (cabbage·radish / onion·garlic), each with altitude-band variants (low / mid / high /
  all).
- **Negative-sample background labeling:** non-target crops are relabeled as background;
  the background ratio is tuned by ablation to suppress false positives.
- **Evaluation:** k-fold cross-validation (mean ± std), realistic time/region splits,
  metrics mIoU / mACC / per-class IoU plus FP analysis.

## Installation

1. Install PyTorch (CUDA build for your machine) — https://pytorch.org
2. Install Detectron2 and build Mask2Former's custom ops:
   ```bash
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   # then build Mask2Former ops:
   cd Mask2Former/mask2former/modeling/pixel_decoder/ops && sh make.sh
   ```
3. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

See https://github.com/facebookresearch/Mask2Former for framework details.

## Usage

> Dataset paths and roots are environment-specific and must be set for your machine.
> The scripts here are templates/entry points; no dataset is bundled.

```bash
# 1) Prepare data: raw COCO -> semantic annotations -> k-fold split
#    (full step-by-step order in datasets/README.md)
python datasets/preprocessing/filter_coco_classes.py ...
python datasets/preprocessing/merge_coco_json.py
# ... instance_to_panoptic -> panoptic_to_semantic ...
python datasets/remap_5to3class.py
python datasets/make_master_5fold.py
python datasets/background_only_sampling.py --bg-ratio 0.10 ...

# 2) Train
python tools/train_net.py --num-gpus 1 \
  --config-file configs/mask2former_swint.yaml

# 3) Evaluate
python tools/eval_semantic.py --config-file configs/mask2former_swint.yaml ...
python tools/fp_evaluation.py --pred-dir ... --no-target-image-list ...
```

## Results

Quantitative results are **withheld pending publication**. See
[`results/README.md`](results/README.md) for the experiments and metrics that were
evaluated (architecture comparison, background-ratio ablation, false-positive
evaluation, altitude-stratified experiment; metrics mIoU / mACC / per-class IoU / FP@k).

## Citation

A citation entry will be added once the accompanying manuscript is published.

## License

Source code is released under the [MIT License](LICENSE). The dataset and any model
weights are **not** included and **not** licensed for redistribution.
