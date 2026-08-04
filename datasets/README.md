# Data preparation

Scripts that turn raw COCO instance annotations into the training-ready semantic
segmentation format, then build the cross-validation split. **No dataset is
bundled** — paths in each script are placeholders (`/path/to/...`) to set for your
environment.

## Pipeline order

1. **`preprocessing/filter_coco_classes.py`** — select / reorder the target crop
   classes from the raw COCO annotations.
2. **`preprocessing/merge_coco_json.py`** — merge per-crop / per-batch COCO JSON
   files into one.
3. **`preprocessing/undersample.py`** — class-balanced undersampling.
4. **`preprocessing/instance_to_panoptic.py`** — instance masks → panoptic masks.
5. **`preprocessing/instance_json_to_panoptic_json.py`** — instance JSON →
   panoptic JSON.
6. **`preprocessing/panoptic_to_semantic.py`** — panoptic → semantic-segmentation
   annotations.
7. **`remap_5to3class.py`** — remap classes and enforce background = 0
   (used to build crop-group–specialized label sets).
8. **`make_master_5fold.py`** — build the master k-fold split
   (group-aware, altitude/region/year-balanced).
9. **`background_only_sampling.py`** — add negative (background-only) images for
   crop-group models (template).

## QC / inspection

- **`preprocessing/visualize_annotations.py`** — overlay annotations on images.
- **`preprocessing/analyze_class_distribution.py`** — per-class pixel/instance
  statistics across splits.
