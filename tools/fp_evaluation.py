"""
fp_evaluation.py  (template / stub)

False-positive evaluation for the negative-sampling (background ratio) study.

Metrics (see results/fp_evaluation.csv):
  - predicted_target_pixel_ratio : fraction of pixels predicted as a target
        crop on images that contain NO target crop (ideal = 0).
  - FP@1% / FP@5% : fraction of no-target images for which the predicted
        target-crop pixel ratio exceeds 1% / 5% (ideal = 0).

Lower is better for all three. These quantify how often a model
hallucinates a target crop on imagery where none is present -- the failure
mode that negative-sample background training is designed to suppress.

--------------------------------------------------------------------------
This is a scaffold. Wire it to your inference outputs (predicted masks) and
the list of "no-target" evaluation images.
--------------------------------------------------------------------------
"""

import argparse


def evaluate_false_positives(
    pred_dir: str,
    no_target_image_list: str,
    target_class_ids=(1, 2),
    thresholds=(0.01, 0.05),
):
    """Compute predicted_target_pixel_ratio and FP@k on no-target images.

    Args:
        pred_dir: directory of predicted semantic masks.
        no_target_image_list: file listing images with no target crop.
        target_class_ids: class ids counted as "target crop".
        thresholds: pixel-ratio thresholds for FP@k (e.g., 1%, 5%).

    Returns:
        dict with predicted_target_pixel_ratio and FP@k values.
    """
    raise NotImplementedError(
        "TODO: implement FP computation over your prediction outputs."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--no-target-image-list", required=True)
    args = ap.parse_args()
    print(evaluate_false_positives(args.pred_dir, args.no_target_image_list))
