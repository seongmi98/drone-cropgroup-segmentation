"""
background_only_sampling.py  (template / stub)

Negative-sample background labeling for crop-group–specialized models.

When training a crop-group model (e.g., cabbage-radish or onion-garlic),
crops that do NOT belong to the target group are relabeled as *background*
instead of being kept as separate classes. A controllable proportion of
"background-only" images (images that contain only non-target crops -> all
background after remapping) is then added to the training set.

This mirrors real deployment, where non-target crops from neighboring fields
appear in an image, and encourages conservative behavior (fewer false
positives). The background ratio is studied via ablation
(see results/background_ablation.csv and results/fp_evaluation.csv).

--------------------------------------------------------------------------
This is a scaffold. Fill in the I/O paths and integrate with your dataset
build pipeline (see make_master_5fold.py and remap_5to3class.py).
--------------------------------------------------------------------------
"""

import argparse


def build_background_only_split(
    target_json: str,
    background_pool_json: str,
    out_json: str,
    bg_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    """Add background-only (negative) images to a target-crop split.

    Args:
        target_json: COCO/semantic annotations for the target crop group.
        background_pool_json: pool of images whose target-group crops are
            absent (all classes -> background after remapping).
        out_json: output annotations with negatives mixed in.
        bg_ratio: number of background-only images added, as a fraction of
            the target-image count (e.g., 0.10 = 10%).
        seed: RNG seed for reproducible sampling.
    """
    raise NotImplementedError(
        "TODO: implement negative-sample mixing for your annotation format."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-json", required=True)
    ap.add_argument("--background-pool-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--bg-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_background_only_split(
        args.target_json,
        args.background_pool_json,
        args.out_json,
        args.bg_ratio,
        args.seed,
    )
