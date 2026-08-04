# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import random
# fmt: on
import json
import tempfile
import time
import warnings
import cv2
import numpy as np
import tqdm
import datetime
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from Mask2Former.mask2former import add_maskformer2_config
from Mask2Former.demo.predictor import VisualizationDemo
from detectron2.data.datasets import register_coco_instances
from PIL import Image

# base = "/path/to/code"
base = "../"
test_json_path = "/path/to/dataset/data/medium_onga/json/real/train1.json"  # coco json format
images_folder = "/path/to/dataset/data/medium_onga/images/train1"
IMG_SIZE = 1024

from detectron2.data.datasets import load_sem_seg
IMAGE_ROOT = "/path/to/dataset/data/medium_onga/images"
GT_ROOT = "/path/to/dataset/data/medium_onga/annotations/sem_seg3"
DatasetCatalog.register(
    "semantic_test",
    lambda: load_sem_seg(os.path.join(GT_ROOT, "train1"), os.path.join(IMAGE_ROOT, "train1"),
                         gt_ext="png", image_ext="JPG")
)
MetadataCatalog.get("semantic_test").set(
    # stuff_classes=["Background", "Cabbage", "Radish"],
    # stuff_colors=[[0, 0, 0], [9, 224, 188], [222, 11, 194]],
    stuff_classes=["background", "Onion", "Garlic"], 
    stuff_colors=[[0, 0, 0], [190, 222, 11], [224, 160, 9]],
    evaluator_type="sem_seg",
    ignore_label=255
)
def _colorize_sem_seg(mask: np.ndarray, stuff_colors: list, ignore_label: int = 255) -> np.ndarray:
    """Semantic mask [H,W] -> BGR 시각화 이미지 [H,W,3]. ignore_label은 검정."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in enumerate(stuff_colors):
        if cid > 255:
            continue
        out[mask == cid] = [color[2], color[1], color[0]]  # RGB -> BGR
    out[mask == ignore_label] = [0, 0, 0]
    return out


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    num_classes = len(MetadataCatalog.get("semantic_test").stuff_classes)
    cfg.DATASETS.TEST = ("semantic_test",)
    # cfg.DATASETS.TEST = ("semantic_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    # cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.2
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    # change this part
    if os.path.exists(test_json_path):
        with open(test_json_path, "r") as f:
            test_data = json.load(f)
        test_image_paths = [os.path.join(images_folder, img["file_name"]) for img in test_data["images"]]
        args.input = test_image_paths
        # args.input = random.sample(test_image_paths, 20)  # 샘플링하려면 주석 해제

        # 단일 이미지만 테스트하려면 아래 주석 해제
        selected_image_names = [
            "/path/to/dataset/data/medium_onga/images/train1/230412_40m_1.JPG"
        ]
        args.input = [os.path.join(images_folder, name) for name in selected_image_names]

        print(f"Loaded {len(test_image_paths)} test image paths from {test_json_path}")
        print(f"Using {len(args.input)} images for evaluation")
    else:
        print(f"Test JSON file '{test_json_path}' not found. Using other inputs if provided.")
    
    # #validation
    # if os.path.exists(val_json_path):
    #     with open(val_json_path, "r") as f:
    #         val_data = json.load(f)
    #     val_image_paths = [os.path.join(images_folder, img["file_name"]) for img in val_data["images"]]
    #     args.input = val_image_paths
    #     print(f"Loaded {len(val_image_paths)} validation image paths from {val_json_path}")
    # else:
    #     print(f"Validation JSON file '{val_json_path}' not found. Using other inputs if provided.")


    now = datetime.datetime.now()
    folder_name = f'test_result_{now.strftime("%Y%m%d_%H%M%S")}'
    # folder_name = f'val_result_{now.strftime("%Y%m%d_%H%M%S")}'
    output_path = os.path.join(args.output, folder_name)
    os.makedirs(output_path, exist_ok=True)
    args.output = output_path
    print(f'args.output: {args.output}')

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            pred_pil = visualized_output.get_image()
            pred_img = cv2.cvtColor(np.array(pred_pil), cv2.COLOR_RGB2BGR)

            # 왼쪽 패널: GT 마스크가 있으면 색상 시각화로 표시, 없으면 입력 이미지
            meta = MetadataCatalog.get("semantic_test")
            gt_split = os.path.basename(os.path.normpath(images_folder))  # e.g. train1
            gt_mask_path = os.path.join(
                GT_ROOT, gt_split,
                os.path.basename(path).replace(".JPG", ".png").replace(".jpg", ".png")
            )
            if os.path.isfile(gt_mask_path):
                gt_mask = np.array(Image.open(gt_mask_path), dtype=np.int64)
                if gt_mask.shape[:2] != (IMG_SIZE, IMG_SIZE):
                    gt_mask = cv2.resize(
                        gt_mask.astype(np.uint8),
                        (IMG_SIZE, IMG_SIZE),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.int64)
                left_img = _colorize_sem_seg(
                    gt_mask,
                    meta.stuff_colors,
                    getattr(meta, "ignore_label", 255),
                )
                left_label = "Ground Truth"
            else:
                left_img = img.copy()
                left_label = "Input"
            cv2.putText(
                left_img,
                left_label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            combined_image = np.hstack([left_img, pred_img])

            # scale_factor = 0.3
            # new_width = int(combined_image.shape[1] * scale_factor)
            # new_height = int(combined_image.shape[0] * scale_factor)
            # combined_image = cv2.resize(combined_image, (new_width, new_height))

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                # visualized_output.save(out_filename)
                cv2.imwrite(out_filename, combined_image)
            else:
                assert False, "Please specify a directory with args.output"
