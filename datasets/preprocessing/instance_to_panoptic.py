"""
This script converts a COCO-style instance segmentation JSON into panoptic PNG annotations
suitable for Detectron2/Mask2Former. The output PNG images are 3-channel RGB images whose colors encode
a panoptic id using the following scheme:

panoptic_id = category_id * 1000 + instance_counter
R = (panoptic_id >> 16) & 255
G = dwwwwwwwwwwdddddddddddw(panoptic_id >> 8) & 255
B = panoptic_id & 255

Usage:
python convert_to_panoptic_detectron2.py --json path/to/your.json --output-dir path/to/panoptic_test2017/

## 터미널 예시
python3 /path/to/code/dataset_converter_instance2panoptic.py \
  --input-dir  /path/to/dataset/data2/all_onga2/json/real \
  --output-dir /path/to/dataset/data2/all_onga2/annotations/panoptic

없으면 기본 인자 사용용
"""

import os
import json
import argparse
import numpy as np
import cv2
from collections import defaultdict

# 기본값(아무 인자 없이 실행 시 사용). 여러 JSON을 한 번에 처리하려면 --input-dir 사용.
DEFAULT_INPUT_JSON = "/path/to/dataset/data2/all/json/real/train1.json"
DEFAULT_OUTPUT_DIR = "/path/to/dataset/data2/all/annotations/panoptic/train1"

def id_to_rgb(panoptic_id):
    """
    Encode an integer panoptic_id into an RGB tuple using bit-shifting.
    """
    r = (panoptic_id >> 16) & 255
    g = (panoptic_id >> 8) & 255
    b = panoptic_id & 255
    return (r, g, b)

def generate_panoptic_annotations(json_file, output_dir):
    # Load the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # Group annotations by image_id
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image
    for image_info in images:
        image_id = image_info["id"]
        file_name = image_info["file_name"]  # e.g., "images/78893af1-dehydration_roi_DSC04530.JPG"
        width = image_info["width"]
        height = image_info["height"]

        # Create an empty RGB image; Detectron2 expects 8-bit PNGs.
        panoptic_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Maintain per-category instance counters to compute unique instance numbers.
        instance_counters = {}

        anns = anns_by_image.get(image_id, [])
        for ann in anns:
            cat_id = ann["category_id"]
            # Increase counter for the category in this image
            instance = instance_counters.get(cat_id, 0) + 1
            instance_counters[cat_id] = instance

            # Compute panoptic id as per a common convention.
            panoptic_id = int(cat_id) * 1000 + instance
            print(f"panoptic_id: {panoptic_id}")
            color = id_to_rgb(panoptic_id)

            segm = ann["segmentation"]
            if not segm:
                continue  # skip if no segmentation available

            # The segmentation is given as a list of polygons (list of lists)
            # Iterate over all polygons in this annotation
            for poly in segm:
                pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                # Fill the polygon with the computed color.
                cv2.fillPoly(panoptic_img, [pts], color)
        
        #Save the resulting panoptic PNG.
        #Use the base filename (without directories) and change extension to .png.
        base = os.path.basename(file_name)
        out_name = os.path.splitext(base)[0] + ".png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, panoptic_img)
        print("Saved panoptic annotation:", out_path)
        # # 추가
        # image_path = image_info["path"]
        # last_folder = os.path.basename(os.path.dirname(image_path))  # 예: '양파'
        # png_name = os.path.splitext(file_name)[0] + ".png"
        # out_path = os.path.join(output_dir, last_folder, png_name)

        # os.makedirs(os.path.dirname(out_path), exist_ok=True)  # 폴더 없으면 생성
        # cv2.imwrite(out_path, panoptic_img)
        # print("Saved panoptic annotation:", out_path)



def _iter_json_files(input_dir: str):
    files = []
    for fn in os.listdir(input_dir):
        if fn.lower().endswith(".json"):
            files.append(os.path.join(input_dir, fn))
    return sorted(files)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def main():
    p = argparse.ArgumentParser(description="COCO instance JSON -> panoptic PNG (단일/폴더 지원)")
    p.add_argument("--input-json", default=None, help="입력 JSON 파일(단일)")
    p.add_argument("--input-dir", default=None, help="입력 폴더(.json 여러 개 처리)")
    p.add_argument("--output-dir", default=None, help="출력 폴더(단일이면 그대로, 폴더 모드면 JSON별 하위폴더 생성)")
    args = p.parse_args()

    input_json = args.input_json or DEFAULT_INPUT_JSON
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR

    if args.input_dir:
        input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
        if not os.path.isdir(input_dir):
            raise SystemExit(f"--input-dir 폴더가 없습니다: {input_dir}")
        out_root = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(out_root, exist_ok=True)

        json_files = _iter_json_files(input_dir)
        if not json_files:
            print(f"폴더에 .json 파일이 없습니다: {input_dir}")
            return

        print(f"폴더 모드: {input_dir} (총 {len(json_files)}개)")
        print(f"출력 루트: {out_root}")
        for i, jf in enumerate(json_files):
            # 파일별로 하위 폴더를 만들어 섞이지 않게 저장 (동일 파일명 충돌 방지)
            out_subdir = os.path.join(out_root, _stem(jf))
            print(f"\n--- [{i+1}/{len(json_files)}] {os.path.basename(jf)} -> {out_subdir} ---")
            generate_panoptic_annotations(jf, out_subdir)
        return

    # 단일 파일 모드
    input_json = os.path.abspath(os.path.expanduser(input_json))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    if not os.path.isfile(input_json):
        raise SystemExit(f"입력 JSON 파일이 없습니다: {input_json}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"단일 모드: {input_json}")
    print(f"출력 폴더: {output_dir}")
    generate_panoptic_annotations(input_json, output_dir)

if __name__ == "__main__":
    main()