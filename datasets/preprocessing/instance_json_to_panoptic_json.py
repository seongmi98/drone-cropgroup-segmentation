"""
Convert instance segmentation JSON (COCO format) to a panoptic-style JSON file.
The output JSON groups annotations by image and assigns each segment a panoptic id,
computed as:
panoptic_id = category_id * 1000 + instance_counter
The expected panoptic PNG file (with the segmentation visualization) is assumed to
be generated separately (e.g., using a conversion script) and this JSON will refer
to that file in its "file_name" field.

사용 예)
단일 파일:
  python3 dataset_convert_instancejson2panopticjson.py \
    --input-json /path/to/input.json \
    --output-json /path/to/output.json

폴더 전체:
  python3 dataset_convert_instancejson2panopticjson.py \
    --input-dir /path/to/dataset/data2/all_onga1/json/real \
    --output-dir /path/to/dataset/data2/all_onga1/annotations
"""

import os
import json
import argparse
from collections import defaultdict
from typing import List, Optional

def convert_instance_to_panoptic(instance_json, output_png_ext=".png"):
    """
    Converts an instance segmentation JSON to a panoptic JSON format.
    Arguments:
    instance_json: (dict) the loaded instance JSON with keys "images", "annotations", "categories", etc.
    output_png_ext: (str) extension for panoptic annotation file names (default ".png")
    
    Returns: a dictionary with the panoptic JSON structure.
    """
    # Copy "info" and "licenses" if present; otherwise, use defaults.
    panoptic = {
        "info": instance_json.get("info", {
            "description": "Panoptic annotations",
            "version": "1.0",
            "year": 2025
        }),
        "licenses": instance_json.get("licenses", []),
    }

    # Use images from the instance JSON.
    images = instance_json["images"]
    panoptic["images"] = []
    # Build a mapping from image id to image info
    image_id_to_info = {}
    for im in images:
        # In COCO panoptic, the "file_name" in the images list often
        # is just the image file name. Here we leave it as is.
        image_id_to_info[im["id"]] = im
        panoptic["images"].append({
            "id": im["id"],
            "width": im["width"],
            "height": im["height"],
            # You might want to change this as needed; typically the image 
            # file (e.g., JPEG) is stored in a separate folder.
            "file_name": os.path.basename(im["file_name"])
        })

    # Group instance annotations by image_id.
    anns_by_image = defaultdict(list)
    for ann in instance_json["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    panoptic_annotations = []

    # For each image, create one panoptic annotation entry.
    for image_id, anns in anns_by_image.items():
        # Use the image's file name (base) modified to a .png.
        image_info = image_id_to_info[image_id]
        base_name = os.path.splitext(os.path.basename(image_info["file_name"]))[0]
        panoptic_png = base_name + output_png_ext  # e.g., "image123.png"
        
        segments_info = []
        # For a unique panoptic id, keep counter per category for this image.
        instance_counters = {}
        
        for ann in anns:
            cat_id = ann["category_id"]
            # Increase counter for this category for the current image.
            instance = instance_counters.get(cat_id, 0) + 1
            instance_counters[cat_id] = instance
            
            panoptic_id = int(cat_id) * 1000 + instance
            
            # Build segment info dictionary.
            seg_info = {
                "id": panoptic_id,                 # panoptic id that matches value in PNG
                "category_id": cat_id,
                "area": ann.get("area", 0),
                "bbox": ann.get("bbox", []),
                "iscrowd": ann.get("iscrowd", 0)
            }
            segments_info.append(seg_info)
        
        panoptic_annotation = {
            "image_id": image_id,
            "file_name": panoptic_png,  # must match the file name of the generated panoptic PNG in your folder
            "segments_info": segments_info
        }
        panoptic_annotations.append(panoptic_annotation)

    panoptic["annotations"] = panoptic_annotations

    # Add categories. We assume the same categories as in the instance JSON.
    # For panoptic annotations, it's common to add an "isthing" flag.
    categories = instance_json.get("categories", [])
    new_categories = []
    for cat in categories:
        new_cat = {
            "id": cat["id"],
            "name": cat["name"],
            # If you know the category is a thing, set isthing to 1; if stuff, use 0.
            # Here we assume all instance categories are "things". Adjust if needed.
            "isthing": 1
        }
        # Optionally, you can add a "color" field.
        new_categories.append(new_cat)
    panoptic["categories"] = new_categories

    print("Processed {} images".format(len(panoptic_annotations)))
    return panoptic

def _iter_json_files(input_path: str) -> List[str]:
    """입력 경로가 파일이면 단일 파일 리스트, 디렉토리면 모든 JSON 파일 리스트 반환"""
    if os.path.isfile(input_path):
        return [input_path]
    out: List[str] = []
    for fn in os.listdir(input_path):
        if fn.lower().endswith(".json"):
            out.append(os.path.join(input_path, fn))
    return sorted(out)


def _save_json(path: str, data: dict) -> None:
    """JSON 파일 저장 (디렉토리 자동 생성)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Convert instance segmentation JSON to panoptic JSON format"
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help="입력 instance JSON 파일 (단일 파일 처리 시)",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="입력 instance JSON 폴더 (여러 파일 처리 시)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="출력 panoptic JSON 파일 (입력이 파일일 때)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="출력 panoptic JSON 폴더 (입력이 폴더일 때)",
    )
    parser.add_argument(
        "--output-png-ext",
        default=".png",
        help="panoptic PNG 파일 확장자 (기본: .png)",
    )
    
    args = parser.parse_args()
    
    # 입력 경로 확인
    input_path = args.input_json or args.input_dir
    if not input_path:
        raise SystemExit("--input-json 또는 --input-dir 중 하나는 필요합니다.")
    
    # 출력 경로 확인
    if os.path.isfile(input_path):
        if not args.output_json:
            raise SystemExit("입력이 파일이면 --output-json을 지정하세요.")
    else:
        if not args.output_dir:
            raise SystemExit("입력이 폴더이면 --output-dir을 지정하세요.")
    
    # JSON 파일 목록 가져오기
    json_files = _iter_json_files(input_path)
    
    if len(json_files) == 0:
        raise SystemExit(f"처리할 JSON 파일을 찾을 수 없습니다: {input_path}")
    
    print(f"처리할 JSON 파일: {len(json_files)}개")
    
    # 각 JSON 파일 처리
    processed = 0
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        print(f"\n처리 중: {file_name}")
        
        # Load the instance JSON.
        with open(json_file, "r", encoding="utf-8") as f:
            instance_data = json.load(f)
        
        # Convert to panoptic JSON.
        panoptic_data = convert_instance_to_panoptic(instance_data, output_png_ext=args.output_png_ext)
        
        # Save the panoptic JSON.
        if os.path.isfile(input_path):
            output_path = args.output_json
        else:
            output_path = os.path.join(args.output_dir, file_name)
        
        _save_json(output_path, panoptic_data)
        print(f"✅ 저장 완료: {output_path}")
        processed += 1
    
    print(f"\n=== 완료 ===")
    print(f"- 처리된 파일: {processed}개")
    print(f"- 입력: {input_path}")
    if os.path.isfile(input_path):
        print(f"- 출력: {args.output_json}")
    else:
        print(f"- 출력: {args.output_dir}")


if __name__ == "__main__":
    main()