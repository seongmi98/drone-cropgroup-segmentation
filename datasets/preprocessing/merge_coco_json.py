import os
import json
from glob import glob

# === 설정 ===
input_root_folder = "/path/to/dataset/data2/all/json/train"
output_path = "/path/to/dataset/data2/all/json/train.json"

# COCO 형식 여부 체크 (Labelme 등 다른 포맷 섞여 있으면 스킵)
def is_coco_like(d):
    return isinstance(d, dict) and isinstance(d.get("images"), list) and isinstance(d.get("annotations"), list)

# === 기본 구조
merged_json = {
    "images": [],
    "annotations": [],
    "categories": []
}

# === 중복 방지용
image_id_offset = 0
annotation_id_offset = 0
category_set = {}

# === 모든 하위 폴더까지 탐색해서 json 파일 찾기
if not os.path.isdir(input_root_folder):
    raise SystemExit(f"❌ 입력 폴더가 존재하지 않습니다: {input_root_folder}")

json_files = glob(os.path.join(input_root_folder, "**", "*.json"), recursive=True)
print(f"🔍 찾은 JSON 파일 수: {len(json_files)}개")
if len(json_files) == 0:
    raise SystemExit(
        "❌ JSON 파일을 찾지 못했습니다.\n"
        f" - 입력 폴더: {input_root_folder}\n"
        " - 확인: 하위 폴더에 *.json이 있는지, 경로/철자(예: radish vs raddish)가 맞는지 확인하세요."
    )

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not is_coco_like(data):
        print(f"⚠️  COCO 형식이 아니라 스킵: {json_file}")
        continue

    # categories
    for cat in data.get("categories", []):
        if cat["id"] not in category_set:
            category_set[cat["id"]] = cat

    # images
    for img in data.get("images", []):
        new_img = img.copy()
        new_img["id"] += image_id_offset
        merged_json["images"].append(new_img)

    # annotations
    for ann in data.get("annotations", []):
        new_ann = ann.copy()
        new_ann["id"] += annotation_id_offset
        new_ann["image_id"] += image_id_offset
        merged_json["annotations"].append(new_ann)

    # offset 업데이트
    if data.get("images"):
        image_id_offset = max(img["id"] for img in merged_json["images"]) + 1
    if data.get("annotations"):
        annotation_id_offset = max(ann["id"] for ann in merged_json["annotations"]) + 1

# categories 정리
merged_json["categories"] = list(category_set.values())

# 저장
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_json, f, indent=4, ensure_ascii=False)

print(f"✅ JSON 병합 완료: {output_path}")
