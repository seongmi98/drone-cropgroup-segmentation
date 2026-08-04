import json
import random
from collections import defaultdict
import os

# === 설정 ===
input_json_path = "/path/to/dataset/low/merged_output.json"  # 합친 JSON 파일 경로
output_json_path = "/path/to/dataset/low/undersampled_output.json"  # undersample된 JSON 저장 경로
max_per_class = 400  # 클래스당 최대 이미지 수

# === JSON 불러오기
with open(input_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# === 이미지별 포함 클래스 매핑
image_to_classes = defaultdict(set)
for ann in data['annotations']:
    image_to_classes[ann['image_id']].add(ann['category_id'])

# === 클래스별 이미지 모으기
class_to_images = defaultdict(set)
for image_id, classes in image_to_classes.items():
    for cls in classes:
        class_to_images[cls].add(image_id)

# === undersample할 이미지 고르기
selected_image_ids = set()

for cls, img_ids in class_to_images.items():
    img_ids = list(img_ids)
    if len(img_ids) > max_per_class:
        sampled = random.sample(img_ids, max_per_class)
    else:
        sampled = img_ids
    selected_image_ids.update(sampled)

print(f"✅ 최종 선택된 이미지 수: {len(selected_image_ids)}")

# === 선택된 이미지/annotation만 추리기
selected_images = [img for img in data['images'] if img['id'] in selected_image_ids]
selected_annotations = [ann for ann in data['annotations'] if ann['image_id'] in selected_image_ids]

# === 최종 저장할 JSON 구성
output_json = {
    "images": selected_images,
    "annotations": selected_annotations,
    "categories": data['categories']
}

# === 저장
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(output_json, f, indent=4, ensure_ascii=False)

print(f"✅ undersample JSON 저장 완료: {output_json_path}")
