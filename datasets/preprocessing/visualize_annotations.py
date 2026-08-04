# import os
# import json
# import cv2
# import numpy as np
# import random
# from tqdm import tqdm

# dataset_dir = "/path/to/dataset/8_2/high/radish/annotations/raw"  # adjust if needed
# images_dir = "/path/to/dataset/8_2/high/radish/images"  # adjust if needed
# # json_files = {'train': os.path.join(dataset_dir, "train.json") ,
# #                 'val': os.path.join(dataset_dir, "val.json"),
# #                 'test': os.path.join(dataset_dir, "test.json")}
# json_files = {'test1': os.path.join(dataset_dir, "test1.json"),
#                  'test2': os.path.join(dataset_dir, "test2.json")}

# CLASS_COLORS = {
#     # 0: np.array([1.0, 0.0, 0.0], dtype=np.float32),  # red
#     # 1: np.array([0.0, 1.0, 0.0], dtype=np.float32),  # green
#     # 2: np.array([0.0, 0.0, 1.0], dtype=np.float32),  # blue
#     # 3: np.array([1.0, 1.0, 0.0], dtype=np.float32),  # yellow
#     # 4: np.array([1.0, 0.0, 1.0], dtype=np.float32),  # magenta

#     0: np.array([9/255, 224/255, 188/255], dtype=np.float32),   # Cabbage
#     1: np.array([222/255, 11/255, 194/255], dtype=np.float32),  # Radish
#     2: np.array([190/255, 222/255, 11/255], dtype=np.float32),  # Onion
#     3: np.array([9/255, 224/255, 56/255], dtype=np.float32),    # Greenonion
#     4: np.array([224/255, 160/255, 9/255], dtype=np.float32),   # Garlic
#     5: np.array([224/255, 9/255, 38/255], dtype=np.float32)     # Chilipepper
# }


# def get_opencv_color(np_color):
#     color = (np_color * 255).astype(np.int32)
#     bgr = (int(color[2]), int(color[1]), int(color[0]))
#     return bgr

# cat_colors = {cat_id: get_opencv_color(color) for cat_id, color in CLASS_COLORS.items()}

# for index, json_file in json_files.items():
#     output_dir = os.path.join(dataset_dir, f"visualization_{index}")
#     os.makedirs(output_dir, exist_ok=True)

#     with open(json_file, "r") as f:
#         coco_data = json.load(f)

#     # Build a mapping from image_id to its annotations.
#     anns_per_image = {}
#     for ann in coco_data["annotations"]:
#         image_id = ann["image_id"]
#         anns_per_image.setdefault(image_id, []).append(ann)

#     # Process each image using tqdm progress bar.
#     for img_info in tqdm(coco_data["images"], desc=f"Processing {index}"):
#         img_id = img_info["id"]
#         # file_name = img_info["file_name"]
#         file_name = img_info["path"]
#         # img_path = os.path.join(images_dir, file_name)
#         img_path = file_name

#         image = cv2.imread(img_path)
#         if image is None:
#             print("Failed to read {}".format(img_path))
#             continue

#         # Create overlay and draw the masks.
#         overlay = image.copy()
#         if img_id in anns_per_image:
#             for ann in anns_per_image[img_id]:
#                 cat_id = ann["category_id"]
#                 # Use fixed color; if cat_id not in CLASS_COLORS,
#                 # default to green (converted).
#                 poly_color = cat_colors.get(cat_id, get_opencv_color(np.array([0.0, 1.0, 0.0], dtype=np.float32)))
#                 for seg in ann["segmentation"]:
#                     pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
#                     cv2.fillPoly(overlay, [pts], color=poly_color)

#         # Blend the mask overlay with the original image.
#         alpha = 0.5
#         blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

#         output_path = os.path.join(output_dir, os.path.basename(file_name))
#         cv2.imwrite(output_path, blended)

# print("Visualization complete.")


import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# === 경로 설정 ===
IMAGE_DIR = "/path/to/dataset/8_2/low/all/images/test"
MASK_DIR = "/path/to/dataset/8_2/low/all/annotations/sem_seg/test"
OUTPUT_DIR = "/path/to/dataset/8_2/low/all/annotations/raw/visualization_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 클래스 색 정의 (OpenCV용 BGR 순서) ===
CLASS_COLORS = {
     0: [255, 255, 255], # Background
     1: [188, 224, 9],    # Cabbage (BGR)
     2: [194, 11, 222],   # Radish
     3: [11, 222, 190],   # Onion
     4: [9, 160, 224],    # Garlic
     5: [38, 9, 224],     # Chilipepper
 }

# CLASS_COLORS = {
#     0: [255, 255, 255], # Background
#     1: [11, 222, 190],   # Onion
#     2: [9, 160, 224],    # Garlic
# }

# === 이미지 목록 불러오기 ===
image_paths = []
for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
    image_paths.extend(glob(os.path.join(IMAGE_DIR, ext)))
image_paths = sorted(image_paths)

# === 시각화 루프 ===
for img_path in tqdm(image_paths, desc="시각화 중"):
    base = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(MASK_DIR, f"{base}.png")

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"❌ 오류: {img_path} 또는 {mask_path}")
        continue

    if mask.ndim != 2:
        print(f"❌ 잘못된 마스크 형식: {mask_path} (ndim={mask.ndim})")
        continue

    # === 색깔 마스크 만들기 ===
    color_mask = np.zeros_like(image)
    for class_id, bgr_color in CLASS_COLORS.items():
        color_mask[mask == class_id] = bgr_color

    # === 합성 ===
    alpha = 0.5
    blended = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    # === 저장 ===
    save_path = os.path.join(OUTPUT_DIR, f"{base}.jpg")
    cv2.imwrite(save_path, blended)

print("✅ PNG 마스크 기반 시각화 완료.")