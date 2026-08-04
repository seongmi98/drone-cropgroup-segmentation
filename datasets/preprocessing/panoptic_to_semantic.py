#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from panopticapi.utils import rgb2id
from PIL import Image


DEFAULT_DATASET_DIR = "/path/to/dataset/data2/all/annotations"
DEFAULT_PANOPTIC_JSON = "/path/to/dataset/data2/all/annotations/train1.json"


def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    for seg in segments:
        cat_id = seg["category_id"]
        # print(f"id_map: {id_map}")
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
    """
    COCO panoptic annotations(panoptic PNG + panoptic JSON)로부터
    semantic segmentation용 sem_seg 마스크(PNG)를 생성합니다.

    ✅ 실제 동작(중요)
    - 출력 마스크는 먼저 전부 255로 채웁니다. (255 = ignore / unlabeled)
    - 각 segment에 대해, JSON의 segments_info[].category_id 를 받아서
      categories 목록의 순서(i=0..N-1)로 "연속 라벨"로 매핑합니다.
        new_label = i  (i는 categories를 enumerate한 인덱스)
      즉, categories의 "id 값" 자체가 출력 라벨이 되는 게 아니라,
      categories 리스트의 순서가 출력 라벨을 결정합니다.

    ⚠️ 주의
    - panoptic JSON의 category_id와 여기 categories의 id는 반드시 일치해야 합니다.
      (예: B안처럼 Garlic을 4→3으로 재매핑했다면, 여기 categories도 id=3으로 맞춰야 함)

    Args:
        panoptic_json (str): COCO panoptic JSON 경로(annotations에 segments_info 포함)
        panoptic_root (str): panoptic PNG들이 있는 폴더
        sem_seg_root (str): 생성될 sem_seg PNG 저장 폴더
        categories (list[dict]): 카테고리 메타데이터 리스트
            - "id": panoptic JSON의 segments_info[].category_id 와 동일해야 함
            - "isthing": 0/1 (여기 스크립트에선 실질적으로 사용하지 않음)
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(categories) <= 254
    for i, k in enumerate(categories):
        id_map[k["id"]] = i
    # what is id = 0?
    # id_map[0] = 255
    # print(id_map)

    with open(panoptic_json) as f:
        obj = json.load(f)

    total_images = len(obj["annotations"])
    print("Total number of images processed:", total_images)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(sem_seg_root, file_name)
            yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


def _iter_json_files(input_dir: str):
    files = []
    for fn in os.listdir(input_dir):
        if fn.lower().endswith(".json"):
            files.append(os.path.join(input_dir, fn))
    return sorted(files)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COCO panoptic JSON(+PNG) → semantic seg PNG 생성 (단일/폴더 JSON 모두 지원)"
    )
    parser.add_argument(
        "--panoptic-json",
        "--json",
        dest="panoptic_json",
        default=None,
        help="입력 panoptic JSON 파일 (단일)",
    )
    parser.add_argument(
        "--panoptic-json-dir",
        "--json-dir",
        dest="panoptic_json_dir",
        default=None,
        help="입력 panoptic JSON 폴더(.json 여러 개 처리)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="dataset 루트 (panoptic/<stem>, sem_seg/<stem> 하위에 PNG 생성)",
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))
    categories = [
        {"id": 0, "name": "Cabbage", "isthing": 1, "color": [9, 224, 188]},
        {"id": 1, "name": "Radish", "isthing": 1, "color": [222, 11, 194]},
        {"id": 2, "name": "Onion", "isthing": 1, "color": [190, 222, 11]},
        {"id": 3, "name": "Garlic", "isthing": 1, "color": [224, 160, 9]},
    ]

    # 폴더 모드: 여러 JSON 한 번에
    if args.panoptic_json_dir:
        json_dir = os.path.abspath(os.path.expanduser(args.panoptic_json_dir))
        if not os.path.isdir(json_dir):
            raise SystemExit(f"panoptic JSON 폴더가 없습니다: {json_dir}")
        json_files = _iter_json_files(json_dir)
        if not json_files:
            raise SystemExit(f"폴더에 .json 파일이 없습니다: {json_dir}")

        print(f"폴더 모드: {json_dir} (총 {len(json_files)}개)")
        print(f"dataset_dir: {dataset_dir}")
        for i, jf in enumerate(json_files):
            name = _stem(jf)
            panoptic_root = os.path.join(dataset_dir, "panoptic", name)
            sem_seg_root = os.path.join(dataset_dir, "sem_seg", name)
            print(f"\n--- [{i+1}/{len(json_files)}] {os.path.basename(jf)} ---")
            print(f"  panoptic_root: {panoptic_root}")
            print(f"  sem_seg_root:  {sem_seg_root}")
            separate_coco_semantic_from_panoptic(jf, panoptic_root, sem_seg_root, categories)
    else:
        # 단일 파일 모드
        panoptic_json = args.panoptic_json or DEFAULT_PANOPTIC_JSON
        panoptic_json = os.path.abspath(os.path.expanduser(panoptic_json))
        if not os.path.isfile(panoptic_json):
            raise SystemExit(f"panoptic JSON 파일이 없습니다: {panoptic_json}")

        name = _stem(panoptic_json)
        panoptic_root = os.path.join(dataset_dir, "panoptic", name)
        sem_seg_root = os.path.join(dataset_dir, "sem_seg", name)
        print(f"단일 모드: {panoptic_json}")
        print(f"  panoptic_root: {panoptic_root}")
        print(f"  sem_seg_root:  {sem_seg_root}")
        separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories)