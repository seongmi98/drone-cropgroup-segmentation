#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO JSON에서 클래스 제거 + category_id 정렬(0,1,2,...)을 한 번에 처리합니다.
(기존 filter_coco_drop_classes + remap_coco_category_ids 통합)

## 터미널 실행 예시

### 제거 + 정렬(기본: 제거한 뒤 남은 id를 0..K-1로 자동 정렬)
  python3 filter_coco_drop_classes.py \
    --input-dir  /path/to/dataset/data2/all/json/real \
    --output-dir /path/to/dataset/data2/all_onga/json/real \
    --drop-ids 0,1 \
    --remap-contiguous

python3 filter_coco_drop_classes.py \
  --input-json  /path/to/dataset/data2/all/json/test/test.json \
  --output-json /path/to/dataset/data2/all_onga/json/test/test.json \
  --drop-ids 3,5 \
  --remove-empty-images \
  --keep-non-dropped-categories \
  --remap-contiguous

### 제거만 하고 id 정렬은 안 함
  python3 filter_coco_drop_classes.py ... --drop-ids 3,5 --no-remap-contiguous

### 합칠 때 categories 일관성 유지(제거+정렬)
  python3 filter_coco_drop_classes.py ... --drop-ids 3,5 --keep-non-dropped-categories


### 매핑만(제거 없이 id만 재매핑, 예: 0,1,2,4 -> 0,1,2,3)
  python3 filter_coco_drop_classes.py \
    --input-json  /path/to/dataset/data2/all/json/train/2022_배추.json \
    --output-json /path/to/dataset/data2/all/json/train/2022_배추.json \
    --mapping "0:0,1:1,2:2,4:3" \
    --drop-unmapped \
    --keep-non-dropped-categories \
    --remove-empty-images

옵션:
  - --drop-ids : 제거할 category id (콤마). 지정하면 제거 후 기본으로 --remap-contiguous 적용
  - --remap-contiguous : 남은 id를 0..K-1로 재매핑(제거 시 기본 True). 끄려면 --no-remap-contiguous
  - --mapping : 제거 대신 매핑 모드. 예: "0:0,1:1,2:2,4:3". --drop-unmapped와 함께 사용
  - --remove-empty-images : annotation 0개인 이미지 제거(추천)
  - --keep-non-dropped-categories : drop 시 나머지 category 메타 유지(JSON 합칠 때 추천)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple


# 프로젝트에서 쓰는 6개 클래스(원본 id 기준) 템플릿
# 일부 JSON에 categories가 일부만 있더라도, 필요하면 이 템플릿으로 categories를 "채워넣을" 수 있음.
KNOWN_CATEGORIES: List[Dict[str, Any]] = [
    {"id": 0, "name": "Cabbage", "color": [9, 224, 188]},
    {"id": 1, "name": "Radish", "color": [222, 11, 194]},
    {"id": 2, "name": "Onion", "color": [190, 222, 11]},
    {"id": 3, "name": "Garlic", "color": [224, 160, 9]},
]
KNOWN_ID_TO_CAT = {int(c["id"]): c for c in KNOWN_CATEGORIES}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _is_coco_like(d: Any) -> bool:
    return isinstance(d, dict) and isinstance(d.get("images"), list) and isinstance(d.get("annotations"), list)


def _iter_json_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    out: List[str] = []
    for fn in os.listdir(input_path):
        if fn.lower().endswith(".json"):
            out.append(os.path.join(input_path, fn))
    return sorted(out)


def _parse_ids_csv(s: str) -> Set[int]:
    ids: Set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        ids.add(int(part))
    return ids


def _parse_mapping(s: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"mapping 형식 오류: {pair} (예: 4:3)")
        a, b = pair.split(":", 1)
        mapping[int(a.strip())] = int(b.strip())
    if not mapping:
        raise ValueError("mapping이 비었습니다.")
    return mapping


def remap_coco(
    data: Dict[str, Any],
    *,
    mapping: Dict[int, int],
    drop_unmapped: bool,
    remove_empty_images: bool,
    keep_non_dropped_categories: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """매핑만 적용(제거 없이 id 재매핑). mapping에 없는 id는 drop_unmapped면 제거."""
    images: List[Dict[str, Any]] = data.get("images", [])
    annotations: List[Dict[str, Any]] = data.get("annotations", [])
    categories: List[Dict[str, Any]] = data.get("categories", [])

    mapped_annotations: List[Dict[str, Any]] = []
    dropped_anns = 0
    for ann in annotations:
        if not isinstance(ann, dict) or ann.get("category_id") is None:
            if drop_unmapped:
                dropped_anns += 1
            continue
        old_cid = int(ann.get("category_id"))
        if old_cid not in mapping:
            if drop_unmapped:
                dropped_anns += 1
                continue
            mapped_annotations.append(ann)
            continue
        new_ann = dict(ann)
        new_ann["category_id"] = mapping[old_cid]
        mapped_annotations.append(new_ann)

    if remove_empty_images:
        kept_image_ids: Set[int] = set()
        for ann in mapped_annotations:
            iid = ann.get("image_id")
            if iid is not None:
                kept_image_ids.add(int(iid))
        mapped_images = [img for img in images if isinstance(img, dict) and img.get("id") in kept_image_ids]
        mapped_annotations = [ann for ann in mapped_annotations if ann.get("image_id") in kept_image_ids]
    else:
        mapped_images = images

    old_cat_by_id: Dict[int, Dict[str, Any]] = {}
    for cat in categories:
        if isinstance(cat, dict) and cat.get("id") is not None:
            try:
                old_cat_by_id[int(cat.get("id"))] = cat
            except Exception:
                pass

    used_new_ids: Set[int] = {int(ann["category_id"]) for ann in mapped_annotations if ann.get("category_id") is not None}
    new_to_old: Dict[int, int] = {}
    for old_id, new_id in mapping.items():
        new_to_old.setdefault(new_id, old_id)

    new_categories: List[Dict[str, Any]] = []
    # mapping 모드에서도 "인스턴스 0개 클래스"를 categories에 남기고 싶으면
    # keep_non_dropped_categories=True로 켜고, mapping 결과 id 집합을 기준으로 categories를 구성한다.
    base_new_ids: Set[int]
    if keep_non_dropped_categories:
        base_new_ids = set(mapping.values())
    else:
        base_new_ids = set(used_new_ids)

    for new_id in sorted(base_new_ids):
        old_id = new_to_old.get(new_id)
        if old_id is not None and old_id in old_cat_by_id:
            c = dict(old_cat_by_id[old_id])
            c["id"] = new_id
            new_categories.append(c)
        else:
            # 원본 categories에 없으면 KNOWN 템플릿을 우선 사용
            if old_id is not None and old_id in KNOWN_ID_TO_CAT:
                c = dict(KNOWN_ID_TO_CAT[old_id])
                c["id"] = new_id
                new_categories.append(c)
            else:
                new_categories.append({"id": new_id, "name": str(new_id)})

    out = dict(data)
    out["images"] = mapped_images
    out["annotations"] = mapped_annotations
    out["categories"] = new_categories
    stats = {
        "images_before": len(images),
        "images_after": len(mapped_images),
        "annotations_before": len(annotations),
        "annotations_after": len(mapped_annotations),
        "categories_before": len(categories),
        "categories_after": len(new_categories),
        "dropped_annotations": dropped_anns,
    }
    return out, stats


def filter_coco(
    data: Dict[str, Any],
    *,
    drop_category_ids: Set[int],
    remove_empty_images: bool,
    remap_contiguous: bool,
    keep_non_dropped_categories: bool,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    images: List[Dict[str, Any]] = data.get("images", [])
    annotations: List[Dict[str, Any]] = data.get("annotations", [])
    categories: List[Dict[str, Any]] = data.get("categories", [])

    # 1) drop 대상 annotation 제거
    kept_annotations = [
        ann
        for ann in annotations
        if isinstance(ann, dict)
        and ann.get("category_id") is not None
        and int(ann.get("category_id")) not in drop_category_ids
    ]

    # 2) 남은 annotation이 가리키는 image_id만 남김(옵션)
    kept_image_ids: Optional[Set[int]] = None
    if remove_empty_images:
        kept_image_ids = set()
        for ann in kept_annotations:
            iid = ann.get("image_id")
            if iid is None:
                continue
            kept_image_ids.add(int(iid))

    if kept_image_ids is None:
        kept_images = images
    else:
        kept_images = [img for img in images if isinstance(img, dict) and img.get("id") in kept_image_ids]
        # 이미지가 삭제되면, 안전하게 annotation도 그 이미지들만 남김
        kept_annotations = [ann for ann in kept_annotations if ann.get("image_id") in kept_image_ids]

    # 3) categories 정리
    # - 기본 동작: 실제 사용되는 category만 남김(used_cat_ids)
    # - 옵션 동작(keep_non_dropped_categories=True): drop만 제외하고, 나머지는 "인스턴스 0개여도" categories에 남김
    used_cat_ids: Set[int] = set()
    for ann in kept_annotations:
        cid = ann.get("category_id")
        if cid is None:
            continue
        used_cat_ids.add(int(cid))

    if keep_non_dropped_categories:
        # 유지할 카테고리 id 집합 = (원본 categories에 있던 것 ∪ KNOWN 템플릿) - drop
        source_cat_ids: Set[int] = set()
        for cat in categories:
            if isinstance(cat, dict) and cat.get("id") is not None:
                try:
                    source_cat_ids.add(int(cat.get("id")))
                except Exception:
                    pass
        keep_ids = (source_cat_ids | set(KNOWN_ID_TO_CAT.keys())) - set(drop_category_ids)

        # 원본 categories에 있던 메타(이름/색 등)를 우선 사용하고, 없으면 KNOWN 템플릿에서 채움
        orig_by_id: Dict[int, Dict[str, Any]] = {}
        for cat in categories:
            if not isinstance(cat, dict) or cat.get("id") is None:
                continue
            try:
                cid = int(cat.get("id"))
            except Exception:
                continue
            orig_by_id[cid] = cat

        kept_categories = []
        for cid in sorted(keep_ids):
            if cid in orig_by_id:
                kept_categories.append(orig_by_id[cid])
            else:
                kept_categories.append(dict(KNOWN_ID_TO_CAT.get(cid, {"id": cid, "name": str(cid)})))
    else:
        kept_categories = [
            cat
            for cat in categories
            if isinstance(cat, dict) and cat.get("id") is not None and int(cat.get("id")) in used_cat_ids
        ]

    # 4) (선택) category_id를 0..K-1로 재매핑
    id_remap: Dict[int, int] = {}
    if remap_contiguous:
        # ⚠️ 주의:
        # - 기본동작(keep_non_dropped_categories=False) 에서는, 실제 annotation에 등장한 id만 기준으로 재매핑.
        # - keep_non_dropped_categories=True 인 경우, categories 에 남겨진 모든 id 집합을 기준으로 재매핑해야
        #   categories 내부에서 id가 중복(예: 0,1,0,1)으로 남지 않음.
        if keep_non_dropped_categories:
            base_ids: Set[int] = set()
            for cat in kept_categories:
                if not isinstance(cat, dict) or cat.get("id") is None:
                    continue
                try:
                    base_ids.add(int(cat.get("id")))
                except Exception:
                    continue
        else:
            base_ids = set(used_cat_ids)

        for new_id, old_id in enumerate(sorted(base_ids)):
            id_remap[old_id] = new_id

        for ann in kept_annotations:
            cid = ann.get("category_id")
            if cid is None:
                continue
            old = int(cid)
            if old in id_remap:
                ann["category_id"] = id_remap[old]

        for cat in kept_categories:
            if not isinstance(cat, dict) or cat.get("id") is None:
                continue
            try:
                old = int(cat.get("id"))
            except Exception:
                continue
            if old in id_remap:
                cat["id"] = id_remap[old]

    out = dict(data)
    out["images"] = kept_images
    out["annotations"] = kept_annotations
    out["categories"] = kept_categories

    stats = {
        "images_before": len(images),
        "images_after": len(kept_images),
        "annotations_before": len(annotations),
        "annotations_after": len(kept_annotations),
        "categories_before": len(categories),
        "categories_after": len(kept_categories),
    }
    return out, stats


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="COCO JSON 클래스 제거 + category_id 정렬(통합)")
    p.add_argument("--input-dir", default=None, help="입력 JSON 폴더")
    p.add_argument("--output-dir", default=None, help="출력 JSON 폴더")
    p.add_argument("--input-json", default=None, help="입력 JSON 파일(단일)")
    p.add_argument("--output-json", default=None, help="출력 JSON 파일(단일 입력 시)")
    p.add_argument(
        "--drop-ids",
        default="",
        help="제거할 category id (콤마). 지정 시 제거 후 기본으로 0..K-1 정렬",
    )
    p.add_argument(
        "--remove-empty-images",
        action="store_true",
        help="필터 후 annotation 0개인 이미지 제거(추천)",
    )
    p.add_argument(
        "--keep-non-dropped-categories",
        action="store_true",
        help="drop 제외한 categories 메타 유지(JSON 합칠 때 추천)",
    )
    p.add_argument(
        "--remap-contiguous",
        action="store_true",
        default=True,
        help="남은 id를 0..K-1로 재매핑(제거 시 기본 켜짐)",
    )
    p.add_argument(
        "--no-remap-contiguous",
        action="store_false",
        dest="remap_contiguous",
        help="정렬(재매핑) 끄기",
    )
    p.add_argument(
        "--mapping",
        default="",
        help='매핑 모드: "0:0,1:1,2:2,4:3" (제거 없이 id만 재매핑)',
    )
    p.add_argument(
        "--drop-unmapped",
        action="store_true",
        help="--mapping 사용 시, 매핑에 없는 category annotation 제거",
    )
    args = p.parse_args(argv)

    input_path = args.input_json or args.input_dir
    if not input_path:
        raise SystemExit("--input-dir 또는 --input-json 필요")
    if os.path.isfile(input_path) and not args.output_json:
        raise SystemExit("입력이 파일이면 --output-json 지정 필요")
    if not os.path.isfile(input_path) and not args.output_dir:
        raise SystemExit("입력이 폴더이면 --output-dir 지정 필요")

    use_mapping_mode = bool(args.mapping.strip())
    if use_mapping_mode:
        mapping = _parse_mapping(args.mapping)
    else:
        if not args.drop_ids.strip():
            raise SystemExit("--drop-ids 또는 --mapping 중 하나 필요")
        drop_ids = _parse_ids_csv(args.drop_ids)

    json_files = _iter_json_files(input_path)
    if not json_files:
        raise SystemExit(f"JSON 파일 없음: {input_path}")

    out_dir = args.output_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    processed = 0
    skipped = 0

    for jf in json_files:
        fn = os.path.basename(jf)
        data = _load_json(jf)
        if not _is_coco_like(data):
            skipped += 1
            print(f"⚠️  스킵(COCO 아님): {fn}")
            continue

        if use_mapping_mode:
            out, stats = remap_coco(
                data,
                mapping=mapping,
                drop_unmapped=args.drop_unmapped,
                remove_empty_images=args.remove_empty_images,
                keep_non_dropped_categories=args.keep_non_dropped_categories,
            )
            extra = f" (drop {stats.get('dropped_annotations', 0)})"
        else:
            out, stats = filter_coco(
                data,
                drop_category_ids=drop_ids,
                remove_empty_images=args.remove_empty_images,
                remap_contiguous=args.remap_contiguous,
                keep_non_dropped_categories=args.keep_non_dropped_categories,
            )
            extra = ""

        if os.path.isfile(input_path):
            out_path = args.output_json
        else:
            out_path = os.path.join(out_dir, fn)
        _save_json(out_path, out)
        processed += 1
        print(
            f"✅ {fn} | images {stats['images_before']}→{stats['images_after']}, "
            f"ann {stats['annotations_before']}→{stats['annotations_after']}{extra}, "
            f"cat {stats['categories_before']}→{stats['categories_after']}"
        )

    print("\n=== 완료 ===")
    print(f"- 처리: {processed}개, 스킵: {skipped}개")
    if use_mapping_mode:
        print(f"- 모드: mapping | mapping: {args.mapping}")
    else:
        print(f"- 모드: drop+remap | drop-ids: {sorted(drop_ids)}, remap-contiguous: {args.remap_contiguous}")
    print(f"- remove-empty-images: {args.remove_empty_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


