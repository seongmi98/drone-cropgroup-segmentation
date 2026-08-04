#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sem_seg PNG 마스크의 픽셀 라벨(id)을 학습용으로 리매핑하는 스크립트입니다.

## 지원 모드
- shift (기본): ignore 라벨(기본 255)을 제외한 나머지 라벨에 일정 값(기본 +1)을 더하는 범용 shift
- custom: old_id → new_id 매핑을 직접 명시하는 커스텀 매핑

### shift 기본 동작 예시 (현재 기본값 기준)
- 옵션: `--mode shift --shift 1 --ignore-label 255 --map-ignore-to 0 --valid-max 3`
- 매핑:
  - 255 → 0  (ignore 를 background 로)
  - 0 → 1
  - 1 → 2
  - 2 → 3
  - 3 → 4

  python3 remap_semseg_keep4.py \
    --input-dir  /path/to/dataset/data2/all/annotations/sem_seg/train4 \
    --output-dir /path/to/dataset/data2/all/annotations/sem_seg2/train4 \
    --valid-max 3

### custom 모드 예시
python3 /path/to/code/remap_semseg_keep4.py \
  --mode custom \
  --input-dir  /path/to/dataset/data/medium/annotations/sem_seg2/test2 \
  --output-dir /path/to/dataset/data/medium_onga/annotations/sem_seg2/test2 \
  --custom-mapping "0:0,1:0,2:1,3:2,255:0"
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image


def _safe_makedirs(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def _iter_pngs(input_dir: str, recursive: bool) -> Tuple[str, str]:
    if recursive:
        for root, _, files in os.walk(input_dir):
            for fn in files:
                if fn.lower().endswith(".png"):
                    src = os.path.join(root, fn)
                    rel = os.path.relpath(src, input_dir)
                    yield src, rel
    else:
        for fn in os.listdir(input_dir):
            if fn.lower().endswith(".png"):
                src = os.path.join(input_dir, fn)
                yield src, fn


def remap_mask_custom(mask: np.ndarray, *, mapping: Dict[int, int], default_value: int) -> np.ndarray:
    """
    커스텀 매핑:
    - mapping에 있는 값은 매핑대로 변환
    - mapping에 없는 값은 default_value로 변환
    """
    out = np.zeros_like(mask, dtype=np.uint8)
    
    # 매핑 적용
    for old_val, new_val in mapping.items():
        out[mask == old_val] = new_val
    
    # 매핑에 없는 값은 default_value로
    mapped_mask = np.zeros_like(mask, dtype=bool)
    for old_val in mapping.keys():
        mapped_mask |= (mask == old_val)
    out[~mapped_mask] = default_value
    
    return out


def remap_mask_shift(
    mask: np.ndarray,
    *,
    shift: int,
    ignore_label: int,
    map_ignore_to: int,
    valid_max: int | None,
) -> np.ndarray:
    """
    범용 shift:
    - ignore_label(기본 255)은 map_ignore_to(보통 0)로 보냄
    - 나머지 라벨은 shift만큼 더함 (예: 0->1, 1->2 ...)
    - valid_max를 넘어간 값은 map_ignore_to로 보냄(안전장치)
    """
    # 핵심: "무시할 픽셀"은 ignore_label(기본 255)만 기준으로 판단해야 함.
    # map_ignore_to가 0인 경우가 많아서, 값==0으로 ignore를 판단하면
    # 원래 클래스 0(예: Cabbage)이 shift에서 빠져버리는 버그가 생김.
    src = mask.astype(np.int16)
    ignore_mask = src == int(ignore_label)

    out = src.copy()
    # ignore는 지정값(보통 0)으로 보냄
    out[ignore_mask] = int(map_ignore_to)

    # ignore가 아닌 픽셀은 "값이 0이어도" 전부 shift
    shift_mask = ~ignore_mask
    out[shift_mask] = out[shift_mask] + int(shift)

    # 유효 범위 밖(원본 라벨 기준)은 background(또는 map_ignore_to)로 보냄
    if valid_max is not None:
        out[(~ignore_mask) & (src > int(valid_max))] = int(map_ignore_to)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def find_max_label(input_dir: str, *, recursive: bool, ignore_label: int) -> int:
    """마스크 파일들을 스캔해서 실제 존재하는 최대 라벨 값 찾기 (ignore_label 제외)"""
    max_label = -1
    scanned = 0
    print(f"🔍 최대 라벨 자동 감지 중... (ignore_label={ignore_label} 제외)")
    
    for src, rel in _iter_pngs(input_dir, recursive):
        mask = np.array(Image.open(src), dtype=np.uint8)
        unique_labels = np.unique(mask)
        # ignore_label 제외한 최대값 찾기
        valid_labels = unique_labels[unique_labels != ignore_label]
        if len(valid_labels) > 0:
            max_label = max(max_label, int(valid_labels.max()))
        scanned += 1
        if scanned % 100 == 0:
            print(f"  ... {scanned}개 파일 스캔, 현재 최대값: {max_label}")
    
    if max_label < 0:
        print("⚠️  유효한 라벨을 찾지 못했습니다. valid-max를 수동으로 지정해주세요.")
        return 0
    
    print(f"✅ 자동 감지 완료: 최대 라벨 = {max_label}")
    return max_label


def process_dir(
    input_dir: str,
    output_dir: str,
    *,
    recursive: bool,
    mode: str,
    shift: int,
    ignore_label: int,
    map_ignore_to: int,
    valid_max: int | None,
    custom_mapping: Optional[Dict[int, int]] = None,
    default_value: int = 0,
) -> Tuple[int, int]:
    # valid_max가 None이면 자동으로 찾기
    if mode == "shift" and valid_max is None:
        valid_max = find_max_label(input_dir, recursive=recursive, ignore_label=ignore_label)
        print()
    elif mode == "shift" and valid_max is not None:
        print(f"📌 --valid-max={valid_max} 지정됨 (자동 감지 스킵)")
    
    total = 0
    written = 0
    for src, rel in _iter_pngs(input_dir, recursive):
        dst = os.path.join(output_dir, rel)
        _safe_makedirs(os.path.dirname(dst))

        mask = np.array(Image.open(src), dtype=np.uint8)
        if mode == "shift":
            remapped = remap_mask_shift(
                mask,
                shift=shift,
                ignore_label=ignore_label,
                map_ignore_to=map_ignore_to,
                valid_max=valid_max,
            )
        elif mode == "custom":
            if custom_mapping is None:
                raise ValueError("custom 모드 사용 시 --custom-mapping이 필요합니다.")
            remapped = remap_mask_custom(mask, mapping=custom_mapping, default_value=default_value)
        else:
            raise ValueError(f"unknown mode: {mode}")
        Image.fromarray(remapped).save(dst)

        total += 1
        written += 1
        if total <= 5:
            print(f"[샘플] {os.path.basename(src)} -> unique labels: {np.unique(remapped)}")

    return total, written


def main() -> int:
    p = argparse.ArgumentParser(description="sem_seg PNG 라벨 id를 리매핑(shift/custom)")
    p.add_argument("--input-dir", required=True, help="원본 sem_seg 폴더")
    p.add_argument("--output-dir", required=True, help="저장할 sem_seg 폴더")
    p.add_argument("--recursive", action="store_true", help="하위 폴더까지 재귀 처리")
    p.add_argument(
        "--mode",
        choices=["shift", "custom"],
        default="shift",
        help="shift=범용 shift (기본: shift 1, ignore 255->0), custom=커스텀 매핑. 기본: shift",
    )
    p.add_argument("--shift", type=int, default=1, help="(shift 전용) ignore가 아닌 라벨에 더할 값. 기본: +1")
    p.add_argument("--ignore-label", type=int, default=255, help="(shift 전용) ignore 라벨 값. 기본: 255")
    p.add_argument("--map-ignore-to", type=int, default=0, help="(shift 전용) ignore를 보낼 값(보통 0). 기본: 0")
    p.add_argument(
        "--valid-max",
        type=int,
        default=None,
        help="(shift 전용) shift 적용 전 유효 라벨의 최대값. 예: 이미 0..3이면 3. 지정하지 않으면 자동 감지 (모든 마스크 파일 스캔)",
    )
    p.add_argument(
        "--no-auto-valid-max",
        action="store_true",
        help="--valid-max가 None일 때 자동 감지를 끄고 기본값(0) 사용 (자동 감지가 느릴 때)",
    )
    p.add_argument(
        "--custom-mapping",
        type=str,
        default=None,
        help='(custom 전용) 커스텀 매핑. 예: "0:1,1:2,2:0,3:0,255:0" (old:new 형식)',
    )
    p.add_argument(
        "--default-value",
        type=int,
        default=0,
        help="(custom 전용) 매핑에 없는 값의 기본값. 기본: 0",
    )

    args = p.parse_args()
    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"input-dir 폴더가 없습니다: {args.input_dir}")

    # custom_mapping 파싱
    custom_mapping: Optional[Dict[int, int]] = None
    if args.mode == "custom":
        if not args.custom_mapping:
            raise SystemExit("custom 모드 사용 시 --custom-mapping이 필요합니다.")
        custom_mapping = {}
        for pair in args.custom_mapping.split(","):
            pair = pair.strip()
            if ":" not in pair:
                raise SystemExit(f"매핑 형식 오류: {pair} (예: 0:1)")
            old_val, new_val = pair.split(":", 1)
            custom_mapping[int(old_val.strip())] = int(new_val.strip())
        print(f"📌 커스텀 매핑: {custom_mapping}")

    # valid_max 처리
    valid_max = args.valid_max
    if args.mode == "shift" and valid_max is None and not args.no_auto_valid_max:
        # 자동 감지 사용 (process_dir에서 처리)
        pass
    elif args.mode == "shift" and valid_max is None:
        # 자동 감지 비활성화, 기본값 0 사용
        valid_max = 0
        print(f"⚠️  valid-max가 지정되지 않았습니다. 기본값 0 사용 (모든 라벨이 shift됨)")

    total, written = process_dir(
        args.input_dir,
        args.output_dir,
        recursive=args.recursive,
        mode=args.mode,
        shift=args.shift,
        ignore_label=args.ignore_label,
        map_ignore_to=args.map_ignore_to,
        valid_max=valid_max,
        custom_mapping=custom_mapping,
        default_value=args.default_value,
    )
    print("✅ 완료")
    print(f"- 처리 대상: {total}개")
    print(f"- 저장됨: {written}개")
    print(f"- output-dir: {args.output_dir}")
    if args.mode == "custom":
        print(f"- 출력 라벨 정의(custom): {custom_mapping}")
        print(f"- 매핑되지 않은 값: {args.default_value}")
    else:
        final_valid_max = valid_max if valid_max is not None else "자동 감지"
        if valid_max is not None:
            print(f"- 출력 라벨 정의(shift): ignore({args.ignore_label})->{args.map_ignore_to}, 0..{valid_max} -> {args.shift}..{valid_max + args.shift}")
        else:
            print(f"- 출력 라벨 정의(shift): ignore({args.ignore_label})->{args.map_ignore_to}, 라벨 자동 감지됨")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


