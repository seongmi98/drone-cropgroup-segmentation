#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic segmentation 마스크의 클래스 불균형 분석 스크립트
"""

import os
import numpy as np
from PIL import Image
from collections import defaultdict

# 데이터셋 경로
GT_Train1 = "/path/to/dataset/data2/all_cara1/annotations/sem_seg_bg0/train1"
GT_Train2 = "/path/to/dataset/data2/all_cara1/annotations/sem_seg_bg0/train2"
GT_Train3 = "/path/to/dataset/data2/all_cara1/annotations/sem_seg_bg0/train3"
GT_Train4 = "/path/to/dataset/data2/all_cara1/annotations/sem_seg_bg0/train4"

GT_TEST1 = "/path/to/dataset/data2/all_cara1/annotations/sem_seg_bg0/test1"
GT_TEST2 = "/path/to/dataset/data2/all_cara1/annotations/sem_seg_bg0/test2"

# 클래스명
CLASS_NAMES = {0: "Background", 1: "Cabbage", 2: "Radish"}

def analyze_directory(dir_path, dir_name):
    """디렉토리의 모든 PNG 파일 분석"""
    print(f"\n{'='*60}")
    print(f"[{dir_name}] 분석 중...")
    print(f"{'='*60}")
    
    class_pixels = defaultdict(int)
    total_pixels = 0
    image_count = 0
    
    if not os.path.exists(dir_path):
        print(f"[ERROR] 경로 없음: {dir_path}")
        return None
    
    png_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
    
    if not png_files:
        print(f"[WARNING] PNG 파일 없음: {dir_path}")
        return None
    
    for idx, filename in enumerate(png_files, 1):
        filepath = os.path.join(dir_path, filename)
        try:
            img = np.array(Image.open(filepath), dtype=np.uint8)
            unique, counts = np.unique(img, return_counts=True)
            
            for cls_id, count in zip(unique, counts):
                class_pixels[cls_id] += count
                total_pixels += count
            
            image_count += 1
            
            if idx % 100 == 0 or idx == len(png_files):
                print(f"  처리 중: {idx}/{len(png_files)}", end='\r')
        except Exception as e:
            print(f"[ERROR] 파일 처리 실패: {filename} - {e}")
    
    print(f"\n이미지 개수: {image_count}")
    print(f"총 픽셀 수: {total_pixels:,}")
    
    # 결과 출력
    print(f"\n클래스별 분포:")
    print(f"{'-'*60}")
    for cls_id in sorted(class_pixels.keys()):
        count = class_pixels[cls_id]
        ratio = (count / total_pixels) * 100
        cls_name = CLASS_NAMES.get(cls_id, f"Unknown_{cls_id}")
        print(f"  {cls_name:15s} (ID={cls_id}): {count:,} px ({ratio:6.2f}%)")
    
    return class_pixels, total_pixels

def main():
    print("\n" + "="*60)
    print("클래스 불균형 분석")
    print("="*60)
    
    # 학습 데이터
    train_dirs = [
        (GT_Train1, "Train1"),
        (GT_Train2, "Train2"),
        (GT_Train3, "Train3"),
        (GT_Train4, "Train4"),
    ]
    
    # 테스트 데이터
    test_dirs = [
        (GT_TEST1, "Val1"),
        (GT_TEST2, "Val2"),
    ]
    
    all_class_pixels = defaultdict(int)
    all_total_pixels = 0
    
    # 학습 데이터 분석
    print("\n[학습 데이터]")
    for dir_path, dir_name in train_dirs:
        result = analyze_directory(dir_path, dir_name)
        if result:
            class_pixels, total_pixels = result
            for cls_id, count in class_pixels.items():
                all_class_pixels[cls_id] += count
                all_total_pixels += count
    
    # 테스트 데이터 분석
    print("\n\n[테스트 데이터]")
    test_class_pixels = defaultdict(int)
    test_total_pixels = 0
    for dir_path, dir_name in test_dirs:
        result = analyze_directory(dir_path, dir_name)
        if result:
            class_pixels, total_pixels = result
            for cls_id, count in class_pixels.items():
                test_class_pixels[cls_id] += count
                test_total_pixels += count
    
    # 전체 통계
    print("\n\n" + "="*60)
    print("[전체 학습 데이터 통계]")
    print("="*60)
    print(f"총 픽셀 수: {all_total_pixels:,}")
    print(f"\n클래스별 분포:")
    print(f"{'-'*60}")
    for cls_id in sorted(all_class_pixels.keys()):
        count = all_class_pixels[cls_id]
        ratio = (count / all_total_pixels) * 100
        cls_name = CLASS_NAMES.get(cls_id, f"Unknown_{cls_id}")
        print(f"  {cls_name:15s} (ID={cls_id}): {count:,} px ({ratio:6.2f}%)")
    
    # 가중치 제안
    print("\n" + "="*60)
    print("[권장 클래스 가중치]")
    print("="*60)
    
    # Inverse frequency 가중치
    print("\n방법 1: Inverse Frequency (더 흔한 클래스에 낮은 가중치)")
    total_classes = len(all_class_pixels)
    weights_inv_freq = {}
    for cls_id in sorted(all_class_pixels.keys()):
        count = all_class_pixels[cls_id]
        # 역빈도 가중치: (전체 클래스 수) / (해당 클래스 픽셀)
        weight = all_total_pixels / (total_classes * count)
        weights_inv_freq[cls_id] = weight
    
    # 정규화 (합 = 클래스 수)
    sum_weights = sum(weights_inv_freq.values())
    weights_inv_freq = {k: v / sum_weights * total_classes for k, v in weights_inv_freq.items()}
    
    for cls_id in sorted(weights_inv_freq.keys()):
        cls_name = CLASS_NAMES.get(cls_id, f"Unknown_{cls_id}")
        weight = weights_inv_freq[cls_id]
        print(f"  {cls_name:15s} (ID={cls_id}): {weight:.4f}")
    
    # Effective number of samples 가중치
    print("\n방법 2: Effective Number (권장 - 더 균형잡힘)")
    beta = 0.9999  # 권장값
    weights_en = {}
    for cls_id in sorted(all_class_pixels.keys()):
        count = all_class_pixels[cls_id]
        weight = (1 - beta) / (1 - beta ** count)
        weights_en[cls_id] = weight
    
    # 정규화
    sum_weights = sum(weights_en.values())
    weights_en = {k: v / sum_weights * total_classes for k, v in weights_en.items()}
    
    for cls_id in sorted(weights_en.keys()):
        cls_name = CLASS_NAMES.get(cls_id, f"Unknown_{cls_id}")
        weight = weights_en[cls_id]
        print(f"  {cls_name:15s} (ID={cls_id}): {weight:.4f}")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
