#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data3/all 전체 이미지에 대해, 모든 실험(전체고도 모델 / 고도별 특화 모델)에서
공통으로 재사용할 "master 5-fold" split을 한 번만 생성합니다.

## 왜 image 단위 랜덤/StratifiedKFold가 아니라 이 방식인가
- 이미지 단위 fold 배정 결과(CSV)를 산출물로 만들지만, 배정 "단위"는 그룹(=같은 날
  같은 지번/collection에서 나온 사진 묶음)입니다. 같은 그룹이 train/val로 쪼개지면
  사실상 거의 동일한 사진을 보고 검증하는 leakage가 생기기 때문입니다.
- crop_class + altitude + region + year를 전부 묶어서 stratify하면 표본이 5장 미만인
  조합이 쏟아져서 StratifiedKFold/StratifiedGroupKFold가 제대로 동작하지 않습니다.
  그래서 "완전 조합 stratify" 대신, 우선순위(altitude > region > year)를 둔
  가중치 기반 greedy LPT(Longest Processing Time) 배정을 사용합니다:

    각 그룹을 크기(이미지 수) 내림차순으로 정렬하고, 매번
        score(fold) = W_ALT   * (해당 altitude_group의 fold 누적 이미지 수)
                    + W_REGION* (해당 (altitude_group, region)의 fold 누적 이미지 수)   [onion/garlic만]
                    + W_YEAR  * (해당 (altitude_group, year)의 fold 누적 이미지 수)
    가 가장 작은 fold에 그 그룹을 배정합니다. W_ALT >> W_REGION >> W_YEAR로 두어
    "altitude를 절대 우선, 그 다음 region, 그 다음 year" 순서를 강제합니다.

  - onion_garlic: region labels are relatively clear (distinct collection
    sites), so the region term is included in the score to balance regions.
  - cabbage_radish / background: region is ambiguous, so it is excluded from
    the score and only reported (year is still reflected).

## 사용법
python3 dataset_make_master_5fold.py \
  --json-dir /path/to/dataset/data3/all/json/real \
  --output   /path/to/dataset/data3/all/master_5fold.csv \
  --k-folds 5

## 산출물 사용 방법
- 전체고도 모델: master_5fold.csv를 fold 컬럼 기준으로 그대로 train/val 분리
- 고도별 특화 모델: master_5fold.csv를 altitude_group으로 먼저 filter한 뒤,
  동일한 fold 컬럼으로 train/val 분리 (즉 fold 배정 자체는 항상 이 파일 하나가 기준)
"""

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

CLASS_NAMES = {0: "Cabbage", 1: "Radish", 2: "Onion", 3: "Garlic"}
ONION_GARLIC = {"Onion", "Garlic"}
# Known collection-site folder names used for region-balanced fold assignment.
# Anonymized for public release — set these to your actual site folder names.
KNOWN_REGIONS = {"region_a", "region_b", "region_c"}
CABBAGE_RADISH = {"Cabbage", "Radish"}

_ALT_LOW = frozenset({"5m", "10m", "20m", "30m"})
_ALT_MEDIUM = frozenset({"40m", "50m", "60m"})
_ALT_HIGH = frozenset({"70m", "80m", "90m", "100m"})

# 우선순위 altitude > region > year 를 강제하기 위한 가중치 (자릿수 차이로 완전히 분리)
W_ALT = 1_000_000
W_REGION = 1_000
W_YEAR = 1


def _altitude_bucket(alt_token: Optional[str]) -> str:
    if alt_token is None:
        return "unknown"
    if alt_token in ("low", "medium", "high"):
        return alt_token
    if alt_token in _ALT_LOW:
        return "low"
    if alt_token in _ALT_MEDIUM:
        return "medium"
    if alt_token in _ALT_HIGH:
        return "high"
    return "unknown"


def _extract_altitude_token(path_str: str) -> Optional[str]:
    parts = re.split(r"[\\/]", path_str)
    for p in parts:
        pl = p.strip().lower()
        if pl in ("low", "medium", "high"):
            return pl
        if re.fullmatch(r"\d+m", pl, flags=re.IGNORECASE):
            return pl
    fname = os.path.basename(path_str)
    stem = os.path.splitext(fname)[0]
    m = re.search(r"_(\d+)m(?:[_.]|$)", fname, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}m"
    m = re.search(r"_(\d+)m$", stem, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}m"
    return None


def _extract_date_token(path_str: str) -> Optional[str]:
    fname = os.path.basename(path_str)
    m = re.search(r"20\d{2}[01]\d[0-3]\d", fname)
    if m:
        return m.group(0)
    m = re.match(r"^(\d{2})([01]\d)([0-3]\d)_", fname)
    if m:
        yy, mm, dd = m.groups()
        if yy in ("22", "23", "24", "25"):
            return f"20{yy}{mm}{dd}"
    return None


def _extract_location_token(fname: str) -> Optional[str]:
    stem = os.path.splitext(fname)[0]
    for seg in stem.split("_"):
        if re.search(r"[가-힣]", seg):
            return seg.strip()
    return None


def _extract_year_token(name: str, date_token: Optional[str]) -> str:
    m = re.match(r"^(20\d{2})_", name)
    if m:
        return m.group(1)
    m = re.match(r"^(\d{2})_", name)
    if m:
        return f"20{m.group(1)}"
    if date_token:
        return date_token[:4]
    return "unknown"


def _extract_region(path_str: str) -> str:
    """Resolve region: match a known collection site from the folder structure
    first; otherwise try an administrative-area token in the filename; otherwise
    fall back to the folder directly under 'newdata'."""
    if not path_str:
        return "unknown"
    parts = re.split(r"[\\/]", path_str)
    top = None
    if "newdata" in parts:
        idx = parts.index("newdata")
        if idx + 1 < len(parts):
            top = parts[idx + 1]
    if top in KNOWN_REGIONS:
        return top
    fname = os.path.basename(path_str)
    m = re.search(r"([가-힣]{2,10}(?:시|군|구))", fname)
    if m:
        return m.group(1)[:-1]  # strip admin suffix (e.g. "-gun") to match folder tag
    return top or "unknown"


def build_records(json_dir: str) -> List[Dict]:
    records = []
    for fn in sorted(os.listdir(json_dir)):
        if not fn.lower().endswith(".json"):
            continue
        name = os.path.splitext(fn)[0]
        with open(os.path.join(json_dir, fn), "r", encoding="utf-8") as f:
            data = json.load(f)

        images = {im["id"]: im for im in data["images"]}
        area_by_image_cat: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for ann in data["annotations"]:
            area_by_image_cat[ann["image_id"]][ann["category_id"]] += float(ann.get("area", 0) or 0)

        for image_id, im in images.items():
            path = im.get("path", "") or ""
            basis = path or im["file_name"]
            alt_token = _extract_altitude_token(basis)
            altitude_group = _altitude_bucket(alt_token)
            date_token = _extract_date_token(basis)
            location_token = _extract_location_token(os.path.basename(basis))
            year = _extract_year_token(name, date_token)
            region = _extract_region(path)

            cats = area_by_image_cat.get(image_id, {})
            if cats:
                dom_cat = max(cats.items(), key=lambda kv: kv[1])[0]
                crop_class = CLASS_NAMES.get(dom_cat, f"cls{dom_cat}")
            else:
                crop_class = "background"

            if crop_class in ONION_GARLIC:
                crop_group = "onion_garlic"
            elif crop_class in CABBAGE_RADISH:
                crop_group = "cabbage_radish"
            else:
                crop_group = "background"

            # altitude_group을 그룹 키에 포함시키는 이유:
            # 같은 날 같은 지번이라도 여러 고도(5m/10m/.../100m)를 한 세션에서 다 찍는 경우가 많아서,
            # 고도를 그룹 키에서 빼면 "고도가 섞인 그룹"이 통째로 한 fold에 배정되어 fold별
            # 고도 분포를 정밀하게 맞출 수가 없음. 반면 실제 leakage 위험(거의 동일한 사진)은
            # "같은 고도 + 같은 패스"에서 연속 촬영된 프레임 사이에서 생기지, 같은 지점을 다른
            # 고도(예: 10m vs 100m)에서 찍은 사진끼리는 화각/디테일이 완전히 달라 leakage 위험이 낮음.
            if location_token or date_token:
                group_key = f"{name}|{location_token or ''}|{date_token or ''}|{altitude_group}"
            elif path:
                group_key = f"{name}|{os.path.dirname(path)}|{altitude_group}"
            else:
                group_key = f"{name}|{altitude_group}"

            stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
            image_uid = f"{name}_{stem}"

            records.append({
                "image_id": image_uid,
                "collection": name,
                "crop_group": crop_group,
                "crop_class": crop_class,
                "altitude_group": altitude_group,
                "region": region,
                "year": year,
                "group_key": group_key,
            })
    return records


# 그룹 하나가 이 크기를 넘으면 CHUNK_SIZE 단위로 쪼갠다.
# (e.g. a single day's capture at one site can form one very large group -> if
#  left as-is, the whole group lands in one fold and region/altitude balance breaks.
#  촬영 순서(=파일 나열 순서, 대체로 연속 프레임)대로 잘라서 쪼개면, 인접 프레임끼리는
#  여전히 같은 fold에 남아 leakage를 막으면서도 fold 배정을 훨씬 유연하게 만들 수 있음.)
MAX_GROUP_SIZE = 80
CHUNK_SIZE = 50


def _split_oversized_groups(groups: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    chunked: Dict[str, List[Dict]] = {}
    for gk, recs in groups.items():
        if len(recs) <= MAX_GROUP_SIZE:
            chunked[gk] = recs
            continue
        for i in range(0, len(recs), CHUNK_SIZE):
            chunk = recs[i:i + CHUNK_SIZE]
            chunked[f"{gk}#chunk{i // CHUNK_SIZE}"] = chunk
    return chunked


def aggregate_groups(records: List[Dict]):
    raw_groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        raw_groups[r["group_key"]].append(r)

    groups = _split_oversized_groups(raw_groups)
    # 쪼갠 결과를 각 record에도 반영해야 CSV의 fold가 실제 배정 단위와 일치함
    key_to_chunked_key = {}
    for gk, recs in groups.items():
        for r in recs:
            key_to_chunked_key[id(r)] = gk
    for r in records:
        r["group_key"] = key_to_chunked_key[id(r)]

    group_info = {}
    for gk, recs in groups.items():
        def majority(key):
            return Counter(r[key] for r in recs).most_common(1)[0][0]

        group_info[gk] = {
            "size": len(recs),
            "altitude_group": majority("altitude_group"),
            "region": majority("region"),
            "year": majority("year"),
            "crop_group": majority("crop_group"),
        }
    return groups, group_info


def assign_folds(group_info: Dict[str, Dict], k: int, seed: int = 42) -> Dict[str, int]:
    rnd = random.Random(seed)

    fold_alt_total: Dict[str, List[int]] = defaultdict(lambda: [0] * k)
    fold_region_total: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0] * k)
    fold_year_total: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0] * k)

    # 그룹 크기 내림차순(LPT) + 동률은 무작위(재현 가능하게 seed 고정)로 섞어서 정렬
    keys = list(group_info.keys())
    rnd.shuffle(keys)
    keys.sort(key=lambda gk: -group_info[gk]["size"])

    fold_of_group: Dict[str, int] = {}
    for gk in keys:
        info = group_info[gk]
        alt, region, year, crop_group = info["altitude_group"], info["region"], info["year"], info["crop_group"]

        best_fold, best_score = 0, None
        for f in range(k):
            score = W_ALT * fold_alt_total[alt][f]
            if crop_group == "onion_garlic":
                score += W_REGION * fold_region_total[(alt, region)][f]
            score += W_YEAR * fold_year_total[(alt, year)][f]
            if best_score is None or score < best_score:
                best_score = score
                best_fold = f

        fold_of_group[gk] = best_fold
        size = info["size"]
        fold_alt_total[alt][best_fold] += size
        fold_region_total[(alt, region)][best_fold] += size
        fold_year_total[(alt, year)][best_fold] += size

    return fold_of_group


def print_distribution(records: List[Dict], key: str, k: int, title: str):
    table: Dict[str, List[int]] = defaultdict(lambda: [0] * k)
    for r in records:
        table[r[key]][r["fold"]] += 1

    print(f"\n[분포] {title}")
    header = f"{'':<14}" + "".join(f"fold{f+1:>8}" for f in range(k)) + f"{'total':>10}"
    print(header)
    for val in sorted(table.keys()):
        counts = table[val]
        total = sum(counts)
        row = f"{val:<14}" + "".join(f"{c:>8}" for c in counts) + f"{total:>10}"
        print(row)
    col_totals = [sum(table[v][f] for v in table) for f in range(k)]
    print(f"{'TOTAL':<14}" + "".join(f"{c:>8}" for c in col_totals) + f"{sum(col_totals):>10}")


def main():
    p = argparse.ArgumentParser(description="data3/all master 5-fold split 생성")
    p.add_argument("--json-dir", default="/path/to/dataset/data3/all/json/real")
    p.add_argument("--output", default="/path/to/dataset/data3/all/master_5fold.csv")
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[Master Fold] json-dir: {args.json_dir}")
    records = build_records(args.json_dir)
    print(f"[Master Fold] 전체 이미지 수: {len(records)}")

    groups, group_info = aggregate_groups(records)
    print(f"[Master Fold] 고유 그룹(촬영 세션) 수: {len(group_info)}")

    # 그룹별 altitude 다양성 체크 (각 fold에 low/mid/high가 모두 들어가려면 그룹이 충분해야 함)
    alt_group_counts = Counter(info["altitude_group"] for info in group_info.values())
    print(f"[Master Fold] altitude_group별 그룹 수: {dict(alt_group_counts)}")
    for alt in ("low", "medium", "high"):
        if alt_group_counts.get(alt, 0) < args.k_folds:
            print(f"  ⚠️  경고: altitude_group={alt} 의 그룹 수({alt_group_counts.get(alt,0)})가 "
                  f"fold 수({args.k_folds})보다 적어서 일부 fold에 이 고도가 안 들어갈 수 있습니다.")

    fold_of_group = assign_folds(group_info, k=args.k_folds, seed=args.seed)
    for r in records:
        r["fold"] = fold_of_group[r["group_key"]]

    # low/mid/high가 fold마다 다 있는지 최종 검증
    print(f"\n[Master Fold] fold별 altitude_group 존재 여부 검증:")
    ok = True
    for f in range(args.k_folds):
        alts_in_fold = {r["altitude_group"] for r in records if r["fold"] == f}
        missing = {"low", "medium", "high"} - alts_in_fold
        status = "OK" if not missing else f"⚠️ 없음: {missing}"
        print(f"  fold {f+1}: {sorted(alts_in_fold)}  -> {status}")
        if missing:
            ok = False
    if ok:
        print("  ✅ 모든 fold에 low/medium/high 고도가 전부 존재합니다.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # collection은 요청받은 필수 컬럼은 아니지만, 학습 스크립트가 실제 이미지/마스크 파일을
    # 찾으려면 어느 <name> 폴더 소속인지 알아야 해서 추가함 (image_id만으로는 역추적이 애매함:
    # image_id = f"{name}_{stem}" 형태인데 name과 stem 둘 다 '_'를 포함할 수 있어서 분리 불가).
    fieldnames = ["image_id", "collection", "crop_group", "crop_class", "altitude_group", "region", "year", "fold"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\n✅ 저장 완료: {args.output} ({len(records)}행)")

    print_distribution(records, "altitude_group", args.k_folds, "altitude_group x fold")
    print_distribution(records, "region", args.k_folds, "region x fold")
    print_distribution(records, "year", args.k_folds, "year x fold")
    print_distribution(records, "crop_class", args.k_folds, "crop_class x fold")
    print_distribution(records, "crop_group", args.k_folds, "crop_group x fold")


if __name__ == "__main__":
    main()
