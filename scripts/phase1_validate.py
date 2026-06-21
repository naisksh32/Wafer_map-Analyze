"""Phase 1 - 데이터 로드 검증 및 기본 통계 생성 스크립트"""

import pickle
import pandas as pd
import numpy as np
import json
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

PKL_PATH = Path(__file__).parent.parent / 'data' / 'raw' / 'LSWMD.pkl'
OUTPUT_DIR = Path(__file__).parent.parent

# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────
print("=" * 55)
print("[1] WM-811K 데이터 로드 중 (약 30초)...")
print("=" * 55)
df = pd.read_pickle(PKL_PATH)
print(f"  로드 완료: {df.shape[0]:,}행 x {df.shape[1]}열")
print(f"  컬럼: {list(df.columns)}")

# ─────────────────────────────────────────
# 2. 레이블 정제 (중첩 리스트 해제)
# ─────────────────────────────────────────
print("\n[2] 레이블 정제 (중첩 리스트 해제)...")

def extract_label(val):
    if isinstance(val, (list, np.ndarray)):
        flat = np.array(val).flatten()
        return str(flat[0]) if len(flat) > 0 else 'unknown'
    return str(val)

df['failureType_clean'] = df['failureType'].apply(extract_label)
df['trianTestLabel_clean'] = df['trianTestLabel'].apply(extract_label)
print(f"  고유 failureType 값: {sorted(df['failureType_clean'].unique())}")
print(f"  고유 trianTestLabel 값: {sorted(df['trianTestLabel_clean'].unique())}")

# ─────────────────────────────────────────
# 3. 전체 통계
# ─────────────────────────────────────────
print("\n[3] 전체 통계")
print("-" * 55)
labeled_mask = df['failureType_clean'] != '0'
n_total = len(df)
n_labeled = labeled_mask.sum()
n_unlabeled = (~labeled_mask).sum()
print(f"  전체 샘플 수:      {n_total:>8,}")
print(f"  레이블 있는 샘플:  {n_labeled:>8,}  ({n_labeled/n_total*100:.1f}%)")
print(f"  레이블 없는 샘플:  {n_unlabeled:>8,}  ({n_unlabeled/n_total*100:.1f}%)")

# ─────────────────────────────────────────
# 4. 클래스 분포 (레이블 있는 데이터만)
# ─────────────────────────────────────────
print("\n[4] 클래스 분포 (레이블 있는 데이터만)")
print("-" * 55)
labeled_df = df[labeled_mask]
dist = labeled_df['failureType_clean'].value_counts()
print(f"  {'클래스':<14} {'샘플 수':>8}  {'비율':>7}")
print(f"  {'-'*35}")
for label, cnt in dist.items():
    pct = cnt / n_labeled * 100
    print(f"  {label:<14} {cnt:>8,}  {pct:>6.2f}%")

# ─────────────────────────────────────────
# 5. Train/Test 분포
# ─────────────────────────────────────────
print("\n[5] Train/Test 레이블 분포")
print("-" * 55)
print(labeled_df['trianTestLabel_clean'].value_counts().to_string())

# ─────────────────────────────────────────
# 6. 웨이퍼 맵 크기 분포
# ─────────────────────────────────────────
print("\n[6] 웨이퍼 맵 크기 분포 (상위 10개)")
print("-" * 55)
shapes = labeled_df['waferMap'].apply(
    lambda x: x.shape if hasattr(x, 'shape') else None
).dropna()
shape_dist = shapes.value_counts()
print(f"  {'크기 (H x W)':<18} {'샘플 수':>8}  {'비율':>7}")
print(f"  {'-'*35}")
for shape, cnt in shape_dist.head(10).items():
    pct = cnt / len(shapes) * 100
    print(f"  {str(shape):<18} {cnt:>8,}  {pct:>6.2f}%")

# 웨이퍼 맵 픽셀값 확인
sample_map = labeled_df['waferMap'].iloc[0]
print(f"\n  픽셀값 예시 (고유값): {np.unique(sample_map)}")
print(f"  픽셀값 설명: 0=빈 영역, 1=정상 다이, 2=불량 다이")

# ─────────────────────────────────────────
# 7. 클래스 불균형 분석
# ─────────────────────────────────────────
print("\n[7] 클래스 불균형 분석")
print("-" * 55)
max_cnt = dist.max()
min_cnt = dist.min()
imbalance_ratio = max_cnt / min_cnt
print(f"  최대 클래스: {dist.index[0]} ({max_cnt:,}개)")
print(f"  최소 클래스: {dist.index[-1]} ({min_cnt:,}개)")
print(f"  Imbalance Ratio: {imbalance_ratio:.1f}x")

# ─────────────────────────────────────────
# 8. data_summary.json 저장
# ─────────────────────────────────────────
print("\n[8] data_summary.json 저장 중...")

summary = {
    "dataset": "WM-811K",
    "source": "Kaggle - qingyi/wm811k-wafer-map",
    "pkl_path": PKL_PATH,
    "total_samples": int(n_total),
    "labeled_samples": int(n_labeled),
    "unlabeled_samples": int(n_unlabeled),
    "label_ratio": round(n_labeled / n_total * 100, 2),
    "num_classes": int(len(dist)),
    "class_distribution": {k: int(v) for k, v in dist.items()},
    "imbalance_ratio": round(float(imbalance_ratio), 2),
    "wafer_map_top5_shapes": [str(s) for s in shape_dist.head(5).index.tolist()],
    "pixel_values": {"0": "empty_area", "1": "normal_die", "2": "defective_die"},
    "columns": list(df.columns),
}

summary_path = OUTPUT_DIR / 'data_summary.json'
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"  저장 완료: {summary_path}")

print("\n" + "=" * 55)
print("Phase 1 검증 완료!")
print(f"  DATA_PATH = r'{PKL_PATH}'")
print("  다음 단계: 02_eda.ipynb")
print("=" * 55)
