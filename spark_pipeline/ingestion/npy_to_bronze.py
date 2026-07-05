"""
Bronze 계층 수집 1: all_maps_resized.npy + split_indices.pkl → Parquet

입력:
  data/processed/all_maps_resized.npy  (172950, 64, 64) uint8
  data/processed/split_indices.pkl     {train_idx, val_idx, test_idx, encoded_labels, ...}

출력:
  data/lake/bronze/wafer_maps/wafer_maps_bronze.parquet  (172,950 레코드)

스키마:
  wafer_id             INT     - 0-based 인덱스 (process_params JOIN 키)
  class_idx            INT     - 0~8 인코딩된 클래스
  failure_type         STRING  - 결함 유형 문자열
  split                STRING  - train / val / test
  wafer_map_flat       BINARY  - 64×64 = 4096 bytes (uint8 직렬화)
  defect_density       DOUBLE  - 불량 다이(pixel=2) 비율 (active 기준)
  active_ratio         DOUBLE  - 활성 다이(pixel>0) 비율 (전체 64×64 기준)
  edge_defect_ratio    DOUBLE  - 에지 영역(반경 75%↑) 내 불량 비율
  center_defect_ratio  DOUBLE  - 중심 영역(반경 38%↓) 내 불량 비율
  radial_mean          DOUBLE  - 불량 다이의 평균 반경 거리
  defect_count         INT     - 불량 다이 수 (pixel=2)
  active_count         INT     - 활성 다이 수 (pixel>0)
"""
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from pyspark.sql import SparkSession

# 64×64 반경 마스크 (전역 사전 계산)
_CX, _CY, _R = 32.0, 32.0, 31.5
_Y, _X = np.ogrid[:64, :64]
_DIST  = np.sqrt((_X - _CX)**2 + (_Y - _CY)**2)
_EDGE_MASK   = _DIST > (_R * 0.75)
_CENTER_MASK = _DIST < (_R * 0.38)


def _spatial_features(flat_bytes: bytes):
    """numpy로 7차원 공간 특징 계산 (Python UDF 없이)."""
    arr    = np.frombuffer(flat_bytes, dtype=np.uint8)
    arr2d  = arr.reshape(64, 64)
    active = (arr > 0).sum()
    defects = (arr == 2).sum()

    if active == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0

    defect_mask   = arr2d == 2
    total_d       = int(defects)
    density       = float(defects / active)
    act_ratio     = float(active / 4096)
    edge_ratio    = float((defect_mask & _EDGE_MASK).sum()   / (total_d + 1e-9))
    center_ratio  = float((defect_mask & _CENTER_MASK).sum() / (total_d + 1e-9))
    radial_mean   = float(_DIST[defect_mask].mean()) if total_d > 0 else 0.0

    return density, act_ratio, edge_ratio, center_ratio, radial_mean, total_d, int(active)


def npy_to_bronze(spark: SparkSession, maps_path: str, split_pkl_path: str, out_path: str):
    """
    172,950개 웨이퍼 맵을 Bronze Parquet으로 저장.

    전략: Python UDF 및 Spark DataFrame.write 대신
    numpy로 공간 특징을 계산하고 PyArrow로 단일 파일 쓰기.
    (Windows PySpark Python worker 충돌 회피)
    """
    print("[Bronze/maps] Loading preprocessed maps...")
    maps = np.load(maps_path, mmap_mode="r")   # (172950, 64, 64)
    n_samples = maps.shape[0]

    with open(split_pkl_path, "rb") as f:
        split_info = pickle.load(f)

    encoded_labels = np.array(split_info["encoded_labels"])
    idx_to_label   = {v: k for k, v in split_info["label_map"].items()}

    split_map: dict[int, str] = {}
    for idx in split_info["train_idx"]: split_map[int(idx)] = "train"
    for idx in split_info["val_idx"]:   split_map[int(idx)] = "val"
    for idx in split_info["test_idx"]:  split_map[int(idx)] = "test"

    n_train = len(split_info["train_idx"])
    n_val   = len(split_info["val_idx"])
    n_test  = len(split_info["test_idx"])
    print(f"  총 {n_samples:,}개 | Train {n_train:,} / Val {n_val:,} / Test {n_test:,}")

    # ── 배치 처리: numpy로 특징 계산 ──────────────────────────────────
    batch_size = 10_000
    records = []

    for start in range(0, n_samples, batch_size):
        end   = min(start + batch_size, n_samples)
        batch = maps[start:end]

        for li in range(end - start):
            gi        = start + li
            flat      = batch[li].flatten().tobytes()
            dens, act_r, edge_r, ctr_r, rad, d_cnt, a_cnt = _spatial_features(flat)
            label_idx = int(encoded_labels[gi])
            records.append({
                "wafer_id":            gi,
                "class_idx":           label_idx,
                "failure_type":        idx_to_label[label_idx],
                "split":               split_map.get(gi, "train"),
                "wafer_map_flat":      flat,
                "defect_density":      dens,
                "active_ratio":        act_r,
                "edge_defect_ratio":   edge_r,
                "center_defect_ratio": ctr_r,
                "radial_mean":         rad,
                "defect_count":        d_cnt,
                "active_count":        a_cnt,
            })

        if (start // batch_size) % 5 == 0:
            print(f"  [{end:>6}/{n_samples}] 처리 중...")

    # ── PyArrow로 단일 파일 쓰기 ───────────────────────────────────────
    print(f"[Bronze/maps] PyArrow로 Parquet 저장 중 ({len(records):,}행)...")
    pdf = pd.DataFrame(records)

    out_dir  = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / "wafer_maps_bronze.parquet")

    table = pa.Table.from_pandas(pdf, preserve_index=False)
    pq.write_table(table, out_file, compression="snappy")

    print(f"[Bronze/maps] 저장 완료 → {out_file}")
    print(f"  파일 크기: {Path(out_file).stat().st_size / 1024**2:.1f} MB")

    # ── Spark로 읽어 클래스 분포 출력 ──────────────────────────────────
    from pyspark.sql import functions as F
    sdf = spark.read.parquet(out_file)
    print(f"\n[Bronze/maps] 클래스 분포 (Spark SQL):")
    total = sdf.count()
    (sdf.groupBy("failure_type")
        .agg(F.count("*").alias("count"),
             F.avg("defect_density").alias("avg_density"))
        .withColumn("비율%", F.round(F.col("count") / total * 100, 2))
        .orderBy("count", ascending=False)
        .show(truncate=False))

    return sdf
