"""
Gold 특징 저장소: Silver에서 경량 Feature Store 추출

Silver Parquet에서 wafer_map_flat(바이너리)을 제거하고
공간 특징 컬럼만 남긴 경량 Feature Store를 생성.
(공간 특징은 Bronze 수집 시 이미 numpy로 계산됨)

출력:
  data/lake/gold/feature_store/features.parquet
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from pyspark.sql import SparkSession


# Bronze에서 이미 계산된 공간 특징 컬럼
SPATIAL_FEATURES = [
    "defect_density",
    "active_ratio",
    "edge_defect_ratio",
    "center_defect_ratio",
    "radial_mean",
    "defect_count",
    "active_count",
]


def build_feature_store(spark: SparkSession, silver_path: str, out_path: str):
    """
    Silver Parquet → Feature Store (wafer_map_flat 제외).

    전략:
    - 공간 특징은 Bronze ingestion 시 numpy로 이미 계산
    - 여기서는 wafer_map_flat 제거 + ML 관련 컬럼 선택만 수행
    - Python UDF 없이 순수 pandas 컬럼 선택
    """
    print("[Gold/features] Building feature store...")

    silver_file = _find_parquet(silver_path)
    pdf         = pd.read_parquet(silver_file)

    # wafer_map_flat 제거 → 경량 Feature Store
    drop_cols = [c for c in pdf.columns if c == "wafer_map_flat"]
    feat_pdf  = pdf.drop(columns=drop_cols)

    print(f"  Silver: {len(pdf):,} rows | Feature Store: {len(feat_pdf.columns)} 컬럼")
    print(f"  컬럼: {list(feat_pdf.columns)}")

    # ── PyArrow로 단일 파일 쓰기 ───────────────────────────────────────
    out_dir  = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / "features.parquet")

    table = pa.Table.from_pandas(feat_pdf, preserve_index=False)
    pq.write_table(table, out_file, compression="snappy")

    print(f"[Gold/features] 저장 완료 → {out_file}")
    print(f"  파일 크기: {Path(out_file).stat().st_size / 1024**2:.1f} MB")

    # ── Spark로 읽어 클래스별 특징 통계 출력 ──────────────────────────
    from pyspark.sql import functions as F
    sdf = spark.read.parquet(out_file)

    print(f"\n[Gold/features] 클래스별 평균 불량 밀도 (Spark SQL):")
    agg_cols = [c for c in SPATIAL_FEATURES if c in sdf.columns]
    agg_exprs = [F.avg(c).alias(f"avg_{c}") for c in agg_cols[:4]]  # 주요 4개
    (sdf.groupBy("failure_type")
        .agg(F.count("*").alias("n"), *agg_exprs)
        .orderBy("avg_defect_density", ascending=False)
        .show(truncate=False))

    return sdf


def _find_parquet(path: str) -> str:
    p = Path(path)
    if p.suffix == ".parquet" and p.is_file():
        return str(p)
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if files:
            return str(files[0])
    raise FileNotFoundError(f"Parquet 파일을 찾을 수 없음: {path}")
