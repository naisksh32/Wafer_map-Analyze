r"""
WM-811K 웨이퍼 데이터 Spark 파이프라인 — 메인 오케스트레이터

실행:
  cd "c:\Users\naisk\Desktop\하이닉스 대비\웨이퍼 불량 분석"
  .venv\Scripts\python.exe spark_pipeline\run_pipeline.py [--step STEP]

  STEP 옵션:
    all       - 전체 파이프라인 (기본값)
    bronze    - Bronze 수집만
    silver    - Silver 변환만 (Bronze 완료 후)
    gold      - Gold 분할만 (Silver 완료 후)
    features  - Feature Store + SPC만 (Silver 완료 후)

흐름:
  [Bronze] npy + csv → Parquet
      ↓
  [Silver] JOIN + 공간통계 → Parquet (failure_type 파티셔닝)
      ↓
  [Gold] Split + Feature Store + SPC 집계 → Parquet
"""
import sys
import time
import argparse
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "spark_pipeline"))

from config.pipeline_config import get_spark, PATHS, spark_read_parquet
from ingestion.npy_to_bronze import npy_to_bronze
from ingestion.csv_to_bronze import csv_to_bronze
from transform.bronze_to_silver import bronze_to_silver
from transform.silver_to_gold import silver_to_gold
from feature_engineering.spatial_features import build_feature_store
from feature_engineering.spc_aggregation import build_spc_statistics


def run_bronze(spark):
    print("=" * 60)
    print("STEP 1: BRONZE 계층 수집")
    print("=" * 60)

    t = time.time()
    npy_to_bronze(
        spark,
        maps_path      = PATHS["maps_npy"],
        split_pkl_path = PATHS["split_pkl"],
        out_path       = PATHS["bronze_maps"],
    )
    print(f"  npy_to_bronze: {time.time()-t:.1f}s\n")

    t = time.time()
    csv_to_bronze(
        spark,
        csv_path = PATHS["process_csv"],
        out_path = PATHS["bronze_params"],
    )
    print(f"  csv_to_bronze: {time.time()-t:.1f}s\n")


def run_silver(spark):
    print("=" * 60)
    print("STEP 2: SILVER 계층 변환")
    print("=" * 60)
    t = time.time()
    bronze_to_silver(
        spark,
        bronze_maps_path   = PATHS["bronze_maps"],
        bronze_params_path = PATHS["bronze_params"],
        out_path           = PATHS["silver_labeled"],
    )
    print(f"  bronze_to_silver: {time.time()-t:.1f}s\n")


def run_gold(spark):
    print("=" * 60)
    print("STEP 3: GOLD 계층 분할")
    print("=" * 60)
    t = time.time()
    silver_to_gold(
        spark,
        silver_path = PATHS["silver_labeled"],
        train_path  = PATHS["gold_train"],
        val_path    = PATHS["gold_val"],
        test_path   = PATHS["gold_test"],
    )
    print(f"  silver_to_gold: {time.time()-t:.1f}s\n")


def run_features(spark):
    print("=" * 60)
    print("STEP 4: FEATURE STORE + SPC 집계")
    print("=" * 60)

    t = time.time()
    build_feature_store(
        spark,
        silver_path = PATHS["silver_labeled"],
        out_path    = PATHS["gold_features"],
    )
    print(f"  build_feature_store: {time.time()-t:.1f}s\n")

    t = time.time()
    build_spc_statistics(
        spark,
        silver_path = PATHS["silver_labeled"],
        out_path    = PATHS["gold_spc"],
    )
    print(f"  build_spc_statistics: {time.time()-t:.1f}s\n")


def print_lake_summary(spark):
    """데이터 레이크 저장 현황 출력."""
    from pathlib import Path
    print("\n" + "=" * 60)
    print("DATA LAKE 저장 현황")
    print("=" * 60)

    lake_root = _ROOT / "data" / "lake"
    total_bytes = 0
    for layer in ["bronze", "silver", "gold"]:
        layer_dir = lake_root / layer
        if not layer_dir.exists():
            continue
        for path in sorted(layer_dir.rglob("*.parquet")):
            size = path.stat().st_size
            total_bytes += size
            rel = path.relative_to(lake_root)
            print(f"  {str(rel):<60} {size/1024/1024:>7.1f} MB")
    print(f"\n  총계: {total_bytes/1024/1024:.1f} MB")

    # Spark로 Gold 분할 통계 조회
    try:
        train = spark_read_parquet(spark, PATHS["gold_train"])
        val   = spark_read_parquet(spark, PATHS["gold_val"])
        test  = spark_read_parquet(spark, PATHS["gold_test"])
        print(f"\n  Gold 분할: Train={train.count():,} / "
              f"Val={val.count():,} / Test={test.count():,}")
    except Exception as e:
        print(f"  Gold 통계 오류: {e}")


def main():
    parser = argparse.ArgumentParser(description="WM-811K Spark 데이터 파이프라인")
    parser.add_argument("--step", default="all",
                        choices=["all","bronze","silver","gold","features"],
                        help="실행할 파이프라인 단계")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("WM-811K WAFER SPARK PIPELINE")
    print(f"Step: {args.step.upper()}")
    print("=" * 60 + "\n")

    spark = get_spark("WaferPipeline")
    print(f"Spark {spark.version} 세션 시작\n")

    total_start = time.time()

    try:
        if args.step in ("all", "bronze"):
            run_bronze(spark)
        if args.step in ("all", "silver"):
            run_silver(spark)
        if args.step in ("all", "gold"):
            run_gold(spark)
        if args.step in ("all", "features"):
            run_features(spark)

        if args.step == "all":
            print_lake_summary(spark)

    finally:
        spark.stop()

    print(f"\n총 소요 시간: {time.time()-total_start:.1f}s")
    print("파이프라인 완료!")


if __name__ == "__main__":
    main()
