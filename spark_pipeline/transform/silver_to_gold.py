"""
Gold 계층: Silver → train / val / test 분할 저장

기존 split_indices.pkl의 분할 인덱스를 그대로 재사용하여
ML 파이프라인(05_finetuning.ipynb 등)과 일관성 보장.

출력:
  data/lake/gold/train/train.parquet         (121,065 rows)
  data/lake/gold/validation/val.parquet      (25,942 rows)
  data/lake/gold/test/test.parquet           (25,943 rows)
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from pyspark.sql import SparkSession


def silver_to_gold(
    spark: SparkSession,
    silver_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
):
    """
    Silver에서 split 컬럼 기준으로 Gold 계층 분할.

    전략: pandas 필터링 + PyArrow 쓰기.
    분할 기준은 Bronze 수집 시 split_indices.pkl로 이미 split 컬럼에 저장됨.
    """
    print("[Gold] Loading Silver layer (pandas)...")

    silver_file = _find_parquet(silver_path)
    silver_pdf  = pd.read_parquet(silver_file)

    print(f"  Silver: {len(silver_pdf):,} rows, {len(silver_pdf.columns)} 컬럼")

    splits = {
        "train": train_path,
        "val":   val_path,
        "test":  test_path,
    }

    for split_name, out_path in splits.items():
        sdf_pd = silver_pdf[silver_pdf["split"] == split_name].reset_index(drop=True)

        out_dir  = Path(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = str(out_dir / f"{split_name}.parquet")

        table = pa.Table.from_pandas(sdf_pd, preserve_index=False)
        pq.write_table(table, out_file, compression="snappy")

        mb = Path(out_file).stat().st_size / 1024**2
        print(f"[Gold] {split_name:5s}: {len(sdf_pd):>7,} rows → {out_file} ({mb:.1f} MB)")

    # ── Spark로 읽어 분할별 클래스 분포 확인 ─────────────────────────
    print("\n[Gold] 분할별 클래스 분포 (Spark SQL):")
    from pyspark.sql import functions as F
    from config.pipeline_config import spark_read_parquet

    train_sdf = spark_read_parquet(spark, train_path)
    val_sdf   = spark_read_parquet(spark, val_path)
    test_sdf  = spark_read_parquet(spark, test_path)

    for name, sdf in [("train", train_sdf), ("val", val_sdf), ("test", test_sdf)]:
        top5 = (sdf.groupBy("failure_type")
                   .count()
                   .orderBy("count", ascending=False)
                   .limit(5)
                   .collect())
        print(f"  [{name}] " + ", ".join(f"{r['failure_type']}:{r['count']:,}" for r in top5))

    return True


def _find_parquet(path: str) -> str:
    p = Path(path)
    if p.suffix == ".parquet" and p.is_file():
        return str(p)
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if files:
            return str(files[0])
    raise FileNotFoundError(f"Parquet 파일을 찾을 수 없음: {path}")
