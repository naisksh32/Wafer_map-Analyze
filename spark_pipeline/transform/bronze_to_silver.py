"""
Silver 계층: Bronze wafer_maps + Bronze process_params → Silver Parquet

변환 내용:
  1. 두 Bronze 테이블을 wafer_id 기준 LEFT JOIN (pandas merge)
  2. 공간 특징은 Bronze에서 이미 계산됨 (UDF 불필요)
  3. any_ooc 통합 플래그 추가
  4. PyArrow 단일 파일로 Silver 저장

출력:
  data/lake/silver/wafer_labeled/wafer_silver.parquet
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from pyspark.sql import SparkSession


def bronze_to_silver(
    spark: SparkSession,
    bronze_maps_path: str,
    bronze_params_path: str,
    out_path: str,
):
    """
    Bronze → Silver: pandas LEFT JOIN + PyArrow 저장.

    전략: Spark DataFrame.join 대신 pandas merge 사용.
    이유: Python UDF 없이 join만 필요 → pandas가 172k 행에서 충분히 빠름.
    Spark는 분석 통계 출력 전용으로 활용.
    """
    print("[Silver] Loading Bronze layers (pandas)...")

    # ── pandas로 읽기 ──────────────────────────────────────────────────
    maps_file   = _find_parquet(bronze_maps_path)
    params_file = _find_parquet(bronze_params_path)

    maps_pdf   = pd.read_parquet(maps_file)
    params_pdf = pd.read_parquet(params_file)

    print(f"  Maps:   {len(maps_pdf):,} rows")
    print(f"  Params: {len(params_pdf):,} rows")

    # ── LEFT JOIN on wafer_id ──────────────────────────────────────────
    # params에서 중복 컬럼 제거 (defect_class, is_defect는 maps와 겹칠 수 있음)
    drop_cols = [c for c in ("defect_class", "is_defect") if c in params_pdf.columns]
    silver_pdf = maps_pdf.merge(
        params_pdf.drop(columns=drop_cols),
        on="wafer_id",
        how="left",
    )

    # ── any_ooc 통합 플래그 ───────────────────────────────────────────
    ooc_cols = [c for c in silver_pdf.columns if c.endswith("_ooc")]
    if ooc_cols:
        silver_pdf["any_ooc"] = silver_pdf[ooc_cols].fillna(0).max(axis=1).astype("int32")

    print(f"[Silver] JOIN 완료: {len(silver_pdf):,} rows, {len(silver_pdf.columns)} 컬럼")

    # ── PyArrow로 단일 파일 쓰기 ───────────────────────────────────────
    out_dir  = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / "wafer_silver.parquet")

    table = pa.Table.from_pandas(silver_pdf, preserve_index=False)
    pq.write_table(table, out_file, compression="snappy")

    print(f"[Silver] 저장 완료 → {out_file}")
    print(f"  파일 크기: {Path(out_file).stat().st_size / 1024**2:.1f} MB")

    # ── Spark로 읽어 클래스 분포 통계 출력 ────────────────────────────
    from pyspark.sql import functions as F
    sdf   = spark.read.parquet(out_file)
    total = sdf.count()

    print(f"\n[Silver] 클래스 분포 (Spark SQL):")
    (sdf.groupBy("failure_type")
        .agg(
            F.count("*").alias("count"),
            F.avg("defect_density").alias("avg_defect_density"),
            F.avg("edge_defect_ratio").alias("avg_edge_ratio"),
        )
        .withColumn("비율%", F.round(F.col("count") / total * 100, 2))
        .orderBy("count", ascending=False)
        .show(truncate=False))

    print(f"[Silver] Total: {total:,} rows")
    return sdf


def _find_parquet(path: str) -> str:
    """경로가 디렉터리이면 내부 첫 번째 .parquet 파일 반환."""
    p = Path(path)
    if p.suffix == ".parquet" and p.is_file():
        return str(p)
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if files:
            return str(files[0])
    raise FileNotFoundError(f"Parquet 파일을 찾을 수 없음: {path}")
