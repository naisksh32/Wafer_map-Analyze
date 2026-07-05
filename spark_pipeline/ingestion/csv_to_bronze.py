"""
Bronze 계층 수집 2: data/process_parameters.csv → Parquet

입력:
  data/process_parameters.csv  (172,950 rows, 12 columns)

출력:
  data/lake/bronze/process_params/process_params_bronze.parquet

스키마 (원본 CSV + OOC 플래그):
  wafer_id         INT
  cmp_pressure     DOUBLE   (CMP 공정 압력, psi)
  polish_time      DOUBLE   (연마 시간, sec)
  slurry_ph        DOUBLE   (슬러리 pH)
  annealing_temp   DOUBLE   (열처리 온도, °C)
  temp_gradient    DOUBLE   (온도 그래디언트, °C/cm)
  etch_depth       DOUBLE   (식각 깊이, Å)
  vacuum_pressure  DOUBLE   (진공 압력, Torr)
  pr_thickness_cv  DOUBLE   (PR 두께 CV, %)
  particle_count   DOUBLE   (파티클 수)
  defect_class     STRING   (명목 결함 클래스)
  is_defect        INT      (0=정상, 1=불량)
  *_ooc            INT      (SPC 관리 이탈 플래그)
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from pyspark.sql import SparkSession

# 공정 파라미터 정상 범위 (SPC 관리 한계: LCL, UCL)
PROCESS_SPEC = {
    "cmp_pressure":    (80.0,  120.0),
    "polish_time":     (25.0,   45.0),
    "slurry_ph":       (5.5,    7.5),
    "annealing_temp":  (1050.0, 1150.0),
    "temp_gradient":   (0.0,    2.0),
    "etch_depth":      (450.0,  550.0),
    "pr_thickness_cv": (0.0,    5.0),
    "particle_count":  (0.0,    10.0),
}


def csv_to_bronze(spark: SparkSession, csv_path: str, out_path: str):
    """
    process_parameters.csv → Bronze Parquet.

    전략: Spark createDataFrame 대신 pandas + PyArrow 직접 쓰기.
    OOC 플래그도 pandas 연산으로 처리 (Spark withColumn 불필요).
    """
    print(f"[Bronze/params] Reading {csv_path} ...")
    pdf = pd.read_csv(csv_path)
    pdf["wafer_id"]  = pdf["wafer_id"].astype("int32")
    pdf["is_defect"] = pdf["is_defect"].astype("int32")

    print(f"  행: {len(pdf):,} / 컬럼: {list(pdf.columns)}")

    # ── OOC (Out-Of-Control) 플래그 계산 ──────────────────────────────
    for col, (lcl, ucl) in PROCESS_SPEC.items():
        if col not in pdf.columns:
            continue
        pdf[f"{col}_ooc"] = ((pdf[col] < lcl) | (pdf[col] > ucl)).astype("int32")

    # ── PyArrow로 단일 파일 쓰기 ───────────────────────────────────────
    out_dir  = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / "process_params_bronze.parquet")

    table = pa.Table.from_pandas(pdf, preserve_index=False)
    pq.write_table(table, out_file, compression="snappy")

    print(f"[Bronze/params] 저장 완료 → {out_file}")
    print(f"  파일 크기: {Path(out_file).stat().st_size / 1024**2:.1f} MB")

    # ── Spark로 읽어 OOC 통계 출력 ────────────────────────────────────
    from pyspark.sql import functions as F
    sdf      = spark.read.parquet(out_file)
    ooc_cols = [c for c in sdf.columns if c.endswith("_ooc")]
    print("\n[Bronze/params] OOC 이탈 건수 (Spark SQL):")
    sdf.select([F.sum(c).alias(c) for c in ooc_cols]).show(truncate=False)

    return sdf
