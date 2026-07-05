"""
SPC (Statistical Process Control) 집계: Spark SQL Window 함수 활용

웨이퍼 ID 순서를 시계열 대용으로 사용하여:
  - 이동 평균 (25매 기준 Rolling Average)
  - 공정 능력 지수 Cpk 계산
  - 관리 이탈(OOC) 패턴 집계

출력:
  data/lake/gold/spc_statistics/spc.parquet
  data/lake/gold/spc_class_summary/spc_summary.parquet
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

# 공정 파라미터 관리 한계 (UCL/LCL)
SPEC_LIMITS = {
    "cmp_pressure":    (80.0,  120.0),
    "polish_time":     (25.0,   45.0),
    "slurry_ph":       (5.5,    7.5),
    "annealing_temp":  (1050.0, 1150.0),
    "temp_gradient":   (0.0,    2.0),
    "etch_depth":      (450.0,  550.0),
    "pr_thickness_cv": (0.0,    5.0),
    "particle_count":  (0.0,    10.0),
}


def build_spc_statistics(spark: SparkSession, silver_path: str, out_path: str):
    """
    Silver 계층에서 SPC 통계 집계.

    핵심 Spark 기법 (Python worker 없는 JVM 전용 연산):
      1. Window 함수: 이동 평균 (ROWS BETWEEN 24 PRECEDING AND CURRENT ROW)
      2. Spark SQL: Cpk = MIN((UCL-μ)/3σ, (μ-LCL)/3σ)
      3. GroupBy + Agg: 결함 유형별 공정 파라미터 통계
    """
    print("[Gold/spc] Loading Silver layer (Spark)...")

    # glob 방식으로 단일 파일 읽기 (NativeIO 우회)
    silver_file = _find_parquet(silver_path)
    silver      = spark.read.parquet(silver_file)

    # wafer_map_flat 제외 (SPC에 불필요)
    param_cols = list(SPEC_LIMITS.keys())
    available  = [c for c in param_cols if c in silver.columns]

    sdf = silver.select(
        "wafer_id", "failure_type", "split",
        "defect_density", *available,
    )

    # ── 1. Window 함수: 25매 이동 평균 ───────────────────────────────
    w25 = Window.orderBy("wafer_id").rowsBetween(-24, 0)

    for col in available:
        sdf = sdf.withColumn(f"{col}_roll25", F.avg(col).over(w25))

    # ── 2. Spark SQL 뷰 등록 후 Cpk 계산 ─────────────────────────────
    sdf.createOrReplaceTempView("spc_data")

    cpk_exprs = []
    for col, (lcl, ucl) in SPEC_LIMITS.items():
        if col not in available:
            continue
        cpk_exprs.append(f"""
            LEAST(
                ({ucl} - AVG({col})) / (3 * STDDEV({col}) + 1e-9),
                (AVG({col}) - {lcl})  / (3 * STDDEV({col}) + 1e-9)
            ) AS cpk_{col}
        """)

    summary_sql = f"""
        SELECT
            failure_type,
            COUNT(*)                                                   AS wafer_count,
            SUM(CASE WHEN failure_type != 'none' THEN 1 ELSE 0 END)   AS defect_count,
            AVG(CASE WHEN failure_type != 'none' THEN 1.0 ELSE 0.0 END) AS defect_rate,
            AVG(defect_density)                                        AS avg_defect_density,
            {', '.join(cpk_exprs)}
        FROM spc_data
        GROUP BY failure_type
        ORDER BY defect_rate DESC
    """

    class_summary = spark.sql(summary_sql)

    print("\n[Gold/spc] 클래스별 Cpk 요약 (Spark SQL):")
    class_summary.show(truncate=False)

    # ── 3. 결함 클래스 OOC 통계 ──────────────────────────────────────
    ooc_sql = """
        SELECT
            failure_type,
            COUNT(*)                                AS n,
            AVG(cmp_pressure)                       AS avg_cmp,
            STDDEV(cmp_pressure)                    AS std_cmp,
            AVG(particle_count)                     AS avg_particles,
            PERCENTILE_APPROX(defect_density, 0.5)  AS median_defect_density,
            PERCENTILE_APPROX(defect_density, 0.95) AS p95_defect_density
        FROM spc_data
        WHERE failure_type != 'none'
        GROUP BY failure_type
        ORDER BY avg_particles DESC
    """
    defect_ooc = spark.sql(ooc_sql)
    print("\n[Gold/spc] 결함 클래스별 공정 통계:")
    defect_ooc.show(truncate=False)

    # ── 4. PyArrow로 저장 ─────────────────────────────────────────────
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / "spc.parquet")

    spc_pdf = sdf.toPandas()
    pq.write_table(pa.Table.from_pandas(spc_pdf, preserve_index=False),
                   out_file, compression="snappy")
    print(f"[Gold/spc] SPC 저장 → {out_file} ({Path(out_file).stat().st_size/1024**2:.1f} MB)")

    # class_summary 저장
    summary_dir  = Path(out_path.replace("spc_statistics", "spc_class_summary"))
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = str(summary_dir / "spc_summary.parquet")
    summary_pdf  = class_summary.toPandas()
    pq.write_table(pa.Table.from_pandas(summary_pdf, preserve_index=False),
                   summary_file, compression="snappy")
    print(f"[Gold/spc] 클래스 요약 저장 → {summary_file}")

    print(f"\n[Gold/spc] {len(spc_pdf):,} rows 처리 완료")
    return sdf, class_summary


def _find_parquet(path: str) -> str:
    p = Path(path)
    if p.suffix == ".parquet" and p.is_file():
        return str(p)
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if files:
            return str(files[0])
    raise FileNotFoundError(f"Parquet 파일을 찾을 수 없음: {path}")
