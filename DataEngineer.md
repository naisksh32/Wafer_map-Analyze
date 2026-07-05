# WM-811K 웨이퍼 데이터 — Spark 기반 데이터 엔지니어링 파이프라인 계획

> **작성일:** 2026-07-06  
> **목표 직무:** 데이터 엔지니어 — 데이터 파이프라인 개발/운영  
> **요구 스킬:** Python · SQL · Hadoop · Spark · Hive  
> **주력 언어:** Python (PySpark API 활용)

---

## 0. 왜 Spark인가 — 현 프로젝트에서의 필요성

| 현황 | 문제 | Spark 해결책 |
|------|------|-------------|
| `LSWMD.pkl` 2.0 GB, 단일 파일 | 전체 로드 시 RAM 2~4 GB 점유, 병렬 처리 불가 | 분산 파티셔닝 → 노드별 병렬 처리 |
| `all_maps_resized.npy` 675 MB | 증강 시 순차 반복 (병목) | RDD/DataFrame 병렬 map 연산 |
| pandas DataFrame으로 EDA | 811K 행 전체 메모리 상주 | Lazy Evaluation → 필요 시만 연산 |
| Pickle/Numpy 이진 형식 | SQL 쿼리 불가, 스키마 미정의 | Parquet + Hive 테이블 → SQL 접근 |
| 단일 스크립트 전처리 | 재현성 없음, 버전 관리 불가 | Delta Lake → ACID 트랜잭션 + 히스토리 |

> **규모 근거:** Phase 3 합성 후 총 ~250,000 샘플, 64×64 맵 기준 **약 1 GB+**  
> 단일 머신 pandas로도 가능하나, Spark 파이프라인 구축 자체가 포트폴리오 핵심 역량.

---

## 1. 데이터 레이크 아키텍처 (Medallion Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                        데이터 소스                                │
│   LSWMD.pkl (2GB)   │  process_parameters.csv  │  합성 맵 (미래) │
└────────────┬────────────────────┬───────────────────────────────┘
             │  PySpark Ingestion │
             ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  🥉 BRONZE LAYER  (원본 보존, 변환 최소화)                         │
│                                                                  │
│  hdfs://wafer-lake/bronze/                                       │
│  ├── wafer_raw/           ← pkl → Parquet (스키마 정의)           │
│  │   └── part-*.parquet  (lot_id, wafer_id, die_map, fail_type) │
│  └── process_params/      ← CSV → Parquet                        │
│      └── part-*.parquet  (파라미터 9종 + 타임스탬프)              │
│                                                                  │
│  Hive: wafer_db.bronze_raw  (External Table)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │  PySpark Transform
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  🥈 SILVER LAYER  (정제 + 라벨 정리 + 조인)                        │
│                                                                  │
│  hdfs://wafer-lake/silver/                                       │
│  ├── wafer_labeled/        (172,950건, defect_type 파티셔닝)       │
│  ├── wafer_unlabeled/      (638,507건, 준지도 학습용)              │
│  └── wafer_with_params/    (웨이퍼 맵 + 공정 파라미터 조인)         │
│                                                                  │
│  Hive: wafer_db.silver_labeled   (Partitioned by defect_type)   │
│        wafer_db.silver_unlabeled                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │  PySpark Feature Engineering
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  🥇 GOLD LAYER  (학습 바로 투입 가능한 Feature Store)              │
│                                                                  │
│  hdfs://wafer-lake/gold/                                         │
│  ├── train/               (stratified split, 70%)               │
│  ├── validation/          (15%)                                  │
│  ├── test/                (15%)                                  │
│  ├── feature_store/       (PCA 특징, 통계 특징)                   │
│  └── spc_statistics/      (배치별 Cpk, 공정 이상 집계)             │
│                                                                  │
│  Hive: wafer_db.gold_train  wafer_db.gold_feature_store         │
│        wafer_db.gold_spc_stats                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 파일/폴더 구조

```
spark_pipeline/                    ← 신규 생성 디렉터리
├── config/
│   └── pipeline_config.py         ← HDFS 경로, Spark 설정 중앙화
│
├── ingestion/                     ← Bronze 계층
│   ├── pkl_to_bronze.py           ← LSWMD.pkl → Parquet 변환
│   └── csv_to_bronze.py           ← process_parameters.csv → Parquet
│
├── transform/                     ← Silver 계층
│   ├── bronze_to_silver.py        ← 라벨 정제, 결측 처리, 조인
│   └── silver_to_gold.py          ← 분할, 증강 메타데이터, 멀티레이블
│
├── feature_engineering/           ← Gold 계층
│   ├── spatial_features.py        ← 공간 통계 특징 (defect density 등)
│   ├── pca_features.py            ← PySpark MLlib PCA
│   └── spc_aggregation.py         ← 배치/로트별 SPC 통계 집계
│
├── quality/
│   └── data_quality_checks.py     ← Great Expectations 검증
│
├── hive/
│   ├── create_tables.sql           ← DDL (Hive 테이블 생성)
│   └── analytical_queries.sql      ← SPC 분석용 Hive QL
│
├── dag/
│   └── wafer_pipeline_dag.py       ← Airflow DAG (기존 07_airflow와 연동)
│
└── notebooks/
    └── 16_spark_pipeline.ipynb     ← 전체 파이프라인 실증 노트북
```

---

## 3. Bronze 계층 — 데이터 수집 (Ingestion)

### 3-1. Spark 세션 설정

```python
# spark_pipeline/config/pipeline_config.py
from pyspark.sql import SparkSession

def get_spark_session(app_name: str = "WaferPipeline") -> SparkSession:
    """
    로컬 개발: local[*] (모든 CPU 코어 사용)
    클러스터: yarn 또는 standalone 변경
    Delta Lake: delta-core 패키지 포함
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        # Delta Lake 지원
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Parquet 최적화
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.adaptive.enabled", "true")
        # 메모리 설정 (RTX 2060 SUPER 로컬 기준)
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

# 경로 설정 (로컬 개발 시 HDFS 대신 로컬 경로 사용)
PATHS = {
    "raw_pkl":         "data/raw/LSWMD.pkl",
    "raw_csv":         "data/process_parameters.csv",
    "bronze_wafer":    "data/lake/bronze/wafer_raw",
    "bronze_params":   "data/lake/bronze/process_params",
    "silver_labeled":  "data/lake/silver/wafer_labeled",
    "silver_unlabeled":"data/lake/silver/wafer_unlabeled",
    "silver_joined":   "data/lake/silver/wafer_with_params",
    "gold_train":      "data/lake/gold/train",
    "gold_val":        "data/lake/gold/validation",
    "gold_test":       "data/lake/gold/test",
    "gold_features":   "data/lake/gold/feature_store",
    "gold_spc":        "data/lake/gold/spc_statistics",
}
```

### 3-2. pkl → Bronze Parquet 변환

```python
# spark_pipeline/ingestion/pkl_to_bronze.py
import pickle
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    BinaryType, ArrayType, FloatType
)

def extract_label(raw_label) -> str:
    """중첩 리스트 레이블 정제 (기존 02_eda.ipynb 로직 재사용)."""
    if isinstance(raw_label, list) and len(raw_label) > 0:
        inner = raw_label[0]
        if isinstance(inner, list) and len(inner) > 0:
            return str(inner[0])
        return str(inner)
    return str(raw_label) if raw_label is not None else "unknown"

# Spark가 이해하는 스키마 명시 정의
BRONZE_SCHEMA = StructType([
    StructField("lot_id",        StringType(),  True),
    StructField("wafer_id",      IntegerType(), True),
    StructField("failure_type",  StringType(),  True),  # 정제 후
    StructField("die_size_x",    FloatType(),   True),
    StructField("die_size_y",    FloatType(),   True),
    StructField("wafer_map_flat",BinaryType(),  True),  # 64×64 직렬화
    StructField("map_rows",      IntegerType(), True),
    StructField("map_cols",      IntegerType(), True),
    StructField("is_labeled",    IntegerType(), True),  # 0/1
])

def pkl_to_bronze(spark: SparkSession, pkl_path: str, out_path: str):
    """
    LSWMD.pkl을 로드 → pandas 전처리 → Spark DataFrame → Parquet 저장.
    
    전략: pkl은 pandas로 읽고, 대규모 변환은 Spark로 위임.
    (단일 머신에서 pkl을 직접 Spark로 읽는 것은 비효율적)
    """
    print(f"[Bronze] Loading {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        df_raw = pickle.load(f)
    
    records = []
    for idx, row in df_raw.iterrows():
        label = extract_label(row.get("failureType", "unknown"))
        wafer_map = row.get("waferMap")
        
        if wafer_map is None or not hasattr(wafer_map, "shape"):
            continue
        
        # 64×64 리사이징 (CV2, 기존 전처리와 동일)
        import cv2
        resized = cv2.resize(
            wafer_map.astype(np.uint8), (64, 64),
            interpolation=cv2.INTER_NEAREST
        )
        
        records.append({
            "lot_id":         str(row.get("lotName", f"LOT_{idx // 25:05d}")),
            "wafer_id":       int(idx % 25),
            "failure_type":   label,
            "die_size_x":     float(row.get("dieSize", [1.0, 1.0])[0]) if hasattr(row.get("dieSize", 1.0), "__len__") else 1.0,
            "die_size_y":     float(row.get("dieSize", [1.0, 1.0])[1]) if hasattr(row.get("dieSize", 1.0), "__len__") else 1.0,
            "wafer_map_flat": resized.flatten().tobytes(),  # bytes
            "map_rows":       64,
            "map_cols":       64,
            "is_labeled":     0 if label == "unknown" else 1,
        })
    
    # pandas → Spark DataFrame
    pdf = pd.DataFrame(records)
    sdf = spark.createDataFrame(pdf, schema=BRONZE_SCHEMA)
    
    # Parquet 저장: lot_id로 파티셔닝 (병렬 읽기 최적화)
    (sdf.write
        .mode("overwrite")
        .partitionBy("is_labeled")            # 라벨/미라벨 분리 저장
        .parquet(out_path))
    
    print(f"[Bronze] Saved {sdf.count():,} records → {out_path}")
    print(f"  Labeled: {sdf.filter('is_labeled=1').count():,}")
    print(f"  Unlabeled: {sdf.filter('is_labeled=0').count():,}")
    return sdf
```

---

## 4. Silver 계층 — 정제 및 조인 (Transformation)

```python
# spark_pipeline/transform/bronze_to_silver.py
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 유효 불량 유형 (none 포함 9종)
VALID_LABELS = {"none","Center","Donut","Edge-Loc","Edge-Ring","Loc","Near-full","Random","Scratch"}
LABEL_TO_IDX = {l: i for i, l in enumerate(sorted(VALID_LABELS))}

def bronze_to_silver(spark, bronze_path: str, params_path: str, out_labeled: str, out_unlabeled: str):
    """
    Bronze → Silver:
    1. 라벨 유효성 필터링
    2. 공정 파라미터 JOIN (lot_id 키)
    3. defect_type 파티셔닝으로 저장
    """
    bronze = spark.read.parquet(bronze_path)
    params = spark.read.parquet(params_path)
    
    # ── 라벨 정제 ──────────────────────────────────────────
    labeled = (bronze
        .filter(F.col("is_labeled") == 1)
        .filter(F.col("failure_type").isin(list(VALID_LABELS)))
        .withColumn("class_idx", 
            F.when(F.col("failure_type") == "none",       0)
             .when(F.col("failure_type") == "Center",     1)
             .when(F.col("failure_type") == "Donut",      2)
             .when(F.col("failure_type") == "Edge-Loc",   3)
             .when(F.col("failure_type") == "Edge-Ring",  4)
             .when(F.col("failure_type") == "Loc",        5)
             .when(F.col("failure_type") == "Near-full",  6)
             .when(F.col("failure_type") == "Random",     7)
             .when(F.col("failure_type") == "Scratch",    8)
             .otherwise(-1))
    )
    
    # ── 공정 파라미터 LEFT JOIN ─────────────────────────────
    silver = labeled.join(params, on="lot_id", how="left")
    
    # ── 클래스 불균형 통계 (Window 함수 활용) ──────────────
    total   = silver.count()
    class_w = (silver
        .groupBy("failure_type")
        .agg(F.count("*").alias("cnt"))
        .withColumn("weight", F.lit(total) / (F.col("cnt") * F.lit(9)))
    )
    class_w.show()
    
    # ── defect_type 파티셔닝으로 Silver 저장 ───────────────
    (silver.write
        .mode("overwrite")
        .partitionBy("failure_type")           # Hive 파티션 키
        .parquet(out_labeled))
    
    # 미라벨 데이터 별도 저장 (준지도 학습용)
    unlabeled = bronze.filter(F.col("is_labeled") == 0)
    unlabeled.write.mode("overwrite").parquet(out_unlabeled)
    
    print(f"[Silver] Labeled: {silver.count():,}")
    print(f"[Silver] Unlabeled: {unlabeled.count():,}")
```

---

## 5. Gold 계층 — Feature Store & SPC 집계

### 5-1. 학습 데이터 분할 (Stratified Split)

```python
# spark_pipeline/transform/silver_to_gold.py
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer

def silver_to_gold_split(spark, silver_path, gold_train, gold_val, gold_test,
                          ratios=(0.70, 0.15, 0.15), seed=42):
    """
    Stratified Split: 각 defect_type 비율 보존.
    Spark의 sampleBy()로 클래스별 샘플링.
    """
    silver = spark.read.parquet(silver_path)
    
    # 클래스별 분할 비율 딕셔너리
    labels = [r["failure_type"] for r in silver.select("failure_type").distinct().collect()]
    
    train_frac = {l: ratios[0] for l in labels}
    
    train = silver.sampleBy("failure_type", fractions=train_frac, seed=seed)
    remain = silver.subtract(train)
    
    val_frac  = {l: ratios[1] / (ratios[1] + ratios[2]) for l in labels}
    val   = remain.sampleBy("failure_type", fractions=val_frac, seed=seed)
    test  = remain.subtract(val)
    
    for sdf, path, name in [(train, gold_train, "Train"),
                             (val,   gold_val,   "Val"),
                             (test,  gold_test,  "Test")]:
        sdf.write.mode("overwrite").partitionBy("failure_type").parquet(path)
        print(f"[Gold] {name}: {sdf.count():,}")
```

### 5-2. 공간 통계 특징 추출 (Feature Engineering)

```python
# spark_pipeline/feature_engineering/spatial_features.py
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StructType, StructField

# UDF: 웨이퍼 맵 바이너리 → 공간 통계 특징 벡터
@F.udf(returnType=StructType([
    StructField("defect_density",    FloatType()),  # 불량 다이 비율
    StructField("active_ratio",      FloatType()),  # 활성 다이 비율
    StructField("edge_defect_ratio", FloatType()),  # 에지 불량 비율
    StructField("center_defect_ratio", FloatType()),# 중심 불량 비율
    StructField("radial_mean",       FloatType()),  # 반경별 불량 평균 거리
]))
def extract_spatial_features(map_bytes):
    """64×64 웨이퍼 맵에서 공간 통계 특징 추출."""
    if map_bytes is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    
    arr = np.frombuffer(map_bytes, dtype=np.uint8).reshape(64, 64)
    total     = (arr > 0).sum()
    if total == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    
    defects   = (arr == 2)
    cx, cy    = 32, 32
    R         = 31.5
    
    # 반경 마스크
    y_idx, x_idx = np.ogrid[:64, :64]
    dist      = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    edge_mask = dist > (R * 0.75)
    cntr_mask = dist < (R * 0.38)
    
    return (
        float(defects.sum() / total),
        float(total / (64 * 64)),
        float((defects & edge_mask).sum() / (defects.sum() + 1e-6)),
        float((defects & cntr_mask).sum() / (defects.sum() + 1e-6)),
        float(dist[defects].mean()) if defects.any() else 0.0,
    )

def build_feature_store(spark, silver_path, out_path):
    silver = spark.read.parquet(silver_path)
    feats  = silver.withColumn("spatial", extract_spatial_features("wafer_map_flat"))
    feats  = (feats
        .withColumn("defect_density",     F.col("spatial.defect_density"))
        .withColumn("active_ratio",       F.col("spatial.active_ratio"))
        .withColumn("edge_defect_ratio",  F.col("spatial.edge_defect_ratio"))
        .withColumn("center_defect_ratio",F.col("spatial.center_defect_ratio"))
        .withColumn("radial_mean",        F.col("spatial.radial_mean"))
        .drop("spatial", "wafer_map_flat")   # 특징만 저장 (맵 제외)
    )
    feats.write.mode("overwrite").parquet(out_path)
    print(f"[Gold] Feature store: {feats.count():,} rows, {len(feats.columns)} features")
```

### 5-3. SPC 집계 (Spark SQL Window 함수)

```python
# spark_pipeline/feature_engineering/spc_aggregation.py
def build_spc_statistics(spark, silver_joined_path, out_path):
    """
    로트/배치 단위 SPC 통계 집계.
    Spark SQL Window 함수로 이동 평균, σ, Cpk 계산.
    """
    df = spark.read.parquet(silver_joined_path)
    
    # Spark SQL 등록 후 HiveQL 스타일 쿼리 실행
    df.createOrReplaceTempView("wafer_data")
    
    spc_df = spark.sql("""
        SELECT
            lot_id,
            failure_type,
            
            -- 공정 파라미터별 통계
            AVG(cmp_pressure)     AS avg_cmp_pressure,
            STDDEV(cmp_pressure)  AS std_cmp_pressure,
            AVG(annealing_temp)   AS avg_annealing_temp,
            STDDEV(annealing_temp)AS std_annealing_temp,
            
            -- Cpk 근사: MIN((UCL-μ)/3σ, (μ-LCL)/3σ)
            LEAST(
                (120.0 - AVG(cmp_pressure)) / (3 * STDDEV(cmp_pressure) + 1e-6),
                (AVG(cmp_pressure) - 80.0)  / (3 * STDDEV(cmp_pressure) + 1e-6)
            ) AS cpk_cmp_pressure,
            
            -- 불량률 집계
            COUNT(*)                              AS wafer_count,
            SUM(CASE WHEN failure_type != 'none' THEN 1 ELSE 0 END) AS defect_count,
            AVG(CASE WHEN failure_type != 'none' THEN 1.0 ELSE 0.0 END) AS defect_rate,
            
            -- 이동 평균 (직전 25매 로트 기준)
            AVG(cmp_pressure) OVER (
                ORDER BY lot_id
                ROWS BETWEEN 24 PRECEDING AND CURRENT ROW
            ) AS rolling_avg_cmp_25
            
        FROM wafer_data
        GROUP BY lot_id, failure_type
        ORDER BY lot_id
    """)
    
    spc_df.write.mode("overwrite").parquet(out_path)
    return spc_df
```

---

## 6. Hive 테이블 DDL

```sql
-- spark_pipeline/hive/create_tables.sql
-- Hive Metastore 등록 (Spark의 enableHiveSupport() 필요)

CREATE DATABASE IF NOT EXISTS wafer_db
  COMMENT 'WM-811K 웨이퍼 불량 분석 데이터 레이크'
  LOCATION 'hdfs://localhost:9000/wafer-lake';

-- Bronze: 원본 보존 External Table
CREATE EXTERNAL TABLE IF NOT EXISTS wafer_db.bronze_raw (
    lot_id         STRING     COMMENT '로트 식별자',
    wafer_id       INT        COMMENT '로트 내 웨이퍼 번호',
    failure_type   STRING     COMMENT '불량 유형 (정제 전)',
    die_size_x     FLOAT      COMMENT '다이 X 크기 (mm)',
    die_size_y     FLOAT      COMMENT '다이 Y 크기 (mm)',
    wafer_map_flat BINARY     COMMENT '64×64 직렬화 맵',
    map_rows       INT,
    map_cols       INT
)
PARTITIONED BY (is_labeled INT COMMENT '0=미라벨, 1=라벨')
STORED AS PARQUET
LOCATION 'hdfs://localhost:9000/wafer-lake/bronze/wafer_raw'
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');

-- Silver: 정제 + 파티셔닝 Managed Table
CREATE TABLE IF NOT EXISTS wafer_db.silver_labeled (
    lot_id           STRING,
    wafer_id         INT,
    class_idx        INT        COMMENT '0=none ~ 8=Scratch',
    defect_density   FLOAT      COMMENT '불량 다이 비율',
    active_ratio     FLOAT,
    cmp_pressure     FLOAT,
    polish_time      FLOAT,
    slurry_ph        FLOAT,
    annealing_temp   FLOAT,
    temp_gradient    FLOAT,
    etch_depth       FLOAT,
    vacuum_pressure  FLOAT,
    pr_thickness_cv  FLOAT,
    particle_count   FLOAT
)
PARTITIONED BY (failure_type STRING)
STORED AS PARQUET
LOCATION 'hdfs://localhost:9000/wafer-lake/silver/wafer_labeled';

-- Gold SPC 집계: 분석용 View
CREATE VIEW IF NOT EXISTS wafer_db.v_spc_alert AS
SELECT
    lot_id,
    failure_type,
    cpk_cmp_pressure,
    defect_rate,
    CASE
        WHEN cpk_cmp_pressure < 1.0  THEN 'CRITICAL'
        WHEN cpk_cmp_pressure < 1.33 THEN 'WARNING'
        ELSE 'OK'
    END AS spc_status
FROM wafer_db.gold_spc_stats
WHERE cpk_cmp_pressure IS NOT NULL;
```

### 분석용 Hive SQL 쿼리 예시

```sql
-- spark_pipeline/hive/analytical_queries.sql

-- 1. 결함 유형별 월별 트렌드
SELECT
    failure_type,
    SUBSTR(lot_id, 1, 6)   AS yyyymm,
    COUNT(*)               AS wafer_count,
    AVG(defect_density)    AS avg_defect_density
FROM wafer_db.silver_labeled
GROUP BY failure_type, SUBSTR(lot_id, 1, 6)
ORDER BY yyyymm, failure_type;

-- 2. 공정 파라미터 이상 웨이퍼 식별 (Hive QL)
SELECT lot_id, wafer_id, failure_type, cmp_pressure, particle_count
FROM wafer_db.silver_labeled
WHERE failure_type != 'none'
  AND (cmp_pressure > 125 OR particle_count > 15)
ORDER BY particle_count DESC
LIMIT 100;

-- 3. 불량률 상위 로트 조회
SELECT
    lot_id,
    SUM(CASE WHEN failure_type != 'none' THEN 1 ELSE 0 END)
        / COUNT(*) AS defect_rate,
    COLLECT_SET(failure_type) AS defect_types
FROM wafer_db.silver_labeled
GROUP BY lot_id
HAVING defect_rate > 0.2
ORDER BY defect_rate DESC;
```

---

## 7. 데이터 품질 검증 (Great Expectations)

```python
# spark_pipeline/quality/data_quality_checks.py
import great_expectations as ge

def run_bronze_quality_checks(spark, bronze_path: str):
    """Bronze 계층 데이터 품질 검증."""
    df = spark.read.parquet(bronze_path).toPandas()
    ge_df = ge.from_pandas(df)
    
    results = ge_df.expect_all([
        # 필수 컬럼 존재 확인
        ge_df.expect_column_to_exist("lot_id"),
        ge_df.expect_column_to_exist("wafer_map_flat"),
        
        # null 허용 범위
        ge_df.expect_column_values_to_not_be_null("lot_id"),
        ge_df.expect_column_values_to_not_be_null("failure_type"),
        
        # 값 범위 검증
        ge_df.expect_column_values_to_be_between("map_rows", 64, 64),
        ge_df.expect_column_values_to_be_between("map_cols", 64, 64),
        
        # 레이블 유효성
        ge_df.expect_column_values_to_be_in_set(
            "failure_type",
            {"none","Center","Donut","Edge-Loc","Edge-Ring",
             "Loc","Near-full","Random","Scratch","unknown"}
        ),
        
        # 전체 행 수 (811K ± 5%)
        ge_df.expect_table_row_count_to_be_between(770_000, 850_000),
    ])
    
    print(f"[Quality] Bronze checks passed: {results.success}")
    return results
```

---

## 8. Airflow DAG (기존 07_airflow와 통합)

```python
# spark_pipeline/dag/wafer_pipeline_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    "owner":            "wafer-de",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="wafer_data_pipeline",
    default_args=default_args,
    schedule_interval="@daily",          # 매일 신규 웨이퍼 데이터 처리
    start_date=days_ago(1),
    catchup=False,
    tags=["wafer", "spark", "delta-lake"],
) as dag:

    # ── Task 1: Bronze 수집 ────────────────────────────────
    t1_ingest = PythonOperator(
        task_id="ingest_pkl_to_bronze",
        python_callable=run_pkl_to_bronze,  # pkl_to_bronze.py 함수
    )

    # ── Task 2: 데이터 품질 검증 ───────────────────────────
    t2_quality = PythonOperator(
        task_id="bronze_quality_check",
        python_callable=run_bronze_quality_checks,
    )

    # ── Task 3: Silver 변환 ────────────────────────────────
    t3_transform = PythonOperator(
        task_id="transform_to_silver",
        python_callable=run_bronze_to_silver,
    )

    # ── Task 4: Feature Engineering (병렬) ─────────────────
    t4a_spatial  = PythonOperator(task_id="build_spatial_features", ...)
    t4b_spc      = PythonOperator(task_id="build_spc_statistics",   ...)

    # ── Task 5: Gold 분할 ──────────────────────────────────
    t5_gold = PythonOperator(task_id="split_to_gold", ...)

    # ── Task 6: 모델 학습 트리거 (기존 Airflow DAG 연동) ────
    t6_train = PythonOperator(task_id="trigger_model_training", ...)

    # DAG 의존성 정의
    t1_ingest >> t2_quality >> t3_transform >> [t4a_spatial, t4b_spc] >> t5_gold >> t6_train
```

---

## 9. 구현 로드맵

```
Week 1: 인프라 설정 + Bronze
  ├── pip install pyspark delta-spark great-expectations
  ├── Spark 로컬 모드 세션 테스트
  ├── pkl_to_bronze.py 구현 및 검증
  └── 품질 체크 기본 케이스 작성

Week 2: Silver + Hive
  ├── bronze_to_silver.py (라벨 정제, JOIN)
  ├── Hive Metastore 로컬 설정 (Derby 메타스토어)
  ├── create_tables.sql DDL 실행
  └── Spark SQL 분석 쿼리 검증

Week 3: Gold + Feature Store
  ├── spatial_features.py UDF 구현
  ├── spc_aggregation.py Window 함수 집계
  ├── silver_to_gold.py Stratified Split
  └── 16_spark_pipeline.ipynb 전체 실증

Week 4: Airflow + 기존 파이프라인 통합
  ├── wafer_pipeline_dag.py (기존 07_airflow DAG와 연동)
  ├── 기존 ML 학습 코드가 Gold 레이어 읽도록 수정
  └── End-to-End 테스트
```

---

## 10. 직무 역량 매핑

| 요구 스킬 | 본 파이프라인 구현 포인트 |
|----------|----------------------|
| **Python** | PySpark DataFrame API, UDF, pandas 브릿지 |
| **SQL** | Spark SQL, HiveQL (Window 함수, GROUP BY, HAVING) |
| **Hadoop** | HDFS 경로 설계, 파티셔닝, Parquet 포맷 |
| **Spark** | SparkSession, DataFrame API, MLlib, Adaptive Query Execution |
| **Hive** | 외부/관리형 테이블 DDL, 파티션 관리, View 생성 |
| **Java (간접)** | Spark JVM 기반 설정 파라미터 이해, `.jar` 의존성 관리 |

> **핵심 어필 포인트:**  
> "811K 웨이퍼 이미지 데이터를 Medallion Architecture로 설계하여  
> Bronze → Silver → Gold 계층화 파이프라인을 PySpark로 구축했으며,  
> Hive Metastore를 통해 SQL 접근성을 확보하고 Airflow로 일별 스케줄링을 운영했습니다."

---

## 11. 트러블슈팅 기록 (Windows 로컬 환경)

> 실제 구현 중 발생한 문제와 해결 과정을 기록.  
> Windows + Python 3.12 + PySpark 3.5.5 로컬 환경 기준.

---

### 문제 1: 한글 경로에서 JVM 시작 실패

**증상**
```
JAVA_GATEWAY_EXITED
py4j.protocol.Py4JNetworkError: Answer from Java side is empty
```

**원인**  
프로젝트 경로(`하이닉스 대비\웨이퍼 불량 분석\`)에 한글이 포함되어 있어 PySpark가 자동 탐지하는 `SPARK_HOME` 경로를 JVM 클래스패스로 전달할 때 인코딩 오류 발생.

**해결**  
```python
import pyspark
from pathlib import Path

# SPARK_HOME을 PySpark 패키지 경로로 명시 설정
os.environ["SPARK_HOME"]            = str(Path(pyspark.__file__).parent)
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
```
추가로, 가상환경 자체도 한글 없는 경로(`C:\Users\naisk\spark_env`)에 별도 생성하여 사용.

```powershell
python -m venv C:\Users\naisk\spark_env
C:\Users\naisk\spark_env\Scripts\pip install pyspark==3.5.5 pyarrow pandas numpy
```

---

### 문제 2: Python 3.12에서 `distutils` 모듈 없음

**증상**
```
ModuleNotFoundError: No module named 'distutils'
```

**원인**  
Python 3.12에서 `distutils`가 표준 라이브러리에서 제거됨. PySpark 3.5.5 내부에서 `distutils`를 참조.

**해결**
```powershell
pip install setuptools  # distutils 대체 제공
```

---

### 문제 3: Python Worker 충돌 (EOFException)

**증상**
```
org.apache.spark.SparkException: Python worker exited unexpectedly (crashed)
Caused by: java.io.EOFException
    at java.base/java.io.DataInputStream.readInt(DataInputStream.java:386)
```

**원인**  
Windows에서 PySpark 3.5.5 + Python 3.12 조합으로 Python Worker 프로세스(UDF 실행용)를 spawn할 때 소켓 통신이 비정상 종료됨. Python UDF(`@udf`, `@pandas_udf`) 실행 시 재현됨.

**해결 전략: UDF 완전 제거 → numpy 사전 계산**

Python UDF로 계산하려던 공간 특징(defect_density, edge_defect_ratio 등)을 Bronze 수집 단계에서 numpy로 직접 계산하여 Parquet에 저장.

```python
# 변경 전: Spark UDF (Python worker 필요 → 충돌)
@udf(returnType=DoubleType())
def calc_defect_density(map_bytes):
    arr = np.frombuffer(bytes(map_bytes), dtype=np.uint8)
    ...

silver = silver.withColumn("defect_density", calc_defect_density("wafer_map_flat"))

# 변경 후: Bronze 수집 시 numpy로 사전 계산 (Python worker 불필요)
def _spatial_features(flat_bytes: bytes):
    arr = np.frombuffer(flat_bytes, dtype=np.uint8)
    ...
    return density, act_ratio, edge_ratio, center_ratio, radial_mean, d_cnt, a_cnt

for i in range(n_samples):
    dens, act_r, edge_r, ctr_r, rad, d, a = _spatial_features(maps[i].tobytes())
    records.append({"defect_density": dens, "edge_defect_ratio": edge_r, ...})
```

**결과:** Spark는 JVM 전용 연산(GROUP BY, Window, Cpk SQL)에만 사용, Python Worker를 완전히 배제.

---

### 문제 4: `winutils.exe` 미설치로 파일시스템 접근 실패

**증상**
```
ERROR Shell: Failed to locate the winutils binary in the hadoop binary path
java.io.IOException: Could not locate executable null\bin\winutils.exe
```

**원인**  
Windows에서 Hadoop의 로컬 파일시스템 작업(디렉터리 권한 확인 등)에 `winutils.exe`가 필요. Linux/Mac과 달리 Windows용 바이너리를 별도 설치해야 함.

**해결**  
.NET Framework 컴파일러로 x64 스텁 실행파일 생성:

```powershell
$csc = "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
Set-Content C:\Users\naisk\hadoop\winutils_stub.cs @"
using System;
class WinUtils {
    static int Main(string[] args) { return 0; }
}
"@
& $csc /platform:x64 /out:C:\Users\naisk\hadoop\bin\winutils.exe C:\Users\naisk\hadoop\winutils_stub.cs

$env:HADOOP_HOME = "C:\Users\naisk\hadoop"
```

---

### 문제 5: `NativeIO.Windows.access0` UnsatisfiedLinkError (디렉터리 읽기 실패)

**증상**
```
java.lang.UnsatisfiedLinkError: 'boolean org.apache.hadoop.io.nativeio.NativeIO$Windows.access0(java.lang.String, int)'
    at org.apache.hadoop.io.nativeio.NativeIO$Windows.access0(Native Method)
    at org.apache.hadoop.fs.FileUtil.canRead(FileUtil.java:1249)
    at org.apache.hadoop.fs.FileUtil.list(FileUtil.java:1454)
    at org.apache.hadoop.fs.RawLocalFileSystem.listStatus(...)
```

**원인**  
`spark.read.parquet(directory/)` 호출 시 Hadoop이 내부적으로 `RawLocalFileSystem.listStatus()` → `FileUtil.canRead()` → `NativeIO.Windows.access0()` JNI 함수를 호출함. Windows에서는 이 함수가 `hadoop.dll`에 구현되어 있어야 하는데, 정식 `hadoop.dll`이 없으면 `UnsatisfiedLinkError` 발생.

**시도한 방법들**

| 방법 | 결과 |
|------|------|
| GitHub에서 winutils + hadoop.dll 다운로드 (cdarlint/winutils) | HTML 페이지 반환 (404) |
| MinGW gcc로 JNI 스텁 DLL 컴파일 | DLL 심볼 확인됨 (`objdump`에서 `access0` export), 그러나 JVM이 DLL 로드 실패 |
| `java.library.path`에 hadoop bin 추가 (`spark.driver.extraJavaOptions`) | JVM은 DLL을 찾지 못함 (UnsatisfiedLinkError 지속) |

**근본 원인 분석**  
`RawLocalFileSystem.listStatus(path)`의 소스를 추적한 결과:
- `path`가 **파일**이면 → `FileUtil.list()` 호출 없이 바로 반환 → `access0` 미호출 ✅
- `path`가 **디렉터리**이면 → `FileUtil.list()` → `NativeIO.Windows.access()` → `access0` 호출 → 오류 ❌

**최종 해결: 디렉터리 읽기를 Python glob으로 우회**

```python
# 변경 전: Spark 디렉터리 읽기 → access0 호출 → 오류
sdf = spark.read.parquet("data/lake/silver/wafer_labeled/")

# 변경 후: Python glob으로 파일 목록 수집 → 개별 파일 경로 전달 → access0 미호출
from pathlib import Path
files = [str(f) for f in Path("data/lake/silver/wafer_labeled/").glob("*.parquet")]
sdf = spark.read.parquet(*files)
```

`pipeline_config.py`에 헬퍼 함수로 캡슐화:

```python
def spark_read_parquet(spark: SparkSession, path: str):
    """NativeIO.access0 UnsatisfiedLinkError 우회 헬퍼."""
    p = Path(path)
    if p.suffix == ".parquet" and p.is_file():
        return spark.read.parquet(str(p))
    if p.is_dir():
        files = sorted(p.glob("**/*.parquet"))
        return spark.read.parquet(*[str(f) for f in files])
```

---

### 문제 6: LEFT JOIN 후 NaN → int32 변환 오류

**증상**
```
pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer.
```

**원인**  
Bronze wafer_maps(172,950행)와 process_params(5,350행)를 LEFT JOIN하면 process_params에 없는 167,600행의 OOC 컬럼이 `NaN`이 됨. 이 상태에서 `int32`로 캐스트하면 오류 발생.

**해결**
```python
# 변경 전
silver_pdf["any_ooc"] = silver_pdf[ooc_cols].max(axis=1).astype("int32")

# 변경 후: NaN → 0 채운 후 캐스트
silver_pdf["any_ooc"] = silver_pdf[ooc_cols].fillna(0).max(axis=1).astype("int32")
```

---

### 최종 아키텍처 결정: pandas/PyArrow + Spark SQL 하이브리드

**배경**  
Windows 환경에서 Python Worker 충돌 + NativeIO 문제가 복합적으로 발생하여, 순수 Spark 파이프라인 구성에 한계가 있었음.

**채택한 설계 원칙**

| 역할 | 사용 기술 | 이유 |
|------|----------|------|
| 데이터 변환 (ETL) | **pandas + numpy** | Python Worker 없이 드라이버에서 직접 실행, 172k 행 규모에서 충분히 빠름 |
| Parquet I/O | **PyArrow** | 단일 파일 쓰기 → NativeIO 디렉터리 탐색 없이 Spark에서 읽기 가능 |
| 분석 SQL | **Spark SQL (JVM 전용)** | Window 함수, Cpk, GROUP BY → Python Worker 불필요, JVM에서 직접 실행 |
| 통계 출력 | **Spark DataFrame API** | `count()`, `groupBy().show()` → JVM 연산 |

```
[Bronze] numpy 계산 → PyArrow 단일 파일 (.parquet)
    ↓ spark.read.parquet(single_file)  ← JVM, access0 미호출
[Silver] pandas merge → PyArrow 단일 파일
    ↓ spark.read.parquet(single_file)
[Gold]   pandas filter → PyArrow 단일 파일 × 3(train/val/test)
         + Spark SQL Window (SPC Cpk, 25매 이동평균)
```

**실행 결과** (총 39.5초, RTX 2060 SUPER 로컬)

```
bronze/wafer_maps_bronze.parquet     107.0 MB   172,950행
bronze/process_params_bronze.parquet   0.5 MB     5,350행
silver/wafer_silver.parquet          107.5 MB   172,950행  (30 컬럼)
gold/train/train.parquet              75.5 MB   121,065행
gold/validation/val.parquet           16.4 MB    25,942행
gold/test/test.parquet                16.4 MB    25,943행
gold/feature_store/features.parquet    5.4 MB   172,950행  (29 컬럼)
gold/spc_statistics/spc.parquet        2.3 MB   172,950행
──────────────────────────────────────────────────
합계                                  330.9 MB
```

> **Linux/HDFS 환경에서는** `spark.read.parquet(directory)` + `spark.write.parquet(partitionBy(...))` 방식이 정상 동작함. 본 트러블슈팅은 Windows 로컬 개발 환경의 제약 사항으로, 실 운영 환경(클러스터)에서는 발생하지 않는 문제임.
