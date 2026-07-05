"""
Spark 세션 설정 및 파이프라인 경로 중앙 관리.
로컬 개발: local[*] 모드 (HDFS 없이 로컬 파일시스템 사용)
"""
import os
import sys
import pyspark
from pathlib import Path

# ── SPARK_HOME을 PySpark 패키지 경로로 명시 설정 ──────────────────────
# 한글 경로 환경에서 자동 감지 실패 방지
_SPARK_HOME = str(Path(pyspark.__file__).parent)
os.environ["SPARK_HOME"]            = _SPARK_HOME
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

# ── 프로젝트 루트 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── 데이터 레이크 경로 (단일 Parquet 파일 방식) ───────────────────────
# Windows 로컬: NativeIO.access0 문제 우회를 위해 단일 파일 쓰기
PATHS = {
    # 소스
    "maps_npy":         str(PROJECT_ROOT / "data" / "processed" / "all_maps_resized.npy"),
    "split_pkl":        str(PROJECT_ROOT / "data" / "processed" / "split_indices.pkl"),
    "raw_pkl":          str(PROJECT_ROOT / "data" / "raw" / "LSWMD.pkl"),
    "process_csv":      str(PROJECT_ROOT / "data" / "process_parameters.csv"),

    # Bronze (디렉터리 경로 — 내부에 단일 .parquet 파일 저장)
    "bronze_maps":      str(PROJECT_ROOT / "data" / "lake" / "bronze" / "wafer_maps"),
    "bronze_params":    str(PROJECT_ROOT / "data" / "lake" / "bronze" / "process_params"),

    # Silver
    "silver_labeled":   str(PROJECT_ROOT / "data" / "lake" / "silver" / "wafer_labeled"),

    # Gold
    "gold_train":       str(PROJECT_ROOT / "data" / "lake" / "gold" / "train"),
    "gold_val":         str(PROJECT_ROOT / "data" / "lake" / "gold" / "validation"),
    "gold_test":        str(PROJECT_ROOT / "data" / "lake" / "gold" / "test"),
    "gold_features":    str(PROJECT_ROOT / "data" / "lake" / "gold" / "feature_store"),
    "gold_spc":         str(PROJECT_ROOT / "data" / "lake" / "gold" / "spc_statistics"),
    "gold_spc_summary": str(PROJECT_ROOT / "data" / "lake" / "gold" / "spc_class_summary"),
}

# ── 클래스 정의 ────────────────────────────────────────────────────────
CLASS_ORDER  = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
                "Loc", "Near-full", "Random", "Scratch"]
LABEL_MAP    = {name: idx for idx, name in enumerate(CLASS_ORDER)}
IDX_TO_LABEL = {idx: name for name, idx in LABEL_MAP.items()}


def get_spark(app_name: str = "WaferPipeline") -> SparkSession:
    """
    PySpark 로컬 세션 반환.
    - local[*]: 전체 CPU 코어 활용
    - adaptive query execution: 자동 최적화
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory",                 "4g")
        .config("spark.driver.maxResultSize",          "2g")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.adaptive.enabled",          "true")
        .config("spark.sql.shuffle.partitions",        "8")
        .config("spark.default.parallelism",           "8")
        .config("spark.ui.showConsoleProgress",        "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def spark_read_parquet(spark: SparkSession, path: str):
    """
    Parquet 디렉터리 또는 단일 파일 읽기 헬퍼.

    Windows 로컬 모드에서 spark.read.parquet(directory)가
    NativeIO.access0 UnsatisfiedLinkError를 발생시키는 문제를
    Python glob으로 파일 목록을 수집 후 개별 경로 전달하여 우회.
    """
    p = Path(path)
    if p.suffix == ".parquet" and p.is_file():
        return spark.read.parquet(str(p))
    if p.is_dir():
        files = sorted(p.glob("**/*.parquet"))
        if not files:
            raise FileNotFoundError(f"Parquet 파일 없음: {path}")
        return spark.read.parquet(*[str(f) for f in files])
    # 파일로 직접 존재하지 않는 경우 디렉터리 탐색
    parent = p.parent
    if parent.is_dir():
        files = sorted(parent.glob("*.parquet"))
        if files:
            return spark.read.parquet(*[str(f) for f in files])
    raise FileNotFoundError(f"경로를 찾을 수 없음: {path}")
