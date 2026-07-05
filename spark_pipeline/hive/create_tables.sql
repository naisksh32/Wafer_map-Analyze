-- WM-811K 웨이퍼 데이터 레이크 Hive 테이블 DDL
-- Spark SQL / Hive Metastore 호환
-- 실행: spark.sql(open("spark_pipeline/hive/create_tables.sql").read())

-- ─────────────────────────────────────────────
-- 데이터베이스 생성
-- ─────────────────────────────────────────────
CREATE DATABASE IF NOT EXISTS wafer_db
  COMMENT 'WM-811K 웨이퍼 불량 분석 데이터 레이크 (Medallion Architecture)';

USE wafer_db;

-- ─────────────────────────────────────────────
-- Bronze: 웨이퍼 맵 원본 테이블
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wafer_db.bronze_wafer_maps (
    wafer_id       INT       COMMENT '웨이퍼 글로벌 인덱스 (0-based)',
    class_idx      INT       COMMENT '인코딩된 클래스 인덱스 (0=none ~ 8=Scratch)',
    failure_type   STRING    COMMENT '결함 유형 문자열',
    wafer_map_flat BINARY    COMMENT '64×64=4096 bytes uint8 직렬화 맵'
)
PARTITIONED BY (split STRING COMMENT 'train/val/test')
STORED AS PARQUET
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');

-- ─────────────────────────────────────────────
-- Bronze: 공정 파라미터 테이블
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wafer_db.bronze_process_params (
    wafer_id         INT     COMMENT '웨이퍼 인덱스 (JOIN 키)',
    cmp_pressure     DOUBLE  COMMENT 'CMP 공정 압력 (psi), 정상: 80~120',
    polish_time      DOUBLE  COMMENT '연마 시간 (sec), 정상: 25~45',
    slurry_ph        DOUBLE  COMMENT '슬러리 pH, 정상: 5.5~7.5',
    annealing_temp   DOUBLE  COMMENT '열처리 온도 (°C), 정상: 1050~1150',
    temp_gradient    DOUBLE  COMMENT '온도 그래디언트 (°C/cm), 정상: 0~2',
    etch_depth       DOUBLE  COMMENT '식각 깊이 (Å), 정상: 450~550',
    vacuum_pressure  DOUBLE  COMMENT '진공 압력 (Torr)',
    pr_thickness_cv  DOUBLE  COMMENT 'PR 두께 CV (%)',
    particle_count   DOUBLE  COMMENT '파티클 수, 정상: 0~10',
    defect_class     STRING  COMMENT '명목 결함 클래스',
    is_defect        INT     COMMENT '0=정상, 1=불량'
)
STORED AS PARQUET
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');

-- ─────────────────────────────────────────────
-- Silver: 정제 + JOIN 통합 테이블
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wafer_db.silver_wafer_labeled (
    wafer_id              INT,
    class_idx             INT,
    split                 STRING,
    wafer_map_flat        BINARY,
    -- 공정 파라미터
    cmp_pressure          DOUBLE,
    polish_time           DOUBLE,
    slurry_ph             DOUBLE,
    annealing_temp        DOUBLE,
    temp_gradient         DOUBLE,
    etch_depth            DOUBLE,
    vacuum_pressure       DOUBLE,
    pr_thickness_cv       DOUBLE,
    particle_count        DOUBLE,
    -- 파생 특징
    defect_density        DOUBLE  COMMENT '불량 다이 비율',
    edge_defect_ratio     DOUBLE  COMMENT '에지 영역 불량 비율',
    center_defect_ratio   DOUBLE  COMMENT '중심 영역 불량 비율',
    any_ooc               INT     COMMENT '공정 이상 플래그'
)
PARTITIONED BY (failure_type STRING COMMENT '결함 유형 파티션')
STORED AS PARQUET
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');

-- ─────────────────────────────────────────────
-- Gold: 특징 저장소 (wafer_map_flat 제외)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wafer_db.gold_feature_store (
    wafer_id              INT,
    class_idx             INT,
    split                 STRING,
    -- 공정 파라미터
    cmp_pressure          DOUBLE,
    annealing_temp        DOUBLE,
    particle_count        DOUBLE,
    -- 공간 특징
    defect_density        DOUBLE,
    active_ratio          DOUBLE,
    edge_defect_ratio     DOUBLE,
    center_defect_ratio   DOUBLE,
    radial_mean           DOUBLE,
    defect_count          INT,
    active_count          INT
)
PARTITIONED BY (failure_type STRING)
STORED AS PARQUET;

-- ─────────────────────────────────────────────
-- 분석용 View: SPC 경보
-- ─────────────────────────────────────────────
CREATE OR REPLACE VIEW wafer_db.v_spc_alert AS
SELECT
    wafer_id,
    failure_type,
    cmp_pressure,
    particle_count,
    defect_density,
    CASE
        WHEN particle_count > 15 OR cmp_pressure > 125 THEN 'CRITICAL'
        WHEN particle_count > 10 OR cmp_pressure > 115 THEN 'WARNING'
        ELSE 'OK'
    END AS spc_status
FROM wafer_db.silver_wafer_labeled
WHERE failure_type != 'none';

-- ─────────────────────────────────────────────
-- 분석 쿼리 예시 (주석 처리됨)
-- ─────────────────────────────────────────────

-- 1. 결함 유형별 집계
-- SELECT failure_type, COUNT(*) AS n, AVG(defect_density) AS avg_density
-- FROM wafer_db.silver_wafer_labeled
-- GROUP BY failure_type ORDER BY n DESC;

-- 2. 공정 파라미터 이상 탐지
-- SELECT wafer_id, failure_type, cmp_pressure, particle_count
-- FROM wafer_db.silver_wafer_labeled
-- WHERE failure_type != 'none' AND (cmp_pressure > 120 OR particle_count > 10)
-- ORDER BY particle_count DESC LIMIT 50;

-- 3. 분할별 클래스 불균형
-- SELECT split, failure_type, COUNT(*) AS n
-- FROM wafer_db.silver_wafer_labeled
-- GROUP BY split, failure_type
-- ORDER BY split, n DESC;
