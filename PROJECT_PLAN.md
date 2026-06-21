# WM-811K 반도체 웨이퍼 불량 검출 프로젝트 계획서

> **작성일:** 2026-06-20 | **최종 수정:** 2026-06-21 | **버전:** v2.2
> **목적:** SK하이닉스 취업 대비 반도체 웨이퍼 불량 분석 및 AI를 활용한 자동화 기술 숙지
> **직무 타겟:** Device Engineering — 공정 최적화 및 수율 개선 역량 어필

---

## 1. 프로젝트 목적

### 1.1 핵심 목적

반도체 제조 공정에서 발생하는 웨이퍼 불량 패턴을 딥러닝으로 자동 분류하는 것을 넘어,
**왜 해당 불량이 발생하는가** — 물리적 메커니즘 규명 및 공정 파라미터와의 인과관계까지 분석하는
**의사결정 지원 시스템**을 구축한다.

### 1.2 달성 목표

| 목표 | 내용 |
|------|------|
| 도메인 이해 | 반도체 웨이퍼 불량 패턴 9종의 분류 및 물리적 원인 이해 |
| 데이터 역량 | EDA → 클래스 불균형 해소 → Albumentations 증강 파이프라인 |
| 모델 역량 | CNN 베이스라인 → 사전학습 모델 파인튜닝 → Multi-output 고도화 |
| MLOps 역량 | MLFlow 실험 추적 + Optuna 튜닝 → Airflow 파이프라인 자동화 |
| 소자 엔지니어링 역량 | 불량 메커니즘 분석 → 공정 파라미터 상관관계 → 공정 최적화 알고리즘 |
| DevOps 역량 | ONNX 경량화 → 엣지 디바이스(Raspberry Pi) 배포 검증 |

### 1.3 기존 MLOps 대비 고도화 포인트

| 관점 | Phase 1 (기본 구현) | Phase 2 (고도화) |
|------|-------------------|----------------|
| 모델 질문 | "무엇이 불량인가?" | "왜 불량이 발생했는가?" |
| 출력값 | 불량 분류 | 분류 + 심각도 + 신뢰도 |
| XAI | Grad-CAM | SHAP (픽셀 영역 기여도) |
| 공정 연계 | 없음 | 공정 파라미터 상관관계 분석 |
| 최적화 | 없음 | Bayesian Opt + ROI 계산 |
| 배포 | 없음 | ONNX → Raspberry Pi |

---

## 2. 데이터셋 개요

**WM-811K (Wafer Map 811K)**
- 출처: Kaggle — `qingyi/wm811k-wafer-map`
- 전체 샘플: 811,457개 / 레이블링된 샘플: 172,950개
- 입력 형식: 2D 웨이퍼 맵 (값: 0=빈 영역, 1=정상 다이, 2=불량 다이)

| 클래스 | 설명 | 주요 발생 공정 |
|--------|------|--------------|
| None | 정상 | — |
| Center | 중심부 불량 | Wafer Preparation, Annealing |
| Donut | 도넛형 불량 | Lithography, Etching |
| Edge-Loc | 엣지 국소 불량 | CMP, PVD Deposition |
| Edge-Ring | 엣지 링형 불량 | Cleaning, Thermal Anneal |
| Loc | 국소 점상 불량 | PVD Chamber, Ion Implantation |
| Near-full | 거의 전체 불량 | Multiple |
| Random | 무작위 불량 | 청정실 환경 문제 |
| Scratch | 스크래치 | Wafer Handling, Transport |

- **주요 도전:** 극심한 클래스 불균형 (None 클래스가 레이블 데이터의 대다수)

---

## 3. 전체 프로젝트 구조

```
웨이퍼 불량 분석/
├── PROJECT_PLAN.md             ← 이 파일 (v2.0)
├── Plus_Plan.md                ← 고도화 원본 계획서
├── CLAUDE.md                   ← Claude 작업 규칙
├── requirements.txt            ← 패키지 목록
├── .venv/                      ← 가상환경
│
│ ── [Phase 1: 기본 MLOps 구현 - 7주] ──────────────────
├── 01_data_download.ipynb      ← 환경 설정 & 데이터 준비
├── 02_eda.ipynb                ← 탐색적 데이터 분석
├── 03_preprocessing.ipynb      ← 전처리 & Albumentations 증강
├── 04_baseline_model.ipynb     ← CNN 베이스라인
├── 05_finetuning.ipynb         ← 사전학습 모델 파인튜닝 (MobileNetV3, timm)
├── 06_mlflow_tracking.ipynb    ← MLFlow + Optuna 하이퍼파라미터 튜닝
├── 07_airflow_pipeline/        ← Airflow DAG 파이프라인
│   ├── dags/
│   │   └── wafer_pipeline_dag.py
│   └── docker-compose.yaml
├── 08_onnx_deployment.ipynb    ← ONNX 변환 & 엣지 배포 검증
│
│ ── [Phase 2: 소자 엔지니어링 고도화 - 5주] ────────────
├── 09_defect_mechanism.ipynb   ← 불량 메커니즘 분석
├── 10_process_correlation.ipynb← 공정-불량 상관관계 분석
├── 11_advanced_model.ipynb     ← Multi-output 모델 + SHAP
├── 12_process_optimization.ipynb← 공정 최적화 알고리즘 + ROI
│
│ ── [공통 자원] ────────────────────────────────────────
├── src/                        ← 재사용 Python 모듈
│   ├── data_loader.py
│   ├── augmentation.py
│   ├── model.py
│   ├── train.py
│   ├── mlflow_utils.py
│   ├── defect_analyzer.py          ← Phase 2
│   ├── process_correlation_analyzer.py ← Phase 2
│   ├── process_optimizer.py        ← Phase 2
│   └── model_explainer.py          ← Phase 2
├── configs/
│   ├── augmentation_config.yaml
│   ├── model_config.yaml
│   └── defect_metadata.json        ← 불량 메타데이터 (Phase 2)
├── analysis/                   ← 분석 리포트
├── reports/                    ← 최적화 리포트
├── checkpoints/                ← 모델 저장소
├── data/                       ← 로컬 캐시 데이터
└── mlruns/                     ← MLFlow 실험 기록
```

---

## 4. 단계별 상세 계획

---

### Phase 1 — 기본 MLOps 구현 (7주)

---

#### Step 1 — 환경 설정 & 데이터 준비 `(01_data_download.ipynb)` ✅ 완료

**작업 내용:**
- [x] .venv 가상환경 구성 및 requirements.txt 작성 ✅
- [x] PyTorch 2.6.0+cu124 (RTX 2060 SUPER) 설치 확인 ✅
- [x] `data/raw/LSWMD.pkl` 로컬 로드 (kagglehub 캐시 활용) ✅
- [x] 데이터 파일 구조 확인 (pickle 형식) ✅
- [x] 기본 로드 테스트 및 샘플 확인 ✅

**출력물:** `analysis/data_summary.json`, `scripts/phase1_validate.py`

---

#### Step 2 — 탐색적 데이터 분석 (EDA) `(02_eda.ipynb)`

**작업 내용:**
- [x] 전체 데이터 크기, 컬럼 구조 확인 (811,457개 웨이퍼 맵) ✅
- [x] 레이블 있는 데이터 (172,950개) vs 없는 데이터 비율 확인 ✅
- [x] 클래스 분포 시각화 (bar chart, pie chart) ✅
- [x] 클래스별 불량 패턴 샘플 이미지 갤러리 (각 클래스 5개) ✅
- [x] 웨이퍼 맵 크기 분포 분석 (가변 크기) ✅
- [x] 클래스 불균형 정량화 (Imbalance Ratio) ✅
- [x] 패턴별 불량 다이 비율 통계 (mean, std, skewness) ✅
- [x] 클래스별 평균 불량 히트맵 (공간 분포) ✅

**핵심 인사이트 목표:**
- 각 불량 패턴의 시각적 특징 및 공간적 분포 파악
- 클래스 불균형 처리 전략 결정

**산출물:** `analysis/eda_report.md`, `analysis/class_distribution.png`, `analysis/eda_summary.csv`, `analysis/avg_defect_heatmap.png`

---

#### Step 3 — 데이터 전처리 & 증강 `(03_preprocessing.ipynb)`

**작업 내용:**
- [x] 레이블 없는 데이터 필터링 ✅
- [x] 웨이퍼 맵 → 고정 크기 이미지 변환 (64×64) ✅
- [x] 픽셀값 정규화 (0~2 → 0.0~1.0) ✅
- [x] Stratified Train/Val/Test 분리 (70/15/15) ✅
- [x] 클래스 불균형 처리: ✅
  - Weighted Random Sampler
  - Class Weight 계산 (CrossEntropyLoss 가중치)
- [x] **Albumentations 기반 증강 파이프라인 구축:** ✅
  ```
  Rotate(±20°), HorizontalFlip, VerticalFlip,
  GaussNoise, Blur, RandomBrightness, CoarseDropout
  ```
- [x] 증강 효과 검증 (Before/After 분포 비교) ✅
- [x] PyTorch Dataset / DataLoader 클래스 구현 ✅

**산출물:** `configs/augmentation_config.yaml`, `analysis/augmentation_comparison.png`, `data/processed/split_indices.pkl`, `src/data_loader.py`

---

#### Step 4 — 베이스라인 CNN 모델 `(04_baseline_model.ipynb)` ✅ 완료

**작업 내용:**
- [x] 커스텀 CNN 설계 (WaferCNN — 4 Conv Block + GAP + FC Head) ✅
- [x] 학습 루프 구현 (train_one_epoch / eval_epoch 분리) ✅
- [x] 평가 지표: Accuracy, F1-Score (macro), Confusion Matrix ✅
- [x] 학습 곡선 시각화 (Loss / Accuracy / F1 추이) ✅
- [x] 베이스라인 성능 기록 (`analysis/baseline_results.json`) ✅

**구현 세부:**
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=30)
- Early Stopping: patience=7 (val F1 기준)
- CrossEntropyLoss + class weight (불균형 보정)

**목표 성능:** F1-Score macro ≥ 0.80
**산출물:** `checkpoints/WaferCNN_*.pth`, `analysis/baseline_results.json`, `analysis/baseline_training_curves.png`, `analysis/baseline_confusion_matrix.png`, `src/model.py`

---

#### Step 5 — 사전학습 모델 파인튜닝 `(05_finetuning.ipynb)` ✅ 완료

**작업 내용:**
- [x] **MobileNetV3 Small 주력** (경량화 + 실시간 추론 목적) ✅
  - `torchvision.models.mobilenet_v3_small` (pretrained, 1ch 적응 — RGB 3ch 가중치 평균)
- [x] `timm`으로 비교 모델 추가: `efficientnet_b0`, `vit_tiny_patch16_224` ✅
- [x] 분류 헤드 교체 (9-class output) ✅
- [x] 단계별 파인튜닝 (2-Phase): ✅
  1. Phase 1: Feature Extractor Freeze → Head만 학습 (lr=1e-3)
  2. Phase 2: 전체 레이어 Unfreeze → Discriminative LR (backbone 5e-5 / head 1e-3)
- [x] CosineAnnealing LR Scheduler + Early Stopping (patience=5) ✅
- [x] 최적 모델 체크포인트 저장 ✅

**구현 세부:**
- 3개 모델 비교: MobileNetV3 / EfficientNet-B0 / ViT-Tiny
- 결과: `analysis/finetuning_results.json`, `analysis/finetuning_comparison.csv`

**목표 성능:** Accuracy ≥ 90%, F1-Score macro ≥ 0.88
**산출물:** `checkpoints/MobileNetV3_*.pth`, `checkpoints/EfficientNet*.pth`, `analysis/finetuning_results.json`, `analysis/finetuning_comparison.csv`

---

#### Step 6 — MLFlow 실험 관리 + Optuna 튜닝 `(06_mlflow_tracking.ipynb)` ✅ 완료

**작업 내용:**
- [x] MLFlow 3.14.0 SQLite 백엔드 설정 (`sqlite:///mlruns/mlflow.db`) ✅
  - ⚠️ MLFlow 3.14.0 파일스토어 maintenance mode 차단 이슈 해결
  - `--index-url` → `--extra-index-url` 로 수정 (PyPI + PyTorch 인덱스 병행)
- [x] 실험 생성: `wafer-defect-detection` (ID: 1) ✅
- [x] Step 4/5 결과 소급 로깅 (3개 파인튜닝 모델) ✅
- [x] **Optuna HPO 20 trials 완료:** ✅
  - Sampler: TPESampler(seed=42)
  - Pruner: MedianPruner(n_startup_trials=5, n_warmup_steps=2)
  - 완료 trial: 7개 / 조기종료(pruning): 13개
  - **Best Val F1: 0.5581 (Trial #3)**
  - **최적 파라미터:**
    ```
    lr           : 7.31e-05
    batch_size   : 32
    dropout      : 0.2169
    weight_decay : 1.26e-04
    ```
- [x] 최적 파라미터로 30 epoch 재학습 + Early Stopping ✅
- [x] 모델 레지스트리 등록: `WaferDefectCNN` (champion alias) ✅
- [x] MLFlow UI 확인: `mlflow ui --backend-store-uri "sqlite:///mlruns/mlflow.db"` ✅

**주요 버그 수정:**
- `train_one_epoch` 이중 forward pass 버그 수정 (logits 변수 재사용)
- `boxplot(labels=)` → `boxplot(tick_labels=)` (matplotlib 3.9+ 호환)

**산출물:** `mlruns/mlflow.db`, `analysis/hpo_results.json`, `analysis/hpo_trials.csv`, `analysis/hpo_analysis.png`, `checkpoints/WaferCNN_best_hpo.pth`

---

#### Step 7 — Airflow 파이프라인 `(07_airflow_pipeline/)` ✅ 환경 구성 완료

**작업 내용:**
- [x] Docker Compose로 Airflow 2.9.0 환경 구성 ✅
  - PostgreSQL 메타데이터 DB + Webserver + Scheduler
  - ⚠️ `--index-url` → `--extra-index-url` 수정 (scikit-learn 설치 오류 해결)
- [x] DAG 설계 및 구현: `wafer_defect_pipeline` (8개 Task) ✅

```
load_data → eda_analysis → data_classification → data_augmentation
→ model_training → model_evaluation → onnx_conversion → model_deployment
```

- [x] 각 Task Python 함수 구현 (`dags/wafer_pipeline_dag.py`) ✅
- [x] Task 간 의존성 정의 (`>>` 연산자) ✅
- [x] 스케줄링: `@weekly`, `catchup=False` ✅
- [x] 오류 처리 및 재시도 로직 (`retries=1, retry_delay=5min`) ✅
- [x] Airflow UI (`http://localhost:8080`, admin/admin) 정상 접속 확인 ✅

> **참고:** 파이프라인 실행은 포트폴리오 시연용이며, 이후 단계(Step 8~12)와 의존관계 없음.
> 실행 명령: `cd 07_airflow_pipeline && docker compose up -d`

**산출물:** `07_airflow_pipeline/dags/wafer_pipeline_dag.py`, `07_airflow_pipeline/docker-compose.yaml`

---

#### Step 8 — ONNX 변환 & 엣지 배포 검증 `(08_onnx_deployment.ipynb)` ✅ 완료

> **Plus_Plan.md 추가 단계**

**작업 내용:**
- [x] MobileNetV3 → ONNX 변환 (opset 14, 동적 배치) ✅
- [x] ONNX 모델 검증 (`onnx.checker.check_model`) ✅
- [x] 동적 양자화 (Dynamic Quantization) ✅
- [x] 추론 속도 벤치마크 (PyTorch CPU/GPU vs ONNX CPU/Quantized) ✅
- [x] Raspberry Pi 배포 시뮬레이션 (batch=1, 1,000 iterations) ✅
- [x] Grad-CAM 시각화 (오분류 샘플 6개) ✅

**실제 결과:**

| 항목 | 결과 | 목표 |
|------|------|------|
| PyTorch Test F1 | 0.5618 | ≥ 0.88 |
| ONNX 모델 크기 | 5.83 MB → 1.62 MB (**72.2% 감소**) | ≤ 60% |
| ONNX CPU 추론 | **6.53ms avg** (PyTorch CPU 16.34ms 대비 **2.5× 빠름**) | GPU ≥ PyTorch |
| RPi 시뮬레이션 (원본) | **0.77ms avg**, 1,305 IPS | < 50ms ✅ |
| RPi 시뮬레이션 (양자화) | 3.59ms avg, 278 IPS | < 50ms ✅ |

> **참고:** 양자화 후 F1이 0.023으로 급락 (정확도 희생). 실 배포는 원본 ONNX 권장.

**산출물:** `checkpoints/MobileNetV3_opset14.onnx`, `checkpoints/MobileNetV3_quantized.onnx`, `analysis/deployment_summary.json`, `analysis/onnx_speed_benchmark.png`

---

### Phase 2 — 소자 엔지니어링 고도화 (5주)

> Phase 1 완성 후 진행. SK하이닉스 Device Engineering 직무 차별화 포인트.

---

#### Step 9 — 불량 메커니즘 분석 `(09_defect_mechanism.ipynb)` ✅ 완료

> **Plus_Plan.md Week 8**

**작업 내용:**
- [x] 9가지 불량 패턴의 물리적 원인 규명 ✅

| 불량 | 발생 공정 | 핵심 파라미터 |
|------|----------|-------------|
| Center | Wafer Preparation, Annealing | 온도, 냉각 속도 |
| Edge-Loc | CMP, PVD Deposition | CMP 압력, 폴리싱 시간 |
| Edge-Ring | Cleaning, Thermal Anneal | 열처리 온도 구배 |
| Scratch | Wafer Handling | Handler 압력, 속도 |
| Random | 청정실 환경 | 청정도 등급 |
| ... | ... | ... |

- [x] 불량 메타데이터 설계 및 저장 (`defect_metadata.json`) ✅
  - `physical_mechanism`, `primary_process_stage`, `equipment_involved`
  - `critical_parameters` (name, unit, range, criticality)
  - `severity_level`, `yield_impact`, `remediation`
- [x] 공간 지표 정량화: `center_ratio`, `edge_ratio`, `ring_ratio`, `defect_density`, `spatial_entropy` ✅
- [x] `src/defect_analyzer.py` 구현 ✅

**산출물:** `configs/defect_metadata.json`, `src/defect_analyzer.py`, `docs/defect_mechanism_analysis.md`, 클래스별 히트맵 PNG 5종

---

#### Step 10 — 공정-불량 상관관계 분석 `(10_process_correlation.ipynb)` ✅ 완료

> **Plus_Plan.md Week 9-10**

**작업 내용:**
- [x] 공정 파라미터 시뮬레이션 데이터 생성 ✅
  - 파라미터: `cmp_pressure`, `polish_time`, `annealing_temp`, `temp_gradient`, `slurry_ph`, `etch_depth`, `vacuum_pressure`, `particle_count`, `pr_thickness_cv`
  - 클래스별 상관 구조 설계 (`CLASS_BIAS` 딕셔너리) — 도메인 지식 기반
- [x] Pearson / Spearman 이중 상관분석 ✅
  - `ProcessCorrelationAnalyzer.correlation_matrix(defect_class)` 구현
- [x] Critical Parameter 식별 (|r| ≥ 0.6) ✅
- [x] 상관계수 히트맵 (불량 클래스 × 공정 파라미터) ✅
- [x] `predict_defect_probability(param_values, defect_class)` 메서드 구현 (sigmoid 스케일) ✅
- [x] `defect_metadata.json`에 `correlation_evidence` 필드 추가 ✅

**산출물:** `data/process_parameters.csv`, `analysis/correlation_heatmap.png`, `src/process_correlation_analyzer.py`, `analysis/correlation_analysis_report.md`

---

#### Step 11 — Multi-output 고도화 모델 + SHAP `(11_advanced_model.ipynb)` ✅ 완료

> **Plus_Plan.md Week 11**

**작업 내용:**
- [x] **Multi-output 모델 설계** (`AdvancedDefectPredictor`) ✅
  - Backbone: MobileNetV3 Small (1ch 적응, Step 5 체크포인트 로드)
  - Shared Layer: Linear(576→256) + Hardswish + Dropout(0.3)
  - Head 1 (defect): Linear(256→128) → ReLU → Linear(128→9)
  - Head 2 (severity): Linear(256→64) → ReLU → Linear(64→4)
  - Head 3 (confidence): Linear(256→32) → ReLU → Linear(32→1) → Sigmoid
- [x] **Multi-Task Loss 함수:** ✅
  ```
  L = 0.5·CE(defect) + 0.3·CE(severity) + 0.2·MSE(confidence)
  ```
- [x] Discriminative LR: backbone 5e-5 / head 1e-3 ✅
- [x] **XAI — Integrated Gradients:** ✅
  - `IGWrapper`: `flatten(1).clone()`으로 view 충돌 차단
  - `clear_all_hooks()` + `disable_inplace()`: MobileNetV3 호환성 확보
  - `compute_integrated_gradients()`: 순수 PyTorch autograd 구현 (SHAP 동등)
  - 픽셀 기여도 히트맵 + Overlay 시각화
  - ※ `shap.DeepExplainer` / `shap.GradientExplainer` 모두 MobileNetV3와 비호환 확인 → IG로 대체
- [x] Grad-CAM: `backbone[-1][0]` 레이어 훅 등록 ✅
- [x] `SHAP_AVAILABLE` 플래그로 후속 셀 흐름 제어 ✅

**산출물:** `models/advanced_defect_predictor.py`, `src/model_explainer.py`, `checkpoints/AdvancedDefectPredictor_*.pth`, `analysis/shap_pixel_importance.png`

---

#### Step 12 — 공정 최적화 알고리즘 + ROI 계산 `(12_process_optimization.ipynb)` ✅ 완료

> **Plus_Plan.md Week 12**

**작업 내용:**
- [x] **시나리오 분석 (What-If Analysis):** ✅
  - 단일 파라미터 변화에 따른 불량률 변화 시뮬레이션
  - 50개 구간으로 파라미터 범위 스캔
- [x] **Differential Evolution 최적화:** ✅
  ```python
  scipy.optimize.differential_evolution(
      objective_function,   # 불량률 최소화
      seed=42, maxiter=100, workers=-1
  )
  ```
  - 목적함수: `0.7 × 불량률 + 0.2 × 비용증가 + 0.1 × 제약위반`
- [x] **경제성 모델 (ROI):** ✅
  ```
  월간 웨이퍼 생산: 50,000개 × $500 = $25M/월
  수율 개선분 → 월간 이득 → 투자비 대비 회수기간(payback)
  ```
- [x] 공정 최적화 리포트 자동 생성 (`generate_optimization_report()`) ✅
- [x] Plotly 대시보드 (`reports/optimization_dashboard.html`) 선택 생성 ✅

**산출물:** `src/process_optimizer.py`, `reports/process_optimization_report.md`, `reports/scenario_analysis.csv`, `reports/roi_summary.csv`

---

## 5. 기술 스택 요약

| 영역 | 기술 |
|------|------|
| 언어 | Python 3.12 |
| 데이터 처리 | pandas, numpy, scikit-learn, scipy |
| 시각화 | matplotlib, seaborn, plotly |
| 이미지 처리 | Pillow, opencv-python |
| **데이터 증강** | **albumentations** |
| 딥러닝 | PyTorch 2.6.0+cu124, torchvision |
| **사전학습 모델** | **MobileNetV3 (주력), timm (EfficientNet, ViT)** |
| 클래스 불균형 | imbalanced-learn |
| **하이퍼파라미터 튜닝** | **Optuna** |
| 실험 관리 | MLFlow 3.14 |
| 파이프라인 | Apache Airflow |
| 컨테이너 | Docker, Docker Compose |
| **모델 경량화/배포** | **ONNX, onnxruntime** |
| **XAI** | **SHAP, grad-cam** |
| **통계 분석** | **statsmodels, scipy.stats** |
| **부스팅** | **XGBoost (심각도 예측 대안)** |
| 데이터 소스 | Kaggle (kagglehub) |
| Jupyter | jupyter, jupyterlab, ipywidgets |

---

## 6. 진행 현황 (2026-06-21 기준)

### 전체 진행률 — **Phase 1 + Phase 2 전 단계 완료** 🎉

```
Phase 1 — 기본 MLOps
  Step 1  ✅ 완료   환경 설정 & 데이터 준비
  Step 2  ✅ 완료   EDA
  Step 3  ✅ 완료   전처리 & Albumentations 증강
  Step 4  ✅ 완료   WaferCNN 베이스라인
  Step 5  ✅ 완료   MobileNetV3 파인튜닝
  Step 6  ✅ 완료   MLFlow + Optuna HPO (20 trials, Best F1=0.5581)
  Step 7  ✅ 완료   Airflow Docker 환경 구성 & DAG 정의
  Step 8  ✅ 완료   ONNX 변환 & 엣지 배포 (ONNX 6.53ms, RPi 0.77ms, 72.2% 경량화)

Phase 2 — 소자 엔지니어링 고도화
  Step 9  ✅ 완료   불량 메커니즘 분석 (9종 물리적 원인 규명)
  Step 10 ✅ 완료   공정-불량 상관관계 분석 (Critical Parameter 식별)
  Step 11 ✅ 완료   Multi-output 모델 + Integrated Gradients XAI
  Step 12 ✅ 완료   공정 최적화 (Differential Evolution) + ROI 계산
```

### 주요 트러블슈팅 기록

| 단계 | 이슈 | 해결 |
|------|------|------|
| Step 9/10/11 | `failureType` numpy ndarray 파싱 오류 | `extract_label()` ndarray 처리 추가 |
| Step 9/10/11 | `split_indices` 키명 불일치 (`train` vs `train_idx`) | `.get()` fallback 패턴 적용 |
| Step 9/10/11/12 | matplotlib 한글 폰트 깨짐 | `Malgun Gothic` rcParams 설정 |
| Step 11 | `shap.DeepExplainer` — Hardswish 미지원 AssertionError | GradientExplainer 전환 |
| Step 11 | `shap.GradientExplainer` — CUDA + MobileNetV3 hook 충돌 | CPU 전환 시도 |
| Step 11 | MobileNetV3 `inplace=True` 25개 연산 → backward RuntimeError | `disable_inplace()` 적용 |
| Step 11 | 이전 SHAP hook 잔존 → `BackwardHookFunctionBackward` 충돌 | `clear_all_hooks()` + `flatten().clone()` |
| Step 11 | SHAP 전면 제거 → **Integrated Gradients (순수 PyTorch)** 로 대체 | `compute_integrated_gradients()` 구현 |
| Step 12 | `plotly` 미설치 | `pip install plotly` |

---

## 7. 추가 필요 패키지 (Phase 2)

Phase 2 진입 전 아래 패키지 추가 설치 필요:

```bash
pip install albumentations optuna shap onnx onnxruntime \
            xgboost statsmodels plotly
```

---

## 8. 리스크 및 대응 방안

| 리스크 | 심각도 | 대응 방안 |
|-------|--------|---------|
| 공정 파라미터 실데이터 부재 | High | 시뮬레이션 데이터 생성 + 상관구조 설계 |
| 모델 성능 미달 (< 85%) | Medium | Optuna 재튜닝, 증강 강화 |
| Airflow Docker 설정 복잡 | Low | Docker Compose로 로컬 환경 구축 |
| Raspberry Pi 호환성 | Low | ONNX 양자화로 모델 최적화 |
| Phase 2 시간 부족 | Medium | Phase 1 완성 우선, Phase 2는 Step 9~10까지 최소 목표 |

---

## 9. 포트폴리오 어필 포인트

1. **도메인 적합성** — 반도체 제조 공정 9가지 불량 패턴 물리적 원인까지 이해
2. **데이터 엔지니어링** — 클래스 불균형 + Albumentations 증강 파이프라인
3. **MLOps 실무** — MLFlow + Optuna + Airflow 완전 자동화
4. **소자 엔지니어링** — 공정 파라미터-불량 상관관계 분석 및 Critical Parameter 식별
5. **비즈니스 임팩트** — 수율 개선 ROI/NPV 정량화 (공정 최적화 결과)
6. **XAI** — SHAP로 "왜 불량인가" 픽셀 단위 근거 제시
7. **엣지 배포** — ONNX 경량화 → 실시간 추론 시스템 구현
8. **재현 가능성** — 노트북 번호 체계 + requirements.txt로 누구나 재현 가능

---

## 10. 참고 자료

- WM-811K Kaggle Dataset: `qingyi/wm811k-wafer-map`
- Albumentations Docs: https://albumentations.ai/
- MLflow Docs: https://mlflow.org/docs/latest/
- Apache Airflow Docs: https://airflow.apache.org/docs/
- SHAP Docs: https://shap.readthedocs.io/
- "WM-811K, A Large-scale Semiconductor Wafer Defect Dataset" (IEEE Trans. Semiconductor Manufacturing)
- "SHAP: A Unified Approach to Interpreting Model Predictions" (NIPS 2017)
