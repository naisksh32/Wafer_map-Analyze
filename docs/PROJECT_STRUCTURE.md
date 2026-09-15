# 프로젝트 구조 및 파일 구성 정리

> 작성일: 2026-09-13 · 기준 커밋: `2f3bcc3` 이후 정리 작업
> 목적: 저장소를 새 머신에서 받은 상태에서 "무엇이 있고, 무엇이 없고, 어떤 순서로 재현하는가"를 한 장으로 파악하기 위함

---

## 1. 한눈에 보는 구성

```
self/                                   ← 저장소 루트 (WM-811K 웨이퍼 불량 검출 프로젝트)
│
├── 01~12, 16 *.ipynb                    ★ 단계별 노트북 (실험 기록 · 시각화 · 산출물 생성 원본)
├── scripts/                             ★ 노트북과 독립적으로 재현 가능한 실행 스크립트
├── src/                                 ★ 재사용 Python 모듈 (모델 · 데이터로더 · 분석기)
├── spark_pipeline/                        PySpark Bronze→Silver→Gold 레이크 파이프라인
├── 07_airflow_pipeline/                   Airflow DAG + docker-compose
├── dashboard/                             Vue 3 What-If 대시보드 (모델 미연동, 규칙 기반)
│
├── data/                                  데이터 (대용량은 gitignore, 아래 §3 참고)
├── checkpoints/                           학습 체크포인트 (gitignore, 재학습으로 생성)
├── analysis/                              성능 지표 JSON/CSV · 그림 (git 추적)
├── reports/                               공정 최적화 리포트 · ROI CSV · HTML 대시보드
├── configs/                               증강 설정 YAML · 불량 메타데이터 JSON
├── docs/                                  문서 (이 파일, 불량 메커니즘 분석, 계획서 모음)
├── logs/                                  재학습 로그
│
├── README.md                              프로젝트 소개 · 성능표 · 한계 (외부 공개용)
├── Report.md                              전체 결과 보고서 (상세)
├── PROJECT_PLAN.md                        단계별 계획 · 진행 현황
├── requirements.txt / requirements-full.txt
└── .gitignore
```

---

## 2. 파일별 역할

### 2.1 노트북 (실행 순서 = 번호 순서)

| 노트북 | 단계 | 하는 일 | 생성 산출물 |
|---|---|---|---|
| `01_data_download.ipynb` | P1-S1 | 환경 확인, kagglehub 로 WM-811K 다운로드 | (캐시) `~/.cache/kagglehub/.../LSWMD.pkl` |
| `02_eda.ipynb` | P1-S2 | 클래스 분포 · 웨이퍼 크기 · 평균 불량 히트맵 | `analysis/eda_*`, `class_distribution.png` 등 |
| `03_preprocessing.ipynb` | P1-S3 | 64×64 리사이즈 · Stratified 70/15/15 분할 · 증강 정의 | `data/processed/all_maps_resized.npy`, `split_indices.pkl`, `configs/augmentation_config.yaml`, **`src/data_loader.py` 를 코드로 생성** |
| `04_baseline_model.ipynb` | P1-S4 | WaferCNN 학습 | `checkpoints/WaferCNN_*.pth`, `analysis/baseline_*`, **`src/model.py` 를 코드로 생성** |
| `05_finetuning.ipynb` | P1-S5 | MobileNetV3 / EfficientNet-B0 / ViT-Tiny 2-Phase 파인튜닝 | `checkpoints/{모델}_*.pth`, `analysis/finetuning_*` |
| `06_mlflow_tracking.ipynb` | P1-S6 | MLflow + Optuna 20 trials HPO | `mlruns/`, `checkpoints/WaferCNN_best_hpo.pth`, `analysis/hpo_*` |
| `07_airflow_pipeline/` | P1-S7 | (노트북 아님) Airflow DAG 8 task | — |
| `08_onnx_deployment.ipynb` | P1-S8 | ONNX 변환 · INT8 양자화 · 속도 벤치마크 | `analysis/deployment_summary.json`, `onnx_speed_benchmark.png` |
| `09_defect_mechanism.ipynb` | P2-S9 | 9종 불량의 물리적 원인 · 공간 통계 | `docs/defect_mechanism_analysis.md`, `configs/defect_metadata.json` |
| `10_process_correlation.ipynb` | P2-S10 | **시뮬레이션** 공정 파라미터 생성 + 상관분석 | `data/process_parameters.csv`, `analysis/correlation_*` |
| `11_advanced_model.ipynb` | P2-S11 | Multi-output 모델 + Integrated Gradients | `checkpoints/AdvancedDefectPredictor_*.pth`, `analysis/advanced_*`, `shap_*` |
| `12_process_optimization.ipynb` | P2-S12 | Differential Evolution 최적화 + ROI | `reports/*` |
| `16_spark_pipeline.ipynb` | DE | PySpark 파이프라인 데모 (13~15 번호는 비어 있음) | `data/lake/` (gitignore) |

> `src/model.py`, `src/data_loader.py` 는 노트북 04/03 이 **문자열로 덤프해서 생성**하는 구조다.
> 두 파일을 직접 수정하면 노트북을 재실행할 때 덮어써지므로, 수정 시 노트북 쪽 문자열도 함께 바꿔야 한다.

### 2.2 `scripts/` — 노트북 없이 재현하는 경로

| 스크립트 | 역할 | 입력 → 출력 |
|---|---|---|
| `preprocess.py` **(신규)** | 노트북 03 의 핵심 단계를 스크립트화 | `data/raw/LSWMD.pkl` → `data/processed/{all_maps_resized.npy, split_indices.pkl, class_weights.npy}` |
| `retrain_baseline.py` | WaferCNN 재학습 (Adam 3e-4, 40ep, patience 10) | → `checkpoints/WaferCNN_{ep}_{valF1}.pth`, `analysis/baseline_results.json` |
| `retrain_finetune.py` | 사전학습 3종 2-Phase 파인튜닝 | → `checkpoints/{MobileNetV3,EfficientNet-B0,ViT-Tiny}_*.pth`, `analysis/finetuning_results.json` |
| `retrain_advanced.py` | Multi-output 모델 (Focal Loss) | → `checkpoints/AdvancedDefectPredictor_best_*.pth`, `analysis/advanced_model_results.json` |
| `evaluate_all.py` | **전 모델을 동일 테스트셋에서 재평가** (README 표의 근거) | 위 체크포인트들 → `analysis/final_evaluation.json`, `per_class_metrics.csv`, `final_confusion_matrices.png`, `baseline_classification_report.txt` |
| `fair_compare.py` **(신규)** | 모든 아키텍처를 **동일 프로토콜**로 학습·평가 (다중 seed, bootstrap CI) | → `analysis/fair_compare/{runs/*.json, summary.md, summary.json, summary.csv}` |
| `export_onnx.py` | MobileNetV3 ONNX FP32/INT8 변환 · 정확도 검증 · 벤치마크 | → `checkpoints/MobileNetV3_deploy*.onnx`, `analysis/deployment_summary.json` |
| `run_all_retrain.py` | 위 retrain 3종 순차 실행 + Report.md 갱신 | 로그 → `logs/` |
| `update_report.py` | 결과 JSON 으로 `Report.md` 의 표 문자열 치환 | (패턴 불일치 시 경고만 출력) |
| `phase1_validate.py` | 원본 pkl 로드 검증 · 기본 통계 | → `analysis/data_summary.json` |

### 2.3 `src/` — 재사용 모듈

| 모듈 | 내용 |
|---|---|
| `data_loader.py` | `WaferMapDataset`, `build_train_transform()` (Albumentations 1.4.x API), `get_dataloaders()`, `CLASS_ORDER` |
| `model.py` | `WaferCNN` (4 Conv block + GAP, 1.21M) |
| `advanced_defect_predictor.py` | MobileNetV3-S 백본 + 3 head (불량 / 심각도 / 신뢰도) |
| `model_explainer.py` | Integrated Gradients · Grad-CAM |
| `defect_analyzer.py` | 불량 공간 통계 (ring_ratio, center_ratio, defect_density …) |
| `process_correlation_analyzer.py` | Pearson/Spearman + 로지스틱 회귀 (시뮬레이션 데이터용) |
| `process_optimizer.py` | Differential Evolution + ROI 계산 |

### 2.4 `analysis/` — 어떤 파일이 "현재 유효한" 수치인가

| 파일 | 상태 | 비고 |
|---|---|---|
| `final_evaluation.json`, `per_class_metrics.csv`, `final_confusion_matrices.png`, `baseline_classification_report.txt` | **유효 (기준 문서)** | `evaluate_all.py` 산출물. README 성능표의 근거 |
| `baseline_results.json`, `finetuning_results.json`, `advanced_model_results.json` | 유효 | 각 retrain 스크립트 산출물. 단, 이전 머신의 절대경로가 남아 있었음 → 재학습 시 상대경로로 저장되도록 수정 |
| `finetuning_comparison.csv` | **폐기 대상 (stale)** | 노트북 05 의 초기 실행값(MobileNetV3 Acc 37.19% 등)으로 `final_evaluation.json` 과 모순. 재학습 후 재생성 |
| `data_summary.json` | 부분 stale | `class_imbalance_strategy` 가 "Sampler + Class Weight" 로 남아 있으나 실제 학습은 Sampler 단독 |
| `hpo_results.json`, `hpo_trials.csv`, `hpo_analysis.png` | 기록용 | HPO 실패 사례. 체크포인트(`WaferCNN_best_hpo.pth`)는 이 머신에 없음 |
| `deployment_summary.json` | 유효 | `export_onnx.py` 산출물 |
| 나머지 `*.png` | 시각화 | 노트북 실행 시 생성 |

---

## 3. 이 머신에 **없는** 것 (gitignore) 와 복원 방법

| 경로 | 크기 | 복원 |
|---|---|---|
| `data/raw/LSWMD.pkl` | 2.0 GB | `kagglehub.dataset_download('qingyi/wm811k-wafer-map')` 후 복사 (노트북 01 또는 `logs/setup_env.log` 절차) |
| `data/processed/all_maps_resized.npy` | 676 MB | `python scripts/preprocess.py` |
| `data/processed/split_indices.pkl` | 4 MB | 〃 (SEED 42 → README 와 동일한 테스트셋 25,943개) |
| `checkpoints/*.pth` | 5~25 MB/개 | `scripts/retrain_*.py` 또는 `scripts/fair_compare.py` |
| `mlruns/` | — | 노트북 06 재실행 |
| `.venv/` | ~6 GB | `python -m venv .venv` → `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124` → `pip install -r requirements.txt` |

---

## 4. 이번 정리에서 변경한 것

| 변경 | 이유 |
|---|---|
| `requirements.txt` UTF-16LE → **UTF-8** 변환 | BOM 없는 UTF-16 은 `pip install -r` 이 읽지 못함 (내용 변화 없음) |
| `upgrade.md` → `docs/plans/upgrade_plan_v1.md` | 루트에 흩어진 계획 문서 4개를 `docs/plans/` 로 모음 |
| `upgrade2.md` → `docs/plans/industry_gap_analysis.md` | 〃 |
| `upgrade3.md` → `docs/plans/phase3_multilabel_plan.md` | 〃 |
| `DataEngineer.md` → `docs/plans/spark_data_engineering_plan.md` | 〃 |
| `scripts/run_remaining.py` 삭제 | `run_all_retrain.py` 에서 베이스라인만 뺀 일회성 복제본 |
| `dashboard/src/components/HelloWorld.vue`, `assets/vue.svg`, `assets/vite.svg` 삭제 | Vite 스캐폴드 잔여물, 어디서도 import 되지 않음 |
| `scripts/preprocess.py` 추가 | 노트북 03 없이 학습 데이터 산출물 재생성 |
| `scripts/fair_compare.py` 추가 | 모델 간 공정 비교 (§5) |
| `scripts/retrain_*.py` | (a) `NUM_WORKERS` 환경변수로 DataLoader 워커 수 제어 (b) 체크포인트 경로를 **저장소 상대경로**로 기록 (c) `checkpoints/`, `analysis/` 자동 생성 |
| `scripts/retrain_advanced.py` | `finetuning_results.json` 의 상대경로 체크포인트를 해석 · 결과 JSON 에 `checkpoint` 필드 추가 |
| `scripts/evaluate_all.py` | 하드코딩된 체크포인트 파일명 제거 → 결과 JSON 에서 동적으로 해석 · macro P/R · **bootstrap 95% CI** 추가 |

변경하지 않았지만 알아둘 것:

- `16_spark_pipeline.ipynb` 는 UTF-8 BOM 이 붙어 있어 `json.load(open(..., encoding='utf-8'))` 로는 열리지 않는다 (`utf-8-sig` 필요). Jupyter 는 정상 로드.
- `spark_pipeline/run_pipeline.py` docstring 에 이전 머신 절대경로(`C:\Users\naisk\...`)가 남아 있다.
- `PROJECT_PLAN.md` 가 참조하는 `Plus_Plan.md`, `CLAUDE.md` 는 저장소에 없다 (`CLAUDE.md` 는 gitignore).
- `logs/retrain_*.log` 는 이전 머신의 로그이며 cp949 로 깨진 문자가 포함. 이번 재학습 로그로 대체된다.

---

## 5. 성능 수치가 "부정확"해 보이는 이유 — 코드 분석

README 는 WaferCNN(1.21M, from-scratch) macro F1 **0.8458** > ViT-Tiny 0.8352 > … > MobileNetV3-Small **0.7369** 이라고 기재한다.
`scripts/retrain_baseline.py` 와 `scripts/retrain_finetune.py` 를 비교하면 이 순서는 **아키텍처가 아닌 학습 레시피 차이**로도 충분히 설명된다.

| 항목 | WaferCNN (`retrain_baseline.py`) | 사전학습 3종 (`retrain_finetune.py`) |
|---|---|---|
| 초기화 | 랜덤 | ImageNet (첫 conv 3→1ch 평균) |
| 입력 해상도 | 64×64 → 4 회 pooling → **4×4** 후 GAP | 64×64 → stride 32 → **2×2** 후 GAP (MobileNetV3/EffNet) · ViT 는 16 토큰 |
| 옵티마이저 / LR | Adam **3e-4** 전체 파라미터 | P1: head 만 1e-3 (5ep) → P2: **backbone 1e-5**, head 1e-4 |
| Epoch 예산 | 40 (patience 10) | 5 + 35 (patience 10) |
| MobileNetV3 best epoch | — | **34 / 35** → 조기종료 직전까지 개선 중 = 미수렴 |

- **입력 해상도**: ImageNet 백본은 224 입력을 가정한다. 64×64 를 그대로 넣으면 마지막 stage 의 공간 해상도가 2×2 로 붕괴해 "링/중심/선형" 같은 전역 배치 정보가 사라진다. WaferCNN 은 64 입력에 맞춰 설계되어 4×4 를 유지한다.
- **백본 LR 1e-5**: 자연 이미지 → 3값 이산 웨이퍼 맵은 도메인 격차가 매우 크다. 백본 LR 이 head 의 1/10, WaferCNN 의 1/30 이면 35 epoch 안에 적응하기 어렵다.
- **단일 seed**: val macro F1 이 epoch 간 0.74~0.85 로 크게 흔들린다(`logs/retrain_baseline.log`). 소수 클래스(val Near-full 22개, Donut 83개)가 macro 평균의 1/9 씩을 차지하므로 한두 샘플 차이가 F1 을 0.01~0.03 움직인다. 단일 seed 의 0.8458 vs 0.8352 차이는 이 잡음 안에 있을 수 있다.

따라서 "커스텀 CNN 이 MobileNet 을 이겼다"는 결론을 검증하려면 (1) 같은 레시피 (2) 입력 해상도 통제 (3) 다중 seed + 신뢰구간이 필요하다. 이것이 `scripts/fair_compare.py` 의 설계이며, 결과는 `analysis/fair_compare/summary.md` 에 기록된다.

**검증 결과 (2026-09-13~14 실행, RTX 4070 Laptop):** 기존 레시피 수치는 재현되었으나(WaferCNN 0.8508 / MobileNetV3 0.7372), 동일 레시피에서는 MobileNetV3-S 64px 0.8341 ± 0.0044 ≥ WaferCNN 0.8311 ± 0.0088 이고, 128px 업샘플 시 0.8756 ± 0.0026 (40 ep 0.8847), EfficientNet-B0 128px 0.8894 로 역전된다. 전체 표는 [`MODEL_PERFORMANCE.md`](MODEL_PERFORMANCE.md), README 의 "모델 성능 비교" 절도 이 결과로 갱신했다.

### 5.-2 3차 — seed 확정 · 엣지 배포 재검증 · 결과 검증 (2026-09-15, 계획서 [`plans/next_steps_2026-09-15.md`](plans/next_steps_2026-09-15.md))

- Stage E `fair_compare.py --seeds 43 44`: mv3_pre160 · mv3_pre128_norm · effb0_pre224 를 3 seeds 로 승격.
- Stage F `export_onnx_v2.py`: MobileNetV3-S 160px 를 업샘플 포함 ONNX 로 변환. FP32 손실 0 · 예측 일치 100% · b=1 0.70ms. INT8 동적 양자화는 F1 0.22 로 기각 → `analysis/deployment_summary_v2.json`, README 배포 절 교체.
- Stage G `verify_results.py`: 체크포인트 재채점(G1·G2) · 집계 일관성(G3) · 문서 수치 대조(G4) · ONNX 동등성(G5) · 테스트셋 무결성(G6) · 그림 재생성(G7) → `analysis/verification_report.json`. GPU 재채점은 cuDNN 비결정성 때문에 F1 ≤ 1e-3 · 예측 불일치 ≤ 0.1% 를 동일성 기준으로 삼는다.

### 5.-1 2차 실험 — 모델별 특화 입력 전처리 (2026-09-14~15)

`scripts/fair_compare.py` 에 `Preprocess` 모듈(GPU 위 해상도·보간·채널·ImageNet 정규화)을 추가해 9개 구성을 seed 42 로 학습했다. 결과: 해상도가 유일하게 큰 변인이며(MobileNetV3-S 64→128→160→224px = 0.834→0.876→0.888→0.889, 160px 포화), 보간·채널·정규화는 잡음 범위, WaferCNN 128px 대조군은 개선 없음(0.8196). 전체 최고는 EfficientNet-B0 224px 0.9072. 그래프는 `scripts/plot_results.py` → `analysis/figures/fig1~6`, 상세는 [`MODEL_PERFORMANCE.md`](MODEL_PERFORMANCE.md) 3~4절.

### 5.0 체크포인트 정리 (2026-09-14)

재학습 스크립트가 val F1 개선마다 새 파일을 저장해 74개(814MB)가 쌓였던 것을 정리했다. 남긴 것은 `analysis/final_evaluation.json` 이 참조하는 기존 레시피 best 5개와 `checkpoints/fair/` 14개(154MB)이며, 경로 ↔ 결과 매핑은 `checkpoints/MANIFEST.json`(git 추적)에 있다. 삭제된 중간·Phase 1 체크포인트는 어떤 코드·문서에서도 참조되지 않음을 삭제 전에 확인했다. 앞으로의 학습 스크립트는 best 1개만 덮어쓰는 방식으로 바꿔야 한다.

### 5.1 실행 중 확인한 환경 이슈 (재현 시 참고)

| 이슈 | 증상 | 조치 |
|---|---|---|
| pypi 기본 `torch` 는 CPU 빌드 | `timm` 설치가 CPU torch 2.14 를 끌어옴 → `cuda False` | `--index-url https://download.pytorch.org/whl/cu124` 로 torch 2.6.0 재설치 |
| `LSWMD.pkl` 이 Python 2 / pandas 0.1x pickle | `No module named 'pandas.indexes'`, `UnicodeDecodeError` | `scripts/preprocess.py` 의 `_LegacyPandasUnpickler(encoding='latin1')` |
| Galaxy Book4 Ultra dGPU 전력 제한 | SW Power Cap 20W · SM 375~750 MHz (정상 55~80W · 2.3 GHz) → 5~8배 느림 | Samsung Settings 성능 모드 "고성능" 필요 (CLI 로 해제 불가). 배터리 충전 완료 후 일부 완화 |
| 노트북 절전 진입 | 학습 35분 정지 (Kernel-General 시간 재동기화 이벤트) | `powercfg /change standby-timeout-ac 0` · `hibernate-timeout-ac 0` |
| Windows DataLoader 워커 | 기본 `num_workers=0` | `NUM_WORKERS=4` 환경변수 + `persistent_workers=True` 로 spawn 비용 1회 |

---

## 6. 재현 순서 (이 머신, GPU)

```powershell
# 0) 가상환경 + CUDA PyTorch
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt

# 1) 데이터
.venv\Scripts\python -c "import kagglehub,shutil; p=kagglehub.dataset_download('qingyi/wm811k-wafer-map'); shutil.copy(p+'/LSWMD.pkl','data/raw/LSWMD.pkl')"
.venv\Scripts\python scripts\preprocess.py

# 2) 기존 레시피 그대로 재학습 → 통합 평가
$env:NUM_WORKERS=4; $env:PYTHONUTF8=1
.venv\Scripts\python scripts\retrain_baseline.py
.venv\Scripts\python scripts\retrain_finetune.py
.venv\Scripts\python scripts\retrain_advanced.py
.venv\Scripts\python scripts\evaluate_all.py          # → analysis/final_evaluation.json

# 3) 동일 프로토콜 공정 비교 (다중 seed)
.venv\Scripts\python scripts\fair_compare.py --models wafercnn mv3_pre64 mv3_pre128 mv3_scratch64 effb0_pre128 vit_pre64 --seeds 42 43 44
```
