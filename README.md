# WM-811K 웨이퍼 불량 검출 — MLOps + 소자 엔지니어링 고도화

> **SK하이닉스 Device Engineering 직무 포트폴리오**  
> 반도체 웨이퍼 불량 패턴 분류를 넘어 물리적 원인 규명 → 공정 파라미터 최적화 → 수율 개선 ROI 정량화까지 연결한 의사결정 지원 시스템

---

## 핵심 성과

| 항목 | 수치 |
|------|------|
| 분석 데이터 | WM-811K 중 레이블 **172,950개** (클래스 불균형 989.5×) |
| 최고 분류 성능 | EfficientNet-B0 **F1-macro 0.667** |
| HPO 최적화 후 | Optuna 20 trials → F1-macro **0.599** |
| Multi-output 모델 | Val F1 **0.8255** (불량종류 + 심각도 + 신뢰도 동시 예측) |
| 모델 경량화 | 5.95MB → 1.62MB (**72.2% 감소**, ONNX 양자화) |
| 추론 속도 | CPU 16.34ms → ONNX **6.53ms** (2.5× 향상) |
| 엣지 배포 (RPi) | **0.77ms / 1,305 IPS** |
| 공정 최적화 ROI | 연간 **$2.23M** · 5년 NPV **$10.66M** |

---

## 프로젝트 구조

```
단순 분류 모델         →        의사결정 지원 시스템
"무엇이 불량인가?"              "왜 발생했고, 어떻게 고치나?"

이미지 분류 (9클래스)
    + 심각도 예측 (4단계)
    + 신뢰도 출력 (0~1)
    + 물리적 원인 규명
    + 공정 파라미터 상관관계 분석
    + 최적 파라미터 자동 계산
    + ROI / NPV 정량화
    + ONNX 엣지 배포
```

---

## 전체 파이프라인

```
┌─────────────────────────────── Phase 1 — 기본 MLOps ───────────────────────────────┐
│                                                                                    │
│  Step 1          Step 2          Step 3          Step 4          Step 5           │
│  데이터 준비  →   EDA       →   전처리/증강  →  베이스라인  →   파인튜닝            │
│  WM-811K         클래스분포       Albumentations   WaferCNN        MobileNetV3      │
│  172,950개       불균형분석       64×64 리사이즈   F1=0.501        EfficientNet-B0  │
│                                  불균형 보정                       ViT-Tiny         │
│                                                                                    │
│  Step 6          Step 7          Step 8                                            │
│  MLFlow/Optuna →  Airflow DAG →  ONNX 배포                                        │
│  20 trials        Docker 8Task   6.53ms CPU                                        │
│  F1=0.599         자동화          0.77ms RPi                                        │
└────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────── Phase 2 — 소자 엔지니어링 ──────────────────────────┐
│                                                                                    │
│  Step 9            Step 10           Step 11            Step 12                   │
│  불량 메커니즘  →  공정 상관관계  →  Multi-output  →   공정 최적화                 │
│  9종 물리 원인     Pearson/Spearman   모델 + XAI        Differential Evolution     │
│  공정 연결         Critical Param.    Integrated Grad.  ROI $2.23M/yr              │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 / 환경 | Python 3.12 · PyTorch 2.6.0+cu124 · CUDA 12.4 |
| 딥러닝 | torchvision · timm (EfficientNet, ViT) |
| 데이터 증강 | Albumentations |
| HPO | Optuna (TPE Sampler + Median Pruner) |
| 실험 관리 | MLFlow 3.14 (SQLite backend) |
| 파이프라인 | Apache Airflow 2.9 · Docker Compose |
| 배포 | ONNX · onnxruntime |
| XAI | Integrated Gradients (PyTorch) · Grad-CAM |
| 통계 | scipy.stats (Pearson / Spearman) |
| 최적화 | scipy.optimize.differential_evolution |
| 시각화 | matplotlib · seaborn · plotly |

---

## 데이터셋

**WM-811K (Wafer Map 811K)** — [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

| 항목 | 값 |
|------|-----|
| 전체 샘플 | 811,457개 |
| 레이블 샘플 (분석 대상) | 172,950개 |
| 클래스 수 | 9종 (none 포함) |
| 클래스 불균형 | **989.5×** (none vs Near-full) |
| 입력 형식 | 2D 웨이퍼 맵 (0=빈 영역 / 1=정상 다이 / 2=불량 다이) |
| 전처리 | 64×64 리사이징 · 픽셀 정규화 · Weighted Random Sampler |

### 클래스 분포

| 클래스 | 샘플 수 | 비율 |
|--------|--------|------|
| none | 147,431 | 85.24% |
| Edge-Ring | 9,680 | 5.60% |
| Edge-Loc | 5,189 | 3.00% |
| Center | 4,294 | 2.48% |
| Loc | 3,593 | 2.08% |
| Scratch | 1,193 | 0.69% |
| Random | 866 | 0.50% |
| Donut | 555 | 0.32% |
| Near-full | 149 | 0.09% |

---

## 모델 성능 비교

| 모델 | Test F1-macro | Test Accuracy | 비고 |
|------|-------------|--------------|------|
| WaferCNN (베이스라인) | 0.501 | 10.65% | 커스텀 4-Conv CNN |
| WaferCNN + Optuna HPO | 0.599 | 43.21% | lr=7.31e-5, batch=32 |
| MobileNetV3 Small | 0.562 | 37.19% | 2-Phase 파인튜닝 |
| ViT-Tiny | 0.647 | 75.57% | 2-Phase 파인튜닝 |
| **EfficientNet-B0** | **0.667** | **79.37%** | **최고 분류 성능** |
| AdvancedDefectPredictor | Val F1 **0.826** | — | Multi-output (분류+심각도+신뢰도) |

---

## Phase 2 — 소자 엔지니어링 고도화

### 불량 메커니즘 분석

9종 불량 패턴의 물리적 원인을 WM-811K 공간 통계 + 반도체 공정 도메인 지식으로 규명:

| 불량 | 원인 공정 | 핵심 지표 | 심각도 |
|------|----------|---------|--------|
| Edge-Ring | Thermal Oxidation / CVD | ring_ratio = **0.780** | High |
| Center | Wafer Preparation / Annealing | center_ratio = **0.372** | Critical |
| Edge-Loc | CMP / PVD Deposition | ring_ratio = 0.621 | High |
| Donut | Lithography / Etch | center_ratio = **0.445** | High |
| Scratch | Wafer Handling / CMP | 직선형 공간 분포 | Medium |
| Near-full | 복합 오염 / 결정 결함 | defect_density = **0.877** | Critical |

### 공정 파라미터 상관관계 (Pearson r, |r| ≥ 0.3)

| 불량 클래스 | Critical Parameter | Pearson r |
|-----------|------------------|----------|
| Edge-Ring | `temp_gradient` | **+0.664** |
| Donut | `pr_thickness_cv` | **+0.608** |
| Center | `annealing_temp` | **+0.527** |
| Loc | `vacuum_pressure` | **+0.534** |
| Edge-Loc | `polish_time` | **-0.554** |

### XAI — Integrated Gradients 픽셀 기여도

```
SHAP DeepExplainer → MobileNetV3 Hardswish 미지원 (AssertionError)
SHAP GradientExplainer → CUDA + inplace op 충돌 (RuntimeError)
Integrated Gradients (순수 PyTorch) → 정상 동작 ✅

IG(x) = (x - baseline) × ∫₀¹ ∇f(baseline + α(x-baseline)) dα
baseline: none 클래스 20개 평균 | steps: 30
```

### 공정 최적화 ROI

**Differential Evolution** 최적화 (`scipy.optimize`):
- 목적함수: `0.7 × 불량률 + 0.2 × 비용증가 + 0.1 × 제약위반`
- 월간 생산: 50,000장 × $500 = $25M

| 불량 | 월 이익 | 연간 이익 | ROI | 투자회수 |
|------|--------|---------|-----|--------|
| Edge-Ring | $58,487 | $701,845 | **10,976%** | **0.1개월** |
| Edge-Loc | $59,668 | $716,011 | 1,966% | 0.6개월 |
| Center | $46,252 | $555,024 | 216% | 3.8개월 |
| Donut | $5,687 | $68,248 | 139% | 5.0개월 |

> **총 연간 이익: $2,225,961 (약 30억 원) · 5년 NPV: $10,657,020 (약 146억 원)**

---

## 프로젝트 디렉토리

```
wafer-defect-analysis/
│
├── 01_data_download.ipynb          # 환경 설정 & 데이터 준비
├── 02_eda.ipynb                    # 탐색적 데이터 분석
├── 03_preprocessing.ipynb          # 전처리 & Albumentations 증강
├── 04_baseline_model.ipynb         # WaferCNN 베이스라인
├── 05_finetuning.ipynb             # MobileNetV3 / EfficientNet / ViT 파인튜닝
├── 06_mlflow_tracking.ipynb        # MLFlow + Optuna HPO
├── 07_airflow_pipeline/            # Apache Airflow DAG (Docker Compose)
│   ├── dags/wafer_pipeline_dag.py
│   └── docker-compose.yaml
├── 08_onnx_deployment.ipynb        # ONNX 변환 & 엣지 배포
├── 09_defect_mechanism.ipynb       # 불량 메커니즘 분석
├── 10_process_correlation.ipynb    # 공정-불량 상관관계 분석
├── 11_advanced_model.ipynb         # Multi-output 모델 + XAI
├── 12_process_optimization.ipynb   # 공정 최적화 + ROI
│
├── src/                            # 재사용 Python 모듈
│   ├── data_loader.py
│   ├── model.py
│   ├── advanced_defect_predictor.py
│   ├── defect_analyzer.py
│   ├── model_explainer.py
│   ├── process_correlation_analyzer.py
│   └── process_optimizer.py
│
├── configs/                        # 설정 파일
│   ├── augmentation_config.yaml
│   └── defect_metadata.json        # 9종 불량 메타데이터
│
├── checkpoints/                    # 최종 모델 가중치
│   ├── WaferCNN_best_hpo.pth
│   ├── MobileNetV3_15_0.5736.pth
│   ├── EfficientNet-B0_25_0.6775.pth
│   ├── ViT-Tiny_23_0.6578.pth
│   ├── AdvancedDefectPredictor_best_0.8255.pth
│   ├── MobileNetV3_opset14.onnx    # 배포 모델
│   └── MobileNetV3_quantized.onnx  # 경량화 모델 (1.62MB)
│
├── analysis/                       # EDA · 학습 결과 · 시각화 (PNG, JSON, CSV)
├── reports/                        # 최적화 리포트 · ROI 분석
├── docs/                           # 불량 메커니즘 상세 문서
├── scripts/                        # 유틸리티 스크립트
├── data/
│   ├── raw/LSWMD.pkl               # WM-811K 원본 (~2GB)
│   └── processed/                  # 전처리 완료 데이터 (NPY, PKL)
│
├── Report.md                       # 전체 결과 보고서
├── PROJECT_PLAN.md                 # 단계별 계획 및 진행 현황
└── requirements.txt
```

---

## 빠른 시작

### 환경 설정

```bash
# 가상환경 생성 및 패키지 설치
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt

# Phase 2 추가 패키지
pip install albumentations optuna shap onnx onnxruntime xgboost statsmodels plotly
```

### 데이터 준비

```bash
# Kaggle API 인증 후 다운로드 (01_data_download.ipynb 참조)
# 또는 data/raw/LSWMD.pkl 직접 배치
```

### 노트북 실행 순서

```
01 → 02 → 03 → 04 → 05 → 06 → 07(선택) → 08 → 09 → 10 → 11 → 12
```

### Airflow 파이프라인 실행 (선택)

```bash
cd 07_airflow_pipeline
docker compose up airflow-init   # 최초 1회
docker compose up -d
# http://localhost:8080  (admin / admin)
```

### MLFlow UI 확인

```bash
mlflow ui --backend-store-uri "sqlite:///mlruns/mlflow.db"
# http://localhost:5000
```

---

## 주요 결과 시각화

| 분석 | 파일 |
|------|------|
| 클래스 분포 | `analysis/class_distribution.png` |
| 클래스별 평균 불량 히트맵 | `analysis/avg_defect_heatmap.png` |
| 파인튜닝 모델 비교 | `analysis/finetuning_model_comparison.png` |
| Optuna HPO 분석 | `analysis/hpo_analysis.png` |
| ONNX 추론 속도 벤치마크 | `analysis/onnx_speed_benchmark.png` |
| 공정-불량 상관계수 히트맵 | `analysis/correlation_heatmap.png` |
| Integrated Gradients 픽셀 기여도 | `analysis/shap_pixel_importance.png` |
| 공정 최적화 시나리오 분석 | `analysis/scenario_analysis.png` |
| Grad-CAM 오분류 시각화 | `analysis/gradcam_misclassified.png` |

---

## 참고 문헌

1. Wu, M.-J. et al. (2015). *Wafer Map Failure Pattern Recognition and Similarity Ranking*. **IEEE Trans. Semiconductor Manufacturing**, 28(1), 1–12.
2. Quirk, M. & Serda, J. (2001). *Semiconductor Manufacturing Technology*. Prentice Hall.
3. Wolf, S. & Tauber, R. (2000). *Silicon Processing for the VLSI Era, Vol.1*. Lattice Press.
4. Sundararajan, M. et al. (2017). *Axiomatic Attribution for Deep Networks*. **ICML 2017**.
5. Shim, J. et al. (2020). *Wafer Defect Pattern Classification Using CNN*. **IEEE Access**, 8, 177499–177507.

---

## 라이선스

본 프로젝트는 포트폴리오 목적으로 작성되었습니다.  
데이터셋: WM-811K (Kaggle, 원본 라이선스 준수)
