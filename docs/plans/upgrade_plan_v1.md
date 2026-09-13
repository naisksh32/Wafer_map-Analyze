# WM-811K 웨이퍼 불량 검출 프로젝트 — 고도화 계획 보고서

> **작성일:** 2026-06-29 | **버전:** v1.0  
> **목적:** 현 프로젝트 성과 검토 + 반도체 산업 직무 연구 기반 발전 방향 제시  
> **타겟 기업:** 삼성전자(DS부문), SK하이닉스, 세메스(SEMES)

---

## 1. 현재 프로젝트 성과 검토

### 1.1 단계별 완료 현황

| 단계 | 내용 | 상태 |
|------|------|------|
| Phase 1 Step 1~3 | 환경 구축, EDA, 전처리 & 증강 | ✅ 완료 |
| Phase 1 Step 4 | WaferCNN 베이스라인 | ✅ 완료 |
| Phase 1 Step 5 | 사전학습 모델 파인튜닝 | ✅ 완료 |
| Phase 1 Step 6 | MLFlow + Optuna HPO | ✅ 완료 |
| Phase 1 Step 7 | Airflow DAG 파이프라인 | ✅ 완료 |
| Phase 1 Step 8 | ONNX 배포 | ✅ 완료 |
| Phase 2 Step 9~12 | 불량 메커니즘~공정 최적화 | ✅ 완료 |

### 1.2 모델 성능 분석

#### 베이스라인 vs 파인튜닝 비교

| 모델 | Val F1 (macro) | Test F1 (macro) | Test Accuracy | 목표 F1 | 달성 여부 |
|------|:-:|:-:|:-:|:-:|:-:|
| WaferCNN (커스텀 CNN) | **0.8485** | **0.8458** | 95.78% | ≥ 0.80 | ✅ 달성 |
| MobileNetV3 Small | 0.7417 | 0.7369 | 90.45% | ≥ 0.88 | ❌ 미달 |
| EfficientNet-B0 | 0.7999 | 0.7944 | 93.96% | ≥ 0.88 | ❌ 미달 |
| ViT-Tiny | 0.8351 | 0.8352 | 95.54% | ≥ 0.88 | ❌ 미달 |
| AdvancedDefectPredictor | 0.8279 | 0.8173 | 95.08% | ≥ 0.90 | ❌ 미달 |

#### 주요 성능 이슈 분석

**① 커스텀 CNN이 사전학습 모델을 앞지른 원인**

WaferCNN이 MobileNetV3를 F1 기준 +0.11 차이로 능가한 것은 역설적이나, 웨이퍼 맵 도메인 특성에서 기인한다. 웨이퍼 맵은 자연 이미지(ImageNet)와 근본적으로 다른 구조를 가진다.
- **픽셀값 3가지뿐** (0=빈영역, 1=정상다이, 2=불량다이) — 텍스처·색상 특징이 없음
- **공간 패턴이 핵심** (링 패턴, 에지 집중, 중심 집중 등) — 전역적 기하학 구조가 분류를 결정
- **64×64 저해상도** — 사전학습 특징이 과도하게 복잡할 수 있음
- 사전학습 모델의 ImageNet 귀납적 편향이 오히려 웨이퍼 맵 패턴 학습을 방해했을 가능성

**② INT8 양자화 정확도 붕괴 문제**

ONNX INT8 양자화 후 F1이 0.5618→0.023으로 붕괴되었다. 이는 단채널(1ch) 그레이스케일 입력에서 동적 양자화의 스케일 추정 오류에 기인한다. 실제 배포는 ONNX 원본(6.53ms, 2.5× 속도 향상) 유지가 권장된다.

**③ SHAP 호환성 문제 → Integrated Gradients 대체**

MobileNetV3의 `Hardswish(inplace=True)` 연산이 SHAP DeepExplainer 및 GradientExplainer 모두와 충돌하여 순수 PyTorch autograd 기반 Integrated Gradients로 대체하였다. 기능적으로는 동등하나, SHAP 생태계와의 표준 통합성이 부족한 점이 한계로 남는다.

**④ Optuna HPO 결과 (Best Val F1=0.5581) 편차 이유**

HPO 탐색 공간이 WaferCNN 기반이었고 20 trials 중 13개가 Pruning 조기종료되어 탐색이 충분하지 않았다. MobileNetV3/ViT 아키텍처에 맞는 별도 HPO가 필요하다.

---

## 2. 반도체 산업 직무 분석

### 2.1 SK하이닉스 — AI 기반 웨이퍼 검사 현황

SK하이닉스는 2023년 Panoptes VM (가우스랩스 AI 솔루션)을 박막 증착 공정에 도입하여 **공정 산포 21.5% 개선, 수율 향상**을 달성했다. 기존 샘플링 검사에서 AI 전수 예측으로 전환한 핵심 사례다.

주요 기술 방향:
- **장비 센서 데이터 → 공정 결과 예측**: 실시간 센서 빅데이터 기반 품질 예측
- **자율형 Fab 2030**: AI가 스스로 학습·의사결정하는 공장 (결함 분석 시간 50% 단축 목표)
- **HBM 생산라인 AI**: 장애 예측·장비 추적 고도화 (2026년 현재 진행 중)
- **초고해상도 TEM 이미지 AI**: 고려대 연구진과 협력, 나노 수준 결함 분석

**Device Engineering 직무에서 필요한 AI 역량:**
1. 공정 파라미터와 불량률 간 상관관계 분석 능력
2. 웨이퍼 맵 패턴 분류 및 불량 원인 규명
3. 수율 개선을 위한 최적 공정 조건 탐색
4. 공정 모니터링 지표 (Cpk, 산포) 이해와 ML 연계

### 2.2 삼성전자 DS부문 — AI 검사 개발

삼성전자 AI센터 내 AI 검사 개발 그룹이 수행하는 주요 직무:
- **비전 검사기 개발**: 광학 검사 장비와 AI 알고리즘 통합
- **AI 플랫폼 개발·운영**: 제조 라인 전체의 검사 데이터 수집·분석 플랫폼
- **신호/시스템 설계**: 장비 신호 처리와 딥러닝 융합

요구 역량:
- Computer Vision (CNN, Transformer)
- Real-time inference 최적화 (ONNX, TensorRT)
- 결함 데이터의 극심한 불균형 처리 경험
- XAI를 통한 모델 해석 가능성 확보

### 2.3 세메스(SEMES) — 장비 내장형 AI

세메스는 삼성전자 자회사이자 국내 최대 반도체 장비 기업으로, 전공정(세정, 포토트랙, 에치)과 후공정 장비 전반을 생산한다. 2025년 하반기 신입 채용 직무 분석:

| 직무 | AI/ML 관련 역량 |
|------|----------------|
| 머신러닝 엔지니어 | 장비 센서 데이터 이상 탐지, 예지 보전(PHM) |
| 소프트웨어 엔지니어 | 검사 알고리즘, SECS/GEM 통신 통합 |
| 공정/품질/테스트 | 공정 파라미터 분석, 수율 개선 |

세메스가 특히 주목하는 기술:
- **인라인 실시간 검사**: 공정 중 즉시 불량 판정 (타 공정 오염 방지)
- **장비 예지 보전(PHM)**: 웨이퍼 불량 패턴 변화로 장비 PM(Preventive Maintenance) 시점 예측
- **SECS/GEM 프로토콜**: 장비-호스트 통신 표준으로 AI 검사 결과 실시간 전송

---

## 3. 현 프로젝트의 갭(Gap) 분석

### 3.1 기술적 갭

| 항목 | 현재 상태 | 산업 표준 | 갭 |
|------|----------|----------|-----|
| 불량 분류 방식 | 단일 레이블 9분류 | **복합 불량(Mixed-type) 멀티레이블** | 실제 웨이퍼는 2가지 이상 불량 동시 발생 |
| 레이블 없는 데이터 활용 | 필터링 후 미사용 (638,507개) | **반지도 학습으로 활용** | 대규모 비레이블 데이터 낭비 |
| 공정 데이터 | 시뮬레이션 생성 | **실제 장비 센서 데이터** | 상관분석 신뢰도 제한 |
| 실시간 서비스 | ONNX 파일만 존재 | **REST API / 스트리밍 추론** | 실제 Fab 시스템 통합 불가 |
| 대시보드 | Plotly HTML 파일 | **실시간 웹 대시보드** | 엔지니어 사용성 부재 |
| 불량 조기 경보 | 없음 | **SPC 연동 EWS** | 수율 저하 사전 대응 불가 |

### 3.2 포트폴리오 표현 갭

| 항목 | 현재 | 개선 방향 |
|------|------|----------|
| 성능 목표 달성률 | 파인튜닝 모델 모두 미달 | 목표 달성 또는 미달 원인을 명확히 분석한 Technical Report |
| 산업 적용 스토리 | 코드 중심 | "이 기능이 Fab에서 어떻게 쓰이는가" 시나리오 서술 |
| 비즈니스 임팩트 | ROI 계산 존재하나 가상 데이터 | 실제 반도체 수율 지표와 연결된 임팩트 정량화 |

---

## 4. 발전 방향 — Phase 3 계획

### 4.1 우선순위 매트릭스

| 항목 | 기술 난이도 | 포트폴리오 임팩트 | 우선순위 |
|------|:-:|:-:|:-:|
| Mixed-type 멀티레이블 분류 | 중 | **최상** | 🔴 1순위 |
| Streamlit 실시간 추론 대시보드 | 하 | **상** | 🔴 1순위 |
| 반지도 학습 (비레이블 데이터 활용) | 상 | 상 | 🟡 2순위 |
| SPC 연동 조기 경보 시스템 | 중 | 상 | 🟡 2순위 |
| Vision Transformer 아키텍처 개선 | 상 | 중 | 🟢 3순위 |
| SECS/GEM 통신 시뮬레이터 | 상 | 중 (세메스 특화) | 🟢 3순위 |
| GAN 기반 소수 클래스 증강 | 상 | 중 | 🟢 3순위 |
| FastAPI 예측 서버 | 하 | 중 | 🟡 2순위 |

---

### 4.2 Phase 3 — Step 13: Mixed-type 불량 멀티레이블 분류

**배경:** 실제 반도체 제조에서 웨이퍼 하나에 2가지 이상의 불량이 동시에 나타나는 Mixed-type 결함이 빈번하게 발생한다. 현재 단일 레이블 분류 시스템으로는 이를 처리하지 못한다. 2025년 최신 연구(SemiWaferNet, ViT-VAE-GAN)에서도 Mixed-type이 핵심 과제로 다루어지고 있다.

**구현 계획 (13_mixed_type_detection.ipynb):**

```
1. 데이터 준비
   - WM-811K에서 Mixed-type 패턴 재레이블링 (규칙 기반 + 수동 검토)
   - 공간 지표(center_ratio, edge_ratio 등)를 이용한 자동 혼합 패턴 생성
   - 멀티레이블 Binary Cross-Entropy 손실 함수 적용

2. 모델 아키텍처
   - Vision Transformer (ViT-Small) — 전역 어텐션이 혼합 패턴에 유리
   - Multi-head 출력 (9개 불량 클래스 × 독립 이진 분류기)
   - 클래스별 임계값 최적화 (macro F1 최대화)

3. 평가 지표
   - Micro/Macro F1 (멀티레이블)
   - Hamming Loss
   - Subset Accuracy (완전 일치율)

4. 산업 연계 서술
   - "Center + Scratch 동시 발생 = 핸들링 중 열 응력 가능성" 등
     복합 불량 → 복합 공정 원인 분석 스토리
```

**기대 효과:** 단일 레이블 시스템 대비 실제 Fab 적용 가능성 대폭 향상, 최신 연구 트렌드(2025 ViT 기반 Mixed-type) 반영

---

### 4.3 Phase 3 — Step 14: Streamlit 실시간 추론 대시보드

**배경:** 현재 프로젝트는 Jupyter Notebook 중심으로 구성되어 있어, 면접관이나 비개발자 엔지니어에게 시연하기 어렵다. 실제 Fab에서 엔지니어가 사용하는 검사 시스템처럼 직관적인 웹 UI가 포트폴리오의 설득력을 크게 높인다.

**구현 계획 (src/dashboard/app.py 또는 14_dashboard.ipynb):**

```
핵심 기능:
├── [탭 1] 웨이퍼 맵 분석기
│   ├── pkl/npy 파일 업로드 또는 랜덤 샘플 선택
│   ├── 불량 유형 예측 (ONNX 모델 추론, <10ms)
│   ├── Grad-CAM 히트맵 오버레이 표시
│   └── 불량 신뢰도 막대그래프
│
├── [탭 2] 공정 파라미터 분석
│   ├── 파라미터 슬라이더 (실시간 불량 확률 업데이트)
│   ├── Critical Parameter 강조 표시
│   └── What-If 시나리오 비교
│
├── [탭 3] 배치 분석 & 수율 대시보드
│   ├── 다수 웨이퍼 일괄 업로드 → 자동 분류
│   ├── 로트(Lot)별 불량 분포 시각화
│   ├── 이상 트렌드 탐지 알림
│   └── CSV 리포트 자동 다운로드
│
└── [탭 4] 모델 성능 모니터링
    ├── 클래스별 정밀도/재현율
    ├── Confusion Matrix 인터랙티브
    └── 모델 드리프트 지표
```

**실행 방법:**
```bash
streamlit run src/dashboard/app.py
# → localhost:8501에서 즉시 시연 가능
```

**기대 효과:** 면접·포트폴리오 시연 시 코드 대신 실동작 앱으로 임팩트 극대화

---

### 4.4 Phase 3 — Step 14-B: Vue.js 공정 시뮬레이션 대시보드 ✨ 신규 추가

**배경:** Streamlit은 Python 서버가 필요하지만, Vue.js 기반 정적 HTML 페이지는 서버 없이 브라우저에서 즉시 실행되어 포트폴리오 시연에 최적이다. 공정 파라미터를 실시간으로 조작하며 웨이퍼 불량 패턴이 변하는 모습을 보여주는 인터랙티브 시뮬레이터는 면접관이나 비개발자 엔지니어에게 프로젝트의 핵심 가치를 즉각적으로 전달한다.

**구현 파일:** `dashboard/wafer_simulation.html` (Vue 3 CDN + Chart.js CDN, 서버 불필요)

**핵심 기능:**

```
[공정 파라미터 제어 패널]
├── 9개 슬라이더 (CMP 압력, 연마 시간, 슬러리 pH, 어닐링 온도,
│   온도 구배, 식각 깊이, 진공 압력, PR 두께 균일도, 파티클 수)
├── 정상 범위 시각화 (슬라이더 내 하이라이트 구간)
├── 파라미터 상태 인디케이터 (정상/경고/위험)
└── 프리셋 버튼 (정상 운전 / 경고 단계 / 위험 단계)

[웨이퍼 불량 맵 (Canvas, 실시간)]
├── 64×64 die 격자 (448px × 448px)
├── 불량 유형별 패턴 렌더링 (Center/Donut/Edge-Ring 등)
├── 공정 흐름 영향 분석 다이어그램
└── 드리프트 애니메이션 (▶ 시뮬레이션)

[불량 위험도 분석 패널]
├── 불량 확률 수평 막대 차트 (Chart.js, 8종 × 실시간 업데이트)
├── 주요 불량 유형 상세 카드 (메커니즘, 심각도, 수율 영향)
├── 개선 조치 권고 사항
└── 수율 & 경제적 영향 (ROI, 월간 손실 추정)
```

**기술 스택:** Vue 3 Composition API (CDN), Chart.js 4.4, Canvas API, CSS Grid

**세메스 타겟 어필:** "인라인 검사 장비가 측정한 공정 파라미터를 이 시스템에 입력하면 즉시 불량 패턴을 예측하고 개선 조치를 제안합니다"

---

### 4.5 Phase 3 — Step 15: FastAPI 예측 서버 + Docker 컨테이너화

**배경:** 세메스 같은 장비 회사는 검사 결과를 다른 시스템(MES, 장비 호스트)에 전송해야 하고, SK하이닉스·삼성전자 내부 AI 플랫폼도 REST API 방식으로 모델을 서빙한다.

**구현 계획 (15_api_deployment/ 폴더):**

```
구조:
15_api_deployment/
├── app/
│   ├── main.py          ← FastAPI 앱 (엔드포인트 정의)
│   ├── model_loader.py  ← ONNX 모델 로드 (싱글턴)
│   ├── schemas.py       ← 요청/응답 Pydantic 스키마
│   └── predict.py       ← 전처리 + 추론 로직
├── Dockerfile
├── docker-compose.yaml
└── tests/
    └── test_api.py      ← pytest 기반 API 단위 테스트

핵심 엔드포인트:
POST /predict           ← 웨이퍼 맵 배열 → 불량 분류 + 신뢰도
POST /predict/batch     ← 다수 웨이퍼 일괄 처리
GET  /health            ← 헬스체크
GET  /model/info        ← 모델 메타데이터
GET  /metrics           ← Prometheus 메트릭 노출
```

**세메스 타겟 강조 포인트:**

현실의 반도체 장비는 SECS/GEM 프로토콜을 통해 호스트와 통신한다. 이 API 서버는 SECS/GEM 게이트웨이와 통합될 수 있는 인터페이스로 설계되었음을 명시하여, 세메스 직무와의 연결고리를 만든다.

```
[SECS/GEM 장비] → [게이트웨이] → [FastAPI 서버] → [AI 불량 판정] → [MES 피드백]
```

---

### 4.5 Phase 3 — Step 16: SPC 연동 조기 경보 시스템 (EWS)

**배경:** SK하이닉스 Panoptes VM의 핵심 기능 중 하나가 공정 산포 모니터링이다. 웨이퍼 불량률이 특정 임계를 넘기 전에 엔지니어에게 알림을 주는 조기 경보 시스템은 실제 Fab에서 가장 필요로 하는 기능이다.

**구현 계획 (16_early_warning_system.ipynb):**

```
1. SPC (Statistical Process Control) 지표 구현
   ├── UCL/LCL (Upper/Lower Control Limit): μ ± 3σ
   ├── Cpk (공정 능력 지수): Cpk < 1.33 → 관리 이탈 경고
   ├── Western Electric Rules: 연속 7점 한 방향, 2/3 경고선 초과 등
   └── EWMA (지수 가중 이동평균) 차트: 소폭 산포 이상 탐지

2. 연속 웨이퍼 스트림 시뮬레이션
   ├── 시계열 웨이퍼 데이터 생성 (정상→이상 전환 포함)
   ├── 슬라이딩 윈도우 기반 실시간 불량률 계산
   └── 임계 초과 시 경보 트리거 (레벨 1/2/3 등급화)

3. 장비 PM 필요 시점 예측
   ├── Edge-Ring 불량 증가 → 특정 장비 열처리 온도 이탈 신호
   ├── Scratch 패턴 증가 → 핸들러 마모 진행 신호
   └── 예측 신뢰도와 함께 PM 스케줄 제안

4. 산업 적용 스토리 (세메스 타겟)
   "세메스 에치 장비에서 Edge-Loc 불량이 3배 증가하면
    이 시스템이 자동으로 CMP 압력 드리프트를 감지하고
    예방 정비 일정을 제안합니다"
```

---

### 4.6 Phase 3 — Step 17: 반지도 학습 (Semi-supervised Learning)

**배경:** WM-811K의 638,507개 비레이블 데이터를 현재 완전히 버리고 있다. 반도체 제조 현장에서 레이블링 비용은 극히 높기 때문에(숙련 엔지니어가 수동 검토), 비레이블 데이터를 학습에 활용하는 반지도 학습은 산업적으로 매우 중요하다.

**구현 계획 (17_semi_supervised.ipynb):**

```
방법 1 — FixMatch (이미지 분류 SOTA)
├── 레이블 데이터: 172,950개 (강한 증강)
├── 비레이블 데이터: 638,507개 (약한 증강 → Pseudo-label 생성)
├── 신뢰도 임계값 0.95 이상 Pseudo-label만 학습에 포함
└── 기대: F1 macro 5~10% 향상 가능

방법 2 — Self-supervised Pretraining (SimCLR/MoCo)
├── 전체 811,457개 웨이퍼 맵으로 대조 학습 사전훈련
├── 학습된 표현을 분류 헤드에 파인튜닝
└── 기대: 소수 클래스(Near-full, Donut, Scratch) F1 대폭 향상

산업 가치 어필:
"실제 Fab에서는 레이블 없는 검사 이미지가 레이블 이미지보다
 수백 배 많습니다. 반지도 학습은 이 데이터를 자산으로 전환합니다"
```

---

### 4.7 Phase 3 — Step 18: 모델 성능 재도전 (목표 달성)

**현재 문제:** 파인튜닝 모델이 모두 F1 0.88 목표에 미달하여 포트폴리오의 신뢰도를 해치고 있다.

**개선 전략:**

```
① 웨이퍼 맵 전용 사전학습 (Domain Adaptation)
   - ImageNet 가중치 대신 웨이퍼 맵 데이터로 자체 사전학습
   - MobileNetV3 1채널 입력의 RGB 평균화 방식 → 독립 채널 학습으로 교체

② 고해상도 입력 실험 (64×64 → 128×128)
   - 해상도 증가로 미세 패턴(Scratch, Loc) 인식률 향상
   - VRAM 8GB 내에서 batch_size 조정 필요

③ Label Smoothing + Focal Loss 결합
   - Focal Loss(γ=2)로 소수 클래스 집중 학습
   - Label Smoothing(ε=0.1)으로 과적합 방지

④ EfficientNet-B0에 Test-Time Augmentation (TTA)
   - 추론 시 8방향 회전/플립 앙상블
   - 기대 F1 향상: +0.03~0.05

⑤ 모델 앙상블 (WaferCNN + ViT-Tiny)
   - Soft Voting: WaferCNN(0.4) + ViT-Tiny(0.6)
   - 기대 F1: 0.87~0.89 (목표 달성 가능)
```

---

## 5. 산업별 포트폴리오 어필 전략

### 5.1 SK하이닉스 Device Engineering 직무

**핵심 메시지:** "웨이퍼 불량 분류 AI를 직접 구현하고, 공정 파라미터와의 인과관계 분석 및 수율 개선 ROI까지 end-to-end로 다뤄본 엔지니어"

강조할 포트폴리오 요소:
1. **Step 9~10**: 9종 불량의 물리적 메커니즘 규명 (CMP 압력, 열처리 온도, 청정도 등)
2. **Step 10**: Critical Parameter 식별 (|r| ≥ 0.6 공정 파라미터)
3. **Step 12**: Differential Evolution + ROI 계산 ($25M/월 생산 기준 수익성 분석)
4. **Step 16 (신규)**: SPC 기반 EWS로 수율 저하 사전 대응

면접 스토리:
> "저는 단순히 불량을 분류하는 것을 넘어, Edge-Ring 불량이 발생하는 원인이 열처리 온도 구배라는 것을 데이터로 입증하고, 해당 파라미터를 최적화했을 때 수율이 얼마나 개선되는지 ROI로 계산했습니다. SK하이닉스의 Panoptes VM과 유사한 접근 방식입니다."

### 5.2 삼성전자 DS부문 AI 검사 개발

**핵심 메시지:** "Computer Vision 기반 비전 검사 시스템을 처음부터 구현하고, XAI로 모델 해석 가능성까지 확보한 엔지니어"

강조할 포트폴리오 요소:
1. **Step 5**: ViT-Tiny vs CNN 비교 실험 (Transformer 기반 검사 시스템 경험)
2. **Step 8**: ONNX 경량화 (실시간 검사 요구사항 충족, 6.53ms 추론)
3. **Step 11**: Integrated Gradients XAI ("왜 불량인가" 픽셀 단위 근거 제시)
4. **Step 13 (신규)**: Mixed-type 멀티레이블 분류 (실제 Fab 적용 가능성)
5. **Step 14 (신규)**: Streamlit 대시보드 (실동작 데모)

### 5.3 세메스(SEMES) 머신러닝/소프트웨어 엔지니어

**핵심 메시지:** "장비 통합을 고려한 AI 추론 서버를 설계하고, 불량 패턴으로 장비 PM 시점까지 예측하는 시스템을 구현한 엔지니어"

강조할 포트폴리오 요소:
1. **Step 15 (신규)**: FastAPI 서버 (SECS/GEM 통합 가능한 REST API 설계)
2. **Step 16 (신규)**: 불량 패턴 변화 → 장비 PM 예측 (장비 회사 핵심 니즈)
3. **Step 7**: Airflow 파이프라인 자동화 (공정 자동화 사고방식)
4. **Step 8**: ONNX + 엣지 배포 (라인 내 인라인 검사 시스템 구현 경험)

---

## 6. 기술 부채 및 즉시 개선 사항

### 6.1 즉시 수정 가능한 문제들 (1~2일 내)

| 문제 | 현상 | 해결 방법 |
|------|------|----------|
| INT8 양자화 F1 붕괴 | 0.5618 → 0.023 | FP16 양자화로 교체, 또는 static quantization + calibration data 제공 |
| Optuna HPO F1 낮음 (0.5581) | WaferCNN 기반 탐색 | MobileNetV3/ViT 각 모델별 독립 HPO 실행 |
| 파인튜닝 목표 미달 | F1 0.88 미달 | TTA + 앙상블 적용 (Step 18) |
| SHAP 미지원 | IG 대체 사용 | 포트폴리오 README에 기술 배경 명확히 서술 |

### 6.2 포트폴리오 README 보완

현재 README가 없거나 미흡할 경우:
```
README.md 구성:
├── 프로젝트 요약 (3줄 executive summary)
├── 성능 지표 표 (모델 비교)
├── 시스템 아키텍처 다이어그램
├── 실행 방법 (Quick Start)
├── 핵심 결과 시각화 (5~6장)
└── 산업 적용 시나리오
```

---

## 7. 최종 발전 로드맵

```
현재 (Phase 1+2 완료)
│
├── [즉시 — 1~2주] 기술 부채 해소
│   ├── INT8 → FP16 양자화 재실험
│   ├── Step 18: 앙상블 + TTA로 F1 0.88+ 달성
│   └── README.md 정비 (시각화, 아키텍처 다이어그램)
│
├── [단기 — 2~4주] 시연 가능한 서비스 구축
│   ├── Step 14: Streamlit 대시보드 (최우선)
│   └── Step 15: FastAPI 서버 + Docker
│
├── [중기 — 1~2개월] 산업 연계 기능 추가
│   ├── Step 13: Mixed-type 멀티레이블 분류
│   ├── Step 16: SPC 기반 EWS
│   └── 공정 파라미터 실데이터 연계 구조 설계
│
└── [장기 — 2~3개월] 연구 수준 고도화
    ├── Step 17: 반지도 학습 (638K 비레이블 활용)
    └── Vision Transformer 기반 Attention Map 분석
```

---

## 8. 결론

현 프로젝트는 **MLOps 전 파이프라인 구축 + 소자 엔지니어링 분석**이라는 두 축에서 상당한 완성도를 갖추고 있다. 그러나 반도체 기업 면접에서 차별화되려면 다음 3가지가 핵심이다.

**① 실동작 데모** — Streamlit 대시보드는 코드보다 100배 강한 인상을 준다  
**② 산업 맥락 서술** — "이 기능이 Fab에서 어떤 문제를 해결하는가"를 코드마다 명시  
**③ 성능 목표 달성** — 앙상블/TTA로 F1 0.88+ 달성 후 "목표 달성" 박스를 채울 것

세메스를 타겟으로 한다면 FastAPI + SECS/GEM 통합 시나리오 작성이 가장 차별화 포인트가 될 것이고, SK하이닉스/삼성전자를 타겟으로 한다면 Mixed-type 분류 + SPC 기반 EWS가 실무 경험을 가장 잘 보여줄 수 있다.

---

*이 보고서는 SK하이닉스 뉴스룸, 세메스 채용 JD, 최신 학술 논문(2025 SemiWaferNet, ViT-VAE-GAN), 삼성전자 DS부문 AI 검사 직무 분석을 종합하여 작성되었습니다.*

---

### 참고 자료

- [SK하이닉스 Panoptes VM AI 솔루션 도입](https://news.skhynix.co.kr/panoptes_vm/)
- [SK하이닉스 HBM 생산라인 AI 적용 (2026)](https://www.newspim.com/news/view/20260625001090)
- [SK하이닉스 AI로 웨이퍼 결함 잡는다 — 서울경제](https://www.sedaily.com/article/20027683)
- [삼성전자 AI센터 신호 및 시스템설계 직무](https://www.samsung-dsrecruit.com/recruits/job_intro/ai_center/signSystem_design.php)
- [세메스 2025 하반기 신입 채용 JD](https://semes.careerlink.kr/product)
- [SemiWaferNet: Hybrid CNN-Transformer for Wafer Defect](https://doi.org/10.3390/electronics15071437)
- [Transformer-Based AI Framework for Wafer Inspection](https://journalcjast.com/index.php/CJAST/article/view/4633)
- [AI-Based Wafer Defect Inspection — RoboVision](https://robovision.ai/blog/ai-based-wafer-defect-inspection-an-accurracy-and-efficiency-boost)
- [Review of wafer defect detection — Springer Nature (2026)](https://link.springer.com/article/10.1007/s10845-026-02845-z)
