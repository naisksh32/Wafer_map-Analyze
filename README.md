# WM-811K 웨이퍼 불량 검출 — MLOps + 소자 엔지니어링 고도화

> **SK하이닉스 Device Engineering 직무 포트폴리오**
> 반도체 웨이퍼 불량 패턴 분류를 넘어, 물리적 원인 규명 → 공정 파라미터 최적화 → 수율 개선 우선순위 도출까지 연결한 의사결정 지원 시스템

---

## 핵심 성과

| 항목 | 수치 | 근거 파일 |
|------|------|----------|
| 분석 데이터 | WM-811K 중 레이블 **172,950개** (클래스 불균형 989.5×) | `analysis/data_summary.json` |
| 최고 분류 성능 | WaferCNN **macro F1 0.8458** / Accuracy 95.78% | `analysis/final_evaluation.json` |
| 사전학습 모델 최고 | ViT-Tiny **macro F1 0.8352** / Accuracy 95.54% | 〃 |
| Multi-output 모델 | **macro F1 0.8173** / Accuracy 95.08% (분류+심각도+신뢰도) | 〃 |
| ONNX 변환 | macro F1 **손실 0.0000** (0.7369 → 0.7369) | `analysis/deployment_summary.json` |
| 추론 속도 | CPU 17.07ms → ONNX **7.02ms** (2.4× · batch=32) | 〃 |
| 단일 웨이퍼 추론 | **0.60ms** (데스크톱 CPU 기준, RPi 추정 3~6ms) | 〃 |
| 공정 최적화 | 개선 우선순위 **Edge-Ring > Edge-Loc > Center** 도출 | `reports/roi_summary.csv` |

> 위 성능 수치는 전부 `scripts/evaluate_all.py` 한 번의 실행으로 재현됩니다.

---

## 성능 측정 기준 (먼저 읽어주세요)

포트폴리오 수치는 **측정 조건이 명시되지 않으면 의미가 없습니다.** 본 프로젝트의 모든 성능은 아래 단일 기준으로 산출했습니다.

| 항목 | 기준 |
|------|------|
| 평가 데이터 | **Test 분할 25,943개** — 증강 없음, WeightedRandomSampler 없음, 원본 불균형 분포 그대로 |
| 지표 산출 | 하나의 체크포인트에서 **Accuracy · macro F1 · weighted F1 를 동시 계산** |
| 표에 기재된 값 | 전부 **Test** 기준 (Validation 값 혼용 없음) |
| 재현 | `python scripts/evaluate_all.py` |

**테스트셋 클래스 분포**
`none 22,115 · Edge-Ring 1,452 · Edge-Loc 779 · Center 644 · Loc 539 · Scratch 179 · Random 130 · Donut 83 · Near-full 22`

> 이전 버전의 README는 학습 도중 기록된 Validation 지표와 재학습 이전의 Test 지표가 섞여 있어
> **macro F1 0.501 / Accuracy 10.65%** 처럼 동시에 성립할 수 없는 조합이 게시되어 있었습니다.
> 전 모델을 동일 체크포인트·동일 테스트셋에서 재측정하여 교체했습니다.

---

## 모델 성능 비교

Test 25,943개 · 동일 조건 측정 (`analysis/final_evaluation.json`)

| 모델 | 파라미터 | Accuracy | **macro F1** | weighted F1 | 비고 |
|------|---------|----------|-------------|-------------|------|
| **WaferCNN** | 1.21M | **95.78%** | **0.8458** | 0.9609 | 커스텀 4-Conv CNN 베이스라인 |
| ViT-Tiny | 5.39M | 95.54% | 0.8352 | 0.9587 | 2-Phase 파인튜닝 |
| AdvancedDefectPredictor | 1.13M | 95.08% | 0.8173 | 0.9549 | Multi-output (분류+심각도+신뢰도) |
| EfficientNet-B0 | 4.02M | 93.96% | 0.7944 | 0.9464 | 2-Phase 파인튜닝 |
| MobileNetV3-Small | 1.53M | 90.45% | 0.7369 | 0.9197 | **엣지 배포 채택 모델** |
| WaferCNN + Optuna HPO | 1.21M | 43.21% | 0.5987 | 0.5329 | HPO 실패 사례 (아래 참조) |

**목표(macro F1 ≥ 0.80) 달성: WaferCNN · ViT-Tiny · AdvancedDefectPredictor**

### 관찰 1 — 1.2M 파라미터 커스텀 CNN이 5.4M ViT를 이겼다

ImageNet 사전학습 백본(ViT-Tiny 5.39M, EfficientNet-B0 4.02M)이 도메인 특화 커스텀 CNN(1.21M)을 넘지 못했습니다. 웨이퍼 맵은 자연 이미지와 통계적 특성이 근본적으로 다르기 때문입니다.

- 입력이 **3-값 이산 데이터**(0=빈 영역 / 1=정상 다이 / 2=불량 다이) — 자연 이미지의 텍스처·색상 사전지식이 전이되지 않음
- 판별 정보가 **국소 텍스처가 아니라 전역 공간 배치**(중심/링/선형)에 있음
- 64×64 해상도에서 ViT의 patch 16 → 4×4=16 토큰으로 공간 해상도가 과도하게 손실

→ **전이학습이 항상 우월하지 않으며, 도메인 특성에 맞춘 아키텍처 설계가 유효하다**는 것을 실측으로 보인 결과입니다.

### 관찰 2 — Accuracy와 macro F1이 함께 움직이지 않는 이유

베이스라인 WaferCNN의 클래스별 지표 (`analysis/baseline_classification_report.txt`):

| 클래스 | Support | Precision | Recall | F1 |
|--------|--------:|----------:|-------:|---:|
| none | 22,115 | 0.9977 | 0.9604 | 0.9787 |
| Edge-Ring | 1,452 | 0.9337 | 0.9897 | 0.9609 |
| Center | 644 | 0.8142 | 0.9596 | 0.8810 |
| Donut | 83 | 0.9059 | 0.9277 | 0.9167 |
| Near-full | 22 | 0.8462 | 1.0000 | 0.9167 |
| Random | 130 | 0.6879 | 0.9154 | 0.7855 |
| Edge-Loc | 779 | 0.6654 | 0.9294 | 0.7756 |
| Loc | 539 | 0.6890 | 0.8219 | 0.7496 |
| **Scratch** | 179 | **0.4927** | 0.9441 | **0.6475** |
| **macro avg** | | **0.7814** | **0.9387** | **0.8458** |

WeightedRandomSampler로 소수 클래스를 과표집한 결과, **macro Recall 0.9387 / macro Precision 0.7814** 로 재현율에 크게 치우쳐 있습니다.

**이것은 반도체 검사 도메인에서 의도한 방향입니다.** 불량 웨이퍼를 놓치는 비용(미검출 → 후공정 낭비 → 필드 불량)이 정상 웨이퍼를 재검사하는 비용보다 훨씬 크기 때문에, Precision을 희생하고 Recall을 확보하는 것이 옳은 트레이드오프입니다.

다만 **Scratch의 Precision 0.4927** 은 실무 도입 시 병목입니다 — Scratch로 분류된 웨이퍼의 절반이 오탐이라 재검사 부하가 2배가 됩니다. 개선 방향은 [알려진 한계](#알려진-한계와-다음-단계)에 정리했습니다.

### 관찰 3 — Optuna HPO는 개선에 실패했다 (기록 목적)

| | Accuracy | macro F1 |
|---|---:|---:|
| WaferCNN 기본 설정 | 95.78% | 0.8458 |
| WaferCNN + Optuna 최적 파라미터 | 43.21% | 0.5987 |

**성능이 하락했습니다.** 포장하지 않고 원인과 함께 기록합니다.

- **탐색 실패**: 20 trials 중 13개가 Median Pruner에 조기 종료되어 **유효 탐색이 7회**에 그침
- **수렴 미달**: 선택된 `lr=7.3e-5`가 기본값(3e-4)의 1/4 수준이라 동일 epoch 예산 내에서 수렴하지 못함
- **결과**: `none` 클래스 Recall이 0.9604 → **0.3462** 로 붕괴, Scratch Precision은 **0.0165**(오탐 폭증). Accuracy가 43%까지 떨어지면서도 macro F1이 0.60을 유지하는 것은, macro 평균이 9개 클래스를 동등 가중하여 소수 클래스의 높은 Recall이 다수 클래스 붕괴를 가려주기 때문입니다.

→ **"Accuracy 43% / macro F1 0.60"은 불균형 데이터에서 실제로 성립하는 조합**이며, 두 지표를 반드시 함께 봐야 하는 이유를 보여주는 사례입니다.
→ 후속 조치: Pruner 완화(`n_warmup_steps` 증가) · 탐색 범위를 `lr ∈ [1e-4, 1e-3]`로 조정 · trial당 epoch 예산 확대.

---

## 엣지 배포 — 모델 선택과 양자화 기각 근거

`analysis/deployment_summary.json` · 재현: `python scripts/export_onnx.py`

### 왜 최고 성능 모델이 아닌 MobileNetV3를 배포했는가

최고 성능은 WaferCNN(macro F1 0.8458)이지만, **엣지 배포 대상은 MobileNetV3-Small(0.7369)** 을 선택했습니다.

**macro F1 0.109 손실을 감수한 의도적 트레이드오프입니다.** 근거는 depthwise separable convolution 기반 구조가 ARM CPU에서 연산 최적화가 검증되어 있고, onnxruntime·TFLite 등 엣지 런타임의 연산자 지원이 가장 성숙하다는 점입니다. 인라인 검사 장비의 실시간 요구를 만족하면서 유지보수 부담이 가장 낮은 선택입니다.

> 정확도가 최우선인 오프라인 배치 분석에는 WaferCNN을, 인라인 실시간 검사에는 MobileNetV3를 쓰는 **이원 배포 전략**이 적절합니다.

### 검증 결과

| 모델 | 크기 | Accuracy | macro F1 | 추론 (batch=32) | 판정 |
|------|-----:|---------:|---------:|----------------:|:----:|
| PyTorch (CPU) | 5.95MB | 90.45% | 0.7369 | 17.07ms | 기준 |
| PyTorch (GPU) | — | 90.45% | 0.7369 | 6.41ms | — |
| **ONNX FP32** | 5.83MB | **90.45%** | **0.7369** | **7.02ms** | ✅ **채택** |
| ONNX INT8 (동적 양자화) | 1.62MB | 2.57% | 0.0279 | 88.06ms | ❌ **기각** |

**ONNX FP32는 macro F1 손실이 정확히 0.0000** — 변환이 성능을 완전히 보존했음을 테스트셋 전체(25,943개)로 확인했습니다. CPU 추론은 2.4배 빨라졌습니다.

**INT8 양자화는 기각했습니다.** 모델 크기가 72.2% 줄어드는 것은 사실이나:

- macro F1 **0.7369 → 0.0279** (−0.709) — 사실상 무작위 예측 수준
- 추론 속도도 7.02ms → **88.06ms 로 12배 느려짐** (크기 감소가 속도 이득으로 이어지지 않음)
- 원인 추정: 단일 채널 grayscale 입력은 활성값 분포의 동적 범위가 좁아, per-tensor 양자화 스케일이 소수 클래스를 구분하는 미세한 활성 차이를 뭉갬

> **"모델 크기 72% 감소"를 성과로 제시하지 않는 이유입니다.** 정확도 검증 없는 경량화는 성과가 아니라 결함입니다. 개선하려면 per-channel 양자화 또는 QAT(Quantization-Aware Training)가 필요합니다.

### 단일 웨이퍼 추론 (batch=1)

| | 평균 | p95 |
|---|---:|---:|
| ONNX FP32 | **0.60ms** | 0.74ms |
| ONNX INT8 | 3.42ms | 3.65ms |

> 데스크톱 CPU 측정값입니다. 실제 Raspberry Pi 4는 5~10배 느리므로 **실기 추정 3~6ms** 수준이며, **실측이 아닌 추정치**입니다.

---

## Phase 2 — 소자 엔지니어링 고도화

> ### ⚠️ 공정 파라미터 데이터에 관한 고지
>
> **WM-811K 데이터셋에는 공정 파라미터가 존재하지 않습니다.** 원본 컬럼은
> `waferMap · dieSize · lotName · waferIndex · trianTestLabel · failureType` 뿐입니다.
>
> 따라서 Phase 2의 공정 파라미터(`temp_gradient`, `annealing_temp`, `vacuum_pressure` 등 9종)는
> **실제 팹 SPC 데이터가 비공개인 환경을 가정하고, 반도체 공정 도메인 지식에 근거해 모사·생성한 시뮬레이션 데이터**입니다.
> (생성 로직: `10_process_correlation.ipynb` 의 `PARAM_BASELINE` / `CLASS_BIAS` / `generate_process_samples`)
>
> **이 섹션의 상관계수와 금액은 "실측 발견"이 아니라, 데이터가 연결되었을 때 동작하도록 설계·검증한
> 분석 파이프라인의 출력 예시입니다.** 실제 팹 SPC 데이터를 `data/process_parameters.csv` 스키마에
> 맞춰 교체하면 코드 수정 없이 동일한 분석이 실행됩니다.
>
> 상관계수의 크기 자체는 생성 시 부여한 클래스별 바이어스에 의해 결정되므로, **파이프라인의 검증 대상은
> 상관계수 값이 아니라 "바이어스가 존재할 때 이를 통계적으로 검출해내는 능력"** 입니다.

### 불량 메커니즘 분석 (WM-811K 실측 기반)

아래 공간 통계량은 **실제 WM-811K 웨이퍼 맵에서 계산한 실측값**입니다.

| 불량 | 원인 공정 | 핵심 지표 (실측) | 심각도 |
|------|----------|-----------------|--------|
| Edge-Ring | Thermal Oxidation / CVD | ring_ratio = **0.780** | High |
| Edge-Loc | CMP / PVD Deposition | ring_ratio = 0.621 | High |
| Donut | Lithography / Etch | center_ratio = **0.445** | High |
| Center | Wafer Preparation / Annealing | center_ratio = **0.372** | Critical |
| Near-full | 복합 오염 / 결정 결함 | defect_density = **0.877** | Critical |
| Scratch | Wafer Handling / CMP | 직선형 공간 분포 | Medium |

### 공정 파라미터 상관분석 파이프라인 *(시뮬레이션 데이터)*

Pearson / Spearman 상관 + 로지스틱 회귀로 클래스별 Critical Parameter를 식별합니다.

| 불량 클래스 | 식별된 Critical Parameter | Pearson r |
|-----------|------------------------|----------|
| Edge-Ring | `temp_gradient` | +0.664 |
| Donut | `pr_thickness_cv` | +0.608 |
| Loc | `vacuum_pressure` | +0.534 |
| Center | `annealing_temp` | +0.527 |
| Edge-Loc | `polish_time` | −0.554 |

> 위 계수는 시뮬레이션 데이터에서 산출된 값이며, 실제 팹 데이터의 상관 강도를 주장하지 않습니다.

### XAI — Integrated Gradients

```
SHAP DeepExplainer   → MobileNetV3 Hardswish 미지원 (AssertionError)
SHAP GradientExplainer → CUDA + inplace op 충돌 (RuntimeError)
Integrated Gradients (순수 PyTorch) → 정상 동작 ✅

IG(x) = (x − baseline) × ∫₀¹ ∇f(baseline + α(x − baseline)) dα
baseline: none 클래스 20개 평균 | steps: 30
```

라이브러리 제약을 우회해 직접 구현한 부분으로, 픽셀 단위 기여도가 각 불량의 물리적 발생 위치(중심/링/선형)와 일치함을 확인했습니다.

### 공정 최적화 — 개선 우선순위 도출 *(시뮬레이션 데이터 기반)*

`scipy.optimize.differential_evolution`으로 불량률을 최소화하는 파라미터 조합을 탐색합니다.

- 목적함수: `0.7 × 불량률 + 0.2 × 비용증가 + 0.1 × 제약위반` *(가중치는 설계 선택값)*
- 생산 가정: 월 50,000장 × $500/장

**본 분석의 산출물은 금액 자체가 아니라 개선 우선순위입니다.**

| 순위 | 불량 | 투자 회수 | ROI | 판단 |
|:---:|------|---------:|----:|------|
| **1** | Edge-Ring | 0.1개월 | 10,975% | 최우선 — 개선 비용이 매우 낮음 (`temp_gradient` 조정만 필요) |
| **2** | Edge-Loc | 0.6개월 | 1,966% | 즉시 착수 |
| **3** | Center | 3.8개월 | 216% | 중기 과제 |
| **4** | Donut | 5.0개월 | 139% | 중기 과제 |
| — | Near-full | 14.8개월 | **−18.8%** | **보류 — 개선 비용이 기대 이익을 초과** |

**Near-full의 ROI가 음수라는 점이 이 분석의 핵심 결론입니다.** 수율 영향도(0.95)가 가장 큰 불량임에도 발생 빈도가 0.086%에 불과해, 개선 투자가 회수되지 않습니다. **"모든 불량을 잡는 것이 아니라, 어떤 불량을 먼저 잡을지 결정하는 것"** 이 공정 엔지니어링의 실제 의사결정입니다.

> ROI 수치는 4단계 가정(시뮬레이션 파라미터 → 모사 상관관계 → 설계된 목적함수 가중치 → 생산량·단가 가정)이 누적된 결과입니다. **절대 금액이 아니라 상대 순위로만 해석해야 합니다.**

---

## 데이터셋

**WM-811K (Wafer Map 811K)** — [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

| 항목 | 값 |
|------|-----|
| 전체 샘플 | 811,457개 |
| 레이블 샘플 (분석 대상) | 172,950개 |
| 미레이블 | 638,507개 (`failureType == 'unknown'` → 제외) |
| 클래스 수 | 9종 (none 포함) |
| 클래스 불균형 | **989.5×** (none vs Near-full) |
| 입력 형식 | 2D 웨이퍼 맵 (0=빈 영역 / 1=정상 다이 / 2=불량 다이), 가변 크기 |
| 전처리 | 64×64 리사이징(INTER_NEAREST) · ÷2 정규화 · Train 70 / Val 15 / Test 15 (Stratified) |
| 불균형 처리 | WeightedRandomSampler **단독** (CrossEntropyLoss weight 병용 시 이중 보정으로 편향 악화) |

### 클래스 분포 (레이블 172,950개)

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

## 알려진 한계와 다음 단계

포트폴리오의 신뢰도는 한계를 숨기지 않는 데서 나온다고 판단하여, 현재 확인된 제약을 명시합니다.

| # | 한계 | 영향 | 개선 방향 |
|:-:|------|------|----------|
| 1 | **Lot 단위 데이터 누수 가능성** — 현재 분할은 `StratifiedShuffleSplit`(웨이퍼 단위). WM-811K는 한 lot의 웨이퍼들이 유사한 불량 패턴을 공유하므로, 동일 lot이 train/test에 동시 포함될 수 있음 | 보고된 성능이 실제 일반화 성능보다 **낙관적일 수 있음** | `GroupShuffleSplit(groups=lotName)` 재분할 후 성능 재측정 · 낙폭을 실제 일반화 성능으로 보고 |
| 2 | **데이터셋 공식 분할 미사용** — 원본의 `trianTestLabel`(Train 54,355 / Test 118,595) 대신 자체 분할 사용 | 선행 논문과 **직접 비교 불가** | 공식 분할 기준 성능을 병기 |
| 3 | **Scratch Precision 0.4927** | 오탐 재검사 부하 2배 | 클래스별 임계값 조정 · Scratch 전용 증강 분리 (현재 `CoarseDropout` 8×8이 선형 패턴을 훼손할 수 있음) |
| 4 | **픽셀값을 연속값으로 처리** — `0/1/2 → 0/0.5/1` 정규화 | 빈 영역·정상 다이·불량 다이는 **순서형이 아닌 범주형** | 3채널 one-hot 인코딩으로 전환 시 Edge 계열 성능 개선 여지 |
| 5 | **Optuna HPO 개선 실패** | 위 [관찰 3](#관찰-3--optuna-hpo는-개선에-실패했다-기록-목적) | Pruner 완화 · 탐색 범위 재설정 후 재실행 |
| 6 | **공정 파라미터가 시뮬레이션 데이터** | Phase 2 상관계수·ROI는 실측 주장 아님 | 실 SPC 데이터 연결 시 코드 수정 없이 재실행 가능 |
| 7 | **서빙 API 부재** | 배포는 ONNX 파일 수준까지 | FastAPI `/predict` + Docker 이미지 |
| 8 | **테스트 코드·CI 부재** | 회귀 검증 수단 없음 | `src/` 모듈 단위 pytest + GitHub Actions |
| 9 | **드리프트 모니터링 부재** | Airflow가 `@weekly` 전체 재학습만 수행 | PSI/KS 기반 드리프트 감지 → 조건부 재학습 트리거 |

---

## 전체 파이프라인

```
┌─────────────────────────── Phase 1 — 기본 MLOps ────────────────────────────┐
│  Step 1        Step 2        Step 3         Step 4         Step 5           │
│  데이터 준비 →  EDA      →   전처리/증강 →  베이스라인  →   파인튜닝          │
│  172,950개     불균형분석    64×64 리사이즈  WaferCNN       MobileNetV3       │
│                             Albumentations  F1 0.8458      EfficientNet-B0  │
│                                                            ViT-Tiny F1 .8352│
│                                                                             │
│  Step 6              Step 7          Step 8                                 │
│  MLFlow/Optuna   →   Airflow DAG →   ONNX 배포                              │
│  20 trials           Docker 8Task    FP32 채택 · F1 손실 0.0000             │
│  (개선 실패 기록)                     7.02ms CPU · INT8 기각                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────── Phase 2 — 소자 엔지니어링 (시뮬레이션 데이터) ──────────────┐
│  Step 9           Step 10          Step 11           Step 12                │
│  불량 메커니즘 →  공정 상관관계 →  Multi-output  →   공정 최적화             │
│  9종 물리 원인    Pearson/Spearman  모델 + XAI        Differential Evolution │
│  (WM-811K 실측)   Critical Param.   Integrated Grad.  개선 우선순위 도출     │
└─────────────────────────────────────────────────────────────────────────────┘
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
| 파이프라인 | Apache Airflow 2.9 · Docker Compose · PySpark |
| 배포 | ONNX · onnxruntime |
| XAI | Integrated Gradients (PyTorch 직접 구현) · Grad-CAM |
| 통계 | scipy.stats (Pearson / Spearman) · statsmodels |
| 최적화 | scipy.optimize.differential_evolution |
| 시각화 | matplotlib · seaborn · plotly · Vue 3 대시보드 |

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
├── 08_onnx_deployment.ipynb        # ONNX 변환 & 엣지 배포
├── 09_defect_mechanism.ipynb       # 불량 메커니즘 분석
├── 10_process_correlation.ipynb    # 공정-불량 상관관계 (시뮬레이션 데이터)
├── 11_advanced_model.ipynb         # Multi-output 모델 + XAI
├── 12_process_optimization.ipynb   # 공정 최적화 + 우선순위 도출
├── 16_spark_pipeline.ipynb         # PySpark 대용량 처리 파이프라인
│
├── scripts/
│   ├── evaluate_all.py             # ★ 전 모델 통합 재평가 (README 성능표 근거)
│   ├── export_onnx.py              # ★ ONNX 변환·정확도 검증·벤치마크
│   ├── retrain_baseline.py         # 베이스라인 재학습
│   ├── retrain_finetune.py         # 파인튜닝 3종 재학습
│   └── retrain_advanced.py         # Multi-output 재학습
│
├── src/                            # 재사용 Python 모듈
│   ├── data_loader.py              # Dataset · DataLoader · 증강
│   ├── model.py                    # WaferCNN
│   ├── advanced_defect_predictor.py# Multi-output 모델
│   ├── defect_analyzer.py          # 불량 공간 통계
│   ├── model_explainer.py          # Integrated Gradients · Grad-CAM
│   ├── process_correlation_analyzer.py
│   └── process_optimizer.py
│
├── spark_pipeline/                 # Bronze → Silver → Gold 레이크 파이프라인
├── dashboard/                      # Vue 3 공정 파라미터 What-If 대시보드
│
├── analysis/                       # 성능·EDA 산출물
│   ├── final_evaluation.json       # ★ 전 모델 통합 성능 (README 근거)
│   ├── per_class_metrics.csv       # ★ 모델 × 클래스별 P/R/F1
│   ├── final_confusion_matrices.png# ★ 모델별 혼동행렬
│   ├── baseline_classification_report.txt
│   └── deployment_summary.json     # ★ ONNX 검증 결과
│
├── configs/                        # augmentation_config.yaml · defect_metadata.json
├── reports/                        # 최적화 리포트 · ROI 분석
├── docs/                           # 불량 메커니즘 상세 문서
├── data/
│   ├── raw/LSWMD.pkl               # WM-811K 원본 (~2GB)
│   ├── processed/                  # 전처리 완료 (NPY, PKL)
│   └── process_parameters.csv      # 시뮬레이션 공정 파라미터
│
├── Report.md                       # 전체 결과 보고서
├── PROJECT_PLAN.md                 # 단계별 계획 및 진행 현황
└── requirements.txt
```

---

## 빠른 시작

### 환경 설정

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

### 데이터 준비

```bash
# Kaggle API 인증 후 다운로드 (01_data_download.ipynb 참조)
# 또는 data/raw/LSWMD.pkl 직접 배치
```

### 성능 재현 (학습 없이 체크포인트로 검증)

```bash
python scripts/evaluate_all.py    # → analysis/final_evaluation.json
python scripts/export_onnx.py     # → analysis/deployment_summary.json
```

### 노트북 실행 순서

```
01 → 02 → 03 → 04 → 05 → 06 → 07(선택) → 08 → 09 → 10 → 11 → 12 → 16(선택)
```

### Airflow / MLFlow / 대시보드

```bash
cd 07_airflow_pipeline && docker compose up -d      # http://localhost:8080
mlflow ui --backend-store-uri "sqlite:///mlruns/mlflow.db"   # http://localhost:5000
cd dashboard && npm install && npm run dev          # http://localhost:5173
```

> **대시보드 주의:** 현재 프론트엔드는 공정 파라미터 이탈도를 도메인 규칙(z-score 기반)으로 계산하는
> **What-If 시뮬레이터**이며, 학습된 모델을 직접 호출하지 않습니다.
> 실제 모델 추론 연동(onnxruntime-web)은 예정 과제입니다.

---

## 주요 결과 시각화

| 분석 | 파일 |
|------|------|
| **모델별 혼동행렬 (최종)** | `analysis/final_confusion_matrices.png` |
| 클래스 분포 | `analysis/class_distribution.png` |
| 클래스별 평균 불량 히트맵 | `analysis/avg_defect_heatmap.png` |
| 파인튜닝 모델 비교 | `analysis/finetuning_model_comparison.png` |
| Optuna HPO 분석 | `analysis/hpo_analysis.png` |
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
