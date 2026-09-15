# 다음 단계 계획 — seed 확정 · 엣지 배포 재검증 · 결과 검증 (2026-09-15)

## 0. 출발점

1·2차 실험(`docs/MODEL_PERFORMANCE.md`)으로 다음이 확정됐다.

- 기존 README 수치는 재현되지만, "커스텀 CNN > MobileNet" 은 레시피(백본 lr 1e-5) + 입력 해상도(64px → 2×2 feature map) 차이가 만든 결과.
- 동일 레시피에서 해상도가 유일하게 큰 변인. MobileNetV3-S 는 160px 에서 포화(0.888), EfficientNet-B0 224px 가 전체 최고(0.907).
- 보간·채널·ImageNet 정규화는 잡음 범위. WaferCNN 은 업샘플에 반응하지 않음(구조 종속 확인).

남은 약점 세 가지가 이 계획의 대상이다.

| # | 약점 | 영향 |
|---|---|---|
| 1 | 상위 구성이 **seed 42 단일** | "EffNet 224 > MV3 160" 같은 2차 결론의 통계적 근거 부족 |
| 2 | README 엣지 배포 표가 **이전 체크포인트(0.7369)** 기준 | 배포 모델과 최신 결과의 불일치. "정확도-속도 트레이드오프" 서술이 낡음 |
| 3 | 결과 파일·문서 수치의 **독립 검증 부재** | run JSON 은 학습 스크립트가 쓴 값 그대로. 체크포인트 재채점·문서 대조가 없음 |

## 1. Stage E — 상위 구성 seed 확정

**목적:** 2차 결론을 mean ± std 로 승격.

| 구성 | 현재 | 추가 seed | 예상 시간 (GPU 정상 / 20W 제한) |
|---|---|---|---|
| `mv3_pre160` MobileNetV3-S 160px | 0.8883 (s42) | 43, 44 | 40분 × 2 / 1.5시간 × 2 |
| `mv3_pre128_norm` 1ch + ImageNet norm | 0.8830 (s42) | 43, 44 | 40분 × 2 / 1.5시간 × 2 |
| `effb0_pre224` EfficientNet-B0 224px | 0.9072 (s42) | 43, 44 | 2시간 × 2 / 7.5시간 × 2 |

실행 순서는 저비용 → 고비용. EfficientNet 224px 는 20W 제한에서 run 당 7.5시간이 걸리므로 Samsung Settings 성능 모드 "고성능" 이 전제다.

```bash
python scripts/fair_compare.py --models mv3_pre160 mv3_pre128_norm --seeds 43 44 --num_workers 4
python scripts/fair_compare.py --models effb0_pre224 --seeds 43 44 --num_workers 4
python scripts/fair_compare.py --summarize
```

**판정 기준**

- 3 seeds std 가 0.01 이하면 구성 간 차이를 "확정" 으로 표기. 두 구성의 평균 차이가 두 std 합보다 작으면 "구분 안 됨".
- `mv3_pre128_norm` 이 `mv3_pre128`(0.8756 ± 0.0026) 을 3 seeds 평균으로도 앞서면 ImageNet 정규화를 기본 전처리로 채택.

## 2. Stage F — 엣지 배포 재검증 (MobileNetV3-S 160px)

**목적:** README 배포 절을 최신 모델로 갱신하고, "정확도를 포기하는 트레이드오프" 서술을 사실에 맞게 교체.

**설계**

- 대상: `checkpoints/fair/mv3_pre160_seed42.pth` (Stage E 후 3 seeds 중 median 으로 교체 가능).
- ONNX 그래프에 **업샘플을 포함** → 배포 입력은 기존과 동일한 (B,1,64,64). 호출측 변경 없음.
- 비교군: 기존 배포 모델(`MobileNetV3_34_0.7431.pth`, 64px) 을 같은 절차로 변환해 속도·정확도 상대값을 같은 머신에서 측정.
- INT8 dynamic quantization 은 정확도 검증 후 채택/기각. 이전에 F1 0.03 으로 붕괴한 이력이 있음.
- 벤치마크는 **GPU 학습이 돌지 않는 시점**에 수행(DataLoader 워커의 CPU 점유가 지연 측정을 오염).

```bash
python scripts/export_onnx_v2.py     # → checkpoints/deploy/*.onnx, analysis/deployment_summary_v2.json
```

**판정 기준**

- FP32 ONNX 와 PyTorch 의 test 예측 일치율 ≥ 99.9%, macro F1 차이 ≤ 0.001.
- INT8: macro F1 손실 ≤ 0.01 이면 채택, 아니면 기각(이유 기록).
- batch=1 CPU 지연을 64px 모델 대비 배수로 기록. 160px 는 픽셀 6.25배이므로 3~5배 느려질 것으로 예상하며, 그래도 수 ms 대면 엣지 요건 충족.

## 3. Stage G — 검증 (필수)

**목적:** 학습 스크립트가 남긴 수치를 독립 경로로 다시 계산해 문서와 대조.

`scripts/verify_results.py` 가 다음을 수행하고 `analysis/verification_report.json` 을 쓴다.

| 검증 | 방법 | 통과 기준 |
|---|---|---|
| G1 체크포인트 재채점 | `checkpoints/fair/*.pth` 전부 로드 → test 25,943개 재추론 → run JSON 의 accuracy·macro F1·confusion matrix 와 비교 | F1 차이 ≤ 1e-4, confusion matrix 완전 일치 |
| G2 기존 레시피 체크포인트 재채점 | `final_evaluation.json` 5개 체크포인트 동일 절차 | 〃 |
| G3 집계 일관성 | `runs.csv` 에서 mean/std 재계산 → `summary.json` 대조 | 차이 ≤ 1e-4 |
| G4 문서 수치 대조 | README C-1·C-2 표와 `MODEL_PERFORMANCE.md` 표의 F1 값을 파싱 → 원본 JSON 대조 | 표기된 모든 값이 원본과 소수 4자리 일치 |
| G5 ONNX 동등성 | ONNX FP32 예측 vs PyTorch 예측 argmax 일치율 | ≥ 99.9% |
| G6 테스트셋 무결성 | `split_indices.pkl` test 분포가 README 기재 분포(none 22,115 … Near-full 22)와 일치, train/test 인덱스 교집합 0 | 완전 일치 |
| G7 그림 재생성 | `plot_results.py` 재실행 → 6개 PNG 존재·크기 > 0 | 6/6 |

실패 항목이 있으면 원인을 보고서에 적고, 문서 쪽 오류면 문서를 고친 뒤 재검증한다.

## 4. 산출물과 문서 반영

- `analysis/fair_compare/summary.*` · `docs/MODEL_PERFORMANCE.md` · `analysis/figures/*` 재생성
- `analysis/deployment_summary_v2.json` → README "엣지 배포" 절 교체
- `analysis/verification_report.json` → `docs/MODEL_PERFORMANCE.md` 5절 "검증" 추가
- `checkpoints/MANIFEST.json` 갱신

## 5. 하지 않는 것

- Optuna HPO: 최고 구성이 3 seeds 로 확정된 뒤 별도 단계로 진행(계획서 `docs/plans/` 후속).
- 첫 conv stride 1 · 224px 이상 해상도: 160px 포화가 확인되어 우선순위 낮음.
- Report.md 갱신: README·MODEL_PERFORMANCE 가 정본. Report.md 는 폐기 여부를 사용자와 결정.
