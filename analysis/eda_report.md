# WM-811K EDA 리포트

## 1. 데이터 현황

| 항목 | 수치 |
|------|------|
| 전체 샘플 수 | 811,457개 |
| 레이블 있는 샘플 | 172,950개 (21.3%) |
| 미레이블 (unknown) | 638,507개 (78.7%) |
| 클래스 수 | 9종 불량 + none |

## 2. 클래스 분포

| 클래스 | 샘플 수 | 비율 |
|--------|--------|------|
| none | 147,431 | 85.24% |
| Center | 4,294 | 2.48% |
| Donut | 555 | 0.32% |
| Edge-Loc | 5,189 | 3.00% |
| Edge-Ring | 9,680 | 5.60% |
| Loc | 3,593 | 2.08% |
| Near-full | 149 | 0.09% |
| Random | 866 | 0.50% |
| Scratch | 1,193 | 0.69% |

**Imbalance Ratio: 989.5x** (none 147,431 vs Near-full 149)

## 3. 불량 다이 비율 통계 (클래스별)

| 클래스 | 평균(%) | 표준편차 | Skewness | Kurtosis |
|--------|---------|---------|----------|----------|
| none | 10.193 | 4.978 | 0.678 | -0.164 |
| Center | 23.016 | 9.936 | -0.049 | 0.401 |
| Donut | 27.716 | 11.192 | 1.261 | 1.764 |
| Edge-Loc | 18.466 | 11.227 | 1.662 | 3.278 |
| Edge-Ring | 15.064 | 5.28 | 0.837 | 1.603 |
| Loc | 14.708 | 10.09 | 1.037 | 2.062 |
| Near-full | 87.691 | 8.354 | -0.106 | -1.2 |
| Random | 48.054 | 10.493 | -0.105 | -0.016 |
| Scratch | 10.21 | 6.222 | 1.125 | 1.949 |

## 4. 웨이퍼 맵 크기

- 고유 크기 종류: 346가지
- 높이 범위: 15 ~ 212 (평균 35.2)
- 너비 범위: 3 ~ 204 (평균 34.8)
- 가변 크기 → **64×64 고정 리사이징** 필수

## 5. 전처리 전략 결정

| 항목 | 결정 사항 |
|------|----------|
| 데이터 필터링 | unknown 제거 → 172,950개 사용 |
| 크기 통일 | 64×64 (cv2.resize, INTER_AREA) |
| 정규화 | 픽셀값 0~2 → 0.0~1.0 |
| 클래스 불균형 처리 | Weighted Random Sampler + Class Weight |
| 데이터 증강 | Albumentations: Rotate±20°, HFlip, VFlip, GaussNoise, Cutout |

## 6. 산출물 목록

- `analysis/class_distribution.png`
- `analysis/sample_gallery.png`
- `analysis/avg_defect_heatmap.png`
- `analysis/wafer_size_distribution.png`
- `analysis/defect_ratio_analysis.png`
- `analysis/eda_summary.csv`
- `analysis/eda_report.md` (이 파일)
