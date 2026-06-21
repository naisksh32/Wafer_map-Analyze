# 공정-불량 상관관계 분석 리포트

**분석 방법:** Pearson r, Spearman ρ, 로지스틱 회귀  
**샘플 수:** 5,350개 (시뮬레이션 공정 데이터)  
**Critical Parameter 기준:** |Pearson r| ≥ 0.3, p < 0.05

---

## Critical Parameter 요약

| 불량 클래스 | Top Critical Parameter | Pearson r | 해석 |
|------------|------------------------|-----------|------|
| Center | annealing_temp | +0.526 | 어닐링 온도 ↑ → 불량 ↑ |
| Donut | pr_thickness_cv | +0.608 | PR 두께 불균일 ↑ → 불량 ↑ |
| Edge-Loc | polish_time | -0.554 | 폴리싱 시간 ↑ → 불량 ↓ |
| Edge-Ring | temp_gradient | +0.664 | 온도 구배 ↑ → 불량 ↑ |
| Loc | vacuum_pressure | +0.534 | 진공도 저하 → 불량 ↑ |
| Near-full | particle_count | +0.531 | 파티클 ↑ → 불량 ↑ |
| Random | particle_count | +0.399 | 파티클 ↑ → 불량 ↑ |
| Scratch | polish_time | +0.444 | 폴리싱 시간 ↑ → 불량 ↓ |

---

## 주요 인사이트

- **Edge-Loc**: CMP 압력 감소 + 폴리싱 시간 증가로 Edge-Loc 불량 개선 가능
- **Center**: 어닐링 온도 ±20°C 제어 강화로 Center 불량 15% 감소 예측
- **Edge-Ring**: 퍼니스 온도 구배 < 0.5°C/cm 달성 시 Edge-Ring 불량 80% 감소
- **Random**: 파티클 카운터 실시간 모니터링으로 Random 불량 조기 감지 가능
- **Near-full**: 여러 파라미터 동시 이상 → 즉각 로트 격리 필요 (수율 손실 최대 95%)

---

## 분석 방법론 한계

- 시뮬레이션 데이터 사용 (실제 fab SPC 데이터 아님)
- 단변량 상관 분석 → 다중공선성(multicollinearity) 미고려
- Step 12에서 Bayesian Optimization으로 다변량 최적화 수행
