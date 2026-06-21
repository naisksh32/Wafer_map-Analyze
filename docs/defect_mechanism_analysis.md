# WM-811K 웨이퍼 불량 메커니즘 분석 리포트

**생성일:** (자동 생성)  
**데이터:** WM-811K (172,950개 레이블 샘플)  
**분석 대상:** 9가지 불량 패턴의 물리적 원인 규명

---

## 클래스별 심각도 요약

| 클래스 | 한국어명 | 심각도 | 수율 영향 | 주요 공정 | 예상 ROI |
|--------|--------|--------|----------|----------|--------|
| none | 정상 | ⚪ None | 0% | N/A | N/A |
| Center | 중심 불량 | 🔴 Critical | 15% | Wafer Preparation / Thermal An | 800% |
| Donut | 도넛형 불량 | 🟠 High | 10% | Lithography / Etch | 650% |
| Edge-Loc | 국소 에지 불량 | 🟠 High | 8% | CMP (Chemical Mechanical Plana | 1200% |
| Edge-Ring | 에지 링 불량 | 🟠 High | 9% | Thermal Oxidation / CVD / Clea | 950% |
| Loc | 국소 클러스터 불량 | 🟡 Medium | 5% | Deposition / Ion Implantation  | 500% |
| Near-full | 웨이퍼 전면 불량 | 🔴 Critical | 95% | Multiple (Contamination / Crys | 3000% |
| Random | 랜덤 분산 불량 | 🟡 Medium | 4% | Cleanroom Environment / Wafer  | 400% |
| Scratch | 스크래치 불량 | 🟡 Medium | 3% | Wafer Handling / CMP | 300% |

---

## 상세 메커니즘 분석

### 🔴 Center (중심 불량)

**물리적 원인:** 웨이퍼 중심부 결정 결함 또는 열처리 불균일. 산소 석출물(SiO2)이 중심에 집중 생성되어 게이트 산화막 품질 저하를 유발

**주요 공정 단계:** Wafer Preparation / Thermal Annealing
**관련 장비:** Czochralski Puller, Annealing Furnace, RTP Chamber
**심각도:** Critical (점수: 5/5)
**수율 영향도:** 15%
**예상 ROI:** 800%

**핵심 파라미터:**
- `Annealing_Temperature` (°C): 정상범위 1050-1150 — Critical
- `Annealing_Time` (h): 정상범위 2-8 — High
- `Cooling_Rate` (°C/min): 정상범위 5-20 — High
- `Oxygen_Concentration` (ppma): 정상범위 10-18 — Medium

**공간 분포 증거 (실측):
- 중심 집중도: 0.372
- 에지 집중도: 0.000
- 불량 밀도: 0.2330

**처방:** 냉각 속도 제어 최적화, 산소 농도 타겟 재설정, RTP 온도 균일도 개선

---

### 🟠 Donut (도넛형 불량)

**물리적 원인:** 링 형태 식각 불균일 또는 포토레지스트 도포 두께 변동. 스핀 코팅 시 중심/에지 경계에서 두께 전이구간 발생

**주요 공정 단계:** Lithography / Etch
**관련 장비:** Spin Coater, Stepper/Scanner, Dry Etcher (ICP/CCP)
**심각도:** High (점수: 4/5)
**수율 영향도:** 10%
**예상 ROI:** 650%

**핵심 파라미터:**
- `PR_Thickness_Uniformity` (%): 정상범위 <2 — Critical
- `Etch_Depth_Uniformity` (%): 정상범위 <3 — Critical
- `Chemical_Concentration` (wt%): 정상범위 ±0.5 — High
- `Spin_Speed` (rpm): 정상범위 2000-4000 — Medium

**공간 분포 증거 (실측):
- 중심 집중도: 0.445
- 에지 집중도: 0.000
- 불량 밀도: 0.2760

**처방:** 스핀 코팅 속도 프로파일 최적화, 식각 가스 유량 균일도 개선, Edge Bead Removal 적용

---

### 🟠 Edge-Loc (국소 에지 불량)

**물리적 원인:** CMP 공정 중 웨이퍼 에지 국소 영역 과도 연마 또는 PVD 증착 시 에지 두께 불균일. 리테이닝 링 압력 불균일로 에지 특정 방위각 구간에만 불량 집중

**주요 공정 단계:** CMP (Chemical Mechanical Planarization) / PVD Deposition
**관련 장비:** CMP Polisher, PVD Sputter, Wafer Chuck
**심각도:** High (점수: 4/5)
**수율 영향도:** 8%
**예상 ROI:** 1200%

**핵심 파라미터:**
- `CMP_Pressure` (psi): 정상범위 80-120 — Critical
- `Polish_Time` (s): 정상범위 20-60 — High
- `Slurry_pH` (-): 정상범위 4-8 — High
- `Retaining_Ring_Pressure` (psi): 정상범위 ±5 — Critical

**공간 분포 증거 (실측):
- 중심 집중도: 0.151
- 에지 집중도: 0.000
- 불량 밀도: 0.1791

**처방:** 리테이닝 링 압력 맵 보정, 슬러리 pH 실시간 모니터링, 에지 프로파일 측정 강화

---

### 🟠 Edge-Ring (에지 링 불량)

**물리적 원인:** 웨이퍼 전체 에지 링 형태의 열처리 불균일. 퍼니스 내 가스 흐름의 에지-센터 온도 구배가 에지 산화막 성장 속도 차이를 유발

**주요 공정 단계:** Thermal Oxidation / CVD / Cleaning
**관련 장비:** Diffusion Furnace, LPCVD Tube, Cleaning Track
**심각도:** High (점수: 4/5)
**수율 영향도:** 9%
**예상 ROI:** 950%

**핵심 파라미터:**
- `Furnace_Temperature_Gradient` (°C): 정상범위 <2 — Critical
- `Gas_Flow_Uniformity` (%): 정상범위 <1 — Critical
- `Process_Time` (min): 정상범위 ±1 — High
- `Boat_Position` (mm): 정상범위 ±0.5 — Medium

**공간 분포 증거 (실측):
- 중심 집중도: 0.095
- 에지 집중도: 0.000
- 불량 밀도: 0.1533

**처방:** 온도 구배 프로파일 보정, 가스 인젝터 유량 균일화, 웨이퍼 보트 위치 최적화

---

### 🟡 Loc (국소 클러스터 불량)

**물리적 원인:** 파티클 오염 또는 마스크 결함에 의한 국소 영역 불량 클러스터. 특정 레티클 좌표 결함이 반복 스텝 패턴으로 나타나거나, 장비 내 파티클 낙하

**주요 공정 단계:** Deposition / Ion Implantation / Lithography
**관련 장비:** Ion Implanter, CVD Chamber, Reticle/Mask
**심각도:** Medium (점수: 3/5)
**수율 영향도:** 5%
**예상 ROI:** 500%

**핵심 파라미터:**
- `Chamber_Pressure` (Torr): 정상범위 <1e-6 — Critical
- `Particle_Count` (ea/wafer): 정상범위 <10 — Critical
- `Deposition_Rate` (nm/s): 정상범위 ±3% — High
- `Implant_Dose_Uniformity` (%): 정상범위 <1 — High

**공간 분포 증거 (실측):
- 중심 집중도: 0.239
- 에지 집중도: 0.000
- 불량 밀도: 0.1512

**처방:** 진공 챔버 파티클 모니터링 강화, 레티클 주기적 검사, 챔버 세정 주기 최적화

---

### 🔴 Near-full (웨이퍼 전면 불량)

**물리적 원인:** 웨이퍼 전체에 걸친 심각한 공정 이상. 슬립 전위(slip dislocation), 적층 결함(stacking fault) 또는 화학물질 오염에 의한 전면적 격자 결함

**주요 공정 단계:** Multiple (Contamination / Crystal Defect)
**관련 장비:** All Process Equipment, Chemical Supply System
**심각도:** Critical (점수: 5/5)
**수율 영향도:** 95%
**예상 ROI:** 3000%

**핵심 파라미터:**
- `Chemical_Purity` (ppb): 정상범위 <1 — Critical
- `Thermal_Shock` (°C/s): 정상범위 <50 — Critical
- `Crystal_Defect_Density` (ea/cm²): 정상범위 <10 — Critical
- `Metal_Contamination` (atoms/cm²): 정상범위 <1e10 — Critical

**공간 분포 증거 (실측):
- 중심 집중도: 0.260
- 에지 집중도: 0.000
- 불량 밀도: 0.8765

**처방:** 즉각 로트 격리 및 근본 원인 분석(RCA), 화학물질 공급 라인 점검, 결정 성장 조건 재검토

---

### 🟡 Random (랜덤 분산 불량)

**물리적 원인:** 무작위 파티클 오염 또는 ESD(정전기 방전)에 의한 산발적 불량. 청정실 클래스 저하 또는 작업자 오염으로 발생

**주요 공정 단계:** Cleanroom Environment / Wafer Handling
**관련 장비:** Cleanroom HVAC, Wafer Handler Robot, FOUP
**심각도:** Medium (점수: 3/5)
**수율 영향도:** 4%
**예상 ROI:** 400%

**핵심 파라미터:**
- `Cleanroom_Class` (ISO): 정상범위 ≤ISO3 — Critical
- `Particle_Concentration` (ea/m³): 정상범위 <1000 — Critical
- `ESD_Voltage` (V): 정상범위 <100 — High
- `Air_Flow_Velocity` (m/s): 정상범위 0.3-0.5 — Medium

**공간 분포 증거 (실측):
- 중심 집중도: 0.259
- 에지 집중도: 0.000
- 불량 밀도: 0.4799

**처방:** 청정실 파티클 카운터 증설, ESD 스트랩 착용 강화, FOUP 세정 주기 단축

---

### 🟡 Scratch (스크래치 불량)

**물리적 원인:** 웨이퍼 핸들링 또는 CMP 공정 중 물리적 접촉에 의한 선형 기계적 손상. 핸들러 암의 압력 불균일 또는 슬러리 내 대형 파티클

**주요 공정 단계:** Wafer Handling / CMP
**관련 장비:** Wafer Handler Robot, CMP Polisher, Load/Unload Station
**심각도:** Medium (점수: 3/5)
**수율 영향도:** 3%
**예상 ROI:** 300%

**핵심 파라미터:**
- `Handler_Pressure` (N): 정상범위 0.5-2.0 — Critical
- `Handler_Speed` (mm/s): 정상범위 <300 — High
- `Slurry_Particle_Size` (nm): 정상범위 <200 — High
- `Pad_Condition` (count): 정상범위 <5000 wafers — Medium

**공간 분포 증거 (실측):
- 중심 집중도: 0.200
- 에지 집중도: 0.000
- 불량 밀도: 0.1003

**처방:** 핸들러 암 압력 캘리브레이션, 슬러리 필터 교환 주기 단축, 패드 드레싱 최적화

---
