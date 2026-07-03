export const DEFECT_META = {
  Center: {
    korean: '중심 불량', color: '#FF6B6B', severity: 5, yieldImpact: 0.15, roi: 800,
    pattern: 'center', stage: '열처리',
    mechanism: '웨이퍼 중심부 열처리 불균일. 산소 석출물(SiO₂)이 중심에 집중 생성되어 게이트 산화막 품질 저하 유발.',
    remediation: '냉각 속도 제어 최적화, RTP 온도 균일도 개선, 산소 농도 타겟 재설정.',
    correlations: { annealing_temp: 0.5265 },
  },
  Donut: {
    korean: '도넛형 불량', color: '#4ECDC4', severity: 4, yieldImpact: 0.10, roi: 650,
    pattern: 'donut', stage: '포토/식각',
    mechanism: '스핀 코팅 시 중심·에지 경계 두께 전이 구간 발생. 링 형태 식각 불균일.',
    remediation: '스핀 코팅 속도 프로파일 최적화, 식각 가스 유량 균일도 개선, Edge Bead Removal 적용.',
    correlations: { pr_thickness_cv: 0.6083, etch_depth: 0.35 },
  },
  'Edge-Loc': {
    korean: '국소 에지 불량', color: '#45B7D1', severity: 4, yieldImpact: 0.08, roi: 1200,
    pattern: 'edge-loc', stage: 'CMP',
    mechanism: 'CMP 리테이닝 링 압력 불균일로 에지 특정 방위각 구간에만 불량 집중. PVD 에지 두께 차이.',
    remediation: '리테이닝 링 압력 맵 보정, 슬러리 pH 실시간 모니터링, 에지 프로파일 측정 강화.',
    correlations: { polish_time: -0.554, cmp_pressure: 0.40, slurry_ph: -0.35 },
  },
  'Edge-Ring': {
    korean: '에지 링 불량', color: '#A29BFE', severity: 4, yieldImpact: 0.09, roi: 950,
    pattern: 'edge-ring', stage: '열처리',
    mechanism: '퍼니스 내 가스 흐름의 에지-센터 온도 구배가 에지 산화막 성장 속도 차이를 유발.',
    remediation: '온도 구배 프로파일 보정, 가스 인젝터 유량 균일화, 웨이퍼 보트 위치 최적화.',
    correlations: { temp_gradient: 0.664 },
  },
  Loc: {
    korean: '국소 클러스터 불량', color: '#FFEAA7', severity: 3, yieldImpact: 0.05, roi: 500,
    pattern: 'loc', stage: '이온주입',
    mechanism: '파티클 오염 또는 마스크 결함에 의한 국소 영역 불량 클러스터. 장비 내 파티클 낙하.',
    remediation: '진공 챔버 파티클 모니터링 강화, 레티클 주기적 검사, 챔버 세정 주기 최적화.',
    correlations: { vacuum_pressure: 0.5337, particle_count: 0.40 },
  },
  'Near-full': {
    korean: '웨이퍼 전면 불량', color: '#FF7675', severity: 5, yieldImpact: 0.95, roi: 3000,
    pattern: 'near-full', stage: '다중 공정',
    mechanism: '슬립 전위, 적층 결함 또는 화학물질 오염에 의한 전면적 격자 결함.',
    remediation: '즉각 로트 격리 및 RCA 수행, 화학물질 공급 라인 점검, 결정 성장 조건 재검토.',
    correlations: { particle_count: 0.5311, annealing_temp: 0.40, temp_gradient: 0.35 },
  },
  Random: {
    korean: '랜덤 분산 불량', color: '#74B9FF', severity: 3, yieldImpact: 0.04, roi: 400,
    pattern: 'random', stage: '청정실',
    mechanism: '무작위 파티클 오염 또는 ESD 방전에 의한 산발적 불량. 청정실 클래스 저하.',
    remediation: '청정실 파티클 카운터 증설, ESD 스트랩 착용 강화, FOUP 세정 주기 단축.',
    correlations: { particle_count: 0.3989 },
  },
  Scratch: {
    korean: '스크래치 불량', color: '#FD79A8', severity: 3, yieldImpact: 0.03, roi: 300,
    pattern: 'scratch', stage: '핸들링',
    mechanism: '웨이퍼 핸들러 암 과압 또는 CMP 슬러리 내 대형 파티클에 의한 선형 기계적 손상.',
    remediation: '핸들러 암 압력 캘리브레이션, 슬러리 필터 교환 주기 단축, 패드 드레싱 최적화.',
    correlations: { polish_time: 0.4439 },
  },
}

export const STAGE_DEFS = [
  { id: 'wafer-prep', icon: '💎', name: '웨이퍼 준비' },
  { id: 'thermal',    icon: '🌡️', name: '열처리' },
  { id: 'litho',      icon: '📸', name: '포토리소' },
  { id: 'etch',       icon: '⚡',  name: '식각' },
  { id: 'cmp',        icon: '💿', name: 'CMP' },
  { id: 'pvd',        icon: '🌫️', name: 'PVD/CVD' },
  { id: 'implant',    icon: '⚛️', name: '이온주입' },
  { id: 'clean',      icon: '🧹', name: '세정' },
  { id: 'handle',     icon: '🤖', name: '핸들링' },
  { id: 'inspect',    icon: '🔬', name: '검사' },
]

export const STAGE_DEFECT_MAP = {
  'wafer-prep': ['Center'],
  thermal:      ['Center', 'Edge-Ring', 'Near-full'],
  litho:        ['Donut', 'Loc'],
  etch:         ['Donut', 'Edge-Loc'],
  cmp:          ['Edge-Loc', 'Scratch'],
  pvd:          ['Loc', 'Edge-Loc'],
  implant:      ['Loc'],
  clean:        ['Edge-Ring', 'Random'],
  handle:       ['Scratch', 'Random'],
  inspect:      [],
}

export const SPC_PARAM_KEYS = ['temp_gradient', 'particle_count', 'annealing_temp', 'cmp_pressure']
