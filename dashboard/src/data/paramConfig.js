export const PARAM_CONFIG = {
  cmp_pressure: {
    label: 'CMP 압력', unit: 'psi',
    min: 60, max: 140, normalMin: 80, normalMax: 120, mean: 100, std: 10, step: 1,
  },
  polish_time: {
    label: '연마 시간', unit: 's',
    min: 10, max: 80, normalMin: 20, normalMax: 60, mean: 40, std: 7, step: 1,
  },
  slurry_ph: {
    label: '슬러리 pH', unit: '',
    min: 3.0, max: 10.0, normalMin: 4.0, normalMax: 8.0, mean: 6.5, std: 0.5, step: 0.1,
  },
  annealing_temp: {
    label: '열처리 온도', unit: '°C',
    min: 1000, max: 1200, normalMin: 1050, normalMax: 1150, mean: 1100, std: 25, step: 1,
  },
  temp_gradient: {
    label: '온도 구배', unit: '°C (에지-센터)',
    min: 0, max: 5.0, normalMin: 0, normalMax: 2.0, mean: 1.0, std: 0.3, step: 0.05,
  },
  etch_depth: {
    label: '식각 깊이', unit: 'nm',
    min: 430, max: 570, normalMin: 450, normalMax: 550, mean: 500, std: 25, step: 1,
  },
  vacuum_pressure: {
    label: '진공 압력 (×10⁻³)', unit: 'Torr',
    min: 0.5, max: 3.0, normalMin: 0.5, normalMax: 2.0, mean: 1.0, std: 0.15, step: 0.05,
  },
  pr_thickness_cv: {
    label: 'PR 두께 균일도', unit: '%CV',
    min: 0.3, max: 5.0, normalMin: 0, normalMax: 2.0, mean: 1.8, std: 0.7, step: 0.1,
  },
  particle_count: {
    label: '파티클 수', unit: 'ea/wafer',
    min: 0, max: 30, normalMin: 0, normalMax: 10, mean: 5, std: 3, step: 0.5,
  },
}

// ─── 기본 상태 프리셋 ───────────────────────────────────────────
export const PRESETS = {
  normal: {
    cmp_pressure: 100, polish_time: 40, slurry_ph: 6.5,
    annealing_temp: 1100, temp_gradient: 1.0, etch_depth: 500,
    vacuum_pressure: 1.0, pr_thickness_cv: 1.8, particle_count: 5,
  },

  // 경고: CMP 3중 이탈 → Edge-Loc 지배 (에지 일부 섹터 패턴)
  // cmp_pressure↑(z=3.0), polish_time↓(z=-3.86), slurry_ph↓(z=-6.0)
  // Edge-Loc score≈5.44 → 확률≈100%, 나머지 불량은 거의 0%
  warning: {
    cmp_pressure: 130, polish_time: 13, slurry_ph: 3.5,
    annealing_temp: 1100, temp_gradient: 1.0, etch_depth: 500,
    vacuum_pressure: 1.0, pr_thickness_cv: 1.8, particle_count: 5,
  },

  // 위험: 파티클+열처리+온도구배 동시 붕괴 → Near-full 지배 (전면 커버리지)
  // particle_count↑(z=6.33), annealing_temp↑(z=3.4), temp_gradient↑(z=6.0)
  // Near-full score≈6.8 → 확률≈100%, 웨이퍼 전면 적색
  critical: {
    cmp_pressure: 130, polish_time: 11, slurry_ph: 3.1,
    annealing_temp: 1185, temp_gradient: 2.8, etch_depth: 556,
    vacuum_pressure: 1.2, pr_thickness_cv: 3.5, particle_count: 24,
  },
}

// ─── 불량 유형별 시나리오 프리셋 ────────────────────────────────
export const DEFECT_PRESETS = {
  Center: {
    cmp_pressure: 100, polish_time: 40, slurry_ph: 6.5,
    annealing_temp: 1190, temp_gradient: 0.3, etch_depth: 500,
    vacuum_pressure: 1.0, pr_thickness_cv: 1.8, particle_count: 3,
  },
  'Edge-Ring': {
    cmp_pressure: 100, polish_time: 40, slurry_ph: 6.5,
    annealing_temp: 1050, temp_gradient: 2.8, etch_depth: 500,
    vacuum_pressure: 1.0, pr_thickness_cv: 1.8, particle_count: 2,
  },
  'Edge-Loc': {
    cmp_pressure: 135, polish_time: 13, slurry_ph: 3.2,
    annealing_temp: 1100, temp_gradient: 1.0, etch_depth: 500,
    vacuum_pressure: 1.0, pr_thickness_cv: 1.8, particle_count: 5,
  },
  Loc: {
    cmp_pressure: 100, polish_time: 40, slurry_ph: 6.5,
    annealing_temp: 1050, temp_gradient: 0.3, etch_depth: 500,
    vacuum_pressure: 2.8, pr_thickness_cv: 1.8, particle_count: 3,
  },
  'Near-full': {
    cmp_pressure: 100, polish_time: 40, slurry_ph: 6.5,
    annealing_temp: 1185, temp_gradient: 2.2, etch_depth: 500,
    vacuum_pressure: 1.0, pr_thickness_cv: 1.8, particle_count: 26,
  },
  Donut: {
    cmp_pressure: 100, polish_time: 40, slurry_ph: 6.5,
    annealing_temp: 1100, temp_gradient: 1.0, etch_depth: 566,
    vacuum_pressure: 1.0, pr_thickness_cv: 4.8, particle_count: 5,
  },
  Random: {
    cmp_pressure: 100, polish_time: 40, slurry_ph: 6.5,
    annealing_temp: 1040, temp_gradient: 0.3, etch_depth: 500,
    vacuum_pressure: 0.6, pr_thickness_cv: 1.8, particle_count: 18,
  },
  Scratch: {
    cmp_pressure: 100, polish_time: 75, slurry_ph: 6.5,
    annealing_temp: 1100, temp_gradient: 1.0, etch_depth: 500,
    vacuum_pressure: 1.0, pr_thickness_cv: 1.8, particle_count: 5,
  },
}
