import { computed } from 'vue'
import { DEFECT_META, STAGE_DEFS, STAGE_DEFECT_MAP } from '../data/defectMeta'
import { PARAM_CONFIG } from '../data/paramConfig'

function sigmoid(x) { return 1 / (1 + Math.exp(-x)) }

function calcDefectProb(params, defectType) {
  const meta = DEFECT_META[defectType]
  if (!meta) return 0
  let score = 0
  for (const [p, r] of Object.entries(meta.correlations)) {
    const cfg = PARAM_CONFIG[p]
    if (!cfg || params[p] === undefined) continue
    const z = (params[p] - cfg.mean) / cfg.std
    score += r * z
  }
  // 정상 파라미터(z=0)일 때 확률 0, 이탈 시 최대 1로 선형 변환
  return Math.max(0, (sigmoid(score) - 0.5) * 2)
}

export function useDefectModel(paramsRef) {
  const defectProbs = computed(() => {
    const out = {}
    for (const k of Object.keys(DEFECT_META)) {
      out[k] = calcDefectProb(paramsRef.value, k)
    }
    return out
  })

  const dominantDefect = computed(() => {
    let best = 'none', bestVal = 0.3
    for (const [k, v] of Object.entries(defectProbs.value)) {
      if (v > bestVal) { bestVal = v; best = k }
    }
    return best
  })

  const totalYieldLoss = computed(() => {
    let loss = 0
    for (const [k, v] of Object.entries(defectProbs.value)) {
      const meta = DEFECT_META[k]
      if (meta && v > 0) loss += meta.yieldImpact * v
    }
    return Math.min(loss, 0.99)
  })

  const estimatedYield = computed(() => Math.max(1, (1 - totalYieldLoss.value) * 100))

  // 월 생산 50,000매 × $500/매 = $25M 기준
  const monthlyLoss = computed(() => 25 * totalYieldLoss.value)

  const alarms = computed(() =>
    Object.entries(paramsRef.value)
      .map(([key, value]) => {
        const s = getParamStatus(key, value)
        if (s === 'ok') return null
        const cfg = PARAM_CONFIG[key]
        return { key, label: cfg.label, status: s, value, unit: cfg.unit }
      })
      .filter(Boolean)
  )

  const processStages = computed(() => {
    const dom = dominantDefect.value
    const probs = defectProbs.value
    return STAGE_DEFS.map(s => {
      const defects = STAGE_DEFECT_MAP[s.id] || []
      const isCrit = dom !== 'none' && defects.includes(dom)
      const isWarn = !isCrit && defects.some(d => (probs[d] || 0) > 0.15)
      return { ...s, riskLevel: isCrit ? 'crit' : isWarn ? 'warn' : '' }
    })
  })

  function getParamStatus(key, value) {
    const cfg = PARAM_CONFIG[key]
    if (!cfg) return 'ok'
    if (value >= cfg.normalMin && value <= cfg.normalMax) return 'ok'
    const range = cfg.normalMax - cfg.normalMin
    const dev = Math.max(cfg.normalMin - value, value - cfg.normalMax, 0)
    return dev / (range * 0.5 + 1e-9) > 0.5 ? 'crit' : 'warn'
  }

  function formatParamVal(key) {
    const v = paramsRef.value[key]
    const cfg = PARAM_CONFIG[key]
    if (!cfg) return String(v)
    if (cfg.step >= 1) return Math.round(v).toString()
    if (cfg.step >= 0.1) return v.toFixed(1)
    return v.toFixed(2)
  }

  function normalRangeStyle(key) {
    const cfg = PARAM_CONFIG[key]
    const total = cfg.max - cfg.min
    const leftPct = ((cfg.normalMin - cfg.min) / total) * 100
    const widthPct = ((cfg.normalMax - cfg.normalMin) / total) * 100
    return `left:${leftPct}%;width:${widthPct}%`
  }

  return {
    defectProbs, dominantDefect, totalYieldLoss,
    estimatedYield, monthlyLoss, alarms, processStages,
    getParamStatus, formatParamVal, normalRangeStyle,
  }
}
