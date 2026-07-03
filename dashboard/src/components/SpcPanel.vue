<script setup>
import { computed } from 'vue'
import { SPC_PARAM_KEYS } from '../data/defectMeta'
import { PARAM_CONFIG } from '../data/paramConfig'

const props = defineProps({ params: Object })

function spcCpk(key) {
  const v   = props.params[key]
  const cfg = PARAM_CONFIG[key]
  const s3  = cfg.std * 3
  const cpu = (cfg.normalMax - v) / s3
  const cpl = (v - cfg.normalMin) / s3
  return Math.min(cpu, cpl)
}

function barWidth(key) {
  return Math.min(Math.max((spcCpk(key) / 2.0) * 100, 0), 100)
}

function barColor(key) {
  const cpk = spcCpk(key)
  if (cpk >= 1.33) return 'var(--green)'
  if (cpk >= 1.0)  return 'var(--yellow)'
  return 'var(--red)'
}

function normalMidPct(key) {
  const cfg = PARAM_CONFIG[key]
  return ((cfg.mean - cfg.min) / (cfg.max - cfg.min)) * 100
}

const rows = computed(() =>
  SPC_PARAM_KEYS.map(key => ({
    key,
    label: PARAM_CONFIG[key].label,
    cpk: spcCpk(key),
    width: barWidth(key),
    color: barColor(key),
    midPct: normalMidPct(key),
  }))
)
</script>

<template>
  <div class="spc-panel">
    <div class="spc-title">SPC 공정 능력 지수 (Cpk 근사)</div>
    <div v-for="row in rows" :key="row.key" class="spc-row">
      <div class="spc-label">{{ row.label }}</div>
      <div class="spc-bar-wrap">
        <div class="spc-bar-fill" :style="`width:${row.width}%;background:${row.color}`"></div>
        <div class="spc-marker" :style="`left:${row.midPct}%`"></div>
      </div>
      <div class="spc-val" :style="`color:${row.color}`">{{ row.cpk.toFixed(2) }}</div>
    </div>
    <div class="spc-legend">
      <span class="c-green">Cpk≥1.33</span> 우수
      <span class="c-yellow" style="margin-left:8px">≥1.0</span> 보통
      <span class="c-red"    style="margin-left:8px">&lt;1.0</span> 불량
    </div>
  </div>
</template>

<style scoped>
.spc-panel {
  background: var(--panel2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px;
  flex-shrink: 0;
}
.spc-title {
  font-size: .62rem;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--sub);
  margin-bottom: 8px;
}
.spc-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.spc-label { font-size: .68rem; color: var(--text); width: 80px; flex-shrink: 0; }
.spc-bar-wrap {
  flex: 1; height: 6px; background: var(--border);
  border-radius: 3px; position: relative; overflow: visible;
}
.spc-bar-fill { height: 100%; border-radius: 3px; transition: width .3s; }
.spc-marker {
  position: absolute; top: -3px;
  width: 2px; height: 12px;
  background: rgba(255,255,255,.25);
  border-radius: 1px;
  transform: translateX(-50%);
}
.spc-val { font-size: .68rem; font-weight: 600; width: 36px; text-align: right; font-variant-numeric: tabular-nums; }
.spc-legend { font-size: .6rem; color: var(--sub); margin-top: 6px; }
</style>
