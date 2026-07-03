<script setup>
import { ref, computed } from 'vue'
import { PRESETS, DEFECT_PRESETS } from './data/paramConfig'
import { useDefectModel } from './composables/useDefectModel'
import AppHeader      from './components/AppHeader.vue'
import ParameterPanel from './components/ParameterPanel.vue'
import WaferCanvas    from './components/WaferCanvas.vue'
import ProcessFlow    from './components/ProcessFlow.vue'
import ProbabilityChart from './components/ProbabilityChart.vue'
import DefectInfoCard from './components/DefectInfoCard.vue'
import SpcPanel       from './components/SpcPanel.vue'

// ── 공정 파라미터 상태 (정상 기본값) ──────────────────────────
const params = ref({ ...PRESETS.normal })

// ── 불량 모델 계산 ────────────────────────────────────────────
const {
  defectProbs, dominantDefect, totalYieldLoss,
  estimatedYield, monthlyLoss, alarms, processStages,
  getParamStatus, formatParamVal, normalRangeStyle,
} = useDefectModel(params)

// ── 헤더용 주요 불량 레이블 ──────────────────────────────────
import { DEFECT_META } from './data/defectMeta'
const dominantLabel = computed(() =>
  dominantDefect.value !== 'none'
    ? DEFECT_META[dominantDefect.value]?.korean ?? dominantDefect.value
    : '정상'
)

// ── 프리셋 ────────────────────────────────────────────────────
function applyPreset(type) {
  stopAnimation()
  if (PRESETS[type]) {
    params.value = { ...PRESETS[type] }
  } else if (DEFECT_PRESETS[type]) {
    params.value = { ...DEFECT_PRESETS[type] }
  }
}

// ── 드리프트 애니메이션 ──────────────────────────────────────
const animating = ref(false)
let animHandle = null
let animTick   = 0

function toggleAnimation() {
  animating.value ? stopAnimation() : startAnimation()
}

function startAnimation() {
  params.value = { ...PRESETS.normal }
  animating.value = true
  animTick = 0
  runAnim()
}

function stopAnimation() {
  animating.value = false
  cancelAnimationFrame(animHandle)
  animHandle = null
}

function runAnim() {
  if (!animating.value) return
  animTick++
  const t = animTick / 200            // 0 → 1 over ~200 frames
  const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t  // ease-in-out

  if (t > 1.15) {
    animTick = 0
    params.value = { ...PRESETS.normal }
  } else {
    params.value = {
      cmp_pressure:    100 + ease * 33,
      polish_time:     40  - ease * 29,
      slurry_ph:       6.5 - ease * 3.4,
      annealing_temp:  1100 + ease * 62,
      temp_gradient:   1.0 + ease * 3.4,
      etch_depth:      500 + ease * 64,
      vacuum_pressure: 1.0 + ease * 1.85,
      pr_thickness_cv: 1.8 + ease * 2.6,
      particle_count:  5   + ease * 19,
    }
  }
  animHandle = requestAnimationFrame(runAnim)
}
</script>

<template>
  <AppHeader
    :estimated-yield="estimatedYield"
    :monthly-loss="monthlyLoss"
    :alarm-count="alarms.length"
    :dominant-label="dominantLabel"
    :animating="animating"
    @preset="applyPreset"
    @toggle-anim="toggleAnimation"
  />

  <main class="main-grid">
    <!-- 좌측: 파라미터 슬라이더 -->
    <ParameterPanel
      :params="params"
      :get-param-status="getParamStatus"
      :format-param-val="formatParamVal"
      :normal-range-style="normalRangeStyle"
      :alarms="alarms"
      @update:params="params = $event"
    />

    <!-- 중앙: 웨이퍼 맵 + 공정 흐름 -->
    <div class="center-col">
      <WaferCanvas
        :defect-probs="defectProbs"
        :dominant-defect="dominantDefect"
      />
      <ProcessFlow :process-stages="processStages" />
    </div>

    <!-- 우측: 분석 패널 -->
    <aside class="analysis-panel">
      <div class="panel-title">불량 위험도 분석</div>
      <ProbabilityChart :defect-probs="defectProbs" />
      <DefectInfoCard
        :dominant-defect="dominantDefect"
        :defect-probs="defectProbs"
        :monthly-loss="monthlyLoss"
        :total-yield-loss="totalYieldLoss"
      />
      <SpcPanel :params="params" />
    </aside>
  </main>

  <footer class="footer">
    <span class="pulse"></span>
    <span>실시간 시뮬레이션 중</span>
    <span class="sep">|</span>
    <span>WM-811K Dataset · 172,950 labeled samples</span>
    <span class="sep">|</span>
    <span>Pearson/Spearman 상관계수 기반 예측 모델</span>
    <span class="sep">|</span>
    <span>WaferCNN F1=0.8458 · ViT-Tiny F1=0.8352</span>
    <span class="footer__right">SK하이닉스 대비 포트폴리오 — WM-811K v2.0</span>
  </footer>
</template>

<style scoped>
.main-grid {
  flex: 1;
  display: grid;
  grid-template-columns: 270px 1fr 300px;
  gap: 1px;
  background: var(--border);
  overflow: hidden;
  min-height: 0;
}

.center-col {
  background: var(--panel);
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 12px 8px;
  gap: 8px;
  overflow-y: auto;
}

.analysis-panel {
  background: var(--panel);
  overflow-y: auto;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.footer {
  background: #06101C;
  border-top: 1px solid var(--border);
  padding: 5px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 0.62rem;
  color: var(--sub);
  flex-shrink: 0;
}
.pulse {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  flex-shrink: 0;
  animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }
.sep { color: var(--border); }
.footer__right { margin-left: auto; color: var(--accent); }
</style>



