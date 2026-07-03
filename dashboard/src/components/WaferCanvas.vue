<script setup>
import { ref, computed } from 'vue'
import { DEFECT_META } from '../data/defectMeta'
import { useWaferCanvas } from '../composables/useWaferCanvas'

const props = defineProps({
  defectProbs:    Object,
  dominantDefect: String,
})

const canvasRef = ref(null)

const defectProbsRef    = computed(() => props.defectProbs)
const dominantDefectRef = computed(() => props.dominantDefect)

useWaferCanvas(canvasRef, defectProbsRef, dominantDefectRef)

const domMeta = computed(() =>
  props.dominantDefect !== 'none' ? DEFECT_META[props.dominantDefect] : null
)
const domProb = computed(() =>
  props.dominantDefect !== 'none' ? Math.round((props.defectProbs[props.dominantDefect] ?? 0) * 100) : 0
)
</script>

<template>
  <section class="wafer-section">
    <div class="panel-title">웨이퍼 불량 분포도 (64×64 Die Grid)</div>

    <!-- 웨이퍼 캔버스 -->
    <canvas ref="canvasRef" width="448" height="448" class="wafer-canvas"></canvas>

    <!-- 범례 -->
    <div class="legend">
      <div class="legend__item">
        <span class="legend__dot" style="background:#2D7A52"></span>정상 다이
      </div>
      <div class="legend__item" v-if="domMeta">
        <span class="legend__dot" :style="`background:${domMeta.color}`"></span>
        불량 다이 ({{ domMeta.korean }})
      </div>
      <div class="legend__item">
        <span class="legend__dot" style="background:#0A0E1A;border:1px solid #1C2D47"></span>비활성
      </div>
    </div>

    <!-- 주요 불량 배지 -->
    <div class="dom-row">
      <span v-if="domMeta" class="dom-badge"
        :style="`border-color:${domMeta.color};color:${domMeta.color}`">
        주요 불량: {{ domMeta.korean }}
      </span>
      <span v-else class="dom-badge dom-badge--ok">정상 전원 범위</span>
      <span v-if="domMeta" class="dom-prob">확률 {{ domProb }}%</span>
    </div>
  </section>
</template>

<style scoped>
.wafer-section {
  background: var(--panel);
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 12px 8px;
  overflow-y: auto;
  gap: 8px;
}

.wafer-canvas {
  display: block;
  border-radius: 50%;
  box-shadow: 0 0 40px rgba(0,212,255,.08), 0 0 0 1px rgba(0,212,255,.2);
  flex-shrink: 0;
}

.legend { display: flex; gap: 14px; justify-content: center; flex-shrink: 0; }
.legend__item { display: flex; align-items: center; gap: 4px; font-size: .65rem; color: var(--sub); }
.legend__dot { width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }

.dom-row { display: flex; align-items: center; gap: 8px; flex-shrink: 0; flex-wrap: wrap; justify-content: center; }
.dom-badge {
  padding: 5px 14px; border-radius: 20px;
  font-size: .78rem; font-weight: 600;
  border: 1px solid; background: rgba(255,255,255,.03);
}
.dom-badge--ok { border-color: var(--green); color: var(--green); }
.dom-prob { font-size: .72rem; color: var(--sub); }
</style>
