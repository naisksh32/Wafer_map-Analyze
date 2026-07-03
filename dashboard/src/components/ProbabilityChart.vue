<script setup>
import { ref, watch, onMounted, onBeforeUnmount } from 'vue'
import { Chart, registerables } from 'chart.js'
import { DEFECT_META } from '../data/defectMeta'

Chart.register(...registerables)

const props = defineProps({ defectProbs: Object })

const canvasRef = ref(null)
let chart = null

const KEYS   = Object.keys(DEFECT_META)
const LABELS = KEYS.map(k => DEFECT_META[k].korean)
const COLORS = KEYS.map(k => DEFECT_META[k].color)

function getProbs() {
  return KEYS.map(k => Math.round((props.defectProbs[k] ?? 0) * 100))
}

function getBgColors() {
  return KEYS.map(k => {
    const p = props.defectProbs[k] ?? 0
    const alpha = Math.round(Math.min(p * 180 + 30, 200)).toString(16).padStart(2, '0')
    return DEFECT_META[k].color + alpha
  })
}

function initChart() {
  chart = new Chart(canvasRef.value, {
    type: 'bar',
    data: {
      labels: LABELS,
      datasets: [{
        data: getProbs(),
        backgroundColor: getBgColors(),
        borderColor: COLORS,
        borderWidth: 1.5,
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0F1A2E',
          borderColor: '#1C2D47',
          borderWidth: 1,
          callbacks: { label: ctx => `위험도 ${ctx.raw}%` },
        },
      },
      scales: {
        x: {
          min: 0, max: 100,
          ticks: { color: '#4A6080', font: { size: 10 }, callback: v => `${v}%` },
          grid: { color: '#1C2D47' },
        },
        y: {
          ticks: { color: '#DDE6F0', font: { size: 10 } },
          grid: { display: false },
        },
      },
    },
  })
}

function updateChart() {
  if (!chart) return
  chart.data.datasets[0].data = getProbs()
  chart.data.datasets[0].backgroundColor = getBgColors()
  chart.update('none')
}

watch(() => props.defectProbs, updateChart, { deep: true })
onMounted(initChart)
onBeforeUnmount(() => chart?.destroy())
</script>

<template>
  <div class="chart-wrap">
    <canvas ref="canvasRef"></canvas>
  </div>
</template>

<style scoped>
.chart-wrap { height: 200px; flex-shrink: 0; }
</style>
