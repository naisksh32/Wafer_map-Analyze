import { watch, onMounted, nextTick } from 'vue'
import { DEFECT_META } from '../data/defectMeta'

function seededRand(n) {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453
  return x - Math.floor(x)
}

// ── 국소(Loc): 클러스터 위치·크기 랜덤 생성 ──────────────────────
function generateLocClusters() {
  const G = 64, R = G / 2 - 0.5, cx = G / 2, cy = G / 2
  const count = 1 + Math.floor(Math.random() * 2)
  const clusters = []
  for (let i = 0; i < count; i++) {
    const angle = Math.random() * Math.PI * 2
    const rFrac = 0.12 + Math.random() * 0.62
    clusters.push({
      lx:   cx + Math.cos(angle) * R * rFrac,
      ly:   cy + Math.sin(angle) * R * rFrac,
      size: 5.5 + Math.random() * 4.5,
    })
  }
  return clusters
}

// ── 에지국소(Edge-Loc): 에지 섹터 시작각·호 폭 랜덤 ──────────────
function generateEdgeLocParams() {
  return {
    startAngle: Math.random() * Math.PI * 2,          // 에지 어느 방향이든
    arcWidth:   0.75 + Math.random() * 0.65,          // 호 폭 0.75~1.4 rad (43~80°)
  }
}

// ── 스크래치(Scratch): 선 방향·중심 오프셋·폭 랜덤 ──────────────
function generateScratchParams() {
  return {
    theta:   Math.random() * Math.PI,                 // 선 방향 0~180°
    kOffset: (Math.random() - 0.5) * 10,              // 중심에서 수직 오프셋 ±5 die
    width:   1.1 + Math.random() * 0.6,               // 스크래치 폭 1.1~1.7 die
  }
}

let locClusters    = generateLocClusters()
let edgeLocParams  = generateEdgeLocParams()
let scratchParams  = generateScratchParams()

function renderWafer(canvas, defectProbs, dominantDefect) {
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  const W = canvas.width, H = canvas.height
  const G = 64
  const cw = W / G, ch = H / G
  const cx = G / 2, cy = G / 2
  const R = G / 2 - 0.5

  ctx.clearRect(0, 0, W, H)

  ctx.fillStyle = '#06101C'
  ctx.beginPath()
  ctx.arc(W / 2, H / 2, W / 2, 0, Math.PI * 2)
  ctx.fill()

  const dom  = dominantDefect
  const domP = dom !== 'none' ? (defectProbs[dom] ?? 0) : 0
  const meta = dom !== 'none' ? DEFECT_META[dom] : null

  for (let r = 0; r < G; r++) {
    for (let c = 0; c < G; c++) {
      const dx = c - cx + 0.5
      const dy = r - cy + 0.5
      const dist  = Math.sqrt(dx * dx + dy * dy)
      const rNorm = dist / R
      if (rNorm > 1.01) continue

      const seed = r * 100 + c
      const rand  = seededRand(seed)
      const rand2 = seededRand(seed + 9999)

      let isDefect = false
      if (meta && domP > 0.3) {
        const angle = Math.atan2(dy, dx)
        switch (meta.pattern) {
          case 'center':
            isDefect = rNorm < 0.38 && rand < domP * 0.90
            break
          case 'donut':
            isDefect = rNorm > 0.24 && rNorm < 0.62 && rand < domP * 0.80
            break
          case 'edge-ring':
            isDefect = rNorm > 0.78 && rand < domP * 0.92
            break

          case 'edge-loc': {
            // 랜덤 섹터: startAngle 기준으로 arcWidth 범위 내 에지 영역
            let a = angle - edgeLocParams.startAngle
            if (a >  Math.PI) a -= Math.PI * 2
            if (a < -Math.PI) a += Math.PI * 2
            isDefect = rNorm > 0.73 && Math.abs(a) < edgeLocParams.arcWidth / 2 && rand < domP * 0.88
            break
          }

          case 'loc': {
            // 랜덤 위치 클러스터 (1~2개)
            const inCluster = locClusters.some(cl =>
              Math.sqrt((c - cl.lx) ** 2 + (r - cl.ly) ** 2) < cl.size
            )
            isDefect = inCluster && rand < domP * 0.88
            break
          }

          case 'near-full':
            isDefect = rand < domP * 0.94
            break
          case 'random':
            isDefect = rand < domP * 0.40 && rand2 > 0.3
            break

          case 'scratch': {
            // 랜덤 방향·위치의 직선: 수직 거리 = |-dx·sinθ + dy·cosθ - kOffset|
            const perp = -dx * Math.sin(scratchParams.theta) + dy * Math.cos(scratchParams.theta)
            const d = Math.abs(perp - scratchParams.kOffset)
            isDefect = d < scratchParams.width && rNorm < 0.94 && rand < domP * 0.88
            break
          }
        }
      }

      if (isDefect) {
        const alpha = Math.round((0.65 + rand * 0.35) * 255).toString(16).padStart(2, '0')
        ctx.fillStyle = meta.color + alpha
      } else {
        const bright = 0.55 + rand * 0.20
        ctx.fillStyle = `rgba(30,${Math.round(100 + bright * 60)},55,${0.75 + rand * 0.15})`
      }

      ctx.fillRect(
        Math.floor(c * cw) + 1,
        Math.floor(r * ch) + 1,
        Math.ceil(cw) - 1.5,
        Math.ceil(ch) - 1.5,
      )
    }
  }

  // 웨이퍼 외곽선
  ctx.beginPath()
  ctx.arc(W / 2, H / 2, W / 2 - 2, 0, Math.PI * 2)
  ctx.strokeStyle = 'rgba(0,212,255,0.35)'
  ctx.lineWidth = 2
  ctx.stroke()

  // 노치 (하단)
  ctx.beginPath()
  ctx.arc(W / 2, H - 3, 5, 0, Math.PI * 2)
  ctx.fillStyle = '#06101C'
  ctx.fill()

  // 중앙 크로스헤어
  ctx.strokeStyle = 'rgba(255,255,255,0.04)'
  ctx.lineWidth = 0.5
  ctx.beginPath(); ctx.moveTo(W / 2, 4);   ctx.lineTo(W / 2, H - 4); ctx.stroke()
  ctx.beginPath(); ctx.moveTo(4, H / 2);   ctx.lineTo(W - 4, H / 2); ctx.stroke()
}

export function useWaferCanvas(canvasRef, defectProbsRef, dominantDefectRef) {
  function render() {
    renderWafer(canvasRef.value, defectProbsRef.value, dominantDefectRef.value)
  }

  let prevDominant = null

  watch(dominantDefectRef, (newVal) => {
    // 불량 유형이 전환될 때마다 해당 유형의 패턴 파라미터 재생성
    if (newVal !== prevDominant) {
      if (newVal === 'Loc')      locClusters   = generateLocClusters()
      if (newVal === 'Edge-Loc') edgeLocParams = generateEdgeLocParams()
      if (newVal === 'Scratch')  scratchParams  = generateScratchParams()
    }
    prevDominant = newVal
    nextTick(render)
  })

  watch(defectProbsRef, () => nextTick(render), { deep: true })
  onMounted(() => nextTick(render))

  return { render }
}
