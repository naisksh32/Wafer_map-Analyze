<script setup>
import { DEFECT_META } from '../data/defectMeta'

defineProps({
  estimatedYield: Number,
  monthlyLoss:    Number,
  alarmCount:     Number,
  dominantLabel:  String,
  animating:      Boolean,
})
defineEmits(['preset', 'toggle-anim'])

const DEFECT_BTNS = [
  { key: 'Center',    label: '중심' },
  { key: 'Edge-Ring', label: '에지링' },
  { key: 'Edge-Loc',  label: '에지국소' },
  { key: 'Loc',       label: '국소' },
  { key: 'Near-full', label: '전면' },
  { key: 'Donut',     label: '도넛' },
  { key: 'Random',    label: '랜덤' },
  { key: 'Scratch',   label: '스크래치' },
]
</script>

<template>
  <header class="header">
    <div class="header__logo">
      <div class="logo-text">⚡ WM-811K <span>Wafer Process Simulator</span></div>
      <div class="logo-sub">SK하이닉스 대비 포트폴리오 | 공정 파라미터 → 불량 위험도 실시간 시뮬레이션</div>
    </div>

    <div class="header__right">
      <div class="stats">
        <div class="stat">
          <span class="stat__val"
            :class="estimatedYield >= 95 ? 'c-green' : estimatedYield >= 85 ? 'c-yellow' : 'c-red'">
            {{ estimatedYield.toFixed(1) }}%
          </span>
          <span class="stat__lbl">추정 수율</span>
        </div>
        <div class="stat">
          <span class="stat__val" :class="monthlyLoss > 0 ? 'c-red' : 'c-green'">
            {{ monthlyLoss > 0 ? `-$${monthlyLoss.toFixed(1)}M` : '$0' }}
          </span>
          <span class="stat__lbl">월 손실</span>
        </div>
        <div class="stat">
          <span class="stat__val"
            :class="alarmCount === 0 ? 'c-green' : alarmCount > 2 ? 'c-red' : 'c-yellow'">
            {{ alarmCount }}
          </span>
          <span class="stat__lbl">경보</span>
        </div>
        <div class="stat">
          <span class="stat__val c-accent">{{ dominantLabel }}</span>
          <span class="stat__lbl">주요 불량</span>
        </div>
      </div>

      <div class="preset-group">
        <!-- 기본 상태 버튼 -->
        <div class="preset-row">
          <span class="row-lbl">상태</span>
          <button class="btn btn--green"  @click="$emit('preset', 'normal')">정상 전환</button>
          <button class="btn btn--yellow" @click="$emit('preset', 'warning')">경고 시나리오</button>
          <button class="btn btn--red"    @click="$emit('preset', 'critical')">위험 시나리오</button>
          <button class="btn btn--accent" :class="{ 'is-active': animating }"
                  @click="$emit('toggle-anim')">
            {{ animating ? '⏹ 정지' : '▶ 드리프트' }}
          </button>
        </div>

        <!-- 불량 유형별 시나리오 버튼 -->
        <div class="preset-row">
          <span class="row-lbl">불량</span>
          <button
            v-for="d in DEFECT_BTNS" :key="d.key"
            class="btn btn--defect"
            :style="`--defect-color:${DEFECT_META[d.key].color}`"
            @click="$emit('preset', d.key)"
          >
            {{ d.label }}
          </button>
        </div>
      </div>
    </div>
  </header>
</template>

<style scoped>
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 20px;
  background: linear-gradient(90deg, #060D1A, #0F1A2E);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.logo-text {
  font-size: 1rem;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: .02em;
}
.logo-text span { color: #5B8CCA; font-weight: 400; }
.logo-sub { font-size: .65rem; color: var(--sub); margin-top: 2px; }

.header__right { display: flex; align-items: center; gap: 16px; }
.stats { display: flex; gap: 20px; }
.stat { display: flex; flex-direction: column; align-items: center; gap: 1px; }
.stat__val { font-size: 1.1rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.stat__lbl { font-size: .6rem; color: var(--sub); text-transform: uppercase; letter-spacing: .08em; }

/* ─── 버튼 그룹 ─── */
.preset-group {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.preset-row {
  display: flex;
  align-items: center;
  gap: 4px;
}
.row-lbl {
  font-size: .55rem;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: var(--sub);
  width: 22px;
  flex-shrink: 0;
  text-align: right;
  padding-right: 4px;
}

/* 불량 유형 버튼 */
.btn--defect {
  padding: 4px 8px;
  border-radius: 4px;
  border: 1px solid rgba(var(--defect-color), .4);
  border-color: color-mix(in srgb, var(--defect-color) 40%, transparent);
  background: color-mix(in srgb, var(--defect-color) 8%, transparent);
  color: var(--defect-color);
  font-size: .65rem;
  cursor: pointer;
  transition: all .15s;
  white-space: nowrap;
  font-family: inherit;
}
.btn--defect:hover {
  background: color-mix(in srgb, var(--defect-color) 18%, transparent);
  transform: translateY(-1px);
}
</style>
