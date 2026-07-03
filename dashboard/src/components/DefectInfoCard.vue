<script setup>
import { computed } from 'vue'
import { DEFECT_META } from '../data/defectMeta'

const props = defineProps({
  dominantDefect: String,
  defectProbs:    Object,
  monthlyLoss:    Number,
  totalYieldLoss: Number,
})

const meta = computed(() =>
  props.dominantDefect !== 'none' ? DEFECT_META[props.dominantDefect] : null
)
</script>

<template>
  <!-- 불량 감지 시 -->
  <template v-if="meta">
    <div class="defect-card">
      <div class="card-hdr" :style="`border-color:${meta.color}`">
        <span class="type-badge"
          :style="`background:${meta.color}22;color:${meta.color}`">
          {{ meta.korean }}
        </span>
        <div class="sev-row">
          <span class="sev-lbl">심각도</span>
          <div class="sev-dots">
            <div v-for="i in 5" :key="i" class="dot" :class="{ 'dot--on': i <= meta.severity }"></div>
          </div>
        </div>
      </div>

      <p class="mechanism">{{ meta.mechanism }}</p>

      <div class="remed-box">
        <div class="remed-title">💡 공정 개선 조치</div>
        <p class="remed-text">{{ meta.remediation }}</p>
      </div>

      <div class="impact-row">
        <div class="impact-item">
          <div class="impact-lbl">수율 영향</div>
          <div class="impact-val c-red">{{ Math.round(meta.yieldImpact * 100) }}%</div>
        </div>
        <div class="impact-item">
          <div class="impact-lbl">개선 ROI</div>
          <div class="impact-val c-accent">{{ meta.roi }}%</div>
        </div>
        <div class="impact-item">
          <div class="impact-lbl">발생 공정</div>
          <div class="impact-val" style="font-size:.68rem;color:var(--sub)">{{ meta.stage }}</div>
        </div>
      </div>
    </div>

    <!-- ROI 카드 -->
    <div v-if="monthlyLoss > 0" class="roi-card">
      <div class="roi-title">경제적 영향 (월 50,000매 × $500/매)</div>
      <div class="roi-row">
        <span>월 총 생산액</span>
        <span class="roi-base">$25.0M</span>
      </div>
      <div class="roi-row">
        <span>불량 수율 손실</span>
        <span class="c-red" style="font-weight:600">-${{ monthlyLoss.toFixed(2) }}M/월</span>
      </div>
      <div class="roi-row">
        <span>개선 시 연간 회수</span>
        <span class="c-green" style="font-weight:600">${{ (monthlyLoss * 12).toFixed(1) }}M/년</span>
      </div>
      <div class="roi-bar">
        <div class="roi-fill" :style="`width:${Math.min(totalYieldLoss * 200, 100)}%`"></div>
      </div>
    </div>
  </template>

  <!-- 정상 상태 -->
  <div v-else class="normal-card">
    <div class="normal-icon">✅</div>
    <div class="normal-txt">모든 파라미터 정상 범위</div>
    <div class="normal-sub">불량 발생 위험 없음. 지속적 SPC 모니터링 권고.</div>
  </div>
</template>

<style scoped>
.defect-card {
  background: var(--panel2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  flex-shrink: 0;
}
.card-hdr {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 1px solid;
}
.type-badge {
  padding: 3px 10px;
  border-radius: 12px;
  font-size: .72rem;
  font-weight: 600;
}
.sev-row { display: flex; align-items: center; gap: 5px; }
.sev-lbl { font-size: .6rem; color: var(--sub); }
.sev-dots { display: flex; gap: 2px; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--border); }
.dot--on { background: var(--red); }

.mechanism { font-size: .7rem; color: var(--sub); line-height: 1.55; margin-bottom: 8px; }

.remed-box {
  background: rgba(0,212,255,.05);
  border-left: 2px solid var(--accent);
  border-radius: 0 4px 4px 0;
  padding: 7px 10px;
  margin-bottom: 8px;
}
.remed-title { font-size: .62rem; color: var(--accent); font-weight: 600; margin-bottom: 3px; }
.remed-text  { font-size: .68rem; color: var(--sub); line-height: 1.5; margin: 0; }

.impact-row { display: flex; gap: 8px; }
.impact-item {
  flex: 1;
  background: var(--panel3);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 7px 10px;
  text-align: center;
}
.impact-lbl { font-size: .6rem; color: var(--sub); margin-bottom: 3px; }
.impact-val { font-size: .95rem; font-weight: 700; }

/* ROI */
.roi-card {
  background: var(--panel2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  flex-shrink: 0;
}
.roi-title { font-size: .62rem; text-transform: uppercase; letter-spacing: .08em; color: var(--sub); margin-bottom: 8px; }
.roi-row { display: flex; justify-content: space-between; align-items: center; font-size: .7rem; margin-bottom: 5px; }
.roi-base { color: var(--sub); }
.roi-bar { height: 4px; background: var(--border); border-radius: 2px; margin-top: 8px; overflow: hidden; }
.roi-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--yellow), var(--red)); transition: width .4s; }

/* 정상 */
.normal-card {
  background: var(--panel2);
  border: 1px solid rgba(0,229,160,.25);
  border-radius: 6px;
  padding: 20px;
  text-align: center;
  flex-shrink: 0;
}
.normal-icon { font-size: 1.8rem; margin-bottom: 6px; }
.normal-txt  { font-size: .78rem; color: var(--green); font-weight: 600; margin-bottom: 4px; }
.normal-sub  { font-size: .68rem; color: var(--sub); }
</style>
