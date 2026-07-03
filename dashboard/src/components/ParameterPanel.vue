<script setup>
import { PARAM_CONFIG } from '../data/paramConfig'

const props = defineProps({
  params:           Object,
  getParamStatus:   Function,
  formatParamVal:   Function,
  normalRangeStyle: Function,
  alarms:           { type: Array, default: () => [] },
})

const emit = defineEmits(['update:params'])

function onSlider(key, e) {
  emit('update:params', { ...props.params, [key]: Number(e.target.value) })
}
</script>

<template>
  <aside class="param-panel">
    <div class="panel-title">공정 파라미터 제어</div>

    <div v-for="(cfg, key) in PARAM_CONFIG" :key="key" class="param-row">
      <div class="param-hdr">
        <span class="param-label">{{ cfg.label }}</span>
        <div class="param-val-wrap">
          <span class="param-val">{{ formatParamVal(key) }}</span>
          <span class="param-unit">{{ cfg.unit }}</span>
          <span class="status-dot" :class="`s-${getParamStatus(key, params[key])}`"></span>
        </div>
      </div>

      <div class="slider-wrap">
        <div class="track"></div>
        <div class="normal-rng" :style="normalRangeStyle(key)"></div>
        <input
          type="range"
          :min="cfg.min" :max="cfg.max" :step="cfg.step"
          :value="params[key]"
          @input="onSlider(key, $event)"
        />
      </div>

      <div class="range-hints">
        <span>{{ cfg.min }}</span>
        <span class="normal-lbl">정상 {{ cfg.normalMin }}&ndash;{{ cfg.normalMax }}</span>
        <span>{{ cfg.max }}</span>
      </div>
    </div>

    <!-- 경고 알람 영역 (좌측 하단 고정) -->
    <div class="alarm-section">
      <div class="alarm-hdr">
        <span class="alarm-hdr-dot" :class="{ 'dot--active': alarms.length > 0 }"></span>
        공정 이상 알람
        <span v-if="alarms.length" class="alarm-count">{{ alarms.length }}</span>
      </div>
      <transition-group name="alarm" tag="div" class="alarm-list">
        <div
          v-for="a in alarms" :key="a.key"
          class="alarm-item" :class="a.status === 'crit' ? 'alarm--crit' : 'alarm--warn'"
        >
          <span class="alarm-icon">{{ a.status === 'crit' ? '🔴' : '🟡' }}</span>
          <div class="alarm-body">
            <span class="alarm-label">{{ a.label }}</span>
            <span class="alarm-val">{{ typeof a.value === 'number'
              ? a.value.toFixed(a.key === 'slurry_ph' ? 1 : a.key === 'vacuum_pressure' ? 2 : 0)
              : a.value }} {{ a.unit }}</span>
          </div>
          <span class="alarm-msg">{{ a.status === 'crit' ? '즉시 조치' : '주의 요망' }}</span>
        </div>
      </transition-group>
      <div v-if="!alarms.length" class="alarm-ok">
        <span class="ok-dot"></span> 모든 파라미터 정상
      </div>
    </div>
  </aside>
</template>

<style scoped>
.param-panel {
  background: var(--panel);
  overflow-y: auto;
  padding: 14px;
  display: flex;
  flex-direction: column;
}

.param-row { margin-bottom: 12px; flex-shrink: 0; }

.param-hdr {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.param-label { font-size: .72rem; color: var(--text); }
.param-val-wrap { display: flex; align-items: center; gap: 4px; }
.param-val {
  font-size: .78rem; font-weight: 600;
  color: var(--accent); font-variant-numeric: tabular-nums;
  min-width: 42px; text-align: right;
}
.param-unit { font-size: .6rem; color: var(--sub); max-width: 56px; line-height: 1.2; }

.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.s-ok   { background: var(--green); }
.s-warn { background: var(--yellow); box-shadow: 0 0 5px rgba(255,179,0,.4); }
.s-crit { background: var(--red); box-shadow: 0 0 6px rgba(255,61,90,.5); animation: blink 1s infinite; }
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: .3; } }

.slider-wrap { position: relative; height: 18px; margin: 2px 0; }
.track { position: absolute; top: 7px; left: 0; right: 0; height: 3px; background: var(--border); border-radius: 2px; }
.normal-rng { position: absolute; top: 0; height: 100%; background: rgba(0,212,255,.12); border-radius: 2px; }

input[type=range] {
  position: absolute; top: 0; left: 0; width: 100%; height: 18px;
  margin: 0; background: transparent;
  cursor: pointer; -webkit-appearance: none; appearance: none; z-index: 1;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 12px; height: 12px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 5px rgba(0,212,255,.5);
  cursor: pointer; margin-top: -4.5px;
}
input[type=range]::-webkit-slider-runnable-track { height: 3px; background: transparent; }

.range-hints {
  display: flex; justify-content: space-between;
  font-size: .58rem; color: var(--sub); margin-top: 1px;
}
.normal-lbl { color: rgba(0,212,255,.5); }

/* ─── 알람 영역 ─── */
.alarm-section {
  margin-top: auto;
  padding-top: 10px;
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}
.alarm-hdr {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: .62rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: var(--sub);
  margin-bottom: 6px;
}
.alarm-hdr-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--border);
  transition: background .3s;
}
.alarm-hdr-dot.dot--active { background: var(--red); animation: blink 1s infinite; }
.alarm-count {
  margin-left: auto;
  background: var(--red);
  color: #fff;
  font-size: .58rem;
  border-radius: 8px;
  padding: 0 5px;
  line-height: 1.6;
}
.alarm-list { display: flex; flex-direction: column; gap: 4px; }
.alarm-item {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 6px 8px;
  border-radius: 5px;
  font-size: .68rem;
}
.alarm--warn { background: rgba(255,179,0,.08);  border: 1px solid rgba(255,179,0,.3); }
.alarm--crit { background: rgba(255,61,90,.08);  border: 1px solid rgba(255,61,90,.3); }
.alarm-icon { font-size: .75rem; flex-shrink: 0; }
.alarm-body { flex: 1; display: flex; flex-direction: column; gap: 1px; min-width: 0; }
.alarm-label { color: var(--text); font-size: .68rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.alarm-val   { color: var(--sub); font-size: .62rem; font-variant-numeric: tabular-nums; }
.alarm-msg   { flex-shrink: 0; font-size: .6rem; font-weight: 600; }
.alarm--warn .alarm-msg { color: var(--yellow); }
.alarm--crit .alarm-msg { color: var(--red); }
.alarm-ok {
  display: flex; align-items: center; gap: 6px;
  font-size: .68rem; color: var(--sub); padding: 5px 2px;
}
.ok-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }

/* 트랜지션 */
.alarm-enter-active { transition: all .25s ease; }
.alarm-leave-active { transition: all .2s ease; }
.alarm-enter-from   { opacity: 0; transform: translateX(-8px); }
.alarm-leave-to     { opacity: 0; transform: translateX(-8px); }
</style>
