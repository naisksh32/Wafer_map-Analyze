<script setup>
defineProps({
  processStages: Array,
})
</script>

<template>
  <div class="flow-box">
    <div class="flow-title">공정 흐름 영향 단계</div>
    <div class="flow-stages">
      <template v-for="(stage, i) in processStages" :key="stage.id">
        <div class="stage" :class="`stage--${stage.riskLevel}`">
          <span class="stage__icon">{{ stage.icon }}</span>
          <span class="stage__name">{{ stage.name }}</span>
        </div>
        <span v-if="i < processStages.length - 1" class="arrow">→</span>
      </template>
    </div>
  </div>
</template>

<style scoped>
.flow-box {
  background: var(--panel2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px;
  width: 100%;
  flex-shrink: 0;
}
.flow-title {
  font-size: .6rem;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: var(--sub);
  margin-bottom: 8px;
}
.flow-stages {
  display: flex;
  align-items: center;
  overflow-x: auto;
  padding-bottom: 2px;
  gap: 2px;
}
.stage {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: 5px 7px;
  border-radius: 4px;
  min-width: 50px;
  flex-shrink: 0;
  font-size: .58rem;
  color: var(--sub);
  border: 1px solid transparent;
  transition: all .25s;
}
.stage--crit {
  background: rgba(255,61,90,.1);
  border-color: rgba(255,61,90,.4);
  color: var(--red);
}
.stage--warn {
  background: rgba(0,212,255,.08);
  border-color: rgba(0,212,255,.3);
  color: var(--accent);
}
.stage__icon { font-size: .95rem; }
.arrow { color: var(--border); font-size: .7rem; flex-shrink: 0; }
</style>
