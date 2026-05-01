<template>
  <div class="llm-status-badge" :class="`status-${status}`" :title="tooltip">
    <span class="dot" />
    <span class="label">{{ label }}</span>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import service from '@/api/index.js'

const { t } = useI18n()

const status = ref('checking')
const checkedAt = ref(null)
const POLL_INTERVAL_MS = 60_000

let timer = null

async function fetchStatus () {
  try {
    const res = await service.get('/api/health/llm')
    status.value = res.status || 'yellow'
    checkedAt.value = res.checked_at ? new Date(res.checked_at) : new Date()
  } catch {
    status.value = 'yellow'
    checkedAt.value = new Date()
  }
}

const label = computed(() => {
  if (status.value === 'checking') return t('llmStatus.checking')
  return t(`llmStatus.${status.value}`)
})

const tooltip = computed(() => {
  if (!checkedAt.value) return ''
  const ageMs = Date.now() - checkedAt.value.getTime()
  const ageMin = Math.max(0, Math.round(ageMs / 60000))
  return t('llmStatus.lastCheck', { minutes: ageMin })
})

onMounted(() => {
  fetchStatus()
  timer = setInterval(fetchStatus, POLL_INTERVAL_MS)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
.llm-status-badge {
  display: inline-flex;
  align-items: center;
  gap: 20px;
  padding: 20px 32px;
  border: 3px solid #000;
  font-family: 'JetBrains Mono', monospace;
  font-size: 26px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  cursor: help;
  user-select: none;
  background: #ffffff;
}

.dot {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: inline-block;
  box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.08);
}

.status-checking .dot { background: #999; animation: pulse 1.2s infinite; }
.status-green    .dot { background: #2ecc40; box-shadow: 0 0 16px rgba(46, 204, 64, 0.7); }
.status-yellow   .dot { background: #f5a623; box-shadow: 0 0 16px rgba(245, 166, 35, 0.7); }
.status-red      .dot { background: #d0021b; box-shadow: 0 0 16px rgba(208, 2, 27, 0.8); }

.status-green    { background: #f4faf4; }
.status-yellow   { background: #fef6e7; }
.status-red      { background: #fdecee; }

@keyframes pulse {
  0%, 100% { opacity: 0.4; }
  50%      { opacity: 1; }
}
</style>
