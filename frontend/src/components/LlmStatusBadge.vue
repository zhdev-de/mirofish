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
  gap: 8px;
  padding: 6px 12px;
  border: 1px solid #000;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  cursor: help;
  user-select: none;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

.status-checking .dot { background: #999; animation: pulse 1.2s infinite; }
.status-green    .dot { background: #2ecc40; }
.status-yellow   .dot { background: #f5a623; }
.status-red      .dot { background: #d0021b; }

@keyframes pulse {
  0%, 100% { opacity: 0.4; }
  50%      { opacity: 1; }
}
</style>
