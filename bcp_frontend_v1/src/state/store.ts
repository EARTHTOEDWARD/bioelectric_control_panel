import { create } from 'zustand'
import type { LiveFrame, MetricsUpdate, GuardrailEvt } from '../api/types'

export type BadgeLevel = 'ok'|'warn'|'err'
export interface LiveState {
  history: LiveFrame[]
  maxPts: number
  current?: MetricsUpdate
  stateBadge?: { text: string, level: BadgeLevel, confidence: number }
  guardrail?: GuardrailEvt
  pushFrame: (f: LiveFrame) => void
  setCurrentMetrics: (m: MetricsUpdate) => void
  setStateBadge: (state: string, confidence: number) => void
  setGuardrail: (g: GuardrailEvt) => void
  reset: () => void
}

function stateLevel(state: string): BadgeLevel {
  if (!state) return 'warn'
  if (state.includes('periodic_entrained')) return 'ok'
  if (state.includes('chaotic')) return 'err'
  return 'warn'
}

export const useLiveStore = create<LiveState>((set, get) => ({
  history: [],
  maxPts: 300,
  current: undefined,
  stateBadge: undefined,
  guardrail: undefined,
  pushFrame: (f) => {
    const arr = get().history.slice()
    arr.push(f)
    if (arr.length > get().maxPts) arr.shift()
    set({ history: arr })
  },
  setCurrentMetrics: (m) => set({ current: m }),
  setStateBadge: (state, confidence) => set({ stateBadge: { text: state, level: stateLevel(state), confidence } }),
  setGuardrail: (g) => set({ guardrail: g }),
  reset: () => set({ history: [], current: undefined, stateBadge: undefined, guardrail: undefined })
}))

