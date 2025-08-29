import { z } from 'zod'

export type MetricRQA = { determinism_mean: number, det_vals?: number[] }
export type MetricBlock = {
  lambda1: number, lambda1_ci: [number, number],
  D2: number, D2_ci: [number, number],
  rqa: MetricRQA
}
export type Stationarity = { ok: boolean, mean_drift_ratio: number, var_ratio_last_first: number, acf1_abs: number }

export type QualityTags = {
  stationarity_pass: boolean
  stationarity_flags: string[]
  lambda1_ci_overlaps_zero: boolean
  lambda1_ci_width: number
  lambda1_seg_len: number
  d2_ci_width: number
  d2_ci_wide: boolean
  d2_scaling_region_len: number
  d2_indeterminate: boolean
  rqa_det_rel_std: number
  rqa_unstable_thresholding: boolean
  resp_phase?: 'insp'|'exp'|'unknown'
  stim_on?: boolean
}
export type EntrainmentTags = {
  determinism_delta: number|null
  entrainment_observed: boolean
  lambda1_drop: boolean
  d2_drop: boolean
}
export type MetricsUpdate = {
  type: 'metrics_update'
  t_window: number
  modality: string
  embedding: { m: number, tau: number }
  stationarity: Stationarity
  metrics: MetricBlock
  quality_ok: boolean
  quality_tags: QualityTags
  entrainment_tags?: EntrainmentTags
  stim_params?: { frequency_hz?: number, pulse_width_us?: number, amplitude_mA?: number } | null
}
export type StateUpdate = {
  type: 'state_update'
  t_window: number
  state: string
  confidence: number
}
export type GuardrailEvt = {
  type: 'guardrail'
  t_window: number
  proposed: 'stim_on'|'stim_off'|'none'
  allowed: boolean
  messages: string[]
  suggestion?: { mode: string, action: 'stim_on'|'stim_off', params: Record<string, any> } | null
}

export type AnyEvt = MetricsUpdate | StateUpdate | GuardrailEvt

export interface LiveFrame {
  t: number
  lambda1: number|null
  D2: number|null
  det: number|null
  state?: string
  quality_ok?: boolean
  resp_phase?: string
  stim_on?: boolean
}

// Zod schemas (Task 1)
export const ZStationarity = z.object({
  ok: z.boolean(),
  mean_drift_ratio: z.number(),
  var_ratio_last_first: z.number(),
  acf1_abs: z.number().nullable().transform(v => (typeof v === 'number' ? v : 0))
})

export const ZMetricRQA = z.object({
  determinism_mean: z.number(),
  det_vals: z.array(z.number()).optional()
})

export const ZMetricBlock = z.object({
  lambda1: z.number(),
  lambda1_ci: z.tuple([z.number(), z.number()]),
  D2: z.number(),
  D2_ci: z.tuple([z.number(), z.number()]),
  rqa: ZMetricRQA
})

export const ZQualityTags = z.object({
  stationarity_pass: z.boolean(),
  stationarity_flags: z.array(z.string()),
  lambda1_ci_overlaps_zero: z.boolean(),
  lambda1_ci_width: z.number().nullable().transform(v => (typeof v === 'number' ? v : NaN)),
  lambda1_seg_len: z.number().int(),
  d2_ci_width: z.number().nullable().transform(v => (typeof v === 'number' ? v : NaN)),
  d2_ci_wide: z.boolean().optional(),
  d2_scaling_region_len: z.number().int(),
  d2_indeterminate: z.boolean(),
  rqa_det_rel_std: z.number().nullable().transform(v => (typeof v === 'number' ? v : NaN)),
  rqa_unstable_thresholding: z.boolean().optional(),
  resp_phase: z.enum(['insp','exp','unknown']).optional(),
  stim_on: z.boolean().optional()
})

export const ZEntrainment = z.object({
  determinism_delta: z.number().nullable(),
  entrainment_observed: z.boolean().optional().default(false),
  lambda1_drop: z.boolean().optional().default(false),
  d2_drop: z.boolean().optional().default(false)
})

export const ZMetricsUpdate = z.object({
  type: z.literal('metrics_update'),
  t_window: z.number(),
  modality: z.string(),
  embedding: z.object({ m: z.number().int(), tau: z.number().int() }),
  stationarity: ZStationarity,
  metrics: ZMetricBlock,
  quality_ok: z.boolean(),
  quality_tags: ZQualityTags,
  entrainment_tags: ZEntrainment.optional(),
  stim_params: z.object({ frequency_hz: z.number().optional(), pulse_width_us: z.number().optional(), amplitude_mA: z.number().optional() }).nullable().optional()
})

export const ZStateUpdate = z.object({
  type: z.literal('state_update'),
  t_window: z.number(),
  state: z.string(),
  confidence: z.number()
})

export const ZGuardrail = z.object({
  type: z.literal('guardrail'),
  t_window: z.number(),
  proposed: z.enum(['stim_on','stim_off','none']),
  allowed: z.boolean(),
  messages: z.array(z.string()).default([]),
  suggestion: z.object({
    mode: z.string(),
    action: z.enum(['stim_on','stim_off']),
    params: z.record(z.any())
  }).nullable().optional()
})

export const ZAnyEvt = z.discriminatedUnion('type', [ZMetricsUpdate, ZStateUpdate, ZGuardrail])

export function parseAnyEvt(raw: unknown): AnyEvt | null {
  const r = ZAnyEvt.safeParse(raw)
  return r.success ? (r.data as AnyEvt) : null
}

