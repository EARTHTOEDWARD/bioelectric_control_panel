import type { AnyEvt, MetricsUpdate, StateUpdate, GuardrailEvt } from './types'
import { useLiveStore } from '../state/store'
import { parseAnyEvt } from './types'

const WS_DEFAULT = (import.meta.env.VITE_WS_URL as string) || 'ws://127.0.0.1:8765/ws'

export class WSClient {
  private url: string
  private ws: WebSocket | null = null
  private backoff = 1000
  private maxBackoff = 8000
  private closedByUser = false

  constructor(url?: string){
    this.url = url || WS_DEFAULT
  }

  connect(){
    this.closedByUser = false
    try{
      this.ws = new WebSocket(this.url)
    }catch(e){
      this.scheduleReconnect()
      return
    }
    this.ws.onopen = () => { this.backoff = 1000 }
    this.ws.onmessage = (evt) => {
      try {
        const obj = JSON.parse(evt.data)
        const ev = parseAnyEvt(obj)
        if (!ev) return
        this.handle(ev)
      } catch(e){ /* ignore */ }
    }
    this.ws.onclose = () => { if (!this.closedByUser) this.scheduleReconnect() }
    this.ws.onerror = () => { this.ws?.close() }
  }

  scheduleReconnect(){
    setTimeout(()=> this.connect(), this.backoff)
    this.backoff = Math.min(this.maxBackoff, this.backoff*1.7)
  }

  handle(ev: AnyEvt){
    const S = useLiveStore.getState()
    if (ev.type === 'metrics_update'){
      const m = (ev as MetricsUpdate).metrics
      S.pushFrame({
        t: ev.t_window,
        lambda1: isFinite(m.lambda1) ? m.lambda1 : null,
        D2: isFinite(m.D2) ? m.D2 : null,
        det: isFinite(m.rqa.determinism_mean) ? m.rqa.determinism_mean : null,
        quality_ok: (ev as MetricsUpdate).quality_ok,
        resp_phase: (ev as MetricsUpdate).quality_tags?.resp_phase,
        stim_on: (ev as MetricsUpdate).quality_tags?.stim_on
      })
      S.setCurrentMetrics(ev as MetricsUpdate)
    }
    if (ev.type === 'state_update'){
      S.setStateBadge((ev as StateUpdate).state, (ev as StateUpdate).confidence)
    }
    if (ev.type === 'guardrail'){
      S.setGuardrail(ev as GuardrailEvt)
    }
  }

  close(){
    this.closedByUser = true
    this.ws?.close()
  }
}

