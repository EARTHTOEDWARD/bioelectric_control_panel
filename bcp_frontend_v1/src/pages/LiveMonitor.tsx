import { useEffect, useMemo, useRef, useState } from 'react'
import { WSClient } from '../api/ws'
import { useLiveStore } from '../state/store'
import Sparkline from '../components/Sparkline'
import Badge from '../components/Badge'
import { Chips } from '../components/Chips'

export default function LiveMonitor(){
  const S = useLiveStore()
  const [url, setUrl] = useState<string>((import.meta.env.VITE_WS_URL as string) || 'ws://127.0.0.1:8765/ws')
  const wsRef = useRef<WSClient|null>(null)

  useEffect(()=>{
    wsRef.current?.close()
    const c = new WSClient(url)
    wsRef.current = c
    c.connect()
    return () => c.close()
  }, [url])

  const lams = useMemo(()=> S.history.map(h=>h.lambda1), [S.history])
  const d2s  = useMemo(()=> S.history.map(h=>h.D2), [S.history])
  const dets = useMemo(()=> S.history.map(h=>h.det), [S.history])

  const overlays = useMemo(()=>{
    // Task 3: shade bands when stim_on=true
    const bands: {startIdx:number, endIdx:number, color?:string}[] = []
    let start: number | null = null
    S.history.forEach((h, i) => {
      if (h.stim_on && start === null) start = i
      if ((!h.stim_on || i === S.history.length-1) && start !== null){
        const end = (h.stim_on && i === S.history.length-1) ? i : i-1
        bands.push({ startIdx: start, endIdx: Math.max(start, end), color: 'rgba(245,158,11,0.12)' })
        start = null
      }
    })
    return bands
  }, [S.history])

  const qualChips = useMemo(()=>{
    const chips: {text:string, level?: 'ok'|'warn'|'err'}[] = []
    const m = S.current
    if (!m) return chips
    const q = m.quality_tags
    if (q.resp_phase) chips.push({ text: `resp:${q.resp_phase}`, level: 'ok' })
    if (q.stim_on) chips.push({ text: 'stim:on', level: 'warn' })
    if (!q.stationarity_pass) chips.push({ text: 'stationarity fail', level: 'err' })
    if (q.lambda1_ci_overlaps_zero) chips.push({ text: 'λ1 CI overlaps 0', level: 'err' })
    if (q.d2_indeterminate) chips.push({ text: 'D2 indeterminate', level: 'err' })
    const e = m.entrainment_tags
    if (e){
      if (Number.isFinite(e.determinism_delta)) chips.push({ text: `ΔDET ${(e.determinism_delta as number).toFixed(2)}`, level: (e.determinism_delta as number) > 0 ? 'ok':'warn' })
      if (e.entrainment_observed) chips.push({ text: 'entrained↑', level: 'ok' })
    }
    return chips
  }, [S.current])

  return <div className="grid" style={{gridTemplateColumns:'repeat(12, 1fr)'}}>
    <div className="card" style={{gridColumn:'span 3'}}>
      <h3>State</h3>
      <div style={{display:'flex', alignItems:'center', gap:8}}>
        <Badge text={S.stateBadge?.text || '—'} level={S.stateBadge?.level || 'warn'} />
        <span className="subtle">conf: {(S.stateBadge?.confidence ?? 0).toFixed(2)}</span>
      </div>
      <div style={{height:8}}/>
      <h3>Quality</h3>
      <Badge text={S.current?.quality_ok ? 'quality_ok' : 'quality_hold'} level={S.current?.quality_ok ? 'ok':'err'} />
      <div style={{height:8}}/>
      <Chips items={qualChips}/>
      <div style={{height:12}}/>
      <div className="kv"><span className="k">WS:</span><span>{url}</span></div>
      <div className="controls">
        <input className="input" style={{width:'100%'}} value={url} onChange={e=>setUrl(e.target.value)} placeholder="ws://host:port/ws"/>
        <button className="button" onClick={()=>{ S.reset(); wsRef.current?.close(); const c = new WSClient(url); wsRef.current = c; c.connect(); }}>Reconnect</button>
      </div>
    </div>

    <div className="card" style={{gridColumn:'span 3'}}>
      <h3>Guardrail</h3>
      <div className="subtle">{S.guardrail ? (S.guardrail.allowed ? 'allowed' : 'blocked') : '—'}</div>
      <div style={{height:4}}/>
      <div className="subtle">{(S.guardrail?.messages || []).join(' ') || '—'}</div>
      {S.guardrail?.suggestion && S.guardrail.allowed && (
        <div style={{marginTop:8}} className="kv"><span className="k">suggest:</span><span>{S.guardrail.suggestion.action === 'stim_on' ? `stim ${S.guardrail.suggestion.params?.frequency_hz || '?'}Hz @ ${S.guardrail.suggestion.params?.pulse_width_us || '?'}µs` : 'stim OFF'}</span></div>
      )}
    </div>

    <div className="card" style={{gridColumn:'span 6'}}>
      <h3>λ₁</h3>
      <div className="metric">
        <span>{Number.isFinite(S.current?.metrics.lambda1) ? S.current?.metrics.lambda1.toFixed(3) : '—'}</span>
        <span className="subtle">CI: [
          {Number.isFinite(S.current?.metrics.lambda1_ci?.[0]) ? S.current?.metrics.lambda1_ci?.[0].toFixed(3) : '—'},
          {Number.isFinite(S.current?.metrics.lambda1_ci?.[1]) ? S.current?.metrics.lambda1_ci?.[1].toFixed(3) : '—'}
        ]</span>
      </div>
      <Sparkline data={lams} overlays={overlays}/>
    </div>

    <div className="card" style={{gridColumn:'span 6'}}>
      <h3>D₂</h3>
      <div className="metric">
        <span>{Number.isFinite(S.current?.metrics.D2) ? S.current?.metrics.D2.toFixed(2) : '—'}</span>
        <span className="subtle">CI: [
          {Number.isFinite(S.current?.metrics.D2_ci?.[0]) ? S.current?.metrics.D2_ci?.[0].toFixed(2) : '—'},
          {Number.isFinite(S.current?.metrics.D2_ci?.[1]) ? S.current?.metrics.D2_ci?.[1].toFixed(2) : '—'}
        ]</span>
      </div>
      <Sparkline data={d2s} overlays={overlays}/>
    </div>

    <div className="card" style={{gridColumn:'span 6'}}>
      <h3>%DET</h3>
      <div className="metric">
        <span>{Number.isFinite(S.current?.metrics.rqa.determinism_mean) ? (100*(S.current?.metrics.rqa.determinism_mean||0)).toFixed(1) + '%' : '—'}</span>
      </div>
      <Sparkline data={dets} overlays={overlays}/>
    </div>
  </div>
}

