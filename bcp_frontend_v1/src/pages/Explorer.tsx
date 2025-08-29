import { useEffect, useMemo, useState } from 'react'
import type { AnyEvt, MetricsUpdate, StateUpdate, GuardrailEvt } from '../api/types'
import { useLiveStore } from '../state/store'
import AttractorCanvas3D from '../components/AttractorCanvas3D'

export default function Explorer(){
  const S = useLiveStore()
  const [file, setFile] = useState<File|null>(null)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1.0)
  const [raw, setRaw] = useState<string>('')
  const [m, setM] = useState(3)
  const [tau, setTau] = useState(10)

  useEffect(()=>{
    let timer: number | null = null
    if (playing && file){
      const reader = new FileReader()
      reader.onload = async () => {
        const text = reader.result as string
        const lines = text.split(/\r?\n/).filter(Boolean)
        let i = 0
        function step(){
          if (i>=lines.length){ setPlaying(false); return }
          try {
            const obj = JSON.parse(lines[i]) as AnyEvt
            if (obj.type === 'metrics_update'){
              const ev = obj as MetricsUpdate
              S.pushFrame({
                t: ev.t_window,
                lambda1: isFinite(ev.metrics.lambda1) ? ev.metrics.lambda1 : null,
                D2: isFinite(ev.metrics.D2) ? ev.metrics.D2 : null,
                det: isFinite(ev.metrics.rqa.determinism_mean) ? ev.metrics.rqa.determinism_mean : null,
                quality_ok: ev.quality_ok,
                resp_phase: ev.quality_tags?.resp_phase,
                stim_on: ev.quality_tags?.stim_on
              })
              S.setCurrentMetrics(ev)
            } else if (obj.type === 'state_update'){
              const s = obj as StateUpdate
              S.setStateBadge(s.state, s.confidence)
            } else if (obj.type === 'guardrail'){
              const g = obj as GuardrailEvt
              S.setGuardrail(g)
            }
          }catch(e){ /* ignore bad line */ }
          i++
          timer = window.setTimeout(step, 300/Math.max(0.1, speed))
        }
        step()
      }
      reader.readAsText(file)
    }
    return () => { if (timer) clearTimeout(timer) }
  }, [playing, file, speed])

  const signal: number[] = useMemo(()=>{
    try {
      if (!raw.trim()) return []
      // accept comma/space separated or JSON array
      if (raw.trim().startsWith('[')) return JSON.parse(raw)
      return raw.split(/[,\s]+/).map(Number).filter(v=>Number.isFinite(v))
    }catch{ return [] }
  }, [raw])

  return <div className="grid" style={{gridTemplateColumns:'repeat(12,1fr)'}}>
    <div className="card" style={{gridColumn:'span 12'}}>
      <h3>Replay NDJSON</h3>
      <div className="controls">
        <input className="input" type="file" accept=".ndjson,.jsonl,.log" onChange={e=> setFile(e.target.files?.[0] || null) }/>
        <button className="button" onClick={()=> setPlaying(p=>!p)}>{playing ? 'Pause' : 'Play'}</button>
        <label className="label">Speed</label>
        <input className="input" type="number" step="0.1" value={speed} onChange={e=> setSpeed(parseFloat(e.target.value || '1'))} style={{width:80}}/>
      </div>
      <div className="subtle" style={{marginTop:8}}>Uses the same Live store; switch to Live to see metrics update.</div>
    </div>

    <div className="card" style={{gridColumn:'span 12'}}>
      <h3>Attractor (x(t), x(t+τ), x(t+2τ))</h3>
      <div className="controls" style={{marginBottom:8}}>
        <label className="label">Embed dim m</label>
        <input className="input" type="number" value={m} onChange={e=> setM(parseInt(e.target.value || '3'))} style={{width:80}}/>
        <label className="label">Delay τ (samples)</label>
        <input className="input" type="number" value={tau} onChange={e=> setTau(parseInt(e.target.value || '10'))} style={{width:100}}/>
      </div>
      <textarea className="input" style={{width:'100%', minHeight:90, fontFamily:'monospace'}} placeholder="Paste a numeric array or CSV values here (raw channel)." value={raw} onChange={e=> setRaw(e.target.value)} />
    </div>

    {signal.length>0 && (
      <AttractorCanvas3D signal={signal} m={m} tau={tau} />
    )}
  </div>
}

