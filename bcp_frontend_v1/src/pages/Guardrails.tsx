import { useMemo } from 'react'
import { useLiveStore } from '../state/store'

export default function Guardrails(){
  const S = useLiveStore()
  const explain = useMemo(()=>{
    const g = S.guardrail
    if (!g) return 'No guardrail events yet.'
    let txt = `${g.allowed ? 'Allowed' : 'Blocked'} — proposed: ${g.proposed}.`
    if (g.messages?.length) txt += ' ' + g.messages.join(' ')
    if (g.allowed && g.suggestion){
      if (g.suggestion.action === 'stim_on'){
        txt += ` Suggest: ${g.suggestion.params?.frequency_hz || '?'}Hz @ ${g.suggestion.params?.pulse_width_us || '?'}µs.`
      } else {
        txt += ` Suggest: stim OFF.`
      }
    }
    return txt
  }, [S.guardrail])

  return <div className="grid" style={{gridTemplateColumns:'repeat(12,1fr)'}}>
    <div className="card" style={{gridColumn:'span 7'}}>
      <h3>Guardrail Reasoning</h3>
      <div className="subtle">{explain}</div>
      <div style={{height:12}}/>
      <ul className="subtle">
        <li>Quality gate requires: stationarity pass + λ₁ CI not overlapping 0 + determinate D₂.</li>
        <li>Stim ON only from <b>vagal_chaotic_rest</b> and below duty limit; Stim OFF only from <b>vagal_periodic_entrained</b>.</li>
        <li>Respiration phase and entrainment tags are displayed as chips in Live view.</li>
      </ul>
    </div>
    <div className="card" style={{gridColumn:'span 5'}}>
      <h3>Current</h3>
      <pre className="subtle" style={{whiteSpace:'pre-wrap'}}>{JSON.stringify(S.guardrail || {}, null, 2)}</pre>
    </div>
  </div>
}

