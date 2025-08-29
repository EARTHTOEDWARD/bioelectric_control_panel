export function Chips({ items }:{ items: { text: string, level?: 'ok'|'warn'|'err' }[] }){
  return <div className="chips">
    {items.map((c, i)=> <span key={i} className="chip" style={{ borderColor: c.level==='err' ? 'rgba(239,68,68,0.35)': (c.level==='warn'?'rgba(245,158,11,0.35)':'rgba(255,255,255,0.12)'), color: (c.level==='err'?'#fca5a5': (c.level==='warn'?'#fcd34d':'#8aa0b8')) }}>{c.text}</span>)}
  </div>
}

