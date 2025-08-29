import { useEffect, useRef } from 'react'

type Overlay = { startIdx: number, endIdx: number, color?: string }

export default function Sparkline({ data, overlays }: { data: (number|null)[], overlays?: Overlay[] }){
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(()=>{
    const el = ref.current; if (!el) return
    const ctx = el.getContext('2d'); if (!ctx) return
    const w = el.clientWidth, h = el.clientHeight
    el.width = w * devicePixelRatio; el.height = h * devicePixelRatio
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0)
    ctx.clearRect(0,0,w,h)
    const xs = data.filter(v => typeof v === 'number') as number[]
    if (!xs.length) return
    const min = Math.min(...xs), max = Math.max(...xs)
    // overlays (Task 3: stimulus bands)
    const bands = overlays || []
    for (const band of bands){
      const s = Math.max(0, Math.min(data.length-1, band.startIdx))
      const e = Math.max(s, Math.min(data.length-1, band.endIdx))
      const x0 = (s/(data.length-1))*(w-2)+1
      const x1 = (e/(data.length-1))*(w-2)+1
      ctx.fillStyle = band.color || 'rgba(245,158,11,0.1)'
      ctx.fillRect(x0, 0, Math.max(2, x1-x0), h)
    }
    // spark
    ctx.beginPath()
    let first = true; const N = data.length
    for (let i=0;i<N;i++){
      const v = data[i]; if (v==null) continue
      const x = (i/(N-1))*(w-2)+1
      const y = h-1 - ((v-min)/(max-min || 1))*(h-2)
      if (first){ ctx.moveTo(x,y); first=false } else { ctx.lineTo(x,y) }
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.8)'
    ctx.lineWidth = 1.2
    ctx.stroke()
  }, [data, overlays])
  return <canvas className="spark" ref={ref}/>
}

