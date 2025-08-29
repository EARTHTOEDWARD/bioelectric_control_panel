import { useEffect, useRef } from 'react'
import * as THREE from 'three'

function embedTakens(x: number[], m: number, tau: number): number[][] {
  const N = x.length - (m-1)*tau
  if (N <= 0) return []
  const Y: number[][] = new Array(N)
  for (let i=0;i<N;i++){
    const row: number[] = new Array(m)
    for (let j=0;j<m;j++) row[j] = x[i + j*tau]
    Y[i] = row
  }
  return Y
}

export default function AttractorCanvas3D({ signal, tau=10, m=3 }:{ signal: number[], tau?: number, m?: number }){
  const ref = useRef<HTMLDivElement>(null)
  useEffect(()=>{
    const mount = ref.current; if (!mount) return
    const width = mount.clientWidth, height = 360
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, width/height, 0.1, 1000)
    camera.position.set(0,0,6)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(window.devicePixelRatio || 1)
    mount.innerHTML = ''
    mount.appendChild(renderer.domElement)
    const light = new THREE.AmbientLight(0xffffff, 0.8)
    scene.add(light)

    const pts = embedTakens(signal, Math.max(2, Math.min(3, m)), Math.max(1, tau))
    if (pts.length){
      // normalize
      let min = Infinity, max = -Infinity
      pts.forEach(p => p.forEach(v => { if (v<min) min=v; if (v>max) max=v; }))
      const scale = (max - min) || 1
      const positions = new Float32Array(pts.length * 3)
      for (let i=0;i<pts.length;i++){
        const p = pts[i]
        const x = ((p[0]-min)/scale - 0.5) * 4
        const y = ((p[1]-min)/scale - 0.5) * 4
        const z = (p[2] != null) ? (((p[2]-min)/scale - 0.5) * 4) : 0
        positions[i*3+0] = x
        positions[i*3+1] = y
        positions[i*3+2] = z
      }
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      const mat = new THREE.PointsMaterial({ size: 0.02, color: 0x86c5ff })
      const cloud = new THREE.Points(geo, mat)
      scene.add(cloud)
    }

    let rotX = 0, rotY = 0, dragging = false, lastX = 0, lastY = 0
    const onDown = (e: MouseEvent) => { dragging = true; lastX = e.clientX; lastY = e.clientY }
    const onUp = () => { dragging = false }
    const onMove = (e: MouseEvent) => {
      if (!dragging) return
      const dx = e.clientX - lastX, dy = e.clientY - lastY
      lastX = e.clientX; lastY = e.clientY
      rotY += dx * 0.005; rotX += dy * 0.005
    }
    renderer.domElement.addEventListener('mousedown', onDown)
    window.addEventListener('mouseup', onUp)
    window.addEventListener('mousemove', onMove)

    const animate = () => {
      scene.rotation.x = rotX
      scene.rotation.y = rotY
      renderer.render(scene, camera)
      id = requestAnimationFrame(animate)
    }
    let id = requestAnimationFrame(animate)
    const onResize = () => {
      const w = mount.clientWidth
      renderer.setSize(w, height)
      camera.aspect = w/height
      camera.updateProjectionMatrix()
    }
    window.addEventListener('resize', onResize)
    return () => {
      cancelAnimationFrame(id)
      window.removeEventListener('resize', onResize)
      renderer.domElement.removeEventListener('mousedown', onDown)
      window.removeEventListener('mouseup', onUp)
      window.removeEventListener('mousemove', onMove)
      mount.removeChild(renderer.domElement)
    }
  }, [signal, tau, m])
  return <div className="card" style={{gridColumn:'span 12'}}>
    <h3>Attractor (Takens embedding)</h3>
    <div ref={ref} />
  </div>
}

