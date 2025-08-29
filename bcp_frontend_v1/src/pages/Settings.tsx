import { useEffect, useState } from 'react'
import { getToken, setToken } from '../utils/auth'

export default function Settings(){
  const [api, setApi] = useState<string>((import.meta.env.VITE_API_URL as string) || 'http://127.0.0.1:8000')
  const [ws, setWs]   = useState<string>((import.meta.env.VITE_WS_URL as string) || 'ws://127.0.0.1:8765/ws')
  const [tok, setTok] = useState<string>('')
  useEffect(()=>{ setTok(getToken() || '') }, [])

  return <div className="card">
    <h3>Endpoints</h3>
    <div className="controls" style={{marginBottom:8}}>
      <label className="label" style={{width:120}}>API URL</label>
      <input className="input" value={api} onChange={e=>setApi(e.target.value)} style={{flex:1}}/>
    </div>
    <div className="controls" style={{marginBottom:8}}>
      <label className="label" style={{width:120}}>WebSocket URL</label>
      <input className="input" value={ws} onChange={e=>setWs(e.target.value)} style={{flex:1}}/>
    </div>
    <h3>Auth (dev stub)</h3>
    <div className="controls">
      <label className="label" style={{width:120}}>Bearer Token</label>
      <input className="input" value={tok} onChange={e=> setTok(e.target.value)} style={{flex:1}}/>
      <button className="button" onClick={()=>{ setToken(tok); alert('Token saved to localStorage') }}>Save</button>
    </div>
    <div className="subtle" style={{marginTop:8}}>Token is sent as <code>Authorization: Bearer &lt;token&gt;</code> in API requests (Ingest). Real OIDC in Sprint 3.</div>
    <div className="subtle" style={{marginTop:8}}>Set env vars <code>VITE_API_URL</code> and <code>VITE_WS_URL</code> in a <code>.env</code> file for persistence.</div>
  </div>
}

